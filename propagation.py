"Split-step propagation, atmospheric phase screens, and receiver statistics."

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import fft


@dataclass(frozen=True)
class LinkConfig:
    name: str = "reference"
    n: int = 512
    delta: float = 1e-3
    wavelength: float = 0.85e-6
    cn2: float = 1e-14
    distance: float = 1e3
    wos_steps: int = 5
    transmitter_radius: float = 5e-2
    receiver_radius: float = 2.5e-2
    inner_scale: float = 3e-3
    outer_scale: float = 1.024
    duale_b: float = 0.2
    power: float = 1000.0
    subharmonic_order: int = 3


class UnifiedWOS:
    def __init__(self, config: LinkConfig, device: str = "cuda") -> None:
        self.config = config
        self.device = torch.device(device)
        self.real_dtype = torch.float32
        self.complex_dtype = torch.complex64
        self.pi = torch.pi
        self.k = 2.0 * self.pi / config.wavelength
        self.delta_z = config.distance / config.wos_steps
        coordinate = torch.linspace(
            -config.n * config.delta / 2.0,
            (config.n / 2.0 - 1.0) * config.delta,
            config.n,
            device=self.device,
            dtype=self.real_dtype,
        )
        self.xx, self.yy = torch.meshgrid(coordinate, coordinate, indexing="ij")
        self.r2 = self.xx.square() + self.yy.square()
        self.rr = torch.sqrt(self.r2)
        frequency = torch.linspace(
            -0.5 / config.delta,
            0.5 / config.delta - 1.0 / (config.n * config.delta),
            config.n,
            device=self.device,
            dtype=self.real_dtype,
        )
        self.fx, self.fy = torch.meshgrid(frequency, frequency, indexing="ij")
        self.ff2 = self.fx.square() + self.fy.square()
        self.receiver_aperture = (self.rr < config.receiver_radius).to(self.real_dtype)
        self.transmitter_aperture = (self.rr < config.transmitter_radius).to(
            self.real_dtype
        )
        grid_width = config.n * config.delta / 2.5
        self.grid_truncation = torch.exp(-((self.rr / grid_width) ** 12))
        transfer_phase = -self.pi * config.wavelength * self.delta_z * self.ff2
        self.transfer = fft.fftshift(
            torch.exp(torch.complex(torch.zeros_like(transfer_phase), transfer_phase))
        )
        self.phase_psd_sqrt = self._high_frequency_psd_sqrt()
        self.subharmonic_terms = self._prepare_subharmonics()

    def _high_frequency_psd_sqrt(self) -> torch.Tensor:
        config = self.config
        fm = 5.92 / config.inner_scale / (2.0 * self.pi)
        f0 = 1.0 / config.outer_scale
        radial_frequency = torch.sqrt(self.ff2)
        phase_psd = (
            2.0
            * self.pi
            * self.k**2
            * self.delta_z
            * 0.033
            * config.cn2
            * (2.0 * self.pi) ** (-11.0 / 3.0)
            * (4.0 * self.pi**2)
            * torch.exp(-((radial_frequency / fm) ** 2))
            / (self.ff2 + f0**2) ** (11.0 / 6.0)
        )
        phase_psd[self.config.n // 2, self.config.n // 2] = 0.0
        return torch.sqrt(phase_psd) / (config.n * config.delta)

    def _prepare_subharmonics(self) -> list[tuple[torch.Tensor, torch.Tensor]]:
        config = self.config
        fm = 5.92 / config.inner_scale / (2.0 * self.pi)
        f0 = 1.0 / config.outer_scale
        terms: list[tuple[torch.Tensor, torch.Tensor]] = []
        for order in range(1, config.subharmonic_order + 1):
            delta_f = 1.0 / (3**order * config.n * config.delta)
            low_frequency = (
                torch.arange(-1, 2, device=self.device, dtype=self.real_dtype) * delta_f
            )
            low_fx, low_fy = torch.meshgrid(low_frequency, low_frequency, indexing="ij")
            low_radius = torch.sqrt(low_fx.square() + low_fy.square())
            low_psd = (
                2.0
                * self.pi
                * self.k**2
                * self.delta_z
                * 0.033
                * config.cn2
                * (2.0 * self.pi) ** (-11.0 / 3.0)
                * (4.0 * self.pi**2)
                * torch.exp(-((low_radius / fm) ** 2))
                / (low_radius.square() + f0**2) ** (11.0 / 6.0)
            )
            low_psd[1, 1] = 0.0
            coefficient_scale = torch.sqrt(low_psd) * delta_f
            basis_phase = (
                2.0
                * self.pi
                * (
                    low_fx[:, :, None, None] * self.xx
                    + low_fy[:, :, None, None] * self.yy
                )
            )
            basis = torch.exp(torch.complex(torch.zeros_like(basis_phase), basis_phase))
            terms.append((coefficient_scale, basis))
        return terms

    def generator(self, seed: int) -> torch.Generator:
        random_generator = torch.Generator(device=self.device)
        random_generator.manual_seed(int(seed))
        return random_generator

    def curvature_phase(
        self, normalized_curvature: torch.Tensor | float
    ) -> torch.Tensor:
        curvature = torch.as_tensor(
            normalized_curvature, device=self.device, dtype=self.real_dtype
        )
        phase = -self.k * self.r2 * curvature / (2.0 * self.config.distance)
        return torch.exp(torch.complex(torch.zeros_like(phase), phase))

    def normalize_each_frame(self, source: torch.Tensor) -> torch.Tensor:
        source = source * self.transmitter_aperture
        frame_power = source.abs().square().sum(dim=(-2, -1), keepdim=True)
        scale = torch.sqrt(self.config.power / frame_power.clamp_min(1e-20))
        return source * scale

    def pcb_source(
        self,
        wg: torch.Tensor,
        lcr: torch.Tensor,
        normalized_curvature: torch.Tensor,
        frame_count: int,
        source_generator: torch.Generator,
    ) -> torch.Tensor:
        base = torch.exp(-self.r2 / wg.square()).to(self.complex_dtype)
        base = base * self.curvature_phase(normalized_curvature)
        sigma_f = 2.5 * lcr
        sigma_r = torch.sqrt(4.0 * self.pi * sigma_f**4 / lcr.square())
        spatial_filter = torch.exp(-self.pi**2 * sigma_f.square() * self.ff2)
        real_noise = torch.randn(
            frame_count,
            self.config.n,
            self.config.n,
            device=self.device,
            generator=source_generator,
        )
        imag_noise = torch.randn(
            frame_count,
            self.config.n,
            self.config.n,
            device=self.device,
            generator=source_generator,
        )
        white_noise = torch.complex(real_noise, imag_noise)
        filtered_spectrum = fft.fftshift(spatial_filter * white_noise, dim=(-2, -1))
        random_phase = fft.ifft2(filtered_spectrum, dim=(-2, -1)).real
        random_phase = (
            random_phase
            * sigma_r
            * (self.config.n * self.config.delta)
            / self.config.delta**2
        )
        phase_screen = torch.exp(
            torch.complex(torch.zeros_like(random_phase), random_phase)
        )
        source = base.unsqueeze(0) * phase_screen
        return self.normalize_each_frame(source)

    def screens(
        self, batch_size: int, screen_generator: torch.Generator
    ) -> torch.Tensor:
        shape = (batch_size, self.config.wos_steps, self.config.n, self.config.n)
        real_noise = torch.randn(shape, device=self.device, generator=screen_generator)
        imag_noise = torch.randn(shape, device=self.device, generator=screen_generator)
        coefficients = torch.complex(real_noise, imag_noise) * self.phase_psd_sqrt
        high = torch.real(
            fft.ifftshift(
                fft.ifft2(fft.ifftshift(coefficients, dim=(-2, -1)), dim=(-2, -1)),
                dim=(-2, -1),
            )
            * self.config.n**2
        )
        high = high - high.mean(dim=(-2, -1), keepdim=True)
        low = torch.zeros_like(high)
        for coefficient_scale, basis in self.subharmonic_terms:
            low_real = torch.randn(
                batch_size,
                self.config.wos_steps,
                3,
                3,
                device=self.device,
                generator=screen_generator,
            )
            low_imag = torch.randn(
                batch_size,
                self.config.wos_steps,
                3,
                3,
                device=self.device,
                generator=screen_generator,
            )
            low_coefficients = torch.complex(low_real, low_imag) * coefficient_scale
            low = low + torch.real(
                torch.einsum("bsij,ijxy->bsxy", low_coefficients, basis)
            )
        low = low - low.mean(dim=(-2, -1), keepdim=True)
        return high + low

    def propagate(self, field: torch.Tensor) -> torch.Tensor:
        spectrum = fft.fft2(fft.fftshift(field, dim=(-2, -1)), dim=(-2, -1))
        propagated = fft.ifft2(self.transfer * spectrum, dim=(-2, -1))
        return fft.ifftshift(propagated, dim=(-2, -1))

    def modal_flux_batch(
        self,
        source: torch.Tensor,
        batch_size: int,
        screen_generator: torch.Generator,
    ) -> torch.Tensor:
        phase_screens = self.screens(batch_size, screen_generator)
        field = source.unsqueeze(0).expand(batch_size, -1, -1, -1)
        for step in range(self.config.wos_steps):
            current_phase = phase_screens[:, None, step]
            field = field * torch.exp(
                torch.complex(torch.zeros_like(current_phase), current_phase)
            )
            field = field * self.grid_truncation
            field = self.propagate(field)
        intensity = (field.abs() * self.receiver_aperture).square()
        return intensity.sum(dim=(-2, -1)).to(torch.float64)

    def loss_from_source(
        self,
        source: torch.Tensor,
        batch_size: int,
        screen_generator: torch.Generator,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        modal_flux = self.modal_flux_batch(source, batch_size, screen_generator)
        # A slow detector averages pseudo-modes through one frozen atmosphere.
        integrated_flux = modal_flux.mean(dim=1)
        mean_flux = integrated_flux.mean()
        scintillation_index = integrated_flux.square().mean() / mean_flux.square() - 1.0
        reciprocal_snr = torch.sqrt(
            self.config.duale_b / mean_flux + scintillation_index.clamp_min(0.0)
        )
        snr_db = 10.0 * torch.log10(1.0 / reciprocal_snr)
        return reciprocal_snr, snr_db, mean_flux, scintillation_index

    @torch.inference_mode()
    def collect_modal_flux(
        self,
        source: torch.Tensor,
        blocks: int,
        batch_size: int,
        screen_seed: int,
    ) -> torch.Tensor:
        screen_generator = self.generator(screen_seed)
        chunks: list[torch.Tensor] = []
        for _ in range(blocks):
            chunks.append(
                self.modal_flux_batch(source, batch_size, screen_generator).cpu()
            )
        return torch.cat(chunks, dim=0)


def metrics_from_modal_flux(
    modal_flux: torch.Tensor,
    duale_b: float,
    bootstrap_seed: int = 20260817,
    bootstrap_repeats: int = 1000,
) -> dict[str, float]:
    # Pool linear powers before estimating moments; do not average batchwise dB.
    modal = modal_flux.detach().cpu().double()
    integrated = modal.mean(dim=1)
    mean_flux = integrated.mean()
    scintillation_index = integrated.square().mean() / mean_flux.square() - 1.0
    reciprocal_snr = torch.sqrt(
        torch.as_tensor(duale_b, dtype=torch.float64) / mean_flux
        + scintillation_index.clamp_min(0.0)
    )
    snr_db = 10.0 * torch.log10(1.0 / reciprocal_snr)
    centered = modal - modal.mean(dim=0, keepdim=True)
    covariance = centered.T @ centered / max(1, modal.shape[0] - 1)
    standard_deviation = torch.sqrt(torch.diag(covariance).clamp_min(1e-30))
    effective_frames = standard_deviation.sum().square() / covariance.sum().clamp_min(
        1e-30
    )
    correlation = covariance / (
        standard_deviation[:, None] * standard_deviation[None, :]
    ).clamp_min(1e-30)
    off_diagonal = ~torch.eye(modal.shape[1], dtype=torch.bool)
    mean_correlation = (
        correlation[off_diagonal].mean() if modal.shape[1] > 1 else torch.tensor(1.0)
    )
    normalized = integrated / mean_flux
    fade_probability_05 = (normalized < 0.5).double().mean()
    fifth_percentile = torch.quantile(normalized, 0.05)
    flux_numpy = integrated.numpy()
    random_generator = np.random.default_rng(bootstrap_seed)
    sample_indices = random_generator.integers(
        0, flux_numpy.size, size=(bootstrap_repeats, flux_numpy.size)
    )
    bootstrap_flux = flux_numpy[sample_indices]
    bootstrap_mean = bootstrap_flux.mean(axis=1)
    bootstrap_si = (
        np.square(bootstrap_flux).mean(axis=1) / np.square(bootstrap_mean) - 1.0
    )
    bootstrap_snr = 10.0 * np.log10(
        1.0 / np.sqrt(duale_b / bootstrap_mean + np.maximum(bootstrap_si, 0.0))
    )
    ci_low, ci_high = np.quantile(bootstrap_snr, [0.025, 0.975])
    return {
        "snr_db": float(snr_db),
        "snr_db_ci95_low": float(ci_low),
        "snr_db_ci95_high": float(ci_high),
        "mean_flux": float(mean_flux),
        "scintillation_index": float(scintillation_index),
        "effective_frame_number": float(effective_frames),
        "mean_pairwise_power_correlation": float(mean_correlation),
        "normalized_power_q05": float(fifth_percentile),
        "fade_probability_below_half_mean": float(fade_probability_05),
        "atmospheric_samples": int(modal.shape[0]),
        "frame_count": int(modal.shape[1]),
    }


def outage_from_modal_flux(
    modal_flux: torch.Tensor,
    duale_b: float,
    transmitted_power: float,
    threshold_fractions: tuple[float, ...] = (0.5, 0.7, 0.9),
) -> dict:
    if duale_b <= 0.0:
        raise ValueError("duale_b must be positive for outage evaluation")
    # Pool linear powers before estimating moments; do not average batchwise dB.
    modal = modal_flux.detach().cpu().double()
    if modal.ndim == 2:
        modal = modal.unsqueeze(0)
    if modal.ndim != 3:
        raise ValueError(
            "modal_flux must have shape [samples, frames] or [repeats, samples, frames]"
        )
    detected_power = modal.mean(dim=-1)
    ideal_energy_snr = math.sqrt(transmitted_power / duale_b)
    rows = []
    for fraction in threshold_fractions:
        if not 0.0 < fraction < 1.0:
            raise ValueError(
                "every threshold fraction must lie strictly between zero and one"
            )
        snr_threshold = fraction * ideal_energy_snr
        power_threshold = duale_b * snr_threshold**2
        per_repeat = (detected_power < power_threshold).double().mean(dim=1)
        rows.append(
            {
                "threshold_fraction": float(fraction),
                "ideal_energy_snr": float(ideal_energy_snr),
                "snr_threshold": float(snr_threshold),
                "power_threshold": float(power_threshold),
                "mean_outage": float(per_repeat.mean()),
                "per_repeat_outage": [float(value) for value in per_repeat],
            }
        )
    return {
        "definition": "gamma_E(Pr)=sqrt(Pr/Be), gamma_th=fraction*sqrt(Pt/Be)",
        "duale_b": float(duale_b),
        "transmitted_power": float(transmitted_power),
        "frame_count": int(modal.shape[-1]),
        "repeats": int(modal.shape[0]),
        "samples_per_repeat": int(modal.shape[1]),
        "thresholds": rows,
    }


def curvature_to_rg(distance: float, normalized_curvature: float) -> float:
    return (
        math.inf
        if abs(normalized_curvature) < 1e-12
        else distance / normalized_curvature
    )
