"Wave-optics experiment: ol structured beams."

from __future__ import annotations

import math

import numpy as np
import torch
from scipy import special

from propagation import UnifiedWOS


DARK_ANTIDARK_ORDERS = (0, -1, 1, -2, 2, -3, 3, -4, 4, 5)


def bessel_mode_orders(mode_count: int) -> tuple[int, ...]:
    if mode_count < 1:
        raise ValueError("mode_count must be at least one")
    orders = [0]
    radial_order = 1
    while len(orders) < mode_count:
        orders.append(-radial_order)
        if len(orders) < mode_count:
            orders.append(radial_order)
        radial_order += 1
    return tuple(orders)


def sharp_source(
    evaluator: UnifiedWOS,
    omega: float,
    alpha: float,
    r0_over_omega: float,
    normalized_curvature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    argument = r0_over_omega - evaluator.rr / omega
    airy_amplitude = torch.special.airy_ai(argument)
    finite_energy_envelope = torch.exp(alpha * argument)
    source = (airy_amplitude * finite_energy_envelope).to(evaluator.complex_dtype)
    source = source * evaluator.curvature_phase(normalized_curvature)
    source = evaluator.normalize_each_frame(source.unsqueeze(0))
    weights = torch.ones(1, dtype=torch.float64)
    return source, weights


def pin_source(
    evaluator: UnifiedWOS, envelope_radius: float, design_distance: float
) -> tuple[torch.Tensor, torch.Tensor]:
    beta = evaluator.config.transmitter_radius / (4.0 * design_distance**2)
    envelope = torch.exp(-evaluator.r2 / envelope_radius**2)
    phase = -(4.0 / 3.0) * evaluator.k * math.sqrt(beta) * evaluator.rr.pow(1.5)
    source = envelope * torch.exp(torch.complex(torch.zeros_like(phase), phase))
    source = evaluator.normalize_each_frame(source.unsqueeze(0))
    weights = torch.ones(1, dtype=torch.float64)
    return source, weights


def inverted_pin_source(
    evaluator: UnifiedWOS, envelope_radius: float, design_distance: float, gamma: float
) -> tuple[torch.Tensor, torch.Tensor]:
    transmitter_radius = evaluator.config.transmitter_radius
    coefficient = transmitter_radius ** (2.0 - gamma) / (gamma * design_distance)
    argument = evaluator.k * coefficient * evaluator.rr.pow(gamma)
    envelope = torch.exp(-evaluator.r2 / envelope_radius**2)
    source = (envelope * torch.special.bessel_j0(argument)).to(evaluator.complex_dtype)
    source = evaluator.normalize_each_frame(source.unsqueeze(0))
    weights = torch.ones(1, dtype=torch.float64)
    return source, weights


def dark_antidark_source(
    evaluator: UnifiedWOS,
    transverse_wavenumber: float,
    envelope_radius: float,
    alpha: float,
    mode_count: int = 10,
) -> tuple[torch.Tensor, torch.Tensor]:
    radial_numpy = evaluator.rr.detach().cpu().numpy().astype(np.float64, copy=False)
    azimuth = torch.atan2(evaluator.yy, evaluator.xx)
    gaussian_envelope = torch.exp(-evaluator.r2 / envelope_radius**2)
    modes = []
    modal_weights = []
    orders = (
        DARK_ANTIDARK_ORDERS if mode_count == 10 else bessel_mode_orders(mode_count)
    )
    for order in orders:
        radial_mode_numpy = special.jv(order, transverse_wavenumber * radial_numpy)
        radial_mode = torch.from_numpy(radial_mode_numpy).to(
            device=evaluator.device, dtype=evaluator.real_dtype
        )
        azimuthal_phase = torch.exp(
            torch.complex(torch.zeros_like(azimuth), order * azimuth)
        )
        modes.append(
            (gaussian_envelope * radial_mode).to(evaluator.complex_dtype)
            * azimuthal_phase
        )
        modal_weights.append(1.0 + ((-1) ** abs(order)) * alpha)
    source = evaluator.normalize_each_frame(torch.stack(modes, dim=0))
    weights = torch.tensor(modal_weights, dtype=torch.float64)
    weights = weights / weights.sum().clamp_min(1e-30)
    return source, weights


def build_structured_source(
    evaluator: UnifiedWOS, candidate: dict
) -> tuple[torch.Tensor, torch.Tensor]:
    family = candidate["family"]
    if family == "sharp":
        return sharp_source(
            evaluator,
            candidate["omega_m"],
            candidate["alpha"],
            candidate["r0_over_omega"],
            candidate["normalized_curvature"],
        )
    if family == "pin":
        return pin_source(
            evaluator, candidate["envelope_radius_m"], candidate["design_distance_m"]
        )
    if family == "inverted_pin":
        return inverted_pin_source(
            evaluator,
            candidate["envelope_radius_m"],
            candidate["design_distance_m"],
            candidate["gamma"],
        )
    if family in ("dark", "antidark"):
        return dark_antidark_source(
            evaluator,
            candidate["transverse_wavenumber_rad_m"],
            candidate["envelope_radius_m"],
            candidate["alpha"],
            int(candidate.get("mode_count", 10)),
        )
    raise ValueError(f"Unknown structured-beam family: {family}")
