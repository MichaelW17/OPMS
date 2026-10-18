"Collimated GSM landscape and live wave-optics Adam trajectories."

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from propagation import LinkConfig, UnifiedWOS

ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "logs" / "landscape"
DEFAULT_FIGURE = (
    ROOT / "figures" / "figures_20260829" / "fig_pcb_collimated_multibasin.png"
)
FRAME_COUNT = 10
WG_MIN = 3e-3
WG_MAX = 180e-3
LCR_MIN = 3e-3
LCR_MAX = 180e-3
COLLIMATED_CURVATURE = 0.0
BASE_SEED = 2026082911


def link_config() -> LinkConfig:
    return LinkConfig(
        name="L1000_C1em14_R25_W850_B0p2_collimated",
        n=512,
        delta=1e-3,
        wavelength=0.850e-6,
        cn2=1e-14,
        distance=1000.0,
        wos_steps=5,
        transmitter_radius=0.05,
        receiver_radius=0.025,
        inner_scale=3e-3,
        outer_scale=1.024,
        duale_b=0.2,
        power=1000.0,
        subharmonic_order=3,
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Expanded collimated GSM landscape with live WOS Adam trajectories."
    )
    parser.add_argument(
        "--stage",
        choices=("landscape", "trajectories", "plot", "all"),
        default="all",
        help="Experiment stage to run.",
    )
    parser.add_argument(
        "--wg-count",
        type=int,
        default=60,
        help="Log-spaced Wg samples from 3 to 180 mm.",
    )
    parser.add_argument(
        "--lcr-count",
        type=int,
        default=60,
        help="Log-spaced Lcr samples from 3 to 180 mm.",
    )
    parser.add_argument(
        "--landscape-samples",
        type=int,
        default=64,
        help="Atmospheres per landscape point.",
    )
    parser.add_argument(
        "--landscape-batch",
        type=int,
        default=8,
        help="Atmospheres per landscape GPU batch.",
    )
    parser.add_argument(
        "--runs", type=int, default=10, help="Number of live WOS Adam trajectories."
    )
    parser.add_argument(
        "--steps", type=int, default=400, help="Adam updates per trajectory."
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.012,
        help="Adam learning rate in bounded log coordinates.",
    )
    parser.add_argument(
        "--train-blocks",
        type=int,
        default=2,
        help="Monte Carlo blocks accumulated per Adam update.",
    )
    parser.add_argument(
        "--train-batch", type=int, default=6, help="Atmospheres per training block."
    )
    parser.add_argument(
        "--grad-clip",
        type=float,
        default=10.0,
        help="Gradient norm clipping threshold.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Independent data directory.",
    )
    parser.add_argument(
        "--trajectory-source",
        type=Path,
        default=None,
        help="Optional directory containing previously computed trajectory files.",
    )
    parser.add_argument(
        "--figure", type=Path, default=DEFAULT_FIGURE, help="Output figure path."
    )
    arguments = parser.parse_args()
    if (
        min(
            arguments.wg_count,
            arguments.lcr_count,
            arguments.landscape_samples,
            arguments.landscape_batch,
            arguments.runs,
            arguments.steps,
            arguments.train_blocks,
            arguments.train_batch,
        )
        < 1
    ):
        raise ValueError("All count and batch arguments must be positive.")
    if arguments.landscape_samples < 2 or arguments.train_batch < 2:
        raise ValueError(
            "SNRP evaluation requires at least two atmospheric realizations."
        )
    return arguments


def snrp_from_modal_flux(modal_flux: torch.Tensor, duale_b: float) -> float:
    integrated_flux = modal_flux.mean(dim=1)
    mean_flux = integrated_flux.mean()
    scintillation_index = integrated_flux.square().mean() / mean_flux.square() - 1.0
    reciprocal_snr = torch.sqrt(
        torch.as_tensor(duale_b, dtype=torch.float64) / mean_flux
        + scintillation_index.clamp_min(0.0)
    )
    return float(10.0 * torch.log10(1.0 / reciprocal_snr))


@torch.inference_mode()
def evaluate_landscape_point(
    evaluator: UnifiedWOS,
    wg: float,
    lcr: float,
    samples: int,
    batch_size: int,
    source_seed: int,
    screen_seed: int,
) -> float:
    source_generator = evaluator.generator(source_seed)
    screen_generator = evaluator.generator(screen_seed)
    wg_tensor = torch.tensor(wg, device=evaluator.device, dtype=evaluator.real_dtype)
    lcr_tensor = torch.tensor(lcr, device=evaluator.device, dtype=evaluator.real_dtype)
    curvature_tensor = torch.tensor(
        COLLIMATED_CURVATURE, device=evaluator.device, dtype=evaluator.real_dtype
    )
    chunks: list[torch.Tensor] = []
    completed = 0
    while completed < samples:
        current_batch = min(batch_size, samples - completed)
        source = evaluator.pcb_source(
            wg_tensor, lcr_tensor, curvature_tensor, FRAME_COUNT, source_generator
        )
        chunks.append(
            evaluator.modal_flux_batch(source, current_batch, screen_generator).cpu()
        )
        completed += current_batch
    return snrp_from_modal_flux(torch.cat(chunks, dim=0), evaluator.config.duale_b)


def landscape_axes(arguments: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    wg_values = np.geomspace(WG_MIN, WG_MAX, arguments.wg_count)
    lcr_values = np.geomspace(LCR_MIN, LCR_MAX, arguments.lcr_count)
    return wg_values, lcr_values


def save_landscape_checkpoint(
    path: Path,
    surface: np.ndarray,
    completed: np.ndarray,
    wg_values: np.ndarray,
    lcr_values: np.ndarray,
    arguments: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "surface_db": surface,
            "completed": completed,
            "wg_values_m": wg_values,
            "lcr_values_m": lcr_values,
            "config": asdict(link_config()),
            "protocol": {
                "curvature": COLLIMATED_CURVATURE,
                "frame_count": FRAME_COUNT,
                "samples_per_point": arguments.landscape_samples,
                "batch_size": arguments.landscape_batch,
                "common_random_numbers_across_grid": True,
            },
        },
        path,
    )


def compute_landscape(arguments: argparse.Namespace) -> Path:
    output_path = arguments.output / "collimated_landscape.pt"
    wg_values, lcr_values = landscape_axes(arguments)
    shape = (wg_values.size, lcr_values.size)
    if output_path.exists():
        saved = torch.load(output_path, map_location="cpu", weights_only=False)
        surface = np.asarray(saved["surface_db"], dtype=float)
        completed = np.asarray(saved["completed"], dtype=bool)
        if surface.shape != shape:
            raise ValueError(
                f"Existing landscape shape {surface.shape} does not match requested shape {shape}."
            )
    else:
        surface = np.full(shape, np.nan, dtype=float)
        completed = np.zeros(shape, dtype=bool)
    evaluator = UnifiedWOS(link_config())
    source_seed = BASE_SEED + 101
    screen_seed = BASE_SEED + 202
    newly_completed = 0
    for i, wg in enumerate(wg_values):
        for j, lcr in enumerate(lcr_values):
            if completed[i, j]:
                continue
            surface[i, j] = evaluate_landscape_point(
                evaluator,
                float(wg),
                float(lcr),
                arguments.landscape_samples,
                arguments.landscape_batch,
                source_seed,
                screen_seed,
            )
            completed[i, j] = True
            newly_completed += 1
            if newly_completed % 12 == 0:
                save_landscape_checkpoint(
                    output_path, surface, completed, wg_values, lcr_values, arguments
                )
        print(f"landscape Wg {i + 1}/{wg_values.size}", flush=True)
    save_landscape_checkpoint(
        output_path, surface, completed, wg_values, lcr_values, arguments
    )
    best_index = np.unravel_index(np.nanargmax(surface), surface.shape)
    summary = {
        "config": asdict(link_config()),
        "curvature": COLLIMATED_CURVATURE,
        "best_grid_point": {
            "wg_m": float(wg_values[best_index[0]]),
            "lcr_m": float(lcr_values[best_index[1]]),
            "snrp_db": float(surface[best_index]),
        },
        "protocol": {
            "wg_count": arguments.wg_count,
            "lcr_count": arguments.lcr_count,
            "samples_per_point": arguments.landscape_samples,
            "frame_count": FRAME_COUNT,
            "common_random_numbers_across_grid": True,
        },
    }
    (arguments.output / "collimated_landscape_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    del evaluator
    torch.cuda.empty_cache()
    return output_path


def inverse_log_bounded(value: float, lower: float, upper: float) -> float:
    fraction = (math.log(value) - math.log(lower)) / (math.log(upper) - math.log(lower))
    fraction = min(max(fraction, 1e-5), 1.0 - 1e-5)
    return math.log(fraction / (1.0 - fraction))


def log_bounded(raw: torch.Tensor, lower: float, upper: float) -> torch.Tensor:
    return torch.exp(math.log(lower) + torch.sigmoid(raw) * math.log(upper / lower))


class CollimatedGSMParameters(nn.Module):
    def __init__(
        self, initial_wg: float, initial_lcr: float, device: torch.device
    ) -> None:
        super().__init__()
        initial = torch.tensor(
            [
                inverse_log_bounded(initial_wg, WG_MIN, WG_MAX),
                inverse_log_bounded(initial_lcr, LCR_MIN, LCR_MAX),
            ],
            device=device,
            dtype=torch.float32,
        )
        self.raw = nn.Parameter(initial)

    def physical(self) -> tuple[torch.Tensor, torch.Tensor]:
        wg = log_bounded(self.raw[0], WG_MIN, WG_MAX)
        lcr = log_bounded(self.raw[1], LCR_MIN, LCR_MAX)
        return wg, lcr


def trajectory_starts(run_count: int) -> list[tuple[float, float]]:
    base = [
        (8e-3, 8e-3),
        (15e-3, 12e-3),
        (28e-3, 8e-3),
        (65e-3, 7e-3),
        (120e-3, 10e-3),
        (10e-3, 45e-3),
        (16e-3, 100e-3),
        (35e-3, 65e-3),
        (75e-3, 35e-3),
        (130e-3, 120e-3),
        (25e-3, 25e-3),
        (50e-3, 15e-3),
    ]
    return [base[index % len(base)] for index in range(run_count)]


def trajectory_path(arguments: argparse.Namespace, run_index: int) -> Path:
    return arguments.output / "trajectories" / f"run_{run_index:02d}.pt"


def compute_trajectory(
    arguments: argparse.Namespace, run_index: int, start: tuple[float, float]
) -> dict:
    path = trajectory_path(arguments, run_index)
    if path.exists():
        return torch.load(path, map_location="cpu", weights_only=False)
    evaluator = UnifiedWOS(link_config())
    model = CollimatedGSMParameters(start[0], start[1], evaluator.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=arguments.learning_rate)
    source_generator = evaluator.generator(BASE_SEED + 10000 * run_index + 303)
    screen_generator = evaluator.generator(BASE_SEED + 10000 * run_index + 404)
    curvature = torch.tensor(
        COLLIMATED_CURVATURE, device=evaluator.device, dtype=evaluator.real_dtype
    )
    history: list[dict] = []
    for step in range(arguments.steps + 1):
        with torch.no_grad():
            current_wg, current_lcr = model.physical()
        if step == 0:
            history.append(
                {
                    "step": 0,
                    "wg_m": float(current_wg),
                    "lcr_m": float(current_lcr),
                    "train_snr_db": math.nan,
                }
            )
            continue
        optimizer.zero_grad(set_to_none=True)
        block_snr: list[float] = []
        for _ in range(arguments.train_blocks):
            wg, lcr = model.physical()
            source = evaluator.pcb_source(
                wg, lcr, curvature, FRAME_COUNT, source_generator
            )
            loss, snr_db, _, _ = evaluator.loss_from_source(
                source, arguments.train_batch, screen_generator
            )
            (loss / arguments.train_blocks).backward()
            block_snr.append(float(snr_db.detach()))
        torch.nn.utils.clip_grad_norm_(model.parameters(), arguments.grad_clip)
        optimizer.step()
        with torch.no_grad():
            updated_wg, updated_lcr = model.physical()
        history.append(
            {
                "step": step,
                "wg_m": float(updated_wg),
                "lcr_m": float(updated_lcr),
                "train_snr_db": float(np.mean(block_snr)),
            }
        )
        if step % 50 == 0:
            print(
                f"trajectory {run_index + 1}/{arguments.runs}: step {step}/{arguments.steps}",
                flush=True,
            )
    result = {
        "run": run_index,
        "initial": {"wg_m": start[0], "lcr_m": start[1]},
        "final": {
            "wg_m": history[-1]["wg_m"],
            "lcr_m": history[-1]["lcr_m"],
            "train_snr_db": history[-1]["train_snr_db"],
        },
        "history": history,
        "protocol": {
            "optimizer": "Adam",
            "objective": "live differentiable WOS, not landscape interpolation",
            "coordinate_parameterization": "bounded logarithmic Wg and Lcr",
            "learning_rate": arguments.learning_rate,
            "steps": arguments.steps,
            "train_blocks": arguments.train_blocks,
            "atmospheres_per_block": arguments.train_batch,
            "fresh_gsm_sequence_before_every_propagation": True,
            "fresh_atmospheres_before_every_propagation": True,
            "curvature": COLLIMATED_CURVATURE,
            "frame_count": FRAME_COUNT,
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, path)
    del evaluator, model, optimizer, source
    torch.cuda.empty_cache()
    return result


def compute_trajectories(arguments: argparse.Namespace) -> list[dict]:
    starts = trajectory_starts(arguments.runs)
    results = [
        compute_trajectory(arguments, run_index, start)
        for run_index, start in enumerate(starts)
    ]
    summary = {
        "config": asdict(link_config()),
        "curvature": COLLIMATED_CURVATURE,
        "protocol": results[0]["protocol"] if results else {},
        "trajectories": [
            {"run": item["run"], "initial": item["initial"], "final": item["final"]}
            for item in results
        ],
    }
    arguments.output.mkdir(parents=True, exist_ok=True)
    (arguments.output / "trajectory_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return results


def load_trajectories(arguments: argparse.Namespace) -> list[dict]:
    trajectory_root = (
        arguments.trajectory_source
        if arguments.trajectory_source is not None
        else arguments.output
    )
    paths = sorted((trajectory_root / "trajectories").glob("run_*.pt"))
    return [torch.load(path, map_location="cpu", weights_only=False) for path in paths]


def plot_result(arguments: argparse.Namespace) -> Path:
    landscape_path = arguments.output / "collimated_landscape.pt"
    if not landscape_path.exists():
        raise FileNotFoundError(
            f"Run the landscape stage before plotting: {landscape_path}"
        )
    payload = torch.load(landscape_path, map_location="cpu", weights_only=False)
    surface = np.asarray(payload["surface_db"], dtype=float)
    wg_values = np.asarray(payload["wg_values_m"], dtype=float) * 1e3
    lcr_values = np.asarray(payload["lcr_values_m"], dtype=float) * 1e3
    trajectories = load_trajectories(arguments)
    if not trajectories:
        raise FileNotFoundError("Run the trajectories stage before plotting.")
    figure, axis = plt.subplots(figsize=(3.35, 2.85), constrained_layout=True)
    mesh = axis.pcolormesh(
        lcr_values, wg_values, surface, shading="auto", cmap="viridis", rasterized=True
    )
    colorbar = figure.colorbar(mesh, ax=axis, pad=0.02)
    colorbar.set_label("SNR (dB)")
    contour_levels = np.linspace(np.nanpercentile(surface, 35.0), np.nanmax(surface), 7)
    axis.contour(
        lcr_values,
        wg_values,
        surface,
        levels=contour_levels,
        colors="white",
        linewidths=0.38,
        alpha=0.65,
    )
    colors = plt.get_cmap("tab10")(np.linspace(0.0, 0.9, len(trajectories)))
    for color, result in zip(colors, trajectories):
        lcr_track = (
            np.asarray([record["lcr_m"] for record in result["history"]], dtype=float)
            * 1e3
        )
        wg_track = (
            np.asarray([record["wg_m"] for record in result["history"]], dtype=float)
            * 1e3
        )
        axis.plot(
            lcr_track,
            wg_track,
            color=color,
            linewidth=0.9,
            alpha=0.90,
            linestyle="-",
            drawstyle="default",
            solid_joinstyle="round",
            zorder=3,
        )
        axis.scatter(
            lcr_track[0],
            wg_track[0],
            s=22,
            marker="o",
            facecolor="white",
            edgecolor=color,
            linewidth=0.9,
            zorder=5,
        )
        axis.scatter(
            lcr_track[-1],
            wg_track[-1],
            s=34,
            marker="*",
            facecolor=color,
            edgecolor="black",
            linewidth=0.35,
            zorder=6,
        )
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlim(lcr_values.min(), lcr_values.max())
    axis.set_ylim(wg_values.min(), wg_values.max())
    axis.set_xlabel(r"Coherence length $l_{\mathrm{cr}}$ (mm)")
    axis.set_ylabel(r"Source radius $w_g$ (mm)")
    axis.text(
        0.025,
        0.97,
        "○ start   ★ end",
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
        color="white",
        bbox={
            "boxstyle": "round,pad=0.18",
            "facecolor": "black",
            "alpha": 0.38,
            "edgecolor": "none",
        },
    )
    arguments.figure.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(arguments.figure, dpi=600)
    figure.savefig(arguments.figure.with_suffix(".pdf"))
    plt.close(figure)
    return arguments.figure


def main() -> None:
    arguments = parse_arguments()
    arguments.output.mkdir(parents=True, exist_ok=True)
    if arguments.stage in ("landscape", "all"):
        compute_landscape(arguments)
    if arguments.stage in ("trajectories", "all"):
        compute_trajectories(arguments)
    if arguments.stage in ("plot", "all"):
        figure_path = plot_result(arguments)
        print(f"Saved figure: {figure_path}", flush=True)


if __name__ == "__main__":
    main()
