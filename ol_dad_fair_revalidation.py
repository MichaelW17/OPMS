"Wave-optics experiment: ol dad fair revalidation."

from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy import special

from ol_frame_budget_complete import dad_orders, load_link
from ol_structured_compare import weighted_metrics
from propagation import LinkConfig, UnifiedWOS


OUTPUT_ROOT = Path(__file__).resolve().parent / "logs/ol_dad_fair_revalidation"
DEFAULT_CASES = ("small_rx", "long_rx20", "strong_small_rx")
DEFAULT_FRAMES = (11, 23, 51)
BETA_VALUES = (90.0, 120.0, 180.0, 240.0, 300.0, 360.0)
ALPHA_VALUES = (-1.0, -0.75, -0.5, -0.25, 0.25, 0.5, 0.75, 1.0)
CURVATURE_VALUES = (0.0, 0.5, 1.0, 1.5)


def write_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=True),
        encoding="utf-8",
    )


@torch.inference_mode()
def build_dad_basis(
    evaluator: UnifiedWOS, frame_count: int, beta: float
) -> torch.Tensor:
    radial_numpy = evaluator.rr.detach().cpu().numpy().astype(np.float64, copy=False)
    squared_index = np.rint(np.square(radial_numpy / evaluator.config.delta)).astype(
        np.int64
    )
    unique_squared, inverse_index = np.unique(squared_index, return_inverse=True)
    unique_radius = evaluator.config.delta * np.sqrt(unique_squared.astype(np.float64))
    azimuth = torch.atan2(evaluator.yy, evaluator.xx)
    modes = []
    for order in dad_orders(frame_count):
        unique_mode = special.jv(order, beta * unique_radius)
        radial_mode_numpy = unique_mode[inverse_index].reshape(radial_numpy.shape)
        radial_mode = torch.from_numpy(radial_mode_numpy).to(
            device=evaluator.device, dtype=evaluator.real_dtype
        )
        phase = torch.exp(torch.complex(torch.zeros_like(azimuth), order * azimuth))
        modes.append(radial_mode.to(evaluator.complex_dtype) * phase)
    return torch.stack(modes, dim=0)


@torch.inference_mode()
def build_source_and_norms(
    evaluator: UnifiedWOS,
    basis: torch.Tensor,
    envelope_radius: float,
    normalized_curvature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    envelope = torch.exp(-evaluator.r2 / envelope_radius**2)
    truncated = basis * envelope.unsqueeze(0) * evaluator.transmitter_aperture
    modal_norms = truncated.abs().square().sum(dim=(-2, -1)).to(torch.float64)
    curved = truncated * evaluator.curvature_phase(normalized_curvature).unsqueeze(0)
    source = evaluator.normalize_each_frame(curved)
    return source, modal_norms


def corrected_dwell_weights(
    frame_count: int, alpha: float, modal_norms: torch.Tensor
) -> torch.Tensor:
    eigenvalues = torch.tensor(
        [1.0 + ((-1) ** abs(order)) * alpha for order in dad_orders(frame_count)],
        dtype=torch.float64,
    )
    physical_weights = eigenvalues * modal_norms.detach().cpu().double()
    return physical_weights / physical_weights.sum().clamp_min(1e-30)


@torch.inference_mode()
def evaluate_source(
    evaluator: UnifiedWOS,
    source: torch.Tensor,
    weights: torch.Tensor,
    atmospheric_samples: int,
    batch_size: int,
    screen_seed: int,
) -> tuple[dict[str, float], torch.Tensor]:
    blocks = int(np.ceil(atmospheric_samples / batch_size))
    modal_flux = evaluator.collect_modal_flux(source, blocks, batch_size, screen_seed)[
        :atmospheric_samples
    ]
    metrics = weighted_metrics(modal_flux, weights, evaluator.config.duale_b)
    metrics["atmospheric_samples"] = int(modal_flux.shape[0])
    return metrics, modal_flux


def candidate_spatial_grid(
    evaluator: UnifiedWOS,
) -> tuple[tuple[float, float, float], ...]:
    radius = float(evaluator.config.transmitter_radius)
    envelopes = (radius, 2.0 * radius, 4.0 * radius, 1000.0 * radius)
    return tuple(itertools.product(BETA_VALUES, envelopes, CURVATURE_VALUES))


def confidence_interval_from_flux(
    modal_flux: torch.Tensor,
    weights: torch.Tensor,
    duale_b: float,
    seed: int,
    repeats: int = 2000,
) -> tuple[float, float]:
    integrated = (
        modal_flux.detach().cpu().double() @ weights.detach().cpu().double()
    ).numpy()
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, integrated.size, size=(repeats, integrated.size))
    resampled = integrated[indices]
    means = resampled.mean(axis=1)
    scintillation = np.square(resampled).mean(axis=1) / np.square(means) - 1.0
    snr_metric_db = 10.0 * np.log10(
        1.0 / np.sqrt(duale_b / means + np.maximum(scintillation, 0.0))
    )
    low, high = np.quantile(snr_metric_db, (0.025, 0.975))
    return float(low), float(high)


def run_one(
    case_stem: str,
    frame_count: int,
    coarse_samples: int,
    refine_samples: int,
    formal_samples: int,
    force: bool,
) -> dict:
    output_directory = OUTPUT_ROOT / f"{case_stem}_DAD_M{frame_count}"
    output_path = output_directory / "dad_fair_result.json"
    flux_path = output_directory / "dad_fair_formal_flux.pt"
    if output_path.exists() and flux_path.exists() and not force:
        return json.loads(output_path.read_text(encoding="utf-8"))
    evaluator = UnifiedWOS(LinkConfig(**load_link(case_stem)))
    started = time.time()
    coarse_rows = []
    coarse_seed = 2026082110 + 1000 * frame_count
    for beta, envelope_radius, curvature in candidate_spatial_grid(evaluator):
        basis = build_dad_basis(evaluator, frame_count, beta)
        source, modal_norms = build_source_and_norms(
            evaluator, basis, envelope_radius, curvature
        )
        _, modal_flux = evaluate_source(
            evaluator,
            source,
            torch.ones(frame_count, dtype=torch.float64) / frame_count,
            coarse_samples,
            4,
            coarse_seed,
        )
        for alpha in ALPHA_VALUES:
            weights = corrected_dwell_weights(frame_count, alpha, modal_norms)
            metrics = weighted_metrics(modal_flux, weights, evaluator.config.duale_b)
            coarse_rows.append(
                {
                    "candidate": {
                        "family": "dark" if alpha < 0.0 else "antidark",
                        "beta_rad_m": beta,
                        "envelope_radius_m": envelope_radius,
                        "normalized_curvature": curvature,
                        "alpha": alpha,
                        "mode_count": frame_count,
                        "first_j0_zero_radius_mm": 1000.0 * 2.4048255577 / beta,
                    },
                    "coarse_metrics": metrics,
                    "modal_norms": modal_norms.detach().cpu().double().tolist(),
                    "dwell_weights": weights.tolist(),
                }
            )
        del basis, source, modal_flux
        torch.cuda.empty_cache()
    top_rows = sorted(
        coarse_rows, key=lambda row: row["coarse_metrics"]["snr_db"], reverse=True
    )[:12]
    refined_rows = []
    refine_seed = 2026082210 + 1000 * frame_count
    for rank, row in enumerate(top_rows):
        candidate = row["candidate"]
        basis = build_dad_basis(evaluator, frame_count, float(candidate["beta_rad_m"]))
        source, modal_norms = build_source_and_norms(
            evaluator,
            basis,
            float(candidate["envelope_radius_m"]),
            float(candidate["normalized_curvature"]),
        )
        weights = corrected_dwell_weights(
            frame_count, float(candidate["alpha"]), modal_norms
        )
        metrics, _ = evaluate_source(
            evaluator, source, weights, refine_samples, 4, refine_seed
        )
        refined_rows.append(
            {
                "candidate": candidate,
                "coarse_metrics": row["coarse_metrics"],
                "refined_metrics": metrics,
                "modal_norms": modal_norms.detach().cpu().double().tolist(),
                "dwell_weights": weights.tolist(),
            }
        )
        print(
            f"{case_stem} M={frame_count} refine {rank + 1}/{len(top_rows)} SNRP={metrics['snr_db']:.4f} dB",
            flush=True,
        )
        del basis, source
        torch.cuda.empty_cache()
    selected = max(refined_rows, key=lambda row: row["refined_metrics"]["snr_db"])
    candidate = selected["candidate"]
    final_basis = build_dad_basis(
        evaluator, frame_count, float(candidate["beta_rad_m"])
    )
    final_source, final_norms = build_source_and_norms(
        evaluator,
        final_basis,
        float(candidate["envelope_radius_m"]),
        float(candidate["normalized_curvature"]),
    )
    final_weights = corrected_dwell_weights(
        frame_count, float(candidate["alpha"]), final_norms
    )
    formal_metrics, formal_flux = evaluate_source(
        evaluator,
        final_source,
        final_weights,
        formal_samples,
        4,
        2026082310 + 1000 * frame_count,
    )
    ci_low, ci_high = confidence_interval_from_flux(
        formal_flux,
        final_weights,
        evaluator.config.duale_b,
        2026082410 + 1000 * frame_count,
    )
    formal_metrics["snr_db_ci95_low"] = ci_low
    formal_metrics["snr_db_ci95_high"] = ci_high
    result = {
        "case": case_stem,
        "frame_count": frame_count,
        "link": evaluator.config.__dict__,
        "selected": selected,
        "formal_metrics": formal_metrics,
        "formal_dwell_weights": final_weights.tolist(),
        "formal_modal_norms": final_norms.detach().cpu().double().tolist(),
        "candidate_count": len(coarse_rows),
        "coarse_samples": coarse_samples,
        "refine_samples": refine_samples,
        "formal_samples": formal_samples,
        "elapsed_seconds": time.time() - started,
        "weight_definition": "dwell_m proportional to lambda_m times the finite-aperture modal norm squared",
        "source_power_constraint": "equal instantaneous launch power per displayed field; weighted mean launch power fixed",
        "selection_rule": "candidate fixed on independent refinement atmospheres before one pooled formal atmosphere ensemble",
    }
    write_json(output_path, result)
    output_directory.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "candidate": candidate,
            "weights": final_weights,
            "modal_norms": final_norms.detach().cpu(),
            "modal_flux": formal_flux,
            "metadata": {
                "case": case_stem,
                "frame_count": frame_count,
                "formal_samples": formal_samples,
            },
        },
        flux_path,
    )
    print(
        f"Completed {case_stem} DAD M={frame_count}: {formal_metrics['snr_db']:.4f} dB",
        flush=True,
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases", nargs="+", default=list(DEFAULT_CASES), choices=DEFAULT_CASES
    )
    parser.add_argument(
        "--frames",
        nargs="+",
        type=int,
        default=list(DEFAULT_FRAMES),
        choices=DEFAULT_FRAMES,
    )
    parser.add_argument("--coarse-samples", type=int, default=16)
    parser.add_argument("--refine-samples", type=int, default=96)
    parser.add_argument("--formal-samples", type=int, default=2000)
    parser.add_argument("--force", action="store_true")
    arguments = parser.parse_args()
    summaries = []
    for case_stem, frame_count in itertools.product(arguments.cases, arguments.frames):
        result = run_one(
            case_stem,
            frame_count,
            arguments.coarse_samples,
            arguments.refine_samples,
            arguments.formal_samples,
            arguments.force,
        )
        summaries.append(
            {
                "case": case_stem,
                "frame_count": frame_count,
                "selected": result["selected"]["candidate"],
                "formal_metrics": result["formal_metrics"],
            }
        )
    write_json(OUTPUT_ROOT / "dad_fair_summary.json", summaries)


if __name__ == "__main__":
    main()
