"Wave-optics experiment: ol structured boundary extension."

from __future__ import annotations

import itertools
import json
import math
from pathlib import Path

import torch

from structured_beams import build_structured_source
from structured_compare import evaluate_repeats, paired_interval, select_family
from propagation import LinkConfig, UnifiedWOS


INPUT_ROOT = Path(__file__).resolve().parent / "logs/ol_structured_comparison_v3"
OUTPUT_ROOT = Path(__file__).resolve().parent / "logs/ol_structured_boundary_v5"
CASE_DIRECTORIES = ("small_rx_M10", "long_rx20_M10", "strong_small_rx_M10")
AIRY_ZERO = 2.404825557695773
MIN_FEATURE_PIXELS = 3.0


def sharp_candidates(evaluator: UnifiedWOS) -> list[dict]:
    omega_values = (7.5e-3, 10e-3, 15e-3, 20e-3)
    alpha_values = (0.0, 0.25, 0.5, 1.0, 2.0)
    ring_values = (6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0)
    curvature_values = (0.5, 0.75, 1.0, 1.25, 1.5)
    candidates = []
    for omega, alpha, ring, curvature in itertools.product(
        omega_values, alpha_values, ring_values, curvature_values
    ):
        edge_argument = max(ring - evaluator.config.transmitter_radius / omega, 1.0)
        local_scale = omega / math.sqrt(edge_argument)
        if local_scale / evaluator.config.delta < MIN_FEATURE_PIXELS:
            continue
        candidates.append(
            {
                "family": "sharp",
                "omega_m": omega,
                "alpha": alpha,
                "r0_over_omega": ring,
                "normalized_curvature": curvature,
            }
        )
    return candidates


def iopb_first_zero_radius(
    evaluator: UnifiedWOS, design_distance: float, gamma: float
) -> float:
    radius = evaluator.config.transmitter_radius
    coefficient = radius ** (2.0 - gamma) / (gamma * design_distance)
    return (AIRY_ZERO / (evaluator.k * coefficient)) ** (1.0 / gamma)


def iopb_candidates(evaluator: UnifiedWOS) -> tuple[list[dict], list[dict]]:
    radius = evaluator.config.transmitter_radius
    distance = evaluator.config.distance
    envelope_values = (radius, 2.0 * radius, 4.0 * radius, 1000.0 * radius)
    design_values = tuple(
        factor * distance for factor in (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0)
    )
    gamma_values = (0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95)
    accepted = []
    rejected = []
    for envelope, design, gamma in itertools.product(
        envelope_values, design_values, gamma_values
    ):
        first_zero = iopb_first_zero_radius(evaluator, design, gamma)
        pixels = first_zero / evaluator.config.delta
        candidate = {
            "family": "inverted_pin",
            "envelope_radius_m": envelope,
            "design_distance_m": design,
            "gamma": gamma,
        }
        if pixels < MIN_FEATURE_PIXELS:
            rejected.append({"candidate": candidate, "first_zero_pixels": pixels})
            continue
        accepted.append(candidate)
    return accepted, rejected


@torch.inference_mode()
def source_diagnostics(evaluator: UnifiedWOS, candidate: dict) -> dict[str, float]:
    source, weights = build_structured_source(evaluator, candidate)
    intensity = source.abs().square().mean(dim=0)
    total = intensity.sum().clamp_min(1e-30)
    outer_shell = (
        evaluator.rr
        >= evaluator.config.transmitter_radius - 3.0 * evaluator.config.delta
    )
    result = {
        "outer_three_pixel_power_fraction": float(intensity[outer_shell].sum() / total),
        "weight_sum": float(weights.sum()),
        "frame_count": int(source.shape[0]),
    }
    if candidate["family"] == "sharp":
        omega = float(candidate["omega_m"])
        edge_argument = max(
            float(candidate["r0_over_omega"])
            - evaluator.config.transmitter_radius / omega,
            1.0,
        )
        result["edge_local_scale_pixels"] = (
            omega / math.sqrt(edge_argument) / evaluator.config.delta
        )
    if candidate["family"] == "inverted_pin":
        first_zero = iopb_first_zero_radius(
            evaluator, float(candidate["design_distance_m"]), float(candidate["gamma"])
        )
        result["first_zero_pixels"] = first_zero / evaluator.config.delta
        result["outer_half_period_pixels"] = (
            math.pi
            * float(candidate["design_distance_m"])
            / (
                evaluator.k
                * evaluator.config.transmitter_radius
                * evaluator.config.delta
            )
        )
    return result


def run_case(case_name: str, case_index: int) -> dict:
    input_path = INPUT_ROOT / case_name / "structured_comparison.json"
    output_directory = OUTPUT_ROOT / case_name
    output_directory.mkdir(parents=True, exist_ok=True)
    output_path = output_directory / "boundary_extension.json"
    if output_path.exists():
        print(f"Skipping completed boundary extension: {case_name}", flush=True)
        return json.loads(output_path.read_text(encoding="utf-8"))
    previous = json.loads(input_path.read_text(encoding="utf-8"))
    evaluator = UnifiedWOS(LinkConfig(**previous["link"]))
    sharp_grid = sharp_candidates(evaluator)
    iopb_grid, rejected = iopb_candidates(evaluator)
    print(f"{case_name}: sharp candidates={len(sharp_grid)}", flush=True)
    sharp_result = select_family(
        evaluator, "sharp", sharp_grid, 2026083001 + 100000 * case_index
    )
    print(
        f"{case_name}: resolved IOPB candidates={len(iopb_grid)}, rejected={len(rejected)}",
        flush=True,
    )
    iopb_result = select_family(
        evaluator, "inverted_pin", iopb_grid, 2026083001 + 100000 * case_index
    )
    sms_metrics = previous["reference_metrics"]["SMS"]
    result = {
        "case": case_name,
        "link": previous["link"],
        "sampling_rule": {
            "minimum_feature_pixels": MIN_FEATURE_PIXELS,
            "iopb_feature": "radius of the first J0 zero",
            "sharp_feature": "Airy positive-tail local scale at the transmitter-aperture edge",
        },
        "sharp": sharp_result,
        "inverted_pin": iopb_result,
        "sharp_source_diagnostics": source_diagnostics(
            evaluator, sharp_result["selected"]["candidate"]
        ),
        "iopb_source_diagnostics": source_diagnostics(
            evaluator, iopb_result["selected"]["candidate"]
        ),
        "rejected_iopb_candidates": rejected,
        "sms_minus_sharp": paired_interval(sms_metrics, sharp_result["formal_metrics"]),
        "sms_minus_inverted_pin": paired_interval(
            sms_metrics, iopb_result["formal_metrics"]
        ),
        "previous_v3": {
            "sharp_snr_db": previous["structured_families"]["sharp"]["formal_metrics"][
                "snr_db"
            ],
            "inverted_pin_snr_db": previous["structured_families"]["inverted_pin"][
                "formal_metrics"
            ]["snr_db"],
            "sharp_candidate": previous["structured_families"]["sharp"]["selected"][
                "candidate"
            ],
            "inverted_pin_candidate": previous["structured_families"]["inverted_pin"][
                "selected"
            ]["candidate"],
        },
    }
    output_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "case": case_name,
                "sharp_snr_db": sharp_result["formal_metrics"]["snr_db"],
                "iopb_snr_db": iopb_result["formal_metrics"]["snr_db"],
                "sms_minus_sharp_db": result["sms_minus_sharp"]["mean_db"],
                "sms_minus_iopb_db": result["sms_minus_inverted_pin"]["mean_db"],
            },
            indent=2,
        ),
        flush=True,
    )
    return result


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    results = [
        run_case(case_name, case_index)
        for case_index, case_name in enumerate(CASE_DIRECTORIES)
    ]
    summary_path = OUTPUT_ROOT / "boundary_extension_summary.json"
    summary = [
        {
            "case": item["case"],
            "sharp_snr_db": item["sharp"]["formal_metrics"]["snr_db"],
            "iopb_snr_db": item["inverted_pin"]["formal_metrics"]["snr_db"],
            "sms_minus_sharp": item["sms_minus_sharp"],
            "sms_minus_inverted_pin": item["sms_minus_inverted_pin"],
            "sharp_candidate": item["sharp"]["selected"]["candidate"],
            "iopb_candidate": item["inverted_pin"]["selected"]["candidate"],
            "sharp_source_diagnostics": item["sharp_source_diagnostics"],
            "iopb_source_diagnostics": item["iopb_source_diagnostics"],
        }
        for item in results
    ]
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Completed boundary extension in {OUTPUT_ROOT.resolve()}", flush=True)


if __name__ == "__main__":
    main()
