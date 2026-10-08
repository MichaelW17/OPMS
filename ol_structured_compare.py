"Wave-optics experiment: ol structured compare."

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import torch
from scipy import stats

from ol_structured_beams import build_structured_source
from propagation import LinkConfig, UnifiedWOS


REFERENCE_ROOT = Path(__file__).resolve().parent / "logs/ol_sms_pcb_3param"
OUTPUT_ROOT = Path(__file__).resolve().parent / "logs/ol_structured_comparison_v3"
CASE_DIRECTORIES = ("small_rx_M10", "long_rx20_M10", "strong_small_rx_M10")
FAMILIES = ("sharp", "pin", "inverted_pin", "dark", "antidark")


def candidate_grids(evaluator: UnifiedWOS) -> dict[str, list[dict]]:
    radius = evaluator.config.transmitter_radius
    distance = evaluator.config.distance
    sharp_candidates = []
    curvature_values = (0.0, 0.5, 2.0 / 3.0, 1.0, 2.0)
    for alpha, r0_over_omega, curvature in itertools.product(
        (0.0, 1.0, 2.0, 3.0, 4.0),
        (0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0),
        curvature_values,
    ):
        sharp_candidates.append(
            {
                "family": "sharp",
                "omega_m": radius / 5.0,
                "alpha": alpha,
                "r0_over_omega": r0_over_omega,
                "normalized_curvature": curvature,
            }
        )
    envelope_values = (
        0.5 * radius,
        radius,
        2.0 * radius,
        4.0 * radius,
        1000.0 * radius,
    )
    design_values = (
        0.5 * distance,
        distance,
        1.5 * distance,
        2.0 * distance,
        3.0 * distance,
    )
    pin_candidates = [
        {"family": "pin", "envelope_radius_m": envelope, "design_distance_m": design}
        for envelope, design in itertools.product(envelope_values, design_values)
    ]
    inverted_candidates = [
        {
            "family": "inverted_pin",
            "envelope_radius_m": envelope,
            "design_distance_m": design,
            "gamma": gamma,
        }
        for envelope, design, gamma in itertools.product(
            envelope_values, design_values, (0.1, 0.2, 0.4, 0.6, 0.8)
        )
    ]
    transverse_values = (60.0, 120.0, 240.0, 480.0, 720.0)
    bessel_envelopes = (
        radius,
        1.5 * radius,
        2.0 * radius,
        4.0 * radius,
        1000.0 * radius,
    )
    dark_candidates = [
        {
            "family": "dark",
            "transverse_wavenumber_rad_m": beta,
            "envelope_radius_m": envelope,
            "alpha": alpha,
        }
        for beta, envelope, alpha in itertools.product(
            transverse_values, bessel_envelopes, (-0.5, -0.8, -1.0)
        )
    ]
    antidark_candidates = [
        {
            "family": "antidark",
            "transverse_wavenumber_rad_m": beta,
            "envelope_radius_m": envelope,
            "alpha": alpha,
        }
        for beta, envelope, alpha in itertools.product(
            transverse_values, bessel_envelopes, (0.5, 0.8, 1.0)
        )
    ]
    return {
        "sharp": sharp_candidates,
        "pin": pin_candidates,
        "inverted_pin": inverted_candidates,
        "dark": dark_candidates,
        "antidark": antidark_candidates,
    }


def weighted_metrics(
    modal_flux: torch.Tensor, weights: torch.Tensor, duale_b: float
) -> dict[str, float]:
    modal = modal_flux.detach().cpu().double()
    normalized_weights = weights.detach().cpu().double()
    integrated = modal @ normalized_weights
    mean_flux = integrated.mean()
    scintillation_index = integrated.square().mean() / mean_flux.square() - 1.0
    reciprocal_snr = torch.sqrt(
        torch.as_tensor(duale_b, dtype=torch.float64) / mean_flux
        + scintillation_index.clamp_min(0.0)
    )
    snr_db = 10.0 * torch.log10(1.0 / reciprocal_snr)
    return {
        "snr_db": float(snr_db),
        "mean_flux": float(mean_flux),
        "scintillation_index": float(scintillation_index),
    }


@torch.inference_mode()
def evaluate_once(
    evaluator: UnifiedWOS,
    candidate: dict,
    blocks: int,
    batch_size: int,
    screen_seed: int,
) -> dict[str, float]:
    source, weights = build_structured_source(evaluator, candidate)
    modal_flux = evaluator.collect_modal_flux(source, blocks, batch_size, screen_seed)
    return weighted_metrics(modal_flux, weights, evaluator.config.duale_b)


@torch.inference_mode()
def evaluate_repeats(
    evaluator: UnifiedWOS,
    candidate: dict,
    repeats: int,
    blocks: int,
    batch_size: int,
    seed: int,
) -> dict:
    per_repeat = []
    for repeat in range(repeats):
        screen_seed = seed + 10000 * repeat + 22
        per_repeat.append(
            evaluate_once(evaluator, candidate, blocks, batch_size, screen_seed)
        )
    metric_names = ("snr_db", "mean_flux", "scintillation_index")
    result = {
        name: float(np.mean([item[name] for item in per_repeat]))
        for name in metric_names
    }
    snr_values = np.asarray([item["snr_db"] for item in per_repeat], dtype=float)
    sem = float(stats.sem(snr_values)) if repeats > 1 else 0.0
    multiplier = float(stats.t.ppf(0.975, repeats - 1)) if repeats > 1 else 0.0
    result["snr_db_ci95_low"] = result["snr_db"] - multiplier * sem
    result["snr_db_ci95_high"] = result["snr_db"] + multiplier * sem
    result["per_repeat"] = per_repeat
    result["repeats"] = repeats
    result["samples_per_repeat"] = blocks * batch_size
    return result


def select_family(
    evaluator: UnifiedWOS, family: str, candidates: list[dict], case_seed: int
) -> dict:
    coarse_rows = []
    for index, candidate in enumerate(candidates):
        metrics = evaluate_once(
            evaluator, candidate, blocks=4, batch_size=6, screen_seed=case_seed + 101
        )
        coarse_rows.append({"candidate": candidate, "coarse_metrics": metrics})
        if (index + 1) % 25 == 0 or index + 1 == len(candidates):
            print(f"  {family}: coarse {index + 1}/{len(candidates)}", flush=True)
    top_rows = sorted(
        coarse_rows, key=lambda item: item["coarse_metrics"]["snr_db"], reverse=True
    )[:8]
    refined_rows = []
    for rank, row in enumerate(top_rows):
        refined = evaluate_repeats(
            evaluator,
            row["candidate"],
            repeats=3,
            blocks=10,
            batch_size=8,
            seed=case_seed + 1000001,
        )
        refined_rows.append(
            {
                "candidate": row["candidate"],
                "coarse_metrics": row["coarse_metrics"],
                "refined_metrics": refined,
            }
        )
        print(
            f"  {family}: refined {rank + 1}/{len(top_rows)} SNR={refined['snr_db']:.3f} dB",
            flush=True,
        )
    selected = max(refined_rows, key=lambda item: item["refined_metrics"]["snr_db"])
    formal = evaluate_repeats(
        evaluator,
        selected["candidate"],
        repeats=10,
        blocks=25,
        batch_size=8,
        seed=2026082001,
    )
    return {
        "family": family,
        "candidate_count": len(candidates),
        "selected": selected,
        "formal_metrics": formal,
        "coarse_grid": coarse_rows,
    }


def paired_interval(first_metrics: dict, second_metrics: dict) -> dict[str, float]:
    first = np.asarray(
        [item["snr_db"] for item in first_metrics["per_repeat"]], dtype=float
    )
    second = np.asarray(
        [item["snr_db"] for item in second_metrics["per_repeat"]], dtype=float
    )
    difference = first - second
    mean = float(difference.mean())
    sem = float(stats.sem(difference))
    multiplier = float(stats.t.ppf(0.975, difference.size - 1))
    return {
        "mean_db": mean,
        "ci95_low_db": mean - multiplier * sem,
        "ci95_high_db": mean + multiplier * sem,
    }


def run_case(directory_name: str, case_index: int) -> dict:
    reference_directory = REFERENCE_ROOT / directory_name
    output_directory = OUTPUT_ROOT / directory_name
    output_directory.mkdir(parents=True, exist_ok=True)
    output_path = output_directory / "structured_comparison.json"
    if output_path.exists():
        print(f"Skipping completed structured comparison: {directory_name}", flush=True)
        return json.loads(output_path.read_text(encoding="utf-8"))
    reference_formal = json.loads(
        (reference_directory / "formal_holdout.json").read_text(encoding="utf-8")
    )
    coherent = json.loads(
        (reference_directory / "coherent_boundary.json").read_text(encoding="utf-8")
    )
    evaluator = UnifiedWOS(LinkConfig(**reference_formal["link"]))
    grids = candidate_grids(evaluator)
    family_results = {}
    for family_index, family in enumerate(FAMILIES):
        print(f"{directory_name}: selecting {family}", flush=True)
        family_seed = 2026090001 + 100000 * case_index + 1000 * family_index
        family_results[family] = select_family(
            evaluator, family, grids[family], family_seed
        )
        torch.cuda.empty_cache()
    comparisons = {}
    comparisons["finite_PCB"] = paired_interval(
        reference_formal["SMS"], reference_formal["PCB"]
    )
    comparisons["coherent_CB"] = paired_interval(
        reference_formal["SMS"], coherent["formal_metrics"]
    )
    for family in FAMILIES:
        comparisons[family] = paired_interval(
            reference_formal["SMS"], family_results[family]["formal_metrics"]
        )
    result = {
        "case": directory_name,
        "link": reference_formal["link"],
        "frame_count": reference_formal["frame_count"],
        "reference_metrics": {
            "SMS": reference_formal["SMS"],
            "finite_PCB": reference_formal["PCB"],
            "coherent_CB": coherent["formal_metrics"],
        },
        "structured_families": family_results,
        "sms_minus_comparator": comparisons,
        "fairness": {
            "transmitter_aperture_and_per_frame_power": "identical through UnifiedWOS.normalize_each_frame",
            "atmospheric_realizations": "paired common screen seeds",
            "static_beam_frames": "one field because duplicating an unchanged coherent field within frozen turbulence is mathematically identical",
            "dark_antidark_frames": 10,
            "dark_antidark_time_average": "lambda_m-weighted modal received powers",
            "formal_samples_per_beam": 2000,
        },
    }
    output_path.write_text(
        json.dumps(result, indent=2, allow_nan=True), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "case": directory_name,
                "sms_snr_db": reference_formal["SMS"]["snr_db"],
                "comparators": {
                    name: data["mean_db"] for name, data in comparisons.items()
                },
            },
            indent=2,
        ),
        flush=True,
    )
    return result


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    results = [
        run_case(directory_name, case_index)
        for case_index, directory_name in enumerate(CASE_DIRECTORIES)
    ]
    (OUTPUT_ROOT / "structured_comparison_all.json").write_text(
        json.dumps(results, indent=2, allow_nan=True), encoding="utf-8"
    )
    print(
        f"Completed {len(results)} structured-beam comparison cases in {OUTPUT_ROOT.resolve()}",
        flush=True,
    )


if __name__ == "__main__":
    main()
