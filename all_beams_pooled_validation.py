"Wave-optics experiment: ol all beams pooled validation."

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from dad_fair_revalidation import (
    build_dad_basis,
    build_source_and_norms,
    corrected_dwell_weights,
)
from fixed_source_ensemble_validation import (
    metric_from_integrated_power,
    modal_covariance_metrics,
    paired_bootstrap,
)
from structured_beams import build_structured_source
from propagation import LinkConfig, UnifiedWOS


REFERENCE_ROOT = Path(__file__).resolve().parent / "logs/ol_sms_pcb_3param"
FIXED_ROOT = (
    Path(__file__).resolve().parent / "logs/ol_fixed_source_ensemble_validation"
)
STRUCTURED_ROOT = Path(__file__).resolve().parent / "logs/ol_structured_comparison_v3"
BOUNDARY_ROOT = Path(__file__).resolve().parent / "logs/ol_structured_boundary_v5"
DAD_ROOT = Path(__file__).resolve().parent / "logs/ol_dad_fair_revalidation"
OUTPUT_ROOT = Path(__file__).resolve().parent / "logs/ol_all_beams_pooled_validation"
CASE_NAMES = ("small_rx_M10", "long_rx20_M10", "strong_small_rx_M10")
CASE_STEMS = ("small_rx", "long_rx20", "strong_small_rx")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=True),
        encoding="utf-8",
    )


@torch.inference_mode()
def propagate_weighted(
    evaluator: UnifiedWOS,
    source: torch.Tensor,
    weights: torch.Tensor,
    atmospheric_samples: int,
    batch_size: int,
    screen_seed: int,
) -> tuple[torch.Tensor, np.ndarray]:
    blocks = int(np.ceil(atmospheric_samples / batch_size))
    modal_flux = evaluator.collect_modal_flux(source, blocks, batch_size, screen_seed)[
        :atmospheric_samples
    ]
    integrated_power = (
        modal_flux.detach().cpu().double() @ weights.detach().cpu().double()
    ).numpy()
    return modal_flux, integrated_power


@torch.inference_mode()
def coherent_gaussian_source(
    evaluator: UnifiedWOS, selected: dict
) -> tuple[torch.Tensor, torch.Tensor]:
    waist = torch.tensor(
        float(selected["wg_m"]), device=evaluator.device, dtype=evaluator.real_dtype
    )
    curvature = torch.tensor(
        float(selected["normalized_curvature"]),
        device=evaluator.device,
        dtype=evaluator.real_dtype,
    )
    field = torch.exp(-evaluator.r2 / waist.square()).to(
        evaluator.complex_dtype
    ) * evaluator.curvature_phase(curvature)
    source = evaluator.normalize_each_frame(field.unsqueeze(0))
    return source, torch.ones(1, dtype=torch.float64)


def selected_candidates(case_name: str) -> dict[str, dict]:
    boundary = read_json(BOUNDARY_ROOT / case_name / "boundary_extension.json")
    structured = read_json(STRUCTURED_ROOT / case_name / "structured_comparison.json")
    return {
        "Sharp": boundary["sharp"]["selected"]["candidate"],
        "Pin": structured["structured_families"]["pin"]["selected"]["candidate"],
        "IOPB": boundary["inverted_pin"]["selected"]["candidate"],
    }


@torch.inference_mode()
def run_case(
    case_name: str,
    case_stem: str,
    case_index: int,
    atmospheric_samples: int,
    batch_size: int,
    bootstrap_repeats: int,
    force: bool,
) -> dict:
    output_directory = OUTPUT_ROOT / case_name
    output_path = output_directory / "all_beams_pooled.json"
    flux_path = output_directory / "all_beams_pooled_flux.pt"
    if output_path.exists() and flux_path.exists() and not force:
        return read_json(output_path)
    fixed_metadata = read_json(FIXED_ROOT / case_name / "fixed_source_ensemble.json")
    fixed_flux = torch.load(
        FIXED_ROOT / case_name / "fixed_source_ensemble_flux.pt",
        map_location="cpu",
        weights_only=False,
    )
    link = fixed_metadata["link"]
    evaluator = UnifiedWOS(LinkConfig(**link))
    screen_seed = int(fixed_metadata["screen_seed_common"])
    results = {"SMS": fixed_metadata["SMS"], "GSM": fixed_metadata["GSM"]}
    integrated_powers = {
        "SMS": fixed_flux["SMS"].mean(dim=1).numpy(),
        "GSM": fixed_flux["GSM"].mean(dim=1).numpy(),
    }
    modal_fluxes = {"SMS": fixed_flux["SMS"], "GSM": fixed_flux["GSM"]}
    coherent = read_json(REFERENCE_ROOT / case_name / "coherent_boundary.json")
    cb_source, cb_weights = coherent_gaussian_source(evaluator, coherent["selected"])
    cb_flux, cb_power = propagate_weighted(
        evaluator, cb_source, cb_weights, atmospheric_samples, batch_size, screen_seed
    )
    results["CB"] = {
        **metric_from_integrated_power(cb_power, evaluator.config.duale_b),
        **modal_covariance_metrics(cb_flux),
    }
    integrated_powers["CB"] = cb_power
    modal_fluxes["CB"] = cb_flux
    for beam_name, candidate in selected_candidates(case_name).items():
        source, weights = build_structured_source(evaluator, candidate)
        modal_flux, integrated_power = propagate_weighted(
            evaluator, source, weights, atmospheric_samples, batch_size, screen_seed
        )
        results[beam_name] = {
            **metric_from_integrated_power(integrated_power, evaluator.config.duale_b),
            **modal_covariance_metrics(modal_flux),
            "candidate": candidate,
        }
        integrated_powers[beam_name] = integrated_power
        modal_fluxes[beam_name] = modal_flux
        del source
        torch.cuda.empty_cache()
    dad_payload = read_json(DAD_ROOT / f"{case_stem}_DAD_M11" / "dad_fair_result.json")
    dad_candidate = dad_payload["selected"]["candidate"]
    dad_basis = build_dad_basis(evaluator, 11, float(dad_candidate["beta_rad_m"]))
    dad_source, dad_norms = build_source_and_norms(
        evaluator,
        dad_basis,
        float(dad_candidate["envelope_radius_m"]),
        float(dad_candidate["normalized_curvature"]),
    )
    dad_weights = corrected_dwell_weights(11, float(dad_candidate["alpha"]), dad_norms)
    dad_flux, dad_power = propagate_weighted(
        evaluator, dad_source, dad_weights, atmospheric_samples, batch_size, screen_seed
    )
    results["DAD"] = {
        **metric_from_integrated_power(dad_power, evaluator.config.duale_b),
        **modal_covariance_metrics(dad_flux),
        "candidate": dad_candidate,
        "dwell_weights": dad_weights.tolist(),
    }
    integrated_powers["DAD"] = dad_power
    modal_fluxes["DAD"] = dad_flux
    paired = {}
    for comparison_index, beam_name in enumerate(
        ("GSM", "CB", "Sharp", "Pin", "IOPB", "DAD")
    ):
        paired[beam_name] = paired_bootstrap(
            integrated_powers["SMS"],
            integrated_powers[beam_name],
            evaluator.config.duale_b,
            2026082801 + 100000 * case_index + 1000 * comparison_index,
            bootstrap_repeats,
        )
    payload = {
        "case": case_name,
        "link": link,
        "atmospheric_samples": atmospheric_samples,
        "screen_seed_common": screen_seed,
        "statistics": "one pooled atmosphere ensemble for each fixed source; paired bootstrap resamples common atmosphere indices",
        "DAD_weight_correction": "dwell_m proportional to lambda_m times the finite-aperture modal norm squared",
        "beams": results,
        "sms_minus_comparator": paired,
    }
    output_directory.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "metadata": {
                "case": case_name,
                "link": link,
                "atmospheric_samples": atmospheric_samples,
                "screen_seed_common": screen_seed,
            },
            "modal_flux": modal_fluxes,
            "integrated_power": {
                name: torch.from_numpy(values)
                for name, values in integrated_powers.items()
            },
        },
        flux_path,
    )
    write_json(output_path, payload)
    print(
        json.dumps(
            {
                "case": case_name,
                "standard_db": {
                    name: value["snr_standard_db_20log"]
                    for name, value in results.items()
                },
                "SMS_minus_best_designed_standard_db": min(
                    paired[name]["sms_minus_gsm_standard_db"]
                    for name in ("Sharp", "Pin", "IOPB", "DAD")
                ),
            }
        ),
        flush=True,
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atmospheric-samples", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--bootstrap-repeats", type=int, default=2000)
    parser.add_argument("--force", action="store_true")
    arguments = parser.parse_args()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    results = [
        run_case(
            case_name,
            case_stem,
            case_index,
            arguments.atmospheric_samples,
            arguments.batch_size,
            arguments.bootstrap_repeats,
            arguments.force,
        )
        for case_index, (case_name, case_stem) in enumerate(zip(CASE_NAMES, CASE_STEMS))
    ]
    write_json(OUTPUT_ROOT / "all_beams_pooled_all.json", results)


if __name__ == "__main__":
    main()
