"Wave-optics experiment: ol fixed source ensemble validation."

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from sources import SMSParameters, synthesize_sms
from propagation import LinkConfig, UnifiedWOS


INPUT_ROOT = Path(__file__).resolve().parent / "logs/ol_sms_pcb_3param"
OUTPUT_ROOT = (
    Path(__file__).resolve().parent / "logs/ol_fixed_source_ensemble_validation"
)
CASE_NAMES = ("small_rx_M10", "long_rx20_M10", "strong_small_rx_M10")


def write_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=True),
        encoding="utf-8",
    )


def metric_from_integrated_power(
    integrated_power: np.ndarray, duale_b: float
) -> dict[str, float]:
    mean_power = float(np.mean(integrated_power))
    scintillation_index = float(
        np.mean(np.square(integrated_power)) / mean_power**2 - 1.0
    )
    linear_amplitude_snr = float(
        1.0 / np.sqrt(duale_b / mean_power + max(scintillation_index, 0.0))
    )
    return {
        "mean_flux": mean_power,
        "scintillation_index": scintillation_index,
        "snr_linear_current_over_rms": linear_amplitude_snr,
        "snrp_legacy_db_10log": float(10.0 * np.log10(linear_amplitude_snr)),
        "snr_standard_db_20log": float(20.0 * np.log10(linear_amplitude_snr)),
        "snr_power_ratio": float(linear_amplitude_snr**2),
    }


def modal_covariance_metrics(modal_flux: torch.Tensor) -> dict[str, float]:
    modal = modal_flux.detach().cpu().double()
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
    return {
        "effective_frame_number": float(effective_frames),
        "mean_pairwise_power_correlation": float(mean_correlation),
        "nominal_frame_number": int(modal.shape[1]),
    }


def paired_bootstrap(
    sms_power: np.ndarray,
    gsm_power: np.ndarray,
    duale_b: float,
    seed: int,
    bootstrap_repeats: int,
) -> dict[str, float]:
    if sms_power.shape != gsm_power.shape:
        raise ValueError(
            "SMS and GSM power arrays must have identical shapes for paired bootstrap"
        )
    generator = np.random.default_rng(seed)
    indices = generator.integers(
        0, sms_power.size, size=(bootstrap_repeats, sms_power.size)
    )
    sms_samples = sms_power[indices]
    gsm_samples = gsm_power[indices]

    def bootstrap_db(samples: np.ndarray, coefficient: float) -> np.ndarray:
        means = samples.mean(axis=1)
        scintillation = np.square(samples).mean(axis=1) / np.square(means) - 1.0
        linear = 1.0 / np.sqrt(coefficient / means + np.maximum(scintillation, 0.0))
        return 10.0 * np.log10(linear)

    legacy_difference = bootstrap_db(sms_samples, duale_b) - bootstrap_db(
        gsm_samples, duale_b
    )
    low, high = np.quantile(legacy_difference, (0.025, 0.975))
    point = (
        metric_from_integrated_power(sms_power, duale_b)["snrp_legacy_db_10log"]
        - metric_from_integrated_power(gsm_power, duale_b)["snrp_legacy_db_10log"]
    )
    return {
        "sms_minus_gsm_legacy_db": float(point),
        "legacy_ci95_low_db": float(low),
        "legacy_ci95_high_db": float(high),
        "sms_minus_gsm_standard_db": float(2.0 * point),
        "standard_ci95_low_db": float(2.0 * low),
        "standard_ci95_high_db": float(2.0 * high),
        "bootstrap_repeats": bootstrap_repeats,
    }


@torch.inference_mode()
def run_case(
    case_name: str,
    case_index: int,
    atmospheric_samples: int,
    batch_size: int,
    bootstrap_repeats: int,
    force: bool,
) -> dict:
    input_path = INPUT_ROOT / case_name / "result.pt"
    output_directory = OUTPUT_ROOT / case_name
    output_path = output_directory / "fixed_source_ensemble.json"
    flux_path = output_directory / "fixed_source_ensemble_flux.pt"
    if output_path.exists() and flux_path.exists() and not force:
        return json.loads(output_path.read_text(encoding="utf-8"))
    saved = torch.load(input_path, map_location="cpu", weights_only=False)
    summary = saved["summary"]
    evaluator = UnifiedWOS(LinkConfig(**summary["link"]))
    frame_count = int(summary["frame_count"])
    selected_gsm = summary["selected"]["PCB"]
    source_seed = 2026082501 + 100000 * case_index
    screen_seed = 2026082601 + 100000 * case_index
    gsm_source = evaluator.pcb_source(
        torch.tensor(selected_gsm["wg_m"], device=evaluator.device),
        torch.tensor(selected_gsm["lcr_m"], device=evaluator.device),
        torch.tensor(selected_gsm["normalized_curvature"], device=evaluator.device),
        frame_count,
        evaluator.generator(source_seed),
    )
    sms_model = SMSParameters(
        frame_count,
        int(summary["optimization"]["sms_n_side"]),
        int(summary["optimization"].get("seed", 20260817)),
        float(summary["optimization"].get("sms_wg_max", 80e-3)),
        evaluator.device,
    )
    sms_model.load_state_dict(saved["selected_model_states"]["SMS"])
    sms_source = synthesize_sms(evaluator, sms_model, normalized_curvature=1.0)
    blocks = int(np.ceil(atmospheric_samples / batch_size))
    sms_flux = evaluator.collect_modal_flux(
        sms_source, blocks, batch_size, screen_seed
    )[:atmospheric_samples]
    gsm_flux = evaluator.collect_modal_flux(
        gsm_source, blocks, batch_size, screen_seed
    )[:atmospheric_samples]
    sms_power = sms_flux.mean(dim=1).numpy()
    gsm_power = gsm_flux.mean(dim=1).numpy()
    result = {
        "case": case_name,
        "link": summary["link"],
        "frame_count": frame_count,
        "atmospheric_samples": atmospheric_samples,
        "source_seed_gsm": source_seed,
        "screen_seed_common": screen_seed,
        "source_definition": "one independently seeded GSM phase-screen sequence and one trained SMS sequence are frozen before the atmospheric ensemble",
        "SMS": {
            **metric_from_integrated_power(sms_power, evaluator.config.duale_b),
            **modal_covariance_metrics(sms_flux),
        },
        "GSM": {
            **metric_from_integrated_power(gsm_power, evaluator.config.duale_b),
            **modal_covariance_metrics(gsm_flux),
        },
        "paired_difference": paired_bootstrap(
            sms_power,
            gsm_power,
            evaluator.config.duale_b,
            2026082701 + 100000 * case_index,
            bootstrap_repeats,
        ),
    }
    output_directory.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "metadata": {
                "case": case_name,
                "source_seed_gsm": source_seed,
                "screen_seed_common": screen_seed,
                "atmospheric_samples": atmospheric_samples,
            },
            "SMS": sms_flux,
            "GSM": gsm_flux,
        },
        flux_path,
    )
    write_json(output_path, result)
    print(
        json.dumps(
            {
                "case": case_name,
                "SMS_standard_db": result["SMS"]["snr_standard_db_20log"],
                "GSM_standard_db": result["GSM"]["snr_standard_db_20log"],
                "gain_standard_db": result["paired_difference"][
                    "sms_minus_gsm_standard_db"
                ],
            }
        ),
        flush=True,
    )
    return result


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
            case_index,
            arguments.atmospheric_samples,
            arguments.batch_size,
            arguments.bootstrap_repeats,
            arguments.force,
        )
        for case_index, case_name in enumerate(CASE_NAMES)
    ]
    write_json(OUTPUT_ROOT / "fixed_source_ensemble_all.json", results)


if __name__ == "__main__":
    main()
