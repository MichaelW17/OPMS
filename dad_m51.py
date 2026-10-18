"""Reproduce the archived 51-mode dark/antidark benchmark and OPMS control."""

from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy import special, stats

from sources import SMSParameters, serializable_result, synthesize_sms, train_sms, validate_result
from structured_beams import bessel_mode_orders, dark_antidark_source
from structured_compare import paired_interval, weighted_metrics
from propagation import LinkConfig, UnifiedWOS


ROOT = Path(__file__).resolve().parent
REFERENCE_ROOT = ROOT / "logs/ol_sms_pcb_3param"
STRUCTURED_ROOT = ROOT / "logs/ol_structured_comparison_v3"
OUTPUT_ROOT = ROOT / "logs/ol_dad_sms_m51"
CASE_DIRECTORIES = ("small_rx_M10", "long_rx20_M10", "strong_small_rx_M10")
FRAME_COUNT = 51
FORMAL_REPEATS = 10
FORMAL_BLOCKS = 50
FORMAL_BATCH_SIZE = 4
FORMAL_SEED = 2026082001


def sms_arguments(case_index: int) -> argparse.Namespace:
    return argparse.Namespace(steps=320, runs=3, mc_blocks=2, train_bs=3, sms_lr=0.02, grad_clip=10.0, n_side=8, sms_wg_max=80e-3, log_every=20, seed=2026090101 + 100000 * case_index, selection_repeats=3, selection_blocks=12, selection_seed=2026091001 + 100000 * case_index, holdout_repeats=3, holdout_blocks=12, holdout_seed=2026092001 + 100000 * case_index, eval_bs=4, bootstrap_repeats=300)


def dad_candidates(evaluator: UnifiedWOS, family: str) -> list[dict]:
    radius = evaluator.config.transmitter_radius
    beta_values = (60.0, 90.0, 120.0, 180.0, 240.0, 300.0, 360.0)
    envelope_values = (radius, 2.0 * radius, 4.0 * radius, 1000.0 * radius)
    alpha_values = (-0.25, -0.5, -0.75, -1.0) if family == "dark" else (0.25, 0.5, 0.75, 1.0)
    return [{"family": family, "transverse_wavenumber_rad_m": beta, "envelope_radius_m": envelope, "alpha": alpha, "mode_count": FRAME_COUNT} for beta, envelope, alpha in itertools.product(beta_values, envelope_values, alpha_values)]


def train_and_select_sms(evaluator: UnifiedWOS, case_index: int) -> tuple[dict, argparse.Namespace]:
    args = sms_arguments(case_index)
    candidates = []
    for run_index in range(args.runs):
        print(f"M51 SMS run {run_index + 1}/{args.runs}", flush=True)
        trained = train_sms(evaluator, FRAME_COUNT, run_index, args)
        trained["selection_metrics"] = validate_result(evaluator, trained, FRAME_COUNT, args, selection=True)
        candidates.append(trained)
        torch.cuda.empty_cache()
    selected = max(candidates, key=lambda item: item["selection_metrics"]["snr_db"])
    selected["all_selection_snr_db"] = [float(item["selection_metrics"]["snr_db"]) for item in candidates]
    return selected, args


@torch.inference_mode()
def dad_basis(evaluator: UnifiedWOS, transverse_wavenumber: float) -> torch.Tensor:
    radial_numpy = evaluator.rr.detach().cpu().numpy().astype(np.float64, copy=False)
    squared_index = np.rint(np.square(radial_numpy / evaluator.config.delta)).astype(np.int64)
    unique_squared, inverse_index = np.unique(squared_index, return_inverse=True)
    unique_radius = evaluator.config.delta * np.sqrt(unique_squared.astype(np.float64))
    azimuth = torch.atan2(evaluator.yy, evaluator.xx)
    modes = []
    for order in bessel_mode_orders(FRAME_COUNT):
        unique_mode = special.jv(order, transverse_wavenumber * unique_radius)
        radial_numpy_mode = unique_mode[inverse_index].reshape(radial_numpy.shape)
        radial_mode = torch.from_numpy(radial_numpy_mode).to(device=evaluator.device, dtype=evaluator.real_dtype)
        azimuthal_phase = torch.exp(torch.complex(torch.zeros_like(azimuth), order * azimuth))
        modes.append(radial_mode.to(evaluator.complex_dtype) * azimuthal_phase)
    return torch.stack(modes, dim=0)


def dad_weights(alpha: float) -> torch.Tensor:
    values = [1.0 + ((-1) ** abs(order)) * alpha for order in bessel_mode_orders(FRAME_COUNT)]
    weights = torch.tensor(values, dtype=torch.float64)
    return weights / weights.sum().clamp_min(1e-30)


@torch.inference_mode()
def dad_source_from_basis(evaluator: UnifiedWOS, basis: torch.Tensor, envelope_radius: float) -> torch.Tensor:
    envelope = torch.exp(-evaluator.r2 / envelope_radius**2)
    return evaluator.normalize_each_frame(basis * envelope.unsqueeze(0))


@torch.inference_mode()
def evaluate_source_repeats(evaluator: UnifiedWOS, source: torch.Tensor, weights: torch.Tensor, repeats: int, blocks: int, batch_size: int, base_seed: int) -> dict:
    per_repeat = []
    for repeat in range(repeats):
        screen_seed = base_seed + 10000 * repeat + 22
        modal_flux = evaluator.collect_modal_flux(source, blocks, batch_size, screen_seed)
        metrics = weighted_metrics(modal_flux, weights, evaluator.config.duale_b)
        metrics["atmospheric_samples"] = blocks * batch_size
        metrics["frame_count"] = int(source.shape[0])
        per_repeat.append(metrics)
    snr_values = np.asarray([item["snr_db"] for item in per_repeat], dtype=float)
    mean_snr = float(snr_values.mean())
    sem = float(stats.sem(snr_values)) if repeats > 1 else 0.0
    multiplier = float(stats.t.ppf(0.975, repeats - 1)) if repeats > 1 else 0.0
    return {"snr_db": mean_snr, "snr_db_ci95_low": mean_snr - multiplier * sem, "snr_db_ci95_high": mean_snr + multiplier * sem, "mean_flux": float(np.mean([item["mean_flux"] for item in per_repeat])), "scintillation_index": float(np.mean([item["scintillation_index"] for item in per_repeat])), "per_repeat": per_repeat, "repeats": repeats, "samples_per_repeat": blocks * batch_size, "frame_count": int(source.shape[0])}


def select_dad_families(evaluator: UnifiedWOS, case_index: int) -> dict[str, dict]:
    candidate_map = {family: dad_candidates(evaluator, family) for family in ("dark", "antidark")}
    coarse_rows = {"dark": [], "antidark": []}
    radius = evaluator.config.transmitter_radius
    beta_values = (60.0, 90.0, 120.0, 180.0, 240.0, 300.0, 360.0)
    envelope_values = (radius, 2.0 * radius, 4.0 * radius, 1000.0 * radius)
    coarse_seed = 2026093001 + 100000 * case_index
    completed_groups = 0
    for beta in beta_values:
        basis = dad_basis(evaluator, beta)
        for envelope in envelope_values:
            source = dad_source_from_basis(evaluator, basis, envelope)
            modal_flux = evaluator.collect_modal_flux(source, 6, 4, coarse_seed)
            for family in ("dark", "antidark"):
                alpha_values = (-0.25, -0.5, -0.75, -1.0) if family == "dark" else (0.25, 0.5, 0.75, 1.0)
                for alpha in alpha_values:
                    candidate = {"family": family, "transverse_wavenumber_rad_m": beta, "envelope_radius_m": envelope, "alpha": alpha, "mode_count": FRAME_COUNT}
                    metrics = weighted_metrics(modal_flux, dad_weights(alpha), evaluator.config.duale_b)
                    coarse_rows[family].append({"candidate": candidate, "coarse_metrics": metrics})
            completed_groups += 1
            print(f"  DAD M51 coarse spatial groups {completed_groups}/{len(beta_values) * len(envelope_values)}", flush=True)
            del source
        del basis
        torch.cuda.empty_cache()
    results = {}
    for family_index, family in enumerate(("dark", "antidark")):
        top_rows = sorted(coarse_rows[family], key=lambda item: item["coarse_metrics"]["snr_db"], reverse=True)[:8]
        refined_rows = []
        for rank, row in enumerate(top_rows):
            candidate = row["candidate"]
            source, weights = dark_antidark_source(evaluator, candidate["transverse_wavenumber_rad_m"], candidate["envelope_radius_m"], candidate["alpha"], mode_count=FRAME_COUNT)
            refined = evaluate_source_repeats(evaluator, source, weights, repeats=3, blocks=20, batch_size=4, base_seed=2026103001 + 100000 * case_index + 10000 * family_index)
            refined_rows.append({"candidate": candidate, "coarse_metrics": row["coarse_metrics"], "refined_metrics": refined})
            print(f"  {family} M51 refined {rank + 1}/{len(top_rows)} SNR={refined['snr_db']:.3f} dB", flush=True)
            del source
        selected = max(refined_rows, key=lambda item: item["refined_metrics"]["snr_db"])
        candidate = selected["candidate"]
        formal_source, formal_weights = dark_antidark_source(evaluator, candidate["transverse_wavenumber_rad_m"], candidate["envelope_radius_m"], candidate["alpha"], mode_count=FRAME_COUNT)
        formal = formal_source_metrics(evaluator, formal_source, formal_weights)
        results[family] = {"family": family, "candidate_count": len(candidate_map[family]), "selected": selected, "formal_metrics": formal, "coarse_grid": coarse_rows[family]}
        del formal_source
        torch.cuda.empty_cache()
    return results


@torch.inference_mode()
def formal_source_metrics(evaluator: UnifiedWOS, source: torch.Tensor, weights: torch.Tensor) -> dict:
    normalized_weights = weights.detach().cpu().double()
    normalized_weights = normalized_weights / normalized_weights.sum().clamp_min(1e-30)
    per_repeat = []
    for repeat in range(FORMAL_REPEATS):
        screen_seed = FORMAL_SEED + 10000 * repeat + 22
        modal_flux = evaluator.collect_modal_flux(source, FORMAL_BLOCKS, FORMAL_BATCH_SIZE, screen_seed)
        modal = modal_flux.detach().cpu().double()
        integrated = modal @ normalized_weights
        mean_flux = integrated.mean()
        scintillation_index = integrated.square().mean() / mean_flux.square() - 1.0
        reciprocal_snr = torch.sqrt(torch.as_tensor(evaluator.config.duale_b, dtype=torch.float64) / mean_flux + scintillation_index.clamp_min(0.0))
        snr_db = 10.0 * torch.log10(1.0 / reciprocal_snr)
        per_repeat.append({"snr_db": float(snr_db), "mean_flux": float(mean_flux), "scintillation_index": float(scintillation_index), "atmospheric_samples": int(integrated.numel()), "frame_count": int(source.shape[0])})
    snr_values = np.asarray([item["snr_db"] for item in per_repeat], dtype=float)
    mean_snr = float(snr_values.mean())
    sem = float(stats.sem(snr_values))
    multiplier = float(stats.t.ppf(0.975, FORMAL_REPEATS - 1))
    return {"snr_db": mean_snr, "snr_db_ci95_low": mean_snr - multiplier * sem, "snr_db_ci95_high": mean_snr + multiplier * sem, "mean_flux": float(np.mean([item["mean_flux"] for item in per_repeat])), "scintillation_index": float(np.mean([item["scintillation_index"] for item in per_repeat])), "per_repeat": per_repeat, "repeats": FORMAL_REPEATS, "samples_per_repeat": FORMAL_BLOCKS * FORMAL_BATCH_SIZE, "frame_count": int(source.shape[0])}


@torch.inference_mode()
def selected_sms_source(evaluator: UnifiedWOS, selected: dict, args: argparse.Namespace) -> torch.Tensor:
    model = SMSParameters(FRAME_COUNT, int(selected["n_side"]), int(args.seed), float(args.sms_wg_max), evaluator.device)
    model.load_state_dict(selected["model_state"])
    return synthesize_sms(evaluator, model, normalized_curvature=1.0)


def run_case(case_name: str, case_index: int) -> dict:
    output_directory = OUTPUT_ROOT / case_name.replace("_M10", "_M51")
    output_directory.mkdir(parents=True, exist_ok=True)
    output_json = output_directory / "m51_comparison.json"
    output_pt = output_directory / "m51_sms_state.pt"
    checkpoint_pt = output_directory / "m51_sms_training_checkpoint.pt"
    if output_json.exists() and output_pt.exists():
        print(f"Skipping completed M51 case: {case_name}", flush=True)
        return json.loads(output_json.read_text(encoding="utf-8"))
    reference_summary = json.loads((REFERENCE_ROOT / case_name / "summary.json").read_text(encoding="utf-8"))
    v3 = json.loads((STRUCTURED_ROOT / case_name / "structured_comparison.json").read_text(encoding="utf-8"))
    evaluator = UnifiedWOS(LinkConfig(**reference_summary["link"]))
    started = time.time()
    args = sms_arguments(case_index)
    if checkpoint_pt.exists():
        checkpoint = torch.load(checkpoint_pt, map_location=evaluator.device, weights_only=False)
        selected_sms = checkpoint["selected_sms"]
        print(f"Resumed M51 SMS checkpoint: {checkpoint_pt}", flush=True)
    else:
        selected_sms, args = train_and_select_sms(evaluator, case_index)
        torch.save({"selected_sms": selected_sms, "arguments": vars(args)}, checkpoint_pt)
    sms_source = selected_sms_source(evaluator, selected_sms, args)
    sms_weights = torch.ones(FRAME_COUNT, dtype=torch.float64) / FRAME_COUNT
    sms_formal = formal_source_metrics(evaluator, sms_source, sms_weights)
    print(f"{case_name}: selecting cached M51 dark/antidark families", flush=True)
    family_results = select_dad_families(evaluator, case_index)
    result = {"case": case_name.replace("_M10", "_M51"), "link": reference_summary["link"], "frame_count": FRAME_COUNT, "fairness_rationale": "The DAD paper states that the first 51 modes (N=25, m=-25...25) reasonably approximate the central dark/antidark structure; SMS is therefore re-optimized at the identical 51-frame budget.", "sms": {"selected": serializable_result(selected_sms), "formal_metrics": sms_formal}, "dark": family_results["dark"], "antidark": family_results["antidark"], "sms_minus_dark": paired_interval(sms_formal, family_results["dark"]["formal_metrics"]), "sms_minus_antidark": paired_interval(sms_formal, family_results["antidark"]["formal_metrics"]), "m10_reference": {"sms_snr_db": v3["reference_metrics"]["SMS"]["snr_db"], "dark_snr_db": v3["structured_families"]["dark"]["formal_metrics"]["snr_db"], "antidark_snr_db": v3["structured_families"]["antidark"]["formal_metrics"]["snr_db"]}, "optimization": vars(args), "elapsed_seconds": time.time() - started}
    torch.save({"selected_sms_state": selected_sms["model_state"], "frame_count": FRAME_COUNT, "n_side": selected_sms["n_side"], "sms_wg_max": args.sms_wg_max, "link": reference_summary["link"]}, output_pt)
    output_json.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(json.dumps({"case": result["case"], "sms_snr_db": sms_formal["snr_db"], "dark_snr_db": family_results["dark"]["formal_metrics"]["snr_db"], "antidark_snr_db": family_results["antidark"]["formal_metrics"]["snr_db"], "sms_minus_dark_db": result["sms_minus_dark"]["mean_db"], "sms_minus_antidark_db": result["sms_minus_antidark"]["mean_db"]}, indent=2), flush=True)
    return result


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    results = [run_case(case_name, case_index) for case_index, case_name in enumerate(CASE_DIRECTORIES)]
    summary = [{"case": item["case"], "sms_snr_db": item["sms"]["formal_metrics"]["snr_db"], "dark_snr_db": item["dark"]["formal_metrics"]["snr_db"], "antidark_snr_db": item["antidark"]["formal_metrics"]["snr_db"], "sms_minus_dark": item["sms_minus_dark"], "sms_minus_antidark": item["sms_minus_antidark"], "sms_selection_snr_db": item["sms"]["selected"]["all_selection_snr_db"], "dark_candidate": item["dark"]["selected"]["candidate"], "antidark_candidate": item["antidark"]["selected"]["candidate"]} for item in results]
    (OUTPUT_ROOT / "m51_comparison_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Completed matched M51 DAD/SMS comparison in {OUTPUT_ROOT.resolve()}", flush=True)


if __name__ == "__main__":
    main()
