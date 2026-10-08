"Wave-optics experiment: ol revalidate results."

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy import stats

from sources import SMSParameters, synthesize_sms
from propagation import LinkConfig, UnifiedWOS, metrics_from_modal_flux


def source_overlap(source: torch.Tensor) -> float:
    flattened = source.detach().flatten(1)
    normalized = flattened / torch.linalg.vector_norm(
        flattened, dim=1, keepdim=True
    ).clamp_min(1e-20)
    overlap = torch.abs(normalized @ normalized.conj().T).square().real
    mask = ~torch.eye(source.shape[0], dtype=torch.bool, device=source.device)
    return float(overlap[mask].mean()) if source.shape[0] > 1 else 1.0


def individual_frame_si(modal_flux: torch.Tensor) -> float:
    modal = modal_flux.detach().cpu().double()
    frame_means = modal.mean(dim=0)
    frame_si = modal.square().mean(dim=0) / frame_means.square() - 1.0
    return float(frame_si.mean())


@torch.inference_mode()
def validate_one(
    result_path: Path,
    repeats: int,
    blocks: int,
    batch_size: int,
    seed: int,
    bootstrap_repeats: int,
    force: bool,
) -> dict:
    output_path = result_path.parent / "formal_holdout.json"
    flux_path = result_path.parent / "formal_holdout_flux.pt"
    if output_path.exists() and flux_path.exists() and not force:
        return json.loads(output_path.read_text(encoding="utf-8"))
    saved = torch.load(result_path, map_location="cpu", weights_only=False)
    summary = saved["summary"]
    link = LinkConfig(**summary["link"])
    evaluator = UnifiedWOS(link)
    frame_count = int(summary["frame_count"])
    n_side = int(summary["optimization"]["sms_n_side"])
    sms_wg_max = float(summary["optimization"].get("sms_wg_max", 80e-3))
    beam_metrics = {}
    saved_flux = {}
    for kind in ("PCB", "SMS"):
        selected = summary["selected"][kind]
        modal_chunks = []
        overlap_values = []
        repeat_metrics = []
        for repeat in range(repeats):
            source_seed = seed + 10000 * repeat + 11
            screen_seed = seed + 10000 * repeat + 22
            if kind == "PCB":
                wg = torch.tensor(selected["wg_m"], device=evaluator.device)
                lcr = torch.tensor(selected["lcr_m"], device=evaluator.device)
                curvature = torch.tensor(
                    selected["normalized_curvature"], device=evaluator.device
                )
                source = evaluator.pcb_source(
                    wg, lcr, curvature, frame_count, evaluator.generator(source_seed)
                )
            else:
                model = SMSParameters(
                    frame_count, n_side, seed, sms_wg_max, evaluator.device
                )
                model.load_state_dict(saved["selected_model_states"]["SMS"])
                source = synthesize_sms(evaluator, model, normalized_curvature=1.0)
            overlap_values.append(source_overlap(source))
            modal_flux = evaluator.collect_modal_flux(
                source, blocks, batch_size, screen_seed
            )
            modal_chunks.append(modal_flux)
            repeat_metric = metrics_from_modal_flux(
                modal_flux,
                link.duale_b,
                bootstrap_seed=seed + repeat,
                bootstrap_repeats=200,
            )
            repeat_metric["mean_individual_frame_si"] = individual_frame_si(modal_flux)
            repeat_metrics.append(repeat_metric)
        metric_names = [
            "snr_db",
            "mean_flux",
            "scintillation_index",
            "effective_frame_number",
            "mean_pairwise_power_correlation",
            "normalized_power_q05",
            "fade_probability_below_half_mean",
            "mean_individual_frame_si",
        ]
        metrics = {
            name: float(np.mean([item[name] for item in repeat_metrics]))
            for name in metric_names
        }
        snr_values = np.asarray(
            [item["snr_db"] for item in repeat_metrics], dtype=float
        )
        snr_sem = float(stats.sem(snr_values)) if repeats > 1 else 0.0
        t_multiplier = float(stats.t.ppf(0.975, repeats - 1)) if repeats > 1 else 0.0
        metrics["snr_db_ci95_low"] = metrics["snr_db"] - t_multiplier * snr_sem
        metrics["snr_db_ci95_high"] = metrics["snr_db"] + t_multiplier * snr_sem
        metrics["snr_db_sem_across_repeats"] = snr_sem
        metrics["mean_transmitter_overlap_sq"] = float(np.mean(overlap_values))
        metrics["temporal_si_reduction_factor"] = metrics[
            "mean_individual_frame_si"
        ] / max(metrics["scintillation_index"], 1e-30)
        metrics["per_repeat_snr_db_std"] = (
            float(np.std([item["snr_db"] for item in repeat_metrics], ddof=1))
            if repeats > 1
            else 0.0
        )
        metrics["per_repeat"] = repeat_metrics
        beam_metrics[kind] = metrics
        saved_flux[kind] = torch.stack(modal_chunks, dim=0)
        torch.cuda.empty_cache()
    formal = {
        "source_result": str(result_path.resolve()),
        "link": summary["link"],
        "frame_count": frame_count,
        "validation_repeats": repeats,
        "blocks_per_repeat": blocks,
        "batch_size": batch_size,
        "atmospheric_samples_per_beam": repeats * blocks * batch_size,
        "seed": seed,
        "PCB": beam_metrics["PCB"],
        "SMS": beam_metrics["SMS"],
        "sms_minus_pcb_snr_db": beam_metrics["SMS"]["snr_db"]
        - beam_metrics["PCB"]["snr_db"],
    }
    torch.save(
        {
            "metadata": {
                key: value for key, value in formal.items() if key not in ("PCB", "SMS")
            },
            "PCB": saved_flux["PCB"],
            "SMS": saved_flux["SMS"],
        },
        flux_path,
    )
    output_path.write_text(
        json.dumps(formal, indent=2, allow_nan=True), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "case": result_path.parent.name,
                "M": frame_count,
                "PCB": beam_metrics["PCB"]["snr_db"],
                "SMS": beam_metrics["SMS"]["snr_db"],
                "gain": formal["sms_minus_pcb_snr_db"],
            }
        ),
        flush=True,
    )
    return formal


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=(Path(__file__).resolve().parent / "logs/ol_sms_pcb_3param"),
    )
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--blocks", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--bootstrap-repeats", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2026082001)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    result_paths = sorted(args.root.glob("*/result.pt"))
    summaries = []
    for result_path in result_paths:
        print(f"Formal validation: {result_path.parent.name}", flush=True)
        summaries.append(
            validate_one(
                result_path,
                args.repeats,
                args.blocks,
                args.batch_size,
                args.seed,
                args.bootstrap_repeats,
                args.force,
            )
        )
    (args.root / "formal_holdout_all.json").write_text(
        json.dumps(summaries, indent=2, allow_nan=True), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
