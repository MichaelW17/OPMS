"""Recompute the fixed-source propagation-step sensitivity experiment."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from scipy import stats

from propagation import LinkConfig, UnifiedWOS, metrics_from_modal_flux
from sources import SMSParameters, synthesize_sms


ROOT = Path(__file__).resolve().parent
ARCHIVE = ROOT / "logs" / "ol_sms_pcb_3param"
DEFAULT_OUT = ROOT / "logs" / "ol_s7_step_sensitivity"
CASES = ("small_rx", "long_rx20", "strong_small_rx")
METHODS = (("OPMS", "SMS"), ("GSM", "PCB"))
FRAME_COUNTS = (1, 2, 4, 10)
WOS_STEPS = (5, 7, 10)


def load_source(case: str, frame_count: int, method: str, evaluator: UnifiedWOS) -> torch.Tensor:
    path = ARCHIVE / f"{case}_M{frame_count}" / "result.pt"
    saved = torch.load(path, map_location="cpu", weights_only=False)
    summary = saved["summary"]
    selected = summary["selected"][method]
    if method == "PCB":
        return evaluator.pcb_source(
            torch.tensor(selected["wg_m"], device=evaluator.device),
            torch.tensor(selected["lcr_m"], device=evaluator.device),
            torch.tensor(selected["normalized_curvature"], device=evaluator.device),
            frame_count,
            evaluator.generator(9021001 + frame_count),
        )
    model = SMSParameters(
        frame_count,
        int(summary["optimization"]["sms_n_side"]),
        9021001 + frame_count,
        float(summary["optimization"].get("sms_wg_max", 80e-3)),
        evaluator.device,
    )
    model.load_state_dict(saved["selected_model_states"][method])
    return synthesize_sms(evaluator, model, normalized_curvature=1.0)


def link_config(case: str, steps: int) -> LinkConfig:
    summary = json.loads((ARCHIVE / f"{case}_M10" / "summary.json").read_text())
    link = dict(summary["link"])
    link["wos_steps"] = steps
    return LinkConfig(**link)


def evaluate_one(source: torch.Tensor, evaluator: UnifiedWOS, seed: int,
                 repeats: int, samples: int, batch_size: int) -> dict:
    if samples % batch_size:
        raise ValueError("samples must be divisible by batch_size")
    blocks = samples // batch_size
    values = []
    for repeat in range(repeats):
        flux = evaluator.collect_modal_flux(
            source, blocks, batch_size, seed + 100000 * repeat
        )
        values.append(metrics_from_modal_flux(flux, evaluator.config.duale_b)["snr_db"])
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    sem = float(stats.sem(values))
    half = float(stats.t.ppf(0.975, repeats - 1) * sem)
    return {
        "mean_snr_db": mean,
        "ci95_low_db": mean - half,
        "ci95_high_db": mean + half,
        "per_repeat_snr_db": values.tolist(),
        "repeats": repeats,
        "atmospheres_per_repeat": samples,
        "batch_size": batch_size,
        "batches_per_repeat": blocks,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", nargs="+", default=list(CASES), choices=CASES)
    parser.add_argument("--repeats", type=int, default=16)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026101701)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for case in args.cases:
        for method, archive_method in METHODS:
            for frame_count in FRAME_COUNTS:
                # Build the field once on the K=5 grid. K changes only propagation.
                source_evaluator = UnifiedWOS(link_config(case, 5))
                source = load_source(case, frame_count, archive_method, source_evaluator)
                for steps in WOS_STEPS:
                    output_path = args.output / f"{case}_{method}_M{frame_count}_K{steps}.json"
                    if output_path.exists():
                        result = json.loads(output_path.read_text())
                    else:
                        evaluator = UnifiedWOS(link_config(case, steps))
                        result = evaluate_one(
                            source, evaluator,
                            args.seed + 1000000 * args.cases.index(case)
                            + 10000 * (0 if method == "OPMS" else 1)
                            + 100 * frame_count + steps,
                            args.repeats, args.samples, args.batch_size,
                        )
                        result.update({"case": case, "method": method,
                                       "M": frame_count, "K": steps})
                        output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
                    rows.append({"case": case, "method": method, "M": frame_count,
                                 "K": steps, **{k: result[k] for k in
                                 ("mean_snr_db", "ci95_low_db", "ci95_high_db",
                                  "repeats", "atmospheres_per_repeat", "batch_size",
                                  "batches_per_repeat")}})
                    print(case, method, f"M={frame_count}", f"K={steps}",
                          f"{result['mean_snr_db']:.4f} dB", flush=True)
    with (args.output / "s7_step_sensitivity.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
