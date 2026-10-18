"""Extended weighted-HG control with 64 coherent modes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import weighted_hg_pseudomode_validation_20260901 as base
import torch


base.MODE_COUNT = 64
base.MODE_INDICES = base.first_hg_indices(base.MODE_COUNT)
base.OUTPUT_ROOT = Path(__file__).resolve().parent / "logs" / "ol_weighted_hg_m64"


def synthesize_weighted_hg_m64(evaluator, model):
    weights, waist, curvature = model.physical()
    repeated_waists = waist.expand(base.MODE_COUNT)
    max_order = max(max(index) for index in base.MODE_INDICES)
    one_dimensional = base.hermite_gaussian_1d(
        evaluator, repeated_waists, max_order + 1
    ).to(evaluator.complex_dtype)
    fields = []
    for mode_index, (x_order, y_order) in enumerate(base.MODE_INDICES):
        field = (
            one_dimensional[mode_index, x_order, :, None]
            * one_dimensional[mode_index, y_order, None, :]
        )
        fields.append(field)
    source = torch.stack(fields, dim=0)
    source = source * evaluator.curvature_phase(curvature)
    source = evaluator.normalize_each_frame(source)
    return source, weights


base.synthesize_weighted_hg = synthesize_weighted_hg_m64


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode-count", type=int, default=64, choices=(64,))
    parser.add_argument("--cases", nargs="+", choices=base.CASE_NAMES, default=list(base.CASE_NAMES))
    parser.add_argument("--runs", type=int, default=8)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--train-bs", type=int, default=6)
    parser.add_argument("--mc-blocks", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=0.025)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--selection-samples", type=int, default=128)
    parser.add_argument("--holdout-samples", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--bootstrap-repeats", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2026101801)
    parser.add_argument("--selection-seed", type=int, default=2026101901)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    base.MODE_COUNT = args.mode_count
    base.MODE_INDICES = base.first_hg_indices(base.MODE_COUNT)
    base.OUTPUT_ROOT = Path(__file__).resolve().parent / "logs" / f"ol_weighted_hg_m{args.mode_count}"
    base.OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    results = [
        base.run_case(case, base.CASE_NAMES.index(case), args)
        for case in args.cases
    ]
    (base.OUTPUT_ROOT / "weighted_hg_m64_all.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False, allow_nan=True),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
