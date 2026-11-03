"""Export wHG validation SNRs and intervals from the saved receiver powers."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from supplement_links import CASE_ORDER, CASE_LABELS, SAMPLING_CASE_INDEX


ROOT = Path(__file__).resolve().parent
CASES = CASE_ORDER


def export(output: Path) -> list[dict]:
    labels = CASE_LABELS
    rows = []
    for modes in (10, 64):
        directory = (
            "ol_weighted_hg_pseudomode_validation_20260901"
            if modes == 10 else "ol_weighted_hg_m64"
        )
        for case in CASES:
            case_index = SAMPLING_CASE_INDEX[case]
            path = ROOT / f"logs/{directory}/{case}_M10/weighted_hg_result.pt"
            record = torch.load(path, map_location="cpu", weights_only=False)
            summary = record["summary"]
            flux = record["modal_flux"].double().numpy()
            weights = record["weights"].double().numpy()
            assert flux.shape == (2000, modes)
            assert np.isfinite(flux).all() and np.isfinite(weights).all()
            assert (weights >= 0).all() and np.isclose(weights.sum(), 1)
            if modes == 64:
                assert summary["holdout"]["mode_indices"] == [
                    [p, q] for p in range(8) for q in range(8)
                ], "Recompute wHG64 with p,q=0,...,7 before exporting."
            power = flux @ weights
            eta = summary["link"]["duale_b"]
            mean = power.mean()
            si = max(np.mean(power ** 2) / mean ** 2 - 1, 0)
            snr = float(10 * np.log10(1 / np.sqrt(eta / mean + si)))
            if modes == 64:
                seed = summary["random_streams"]["bootstrap_seed"]
            else:
                seed = 2026090101 + 200000 + case_index
            indices = np.random.default_rng(seed).integers(
                0, len(power), size=(2000, len(power))
            )
            samples = power[indices]
            means = samples.mean(axis=1)
            sis = np.maximum(np.mean(samples ** 2, axis=1) / means ** 2 - 1, 0)
            values = 10 * np.log10(1 / np.sqrt(eta / means + sis))
            low, high = np.quantile(values, [0.025, 0.975])
            for computed, key in (
                (snr, "snrp_legacy_db_10log"),
                (low, "snr_db_ci95_low"),
                (high, "snr_db_ci95_high"),
            ):
                assert abs(computed - summary["holdout"][key]) < 1e-9, (path, key)
            rows.append(dict(
                link=labels[case], case=case, mode_count=modes, snr_db=snr,
                ci95_low_db=float(low), ci95_high_db=float(high),
                atmospheric_samples=len(power), interval="percentile bootstrap",
            ))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "data/controls/weighted_hg_comparison.csv")
    args = parser.parse_args()
    rows = export(args.output)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
