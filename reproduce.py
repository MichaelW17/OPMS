"""Rebuild the paper's numerical figures from the archived simulation data."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform

ROOT = Path(__file__).resolve().parent
COLORS = {"GSM": "#3478A8", "OPMS": "#D46A3A"}
CASES = ("small_rx", "long_rx20", "strong_small_rx")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def save(fig, output, name):
    fig.savefig(output / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(output / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def wilson(count, total):
    z = 1.959963984540054
    p = count / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = (
        z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    )
    return max(0, center - half), min(1, center + half)


def endpoint_statistics(output, reference_count=16):
    """Use top-16 for manuscript numbers or top-100 for its methods text."""
    rows, scores = [], {}
    for link in "ABCD":
        for beam in ("GSM", "OPMS"):
            records = read_json(
                ROOT / "data/multistart" / f"{link}_{beam.lower()}_metrics.json"
            )
            values = np.array([r["snr_db"] for r in records], dtype=np.float64)
            if values.shape != (1024,) or not np.isfinite(values).all():
                raise ValueError(
                    f"Incomplete or non-finite endpoint data: {link} {beam}"
                )
            reference = np.sort(values)[-reference_count:].mean()
            count = int((values >= reference - 0.5).sum())
            lower, upper = wilson(count, len(values))
            rows.append(
                dict(
                    link=link,
                    beam=beam,
                    starts=len(values),
                    reference_count=reference_count,
                    reference_snr_db=float(reference),
                    threshold_snr_db=float(reference - 0.5),
                    successes=count,
                    success_rate=count / len(values),
                    ci95_low=lower,
                    ci95_high=upper,
                    median_snr_db=float(np.median(values)),
                    sd_snr_db=float(values.std(ddof=1)),
                    iqr_snr_db=float(
                        np.percentile(values, 75) - np.percentile(values, 25)
                    ),
                )
            )
            scores[link, beam] = values
    write_csv(output / "fig3_summary.csv", rows)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), constrained_layout=True)
    for j, beam in enumerate(("GSM", "OPMS")):
        positions = np.arange(4) + (j - 0.5) * 0.32
        parts = axes[0].violinplot(
            [scores[link, beam] for link in "ABCD"],
            positions,
            widths=0.29,
            points=512,
            showmedians=True,
            showextrema=False,
        )
        for body in parts["bodies"]:
            body.set_facecolor(COLORS[beam])
            body.set_alpha(0.4)
        parts["cmedians"].set_color(COLORS[beam])
        group = [r for r in rows if r["beam"] == beam]
        p = np.array([r["success_rate"] for r in group])
        err = [
            [r["success_rate"] - r["ci95_low"] for r in group],
            [r["ci95_high"] - r["success_rate"] for r in group],
        ]
        axes[1].bar(positions, p, width=0.28, color=COLORS[beam], label=beam)
        axes[1].errorbar(positions, p, yerr=err, fmt="none", color="black", capsize=2)
    axes[0].set_ylabel("Validated SNR (dB)")
    axes[0].set_title("(a) Endpoint distributions")
    axes[1].set_ylabel("Success fraction")
    axes[1].set_ylim(0, 1.08)
    axes[1].set_title(f"(b) Within 0.5 dB of top-{reference_count} mean")
    axes[1].legend(frameon=False)
    for ax in axes:
        ax.set_xticks(range(4), list("ABCD"))
        ax.grid(axis="y", alpha=0.2)
    save(fig, output, "fig3_multistart")
    return rows


def verify_flux(output):
    """Recompute each SNR from raw powers, without importing PyTorch."""
    configuration = read_json(ROOT / "data/multistart/configuration.json")
    rows = []
    for link in "ABCD":
        b = configuration["links"][link]["duale_b"]
        for beam in ("gsm", "opms"):
            with np.load(
                ROOT / "data/multistart" / f"{link}_{beam}_flux.npz"
            ) as archive:
                modal = archive["modal_flux"]
            if modal.shape != (1024, 1000, 10):
                raise ValueError(f"Unexpected raw power shape: {modal.shape}")
            power = modal.astype(np.float64, copy=False).mean(axis=-1)
            mu = power.mean(axis=-1)
            si = (power * power).mean(axis=-1) / (mu * mu) - 1
            snr = 10 * np.log10(1 / np.sqrt(b / mu + np.maximum(si, 0)))
            recorded = np.array(
                [
                    r["snr_db"]
                    for r in read_json(
                        ROOT / "data/multistart" / f"{link}_{beam}_metrics.json"
                    )
                ]
            )
            error = float(np.max(np.abs(snr - recorded)))
            if error > 1e-9:
                raise AssertionError(
                    f"Raw powers disagree with archived SNR: {link} {beam}: {error}"
                )
            rows.append(
                dict(
                    link=link,
                    beam=beam,
                    starts=1024,
                    atmospheres_per_endpoint=1000,
                    frames=10,
                    maximum_snr_error_db=error,
                )
            )
            print(f"{link} {beam}: raw-power SNR agreement {error:.3g} dB", flush=True)
    write_csv(output / "raw_power_verification.csv", rows)


def csd_plot(output, reference_count=16):
    summary = read_json(ROOT / "data/csd/summary.json")
    fig, axes = plt.subplots(1, 3, figsize=(10, 3), constrained_layout=True)
    rows = []
    for j, link in enumerate("ABCD"):
        for beam in ("GSM", "OPMS"):
            suffix = "262144" if beam == "GSM" else "grid512"
            matrix = np.load(
                ROOT / "data/csd" / f"link_{link}_{beam.lower()}_distances_{suffix}.npy"
            )
            record = summary["links"][link][beam]
            if matrix.shape != (record["selected_count"], record["selected_count"]):
                raise ValueError("CSD matrix and endpoint selection disagree")
            if not np.allclose(matrix, matrix.T, atol=1e-6) or not np.allclose(
                np.diag(matrix), 0, atol=1e-6
            ):
                raise ValueError(
                    "CSD distance matrix must be symmetric with zero diagonal"
                )
            if reference_count != 100:
                metrics = read_json(
                    ROOT / "data/multistart" / f"{link}_{beam.lower()}_metrics.json"
                )
                scores = np.array([r["snr_db"] for r in metrics])
                threshold = np.sort(scores)[-reference_count:].mean() - 0.5
                selected_ids = np.flatnonzero(scores >= threshold)
                archived_ids = record["selected_endpoint_indices_zero_based"]
                positions = np.array(
                    [archived_ids.index(int(index)) for index in selected_ids]
                )
                matrix = matrix[np.ix_(positions, positions)]
            values = np.sort(matrix[np.triu_indices(len(matrix), 1)])
            axes[0].plot(
                values,
                np.arange(1, len(values) + 1) / len(values),
                linestyle="--" if beam == "GSM" else "-",
                color=f"C{j}",
                label=f"{link} {beam}",
            )
            rows.append(
                dict(
                    link=link,
                    beam=beam,
                    selected_count=len(matrix),
                    median_distance=float(np.median(values)),
                )
            )
            if link == "B":
                order = leaves_list(
                    linkage(squareform(matrix, checks=False), method="average")
                )
                ax = axes[1 if beam == "GSM" else 2]
                im = ax.imshow(
                    matrix[np.ix_(order, order)],
                    vmin=0,
                    vmax=1,
                    cmap="viridis",
                    rasterized=True,
                )
                ax.set_title(f"B {beam}: {len(matrix)} endpoints")
                ax.set_xticks([])
                ax.set_yticks([])
                fig.colorbar(im, ax=ax, shrink=0.75)
    axes[0].set_xlabel("Normalized Frobenius distance")
    axes[0].set_ylabel("Empirical CDF")
    axes[0].legend(fontsize=7, frameon=False, ncol=2)
    save(fig, output, "fig4_csd")
    write_csv(output / "fig4_summary.csv", rows)


def controls(output):
    rows = read_csv(ROOT / "data/controls/fig3_performance.csv")
    weighted = read_json(
        ROOT / "logs/ol_weighted_hg_pseudomode_validation_20260901/weighted_hg_all.json"
    )
    for record in weighted:
        metrics = record["holdout"]
        rows.append(
            dict(
                case=record["case"].removesuffix("_M10"),
                method="wHG",
                snr_p_db=metrics["snrp_legacy_db_10log"],
                ci95_low_db=metrics["snr_db_ci95_low"],
                ci95_high_db=metrics["snr_db_ci95_high"],
            )
        )
    write_csv(
        output / "fig5_performance.csv",
        [
            {
                k: r[k]
                for k in ("case", "method", "snr_p_db", "ci95_low_db", "ci95_high_db")
            }
            for r in rows
        ],
    )
    methods = ("SMS", "GSM", "CB", "Sharp", "Pin", "IOPB", "DAD", "wHG")
    fig, ax = plt.subplots(figsize=(6.0, 3.3), constrained_layout=True)
    for j, case in enumerate(CASES):
        group = [
            next(r for r in rows if r["case"] == case and r["method"] == m)
            for m in methods
        ]
        center = np.array([float(r["snr_p_db"]) for r in group])
        err = np.array(
            [
                [float(r["snr_p_db"]) - float(r["ci95_low_db"]) for r in group],
                [float(r["ci95_high_db"]) - float(r["snr_p_db"]) for r in group],
            ]
        )
        ax.errorbar(
            center,
            np.arange(len(methods)) + (j - 1) * 0.15,
            xerr=err,
            fmt="os^"[j],
            label=f"S{j+1}",
            capsize=2,
        )
    ax.set_yticks(range(len(methods)), ["OPMS" if m == "SMS" else m for m in methods])
    ax.invert_yaxis()
    ax.set_xlabel("Validated SNR (dB)")
    ax.legend(frameon=False)
    ax.grid(axis="x", alpha=0.2)
    save(fig, output, "fig5_structured_beams")
    frames = read_csv(ROOT / "data/controls/fig5_frame_budget.csv")
    basis = read_csv(ROOT / "data/controls/fig6_basis_performance.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7, 3), constrained_layout=True)
    for j, case in enumerate(CASES):
        for method in ("SMS", "GSM"):
            group = sorted(
                [r for r in frames if r["case"] == case and r["method"] == method],
                key=lambda r: int(r["frame_count"]),
            )
            axes[0].plot(
                [int(r["frame_count"]) for r in group],
                [float(r["snr_p_db"]) for r in group],
                linestyle="-" if method == "SMS" else "--",
                marker="os^"[j],
                color=f"C{j}",
                label=f'S{j+1} {"OPMS" if method=="SMS" else method}',
            )
    center = np.array([float(r["lg_minus_hg_snr_p_db"]) for r in basis])
    axes[1].errorbar(
        range(3),
        center,
        yerr=[
            [float(r["lg_minus_hg_snr_p_db"]) - float(r["ci95_low_db"]) for r in basis],
            [
                float(r["ci95_high_db"]) - float(r["lg_minus_hg_snr_p_db"])
                for r in basis
            ],
        ],
        fmt="o",
        capsize=3,
    )
    axes[1].axhline(0, color="gray", lw=0.6)
    axes[1].set_xticks(range(3), ["S1", "S2", "S3"])
    axes[1].set_ylabel("LG minus HG (dB)")
    axes[0].set_xlabel("Frame count")
    axes[0].set_ylabel("SNR (dB)")
    axes[0].legend(fontsize=7, ncol=2, frameon=False)
    save(fig, output, "fig6_frame_and_basis")


def workflow(output):
    from matplotlib.patches import FancyBboxPatch
    fig, ax = plt.subplots(figsize=(10, 2.2))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    labels = [
        "Source parameters",
        "Pseudo-mode\nsynthesis",
        "Split-step\npropagation",
        "Aperture powers\nand inverse SNR",
    ]
    for i, label in enumerate(labels):
        x = 0.125 + i * 0.25
        ax.add_patch(FancyBboxPatch((x - 0.10, 0.52), 0.20, 0.30,
                                   boxstyle='round,pad=0.01', facecolor='#eaf1f7', edgecolor='#3478A8'))
        ax.text(
            x,
            0.67,
            label,
            ha="center",
            va="center",
            fontsize=11,
        )
        if i < 3:
            ax.annotate(
                "",
                xy=(x + 0.14, 0.67),
                xytext=(x + 0.11, 0.67),
                arrowprops=dict(arrowstyle="->"),
            )
    ax.plot([0.875, 0.875, 0.125], [0.51, 0.3, 0.3], color='black', lw=1)
    ax.annotate(
        "",
        xy=(0.125, 0.51),
        xytext=(0.125, 0.3),
        arrowprops=dict(arrowstyle="->"),
    )
    ax.text(0.5, 0.13, "Automatic differentiation and Adam update", ha="center", fontsize=11)
    save(fig, output, "fig1_workflow")


def integrity():
    manifest = read_json(ROOT / "data_manifest.json")
    for record in manifest["files"]:
        path = ROOT / record["path"]
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != record["sha256"]:
            raise ValueError(f'Checksum mismatch: {record["path"]}')
    print(f'Verified {len(manifest["files"])} data files.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    parser.add_argument(
        "--reference-count",
        type=int,
        choices=(16, 100),
        default=16,
        help="16 matches manuscript numbers; 100 follows its methods text. See MANUSCRIPT_AUDIT.md.",
    )
    parser.add_argument(
        "--verify-flux",
        action="store_true",
        help="Recompute all 8192 endpoint SNRs from raw powers.",
    )
    parser.add_argument(
        "--verify-checksums",
        action="store_true",
        help="Check the archived data against SHA-256 hashes.",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.verify_checksums:
        integrity()
    if args.verify_flux:
        verify_flux(args.output)
    workflow(args.output)
    import landscape

    landscape.plot_result(
        SimpleNamespace(
            output=ROOT / "logs/ol_pcb_collimated_multibasin_100x100_20260829",
            trajectory_source=ROOT / "logs/ol_pcb_collimated_multibasin_20260829",
            figure=args.output / "fig2_landscape.png",
        )
    )
    endpoint_statistics(args.output, args.reference_count)
    csd_plot(args.output, args.reference_count)
    controls(args.output)
    print(f"Figures and statistics saved to {args.output}", flush=True)


if __name__ == "__main__":
    main()
