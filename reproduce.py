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


def controls(output):
    from supplementary_figures import draw
    draw(output)


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
        default=100,
        help="100 uses the current success criterion; 16 reproduces the earlier classification.",
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
        import subprocess, sys
        subprocess.run([sys.executable,str(ROOT/"verify_submission.py")],check=True)
    workflow(args.output)
    import landscape

    landscape.plot_result(
        SimpleNamespace(
            output=ROOT / "logs/ol_pcb_collimated_multibasin_100x100_20260829",
            trajectory_source=ROOT / "logs/ol_pcb_collimated_multibasin_20260829",
            figure=args.output / "fig2_landscape.png",
        )
    )
    import plot_fig3
    plot_fig3.main(args.output)
    import runpy, os
    os.environ["OPMS_RESULTS"]=str(args.output.resolve())
    runpy.run_path(str(ROOT / "plot_fig4.py"), run_name="__main__")
    import subprocess, sys
    subprocess.run([sys.executable, str(ROOT/"experiments/local_manifolds/plot_archived.py")],check=True)
    controls(args.output)
    print(f"Figures and statistics saved to {args.output}", flush=True)


if __name__ == "__main__":
    main()
