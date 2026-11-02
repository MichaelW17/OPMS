"""Plot six source/frame conditions after the full M3 extension passes verification."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


REPO = Path(__file__).resolve().parent
OUT = REPO / "results"
LINKS = "ABCD"
CONDITIONS = [
    ("GSM, M=3", "#70ACC4"),
    ("GSM, M=10", "#235E8D"),
    ("9-HG, M=3", "#DEA62F"),
    ("9-HG, M=10", "#C66435"),
    ("64-HG, M=3", "#66A268"),
    ("64-HG, M=10", "#79549C"),
]
CAPTION = (
    "Fig. 3. Optimization performance under 1024 scrambled Sobol initializations for Links A-D, "
    "comparing GSM and OPMS with 9 or 64 HG modes at M=3 and M=10. "
    "(a) Validated endpoint SNR distributions; horizontal bars mark the medians. "
    "(b) Optimization success rates. A run is successful if its validated SNR is no more than "
    "0.5 dB below the condition- and link-specific mean SNR of the top 100 endpoints. "
    "Error bars denote 95% Wilson score intervals for the binomial success proportions "
    "(1024 starts per condition and link)."
)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def scores_from_screening(root: Path, link: str, kind="opms") -> np.ndarray:
    folder = root / link / kind
    files = sorted(folder.glob("run_*_screening.json"), key=lambda p: int(p.name.split("_")[1]))
    if [p.name for p in files] != [f"run_{i:03d}_screening.json" for i in range(1024)]:
        raise RuntimeError(f"Expected 1024 screening records for {link} in {root}")
    values = np.array([float(read_json(p)["snr_db"]) for p in files], dtype=np.float64)
    if not np.isfinite(values).all():
        raise RuntimeError(f"Non-finite SNR values for {link} in {root}")
    return values


def top100(values: np.ndarray) -> float:
    return float(np.sort(values)[-100:].mean())


def wilson(successes: int, n: int, z: float = 1.959963984540054):
    p = successes / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def main(output=None):
    global OUT
    if output is not None: OUT=Path(output)
    OUT.mkdir(parents=True,exist_ok=True)
    with (REPO/'data/fig3_six_conditions/gsm_hg_success_summary.csv').open(encoding='utf-8-sig') as f: summary=list(csv.DictReader(f))
    for row in summary:
        for key in ['success_rate','ci95_low','ci95_high']:row[key]=float(row[key])
    with (REPO/'data/fig3_six_conditions/gsm_hg_endpoint_scores.csv').open(encoding='utf-8-sig') as f: endpoints=list(csv.DictReader(f))
    values={(l,label):np.array([float(x['snr_db']) for x in endpoints if x['link']==l and x['condition']==label]) for l in LINKS for label,_ in CONDITIONS}
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 15, "axes.labelsize": 16,
                         "xtick.labelsize": 14, "ytick.labelsize": 14, "legend.fontsize": 14,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42})
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 6.2), sharex=True)
    fig.subplots_adjust(left=0.105, right=0.985, bottom=0.095, top=0.86, hspace=0.13)
    positions0 = np.arange(4)
    offsets = np.linspace(-0.30, 0.30, len(CONDITIONS))
    width = 0.095
    for j, (label, color) in enumerate(CONDITIONS):
        pos = positions0 + offsets[j]
        violins = axes[0].violinplot([values[(link, label)] for link in LINKS], pos,
                                     widths=width, points=512, showmedians=True, showextrema=False)
        for body in violins["bodies"]:
            body.set_facecolor(color); body.set_edgecolor(color); body.set_alpha(0.55)
        violins["cmedians"].set_color(color)
        violins["cmedians"].set_linewidth(1.5)
        rates = np.array([next(r["success_rate"] for r in summary if r["link"] == link and r["condition"] == label) for link in LINKS])
        lows = np.array([next(r["ci95_low"] for r in summary if r["link"] == link and r["condition"] == label) for link in LINKS])
        highs = np.array([next(r["ci95_high"] for r in summary if r["link"] == link and r["condition"] == label) for link in LINKS])
        axes[1].bar(pos, rates, width=width, color=color, alpha=0.85, zorder=2)
        axes[1].errorbar(pos, rates, yerr=[np.maximum(rates - lows,0), np.maximum(highs - rates,0)], fmt="none", ecolor="black", capsize=3, lw=1.2, zorder=3)

    handles = [Line2D([0], [0], color=color, marker="s", linestyle="none", markersize=6, label=label)
               for label, color in CONDITIONS]
    fig.legend(handles=handles, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=3,
               handlelength=1.2, columnspacing=1.0)
    axes[0].set_ylabel("Validated SNR (dB)")
    axes[0].text(0.015, 0.97, "(a)", transform=axes[0].transAxes, ha="left", va="top", fontsize=16)
    axes[1].set_ylabel("Success rate")
    axes[1].set_xlabel("Link")
    axes[1].set_ylim(0, 1.08)
    axes[1].set_yticks(np.linspace(0, 1, 6))
    axes[1].text(0.015, 0.96, "(b)", transform=axes[1].transAxes, ha="left", va="top", fontsize=16)
    for ax in axes:
        ax.set_xticks(positions0)
        ax.set_xticklabels(list(LINKS))
        ax.grid(axis="y", alpha=0.2); ax.set_axisbelow(True)
    axes[0].tick_params(axis="x", labelbottom=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig3_gsm_hg_six_conditions.{ext}", dpi=400, bbox_inches="tight")
    plt.close(fig)


if __name__=='__main__':main()
