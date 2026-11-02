from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerBase
from matplotlib.lines import Line2D
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform

ROOT = Path(__file__).resolve().parent / 'data/csd'
OUT = Path(__file__).resolve().parent / 'results'
OUT.mkdir(parents=True, exist_ok=True)
summary = json.loads((ROOT / "summary.json").read_text(encoding="utf-8"))
links = ["A", "B", "C", "D"]
colors = {"A": "#0072B2", "B": "#D55E00", "C": "#009E73", "D": "#CC79A7"}

def upper_triangle(path):
    a = np.load(path, mmap_mode="r")
    return np.asarray(a[np.triu_indices(a.shape[0], k=1)], dtype=np.float32)

def ordered_matrix(path):
    a = np.asarray(np.load(path, mmap_mode="r"), dtype=np.float64)
    order = leaves_list(linkage(squareform(a, checks=False), method="average", optimal_ordering=False))
    return a[np.ix_(order, order)]

plt.rcParams.update({"font.family": "Arial", "font.size": 7.4, "axes.labelsize": 7.6,
                     "axes.titlesize": 8, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
                     "legend.fontsize": 6.4, "axes.linewidth": 0.65, "pdf.fonttype": 42,
                     "ps.fonttype": 42, "savefig.dpi": 500})
fig = plt.figure(figsize=(3.25, 2.75), constrained_layout=False)
gs = fig.add_gridspec(2, 2, height_ratios=[0.82, 1.45], hspace=0.42, wspace=0.33,
                      left=0.12, right=0.88, top=0.94, bottom=0.10)
ax0 = fig.add_subplot(gs[0, :])
for l in links:
    for kind, ls in (("GSM", "--"), ("OPMS", "-")):
        suffix = "262144.npy" if kind == "GSM" else "grid512.npy"
        vals = upper_triangle(ROOT / f"link_{l}_{kind.lower()}_distances_{suffix}")
        vals.sort()
        y = np.arange(1, len(vals) + 1, dtype=np.float64) / len(vals)
        ax0.plot(vals, y, color=colors[l], ls=ls, lw=0.9,
                 label=f"Link {l}" if kind == "GSM" else "_nolegend_")
ax0.set_xlim(0, 1)
ax0.set_ylim(0, 1)
ax0.set_xlabel(r"Normalized Frobenius distance, $d_{ij}$")
ax0.set_ylabel("Cumulative fraction")
ax0.text(0.02, 0.98, "(a)", transform=ax0.transAxes, ha="left", va="top",
         bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=1))
ax0.grid(True, color="0.88", lw=0.45)
class DualLineHandler(HandlerBase):
    def create_artists(self, legend, original, xdescent, ydescent, width, height, fontsize, trans):
        color = original.get_color()
        return [Line2D([-xdescent, width-xdescent], [y,y], color=color,
                       linestyle=style, linewidth=0.9, transform=trans)
                for y,style in [(-ydescent+height*0.82,"-"),(-ydescent+height*0.18,"--")]]

handles=[Line2D([],[],color=colors[l]) for l in links]
ax0.legend(handles, [f"Link {l}" for l in links],
           handler_map={Line2D:DualLineHandler()}, ncol=4, frameon=False,
           loc="lower right", columnspacing=0.6, handlelength=1.4, handleheight=1.15)

im = None
for ax, kind in ((fig.add_subplot(gs[1, 0]), "gsm"),
                 (fig.add_subplot(gs[1, 1]), "opms")):
    suffix = "262144.npy" if kind == "gsm" else "grid512.npy"
    mat = ordered_matrix(ROOT / f"link_B_{kind}_distances_{suffix}")
    im = ax.imshow(mat, origin="lower", vmin=0, vmax=1, cmap="viridis", interpolation="nearest", rasterized=True)
    ax.text(0.02, 0.98, "(b)" if kind == "gsm" else "(c)", transform=ax.transAxes,
            ha="left", va="top", color="white",
            bbox=dict(facecolor="black", edgecolor="none", alpha=0.45, pad=1))
    ax.set_xlabel("Average-linkage order")
    ax.set_ylabel("Linkage order" if kind == "gsm" else "")
    if kind == "gsm":
        ax.yaxis.label.set_size(6.5)
    ax.set_xticks([0, (mat.shape[0]-1)//2, mat.shape[0]-1], ["1", str(mat.shape[0]//2), str(mat.shape[0])])
    ax.set_yticks([0, (mat.shape[0]-1)//2, mat.shape[0]-1], ["1", str(mat.shape[0]//2), str(mat.shape[0])])
    ax.tick_params(length=2, pad=1.5)
cax = fig.add_axes([0.905, 0.145, 0.018, 0.31])
cb = fig.colorbar(im, cax=cax)
cb.set_label(r"$d_{ij}$", rotation=90, labelpad=3)
cb.ax.tick_params(labelsize=6, length=2, pad=1)

for ext in ("png", "pdf"):
    fig.savefig(OUT / f"fig4_csd.{ext}", dpi=500 if ext == "png" else None,
                bbox_inches="tight", facecolor="white")
plt.close(fig)
print(OUT / "fig4_csd.png")
