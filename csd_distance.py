"""Deterministic, grid-based CSD-distance analysis for the completed 1024-start study.

GSM kernels are evaluated directly at deterministic, nested Sobol quadrature
points snapped to the experiment's 512x512 aperture grid.  OPMS distances use
the frozen optimized fields and exact discrete grid inner products.  The script
records nested-quadrature/grid-coarsening checks alongside all outputs.
"""

from __future__ import annotations

import csv
import json
import math
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from sources import SMSParameters, synthesize_sms
from propagation import LinkConfig, UnifiedWOS

ROOT = Path(__file__).resolve().parent
RUN_ROOT = ROOT / "logs" / "ol_joint_bs16_sobol1024_common1000_20260924"
OUT = RUN_ROOT / "analysis_csd_deterministic"
FIG_DIR = ROOT / "results"
FRAMES = 10
N_SIDE = 8
OPMS_WG_MAX = 80e-3
Q_LEVELS = (16_384, 65_536, 262_144)
GRID_LEVELS = (1, 2, 4)
BLOCK_ENDPOINTS = 12


def write_progress(
    stage: str,
    link: str | None = None,
    kind: str | None = None,
    done: int | None = None,
    total: int | None = None,
    detail: str | None = None,
) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "running",
        "stage": stage,
        "link": link,
        "kind": kind,
        "done": done,
        "total": total,
        "detail": detail,
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (OUT / "progress.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_inputs():
    config = json.loads((RUN_ROOT / "configuration.json").read_text(encoding="utf-8"))
    rows = list(
        csv.DictReader(
            (RUN_ROOT / "candidate_summary.csv").open(
                "r", encoding="utf-8-sig", newline=""
            )
        )
    )
    by_key = {(r["link"], r["kind"], int(r["run"])): r for r in rows}
    if len(rows) != 8192:
        raise RuntimeError(f"Expected 8192 candidate rows, found {len(rows)}")
    return config, by_key


def selection(config, by_key, link: str, kind: str):
    scores = np.asarray(
        [float(by_key[(link, kind, i)]["screening_snr_db"]) for i in range(1024)]
    )
    baseline = float(np.mean(np.sort(scores)[-16:]))
    threshold = baseline - 0.5
    mask = scores >= threshold
    return scores, baseline, threshold, np.flatnonzero(mask)


def deterministic_aperture_pairs(
    evaluator: UnifiedWOS, count: int
) -> tuple[torch.Tensor, ...]:
    """Nested unscrambled Sobol points, mapped to the common 512-grid disk."""
    device = evaluator.device
    n = evaluator.config.n
    delta = evaluator.config.delta
    radius = evaluator.config.transmitter_radius
    # Rejection caused by snapping to a grid cell outside the hard pupil is tiny.
    # Oversample deterministically and retain the first `count` valid pairs.
    engine = torch.quasirandom.SobolEngine(dimension=4, scramble=False)
    accepted1: list[torch.Tensor] = []
    accepted2: list[torch.Tensor] = []
    kept = 0
    while kept < count:
        need = count - kept
        u = engine.draw(max(need * 2, 4096)).to(torch.float64)
        r1 = radius * torch.sqrt(u[:, 0])
        t1 = (2.0 * math.pi) * u[:, 1]
        r2 = radius * torch.sqrt(u[:, 2])
        t2 = (2.0 * math.pi) * u[:, 3]
        x1, y1 = r1 * torch.cos(t1), r1 * torch.sin(t1)
        x2, y2 = r2 * torch.cos(t2), r2 * torch.sin(t2)
        ix1 = torch.round((x1 + n * delta / 2) / delta).long().clamp(0, n - 1)
        iy1 = torch.round((y1 + n * delta / 2) / delta).long().clamp(0, n - 1)
        ix2 = torch.round((x2 + n * delta / 2) / delta).long().clamp(0, n - 1)
        iy2 = torch.round((y2 + n * delta / 2) / delta).long().clamp(0, n - 1)
        valid = ((ix1 - n // 2) ** 2 + (iy1 - n // 2) ** 2 < (radius / delta) ** 2) & (
            (ix2 - n // 2) ** 2 + (iy2 - n // 2) ** 2 < (radius / delta) ** 2
        )
        idx1 = ix1[valid] * n + iy1[valid]
        idx2 = ix2[valid] * n + iy2[valid]
        take = min(need, idx1.numel())
        if take:
            accepted1.append(idx1[:take])
            accepted2.append(idx2[:take])
            kept += take
    i1 = torch.cat(accepted1).to(device)
    i2 = torch.cat(accepted2).to(device)
    flat_x = evaluator.xx.reshape(-1)
    flat_y = evaluator.yy.reshape(-1)
    flat_r2 = evaluator.r2.reshape(-1)
    aperture = evaluator.transmitter_aperture.reshape(-1)
    return i1, i2, flat_x, flat_y, flat_r2, aperture


@torch.inference_mode()
def gsm_features(evaluator, link, endpoint_ids, pairs, by_key, qcount):
    first, second, x, y, r2, aperture = pairs
    first, second = first[:qcount], second[:qcount]
    wg_idx = torch.tensor(endpoint_ids, device=evaluator.device, dtype=torch.long)
    result = torch.empty(
        (len(endpoint_ids), qcount), dtype=torch.complex64, device=evaluator.device
    )
    dx = x[first] - x[second]
    dy = y[first] - y[second]
    sep2 = dx.square() + dy.square()
    r2a, r2b = r2[first], r2[second]
    ap_a, ap_b = aperture[first], aperture[second]
    power = evaluator.config.power
    for row, run in enumerate(endpoint_ids):
        endpoint = torch.load(
            RUN_ROOT / link / "gsm" / f"run_{int(run):03d}.pt",
            map_location="cpu",
            weights_only=False,
        )
        wg = float(endpoint["wg_m"])
        lcr = float(endpoint["lcr_m"])
        curv = float(endpoint["normalized_curvature"])
        base = torch.exp(-r2 / (wg * wg)) * aperture
        base = base * math.sqrt(power / float(base.square().sum().clamp_min(1e-30)))
        mag = base[first] * base[second] * torch.exp(-sep2 / (lcr * lcr))
        phi1 = -evaluator.k * r2a * curv / (2.0 * evaluator.config.distance)
        phi2 = -evaluator.k * r2b * curv / (2.0 * evaluator.config.distance)
        result[row] = torch.polar(mag, phi1 - phi2)
    return result


def distance_from_csd_inner(inner: torch.Tensor) -> torch.Tensor:
    norm = torch.diagonal(inner).clamp_min(1e-30)
    d2 = (norm[:, None] + norm[None, :] - 2.0 * inner).clamp_min(0.0)
    return torch.sqrt(d2 / (norm[:, None] + norm[None, :]).clamp_min(1e-30))


def qmc_distance(features: torch.Tensor, count: int) -> np.ndarray:
    f = features[:, :count]
    inner = torch.real(f @ f.conj().T).to(torch.float64)
    return distance_from_csd_inner(inner).cpu().numpy()


@torch.inference_mode()
def opms_fields(evaluator, link: str, endpoint_ids):
    # Preallocate once: torch.stack(list_of_fields) would transiently double
    # the already sizeable 10 x 512 x 512 complex field store.
    npx = evaluator.config.n * evaluator.config.n
    fields = torch.empty(
        (len(endpoint_ids), FRAMES, npx), dtype=torch.complex64, device="cpu"
    )
    for k, run in enumerate(endpoint_ids):
        path = RUN_ROOT / link / "opms" / f"run_{int(run):03d}.pt"
        ck = torch.load(path, map_location="cpu", weights_only=False)
        model = SMSParameters(FRAMES, N_SIDE, 0, OPMS_WG_MAX, evaluator.device)
        model.load_state_dict(ck["model_state"])
        model.eval()
        source = synthesize_sms(evaluator, model, normalized_curvature=1.0)
        fields[k].copy_(source.reshape(FRAMES, -1).cpu())
        if (k + 1) % 32 == 0 or k + 1 == len(endpoint_ids):
            write_progress("synthesizing_opms", link, "OPMS", k + 1, len(endpoint_ids))
    return fields  # [endpoint, frame, 512*512]


@torch.inference_mode()
def opms_distance(fields_cpu: torch.Tensor, stride: int, link: str):
    e, m, npx = fields_cpu.shape
    nside = int(round(math.sqrt(npx)))
    grid = fields_cpu.reshape(e, m, nside, nside)
    out = torch.empty((e, e), dtype=torch.float64)
    blocks = math.ceil(e / BLOCK_ENDPOINTS)
    total = blocks * (blocks + 1) // 2
    step = 0
    for bi in range(0, e, BLOCK_ENDPOINTS):
        a = (
            grid[bi : bi + BLOCK_ENDPOINTS, :, ::stride, ::stride]
            .reshape(min(BLOCK_ENDPOINTS, e - bi) * m, -1)
            .to("cuda", non_blocking=False)
        )
        ba = a.shape[0]
        ba //= m
        for bj in range(bi, e, BLOCK_ENDPOINTS):
            if bj == bi:
                b = a
            else:
                bb_endpoints = min(BLOCK_ENDPOINTS, e - bj)
                b = (
                    grid[bj : bj + BLOCK_ENDPOINTS, :, ::stride, ::stride]
                    .reshape(bb_endpoints * m, -1)
                    .to("cuda", non_blocking=False)
                )
            bb = b.shape[0] // m
            gram = a @ b.conj().T
            inner = gram.abs().square().reshape(ba, m, bb, m).sum(
                dim=(1, 3), dtype=torch.float64
            ) / float(m * m)
            i0, j0 = bi, bj
            out[i0 : i0 + ba, j0 : j0 + bb] = inner.cpu()
            if bi != bj:
                out[j0 : j0 + bb, i0 : i0 + ba] = inner.T.cpu()
            step += 1
            write_progress(
                "opms_grid_gram", link, "OPMS", step, total, f"stride={stride}"
            )
            if bj != bi:
                del b
        del a
    d = distance_from_csd_inner(out.to("cuda")).cpu().numpy()
    return d


def unique_upper(d, ids):
    block = d[np.ix_(np.arange(len(ids)), np.arange(len(ids)))]
    return block[np.triu_indices(block.shape[0], 1)]


def convergence_metrics(reference, current, ids):
    a = reference[np.ix_(ids, ids)][np.triu_indices(len(ids), 1)]
    b = current[np.ix_(ids, ids)][np.triu_indices(len(ids), 1)]
    return {
        "pearson_r": float(np.corrcoef(a, b)[0, 1]),
        "median_absolute_difference": float(np.median(np.abs(a - b))),
        "max_absolute_difference": float(np.max(np.abs(a - b))),
        "median_reference": float(np.median(a)),
        "median_current": float(np.median(b)),
    }


def save_figure(results):
    colors = {"A": "#0072B2", "B": "#D55E00", "C": "#009E73", "D": "#CC79A7"}
    fig, ax = plt.subplots(figsize=(7.0, 4.7), constrained_layout=True)
    for link in "ABCD":
        for kind, style in (("GSM", "--"), ("OPMS", "-")):
            vals = np.asarray(
                results[link][kind]["selected_pair_distances"], dtype=float
            )
            vals.sort()
            p = np.arange(1, len(vals) + 1) / len(vals)
            ax.plot(
                vals,
                p,
                color=colors[link],
                linestyle=style,
                linewidth=1.4,
                label=f"{link} {kind} (n={results[link][kind]['selected_count']})",
            )
    ax.set(
        xlabel=r"Normalized CSD Frobenius distance $d_{ij}$",
        ylabel="Cumulative fraction of endpoint pairs",
        xlim=(0, 1.02),
        ylim=(0, 1.01),
    )
    ax.grid(alpha=0.2, linewidth=0.5)
    ax.legend(ncol=2, fontsize=8, frameon=False)
    fig.savefig(FIG_DIR / "Fig6_CSD_distance_sobol1024.png", dpi=300)
    fig.savefig(FIG_DIR / "Fig6_CSD_distance_sobol1024.pdf")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    cfg, by_key = load_inputs()
    results = {}
    all_progress = {
        "status": "running",
        "stage": "initializing",
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (OUT / "progress.json").write_text(
        json.dumps(all_progress, indent=2), encoding="utf-8"
    )
    torch.backends.cuda.matmul.allow_tf32 = False
    for link in "ABCD":
        linkcfg = LinkConfig(**cfg["links"][link])
        evaluator = UnifiedWOS(linkcfg, device="cuda")
        pairs = deterministic_aperture_pairs(evaluator, max(Q_LEVELS))
        results[link] = {}
        for kind in ("GSM", "OPMS"):
            scores, baseline, threshold, ids = selection(cfg, by_key, link, kind)
            record = {
                "selected_count": int(len(ids)),
                "all_endpoints": 1024,
                "reference_mean_top16_snr_db": baseline,
                "selection_threshold_snr_db": threshold,
                "selected_endpoint_indices_zero_based": ids.tolist(),
                "selected_validation_snr_db": scores[ids].tolist(),
            }
            if kind == "GSM":
                write_progress(
                    "gsm_deterministic_aperture_quadrature", link, kind, 0, 1024
                )
                f = gsm_features(evaluator, link, ids, pairs, by_key, max(Q_LEVELS))
                matrices = {str(q): qmc_distance(f, q) for q in Q_LEVELS}
                final = matrices[str(max(Q_LEVELS))]
                record["integration"] = {
                    "method": "unscrambled 4-D Sobol deterministic quadrature; each point snapped to the 512x512 hard-aperture grid",
                    "nested_pair_counts": list(Q_LEVELS),
                    "no_random_pair_sampling": True,
                }
                record["convergence"] = {
                    str(q): convergence_metrics(
                        final, matrices[str(q)], np.arange(len(ids))
                    )
                    for q in Q_LEVELS[:-1]
                }
                record["median_pair_distance"] = float(
                    np.median(final[np.triu_indices(len(ids), 1)])
                )
                record["pair_distance_iqr"] = float(
                    np.percentile(final[np.triu_indices(len(ids), 1)], 75)
                    - np.percentile(final[np.triu_indices(len(ids), 1)], 25)
                )
                record["selected_pair_distances"] = final[
                    np.triu_indices(len(ids), 1)
                ].tolist()
                np.save(
                    OUT / f"link_{link}_gsm_distances_{max(Q_LEVELS)}.npy",
                    final.astype(np.float32),
                )
            else:
                fields = opms_fields(evaluator, link, ids)
                mats = {str(s): opms_distance(fields, s, link) for s in GRID_LEVELS}
                final = mats["1"]
                record["integration"] = {
                    "method": "exact empirical CSD Gram identity on all frozen fields on the 512x512 grid",
                    "grid_spacing_m": linkcfg.delta,
                    "grid_side": linkcfg.n,
                    "grid_refinement_strides": list(GRID_LEVELS),
                    "no_random_field_sampling": True,
                }
                record["convergence"] = {
                    str(s): convergence_metrics(
                        final, mats[str(s)], np.arange(len(ids))
                    )
                    for s in GRID_LEVELS[1:]
                }
                vals = final[np.triu_indices(len(ids), 1)]
                record["median_pair_distance"] = float(np.median(vals))
                record["pair_distance_iqr"] = float(
                    np.percentile(vals, 75) - np.percentile(vals, 25)
                )
                record["selected_pair_distances"] = vals.tolist()
                np.save(
                    OUT / f"link_{link}_opms_distances_grid512.npy",
                    final.astype(np.float32),
                )
                del fields, mats
            results[link][kind] = record
            print(
                f"{link} {kind}: selected={len(ids)}, median d={record['median_pair_distance']:.5f}, IQR={record['pair_distance_iqr']:.5f}",
                flush=True,
            )
            (OUT / "summary_partial.json").write_text(
                json.dumps(results, indent=2), encoding="utf-8"
            )
        del evaluator, pairs
        torch.cuda.empty_cache()
    summary = {
        "protocol": {
            "experiment_root": str(RUN_ROOT),
            "starts_per_link_class": 1024,
            "screening_atmosphere_pairs_per_endpoint": 1000,
            "high_performance_reference": "mean of top 16 screening SNR values among all 1024 endpoints in each link/class",
            "selection": "validated screening SNR >= reference - 0.5 dB",
            "distance": "d_ij=sqrt(||W_i-W_j||_F^2/(||W_i||_F^2+||W_j||_F^2))",
            "gsm": "analytic GSM CSD kernel evaluated by nested deterministic quadrature on the 512x512 transmitter grid",
            "opms": "empirical CSD from frozen M=10 fields; full 512x512 grid discrete sums",
            "common_transmitter_aperture_radius_m": 0.05,
            "outputs": str(OUT),
        },
        "links": results,
    }
    (OUT / "csd_sobol1024_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    with (OUT / "csd_sobol1024_summary.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as h:
        w = csv.writer(h)
        w.writerow(
            [
                "link",
                "class",
                "selected_count",
                "baseline_top16_snr_db",
                "threshold_snr_db",
                "median_distance",
                "iqr_distance",
            ]
        )
        for link in "ABCD":
            for kind in ("GSM", "OPMS"):
                r = results[link][kind]
                w.writerow(
                    [
                        link,
                        kind,
                        r["selected_count"],
                        r["reference_mean_top16_snr_db"],
                        r["selection_threshold_snr_db"],
                        r["median_pair_distance"],
                        r["pair_distance_iqr"],
                    ]
                )
    save_figure(results)
    (OUT / "progress.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "summary": str(OUT / "csd_sobol1024_summary.json"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
