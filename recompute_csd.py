"""Recompute top-100-selected CSD distances from the frozen endpoint parameters."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import csd_distance as csd
from archive import unpack
from propagation import LinkConfig, UnifiedWOS
from sources import SMSParameters, synthesize_sms

ROOT = Path(__file__).resolve().parent


@torch.inference_mode()
def opms_distance(evaluator, link, ids, endpoint_root):
    # Cropping exactly zero pixels keeps the full-grid trace identity unchanged.
    mask = evaluator.transmitter_aperture.bool().reshape(-1)
    fields = torch.empty(
        (len(ids), 10, int(mask.sum())), dtype=torch.complex64, device=evaluator.device
    )
    for j, index in enumerate(ids):
        endpoint = torch.load(
            endpoint_root / link / "opms" / f"run_{index:03d}.pt",
            map_location="cpu",
            weights_only=False,
        )
        model = SMSParameters(10, 8, 0, 0.08, evaluator.device)
        model.load_state_dict(endpoint["model_state"])
        model.eval()
        fields[j] = synthesize_sms(evaluator, model, normalized_curvature=1.0).reshape(
            10, -1
        )[:, mask]
    norms = (fields @ fields.conj().transpose(1, 2)).abs().square().sum(
        (1, 2), dtype=torch.float64
    ) / 100
    matrix = np.empty((len(ids), len(ids)), dtype=np.float32)
    for start in range(0, len(ids), 10):
        group = fields[start : start + 10]
        gram = group.flatten(0, 1) @ fields.flatten(0, 1).conj().T
        inner = (
            gram.abs()
            .square()
            .reshape(len(group), 10, len(ids), 10)
            .sum((1, 3), dtype=torch.float64)
            / 100
        )
        denominator = norms[start : start + len(group), None] + norms[None, :]
        matrix[start : start + len(group)] = (
            torch.sqrt(((denominator - 2 * inner) / denominator).clamp_min(0))
            .cpu()
            .numpy()
        )
    np.fill_diagonal(matrix, 0)
    return matrix


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--links", nargs="+", choices=list("ABCD"), default=list("ABCD")
    )
    parser.add_argument("--output", type=Path, default=ROOT / "results/recomputed_csd")
    parser.add_argument("--endpoint-root", type=Path, default=ROOT / "work/endpoints")
    parser.add_argument("--quadrature-pairs", type=int, default=262144)
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit selected endpoints for an integration check; not a paper run.",
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    csd.RUN_ROOT = unpack(args.endpoint_root)
    csd.OUT = args.output
    config, by_key = csd.load_inputs()
    summary = {}
    for link in args.links:
        evaluator = UnifiedWOS(LinkConfig(**config["links"][link]), device=args.device)
        summary[link] = {}
        for kind in ("GSM", "OPMS"):
            scores = np.array(
                [float(by_key[link, kind, i]["screening_snr_db"]) for i in range(1024)]
            )
            reference = float(np.sort(scores)[-100:].mean())
            ids = np.flatnonzero(scores >= reference - 0.5)
            if args.limit is not None:
                ids = ids[: args.limit]
            if kind == "GSM":
                pairs = csd.deterministic_aperture_pairs(
                    evaluator, args.quadrature_pairs
                )
                features = csd.gsm_features(
                    evaluator, link, ids, pairs, by_key, args.quadrature_pairs
                )
                matrix = csd.qmc_distance(features, args.quadrature_pairs)
                del features, pairs
            else:
                matrix = opms_distance(evaluator, link, ids, args.endpoint_root)
            np.fill_diagonal(matrix, 0)
            suffix = str(args.quadrature_pairs) if kind == "GSM" else "grid512"
            np.save(
                args.output / f"link_{link}_{kind.lower()}_distances_{suffix}.npy",
                matrix,
            )
            summary[link][kind] = {
                "reference_top100_snr_db": reference,
                "selected_endpoint_indices_zero_based": ids.tolist(),
                "selected_count": len(ids),
                "median_distance": float(
                    np.median(matrix[np.triu_indices(len(ids), 1)])
                ),
            }
            print(link, kind, len(ids), "endpoints", flush=True)
        del evaluator
        torch.cuda.empty_cache()
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
