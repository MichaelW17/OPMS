"Wave-optics experiment: ol weighted hg pseudomode validation 20260901."

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from fixed_source_ensemble_validation import metric_from_integrated_power
from sources import WG_MIN, bounded, hermite_gaussian_1d, inverse_bounded
from propagation import LinkConfig, UnifiedWOS


ROOT = Path(__file__).resolve().parent
REFERENCE_ROOT = ROOT / "logs" / "ol_sms_pcb_3param"
GSM_ROOT = ROOT / "logs" / "ol_resampled_gsm_ensemble_validation"
OUTPUT_ROOT = ROOT / "logs" / "ol_weighted_hg_pseudomode_validation_20260901"
CASE_NAMES = ("small_rx_M10", "long_rx20_M10", "strong_small_rx_M10")
MODE_COUNT = 10
WAIST_MAX = 100e-3
CURVATURE_MIN = 0.0
CURVATURE_MAX = 2.0


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=True),
        encoding="utf-8",
    )


def first_hg_indices(mode_count: int) -> tuple[tuple[int, int], ...]:
    indices: list[tuple[int, int]] = []
    total_order = 0
    while len(indices) < mode_count:
        for x_order in range(total_order + 1):
            y_order = total_order - x_order
            indices.append((x_order, y_order))
            if len(indices) == mode_count:
                break
        total_order += 1
    return tuple(indices)


MODE_INDICES = first_hg_indices(MODE_COUNT)


class WeightedHGParameters(nn.Module):
    def __init__(self, sobol_point: torch.Tensor, device: torch.device) -> None:
        super().__init__()
        probabilities = sobol_point[:MODE_COUNT].clamp(1e-4, 1.0 - 1e-4)
        logits = torch.log(probabilities) - torch.log1p(-probabilities)
        logits = logits - logits.mean()
        waist = WG_MIN + (WAIST_MAX - WG_MIN) * sobol_point[MODE_COUNT]
        curvature = (
            CURVATURE_MIN
            + (CURVATURE_MAX - CURVATURE_MIN) * sobol_point[MODE_COUNT + 1]
        )
        self.weight_logits = nn.Parameter(logits.to(device=device, dtype=torch.float32))
        self.raw_waist = nn.Parameter(
            torch.tensor(
                inverse_bounded(float(waist), WG_MIN, WAIST_MAX),
                device=device,
                dtype=torch.float32,
            )
        )
        self.raw_curvature = nn.Parameter(
            torch.tensor(
                inverse_bounded(float(curvature), CURVATURE_MIN, CURVATURE_MAX),
                device=device,
                dtype=torch.float32,
            )
        )

    def physical(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        weights = torch.softmax(self.weight_logits, dim=0)
        waist = bounded(self.raw_waist, WG_MIN, WAIST_MAX)
        curvature = bounded(self.raw_curvature, CURVATURE_MIN, CURVATURE_MAX)
        return weights, waist, curvature


def synthesize_weighted_hg(
    evaluator: UnifiedWOS, model: WeightedHGParameters
) -> tuple[torch.Tensor, torch.Tensor]:
    weights, waist, curvature = model.physical()
    repeated_waists = waist.expand(MODE_COUNT)
    one_dimensional = hermite_gaussian_1d(evaluator, repeated_waists, 4).to(
        evaluator.complex_dtype
    )
    fields = []
    for mode_index, (x_order, y_order) in enumerate(MODE_INDICES):
        field = (
            one_dimensional[mode_index, x_order, :, None]
            * one_dimensional[mode_index, y_order, None, :]
        )
        fields.append(field)
    source = torch.stack(fields, dim=0)
    source = source * evaluator.curvature_phase(curvature)
    source = evaluator.normalize_each_frame(source)
    return source, weights


def weighted_objective(
    evaluator: UnifiedWOS,
    source: torch.Tensor,
    weights: torch.Tensor,
    batch_size: int,
    screen_generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    modal_flux = evaluator.modal_flux_batch(source, batch_size, screen_generator)
    integrated_flux = modal_flux @ weights.to(dtype=modal_flux.dtype)
    mean_flux = integrated_flux.mean()
    scintillation_index = integrated_flux.square().mean() / mean_flux.square() - 1.0
    reciprocal_snr = torch.sqrt(
        evaluator.config.duale_b / mean_flux + scintillation_index.clamp_min(0.0)
    )
    snr_db = 10.0 * torch.log10(1.0 / reciprocal_snr)
    return reciprocal_snr, snr_db, mean_flux, scintillation_index


def weighted_metrics(
    modal_flux: torch.Tensor,
    weights: torch.Tensor,
    duale_b: float,
    bootstrap_seed: int,
    bootstrap_repeats: int,
) -> dict[str, float]:
    modal = modal_flux.detach().cpu().double()
    dwell = weights.detach().cpu().double()
    integrated = modal @ dwell
    base = metric_from_integrated_power(integrated.numpy(), duale_b)
    centered = modal - modal.mean(dim=0, keepdim=True)
    covariance = centered.T @ centered / max(1, modal.shape[0] - 1)
    standard_deviation = torch.sqrt(torch.diag(covariance).clamp_min(1e-30))
    weighted_variance = dwell @ covariance @ dwell
    effective_frames = (
        dwell @ standard_deviation
    ).square() / weighted_variance.clamp_min(1e-30)
    random_generator = np.random.default_rng(bootstrap_seed)
    sample_indices = random_generator.integers(
        0, integrated.numel(), size=(bootstrap_repeats, integrated.numel())
    )
    bootstrap_power = integrated.numpy()[sample_indices]
    bootstrap_mean = bootstrap_power.mean(axis=1)
    bootstrap_si = (
        np.square(bootstrap_power).mean(axis=1) / np.square(bootstrap_mean) - 1.0
    )
    bootstrap_snr = 10.0 * np.log10(
        1.0 / np.sqrt(duale_b / bootstrap_mean + np.maximum(bootstrap_si, 0.0))
    )
    ci_low, ci_high = np.quantile(bootstrap_snr, (0.025, 0.975))
    base.update(
        {
            "effective_frame_number": float(effective_frames),
            "snr_db_ci95_low": float(ci_low),
            "snr_db_ci95_high": float(ci_high),
            "frame_count": int(modal.shape[1]),
            "atmospheric_samples": int(modal.shape[0]),
        }
    )
    return base


def train_candidate(
    evaluator: UnifiedWOS,
    sobol_point: torch.Tensor,
    run_index: int,
    case_index: int,
    arguments: argparse.Namespace,
) -> dict:
    model = WeightedHGParameters(sobol_point, evaluator.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=arguments.learning_rate)
    screen_generator = evaluator.generator(
        arguments.seed + 100000 * case_index + 1000 * run_index + 17
    )
    history: list[dict] = []
    started = time.time()
    for step in range(1, arguments.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        accumulated_loss = 0.0
        latest = None
        for _ in range(arguments.mc_blocks):
            source, weights = synthesize_weighted_hg(evaluator, model)
            loss, snr_db, mean_flux, scintillation_index = weighted_objective(
                evaluator, source, weights, arguments.train_bs, screen_generator
            )
            (loss / arguments.mc_blocks).backward()
            accumulated_loss += float(loss.detach()) / arguments.mc_blocks
            latest = (snr_db, mean_flux, scintillation_index)
        torch.nn.utils.clip_grad_norm_(model.parameters(), arguments.grad_clip)
        optimizer.step()
        if step == 1 or step % arguments.log_every == 0 or step == arguments.steps:
            weights, waist, curvature = model.physical()
            history.append(
                {
                    "step": step,
                    "loss": accumulated_loss,
                    "train_snr_db": float(latest[0]),
                    "train_mean_power": float(latest[1]),
                    "train_si": float(latest[2]),
                    "waist_m": float(waist),
                    "normalized_curvature": float(curvature),
                    "weights": [float(value) for value in weights],
                }
            )
    return {
        "run": run_index,
        "elapsed_seconds": time.time() - started,
        "history": history,
        "model_state": model.state_dict(),
    }


@torch.inference_mode()
def evaluate_candidate(
    evaluator: UnifiedWOS,
    candidate: dict,
    sobol_point: torch.Tensor,
    atmospheric_samples: int,
    batch_size: int,
    screen_seed: int,
    bootstrap_seed: int,
    bootstrap_repeats: int,
) -> tuple[dict, torch.Tensor, torch.Tensor]:
    model = WeightedHGParameters(sobol_point, evaluator.device)
    model.load_state_dict(candidate["model_state"])
    source, weights = synthesize_weighted_hg(evaluator, model)
    blocks = int(math.ceil(atmospheric_samples / batch_size))
    modal_flux = evaluator.collect_modal_flux(source, blocks, batch_size, screen_seed)[
        :atmospheric_samples
    ]
    metrics = weighted_metrics(
        modal_flux, weights, evaluator.config.duale_b, bootstrap_seed, bootstrap_repeats
    )
    physical_weights, waist, curvature = model.physical()
    metrics.update(
        {
            "weights": [float(value) for value in physical_weights],
            "waist_m": float(waist),
            "normalized_curvature": float(curvature),
            "mode_indices": [list(index) for index in MODE_INDICES],
        }
    )
    return metrics, modal_flux, weights.detach().cpu()


def serializable(candidate: dict) -> dict:
    return {key: value for key, value in candidate.items() if key != "model_state"}


def run_case(case_name: str, case_index: int, arguments: argparse.Namespace) -> dict:
    output_directory = OUTPUT_ROOT / case_name
    result_path = output_directory / "weighted_hg_result.pt"
    summary_path = output_directory / "weighted_hg_summary.json"
    if result_path.exists() and summary_path.exists() and not arguments.force:
        return read_json(summary_path)
    reference = read_json(REFERENCE_ROOT / case_name / "summary.json")
    evaluator = UnifiedWOS(LinkConfig(**reference["link"]))
    sobol_engine = torch.quasirandom.SobolEngine(
        MODE_COUNT + 2, scramble=True, seed=arguments.seed + case_index
    )
    sobol_points = sobol_engine.draw(arguments.runs)
    candidates: list[dict] = []
    case_started = time.time()
    for run_index in range(arguments.runs):
        print(
            f"{case_name}: weighted-HG run {run_index + 1}/{arguments.runs}", flush=True
        )
        candidate = train_candidate(
            evaluator, sobol_points[run_index], run_index, case_index, arguments
        )
        selection_metrics, _, _ = evaluate_candidate(
            evaluator,
            candidate,
            sobol_points[run_index],
            arguments.selection_samples,
            arguments.batch_size,
            arguments.selection_seed + 100000 * case_index,
            arguments.seed + 9000 + run_index,
            300,
        )
        candidate["selection_metrics"] = selection_metrics
        candidates.append(candidate)
        checkpoint = {
            "case": case_name,
            "completed_runs": len(candidates),
            "sobol_points": sobol_points,
            "candidates": candidates,
        }
        output_directory.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, output_directory / "training_checkpoint.pt")
        torch.cuda.empty_cache()
    selected_index = int(
        np.argmax(
            [item["selection_metrics"]["snrp_legacy_db_10log"] for item in candidates]
        )
    )
    selected = candidates[selected_index]
    gsm_metadata = read_json(GSM_ROOT / case_name / "resampled_gsm_ensemble.json")
    holdout_seed = int(gsm_metadata["screen_seed_common"])
    holdout_metrics, modal_flux, weights = evaluate_candidate(
        evaluator,
        selected,
        sobol_points[selected_index],
        arguments.holdout_samples,
        arguments.batch_size,
        holdout_seed,
        arguments.seed + 200000 + case_index,
        arguments.bootstrap_repeats,
    )
    summary = {
        "case": case_name,
        "method": "directly weighted pure HG pseudo-modes",
        "adaptation_of": "Schulz--Liu modal-intensity optimization",
        "dimensional_adaptation": "first ten 2D HG modes through complete total order 3",
        "link": reference["link"],
        "frame_count": MODE_COUNT,
        "optimization": {
            "runs": arguments.runs,
            "steps": arguments.steps,
            "train_bs": arguments.train_bs,
            "mc_blocks": arguments.mc_blocks,
            "learning_rate": arguments.learning_rate,
            "optimized_parameters": [
                "ten nonnegative HG modal intensities summing to one",
                "common HG waist",
                "common normalized curvature",
            ],
        },
        "selection": {
            "samples": arguments.selection_samples,
            "selected_run": selected_index,
            "selected_metrics": selected["selection_metrics"],
        },
        "holdout": holdout_metrics,
        "all_candidates": [serializable(candidate) for candidate in candidates],
        "elapsed_seconds": time.time() - case_started,
    }
    torch.save(
        {
            "summary": summary,
            "selected_model_state": selected["model_state"],
            "selected_sobol_point": sobol_points[selected_index],
            "modal_flux": modal_flux,
            "weights": weights,
        },
        result_path,
    )
    write_json(summary_path, summary)
    print(
        json.dumps(
            {
                "case": case_name,
                "selected_run": selected_index,
                "snrp_db": holdout_metrics["snrp_legacy_db_10log"],
                "mean_power": holdout_metrics["mean_flux"],
                "si": holdout_metrics["scintillation_index"],
                "meff": holdout_metrics["effective_frame_number"],
            },
            indent=2,
        ),
        flush=True,
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases", nargs="+", choices=CASE_NAMES, default=list(CASE_NAMES)
    )
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
    parser.add_argument("--seed", type=int, default=2026090101)
    parser.add_argument("--selection-seed", type=int, default=2026090201)
    parser.add_argument("--force", action="store_true")
    arguments = parser.parse_args()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    results = [
        run_case(case_name, CASE_NAMES.index(case_name), arguments)
        for case_name in arguments.cases
    ]
    write_json(OUTPUT_ROOT / "weighted_hg_all.json", results)


if __name__ == "__main__":
    main()
