"Constrained GSM parameters and separable Hermite-Gaussian pseudo-mode synthesis."

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as functional

from propagation import LinkConfig, UnifiedWOS, curvature_to_rg, metrics_from_modal_flux


WG_MIN = 3e-3
WG_MAX_PCB = 100e-3
WG_MAX_SMS = 80e-3
LCR_MIN = 3e-3
LCR_MAX = 50e-3
CURVATURE_MIN = 0.0
CURVATURE_MAX = 2.0


def bounded(raw: torch.Tensor, lower: float, upper: float) -> torch.Tensor:
    return lower + (upper - lower) * torch.sigmoid(raw)


def inverse_bounded(value: float, lower: float, upper: float) -> float:
    fraction = (value - lower) / (upper - lower)
    fraction = min(max(fraction, 1e-5), 1.0 - 1e-5)
    return math.log(fraction / (1.0 - fraction))


def inverse_softplus(value: torch.Tensor) -> torch.Tensor:
    return torch.log(torch.expm1(value.clamp_min(1e-6)))


class PCBParameters(nn.Module):
    """GSM width, coherence length, and normalized curvature in bounded coordinates."""

    def __init__(
        self,
        initial_wg: float,
        initial_lcr: float,
        initial_curvature: float,
        wg_max: float,
        lcr_max: float,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.wg_max = wg_max
        self.lcr_max = lcr_max
        initial = torch.tensor(
            [
                inverse_bounded(initial_wg, WG_MIN, self.wg_max),
                inverse_bounded(initial_lcr, LCR_MIN, self.lcr_max),
                inverse_bounded(initial_curvature, CURVATURE_MIN, CURVATURE_MAX),
            ],
            device=device,
            dtype=torch.float32,
        )
        self.raw = nn.Parameter(initial)

    def physical(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        wg = bounded(self.raw[0], WG_MIN, self.wg_max)
        lcr = bounded(self.raw[1], LCR_MIN, self.lcr_max)
        curvature = bounded(self.raw[2], CURVATURE_MIN, CURVATURE_MAX)
        return wg, lcr, curvature


class SMSParameters(nn.Module):
    """Frame-specific HG amplitudes, relative phases, and transverse scales."""

    def __init__(
        self,
        frame_count: int,
        n_side: int,
        seed: int,
        wg_max: float,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.frame_count = frame_count
        self.n_side = n_side
        self.mode_count = n_side**2
        self.wg_max = wg_max
        cpu_generator = torch.Generator(device="cpu").manual_seed(seed)
        amplitudes = 0.015 + 0.015 * torch.rand(
            frame_count, self.mode_count, generator=cpu_generator
        )
        preferred_modes = [
            i * n_side + j for i in range(0, n_side, 2) for j in range(0, n_side, 2)
        ]
        remaining_modes = [
            index for index in range(self.mode_count) if index not in preferred_modes
        ]
        ordered_modes = preferred_modes + remaining_modes
        for frame in range(frame_count):
            amplitudes[frame, ordered_modes[frame % len(ordered_modes)]] = 0.8
        phase_fraction = 0.8 * (
            2.0 * torch.rand(frame_count, self.mode_count - 1, generator=cpu_generator)
            - 1.0
        )
        self.raw_amplitudes = nn.Parameter(inverse_softplus(amplitudes).to(device))
        self.raw_phases = nn.Parameter(torch.atanh(phase_fraction).to(device))
        initial_waist_value = min(35e-3, 0.7 * self.wg_max)
        initial_waist = torch.full(
            (frame_count,),
            inverse_bounded(initial_waist_value, WG_MIN, self.wg_max),
            device=device,
        )
        self.raw_waists = nn.Parameter(initial_waist)

    def physical(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        amplitudes = functional.softplus(self.raw_amplitudes)
        relative_phases = math.pi * torch.tanh(self.raw_phases)
        reference_phase = torch.zeros(
            self.frame_count, 1, device=self.raw_phases.device
        )
        phases = torch.cat((reference_phase, relative_phases), dim=1)
        waists = bounded(self.raw_waists, WG_MIN, self.wg_max)
        return amplitudes, phases, waists


def hermite_gaussian_1d(
    evaluator: UnifiedWOS, waists: torch.Tensor, n_side: int
) -> torch.Tensor:
    x_coordinate = evaluator.xx[:, 0].reshape(1, -1)
    waist_column = waists.reshape(-1, 1)
    normalized_coordinate = math.sqrt(2.0) * x_coordinate / waist_column
    fundamental = math.pi ** (-0.25) * torch.exp(-0.5 * normalized_coordinate.square())
    modes = [fundamental]
    if n_side > 1:
        modes.append(math.sqrt(2.0) * normalized_coordinate * fundamental)
    for order in range(1, n_side - 1):
        next_mode = (
            math.sqrt(2.0 / (order + 1.0)) * normalized_coordinate * modes[-1]
            - math.sqrt(order / (order + 1.0)) * modes[-2]
        )
        modes.append(next_mode)
    dimensionless = torch.stack(modes, dim=1)
    dimensional_scale = torch.sqrt(math.sqrt(2.0) / waist_column).unsqueeze(1)
    return dimensionless * dimensional_scale


def synthesize_sms(
    evaluator: UnifiedWOS, parameters: SMSParameters, normalized_curvature: float = 1.0
) -> torch.Tensor:
    amplitudes, phases, waists = parameters.physical()
    one_dimensional = hermite_gaussian_1d(evaluator, waists, parameters.n_side).to(
        torch.complex64
    )
    # Contract separable 1D functions instead of allocating every 2D basis image.
    complex_weights = torch.polar(amplitudes, phases).reshape(
        parameters.frame_count, parameters.n_side, parameters.n_side
    )
    source = torch.einsum(
        "mij,mix,mjy->mxy", complex_weights, one_dimensional, one_dimensional
    )
    source = source * evaluator.curvature_phase(normalized_curvature)
    return evaluator.normalize_each_frame(source)


def pcb_initial_point(run_index: int, runs: int) -> tuple[float, float, float]:
    candidates = [
        (75e-3, 6e-3, 0.8),
        (15e-3, 35e-3, 1.2),
        (35e-3, 12e-3, 0.3),
        (90e-3, 30e-3, 1.7),
        (8e-3, 5e-3, 1.0),
    ]
    return candidates[run_index % min(len(candidates), max(1, runs))]


def train_pcb(
    evaluator: UnifiedWOS, frame_count: int, run_index: int, args: argparse.Namespace
) -> dict:
    initial_wg, initial_lcr, initial_curvature = pcb_initial_point(run_index, args.runs)
    initial_wg = min(initial_wg, 0.9 * args.pcb_wg_max)
    initial_lcr = min(initial_lcr, 0.9 * args.lcr_max)
    model = PCBParameters(
        initial_wg,
        initial_lcr,
        initial_curvature,
        args.pcb_wg_max,
        args.lcr_max,
        evaluator.device,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.pcb_lr)
    source_generator = evaluator.generator(args.seed + 10000 * run_index + 101)
    screen_generator = evaluator.generator(args.seed + 10000 * run_index + 202)
    history = []
    started = time.time()
    for step in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        accumulated_loss = 0.0
        latest_metrics = None
        for _ in range(args.mc_blocks):
            wg, lcr, curvature = model.physical()
            source = evaluator.pcb_source(
                wg, lcr, curvature, frame_count, source_generator
            )
            loss, snr_db, mean_flux, scintillation_index = evaluator.loss_from_source(
                source, args.train_bs, screen_generator
            )
            (loss / args.mc_blocks).backward()
            accumulated_loss += float(loss.detach()) / args.mc_blocks
            latest_metrics = (snr_db, mean_flux, scintillation_index)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            wg, lcr, curvature = model.physical()
            history.append(
                {
                    "step": step,
                    "loss": accumulated_loss,
                    "train_snr_db": float(latest_metrics[0]),
                    "train_mean_flux": float(latest_metrics[1]),
                    "train_si": float(latest_metrics[2]),
                    "wg_m": float(wg),
                    "lcr_m": float(lcr),
                    "normalized_curvature": float(curvature),
                    "rg_m": curvature_to_rg(
                        evaluator.config.distance, float(curvature)
                    ),
                }
            )
    wg, lcr, curvature = model.physical()
    return {
        "kind": "PCB",
        "run": run_index,
        "elapsed_seconds": time.time() - started,
        "wg_m": float(wg),
        "lcr_m": float(lcr),
        "normalized_curvature": float(curvature),
        "rg_m": curvature_to_rg(evaluator.config.distance, float(curvature)),
        "model_state": model.state_dict(),
        "history": history,
    }


def train_sms(
    evaluator: UnifiedWOS, frame_count: int, run_index: int, args: argparse.Namespace
) -> dict:
    model = SMSParameters(
        frame_count,
        args.n_side,
        args.seed + 10000 * run_index + 303,
        args.sms_wg_max,
        evaluator.device,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.sms_lr)
    screen_generator = evaluator.generator(args.seed + 10000 * run_index + 404)
    history = []
    started = time.time()
    for step in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        accumulated_loss = 0.0
        latest_metrics = None
        for _ in range(args.mc_blocks):
            source = synthesize_sms(evaluator, model, normalized_curvature=1.0)
            loss, snr_db, mean_flux, scintillation_index = evaluator.loss_from_source(
                source, args.train_bs, screen_generator
            )
            (loss / args.mc_blocks).backward()
            accumulated_loss += float(loss.detach()) / args.mc_blocks
            latest_metrics = (snr_db, mean_flux, scintillation_index)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            _, _, waists = model.physical()
            history.append(
                {
                    "step": step,
                    "loss": accumulated_loss,
                    "train_snr_db": float(latest_metrics[0]),
                    "train_mean_flux": float(latest_metrics[1]),
                    "train_si": float(latest_metrics[2]),
                    "mean_wg_m": float(waists.mean()),
                    "min_wg_m": float(waists.min()),
                    "max_wg_m": float(waists.max()),
                }
            )
    return {
        "kind": "SMS",
        "run": run_index,
        "elapsed_seconds": time.time() - started,
        "normalized_curvature": 1.0,
        "rg_m": evaluator.config.distance,
        "model_state": model.state_dict(),
        "history": history,
        "n_side": args.n_side,
    }


@torch.inference_mode()
def source_for_result(
    evaluator: UnifiedWOS,
    result: dict,
    frame_count: int,
    args: argparse.Namespace,
    source_seed: int,
) -> torch.Tensor:
    if result["kind"] == "PCB":
        wg = torch.tensor(result["wg_m"], device=evaluator.device)
        lcr = torch.tensor(result["lcr_m"], device=evaluator.device)
        curvature = torch.tensor(
            result["normalized_curvature"], device=evaluator.device
        )
        return evaluator.pcb_source(
            wg, lcr, curvature, frame_count, evaluator.generator(source_seed)
        )
    model = SMSParameters(
        frame_count, result["n_side"], args.seed, args.sms_wg_max, evaluator.device
    )
    model.load_state_dict(result["model_state"])
    return synthesize_sms(evaluator, model, normalized_curvature=1.0)


@torch.inference_mode()
def validate_result(
    evaluator: UnifiedWOS,
    result: dict,
    frame_count: int,
    args: argparse.Namespace,
    selection: bool,
) -> dict:
    repeats = args.selection_repeats if selection else args.holdout_repeats
    blocks = args.selection_blocks if selection else args.holdout_blocks
    base_seed = args.selection_seed if selection else args.holdout_seed
    modal_chunks = []
    per_repeat_metrics = []
    for repeat in range(repeats):
        source_seed = base_seed + 1000 * repeat + 11
        screen_seed = base_seed + 1000 * repeat + 22
        source = source_for_result(evaluator, result, frame_count, args, source_seed)
        modal_flux = evaluator.collect_modal_flux(
            source, blocks, args.eval_bs, screen_seed
        )
        modal_chunks.append(modal_flux)
        per_repeat_metrics.append(
            metrics_from_modal_flux(
                modal_flux,
                evaluator.config.duale_b,
                bootstrap_seed=base_seed + repeat,
                bootstrap_repeats=200,
            )
        )
    pooled_flux = torch.cat(modal_chunks, dim=0)
    pooled_metrics = metrics_from_modal_flux(
        pooled_flux,
        evaluator.config.duale_b,
        bootstrap_seed=base_seed + 77,
        bootstrap_repeats=args.bootstrap_repeats,
    )
    pooled_metrics["source_screen_repeats"] = repeats
    pooled_metrics["per_repeat_snr_db_std"] = (
        float(np.std([item["snr_db"] for item in per_repeat_metrics], ddof=1))
        if repeats > 1
        else 0.0
    )
    pooled_metrics["per_repeat"] = per_repeat_metrics
    return pooled_metrics


def serializable_result(result: dict) -> dict:
    return {key: value for key, value in result.items() if key != "model_state"}


def run_experiment(args: argparse.Namespace) -> dict:
    config = LinkConfig(
        name=args.name,
        n=512,
        delta=1e-3,
        wavelength=args.wavelength,
        cn2=args.cn2,
        distance=args.distance,
        wos_steps=args.wos_steps,
        transmitter_radius=args.transmitter_radius,
        receiver_radius=args.receiver_radius,
        inner_scale=args.inner_scale,
        outer_scale=args.outer_scale,
        duale_b=args.duale_b,
        power=args.power,
        subharmonic_order=3,
    )
    evaluator = UnifiedWOS(config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_results = {"PCB": [], "SMS": []}
    started = time.time()
    for kind, trainer in (("PCB", train_pcb), ("SMS", train_sms)):
        for run_index in range(args.runs):
            print(
                f"{args.name} M={args.frames} {kind} run {run_index + 1}/{args.runs}",
                flush=True,
            )
            trained = trainer(evaluator, args.frames, run_index, args)
            trained["selection_metrics"] = validate_result(
                evaluator, trained, args.frames, args, selection=True
            )
            all_results[kind].append(trained)
            torch.cuda.empty_cache()
    selected = {}
    for kind in ("PCB", "SMS"):
        best = max(
            all_results[kind], key=lambda item: item["selection_metrics"]["snr_db"]
        )
        best["holdout_metrics"] = validate_result(
            evaluator, best, args.frames, args, selection=False
        )
        selected[kind] = best
    pcb_snr = selected["PCB"]["holdout_metrics"]["snr_db"]
    sms_snr = selected["SMS"]["holdout_metrics"]["snr_db"]
    summary = {
        "schema_version": 2,
        "link": config.__dict__,
        "frame_count": args.frames,
        "optimization": {
            "steps": args.steps,
            "runs": args.runs,
            "mc_blocks": args.mc_blocks,
            "train_bs": args.train_bs,
            "pcb_lr": args.pcb_lr,
            "sms_lr": args.sms_lr,
            "sms_n_side": args.n_side,
            "pcb_wg_max": args.pcb_wg_max,
            "sms_wg_max": args.sms_wg_max,
            "lcr_max": args.lcr_max,
            "sms_rg_rule": "Rg=distance",
            "sms_hg_implementation": "separable_1d_direct_synthesis",
            "pcb_parameters": ["Wg", "Lcr", "Rg"],
        },
        "validation": {
            "selection_repeats": args.selection_repeats,
            "selection_blocks": args.selection_blocks,
            "holdout_repeats": args.holdout_repeats,
            "holdout_blocks": args.holdout_blocks,
            "eval_bs": args.eval_bs,
        },
        "selected": {
            kind: serializable_result(result) for kind, result in selected.items()
        },
        "all_candidates": {
            kind: [serializable_result(item) for item in items]
            for kind, items in all_results.items()
        },
        "sms_minus_pcb_snr_db": sms_snr - pcb_snr,
        "elapsed_seconds": time.time() - started,
    }
    torch.save(
        {
            "summary": summary,
            "selected_model_states": {
                kind: selected[kind]["model_state"] for kind in ("PCB", "SMS")
            },
        },
        args.output_dir / "result.pt",
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=True), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "name": args.name,
                "M": args.frames,
                "pcb_snr_db": pcb_snr,
                "sms_snr_db": sms_snr,
                "gain_db": sms_snr - pcb_snr,
            },
            indent=2,
        ),
        flush=True,
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="reference")
    parser.add_argument("--frames", type=int, default=10)
    parser.add_argument("--wavelength", type=float, default=0.85e-6)
    parser.add_argument("--cn2", type=float, default=1e-14)
    parser.add_argument("--distance", type=float, default=1000.0)
    parser.add_argument("--transmitter-radius", type=float, default=0.05)
    parser.add_argument("--receiver-radius", type=float, default=0.025)
    parser.add_argument("--inner-scale", type=float, default=0.003)
    parser.add_argument("--outer-scale", type=float, default=1.024)
    parser.add_argument("--duale-b", type=float, default=0.2)
    parser.add_argument("--power", type=float, default=1000.0)
    parser.add_argument("--wos-steps", type=int, default=5)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--mc-blocks", type=int, default=2)
    parser.add_argument("--train-bs", type=int, default=6)
    parser.add_argument("--pcb-lr", type=float, default=0.04)
    parser.add_argument("--sms-lr", type=float, default=0.02)
    parser.add_argument("--grad-clip", type=float, default=10.0)
    parser.add_argument("--n-side", type=int, default=8)
    parser.add_argument("--pcb-wg-max", type=float, default=WG_MAX_PCB)
    parser.add_argument("--sms-wg-max", type=float, default=WG_MAX_SMS)
    parser.add_argument("--lcr-max", type=float, default=LCR_MAX)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--selection-repeats", type=int, default=2)
    parser.add_argument("--selection-blocks", type=int, default=8)
    parser.add_argument("--holdout-repeats", type=int, default=5)
    parser.add_argument("--holdout-blocks", type=int, default=25)
    parser.add_argument("--eval-bs", type=int, default=8)
    parser.add_argument("--bootstrap-repeats", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2026081701)
    parser.add_argument("--selection-seed", type=int, default=2026081801)
    parser.add_argument("--holdout-seed", type=int, default=2026081901)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            Path(__file__).resolve().parent / "logs/ol_sms_pcb_3param/reference_M10"
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    run_experiment(parse_args())
