"Scrambled Sobol multistart optimization with independent source-atmosphere validation."

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.quasirandom import SobolEngine

from sources import (
    CURVATURE_MAX,
    CURVATURE_MIN,
    LCR_MIN,
    PCBParameters,
    SMSParameters,
    WG_MIN,
    inverse_bounded,
    inverse_softplus,
    synthesize_sms,
)
from propagation import LinkConfig, UnifiedWOS, curvature_to_rg, metrics_from_modal_flux

ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "logs" / "multistart"
PCB_WG_MAX = 150e-3
LCR_MAX = 150e-3
OPMS_WG_MAX = 80e-3
N_SIDE = 8
GRAD_CLIP = 10.0
BASE_SEED = 2026091101


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--links",
        nargs="+",
        default=["A", "B", "C", "D"],
        choices=["A", "B", "C", "D"],
        help="Link labels, in paper order.",
    )
    parser.add_argument(
        "--starts",
        type=int,
        default=1024,
        help="Sobol starts per link and source class.",
    )
    parser.add_argument(
        "--steps", type=int, default=300, help="Adam updates per trajectory."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Atmospheres in each joint training loss.",
    )
    parser.add_argument(
        "--frames", type=int, default=10, help="Pseudo-modes in each detector window."
    )
    parser.add_argument(
        "--screening-samples",
        type=int,
        default=1000,
        help="Common unseen atmospheres for every endpoint.",
    )
    parser.add_argument(
        "--validation-samples",
        type=int,
        default=1000,
        help="Additional unseen atmospheres for the selected winner.",
    )
    parser.add_argument(
        "--validation-batch",
        type=int,
        default=16,
        help="Propagation batch size during validation.",
    )
    parser.add_argument("--pcb-lr", type=float, default=0.04, help="GSM learning rate.")
    parser.add_argument(
        "--opms-lr", type=float, default=0.02, help="OPMS learning rate."
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=25,
        help="Updates between resumable checkpoints.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=5,
        help="Updates between recorded trajectory samples.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "work/multistart",
        help="Directory for a new experiment.",
    )
    parser.add_argument(
        "--device", default="cuda", help="PyTorch device, for example cuda or cpu."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2026092401,
        help="Base seed of the archived 1024-start experiment.",
    )
    return parser.parse_args()


def link_table() -> dict[str, LinkConfig]:
    common = dict(
        n=512,
        delta=1e-3,
        wos_steps=5,
        transmitter_radius=50e-3,
        inner_scale=3e-3,
        outer_scale=1.024,
        power=1000.0,
        subharmonic_order=3,
    )
    return {
        "A": LinkConfig(
            name="A",
            cn2=5e-15,
            distance=1000.0,
            receiver_radius=10e-3,
            wavelength=0.632e-6,
            duale_b=0.1,
            **common,
        ),
        "B": LinkConfig(
            name="B",
            cn2=5e-14,
            distance=1000.0,
            receiver_radius=10e-3,
            wavelength=0.632e-6,
            duale_b=0.1,
            **common,
        ),
        "C": LinkConfig(
            name="C",
            cn2=1e-14,
            distance=1000.0,
            receiver_radius=10e-3,
            wavelength=0.850e-6,
            duale_b=0.2,
            **common,
        ),
        "D": LinkConfig(
            name="D",
            cn2=5e-15,
            distance=2000.0,
            receiver_radius=20e-3,
            wavelength=0.850e-6,
            duale_b=0.2,
            **common,
        ),
    }


def write_json(path: Path, value: object) -> None:
    temporary = path.with_name(f"{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=True),
        encoding="utf-8",
    )
    for attempt in range(12):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 11:
                raise
            time.sleep(1.0)


def cpu_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone() for name, value in model.state_dict().items()
    }


def pcb_sobol_points(starts: int, seed: int) -> torch.Tensor:
    unit = (
        SobolEngine(dimension=3, scramble=True, seed=seed)
        .draw(starts)
        .clamp(1e-5, 1.0 - 1e-5)
    )
    wg = torch.exp(math.log(WG_MIN) + unit[:, 0] * math.log(PCB_WG_MAX / WG_MIN))
    lcr = torch.exp(math.log(LCR_MIN) + unit[:, 1] * math.log(LCR_MAX / LCR_MIN))
    curvature = CURVATURE_MIN + unit[:, 2] * (CURVATURE_MAX - CURVATURE_MIN)
    return torch.stack((wg, lcr, curvature), dim=1)


def opms_sobol_points(starts: int, frames: int, seed: int) -> torch.Tensor:
    mode_count = N_SIDE**2
    dimension = frames * mode_count + frames * (mode_count - 1) + frames
    return (
        SobolEngine(dimension=dimension, scramble=True, seed=seed)
        .draw(starts)
        .clamp(1e-5, 1.0 - 1e-5)
    )


def initialize_opms(model: SMSParameters, unit: torch.Tensor, run_index: int) -> None:
    frames = model.frame_count
    mode_count = model.mode_count
    amplitude_count = frames * mode_count
    phase_count = frames * (mode_count - 1)
    amplitudes = (0.01 + 0.04 * unit[:amplitude_count]).reshape(frames, mode_count)
    preferred = [
        i * N_SIDE + j for i in range(0, N_SIDE, 2) for j in range(0, N_SIDE, 2)
    ]
    remaining = [index for index in range(mode_count) if index not in preferred]
    ordered = preferred + remaining
    for frame in range(frames):
        dominant = ordered[(frame + run_index) % len(ordered)]
        amplitudes[frame, dominant] = 0.50 + 0.50 * unit[frame * mode_count + dominant]
    phase_start = amplitude_count
    phase_fraction = 0.90 * (2.0 * unit[phase_start : phase_start + phase_count] - 1.0)
    waist_unit = unit[phase_start + phase_count :]
    waists = torch.exp(math.log(8e-3) + waist_unit * math.log(55e-3 / 8e-3))
    waist_raw = torch.tensor(
        [inverse_bounded(float(value), WG_MIN, OPMS_WG_MAX) for value in waists],
        dtype=torch.float32,
    )
    with torch.no_grad():
        model.raw_amplitudes.copy_(
            inverse_softplus(amplitudes).to(model.raw_amplitudes.device)
        )
        model.raw_phases.copy_(
            torch.atanh(phase_fraction)
            .reshape(frames, mode_count - 1)
            .to(model.raw_phases.device)
        )
        model.raw_waists.copy_(waist_raw.to(model.raw_waists.device))


def build_model(
    kind: str,
    run_index: int,
    evaluator: UnifiedWOS,
    pcb_points: torch.Tensor,
    opms_points: torch.Tensor,
    args: argparse.Namespace,
    link_seed: int,
) -> torch.nn.Module:
    if kind == "GSM":
        initial = pcb_points[run_index]
        return PCBParameters(
            float(initial[0]),
            float(initial[1]),
            float(initial[2]),
            PCB_WG_MAX,
            LCR_MAX,
            evaluator.device,
        )
    model = SMSParameters(
        args.frames, N_SIDE, link_seed + 5000 + run_index, OPMS_WG_MAX, evaluator.device
    )
    initialize_opms(model, opms_points[run_index], run_index)
    return model


def sources_for_batch(
    kind: str,
    model: torch.nn.Module,
    evaluator: UnifiedWOS,
    batch_size: int,
    frames: int,
    source_generator: torch.Generator,
) -> torch.Tensor:
    if kind == "GSM":
        wg, lcr, curvature = model.physical()
        sources = [
            evaluator.pcb_source(wg, lcr, curvature, frames, source_generator)
            for _ in range(batch_size)
        ]
        return torch.stack(sources, dim=0)
    source = synthesize_sms(evaluator, model, normalized_curvature=1.0)
    return source.unsqueeze(0).expand(batch_size, -1, -1, -1)


def modal_flux_for_pairs(
    evaluator: UnifiedWOS, sources: torch.Tensor, screen_generator: torch.Generator
) -> torch.Tensor:
    batch_size = sources.shape[0]
    phase_screens = evaluator.screens(batch_size, screen_generator)
    field = sources
    for step in range(evaluator.config.wos_steps):
        current_phase = phase_screens[:, None, step]
        field = field * torch.exp(
            torch.complex(torch.zeros_like(current_phase), current_phase)
        )
        field = field * evaluator.grid_truncation
        field = evaluator.propagate(field)
    intensity = (field.abs() * evaluator.receiver_aperture).square()
    return intensity.sum(dim=(-2, -1)).to(torch.float64)


def loss_from_independent_pairs(
    kind: str,
    model: torch.nn.Module,
    evaluator: UnifiedWOS,
    batch_size: int,
    frames: int,
    source_generator: torch.Generator,
    screen_generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    sources = sources_for_batch(
        kind, model, evaluator, batch_size, frames, source_generator
    )
    modal_flux = modal_flux_for_pairs(evaluator, sources, screen_generator)
    integrated_flux = modal_flux.mean(dim=1)
    mean_flux = integrated_flux.mean()
    scintillation_index = integrated_flux.square().mean() / mean_flux.square() - 1.0
    loss = torch.sqrt(
        evaluator.config.duale_b / mean_flux + scintillation_index.clamp_min(0.0)
    )
    snr_db = 10.0 * torch.log10(1.0 / loss)
    return loss, snr_db, mean_flux, scintillation_index


def checkpoint_path(output: Path, link: str, kind: str, run_index: int) -> Path:
    return output / link / kind.lower() / f"run_{run_index:03d}.pt"


def running_path(output: Path, link: str, kind: str, run_index: int) -> Path:
    return output / link / kind.lower() / f"run_{run_index:03d}.running.pt"


def train_candidate(
    kind: str,
    run_index: int,
    evaluator: UnifiedWOS,
    pcb_points: torch.Tensor,
    opms_points: torch.Tensor,
    args: argparse.Namespace,
    link_seed: int,
) -> dict:
    final_path = checkpoint_path(args.output, evaluator.config.name, kind, run_index)
    partial_path = running_path(args.output, evaluator.config.name, kind, run_index)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    if final_path.exists():
        return torch.load(final_path, map_location="cpu", weights_only=False)
    torch.manual_seed(link_seed + 1000 * run_index + (11 if kind == "GSM" else 22))
    torch.cuda.manual_seed_all(
        link_seed + 1000 * run_index + (11 if kind == "GSM" else 22)
    )
    model = build_model(
        kind, run_index, evaluator, pcb_points, opms_points, args, link_seed
    )
    learning_rate = args.pcb_lr if kind == "GSM" else args.opms_lr
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    source_generator = evaluator.generator(
        link_seed + 100000 * run_index + (101 if kind == "GSM" else 111)
    )
    screen_generator = evaluator.generator(link_seed + 100000 * run_index + 202)
    history: list[dict] = []
    start_step = 1
    elapsed_before = 0.0
    if partial_path.exists():
        partial = torch.load(
            partial_path, map_location=evaluator.device, weights_only=False
        )
        model.load_state_dict(partial["model_state"])
        optimizer.load_state_dict(partial["optimizer_state"])
        source_generator.set_state(partial["source_generator_state"].cpu())
        screen_generator.set_state(partial["screen_generator_state"].cpu())
        history = partial["history"]
        start_step = int(partial["step"]) + 1
        elapsed_before = float(partial["elapsed_seconds"])
    started = time.time()
    latest = None
    for step in range(start_step, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        loss, snr_db, mean_flux, scintillation_index = loss_from_independent_pairs(
            kind,
            model,
            evaluator,
            args.batch_size,
            args.frames,
            source_generator,
            screen_generator,
        )
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        latest = (
            float(loss.detach()),
            float(snr_db.detach()),
            float(mean_flux.detach()),
            float(scintillation_index.detach()),
            float(gradient_norm.detach()),
        )
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            record = {
                "step": step,
                "loss": latest[0],
                "optimization_snr_db": latest[1],
                "mean_flux": latest[2],
                "scintillation_index": latest[3],
                "gradient_norm": latest[4],
            }
            if kind == "GSM":
                wg, lcr, curvature = model.physical()
                record.update(
                    {
                        "wg_m": float(wg),
                        "lcr_m": float(lcr),
                        "normalized_curvature": float(curvature),
                        "rg_m": curvature_to_rg(
                            evaluator.config.distance, float(curvature)
                        ),
                    }
                )
            else:
                _, _, waists = model.physical()
                record.update(
                    {
                        "mean_wg_m": float(waists.mean()),
                        "min_wg_m": float(waists.min()),
                        "max_wg_m": float(waists.max()),
                    }
                )
            history.append(record)
        if step % args.checkpoint_every == 0 and step < args.steps:
            partial = {
                "step": step,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "source_generator_state": source_generator.get_state(),
                "screen_generator_state": screen_generator.get_state(),
                "history": history,
                "elapsed_seconds": elapsed_before + time.time() - started,
            }
            torch.save(partial, partial_path)
    if latest is None:
        last_record = history[-1]
        latest = (
            float(last_record["loss"]),
            float(last_record["optimization_snr_db"]),
            float(last_record["mean_flux"]),
            float(last_record["scintillation_index"]),
            float(last_record["gradient_norm"]),
        )
    tail_values = [
        float(item["optimization_snr_db"])
        for item in history
        if int(item["step"]) > max(0, args.steps - 30)
    ]
    result = {
        "link": evaluator.config.name,
        "kind": kind,
        "run": run_index,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "frames": args.frames,
        "elapsed_seconds": elapsed_before + time.time() - started,
        "optimization_terminal_snr_db": latest[1],
        "optimization_tail_median_snr_db": float(np.median(tail_values)),
        "optimization_tail_mean_snr_db": float(np.mean(tail_values)),
        "model_state": cpu_state_dict(model),
        "history": history,
    }
    if kind == "GSM":
        wg, lcr, curvature = model.physical()
        result.update(
            {
                "wg_m": float(wg),
                "lcr_m": float(lcr),
                "normalized_curvature": float(curvature),
                "rg_m": curvature_to_rg(evaluator.config.distance, float(curvature)),
            }
        )
    else:
        _, _, waists = model.physical()
        result.update(
            {
                "mean_wg_m": float(waists.mean()),
                "min_wg_m": float(waists.min()),
                "max_wg_m": float(waists.max()),
            }
        )
    torch.save(result, final_path)
    if partial_path.exists():
        partial_path.unlink()
    return result


@torch.inference_mode()
def collect_independent_pairs(
    kind: str,
    model: torch.nn.Module,
    evaluator: UnifiedWOS,
    samples: int,
    batch_size: int,
    frames: int,
    source_seed: int,
    screen_seed: int,
) -> torch.Tensor:
    source_generator = evaluator.generator(source_seed)
    screen_generator = evaluator.generator(screen_seed)
    chunks: list[torch.Tensor] = []
    completed = 0
    while completed < samples:
        current_batch = min(batch_size, samples - completed)
        sources = sources_for_batch(
            kind, model, evaluator, current_batch, frames, source_generator
        )
        chunks.append(modal_flux_for_pairs(evaluator, sources, screen_generator).cpu())
        completed += current_batch
    return torch.cat(chunks, dim=0)


def load_endpoint_model(
    kind: str,
    endpoint: dict,
    evaluator: UnifiedWOS,
    args: argparse.Namespace,
    link_seed: int,
) -> torch.nn.Module:
    dummy_pcb = torch.tensor([[20e-3, 20e-3, 1.0]], dtype=torch.float32)
    dummy_opms = opms_sobol_points(1, args.frames, link_seed + 909)
    model = build_model(kind, 0, evaluator, dummy_pcb, dummy_opms, args, link_seed)
    model.load_state_dict(endpoint["model_state"])
    model.eval()
    return model


def validate_endpoint(
    kind: str,
    endpoint: dict,
    evaluator: UnifiedWOS,
    samples: int,
    args: argparse.Namespace,
    link_seed: int,
    stage_seed: int,
    bootstrap_repeats: int,
) -> tuple[dict, torch.Tensor]:
    model = load_endpoint_model(kind, endpoint, evaluator, args, link_seed)
    source_seed = stage_seed + 301
    screen_seed = stage_seed + 401
    started = time.time()
    modal_flux = collect_independent_pairs(
        kind,
        model,
        evaluator,
        samples,
        args.validation_batch,
        args.frames,
        source_seed,
        screen_seed,
    )
    metrics = metrics_from_modal_flux(
        modal_flux,
        evaluator.config.duale_b,
        bootstrap_seed=stage_seed + 501,
        bootstrap_repeats=bootstrap_repeats,
    )
    metrics.update(
        {
            "elapsed_seconds": time.time() - started,
            "source_sampling": (
                "independent_sequence_per_atmosphere"
                if kind == "GSM"
                else "fixed_optimized_sequence"
            ),
            "atmosphere_pairing": "one_independent_atmosphere_per_integration_window",
        }
    )
    return metrics, modal_flux


def distribution_summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "median": float(np.median(array)),
        "mean": float(np.mean(array)),
        "sd": float(np.std(array, ddof=1)) if array.size > 1 else 0.0,
        "iqr": float(np.quantile(array, 0.75) - np.quantile(array, 0.25)),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
    }


def update_progress(
    args: argparse.Namespace,
    status: str,
    link: str | None = None,
    kind: str | None = None,
    run: int | None = None,
    detail: str | None = None,
) -> None:
    completed_files = (
        [
            path
            for path in args.output.glob("[ABCD]/*/run_*.pt")
            if re.fullmatch(r"run_\d+\.pt", path.name)
        ]
        if args.output.exists()
        else []
    )
    payload = {
        "status": status,
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "pid": os.getpid(),
        "links_requested": args.links,
        "starts_per_class": args.starts,
        "completed_candidates": len(completed_files),
        "total_candidates": len(args.links) * 2 * args.starts,
        "current_link": link,
        "current_kind": kind,
        "current_run": run,
        "detail": detail,
    }
    write_json(args.output / "progress.json", payload)


def write_candidate_csv(output: Path, rows: list[dict]) -> None:
    fieldnames = [
        "link",
        "kind",
        "run",
        "optimization_terminal_snr_db",
        "optimization_tail_median_snr_db",
        "screening_snr_db",
        "screening_mean_flux",
        "screening_si",
        "screening_meff",
        "elapsed_seconds",
    ]
    with (output / "candidate_summary.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_arguments()
    if args.batch_size != 16:
        raise ValueError("The paper protocol requires --batch-size 16.")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    configuration = {
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "links": {name: asdict(config) for name, config in link_table().items()},
        "sampling_definition": {
            "optimization": "one bs=16 loss; every GSM atmosphere receives an independently generated M-frame source sequence",
            "screening_and_validation": "independent source-atmosphere pairing; OPMS sequence frozen after optimization",
            "selection": f"all endpoints screened with {args.screening_samples} unseen pairs; class winner receives {args.validation_samples}-pair additional validation",
        },
    }
    write_json(args.output / "configuration.json", configuration)
    update_progress(args, "running", detail="initializing")
    links = link_table()
    all_rows: list[dict] = []
    complete_summary: dict[str, dict] = {}
    for link_position, link_name in enumerate(args.links):
        config = links[link_name]
        evaluator = UnifiedWOS(config, device=args.device)
        # Keep A-D order when matching archived seeds: the position enters the seed.
        link_seed = args.seed + 10000000 * (link_position + 1) + 100000 * ord(link_name)
        pcb_points = pcb_sobol_points(args.starts, link_seed + 11)
        opms_points = opms_sobol_points(args.starts, args.frames, link_seed + 22)
        complete_summary[link_name] = {"config": asdict(config), "classes": {}}
        for kind in ("GSM", "OPMS"):
            endpoints: list[dict] = []
            for run_index in range(args.starts):
                update_progress(
                    args,
                    "running",
                    link_name,
                    kind,
                    run_index,
                    "optimizing with one bs=16 joint loss",
                )
                endpoint = train_candidate(
                    kind, run_index, evaluator, pcb_points, opms_points, args, link_seed
                )
                endpoints.append(endpoint)
            screening_results: list[dict] = []
            for endpoint in endpoints:
                run_index = int(endpoint["run"])
                screening_file = (
                    args.output
                    / link_name
                    / kind.lower()
                    / f"run_{run_index:03d}_screening.json"
                )
                flux_file = (
                    args.output
                    / link_name
                    / kind.lower()
                    / f"run_{run_index:03d}_screening_flux.pt"
                )
                if screening_file.exists() and flux_file.exists():
                    metrics = json.loads(screening_file.read_text(encoding="utf-8"))
                else:
                    update_progress(
                        args,
                        "running",
                        link_name,
                        kind,
                        run_index,
                        "screening endpoint on independent source-atmosphere pairs",
                    )
                    stage_seed = link_seed + 700000
                    metrics, modal_flux = validate_endpoint(
                        kind,
                        endpoint,
                        evaluator,
                        args.screening_samples,
                        args,
                        link_seed,
                        stage_seed,
                        bootstrap_repeats=400,
                    )
                    torch.save(modal_flux, flux_file)
                    write_json(screening_file, metrics)
                screening_results.append(metrics)
                all_rows.append(
                    {
                        "link": link_name,
                        "kind": kind,
                        "run": run_index,
                        "optimization_terminal_snr_db": endpoint[
                            "optimization_terminal_snr_db"
                        ],
                        "optimization_tail_median_snr_db": endpoint[
                            "optimization_tail_median_snr_db"
                        ],
                        "screening_snr_db": metrics["snr_db"],
                        "screening_mean_flux": metrics["mean_flux"],
                        "screening_si": metrics["scintillation_index"],
                        "screening_meff": metrics["effective_frame_number"],
                        "elapsed_seconds": endpoint["elapsed_seconds"],
                    }
                )
            winner_index = int(
                np.argmax([item["snr_db"] for item in screening_results])
            )
            winner = endpoints[winner_index]
            final_file = (
                args.output / link_name / kind.lower() / "winner_validation.json"
            )
            final_flux_file = (
                args.output / link_name / kind.lower() / "winner_validation_flux.pt"
            )
            if final_file.exists() and final_flux_file.exists():
                final_metrics = json.loads(final_file.read_text(encoding="utf-8"))
            else:
                update_progress(
                    args,
                    "running",
                    link_name,
                    kind,
                    winner_index,
                    "validating class winner on 1000 independent source-atmosphere pairs",
                )
                final_stage_seed = link_seed + 900000
                final_metrics, final_modal_flux = validate_endpoint(
                    kind,
                    winner,
                    evaluator,
                    args.validation_samples,
                    args,
                    link_seed,
                    final_stage_seed,
                    bootstrap_repeats=1000,
                )
                torch.save(final_modal_flux, final_flux_file)
                write_json(final_file, final_metrics)
            optimization_values = [
                float(item["optimization_tail_median_snr_db"]) for item in endpoints
            ]
            screening_values = [float(item["snr_db"]) for item in screening_results]
            gaps = [
                screening_values[index] - optimization_values[index]
                for index in range(len(endpoints))
            ]
            complete_summary[link_name]["classes"][kind] = {
                "optimization_distribution": distribution_summary(optimization_values),
                "screening_distribution": distribution_summary(screening_values),
                "screening_minus_optimization_distribution": distribution_summary(gaps),
                "winner_run": winner_index,
                "winner_screening": screening_results[winner_index],
                "winner_final_validation": final_metrics,
            }
            write_json(args.output / "summary_partial.json", complete_summary)
            write_candidate_csv(args.output, all_rows)
        del evaluator
        torch.cuda.empty_cache()
    write_json(args.output / "summary.json", complete_summary)
    write_candidate_csv(args.output, all_rows)
    update_progress(
        args,
        "complete",
        detail="pilot optimization, endpoint screening, and winner validation completed",
    )


if __name__ == "__main__":
    main()
