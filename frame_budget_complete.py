"Wave-optics experiment: ol frame budget complete."

from __future__ import annotations

import argparse
import csv
import itertools
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy import special, stats

from revalidate_results import validate_one
from sources import serializable_result, train_pcb, validate_result
from sources import run_experiment
from structured_compare import weighted_metrics
from propagation import LinkConfig, UnifiedWOS


REFERENCE_ROOT = Path(__file__).resolve().parent / "logs/ol_sms_pcb_3param"
STRUCTURED_ROOT = Path(__file__).resolve().parent / "logs/ol_structured_comparison_v3"
BOUNDARY_ROOT = Path(__file__).resolve().parent / "logs/ol_structured_boundary_v5"
M51_ROOT = Path(__file__).resolve().parent / "logs/ol_dad_sms_m51"
OUTPUT_ROOT = Path(__file__).resolve().parent / "logs/ol_frame_budget_complete"
CASE_STEMS = ("small_rx", "long_rx20", "strong_small_rx")
CASE_LABELS = {
    "small_rx": "1 km, Rx 10 mm",
    "long_rx20": "2 km, Rx 20 mm",
    "strong_small_rx": "1 km, strong turbulence",
}
FRAME_VALUES = (1, 2, 4, 10, 20, 51)
DAD_FRAME_VALUES = (11, 23, 51)
FORMAL_REPEATS = 10
FORMAL_SAMPLES = 200
FORMAL_SEED = 2026082001
DAD_BETAS = (60.0, 90.0, 120.0, 180.0, 240.0, 300.0, 360.0)
DAD_ALPHA_ABS = (0.25, 0.5, 0.75, 1.0)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=True),
        encoding="utf-8",
    )


def reference_directory(case_stem: str, frame_count: int) -> Path:
    return REFERENCE_ROOT / f"{case_stem}_M{frame_count}"


def load_link(case_stem: str) -> dict:
    return read_json(reference_directory(case_stem, 10) / "summary.json")["link"]


def m20_arguments(case_stem: str, case_index: int) -> argparse.Namespace:
    link = load_link(case_stem)
    return argparse.Namespace(
        name=f"{case_stem}_M20",
        frames=20,
        wavelength=float(link["wavelength"]),
        cn2=float(link["cn2"]),
        distance=float(link["distance"]),
        transmitter_radius=float(link["transmitter_radius"]),
        receiver_radius=float(link["receiver_radius"]),
        inner_scale=float(link["inner_scale"]),
        outer_scale=float(link["outer_scale"]),
        duale_b=float(link["duale_b"]),
        power=float(link["power"]),
        wos_steps=int(link["wos_steps"]),
        steps=250,
        runs=3,
        mc_blocks=2,
        train_bs=6,
        pcb_lr=0.04,
        sms_lr=0.02,
        grad_clip=10.0,
        n_side=8,
        pcb_wg_max=100e-3,
        sms_wg_max=80e-3,
        lcr_max=50e-3,
        log_every=20,
        selection_repeats=2,
        selection_blocks=8,
        holdout_repeats=3,
        holdout_blocks=20,
        eval_bs=8,
        bootstrap_repeats=1000,
        seed=2026110101 + 100000 * case_index,
        selection_seed=2026111001 + 100000 * case_index,
        holdout_seed=2026112001 + 100000 * case_index,
        output_dir=OUTPUT_ROOT / f"{case_stem}_M20",
    )


def ensure_m20(case_stem: str, case_index: int) -> Path:
    arguments = m20_arguments(case_stem, case_index)
    result_path = arguments.output_dir / "result.pt"
    if not result_path.exists():
        run_experiment(arguments)
    validate_one(
        result_path, FORMAL_REPEATS, FORMAL_SAMPLES // 8, 8, FORMAL_SEED, 1000, False
    )
    return result_path


def m51_pcb_arguments(case_index: int) -> argparse.Namespace:
    return argparse.Namespace(
        runs=3,
        steps=320,
        pcb_lr=0.04,
        mc_blocks=2,
        train_bs=3,
        grad_clip=10.0,
        pcb_wg_max=100e-3,
        lcr_max=50e-3,
        log_every=20,
        seed=2026120101 + 100000 * case_index,
        selection_repeats=3,
        selection_blocks=12,
        selection_seed=2026121001 + 100000 * case_index,
        holdout_repeats=3,
        holdout_blocks=12,
        holdout_seed=2026122001 + 100000 * case_index,
        eval_bs=4,
        bootstrap_repeats=300,
    )


def ensure_m51_pair(case_stem: str, case_index: int) -> Path:
    output_directory = OUTPUT_ROOT / f"{case_stem}_M51"
    output_directory.mkdir(parents=True, exist_ok=True)
    result_path = output_directory / "result.pt"
    if not result_path.exists():
        link = load_link(case_stem)
        link["name"] = f"{case_stem}_M51"
        evaluator = UnifiedWOS(LinkConfig(**link))
        arguments = m51_pcb_arguments(case_index)
        candidates = []
        for run_index in range(arguments.runs):
            print(
                f"{case_stem} M=51 PCB run {run_index + 1}/{arguments.runs}", flush=True
            )
            trained = train_pcb(evaluator, 51, run_index, arguments)
            trained["selection_metrics"] = validate_result(
                evaluator, trained, 51, arguments, selection=True
            )
            candidates.append(trained)
            torch.cuda.empty_cache()
        selected_pcb = max(
            candidates, key=lambda item: item["selection_metrics"]["snr_db"]
        )
        sms_state_path = M51_ROOT / f"{case_stem}_M51" / "m51_sms_state.pt"
        sms_state = torch.load(sms_state_path, map_location="cpu", weights_only=False)
        m51_comparison = read_json(
            M51_ROOT / f"{case_stem}_M51" / "m51_comparison.json"
        )
        sms_selected = dict(m51_comparison["sms"]["selected"])
        sms_selected["kind"] = "SMS"
        sms_selected["n_side"] = int(sms_state["n_side"])
        summary = {
            "schema_version": 3,
            "link": link,
            "frame_count": 51,
            "optimization": {
                "steps": arguments.steps,
                "runs": arguments.runs,
                "mc_blocks": arguments.mc_blocks,
                "train_bs": arguments.train_bs,
                "pcb_lr": arguments.pcb_lr,
                "sms_lr": 0.02,
                "sms_n_side": int(sms_state["n_side"]),
                "pcb_wg_max": 100e-3,
                "sms_wg_max": float(sms_state["sms_wg_max"]),
                "lcr_max": 50e-3,
                "sms_rg_rule": "Rg=distance",
                "sms_hg_implementation": "separable_1d_direct_synthesis",
                "pcb_parameters": ["Wg", "Lcr", "Rg"],
            },
            "selected": {"PCB": serializable_result(selected_pcb), "SMS": sms_selected},
            "all_candidates": {
                "PCB": [serializable_result(item) for item in candidates],
                "SMS": [sms_selected],
            },
            "provenance": {
                "sms_state": str(sms_state_path.resolve()),
                "pcb_generated_by": str(Path(__file__).resolve()),
            },
        }
        torch.save(
            {
                "summary": summary,
                "selected_model_states": {
                    "PCB": selected_pcb["model_state"],
                    "SMS": sms_state["selected_sms_state"],
                },
            },
            result_path,
        )
        write_json(output_directory / "pcb_training_summary.json", summary)
    validate_one(
        result_path, FORMAL_REPEATS, FORMAL_SAMPLES // 4, 4, FORMAL_SEED, 1000, False
    )
    return result_path


def dad_orders(frame_count: int) -> tuple[int, ...]:
    half_width = (frame_count - 1) // 2
    return tuple(range(-half_width, half_width + 1))


def dad_weights(frame_count: int, alpha: float) -> torch.Tensor:
    values = [1.0 + ((-1) ** abs(order)) * alpha for order in dad_orders(frame_count)]
    weights = torch.tensor(values, dtype=torch.float64)
    return weights / weights.sum().clamp_min(1e-30)


@torch.inference_mode()
def dad_basis(evaluator: UnifiedWOS, frame_count: int, beta: float) -> torch.Tensor:
    radial_numpy = evaluator.rr.detach().cpu().numpy().astype(np.float64, copy=False)
    squared_index = np.rint(np.square(radial_numpy / evaluator.config.delta)).astype(
        np.int64
    )
    unique_squared, inverse_index = np.unique(squared_index, return_inverse=True)
    unique_radius = evaluator.config.delta * np.sqrt(unique_squared.astype(np.float64))
    azimuth = torch.atan2(evaluator.yy, evaluator.xx)
    modes = []
    for order in dad_orders(frame_count):
        unique_mode = special.jv(order, beta * unique_radius)
        radial_mode_numpy = unique_mode[inverse_index].reshape(radial_numpy.shape)
        radial_mode = torch.from_numpy(radial_mode_numpy).to(
            device=evaluator.device, dtype=evaluator.real_dtype
        )
        azimuthal_phase = torch.exp(
            torch.complex(torch.zeros_like(azimuth), order * azimuth)
        )
        modes.append(radial_mode.to(evaluator.complex_dtype) * azimuthal_phase)
    return torch.stack(modes, dim=0)


@torch.inference_mode()
def dad_source_from_basis(
    evaluator: UnifiedWOS, basis: torch.Tensor, envelope_radius: float
) -> torch.Tensor:
    envelope = torch.exp(-evaluator.r2 / envelope_radius**2)
    return evaluator.normalize_each_frame(basis * envelope.unsqueeze(0))


@torch.inference_mode()
def evaluate_weighted_repeats(
    evaluator: UnifiedWOS,
    source: torch.Tensor,
    weights: torch.Tensor,
    repeats: int,
    blocks: int,
    batch_size: int,
    seed: int,
) -> dict:
    per_repeat = []
    for repeat in range(repeats):
        screen_seed = seed + 10000 * repeat + 22
        modal_flux = evaluator.collect_modal_flux(
            source, blocks, batch_size, screen_seed
        )
        per_repeat.append(
            weighted_metrics(modal_flux, weights, evaluator.config.duale_b)
        )
    snr_values = np.asarray([item["snr_db"] for item in per_repeat], dtype=float)
    mean_snr = float(snr_values.mean())
    sem = float(stats.sem(snr_values)) if repeats > 1 else 0.0
    multiplier = float(stats.t.ppf(0.975, repeats - 1)) if repeats > 1 else 0.0
    return {
        "snr_db": mean_snr,
        "snr_db_ci95_low": mean_snr - multiplier * sem,
        "snr_db_ci95_high": mean_snr + multiplier * sem,
        "mean_flux": float(np.mean([item["mean_flux"] for item in per_repeat])),
        "scintillation_index": float(
            np.mean([item["scintillation_index"] for item in per_repeat])
        ),
        "per_repeat": per_repeat,
        "repeats": repeats,
        "samples_per_repeat": blocks * batch_size,
        "frame_count": int(source.shape[0]),
    }


def ensure_dad(case_stem: str, case_index: int, frame_count: int) -> dict:
    output_path = OUTPUT_ROOT / f"{case_stem}_DAD_M{frame_count}" / "dad_result.json"
    if output_path.exists():
        return read_json(output_path)
    evaluator = UnifiedWOS(LinkConfig(**load_link(case_stem)))
    radius = evaluator.config.transmitter_radius
    envelope_values = (radius, 2.0 * radius, 4.0 * radius, 1000.0 * radius)
    alpha_values = tuple(-value for value in DAD_ALPHA_ABS) + DAD_ALPHA_ABS
    coarse_rows = []
    coarse_seed = 2026130101 + 100000 * case_index + 1000 * frame_count
    started = time.time()
    for beta, envelope_radius in itertools.product(DAD_BETAS, envelope_values):
        basis = dad_basis(evaluator, frame_count, beta)
        source = dad_source_from_basis(evaluator, basis, envelope_radius)
        modal_flux = evaluator.collect_modal_flux(source, 6, 4, coarse_seed)
        for alpha in alpha_values:
            metrics = weighted_metrics(
                modal_flux, dad_weights(frame_count, alpha), evaluator.config.duale_b
            )
            coarse_rows.append(
                {
                    "candidate": {
                        "family": "dark" if alpha < 0.0 else "antidark",
                        "transverse_wavenumber_rad_m": beta,
                        "envelope_radius_m": envelope_radius,
                        "alpha": alpha,
                        "mode_count": frame_count,
                    },
                    "coarse_metrics": metrics,
                }
            )
        del basis, source, modal_flux
        torch.cuda.empty_cache()
    top_rows = sorted(
        coarse_rows, key=lambda item: item["coarse_metrics"]["snr_db"], reverse=True
    )[:12]
    refined_rows = []
    refine_seed = 2026131001 + 100000 * case_index + 1000 * frame_count
    for rank, row in enumerate(top_rows):
        candidate = row["candidate"]
        basis = dad_basis(
            evaluator, frame_count, float(candidate["transverse_wavenumber_rad_m"])
        )
        source = dad_source_from_basis(
            evaluator, basis, float(candidate["envelope_radius_m"])
        )
        refined = evaluate_weighted_repeats(
            evaluator,
            source,
            dad_weights(frame_count, float(candidate["alpha"])),
            3,
            20,
            4,
            refine_seed,
        )
        refined_rows.append(
            {
                "candidate": candidate,
                "coarse_metrics": row["coarse_metrics"],
                "refined_metrics": refined,
            }
        )
        print(
            f"{case_stem} DAD M={frame_count} refined {rank + 1}/{len(top_rows)} SNR={refined['snr_db']:.3f} dB",
            flush=True,
        )
        del basis, source
    selected = max(refined_rows, key=lambda item: item["refined_metrics"]["snr_db"])
    candidate = selected["candidate"]
    final_basis = dad_basis(
        evaluator, frame_count, float(candidate["transverse_wavenumber_rad_m"])
    )
    final_source = dad_source_from_basis(
        evaluator, final_basis, float(candidate["envelope_radius_m"])
    )
    formal = evaluate_weighted_repeats(
        evaluator,
        final_source,
        dad_weights(frame_count, float(candidate["alpha"])),
        FORMAL_REPEATS,
        FORMAL_SAMPLES // 4,
        4,
        FORMAL_SEED,
    )
    result = {
        "case": case_stem,
        "frame_count": frame_count,
        "link": evaluator.config.__dict__,
        "candidate_count": len(coarse_rows),
        "selected": selected,
        "formal_metrics": formal,
        "coarse_grid": coarse_rows,
        "elapsed_seconds": time.time() - started,
        "selection_rule": "dark and antidark candidates were pooled before formal evaluation; the branch was fixed on the independent refinement set",
    }
    write_json(output_path, result)
    return result


def existing_formal(case_stem: str, frame_count: int) -> dict:
    return read_json(
        reference_directory(case_stem, frame_count) / "formal_holdout.json"
    )


def coherent_metrics(case_stem: str) -> dict:
    return read_json(reference_directory(case_stem, 10) / "coherent_boundary.json")[
        "formal_metrics"
    ]


def append_metric(
    rows: list[dict],
    case_stem: str,
    frame_count: int,
    beam: str,
    metrics: dict,
    branch: str = "",
) -> None:
    rows.append(
        {
            "case": case_stem,
            "case_label": CASE_LABELS[case_stem],
            "frame_count": frame_count,
            "beam": beam,
            "snr_db": float(metrics["snr_db"]),
            "ci95_low_db": float(metrics.get("snr_db_ci95_low", metrics["snr_db"])),
            "ci95_high_db": float(metrics.get("snr_db_ci95_high", metrics["snr_db"])),
            "mean_flux": float(metrics.get("mean_flux", np.nan)),
            "scintillation_index": float(metrics.get("scintillation_index", np.nan)),
            "effective_frame_number": float(
                metrics.get("effective_frame_number", np.nan)
            ),
            "selected_branch": branch,
        }
    )


def structured_static_metrics(case_stem: str) -> dict[str, dict]:
    structured = read_json(
        STRUCTURED_ROOT / f"{case_stem}_M10" / "structured_comparison.json"
    )
    boundary = read_json(BOUNDARY_ROOT / f"{case_stem}_M10" / "boundary_extension.json")
    return {
        "Sharp": boundary["sharp"]["formal_metrics"],
        "Pin": structured["structured_families"]["pin"]["formal_metrics"],
        "IOPB": boundary["inverted_pin"]["formal_metrics"],
    }


def m51_dad_existing(case_stem: str) -> dict:
    payload = read_json(M51_ROOT / f"{case_stem}_M51" / "m51_comparison.json")
    dark_selection = float(payload["dark"]["selected"]["refined_metrics"]["snr_db"])
    antidark_selection = float(
        payload["antidark"]["selected"]["refined_metrics"]["snr_db"]
    )
    return (
        payload["dark"] if dark_selection >= antidark_selection else payload["antidark"]
    )


def aggregate_results() -> list[dict]:
    rows = []
    for case_stem in CASE_STEMS:
        cb_metrics = coherent_metrics(case_stem)
        static_metrics = structured_static_metrics(case_stem)
        for frame_count in FRAME_VALUES:
            formal = (
                existing_formal(case_stem, frame_count)
                if frame_count <= 10
                else read_json(
                    OUTPUT_ROOT / f"{case_stem}_M{frame_count}" / "formal_holdout.json"
                )
            )
            finite_pcb = formal["PCB"]
            if float(cb_metrics["snr_db"]) > float(finite_pcb["snr_db"]):
                pcb_metrics = cb_metrics
                pcb_branch = "coherent boundary"
            else:
                pcb_metrics = finite_pcb
                pcb_branch = "finite coherence"
            append_metric(rows, case_stem, frame_count, "SMS", formal["SMS"])
            append_metric(rows, case_stem, frame_count, "GSM", pcb_metrics, pcb_branch)
            append_metric(
                rows, case_stem, frame_count, "CB", cb_metrics, "coherent Gaussian"
            )
            for beam, metrics in static_metrics.items():
                append_metric(
                    rows, case_stem, frame_count, beam, metrics, "single coherent field"
                )
        for frame_count in DAD_FRAME_VALUES[:-1]:
            payload = read_json(
                OUTPUT_ROOT / f"{case_stem}_DAD_M{frame_count}" / "dad_result.json"
            )
            append_metric(
                rows,
                case_stem,
                frame_count,
                "DAD",
                payload["formal_metrics"],
                payload["selected"]["candidate"]["family"],
            )
        m51_dad = m51_dad_existing(case_stem)
        append_metric(
            rows, case_stem, 51, "DAD", m51_dad["formal_metrics"], m51_dad["family"]
        )
    return rows


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    for case_index, case_stem in enumerate(CASE_STEMS):
        ensure_m20(case_stem, case_index)
        ensure_m51_pair(case_stem, case_index)
        for frame_count in DAD_FRAME_VALUES[:-1]:
            ensure_dad(case_stem, case_index, frame_count)
    rows = aggregate_results()
    write_csv(rows, OUTPUT_ROOT / "frame_budget_all_beams.csv")
    write_json(OUTPUT_ROOT / "frame_budget_all_beams.json", rows)
    print(f"Completed full frame-budget data in {OUTPUT_ROOT.resolve()}", flush=True)


if __name__ == "__main__":
    main()
