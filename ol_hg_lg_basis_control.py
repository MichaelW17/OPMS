"Wave-optics experiment: ol hg lg basis control."

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from scipy import stats
from torch import nn
from torch.nn import functional as functional

from sources import (
    WG_MIN,
    bounded,
    hermite_gaussian_1d,
    inverse_bounded,
    inverse_softplus,
)
from propagation import LinkConfig, UnifiedWOS, metrics_from_modal_flux


REFERENCE_ROOT = Path(__file__).resolve().parent / "logs/ol_sms_pcb_3param"
OUTPUT_ROOT = Path(__file__).resolve().parent / "logs/ol_hg_lg_basis_control"
CASE_STEMS = ("small_rx", "long_rx20", "strong_small_rx")
FRAME_COUNT = 10
TOTAL_ORDER_MAX = 10
MODE_COUNT = (TOTAL_ORDER_MAX + 1) * (TOTAL_ORDER_MAX + 2) // 2
WAIST_MAX = 80e-3
TRAIN_STEPS = 250
RUNS = 3
FORMAL_REPEATS = 10
FORMAL_BLOCKS = 25
FORMAL_BATCH_SIZE = 8
FORMAL_SEED = 2026082001


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=True),
        encoding="utf-8",
    )


def hg_mode_indices() -> tuple[tuple[int, int], ...]:
    return tuple(
        (m_order, total_order - m_order)
        for total_order in range(TOTAL_ORDER_MAX + 1)
        for m_order in range(total_order + 1)
    )


def circular_to_cartesian_matrix(device: torch.device) -> torch.Tensor:
    matrix = torch.zeros(MODE_COUNT, MODE_COUNT, dtype=torch.complex64, device=device)
    offset = 0
    for total_order in range(TOTAL_ORDER_MAX + 1):
        block_size = total_order + 1
        for n_plus in range(block_size):
            n_minus = total_order - n_plus
            for m_order in range(block_size):
                lower = max(0, m_order - n_minus)
                upper = min(n_plus, m_order)
                polynomial_coefficient = 0.0j
                for x_from_plus in range(lower, upper + 1):
                    x_from_minus = m_order - x_from_plus
                    y_from_plus = n_plus - x_from_plus
                    y_from_minus = n_minus - x_from_minus
                    polynomial_coefficient += (
                        math.comb(n_plus, x_from_plus)
                        * math.comb(n_minus, x_from_minus)
                        * (1j**y_from_plus)
                        * ((-1j) ** y_from_minus)
                    )
                normalization = 2.0 ** (-0.5 * total_order) * math.sqrt(
                    math.factorial(m_order)
                    * math.factorial(total_order - m_order)
                    / (math.factorial(n_plus) * math.factorial(n_minus))
                )
                matrix[offset + m_order, offset + n_plus] = complex(
                    normalization * polynomial_coefficient
                )
        offset += block_size
    identity = torch.eye(MODE_COUNT, dtype=torch.complex64, device=device)
    unitary_error = torch.linalg.matrix_norm(matrix.conj().T @ matrix - identity).item()
    if unitary_error > 2e-5:
        raise RuntimeError(
            f"HG/LG basis transform is not unitary: error={unitary_error}"
        )
    return matrix


def initial_hg_weights(run_index: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(
        2026140101 + 10000 * run_index
    )
    amplitudes = 0.015 + 0.015 * torch.rand(
        FRAME_COUNT, MODE_COUNT, generator=generator
    )
    for frame in range(FRAME_COUNT):
        amplitudes[frame, frame % MODE_COUNT] = 0.8
    phases = (
        0.8
        * math.pi
        * (2.0 * torch.rand(FRAME_COUNT, MODE_COUNT, generator=generator) - 1.0)
    )
    phases[:, 0] = 0.0
    return torch.polar(amplitudes, phases)


class MatchedBasisSMS(nn.Module):
    def __init__(
        self,
        basis_name: str,
        run_index: int,
        transform: torch.Tensor,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.basis_name = basis_name
        self.register_buffer("lg_to_hg", transform)
        hg_weights = initial_hg_weights(run_index).to(device)
        coordinate_weights = (
            hg_weights if basis_name == "HG" else hg_weights @ transform.conj()
        )
        amplitudes = coordinate_weights.abs().clamp_min(1e-6)
        phases = torch.angle(coordinate_weights)
        phases = phases - phases[:, :1]
        phases = torch.atan2(torch.sin(phases), torch.cos(phases))
        self.raw_amplitudes = nn.Parameter(inverse_softplus(amplitudes))
        phase_fraction = (phases[:, 1:] / math.pi).clamp(-0.999, 0.999)
        self.raw_phases = nn.Parameter(torch.atanh(phase_fraction))
        initial_waist = torch.full(
            (FRAME_COUNT,), inverse_bounded(35e-3, WG_MIN, WAIST_MAX), device=device
        )
        self.raw_waists = nn.Parameter(initial_waist)

    def physical(self) -> tuple[torch.Tensor, torch.Tensor]:
        amplitudes = functional.softplus(self.raw_amplitudes)
        relative_phases = math.pi * torch.tanh(self.raw_phases)
        reference = torch.zeros(FRAME_COUNT, 1, device=self.raw_phases.device)
        phases = torch.cat((reference, relative_phases), dim=1)
        coordinate_weights = torch.polar(amplitudes, phases)
        hg_weights = (
            coordinate_weights
            if self.basis_name == "HG"
            else coordinate_weights @ self.lg_to_hg.T
        )
        waists = bounded(self.raw_waists, WG_MIN, WAIST_MAX)
        return hg_weights, waists


def synthesize_source(evaluator: UnifiedWOS, model: MatchedBasisSMS) -> torch.Tensor:
    hg_weights, waists = model.physical()
    one_dimensional = hermite_gaussian_1d(evaluator, waists, TOTAL_ORDER_MAX + 1).to(
        torch.complex64
    )
    flat_indices = torch.tensor(
        [
            m_order * (TOTAL_ORDER_MAX + 1) + n_order
            for m_order, n_order in hg_mode_indices()
        ],
        device=evaluator.device,
        dtype=torch.long,
    )
    square_weights = torch.zeros(
        FRAME_COUNT,
        (TOTAL_ORDER_MAX + 1) ** 2,
        dtype=torch.complex64,
        device=evaluator.device,
    )
    square_weights = square_weights.index_copy(1, flat_indices, hg_weights)
    square_weights = square_weights.reshape(
        FRAME_COUNT, TOTAL_ORDER_MAX + 1, TOTAL_ORDER_MAX + 1
    )
    source = torch.einsum(
        "mij,mix,mjy->mxy", square_weights, one_dimensional, one_dimensional
    )
    source = source * evaluator.curvature_phase(1.0)
    return evaluator.normalize_each_frame(source)


def train_basis(
    evaluator: UnifiedWOS,
    basis_name: str,
    run_index: int,
    transform: torch.Tensor,
    case_index: int,
) -> dict:
    model = MatchedBasisSMS(basis_name, run_index, transform, evaluator.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
    screen_generator = evaluator.generator(
        2026141001 + 100000 * case_index + 10000 * run_index
    )
    history = []
    started = time.time()
    for step in range(1, TRAIN_STEPS + 1):
        optimizer.zero_grad(set_to_none=True)
        accumulated_loss = 0.0
        latest_snr = float("nan")
        for _ in range(2):
            source = synthesize_source(evaluator, model)
            loss, snr_db, _, _ = evaluator.loss_from_source(source, 6, screen_generator)
            (loss / 2.0).backward()
            accumulated_loss += float(loss.detach()) / 2.0
            latest_snr = float(snr_db.detach())
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
        if step == 1 or step % 20 == 0 or step == TRAIN_STEPS:
            history.append(
                {"step": step, "loss": accumulated_loss, "train_snr_db": latest_snr}
            )
    return {
        "basis": basis_name,
        "run": run_index,
        "elapsed_seconds": time.time() - started,
        "model_state": model.state_dict(),
        "history": history,
    }


@torch.inference_mode()
def evaluate_model(
    evaluator: UnifiedWOS,
    result: dict,
    transform: torch.Tensor,
    case_index: int,
    selection: bool,
) -> dict:
    model = MatchedBasisSMS(
        result["basis"], int(result["run"]), transform, evaluator.device
    )
    model.load_state_dict(result["model_state"])
    source = synthesize_source(evaluator, model)
    repeats = 3 if selection else FORMAL_REPEATS
    blocks = 10 if selection else FORMAL_BLOCKS
    base_seed = (2026142001 + 100000 * case_index) if selection else FORMAL_SEED
    per_repeat = []
    for repeat in range(repeats):
        screen_seed = base_seed + 10000 * repeat + 22
        modal_flux = evaluator.collect_modal_flux(
            source, blocks, FORMAL_BATCH_SIZE, screen_seed
        )
        per_repeat.append(
            metrics_from_modal_flux(
                modal_flux,
                evaluator.config.duale_b,
                bootstrap_seed=base_seed + repeat,
                bootstrap_repeats=200,
            )
        )
    metric_names = (
        "snr_db",
        "mean_flux",
        "scintillation_index",
        "effective_frame_number",
        "mean_pairwise_power_correlation",
        "mean_individual_frame_si",
    )
    averaged = {
        name: float(np.mean([item[name] for item in per_repeat]))
        for name in metric_names
        if name in per_repeat[0]
    }
    snr_values = np.asarray([item["snr_db"] for item in per_repeat], dtype=float)
    sem = float(stats.sem(snr_values)) if repeats > 1 else 0.0
    multiplier = float(stats.t.ppf(0.975, repeats - 1)) if repeats > 1 else 0.0
    averaged.update(
        {
            "snr_db_ci95_low": averaged["snr_db"] - multiplier * sem,
            "snr_db_ci95_high": averaged["snr_db"] + multiplier * sem,
            "per_repeat": per_repeat,
            "repeats": repeats,
            "samples_per_repeat": blocks * FORMAL_BATCH_SIZE,
        }
    )
    return averaged


@torch.inference_mode()
def representation_error(evaluator: UnifiedWOS, transform: torch.Tensor) -> float:
    hg_model = MatchedBasisSMS("HG", 0, transform, evaluator.device)
    lg_model = MatchedBasisSMS("LG", 0, transform, evaluator.device)
    hg_source = synthesize_source(evaluator, hg_model)
    lg_source = synthesize_source(evaluator, lg_model)
    frame_phase = (hg_source.flatten(1).conj() * lg_source.flatten(1)).sum(dim=1)
    frame_phase = frame_phase / frame_phase.abs().clamp_min(1e-20)
    aligned_lg = lg_source * frame_phase.conj().reshape(-1, 1, 1)
    relative_error = torch.linalg.vector_norm(
        (hg_source - aligned_lg).flatten()
    ) / torch.linalg.vector_norm(hg_source.flatten()).clamp_min(1e-20)
    return float(relative_error)


def run_case(case_stem: str, case_index: int) -> dict:
    output_directory = OUTPUT_ROOT / case_stem
    output_directory.mkdir(parents=True, exist_ok=True)
    output_path = output_directory / "basis_control.json"
    state_path = output_directory / "basis_control_states.pt"
    if output_path.exists() and state_path.exists():
        return read_json(output_path)
    link = read_json(REFERENCE_ROOT / f"{case_stem}_M10" / "summary.json")["link"]
    evaluator = UnifiedWOS(LinkConfig(**link))
    transform = circular_to_cartesian_matrix(evaluator.device)
    equivalence_error = representation_error(evaluator, transform)
    candidates = {"HG": [], "LG": []}
    for basis_name in ("HG", "LG"):
        for run_index in range(RUNS):
            print(
                f"{case_stem} {basis_name} basis run {run_index + 1}/{RUNS}", flush=True
            )
            trained = train_basis(
                evaluator, basis_name, run_index, transform, case_index
            )
            trained["selection_metrics"] = evaluate_model(
                evaluator, trained, transform, case_index, selection=True
            )
            candidates[basis_name].append(trained)
            torch.cuda.empty_cache()
    selected = {
        basis_name: max(
            candidates[basis_name], key=lambda item: item["selection_metrics"]["snr_db"]
        )
        for basis_name in ("HG", "LG")
    }
    formal = {
        basis_name: evaluate_model(
            evaluator, selected[basis_name], transform, case_index, selection=False
        )
        for basis_name in ("HG", "LG")
    }
    differences = np.asarray(
        [
            formal["LG"]["per_repeat"][index]["snr_db"]
            - formal["HG"]["per_repeat"][index]["snr_db"]
            for index in range(FORMAL_REPEATS)
        ],
        dtype=float,
    )
    sem = float(stats.sem(differences))
    multiplier = float(stats.t.ppf(0.975, FORMAL_REPEATS - 1))
    paired = {
        "mean_db": float(differences.mean()),
        "ci95_low_db": float(differences.mean() - multiplier * sem),
        "ci95_high_db": float(differences.mean() + multiplier * sem),
    }
    serializable_candidates = {
        basis_name: [
            {key: value for key, value in item.items() if key != "model_state"}
            for item in items
        ]
        for basis_name, items in candidates.items()
    }
    result = {
        "case": case_stem,
        "link": link,
        "frame_count": FRAME_COUNT,
        "total_order_max": TOTAL_ORDER_MAX,
        "mode_count": MODE_COUNT,
        "subspace_definition": "HG: m+n<=Nmax; LG: 2p+|ell|<=Nmax",
        "representation_relative_l2_error": equivalence_error,
        "formal_metrics": formal,
        "lg_minus_hg": paired,
        "selected_runs": {
            basis_name: int(selected[basis_name]["run"]) for basis_name in ("HG", "LG")
        },
        "all_candidates": serializable_candidates,
    }
    torch.save(
        {
            "result": result,
            "selected_states": {
                basis_name: selected[basis_name]["model_state"]
                for basis_name in ("HG", "LG")
            },
            "all_candidate_states": {
                basis_name: [item["model_state"] for item in items]
                for basis_name, items in candidates.items()
            },
            "lg_to_hg": transform.detach().cpu(),
        },
        state_path,
    )
    write_json(output_path, result)
    return result


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    results = [
        run_case(case_stem, case_index)
        for case_index, case_stem in enumerate(CASE_STEMS)
    ]
    write_json(OUTPUT_ROOT / "basis_control_summary.json", results)
    print(
        f"Completed matched HG/LG basis control in {OUTPUT_ROOT.resolve()}", flush=True
    )


if __name__ == "__main__":
    main()
