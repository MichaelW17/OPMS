# OPMS

Code and simulation data for **Initialization-robust pseudo-mode synthesis for free-space optical communications**, by Qianyi Zhang and Minghao Wang.

This repository contains the differentiable split-step propagation model, GSM and OPMS source synthesis, scrambled Sobol multistart optimization, frozen endpoint parameters, raw receiver powers, and the data used for the landscape, CSD, structured-beam, frame-budget, and basis comparisons.

The release was checked against the `1007.docx` manuscript available on 8 October 2026. Two inconsistencies between that draft's methods text and its numerical results are documented in [MANUSCRIPT_AUDIT.md](MANUSCRIPT_AUDIT.md). The default figure command reproduces the reported numerical success rates; an explicit option reproduces the top-100 criterion written in the methods. The archived multistart validation uses **1000 atmospheres per endpoint**, not 32,000.

## Quick start

Python 3.10 and PyTorch 2.1.0 were used for the original calculations. Figure reproduction runs on a CPU. Full simulation is intended for a CUDA GPU; the tested device was an NVIDIA RTX 4090 D.

```bash
git clone https://github.com/MichaelW17/OPMS.git
cd OPMS
python -m venv .venv
```

Activate the environment with `.venv\Scripts\Activate.ps1` on Windows PowerShell, or `source .venv/bin/activate` on Linux/macOS. Install the dependencies:

```bash
python -m pip install -r requirements.txt
python reproduce.py --verify-flux --verify-checksums
```

The command rebuilds six figures as PNG and PDF in `results/`, exports the endpoint and CSD statistics, checks data integrity, and recomputes all 8192 endpoint SNRs from the raw receiver powers. It does not run wave-optics propagation. Fig. 1 is a regenerated workflow schematic rather than the original Visio artwork.

For the alternative success reference described in the manuscript's methods:

```bash
python reproduce.py --reference-count 100 --output results/top100
```

## Run the multistart simulation

Use a CUDA-enabled PyTorch installation for the following commands. A short integration check keeps the paper's 512×512 grid, ten pseudo-modes and joint batch of sixteen atmospheres, but reduces the number of starts and updates:

```bash
python multistart.py --links A --starts 1 --steps 2 --screening-samples 16 --validation-samples 16 --output work/integration_check
```

The complete archived experiment is specified by:

```bash
python multistart.py --links A B C D --starts 1024 --steps 300 --batch-size 16 --frames 10 --screening-samples 1000 --validation-samples 1000 --validation-batch 16 --pcb-lr 0.04 --opms-lr 0.02 --seed 2026092401 --output work/multistart
```

Keep links in A–D order when reproducing the archived random streams: link position enters the seed. Repeating the same command resumes checkpoints and reuses completed endpoint evaluations. Use a fresh output directory after changing parameters. `--screening-samples` is the common unseen ensemble used for **every** endpoint; `--validation-samples` is an additional ensemble for the selected class winner. They are separate evaluations.

For a new run with the sample count stated in the manuscript, use `--screening-samples 32000 --validation-samples 32000` and a new output directory. This is an additional experiment, not the source of the archived manuscript numbers. Full multistart simulation was not rerun during release preparation.

## Model and data layout

| File or directory | Purpose |
|---|---|
| `propagation.py` | Modified von Kármán phase screens, subharmonics, split-step propagation and aperture statistics |
| `sources.py` | Bounded GSM parameters, HG recurrence and OPMS synthesis |
| `multistart.py` | Sobol initialization, Adam, independent source-atmosphere pairing and endpoint validation |
| `landscape.py` | Collimated GSM landscape and ten live Adam trajectories |
| `recompute_csd.py` | CSD distance computation from frozen source parameters |
| `reproduce.py` | Figure and statistics reproduction from archived data |
| `data/multistart/` | Eight compact endpoint, power and metric archives; original experiment configuration |
| `data/csd/` | Pairwise CSD distances and selected endpoint identifiers |
| `data/controls/` | Structured-beam, frame-budget and HG/LG figure tables |
| `logs/` | Supporting control experiment parameters, source states and receiver powers |
| `source_map.json` | Mapping from released modules to research script names |
| `data_manifest.json` | SHA-256 checksums and sizes of all archived data files |
| `results/` | Reproduced figures, statistics and release verification results |

Each `{link}_{kind}_flux.npz` contains `modal_flux` with shape `(1024, 1000, 10)`: endpoint, independent atmospheric window, pseudo-mode. Values preserve the original saved floating-point precision. Each endpoint archive contains 1024 dictionaries with source parameters and optimization histories. Endpoint indices are zero-based. The archives include all endpoints, not only successful ones.

Example reading the data:

```python
import numpy as np

with np.load("data/multistart/A_opms_flux.npz") as archive:
    modal_flux = archive["modal_flux"]
power = modal_flux.mean(axis=-1)  # Detector integrates ten pseudo-modes.
mean_power = power.mean(axis=-1)
si = (power ** 2).mean(axis=-1) / mean_power ** 2 - 1
snr_db = 10 * np.log10(1 / np.sqrt(0.1 / mean_power + np.maximum(si, 0)))
```

The numerical convention is `10 log10(1 / sqrt(Λe / mean_power + SI))`, consistent with the archived results. Pool linear powers before estimating moments. Averaging SNR values computed from small batches gives a different estimator.

GSM generates an independent ten-frame source sequence for every atmospheric window. OPMS freezes its optimized sequence across validation atmospheres. Within each detector window, every pseudo-mode sees the same frozen atmosphere. Both sources are truncated by the same transmitter aperture and normalized frame by frame.

## Additional reproduction commands

See [REPRODUCTION.md](REPRODUCTION.md) for the landscape, CSD and control experiments, their inputs, and the scope of release testing. The standalone code and data are under the repository's existing Apache-2.0 license. Please cite the manuscript using [CITATION.cff](CITATION.cff).
