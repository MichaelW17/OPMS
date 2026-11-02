# OPMS: Over-parameterized Pseudo-mode Synthesis

Code and simulation data for **Over-parameterized pseudo-mode synthesis for free-space optical communication**, by Minghao Wang and Qianyi Zhang.

This repository provides the differentiable wave-optics model, GSM and OPMS source synthesis, Sobol-initialized optimization, and the data and commands used to reproduce the paper and Supplement 1. The experiments examine receiver SNR, initialization robustness, structural diversity of source cross-spectral densities (CSDs), low-loss connections between optimized sources, and local source manifolds.

## Quick start

The calculations use Python 3.10 and PyTorch 2.1.0. Figures and statistics can be reproduced on a CPU. Wave-optics optimization requires a CUDA-enabled PyTorch installation; the tested GPU was an NVIDIA RTX 4090 D.

```bash
git clone https://github.com/MichaelW17/OPMS.git
cd OPMS
python -m venv .venv
```

Activate the environment with `.venv\Scripts\Activate.ps1` on Windows PowerShell, or `source .venv/bin/activate` on Linux/macOS, then run:

```bash
python -m pip install -r requirements.txt
python reproduce.py --verify-flux --verify-checksums
```

This command generates the main-paper and supplementary figures as PNG and PDF files in `results/`, checks the data checksums, and recomputes all **24,576 endpoint SNRs** from the archived receiver powers. To use a different output directory:

```bash
python reproduce.py --output results/reproduced
```

## Experiments and figures

| Result | Description | Archived-data command |
|---|---|---|
| Fig. 1 | Beam-optimization workflow | `python reproduce.py` |
| Fig. 2 | Collimated GSM SNR landscape and ten Adam trajectories | `python reproduce.py` |
| Fig. 3 | Endpoint distributions and success rates for GSM, 9-HG OPMS, and 64-HG OPMS, each at M=3 and M=10 | `python plot_fig3.py` |
| Fig. 4 | Pairwise CSD distances among successful M=10 GSM and 64-HG OPMS sources | `python plot_fig4.py` |
| Low-loss connections | Sixteen quadratic Bézier paths between distant successful OPMS sources | `python experiments/paths/summarize_paths.py` |
| Fig. 5 | Local beam-parameter variations around selected 64-HG OPMS sources | `python experiments/local_manifolds/plot_archived.py` |
| Fig. S1 | SNR versus modal truncation order M | `python reproduce.py` |
| Fig. S2 | Paired LG-minus-HG SNR differences | `python reproduce.py` |
| Fig. S3 | OPMS, GSM, CB, Sharp, Pin, IOPB, and DAD benchmarks | `python benchmark_screening.py`, then `python reproduce.py` |

Supplementary tables include propagation-step sensitivity and weighted pure-HG-mode (wHG) comparisons. [REPRODUCTION.md](REPRODUCTION.md) gives the simulation commands and input/output details; [SUPPLEMENT_BENCHMARK.md](SUPPLEMENT_BENCHMARK.md) specifies the structured-source models and evaluation protocols.

### Multistart optimization

Each source configuration and link has 1024 scrambled Sobol initializations, 300 Adam updates, and validation over 1000 independent atmospheric realizations. The six configurations produce 24,576 endpoints across Links A–D. For each configuration and link, the reference SNR is the mean of the highest 100 validated endpoint SNRs. An endpoint is successful when its SNR is at least this reference minus 0.5 dB. Fig. 3 shows the SNR distributions and success rates, with median markers and 95% Wilson score intervals.

To independently recompute the endpoint scores from receiver powers:

```bash
python verify_submission.py
```

### CSD distances and low-loss paths

GSM distances use the analytical ensemble CSD evaluated by Sobol quadrature. OPMS distances use the CSD of the optimized fixed pseudo-mode sequence and finite-field Gram identities. The source states, distance matrices, and endpoint identifiers are provided together.

The path data contain four distant endpoint pairs per link. Each pair is connected by a quadratic Bézier curve whose endpoints remain fixed while its control point is optimized. Validation uses 1000 independent atmospheres at 33 equally spaced path positions. The path barrier is the SNR of the poorer endpoint minus the minimum SNR over these positions, floored at zero. All sixteen optimized paths have barriers below 0.5 dB at the evaluated positions.

### Local source manifolds

For each link, the central source is the highest-validated-SNR endpoint among the 1024 M=10, 64-HG OPMS designs. Each frame retains its optimized waist and is normalized separately to the launch power. Two beam-parameter directions are selected from the SNR Hessian using 4800 independent atmospheric realizations; a separate ensemble of 1000 atmospheres evaluates the local variations.

The archived data contain the centers, perturbation directions, curvature matrices, source representations, and receiver powers at a 9×9 grid over `u,v ∈ [-0.1,0.1]`. The maximum sampled absolute SNR changes for Links A–D are 0.0177, 0.0146, 0.0085, and 0.0069 dB. Exact-rational Bernstein bounds certify a one-to-one CSD parameterization with two independent tangent directions throughout the full square, and an absolute SNR change below **0.05 dB** throughout this region in the stored 1000-atmosphere response model.

The Hessian is computed in 1260 real coefficient coordinates after removing frame-wise amplitude and phase redundancies. At a relative eigenvalue cutoff of 1e-10, the CSD metric resolves 1260 directions for A, B, and D and 1259 for C. The cutoff specifies numerical resolution.

To recompute the continuous certificates:

```bash
python experiments/local_manifolds/certify_native_geometry.py
python experiments/local_manifolds/certify_native_snr.py
```

The exact-rational SNR certification takes approximately one hour on the tested workstation. Plotting uses the archived receiver powers and does not require the full propagation-response banks.

## Run wave-optics simulations

A short integration check retains the 512×512 grid, ten pseudo-modes, and a batch of sixteen atmospheres, while reducing the number of starts and updates:

```bash
python multistart.py --links A --starts 1 --steps 2 --screening-samples 16 --validation-samples 16 --output work/integration_check
```

The M=10 GSM and 64-HG multistart calculation uses:

```bash
python multistart.py --links A B C D --starts 1024 --steps 300 --batch-size 16 --frames 10 --screening-samples 1000 --validation-samples 1000 --validation-batch 16 --pcb-lr 0.04 --opms-lr 0.02 --seed 2026092401 --output work/multistart
```

`--screening-samples` specifies the independent ensemble used for every endpoint. The separate `--validation-samples` option evaluates the selected class winner. Fig. 3 uses the endpoint ensemble. Keep Links A–D in this order to reproduce the random streams. Repeating a command resumes checkpoints; use a fresh output directory when changing parameters.

The 9-HG calculations and M=3 GSM/64-HG calculations use:

```bash
python basis_size_multistart.py --frames 3 --links A B C D --output work/hg9_m3
python basis_size_multistart.py --frames 10 --links A B C D --output work/hg9_m10
python m3_fig3_extension.py
```

The path and local-manifold calculation entry points are:

```bash
python experiments/paths/native_paths.py
python experiments/local_manifolds/run_native_local.py
```

These commands reuse completed results when available. For a full recomputation, use a separate clone and move the corresponding archived per-link results aside. The local-manifold calculation regenerates the full atmospheric response banks, which require several GB of disk space. Further commands for the landscape, CSD distances, and supplementary simulations are documented in [REPRODUCTION.md](REPRODUCTION.md).

## Code and data layout

| File or directory | Purpose |
|---|---|
| `propagation.py` | Modified von Kármán phase screens, subharmonics, split-step propagation, and aperture statistics |
| `sources.py` | Constrained GSM parameters, HG recurrence, and OPMS synthesis |
| `multistart.py` | Sobol initialization, Adam optimization, and endpoint evaluation |
| `basis_size_multistart.py` | 9-HG experiments at M=3 and M=10 |
| `m3_fig3_extension.py` | GSM and 64-HG experiments at M=3 |
| `landscape.py` | Collimated GSM landscape and Adam trajectories |
| `recompute_csd.py` | CSD distances from saved source parameters |
| `reproduce.py` | Figures, statistics, and integrity checks |
| `data/multistart/` | M=10 GSM and 64-HG endpoint states, powers, and metrics |
| `data/fig3_six_conditions/` | Six-condition summary tables and 9-HG/M=3 endpoint and power archives |
| `data/csd/` | Pairwise CSD distances and selected endpoint identifiers |
| `data/controls/` | Structured-beam, frame-count, and HG/LG comparison tables |
| `experiments/paths/` | Selected endpoint pairs, Bézier parameters, validation powers, and path calculations |
| `experiments/local_manifolds/` | Local source data, Hessian calculations, powers, and continuous certificates |
| `logs/` | Supplementary experiment configurations, source states, and receiver powers |
| `data_manifest.json` | SHA-256 checksums and file sizes |
| `results/` | Generated figures and numerical summaries |

Each endpoint power archive contains `modal_flux` with shape `(1024,1000,M)`: optimization endpoint, atmospheric realization, and pseudo-mode. Endpoint identifiers are zero-based, and the archives include all endpoints. The parameter archives also contain optimization histories.

GSM generates an independent M-frame sequence for each atmospheric realization; OPMS keeps its optimized sequence fixed during validation. All M frames within an integration window share the same atmosphere. Both source types use the same transmitter aperture and frame-wise power normalization. See [Data_conventions.md](Data_conventions.md) for details.

For example, the M=10 OPMS SNR for Link A can be recomputed as follows:

```python
import numpy as np

with np.load("data/multistart/A_opms_flux.npz") as archive:
    power = archive["modal_flux"].astype(np.float64).mean(axis=-1)
mean_power = power.mean(axis=-1)
si = power.var(axis=-1) / mean_power**2
snr_db = -5 * np.log10(0.1 / mean_power + si)
```

Powers are pooled before estimating their mean and variance. Averaging SNR estimates from smaller batches produces a different estimator.

## License and citation

Source code is licensed under [Apache-2.0](LICENSE). Research data and figures are licensed under [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/). Give credit to Minghao Wang and Qianyi Zhang, link to this repository and the license, and indicate any changes. See [LICENSING.md](LICENSING.md) for scope and [CITATION.cff](CITATION.cff) and [data/CITATION.cff](data/CITATION.cff) for citation metadata.
