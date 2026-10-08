# Experiment reproduction

Run commands from the repository root. The released numerical kernels keep the original physical formulas, floating-point operations, parameter constraints and random-stream policy. The release removes line-by-line narration, adds concise English documentation, formats the code, separates figure reproduction from simulation, and replaces machine-specific output paths. Historical control module and dataset names remain in the source map for traceability. `SMS` is the historical internal name for OPMS; `PCB` denotes the GSM benchmark.

## Figures 1–6 from archived data

```bash
python reproduce.py --verify-flux --verify-checksums
```

This reconstructs the figure contents and numerical results; it is not a pixel-for-pixel reproduction of Word's layout or the author's Visio drawing. Default Fig. 3 and Fig. 4 use top-16 selection to match the reported values. Use `--reference-count 100 --output results/top100` to reproduce the methods paragraph's alternate definition. Fig. 5 combines the archived structured-beam table with the weighted-HG holdout results. Fig. 6 uses the archived frame-budget and matched-basis tables.

## Figure 2: new landscape and trajectories

```bash
python landscape.py --stage all --wg-count 100 --lcr-count 100 --landscape-samples 64 --landscape-batch 8 --runs 10 --steps 400 --learning-rate 0.012 --train-blocks 2 --train-batch 6 --output work/landscape --figure results/new_landscape.png
```

The link is collimated, L=1 km, wavelength 850 nm, Cn²=1e-14, receiver radius 25 mm. Wg and Lcr span 3–180 mm logarithmically. Each grid point uses 64 atmospheres. Adam gradients are obtained from live wave-optics propagation, not interpolated from the grid. The saved 100×100 landscape and saved ten-trajectory dataset came from separate runs and are reused together by `reproduce.py`.

## Figure 3: new multistart optimization

Use the full command in README.md. The main numerical settings are 1024 starts per link/class, 300 Adam updates, 16 joint atmosphere samples, M=10, grid 512×512, spacing 1 mm, five propagation steps, transmitter radius 50 mm, inner scale 3 mm, outer scale 1.024 m and three subharmonic levels.

| Link | L (m) | Cn² | Receiver radius (m) | Wavelength (m) | Λe |
|---|---:|---:|---:|---:|---:|
| A | 1000 | 5e-15 | 0.01 | 632e-9 | 0.1 |
| B | 1000 | 5e-14 | 0.01 | 632e-9 | 0.1 |
| C | 1000 | 1e-14 | 0.01 | 850e-9 | 0.2 |
| D | 2000 | 5e-15 | 0.02 | 850e-9 | 0.2 |

GSM Wg and Lcr are bounded at 3–150 mm and normalized curvature L/Rg at 0–2. OPMS uses an 8×8 HG basis, frame-specific waists bounded at 3–80 mm and fixed normalized curvature 1.0. Source fields are normalized to power 1000 within the transmitter aperture.

Different CUDA/PyTorch versions can produce small numerical changes that accumulate during stochastic optimization. Archived powers and source states provide an exact reference for the published computations. No claim of bitwise equality across GPU architectures is made.

## Figure 4: recompute CSD distances

```bash
python recompute_csd.py --links A B C D --quadrature-pairs 262144 --output work/recomputed_csd
```

This expands the compact source archives into `work/endpoints`. GSM uses deterministic Sobol quadrature over pairs of pixels in the transmitter aperture. OPMS uses the exact finite-field trace identity, with exactly zero pixels outside the aperture removed. The normalized distance is

`sqrt((||Wi||² + ||Wj||² - 2 Re <Wi,Wj>) / (||Wi||² + ||Wj||²))`.

The recomputation command selects the broader top-100 set. `reproduce.py` can subsequently visualize either stored selection by endpoint identifier. For a short integration check:

```bash
python recompute_csd.py --links A --limit 2 --output work/csd_check
```

The full pairwise computation needs several GB of device memory for OPMS fields. `--limit` is only an integration-check setting and does not reproduce the complete pair distribution.

## Figures 5–6: supporting controls

The release includes the actual selected source states, raw validation powers where saved, control configurations, and the original simulation modules. These experiments use three cases named `small_rx`, `long_rx20`, and `strong_small_rx` (S1–S3). Their calculations predate the September joint-sampling multistart experiment and retain their own sampling policies and seeds; they must not be silently replaced with its training configuration.

The following entry points consume the included `logs/` inputs:

```bash
python ol_all_beams_pooled_validation.py --atmospheric-samples 2000 --batch-size 4 --bootstrap-repeats 2000 --force
python ol_weighted_hg_pseudomode_validation_20260901.py --force
python ol_frame_budget_complete.py
python ol_hg_lg_basis_control.py
```

The first command validates selected fixed sources on a common atmospheric pool. The second optimizes and validates weighted pure HG pseudo-modes; use `--help` for available case and sample options. The last two retain completed cached source optimizations and aggregate the supplied frame/basis results. They are supporting research entry points; the release preparation did not rerun their full training campaigns. To regenerate their training, work in a separate copy and move the corresponding completed output directory aside before invoking its entry point. Preserve input directories identified by `REFERENCE_ROOT` and `INPUT_ROOT`.

The canonical reproduction of Fig. 5 and Fig. 6 is the archived-data command above. All archived validation policies and confidence-interval definitions are retained in their JSON records. The release does not claim that these supporting controls share the same protocol as the main multistart experiment.

## Recorded environment and verification

The release checks ran in Python 3.10.13, torch 2.1.0, NumPy 1.26.4, SciPy 1.11.3, Matplotlib 3.10.0, on an NVIDIA RTX 4090 D. Runtime dependencies are pinned in `requirements.txt`. `results/kernel_equivalence.json` records numerical comparison with the research code; `results/raw_power_verification.csv` records all raw-power checks; `results/csd_verification.json` records the CSD subset check. See MANUSCRIPT_AUDIT.md for the limits of what was rerun.
