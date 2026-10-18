# Data conventions

Updated on 18 October 2026. The original audit concerned the 1007 manuscript draft. Its inconsistencies are recorded below for traceability; the current reproduction defaults use top-100 selection. Manuscript files are not bundled.

## Current success criterion

For each source class and link, average the highest 100 validated endpoint SNRs among 1024 starts. An endpoint is successful if its SNR is at least this reference minus 0.5 dB. `reproduce.py` defaults to this criterion. The CSD archive contains the corresponding endpoint selection. No raw scores or source parameters were changed.

The earlier draft used top-16 classifications in some reported results. The retained `--reference-count 16` option reproduces those historical results:

| Link | GSM, top-16 reference | OPMS, top-16 reference | GSM, top-100 reference | OPMS, top-100 reference |
|---|---:|---:|---:|---:|
| A | 509/1024 = 49.71% | 925/1024 = 90.33% | 511/1024 = 49.90% | 975/1024 = 95.21% |
| B | 831/1024 = 81.15% | 1024/1024 = 100% | 832/1024 = 81.25% | 1024/1024 = 100% |
| C | 165/1024 = 16.11% | 1012/1024 = 98.83% | 165/1024 = 16.11% | 1023/1024 = 99.90% |
| D | 724/1024 = 70.70% | 1024/1024 = 100% | 725/1024 = 70.80% | 1024/1024 = 100% |

## What one atmospheric realization means

Each endpoint has a receiver-power array of shape `(1000, 10)`. Its rows represent 1000 independent atmospheres; its columns represent ten pseudo-mode frames. All ten frames in a row propagate through the same set of five turbulence phase screens. Thus each endpoint uses 1000 atmospheres, not 10,000. Five propagation screens per atmosphere also do not multiply the number of independent atmospheres.

First average the ten frame powers within each atmosphere. Then compute the mean power and scintillation index over the 1000 atmosphere-averaged powers, and evaluate the nonlinear SNR expression once. This is what the original audit called a pooled-power estimate. It does not mean averaging SNR estimates from separate processing batches.

The main multistart validation uses a batch size of 16: 62 batches of 16 realizations followed by 8 realizations. Batch size controls processing only; it does not multiply the total sample count. Screening evaluates every endpoint on 1000 atmospheres. The selected class winner receives an additional, separate 1000-atmosphere evaluation. These two evaluations must not be conflated with the ten source frames.

The 1007 draft stated 32,000 atmospheres per endpoint. This does not describe the archived endpoint arrays. The data and reproduction commands use 1000.

## OPMS parameter count

The implementation optimizes 640 amplitude coordinates, 630 relative-phase coordinates, and ten frame-specific waists: 1280 real trainable coordinates for ten frames and an 8×8 HG basis. The first phase in each frame is fixed to eliminate a global phase.


## Current supplementary benchmark

The structured-beam figure now excludes wHG and replaces its older DAD points with the archived 51-mode branch results. DAD uses ten repetitions of 200 atmospheres and Student-t intervals; the other structured-beam points retain their original 2000-atmosphere pooled estimates and bootstrap intervals. The winning DAD branch is dark for L1/L3 and antidark for L2. Both branches and the selection are exported in `data/controls/dad_m51_branches.csv`.

wHG M=10 gives 10.5142, 7.3213 and 6.3736 dB; M=64 gives 10.6470, 7.3904 and 7.2301 dB on L1–L3. These values are extracted from the archived holdout records. Earlier manuscript text giving 7.36 and 6.12 dB for the L2/L3 ten-mode wHG control does not match these records and is superseded.

The M=10 OPMS structured-benchmark table gives 10.8481, 7.5727 and 6.8787 dB. Do not substitute the repetition-averaged frame-budget results (10.7988, 7.6711 and 6.9934 dB) for these pooled-power figure values. Source budgets and validation estimators must be identified when comparing different controls.

The archived DAD-51 generator applies the Gaussian envelope and angular Bessel field without an additional optimized quadratic phase. The quadratic-phase DAD expression in a working Supplement draft is therefore not a description of these archived DAD-51 runs. The release documents this explicitly rather than changing the data or silently adding focusing to the reproduction code. A focused DAD-51 control would require a new search and validation experiment.
