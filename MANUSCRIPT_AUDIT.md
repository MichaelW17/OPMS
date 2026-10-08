# Manuscript-to-data audit

Reference manuscript: `1007.docx`, inspected on 8 October 2026. The manuscript itself is not bundled.

## Success-reference mismatch

The methods paragraph defines the reference as the mean SNR of the highest **100** validated endpoints among 1024 starts. The reported percentages and the Fig. 4 caption instead match the highest **16** endpoints. Both definitions are exposed in `reproduce.py`; no saved scores or source parameters were modified.

| Link | GSM, top-16 reference | OPMS, top-16 reference | GSM, top-100 reference | OPMS, top-100 reference |
|---|---:|---:|---:|---:|
| A | 509/1024 = 49.71% | 925/1024 = 90.33% | 511/1024 = 49.90% | 975/1024 = 95.21% |
| B | 831/1024 = 81.15% | 1024/1024 = 100% | 832/1024 = 81.25% | 1024/1024 = 100% |
| C | 165/1024 = 16.11% | 1012/1024 = 98.83% | 165/1024 = 16.11% | 1023/1024 = 99.90% |
| D | 724/1024 = 70.70% | 1024/1024 = 100% | 725/1024 = 70.80% | 1024/1024 = 100% |

The default `--reference-count 16` matches the numbers currently reported by the manuscript, including 831 GSM endpoints in the link-B CSD panel. `--reference-count 100` follows the written methods paragraph. Both compare all 1024 endpoints to a class-specific reference minus 0.5 dB. Neither uses a separate calibration subset. The CSD archive stores the broader top-100 selection; the default plot extracts the corresponding top-16 subset using the saved endpoint identifiers.

Before final publication, choose one definition and align the methods, percentages, captions and repository invocation. The release does not edit the paper.

## Atmospheric sample-count mismatch

The draft describes 32,000 atmospheric realizations per endpoint (2000 batches of 16). Every archived multistart power tensor has shape `(1000, 10)`, and the experiment configuration sets both screening and additional winner validation to 1000. The archive contains no evidence that the endpoint distributions were computed from 32,000 realizations.

The reported endpoint medians and standard deviations match the archived **1000-window pooled-power estimator**. Increasing the sample count or averaging batchwise SNR estimates would change the estimator or random sample, and cannot be presented as a reproduction of these saved numbers.

## OPMS parameter count

The implementation optimizes 640 amplitude coordinates, 630 relative-phase coordinates, and 10 frame-specific waists: 1280 real trainable coordinates for ten frames and an 8×8 HG basis. The first phase in each frame is fixed to eliminate a global phase. Thus the manuscript's 1280-parameter total agrees with the code, but describing all 1280 as modal coefficients omits the optimized waist parameters.

## Configuration metadata

The original configuration's free-text selection description mentions 128 screening samples. Its actual argument value, recorded power shapes and metrics use 1000. The archival JSON is retained with local absolute paths removed; the new simulation entry point writes its selection description from its actual arguments.

## Verification completed for this release

- Recomputed all 8192 endpoint SNRs directly from raw receiver powers; maximum absolute discrepancy was below 1e-12 dB.
- Matched the original and released 512×512 numerical kernels on CUDA: OPMS fields, loss, gradients, independently paired GSM fields and both Sobol sequences agreed exactly in the tested cases.
- Completed one GSM and one OPMS trajectory with two Adam updates, a joint batch of sixteen and sixteen-window screening/final validation.
- Recomputed two GSM and two OPMS CSD endpoints on link A and compared them with the corresponding stored matrices.
- Rebuilt all six figure artifacts from the archived data.

These checks verify the data-processing and integration paths. They do not constitute a new 8192-trajectory simulation or a new 32,000-window validation campaign.
