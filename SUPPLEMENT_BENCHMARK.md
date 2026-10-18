# Supplementary benchmark definitions

The three supplementary links are L1 (`small_rx`), L2 (`long_rx20`) and L3 (`strong_small_rx`). These are distinct from the main multistart link identifiers A–D. All use wavelength 850 nm, transmitter radius 50 mm, launch power 1000 in the numerical convention, and noise coefficient 0.2. Their distance, turbulence strength and receiver radius are:

| Link | Distance | Cn² (m⁻²ᐟ³) | Receiver radius |
|---|---:|---:|---:|
| L1 | 1 km | 1e-14 | 10 mm |
| L2 | 2 km | 1e-14 | 20 mm |
| L3 | 1 km | 5e-14 | 10 mm |

## Source expressions and symbols

The following expressions specify unnormalized source-plane fields before transmitter-aperture truncation and per-field launch-power normalization. The radial and azimuthal coordinates are r and φ, k=2π/λ is the optical wavenumber, L is the fixed link length, and R_t is the transmitter radius.

### Sharp source (Supplement S12)

```latex
v_{\mathrm{Sharp}}(r)=\operatorname{Ai}\left(\frac{r_0-r}{\omega}\right)
\exp\left[\alpha_{\mathrm{A}}\frac{r_0-r}{\omega}\right]
\exp\left(-\frac{i k\kappa r^2}{2L}\right).
```

Ai is the Airy function of the first kind, ω is the radial scale, r₀ locates its zero argument, and α_A is the dimensionless exponential apodization. The dimensionless curvature κ controls the quadratic phase, with κ=0 for no added phase and κ=1 for nominal focusing at the receiver distance. The underlying implementation uses the candidate key `alpha` for α_A and `normalized_curvature` for κ.

### Pin source (Supplement S13)

```latex
v_{\mathrm{Pin}}(r)=\exp\left(-\frac{r^2}{w_e^2}\right)
\exp\left(-i\frac{4}{3}k\sqrt{\beta_{\mathrm{Pin}}}\,r^{3/2}\right),
\qquad \beta_{\mathrm{Pin}}=\frac{R_t}{4z_d^2}.
```

w_e is the Gaussian-envelope radius. The design distance z_d determines the phase profile and is varied during source selection; it need not equal the physical propagation distance L. The original Pin grid uses z_d/L in {0.5,1,1.5,2,3}. β_Pin is a radial phase coefficient, not the transverse wavenumber β used for DAD.

### Inverted pin source (Supplement S14)

```latex
v_{\mathrm{IOPB}}(r)=\exp\left(-\frac{r^2}{w_e^2}\right)
J_0\left(k C_\gamma r^\gamma\right),
\qquad C_\gamma=\frac{R_t^{2-\gamma}}{\gamma z_d}.
```

J₀ is the Bessel function of the first kind of order zero; γ is the dimensionless radial exponent. C_γ sets the radial argument. w_e, R_t and z_d have the same meanings as above. The source search also includes the boundary extension implemented in `structured_boundary_extension.py`.

### DAD components and dwell weights (Supplement S15–S16)

The archived **51-mode calculation** uses:

```latex
v_\ell(r,\varphi)=\exp\left(-\frac{r^2}{w_e^2}\right)
J_\ell(\beta r)\exp(i\ell\varphi),\qquad \ell=-25,\ldots,25.

\lambda_\ell=
\frac{1+(-1)^{|\ell|}\alpha_{\mathrm{D}}}
{\sum_{j=-25}^{25}[1+(-1)^{|j|}\alpha_{\mathrm{D}}]},
\qquad \sum_{\ell=-25}^{25}\lambda_\ell=1.

P_s=\sum_{\ell=-25}^{25}\lambda_\ell F_{s\ell}.
```

J_ℓ is the Bessel function of the first kind of integer order ℓ, β is the transverse wavenumber, and w_e is the Gaussian-envelope radius. λ_ℓ is the normalized **intensity/dwell-duration weight**, not a coherent complex-amplitude coefficient. F_sℓ is the aperture power of normalized component ℓ after propagation through atmosphere s. The contrast α_D controls the relative even/odd-order weights. Its searched values are −0.25, −0.5, −0.75, −1 for dark and 0.25, 0.5, 0.75, 1 for antidark.

The archived DAD-51 generator does **not** multiply these fields by an additional optimized curvature phase. A working Supplement expression containing such a phase must use κ=0 to describe these data. Optimizing DAD-51 curvature requires a new experiment; the existing code and records are preserved as calculated.

Both branches were searched separately. Their previously selected parameters are frozen for the updated validation: each branch is evaluated on the same 2000 atmospheric realizations as the original pooled benchmark, using the recorded common seed and processing batch size four. Modal powers are first combined using the dwell weights; all 2000 integrated powers are then pooled to compute one SNR. Percentile bootstrap resampling of these powers, with 2000 resamples, gives the 95% interval. The plotted branch is the one with the higher pooled validated SNR within each link. The reported interval belongs to that branch and is not adjusted for branch selection.

The current `data/controls/dad_m51_branches.csv` contains the updated pooled-power estimates and branch choices. The saved `logs/ol_dad_m51_pooled_validation/*_powers.npz` files contain every modal power, the dwell weights and all integrated powers. The older repetition-averaged Student-t estimates remain in `logs/ol_dad_sms_m51` for source-selection provenance, but are not plotted in the current benchmark.

Suggested Supplement description: For DAD, 51 angular components with orders from −25 to 25 were used. The dark and antidark branches were searched separately, and the branch yielding the higher validated SNR was reported for each link.

## Weighted pure HG control

wHG optimizes nonnegative dwell weights over pure HG modes, a common waist and a common curvature. OPMS optimizes the complex HG superposition within every equal-duration pseudo-mode. The first 64 HG index pairs are enumerated in total-order sequence by `first_hg_indices`; they are not a complete 66-function total-order-10 set or the 8×8 tensor-product set used to synthesize OPMS fields.

| Link | OPMS M=10 | wHG M=10 | wHG M=64 |
|---|---:|---:|---:|
| L1 | 10.8481 dB | 10.5142 dB | 10.6470 dB |
| L2 | 7.5727 dB | 7.3213 dB | 7.3904 dB |
| L3 | 6.8787 dB | 6.3736 dB | 7.2301 dB |

At the same ten-field budget, OPMS gives higher validated SNRs on all three links. Even after wHG increases to 64 modes, ten-frame OPMS gives higher point estimates on L1 and L2. The 64-mode sequence requires 6.4 times the displayed-field budget: at fixed integration time it requires a proportionally higher update rate, or at fixed update rate a longer integration time. This is a frame-budget comparison, not a measured hardware-cost estimate. The table does not claim OPMS superiority on L3 at M=64 or statistical significance of the small between-source differences.

Each wHG holdout estimate uses 2000 independent atmospheres and 2000 percentile-bootstrap resamples. The M=64 search uses eight starts, 300 Adam steps, a 128-atmosphere selection set, and the separate holdout set. The new M=64 run uses its recorded random streams; it is not a paired-bootstrap comparison with the earlier OPMS data. All values retain the manuscript convention `10 log10(1/sqrt(Λe/mean_power+SI))`.

`data/controls/weighted_hg_comparison.csv` and the saved model/receiver-power tensors under `logs/ol_weighted_hg_*` provide the corresponding numerical records. wHG is discussed in text rather than included in the structured-beam plot.

## Propagation-step sensitivity

`data/controls/propagation_step_sensitivity.csv` records all 72 combinations of three links, two source families, M=1,2,4,10 and K=5,7,10. Each M has its own archived optimized source. That source is held fixed while K changes. Every estimate uses 16 independent repetitions of 1000 atmospheres, with a 95% Student-t interval on repetition-level absolute SNR. Batch size is eight, giving 125 processing batches per repetition. These intervals differ from the pooled-power bootstrap intervals used in the structured-beam benchmark.
