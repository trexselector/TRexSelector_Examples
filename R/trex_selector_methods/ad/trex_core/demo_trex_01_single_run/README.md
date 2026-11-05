# Demo 01: Single-Run Classical T-Rex Selector

## Purpose

Demonstrate basic Terminating Random Experiments (T-Rex) selector usage and its output format on a single random
draw, using the TLARS base solver. The demo runs two contrasting scenarios to show
the reporting of the internal T-Rex diagnostics.

## Data-generating process

`dgp_gauss_snr` builds an independent and identically distributed (i.i.d.) Gaussian design `X ~ N(0,1)` and
`y = X beta + eps` with noise calibrated to a target signal-to-noise ratio (SNR). Fixed true support
`TRUE_SUPPORT_P1 = c(28, 150, 44, 129, 43, 5)` (1-based, while the C++ 0-based set is
`{27, 149, 43, 128, 42, 4}`), fixed coefficients `beta_j = 1` (`amplitude = 1`),
`SNR = 1.0`, `tFDR = 0.1`, data seed `58`, selector seed `42`. Two dimension
settings:

- High-dimensional (`p > n`): `n = 150`, `p = 300` — run first (as in C++).
- Low-dimensional (`n > p`): `n = 5000`, `p = 1000` — run second.

The design columns are drawn independently, so the design carries no correlation
structure — the empirical column-correlation matrix is essentially the identity:

![Design-matrix correlation of the shared i.i.d. Gaussian DGP (natural order, no clustering)](../dgp_corr_plots/trex_gauss.png)

## Control

`trex_control(solver = "TLARS", K = 20, max_dummy_multiplier = 10,
use_max_T_stop = TRUE, opt_threshold = 0.75, lloop_strategy = "STANDARD",
tloop_stagnation_stop = FALSE, tloop_max_stagnant_steps = 5, parallel_rnd_experiments = FALSE,
use_memory_mapping = FALSE, lambda2 = 0.1, rho_afs = 0, ncgmp_variant = 0)` —
every option explicit, taken from the C++ reference.

## What it computes

- Part 1 (`run_part_01`): one `TRexSelector$new(...)$select()` per dimension setting.
- `.print_single_run` reports the selected indices, the true positive proportion (TPP), the false discovery
  proportion (FDP), and the internal T-Rex statistics `phi_prime` and `phi_mat` (printed sparsely), `fdp_hat_mat`,
  `r_mat`, and the voting grid.

This is not a Monte Carlo study. Results are from a single seed.

## Running

```bash
Rscript R/trex_selector_methods/ad/trex_core/demo_trex_01_single_run/demo_trex_01_single_run.R
```

The output goes to the console and, for each dimension setting, to
`simulation_results/data/d01_trex_basic_n{n}_p{p}.txt`. The script sources the shared
`../../../support_generators.R` and `../trex_sim_utils.R`.
