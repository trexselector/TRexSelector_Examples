# Demo 09: Achieved FDR and TPR vs. Target FDR

## Purpose

The Terminating Random Experiments (T-Rex) selector controls the false discovery rate (FDR) at a target level
$\mathrm{tFDR}$ that the user chooses. This demo sweeps $\mathrm{tFDR}$ and records the achieved FDR, the mean false
discovery proportion over the trials, and the true positive rate (TPR) for three base solvers and four
signal-to-noise ratios (SNR). FDR control shows as an FDR curve on or below the diagonal $\mathrm{FDR} = \mathrm{tFDR}$,
and the TPR curve shows how much power each level of control costs.

The demo mirrors the
[C++ demo 09](../../../../../cpp/trex_selector_methods/ad/trex_core/demo_trex_09_mc_sim_target_fdr/README.md).

---

## Data Generation Parameters

The data follow [Demo 03](../demo_trex_03_mc_sim_variable_support/README.md).

- $n = 300$ observations and $p = 1{,}000$ predictors, Normal predictors and Normal noise.
- A true support of $s = 10$ variables, redrawn in every trial with `set.seed(seed + 500000L)` and
  `sample.int()`, with unit coefficients $\beta_j = 1$.
- $\mathrm{SNR} \in \lbrace0.5, 1, 2, 5\rbrace$, calibrated as in all demos of this suite.

---

## Selector

The selector follows [Demo 05](../demo_trex_05_mc_sim_dummy_distributions/README.md) with Normal dummies.

```text
K                    = 20         # random experiments
max_dummy_multiplier = 10         # cap on L
dummy_distribution   = Normal
lloop_strategy       = STANDARD   # fresh dummies in every L-loop iteration
use_max_T_stop       = TRUE
stagnation stop      = AUTO       # off for TLARS, on for TOMP and TAFS
rho_afs              = 0.3        # TAFS only
```

- **Base solvers**: TLARS, TOMP and TAFS ($\rho = 0.3$).
- **Target FDR grid**: $\mathrm{tFDR} \in \lbrace0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5\rbrace$.

---

## Monte Carlo Design

- 200 Monte Carlo (MC) trials per (solver, SNR, tFDR) cell, 132 cells and 26,400 selector runs in total.
- Trial $m$ of an SNR uses the data seed $24 + 1000 \cdot i_{\mathrm{SNR}} + m$ for every tFDR and every solver, so
  the curves along tFDR compare the same data sets and differ only through the target level.
- The selector is seeded with the trial seed (`seeded_selector = TRUE` in `.run_mc_trex()`), so every cell is
  reproducible.
- Every finished cell is cached in `simulation_results/cache/`, and a stopped run resumes at the first missing cell.
  A completed run removes the cache.

The R and C++ random number generators differ, so the two languages draw different data sets from the same
seeds and agree statistically, not bit for bit.

---

## Output Files

Written to `simulation_results/data/`:

- `demo_trex_09_mc_sim_target_fdr.txt` — one table per solver, with an FDR and a TPR row per SNR and one column per
  tFDR.
- `demo_trex_09_mc_sim_target_fdr.csv` — tidy long table with the columns `solver,snr,tfdr,metric,mean,sd`. The
  metrics are `FDR`, `TPR`, `AvgL` (mean calibrated dummy multiplier) and `AvgT` (mean stopping time), and `sd` is
  filled for FDR and TPR.

`./generate_plots.sh` renders `simulation_results/plots/demo_trex_09_mc_sim_target_fdr.{png,pdf}` with the `sweep`
mode of [`../trex_plt_utils.R`](../trex_plt_utils.R). The top row shows the achieved FDR against tFDR with the
diagonal, and the bottom row shows the TPR, one column per solver and one line per SNR.

---

## Running

```bash
Rscript R/trex_selector_methods/ad/trex_core/demo_trex_09_mc_sim_target_fdr/demo_trex_09_mc_sim_target_fdr.R
Rscript R/trex_selector_methods/ad/trex_core/demo_trex_09_mc_sim_target_fdr/demo_trex_09_mc_sim_target_fdr.R 8 2
R/trex_selector_methods/ad/trex_core/demo_trex_09_mc_sim_target_fdr/generate_plots.sh
```

The optional arguments are the number of worker processes and the number of trials per cell. Without them the demo
runs 200 trials per cell on `TREX_NUM_WORKERS` spawned parallel socket (PSOCK) workers (default 8), and `8 2` is a
smoke run with 2 trials. The script sources `../../../mc_runtime.R`, `../../../support_generators.R`,
`../../../simulation_utils.R` and `../trex_sim_utils.R`.

---

## Simulation Results

The full run is pending.
