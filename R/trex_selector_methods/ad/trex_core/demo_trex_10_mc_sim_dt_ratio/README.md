# Demo 10: FDR and TPR vs. the Sample-Size Ratio n / (s ln p)

## Purpose

Sparse recovery needs the number of observations $n$ to grow like $s \log p$, with $s$ true variables among $p$
predictors (the Donoho–Tanner scaling). This demo sets

```math
n = \operatorname{round}\!\left(c \cdot s \ln p\right)
```

for a grid of ratios $c$ and two numbers of predictors, and records the false discovery rate (FDR) and the true
positive rate (TPR) of the Terminating Random Experiments (T-Rex) selector at a target FDR of $0.1$. Curves of the two
$p$ that fall onto each other on the $c$ axis confirm that $c$ governs the power of the FDR-controlled selection.

The demo mirrors the
[C++ demo 10](../../../../../cpp/trex_selector_methods/ad/trex_core/demo_trex_10_mc_sim_dt_ratio/README.md).
[Demo 08](../demo_trex_08_mc_sim_scalability/README.md) already shows the drop of the TPR between $c \approx 2$ and
$c \approx 8$ at a signal-to-noise ratio (SNR) of 1.

---

## Data Generation Parameters

The data follow [Demo 03](../demo_trex_03_mc_sim_variable_support/README.md) except for $n$.

- $p \in \lbrace1{,}000, 10{,}000\rbrace$ predictors, Normal predictors and Normal noise.
- A true support of $s = 10$ variables, redrawn in every trial with `set.seed(seed + 500000L)` and
  `sample.int()`, with unit coefficients $\beta_j = 1$.
- $c \in \lbrace1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10\rbrace$, so $n$ runs from 69 to 691 at $p = 1{,}000$ and from 92 to
  921 at $p = 10{,}000$.
- $\mathrm{SNR} \in \lbrace0.5, 1, 2\rbrace$.

---

## Selector

The selector follows [Demo 05](../demo_trex_05_mc_sim_dummy_distributions/README.md) with Normal dummies, at a target
FDR of $0.1$.

```text
tFDR                 = 0.1
K                    = 20         # random experiments
max_dummy_multiplier = 10         # cap on L
dummy_distribution   = Normal
lloop_strategy       = STANDARD   # fresh dummies in every L-loop iteration
use_max_T_stop       = TRUE
stagnation stop      = AUTO       # off for TLARS, on for TOMP and TAFS
rho_afs              = 0.3        # TAFS only
```

The base solvers are TLARS, TOMP and TAFS ($\rho = 0.3$).

---

## Monte Carlo Design

- 200 Monte Carlo (MC) trials per (solver, $p$, $c$, SNR) cell, 180 cells and 36,000 selector runs in total.
- Trial $m$ of a cell uses the data seed $24 + 1000 \cdot i_{\mathrm{SNR}} + 100{,}000 \cdot i_c + 10^7 \cdot i_p + m$
  for every solver, so the solvers see the same data sets.
- The selector is seeded with the trial seed (`seeded_selector = TRUE` in `.run_mc_trex()`), so every cell is
  reproducible.
- Every finished cell is cached in `simulation_results/cache/`, and a stopped run resumes at the first missing cell.
  A completed run removes the cache.

The R and C++ random number generators differ, so the two languages draw different data sets from the same
seeds and agree statistically, not bit for bit.

---

## Output Files

Written to `simulation_results/data/`:

- `demo_trex_10_mc_sim_dt_ratio.txt` — one table per solver and SNR, with the sample sizes and an FDR and a TPR row
  per $p$, one column per $c$.
- `demo_trex_10_mc_sim_dt_ratio.csv` — tidy long table with the columns `solver,snr,p,n,c,metric,mean,sd`. The
  metrics are `FDR`, `TPR`, `AvgL` (mean calibrated dummy multiplier) and `AvgT` (mean stopping time), and `sd` is
  filled for FDR and TPR.

`./generate_plots.sh` renders one figure per SNR,
`simulation_results/plots/demo_trex_10_mc_sim_dt_ratio_snr*.{png,pdf}`, with the `sweep` mode of
[`../trex_plt_utils.R`](../trex_plt_utils.R). The top row shows the FDR against $c$ with the target line, and the
bottom row shows the TPR, one column per solver and one line per $p$.

---

## Running

```bash
Rscript R/trex_selector_methods/ad/trex_core/demo_trex_10_mc_sim_dt_ratio/demo_trex_10_mc_sim_dt_ratio.R
Rscript R/trex_selector_methods/ad/trex_core/demo_trex_10_mc_sim_dt_ratio/demo_trex_10_mc_sim_dt_ratio.R 8 2
R/trex_selector_methods/ad/trex_core/demo_trex_10_mc_sim_dt_ratio/generate_plots.sh
```

The optional arguments are the number of worker processes and the number of trials per cell. Without them the demo
runs 200 trials per cell on `TREX_NUM_WORKERS` spawned parallel socket (PSOCK) workers (default 8), and `8 2` is a
smoke run with 2 trials. The script sources `../../../mc_runtime.R`, `../../../support_generators.R`,
`../../../simulation_utils.R` and `../trex_sim_utils.R`.

---

## Simulation Results

The full run is pending.
