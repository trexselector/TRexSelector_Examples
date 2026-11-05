# Demo 01: Screen-TRex on an i.i.d. Gaussian Design

## Purpose

The demo establishes the **baseline behaviour of Screen-TRex** [[1]](#references) on the easiest possible design,
independent and identically distributed (i.i.d.) Gaussian predictors with no correlation structure at all.
Two thresholding rules are compared over a signal-to-noise ratio (SNR) sweep. The **Ordinary** rule is a majority vote
($\Phi_j > 0.5$), and the **Bootstrap-CI** rule picks its threshold from a bootstrap confidence interval (CI) band.
Everything the later demos add (correlation, dependency-aware corrections, biobank routing, solver backends) is
measured against the numbers established here.
Screening returns a *candidate set*, and the false discovery rate (FDR) and the true positive rate (TPR) are evaluated
on the individual selected variables (see
[What is actually measured](../../../../../cpp/trex_selector_methods/ad/trex_screening/README.md#what-is-actually-measured-in-these-demos)
in the C++ suite README).

This is the R port of the C++ demo
[demo_trex_scr_01_mc_sim_screen_trex](../../../../../cpp/trex_selector_methods/ad/trex_screening/demo_trex_scr_01_mc_sim_screen_trex/README.md),
using the `TRexScreeningSelector` class of the TRexSelector package. It keeps the data-generating process (DGP), the
control parameters, the sweep grid, the console tables and the result-file naming of the C++ demo. Indices are 1-based
(R convention). This is the in-memory variant, and Demo 02 is the memory-mapped counterpart.

---

## Data Generation Parameters (`dgp_iid_snr`)

We consider the linear model:

```math
\boldsymbol{y} = \boldsymbol{X}\boldsymbol{\beta} + \boldsymbol{\epsilon},
\qquad \boldsymbol{\epsilon} \sim \mathcal{N}(\boldsymbol{0}, \sigma_{\varepsilon}^2 \boldsymbol{I}_n)
```

- $\boldsymbol{y} \in \mathbb{R}^n$ is the response vector.
- $\boldsymbol{X} \in \mathbb{R}^{n \times p}$ is the design matrix.
- $\boldsymbol{\beta} \in \mathbb{R}^p$ is the coefficient vector, with $s$ nonzero entries.
- $\boldsymbol{\epsilon}$ is the noise vector, i.i.d. normal.
- $\sigma_{\varepsilon}^2$ is the noise variance, calibrated to achieve a target linear SNR.
- $n = 300$, $p = 1{,}000$, $s = 10$ (high-dimensional, $p > n$).

The design matrix has **no correlation structure**:

```math
X_{ij} \sim \mathcal{N}(0,1) \quad \text{i.i.d.}
```

- The active support is drawn uniformly at random *per Monte Carlo (MC) trial*, so the results are not tied to one
  support pattern. The support draw uses its own random number generator (RNG) seed, the trial seed plus `500000`, so it
  stays independent of the design and noise draws.
- All active coefficients are $\beta_j = 1$.
- Noise calibration
  $\sigma_{\varepsilon} = \sqrt{\lVert \boldsymbol{X}\boldsymbol{\beta} \rVert^2 / (n \cdot \mathrm{SNR})}$, and the
  response is centred to mean zero.

![Design-matrix correlation — i.i.d. Gaussian design (no correlation structure)](../dgp_corr_plots/scr_iid.png)

Empirical Pearson correlation of an exemplary design drawn from `dgp_iid_snr` ($n=300$, $p=1{,}000$, seed 2026), in
**natural variable order without hierarchical clustering**. Only the unit diagonal stands out, and the off-diagonal
noise is the $O(1/\sqrt{n})$ sampling fluctuation expected of independent columns. One figure per DGP lives in
[`../dgp_corr_plots/`](../dgp_corr_plots/). Regenerate the whole suite with
[`../generate_corr_plots.sh`](../generate_corr_plots.sh), which drives the shared ggplot2 plotter
[`../../../corr_plt_utils.R`](../../../corr_plt_utils.R).

---

## Control Parameters

```text
K = 20                        # Random experiments per T-loop iteration
max_dummy_multiplier = 10     # Passed on, screening pins the dummy count to p
use_max_T_stop = TRUE         # Passed on, screening runs a single T-loop step
lloop_strategy = "STANDARD"   # L-loop strategy
R_boot = 1000                 # Bootstrap replicates (Bootstrap-CI rule only)
ci_grid_step = 0.001          # Bootstrap-CI threshold grid granularity
solver = "TLARS"              # T-Rex solver backend
use_memory_mapping = FALSE    # In-memory variant (Demo 02 flips this)
NUM_MC = 200                  # Monte Carlo repetitions per grid point
```

Screen-TRex has **no target-FDR parameter**. Unlike the classical Terminating Random Experiments (T-Rex) selector,
screening thresholds the voting statistic instead of calibrating to a user-specified level.

---

## Methods Compared

Two Screen-TRex thresholding rules [[1]](#references), both with `trex_method = "TREX"`:

- **Screen-TRex Ordinary: TLARS** selects $\lbrace j : \Phi_j > 0.5 \rbrace$, a simple majority vote of the random
  experiments (`use_bootstrap_CI = FALSE`).
- **Screen-TRex Bootstrap-CI: TLARS** builds a bootstrap confidence band around the estimated FDR curve
  (`R_boot = 1000` replicates) and picks its threshold from that band (`use_bootstrap_CI = TRUE`).

Both report an **estimated FDR** alongside the realized FDR and TPR. Each Monte Carlo trial builds the selector through
the TRexSelector class interface, here for the Bootstrap-CI rule:

```r
library(TRexSelector)

ctrl <- trex_control(
  solver = "TLARS", K = 20L, max_dummy_multiplier = 10L, use_max_T_stop = TRUE,
  lloop_strategy = "STANDARD", use_memory_mapping = FALSE
)
sel <- TRexScreeningSelector$new(
  X, y, seed = -1L, verbose = FALSE,
  screen_control = trex_screen_control(trex_method = "TREX", use_bootstrap_CI = TRUE),
  control = ctrl
)
sel$select()
fdp <- trex_fdp(sel$selected_indices, true_support)
tpp <- trex_tpp(sel$selected_indices, true_support)
est <- sel$estimated_FDR
```

`make_scr_trex_ctrl()` and `run_mc_screen()` in [`../trex_scr_sim_utils.R`](../trex_scr_sim_utils.R) build exactly
this per trial. `R_boot` and `ci_grid_step` stay at the defaults of `trex_screen_control()`, which equal the values the
C++ and Python demos set. The selector seed `-1L` draws the dummies from hardware entropy in every trial.

---

## The Sweep

A single **SNR sweep** over $\mathrm{SNR} \in \lbrace0.01, 0.1, 0.2, 0.5, 0.6, 1, 2, 5\rbrace$, 200 MC trials per
point, with a fresh design, support and noise draw in every trial. The trial seeds carry the same labels as the C++
source. The `si`-th SNR value uses `base_seed = 24 + (si - 1) * 1000`, and trial `mc = 0, ..., 199` uses
`base_seed + mc`. The seeds do not depend on the method, so both rules screen the same 200 datasets at each SNR point
and differ only in their own random draws.

---

## Output Files

Written to `simulation_results/data/`:

- `scr_screen_trex_snr_n300_p1000.txt` / `.csv`, the FDR, the TPR and the estimated FDR per method and SNR level. The
  comma-separated values (CSV) file is a tidy long table with the columns `method,metric,SNR,value`.

The file names and the CSV schema match the C++ and Python demos. Figures in Portable Network Graphics (PNG) and
Portable Document Format (PDF) go to `simulation_results/plots/`, produced by [`generate_plots.sh`](generate_plots.sh).

---

## Running the Demo

```bash
demo=R/trex_selector_methods/ad/trex_screening/demo_trex_scr_01_mc_sim_screen_trex/demo_trex_scr_01_mc_sim_screen_trex.R

Rscript "$demo"      # full run, TREX_NUM_WORKERS workers (default 8)
Rscript "$demo" 8    # 8 worker cores
```

Monte Carlo trials run in parallel on a parallel socket (PSOCK) cluster of spawned R worker processes
([`../../../mc_runtime.R`](../../../mc_runtime.R)). The worker count is the first command-line argument, else
`TREX_NUM_WORKERS`, else 8, and `0` uses every core. Every finished grid cell is appended to a cache in
`simulation_results/cache/`, so a stopped run resumes where it left off, and a completed run removes the cache. The
script also sources [`../trex_scr_sim_utils.R`](../trex_scr_sim_utils.R). For a quick smoke test, reduce `NUM_MC` and
`SNR_GRID` at the top of the script. The recorded run (library commit `44cf1ab`, 8 workers) took about 2 minutes.

Afterwards, regenerate the figure from the CSV file with [`generate_plots.sh`](generate_plots.sh), which drives the
suite's ggplot2 plotting module [`../trex_scr_plt_utils.R`](../trex_scr_plt_utils.R) through `Rscript` (set `RSCRIPT`
to use another R installation).

---

## Simulation Results

The numbers and figure below are this R run's own results (200 MC trials per grid point). R draws its own random
numbers, so they agree with the
[C++ reference](../../../../../cpp/trex_selector_methods/ad/trex_screening/demo_trex_scr_01_mc_sim_screen_trex/README.md#simulation-results)
up to Monte Carlo noise, not exactly. The screening tables store only the 200-trial means, without standard
deviations, so agreement is judged by size. The mean absolute difference from the C++ run is $0.023$ in the FDR,
$0.009$ in the TPR and $0.013$ in the estimated FDR, against $0.022$, $0.015$ and $0.012$ between the Python and C++
runs. Both rules screen the same 200 datasets per SNR point, so a deviation at one point repeats in both series and
stems from one dataset realization.

- **Screening is not FDR-controlling at low SNR.** At $\mathrm{SNR} \le 0.2$ the realized FDR runs between $0.24$
  and $0.48$ for both rules, while the TPR stays at or below $0.115$. Screening has no target level that would hold
  the FDR down when almost no signal can be recovered. At $\mathrm{SNR} = 0.1$ the FDR of both rules lies above the
  C++ run, by $0.118$ for Ordinary ($0.447$ against $0.328$) and $0.068$ for Bootstrap-CI. Rerunning the Ordinary
  rule on the same datasets with fresh dummies reproduces the excess ($0.464$ and $0.458$), while three fresh seed
  bands give $0.367$, $0.404$ and $0.434$, against $0.328$ in C++ and $0.396$ in Python. The datasets of one band
  thus move the FDR by about $0.1$, so the excess is no evidence of a method difference.
- **Both rules become reliable once signal is present.** From $\mathrm{SNR} = 0.5$ the Ordinary rule's FDR drops to
  $0.112$ and settles at $0.056\text{–}0.066$ for $\mathrm{SNR} \ge 1$, while its TPR climbs from $0.445$ to $1.000$.
  At $\mathrm{SNR} = 5$ every trial recovers all ten active variables.
- **Bootstrap-CI is the conservative rule, with lower FDR and clearly less power.** Its FDR stays at or below $0.048$
  for $\mathrm{SNR} \ge 0.5$, less than half the Ordinary rule's up to $\mathrm{SNR} = 1$ ($0.033$ against $0.066$
  at $\mathrm{SNR} = 1$). The price is a TPR of $0.850$ against $1.000$ at $\mathrm{SNR} = 5$ and a much wider gap
  mid-sweep ($0.420$ against $0.815$ at $\mathrm{SNR} = 1$). Choose it when false discoveries are costlier than
  missed ones. It also runs 1,000 bootstrap replicates per trial on top of the voting step.
- **The internal FDR estimate is conservative here.** For $\mathrm{SNR} \ge 0.5$ it sits *above* the realized FDR for
  both rules (for example $0.118$ against $0.066$ for Ordinary at $\mathrm{SNR} = 1$), so it over-states the error
  rate rather than hiding it. At the lowest SNR points it is unreliable in both directions. At $\mathrm{SNR} = 0.01$
  it lies below the realized FDR for both rules ($0.415$ against $0.478$ for Ordinary), and at
  $\mathrm{SNR} = 0.2$ well above it ($0.463$ against $0.286$). Demo 03 shows this same estimate failing far more
  dramatically under correlation.

TPR (left) and FDR (right) against SNR, one line per thresholding rule. On the FDR panel the solid line is the
realized FDR and the dashed line the procedure's own estimated FDR. The dotted line at $0.1$ is a visual reference
only, since screening has no target FDR.

![Screen-TRex on an i.i.d. design: TPR/FDR vs SNR](simulation_results/plots/scr_screen_trex_snr_n300_p1000.png)

---

## References

1. Machkour, J., Muma, M., & Palomar, D. P., "False Discovery Rate Control for Fast Screening of
   Large-Scale Genomics Biobanks.", IEEE Statistical Signal Processing Workshop (SSP), 2023,
    pp. 666–670, IEEE.
    [DOI-Link](https://doi.org/10.1109/SSP53291.2023.10207957)
