# Demo 06: Solver Backends for Screen-TRex — TLARS, TAFS, TOMP (Python)

## Purpose

The demo asks whether the **choice of Terminating Random Experiments (T-Rex) solver backend** changes screening
performance, or whether it
 is merely a speed knob.
 The same independent and identically distributed (i.i.d.) Gaussian design as
 [Demo 01](../demo_trex_scr_01_mc_sim_screen_trex/README.md) is used, so the numbers here are directly comparable to
 the baseline established there.
 Three backends are swept over the signal-to-noise ratio (SNR). They are **TLARS**, the terminating form of
 least-angle regression (LARS), **TAFS** with $\rho = 0.3$ (terminating adaptive forward stepwise) and **TOMP**
 (terminating orthogonal matching pursuit), first under the Ordinary thresholding rule alone and then crossed with
 the Bootstrap-CI rule, whose threshold comes from a bootstrap confidence interval (CI).
 Screening returns a *candidate set*, and the false discovery rate (FDR) and the true positive rate (TPR) are
 evaluated on the individual selected variables
 (see [What is actually measured](../README.md#what-is-actually-measured-in-these-demos)).

This is the Python port of the C++ demo
[demo_trex_scr_06_mc_sim_solvers](../../../../../cpp/trex_selector_methods/ad/trex_screening/demo_trex_scr_06_mc_sim_solvers/README.md),
using `trexselector`'s `TRexScreeningSelector`.

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
- $\sigma_{\varepsilon}^2$ is the noise variance, calibrated to the target SNR,
   $\sigma_{\varepsilon}^2 = \lVert \boldsymbol{X}\boldsymbol{\beta} \rVert_2^2 / (n \cdot \mathrm{SNR})$, and the
   response is centred.
- $n = 300$, $p = 1{,}000$, $s = 10$ (high-dimensional, $p > n$).

The design matrix has **no correlation structure**:

```math
X_{ij} \sim \mathcal{N}(0,1) \quad \text{i.i.d.}
```

- The active support is drawn uniformly at random *per Monte Carlo (MC) trial*, so the results are not tied to
   one support pattern.
- All active coefficients are $\beta_j = 1$, and the support comes from a random number generator (RNG) seeded at
   the trial seed plus `500000`, so it stays apart from the design and noise draws of the trial seed.
- This is the **same data-generating process (DGP) as Demo 01** with the same seeds. Trial `mc` at grid index `si`
   uses seed `24 + 1000 * si + mc`, so every series of both demos screens the **same 200 datasets** at one SNR
   point. Differences between the series at one SNR point therefore come from the solver, the thresholding rule and
   the random experiments, not from the data.

![Design-matrix correlation — i.i.d. Gaussian design (no correlation structure)](../dgp_corr_plots/scr_iid.png)

Empirical Pearson correlation of an exemplary design drawn from this DGP (`dgp_iid_snr`, $n=300$,
$p=1{,}000$, seed 2026), in **natural variable order — no hierarchical clustering**. One figure per DGP lives
in [`../dgp_corr_plots/`](../dgp_corr_plots/). Regenerate the whole suite with
[`../generate_corr_plots.sh`](../generate_corr_plots.sh) (shared plotter
[`../../../corr_plt_utils.py`](../../../corr_plt_utils.py)).

---

## Control Parameters

```text
K = 20                       # Random experiments per T-loop iteration
R_boot = 1000                # Bootstrap replicates (Bootstrap-CI rule only)
ci_grid_step = 0.001         # Bootstrap-CI threshold grid granularity
solver = TLARS / TAFS / TOMP # T-Rex solver backends compared
rho_afs = 0.3                # Adaptive forward stepwise shrinkage step (read by TAFS only)
tloop_stagnation_stop = True # Stagnation stop enabled for every backend here
tloop_max_stagnant_steps = 5 # Stagnant T-loop steps before the stop
MC = 200                     # Monte Carlo repetitions per grid point
```

Note that Screen-TRex has **no target-FDR parameter**, because unlike the classical T-Rex selector, screening
 thresholds the voting statistic instead of calibrating to a user-specified level.

---

## Methods Compared

Three T-Rex solver backends [[1]](#references), all driving `ScreenTRexMethod.TREX`:

- **TLARS** — terminating LARS, the default backend, which follows the least-angle regression path until the
   dummy-based stopping criterion fires.
- **TAFS-0.3** — terminating adaptive forward stepwise with shrinkage step $\rho = 0.3$, which moves the fit only a
   fraction $\rho$ toward the least-squares fit on the active set and so lies between forward stepwise ($\rho = 1$)
   and least-angle regression ($\rho \to 0$).
- **TOMP** — terminating orthogonal matching pursuit, the greediest of the three, taking the predictor most
   correlated with the current residual at each step.

Each backend is combined with the Screen-TRex thresholding rules:

- **Ordinary** — selects $\lbrace j : \Phi_j > 0.5 \rbrace$, a simple majority vote of the random experiments.
- **Bootstrap-CI** — builds a bootstrap confidence band around the estimated FDR curve (`R_boot = 1000`
   replicates) and picks its threshold from that band.

---

## The Two Parts

- **Part 1 — solvers under the Ordinary rule.** Three series (TLARS, TAFS-0.3, TOMP), Ordinary screening
   only, over $\mathrm{SNR} \in \lbrace0.01, 0.1, 0.2, 0.5, 0.6, 1, 2, 5\rbrace$ with 200 MC trials per grid point.
- **Part 2 — solvers crossed with both rules.** The same three backends under both Ordinary and
   Bootstrap-CI thresholding, giving six series over the same SNR grid and MC budget.

Every trial draws a fresh design, support, and noise realization. Per-cell trial seeds follow the C++
source (`base_seed = 24 + snr_idx * 1000`).

---

## Output Files

Written to `simulation_results/data/`:

- `scr_solvers_snr_n300_p1000_s10.txt` / `.csv` — Part 1: FDR, TPR, and estimated FDR per solver and SNR
   level.
- `scr_solver_method_snr_n300_p1000_s10.txt` / `.csv` — Part 2: the same metrics for all six
   solver × rule combinations.

The file names and comma-separated values (CSV) schemas match the C++ demo, and the suite plotter
[`../trex_scr_plt_utils.py`](../trex_scr_plt_utils.py) is the same module the C++ suite uses.

Figures go to `simulation_results/plots/` as Portable Network Graphics (PNG) and Portable Document Format (PDF)
files, produced by `./generate_plots.sh`.

---

## Running the Demo

```bash
python demo_trex_scr_06_mc_sim_solvers.py   # full run (~7 min with 8 workers)
./generate_plots.sh                         # render the figures below from the saved CSVs
```

Monte Carlo trials are parallelized with `ProcessPoolExecutor` over `TREX_NUM_WORKERS` worker processes
(default 8, capped at the usable cores, and `0` uses all of them). The suite's older `SCR_NUM_WORKERS` still
works and takes precedence.

---

## Simulation Results

The numbers and figures below are this Python run's own results (200 MC trials per grid point).
The two ports draw their data from independent RNG streams (NumPy here, the C++ engine there), so they are not
expected to match the
[C++ reference](../../../../../cpp/trex_selector_methods/ad/trex_screening/demo_trex_scr_06_mc_sim_solvers/README.md#simulation-results)
exactly. The tables carry no standard deviations, so the agreement is described by size. The mean absolute
difference to C++ is $0.019$ for the FDR and $0.006$ for the TPR in Part 1, and $0.012$ and $0.015$ in Part 2. From
$\mathrm{SNR} = 0.5$ on the largest differences are the Bootstrap-CI TPR values at $\mathrm{SNR} = 1$, TAFS-0.3 at
$0.509$ against $0.635$ and TLARS at $0.444$ against $0.341$.

### Part 1 — Three Solvers, Ordinary Screening

- **The solver is not a pure speed knob.** At mid SNR the greedier backends are markedly more powerful than
   TLARS at comparable FDR. At $\mathrm{SNR} = 1$ the TPR is $0.938$ (TAFS-0.3) and $0.958$ (TOMP) against
   $0.836$ (TLARS), while the realized FDR is $0.057$, $0.062$, and $0.057$ respectively. The gap is already
   visible at $\mathrm{SNR} = 0.5\text{–}0.6$ ($0.464/0.475$ vs. $0.439$, and $0.620/0.624$ vs. $0.539$).
- **All three converge at high SNR.** By $\mathrm{SNR} = 2$ TOMP reaches TPR $1.000$, TAFS-0.3 $0.9995$ and
   TLARS $0.995$, and at $\mathrm{SNR} = 5$ all three recover every active variable at FDR
   $0.057\text{–}0.059$. The choice of backend therefore only matters where the signal is neither absent nor
   obvious.
- **At very low SNR the backends are indistinguishable, and all uncontrolled.** For $\mathrm{SNR} \le 0.2$
   the realized FDR runs $0.217\text{–}0.473$ for all three with TPR $\le 0.125$, reproducing the Demo 01 finding
   that screening is not FDR-controlling when there is no recoverable signal.
- **The internal FDR estimate behaves the same way for every backend.** It peaks at $\mathrm{SNR} = 0.2$
   ($0.516\text{–}0.530$) and then settles at $0.094$ for all three at $\mathrm{SNR} = 5$, sitting *above* the
   realized FDR from $\mathrm{SNR} = 0.5$ on.

TPR (left) and FDR (right) vs. SNR, one line per solver backend. On the FDR panel the solid line
is the realized FDR and the dashed line the procedure's own estimated FDR.

![Solver comparison under Ordinary screening: TPR/FDR vs SNR](simulation_results/plots/scr_solvers_snr_n300_p1000_s10.png)

### Part 2 — Three Solvers × Two Thresholding Rules

- **Under Bootstrap-CI the backend matters far more than under the Ordinary rule.** At $\mathrm{SNR} = 1$
   the bootstrap TPR spans $0.444$ (TLARS), $0.509$ (TAFS-0.3), and $0.892$ (TOMP) — nearly half the
   achievable power — against a spread of roughly $0.11$ for the same three backends under the Ordinary
   rule ($0.845$ / $0.943$ / $0.956$).
- **TLARS and TAFS-0.3 are the conservative backends under Bootstrap-CI.** Both keep FDR at or below $0.060$ from
   $\mathrm{SNR} = 0.5$ onwards. TLARS gives up the most power around $\mathrm{SNR} = 1$ (TPR $0.44$ vs. $0.51$ for
   TAFS-0.3) but recovers at high SNR, where its TPR reaches $0.905$ at $\mathrm{SNR} = 5$. TOMP under
   Bootstrap-CI, in contrast, behaves almost like Ordinary screening. At $\mathrm{SNR} = 1$ its TPR is
   $0.892$ vs. $0.956$ ordinary, at FDR $0.066$ vs. $0.065$.
- **Bootstrap-CI stops short of full power.** TAFS-0.3 levels off at TPR $0.665\text{–}0.690$ from
   $\mathrm{SNR} = 2$ on and TOMP reaches $0.925\text{–}0.955$, while every Ordinary series reaches $1.000$.
- **The low-SNR regime is unaffected by either choice.** For $\mathrm{SNR} \le 0.2$ all six series have
   TPR $\le 0.122$ and FDR between $0.124$ and $0.471$. Neither the backend nor the thresholding rule
   rescues screening when there is nothing to find.
- **Practical takeaway.** Under the Ordinary rule TOMP and TAFS-0.3 buy substantial power over TLARS at mid SNR
   at a comparable error rate. Under Bootstrap-CI TLARS and TAFS-0.3 keep the lowest FDR, while TOMP keeps most of
   the power. At $\mathrm{SNR} = 1$ the backend moves the Bootstrap-CI TPR by $0.448$, a little more than switching
   TLARS from Ordinary to Bootstrap-CI ($0.845 \to 0.444$), so there the solver matters as much as the rule.

TPR (left) and FDR (right) vs. SNR, one line per solver × rule combination. On the FDR panel the
solid line is the realized FDR and the dashed line the procedure's own estimated FDR.

![Solver × thresholding rule: TPR/FDR vs SNR](simulation_results/plots/scr_solver_method_snr_n300_p1000_s10.png)

---

## References

1. Machkour, J., Muma, M., & Palomar, D. P., "False Discovery Rate Control for Fast Screening of
   Large-Scale Genomics Biobanks.", IEEE Statistical Signal Processing Workshop (SSP), 2023,
    pp. 666–670, IEEE.
    [DOI-Link](https://doi.org/10.1109/SSP53291.2023.10207957)
