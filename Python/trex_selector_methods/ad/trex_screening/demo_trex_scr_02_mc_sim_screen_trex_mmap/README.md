# Demo 02: Screen-TRex with Memory-Mapped Dummy Matrices (Python)

## Purpose

The demo shows that the **memory-mapped Screen-TRex workflow** reproduces the in-memory baseline while keeping the
 dummy (D) matrices on disk instead of in random-access memory (RAM).
The study is the same experiment as [Demo 01](../demo_trex_scr_01_mc_sim_screen_trex/README.md) — same
 independent and identically distributed (i.i.d.) Gaussian design, same signal-to-noise ratio (SNR) sweep, same two
 thresholding rules (**Ordinary** and **Bootstrap-CI**, the latter built on a bootstrap confidence interval (CI)) —
 with the single change `TRexControlParameter.use_memory_mapping = True`.
The point of this demo is therefore *equivalence at a lower memory footprint*, not a different statistical
 regime, because memory mapping is what makes very large $p$ feasible on machines that cannot hold the dummy
 matrices in RAM.
Screening returns a *candidate set*, and the false discovery rate (FDR) and the true positive rate (TPR) are
 evaluated on the individual selected variables
 (see [What is actually measured](../README.md#what-is-actually-measured-in-these-demos)).

This is the Python port of the C++ demo
[demo_trex_scr_02_mc_sim_screen_trex_mmap](../../../../../cpp/trex_selector_methods/ad/trex_screening/demo_trex_scr_02_mc_sim_screen_trex_mmap/README.md),
using `trexselector`'s `TRexScreeningSelector`. As in the C++ source, the design matrix `X` still
lives in RAM. Only the dummy matrices move to disk, and the library manages those temporary files in its
own scratch directories (nothing to clean up in the script).

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

- The active support is drawn uniformly at random *per Monte Carlo (MC) trial*, so the results are not tied to
   one support pattern.
- All active coefficients are $\beta_j = 1$. The support draw uses its own random number generator (RNG) seed,
   the trial seed plus `500000`, so it stays independent of the design and noise draws.

![Design-matrix correlation — i.i.d. Gaussian design (no correlation structure)](../dgp_corr_plots/scr_iid.png)

Empirical Pearson correlation of an exemplary design drawn from this data-generating process (DGP)
`dgp_iid_snr` ($n=300$, $p=1{,}000$, seed 2026), in **natural variable order — no hierarchical clustering**.
The design is identical to Demo 01's, and only the unit diagonal stands out. One figure per DGP lives in
[`../dgp_corr_plots/`](../dgp_corr_plots/). Regenerate the whole suite with
[`../generate_corr_plots.sh`](../generate_corr_plots.sh) (shared plotter
[`../../../corr_plt_utils.py`](../../../corr_plt_utils.py)).

---

## Control Parameters

```text
K = 20                       # Random experiments per T-loop iteration
use_memory_mapping = True    # Dummy (D) matrices backed by on-disk files, not RAM
R_boot = 1000                # Bootstrap replicates (Bootstrap-CI rule only)
ci_grid_step = 0.001         # Bootstrap-CI threshold grid granularity
solver = TLARS               # T-Rex solver backend
MC = 200                     # Monte Carlo repetitions per grid point
```

Note that Screen-TRex has **no target-FDR parameter**, because unlike the classical Terminating Random Experiments
(T-Rex) selector, screening
 thresholds the voting statistic instead of calibrating to a user-specified level.

---

## Methods Compared

Two Screen-TRex thresholding rules [[1]](#references), both using `ScreenTRexMethod.TREX`:

- **Screen-TRex Ordinary: TLARS** — selects $\lbrace j : \Phi_j > 0.5 \rbrace$, a simple majority vote of the
   random experiments (`use_bootstrap_CI = False`).
- **Screen-TRex Bootstrap-CI: TLARS** — builds a bootstrap confidence band around the estimated FDR
   curve (`R_boot = 1000` replicates) and picks its threshold from that band
   (`use_bootstrap_CI = True`).

Both report an **estimated FDR** alongside the realized FDR/TPR.

---

## The Sweep

A single **SNR sweep** over $\mathrm{SNR} \in \lbrace0.01, 0.1, 0.2, 0.5, 0.6, 1, 2, 5\rbrace$, 200 MC trials per
point, with a fresh design, support, and noise draw in every trial — identical to Demo 01 apart from the
memory-mapped dummy storage. Per-cell trial seeds follow the C++ source
(`base_seed = 24 + snr_idx * 1000`).

---

## Output Files

Written to `simulation_results/data/`:

- `scr_screen_trex_mmap_snr_n300_p1000.txt` / `.csv` — FDR, TPR, and estimated FDR per method and SNR
  level. The `.csv` is a tidy long comma-separated values (CSV) table with the columns
  `method, metric, SNR, value`.

The file names and CSV schemas match the C++ demo, and the suite plotter
[`../trex_scr_plt_utils.py`](../trex_scr_plt_utils.py) is the same module the C++ suite uses.

Figures go to `simulation_results/plots/` as Portable Network Graphics (PNG) and Portable Document Format (PDF)
files, produced by `./generate_plots.sh`.

---

## Running the Demo

```bash
python demo_trex_scr_02_mc_sim_screen_trex_mmap.py   # full run (~2 min with 8 workers)
./generate_plots.sh                                  # render the figure below from the saved CSV
```

Monte Carlo trials are parallelized with `ProcessPoolExecutor` over `TREX_NUM_WORKERS` worker processes
(default 8, capped at the usable cores, and `0` uses all of them). The suite's older `SCR_NUM_WORKERS` still
works and takes precedence. Each worker pins its inner Open Multi-Processing (OpenMP) pool to one thread, so
the demo never exceeds the worker count in total cores. Note that every worker process drives its own
memory-mapped dummy files, so disk input/output (I/O) — not RAM — is what scales with the worker count here.

---

## Simulation Results

The numbers and figure below are this Python run's own results (200 MC trials per grid point).
They are compared with the
[C++ reference](../../../../../cpp/trex_selector_methods/ad/trex_screening/demo_trex_scr_02_mc_sim_screen_trex_mmap/README.md#simulation-results)
by size, not digit for digit, because the two ports draw from independent RNG streams (NumPy here, the C++
engine there). The tables store only the 200-trial means without standard deviations. The mean absolute
difference from the C++ run is $0.017$ in the FDR, $0.017$ in the TPR and $0.024$ in the estimated FDR.

- **Memory mapping changes the memory footprint, not the statistical behaviour.** Across the sweep the
   memory-mapped curves track this port's own in-memory Demo 01 curves closely. For $\mathrm{SNR} \ge 0.5$
   the largest FDR difference is about $0.014$ (Bootstrap-CI at $\mathrm{SNR} = 0.5$), and the Ordinary rule's
   TPR curves coincide to within $0.005$ ($0.842$ vs. $0.837$ at $\mathrm{SNR} = 1$, and $1.000$ in both at
   $\mathrm{SNR} = 5$). It is the same procedure on the same statistics — only the D matrices live on disk.
- **Exact equality is not expected.** Both demos screen the same datasets (same trial seeds) but use
   nondeterministic selector seeds (the library resolves `seed = -1` from hardware entropy), so the two runs
   draw different random experiments. The tables carry no standard deviations, so the gaps cannot be tested
   against Monte Carlo noise. The Bootstrap-CI rule shows the widest of them (TPR $0.170$ here vs. $0.214$ in
   Demo 01 at $\mathrm{SNR} = 0.5$). Its threshold is itself a bootstrap estimate, which plausibly adds a
   second layer of sampling variability.
- **The baseline conclusions carry over unchanged.** Screening is not FDR-controlling at low SNR — at
   $\mathrm{SNR} \le 0.2$ the realized FDR runs between $0.169$ and $0.475$ with TPR $\le 0.123$ — and both
   rules become reliable from $\mathrm{SNR} \ge 0.5$, where the Ordinary rule's FDR falls to $0.127$ and
   settles around $0.05\text{–}0.06$ while its TPR climbs $0.426 \to 1.000$.
- **Bootstrap-CI remains the conservative rule, with lower FDR and clearly less power.** Its FDR stays at or below
   $0.050$ for $\mathrm{SNR} \ge 0.5$, less than half the Ordinary rule's up to $\mathrm{SNR} = 1$, paid for
   with TPR $0.905$ vs. $1.000$ at $\mathrm{SNR} = 5$ and a wide mid-sweep gap ($0.387$ vs. $0.842$ at
   $\mathrm{SNR} = 1$).
- **The internal FDR estimate is conservative here.** For $\mathrm{SNR} \ge 0.5$ it sits *above* the
   realized FDR for both rules (e.g. $0.114$ vs. $0.063$ for Ordinary at $\mathrm{SNR} = 1$), over-stating
   the error rate rather than hiding it. At the lowest SNR points it becomes unreliable in both directions.

TPR (left) and FDR (right) vs. SNR, one line per thresholding rule. On the FDR panel the solid
line is the realized FDR and the dashed line the procedure's own estimated FDR.

![Memory-mapped Screen-TRex on an i.i.d. design: TPR/FDR vs SNR](simulation_results/plots/scr_screen_trex_mmap_snr_n300_p1000.png)

---

## References

1. Machkour, J., Muma, M., & Palomar, D. P., "False Discovery Rate Control for Fast Screening of
   Large-Scale Genomics Biobanks.", IEEE Statistical Signal Processing Workshop (SSP), 2023,
    pp. 666–670, IEEE.
    [DOI-Link](https://doi.org/10.1109/SSP53291.2023.10207957)
