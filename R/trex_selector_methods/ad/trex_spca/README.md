# T-Rex SPCA: Sparse Principal Component Analysis — R Demonstration Suite

## Overview

This folder contains the R example program for **T-Rex SPCA**, a sparse principal component analysis (SPCA) method built
on top of **T-Rex+GVS** (see [../trex_gvs/](../trex_gvs/README.md)), the Terminating Random Experiments (T-Rex) selector
for grouped variable selection (GVS). Instead of selecting sparse *variables* in a regression model, T-Rex SPCA selects
sparse *loadings* for each principal component (PC), applying the T-Rex+GVS elastic-net machinery per component to
control the false discovery rate (FDR) of the estimated loading support.

The two demos in this folder compare T-Rex SPCA against two principal component analysis (PCA) baselines on a
synthetic sparse-factor-model dataset. They are the R counterparts of the C++ demos `demo_trex_spca_01_mc_sim.cpp` and
`demo_trex_spca_02_mc_sim_pev.cpp` and mirror that suite's folder structure one-to-one.

---

## What this folder covers

- A **sparse M-factor data-generating process** $X = ZV^\top + E$, where the loading matrix $V$ has exactly `p1`
  nonzero entries per factor, drawn from a shared overlap pool so factor supports can partially coincide.
- Four **T-Rex SPCA** solver/mode combinations (elastic-net solver `TENET` vs. `TENET_AUG`, and loading-assembly mode
  `ActiveSet` vs. `Thresholded`) benchmarked against **ordinary PCA** (no sparsity) and **oracle-thresholded PCA**
  (knows the true support size). The PCA baselines use base-R `svd()`.
- The cumulative proportion of explained variance under the reference paper's Definition 1, swept over four axes as in
  its Fig. 3, and the realized error rate and power of the first component's support over a grid of target levels and
  noise levels (demo 02).

---

## Statistical model and notation

```math
\boldsymbol{X} = \boldsymbol{Z}\boldsymbol{V}^\top + \boldsymbol{E},
```

- $\boldsymbol{X} \in \mathbb{R}^{n \times p}$: centered observation matrix.
- $\boldsymbol{Z} \in \mathbb{R}^{n \times M}$: true latent factor scores, column $m$ drawn $\mathcal{N}(0, \sigma_m^2)$
  with decreasing factor strength $\sigma_m \in \lbrace5, 3, 1\rbrace$ (falling back to $1$ for $M > 3$).
- $\boldsymbol{V} \in \mathbb{R}^{p \times M}$: true sparse loading matrix, where each column has exactly `p1` nonzero
  entries (value $0.9$), drawn without replacement from a shared candidate pool of size `overlap_pool_size` — so
  different factors' active sets can overlap.
- $\boldsymbol{E} \in \mathbb{R}^{n \times p}$: independent and identically distributed (i.i.d.) Gaussian noise, scaled
  so that the realized signal-to-noise ratio (SNR) matches a target value **in decibels**,

```math
\mathrm{SNR}_{\mathrm{dB}}
= 10 \log_{10}\!\left(\frac{\mathrm{Var}(\boldsymbol{Z}\boldsymbol{V}^\top)}{\mathrm{Var}(\boldsymbol{E})}\right).
```

All methods are evaluated on the same $\boldsymbol{X}$ after a shared **center-only** preprocessing step (mean
subtraction, no column scaling — a covariance-PCA footing), so that OrdPCA, OraclePCA, and T-Rex SPCA are compared on
equal preprocessing. The column scales must **not** be normalized, because in this factor model the amplitude signal
lives in the column variances, and z-scoring the columns (correlation PCA) destroys it. The measured effect at $-10$ dB
is that the T-Rex SPCA FDR degrades from $\approx 0.13$ to $\approx 0.53$ (see the C++
`validation_trex_spca_06_handrolled_comparison`).

---

## T-Rex SPCA specific concepts

- **`mode`** (in `trex_spca_control()`) — how the sparse loading vector for each principal component is assembled
  from the underlying T-Rex+GVS selection:
  - `"ActiveSet"`: refit the plug-in PC on the T-Rex+GVS *selected* variables by ridge regression and use
    those coefficients as loadings.
  - `"Thresholded"`: keep the largest-magnitude entries of the ordinary-PCA loading vector, as many as
    T-Rex+GVS selected, so only the support size comes from the selector.
- **`en_solver`** — the elastic-net solver driving each per-component T-Rex+GVS selection: `"TENET"` (Gram-based) or
  `"TENET_AUG"` (augmented least absolute shrinkage and selection operator (LASSO) formulation). See
  [../trex_gvs/README.md](../trex_gvs/README.md) for the shared elastic-net background.
- **PC1-only FDR/TPR** — FDR and the true positive rate (TPR) are evaluated **strictly on the loading support of the
  first principal component (PC1)**. Components 2 and 3 are intentionally excluded from these metrics because ordinary
  PCA's orthogonality constraint mixes their loading supports across the true factors, making a per-component "true
  support" comparison ambiguous beyond PC1. This is a deliberate scope limitation of the evaluation, not a limitation
  of T-Rex SPCA itself.
- **Cumulative PEV** — the proportion of explained variance (PEV) is instead evaluated cumulatively across **all** $M$
  components (via an orthogonal-triangular (QR) decomposition of the estimated score matrix), since this metric does
  not suffer from the same per-component ambiguity.
- **Internal scaling (R binding note)** — `trex_spca_control()` does not expose the per-PC selector's internal
  `scaling_mode`, but this is moot, because the C++ demo and the Python port also use the library default (unit-L2
  column scaling), which matches the legacy `lm_dummy.R` centering + L2 normalization.

---

## Statistical targets

```math
\mathrm{FDR}
=
\mathbb{E}
\left[
\frac{|\widehat{\mathcal{S}}_1 \setminus \mathcal{S}_1|}
{\max\{1, |\widehat{\mathcal{S}}_1|\}}
\right],
\qquad
\mathrm{TPR}
=
\mathbb{E}
\left[
\frac{|\mathcal{S}_1 \cap \widehat{\mathcal{S}}_1|}
{\max\{1, |\mathcal{S}_1|\}}
\right],
```

where $\mathcal S_1$ is PC1's true active loading support and $\widehat{\mathcal S_1}$ its estimated counterpart (both
**1-based** in R). **Ordinary PCA** trivially achieves $\mathrm{TPR}=1$ and $\mathrm{FDR}\approx(p-p_1)/p$ since it
retains all $p$ loadings — it is a non-sparse reference point, not a genuine competitor on these two metrics.
**Oracle PCA** (knows the true support cardinality $p_1$) is a near-ideal sparse reference, though not a strict bound,
because the T-Rex variants match or pass its TPR and explained variance at some grid points.

---

## Start here

Start with **`demo_trex_spca_01_mc_sim/`**, which compares all methods via a Monte Carlo (MC) simulation over an SNR
sweep, and continue with **`demo_trex_spca_02_mc_sim_pev/`** for the explained variance. Each demo's local README holds
the full parameter breakdown.

---

## Demo descriptions

| # | Name | Purpose | Setting | Main parameters |
| --- | --- | --- | --- | --- |
| **01** | MC Sim | Compare T-Rex SPCA (2 solvers × 2 modes) against OrdPCA/OraclePCA, plus the union-support FDR per PC | Sparse 3-factor model, $n=50$, $p=100$ | $p_1=5$, `overlap_pool_size=30`, target FDR $\mathrm{tFDR}=0.10$, SNR $\in \lbrace-10, \ldots, 10\rbrace$ dB (9 points), MC = 200 |
| **02** | MC Sim PEV | Cumulative PEV under four sweeps (Fig. 3) and FDR/TPR heatmaps over target FDR $\times$ SNR | Sparse 3-factor model, $n=50$, $p=100$ | $p_1=10$, sweeps over the number of components, SNR, $p_1$ and target FDR, $11 \times 9$ heatmap grid, MC = 200 |

---

## Folder contents

```txt
trex_spca/
  ├── README.md
  ├── trex_spca_sim_utils.R
  ├── trex_spca_plt_utils.R
  ├── generate_corr_plots.sh
  ├── run_overnight.sh
  ├── dgp_corr_plots/                (spca_factor.png)
  ├── demo_trex_spca_01_mc_sim/
  │   ├── demo_trex_spca_01_mc_sim.R
  │   ├── generate_plots.sh
  │   ├── README.md
  │   └── simulation_results/
  │       ├── data/
  │       │   ├── demo_trex_spca_01_mc_sim.txt
  │       │   ├── demo_trex_spca_01_mc_sim.csv
  │       │   └── demo_trex_spca_01_mc_sim_union_fdr.csv
  │       └── plots/
  └── demo_trex_spca_02_mc_sim_pev/
      ├── demo_trex_spca_02_mc_sim_pev.R
      ├── generate_plots.sh
      ├── README.md
      └── simulation_results/
          ├── data/          (pcs / snr / loadings / tfdr .txt + .csv, fdr_heatmap.csv)
          └── plots/
```

The suite-level [trex_spca_sim_utils.R](trex_spca_sim_utils.R) mirrors the C++ `trex_spca_sim_utils.hpp`:
`generate_sparse_factor_model()`, `center_columns()`, `evaluate_spca()` (TPR/FDR on PC1 plus cumulative PEV under three
normalizations), the Monte Carlo runners `run_mc_trials_pca()` / `run_mc_trials_oracle_pca()` /
`run_mc_trials_trex_spca()`, and the writers `save_and_print_spca_sweep_results()` /
`save_and_print_spca_mc_results()` (console + `.txt` table plus a tidy comma-separated values (CSV) file
`method,metric,<sweep>,value`). The trials run on spawned R worker processes, a parallel socket (PSOCK) cluster via
`mc_map()` in the family-level [../../mc_runtime.R](../../mc_runtime.R), and both demos are resumable through its cell
cache. Figures are rendered by [trex_spca_plt_utils.R](trex_spca_plt_utils.R) with ggplot2 as Portable Network Graphics
(PNG) and Portable Document Format (PDF) files via each demo's `generate_plots.sh`, with the same file names as the
Python suite.

---

## Recorded results

The committed results are the first R run of both demos on library commit 44cf1ab (200 trials per grid point,
8 workers, 84 minutes in total, 9 for demo 01 and 76 for demo 02). Each demo README shows its R numbers and figures.
R's generator draws other data from the same seeds, and the per-trial dummies use hardware entropy (selector
`seed = -1L`), as required for a valid Monte Carlo FDR estimate. The R cells therefore agree with C++ and Python up
to Monte Carlo noise, not exactly.

The result tables carry no standard deviations, so the comparison with C++ measures the size of the gaps and sets it
against the gaps between Python and C++. Over the 951 FDR and TPR cells of both demos, the mean gap per table is
0.002–0.010 between R and C++ and 0.003–0.011 between Python and C++. The explained-variance cells differ by
0.002–0.003 on average in both comparisons, and the signed bias of R against C++ stays below 0.003 in every table.

> **History note (z-scoring episode)**: an earlier version of this pipeline z-score-standardized
> $\boldsymbol{X}$, which destroyed the factor amplitude signal and inflated the measured T-Rex FDR to $\approx 0.53$ at
> $-10$ dB. The C++ `validation_trex_spca_06_handrolled_comparison` isolated the cause, and the pipeline is back to
> center-only.

---

## Running the demos

```bash
# Optional first argument = worker processes (default TREX_NUM_WORKERS, else 8, 0 = all cores)
Rscript R/trex_selector_methods/ad/trex_spca/demo_trex_spca_01_mc_sim/demo_trex_spca_01_mc_sim.R
Rscript R/trex_selector_methods/ad/trex_spca/demo_trex_spca_02_mc_sim_pev/demo_trex_spca_02_mc_sim_pev.R 8
TREX_SPCA_NUM_MC=10 Rscript R/trex_selector_methods/ad/trex_spca/demo_trex_spca_01_mc_sim/demo_trex_spca_01_mc_sim.R
R/trex_selector_methods/ad/trex_spca/run_overnight.sh                  # both demos
SKIP_HEAVY=1 R/trex_selector_methods/ad/trex_spca/run_overnight.sh     # demo 01 only
```

`TREX_SPCA_NUM_MC=<k>` overrides the Monte Carlo count for quick pilot runs, and `TREX_SPCA_PARTS` selects the parts
to run (default `"12"` in demo 01 and `"12345"` in demo 02), as in the C++ and Python demos. Every finished grid point
is stored in the demo's cell cache (`simulation_results/cache/`, gitignored), so a restarted demo resumes where it
stopped. `TREX_FRESH=1` starts from scratch, and a finished demo removes its cache.

Runtimes of the committed run at 8 workers:

| Demo | 01 | 02 | total |
| :--- | :--- | :--- | :--- |
| Runtime | 9 min | 76 min | 84 min |

Each demo folder has a `generate_plots.sh` that renders its figures from the saved CSVs into
`simulation_results/plots/`, and `generate_corr_plots.sh` renders the design-correlation heatmap of the factor model
into `dgp_corr_plots/`:

```bash
cd R/trex_selector_methods/ad/trex_spca/demo_trex_spca_01_mc_sim && ./generate_plots.sh
R/trex_selector_methods/ad/trex_spca/generate_corr_plots.sh
```

Dependencies are `TRexSelector` and base-R `parallel` for the demos plus ggplot2 and patchwork for the figures.
The old `elasticnet`/`glmnet` dependencies are gone from this suite.

---

## Cross-check probes (relocated)

The R↔C++ cross-check material that used to live in this folder — `demo_trex_spca_02.R` (R reference-dump generator),
`lambda2_probe.R`, `lambda2_foldmatch.R`, `probe_R_dummy_variance.R`, `r_dummy_variance_probe.R`, and the baked
`rdump/` / `rdump10/` data folders — moved to the TRexSelector library test suite
(`TRexSelector/cpp/tests/validation/trex_selector_methods/ad/trex_spca/`). The `rdump_snr7_backup/` folder was
intentionally dropped, not carried over. See that suite for the stage-by-stage pipeline comparisons. This README no
longer reproduces the old probe tables.

---

## Notes for new users

- Start with `demo_trex_spca_01_mc_sim/`, then `demo_trex_spca_02_mc_sim_pev/`.
- Read the PC1-only FDR/TPR rationale above before interpreting results — this is a deliberate evaluation-scope choice,
  not a bug.
- Support sets and the indices in the `active_sets` field are **1-based** in R (the C++ sources are 0-based).
- The counterparts are the C++ suite at
  [../../../../cpp/trex_selector_methods/ad/trex_spca/](../../../../cpp/trex_selector_methods/ad/trex_spca/), the
  Python suite at
  [../../../../Python/trex_selector_methods/ad/trex_spca/](../../../../Python/trex_selector_methods/ad/trex_spca/),
  and the C++/R validation probes, now in the TRexSelector library test suite
  (`TRexSelector/cpp/tests/validation/trex_selector_methods/ad/trex_spca/`).

---

## References

- See [../../README.md](../../README.md) for the category overview of all T-Rex selector variants.
- See [../trex_gvs/README.md](../trex_gvs/README.md) for the shared T-Rex+GVS elastic-net background (`gvs_type`,
  `en_solver`, `lambda2_method`).
