# T-Rex SPCA: Sparse Principal Component Analysis — Python Demos

## Overview

This folder introduces the sparse principal component analysis (SPCA) version of the Terminating Random
 Experiments (T-Rex) selector, the **T-Rex SPCA** according to [[1]](#references).
A sparse principal component analysis method built on the elastic-net machinery of T-Rex+GVS, the T-Rex selector for
 grouped variable selection (GVS) (see [../trex_gvs/](../trex_gvs/README.md)), according to [[2]](#references).

Instead of selecting sparse *variables* in a regression model, T-Rex SPCA selects sparse *loadings* for each
 principal component. The T-Rex+GVS selector is applied per component to control the false discovery rate (FDR) of
 the estimated loading support. The result is a principal component analysis (PCA) whose components are supported on
 a small, FDR-controlled set of variables rather than on all $p$ of them.

The demos are the Python counterparts of the C++ suite in
 [`cpp/trex_selector_methods/ad/trex_spca/`](../../../../cpp/trex_selector_methods/ad/trex_spca/README.md)
 (and the R suite in `R/trex_selector_methods/ad/trex_spca/`), using the `trexselector` root-level
 `trex.TRexSPCASelector`. They are designed to help users understand:

1. whether the FDR of the estimated loading support is actually held at the target,
2. what that control costs in explained variance, measured against ordinary PCA and an oracle that is told
    the true support size,
3. how the elastic-net solver (`TENET` vs. `TENET_AUG`) and the loading-assembly mode (`ActiveSet` vs.
    `Thresholded`) affect the outcome,
4. and why the *choice of denominator* for the explained variance decides how the same numbers read.

---

## What this folder covers

- A **sparse $M$-factor data-generating process** (DGP) where each column of the loading matrix has exactly $p_1$
   nonzero entries drawn from a shared overlap pool, so factor supports can partially coincide.
- Two **T-Rex SPCA** solver variants crossed with two loading-assembly modes, benchmarked against **ordinary
   PCA** (no sparsity) and **oracle-thresholded PCA** (knows the true support size).
- A sweep over the signal-to-noise ratio (SNR) **in decibels**, the scale this suite reports on throughout
   (demo 01), plus sweeps over the number of extracted principal components (PCs), the number of true active loadings
   and the target FDR (demo 02).

---

## Statistical model and targets

### The factor model

```math
\boldsymbol{X} = \boldsymbol{Z}\boldsymbol{V}^\top + \boldsymbol{E},
\qquad
\boldsymbol{Z}_{\cdot,m} \sim \mathcal{N}(\boldsymbol{0}, \sigma_m^2 \boldsymbol{I}_n).
```

### Notation

- $n$: number of observations, $p$: number of variables, $M$: number of latent factors.
- $\boldsymbol{X} \in \mathbb{R}^{n \times p}$: centered observation matrix.
- $\boldsymbol{V} \in \mathbb{R}^{p \times M}$: sparse loading matrix, where each column carries exactly $p_1$
   nonzero entries of value $0.9$, drawn without replacement from a shared candidate pool of size
   `overlap_pool_size`.
- $\sigma_m$: amplitude of factor $m$, decreasing over $\lbrace5, 3, 1\rbrace$ (falling back to $1$ for $M > 3$), so
   the components are ordered by strength.
- $\mathcal S_1$: the true active loading support of the first PC (PC1), and $\widehat{\mathcal S_1}$ its estimated
   counterpart.
- $\boldsymbol{E}$: independent and identically distributed (i.i.d.) Gaussian noise, scaled to hit the target SNR.

### Signal-to-noise ratio in decibels

```math
\mathrm{SNR}_{\mathrm{dB}} = 10 \log_{10}\!\left(\frac{\mathrm{Var}(\boldsymbol{Z}\boldsymbol{V}^\top)}{\mathrm{Var}(\boldsymbol{E})}\right).
```

Both the tables and the figures use this decibel axis directly, and it is never converted back to a linear ratio.

### False discovery rate, true positive rate, explained variance

```math
\mathrm{FDR} = \mathbb{E}\left[\frac{|\widehat{\mathcal{S}}_1 \setminus \mathcal{S}_1|}{\max\{1, |\widehat{\mathcal{S}}_1|\}}\right],
\qquad
\mathrm{TPR} = \mathbb{E}\left[\frac{|\mathcal{S}_1 \cap \widehat{\mathcal{S}}_1|}{\max\{1, |\mathcal{S}_1|\}}\right].
```

The FDR and the true positive rate (TPR) are evaluated on the **first PC only**, and FDR control is assumed only
 there, because the factor supports are drawn from a shared pool and overlap, so the plug-in PCs that supervise
 the later components' selectors already blend several factors' supports — beyond PC1 there is no unambiguous
 per-component ground truth to control against. (Demo 01's union-FDR section covers what *can* be measured
 beyond PC1.)

The **proportion of explained variance (PEV)** — computed cumulatively across all $M$ components — is the
 quantity the sparsity is bought against, and the place where the cost of FDR control becomes visible.

Classically the explained variance (EV) is the plain EV of the estimated components,
 $\mathrm{EV} = \mathrm{tr}(\widehat{\boldsymbol{Z}}^\top \widehat{\boldsymbol{Z}}) / (n-1)$. Sparse loading vectors
 are not orthogonal, however, and correlated components double-count shared variance under that trace —
 Zou, Hastie & Tibshirani [[3]](#references) therefore correct it with the **adjusted** EV, which takes an
 orthogonal-triangular (QR) decomposition of the estimated score matrix, whose diagonal carries each component's
 *incremental* contribution,

```math
\mathrm{EV}_{\mathrm{adj}} = \frac{1}{n-1} \sum_{m=1}^{M} r_{mm}^2,
\qquad \widehat{\boldsymbol{Z}} = \boldsymbol{Q}\boldsymbol{R}.
```

*What one divides it by* is then what decides what the number means, and this suite reports **both** choices.

**The straightforward reading** divides by the variance of the whole data matrix,

```math
\mathrm{PEV}_{\mathrm{total}} = \mathrm{EV}_{\mathrm{adj}} / \mathrm{Var}(\boldsymbol{X}).
```

It answers "how much of what is in front of us did the $M$ components reproduce", and it is **bounded by 1** by
 construction — no set of $M$ components can explain more variance than $\boldsymbol{X}$ contains.

**The reference paper's reading** (Definition 1 of [[1]](#references)) divides by the **Signal + Mixed EV of
 the data**: the variance carried by the true active variables, with
 $\mathcal{A}$ the union of the factor supports,

```math
\mathrm{PEV}_{\mathrm{sig}} = \frac{\mathrm{EV}_{\mathrm{adj}}}
{\lVert \boldsymbol{X}_{\mathcal{A}} \rVert_F^2 \,/\, (n-1)}.
```

The denominator is **fixed per dataset and shared by all methods** — that is what lets the metric compare
 methods on a common scale, and what the paper's Fig. 3 requires, because its curves *rise* with SNR from far below
 $100\%$, which no method-specific (self-normalized) denominator can produce. This convention follows the legacy
 R reference (`TRex_Simulations/.../trex_spca/demo_trex_spca_03_fig3.R`, denominator mode `"active"`), which
 reproduces the rise of the published Fig. 3(b) from about $40\%$. No constructible denominator reproduces all
 published anchors of Fig. 3. An earlier revision of the cpp suite normalized each method by its *own*
 Signal + Mixed EV instead ("Definition 1 as printed"), which is structurally $\geq 100\%$ for null-capturing
 methods and inverts the SNR trend — see the
 [cpp demo-02 README](../../../../cpp/trex_selector_methods/ad/trex_spca/demo_trex_spca_02_mc_sim_pev/README.md)
 for the full account.

The consequence is the whole point of the metric. It is **not capped**. A method that spends loadings on null
 variables accumulates variance the denominator does not contain, so its PEV climbs past $100\%$ — and, as
 [[1]](#references) puts it, exceeding $100\%$ "indicates an inferior performance of the respective method".
 Ordinary PCA, using all $p$ loadings, overshoots hardest. The T-Rex variants stay below $100\%$ in every sweep
 of both demos. The gap to $100\%$ is variance on the active columns that the $M$ sparse components do not
 capture, and at high SNR it is larger for `ActiveSet` than for `Thresholded` (demo 01, $81.4\text{–}81.6\%$ vs.
 $95.9\%$ at $+10$ dB).

A third reading, the metric `PEVsigmix`, reports a method's *own* Signal + Mixed part — its raw EV
 minus the Null EV of its loadings, split by variable — over the same shared denominator. Read for ordinary
 PCA it is the paper's **"Ordinary PCA (Sig + Mix)"** reference curve, which levels off near $108\%$ once all
 components are extracted ($108.0\%$ at 40 and $107.8\%$ at 49 components in demo 02), while plain ordinary PCA
 keeps climbing on Null EV.

Keeping all readings is what makes the trade-off legible. The first prices the sparsity, and the second exposes
 where a non-sparse method's apparent advantage actually comes from.

---

## T-Rex SPCA specific concepts

- **Plug-in supervision (and the scope of the FDR guarantee)** — each component's selector regresses on
   the *plug-in* ordinary PC $\boldsymbol z_m = \boldsymbol X \boldsymbol v_m$, a one-shot construction
   rather than the iterative refinement most sparse-PCA methods use. The FDR guarantee is therefore
   **conditional on the plug-in PC being a faithful proxy for its factor**, because the selector's guarantee
   refers to the regression it is handed, and that transfers to the factor model only where the proxy is
   good. Demo 01 (union-FDR section) and demo 02 (FDR-heatmap section) measure where the condition holds
   and where it fails.
- **`trex.SPCAMode`** — how the sparse loading vector is assembled once a support has been selected:
  - `ActiveSet`: refit the plug-in PC on the T-Rex+GVS selected variables by ridge regression
     (`lambda2_ridge_loadings`, default $10^{-6}$) and use those coefficients as loadings,
  - `Thresholded`: keep the largest-magnitude entries of the ordinary-PCA loading vector, as many as T-Rex+GVS
     selected, so only the support size comes from the selector.
- **`trex.ENSolverType`** — the elastic-net solver driving each per-component selection: `TENET`
   (Gram-based) or `TENET_AUG` (augmented least absolute shrinkage and selection operator (LASSO) formulation).
   See [../trex_gvs/](../trex_gvs/README.md) for the shared elastic-net background.
- **`lambda_2`** — the ridge penalty (`spca_ctrl.gvs_ctrl.lambda_2`): a negative value selects it by
   $k$-fold cross-validation, $0$ is the degenerate no-ridge case (pure terminating LASSO (T-LASSO), and the
   wrapper warns), and a positive value fixes it.
- **`tFDR`** — the target FDR handed to the per-component T-Rex selector (an argument of
   `sel.select(M, tFDR)`).

---

## Folder contents

```txt
trex_spca/
  ├── README.md
  ├── trex_spca_sim_utils.py           # DGP, MC loops, table + CSV output (Python
  │                                    #   counterpart of trex_spca_sim_utils.hpp)
  ├── trex_spca_plt_utils.py           # shared plotting module (dB sweep figures;
  │                                    #   identical to the cpp suite's module)
  ├── generate_corr_plots.sh           # renders dgp_corr_plots/ (one figure per DGP)
  ├── dgp_corr_plots/                  # design-correlation heatmap of the factor DGP
  ├── run_overnight.sh                 # batch runner: both demos, then the heatmap
  │
  ├── demo_trex_spca_01_mc_sim/
  │   ├── demo_trex_spca_01_mc_sim.py  # the demo source
  │   ├── README.md                    # scenario description and results
  │   ├── generate_plots.sh            # renders this demo's figures from its CSVs
  │   └── simulation_results/
  │       ├── data/                    # .txt summary + .csv tables
  │       └── plots/                   # .png + .pdf figures
  │
  └── demo_trex_spca_02_mc_sim_pev/
      ├── demo_trex_spca_02_mc_sim_pev.py
      ├── README.md
      ├── generate_plots.sh            # combines the four sweep CSVs into panels
      └── simulation_results/
          ├── data/                    # one .txt + .csv pair per sweep
          └── plots/                   # .png + .pdf figures
```

The layout mirrors the C++ suite one-to-one.
 [trex_spca_sim_utils.py](trex_spca_sim_utils.py) is the suite-level shared module:
 `generate_sparse_factor_model()` (numpy `default_rng`), `center_columns()`, `evaluate_spca()` (PC1 TPR/FDR
 plus the three PEV readings `PEV`/`PEVsig`/`PEVsigmix`), the parallel Monte Carlo (MC) loops `run_mc_trials_pca()` /
 `run_mc_trials_oracle_pca()` / `run_mc_trials_trex_spca()` (`ProcessPoolExecutor`, worker count from
 `resolve_workers()`, default 8), and `save_and_print_spca_sweep_results()` (console + `.txt` table with
 per-sweep-column axis precision, plus a tidy pandas comma-separated values (CSV) file
 `method,metric,<sweep_col>,value`) with the SNR convenience wrapper `save_and_print_spca_mc_results()`.
 [generate_corr_plots.sh](generate_corr_plots.sh) renders the design-correlation heatmap of the factor DGP
 into [dgp_corr_plots/](dgp_corr_plots/) via the category-level plotter
 [../../corr_plt_utils.py](../../corr_plt_utils.py). The heatmap is embedded in the demo-01 README alongside the
 DGP description. [run_overnight.sh](run_overnight.sh) runs both demos back to back and then the heatmap,
 logging into `_overnight_logs/`.

Correctness probes for the individual pipeline stages — TENET vs. TENET_AUG equivalence, scaling
 comparisons, step-by-step C++/R diffs, and a dump of the GVS selector's internal calibration matrices —
 live in the library's test suite under
 `TRexSelector/cpp/tests/validation/trex_selector_methods/ad/trex_spca/`. The C++ and R counterparts of these
 demos run the identical simulation and can be used to cross-check the numbers (on their own random number
 generator (RNG) streams, numpy's permuted congruential generator (PCG64) vs. C++ `std::mt19937`, so exact
 per-trial matches across languages are impossible).

---

## Demo descriptions

| # | Name | Purpose | Setting | Main parameters |
|---|------|---------|---------|-----------------|
| **01** | MC Sim | Compare T-Rex SPCA (2 solvers × 2 modes) against OrdPCA/OraclePCA, plus union-support FDR per PC | Sparse 3-factor model, $n=50$, $p=100$ | $p_1=5$, `overlap_pool_size=30`, tFDR=0.10, SNR sweep in dB ($-10 \ldots +10$) |
| **02** | MC Sim PEV | Fig.-3-style cumulative-PEV sweeps (Definition 1 vs. total variance) + FDR/TPR heatmap study | Sparse 3-factor model, $n=50$, $p=100$ | $p_1=10$, sweeps over #PCs / SNR / $p_1$ / tFDR, and a (tFDR × SNR) grid in part 5 |

---

## Running a demo

```bash
cd Python/trex_selector_methods/ad/trex_spca/demo_trex_spca_01_mc_sim
python demo_trex_spca_01_mc_sim.py                # TREX_NUM_WORKERS workers (default 8)
python demo_trex_spca_01_mc_sim.py 12             # 12 workers (0 = all usable cores)
```

Run inside an environment that carries `trexselector` (e.g. the `trex-dev` conda environment or the
repo `.venv`). Each demo inserts both its own directory and the parent suite directory onto `sys.path`, so
the shared `trex_spca_sim_utils` module resolves from the nested location and is re-importable by the
spawn-launched Monte Carlo workers — run demos **as files** (not via `python -` / piped stdin), matching the
other Python suites. The demos use `TREX_NUM_WORKERS` worker processes (default 8, and `0` uses every usable
core, see `resolve_workers()` in [trex_spca_sim_utils.py](trex_spca_sim_utils.py)). The optional first
command-line interface (CLI) argument overrides that count and goes through the same `resolve_workers()`.

Both demos honour the same environment switches as their cpp counterparts: `TREX_SPCA_NUM_MC=<k>` overrides
the Monte Carlo count for quick pilot runs, and `TREX_SPCA_PARTS` selects the sections/parts to run.

The demos write a human-readable `.txt` summary and tidy `.csv` tables into their local
`simulation_results/data/` folder. Note that only the data is seeded deterministically, while each trial's dummies
are drawn from hardware entropy by design, which is what makes the Monte Carlo FDR estimate valid. A re-run
therefore reproduces the reported behaviour to within Monte Carlo noise rather than exactly.

Indices are **0-based** (Python convention), while the R counterpart is 1-based.

### Figures

Each demo folder has a `generate_plots.sh` that renders its figures from the saved CSVs into
`simulation_results/plots/` using the shared `trex_spca_plt_utils.py` and the repo-root `.venv`
(matplotlib + pandas + numpy):

```bash
cd demo_trex_spca_01_mc_sim
./generate_plots.sh
```

For demo 01, two figures are rendered: TPR and FDR, the latter against the target FDR (tFDR), over the decibel sweep
axis, and — with a `_pev` suffix — the two PEV normalizations side by side, with the 100 % line and OrdPCA's dashed
"Sig + Mix" reference in the Definition-1 panel. A third figure covers the union-FDR CSV when present. For demo 02,
the four sweep CSVs are combined into one Fig.-3-style panel figure per normalization, plus the part-5 heatmap figure.
`--xscale` is accepted for parity with the other suites, but the axis stays in dB either way.

---

## Notes for new users

- Start with `demo_trex_spca_01_mc_sim/`, then move to demo 02 for the explained-variance story.
- Read the PC1-only FDR/TPR rationale above before interpreting results — this is a deliberate
   evaluation-scope choice, not a bug.
- All languages run the per-PC selector with the library default scaling (unit-L2, matching the legacy
   `lm_dummy.R` centering + L2 normalization). Do **not** z-score $\boldsymbol{X}$, because the factor amplitude
   signal lives in the column variances and z-scoring destroys it. An earlier z-scoring pipeline pushed the
   T-Rex SPCA FDR at $-10$ dB to about $0.5$ (noted in the header of the C++
   `validation_trex_spca_06_handrolled_comparison`, which stores no output of that run).
- Cross-check numbers against the language counterparts:
   [`cpp/trex_selector_methods/ad/trex_spca/`](../../../../cpp/trex_selector_methods/ad/trex_spca/README.md) and
   `R/trex_selector_methods/ad/trex_spca/`.

---

## References

1. Machkour, J., Breloy, A., Muma, M., Palomar, D. P., & Pascal, F., "Sparse PCA with False Discovery Rate Controlled
   Variable Selection.", IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2024,
   pp. 9716–9720, IEEE.
   [DOI-Link](https://doi.org/10.1109/ICASSP48485.2024.10448237)
2. Machkour, J., Muma, M., & Palomar, D. P., "False Discovery Rate Control for Grouped Variable Selection
   in High-Dimensional Linear Models using the T-Knock Filter.", European Signal Processing Conference (EUSIPCO), 2022,
    pp. 892–896, European Association for Signal Processing (EURASIP).
    [DOI-Link](https://doi.org/10.23919/EUSIPCO55093.2022.9909883)
3. Zou, H., Hastie, T., & Tibshirani, R., "Sparse Principal Component Analysis.", Journal of Computational
   and Graphical Statistics, vol. 15, no. 2, 2006, pp. 265–286, Taylor & Francis.
   [DOI-Link](https://doi.org/10.1198/106186006X113430)
