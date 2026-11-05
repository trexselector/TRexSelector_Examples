# Screen-TRex: Variable Screening Demonstration Suite

## Overview

This folder contains C++ example programs for **Screen-TRex**, the Terminating Random Experiments (T-Rex) Selector
extended with a
 variable-**screening** step for ultra-high-dimensional data, according to [[1]](#references).

Screen-TRex targets settings where $p$ is so large that running the classical T-Rex selector directly is
 impractical — genome-wide or biobank-scale designs — by first thresholding a dummy-based voting statistic
 to reduce the candidate set.
 Unlike the classical T-Rex selector, **screening has no target parameter for the false discovery rate (FDR)**.
 It thresholds the voting statistic instead of calibrating to a user-specified level, and reports its own
 *estimated FDR* alongside the selection. Whether that self-estimate can be trusted is a central question of this
 suite, and the answer depends sharply on the correlation structure of the design (see Demos 01 and 03).

The demos in this folder are designed to help users understand:

1. how the **Ordinary** (majority-vote) rule and the **Bootstrap-CI** rule, built on a bootstrap confidence
    interval (CI), behave in memory and memory-mapped,
2. how screening degrades under **correlated designs**, and how far the **dependency-aware (DA)** variants
    can correct it,
3. how the **Biobank** workflow ("Algorithm 1") routes each of many phenotypes to the Ordinary or the
    Bootstrap-CI selection, or to a classical T-Rex fallback when neither estimate is acceptable,
4. how the choice of underlying T-Rex **solver backend** (TLARS, TAFS, TOMP) shifts the power/FDR operating
    point.

If you are new to this folder, start with **Demo 01** for the baseline comparison, then **Demo 03** for the
 correlation results, then **Demo 04** for the biobank workflow.

---

## What this folder covers

- **Baseline screening** (Demos 01–02): Ordinary vs. Bootstrap-CI on an independent and identically
   distributed (i.i.d.) Gaussian design, in memory
   and memory-mapped.
- **Correlation robustness** (Demo 03): plain screening vs. the dependency-aware variants under first-order
   autoregressive (AR(1)), equicorrelated, and block-equicorrelated designs, swept over both signal-to-noise ratio (SNR)
   and correlation strength.
- **Multi-phenotype biobank screening** (Demos 04–05): Algorithm 1's adaptive routing applied to several
   phenotypes sharing one design matrix, in memory and memory-mapped.
- **Solver sensitivity** (Demo 06): whether the T-Rex backend (TLARS, TAFS, TOMP) changes screening
   performance.

---

## Statistical model and targets

### The linear model

All demos use the same Gaussian linear model as the classical T-Rex selector:

```math
\boldsymbol{y} = \boldsymbol{X}\boldsymbol{\beta} + \boldsymbol{\varepsilon},
\qquad
\boldsymbol{\varepsilon} \sim \mathcal{N}(\boldsymbol{0}, \sigma_{\varepsilon}^2 \boldsymbol{I}_n).
```

### Notation

- $n$: number of observations, $p$: number of variables, $s$: number of truly active variables.
- $\boldsymbol{X} \in \mathbb{R}^{n \times p}$: design matrix, $\boldsymbol{y} \in \mathbb{R}^n$: response.
- $\mathcal{A} = \lbrace j : \beta_j \neq 0 \rbrace$: true active support, $\widehat{\mathcal{A}}$: selected set.
- $\Phi_j$: dummy-based voting proportion for variable $j$ — the evidence T-Rex accumulates internally.
- $\rho$: correlation strength (AR(1) lag-1, equicorrelation, or within-block equicorrelation).
- $\sigma_{\varepsilon}^2$: noise variance, calibrated to a target SNR.

### Correlation models

- **i.i.d. Gaussian** (Demos 01, 02, 04–06): $X_{ij} \sim \mathcal{N}(0,1)$, no correlation structure.
- **AR(1)** (Demo 03): $x_j = \rho\, x_{j-1} + \sqrt{1-\rho^2}\, w_j$, so columns correlate as
  $\rho^{|j - j'|}$.
- **Equicorrelated** (Demo 03): $x_j = \sqrt{\rho}\, z + \sqrt{1-\rho}\, w_j$ with one shared latent factor
  $z$, so every pair of columns correlates as $\rho$.
- **Block-equicorrelated** (Demo 03): the same construction with one latent factor per block, so columns
  correlate as $\rho$ within a block and are independent across blocks.

### Signal-to-noise ratio

As elsewhere in this project,
 $\mathrm{SNR} = \mathrm{Var}(\boldsymbol{X}\boldsymbol{\beta}) / \mathrm{Var}(\boldsymbol{\varepsilon})$,
 with $\sigma_{\varepsilon}$ calibrated to hit a target SNR for a fixed $(\boldsymbol{X}, \boldsymbol{\beta})$.

### False discovery rate and true positive rate

```math
\mathrm{FDR} = \mathbb{E}\left[\frac{|\widehat{\mathcal{A}} \setminus \mathcal{A}|}{\max\{1, |\widehat{\mathcal{A}}|\}}\right],
\qquad
\mathrm{TPR} = \mathbb{E}\left[\frac{|\mathcal{A} \cap \widehat{\mathcal{A}}|}{\max\{1, |\mathcal{A}|\}}\right].
```

### What is actually measured in these demos

> **Screening returns a candidate set, and every rate is evaluated on individual variables.** The quantity
> Screen-TRex thresholds is the voting proportion $\Phi_j$, and what it returns is a set of variable
> indices. $\widehat{\mathcal{A}}$ and $\mathcal{A}$ above are therefore sets of *variables*. A false
> discovery is a selected variable that is not truly active.
>
> Screen-TRex additionally reports an **estimated FDR** — its own internal assessment of the selection's
> error rate. Every figure in this folder plots it (dashed) against the realized FDR (solid), because the
> two can diverge severely. The estimate is conservative on i.i.d. designs, except at the weakest signal, but
> wildly optimistic under equicorrelation (Demo 03).
>
> In the biobank demos, the FDR and the true positive rate (TPR) are accumulated **conditionally** on which method a
> phenotype was routed to, alongside the unconditional **Usage %**. A conditional rate over a route that handled no
> phenotypes at a given SNR is an average over an empty set, not a failure.

---

## Screen-TRex specific concepts

- **`ScreenTRexMethod`** — the screening variant: `TREX` (ordinary voting-based screening, with or without
   the bootstrap rule) and the dependency-aware variants `TREX_DA_AR1`, `TREX_DA_EQUI`,
   `TREX_DA_BLOCK_EQUI` (Demo 03).
- **Ordinary vs. Bootstrap-CI** (`ScreenTRexControlParameter::use_bootstrap_CI`) — the **Ordinary** rule
   thresholds at $\Phi_j > 0.5$, while the **Bootstrap-CI** rule builds a bootstrap confidence band
   (`R_boot = 1000` replicates, grid granularity `ci_grid_step = 0.001`) around the estimated FDR curve and
   picks its threshold from that band.
- **Dependency-aware (DA) screening** (`rho_thr_DA`, `n_blocks`) — deflates each voting proportion by its
   distance to the nearest one among its correlated neighbours (the AR(1) window, all variables, or the
   variables of the same block), so redundant correlated variables cannot pass the vote together. Screen-TRex
   runs a single step, so the proportions are coarse, and under (block-)equicorrelation many of them tie. A
   tied proportion is halved onto the threshold and rejected, which makes DA-EQUI select nothing and
   DA-BLOCK-EQUI select very little in Demo 03. For DA-EQUI the reference R package selects nothing on
   identical data as well.
- **Biobank / "Algorithm 1"** (`BiobankScreenTRexControl`, Demos 04–05) — screens **multiple phenotypes
   against one shared design matrix**. Per phenotype, one Screen-TRex run yields an Ordinary and a
   Bootstrap-CI estimated FDR. A rule whose estimate lies in `[lower_bound_FDR, upper_bound_FDR]` is used, and
   if both do, the larger estimate wins (a tie goes to Bootstrap-CI). If neither does, the classical T-Rex
   selector runs as the fallback at `target_FDR_trex`. The share routed to each method is reported as
   **Usage %**.

---

## Folder contents

```txt
trex_screening/
  ├── README.md
  ├── CMakeLists.txt
  ├── trex_screening_dgps.hpp               # ScrDGPData + all DGP generators
  ├── trex_screening_simulation_utils.hpp   # controls, MC loop, table + CSV output
  ├── trex_scr_plt_utils.py                 # shared plotting module (sweep / biobank figures)
  ├── reformat_result_txt.py                # re-render saved .txt tables from CSVs (maintenance)
  │
  ├── demo_trex_scr_01_mc_sim_screen_trex/
  │   ├── demo_trex_scr_01_mc_sim_screen_trex.cpp   # the demo source
  │   ├── README.md                                 # scenario description and results
  │   ├── generate_plots.sh                         # renders this demo's figures from its CSVs
  │   └── simulation_results/
  │       ├── data/                                 # .txt summaries + .csv tables
  │       └── plots/                                # .png + .pdf figures
  │
  │   ... demos 02–06 follow the same layout ...
  │
  └── demo_trex_scr_06_mc_sim_solvers/
      ├── demo_trex_scr_06_mc_sim_solvers.cpp
      ├── README.md
      ├── generate_plots.sh
      └── simulation_results/
          ├── data/
          └── plots/
```

All six demos are Monte Carlo studies (200 repetitions per grid point), and every result in this folder was
 produced by the committed sources.

---

## Building the demos

From the C++ workspace root:

```bash
cd TRexSelector_Examples/cpp
cmake -S . -B build/release -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="<path-to-TRexSelector>/cpp/install"
cmake --build build/release
```

The executables mirror the source tree under the `bin/` directory:

```txt
build/release/bin/trex_selector_methods/ad/trex_screening/<demo_folder>/<demo_name>
```

---

## Running a demo

For example, to run Demo 01:

```bash
./build/release/bin/trex_selector_methods/ad/trex_screening/demo_trex_scr_01_mc_sim_screen_trex/demo_trex_scr_01_mc_sim_screen_trex
```

Every demo writes a human-readable `.txt` summary and a tidy `.csv` table into its local
`simulation_results/data/` folder. To run all six back to back as an unattended job, use the family runner
[../../run_all_overnight.sh](../../run_all_overnight.sh) with `trex_screening` as its argument (`--build` rebuilds the
targets first). It logs each demo with its exit code and wall-clock time, keeps the machine awake, and resumes a
stopped batch at the demo it was in.

Note that the selector draws its dummies from a nondeterministic seed, so re-running a demo reproduces the
 reported behaviour but not the exact digits.

### Figures

Each demo folder has a `generate_plots.sh` that renders its figures from the saved comma-separated values (CSV) files
into `simulation_results/plots/` using the shared `trex_scr_plt_utils.py` and the repo-root `.venv`
(matplotlib + pandas + numpy):

```bash
cd demo_trex_scr_01_mc_sim_screen_trex
./generate_plots.sh
```

`trex_scr_plt_utils.py` auto-detects the CSV shape, so sweep results render as a TPR panel and an FDR panel
(realized solid, estimated dashed), while the biobank results add a third Usage-% panel and shade the
routing window.

---

## References

1. Machkour, J., Muma, M., & Palomar, D. P., "False Discovery Rate Control for Fast Screening of
   Large-Scale Genomics Biobanks.", IEEE Statistical Signal Processing Workshop (SSP), 2023,
    pp. 666–670, IEEE.
    [DOI-Link](https://doi.org/10.1109/SSP53291.2023.10207957)
