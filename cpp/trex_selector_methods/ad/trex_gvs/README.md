# T-Rex+GVS: Grouped Variable Selection Demonstration Suite

## Overview

This folder contains C++ example programs for the **Terminating Random Experiments (T-Rex) Selector with Grouped
Variable Selection (GVS)**, in short T-Rex+GVS, according to [[1,2]](#references).

T-Rex+GVS extends the classical T-Rex selector (see [../trex_core/](../trex_core/README.md)) to settings where
predictors form known or discoverable **groups** of correlated variables.
 Examples are genes in the same pathway, or sensor channels driven by a shared latent signal.
 The selector aggregates the evidence of the random experiments at the group level while it runs, so that correlated
 variables reinforce each other instead of splitting the vote. However, the selection is based on individual variables,
 and that is also the level at which false discovery rate (FDR) and true positive rate (TPR) are evaluated
 (see [What is actually measured](#what-is-actually-measured-in-these-demos)).

The demos in this folder are designed to help users understand:

1. how grouped-signal recovery behaves under different within-group correlation structures (equicorrelated, first-order
    autoregressive (AR(1)), autoregressive moving-average (ARMA), heavy-tailed),
2. how group *discovery* via hierarchical agglomerative clustering (HAC) compares to using known ("oracle") groups,
3. how the elastic-net-based (`EN`) and informed-elastic-net-based (`IEN`) group solvers behave under scattered
    layouts, correlation traps, and negative correlation structure.

If you are new to this folder, start with **Demo 01** for the canonical equicorrelated-blocks scenario.

---

## What this folder covers

All demos study sparse variable selection in a Gaussian (or near-Gaussian) linear model where the active variables
 cluster into groups:

- **equicorrelated blocks** (Demos 01–04, 08): variables within a group share a common latent factor,
- **scattered groups** (Demo 02): group members are permuted across the column space instead of being contiguous,
- **gapped, randomly-ordered blocks with an inactive trap** (Demo 03, and reused by Demos 05–07): four unequal-size
   blocks, three active and one inactive equicorrelated trap,
- **negative-correlation structure** (Demo 04): groups with sign-flipped halves and spatially separated inactive traps,
- **non-Gaussian / non-equicorrelated structure** (Demos 05–07): Student-t(3) heavy tails, AR(1) Toeplitz covariance,
   and heterogeneous ARMA processes per block,
- **value of prior group information** (Demo 08): a larger 100-block benchmark comparing HAC-discovered groups
   against known ("oracle") group labels, sweeping the within-block correlation across the HAC discoverability
   boundary.

---

## Statistical model and targets

### The linear model

We work with the same Gaussian linear model as the classical T-Rex selector, but with predictors organized into
 groups:

```math
\boldsymbol{y} = \boldsymbol{X}\boldsymbol{\beta} + \boldsymbol{\varepsilon},
\qquad
\boldsymbol{\varepsilon} \sim \mathcal{N}(\boldsymbol{0}, \sigma_{\varepsilon}^2 \boldsymbol{I}_n).
```

### Notation

- $n$: Number of observations, $p$: Number of variables.
- $\boldsymbol{X} \in \mathbb{R}^{n \times p}$: Design matrix, $\boldsymbol{y} \in \mathbb{R}^n$: Response vector.
- $\boldsymbol{\beta} \in \mathbb{R}^p$: Regression coefficient vector.
- $g(j) \in \lbrace1, \dots, G\rbrace$: Group assignment of variable $j$ (the `prior_groups` vector), and background or
  independent and identically distributed (i.i.d.) variables each receive a unique singleton group ID.
- $G$ (or $M$): Number of groups.
- $\mathcal{A} = \lbrace j : \beta_j \neq 0 \rbrace$: True active-variable support, and the corresponding set of
  **active groups** is $\lbrace g(j) : j \in \mathcal{A} \rbrace$.
- $\rho$: Within-group correlation in equicorrelated data-generating processes (DGPs), induced through a shared latent
  factor per group.
- $Z_{\cdot,k}$: Latent factor shared by group $k$, $\xi_{ij}$: idiosyncratic per-entry term of the design matrix.
- $\sigma_x$: Idiosyncratic scale in the latent-factor design, $\sigma_{\varepsilon}^2$: noise variance.

### Within-group correlation models

- **Equicorrelated, unnormalized latent factor** (Demos 01–04):
  $X_{ij} = Z_{i,g(j)} + \sigma_x\, \xi_{ij}$, with $Z_{\cdot,k}, \xi_{ij} \sim \mathcal{N}(0,1)$.
  This induces within-group correlation $\rho = 1/(1+\sigma_x^2)$. The columns are *not* unit-variance
  (they have variance $1+\sigma_x^2$). To target a given $\rho$, set $\sigma_x = \sqrt{(1-\rho)/\rho}$.
- **Equicorrelated, unit-variance** (Demo 08, and the heavy-tailed blocks of Demo 05):
  $X_{ij} = \sqrt{\rho}\, Z_{i,g(j)} + \sqrt{1-\rho}\, \xi_{ij}$, giving exactly unit-variance columns with
  within-group correlation $\rho$.
- **AR(1) Toeplitz** (Demo 06): within-block covariance $\Sigma_k(i,j) = \rho^{|i-j|}$, generated via a Cholesky
  factor.
- **Heterogeneous ARMA** (Demo 07): each block follows a different ARMA specification (AR(1), moving-average (MA(3)),
  ARMA(2,1)), swept over an AR coefficient `ar_coef` instead of $\rho$.
- **Heavy-tailed** (Demo 05): the same latent-factor construction as the equicorrelated model, but with
  $\text{Student-}t(3)/\sqrt{3}$ (unit-variance) latent factors and noise.

### Signal-to-noise ratio

As in the classical T-Rex demos, the signal-to-noise ratio (SNR) is
 $\mathrm{SNR} = \mathrm{Var}(\boldsymbol{X}\boldsymbol{\beta}) / \mathrm{Var}(\boldsymbol{\varepsilon})$,
 with $\sigma_{\varepsilon}^2$ calibrated to hit a target SNR for a fixed $(\boldsymbol{X}, \boldsymbol{\beta})$.

### False discovery rate and true positive rate

The same FDR and TPR definitions from the classical T-Rex demos apply here, with $\widehat{\mathcal{A}}$ the
 selected support:

```math
\mathrm{FDR} = \mathbb{E}\left[\frac{|\widehat{\mathcal{A}} \setminus \mathcal{A}|}{\max\{1, |\widehat{\mathcal{A}}|\}}\right],
\qquad
\mathrm{TPR} = \mathbb{E}\left[\frac{|\mathcal{A} \cap \widehat{\mathcal{A}}|}{\max\{1, |\mathcal{A}|\}}\right].
```

### What is actually measured in these demos

> **Selection and evaluation are at the level of individual variables.** Although T-Rex+GVS
> *aggregates the evidence of the random experiments at the group level* while selecting, what the selector returns
> is a set of individual variable indices, and every FDR/TPR number reported in Demos 01–07 is
> **coordinate-level**: $\widehat{\mathcal{A}}$ and $\mathcal{A}$ above are sets of *variables*,
> not of groups. A false discovery is a selected variable that is not truly active.
>
> A genuinely **group-level FDR** — treating a whole group as the unit of discovery — is a
> different quantity and needs its own definition of what counts as a false discovery (e.g. a
> selected group containing no active variable), together with a matching notion of the group
> null hypothesis. See [[3]](#references) for such a formulation in the knockoff setting. It is
> *not* obtainable by reinterpreting the coordinate-level numbers.

Demo 01 Part 4 and Demo 08 additionally report group-level rates. Demo 01 Part 4 compares clustering cuts and
 gives the share of active groups hit, the share recovered in full, the group FDR (share of selected groups without
 an active variable) and the number of groups formed. Demo 08 gives the **block hit rate** (share of active blocks
 with at least one selected variable — a block-level TPR), **block FDR** (share of hit blocks that are actually
 null), **full-block rate** (share of active blocks recovered in their entirety), and a **block-purity** diagnostic
 (how consistently a true block maps to a single discovered group — always $1.0$ for oracle groups).

> **T-Rex+GVS as implemented is not designed to control a group-level FDR.** It controls the FDR of the selected
> variables, and $\mathrm{tFDR}$ bounds that rate only. The group- and block-level rates are descriptive, carry no
> target and no guarantee, and a value above $\mathrm{tFDR}$ does not indicate a loss of FDR control. With many
> singleton background groups next to a few active groups, the group FDR can be large (Demo 01 Part 4) while the
> variable-level FDR stays below the target.

---

## T-Rex+GVS specific concepts

Grouped selection introduces a few additional control parameters and types beyond the classical T-Rex selector
 (from `trex_gvs_simulation_utils.hpp` / `TRexGVSControlParameter`):

- **`GVSType`** — the group-level base method:
  - `EN`: elastic-net-based grouped selection [[1]](#references),
  - `IEN`: informed (iterative) elastic-net-based grouped selection [[2]](#references).
- **`ENSolverType`** (only applies to `GVSType::EN`) — which elastic-net solver drives the random experiments:
  - `TENET`: the base Gram-based elastic-net solver,
  - `TENET_AUG`: the augmented least absolute shrinkage and selection operator (LASSO) formulation of the
    elastic net.
- **`LambdaSelectionMethod`** — how the elastic-net penalty $\lambda_2$ is chosen: `CV_1SE_CCD` (cross-validation,
   the one-standard-error rule, via coordinate descent) or `CV_1SE_SVD` (via singular value decomposition (SVD)).
   The rules `CV_1SE_IEN_CCD` and `CV_1SE_TIK_SVD`, built for the informed elastic net (IEN) geometry, select
   $\lambda_2$ on the group structure and apply to `GVSType::IEN` only (on the IEN track the library maps
   `CV_1SE_CCD` to `CV_1SE_IEN_CCD`). All IEN rows use `CV_1SE_IEN_CCD`, which cross-validates $\lambda_2$ jointly
   with $\lambda_1$. Demos 01–07 raise its $\lambda_1$ support cap to `cv_dfmax_ratio = 1.0`
   ($n_{\mathrm{train}} = 180$ variables), because six of them have a true support of $s = 150$, above the default
   cap of $0.8\, n_{\mathrm{train}} = 144$. Demo 08 keeps the default. `CV_1SE_TIK_SVD` appears only in the Demo 01
   Part 2 comparison, because the library refuses it once the groups hold $n_{\mathrm{train}}$ or more within-group
   contrasts.
- **`prior_groups`** — the 0-based per-column group-ID vector. When groups are known in advance (an "oracle" run),
   this vector is taken directly from the data-generating process. When groups must be discovered from the data,
   HAC clustering (with a `corr_max` correlation threshold and a chosen linkage method) is used to build
   `prior_groups` automatically — this is what Demo 08 compares directly against oracle groups.

---

## Folder contents

```txt
trex_gvs/
  ├── README.md
  ├── CMakeLists.txt
  ├── trex_gvs_dgps.hpp
  ├── trex_gvs_simulation_utils.hpp
  ├── trex_gvs_plt_utils.py           # shared plotting module (heatmaps / bars / line plots)
  │
  ├── demo_trex_gvs_01_mc_sim_hastie_en_blocks/
  │   ├── demo_trex_gvs_01_mc_sim_hastie_en_blocks.cpp   # the demo source
  │   ├── README.md                    # scenario description and interpretation
  │   ├── generate_plots.sh            # renders this demo's figures from its CSVs
  │   └── simulation_results/
  │       ├── data/                    # .txt summaries + .csv tables
  │       └── plots/                   # .png + .pdf figures
  │
  │   ... demos 02–08 follow the same layout ...
  │
  └── demo_trex_gvs_08_mc_sim_block_bench/
      ├── demo_trex_gvs_08_mc_sim_block_bench.cpp
      ├── README.md
      ├── generate_plots.sh
      └── simulation_results/
          ├── data/
          └── plots/
```

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
build/release/bin/trex_selector_methods/ad/trex_gvs/<demo_folder>/<demo_name>
```

---

## Running a demo

For example, to run Demo 01:

```bash
./build/release/bin/trex_selector_methods/ad/trex_gvs/demo_trex_gvs_01_mc_sim_hastie_en_blocks/demo_trex_gvs_01_mc_sim_hastie_en_blocks
```

Every demo writes a human-readable `.txt` summary and a tidy `.csv` table into its local
`simulation_results/data/` folder. To run all eight back to back as an unattended job, use the family runner
[../../run_all_overnight.sh](../../run_all_overnight.sh) with `trex_gvs` as its argument (`--build` rebuilds the
targets first). It logs each demo, keeps the machine awake, and resumes a stopped batch at the demo it was in.

### Figures

Each demo folder has a `generate_plots.sh` that renders its figures from the saved comma-separated values (CSV) files
into `simulation_results/plots/` using the shared `trex_gvs_plt_utils.py` and the repo-root `.venv`
(matplotlib + pandas + numpy):

```bash
cd demo_trex_gvs_01_mc_sim_hastie_en_blocks
./generate_plots.sh
```

`trex_gvs_plt_utils.py` auto-detects the CSV shape and renders 2-D sweeps as TPR/FDR heatmap
grids, the demo-01 $\lambda_2$/linkage studies as grouped bar charts, and the demo-08 block
benchmark as TPR/FDR line plots (vs. SNR or, via `--vs rho`, vs. the within-block correlation).

---

## References

1. Machkour, J., Muma, M., & Palomar, D. P., "False Discovery Rate Control for Grouped Variable Selection
   in High-Dimensional Linear Models using the T-Knock Filter.", European Signal Processing Conference (EUSIPCO), 2022,
    pp. 892–896, European Association for Signal Processing (EURASIP).
    [DOI-Link](https://doi.org/10.23919/EUSIPCO55093.2022.9909883)
2. Machkour, J., Muma, M., & Palomar, D. P., "The Informed Elastic Net for Fast Grouped Variable Selection and
   FDR Control in Genomics Research.", Workshop on Computational Advances in Multi-Sensor Adaptive Processing (CAMSAP),
    2023, pp. 466–470, IEEE.
    [DOI-Link](https://doi.org/10.1109/CAMSAP58249.2023.10403489)
3. Dai, R., & Barber, R. F., "The knockoff filter for FDR control in group-sparse and multitask regression."
   Proceedings of the 33rd International Conference on Machine Learning (ICML), vol. 48, pp. 1851–1859, 2016.
   [DOI-Link](http://proceedings.mlr.press/v48/daia16.pdf)
