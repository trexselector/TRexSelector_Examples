# TRexSelector_Examples — R Sources

## Overview

This directory contains the standalone R examples for the **TRexSelector** project.
It mirrors the sibling [cpp/](../cpp/README.md) examples tree and covers the same
library stack from the R side, with memory-mapped (mmap) matrices, data-preprocessing
building blocks (ml_methods) and Terminating Random Experiments (T-Rex) selector variants.

The demos use the **TRexSelector** R package (2.0.0), the Rcpp/R6 package from the main repo
(`TRexSelector/R/TRexSelector`, requires R >= 4.3). Its R6 application programming interface (API) is
`TRexSelector$new(X, y, tFDR, control = trex_control(...))$select()`, plus
helpers such as `trex_fdp()` / `trex_tpp()`, `mem_matrix()` /
`mem_matrix_from()`, the scaler classes, and
`ml_agglomerative_cluster()`. Version 2.0.0 replaces the pure-R package of the same name (1.0.0 on the
Comprehensive R Archive Network (CRAN), with the functional API `trex()`, `FDP()`, `TPP()`,
`add_dummies_GVS()`) in place. No demo in this tree depends on the pure-R package any more, because its only
consumer, the path-equivalence reference generator `demo_ts_compare_tlars_tlasso.R`, moved to the TRexSelector
library test suite (`TRexSelector/cpp/tests/validation/tsolvers/`), and all the selector suites (trex_da,
trex_gvs, trex_spca, and the new trex_screening) run on the R6 API.

---

## Prerequisites

Install **TRexSelector** from the main repository:

```bash
R CMD INSTALL --preclean <path-to-TRexSelector>/R/TRexSelector
# or, from R:
# devtools::install("<path-to-TRexSelector>/R/TRexSelector")
```

Additional CRAN packages, depending on which demos you run:

| Package | Needed by |
|---|---|
| `TRexSelector` (1.0.0) | the reference generator `demo_ts_compare_tlars_tlasso.R`, now in the TRexSelector library test suite (`TRexSelector/cpp/tests/validation/tsolvers/`) |
| `plotly` | Correlation-matrix heatmaps in the trex_da / trex_gvs Monte Carlo (MC) demos |
| `parallel` | Monte Carlo runners (shipped with base R) |
| `glmnet` | the model-selection and trex_spca cross-check probes, now in the TRexSelector library test suite (`TRexSelector/cpp/tests/validation/ml_methods/model_selection/` and `.../trex_selector_methods/ad/trex_spca/`) |
| `tlars` | the reference generator `demo_ts_compare_tlars_tlasso.R`, now in the TRexSelector library test suite (`TRexSelector/cpp/tests/validation/tsolvers/`) |

---

## Running the demos

Every demo is a standalone script and can be run from **any** working directory:

```bash
Rscript R/trex_selector_methods/ad/trex_core/demo_trex_01_single_run/demo_trex_01_single_run.R
```

Each script resolves its own location (via `sys.frame(1L)$ofile` with a
`--file=` fallback), so relative `source()` calls and output paths work under
both `Rscript` and interactive `source()`. Monte Carlo demos write their
results into a `simulation_results/` folder next to the demo file.

All supports and selected variables are **1-based** in R, while
the C++ and Python counterparts are 0-based. Demo headers state the conversion
explicitly where fixed supports are mirrored from C++.

---

## Contents

| Area | Status | What it covers | cpp counterpart |
|---|---|---|---|
| [memory_mapping/](memory_mapping/README.md) | New (TRexSelector 2.0.0) | `mem_matrix` / `mem_matrix_from` walkthrough: basics, out-of-core streaming, per-column seeded fill, element access (folder-per-demo) | [cpp/memory_mapping/](../cpp/memory_mapping/) |
| [ml_methods/hac_clustering/](ml_methods/hac_clustering/README.md) | New (TRexSelector 2.0.0) | Agglomerative hierarchical clustering: six in-memory scenarios (`demo_mlm_hac_01`) plus the out-of-core locality-sensitive hashing (LSH) linkage comparison (`demo_mlm_hac_02_mmap`), adjusted Rand index (ARI) evaluation | [cpp/ml_methods/hac_clustering/](../cpp/ml_methods/hac_clustering/) |
| [ml_methods/scaler_methods/](ml_methods/scaler_methods/README.md) | New (TRexSelector 2.0.0) | `MLZScoreScaler` / `MLLpNormScaler`: center/scale switches, inverse transform, serialization, mmap-backed scaling | [cpp/ml_methods/scaler_methods/](../cpp/ml_methods/scaler_methods/) |
| [ml_methods/pca/](ml_methods/pca/README.md) | New (TRexSelector 2.0.0) | `MLPCA` R6 class: fit, restore round-trip, out-of-sample transform | [cpp/ml_methods/pca/](../cpp/ml_methods/pca/) |
| [ml_methods/svd/](ml_methods/svd/README.md) | New (TRexSelector 2.0.0) | Truncated singular value decomposition (SVD) via `MLSVD` | [cpp/ml_methods/svd/](../cpp/ml_methods/svd/) |
| [ml_methods/ridge_regression/](ml_methods/ridge_regression/README.md) | New (TRexSelector 2.0.0) | L2-penalized least squares via `MLRidge` | [cpp/ml_methods/ridge_regression/](../cpp/ml_methods/ridge_regression/) |
| [ml_methods/model_selection/](ml_methods/model_selection/README.md) | New (TRexSelector 2.0.0) | K-fold cross-validation (CV) ridge via `MLRidgeCV` incl. a memory-mapped design matrix, plus elastic-net path fitting and CV via `MLElasticNet` / `MLElasticNetCV` | [cpp/ml_methods/model_selection/](../cpp/ml_methods/model_selection/) |
| [trex_selector_methods/ad/trex_core/](trex_selector_methods/ad/trex_core/README.md) | New (TRexSelector 2.0.0) | Classical T-Rex: single runs, MC studies, l-loop strategies, mmap, scalability (demos 01–08) | [cpp/trex_selector_methods/ad/trex_core/](../cpp/trex_selector_methods/ad/trex_core/) |
| [trex_selector_methods/ad/trex_da/](trex_selector_methods/ad/trex_da/README.md) | New (TRexSelector 2.0.0) | Dependency-Aware (DA) T-Rex: first-order autoregressive (AR(1)), equi, nearest-neighbor (NN), Binary-Tree (BT) block, heavy-tailed data-generating processes (DGPs) (demos 01–08) | [cpp/trex_selector_methods/ad/trex_da/](../cpp/trex_selector_methods/ad/trex_da/) |
| [trex_selector_methods/ad/trex_gvs/](trex_selector_methods/ad/trex_gvs/README.md) | New (TRexSelector 2.0.0) | Grouped Variable Selection T-Rex: the terminating elastic net (TENET), its augmented form TENET_AUG and the informed elastic net variant TIENET_AUG over grouped DGPs (demos 01–08) | [cpp/trex_selector_methods/ad/trex_gvs/](../cpp/trex_selector_methods/ad/trex_gvs/) |
| [trex_selector_methods/ad/trex_spca/](trex_selector_methods/ad/trex_spca/README.md) | New (TRexSelector 2.0.0) | T-Rex sparse principal component analysis (PCA) MC comparison vs. ordinary/oracle PCA (`TRexSPCASelector`, demos 01–02, folder-per-demo), C++/R cross-check probes moved to the TRexSelector library test suite (`TRexSelector/cpp/tests/validation/trex_selector_methods/ad/trex_spca/`) | [cpp/trex_selector_methods/ad/trex_spca/](../cpp/trex_selector_methods/ad/trex_spca/) |
| [trex_selector_methods/ad/trex_screening/](trex_selector_methods/ad/trex_screening/README.md) | New (TRexSelector 2.0.0) | Screen-TRex: in-memory / mmap screening, correlated designs, biobank-scale, solver comparison (demos 01–06) | [cpp/trex_selector_methods/ad/trex_screening/](../cpp/trex_selector_methods/ad/trex_screening/) |
| [tsolvers/](tsolvers/README.md) | New (TRexSelector 2.0.0) | 12 standalone terminating-solver demos (`demo_ts_01` … `demo_ts_12`: early stopping, external normalization, serialization, mmap, terminated solution paths with ggplot2 figures), the TENET/TENETAug equivalence check and CRAN tlars path-equivalence reference generator moved to the TRexSelector library test suite (`TRexSelector/cpp/tests/validation/tsolvers/`) | [cpp/tsolvers/](../cpp/tsolvers/) |

Shared helpers for the selector demos live in
[trex_selector_methods/](trex_selector_methods/README.md)
(`support_generators.R`, `simulation_utils.R`).

The `tsolvers/` and `trex_selector_methods/` folders follow the library layout by dummy type. `ad/` holds the
augmented-dummy (AD) demos, while `sd/` (sparse dummies) and `vd/` (virtual dummies) hold a README until the R bindings
expose these methods.

---

## Coverage relative to the C++ tree

The R tree now mirrors the selector suites — `trex_core`, `trex_da`, `trex_gvs`,
`trex_spca`, and `trex_screening` — on the new TRexSelector R6 API. The notes
below detail the coverage of a few parts relative to [cpp/](../cpp/README.md):

- **trex+DA+NN** — the R binding exposes `da_method = "NN"`, so the nearest-neighbor
  demo `demo_trex_da_07_mc_sim_nn` runs the NN method like its C++ counterpart and
  has committed results.
- **tsolvers** — all 12 per-solver demos and the four performance demos are mirrored
  (`demo_ts_01` … `demo_ts_12`, `demo_ts_perf_01` … `demo_ts_perf_04`). The validation programs
  (`validation_ts_01_tenet_aug_comparison` and the CRAN tlars cross-comparison
  generator `validation_ts_02_tlars_tlasso_rcompare`) moved to the TRexSelector
  library test suite (`TRexSelector/cpp/tests/validation/tsolvers/`).
- **ml_methods** is fully mirrored, folder-per-demo, under the C++ demo names
  (`scaler_methods/demo_mlm_scaler_01`, `pca/demo_mlm_pca_01`,
  `svd/demo_mlm_svd_01`, `ridge_regression/demo_mlm_ridge_01`,
  `model_selection/demo_mlm_ms_01_ridge_cv_svd` and `…_ms_02_enet_cv_ccd`,
  `hac_clustering/demo_mlm_hac_01` and `…_hac_02_mmap`). Both parts of the C++
  elastic-net demo are covered, the standalone path fit via `MLElasticNet` and
  K-fold CV via `MLElasticNetCV`. The memory-mapped subproblems bind the mapped
  buffer zero-copy exactly as in C++. The scalers and `MLRidgeCV` accept
  `mmap_matrix` handles directly (scalers need `"readwrite"` mode, `MLRidgeCV`
  also fits from `"readonly"` handles).
