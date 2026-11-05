# TRexSelector_Examples — C++ Sources

## Overview

This directory contains the standalone C++ examples project. It consumes the
installed **TRexSelector** library via CMake's `find_package()` and covers the
full library stack: data-preprocessing utilities, T-Algorithm solvers, and T-Rex
selector variants. Demos that produce output files write them to a
`simulation_results/` folder next to their source, while the `ml_methods` and
`memory_mapping` demos print to the console.

---

## Prerequisites

Install the TRexSelector library before configuring this project:

```bash
cmake --build <TRexSelector-build-dir> --target install
```

The CMake presets in this directory expect the install tree at
`../../TRexSelector-beyond/cpp/install`. Adjust `CMAKE_PREFIX_PATH` in
`CMakePresets.json` if your TRexSelector installation is elsewhere.

---

## Build System

### Presets (`CMakePresets.json`)

| Configure preset | Build preset | Build type | Binary output directory |
| --- | --- | --- | --- |
| `debug` | `debug-build` | Debug (`-g -O0`) | `build/debug/bin/` |
| `release` | `release-build` | Release (`-O3 -march=native`) | `build/release/bin/` |

Quick start (run from this `cpp/` directory):

```bash
# Full debug build
cmake --preset debug
cmake --build --preset debug-build

# Full release build
cmake --preset release
cmake --build --preset release-build
```

The VS Code tasks **CMake Full Build (Debug/Release)** in
`.vscode/tasks.json` run the same two-step configure-then-build sequence.

### `add_demo_target` macro

All demo executables are registered through the
`add_demo_target(target_name source_file library_set)` macro in
[CMakeLists.txt](CMakeLists.txt). It:

- Creates the executable and links Eigen3, Open Multi-Processing (OpenMP), and a chosen library set
- Routes each binary into a mirrored sub-tree under `build/<type>/bin/`
  (e.g. `tsolvers/ad/demo_ts_01_tlars/demo_ts_01_tlars`)
- Injects `DEMO_OUTPUT_DIR` as a compile-time constant pointing to the
  source-side `simulation_results/` folder next to the demo source

### Library sets

| Variable | Linked TRexSelector components |
| --- | --- |
| `DATA_PREPROCESSING_LIBS` | `trex_ml_methods` |
| `T_ALGORITHM_LIBS` | `trex_ml_methods`, `trex_tsolvers` |
| `T_REX_SELECTOR_LIBS` | `trex_selector_methods` (transitively pulls in tsolvers, ml\_methods, utils) |

---

## Module Structure

### `memory_mapping/`

Memory-mapped matrix demos using disk-backed `Eigen::Map`. Demonstrates
out-of-core matrix creation, zero-copy read/write access, and OpenMP parallel
filling via Portable Operating System Interface (POSIX) `mmap`.

| Demo | Description |
| --- | --- |
| `demo_memory_mapping` | Create, access, parallel-fill, and clean up a memory-mapped matrix |

---

### `ml_methods/`

Data-preprocessing and numerical building blocks used by the solver and
selector layers.

| Demo | Description |
| --- | --- |
| `demo_mlm_scaler_01` | Column normalization — Z-score and Lp-norm scalers |
| `demo_mlm_pca_01` | Principal component analysis (PCA) |
| `demo_mlm_svd_01` | Singular value decomposition (SVD) |
| `demo_mlm_ridge_01` | Ridge regression (L2-penalized least squares) |
| `demo_mlm_ms_01_ridge_cv_svd` | Cross-validated ridge regression via SVD |
| `demo_mlm_ms_02_enet_cv_ccd` | Cross-validated elastic net via cyclic coordinate descent (CCD) |
| `demo_mlm_hac_01` | Hierarchical agglomerative clustering (HAC) |
| `demo_mlm_hac_02_mmap` | HAC on a memory-mapped design matrix |

Cross-language validation of HAC output against R now lives in the TRexSelector
library test suite (`TRexSelector/cpp/tests/validation/hac_clustering/`), run
on demand via its `run_validations` target.

---

### `tsolvers/`

Terminating T-Algorithm solvers for high-dimensional linear-model variable
selection. Each solver stops at $T_{\text{stop}}$ level rather than computing
the full regularization path. The demos live in `tsolvers/ad/` (augmented dummies), while
`tsolvers/sd/` (sparse dummies) and `tsolvers/vd/` (virtual dummies) are reserved for later demos.

| Demo | Solver |
| --- | --- |
| `demo_ts_01_tlars` | T-LARS (Least Angle Regression) |
| `demo_ts_02_tlasso` | T-LASSO |
| `demo_ts_03_tstepwise` | T-Stepwise (forward stepwise) |
| `demo_ts_04_tenet` | T-ENET (elastic net) |
| `demo_ts_05_tstagewise` | T-Stagewise (forward stagewise) |
| `demo_ts_06_tomp` | T-OMP (orthogonal matching pursuit) |
| `demo_ts_07_tgp` | T-GP (gradient pursuit) |
| `demo_ts_08_tacgp` | T-ACGP (approximate conjugate gradient pursuit) |
| `demo_ts_09_tmp` | T-MP (matching pursuit) |
| `demo_ts_10_tncgmp` | T-NCGMP (norm-corrective generalized matching pursuit) |
| `demo_ts_11_tools` | T-OLS (orthogonal least squares) |
| `demo_ts_12_tafs` | T-AFS (adaptive forward stepwise) |

Cross-language solver-equivalence checks (TLARS/TLASSO/TENET vs R, plus the
TENET/TENET_AUG probe) moved to the TRexSelector library test suite
(`TRexSelector/cpp/tests/validation/tsolvers/`).

---

### `trex_selector_methods/`

Five T-Rex selector variants, each in its own sub-directory of `ad/` (augmented dummies), while `sd/` (sparse
dummies) and `vd/` (virtual dummies) are reserved for later demos. Demos run Monte Carlo simulations and write
results to source-side `simulation_results/` folders.

#### `ad/trex_core/`

Classical T-Rex selector. Demos cover single runs, Monte Carlo simulations
with fixed and variable support, l-loop strategies, and memory-mapped designs
(10 demos).

#### `ad/trex_da/`

Dependency-Aware T-Rex. Demos explore first-order autoregressive (AR(1)), equicorrelated, nearest-neighbor,
block-diagonal, and group correlation structures (8 demos).

#### `ad/trex_gvs/`

Grouped Variable Selection T-Rex. Demos benchmark elastic net (EN), EN in augmented-lasso form (EN-AUG), and
informed elastic net (IEN) grouping strategies across Hastie, scattered, mixed, negative-trap, t-distributed, AR(1),
autoregressive moving-average (ARMA), and block-benchmark designs (8 demos).

#### `ad/trex_screening/`

Ultra-high-dimensional screening with T-Rex. Demos cover in-memory screening,
memory-mapped screening, correlated designs, and biobank-scale pipelines
(6 demos).

#### `ad/trex_spca/`

Sparse PCA with T-Rex. Monte Carlo simulation demo (1 demo).
