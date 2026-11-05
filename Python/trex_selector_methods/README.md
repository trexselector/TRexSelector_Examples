# T-Rex Selector Methods: Demos

## Overview

This folder contains the Python demos for all **T-Rex selector variants**, built on the `trexselector` package
(the pybind11 binding layer over the C++ `tsolvers` and `ml_methods` stack).

Each variant addresses a different variable-selection scenario — the classical baseline, dependent/correlated designs,
grouped-variable structures, ultra-high-dimensional screening, and sparse principal component analysis (PCA) — while
sharing the same core idea of controlling the false discovery rate (FDR) with the Terminating Random Experiments
(T-Rex) framework. The demos mirror their C++ counterparts in
[cpp/trex_selector_methods/](../../cpp/trex_selector_methods/): same data-generating processes, same control
parameters, same result summaries, following the same **folder-per-demo** layout (one subfolder per demo, each with
its own `README.md` and `simulation_results/`, plus suite-level shared modules).

The C++ tree's `validation/` programs (cross-checks against R reference outputs) are validation infrastructure rather
than demos and have no Python port by design. They live in the TRexSelector library test suite.

The folders follow the layout of the library, which sorts the selectors by the way they handle the dummies. [ad/](ad/)
holds the variants with augmented dummies (AD), which append an explicit dummy matrix to the predictors, and all current
demos. [sd/](sd/README.md) is reserved for the sparse-dummy (SD) selector and [vd/](vd/README.md) for the virtual-dummy
(VD) selector.

---

## Category overview

The table uses the short names of the variants. T-Rex+GVS is the T-Rex selector for Grouped Variable Selection (GVS),
and T-Rex SPCA is its sparse PCA (SPCA) variant.

| Category | Folder | Purpose |
|----------|--------|---------|
| **Classical T-Rex** | [ad/trex_core/](ad/trex_core/README.md) | Terminating Random Experiments Selector — baseline FDR-controlled variable selection in low- and high-dimensional Gaussian linear models (demos 01–08) |
| **Dependency-Aware T-Rex (DA-TRex)** | [ad/trex_da/](ad/trex_da/README.md) | Variable selection under correlated/dependent design structures: first-order autoregressive (AR(1)) Toeplitz, equicorrelated, banded nearest-neighbor, and hierarchical block-covariance designs (demos 01–08) |
| **Grouped Variable Selection (T-Rex+GVS)** | [ad/trex_gvs/](ad/trex_gvs/README.md) | Group-structured selection via the elastic net (EN), the elastic net in augmented-lasso form (EN-AUG) and the informed elastic net (IEN) based on hierarchical agglomerative clustering (HAC) across equicorrelated, scattered, and block-covariance patterns (demos 01–08) |
| **Screening (Screen-TRex)** | [ad/trex_screening/](ad/trex_screening/README.md) | Ultra-high-dimensional pre-screening ahead of T-Rex selection, with in-memory, memory-mapped, and biobank-scale variants (demos 01–06) |
| **Sparse PCA (T-Rex SPCA)** | [ad/trex_spca/](ad/trex_spca/README.md) | Sparse PCA via per-component T-Rex+GVS selection on plug-in principal components (PCs), benchmarked against ordinary PCA and oracle-thresholded baselines: Fig.-2/Fig.-3-style support-recovery and Definition-1 proportion of explained variance (PEV) studies, plus mechanism studies of the plug-in construction (plug-in vs. oracle response, union-support FDR per PC) (demos 01–02) |

---

## Architecture context

The demos exercise the `trexselector` selector layer, which sits on top of the solver and preprocessing layers:

```txt
utils -> ml_methods -> tsolvers -> trex_selector_methods -> demos
```

In particular:

- each selector variant dispatches to one or more base solvers from `tsolvers` (e.g. TLARS, TLASSO, TENET) within
  its T-loop,
- variants that rely on preprocessing reuse components from `ml_methods`, for example HAC clustering for group
  construction in `trex_gvs`, or scaling utilities shared across variants,
- `trex_screening` and the memory-mapped demos in `trex_core` build on the memory-mapping utilities for large design
  matrices.

All selector-layer classes and enums are accessed through the package root (`import trexselector as trex`).
See the [import conventions](../README.md#import-conventions) for details.

---

## Running

Each demo is a plain script, run directly with the Python interpreter:

```bash
python ad/trex_core/demo_trex_01_single_run/demo_trex_01_single_run.py
```

For each variant's demo lineup and required parameters, see the corresponding subfolder README.

To run the Monte Carlo demos back to back as an unattended job, use the batch runner, which runs every suite's
`run_overnight.sh` (or the ones named as arguments) on the repository's `.venv`:

```bash
nohup Python/trex_selector_methods/run_all_overnight.sh > /dev/null 2>&1 &
```

Both levels are resumable. A demo that finishes is recorded under the run tag (the pinned library commit of
`LIBRARY_TARGET.txt`) and skipped by the next run, and the demo in flight keeps every finished grid cell in
`simulation_results/cache/cells.tsv` ([cell_cache.py](cell_cache.py)), so a stopped batch continues where it
stopped. `FRESH=1` reruns everything.

---

## Folder contents

```txt
trex_selector_methods/
  ├── README.md
  ├── cell_cache.py           # resumable grid cells of the Monte Carlo demos
  ├── run_all_overnight.sh    # batch runner over the suites' run_overnight.sh
  ├── ad/   # augmented dummies (AD)
  │   ├── trex_core/
  │   ├── trex_screening/
  │   ├── trex_da/
  │   ├── trex_gvs/
  │   └── trex_spca/
  ├── sd/   # sparse dummies (SD), README only
  └── vd/   # virtual dummies (VD), README only
```
