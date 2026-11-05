# T-Rex Selector Demos and Simulations

This repository contains runnable examples, demos, and Monte Carlo simulation studies for the
[TRexSelector](https://github.com/G4Lactus/TRexSelector) software — a high-performance C++ implementation of the T-Rex
selector family with R and Python bindings. The examples are organized per language and are
intended as the first stop for new users.
Each demo is a small, self-contained script or program that generates data, runs a selector
(or building block), and explains the output.

---

## About the T-Rex Selector

The T-Rex (Terminating-Random EXperiments) selector performs fast variable selection with false discovery rate
(FDR) control in high-dimensional regression problems (Machkour, Muma, and Palomar, 2022+). It runs
K random experiments in which the design matrix is augmented with computer-generated dummy
variables and a terminating solver (e.g. T-LARS) stops early once enough dummies enter the path.
A calibration procedure then selects a voting threshold that provably controls the false
discovery rate at a user-chosen target level while maximizing the number of selected variables.

The software implements the classical T-Rex selector plus several extensions:

- **T-Rex DA** — dependency-aware (DA) variants for correlated designs, with first-order autoregressive (AR(1)),
  equicorrelation, nearest-neighbor, and binary-tree/dendrogram corrections.
- **T-Rex GVS** — grouped variable selection (GVS) with elastic-net-type solvers and
  clustering-informed dummies.
- **T-Rex Screening / Biobank Screening** — ultra-high-dimensional screening workflows.
- **T-Rex SPCA** — sparse principal component analysis (SPCA) with FDR control on the loadings.
- **Terminating solvers (tsolvers)** — T-LARS, T-LASSO, T-ENET, matching-pursuit and stepwise
  families, all with early termination at a dummy-count threshold.
- **ML methods** — the numerical machine learning (ML) building blocks (scalers, principal component analysis (PCA),
  singular value decomposition (SVD), ridge, elastic-net cross-validation (CV), hierarchical agglomerative clustering)
  and disk-backed memory-mapped matrices for problems larger than the random-access memory (RAM).

---

## Contents

| Folder | Language | Status | Description |
|---|---|---|---|
| [cpp/](cpp/) | C++20 | reference | All selector variants, all terminating solvers, ML building blocks, memory mapping, and validation programs. Has its own CMake build (see [cpp/README.md](cpp/README.md)). |
| [R/](R/) | R ≥ 4.3 | mirrors cpp | Demos for the `TRexSelector` R package with its R6 application programming interface (API): all selector suites (trex_core, trex_da, trex_gvs, trex_spca, trex_screening), all terminating-solver demos (`tsolvers/`) + validation, memory mapping, and the full ml_methods set. Version 2.0.0 replaces the pure-R 1.0.0 of the Comprehensive R Archive Network (CRAN) in place, and no demo depends on the pure-R package any more. See [R/README.md](R/README.md). |
| [Python/](Python/) | Python ≥ 3.10 | mirrors cpp | Demos for the `trexselector` Python package (`import trexselector`): all selector suites (trex_core, trex_da, trex_gvs, trex_spca, trex_screening), all terminating-solver demos (`tsolvers/`) + validation, memory mapping, and the full ml_methods set. Only the `ml_methods/validation/` reference programs are not ported. See [Python/README.md](Python/README.md). |

Demo naming follows the C++ convention `demo_<area>_NN_<topic>`.
R and Python demos refer to their C++ counterpart in the file header, which simplifies cross-checking.
Inside each language folder, `tsolvers/` and `trex_selector_methods/` follow the library layout by dummy type.
`ad/` holds the augmented-dummy (AD) demos, and `sd/` (sparse dummies) and `vd/` (virtual dummies) are reserved
for later demos.

---

## Overview of supported Languages and Packages

All three languages are backed by the same C++ core, so results agree across languages up to the
documented random number generator (RNG) and indexing conventions (C++/Python are 0-based, R is 1-based).

### R Package

`TRexSelector` (R ≥ 4.3, Rcpp/RcppEigen, R6 classes). Install from the main repo:

```sh
R CMD INSTALL <path-to>/TRexSelector/R/TRexSelector
```

Typical usage: `TRexSelector$new(X, y, tFDR = 0.1, control = trex_control(...))$select()`.
See [R/README.md](R/README.md) for more details.

### Python Package

`trexselector` (Python ≥ 3.10, pybind11/scikit-build-core), distributed on PyPI and imported under that name:

```sh
pip install <path-to>/TRexSelector/Python/trexselector
```

Typical usage: `trexselector.TRexSelector(X, y, tFDR=0.1, trex_control=ctrl).select()`.

### Cpp Package

The C++ library `TRexSelector` is consumed via CMake (`find_package(TRexSelector REQUIRED)`).
The [cpp/](cpp/) folder is a standalone CMake project. Build it with the provided presets
(see [cpp/README.md](cpp/README.md)) and run the demo executables from `bin/`.
