# ML Methods: Demos

## Overview

This folder contains the Python demos for the `ml_methods` layer of the TRexSelector library, used through the
`trexselector.ml_methods` module of the TRexSelector package.

The layer provides the numerical and preprocessing components that support the higher-level workflows of the
Terminating-Random Experiments (T-Rex) selector, including column scaling, singular value decomposition (SVD),
principal component analysis (PCA), ridge regression, model parameter selection based on k-fold cross-validation, and
hierarchical agglomerative clustering (HAC).

Each method family has its own subfolder. Demo scripts illustrate usage and interface behavior. Cross-language
correctness checks against R references live in the TRexSelector library's own test suite (`cpp/tests/validation/`),
run on demand via its `run_validations` target, rather than in this examples package.

The demos mirror the C++ suite in [cpp/ml_methods/](../../cpp/ml_methods/README.md) with the same demo names,
subproblems and console layout, and they replace the Eigen matrices with Fortran-ordered `float64` NumPy arrays. The
HAC demos and the memory-mapped part of the scaler demo scale the C++ problem sizes down, as each script's header
comment notes. A few demos add checks that the C++ versions lack, such as the full-rank reconstruction in the SVD demo
and the monotone shrinkage in the ridge demo.

---

## Category overview

| Category | Folder | Demos | Purpose |
| -------- | ------ | ----- | ------- |
| **Scaler Methods** | [scaler_methods/](scaler_methods/README.md) | `demo_mlm_scaler_01` | Column normalization utilities, including Z-score and Lp-norm scaling |
| **PCA** | [pca/](pca/README.md) | `demo_mlm_pca_01` | Principal component analysis |
| **SVD** | [svd/](svd/README.md) | `demo_mlm_svd_01` | Singular value decomposition for stable matrix computations |
| **Ridge Regression** | [ridge_regression/](ridge_regression/README.md) | `demo_mlm_ridge_01` | L2-penalized least-squares regression |
| **Model Selection** | [model_selection/](model_selection/README.md) | `demo_mlm_ms_01_ridge_cv_svd`, `demo_mlm_ms_02_enet_cv_ccd` | Cross-validation workflows for regularized regression methods |
| **HAC Clustering** | [hac_clustering/](hac_clustering/README.md) | `demo_mlm_hac_01`, `demo_mlm_hac_02_mmap` | Hierarchical agglomerative clustering, including an out-of-core memory-mapped workflow |

---

## Architecture context

These components form the `ml_methods` layer in the project structure:

```txt
utils -> ml_methods -> tsolvers -> trex_selector_methods -> demos
```

In particular:

- scaler methods support feature normalization before downstream modeling,
- SVD and ridge-related tools support stable regression and dimensionality-reduction workflows,
- HAC clustering supports grouped-variable workflows such as grouped variable selection (GVS).

---

## Installation

The demos are plain scripts without a build step. They need TRexSelector 1.0.0 in the repository virtual
environment `.venv`, installed as described in the [Python README](../README.md):

```bash
python3 -m venv .venv
.venv/bin/pip install <path-to-TRexSelector>/Python/trexselector
```

---

## Running

Run any demo from the repository root with the interpreter of the virtual environment. For example:

```bash
.venv/bin/python Python/ml_methods/pca/demo_mlm_pca_01/demo_mlm_pca_01.py
```

---

## Folder contents

```txt
ml_methods/
  ├── README.md
  ├── scaler_methods/
  ├── pca/
  ├── svd/
  ├── ridge_regression/
  ├── model_selection/
  └── hac_clustering/
```
