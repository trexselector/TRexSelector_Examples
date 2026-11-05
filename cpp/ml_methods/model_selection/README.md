# Model Selection: Demonstration Suite

## Overview

This folder holds the C++ examples for model selection in the `ml_methods` module: choosing the penalty of a
regularized Gaussian regression by K-fold cross-validation (CV). In TRexSelector, these routines tune the ridge
penalty of Grouped Variable Selection (GVS). Both demos draw synthetic data from

```math
\boldsymbol{y} = \boldsymbol{X}\boldsymbol{\beta} + \boldsymbol{\varepsilon},
\qquad
\boldsymbol{\varepsilon} \sim \mathcal{N}(\boldsymbol{0}, \sigma^2 \boldsymbol{I}_n),
```

with ten active coefficients, and report the CV-optimal penalty $\lambda_{\min}$ together with the more heavily
regularized one-standard-error pick $\lambda_{1\mathrm{se}}$.

The folder has two demos, each in its own subfolder with the source file and a README that walks through the
console output. `demo_mlm_ms_01_ridge_cv_svd/` runs ridge CV with a singular value decomposition (SVD) per fold, from
$n > p$ to $p \gg n$ and on a memory-mapped design matrix, while `demo_mlm_ms_02_enet_cv_ccd/` fits elastic-net paths by
cyclic coordinate descent (CCD) and selects $\lambda$ for ridge, elastic net and lasso, including the ridge conversion
used by GVS. Both print to the console only and finish in under a minute. The Python and R packages have the same
demos (see [Python/ml_methods/model_selection/](../../../Python/ml_methods/model_selection/README.md) and
[R/ml_methods/model_selection/](../../../R/ml_methods/model_selection/README.md)).

---

## Demos

| Folder | Purpose |
| ------ | ------- |
| [`demo_mlm_ms_01_ridge_cv_svd/`](demo_mlm_ms_01_ridge_cv_svd/) | Ridge K-fold CV via `Eigen::JacobiSVD` in four regimes, including a memory-mapped design matrix and a grid-edge check |
| [`demo_mlm_ms_02_enet_cv_ccd/`](demo_mlm_ms_02_enet_cv_ccd/) | Elastic-net paths and K-fold CV via coordinate descent for ridge, elastic net and lasso, with the TRexGVS ridge conversion |

---

## Running

From the repository root, after `cmake --preset release && cmake --build --preset release-build` in `cpp/`:

```bash
cpp/build/release/bin/ml_methods/model_selection/demo_mlm_ms_01_ridge_cv_svd/demo_mlm_ms_01_ridge_cv_svd
cpp/build/release/bin/ml_methods/model_selection/demo_mlm_ms_02_enet_cv_ccd/demo_mlm_ms_02_enet_cv_ccd
```
