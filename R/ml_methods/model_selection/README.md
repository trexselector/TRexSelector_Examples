# Model Selection: Demonstration Suite

## Overview

This folder holds the R examples for model selection in the TRexSelector R package, which choose the penalty of a
regularized Gaussian regression by K-fold cross-validation (CV). In TRexSelector, these routines tune the ridge penalty
of Grouped Variable Selection (GVS). Both demos draw synthetic data from

```math
\boldsymbol{y} = \boldsymbol{X}\boldsymbol{\beta} + \boldsymbol{\varepsilon},
\qquad
\boldsymbol{\varepsilon} \sim \mathcal{N}(\boldsymbol{0}, \sigma^2 \boldsymbol{I}_n),
```

with ten active coefficients, and report the CV-optimal penalty $\lambda_{\min}$ together with the more heavily
regularized one-standard-error pick $\lambda_{1\mathrm{se}}$.

The folder has two demos, each in its own subfolder with the script and a README that walks through the console
output. `demo_mlm_ms_01_ridge_cv_svd/` runs ridge CV with `MLRidgeCV`, which takes a singular value decomposition
(SVD) per fold, from $n > p$ to $p \gg n$ and on a memory-mapped design matrix. In contrast,
`demo_mlm_ms_02_enet_cv_ccd/` fits elastic-net paths with `MLElasticNet` by cyclic coordinate descent (CCD) and
selects $\lambda$ with `MLElasticNetCV` for ridge, elastic net and lasso, including the ridge conversion used by GVS.
Both print to the console only and finish in under a minute. The C++ and Python examples have the same demos (see
[cpp/ml_methods/model_selection/](../../../cpp/ml_methods/model_selection/README.md) and
[Python/ml_methods/model_selection/](../../../Python/ml_methods/model_selection/README.md)).

---

## Demos

| Folder | Purpose |
| ------ | ------- |
| [`demo_mlm_ms_01_ridge_cv_svd/`](demo_mlm_ms_01_ridge_cv_svd/) | Ridge K-fold CV with `MLRidgeCV` (JacobiSVD per fold) in four regimes, including a memory-mapped design matrix and a grid-edge check |
| [`demo_mlm_ms_02_enet_cv_ccd/`](demo_mlm_ms_02_enet_cv_ccd/) | Elastic-net paths with `MLElasticNet` and K-fold CV with `MLElasticNetCV` for ridge, elastic net and lasso, with the TRexGVS ridge conversion |

---

## Running

From the repository root, with the TRexSelector R package installed as described in
[`LIBRARY_TARGET.txt`](../../../LIBRARY_TARGET.txt):

```bash
Rscript R/ml_methods/model_selection/demo_mlm_ms_01_ridge_cv_svd/demo_mlm_ms_01_ridge_cv_svd.R
Rscript R/ml_methods/model_selection/demo_mlm_ms_02_enet_cv_ccd/demo_mlm_ms_02_enet_cv_ccd.R
```

The classes take ordinary numeric R matrices, which are column-major like the library's Eigen matrices, so no
conversion is needed. Grid indices such as `$index_min()` are 1-based, the R convention, while the C++ and Python
accessors are 0-based. Both scripts assert their invariants with `stopifnot()`, so a violated one stops the script. The
data seeds match the C++ demos, but R's Mersenne-Twister generator draws other numbers than `std::mt19937`, so the
results agree with C++ qualitatively, not bitwise.
