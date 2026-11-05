# Ridge Regression: Demonstration Suite

## Overview

This folder holds the C++ example for ridge regression in the `ml_methods` module. Ridge regression adds an
$\ell_2$-penalty to least squares, which stabilizes the estimate and shrinks the coefficients toward zero:

```math
\widehat{\boldsymbol{\beta}}(\lambda) =
\arg\min_{\boldsymbol{\beta} \in \mathbb{R}^p}
\lVert \boldsymbol{y} - \boldsymbol{X}\boldsymbol{\beta} \rVert_2^2
+
\lambda \lVert \boldsymbol{\beta} \rVert_2^2 .
```

The folder has one demo, `demo_mlm_ridge_01/`, with the source file and a README that walks through the console
output. It runs `RidgeSolver` on the primal path for $n \ge p$, on the dual path for $n < p$ and over a sweep of
$\lambda$. The demo prints to the console only and finishes in under a second. Choosing $\lambda$ by
cross-validation is covered in [model_selection/](../model_selection/README.md). The Python and R packages have the
same demo in [Python/ml_methods/ridge_regression/](../../../Python/ml_methods/ridge_regression/README.md) and
[R/ml_methods/ridge_regression/](../../../R/ml_methods/ridge_regression/README.md).

---

## Demos

| Folder | Purpose |
| ------ | ------- |
| [`demo_mlm_ridge_01/`](demo_mlm_ridge_01/) | Ridge regression on the primal ($n \ge p$) and dual ($n < p$) paths, plus a lambda sweep |

---

## Running

From the repository root, after `cmake --preset release && cmake --build --preset release-build` in `cpp/`:

```bash
cpp/build/release/bin/ml_methods/ridge_regression/demo_mlm_ridge_01/demo_mlm_ridge_01
```
