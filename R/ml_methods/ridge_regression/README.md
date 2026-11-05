# Ridge Regression: Demonstration Suite

## Overview

This folder holds the R example for ridge regression in the TRexSelector package, where the R6 class `MLRidge`
wraps the C++ `RidgeSolver`. Ridge regression adds an $\ell_2$-penalty to least squares, which stabilizes the estimate
and shrinks the coefficients toward zero:

```math
\widehat{\boldsymbol{\beta}}(\lambda) =
\arg\min_{\boldsymbol{\beta} \in \mathbb{R}^p}
\lVert \boldsymbol{y} - \boldsymbol{X}\boldsymbol{\beta} \rVert_2^2
+
\lambda \lVert \boldsymbol{\beta} \rVert_2^2 .
```

The folder has one demo, `demo_mlm_ridge_01/`, with the script and a README that walks through the console output. It
runs `MLRidge` on the primal path for $n \ge p$, on the dual path for $n < p$ and over a sweep of $\lambda$. Like the
Python version, the R version adds a monotone-shrinkage check to the sweep, and it also checks the coefficient error of
the primal fit. The demo prints to the console only and finishes in about a second. Choosing $\lambda$ by
cross-validation is covered in [model_selection/](../model_selection/README.md). The C++ and Python packages have the
same demo in [cpp/ml_methods/ridge_regression/](../../../cpp/ml_methods/ridge_regression/README.md) and
[Python/ml_methods/ridge_regression/](../../../Python/ml_methods/ridge_regression/README.md).

---

## Demos

| Folder | Purpose |
| ------ | ------- |
| [`demo_mlm_ridge_01/`](demo_mlm_ridge_01/) | Ridge regression on the primal ($n \ge p$) and dual ($n < p$) paths, plus a lambda sweep with a monotone-shrinkage check |

---

## Running

From the repository root, with the TRexSelector R package installed:

```bash
Rscript R/ml_methods/ridge_regression/demo_mlm_ridge_01/demo_mlm_ridge_01.R
```
