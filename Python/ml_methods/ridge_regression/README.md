# Ridge Regression: Demonstration Suite

## Overview

This folder holds the Python example for ridge regression in the `trexselector.ml_methods` module. Ridge
regression adds an $\ell_2$-penalty to least squares, which stabilizes the estimate and shrinks the coefficients
toward zero:

```math
\widehat{\boldsymbol{\beta}}(\lambda) =
\arg\min_{\boldsymbol{\beta} \in \mathbb{R}^p}
\lVert \boldsymbol{y} - \boldsymbol{X}\boldsymbol{\beta} \rVert_2^2
+
\lambda \lVert \boldsymbol{\beta} \rVert_2^2 .
```

The folder has one demo, `demo_mlm_ridge_01/`, with the script and a README that walks through the console output. It
runs `RidgeSolver` on the primal path for $n \ge p$, on the dual path for $n < p$ and over a sweep of $\lambda$, and
the Python version adds a monotone-shrinkage check to the sweep. The demo prints to the console only and finishes in
well under a second. Choosing $\lambda$ by cross-validation is covered in
[model_selection/](../model_selection/README.md). The C++ and R packages have the same demo in
[cpp/ml_methods/ridge_regression/](../../../cpp/ml_methods/ridge_regression/README.md) and
[R/ml_methods/ridge_regression/](../../../R/ml_methods/ridge_regression/README.md).

---

## Demos

| Folder | Purpose |
| ------ | ------- |
| [`demo_mlm_ridge_01/`](demo_mlm_ridge_01/) | Ridge regression on the primal ($n \ge p$) and dual ($n < p$) paths, plus a lambda sweep with a monotone-shrinkage check |

---

## Running

From the repository root, with TRexSelector installed in the repository virtual environment `.venv`:

```bash
.venv/bin/python Python/ml_methods/ridge_regression/demo_mlm_ridge_01/demo_mlm_ridge_01.py
```
