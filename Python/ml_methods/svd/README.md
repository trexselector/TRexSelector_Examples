# SVD: Demonstration Suite

## Overview

This folder holds the Python example for the singular value decomposition (SVD) in the
`trexselector.ml_methods` module. For $\mathbf{X} \in \mathbb{R}^{n \times p}$, the truncated SVD keeps the
leading $M$ singular components,

```math
\mathbf{X}_M = \mathbf{U}_M \boldsymbol{\Sigma}_M \mathbf{V}_M^\top,
```

which form the best approximation of rank $M$ in the Frobenius norm. In TRexSelector, SVD underlies principal
component analysis (PCA) and the numerically stable ridge cross-validation.

The folder has one demo, `demo_mlm_svd_01/`, with the script and a README that walks through the console output. It
runs `SVDSolver` on the direct path for $n \ge p$ and on the Gram path for $p > 2n$, checks each truncation error
against the Eckart-Young optimum and tests the singular vectors for orthogonality. As a Python addition, it also checks
the full-rank reconstruction. The demo prints to the console only and finishes in well under a second. The C++ and R
packages have the same demo in [cpp/ml_methods/svd/](../../../cpp/ml_methods/svd/README.md) and
[R/ml_methods/svd/](../../../R/ml_methods/svd/README.md).

---

## Demos

| Folder | Purpose |
| ------ | ------- |
| [`demo_mlm_svd_01/`](demo_mlm_svd_01/) | Truncated SVD on the direct and Gram paths with Eckart-Young checks, a full-rank reconstruction check and an orthogonality check |

---

## Running

From the repository root, with TRexSelector installed in the repository virtual environment `.venv`:

```bash
.venv/bin/python Python/ml_methods/svd/demo_mlm_svd_01/demo_mlm_svd_01.py
```
