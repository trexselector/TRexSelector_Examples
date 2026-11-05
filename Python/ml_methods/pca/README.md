# PCA: Demonstration Suite

## Overview

This folder holds the Python example for principal component analysis (PCA) in the `trexselector.ml_methods`
module. PCA projects a data matrix $\boldsymbol{X} \in \mathbb{R}^{n \times p}$ onto $M \le \min(n, p)$ orthogonal
directions of largest variance,

```math
\boldsymbol{X} \approx \boldsymbol{Z}\boldsymbol{V}^\top,
\qquad
\boldsymbol{Z} \in \mathbb{R}^{n \times M},\;
\boldsymbol{V} \in \mathbb{R}^{p \times M},
```

and serves in TRexSelector as a dimensionality-reduction and preprocessing step for high-dimensional data.

The folder has one demo, `demo_mlm_pca_01/`, with the script and a README that walks through the console output. It
fits PCA and reports the explained variance (EV) as a share of the total, restores the input after the in-place
preprocessing and transforms new data. The demo prints to the console only and finishes in well under a second. The
C++ and R packages have the same demo in [cpp/ml_methods/pca/](../../../cpp/ml_methods/pca/README.md) and
[R/ml_methods/pca/](../../../R/ml_methods/pca/README.md).

---

## Demos

| Folder | Purpose |
| ------ | ------- |
| [`demo_mlm_pca_01/`](demo_mlm_pca_01/) | PCA fit with EV, restore round-trip and transform of new data |

---

## Running

From the repository root, with TRexSelector installed in the repository virtual environment `.venv`:

```bash
.venv/bin/python Python/ml_methods/pca/demo_mlm_pca_01/demo_mlm_pca_01.py
```
