# Scaler Methods: Demonstration Suite

## Overview

This folder contains C++ examples for **column-wise scaling and normalization** utilities in the `ml_methods` module.

These preprocessing methods are used to standardize features before downstream tasks such as penalized regression, and
other numerical linear-algebra workflows. The current demo illustrates both variance-based standardization and
norm-based normalization.

---

## Key concepts

Given a data matrix $\boldsymbol{X} \in \mathbb{R}^{n \times p}$, scaling is applied column by column.

Two core transformations are demonstrated:

### Z-score scaling

```math
\tilde{x}_{ij} = \frac{x_{ij} - \bar{x}_j}{s_j},
```

where $\bar x_j$ is the column mean and $s_j$ is the column scale.

### Lp-norm scaling

```math
\tilde{\boldsymbol{x}}_j =
\frac{\boldsymbol{x}_j - \bar{x}_j \mathbf{1}}
{\lVert \boldsymbol{x}_j - \bar{x}_j \mathbf{1} \rVert_p},
```

when centering is enabled.

---

## Start here

| Folder | Purpose |
| ------ | ------- |
| [demo_mlm_scaler_01/](demo_mlm_scaler_01/README.md) | Demonstrates `ZScoreScaler` and `LpNormScaler`, including inverse transforms, serialization, and memory-mapped usage |

---

## Running

```bash
./build/debug/bin/ml_methods/scaler_methods/demo_mlm_scaler_01/demo_mlm_scaler_01
```
