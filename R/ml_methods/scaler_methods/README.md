# Scaler Methods: Demonstration Suite

## Overview

This folder holds the R example for column-wise scaling and normalization in the TRexSelector package, where the
R6 classes `MLZScoreScaler` and `MLLpNormScaler` wrap the C++ `ZScoreScaler` and `LpNormScaler`. Both scalers
subtract a center $c_j$ from every column of $\boldsymbol{X} \in \mathbb{R}^{n \times p}$ and divide it by a scale
$s_j$,

```math
\tilde{\boldsymbol{x}}_j = \frac{\boldsymbol{x}_j - c_j \mathbf{1}}{s_j},
```

where $c_j$ is the column mean with centering on and 0 with centering off. `MLZScoreScaler` divides by the root mean
square around that center, which is the standard deviation with centering on, and `MLLpNormScaler` divides by the
$L_2$- or $L_1$-norm of the centered column. The preprocessing prepares the features for penalized regression and
other numerical linear-algebra workflows.

The folder has one demo, `demo_mlm_scaler_01/`, with the script and a README that walks through the console output.
It compares the `center` and `scale` switches of both scalers, checks the inverse transform, saves and reloads a
fitted scaler and scales a memory-mapped matrix. The demo prints to the console only and finishes in about a second.
The C++ and Python packages have the same demo in
[cpp/ml_methods/scaler_methods/](../../../cpp/ml_methods/scaler_methods/README.md) and
[Python/ml_methods/scaler_methods/](../../../Python/ml_methods/scaler_methods/README.md).

---

## Demos

| Folder | Purpose |
| ------ | ------- |
| [`demo_mlm_scaler_01/`](demo_mlm_scaler_01/) | `MLZScoreScaler` and `MLLpNormScaler` under the center and scale switches, with inverse transforms, serialization and memory-mapped usage |

---

## Running

From the repository root, with the TRexSelector R package installed:

```bash
Rscript R/ml_methods/scaler_methods/demo_mlm_scaler_01/demo_mlm_scaler_01.R
```
