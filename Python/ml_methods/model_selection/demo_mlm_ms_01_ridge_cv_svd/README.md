# Demo: Ridge Regression K-Fold CV via JacobiSVD

## Purpose

This demo shows `RidgeCV` from `trexselector.ml_methods`, the Python binding of the library's `ridge_cv_svd`.
It is the K-fold cross-validation (CV) routine for ridge regression that solves every fold with a singular value
decomposition (SVD), and it selects the penalty $\lambda$ from the cross-validated prediction error in four synthetic
Gaussian settings:

1. **Low-dimensional**: $n = 1{,}000$, $p = 500$, 10-fold CV.
2. **High-dimensional**: $n = 300$, $p = 500$, 10-fold CV.
3. **Very high-dimensional**: $n = 300$, $p = 1{,}000$, 5-fold CV.
4. **Memory-mapped dataset**: $n = 5{,}000$, $p = 500$, 10-fold CV, the design matrix stored on disk.

The same workflow is implemented in
[C++](../../../../cpp/ml_methods/model_selection/demo_mlm_ms_01_ridge_cv_svd/) and
[R](../../../../R/ml_methods/model_selection/demo_mlm_ms_01_ridge_cv_svd/).

---

## Statistical model and notation

Let

- $\boldsymbol{X} \in \mathbb{R}^{n \times p}$ denote the design matrix,
- $\boldsymbol{y} \in \mathbb{R}^n$ the response vector,
- $\boldsymbol{\beta} \in \mathbb{R}^p$ the regression coefficients,
- $\lambda \ge 0$ the ridge penalty parameter.

The synthetic data follow the Gaussian linear model

```math
\boldsymbol{y} = \boldsymbol{X}\boldsymbol{\beta}_{\mathrm{true}} + \boldsymbol{\varepsilon},
\qquad
\boldsymbol{\varepsilon} \sim \mathcal{N}(\boldsymbol{0}, \sigma^2 \boldsymbol{I}_n),
\qquad
\sigma = 1.
```

The entries of $\boldsymbol{X}$ are independent standard normal and are drawn into a Fortran-ordered `float64` array,
the layout `RidgeCV.fit()` requires. Ten entries of $\boldsymbol\beta_{\mathrm{true}}$ are set to 1 at evenly spaced
coordinates, the rest are zero, so $\lVert \boldsymbol\beta_{\mathrm{true}} \rVert_2^2 = 10$ and the response variance
is about $10 + \sigma^2 = 11$. That is the prediction error of the null model, which predicts the mean only.

---

## Ridge regression objective

For a given penalty parameter $\lambda$, ridge regression estimates the coefficients by solving

```math
\widehat{\boldsymbol{\beta}}(\lambda)
=
\underset{\boldsymbol{\beta} \in \mathbb{R}^p}{\arg\min}
\lVert \boldsymbol{y} - \boldsymbol{X}\boldsymbol{\beta} \rVert_2^2
+ \lambda \lVert \boldsymbol{\beta} \rVert_2^2
.
```

`RidgeCV` centers every training fold and scales its columns to unit $\ell_2$-norm before the solve, so $\lambda$
lives on that unit-norm scale. The fold solve runs in the compiled library with `Eigen::JacobiSVD`.

---

## Cross-validation criterion

The data are split into $K$ folds. For every $\lambda$ on the grid the model is fit on $K-1$ folds and the prediction
error is measured on the held-out fold. Averaging over the folds gives the curve $\mathrm{CV\text{-}MSE}(\lambda)$ of
the cross-validated mean squared error (MSE), with a standard error (SE) per grid point. The demo reports

- $\lambda_{\min}$: the value minimizing the CV-MSE,
- $\lambda_{1\mathrm{se}}$: the largest value of $\lambda$ whose CV-MSE lies within one SE of the minimum.

The grid holds 100 values, geometrically spaced from
$\lambda_{\max} = \max_j \lvert \boldsymbol x_{s,j}^\top \boldsymbol y_c \rvert$, the largest absolute inner
product between a centered unit-norm column and the centered response, down to $\lambda_{\max}/1{,}000$. Scenario 4
widens the range to $\lambda_{\max}/100{,}000$. Two checks close every scenario: the ordering
$\lambda_{\min} \le \lambda_{1\mathrm{se}}$, and the grid-edge check `selection_at_grid_edge()`, which fails
when a pick lands on the first or last grid point. Such a pick is the end of the grid, not a minimum.

A reference value for $\lambda_{\min}$ comes from ridge theory. For an isotropic Gaussian design, the penalty that
minimizes the prediction error is about

```math
\lambda^\ast \approx \frac{p\,\sigma^2}{n_{\mathrm{train}}\,\lVert \boldsymbol{\beta}_{\mathrm{true}} \rVert_2^2}
```

on the unit-norm scale, with $n_{\mathrm{train}}$ the number of training rows per fold. The walk-through compares the
CV picks with it.

---

## Running the demo

From the repository root, with the repository's virtual environment `.venv` set up as described in
[`LIBRARY_TARGET.txt`](../../../../LIBRARY_TARGET.txt):

```bash
.venv/bin/python Python/ml_methods/model_selection/demo_mlm_ms_01_ridge_cv_svd/demo_mlm_ms_01_ridge_cv_svd.py
```

The demo prints to the console only and takes about 40 seconds. The output below is a run against library b95b984,
with the package TRexSelector 1.0.0 built from that commit. The data come from NumPy's permuted congruential
generator (PCG64) via `numpy.random.default_rng`, with the same data seeds as the C++ demo. The C++ demo draws from
`std::mt19937`, so the two languages see different data, and their numbers agree qualitatively, not bitwise.

---

## Console walk-through

The run opens with a banner for the suite and closes with `All ridge_cv_svd demos complete.`, and every scenario
prints its own block. The CV-MSE curve lists every tenth grid point in increasing $\lambda$, followed by the rows of
the two picks.

### Scenario 1: Low-dimensional ($n > p$)

With $n = 1{,}000 > p = 500$, the problem is overdetermined and the penalty only adds shrinkage. Each fold trains on
900 rows, so the reference value is $\lambda^\ast \approx 500/(900 \cdot 10) = 0.056$, and CV picks
$\lambda_{\min} = 0.0603$ (C++ picks 0.0551 on its own draw). The CV-MSE of 2.16 is more than twice the noise
variance, the price of estimating 500 coefficients from 900 rows. $\lambda_{1\mathrm{se}} = 0.1130$ nearly doubles
the penalty for an error of 2.23, still within one SE of the minimum.

```text
==============================================================================
 Demo MS-01 | ridge_cv_svd | Low-Dimensional (n=1000, p=500, 10-fold)
==============================================================================

Data: n=1000, p=500, folds=10, active coefs=10

  lambda.min : 0.0603  (CV-MSE=2.1580 +/- 0.0906)
  lambda.1se : 0.1130  (CV-MSE=2.2330 +/- 0.0955)

  CV-MSE curve (every 10 steps):
    lambda        CV-MSE      SEM         marker
    0.0397        2.1727      0.0885
    0.0797        2.1721      0.0925
    0.1602        2.3611      0.0994
    0.3218        2.8976      0.1124
    0.6466        3.8975      0.1379
    1.2992        5.3422      0.1789
    2.6103        7.0117      0.2319
    5.2447        8.5587      0.2857
    10.5379       9.7273      0.3287
    21.1731       10.4776     0.3571
    0.0603        2.1580      0.0906        <-- min
    0.1130        2.2330      0.0955        <-- 1se

  Ordering check (lambda.min <= lambda.1se): PASS
  Grid-edge check (both picks inside the grid): PASS
```

### Scenario 2: High-dimensional ($p > n$)

With $p = 500 > n = 300$, the unpenalized least-squares problem has no unique solution, and the penalty makes it well
posed. Ridge spreads the fit over all 500 coefficients and cannot single out the ten active ones from 270 training
rows. The minimum CV-MSE of 6.11 removes about half of the signal variance. The null model errs by about 11,
and the noise floor is 1. The reference value 0.19 lies below $\lambda_{\min} = 0.2967$, but the curve is flat there.
At $\lambda = 0.1583$ the CV-MSE is 6.16, about a tenth of one SE above the minimum.

```text
==============================================================================
 Demo MS-01 | ridge_cv_svd | High-Dimensional (n=300, p=500, 10-fold)
==============================================================================

Data: n=300, p=500, folds=10, active coefs=10

  lambda.min : 0.2967  (CV-MSE=6.1093 +/- 0.4267)
  lambda.1se : 1.1170  (CV-MSE=6.5036 +/- 0.5849)

  CV-MSE curve (every 10 steps):
    lambda        CV-MSE      SEM         marker
    0.0195        6.4811      0.3994
    0.0392        6.3957      0.3961
    0.0788        6.2777      0.3938
    0.1583        6.1582      0.4004
    0.3181        6.1105      0.4314
    0.6392        6.2323      0.5015
    1.2843        6.5970      0.6082
    2.5805        7.1906      0.7277
    5.1847        7.8855      0.8326
    10.4173       8.5126      0.9091
    0.2967        6.1093      0.4267        <-- min
    1.1170        6.5036      0.5849        <-- 1se

  Ordering check (lambda.min <= lambda.1se): PASS
  Grid-edge check (both picks inside the grid): PASS
```

### Scenario 3: Very high-dimensional ($p \gg n$)

With $p = 1{,}000$ and $n = 300$, the demo uses 5 folds, so each training fold keeps 240 rows. Toward
$\lambda \to 0$ the fit approaches the minimum-norm interpolator of the training fold, and the CV-MSE of about 10.6 at
the lower grid end is barely above the minimum of 10.48. The curve stays within one SE of the minimum up to
$\lambda \approx 8.7$, so $\lambda_{1\mathrm{se}}$ is about 11 times $\lambda_{\min}$. The reference value 0.42 falls
inside the same flat valley. On this draw the minimum stays close to the null-model error of about 11, so ridge
recovers little of the signal from 240 rows, while the C++ draw reaches 8.92.

```text
==============================================================================
 Demo MS-01 | ridge_cv_svd | Very High-Dimensional (n=300, p=1000, 5-fold)
==============================================================================

Data: n=300, p=1000, folds=5, active coefs=10
  Note: 5 folds used to ensure each fold's training set is large enough relative to n.

  lambda.min : 0.7564  (CV-MSE=10.4831 +/- 0.8127)
  lambda.1se : 8.6971  (CV-MSE=11.2559 +/- 0.8617)

  CV-MSE curve (every 10 steps):
    lambda        CV-MSE      SEM         marker
    0.0266        10.5980     0.8149
    0.0534        10.5873     0.8146
    0.1072        10.5681     0.8140
    0.2154        10.5375     0.8131
    0.4329        10.5002     0.8123
    0.8697        10.4842     0.8132
    1.7474        10.5521     0.8190
    3.5110        10.7690     0.8334
    7.0545        11.1315     0.8549
    14.1741       11.5452     0.8768
    0.7564        10.4831     0.8127        <-- min
    8.6971        11.2559     0.8617        <-- 1se

  Ordering check (lambda.min <= lambda.1se): PASS
  Grid-edge check (both picks inside the grid): PASS
```

### Scenario 4: Memory-mapped design matrix

The design matrix is written with `numpy_to_memmap()` to a memory-mapped file of
$5{,}000 \cdot 500 \cdot 8 = 20{,}000{,}000$ bytes (19.07 MB in the console's units of $2^{20}$ bytes). The file comes
from `tempfile.mkstemp()` in the system temporary directory, and the demo prints its path as `<tmp>/` plus the file
name. After the writer is released, the file is reopened as a `MemoryMappedMatrix`, and its `to_numpy()` view, a
zero-copy Fortran-ordered `float64` array, goes to `RidgeCV.fit()` without a copy at the interface. The C++ demo
reopens the file read-only as a `ConstMapType`. The Python zero-copy view requires `AccessMode.ReadWrite`, so the
demo reopens the file in that mode and only reads from it. The CV routine still gathers every training and test fold
into random-access memory (RAM), so the demo is not out-of-core.

With $n = 10p$, the reference value is $\lambda^\ast \approx 500/(4{,}500 \cdot 10) = 0.011$, while the default grid
would end at $\lambda_{\max}/1{,}000 \approx 0.075$ (here $\lambda_{\max} \approx 75$). The curve rises steadily above
$\lambda \approx 0.01$, so on that grid $\lambda_{\min}$ would land on the first grid point, which the grid-edge check
would flag. This scenario therefore widens the range to $\lambda_{\max}/100{,}000$. The minimum now sits inside the
grid at $\lambda_{\min} = 0.0097$. Below it the curve flattens at about 1.12, because the fit approaches ordinary
least squares (OLS), which is already close to the noise floor $\sigma^2 = 1$ at this sample size.

```text
==============================================================================
 Demo MS-01 | ridge_cv_svd | Memory-Mapped Dataset (n=5000, p=500, 10-fold)
==============================================================================

Writing X to mmap file: <tmp>/trex_mlm_demo_ms01_svd_mmap_83vn1k38.bin
  File size: 19.07 MB
  Active coefs: 10

Re-opening mmap and fitting ridge CV (10-fold)...

  lambda.min : 0.0097  (CV-MSE=1.1205 +/- 0.0247)
  lambda.1se : 0.0496  (CV-MSE=1.1397 +/- 0.0248)

  CV-MSE curve (every 10 steps):
    lambda        CV-MSE      SEM         marker
    0.0008        1.1217      0.0249
    0.0024        1.1213      0.0249
    0.0077        1.1206      0.0248
    0.0247        1.1233      0.0246
    0.0790        1.1748      0.0254
    0.2527        1.5809      0.0336
    0.8083        3.2521      0.0708
    2.5861        6.3521      0.1384
    8.2736        8.9944      0.1938
    26.4694       10.2885     0.2202
    0.0097        1.1205      0.0247        <-- min
    0.0496        1.1397      0.0248        <-- 1se

  Ordering check (lambda.min <= lambda.1se): PASS
  Grid-edge check (both picks inside the grid): PASS

Cleaned up mmap file: <tmp>/trex_mlm_demo_ms01_svd_mmap_83vn1k38.bin
```

---

## Results at a glance

| Scenario | Size (n x p) | Folds | $\lambda^\ast$ | $\lambda_{\min}$ | $\lambda_{1\mathrm{se}}$ | CV-MSE at $\lambda_{\min}$ |
| --- | --- | --- | --- | --- | --- | --- |
| 1 Low-dimensional | 1,000 x 500 | 10 | 0.056 | 0.0603 | 0.1130 | 2.16 |
| 2 High-dimensional | 300 x 500 | 10 | 0.19 | 0.2967 | 1.1170 | 6.11 |
| 3 Very high-dimensional | 300 x 1,000 | 5 | 0.42 | 0.7564 | 8.6971 | 10.48 |
| 4 Memory-mapped | 5,000 x 500 | 10 | 0.011 | 0.0097 | 0.0496 | 1.12 |

---

## Technical notes

- `RidgeCV` runs the compiled `ridge_cv_svd`, which uses `Eigen::JacobiSVD` rather than `BDCSVD` for numerical safety
  under `-O3 -march=native`.
- The demo calls `RidgeCV().fit(X, y, n_folds=..., n_lambda=100, lambda_ratio=..., seed=...)`. The binding accepts
  only `float64` arrays in Fortran (column-major) order, which `np.asfortranarray()` provides.
- The grid settings are the `fit()` arguments `n_lambda` (100, the default of the package, the library and the R
  package) and `lambda_ratio` (1,000, Scenario 4: 100,000).
- The fold permutation takes the data seed of each scenario (`seed=111`, `222`, `333` and `444`), while the C++ demo
  keeps the default fold seed 0. So the Python run differs from the C++ one in the folds as well as in the data.
- For $p \ge n_{\mathrm{train}}$, `interpolates()` reports that the low end of the grid is an interpolation regime,
  and together with `selection_at_grid_edge()` it tells whether $\lambda_{\min}$ is a real minimum. The demo prints
  only the grid-edge check.
- The backing file gets the prefix `trex_mlm_demo_ms01_svd_mmap_` and a random suffix from `tempfile.mkstemp()`. It is
  removed in a `finally` block after the view and the map are deleted, while the C++ demo writes to the path in
  `TREX_DEMO_MMAP_PATH` or to a file in the working directory.
- `RidgeCV` also offers a kernel engine (`set_low_memory(True)`) that reads the folds from $\boldsymbol{X}$ in place.
  The demo keeps the default engine, which gathers them.
