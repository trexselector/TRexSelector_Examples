# Demo: Ridge Regression K-Fold CV via JacobiSVD

## Purpose

This demo shows `ridge_cv_svd`, the K-fold cross-validation (CV) routine for ridge regression that solves every fold
with a singular value decomposition (SVD). It selects the penalty $\lambda$ from the cross-validated prediction error
in four synthetic Gaussian settings:

1. **Low-dimensional**: $n = 1{,}000$, $p = 500$, 10-fold CV.
2. **High-dimensional**: $n = 300$, $p = 500$, 10-fold CV.
3. **Very high-dimensional**: $n = 300$, $p = 1{,}000$, 5-fold CV.
4. **Memory-mapped dataset**: $n = 5{,}000$, $p = 500$, 10-fold CV, the design matrix stored on disk.

The same workflow is implemented in
[Python](../../../../Python/ml_methods/model_selection/demo_mlm_ms_01_ridge_cv_svd/) and
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

The entries of $\boldsymbol{X}$ are independent standard normal. Ten entries of $\boldsymbol\beta_{\mathrm{true}}$
are set to 1 at evenly spaced coordinates, the rest are zero, so
$\lVert \boldsymbol\beta_{\mathrm{true}} \rVert_2^2 = 10$ and the response variance is about $10 + \sigma^2 = 11$.
That is the prediction error of the null model, which predicts the mean only.

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

`ridge_cv_svd` centers every training fold and scales its columns to unit $\ell_2$-norm before the solve, so
$\lambda$ lives on that unit-norm scale. The fold solve uses `Eigen::JacobiSVD`.

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
$\lambda_{\min} \le \lambda_{1\mathrm{se}}$, and the library's grid-edge check `selection_at_grid_edge()`, which fails
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

From the repository root, after `cmake --preset release && cmake --build --preset release-build` in `cpp/`:

```bash
cpp/build/release/bin/ml_methods/model_selection/demo_mlm_ms_01_ridge_cv_svd/demo_mlm_ms_01_ridge_cv_svd
```

The demo prints to the console only. The output below is a run against library 44cf1ab on an Apple Silicon machine (the
earlier run against 8ad3376 printed the same lines), and it takes about 40 s on a single core. The data come from
`std::mt19937` and `std::normal_distribution`. The latter is implementation-defined, so another standard library
(libstdc++ on Linux, for example) may draw different data and print different numbers.

---

## Console walk-through

The CV-MSE curve lists every tenth grid point in increasing $\lambda$, followed by the rows of the two picks.

### Scenario 1: Low-dimensional ($n > p$)

With $n = 1{,}000 > p = 500$, the problem is overdetermined and the penalty only adds shrinkage. Each fold trains on
900 rows, so the reference value is $\lambda^\ast \approx 500/(900 \cdot 10) = 0.056$, and CV picks
$\lambda_{\min} = 0.0551$. The CV-MSE of 2.28 is more than twice the noise variance, the price of estimating 500
coefficients from 900 rows. $\lambda_{1\mathrm{se}}$ about doubles the penalty for an error still within one SE.

```text
============================================================
Demo MS-01 | ridge_cv_svd | Low-Dimensional (n=1000, p=500, 10-fold)
============================================================
Data: n=1000, p=500, folds=10, active coefs=10

  lambda.min : 0.0551  (CV-MSE=2.2765 +/- 0.1237)
  lambda.1se : 0.1186  (CV-MSE=2.3902 +/- 0.1339)

  CV-MSE curve (every 10 steps):
    lambda        CV-MSE      SEM         marker
    0.0338        2.2989      0.1186
    0.0679        2.2837      0.1263
    0.1364        2.4434      0.1358
    0.2740        2.9245      0.1451
    0.5506        3.8194      0.1576
    1.1063        5.1002      0.1831
    2.2228        6.5900      0.2249
    4.4662        8.0059      0.2721
    8.9736        9.1101      0.3113
    18.0301       9.8384      0.3376
    0.0551        2.2765      0.1237        <-- min
    0.1186        2.3902      0.1339        <-- 1se

  Ordering check (lambda.min <= lambda.1se): PASS
  Grid-edge check (both picks inside the grid): PASS
```

### Scenario 2: High-dimensional ($p > n$)

With $p = 500 > n = 300$, the unpenalized least-squares problem has no unique solution, and the penalty makes it well
posed. Ridge spreads the fit over all 500 coefficients and cannot single out the ten active ones from 270 training
rows. The minimum CV-MSE of 7.36 removes only about a third of the signal variance. The null model errs by about 11,
and the noise floor is 1. The reference value 0.19 lies below $\lambda_{\min} = 0.3082$, but the curve is flat there.
At $\lambda = 0.1891$ the CV-MSE is 7.42, less than a tenth of one SE above the minimum.

```text
============================================================
Demo MS-01 | ridge_cv_svd | High-Dimensional (n=300, p=500, 10-fold)
============================================================
Data: n=300, p=500, folds=10, active coefs=10

  lambda.min : 0.3082  (CV-MSE=7.3584 +/- 0.8170)
  lambda.1se : 1.4306  (CV-MSE=8.1390 +/- 0.9298)

  CV-MSE curve (every 10 steps):
    lambda        CV-MSE      SEM         marker
    0.0233        8.0213      0.8040
    0.0468        7.8521      0.8050
    0.0941        7.6267      0.8063
    0.1891        7.4184      0.8098
    0.3800        7.3702      0.8227
    0.7635        7.6210      0.8607
    1.5340        8.2117      0.9397
    3.0821        9.0524      1.0533
    6.1927        9.9572      1.1700
    12.4426       10.7347     1.2616
    0.3082        7.3584      0.8170        <-- min
    1.4306        8.1390      0.9298        <-- 1se

  Ordering check (lambda.min <= lambda.1se): PASS
  Grid-edge check (both picks inside the grid): PASS
```

### Scenario 3: Very high-dimensional ($p \gg n$)

With $p = 1{,}000$ and $n = 300$, the demo uses 5 folds, so each training fold keeps 240 rows. Toward
$\lambda \to 0$ the fit approaches the minimum-norm interpolator of the training fold, and the CV-MSE of about 9.0 at
the lower grid end is barely above the minimum of 8.92. The curve stays within one SE of the minimum up to
$\lambda \approx 7.4$, so $\lambda_{1\mathrm{se}}$ is 14 times $\lambda_{\min}$. The reference value 0.42 falls
inside the same flat valley.

```text
============================================================
Demo MS-01 | ridge_cv_svd | Very High-Dimensional (n=300, p=1000, 5-fold)
============================================================
Data: n=300, p=1000, folds=5, active coefs=10
  Note: 5 folds used to ensure each fold's training set is large enough relative to n.

  lambda.min : 0.5197  (CV-MSE=8.9209 +/- 0.9752)
  lambda.1se : 7.3659  (CV-MSE=9.8830 +/- 0.8757)

  CV-MSE curve (every 10 steps):
    lambda        CV-MSE      SEM         marker
    0.0210        8.9989      1.0137
    0.0422        8.9911      1.0117
    0.0847        8.9772      1.0077
    0.1702        8.9550      1.0002
    0.3419        8.9291      0.9869
    0.6869        8.9249      0.9657
    1.3802        9.0048      0.9372
    2.7732        9.2475      0.9067
    5.5721        9.6759      0.8825
    11.1956       10.2032     0.8684
    0.5197        8.9209      0.9752        <-- min
    7.3659        9.8830      0.8757        <-- 1se

  Ordering check (lambda.min <= lambda.1se): PASS
  Grid-edge check (both picks inside the grid): PASS
```

### Scenario 4: Memory-mapped design matrix

The design matrix is written to a memory-mapped file of $5{,}000 \cdot 500 \cdot 8 = 20{,}000{,}000$ bytes (19.07 MB in
the console's units of $2^{20}$ bytes), reopened read-only, and handed to `fit()` as a `ConstMapType`. `fit()` takes an
`Eigen::Ref`, so the mapped matrix binds without a copy at the interface. The CV routine still gathers every
training and test fold into random-access memory (RAM), so the demo is not out-of-core.

With $n = 10p$, the reference value is $\lambda^\ast \approx 500/(4{,}500 \cdot 10) = 0.011$, while the default grid
would end at $\lambda_{\max}/1{,}000 \approx 0.075$. On that grid, $\lambda_{\min}$ lands on the first grid point,
which the grid-edge check flags, so this scenario widens the range to $\lambda_{\max}/100{,}000$. The minimum now sits
inside the grid at $\lambda_{\min} = 0.0109$. Below it the curve flattens at about 1.14, because the fit approaches
ordinary least squares (OLS), which is already close to the noise floor $\sigma^2 = 1$ at this sample size.

```text
============================================================
Demo MS-01 | ridge_cv_svd | Memory-Mapped Dataset (n=5000, p=500, 10-fold)
============================================================
Writing X to mmap file: .../cpp/ml_methods/model_selection/demo_mlm_ms_01_ridge_cv_svd/trex_mlm_demo_ms01_svd_mmap.bin
  File size: 19.07 MB
  Active coefs: 10

Re-opening mmap read-only and fitting ridge_cv_svd (10-fold)...

  lambda.min : 0.0109  (CV-MSE=1.1421 +/- 0.0227)
  lambda.1se : 0.0494  (CV-MSE=1.1606 +/- 0.0218)

  CV-MSE curve (every 10 steps):
    lambda        CV-MSE      SEM         marker
    0.0008        1.1435      0.0232
    0.0024        1.1430      0.0231
    0.0077        1.1422      0.0228
    0.0246        1.1446      0.0222
    0.0787        1.1950      0.0220
    0.2519        1.5941      0.0301
    0.8058        3.2294      0.0622
    2.5778        6.2652      0.1186
    8.2472        8.8761      0.1679
    26.3851       10.1660     0.1925
    0.0109        1.1421      0.0227        <-- min
    0.0494        1.1606      0.0218        <-- 1se

  Ordering check (lambda.min <= lambda.1se): PASS
  Grid-edge check (both picks inside the grid): PASS

Cleaned up mmap file: .../cpp/ml_methods/model_selection/demo_mlm_ms_01_ridge_cv_svd/trex_mlm_demo_ms01_svd_mmap.bin
```

---

## Results at a glance

| Scenario | Size (n x p) | Folds | $\lambda^\ast$ | $\lambda_{\min}$ | $\lambda_{1\mathrm{se}}$ | CV-MSE at $\lambda_{\min}$ |
| --- | --- | --- | --- | --- | --- | --- |
| 1 Low-dimensional | 1,000 x 500 | 10 | 0.056 | 0.0551 | 0.1186 | 2.28 |
| 2 High-dimensional | 300 x 500 | 10 | 0.19 | 0.3082 | 1.4306 | 7.36 |
| 3 Very high-dimensional | 300 x 1,000 | 5 | 0.42 | 0.5197 | 7.3659 | 8.92 |
| 4 Memory-mapped | 5,000 x 500 | 10 | 0.011 | 0.0109 | 0.0494 | 1.14 |

---

## Technical notes

- `ridge_cv_svd` uses `Eigen::JacobiSVD` rather than `BDCSVD` for numerical safety under `-O3 -march=native`.
- The grid settings are the `fit()` arguments `n_lambda` (100, the default of the library, the R package and the
  Python package) and `lambda_ratio` (1,000, Scenario 4: 100,000). The fold permutation uses the default seed 0.
- For $p \ge n_{\mathrm{train}}$, `interpolates()` reports that the low end of the grid is an interpolation regime,
  and together with `selection_at_grid_edge()` it tells whether $\lambda_{\min}$ is a real minimum.
- The backing file path is injected through `TREX_DEMO_MMAP_PATH` when the build defines it, with a file name in the
  working directory as the fallback. The file is removed at the end of Scenario 4.
