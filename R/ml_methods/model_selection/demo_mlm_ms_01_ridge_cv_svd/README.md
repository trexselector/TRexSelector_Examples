# Demo: Ridge Regression K-Fold CV via JacobiSVD

## Purpose

This demo shows `MLRidgeCV` from the TRexSelector R package, the R6 binding of the library's `ridge_cv_svd`. It is
the K-fold cross-validation (CV) routine for ridge regression that solves every fold with a singular value
decomposition (SVD), and it selects the penalty $\lambda$ from the cross-validated prediction error in four synthetic
Gaussian settings:

1. **Low-dimensional**: $n = 1{,}000$, $p = 500$, 10-fold CV.
2. **High-dimensional**: $n = 300$, $p = 500$, 10-fold CV.
3. **Very high-dimensional**: $n = 300$, $p = 1{,}000$, 5-fold CV.
4. **Memory-mapped dataset**: $n = 5{,}000$, $p = 500$, 10-fold CV, the design matrix stored on disk.

The same workflow is implemented in
[C++](../../../../cpp/ml_methods/model_selection/demo_mlm_ms_01_ridge_cv_svd/) and
[Python](../../../../Python/ml_methods/model_selection/demo_mlm_ms_01_ridge_cv_svd/).

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

The entries of $\boldsymbol{X}$ are independent standard normal and are drawn with `rnorm()` into an ordinary numeric
R matrix. It is column-major like the library's Eigen matrices, so `$fit()` binds it without a copy. Ten entries of
$\boldsymbol\beta_{\mathrm{true}}$ are set to 1 at evenly spaced coordinates, the rest are zero, so
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

`MLRidgeCV` centers every training fold and scales its columns to unit $\ell_2$-norm before the solve, so $\lambda$
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
widens the range to $\lambda_{\max}/100{,}000$. Two checks close every scenario. The first is the ordering
$\lambda_{\min} \le \lambda_{1\mathrm{se}}$, and the second is the grid-edge check, which fails when a pick lands on
the first or last grid point. Such a pick is the end of the grid, not a minimum. The R binding does not expose the
library's `selection_at_grid_edge()`, so the demo applies the same rule itself and tests the 1-based indices
`$index_min()` and `$index_1se()` against 1 and the grid length.

A reference value for $\lambda_{\min}$ comes from ridge theory. For an isotropic Gaussian design, the penalty that
minimizes the prediction error is about

```math
\lambda^\ast \approx \frac{p\,\sigma^2}{n_{\mathrm{train}}\,\lVert \boldsymbol{\beta}_{\mathrm{true}} \rVert_2^2}
```

on the unit-norm scale, with $n_{\mathrm{train}}$ the number of training rows per fold. The walk-through compares the
CV picks with it.

---

## Running the demo

From the repository root, with the TRexSelector R package installed as described in
[`LIBRARY_TARGET.txt`](../../../../LIBRARY_TARGET.txt):

```bash
Rscript R/ml_methods/model_selection/demo_mlm_ms_01_ridge_cv_svd/demo_mlm_ms_01_ridge_cv_svd.R
```

The demo prints to the console only and takes about 40 seconds. The output below is a run against library b95b984,
with the package TRexSelector 2.0.0 built from that commit. The data come from R's Mersenne-Twister generator via
`set.seed()` and `rnorm()`, which draws normals by inversion, with the same data seeds as the C++ demo. The C++ demo
draws from `std::mt19937` and the Python demo from NumPy's `default_rng`, so the three languages see different data,
and their numbers agree qualitatively, not bitwise.

---

## Console walk-through

The run opens with a banner for the suite and closes with `All ridge_cv_svd demos complete.`, and every scenario
prints its own block. The CV-MSE curve lists every tenth grid point in increasing $\lambda$, as its header
`lambda ascending` states, followed by the rows of the two picks.

### Scenario 1: Low-dimensional ($n > p$)

With $n = 1{,}000 > p = 500$, the problem is overdetermined and the penalty only adds shrinkage. Each fold trains on
900 rows, so the reference value is $\lambda^\ast \approx 500/(900 \cdot 10) = 0.056$, and CV picks
$\lambda_{\min} = 0.0526$ (C++ picks 0.0551 and Python 0.0603 on their own draws). The CV-MSE of 2.16 is more than
twice the noise variance, the price of estimating 500 coefficients from 900 rows. $\lambda_{1\mathrm{se}} = 0.1133$
about doubles the penalty for an error of 2.27, still within one SE of the minimum.

```text
==============================================================================
 Demo MS-01 | ridge_cv_svd | Low-Dimensional (n=1000, p=500, 10-fold)
==============================================================================

Data: n=1000, p=500, folds=10, active coefs=10

  lambda.min : 0.0526  (CV-MSE=2.1579 +/- 0.1216)
  lambda.1se : 0.1133  (CV-MSE=2.2663 +/- 0.1218)

  CV-MSE curve (every 10 steps, lambda ascending):
    lambda        CV-MSE      SEM         marker
    0.0398        2.1680      0.1234
    0.0799        2.1844      0.1202
    0.1606        2.4251      0.1283
    0.3227        3.0468      0.1649
    0.6483        4.1393      0.2383
    1.3026        5.6593      0.3383
    2.6173        7.3877      0.4421
    5.2587        8.9871      0.5279
    10.5659       10.2011     0.5874
    21.2294       10.9845     0.6235
    0.0526        2.1579      0.1216        <-- min
    0.1133        2.2663      0.1218        <-- 1se

  Ordering check (lambda.min <= lambda.1se): PASS
  Grid-edge check (both picks inside the grid): PASS
```

### Scenario 2: High-dimensional ($p > n$)

With $p = 500 > n = 300$, the unpenalized least-squares problem has no unique solution, and the penalty makes it well
posed. Ridge spreads the fit over all 500 coefficients and cannot single out the ten active ones from 270 training
rows. The minimum CV-MSE of 5.95 removes about half of the signal variance. The null model errs by about 11, and the
noise floor is 1. In this run $\lambda_{\min} = 0.1166$ lies below the reference value 0.19, while the C++ and Python
picks (0.3082 and 0.2967) lie above it. The curve is flat in between, and at $\lambda = 0.1773$ the CV-MSE is 5.96,
less than a twentieth of one SE above the minimum.

```text
==============================================================================
 Demo MS-01 | ridge_cv_svd | High-Dimensional (n=300, p=500, 10-fold)
==============================================================================

Data: n=300, p=500, folds=10, active coefs=10

  lambda.min : 0.1166  (CV-MSE=5.9461 +/- 0.3976)
  lambda.1se : 0.5414  (CV-MSE=6.3366 +/- 0.4752)

  CV-MSE curve (every 10 steps, lambda ascending):
    lambda        CV-MSE      SEM         marker
    0.0219        6.0455      0.3842
    0.0439        5.9986      0.3869
    0.0882        5.9533      0.3932
    0.1773        5.9635      0.4077
    0.3562        6.1242      0.4401
    0.7157        6.5371      0.5081
    1.4380        7.2507      0.6282
    2.8892        8.1990      0.7864
    5.8050        9.1971      0.9408
    11.6637       10.0433     1.0587
    0.1166        5.9461      0.3976        <-- min
    0.5414        6.3366      0.4752        <-- 1se

  Ordering check (lambda.min <= lambda.1se): PASS
  Grid-edge check (both picks inside the grid): PASS
```

### Scenario 3: Very high-dimensional ($p \gg n$)

With $p = 1{,}000$ and $n = 300$, the demo uses 5 folds, so each training fold keeps 240 rows. Toward
$\lambda \to 0$ the fit approaches the minimum-norm interpolator of the training fold, and the CV-MSE of about 8.85 at
the lower grid end is barely above the minimum of 8.70. The curve stays within one SE of the minimum up to
$\lambda \approx 10.3$, so $\lambda_{1\mathrm{se}}$ is about 12 times $\lambda_{\min}$. The reference value 0.42 falls
inside the same flat valley. On this draw the minimum lies about 2.3 below the null-model error of about 11, close to
the C++ draw (8.92), while the Python draw stays at 10.48 and recovers little of the signal.

```text
==============================================================================
 Demo MS-01 | ridge_cv_svd | Very High-Dimensional (n=300, p=1000, 5-fold)
==============================================================================

Data: n=300, p=1000, folds=5, active coefs=10
  Note: 5 folds used to ensure each fold's training set is large enough relative to n.

  lambda.min : 0.8358  (CV-MSE=8.7001 +/- 0.8961)
  lambda.1se : 10.3036  (CV-MSE=9.5858 +/- 1.0105)

  CV-MSE curve (every 10 steps, lambda ascending):
    lambda        CV-MSE      SEM         marker
    0.0207        8.8454      0.8612
    0.0416        8.8351      0.8625
    0.0836        8.8161      0.8651
    0.1679        8.7842      0.8699
    0.3374        8.7398      0.8781
    0.6779        8.7025      0.8912
    1.3621        8.7282      0.9100
    2.7367        8.8918      0.9365
    5.4987        9.2168      0.9724
    11.0482       9.6271      1.0148
    0.8358        8.7001      0.8961        <-- min
    10.3036       9.5858      1.0105        <-- 1se

  Ordering check (lambda.min <= lambda.1se): PASS
  Grid-edge check (both picks inside the grid): PASS
```

### Scenario 4: Memory-mapped design matrix

The design matrix is written with `mem_matrix_from()` to a memory-mapped file of
$5{,}000 \cdot 500 \cdot 8 = 20{,}000{,}000$ bytes (19.07 MB in the console's units of $2^{20}$ bytes). The file comes
from `tempfile()` in the temporary directory of the R session, and the demo prints its path as `<tmp>/` plus the file
name. The demo drops the writer handle with `rm()` and reopens the file read-only with
`mem_matrix(..., mode = "readonly")`. `$fit()` takes this `mmap_matrix` handle directly, and the binding hands the
library a read-only map (`ConstMapType`) of the file, which binds to its `Eigen::Ref` without a copy. So the R demo
follows the C++ demo, while the Python zero-copy view requires read-write mode. The CV routine still gathers every
training and test fold into random-access memory (RAM), so the demo is not out-of-core.

With $n = 10p$, the reference value is $\lambda^\ast \approx 500/(4{,}500 \cdot 10) = 0.011$, while the default grid
would end at $\lambda_{\max}/1{,}000 \approx 0.079$ (here $\lambda_{\max} \approx 79$). The curve rises steadily above
$\lambda \approx 0.01$, so on that grid $\lambda_{\min}$ would land on the first grid point, which the grid-edge check
would flag. This scenario therefore widens the range to $\lambda_{\max}/100{,}000$. The minimum now sits inside the
grid at $\lambda_{\min} = 0.0114$. Below it the curve flattens at about 1.16, because the fit approaches ordinary
least squares (OLS), which is already close to the noise floor $\sigma^2 = 1$ at this sample size.

```text
==============================================================================
 Demo MS-01 | ridge_cv_svd | Memory-Mapped Dataset (n=5000, p=500, 10-fold)
==============================================================================

Writing X to mmap file: <tmp>/trex_mlm_demo_ms01_svd_mmap_c91366a5db7.bin
  File size: 19.07 MB
  Active coefs: 10

Re-opening mmap read-only and fitting RidgeCV (10-fold)...

  lambda.min : 0.0114  (CV-MSE=1.1589 +/- 0.0182)
  lambda.1se : 0.0461  (CV-MSE=1.1741 +/- 0.0178)

  CV-MSE curve (every 10 steps, lambda ascending):
    lambda        CV-MSE      SEM         marker
    0.0008        1.1603      0.0186
    0.0025        1.1598      0.0185
    0.0081        1.1590      0.0183
    0.0258        1.1618      0.0179
    0.0825        1.2161      0.0185
    0.2641        1.6418      0.0307
    0.8449        3.3749      0.0762
    2.7029        6.5592      0.1523
    8.6474        9.2521      0.2139
    27.6654       10.5632     0.2432
    0.0114        1.1589      0.0182        <-- min
    0.0461        1.1741      0.0178        <-- 1se

  Ordering check (lambda.min <= lambda.1se): PASS
  Grid-edge check (both picks inside the grid): PASS

Cleaned up mmap file: <tmp>/trex_mlm_demo_ms01_svd_mmap_c91366a5db7.bin
```

---

## Results at a glance

| Scenario | Size (n x p) | Folds | $\lambda^\ast$ | $\lambda_{\min}$ | $\lambda_{1\mathrm{se}}$ | CV-MSE at $\lambda_{\min}$ |
| --- | --- | --- | --- | --- | --- | --- |
| 1 Low-dimensional | 1,000 x 500 | 10 | 0.056 | 0.0526 | 0.1133 | 2.16 |
| 2 High-dimensional | 300 x 500 | 10 | 0.19 | 0.1166 | 0.5414 | 5.95 |
| 3 Very high-dimensional | 300 x 1,000 | 5 | 0.42 | 0.8358 | 10.3036 | 8.70 |
| 4 Memory-mapped | 5,000 x 500 | 10 | 0.011 | 0.0114 | 0.0461 | 1.16 |

---

## Technical notes

- `MLRidgeCV` runs the compiled `ridge_cv_svd`, which uses `Eigen::JacobiSVD` rather than `BDCSVD`, because the
  latter can silently return wrong results under aggressive optimization flags.
- The demo calls `MLRidgeCV$new()` and `$fit(X, y, n_folds = ..., n_lambda = 100L, lambda_ratio = ..., seed = ...)`
  and reads the results with `$cv_min()`, `$cv_1se()`, `$index_min()`, `$index_1se()`, `$get_lambdas()`,
  `$get_cv_errors()` and `$get_cv_std()`. The fold count is `n_folds` on every surface, as in `MLElasticNetCV` and
  the Python `RidgeCV`.
- The grid settings are the `$fit()` arguments `n_lambda` (100, the default of the library, the R package and the
  Python package) and `lambda_ratio` (1,000, Scenario 4: 100,000).
- `$index_min()` and `$index_1se()` return 1-based indices into the ascending grid, the R convention, because the
  binding adds 1 to the library's 0-based index. The C++ and Python accessors are 0-based.
- The fold permutation takes the data seed of each scenario (`seed = 111L`, `222L`, `333L` and `444L`), as in the
  Python demo, while the C++ demo keeps the default fold seed 0. The permutation runs in the library with
  `std::mt19937`, so it does not depend on R's generator.
- `stopifnot()` also asserts the ordering, so a violation stops the script. The grid-edge check only prints PASS or
  FAIL.
- The backing file gets the prefix `trex_mlm_demo_ms01_svd_mmap_` and a random suffix from `tempfile()`. An
  `on.exit()` handler, registered before the file is written, removes it with `unlink()` when Scenario 4 returns,
  while the C++ demo writes to the path in `TREX_DEMO_MMAP_PATH` or to a file in the working directory.
- `MLRidgeCV$new(low_memory = TRUE)` selects a kernel engine that reads the folds from $\boldsymbol{X}$ in place. The
  demo keeps the default engine, which gathers them.
