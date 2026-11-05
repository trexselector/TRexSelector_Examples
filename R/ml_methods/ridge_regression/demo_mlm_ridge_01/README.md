# Demo: Ridge Regression via `MLRidge`

## Purpose

This demo shows `MLRidge`, the R6 class around the C++ `RidgeSolver`, which solves ridge regression for a single
penalty, in three settings:

1. the **primal path** for $n \ge p$, with a check of the coefficient error that the R version adds,
2. the **dual path** for $n < p$,
3. a **lambda sweep** showing how the solution changes as the penalty grows, with a monotone-shrinkage check that the
   R and Python versions add.

It introduces the solver, while the [model-selection demos](../../model_selection/README.md) choose $\lambda$ by
cross-validation. The same workflow runs in [C++](../../../../cpp/ml_methods/ridge_regression/demo_mlm_ridge_01/)
and [Python](../../../../Python/ml_methods/ridge_regression/demo_mlm_ridge_01/). The R version replaces the Eigen
matrices with ordinary double-storage R matrices.

---

## Statistical model and notation

Let

- $\boldsymbol{X} \in \mathbb{R}^{n \times p}$ denote the design matrix with independent standard normal entries,
- $\boldsymbol{y} \in \mathbb{R}^n$ the response vector,
- $\boldsymbol{\beta} \in \mathbb{R}^p$ the regression coefficients,
- $\lambda \ge 0$ the ridge penalty.

The synthetic data follow

```math
\boldsymbol{y} = \boldsymbol{X}\boldsymbol{\beta}_{\mathrm{true}} + \boldsymbol{\varepsilon},
\qquad
\boldsymbol{\varepsilon} \sim \mathcal{N}(\boldsymbol{0}, \sigma^2 \boldsymbol{I}_n).
```

---

## Ridge objective

For a fixed penalty $\lambda$, ridge regression solves

```math
\widehat{\boldsymbol{\beta}}(\lambda) =
\arg\min_{\boldsymbol{\beta} \in \mathbb{R}^p}
\lVert \boldsymbol{y} - \boldsymbol{X}\boldsymbol{\beta} \rVert_2^2
+
\lambda \lVert \boldsymbol{\beta} \rVert_2^2 .
```

`RidgeSolver` works on $\boldsymbol{X}$ as given, without centering or scaling, and solves the smaller of two
equivalent linear systems:

```math
\widehat{\boldsymbol{\beta}}
= (\boldsymbol{X}^\top \boldsymbol{X} + \lambda \boldsymbol{I}_p)^{-1}\boldsymbol{X}^\top \boldsymbol{y}
\quad (n \ge p,\ \text{primal}),
\qquad
\widehat{\boldsymbol{\beta}}
= \boldsymbol{X}^\top (\boldsymbol{X}\boldsymbol{X}^\top + \lambda \boldsymbol{I}_n)^{-1} \boldsymbol{y}
\quad (n < p,\ \text{dual}).
```

Both systems are solved by a Cholesky factorization, with a pivoted $\boldsymbol{L}\boldsymbol{D}\boldsymbol{L}^\top$
factorization as the fallback. On this scale, the penalty that minimizes the coefficient error for an isotropic
Gaussian design is about $\lambda^\ast \approx p\,\sigma^2 / \lVert \boldsymbol\beta_{\mathrm{true}} \rVert_2^2$.

---

## Running the demo

From the repository root, with the TRexSelector R package 2.0.0 installed:

```bash
Rscript R/ml_methods/ridge_regression/demo_mlm_ridge_01/demo_mlm_ridge_01.R
```

The demo prints to the console only and finishes in about a second. The output below is a run against library
b95b984. The data come from R's default generator, a Mersenne-Twister with normals by inversion, seeded through
`set.seed()` with the values of the C++ demo (42, 7 and 99). The C++ demo uses `std::mt19937` with
`std::normal_distribution`, and `set.seed()` scrambles its seed before use. The two languages therefore print
different numbers that agree qualitatively, not bitwise.

---

## Console walk-through

### Subproblem 1: Primal path ($n \ge p$)

With $n = 500$, $p = 50$, all 50 true coefficients equal to 1 and noise $\sigma = 0.1$, the problem is well
determined. Ordinary least squares (OLS) reaches a coefficient error of about $\sigma\sqrt{p/(n-p-1)} \approx 0.033$,
and the penalty $\lambda = 1$ is small next to $\boldsymbol{X}^\top \boldsymbol{X} \approx 500\,\boldsymbol{I}$. It
still adds a bias of about $\lambda \lVert \boldsymbol\beta_{\mathrm{true}} \rVert_2 / 500 \approx 0.014$, which
combines with the OLS error in quadrature to about 0.036. The coefficient error of 0.045 lies somewhat above this
average-case value, which is a matter of the particular draw, while the C++ and Python runs print 0.035 and 0.038.
The residual norm of 2.13 is close to the noise level $\sigma\sqrt{n-p} \approx 2.12$. The `stopifnot()` check that
the R version adds requires an error below 0.5, and the ✓ line confirms it.

```text
==============================================================================
 RidgeSolver - Primal Path (n >= p)
==============================================================================

Matrix size : 500 x 50
Lambda      : 1

||beta_hat - beta_true||_2 : 4.4950e-02
Residual ||y - X*beta||_2  : 2.1293e+00
✓ Ridge estimate close to the true coefficients
```

### Subproblem 2: Dual path ($n < p$)

With $n = 80$, $p = 300$ and only the first ten coefficients equal to 1, the dual solution
$\widehat{\boldsymbol\beta} = \boldsymbol X^\top \boldsymbol\alpha$ lies in the 80-dimensional row space of
$\boldsymbol{X}$. The part of $\boldsymbol\beta_{\mathrm{true}}$ outside that space is out of reach for any $\lambda$,
and its norm is $\sqrt{1 - n/p}\,\lVert \boldsymbol\beta_{\mathrm{true}} \rVert_2 \approx 2.71$ on average. The
coefficient error of 2.74 therefore stays close to $\lVert \boldsymbol\beta_{\mathrm{true}} \rVert_2 = 3.16$. It sits
slightly above the average of 2.71 because the row space of this draw holds a little less of
$\boldsymbol\beta_{\mathrm{true}}$ than a typical one, while the Python and C++ draws give 2.59 and 2.82. The residual
norm of 0.96, in contrast, is of the order of the noise level $\sigma\sqrt{n} \approx 0.89$. It equals the norm of
$\lambda(\boldsymbol X\boldsymbol X^\top + \lambda\boldsymbol I)^{-1}\boldsymbol y$ and would drop to zero as
$\lambda \to 0$. With 300 coefficients for 80 rows, ridge reproduces $\boldsymbol{y}$ without finding the ten active
coefficients, which calls for a sparse method such as the lasso.

```text
==============================================================================
 RidgeSolver - Dual Path (n < p)
==============================================================================

Matrix size : 80 x 300
Lambda      : 10

||beta_hat - beta_true||_2 : 2.7416e+00
Residual ||y - X*beta||_2  : 9.6179e-01
```

### Subproblem 3: Lambda sweep

With $n = 200$, $p = 40$, all 40 coefficients equal to 1 and $\sigma = 0.5$, the solver runs over
$\lambda \in \lbrace 10^{-4}, 10^{-2}, 0.1, 1, 10, 100, 1{,}000 \rbrace$. The coefficient error starts at its near-OLS
value of 0.2741 for $\lambda = 10^{-4}$, close to $\sigma\sqrt{p/(n-p-1)} \approx 0.25$. It falls to its grid minimum
of 0.2670 at $\lambda = 1$ and then grows quickly from $\lambda = 10$ on. At $\lambda = 1{,}000$ it reaches 5.25 and
approaches $\lVert \boldsymbol\beta_{\mathrm{true}} \rVert_2 = \sqrt{40} \approx 6.32$ as the estimate shrinks toward
zero. The reference value $\lambda^\ast \approx 40 \cdot 0.25/40 = 0.25$ lies between the grid points 0.1 and 1. The
R draw has its minimum at $\lambda = 1$ like the C++ draw, while the Python draw has it at $\lambda = 0.1$, on the
other side of $\lambda^\ast$. The gain over OLS stays small because $\lambda^\ast$ is small next to
$\boldsymbol X^\top \boldsymbol X \approx 200\,\boldsymbol I$.

The column $\lVert \widehat{\boldsymbol\beta} \rVert_2$, which only the R version prints, falls from 6.38 at
$\lambda = 10^{-4}$ to 1.15 at $\lambda = 1{,}000$. The residual norm, in contrast, rises with $\lambda$ over the whole
grid, although the first two values agree in the printed digits. Shrinkage gives up fit to the observed $\boldsymbol{y}$
in exchange for a smaller coefficient error, and past the optimum both get worse.

The R addition then checks with two `stopifnot()` calls that $\lVert \widehat{\boldsymbol\beta} \rVert_2$ never
increases and the residual never decreases along the grid, up to a tolerance of $10^{-10}$. Both hold for ridge
regression in general, so the check guards the solver rather than the data, and the ✓ line shows that the run passes.

```text
==============================================================================
 RidgeSolver - Lambda Sweep
==============================================================================

Matrix size : 200 x 40

lambda      ||beta_hat - true||_2 ||beta_hat||_2    ||residual||_2
----------------------------------------------------------------------
1.0e-04     2.7412e-01            6.3811e+00        5.8348e+00
1.0e-02     2.7402e-01            6.3807e+00        5.8348e+00
1.0e-01     2.7316e-01            6.3774e+00        5.8350e+00
1.0e+00     2.6702e-01            6.3450e+00        5.8544e+00
1.0e+01     3.9990e-01            6.0418e+00        7.3782e+00
1.0e+02     2.2501e+00            4.2029e+00        3.0353e+01
1.0e+03     5.2531e+00            1.1513e+00        7.5016e+01

✓ Coefficients shrink and the residual grows with lambda
```

---

## Technical notes

- `MLRidge$new()$solve(X, y, lambda)` wraps the static C++ `RidgeSolver::solve()` and reads `X` through a read-only
  map. A double-storage R matrix binds without a copy, and an `mmap_matrix` handle binds zero-copy in `"readonly"`
  mode. The result is a numeric vector of length $p$.
- `$solve()` stops with an error unless $\lambda \ge 0$ and `nrow(X)` equals `length(y)`.
- The solver fits no intercept and does not standardize. Center and scale the data first if the penalty should
  treat all columns alike.
- For a penalty grid with cross-validation, `model_selection/` offers `MLRidgeCV`, which reuses one singular value
  decomposition (SVD) per fold for the whole grid, and `MLElasticNetCV`, which runs warm-started coordinate-descent
  paths.
- The sweep table adds a $\lVert \widehat{\boldsymbol\beta} \rVert_2$ column that the C++ and Python demos do not
  print, because the monotonicity check is stated in terms of that norm.
- The checks use `stopifnot()`, so a violation would stop the script with an error.
