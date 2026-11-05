# Demo: Ridge Regression via `RidgeSolver`

## Purpose

This demo shows `RidgeSolver`, the ridge regression solver for a single penalty, in three settings:

1. the **primal path** for $n \ge p$,
2. the **dual path** for $n < p$,
3. a **lambda sweep** showing how the solution changes as the penalty grows, with a monotone-shrinkage check that the
   Python version adds.

It introduces the solver, while the [model-selection demos](../../model_selection/README.md) choose $\lambda$ by
cross-validation. The same workflow runs in [C++](../../../../cpp/ml_methods/ridge_regression/demo_mlm_ridge_01/)
and [R](../../../../R/ml_methods/ridge_regression/demo_mlm_ridge_01/). The Python version replaces the Eigen
matrices with Fortran-ordered `float64` NumPy arrays.

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

From the repository root, with TRexSelector 1.0.0 installed in the repository virtual environment `.venv`:

```bash
.venv/bin/python Python/ml_methods/ridge_regression/demo_mlm_ridge_01/demo_mlm_ridge_01.py
```

The demo prints to the console only and finishes in well under a second. The output below is a run against library
b95b984. The data are drawn with NumPy's `default_rng`, a permuted congruential generator (PCG64), seeded with the
values of the C++ demo (42, 7 and 99). The C++ demo draws with `std::mt19937`, so the two languages print different
numbers that agree qualitatively, not bitwise.

---

## Console walk-through

### Subproblem 1: Primal path ($n \ge p$)

With $n = 500$, $p = 50$, all 50 true coefficients equal to 1 and noise $\sigma = 0.1$, the problem is well
determined. The coefficient error of 0.038 lies close to the level of ordinary least squares (OLS),
$\sigma\sqrt{p/(n-p-1)} \approx 0.033$, because the penalty $\lambda = 1$ is small next to
$\boldsymbol{X}^\top \boldsymbol{X} \approx 500\,\boldsymbol{I}$. It still adds a bias of about
$\lambda \lVert \boldsymbol\beta_{\mathrm{true}} \rVert_2 / 500 \approx 0.014$, which combines with the OLS error in
quadrature to about 0.036. The residual norm of 2.14 is close to the noise level $\sigma\sqrt{n-p} \approx 2.12$.

```text
==============================================================================
 RidgeSolver - Primal Path (n >= p)
==============================================================================

Matrix size : 500 x 50
Lambda      : 1

||beta_hat - beta_true||_2 : 3.8124e-02
Residual ||y - X*beta||_2  : 2.1398e+00
```

### Subproblem 2: Dual path ($n < p$)

With $n = 80$, $p = 300$ and only the first ten coefficients equal to 1, the dual solution
$\widehat{\boldsymbol\beta} = \boldsymbol X^\top \boldsymbol\alpha$ lies in the 80-dimensional row space of
$\boldsymbol{X}$. The part of $\boldsymbol\beta_{\mathrm{true}}$ outside that space is out of reach for any $\lambda$,
and its norm is $\sqrt{1 - n/p}\,\lVert \boldsymbol\beta_{\mathrm{true}} \rVert_2 \approx 2.71$ on average. The
coefficient error of 2.59 therefore stays close to $\lVert \boldsymbol\beta_{\mathrm{true}} \rVert_2 = 3.16$. It sits
slightly below the average of 2.71 because the row space of this draw holds a little more of
$\boldsymbol\beta_{\mathrm{true}}$ than a typical one, while the C++ draw gives 2.82. The residual norm of 1.18, in
contrast, is of the order of the noise level $\sigma\sqrt{n} \approx 0.89$. It equals the norm of
$\lambda(\boldsymbol X\boldsymbol X^\top + \lambda\boldsymbol I)^{-1}\boldsymbol y$ and would drop to zero as
$\lambda \to 0$. With 300 coefficients for 80 rows, ridge reproduces $\boldsymbol{y}$ without finding the ten active
coefficients, which calls for a sparse method such as the lasso.

```text
==============================================================================
 RidgeSolver - Dual Path (n < p)
==============================================================================

Matrix size : 80 x 300
Lambda      : 10

||beta_hat - beta_true||_2 : 2.5904e+00
Residual ||y - X*beta||_2  : 1.1827e+00
```

### Subproblem 3: Lambda sweep

With $n = 200$, $p = 40$, all 40 coefficients equal to 1 and $\sigma = 0.5$, the solver runs over
$\lambda \in \lbrace 10^{-4}, 10^{-2}, 0.1, 1, 10, 100, 1{,}000 \rbrace$. The coefficient error starts at its near-OLS
value of 0.2578 for $\lambda = 10^{-4}$, close to $\sigma\sqrt{p/(n-p-1)} \approx 0.25$. It reaches its grid minimum
of 0.2575 at $\lambda = 0.1$ and then grows, slowly at $\lambda = 1$ and quickly from $\lambda = 10$ on. At
$\lambda = 1{,}000$ it approaches $\lVert \boldsymbol\beta_{\mathrm{true}} \rVert_2 = \sqrt{40} \approx 6.32$ as the
estimate shrinks toward zero. The reference value $\lambda^\ast \approx 40 \cdot 0.25/40 = 0.25$ lies between the grid
points 0.1 and 1, and the C++ draw has its minimum at $\lambda = 1$, on the other side of $\lambda^\ast$. The gain
over OLS is tiny because $\lambda^\ast$ is small next to $\boldsymbol X^\top \boldsymbol X \approx 200\,\boldsymbol I$.
The residual norm, in contrast, rises with $\lambda$ over the whole grid, although the first two values agree in the
printed digits. Shrinkage gives up fit to the observed $\boldsymbol{y}$ in exchange for a smaller coefficient error,
and past the optimum both get worse.

The Python addition then asserts that $\lVert \widehat{\boldsymbol\beta} \rVert_2$ strictly decreases and the residual
strictly increases along the grid. Both hold for ridge regression in general, so the check guards the solver rather
than the data, and the run passes it.

```text
==============================================================================
 RidgeSolver - Lambda Sweep
==============================================================================

Matrix size : 200 x 40

      lambda ||beta_hat - true||_2        ||residual||_2
--------------------------------------------------------
     1.0e-04            2.5779e-01            6.1168e+00
     1.0e-02            2.5776e-01            6.1168e+00
     1.0e-01            2.5747e-01            6.1171e+00
     1.0e+00            2.5828e-01            6.1381e+00
     1.0e+01            4.7746e-01            7.7584e+00
     1.0e+02            2.4751e+00            3.0589e+01
     1.0e+03            5.3506e+00            7.2096e+01

Shrinkage check passed: ||beta_hat||_2 decreases and the residual
increases monotonically in lambda.
```

---

## Technical notes

- `RidgeSolver.solve(X, y, lambda_val)` is a static method with read-only `Eigen::Ref` arguments. Fortran-ordered
  `float64` arrays bind without a copy, while a C-ordered array is copied once. The result is a NumPy vector.
- The solver fits no intercept and does not standardize. Center and scale the data first if the penalty should
  treat all columns alike.
- For a penalty grid with cross-validation, `model_selection/` offers `RidgeCV`, which reuses one singular value
  decomposition (SVD) per fold for the whole grid, and `ElasticNetCV`, which runs warm-started coordinate-descent
  paths.
- The shrinkage check uses `assert`, so a violation would stop the script with an `AssertionError`.
