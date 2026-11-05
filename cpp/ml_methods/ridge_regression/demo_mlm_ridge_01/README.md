# Demo: Ridge Regression via `RidgeSolver`

## Purpose

This demo shows `RidgeSolver`, the ridge regression solver for a single penalty, in three settings:

1. the **primal path** for $n \ge p$,
2. the **dual path** for $n < p$,
3. a **lambda sweep** showing how the solution changes as the penalty grows.

It introduces the solver, while the [model-selection demos](../../model_selection/README.md) choose $\lambda$ by
cross-validation. The same workflow runs in [Python](../../../../Python/ml_methods/ridge_regression/demo_mlm_ridge_01/)
and [R](../../../../R/ml_methods/ridge_regression/demo_mlm_ridge_01/).

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

From the repository root, after `cmake --preset release && cmake --build --preset release-build` in `cpp/`:

```bash
cpp/build/release/bin/ml_methods/ridge_regression/demo_mlm_ridge_01/demo_mlm_ridge_01
```

The demo prints to the console only and finishes in under a second. The output below comes from a run against
library b95b984 on an Apple Silicon machine. The data are drawn with `std::mt19937` and `std::normal_distribution`.
The latter is implementation-defined, so another standard library, such as libstdc++ on Linux, may draw different
data and print different numbers.

---

## Console walk-through

### Subproblem 1: Primal path ($n \ge p$)

With $n = 500$, $p = 50$, all 50 true coefficients equal to 1 and noise $\sigma = 0.1$, the problem is well
determined. The coefficient error of 0.035 matches the level of ordinary least squares (OLS),
$\sigma\sqrt{p/(n-p-1)} \approx 0.033$, because the penalty $\lambda = 1$ adds little bias next to
$\boldsymbol{X}^\top \boldsymbol{X} \approx 500\,\boldsymbol{I}$. The residual norm of 2.22 is close to the noise
level $\sigma\sqrt{n-p} \approx 2.12$.

```text
============================================================
RidgeSolver - Primal Path (n >= p)
============================================================
Matrix size : 500 x 50
Lambda      : 1

||beta_hat - beta_true||_2 : 3.4798e-02
Residual ||y - X*beta||_2  : 2.2157e+00
```

### Subproblem 2: Dual path ($n < p$)

With $n = 80$, $p = 300$ and only the first ten coefficients equal to 1, the dual solution
$\widehat{\boldsymbol\beta} = \boldsymbol X^\top \boldsymbol\alpha$ lies in the 80-dimensional row space of
$\boldsymbol{X}$. The part of $\boldsymbol\beta_{\mathrm{true}}$ outside that space, about
$\sqrt{1 - n/p}\,\lVert \boldsymbol\beta_{\mathrm{true}} \rVert_2 = 2.71$, is out of reach for any $\lambda$,
and so the coefficient error of 2.82 stays close to $\lVert \boldsymbol\beta_{\mathrm{true}} \rVert_2 = 3.16$. The
residual norm of 0.90, in contrast, is as small as the noise ($\sigma\sqrt{n} \approx 0.89$) and would drop to zero
as $\lambda \to 0$. With 300 coefficients for 80 rows, ridge reproduces $\boldsymbol{y}$ without finding the ten
active coefficients, which calls for a sparse method such as the lasso.

```text
============================================================
RidgeSolver - Dual Path (n < p)
============================================================
Matrix size : 80 x 300
Lambda      : 10

||beta_hat - beta_true||_2 : 2.8213e+00
Residual ||y - X*beta||_2  : 9.0392e-01
```

### Subproblem 3: Lambda sweep

With $n = 200$, $p = 40$, all 40 coefficients equal to 1 and $\sigma = 0.5$, the solver runs over
$\lambda \in \lbrace 10^{-4}, 10^{-2}, 0.1, 1, 10, 100, 1{,}000 \rbrace$. The coefficient error first falls from its
near-OLS value of 0.227 at $\lambda = 10^{-4}$ to 0.215 at $\lambda = 1$ and then grows quickly from $\lambda = 10$
on. The reference value $\lambda^\ast \approx 40 \cdot 0.25/40 = 0.25$ lies between the grid points 0.1 and 1. The
residual norm, in contrast, never decreases with $\lambda$, since shrinkage gives up fit to the observed
$\boldsymbol{y}$ in exchange for a smaller coefficient error. Past the optimum, both get worse.

```text
============================================================
RidgeSolver - Lambda Sweep
============================================================
Matrix size : 200 x 40

      lambda ||beta_hat - true||_2        ||residual||_2
--------------------------------------------------------
     1.0e-04            2.2659e-01            5.4832e+00
     1.0e-02            2.2645e-01            5.4832e+00
     1.0e-01            2.2517e-01            5.4834e+00
     1.0e+00            2.1547e-01            5.5047e+00
     1.0e+01            3.6360e-01            7.1417e+00
     1.0e+02            2.2770e+00            3.0543e+01
     1.0e+03            5.2962e+00            7.3895e+01
```

---

## Technical notes

- `RidgeSolver::solve(X, y, lambda)` is static and takes `Eigen::Ref` arguments, so mapped matrices bind without a
  copy.
- The solver fits no intercept and does not standardize. Center and scale the data first if the penalty should
  treat all columns alike.
- For a penalty grid with cross-validation, `model_selection/` offers `ridge_cv_svd`, which reuses one singular
  value decomposition (SVD) per fold for the whole grid, and `enet_cv_ccd`, which runs warm-started
  coordinate-descent paths.
