# Demo: Elastic-Net Path Fitting and K-Fold CV via Coordinate Descent

## Purpose

This demo shows `ElasticNet`, which fits elastic-net regularization paths by cyclic coordinate descent (CCD), and
`ElasticNetCV`, which selects the penalty $\lambda$ by K-fold cross-validation (CV). Both come from
`trexselector.ml_methods`, bind the library's `enet_gaussian` and `enet_cv_ccd`, and follow the semantics of the
R package glmnet. The demo shows how the path changes with the mixing parameter $\alpha$ and how CV picks $\lambda$
for ridge, elastic net and lasso, including the ridge conversion used by Grouped Variable Selection (GVS) in
TRexSelector (`TRexGVS`). It has four scenarios:

1. **Low-dimensional path fitting**: paths for $\alpha \in \lbrace 0, 0.5, 1 \rbrace$ with $n = 300$, $p = 100$.
2. **High-dimensional path fitting**: paths for $\alpha \in \lbrace 0, 1 \rbrace$ with $n = 200$, $p = 500$.
3. **Low-dimensional cross-validation**: 10-fold CV for $\alpha \in \lbrace 0, 0.5, 1 \rbrace$ with $n = 300$,
   $p = 100$.
4. **Ridge CV with the TRexGVS conversion**: 10-fold CV for $\alpha = 0$ with $n = 300$, $p = 200$, and the conversion
   of the CV pick to the `lambda_2_lars` scale used downstream.

The same workflow is implemented in
[C++](../../../../cpp/ml_methods/model_selection/demo_mlm_ms_02_enet_cv_ccd/) and
[R](../../../../R/ml_methods/model_selection/demo_mlm_ms_02_enet_cv_ccd/). The Python demo adds a reproduction check
to every path fit, which the C++ demo does not have.

---

## Statistical model and notation

Let

- $\boldsymbol{X} \in \mathbb{R}^{n \times p}$ denote the design matrix,
- $\boldsymbol{y} \in \mathbb{R}^n$ the response vector,
- $\boldsymbol{\beta} \in \mathbb{R}^p$ the regression coefficients,
- $\lambda \ge 0$ the penalty parameter,
- $\alpha \in [0,1]$ the elastic-net mixing parameter.

The synthetic data follow the Gaussian linear model

```math
\boldsymbol{y} = \boldsymbol{X}\boldsymbol{\beta}_{\mathrm{true}} + \boldsymbol{\varepsilon},
\qquad
\boldsymbol{\varepsilon} \sim \mathcal{N}(\boldsymbol{0}, \sigma^2 \boldsymbol{I}_n),
\qquad
\sigma = 1.
```

The entries of $\boldsymbol{X}$ are independent standard normal and are drawn into a Fortran-ordered `float64` array,
the layout both classes require. As in the ridge demo, ten coefficients of $\boldsymbol\beta_{\mathrm{true}}$ are set
to 1 at evenly spaced positions and the rest are zero, so the response variance is about $10 + \sigma^2 = 11$ and the
true fraction of explained variance (EV) is $10/11 \approx 0.91$.

---

## Elastic-net objective

For fixed $\alpha$ and $\lambda$, the elastic-net estimator solves

```math
\widehat{\boldsymbol{\beta}}(\lambda,\alpha) =
\arg\min_{\boldsymbol{\beta} \in \mathbb{R}^p}
\left\lbrace
\frac{1}{2n}\lVert \boldsymbol{y} - \boldsymbol{X}\boldsymbol{\beta} \rVert_2^2
+
\lambda
\left[
\alpha \lVert \boldsymbol{\beta} \rVert_1
+
\frac{1-\alpha}{2}\lVert \boldsymbol{\beta} \rVert_2^2
\right]
\right\rbrace.
```

The residual sum of squares carries glmnet's $1/(2n)$ normalization, and the $\lambda$ grid and the conversion in
Scenario 4 are defined on this scale. The parameter $\alpha$ sets the penalty type. $\alpha = 0$ is ridge regression,
$\alpha = 1$ the lasso, and $0 < \alpha < 1$ the elastic net.

---

## Path fitting

For a fixed $\alpha$, `ElasticNet().fit(X, y, alpha)` fits a decreasing sequence
$\lambda_1 > \lambda_2 > \cdots > \lambda_K$ and returns the coefficient vectors
$\widehat{\boldsymbol\beta}(\lambda_k,\alpha)$. As in glmnet:

- the grid holds up to 100 values from $\lambda_{\max}$, the smallest penalty that keeps all coefficients at zero, down
  to $\lambda_{\max}/10{,}000$ for $n > p$ or $\lambda_{\max}/100$ for $n < p$,
- for ridge, $\lambda_{\max}$ is computed as for $\alpha = 0.001$, since no finite penalty zeroes a ridge fit,
- the path stops early once the explained fraction of the null deviance exceeds 0.999 or grows by less than a relative
  $10^{-5}$ between neighbors.

A sequential strong rule for screening coefficients is available through `use_strong_rule=True`, and the demo keeps
the default `False`. It prints every tenth path point and the last one, with the number of nonzero coefficients
(`nnz`, magnitude above $10^{-9}$), the $\ell_1$-norm of the coefficient vector and the deviance ratio (`dev.ratio`),
the fraction of the null deviance the fit explains.

Two checks follow every path, and both are Python additions. The demo refits the model with
`ElasticNet().fit_grid()` at the grid the automatic fit produced and prints the largest absolute coefficient
difference between the two fits. It also asserts, without printing, that the deviance ratio never decreases along
the path (with a tolerance of $10^{-12}$), because less penalty can only improve the training fit.

---

## Cross-validation criterion

For a fixed $\alpha$, `ElasticNetCV().fit(X, y, alpha=alpha, n_folds=10, seed=0)` builds the $\lambda$ grid once on
the full data and reuses it in every fold, as glmnet does. Averaging the held-out errors gives the curve of the
cross-validated mean squared error (MSE), $\mathrm{CV\text{-}MSE}(\lambda)$, with a standard error (SE) per grid
point. The demo reports

- $\lambda_{\min}$: the value minimizing the CV-MSE,
- $\lambda_{1\mathrm{se}}$: the largest value whose CV-MSE lies within one SE of the minimum,

and checks the ordering $\lambda_{\min} \le \lambda_{1\mathrm{se}}$.

Like glmnet, the routine divides $\boldsymbol{y}$ by its standard deviation $\mathrm{sd}(y)$ before fitting and
reports $\lambda$ in the original units. The lasso term is unaffected, but the ridge penalty actually applied is
$\lambda/\mathrm{sd}(y)$.
With the ridge reference from the
[ridge CV demo](../demo_mlm_ms_01_ridge_cv_svd/README.md#cross-validation-criterion), the expected ridge pick on the
reported scale is therefore about

```math
\lambda^\ast \approx
\frac{p\,\sigma^2\,\mathrm{sd}(y)}{n_{\mathrm{train}}\,\lVert \boldsymbol{\beta}_{\mathrm{true}} \rVert_2^2},
\qquad
\mathrm{sd}(y) \approx \sqrt{11} \approx 3.3,
```

with $n_{\mathrm{train}} = 270$ training rows per fold in Scenarios 3 and 4.

---

## Running the demo

From the repository root, with the repository's virtual environment `.venv` set up as described in
[`LIBRARY_TARGET.txt`](../../../../LIBRARY_TARGET.txt):

```bash
.venv/bin/python Python/ml_methods/model_selection/demo_mlm_ms_02_enet_cv_ccd/demo_mlm_ms_02_enet_cv_ccd.py
```

The demo prints to the console only and finishes in about a second. The output below is a run against library
b95b984, with the package TRexSelector 1.0.0 built from that commit. The data come from NumPy's permuted
congruential generator (PCG64) via `numpy.random.default_rng`, with the same data seeds as the C++ demo. The C++ demo
draws from `std::mt19937`, so the two languages see different data, and their numbers agree qualitatively, not
bitwise.

---

## Console walk-through

The run opens with a banner for the suite and closes with `All enet_cv_ccd demos complete.`, and every scenario
prints its own block.

### Scenario 1: Low-dimensional path fitting

With $n = 300 > p = 100$, the ridge path keeps all 100 coefficients nonzero from its first point on and only shrinks
them. Its $\lambda_{\max}$ of 1,318.19 is 1,000 times the lasso's 1.318 because of the $\alpha = 0.001$ rule. The
elastic-net and lasso paths start empty and hold 10 nonzeros, the number of active coefficients, at the second printed
point, before a slower spread to all 100 coefficients. They stop early after 74 and 73 points, when the deviance ratio
stops growing. All three paths end at a deviance ratio of 0.93, above the true EV fraction of 0.91, because at the
small end of the $\lambda$ grid, 100 coefficients also fit part of the noise. The `-0.00000` at the first point is a
rounded tiny negative value, since the empty model explains no deviance. Each `fit_grid()` check prints a difference
of exactly zero, so the explicit-grid fit reproduces the automatic path bit for bit.

```text
==============================================================================
 Demo MS-02 | enet_gaussian path | Low-Dimensional (n=300, p=100)
==============================================================================

Data: n=300, p=100, active coefs=10

--- alpha=0.0 ---
  Path: 100 lambdas fitted
  lambda      nnz     L1(beta)    dev.ratio
  1318.18879  100     0.05945     0.00615
  519.92082   100     0.14971     0.01545
  205.06749   100     0.37324     0.03831
  80.88284    100     0.90784     0.09199
  31.90186    100     2.08700     0.20554
  12.58275    100     4.28601     0.39944
  4.96290     100     7.36802     0.63111
  1.95747     100     10.31741    0.80733
  0.77207     100     12.10737    0.89430
  0.30452     100     13.41216    0.92430
  0.13182     100     14.16185    0.93163

  fit_grid() reproduction check (max |coef diff| = 0.00e+00): PASS

--- alpha=0.5 ---
  Path: 74 lambdas fitted
  lambda      nnz     L1(beta)    dev.ratio
  2.63638     0       0.00000     -0.00000
  1.03984     10      3.94838     0.58460
  0.41013     11      7.39752     0.84853
  0.16177     20      9.19495     0.90388
  0.06380     56      11.18330    0.92441
  0.02517     83      12.93661    0.93152
  0.00993     98      14.11088    0.93343
  0.00391     100     14.67177    0.93380
  0.00296     100     14.76274    0.93383

  fit_grid() reproduction check (max |coef diff| = 0.00e+00): PASS

--- alpha=1.0 ---
  Path: 73 lambdas fitted
  lambda      nnz     L1(beta)    dev.ratio
  1.31819     0       0.00000     -0.00000
  0.51992     10      4.58271     0.64913
  0.20507     11      7.86957     0.86851
  0.08088     19      9.42237     0.90779
  0.03190     54      11.29675    0.92519
  0.01258     84      13.00362    0.93170
  0.00496     97      14.14420    0.93346
  0.00196     100     14.68642    0.93380
  0.00163     100     14.74743    0.93382

  fit_grid() reproduction check (max |coef diff| = 0.00e+00): PASS
```

### Scenario 2: High-dimensional path fitting

With $p = 500 > n = 200$, the grid ends at $\lambda_{\max}/100$. The ridge path reaches a deviance ratio of only 0.63
there, because ridge needs a far smaller penalty to fit 500 coefficients to 200 rows. The lasso path runs its full 100
points, holds 10 nonzeros at $\lambda = 0.547$, and ends with 173 nonzeros and a deviance ratio of 0.996, close to
interpolation. A lasso fit has at most $n = 200$ nonzero coefficients. Here the empty first lasso point prints as
`0.00000` without a sign, and both `fit_grid()` checks again print a difference of zero.

```text
==============================================================================
 Demo MS-02 | enet_gaussian path | High-Dimensional (n=200, p=500)
==============================================================================

Data: n=200, p=500, active coefs=10

--- alpha=0.0 ---
  Path: 100 lambdas fitted
  lambda      nnz     L1(beta)    dev.ratio
  1387.34257  500     0.22520     0.01533
  871.29157   500     0.35655     0.02421
  547.19650   500     0.56264     0.03805
  343.65535   500     0.88333     0.05936
  215.82557   500     1.37597     0.09155
  135.54475   500     2.11817     0.13883
  85.12605    500     3.20672     0.20544
  53.46164    500     4.74235     0.29399
  33.57547    500     6.79952     0.40289
  21.08637    500     9.38324     0.52454
  13.87343    500     12.10056    0.63456

  fit_grid() reproduction check (max |coef diff| = 0.00e+00): PASS

--- alpha=1.0 ---
  Path: 100 lambdas fitted
  lambda      nnz     L1(beta)    dev.ratio
  1.38734     0       0.00000     0.00000
  0.87129     6       0.80564     0.16637
  0.54720     10      3.65436     0.54578
  0.34366     11      6.20692     0.76410
  0.21583     14      7.83674     0.85188
  0.13554     36      9.28781     0.89994
  0.08513     75      11.30332    0.94181
  0.05346     108     13.37474    0.96903
  0.03358     139     15.23963    0.98448
  0.02109     157     16.80667    0.99267
  0.01387     173     17.92426    0.99644

  fit_grid() reproduction check (max |coef diff| = 0.00e+00): PASS
```

The printout counts nonzero coefficients but does not show which ones entered the path.

### Scenario 3: Low-dimensional cross-validation

The three penalties see the same data and the same folds. Lasso and elastic net reach a minimum CV-MSE of 1.17 and
1.21, close to the noise floor of 1, because the truth is sparse. Ridge keeps all 100 coefficients and stays at 1.65.
The ridge pick $\lambda_{\min} = 0.16251$ lies above the reference value $100 \cdot 3.3/(270 \cdot 10) \approx 0.12$,
but in a flat part of the curve. At the last printed point $\lambda = 0.25876$, the CV-MSE of 1.67218 is only about a
fifth of one SE above the minimum. The pick sits near the low end of the ridge grid, which the $n > p$ rule puts at
$\lambda_{\max}/10{,}000 \approx 0.112$, while the elastic-net and lasso picks lie well inside their grids. The lasso
minimum falls on a printed grid point, so its marker appears inside the curve.

```text
==============================================================================
 Demo MS-02 | enet_cv_ccd | Low-Dimensional CV (n=300, p=100, 10-fold)
==============================================================================

Data: n=300, p=100, folds=10, active coefs=10

--- alpha=0.0 ---
  lambda.min : 0.16251  (CV-MSE=1.65219 +/- 0.09495)
  lambda.1se : 0.37541  (CV-MSE=1.73275 +/- 0.08964)

  CV-MSE curve (every 10 steps):
    lambda        CV-MSE      SEM         marker
    1120.09396    10.08618    0.98595
    441.78814     10.01046    0.98049
    174.25035     9.82436     0.96689
    68.72793      9.38691     0.93400
    27.10771      8.45649     0.85954
    10.69184      6.82827     0.71349
    4.21708       4.73775     0.49164
    1.66330       2.93609     0.25802
    0.65604       1.96059     0.11282
    0.25876       1.67218     0.09002
    0.16251       1.65219     0.09495       <-- min
    0.37541       1.73275     0.08964       <-- 1se

  Ordering check (lambda.min <= lambda.1se): PASS

--- alpha=0.5 ---
  lambda.min : 0.12524  (CV-MSE=1.20788 +/- 0.04728)
  lambda.1se : 0.18171  (CV-MSE=1.24271 +/- 0.05931)

  CV-MSE curve (every 10 steps):
    lambda        CV-MSE      SEM         marker
    2.24019       10.09268    0.99323
    0.88358       3.87802     0.47009
    0.34850       1.58176     0.12523
    0.13746       1.21042     0.04928
    0.05422       1.29278     0.05836
    0.02138       1.47656     0.08139
    0.00843       1.61180     0.09680
    0.00333       1.67973     0.10441
    0.12524       1.20788     0.04728       <-- min
    0.18171       1.24271     0.05931       <-- 1se

  Ordering check (lambda.min <= lambda.1se): PASS

--- alpha=1.0 ---
  lambda.min : 0.06873  (CV-MSE=1.17044 +/- 0.04798)
  lambda.1se : 0.10943  (CV-MSE=1.20713 +/- 0.05855)

  CV-MSE curve (every 10 steps):
    lambda        CV-MSE      SEM         marker
    1.12009       10.07874    0.99412
    0.44179       3.28051     0.39537
    0.17425       1.40841     0.09578
    0.06873       1.17044     0.04798       <-- min
    0.02711       1.28859     0.06053
    0.01069       1.48095     0.08192
    0.00422       1.61506     0.09715
    0.00166       1.68143     0.10457
    0.10943       1.20713     0.05855       <-- 1se

  Ordering check (lambda.min <= lambda.1se): PASS
```

### Scenario 4: Ridge CV and the TRexGVS conversion

TRexGVS can take its ridge penalty from a ridge CV fit. It converts the pick from glmnet's reported scale to the scale
of its least-angle regression (LARS) solver, as the R function `lm_dummy` does:

```math
\lambda_{2,\mathrm{lars}} = \lambda_{1\mathrm{se}} \cdot \frac{p}{2}.
```

With $p = 200$, the factor is 100. The ridge pick $\lambda_{\min} = 0.2233$ is close to the reference value
$200 \cdot 3.3/(270 \cdot 10) \approx 0.24$. In its `CV_1SE_CCD` mode, TRexGVS uses the more heavily regularized
$\lambda_{1\mathrm{se}} = 0.5158$, so its penalty is $\lambda_{2,\mathrm{lars}} = 51.58$ (under the default unit-norm
column scaling).

```text
==============================================================================
 Demo MS-02 | enet_cv_ccd | Ridge CV + lambda_2_lars Conversion
==============================================================================

Data: n=300, p=200, folds=10, active coefs=10

  lambda.min : 0.22326  (CV-MSE=2.32147 +/- 0.28765)
  lambda.1se : 0.51577  (CV-MSE=2.56116 +/- 0.29360)

  CV-MSE curve (every 10 steps):
    lambda        CV-MSE      SEM         marker
    1277.60431    11.95011    0.72330
    503.91347     11.86406    0.72031
    198.75385     11.65318    0.71300
    78.39262      11.16090    0.69604
    30.91966      10.12878    0.66062
    12.19535      8.36396     0.59804
    4.81010       6.13649     0.50559
    1.89720       4.13449     0.39438
    0.74830       2.84879     0.30945
    0.29514       2.34587     0.28652
    0.22326       2.32147     0.28765       <-- min
    0.51577       2.56116     0.29360       <-- 1se

  lambda_2_lars (min) = cv_min  * p/2 = 0.223265 * 200/2 = 22.326498
  lambda_2_lars (1se) = cv_1se  * p/2 = 0.515771 * 200/2 = 51.577106

  (This is the value TRexGVS uses internally via CV_1SE_CCD.)
```

---

## Results at a glance

| Scenario | Size (n x p) | $\alpha$ | $\lambda_{\min}$ | $\lambda_{1\mathrm{se}}$ | CV-MSE at $\lambda_{\min}$ |
| --- | --- | --- | --- | --- | --- |
| 3 Low-dimensional CV | 300 x 100 | 0 (ridge) | 0.1625 | 0.3754 | 1.65 |
| 3 Low-dimensional CV | 300 x 100 | 0.5 (elastic net) | 0.1252 | 0.1817 | 1.21 |
| 3 Low-dimensional CV | 300 x 100 | 1 (lasso) | 0.0687 | 0.1094 | 1.17 |
| 4 Ridge + conversion | 300 x 200 | 0 (ridge) | 0.2233 | 0.5158 | 2.32 |

---

## Technical notes

- `ElasticNet` uses pathwise cyclic coordinate descent with glmnet's path construction and stopping rules, while
  `n_lambda` and `lambda_min_ratio` override the grid defaults. The sequential strong rule is off by default.
- `ElasticNetCV` builds the $\lambda$ grid once on the full dataset and reuses it across folds, rather than
  rebuilding a grid for every fold. The grid is stored in glmnet's descending order (largest penalty first), unlike
  the ascending grid of `RidgeCV`.
- Both classes accept only `float64` arrays in Fortran (column-major) order, which `np.asfortranarray()` provides.
- The data seeds are 101, 202, 303 and 404 as in the C++ demo, and both CV scenarios fix the fold seed to 0, which is
  also the C++ default.
- The `fit_grid()` reproduction check and the silent deviance-ratio assertion are Python additions. A failing check
  stops the demo with an `AssertionError`.
- The demo explains path behavior and penalty selection but does not benchmark prediction accuracy over replicated
  simulations.
