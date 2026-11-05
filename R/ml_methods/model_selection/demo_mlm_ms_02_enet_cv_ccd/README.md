# Demo: Elastic-Net Path Fitting and K-Fold CV via Coordinate Descent

## Purpose

This demo shows `MLElasticNet`, which fits elastic-net regularization paths by cyclic coordinate descent (CCD), and
`MLElasticNetCV`, which selects the penalty $\lambda$ by K-fold cross-validation (CV). Both are R6 classes of the
TRexSelector R package, bind the library's `enet_gaussian` and `enet_cv_ccd`, and follow the semantics of the R
package glmnet. The demo shows how the path changes with the mixing parameter $\alpha$ and how CV picks $\lambda$ for
ridge, elastic net and lasso, including the ridge conversion used by Grouped Variable Selection (GVS) in TRexSelector
(`TRexGVS`). It has four scenarios:

1. **Low-dimensional path fitting**: paths for $\alpha \in \lbrace 0, 0.5, 1 \rbrace$ with $n = 300$, $p = 100$.
2. **High-dimensional path fitting**: paths for $\alpha \in \lbrace 0, 1 \rbrace$ with $n = 200$, $p = 500$.
3. **Low-dimensional cross-validation**: 10-fold CV for $\alpha \in \lbrace 0, 0.5, 1 \rbrace$ with $n = 300$,
   $p = 100$.
4. **Ridge CV with the TRexGVS conversion**: 10-fold CV for $\alpha = 0$ with $n = 300$, $p = 200$, and the conversion
   of the CV pick to the `lambda_2_lars` scale used downstream.

The same workflow is implemented in
[C++](../../../../cpp/ml_methods/model_selection/demo_mlm_ms_02_enet_cv_ccd/) and
[Python](../../../../Python/ml_methods/model_selection/demo_mlm_ms_02_enet_cv_ccd/). Like the Python demo, the R demo
adds a reproduction check to every path fit, which the C++ demo does not have.

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

The entries of $\boldsymbol{X}$ are independent standard normal and are drawn with `rnorm()` into an ordinary numeric
R matrix, which both classes bind without a copy. As in the ridge demo, ten coefficients of
$\boldsymbol\beta_{\mathrm{true}}$ are set to 1 at evenly spaced positions and the rest are zero, so the response
variance is about $10 + \sigma^2 = 11$ and the true fraction of explained variance (EV) is $10/11 \approx 0.91$.

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

For a fixed $\alpha$, `MLElasticNet$new()$fit(X, y, alpha)` fits a decreasing sequence
$\lambda_1 > \lambda_2 > \cdots > \lambda_K$ and returns the coefficient vectors
$\widehat{\boldsymbol\beta}(\lambda_k,\alpha)$. As in glmnet:

- the grid holds up to 100 values from $\lambda_{\max}$, the smallest penalty that keeps all coefficients at zero, down
  to $\lambda_{\max}/10{,}000$ for $n > p$ or $\lambda_{\max}/100$ for $n < p$,
- for ridge, $\lambda_{\max}$ is computed as for $\alpha = 0.001$, since no finite penalty zeroes a ridge fit,
- the path stops early once the explained fraction of the null deviance exceeds 0.999 or grows by less than a relative
  $10^{-5}$ between neighbors.

A sequential strong rule for screening coefficients is available through `use_strong_rule = TRUE`, and the demo keeps
the default `FALSE`. It prints every tenth path point and the last one, with the number of nonzero coefficients
(`nnz`, magnitude above $10^{-9}$), the $\ell_1$-norm of the coefficient vector and the deviance ratio (`dev.ratio`),
the fraction of the null deviance the fit explains.

Two checks follow every path, and the R and Python demos share both. The demo refits the model with `$fit_grid()` at
the grid the automatic fit produced and prints the largest absolute coefficient difference between the two fits,
which `stopifnot()` requires to stay below $10^{-12}$. It also asserts, without printing, that the deviance ratio
never decreases along the path (with a tolerance of $10^{-12}$), because less penalty can only improve the training
fit.

---

## Cross-validation criterion

For a fixed $\alpha$, `MLElasticNetCV$new()` with `$fit(X, y, alpha = alpha, n_folds = 10L, seed = 0L)` builds the
$\lambda$ grid once on the full data and reuses it in every fold, as glmnet does. Averaging the held-out errors gives
the curve of the cross-validated mean squared error (MSE), $\mathrm{CV\text{-}MSE}(\lambda)$, with a standard error
(SE) per grid point. The demo reports

- $\lambda_{\min}$: the value minimizing the CV-MSE,
- $\lambda_{1\mathrm{se}}$: the largest value whose CV-MSE lies within one SE of the minimum,

and checks the ordering $\lambda_{\min} \le \lambda_{1\mathrm{se}}$. In Scenario 3 it also asserts, without
printing, that the CV-MSE and SE vectors hold one entry per grid point, since early termination can leave fewer than
100 of them.

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

From the repository root, with the TRexSelector R package installed as described in
[`LIBRARY_TARGET.txt`](../../../../LIBRARY_TARGET.txt):

```bash
Rscript R/ml_methods/model_selection/demo_mlm_ms_02_enet_cv_ccd/demo_mlm_ms_02_enet_cv_ccd.R
```

The demo prints to the console only and finishes in about a second. The output below is a run against library
b95b984, with the package TRexSelector 2.0.0 built from that commit. The data come from R's Mersenne-Twister
generator via `set.seed()` and `rnorm()`, which draws normals by inversion, with the same data seeds as the C++ demo.
The C++ demo draws from `std::mt19937` and the Python demo from NumPy's `default_rng`, so the three languages see
different data, and their numbers agree qualitatively, not bitwise.

---

## Console walk-through

The run opens with a banner for the suite and closes with `All enet_cv_ccd demos complete.`, and every scenario
prints its own block.

### Scenario 1: Low-dimensional path fitting

With $n = 300 > p = 100$, the ridge path keeps all 100 coefficients nonzero from its first point on and only shrinks
them. Its $\lambda_{\max}$ of 1,261.30 is 1,000 times the lasso's 1.261 because of the $\alpha = 0.001$ rule. The
elastic-net and lasso paths start empty and hold 10 nonzeros, the number of active coefficients, at the second and
third printed points, before a slower spread to 98 of the 100 coefficients. Both stop early after 74 points, when the
deviance ratio stops growing. All three paths end at a deviance ratio of about 0.94, above the true EV fraction of
0.91, because at the small end of the $\lambda$ grid, 100 coefficients also fit part of the noise. The `-0.00000` at
the first point is a rounded tiny negative value, since the empty model explains no deviance. Each `fit_grid()` check
prints a difference of exactly zero, so the explicit-grid fit reproduces the automatic path bit for bit.

```text
==============================================================================
 Demo MS-02 | enet_gaussian path | Low-Dimensional (n=300, p=100)
==============================================================================

Data: n=300, p=100, active coefs=10

--- alpha=0.0 ---
  Path: 100 lambdas fitted
  lambda      nnz     L1(beta)    dev.ratio
  1261.29501  100     0.05768     0.00613
  497.48074   100     0.14520     0.01539
  196.21666   100     0.36161     0.03818
  77.39189    100     0.87740     0.09169
  30.52496    100     2.00590     0.20498
  12.03967    100     4.07718     0.39916
  4.74869     100     6.91939     0.63372
  1.87298     100     9.62125     0.81483
  0.73874     100     11.67464    0.90338
  0.29138     100     13.21218    0.93219
  0.12613     100     14.15331    0.93869

  fit_grid() reproduction check (max |coef diff| = 0.00e+00): PASS

--- alpha=0.5 ---
  Path: 74 lambdas fitted
  lambda      nnz     L1(beta)    dev.ratio
  2.52259     0       0.00000     -0.00000
  0.99496     10      4.18002     0.61493
  0.39243     10      7.45464     0.85798
  0.15478     20      9.34286     0.91261
  0.06105     48      11.05670    0.93086
  0.02408     82      12.82681    0.93802
  0.00950     93      14.10545    0.94007
  0.00375     98      14.71645    0.94046
  0.00283     98      14.82185    0.94050

  fit_grid() reproduction check (max |coef diff| = 0.00e+00): PASS

--- alpha=1.0 ---
  Path: 74 lambdas fitted
  lambda      nnz     L1(beta)    dev.ratio
  1.26130     0       0.00000     -0.00000
  0.49748     10      4.80621     0.67674
  0.19622     10      7.90536     0.87631
  0.07739     22      9.57448     0.91638
  0.03052     52      11.18013    0.93165
  0.01204     81      12.89473    0.93819
  0.00475     93      14.14058    0.94011
  0.00187     98      14.73314    0.94047
  0.00142     98      14.83462    0.94050

  fit_grid() reproduction check (max |coef diff| = 0.00e+00): PASS
```

### Scenario 2: High-dimensional path fitting

With $p = 500 > n = 200$, the grid ends at $\lambda_{\max}/100$. The ridge path reaches a deviance ratio of only 0.68
there, because ridge needs a far smaller penalty to fit 500 coefficients to 200 rows. The lasso path runs its full 100
points, holds 10 nonzeros at $\lambda = 0.718$ and $\lambda = 0.451$, and ends with 177 nonzeros and a deviance ratio
of 0.997, close to interpolation. A lasso fit has at most $n = 200$ nonzero coefficients. The empty first lasso point
again prints as `-0.00000`, and both `fit_grid()` checks again print a difference of zero.

```text
==============================================================================
 Demo MS-02 | enet_gaussian path | High-Dimensional (n=200, p=500)
==============================================================================

Data: n=200, p=500, active coefs=10

--- alpha=0.0 ---
  Path: 100 lambdas fitted
  lambda      nnz     L1(beta)    dev.ratio
  1143.09525  500     0.27189     0.01807
  717.89713   500     0.42998     0.02850
  450.86032   500     0.67733     0.04468
  283.15342   500     1.06056     0.06946
  177.82860   500     1.64549     0.10655
  111.68154   500     2.51860     0.16033
  70.13926    500     3.78115     0.23473
  44.04950    500     5.52891     0.33119
  27.66437    500     7.81515     0.44620
  17.37403    500     10.61746    0.57013
  11.43095    500     13.47990    0.67812

  fit_grid() reproduction check (max |coef diff| = 0.00e+00): PASS

--- alpha=1.0 ---
  Path: 100 lambdas fitted
  lambda      nnz     L1(beta)    dev.ratio
  1.14310     0       0.00000     -0.00000
  0.71790     10      2.27386     0.39054
  0.45086     10      5.10984     0.70713
  0.28315     11      6.89499     0.83221
  0.17783     20      8.12666     0.88577
  0.11168     42      9.56650     0.92506
  0.07014     77      11.32857    0.95490
  0.04405     110     13.19399    0.97501
  0.02766     143     14.98709    0.98713
  0.01737     155     16.58188    0.99397
  0.01143     177     17.72414    0.99710

  fit_grid() reproduction check (max |coef diff| = 0.00e+00): PASS
```

The printout counts nonzero coefficients but does not show which ones entered the path.

### Scenario 3: Low-dimensional cross-validation

The three penalties see the same data and the same folds. Lasso and elastic net reach a minimum CV-MSE of 1.28 and
1.32, close to the noise floor of 1, because the truth is sparse. Ridge keeps all 100 coefficients and stays at 1.69.
The ridge pick $\lambda_{\min} = 0.11893$ agrees with the reference value $100 \cdot 3.3/(270 \cdot 10) \approx 0.12$.
It sits near the low end of the ridge grid, which the $n > p$ rule puts at $\lambda_{\max}/10{,}000 \approx 0.099$,
while the elastic-net and lasso picks lie well inside their grids. The curves list the grid in decreasing $\lambda$,
as their header `lambda descending` states. All six picks fall between printed grid points, so their marker rows
follow the curve.

```text
==============================================================================
 Demo MS-02 | enet_cv_ccd | Low-Dimensional CV (n=300, p=100, 10-fold)
==============================================================================

Data: n=300, p=100, folds=10, active coefs=10

--- alpha=0.0 ---
  lambda.min : 0.11893  (CV-MSE=1.68786 +/- 0.09168)
  lambda.1se : 0.30153  (CV-MSE=1.77645 +/- 0.09720)

  CV-MSE curve (every 10 steps, lambda descending):
    lambda        CV-MSE      SEM         marker
    987.36597     8.97020     0.67298
    389.43748     8.91464     0.66960
    153.60217     8.77769     0.66122
    60.58386      8.45370     0.64107
    23.89552      7.75373     0.59589
    9.42489       6.48538     0.50732
    3.71737       4.74499     0.36814
    1.46621       3.08502     0.21270
    0.57830       2.06898     0.11661
    0.22809       1.72316     0.09435
    0.11893       1.68786     0.09168       <-- min
    0.30153       1.77645     0.09720       <-- 1se

  Ordering check (lambda.min <= lambda.1se): PASS

--- alpha=0.5 ---
  lambda.min : 0.11040  (CV-MSE=1.31620 +/- 0.05274)
  lambda.1se : 0.17579  (CV-MSE=1.34961 +/- 0.07107)

  CV-MSE curve (every 10 steps, lambda descending):
    lambda        CV-MSE      SEM         marker
    1.97473       8.99001     0.67633
    0.77887       3.76065     0.28253
    0.30720       1.62653     0.09909
    0.12117       1.31649     0.05668
    0.04779       1.37695     0.04435
    0.01885       1.53148     0.06203
    0.00743       1.65532     0.07945
    0.00293       1.71805     0.08864
    0.11040       1.31620     0.05274       <-- min
    0.17579       1.34961     0.07107       <-- 1se

  Ordering check (lambda.min <= lambda.1se): PASS

--- alpha=1.0 ---
  lambda.min : 0.07297  (CV-MSE=1.27552 +/- 0.06085)
  lambda.1se : 0.11619  (CV-MSE=1.32935 +/- 0.07572)

  CV-MSE curve (every 10 steps, lambda descending):
    lambda        CV-MSE      SEM         marker
    0.98737       8.98475     0.67662
    0.38944       3.17226     0.23426
    0.15360       1.45441     0.08606
    0.06058       1.27841     0.05390
    0.02390       1.37290     0.04478
    0.00942       1.53606     0.06248
    0.00372       1.65884     0.07981
    0.00147       1.71966     0.08875
    0.07297       1.27552     0.06085       <-- min
    0.11619       1.32935     0.07572       <-- 1se

  Ordering check (lambda.min <= lambda.1se): PASS
```

### Scenario 4: Ridge CV and the TRexGVS conversion

TRexGVS can take its ridge penalty from a ridge CV fit. It converts the pick from glmnet's reported scale to the scale
of its least-angle regression (LARS) solver, as the R function `lm_dummy` does:

```math
\lambda_{2,\mathrm{lars}} = \lambda_{1\mathrm{se}} \cdot \frac{p}{2}.
```

With $p = 200$, the factor is 100. The ridge pick $\lambda_{\min} = 0.2585$ is close to the reference value
$200 \cdot 3.3/(270 \cdot 10) \approx 0.24$. In its `CV_1SE_CCD` mode, TRexGVS uses the more heavily regularized
$\lambda_{1\mathrm{se}} = 0.4958$, so its penalty is $\lambda_{2,\mathrm{lars}} = 49.58$ (under the default unit-norm
column scaling).

```text
==============================================================================
 Demo MS-02 | enet_cv_ccd | Ridge CV + lambda_2_lars Conversion
==============================================================================

Data: n=300, p=200, folds=10, active coefs=10

  lambda.min : 0.25851  (CV-MSE=2.80921 +/- 0.17311)
  lambda.1se : 0.49579  (CV-MSE=2.95820 +/- 0.16855)

  CV-MSE curve (every 10 steps, lambda descending):
    lambda        CV-MSE      SEM         marker
    1228.11544    10.66814    0.56723
    484.39404     10.59228    0.56204
    191.05499     10.40665    0.54942
    75.35602      9.97478     0.52050
    29.72197      9.07620     0.46247
    11.72296      7.56353     0.37210
    4.62378       5.71218     0.27679
    1.82371       4.13572     0.20922
    0.71931       3.17089     0.17263
    0.28371       2.81358     0.17197
    0.25851       2.80921     0.17311       <-- min
    0.49579       2.95820     0.16855       <-- 1se

  lambda_2_lars (min) = cv_min  * p/2 = 0.258507 * 200/2 = 25.850653
  lambda_2_lars (1se) = cv_1se  * p/2 = 0.495792 * 200/2 = 49.579232

  (This is the value TRexGVS uses internally via CV_1SE_CCD.)
```

---

## Results at a glance

| Scenario | Size (n x p) | $\alpha$ | $\lambda_{\min}$ | $\lambda_{1\mathrm{se}}$ | CV-MSE at $\lambda_{\min}$ |
| --- | --- | --- | --- | --- | --- |
| 3 Low-dimensional CV | 300 x 100 | 0 (ridge) | 0.1189 | 0.3015 | 1.69 |
| 3 Low-dimensional CV | 300 x 100 | 0.5 (elastic net) | 0.1104 | 0.1758 | 1.32 |
| 3 Low-dimensional CV | 300 x 100 | 1 (lasso) | 0.0730 | 0.1162 | 1.28 |
| 4 Ridge + conversion | 300 x 200 | 0 (ridge) | 0.2585 | 0.4958 | 2.81 |

---

## Technical notes

- `MLElasticNet` uses pathwise cyclic coordinate descent with glmnet's path construction and stopping rules, while
  the `$fit()` arguments `n_lambda` and `lambda_min_ratio` override the grid defaults. The sequential strong rule is
  off by default.
- `MLElasticNetCV` builds the $\lambda$ grid once on the full dataset and reuses it across folds, rather than
  rebuilding a grid for every fold. The grid is stored in glmnet's descending order (largest penalty first), unlike
  the ascending grid of `MLRidgeCV`.
- The demo reads the paths with `$get_lambdas()`, `$get_coef()` (a $p \times K$ matrix on the original predictor
  scale) and `$get_dev_ratio()`, and the CV results with `$cv_min()`, `$cv_1se()`, `$index_min()`, `$index_1se()`,
  `$get_cv_errors()` and `$get_cv_std()`.
- `$index_min()` and `$index_1se()` return 1-based indices into the descending grid, the R convention, while the C++
  and Python accessors are 0-based.
- The data seeds are 101, 202, 303 and 404 as in the C++ demo, and both CV scenarios fix the fold seed to 0, which is
  also the C++ default. The fold permutation runs in the library, so it does not depend on R's generator.
- The `fit_grid()` reproduction check and the silent deviance-ratio assertion also run in the Python demo, while the
  length assertions of Scenario 3 are R additions. A failing check stops the demo with a `stopifnot()` error.
- The demo explains path behavior and penalty selection but does not benchmark prediction accuracy over replicated
  simulations.
