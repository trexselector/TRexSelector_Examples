# Demo: Elastic-Net Path Fitting and K-Fold CV via Coordinate Descent

## Purpose

This demo shows `enet_gaussian`, which fits elastic-net regularization paths by cyclic coordinate descent (CCD), and
`enet_cv_ccd`, which selects the penalty $\lambda$ by K-fold cross-validation (CV). Both follow the semantics of the R
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
[Python](../../../../Python/ml_methods/model_selection/demo_mlm_ms_02_enet_cv_ccd/) and
[R](../../../../R/ml_methods/model_selection/demo_mlm_ms_02_enet_cv_ccd/).

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

The entries of $\boldsymbol{X}$ are independent standard normal. As in the ridge demo, ten coefficients of
$\boldsymbol\beta_{\mathrm{true}}$ are set to 1 at evenly spaced positions and the rest are zero, so the response
variance is about $10 + \sigma^2 = 11$ and the true fraction of explained variance is $10/11 \approx 0.91$.

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
Scenario 4 are defined on this scale. The parameter $\alpha$ sets the penalty type: $\alpha = 0$ is ridge regression,
$\alpha = 1$ the lasso, and $0 < \alpha < 1$ the elastic net.

---

## Path fitting

For a fixed $\alpha$, `enet_gaussian` fits a decreasing sequence $\lambda_1 > \lambda_2 > \cdots > \lambda_K$ and
returns the coefficient vectors $\widehat{\boldsymbol\beta}(\lambda_k,\alpha)$. As in glmnet:

- the grid holds up to 100 values from $\lambda_{\max}$, the smallest penalty that keeps all coefficients at zero, down
  to $\lambda_{\max}/10{,}000$ for $n > p$ or $\lambda_{\max}/100$ for $n < p$,
- for ridge, $\lambda_{\max}$ is computed as for $\alpha = 0.001$, since no finite penalty zeroes a ridge fit,
- the path stops early once the explained fraction of the null deviance exceeds 0.999 or grows by less than a
  relative $10^{-5}$ between neighbors, while the optional sequential strong rule stays off by default, as in glmnet.

The demo prints every tenth path point and the last one, with the number of nonzero coefficients (`nnz`), the
$\ell_1$-norm of the coefficient vector and the deviance ratio (`dev.ratio`), the fraction of the null deviance the
fit explains.

---

## Cross-validation criterion

For a fixed $\alpha$, `enet_cv_ccd` builds the $\lambda$ grid once on the full data and reuses it in every fold, as
glmnet does. Averaging the held-out errors gives the curve of the cross-validated mean squared error (MSE),
$\mathrm{CV\text{-}MSE}(\lambda)$, with a standard error (SE) per grid point. The demo reports

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

From the repository root, after `cmake --preset release && cmake --build --preset release-build` in `cpp/`:

```bash
cpp/build/release/bin/ml_methods/model_selection/demo_mlm_ms_02_enet_cv_ccd/demo_mlm_ms_02_enet_cv_ccd
```

The demo prints to the console only and finishes in under a second. The output below is a run against library
b95b984 on an Apple Silicon machine. The data come from `std::mt19937` and `std::normal_distribution`. The latter is
implementation-defined, so another standard library (libstdc++ on Linux, for example) may draw different data and
print different numbers.

---

## Console walk-through

### Scenario 1: Low-dimensional path fitting

With $n = 300 > p = 100$, the ridge path keeps all 100 coefficients nonzero from its first point on and only shrinks
them. Its $\lambda_{\max}$ of 1,173.7 is 1,000 times the lasso's because of the $\alpha = 0.001$ rule. The elastic-net
and lasso paths start empty and hold 10 nonzeros, the number of active coefficients, at the second printed point, before
a slower spread to 97 and 96 nonzeros. They stop early after 72 and 71 points, when the deviance ratio no longer grows.
All three paths end at a deviance ratio of 0.94, above the true 0.91, because at the small end of the $\lambda$ grid,
100 coefficients also fit part of the noise. The `-0.00000` at the first point is a rounded tiny negative value. The
empty model explains no deviance.

```text
============================================================
Demo MS-02 | enet_gaussian path | Low-Dimensional (n=300, p=100)
============================================================
Data: n=300, p=100, active coefs=10

--- alpha=0.0 ---
  Path: 100 lambdas fitted
  lambda      nnz     L1(beta)    dev.ratio
  1173.69457  100     0.06948     0.00726
  462.92932   100     0.17474     0.01821
  182.58886   100     0.43423     0.04496
  72.01681    100     1.04825     0.10691
  28.40491    100     2.37115     0.23407
  11.20348    100     4.73590     0.44008
  4.41888     100     7.89850     0.67016
  1.74290     100     10.73163    0.83430
  0.68744     100     12.67387    0.91037
  0.27114     100     13.92427    0.93474
  0.11737     100     14.67894    0.94029

--- alpha=0.5 ---
  Path: 72 lambdas fitted
  lambda      nnz     L1(beta)    dev.ratio
  2.34739     0       0.00000     -0.00000
  0.92586     10      4.72723     0.65809
  0.36518     12      7.91820     0.86876
  0.14403     32      9.70893     0.91396
  0.05681     67      11.88747    0.93360
  0.02241     87      13.70997    0.94018
  0.00884     93      14.69553    0.94161
  0.00349     97      15.12643    0.94185
  0.00318     97      15.15174    0.94186

--- alpha=1.0 ---
  Path: 71 lambdas fitted
  lambda      nnz     L1(beta)    dev.ratio
  1.17369     0       0.00000     -0.00000
  0.46293     10      5.38662     0.71605
  0.18259     12      8.33556     0.88394
  0.07202     30      9.90989     0.91691
  0.02840     68      11.99141    0.93421
  0.01120     87      13.77344    0.94032
  0.00442     92      14.72511    0.94163
  0.00174     96      15.13829    0.94186
```

### Scenario 2: High-dimensional path fitting

With $p = 500 > n = 200$, the grid ends at $\lambda_{\max}/100$. The ridge path reaches a deviance ratio of only 0.68
there, because ridge needs a far smaller penalty to fit 500 coefficients to 200 rows. The lasso path runs its full 100
points, holds 10 nonzeros at $\lambda = 0.501$, and ends with 172 nonzeros and a deviance ratio of 0.997, close to
interpolation. A lasso fit has at most $n = 200$ nonzero coefficients.

```text
============================================================
Demo MS-02 | enet_gaussian path | High-Dimensional (n=200, p=500)
============================================================
Data: n=200, p=500, active coefs=10

--- alpha=0.0 ---
  Path: 100 lambdas fitted
  lambda      nnz     L1(beta)    dev.ratio
  1270.38415  500     0.25794     0.01747
  797.83827   500     0.40820     0.02756
  501.06569   500     0.64369     0.04327
  314.68385   500     1.00949     0.06738
  197.63063   500     1.57000     0.10366
  124.11780   500     2.41114     0.15660
  77.94959    500     3.63623     0.23046
  48.95462    500     5.34806     0.32724
  30.74493    500     7.61095     0.44389
  19.30871    500     10.40321    0.57067
  12.70384    500     13.26322    0.68140

--- alpha=1.0 ---
  Path: 100 lambdas fitted
  lambda      nnz     L1(beta)    dev.ratio
  1.27038     0       0.00000     -0.00000
  0.79784     7       1.76077     0.32372
  0.50107     10      4.36632     0.63259
  0.31468     11      6.52029     0.79631
  0.19763     18      8.03204     0.86803
  0.12412     45      9.54604     0.91195
  0.07795     88      11.55155    0.94860
  0.04895     119     13.65292    0.97289
  0.03074     144     15.58070    0.98696
  0.01931     165     17.05752    0.99366
  0.01270     172     18.17358    0.99693
```

The printout counts nonzero coefficients but does not show which ones entered the path.

### Scenario 3: Low-dimensional cross-validation

The three penalties see the same data. Lasso and elastic net reach a minimum CV-MSE of 1.30 and 1.34, close to the noise
floor of 1, because the truth is sparse. Ridge keeps all 100 coefficients and stays at 1.81. The ridge pick
$\lambda_{\min} = 0.1252$ agrees with the reference value $100 \cdot 3.3/(270 \cdot 10) \approx 0.12$. It is the
second-to-last grid point, because the $n > p$ grid ends at $\lambda_{\max}/10{,}000 \approx 0.114$, just below the
ridge optimum. The elastic-net and lasso picks lie well inside their grids.

```text
============================================================
Demo MS-02 | enet_cv_ccd | Low-Dimensional CV (n=300, p=100, 10-fold)
============================================================
Data: n=300, p=100, folds=10, active coefs=10

--- alpha=0.0 ---
  lambda.min : 0.12520  (CV-MSE=1.80910 +/- 0.19422)
  lambda.1se : 0.41962  (CV-MSE=1.96959 +/- 0.18633)

  CV-MSE curve (every 10 steps):
    lambda        CV-MSE      SEM         marker
    1140.77093    10.29014    0.54589
    449.94356     10.22028    0.54168
    177.46701     10.04827    0.53127
    69.99665      9.64222     0.50649
    27.60812      8.77009     0.45233
    10.88921      7.21366     0.35326
    4.29493       5.14979     0.22446
    1.69401       3.28933     0.15272
    0.66815       2.21471     0.17443
    0.26353       1.85377     0.19230
    0.12520       1.80910     0.19422       <-- min
    0.41962       1.96959     0.18633       <-- 1se

  Ordering check (lambda.min <= lambda.1se): PASS

--- alpha=0.5 ---
  lambda.min : 0.13999  (CV-MSE=1.33819 +/- 0.13901)
  lambda.1se : 0.24464  (CV-MSE=1.45915 +/- 0.13150)

  CV-MSE curve (every 10 steps):
    lambda        CV-MSE      SEM         marker
    2.28154       10.31803    0.55153
    0.89989       4.16226     0.23041
    0.35493       1.72790     0.13156
    0.13999       1.33819     0.13901       <-- min
    0.05522       1.44268     0.16233
    0.02178       1.59852     0.17689
    0.00859       1.73822     0.18631
    0.00339       1.81125     0.18928
    0.24464       1.45915     0.13150       <-- 1se

  Ordering check (lambda.min <= lambda.1se): PASS

--- alpha=1.0 ---
  lambda.min : 0.07682  (CV-MSE=1.29723 +/- 0.13887)
  lambda.1se : 0.14734  (CV-MSE=1.42860 +/- 0.13275)

  CV-MSE curve (every 10 steps):
    lambda        CV-MSE      SEM         marker
    1.14077       10.31235    0.55242
    0.44994       3.54394     0.20097
    0.17747       1.54042     0.13295
    0.07000       1.30109     0.14043
    0.02761       1.43688     0.16266
    0.01089       1.60096     0.17686
    0.00429       1.74103     0.18613
    0.00169       1.81273     0.18917
    0.07682       1.29723     0.13887       <-- min
    0.14734       1.42860     0.13275       <-- 1se

  Ordering check (lambda.min <= lambda.1se): PASS
```

### Scenario 4: Ridge CV and the TRexGVS conversion

TRexGVS can take its ridge penalty from a ridge CV fit. It converts the pick from glmnet's reported scale to the scale
of its least-angle regression (LARS) solver, as the R function `lm_dummy` does:

```math
\lambda_{2,\mathrm{lars}} = \lambda_{1\mathrm{se}} \cdot \frac{p}{2}.
```

With $p = 200$, the factor is 100. The ridge pick $\lambda_{\min} = 0.2809$ is close to the reference value
$200 \cdot 3.3/(270 \cdot 10) \approx 0.24$. In its `CV_1SE_CCD` mode, TRexGVS uses the more heavily regularized
$\lambda_{1\mathrm{se}}$, so its penalty is $\lambda_{2,\mathrm{lars}} = 64.90$ (under the default unit-norm column
scaling).

```text
============================================================
Demo MS-02 | enet_cv_ccd | Ridge CV + lambda_2_lars Conversion
============================================================
Data: n=300, p=200, folds=10, active coefs=10

  lambda.min : 0.28094  (CV-MSE=2.64430 +/- 0.29186)
  lambda.1se : 0.64901  (CV-MSE=2.90630 +/- 0.34961)

  CV-MSE curve (every 10 steps):
    lambda        CV-MSE      SEM         marker
    1216.12889    10.50160    1.23680
    479.66629     10.43015    1.23041
    189.19027     10.25521    1.21462
    74.62054      9.84771     1.17694
    29.43188      8.99670     1.09425
    11.60854      7.54805     0.94100
    4.57865       5.71659     0.72583
    1.80591       4.04688     0.51203
    0.71229       2.97251     0.36038
    0.28094       2.64430     0.29186       <-- min
    0.64901       2.90630     0.34961       <-- 1se

  lambda_2_lars (min) = cv_min  * p/2 = 0.280942 * 200/2 = 28.094155
  lambda_2_lars (1se) = cv_1se  * p/2 = 0.649011 * 200/2 = 64.901141

  (This is the value TRexGVS uses internally via CV_1SE_CCD.)
```

---

## Results at a glance

| Scenario | Size (n x p) | $\alpha$ | $\lambda_{\min}$ | $\lambda_{1\mathrm{se}}$ | CV-MSE at $\lambda_{\min}$ |
| --- | --- | --- | --- | --- | --- |
| 3 Low-dimensional CV | 300 x 100 | 0 (ridge) | 0.1252 | 0.4196 | 1.81 |
| 3 Low-dimensional CV | 300 x 100 | 0.5 (elastic net) | 0.1400 | 0.2446 | 1.34 |
| 3 Low-dimensional CV | 300 x 100 | 1 (lasso) | 0.0768 | 0.1473 | 1.30 |
| 4 Ridge + conversion | 300 x 200 | 0 (ridge) | 0.2809 | 0.6490 | 2.64 |

---

## Technical notes

- `enet_gaussian` uses pathwise cyclic coordinate descent with glmnet's path construction and stopping rules (the
  optional sequential strong rule is off by default), while `n_lambda` and `lambda_min_ratio` override the grid
  defaults.
- `enet_cv_ccd` builds the $\lambda$ grid once on the full dataset and reuses it across folds, rather than rebuilding a
  grid for every fold.
- The demo explains path behavior and penalty selection but does not benchmark prediction accuracy over replicated
  simulations.
