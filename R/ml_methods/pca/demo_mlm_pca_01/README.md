# Demo: PCA Fit, Restore, and Transform

## Purpose

This demo shows the `MLPCA` class for principal component analysis (PCA) in three workflows:

1. fitting PCA and reading the explained variance (EV),
2. restoring the input matrix after the in-place preprocessing,
3. transforming new data into the learned component space.

The data are pure Gaussian noise, so the output shows the interface and the noise baseline rather than a recovered
structure. The same workflow runs in [C++](../../../../cpp/ml_methods/pca/demo_mlm_pca_01/) and
[Python](../../../../Python/ml_methods/pca/demo_mlm_pca_01/). `MLPCA` is the R6 class around the C++ `PCA`, and the R
version replaces the Eigen matrices with ordinary double-storage R matrices. It also checks that the EV does not grow
from one component to the next and prints how far the in-place preprocessing moves the input.

---

## Statistical model and notation

Let

- $\boldsymbol{X} \in \mathbb{R}^{n \times p}$ denote the data matrix with independent standard normal entries,
- $M \le \min(n, p)$ the number of retained principal components (PCs),
- $\boldsymbol{Z} \in \mathbb{R}^{n \times M}$ the score matrix,
- $\boldsymbol{V} \in \mathbb{R}^{p \times M}$ the loading matrix.

---

## PCA representation

`MLPCA$new(X, M, center = TRUE, normalize = TRUE)` binds the object to $\boldsymbol{X}$ and, like the C++ constructor,
centers every column in place and scales it to unit $\ell_2$-norm right away. `$fit()` then computes the truncated
singular value decomposition (SVD) $\boldsymbol X \approx \boldsymbol U_M \boldsymbol\Sigma_M \boldsymbol V^\top$ of
the preprocessed matrix and caches the result, which `$get_scores()`, `$get_loadings()` and
`$get_explained_variance()` return,

```math
\boldsymbol{Z} = \boldsymbol{U}_M \boldsymbol{\Sigma}_M,
\qquad
\boldsymbol{X} \approx \boldsymbol{Z}\boldsymbol{V}^\top,
\qquad
\mathrm{EV}_k = \frac{s_k^2}{n-1},
```

where $s_k$ is the $k$-th singular value. After the normalization the total variance is $p/(n-1)$, so the EV of
component $k$ amounts to the share $s_k^2/p$. `$restore()` undoes the normalization and then the centering of the
bound matrix, and `$transform(X_new)` applies the stored training means and norms to new rows before projecting them
onto $\boldsymbol{V}$.

---

## Running the demo

From the repository root, with the TRexSelector R package 2.0.0 installed:

```bash
Rscript R/ml_methods/pca/demo_mlm_pca_01/demo_mlm_pca_01.R
```

The demo prints to the console only and finishes in about a second. The output below is a run against library
b95b984. The data come from R's default generator, a Mersenne-Twister with normals by inversion, seeded through
`set.seed()` with the values of the C++ demo (42, 123 and 7). The C++ demo uses `std::mt19937` with
`std::normal_distribution`, and `set.seed()` scrambles its seed before use. The two languages therefore print
different numbers that agree qualitatively, not bitwise.

---

## Console walk-through

### Subproblem 1: Basic fit

With $n = 500$, $p = 200$ and $M = 10$, the fit returns a $500 \times 10$ score matrix and a $200 \times 10$ loading
matrix. The data hold no structure, so an infinite sample would spread the variance evenly at 0.5% per component, or
5% for ten components. In a finite sample, the eigenvalues of the sample correlation matrix fill the Marchenko-Pastur
interval $[(1-\sqrt{\gamma})^2, (1+\sqrt{\gamma})^2]$ with $\gamma = p/n = 0.4$. The first principal component (PC1)
holds 1.30%, an eigenvalue of about 2.60 (1.30% of 200). That is just below the upper edge of 2.66, where the largest
eigenvalue of a finite sample typically lies, and the Python run prints the same 1.30% while the C++ run prints 1.33%.
The ten leading of 200 components correspond to the top 5% of the Marchenko-Pastur law, which predicts that they hold
11.97% of the variance. The R run finds 11.89%, all of it noise, while the Python and C++ runs find 11.92% and 12.00%.

The `stopifnot()` check that the R version adds confirms that the EV does not increase from one component to the
next, up to a tolerance of $10^{-12}$.

```text
==============================================================================
 PCA Basic Fit
==============================================================================

Matrix size  : 500 x 200
Components M : 10

Z dimensions : 500 x 10
V dimensions : 200 x 10
explained_variance size: 10

Explained variance per component (% of total):
  PC1: 1.30%
  PC2: 1.26%
  PC3: 1.23%
  PC4: 1.21%
  PC5: 1.18%
  PC6: 1.16%
  PC7: 1.15%
  PC8: 1.14%
  PC9: 1.13%
  PC10: 1.12%
  Cumulative (10 of 200): 11.89%
✓ Explained variance is non-increasing across PCs
```

### Subproblem 2: Restore round-trip

`MLPCA$new()` preprocesses a $300 \times 100$ matrix in place, and the demo keeps a deep copy from before, taken with
`X + 0`. After the fit, the R version prints the largest deviation of the bound matrix from that copy. It amounts to
4.07, because the unit-norm scaling shrinks the entries to the order of $1/\sqrt{300} \approx 0.06$, so the deviation
reflects the largest entries of the draw. `$restore()` then brings the matrix back. The largest remaining deviation
from the original, $4.4 \cdot 10^{-16}$, is one unit in the last place of the entries between 2 and 4. The mean
deviation of $1.1 \cdot 10^{-17}$ lies far below one unit in the last place, so most entries come back bit for bit and
dividing by the column norm and multiplying back is exact up to rounding. The check tests the interface, not the
low-rank approximation error.

```text
==============================================================================
 PCA Restore Round-Trip Accuracy
==============================================================================

Matrix size  : 300 x 100
Components M : 5

Max |X - X_orig| after fit()     : 4.0688e+00  (X standardized in place)

Reconstruction error after restore():
  Max Error  : 4.4409e-16
  Mean Error : 1.1390e-17

✓ Restore round-trip passed
```

### Subproblem 3: Transform new data

PCA is fit on a $400 \times 80$ training matrix and then applied to 50 new rows, which yields $50 \times 8$ scores. The
demo keeps a deep copy of the training data from before the in-place preprocessing. On this copy, `$transform()`
reproduces the scores of the fit up to $4.8 \cdot 10^{-15}$, which confirms that it uses the same means, norms and
loadings. The Python and C++ runs print $3.6 \cdot 10^{-15}$ and $3.5 \cdot 10^{-15}$.

```text
==============================================================================
 PCA Transform New Data
==============================================================================

Train size   : 400 x 80
Test size    : 50 x 80
Components M : 8

Train scores Z dimensions: 400 x 8
Test  scores Z dimensions: 50 x 8

✓ Transform output shape correct (50 x 8)
Max diff (fit scores vs transform on same data): 4.8295e-15
✓ transform() reproduces fit scores
```

---

## Technical notes

- `MLPCA$new(X, M)` maps the memory of the R matrix through an `Eigen::Map` and preprocesses it in place at
  construction. The object stays bound to `X`, so `$restore()` takes no argument, unlike the C++ `restore(X)`.
  `$restore()` may run before or after `$fit()` and is idempotent, while `$fit()` is single-use.
- If the object is garbage-collected before `$restore()`, its finalizer writes the original data back.
- A plain alias `X_orig <- X` shares the buffer that the library changes, so the demo takes deep copies with `X + 0`
  where the original values are needed while the object is bound. A matrix of another storage mode, such as integer,
  is converted to a double copy first, and only that copy is preprocessed.
- `MLPCA$new()` also accepts an `mmap_matrix` handle, which must be open in `"readwrite"` mode because the
  preprocessing writes to the file. The binding has no counterpart of the C++ `PCA::readOnly(X, M)`.
- The in-place, zero-copy design lets the same interface serve in-memory matrices and memory-mapped data larger than
  random-access memory (RAM), because the library never takes a hidden copy of `X`.
- `$fit()` relies on the dimensionality-aware `SVDSolver`, which R exposes as `MLSVD` and
  [demo_mlm_svd_01](../../svd/demo_mlm_svd_01/README.md) shows.
