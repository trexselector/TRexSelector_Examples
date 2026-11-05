# Demo: PCA Fit, Restore, and Transform

## Purpose

This demo shows the `PCA` class for principal component analysis (PCA) in three workflows:

1. fitting PCA and reading the explained variance (EV),
2. restoring the input matrix after the in-place preprocessing,
3. transforming new data into the learned component space.

The data are pure Gaussian noise, so the output shows the interface and the noise baseline rather than a recovered
structure. The same workflow runs in [C++](../../../../cpp/ml_methods/pca/demo_mlm_pca_01/) and
[R](../../../../R/ml_methods/pca/demo_mlm_pca_01/). The Python version replaces the Eigen matrices with
Fortran-ordered `float64` NumPy arrays.

---

## Statistical model and notation

Let

- $\boldsymbol{X} \in \mathbb{R}^{n \times p}$ denote the data matrix with independent standard normal entries,
- $M \le \min(n, p)$ the number of retained principal components (PCs),
- $\boldsymbol{Z} \in \mathbb{R}^{n \times M}$ the score matrix,
- $\boldsymbol{V} \in \mathbb{R}^{p \times M}$ the loading matrix.

---

## PCA representation

`PCA(center=True, normalize=True)` holds the configuration only, and the matrix goes to `fit(X, M)`. The fit centers
every column of $\boldsymbol{X}$ in place and scales it to unit $\ell_2$-norm. It then computes the truncated singular
value decomposition (SVD) $\boldsymbol X \approx \boldsymbol U_M \boldsymbol\Sigma_M \boldsymbol V^\top$ of the
preprocessed matrix and returns a result with the fields `Z`, `V` and `explained_variance`,

```math
\boldsymbol{Z} = \boldsymbol{U}_M \boldsymbol{\Sigma}_M,
\qquad
\boldsymbol{X} \approx \boldsymbol{Z}\boldsymbol{V}^\top,
\qquad
\mathrm{EV}_k = \frac{s_k^2}{n-1},
```

where $s_k$ is the $k$-th singular value. After the normalization the total variance is $p/(n-1)$, so the EV of
component $k$ amounts to the share $s_k^2/p$. `restore()` undoes the normalization and then the centering of the
fitted matrix, and `transform(X_new)` applies the stored training means and norms to new rows before projecting them
onto $\boldsymbol{V}$.

---

## Running the demo

From the repository root, with TRexSelector 1.0.0 installed in the repository virtual environment `.venv`:

```bash
.venv/bin/python Python/ml_methods/pca/demo_mlm_pca_01/demo_mlm_pca_01.py
```

The demo prints to the console only and finishes in well under a second. The output below is a run against library
b95b984. The data are drawn with NumPy's `default_rng`, a permuted congruential generator (PCG64), seeded with the
values of the C++ demo (42, 123 and 7). The C++ demo draws with `std::mt19937`, so the two languages print different
numbers that agree qualitatively, not bitwise.

---

## Console walk-through

### Subproblem 1: Basic fit

With $n = 500$, $p = 200$ and $M = 10$, the fit returns a $500 \times 10$ score matrix and a $200 \times 10$ loading
matrix. The data hold no structure, so an infinite sample would spread the variance evenly at 0.5% per component, or
5% for ten components. In a finite sample, the eigenvalues of the sample correlation matrix fill the Marchenko-Pastur
interval $[(1-\sqrt{\gamma})^2, (1+\sqrt{\gamma})^2]$ with $\gamma = p/n = 0.4$. The first principal component (PC1)
holds 1.30%, an eigenvalue of about 2.60 (1.30% of 200). That is just below the upper edge of 2.66, where the largest
eigenvalue of a finite sample typically lies (the C++ run prints 1.33%). The ten leading of 200 components correspond to
the top 5% of the Marchenko-Pastur law, which predicts that they hold 11.97% of the variance. The run finds 11.92%, all
of it noise, while the C++ run finds 12.00%.

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
  PC2: 1.28%
  PC3: 1.25%
  PC4: 1.21%
  PC5: 1.18%
  PC6: 1.17%
  PC7: 1.15%
  PC8: 1.13%
  PC9: 1.12%
  PC10: 1.11%
  Cumulative (10 of 200): 11.92%
```

### Subproblem 2: Restore round-trip

PCA is fit on a working copy of a $300 \times 100$ matrix, and `restore()` brings the copy back. The largest deviation
from the original, $4.4 \cdot 10^{-16}$, is one unit in the last place of the entries between 2 and 4. The mean
deviation of $1.2 \cdot 10^{-17}$ lies far below one unit in the last place, so most entries come back bit for bit and
dividing by the column norm and multiplying back is exact up to rounding. The check tests the interface, not the
low-rank approximation error.

```text
==============================================================================
 PCA Restore Round-Trip Accuracy
==============================================================================

Matrix size  : 300 x 100
Components M : 5

Reconstruction error after restore():
  Max Error  : 4.4409e-16
  Mean Error : 1.2210e-17

✓ Restore round-trip passed
```

### Subproblem 3: Transform new data

PCA is fit on a $400 \times 80$ training matrix and then applied to 50 new rows, which yields $50 \times 8$ scores. The
demo keeps an untouched copy of the training data from before the in-place fit. On this copy, `transform()` reproduces
the scores of the fit up to $3.6 \cdot 10^{-15}$, which confirms that it uses the same means, norms and loadings.

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
Max diff (fit scores vs transform on same data): 3.5805e-15
✓ transform() reproduces fit scores
```

---

## Technical notes

- `fit(X, M)` keeps an `Eigen::Map` into the NumPy buffer until `restore()`, so `X` must be a writeable,
  Fortran-ordered `float64` array (`np.asfortranarray`). Unlike the C++ `restore(X)`, the Python `restore()` takes no
  argument.
- If the `PCA` object is destroyed before `restore()`, the C++ destructor writes the original data back, and `fit()`
  keeps `X` alive for as long as the `PCA` object exists.
- The binding has no counterpart of the C++ `PCA::readOnly(X, M)`, so `fit()` rejects a read-only array. Pass
  `X.copy(order="F")` for read-only data or where the original values are needed while the fit is active.
- The in-place, zero-copy design lets the same interface serve in-memory arrays and memory-mapped data larger than
  random-access memory (RAM), because the library never takes a hidden copy of `X`.
- `fit()` relies on the dimensionality-aware `SVDSolver`, shown in
  [demo_mlm_svd_01](../../svd/demo_mlm_svd_01/README.md).
