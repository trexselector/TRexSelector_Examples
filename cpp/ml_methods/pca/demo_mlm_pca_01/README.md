# Demo: PCA Fit, Restore, and Transform

## Purpose

This demo shows the `PCA` class for principal component analysis (PCA) in three workflows:

1. fitting PCA and reading the explained variance (EV),
2. restoring the input matrix after the in-place preprocessing,
3. transforming new data into the learned component space.

The data are pure Gaussian noise, so the output shows the interface and the noise baseline rather than a recovered
structure. The same workflow runs in [Python](../../../../Python/ml_methods/pca/demo_mlm_pca_01/) and
[R](../../../../R/ml_methods/pca/demo_mlm_pca_01/).

---

## Statistical model and notation

Let

- $\boldsymbol{X} \in \mathbb{R}^{n \times p}$ denote the data matrix with independent standard normal entries,
- $M \le \min(n, p)$ the number of retained principal components (PCs),
- $\boldsymbol{Z} \in \mathbb{R}^{n \times M}$ the score matrix,
- $\boldsymbol{V} \in \mathbb{R}^{p \times M}$ the loading matrix.

---

## PCA representation

At construction, `PCA(X, M)` centers every column of $\boldsymbol{X}$ in place and scales it to unit $\ell_2$-norm.
`fit()` then computes the truncated singular value decomposition (SVD)
$\boldsymbol X \approx \boldsymbol U_M \boldsymbol\Sigma_M \boldsymbol V^\top$ of the preprocessed matrix and returns

```math
\boldsymbol{Z} = \boldsymbol{U}_M \boldsymbol{\Sigma}_M,
\qquad
\boldsymbol{X} \approx \boldsymbol{Z}\boldsymbol{V}^\top,
\qquad
\mathrm{EV}_k = \frac{s_k^2}{n-1},
```

where $s_k$ is the $k$-th singular value. After the normalization the total variance is $p/(n-1)$, so the EV of
component $k$ amounts to the share $s_k^2/p$. `restore(X)` undoes the normalization and then the centering, and
`transform(X_new)` applies the stored training means and norms to new rows before projecting them onto
$\boldsymbol{V}$.

---

## Running the demo

From the repository root, after `cmake --preset release && cmake --build --preset release-build` in `cpp/`:

```bash
cpp/build/release/bin/ml_methods/pca/demo_mlm_pca_01/demo_mlm_pca_01
```

The demo prints to the console only and finishes in under a second. The output below comes from a run against
library b95b984 on an Apple Silicon machine. The data are drawn with `std::mt19937` and `std::normal_distribution`.
The latter is implementation-defined, so another standard library, such as libstdc++ on Linux, may draw different
data and print different numbers.

---

## Console walk-through

### Subproblem 1: Basic fit

With $n = 500$, $p = 200$ and $M = 10$, the fit returns a $500 \times 10$ score matrix and a $200 \times 10$ loading
matrix. The data hold no structure, so an infinite sample would spread the variance evenly at 0.5% per component. In
a finite sample, the eigenvalues of the sample correlation matrix fill the Marchenko-Pastur interval
$[(1-\sqrt{\gamma})^2, (1+\sqrt{\gamma})^2]$ with $\gamma = p/n = 0.4$. Its upper edge of 2.66 matches the 1.33%
(2.66/200) of the first principal component (PC1), and together the ten leading components hold 12%, all of it
noise.

```text
============================================================
PCA Basic Fit
============================================================
Matrix size  : 500 x 200
Components M : 10

Z dimensions : 500 x 10
V dimensions : 200 x 10
explained_variance size: 10

Explained variance per component (% of total):
  PC1: 1.33%
  PC2: 1.27%
  PC3: 1.24%
  PC4: 1.21%
  PC5: 1.20%
  PC6: 1.18%
  PC7: 1.16%
  PC8: 1.15%
  PC9: 1.13%
  PC10: 1.12%
  Cumulative (10 of 200): 12.00%
```

### Subproblem 2: Restore round-trip

PCA is fit on a working copy of a $300 \times 100$ matrix, and `restore()` brings the copy back. The largest
deviation from the original, $4.4 \cdot 10^{-16}$, is one unit in the last place of the larger entries, so
dividing by the column norm and multiplying back is exact up to rounding. The check tests the interface, not the
low-rank approximation error.

```text
============================================================
PCA Restore Round-Trip Accuracy
============================================================
Matrix size  : 300 x 100
Components M : 5

Reconstruction error after restore():
  Max Error  : 4.4409e-16
  Mean Error : 1.3604e-17

✓ Restore round-trip passed
```

### Subproblem 3: Transform new data

PCA is fit on a $400 \times 80$ training matrix and then applied to 50 new rows, which yields $50 \times 8$ scores.
On an untouched copy of the training data, `transform()` reproduces the scores of the fit up to $3.5 \cdot 10^{-15}$,
which confirms that it uses the same means, norms and loadings.

```text
============================================================
PCA Transform New Data
============================================================
Train size   : 400 x 80
Test size    : 50 x 80
Components M : 8

Train scores Z dimensions: 400 x 8
Test  scores Z dimensions: 50 x 8

✓ Transform output shape correct (50 x 8)
Max diff (fit scores vs transform on same data): 3.4694e-15
✓ transform() reproduces fit scores
```

---

## Technical notes

- `PCA(X, M)` takes an `Eigen::Ref` to the caller's matrix, here an `Eigen::Map`, and preprocesses it in place. If the
  object is destroyed before `restore()`, its destructor writes the original data back.
- `PCA::readOnly(X, M)` never writes to $\boldsymbol{X}$ and suits data that are already centered and scaled, as well
  as read-only memory-mapped files.
- `fit()` relies on the dimensionality-aware `SVDSolver`, shown in
  [demo_mlm_svd_01](../../svd/demo_mlm_svd_01/README.md).
