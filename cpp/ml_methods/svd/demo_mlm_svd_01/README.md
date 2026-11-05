# Demo: `SVDSolver` Usage

## Purpose

This demo shows `SVDSolver`, which computes a truncated singular value decomposition (SVD) and chooses the algorithm
by the shape of the matrix. It covers three subproblems:

1. the direct path for $n \ge p$,
2. the Gram path for a wide matrix with $p > 2n$,
3. an orthogonality check of the returned singular vectors.

It checks the interface numerically and is not a performance benchmark. The same workflow runs in
[Python](../../../../Python/ml_methods/svd/demo_mlm_svd_01/) and [R](../../../../R/ml_methods/svd/demo_mlm_svd_01/).

---

## Mathematical setup

For a matrix $\mathbf{X} \in \mathbb{R}^{n \times p}$, the singular value decomposition is
$\mathbf{X} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top$ with singular values $s_1 \ge s_2 \ge \dots \ge 0$.
Keeping the top $M$ of them gives the truncated decomposition

```math
\mathbf{X}_M = \mathbf{U}_M \boldsymbol{\Sigma}_M \mathbf{V}_M^\top,
\qquad
\mathbf{U}_M \in \mathbb{R}^{n \times M},\;
\boldsymbol{\Sigma}_M \in \mathbb{R}^{M \times M},\;
\mathbf{V}_M \in \mathbb{R}^{p \times M}.
```

By the Eckart-Young theorem, $\mathbf X_M$ is the best approximation of rank $M$ in the Frobenius norm, and its error
is fixed by the discarded singular values:

```math
\frac{\lVert \mathbf{X}_M - \mathbf{X} \rVert_F}{\lVert \mathbf{X} \rVert_F}
=
\frac{\sqrt{\sum_{k > M} s_k^2}}{\lVert \mathbf{X} \rVert_F}.
```

The demo checks this identity against the singular values of a full `Eigen::JacobiSVD`. `SVDSolver` takes the Gram
path for $p > 2n$ and a randomized SVD when both $n$ and $p$ exceed 10,000 and $M$ stays below a tenth of
$\min(n, p)$. All other shapes go through a thin Jacobi SVD.

---

## Running the demo

From the repository root, after `cmake --preset release && cmake --build --preset release-build` in `cpp/`:

```bash
cpp/build/release/bin/ml_methods/svd/demo_mlm_svd_01/demo_mlm_svd_01
```

The demo prints to the console only and finishes in under a second. The output below comes from a run against
library b95b984 on an Apple Silicon machine. The data are drawn with `std::mt19937` and `std::normal_distribution`.
The latter is implementation-defined, so another standard library, such as libstdc++ on Linux, may draw different
data and print different numbers.

---

## Console walk-through

### Subproblem 1: Direct path

A $300 \times 80$ Gaussian matrix goes through the thin Jacobi SVD with $M = 10$. The top singular value of 25.97
sits just below $\sqrt{n} + \sqrt{p} \approx 26.26$, the upper edge for a Gaussian matrix, and the spectrum falls off
slowly because the matrix has no low-rank structure. The ten leading components therefore keep only
$1 - 0.8709^2 \approx 24\%$ of the squared Frobenius norm, and the relative error of 0.8709 equals the Eckart-Young
optimum to within $1.1 \cdot 10^{-14}$.

```text
============================================================
SVDSolver - Direct Path (n >= p)
============================================================
Matrix size : 300 x 80
M           : 10

U dimensions : 300 x 10
S size       : 10
V dimensions : 80 x 10

Relative reconstruction error ||X_M - X||_F / ||X||_F : 8.7087e-01
Eckart-Young optimum sqrt(sum_{k>M} s_k^2) / ||X||_F  : 8.7087e-01
Eckart-Young check (|difference| = 1.1324e-14 < 1e-10): PASS

Top 10 singular values:
  s1 = 25.9719
  s2 = 25.4824
  s3 = 25.4040
  s4 = 24.1664
  s5 = 23.8591
  s6 = 23.5689
  s7 = 23.5130
  s8 = 23.2767
  s9 = 22.6883
  s10 = 22.5729
```

### Subproblem 2: Gram path

For the wide $100 \times 500$ matrix, the solver computes the eigendecomposition of the $100 \times 100$ Gram matrix
$\mathbf{X}\mathbf{X}^\top$ instead of working on the 500 columns, at a cost of order $n^2 p + n^3$. Forming
$\mathbf{X}\mathbf{X}^\top$ squares the condition number and can cost accuracy in the small singular values. For the
eight retained components, however, the Eckart-Young check finds no loss, as the error of 0.9226 matches the optimum
to within $1.5 \cdot 10^{-14}$. These eight of 100 components keep about 15% of the squared norm.

```text
============================================================
SVDSolver - Gram Path (p > 2*n)
============================================================
Matrix size : 100 x 500  (p > 2*n → Gram path)
M           : 8

U dimensions : 100 x 8
S size       : 8
V dimensions : 500 x 8

Relative reconstruction error ||X_M - X||_F / ||X||_F : 9.2257e-01
Eckart-Young optimum sqrt(sum_{k>M} s_k^2) / ||X||_F  : 9.2257e-01
Eckart-Young check (|difference| = 1.4766e-14 < 1e-10): PASS
```

### Subproblem 3: Orthogonality check

For a $200 \times 150$ matrix with $M = 20$, $\mathbf{U}^\top\mathbf{U}$ and $\mathbf{V}^\top\mathbf{V}$ deviate from
the $20 \times 20$ identity by $1.5 \cdot 10^{-13}$ and $1.6 \cdot 10^{-13}$ in the Frobenius norm, far below the
threshold of $10^{-10}$.

```text
============================================================
SVDSolver - Orthogonality Check
============================================================
Matrix size : 200 x 150
M           : 20

||U^T U - I||_F : 1.4629e-13
||V^T V - I||_F : 1.5536e-13

✓ U and V are orthonormal
```

---

## Technical notes

- The Jacobi SVD is preferred over a divide-and-conquer SVD because it stays accurate under aggressive
  floating-point optimization flags.
- The randomized path is not reached at these sizes. `SVDSolver(min_dim_thresh, block_size)` sets its dimension
  threshold (default 10,000) and the block width of the matrix products (default 1,024).
- `compute()` throws unless $1 \le M \le \min(n, p)$.
