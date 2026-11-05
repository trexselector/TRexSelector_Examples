# Demo: `MLSVD` Usage

## Purpose

This demo shows `MLSVD`, the R6 class around the C++ `SVDSolver`, which computes a truncated singular value
decomposition (SVD) and chooses the algorithm by the shape of the matrix. It covers three subproblems:

1. the direct path for $n \ge p$, with a full-rank reconstruction check that the R and Python versions add,
2. the Gram path for a wide matrix with $p > 2n$,
3. an orthogonality check of the returned singular vectors.

It checks the interface numerically and is not a performance benchmark. The same workflow runs in
[C++](../../../../cpp/ml_methods/svd/demo_mlm_svd_01/) and [Python](../../../../Python/ml_methods/svd/demo_mlm_svd_01/).
The R version replaces the Eigen matrices with ordinary double-storage R matrices.

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

The demo checks this identity against the singular values of `base::svd`, which calls the Linear Algebra PACKage
(LAPACK), so the reference is independent of the Eigen code in the library. At full rank $M = p$ the discarded sum is
empty, and the R addition checks that $\mathbf X_p$ reproduces $\mathbf X$ up to rounding. `SVDSolver` takes the Gram
path for $p > 2n$ and a randomized SVD when both $n$ and $p$ exceed 10,000 and $M$ stays below a tenth of
$\min(n, p)$. All other shapes go through a thin Jacobi SVD.

---

## Running the demo

From the repository root, with the TRexSelector R package 2.0.0 installed:

```bash
Rscript R/ml_methods/svd/demo_mlm_svd_01/demo_mlm_svd_01.R
```

The demo prints to the console only and finishes in about a second. The output below is a run against library
b95b984. The data come from R's default generator, a Mersenne-Twister with normals by inversion, seeded through
`set.seed()` with the values of the C++ demo (42, 7 and 99). The C++ demo uses `std::mt19937` with
`std::normal_distribution`, and `set.seed()` scrambles its seed before use. The two languages therefore print
different numbers that agree qualitatively, not bitwise.

---

## Console walk-through

### Subproblem 1: Direct path

A $300 \times 80$ Gaussian matrix goes through the thin Jacobi SVD with $M = 10$. The top singular value of 26.03 sits
just below $\sqrt{n} + \sqrt{p} \approx 26.26$, the edge of the spectrum for large Gaussian matrices, where the largest
singular value of a sample of this size typically lies. The Python and C++ runs print 26.68 and 25.97. The spectrum
falls off slowly because the matrix has no low-rank structure. The ten leading components therefore keep only
$1 - 0.8712^2 \approx 24\%$ of the squared Frobenius norm. The relative error of 0.8712 equals the Eckart-Young
optimum to within $2.2 \cdot 10^{-16}$, two units in the last place, and the C++ run shows $1.1 \cdot 10^{-14}$.

The R addition then computes the decomposition at full rank ($M = 80$). It rebuilds $\mathbf X$ with a relative error
of $2.5 \cdot 10^{-14}$, far below the threshold of $10^{-10}$ that its `stopifnot()` enforces. A second
`stopifnot()`, which prints nothing, checks that the singular values come sorted in decreasing order.

```text
==============================================================================
 SVDSolver - Direct Path (n >= p)
==============================================================================

Matrix size : 300 x 80
M           : 10

U dimensions : 300 x 10
S size       : 10
V dimensions : 80 x 10

Relative reconstruction error ||X_M - X||_F / ||X||_F : 8.7120e-01
Eckart-Young optimum sqrt(sum_{k>M} s_k^2) / ||X||_F  : 8.7120e-01
Eckart-Young check (|difference| = 2.2204e-16 < 1e-10): PASS

Top 10 singular values:
  s1 = 26.0334
  s2 = 25.4062
  s3 = 24.8067
  s4 = 24.3979
  s5 = 24.3162
  s6 = 23.9276
  s7 = 23.7395
  s8 = 23.5194
  s9 = 23.1935
  s10 = 22.8344

Full-rank (M = p) relative reconstruction error : 2.5317e-14
✓ Full-rank reconstruction is exact
```

### Subproblem 2: Gram path

For the wide $100 \times 500$ matrix, the solver computes the eigendecomposition of the $100 \times 100$ Gram matrix
$\mathbf{X}\mathbf{X}^\top$ instead of working on the 500 columns, at a cost of order $n^2 p + n^3$. Forming
$\mathbf{X}\mathbf{X}^\top$ squares the condition number and can cost accuracy in the small singular values. For the
eight retained components, however, the Eckart-Young check finds no loss, as the error of 0.9226 matches the optimum
to within $5.6 \cdot 10^{-16}$, five units in the last place. These eight of 100 components keep
$1 - 0.9226^2 \approx 15\%$ of the squared norm.

```text
==============================================================================
 SVDSolver - Gram Path (p > 2*n)
==============================================================================

Matrix size : 100 x 500  (p > 2*n -> Gram path)
M           : 8

U dimensions : 100 x 8
S size       : 8
V dimensions : 500 x 8

Relative reconstruction error ||X_M - X||_F / ||X||_F : 9.2261e-01
Eckart-Young optimum sqrt(sum_{k>M} s_k^2) / ||X||_F  : 9.2261e-01
Eckart-Young check (|difference| = 5.5511e-16 < 1e-10): PASS
```

### Subproblem 3: Orthogonality check

For a $200 \times 150$ matrix with $M = 20$, $\mathbf{U}^\top\mathbf{U}$ and $\mathbf{V}^\top\mathbf{V}$ deviate from
the $20 \times 20$ identity by $1.3 \cdot 10^{-13}$ each in the Frobenius norm, far below the threshold of
$10^{-10}$.

```text
==============================================================================
 SVDSolver - Orthogonality Check
==============================================================================

Matrix size : 200 x 150
M           : 20

||U^T U - I||_F : 1.2787e-13
||V^T V - I||_F : 1.3196e-13

✓ U and V are orthonormal
```

---

## Technical notes

- The Jacobi SVD is preferred over a divide-and-conquer SVD because it stays accurate under aggressive
  floating-point optimization flags.
- The randomized path is not reached at these sizes. `MLSVD$new(min_dim_thresh = 10000, block_size = 1024)` sets its
  dimension threshold (default 10,000) and the block width of the matrix products (default 1,024).
- `$compute(X, M)` stops with an error unless $1 \le M \le \min(n, p)$. It returns a named list with `U`, `S` and
  `V` of sizes $n \times M$, $M$ and $p \times M$.
- `$compute()` reads `X` through a read-only map. A double-storage R matrix binds without a copy, a matrix of another
  storage mode is converted to double first, and an `mmap_matrix` handle binds zero-copy in `"readonly"` mode.
- The reconstruction errors, the Eckart-Young reference and the two added checks are computed on the R side with base
  R (`norm()`, `svd()` and `crossprod()`).
