# Demo: `MLZScoreScaler` and `MLLpNormScaler`

## Purpose

This demo shows two column-scaling utilities used for data normalization, the R6 classes around the C++
`ZScoreScaler` and `LpNormScaler`:

- `MLZScoreScaler`, which centers columns and divides them by their standard deviation (SD),
- `MLLpNormScaler`, which centers columns and rescales them to unit $L_2$- or $L_1$-norm.

Both follow the `center` and `scale` switches of R's `scale()`. Beyond the in-memory comparison of these switches,
the demo covers inverse-transform accuracy, serialization of fitted scaler parameters and scaling on a memory-mapped
matrix. The same workflow runs in [C++](../../../../cpp/ml_methods/scaler_methods/demo_mlm_scaler_01/) and
[Python](../../../../Python/ml_methods/scaler_methods/demo_mlm_scaler_01/). The R version replaces the Eigen matrices
with ordinary double-storage R matrices and runs the memory-mapped subproblem at a twentieth of the C++ size in each
dimension. It also measures the round-trip error on the mapped matrix, where the C++ and Python demos only list the
steps.

---

## Mathematical setup

Let $\boldsymbol{X} \in \mathbb{R}^{n \times p}$ denote a data matrix with columns
$\boldsymbol x_1, \dots, \boldsymbol x_p$. For each column $j$, both scalers subtract a center $c_j$ and divide by a
scale $s_j$,

```math
\tilde{\boldsymbol{x}}_j = \frac{\boldsymbol{x}_j - c_j \mathbf{1}}{s_j},
\qquad
c_j =
\begin{cases}
\bar{x}_j & \text{center on},\\
0 & \text{center off}.
\end{cases}
```

`MLZScoreScaler` divides by the root mean square around the center,

```math
s_j = \sqrt{\frac{1}{n-1} \sum_{i=1}^{n} (x_{ij} - c_j)^2},
```

which is the SD when centering is on, and by $s_j = 1$ when `scale` is off. `MLLpNormScaler` divides by the norm
$s_j = \lVert \boldsymbol x_j - c_j \mathbf 1 \rVert_q$ with $q \in \lbrace 1, 2 \rbrace$. The inverse transform
multiplies by $s_j$ and adds $c_j$ back.

---

## Running the demo

From the repository root, with the TRexSelector R package 2.0.0 installed:

```bash
Rscript R/ml_methods/scaler_methods/demo_mlm_scaler_01/demo_mlm_scaler_01.R
```

The demo prints to the console only and finishes in about a second. The output below is a run against library
b95b984. The data come from R's default generator, a Mersenne-Twister with normals by inversion, seeded through
`set.seed()` with the values of the C++ demo (123, 456 and 999). The C++ demo uses `std::mt19937` with
`std::normal_distribution`, and `set.seed()` scrambles its seed before use. The two languages therefore print
different numbers that agree qualitatively, not bitwise. The serialization and memory-mapped files go to the temporary
directory of the R session and are removed at the end.

---

## Console walk-through

The suite opens with a banner and sets the thread count of the library backend to six, as the C++ and Python demos
do.

```text
==============================================================================
 DataTransformer Demo Suite
==============================================================================

Running with 6 threads
```

### Subproblem 1: Center and scale switches

The demo draws a $1{,}000 \times 500$ matrix from $\mathcal N(5, 3^2)$, so the effect of each switch is visible. It
prints three column statistics averaged over all columns, the absolute mean, the SD and the $L_2$-norm. The
untransformed columns have mean 5.00, SD 3.0024 and norm 184.42, close to
$\sqrt{n(\mu^2 + \sigma^2)} = \sqrt{34{,}000} \approx 184.39$. The seven scaled copies read as follows:

- Full z-scoring gives the mean $3.2 \cdot 10^{-16}$, the SD 1.0000 and the norm $31.61 = \sqrt{999}$, which shows
  the $n - 1$ denominator.
- Without centering, the scaler divides by the root mean square around 0, about $\sqrt{34{,}000/999} \approx 5.83$,
  as R's `scale()` does. The mean becomes $5/5.83 \approx 0.857$ and the SD $3/5.83 \approx 0.515$, while the norm
  is again $\sqrt{999}$.
- Centering alone leaves the SD at 3.0024 and gives the norm $\sqrt{999} \cdot 3.0024 \approx 94.90$.
- With both switches off, the transform is the identity and reproduces the original row.
- The centered $L_2$ scaler gives unit norms and the SD $1/\sqrt{999} \approx 0.0316$.
- The uncentered $L_2$ scaler divides by the raw norm, so the mean becomes $5/184.42 \approx 0.0271$ and the SD
  $3/184.42 \approx 0.0163$.
- The centered $L_1$ scaler divides by an $L_1$-norm of about $n\sigma\sqrt{2/\pi} \approx 2{,}394$, which leaves
  the SD 0.0013 and the $L_2$-norm 0.04.

```text
==============================================================================
 Scaler Comparison - center / scale switches
==============================================================================

Data: 1000 x 500 drawn from N(mean=5, sd=3)

  original (untransformed)            avg|mean| = 5.00e+00   avg SD = 3.0024   avg L2 = 184.42

  ZScore(center=TRUE,  scale=TRUE )   avg|mean| = 3.22e-16   avg SD = 1.0000   avg L2 = 31.61
  ZScore(center=FALSE, scale=TRUE )   avg|mean| = 8.57e-01   avg SD = 0.5146   avg L2 = 31.61
  ZScore(center=TRUE,  scale=FALSE)   avg|mean| = 9.69e-16   avg SD = 3.0024   avg L2 = 94.90
  ZScore(center=FALSE, scale=FALSE)   avg|mean| = 5.00e+00   avg SD = 3.0024   avg L2 = 184.42

  LpNorm(L2, center=TRUE )            avg|mean| = 1.02e-17   avg SD = 0.0316   avg L2 = 1.00
  LpNorm(L2, center=FALSE)            avg|mean| = 2.71e-02   avg SD = 0.0163   avg L2 = 1.00
  LpNorm(L1, center=TRUE )            avg|mean| = 4.04e-19   avg SD = 0.0013   avg L2 = 0.04
```

### Subproblem 2: Inverse-transform accuracy

On a $1{,}000 \times 5{,}000$ standard normal matrix, `$fit()`, `$transform_inplace()` and
`$inverse_transform_inplace()` run in sequence for `MLZScoreScaler` and the centered $L_2$ `MLLpNormScaler`. Both
round trips end with a largest error of $8.9 \cdot 10^{-16}$, one unit in the last place of the entries between 4 and
8, which the largest of the five million draws reach. The mean error of about $4 \cdot 10^{-18}$ lies far below one
unit in the last place, so most entries come back bit for bit.

```text
==============================================================================
 Scaler Inverse Transform Accuracy
==============================================================================

Matrix size: 1000 x 5000

ZScoreScaler:
  Max Error: 8.881784e-16
  Mean Error: 4.200594e-18

LpNormScaler L2:
  Max Error: 8.881784e-16
  Mean Error: 4.055381e-18
```

### Subproblem 3: Serialization

An `MLZScoreScaler` is fit on a $1{,}000 \times 500$ standard normal matrix, saved with `$save()` to a temporary file
and loaded into a fresh object with `$load()`. The stored means and scales differ by exactly zero, so the binary file
holds the doubles without loss. The console shows the file as `<tmp>/` plus the name that `tempfile()` draws, so the
name changes from run to run.

```text
==============================================================================
 Scaler Serialization
==============================================================================

✓ Saved ZScoreScaler to '<tmp>/trex_scaler_demo_f32d7775cbca.bin'
✓ Loaded ZScoreScaler from '<tmp>/trex_scaler_demo_f32d7775cbca.bin'

Verification:
  Mean difference: 0.000000e+00
  Scale difference: 0.000000e+00

✓ Serialization perfect!
```

### Subproblem 4: Memory-mapped matrix

The demo creates a writable $1{,}000 \times 5{,}000$ `mmap_matrix` of 40 MB with `mem_matrix()` and fills it column by
column through the handle with $x_{ij} = 2(i-1) + (j-1)/100$ for the 1-based indices of R. This is the pattern
$2i + j/100$ of the C++ demo in its 0-based indices. `$fit()`, `$transform_inplace()` and
`$inverse_transform_inplace()` of the centered $L_2$ `MLLpNormScaler` then run directly on the handle, which the
binding maps without a copy, while the C++ demo runs the same steps on a $20{,}000 \times 100{,}000$ matrix of 16 GB.

As an R addition, the demo compares column 1 after the round trip with its analytic values $2(i-1)$, which range from
0 to 1,998. The largest deviation of $1.1 \cdot 10^{-13}$ is one unit in the last place of the entries between 512 and
1,024, so the round trip on disk is exact up to rounding. At the end, the demo removes the file.

```text
==============================================================================
 Scaler on Memory-Mapped Matrix
==============================================================================

Creating memory-mapped matrix...
Matrix size: 1000 x 5000
Fitting L2 LpNormScaler on memory-mapped matrix...
✓ LpNormScaler fitted
✓ Data transformed (in place, on disk)
✓ Data inverse transformed
  Round-trip max|diff| on column 1: 1.136868e-13
✓ Memory-mapped file removed
```

The suite closes with its final banner.

```text
==============================================================================
 All Demos Completed Successfully
==============================================================================
```

---

## Technical notes

- The constructors are `MLZScoreScaler$new(center = TRUE, scale = TRUE)` and
  `MLLpNormScaler$new(norm_type = 2, center = TRUE, scale = TRUE)`, shown here with their defaults.
- `$fit()` leaves `X` unchanged. The R classes offer only in-place transforms (`$transform_inplace()`,
  `$inverse_transform_inplace()` and `$fit_transform_inplace()`), which write into a double-storage R matrix without a
  copy and return it. A plain alias (`X2 <- X`) shares that buffer, so the demo takes a deep copy with `X + 0` before
  each in-place transform, and the original data survive for the comparison. A matrix of another storage mode is
  converted to a double copy first, and that copy is modified and returned.
- An `mmap_matrix` handle binds zero-copy through the same `Eigen::Map` interface as in the C++ demo. All scaler
  methods, `$fit()` included, need the handle open in `"readwrite"` mode because the scaler core binds a mutable map.
- The serialization and memory-mapped files come from `tempfile()` and are removed through `on.exit()`. The C++ demo
  writes `scaler_test.bin` and `mmap_matrix.bin` unless the compile-time macros `TREX_DEMO_SCALER_SER_PATH` and
  `TREX_DEMO_MMAP_PATH` override them.
- The R version fills the mapped matrix column by column, where the C++ demo runs a parallel loop with Open
  Multi-Processing (OpenMP). `trex_set_num_threads(min(6L, trex_get_max_threads()))` from TRexSelector pins six
  OpenMP threads for the column loops of the scalers, or fewer on a smaller machine, as the C++ and Python demos do.
  The console line `Running with 6 threads` reports the count.
- The labels print R's logicals (`center=TRUE`), where the C++ output prints `center=true` and the Python output
  `center=True`.
