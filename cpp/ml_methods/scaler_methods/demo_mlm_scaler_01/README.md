# Demo: `ZScoreScaler` and `LpNormScaler`

## Purpose

This demo shows two column-scaling utilities used for data normalization:

- `ZScoreScaler`, which centers columns and divides them by their standard deviation (SD),
- `LpNormScaler`, which centers columns and rescales them to unit $L_2$- or $L_1$-norm.

Both follow the `center` and `scale` switches of R's `scale()`. Beyond the in-memory comparison of these switches,
the demo covers inverse-transform accuracy, serialization of fitted scaler parameters and scaling on a memory-mapped
matrix of 16 GB. The same workflow runs in [Python](../../../../Python/ml_methods/scaler_methods/demo_mlm_scaler_01/)
and [R](../../../../R/ml_methods/scaler_methods/demo_mlm_scaler_01/).

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

`ZScoreScaler` divides by the root mean square around the center,

```math
s_j = \sqrt{\frac{1}{n-1} \sum_{i=1}^{n} (x_{ij} - c_j)^2},
```

which is the SD when centering is on, and by $s_j = 1$ when `scale` is off. `LpNormScaler` divides by the norm
$s_j = \lVert \boldsymbol x_j - c_j \mathbf 1 \rVert_q$ with $q \in \lbrace 1, 2 \rbrace$. The inverse transform
multiplies by $s_j$ and adds $c_j$ back.

---

## Running the demo

From the `cpp/` folder, after building the Release preset:

```bash
./build/release/bin/ml_methods/scaler_methods/demo_mlm_scaler_01/demo_mlm_scaler_01
```

The demo prints to the console only. The output below is a run against library 44cf1ab on an Apple Silicon machine.
The data come from `std::mt19937` with the seeds 123, 456 and 999 and from `std::normal_distribution`, which is
implementation-defined, so another standard library (libstdc++ on Linux, for example) may draw different data and
print different numbers. Subproblem 4 writes a 16 GB file, so the disk needs that much free space. The build places
the serialization file and the mapped file next to this README (the compile-time macros `TREX_DEMO_SCALER_SER_PATH`
and `TREX_DEMO_MMAP_PATH` in `cpp/ml_methods/CMakeLists.txt`), and the demo removes both at the end.

---

## Console walk-through

The suite opens with a banner and the Open Multi-Processing (OpenMP) information of the build. It reports twelve
available threads and then sets six, the thread count of all Examples demos.

```text
============================================================
DataTransformer Demo Suite
============================================================
OpenMP version: 202011 (5.1+)
Max threads: 12
Num processors: 12
Custom reductions: Supported
Running with 6 threads
```

### Subproblem 1: Center and scale switches

The demo draws a $1{,}000 \times 500$ matrix from $\mathcal N(5, 3^2)$, so the effect of each switch is visible. It
prints three column statistics averaged over all columns, the absolute mean, the SD and the $L_2$-norm. The
untransformed columns have mean 5.00, SD 3.0029 and norm 184.56, close to
$\sqrt{n(\mu^2 + \sigma^2)} = \sqrt{34{,}000} \approx 184.39$. The seven scaled copies read as follows:

- Full z-scoring gives the mean $3.1 \cdot 10^{-16}$, the SD 1.0000 and the norm $31.61 = \sqrt{999}$, which shows
  the $n - 1$ denominator.
- Without centering, the scaler divides by the root mean square around 0, about $184.56/\sqrt{999} \approx 5.84$,
  as R's `scale()` does. The mean becomes $5/5.84 \approx 0.857$ and the SD $3.0029/5.84 \approx 0.514$, while the
  norm is again $\sqrt{999}$.
- Centering alone leaves the SD at 3.0029 and gives the norm $\sqrt{999} \cdot 3.0029 \approx 94.91$.
- With both switches off, the transform is the identity and reproduces the original row.
- The centered $L_2$ scaler gives unit norms and the SD $1/\sqrt{999} \approx 0.0316$.
- The uncentered $L_2$ scaler divides by the raw norm, so the mean becomes $5/184.56 \approx 0.0271$ and the SD
  $3.0029/184.56 \approx 0.0163$.
- The centered $L_1$ scaler divides by an $L_1$-norm of about $n\sigma\sqrt{2/\pi} \approx 2{,}396$, which leaves
  the SD 0.0013 and the $L_2$-norm 0.04.

```text
============================================================
Scaler Comparison - center / scale switches
============================================================
Data: 1000 x 500 drawn from N(mean=5, sd=3)

  original (untransformed)            avg|mean| = 5.00e+00   avg SD = 3.0029   avg L2 = 184.56

  ZScore(center=true,  scale=true )   avg|mean| = 3.11e-16   avg SD = 1.0000   avg L2 = 31.61
  ZScore(center=false, scale=true )   avg|mean| = 8.57e-01   avg SD = 0.5143   avg L2 = 31.61
  ZScore(center=true,  scale=false)   avg|mean| = 9.37e-16   avg SD = 3.0029   avg L2 = 94.91
  ZScore(center=false, scale=false)   avg|mean| = 5.00e+00   avg SD = 3.0029   avg L2 = 184.56

  LpNorm(L2, center=true )            avg|mean| = 9.85e-18   avg SD = 0.0316   avg L2 = 1.00
  LpNorm(L2, center=false)            avg|mean| = 2.71e-02   avg SD = 0.0163   avg L2 = 1.00
  LpNorm(L1, center=true )            avg|mean| = 3.89e-19   avg SD = 0.0013   avg L2 = 0.04
```

### Subproblem 2: Inverse-transform accuracy

On a $1{,}000 \times 5{,}000$ standard normal matrix, `fit()`, `transform_inplace()` and
`inverse_transform_inplace()` run in sequence for `ZScoreScaler` and the centered $L_2$ `LpNormScaler`. Both round
trips end with a largest error of $8.88 \cdot 10^{-16} = 2^{-50}$, one unit in the last place of the entries between 4
and 8, which the largest of the five million draws reach. The mean error of about $4 \cdot 10^{-18}$ lies far below
one unit in the last place, so most entries come back bit for bit.

```text
============================================================
Scaler Inverse Transform Accuracy
============================================================
Matrix size: 1000 x 5000

ZScoreScaler:
  Max Error: 8.88e-16
  Mean Error: 4.13e-18

LpNormScaler L2:
  Max Error: 8.88e-16
  Mean Error: 4.03e-18
```

### Subproblem 3: Serialization

A `ZScoreScaler` is fit on a $1{,}000 \times 500$ standard normal matrix, saved with `save()` and loaded into a fresh
object with `load()`. The stored means and scales differ by exactly zero, so the binary file holds the doubles without
loss. The console shows the path from `cpp/` on, without the machine-specific prefix.

```text
============================================================
Scaler Serialization
============================================================
✓ Saved ZScoreScaler to '.../cpp/ml_methods/scaler_methods/demo_mlm_scaler_01/trex_mlm_demo_scaler01_ser.bin'
✓ Loaded ZScoreScaler from '.../cpp/ml_methods/scaler_methods/demo_mlm_scaler_01/trex_mlm_demo_scaler01_ser.bin'

Verification:
  Mean difference: 0.00e+00
  Scale difference: 0.00e+00

✓ Serialization perfect!
```

### Subproblem 4: Memory-mapped matrix

The demo creates a writable $20{,}000 \times 100{,}000$ `MemoryMappedMatrix<double>` of $1.6 \cdot 10^{10}$ bytes
(16 GB) and fills it in parallel with $x_{ij} = 2i + j/100$ for 0-based indices, one column per loop iteration. It
then runs `fit()`, `transform_inplace()` and `inverse_transform_inplace()` of the centered $L_2$ `LpNormScaler`
directly on the mapped buffer, without a copy in memory. The output lists the steps only and measures no error,
because Subproblem 2 covers the accuracy. At the end, the demo removes the file.

```text
============================================================
Scaler on Memory-Mapped Matrix
============================================================
Creating memory-mapped matrix...
Matrix size: 20000 x 100000
Fitting L2 LpNormScaler on memory-mapped matrix...
✓ LpNormScaler fitted
✓ Data transformed
✓ Data inverse transformed
✓ Memory-mapped file removed
```

The suite closes with its final banner.

```text
============================================================
All Demos Completed Successfully
============================================================
```

---

## Technical notes

- The constructors are `ZScoreScaler(center, scale)` and `LpNormScaler(norm_type, center, scale)`, with both switches
  on and the $L_2$-norm by default. The demo copies the data before each in-place transform (`Eigen::MatrixXd Xt = X`),
  so the original survives for the comparison.
- `fit()` reads the data through an `Eigen::Ref` to a constant matrix, and `transform_inplace()` and
  `inverse_transform_inplace()` work on an `Eigen::Map`, so the same calls serve an in-memory matrix and the map of a
  `MemoryMappedMatrix`.
- The serialization uses `ZScoreScaler::save()` and `ZScoreScaler::load()` and compares the stored centers and scales
  through the norms of their differences.
- Without the compile-time macros, the demo writes `scaler_test.bin` and `mmap_matrix.bin` into the working
  directory.
- The mapped file is unlinked with `std::remove()` while the map is still open. On macOS and Linux the disk space is
  released when the map closes at the end of the function.
