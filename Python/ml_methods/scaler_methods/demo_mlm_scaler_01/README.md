# Demo: `ZScoreScaler` and `LpNormScaler`

## Purpose

This demo shows two column-scaling utilities used for data normalization:

- `ZScoreScaler`, which centers columns and divides them by their standard deviation (SD),
- `LpNormScaler`, which centers columns and rescales them to unit $L_2$- or $L_1$-norm.

Both follow the `center` and `scale` switches of R's `scale()`. Beyond the in-memory comparison of these switches,
the demo covers inverse-transform accuracy, serialization of fitted scaler parameters and scaling on a memory-mapped
matrix. The same workflow runs in [C++](../../../../cpp/ml_methods/scaler_methods/demo_mlm_scaler_01/) and
[R](../../../../R/ml_methods/scaler_methods/demo_mlm_scaler_01/). The Python version replaces the Eigen matrices
with Fortran-ordered `float64` NumPy arrays and runs the memory-mapped subproblem at a tenth of the C++ size in each
dimension.

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

From the repository root, with TRexSelector 1.0.0 installed in the repository virtual environment `.venv`:

```bash
.venv/bin/python Python/ml_methods/scaler_methods/demo_mlm_scaler_01/demo_mlm_scaler_01.py
```

The demo prints to the console only and finishes in well under a second. The output below is a run against library
b95b984. The data are drawn with NumPy's `default_rng`, a permuted congruential generator (PCG64), seeded with the
values of the C++ demo (123, 456 and 999). The C++ demo draws with `std::mt19937`, so the two languages print
different numbers that agree qualitatively, not bitwise. The serialization and memory-mapped files go to the system
temporary directory and are removed at the end.

---

## Console walk-through

The suite opens with a banner and sets the thread count of the library backend to six, as the C++ demo does.

```text
==============================================================================
 DataTransformer Demo Suite
==============================================================================

Running with 6 threads
```

### Subproblem 1: Center and scale switches

The demo draws a $1{,}000 \times 500$ matrix from $\mathcal N(5, 3^2)$, so the effect of each switch is visible. It
prints three column statistics averaged over all columns, the absolute mean, the SD and the $L_2$-norm. The
untransformed columns have mean 5.00, SD 2.9981 and norm 184.37, close to
$\sqrt{n(\mu^2 + \sigma^2)} = \sqrt{34{,}000} \approx 184.39$. The seven scaled copies read as follows:

- Full z-scoring gives the mean $3.4 \cdot 10^{-16}$, the SD 1.0000 and the norm $31.61 = \sqrt{999}$, which shows
  the $n - 1$ denominator.
- Without centering, the scaler divides by the root mean square around 0, about $\sqrt{34{,}000/999} \approx 5.83$,
  as R's `scale()` does. The mean becomes $5/5.83 \approx 0.857$ and the SD $3/5.83 \approx 0.514$, while the norm
  is again $\sqrt{999}$.
- Centering alone leaves the SD at 2.9981 and gives the norm $\sqrt{999} \cdot 2.9981 \approx 94.76$.
- With both switches off, the transform is the identity and reproduces the original row.
- The centered $L_2$ scaler gives unit norms and the SD $1/\sqrt{999} \approx 0.0316$.
- The uncentered $L_2$ scaler divides by the raw norm, so the mean becomes $5/184.37 \approx 0.0271$ and the SD
  $3/184.37 \approx 0.0163$.
- The centered $L_1$ scaler divides by an $L_1$-norm of about $n\sigma\sqrt{2/\pi} \approx 2{,}394$, which leaves
  the SD 0.0013 and the $L_2$-norm 0.04.

```text
==============================================================================
 Scaler Comparison - center / scale switches
==============================================================================

Data: 1000 x 500 drawn from N(mean=5, sd=3)

  original (untransformed)            avg|mean| = 5.00e+00   avg SD = 2.9981   avg L2 = 184.37

  ZScore(center=True,  scale=True )   avg|mean| = 3.41e-16   avg SD = 1.0000   avg L2 = 31.61
  ZScore(center=False, scale=True )   avg|mean| = 8.57e-01   avg SD = 0.5140   avg L2 = 31.61
  ZScore(center=True,  scale=False)   avg|mean| = 1.02e-15   avg SD = 2.9981   avg L2 = 94.76
  ZScore(center=False, scale=False)   avg|mean| = 5.00e+00   avg SD = 2.9981   avg L2 = 184.37

  LpNorm(L2, center=True )            avg|mean| = 1.08e-17   avg SD = 0.0316   avg L2 = 1.00
  LpNorm(L2, center=False)            avg|mean| = 2.71e-02   avg SD = 0.0163   avg L2 = 1.00
  LpNorm(L1, center=True )            avg|mean| = 4.27e-19   avg SD = 0.0013   avg L2 = 0.04
```

### Subproblem 2: Inverse-transform accuracy

On a $1{,}000 \times 5{,}000$ standard normal matrix, `fit()`, `transform_inplace()` and
`inverse_transform_inplace()` run in sequence for `ZScoreScaler` and the centered $L_2$ `LpNormScaler`. Both round
trips end with a largest error of $8.9 \cdot 10^{-16}$, one unit in the last place of the entries between 4 and 8,
which the largest of the five million draws reach. The mean error of about $4 \cdot 10^{-18}$ lies far below one unit
in the last place, so most entries come back bit for bit.

```text
==============================================================================
 Scaler Inverse Transform Accuracy
==============================================================================

Matrix size: 1000 x 5000

ZScoreScaler:
  Max Error: 8.881784e-16
  Mean Error: 4.167951e-18

LpNormScaler L2:
  Max Error: 8.881784e-16
  Mean Error: 4.070167e-18
```

### Subproblem 3: Serialization

A `ZScoreScaler` is fit on a $1{,}000 \times 500$ standard normal matrix, saved with `save()` to a temporary file and
loaded into a fresh object with `load()`. The stored means and scales differ by exactly zero, so the binary file holds
the doubles without loss. The console shows the file as `<tmp>/` plus the name that `tempfile.mkstemp()` draws, so
the name changes from run to run.

```text
==============================================================================
 Scaler Serialization
==============================================================================

✓ Saved ZScoreScaler to '<tmp>/trex_scaler_demo_c_7ni3ez.bin'
✓ Loaded ZScoreScaler from '<tmp>/trex_scaler_demo_c_7ni3ez.bin'

Verification:
  Mean difference: 0.000000e+00
  Scale difference: 0.000000e+00

✓ Serialization perfect!
```

### Subproblem 4: Memory-mapped matrix

The demo creates a writable $2{,}000 \times 10{,}000$ `MemoryMappedMatrix` of 160 MB and fills its zero-copy
`to_numpy()` view with $x_{ij} = 2i + j/100$ for 0-based indices. It then runs `fit()`, `transform_inplace()` and
`inverse_transform_inplace()` of the centered $L_2$ `LpNormScaler` directly on that view, while the C++ demo runs the
same steps on a $20{,}000 \times 100{,}000$ matrix of 16 GB. The output lists the steps only and measures no error,
because Subproblem 2 covers the accuracy. At the end, the demo releases the view and the map and removes the file.

```text
==============================================================================
 Scaler on Memory-Mapped Matrix
==============================================================================

Creating memory-mapped matrix...
Matrix size: 2000 x 10000
Fitting L2 LpNormScaler on memory-mapped matrix...
✓ LpNormScaler fitted
✓ Data transformed
✓ Data inverse transformed
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

- The constructors are `ZScoreScaler(center=True, scale=True)` and
  `LpNormScaler(norm_type=NormType.L2, center=True, scale=True)`, shown here with their defaults.
- `fit()` and the copying `transform()`, `fit_transform()` and `inverse_transform()` accept a `float64` array of any
  memory order and leave it unchanged. The `*_inplace()` methods work on the caller's buffer without a copy and
  therefore need a writeable, Fortran-ordered `float64` array. The demo copies with `X.copy(order="F")` before each
  in-place transform, so the original data survive for the comparison.
- The serialization and memory-mapped files come from `tempfile.mkstemp()` and are removed in `try`/`finally`
  blocks. The C++ demo writes `scaler_test.bin` and `mmap_matrix.bin` unless the compile-time macros
  `TREX_DEMO_SCALER_SER_PATH` and `TREX_DEMO_MMAP_PATH` override them.
- The Python demo fills the mapped matrix with a vectorized NumPy broadcast, where the C++ demo runs a parallel loop
  with Open Multi-Processing (OpenMP). `set_num_threads(min(6, get_max_threads()))` from `trexselector.utils`
  mirrors the six OpenMP threads of the C++ demo.
- The labels print Python booleans (`center=True`), where the C++ output prints `center=true`.
