# Demo: T-GP Solver

## Purpose

This demo runs the terminating gradient pursuit (T-GP) solver on its own, outside the T-Rex selector. Gradient pursuit
selects like orthogonal matching pursuit, but instead of the exact least-squares refit it takes one step along the
gradient of the residual sum of squares on the active set, with the optimal step length.

The demo has five parts:

1. early stopping with internal normalization, for $p > n$ and $n > p$,
2. the same data normalized outside the solver,
3. a checkpoint saved to disk, reloaded and continued,
4. data memory-mapped from disk,
5. terminated solution paths of one warm-started object, drawn as a figure.

---

## Key idea

A terminating solver does not compute the whole path. It stops once $T$ dummies have entered the active set:

```math
\hat{\boldsymbol{\beta}}^{(T)} = \mathrm{TGP}(\mathbf{X}, \mathbf{D}, \mathbf{y}; T).
```

The demo calls

```cpp
executeStep(T_stop, /*early_stop=*/true)
```

and the path stops as soon as the number of active dummies reaches `T_stop`.

---

## Running the demo

From the repository root, after `cmake --preset release && cmake --build --preset release-build` in `cpp/`:

```bash
cpp/build/release/bin/tsolvers/ad/demo_ts_07_tgp/demo_ts_07_tgp        # Parts 1-5
cpp/build/release/bin/tsolvers/ad/demo_ts_07_tgp/demo_ts_07_tgp 1 2 3  # Parts 1 to 3, as below
cpp/build/release/bin/tsolvers/ad/demo_ts_07_tgp/demo_ts_07_tgp 5      # Part 5 only
cpp/tsolvers/ad/demo_ts_07_tgp/generate_plots.sh                       # path figure from the Part 5 data
```

The optional arguments select the parts to run. Part 4 writes about 44 GB of temporary files, so `1 2 3` and `5` leave
it out. `generate_plots.sh` uses the repository's `.venv` and writes the figure as Portable Network Graphics (PNG)
and Portable Document Format (PDF) files to `simulation_results/plots/`. The output below is a run against
library b95b984.

---

## Console walk-through

### Part 1: early stopping

Part 1 draws $n = 1{,}000$ observations of $p = 5{,}000$ variables and then $n = 5{,}000$ observations of $p = 1{,}000$
variables, each time with $L = 10p$ standard normal dummies (seed 42). Five true coefficients equal to 1 sit on the
variables 4, 27, 149, 398 and 420, and `snr = 1` gives a noise standard deviation of about 2.2. T-GP centers and scales
the data itself (`.normalize = true, .intercept = true`) and runs to $T = 10$.

```text
============================================================
Demo 1: Basic T-GP with Early Stopping
============================================================
High-dimensional (p > n)
Configuration:
  n = 1000, p = 5000
  num_dummies = 50000
  T_stop = 10
  True support size = 5
  True support: 27 149 398 420 4
  True coefficients: 1 1 1 1 1
  SNR = 1

[Info] TGP sequence
  Samples (n): 1000
  Features (p): 55000 (5000 real + 50000 dummies)
  Max steps: 7992

Selection:
  Predictors: {149, 398, 27, 4, 420}
  Dummies: {50753, 5513, 22780, 14004, 19036, 13141, 30587, 9284, 25555, 50308}
  Path: [+v149, +v398, +v27, +v4, +v420, +D45753, +D513, +D17780, +D9004, +D14036, +D8141, +D25587, +D4284, +D20555, +D45308]

Selection Quality:
  True Positives:  5
  False Positives: 0
  False Negatives: 0
  FDP:             0
  TPP:             1
  Precision:       1
  Recall:          1
  F1 Score:        1

============================================================
Demo 1: Basic T-GP with Early Stopping
============================================================
Low-dimensional (n > p)
Configuration:
  n = 5000, p = 1000
  num_dummies = 10000
  T_stop = 10
  True support size = 5
  True support: 27 149 398 420 4
  True coefficients: 1 1 1 1 1
  SNR = 1

[Info] TGP sequence
  Samples (n): 5000
  Features (p): 11000 (1000 real + 10000 dummies)
  Max steps: 39992

Selection:
  Predictors: {149, 27, 420, 398, 4}
  Dummies: {9675, 5174, 9011, 3702, 5655, 1358, 3412, 7113, 9141, 1699}
  Path: [+v149, +v27, +v420, +v398, +v4, +D8675, +D4174, +D8011, +D2702, +D4655, +D358, +D2412, +D6113, +D8141, +D699]

Selection Quality:
  True Positives:  5
  False Positives: 0
  False Negatives: 0
  FDP:             0
  TPP:             1
  Precision:       1
  Recall:          1
  F1 Score:        1
```

T-GP takes exactly the path of T-OMP in both cases. The five true variables enter first and ten dummies follow, so the
selection is exactly the true support, with a false discovery proportion (FDP) of 0 and a true positive proportion (TPP)
of 1.

The `Dummies` line counts the columns of the combined matrix $[\mathbf{X}, \mathbf{D}]$ from 0, while the `Path` line
numbers the dummies on their own, so column `50753` is dummy `D45753`, because the $p = 5{,}000$ columns of $\mathbf{X}$
come first. The step limit `Max steps` is $8 \min(n - 1, p + L)$, because the intercept takes one degree of freedom,
which gives 7,992 and 39,992.

### Part 2: external normalization

Part 2 uses the same data but centers and scales $\mathbf{X}$ and $\mathbf{D}$ outside the solver with `LpNormScaler`
(unit $L_2$ norm) and centers $\mathbf{y}$. T-GP then runs without its own normalization and intercept
(`.normalize = false, .intercept = false`). The part starts with the case $n > p$.

```text
============================================================
=== Demo 2: T-GP with External Normalization ===
============================================================
Low-dimensional (n > p)
Configuration:
  n = 5000, p = 1000
  num_dummies = 10000
  T_stop = 10
  True support size = 5
  True support: 4 27 149 398 420
  True coefficients: 1 1 1 1 1
  SNR = 1

Applying external L2 normalization...
[Info] TGP sequence
  Samples (n): 5000
  Features (p): 11000 (1000 real + 10000 dummies)
  Max steps: 40000

Selection:
  Predictors: {149, 27, 420, 398, 4}
  Dummies: {9675, 5174, 9011, 3702, 5655, 1358, 3412, 7113, 9141, 1699}
  Path: [+v149, +v27, +v420, +v398, +v4, +D8675, +D4174, +D8011, +D2702, +D4655, +D358, +D2412, +D6113, +D8141, +D699]

Selection Quality:
  True Positives:  5
  False Positives: 0
  False Negatives: 0
  FDP:             0
  TPP:             1
  Precision:       1
  Recall:          1
  F1 Score:        1
```

The selection and the path equal those of Part 1, and the case $p > n$ that follows repeats the Part 1 path as well. So
the external normalization and the internal one lead to the same solver run. Only the step limit changes to 40,000 and
8,000, because the data are not marked as precentered and the solver takes the rank as $n$ instead of $n - 1$.

### Part 3: checkpoint and warm start

Part 3 uses a small problem with $n = 100$, $p = 50$, $L = 50$ and the three true coefficients 2.5, −1.8 and 3.2 on the
variables 10, 25 and 40. It scales the data once in front, and three solvers read the same buffers without copies. A
reference solver runs to $T = 15$. A second solver runs to $T = 7$ and saves its state to `tgp_checkpoint.bin`, and a
third one loads that file and continues to $T = 15$.

```text
============================================================
=== Demo 3: T-GP Serialization & Warm Start ===
============================================================
Configuration:
  n = 100, p = 50
  num_dummies = 50
  T_stop = 15
  True support size = 3
  True support: 10 25 40
  True coefficients: 2.5 -1.8 3.2
  SNR = 1

✓ Reference completed with 31 steps

Checkpoint at partial stop: 18 steps
✓ Checkpoint saved at 'tgp_checkpoint.bin'
Loaded from checkpoint: 18 steps

COMPARISON:
Reference steps: 31
Reloaded steps:  31
RSS diff:   0
R2 diff:    0
✓ Paths match
✓ Checkpoint file removed
```

The checkpoint holds the first 18 steps, up to $T = 7$. The reloaded solver continues to the same 31 steps as the
reference, with differences of exactly 0 in the residual sum of squares (RSS) and in $R^2$ and the same actions. T-OMP
needs the same 31 steps on this data. Writers would not work here, because each of them centers and scales the buffers
in place, and a second writer on a held buffer raises an error under the in-place contract of the library.

### Part 4: memory-mapped data

Part 4 generates $n = 1{,}000$ observations of $p = 500{,}000$ variables with $L = 10p = 5{,}000{,}000$ dummies directly
on disk, about 44 GB, and runs T-GP as a writer on memory-mapped views of the files to $T = 10$. It then removes the
files. Its output has the form of Part 1, and it is left out here because of the disk space and the time it takes.

### Part 5: terminated solution paths

Part 5 repeats the quick start of the R tlars vignette. With $n = 150$, $p = 300$, five true coefficients equal to 1 and
$L = p = 300$ dummies, one T-GP object runs to $T = 1$ and is then warm-started to $T = 2$ and $T = 5$. Each call
continues the stored path, and a snapshot of the path after each call goes to `simulation_results/data/`. The setting
`snr = 5` gives a noise standard deviation of about 1, as in the vignette.

```text
=== Demo 5: T-GP Terminated Solution Paths ===
============================================================
Configuration:
  n = 150, p = 300
  num_dummies = 300
  T_stop = 5
  True support size = 5
  True support: 0 1 2 3 4
  True coefficients: 1 1 1 1 1
  SNR = 5

T_stop = 1: steps =  9, predictors =  8, dummies = 1, FDP = 0.375, TPP = 1.000
T_stop = 2: steps = 12, predictors = 10, dummies = 2, FDP = 0.500, TPP = 1.000
T_stop = 5: steps = 16, predictors = 11, dummies = 5, FDP = 0.545, TPP = 1.000

Selection:
  Predictors: {0, 1, 4, 2, 3, 72, 272, 183, 106, 201, 194}
  Dummies: {383, 399, 513, 339, 366}
  Path: [+v0, +v1, +v4, +v2, +v3, +v72, +v272, +v183, +D83, +v106, +v201, +D99, +v194, +D213, +D39, +D66]

Path data written to:
  .../cpp/tsolvers/ad/demo_ts_07_tgp/simulation_results/data/tgp_terminated_paths_path.csv
  .../cpp/tsolvers/ad/demo_ts_07_tgp/simulation_results/data/tgp_terminated_paths_actions.csv
```

Gradient pursuit (GP) selects like orthogonal matching pursuit (OMP) but replaces the exact least-squares refit by a
single gradient step on the active set. That avoids the least-squares solve, but on short terminated paths the scan
over all columns dominates each step, so T-GP is not faster than T-OMP there (see
[demo_ts_perf_01](../demo_ts_perf_01_fdr_tpr_vs_T/README.md)). The coefficients stay close to the OMP values, with the
five true ones at 0.78 to 1.03 at $T = 5$. The entry order equals T-OMP for the first six variables. Then variable 272
enters before 183, and the later order differs. The FDP is 0.375 at $T = 1$ and 0.545 at $T = 5$, and the TPP stays
at 1. T-ACGP in [demo_ts_08_tacgp](../demo_ts_08_tacgp/README.md) takes the same entry order. The figure shows the three
snapshots. Original variables are black, dummies red and dashed, a red vertical line marks each included dummy, and the
top axis names the variable that enters at each step.

![T-GP terminated solution paths for T = 1, 2 and 5](simulation_results/plots/tgp_terminated_paths.png)

---

## Technical notes

- Part 3 writes the checkpoint `tgp_checkpoint.bin` into the working directory and removes it after the
  comparison.
- Part 4 writes $\mathbf{X}$, $\mathbf{D}$ and $\mathbf{y}$ into three temporary files of about 44 GB in total in the
  working directory, runs the solver on memory-mapped views and removes the files at the end.
- The path data and figures under `simulation_results/` are committed, so the figure is visible without a run.
