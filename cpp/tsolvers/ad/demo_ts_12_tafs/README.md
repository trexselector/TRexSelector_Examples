# Demo: T-AFS (Adaptive Forward Stepwise) Solver

## Purpose

This demo runs the terminating adaptive forward stepwise (T-AFS) solver on its own, outside the T-Rex selector. Its
shrinkage parameter $\rho \in (0, 1]$ sets how far each step goes. With $\rho = 1$ T-AFS equals T-OMP step for step, and
small values of $\rho$ bring it close to T-LARS. Parts 1 to 4 use $\rho = 1$, and Part 5 draws four values.

The demo has five parts:

1. early stopping with internal normalization, for $p > n$ and $n > p$,
2. the same data normalized outside the solver,
3. a checkpoint saved to disk, reloaded and continued,
4. data memory-mapped from disk,
5. terminated solution paths for four values of $\rho$, drawn as a figure.

---

## Key idea

A terminating solver does not compute the whole path. It stops once $T$ dummies have entered the active set:

```math
\hat{\boldsymbol{\beta}}^{(T)} = \mathrm{TAFS}(\mathbf{X}, \mathbf{D}, \mathbf{y}; T, \rho).
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
cpp/build/release/bin/tsolvers/ad/demo_ts_12_tafs/demo_ts_12_tafs        # Parts 1-5
cpp/build/release/bin/tsolvers/ad/demo_ts_12_tafs/demo_ts_12_tafs 1 2 3  # Parts 1 to 3, as below
cpp/build/release/bin/tsolvers/ad/demo_ts_12_tafs/demo_ts_12_tafs 5      # Part 5 only
cpp/tsolvers/ad/demo_ts_12_tafs/generate_plots.sh                        # path figure from the Part 5 data
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
variables 4, 27, 149, 398 and 420, and `snr = 1` gives a noise standard deviation of about 2.2. T-AFS centers and scales
the data itself (`.normalize = true, .intercept = true`) with $\rho = 1$ and runs to $T = 10$.

```text
============================================================
Demo 1: Basic T-AFS with Early Stopping
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

[Info] TAFS sequence
  Samples (n): 1000
  Features (p): 55000
  Shrinkage (rho): 1
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
Demo 1: Basic T-AFS with Early Stopping
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

[Info] TAFS sequence
  Samples (n): 5000
  Features (p): 11000
  Shrinkage (rho): 1
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

With $\rho = 1$ T-AFS takes exactly the path of T-OMP in both cases. The five true variables enter first and ten dummies
follow, so the selection is exactly the true support, with a false discovery proportion (FDP) of 0 and a true positive
proportion (TPP) of 1.

The `Dummies` line counts the columns of the combined matrix $[\mathbf{X}, \mathbf{D}]$ from 0, while the `Path` line
numbers the dummies on their own, so column `50753` is dummy `D45753`, because the $p = 5{,}000$ columns of $\mathbf{X}$
come first. The step limit `Max steps` is $8 \min(n - 1, p + L)$, because the intercept takes one degree of freedom,
which gives 7,992 and 39,992.

### Part 2: external normalization

Part 2 uses the same data but centers and scales $\mathbf{X}$ and $\mathbf{D}$ outside the solver with `LpNormScaler`
(unit $L_2$ norm) and centers $\mathbf{y}$. T-AFS then runs without its own normalization and intercept
(`.normalize = false, .intercept = false`), again with $\rho = 1$. The part starts with the case $n > p$.

```text
============================================================
=== Demo 2: T-AFS with External Normalization ===
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
[Info] TAFS sequence
  Samples (n): 5000
  Features (p): 11000
  Shrinkage (rho): 1
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
variables 10, 25 and 40. It scales the data once in front, and three solvers with $\rho = 1$ read the same buffers
without copies. A reference solver runs to $T = 15$. A second solver runs to $T = 7$ and saves its state to
`tafs_checkpoint.bin`, and a third one loads that file and continues to $T = 15$.

```text
============================================================
=== Demo 3: T-AFS Serialization & Warm Start ===
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
✓ Checkpoint saved at 'tafs_checkpoint.bin'
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
on disk, about 44 GB, and runs T-AFS as a writer on memory-mapped views of the files to $T = 10$. It then removes the
files. Its output has the form of Part 1, and it is left out here because of the disk space and the time it takes.

### Part 5: terminated solution paths

Part 5 repeats the quick start of the R tlars vignette. With $n = 150$, $p = 300$, five true coefficients equal to 1 and
$L = p = 300$ dummies, one T-AFS object per $\rho \in \lbrace 0.1, 0.3, 0.5, 1 \rbrace$ runs to $T = 1$ and is then
warm-started to $T = 2$ and $T = 5$. Each call continues the stored path, and a snapshot of the path after each call
goes to `simulation_results/data/`. The setting `snr = 5` gives a noise standard deviation of about 1, as in the
vignette.

```text
=== Demo 5: T-AFS Terminated Solution Paths (rho = 0.1) ===
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
T_stop = 2: steps = 11, predictors =  9, dummies = 2, FDP = 0.444, TPP = 1.000
T_stop = 5: steps = 17, predictors = 12, dummies = 5, FDP = 0.583, TPP = 1.000

Selection:
  Predictors: {0, 4, 1, 2, 3, 106, 72, 183, 124, 282, 5, 22}
  Dummies: {315, 564, 321, 399, 355}
  Path: [+v0, +v4, +v1, +v2, +v3, +v106, +v72, +v183, +D15, +v124, +D264, +v282, +D21, +v5, +D99, +v22, +D55]

Path data written to:
  .../cpp/tsolvers/ad/demo_ts_12_tafs/simulation_results/data/tafs_rho_0.1_terminated_paths_path.csv
  .../cpp/tsolvers/ad/demo_ts_12_tafs/simulation_results/data/tafs_rho_0.1_terminated_paths_actions.csv


============================================================
=== Demo 5: T-AFS Terminated Solution Paths (rho = 0.3) ===
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
T_stop = 2: steps = 10, predictors =  8, dummies = 2, FDP = 0.375, TPP = 1.000
T_stop = 5: steps = 20, predictors = 15, dummies = 5, FDP = 0.667, TPP = 1.000

Selection:
  Predictors: {0, 4, 1, 2, 3, 72, 183, 106, 192, 22, 201, 194, 285, 211, 88}
  Dummies: {564, 383, 399, 339, 537}
  Path: [+v0, +v4, +v1, +v2, +v3, +v72, +v183, +v106, +D264, +D83, +D99, +v192, +v22, +v201, +v194, +v285, +v211, +D39, +v88, +D237]

Path data written to:
  .../cpp/tsolvers/ad/demo_ts_12_tafs/simulation_results/data/tafs_rho_0.3_terminated_paths_path.csv
  .../cpp/tsolvers/ad/demo_ts_12_tafs/simulation_results/data/tafs_rho_0.3_terminated_paths_actions.csv


============================================================
=== Demo 5: T-AFS Terminated Solution Paths (rho = 0.5) ===
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
T_stop = 2: steps = 11, predictors =  9, dummies = 2, FDP = 0.444, TPP = 1.000
T_stop = 5: steps = 16, predictors = 11, dummies = 5, FDP = 0.545, TPP = 1.000

Selection:
  Predictors: {0, 4, 1, 2, 3, 72, 183, 106, 194, 201, 92}
  Dummies: {383, 399, 339, 553, 308}
  Path: [+v0, +v4, +v1, +v2, +v3, +v72, +v183, +v106, +D83, +v194, +D99, +v201, +D39, +D253, +v92, +D8]

Path data written to:
  .../cpp/tsolvers/ad/demo_ts_12_tafs/simulation_results/data/tafs_rho_0.5_terminated_paths_path.csv
  .../cpp/tsolvers/ad/demo_ts_12_tafs/simulation_results/data/tafs_rho_0.5_terminated_paths_actions.csv


============================================================
=== Demo 5: T-AFS Terminated Solution Paths (rho = 1.0) ===
============================================================
Configuration:
  n = 150, p = 300
  num_dummies = 300
  T_stop = 5
  True support size = 5
  True support: 0 1 2 3 4
  True coefficients: 1 1 1 1 1
  SNR = 5

T_stop = 1: steps =  8, predictors =  7, dummies = 1, FDP = 0.286, TPP = 1.000
T_stop = 2: steps = 11, predictors =  9, dummies = 2, FDP = 0.444, TPP = 1.000
T_stop = 5: steps = 20, predictors = 15, dummies = 5, FDP = 0.667, TPP = 1.000

Selection:
  Predictors: {0, 1, 4, 2, 3, 72, 183, 106, 201, 5, 203, 142, 269, 80, 138}
  Dummies: {383, 399, 357, 372, 388}
  Path: [+v0, +v1, +v4, +v2, +v3, +v72, +v183, +D83, +v106, +v201, +D99, +v5, +D57, +v203, +v142, +v269, +v80, +v138, +D72, +D88]

Path data written to:
  .../cpp/tsolvers/ad/demo_ts_12_tafs/simulation_results/data/tafs_rho_1.0_terminated_paths_path.csv
  .../cpp/tsolvers/ad/demo_ts_12_tafs/simulation_results/data/tafs_rho_1.0_terminated_paths_actions.csv
```

Adaptive forward stepwise (AFS) shrinks each forward-stepwise update by the factor $\rho$. It is forward stepwise at
$\rho = 1$ and approaches least angle regression (LAR) and the lasso as $\rho \to 0$. At $\rho = 0.1$ the coefficients
grow gradually over many steps as in T-LARS, the first five variables enter in the T-LARS order 0, 4, 1, 2, 3, and the
five true coefficients end at 0.57 to 0.80, close to the 0.57 to 0.83 of T-LARS. At $\rho = 1$ every entering variable
jumps to its least-squares size, and the path equals the T-OMP path of [demo_ts_06_tomp](../demo_ts_06_tomp/README.md)
to the last digit, with true coefficients of 0.67 to 1.01. The values 0.3 and 0.5 lie in between. The true positive
proportion (TPP) is 1 in all runs, while the FDP at $T = 5$ ranges from 0.545 at $\rho = 0.5$ to 0.667 at $\rho = 0.3$
and $\rho = 1$. The figure shows each path at $T = 5$, where the red lines also mark where the runs to $T = 1$ and
$T = 2$ ended. Original variables are black, dummies red and dashed, a red vertical line marks each included dummy, and
the top axis names the variable that enters at each step.

![T-AFS paths at T = 5 for four values of rho](simulation_results/plots/tafs_terminated_paths_rho.png)

---

## Technical notes

- Part 3 writes the checkpoint `tafs_checkpoint.bin` into the working directory and removes it after the
  comparison.
- Part 4 writes $\mathbf{X}$, $\mathbf{D}$ and $\mathbf{y}$ into three temporary files of about 44 GB in total in the
  working directory, runs the solver on memory-mapped views and removes the files at the end.
- The path data and figures under `simulation_results/` are committed, so the figure is visible without a run.
