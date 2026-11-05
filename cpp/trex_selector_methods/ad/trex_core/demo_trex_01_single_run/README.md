# Demo 01: Single-Run T-Rex Selector

## Purpose

The demo demonstrates the basic Terminating Random Experiments (T-Rex) selector as single run example, illustrating
correct application programming interface (API) usage and the resulting console/file output. Suppose we have $n$
observations and $p$ predictors, with a true support $\mathcal{S}^*$ of size $s$ and fixed coefficients $\beta_j = 1$
for all active variables. We present two contrasting scenarios:

1. **High-dimensional** ($p > n$): Moderate sample, many predictors
2. **Low-dimensional** ($n > p$): Large sample, moderate predictors

This demo introduces a small single run example for the classical T-Rex selector.
It verifies correct API usage and produces as result a human-readable console/file output for inspection.
The `main()` runs the high-dimensional scenario first, then the low-dimensional one.
Both use fixed $+1$ coefficients (`rnd_coef = false`).

---

## Data Generation Parameters

Both scenarios consider the linear model

```math
\mathbf{y} = \mathbf{X}\boldsymbol{\beta} + \boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim N(0, \sigma^2 I_n)
```

and share the same true support and coefficients. Only the dimensions differ.

### Scenario A: High-Dimensional ($p > n$)

- $n = 150$ observations
- $p = 300$ features
- True support: $\mathcal{S}^* = \lbrace27, 149, 43, 128, 42, 4\rbrace$ ($s = 6$ true actives)
- True coefficients: fixed $\beta_j = 1$ for all active variables
- Linear signal-to-noise ratio (SNR): $1.0$

### Scenario B: Low-Dimensional ($n > p$)

- $n = 5{,}000$ observations
- $p = 1{,}000$ features
- True support: Same as scenario A
- True coefficients: Same as scenario A
- SNR (linear): $1.0$

Both scenarios share the deterministic data seed `58` and the selector seed `42`, so the example is reproducible.

**Coefficient note**: the `rnd_coef` code path does **not** draw from a uniform distribution — instead it selects
a fixed hardcoded vector $\lbrace-0.4, -0.25, -0.8, 1.1, 2.5, -1.2\rbrace$ which mimics randomly drawn coefficients.
Both runs in `main()` use `rnd_coef = false`, i.e. all coefficients equal $1$.

---

## Control Parameters

```cpp
K = 20                           # Random experiments per T-loop iteration
max_dummy_multiplier = 10        # Max L = 10 × p
use_max_T_stop = true            # Cap T ≤ ceil(n/2)
dummy_distribution = Normal      # Dummy predictors drawn from N(0,1)
lloop_strategy = STANDARD        # Standard L-loop strategy: Fresh dummies each L-loop iteration
tloop_stagnation_stop = false    # No stagnation early-exit
tloop_max_stagnant_steps = 5     # (inactive: stagnation stop disabled)
parallel_rnd_experiments = false # Sequential random experiments
solver_type = TLARS              # Base solver
tFDR = 0.1                       # Target FDR control level
```

---

## Output Files

Two output files are generated in `simulation_results/`, one per scenario. The file stem is built as
`d01_trex_basic_n{n}_p{p}.txt`:

### `d01_trex_basic_n150_p300.txt`

**Scenario A output** (high-dimensional case)

The program prints (both to console and file), in order: the selected indices, the false discovery proportion (FDP),
the true positive proportion (TPP), and the internal T-Rex matrices.
Actual contents:

```txt
High-dimensional (p > n)
Generating synthetic data...
Creating maps of data...
Creating T-Rex Selector instance...
Executing T-Rex Selector...
Selected indices: 4 27 42 43
False Discovery Proportion (FDP): 0.0000
True Positive Proportion (TPP):   0.6667

Adjusted Relative Occurrences (Phi_prime): 6 of 300 entries nonzero
...

Phi Matrix (Phi): 3 x 300, 18 of 900 entries nonzero
...

Estimated FDP Matrix (FDP_hat):
...

R Matrix (R):
...

Voting Grid:
 0.5 0.55  0.6 0.65  0.7 0.75  0.8 0.85  0.9 0.95    1
```

### `d01_trex_basic_n5000_p1000.txt`

**Scenario B output** (low-dimensional case), same block structure.

There is no timing block and no "number of selected variables / false positives" summary.
The program reports the selected index list plus FDP/TPP and the four diagnostic matrices (Phi', Phi, FDP_hat, R)
and the voting grid.

---

## Running the Demo

```bash
./build/debug/bin/trex_selector_methods/ad/trex_core/demo_trex_01_single_run/demo_trex_01_single_run
```

Output appears on the console and is simultaneously written to the files above.

---

## Expected Results

- **FDP** is well controlled. The committed high-dimensional run reports $\text{FDP} = 0.0$ (no false selections).
- **TPP** at SNR $= 1.0$ is partial in the harder high-dimensional case (the committed run recovers 4 of the 6 actives,
   $\text{TPP} = 0.6667$), while the easier low-dimensional case recovers all six ($\text{TPP} = 1$).
- The **voting grid** spans $\lbrace0.5, 0.55, \ldots, 1\rbrace$ and the **R** / **FDP_hat** matrices show how the
   relative-occurrence threshold and dummy multiplier $L$ evolve across the T-loop.

---

## Interpretation

- **Console output** shows real-time progress and the final selection summary.
- **File output** provides a permanent record for the single run.
- This demo is **not** a Monte Carlo study. Results come from a single run with fixed seeds and should not be
   interpreted as population averages.
- Compare the false discovery rate (FDR) and the true positive rate (TPR) with Demos 02–04 for empirical performance
   over many repetitions.
