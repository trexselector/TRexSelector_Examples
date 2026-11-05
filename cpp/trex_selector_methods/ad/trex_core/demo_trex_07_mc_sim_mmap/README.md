# Demo 07: Monte Carlo Simulation with Memory-Mapped Data

## Purpose

Validate **Terminating Random Experiments (T-Rex) selector performance under memory-mapped (mmap) input/output (I/O)**
over many Monte Carlo (MC) trials, mirroring the single-run patterns of Demo 06. Two scenarios are run: (C) in-memory
$\mathbf{X}$ with mmap dummy matrices $\mathbf{D}$ + solver serialization, and (D) a fully memory-mapped pipeline where
each MC iteration creates and destroys its own mmap-backed $\mathbf{X}/\mathbf{y}$ files.

---

## Data Generation Parameters

Both scenarios use the same high-dimensional configuration:

- **Sample size**: $n = 300$
- **Number of features**: $p = 1{,}000$
- **True support**: a set of **10 unique random indices** drawn once from `std::mt19937 rng(24)` (uniform over
  $\lbrace0, \ldots, p-1\rbrace$), fixed across all MC trials
- **True coefficients**: fixed $\beta_j = 1$ (`rnd_coef = false`)
- **Signal-to-noise ratio (SNR) range**: $\lbrace0.1, 0.5, 1.0, 2.0, 5.0\rbrace$
- **Monte Carlo repetitions**: `num_MC = 200` for both scenarios — the two runs are directly comparable
- **Base solver**: TLARS only
- **Data-generating process (DGP)**: $\mathbf{y} = \mathbf{X}\boldsymbol{\beta} + \boldsymbol{\epsilon}$,
  $\boldsymbol{\epsilon} \sim N(0, \sigma^2 I_n)$
- **Target false discovery rate (tFDR)**: $0.1$

---

## Scenarios

### Demo C: MC — In-Memory $\mathbf{X}$ + `use_memory_mapping = true`

- $\mathbf{X}$ generated in random-access memory (RAM) per trial, while dummy matrices $\mathbf{D}$ are memory-mapped
  and solver least-angle regression (LARS) path state is serialized to disk.
- The purpose is to verify that the D-mmap + solver-serialization pipeline yields a stable, reproducible false
  discovery rate (FDR) and true positive rate (TPR) over many runs.

### Demo D: MC — Fully Memory-Mapped Pipeline ($\mathbf{X}$ + $\mathbf{D}$ + solver serialization)

- Each MC iteration writes its own mmap-backed $\mathbf{X}/\mathbf{y}$ files (per-thread paths derived from the
  filestem `X_mmap_mc` and `omp_get_thread_num()`), which are removed by a Resource Acquisition Is Initialization
  (RAII) `MmapFileGuard` at the end of the iteration scope — exception-safe.

---

## Control Parameters

```text
K = 20                           # Random experiments per T-loop iteration
max_dummy_multiplier = 10        # Max L = 10p
use_max_T_stop = true            # Cap T ≤ ceil(n/2)
dummy_distribution = Normal      # Dummy predictors drawn from N(0,1)
lloop_strategy = HCONCAT         # Horizontally concatenated dummy columns
tloop_stagnation_stop = true     # Early exit when R_mat stagnates
tloop_max_stagnant_steps = 7     # Stagnation window (7) → "sw7" in the output stem
opt_threshold = 0.75             # Optimization grid point
parallel_rnd_experiments = false # Sequential random experiments
use_memory_mapping = true        # D mmap + solver serialization
solver_type = TLARS              # Base solver
tFDR = 0.1                       # Target FDR control level
```

The MC loop is parallelized with Open Multi-Processing (OpenMP) (`TREX_NUM_WORKERS` threads, default 8).

---

## Output Files

Written to `simulation_results/data/`. Each scenario writes one `.txt` and one `.csv`, with stems encoding `n`, `p`,
and the stagnation window (`sw7`):

- **Demo C**: `d03_trex_mmap_demo_c_n300_p1000_sw7.{txt,csv}`
- **Demo D**: `d03_trex_mmap_demo_d_n300_p1000_sw7.{txt,csv}`

Figures are written separately to `simulation_results/plots/` by `./generate_plots.sh` (see *Results Visualization*),
while the demo binary itself writes only the `.txt` / `.csv` pair per scenario.

The `.txt` file (written by the shared `save_and_print_mc_results`) is an aligned table with FDR, TPR, Avg L, and
Avg T rows for the single TLARS solver across the 5 SNR columns:

```text
======================================================================
=== T-Rex Results (averaged over 200 Monte Carlo runs) ===
======================================================================

  SNR                          0.10      0.50      1.00      2.00      5.00
---------------------------------------------------------------------------

TLARS
  FDR                        0.0250    0.0343    0.0412    0.0314    0.0307
  TPR                        0.0015    0.2970    0.7320    0.9815    1.0000
  Avg L                      2.7900    6.6300    3.5350    1.8850    1.1350
  Avg T                      3.7600    5.6450    5.9800    5.5500    4.5800
```

Both scenarios report "averaged over 200 Monte Carlo runs". The `.csv` uses the tidy header
**`solver,metric,snr,value`** with `FDR`, `TPR`, `AvgL`, `AvgT` rows.

---

## Results Visualization

The suite-level plotting module [../trex_plt_utils.py](../trex_plt_utils.py) renders the tidy comma-separated values
(CSV) files into `simulation_results/plots/`. Only the `FDR` and `TPR` rows are plotted, while the `AvgL` / `AvgT` rows
are ignored by the plotter (they remain in the CSV and the `.txt` table).

### Storage scenarios compared (headline figure)

Both scenarios run the identical sweep, so they belong on one pair of axes, because this demo's question is whether
the storage medium is visible in the *results*, and two curves that lie on top of each other answer it directly. The
scenario takes the place of the "solver" row that the other demos sweep:

![FDR and TPR versus SNR for Demo C and Demo D, the two curves coinciding](simulation_results/plots/d03_trex_mmap_demo_c_vs_d.png)

*Left:* FDR stays far below the target `tFDR = 0.1` (black dotted line) across the whole SNR grid under **both**
storage pipelines, peaking at only $0.041$ (C) / $0.044$ (D) at $\mathrm{SNR}=1$. *Right:* the power curves are
indistinguishable — TPR climbs from at most $0.003$ at $\mathrm{SNR}=0.1$ to full recovery ($1.000$ in both) by
$\mathrm{SNR}=5$.

The two pipelines agree to within **0.005 (FDR)** and **0.020 (TPR)** at every grid point — the scale of Monte
Carlo noise at $\mathrm{MC}=200$, not of a systematic effect. This is the result the demo exists to show. Memory
mapping $\mathbf{X}$ and $\mathbf{D}$ is transparent to the selector, so it buys its memory savings without moving
the statistics. Note the curves are *not* expected to coincide exactly — the two scenarios drive different data
paths, so their MC trials are not bit-identical draws.

### Per-scenario overviews

The same data one scenario at a time, for reading a single pipeline's numbers off a figure. A single TLARS row per
scenario means there is nothing for the grouped 2×2 view to de-clutter, so it is skipped:

![Demo C: FDR and TPR versus SNR](simulation_results/plots/d03_trex_mmap_demo_c_n300_p1000_sw7_fdr_tpr_vs_snr.png)

![Demo D: FDR and TPR versus SNR](simulation_results/plots/d03_trex_mmap_demo_d_n300_p1000_sw7_fdr_tpr_vs_snr.png)

### Interactive (Plotly HTML, on request)

With `--plotly`, each static figure gets a self-contained `.html` twin — hover for exact values, click a scenario in
the legend to isolate its curve in both panels. Open it directly in a browser (it is not rendered inline on GitHub).
The `.html` figures are off by default and never committed, since each embeds a full copy
of plotly.js (about 5 MB). Run `./generate_plots.sh --plotly` to produce them locally.

```bash
open simulation_results/plots/d03_trex_mmap_demo_c_vs_d.html
```

Vector `.pdf` copies of every static figure sit alongside the `.png` files.

### Regenerating the figures

```bash
# From this demo folder:
./generate_plots.sh                 # overlay + per-scenario (png+pdf)
./generate_plots.sh --plotly        # plus the interactive html (not committed)
./generate_plots.sh --formats png   # extra args pass through to trex_plt_utils.py
```

The overlay is assembled by merging the two per-scenario CSVs into one tidy frame whose `solver` column carries the
scenario name — that column is the plotting module's generic "what this demo varies" dimension. The merge is a
temporary file, so `simulation_results/data/` keeps only what the demo binary writes.

---

## Running the Demo

```bash
./build/debug/bin/trex_selector_methods/ad/trex_core/demo_trex_07_mc_sim_mmap/demo_trex_07_mc_sim_mmap
```

`main()` runs Demo C then Demo D, each at `num_MC = 200`.

---

## Key Questions Addressed

1. **Does FDR control hold with mmap storage?**
   - Expected: FDR $\leq$ tFDR — the storage medium is transparent to the algorithm.

2. **Does the fully-mmapped pipeline (Demo D) manage its per-iteration files safely?**
   - Expected: yes, because the RAII `MmapFileGuard` cleanup runs even if `select()` throws.

3. **Do the two storage pipelines agree?**
   - Expected: yes, up to MC noise. Both scenarios run the same sweep at `num_MC = 200`, so any systematic
     gap would point at the mmap pipeline rather than at sampling variability.
   - Observed: max $|\text{C} - \text{D}|$ of $0.005$ (FDR) and $0.020$ (TPR) — see *Results Visualization*.

---

## Interpretation Guide

**What to look for:**

- **FDR stability**: FDR should stay $\leq$ tFDR across SNR.
- **TPR progression**: increases with SNR toward full recovery.
- **Avg L / Avg T**: dummy-multiplier and stopping-time behaviour under the mmap pipeline.
- **Reliability**: no leaked mmap files, no crashes.

**Comparison with Demo 06 (single-run mmap):**

- Demo 06 validates basic mmap correctness (one run per scenario).
- Demo 07 validates robustness and FDR guarantees over many MC trials.
