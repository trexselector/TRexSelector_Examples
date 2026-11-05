# Demo 06: Memory-Mapped Data — Single Run

## Purpose

Demonstrate **memory-mapped (mmap) input/output (I/O) workflows** for the Terminating Random Experiments (T-Rex)
selector via two single-run usage patterns. Enabling `use_memory_mapping = true` couples two behaviours. The internal
dummy matrices $\mathbf{D}$ are memory-mapped, and solver warm-start (least-angle regression (LARS) path) state is
serialized to disk between T-loop iterations. Demo B additionally memory-maps the design matrix $\mathbf{X}$ itself.

---

## Data Generation Parameters

Each demo function runs twice — once low-dimensional and once high-dimensional (`make_mmap_demo_config`):

- **Low-dimensional** ($n > p$): $n = 5{,}000$, $p = 1{,}000$
- **High-dimensional** ($p > n$): $n = 1{,}000$, $p = 5{,}000$
- **True support**: fixed index set $\mathcal{S}^* = \lbrace27, 149, 398, 420, 4\rbrace$ ($s = 5$)
- **True coefficients**: fixed $\beta_j = 1$ (`rnd_coef = false`, while the `rnd_coef` path uses a fixed hardcoded
  vector $\lbrace-0.4, -0.25, -0.8, 1.1, 2.5\rbrace$, not a random draw)
- **Signal-to-noise ratio (SNR)**: $1.0$
- **Target false discovery rate (tFDR)**: $0.1$
- Data seed: `58`

**Data size (largest config, $n = 1{,}000$, $p = 5{,}000$)**: the design matrix $\mathbf{X}$ is
$1{,}000 \times 5{,}000 \times 8$ bytes $\approx 40$ MB — comfortably in random-access memory (RAM). The motivation for
mmap here is the **augmented dummy space**, because with up to $L = 10p$ dummy columns, the internal dummy matrices
$\mathbf{D}$ and serialized solver state are the dominant memory consumers (on the order of a few hundred MB), which is
what `use_memory_mapping = true` offloads to disk. (This is an ~MB/hundreds-of-MB workload, not the hundreds-of-GB
range.)

---

## Scenarios

### Demo A: In-Memory $\mathbf{X}$ + `use_memory_mapping = true`

- Design matrix $\mathbf{X}$ is generated and held in RAM (`SyntheticData`).
- Internal dummy matrices $\mathbf{D}$ are memory-mapped, and solver LARS-path checkpoints are serialized to disk
  between T-loop iterations.

**Use case**: $\mathbf{X}$ fits in RAM but the dummy/solver working set is large.

### Demo B: Fully Memory-Mapped Pipeline ($\mathbf{X}$ + $\mathbf{D}$ + solver serialization)

- $\mathbf{X}$ itself is backed by a Boost memory-mapped file (`SyntheticDataMapped`) written to `X_mmap.dat` /
  `y_mmap.dat` in the **current working directory**.
- These backing files are removed automatically on scope exit by a Resource Acquisition Is Initialization (RAII)
  `MmapFileGuard` (declared before the data object so mmap handles close before the files are deleted —
  exception-safe).
- Combined with `use_memory_mapping = true`, this exercises the full pipeline: X mmap + D mmap + solver serialization.

**Use case**: $\mathbf{X}$ too large for RAM and maximum memory efficiency.

---

## Control Parameters

```text
K = 20                           # Random experiments per T-loop iteration
max_dummy_multiplier = 10        # Max L = 10p
use_max_T_stop = true            # Cap T ≤ ceil(n/2)
dummy_distribution = Normal      # Dummy predictors drawn from N(0,1)
lloop_strategy = HCONCAT         # Horizontally concatenated dummy columns
tloop_stagnation_stop = true     # Early exit when R_mat stagnates
use_memory_mapping = true        # D mmap + solver serialization (both demos)
solver_type = TLARS              # Base solver
tFDR = 0.1                       # Target FDR control level
```

---

## Output Files

Written to `simulation_results/data/`. Each demo writes one `.txt` (dual console + file) and one `.csv` per
configuration. Stems are `d02_trex_mmap_demo_a_n{n}_p{p}` (Demo A) and `d02_trex_mmap_demo_b_n{n}_p{p}` (Demo B):

- `d02_trex_mmap_demo_a_n5000_p1000.{txt,csv}`
- `d02_trex_mmap_demo_a_n1000_p5000.{txt,csv}`
- `d02_trex_mmap_demo_b_n5000_p1000.{txt,csv}`
- `d02_trex_mmap_demo_b_n1000_p5000.{txt,csv}`

The `.txt` file records the run (selected indices, false discovery proportion (FDP), true positive proportion (TPP),
and the Phi', Phi, FDP_hat, R matrices and voting grid — via the shared `print_results`). There is **no** timing block.
The `.csv` is a per-variable selection table written by `save_selection_csv`, with header:

```text
variable_index,phi_prime,selected,true_positive
```

The Demo B mmap backing files `X_mmap.dat` / `y_mmap.dat` are created in the working directory and deleted on
scope exit. They are not written under `/tmp/trex_mmaps/`.

---

## Running the Demo

```bash
./build/debug/bin/trex_selector_methods/ad/trex_core/demo_trex_06_mmap/demo_trex_06_mmap
```

`main()` runs Demo A (low-dim then high-dim) followed by Demo B (low-dim then high-dim).

---

## Expected Results

- **Selection quality**: the mmap pipeline produces the same kind of FDP/TPP result as an in-memory run — storage
  medium does not change the algorithm's selection.
- **Memory footprint**: with `use_memory_mapping = true`, the dummy matrices and solver state live on disk, keeping
  RAM usage low, while Demo B additionally keeps $\mathbf{X}$ off the heap.
- Disk speed, solid-state drive (SSD) vs. hard disk drive (HDD), drives the wall-clock cost of the page-fault-driven
  access.

---

## Interpretation Guide

**What to look for:**

1. **Correctness**: Demo A and Demo B select consistently for the same configuration.
2. **Resource behaviour**: RAM stays low while dummy/solver state is offloaded to disk.
3. **Cleanup**: Demo B's backing files are removed automatically by the RAII guard.

**Practical guidance:**

- Enable mmap when the dummy/solver working set (or $\mathbf{X}$ itself) does not fit comfortably in RAM.
- Prefer an SSD for the backing files to reduce page-fault latency.
