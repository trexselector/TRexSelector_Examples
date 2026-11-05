# Demo 06: Memory-Mapped Data — Single Run (Python)

## Purpose

Demonstrate **memory-mapped (mmap) input/output (I/O) workflows** for the
Terminating Random Experiments (T-Rex) selector via two single-run usage
patterns. Setting `use_memory_mapping = True` couples two behaviours. The
internal dummy matrices $\mathbf{D}$ are memory-mapped, and the solver's
least-angle regression (LARS) path warm-start state is serialized to disk between
T-loop iterations. Part B additionally memory-maps the design matrix
$\mathbf{X}$ itself. Python port of
[cpp/trex_selector_methods/ad/trex_core/demo_trex_06_mmap](../../../../../cpp/trex_selector_methods/ad/trex_core/demo_trex_06_mmap/),
using `trexselector`.

---

## Data Generation Parameters

Each part runs twice (low- then high-dimensional) via the shared Gaussian
data-generating process (DGP) `dgp_gauss_snr`. It draws an independent and
identically distributed (i.i.d.) standard-normal design with independent
columns, the shared design family whose column-correlation is essentially the
identity:

![Design-matrix correlation of the shared i.i.d. Gaussian DGP (shown at n=300 and p=1,000 in natural order, no clustering)](../dgp_corr_plots/trex_gauss.png)

- **Low-dimensional** ($n > p$): $n = 5{,}000$, $p = 1{,}000$
- **High-dimensional** ($p > n$): $n = 1{,}000$, $p = 5{,}000$
- **True support** (0-based): fixed $\mathcal{S}^\ast = \lbrace4, 27, 149, 398, 420\rbrace$ ($s = 5$)
- **True coefficients**: fixed $\beta_j = 1$ (`rnd_coef = False`)
- Signal-to-noise ratio (SNR) $= 1.0$, target false discovery rate (tFDR) $= 0.1$, seed $= 58$

**Data size** (largest config, $n = 1{,}000$, $p = 5{,}000$): $\mathbf{X}$ is
$\approx 40$ MB — comfortably in random-access memory (RAM). The motivation for
mmap here is the augmented dummy space, because with up to $L = 10p$ dummy
columns, the internal dummy matrices $\mathbf{D}$ and serialized solver state
dominate memory (hundreds of MB), which is what `use_memory_mapping = True`
offloads to disk.

---

## Control Parameters

```text
solver_name           = TLARS      # Base solver
K                     = 20         # Random experiments per T-loop iteration
max_dummy_multiplier  = 10         # Max L = 10p
use_max_T_stop        = True       # Cap T <= ceil(n/2)
dummy_distribution    = Normal     # Dummy predictors drawn from N(0,1)
lloop_strategy        = HCONCAT    # Horizontally concatenated dummy columns
tloop_stagnation_stop = True       # Early exit when R_mat stagnates
use_memory_mapping    = True       # D mmap + solver serialization (both parts)
tFDR                  = 0.1        # Target FDR control level
```

---

## Parts

- **Part A** (`part_a_d_mmap_solver_serial`) — $\mathbf{X}$ stays in RAM, and
  `use_memory_mapping = True` gives D-mmap + solver serialization. *Use case:*
  $\mathbf{X}$ fits in RAM but the dummy/solver working set is large.
- **Part B** (`part_b_full_mmap`) — the **fully memory-mapped pipeline**, where
  $\mathbf{X}$ is written to a temporary binary file via `numpy_to_memmap`, and
  the zero-copy `X_mmap.to_numpy()` view is passed to `TRexSelector`. The temp
  file is removed in a `finally` block, which is exception-safe and mirrors the
  C++ Resource Acquisition Is Initialization (RAII) guard `MmapFileGuard`.
  *Use case:* $\mathbf{X}$ too large for RAM, with maximum memory efficiency.

`main()` runs Part A (low then high dim) followed by Part B (low then high dim).

---

## Output Files

Dual console + file output. Per configuration, one `.txt` (full selection block —
selected indices, false discovery proportion (FDP) / true positive proportion
(TPP), sparse `phi_prime` / `phi_mat`, `fdp_hat_mat`,
`r_mat`, voting grid, plus $T_{\text{stop}}$ and $L$) and one per-variable
selection `.csv` (`variable_index,phi_prime,selected,true_positive`), in
`simulation_results/data/`, stems `d02_trex_mmap_demo_a_n{n}_p{p}` and
`d02_trex_mmap_demo_b_n{n}_p{p}` — matching the C++ result files.

---

## Recorded Results

The committed recordings (seed $= 58$) show that the mmap pipeline recovers the
support cleanly in both regimes, and the selected set is exactly the true support
$\lbrace4, 27, 149, 398, 420\rbrace$ in all four cases (storage medium does not
change the selection). The table counts true positives (TP) and false positives
(FP):

| Part | Config | Selected | TP | FP | FDP | TPP | Calibration |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A (D-mmap) | $n{=}5{,}000$, $p{=}1{,}000$ | 5 | 5 | 0 | 0.000 | 1.000 | $T_{\text{stop}}{=}3$, $L{=}2$ |
| A (D-mmap) | $n{=}1{,}000$, $p{=}5{,}000$ | 5 | 5 | 0 | 0.000 | 1.000 | $T_{\text{stop}}{=}4$, $L{=}2$ |
| B (full mmap) | $n{=}5{,}000$, $p{=}1{,}000$ | 5 | 5 | 0 | 0.000 | 1.000 | $T_{\text{stop}}{=}3$, $L{=}2$ |
| B (full mmap) | $n{=}1{,}000$, $p{=}5{,}000$ | 5 | 5 | 0 | 0.000 | 1.000 | $T_{\text{stop}}{=}4$, $L{=}2$ |

- **Selection quality** matches an in-memory run — all five actives recovered
  with no false positive in all four cases.
- **A ≡ B (same seed)**: for each dimension, the in-memory-mmap (Part A) and
  fully-memory-mapped (Part B) pipelines produce **identical** output — same
  selected indices, same `Phi_prime`, and the same calibration ($T_{\text{stop}}$,
  $L$). The backing storage of $\mathbf{X}$ does not alter the selection.
- With `use_memory_mapping = True`, the dummy matrices and solver state live on
  disk and keep RAM usage low, while Part B additionally keeps $\mathbf{X}$ off
  the heap. Disk speed, solid-state drive (SSD) vs. hard disk drive (HDD), drives
  the wall-clock cost of the page-fault-driven access — prefer an SSD for the
  backing files.

---

## Running

```bash
python Python/trex_selector_methods/ad/trex_core/demo_trex_06_mmap/demo_trex_06_mmap.py
```

Single-run demo — no `multiprocessing`.
