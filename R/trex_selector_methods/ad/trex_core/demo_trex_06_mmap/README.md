# Demo 06: Memory-Mapped Data — Single Run

## Purpose

Demonstrate memory-mapped (mmap) input/output (I/O) workflows for the Terminating Random Experiments (T-Rex) selector
via two single-run usage patterns. `use_memory_mapping = TRUE` memory-maps the internal dummy
matrices `D` and serializes solver least-angle regression (LARS) path state to disk between T-loop iterations.
Part B additionally memory-maps the design matrix `X` itself.

## Data-generating process

`dgp_gauss_snr` — independent and identically distributed (i.i.d.) Gaussian design, `y = X beta + eps` calibrated
to a signal-to-noise ratio (SNR). Shared config (`DEMO_CFG`): fixed true support `c(28, 150, 399, 421, 5)` (1-based,
C++ 0-based `{27, 149, 398, 420, 4}`, `s = 5`), coefficients `beta_j = 1`, `SNR = 1.0`,
`tFDR = 0.1`, `seed = 58`. Each part runs twice, first low-dimensional `n = 5000, p = 1000`
and then high-dimensional `n = 1000, p = 5000`. Control via `make_mmap_trex_ctrl(solver = "TLARS")`.

The design belongs to the suite's shared i.i.d. Gaussian family — independent
columns, so the empirical column-correlation matrix is essentially the identity
at every `(n, p)` this demo runs:

![Design-matrix correlation of the shared i.i.d. Gaussian DGP (shown at n=300, p=1,000, natural order, no clustering)](../dgp_corr_plots/trex_gauss.png)

## Parts

- Part A (`run_part_a`) — in-memory `X` + `use_memory_mapping = TRUE`, which activates D-mmap
  and solver serialization inside `TRexSelector`. Mirror of the C++ Demo A.
- Part B (`run_part_b`) — memory-mapped `X` + `use_memory_mapping = TRUE`. Here `X` is written
  to a temporary mmap file via `execute_with_temp_mmap` before being passed to the
  selector (full pipeline: X mmap + D mmap + solver serialization). The R interface
  requires writing the generated matrix to disk first (C++ uses `SyntheticDataMapped`,
  where `X` is never fully in random-access memory (RAM)).

`.print_and_save_d02` prints (dual console + file) the selected indices, the false discovery proportion (FDP), the
true positive proportion (TPP), the sparse `Phi_prime` / `Phi` structures, the dense `FDP_hat` / `R` matrices, the
voting grid, and `T_stop` / `L`.

## Running

```bash
Rscript R/trex_selector_methods/ad/trex_core/demo_trex_06_mmap/demo_trex_06_mmap.R
```

Each run writes a `.txt` result block (dual console + file, with the sparse
`Phi_prime`/`Phi` section) and a per-variable selection **comma-separated values (CSV)** file to this demo's own
`simulation_results/` folder — CSV header `variable_index,phi_prime,selected,true_positive`,
stems `d02_trex_mmap_demo_a_n{n}_p{p}` and `d02_trex_mmap_demo_b_n{n}_p{p}`. The
script sources `../../../simulation_utils.R` and `../trex_sim_utils.R`.
