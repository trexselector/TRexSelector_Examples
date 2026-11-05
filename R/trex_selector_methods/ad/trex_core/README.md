# Classical T-Rex Selector: R Demonstration Suite

## Purpose

R example programs for the classical **Terminating Random Experiments Selector
(T-Rex Selector)**, built on the **TRexSelector** R6 application programming interface (API):

```r
library(TRexSelector)

ctrl     <- trex_control(solver = "TLARS", K = 20L)
selector <- TRexSelector$new(X, y, tFDR = 0.1, control = ctrl)
selector$select()
selector$selected_indices          # 1-based selected support
trex_fdp(selector$selected_indices, true_support)
trex_tpp(selector$selected_indices, true_support)
```

The suite mirrors the C++ demos in
[cpp/trex_selector_methods/ad/trex_core/](../../../../cpp/trex_selector_methods/ad/trex_core/README.md)
(see that README for the statistical model, the notation, and the definitions of the false discovery rate (FDR)
and the true positive rate (TPR)) and covers single runs, Monte Carlo (MC) studies with fixed and variable support,
l-loop strategy and dummy distribution comparisons, memory-mapped (mmap) workflows, and a scalability benchmark.

The folder layout mirrors the C++ suite, with one subfolder per demo (each with its own `README.md` and
`simulation_results/`) plus the suite-level `trex_sim_utils.R` helper.

```txt
trex_core/
  ├── README.md
  ├── trex_sim_utils.R                 <- suite-level MC / DGP helpers
  ├── demo_trex_01_single_run/
  │   ├── demo_trex_01_single_run.R
  │   ├── README.md
  │   └── simulation_results/
  ├── demo_trex_02_mc_sim_fixed_support/
  │   └── ...
  ├── demo_trex_08_mc_sim_scalability/
  ├── demo_trex_09_mc_sim_target_fdr/
  └── demo_trex_10_mc_sim_dt_ratio/
```

---

## Demos

| # | Folder | Description | cpp counterpart |
|---|---|---|---|
| 01 | [demo_trex_01_single_run](demo_trex_01_single_run/) | Single run in a low- and a high-dimensional setting with a fixed true support (TLARS), illustrating the reporting of the internal statistics `phi_prime`, `phi_mat`, `fdp_hat_mat`, `r_mat`, `voting_grid` | `demo_trex_01_single_run` |
| 02 | [demo_trex_02_mc_sim_fixed_support](demo_trex_02_mc_sim_fixed_support/) | MC simulation with fixed support (drawn once with seed 24): 14 solvers x signal-to-noise ratio (SNR) sweep {0.1, 0.5, 1.0, 2.0, 5.0}, with averaged FDR/TPR per solver x SNR | `demo_trex_02_mc_sim_fixed_support` |
| 03 | [demo_trex_03_mc_sim_variable_support](demo_trex_03_mc_sim_variable_support/) | MC simulation with support (and optionally coefficients) redrawn each trial: 14 solvers x 21 SNR values {0.1, 0.2, …, 2.0, 5.0}, with averaged FDR / TPR / Avg L / Avg T | `demo_trex_03_mc_sim_variable_support` |
| 04 | [demo_trex_04_mc_sim_lloop_strategies](demo_trex_04_mc_sim_lloop_strategies/) | MC comparison of the six L-loop strategies (STANDARD, HCONCAT, PERMUTATION, PERMUTATION_SEEDED, SEEDED, SKIPL at 5p/10p/20p) across 4 base solvers (TLARS/TOMP/TAFS rho_afs=0.3 and TLARS_STAG, TLARS with the stagnation stop). The setting is n=300, p=1,000, s=10 with 21 SNR values and random support (a block-support call is available), and each solver writes one result file pair. The fixed-budget SKIPL FDR overshoot under TLARS comes from the missing stagnation stop, not from the least-angle regression (LARS) path | `demo_trex_04_mc_sim_lloop_strategies` |
| 05 | [demo_trex_05_mc_sim_dummy_distributions](demo_trex_05_mc_sim_dummy_distributions/) | MC comparison of 12 dummy distribution variants (Normal, Uniform, Rademacher, StudentT df 3/5, Laplace, Gumbel, Triangle, Logistic, Mammen, sparse Rademacher s=0.1, UniformSphere dim=5) across 3 base solvers (TLARS/TOMP/TAFS rho_afs=0.3) with the STANDARD L-loop strategy held fixed. The setting is n=300, p=1,000, s=10 with 7 SNR values, and each solver writes one result file pair | `demo_trex_05_mc_sim_dummy_distributions` |
| 06 | [demo_trex_06_mmap](demo_trex_06_mmap/) | Memory-mapped usage patterns. Part A: in-memory X + `use_memory_mapping = TRUE` (internal D-mmap + solver serialization). Part B: fully memory-mapped X via `mem_matrix_from()` / `mem_matrix()` | `demo_trex_06_mmap` |
| 07 | [demo_trex_07_mc_sim_mmap](demo_trex_07_mc_sim_mmap/) | MC verification of the D-mmap + solver-serialization pipeline (TLARS only, fixed support seed 24, SNR {0.1, 0.5, 1.0, 2.0, 5.0}, `tloop_max_stagnant_steps = 7`, `opt_threshold = 0.75`). Part A: in-memory X. Part B: per-trial temporary mmap-backed X/y with exception-safe cleanup | `demo_trex_07_mc_sim_mmap` |
| 08 | [demo_trex_08_mc_sim_scalability](demo_trex_08_mc_sim_scalability/) | Scalability benchmark on the fully memory-mapped pipeline in two blocks. The base block crosses n in {300, 1,000, 2,000, 5,000} with p in {1,000, 10,000, 100,000, 150,000} (dummy cap 10), and the large-p block runs n = 300 with p in {100,000, 1,000,000} (dummy cap 5). It runs 3 base solvers (TLARS/TOMP/TAFS rho_afs=0.3) with num_MC=10 sequential trials per cell, a resumable per-trial cache and a disk pre-flight per point, and it records timing, memory, and FDR/TPR | `demo_trex_08_mc_sim_scalability` |
| 09 | [demo_trex_09_mc_sim_target_fdr](demo_trex_09_mc_sim_target_fdr/) | Achieved FDR and TPR vs. the target FDR for 3 base solvers (TLARS/TOMP/TAFS rho_afs=0.3) on the demo-03 data (n = 300, p = 1,000, s = 10), SNR in {0.5, 1, 2, 5} x tFDR in {0.01, ..., 0.5} (11 levels), num_MC=200 and a seeded selector (`seeded_selector = TRUE`). | `demo_trex_09_mc_sim_target_fdr` |
| 10 | [demo_trex_10_mc_sim_dt_ratio](demo_trex_10_mc_sim_dt_ratio/) | FDR and TPR at tFDR = 0.1 vs. the sample-size ratio c = n / (s ln p) for 3 base solvers, p in {1,000, 10,000} x c in {1, ..., 10} (10 values) x SNR in {0.5, 1, 2}, num_MC=200 and a seeded selector. | `demo_trex_10_mc_sim_dt_ratio` |

Each demo folder has its own `README.md`.

---

## Shared utilities: `trex_sim_utils.R`

[trex_sim_utils.R](trex_sim_utils.R) holds the suite-level MC and data-generating process (DGP) helpers. It mirrors the
C++ `trex_sim_utils.hpp` and is sourced by every demo in this folder:

- **Section 1 — DGP (in-memory)**: `dgp_gauss_snr()` — independent and identically distributed (i.i.d.) N(0,1)
  predictors with SNR-calibrated noise.
- **Section 2 — DGP (out-of-core / mmap)**: `dgp_chunked_gauss_snr()` —
  chunked out-of-core DGP for massive datasets.
- **Section 3 — print helpers**: `.print_single_run()`.
- **Section 4 — MC infrastructure**:
  - `SOLVERS_DEFAULT` — 14 solver descriptors (TLARS, TLASSO, TENET,
    TSTAGEWISE, TSTEPWISE, TOMP, TGP, TACGP, TMP, TAFS with rho_afs 0.3 and
    1.0, TNCGMP variants v0/v1, TOOLS), mirroring
    `make_default_solvers_to_test()` in C++.
  - `make_mmap_trex_ctrl()` — `trex_control` builder for memory-mapped scenarios.
  - `execute_with_temp_mmap()` — lifecycle manager for temporary mmap files, in the style of Resource Acquisition Is
    Initialization (RAII).
  - `.make_trex_ctrl()` — builds a `trex_control()` from a solver descriptor.
  - `.run_mc_trex()` — parallel MC runner over the R6 API on parallel socket (PSOCK) workers (`mc_map()` of
    `../../mc_runtime.R`), resumable per grid cell.
  - `.print_solver_table()`, `.save_mc_csv()`, `.save_and_print_mc()` —
    console table plus combined `.txt` / `.csv` output.

Because each demo now lives one level deeper (in its own subfolder), the
suite-level helper is sourced as `../trex_sim_utils.R`. Some demos additionally
source the cross-suite shared
[../../support_generators.R](../../support_generators.R) and
[../../simulation_utils.R](../../simulation_utils.R) (at the
`trex_selector_methods/` level).

---

## Control parameters

All selector configuration goes through `trex_control()`. Parameters used
across these demos include:

- `solver` (e.g. `"TLARS"`), plus solver-specific `lambda2`, `rho_afs`,
  `ncgmp_variant`,
- `K` (number of random experiments) and `max_dummy_multiplier`,
- `lloop_strategy` (`"STANDARD"`, `"HCONCAT"`, `"PERMUTATION"`,
  `"PERMUTATION_SEEDED"`, `"SEEDED"`, `"SKIPL"`),
- `dummy_distribution` (built with `trex_dummy_distribution()`, e.g.
  `trex_dummy_distribution("StudentT", df = 3)` — see demo 05),
- `use_max_T_stop`, `tloop_stagnation_stop`, `tloop_max_stagnant_steps`,
  `opt_threshold`,
- `use_memory_mapping` (activates D-mmap + solver serialization),
- `parallel_rnd_experiments`, `seed`.

The target FDR is not part of the control object and is passed as `tFDR` to
`TRexSelector$new()` instead.

---

## Seed conventions

- **Per-trial DGP seed**: `.run_mc_trex()` derives `trial_seed = base_seed + mc - 1` and hands it to the
  data-generating factory, so trials are reproducible and independent.
- **Dummy generation**: inside the MC runner, `trex_control(seed = -1L)` is
  used — hardware entropy per trial, giving independent random experiments.
- **Random supports**: `make_support_random()` seeds with
  `seed + 500000L` (mirrors the C++ `random_support()`).
- **Fixed supports**: demos 02 and 07 draw the support once with seed 24,
  while demo 01 uses the fixed C++ support {27, 149, 43, 128, 42, 4} (0-based),
  converted to 1-based R indices {28, 150, 44, 129, 43, 5}.

---

## Running and outputs

```bash
Rscript R/trex_selector_methods/ad/trex_core/demo_trex_02_mc_sim_fixed_support/demo_trex_02_mc_sim_fixed_support.R
```

Demos run from any working directory. As in the `trex_da` / `trex_gvs` demos,
the optional first command-line argument sets the number of worker processes, else
`TREX_NUM_WORKERS`, else 8 (`0` uses every core). The MC demos write into each demo's own
`simulation_results/data/` subfolder (the `simulation_results/` umbrella holds
generated artifacts only — `data/` for tables, `plots/` for figures, matching
the C++ suite convention):

- `.txt` files — human-readable summaries,
- `.csv` files — tidy long-format tables for plotting and post-processing (e.g.
  `demo_trex_02_mc_sim_fixed_support_trex_results_n300_p1000_stagnation_window_7.csv` in
  `demo_trex_02_mc_sim_fixed_support/simulation_results/data/`).

All demos persist result files (`.csv`/`.txt`, or the mmap selection/benchmark
comma-separated values (CSV) files), and demo 01 (single run) writes its result block to a `.txt` per dimension
setting (dual console + file).
