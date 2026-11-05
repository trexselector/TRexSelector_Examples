# T-Rex Selector Methods — R Demos

R demonstration suites for the Terminating Random Experiments (T-Rex) selector variants, mirroring
[cpp/trex_selector_methods/](../../cpp/trex_selector_methods/). Each variant
lives in its own sub-directory, and Monte Carlo (MC) demos write results to a
source-side `simulation_results/` folder next to the demo file.

The folders follow the layout of the library, which sorts the selectors by the way they handle the dummies. [ad/](ad/)
holds the variants with augmented dummies (AD), which append an explicit dummy matrix to the predictors, and all current
demos. [sd/](sd/README.md) is reserved for the sparse-dummy (SD) selector and [vd/](vd/README.md) for the virtual-dummy
(VD) selector.

---

## Contents

| Folder | Status | What it covers |
|---|---|---|
| [ad/trex_core/](ad/trex_core/README.md) | New — TRexSelector R6 application programming interface (API) | Classical T-Rex: single runs, MC studies (fixed/variable support), l-loop strategies, dummy distributions, memory-mapped workflows, scalability benchmark (demos 01–08, folder-per-demo) |
| [ad/trex_da/](ad/trex_da/README.md) | New — TRexSelector R6 API | Dependency-Aware T-Rex (DA-TRex) via `TRexDASelector` over first-order autoregressive (AR(1)), equicorrelated, nearest-neighbor, block-diagonal, and heavy-tailed designs (demos 01–08, folder-per-demo) |
| [ad/trex_gvs/](ad/trex_gvs/README.md) | New — TRexSelector R6 API | Grouped Variable Selection (GVS) T-Rex via `TRexGVSSelector` with the solver variants TENET (elastic net), TENET_AUG (elastic net in augmented-lasso form) and TIENET_AUG (informed elastic net) over Hastie, scattered, mixed, negative-trap, equi-block, t-distributed, AR(1), and autoregressive moving-average (ARMA) designs (demos 01–08, folder-per-demo) |
| [ad/trex_spca/](ad/trex_spca/README.md) | New — TRexSelector R6 API | T-Rex sparse principal component analysis (PCA) with `TRexSPCASelector` (2 solvers × 2 modes) vs. ordinary/oracle PCA on a sparse 3-factor model (demos 01–02, folder-per-demo) |
| [ad/trex_screening/](ad/trex_screening/README.md) | New — TRexSelector R6 API | Screen-TRex (`TRexScreeningSelector` / `TRexBiobankScreeningSelector`): in-memory and memory-mapped screening, correlated designs, biobank-scale, and solver comparison (demos 01–06, folder-per-demo) |

The R-side cross-check probes for trex_spca (the rdump generator and its C++/R
comparison probes) moved to the TRexSelector library test suite
(`TRexSelector/cpp/tests/validation/trex_selector_methods/ad/trex_spca/`).

---

## Shared helpers

| File | Status | Description |
|---|---|---|
| [mc_runtime.R](mc_runtime.R) | Shared | Monte Carlo runtime of every MC demo: `mc_map()` runs the trials on a persistent parallel socket (PSOCK) cluster of spawned R worker processes (no fork) with single-threaded solvers, and `cell_cached()` / `cell_cache_done()` make each grid cell resumable (see [Running](#running)). |
| [support_generators.R](support_generators.R) | Shared | Support-index construction helpers, sourced by demos across all variants: `make_support_capped_spread()` (deterministic evenly-spaced support with capped gap), `make_support_random()` (uniformly random support, internally using `set.seed(seed + 500000L)`), `make_support_bt_one_per_block()` (one active predictor per Binary-Tree (BT) block). All return 1-based R indices and mirror the C++ helpers. |
| [simulation_utils.R](simulation_utils.R) | New — migrated to the TRexSelector R6 API (`TRexGVSSelector` + `trex_fdp()` / `trex_tpp()`) | Shared MC engine for the trex_gvs demos plus generic output helpers: `plot_cormat()` (Plotly heatmap), table printers, and the GVS wrappers `.run_mc_hastie()` … `.run_mc_arma_blocks()` behind the TENET / TENET_AUG / TIENET_AUG rows, plus the `.gvs_select()` adapter. (The DA-TRex MC engine lives in [ad/trex_da/da_sim_common.R](ad/trex_da/da_sim_common.R), sourced by the trex_da demos directly.) |

The new-API trex demos additionally use their own utility file,
[ad/trex_core/trex_sim_utils.R](ad/trex_core/trex_sim_utils.R) (see the
[trex_core README](ad/trex_core/README.md)).

---

## Running

All demos run from any working directory, e.g.

```bash
Rscript R/trex_selector_methods/ad/trex_core/demo_trex_02_mc_sim_fixed_support/demo_trex_02_mc_sim_fixed_support.R
Rscript R/trex_selector_methods/ad/trex_da/demo_trex_da_01_mc_sim_ar1/demo_trex_da_01_mc_sim_ar1.R
```

To run the Monte Carlo demos back to back as an unattended job, use the batch runner, which runs every suite's
`run_overnight.sh` (or the ones named as arguments) with `TREX_NUM_WORKERS` workers per demo (default 8):

```bash
nohup R/trex_selector_methods/run_all_overnight.sh > /dev/null 2>&1 &
```

Both levels are resumable. A demo that finishes is recorded under the run tag (the pinned library commit of
`LIBRARY_TARGET.txt`) and skipped by the next run, and the demo in flight keeps every finished grid cell in
`simulation_results/cache/cells.tsv`, so a stopped batch continues where it stopped. `FRESH=1` reruns everything.

The gvs MC suite additionally requires `plotly` (correlation heatmaps), and all MC suites use base-R `parallel`
for their worker processes. The trex_spca cross-check programs (which
additionally require `glmnet`) moved to the TRexSelector library test suite
(`TRexSelector/cpp/tests/validation/trex_selector_methods/ad/trex_spca/`).
