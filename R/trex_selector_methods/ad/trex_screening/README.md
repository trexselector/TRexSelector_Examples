# Screen-TRex Selector: R Demonstration Suite

## Purpose

R example programs for **Screen-TRex**, the Terminating Random Experiments (T-Rex) selector extended with a
variable-**screening** step for ultra-high-dimensional data. Screen-TRex is built for the regime where `p` is so large
that running the classical T-Rex selector directly is impractical (genome- or biobank-scale designs). It first
thresholds a dummy-based voting statistic `Phi_j` to screen down to a small candidate set, using either the
**Ordinary** majority-vote rule (`Phi_j > 0.5`) or a **Bootstrap-CI** rule built on a bootstrap confidence interval
(CI), plus **dependency-aware (DA)** variants meant for correlated designs. Unlike the classical T-Rex selector,
screening has no target for the false discovery rate (FDR). It reports its own *estimated FDR* alongside the
selection.

All six demos are **Monte Carlo (MC) studies** with 200 repetitions per grid point. They run their trials on spawned
R worker processes and write per-method tables (`.txt` + tidy `.csv`) into each demo's own
`simulation_results/data/` folder, with figures in `simulation_results/plots/`.

```r
library(TRexSelector)

# Ordinary / DA screening
sel <- TRexScreeningSelector$new(
  X, y, seed = -1L, verbose = FALSE,
  screen_control = trex_screen_control(trex_method = "TREX"),  # or a DA variant
  control        = trex_control(solver = "TLARS", K = 20L)
)
sel$select()
sel$selected_indices        # 1-based screened support
sel$estimated_FDR           # procedure's self-reported FDR
sel$estimated_correlation   # estimated rho (DA variants)
trex_fdp(sel$selected_indices, true_support)
trex_tpp(sel$selected_indices, true_support)

# Biobank / Algorithm 1 (many phenotypes, one shared X)
bio <- TRexBiobankScreeningSelector$new(
  X, Y, tFDR = 0.10, seed = -1L, verbose = FALSE,
  biobank_control = trex_biobank_control(lower_bound_FDR = 0.05, upper_bound_FDR = 0.15),
  screen_control  = trex_screen_control(R_boot = 1000L),
  control         = trex_control()
)
res <- bio$select()     # vector Y -> one record, matrix Y -> list of records
res[[1]]$method_used         # routing decision for phenotype 1
res[[1]]$selected_indices    # phenotype 1's 1-based selected indices
res[[1]]$estimated_FDR
```

The suite mirrors the C++ demos in
[cpp/trex_selector_methods/ad/trex_screening/](../../../../cpp/trex_selector_methods/ad/trex_screening/README.md)
one-to-one, with the same data-generating processes (DGPs), grids, Monte Carlo counts, output file names and
comma-separated values (CSV) schemas. That README explains the statistical model, the notation, `Phi_j`, DA screening
and Algorithm 1. Support sets and selected indices are **1-based** in R, while the C++ and Python sources are 0-based.

The `trex_method` strings accepted by `trex_screen_control()` are `"TREX"` (ordinary voting-based screening) and the
dependency-aware variants `"TREX_DA_AR1"`, `"TREX_DA_EQUI"` and `"TREX_DA_BLOCK_EQUI"`. An unknown string raises
`Unknown ScreenTRexMethod: ...` when the selector is constructed.

---

## What is actually measured

Screen-TRex returns a set of variable indices, so every rate is evaluated on individual variables. A false discovery
is a selected variable that is not truly active. The figures plot the estimated FDR (dashed) against the realized FDR
(solid), because the two can diverge. The estimate is conservative on independent designs, except at the weakest
signal, and far too optimistic under equicorrelation (demo 03).

In the biobank demos 04 and 05, the FDR and the true positive rate (TPR) of each rule are averaged **only over the
cases routed to that rule**, next to the unconditional **Usage** share. A rule that handled few cases at a
signal-to-noise ratio (SNR) point therefore gives a noisy average, and a rule that handled none gives an average over
an empty set.

---

## Folder contents

```txt
trex_screening/
  ├── README.md
  ├── trex_scr_sim_utils.R            <- DGPs, control factories, MC runners, table output
  ├── trex_scr_plt_utils.R            <- shared plotting module (sweep / biobank figures)
  ├── generate_corr_plots.sh          <- one design-correlation heatmap per DGP
  ├── run_overnight.sh                <- unattended batch runner for demos 01–06
  ├── dgp_corr_plots/                 <- scr_iid / scr_ar1 / scr_equi / scr_block_equi .png
  ├── demo_trex_scr_01_mc_sim_screen_trex/
  │   ├── demo_trex_scr_01_mc_sim_screen_trex.R
  │   ├── README.md
  │   ├── generate_plots.sh
  │   └── simulation_results/
  │       ├── data/                   # .txt summaries + .csv tables
  │       └── plots/                  # .png + .pdf figures
  │
  │   ... demos 02–06 follow the same layout ...
  │
  └── demo_trex_scr_06_mc_sim_solvers/
```

---

## Demos

Each demo folder has its own `README.md` with its DGP, control parameters, recorded results and figures.

| # | Folder | Description | Key functions |
| --- | --- | --- | --- |
| 01 | [demo_trex_scr_01_mc_sim_screen_trex/](demo_trex_scr_01_mc_sim_screen_trex/) | Ordinary vs. Bootstrap-CI screening in memory, an 8-point SNR sweep reporting FDR, TPR and estimated FDR | `TRexScreeningSelector`, `trex_screen_control(use_bootstrap_CI = )` |
| 02 | [demo_trex_scr_02_mc_sim_screen_trex_mmap/](demo_trex_scr_02_mc_sim_screen_trex_mmap/) | The same SNR sweep with `use_memory_mapping = TRUE`, which keeps the dummy matrices in files on disk | `trex_control(use_memory_mapping = )` |
| 03 | [demo_trex_scr_03_mc_sim_correlated/](demo_trex_scr_03_mc_sim_correlated/) | Ordinary/Bootstrap-CI vs. DA screening on correlated designs in 5 sweeps (SNR sweeps on first-order autoregressive (AR(1)) and equicorrelated data, rho sweeps on AR(1), equicorrelated and block-equicorrelated data) | `trex_method = "TREX_DA_*"`, `estimated_correlation` |
| 04 | [demo_trex_scr_04_mc_sim_biobank/](demo_trex_scr_04_mc_sim_biobank/) | Biobank Algorithm 1 in memory, SNR sweeps for one phenotype and for 5 phenotypes sharing one design, reporting Usage, FDR, TPR and estimated FDR per rule | `TRexBiobankScreeningSelector`, `trex_biobank_control` |
| 05 | [demo_trex_scr_05_mc_sim_biobank_mmap/](demo_trex_scr_05_mc_sim_biobank_mmap/) | The same biobank sweeps with `use_memory_mapping = TRUE` | `TRexBiobankScreeningSelector` + memory mapping |
| 06 | [demo_trex_scr_06_mc_sim_solvers/](demo_trex_scr_06_mc_sim_solvers/) | Solver backends under screening, an SNR sweep for terminating least-angle regression (TLARS), terminating adaptive forward stepwise (TAFS, shrinkage step `rho_afs = 0.3`) and terminating orthogonal matching pursuit (TOMP), Part 1 Ordinary only (3 series), Part 2 Ordinary and Bootstrap-CI (6 series) | `trex_control(solver = , rho_afs = )` |

The C++ counterpart of each demo lives in the folder of the same name under
[cpp/trex_selector_methods/ad/trex_screening/](../../../../cpp/trex_selector_methods/ad/trex_screening/).

---

## Shared utilities

[trex_scr_sim_utils.R](trex_scr_sim_utils.R) mirrors the Python modules `trex_scr_common.py` and
`trex_scr_bb_common.py` and, through them, the C++ headers `trex_screening_dgps.hpp` and
`trex_screening_simulation_utils.hpp`. Every demo sources it after the family-level
[../../mc_runtime.R](../../mc_runtime.R).

- **Section 1, DGPs.** `dgp_iid_snr()` (independent and identically distributed (i.i.d.) Gaussian), `dgp_ar1()`,
  `dgp_equi()`, `dgp_block_equi()` and `make_beta()` (evenly spaced sparse coefficients). All use SNR-calibrated
  noise, `sigma = sqrt(||X beta||^2 / (n * SNR))`.
- **Section 2, control factories.** `make_scr_trex_ctrl()` (TLARS, K = 20, `use_max_T_stop = TRUE`) and
  `default_scr_methods()`.
- **Section 3, Monte Carlo.** `run_mc_screen()` runs the trials of one grid point on parallel socket (PSOCK) workers
  via `mc_map()` and is a resumable cell of the cell cache. `save_and_print_scr_mc()` writes the console table, the
  `.txt` and the tidy `.csv`.
- **Section 4, biobank.** `biobank_single_trial()`, `biobank_multi_trial()`, `run_mc_biobank()`,
  `run_biobank_snr_cell()`, `accumulate_snr()` (Usage share plus conditional FDR and TPR per rule) and
  `save_and_print_biobank_mc()`.

[trex_scr_plt_utils.R](trex_scr_plt_utils.R) renders the figures with ggplot2 and writes the same file names as the
Python suite.

---

## Control parameters

- `trex_screen_control(use_bootstrap_CI, R_boot, ci_grid_step, trex_method)` sets the screening rule and variant.
  Unlike the C++ `make_screen_control()`, the R binding does **not** expose `rho_thr_DA` or `n_blocks`. The DA
  variants, including `TREX_DA_BLOCK_EQUI`, use the library defaults `rho_thr_DA = 0.02` and `n_blocks = 5`, which
  are the values the C++ and Python demos set, so demo 03 screens with the same settings in all three ports.
- `trex_control(solver, K, rho_afs, use_memory_mapping, ...)` configures the underlying T-Rex solver. It is the same
  object as in the classical `trex_core/` demos.
- `trex_biobank_control(lower_bound_FDR, upper_bound_FDR)` sets the acceptable-FDR window for Algorithm 1. Per
  phenotype, one Screen-TRex run yields an Ordinary and a Bootstrap-CI estimated FDR. A rule whose estimate lies in
  the window is used, and if both do, the larger estimate wins (a tie goes to Bootstrap-CI). If neither does, the
  classical T-Rex selector runs as the fallback, at the target FDR given by the selector's `tFDR` argument.

The biobank `select()` returns **one record for a vector `Y`** (single phenotype) or **a plain list of records for a
matrix `Y`** (one per phenotype). Each record is a named list with the fields `phenotype_index`, `estimated_FDR`,
`method_used` (`"Screen-TRex (ordinary)"`, `"Screen-TRex (bootstrap-CI)"`, `"T-Rex (fallback)"`),
`used_fallback_trex`, `estimated_FDR_screen_ordinary`, `estimated_FDR_screen_bootstrap`, `selected_indices` (1-based)
and `selected_indices_screen_ordinary` / `selected_indices_screen_bootstrap`. This matches the Python binding. A
summary table takes one line:

```r
do.call(rbind, lapply(res, function(r) {
  as.data.frame(r[c("phenotype_index", "estimated_FDR", "method_used", "used_fallback_trex")])
}))
```

---

## Monte Carlo counts

Every demo uses `num_MC = 200`, matching the C++ and Python references, and the biobank demos 04 and 05 add
`num_MC_multi = 200` and `R_boot = 1000`. Each trial builds its selector with `seed = -1L`, so the dummies come from
hardware entropy, as a valid Monte Carlo FDR estimate requires. The three ports use different random number
generator (RNG) streams and mirror the seeds as labels only, so results are statistically comparable but not
bit-identical.

---

## Recorded results

The committed results are the first R run of all six demos on library commit 44cf1ab (200 trials per grid point,
8 workers, 77 minutes in total). R draws its own data from the mirrored seeds, so its cells agree with C++ and Python
up to Monte Carlo noise, not exactly.

The result tables carry no standard deviations, so the comparison with C++ measures the size of the gaps and sets it
against the gaps between Python and C++. Over the 552 FDR and TPR cells of demos 01–03 and 06, the mean gap is 0.012
between R and C++ and 0.013 between Python and C++. Over the 96 FDR and TPR cells of the biobank demos in which every
port routed at least one case, it is 0.010 against 0.008, and the routing shares agree within binomial noise. The 24
cells left out have no routed case in any port, except Bootstrap-CI at SNR 0.1 in the multi-phenotype table of demo
05, which rests on a single routed case in R and none in C++ or Python.

Within one port, all series of a sweep share the same 200 datasets per grid point, so a gap at one grid point repeats
in every series. R's FDR at SNR 0.1 in demos 01, 02 and 06 is such a case. It lies 0.06–0.16 above C++ in every
series. A rerun of that grid point for Ordinary TLARS gave an FDR of 0.46 again on the original seed band and 0.367,
0.404 and 0.434 on three fresh bands, so the 200 datasets of one band move the FDR by about 0.1. The gap is of the
size the draw of R's datasets alone produces, so it is no evidence of a difference between the ports.

---

## Running and outputs

```bash
# Optional first argument = worker processes (default TREX_NUM_WORKERS, else 8, 0 = all cores)
Rscript R/trex_selector_methods/ad/trex_screening/demo_trex_scr_01_mc_sim_screen_trex/demo_trex_scr_01_mc_sim_screen_trex.R
Rscript R/trex_selector_methods/ad/trex_screening/demo_trex_scr_02_mc_sim_screen_trex_mmap/demo_trex_scr_02_mc_sim_screen_trex_mmap.R
Rscript R/trex_selector_methods/ad/trex_screening/demo_trex_scr_03_mc_sim_correlated/demo_trex_scr_03_mc_sim_correlated.R
Rscript R/trex_selector_methods/ad/trex_screening/demo_trex_scr_04_mc_sim_biobank/demo_trex_scr_04_mc_sim_biobank.R
Rscript R/trex_selector_methods/ad/trex_screening/demo_trex_scr_05_mc_sim_biobank_mmap/demo_trex_scr_05_mc_sim_biobank_mmap.R
Rscript R/trex_selector_methods/ad/trex_screening/demo_trex_scr_06_mc_sim_solvers/demo_trex_scr_06_mc_sim_solvers.R 8
R/trex_selector_methods/ad/trex_screening/run_overnight.sh               # all six demos
SKIP_HEAVY=1 R/trex_selector_methods/ad/trex_screening/run_overnight.sh  # without the biobank demos 04 and 05
```

Demos run from any working directory, because each resolves the shared files relative to its own location. Every
finished grid point is stored in the demo's cell cache (`simulation_results/cache/`, gitignored), so a restarted demo
resumes where it stopped. `TREX_FRESH=1` starts from scratch, and a finished demo removes its cache.

Runtimes of the committed run at 8 workers:

| Demo | 01 | 02 | 03 | 04 | 05 | 06 | total |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Runtime | 2 min | 2 min | 22 min | 20 min | 24 min | 8 min | 77 min |

Each demo writes into its own `simulation_results/` subfolder:

- `data/*.txt` with human-readable summary tables,
- `data/*.csv` with tidy long-format tables (`method, metric, SNR|rho, value` for the screening demos and
  `method, metric, snr, value` with a `Usage` metric for the biobank demos),
- `plots/*.png` / `*.pdf` with the figures embedded in the demo READMEs.

### Figures

Each demo folder has a `generate_plots.sh` that renders its figures from the saved CSVs into
`simulation_results/plots/` via the shared [trex_scr_plt_utils.R](trex_scr_plt_utils.R). Sweep results render as a
TPR panel and an FDR panel (realized solid, estimated dashed), while the biobank results add a Usage panel and shade
the routing window. The design-correlation heatmaps are suite-wide:

```bash
cd R/trex_selector_methods/ad/trex_screening/demo_trex_scr_01_mc_sim_screen_trex && ./generate_plots.sh
R/trex_selector_methods/ad/trex_screening/generate_corr_plots.sh   # -> dgp_corr_plots/*.png
```

---

## References

1. Machkour, J., Muma, M., & Palomar, D. P., "False Discovery Rate Control for Fast Screening of
   Large-Scale Genomics Biobanks.", IEEE Statistical Signal Processing Workshop (SSP), 2023,
   pp. 666–670, IEEE.
   [DOI-Link](https://doi.org/10.1109/SSP53291.2023.10207957)
