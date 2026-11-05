# T-Solvers: Demos (R)

## Overview

This folder holds the R demos of the terminating solvers (T-solvers) that the TRexSelector R package exposes as R6
classes. They mirror [cpp/tsolvers/](../../cpp/tsolvers/README.md) and
[Python/tsolvers/](../../Python/tsolvers/README.md) demo by demo, with the same numbering, parts, settings, seed values
and console format, while R's Mersenne-Twister draws the data and all indices are 1-based, so the numbers differ from
the C++ and Python runs. The solvers are variable-selection algorithms for high-dimensional linear models inside the
Terminating-Random Experiments (T-Rex) framework. Each one follows its own path-following, greedy or pursuit strategy,
and all of them share the terminating principle. Instead of the whole path they compute only its start, up to the
$T$-th dummy that enters the active set.

Five solvers build on the machinery of least angle regression, namely terminating least angle regression (T-LARS), the
terminating least absolute shrinkage and selection operator (T-LASSO), terminating forward stepwise regression
(T-Stepwise), the terminating elastic net (T-ENET) and terminating forward stagewise regression (T-Stagewise). Six are
greedy pursuit methods, namely terminating orthogonal matching pursuit (T-OMP), terminating gradient pursuit (T-GP),
terminating approximate conjugate gradient pursuit (T-ACGP), terminating matching pursuit (T-MP), terminating
norm-corrective generalized matching pursuit (T-NCGMP) and terminating orthogonal least squares (T-OLS). Terminating
adaptive forward stepwise (T-AFS) moves between the two families with its shrinkage parameter $\rho$.

---

## Demo overview

| # | Solver | Folder | Strategy | R6 class |
| --- | ------ | ------ | -------- | -------- |
| 01 | **T-LARS** | [demo_ts_01_tlars/](ad/demo_ts_01_tlars/README.md) | Least angle regression | `TLARS_Solver` |
| 02 | **T-LASSO** | [demo_ts_02_tlasso/](ad/demo_ts_02_tlasso/README.md) | Lasso path | `TLASSO_Solver` |
| 03 | **T-Stepwise** | [demo_ts_03_tstepwise/](ad/demo_ts_03_tstepwise/README.md) | Forward stepwise regression | `TSTEPWISE_Solver` |
| 04 | **T-ENET** | [demo_ts_04_tenet/](ad/demo_ts_04_tenet/README.md) | Elastic net | `TENET_Solver` |
| 05 | **T-Stagewise** | [demo_ts_05_tstagewise/](ad/demo_ts_05_tstagewise/README.md) | Forward stagewise regression | `TSTAGEWISE_Solver` |
| 06 | **T-OMP** | [demo_ts_06_tomp/](ad/demo_ts_06_tomp/README.md) | Orthogonal matching pursuit | `TOMP_Solver` |
| 07 | **T-GP** | [demo_ts_07_tgp/](ad/demo_ts_07_tgp/README.md) | Gradient pursuit | `TGP_Solver` |
| 08 | **T-ACGP** | [demo_ts_08_tacgp/](ad/demo_ts_08_tacgp/README.md) | Approximate conjugate gradient pursuit | `TACGP_Solver` |
| 09 | **T-MP** | [demo_ts_09_tmp/](ad/demo_ts_09_tmp/README.md) | Matching pursuit | `TMP_Solver` |
| 10 | **T-NCGMP** | [demo_ts_10_tncgmp/](ad/demo_ts_10_tncgmp/README.md) | Norm-corrective generalized matching pursuit | `TNCGMP_Solver` |
| 11 | **T-OLS** | [demo_ts_11_tools/](ad/demo_ts_11_tools/README.md) | Orthogonal least squares | `TOOLS_Solver` |
| 12 | **T-AFS** | [demo_ts_12_tafs/](ad/demo_ts_12_tafs/README.md) | Adaptive forward stepwise | `TAFS_Solver` |

Every demo has the five parts of its C++ counterpart:

1. early stopping with internal normalization, for $p > n$ and $n > p$,
2. the same data normalized outside the solver with `MLLpNormScaler`,
3. a checkpoint saved to disk with `$save()`, loaded into a new solver with `$load()` and continued,
4. the predictor matrix memory-mapped from disk with `mem_matrix_from()` (the C++ demos map $\mathbf{X}$,
   $\mathbf{D}$ and $\mathbf{y}$ with $p = 500{,}000$, the R port maps $\mathbf{X}$ with $p = 5{,}000$),
5. terminated solution paths of one warm-started object, drawn as a figure.

The performance demos compare all solvers side by side, for example by false discovery rate (FDR) and true positive
rate (TPR). They keep the designs, the seed values and the comma-separated values (CSV) columns of the C++ ones, and
`ts_plt_utils.R` draws their figures:

| # | Folder | Question |
| --- | ------ | -------- |
| P1 | [demo_ts_perf_01_fdr_tpr_vs_T/](ad/demo_ts_perf_01_fdr_tpr_vs_T/README.md) | FDR and TPR of the terminated paths vs. $T$ (the simulation of the R tlars vignette) |
| P2 | [demo_ts_perf_02_runtime_vs_dummies/](ad/demo_ts_perf_02_runtime_vs_dummies/README.md) | Run time, steps and time per step vs. the number of dummies $L$, and the gain of the warm start |
| P3 | [demo_ts_perf_03_memory_vs_np/](ad/demo_ts_perf_03_memory_vs_np/README.md) | Solver memory vs. the ratio $n/p$, and its share of the data |
| P4 | [demo_ts_perf_04_recovery_vs_dt_ratio/](ad/demo_ts_perf_04_recovery_vs_dt_ratio/README.md) | Exact support recovery, TPR and FDR at the first dummy vs. $n/(s \log p)$, and the Donoho–Tanner diagram |
| P5 | [demo_ts_perf_05_null_counts_vs_nhg/](ad/demo_ts_perf_05_null_counts_vs_nhg/README.md) | Included and selected null variables vs. the negative hypergeometric distribution of the T-Rex theory |

---

## Key concepts

### Terminating principle

A terminating solver runs its path with `solver$execute_step(T_stop, early_stop = TRUE)` and stops as soon as `T_stop`
dummies are in the active set, usually long before the whole path. A later call with a larger `T_stop` continues the
stored path. The coefficients along the path serve the selection only.

### Normalization modes

The solvers work on the caller's matrices without copying, in one of two modes:

- A **writer** (`normalize = TRUE` or `intercept = TRUE` in `tsolver_options()`) centers and scales $\mathbf{X}$ and
  $\mathbf{D}$ in place, which shows on the caller's matrix or memory-mapped file. It keeps the statistics and writes
  the data back on `$restore()` or at garbage collection, and a second writer on a matrix that another writer holds
  raises an error.
- A **reader** (`normalize = FALSE, intercept = FALSE`) runs on data that the caller prepared once, for example with
  `MLLpNormScaler`, and any number of readers share the same double matrices without copies. With
  `precentered = TRUE` the solver knows that the data are centered and caps the active set at the rank $n - 1$.

In both modes $\mathbf{y}$ stays untouched, because the solver centers an internal copy.

### Checkpoints

The binding saves the solver state with `solver$save(path)`. A new solver constructed with `$new()` on the same
matrices loads it with `solver$load(path)` and continues from the same point, while the C++ demos use the static
factory `load(filename, X, D)`. `$load()` expects $\mathbf{X}$ and $\mathbf{D}$ in the state they were saved in, so
Part 3 loads into a reader on the pre-scaled data.

### Indices

All indices are 1-based. The `Predictors` and `Dummies` lines of the console output list columns of the combined
matrix $[\mathbf{X}, \mathbf{D}]$, with the dummies at the columns $p + 1$ to $p + L$, and the `Path` line numbers the
dummies on their own, so column $p + k$ is dummy `Dk`. The true supports mirror the 0-based supports of the C++ demos,
so the C++ variable 4 is the R variable 5.

### Shared helpers

[`ts_demo_utils.R`](ts_demo_utils.R) mirrors the C++ demo utilities. It draws the data (`gen_synthetic_data()`, the R
version of `datagen::SyntheticData`), prints the configuration, the selection and its quality in the format of
`cdiagnostics`, reads the selection from the coefficient path as the C++ diagnostics do, and holds `part_selection()`,
`path_recorder()` and `display_path()`, the counterparts of `PartSelection`, `PathRecorder` and `display_path()` in
[`ts_perf_utils.hpp`](../../cpp/tsolvers/ts_perf_utils.hpp). [`ts_plt_utils.R`](ts_plt_utils.R) is the ggplot2 and
patchwork port of the C++ plotting module, and its Portable Document Format (PDF) files carry no creation date, so an
unchanged figure stays byte-identical. The performance demos add [`ts_perf_solvers.R`](ts_perf_solvers.R), the 15
compared solver configurations of [`ts_perf_solvers.hpp`](../../cpp/tsolvers/ts_perf_solvers.hpp), and
[`ts_perf_utils.R`](ts_perf_utils.R) with the fixed random support, reader-ready data, the selection read from the
coefficients and `run_parallel()`, which spreads the runs over spawned R worker processes with single-threaded
solvers.

### Terminated solution paths

Part 5 repeats the quick start of the R tlars vignette. One solver object with $n = 150$, $p = 300$, five true
coefficients equal to 1 and $L = p = 300$ dummies is warm-started $T = 1 \to 2 \to 5$, and `path_recorder()` writes
each snapshot of its coefficient path in the CSV format of the C++ demos, with 1-based indices. `generate_plots.sh`
then draws the coefficients against the number of included dummies, and each demo README embeds its figure. On the R
data several solvers share one path, and their path files are byte-identical, so the coefficients agree in all ten
written digits:

| Shared path | Solvers |
| ----------- | ------- |
| T-LARS | T-LARS, T-LASSO, T-Stagewise |
| T-OMP | T-OMP, T-Stepwise, T-OLS, T-AFS ($\rho = 1$), T-NCGMP (fully corrective variant) |
| T-MP | T-MP, T-NCGMP (line-search variant) |

T-ENET, T-GP and T-ACGP take paths of their own. In contrast to the Python data, T-GP and T-ACGP enter the variables in
different orders here, and in contrast to the C++ data, T-OLS follows T-OMP to the end. Parts 1 and 2 show the same
groups, and there T-GP, T-ACGP and T-OLS take the T-OMP path as well, while T-MP leaves it after the fifth dummy for
$p > n$. The T-AFS figure shows four values of $\rho$ from 0.1 to 1, and the T-NCGMP figure both variants.

---

## Running

Run any demo from the repository root, with the TRexSelector R package 2.0.0 installed:

```bash
Rscript R/tsolvers/ad/demo_ts_01_tlars/demo_ts_01_tlars.R      # Parts 1-5
Rscript R/tsolvers/ad/demo_ts_01_tlars/demo_ts_01_tlars.R 5    # Part 5 only
R/tsolvers/ad/demo_ts_01_tlars/generate_plots.sh               # path figure from the Part 5 data
```

If the package is installed in a separate library, prefix the commands with `R_LIBS=<R library>`. Every solver demo
takes the parts to run as optional arguments and sets a budget of six Open Multi-Processing (OpenMP) threads with
`trex_set_num_threads(min(6L, trex_get_max_threads()))`. The demos print `Running with 6 threads` only when TRexSelector
was built with OpenMP, and a build without it compiles single-threaded, where `trex_get_max_threads()` is 1, so the
demos set a budget of 1 and print `Running with 1 threads`. Linux and Windows builds use R's own OpenMP flags, while on
macOS, where R from the Comprehensive R Archive Network (CRAN) has none, the configure script of the package links R's
own OpenMP runtime when an `omp.h` is available (from Homebrew or the OpenMP package of mac.r-project.org), otherwise
Homebrew's libomp, and builds single-threaded without either.

The library log lines go through `Rcpp::Rcout`, so they appear in order with the output of the demo, also in a file
or a pipe. The demos clean up the checkpoint and memory-mapped files they create inside their own folder.
`generate_plots.sh` needs the ggplot2 and patchwork packages and writes Portable Network Graphics (PNG) and PDF files
to `simulation_results/plots/`. The performance demos take the number of runs and of worker processes (500 and 6 by
default), and P1, P2 and P4 spawn their workers with `parallel::makePSOCKcluster()`, each with one thread.

---

## Folder contents

```txt
tsolvers/
  ├── README.md
  ├── ts_demo_utils.R        # shared helpers (data, diagnostics, part selection, path recorder)
  ├── ts_plt_utils.R         # shared ggplot2 figures (port of cpp/tsolvers/ts_plt_utils.py)
  ├── ts_perf_solvers.R      # the solvers compared by the performance demos
  ├── ts_perf_utils.R        # shared helpers of the performance demos
  ├── ad/                    # augmented dummies (AD)
  │   ├── demo_ts_01_tlars/
  │   ├── demo_ts_02_tlasso/
  │   ├── demo_ts_03_tstepwise/
  │   ├── demo_ts_04_tenet/
  │   ├── demo_ts_05_tstagewise/
  │   ├── demo_ts_06_tomp/
  │   ├── demo_ts_07_tgp/
  │   ├── demo_ts_08_tacgp/
  │   ├── demo_ts_09_tmp/
  │   ├── demo_ts_10_tncgmp/
  │   ├── demo_ts_11_tools/
  │   ├── demo_ts_12_tafs/
  │   ├── demo_ts_perf_01_fdr_tpr_vs_T/
  │   ├── demo_ts_perf_02_runtime_vs_dummies/
  │   ├── demo_ts_perf_03_memory_vs_np/
  │   ├── demo_ts_perf_04_recovery_vs_dt_ratio/
  │   └── demo_ts_perf_05_null_counts_vs_nhg/
  ├── sd/                    # sparse dummies (SD), README only
  └── vd/                    # virtual dummies (VD), README only
```
