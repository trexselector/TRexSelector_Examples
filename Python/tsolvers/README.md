# T-Solvers: Demos (Python)

## Overview

This folder holds the Python demos of the terminating solvers (T-solvers) exposed by `trexselector.tsolvers`. They
mirror [cpp/tsolvers/](../../cpp/tsolvers/README.md) demo by demo, with the same numbering, parts, settings, seeds and
console format, while NumPy draws the data, so the numbers differ from the C++ runs. The solvers are variable-selection
algorithms for high-dimensional linear models inside the T-Rex framework. Each one follows its own path-following,
greedy or pursuit strategy, and all of them share the terminating principle. Instead of the whole path they compute
only its start, up to the $T$-th dummy that enters the active set.

The folders follow the layout of the library, which sorts the solvers by the way they handle the dummies. [ad/](ad/)
holds the augmented-dummy (AD) solvers, which append an explicit dummy matrix to the predictors, and all current demos.
[sd/](sd/README.md) is reserved for the sparse-dummy (SD) solvers and [vd/](vd/README.md) for the virtual-dummy (VD)
forward selectors.

---

## Demo overview

| # | Solver | Folder | Strategy | Binding class |
|---|--------|--------|----------|---------------|
| 01 | **T-LARS** | [demo_ts_01_tlars/](ad/demo_ts_01_tlars/README.md) | Least angle regression | `tsolvers.ad.lars_based.TLARS_Solver` |
| 02 | **T-LASSO** | [demo_ts_02_tlasso/](ad/demo_ts_02_tlasso/README.md) | Lasso path | `tsolvers.ad.lars_based.TLASSO_Solver` |
| 03 | **T-Stepwise** | [demo_ts_03_tstepwise/](ad/demo_ts_03_tstepwise/README.md) | Forward stepwise regression | `tsolvers.ad.lars_based.TSTEPWISE_Solver` |
| 04 | **T-ENET** | [demo_ts_04_tenet/](ad/demo_ts_04_tenet/README.md) | Elastic net | `tsolvers.ad.lars_based.TENET_Solver` |
| 05 | **T-Stagewise** | [demo_ts_05_tstagewise/](ad/demo_ts_05_tstagewise/README.md) | Forward stagewise regression | `tsolvers.ad.lars_based.TSTAGEWISE_Solver` |
| 06 | **T-OMP** | [demo_ts_06_tomp/](ad/demo_ts_06_tomp/README.md) | Orthogonal matching pursuit | `tsolvers.ad.omp_based.TOMP_Solver` |
| 07 | **T-GP** | [demo_ts_07_tgp/](ad/demo_ts_07_tgp/README.md) | Gradient pursuit | `tsolvers.ad.omp_based.TGP_Solver` |
| 08 | **T-ACGP** | [demo_ts_08_tacgp/](ad/demo_ts_08_tacgp/README.md) | Approximate conjugate gradient pursuit | `tsolvers.ad.omp_based.TACGP_Solver` |
| 09 | **T-MP** | [demo_ts_09_tmp/](ad/demo_ts_09_tmp/README.md) | Matching pursuit | `tsolvers.ad.omp_based.TMP_Solver` |
| 10 | **T-NCGMP** | [demo_ts_10_tncgmp/](ad/demo_ts_10_tncgmp/README.md) | Norm-corrective generalized matching pursuit | `tsolvers.ad.omp_based.TNCGMP_Solver` |
| 11 | **T-OLS** | [demo_ts_11_tools/](ad/demo_ts_11_tools/README.md) | Orthogonal least squares | `tsolvers.ad.omp_based.TOOLS_Solver` |
| 12 | **T-AFS** | [demo_ts_12_tafs/](ad/demo_ts_12_tafs/README.md) | Adaptive forward stepwise | `tsolvers.ad.afs_based.TAFS_Solver` |

Every demo has the five parts of its C++ counterpart:

1. early stopping with internal normalization, for $p > n$ and $n > p$,
2. the same data normalized outside the solver with `LpNormScaler`,
3. a checkpoint saved to disk, reloaded and continued,
4. data memory-mapped from disk with `numpy_to_memmap()` (the C++ demos use $p = 500{,}000$, the Python port keeps
   $p = 5{,}000$),
5. terminated solution paths of one warm-started object, drawn as a figure.

The performance demos compare all solvers side by side, for example by false discovery rate (FDR) and true positive rate
(TPR), with the designs, seeds and comma-separated values (CSV) columns of the C++ ones and figures drawn by
`ts_plt_utils.py`:

| # | Folder | Question |
|---|--------|----------|
| P1 | [demo_ts_perf_01_fdr_tpr_vs_T/](ad/demo_ts_perf_01_fdr_tpr_vs_T/README.md) | FDR and TPR of the terminated paths vs. $T$ (the simulation of the R tlars vignette) |
| P2 | [demo_ts_perf_02_runtime_vs_dummies/](ad/demo_ts_perf_02_runtime_vs_dummies/README.md) | Run time, steps and time per step vs. the number of dummies $L$, and the gain of the warm start |
| P3 | [demo_ts_perf_03_memory_vs_np/](ad/demo_ts_perf_03_memory_vs_np/README.md) | Solver memory vs. the ratio $n/p$, and its share of the data |
| P4 | [demo_ts_perf_04_recovery_vs_dt_ratio/](ad/demo_ts_perf_04_recovery_vs_dt_ratio/README.md) | Exact support recovery, TPR and FDR at the first dummy vs. $n/(s \log p)$, and the Donoho–Tanner diagram |
| P5 | [demo_ts_perf_05_null_counts_vs_nhg/](ad/demo_ts_perf_05_null_counts_vs_nhg/README.md) | Included and selected null variables vs. the negative hypergeometric distribution of the T-Rex theory |

---

## Key concepts

### Terminating principle

A terminating solver runs its path with `executeStep(T_stop, early_stop=True)` and stops as soon as `T_stop` dummies
are in the active set, usually long before the whole path. The coefficients along the path serve the selection only.

### Normalization modes

The solvers work in two modes:

- **Writer** (`normalize=True` or `intercept=True`): the solver centers and scales $\mathbf{X}$ and $\mathbf{D}$ in
  place, restores them when it is released and keeps a centered copy of $\mathbf{y}$. A second writer on an array
  that another writer holds raises an error.
- **Reader** (`normalize=False, intercept=False`): the caller prepares the data once, for example with
  `LpNormScaler`, and any number of solvers share the Fortran-ordered `float64` arrays without copies. With
  `precentered=True` the solver knows that the data are centered and caps the active set at the rank $n - 1$.

### Checkpoints

The binding saves the solver state with `solver.save(path)`. A new solver constructed on the same arrays loads it with
`solver.load(path)` and continues from the same point (the C++ demos use the static factory `load(filename, X, D)`).

### Shared helpers

[`ts_demo_utils.py`](ts_demo_utils.py) mirrors the C++ demo utilities. It draws the data (`gen_synthetic_data()`, the
NumPy version of `datagen::SyntheticData`), prints the configuration, the selection and its quality in the format of
`cdiagnostics`, reads the selection from the coefficient path as the C++ diagnostics do, and holds `PartSelection`,
`PathRecorder` and `display_path()` of [`ts_perf_utils.hpp`](../../cpp/tsolvers/ts_perf_utils.hpp). It also sets
Python's and the C library's standard output to line buffering, so the log lines of the C++ backend appear in their
place when the output goes to a file or a pipe. [`ts_plt_utils.py`](ts_plt_utils.py) is a byte-identical copy of the
C++ plotting module. The performance demos add [`ts_perf_solvers.py`](ts_perf_solvers.py), the 15 compared solver
configurations of [`ts_perf_solvers.hpp`](../../cpp/tsolvers/ts_perf_solvers.hpp), and
[`ts_perf_utils.py`](ts_perf_utils.py) with the fixed random support, reader-ready data, the selection read from the
coefficients and a pool of worker processes with single-threaded solvers.

### Terminated solution paths

Part 5 repeats the quick start of the R tlars vignette. One solver object with $n = 150$, $p = 300$, five true
coefficients equal to 1 and $L = p = 300$ dummies is warm-started $T = 1 \to 2 \to 5$, and `PathRecorder` writes each
snapshot of its coefficient path in the CSV format of the C++ demos. `generate_plots.sh` then draws the coefficients
against the number of included dummies, and each demo README embeds its figure. On the Python data several solvers
share one path, to all ten digits of the written coefficients:

| Shared path | Solvers |
| ----------- | ------- |
| T-LARS | T-LARS, T-LASSO, T-Stagewise |
| T-OMP | T-OMP, T-Stepwise, T-OLS, T-AFS ($\rho = 1$), T-NCGMP (fully corrective variant) |
| T-MP | T-MP, T-NCGMP (line-search variant) |
| T-GP (same entries, coefficients within 0.089) | T-GP, T-ACGP |

T-ENET takes a path of its own. The T-AFS figure shows four values of $\rho$ from 0.1 to 1, and the T-NCGMP figure both
variants. In contrast to the C++ data, T-OLS follows T-OMP to the end here.

---

## Running

Run any demo from the repository root, with the package installed in the repository's virtual environment:

```bash
.venv/bin/python Python/tsolvers/ad/demo_ts_01_tlars/demo_ts_01_tlars.py      # Parts 1-5
.venv/bin/python Python/tsolvers/ad/demo_ts_01_tlars/demo_ts_01_tlars.py 5    # Part 5 only
Python/tsolvers/ad/demo_ts_01_tlars/generate_plots.sh                          # path figure from the Part 5 data
```

Every solver demo takes the parts to run as optional arguments, and all five parts run in a few seconds. The demos
clean up the checkpoint and memory-mapped files they create inside their own folder. The performance demos take the
number of runs and of worker processes (500 and 6 by default), and their READMEs give the run times.

---

## Folder contents

```txt
tsolvers/
  ├── README.md
  ├── ts_demo_utils.py       # shared helpers (data, diagnostics, part selection, path recorder)
  ├── ts_plt_utils.py        # shared matplotlib figures (copy of cpp/tsolvers/ts_plt_utils.py)
  ├── ts_perf_solvers.py     # the solvers compared by the performance demos
  ├── ts_perf_utils.py       # shared helpers of the performance demos
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
