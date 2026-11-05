# T-Solvers: Demos

## Overview

This folder holds the demos of the terminating solvers (T-solvers) in `tsolvers/` of the TRexSelector library. They
are variable-selection algorithms for high-dimensional linear models inside the T-Rex framework. Each one follows its
own path-following, greedy or pursuit strategy, and all of them share the terminating principle. Instead of the whole
path they compute only its start, up to the $T$-th dummy that enters the active set.

The solver demos run each solver on its own, outside the calibration of the T-Rex selector, so its behavior, the
normalization modes and the checkpoints can be inspected in isolation. The performance demos compare all solvers side
by side.

The folders follow the layout of the library, which sorts the solvers by the way they handle the dummies. [ad/](ad/)
holds the augmented-dummy (AD) solvers, which append an explicit dummy matrix to the predictors, and all current demos.
[sd/](sd/README.md) is reserved for the sparse-dummy (SD) solvers and [vd/](vd/README.md) for the virtual-dummy (VD)
forward selectors.

---

## Demo overview

| # | Solver | Folder | Strategy |
|---|--------|--------|----------|
| 01 | **T-LARS** | [demo_ts_01_tlars/](ad/demo_ts_01_tlars/README.md) | Least angle regression |
| 02 | **T-LASSO** | [demo_ts_02_tlasso/](ad/demo_ts_02_tlasso/README.md) | Lasso path |
| 03 | **T-Stepwise** | [demo_ts_03_tstepwise/](ad/demo_ts_03_tstepwise/README.md) | Forward stepwise regression |
| 04 | **T-ENET** | [demo_ts_04_tenet/](ad/demo_ts_04_tenet/README.md) | Elastic net |
| 05 | **T-Stagewise** | [demo_ts_05_tstagewise/](ad/demo_ts_05_tstagewise/README.md) | Forward stagewise regression |
| 06 | **T-OMP** | [demo_ts_06_tomp/](ad/demo_ts_06_tomp/README.md) | Orthogonal matching pursuit |
| 07 | **T-GP** | [demo_ts_07_tgp/](ad/demo_ts_07_tgp/README.md) | Gradient pursuit |
| 08 | **T-ACGP** | [demo_ts_08_tacgp/](ad/demo_ts_08_tacgp/README.md) | Approximate conjugate gradient pursuit |
| 09 | **T-MP** | [demo_ts_09_tmp/](ad/demo_ts_09_tmp/README.md) | Matching pursuit |
| 10 | **T-NCGMP** | [demo_ts_10_tncgmp/](ad/demo_ts_10_tncgmp/README.md) | Norm-corrective generalized matching pursuit |
| 11 | **T-OLS** | [demo_ts_11_tools/](ad/demo_ts_11_tools/README.md) | Orthogonal least squares |
| 12 | **T-AFS** | [demo_ts_12_tafs/](ad/demo_ts_12_tafs/README.md) | Adaptive forward stepwise |

The performance demos compare all solvers side by side, for example by false discovery rate (FDR) and true
positive rate (TPR), with figures drawn by `ts_plt_utils.py`:

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

A terminating solver runs its path with `executeStep(T_stop, /*early_stop=*/true)` and stops as soon as `T_stop`
dummies are in the active set, usually long before the whole path. The coefficients along the path serve the
selection only.

### Normalization modes

The solvers work in two modes:

- **Writer** (`.normalize = true` or `.intercept = true`): the solver centers and scales $\mathbf{X}$ and
  $\mathbf{D}$ in place and restores them when it is released, and it keeps a centered copy of $\mathbf{y}$. A
  second writer on a buffer that another writer holds raises an error.
- **Reader** (`.normalize = false, .intercept = false`): the caller prepares the data once, for example with
  `LpNormScaler`, and any number of solvers share the buffers without copies. With `.precentered = true` the solver
  knows that the data are centered and caps the active set at the rank $n - 1$.

Part 1 of every solver demo runs a writer, and Part 2 runs a reader on the same data.

### Checkpoints

Every solver can save its state with `save(...)` and continue after `load(...)` from the same point. Part 3 of every
solver demo checks this against an uninterrupted run.

### Terminated solution paths

Part 5 of every demo repeats the quick start of the R tlars vignette. One solver object with $n = 150$, $p = 300$,
five true coefficients equal to 1 and $L = p = 300$ dummies is warm-started $T = 1 \to 2 \to 5$, and
`ts_perf_utils.hpp` records each snapshot of its coefficient path. `ts_plt_utils.py` then draws the coefficients
against the number of included dummies, and each demo README embeds its figure. On this data several solvers share
one path to the last digit:

| Shared path | Solvers |
| ----------- | ------- |
| T-LARS | T-LARS, T-LASSO, T-Stagewise |
| T-OMP | T-OMP, T-Stepwise, T-AFS ($\rho = 1$), T-NCGMP (fully corrective variant) |
| T-MP | T-MP, T-NCGMP (line-search variant) |
| T-GP (same entry order, coefficients within 0.074) | T-GP, T-ACGP |

T-ENET and T-OLS take paths of their own. The T-AFS figure shows four values of $\rho$ from 0.1 to 1,
and the T-NCGMP figure both variants.

Over full paths an identity holds as far as the rules of the solvers agree. In
[demo_ts_perf_01](ad/demo_ts_perf_01_fdr_tpr_vs_T/) (100 runs up to $T = 100$, precentered readers, seeded
tie-breaking):

| Solvers | Full path |
| ------- | --------- |
| T-OMP, T-AFS ($\rho = 1$), T-NCGMP (fully corrective) | every step, bit for bit |
| T-OMP, T-Stepwise | the same steps, coefficients within $1.1 \cdot 10^{-12}$ |
| T-MP, T-NCGMP (line search) | the same steps, coefficients within $2.1 \cdot 10^{-14}$ (T-NCGMP divides each step by $\lVert x_j \rVert_2^2$, which is 1 up to rounding) |
| T-LARS, T-LASSO | bit for bit up to the first T-LASSO drop (a coefficient crossing zero) |
| T-LARS, T-Stagewise | within $2.7 \cdot 10^{-13}$ up to the first T-Stagewise removal (the nonnegative least-squares reduction) |

After its first drop T-LASSO takes its own path, and so does T-Stagewise after its first removal. A variable
that T-Stagewise removes keeps its coefficient (frozen), so its selection, read from the coefficient path, can
exceed the active set, and `cdiagnostics::print_selection` lists such variables separately.

---

## Building

```bash
cd cpp
cmake --preset release
cmake --build --preset release-build
```

---

## Running

Run any demo from the repository root. For example:

```bash
cpp/build/release/bin/tsolvers/ad/demo_ts_01_tlars/demo_ts_01_tlars      # Parts 1-5
cpp/build/release/bin/tsolvers/ad/demo_ts_01_tlars/demo_ts_01_tlars 5    # Part 5 only
cpp/tsolvers/ad/demo_ts_01_tlars/generate_plots.sh                        # path figure from the Part 5 data
```

Every demo takes the parts to run as optional arguments. Part 4 writes about 44 GB of temporary files, so `5`
refreshes the path data in well under a second without it.

---

## Folder contents

```txt
tsolvers/
  ├── README.md
  ├── CMakeLists.txt
  ├── ts_perf_utils.hpp      # shared C++ helpers (part selection, path recorder, selection, medians)
  ├── ts_perf_solvers.hpp    # the solvers compared by the performance demos
  ├── ts_plt_utils.py        # shared matplotlib figures
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
