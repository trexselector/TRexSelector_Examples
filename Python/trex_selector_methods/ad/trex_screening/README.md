# Screen-TRex Selector: Python Demonstration Suite

## Purpose

Python example programs for **Screen-TRex**, the Terminating Random Experiments (T-Rex) Selector extended with a
variable-**screening** step for ultra-high-dimensional data, according to
[[1]](#references). Screen-TRex is built for the regime where `p` is so large
that running the classical T-Rex selector directly is impractical (genome- or
biobank-scale designs). It first thresholds a dummy-based voting statistic
`Phi_j` to screen down to a small candidate set, using either the **Ordinary**
majority-vote rule (`Phi_j > 0.5`) or a **Bootstrap-CI** rule built on a bootstrap
confidence interval (CI), plus
**dependency-aware (DA)** variants meant for correlated designs.
Unlike the classical T-Rex selector, screening has **no target parameter for the
false discovery rate (FDR)**. It reports its own *estimated FDR* alongside the
selection, and whether that self-estimate can be trusted depends sharply on the
correlation structure (see Demos 01 and 03).

All six demos are **Monte Carlo (MC) studies** (200 repetitions per grid point) that
parallelize their trials with `concurrent.futures.ProcessPoolExecutor` and write
per-method sweep tables (`.txt` + tidy `.csv`) into each demo's own
`simulation_results/data/` folder, with figures in `simulation_results/plots/`.

```python
import trexselector as trex

# Ordinary / DA screening
sc = trex.ScreenTRexControlParameter()
sc.trex_method = trex.ScreenTRexMethod.TREX          # or a TREX_DA_* variant
tc = trex.TRexControlParameter(); tc.K = 20
sel = trex.TRexScreeningSelector(X, y, screen_control=sc, trex_control=tc,
                                seed=-1, verbose=False)
res = sel.select()
sel.selected_indices          # 0-based screened support (numpy array)
res.estimated_FDR             # procedure's self-reported FDR
res.estimated_correlation     # estimated rho (DA variants)

from trexselector.utils import compute_fdp, compute_tpp
compute_fdp(list(sel.selected_indices), true_support)
compute_tpp(list(sel.selected_indices), true_support)

# Biobank / Algorithm 1 (many phenotypes, one shared X)
bc = trex.BiobankScreenTRexControl()
bc.lower_bound_FDR = 0.05; bc.upper_bound_FDR = 0.15; bc.target_FDR_trex = 0.10
bio = trex.TRexBiobankScreeningSelector(X, Y, bio_ctrl=bc, seed=-1, verbose=False)
results = bio.select()        # BiobankScreenTRexResult, or a list for m-D Y
results[0].method_used        # routing decision per phenotype
results[0].selected_indices   # per-phenotype 0-based selected indices
```

The suite mirrors the C++ demos in
[cpp/trex_selector_methods/ad/trex_screening/](../../../../cpp/trex_selector_methods/ad/trex_screening/README.md)
one-to-one — same data-generating processes (DGPs), same grids, same Monte Carlo
counts, same output file names and comma-separated values (CSV) schemas, and the
same plotting module. Support sets and selected indices are **0-based** in Python
(matching the C++ sources, while the R suite is 1-based).

The `ScreenTRexMethod` values accepted by `ScreenTRexControlParameter.trex_method`
are `trex.ScreenTRexMethod.TREX` (ordinary voting-based screening), `.TREX_DA_AR1`,
`.TREX_DA_EQUI`, and `.TREX_DA_BLOCK_EQUI` (dependency-aware variants).

---

## What this folder covers

- **Baseline screening** (Demos 01–02): Ordinary vs. Bootstrap-CI on an independent and identically
   distributed (i.i.d.) Gaussian design, in memory and memory-mapped.
- **Correlation robustness** (Demo 03): plain screening vs. the dependency-aware variants under first-order
   autoregressive (AR(1)), equicorrelated, and block-equicorrelated designs, swept over both signal-to-noise
   ratio (SNR) and correlation strength.
- **Multi-phenotype biobank screening** (Demos 04–05): Algorithm 1's adaptive routing applied to several
   phenotypes sharing one design matrix, in memory and memory-mapped.
- **Solver sensitivity** (Demo 06): whether the T-Rex backend (TLARS, TAFS, TOMP) changes screening
   performance.

If you are new to this folder, start with **Demo 01** for the baseline comparison, then **Demo 03** for the
 correlation results, then **Demo 04** for the biobank workflow.

---

## Statistical model and targets

### The linear model

All demos use the same Gaussian linear model as the classical T-Rex selector:

```math
\boldsymbol{y} = \boldsymbol{X}\boldsymbol{\beta} + \boldsymbol{\varepsilon},
\qquad
\boldsymbol{\varepsilon} \sim \mathcal{N}(\boldsymbol{0}, \sigma_{\varepsilon}^2 \boldsymbol{I}_n).
```

### Notation

- $n$: number of observations, $p$: number of variables, $s$: number of truly active variables.
- $\boldsymbol{X} \in \mathbb{R}^{n \times p}$: design matrix, $\boldsymbol{y} \in \mathbb{R}^n$: response.
- $\mathcal{A} = \lbrace j : \beta_j \neq 0 \rbrace$: true active support, $\widehat{\mathcal{A}}$: selected set.
- $\Phi_j$: dummy-based voting proportion for variable $j$ — the evidence T-Rex accumulates internally.
- $\rho$: correlation strength (AR(1) lag-1, equicorrelation, or within-block equicorrelation).
- $\sigma_{\varepsilon}^2$: noise variance, calibrated to a target SNR.

### Correlation models

- **i.i.d. Gaussian** (Demos 01, 02, 04–06): $X_{ij} \sim \mathcal{N}(0,1)$, no correlation structure.
- **AR(1)** (Demo 03): $x_j = \rho\, x_{j-1} + \sqrt{1-\rho^2}\, w_j$, so columns correlate as
  $\rho^{|j - j'|}$.
- **Equicorrelated** (Demo 03): $x_j = \sqrt{\rho}\, z + \sqrt{1-\rho}\, w_j$ with one shared latent factor
  $z$, so every pair of columns correlates as $\rho$.
- **Block-equicorrelated** (Demo 03): the same construction with one latent factor per block, so columns
  correlate as $\rho$ within a block and are independent across blocks.

One correlation heatmap per DGP is rendered into [dgp_corr_plots/](dgp_corr_plots/) by
[generate_corr_plots.sh](generate_corr_plots.sh) (shared plotter
[../../corr_plt_utils.py](../../corr_plt_utils.py)) and embedded in the demo READMEs.

### Signal-to-noise ratio

As elsewhere in this project,
 $\mathrm{SNR} = \mathrm{Var}(\boldsymbol{X}\boldsymbol{\beta}) / \mathrm{Var}(\boldsymbol{\varepsilon})$,
 with $\sigma_{\varepsilon}$ calibrated to hit a target SNR for a fixed $(\boldsymbol{X}, \boldsymbol{\beta})$
 — in code, `sigma = sqrt(||X beta||^2 / (n * SNR))`.

### False discovery rate and true positive rate

```math
\mathrm{FDR} = \mathbb{E}\left[\frac{|\widehat{\mathcal{A}} \setminus \mathcal{A}|}{\max\{1, |\widehat{\mathcal{A}}|\}}\right],
\qquad
\mathrm{TPR} = \mathbb{E}\left[\frac{|\mathcal{A} \cap \widehat{\mathcal{A}}|}{\max\{1, |\mathcal{A}|\}}\right].
```

### What is actually measured in these demos

> **Screening returns a candidate set, and every rate is evaluated on individual variables.** The quantity
> Screen-TRex thresholds is the voting proportion $\Phi_j$, and what it returns is a set of variable
> indices. $\widehat{\mathcal{A}}$ and $\mathcal{A}$ above are therefore sets of *variables*. A false
> discovery is a selected variable that is not truly active.
>
> Screen-TRex additionally reports an **estimated FDR** — its own internal assessment of the selection's
> error rate. Every figure in this folder plots it (dashed) against the realized FDR (solid), because the
> two can diverge severely. The estimate is conservative on i.i.d. designs, except at the weakest signal, but
> wildly optimistic under equicorrelation (Demo 03).
>
> In the biobank demos, the FDR and the true positive rate (TPR) are accumulated **conditionally** on which
> method a phenotype was routed to, alongside the unconditional **Usage %**. A conditional rate over a route
> that handled no phenotypes at a given SNR is an average over an empty set, not a failure.

---

## Screen-TRex specific concepts

- **`ScreenTRexMethod`** — the screening variant: `TREX` (ordinary voting-based screening, with or without
   the bootstrap rule) and the dependency-aware variants `TREX_DA_AR1`, `TREX_DA_EQUI`,
   `TREX_DA_BLOCK_EQUI` (Demo 03).
- **Ordinary vs. Bootstrap-CI** (`ScreenTRexControlParameter.use_bootstrap_CI`) — the **Ordinary** rule
   thresholds at $\Phi_j > 0.5$, while the **Bootstrap-CI** rule builds a bootstrap confidence band
   (`R_boot = 1000` replicates, grid granularity `ci_grid_step = 0.001`) around the estimated FDR curve and
   picks its threshold from that band.
- **Dependency-aware (DA) screening** (`rho_thr_DA`, `n_blocks`) — deflates each voting proportion by its
   distance to the nearest one among its correlated neighbours (the AR(1) window, all variables, or the
   variables of the same block), so redundant correlated variables cannot pass the vote together. Screen-TRex
   runs a single step, so the proportions are coarse, and under (block-)equicorrelation many of them tie. A
   tied proportion is halved onto the threshold and rejected, which makes `TREX_DA_EQUI` select nothing and
   `TREX_DA_BLOCK_EQUI` select very little in Demo 03. For `TREX_DA_EQUI` the reference R package selects
   nothing on identical data as well.
- **Biobank / "Algorithm 1"** (`BiobankScreenTRexControl`, Demos 04–05) — screens **multiple phenotypes
   against one shared design matrix**. Per phenotype, one Screen-TRex run yields an Ordinary and a
   Bootstrap-CI estimated FDR. A rule whose estimate lies in `[lower_bound_FDR, upper_bound_FDR]` is used, and
   if both do, the larger estimate wins (a tie goes to Bootstrap-CI). If neither does, the classical T-Rex
   selector runs as the fallback at `target_FDR_trex`. The share routed to each method is reported as
   **Usage %**.

---

## Folder contents

The folder layout mirrors the C++ suite one-to-one, with one subfolder per demo
plus the shared helper modules at the suite level.

```txt
trex_screening/
  ├── README.md
  ├── trex_scr_common.py              <- DGPs, control builders, MC runner, table output
  ├── trex_scr_bb_common.py           <- Biobank (Algorithm 1) MC runner + table output
  ├── trex_scr_plt_utils.py           <- shared plotting module (sweep / biobank figures)
  ├── generate_corr_plots.sh          <- one design-correlation heatmap per DGP
  ├── run_overnight.sh                <- unattended batch runner for demos 01–06
  ├── dgp_corr_plots/                 <- scr_iid / scr_ar1 / scr_equi / scr_block_equi .png
  │
  ├── demo_trex_scr_01_mc_sim_screen_trex/
  │   ├── demo_trex_scr_01_mc_sim_screen_trex.py    # the demo source
  │   ├── README.md                                 # scenario description and results
  │   ├── generate_plots.sh                         # renders this demo's figures from its CSVs
  │   └── simulation_results/
  │       ├── data/                                 # .txt summaries + .csv tables
  │       └── plots/                                # .png + .pdf figures
  │
  │   ... demos 02–06 follow the same layout ...
  │
  └── demo_trex_scr_06_mc_sim_solvers/
      ├── demo_trex_scr_06_mc_sim_solvers.py
      ├── README.md
      ├── generate_plots.sh
      └── simulation_results/
          ├── data/
          └── plots/
```

---

## Demos

Each demo folder has its own `README.md` with its data-generating process, control parameters, recorded
results and figures.

| # | Folder | Description | cpp counterpart | Key classes and fields |
| --- | --- | --- | --- | --- |
| 01 | [demo_trex_scr_01_mc_sim_screen_trex/](demo_trex_scr_01_mc_sim_screen_trex/) | In-memory Ordinary vs. Bootstrap-CI screening: an 8-point SNR-sweep Monte Carlo reporting FDR / TPR / Estimated FDR | [demo_trex_scr_01_mc_sim_screen_trex/](../../../../cpp/trex_selector_methods/ad/trex_screening/demo_trex_scr_01_mc_sim_screen_trex/) | `TRexScreeningSelector`, `ScreenTRexControlParameter.use_bootstrap_CI` |
| 02 | [demo_trex_scr_02_mc_sim_screen_trex_mmap/](demo_trex_scr_02_mc_sim_screen_trex_mmap/) | The same SNR sweep with `use_memory_mapping=True` (dummy matrices on disk), equivalent to Demo 01 at a lower memory footprint | [demo_trex_scr_02_mc_sim_screen_trex_mmap/](../../../../cpp/trex_selector_methods/ad/trex_screening/demo_trex_scr_02_mc_sim_screen_trex_mmap/) | `TRexControlParameter.use_memory_mapping` |
| 03 | [demo_trex_scr_03_mc_sim_correlated/](demo_trex_scr_03_mc_sim_correlated/) | Ordinary/Bootstrap vs. DA screening on correlated designs: 5 MC sweeps (AR(1)/equi SNR sweeps and AR(1)/equi/block-equi rho sweeps) | [demo_trex_scr_03_mc_sim_correlated/](../../../../cpp/trex_selector_methods/ad/trex_screening/demo_trex_scr_03_mc_sim_correlated/) | `ScreenTRexMethod.TREX_DA_*`, `estimated_correlation` |
| 04 | [demo_trex_scr_04_mc_sim_biobank/](demo_trex_scr_04_mc_sim_biobank/) | Biobank "Algorithm 1" (in-memory): MC SNR sweeps (single and multi-phenotype) reporting per-method Usage % / FDR / TPR / Est. FDR | [demo_trex_scr_04_mc_sim_biobank/](../../../../cpp/trex_selector_methods/ad/trex_screening/demo_trex_scr_04_mc_sim_biobank/) | `TRexBiobankScreeningSelector`, `BiobankScreenTRexControl` |
| 05 | [demo_trex_scr_05_mc_sim_biobank_mmap/](demo_trex_scr_05_mc_sim_biobank_mmap/) | The same biobank MC sweeps with `use_memory_mapping=True` | [demo_trex_scr_05_mc_sim_biobank_mmap/](../../../../cpp/trex_selector_methods/ad/trex_screening/demo_trex_scr_05_mc_sim_biobank_mmap/) | `TRexBiobankScreeningSelector` + D-mmap |
| 06 | [demo_trex_scr_06_mc_sim_solvers/](demo_trex_scr_06_mc_sim_solvers/) | Solver-backend comparison under screening: SNR sweep for TLARS / TAFS(rho_afs=0.3) / TOMP, Part 1 Ordinary only (3 series), Part 2 x {Ordinary, Bootstrap-CI} (6 series) | [demo_trex_scr_06_mc_sim_solvers/](../../../../cpp/trex_selector_methods/ad/trex_screening/demo_trex_scr_06_mc_sim_solvers/) | `TRexControlParameter.solver_type`, `solver_params.rho_afs` |

---

## Shared modules

Both live at the suite level and mirror the C++ demo-internal headers
(`trex_screening_dgps.hpp`, `trex_screening_simulation_utils.hpp`). Each demo
resolves them via a `sys.path` bootstrap that inserts both its own directory and
the parent (suite) directory, so the shared modules import cleanly from the
nested demo location and — because the pools use the `spawn` start method —
from every `ProcessPoolExecutor` child.

- [trex_scr_common.py](trex_scr_common.py) — DGPs (`dgp_iid_snr`, `dgp_ar1`,
  `dgp_equi`, `dgp_block_equi`, `make_beta`), flat-dict control builders
  (`_build_trex_ctrl`, `_build_screen_ctrl`, `default_scr_methods`), the
  module-level ProcessPoolExecutor worker `_screen_trial_worker`, the parallel
  Monte Carlo runner `run_mc_screen`, the shared worker pool
  (`resolve_workers`, `_get_executor`), and the table writer
  `save_and_print_scr_mc` (console + `.txt` + tidy pandas `.csv`). All DGPs use
  SNR-calibrated noise, `sigma = sqrt(||X beta||^2 / (n * SNR))`.
- [trex_scr_bb_common.py](trex_scr_bb_common.py) — Biobank helpers: the control
  builder `_build_biobank_ctrl`, the module-level workers
  `_biobank_single_worker` / `_biobank_multi_worker`, the parallel runner
  `run_mc_biobank`, the per-SNR aggregator `accumulate_snr` (conditional FDR/TPR +
  unconditional Usage % and estimated FDR), and the table writer
  `save_and_print_biobank_mc`. It shares `trex_scr_common`'s worker pool.
- [trex_scr_plt_utils.py](trex_scr_plt_utils.py) — the figure renderer, a
  byte-identical copy of the C++ suite's module, so both ports produce visually
  identical figures from identically-named CSVs.

Both runners follow the suite convention, in which workers are module-level functions
that receive only picklable flat dicts. The `trexselector` enums and controls
are rebuilt inside each worker and never sent across a process boundary. Every
trial constructs its selector with `seed = -1` (hardware entropy per trial),
which the library requires for valid Monte Carlo FDR estimates.

---

## Control parameters

- `ScreenTRexControlParameter` — `trex_method`, `use_bootstrap_CI`, `R_boot`
  (Bootstrap-CI replicates, default 1,000), `ci_grid_step`, and the DA knobs
  `rho_thr_DA`, `n_blocks`, `cor_coef`. It nests its own `trex_ctrl`
  (a `TRexControlParameter`). The Python `TRexScreeningSelector` also accepts a
  separate `trex_control` argument for convenience.
- `TRexControlParameter` — `solver_type`, `K`, `use_memory_mapping`,
  `solver_params.rho_afs`, `tloop_stagnation_stop`, `tloop_max_stagnant_steps`,
  … (the same object as the classical `trex_core/` demos).
- `BiobankScreenTRexControl` — `lower_bound_FDR`, `upper_bound_FDR` (the
  acceptable-FDR window that decides between the Ordinary result, the
  Bootstrap-CI result and the T-Rex fallback in Algorithm 1), and
  `target_FDR_trex` (the fallback target). It nests
  `trex_screen_ctrl` (a `ScreenTRexControlParameter`, which in turn nests
  `trex_ctrl`).

`TRexBiobankScreeningSelector.select()` returns a single
`BiobankScreenTRexResult` for a 1-D response `y`, or a
`list[BiobankScreenTRexResult]` for a 2-D response matrix `Y`. Each result
exposes `phenotype_index`, `method_used` (`"Screen-TRex (ordinary)"`,
`"Screen-TRex (bootstrap-CI)"`, `"T-Rex (fallback)"`), `estimated_FDR`,
`estimated_FDR_screen_ordinary`, `estimated_FDR_screen_bootstrap`,
`selected_indices` (and `selected_indices_screen_ordinary` /
`selected_indices_screen_bootstrap`), and `used_fallback_trex`. The binding
returns the per-phenotype result objects directly (mirroring the C++
`BiobankScreenTRexResult` vector), and the R binding returns one record per
phenotype in the same shape.

---

## Monte Carlo counts

Every demo uses the **C++ reference counts**: `num_MC = 200` per grid point
(biobank `num_MC_multi = 200`) and `R_boot = 1000` bootstrap replicates, so the
recorded results reproduce the C++ suite's fidelity rather than a downscaled
approximation. Results are statistically comparable to the C++/R reference,
**not** bit-identical, because the three ports use different random number
generator (RNG) streams. The mirrored seeds are therefore labels only, and each
trial's selector is additionally constructed with `seed = -1`.

For a quick smoke test, reduce `_NUM_MC` and the sweep grids at the top of the
demo you are running.

---

## Parallelism and worker budget

Monte Carlo trials run in a `ProcessPoolExecutor` shared across a demo's parts.
The worker count defaults to `min(8, usable CPUs)` — matching the C++ demos'
default of 8 threads so runtimes stay comparable across the ports — and each
worker pins its own Open Multi-Processing (OpenMP) pool to a single thread, so a
demo never exceeds the worker count in total cores. Override per run, without editing sources:

```bash
SCR_NUM_WORKERS=12 python <demo>.py   # this suite's variable (takes precedence)
TREX_NUM_WORKERS=12 python <demo>.py  # the project-wide variable
TREX_NUM_WORKERS=0 python <demo>.py   # use every usable CPU
```

"Usable" honours central processing unit (CPU) affinity and cgroup quotas where the platform reports them,
so a container granted 4 processors yields 4, not the host's core count.

---

## Running and outputs

Each demo runs from any working directory:

```bash
python demo_trex_scr_01_mc_sim_screen_trex/demo_trex_scr_01_mc_sim_screen_trex.py
python demo_trex_scr_02_mc_sim_screen_trex_mmap/demo_trex_scr_02_mc_sim_screen_trex_mmap.py
python demo_trex_scr_03_mc_sim_correlated/demo_trex_scr_03_mc_sim_correlated.py
python demo_trex_scr_04_mc_sim_biobank/demo_trex_scr_04_mc_sim_biobank.py
python demo_trex_scr_05_mc_sim_biobank_mmap/demo_trex_scr_05_mc_sim_biobank_mmap.py
python demo_trex_scr_06_mc_sim_solvers/demo_trex_scr_06_mc_sim_solvers.py
```

To regenerate the whole suite as an unattended job, use
[run_overnight.sh](run_overnight.sh), which runs all six demos end to end
(continue-on-error, per-demo timestamped logs and a run summary under
`_overnight_logs/`, stray-worker reaping) and then renders the DGP correlation
heatmaps. `SKIP_HEAVY=1` skips the two biobank demos.

Approximate wall-clock times for the committed results, at the default eight
workers:

| Demo | 01 | 02 | 03 | 04 | 05 | 06 | total |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Runtime | ~1.7 min | ~2 min | ~20 min | ~20 min | ~23 min | ~7 min | ~1.2 h |

Each demo writes into its own `simulation_results/` subfolder:

- `data/*.txt` — human-readable summary tables,
- `data/*.csv` — tidy long-format tables (`method, metric, SNR|rho, value` for the
  screening demos and `method, metric, snr, value` with a `Usage` metric for the
  biobank demos) for plotting and post-processing,
- `plots/*.png` / `*.pdf` — the figures embedded in the demo READMEs.

Note that the selector draws its dummies from a nondeterministic seed, so
re-running a demo reproduces the reported behaviour but not the exact digits.

### Figures

Each demo folder has a `generate_plots.sh` that renders its figures from the
saved CSVs into `simulation_results/plots/` using the shared
`trex_scr_plt_utils.py` and the repo-root `.venv` (matplotlib + pandas + numpy):

```bash
cd demo_trex_scr_01_mc_sim_screen_trex
./generate_plots.sh
```

`trex_scr_plt_utils.py` auto-detects the CSV shape, so sweep results render as a TPR
panel and an FDR panel (realized solid, estimated dashed), while the biobank
results add a third Usage-% panel and shade the routing window. The design
correlation heatmaps are separate and suite-wide:

```bash
./generate_corr_plots.sh    # -> dgp_corr_plots/{scr_iid,scr_ar1,scr_equi,scr_block_equi}.png
```

---

## References

1. Machkour, J., Muma, M., & Palomar, D. P., "False Discovery Rate Control for Fast Screening of
   Large-Scale Genomics Biobanks.", IEEE Statistical Signal Processing Workshop (SSP), 2023,
    pp. 666–670, IEEE.
    [DOI-Link](https://doi.org/10.1109/SSP53291.2023.10207957)
