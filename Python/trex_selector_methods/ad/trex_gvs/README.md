# T-Rex+GVS: Grouped Variable Selection — Python Demos

## Overview

This folder contains the Python ports of the demonstration suite for the **Terminating Random Experiments (T-Rex)
Selector with Grouped Variable Selection (GVS)**, in short T-Rex+GVS, according to [[1,2]](#references), using
`trexselector`'s `TRexGVSSelector`. They mirror the authoritative C++ suite in
[cpp/trex_selector_methods/ad/trex_gvs/](../../../../cpp/trex_selector_methods/ad/trex_gvs/README.md) one-to-one (folder
names, scenarios, sweep grids, console tables, and result-file names all match).

T-Rex+GVS extends the classical T-Rex selector (see [../trex_core/](../trex_core/README.md)) to settings where
predictors form known or discoverable **groups** of correlated variables.
 Examples are genes in the same pathway, or sensor channels driven by a shared latent signal.
 The selector aggregates the evidence of the random experiments at the group level while it runs, so that correlated
 variables reinforce each other instead of splitting the vote.
 However, the selection is based on individual variables, and that is also the level at which false discovery rate
 (FDR) and true positive rate (TPR) are evaluated
 (see [What is actually measured](#what-is-actually-measured-in-these-demos)).

The demos in this folder are designed to help users understand:

1. how grouped-signal recovery behaves under different within-group correlation structures (equicorrelated, first-order
    autoregressive (AR(1)), autoregressive moving-average (ARMA), heavy-tailed),
2. how group *discovery* via hierarchical agglomerative clustering (HAC) compares to using known ("oracle") groups,
3. how the elastic-net-based (`EN`) and informed-elastic-net-based (`IEN`) group solvers behave under scattered
    layouts, correlation traps, and negative correlation structure.

If you are new to this folder, start with **Demo 01** for the canonical equicorrelated-blocks scenario.

---

## What this folder covers

All demos study sparse variable selection in a Gaussian (or near-Gaussian) linear model where the active variables
 cluster into groups:

- **equicorrelated blocks** (Demos 01–04, 08): variables within a group share a common latent factor,
- **scattered groups** (Demo 02): group members are permuted across the column space instead of being contiguous,
- **gapped, randomly-ordered blocks with an inactive trap** (Demo 03, and reused by Demos 05–07): four unequal-size
   blocks, three active and one inactive equicorrelated trap,
- **negative-correlation structure** (Demo 04): groups with sign-flipped halves and spatially separated inactive traps,
- **non-Gaussian / non-equicorrelated structure** (Demos 05–07): Student-t(3) heavy tails, AR(1) Toeplitz covariance,
   and heterogeneous ARMA processes per block,
- **value of prior group information** (Demo 08): a larger 100-block benchmark comparing HAC-discovered groups
   against known ("oracle") group labels, sweeping the within-block correlation across the HAC discoverability
   boundary.

| # | Demo | Scenario | Output stem | cpp counterpart |
|---|---|---|---|---|
| 01 | [demo_trex_gvs_01_mc_sim_hastie_en_blocks](demo_trex_gvs_01_mc_sim_hastie_en_blocks/) | Hastie: 3 equicorrelated active groups of 50 (s = 150), 2-D grid + lambda2/linkage/cut studies | `gvs_Hastie_*` | [cpp demo 01](../../../../cpp/trex_selector_methods/ad/trex_gvs/demo_trex_gvs_01_mc_sim_hastie_en_blocks/README.md) |
| 02 | [demo_trex_gvs_02_mc_sim_scattered_blocks](demo_trex_gvs_02_mc_sim_scattered_blocks/) | Same 3 groups, members permuted across columns | `gvs_scattered_grouped_2d` | [cpp demo 02](../../../../cpp/trex_selector_methods/ad/trex_gvs/demo_trex_gvs_02_mc_sim_scattered_blocks/README.md) |
| 03 | [demo_trex_gvs_03_mc_sim_mixed_blocks](demo_trex_gvs_03_mc_sim_mixed_blocks/) | 3 active blocks (20/50/80) + 1 trap (65), random order, noise gaps | `gvs_mixed_blocks_2d` | [cpp demo 03](../../../../cpp/trex_selector_methods/ad/trex_gvs/demo_trex_gvs_03_mc_sim_mixed_blocks/README.md) |
| 04 | [demo_trex_gvs_04_mc_sim_neg_traps](demo_trex_gvs_04_mc_sim_neg_traps/) | Active +Z/−Z group + two inactive +Z/−Z traps (s = 100) | `gvs_Neg-Traps_2d` | [cpp demo 04](../../../../cpp/trex_selector_methods/ad/trex_gvs/demo_trex_gvs_04_mc_sim_neg_traps/README.md) |
| 05 | [demo_trex_gvs_05_mc_sim_t3_blocks](demo_trex_gvs_05_mc_sim_t3_blocks/) | Demo-03 geometry with Student-t(3) factors/noise | `gvs_t3_blocks_2d` | [cpp demo 05](../../../../cpp/trex_selector_methods/ad/trex_gvs/demo_trex_gvs_05_mc_sim_t3_blocks/README.md) |
| 06 | [demo_trex_gvs_06_mc_sim_ar1_blocks](demo_trex_gvs_06_mc_sim_ar1_blocks/) | Demo-03 geometry with within-block Cor = rho^\|j−k\| | `gvs_ar1_blocks_2d` | [cpp demo 06](../../../../cpp/trex_selector_methods/ad/trex_gvs/demo_trex_gvs_06_mc_sim_ar1_blocks/README.md) |
| 07 | [demo_trex_gvs_07_mc_sim_arma_blocks](demo_trex_gvs_07_mc_sim_arma_blocks/) | AR(1) / moving-average (MA(3)) / ARMA(2,1) active blocks + AR(1) trap, ar_coef axis | `gvs_arma_blocks_2d` | [cpp demo 07](../../../../cpp/trex_selector_methods/ad/trex_gvs/demo_trex_gvs_07_mc_sim_arma_blocks/README.md) |
| 08 | [demo_trex_gvs_08_mc_sim_block_bench](demo_trex_gvs_08_mc_sim_block_bench/) | 100 blocks × 5, 10 active, HAC vs oracle groups (M1–M4), WHOLE + INDIVIDUAL truths | `gvs_block_bench_*` | [cpp demo 08](../../../../cpp/trex_selector_methods/ad/trex_gvs/demo_trex_gvs_08_mc_sim_block_bench/README.md) |

Demos 01–07 compare the three solver variants **TENET** (elastic net (EN)), **TENET_AUG** (EN, row-augmented
solver), and **TIENET_AUG** (informed elastic net (IEN)) on a 2-D signal-to-noise ratio (SNR) × correlation grid
(fixed settings: `n = 200`, `p = 500`, `tFDR = 0.1`, `K = 20`, `corr_max = 0.98`, 200 Monte Carlo (MC) trials per
grid point). Demo 01 adds `lambda2_method` and `hc_linkage` robustness checks and a clustering-cut comparison
(Part 4). Demo 08 compares EN/IEN × HAC/oracle
(M1–M4) at `corr_max = 0.5`.

---

## What is actually measured in these demos

> **Selection and evaluation are at the level of individual variables.** Although T-Rex+GVS
> *aggregates the evidence of the random experiments at the group level* while selecting, what the selector returns
> is a set of individual variable indices, and every FDR/TPR number reported in Demos 01–07 is
> **coordinate-level**. A false discovery is a selected variable that is not truly active.
>
> A genuinely **group-level FDR** — treating a whole group as the unit of discovery — is a
> different quantity and needs its own definition of what counts as a false discovery, together with a matching
> notion of the group null hypothesis. See [[3]](#references) for such a formulation in the knockoff setting. It is
> *not* obtainable by reinterpreting the coordinate-level numbers.

Demo 01 Part 4 (group hit, group full, group FDR, groups formed under three clustering cuts) and Demo 08 (block hit
 rate, block FDR, full-block rate, block purity) additionally report group-level rates.

> **T-Rex+GVS as implemented is not designed to control a group-level FDR.** It controls the FDR of the selected
> variables, and $\mathrm{tFDR}$ bounds that rate only. The group- and block-level rates are descriptive, carry no
> target and no guarantee, and a value above $\mathrm{tFDR}$ does not indicate a loss of FDR control. With many
> singleton background groups next to a few active groups, the group FDR can be large (Demo 01 Part 4) while the
> variable-level FDR stays below the target.

For the statistical model, the within-group correlation models, and the FDR/TPR definitions, see the
[C++ suite README](../../../../cpp/trex_selector_methods/ad/trex_gvs/README.md#statistical-model-and-targets) — the
Python data-generating processes (DGPs) implement the same constructions (with NumPy's pseudo-random number generator
(PRNG), so per-cell numbers differ slightly from the committed C++ results while reproducing the same qualitative
patterns).

---

## T-Rex+GVS specific concepts

Grouped selection introduces a few additional control parameters and types beyond the classical T-Rex selector
 (all accessed via `import trexselector as trex`, e.g. `trex.TRexGVSControlParameter`):

- **`trex.GVSType`** — the group-level base method:
  - `EN`: elastic-net-based grouped selection [[1]](#references),
  - `IEN`: informed (iterative) elastic-net-based grouped selection [[2]](#references).
- **`trex.ENSolverType`** (only applies to `GVSType.EN`) — which elastic-net solver drives the random experiments:
  - `TENET`: the base Gram-based elastic-net solver,
  - `TENET_AUG`: the augmented least absolute shrinkage and selection operator (LASSO) formulation of the
    elastic net.
- **`trex.LambdaSelectionMethod`** — how the elastic-net penalty $\lambda_2$ is chosen: `CV_1SE_CCD`
   (cross-validation, rule of one standard error (SE), via coordinate descent) or `CV_1SE_SVD` (via singular value
   decomposition (SVD)). The rules `CV_1SE_IEN_CCD` and `CV_1SE_TIK_SVD`, built for the IEN geometry, select
   $\lambda_2$ on the group structure and apply to `GVSType.IEN` only (on the IEN track the library maps
   `CV_1SE_CCD` to `CV_1SE_IEN_CCD`). All IEN rows use `CV_1SE_IEN_CCD`, which cross-validates $\lambda_2$ jointly
   with $\lambda_1$. Demos 01–07 raise its $\lambda_1$ support cap to `cv_dfmax_ratio = 1.0`
   ($n_{\mathrm{train}} = 180$ variables), because six of them have a true support of $s = 150$, above the default
   cap of $0.8\, n_{\mathrm{train}} = 144$ (see the Demo 01 Part 2 comparison). Demo 08 keeps the default.
   `CV_1SE_TIK_SVD` appears only in the Demo 01 Part 2 comparison, because the library refuses it once the groups
   hold $n_{\mathrm{train}}$ or more within-group contrasts.
- **`prior_groups`** — the 0-based per-column group-ID vector. When groups are known in advance (an "oracle" run),
   this vector is taken directly from the data-generating process. When groups must be discovered from the data,
   HAC clustering (with a `corr_max` correlation threshold and a chosen linkage method) is used to build
   `prior_groups` automatically — this is what Demo 08 compares directly against oracle groups.

The demos build the selector through `build_gvs_selector` in [gvs_sim_common.py](gvs_sim_common.py), which calls
the raw `TRexGVSSelector` binding. For EN the binding derives the solver (TENET or TENET_AUG) from `en_solver`, and
for IEN the helper sets `trex_control.solver_type` to TIENET_AUG explicitly, while the demos name the tuner
`CV_1SE_IEN_CCD`. The Python wrapper of the same name would run the IEN track with the native TIENET solver by
default, so the demos bypass it and run exactly the configuration of the C++ suite.

---

## Folder contents

```txt
trex_gvs/
  ├── README.md
  ├── trex_gvs_dgps.py                # all scenario DGPs (mirrors trex_gvs_dgps.hpp)
  ├── gvs_sim_common.py               # selector adapter, parallel MC runners,
  │                                   # table printers, .txt/.csv writers
  │                                   # (mirrors trex_gvs_simulation_utils.hpp)
  ├── trex_gvs_plt_utils.py           # shared plotting module (heatmaps / bars / line plots)
  ├── generate_corr_plots.sh          # renders dgp_corr_plots/ (one figure per DGP)
  ├── dgp_corr_plots/                 # design-correlation heatmaps per DGP
  ├── run_overnight.sh                # batch runner: all 8 demos, then the correlation heatmaps
  │
  ├── demo_trex_gvs_01_mc_sim_hastie_en_blocks/
  │   ├── demo_trex_gvs_01_mc_sim_hastie_en_blocks.py   # the demo script
  │   ├── README.md                    # scenario description and interpretation
  │   ├── generate_plots.sh            # renders this demo's figures from its CSVs
  │   └── simulation_results/
  │       ├── data/                    # .txt summaries + .csv tables
  │       └── plots/                   # .png + .pdf figures
  │
  │   ... demos 02–08 follow the same layout ...
  │
  └── demo_trex_gvs_08_mc_sim_block_bench/
      ├── demo_trex_gvs_08_mc_sim_block_bench.py
      ├── README.md
      ├── generate_plots.sh
      └── simulation_results/
          ├── data/
          └── plots/
```

Shared infrastructure:

- [trex_gvs_dgps.py](trex_gvs_dgps.py) — all scenario data-generating processes in one module:
  `dgp_hastie_snr`, `dgp_scattered_grouped_snr`, `dgp_mixed_blocks_snr`, `dgp_neg_traps_snr`,
  `dgp_t3_equi_blocks_snr`, `dgp_ar1_blocks_snr`, `dgp_arma_blocks_snr`, and `dgp_block_equicorr`.
  All are SNR-calibrated (`sigma_y = sqrt(var(signal) / snr)`, `ddof=1`, while the block benchmark uses
  the biased `ddof=0` denominator, matching the cpp DGP). Support / group indices are 0-based.
- [gvs_sim_common.py](gvs_sim_common.py) — the `build_gvs_selector` adapter (EN / EN_AUG / IEN, HAC or
  oracle `prior_groups`), the parallel MC runners `run_gvs_mc_trials` / `run_gvs_2d_sweep` /
  `run_block_mc` (`ProcessPoolExecutor`, mirroring the cpp Open Multi-Processing (OpenMP) loops), and the table
  printers and `.txt`/`.csv` writers whose output matches the cpp layout and schemas.
- [trex_gvs_plt_utils.py](trex_gvs_plt_utils.py) — the suite plotter, byte-identical to the cpp suite's
  module. It auto-detects the shape of the comma-separated values (CSV) tables and renders 2-D sweeps as TPR/FDR
  heatmap grids, the demo-01 $\lambda_2$/linkage studies as grouped bar charts, and the demo-08 block benchmark as
  TPR/FDR line plots (vs. SNR or, via `--vs rho`, vs. the within-block correlation).
- [generate_corr_plots.sh](generate_corr_plots.sh) — renders one design-correlation heatmap per DGP into
  [dgp_corr_plots/](dgp_corr_plots/) via the category-level plotter [../../corr_plt_utils.py](../../corr_plt_utils.py).
  These are embedded in the demo READMEs alongside the DGP description.

---

## Running a demo

Each demo is run directly with the Python interpreter (with `trexselector`, `numpy`, and
`pandas` installed — see the [Python examples README](../../../README.md)):

```bash
cd Python/trex_selector_methods/ad/trex_gvs/demo_trex_gvs_01_mc_sim_hastie_en_blocks
python demo_trex_gvs_01_mc_sim_hastie_en_blocks.py       # full run (hours)
python demo_trex_gvs_01_mc_sim_hastie_en_blocks.py 8     # with 8 worker processes
```

Notes:

- Every demo writes a human-readable `.txt` summary and a tidy `.csv` table into its local
  `simulation_results/data/` folder, using the same file names and CSV schemas as the C++ suite
  (so either suite's `trex_gvs_plt_utils.py` renders the same figures from either side's CSVs).
- Each demo inserts its own directory and the parent suite directory onto `sys.path`, so the shared
  `trex_gvs_dgps` / `gvs_sim_common` modules resolve from the nested location and are re-importable
  by the `spawn`-launched Monte Carlo workers — run the demos **as files**, not piped via `python -`.
- Monte Carlo trials are parallelized with `concurrent.futures.ProcessPoolExecutor`. The optional
  first command-line interface (CLI) argument sets the number of worker processes. Without it the demos use
  `TREX_NUM_WORKERS` workers (default 8, and `0` uses every usable core).
- The committed settings match the C++ suite (200 MC trials per grid point) and take hours per demo.
  For a quick smoke test, temporarily reduce `num_MC` and the sweep grids at the top of a script.
- Seeding mirrors the cpp MC loops. Trial i uses data seed `base_seed + i`, which is also passed as
  the selector seed (and as `cv_seed` in demos 01–07), while demo 08 staggers seeds per grid cell.
- Indices are **0-based** (Python convention), while the R counterparts are 1-based.

### Figures

Each demo folder has a `generate_plots.sh` that renders its figures from the saved CSVs into
`simulation_results/plots/` using the shared [trex_gvs_plt_utils.py](trex_gvs_plt_utils.py) and the
repo-root `.venv` (matplotlib + pandas + numpy):

```bash
cd demo_trex_gvs_01_mc_sim_hastie_en_blocks
./generate_plots.sh                 # this demo's full figure set (PNG + PDF)
./generate_plots.sh --formats png   # extra args pass through to the plotter
```

The design-correlation heatmaps embedded in each demo README's DGP section are separate — they
depend only on the DGP, not on a simulation run, and are regenerated suite-wide with
[generate_corr_plots.sh](generate_corr_plots.sh).

---

## References

1. Machkour, J., Muma, M., & Palomar, D. P., "False Discovery Rate Control for Grouped Variable Selection
   in High-Dimensional Linear Models using the T-Knock Filter.", European Signal Processing Conference (EUSIPCO), 2022,
    pp. 892–896, European Association for Signal Processing (EURASIP).
    [DOI-Link](https://doi.org/10.23919/EUSIPCO55093.2022.9909883)
2. Machkour, J., Muma, M., & Palomar, D. P., "The Informed Elastic Net for Fast Grouped Variable Selection and
   FDR Control in Genomics Research.", Workshop on Computational Advances in Multi-Sensor Adaptive Processing (CAMSAP),
    2023, pp. 466–470, IEEE.
    [DOI-Link](https://doi.org/10.1109/CAMSAP58249.2023.10403489)
3. Dai, R., & Barber, R. F., "The knockoff filter for FDR control in group-sparse and multitask regression."
   Proceedings of the 33rd International Conference on Machine Learning (ICML), vol. 48, pp. 1851–1859, 2016.
   [DOI-Link](http://proceedings.mlr.press/v48/daia16.pdf)
