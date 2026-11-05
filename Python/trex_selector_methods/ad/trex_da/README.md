# DA-TRex: Dependency-Aware T-Rex Demonstration Suite — Python Demos

## Overview

This folder contains the Python example programs for the **Dependency-Aware T-Rex (DA-TRex)**, the Terminating Random
Experiments (T-Rex) selector extended with a **dependency-aware (DA) correction** for correlated/dependent design
matrices. Where the classical T-Rex selector (see [../trex_core/](../trex_core/README.md)) assumes exchangeable, weakly
correlated predictors, DA-TRex explicitly models the correlation structure among nearby or grouped variables so that a
single true signal does not get "smeared" into false discoveries among its correlated neighbors.

The demos mirror the authoritative C++ suite in
[cpp/trex_selector_methods/ad/trex_da/](../../../../cpp/trex_selector_methods/ad/trex_da/README.md)
one-to-one — same data-generating processes (DGPs), control parameters, sweep axes, console summaries, and result-file
naming. They cover four dependency structures and their matching DA methods: **first-order autoregressive (AR(1))**
(Toeplitz decay), **equicorrelation (EQUI)**, **banded nearest-neighbor (NN)**, and **binary-tree (BT)**, a
dendrogram-based hierarchical block detection via hierarchical agglomerative clustering (HAC). In addition, the
**PRIOR_GROUPS** constrained sub-clustering method covers a priori known groups (AKG).

---

## What this folder covers

- **AR(1) Toeplitz correlation** (Demo 01): the classical case DA-TRex was
  designed for — correlation decays geometrically with column distance.
- **Block-diagonal AR(1) designs**, with and without appended white-noise
  columns (Demos 03 and 02), using BT aggregation with a sweep over HAC
  linkage methods.
- **Heavy-tailed (Student-t) block designs** (Demos 04-05): testing DA-TRex
  robustness when predictors and/or noise depart from Gaussianity.
- **Multi-level nested group structures** (Demo 06): three-level hierarchical
  latent factors with a non-exchangeable Toeplitz leaf layer, driven through
  the `prior_groups` merge constraints.
- **Banded/nearest-neighbor correlation** plus a **method-mismatch stress
  test** (Demo 07: NN data, and the NN correction applied to AR(1) data).
- **Equicorrelation** (Demo 08): compound-symmetry correlation via a single
  shared latent factor.

---

## DA-TRex specific concepts

- **`trex.DAMethod`** — the dependency-correction strategy: `AR1`, `EQUI`, `NN`, `BT` (plus `PRIOR_GROUPS`, engaged
  automatically whenever `da_control.prior_groups` is non-empty). Each targets the matching correlation structure, and
  DA-TRex uses this structure to widen or narrow the "correction window" around each candidate variable when
  accumulating dummy-based voting evidence.
- **BT linkage method** (`LinkageMethod` from `trexselector.ml_methods.clustering`) — for the `BT` method,
  variables are grouped via hierarchical agglomerative clustering, and several demos sweep **Single**, **Complete**, and
  **Average** linkage to study sensitivity to this choice.
- **Prior groups as merge constraints**: the `PRIOR_GROUPS` method sub-clusters *within* the finest
  common refinement of the supplied label vectors and calibrates over a data-driven ascending rho grid anchored at the
  conservative rho = 1 singleton point. Negative labels are rejected.
- **Support placement policies**: `CappedSpread` (deterministic, evenly spaced
  with a capped gap), `Random` (redrawn per trial), `OnePerBlock` (one active
  variable per block — the natural choice for block-structured DGPs). See
  `da_sim_common.make_support_*`.
- **Base T-Rex comparison rows**: Demos 01, 07, and 08 additionally run the
  *classical* T-Rex selector (no DA correction) side by side with DA-TRex on
  the same correlated data (`TREX: <solver>` rows).
- **Exchangeable tie-breaking for greedy solvers** (`exch_tie_alpha`): `default_solvers()` sets
  `exch_tie_alpha = 0.25` for TAFS/TOMP so that statistically indistinguishable top candidates within highly correlated
  clusters are picked uniformly at random per random experiment — restoring the occurrence spread the DA deflation's
  false discovery rate (FDR) control relies on. TLARS and TLARS_STAG keep `0` (path solvers spread naturally). See
  the `exch_tie_alpha` entry in HISTORY.md of the TRexSelector repository.
- **TAFS' adaptive forward-selection (AFS) correlation parameter**: `default_solvers()` runs TAFS with
  `rho_afs = 0.3`, while TLARS, TLARS_STAG and TOMP take `0` (unused).
- **TLARS_STAG, TLARS with the stagnation stop**: `default_solvers()` runs TLARS without the T-loop stagnation stop,
  because its monotone path needs none, and in correlated designs its T-loop then often runs far, up to the cap of
  $T = 150$ (Demo 08). The fourth pool entry TLARS_STAG is the same TLARS with the stop on (window 7), on the same
  data. In every demo it matches TLARS within Monte Carlo noise (averaged over a demo's cells, the true positive rate
  (TPR) changes by at most $0.003$, and the FDR rises by at most $0.001$ and falls by at most $0.009$). It breaks the
  target in about as many cells as TLARS (56 against 57 DA cells over all demos), and the cells where only one of the
  two crosses the target lie within $0.02$ of it. Below the correction-window boundary of Demo 01 it raises fewer false
  alarms than TLARS. Its T-loop stops after at most $8.0$ steps on average in the DA rows. The `TREX: TLARS_STAG` base
  rows of Demos 01, 07 and 08 match `TREX: TLARS` in the same way. The figures draw TLARS_STAG dashed in TLARS' colour.

---

## Demo descriptions

Each row lists the DA method, the DGP generator and the key parameters, including the sweeps over the signal-to-noise
ratio (SNR) and, in Demos 04–05, over the target FDR (tFDR):

| # | Name | DA Method | DGP | Key parameters |
|---|------|-----------|-----|-----------------|
| **01** | [AR(1)](demo_trex_da_01_mc_sim_ar1/) | `AR1` | `dgp_ar1_snr` | n=300, p=1,000, s=10, rho=0.7, with SNR & rho sweeps + 2D gap x rho |
| **02** | [BT + white noise](demo_trex_da_02_mc_sim_ar1_blocks_plus_white/) | `BT` | `dgp_ar1_block_white_snr` | n=300, p_total=1,000, M=5, Q=5 (p_ar=25), rho=0.7, with linkage x {SNR, rho, Q, M} sweeps |
| **03** | [BT](demo_trex_da_03_mc_sim_ar1_blocks/) | `BT` | `dgp_ar1_block_snr` | n=150, M=5, Q=5 (p=25, s=5), rho=0.7, with linkage x {SNR, rho, Q, M} sweeps |
| **04** | [BT, heavy-tailed](demo_trex_da_04_mc_sim_ht_blocks/) | `BT` | `dgp_block_toeplitz_hvt_snr` | n=150, M=5, Q=5, rho=0.8, nu=3, with Gauss/Heavy x linkage x {SNR, rho, Q, M, tFDR} |
| **05** | [BT, heavy-tailed + white](demo_trex_da_05_mc_sim_ht_blocks_plus_white/) | `BT` | `dgp_ht_block_white_snr` | n=150, p_total=500, M=5, Q=5, rho=0.8, nu=3, with Gauss/Heavy x linkage x {SNR, rho, Q, M, tFDR} |
| **06** | [Prior groups (AKG)](demo_trex_da_06_mc_sim_groups/) | `PRIOR_GROUPS` (a priori known groups) | `dgp_groups_toeplitz_leaf` | n=300, p=1,000, s=10, with 3-level groups {10, 50, 250}, rho={0.55, 0.25, 0.10}, phi=0.5 |
| **07** | [NN + mismatch test](demo_trex_da_07_mc_sim_nn/) | `NN` | `dgp_nn_snr`, `dgp_ar1_snr` | n=300, p=1,000, kappa=3, rho=0.7, with SNR sweep + 2D kappa x rho + 2D SNR x rho on AR(1) data (NN misspecified) |
| **08** | [Equi](demo_trex_da_08_mc_sim_equi_and_bt/) | `EQUI` | `dgp_equi_snr` | n=300, p=1,000, s=10, with rho sweep x SNR in {0.5, 1, 2, 5} |

---

## Folder contents

```txt
trex_da/
  ├── README.md
  ├── dgp_generators.py       ┐  suite-level shared modules
  ├── da_sim_common.py        ┘  (imported by every demo)
  ├── trex_da_plt_utils.py       <- shared plotting module (all DA demos)
  ├── generate_corr_plots.sh     <- renders dgp_corr_plots/ (one per DGP)
  ├── dgp_corr_plots/            <- design-correlation heatmaps per DGP
  ├── demo_trex_da_01_mc_sim_ar1/
  ├── demo_trex_da_02_mc_sim_ar1_blocks_plus_white/
  ├── demo_trex_da_03_mc_sim_ar1_blocks/
  ├── demo_trex_da_04_mc_sim_ht_blocks/
  ├── demo_trex_da_05_mc_sim_ht_blocks_plus_white/
  ├── demo_trex_da_06_mc_sim_groups/
  ├── demo_trex_da_07_mc_sim_nn/
  └── demo_trex_da_08_mc_sim_equi_and_bt/
```

Each demo subfolder contains its `.py` script (named after the folder), a `README.md`, a `generate_plots.sh`, and a
`simulation_results/` folder split exactly like the cpp suite into `data/` — the `.txt` (pretty-printed tables) and
`.csv` (tidy long format) files written by the demo, following the cpp naming pattern
`da_trex_mc_{scenario_tag}.{txt,csv}` — and `plots/`, the figures rendered from those comma-separated values (CSV)
files.

Shared infrastructure:

- [dgp_generators.py](dgp_generators.py) — the DA data-generating processes, all SNR-calibrated with 0-based support
  and mirroring `cpp/.../dgp_generators.hpp` one generator per cpp DGP. They cover AR(1), equicorrelated, NN or
  moving-average (MA(kappa)), AR(1) blocks with and without a white block, heavy-tailed blocks with and without a
  heavy-tailed white block, and prior groups with Toeplitz leaves.
- [da_sim_common.py](da_sim_common.py) — support generators, the default solver set (TLARS/TAFS/TOMP/TLARS_STAG with
  per-solver `exch_tie_alpha` / `rho_afs` / stagnation), parallel Monte-Carlo (MC) runners for DA-TRex and base
  T-Rex, generic parameter sweeps, and the console / `.txt` / `.csv` writers, mirroring
  `cpp/.../trex_da_sim_common.hpp`.
- [trex_da_plt_utils.py](trex_da_plt_utils.py) — the shared results plotter (identical to
  `cpp/.../trex_da_plt_utils.py`), driven by each demo's `generate_plots.sh`.
- [generate_corr_plots.sh](generate_corr_plots.sh) — renders one design-correlation heatmap per DGP into
  [dgp_corr_plots/](dgp_corr_plots/) via the category-level plotter
  [../../corr_plt_utils.py](../../corr_plt_utils.py), and the demo READMEs embed these heatmaps alongside the DGP
  description.

---

## Running

```bash
python Python/trex_selector_methods/ad/trex_da/demo_trex_da_01_mc_sim_ar1/demo_trex_da_01_mc_sim_ar1.py
```

Each demo exposes `RUN_PART_*` flags at the top. All parts are enabled by default (mirroring the cpp `if (true)`
toggles), and each runs `NUM_MC = 200` Monte Carlo trials per grid point. Reduce `NUM_MC` to preview. Each demo inserts
both its own directory and its parent suite directory onto `sys.path`, so the shared `dgp_generators` /
`da_sim_common` modules resolve from the nested location and are importable by the spawned workers. The demos must
therefore be run as files, not piped via `python -`. Monte Carlo trials run in parallel via
`concurrent.futures.ProcessPoolExecutor` (one persistent pool per demo run with `TREX_NUM_WORKERS` worker processes,
8 by default and every available core for `0`).

Seed spaces mirror the cpp suite: DGP seed = base + grid offset + trial,
random support seed = trial seed + 500000, selector seed = -1 (hardware
entropy, required for valid MC FDR estimates).

Indices are **0-based** (Python convention), while the R counterparts are 1-based.

---

## Plotting the results

Each demo has a `generate_plots.sh` that renders its figures from the committed
`simulation_results/data/*.csv` into `simulation_results/plots/` (as `png` /
`pdf`, plus interactive `html` with `--plotly`), via the suite-level plotting module
[trex_da_plt_utils.py](trex_da_plt_utils.py). The scripts use the repo-root
`.venv` (`pip install matplotlib pandas plotly`) and are byte-identical to
their cpp counterparts, so both suites render the same figure set from their
own data:

```bash
cd TRexSelector_Examples/Python/trex_selector_methods/ad/trex_da
./demo_trex_da_01_mc_sim_ar1/generate_plots.sh          # this demo's full figure set
./demo_trex_da_03_mc_sim_ar1_blocks/generate_plots.sh
# ... one per demo; pass a single CSV to plot just that file, or extra flags
# (e.g. --formats png, --bands) which pass straight through to the plotter.
```

All the tidy demos share the CSV schema `solver,metric,<SWEEP>,value`, where
`<SWEEP>` is whatever that scenario sweeps (`SNR`, `Rho`, `M`, `Q`, `tFDR`, or a
`Linkage` index) — the plotter auto-detects the sweep column and picks the
matching x-axis. It offers four figure patterns:

- **per-CSV FDR and TPR overview** — side-by-side FDR | TPR vs the swept quantity,
  one line per method (`TREX-DA:` TLARS/TAFS/TOMP/TLARS_STAG with TLARS_STAG dashed, and the classical `TREX:`
  baseline in Demos 01, 07 and 08), with the `tFDR` target drawn as a reference
  (a horizontal line, or the identity line for a `tFDR` sweep),
- **comparison grid** (`grid` mode) — a 2×N grid sharing FDR/TPR scales across
  columns, used for the HAC-linkage sensitivity study (Single/Complete/Average
  columns, Demos 02–05), the Gaussian-vs-heavy-tailed scenario contrast
  (Demos 04–05), and the CappedSpread-vs-Random support contrast (Demos 01
  and 07),
- **gap × ρ study** (`gaprho` mode, Demo 01) — `mean_TPP` / `mean_FDP` heatmaps
  over the (gap, ρ) grid with the DA+AR1 correction-window (κ-boundary) overlaid,
- **generic 2D sweep** (`sweep2d` mode, Demo 07) — per-solver TPR/FDR heatmaps
  over a 2D grid (κ × ρ on NN data, and SNR × ρ for the AR(1) method-mismatch
  study), one figure per support placement, with FDR cells above the `tFDR`
  target outlined in cyan.

---

## What to expect

Every demo ships with real committed data and figures, and the pattern matches the C++ reference throughout
(independent random number generator (RNG) streams, so cells agree up to Monte-Carlo noise). All eight demos run on
library commit `09debc6` (200 Monte Carlo trials per cell, 8 workers, about 23.5 h in total). Of the 9,344 FDR and TPR
cells, 29 (0.31 %) lie more than three Monte Carlo standard errors from C++ and 17 (0.18 %) from R, the rate noise
alone produces. DA-TRex's FDR stays far closer to controlled than the classical (no-DA) T-Rex selector on the same
correlated data — in Demo 01 at $\mathrm{SNR}=2$, DA-TRex realizes $\mathrm{FDR}\approx 0.03\text{–}0.04$ against base
T-Rex TLARS's $\approx 0.27\text{–}0.28$ — at some cost in TPR, largest around $\mathrm{SNR} = 1$ and under `Random`
support. The greedy DA solvers (TAFS/TOMP) are FDR-controlled across the full $\rho$ grid and both support placements,
which the C++ README traces to the exchangeable-tie fix, and they generally outperform DA-TLARS on block designs.
TLARS_STAG tracks DA-TLARS in FDR and TPR in every demo while cutting its long T-loops short. The demo READMEs document
the known rough edges. TLARS, and TLARS_STAG with it, breaks the target at $\rho = 0.9$ on block designs (Demos 03/04)
and on the multi-level prior-groups DGP (Demo 06, flat in SNR — consistent with a structural property of the symmetric
nearest-partner deflation under least-angle regression (LARS) type occurrence profiles). Heavy-tailed noise raises the
FDR in Demo 05 (mean over the DA cells $0.115$ against $0.072$ with Gaussian noise) without costing power, while in
Demo 04 the two noise models give the same averages. Against the Gaussian Demo 03, the heavy-tailed predictors of
Demo 04 cost power more than FDR control. The NN correction breaks down when applied to AR(1) data at $\rho \geq 0.8$
(Demo 07), and equicorrelation suppresses the DA selector entirely from $\rho = 0.025$ on (Demo 08).

---

## Start here

1. **Demo 01 — AR(1)**: the foundational scenario, including a direct
   DA-TRex vs. base-T-Rex comparison and a "gap x rho" sweep showing when
   correlated neighbors start hurting the classical selector.
2. **Demo 03 — BT on block AR(1)**: introduces the `BT` method and HAC
   linkage sweep on a clean block-diagonal design.
3. **Demo 06 — Prior groups**: the most structurally rich DGP (three nested
   levels of correlation), driven through the constrained sub-clustering.

Cross-check against the C++ suite (committed reference data and figures) in
[cpp/trex_selector_methods/ad/trex_da/](../../../../cpp/trex_selector_methods/ad/trex_da/README.md)
and the R suite in `R/trex_selector_methods/ad/trex_da/`. Note that the R suite
keeps its own legacy numbering (e.g. the NN demos are `03`/`03b` there), so
match demos by name, not number.
