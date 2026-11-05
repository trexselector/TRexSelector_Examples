# DA-TRex: Dependency-Aware T-Rex Demonstration Suite

## Overview

This folder contains C++ example programs for **DA-TRex**, the Terminating Random Experiments (T-Rex) Selector extended
with **dependency-aware (DA) correction** for correlated/dependent design matrices. Where the classical T-Rex selector
(see [../trex_core/](../trex_core/README.md)) assumes exchangeable, weakly correlated predictors, DA-TRex explicitly
models the correlation structure among nearby or grouped variables so that a single true signal does not get "smeared"
into false discoveries among its correlated neighbors.

The demos cover four dependency structures and their matching DA methods: **first-order autoregressive (AR(1))**
(Toeplitz decay), **equicorrelation (EQUI)**, **banded nearest-neighbor (NN)**, and **binary-tree (BT)**, a
dendrogram-based hierarchical block detection via hierarchical agglomerative clustering (HAC).

---

## What this folder covers

- **AR(1) Toeplitz correlation** (Demo 01): the classical case DA-TRex was designed for — correlation decays
  geometrically with column distance.
- **Block-diagonal AR(1) designs**, with and without appended white-noise columns (Demos 03 and 02), using BT
  aggregation with a sweep over HAC linkage methods.
- **Heavy-tailed (Student-t) block designs** (Demos 04–05): testing DA-TRex robustness when predictors and/or noise
  depart from Gaussianity.
- **Multi-level nested group structures** (Demo 06): three-level hierarchical latent factors with a non-exchangeable
  Toeplitz leaf layer.
- **Banded/nearest-neighbor correlation** plus a **method-mismatch stress test** (Demo 07: NN data, and the NN
  correction applied to AR(1) data).
- **Equicorrelation** (Demo 08): compound-symmetry correlation via a single shared latent factor.

A correctness diagnostic comparing the BT clustering pipeline step-by-step against an R reference implementation lives
in the TRexSelector library test suite (`TRexSelector/cpp/tests/validation/trex_selector_methods/ad/trex_da/`), not in
this demo suite.

---

## Statistical model and notation

DA-TRex uses the same Gaussian linear model as the classical T-Rex selector,

```math
\boldsymbol{y} = \boldsymbol{X}\boldsymbol{\beta} + \boldsymbol{\varepsilon},
\qquad
\boldsymbol{\varepsilon} \sim \mathcal{N}(0, \sigma^2 \boldsymbol{I}_n),
```

but $\boldsymbol{X}$'s columns are no longer (close to) exchangeable — they follow one of several explicit dependency
structures:

- **AR(1) / Toeplitz**: $\Sigma_{jk} = \rho^{|j-k|}$, built via the recursion
  $X_{i,j} = \rho X_{i,j-1} + \sqrt{1-\rho^2}\,\eta_{i,j}$.
- **Equicorrelated**: $X_{i,j} = \sqrt{\rho}\, f_i + \sqrt{1-\rho}\,\eta_{i,j}$, a single shared latent factor $f_i$
  across all columns.
- **Banded / moving-average (MA($\kappa$))**: $X_{i,j} = \sum_{l=0}^{\kappa} \theta_l\, \eta_{i,j+l}$, with
  $\theta_l \propto \rho^l$ normalized to unit variance — only the $\kappa$ nearest neighboring columns are correlated.
- **Binary-tree / two-level block**, requiring $\rho_{\text{within}} > \rho_{\text{between}}$:

```math
X_{i,j} = \sqrt{\rho_{\text{between}}}\, f_0 + \sqrt{\rho_{\text{within}} - \rho_{\text{between}}}\, f_{\text{block}(j)} + \sqrt{1-\rho_{\text{within}}}\,\eta_{i,j}
```

- **Multi-level nested groups**:
  $X_{i,j} = \sum_{x} \sqrt{\rho_x - \rho_{x+1}}\, f_{g_x(j)} + \sqrt{1-\rho_0}\,\eta_{i,j}$, with strictly decreasing
  $\rho$ levels from fine to coarse groupings.

Signal-to-noise ratio (SNR) follows the same convention as elsewhere:
$\mathrm{SNR} = \mathrm{Var}(\boldsymbol{X}\boldsymbol{\beta})/\mathrm{Var}(\boldsymbol{\varepsilon})$.

---

## DA-TRex specific concepts

- **`DAMethod`** — the dependency-correction strategy: `AR1`, `EQUI`, `NN`, `BT`. Each targets the matching correlation
  structure above, and DA-TRex uses this structure to widen or narrow the "correction window" around each candidate
  variable when accumulating dummy-based voting evidence.
- **BT linkage method** (`hac::LinkageMethod`) — for the `BT` method, variables are grouped via hierarchical
  agglomerative clustering, and several demos sweep **Single**, **Complete**, and **Average** linkage to study
  sensitivity to this choice.
- **Support placement policies** (`SupportPolicy`): `CappedSpread` (deterministic, evenly spaced with a capped gap —
  used to probe how spacing interacts with a correction window), `Random` (redrawn per trial), `Clustered` (stochastic
  small groups), `OnePerBlock` (one active variable per block — the natural choice for block-structured data-generating
  processes (DGPs)).
- **Base T-Rex comparison row**: several demos additionally run the *classical* T-Rex selector (no DA correction) side
  by side with DA-TRex on the same correlated data, to directly show the benefit for false discovery rate (FDR) control,
  or the cost in true positive rate (TPR), of the dependency-aware correction.
- **Exchangeable tie-breaking for greedy solvers** (`exch_tie_alpha`): the DA deflation
  $\delta_j = 2 - \min_k |\Phi_j - \Phi_k|$ suppresses collinear shadows only when exchangeable cluster members end up
  with *similar occurrence values within a trial*. Path solvers of the least-angle regression (LARS) type produce that
  spread naturally, while greedy solvers (TOMP/TAFS) are winner-take-all — the in-sample cluster winner is a
  deterministic function of $(X, y)$, so the deflation degenerates to the identity and shadows that beat their active
  in-sample survive as false positives (realized FDR $\approx$ the intrinsic shadow-win rate, e.g. $\approx 34\,\%$ per
  cluster at $\rho = 0.9$, above the target FDR (tFDR) of $\mathrm{tFDR} = 0.2$). `default_solvers()` therefore sets
  `exch_tie_alpha = 0.25` for TAFS/TOMP, so statistically indistinguishable top candidates within highly correlated
  clusters are picked uniformly at random per random experiment, which restores the occurrence spread the DA correction
  relies on. TLARS keeps `0` (unneeded). See the `exch_tie_alpha` entry in `HISTORY.md` of the TRexSelector repository
  for details and validation.
- **TAFS' adaptive forward-selection (AFS) correlation parameter** (`rho_afs = 0.3`): `default_solvers()` runs TAFS with
  `rho_afs = 0.3`, the correlation threshold of its AFS step, while TLARS and TOMP take `0` (unused). Because it is a
  defining part of the method rather than an incidental tuning knob, the plotting module annotates every legend entry as
  `TAFS (rho = 0.3)` (`TAFS_RHO` in `trex_da_plt_utils.py` — keep it in sync with `default_solvers()` in
  `trex_da_sim_common.hpp`). The comma-separated values (CSV) files keep the raw `TREX-DA...: TAFS` label the demo
  binaries write, so the annotation is applied at plot time and needs no re-run.
- **TLARS_STAG, TLARS with the stagnation stop**: `default_solvers()` runs TLARS without the T-loop stagnation stop,
  because its monotone path needs none, and in correlated designs its T-loop then often runs far, up to the cap of
  $T = 150$. The fourth pool entry TLARS_STAG is the same TLARS with the stop on (window 7), on the same data and seeds.
  In every demo it matches TLARS within Monte Carlo noise (mean TPR change at most $0.002$, mean FDR change at most
  $+0.007$). It breaks the target about as often, e.g. in demo 04 in $27$ against $26$ DA cells above the target and
  in $8$ against $11$ cells more than two standard errors above it. Its T-loop stops after at most $8$ steps in the DA
  rows. The figures draw it dashed in TLARS' colour. Its numbers were computed in a separate run with
  `TREX_SOLVERS=TLARS_STAG` and merged into the tables by [merge_result_files.py](merge_result_files.py).

---

## Statistical targets

Same FDR/TPR definitions as throughout this project:

```math
\mathrm{FDR}
=
\mathbb{E}
\left[
\frac{|\widehat{\mathcal{A}} \setminus \mathcal{A}|}
{\max\{1, |\widehat{\mathcal{A}}|\}}
\right],
\qquad
\mathrm{TPR}
=
\mathbb{E}
\left[
\frac{|\mathcal{A} \cap \widehat{\mathcal{A}}|}
{\max\{1, |\mathcal{A}|\}}
\right].
```

The TRexSelector library test suite (`TRexSelector/cpp/tests/validation/trex_selector_methods/ad/trex_da/`) holds a
**correctness diagnostic** — not a statistical-power study — that compares C++ and R dendrogram heights, group
memberships, and selection results on one identical dataset to check they agree, rather than measuring FDR/TPR over many
trials.

---

## Start here

1. **Demo 01 — AR(1)**: the foundational scenario, including a direct DA-TRex vs. base-T-Rex comparison and a "gap ×
   rho" sweep showing when correlated neighbors start hurting the classical selector.
2. **Demo 03 — BT on block AR(1)**: introduces the `BT` method and HAC linkage sweep on a clean block-diagonal design.
3. **Demo 06 — Prior groups**: the most structurally rich DGP (three nested levels of correlation).
4. **The BT diagnostic** in the library test suite
   (`TRexSelector/cpp/tests/validation/trex_selector_methods/ad/trex_da/`) — if you want to verify the BT clustering
   pipeline itself against an R reference on one fixed dataset.

---

## Demo descriptions

| # | Name | DA Method | DGP | Key parameters | Status |
|---|------|-----------|-----|-----------------|--------|
| **01** | AR(1) | `AR1` | `dgp_ar1` | $n=300,p=1{,}000,s=10,\rho=0.7$, with SNR & $\rho$ sweeps + 2D gap × $\rho$ | Real data |
| **02** | BT + white noise | `BT` | `dgp_ar1_block_white` | $n=300,p_{\text{total}}=1{,}000,M=5,Q=5(p_{ar}=25),\rho=0.7$, with linkage × {SNR, $\rho$,Q,M} sweeps | Real data |
| **03** | BT | `BT` | `dgp_ar1_block` | $n=150,M=5,Q=5(p=25,s=5),\rho=0.7$, with linkage × {SNR, $\rho$,Q,M} sweeps | Real data |
| **04** | BT, heavy-tailed | `BT` | `dgp_block_toeplitz_hvt` | $n=150,M=5,Q=5,\rho=0.8,\nu=3$, with Gauss/Heavy × linkage × {SNR, $\rho$, Q, M,tFDR} | Real data |
| **05** | BT, heavy-tailed + white | `BT` | `dgp_ht_block_white` | $n=150,p_{\text{total}}=500,M=5,Q=5,\rho=0.8,\nu=3$, with Gauss/Heavy × linkage × {SNR, $\rho$, Q, M,tFDR} | Real data |
| **06** | Prior groups (AKG) | `PRIOR_GROUPS` (a priori known groups) | `dgp_groups_toeplitz_leaf` | $n=300,p=1{,}000,s=10$, with 3-level groups $\lbrace10,50,250\rbrace$, $\rho=\lbrace0.55,0.25,0.10\rbrace$, $\phi=0.5$ | Real data |
| **07** | NN + mismatch test | `NN` | `dgp_nn`, `dgp_ar1` | $n=300,p=1{,}000,\kappa=3,\rho=0.7$, with SNR sweep + 2D $\kappa\times\rho$ + 2D SNR × $\rho$ on AR(1) data (NN misspecified) | Real data |
| **08** | Equi | `EQUI` | `dgp_equi` | $n=300,p=1{,}000,s=10$, with $\rho$ sweep × SNR $\in\lbrace0.5,1,2,5\rbrace$ | Real data |

(A correctness diagnostic previously numbered Demo 09 now lives in the TRexSelector library test suite as
`TRexSelector/cpp/tests/validation/trex_selector_methods/ad/trex_da/validation_trex_da_01_bt_dendro_diag/`.)

---

## Folder contents

```txt
trex_da/
  ├── README.md
  ├── CMakeLists.txt
  ├── dgp_generators.hpp
  ├── trex_da_sim_common.hpp            <- shared MC / output helpers (C++)
  ├── trex_da_plt_utils.py              <- shared plotting module (all DA demos)
  ├── merge_result_files.py             <- adds a TREX_SOLVERS selection run to the full tables
  ├── demo_trex_da_01_mc_sim_ar1/
  ├── demo_trex_da_02_mc_sim_ar1_blocks_plus_white/
  ├── demo_trex_da_03_mc_sim_ar1_blocks/
  ├── demo_trex_da_04_mc_sim_ht_blocks/
  ├── demo_trex_da_05_mc_sim_ht_blocks_plus_white/
  ├── demo_trex_da_06_mc_sim_groups/
  ├── demo_trex_da_07_mc_sim_nn/
  └── demo_trex_da_08_mc_sim_equi_and_bt/
```

(Each demo subfolder above also contains its own `.cpp` file, `README.md`, a
`generate_plots.sh`, and `simulation_results/` split into `data/` — the `.txt` /
`.csv` written by the demo binary — and `plots/` — the figures rendered from
those CSVs.)

---

## What to expect

Every demo now ships with real committed data, and a consistent pattern emerges. DA-TRex's FDR stays much closer to
well-controlled than the classical (no-DA) T-Rex selector on the same correlated data — for example, in Demo 01 at
$\mathrm{SNR}=2.0$, DA-TRex achieves $\mathrm{FDR}\approx0.04$ vs. base T-Rex's $\mathrm{FDR}\approx0.29$ at the same
TPR ballpark — at some cost in TPR, largest around $\mathrm{SNR} = 1$ and under `Random` support. Since the library
added exchangeable tie-breaking (see *DA-TRex specific concepts* above), Demo 01's DA rows are FDR-controlled for
**all four solvers across the full $\rho$ grid and both support placements** — including the previously violating
greedy-solver cells at $\rho=0.9$ with power (e.g. for gap=100: TAFS $0.24\to0.15$, TOMP $0.21\to0.12$, and for Random
support: TAFS $0.26\to0.18$, TOMP $0.27\to0.15$), where the greedy DA solvers now also retain *more* power than
DA-TLARS. Heavy-tailed predictors cost power more than FDR control (Demo 04 against the Gaussian Demo 03 at the same
$\rho$). Heavy-tailed noise raises the FDR in Demo 05 (mean over the DA cells $0.118$ against $0.074$ with Gaussian
noise) without costing power, while in Demo 04 the two noise models give the same averages ($0.094$ / $0.096$). The
multi-level grouped design of Demo 06 pushes DA-TLARS above the $\mathrm{tFDR}$ target (FDR $0.33\text{–}0.38$ from
$\mathrm{SNR}=0.5$ on), while TAFS and TOMP stay below it.

---

## Building the demos

```bash
cd TRexSelector_Examples/cpp
cmake -S . -B build/debug -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_PREFIX_PATH="<path-to-TRexSelector>/cpp/install"
cmake --build build/debug
```

---

## Running a demo

```bash
./build/debug/bin/trex_selector_methods/ad/trex_da/demo_trex_da_01_mc_sim_ar1/demo_trex_da_01_mc_sim_ar1
```

(Each demo's binary mirrors the source tree:
`./build/debug/bin/trex_selector_methods/ad/trex_da/<demo_folder>/<demo_name>`.)

Most demos write both `.txt` and `.csv` files per scenario into their local `simulation_results/` folder, following the
naming pattern `da_trex_mc_{scenario_tag}.{txt,csv}`.

---

## Plotting the results

Each demo has a `generate_plots.sh` that renders its figures from the committed
`simulation_results/data/*.csv` into `simulation_results/plots/` (as
`png` / `pdf`, plus interactive `html` with `--plotly`), via the suite-level plotting module
[trex_da_plt_utils.py](trex_da_plt_utils.py). The scripts use the repo-root
`.venv` (`pip install matplotlib pandas plotly`):

```bash
cd TRexSelector_Examples/cpp/trex_selector_methods/ad/trex_da
./demo_trex_da_01_mc_sim_ar1/generate_plots.sh          # this demo's full figure set
./demo_trex_da_03_mc_sim_ar1_blocks/generate_plots.sh
# ... one per demo; pass a single CSV to plot just that file, or extra flags
# (e.g. --formats png, --bands) which pass straight through to the plotter.
```

All the tidy demos share the CSV schema `solver,metric,<SWEEP>,value`, where
`<SWEEP>` is whatever that scenario sweeps (`SNR`, `Rho`, `M`, `Q`, `tFDR`, or a
`Linkage` index) — the plotter auto-detects the sweep column and picks the
matching x-axis. It offers four figure patterns:

- **per-CSV FDR/TPR overview** — side-by-side FDR | TPR vs the swept quantity,
  one line per method (`TREX-DA:` TLARS/TAFS/TOMP/TLARS_STAG, and the classical `TREX:`
  baseline in Demos 01, 07 and 08), with the `tFDR` target drawn as a reference (a
  horizontal line, or the identity line for a `tFDR` sweep),
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

## Notes for new users

- Start with **Demo 01** to see the core AR(1) scenario and the DA-vs-base-T-Rex comparison.
- Use **Demos 02–05** to explore the `BT` method and HAC linkage sensitivity on block-structured designs, including
  heavy-tailed robustness.
- Use **Demo 06** for the richest hierarchical dependency structure.
- See the BT diagnostic in the library test suite
  (`TRexSelector/cpp/tests/validation/trex_selector_methods/ad/trex_da/`) if you need to debug or verify the BT
  clustering pipeline itself against the R reference.
- Cross-check against the R implementation in `R/trex_selector_methods/ad/trex_da/` (one folder per demo, plus the
  shared `da_sim_common.R` and `trex_da_dgps.R`). The shared `simulation_utils.R` and `support_generators.R` helpers
  live one level up, in `R/trex_selector_methods/`. The R and Python suites use the same demo folders and numbers as
  this one.

---

## References

- See [../../README.md](../../README.md) for the category overview of all T-Rex selector variants.
- See [../trex_core/README.md](../trex_core/README.md) for the classical (non-DA) T-Rex selector and shared statistical
  background.
