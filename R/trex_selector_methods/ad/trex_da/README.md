# DA-TRex: Dependency-Aware T-Rex — R Demonstration Suite

> **Status**: New — migrated to the **TRexSelector** R6 application programming interface (API)
> (`TRexDASelector$new(..., da_control = trex_da_control(...))$select()`, with `trex_fdp()` / `trex_tpp()`). No
> longer depends on the `TRexSelector` package from the Comprehensive R Archive Network (CRAN).

## Purpose

R demos and Monte Carlo (MC) simulations for the **Dependency-Aware T-Rex (DA-TRex)** selector across correlated
designs: first-order autoregressive (AR) Toeplitz, AR(1) for short, equicorrelation, nearest-neighbor (NN) with a banded
moving-average (MA) structure, block-diagonal AR(1) (with and without white-noise augmentation), and heavy-tailed
Student-t blocks.

The folder layout mirrors the C++ / Python suites, with one subfolder per demo, plus a single consolidated
data-generating process (DGP) file and the shared MC / writer infrastructure.

```txt
trex_da/
  ├── README.md
  ├── da_sim_common.R                  <- solver set, MC runners, sweeps, writers
  ├── trex_da_dgps.R                   <- all scenario DGPs (one file)
  ├── trex_da_plt_utils.R              <- shared plotter (figures of every demo)
  ├── demo_trex_da_01_mc_sim_ar1/
  │   ├── demo_trex_da_01_mc_sim_ar1.R
  │   ├── README.md
  │   ├── generate_plots.sh            <- renders this demo's figures
  │   └── simulation_results/
  │       ├── data/                    <- da_trex_mc_*.{txt,csv}
  │       └── plots/                   <- *.{png,pdf}
  ├── demo_trex_da_02_mc_sim_ar1_blocks_plus_white/
  │   └── ...
  └── demo_trex_da_08_mc_sim_equi_and_bt/
```

Dependencies: `TRexSelector`, `parallel` (MC loops). Each demo sources the
consolidated [trex_da_dgps.R](trex_da_dgps.R) (all scenario generators in one
file, mirroring the C++ `dgp_generators.hpp` / Python `dgp_generators.py`), the
shared [da_sim_common.R](da_sim_common.R) (default solver set, parallel MC
runners, parameter sweeps, and the `.txt`/`.csv` result writers, mirroring the Python
`da_sim_common.py` / C++ `trex_da_sim_common.hpp`), and
[../../support_generators.R](../../support_generators.R).

---

## Demos

| File | Description | cpp counterpart |
|---|---|---|
| [demo_trex_da_01_mc_sim_ar1](demo_trex_da_01_mc_sim_ar1/) | AR(1) Toeplitz DGP: signal-to-noise ratio (SNR) sweep and rho sweep (CappedSpread and Random support, dependency-aware (DA) + base T-Rex rows), plus a 2D gap x rho sweep exploring the kappa boundary | `cpp/trex_selector_methods/ad/trex_da/demo_trex_da_01_mc_sim_ar1` |
| [demo_trex_da_02_mc_sim_ar1_blocks_plus_white](demo_trex_da_02_mc_sim_ar1_blocks_plus_white/) | AR(1)-block + white-noise DGP (fixed p_total = 1,000, p_white = p_total − M*Q): SNR, rho, Q, and M sweeps, each with an outer linkage loop | `cpp/trex_selector_methods/ad/trex_da/demo_trex_da_02_mc_sim_ar1_blocks_plus_white` |
| [demo_trex_da_03_mc_sim_ar1_blocks](demo_trex_da_03_mc_sim_ar1_blocks/) | Pure AR(1)-block DGP (p = M*Q): SNR, rho, Q, and M sweeps, each with an outer linkage loop | `cpp/trex_selector_methods/ad/trex_da/demo_trex_da_03_mc_sim_ar1_blocks` |
| [demo_trex_da_04_mc_sim_ht_blocks](demo_trex_da_04_mc_sim_ht_blocks/) | Heavy-tailed Student-t(3) AR(1)-block data, Gaussian vs Student-t noise scenarios: SNR, rho, Q, M, target false discovery rate (tFDR), and linkage sweeps | `cpp/trex_selector_methods/ad/trex_da/demo_trex_da_04_mc_sim_ht_blocks` |
| [demo_trex_da_05_mc_sim_ht_blocks_plus_white](demo_trex_da_05_mc_sim_ht_blocks_plus_white/) | Heavy-tailed blocks + heavy-tailed white augmentation, Gaussian vs heavy-tailed noise: SNR, rho, Q, M, and tFDR sweeps | `cpp/trex_selector_methods/ad/trex_da/demo_trex_da_05_mc_sim_ht_blocks_plus_white` |
| [demo_trex_da_06_mc_sim_groups](demo_trex_da_06_mc_sim_groups/) | Prior-groups method: 3-level hierarchy (groups of 10 / 50 / 250) over a multi-level latent-factor DGP with Toeplitz leaf blocks, SNR sweep. The method is selected by supplying a non-empty `prior_groups` to `trex_da_control()` | `cpp/trex_selector_methods/ad/trex_da/demo_trex_da_06_mc_sim_groups` |
| [demo_trex_da_07_mc_sim_nn](demo_trex_da_07_mc_sim_nn/) | Nearest-Neighbours / MA(kappa) banded DGP: SNR sweep (DA + base T-Rex rows), a 2D kappa x rho sweep, plus TREX+DA+NN applied to AR(1) data (method-mismatch test) as a 2D SNR x rho sweep | `cpp/trex_selector_methods/ad/trex_da/demo_trex_da_07_mc_sim_nn` |
| [demo_trex_da_08_mc_sim_equi_and_bt](demo_trex_da_08_mc_sim_equi_and_bt/) | Equicorrelated DGP (single latent factor): one 2D SNR x rho sweep with a rho grid dense near 0, one `.txt`/`.csv` pair per SNR column, DA + base T-Rex rows | `cpp/trex_selector_methods/ad/trex_da/demo_trex_da_08_mc_sim_equi_and_bt` |

Notes:

- Every MC part runs four DA solver rows — `TREX-DA[+variant]: TLARS / TAFS / TOMP / TLARS_STAG` (variant tags
  `+AR1`, `+BT`, `+NN`, `+EQUI`, while the prior-groups demo 06 uses plain `TREX-DA:`). The greedy TAFS and TOMP use
  exchangeable tie-breaking (`exch_tie_alpha = 0.25`) and the T-loop stagnation stop, and TAFS runs with
  `rho_afs = 0.3`. TLARS runs without the stagnation stop, and TLARS_STAG is the same TLARS with the stop on
  (window 7). Demos 01, 07, and 08 additionally run the base (no-DA) `TREX: TLARS / TAFS / TOMP / TLARS_STAG`
  comparison rows.
- Hierarchical agglomerative clustering (HAC) linkage labels are capitalized (`Single`, `Complete`, `Average`) and also
  appear as filename tokens in the block demos' result stems.
- The prior-groups method (demo 06) is driven by passing `prior_groups` to `trex_da_control()`. The C++ core routes on
  `prior_groups` being non-empty, so `da_method` is then irrelevant.

---

## DGPs

All scenario generators live in the single [trex_da_dgps.R](trex_da_dgps.R)
(mirroring the C++ `dgp_generators.hpp` / Python `dgp_generators.py`):

- `dgp_ar1_snr` — AR(1) Toeplitz design, SNR-calibrated noise (demos 01, 07).
- `dgp_ar1_block_white_snr` / `dgp_ar1_block_snr` — AR(1)-block covariance with
  and without white augmentation (demos 02, 03).
- `dgp_block_toeplitz_hvt_snr` / `dgp_ht_block_white_snr` — heavy-tailed
  Student-t AR(1) blocks via Gaussian scale mixture, with and without t(nu)
  white augmentation (demos 04, 05).
- `dgp_groups_toeplitz_leaf` — multi-level latent-factor design with Toeplitz
  leaf blocks (prior-groups demo 06).
- `dgp_nn_snr` — banded MA(kappa) covariance via causal convolution of a shared
  innovation sheet (demo 07).
- `dgp_equi_snr` — pure equicorrelation via a single latent factor (demo 08).

All generators calibrate the noise variance to a target SNR = Var(signal)/Var(noise)
and take the true support as 1-based R indices. (`dgp_bt_snr` and
`dgp_nn_block_snr` are demo-less leftovers kept for reference.)

---

## Results

Each demo writes its result tables to its own `simulation_results/data/` folder as `da_trex_mc_<scenario_tag>.{txt,csv}`
— the same stems and tidy comma-separated values (CSV) schemas as the C++ / Python suites. Each demo's
`./generate_plots.sh` renders the figures from these tables into `simulation_results/plots/` via the shared
[trex_da_plt_utils.R](trex_da_plt_utils.R), with the same file names as the Python suite.

The committed results are the R run of all eight demos on library commit `09debc6` (200 Monte Carlo trials per cell,
8 workers, about 25 h in total). R draws its own random numbers, so its cells agree with C++ and Python up to Monte
Carlo noise, not exactly. Of 9,344 cells compared with C++, 29 (0.31 %) lie more than three Monte Carlo standard
errors away, the rate noise alone produces. The largest single differences are 0.120 in the false discovery rate
(FDR) (Demo 01, DA-TLARS at gap 10 and $\rho = 0.9$, a high-variance false-alarm cell below the correction-window
boundary) and 0.122 in the true positive rate (TPR) (Demo 04, DA-TLARS_STAG at $M = 3$ with heavy-tailed noise and
Single linkage).

## What to expect

DA-TRex keeps the FDR far closer to the target than the classical (no-DA) T-Rex selector on the same correlated
data. In Demo 01 at $\mathrm{SNR} = 2$ and $\rho = 0.7$ the DA rows realize an FDR of $0.03\text{–}0.05$, while base
T-Rex TLARS reaches $0.27\text{–}0.29$. The greedy DA solvers stay below the target of $0.2$ across the whole $\rho$
grid of Demo 01 for both support placements (at most $0.19$).

TLARS_STAG matches TLARS within Monte Carlo noise in every demo, except below the correction-window boundary of
Demo 01, where it raises fewer false alarms (FDR at most $0.015$ against $0.050\text{–}0.115$ at $\rho = 0.9$).
Averaged over a demo's DA cells, the TPR changes by at most $0.004$ and the FDR by at most $+0.003$. TLARS breaks the
target in 55 DA cells over all demos and TLARS_STAG in 52, the cells where only one of the two crosses it lie within
$0.018$ of the target, and the T-loop of TLARS_STAG stops after at most $8.0$ steps on average.

The demo READMEs document the known rough edges.

- TLARS, and TLARS_STAG with it, breaks the target at $\rho = 0.9$ on the block designs under Average and Complete
  linkage (Demos 03 and 04, FDR $0.21\text{–}0.25$), and on the multi-level prior-groups design of Demo 06, where its
  FDR stays near $0.36$ from $\mathrm{SNR} = 0.5$ on while TAFS and TOMP stay below $0.2$.
- Heavy-tailed noise raises the FDR in Demo 05 (mean over the DA cells $0.113$ against $0.071$ with Gaussian noise)
  without costing power, while in Demo 04 the two noise models give the same averages.
- The NN correction breaks down when applied to AR(1) data at $\rho \geq 0.8$ (Demo 07, FDR up to $0.62$ for TLARS
  at $\rho = 0.9$).
- Equicorrelation suppresses the DA selector from $\rho = 0.025$ on (Demo 08, TPR at most $0.005$).

---

## Running

```bash
Rscript R/trex_selector_methods/ad/trex_da/demo_trex_da_01_mc_sim_ar1/demo_trex_da_01_mc_sim_ar1.R      # full run
Rscript R/trex_selector_methods/ad/trex_da/demo_trex_da_01_mc_sim_ar1/demo_trex_da_01_mc_sim_ar1.R 8    # 8 worker cores
```

The optional first command-line argument sets the number of parallel worker cores
(default `TREX_NUM_WORKERS`, else 8). Scripts resolve their own directory, so they run from any working
directory. Individual parts are gated behind `RUN_PART_*` flags at the top of
each file. Each demo folder has its own `README.md`, and
[run_overnight.sh](run_overnight.sh) regenerates the whole suite's result data.
