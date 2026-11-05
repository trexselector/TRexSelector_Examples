# Demo 06: T-Rex+GVS on AR(1) Blocks (Python)

## Purpose

Evaluate the Terminating Random Experiments (T-Rex) selector with Grouped Variable Selection (GVS), in short T-Rex+GVS,
when within-block correlation follows a
 **first-order autoregressive (AR(1)) time-series structure** instead of equicorrelation.
 This uses the same four-block active/trap geometry as Demo 03, but replaces the shared-latent-factor
 equicorrelated model with an AR(1) Toeplitz covariance within each block, testing robustness to more realistic,
 decaying correlation patterns.
 Selection and evaluation are per variable (see
 [What is actually measured](../README.md#what-is-actually-measured-in-these-demos)).

This is the Python port of the C++ demo
[demo_trex_gvs_06_mc_sim_ar1_blocks](../../../../../cpp/trex_selector_methods/ad/trex_gvs/demo_trex_gvs_06_mc_sim_ar1_blocks/),
using `trexselector`'s `TRexGVSSelector`.

---

## Data Generation Parameters (`dgp_ar1_blocks_snr`)

We consider again the linear model:

```math
\boldsymbol{y} = \boldsymbol{X}\boldsymbol{\beta} + \boldsymbol{\epsilon},
\qquad \boldsymbol{\epsilon} \sim \mathcal{N}(\boldsymbol{0}, \sigma_{\varepsilon}^2 \boldsymbol{I}_n)
```

- $\boldsymbol{y} \in \mathbb{R}^n$ is the response vector.
- $\boldsymbol{X} \in \mathbb{R}^{n \times p}$ is the design matrix.
- $\boldsymbol{\beta} \in \mathbb{R}^p$ is the coefficient vector, with $s$ nonzero entries.
- $\boldsymbol{\epsilon}$ is the noise vector, independent and identically distributed (i.i.d.) standard normal.
- $\sigma_{\varepsilon}^2$ is the noise variance, calibrated to achieve a target linear signal-to-noise ratio (SNR).
- $n = 200$, $p = 500$, $s = 150$ (the sum of the three active blocks).

Unlike Demos 01–05, the within-block structure is **not** equicorrelated.
 Each block $k$ follows an AR(1) Toeplitz covariance,

```math
\Sigma_k(i,j) = \rho^{\lvert i-j \rvert},
```

generated as $X_{\text{block}} = Z L^\top$, where $Z \sim \mathcal{N}(0, I_{n \times b_k})$ and $L$ is the
 Cholesky factor of $\Sigma_k$.

- Same 4-block geometry as Demo 03: sizes $\lbrace20, 50, 80, 65\rbrace$ (215 grouped columns in total).
- **Blocks 0–2 are active** ($\beta_j = 3$), while **block 3 is an inactive AR(1) trap** ($\beta_j = 0$).
- Gap columns (between and around the blocks) are i.i.d. $\mathcal{N}(0,1)$.
- Correlation between two variables in the same block decays with their column distance, so a block no longer
   behaves like a single tight cluster — variables at opposite ends of a large block can be nearly uncorrelated.

![Design-matrix correlation — block-diagonal AR(1), natural order (no clustering)](../dgp_corr_plots/gvs_ar1_blocks.png)

Empirical Pearson correlation of an exemplary design drawn from the data-generating process (DGP) `dgp_ar1_blocks_snr`
($n=200$, $p=500$, $\rho = 0.85$, seed 2026), in **natural variable order — no hierarchical clustering**. Instead of
the solid squares of the equicorrelated demos, each block shows as a **band fading off the diagonal** — the
$\rho^{|i-j|}$ decay. In the two large blocks the band dies out well before the block ends. At this $\rho$ the
fixed cut of the hierarchical agglomerative clustering (HAC) at `corr_max = 0.98` lies above even the correlation
of neighboring columns, so it leaves every variable in its own cluster. One figure per DGP lives in
[`../dgp_corr_plots/`](../dgp_corr_plots/). Regenerate the whole suite with
[`../generate_corr_plots.sh`](../generate_corr_plots.sh) (shared plotter
[`../../../corr_plt_utils.py`](../../../corr_plt_utils.py)).

---

## Control Parameters

```text
K = 20                          # Random experiments per T-loop iteration
tFDR = 0.1                      # Target FDR
corr_max = 0.98                 # HAC auto-clustering correlation threshold
hc_linkage = Single             # Single-linkage HAC
lambda2_method = CV_1SE_CCD     # Elastic-net penalty selection, EN rows (TENET, TENET_AUG)
lambda2_method = CV_1SE_IEN_CCD # IEN rows (TIENET_AUG)
cv_dfmax_ratio = 1.0            # IEN support cap 1.0 * n_train = 180 (default 0.8 gives 144)
MC = 200                        # Monte Carlo repetitions per grid point
```

---

## Methods Compared

Three T-Rex+GVS solver variants: **TENET** (elastic net) [[3]](#references),
**TENET_AUG** (row-augmented elastic net) [[1]](#references), and **TIENET_AUG** (informed
elastic net) [[2]](#references). All use single-linkage HAC grouping. TENET and TENET_AUG select
$\lambda_2$ with `CV_1SE_CCD`, and TIENET_AUG with `CV_1SE_IEN_CCD`, the rule built for the informed elastic net
geometry, with its support cap raised to $180$ variables (see
[Demo 01](../demo_trex_gvs_01_mc_sim_hastie_en_blocks/README.md), Part 2).

---

## The Sweep

A single **2-D SNR $\times$ $\rho$ grid**, over
$\mathrm{SNR} \in \lbrace0.2, 0.5, 1, 2, 5\rbrace$ and
$\rho \in \lbrace0.30, 0.50, 0.70, 0.85, 0.90, 0.99\rbrace$, 200 Monte Carlo (MC) trials per cell. Here $\rho$ is
the AR(1) coefficient of the within-block Toeplitz covariance, so it controls how fast correlation decays
with column distance rather than a uniform within-block correlation level.

---

## Output Files

Written to `simulation_results/data/`:

- `gvs_ar1_blocks_2d.txt` / `.csv` — mean false discovery proportion (FDP) / true positive proportion (TPP) over the SNR
  $\times$ $\rho$ grid.

The file names and comma-separated values (CSV) schema match the C++ demo, and the suite plotter
[`../trex_gvs_plt_utils.py`](../trex_gvs_plt_utils.py) is the same module the C++ suite uses.

Figures in Portable Network Graphics (PNG) and Portable Document Format (PDF) go to `simulation_results/plots/`,
produced by `./generate_plots.sh`.

---

## Running the Demo

```bash
python demo_trex_gvs_06_mc_sim_ar1_blocks.py       # full run (hours)
python demo_trex_gvs_06_mc_sim_ar1_blocks.py 8     # with 8 worker processes
./generate_plots.sh                                # render the figure below from the saved CSV
```

Monte Carlo trials are parallelized with `ProcessPoolExecutor` on `TREX_NUM_WORKERS` worker processes (default 8,
and `0` uses every usable core, see `resolve_workers()` in [`../gvs_sim_common.py`](../gvs_sim_common.py)). The
optional first command-line interface (CLI) argument overrides this count. A full run takes about 2.2 hours on
8 workers. For a quick smoke test, temporarily reduce `num_MC` and the sweep grids at the top of the script.

---

## Simulation Results

The numbers and figure below are this Python run's own results (200 MC trials per cell). They agree with the
[C++ reference](../../../../../cpp/trex_selector_methods/ad/trex_gvs/demo_trex_gvs_06_mc_sim_ar1_blocks/README.md#simulation-results)
up to Monte-Carlo noise — the two ports draw from independent random number generator (RNG) streams.

- The false discovery rate (FDR) stays under control everywhere (no cyan cells). **TENET** and **TENET_AUG** are
   numerically identical and peak at $0.087$, just below the $\mathrm{tFDR} = 0.1$ target, while **TIENET_AUG** is
   markedly more conservative (FDR $\le 0.054$).
- **The true positive rate (TPR) recovers much more slowly with SNR than in the equicorrelated Hastie/Mixed-Blocks
   demos.** At $\mathrm{SNR} = 5$, TENET's TPR runs $0.02 / 0.12 / 0.51 / 0.88 / 0.95 / 0.96$ across
   $\rho \in \lbrace0.3, 0.5, 0.7, 0.85, 0.9, 0.99\rbrace$, so the $\text{low-}\rho$ columns barely lift off zero.
   At $\rho = 0.5$ and $\mathrm{SNR} = 5$ it reaches $0.12$ here, against $0.93$ in the equicorrelated
   [Demo 03](../demo_trex_gvs_03_mc_sim_mixed_blocks/README.md). The HAC grouping is not the cause, because below
   $\rho = 0.99$ the fixed cut at `corr_max = 0.98` lies above the correlation of even neighboring columns and
   leaves every variable in its own cluster, here as in Demo 03. A plausible reading is that the AR(1) decay keeps
   only nearby columns of a block strongly correlated, so the active columns share no common factor and each carries
   too little of the signal on its own at $n = 200$. Full recovery needs both a high $\rho$ and a high SNR.
- **TIENET_AUG** again has less power than the TENET-based methods, as in most cells of the other block demos. Its
   TPR rises with SNR in every column (to $0.86$ at $\rho = 0.9$, $\mathrm{SNR} = 5$, against $0.95$ for TENET). In
   the $\rho = 0.99$ column, where neighboring columns are nearly collinear, it rises too, from $0.010$ to $0.104$,
   but stays far below TENET's $0.754\text{–}0.958$.
- The $\rho$ axis is the most informative view of this demo. Recovery degrades sharply as $\rho$ decreases and the
   AR(1) decay becomes steeper (correlation vanishing faster with column distance).

TPR (top) and FDR (bottom) heatmaps over the SNR $\times$ $\rho$ grid, one column per solver.
FDR cells above the $\mathrm{tFDR} = 0.1$ target would be outlined in cyan.

![AR(1) blocks SNR × rho sweep](simulation_results/plots/gvs_ar1_blocks_2d.png)

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
3. Zou, H., & Hastie, T. (2005). "Regularization and variable selection via the elastic net." *Journal of the Royal
   Statistical Society: Series B (Statistical Methodology)*, 67(2), pp. 301–320.
   [DOI-Link](https://doi.org/10.1111/j.1467-9868.2005.00503.x)
