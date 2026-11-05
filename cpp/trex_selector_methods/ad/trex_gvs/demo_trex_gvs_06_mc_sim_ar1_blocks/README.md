# Demo 06: T-Rex+GVS on AR(1) Blocks

## Purpose

Evaluate the Terminating Random Experiments (T-Rex) selector with Grouped Variable Selection (GVS), in short T-Rex+GVS,
when within-block correlation follows a
 **first-order autoregressive (AR(1)) time-series structure** instead of equicorrelation.
 This uses the same four-block active/trap geometry as Demo 03, but replaces the shared-latent-factor
 equicorrelated model with an AR(1) Toeplitz covariance within each block, testing robustness to more realistic,
 decaying correlation patterns.
 Selection and evaluation are per variable (see
 [What is actually measured](../README.md#what-is-actually-measured-in-these-demos)).

---

## Data Generation Parameters (`make_ar1_blocks_dgp`)

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

- Same 4-block geometry as Demo 03, with sizes $\lbrace20, 50, 80, 65\rbrace$ (215 grouped columns in total).
- **Blocks 0–2 are active** ($\beta_j = 3$), while **block 3 is an inactive AR(1) trap** ($\beta_j = 0$).
- Gap columns (between and around the blocks) are i.i.d. $\mathcal{N}(0,1)$.
- Correlation between two variables in the same block decays with their column distance, so a block no longer
   behaves like a single tight cluster — variables at opposite ends of a large block can be nearly uncorrelated.

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
elastic net) [[2]](#references). All use single-linkage hierarchical agglomerative clustering
(HAC) grouping. TENET and TENET_AUG select $\lambda_2$ with `CV_1SE_CCD`, and TIENET_AUG with
`CV_1SE_IEN_CCD`, the rule built for the informed elastic net geometry, with its support cap raised to
$180$ variables (see [Demo 01](../demo_trex_gvs_01_mc_sim_hastie_en_blocks/README.md), Part 2).

---

## The Sweep

A single **2-D SNR $\times$ $\rho$ grid**, over
$\mathrm{SNR} \in \lbrace0.2, 0.5, 1, 2, 5\rbrace$ and
$\rho \in \lbrace0.30, 0.50, 0.70, 0.85, 0.90, 0.99\rbrace$, 200 Monte Carlo (MC) trials per cell. Here $\rho$ is the
AR(1) coefficient of the within-block Toeplitz covariance, so it controls how fast correlation decays
with column distance rather than a uniform within-block correlation level.

---

## Output Files

Written to `simulation_results/data/`:

- `gvs_ar1_blocks_2d.txt` / `.csv` — mean false discovery proportion (FDP) / true positive proportion (TPP) over the SNR
  $\times$ $\rho$ grid.

Figures in Portable Network Graphics (PNG) and Portable Document Format (PDF) go to `simulation_results/plots/`,
produced by `./generate_plots.sh`.

---

## Running the Demo

```bash
./build/release/bin/trex_selector_methods/ad/trex_gvs/demo_trex_gvs_06_mc_sim_ar1_blocks/demo_trex_gvs_06_mc_sim_ar1_blocks
./generate_plots.sh   # render the figure below from the saved CSV
```

---

## Simulation Results

- The false discovery rate (FDR) stays under control everywhere (no cyan cells). **TENET** and **TENET_AUG** track
   each other and peak near $0.09$, just below the $\mathrm{tFDR} = 0.1$ target, while **TIENET_AUG** is markedly more
   conservative (FDR at most $0.056$).
- **The true positive rate (TPR) recovers much more slowly with SNR than in the equicorrelated Hastie/Mixed-Blocks
   demos.** At $\mathrm{SNR} = 5$ and $\rho = 0.5$ TENET reaches $0.13$ here, against $0.93$ in the equicorrelated
   [Demo 03](../demo_trex_gvs_03_mc_sim_mixed_blocks/README.md). The grouping is not the cause, because below
   $\rho = 0.99$ the fixed cut at `corr_max = 0.98` lies above the correlation of even neighboring columns, so it
   leaves every variable in its own cluster, here as in Demo 03. The
   AR(1) decay keeps only nearby columns of a block strongly correlated, so the active columns share no common factor
   and plausibly each carries too little of the signal on its own at $n = 200$. Full recovery needs both high $\rho$
   and high SNR, and even at $\mathrm{SNR} = 5$ the $\text{low-}\rho$ columns barely lift off zero.
- **TIENET_AUG** again has less power than the TENET-based methods (TPR up to $0.86$ against $0.96$), as in most
   cells of the other block demos. Its TPR rises with SNR in every column, but in the $\rho = 0.99$ column it stays
   far below TENET's at $0.010\text{–}0.104$. That is the only column the fixed cut at `corr_max = 0.98` splits into
   groups. Neighboring columns, correlated at $0.99$, lie above the cut, so single-linkage HAC chains each block into
   one cluster. Only neighboring columns are nearly collinear there, because the correlation still decays with column
   distance.
- The $\rho$ axis is the most informative view of this demo. Watch how quickly recovery degrades as $\rho$
   decreases and the AR(1) decay becomes steeper (correlation vanishing faster with column distance).

TPR (top) and FDR (bottom) heatmaps over the SNR $\times$ $\rho$ grid, one column per solver.
FDR cells above the $\mathrm{tFDR} = 0.1$ target are outlined in cyan.

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
