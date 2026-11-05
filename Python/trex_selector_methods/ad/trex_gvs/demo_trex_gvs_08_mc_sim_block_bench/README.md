# Demo 08: T-Rex+GVS Block Benchmark — Value of Prior Group Information (Python)

## Purpose

Quantify what **prior group knowledge is actually worth** to the Terminating Random Experiments (T-Rex) selector with
Grouped Variable Selection
 (GVS), in short T-Rex+GVS, as a function of how easily the groups could have been discovered from the data anyway.
 Four method variants factorize the question: elastic net (EN) vs. informed elastic net (IEN), each run once with
 groups discovered by hierarchical agglomerative clustering (HAC) and once with the true ("oracle") groups from the
 data-generating process (DGP).
 The headline sweep moves the within-block correlation $\rho$ across the HAC discoverability boundary set by
 `corr_max = 0.5`. Below it the clustered methods see near-singletons and the oracle prior is the only source of
 group structure, while above it HAC rediscovers the blocks and the prior becomes redundant.
 Selection and evaluation are per variable (see
 [What is actually measured](../README.md#what-is-actually-measured-in-these-demos)).

This is the Python port of the C++ demo
[demo_trex_gvs_08_mc_sim_block_bench](../../../../../cpp/trex_selector_methods/ad/trex_gvs/demo_trex_gvs_08_mc_sim_block_bench/),
using `trexselector`'s `TRexGVSSelector`.

---

## Data Generation Parameters (`dgp_block_equicorr`)

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
- $n = 200$, $p = 500$, and $s$ depends on the truth scenario below.

The design matrix consists of $G = 100$ equal-size blocks of `block_size` $= 5$ variables each, drawn from the
 unit-variance equicorrelated latent-factor model:

```math
X_{ij} = \sqrt{\rho}\, Z_{i,g(j)} + \sqrt{1-\rho}\, \xi_{ij},
\qquad Z_{\cdot,k},\ \xi_{ij} \sim \mathcal{N}(0,1).
```

- `n_active_blocks` $= 10$ blocks are chosen at random per trial, and each receives a random-sign coefficient
   $\beta = \pm 3$.
- The DGP exports `prior_groups[j] = j // 5` — the exact generating block of every column — which the oracle
   variants pass to `TRexGVSControlParameter.prior_groups`.

![Design-matrix correlation — 100 equicorrelated blocks of size 5, natural order (no clustering)](../dgp_corr_plots/gvs_block_equicorr.png)

Empirical Pearson correlation of an exemplary design drawn from this DGP (`dgp_block_equicorr`, $n=200$, $p=500$,
$\rho = 0.9$, seed 2026), in **natural variable order — no hierarchical clustering**. Only the leading
$75 \times 75$ corner is shown, because 100 blocks of 5 columns are sub-pixel over the full $p = 500$. The chain
of tiny $5 \times 5$ squares along the diagonal is the block structure the oracle prior encodes exactly and HAC has
to rediscover — at $\rho = 0.9$ it does, but at $\rho \le 0.35$ the same blocks fall below the `corr_max` $= 0.5$
cut and dissolve into singletons. One figure per DGP lives in
[`../dgp_corr_plots/`](../dgp_corr_plots/). Regenerate the whole suite with
[`../generate_corr_plots.sh`](../generate_corr_plots.sh) (shared plotter
[`../../../corr_plt_utils.py`](../../../corr_plt_utils.py)).

### Truth scenarios

- **WHOLE** (headline): all 5 variables of each active block are active, $s = 50$. This matches the grouped-selection
   premise of the T-Rex+GVS papers — active groups are fully active — so coordinate-level false discovery rate (FDR)
   and true positive rate (TPR) are the right metrics.
- **INDIVIDUAL** (cautionary appendix): exactly 1 of the 5 variables per active block is active, $s = 10$. The four
   null siblings of each active variable are exchangeable with it up to the $(1-\rho)$ idiosyncratic component, so
   no selector that aggregates evidence across correlated variables can control the coordinate-level FDR here —
   this scenario is included to show that failure mode explicitly, not as a benchmark to win.

(The DGP additionally supports a REPRESENTATIVE scenario — 2–3 active variables per block — which this benchmark,
 like the C++ one, does not run.)

---

## Four Methods Compared

| Method | `GVSType` | Groups |
| --- | --- | --- |
| **M1 (EN + HAC)** | EN [[1]](#references) | discovered per trial via single-linkage HAC at `corr_max` |
| **M2 (EN + oracle)** | EN [[1]](#references) | true `prior_groups` from the DGP |
| **M3 (IEN + HAC)** | IEN [[2]](#references) | discovered per trial via single-linkage HAC at `corr_max` |
| **M4 (IEN + oracle)** | IEN [[2]](#references) | true `prior_groups` from the DGP |

The group information enters the two solvers differently. The EN variants use it **only** to draw
 group-correlated dummy (random-experiment) columns, while IEN additionally ties the group members together
 through its group-indicator row augmentation — and those indicator rows are tiled across the dummy layers as
 well, so every variable's group-aligned dummies share its group geometry. The oracle prior can therefore
 matter more — in either direction — for IEN than for EN.

The two tracks also choose $\lambda_2$ differently. The EN variants cross-validate it on the original
 $(\boldsymbol{X}, \boldsymbol{y})$ with the ridge-shaped rule `CV_1SE_CCD`, so their penalty is identical for the
 clustered and oracle groups. The IEN variants use the IEN-geometry rule `CV_1SE_IEN_CCD`, which cross-validates
 $\lambda_2$ jointly with $\lambda_1$ on the group structure the variant is given, so the clustered and oracle runs
 get different penalties (e.g. $0.39$ vs. $0.13$ at $\rho = 0.35$, $\mathrm{SNR} = 5$). Its $\lambda_1$ path stops at
 $0.8\, n_{\mathrm{train}} = 144$ selected variables, well above the $s \le 50$ true variables here. Demos 01–07 use
 the same rule for their IEN rows, with the cap raised to $n_{\mathrm{train}} = 180$ because most of them have
 $s = 150$ (see [Demo 01](../demo_trex_gvs_01_mc_sim_hastie_en_blocks/README.md)). Here the IEN rule returns a
 boundary pick in about $8{,}300$ of the $19{,}200$ IEN selections ($43\,\%$, as in C++). The library reports such
 picks on the console and keeps them.

---

## Control Parameters

```text
K = 20                # Random experiments per T-loop iteration
tFDR = 0.1            # Target FDR
corr_max = 0.5        # HAC discoverability boundary (cut height 1 - corr_max)
hc_linkage = Single   # Single-linkage HAC
lambda2_method        # CV_1SE_CCD (EN rows M1/M2), CV_1SE_IEN_CCD (IEN rows M3/M4)
b = 3.0               # Nonzero coefficient magnitude (random sign per block)
MC = 200              # Monte Carlo repetitions per grid cell
```

Seeds are staggered per grid cell (`cell_base_seed = base_seed + cell_index * num_MC`, with
`cell_index = i_rho * n_snr + i_snr` over the scenario's own $\rho$ grid), so each $(\rho, \mathrm{SNR})$ cell
draws from a distinct seed band, and all four methods see identical data within a trial.

---

## The Sweep

- **WHOLE**: a 2-D $\rho \times \mathrm{SNR}$ grid over $\rho \in \lbrace0.2, 0.35, 0.5, 0.65, 0.8, 0.9\rbrace$ and
  $\mathrm{SNR} \in \lbrace0.1, 0.2, 0.5, 1, 2, 5\rbrace$, 200 Monte Carlo (MC) trials per cell. The $\rho$ grid
  deliberately brackets `corr_max` $= 0.5$, because single-linkage merges within-block pairs at distance
  $\approx 1 - \rho$, so for $\rho \le 0.35$ every merge lies above the fixed cut height $1 -$ `corr_max` $= 0.5$
  and no group survives the cut (purity $0$, $M \approx 500$ singletons). The point $\rho = 0.5$ sits on the
  boundary, while $\rho \ge 0.65$ is fully discovered (purity $1.0$, $M = 100$).
  Holding `corr_max` fixed while $\rho$ varies models a **mis-specified discovery threshold** — the practical
  situation where the true correlation level is unknown — not a limitation of the clustering itself, which
  recovers these blocks whenever the cut is set below the within-block correlation.
- **INDIVIDUAL**: the same SNR grid at the low/high pair $\rho \in \lbrace0.35, 0.9\rbrace$.

---

## Metrics Reported

The primary metrics are the usual **coordinate-level FDR/TPR** — the quantities the T-Rex selector actually
 controls and targets. The saved tables additionally report block-level and grouping diagnostics:

- **`blk_hit`** — share of active blocks with at least one selected variable (a block-level TPR), or
  **`full_blk`** — share of active blocks recovered in their entirety (WHOLE tables).
- **`block_FDR`** — share of hit blocks that are truly null.
- **`purity`** — share of true blocks whose members all land in one discovered group ($1.0$ by construction for
  oracle runs).
- **`T_stop`**, **`M_found`** — selector diagnostics.

> The block-level numbers are **descriptive diagnostics only**. T-Rex+GVS as implemented controls the FDR of the
> selected variables and is not designed to control a block-level FDR, so no target and no guarantee attach to
> `block_FDR`, and a value above $\mathrm{tFDR}$ does not indicate a loss of FDR control. A block-level error rate
> is a different quantity with its own definition of a false discovery (cf.
> [the group-FDR discussion](../README.md#what-is-actually-measured-in-these-demos)).

The grouping diagnostics come straight from the `GVSSelectionResult` returned by `select()`: `max_clusters`
(→ `M_found`) and `groups_vec` (per-variable cluster labels, from which `purity` is computed for HAC and oracle
runs alike).

---

## Output Files

Written to `simulation_results/data/`, one pair per truth scenario:

- `gvs_block_bench_whole.txt` / `.csv`
- `gvs_block_bench_individual.txt` / `.csv`

Each `.txt` reproduces the console grid tables, and each `.csv` is a tidy long-format table with all four methods'
aggregates per $(\rho, \mathrm{SNR})$ cell. The file names and comma-separated values (CSV) schemas match the C++
demo, and the suite plotter [`../trex_gvs_plt_utils.py`](../trex_gvs_plt_utils.py) is the same module the C++ suite
uses.

Figures in Portable Network Graphics (PNG) and Portable Document Format (PDF) go to `simulation_results/plots/`,
produced by `./generate_plots.sh`.

---

## Running the Demo

```bash
python demo_trex_gvs_08_mc_sim_block_bench.py       # full run (about 7 h on 8 workers)
python demo_trex_gvs_08_mc_sim_block_bench.py 8     # with 8 worker processes
./generate_plots.sh                                 # render the figures below from the saved CSVs
```

Both scenarios run in the same invocation and print one grid table per scenario, per $\rho$ value.
Monte Carlo trials are parallelized with `ProcessPoolExecutor`, with `TREX_NUM_WORKERS` worker processes (default 8,
and `0` uses every usable core, see `resolve_workers()` in `../gvs_sim_common.py`). The optional first
command-line interface (CLI) argument overrides that count in the same way. For a quick smoke test, temporarily
reduce `num_MC` and the grids at the top of the script.

---

## Simulation Results

The numbers and figures below are this Python run's own results (200 MC trials per cell). They agree with the
[C++ reference](../../../../../cpp/trex_selector_methods/ad/trex_gvs/demo_trex_gvs_08_mc_sim_block_bench/README.md#simulation-results)
up to Monte-Carlo noise — the two ports draw from independent random number generator (RNG) streams.

### WHOLE truth — value of the prior vs. $\rho$ (headline)

- **FDR is controlled everywhere**. The realized coordinate-level FDR never exceeds $0.071$ against the
   $\mathrm{tFDR} = 0.1$ target, for all four methods across the entire grid.
- **For the EN variants the prior is inert**. M1 and M2 differ by at most $0.004$ TPR at every grid point. Since
   EN consumes the group information only through the dummy correlation structure, knowing the true groups
   neither helps nor hurts it here.
- **For IEN the oracle prior costs some power below the boundary.** There the fixed `corr_max` $= 0.5$ cut lies
   above the within-block correlation, so it leaves almost every variable in its own cluster (purity about $0$,
   $M \approx 495\text{–}500$) and clustered IEN works with per-variable rows, while the oracle variant gets the true
   5-member groups. At $\rho = 0.35$ clustered IEN reaches TPR $0.099$, $0.266$ and $0.496$ at
   $\mathrm{SNR} = 1$, $2$ and $5$, against $0.061$, $0.207$ and $0.440$ for oracle IEN, a gap of more than five
   combined Monte Carlo standard errors in each cell ($|z| \approx 5.1$, $6.1$ and $5.1$). At $\rho = 0.2$ the gap is
   at most $0.026$ ($0.120$ vs. $0.094$ at $\mathrm{SNR} = 2$), and at the boundary ($\rho = 0.5$, purity $0.58$,
   $M \approx 179$) at most $0.010$ ($|z| < 1.6$). From $\rho \ge 0.65$ both IEN variants are numerically identical
   because HAC recovers exactly the true blocks (purity $1.0$, $M = 100$).
- **The IEN-geometry $\lambda_2$ rule shrinks the oracle penalty of the earlier setup.** With the ridge-shaped
   `CV_1SE_CCD`, which gave both IEN variants the same $\lambda_2$, the oracle's 5-member ties coupled each active with
   its group-aligned dummies. The false discovery proportion (FDP) estimate then stayed high and the T-loop
   terminated early, so oracle IEN lost most of its power (TPR $0.11$ vs. $0.30$ clustered at $\rho = 0.35$,
   $\mathrm{SNR} = 5$, with $\overline T_{\mathrm{stop}} = 6.3$ vs. $13.1$, earlier run). The IEN rule picks a smaller
   $\lambda_2$ for the oracle groups ($0.13$ vs. $0.39$), but the oracle T-loop still stops earlier ($13.5$ vs.
   $14.3$ steps at $\rho = 0.35$, $\mathrm{SNR} = 5$, and $10.9$ vs. $12.7$ at $\mathrm{SNR} = 2$). A plausible
   reading is a weaker form of the same effect.
- **IEN gains a lot but stays more conservative than EN** (TPR at most $0.56$ for $\rho \ge 0.5$, up from $0.15$ with
   `CV_1SE_CCD`, vs. up to $0.90$ for EN), at FDR at most $0.059$.

TPR (top) and FDR (bottom) vs. $\rho$, one panel per SNR $\in \lbrace0.5, 2, 5\rbrace$, one line per method.
The dashed red line marks the $\mathrm{tFDR} = 0.1$ target.

![Block benchmark — WHOLE, value of prior groups vs rho](simulation_results/plots/gvs_block_bench_whole_vs_rho.png)

- The supplementary vs.-SNR view below shows the same data at three discoverability regimes
   ($\rho = 0.2$ undiscoverable, $0.5$ partial, $0.9$ fully discovered). The EN pair overlaps in every panel,
   and EN recovery grows with both SNR and $\rho$ (TPR $0.250$, $0.645$ and $0.901$ at $\mathrm{SNR} = 5$). A
   plausible reading is that stronger within-block correlation concentrates the signal on fewer effective
   directions, which helps whole-block recovery. IEN rises with SNR too, but not with $\rho$ above the boundary. At
   $\mathrm{SNR} = 5$ it falls from $0.562$ at $\rho = 0.65$ to $0.445$ at $\rho = 0.9$.

![Block benchmark — WHOLE, TPR/FDR vs SNR](simulation_results/plots/gvs_block_bench_whole.png)

### INDIVIDUAL truth — cautionary appendix

- **Coordinate-level FDR control collapses by construction once siblings correlate strongly.** At $\rho = 0.9$
   the EN methods realize FDR $\approx 0.77\text{–}0.80$ from $\mathrm{SNR} = 1$ on ($0.05$, $0.17$ and $0.60$ at
   $\mathrm{SNR} = 0.1$, $0.2$ and $0.5$, target $0.1$) while selecting the active variable's whole
   block. IEN sits at $0.40\text{–}0.50$ from $\mathrm{SNR} = 1$ on, with TPR rising to $0.93$ at $\mathrm{SNR} = 5$.
   Oracle and clustered variants are identical here — no amount of correct group knowledge identifies *which*
   exchangeable member is active.
- At the weaker $\rho = 0.35$ the violation is mild for EN (FDR up to $\approx 0.17$ at high SNR), while both IEN
   variants keep FDR below the target (at most $0.074$ clustered and $0.072$ oracle from $\mathrm{SNR} = 1$ on) and
   reach TPR $0.97\text{–}0.98$ at $\mathrm{SNR} = 5$, against $0.77$ for EN. The group ties concentrate the selection
   on one representative per block.
- The block-level diagnostics in the saved tables stay small ($\mathrm{block\_FDR} \le 0.13$ at
   $\rho = 0.9$ for EN, $\le 0.05$ for IEN), so the method still finds the right *blocks* while the
   coordinate-level FDR fails. T-Rex+GVS is not designed to control a block-level FDR, so this value is
   descriptive and carries no guarantee.

TPR (top) and FDR (bottom) vs. SNR, one panel per $\rho \in \lbrace0.35, 0.9\rbrace$. Note the different FDR
axis scales.

![Block benchmark — INDIVIDUAL](simulation_results/plots/gvs_block_bench_individual.png)

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
