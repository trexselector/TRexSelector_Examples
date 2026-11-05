# Demo 03: DA-TRex+BT (Binary-Tree Dependency-Aware T-Rex) on Block-Diagonal AR(1) Data (Python)

Monte-Carlo (MC) results for **DA-TRex+BT** — the Dependency-Aware (DA) Terminating Random Experiments (T-Rex)
selector (DA-TRex) with the Binary-Tree (BT) group design (`DAMethod.BT`) — on a clean block-diagonal first-order
autoregressive (AR(1)) design (no white-noise padding), sweeping the signal-to-noise ratio (SNR), $\rho$, block size
$Q$, and number of blocks $M$, each across the three hierarchical agglomerative clustering (HAC) linkage methods
(Single / Complete / Average).
Common setup: $n=150$, amplitude $1.0$, target false discovery rate (FDR) $\mathrm{tFDR}=0.2$, $K=20$ random
experiments, $\mathrm{MC}=200$ per grid point, and solvers TLARS / TAFS / TOMP, plus TLARS_STAG (TLARS with the
T-loop stagnation stop on, window 7, see the
[C++ suite README](../../../../../cpp/trex_selector_methods/ad/trex_da/README.md)).

Python port of the C++ demo
[cpp/trex_selector_methods/ad/trex_da/demo_trex_da_03_mc_sim_ar1_blocks](../../../../../cpp/trex_selector_methods/ad/trex_da/demo_trex_da_03_mc_sim_ar1_blocks/README.md)
— same data-generating process (DGP), control parameters, sweep axes, console summaries, and result-file naming. The
numbers below are this Python run's own results, and they agree with the C++ reference up to Monte-Carlo noise
(independent random number generator (RNG) streams).

The greedy solvers use *exchangeable tie-breaking* (`exch_tie_alpha = 0.25` for TAFS/TOMP, `0` for
TLARS/TLARS_STAG). See `Exchangeable_Tie_Breaking_DA_TRex.md` in the TRex_Research documentation.
TAFS additionally runs with its adaptive forward-selection (AFS) correlation parameter `rho_afs = 0.3`
(`0` for TLARS/TOMP), which is why the figures label it `TAFS (rho = 0.3)`.

---

## Setup — the DA-TRex+BT selector

The selector deflates each variable's ordinary relative occurrence $\Phi_{T,L}(j)$ by the penalty
$\Psi_{T,L}^{\mathrm{BT}}(j) \in [1/2, 1]$ built from its most similar competitor within its
variable group,

```math
\Phi^{\mathrm{BT}}_{T,L}(j) := \Psi^{\mathrm{BT}}_{T,L}(j) \cdot \Phi_{T,L}(j),
```

where the **BT group design** obtains the groups $\mathrm{Gr}(j)$ from a **binary tree**. Variables
are clustered by hierarchical agglomerative clustering (HAC) on the dissimilarity
$1 - |\mathrm{corr}(\boldsymbol x_j, \boldsymbol x_{j'})|$ and the dendrogram cut defines disjoint
groups (see [Demo 02](../demo_trex_da_02_mc_sim_ar1_blocks_plus_white/README.md) for the full penalty
formula). The **linkage method** sets the between-cluster distance during agglomeration —
**Single** ($\min$, prone to chaining), **Complete** ($\max$, compact clusters), **Average**
(mean, unweighted pair group method with arithmetic mean (UPGMA)) — and every sweep below runs once per linkage to
probe its influence.

## Setup — data generating process (`dgp_ar1_block_snr`)

The design consists of $M$ statistically independent AR(1) blocks of size $Q$ — and nothing else.
Unlike Demo 02 there is no white-noise padding, so *every* column belongs to a correlated block and
$p = M \cdot Q$. The within-block correlation follows the $Q \times Q$ Toeplitz matrix

```math
\left[\boldsymbol{\Sigma}_{m}(\rho)\right]_{j,k} = \rho^{|j-k|},
\qquad m = 1, \ldots, M,
```

generated per block via the recursion $z_{j} = \rho\, z_{j-1} + \sqrt{1-\rho^{2}}\,\varepsilon_{j}$
with $z_1, \varepsilon_j \sim \mathcal{N}(0,1)$, giving the block-diagonal covariance

```math
\boldsymbol{\Sigma} =
\begin{bmatrix}
\boldsymbol{\Sigma}_1(\rho) & & \\
& \ddots & \\
& & \boldsymbol{\Sigma}_M(\rho)
\end{bmatrix} \in \mathbb{R}^{p \times p},
\qquad p = M \cdot Q.
```

At the base point the problem is *low-dimensional* ($p = 25 \ll n = 150$). The $Q$ sweep pushes it
past $p > n$ ($p = 250$ at $Q = 50$), and the $M$ sweep reaches $p = n = 150$ at $M = 30$.

The block core is exactly the one shown in **Demo 02**'s design-correlation heatmap
([`../dgp_corr_plots/da_ar1_block_white.png`](../dgp_corr_plots/da_ar1_block_white.png)), minus the
white padding that follows it. The `da_ar1_block` figure in
[`../dgp_corr_plots/`](../dgp_corr_plots/) renders the same generator at an illustrative
$p = 1{,}000$ / 5 blocks of 200, where the within-block $\rho^{|j-k|}$ decay makes the blocks all but
invisible — the $\text{small-}Q$ view of Demo 02 is the more informative picture for this demo's base point.

**Support (`OnePerBlock`).** Exactly one representative per AR(1) block,

```math
\mathcal{S} = \{s_1, \ldots, s_M\}, \qquad s_m \in \{(m-1)Q, \ldots, mQ - 1\},
```

(0-based Python indices, while the R suite is 1-based), so $s = M$ and every block carries a signal.

**Linear model and SNR control.** Each trial draws $y = X\beta + \varepsilon$ with active amplitude
$\beta_{s_m} = 1.0$, $\varepsilon_i \stackrel{\text{iid}}{\sim} \mathcal{N}(0, \sigma^2)$, and
$\sigma^2 = \widehat{\mathrm{Var}}(X\beta)/\mathrm{SNR}$.

**Base parameters** (each sweep varies one dimension, ceteris paribus): $M=5$, $Q=5$ ($p=25$,
$s=5$), $\rho=0.7$, $\mathrm{SNR}=2.0$, seed $2026$.

---

## Running the Demo

```bash
python Python/trex_selector_methods/ad/trex_da/demo_trex_da_03_mc_sim_ar1_blocks/demo_trex_da_03_mc_sim_ar1_blocks.py
```

The four parts are toggled by the `RUN_PART_*` flags at the top of the file (all `True`, mirroring the
cpp `if (true)` toggles). Reduce `NUM_MC` to preview. The file inserts both its own folder and the
parent suite directory onto `sys.path` so `dgp_generators` / `da_sim_common` resolve in the demo *and*
in the spawned workers — run it as a file, not piped via `python -`. Monte-Carlo trials run in
parallel through `concurrent.futures.ProcessPoolExecutor` on `TREX_NUM_WORKERS` worker processes (default 8, and `0`
uses every usable core, see `resolve_workers()` in [`../da_sim_common.py`](../da_sim_common.py)).

Afterwards, regenerate the figures from the comma-separated values (CSV) files with
[`generate_plots.sh`](generate_plots.sh), which drives the suite plotting module
[`../trex_da_plt_utils.py`](../trex_da_plt_utils.py) through the repo-root `.venv`
(`pip install matplotlib pandas plotly`).

---

## Output Files

Data tables are written to `simulation_results/data/` (24 files = 12 scenario stems, one
`.txt`+`.csv` pair each):

- `da_trex_mc_da_ar1_blocks_snr_{Single,Complete,Average}.txt` / `.csv`
- `da_trex_mc_da_ar1_blocks_rho_{Single,Complete,Average}.txt` / `.csv`
- `da_trex_mc_da_ar1_blocks_Q_{Single,Complete,Average}.txt` / `.csv`
- `da_trex_mc_da_ar1_blocks_M_{Single,Complete,Average}.txt` / `.csv`

Figures go to `simulation_results/plots/`: one overview of the FDR and the true positive rate (TPR) per sweep ×
linkage (Portable Network Graphics (PNG) and Portable Document Format (PDF), plus interactive Plotly `.html` with
`--plotly`), plus one linkage-comparison grid (Single/Complete/Average columns) per sweep — the four grids embedded
below.

---

## Part 1 — SNR sweep ($\mathrm{SNR} \in \lbrace0.1, 0.2, 0.5, 0.6, 1, 2, 5\rbrace$)

- **FDR stays under the $\mathrm{tFDR}=0.2$ target at every SNR** (worst cell $0.113$, TLARS at
  $\mathrm{SNR}=2$ under Complete linkage) — but visibly higher than the white-noise-diluted Demo 02
  ($\leq 0.027$ over its SNR sweep). Demo 02 runs at its stricter target of $0.1$ and twice the sample size, so the
  gap reflects the target and $n$ as well as the null structure. In an all-block design every false discovery
  candidate is a correlated shadow.
- In TPR, TOMP is the strongest from $\mathrm{SNR}=1$ on ($0.98\text{–}1.0$ from $\mathrm{SNR}=2$), while TLARS and
  TLARS_STAG reach $0.77\text{–}0.84$. TAFS peaks at $\mathrm{SNR}=1\text{–}2$ ($0.85\text{–}0.90$) and then
  *declines* to $\approx 0.70\text{–}0.75$ at $\mathrm{SNR}=5$ — the same non-monotonicity seen in Demo 02.
- Linkage has only a mild effect here, and Complete/Average buy a few TPR points over Single.

![SNR sweep across HAC linkage](simulation_results/plots/da_trex_mc_da_ar1_blocks_snr_linkage_grid.png)

---

## Part 2 — $\rho$ sweep ($\rho \in \lbrace0.0, 0.1, \ldots, 0.9\rbrace$)

- **FDR rises steadily with $\rho$ and TLARS breaks the target at $\rho=0.9$** under Complete and
  Average linkage ($0.260$ / $0.259$), while Single linkage sits right at it ($0.207$). TAFS remains controlled
  ($0.13\text{–}0.15$) and TOMP comfortably so ($\approx 0.07$). TLARS_STAG shows nearly the same values
  ($0.264$ / $0.260$ / $0.211$), so the stagnation stop does not cure this violation. Strong within-block
  correlation makes the block shadows progressively harder to deflate — the block-design counterpart of the
  $\text{high-}\rho$ corner seen in Demos 01 and 07.
- **Single linkage again loses power at low $\rho$**: TPR $0.38\text{–}0.40$ at $\rho \leq 0.1$ vs.
  $0.67\text{–}0.76$ for Complete/Average — with weak within-block correlation the Single-linkage cut
  groups variables by noise. The linkages come close at $\rho \approx 0.4\text{–}0.5$, while Single falls
  behind again at high $\rho$, most for TLARS at $\rho = 0.9$ ($0.52$ vs. $0.67\text{–}0.68$).
- TPR peaks $\text{mid-}\rho$ and falls toward $\rho=0.9$ (TOMP $0.75\text{–}0.79$, TAFS $0.67\text{–}0.73$,
  TLARS and TLARS_STAG $0.52\text{–}0.68$).

![rho sweep across HAC linkage](simulation_results/plots/da_trex_mc_da_ar1_blocks_rho_linkage_grid.png)

---

## Part 3 — Block-size $Q$ sweep ($Q \in \lbrace5, 10, \ldots, 50\rbrace$, $p = 5Q$ grows to $250 > n$)

- FDR is comfortably controlled over the whole sweep (max $0.112$, TLARS under Complete linkage at
  $Q=10$) even as the design crosses into the high-dimensional regime ($p = 250 > n = 150$ at $Q=50$).
- Complete/Average linkage handle growing blocks gracefully (TOMP $0.97\text{–}0.99$,
  TAFS $0.87\text{–}0.92$, TLARS and TLARS_STAG $0.80\text{–}0.88$ throughout), while **Single linkage loses power
  once blocks grow** and sits $0.07\text{–}0.18$ TPR below them from $Q = 15$ on (TOMP $0.85\text{–}0.90$,
  TLARS $0.69\text{–}0.75$) — the same fragility as in Demo 02's $Q$ sweep, where Single trails by up to $0.19$.

![Q sweep across HAC linkage](simulation_results/plots/da_trex_mc_da_ar1_blocks_Q_linkage_grid.png)

---

## Part 4 — Number-of-blocks $M$ sweep ($M \in \lbrace1, 3, 5, 10, 15, 20, 30\rbrace$, $s = M$, $p = 5M$)

- This is the stress dimension. Here $s = M$ grows while $n = 150$ and the amplitude stay fixed.
  **TPR collapses for $M \gtrsim 20$** ($\approx 0.12\text{–}0.19$ at $M=20$, $\leq 0.05$ at $M=30$, where $p = n$),
  while FDR stays below target throughout (max $0.135$, TLARS at $M=20$) — the selector fails *safe*.
- The $M=1$ corner is degenerate ($p=5$, one active variable), so all solvers are weak
  (TPR $\leq 0.29$). TAFS never selects the active variable (TPR $0$) and selects only null variables in
  $5.0\text{–}6.5\,\%$ of the runs (FDR $0.050\text{–}0.065$) — with a single signal there is no occurrence
  spread for the DA deflation to work with.
- The sweet spot sits at $M \approx 5\text{–}10$, where TOMP holds $0.86\text{–}0.98$ TPR with FDR
  $0.02\text{–}0.05$.

![M sweep across HAC linkage](simulation_results/plots/da_trex_mc_da_ar1_blocks_M_linkage_grid.png)

---

## TLARS_STAG

TLARS_STAG (TLARS with the T-loop stagnation stop) matches TLARS in every sweep. Over all $102$ cells its mean FDR
and TPR changes stay below $0.002$, and a single cell loses at most $0.032$ TPR ($\mathrm{SNR} = 0.6$, Complete
linkage). Its largest FDR increase, $0.021$ at $\mathrm{SNR} = 0.6$ under Single linkage, stays far below target. Its
mean T-loop length stays at or below $7.3$ steps in every cell, where TLARS runs to a median of $30$ and up to $43$
steps. It also inherits the TLARS violation at $\rho = 0.9$, so that violation does not come from the long T-loop.
The figures draw TLARS_STAG dashed in the TLARS colour.

---

## Interpretation

- Read this demo against **Demo 02** (same block AR(1) core plus $975$ white-noise columns,
  $n=300$, stricter target $\mathrm{tFDR}=0.1$). The pure block design at half the sample size shows a higher
  realized FDR in $396$ of the $408$ shared cells — culminating in TLARS's $\rho=0.9$ violation — and a lower TPR
  in $353$ of them, with TAFS on the $Q$ sweep as the main exception. So the pure block design is the harder
  problem, although the two demos differ in more than the white padding.
- The linkage story is consistent across both demos. Linkage is irrelevant at the well-correlated base point,
  but **Single linkage is the fragile choice** at low $\rho$ and large $Q$, so Complete or Average are
  the safer defaults.
- Solver ranking on block designs: **TOMP is the most robust** (best TPR *and* lowest FDR in $80$ of
  the $102$ cells), TAFS in between with its high-SNR power decline, TLARS the most FDR-fragile at strong
  within-block correlation, with or without the stagnation stop.
