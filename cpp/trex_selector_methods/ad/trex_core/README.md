# Classical T-Rex Selector: Demonstration Suite

## Overview

This folder contains C++ example programs for the classical **Terminating Random Experiments Selector
(T-Rex Selector)**.

The demos are designed to help first-time users understand three things:

1. how to generate synthetic regression data,
2. how to run the classical T-Rex Selector in different settings,
3. how to interpret the resulting variable selections and simulation summaries.

If you are new to this folder, start with **Demo 01** for a basic single run and then
continue with **Demo 02** for a Monte Carlo (MC) study across several signal-to-noise ratios.

---

## What this folder covers

The examples study sparse variable selection in Gaussian linear regression in both:

- **low-dimensional settings**, where $(n > p)$,
- **high-dimensional settings**, where $(p > n)$.

Across the demos, we vary:

- the sparsity pattern of the true coefficient vector,
- the signal strength,
- the support construction,
- and the computational strategy (standard in-memory or memory-mapped workflows).

The main statistical goal is to recover the active variables while controlling the false discovery rate (FDR).

---

## Statistical model and notation

We work with the classical Gaussian linear model

```math
\boldsymbol{y} = \boldsymbol{X}\boldsymbol{\beta} + \boldsymbol{\varepsilon},
\qquad
\boldsymbol{\varepsilon} \sim \mathcal{N}(0, \sigma^2 \boldsymbol{I}_n).
```

### Notation

- $n$: Number of observations.
- $p$: Number of variables.
- $\boldsymbol{y} \in \mathbb{R}^n$: Response vector.
- $\boldsymbol{X} \in \mathbb{R}^{n \times p}$: Design matrix.
- $\boldsymbol{\beta} \in \mathbb{R}^p$: Regression coefficient vector, $\beta_{j}$ denotes the $j$-th coefficient.
- $\boldsymbol{\varepsilon} \in \mathbb{R}^n$: Gaussian noise vector.
- $s = |\mathcal{A}|$: Number of truly active variables.
- $\mathcal{A} = \lbrace j : \beta_j \neq 0\rbrace$: True support set.
- $\widehat{\mathcal{A}}$: Estimated support set returned by the selector.

### Data-generating variations

Depending on the demo, the simulation setup may use:

- **fixed or random support indices**,
- **deterministic nonzero coefficients** such as $\beta_j = 1$,
- **random nonzero coefficients**, for example drawn from $\mathcal{N}(0,1)$.

### Signal-to-noise ratio

The linear signal-to-noise ratio (SNR) is defined as

```math
\mathrm{SNR} = \frac{\mathrm{Var}(\boldsymbol{X}\boldsymbol{\beta})}{\mathrm{Var}(\boldsymbol{\varepsilon})}.
```

For a fixed design matrix $\boldsymbol{X}$ and coefficient vector $\boldsymbol{\beta}$,
 the noise variance $\sigma^2$ is adjusted to match a desired target SNR.

Typical SNR values used in the demos are

```math
\left\lbrace 0.1, 0.5, 1, 2, 5 \right\rbrace,
```

which range from weak to strong signal settings.

---

## Statistical targets

The selector aims to identify relevant variables while controlling false discoveries.

### False Discovery Rate (FDR)

```math
\mathrm{FDR}
=
\mathbb{E}
\left[
\frac{|\widehat{\mathcal{A}} \setminus \mathcal{A}|}
{\max\{1, |\widehat{\mathcal{A}}|\}}
\right].
```

In the simulations, the target level is called the **target FDR (tFDR)**, typically

```math
\alpha = 0.1.
```

### True Positive Rate (TPR)

```math
\mathrm{TPR}
=
\mathbb{E}
\left[
\frac{|\mathcal{A} \cap \widehat{\mathcal{A}}|}
{\max\{1, |\mathcal{A}|\}}
\right].
```

Informally:

- **FDR** expected proportion of false discoveries among all discoveries on average.
- **True positive rate (TPR)** expected proportion of true positives among all true signals on average.

A good procedure keeps FDR below the target level while achieving high TPR.

---

## Start here

For first use, the following order is recommended:

1. **Demo 01 — Single Run**  
   Best starting point. Shows the basic workflow in one low-dimensional and one high-dimensional example.

2. **Demo 02 — Monte Carlo, Fixed Support**
   Shows how performance changes across SNR values and provides aggregated FDR/TPR output.

3. **Demo 06 — Memory-Mapped Single Run**  
   Useful if you are interested in large-data workflows or settings with limited random-access memory (RAM). To reveal
   its full power, we recommend using a solid-state drive (SSD).

---

## Demonstration structure

Each demo subfolder typically contains:

- **`demo_*.cpp`**: the source file for the demo.
- **`README.md`**: a description of the scenario and the main interpretation.
- **`generate_plots.sh`** (MC demos 02–05, 07, 08): thin wrapper that renders the
  demo's figures via the suite-level [trex_plt_utils.py](trex_plt_utils.py).
- **`simulation_results/`**: generated artifacts only, split into
  - **`data/`** — text summaries and comma-separated values (CSV) tables written by the demo binary,
  - **`plots/`** — figures (png/pdf, html on request) written by the plotting scripts.

Everything under `simulation_results/` is machine-generated and can be
regenerated. Source files (demo code, plot scripts) never live there. This makes
each demo self-contained and easier to inspect independently.

---

## Demo descriptions

| # | Name | Purpose | Setting | Main parameters |
|---|------|---------|---------|-----------------|
| **01** | Single Run | Demonstrates basic T-Rex usage | One low-dimensional and one high-dimensional scenario | $n \in \lbrace150, 5{,}000\rbrace$, $p \in \lbrace300, 1{,}000\rbrace$, $s = 6$, SNR = 1.0, tFDR = 0.1 |
| **02** | MC: Fixed Support | Studies FDR/TPR over an SNR range for 14 base solvers | Moderate dimension, fixed random support | $n = 300$, $p = 1{,}000$, $s = 10$ (fixed, random indices, seed 24), SNR $\in \lbrace0.1, 0.5, 1, 2, 5\rbrace$, tFDR = 0.1, MC = 200 |
| **03** | MC: Variable Support | Studies FDR/TPR/L/T when support indices are redrawn per trial | Moderate dimension, variable support indices | $n = 300$, $p = 1{,}000$, $s = 10$ (cardinality fixed, indices vary per trial), 21-point SNR grid $\lbrace0.1,\ldots,2.0, 5.0\rbrace$, tFDR = 0.1, MC = 200 |
| **04** | MC: L-loop Strategies | Compares the 6 L-loop strategy types across 4 base solvers (TLARS/TOMP/TAFS and TLARS_STAG, TLARS with the stagnation stop) | Moderate dimension, random support | $n = 300$, $p = 1{,}000$, $s = 10$, 21-point SNR grid, tFDR = 0.1, MC = 200, 8 strategy rows (incl. SKIPL 5p/10p/20p) × 4 solvers (one file pair each) |
| **05** | MC: Dummy Distributions | Compares dummy distribution shapes across 3 base solvers (TLARS/TOMP/TAFS, STANDARD L-loop fixed) | Moderate dimension, random support | $n = 300$, $p = 1{,}000$, $s = 10$, SNR $\in \lbrace0.1, 0.2, 0.5, 0.6, 1, 2, 5\rbrace$, tFDR = 0.1, MC = 200, 12 distribution rows (Normal … UnifSphere_d5) × 3 solvers (one file pair each) |
| **06** | Memory-Mapped Single Run | Demonstrates memory-mapped (mmap) workflows | Larger data and reduced RAM usage | Two scenarios ($n=5{,}000,p=1{,}000$ and $n=1{,}000,p=5{,}000$): in-memory X + D mmap, and full X+D mmap pipeline |
| **07** | MC: Memory-Mapped | Studies FDR/TPR under memory-mapped execution (TLARS) | Moderate dimension, mmap pipeline | $n = 300$, $p = 1{,}000$, $s = 10$, SNR $\in \lbrace0.1,0.5,1,2,5\rbrace$, MC = 200 (Demo C and Demo D), mmap storage on disk |
| **08** | MC: Scalability | Measures runtime and memory as $n$ and $p$ grow, up to designs larger than RAM (TLARS/TOMP/TAFS) | Fully memory-mapped pipeline, out-of-core | Base block $n \in \lbrace300, \ldots, 5{,}000\rbrace$, $p \in \lbrace1{,}000, \ldots, 150{,}000\rbrace$ (dummy cap 10) and large-p block $n = 300$, $p \in \lbrace100{,}000, 1{,}000{,}000\rbrace$ (dummy cap 5), $s = 10$, SNR = 1, tFDR = 0.1, MC = 10 sequential trials per grid point |
| **09** | MC: Target FDR | Measures the achieved FDR and TPR as the target FDR is swept (TLARS/TOMP/TAFS) | Moderate dimension, variable support indices, paired data along tFDR | $n = 300$, $p = 1{,}000$, $s = 10$, SNR $\in \lbrace0.5, 1, 2, 5\rbrace$, tFDR $\in \lbrace0.01, \ldots, 0.5\rbrace$ (11 values), MC = 200 |
| **10** | MC: Sample-Size Ratio | Measures FDR and TPR against $n / (s \ln p)$ at tFDR = 0.1 (TLARS/TOMP/TAFS) | Varying $n$ for two $p$, variable support indices | $p \in \lbrace1{,}000, 10{,}000\rbrace$, $n = \operatorname{round}(c \, s \ln p)$ with $c \in \lbrace1, \ldots, 10\rbrace$ (10 values), $s = 10$, SNR $\in \lbrace0.5, 1, 2\rbrace$, MC = 200 |

---

## Folder contents

```txt
trex_core/
  ├── README.md
  ├── CMakeLists.txt
  ├── trex_sim_utils.hpp                <- shared MC / output helpers (C++)
  ├── trex_plt_utils.py                 <- shared plotting module (all tidy-CSV demos)
  │
  ├── demo_trex_01_single_run/
  │   ├── demo_trex_01_single_run.cpp
  │   ├── README.md
  │   └── simulation_results/
  │       └── data/
  │           ├── d01_trex_basic_n150_p300.txt
  │           └── d01_trex_basic_n5000_p1000.txt
  │
  ├── demo_trex_02_mc_sim_fixed_support/
  │   ├── demo_trex_02_mc_sim_fixed_support.cpp
  │   ├── README.md
  │   ├── generate_plots.sh
  │   └── simulation_results/
  │       ├── data/
  │       │   ├── demo_trex_02_mc_sim_fixed_support_trex_results_n300_p1000_stagnation_window_7.txt
  │       │   └── demo_trex_02_mc_sim_fixed_support_trex_results_n300_p1000_stagnation_window_7.csv
  │       └── plots/
  │           ├── ..._fdr_tpr_vs_snr.{png,pdf}
  │           └── ..._fdr_tpr_vs_snr_grouped.{png,pdf}
  │
  ├── demo_trex_03_mc_sim_variable_support/
  │   ├── demo_trex_03_mc_sim_variable_support.cpp
  │   ├── README.md
  │   ├── generate_plots.sh
  │   └── simulation_results/          <- data/ + plots/
  │
  ├── demo_trex_04_mc_sim_lloop_strategies/
  │   ├── demo_trex_04_mc_sim_lloop_strategies.cpp
  │   ├── README.md
  │   ├── generate_plots.sh
  │   └── simulation_results/          <- data/ + plots/
  │
  ├── demo_trex_05_mc_sim_dummy_distributions/
  │   ├── demo_trex_05_mc_sim_dummy_distributions.cpp
  │   ├── README.md
  │   ├── generate_plots.sh
  │   └── simulation_results/          <- data/ + plots/
  │
  ├── demo_trex_06_mmap/
  │   ├── demo_trex_06_mmap.cpp
  │   ├── README.md
  │   └── simulation_results/          <- data/
  │
  ├── demo_trex_07_mc_sim_mmap/
  │   ├── demo_trex_07_mc_sim_mmap.cpp
  │   ├── generate_plots.sh
  │   ├── README.md
  │   └── simulation_results/          <- data/ + plots/
  │
  ├── demo_trex_08_mc_sim_scalability/
  │   ├── demo_trex_08_mc_sim_scalability.cpp
  │   ├── README.md
  │   ├── generate_plots.sh
  │   └── simulation_results/          <- data/ + plots/
  │
  ├── demo_trex_09_mc_sim_target_fdr/
  │   ├── demo_trex_09_mc_sim_target_fdr.cpp
  │   ├── README.md
  │   ├── generate_plots.sh
  │   └── simulation_results/          <- data/ + plots/
  │
  └── demo_trex_10_mc_sim_dt_ratio/
      ├── demo_trex_10_mc_sim_dt_ratio.cpp
      ├── README.md
      ├── generate_plots.sh
      └── simulation_results/          <- data/ + plots/
```

---

## Example results

The following comes from **Demo 02**, which evaluates performance across several SNR levels for multiple base
solvers within the T-Rex framework. The table quotes the committed TLARS results (200 Monte Carlo trials per SNR
level), and the demo's `simulation_results/` files hold all 14 solvers.

| SNR | FDR (TLARS) | TPR (TLARS) |
| ----- | ------------- | ------------- |
| 0.1 | 0.0200      | 0.0035      |
| 0.5 | 0.0342      | 0.2730      |
| 1.0 | 0.0367      | 0.7220      |
| 2.0 | 0.0317      | 0.9820      |
| 5.0 | 0.0317      | 1.0000      |

### Interpretation

- FDR remains below the target level $\text{tFDR} = 0.1$ across all listed SNR values.
- TPR increases strongly as the signal becomes easier to detect.
- At very low SNR, recovery is difficult.
- At moderate to high SNR, the procedure recovers most or all active variables.

This is the expected pattern for a method that prioritizes controlled false discoveries
while gaining power as signal strength increases.

---

## Building the demos

From the C++ workspace root:

```bash
cd TRexSelector_Examples/cpp
cmake -S . -B build/debug -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_PREFIX_PATH="<path-to-TRexSelector>/cpp/install"
cmake --build build/debug
```

The executables are written under `build/debug/bin/`, mirroring the source-tree layout:

```txt
build/debug/bin/trex_selector_methods/ad/trex_core/<demo_folder>/<demo_name>
```

---

## Running a demo

For example, to run Demo 02:

```bash
./build/debug/bin/trex_selector_methods/ad/trex_core/demo_trex_02_mc_sim_fixed_support/demo_trex_02_mc_sim_fixed_support
```

The program writes its output to the console and also stores result files in the corresponding demo folder.
From here the results can be used in a graphics library of choice, such as Python's `matplotlib`, R's `ggplot2`,
Cpp's `matplot++` or any other plotting tool.

Example output location:

```txt
demo_trex_02_mc_sim_fixed_support/simulation_results/data/
  ├── demo_trex_02_mc_sim_fixed_support_trex_results_n300_p1000_stagnation_window_7.txt
  └── demo_trex_02_mc_sim_fixed_support_trex_results_n300_p1000_stagnation_window_7.csv
```

### Output files

Each demo's `simulation_results/` folder holds machine-generated artifacts only:

- **`data/`** — `.txt` files (human-readable summaries) and `.csv` files
  (tidy tables for plotting and post-processing), written by the demo binary.
- **`plots/`** — figures (`.png`/`.pdf`) rendered from the `data/` CSVs by each
  demo's `generate_plots.sh` via the shared
  [trex_plt_utils.py](trex_plt_utils.py). With `--plotly` the same script also writes
  interactive Plotly `.html` twins, which are **not committed** (each embeds a full copy
  of plotly.js).

---

## Notes for new users

- Start with **Demo 01** if you only want to verify that the pipeline works.
- Use **Demo 02** and **Demo 03** to study statistical behavior across repeated simulations.
- Use **Demo 06** and **Demo 07** if you are interested in larger problem sizes or disk-backed workflows.
- Check the local `README.md` inside each demo folder for scenario-specific details.

---

## References

- The main T-Rex Selector implementation and methodological background are documented in the parent project.
- See [TRexSelector/README.md](../../../../README.md) for architecture, dependencies, and package-level information.
