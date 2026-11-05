# Simulations `TREX+GVS` - Part 01

## Scenarios inspired by testbeds used for the Elastic-Network in Zhu & Hastie (2005)

The following design was proposed in Zhu & Hastie (2005) to evaluate the Elastic Net's ability to
select the entirety of the highly correlated block variables and underline the grouping propoerty
of the elastic network.
We adapt it to evaluate the `TREX+GVS` selector with selector methods `EN` and `IEN`.

We consider the high-dimensional linear regression model:
$$
\boldsymbol{y} = \boldsymbol{X} \boldsymbol{\beta} + \boldsymbol{\varepsilon},
$$
$\boldsymbol{y} \in \mathbb{R}^{n}$ is the response vector,
$\boldsymbol{X} \in \mathbb{R}^{n \times p}$ is the predictor matrix,
and homoscedastic noise $\boldsymbol{\varepsilon} \in \mathbb{R}^{n}$ with
$\boldsymbol{\varepsilon} \sim \mathcal{N}(\boldsymbol{0}, \sigma^2_{\varepsilon} \boldsymbol{I}_n)$.
The coefficient vector
$\boldsymbol{\beta} \in \mathbb{R}^p$ is sparse true with cardinality
$\text{card}\{\boldsymbol{\beta}\}=s$.

The predictors are structured into $K=3$ active, highly correlated blocks, $\{ G_k \}_{k=1}^{K}$,
each of size $m=50$, driven by independent latent factors.
The feature space is index with $j = 1, \ldots, p$, where the first $p_{\text{active}}$ variables
belong to the active groups, and the remaining $p_{\text{noise}} = p - K \cdot m$ variables are
independent noise.

1. **Latent Factors:** Let $Z_k \sim \mathcal{N}(0, 1)$ for $k \in \{1, 2, 3\}$.
2. **Active Grouped Predictors ($j \le 150$):**
   For each block $G_k$, the predictors are generated as:
   $$
    X_j = Z_k + \xi_j,
   $$
   $$
    \xi_j \sim \mathcal{N}(0, \sigma_x^2),
   $$
   where $\sigma_x^2 = 0.01$. This induces a near-perfect intra-group correlation of
   $\rho = \frac{1}{1 + 0.01} \approx 0.99$.
3. **Null Predictors ($j > 150$):**
   $X_j \sim \mathcal{N}(0, 1)$.
4. **True Support ($\beta$):**
   All variables within the three latent groups are genuinely active and contribute equally to the
   response. The coefficients within $\boldsymbol{\beta}$ are
   $$
   \beta_j = \begin{cases}
        3 & \text{if } j \le 150, \\
        0 & \text{if } j > 150.
    \end{cases}
   $$

## Concepts for the Monte Carlo Demos

### Part 1: Single-Run Validation \& Visualization

* **Objective:** Confirm the integrity of the data generation and demonstrate application of the
  T-Rex GVS selector.
* **Execution:** Generate one instance of the $200 \times 500$ matrix.
  Plot the correlation matrix `cor(X_train)` to visually confirm the three dense $50 \times 50$
  blocks on the diagonal.
* **Evaluation:** Run `TRexSelector` with `method = "trex+GVS"` for both `EN` and `IEN`.
  Output the single-run TPP and FDP to the console.

---

### Part 2: The Signal-to-Noise Ratio (SNR) Sweep (1D)

* **Objective:** Investigate the robustness under signal-to-noise (SNR) variations.
* **Execution:** Hold the block structure and $\sigma_x^2 = 0.01$ constant.
  We consider a sweep in linear $\text{SNR} \in \{0.1, 0.2, 0.5, 1, 2, 5 \}$ according to
  $$
    \sigma_{\varepsilon} = \sqrt{\frac{\text{Var}(\boldsymbol{X} \boldsymbol{\beta})}{\text{SNR}}}.
  $$
* **Evaluation:** Track `TREX+GVS`'s behavior as function of the SNR.

Although, we rely on the empirical signal power $\text{Var}(\boldsymbol{X} \boldsymbol{\beta})$ to
adjust the SNR, we present for the sake of completness also the theoretical quantities for this
scenario.
For a particular
$$
    X_{j} = Z_{k} + \varepsilon_{k},
$$
we have the expectation
$$
\mathbb{E}[X_j] = \mathbb{E}\left[ Z_k \right] + \mathbb{E} \left[\varepsilon_{k} \right],
$$
variance
$$
\text{Var}[X_j] = \text{Var}[Z_k] + \text{Var}[\varepsilon_j] = 1 +\sigma^{2}_x,
$$
and covariance
$$
\text{Cov}[X_i, X_j] = \text{Cov}[Z_k + \varepsilon_i, Z_k + \varepsilon_j]
= \mathbb{E}\left[ (Z_k + \varepsilon_i) (Z_k + \varepsilon_j) \right]
= \mathbb{E}[Z^{2}_k] = 1.
$$
For $\boldsymbol{\beta}_{G_{k}} = c \boldsymbol{1}_{m}$ it follows
$$
\text{Var}[\boldsymbol{X}_{G_k} \boldsymbol{\beta}_{G_k}] 
= \boldsymbol{\beta}^{\top}_{G_k} \boldsymbol{\Sigma}_{G_{k}} \boldsymbol{\beta}_{G_k}
= c^2 \boldsymbol{1}^{\top}_{m} \Sigma_{G_k} \boldsymbol{1}_{m}
= c^2 (m ( 1 + \sigma^{2}_{x}) + m(m - 1)) = c^2 m (m + \sigma_{x}^{2}).
$$
Thus, we can express the desired ratio also as
$$
    \sigma_{\varepsilon} = \sqrt{  \frac{K c^2 m (m + \sigma_{x}^{2})}{\text{SNR}} }.
$$

### Simulation based on Zhu & Hastie (2005) for the `TREX+GVS` selector: Monte Carlo — SNR sweep

`Results — EN`: We examine $n=200$, $p=500$, $\alpha := \text{tFDR} = 0.1$,
$\sigma_{x} = 0.1 \equiv \rho \sim 0.99$,
the support is chosen according to the three active groups with:
$s_{G_1} \in \{1, \ldots, 50 \}$,
$s_{G_2} \in \{51, \ldots, 100 \}$,
$s_{G_3} \in \{101, \ldots, 150 \}$,
the GVS type is `EN`, $200$ Monte Carlo trials are conducted per SNR grid point:

| SNR    | mean_FDP | mean_TPP | sd_FDP | sd_TPP |
|--------|----------|----------|--------|--------|
| 0.1000 | 0.0705   | 0.4690   | 0.0840 | 0.3641 |
| 0.2000 | 0.1031   | 0.8574   | 0.0787 | 0.2480 |
| 0.5000 | 0.1151   | 0.9984   | 0.0248 | 0.0231 |
| 1.0000 | 0.1176   | 1.0000   | 0.0232 | 0.0000 |
| 2.0000 | 0.1175   | 1.0000   | 0.0247 | 0.0000 |
| 5.0000 | 0.1178   | 1.0000   | 0.0250 | 0.0000 |

`Results — IEN`:
For comparison we the same evaluation with GVS type `IEN`:

| SNR    | mean_FDP | mean_TPP | sd_FDP | sd_TPP |
|--------|----------|----------|--------|--------|
| 0.1000 | 0.0000   | 0.0067   | 0.0000 | 0.0061 |
| 0.2000 | 0.0000   | 0.0126   | 0.0000 | 0.0056 |
| 0.5000 | 0.0000   | 0.0180   | 0.0000 | 0.0035 |
| 1.0000 | 0.0000   | 0.0193   | 0.0000 | 0.0021 |
| 2.0000 | 0.0000   | 0.0198   | 0.0000 | 0.0016 |
| 5.0000 | 0.0000   | 0.0202   | 0.0000 | 0.0012 |

---

### Part 3: The Intra-Group Correlation ($\rho$) Sweep (1D)

* **Objective:** Map the phase transition of the hierarchical clustering due to changes in $\rho$
  for the `TREX+GVS` selector.
* **Execution:** The SNR is hold constant and we sweep the within-group noise $\sigma_x^2$ to
  effectively sweep the intra-group correlation $\rho \in [0.1, 0.99]$.
* **Evaluation:** As $\rho$ drops, the blocks become diffuse. We identify the critical $\rho$
  threshold where the single-linkage clustering shatters the groups, causing the Elastic Net to
  degenerate into Lasso-like behavior (evidenced by a sharp drop in TPP).

### Simulation based on Zhu & Hastie (2005) for the `TREX+GVS` selector: Monte Carlo — $\rho$ sweep

We examine $n=200$, $p=500$, $\alpha := \text{tFDR} = 0.1$,
$\text{SNR} = 2.0$,
the support is chosen according to the three active groups with:
$s_{G_1} \in \{1, \ldots, 50 \}$,
$s_{G_2} \in \{51, \ldots, 100 \}$,
$s_{G_3} \in \{101, \ldots, 150 \}$,
the GVS type is `EN`, $200$ Monte Carlo trials are conducted per SNR grid point.
The sweep goes over $\rho$.

`Results — EN`:

| rho  | mean_FDP | mean_TPP | sd_FDP | sd_TPP |
|------|----------|----------|--------|--------|
| 0.10 | 0.0673   | 0.4436   | 0.0320 | 0.1042 |
| 0.20 | 0.0841   | 0.8675   | 0.0243 | 0.0597 |
| 0.30 | 0.0913   | 0.9735   | 0.0230 | 0.0261 |
| 0.50 | 0.0942   | 0.9995   | 0.0225 | 0.0029 |
| 0.70 | 0.0946   | 1.0000   | 0.0228 | 0.0000 |
| 0.80 | 0.0944   | 1.0000   | 0.0225 | 0.0000 |
| 0.90 | 0.0939   | 1.0000   | 0.0223 | 0.0000 |
| 0.95 | 0.0942   | 1.0000   | 0.0226 | 0.0000 |
| 0.99 | 0.1174   | 1.0000   | 0.0245 | 0.0000 |

`Results — IEN`:
For comparison we the same evaluation with GVS type `IEN`:

| rho  | mean_FDP | mean_TPP | sd_FDP | sd_TPP |
|------|----------|----------|--------|--------|
| 0.10 | 0.0435   | 0.3413   | 0.0268 | 0.0798 |
| 0.20 | 0.0302   | 0.7057   | 0.0158 | 0.0730 |
| 0.30 | 0.0287   | 0.8730   | 0.0132 | 0.0503 |
| 0.50 | 0.0231   | 0.9823   | 0.0097 | 0.0175 |
| 0.70 | 0.0229   | 0.9996   | 0.0074 | 0.0015 |
| 0.80 | 0.0258   | 1.0000   | 0.0082 | 0.0000 |
| 0.90 | 0.0292   | 1.0000   | 0.0081 | 0.0000 |
| 0.95 | 0.0307   | 1.0000   | 0.0080 | 0.0000 |
| 0.99 | 0.0000   | 0.0198   | 0.0000 | 0.0016 |

---

### Part 4: 2D Phase Transition Sweep (SNR $\times \rho$)

* **Objective:** Comprehensive picture of phase-transitions towards the performance of the
  `TREX+GVS` within the presented data scenario.
* **Execution:** Perform a 2D grid search across $\sigma_y$ (y-axis) and $\rho$ (x-axis).
* **Evaluation:** Output combined matrices (like `matTPP_combined` and `matFDP_combined`) to
  visually trace the exact non-linear boundaries separating the `Safe Zone` (FDP $\le 0.10$) from
  the `Failure Zone`.

#### Elastic Network [EN] Selector

mean_TPP [EN]  (rows: SNR, cols: rho):

|          | rho=0.30 | rho=0.50 | rho=0.70 | rho=0.90 | rho=0.95 | rho=0.99
|----------|----------|----------|----------|----------|----------|----------
| snr=0.20 | 0.164    | 0.478    | 0.697    | 0.821    | 0.830    | 0.858
| snr=0.50 | 0.660    | 0.925    | 0.985    | 0.997    | 0.998    | 0.998
| snr=1.00 | 0.902    | 0.992    | 0.999    | 1.000    | 1.000    | 1.000
| snr=2.00 | 0.974    | 0.999    | 1.000    | 1.000    | 1.000    | 1.000
| snr=5.00 | 0.993    | 1.000    | 1.000    | 1.000    | 1.000    | 1.000

mean_FDP [EN]  (rows: SNR, cols: rho):

|          | rho=0.30 |rho=0.50| rho=0.70| rho=0.90| rho=0.95 |rho=0.99
|----------|----------|--------|---------|---------|----------|----------
| snr=0.20 | 0.050    |0.062   | 0.072   | 0.078   | 0.078    |0.105
| snr=0.50 | 0.074    |0.085   | 0.091   | 0.093   | 0.093    |0.115
| snr=1.00 | 0.084    |0.093   | 0.095   | 0.095   | 0.094    |0.117
| snr=2.00 | 0.091    |0.094   | 0.095   | 0.094   | 0.094    |0.117
| snr=5.00 | 0.094    |0.095   | 0.095   | 0.095   | 0.094    |0.117

#### Informed Elastic Network (IEN) Selector

mean_TPP [IEN] (rows: SNR, cols: rho):

|          | rho=0.30| rho=0.50| rho=0.70| rho=0.90| rho=0.95| rho=0.99
|----------|---------|---------|---------|---------|---------|-----------
| snr=0.20 | 0.098   | 0.265   | 0.446   | 0.619   | 0.657   | 0.013
| snr=0.50 | 0.454   | 0.727   | 0.901   | 0.982   | 0.987   | 0.018
| snr=1.00 | 0.721   | 0.923   | 0.989   | 1.000   | 1.000   | 0.019
| snr=2.00 | 0.873   | 0.982   | 1.000   | 1.000   | 1.000   | 0.020
| snr=5.00 | 0.951   | 0.998   | 1.000   | 1.000   | 1.000   | 0.020

mean_FDP [IEN] (rows: SNR, cols: rho):

|          | rho=0.30 |rho=0.50| rho=0.70| rho=0.90| rho=0.95| rho=0.99
|----------|----------|--------|---------|---------|---------|-----------
| snr=0.20 | 0.040    |0.037   | 0.039   | 0.037   | 0.036   | 0.000
| snr=0.50 | 0.046    |0.047   | 0.044   | 0.044   | 0.046   | 0.000
| snr=1.00 | 0.038    |0.039   | 0.032   | 0.037   | 0.039   | 0.000
| snr=2.00 | 0.029    |0.023   | 0.023   | 0.029   | 0.031   | 0.000
| snr=5.00 | 0.017    |0.012   | 0.017   | 0.020   | 0.021   | 0.000
