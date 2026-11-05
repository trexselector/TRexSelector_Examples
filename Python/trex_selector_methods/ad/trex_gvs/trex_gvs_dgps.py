# ==============================================================================
# trex_gvs_dgps.py
# ==============================================================================
#
# Consolidated data-generating processes for the T-Rex+GVS demonstration suite
# (mirrors cpp trex_gvs_dgps.hpp and R trex_gvs_dgps.R). One section per scenario;
# every demo imports this single module.
#
#   Section 01 — Hastie equicorrelated groups   -> demo_trex_gvs_01_mc_sim_hastie_en_blocks
#   Section 02 — Scattered-grouped              -> demo_trex_gvs_02_mc_sim_scattered_blocks
#   Section 03 — Mixed blocks (unequal, trap)   -> demo_trex_gvs_03_mc_sim_mixed_blocks
#   Section 04 — Negative traps                 -> demo_trex_gvs_04_mc_sim_neg_traps
#   Section 05 — Heavy-tailed (t3) equi blocks  -> demo_trex_gvs_05_mc_sim_t3_blocks
#   Section 06 — AR(1) blocks                   -> demo_trex_gvs_06_mc_sim_ar1_blocks
#   Section 07 — ARMA mixed-structure blocks    -> demo_trex_gvs_07_mc_sim_arma_blocks
#   Section 08 — Block-equicorrelated benchmark -> demo_trex_gvs_08_mc_sim_block_bench
#
# All noise is SNR-calibrated: sigma_y = sqrt(var(signal) / snr), with the n-1
# (ddof=1) sample-variance denominator to match the R ports and the C++
# make_snr_response (Section 08 uses the biased n denominator, as in cpp).
# Support / group indices are 0-based throughout (Python convention).
# ==============================================================================

import math

import numpy as np

_SD_X_DEFAULT = math.sqrt(0.01)   # within-group noise sd -> rho ~ 0.99


def _snr_sigma(signal, snr, ddof=1):
    """Noise sd calibrated to a target SNR = Var(signal) / sigma^2."""
    return math.sqrt(np.var(signal, ddof=ddof) / snr)


# ==============================================================================
# Section 01 — DGP: hastie
# ==============================================================================

def dgp_hastie_snr(n, p=500, snr=2.0, sd_x=_SD_X_DEFAULT, seed=None):
    """Hastie (Zou & Hastie 2005, Example 4) equicorrelated-groups DGP.

    Three latent-factor groups of 50 variables each (columns 0-49, 50-99,
    100-149), all active with beta = 3, plus p-150 i.i.d. background columns.
    Each grouped column is X_j = Z_g + sd_x * xi, giving within-group correlation
    rho = 1 / (1 + sd_x^2). SNR-calibrated noise. prior_groups labels the three
    blocks 0-2 and every background column as its own singleton group.
    """
    if p < 150:
        raise ValueError("'p' must be at least 150 to accommodate the 3 groups.")
    rng = np.random.default_rng(seed)

    beta = np.zeros(p)
    beta[:150] = 3.0

    X = np.zeros((n, p))
    Z1 = rng.standard_normal(n)
    Z2 = rng.standard_normal(n)
    Z3 = rng.standard_normal(n)
    X[:, 0:50]    = Z1[:, None] + rng.standard_normal((n, 50)) * sd_x
    X[:, 50:100]  = Z2[:, None] + rng.standard_normal((n, 50)) * sd_x
    X[:, 100:150] = Z3[:, None] + rng.standard_normal((n, 50)) * sd_x
    if p > 150:
        X[:, 150:p] = rng.standard_normal((n, p - 150))

    signal = X @ beta
    sigma_y = _snr_sigma(signal, snr)
    y = signal + rng.standard_normal(n) * sigma_y

    # Oracle groups: 0-2 for the three blocks, background columns as singletons
    prior_groups = np.concatenate([np.repeat(np.arange(3), 50),
                                   np.arange(3, p - 147)])

    return dict(X=X, y=y, beta=beta, prior_groups=prior_groups, n=n, p=p,
                s=150, snr=snr, sigma_y=sigma_y, sd_x=sd_x)


# ==============================================================================
# Section 02 — DGP: scattered_grouped
# ==============================================================================

def dgp_scattered_grouped_snr(n, p=500, snr=2.0, sd_x=_SD_X_DEFAULT,
                              n_groups=3, group_size=50, seed=None):
    """Scattered-grouped DGP: n_groups equicorrelated groups whose members are
    randomly distributed across the p columns (rather than contiguous). Each
    group shares a latent factor (within-group rho = 1/(1+sd_x^2)); all group
    members are active (beta = 3). SNR-calibrated noise.
    """
    if n_groups * group_size > p:
        raise ValueError("'n_groups * group_size' must not exceed 'p'.")
    rng = np.random.default_rng(seed)

    # Random partition of the column indices into groups and noise.
    perm = rng.permutation(p)
    group_indices = [np.sort(perm[k * group_size:(k + 1) * group_size])
                     for k in range(n_groups)]
    noise_indices = np.sort(perm[n_groups * group_size:])

    beta = np.zeros(p)
    X = np.zeros((n, p))
    for gi in group_indices:
        Zk = rng.standard_normal(n)
        X[:, gi] = Zk[:, None] + rng.standard_normal((n, gi.size)) * sd_x
        beta[gi] = 3.0
    if noise_indices.size > 0:
        X[:, noise_indices] = rng.standard_normal((n, noise_indices.size))

    signal = X @ beta
    sigma_y = _snr_sigma(signal, snr)
    y = signal + rng.standard_normal(n) * sigma_y

    return dict(X=X, y=y, beta=beta, n=n, p=p, s=n_groups * group_size,
                snr=snr, sigma_y=sigma_y, sd_x=sd_x, n_groups=n_groups,
                group_size=group_size, group_indices=group_indices,
                noise_indices=noise_indices)


# ==============================================================================
# Section 03 — DGP: mixed_blocks
# ==============================================================================

def _distribute_gaps(rng, p, block_sizes):
    """Randomly distribute p - sum(block_sizes) noise columns across the
    (num_blocks + 1) gaps, adding one to each internal gap to guarantee block
    separation. Mirrors the R rmultinom + internal-gap +1 construction.
    Returns (block_order, gaps) with gaps summing to the noise budget.
    """
    num_blocks = len(block_sizes)
    total_grouped = int(np.sum(block_sizes))
    block_order = rng.permutation(num_blocks)

    num_noise = p - total_grouped
    rem_noise = num_noise - (num_blocks - 1)
    gaps = rng.multinomial(rem_noise, np.ones(num_blocks + 1) / (num_blocks + 1))
    gaps[1:num_blocks] += 1   # internal gaps (R indices 2..num_blocks) get +1
    return block_order, gaps


def dgp_mixed_blocks_snr(n, p=500, snr=2.0, sd_x=_SD_X_DEFAULT, seed=None):
    """Mixed-blocks DGP: 3 active equicorrelated blocks (sizes 20, 50, 80;
    beta = 3, s = 150) plus 1 inactive trap block (size 65), randomly ordered
    and separated by white-noise gaps. Within-block rho = 1/(1+sd_x^2).
    SNR-calibrated noise.
    """
    rng = np.random.default_rng(seed)
    block_sizes = np.array([20, 50, 80, 65])
    is_active = np.array([True, True, True, False])
    num_blocks = len(block_sizes)

    block_order, gaps = _distribute_gaps(rng, p, block_sizes)

    X = np.zeros((n, p))
    beta = np.zeros(p)
    latent = [rng.standard_normal(n) for _ in range(num_blocks)]
    cur = 0
    block_indices = [None] * num_blocks

    for i in range(num_blocks + 1):
        g = int(gaps[i])
        if g > 0:
            X[:, cur:cur + g] = rng.standard_normal((n, g))
            cur += g
        if i < num_blocks:
            b = int(block_order[i])
            bs = int(block_sizes[b])
            X[:, cur:cur + bs] = latent[b][:, None] + rng.standard_normal((n, bs)) * sd_x
            if is_active[b]:
                beta[cur:cur + bs] = 3.0
            block_indices[b] = np.arange(cur, cur + bs)
            cur += bs

    signal = X @ beta
    sigma_y = _snr_sigma(signal, snr)
    y = signal + rng.standard_normal(n) * sigma_y

    return dict(X=X, y=y, beta=beta, n=n, p=p,
                s=int(np.sum(block_sizes[is_active])), snr=snr, sigma_y=sigma_y,
                sd_x=sd_x, block_sizes=block_sizes, is_active=is_active,
                block_order=block_order, block_indices=block_indices)


# ==============================================================================
# Section 04 — DGP: neg_traps
# ==============================================================================

def dgp_neg_traps_snr(n, p=500, snr=2.0, sd_x=_SD_X_DEFAULT, seed=None):
    """Negative-traps DGP with an active sign-flipped group and two inactive
    sign-flipped traps (all 0-based column ranges):

      cols   0-99 : active,  +Z1 / -Z1, beta = +3 / -3  (s = 100)
      cols 100-199: Trap 1,  +Z2 / -Z2, beta = 0
      cols 200-299: white noise
      cols 300-359: Trap 2,  +Z3 / -Z3, beta = 0
      cols 360-p  : white noise

    Sign-flipped halves induce strong negative within-group correlation.
    SNR-calibrated noise.
    """
    rng = np.random.default_rng(seed)
    Z1 = rng.standard_normal(n)
    Z2 = rng.standard_normal(n)
    Z3 = rng.standard_normal(n)

    X = np.zeros((n, p))
    beta = np.zeros(p)

    # Group 1: active (0-99)
    X[:, 0:50]   = Z1[:, None] + rng.standard_normal((n, 50)) * sd_x
    X[:, 50:100] = -Z1[:, None] + rng.standard_normal((n, 50)) * sd_x
    beta[0:50] = 3.0
    beta[50:100] = -3.0

    # Group 2: Trap 1 (100-199)
    X[:, 100:150] = Z2[:, None] + rng.standard_normal((n, 50)) * sd_x
    X[:, 150:200] = -Z2[:, None] + rng.standard_normal((n, 50)) * sd_x

    # White noise (200-299)
    X[:, 200:300] = rng.standard_normal((n, 100))

    # Group 3: Trap 2 (300-359)
    X[:, 300:330] = Z3[:, None] + rng.standard_normal((n, 30)) * sd_x
    X[:, 330:360] = -Z3[:, None] + rng.standard_normal((n, 30)) * sd_x

    # White noise (360-p)
    if p > 360:
        X[:, 360:p] = rng.standard_normal((n, p - 360))

    signal = X @ beta
    sigma_y = _snr_sigma(signal, snr)
    y = signal + rng.standard_normal(n) * sigma_y

    return dict(X=X, y=y, beta=beta, n=n, p=p, s=100, snr=snr,
                sigma_y=sigma_y, sd_x=sd_x)


# ==============================================================================
# Section 05 — DGP: t3_equi_blocks
# ==============================================================================

def dgp_t3_equi_blocks_snr(n, p=500, snr=2.0, rho=0.75, df=3, seed=None):
    """Heavy-tailed equicorrelated-blocks DGP: same 4-block geometry as
    dgp_mixed_blocks_snr (sizes 20/50/80/65; 3 active, s = 150, 1 trap), but the
    latent factors, within-block perturbations, and response noise are drawn from
    a scaled Student-t(df) distribution (scaled to unit variance). Within-block
    equicorrelation X_j = sqrt(rho) * Z + sqrt(1-rho) * eps. SNR-calibrated noise.
    """
    rng = np.random.default_rng(seed)

    def rt_scaled(size):
        return rng.standard_t(df, size) / math.sqrt(df / (df - 2))

    block_sizes = np.array([20, 50, 80, 65])
    is_active = np.array([True, True, True, False])
    num_blocks = len(block_sizes)

    block_order, gaps = _distribute_gaps(rng, p, block_sizes)

    X = np.zeros((n, p))
    beta = np.zeros(p)
    latent = [rt_scaled(n) for _ in range(num_blocks)]
    cur = 0
    block_indices = [None] * num_blocks
    w_Z = math.sqrt(rho)
    w_E = math.sqrt(1.0 - rho)

    for i in range(num_blocks + 1):
        g = int(gaps[i])
        if g > 0:
            X[:, cur:cur + g] = rt_scaled((n, g))
            cur += g
        if i < num_blocks:
            b = int(block_order[i])
            bs = int(block_sizes[b])
            X[:, cur:cur + bs] = w_Z * latent[b][:, None] + w_E * rt_scaled((n, bs))
            if is_active[b]:
                beta[cur:cur + bs] = 3.0
            block_indices[b] = np.arange(cur, cur + bs)
            cur += bs

    signal = X @ beta
    sigma_y = _snr_sigma(signal, snr)
    y = signal + rt_scaled(n) * sigma_y

    return dict(X=X, y=y, beta=beta, n=n, p=p, s=150, snr=snr, sigma_y=sigma_y,
                rho=rho, df=df, block_indices=block_indices,
                block_order=block_order)


# ==============================================================================
# Section 06 — DGP: ar1_blocks
# ==============================================================================

def dgp_ar1_blocks_snr(n, p=500, snr=2.0, rho=0.85, seed=None):
    """Block-structured AR(1) DGP: same 4-block geometry as
    dgp_mixed_blocks_snr, but the within-block correlation is AR(1)/Toeplitz:
    Cor(X_j, X_k) = rho^|j-k|. 3 active blocks (s = 150) + 1 trap.
    SNR-calibrated noise.
    """
    rng = np.random.default_rng(seed)
    block_sizes = np.array([20, 50, 80, 65])
    is_active = np.array([True, True, True, False])
    num_blocks = len(block_sizes)

    block_order, gaps = _distribute_gaps(rng, p, block_sizes)

    X = np.zeros((n, p))
    beta = np.zeros(p)
    cur = 0
    block_indices = [None] * num_blocks

    for i in range(num_blocks + 1):
        g = int(gaps[i])
        if g > 0:
            X[:, cur:cur + g] = rng.standard_normal((n, g))
            cur += g
        if i < num_blocks:
            b = int(block_order[i])
            bs = int(block_sizes[b])
            r_idx = np.arange(bs)
            sigma_ar1 = rho ** np.abs(np.subtract.outer(r_idx, r_idx))
            L = np.linalg.cholesky(sigma_ar1)       # lower: L L^T = Sigma
            Z = rng.standard_normal((n, bs))
            X[:, cur:cur + bs] = Z @ L.T             # Cov(rows) = L L^T = Sigma
            if is_active[b]:
                beta[cur:cur + bs] = 3.0
            block_indices[b] = np.arange(cur, cur + bs)
            cur += bs

    signal = X @ beta
    sigma_y = _snr_sigma(signal, snr)
    y = signal + rng.standard_normal(n) * sigma_y

    return dict(X=X, y=y, beta=beta, n=n, p=p, s=150, snr=snr, sigma_y=sigma_y,
                rho=rho, block_indices=block_indices, block_order=block_order)


# ==============================================================================
# Section 07 — DGP: arma_blocks
# ==============================================================================

def _sim_arma(rng, ar, ma, length):
    """Simulate one ARMA(p, q) series via a direct recursion with burn-in:
    x_t = eps_t + sum_k ar[k] x_{t-k} + sum_k ma[k] eps_{t-k}, burn-in
    max(100, p + q + 10) discarded. Does not enforce stationarity (matches the
    R/cpp inline generator), so the callers pass stationary coefficients;
    downstream column standardisation keeps unit variance.
    """
    p_ar = len(ar)
    q_ma = len(ma)
    burn = max(100, p_ar + q_ma + 10)
    total = burn + length

    eps = rng.standard_normal(total)
    x = np.zeros(total)
    for t in range(total):
        val = eps[t]
        for k in range(1, p_ar + 1):
            if t - k >= 0:
                val += ar[k - 1] * x[t - k]
        for k in range(1, q_ma + 1):
            if t - k >= 0:
                val += ma[k - 1] * eps[t - k]
        x[t] = val
    return x[burn:]


def dgp_arma_blocks_snr(n, p=500, snr=2.0, ar_coef=0.8, ma_coefs=(0.5, 0.3, 0.1),
                        seed=None):
    """ARMA-structured blocks DGP: same 4-block geometry, each block governed by
    a different ARMA process (columns standardised to unit variance):
        Block 0 — AR(1);  Block 1 — MA(3);  Block 2 — ARMA(2,1);
        Block 3 (trap) — AR(1).
    3 active blocks (sizes 20, 50, 80; s = 150), 1 inactive trap (size 65).
    SNR-calibrated noise.
    """
    rng = np.random.default_rng(seed)
    ma_coefs = list(ma_coefs)
    block_sizes = np.array([20, 50, 80, 65])
    is_active = np.array([True, True, True, False])
    num_blocks = len(block_sizes)

    models = [
        (list([ar_coef]), []),                          # Block 0: AR(1)
        ([], ma_coefs),                                 # Block 1: MA(3)
        # Block 2: ARMA(2,1), AR terms 5:1 summing to ar_coef (stationary on (0, 1))
        ([ar_coef * 5.0 / 6.0, ar_coef / 6.0], [ma_coefs[0]]),
        ([ar_coef], []),                                # Block 3: AR(1) trap
    ]

    block_order, gaps = _distribute_gaps(rng, p, block_sizes)

    X = np.zeros((n, p))
    beta = np.zeros(p)
    cur = 0
    block_indices = [None] * num_blocks

    for i in range(num_blocks + 1):
        g = int(gaps[i])
        if g > 0:
            X[:, cur:cur + g] = rng.standard_normal((n, g))
            cur += g
        if i < num_blocks:
            b = int(block_order[i])
            bs = int(block_sizes[b])
            ar, ma = models[b]
            block = np.empty((n, bs))
            for row in range(n):
                block[row, :] = _sim_arma(rng, ar, ma, bs)
            # Standardise columns to mean 0 / variance 1 (ddof=1, as R scale()).
            block = (block - block.mean(axis=0)) / block.std(axis=0, ddof=1)
            X[:, cur:cur + bs] = block
            if is_active[b]:
                beta[cur:cur + bs] = 3.0
            block_indices[b] = np.arange(cur, cur + bs)
            cur += bs

    signal = X @ beta
    sigma_y = _snr_sigma(signal, snr)
    y = signal + rng.standard_normal(n) * sigma_y

    return dict(X=X, y=y, beta=beta, n=n, p=p, s=150, snr=snr, sigma_y=sigma_y,
                ar_coef=ar_coef, block_indices=block_indices,
                block_order=block_order)


# ==============================================================================
# Section 08 — DGP: block_equicorr
# ==============================================================================

def dgp_block_equicorr(n, p, G=100, block_size=5, n_active_blocks=10, rho=0.9,
                       snr=2.0, scenario="INDIVIDUAL", seed=None, b=3.0):
    """Block-equicorrelated benchmark DGP. One latent factor per block gives
    unit-variance columns with within-block correlation rho:
        X_j = sqrt(rho) * Z_block(j) + sqrt(1 - rho) * E_j.
    All p = G * block_size columns are tiled by the G blocks; n_active_blocks are
    chosen at random to carry signal (magnitude b, random sign per block). The
    within-block active pattern is set by `scenario`:
        INDIVIDUAL     — 1 active var/block
        REPRESENTATIVE — 2-3 active vars/block
        WHOLE          — all block_size vars active

    Group / block IDs are 0-based. The response uses the biased (n-denominator)
    signal variance, matching the cpp benchmark DGP.

    Returns a dict with X, y, beta, prior_groups (contiguous 0..G-1 labels),
    active_blocks (0-based block IDs), true_support, n, p, G, block_size,
    n_active_blocks, s, snr, sigma_y, rho, scenario.
    """
    if scenario not in ("INDIVIDUAL", "REPRESENTATIVE", "WHOLE"):
        raise ValueError("scenario must be INDIVIDUAL, REPRESENTATIVE, or WHOLE.")
    if p != G * block_size:
        raise ValueError("dgp_block_equicorr: p must equal G * block_size.")
    if n_active_blocks > G:
        raise ValueError("dgp_block_equicorr: n_active_blocks must be <= G.")
    if not (0.0 < rho < 1.0):
        raise ValueError("dgp_block_equicorr: rho must be in (0, 1).")
    rng = np.random.default_rng(seed)

    # Design matrix: one latent factor per block (unit-variance columns).
    Z = rng.standard_normal((n, G))
    X = np.zeros((n, p))
    prior_groups = np.zeros(p, dtype=int)
    sqrt_rho = math.sqrt(rho)
    sqrt_omr = math.sqrt(1.0 - rho)
    for k in range(G):
        cols = slice(k * block_size, (k + 1) * block_size)
        prior_groups[cols] = k
        X[:, cols] = sqrt_rho * Z[:, [k]] + \
            sqrt_omr * rng.standard_normal((n, block_size))

    # Choose active blocks without replacement.
    active_blocks = np.sort(rng.choice(G, size=n_active_blocks, replace=False))

    # Build beta according to the truth scenario.
    beta = np.zeros(p)
    for blk in active_blocks:
        base = blk * block_size
        sign = 1.0 if rng.integers(2) == 0 else -1.0
        if scenario == "INDIVIDUAL":
            beta[base] = sign * b
        elif scenario == "REPRESENTATIVE":
            n_act = int(rng.integers(2, 4))          # 2 or 3
            pos = rng.choice(block_size, size=n_act, replace=False)
            beta[base + pos] = sign * b
        else:  # WHOLE
            beta[base:base + block_size] = sign * b
    true_support = np.flatnonzero(beta)

    # SNR-calibrated response (biased signal variance, as in the cpp DGP).
    signal = X @ beta
    sig_var = np.mean((signal - signal.mean()) ** 2)
    sigma_y = math.sqrt(sig_var / snr)
    y = signal + rng.standard_normal(n) * sigma_y

    return dict(X=X, y=y, beta=beta, prior_groups=prior_groups,
                active_blocks=active_blocks, true_support=true_support,
                n=n, p=p, G=G, block_size=block_size,
                n_active_blocks=n_active_blocks, s=int(true_support.size),
                snr=snr, sigma_y=sigma_y, rho=rho, scenario=scenario)
