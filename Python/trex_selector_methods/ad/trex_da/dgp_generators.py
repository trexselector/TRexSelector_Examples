# ==============================================================================
# dgp_generators.py
# ==============================================================================
#
# Data-generating processes for the DA-TRex demos, with SNR-calibrated noise
# (SNR = Var(signal) / Var(noise), sample variance with the n-1 denominator to
# match the C++ make_snr_response).
#
# Support indices are 0-based throughout (Python convention).
#
# Mirrors cpp/trex_selector_methods/ad/trex_da/dgp_generators.hpp. One generator
# per cpp DGP used by the current demo suite (cpp name in parentheses):
#
#   dgp_ar1_snr                (dgp_ar1)                — Demos 01, 07
#   dgp_ar1_block_white_snr    (dgp_ar1_block_white)    — Demo 02
#   dgp_ar1_block_snr          (dgp_ar1_block)          — Demo 03
#   dgp_block_toeplitz_hvt_snr (dgp_block_toeplitz_hvt) — Demo 04
#   dgp_ht_block_white_snr     (dgp_ht_block_white)     — Demo 05
#   dgp_groups_toeplitz_leaf   (dgp_groups_toeplitz_leaf) — Demo 06
#   dgp_nn_snr                 (dgp_nn)                 — Demo 07
#   dgp_equi_snr               (dgp_equi)               — Demo 08
#
# The generators no longer used by the reworked suite (the two-level factor
# model dgp_bt_snr) are archived in
# ~/Documents/C++/TRex_Research/examples_python_rework_2026-07/trex_da/.
# ==============================================================================

import numpy as np


def _snr_response(X, beta, snr, rng):
    """y = X @ beta + noise, with noise_sigma = sqrt(var(signal) / snr)."""
    signal = X @ beta
    noise_sigma = np.sqrt(np.var(signal, ddof=1) / snr)
    y = signal + rng.standard_normal(X.shape[0]) * noise_sigma
    return y


def _pack(X, y, beta, support, **extra):
    return dict(X=X, y=y, beta=beta,
                true_support=np.asarray(support, dtype=np.intp),
                n=X.shape[0], p=X.shape[1], s=len(support), **extra)


# ==============================================================================
# AR(1) Toeplitz DGP (cpp: dgp_ar1)
# ==============================================================================

def dgp_ar1_snr(n, p, support, amplitude=3.0, rho=0.7, snr=2.0, seed=None):
    """AR(1) Toeplitz DGP: Sigma[j, k] = rho^|j-k|, SNR-calibrated noise.

    X[:, 0] = eta_0;  X[:, j] = rho * X[:, j-1] + sqrt(1 - rho^2) * eta_j.
    """
    assert abs(rho) < 1 and snr > 0
    rng = np.random.default_rng(seed)
    scale = np.sqrt(1.0 - rho * rho)

    X = np.empty((n, p))
    X[:, 0] = rng.standard_normal(n)
    for j in range(1, p):
        X[:, j] = rho * X[:, j - 1] + scale * rng.standard_normal(n)

    beta = np.zeros(p)
    beta[support] = amplitude
    y = _snr_response(X, beta, snr, rng)
    return _pack(X, y, beta, support, snr=snr, rho=rho)


# ==============================================================================
# Equicorrelation DGP (cpp: dgp_equi)
# ==============================================================================

def dgp_equi_snr(n, p, support, amplitude=3.0, rho=0.25, snr=2.0, seed=None):
    """Compound symmetry via a single latent factor:
    X[i, j] = sqrt(rho) * f_i + sqrt(1 - rho) * eta[i, j]. SNR-calibrated noise.
    """
    assert 0.0 <= rho < 1.0 and snr > 0
    rng = np.random.default_rng(seed)

    f = rng.standard_normal(n)
    X = np.sqrt(rho) * f[:, None] + np.sqrt(1.0 - rho) * rng.standard_normal((n, p))

    beta = np.zeros(p)
    beta[support] = amplitude
    y = _snr_response(X, beta, snr, rng)
    return _pack(X, y, beta, support, snr=snr, rho=rho)


# ==============================================================================
# Nearest-Neighbours / MA(kappa) DGP (cpp: dgp_nn)
# ==============================================================================

def dgp_nn_snr(n, p, support, amplitude=3.0, kappa=3, rho=0.7, snr=2.0, seed=None):
    """Nearest-Neighbours / MA(kappa) banded DGP: each column is a causal MA(kappa)
    convolution of a shared innovation sheet, giving Sigma[j,k] != 0 only for
    |j-k| <= kappa. theta[l] = rho^l, normalised to unit variance.
    """
    assert kappa >= 1 and 0 <= rho < 1 and snr > 0
    rng = np.random.default_rng(seed)

    theta = rho ** np.arange(kappa + 1)
    theta = theta / np.sqrt(np.sum(theta ** 2))

    eta = rng.standard_normal((n, p + kappa))       # padded innovation sheet
    X = np.empty((n, p))
    for j in range(p):
        X[:, j] = eta[:, j:j + kappa + 1] @ theta

    beta = np.zeros(p)
    beta[support] = amplitude
    y = _snr_response(X, beta, snr, rng)
    return _pack(X, y, beta, support, kappa=kappa, rho=rho, snr=snr)


# ==============================================================================
# Block-diagonal AR(1) DGP (cpp: dgp_ar1_block)
# ==============================================================================

def dgp_ar1_block_snr(n, p, support, amplitude=1.0, n_blocks=5, rho=0.7, snr=2.0,
                      seed=None):
    """Block-diagonal AR(1) DGP: p predictors split into n_blocks blocks; within
    each block the columns follow an AR(1) recursion (Toeplitz rho^|j-k|), zero
    correlation between blocks. SNR-calibrated noise.
    """
    assert p % n_blocks == 0 and abs(rho) < 1 and snr > 0
    rng = np.random.default_rng(seed)
    block_size = p // n_blocks
    innov_sd = np.sqrt(1.0 - rho * rho)

    X = np.zeros((n, p))
    for m in range(n_blocks):
        cs = m * block_size
        X[:, cs] = rng.standard_normal(n)
        for j in range(cs + 1, cs + block_size):
            X[:, j] = rho * X[:, j - 1] + innov_sd * rng.standard_normal(n)

    beta = np.zeros(p)
    beta[support] = amplitude
    y = _snr_response(X, beta, snr, rng)
    return _pack(X, y, beta, support, n_blocks=n_blocks, rho=rho, snr=snr)


# ==============================================================================
# Block-diagonal AR(1) + white-noise block DGP (cpp: dgp_ar1_block_white)
# ==============================================================================

def dgp_ar1_block_white_snr(n, p, support, amplitude=1.0, p_ar=25, p_white=975,
                            n_blocks=5, rho=0.7, snr=2.0, seed=None):
    """M AR(1) blocks (first p_ar columns) plus p_white i.i.d. N(0,1) white-noise
    columns; total width p_ar + p_white. Actives lie in the AR part. SNR-calibrated
    noise. (`p` is the total width supplied by the generic runner; p_ar/p_white
    carry the split.)
    """
    assert p_ar % n_blocks == 0 and p_white >= 1 and abs(rho) < 1 and snr > 0
    rng = np.random.default_rng(seed)
    block_size = p_ar // n_blocks
    innov_sd = np.sqrt(1.0 - rho * rho)

    X_ar = np.zeros((n, p_ar))
    for m in range(n_blocks):
        cs = m * block_size
        X_ar[:, cs] = rng.standard_normal(n)
        for j in range(cs + 1, cs + block_size):
            X_ar[:, j] = rho * X_ar[:, j - 1] + innov_sd * rng.standard_normal(n)
    X_white = rng.standard_normal((n, p_white))
    X = np.hstack([X_ar, X_white])

    beta = np.zeros(p_ar + p_white)
    beta[support] = amplitude
    y = _snr_response(X, beta, snr, rng)
    return _pack(X, y, beta, support, p_ar=p_ar, p_white=p_white,
                 n_blocks=n_blocks, rho=rho, snr=snr)


# ==============================================================================
# Block-diagonal Toeplitz heavy-tailed DGP (cpp: dgp_block_toeplitz_hvt)
# ==============================================================================

def dgp_block_toeplitz_hvt_snr(n, p, support, amplitude=1.0, n_blocks=5, rho=0.8,
                               snr=2.0, nu=3.0, heavy_X=True, heavy_noise=False,
                               seed=None):
    """Heavy-tailed block DGP: n_blocks AR(1)-Toeplitz blocks; when heavy_X, each
    block is converted to multivariate Student-t(nu) via a Gaussian scale mixture
    with an independent chi-squared mixing variable per (row, block) — extreme
    events are block-local, mirroring cpp dgp_block_toeplitz_hvt. Noise is
    Gaussian, or variance-matched Student-t(nu) when heavy_noise. SNR-calibrated.
    """
    assert p % n_blocks == 0 and abs(rho) < 1 and snr > 0 and nu > 2
    rng = np.random.default_rng(seed)
    block_size = p // n_blocks
    innov_sd = np.sqrt(1.0 - rho * rho)

    X = np.zeros((n, p))
    for m in range(n_blocks):
        cs = m * block_size
        Zb = np.zeros((n, block_size))
        Zb[:, 0] = rng.standard_normal(n)
        for j in range(1, block_size):
            Zb[:, j] = rho * Zb[:, j - 1] + innov_sd * rng.standard_normal(n)
        if heavy_X:
            # One mixing variable per (row, block): sqrt(nu / chi2(nu))
            u = rng.chisquare(nu, size=n)
            Zb = Zb * np.sqrt(nu / u)[:, None]
        X[:, cs:cs + block_size] = Zb

    beta = np.zeros(p)
    beta[support] = amplitude
    signal = X @ beta
    noise_sigma = np.sqrt(np.var(signal, ddof=1) / snr)
    if heavy_noise:
        # Var[t(nu)] = nu / (nu - 2): rescale to match the target variance
        eps = rng.standard_t(nu, size=n) * (noise_sigma / np.sqrt(nu / (nu - 2)))
    else:
        eps = rng.standard_normal(n) * noise_sigma
    y = signal + eps
    return _pack(X, y, beta, support, n_blocks=n_blocks, rho=rho, snr=snr,
                 nu=nu, heavy_noise=heavy_noise)


# ==============================================================================
# Heavy-tailed AR(1)-block + white-noise DGP (cpp: dgp_ht_block_white)
# ==============================================================================

def dgp_ht_block_white_snr(n, p, support, amplitude=1.0, p_ar=25, p_white=475,
                           n_blocks=5, rho=0.8, snr=2.0, nu=3.0, heavy_noise=False,
                           seed=None):
    """Heavy-tailed AR(1)-Toeplitz t(nu) blocks (p_ar columns, one chi-squared
    mixing variable per (row, block)) plus p_white i.i.d. Student-t(nu)
    white-noise columns (element-wise independent, mirroring cpp
    dgp_ht_block_white). Actives lie in the AR part. SNR-calibrated.
    """
    assert p_ar % n_blocks == 0 and abs(rho) < 1 and snr > 0 and nu > 2
    rng = np.random.default_rng(seed)
    ptot = p_ar + p_white
    block_size = p_ar // n_blocks
    innov_sd = np.sqrt(1.0 - rho * rho)

    X = np.zeros((n, ptot))
    for m in range(n_blocks):
        cs = m * block_size
        Zb = np.zeros((n, block_size))
        Zb[:, 0] = rng.standard_normal(n)
        for j in range(1, block_size):
            Zb[:, j] = rho * Zb[:, j - 1] + innov_sd * rng.standard_normal(n)
        u = rng.chisquare(nu, size=n)
        X[:, cs:cs + block_size] = Zb * np.sqrt(nu / u)[:, None]
    if p_white > 0:
        X[:, p_ar:] = rng.standard_t(nu, size=(n, p_white))

    beta = np.zeros(ptot)
    beta[support] = amplitude
    signal = X @ beta
    noise_sigma = np.sqrt(np.var(signal, ddof=1) / snr)
    if heavy_noise:
        eps = rng.standard_t(nu, size=n) * (noise_sigma / np.sqrt(nu / (nu - 2)))
    else:
        eps = rng.standard_normal(n) * noise_sigma
    y = signal + eps
    return _pack(X, y, beta, support, p_ar=p_ar, p_white=p_white,
                 n_blocks=n_blocks, rho=rho, snr=snr, nu=nu,
                 heavy_noise=heavy_noise)


# ==============================================================================
# Prior Groups + Toeplitz leaf DGP (cpp: dgp_groups_toeplitz_leaf)
# ==============================================================================

def dgp_groups_toeplitz_leaf(n, p, support, amplitude=3.0, groups=None,
                             rho_levels=None, phi=0.5, snr=2.0, seed=None):
    """Multi-level latent-factor DGP with Toeplitz leaf blocks. `groups` is a list
    of L label-vectors (each length p, fine->coarse); rho_levels is strictly
    decreasing in (0,1). Coarse levels x=1..L-1 add a shared factor per group
    with loading sqrt(rho_ext[x]-rho_ext[x+1]); the finest level adds a Toeplitz(phi)
    block within each leaf; plus idiosyncratic noise. SNR-calibrated.
    """
    assert groups is not None and rho_levels is not None
    assert len(groups) == len(rho_levels) >= 2
    assert all(rho_levels[i] > rho_levels[i + 1] for i in range(len(rho_levels) - 1))
    assert rho_levels[0] < 1.0 and rho_levels[-1] > 0.0 and 0.0 <= phi < 1.0

    rng = np.random.default_rng(seed)
    groups = [np.asarray(g) for g in groups]
    rho_ext = list(rho_levels) + [0.0]
    X = np.zeros((n, p))

    # Coarser latent-factor levels x = 1 .. L-1
    for x in range(1, len(groups)):
        loading = np.sqrt(rho_ext[x] - rho_ext[x + 1])
        for gid in np.unique(groups[x]):
            f = rng.standard_normal(n)
            X[:, groups[x] == gid] += loading * f[:, None]

    # Finest level: Toeplitz(phi) within each leaf block
    leaf_scale = np.sqrt(rho_levels[0] - rho_levels[1])
    for gid in np.unique(groups[0]):
        cols = np.flatnonzero(groups[0] == gid)
        g = len(cols)
        idx = np.arange(g)
        Sigma = phi ** np.abs(idx[:, None] - idx[None, :])
        L = np.linalg.cholesky(Sigma)
        X[:, cols] += leaf_scale * (rng.standard_normal((n, g)) @ L.T)

    # Idiosyncratic noise
    X += np.sqrt(1.0 - rho_levels[0]) * rng.standard_normal((n, p))

    beta = np.zeros(p)
    beta[support] = amplitude
    y = _snr_response(X, beta, snr, rng)
    return _pack(X, y, beta, support, snr=snr)
