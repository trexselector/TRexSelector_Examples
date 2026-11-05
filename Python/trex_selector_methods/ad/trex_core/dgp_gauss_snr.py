# ==============================================================================
# dgp_gauss_snr.py
# ==============================================================================
#
# Data generating process for the standard i.i.d. Gaussian scenario, using
# SNR-based noise calibration.
#
# Mirrors R/trex_selector_methods/ad/trex_core/dgp_gauss_snr.R and
# C++ datagen::SyntheticData with predictor_policy::Normal() and
# noisegen::noise_policy::Normal().
#
# Public functions:
#   dgp_gauss_snr(n, p, support, amplitude, coefs, snr, seed)
#
# ==============================================================================

import numpy as np


def dgp_gauss_snr(n, p, support, amplitude=1.0, coefs=None, snr=1.0, seed=None):
    """
    DGP for the i.i.d. Gaussian scenario with SNR-calibrated noise.

    Predictors are drawn i.i.d. from N(0, 1):

        X[i, j]  i.i.d. N(0, 1)

    The response is:

        y = X @ beta + eps,   eps ~ N(0, noise_sigma^2 * I_n)

    where beta has entries equal to `amplitude` (or `coefs`) at the positions
    given by `support` (0-based) and zero elsewhere.

    The noise standard deviation is chosen according to the target SNR:

        noise_sigma = sqrt( Var(X @ beta) / snr )

    with Var() using ddof=1 (Bessel correction), matching
    C++ noisegen::calculate_noise_params() and R's var().

    Parameters
    ----------
    n         : int
        Number of observations.
    p         : int
        Number of predictors.
    support   : array-like of int
        0-based indices of active predictors.
    amplitude : float, optional
        Scalar signal coefficient applied to all active predictors.
        Ignored when `coefs` is supplied. Default 1.0.
    coefs     : array-like of float or None, optional
        Per-predictor active coefficients. Overrides `amplitude`.
    snr       : float, optional
        Target signal-to-noise ratio  Var(signal) / Var(noise). Default 1.0.
    seed      : int or None, optional
        Seed for np.random.default_rng (governs both X and noise).
        When None, the RNG is seeded from OS entropy.

    Returns
    -------
    dict with keys:
        X            : np.ndarray, shape (n, p), float64
        y            : np.ndarray, shape (n,), float64
        beta         : np.ndarray, shape (p,), float64
        true_support : np.ndarray of int, 0-based, sorted ascending
        n, p, s, snr : scalars
    """
    support = np.asarray(support, dtype=np.intp)
    assert support.ndim == 1, "support must be a 1-D array"
    assert np.all(support >= 0) and np.all(support < p), "support indices out of range"
    assert snr > 0, "snr must be positive"

    rng = np.random.default_rng(seed)

    # X: n x p matrix of i.i.d. N(0, 1)
    X = rng.standard_normal((n, p))

    # Coefficient vector
    beta = np.zeros(p, dtype=np.float64)
    active_coefs = (
        np.asarray(coefs, dtype=np.float64)
        if coefs is not None
        else np.full(len(support), float(amplitude), dtype=np.float64)
    )
    beta[support] = active_coefs

    # SNR-calibrated response
    signal      = X[:, support] @ beta[support]
    var_sig     = np.var(signal, ddof=1)          # Bessel correction matches R var() and C++
    noise_sigma = np.sqrt(var_sig / snr)
    y           = signal + rng.standard_normal(n) * noise_sigma

    return {
        "X":            X,
        "y":            y,
        "beta":         beta,
        "true_support": np.sort(support),
        "n":            n,
        "p":            p,
        "s":            len(support),
        "snr":          snr,
    }
