# ==============================================================================
# support_generators.py
# ==============================================================================
#
# Support-index construction helpers for TRex simulation demos.
# Mirrors R/trex_selector_methods/support_generators.R.
#
# Contents:
#   make_support_capped_spread()  — Deterministic evenly-spaced support.
#   make_support_random()         — Uniformly random support.
#
# ==============================================================================

import numpy as np


def make_support_capped_spread(s, p, max_gap=10):
    """
    Build an evenly-spaced support with a capped gap (deterministic).

    Mirrors R make_support_capped_spread():
        gap = min(p // s, max_gap)
        indices = {0, gap, 2*gap, ..., (s-1)*gap}  (0-based)

    Parameters
    ----------
    s       : int  Number of active predictors.
    p       : int  Total number of predictors.
    max_gap : int  Maximum gap between consecutive active indices. Default 10.

    Returns
    -------
    np.ndarray of int, shape (s,), 0-based, sorted ascending.

    Examples
    --------
    >>> make_support_capped_spread(10, 1000, 20)
    array([  0,  20,  40,  60,  80, 100, 120, 140, 160, 180])
    """
    gap = min(p // s, max_gap)
    return np.arange(s, dtype=np.intp) * gap


def make_support_random(s, p, seed):
    """
    Build a uniformly random support without replacement (stochastic).

    Mirrors R make_support_random():
        rng = np.random.default_rng(seed + 500_000)
        support = sorted sample of size s from {0, ..., p-1}

    The +500_000 offset keeps the support-draw RNG stream disjoint
    from the DGP RNG stream (which uses seed directly), and matches
    the R convention of set.seed(seed + 500000L).

    Parameters
    ----------
    s    : int  Number of active predictors.
    p    : int  Total number of predictors.
    seed : int  Base seed (500_000 will be added internally).

    Returns
    -------
    np.ndarray of int, shape (s,), 0-based, sorted ascending.
    """
    rng = np.random.default_rng(int(seed) + 500_000)
    return np.sort(rng.choice(p, size=s, replace=False).astype(np.intp))
