# ==============================================================================
# ts_perf_utils.py
# ==============================================================================
#
# Shared helpers for the performance demos (demo_ts_perf_*), mirroring the
# corresponding parts of cpp/tsolvers/ts_perf_utils.hpp: the fixed random
# support, reader-ready data, the selection read from the coefficients,
# positive integer arguments and the worker pool.
#
# The memory demo reads the solvers' own memory report (getMemoryBytes())
# instead of a peak resident set, so no helper here needs the Unix-only
# resource module: portable to Windows, and deterministic for given data.
#
# ==============================================================================

from multiprocessing import Pool

import numpy as np
from ts_demo_utils import gen_synthetic_data

from trexselector.ml_methods import LpNormScaler, NormType


def draw_support(p, s, seed):
    """Fixed random support: s of the p indices, sorted (NumPy draw of the C++ std::sample)."""
    rng = np.random.default_rng(seed)
    return sorted(int(j) for j in rng.choice(p, size=s, replace=False))


def reader_data(n, p, num_dummies, support, coefs, snr, seed):
    """
    Draw X, D and y and prepare them once for precentered readers: X and D
    centered and scaled to unit L2 norm in place, y centered.
    """
    X, D, y = gen_synthetic_data(n, p, num_dummies, support, coefs, snr, seed=seed)
    LpNormScaler(NormType.L2, center=True).fit_transform_inplace(X)
    if num_dummies > 0:
        LpNormScaler(NormType.L2, center=True).fit_transform_inplace(D)
    y -= y.mean()
    return X, D, y


def selected_predictors(solver, p):
    """
    Selected original variables: nonzero coefficients among the first p, as the
    T-Rex selector reads them (T-Stagewise keeps the coefficient of a variable
    it removes from the active set, so the active set can be smaller).
    """
    beta = np.asarray(solver.getBeta()).ravel()[:p]
    return [int(j) for j in np.flatnonzero(beta)]


def included_dummies(solver, p):
    """
    Included dummies: nonzero coefficients beyond the first p, the count that
    stops a T-solver at T (the dummies of the support, as the T-Rex selector
    counts them). The active set can hold fewer: T-Stagewise keeps the
    coefficient of a dummy it drops.
    """
    return int(np.count_nonzero(np.asarray(solver.getBeta()).ravel()[p:]))


def parse_positive(argv, index, fallback):
    """Optional positive integer argument argv[index], or fallback when it is missing."""
    if len(argv) <= index:
        return fallback
    text = argv[index]
    if not text.isdigit() or int(text) < 1:
        raise SystemExit(f"expected a positive integer, got: {text}")
    return int(text)


def run_parallel(func, tasks, workers, chunksize=1):
    """Map func over tasks on `workers` processes, in task order (the solvers run serially by default)."""
    if workers == 1:
        return [func(t) for t in tasks]
    with Pool(processes=workers) as pool:
        return pool.map(func, tasks, chunksize=chunksize)

