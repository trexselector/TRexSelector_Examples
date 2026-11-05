# ==============================================================================
# demo_ts_perf_02_runtime_vs_dummies.py
# ==============================================================================
#
# Run time of the terminated solution paths vs. the number of dummies L, for
# all demo T-solvers.
#
# Mirrors cpp/tsolvers/ad/demo_ts_perf_02_runtime_vs_dummies/demo_ts_perf_02_runtime_vs_dummies.cpp:
# n = 500, p = 1,000, 10 true coefficients equal to 3 on a fixed random support,
# noise sigma about 1, T = 10 and L in {1, 2, 5, 10, 20} x p. Every run draws
# 20p dummies once, and each L uses the first L of them (nested dummy sets, one
# contiguous block of the Fortran-ordered array). The runs are spread over the
# worker processes, and each solver runs single-threaded inside its run. Per L
# and solver: the time to T with its steps, the warm start T -> T + 1, and a
# fresh run to T + 1.
#
# Usage: demo_ts_perf_02_runtime_vs_dummies.py [num_MC] [workers]   (default 500, 6)
#
# ==============================================================================

import os
import sys
import time

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.dirname(os.path.dirname(_THIS_DIR))  # tsolvers/, the shared helpers
if _PARENT_DIR not in sys.path:
    sys.path.insert(0, _PARENT_DIR)

from ts_demo_utils import display_path, output_dir, print_section_header
from ts_perf_solvers import demo_solvers, file_suffix
from ts_perf_utils import (
    draw_support,
    parse_positive,
    reader_data,
    run_parallel,
)

# Setup
n, p = 500, 1_000
num_active = 10
beta_active = 3.0
snr = 90.0                    # signal variance 10 * 3^2 over noise variance 1
T_stop = 10
support_seed = 12345
data_seed_offset = 1000       # run mc uses seed data_seed_offset + mc
L_factors = [1, 2, 5, 10, 20]
FIELDS = ["ms", "steps", "warm_ms", "fresh_ms"]

SUPPORT = draw_support(p, num_active, support_seed)
COEFS = [beta_active] * num_active


def run_one(mc):
    """One run: every solver at every L on the same nested dummy sets."""
    X, D_all, y = reader_data(n, p, L_factors[-1] * p, SUPPORT, COEFS, snr, seed=data_seed_offset + mc)
    seed = data_seed_offset + mc
    results = []
    for spec in demo_solvers():
        r = {f: [] for f in FIELDS}
        try:
            for factor in L_factors:
                D = D_all[:, :factor * p]  # the first L dummy columns, a contiguous view

                # Path to T, then a warm start to T + 1
                t0 = time.perf_counter()
                solver = spec.make(X, D, y)
                solver.setTieSeed(seed)
                solver.executeStep(T_stop, True)
                to_T = 1e3 * (time.perf_counter() - t0)
                steps = float(solver.getNumSteps())
                t1 = time.perf_counter()
                solver.executeStep(T_stop + 1, True)
                warm = 1e3 * (time.perf_counter() - t1)

                # A fresh run to T + 1 for comparison
                t2 = time.perf_counter()
                fresh = spec.make(X, D, y)
                fresh.setTieSeed(seed)
                fresh.executeStep(T_stop + 1, True)
                fresh_ms = 1e3 * (time.perf_counter() - t2)

                for f, v in zip(FIELDS, (to_T, steps, warm, fresh_ms), strict=True):
                    r[f].append(v)
        except (RuntimeError, ValueError):  # the whole run of this solver is left out
            r = None
        results.append(r)
    return results


def main():
    num_MC = parse_positive(sys.argv, 1, 500)
    workers = parse_positive(sys.argv, 2, 6)
    solvers = demo_solvers()
    S, K = len(solvers), len(L_factors)

    print()
    print_section_header("T-Solver Performance: Runtime vs. Number of Dummies")
    print(f"n = {n}, p = {p}, L in {{1, 2, 5, 10, 20}} x p, T = {T_stop}, true coefficients: {num_active} x "
          f"{beta_active:g}, SNR = {snr:g}")
    print(f"{S} solvers, {num_MC} runs on {workers} worker processes (single-threaded solvers)\n")

    t_start = time.perf_counter()
    runs = run_parallel(run_one, range(num_MC), workers)
    wall = time.perf_counter() - t_start
    failures = sum(r is None for results in runs for r in results)

    def column(s, k, field):
        return np.array([results[s][field][k] for results in runs if results[s] is not None])

    def per_step(s, k):
        return 1e3 * column(s, k, "ms") / np.maximum(1.0, column(s, k, "steps"))

    def speedup(s, k):
        return column(s, k, "fresh_ms") / np.maximum(1e-6, column(s, k, "warm_ms"))

    out = []

    def table(title, values, precision):
        out.append(title)
        out.append(f"{'solver':<22}" + "".join(f"{'L=' + ('' if f == 1 else str(f)) + 'p':>9}" for f in L_factors))
        out.append("-" * (22 + 9 * K))
        for s, spec in enumerate(solvers):
            out.append(f"{spec.name:<22}" + "".join(f"{np.median(values(s, k)):>9.{precision}f}" for k in range(K)))
        out.append("")

    table("Median time to T = 10 (ms)", lambda s, k: column(s, k, "ms"), 2)
    table("Median steps to T = 10", lambda s, k: column(s, k, "steps"), 1)
    table("Median time per step (us)", per_step, 1)
    table("Median warm start T = 10 -> 11 (ms)", lambda s, k: column(s, k, "warm_ms"), 3)
    table("Median fresh run to T = 11 over warm start 10 -> 11 (ratio)", speedup, 1)
    out.append(f"Failed solver runs: {failures}")
    out.append(f"Wall time: {wall:.1f} s")
    text = "\n".join(out) + "\n"
    print(text, end="")

    stem = f"runtime_vs_dummies_mc{num_MC}{file_suffix()}"  # suffixed for a TREX_SOLVERS selection
    data_dir = output_dir(_THIS_DIR)
    csv_path = os.path.join(data_dir, stem + ".csv")
    with open(csv_path, "w") as f:
        f.write("solver,family,L_factor,L,ms,steps,us_per_step,warm_ms,fresh_ms,fresh_over_warm\n")
        for s, spec in enumerate(solvers):
            for k, factor in enumerate(L_factors):
                row = [np.median(column(s, k, "ms")), np.median(column(s, k, "steps")), np.median(per_step(s, k)),
                       np.median(column(s, k, "warm_ms")), np.median(column(s, k, "fresh_ms")),
                       np.median(speedup(s, k))]
                f.write(f"{spec.name},{spec.family},{factor},{factor * p}," + ",".join(f"{v:.6g}" for v in row) + "\n")
    txt_path = os.path.join(data_dir, stem + ".txt")
    with open(txt_path, "w") as f:
        f.write(text)

    print(f"\nResults written to:\n  {display_path(csv_path)}\n  {display_path(txt_path)}\n")


if __name__ == "__main__":
    main()
