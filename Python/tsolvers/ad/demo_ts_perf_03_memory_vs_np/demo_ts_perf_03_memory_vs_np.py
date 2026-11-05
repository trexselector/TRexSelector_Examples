# ==============================================================================
# demo_ts_perf_03_memory_vs_np.py
# ==============================================================================
#
# Solver memory vs. the ratio n/p, for all demo T-solvers.
#
# Mirrors cpp/tsolvers/ad/demo_ts_perf_03_memory_vs_np/demo_ts_perf_03_memory_vs_np.cpp:
# p = 1,000, L = p, n in {50, 100, 200, 500, 1,000, 2,000}, 10 true
# coefficients equal to 3 on a fixed random support, noise sigma about 1,
# T = 10. Per run and n the data are drawn and scaled once, and every solver
# runs on them as a precentered reader. The solver memory is the library's
# memory report getMemoryBytes() after T = 10, the heap bytes the C++ solver
# holds without the borrowed X and D (the binding and the interpreter are not
# counted). Unlike the peak resident set of a forked child, the report is
# portable (no fork, no getrusage()) and deterministic for given data, so the
# runs go to worker processes with single-threaded solvers.
#
# Usage: demo_ts_perf_03_memory_vs_np.py [num_MC] [workers]   (default 500, 6)
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
p = 1_000
num_dummies = p
num_active = 10
beta_active = 3.0
snr = 90.0                    # signal variance 10 * 3^2 over noise variance 1
T_stop = 10
support_seed = 12345
data_seed_offset = 1000       # run mc uses seed data_seed_offset + mc
n_grid = [50, 100, 200, 500, 1_000, 2_000]
BYTES_PER_MIB = 1024.0 * 1024.0

SUPPORT = draw_support(p, num_active, support_seed)
COEFS = [beta_active] * num_active


def run_one(mc):
    """
    One run: every solver on the data of every n. Returns records[s][k] as
    (memory in MiB, steps), or None when the solver fails.
    """
    seed = data_seed_offset + mc
    solvers = demo_solvers()
    records = [[None] * len(n_grid) for _ in solvers]
    for k, n in enumerate(n_grid):
        X, D, y = reader_data(n, p, num_dummies, SUPPORT, COEFS, snr, seed=seed)
        for s, spec in enumerate(solvers):
            try:
                solver = spec.make(X, D, y)
                solver.setTieSeed(seed)
                solver.executeStep(T_stop, True)
                records[s][k] = (solver.getMemoryBytes() / BYTES_PER_MIB, float(solver.getNumSteps()))
            except (RuntimeError, ValueError):  # a failing solver is left out of its medians
                records[s][k] = None
    return records


def main():
    num_MC = parse_positive(sys.argv, 1, 500)
    workers = parse_positive(sys.argv, 2, 6)
    solvers = demo_solvers()
    S, K = len(solvers), len(n_grid)

    print()
    print_section_header("T-Solver Performance: Memory vs. n/p")
    print(f"p = {p}, L = p, n in {{50, 100, 200, 500, 1000, 2000}}, T = {T_stop}, true coefficients: "
          f"{num_active} x {beta_active:g}, SNR = {snr:g}")
    print(f"Solver memory: held heap bytes after T = {T_stop} (getMemoryBytes; X, D borrowed, not counted)")
    print(f"{S} solvers, {num_MC} runs on {workers} worker processes (single-threaded solvers)\n")

    t_start = time.perf_counter()
    runs = run_parallel(run_one, range(num_MC), workers)
    wall = time.perf_counter() - t_start

    # memory[s][k] and steps[s][k] collect the finished runs in run order
    memory = [[[r[s][k][0] for r in runs if r[s][k] is not None] for k in range(K)] for s in range(S)]
    steps = [[[r[s][k][1] for r in runs if r[s][k] is not None] for k in range(K)] for s in range(S)]
    failures = sum(rec is None for r in runs for row in r for rec in row)

    # Data size per n: X, D and y as doubles
    data_mib = [8.0 * n * (p + num_dummies + 1) / BYTES_PER_MIB for n in n_grid]

    def header(title):
        return [title, f"{'solver':<22}" + "".join(f"{'n=' + str(n):>9}" for n in n_grid), "-" * (22 + 9 * K)]

    out = [""] + header("Data X, D and y (MiB)")
    out.append(f"{'all solvers':<22}" + "".join(f"{v:>9.2f}" for v in data_mib))
    out.append("")

    def table(title, value, precision):
        out.extend(header(title))
        for s, spec in enumerate(solvers):
            out.append(f"{spec.name:<22}" + "".join(f"{value(s, k):>9.{precision}f}" for k in range(K)))
        out.append("")

    def largest(s, k):
        return max(memory[s][k]) if memory[s][k] else 0.0

    table("Median solver memory: held heap after T = 10 (MiB)", lambda s, k: float(np.median(memory[s][k])), 2)
    table("Largest solver memory over the runs: held heap after T = 10 (MiB)", largest, 2)
    table("Median solver memory relative to the data X, D and y (%)",
          lambda s, k: 100.0 * float(np.median(memory[s][k])) / data_mib[k], 1)
    table("Median steps to T = 10", lambda s, k: float(np.median(steps[s][k])), 1)
    out.append(f"Failed solver runs: {failures}")
    out.append(f"Wall time: {wall:.1f} s")
    text = "\n".join(out) + "\n"
    print(text, end="")

    stem = f"memory_vs_np_mc{num_MC}{file_suffix()}"  # suffixed for a TREX_SOLVERS selection
    data_dir = output_dir(_THIS_DIR)
    csv_path = os.path.join(data_dir, stem + ".csv")
    with open(csv_path, "w") as f:
        f.write("solver,family,n,n_over_p,data_mib,solver_mib,solver_max_mib,steps\n")
        for s, spec in enumerate(solvers):
            for k, n in enumerate(n_grid):
                row = [n / p, data_mib[k], float(np.median(memory[s][k])), largest(s, k), float(np.median(steps[s][k]))]
                f.write(f"{spec.name},{spec.family},{n}," + ",".join(f"{v:.6g}" for v in row) + "\n")
    txt_path = os.path.join(data_dir, stem + ".txt")
    with open(txt_path, "w") as f:
        f.write(text)

    print(f"\nResults written to:\n  {display_path(csv_path)}\n  {display_path(txt_path)}\n")


if __name__ == "__main__":
    main()
