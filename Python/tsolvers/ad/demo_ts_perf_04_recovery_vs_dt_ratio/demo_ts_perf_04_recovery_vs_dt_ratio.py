# ==============================================================================
# demo_ts_perf_04_recovery_vs_dt_ratio.py
# ==============================================================================
#
# Support recovery of the T-solvers vs. the ratio n/(s log p), and the
# Donoho-Tanner phase transition of T-LARS, T-OMP and T-AFS.
#
# Mirrors cpp/tsolvers/ad/demo_ts_perf_04_recovery_vs_dt_ratio/demo_ts_perf_04_recovery_vs_dt_ratio.cpp.
# Exact recovery: the first s distinct variables that enter the path are the
# true support (a dummy among them counts as a miss). Every solver runs as a
# precentered reader to the first dummy (T = 1), which also gives TPR and FDR
# at T = 1. L = p, coefficients equal to 3 on a fixed random support, noise
# sigma about 1.
#
#   Part 1: p in {250, 500, 1,000, 2,000, 5,000}, s = 10, n = n/(s log p) * s log p
#           for n/(s log p) in {0.25, 0.5, 1, 1.5, 2, 3, 4}, all demo solvers.
#   Part 2: p = 400, 20 x 20 grid of n/p and s/n in {0.05, ..., 1},
#           T-LARS, T-OMP and T-AFS (rho 0.3).
#
# Usage: demo_ts_perf_04_recovery_vs_dt_ratio.py [num_MC] [workers] [parts...]   (default 500, 6, all)
#
# ==============================================================================

import math
import os
import sys
import time

import numpy as np

from trexselector.utils import compute_fdp, compute_tpp

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.dirname(os.path.dirname(_THIS_DIR))  # tsolvers/, the shared helpers
if _PARENT_DIR not in sys.path:
    sys.path.insert(0, _PARENT_DIR)

from ts_demo_utils import (
    PartSelection,
    display_path,
    output_dir,
    print_section_header,
)
from ts_perf_solvers import demo_solvers, file_suffix, selected
from ts_perf_utils import (
    draw_support,
    parse_positive,
    reader_data,
    run_parallel,
    included_dummies,
    selected_predictors,
)

beta_active = 3.0
support_seed = 12345
data_seed_offset = 1000        # Part 1, run mc: seed data_seed_offset + mc
phase_seed_offset = 3_000_000  # Part 2, run mc and cell c: seed offset + mc * cells + c

# Part 1
s_curves = 10
p_grid = [250, 500, 1_000, 2_000, 5_000]
ratio_grid = [0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0]

# Part 2
p_phase = 400
phase_steps = 20  # n/p and s/n in {1, ..., 20} / 20
# Only the ones TREX_SOLVERS selects (all without a selection)
phase_solvers = [name for name in ("T-LARS", "T-OMP", "T-AFS (rho 0.3)") if selected(name)]


def n_for(p, ratio):
    """Number of observations for p and the ratio n/(s log p), natural log."""
    return round(ratio * s_curves * math.log(p))


def first_entries_exact(solver, support):
    """True when the first distinct variables entering the path are the support."""
    truth, seen = set(support), set()
    for step in solver.getActions():
        for a in step:
            if a <= 0:
                continue  # a drop
            index = a - 1
            if index in seen:
                continue
            seen.add(index)
            if index not in truth:
                return False
            if len(seen) == len(truth):
                return True
    return False  # the path ended before s variables entered


def run_problem(n, p, support, seed, specs):
    """Draw one data set, scale it and run `specs` to the first dummy: (exact, tpp, fdp, reached, ok) per spec."""
    coefs = [beta_active] * len(support)
    snr = len(support) * beta_active ** 2  # noise sigma 1
    X, D, y = reader_data(n, p, p, support, coefs, snr, seed=seed)
    outcomes = []
    for spec in specs:
        try:
            solver = spec.make(X, D, y)
            solver.setTieSeed(seed)
            solver.executeStep(1, True)
            selected = selected_predictors(solver, p)
            outcomes.append((1.0 if first_entries_exact(solver, support) else 0.0,
                             compute_tpp(selected, support), compute_fdp(selected, support),
                             1.0 if included_dummies(solver, p) > 0 else 0.0, True))
        except (RuntimeError, ValueError):
            outcomes.append((0.0, 0.0, 0.0, 0.0, False))
    return outcomes


def std_error(v):
    """Standard error of the mean (Bessel-corrected sd over sqrt(size))."""
    return float(np.std(v, ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0


# ------------------------------------------------------------------------------
# Part 1
# ------------------------------------------------------------------------------

SUPPORTS = {p: draw_support(p, s_curves, support_seed) for p in p_grid}


def part1_task(task):
    c, mc = task
    p, ratio = p_grid[c // len(ratio_grid)], ratio_grid[c % len(ratio_grid)]
    return run_problem(n_for(p, ratio), p, SUPPORTS[p], data_seed_offset + mc, demo_solvers())


def run_part1(num_MC, workers):
    print_section_header("Part 1: Recovery vs. n/(s log p)")
    solvers = demo_solvers()
    S, P, R = len(solvers), len(p_grid), len(ratio_grid)
    C = P * R
    print(f"p in {{250, 500, 1000, 2000, 5000}}, L = p, s = {s_curves}, n/(s log p) in {{0.25, 0.5, 1, 1.5, 2, 3, 4}}, "
          f"T = 1\n{S} solvers, {num_MC} runs\n")

    t_start = time.perf_counter()
    tasks = [(c, mc) for c in range(C) for mc in range(num_MC)]
    flat = run_parallel(part1_task, tasks, workers)
    wall = time.perf_counter() - t_start
    outcomes = [flat[c * num_MC:(c + 1) * num_MC] for c in range(C)]
    failures = sum(not o[4] for per_c in outcomes for per_mc in per_c for o in per_mc)

    def column(c, s, field):
        return [per_mc[s][field] for per_mc in outcomes[c] if per_mc[s][4]]

    out = []

    def table(title, i, field):
        p = p_grid[i]
        out.append(f"{title}, p = {p}")
        out.append(f"{'n/(s log p)':<22}" + "".join(f"{r:>8g}" for r in ratio_grid))
        out.append(f"{'n':<22}" + "".join(f"{n_for(p, r):>8}" for r in ratio_grid))
        out.append("-" * (22 + 8 * R))
        for s, spec in enumerate(solvers):
            out.append(f"{spec.name:<22}" + "".join(f"{100.0 * np.mean(column(i * R + j, s, field)):>8.1f}"
                                                    for j in range(R)))
        out.append("")

    for i in range(P):
        table("Exact recovery by the first s = 10 entries (%)", i, 0)
        table("TPR at T = 1 (%)", i, 1)
        table("FDR at T = 1 (%)", i, 2)
    out.append(f"Failed solver runs: {failures}")
    out.append(f"Wall time: {wall:.1f} s")
    text = "\n".join(out) + "\n"
    print(text, end="")

    stem = f"recovery_vs_dt_ratio_mc{num_MC}{file_suffix()}"  # suffixed for a TREX_SOLVERS selection
    data_dir = output_dir(_THIS_DIR)
    csv_path = os.path.join(data_dir, stem + ".csv")
    with open(csv_path, "w") as f:
        f.write("solver,family,p,s,n,ratio,exact,exact_se,TPR,TPR_se,FDR,FDR_se,reached\n")
        for s, spec in enumerate(solvers):
            for i, p in enumerate(p_grid):
                for j, ratio in enumerate(ratio_grid):
                    c = i * R + j
                    ex, tp, fd = column(c, s, 0), column(c, s, 1), column(c, s, 2)
                    row = [np.mean(ex), std_error(ex), np.mean(tp), std_error(tp), np.mean(fd), std_error(fd),
                           np.mean(column(c, s, 3))]
                    f.write(f"{spec.name},{spec.family},{p},{s_curves},{n_for(p, ratio)},{ratio:g},"
                            + ",".join(f"{v:.6g}" for v in row) + "\n")
    txt_path = os.path.join(data_dir, stem + ".txt")
    with open(txt_path, "w") as f:
        f.write(text)
    print(f"\nResults written to:\n  {display_path(csv_path)}\n  {display_path(txt_path)}\n")


# ------------------------------------------------------------------------------
# Part 2
# ------------------------------------------------------------------------------

def n_of(a):
    return (a + 1) * p_phase // phase_steps


def s_of(a, b):
    return (b + 1) * n_of(a) // phase_steps


CELLS = phase_steps * phase_steps
PHASE_SUPPORTS = [draw_support(p_phase, s_of(c // phase_steps, c % phase_steps), support_seed + c)
                  for c in range(CELLS)]


def part2_task(task):
    mc, c = task
    specs = [next(sp for sp in demo_solvers() if sp.name == name) for name in phase_solvers]
    seed = phase_seed_offset + mc * CELLS + c
    outcomes = run_problem(n_of(c // phase_steps), p_phase, PHASE_SUPPORTS[c], seed, specs)
    return tuple(int(o[0]) if o[4] else -1 for o in outcomes)


def run_part2(num_MC, workers):
    print_section_header("Part 2: Donoho-Tanner phase transition")
    if not phase_solvers:
        print("TREX_SOLVERS selects none of its solvers, Part 2 skipped\n")
        return
    S, G = len(phase_solvers), phase_steps
    print(f"p = {p_phase}, L = p, n/p and s/n in {{0.05, 0.10, ..., 1}}, T = 1\n"
          f"T-LARS, T-OMP and T-AFS (rho 0.3), {num_MC} runs per cell\n")

    t_start = time.perf_counter()
    tasks = [(mc, c) for mc in range(num_MC) for c in range(CELLS)]
    flat = run_parallel(part2_task, tasks, workers, chunksize=64)
    wall = time.perf_counter() - t_start
    exact = np.array(flat, dtype=int).reshape(num_MC, CELLS, S)  # 1 exact, 0 miss, -1 failed
    failures = int(np.sum(exact < 0))

    def cell_values(c, k):
        v = exact[:, c, k]
        return v[v >= 0].astype(float)

    out = []
    for k, name in enumerate(phase_solvers):
        out.append(f"Exact recovery by the first s entries (%), {name}")
        out.append(f"{'s/n \\ n/p':>10}" + "".join(f"{(a + 1) / G:>5.2f}" for a in range(G)))
        out.append("-" * (10 + 5 * G))
        for b in reversed(range(G)):
            out.append(f"{(b + 1) / G:>10.2f}" + "".join(f"{100.0 * np.mean(cell_values(a * G + b, k)):>5.0f}"
                                                        for a in range(G)))
        out.append("")
    out.append(f"Failed solver runs: {failures}")
    out.append(f"Wall time: {wall:.1f} s")
    text = "\n".join(out) + "\n"
    print(text, end="")

    families = {sp.name: sp.family for sp in demo_solvers()}
    stem = f"phase_transition_mc{num_MC}{file_suffix()}"  # suffixed for a TREX_SOLVERS selection
    data_dir = output_dir(_THIS_DIR)
    csv_path = os.path.join(data_dir, stem + ".csv")
    with open(csv_path, "w") as f:
        f.write("solver,family,p,n,s,n_over_p,s_over_n,exact,exact_se\n")
        for k, name in enumerate(phase_solvers):
            for c in range(CELLS):
                a, b = c // G, c % G
                v = cell_values(c, k)
                f.write(f"{name},{families[name]},{p_phase},{n_of(a)},{s_of(a, b)},{(a + 1) / G:.6g},"
                        f"{(b + 1) / G:.6g},{np.mean(v):.6g},{std_error(v):.6g}\n")
    txt_path = os.path.join(data_dir, stem + ".txt")
    with open(txt_path, "w") as f:
        f.write(text)
    print(f"\nResults written to:\n  {display_path(csv_path)}\n  {display_path(txt_path)}\n")


def main():
    num_MC = parse_positive(sys.argv, 1, 500)
    workers = parse_positive(sys.argv, 2, 6)
    run = PartSelection(sys.argv[3:])
    print()
    print_section_header("T-Solver Performance: Recovery vs. n/(s log p)")
    print(f"{num_MC} runs on {workers} worker processes\n")
    if run(1):
        run_part1(num_MC, workers)
    if run(2):
        run_part2(num_MC, workers)


if __name__ == "__main__":
    main()
