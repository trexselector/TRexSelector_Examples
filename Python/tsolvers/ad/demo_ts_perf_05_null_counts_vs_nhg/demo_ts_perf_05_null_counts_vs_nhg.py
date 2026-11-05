# ==============================================================================
# demo_ts_perf_05_null_counts_vs_nhg.py
# ==============================================================================
#
# Number of included and selected null variables vs. the negative hypergeometric
# (NHG) distribution, for all demo T-solvers (numerical verification of
# Corollary 2, A-I and A-II of the T-Rex supplement, Figures 10 and 11).
#
# Mirrors cpp/tsolvers/ad/demo_ts_perf_05_null_counts_vs_nhg/demo_ts_perf_05_null_counts_vs_nhg.cpp:
# n = 150, p = 300, 5 true coefficients equal to 1 on a fixed random support,
# SNR = 1. Every run draws one data set and K = 20 random experiments, each with
# L = p fresh standard normal dummies, and runs every solver to T = 20 included
# dummies. The included null variables of one experiment follow
# NHG(p0 + L, p0, T) when nulls and dummies are exchangeable. Their relative
# occurrences over the K experiments give the selected null variables V(v) at
# the voting levels v -> 1, 0.75 and 0.5, which the NHG should dominate.
#
# Usage: demo_ts_perf_05_null_counts_vs_nhg.py [num_MC] [workers]   (default 500, 6)
#
# ==============================================================================

import math
import os
import sys
import time

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.dirname(os.path.dirname(_THIS_DIR))  # tsolvers/, the shared helpers
if _PARENT_DIR not in sys.path:
    sys.path.insert(0, _PARENT_DIR)

from ts_demo_utils import (
    display_path,
    gen_synthetic_data,
    output_dir,
    print_section_header,
)
from ts_perf_solvers import demo_solvers, file_suffix
from ts_perf_utils import (
    draw_support,
    parse_positive,
    run_parallel,
    included_dummies,
    selected_predictors,
)

# Setup
n, p = 150, 300
num_active = 5
p0 = p - num_active             # null variables
num_dummies = p                 # L = p per random experiment
K = 20                          # random experiments
T_stop = 20                     # included dummies
beta_active = 1.0
snr = 1.0
support_seed = 12345
data_seed_offset = 1000         # run mc uses seed data_seed_offset + mc

# Voting levels v -> 1 (all K experiments), 0.75 and 0.5 as minimum occurrence counts (Phi > v)
LEVEL_NAMES = ["v1", "v0.75", "v0.5"]
LEVEL_MIN_COUNTS = [K, 16, 11]

SUPPORT = draw_support(p, num_active, support_seed)
SUPPORT_SET = set(SUPPORT)
COEFS = [beta_active] * num_active


def nhg_pmf():
    """PMF of the number of nulls drawn before the T-th dummy, NHG(p0 + L, p0, T), for 0, ..., p0."""
    total = math.comb(p0 + num_dummies, p0)
    return np.array([math.comb(v + T_stop - 1, v) * math.comb(p0 + num_dummies - v - T_stop, p0 - v) / total
                     for v in range(p0 + 1)])


def run_one(mc):
    """One run: one data set with K * L dummies, block k for experiment k, every solver to T."""
    from trexselector.ml_methods import LpNormScaler, NormType

    seed = data_seed_offset + mc
    X, D_all, y = gen_synthetic_data(n, p, K * num_dummies, SUPPORT, COEFS, snr, seed=seed)
    # Center and scale once in front (column-wise, so the blocks stay independent)
    LpNormScaler(NormType.L2, center=True).fit_transform_inplace(X)
    LpNormScaler(NormType.L2, center=True).fit_transform_inplace(D_all)
    y -= y.mean()

    results = []
    for spec in demo_solvers():
        occurrences = np.zeros(p, dtype=int)
        included, reached = [], 0.0
        try:
            for k in range(K):
                D = D_all[:, k * num_dummies:(k + 1) * num_dummies]  # column block, still Fortran-contiguous
                solver = spec.make(X, D, y)
                solver.setTieSeed(seed * K + k)
                solver.executeStep(T_stop, True)
                selected = selected_predictors(solver, p)
                occurrences[selected] += 1
                included.append(sum(j not in SUPPORT_SET for j in selected))
                reached += 1.0 if included_dummies(solver, p) >= T_stop else 0.0
            null_mask = np.ones(p, dtype=bool)
            null_mask[SUPPORT] = False
            selected_nulls = [int(np.sum((occurrences >= c) & null_mask)) for c in LEVEL_MIN_COUNTS]
            results.append({"included": included, "selected": selected_nulls, "reached": reached / K})
        except (RuntimeError, ValueError):  # a solver that throws in one experiment leaves the run out
            results.append(None)
    return results


def frequencies(values):
    """Frequencies of the counts 0, ..., p0."""
    return np.bincount(np.asarray(values, dtype=int), minlength=p0 + 1)[:p0 + 1]


def moments(weights):
    """Mean and standard deviation of a histogram over 0, ..., p0."""
    w = np.asarray(weights, dtype=float)
    total = w.sum()
    if total <= 0:
        return float("nan"), float("nan")
    x = np.arange(w.size)
    mean = float((w * x).sum() / total)
    return mean, float(np.sqrt(max(0.0, (w * x * x).sum() / total - mean * mean)))


def ecdf(freq):
    """Empirical CDF over 0, ..., p0 from frequencies."""
    total = freq.sum()
    return np.cumsum(freq) / total if total > 0 else np.full(freq.size, np.nan)


def main():
    num_MC = parse_positive(sys.argv, 1, 500)
    workers = parse_positive(sys.argv, 2, 6)
    solvers = demo_solvers()
    pmf = nhg_pmf()
    nhg_cdf = np.cumsum(pmf)
    nhg_mean, nhg_sd = moments(pmf)

    print()
    print_section_header("T-Solver Performance: Null Counts vs. NHG")
    print(f"n = {n}, p = {p}, p0 = {p0}, L = p = {num_dummies}, true coefficients: {num_active} x {beta_active:g}, "
          f"SNR = {snr:g}")
    print(f"K = {K} random experiments, T = {T_stop}, voting levels v -> 1, 0.75, 0.5")
    print(f"NHG(p0 + L, p0, T): mean T p0 / (L + 1) = {nhg_mean:.2f}, sd = {nhg_sd:.2f}")
    print("True support: " + " ".join(str(j) for j in SUPPORT))
    print(f"{len(solvers)} solvers, {num_MC} runs on {workers} worker processes (single-threaded solvers)\n")

    t_start = time.perf_counter()
    runs = run_parallel(run_one, range(num_MC), workers)
    wall = time.perf_counter() - t_start

    failures = sum(r is None for results in runs for r in results)
    freq_included, freq_selected, reached = [], [], []
    for s in range(len(solvers)):
        ok = [results[s] for results in runs if results[s] is not None]
        freq_included.append(frequencies([c for r in ok for c in r["included"]]))
        freq_selected.append([frequencies([r["selected"][lv] for r in ok]) for lv in range(len(LEVEL_NAMES))])
        reached.append(float(np.mean([r["reached"] for r in ok])) if ok else float("nan"))

    out = [f"Included null variables at T = {T_stop} (pooled over runs and experiments)",
           f"{'solver':<22}{'mean':>8}{'sd':>8}{'KS D':>8}{'reached':>10}",
           "-" * (22 + 8 * 3 + 10),
           f"{'NHG':<22}{nhg_mean:>8.2f}{nhg_sd:>8.2f}"]
    for s, spec in enumerate(solvers):
        mean, sd = moments(freq_included[s])
        ks = float(np.max(np.abs(ecdf(freq_included[s]) - nhg_cdf)))
        out.append(f"{spec.name:<22}{mean:>8.2f}{sd:>8.2f}{ks:>8.3f}{100.0 * reached[s]:>8.1f} %")
    out.append("")
    # Stochastic dominance: excess = max over x of F_NHG(x) - F_V(x), positive where the NHG CDF lies above
    out.append("Selected null variables V(v) over the runs: mean and largest excess of the NHG CDF over the ECDF")
    out.append(f"{'solver':<22}" + "".join(f"{'mean ' + nm:>13}" for nm in LEVEL_NAMES)
               + "".join(f"{'excess ' + nm:>13}" for nm in LEVEL_NAMES))
    out.append("-" * (22 + 13 * 2 * len(LEVEL_NAMES)))
    for s, spec in enumerate(solvers):
        means = "".join(f"{moments(freq_selected[s][lv])[0]:>13.2f}" for lv in range(len(LEVEL_NAMES)))
        excess = "".join(f"{max(0.0, float(np.max(nhg_cdf - ecdf(freq_selected[s][lv])))):>13.3f}"
                         for lv in range(len(LEVEL_NAMES)))
        out.append(f"{spec.name:<22}" + means + excess)
    out.append("")
    out.append(f"Failed solver runs: {failures}")
    out.append(f"Wall time: {wall:.1f} s")
    text = "\n".join(out) + "\n"
    print(text, end="")

    # Result files: frequencies of the counts, the NHG reference is computed by the plotter
    stem = f"null_counts_vs_nhg_mc{num_MC}{file_suffix()}"  # suffixed for a TREX_SOLVERS selection
    data_dir = output_dir(_THIS_DIR)
    csv_path = os.path.join(data_dir, stem + ".csv")
    with open(csv_path, "w") as f:
        f.write("solver,family,p0,L,T,K,statistic,count,frequency\n")

        def write_rows(spec, statistic, freq):
            f.writelines(f"{spec.name},{spec.family},{p0},{num_dummies},{T_stop},{K},{statistic},{v},{freq[v]}\n" for v in np.flatnonzero(freq))

        for s, spec in enumerate(solvers):
            write_rows(spec, "included", freq_included[s])
            for lv, nm in enumerate(LEVEL_NAMES):
                write_rows(spec, "selected_" + nm, freq_selected[s][lv])
    txt_path = os.path.join(data_dir, stem + ".txt")
    with open(txt_path, "w") as f:
        f.write(text)

    print(f"\nResults written to:\n  {display_path(csv_path)}\n  {display_path(txt_path)}\n")


if __name__ == "__main__":
    main()
