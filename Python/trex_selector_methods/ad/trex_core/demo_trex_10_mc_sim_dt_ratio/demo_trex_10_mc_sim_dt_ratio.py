# ==============================================================================
# demo_trex_10_mc_sim_dt_ratio.py
# ==============================================================================
#
# Classical T-Rex selector demo — FDR and TPR as a function of the sample-size
# ratio c = n / (s ln p), for three base solvers (TLARS, TOMP, TAFS).
#
# Mirrors:
#  cpp/trex_selector_methods/ad/trex_core/demo_trex_10_mc_sim_dt_ratio.cpp.
#
# Sparse recovery needs n to grow like s log p (Donoho-Tanner scaling). The demo
# sets n = round(c * s * ln p) for a grid of ratios c and two p, and records FDR
# and TPR at tFDR = 0.1, so curves of the two p that fall onto each other on the
# c axis confirm that c governs the power of the FDR-controlled selection.
#
# Data as in demo 03 except for n: s = 10 unit coefficients on a support
# redrawn per trial, Normal predictors and noise.
# Selector as in demo 05: K = 20, max_dummy_multiplier = 10, Normal dummies,
# STANDARD L-loop, use_max_T_stop, stagnation stop AUTO (off for TLARS, on for
# TOMP and TAFS), rho_afs = 0.3 for TAFS.
#
# Grid: p in {1,000, 10,000} x c in {1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10} x
# SNR in {0.5, 1, 2}, num_MC = 200 trials per cell (the first command-line
# argument overrides it, e.g. 2 for a smoke run).
#
# Seeds: trial mc of a cell uses the data seed
# 24 + 1000 * snr_index + 100000 * c_index + 10000000 * p_index + mc for every
# solver, and the selector is seeded with the same trial seed (reproducible).
#
# Output (simulation_results/data/): one table per solver and SNR in
# demo_trex_10_mc_sim_dt_ratio.txt and the tidy CSV
# demo_trex_10_mc_sim_dt_ratio.csv with the columns
# solver, snr, p, n, c, metric, mean, sd (metrics FDR, TPR, AvgL, AvgT; sd for
# FDR and TPR only). Every finished cell is cached, so a stopped run resumes.
#
# ==============================================================================

import math
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.dirname(_THIS_DIR)
for _p in (_THIS_DIR, _PARENT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from trex_sim_common import resolve_workers, run_mc_trex

# ==============================================================================
# Global parameters
# ==============================================================================

_OUT_DIR     = os.path.join(_THIS_DIR, "simulation_results", "data")
_FILE_STEM   = "demo_trex_10_mc_sim_dt_ratio"
_NUM_WORKERS = resolve_workers()
_NUM_MC      = 200

S_TRUE      = 10
TFDR        = 0.1
SEED_OFFSET = 24  # as demos 02-05

P_VALUES   = [1000, 10000]
C_VALUES   = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]
SNR_VALUES = [0.5, 1.0, 2.0]

# Base solvers (rho_afs only meaningful for TAFS)
_SOLVERS = [
    dict(solver_name="TLARS", name="TLARS", lambda2=0.0, rho_afs=0.0, ncgmp_variant=0),
    dict(solver_name="TOMP",  name="TOMP",  lambda2=0.0, rho_afs=0.0, ncgmp_variant=0),
    dict(solver_name="TAFS",  name="TAFS",  lambda2=0.0, rho_afs=0.3, ncgmp_variant=0),
]

# Demo 05 configuration with Normal dummies, every option explicit (C++ reference).
# tloop_stagnation_stop = None is the tri-state AUTO (C++ nullopt).
_BASE_CTRL = dict(
    K                        = 20,
    max_dummy_multiplier     = 10,
    use_max_T_stop           = True,
    opt_threshold            = 0.75,
    lloop_strategy           = "STANDARD",
    dummy_distribution       = dict(type="Normal"),
    tloop_stagnation_stop    = None,    # AUTO (C++ nullopt)
    tloop_max_stagnant_steps = 5,
    parallel_rnd_experiments = False,
    use_memory_mapping       = False,
)


def sample_size(c, p):
    """Sample size of a grid point: n = round(c * s * ln p), halves away from zero as C++ std::lround."""
    return math.floor(c * S_TRUE * math.log(p) + 0.5)


# ==============================================================================
# Output: per-solver tables (console + .txt) and tidy CSV
# ==============================================================================

def save_results(results, num_MC):
    """Print the per-solver tables and write the .txt and .csv files.

    results[solver_name][si][pi][ci] is the run_mc_trex() dict of one cell.
    """
    os.makedirs(_OUT_DIR, exist_ok=True)
    lines = [
        "",
        "=" * 78,
        f"=== T-Rex FDR / TPR vs. c = n / (s ln p) (mean over {num_MC} trials) ===",
        f"    s = {S_TRUE}, tFDR = {TFDR:g}, K = 20, STANDARD L-loop, Normal dummies",
        "=" * 78,
    ]
    label_w, col_w = 18, 8
    for solver in _SOLVERS:
        res = results[solver["name"]]
        for si, snr in enumerate(SNR_VALUES):
            lines.append("")
            lines.append(f"{solver['name']}  |  SNR = {snr:g}")
            lines.append(f"{'  c':<{label_w}}" + "".join(f"{c:>{col_w}.1f}" for c in C_VALUES))
            for pi, p in enumerate(P_VALUES):
                lines.append(f"{'  n  p=' + str(p):<{label_w}}"
                             + "".join(f"{sample_size(c, p):>{col_w}d}" for c in C_VALUES))
                for metric, key in (("FDR", "mean_FDR"), ("TPR", "mean_TPR")):
                    label = f"  {metric} p={p}"
                    lines.append(f"{label:<{label_w}}"
                                 + "".join(f"{cell[key]:>{col_w}.3f}" for cell in res[si][pi]))
    lines.append("")
    text = "\n".join(lines) + "\n"
    print(text)

    txt_path = os.path.join(_OUT_DIR, f"{_FILE_STEM}.txt")
    with open(txt_path, "w") as fh:
        fh.write(text)
    print(f"[Info] TXT saved to: {txt_path}")

    csv_path = os.path.join(_OUT_DIR, f"{_FILE_STEM}.csv")
    with open(csv_path, "w") as fh:
        fh.write("solver,snr,p,n,c,metric,mean,sd\n")
        for solver in _SOLVERS:
            res = results[solver["name"]]
            for si, snr in enumerate(SNR_VALUES):
                for pi, p in enumerate(P_VALUES):
                    for ci, c in enumerate(C_VALUES):
                        cell = res[si][pi][ci]
                        key = f"{solver['name']},{snr:.6f},{p},{sample_size(c, p)},{c:.6f},"
                        fh.write(f"{key}FDR,{cell['mean_FDR']:.6f},{cell['sd_FDR']:.6f}\n")
                        fh.write(f"{key}TPR,{cell['mean_TPR']:.6f},{cell['sd_TPR']:.6f}\n")
                        fh.write(f"{key}AvgL,{cell['mean_L']:.6f},\n")
                        fh.write(f"{key}AvgT,{cell['mean_T']:.6f},\n")
    print(f"[Info] CSV saved to: {csv_path}\n")


# ==============================================================================
# MC simulation — FDR / TPR vs. c = n / (s ln p)
# ==============================================================================

def trx_10_dt_ratio(num_MC=_NUM_MC, num_workers=_NUM_WORKERS):
    """Sweep p x c x SNR for TLARS, TOMP and TAFS at tFDR = 0.1."""
    print(f"\n{'=' * 70}")
    print("MC Simulation — FDR / TPR vs. n / (s ln p)  |  TLARS / TOMP / TAFS")
    print(f"  s = {S_TRUE},  tFDR = {TFDR:g},  num_MC = {num_MC}  [support redrawn per trial]")
    for p in P_VALUES:
        print(f"  p = {p}:  n = " + " ".join(str(sample_size(c, p)) for c in C_VALUES))
    print()

    results = {}
    for solver in _SOLVERS:
        print(f"{'=' * 60}\nSolver: {solver['name']}\n{'=' * 60}")
        res = []
        for si, snr in enumerate(SNR_VALUES):
            per_p = []
            for pi, p in enumerate(P_VALUES):
                row = []
                for ci, c in enumerate(C_VALUES):
                    n = sample_size(c, p)
                    base_seed = SEED_OFFSET + si * 1000 + ci * 100_000 + pi * 10_000_000
                    row.append(run_mc_trex(
                        n=n, p=p, s=S_TRUE, snr=snr,
                        solver_desc=solver,
                        base_ctrl_dict=_BASE_CTRL,
                        tFDR=TFDR,
                        base_seed=base_seed,
                        num_MC=num_MC,
                        max_workers=num_workers,
                        fixed_support=None,   # redrawn per trial
                        amplitude=1.0,
                        track_L_T=True,
                        label=f"SNR={snr:.2f} p={p} c={c:.1f} n={n} [{solver['name']}]",
                        seeded_selector=True,
                    ))
                per_p.append(row)
            res.append(per_p)
        results[solver["name"]] = res

    save_results(results, num_MC)


# ==============================================================================
# Main
# ==============================================================================

if __name__ == "__main__":
    _num_MC = int(sys.argv[1]) if len(sys.argv) > 1 else _NUM_MC
    if _num_MC < 1:
        sys.exit("num_MC must be a positive integer")
    print(f"Running with {_NUM_WORKERS} workers")
    trx_10_dt_ratio(num_MC=_num_MC)
    print("\ndemo_trex_10_mc_sim_dt_ratio.py complete.")
