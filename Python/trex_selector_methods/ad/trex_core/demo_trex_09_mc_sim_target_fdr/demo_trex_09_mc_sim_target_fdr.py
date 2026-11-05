# ==============================================================================
# demo_trex_09_mc_sim_target_fdr.py
# ==============================================================================
#
# Classical T-Rex selector demo — achieved FDR and TPR as a function of the
# target FDR, for three base solvers (TLARS, TOMP, TAFS) and four SNRs.
#
# Mirrors:
#  cpp/trex_selector_methods/ad/trex_core/demo_trex_09_mc_sim_target_fdr.cpp.
#
# The T-Rex selector controls the FDR at the target level tFDR. The demo sweeps
# tFDR and records the achieved FDR (the mean false discovery proportion) and
# TPR, so FDR control shows as the FDR curve on or below the diagonal
# FDR = tFDR, and the TPR curve shows what each level of control costs in power.
#
# Data as in demo 03: n = 300, p = 1,000, s = 10 unit coefficients on a support
# redrawn per trial, Normal predictors and noise.
# Selector as in demo 05: K = 20, max_dummy_multiplier = 10, Normal dummies,
# STANDARD L-loop, use_max_T_stop, stagnation stop AUTO (off for TLARS, on for
# TOMP and TAFS), rho_afs = 0.3 for TAFS.
#
# Grid: SNR in {0.5, 1, 2, 5} x tFDR in {0.01, 0.025, 0.05, 0.075, 0.1, 0.15,
# 0.2, 0.25, 0.3, 0.4, 0.5}, num_MC = 200 trials per cell (the first
# command-line argument overrides it, e.g. 2 for a smoke run).
#
# Seeds: trial mc of an SNR uses the data seed 24 + 1000 * snr_index + mc for
# every tFDR and every solver, so the curves along tFDR compare the same data
# sets, and the selector is seeded with the same trial seed (reproducible).
#
# Output (simulation_results/data/): one table per solver in
# demo_trex_09_mc_sim_target_fdr.txt and the tidy CSV
# demo_trex_09_mc_sim_target_fdr.csv with the columns
# solver, snr, tfdr, metric, mean, sd (metrics FDR, TPR, AvgL, AvgT; sd for FDR
# and TPR only). Every finished cell is cached, so a stopped run resumes.
#
# ==============================================================================

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
_FILE_STEM   = "demo_trex_09_mc_sim_target_fdr"
_NUM_WORKERS = resolve_workers()
_NUM_MC      = 200

N_OBS       = 300
P_VARS      = 1000
S_TRUE      = 10
SEED_OFFSET = 24  # as demos 02-05

SNR_VALUES  = [0.5, 1.0, 2.0, 5.0]
TFDR_VALUES = [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]

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


# ==============================================================================
# Output: per-solver table (console + .txt) and tidy CSV
# ==============================================================================

def save_results(results, num_MC):
    """Print the per-solver tables and write the .txt and .csv files.

    results[solver_name][si][ti] is the run_mc_trex() dict of one cell.
    """
    os.makedirs(_OUT_DIR, exist_ok=True)
    lines = [
        "",
        "=" * 78,
        f"=== T-Rex achieved FDR / TPR vs. target FDR (mean over {num_MC} trials) ===",
        f"    n = {N_OBS}, p = {P_VARS}, s = {S_TRUE}, K = 20, STANDARD L-loop, Normal dummies",
        "=" * 78,
    ]
    label_w, col_w = 16, 8
    for solver in _SOLVERS:
        res = results[solver["name"]]
        lines.append("")
        lines.append(solver["name"])
        lines.append(f"{'  tFDR':<{label_w}}" + "".join(f"{t:>{col_w}.3f}" for t in TFDR_VALUES))
        for si, snr in enumerate(SNR_VALUES):
            for metric, key in (("FDR", "mean_FDR"), ("TPR", "mean_TPR")):
                label = f"  {metric} SNR {snr:g}"
                lines.append(f"{label:<{label_w}}" + "".join(f"{cell[key]:>{col_w}.3f}" for cell in res[si]))
    lines.append("")
    text = "\n".join(lines) + "\n"
    print(text)

    txt_path = os.path.join(_OUT_DIR, f"{_FILE_STEM}.txt")
    with open(txt_path, "w") as fh:
        fh.write(text)
    print(f"[Info] TXT saved to: {txt_path}")

    csv_path = os.path.join(_OUT_DIR, f"{_FILE_STEM}.csv")
    with open(csv_path, "w") as fh:
        fh.write("solver,snr,tfdr,metric,mean,sd\n")
        for solver in _SOLVERS:
            res = results[solver["name"]]
            for si, snr in enumerate(SNR_VALUES):
                for ti, tfdr in enumerate(TFDR_VALUES):
                    cell = res[si][ti]
                    key = f"{solver['name']},{snr:.6f},{tfdr:.6f},"
                    fh.write(f"{key}FDR,{cell['mean_FDR']:.6f},{cell['sd_FDR']:.6f}\n")
                    fh.write(f"{key}TPR,{cell['mean_TPR']:.6f},{cell['sd_TPR']:.6f}\n")
                    fh.write(f"{key}AvgL,{cell['mean_L']:.6f},\n")
                    fh.write(f"{key}AvgT,{cell['mean_T']:.6f},\n")
    print(f"[Info] CSV saved to: {csv_path}\n")


# ==============================================================================
# MC simulation — achieved FDR / TPR vs. target FDR
# ==============================================================================

def trx_09_target_fdr(num_MC=_NUM_MC, num_workers=_NUM_WORKERS):
    """Sweep tFDR x SNR for TLARS, TOMP and TAFS."""
    print(f"\n{'=' * 70}")
    print("MC Simulation — achieved FDR / TPR vs. target FDR  |  TLARS / TOMP / TAFS")
    print(f"  n = {N_OBS},  p = {P_VARS},  s = {S_TRUE},  num_MC = {num_MC}  [support redrawn per trial]")
    print(f"  SNR grid: {len(SNR_VALUES)} values,  tFDR grid: {len(TFDR_VALUES)} values\n")

    results = {}
    for solver in _SOLVERS:
        print(f"{'=' * 60}\nSolver: {solver['name']}\n{'=' * 60}")
        res = []
        for si, snr in enumerate(SNR_VALUES):
            base_seed = SEED_OFFSET + si * 1000
            row = []
            for tfdr in TFDR_VALUES:
                row.append(run_mc_trex(
                    n=N_OBS, p=P_VARS, s=S_TRUE, snr=snr,
                    solver_desc=solver,
                    base_ctrl_dict=_BASE_CTRL,
                    tFDR=tfdr,
                    base_seed=base_seed,
                    num_MC=num_MC,
                    max_workers=num_workers,
                    fixed_support=None,   # redrawn per trial
                    amplitude=1.0,
                    track_L_T=True,
                    label=f"SNR={snr:.2f} tFDR={tfdr:.3f} [{solver['name']}]",
                    seeded_selector=True,
                ))
            res.append(row)
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
    trx_09_target_fdr(num_MC=_num_MC)
    print("\ndemo_trex_09_mc_sim_target_fdr.py complete.")
