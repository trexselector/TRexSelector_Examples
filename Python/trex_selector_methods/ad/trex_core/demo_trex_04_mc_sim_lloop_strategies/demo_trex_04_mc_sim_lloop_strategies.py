# ==============================================================================
# demo_trex_04_mc_sim_lloop_strategies.py
# ==============================================================================
#
# Classical T-Rex selector demo — L-loop strategy comparison, repeated across
# four base solvers (TLARS / TOMP / TAFS / TLARS_STAG = TLARS with the stagnation
# stop on).
#
# Mirrors:
# cpp/trex_selector_methods/ad/trex_core/demo_trex_04_mc_sim_lloop_strategies.cpp
# and
# R/trex_selector_methods/ad/trex_core/demo_trex_04_mc_sim_lloop_strategies.R
#
# L-loop strategies (8 rows):
#
#    STANDARD             — Stored; fresh i.i.d. dummy matrix at each L-loop
#                           iteration (conservative default).
#    HCONCAT              — Stored; horizontally expand dummy columns from the
#                           same base draw (faster, prefix-stable).
#    PERMUTATION          — Stored base dummy matrix re-used via deterministic
#                           row permutations per experiment.
#    PERMUTATION_SEEDED   — Seed-derived base + row permutations; nothing
#                           stored. Bit-identical to PERMUTATION per seed.
#    SEEDED               — Seed-derived i.i.d. draws; nothing stored.
#    SKIPL_{5p,10p,20p}   — Skip the L-loop entirely; fixed
#                           L = max_dummy_multiplier in {5, 10, 20} x p.
#
#  Scenario:
#  High-dimensional (n = 300, p = 1000, s = 10).
#  SNR sweep: {0.1, 0.2, ..., 2.0, 5.0}  (21 values).
#  Base solvers (outer sweep, one result file pair per solver):
#    TLARS, TOMP, TAFS (rho_afs = 0.3), TLARS_STAG (stagnation stop on).
#  Two support scenarios:
#    random support — redrawn per trial;
#    block support  — contiguous block {0, 1, ..., s-1}  (0-based).
#  Reports averaged FDR / TPR / Avg L / Avg T per L-strategy x SNR.
#
# ==============================================================================

import sys
import os

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.dirname(_THIS_DIR)
for _p in (_THIS_DIR, _PARENT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402

from trex_sim_common import run_mc_trex, save_mc_results  # noqa: E402
from trex_sim_common import resolve_workers  # noqa: E402

# ==============================================================================
# Global parameters
# ==============================================================================

_OUT_DIR     = os.path.join(_THIS_DIR, "simulation_results", "data")
_NUM_WORKERS = resolve_workers()
_NUM_MC      = 200

# ==============================================================================
# L-loop strategies to compare (mirrors make_lloop_strategies() in C++)
# ==============================================================================
#
# name           : row label in output tables and CSV.
# strategy       : string passed to lloop_strategy in _make_trex_ctrl_from_dict.
# max_dummy_mult : upper bound for the adaptive L-loop (non-SKIPL) or the fixed
#                  L value for SKIPL (as a multiple of p).

def _strat(name, strategy, max_dummy_mult=10):
    return dict(name=name, strategy=strategy, max_dummy_mult=max_dummy_mult)

L_STRATEGIES = [
    _strat("STANDARD",             "STANDARD"),
    _strat("HCONCAT",              "HCONCAT"),
    _strat("PERMUTATION",          "PERMUTATION"),
    _strat("PERMUTATION_SEEDED", "PERMUTATION_SEEDED"),
    _strat("SEEDED",             "SEEDED"),
    _strat("SKIPL_5p",  "SKIPL",  5),
    _strat("SKIPL_10p", "SKIPL", 10),
    _strat("SKIPL_20p", "SKIPL", 20),
]

# Base solvers for the outer sweep (rho_afs only meaningful for TAFS).
# Deviations between the solvers will be detectable in the results.
# TLARS_STAG is TLARS with the stagnation stop ON (AUTO leaves it off for
# TLARS): does the SKIPL overshoot come from the LARS path or the missing stop?
_SOLVERS = [
    dict(solver_name="TLARS", name="TLARS", lambda2=0.0, rho_afs=0.0, ncgmp_variant=0),
    dict(solver_name="TOMP",  name="TOMP",  lambda2=0.0, rho_afs=0.0, ncgmp_variant=0),
    dict(solver_name="TAFS",  name="TAFS",  lambda2=0.0, rho_afs=0.3, ncgmp_variant=0),
    dict(solver_name="TLARS", name="TLARS_STAG", lambda2=0.0, rho_afs=0.0, ncgmp_variant=0,
         stagnation_stop=True),
]


# ==============================================================================
# MC simulation — L-loop strategy comparison
# ==============================================================================

def trx_04_lloop_strategies(num_MC=_NUM_MC, num_workers=_NUM_WORKERS,
                            rnd_coef=False, block_support=False):

    n = 300
    p = 1000
    s = 10
    tFDR       = 0.1
    snr_values = list(np.round(np.arange(0.1, 2.1, 0.1), 1)) + [5.0]  # 21 values

    support_label = "block" if block_support else "random"
    print(f"\n{'=' * 70}")
    print("MC Simulation — L-Strategy Comparison  |  TLARS / TOMP / TAFS / TLARS_STAG")
    print("  High-dimensional (p > n)")
    print(f"  n = {n},  p = {p},  s = {s},  num_MC = {num_MC}  [{support_label} support]\n")

    # Block support: contiguous 0-based {0, 1, ..., s-1}  (fixed across all trials)
    block_supp = np.arange(s, dtype=np.intp) if block_support else None

    n_strat = len(L_STRATEGIES)
    n_snr   = len(snr_values)
    strat_names = [st["name"] for st in L_STRATEGIES]

    # Solver sweep — one result file pair (txt + csv) per base solver
    for solver_desc in _SOLVERS:

        # Result containers: strategy x SNR (reset per solver)
        fdr_mat   = np.zeros((n_strat, n_snr))
        tpr_mat   = np.zeros((n_strat, n_snr))
        avg_L_mat = np.zeros((n_strat, n_snr))
        avg_T_mat = np.zeros((n_strat, n_snr))

        # Strategy sweep: iterate over L-loop strategies and SNR values
        for si, strat in enumerate(L_STRATEGIES):
            print(f"{'=' * 50}")
            print(f"Solver: {solver_desc['name']}  |  L-strategy: {strat['name']}")
            print(f"{'=' * 50}\n")

            # Per-strategy ctrl — max_dummy_multiplier and lloop_strategy vary.
            # Values taken from the C++ reference — every option explicit.
            # tloop_stagnation_stop = None is the tri-state AUTO (C++ nullopt):
            # the selector resolves it per solver family (disabled for TLARS,
            # enabled for the greedy TOMP/TAFS). TLARS_STAG sets it to True.
            # tloop_max_stagnant_steps = 5 is the C++ struct default (only
            # active when the stop is on).
            base_ctrl = dict(
                K                        = 20,
                max_dummy_multiplier     = strat["max_dummy_mult"],
                use_max_T_stop           = True,
                opt_threshold            = 0.75,
                lloop_strategy           = strat["strategy"],
                tloop_stagnation_stop    = solver_desc.get("stagnation_stop"),  # None: AUTO
                tloop_max_stagnant_steps = 5,
                parallel_rnd_experiments = False,
                use_memory_mapping       = False,
            )

            for snr_idx, snr in enumerate(snr_values):
                base_seed = 24 + snr_idx * 1000
                lbl = f"SNR={snr:.2f} [{strat['name']}]"
                result = run_mc_trex(
                    n=n, p=p, s=s, snr=snr,
                    solver_desc=solver_desc,
                    base_ctrl_dict=base_ctrl,
                    tFDR=tFDR,
                    base_seed=base_seed,
                    num_MC=num_MC,
                    max_workers=num_workers,
                    fixed_support=block_supp,   # None → random per trial
                    amplitude=1.0,
                    rnd_coef=rnd_coef,
                    track_L_T=True,
                    label=lbl,
                )
                fdr_mat[si, snr_idx]   = result["mean_FDR"]
                tpr_mat[si, snr_idx]   = result["mean_TPR"]
                avg_L_mat[si, snr_idx] = result["mean_L"]
                avg_T_mat[si, snr_idx] = result["mean_T"]
            print()

        stem = (f"demo_trex_04_lloop_strategies_results_n{n}_p{p}"
                f"_{str(solver_desc['name']).lower()}_{support_label}_support")
        save_mc_results(
            _OUT_DIR, stem, num_MC,
            strat_names, snr_values,
            fdr_mat, tpr_mat, avg_L_mat, avg_T_mat,
        )
        print()


# ==============================================================================
# Main
# ==============================================================================

if __name__ == "__main__":
    trx_04_lloop_strategies(block_support=False)
    print("\ndemo_trex_04_mc_sim_lloop_strategies.py complete.")
