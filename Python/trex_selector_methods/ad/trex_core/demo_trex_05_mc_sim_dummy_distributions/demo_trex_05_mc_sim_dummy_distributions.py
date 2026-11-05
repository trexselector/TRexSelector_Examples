# ==============================================================================
# demo_trex_05_mc_sim_dummy_distributions.py
# ==============================================================================
#
# Classical T-Rex selector demo — dummy distribution comparison across three
# base solvers (TLARS, TOMP, TAFS), with the STANDARD L-loop strategy fixed.
#
# Mirrors:
# cpp/trex_selector_methods/ad/trex_core/demo_trex_05_mc_sim_dummy_distributions.cpp
# and
# R/trex_selector_methods/ad/trex_core/demo_trex_05_mc_sim_dummy_distributions.R
#
# Base solvers compared (one result file pair per solver):
#   TLARS — equiangular LARS path (stagnation stop AUTO-resolves to disabled).
#   TOMP  — greedy orthogonal matching pursuit (stagnation stop AUTO-resolves
#           to enabled).
#   TAFS  — greedy adaptive forward selection, rho_afs = 0.3 (stagnation stop
#           AUTO-resolves to enabled).
#
# The T-Rex FDR calibration rests on exchangeability between real null
# predictors and the injected dummy variables. Since dummies are centered and
# L2-normalized before entering the solver, their scale is immaterial — this
# demo probes whether the SHAPE of the dummy distribution (tails, discreteness,
# skewness, sparsity, cross-column dependence) affects the achieved FDR / TPR
# and the calibrated dummy multiplier L / stopping time T.
#
# Demo content (resolved by trex_sim_common._make_dummy_distribution):
#
#    Normal              — N(0, 1); the reference choice.
#    Uniform             — U(-sqrt(3), sqrt(3)); bounded, light tails.
#    Rademacher          — {-1, +1} equiprobable; discrete two-point.
#    StudentT (df 3 / 5) — heavy tails (unit-variance scaled for df > 2).
#    Laplace             — double exponential; heavier-than-normal tails.
#    Gumbel              — extreme-value; skewed (centered at 0).
#    Triangle            — bounded, symmetric; (-sqrt(6), 0, sqrt(6)).
#    Logistic            — between normal and Laplace tails.
#    Mammen              — asymmetric golden-ratio two-point distribution.
#    SparseRademacher    — constrained sparse Rademacher, sparsity s = 0.1.
#    UniformSphere       — uniform on the unit 5-sphere; consecutive 5-column
#                          groups are dependent (norm constraint), a deliberate
#                          probe of the exchangeability axis. The library
#                          requests dummies in multiples of p, so dim must
#                          divide p (the default dim = 3 would throw at
#                          p = 1000, hence dim = 5).
#
#  Scenario:
#  High-dimensional (n = 300, p = 1000, s = 10).
#  SNR sweep: {0.1, 0.2, 0.5, 0.6, 1.0, 2.0, 5.0}  (7 values, mirrors the
#  current C++ grid and its committed results).
#
#  Two support scenarios:
#    random support — redrawn per trial;
#    block support  — contiguous block {0, 1, ..., s-1}  (0-based).
#  Reports averaged FDR / TPR / Avg L / Avg T per distribution x SNR.
#
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
# Dummy distributions to compare
# ==============================================================================
#
# name : row label in output tables and CSV.
# spec : plain (picklable) dict resolved to a ts.DummyDistribution inside each
#        worker by trex_sim_common._make_dummy_distribution().

def _dist(name, dist_type, **params):
    return dict(name=name, spec=dict(type=dist_type, **params))

DISTRIBUTIONS = [
    _dist("Normal",         "Normal"),
    _dist("Uniform",        "Uniform"),
    _dist("Rademacher",     "Rademacher"),
    _dist("StudentT_df3",   "StudentT", df=3.0),
    _dist("StudentT_df5",   "StudentT", df=5.0),
    _dist("Laplace",        "Laplace"),
    _dist("Gumbel",         "Gumbel"),
    _dist("Triangle",       "Triangle"),
    _dist("Logistic",       "Logistic"),
    _dist("Mammen",         "Mammen"),
    _dist("SparseRad_s0.1", "ConstrainedSparseRademacher", s=0.1),
    _dist("UnifSphere_d5",  "UniformSphere", dim=5),
]

# Base solvers for the outer sweep (rho_afs only meaningful for TAFS)
_SOLVERS = [
    dict(solver_name="TLARS", name="TLARS", lambda2=0.0, rho_afs=0.0, ncgmp_variant=0),
    dict(solver_name="TOMP",  name="TOMP",  lambda2=0.0, rho_afs=0.0, ncgmp_variant=0),
    dict(solver_name="TAFS",  name="TAFS",  lambda2=0.0, rho_afs=0.3, ncgmp_variant=0),
]


# ==============================================================================
# MC simulation — dummy distribution comparison
# ==============================================================================

def trx_05_dummy_distributions(num_MC=_NUM_MC, num_workers=_NUM_WORKERS,
                               rnd_coef=False, block_support=False):

    n = 300
    p = 1000
    s = 10
    tFDR       = 0.1
    snr_values = [0.1, 0.2, 0.5, 0.6, 1.0, 2.0, 5.0]  # mirrors current C++ grid

    support_label = "block" if block_support else "random"
    print(f"\n{'=' * 70}")
    print("MC Simulation — Dummy Distribution Comparison  |  TLARS / TOMP / TAFS")
    print("  High-dimensional (p > n)")
    print(f"  n = {n},  p = {p},  s = {s},  num_MC = {num_MC}  [{support_label} support]\n")

    # Block support: contiguous 0-based {0, 1, ..., s-1}  (fixed across all trials)
    block_supp = np.arange(s, dtype=np.intp) if block_support else None

    n_dist = len(DISTRIBUTIONS)
    n_snr  = len(snr_values)
    dist_names = [d["name"] for d in DISTRIBUTIONS]

    # Solver sweep — one result file pair (txt + csv) per solver
    for solver_desc in _SOLVERS:

        # Result containers: distribution x SNR (reset per solver)
        fdr_mat   = np.zeros((n_dist, n_snr))
        tpr_mat   = np.zeros((n_dist, n_snr))
        avg_L_mat = np.zeros((n_dist, n_snr))
        avg_T_mat = np.zeros((n_dist, n_snr))

        # Simulation sweep: iterate over dummy distributions and SNR values
        for di, dist in enumerate(DISTRIBUTIONS):
            print(f"{'=' * 50}")
            print(f"Solver: {solver_desc['name']}  |  Dummy distribution: {dist['name']}")
            print(f"{'=' * 50}\n")

            # Per-distribution ctrl — only dummy_distribution varies. Values taken
            # from the C++ reference (demo_trex_05_mc_sim_dummy_distributions.cpp)
            # — every option explicit.
            # tloop_stagnation_stop = None is the tri-state AUTO (C++ nullopt):
            # resolved per solver family (disabled for TLARS, enabled for the greedy TOMP/TAFS).
            base_ctrl = dict(
                K                        = 20,
                max_dummy_multiplier     = 10,
                use_max_T_stop           = True,
                opt_threshold            = 0.75,
                lloop_strategy           = "STANDARD",
                dummy_distribution       = dist["spec"],
                tloop_stagnation_stop    = None,    # AUTO (C++ nullopt)
                tloop_max_stagnant_steps = 5,
                parallel_rnd_experiments = False,
                use_memory_mapping       = False,
            )

            for snr_idx, snr in enumerate(snr_values):
                base_seed = 24 + snr_idx * 1000
                lbl = f"SNR={snr:.2f} [{solver_desc['name']}/{dist['name']}]"
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
                fdr_mat[di, snr_idx]   = result["mean_FDR"]
                tpr_mat[di, snr_idx]   = result["mean_TPR"]
                avg_L_mat[di, snr_idx] = result["mean_L"]
                avg_T_mat[di, snr_idx] = result["mean_T"]
            print()

        stem = (f"demo_trex_05_dummy_distributions_results_n{n}_p{p}"
                f"_{str(solver_desc['name']).lower()}_{support_label}_support")
        save_mc_results(
            _OUT_DIR, stem, num_MC,
            dist_names, snr_values,
            fdr_mat, tpr_mat, avg_L_mat, avg_T_mat,
        )
        print()


# ==============================================================================
# Main
# ==============================================================================

if __name__ == "__main__":
    trx_05_dummy_distributions(block_support=False)
    print("\ndemo_trex_05_mc_sim_dummy_distributions.py complete.")
