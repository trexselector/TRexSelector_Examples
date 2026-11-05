# ==============================================================================
# demo_trex_da_06_mc_sim_groups.py
# ==============================================================================
#
# DA-TRex MC simulation for the prior-groups DGP.
#
# The predictors carry a known 3-level hierarchical group structure (groups of
# 10 / 50 / 250, fine -> coarse), passed to the selector via
# da_control.prior_groups. The PRIOR_GROUPS method (redefined 2026-07-17)
# treats these labels as merge CONSTRAINTS: the selector sub-clusters within
# their finest common refinement (hc_linkage, correlation distance) and
# calibrates over a data-driven ascending rho grid terminated by the
# conservative rho = 1 singleton anchor. Data is drawn from a multi-level
# latent-factor model with Toeplitz leaf blocks.
#
#   Part 1: SNR sweep (n=300, p=1000, s=10, Random support, 200 MC, 3 solvers).
#
# Mirrors:
# cpp/trex_selector_methods/ad/trex_da/demo_trex_da_06_mc_sim_groups
# (same DGP, control parameters, sweep axis, and result files).
#
# Results are saved to simulation_results/data/ as TXT and CSV files.
# ==============================================================================

import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.dirname(_THIS_DIR)
for _p in (_THIS_DIR, _PARENT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from da_sim_common import default_solvers, fmt_num, run_snr_sweep  # noqa: E402

RUN_PART_1 = True

OUT_DIR = os.path.join(_THIS_DIR, "simulation_results", "data")

# Simulation config (mirrors cpp SimConfig defaults)
N, P, S = 300, 1000, 10
AMPLITUDE = 3.0
TFDR = 0.2
K = 20
NUM_MC = 200
BASE_SEED = 2026
MAX_GAP = 20   # unused by the Random policy; kept for cpp parity

# 3-level hierarchy: groups of 10, 50, 250 (fine -> coarse) and their
# cumulative variance masses, fine -> coarse.
RHO_LEVELS = [0.55, 0.25, 0.10]
PHI_LEAF = 0.50


def make_group_structure(p):
    """Level 1: groups of 10; level 2: groups of 50; level 3: groups of 250."""
    return [[j // 10 for j in range(p)],
            [j // 50 for j in range(p)],
            [j // 250 for j in range(p)]]


# ==============================================================================
# Part 1: MC SNR sweep
# DGP: dgp_groups_toeplitz_leaf with 3-level hierarchy, Random support.
# Fixed: n=300, p=1000, s=10, tFDR=0.2, K=20, num_MC=200.
# Swept: SNR in {0.1, 0.2, 0.5, 1.0, 2.0, 5.0}.
# ==============================================================================

def part_1_snr_sweep():
    groups = make_group_structure(P)

    # The prior groups act as merge CONSTRAINTS: the selector sub-clusters
    # within their finest common refinement (the groups of 10) and calibrates
    # over a data-driven rho grid ending at the conservative rho = 1 anchor.
    da_spec = {"prior_groups": groups}

    run_snr_sweep(
        OUT_DIR, "da_groups_toeplitz_leaf",
        [0.1, 0.2, 0.5, 1.0, 2.0, 5.0], NUM_MC, TFDR, BASE_SEED,
        default_solvers(),
        lambda snr: dict(n=N, p=P, dgp="dgp_groups_toeplitz_leaf",
                         kwargs=dict(amplitude=AMPLITUDE, groups=groups,
                                     rho_levels=RHO_LEVELS, phi=PHI_LEAF,
                                     snr=snr),
                         support=None, s=S),
        da_spec, K,
        f"Prior Groups + Toeplitz leaf, n={N}, p={P}, support=Random"
        f", phi_leaf={fmt_num(PHI_LEAF)}, max_gap={MAX_GAP}")


# ==============================================================================
# Main
# ==============================================================================

if __name__ == "__main__":
    if RUN_PART_1:
        part_1_snr_sweep()
    print("\nPrior Groups MC simulations complete.")
