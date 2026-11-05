# ==============================================================================
# demo_trex_da_08_mc_sim_equi_and_bt.py
# ==============================================================================
#
# DA-TRex+EQUI Monte Carlo 2D sweep (SNR x rho) for the equicorrelated DGP.
#
# One 2D sweep: for each SNR column in {0.5, 1, 2, 5}, run a full rho sweep
# over {0, 0.025, 0.05, 0.1, 0.2, 0.4, 0.7, 0.9} (dense near 0 to locate the
# DA suppression cliff) with the Equi DGP (n=300, p=1000, Random support),
# 3 DA solvers + the 3 base T-Rex comparison rows.
#
# Rationale: equicorrelation ties every predictor to one shared factor, so
# DA-EQUI suppresses all candidates as soon as rho is noticeably above 0,
# independent of signal strength, while base T-Rex loses FDR control at the
# same tiny rho. The 2D layout shows the cliff location, its SNR independence,
# and the rho=0 working anchor in a single figure.
#
# Earlier revisions of this demo also carried BT (hierarchical-block) SNR and
# linkage sweeps; those were dropped as redundant — the BT method is covered
# on better-suited block designs in Demos 02-05. (The folder keeps the cpp
# name "equi_and_bt" for parity.)
#
# Mirrors cpp/trex_selector_methods/ad/trex_da/demo_trex_da_08_mc_sim_equi_and_bt
# (same DGP, control parameters, sweep axes, and result files — one TXT/CSV
# pair per SNR column).
# Results are saved to simulation_results/data/.
# ==============================================================================

import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.dirname(_THIS_DIR)
for _p in (_THIS_DIR, _PARENT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from da_sim_common import default_solvers, fmt_num, run_param_sweep  # noqa: E402

RUN_PART_1 = True

OUT_DIR = os.path.join(_THIS_DIR, "simulation_results", "data")

# Simulation config (mirrors cpp)
N, P, S = 300, 1000, 10
AMPLITUDE = 3.0
TFDR = 0.2
K = 20
NUM_MC = 200
BASE_SEED = 2026

DA_SPEC = {"method": "EQUI"}


# ==============================================================================
# 2D sweep: SNR (outer, one output file per value) x rho (inner sweep)
# ==============================================================================

def part_1_2d_snr_rho_sweep():
    snr_grid = [0.5, 1.0, 2.0, 5.0]
    rho_grid = [0.0, 0.025, 0.05, 0.1, 0.2, 0.4, 0.7, 0.9]

    for si, snr in enumerate(snr_grid):
        # Filesystem-safe SNR tag: 0.5 -> "0p5", 2.0 -> "2".
        snr_tag = fmt_num(snr).replace(".", "p")

        # Disjoint seed blocks per SNR column (inner sweep spans gi * 10000),
        # so the columns are statistically independent.
        col_seed = BASE_SEED + si * 1000000

        run_param_sweep(
            OUT_DIR, f"da_equi_rho_snr{snr_tag}", "Rho", rho_grid, NUM_MC,
            TFDR, col_seed, default_solvers("+EQUI"),
            lambda rho, snr=snr: dict(n=N, p=P, dgp="dgp_equi_snr",
                                      kwargs=dict(amplitude=AMPLITUDE, rho=rho,
                                                  snr=snr),
                                      support=None, s=S),
            DA_SPEC, K,
            f"Equi 2D sweep, SNR={fmt_num(snr)}, n={N}, p={P}, support=Random",
            include_base_trex=True)


# ==============================================================================
# Main
# ==============================================================================

if __name__ == "__main__":
    if RUN_PART_1:
        part_1_2d_snr_rho_sweep()
    print("\nEqui 2D (SNR x rho) MC simulation complete.")
