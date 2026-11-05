# ==============================================================================
# demo_trex_da_03_mc_sim_ar1_blocks.py
# ==============================================================================
#
# DA-TRex+BT MC simulations for the AR(1)-block DGP using BT aggregation.
#
# DGP: dgp_ar1_block_snr — block-diagonal AR(1) covariance. One active variable
# per block (OnePerBlock support). All M blocks are active (s = M).
#
# Every part loops over linkage in {Single, Complete, Average}. All 3 solvers
# (TLARS, TAFS, TOMP) run inside each linkage iteration.
#
#   Part 1: SNR sweep    SNR in {0.1, 0.2, 0.5, 0.6, 1.0, 2.0, 5.0}.
#   Part 2: rho sweep    rho in {0.0, 0.1, ..., 0.9}.
#   Part 3: Q sweep      Q in {5, 10, ..., 50}; p = M*Q varies; s = M = 5 fixed.
#   Part 4: M sweep      M in {1, 3, 5, 10, 15, 20, 30}; p = M*Q and s = M vary.
#
# Mirrors:
# cpp/trex_selector_methods/ad/trex_da/demo_trex_da_03_mc_sim_ar1_blocks
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

from da_sim_common import (default_solvers, fmt_num,  # noqa: E402
                           make_support_one_per_block, run_param_sweep,
                           run_snr_sweep)

RUN_PART_1 = True
RUN_PART_2 = True
RUN_PART_3 = True
RUN_PART_4 = True

OUT_DIR = os.path.join(_THIS_DIR, "simulation_results", "data")

# Shared base parameters (mirror cpp)
N = 150
M = 5              # number of AR(1) blocks
Q = 5              # block size; base p = M * Q = 25
BASE_P = M * Q     # 25
AMPLITUDE = 1.0
RHO = 0.7
SNR = 2.0
TFDR = 0.2
K = 20
NUM_MC = 200
BASE_SEED = 2026

LINKAGES = ("Single", "Complete", "Average")


# ==============================================================================
# Part 1: MC SNR sweep
# Fixed: n=150, M=5, Q=5 (p=25, s=5), rho=0.7.
# ==============================================================================

def part_1_snr_sweep():
    snr_grid = [0.1, 0.2, 0.5, 0.6, 1.0, 2.0, 5.0]
    support_base = make_support_one_per_block(M, BASE_P, Q)

    for lnk in LINKAGES:
        run_snr_sweep(
            OUT_DIR, f"da_ar1_blocks_snr_{lnk}", snr_grid, NUM_MC, TFDR,
            BASE_SEED, default_solvers("+BT"),
            lambda snr: dict(n=N, p=BASE_P, dgp="dgp_ar1_block_snr",
                             kwargs=dict(amplitude=AMPLITUDE, n_blocks=M,
                                         rho=RHO, snr=snr),
                             support=support_base, s=M),
            {"method": "BT", "hc_linkage": lnk}, K,
            f"AR(1)-block SNR sweep  n={N}  M={M}  Q={Q}  p={BASE_P}  s={M}"
            f"  rho={fmt_num(RHO)}  linkage={lnk}",
            include_base_trex=False)


# ==============================================================================
# Part 2: MC rho sweep
# Fixed: n=150, M=5, Q=5 (p=25, s=5), SNR=2.
# ==============================================================================

def part_2_rho_sweep():
    rho_grid = [round(0.1 * i, 1) for i in range(10)]
    support_base = make_support_one_per_block(M, BASE_P, Q)

    for lnk in LINKAGES:
        run_param_sweep(
            OUT_DIR, f"da_ar1_blocks_rho_{lnk}", "Rho", rho_grid, NUM_MC, TFDR,
            BASE_SEED, default_solvers("+BT"),
            lambda rho: dict(n=N, p=BASE_P, dgp="dgp_ar1_block_snr",
                             kwargs=dict(amplitude=AMPLITUDE, n_blocks=M,
                                         rho=rho, snr=SNR),
                             support=support_base, s=M),
            {"method": "BT", "hc_linkage": lnk}, K,
            f"AR(1)-block rho sweep  n={N}  M={M}  Q={Q}  p={BASE_P}  s={M}"
            f"  SNR={fmt_num(SNR)}  linkage={lnk}",
            include_base_trex=False)


# ==============================================================================
# Part 3: MC Q sweep (block size)
# Fixed: n=150, M=5, rho=0.7, SNR=2.  s = M = 5 throughout.
# Swept: Q in {5, 10, ..., 50}; p = M*Q varies.
# Support recomputed per grid point (OnePerBlock on p_val = M*Q).
# ==============================================================================

def part_3_q_sweep():
    q_grid = list(range(5, 51, 5))

    def make_dgp(q):
        q_int = int(q)
        p_val = M * q_int
        return dict(n=N, p=p_val, dgp="dgp_ar1_block_snr",
                    kwargs=dict(amplitude=AMPLITUDE, n_blocks=M, rho=RHO,
                                snr=SNR),
                    support=make_support_one_per_block(M, p_val, q_int), s=M)

    for lnk in LINKAGES:
        run_param_sweep(
            OUT_DIR, f"da_ar1_blocks_Q_{lnk}", "Q", q_grid, NUM_MC, TFDR,
            BASE_SEED, default_solvers("+BT"), make_dgp,
            {"method": "BT", "hc_linkage": lnk}, K,
            f"AR(1)-block Q sweep  n={N}  M={M}  s={M}  rho={fmt_num(RHO)}"
            f"  SNR={fmt_num(SNR)}  p=M*Q varies  linkage={lnk}",
            include_base_trex=False)


# ==============================================================================
# Part 4: MC M sweep (number of blocks)
# Fixed: n=150, Q=5, rho=0.7, SNR=2.
# Swept: M in {1, 3, 5, 10, 15, 20, 30}; p = M*Q and s = M both vary.
# Support recomputed per grid point.
# ==============================================================================

def part_4_m_sweep():
    m_grid = [1, 3, 5, 10, 15, 20, 30]

    def make_dgp(m):
        m_int = int(m)
        p_val = m_int * Q
        return dict(n=N, p=p_val, dgp="dgp_ar1_block_snr",
                    kwargs=dict(amplitude=AMPLITUDE, n_blocks=m_int, rho=RHO,
                                snr=SNR),
                    support=make_support_one_per_block(m_int, p_val, Q),
                    s=m_int)

    for lnk in LINKAGES:
        run_param_sweep(
            OUT_DIR, f"da_ar1_blocks_M_{lnk}", "M", m_grid, NUM_MC, TFDR,
            BASE_SEED, default_solvers("+BT"), make_dgp,
            {"method": "BT", "hc_linkage": lnk}, K,
            f"AR(1)-block M sweep  n={N}  Q={Q}  rho={fmt_num(RHO)}"
            f"  SNR={fmt_num(SNR)}  p=M*Q and s=M vary  linkage={lnk}",
            include_base_trex=False)


# ==============================================================================
# Main
# ==============================================================================

if __name__ == "__main__":
    if RUN_PART_1:
        part_1_snr_sweep()
    if RUN_PART_2:
        part_2_rho_sweep()
    if RUN_PART_3:
        part_3_q_sweep()
    if RUN_PART_4:
        part_4_m_sweep()
    print("\nAR(1)-block BT MC simulation complete.")
