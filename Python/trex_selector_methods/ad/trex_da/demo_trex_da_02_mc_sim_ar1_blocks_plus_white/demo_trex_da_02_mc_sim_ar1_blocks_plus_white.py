# ==============================================================================
# demo_trex_da_02_mc_sim_ar1_blocks_plus_white.py
# ==============================================================================
#
# DA-TRex+BT MC simulations for the AR(1)-block + white-noise DGP.
#
# DGP: dgp_ar1_block_white_snr — M AR(1) blocks (p_ar columns) plus p_white
# i.i.d. N(0,1) white-noise columns; p_total = p_ar + p_white = 1000 fixed.
# Active variables (s = M, one per AR block) lie in the AR part only.
#
# Every part loops over linkage in {Single, Complete, Average}. All 3 solvers
# (TLARS, TAFS, TOMP) run inside each linkage iteration.
#
#   Part 1: SNR sweep    SNR in {0.1, 0.2, 0.5, 0.6, 1.0, 2.0, 5.0}.
#   Part 2: rho sweep    rho in {0.0, 0.1, ..., 0.9}.
#   Part 3: Q sweep      Q in {5, 10, ..., 50}; p_ar = M*Q; p_white = 1000-p_ar.
#   Part 4: M sweep      M in {1, 3, 5, 10, 15, 20, 30}; p_ar, p_white, s = M vary.
#
# Mirrors:
# cpp/trex_selector_methods/ad/trex_da/demo_trex_da_02_mc_sim_ar1_blocks_plus_white
#
# Same DGP, control parameters, sweep axes, and result files.
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
N = 300
P_TOTAL = 1000
M = 5              # number of AR(1) blocks
Q = 5              # block size; base p_ar = M * Q = 25
P_AR_BASE = M * Q  # 25
AMPLITUDE = 1.0
RHO = 0.7
SNR = 2.0
TFDR = 0.1
K = 20
NUM_MC = 200
BASE_SEED = 2026

LINKAGES = ("Single", "Complete", "Average")


def _support_one_per_block(p_ar, m):
    """OnePerBlock support on the AR part: block size Q, one active per block."""
    return make_support_one_per_block(m, p_ar, p_ar // m)


# ==============================================================================
# Part 1: MC SNR sweep
# Fixed: n=300, p_total=1000, M=5, Q=5 (p_ar=25, p_white=975, s=5), rho=0.7.
# ==============================================================================

def part_1_snr_sweep():
    snr_grid = [0.1, 0.2, 0.5, 0.6, 1.0, 2.0, 5.0]
    p_white_base = P_TOTAL - P_AR_BASE
    support_base = _support_one_per_block(P_AR_BASE, M)

    for lnk in LINKAGES:
        run_snr_sweep(
            OUT_DIR, f"da_ar1_blocks_plus_white_snr_{lnk}", snr_grid, NUM_MC,
            TFDR, BASE_SEED, default_solvers("+BT"),
            lambda snr: dict(n=N, p=P_TOTAL, dgp="dgp_ar1_block_white_snr",
                             kwargs=dict(amplitude=AMPLITUDE, p_ar=P_AR_BASE,
                                         p_white=p_white_base, n_blocks=M,
                                         rho=RHO, snr=snr),
                             support=support_base, s=M),
            {"method": "BT", "hc_linkage": lnk}, K,
            f"AR(1)+white SNR sweep  n={N}  p_total={P_TOTAL}  M={M}  Q={Q}"
            f"  p_ar={P_AR_BASE}  s={M}  rho={fmt_num(RHO)}  linkage={lnk}",
            include_base_trex=False)


# ==============================================================================
# Part 2: MC rho sweep
# Fixed: n=300, p_total=1000, M=5, Q=5, SNR=2.
# ==============================================================================

def part_2_rho_sweep():
    rho_grid = [round(0.1 * i, 1) for i in range(10)]
    p_white_base = P_TOTAL - P_AR_BASE
    support_base = _support_one_per_block(P_AR_BASE, M)

    for lnk in LINKAGES:
        run_param_sweep(
            OUT_DIR, f"da_ar1_blocks_plus_white_rho_{lnk}", "Rho", rho_grid,
            NUM_MC, TFDR, BASE_SEED, default_solvers("+BT"),
            lambda rho: dict(n=N, p=P_TOTAL, dgp="dgp_ar1_block_white_snr",
                             kwargs=dict(amplitude=AMPLITUDE, p_ar=P_AR_BASE,
                                         p_white=p_white_base, n_blocks=M,
                                         rho=rho, snr=SNR),
                             support=support_base, s=M),
            {"method": "BT", "hc_linkage": lnk}, K,
            f"AR(1)+white rho sweep  n={N}  p_total={P_TOTAL}  M={M}  Q={Q}"
            f"  s={M}  SNR={fmt_num(SNR)}  linkage={lnk}",
            include_base_trex=False)


# ==============================================================================
# Part 3: MC Q sweep (block size)
# Fixed: n=300, p_total=1000, M=5, rho=0.7, SNR=2.  s = M = 5 throughout.
# Swept: Q in {5, 10, ..., 50}; p_ar = M*Q varies; p_white = 1000-p_ar.
# Support recomputed per grid point (OnePerBlock on p_ar).
# ==============================================================================

def part_3_q_sweep():
    q_grid = list(range(5, 51, 5))

    def make_dgp(q):
        q_int = int(q)
        p_ar = M * q_int
        p_white = P_TOTAL - p_ar
        return dict(n=N, p=P_TOTAL, dgp="dgp_ar1_block_white_snr",
                    kwargs=dict(amplitude=AMPLITUDE, p_ar=p_ar,
                                p_white=p_white, n_blocks=M, rho=RHO, snr=SNR),
                    support=_support_one_per_block(p_ar, M), s=M)

    for lnk in LINKAGES:
        run_param_sweep(
            OUT_DIR, f"da_ar1_blocks_plus_white_Q_{lnk}", "Q", q_grid, NUM_MC,
            TFDR, BASE_SEED, default_solvers("+BT"), make_dgp,
            {"method": "BT", "hc_linkage": lnk}, K,
            f"AR(1)+white Q sweep  n={N}  p_total={P_TOTAL}  M={M}  s={M}"
            f"  rho={fmt_num(RHO)}  SNR={fmt_num(SNR)}  p_ar=M*Q varies"
            f"  linkage={lnk}",
            include_base_trex=False)


# ==============================================================================
# Part 4: MC M sweep (number of blocks)
# Fixed: n=300, p_total=1000, Q=5, rho=0.7, SNR=2.
# Swept: M in {1, 3, 5, 10, 15, 20, 30}; p_ar = M*Q, p_white = 1000-p_ar,
# s = M all vary. Support recomputed per grid point.
# ==============================================================================

def part_4_m_sweep():
    m_grid = [1, 3, 5, 10, 15, 20, 30]

    def make_dgp(m):
        m_int = int(m)
        p_ar = m_int * Q
        p_white = P_TOTAL - p_ar
        return dict(n=N, p=P_TOTAL, dgp="dgp_ar1_block_white_snr",
                    kwargs=dict(amplitude=AMPLITUDE, p_ar=p_ar,
                                p_white=p_white, n_blocks=m_int, rho=RHO,
                                snr=SNR),
                    support=_support_one_per_block(p_ar, m_int), s=m_int)

    for lnk in LINKAGES:
        run_param_sweep(
            OUT_DIR, f"da_ar1_blocks_plus_white_M_{lnk}", "M", m_grid, NUM_MC,
            TFDR, BASE_SEED, default_solvers("+BT"), make_dgp,
            {"method": "BT", "hc_linkage": lnk}, K,
            f"AR(1)+white M sweep  n={N}  p_total={P_TOTAL}  Q={Q}"
            f"  rho={fmt_num(RHO)}  SNR={fmt_num(SNR)}  p_ar=M*Q and s=M vary"
            f"  linkage={lnk}",
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
    print("\nAR(1)+white-noise block BT MC simulation complete.")
