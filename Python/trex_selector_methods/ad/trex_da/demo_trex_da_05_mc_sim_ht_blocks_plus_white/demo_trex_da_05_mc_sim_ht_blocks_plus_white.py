# ==============================================================================
# demo_trex_da_05_mc_sim_ht_blocks_plus_white.py
# ==============================================================================
#
# DA-TRex+BT MC simulations for the heavy-tailed block + white-noise DGP.
#
# DGP: dgp_ht_block_white_snr — M AR(1)-Toeplitz t(nu) blocks (p_ar columns)
# plus p_white i.i.d. t(nu) white-noise columns; p_total = p_ar + p_white = 500.
# Active variables (s = M = 5, one per AR block) lie in the heavy-tailed AR part.
#
# Outer loops: h_noise in {False ("s1_Gauss"), True ("s2_Heavy")} and linkage
# in {Single, Complete, Average}. All 3 solvers (TLARS, TAFS, TOMP) run inside
# each linkage iteration.
#
#   Part 1: SNR sweep      SNR in {0.1, ..., 5.0}.
#   Part 2: rho sweep      rho in {0.0, ..., 0.9}.
#   Part 3: Q sweep        Q in {5, ..., 50}; p_ar = M*Q; p_white = 500-p_ar.
#   Part 4: M sweep        M in {1, ..., 30}; p_ar, p_white, s = M vary.
#   Part 5: tFDR sweep     tFDR in {0.05, ..., 0.50}.
#
# Mirrors:
# cpp/trex_selector_methods/ad/trex_da/demo_trex_da_05_mc_sim_ht_blocks_plus_white
# (same DGP, control parameters, sweep axes, and result files).
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

import numpy as np  # noqa: E402

from da_sim_common import (default_solvers, fmt_num,  # noqa: E402
                           make_support_one_per_block, run_mc_trials,
                           run_param_sweep, run_snr_sweep,
                           save_and_print_grid_results)

RUN_PART_1 = True
RUN_PART_2 = True
RUN_PART_3 = True
RUN_PART_4 = True
RUN_PART_5 = True

OUT_DIR = os.path.join(_THIS_DIR, "simulation_results", "data")

# Shared base parameters (mirror cpp)
N = 150
P_TOTAL = 500
M = 5
Q = 5
AMPLITUDE = 1.0
RHO = 0.8
NU = 3.0
SNR = 2.0
TFDR = 0.2
K = 20
NUM_MC = 200
BASE_SEED = 2026
P_AR_BASE = M * Q                  # 25
P_WHITE_BASE = P_TOTAL - P_AR_BASE  # 475

SCENARIOS = (("s1_Gauss", False), ("s2_Heavy", True))
LINKAGES = ("Single", "Complete", "Average")

SUPPORT_BASE = make_support_one_per_block(M, P_AR_BASE, Q)


def _kwargs(p_ar, p_white, n_blocks, rho, snr, h_noise):
    return dict(amplitude=AMPLITUDE, p_ar=p_ar, p_white=p_white,
                n_blocks=n_blocks, rho=rho, snr=snr, nu=NU,
                heavy_noise=h_noise)


# ==============================================================================
# Part 1: MC SNR sweep
# Fixed: n=150, p_total=500, M=5, Q=5 (p_ar=25, p_white=475, s=5), rho=0.8,
# nu=3.0.
# ==============================================================================

def part_1_snr_sweep():
    snr_grid = [0.1, 0.2, 0.5, 0.6, 1.0, 2.0, 5.0]

    for scen, h_noise in SCENARIOS:
        for lnk in LINKAGES:
            run_snr_sweep(
                OUT_DIR, f"da_ht_blocks_plus_white_snr_{scen}_{lnk}", snr_grid,
                NUM_MC, TFDR, BASE_SEED, default_solvers("+BT"),
                lambda snr: dict(n=N, p=P_TOTAL, dgp="dgp_ht_block_white_snr",
                                 kwargs=_kwargs(P_AR_BASE, P_WHITE_BASE, M,
                                                RHO, snr, h_noise),
                                 support=SUPPORT_BASE, s=M),
                {"method": "BT", "hc_linkage": lnk}, K,
                f"HT+white SNR sweep  n={N}  p_total={P_TOTAL}  M={M}  Q={Q}"
                f"  p_ar={P_AR_BASE}  s={M}  rho={fmt_num(RHO)}"
                f"  nu={fmt_num(NU)}  noise={scen}  linkage={lnk}",
                include_base_trex=False)


# ==============================================================================
# Part 2: MC rho sweep
# Fixed: n=150, p_total=500, M=5, Q=5 (p_ar=25, p_white=475, s=5), SNR=2,
# nu=3.0.
# ==============================================================================

def part_2_rho_sweep():
    rho_grid = [round(0.1 * i, 1) for i in range(10)]

    for scen, h_noise in SCENARIOS:
        for lnk in LINKAGES:
            run_param_sweep(
                OUT_DIR, f"da_ht_blocks_plus_white_rho_{scen}_{lnk}", "Rho",
                rho_grid, NUM_MC, TFDR, BASE_SEED, default_solvers("+BT"),
                lambda rho: dict(n=N, p=P_TOTAL, dgp="dgp_ht_block_white_snr",
                                 kwargs=_kwargs(P_AR_BASE, P_WHITE_BASE, M,
                                                rho, SNR, h_noise),
                                 support=SUPPORT_BASE, s=M),
                {"method": "BT", "hc_linkage": lnk}, K,
                f"HT+white rho sweep  n={N}  p_total={P_TOTAL}  M={M}  Q={Q}"
                f"  s={M}  SNR={fmt_num(SNR)}  nu={fmt_num(NU)}  noise={scen}"
                f"  linkage={lnk}",
                include_base_trex=False)


# ==============================================================================
# Part 3: MC Q sweep (block size)
# Fixed: n=150, p_total=500, M=5, rho=0.8, SNR=2, nu=3.0.  s = M = 5.
# Swept: Q in {5, 10, ..., 50}; p_ar = M*Q varies; p_white = 500-p_ar.
# Support recomputed per grid point (OnePerBlock on p_ar).
# ==============================================================================

def part_3_q_sweep():
    q_grid = list(range(5, 51, 5))

    def make_dgp(q):
        q_int = int(q)
        p_ar = M * q_int
        p_white = P_TOTAL - p_ar
        return dict(n=N, p=P_TOTAL, dgp="dgp_ht_block_white_snr",
                    kwargs=_kwargs(p_ar, p_white, M, RHO, SNR,
                                   make_dgp.h_noise),
                    support=make_support_one_per_block(M, p_ar, q_int), s=M)

    for scen, h_noise in SCENARIOS:
        make_dgp.h_noise = h_noise
        for lnk in LINKAGES:
            run_param_sweep(
                OUT_DIR, f"da_ht_blocks_plus_white_Q_{scen}_{lnk}", "Q",
                q_grid, NUM_MC, TFDR, BASE_SEED, default_solvers("+BT"),
                make_dgp, {"method": "BT", "hc_linkage": lnk}, K,
                f"HT+white Q sweep  n={N}  p_total={P_TOTAL}  M={M}  s={M}"
                f"  rho={fmt_num(RHO)}  SNR={fmt_num(SNR)}  nu={fmt_num(NU)}"
                f"  p_ar=M*Q varies  noise={scen}  linkage={lnk}",
                include_base_trex=False)


# ==============================================================================
# Part 4: MC M sweep (number of blocks)
# Fixed: n=150, p_total=500, Q=5, rho=0.8, SNR=2, nu=3.0.
# Swept: M in {1, 3, 5, 10, 15, 20, 30}; p_ar = M*Q, p_white = 500-p_ar,
# s = M all vary. Support recomputed per grid point.
# ==============================================================================

def part_4_m_sweep():
    m_grid = [1, 3, 5, 10, 15, 20, 30]

    def make_dgp(m):
        m_int = int(m)
        p_ar = m_int * Q
        p_white = P_TOTAL - p_ar
        return dict(n=N, p=P_TOTAL, dgp="dgp_ht_block_white_snr",
                    kwargs=_kwargs(p_ar, p_white, m_int, RHO, SNR,
                                   make_dgp.h_noise),
                    support=make_support_one_per_block(m_int, p_ar, Q),
                    s=m_int)

    for scen, h_noise in SCENARIOS:
        make_dgp.h_noise = h_noise
        for lnk in LINKAGES:
            run_param_sweep(
                OUT_DIR, f"da_ht_blocks_plus_white_M_{scen}_{lnk}", "M",
                m_grid, NUM_MC, TFDR, BASE_SEED, default_solvers("+BT"),
                make_dgp, {"method": "BT", "hc_linkage": lnk}, K,
                f"HT+white M sweep  n={N}  p_total={P_TOTAL}  Q={Q}"
                f"  rho={fmt_num(RHO)}  SNR={fmt_num(SNR)}  nu={fmt_num(NU)}"
                f"  p_ar=M*Q and s=M vary  noise={scen}  linkage={lnk}",
                include_base_trex=False)


# ==============================================================================
# Part 5: MC tFDR sweep
# Fixed: n=150, p_total=500, M=5, Q=5 (p_ar=25, p_white=475, s=5), rho=0.8,
# SNR=2, nu=3.0.
# Swept: tFDR in {0.05, 0.10, ..., 0.50}.
# (tFDR is a selector parameter — run_mc_trials called per solver x grid point,
#  each solver with its own exch_tie_alpha as in the cpp section.)
# ==============================================================================

def part_5_tfdr_sweep():
    tfdr_grid = [round(0.05 * i, 2) for i in range(1, 11)]

    for scen, h_noise in SCENARIOS:
        for lnk in LINKAGES:
            da_spec = {"method": "BT", "hc_linkage": lnk}
            solvers = default_solvers("+BT")
            names = [sv["name"] for sv in solvers]
            fdr, tpr, sd_fdr, sd_tpr, avg_L, avg_T = (
                {nm: np.zeros(len(tfdr_grid)) for nm in names}
                for _ in range(6))

            for sv in solvers:
                for gi, tfdr in enumerate(tfdr_grid):
                    res = run_mc_trials(
                        NUM_MC, f"tFDR = {fmt_num(tfdr)}", N, P_TOTAL,
                        "dgp_ht_block_white_snr",
                        _kwargs(P_AR_BASE, P_WHITE_BASE, M, RHO, SNR, h_noise),
                        SUPPORT_BASE, M, tfdr, sv, K, da_spec,
                        BASE_SEED + gi * 10000)
                    nm = sv["name"]
                    fdr[nm][gi] = res["avg_fdr"]
                    tpr[nm][gi] = res["avg_tpr"]
                    sd_fdr[nm][gi] = res["sd_fdr"]
                    sd_tpr[nm][gi] = res["sd_tpr"]
                    avg_L[nm][gi] = res["avg_L"]
                    avg_T[nm][gi] = res["avg_T"]

            save_and_print_grid_results(
                OUT_DIR, f"da_ht_blocks_plus_white_tFDR_{scen}_{lnk}", "tFDR",
                tfdr_grid, NUM_MC, names, fdr, tpr, sd_fdr, sd_tpr, avg_L,
                avg_T,
                f"HT+white tFDR sweep  n={N}  p_total={P_TOTAL}  M={M}  Q={Q}"
                f"  p_ar={P_AR_BASE}  s={M}  rho={fmt_num(RHO)}"
                f"  SNR={fmt_num(SNR)}  nu={fmt_num(NU)}  noise={scen}"
                f"  linkage={lnk}")


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
    if RUN_PART_5:
        part_5_tfdr_sweep()
    print("\nHeavy-tailed+white-noise block BT MC simulations complete.")
