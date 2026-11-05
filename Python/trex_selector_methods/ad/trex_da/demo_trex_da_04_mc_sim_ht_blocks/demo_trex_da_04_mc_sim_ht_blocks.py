# ==============================================================================
# demo_trex_da_04_mc_sim_ht_blocks.py
# ==============================================================================
#
# DA-TRex+BT MC simulations for the heavy-tailed block DGP (BT aggregation).
#
# DGP: dgp_block_toeplitz_hvt_snr — block-diagonal Toeplitz covariance;
# predictors are multivariate t(nu=3) in both scenarios, and the response noise
# is Gaussian ("s1_Gauss") or t(nu=3) ("s2_Heavy"). M blocks of size Q each,
# one active per block (s = M).
#
# Outer loops: h_noise in {False ("s1_Gauss"), True ("s2_Heavy")} and linkage
# in {Single, Complete, Average}. All 3 solvers (TLARS, TAFS, TOMP) run inside
# each linkage iteration.
#
#   Part 1: SNR sweep      SNR in {0.1, ..., 5.0}.
#   Part 2: rho sweep      rho in {0.0, ..., 0.9}.
#   Part 3: Q sweep        Q in {5, ..., 50}.
#   Part 4: M sweep        M in {1, ..., 30}.
#   Part 5: tFDR sweep     tFDR in {0.05, ..., 0.50}.
#   Part 6: linkage sweep  linkage codes {1=Single, 2=Complete, 3=Average}.
#
# Mirrors:
# cpp/trex_selector_methods/ad/trex_da/demo_trex_da_04_mc_sim_ht_blocks
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
RUN_PART_6 = True

OUT_DIR = os.path.join(_THIS_DIR, "simulation_results", "data")

# Shared base parameters (mirror cpp)
N = 150
M = 5
Q = 5
BASE_P = M * Q     # 25
AMPLITUDE = 1.0
RHO = 0.8
NU = 3.0
SNR = 2.0
TFDR = 0.2
K = 20
NUM_MC = 200
BASE_SEED = 2026

SCENARIOS = (("s1_Gauss", False), ("s2_Heavy", True))
LINKAGES = ("Single", "Complete", "Average")

SUPPORT_BASE = make_support_one_per_block(M, BASE_P, Q)


def _kwargs(n_blocks, q, rho, snr, h_noise):
    return dict(amplitude=AMPLITUDE, n_blocks=n_blocks, rho=rho, snr=snr,
                nu=NU, heavy_X=True, heavy_noise=h_noise)


# ==============================================================================
# Part 1: MC SNR sweep
# Fixed: n=150, M=5, Q=5 (p=25, s=5), rho=0.8, nu=3.0.
# ==============================================================================

def part_1_snr_sweep():
    snr_grid = [0.1, 0.2, 0.5, 0.6, 1.0, 2.0, 5.0]

    for scen, h_noise in SCENARIOS:
        for lnk in LINKAGES:
            run_snr_sweep(
                OUT_DIR, f"da_ht_blocks_snr_{scen}_{lnk}", snr_grid, NUM_MC,
                TFDR, BASE_SEED, default_solvers("+BT"),
                lambda snr: dict(n=N, p=BASE_P,
                                 dgp="dgp_block_toeplitz_hvt_snr",
                                 kwargs=_kwargs(M, Q, RHO, snr, h_noise),
                                 support=SUPPORT_BASE, s=M),
                {"method": "BT", "hc_linkage": lnk}, K,
                f"HT-block SNR sweep  n={N}  M={M}  Q={Q}  p={BASE_P}  s={M}"
                f"  rho={fmt_num(RHO)}  nu={fmt_num(NU)}  noise={scen}"
                f"  linkage={lnk}",
                include_base_trex=False)


# ==============================================================================
# Part 2: MC rho sweep
# Fixed: n=150, M=5, Q=5 (p=25, s=5), SNR=2, nu=3.0.
# ==============================================================================

def part_2_rho_sweep():
    rho_grid = [round(0.1 * i, 1) for i in range(10)]

    for scen, h_noise in SCENARIOS:
        for lnk in LINKAGES:
            run_param_sweep(
                OUT_DIR, f"da_ht_blocks_rho_{scen}_{lnk}", "Rho", rho_grid,
                NUM_MC, TFDR, BASE_SEED, default_solvers("+BT"),
                lambda rho: dict(n=N, p=BASE_P,
                                 dgp="dgp_block_toeplitz_hvt_snr",
                                 kwargs=_kwargs(M, Q, rho, SNR, h_noise),
                                 support=SUPPORT_BASE, s=M),
                {"method": "BT", "hc_linkage": lnk}, K,
                f"HT-block rho sweep  n={N}  M={M}  Q={Q}  s={M}"
                f"  SNR={fmt_num(SNR)}  nu={fmt_num(NU)}  noise={scen}"
                f"  linkage={lnk}",
                include_base_trex=False)


# ==============================================================================
# Part 3: MC Q sweep (block size)
# Fixed: n=150, M=5, rho=0.8, SNR=2, nu=3.0.  s = M = 5 throughout.
# Swept: Q in {5, 10, ..., 50}; p = M*Q varies.
# ==============================================================================

def part_3_q_sweep():
    q_grid = list(range(5, 51, 5))

    def make_dgp(q):
        q_int = int(q)
        p_val = M * q_int
        return dict(n=N, p=p_val, dgp="dgp_block_toeplitz_hvt_snr",
                    kwargs=_kwargs(M, q_int, RHO, SNR, make_dgp.h_noise),
                    support=make_support_one_per_block(M, p_val, q_int), s=M)

    for scen, h_noise in SCENARIOS:
        make_dgp.h_noise = h_noise
        for lnk in LINKAGES:
            run_param_sweep(
                OUT_DIR, f"da_ht_blocks_Q_{scen}_{lnk}", "Q", q_grid, NUM_MC,
                TFDR, BASE_SEED, default_solvers("+BT"), make_dgp,
                {"method": "BT", "hc_linkage": lnk}, K,
                f"HT-block Q sweep  n={N}  M={M}  s={M}  rho={fmt_num(RHO)}"
                f"  SNR={fmt_num(SNR)}  nu={fmt_num(NU)}  p=M*Q varies"
                f"  noise={scen}  linkage={lnk}",
                include_base_trex=False)


# ==============================================================================
# Part 4: MC M sweep (number of blocks)
# Fixed: n=150, Q=5, rho=0.8, SNR=2, nu=3.0.
# Swept: M in {1, 3, 5, 10, 15, 20, 30}; p = M*Q and s = M both vary.
# ==============================================================================

def part_4_m_sweep():
    m_grid = [1, 3, 5, 10, 15, 20, 30]

    def make_dgp(m):
        m_int = int(m)
        p_val = m_int * Q
        return dict(n=N, p=p_val, dgp="dgp_block_toeplitz_hvt_snr",
                    kwargs=_kwargs(m_int, Q, RHO, SNR, make_dgp.h_noise),
                    support=make_support_one_per_block(m_int, p_val, Q),
                    s=m_int)

    for scen, h_noise in SCENARIOS:
        make_dgp.h_noise = h_noise
        for lnk in LINKAGES:
            run_param_sweep(
                OUT_DIR, f"da_ht_blocks_M_{scen}_{lnk}", "M", m_grid, NUM_MC,
                TFDR, BASE_SEED, default_solvers("+BT"), make_dgp,
                {"method": "BT", "hc_linkage": lnk}, K,
                f"HT-block M sweep  n={N}  Q={Q}  rho={fmt_num(RHO)}"
                f"  SNR={fmt_num(SNR)}  nu={fmt_num(NU)}  p=M*Q and s=M vary"
                f"  noise={scen}  linkage={lnk}",
                include_base_trex=False)


# ==============================================================================
# Part 5: MC tFDR sweep
# Fixed: n=150, M=5, Q=5 (p=25, s=5), rho=0.8, SNR=2, nu=3.0.
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
                        NUM_MC, f"tFDR = {fmt_num(tfdr)}", N, BASE_P,
                        "dgp_block_toeplitz_hvt_snr",
                        _kwargs(M, Q, RHO, SNR, h_noise),
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
                OUT_DIR, f"da_ht_blocks_tFDR_{scen}_{lnk}", "tFDR", tfdr_grid,
                NUM_MC, names, fdr, tpr, sd_fdr, sd_tpr, avg_L, avg_T,
                f"HT-block tFDR sweep  n={N}  M={M}  Q={Q}  p={BASE_P}  s={M}"
                f"  rho={fmt_num(RHO)}  SNR={fmt_num(SNR)}  nu={fmt_num(NU)}"
                f"  noise={scen}  linkage={lnk}")


# ==============================================================================
# Part 6: Linkage sweep
# Fixed: n=150, M=5, Q=5 (p=25, s=5), rho=0.8, SNR=2, nu=3.0, tFDR=0.2.
# Swept: linkage encoded as {1.0=Single, 2.0=Complete, 3.0=Average}.
# (Each solver runs with its own exch_tie_alpha, as in the cpp section.)
# ==============================================================================

def part_6_linkage_sweep():
    lnk_vals = [1.0, 2.0, 3.0]

    for scen, h_noise in SCENARIOS:
        solvers = default_solvers("+BT")
        names = [sv["name"] for sv in solvers]
        fdr, tpr, sd_fdr, sd_tpr, avg_L, avg_T = (
            {nm: np.zeros(len(lnk_vals)) for nm in names}
            for _ in range(6))

        for sv in solvers:
            for gi, lnk in enumerate(LINKAGES):
                res = run_mc_trials(
                    NUM_MC, f"Linkage = {lnk}", N, BASE_P,
                    "dgp_block_toeplitz_hvt_snr",
                    _kwargs(M, Q, RHO, SNR, h_noise),
                    SUPPORT_BASE, M, TFDR, sv, K,
                    {"method": "BT", "hc_linkage": lnk},
                    BASE_SEED + gi * 10000)
                nm = sv["name"]
                fdr[nm][gi] = res["avg_fdr"]
                tpr[nm][gi] = res["avg_tpr"]
                sd_fdr[nm][gi] = res["sd_fdr"]
                sd_tpr[nm][gi] = res["sd_tpr"]
                avg_L[nm][gi] = res["avg_L"]
                avg_T[nm][gi] = res["avg_T"]

        save_and_print_grid_results(
            OUT_DIR, f"da_ht_blocks_linkage_{scen}", "Linkage", lnk_vals,
            NUM_MC, names, fdr, tpr, sd_fdr, sd_tpr, avg_L, avg_T,
            f"HT-block Linkage sweep  n={N}  M={M}  Q={Q}  p={BASE_P}  s={M}"
            f"  rho={fmt_num(RHO)}  SNR={fmt_num(SNR)}  nu={fmt_num(NU)}"
            f"  tFDR={fmt_num(TFDR)}  noise={scen}"
            "\n  Linkage codes: 1=Single  2=Complete  3=Average")


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
    if RUN_PART_6:
        part_6_linkage_sweep()
    print("\nHeavy-tailed block BT MC simulations complete.")
