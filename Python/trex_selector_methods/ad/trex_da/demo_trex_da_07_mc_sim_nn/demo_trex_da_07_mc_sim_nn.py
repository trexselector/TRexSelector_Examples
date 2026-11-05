# ==============================================================================
# demo_trex_da_07_mc_sim_nn.py
# ==============================================================================
#
# DA-TRex+NN Monte Carlo simulations for the nearest-neighbours (NN) DA method:
# banded MA(kappa) data, plus an AR(1) method-mismatch study.
#
# Parts 1-2 use the NN DGP, where all p predictors share a banded MA(kappa)
# covariance structure (the correctly-specified case for DAMethod.NN). Part 3
# applies the same NN correction to AR(1) data — a deliberate misspecification
# stress test (formerly its own demo, 03b, in the legacy Python numbering).
#
#   Part 1: MC SNR sweep on NN data (n=300, p=1000, kappa=3, rho=0.7;
#           CappedSpread and Random support; 3 solvers + base T-Rex comparison).
#   Part 2: 2D kappa x rho sweep on NN data (SNR=2.0; all 3 solvers;
#           CappedSpread and Random sub-sections).
#   Part 3: 2D SNR x rho sweep on AR(1) data with DA+NN (all 3 solvers;
#           CappedSpread and Random sub-sections).
#
# Mirrors:
#  cpp/trex_selector_methods/ad/trex_da/demo_trex_da_07_mc_sim_nn
#
# (same DGPs, control parameters, sweep axes, and result files).
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

from da_sim_common import (default_solvers, fmt_num, make_support_capped_spread,  # noqa: E402
                           print_section_header, run_mc_trials, run_snr_sweep,
                           save_and_print_2d_two_support_results)

RUN_PART_1 = True
RUN_PART_2 = True
RUN_PART_3 = True

OUT_DIR = os.path.join(_THIS_DIR, "simulation_results", "data")

# Shared MC parameters (mirror cpp)
N, P, S = 300, 1000, 10
KAPPA = 3
RHO = 0.7
SNR = 2.0
AMPLITUDE = 3.0
TFDR = 0.2
K = 20
NUM_MC = 200
BASE_SEED = 2026
MAX_GAP = 20

DA_SPEC = {"method": "NN"}


# ==============================================================================
# Part 1: MC SNR sweep — NN DGP
# CappedSpread(max_gap=20) and Random support; base T-Rex comparison rows.
# ==============================================================================

def part_1_snr_sweep():
    snr_grid = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
    support_fixed = make_support_capped_spread(S, P, MAX_GAP)
    hdr_base = f"NN kappa={KAPPA}, rho={fmt_num(RHO)}, n={N}, p={P}"

    # A — CappedSpread(max_gap=20)
    run_snr_sweep(
        OUT_DIR, "da_nn_snr_capped", snr_grid, NUM_MC, TFDR, BASE_SEED,
        default_solvers("+NN"),
        lambda snr: dict(n=N, p=P, dgp="dgp_nn_snr",
                         kwargs=dict(amplitude=AMPLITUDE, kappa=KAPPA, rho=RHO,
                                     snr=snr),
                         support=support_fixed, s=S),
        DA_SPEC, K,
        hdr_base + f", support=CappedSpread(max_gap={MAX_GAP})",
        include_base_trex=True)

    # B — Random (redrawn each trial)
    run_snr_sweep(
        OUT_DIR, "da_nn_snr_random", snr_grid, NUM_MC, TFDR, BASE_SEED,
        default_solvers("+NN"),
        lambda snr: dict(n=N, p=P, dgp="dgp_nn_snr",
                         kwargs=dict(amplitude=AMPLITUDE, kappa=KAPPA, rho=RHO,
                                     snr=snr),
                         support=None, s=S),
        DA_SPEC, K,
        hdr_base + ", support=Random (per-trial)",
        include_base_trex=True)


# ==============================================================================
# Generic 2D two-support sweep driver (Parts 2 and 3)
#
# Each solver runs with its own exch_tie_alpha, as in the cpp loops. The seed
# offset stays constant across cells (mirrors cpp base_offset = base_seed).
# ==============================================================================

def _run_2d_two_support(scenario_tag, row_label, col_label, row_grid, col_grid,
                        make_cell, header_extra, label_fns):
    solvers = default_solvers("+NN")
    names = [sv["name"] for sv in solvers]
    n_row, n_col = len(row_grid), len(col_grid)

    tpp_cs = {nm: np.zeros((n_row, n_col)) for nm in names}
    fdp_cs = {nm: np.zeros((n_row, n_col)) for nm in names}
    sd_tpp_cs = {nm: np.zeros((n_row, n_col)) for nm in names}
    sd_fdp_cs = {nm: np.zeros((n_row, n_col)) for nm in names}
    tpp_rand = {nm: np.zeros((n_row, n_col)) for nm in names}
    fdp_rand = {nm: np.zeros((n_row, n_col)) for nm in names}
    sd_tpp_rand = {nm: np.zeros((n_row, n_col)) for nm in names}
    sd_fdp_rand = {nm: np.zeros((n_row, n_col)) for nm in names}

    support_cs = make_support_capped_spread(S, P, MAX_GAP)
    total_cells = n_row * n_col

    for sv in solvers:
        nm = sv["name"]
        print(f"\n  Solver: {nm}\n")

        print(f"  [A] CappedSpread(max_gap={MAX_GAP}) ...\n")
        cell_idx = 0
        for ir, rv in enumerate(row_grid):
            for ic, cv in enumerate(col_grid):
                cell_idx += 1
                dgp, kwargs = make_cell(rv, cv)
                lbl = f"[{cell_idx}/{total_cells}] {nm}  {label_fns(rv, cv)}"
                res = run_mc_trials(NUM_MC, lbl, N, P, dgp, kwargs, support_cs,
                                    S, TFDR, sv, K, DA_SPEC, BASE_SEED)
                tpp_cs[nm][ir, ic] = res["avg_tpr"]
                fdp_cs[nm][ir, ic] = res["avg_fdr"]
                sd_tpp_cs[nm][ir, ic] = res["sd_tpr"]
                sd_fdp_cs[nm][ir, ic] = res["sd_fdr"]

        print("\n  [B] Random support ...\n")
        cell_idx = 0
        for ir, rv in enumerate(row_grid):
            for ic, cv in enumerate(col_grid):
                cell_idx += 1
                dgp, kwargs = make_cell(rv, cv)
                lbl = (f"[{cell_idx}/{total_cells}] {nm}  {label_fns(rv, cv)}"
                       "  Random")
                res = run_mc_trials(NUM_MC, lbl, N, P, dgp, kwargs, None,
                                    S, TFDR, sv, K, DA_SPEC, BASE_SEED)
                tpp_rand[nm][ir, ic] = res["avg_tpr"]
                fdp_rand[nm][ir, ic] = res["avg_fdr"]
                sd_tpp_rand[nm][ir, ic] = res["sd_tpr"]
                sd_fdp_rand[nm][ir, ic] = res["sd_fdr"]

    save_and_print_2d_two_support_results(
        OUT_DIR, scenario_tag, row_label, col_label, row_grid, col_grid,
        NUM_MC, names, tpp_cs, fdp_cs, sd_tpp_cs, sd_fdp_cs,
        tpp_rand, fdp_rand, sd_tpp_rand, sd_fdp_rand, header_extra)


# ==============================================================================
# Part 2: 2D kappa x rho sweep — NN DGP
# Fixed: SNR=2.0, amplitude=3.0.
# ==============================================================================

def part_2_kappa_rho_sweep():
    kappa_grid = [1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0]
    rho_grid = [0.1, 0.3, 0.5, 0.7, 0.8, 0.9]

    print_section_header("NN Part 2: 2D Kappa x Rho sweep")
    print(f"  n={N}  p={P}  s={S}  SNR={fmt_num(SNR)}  MC={NUM_MC}"
          f"  tFDR={TFDR}  amplitude={AMPLITUDE}")
    print("  kappa_grid: {" + ", ".join(str(int(v)) for v in kappa_grid) + "}")
    print("  rho_grid:   {" + ", ".join(f"{r:.1f}" for r in rho_grid) + "}\n")

    _run_2d_two_support(
        "da_nn_kappa_rho", "Kappa", "Rho", kappa_grid, rho_grid,
        lambda kappa, rho: ("dgp_nn_snr",
                            dict(amplitude=AMPLITUDE, kappa=int(kappa),
                                 rho=rho, snr=SNR)),
        f"n={N}, p={P}, s={S}, SNR={fmt_num(SNR)}, tFDR={fmt_num(TFDR)}"
        ", amplitude=3.0",
        lambda kappa, rho: f"kappa={int(kappa)}  rho={rho:.1f}")


# ==============================================================================
# Part 3: 2D SNR x rho sweep — AR(1) data with DA+NN (method-mismatch study)
# Uses the AR(1) DGP but keeps DAMethod.NN, exploring whether the
# nearest-neighbour aggregation can track the AR(1) correlation structure.
# Fixed: amplitude=3.0.
# ==============================================================================

def part_3_nn_ar_snr_rho_sweep():
    snr_grid = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
    rho_grid = [round(0.1 * i, 1) for i in range(1, 10)]

    print_section_header("AR(1)+NN Part 3: 2D SNR x Rho sweep")
    print(f"  n={N}  p={P}  s={S}  MC={NUM_MC}"
          f"  tFDR={TFDR}  amplitude={AMPLITUDE}")
    print("  snr_grid: {" + ", ".join(f"{v:.2f}" for v in snr_grid) + "}")
    print("  rho_grid: {" + ", ".join(f"{r:.1f}" for r in rho_grid) + "}\n")

    _run_2d_two_support(
        "da_nn_ar_snr_rho", "SNR", "Rho", snr_grid, rho_grid,
        lambda snr, rho: ("dgp_ar1_snr",
                          dict(amplitude=AMPLITUDE, rho=rho, snr=snr)),
        f"n={N}, p={P}, s={S}, tFDR={fmt_num(TFDR)}, amplitude=3.0",
        lambda snr, rho: f"SNR={snr:.2f}  rho={rho:.1f}")


# ==============================================================================
# Main
# ==============================================================================

if __name__ == "__main__":
    if RUN_PART_1:
        part_1_snr_sweep()
    if RUN_PART_2:
        part_2_kappa_rho_sweep()
    if RUN_PART_3:
        part_3_nn_ar_snr_rho_sweep()
    print("\nNN MC simulation complete.")
