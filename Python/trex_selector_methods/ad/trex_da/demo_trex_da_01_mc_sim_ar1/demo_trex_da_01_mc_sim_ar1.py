# ==============================================================================
# demo_trex_da_01_mc_sim_ar1.py
# ==============================================================================
#
# DA-TRex+AR1 Monte Carlo simulations for the AR(1) Toeplitz DGP.
#
#   Part 1: MC SNR sweep (n=300, p=1000, s=10, rho=0.7, 200 MC; CappedSpread
#           and Random support; 3 DA solvers + base T-Rex comparison rows).
#   Part 2: MC rho sweep (SNR=2.0; same layout as Part 1).
#   Part 3: 2D Gap x Rho sweep — probes the kappa boundary where gap < kappa
#           collapses TPP (CappedSpread(gap) matrices + Random column).
#
# Mirrors:
# cpp/trex_selector_methods/ad/trex_da/demo_trex_da_01_mc_sim_ar1
#
# Results are saved to simulation_results/data/ as TXT and CSV files.
# Every part runs num_MC=200 trials per grid point.
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
                           print_section_header, run_mc_trials, run_param_sweep,
                           run_snr_sweep, save_and_print_2d_gap_rho_results)

RUN_PART_1 = True
RUN_PART_2 = True
RUN_PART_3 = True

OUT_DIR = os.path.join(_THIS_DIR, "simulation_results", "data")

# Simulation config (mirrors cpp SimConfig + per-part constants)
N, P, S = 300, 1000, 10
AMPLITUDE = 3.0
TFDR = 0.2
K = 20
NUM_MC = 200
BASE_SEED = 2026
MAX_GAP = 20

DA_SPEC = {"method": "AR1"}


# ==============================================================================
# Part 1: MC simulation — SNR sweep, 2 support types (CappedSpread + Random)
# ==============================================================================

def part_1_snr_sweep():
    rho = 0.7
    snr_grid = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
    hdr_base = f"AR(1) rho={fmt_num(rho)}, n={N}, p={P}"
    support_capped = make_support_capped_spread(S, P, MAX_GAP)

    # --- Run A: CappedSpread(max_gap=20) ---
    run_snr_sweep(
        OUT_DIR, "da_ar1_snr_capped", snr_grid, NUM_MC, TFDR, BASE_SEED,
        default_solvers("+AR1"),
        lambda snr: dict(n=N, p=P, dgp="dgp_ar1_snr",
                         kwargs=dict(amplitude=AMPLITUDE, rho=rho, snr=snr),
                         support=support_capped, s=S),
        DA_SPEC, K,
        hdr_base + f", support=CappedSpread(max_gap={MAX_GAP})",
        include_base_trex=True)

    # --- Run B: Random support (redrawn per trial) ---
    run_snr_sweep(
        OUT_DIR, "da_ar1_snr_random", snr_grid, NUM_MC, TFDR, BASE_SEED,
        default_solvers("+AR1"),
        lambda snr: dict(n=N, p=P, dgp="dgp_ar1_snr",
                         kwargs=dict(amplitude=AMPLITUDE, rho=rho, snr=snr),
                         support=None, s=S),
        DA_SPEC, K,
        hdr_base + ", support=Random (redrawn per trial)",
        include_base_trex=True)


# ==============================================================================
# Part 2: MC simulation — Rho sweep, 2 support types (CappedSpread + Random)
# ==============================================================================

def part_2_rho_sweep():
    snr_fixed = 2.0
    rho_grid = [round(0.1 * i, 1) for i in range(10)]   # 0.0 .. 0.9
    hdr_base = f"AR(1) SNR={fmt_num(snr_fixed)}, n={N}, p={P}"
    support_capped = make_support_capped_spread(S, P, MAX_GAP)

    # --- Run A: CappedSpread(max_gap=20) ---
    run_param_sweep(
        OUT_DIR, "da_ar1_rho_capped", "Rho", rho_grid, NUM_MC, TFDR, BASE_SEED,
        default_solvers("+AR1"),
        lambda rho: dict(n=N, p=P, dgp="dgp_ar1_snr",
                         kwargs=dict(amplitude=AMPLITUDE, rho=rho, snr=snr_fixed),
                         support=support_capped, s=S),
        DA_SPEC, K,
        hdr_base + f", support=CappedSpread(max_gap={MAX_GAP})",
        include_base_trex=True)

    # --- Run B: Random support (redrawn per trial) ---
    run_param_sweep(
        OUT_DIR, "da_ar1_rho_random", "Rho", rho_grid, NUM_MC, TFDR, BASE_SEED,
        default_solvers("+AR1"),
        lambda rho: dict(n=N, p=P, dgp="dgp_ar1_snr",
                         kwargs=dict(amplitude=AMPLITUDE, rho=rho, snr=snr_fixed),
                         support=None, s=S),
        DA_SPEC, K,
        hdr_base + ", support=Random (redrawn per trial)",
        include_base_trex=True)


# ==============================================================================
# Part 3: 2D Gap x Rho sweep
#
#  Runs num_MC trials per (solver, gap, rho) cell for two support types:
#    A) CappedSpread(gap) — deterministic, evenly-spaced support.
#    B) Random            — support redrawn each trial (not gap-dependent).
#
#  The kappa boundary (kappa = ceil(log(0.02) / log(rho))) is the DA+AR1
#  correction window half-width. When gap < kappa, active predictors fall
#  inside each other's correction windows and TPP collapses.
# ==============================================================================

def part_3_gap_rho_sweep():
    snr_fixed = 2.0
    gap_grid = [100, 50, 20, 15, 10, 5, 1]
    rho_grid = [round(0.1 * i, 1) for i in range(10)]

    kappa_boundary = [0 if rho <= 0.0
                      else int(np.ceil(np.log(0.02) / np.log(rho)))
                      for rho in rho_grid]

    solvers = default_solvers("+AR1")
    names = [sv["name"] for sv in solvers]
    n_rho, n_gap = len(rho_grid), len(gap_grid)

    tpp_cs = {nm: np.zeros((n_rho, n_gap)) for nm in names}
    fdp_cs = {nm: np.zeros((n_rho, n_gap)) for nm in names}
    sd_tpp_cs = {nm: np.zeros((n_rho, n_gap)) for nm in names}
    sd_fdp_cs = {nm: np.zeros((n_rho, n_gap)) for nm in names}
    tpp_rand = {nm: np.zeros(n_rho) for nm in names}
    fdp_rand = {nm: np.zeros(n_rho) for nm in names}
    sd_tpp_rand = {nm: np.zeros(n_rho) for nm in names}
    sd_fdp_rand = {nm: np.zeros(n_rho) for nm in names}

    print_section_header("AR(1) Part 3: Gap x Rho sweep")
    print(f"  n={N}  p={P}  s={S}  SNR={fmt_num(snr_fixed)}  MC={NUM_MC}"
          f"  tFDR={TFDR}  amplitude={fmt_num(AMPLITUDE)}")
    print("  gap_grid: {" + ", ".join(map(str, gap_grid)) + "}")
    print("  rho_grid: {" + ", ".join(f"{r:.1f}" for r in rho_grid) + "}\n")

    # Constant seed offset across cells (mirrors cpp base_offset = base_seed)
    base_offset = BASE_SEED

    for sv in solvers:
        nm = sv["name"]
        print(f"\n  Solver: {nm}\n")

        # ---- Section A: CappedSpread(gap) x rho ----
        print("  [A] CappedSpread(gap) sweep ...\n")
        cell_idx, total_cells = 0, n_rho * n_gap
        for ir, rho in enumerate(rho_grid):
            for ig, gap in enumerate(gap_grid):
                cell_idx += 1
                support = make_support_capped_spread(S, P, gap)
                lbl = (f"[{cell_idx}/{total_cells}] {nm}  rho={rho:.1f}"
                       f"  gap={gap:3d}  kappa={kappa_boundary[ir]}")
                res = run_mc_trials(
                    NUM_MC, lbl, N, P, "dgp_ar1_snr",
                    dict(amplitude=AMPLITUDE, rho=rho, snr=snr_fixed),
                    support, S, TFDR, sv, K, DA_SPEC, base_offset)
                tpp_cs[nm][ir, ig] = res["avg_tpr"]
                fdp_cs[nm][ir, ig] = res["avg_fdr"]
                sd_tpp_cs[nm][ir, ig] = res["sd_tpr"]
                sd_fdp_cs[nm][ir, ig] = res["sd_fdr"]

        # ---- Section B: Random support sweep (rho only) ----
        print("\n  [B] Random support sweep ...\n")
        for ir, rho in enumerate(rho_grid):
            lbl = (f"[{ir + 1}/{n_rho}] {nm}  rho={rho:.1f}"
                   f"  kappa={kappa_boundary[ir]}  Random")
            res = run_mc_trials(
                NUM_MC, lbl, N, P, "dgp_ar1_snr",
                dict(amplitude=AMPLITUDE, rho=rho, snr=snr_fixed),
                None, S, TFDR, sv, K, DA_SPEC, base_offset)
            tpp_rand[nm][ir] = res["avg_tpr"]
            fdp_rand[nm][ir] = res["avg_fdr"]
            sd_tpp_rand[nm][ir] = res["sd_tpr"]
            sd_fdp_rand[nm][ir] = res["sd_fdr"]

    header_extra = (f"n={N}, p={P}, s={S}, SNR={fmt_num(snr_fixed)}"
                    f", tFDR={fmt_num(TFDR)}, amplitude={fmt_num(AMPLITUDE)}")
    save_and_print_2d_gap_rho_results(
        OUT_DIR, "da_ar1_gap_rho", gap_grid, rho_grid, kappa_boundary, NUM_MC,
        names, tpp_cs, fdp_cs, sd_tpp_cs, sd_fdp_cs,
        tpp_rand, fdp_rand, sd_tpp_rand, sd_fdp_rand, header_extra)


# ==============================================================================
# Main
# ==============================================================================

if __name__ == "__main__":
    if RUN_PART_1:
        part_1_snr_sweep()
    if RUN_PART_2:
        part_2_rho_sweep()
    if RUN_PART_3:
        part_3_gap_rho_sweep()
    print("\nAR(1) MC simulation complete.")
