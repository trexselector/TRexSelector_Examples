# ==============================================================================
# demo_trex_gvs_04_mc_sim_neg_traps.py
# ==============================================================================
#
# MC simulation: T-Rex+GVS on the negative-traps DGP.
# This scenario evaluates recovery of an active negatively correlated group in
# the presence of two spatially separated inactive trap groups with the same
# sign-flipped correlation structure.
#
# Mirrors cpp/trex_selector_methods/ad/trex_gvs/demo_trex_gvs_04_mc_sim_neg_traps.
#
# One part:
#   sim_neg_traps_part1() — 2-D SNR x rho grid, sd_x = sqrt((1 - rho)/rho)
#   (subsumes the former 1-D SNR and rho sweeps as its rho = 0.99 column and
#    SNR = 2 row)
#
# DGP: active group (0-99) + Trap1 (100-199) + noise (200-299) +
#      Trap2 (300-359) + noise (360+); s = 100.
# In each correlated group, one half loads positively and the other half
# negatively on a latent factor, inducing strong within-group sign structure.
# Selector: K=20, tFDR=0.1, corr_max=0.98, lambda2=CV_1SE_CCD (EN),
#   CV_1SE_IEN_CCD (IEN).
# Methods compared: TENET, TENET_AUG, TIENET_AUG.
# MC: 200 trials per grid point (the full run takes hours — reduce num_MC and
#     the grids for a quick smoke test). Optional first CLI argument sets the
#     number of parallel worker processes (default TREX_NUM_WORKERS or 8).
# Indices are 0-based (Python convention); the R counterpart is 1-based.
# ==============================================================================

import math
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.dirname(_THIS_DIR)
for _p in (_THIS_DIR, _PARENT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from gvs_sim_common import (N_CORES, resolve_workers, fmt_num, print_mc_matrix,  # noqa: E402
                            num_mc, print_section_header, run_gvs_2d_sweep,
                            save_mc_2d_tables)

# Optional CLI worker count, resolved like TREX_NUM_WORKERS (0 = all usable cores).
N_WORKERS = resolve_workers(sys.argv[1]) if len(sys.argv) > 1 else N_CORES

OUT_DIR = os.path.join(_THIS_DIR, "simulation_results", "data")

# Simulation parameters (mirrors the cpp GVSSimConfig block in main()).
CFG = dict(
    n=200, p=500,
    tFDR=0.1, K=20,
    num_MC=num_mc(200),  # TREX_NUM_MC overrides it for smoke runs
    base_seed=2026,
    corr_max=0.98,
)

SCENARIO_TAG = "Neg-Traps"


# ==============================================================================
#  Part 1 — 2-D SNR x rho sweep
# ==============================================================================

def sim_neg_traps_part1(cfg, tag):
    """2-D MC sweep over SNR and rho on the negative-traps DGP."""
    snr_grid_2d = [0.2, 0.5, 1.0, 2.0, 5.0]
    rho_grid_2d = [0.30, 0.50, 0.70, 0.90, 0.95, 0.99]

    def cell_kwargs(snr, rho):
        sd_x = math.sqrt((1.0 - rho) / rho)
        return dict(n=cfg["n"], p=cfg["p"], snr=snr, sd_x=sd_x)

    print_section_header(f"Part 1: 2-D SNR x rho Sweep  |  {tag}")
    print(f"  n={cfg['n']}  p={cfg['p']}  s=100  MC={cfg['num_MC']}"
          f"  tFDR={fmt_num(cfg['tFDR'])}  K={cfg['K']}"
          f"  corr_max={fmt_num(cfg['corr_max'])}\n"
          "  methods: TENET / TENET_AUG / TIENET_AUG"
          "  |  lambda2=CV_1SE_CCD (EN), CV_1SE_IEN_CCD (IEN)  linkage=Single\n"
          "  snr_grid: {" + ", ".join(fmt_num(s) for s in snr_grid_2d) + "}\n"
          "  rho_grid: {" + ", ".join(fmt_num(r) for r in rho_grid_2d) + "}")

    en_2d = run_gvs_2d_sweep("dgp_neg_traps_snr", cell_kwargs, snr_grid_2d,
                             rho_grid_2d, cfg, "EN", "TENET", "TENET",
                             hc_linkage="Single", lambda2_method="CV_1SE_CCD",
                             max_workers=N_WORKERS)
    en_aug_2d = run_gvs_2d_sweep("dgp_neg_traps_snr", cell_kwargs, snr_grid_2d,
                                 rho_grid_2d, cfg, "EN", "TENET_AUG",
                                 "TENET_AUG", hc_linkage="Single",
                                 lambda2_method="CV_1SE_CCD",
                                 max_workers=N_WORKERS)
    # support cap 180 as in demos 01-07 (library default 0.8 gives 144)
    ien_2d = run_gvs_2d_sweep("dgp_neg_traps_snr", cell_kwargs, snr_grid_2d,
                              rho_grid_2d, cfg, "IEN", None, "TIENET_AUG",
                              hc_linkage="Single",
                              lambda2_method="CV_1SE_IEN_CCD",
                              max_workers=N_WORKERS, cv_dfmax_ratio=1.0)

    snr_labels = [f"snr={fmt_num(s)}" for s in snr_grid_2d]
    rho_labels = [f"rho={fmt_num(r)}" for r in rho_grid_2d]

    print_mc_matrix("mean_TPP [TENET]", snr_labels, rho_labels, en_2d, True)
    print_mc_matrix("mean_FDP [TENET]", snr_labels, rho_labels, en_2d, False)
    print_mc_matrix("mean_TPP [TENET_AUG]", snr_labels, rho_labels, en_aug_2d, True)
    print_mc_matrix("mean_FDP [TENET_AUG]", snr_labels, rho_labels, en_aug_2d, False)
    print_mc_matrix("mean_TPP [TIENET_AUG]", snr_labels, rho_labels, ien_2d, True)
    print_mc_matrix("mean_FDP [TIENET_AUG]", snr_labels, rho_labels, ien_2d, False)
    save_mc_2d_tables(tag, snr_labels, rho_labels, en_2d, en_aug_2d, ien_2d,
                      cfg, OUT_DIR, f"gvs_{tag}_2d")


# ==============================================================================
#  main
# ==============================================================================

if __name__ == "__main__":
    print(f"Running with {N_WORKERS} worker processes\n")

    sim_neg_traps_part1(CFG, SCENARIO_TAG)

    print(f"{SCENARIO_TAG} GVS MC simulations complete.")
