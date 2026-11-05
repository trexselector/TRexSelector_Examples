# ==============================================================================
# demo_trex_gvs_02_mc_sim_scattered_blocks.py
# ==============================================================================
#
# MC simulation: T-Rex+GVS on the scattered-grouped DGP.
#
# This scenario evaluates grouped-signal recovery when active variables are
# randomly scattered across the design rather than arranged in contiguous
# blocks. Within-group correlation is controlled through sd_x via
# rho = 1 / (1 + sd_x^2).
#
# Mirrors cpp/trex_selector_methods/ad/trex_gvs/demo_trex_gvs_02_mc_sim_scattered_blocks.
#
# One part:
#   1. 2-D SNR x rho sweep, using sd_x = sqrt((1 - rho)/rho)
#      (subsumes the former 1-D SNR and rho sweeps as its rho = 0.99 column and
#       SNR = 2 row)
#
# DGP: dgp_scattered_grouped_snr — n_groups = 3, group_size = 50, all grouped
#      variables active (s = 150), p = 500, remaining columns i.i.d. noise.
# Selector: K=20, tFDR=0.1, corr_max=0.98, single-linkage HAC,
#   lambda2 = CV_1SE_CCD (EN rows), CV_1SE_IEN_CCD (IEN rows).
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

# Preset (mirrors cpp make_scattered_grouped_preset()).
PRESET = dict(
    scenario_tag="Scattered-Grouped",
    file_stem_prefix="gvs_scattered_grouped",
    n_groups=3,
    group_size=50,
    snr_grid_2d=[0.2, 0.5, 1.0, 2.0, 5.0],
    rho_grid_2d=[0.30, 0.50, 0.70, 0.90, 0.95, 0.99],
)


# =============================================================================
# Part 1 — 2-D SNR x rho sweep
# =============================================================================

def run_scattered_part1_2d_sweep(cfg, preset):
    """2-D MC sweep over SNR and rho on the scattered-grouped DGP."""
    n_groups = preset["n_groups"]
    group_size = preset["group_size"]

    def cell_kwargs(snr, rho):
        sd_x = math.sqrt((1.0 - rho) / rho)
        return dict(n=cfg["n"], p=cfg["p"], snr=snr, sd_x=sd_x,
                    n_groups=n_groups, group_size=group_size)

    print_section_header(
        f"Part 1: 2-D SNR x rho Sweep | {preset['scenario_tag']}")
    print(f"  n={cfg['n']}  p={cfg['p']}  s={n_groups * group_size}"
          f"  MC={cfg['num_MC']}  tFDR={fmt_num(cfg['tFDR'])}  K={cfg['K']}"
          f"  corr_max={fmt_num(cfg['corr_max'])}\n"
          "  methods: TENET / TENET_AUG / TIENET_AUG"
          "  |  lambda2=CV_1SE_CCD (EN), CV_1SE_IEN_CCD (IEN)  linkage=Single\n"
          "  snr_grid: {"
          + ", ".join(fmt_num(s) for s in preset["snr_grid_2d"]) + "}\n"
          "  rho_grid: {"
          + ", ".join(fmt_num(r) for r in preset["rho_grid_2d"]) + "}")

    en = run_gvs_2d_sweep("dgp_scattered_grouped_snr", cell_kwargs,
                          preset["snr_grid_2d"], preset["rho_grid_2d"], cfg,
                          "EN", "TENET", "TENET",
                          hc_linkage="Single", lambda2_method="CV_1SE_CCD",
                          max_workers=N_WORKERS)
    en_aug = run_gvs_2d_sweep("dgp_scattered_grouped_snr", cell_kwargs,
                              preset["snr_grid_2d"], preset["rho_grid_2d"],
                              cfg, "EN", "TENET_AUG", "TENET_AUG",
                              hc_linkage="Single", lambda2_method="CV_1SE_CCD",
                              max_workers=N_WORKERS)
    # support cap 180 > s = 150 (library default 0.8 gives 144)
    ien = run_gvs_2d_sweep("dgp_scattered_grouped_snr", cell_kwargs,
                           preset["snr_grid_2d"], preset["rho_grid_2d"], cfg,
                           "IEN", None, "TIENET_AUG",
                           hc_linkage="Single",
                           lambda2_method="CV_1SE_IEN_CCD",
                           max_workers=N_WORKERS, cv_dfmax_ratio=1.0)

    snr_labels = [f"snr={fmt_num(s)}" for s in preset["snr_grid_2d"]]
    rho_labels = [f"rho={fmt_num(r)}" for r in preset["rho_grid_2d"]]

    print_mc_matrix("mean_TPP [TENET]", snr_labels, rho_labels, en, True)
    print_mc_matrix("mean_FDP [TENET]", snr_labels, rho_labels, en, False)
    print_mc_matrix("mean_TPP [TENET_AUG]", snr_labels, rho_labels, en_aug, True)
    print_mc_matrix("mean_FDP [TENET_AUG]", snr_labels, rho_labels, en_aug, False)
    print_mc_matrix("mean_TPP [TIENET_AUG]", snr_labels, rho_labels, ien, True)
    print_mc_matrix("mean_FDP [TIENET_AUG]", snr_labels, rho_labels, ien, False)

    save_mc_2d_tables(preset["scenario_tag"], snr_labels, rho_labels, en,
                      en_aug, ien, cfg, OUT_DIR,
                      preset["file_stem_prefix"] + "_2d")


# =============================================================================
# Top-level driver
# =============================================================================

def run_scattered_grouped_benchmark(cfg, preset):
    """Run the scattered-grouped benchmark (2-D SNR x rho sweep)."""
    print_section_header(
        f"Scattered-grouped benchmark | {preset['scenario_tag']}\n"
        "DGP = dgp_scattered_grouped_snr\n"
        f"n_groups = {preset['n_groups']}, group_size = {preset['group_size']},"
        f" s = {preset['n_groups'] * preset['group_size']}")

    run_scattered_part1_2d_sweep(cfg, preset)

    print(f"{preset['scenario_tag']} GVS MC simulations complete.")


# =============================================================================
# main
# =============================================================================

if __name__ == "__main__":
    print(f"Running with {N_WORKERS} worker processes\n")

    run_scattered_grouped_benchmark(CFG, PRESET)

    print("All scattered-grouped benchmark simulations complete.")
