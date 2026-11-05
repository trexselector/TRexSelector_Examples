# ==============================================================================
# demo_trex_gvs_05_mc_sim_t3_blocks.py
# ==============================================================================
#
# MC simulation: T-Rex+GVS on heavy-tailed equicorrelated blocks.
#
# This benchmark uses the same four-block active/trap geometry as the
# mixed-blocks / equicorrelated-blocks Gaussian scenarios, but replaces the
# Gaussian latent factors, within-block perturbations, and response noise by
# scaled Student-t(3) variables via dgp_t3_equi_blocks_snr.
#
# Mirrors cpp/trex_selector_methods/ad/trex_gvs/demo_trex_gvs_05_mc_sim_t3_blocks.
#
# One part:
#   1. 2-D SNR x rho sweep
#      (subsumes the former 1-D SNR and rho sweeps; the former fixed operating
#       points rho = 0.75 and SNR = 2 lie on or between the grid's columns/rows)
#
# DGP: block sizes = {20, 50, 80, 65}; blocks 0,1,2 active (beta = 3); block 3
#      inactive heavy-tailed trap; total active support size s = 150.
# Selector: K=20, tFDR=0.1, corr_max=0.98, single-linkage HAC,
#   lambda2 = CV_1SE_CCD (EN rows), CV_1SE_IEN_CCD (IEN rows).
# Methods compared: TENET, TENET_AUG, TIENET_AUG.
# MC: 200 trials per grid point (the full run takes hours — reduce num_MC and
#     the grids for a quick smoke test). Optional first CLI argument sets the
#     number of parallel worker processes (default TREX_NUM_WORKERS or 8).
# Indices are 0-based (Python convention); the R counterpart is 1-based.
# ==============================================================================

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

# Preset (mirrors cpp make_t3_blocks_preset()).
PRESET = dict(
    scenario_tag="t3-Blocks",
    file_stem_prefix="gvs_t3_blocks",
    snr_grid_2d=[0.2, 0.5, 1.0, 2.0, 5.0],
    rho_grid_2d=[0.30, 0.50, 0.70, 0.85, 0.90, 0.99],
)


# =============================================================================
# Part 1 — 2-D SNR x rho sweep
# =============================================================================

def run_t3_part1_2d_sweep(cfg, preset):
    """2-D MC sweep over SNR and rho on the heavy-tailed t(3)-blocks DGP."""

    def cell_kwargs(snr, rho):
        # rho enters the DGP directly (no sd_x reparameterization).
        return dict(n=cfg["n"], p=cfg["p"], snr=snr, rho=rho)

    print_section_header(
        f"Part 1: 2-D SNR x rho Sweep | {preset['scenario_tag']}")
    print(f"  n={cfg['n']}  p={cfg['p']}  s=150"
          f"  MC={cfg['num_MC']}  tFDR={fmt_num(cfg['tFDR'])}  K={cfg['K']}"
          f"  corr_max={fmt_num(cfg['corr_max'])}\n"
          "  methods: TENET / TENET_AUG / TIENET_AUG"
          "  |  lambda2=CV_1SE_CCD (EN), CV_1SE_IEN_CCD (IEN)  linkage=Single\n"
          "  snr_grid: {"
          + ", ".join(fmt_num(s) for s in preset["snr_grid_2d"]) + "}\n"
          "  rho_grid: {"
          + ", ".join(fmt_num(r) for r in preset["rho_grid_2d"]) + "}")

    en = run_gvs_2d_sweep("dgp_t3_equi_blocks_snr", cell_kwargs,
                          preset["snr_grid_2d"], preset["rho_grid_2d"], cfg,
                          "EN", "TENET", "TENET",
                          hc_linkage="Single", lambda2_method="CV_1SE_CCD",
                          max_workers=N_WORKERS)
    en_aug = run_gvs_2d_sweep("dgp_t3_equi_blocks_snr", cell_kwargs,
                              preset["snr_grid_2d"], preset["rho_grid_2d"],
                              cfg, "EN", "TENET_AUG", "TENET_AUG",
                              hc_linkage="Single", lambda2_method="CV_1SE_CCD",
                              max_workers=N_WORKERS)
    # support cap 180 > s = 150 (library default 0.8 gives 144)
    ien = run_gvs_2d_sweep("dgp_t3_equi_blocks_snr", cell_kwargs,
                           preset["snr_grid_2d"], preset["rho_grid_2d"], cfg,
                           "IEN", None, "TIENET_AUG",
                           hc_linkage="Single",
                           lambda2_method="CV_1SE_IEN_CCD",
                           max_workers=N_WORKERS, cv_dfmax_ratio=1.0)

    snr_labels = [f"snr={fmt_num(s)}" for s in preset["snr_grid_2d"]]
    rho_labels = [f"rho={fmt_num(r)}" for r in preset["rho_grid_2d"]]

    print_mc_matrix("mean_FDP [TENET]", snr_labels, rho_labels, en, False)
    print_mc_matrix("mean_TPP [TENET]", snr_labels, rho_labels, en, True)
    print_mc_matrix("mean_FDP [TENET_AUG]", snr_labels, rho_labels, en_aug, False)
    print_mc_matrix("mean_TPP [TENET_AUG]", snr_labels, rho_labels, en_aug, True)
    print_mc_matrix("mean_FDP [TIENET_AUG]", snr_labels, rho_labels, ien, False)
    print_mc_matrix("mean_TPP [TIENET_AUG]", snr_labels, rho_labels, ien, True)

    save_mc_2d_tables(preset["scenario_tag"], snr_labels, rho_labels, en,
                      en_aug, ien, cfg, OUT_DIR,
                      preset["file_stem_prefix"] + "_2d")


# =============================================================================
# Top-level driver
# =============================================================================

def run_t3_blocks_benchmark(cfg, preset):
    """Run the heavy-tailed t(3)-blocks benchmark (2-D SNR x rho sweep)."""
    print_section_header(
        f"Heavy-tailed mixed-blocks benchmark | {preset['scenario_tag']}\n"
        "DGP = dgp_t3_equi_blocks_snr\n"
        "blocks = {20, 50, 80, 65}, active = {0,1,2}, inactive trap = {3}")

    run_t3_part1_2d_sweep(cfg, preset)

    print(f"{preset['scenario_tag']} GVS MC simulations complete.")


# =============================================================================
# main
# =============================================================================

if __name__ == "__main__":
    print(f"Running with {N_WORKERS} worker processes\n")

    run_t3_blocks_benchmark(CFG, PRESET)

    print("All t(3)-blocks benchmark simulations complete.")
