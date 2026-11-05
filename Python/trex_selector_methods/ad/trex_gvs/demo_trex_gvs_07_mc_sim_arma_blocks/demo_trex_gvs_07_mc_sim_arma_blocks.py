# ==============================================================================
# demo_trex_gvs_07_mc_sim_arma_blocks.py
# ==============================================================================
#
# MC simulation: T-Rex+GVS on heterogeneous ARMA blocks.
#
# Mirrors cpp/trex_selector_methods/ad/trex_gvs/demo_trex_gvs_07_mc_sim_arma_blocks.
#
# One part:
#   1. 2-D SNR x ar_coef sweep
#      (subsumes the former 1-D SNR and ar_coef sweeps as its ar_coef = 0.8
#       column and SNR = 2 row)
#
# DGP: dgp_arma_blocks_snr — 4 blocks of sizes {20, 50, 80, 65} with
#      heterogeneous within-block ARMA structure (AR(1) / MA(3) / ARMA(2,1) /
#      AR(1) trap); blocks 0, 1, 2 active with beta = 3; block 3 is an inactive
#      AR(1) trap; support size s = 150.
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

# Preset (mirrors cpp make_arma_blocks_preset()).
PRESET = dict(
    scenario_tag="ARMA-Blocks",
    file_stem_prefix="gvs_arma_blocks",
    snr_grid_2d=[0.2, 0.5, 1.0, 2.0, 5.0],
    ar_coef_grid_2d=[0.30, 0.50, 0.70, 0.80, 0.90, 0.95],
)


# =============================================================================
# Part 1 — 2-D SNR x ar_coef sweep
# =============================================================================

def run_arma_part1_2d_sweep(cfg, preset):
    """2-D MC sweep over SNR and ar_coef on the ARMA-blocks DGP."""

    def cell_kwargs(snr, ar_coef):
        return dict(n=cfg["n"], p=cfg["p"], snr=snr, ar_coef=ar_coef)

    print_section_header(
        f"Part 1: 2-D SNR x ar_coef Sweep | {preset['scenario_tag']}")
    print(f"  n={cfg['n']}  p={cfg['p']}  s=150"
          f"  MC={cfg['num_MC']}  tFDR={fmt_num(cfg['tFDR'])}  K={cfg['K']}"
          f"  corr_max={fmt_num(cfg['corr_max'])}\n"
          "  methods: TENET / TENET_AUG / TIENET_AUG"
          "  |  lambda2=CV_1SE_CCD (EN), CV_1SE_IEN_CCD (IEN)  linkage=Single\n"
          "  snr_grid: {"
          + ", ".join(fmt_num(s) for s in preset["snr_grid_2d"]) + "}\n"
          "  ar_coef_grid: {"
          + ", ".join(fmt_num(a) for a in preset["ar_coef_grid_2d"]) + "}")

    en = run_gvs_2d_sweep("dgp_arma_blocks_snr", cell_kwargs,
                          preset["snr_grid_2d"], preset["ar_coef_grid_2d"],
                          cfg, "EN", "TENET", "TENET",
                          hc_linkage="Single", lambda2_method="CV_1SE_CCD",
                          max_workers=N_WORKERS)
    en_aug = run_gvs_2d_sweep("dgp_arma_blocks_snr", cell_kwargs,
                              preset["snr_grid_2d"], preset["ar_coef_grid_2d"],
                              cfg, "EN", "TENET_AUG", "TENET_AUG",
                              hc_linkage="Single", lambda2_method="CV_1SE_CCD",
                              max_workers=N_WORKERS)
    # support cap 180 > s = 150 (library default 0.8 gives 144)
    ien = run_gvs_2d_sweep("dgp_arma_blocks_snr", cell_kwargs,
                           preset["snr_grid_2d"], preset["ar_coef_grid_2d"],
                           cfg, "IEN", None, "TIENET_AUG",
                           hc_linkage="Single",
                           lambda2_method="CV_1SE_IEN_CCD",
                           max_workers=N_WORKERS, cv_dfmax_ratio=1.0)

    snr_labels = [f"snr={fmt_num(s)}" for s in preset["snr_grid_2d"]]
    ar_labels = [f"ac={fmt_num(a)}" for a in preset["ar_coef_grid_2d"]]

    print_mc_matrix("mean_FDP [TENET]", snr_labels, ar_labels, en, False)
    print_mc_matrix("mean_TPP [TENET]", snr_labels, ar_labels, en, True)
    print_mc_matrix("mean_FDP [TENET_AUG]", snr_labels, ar_labels, en_aug, False)
    print_mc_matrix("mean_TPP [TENET_AUG]", snr_labels, ar_labels, en_aug, True)
    print_mc_matrix("mean_FDP [TIENET_AUG]", snr_labels, ar_labels, ien, False)
    print_mc_matrix("mean_TPP [TIENET_AUG]", snr_labels, ar_labels, ien, True)

    save_mc_2d_tables(preset["scenario_tag"], snr_labels, ar_labels, en,
                      en_aug, ien, cfg, OUT_DIR,
                      preset["file_stem_prefix"] + "_2d")


# =============================================================================
# Top-level driver
# =============================================================================

def run_arma_blocks_benchmark(cfg, preset):
    """Run the heterogeneous ARMA-blocks benchmark (2-D SNR x ar_coef sweep)."""
    print_section_header(
        f"ARMA-blocks benchmark | {preset['scenario_tag']}\n"
        "DGP = dgp_arma_blocks_snr\n"
        "blocks = {20, 50, 80, 65}, s = 150\n"
        "active blocks = {0, 1, 2}, trap block = 3")

    run_arma_part1_2d_sweep(cfg, preset)

    print(f"{preset['scenario_tag']} GVS MC simulations complete.")


# =============================================================================
# main
# =============================================================================

if __name__ == "__main__":
    print(f"Running with {N_WORKERS} worker processes\n")

    run_arma_blocks_benchmark(CFG, PRESET)

    print("All ARMA-block benchmark simulations complete.")
