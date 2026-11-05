# ==============================================================================
# demo_trex_gvs_01_mc_sim_hastie_en_blocks.py
# ==============================================================================
#
# MC simulation: T-Rex+GVS on the Hastie DGP.
# This scenario evaluates grouped-signal recovery under strong equicorrelation,
# not sparse within-group support identification. Within-group correlation is
# controlled through sd_x, with rho = 1 / (1 + sd_x^2).
#
# Mirrors cpp/trex_selector_methods/ad/trex_gvs/demo_trex_gvs_01_mc_sim_hastie_en_blocks.
#
# Four parts, each in its own function:
#   sim_hastie_part_1() — 2-D SNR x rho grid
#   sim_hastie_part_2() — lambda2_method comparison: SVD / CCD (EN),
#                         TIK_SVD / IEN_CCD (IEN)
#   sim_hastie_part_3() — hc_linkage comparison: Single / Complete / Average / WPGMA
#   sim_hastie_part_4() — clustering cut: corr_max 0.98 / noise floor / oracle groups
#
# DGP: three equicorrelated groups of 50 variables each, all active (s = 150),
#      plus 350 null white-noise variables; p = 500 and n = 200.
# Selector: K=20, tFDR=0.1, corr_max=0.98, lambda2=CV_1SE_CCD on the EN track
#   and the IEN-geometry CV_1SE_IEN_CCD on the IEN track (Part 2 compares it
#   with CV_1SE_TIK_SVD).
# Methods compared: TENET, TENET_AUG, TIENET_AUG (Part 4: TENET, TIENET_AUG).
# Part selection: TREX_GVS_PARTS (default "1234"), e.g. TREX_GVS_PARTS=4 runs
#   only the cut comparison.
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
                            noise_floor_corr_max, num_mc, print_section_header,
                            run_gvs_2d_sweep, run_gvs_mc_trials,
                            run_gvs_mc_trials_cut, save_mc_1d_method_tables,
                            save_mc_2d_tables, save_mc_cut_tables)

# Optional CLI worker count, resolved like TREX_NUM_WORKERS (0 = all usable cores).
N_WORKERS = resolve_workers(sys.argv[1]) if len(sys.argv) > 1 else N_CORES

OUT_DIR = os.path.join(_THIS_DIR, "simulation_results", "data")

# Simulation parameters (mirrors the cpp GVSSimConfig block in main()).
CFG = dict(
    n=200, p=500,
    sd_x=0.1,           # sqrt(0.01) => rho ~ 0.99 (Parts fix rho locally)
    tFDR=0.1, K=20,
    num_MC=num_mc(200),  # TREX_NUM_MC overrides it for smoke runs
    base_seed=2026,
    corr_max=0.98,
    snr=2.0,           # unused here; Parts 2-3 fix SNR = 2.0 locally
)

SCENARIO_TAG = "Hastie"

# Part selection (default "1234"); not part of the cache fingerprint
PARTS = os.environ.get("TREX_GVS_PARTS", "") or "1234"


# ==============================================================================
#  Part 1 — 2-D SNR x rho sweep
# ==============================================================================

def sim_hastie_part_1(cfg, tag):
    """2-D MC sweep over SNR and rho: TENET / TENET_AUG / TIENET_AUG."""
    snr_grid_2d = [0.2, 0.5, 1.0, 2.0, 5.0]
    rho_grid_2d = [0.30, 0.50, 0.70, 0.90, 0.95, 0.99]

    def cell_kwargs(snr, rho):
        sd_x = math.sqrt((1.0 - rho) / rho)
        return dict(n=cfg["n"], p=cfg["p"], snr=snr, sd_x=sd_x)

    print_section_header(f"Part 1/4: 2-D SNR x rho Sweep  |  {tag}")
    print(f"  n={cfg['n']}  p={cfg['p']}  s=150  MC={cfg['num_MC']}"
          f"  tFDR={fmt_num(cfg['tFDR'])}  K={cfg['K']}"
          f"  corr_max={fmt_num(cfg['corr_max'])}\n"
          "  methods: TENET / TENET_AUG / TIENET_AUG"
          "  |  lambda2=CV_1SE_CCD (EN), CV_1SE_IEN_CCD (IEN)"
          "  linkage=Single\n"
          "  snr_grid: {" + ", ".join(fmt_num(s) for s in snr_grid_2d) + "}\n"
          "  rho_grid: {" + ", ".join(fmt_num(r) for r in rho_grid_2d) + "}\n")

    en_2d = run_gvs_2d_sweep("dgp_hastie_snr", cell_kwargs, snr_grid_2d,
                             rho_grid_2d, cfg, "EN", "TENET", "TENET",
                             hc_linkage="Single", lambda2_method="CV_1SE_CCD",
                             max_workers=N_WORKERS)
    en_aug_2d = run_gvs_2d_sweep("dgp_hastie_snr", cell_kwargs, snr_grid_2d,
                                 rho_grid_2d, cfg, "EN", "TENET_AUG",
                                 "TENET_AUG", hc_linkage="Single",
                                 lambda2_method="CV_1SE_CCD",
                                 max_workers=N_WORKERS)
    # support cap 180 > s = 150 (library default 0.8 gives 144)
    ien_2d = run_gvs_2d_sweep("dgp_hastie_snr", cell_kwargs, snr_grid_2d,
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
#  Part 2 — lambda2_method comparison
# ==============================================================================

def sim_hastie_part_2(cfg, tag):
    """Compare lambda2 selection methods at the fixed operating point
    SNR = 2.0, rho = 0.7: CV_1SE_SVD vs CV_1SE_CCD for TENET, and the two
    IEN-geometry tuners CV_1SE_TIK_SVD vs CV_1SE_IEN_CCD for TIENET_AUG (the
    EN track rejects them, and an EN tuner on the IEN track would fit the
    wrong objective). With cv_dfmax_ratio = 1.0, CV_1SE_IEN_CCD caps its
    lambda1 path at n_train = 180 variables, above this design's s = 150
    (the library default 0.8 gives 144). CV_1SE_TIK_SVD ignores the cap and
    needs p - M < n_train (M clusters, so p - M within-group contrasts). The
    library refuses it otherwise, e.g. on the trap blocks of demo 04 at
    rho = 0.99, so the sweeps of demos 01-07 use CV_1SE_IEN_CCD."""
    fixed_snr, fixed_rho = 2.0, 0.7
    fixed_sdx = math.sqrt((1.0 - fixed_rho) / fixed_rho)
    dgp_kwargs = dict(n=cfg["n"], p=cfg["p"], snr=fixed_snr, sd_x=fixed_sdx)

    # Rules per track: the IEN-geometry tuners are valid on the IEN track only.
    en_methods = ["CV_1SE_SVD", "CV_1SE_CCD"]
    ien_methods = ["CV_1SE_TIK_SVD", "CV_1SE_IEN_CCD"]

    print_section_header(f"Part 2/4: lambda2_method Comparison  |  {tag}")
    print(f"  n={cfg['n']}  p={cfg['p']}  s=150  MC={cfg['num_MC']}"
          f"  tFDR={fmt_num(cfg['tFDR'])}  K={cfg['K']}\n"
          f"  fixed operating point: SNR={fmt_num(fixed_snr)}"
          f"  rho={fmt_num(fixed_rho)}\n"
          "  lambda2 methods: TENET CV_1SE_SVD / CV_1SE_CCD,"
          " TIENET_AUG CV_1SE_TIK_SVD / CV_1SE_IEN_CCD  (4 runs)\n")

    en_results, ien_results = [], []
    for lm_en, lm_ien in zip(en_methods, ien_methods):
        en_results.append(run_gvs_mc_trials(
            "dgp_hastie_snr", dgp_kwargs, cfg["num_MC"], f"TENET+{lm_en}",
            cfg["tFDR"], cfg["K"], "EN", cfg["corr_max"],
            hc_linkage="Single", en_solver="TENET", lambda2_method=lm_en,
            base_seed=cfg["base_seed"], max_workers=N_WORKERS))
        ien_results.append(run_gvs_mc_trials(
            "dgp_hastie_snr", dgp_kwargs, cfg["num_MC"],
            f"TIENET_AUG+{lm_ien}",
            cfg["tFDR"], cfg["K"], "IEN", cfg["corr_max"],
            hc_linkage="Single", en_solver=None, lambda2_method=lm_ien,
            base_seed=cfg["base_seed"], max_workers=N_WORKERS,
            cv_dfmax_ratio=1.0))

    save_mc_1d_method_tables(
        tag, "lambda2_method", "lambda2_method", en_methods, en_results,
        ien_results, f"SNR={fmt_num(fixed_snr)}, rho={fmt_num(fixed_rho)}",
        cfg, OUT_DIR, f"gvs_{tag}_lambda2_method",
        ien_param_labels=ien_methods)


# ==============================================================================
#  Part 3 — hc_linkage comparison
# ==============================================================================

def sim_hastie_part_3(cfg, tag):
    """Compare HAC linkage rules (Single / Complete / Average / WPGMA) for
    TENET and TIENET_AUG at the same fixed operating point as Part 2, with
    lambda2 fixed at CV_1SE_CCD (EN) and CV_1SE_IEN_CCD (IEN) to isolate the
    linkage effect."""
    fixed_snr, fixed_rho = 2.0, 0.7
    fixed_sdx = math.sqrt((1.0 - fixed_rho) / fixed_rho)
    dgp_kwargs = dict(n=cfg["n"], p=cfg["p"], snr=fixed_snr, sd_x=fixed_sdx)

    linkages = ["Single", "Complete", "Average", "WPGMA"]

    print_section_header(f"Part 3/4: hc_linkage Comparison  |  {tag}")
    print(f"  n={cfg['n']}  p={cfg['p']}  s=150  MC={cfg['num_MC']}"
          f"  tFDR={fmt_num(cfg['tFDR'])}  K={cfg['K']}"
          "  lambda2=CV_1SE_CCD (EN), CV_1SE_IEN_CCD (IEN)\n"
          f"  fixed operating point: SNR={fmt_num(fixed_snr)}"
          f"  rho={fmt_num(fixed_rho)}\n"
          "  linkages: Single / Complete / Average / WPGMA"
          "  |  selectors: TENET, TIENET_AUG  (8 runs)\n")

    en_results, ien_results = [], []
    for lk in linkages:
        en_results.append(run_gvs_mc_trials(
            "dgp_hastie_snr", dgp_kwargs, cfg["num_MC"], f"TENET+{lk}",
            cfg["tFDR"], cfg["K"], "EN", cfg["corr_max"],
            hc_linkage=lk, en_solver="TENET", lambda2_method="CV_1SE_CCD",
            base_seed=cfg["base_seed"], max_workers=N_WORKERS))
        ien_results.append(run_gvs_mc_trials(
            "dgp_hastie_snr", dgp_kwargs, cfg["num_MC"], f"TIENET_AUG+{lk}",
            cfg["tFDR"], cfg["K"], "IEN", cfg["corr_max"],
            hc_linkage=lk, en_solver=None, lambda2_method="CV_1SE_IEN_CCD",
            base_seed=cfg["base_seed"], max_workers=N_WORKERS,
            cv_dfmax_ratio=1.0))

    save_mc_1d_method_tables(
        tag, "hc_linkage", "hc_linkage", linkages, en_results, ien_results,
        f"SNR={fmt_num(fixed_snr)}, rho={fmt_num(fixed_rho)}",
        cfg, OUT_DIR, f"gvs_{tag}_hc_linkage")


# ==============================================================================
#  Part 4 — clustering-cut comparison
# ==============================================================================

def sim_hastie_part_4(cfg, tag):
    """Compare the clustering cut at the fixed operating point SNR = 2.0,
    rho = 0.7: the demos' fixed corr_max = 0.98 (no groups below rho = 0.99),
    the Bonferroni noise floor (about 0.35 here) and the DGP's oracle groups,
    for TENET and TIENET_AUG with lambda2 = CV_1SE_CCD (EN) or CV_1SE_IEN_CCD
    (IEN) and single linkage.
    Records variable-level FDP / TPP next to group-level rates."""
    fixed_snr, fixed_rho = 2.0, 0.7
    fixed_sdx = math.sqrt((1.0 - fixed_rho) / fixed_rho)
    dgp_kwargs = dict(n=cfg["n"], p=cfg["p"], snr=fixed_snr, sd_x=fixed_sdx)
    floor_cut = noise_floor_corr_max(cfg["n"], cfg["p"])

    cuts = [("fixed_0.98", cfg["corr_max"], False),
            ("noise_floor", floor_cut, False),
            ("oracle", cfg["corr_max"], True)]

    print_section_header(f"Part 4/4: Clustering-cut Comparison  |  {tag}")
    print(f"  n={cfg['n']}  p={cfg['p']}  s=150  MC={cfg['num_MC']}"
          f"  tFDR={fmt_num(cfg['tFDR'])}  K={cfg['K']}"
          "  lambda2=CV_1SE_CCD (EN), CV_1SE_IEN_CCD (IEN)  linkage=Single\n"
          f"  fixed operating point: SNR={fmt_num(fixed_snr)}"
          f"  rho={fmt_num(fixed_rho)}\n"
          f"  cuts: corr_max {fmt_num(cfg['corr_max'])} / noise floor"
          f" {floor_cut:.3f} / oracle groups"
          "  |  selectors: TENET, TIENET_AUG  (6 runs)\n")

    en_results, ien_results = [], []
    for label, corr_max, oracle in cuts:
        en_results.append(run_gvs_mc_trials_cut(
            "dgp_hastie_snr", dgp_kwargs, cfg["num_MC"], f"TENET+{label}",
            cfg["tFDR"], cfg["K"], "EN", corr_max, hc_linkage="Single",
            en_solver="TENET", lambda2_method="CV_1SE_CCD",
            base_seed=cfg["base_seed"], oracle=oracle, max_workers=N_WORKERS))
        ien_results.append(run_gvs_mc_trials_cut(
            "dgp_hastie_snr", dgp_kwargs, cfg["num_MC"], f"TIENET_AUG+{label}",
            cfg["tFDR"], cfg["K"], "IEN", corr_max, hc_linkage="Single",
            en_solver=None, lambda2_method="CV_1SE_IEN_CCD",
            base_seed=cfg["base_seed"], oracle=oracle, max_workers=N_WORKERS,
            cv_dfmax_ratio=1.0))

    save_mc_cut_tables(tag, [c[0] for c in cuts], en_results, ien_results,
                       fixed_snr, fixed_rho, floor_cut, cfg, OUT_DIR,
                       f"gvs_{tag}_cut")


# ==============================================================================
#  main
# ==============================================================================

if __name__ == "__main__":
    print(f"Running with {N_WORKERS} worker processes\n")

    if "1" in PARTS:
        sim_hastie_part_1(CFG, SCENARIO_TAG)
    if "2" in PARTS:
        sim_hastie_part_2(CFG, SCENARIO_TAG)
    if "3" in PARTS:
        sim_hastie_part_3(CFG, SCENARIO_TAG)
    if "4" in PARTS:
        sim_hastie_part_4(CFG, SCENARIO_TAG)

    print(f"\n{SCENARIO_TAG} GVS MC simulations complete.")
