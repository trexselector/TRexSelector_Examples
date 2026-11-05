# ==============================================================================
# demo_trex_gvs_08_mc_sim_block_bench.py
# ==============================================================================
#
# Monte Carlo benchmark: value of prior group information vs. HAC
# discoverability on the block-equicorrelated DGP.
#
# Mirrors cpp/trex_selector_methods/ad/trex_gvs/demo_trex_gvs_08_mc_sim_block_bench.
#
# Tests four method variants over a (rho x snr) grid:
#
#   M1 = T-Rex + EN, HAC-clustered groups
#   M2 = T-Rex + EN, oracle groups
#   M3 = T-Rex + IEN, HAC-clustered groups
#   M4 = T-Rex + IEN, oracle groups
#
# Headline scenario (WHOLE truth — active blocks are fully active, matching
# the grouped-selection premise of the T-Rex+GVS papers):
# rho sweeps across the HAC discoverability boundary set by corr_max = 0.5.
# At low rho the clustered methods see near-singletons and the oracle prior
# is the only source of group structure; as rho grows, HAC rediscovers the
# blocks and the oracle-vs-clustered gap closes.
#
# Cautionary appendix (INDIVIDUAL truth — 1 of block_size variables active):
# within a block the null siblings are exchangeable with the active member,
# so coordinate-level FDR control is unattainable by construction; run at a
# low/high rho pair to show the collapse scaling with correlation.
#
# Fixed benchmark constants:
#   n = 200, p = 500, G = 100, block_size = 5, n_active_blocks = 10
#   tFDR = 0.1, K = 20, corr_max = 0.5, b = 3.0
#   snr_grid = {0.1, 0.2, 0.5, 1.0, 2.0, 5.0}
#   rho_grid (WHOLE)      = {0.2, 0.35, 0.5, 0.65, 0.8, 0.9}
#   rho_grid (INDIVIDUAL) = {0.35, 0.9}
#   num_MC = 200 (the full run takes hours — reduce num_MC / the grids for a
#   quick smoke test; the optional first CLI argument sets the number of
#   parallel worker processes, default TREX_NUM_WORKERS or 8)
#
# Seeds are staggered by grid-cell index so that each (rho, snr) cell uses a
# distinct seed band:
#
#   cell_base_seed = base_seed + cell_index * num_MC
#
# where cell_index = i_rho * n_snr + i_snr (i_rho indexes the scenario's own
# rho grid). All four methods see identical data within a trial.
# Indices are 0-based (Python convention); the R counterpart is 1-based.
# ==============================================================================

import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.dirname(_THIS_DIR)
for _p in (_THIS_DIR, _PARENT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from gvs_sim_common import (N_CORES, resolve_workers, fmt_num, print_block_grid_table,  # noqa: E402
                            print_section_header, run_block_mc)

# Optional CLI worker count, resolved like TREX_NUM_WORKERS (0 = all usable cores).
N_WORKERS = resolve_workers(sys.argv[1]) if len(sys.argv) > 1 else N_CORES

OUT_DIR = os.path.join(_THIS_DIR, "simulation_results", "data")

# Preset (mirrors cpp BlockBenchPreset).
PRESET = dict(
    n=200, p=500,
    G=100, block_size=5, n_active_blocks=10,
    tFDR=0.1, K=20,
    corr_max=0.5,       # HAC discoverability boundary (much lower than 0.98)
    base_seed=2026,
    num_MC=200,
    b=3.0,
    en_solver="TENET_AUG",   # EN solver variant for the M1/M2 (EN) methods
    snr_grid=[0.1, 0.2, 0.5, 1.0, 2.0, 5.0],
    # Headline WHOLE-truth sweep: crosses the corr_max = 0.5 discoverability
    # boundary (0.2/0.35 undiscoverable, 0.5 partial, 0.65-0.9 discovered).
    rho_grid_whole=[0.2, 0.35, 0.5, 0.65, 0.8, 0.9],
    # Cautionary INDIVIDUAL-truth pair: weak vs. strong sibling correlation.
    rho_grid_individual=[0.35, 0.9],
)


# =============================================================================
# One scenario
# =============================================================================

def run_block_bench_scenario(preset, scenario, rho_grid):
    """Run the full (rho x snr) grid for one truth scenario.

    `rho_grid` is scenario-specific: WHOLE sweeps the full discoverability
    range; INDIVIDUAL uses a low/high pair.
    """
    tag = scenario

    print_section_header(
        f"Block Bench MC: {tag}\n"
        f"n={preset['n']} p={preset['p']} G={preset['G']}"
        f" block={preset['block_size']} active={preset['n_active_blocks']}"
        f" tFDR={fmt_num(preset['tFDR'])} K={preset['K']}"
        f" MC={preset['num_MC']} corr_max={fmt_num(preset['corr_max'])}")

    n_rho = len(rho_grid)
    n_snr = len(preset["snr_grid"])
    results = [[None] * n_snr for _ in range(n_rho)]

    for i_rho, rho in enumerate(rho_grid):
        for i_snr, snr in enumerate(preset["snr_grid"]):
            cell_index = i_rho * n_snr + i_snr
            cell_base_seed = preset["base_seed"] + cell_index * preset["num_MC"]
            label = (f"{tag} rho={rho:.2f} snr={snr:.2f}"
                     f" [{cell_index + 1}/{n_rho * n_snr}]")
            results[i_rho][i_snr] = run_block_mc(
                preset, scenario, rho, snr, cell_base_seed, label,
                max_workers=N_WORKERS)

    print_block_grid_table(tag, scenario, rho_grid, preset["snr_grid"],
                           results, preset, OUT_DIR,
                           "gvs_block_bench_" + tag.lower())


# =============================================================================
# Top-level benchmark
# =============================================================================

def run_block_benchmark(preset):
    """Run the full block-equicorrelated benchmark.

    WHOLE is the headline scenario (prior-information value vs. HAC
    discoverability); INDIVIDUAL is the cautionary appendix (coordinate-level
    FDR under exchangeable null siblings).
    """
    print_section_header(
        "Block-equicorrelated benchmark\n"
        "DGP = dgp_block_equicorr\n"
        "methods = {EN clustered, EN oracle, IEN clustered, IEN oracle}\n"
        "scenarios = {WHOLE (headline), INDIVIDUAL (appendix)}")

    run_block_bench_scenario(preset, "WHOLE", preset["rho_grid_whole"])
    run_block_bench_scenario(preset, "INDIVIDUAL",
                             preset["rho_grid_individual"])

    print("Block benchmark MC simulations complete.")


# =============================================================================
# main
# =============================================================================

if __name__ == "__main__":
    print(f"Running with {N_WORKERS} worker processes\n")

    run_block_benchmark(PRESET)
