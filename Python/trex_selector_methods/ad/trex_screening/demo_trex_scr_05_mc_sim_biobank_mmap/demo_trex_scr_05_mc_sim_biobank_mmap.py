# ==============================================================================
# demo_trex_scr_05_mc_sim_biobank_mmap.py
# ==============================================================================
#
# Biobank Screen-TRex Selector ("Algorithm 1"), memory-mapped Monte Carlo variant.
#
# Statistically identical to demo 04, but the internal dummy matrices are backed
# by memory-mapped files (use_memory_mapping = True), the storage mode intended
# for biobank-scale problems where holding the dummy matrices in RAM is
# infeasible. Everything else about Algorithm 1 - the FDR-window routing to
# Ordinary, Bootstrap-CI or the T-Rex fallback and its per-method Usage % - is
# unchanged. As in the cpp source, the design matrix X still lives in RAM; only
# the dummy matrices move to disk, and the library manages those temporary files
# in its own scratch directories (nothing to clean up in this script).
#
# Mirrors:
#   cpp/trex_selector_methods/ad/trex_screening/demo_trex_scr_05_mc_sim_biobank_mmap/
#       demo_trex_scr_05_mc_sim_biobank_mmap.cpp
#
# Demo content (two Monte Carlo SNR sweeps, matching the cpp Parts 1-2):
#   Part 1: single phenotype (s=10): per-method Usage %, FDR, TPR, Est. FDR.
#   Part 2: q=5 phenotypes sharing one X (s=5): same per-method metrics.
#
# The Monte Carlo counts and R_boot match the cpp reference (num_MC = 200,
# num_MC_multi = 200, R_boot = 1000). Support indices are 0-based.
# ==============================================================================

import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.dirname(_THIS_DIR)
for _p in (_THIS_DIR, _PARENT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from trex_scr_bb_common import (BIOBANK_METHODS, BIOBANK_DISPLAY,  # noqa: E402
                                _biobank_single_worker, _biobank_multi_worker,
                                run_biobank_snr_cell,
                                save_and_print_biobank_mc)
from trex_scr_common import resolve_workers  # noqa: E402

_OUT_DIR = os.path.join(_THIS_DIR, "simulation_results", "data")

# Monte Carlo counts mirror the cpp reference (num_MC = 200, num_MC_multi = 200).
_NUM_MC       = 200
_NUM_MC_MULTI = 200
_NUM_WORKERS  = resolve_workers(os.environ.get("SCR_NUM_WORKERS"))

N, P = 300, 1000
SNR_GRID = [0.1, 0.5, 1.0, 2.0, 5.0]
TARGET_FDR_TREX = 0.10
R_BOOT = 1000   # mirrors the cpp reference (R_boot = 1000)
USE_MMAP = True              # dummy (D) matrices backed by on-disk files


def _bb_args(**extra):
    """Flat biobank-control dict shared by the workers."""
    d = {"use_mmap": USE_MMAP, "K": 20, "R_boot": R_BOOT,
         "lower_bound_FDR": 0.05, "upper_bound_FDR": 0.15,
         "target_FDR_trex": TARGET_FDR_TREX}
    d.update(extra)
    return d


def _empty_maps():
    return {nm: [0.0] * len(SNR_GRID) for nm in BIOBANK_DISPLAY}


def _store(agg, maps, si):
    """Copy one SNR level's aggregate into the display-keyed maps."""
    for canon, disp in zip(BIOBANK_METHODS, BIOBANK_DISPLAY):
        maps["fdp"][disp][si]   = agg[canon]["fdp"]
        maps["tpp"][disp][si]   = agg[canon]["tpp"]
        maps["est"][disp][si]   = agg[canon]["est"]
        maps["usage"][disp][si] = agg[canon]["usage"]


# ==============================================================================
# Part 1 - MC SNR sweep, single phenotype (memory-mapped)
# ==============================================================================

def part_1(num_MC=_NUM_MC, num_workers=_NUM_WORKERS):
    print("\n" + "=" * 70)
    print("Part 1: Biobank Screen-TRex - MC SNR Sweep (Memory-Mapped)")
    print("=" * 70)
    support_size = 10
    print(f"n = {N}, p = {P}, s = {support_size}, num_MC = {num_MC}\n")

    maps = {k: _empty_maps() for k in ("fdp", "tpp", "est", "usage")}
    for si, snr in enumerate(SNR_GRID):
        args_list = [
            _bb_args(n=N, p=P, support_size=support_size, snr=snr,
                     trial_seed=1000 * si + mc)          # mirrors cpp seed = 1000*snr_idx + mc
            for mc in range(num_MC)
        ]
        _store(run_biobank_snr_cell(_biobank_single_worker, args_list, num_workers, TARGET_FDR_TREX),
               maps, si)
        print(f"  SNR {snr:.2f} - completed {num_MC} runs.")

    save_and_print_biobank_mc(num_MC, _OUT_DIR,
                              f"scr_biobank_mmap_snr_n{N}_p{P}_s{support_size}",
                              SNR_GRID, BIOBANK_DISPLAY,
                              maps["fdp"], maps["tpp"], maps["est"], maps["usage"])


# ==============================================================================
# Part 2 - MC SNR sweep, multiple phenotypes (memory-mapped)
# ==============================================================================

def part_2(num_MC=_NUM_MC_MULTI, num_workers=_NUM_WORKERS):
    print("\n" + "=" * 70)
    print("Part 2: Biobank Screen-TRex - MC SNR Sweep, Multiple Phenotypes (Memory-Mapped)")
    print("=" * 70)
    q, support_size = 5, 5
    print(f"n = {N}, p = {P}, q = {q}, s = {support_size}, num_MC = {num_MC}\n")

    maps = {k: _empty_maps() for k in ("fdp", "tpp", "est", "usage")}
    for si, snr in enumerate(SNR_GRID):
        args_list = [
            _bb_args(n=N, p=P, q=q, support_size=support_size, snr=snr,
                     trial_seed=5000 * si + mc + 1,       # x_seed label (shared X per trial)
                     sup_seed_base=770000 + q * (num_MC * si + mc))
            for mc in range(num_MC)
        ]
        _store(run_biobank_snr_cell(_biobank_multi_worker, args_list, num_workers, TARGET_FDR_TREX),
               maps, si)
        print(f"  SNR {snr:.2f} - completed {num_MC} runs ({q} phenotypes each).")

    save_and_print_biobank_mc(num_MC, _OUT_DIR,
                              f"scr_biobank_mmap_multi_n{N}_p{P}_q{q}_s{support_size}",
                              SNR_GRID, BIOBANK_DISPLAY,
                              maps["fdp"], maps["tpp"], maps["est"], maps["usage"])


# ==============================================================================
# Main
# ==============================================================================

def main():
    part_1()
    part_2()
    print("\ndemo_trex_scr_05_mc_sim_biobank_mmap.py complete.")


if __name__ == "__main__":
    main()
