# ==============================================================================
# demo_trex_scr_02_mc_sim_screen_trex_mmap.py
# ==============================================================================
#
# Screen-TRex Selector, demo 02: memory-mapped Monte Carlo variant.
#
# Statistically identical to demo 01, but the dummy (D) matrices are backed by
# on-disk files rather than held in RAM: TRexControlParameter.use_memory_mapping
# = True activates the internal D-mmap + solver-serialization pipeline. This is
# the regime Screen-TRex is built for: p so large that a plain in-core T-Rex run
# is impractical. As in the cpp source, the design matrix X still lives in RAM;
# only the dummy matrices move to disk, and the library manages those temporary
# files in its own scratch directories (nothing to clean up in this script).
#
# Mirrors:
#   cpp/trex_selector_methods/ad/trex_screening/demo_trex_scr_02_mc_sim_screen_trex_mmap/
#       demo_trex_scr_02_mc_sim_screen_trex_mmap.cpp
#
# Demo content (single Monte Carlo SNR sweep, memory-mapped):
#   Ordinary vs. Bootstrap-CI Screen-TRex over an SNR grid with the mmap pipeline
#   enabled - equivalence to demo 01 at a lower memory footprint.
#
# The Monte Carlo count matches the cpp reference (num_MC = 200). Support
# indices are 0-based.
# ==============================================================================

import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.dirname(_THIS_DIR)
for _p in (_THIS_DIR, _PARENT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from trex_scr_common import default_scr_methods, run_mc_screen, save_and_print_scr_mc  # noqa: E402
from trex_scr_common import resolve_workers  # noqa: E402

_OUT_DIR = os.path.join(_THIS_DIR, "simulation_results", "data")

# Monte Carlo count mirrors the cpp reference (num_MC = 200).
_NUM_MC = 200
_NUM_WORKERS = resolve_workers(os.environ.get("SCR_NUM_WORKERS"))

N, P, SUPPORT_SIZE = 300, 1000, 10
SNR_GRID = [0.01, 0.1, 0.2, 0.5, 0.6, 1.0, 2.0, 5.0]


# ==============================================================================
# Monte Carlo SNR sweep - Ordinary vs. Bootstrap-CI (memory-mapped)
# ==============================================================================

def demo_screen_trex_mmap_monte_carlo(num_MC=_NUM_MC, num_workers=_NUM_WORKERS):
    print("\n" + "=" * 70)
    print("Demo: Screen-TRex Monte Carlo - Memory-Mapped")
    print("=" * 70)
    print("High-dimensional (p > n)  |  use_memory_mapping = True")
    print(f"n = {N}, p = {P}, s = {SUPPORT_SIZE}, num_MC = {num_MC}\n")

    methods = default_scr_methods()
    method_names = [m["name"] for m in methods]

    def init():
        return {nm: [0.0] * len(SNR_GRID) for nm in method_names}
    fdr_map, tpr_map, est_fdr_map = init(), init(), init()

    trex_args = {"solver": "TLARS", "use_memory_mapping": True, "K": 20,
                 "tloop_stagnation_stop": None,   # AUTO (C++ nullopt)
                 "tloop_max_stagnant_steps": 5}

    for m in methods:
        print("=" * 70)
        print(f"Method: {m['name']}")
        print("=" * 70)
        screen_args = {"trex_method": m["trex_method"], "bootstrap": m["bootstrap"]}
        for si, snr in enumerate(SNR_GRID):
            dgp_args = {"dgp_kind": "iid", "n": N, "p": P,
                        "support_size": SUPPORT_SIZE, "snr": snr, "rnd_coef": False}
            base_seed = 24 + si * 1000        # mirrors cpp base_seed = 24 + snr_idx * 1000
            res = run_mc_screen(num_MC, f"{m['name']}  SNR={snr:.2f}",
                                dgp_args, trex_args, screen_args,
                                base_seed=base_seed, max_workers=num_workers)
            fdr_map[m["name"]][si] = res["avg_fdr"]
            tpr_map[m["name"]][si] = res["avg_tpr"]
            est_fdr_map[m["name"]][si] = res["avg_est_fdr"]

    file_stem = f"scr_screen_trex_mmap_snr_n{N}_p{P}"
    save_and_print_scr_mc(num_MC, _OUT_DIR, file_stem, SNR_GRID,
                          method_names, fdr_map, tpr_map, est_fdr_map)


# ==============================================================================
# Main
# ==============================================================================

def main():
    demo_screen_trex_mmap_monte_carlo()
    print("\ndemo_trex_scr_02_mc_sim_screen_trex_mmap.py complete.")


if __name__ == "__main__":
    main()
