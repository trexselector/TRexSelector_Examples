# ==============================================================================
# demo_trex_scr_01_mc_sim_screen_trex.py
# ==============================================================================
#
# Screen-TRex Selector, demo 01: Monte Carlo baseline on an i.i.d. Gaussian design.
#
# Screen-TRex extends the classical T-Rex Selector with a fast variable-
# *screening* step for ultra-high-dimensional data. It thresholds the dummy-based
# voting proportion Phi_j that T-Rex accumulates internally:
#   * Ordinary    rule: select {j : Phi_j > 0.5}  (a majority vote).
#   * Bootstrap-CI rule: build a bootstrap confidence band around the estimated
#     FDR curve (R_boot replicates) and pick the threshold from that band.
# Unlike the classical T-Rex selector, screening has *no target-FDR parameter*;
# it reports its own estimated FDR alongside the selection.
#
# Mirrors:
#   cpp/trex_selector_methods/ad/trex_screening/demo_trex_scr_01_mc_sim_screen_trex/
#       demo_trex_scr_01_mc_sim_screen_trex.cpp
#
# Demo content (single Monte Carlo SNR sweep, in-memory):
#   Ordinary vs. Bootstrap-CI Screen-TRex over an SNR grid, reporting realized
#   FDR, TPR, and the procedure's self-reported estimated FDR.
#
# The Monte Carlo count matches the cpp reference (num_MC = 200). Results are
# statistically comparable, not bit-identical (different RNG streams).
# Support indices are 0-based.
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
# Monte Carlo SNR sweep - Ordinary vs. Bootstrap-CI
# ==============================================================================

def demo_screen_trex_monte_carlo(num_MC=_NUM_MC, num_workers=_NUM_WORKERS):
    print("\n" + "=" * 70)
    print("Demo: Screen-TRex Monte Carlo Comparison")
    print("=" * 70)
    print("High-dimensional (p > n)")
    print(f"n = {N}, p = {P}, s = {SUPPORT_SIZE}, num_MC = {num_MC}\n")

    methods = default_scr_methods()
    method_names = [m["name"] for m in methods]

    def init():
        return {nm: [0.0] * len(SNR_GRID) for nm in method_names}
    fdr_map, tpr_map, est_fdr_map = init(), init(), init()

    trex_args = {"solver": "TLARS", "use_memory_mapping": False, "K": 20,
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

    file_stem = f"scr_screen_trex_snr_n{N}_p{P}"
    save_and_print_scr_mc(num_MC, _OUT_DIR, file_stem, SNR_GRID,
                          method_names, fdr_map, tpr_map, est_fdr_map)


# ==============================================================================
# Main
# ==============================================================================

def main():
    demo_screen_trex_monte_carlo()
    print("\ndemo_trex_scr_01_mc_sim_screen_trex.py complete.")


if __name__ == "__main__":
    main()
