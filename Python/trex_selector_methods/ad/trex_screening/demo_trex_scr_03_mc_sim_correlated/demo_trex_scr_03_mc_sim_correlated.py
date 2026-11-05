# ==============================================================================
# demo_trex_scr_03_mc_sim_correlated.py
# ==============================================================================
#
# Screen-TRex Selector, demo 03: correlated designs & dependency-aware screening.
#
# Ordinary screening thresholds each variable's voting proportion Phi_j > 0.5
# independently. Under correlated predictors, redundant correlated variables
# *split* the voting evidence, so Phi drops below 0.5 for all of them and power
# collapses. The dependency-aware (DA) variants pre-group strongly correlated
# variables before voting so the evidence is not split:
#   * TREX_DA_AR1        - for AR(1) (banded) correlation,
#   * TREX_DA_EQUI       - for equicorrelation (single shared factor),
#   * TREX_DA_BLOCK_EQUI - for block-equicorrelation (per-block factors).
#
# Mirrors:
#   cpp/trex_selector_methods/ad/trex_screening/demo_trex_scr_03_mc_sim_correlated/
#       demo_trex_scr_03_mc_sim_correlated.cpp
#
# Demo content (five Monte Carlo studies, matching the cpp Parts 1-5):
#   Part 1: SNR sweep on AR(1) data (rho=0.5): Ordinary/Bootstrap vs. +DA-AR1.
#   Part 2: SNR sweep on equicorrelated data (rho=0.4): vs. +DA-EQUI.
#   Part 3: rho sweep on AR(1) data (SNR=1): vs. +DA-AR1.
#   Part 4: rho sweep on equicorrelated data (SNR=1): vs. +DA-EQUI.
#   Part 5: rho sweep on block-equicorrelated data (SNR=1, 5 blocks): vs. +DA-BLOCK-EQUI.
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

from trex_scr_common import run_mc_screen, save_and_print_scr_mc  # noqa: E402
from trex_scr_common import resolve_workers  # noqa: E402

_OUT_DIR = os.path.join(_THIS_DIR, "simulation_results", "data")

# Monte Carlo count mirrors the cpp reference (num_MC = 200).
_NUM_MC = 200
_NUM_WORKERS = resolve_workers(os.environ.get("SCR_NUM_WORKERS"))

N, P, P1 = 300, 1000, 10
SNR_GRID = [0.01, 0.1, 0.2, 0.5, 0.6, 1.0, 2.0, 5.0]
RHO_GRID = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def _methods_for(da_method, da_label):
    """Four cpp-style method descriptors: Ordinary/Bootstrap, plain and +DA."""
    return [
        {"name": "Screen-TRex Ordinary: TLARS",              "trex_method": "TREX",     "bootstrap": False},
        {"name": "Screen-TRex Bootstrap-CI: TLARS",          "trex_method": "TREX",     "bootstrap": True},
        {"name": f"Screen-TRex Ordinary+{da_label}: TLARS",     "trex_method": da_method, "bootstrap": False},
        {"name": f"Screen-TRex Bootstrap-CI+{da_label}: TLARS", "trex_method": da_method, "bootstrap": True},
    ]


def run_sweep(part_title, methods, dgp_kind, sweep_name, sweep_values,
              fixed, beta_val, file_stem, screen_n_blocks=None,
              num_MC=_NUM_MC, num_workers=_NUM_WORKERS):
    """Generic MC sweep over SNR or rho for a list of methods (mirrors the cpp parts)."""
    print("\n" + "=" * 70)
    print(part_title)
    print("=" * 70 + "\n")
    method_names = [m["name"] for m in methods]

    def init():
        return {nm: [0.0] * len(sweep_values) for nm in method_names}
    fdr_map, tpr_map, est_fdr_map = init(), init(), init()

    trex_args = {"solver": "TLARS", "use_memory_mapping": False, "K": 20,
                 "tloop_stagnation_stop": None,   # AUTO (C++ nullopt)
                 "tloop_max_stagnant_steps": 5}

    for m in methods:
        print("=" * 70)
        print(f"Method: {m['name']}")
        print("=" * 70)
        screen_args = {"trex_method": m["trex_method"], "bootstrap": m["bootstrap"]}
        if screen_n_blocks is not None:
            screen_args["screen_n_blocks"] = screen_n_blocks
        for vi, v in enumerate(sweep_values):
            snr = v if sweep_name == "SNR" else fixed["snr"]
            rho = v if sweep_name == "rho" else fixed["rho"]
            dgp_args = {"dgp_kind": dgp_kind, "n": N, "p": P, "p1": P1,
                        "snr": snr, "rho": rho, "beta_val": beta_val}
            if dgp_kind == "block_equi":
                dgp_args["dgp_n_blocks"] = fixed["n_blocks"]
            base_seed = 42 + vi * 1000        # mirrors cpp base_seed = 42 + idx * 1000
            res = run_mc_screen(num_MC, f"{m['name']}  {sweep_name}={v:.2f}",
                                dgp_args, trex_args, screen_args,
                                base_seed=base_seed, max_workers=num_workers)
            fdr_map[m["name"]][vi] = res["avg_fdr"]
            tpr_map[m["name"]][vi] = res["avg_tpr"]
            est_fdr_map[m["name"]][vi] = res["avg_est_fdr"]

    save_and_print_scr_mc(num_MC, _OUT_DIR, file_stem, sweep_values,
                          method_names, fdr_map, tpr_map, est_fdr_map,
                          sweep_label=sweep_name)


# ==============================================================================
# Main
# ==============================================================================

def main():
    # Part 1: SNR sweep - AR(1), rho=0.5, beta=1.0
    run_sweep("Part 1: DA-AR1 Screen-TRex Monte Carlo on AR(1) Data",
              _methods_for("TREX_DA_AR1", "DA-AR1"), "ar1", "SNR", SNR_GRID,
              fixed={"rho": 0.5}, beta_val=1.0,
              file_stem="scr_ar1_snr_n300_p1000_rho0.50")

    # Part 2: SNR sweep - equicorrelated, rho=0.4, beta=3.0
    run_sweep("Part 2: DA-EQUI Screen-TRex Monte Carlo on Equicorrelated Data",
              _methods_for("TREX_DA_EQUI", "DA-EQUI"), "equi", "SNR", SNR_GRID,
              fixed={"rho": 0.4}, beta_val=3.0,
              file_stem="scr_equi_snr_n300_p1000_rho0.40")

    # Part 3: rho sweep - AR(1), SNR=1.0, beta=1.0
    run_sweep("Part 3: DA-AR1 Screen-TRex rho Sweep on AR(1) Data",
              _methods_for("TREX_DA_AR1", "DA-AR1"), "ar1", "rho", RHO_GRID,
              fixed={"snr": 1.0}, beta_val=1.0,
              file_stem="scr_ar1_rho_n300_p1000_snr1.00")

    # Part 4: rho sweep - equicorrelated, SNR=1.0, beta=3.0
    run_sweep("Part 4: DA-EQUI Screen-TRex rho Sweep on Equicorrelated Data",
              _methods_for("TREX_DA_EQUI", "DA-EQUI"), "equi", "rho", RHO_GRID,
              fixed={"snr": 1.0}, beta_val=3.0,
              file_stem="scr_equi_rho_n300_p1000_snr1.00")

    # Part 5: rho sweep - block-equicorrelated, SNR=1.0, beta=3.0, 5 blocks
    run_sweep("Part 5: DA-BLOCK-EQUI Screen-TRex rho Sweep on Block-Equicorr Data",
              _methods_for("TREX_DA_BLOCK_EQUI", "DA-BLOCK-EQUI"), "block_equi",
              "rho", RHO_GRID, fixed={"snr": 1.0, "n_blocks": 5}, beta_val=3.0,
              file_stem="scr_block_equi_rho_n300_p1000_blocks5_snr1.00",
              screen_n_blocks=5)

    print("\ndemo_trex_scr_03_mc_sim_correlated.py complete.")


if __name__ == "__main__":
    main()
