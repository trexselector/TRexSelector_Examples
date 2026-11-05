# ==============================================================================
# demo_trex_scr_06_mc_sim_solvers.py
# ==============================================================================
#
# Screen-TRex Selector, demo 06: solver-backend comparison (TLARS / TAFS / TOMP).
#
# Screening thresholds the dummy-based voting proportion Phi_j, but Phi itself is
# produced by an underlying T-Rex solver. This demo asks whether the choice of
# solver backend changes screening performance:
#   * TLARS    - terminating LARS (the default),
#   * TAFS-0.3 - terminating adaptive forward stepwise (shrinkage step rho_afs = 0.3),
#   * TOMP     - terminating orthogonal matching pursuit,
# each combined with Ordinary and Bootstrap-CI screening. The same i.i.d.
# Gaussian design as demo 01 is used, so the numbers are directly comparable.
#
# Mirrors:
#   cpp/trex_selector_methods/ad/trex_screening/demo_trex_scr_06_mc_sim_solvers/
#       demo_trex_scr_06_mc_sim_solvers.cpp
#
# Demo content (two Monte Carlo SNR sweeps, matching the cpp Demos 01-02):
#   Part 1: three solvers, Ordinary screening only (3 series).
#   Part 2: three solvers x {Ordinary, Bootstrap-CI} (6 series).
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

N, P, SUPPORT_SIZE = 300, 1000, 10
SNR_GRID = [0.01, 0.1, 0.2, 0.5, 0.6, 1.0, 2.0, 5.0]


def _sv(name, solver, rho_afs=None, bootstrap=False):
    """A solver x screening-variant descriptor (mirrors cpp SolverVariant)."""
    return {"name": name, "solver": solver, "rho_afs": rho_afs, "bootstrap": bootstrap}


def _ordinary_solver_variants():
    """Three solvers, Ordinary screening only (mirrors ordinary_solver_variants())."""
    return [
        _sv("Screen-TRex Ordinary: TLARS",    "TLARS"),
        _sv("Screen-TRex Ordinary: TAFS-0.3", "TAFS", rho_afs=0.3),
        _sv("Screen-TRex Ordinary: TOMP",     "TOMP"),
    ]


def _all_solver_variants():
    """Six combined variants - three solvers x {Ordinary, Bootstrap-CI}."""
    variants = []
    for bs in (False, True):
        prefix = "Screen-TRex Bootstrap-CI: " if bs else "Screen-TRex Ordinary: "
        variants.append(_sv(prefix + "TLARS",    "TLARS",           bootstrap=bs))
        variants.append(_sv(prefix + "TAFS-0.3", "TAFS", rho_afs=0.3, bootstrap=bs))
        variants.append(_sv(prefix + "TOMP",     "TOMP",            bootstrap=bs))
    return variants


def run_solver_sweep(title, variants, file_stem, num_MC=_NUM_MC, num_workers=_NUM_WORKERS):
    """MC sweep over a list of solver x method variants (SNR grid)."""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)
    print("High-dimensional (p > n)")
    print(f"n = {N}, p = {P}, s = {SUPPORT_SIZE}, num_MC = {num_MC}\n")

    method_names = [v["name"] for v in variants]

    def init():
        return {nm: [0.0] * len(SNR_GRID) for nm in method_names}
    fdr_map, tpr_map, est_fdr_map = init(), init(), init()

    for v in variants:
        print("=" * 70)
        print(f"Variant: {v['name']}")
        print("=" * 70)
        # make_trex_ctrl_for(): stagnation stop enabled for every backend.
        trex_args = {"solver": v["solver"], "use_memory_mapping": False, "K": 20,
                     "tloop_stagnation_stop": True, "tloop_max_stagnant_steps": 5,
                     "rho_afs": v["rho_afs"]}
        screen_args = {"trex_method": "TREX", "bootstrap": v["bootstrap"]}
        for si, snr in enumerate(SNR_GRID):
            dgp_args = {"dgp_kind": "iid", "n": N, "p": P,
                        "support_size": SUPPORT_SIZE, "snr": snr, "rnd_coef": False}
            base_seed = 24 + si * 1000        # mirrors cpp base_seed = 24 + snr_idx * 1000
            res = run_mc_screen(num_MC, f"{v['name']}  SNR={snr:.2f}",
                                dgp_args, trex_args, screen_args,
                                base_seed=base_seed, max_workers=num_workers)
            fdr_map[v["name"]][si] = res["avg_fdr"]
            tpr_map[v["name"]][si] = res["avg_tpr"]
            est_fdr_map[v["name"]][si] = res["avg_est_fdr"]

    save_and_print_scr_mc(num_MC, _OUT_DIR, file_stem, SNR_GRID,
                          method_names, fdr_map, tpr_map, est_fdr_map)


# ==============================================================================
# Main
# ==============================================================================

def main():
    # Part 1: three solvers, ordinary Screen-TRex
    run_solver_sweep(
        "Demo: Screen-TRex MC - Solver Comparison (TLARS / TAFS-0.3 / TOMP)",
        _ordinary_solver_variants(),
        f"scr_solvers_snr_n{N}_p{P}_s{SUPPORT_SIZE}")

    # Part 2: three solvers x {Ordinary, Bootstrap-CI} (6 series)
    run_solver_sweep(
        "Demo: Screen-TRex MC - Solver x Method Comparison (6 series)",
        _all_solver_variants(),
        f"scr_solver_method_snr_n{N}_p{P}_s{SUPPORT_SIZE}")

    print("\ndemo_trex_scr_06_mc_sim_solvers.py complete.")


if __name__ == "__main__":
    main()
