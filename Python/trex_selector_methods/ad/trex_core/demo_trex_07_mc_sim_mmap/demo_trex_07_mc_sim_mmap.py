# ==============================================================================
# demo_trex_07_mc_sim_mmap.py
# ==============================================================================
#
# Classical T-Rex selector demo — Monte Carlo simulations with memory-mapped
# workflows.
#
# Mirrors:
# cpp/trex_selector_methods/ad/trex_core/demo_trex_07_mc_sim_mmap.cpp
# and
# R/trex_selector_methods/ad/trex_core/demo_trex_07_mc_sim_mmap.R
#
# Demo content:
#
#  Part C: MC — In-memory X + use_memory_mapping=True.
#          Activates D-mmap and solver serialization inside TRexSelector.
#          Purpose: verify that the D-mmap + solver-serialization pipeline
#          produces stable, reproducible FDR/TPR statistics over many runs.
#          Uses run_mc_trex() (existing in-memory worker).
#
#  Part D: MC — Fully memory-mapped X + use_memory_mapping=True.
#          Each MC trial writes X to a unique temporary binary file, passes
#          X_mmap.to_numpy() to TRexSelector, and deletes the file in a
#          finally block — exception-safe RAII equivalent.
#          Uses run_mc_trex_full_mmap() (new full-mmap worker).
#
# Both parts should produce statistically equivalent FDR/TPR results since the
# data and algorithm are identical — the equivalence is the verification goal.
#
# Shared configuration:
#   n=300, p=1000, s=10  (high-dim)
#   TLARS only
#   Fixed support drawn once with seed=24 (shared across all MC trials)
#   SNR sweep: {0.1, 0.5, 1.0, 2.0, 5.0}
#   tloop_max_stagnant_steps=7,  opt_threshold=0.75
#   num_MC=200.
#
# Seed spaces (no clashes):
#   DGP (X, noise)   : trial_seed = base_seed + mc - 1
#   Fixed support     : make_support_random(seed=24) → internal seed+500_000
#   TRexSelector      : seed=-1 (hardware entropy, required for valid MC FDR)
#   Temp mmap paths   : tempfile.mkstemp() — OS-guaranteed unique per worker
#
# ==============================================================================

import sys
import os

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.dirname(_THIS_DIR)
for _p in (_THIS_DIR, _PARENT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from support_generators import make_support_random # noqa: E402
from trex_sim_common import ( # noqa: E402
    run_mc_trex,
    run_mc_trex_full_mmap,
    save_mc_results,
)
from trex_sim_common import resolve_workers  # noqa: E402

# ==============================================================================
# Global parameters
# ==============================================================================

_OUT_DIR     = os.path.join(_THIS_DIR, "simulation_results", "data")
_NUM_WORKERS = resolve_workers()
_NUM_MC      = 200

# TLARS solver descriptor (only solver used in mmap demos)
_TLARS_SOLVER = dict(
    solver_name   = "TLARS",
    name          = "TLARS",
    lambda2       = 0.1,
    rho_afs       = 0.0,   # non-TAFS solver; rho_afs is unused
    ncgmp_variant = 0,
)

# Shared base control for both Part C and Part D.
# use_memory_mapping=True: enables D-mmap + solver serialization (Part C's
# primary mode, and the "internal mmap layer" on top of X-mmap in Part D).
_MMAP_BASE_CTRL = dict(
    K                        = 20,
    max_dummy_multiplier     = 10,
    use_max_T_stop           = True,
    lloop_strategy           = "HCONCAT",
    tloop_stagnation_stop    = True,
    tloop_max_stagnant_steps = 7,
    opt_threshold            = 0.75,
    use_memory_mapping       = True,    # D-mmap + solver serialization
    parallel_rnd_experiments = False,   # mirrors R parallel_rnd_experiments = FALSE
)


# ==============================================================================
# Part C: MC — In-memory X + use_memory_mapping=True
# ==============================================================================

def trx_03_part_c(num_MC=_NUM_MC, num_workers=_NUM_WORKERS, high_dim=True):
    """
    Mirrors C++ demo_TRexSelector_d_mmap_MonteCarlo() and R trx_03_part_c().

    Uses run_mc_trex() (existing in-memory worker) with use_memory_mapping=True
    in the control dict — the C++ backend memory-maps the dummy matrices D and
    serializes solver checkpoints, but X stays in RAM.
    """
    n = 300  if high_dim else 1000
    p = 1000 if high_dim else 300
    s = 10
    tFDR       = 0.1
    snr_values = [0.1, 0.5, 1.0, 2.0, 5.0]

    dim_label = "High-dimensional (p > n)" if high_dim else "Low-dimensional (n > p)"
    print(f"\n{'=' * 70}")
    print("Part C: MC \u2014 In-Memory X + use_memory_mapping=True")
    print("  Verifies D-mmap + solver-serialization pipeline over many runs.")
    print(f"  {dim_label}  (n = {n}, p = {p},  num_MC = {num_MC})\n")

    # Fixed support shared across all MC runs — mirrors C++ rng(24) and R seed=24
    true_support = make_support_random(s, p, seed=24)
    print(f"True support (cardinality {s}): {{{', '.join(str(x) for x in true_support)}}}\n")

    n_snr     = len(snr_values)
    fdr_vec   = np.zeros(n_snr)
    tpr_vec   = np.zeros(n_snr)
    avg_L_vec = np.zeros(n_snr)
    avg_T_vec = np.zeros(n_snr)

    for snr_idx, snr in enumerate(snr_values):
        base_seed = 24 + snr_idx * 1000
        lbl = f"SNR={snr:.2f} [TLARS/mmap-D]"
        result = run_mc_trex(
            n=n, p=p, s=s, snr=snr,
            solver_desc=_TLARS_SOLVER,
            base_ctrl_dict=_MMAP_BASE_CTRL,
            tFDR=tFDR,
            base_seed=base_seed,
            num_MC=num_MC,
            max_workers=num_workers,
            fixed_support=true_support,
            track_L_T=True,
            label=lbl,
        )
        fdr_vec[snr_idx]   = result["mean_FDR"]
        tpr_vec[snr_idx]   = result["mean_TPR"]
        avg_L_vec[snr_idx] = result["mean_L"]
        avg_T_vec[snr_idx] = result["mean_T"]

    # Wrap into 1-row matrices to reuse save_mc_results()
    stagnant_sw = _MMAP_BASE_CTRL["tloop_max_stagnant_steps"]
    stem = f"d03_trex_mmap_demo_c_n{n}_p{p}_sw{stagnant_sw}"
    save_mc_results(
        _OUT_DIR, stem, num_MC,
        row_names=["TLARS"],
        snr_values=snr_values,
        fdr_mat=fdr_vec.reshape(1, -1),
        tpr_mat=tpr_vec.reshape(1, -1),
        avg_L_mat=avg_L_vec.reshape(1, -1),
        avg_T_mat=avg_T_vec.reshape(1, -1),
    )
    print()


# ==============================================================================
# Part D: MC — Fully memory-mapped X + use_memory_mapping=True
# ==============================================================================

def trx_03_part_d(num_MC=_NUM_MC, num_workers=_NUM_WORKERS, high_dim=True):
    """
    Mirrors C++ demo_TRexSelector_full_mmap_MonteCarlo() and R trx_03_part_d().

    Uses run_mc_trex_full_mmap() — each worker writes X to a unique per-trial
    temporary binary file via numpy_to_memmap(), passes X_mmap.to_numpy() to
    TRexSelector, and deletes the temp file in a finally block.

    use_memory_mapping=True in the control dict additionally activates D-mmap
    and solver serialization inside TRexSelector on top of the X-mmap layer.
    This is the full mmap pipeline: X mmap + D mmap + solver serialization.
    """
    n = 300  if high_dim else 1000
    p = 1000 if high_dim else 300
    s = 10
    tFDR       = 0.1
    snr_values = [0.1, 0.5, 1.0, 2.0, 5.0]

    dim_label = "High-dimensional (p > n)" if high_dim else "Low-dimensional (n > p)"
    print(f"\n{'=' * 70}")
    print("Part D: MC \u2014 Fully Memory-Mapped X + use_memory_mapping=True")
    print("  Full mmap pipeline: X mmap + D mmap + solver serialization.")
    print(f"  {dim_label}  (n = {n}, p = {p},  num_MC = {num_MC})\n")

    # Fixed support — same as Part C, enabling direct comparison
    true_support = make_support_random(s, p, seed=24)
    print(f"True support (cardinality {s}): {{{', '.join(str(x) for x in true_support)}}}\n")

    n_snr     = len(snr_values)
    fdr_vec   = np.zeros(n_snr)
    tpr_vec   = np.zeros(n_snr)
    avg_L_vec = np.zeros(n_snr)
    avg_T_vec = np.zeros(n_snr)

    for snr_idx, snr in enumerate(snr_values):
        base_seed = 24 + snr_idx * 1000
        lbl = f"SNR={snr:.2f} [TLARS/mmap-XD]"
        result = run_mc_trex_full_mmap(
            n=n, p=p, s=s, snr=snr,
            solver_desc=_TLARS_SOLVER,
            base_ctrl_dict=_MMAP_BASE_CTRL,
            tFDR=tFDR,
            base_seed=base_seed,
            num_MC=num_MC,
            max_workers=num_workers,
            fixed_support=true_support,
            track_L_T=True,
            label=lbl,
        )
        fdr_vec[snr_idx]   = result["mean_FDR"]
        tpr_vec[snr_idx]   = result["mean_TPR"]
        avg_L_vec[snr_idx] = result["mean_L"]
        avg_T_vec[snr_idx] = result["mean_T"]

    stagnant_sw = _MMAP_BASE_CTRL["tloop_max_stagnant_steps"]
    stem = f"d03_trex_mmap_demo_d_n{n}_p{p}_sw{stagnant_sw}"
    save_mc_results(
        _OUT_DIR, stem, num_MC,
        row_names=["TLARS"],
        snr_values=snr_values,
        fdr_mat=fdr_vec.reshape(1, -1),
        tpr_mat=tpr_vec.reshape(1, -1),
        avg_L_mat=avg_L_vec.reshape(1, -1),
        avg_T_mat=avg_T_vec.reshape(1, -1),
    )
    print()


# ==============================================================================
# Main
# ==============================================================================

if __name__ == "__main__":
    trx_03_part_c()
    trx_03_part_d()
    print("\ndemo_trex_07_mc_sim_mmap.py complete.")
