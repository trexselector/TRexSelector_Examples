# ==============================================================================
# demo_trex_06_mmap.py
# ==============================================================================
#
# Classical T-Rex Selector demo — memory-mapped workflows.
#
# Mirrors:
# cpp/trex_selector_methods/ad/trex_core/demo_trex_06_mmap.cpp.
# and
# R/trex_selector_methods/ad/trex_core/demo_trex_06_mmap.R
#
# Demo content:
#
#  Part A: In-memory X + use_memory_mapping=True.
#          Setting use_memory_mapping=True activates both:
#            (1) Memory-mapped storage for internal dummy matrices D.
#            (2) Solver LARS-path serialization to disk between T-loop iterations.
#          X itself stays fully in RAM.
#          Exact mirror of C++ Demo A.
#
#  Part B: Fully memory-mapped pipeline.
#          X is written to a temporary binary file via numpy_to_memmap().
#          X_mmap.to_numpy() (a zero-copy OS-paged view) is passed to
#          TRexSelector; use_memory_mapping=True adds D-mmap + solver
#          serialization on top.
#          Mirrors C++ Demo B (SyntheticDataMapped).
#
# Seed note:
#   Both parts use seed=58 for DGP and TRexSelector (single-run reproducibility,
#   mirrors R Part A/B).  This is intentional for a single-run demo; MC demos
#   use seed=-1 (hardware entropy) for valid FDR estimates.
#
# Output (mirrors C++): per configuration one .txt (dual console + file, with
# the sparse Phi_prime/Phi block) and one per-variable selection .csv, both in
# simulation_results/data/.
#
# ==============================================================================

import sys
import os

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.dirname(_THIS_DIR)
for _p in (_THIS_DIR, _PARENT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import tempfile  # noqa: E402

import numpy as np  # noqa: E402

import trexselector as trex  # noqa: E402
from trexselector.utils import numpy_to_memmap  # noqa: E402

from dgp_gauss_snr import dgp_gauss_snr  # noqa: E402
from trex_helpers import make_dual_out, print_selection_results, save_selection_csv  # noqa: E402
from trex_sim_common import _make_trex_ctrl_from_dict  # noqa: E402

# ==============================================================================
# Shared demo configuration
# ==============================================================================
# Mirrors make_mmap_demo_config() in C++ and DEMO_CFG in R.
#
# Fixed true support (0-based) — note: different from demo_trex_01.
# C++ (0-based): {27, 149, 398, 420, 4}
# R   (1-based): {28, 150, 399, 421, 5}
_TRUE_SUPPORT = np.array([27, 149, 398, 420, 4], dtype=np.intp)
_TRUE_COEFS   = np.ones(5, dtype=np.float64)     # rnd_coef = False
_SNR          = 1.0
_TFDR         = 0.1
_SEED         = 58

_OUT_DIR = os.path.join(_THIS_DIR, "simulation_results", "data")

# Shared control: mirrors make_mmap_trex_ctrl() in C++ (demo_trex_06_mmap.cpp).
# Every option explicit, values from the C++ reference — no reliance on
# binding/helper defaults.
_MMAP_CTRL_DICT = dict(
    solver_name               = "TLARS",
    K                         = 20,
    max_dummy_multiplier      = 10,
    use_max_T_stop            = True,
    opt_threshold             = 0.75,
    lloop_strategy            = "HCONCAT",
    tloop_stagnation_stop     = True,
    tloop_max_stagnant_steps  = 5,
    use_memory_mapping        = True,   # activates D-mmap + solver serialization
    lambda2                   = 0.1,
    rho_afs                   = 0.0,   # non-TAFS solver; rho_afs is unused
    ncgmp_variant             = 0,
    parallel_rnd_experiments  = False,
)


# ==============================================================================
# Shared print helper (mirrors C++ print_results + save_selection_csv,
# written as dual console + file output like the C++ print_dual lambda)
# ==============================================================================

def _print_and_save(stem, dat, sel, tFDR, out):
    true_support = dat["true_support"]
    selected     = list(sel.selected_indices)

    print_selection_results(selected, true_support.tolist(), sel, out)
    out(f"\nT_stop = {sel.T_stop},  L = {sel.L}  (target tFDR <= {tFDR:.2f})\n\n")

    save_selection_csv(
        _OUT_DIR, stem,
        p=dat["p"],
        phi_prime=sel.phi_prime,
        selected_0based=selected,
        true_support_0based=true_support.tolist(),
    )


# ==============================================================================
# Part A: In-memory X + use_memory_mapping=True
# ==============================================================================

def part_a_d_mmap_solver_serial():
    """
    Mirrors C++ demo_TRexSelector_d_mmap_solver_serial() and R trx_02_part_a().
    """
    def run_single(high_dim):
        n = 1000 if high_dim else 5000
        p = 5000 if high_dim else 1000
        dim_label = "High-dimensional (p > n)" if high_dim else "Low-dimensional (n > p)"

        stem = f"d02_trex_mmap_demo_a_n{n}_p{p}"
        os.makedirs(_OUT_DIR, exist_ok=True)
        txt_path = os.path.join(_OUT_DIR, f"{stem}.txt")

        with open(txt_path, "w") as fh:
            out = make_dual_out(fh)

            out(f"\n{'=' * 70}\n")
            out("Part A: In-Memory X  +  use_memory_mapping=True\n")
            out("  Activates D-mmap and solver serialization inside TRexSelector.\n")
            out(f"  {dim_label}  (n = {n}, p = {p})\n\n")

            out("Generating synthetic data (in-memory) ...\n")
            dat = dgp_gauss_snr(n, p, _TRUE_SUPPORT, coefs=_TRUE_COEFS,
                                snr=_SNR, seed=_SEED)

            ctrl = _make_trex_ctrl_from_dict(_MMAP_CTRL_DICT)

            out("Running T-Rex Selector ...\n\n")
            sel = trex.TRexSelector(
                np.asfortranarray(dat["X"]),
                dat["y"],
                tFDR=_TFDR,
                trex_control=ctrl,
                seed=_SEED,
                verbose=True,
            )
            sel.select()

            _print_and_save(stem, dat, sel, _TFDR, out)

        print(f"[Info] Results saved to: {txt_path}")

    run_single(high_dim=False)
    run_single(high_dim=True)


# ==============================================================================
# Part B: Memory-mapped X + use_memory_mapping=True
# ==============================================================================

def part_b_full_mmap():
    """
    Mirrors C++ demo_TRexSelector_full_mmap() and R trx_02_part_b().

    X is written to a temporary binary file; X_mmap.to_numpy() (a zero-copy
    Fortran-contiguous float64 view backed by the mmap buffer) is passed to
    TRexSelector.  The temp file is removed in a finally block — exception-safe,
    mirrors C++ RAII MmapFileGuard and R execute_with_temp_mmap / on.exit().
    """
    def run_single(high_dim):
        n = 1000 if high_dim else 5000
        p = 5000 if high_dim else 1000
        dim_label = "High-dimensional (p > n)" if high_dim else "Low-dimensional (n > p)"

        stem = f"d02_trex_mmap_demo_b_n{n}_p{p}"
        os.makedirs(_OUT_DIR, exist_ok=True)
        txt_path = os.path.join(_OUT_DIR, f"{stem}.txt")

        with open(txt_path, "w") as fh:
            out = make_dual_out(fh)

            out(f"\n{'=' * 70}\n")
            out("Part B: Memory-Mapped X  +  use_memory_mapping=True\n")
            out("  Full mmap pipeline: X mmap + D mmap + solver serialization.\n")
            out(f"  {dim_label}  (n = {n}, p = {p})\n\n")

            out("Generating synthetic data ...\n")
            dat = dgp_gauss_snr(n, p, _TRUE_SUPPORT, coefs=_TRUE_COEFS,
                                snr=_SNR, seed=_SEED)

            ctrl = _make_trex_ctrl_from_dict(_MMAP_CTRL_DICT)

            # Create temporary backing file.  os.close(fd) releases the OS file
            # descriptor; numpy_to_memmap re-opens the path for writing.
            fd, mmap_path = tempfile.mkstemp(suffix=".bin")
            os.close(fd)

            try:
                # The temp path stays out of the output: it is machine-specific.
                out("Converting X to memory-mapped file (mmap_path) ...\n")
                X_mmap = numpy_to_memmap(mmap_path, dat["X"])

                out("Running T-Rex Selector ...\n\n")
                # X_mmap.to_numpy() → Fortran-contiguous float64 mmap view.
                # TRexSelector.__init__ calls np.asfortranarray(X) which returns
                # the view unchanged (already F-contiguous float64, zero copy).
                sel = trex.TRexSelector(
                    X_mmap.to_numpy(),
                    dat["y"],
                    tFDR=_TFDR,
                    trex_control=ctrl,
                    seed=_SEED,
                    verbose=True,
                )
                sel.select()
            finally:
                # Exception-safe cleanup.  On CPython the MemoryMappedMatrix is
                # reference-counted; it goes out of scope (and releases the mmap
                # mapping) before this finally block runs.
                try:
                    os.unlink(mmap_path)
                    print(f"[Info] Temp mmap file removed: {mmap_path}")
                except OSError:
                    pass

            _print_and_save(stem, dat, sel, _TFDR, out)

        print(f"[Info] Results saved to: {txt_path}")

    run_single(high_dim=False)
    run_single(high_dim=True)


# ==============================================================================
# Main
# ==============================================================================

if __name__ == "__main__":
    part_a_d_mmap_solver_serial()
    part_b_full_mmap()
    print("\ndemo_trex_06_mmap.py complete.")
