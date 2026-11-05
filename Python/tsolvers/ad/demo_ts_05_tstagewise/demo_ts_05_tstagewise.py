# ==============================================================================
# demo_ts_05_tstagewise.py
# ==============================================================================
#
# Demonstration of the T-Stagewise (Terminating Forward Stagewise Regression) solver.
#
# Mirrors cpp/tsolvers/ad/demo_ts_05_tstagewise/demo_ts_05_tstagewise.cpp.
#
# Parts:
#   Demo 1: Basic T-Stagewise with early stopping (internal normalization),
#           high-dimensional (p > n) and low-dimensional (n > p) regimes.
#   Demo 2: External L2 normalization (LpNormScaler) + centered y,
#           solver run with normalize=False, intercept=False.
#   Demo 3: Serialization & warm start — save a partial path, reload,
#           continue, and compare against a reference run.
#   Demo 4: Memory-mapped X/D/y via numpy_to_memmap.
#           (cpp generates p = 500,000 on disk; scaled to p = 5,000 here.)
#
# ==============================================================================

import os
import sys

import numpy as np

from trexselector.ml_methods import LpNormScaler, NormType
from trexselector.tsolvers.ad import SolverOptions
from trexselector.tsolvers.ad.lars_based import TSTAGEWISE_Solver
from trexselector.utils import (
    compute_fdp,
    compute_tpp,
    get_max_threads,
    get_num_threads,
    numpy_to_memmap,
    set_num_threads,
)

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.dirname(os.path.dirname(_THIS_DIR))  # tsolvers/, the shared helpers
for _p in (_THIS_DIR, _PARENT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ts_demo_utils import (
    PartSelection,
    PathRecorder,
    display_path,
    gen_synthetic_data,
    output_dir,
    print_demo_config,
    print_section_header,
    print_selection,
    print_selection_quality,
)

# ==============================================================================
# Demo 1: Basic T-Stagewise with Early Stopping
# ==============================================================================

def demo_early_stopping(high_dim, rnd_coef, T_stop):
    print_section_header("Demo 1: Basic T-Stagewise with Early Stopping")

    n = 1000 if high_dim else 5000
    p = 5000 if high_dim else 1000
    num_dummies = 10 * p

    true_support = [27, 149, 398, 420, 4]
    true_coefs = [-0.4, -0.2, -0.8, 1.1, 2.5] if rnd_coef else [1.0] * 5
    snr = 1.0

    print("High-dimensional (p > n)" if high_dim else "Low-dimensional (n > p)")
    print_demo_config(n, p, num_dummies, T_stop, true_support, true_coefs, snr)

    X, D, y = gen_synthetic_data(n, p, num_dummies, true_support, true_coefs,
                                 snr, seed=42)

    solver = TSTAGEWISE_Solver(
        X, D, y,
        SolverOptions(normalize=True, intercept=True, verbose=True))
    solver.executeStep(T_stop, early_stop=True)

    print_selection(solver, true_support)
    print_selection_quality(solver, true_support)
    print("\nT-Stagewise Diagnostics:")
    print(f"  Removals: {solver.getNumRemovals()}")
    print(f"  Cycling ratio: {solver.getCyclingRatio():.4f}")
    print("\n")


# ==============================================================================
# Demo 2: External Normalization (LpNormScaler)
# ==============================================================================

def demo_with_external_normalizer(high_dim, rnd_coef, T_stop):
    print_section_header("=== Demo 2: T-Stagewise with External Normalization ===")

    n = 1000 if high_dim else 5000
    p = 5000 if high_dim else 1000
    num_dummies = 10 * p

    true_support = [4, 27, 149, 398, 420]
    true_coefs = [-0.4, -0.2, -0.8, 1.1, 2.5] if rnd_coef else [1.0] * 5
    snr = 1.0

    print("High-dimensional (p > n)" if high_dim else "Low-dimensional (n > p)")
    print_demo_config(n, p, num_dummies, T_stop, true_support, true_coefs, snr)

    X, D, y = gen_synthetic_data(n, p, num_dummies, true_support, true_coefs,
                                 snr, seed=42)

    # External normalization: center + unit-L2 columns for X and D, center y.
    print("Applying external L2 normalization...")
    l2scaler_X = LpNormScaler(NormType.L2, center=True)
    l2scaler_X.fit(X)
    l2scaler_X.transform_inplace(X)

    l2scaler_D = LpNormScaler(NormType.L2, center=True)
    l2scaler_D.fit(D)
    l2scaler_D.transform_inplace(D)

    y -= y.mean()

    solver = TSTAGEWISE_Solver(
        X, D, y,
        SolverOptions(normalize=False, intercept=False, verbose=True))
    solver.executeStep(T_stop, early_stop=True)

    print_selection(solver, true_support)
    print_selection_quality(solver, true_support)
    print("\nT-Stagewise Diagnostics:")
    print(f"  Removals: {solver.getNumRemovals()}")
    print(f"  Cycling ratio: {solver.getCyclingRatio():.4f}")
    print("\n")


# ==============================================================================
# Demo 3: Serialization & Warm Start
# ==============================================================================

def demo_serialization():
    print_section_header("=== Demo 3: T-Stagewise Serialization & Warm Start ===")

    n, p = 100, 50
    num_dummies = p
    T_stop_final = 15
    T_stop_partial = 7
    snr = 1.0

    true_support = [10, 25, 40]
    true_coefs = [2.5, -1.8, 3.2]

    print_demo_config(n, p, num_dummies, T_stop_final, true_support,
                      true_coefs, snr)

    X, D, y = gen_synthetic_data(n, p, num_dummies, true_support, true_coefs,
                                 snr, seed=42)

    # Normalize ONCE in front; the three solvers below are READERS.
    # The reference, checkpoint and resumed solvers share X and D (Fortran-
    # ordered float64 arrays are shared, not copied). A solver constructed
    # with normalize=True / intercept=True is a WRITER that preprocesses them
    # in place, so three writers would center and scale the data three times,
    # which moves it by rounding and made the paths differ (the 2026-09-25
    # finding; a second writer on a held buffer raises since the library's
    # in-place contract). So: the statistics live in the scalers, the solvers
    # run as constructed (normalize=False, intercept=False), no copies,
    # bit-identical paths.
    l2scaler_X = LpNormScaler(NormType.L2, center=True)
    l2scaler_X.fit_transform_inplace(X)
    l2scaler_D = LpNormScaler(NormType.L2, center=True)
    l2scaler_D.fit_transform_inplace(D)
    y_mean = float(y.mean())
    y -= y_mean

    # Reference solver: run until T_stop_final.
    solver_ref = TSTAGEWISE_Solver(
        X, D, y,
        SolverOptions(normalize=False, intercept=False, verbose=False))
    solver_ref.executeStep(T_stop_final, early_stop=True)
    print(f"\u2713 Reference completed with {solver_ref.getNumSteps()} steps\n")

    # relative to the working directory, so the logs carry no machine path
    filename = os.path.relpath(os.path.join(_THIS_DIR, "tstagewise_checkpoint.bin"))

    # STEP 1: Run partial path and save checkpoint.
    solver1 = TSTAGEWISE_Solver(
        X, D, y,
        SolverOptions(normalize=False, intercept=False, verbose=False))
    solver1.executeStep(T_stop_partial, early_stop=True)
    print(f"Checkpoint at partial stop: {solver1.getNumSteps()} steps")
    solver1.save(filename)
    print(f"\u2713 Checkpoint saved at '{os.path.basename(filename)}'")

    # STEP 2: Load from checkpoint and continue.
    # The Python binding hydrates an existing instance (cpp: static
    # load(filename, X, D)); the constructor re-binds the data views. The
    # receiving solver is a reader on the pre-scaled data, the state the
    # checkpoint was saved in.
    solver2 = TSTAGEWISE_Solver(
        X, D, y,
        SolverOptions(normalize=False, intercept=False, verbose=False))
    solver2.load(filename)
    print(f"Loaded from checkpoint: {solver2.getNumSteps()} steps")
    solver2.executeStep(T_stop_final, early_stop=True)

    print("\nCOMPARISON:")
    print(f"Reference steps: {solver_ref.getNumSteps()}")
    print(f"Reloaded steps:  {solver2.getNumSteps()}")
    print(f"RSS diff:   {abs(solver_ref.getRSS()[-1] - solver2.getRSS()[-1]):g}")
    print(f"R2 diff:    {abs(solver_ref.getR2()[-1] - solver2.getR2()[-1]):g}")
    paths_match = solver_ref.getActions() == solver2.getActions()
    print("\u2713 Paths match" if paths_match else "\u2717 Paths differ!")

    # Give the data back (the statistics are in the scalers)
    l2scaler_X.inverse_transform_inplace(X)
    l2scaler_D.inverse_transform_inplace(D)
    y += y_mean

    os.remove(filename)
    print("\u2713 Checkpoint file removed")
    print("\n")


# ==============================================================================
# Demo 4: Memory-Mapped Data
# ==============================================================================

def demo_memory_mapped(high_dim, rnd_coef, T_stop):
    print_section_header("=== Demo 4: T-Stagewise with Memory-Mapped Data ===")

    n = 1000 if high_dim else 5000
    p = 5000 if high_dim else 1000  # cpp: 500,000 on disk; scaled for the port
    num_dummies = 10 * p
    snr = 1.0

    true_support = [4, 27, 149, 398, 420]
    true_coefs = [2.5, -0.4, -0.2, -0.8, 1.1] if rnd_coef else [1.0] * 5

    print_demo_config(n, p, num_dummies, T_stop, true_support, true_coefs, snr)

    print("Generating memory-mapped data...")
    X_file = os.path.join(_THIS_DIR, "demo_tstagewise_X.bin")
    D_file = os.path.join(_THIS_DIR, "demo_tstagewise_D.bin")
    y_file = os.path.join(_THIS_DIR, "demo_tstagewise_y.bin")

    X, D, y = gen_synthetic_data(n, p, num_dummies, true_support, true_coefs,
                                 snr, seed=42)

    try:
        # Write X/D/y to disk (cpp: SyntheticDataMapped) and drop the
        # in-memory copies; the solver reads the zero-copy mmap views.
        X_mmap = numpy_to_memmap(X_file, X)
        D_mmap = numpy_to_memmap(D_file, D)
        y_mmap = numpy_to_memmap(y_file, np.asfortranarray(y.reshape(-1, 1)))
        del X, D
        print("\u2713 Data generated on disk\n")

        print("Running T-Stagewise on memory-mapped data...")
        solver = TSTAGEWISE_Solver(
            X_mmap.to_numpy(), D_mmap.to_numpy(), y_mmap.to_numpy(),
            SolverOptions(normalize=True, intercept=True, verbose=True))
        solver.executeStep(T_stop, early_stop=True)
        print("\u2713 T-Stagewise completed\n")

        print_selection(solver, true_support)
        print_selection_quality(solver, true_support)
        print("\nT-Stagewise Diagnostics:")
        print(f"  Removals: {solver.getNumRemovals()}")
        print(f"  Cycling ratio: {solver.getCyclingRatio():.4f}")
    finally:
        print("\nCleaning up files...")
        for f in (X_file, D_file, y_file):
            if os.path.exists(f):
                os.remove(f)
        print("\u2713 All files removed\n")



# ==============================================================================
# Demo 5: Terminated Solution Paths
# ==============================================================================

def demo_terminated_paths():
    """
    Record the terminated solution paths of one T-Stagewise object that is
    warm-started T = 1 -> 2 -> 5 (setup of the R tlars vignette). Writes
    tstagewise_terminated_paths_{path,actions}.csv to simulation_results/data/;
    generate_plots.sh draws the path figure from them.
    """
    print_section_header("=== Demo 5: T-Stagewise Terminated Solution Paths ===")

    # Setup of the tlars vignette: 5 actives with beta = 1 and noise sigma about 1
    n, p = 150, 300
    num_dummies = p
    true_support = [0, 1, 2, 3, 4]
    true_coefs = [1.0] * 5
    snr = 5.0  # signal variance 5 over noise variance 1
    T_stops = [1, 2, 5]

    print_demo_config(n, p, num_dummies, T_stops[-1], true_support, true_coefs, snr)

    X, D, y = gen_synthetic_data(n, p, num_dummies, true_support, true_coefs,
                                 snr, seed=42)

    # One solver object; every executeStep() call continues its path
    solver = TSTAGEWISE_Solver(
        X, D, y,
        SolverOptions(normalize=True, intercept=True, verbose=False))

    recorder = PathRecorder("tstagewise_terminated_paths", output_dir(_THIS_DIR))
    for T_stop in T_stops:
        solver.executeStep(T_stop, early_stop=True)
        recorder.record(solver, T_stop)

        selected = solver.getActivePredictorIndices()
        print(f"T_stop = {T_stop}: steps = {solver.getNumSteps():2d}, "
              f"predictors = {len(selected):2d}, "
              f"dummies = {len(solver.getActiveDummyIndices())}, "
              f"FDP = {compute_fdp(selected, true_support):.3f}, "
              f"TPP = {compute_tpp(selected, true_support):.3f}")

    print_selection(solver, true_support)

    print(f"\nPath data written to:\n  {display_path(recorder.path_file)}"
          f"\n  {display_path(recorder.actions_file)}\n")

# ==============================================================================
# Main
# ==============================================================================

if __name__ == "__main__":
    print()
    print_section_header("T-Stagewise Demo Suite")

    set_num_threads(min(6, get_max_threads()))
    print(f"Running with {get_num_threads()} threads\n")

    # Optional arguments: the parts to run (default: all)
    run = PartSelection()

    # Demo 1: Basic usage with internal normalization
    if run(1):
        demo_early_stopping(high_dim=True, rnd_coef=False, T_stop=10)
        demo_early_stopping(high_dim=False, rnd_coef=False, T_stop=10)

    # Demo 2: External normalization
    if run(2):
        demo_with_external_normalizer(high_dim=False, rnd_coef=False, T_stop=10)
        demo_with_external_normalizer(high_dim=True, rnd_coef=False, T_stop=10)

    # Demo 3: Serialization
    if run(3):
        demo_serialization()

    # Demo 4: Memory-mapped
    if run(4):
        demo_memory_mapped(high_dim=True, rnd_coef=False, T_stop=10)

    # Demo 5: Terminated solution paths (data for the path figure)
    if run(5):
        demo_terminated_paths()

    print_section_header("\u2713 All demos completed successfully")
