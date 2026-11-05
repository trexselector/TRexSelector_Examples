# ==============================================================================
# demo_mlm_scaler_01.py
# ==============================================================================
#
# Demonstration of ZScoreScaler and LpNormScaler usage.
#
# Mirrors cpp/ml_methods/scaler_methods/demo_mlm_scaler_01/demo_mlm_scaler_01.cpp.
#
# Four demos which illustrate the column-scaling utilities:
#
#   Demo 1 | Scaler comparison  : ZScoreScaler under all four center / scale
#                                 combinations and LpNormScaler (L2 / L1),
#                                 summarized through aggregate column stats.
#
#   Demo 2 | Inverse transform  : fit -> transform_inplace ->
#                                 inverse_transform_inplace round-trip accuracy
#                                 for ZScoreScaler and the L2 LpNormScaler.
#
#   Demo 3 | Serialization      : save() a fitted ZScoreScaler to disk, load()
#                                 it into a fresh object, compare parameters.
#
#   Demo 4 | Memory-mapped      : L2 LpNormScaler fit / transform / inverse
#                                 transform directly on a MemoryMappedMatrix
#                                 zero-copy view.
#                                 (cpp: n = 20,000, p = 100,000, filled with an
#                                 OpenMP parallel loop; scaled to n = 2,000,
#                                 p = 10,000 here and filled with a vectorized
#                                 NumPy broadcast instead.)
#
# ==============================================================================

import os
import tempfile

import numpy as np

from trexselector.ml_methods import LpNormScaler, NormType, ZScoreScaler
from trexselector.utils import (
    AccessMode,
    MemoryMappedMatrix,
    get_max_threads,
    get_num_threads,
    set_num_threads,
)


def print_section_header(title):
    print("=" * 78)
    print(f" {title}")
    print("=" * 78 + "\n")


def print_column_stats(label, X):
    """Print compact per-column statistics, averaged over all columns.

    Reveals the effect of the center / scale switches at a glance:
      - centering drives the average absolute column mean toward 0,
      - z-score scaling drives the average column SD toward 1,
      - L2 normalization drives the average column L2 norm toward 1.
    """
    col_mean = X.mean(axis=0)
    col_sd = X.std(axis=0, ddof=1)
    col_l2 = np.linalg.norm(X, axis=0)
    print(f"  {label:<36}avg|mean| = {np.abs(col_mean).mean():.2e}"
          f"   avg SD = {col_sd.mean():.4f}"
          f"   avg L2 = {col_l2.mean():.2f}")


# ==============================================================================
# Demo 1: center / scale switches (R scale() semantics)
# ==============================================================================

def demo_scaler_comparison():
    print_section_header("Scaler Comparison - center / scale switches")

    # Step 1. Generate data with a deliberately non-zero mean (5.0) and
    #         non-unit spread (SD 3.0), so the effect of each switch is visible.
    n, p = 1_000, 500
    rng = np.random.default_rng(123)
    X = np.asfortranarray(rng.normal(5.0, 3.0, (n, p)))

    print(f"Data: {n} x {p} drawn from N(mean=5, sd=3)\n")
    print_column_stats("original (untransformed)", X)
    print()

    # Step 2. ZScoreScaler under all four center / scale combinations.
    #         The (center, scale) booleans follow R's scale(): centering drives
    #         avg|mean| -> 0, scaling drives avg SD -> 1. With center=False the
    #         scale statistic is the root-mean-square around 0, not the SD.
    z_cases = [
        (True,  True,  "ZScore(center=True,  scale=True )"),
        (False, True,  "ZScore(center=False, scale=True )"),
        (True,  False, "ZScore(center=True,  scale=False)"),
        (False, False, "ZScore(center=False, scale=False)"),
    ]
    for center, scale, label in z_cases:
        Xt = X.copy(order="F")
        zscaler = ZScoreScaler(center=center, scale=scale)
        zscaler.fit(Xt)
        zscaler.transform_inplace(Xt)
        print_column_stats(label, Xt)
    print()

    # Step 3. LpNormScaler: L2 vs L1, with centering on and off (scale kept on).
    #         With scale=True each column is divided by its Lp norm around the
    #         applied center, so the average column L2 norm collapses toward 1
    #         for the L2 variant.
    l_cases = [
        (NormType.L2, True,  "LpNorm(L2, center=True )"),
        (NormType.L2, False, "LpNorm(L2, center=False)"),
        (NormType.L1, True,  "LpNorm(L1, center=True )"),
    ]
    for norm, center, label in l_cases:
        Xt = X.copy(order="F")
        lscaler = LpNormScaler(norm, center, True)
        lscaler.fit(Xt)
        lscaler.transform_inplace(Xt)
        print_column_stats(label, Xt)

    print("\n")


# ==============================================================================
# Demo 2: Transform and Inverse Transform Test
# ==============================================================================

def demo_scaler_inverse_transform():
    print_section_header("Scaler Inverse Transform Accuracy")

    # Step 1. Generate random data
    n, p = 1_000, 5_000
    print(f"Matrix size: {n} x {p}\n")

    rng = np.random.default_rng(456)
    X_orig = np.asfortranarray(rng.standard_normal((n, p)))

    # Step 2. Test ZScoreScaler
    X = X_orig.copy(order="F")
    zscaler = ZScoreScaler()
    zscaler.fit(X)
    zscaler.transform_inplace(X)
    zscaler.inverse_transform_inplace(X)

    err = np.abs(X - X_orig)
    print("ZScoreScaler:")
    print(f"  Max Error: {err.max():e}")
    print(f"  Mean Error: {err.mean():e}\n")

    # Step 3. Test LpNormScaler L2
    X = X_orig.copy(order="F")
    l2scaler = LpNormScaler(NormType.L2, True)
    l2scaler.fit(X)
    l2scaler.transform_inplace(X)
    l2scaler.inverse_transform_inplace(X)

    err = np.abs(X - X_orig)
    print("LpNormScaler L2:")
    print(f"  Max Error: {err.max():e}")
    print(f"  Mean Error: {err.mean():e}\n")
    print("\n")


# ==============================================================================
# Demo 3: Serialization
# ==============================================================================

def demo_scaler_serialization():
    print_section_header("Scaler Serialization")

    # Step 1. Generate data
    n, p = 1_000, 500
    rng = np.random.default_rng(999)
    X = np.asfortranarray(rng.standard_normal((n, p)))

    # Step 2. Fit and save (cpp: fallback filename "scaler_test.bin";
    #         a temporary file in the system temp directory is used here)
    fd, filename = tempfile.mkstemp(prefix="trex_scaler_demo_", suffix=".bin")
    os.close(fd)
    try:
        scaler = ZScoreScaler()
        scaler.fit(X)
        scaler.save(filename)
        print(f"✓ Saved ZScoreScaler to '<tmp>/{os.path.basename(filename)}'")

        # Step 3. Load and verify
        loaded_scaler = ZScoreScaler()
        loaded_scaler.load(filename)
        print(f"✓ Loaded ZScoreScaler from '<tmp>/{os.path.basename(filename)}'\n")

        # Verify means and scales match
        mean_diff = np.linalg.norm(scaler.get_centers() - loaded_scaler.get_centers())
        scale_diff = np.linalg.norm(scaler.get_scales() - loaded_scaler.get_scales())

        print("Verification:")
        print(f"  Mean difference: {mean_diff:e}")
        print(f"  Scale difference: {scale_diff:e}\n")

        if mean_diff < 1e-10 and scale_diff < 1e-10:
            print("✓ Serialization perfect!\n")
    finally:
        # Step 4. Clean up
        try:
            os.unlink(filename)
        except OSError:
            pass
    print("\n")


# ==============================================================================
# Demo 4: Memory-Mapped Matrix
# ==============================================================================

def demo_scaler_on_mmap_matrix():
    print_section_header("Scaler on Memory-Mapped Matrix")

    # Step 1. Create memory-mapped matrix
    #         (cpp: n = 20,000, p = 100,000; scaled to 2,000 x 10,000 here)
    print("Creating memory-mapped matrix...")
    n, p = 2_000, 10_000
    print(f"Matrix size: {n} x {p}")
    fd, mmap_filename = tempfile.mkstemp(prefix="trex_scaler_mmap_", suffix=".bin")
    os.close(fd)
    try:
        mmap_matrix = MemoryMappedMatrix(mmap_filename, n, p, AccessMode.ReadWrite)
        X_map = mmap_matrix.to_numpy()  # zero-copy view into the file

        # Step 2. Populate matrix: X(i, j) = 2*i + j/100
        #         (cpp fills the columns with an OpenMP parallel loop — that
        #         part is C++-exclusive; a vectorized broadcast is used here)
        i = np.arange(n, dtype=np.float64)[:, None]
        j = np.arange(p, dtype=np.float64)[None, :]
        X_map[:, :] = 2.0 * i + j / 100.0

        # Step 3. Fit and run LpNormScaler
        l2scaler = LpNormScaler(NormType.L2, True)
        print("Fitting L2 LpNormScaler on memory-mapped matrix...")
        l2scaler.fit(X_map)
        print("✓ LpNormScaler fitted")
        l2scaler.transform_inplace(X_map)
        print("✓ Data transformed")
        l2scaler.inverse_transform_inplace(X_map)
        print("✓ Data inverse transformed")
        del X_map, mmap_matrix  # unmap before the file is removed
    finally:
        # Step 4. Clean up
        try:
            os.unlink(mmap_filename)
        except OSError:
            pass
        print("✓ Memory-mapped file removed")
    print("\n")


# ==============================================================================
# Main
# ==============================================================================

if __name__ == "__main__":
    print()
    print_section_header("DataTransformer Demo Suite")

    # cpp initializes OpenMP with 6 threads; the backend thread pool is
    # configured here through the same utils entry points.
    set_num_threads(min(6, get_max_threads()))
    print(f"Running with {get_num_threads()} threads\n")

    demo_scaler_comparison()
    demo_scaler_inverse_transform()
    demo_scaler_serialization()
    demo_scaler_on_mmap_matrix()

    print()
    print_section_header("All Demos Completed Successfully")
