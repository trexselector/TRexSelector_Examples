# ==============================================================================
# demo_mlm_hac_02_mmap.py
# ==============================================================================
#
# LSH Linkage Comparison on memory-mapped data.
#
# Mirrors cpp/ml_methods/hac_clustering/demo_mlm_hac_02_mmap/
#         demo_mlm_hac_02_mmap.cpp.
#
# The cpp demo replicates the LSH linkage comparison at biobank scale
# (N = 100,000 samples, P = 400,000 variables, K = 10 clusters, ~298 GB on
# disk) without ever allocating the full matrix in RAM. This Python mirror
# keeps the same three-phase out-of-core workflow at an illustrative scale:
# N = 2,000, P = 4,000, K = 10 (~61 MB on disk).
#
# The demo is organized into three sequential phases:
#
#   Phase 1 | Generation   : Latent-factor correlated blocks are written
#                            variable-by-variable into the mmap file. RAM cost
#                            is bounded by the latent signal matrix (N x K)
#                            plus one variable at a time.
#
#   Phase 2 | Standardize  : Each variable is mean-centered and L2-normalized
#                            in place (required for Pearson-correlation
#                            distances).
#
#   Phase 3 | Clustering   : Linkage strategies are benchmarked under the pure
#                            O(1) LSH Correlation Approximation:
#                              A. Single Linkage   - SLINK
#                              B. Complete/WPGMA   - [SKIPPED] the O(p^2)
#                                 Lance-Williams matrix is intractable at the
#                                 cpp scale (P = 400,000); the skip is kept to
#                                 mirror the cpp console output.
#                              C. Average Linkage  - UPGMA [SKIPPED as in cpp]
#                              D. Ward Linkage     - minimum variance
#                                                    (projected engine)
#
# Expected result: the executed methods achieve ARI ~ 1.0000 because the block
# geometry creates well-separated clusters in correlation space.
#
# Notes on the port:
#   - agglomerative_cluster() clusters the ROWS of its input, so the matrix is
#     stored TRANSPOSED on disk relative to cpp: P x N, one variable per row.
#     The zero-copy to_numpy() view of the mmap file is passed directly to the
#     clustering call; use_mmap=True additionally keeps the backend's
#     intermediate distance data on disk.
#   - The cpp demo generates and standardizes with OpenMP parallel loops
#     (C++-exclusive) and skips Phases 1-2 when a file of the expected size
#     already exists; this mirror uses a fresh temporary file every run.
#
# ==============================================================================

import os
import tempfile
import time

import numpy as np

from trexselector.ml_methods.clustering import (
    DistanceMetric,
    LinkageMethod,
    agglomerative_cluster,
    cut_tree,
)
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


# ==============================================================================
# Part 1: Data Generation, Preprocessing, and Evaluation
# ==============================================================================

def compute_ari(true_labels, pred_labels):
    """Adjusted Rand Index (ARI) in [-1, 1]; perfect agreement yields 1.0."""
    true_labels = np.asarray(true_labels)
    pred_labels = np.asarray(pred_labels)
    if true_labels.size == 0 or true_labels.size != pred_labels.size:
        return 0.0

    n = true_labels.size
    contingency = np.zeros((true_labels.max() + 1, pred_labels.max() + 1))
    np.add.at(contingency, (true_labels, pred_labels), 1.0)

    def choose2(x):
        return x * (x - 1.0) / 2.0

    sum_comb_c = choose2(contingency).sum()
    sum_comb_a = choose2(contingency.sum(axis=1)).sum()
    sum_comb_b = choose2(contingency.sum(axis=0)).sum()

    expected_index = (sum_comb_a * sum_comb_b) / choose2(float(n))
    max_index = (sum_comb_a + sum_comb_b) / 2.0

    if max_index == expected_index:
        return 1.0
    return (sum_comb_c - expected_index) / (max_index - expected_index)


def build_true_labels(block_sizes):
    """Per-variable ground-truth cluster labels from the block sizes."""
    return np.repeat(np.arange(len(block_sizes)), block_sizes)


def generate_correlated_variables_to_mmap(view, n_samples, k_clusters,
                                          true_labels, centers, std_devs,
                                          seed=42):
    """Generate correlated variable blocks directly into the mmap view.

    Latent factor model: X(j, i) = mu_k + sigma_k * [signal * latent(i, k) +
    noise * eps(j, i)] with k = block(j). The view is P x N (one variable per
    row); only the latent matrix (N x K) and one variable at a time are held
    in RAM.
    """
    rng = np.random.default_rng(seed)
    latent = rng.standard_normal((n_samples, k_clusters))  # N x K in RAM

    signal_strength = 0.85
    noise_strength = 0.15

    p_features = view.shape[0]
    for j in range(p_features):
        blk = true_labels[j]
        base = (signal_strength * latent[:, blk]
                + noise_strength * rng.standard_normal(n_samples))
        view[j, :] = centers[blk] + std_devs[blk] * base


def standardize_rows_for_correlation(view):
    """In-place mean-centering and L2-normalization, one variable (row) at a
    time. In the column-major P x N file a row is strided across the whole
    file, which costs nothing at this size but would not stream at scale."""
    for j in range(view.shape[0]):
        row = view[j, :]
        row -= row.mean()
        norm = np.linalg.norm(row)
        if norm > 1e-12:
            row /= norm
        else:
            row[:] = 0.0
        view[j, :] = row


# ==============================================================================
# Part 2: Demo
# ==============================================================================

def demo_lsh_linkage_comparison_mmap():
    """Mmap-backed LSH Linkage Comparison.

    Generates P x N correlated variables in K imbalanced blocks directly in a
    memory-mapped file, standardizes them in place, then benchmarks HAC
    linkage strategies under the O(1) LSH Correlation Approximation.
    """
    print_section_header(
        "Demo 03: LSH Linkage Comparison via Memory-Mapped I/O "
        "(N=2k, P=4k, K=10)")

    # -------------------------------------------------------------------------
    # 0. Configuration
    # -------------------------------------------------------------------------

    # cpp: N = 100,000, P = 400,000 (~298 GB); scaled by 1/50 and 1/100 to
    # N = 2,000, P = 4,000 (~61 MB) here.
    n_samples, p_features, k_clusters = 2_000, 4_000, 10

    # Imbalanced block sizes — 10 blocks summing exactly to 4,000 (cpp block
    # sizes scaled by 1/100). The deliberate imbalance tests robustness of the
    # LSH approximation to non-uniform cluster sizes.
    block_sizes = [500, 480, 450, 420, 400, 380, 350, 330, 350, 340]
    assert sum(block_sizes) == p_features, "block_sizes must sum to p_features"

    # Per-block cluster centers — spread across [-5.0, +5.0]
    centers = [0.0, 2.0, -2.0, 4.0, -4.0, 1.5, -1.5, 3.0, -3.0, 5.0]

    # Per-block noise scales — varied to create realistic heteroscedasticity
    std_devs = [1.5, 1.2, 0.8, 2.0, 3.5, 1.0, 1.8, 0.6, 2.5, 1.3]

    # Precompute true labels — deterministic from block_sizes, O(P)
    true_labels = build_true_labels(block_sizes)

    # Disk footprint (cpp injects the path via TREX_DEMO_MMAP_PATH; a
    # temporary file in the system temp directory is used here)
    fd, mmap_path = tempfile.mkstemp(prefix="trex_demo03_hac_mmap_",
                                     suffix=".bin")
    os.close(fd)
    file_bytes = n_samples * p_features * 8

    print(f"Memory-mapped file : <tmp>/{os.path.basename(mmap_path)}")
    print(f"File size          : {file_bytes / (1 << 20):.2f} MB")
    print(f"Latent RAM (N x K) : "
          f"{n_samples * k_clusters * 8 / (1 << 20):.2f} MB\n")

    try:
        # ---------------------------------------------------------------------
        # Phase 1 — Data generation (ReadWrite)
        # Phase 2 — Standardization (in-place, same ReadWrite mapping)
        # ---------------------------------------------------------------------
        print("=== Phase 1: Generating correlated variables into mmap ===")
        print(f"    N={n_samples}, P={p_features}, K={k_clusters}")

        # Stored transposed relative to cpp: P x N, one variable per row.
        mmap_matrix = MemoryMappedMatrix(mmap_path, p_features, n_samples,
                                         AccessMode.ReadWrite)
        X_map = mmap_matrix.to_numpy()  # zero-copy view into the file

        t0 = time.perf_counter()
        generate_correlated_variables_to_mmap(
            X_map, n_samples, k_clusters, true_labels, centers, std_devs,
            seed=42)
        print(f"    Done. ({time.perf_counter() - t0:.1f} s)\n")

        print("=== Phase 2: Standardizing variables (mean=0, L2=1) in-place ===")
        t0 = time.perf_counter()
        standardize_rows_for_correlation(X_map)
        print(f"    Done. ({time.perf_counter() - t0:.1f} s)\n")

        # ---------------------------------------------------------------------
        # Phase 3 — Clustering (HAC does not mutate the data)
        # ---------------------------------------------------------------------
        print("=== Phase 3: Clustering the memory-mapped variables ===")
        print("    Distance policy: Correlation LSH Approximation "
              "(O(1) SimHash)\n")

        def run_test(title, method, X):
            print(f"{title}:")
            t0 = time.perf_counter()
            merges = agglomerative_cluster(
                X, method=method,
                metric=DistanceMetric.Correlation_LSH_Approx,
                use_mmap=True)  # backend intermediates also stay on disk
            elapsed = time.perf_counter() - t0
            labels = cut_tree(merges, p_features, k_clusters)
            ari = compute_ari(true_labels, labels)
            print(f"    Time : {elapsed:.4f} s")
            print(f"    ARI  : {ari:.4f} / 1.0000\n")

        # Test A — Single Linkage (routes to SLINK)
        run_test("[Test A] Single Linkage (SLINK + LSH)", LinkageMethod.Single, X_map)

        # Test B — Complete Linkage & WPGMA (skipped, mirroring cpp)
        print("[Test B] Complete Linkage & WPGMA:")
        print("    [SKIPPED] These methods lack a geometric center of mass.")
        print("    They strictly require the O(P^2) Lance-Williams matrix.")
        print("    At the cpp scale (P=400,000) this requires 1.2 TB of NVMe "
              "I/O, which is intractable.\n")

        # Test C — Average Linkage (UPGMA), skipped as in cpp: the dispatcher
        # routes Average to the exact Lance-Williams matrix engine for every
        # metric, and the projected 64D engine is valid for Ward only
        print("[Test C] UPGMA Average Linkage:")
        print("    [SKIPPED] Average linkage runs on the O(P^2) Lance-Williams matrix as well;")
        print("    the geometric centroid update of the projected engine does not implement")
        print("    the UPGMA formula. At the cpp scale (P=400,000) the matrix is 1.28 TB, which")
        print("    is intractable.\n")

        # Test D — Ward's Minimum Variance (projected RAM engine)
        run_test("[Test D] Ward's Minimum Variance "
                 "(Projected 64D RAM Engine + LSH)", LinkageMethod.Ward, X_map)

        print("[Demo 03 Complete]\n")

        del X_map, mmap_matrix  # unmap before the file is removed
    finally:
        # ---------------------------------------------------------------------
        # Cleanup: remove the mmap file
        # ---------------------------------------------------------------------
        try:
            os.unlink(mmap_path)
            print(f"Cleaned up mmap file: <tmp>/{os.path.basename(mmap_path)}")
        except OSError as e:
            print(f"[WARN] Could not remove mmap file: {e}")


# ==============================================================================
# 3. Main
# ==============================================================================

if __name__ == "__main__":
    print()
    print_section_header("HAC Memory-Mapped Demo Suite (N=2k x P=4k)")

    # cpp initializes OpenMP with 6 threads; mirrored via the utils entry points.
    set_num_threads(min(6, get_max_threads()))
    print(f"Running with {get_num_threads()} threads\n")

    demo_lsh_linkage_comparison_mmap()

    print_section_header("Demo 03 completed successfully")
