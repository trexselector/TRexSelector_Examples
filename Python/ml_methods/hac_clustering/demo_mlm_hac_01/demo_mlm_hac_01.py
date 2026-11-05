# ==============================================================================
# demo_mlm_hac_01.py
# ==============================================================================
#
# Demonstration of hierarchical agglomerative clustering (HAC).
#
# Mirrors cpp/ml_methods/hac_clustering/demo_mlm_hac_01/demo_mlm_hac_01.cpp.
#
# Six demos which illustrate the clustering scenarios:
#
#   Demo 1 | Sample clustering    : Single linkage with Euclidean distance.
#                                   (cpp: N = 1,500, P = 50,000; scaled to
#                                   N = 300, P = 1,000 here.)
#
#   Demo 2 | Variable clustering  : Single linkage with Pearson correlation on
#                                   latent-factor variable blocks.
#                                   (cpp: N = 10,000, P = 50,000; scaled to
#                                   N = 500, P = 500 here.)
#
#   Demo 3 | LSH benchmark        : Exact correlation vs. the LSH filter vs.
#                                   the pure O(1) SimHash LSH approximation.
#                                   (cpp: N = 40,000, P = 50,000; scaled to
#                                   N = 800, P = 500 here.)
#
#   Demo 4 | Linkage comparison   : Single / Complete / Average / WPGMA under
#                                   exact correlation, plus Ward / Centroid /
#                                   Median under Euclidean distance.
#                                   (cpp: N = 1,500, P = 50,000; scaled to
#                                   N = 300, P = 500 here. The cpp demo runs
#                                   Single linkage twice — direct SLINK and
#                                   via the dispatcher; the Python binding has
#                                   a single entry point, so it is run once.)
#
#   Demo 5 | LSH linkage compare  : Single / Complete / Average / WPGMA under
#                                   the pure O(1) LSH approximation.
#                                   (cpp: N = 50,000, P = 100,000; scaled to
#                                   N = 500, P = 1,000 here.)
#
#   Demo 6 | AR(1) blocks         : Single / Complete / Average under exact
#                                   correlation on AR(1) Toeplitz chains.
#                                   (cpp: N = 3,000, P = 50,000; scaled to
#                                   N = 300, P = 500 here.)
#
# Notes:
#   - agglomerative_cluster() clusters the ROWS of its input.
#     Sample clustering passes X directly.
#     Variable clustering passes the transposed matrix so that variables
#     become the clustered objects.
#
#   - The cpp demo fills matrices with OpenMP-parallel per-thread RNGs — that
#     part is C++-exclusive; vectorized NumPy generation is used here, so
#     results match the cpp output qualitatively, not bitwise.
#
# ==============================================================================

import time

import numpy as np

from trexselector.ml_methods.clustering import (
    DistanceMetric,
    LinkageMethod,
    agglomerative_cluster,
    cut_tree,
)
from trexselector.utils import get_max_threads, get_num_threads, set_num_threads


def print_section_header(title):
    print("=" * 78)
    print(f" {title}")
    print("=" * 78 + "\n")


# ==============================================================================
# Part 1: Data Generation & Preprocessing & Evaluation Metrics
# ==============================================================================

def compute_ari(true_labels, pred_labels):
    """Adjusted Rand Index (ARI) between two labelings.

    Returns a score between -1.0 (worst) and 1.0 (perfect agreement).
    Mirrors the C++ compute_ari() helper (contingency-table formulation).
    """
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


def generate_hierarchical_row_data(n_samples, p_features, k_clusters,
                                   centers, std_devs, seed=42):
    """Row-wise clusters with custom centers and distinct variances.

    Each row i belongs to cluster z_i and follows the location-scale model
    X[i, j] = mu_{z_i} + sigma_{z_i} * eps with eps ~ N(0, 1). Returns the
    matrix and the true labels for the SAMPLES (rows).
    """
    if n_samples % k_clusters != 0:
        raise ValueError("n_samples must be perfectly divisible by k_clusters.")

    rng = np.random.default_rng(seed)
    cluster_size = n_samples // k_clusters
    true_labels = np.arange(n_samples) // cluster_size

    mu = np.asarray(centers)[true_labels][:, None]
    sigma = np.asarray(std_devs)[true_labels][:, None]
    X = mu + sigma * rng.standard_normal((n_samples, p_features))
    return X, true_labels


def generate_correlated_variables(n_samples, p_features, k_clusters,
                                  block_sizes, centers, std_devs, seed=42):
    """Correlated variables in distinct blocks via a latent-factor model.

    X[i, j] = mu_k + sigma_k * (signal * latent[i, k] + noise * eps[i, j])
    with k = block(j), signal = 0.85, noise = 0.15 — all variables within a
    block move together based on a hidden shared signal. Returns the matrix
    and the true labels of the VARIABLES (columns).
    """
    if sum(block_sizes) != p_features:
        raise ValueError("Sum of block_sizes must exactly equal p_features.")

    rng = np.random.default_rng(seed)
    true_labels = np.repeat(np.arange(k_clusters), block_sizes)

    latent = rng.standard_normal((n_samples, k_clusters))
    signal_strength = 0.85   # within-group shared signal
    noise_strength = 0.15    # independent noise

    base = (signal_strength * latent[:, true_labels]
            + noise_strength * rng.standard_normal((n_samples, p_features)))
    X = np.asarray(centers)[true_labels] + np.asarray(std_devs)[true_labels] * base
    return X, true_labels


def generate_ar1_block_variables(n_samples, p_features, k_clusters,
                                 block_sizes, rhos, seed=42):
    """Variables with block-diagonal AR(1) Toeplitz correlation structure.

    Within block k, adjacent columns satisfy
    x_j = rho_k * x_{j-1} + sqrt(1 - rho_k^2) * eps_j, so the correlation
    decays exactly as rho_k^m with the index separation m and vanishes
    between blocks. Returns the matrix and the true block labels (columns).
    """
    if sum(block_sizes) != p_features:
        raise ValueError("Sum of block_sizes must exactly equal p_features.")

    rng = np.random.default_rng(seed)
    true_labels = np.repeat(np.arange(k_clusters), block_sizes)

    X = rng.standard_normal((n_samples, p_features))

    start = 0
    for k in range(k_clusters):
        rho = rhos[k]
        if not -1.0 < rho < 1.0:
            raise ValueError("Rho must be strictly between -1.0 and 1.0.")
        scale = np.sqrt(1.0 - rho * rho)
        for j in range(1, block_sizes[k]):
            X[:, start + j] = rho * X[:, start + j - 1] + scale * X[:, start + j]
        start += block_sizes[k]

    return X, true_labels


def standardize_for_correlation(X):
    """In-place mean-centering and L2-normalization of the columns."""
    X -= X.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(X, axis=0, keepdims=True)
    np.divide(X, norms, out=X, where=norms > 1e-12)
    X[:, (norms <= 1e-12).ravel()] = 0.0


def cluster_and_score(objects, method, metric, true_labels, k_clusters):
    """Cluster the rows of `objects`, cut into k clusters, report time + ARI."""
    n_objs = objects.shape[0]
    t0 = time.perf_counter()
    merges = agglomerative_cluster(objects, method=method, metric=metric)
    elapsed = time.perf_counter() - t0
    pred_labels = cut_tree(merges, n_objs, k_clusters)
    ari = compute_ari(true_labels, pred_labels)
    return elapsed, ari, pred_labels


def print_cluster_sizes(pred_labels, k_clusters, object_name="variables"):
    counts = np.bincount(np.asarray(pred_labels), minlength=k_clusters)
    for i in range(k_clusters):
        print(f"  Predicted Cluster {i}: {counts[i]} {object_name}")


# ==============================================================================
# Part 2: Demos
# ==============================================================================

def demo_slink_sample_clustering():
    """Demo 1: Clusters samples according to Euclidean distances."""
    print_section_header("Demo 1: SLINK Sample Clustering (Euclidean)")

    # Data Setup (cpp: N = 1,500, P = 50,000; scaled to 300 x 1,000)
    n_samples, p_features, k_clusters = 300, 1_000, 3
    cluster_centers = [0.0, 3.0, 20.0]
    cluster_std_devs = [1.0, 1.0, 1.0]

    print(f"Generating data (N={n_samples}, P={p_features})...")
    X, true_labels = generate_hierarchical_row_data(
        n_samples, p_features, k_clusters, cluster_centers, cluster_std_devs,
        seed=42)

    # Samples are the rows, which is exactly what the binding clusters.
    print("Building Single Linkage tree (SLINK)...")
    elapsed, ari_score, pred_labels = cluster_and_score(
        np.asfortranarray(X), LinkageMethod.Single, DistanceMetric.Euclidean,
        true_labels, k_clusters)
    print(f"✓ Tree built successfully in {elapsed:.6f} seconds.\n")

    print(f"Cutting tree to form {k_clusters} clusters...")
    print("Resulting Sample Clustering:")
    print_cluster_sizes(pred_labels, k_clusters, "samples")

    print("\n--- CLUSTERING PERFORMANCE ---")
    print(f"Adjusted Rand Index (ARI): {ari_score:.4f} / 1.0000")
    print("\n[Demo 1 Complete]\n")


def demo_slink_variable_clustering():
    """Demo 2: Cluster variables using Pearson Correlation."""
    print_section_header("Demo 2: SLINK Variable Clustering (Correlation)")

    # cpp: N = 10,000, P = 50,000, blocks {20k, 10k, 8k, 7k, 5k};
    # scaled by 1/100 to N = 500, P = 500.
    n_samples, p_features, k_clusters = 500, 500, 5
    block_sizes = [200, 100, 80, 70, 50]
    centers = [0.0, 5.0, -5.0, 15.0, -15.0]
    std_devs = [1.0, 1.2, 0.8, 2.0, 3.5]

    print(f"Generating Correlated Variables (N={n_samples}, P={p_features})...")
    X, true_labels = generate_correlated_variables(
        n_samples, p_features, k_clusters, block_sizes, centers, std_devs,
        seed=42)

    print("Standardizing variables (Mean=0, L2=1) for correlation...")
    standardize_for_correlation(X)

    # Variables are the columns; transpose so they become the clustered rows.
    print("Building Single Linkage tree for Variables...")
    elapsed, ari_score, pred_labels = cluster_and_score(
        np.asfortranarray(X.T), LinkageMethod.Single, DistanceMetric.Correlation,
        true_labels, k_clusters)
    print(f"✓ Tree built successfully in {elapsed:.6f} seconds.\n")

    print(f"Cutting tree to form {k_clusters} clusters...")
    print("Resulting Variable Cluster Sizes:")
    print_cluster_sizes(pred_labels, k_clusters)

    print("\n--- CLUSTERING PERFORMANCE ---")
    print(f"Adjusted Rand Index (ARI): {ari_score:.4f} / 1.0000")
    print("\n[Demo 2 Complete]\n")


def demo_slink_variable_clustering_lsh():
    """Demo 3: SLINK Variable Clustering with LSH Comparison."""
    print_section_header("Demo 3: SLINK Variable Clustering (LSH Benchmark)")

    # cpp: N = 40,000, P = 50,000; scaled to N = 800, P = 500.
    n_samples, p_features, k_clusters = 800, 500, 5
    block_sizes = [200, 100, 80, 70, 50]
    centers = [0.0, 2.5, -2.5, 15.0, -15.0]
    std_devs = [1.0, 1.2, 0.8, 2.0, 3.5]

    print(f"Generating Imbalanced Correlated Variables (N={n_samples}, "
          f"P={p_features})...")
    X, true_labels = generate_correlated_variables(
        n_samples, p_features, k_clusters, block_sizes, centers, std_devs,
        seed=42)

    print("Standardizing variables (Mean=0, L2=1) for correlation...")
    standardize_for_correlation(X)
    XT = np.asfortranarray(X.T)

    tests = [
        ("[Test A] Exact Pearson Correlation (The Baseline)",
         DistanceMetric.Correlation, False),
        ("[Test B] LSH Filter (Gatekeeper to Exact Dot Product)",
         DistanceMetric.Correlation_LSH_Filter, False),
        ("[Test C] LSH Approx (Pure O(1) SimHash Angular Estimation)",
         DistanceMetric.Correlation_LSH_Approx, True),
    ]
    for title, metric, show_sizes in tests:
        print("\n" + "-" * 60)
        print(title)
        elapsed, ari_score, pred_labels = cluster_and_score(
            XT, LinkageMethod.Single, metric, true_labels, k_clusters)
        print(f"✓ Tree built successfully in {elapsed:.6f} seconds.")
        if show_sizes:
            print("\nResulting Variable Cluster Sizes (Approx):")
            print_cluster_sizes(pred_labels, k_clusters)
        print(f"ARI: {ari_score:.4f} / 1.0000")

    print("\n[Demo 3 Complete]\n")


def demo_compare_linkage():
    """Demo 4: Comparison of Linkage Methods."""
    print_section_header("Demo 4: Linkage Comparison")

    # cpp: N = 1,500, P = 50,000; scaled to N = 300, P = 500.
    n_samples, p_features, k_clusters = 300, 500, 5
    block_sizes = [200, 100, 80, 70, 50]
    centers = [0.0, 5.0, -5.0, 15.0, -15.0]
    std_devs = [1.0, 1.2, 0.8, 2.0, 3.5]

    print(f"1. Generating Clean Correlated Variables (N={n_samples}, "
          f"P={p_features})...")
    X, true_labels = generate_correlated_variables(
        n_samples, p_features, k_clusters, block_sizes, centers, std_devs,
        seed=42)

    print("2. Standardizing variables (Mean=0, L2=1) for correlation...")
    standardize_for_correlation(X)
    XT = np.asfortranarray(X.T)

    print("3. Setting distance policy (Correlation)...")
    # cpp runs Single linkage twice (direct SLINK and via the dispatcher);
    # the Python binding exposes one entry point, so it appears once here.
    corr_tests = [
        ("[Test A] Single Linkage (SLINK)", LinkageMethod.Single),
        ("[Test B] Complete Linkage (Disk/Matrix Bound)", LinkageMethod.Complete),
        ("[Test C] UPGMA (RAM/Geometric Bound)", LinkageMethod.Average),
        ("[Test D] WPGMA (Disk/Matrix Bound)", LinkageMethod.WPGMA),
    ]
    for title, method in corr_tests:
        print(f"\n{title}:")
        elapsed, ari, _ = cluster_and_score(
            XT, method, DistanceMetric.Correlation, true_labels, k_clusters)
        print(f"Time: {elapsed:.6f} seconds")
        print(f"ARI:  {ari:.4f} / 1.0000")

    # Ward, Centroid, and Median require Euclidean geometry.
    euclid_tests = [
        ("[Test E] WARD (RAM/Geometric Bound)", LinkageMethod.Ward),
        ("[Test F] Centroid Linkage (Generic Linkage)", LinkageMethod.Centroid),
        ("[Test G] Median Linkage (Generic Linkage)", LinkageMethod.Median),
    ]
    for title, method in euclid_tests:
        print(f"\n{title}:")
        if method == LinkageMethod.Ward:
            print(" -> Note: Switching to Squared Euclidean Distance!")
        elapsed, ari, _ = cluster_and_score(
            XT, method, DistanceMetric.Euclidean, true_labels, k_clusters)
        print(f"Time: {elapsed:.6f} seconds")
        print(f"ARI:  {ari:.4f} / 1.0000")

    print("\n[Demo 4 Complete]\n")


def demo_lsh_linkage_comparison():
    """Demo 5: Impact of LSH Approximation across different Linkage Methods.

    This demo illustrates the geometry of latent factors: dense variable
    blocks driven by hidden signals form distinct clusters, where the O(1)
    LSH SimHash approximation excels.
    """
    print_section_header("Demo 5: LSH Approx Linkage Comparison")

    # cpp: N = 50,000, P = 100,000; scaled by 1/100 to N = 500, P = 1,000.
    n_samples, p_features, k_clusters = 500, 1_000, 5
    block_sizes = [200, 150, 180, 170, 300]
    centers = [0.0, 1.0, -1.0, 5.0, -5.0]
    std_devs = [1.5, 1.2, 0.8, 2.0, 3.5]

    print(f"1. Generating Imbalanced Correlated Variables (N={n_samples}, "
          f"P={p_features})...")
    X, true_labels = generate_correlated_variables(
        n_samples, p_features, k_clusters, block_sizes, centers, std_devs,
        seed=42)

    print("2. Standardizing variables (Mean=0, L2=1) for correlation...")
    standardize_for_correlation(X)
    XT = np.asfortranarray(X.T)

    print("3. Setting distance policy (Pure O(1) Correlation LSH Approx)...")
    tests = [
        ("[Test A] Single Linkage (SLINK with LSH)", LinkageMethod.Single),
        ("[Test B] Complete Linkage (Disk/Matrix Bound)", LinkageMethod.Complete),
        ("[Test C] UPGMA (RAM/Geometric Bound)", LinkageMethod.Average),
        ("[Test D] WPGMA (Disk/Matrix Bound)", LinkageMethod.WPGMA),
    ]
    for title, method in tests:
        print(f"\n{title}:")
        elapsed, ari, _ = cluster_and_score(
            XT, method, DistanceMetric.Correlation_LSH_Approx,
            true_labels, k_clusters)
        print(f"Time: {elapsed:.6f} seconds")
        print(f"ARI:  {ari:.4f} / 1.0000")

    print("\n[Demo 5 Complete]\n")


def demo_ar1_linkage_comparison():
    """Demo 6: AR(1) Toeplitz Correlation Clustering.

    The geometry of AR(1) chains: correlation decays over index distance
    (variable 1 is close to 2, but far from 100).
      - Algorithmic rule: Single Linkage (SLINK) perfectly chains this
        together; Complete and Average linkage shatter the clusters.
      - LSH rule: do not use LSH here — the 64-bit SimHash quantization
        destroys the continuous correlation gradient. Use Exact Correlation.
    """
    print_section_header("Demo 6: AR(1) Block Linkage Comparison")

    # cpp: N = 3,000, P = 50,000; scaled to N = 300, P = 500.
    n_samples, p_features, k_clusters = 300, 500, 5
    block_sizes = [200, 100, 80, 70, 50]
    # rhos: correlation between strictly adjacent variables in each block.
    rhos = [0.95, 0.90, 0.85, 0.80, 0.75]

    print(f"1. Generating AR(1) Toeplitz Variables (N={n_samples}, "
          f"P={p_features})...")
    X, true_labels = generate_ar1_block_variables(
        n_samples, p_features, k_clusters, block_sizes, rhos, seed=42)

    print("2. Standardizing variables (Mean=0, L2=1) for correlation...")
    standardize_for_correlation(X)
    XT = np.asfortranarray(X.T)

    print("3. Setting distance policy (Exact Pearson Correlation)...")
    tests = [
        ("[Test A] Single Linkage (SLINK)", LinkageMethod.Single),
        ("[Test B] Complete Linkage (Disk/Matrix Bound)", LinkageMethod.Complete),
        ("[Test C] UPGMA (RAM/Geometric Bound)", LinkageMethod.Average),
    ]
    for title, method in tests:
        print(f"\n{title}:")
        elapsed, ari, _ = cluster_and_score(
            XT, method, DistanceMetric.Correlation, true_labels, k_clusters)
        print(f"Time: {elapsed:.6f} seconds")
        print(f"ARI:  {ari:.4f} / 1.0000")

    print("\n[Demo 6 Complete]\n")


# ==============================================================================
# 3. Main
# ==============================================================================

if __name__ == "__main__":
    print()
    print_section_header("Dendrograms Demo Suite")

    # cpp initializes OpenMP with 6 threads; mirrored via the utils entry points.
    set_num_threads(min(6, get_max_threads()))
    print(f"Running with {get_num_threads()} threads\n")

    # Each demo is guarded by its own if-toggle so individual parts can be
    # switched off while iterating on one of them (mirrors the cpp main()).
    if True:
        demo_slink_sample_clustering()
    if True:
        demo_slink_variable_clustering()
    if True:
        demo_slink_variable_clustering_lsh()
    if True:
        demo_compare_linkage()
    if True:
        demo_lsh_linkage_comparison()
    if True:
        demo_ar1_linkage_comparison()

    print_section_header("All clustering demos completed successfully")
