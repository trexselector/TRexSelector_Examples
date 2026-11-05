# ==============================================================================
# demo_mlm_hac_02_mmap.R
# ==============================================================================
#
# LSH Linkage Comparison on memory-mapped data.
#
# Mirrors cpp/ml_methods/hac_clustering/demo_mlm_hac_02_mmap/
#         demo_mlm_hac_02_mmap.cpp.
#
# The cpp demo replicates the LSH linkage comparison at biobank scale
# (N = 100,000 samples, P = 400,000 variables, K = 10 clusters, ~298 GB on
# disk) without ever allocating the full matrix in RAM. This R mirror keeps the
# same three-phase out-of-core workflow at an illustrative scale:
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
#   - ml_agglomerative_cluster() clusters the ROWS of its input, so the matrix is
#     stored TRANSPOSED on disk relative to cpp: P x N, one variable per row.
#     The mmap_matrix handle is passed directly to the clustering call, which
#     keeps the backend's intermediate distance data on disk as well.
#   - Ward is permitted with Correlation_LSH_Approx: the dispatcher routes it
#     to a 64-dimensional SimHash-projected Euclidean space, which makes it a
#     defined approximate operation.
#   - The cpp demo generates and standardizes with OpenMP parallel loops
#     (C++-exclusive) and skips Phases 1-2 when a file of the expected size
#     already exists; this mirror uses a fresh temporary file every run.
#
# ==============================================================================

library(TRexSelector)


print_section_header <- function(title) {
  cat(strrep("=", 78), "\n", sep = "")
  cat(" ", title, "\n", sep = "")
  cat(strrep("=", 78), "\n\n", sep = "")
}


# ==============================================================================
# Part 1: Data Generation, Preprocessing, and Evaluation
# ==============================================================================

# Adjusted Rand Index (ARI) in [-1, 1]; perfect agreement yields 1.0.
compute_ari <- function(true_labels, pred_labels) {
  n <- length(true_labels)
  if (n == 0L || n != length(pred_labels)) return(0.0)

  choose2 <- function(x) x * (x - 1.0) / 2.0

  tbl        <- table(true = true_labels, pred = pred_labels)
  sum_comb_c <- sum(choose2(as.double(tbl)))
  sum_comb_a <- sum(choose2(as.double(rowSums(tbl))))
  sum_comb_b <- sum(choose2(as.double(colSums(tbl))))

  expected  <- (sum_comb_a * sum_comb_b) / choose2(as.double(n))
  max_index <- (sum_comb_a + sum_comb_b) / 2.0

  if (max_index == expected) return(1.0)
  (sum_comb_c - expected) / (max_index - expected)
}


# Per-variable ground-truth cluster labels from the block sizes.
build_true_labels <- function(block_sizes) {
  rep(seq_along(block_sizes), times = block_sizes)
}


# Generate correlated variable blocks directly into the mmap file.
#
# Latent factor model: X(j, i) = mu_k + sigma_k * [signal * latent(i, k) +
# noise * eps(j, i)] with k = block(j). The mapped matrix is P x N (one
# variable per row); only the latent matrix (N x K) and one variable at a time
# are held in R memory.
generate_correlated_variables_to_mmap <- function(mmap_mat, n_samples,  # nolint: object_length_linter. C++ name
                                                  k_clusters, true_labels,
                                                  centers, std_devs,
                                                  seed = 42L) {
  set.seed(seed)
  latent <- matrix(rnorm(n_samples * k_clusters), nrow = n_samples,
                   ncol = k_clusters)   # N x K in RAM

  signal_strength <- 0.85
  noise_strength  <- 0.15

  p_features <- dim(mmap_mat)[1]
  for (j in seq_len(p_features)) {
    blk  <- true_labels[j]
    base <- signal_strength * latent[, blk] +
      noise_strength * rnorm(n_samples)
    mmap_mat[j, 1:n_samples] <- centers[blk] + std_devs[blk] * base
  }
  invisible(NULL)
}


# In-place mean-centering and L2-normalization, one variable (row) at a time —
# sequential-streaming-friendly for the memory-mapped file.
standardize_rows_for_correlation <- function(mmap_mat) {  # nolint: object_length_linter. C++ name
  dims <- dim(mmap_mat)
  for (j in seq_len(dims[1])) {
    row <- mmap_mat[j, 1:dims[2]]
    row <- row - mean(row)
    nrm <- sqrt(sum(row^2))
    row <- if (nrm > 1e-12) row / nrm else rep(0.0, dims[2])
    mmap_mat[j, 1:dims[2]] <- row
  }
  invisible(NULL)
}


# ==============================================================================
# Part 2: Demo
# ==============================================================================

# Mmap-backed LSH Linkage Comparison.
#
# Generates P x N correlated variables in K imbalanced blocks directly in a
# memory-mapped file, standardizes them in place, then benchmarks HAC linkage
# strategies under the O(1) LSH Correlation Approximation.
demo_lsh_linkage_comparison_mmap <- function() {  # nolint: object_length_linter. C++ name
  print_section_header(
    "Demo 03: LSH Linkage Comparison via Memory-Mapped I/O (N=2k, P=4k, K=10)"
  )

  # ---------------------------------------------------------------------------
  # 0. Configuration
  # ---------------------------------------------------------------------------

  # cpp: N = 100,000, P = 400,000 (~298 GB); scaled by 1/50 and 1/100 to
  # N = 2,000, P = 4,000 (~61 MB) here.
  n_samples <- 2000L
  p_features <- 4000L
  k_clusters <- 10L

  # Imbalanced block sizes — 10 blocks summing exactly to 4,000 (cpp block
  # sizes scaled by 1/100). The deliberate imbalance tests robustness of the
  # LSH approximation to non-uniform cluster sizes.
  block_sizes <- c(500L, 480L, 450L, 420L, 400L, 380L, 350L, 330L, 350L, 340L)
  stopifnot(sum(block_sizes) == p_features)

  # Per-block cluster centers — spread across [-5.0, +5.0]
  centers <- c(0.0, 2.0, -2.0, 4.0, -4.0, 1.5, -1.5, 3.0, -3.0, 5.0)

  # Per-block noise scales — varied to create realistic heteroscedasticity
  std_devs <- c(1.5, 1.2, 0.8, 2.0, 3.5, 1.0, 1.8, 0.6, 2.5, 1.3)

  # Precompute true labels — deterministic from block_sizes, O(P)
  true_labels <- build_true_labels(block_sizes)

  # Disk footprint (cpp injects the path via TREX_DEMO_MMAP_PATH; a temporary
  # file in the session temp directory is used here)
  mmap_path <- tempfile(pattern = "trex_demo03_hac_mmap_", fileext = ".bin")
  on.exit({
    if (file.exists(mmap_path)) {
      unlink(mmap_path)
      cat(sprintf("Cleaned up mmap file: <tmp>/%s\n", basename(mmap_path)))
    }
  }, add = TRUE)

  file_bytes <- as.numeric(n_samples) * as.numeric(p_features) * 8

  cat(sprintf("Memory-mapped file : <tmp>/%s\n", basename(mmap_path)))
  cat(sprintf("File size          : %.2f MB\n", file_bytes / 2^20))
  cat(sprintf("Latent RAM (N x K) : %.2f MB\n\n",
              n_samples * k_clusters * 8 / 2^20))

  # ---------------------------------------------------------------------------
  # Phase 1 — Data generation (readwrite)
  # Phase 2 — Standardization (in place, same readwrite mapping)
  # ---------------------------------------------------------------------------
  cat("=== Phase 1: Generating correlated variables into mmap ===\n")
  cat(sprintf("    N=%d, P=%d, K=%d\n", n_samples, p_features, k_clusters))

  # Stored transposed relative to cpp: P x N, one variable per row.
  mmap_mat <- mem_matrix(mmap_path, rows = p_features, cols = n_samples)

  t0 <- Sys.time()
  generate_correlated_variables_to_mmap(mmap_mat, n_samples, k_clusters,
                                        true_labels, centers, std_devs,
                                        seed = 42L)
  cat(sprintf("    Done. (%.1f s)\n\n",
              as.numeric(difftime(Sys.time(), t0, units = "secs"))))

  cat("=== Phase 2: Standardizing variables (mean=0, L2=1) in-place ===\n")
  t0 <- Sys.time()
  standardize_rows_for_correlation(mmap_mat)
  cat(sprintf("    Done. (%.1f s)\n\n",
              as.numeric(difftime(Sys.time(), t0, units = "secs"))))

  # ---------------------------------------------------------------------------
  # Phase 3 — Clustering (HAC does not mutate the data)
  # ---------------------------------------------------------------------------
  cat("=== Phase 3: Clustering the memory-mapped variables ===\n")
  cat("    Distance policy: Correlation LSH Approximation (O(1) SimHash)\n\n")

  run_test <- function(title, method) {
    cat(sprintf("%s:\n", title))
    t0 <- Sys.time()
    # The mmap handle is passed straight through: the backend reads the
    # variables from disk and keeps its intermediate distance data there too.
    merges <- ml_agglomerative_cluster(
      data = mmap_mat, method = method,
      metric = ml_distance_metric$Correlation_LSH_Approx
    )
    elapsed <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
    labels <- ml_cut_tree(merges, num_orig_objs = p_features,
                          num_clusters = k_clusters)
    cat(sprintf("    Time : %.4f s\n", elapsed))
    cat(sprintf("    ARI  : %.4f / 1.0000\n\n",
                compute_ari(true_labels, labels)))
  }

  # Test A — Single Linkage (routes to SLINK)
  run_test("[Test A] Single Linkage (SLINK + LSH)", ml_linkage_method$Single)

  # Test B — Complete Linkage & WPGMA (skipped, mirroring cpp)
  cat("[Test B] Complete Linkage & WPGMA:\n")
  cat("    [SKIPPED] These methods lack a geometric center of mass.\n")
  cat("    They strictly require the O(P^2) Lance-Williams matrix.\n")
  cat("    At the cpp scale (P=400,000) this requires 1.2 TB of NVMe I/O, ")
  cat("which is intractable.\n\n")

  # Test C — Average Linkage (UPGMA), skipped as in cpp: the dispatcher routes
  # Average to the exact Lance-Williams matrix engine for every metric, and the
  # projected 64D engine is valid for Ward only
  cat("[Test C] UPGMA Average Linkage:\n")
  cat("    [SKIPPED] Average linkage runs on the O(P^2) Lance-Williams matrix as well;\n")
  cat("    the geometric centroid update of the projected engine does not implement\n")
  cat("    the UPGMA formula. At the cpp scale (P=400,000) the matrix is 1.28 TB, which\n")
  cat("    is intractable.\n\n")

  # Test D — Ward's Minimum Variance (projected RAM engine)
  run_test("[Test D] Ward's Minimum Variance (Projected 64D RAM Engine + LSH)",
           ml_linkage_method$Ward)

  cat("[Demo 03 Complete]\n\n")
  invisible(NULL)
}


# ==============================================================================
# 3. Main
# ==============================================================================

cat("\n")
print_section_header("HAC Memory-Mapped Demo Suite (N=2k x P=4k)")

# Six OpenMP threads, as the C++ demo
trex_set_num_threads(min(6L, trex_get_max_threads()))
cat(sprintf("Running with %d threads\n\n", trex_get_num_threads()))

demo_lsh_linkage_comparison_mmap()

print_section_header("Demo 03 completed successfully")
