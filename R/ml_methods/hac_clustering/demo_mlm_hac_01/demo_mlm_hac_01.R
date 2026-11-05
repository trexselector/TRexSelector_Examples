# ==============================================================================
# demo_mlm_hac_01.R
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
#                                   via the dispatcher; the R binding has a
#                                   single entry point, so it is run once.)
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
# Notes on the port:
#   - ml_agglomerative_cluster() clusters the ROWS of its input.
#     Sample clustering passes X directly.
#     Variable clustering passes t(X) so that variables become the clustered
#     objects.
#   - Ward / Median / Centroid require Euclidean geometry: their Lance-Williams
#     updates are derived from the law of cosines. The one exception is
#     Ward + Correlation_LSH_Approx, which the dispatcher routes through a
#     64-dimensional SimHash-projected Euclidean space.
#   - ml_cut_tree() returns 1-based cluster labels in R (0-based in C++/Python);
#     compute_ari() below is invariant to the label encoding.
#   - The cpp demo fills matrices with OpenMP-parallel per-thread RNGs — that
#     part is C++-exclusive; vectorized R generation is used here, so results
#     match the cpp output qualitatively, not bitwise.
#
# ==============================================================================

library(TRexSelector)


print_section_header <- function(title) {
  cat(strrep("=", 78), "\n", sep = "")
  cat(" ", title, "\n", sep = "")
  cat(strrep("=", 78), "\n\n", sep = "")
}


# ==============================================================================
# Part 1: Data Generation & Preprocessing & Evaluation Metrics
# ==============================================================================

# Adjusted Rand Index (ARI) between two labelings; returns a score between
# -1.0 (worst) and 1.0 (perfect agreement). R port of the inline C++
# compute_ari() helper (contingency-table formulation).
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


# Row-wise clusters with custom centers and distinct variances.
#
# Each row i belongs to cluster z_i and follows the location-scale model
# X[i, j] = mu_{z_i} + sigma_{z_i} * eps with eps ~ N(0, 1). Returns the
# matrix and the true labels for the SAMPLES (rows).
generate_hierarchical_row_data <- function(n_samples, p_features, k_clusters,
                                           centers, std_devs, seed = 42L) {
  if (n_samples %% k_clusters != 0L) {
    stop("n_samples must be perfectly divisible by k_clusters.")
  }
  set.seed(seed)
  cluster_size <- n_samples %/% k_clusters
  true_labels  <- rep(seq_len(k_clusters), each = cluster_size)

  X <- matrix(rnorm(n_samples * p_features), nrow = n_samples, ncol = p_features)
  X <- centers[true_labels] + std_devs[true_labels] * X
  list(X = X, true_labels = true_labels)
}


# Correlated variables in distinct blocks via a latent-factor model.
#
# X[i, j] = mu_k + sigma_k * (signal * latent[i, k] + noise * eps[i, j])
# with k = block(j), signal = 0.85, noise = 0.15 — all variables within a
# block move together based on a hidden shared signal. Returns the matrix and
# the true labels of the VARIABLES (columns).
generate_correlated_variables <- function(n_samples, p_features, k_clusters,
                                          block_sizes, centers, std_devs,
                                          seed = 42L) {
  if (sum(block_sizes) != p_features) {
    stop("Sum of block_sizes must exactly equal p_features.")
  }
  set.seed(seed)
  true_labels <- rep(seq_len(k_clusters), times = block_sizes)

  latent <- matrix(rnorm(n_samples * k_clusters), nrow = n_samples,
                   ncol = k_clusters)
  signal_strength <- 0.85   # within-group shared signal
  noise_strength  <- 0.15   # independent noise

  base <- signal_strength * latent[, true_labels, drop = FALSE] +
    noise_strength * matrix(rnorm(n_samples * p_features), nrow = n_samples,
                            ncol = p_features)
  X <- sweep(sweep(base, 2L, std_devs[true_labels], "*"),
             2L, centers[true_labels], "+")
  list(X = X, true_labels = true_labels)
}


# Variables with block-diagonal AR(1) Toeplitz correlation structure.
#
# Within block k, adjacent columns satisfy
# x_j = rho_k * x_{j-1} + sqrt(1 - rho_k^2) * eps_j, so the correlation decays
# exactly as rho_k^m with the index separation m and vanishes between blocks.
# Returns the matrix and the true block labels (columns).
generate_ar1_block_variables <- function(n_samples, p_features, k_clusters,
                                         block_sizes, rhos, seed = 42L) {
  if (sum(block_sizes) != p_features) {
    stop("Sum of block_sizes must exactly equal p_features.")
  }
  set.seed(seed)
  true_labels <- rep(seq_len(k_clusters), times = block_sizes)

  X <- matrix(rnorm(n_samples * p_features), nrow = n_samples, ncol = p_features)

  start <- 0L
  for (k in seq_len(k_clusters)) {
    rho <- rhos[k]
    if (rho <= -1.0 || rho >= 1.0) {
      stop("Rho must be strictly between -1.0 and 1.0.")
    }
    scale_k <- sqrt(1.0 - rho * rho)
    if (block_sizes[k] > 1L) {
      for (j in 2L:block_sizes[k]) {
        X[, start + j] <- rho * X[, start + j - 1L] + scale_k * X[, start + j]
      }
    }
    start <- start + block_sizes[k]
  }
  list(X = X, true_labels = true_labels)
}


# In-place-style mean-centering and L2-normalization of the columns.
standardize_for_correlation <- function(X) {
  X <- sweep(X, 2L, colMeans(X), "-")
  col_norms <- sqrt(colSums(X^2))
  col_norms[col_norms <= 1e-12] <- 1.0   # guard against zero-variance columns
  sweep(X, 2L, col_norms, "/")
}


# Cluster the rows of `objects`, cut into k clusters, report time + ARI.
cluster_and_score <- function(objects, method, metric, true_labels, k_clusters) {
  n_objs <- nrow(objects)
  t0 <- Sys.time()
  merges <- ml_agglomerative_cluster(objects, method = method, metric = metric)
  elapsed <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
  pred_labels <- ml_cut_tree(merges, num_orig_objs = n_objs,
                             num_clusters = k_clusters)
  list(elapsed = elapsed,
       ari = compute_ari(true_labels, pred_labels),
       pred_labels = pred_labels)
}


print_cluster_sizes <- function(pred_labels, k_clusters,
                                object_name = "variables") {
  # Cluster ids are 1-based in R (the cpp/Python demos print 0-based ids).
  counts <- tabulate(pred_labels, nbins = k_clusters)
  for (i in seq_len(k_clusters)) {
    cat(sprintf("  Predicted Cluster %d: %d %s\n", i, counts[i], object_name))
  }
}


# ==============================================================================
# Part 2: Demos
# ==============================================================================

# Demo 1: Clusters samples according to Euclidean distances.
demo_slink_sample_clustering <- function() {
  print_section_header("Demo 1: SLINK Sample Clustering (Euclidean)")

  # Data setup (cpp: N = 1,500, P = 50,000; scaled to 300 x 1,000)
  n_samples <- 300L
  p_features <- 1000L
  k_clusters <- 3L
  cluster_centers  <- c(0.0, 3.0, 20.0)
  cluster_std_devs <- c(1.0, 1.0, 1.0)

  cat(sprintf("Generating data (N=%d, P=%d)...\n", n_samples, p_features))
  dat <- generate_hierarchical_row_data(n_samples, p_features, k_clusters,
                                        cluster_centers, cluster_std_devs,
                                        seed = 42L)

  # Samples are the rows, which is exactly what the binding clusters.
  cat("Building Single Linkage tree (SLINK)...\n")
  res <- cluster_and_score(dat$X, ml_linkage_method$Single, ml_distance_metric$Euclidean,
                           dat$true_labels, k_clusters)
  cat(sprintf("✓ Tree built successfully in %.6f seconds.\n\n", res$elapsed))

  cat(sprintf("Cutting tree to form %d clusters...\n", k_clusters))
  cat("Resulting Sample Clustering:\n")
  print_cluster_sizes(res$pred_labels, k_clusters, "samples")

  cat("\n--- CLUSTERING PERFORMANCE ---\n")
  cat(sprintf("Adjusted Rand Index (ARI): %.4f / 1.0000\n", res$ari))
  cat("\n[Demo 1 Complete]\n\n")
  invisible(NULL)
}


# Demo 2: Cluster variables using Pearson correlation.
demo_slink_variable_clustering <- function() {
  print_section_header("Demo 2: SLINK Variable Clustering (Correlation)")

  # cpp: N = 10,000, P = 50,000, blocks {20k, 10k, 8k, 7k, 5k};
  # scaled by 1/100 to N = 500, P = 500.
  n_samples <- 500L
  p_features <- 500L
  k_clusters <- 5L
  block_sizes <- c(200L, 100L, 80L, 70L, 50L)
  centers     <- c(0.0, 5.0, -5.0, 15.0, -15.0)
  std_devs    <- c(1.0, 1.2, 0.8, 2.0, 3.5)

  cat(sprintf("Generating Correlated Variables (N=%d, P=%d)...\n",
              n_samples, p_features))
  dat <- generate_correlated_variables(n_samples, p_features, k_clusters,
                                       block_sizes, centers, std_devs,
                                       seed = 42L)

  cat("Standardizing variables (Mean=0, L2=1) for correlation...\n")
  X_std <- standardize_for_correlation(dat$X)

  # Variables are the columns; transpose so they become the clustered rows.
  cat("Building Single Linkage tree for Variables...\n")
  res <- cluster_and_score(t(X_std), ml_linkage_method$Single,
                           ml_distance_metric$Correlation, dat$true_labels,
                           k_clusters)
  cat(sprintf("✓ Tree built successfully in %.6f seconds.\n\n", res$elapsed))

  cat(sprintf("Cutting tree to form %d clusters...\n", k_clusters))
  cat("Resulting Variable Cluster Sizes:\n")
  print_cluster_sizes(res$pred_labels, k_clusters)

  cat("\n--- CLUSTERING PERFORMANCE ---\n")
  cat(sprintf("Adjusted Rand Index (ARI): %.4f / 1.0000\n", res$ari))
  cat("\n[Demo 2 Complete]\n\n")
  invisible(NULL)
}


# Demo 3: SLINK variable clustering with LSH comparison.
demo_slink_variable_clustering_lsh <- function() {  # nolint: object_length_linter. C++ name
  print_section_header("Demo 3: SLINK Variable Clustering (LSH Benchmark)")

  # cpp: N = 40,000, P = 50,000; scaled to N = 800, P = 500.
  n_samples <- 800L
  p_features <- 500L
  k_clusters <- 5L
  block_sizes <- c(200L, 100L, 80L, 70L, 50L)
  centers     <- c(0.0, 2.5, -2.5, 15.0, -15.0)
  std_devs    <- c(1.0, 1.2, 0.8, 2.0, 3.5)

  cat(sprintf("Generating Imbalanced Correlated Variables (N=%d, P=%d)...\n",
              n_samples, p_features))
  dat <- generate_correlated_variables(n_samples, p_features, k_clusters,
                                       block_sizes, centers, std_devs,
                                       seed = 42L)

  cat("Standardizing variables (Mean=0, L2=1) for correlation...\n")
  XT <- t(standardize_for_correlation(dat$X))

  tests <- list(
    list(title = "[Test A] Exact Pearson Correlation (The Baseline)",
         metric = ml_distance_metric$Correlation, show_sizes = FALSE),
    list(title = "[Test B] LSH Filter (Gatekeeper to Exact Dot Product)",
         metric = ml_distance_metric$Correlation_LSH_Filter, show_sizes = FALSE),
    list(title = "[Test C] LSH Approx (Pure O(1) SimHash Angular Estimation)",
         metric = ml_distance_metric$Correlation_LSH_Approx, show_sizes = TRUE)
  )
  for (tst in tests) {
    cat("\n", strrep("-", 60), "\n", sep = "")
    cat(tst$title, "\n", sep = "")
    res <- cluster_and_score(XT, ml_linkage_method$Single, tst$metric,
                             dat$true_labels, k_clusters)
    cat(sprintf("✓ Tree built successfully in %.6f seconds.\n", res$elapsed))
    if (tst$show_sizes) {
      cat("\nResulting Variable Cluster Sizes (Approx):\n")
      print_cluster_sizes(res$pred_labels, k_clusters)
    }
    cat(sprintf("ARI: %.4f / 1.0000\n", res$ari))
  }

  cat("\n[Demo 3 Complete]\n\n")
  invisible(NULL)
}


# Demo 4: Comparison of linkage methods.
demo_compare_linkage <- function() {
  print_section_header("Demo 4: Linkage Comparison")

  # cpp: N = 1,500, P = 50,000; scaled to N = 300, P = 500.
  n_samples <- 300L
  p_features <- 500L
  k_clusters <- 5L
  block_sizes <- c(200L, 100L, 80L, 70L, 50L)
  centers     <- c(0.0, 5.0, -5.0, 15.0, -15.0)
  std_devs    <- c(1.0, 1.2, 0.8, 2.0, 3.5)

  cat(sprintf("1. Generating Clean Correlated Variables (N=%d, P=%d)...\n",
              n_samples, p_features))
  dat <- generate_correlated_variables(n_samples, p_features, k_clusters,
                                       block_sizes, centers, std_devs,
                                       seed = 42L)

  cat("2. Standardizing variables (Mean=0, L2=1) for correlation...\n")
  XT <- t(standardize_for_correlation(dat$X))

  cat("3. Setting distance policy (Correlation)...\n")
  # cpp runs Single linkage twice (direct SLINK and via the dispatcher);
  # the R binding exposes one entry point, so it appears once here.
  corr_tests <- list(
    list(title = "[Test A] Single Linkage (SLINK)", method = ml_linkage_method$Single),
    list(title = "[Test B] Complete Linkage (Disk/Matrix Bound)",
         method = ml_linkage_method$Complete),
    list(title = "[Test C] UPGMA (RAM/Geometric Bound)",
         method = ml_linkage_method$Average),
    list(title = "[Test D] WPGMA (Disk/Matrix Bound)", method = ml_linkage_method$WPGMA)
  )
  for (tst in corr_tests) {
    cat(sprintf("\n%s:\n", tst$title))
    res <- cluster_and_score(XT, tst$method, ml_distance_metric$Correlation,
                             dat$true_labels, k_clusters)
    cat(sprintf("Time: %.6f seconds\n", res$elapsed))
    cat(sprintf("ARI:  %.4f / 1.0000\n", res$ari))
  }

  # Ward, Centroid, and Median require Euclidean geometry.
  euclid_tests <- list(
    list(title = "[Test E] WARD (RAM/Geometric Bound)", method = ml_linkage_method$Ward),
    list(title = "[Test F] Centroid Linkage (Generic Linkage)",
         method = ml_linkage_method$Centroid),
    list(title = "[Test G] Median Linkage (Generic Linkage)",
         method = ml_linkage_method$Median)
  )
  for (tst in euclid_tests) {
    cat(sprintf("\n%s:\n", tst$title))
    if (identical(tst$method, ml_linkage_method$Ward)) {
      cat(" -> Note: Switching to Squared Euclidean Distance!\n")
    }
    res <- cluster_and_score(XT, tst$method, ml_distance_metric$Euclidean,
                             dat$true_labels, k_clusters)
    cat(sprintf("Time: %.6f seconds\n", res$elapsed))
    cat(sprintf("ARI:  %.4f / 1.0000\n", res$ari))
  }

  cat("\n[Demo 4 Complete]\n\n")
  invisible(NULL)
}


# Demo 5: Impact of the LSH approximation across linkage methods.
#
# This demo illustrates the geometry of latent factors: dense variable blocks
# driven by hidden signals form distinct clusters, where the O(1) LSH SimHash
# approximation excels.
demo_lsh_linkage_comparison <- function() {
  print_section_header("Demo 5: LSH Approx Linkage Comparison")

  # cpp: N = 50,000, P = 100,000; scaled by 1/100 to N = 500, P = 1,000.
  n_samples <- 500L
  p_features <- 1000L
  k_clusters <- 5L
  block_sizes <- c(200L, 150L, 180L, 170L, 300L)
  centers     <- c(0.0, 1.0, -1.0, 5.0, -5.0)
  std_devs    <- c(1.5, 1.2, 0.8, 2.0, 3.5)

  cat(sprintf("1. Generating Imbalanced Correlated Variables (N=%d, P=%d)...\n",
              n_samples, p_features))
  dat <- generate_correlated_variables(n_samples, p_features, k_clusters,
                                       block_sizes, centers, std_devs,
                                       seed = 42L)

  cat("2. Standardizing variables (Mean=0, L2=1) for correlation...\n")
  XT <- t(standardize_for_correlation(dat$X))

  cat("3. Setting distance policy (Pure O(1) Correlation LSH Approx)...\n")
  tests <- list(
    list(title = "[Test A] Single Linkage (SLINK with LSH)",
         method = ml_linkage_method$Single),
    list(title = "[Test B] Complete Linkage (Disk/Matrix Bound)",
         method = ml_linkage_method$Complete),
    list(title = "[Test C] UPGMA (RAM/Geometric Bound)",
         method = ml_linkage_method$Average),
    list(title = "[Test D] WPGMA (Disk/Matrix Bound)",
         method = ml_linkage_method$WPGMA)
  )
  for (tst in tests) {
    cat(sprintf("\n%s:\n", tst$title))
    res <- cluster_and_score(XT, tst$method,
                             ml_distance_metric$Correlation_LSH_Approx,
                             dat$true_labels, k_clusters)
    cat(sprintf("Time: %.6f seconds\n", res$elapsed))
    cat(sprintf("ARI:  %.4f / 1.0000\n", res$ari))
  }

  cat("\n[Demo 5 Complete]\n\n")
  invisible(NULL)
}


# Demo 6: AR(1) Toeplitz correlation clustering.
#
# The geometry of AR(1) chains: correlation decays over index distance
# (variable 1 is close to 2, but far from 100).
#   - Algorithmic rule: Single Linkage (SLINK) perfectly chains this together;
#     Complete and Average linkage shatter the clusters.
#   - LSH rule: do not use LSH here — the 64-bit SimHash quantization destroys
#     the continuous correlation gradient. Use exact correlation.
demo_ar1_linkage_comparison <- function() {
  print_section_header("Demo 6: AR(1) Block Linkage Comparison")

  # cpp: N = 3,000, P = 50,000; scaled to N = 300, P = 500.
  n_samples <- 300L
  p_features <- 500L
  k_clusters <- 5L
  block_sizes <- c(200L, 100L, 80L, 70L, 50L)
  # rhos: correlation between strictly adjacent variables in each block.
  rhos <- c(0.95, 0.90, 0.85, 0.80, 0.75)

  cat(sprintf("1. Generating AR(1) Toeplitz Variables (N=%d, P=%d)...\n",
              n_samples, p_features))
  dat <- generate_ar1_block_variables(n_samples, p_features, k_clusters,
                                      block_sizes, rhos, seed = 42L)

  cat("2. Standardizing variables (Mean=0, L2=1) for correlation...\n")
  XT <- t(standardize_for_correlation(dat$X))

  cat("3. Setting distance policy (Exact Pearson Correlation)...\n")
  tests <- list(
    list(title = "[Test A] Single Linkage (SLINK)", method = ml_linkage_method$Single),
    list(title = "[Test B] Complete Linkage (Disk/Matrix Bound)",
         method = ml_linkage_method$Complete),
    list(title = "[Test C] UPGMA (RAM/Geometric Bound)",
         method = ml_linkage_method$Average)
  )
  for (tst in tests) {
    cat(sprintf("\n%s:\n", tst$title))
    res <- cluster_and_score(XT, tst$method, ml_distance_metric$Correlation,
                             dat$true_labels, k_clusters)
    cat(sprintf("Time: %.6f seconds\n", res$elapsed))
    cat(sprintf("ARI:  %.4f / 1.0000\n", res$ari))
  }

  cat("\n[Demo 6 Complete]\n\n")
  invisible(NULL)
}


# ==============================================================================
# 3. Main
# ==============================================================================

cat("\n")
print_section_header("Dendrograms Demo Suite")

# Six OpenMP threads, as the C++ demo
trex_set_num_threads(min(6L, trex_get_max_threads()))
cat(sprintf("Running with %d threads\n\n", trex_get_num_threads()))

# Each demo is guarded by its own if-toggle so individual parts can be
# switched off while iterating on one of them (mirrors the cpp main()).
if (TRUE) demo_slink_sample_clustering()
if (TRUE) demo_slink_variable_clustering()
if (TRUE) demo_slink_variable_clustering_lsh()
if (TRUE) demo_compare_linkage()
if (TRUE) demo_lsh_linkage_comparison()
if (TRUE) demo_ar1_linkage_comparison()

print_section_header("All clustering demos completed successfully")
