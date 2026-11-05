# ==============================================================================
# demo_mlm_pca_01.R
# ==============================================================================
#
# Demonstration of PCA usage.
#
# Mirrors cpp/ml_methods/pca/demo_mlm_pca_01/demo_mlm_pca_01.cpp.
#
# Three demos which illustrate the PCA interface:
#
#   Demo 1 | Basic fit     : n = 500, p = 200, M = 10 — score matrix Z,
#                            loading matrix V, and explained variance per
#                            retained component.
#
#   Demo 2 | Restore       : n = 300, p = 100, M = 5 — MLPCA$new() preprocesses
#                            X in place; $restore() must return it to its
#                            original state.
#
#   Demo 3 | Transform     : n_train = 400, n_test = 50, p = 80, M = 8 —
#                            project unseen observations onto the learned
#                            components and verify $transform() reproduces the
#                            fit scores on the training data.
#
# Notes on the port:
#   - In-place (zero-copy) design: MLPCA$new(X, ...) maps the memory of the R
#     matrix X directly, the same Eigen::Map interface that serves both
#     in-memory and memory-mapped data. This is deliberate — the library
#     targets data volumes that can exceed RAM, so it never takes hidden
#     copies of X. Consequently MLPCA$new() standardizes X in place and
#     $restore() undoes the preprocessing exactly. A plain alias (`X_orig <- X`) shares
#     the same buffer; where a pristine copy is needed, take an explicit deep
#     copy with `X_orig <- X + 0`.
#   - Same n / p / M values and seeds as the C++ demo; R's Mersenne-Twister
#     stream differs from std::mt19937, so results match qualitatively,
#     not bitwise.
#
# ==============================================================================

library(TRexSelector)


print_section_header <- function(title) {
  cat(strrep("=", 78), "\n", sep = "")
  cat(" ", title, "\n", sep = "")
  cat(strrep("=", 78), "\n\n", sep = "")
}


# ==============================================================================
# Demo 1: Basic PCA Fit
# ==============================================================================

demo_pca_fit <- function() {
  print_section_header("PCA Basic Fit")

  # Step 1. Generate random data
  n <- 500L
  p <- 200L
  M <- 10L
  cat(sprintf("Matrix size  : %d x %d\n", n, p))
  cat(sprintf("Components M : %d\n\n", M))

  set.seed(42)
  X <- matrix(rnorm(n * p), nrow = n, ncol = p)

  # Step 2. Construct and fit. center = TRUE subtracts the column means,
  #         normalize = TRUE scales the columns; the constructor does both
  #         in place on X, and $restore() undoes them.
  pca <- MLPCA$new(X, M = M, center = TRUE, normalize = TRUE)
  pca$fit()

  # Step 3. Print result dimensions. Z holds the principal-component scores:
  #         each row is an observation expressed in the M-dimensional
  #         component space. V holds the loadings: each column is a
  #         unit-length direction in the original p-dimensional space.
  Z  <- pca$get_scores()
  V  <- pca$get_loadings()
  ev <- pca$get_explained_variance()

  cat(sprintf("Z dimensions : %d x %d\n", nrow(Z), ncol(Z)))
  cat(sprintf("V dimensions : %d x %d\n", nrow(V), ncol(V)))
  cat(sprintf("explained_variance size: %d\n\n", length(ev)))

  # Step 4. Explained variance as a share of the total variance of the
  #         preprocessed X (MLPCA$new() centered and scaled X in place)
  total_var <- sum(X^2) / (n - 1)
  pct <- 100 * ev[seq_len(M)] / total_var
  cat("Explained variance per component (% of total):\n")
  for (k in seq_len(M)) {
    cat(sprintf("  PC%d: %.2f%%\n", k, pct[k]))
  }
  cat(sprintf("  Cumulative (%d of %d): %.2f%%\n", M, p, sum(pct)))

  # Didactic check: components are ordered by decreasing explained variance.
  stopifnot(all(diff(ev) <= 1e-12))
  cat("✓ Explained variance is non-increasing across PCs\n")

  # Restore X to its original (pre-preprocessing) state
  pca$restore()
  cat("\n\n")

  invisible(NULL)
}


# ==============================================================================
# Demo 2: Restore Round-Trip Accuracy: Data -> PCA -> Restore -> Data
# ==============================================================================

demo_pca_restore <- function() {
  print_section_header("PCA Restore Round-Trip Accuracy")

  # Step 1. Generate random data
  n <- 300L
  p <- 100L
  M <- 5L
  cat(sprintf("Matrix size  : %d x %d\n", n, p))
  cat(sprintf("Components M : %d\n\n", M))

  set.seed(123)
  X <- matrix(rnorm(n * p), nrow = n, ncol = p)

  # Step 2. Forced deep copy (`+ 0`): a plain `X_orig <- X` would alias the
  #         same buffer and be mutated together with X by MLPCA$new().
  X_orig <- X + 0

  pca <- MLPCA$new(X, M = M, center = TRUE, normalize = TRUE)
  pca$fit()

  # X holds the standardized data since MLPCA$new() (in-place preprocessing).
  cat(sprintf("Max |X - X_orig| after fit()     : %.4e  (X standardized in place)\n",
              max(abs(X - X_orig))))

  # Step 3. Restore and measure reconstruction error. $restore() undoes the
  #         centering / normalization. The point is interface correctness (a
  #         round-trip on the preprocessed training matrix), not low-rank
  #         approximation error.
  pca$restore()

  max_error  <- max(abs(X - X_orig))
  mean_error <- mean(abs(X - X_orig))

  cat("\nReconstruction error after restore():\n")
  cat(sprintf("  Max Error  : %.4e\n", max_error))
  cat(sprintf("  Mean Error : %.4e\n\n", mean_error))

  if (max_error < 1e-10) {
    cat("✓ Restore round-trip passed\n")
  } else {
    cat("✗ Restore round-trip FAILED\n")
  }
  cat("\n\n")

  invisible(NULL)
}


# ==============================================================================
# Demo 3: Transform New Data: Data -> PCA -> Transform -> Scores
# ==============================================================================

demo_pca_transform <- function() {
  print_section_header("PCA Transform New Data")

  # Step 1. Generate train and test data
  n_train <- 400L
  n_test  <- 50L
  p       <- 80L
  M       <- 8L
  cat(sprintf("Train size   : %d x %d\n", n_train, p))
  cat(sprintf("Test size    : %d x %d\n", n_test, p))
  cat(sprintf("Components M : %d\n\n", M))

  set.seed(7)
  X_train <- matrix(rnorm(n_train * p), nrow = n_train, ncol = p)
  # Keep a pristine copy before PCA preprocesses X_train in place
  X_train_orig <- X_train + 0
  X_test <- matrix(rnorm(n_test * p), nrow = n_test, ncol = p)

  # Step 2. Fit PCA on training data
  pca <- MLPCA$new(X_train, M = M, center = TRUE, normalize = TRUE)
  pca$fit()
  Z_train <- pca$get_scores()

  cat(sprintf("Train scores Z dimensions: %d x %d\n", nrow(Z_train), ncol(Z_train)))

  # Step 3. Transform test data — $transform() applies the training-set
  #         centering / normalization to new data and projects it onto the
  #         learned loadings; no refitting takes place.
  Z_test <- pca$transform(X_test)

  cat(sprintf("Test  scores Z dimensions: %d x %d\n\n", nrow(Z_test), ncol(Z_test)))

  if (nrow(Z_test) == n_test && ncol(Z_test) == M) {
    cat(sprintf("✓ Transform output shape correct (%d x %d)\n", n_test, M))
  } else {
    cat("✗ Transform output shape FAILED\n")
  }

  # Step 4. Verify $transform() on the original training data reproduces the
  #         fit scores
  Z_train_check <- pca$transform(X_train_orig)
  max_diff <- max(abs(Z_train_check - Z_train))
  cat(sprintf("Max diff (fit scores vs transform on same data): %.4e\n", max_diff))

  if (max_diff < 1e-10) {
    cat("✓ transform() reproduces fit scores\n")
  } else {
    cat("✗ transform() mismatch\n")
  }

  # Restore X_train
  pca$restore()
  cat("\n\n")

  invisible(NULL)
}


# ==============================================================================
# Main
# ==============================================================================

demo_pca_fit()
demo_pca_restore()
demo_pca_transform()
