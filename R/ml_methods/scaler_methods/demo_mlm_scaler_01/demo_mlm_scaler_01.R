# ==============================================================================
# demo_mlm_scaler_01.R
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
#   Demo 3 | Serialization      : $save() a fitted ZScoreScaler to disk,
#                                 $load() it into a fresh object, compare the
#                                 stored parameters.
#
#   Demo 4 | Memory-mapped      : L2 LpNormScaler fit / transform / inverse
#                                 transform directly on the mmap_matrix handle
#                                 (zero-copy through the same Eigen map
#                                 interface as the C++ demo).
#                                 (cpp: n = 20,000, p = 100,000, filled with an
#                                 OpenMP parallel loop; scaled to n = 1,000,
#                                 p = 5,000 here and filled column-wise.)
#
# Notes on the port:
#   - The `*_inplace()` methods mutate a double-storage R matrix directly
#     (zero-copy) and also return it. A plain alias (`X2 <- X`) therefore
#     shares the same buffer; take an explicit deep copy with `X + 0` where a
#     pristine matrix is needed afterwards.
#   - Passing an mmap_matrix handle to the scaler binds the disk-backed buffer
#     zero-copy; the scaler methods (including $fit()) require the handle to
#     be open in "readwrite" mode.
#   - Same n / p values and seeds as the C++ demo where the sizes are shared;
#     R's Mersenne-Twister stream differs from std::mt19937, so results match
#     qualitatively, not bitwise.
#
# ==============================================================================

library(TRexSelector)


print_section_header <- function(title) {
  cat(strrep("=", 78), "\n", sep = "")
  cat(" ", title, "\n", sep = "")
  cat(strrep("=", 78), "\n\n", sep = "")
}

# Compact per-column statistics, averaged over all columns.
#
# Reveals the effect of the center / scale switches at a glance:
#   - centering drives the average absolute column mean toward 0,
#   - z-score scaling drives the average column SD toward 1,
#   - L2 normalization drives the average column L2 norm toward 1.
print_column_stats <- function(label, X) {
  col_mean <- colMeans(X)
  col_sd   <- apply(X, 2L, sd)
  col_l2   <- sqrt(colSums(X^2))
  cat(sprintf("  %-36savg|mean| = %.2e   avg SD = %.4f   avg L2 = %.2f\n",
              label, mean(abs(col_mean)), mean(col_sd), mean(col_l2)))
}


# ==============================================================================
# Demo 1: center / scale switches (R scale() semantics)
# ==============================================================================

demo_scaler_comparison <- function() {
  print_section_header("Scaler Comparison - center / scale switches")

  # Step 1. Generate data with a deliberately non-zero mean (5.0) and
  #         non-unit spread (SD 3.0), so the effect of each switch is visible.
  n <- 1000L
  p <- 500L
  set.seed(123)
  X <- matrix(rnorm(n * p, mean = 5.0, sd = 3.0), nrow = n, ncol = p)

  cat(sprintf("Data: %d x %d drawn from N(mean=5, sd=3)\n\n", n, p))
  print_column_stats("original (untransformed)", X)
  cat("\n")

  # Step 2. ZScoreScaler under all four center / scale combinations.
  #         The (center, scale) flags follow R's scale(): centering drives
  #         avg|mean| -> 0, scaling drives avg SD -> 1. With center = FALSE the
  #         scale statistic is the root-mean-square around 0, not the SD.
  z_cases <- list(
    list(center = TRUE,  scale = TRUE,  label = "ZScore(center=TRUE,  scale=TRUE )"),
    list(center = FALSE, scale = TRUE,  label = "ZScore(center=FALSE, scale=TRUE )"),
    list(center = TRUE,  scale = FALSE, label = "ZScore(center=TRUE,  scale=FALSE)"),
    list(center = FALSE, scale = FALSE, label = "ZScore(center=FALSE, scale=FALSE)")
  )
  for (case in z_cases) {
    Xt <- X + 0                      # forced deep copy: the transform is in-place
    zscaler <- MLZScoreScaler$new(center = case$center, scale = case$scale)
    zscaler$fit(Xt)
    Xt <- zscaler$transform_inplace(Xt)
    print_column_stats(case$label, Xt)
  }
  cat("\n")

  # Step 3. LpNormScaler: L2 vs L1, with centering on and off (scale kept on).
  #         With scale = TRUE each column is divided by its Lp norm around the
  #         applied center, so the average column L2 norm collapses toward 1
  #         for the L2 variant.
  l_cases <- list(
    list(norm = 2L, center = TRUE,  label = "LpNorm(L2, center=TRUE )"),
    list(norm = 2L, center = FALSE, label = "LpNorm(L2, center=FALSE)"),
    list(norm = 1L, center = TRUE,  label = "LpNorm(L1, center=TRUE )")
  )
  for (case in l_cases) {
    Xt <- X + 0
    lscaler <- MLLpNormScaler$new(norm_type = case$norm, center = case$center,
                                  scale = TRUE)
    lscaler$fit(Xt)
    Xt <- lscaler$transform_inplace(Xt)
    print_column_stats(case$label, Xt)
  }

  cat("\n\n")
  invisible(NULL)
}


# ==============================================================================
# Demo 2: Transform and Inverse Transform Test
# ==============================================================================

demo_scaler_inverse_transform <- function() {
  print_section_header("Scaler Inverse Transform Accuracy")

  # Step 1. Generate random data
  n <- 1000L
  p <- 5000L
  cat(sprintf("Matrix size: %d x %d\n\n", n, p))

  set.seed(456)
  X_orig <- matrix(rnorm(n * p), nrow = n, ncol = p)

  # Step 2. Test ZScoreScaler
  X <- X_orig + 0
  zscaler <- MLZScoreScaler$new()
  zscaler$fit(X)
  X <- zscaler$transform_inplace(X)
  X <- zscaler$inverse_transform_inplace(X)

  err <- abs(X - X_orig)
  cat("ZScoreScaler:\n")
  cat(sprintf("  Max Error: %e\n", max(err)))
  cat(sprintf("  Mean Error: %e\n\n", mean(err)))

  # Step 3. Test LpNormScaler L2
  X <- X_orig + 0
  l2scaler <- MLLpNormScaler$new(norm_type = 2L, center = TRUE)
  l2scaler$fit(X)
  X <- l2scaler$transform_inplace(X)
  X <- l2scaler$inverse_transform_inplace(X)

  err <- abs(X - X_orig)
  cat("LpNormScaler L2:\n")
  cat(sprintf("  Max Error: %e\n", max(err)))
  cat(sprintf("  Mean Error: %e\n\n", mean(err)))
  cat("\n\n")

  invisible(NULL)
}


# ==============================================================================
# Demo 3: Serialization
# ==============================================================================

demo_scaler_serialization <- function() {
  print_section_header("Scaler Serialization")

  # Step 1. Generate data
  n <- 1000L
  p <- 500L
  set.seed(999)
  X <- matrix(rnorm(n * p), nrow = n, ncol = p)

  # Step 2. Fit and save (cpp: fallback filename "scaler_test.bin"; a
  #         temporary file in the session temp directory is used here)
  filename <- tempfile(pattern = "trex_scaler_demo_", fileext = ".bin")
  on.exit(unlink(filename), add = TRUE)

  scaler <- MLZScoreScaler$new()
  scaler$fit(X)
  scaler$save(filename)
  cat(sprintf("✓ Saved ZScoreScaler to '<tmp>/%s'\n", basename(filename)))

  # Step 3. Load and verify
  loaded_scaler <- MLZScoreScaler$new()
  loaded_scaler$load(filename)
  cat(sprintf("✓ Loaded ZScoreScaler from '<tmp>/%s'\n\n", basename(filename)))

  # Verify centers and scales match
  mean_diff  <- sqrt(sum((scaler$get_centers() - loaded_scaler$get_centers())^2))
  scale_diff <- sqrt(sum((scaler$get_scales()  - loaded_scaler$get_scales())^2))

  cat("Verification:\n")
  cat(sprintf("  Mean difference: %e\n", mean_diff))
  cat(sprintf("  Scale difference: %e\n\n", scale_diff))

  if (mean_diff < 1e-10 && scale_diff < 1e-10) {
    cat("✓ Serialization perfect!\n\n")
  }

  cat("\n\n")
  invisible(NULL)
}


# ==============================================================================
# Demo 4: Memory-Mapped Matrix
# ==============================================================================

demo_scaler_on_mmap_matrix <- function() {
  print_section_header("Scaler on Memory-Mapped Matrix")

  # Step 1. Create memory-mapped matrix
  #         (cpp: n = 20,000, p = 100,000; scaled to 1,000 x 5,000 here)
  cat("Creating memory-mapped matrix...\n")
  n <- 1000L
  p <- 5000L
  cat(sprintf("Matrix size: %d x %d\n", n, p))

  mmap_filename <- tempfile(pattern = "trex_scaler_mmap_", fileext = ".bin")
  on.exit({
    unlink(mmap_filename)
    cat("✓ Memory-mapped file removed\n\n\n")
  }, add = TRUE)

  X_mmap <- mem_matrix(mmap_filename, rows = n, cols = p)

  # Step 2. Populate the file column by column: X(i, j) = 2*i + j/100
  #         (0-based i, j — matching the C++ demo, which fills the mapped
  #         buffer with an OpenMP parallel loop; that part is C++-exclusive)
  rows_0based <- as.double(seq_len(n) - 1L)
  for (j in seq_len(p)) {
    X_mmap[1:n, j] <- 2.0 * rows_0based + (j - 1L) / 100.0
  }

  # Step 3. Fit and run LpNormScaler directly on the mmap handle. The scaler
  #         binds the disk-backed buffer zero-copy through the same Eigen map
  #         interface as the C++ demo — no block is materialized in R memory.
  l2scaler <- MLLpNormScaler$new(norm_type = 2L, center = TRUE)
  cat("Fitting L2 LpNormScaler on memory-mapped matrix...\n")
  l2scaler$fit(X_mmap)
  cat("✓ LpNormScaler fitted\n")

  l2scaler$transform_inplace(X_mmap)
  cat("✓ Data transformed (in place, on disk)\n")

  l2scaler$inverse_transform_inplace(X_mmap)
  cat("✓ Data inverse transformed\n")

  # Round-trip check against the analytic fill pattern
  max_err <- max(abs(X_mmap[1:n, 1] - (2.0 * rows_0based)))
  cat(sprintf("  Round-trip max|diff| on column 1: %e\n", max_err))

  # Step 4. Clean up (handled by on.exit)
  invisible(NULL)
}


# ==============================================================================
# Main
# ==============================================================================

cat("\n")
print_section_header("DataTransformer Demo Suite")

# Six OpenMP threads, as the C++ demo
trex_set_num_threads(min(6L, trex_get_max_threads()))
cat(sprintf("Running with %d threads\n\n", trex_get_num_threads()))

demo_scaler_comparison()
demo_scaler_inverse_transform()
demo_scaler_serialization()
demo_scaler_on_mmap_matrix()

cat("\n")
print_section_header("All Demos Completed Successfully")
