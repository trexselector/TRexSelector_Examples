# ==============================================================================
# demo_mlm_ms_01_ridge_cv_svd.R
# ==============================================================================
#
# Demo of Ridge Regression K-Fold Cross-Validation via the SVD-based RidgeCV
# (per-fold centering and column L2-normalization).
#
# Mirrors cpp/ml_methods/model_selection/demo_mlm_ms_01_ridge_cv_svd/
#         demo_mlm_ms_01_ridge_cv_svd.cpp.
#
# Scenarios:
#
#   Demo 1 | Low-dimensional  : n=1000, p=500  (n > p,  10-fold)
#   Demo 2 | High-dimensional : n=300,  p=500  (p > n,  10-fold)
#   Demo 3 | Very high-dim    : n=300,  p=1000 (p >> n,  5-fold)
#   Demo 4 | Memory-mapped    : n=5000, p=500  — the design matrix lives on
#                               disk in an mmap file, is re-opened read-only,
#                               and is bound zero-copy for the fit; grid
#                               widened to a 1e5 range (n >> p), as in cpp.
#
# Algorithm notes:
#   RidgeCV runs an SVD per fold with per-fold centering and column
#   L2-normalization. The lambda grid is anchored at the largest useful
#   penalty and spans a `lambda_ratio` range; `seed` fixes the fold assignment.
#
# Notes on the port:
#   - Same n / p / fold values and data seeds as the C++ demo; R's
#     Mersenne-Twister stream differs from std::mt19937, so results match
#     qualitatively, not bitwise.
#   - In Demo 4 the mmap_matrix handle is passed straight to $fit(), which
#     binds the disk-backed buffer zero-copy through a const Eigen map —
#     "readonly" mode suffices, exactly as in the C++ demo. Internally each
#     fold materializes dense X_train / X_test sub-matrices in RAM, so there
#     is no out-of-core streaming path in either port.
#
# ==============================================================================

library(TRexSelector)


print_section_header <- function(title) {
  cat(strrep("=", 78), "\n", sep = "")
  cat(" ", title, "\n", sep = "")
  cat(strrep("=", 78), "\n\n", sep = "")
}


# ==============================================================================
# Shared data-generation helper
# ==============================================================================

# Synthetic regression data with known sparse signal: n_active coefficients
# equal to 1.0 at evenly spaced column indices, y = X beta_true + N(0, sigma^2).
# Mirrors the C++ generate_regression_data() helper.
generate_regression_data <- function(n, p, sigma, seed, n_active = 10L) {
  set.seed(seed)
  X <- matrix(rnorm(n * p), nrow = n, ncol = p)
  beta_true <- numeric(p)
  stride <- max(1L, p %/% n_active)
  beta_true[1L + (seq_len(n_active) - 1L) * stride] <- 1.0
  y <- as.numeric(X %*% beta_true) + sigma * rnorm(n)
  list(X = X, y = y, beta_true = beta_true)
}


# ==============================================================================
# Shared print helper
# ==============================================================================

# Condensed summary of a fitted RidgeCV object:
#   - lambda.min: grid value minimizing the CV mean-squared error
#   - lambda.1se: largest lambda whose CV-MSE is within one standard error
#     of the minimum (more parsimonious / more heavily regularized choice)
# The lambda grid is stored in ascending order; the curve is printed every
# `step_size` grid points with the min/1se positions marked.
print_cv_summary <- function(cv, step_size = 10L) {
  lambdas <- cv$get_lambdas()
  mse     <- cv$get_cv_errors()
  sem     <- cv$get_cv_std()
  K       <- length(lambdas)
  i_min   <- cv$index_min()
  i_1se   <- cv$index_1se()

  cat(sprintf("  lambda.min : %.4f  (CV-MSE=%.4f +/- %.4f)\n",
              cv$cv_min(), mse[i_min], sem[i_min]))
  cat(sprintf("  lambda.1se : %.4f  (CV-MSE=%.4f +/- %.4f)\n\n",
              cv$cv_1se(), mse[i_1se], sem[i_1se]))

  cat(sprintf("  CV-MSE curve (every %d steps, lambda ascending):\n", step_size))
  cat(sprintf("    %-14s%-12s%-12s%s\n", "lambda", "CV-MSE", "SEM", "marker"))

  print_row <- function(k, marker) {
    cat(sprintf("    %-14.4f%-12.4f%-12.4f%s\n",
                lambdas[k], mse[k], sem[k], marker))
  }
  shown <- seq(1L, K, by = step_size)
  for (k in shown) {
    marker <- if (k == i_min) "  <-- min" else if (k == i_1se) "  <-- 1se" else ""
    print_row(k, marker)
  }
  if (!(i_min %in% shown)) print_row(i_min, "  <-- min")
  if (!(i_1se %in% shown) && i_1se != i_min) print_row(i_1se, "  <-- 1se")
  cat("\n")
}

# Fit RidgeCV, print the summary, and run the two checks: the ordering
# lambda.1se >= lambda.min (the 1se rule always regularizes at least as
# strongly as the pure minimizer), and both picks inside the grid.
fit_and_report <- function(X, y, n_folds, seed, lambda_ratio = 1000) {
  # Grid parameters mirror the C++ fit(): a 100-lambda grid anchored at the
  # largest useful penalty, spanning [lambda_max / lambda_ratio, lambda_max];
  # `seed` fixes the fold assignment.
  cv <- MLRidgeCV$new()
  cv$fit(X, y, n_folds = n_folds, n_lambda = 100L,
         lambda_ratio = lambda_ratio, seed = seed)

  print_cv_summary(cv, step_size = 10L)

  order_ok <- cv$cv_min() <= cv$cv_1se()
  cat(sprintf("  Ordering check (lambda.min <= lambda.1se): %s\n",
              if (order_ok) "PASS" else "FAIL"))
  # A pick on the first or last grid point is the grid boundary, not a minimum
  K <- length(cv$get_lambdas())
  at_edge <- any(c(cv$index_min(), cv$index_1se()) %in% c(1L, K))
  cat(sprintf("  Grid-edge check (both picks inside the grid): %s\n\n",
              if (at_edge) "FAIL" else "PASS"))
  stopifnot(order_ok)
  invisible(cv)
}


# ==============================================================================
# Demo 1: Low-dimensional scenario (n=1000, p=500, n > p, 10-fold)
# ==============================================================================

demo_kfold_low_dim <- function() {
  print_section_header(
    "Demo MS-01 | ridge_cv_svd | Low-Dimensional (n=1000, p=500, 10-fold)"
  )

  n <- 1000L
  p <- 500L
  sigma <- 1.0
  n_folds <- 10L

  dat <- generate_regression_data(n, p, sigma, seed = 111L)
  cat(sprintf("Data: n=%d, p=%d, folds=%d, active coefs=%d\n\n",
              n, p, n_folds, sum(dat$beta_true != 0)))

  fit_and_report(dat$X, dat$y, n_folds, seed = 111L)
  invisible(NULL)
}


# ==============================================================================
# Demo 2: High-dimensional scenario (n=300, p=500, p > n, 10-fold)
# ==============================================================================

demo_kfold_high_dim <- function() {
  print_section_header(
    "Demo MS-01 | ridge_cv_svd | High-Dimensional (n=300, p=500, 10-fold)"
  )

  n <- 300L
  p <- 500L
  sigma <- 1.0
  n_folds <- 10L

  dat <- generate_regression_data(n, p, sigma, seed = 222L)
  cat(sprintf("Data: n=%d, p=%d, folds=%d, active coefs=%d\n\n",
              n, p, n_folds, sum(dat$beta_true != 0)))

  fit_and_report(dat$X, dat$y, n_folds, seed = 222L)
  invisible(NULL)
}


# ==============================================================================
# Demo 3: Very high-dimensional scenario (n=300, p=1000, p >> n, 5-fold)
# ==============================================================================

demo_kfold_very_high_dim <- function() {
  print_section_header(
    "Demo MS-01 | ridge_cv_svd | Very High-Dimensional (n=300, p=1000, 5-fold)"
  )

  n <- 300L
  p <- 1000L
  sigma <- 1.0
  n_folds <- 5L

  dat <- generate_regression_data(n, p, sigma, seed = 333L)
  cat(sprintf("Data: n=%d, p=%d, folds=%d, active coefs=%d\n",
              n, p, n_folds, sum(dat$beta_true != 0)))
  cat("  Note: 5 folds used to ensure each fold's training set is large ")
  cat("enough relative to n.\n\n")

  fit_and_report(dat$X, dat$y, n_folds, seed = 333L)
  invisible(NULL)
}


# ==============================================================================
# Demo 4: Memory-mapped scenario (n=5000, p=500, 10-fold)
# ==============================================================================

demo_kfold_mmap <- function() {
  print_section_header(
    "Demo MS-01 | ridge_cv_svd | Memory-Mapped Dataset (n=5000, p=500, 10-fold)"
  )

  n <- 5000L
  p <- 500L
  sigma <- 1.0
  n_folds <- 10L
  # n >> p: the CV optimum lies below lambda_max / 1000, so the grid is widened
  lambda_ratio <- 1e5

  mmap_path <- tempfile(pattern = "trex_mlm_demo_ms01_svd_mmap_",
                        fileext = ".bin")
  on.exit({
    if (file.exists(mmap_path)) {
      unlink(mmap_path)
      cat(sprintf("Cleaned up mmap file: <tmp>/%s\n\n", basename(mmap_path)))
    }
  }, add = TRUE)

  dat <- generate_regression_data(n, p, sigma, seed = 444L)

  file_bytes <- as.numeric(n) * as.numeric(p) * 8
  cat(sprintf("Writing X to mmap file: <tmp>/%s\n", basename(mmap_path)))
  cat(sprintf("  File size: %.2f MB\n", file_bytes / 2^20))
  cat(sprintf("  Active coefs: %d\n\n", sum(dat$beta_true != 0)))

  X_mmap <- mem_matrix_from(dat$X, mmap_path)
  rm(X_mmap)  # release the writer handle before re-opening

  # Re-open the file read-only and pass the handle straight to $fit(): the
  # binding binds the disk-backed buffer zero-copy through a const Eigen map,
  # mirroring the cpp demo's ConstMapType / Eigen::Ref path.
  cat(sprintf("Re-opening mmap read-only and fitting RidgeCV (%d-fold)...\n\n",
              n_folds))
  X_ro <- mem_matrix(mmap_path, rows = n, cols = p, mode = "readonly")

  fit_and_report(X_ro, dat$y, n_folds, seed = 444L, lambda_ratio = lambda_ratio)
  invisible(NULL)
}


# ==============================================================================
# Main
# ==============================================================================

print_section_header("ridge_cv_svd Demo Suite  (demo_mlm_ms_01_ridge_cv_svd)")

demo_kfold_low_dim()
demo_kfold_high_dim()
demo_kfold_very_high_dim()
demo_kfold_mmap()

cat("\nAll ridge_cv_svd demos complete.\n\n")
