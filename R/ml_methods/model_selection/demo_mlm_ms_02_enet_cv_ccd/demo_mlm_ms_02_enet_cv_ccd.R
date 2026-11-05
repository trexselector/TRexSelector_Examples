# ==============================================================================
# demo_mlm_ms_02_enet_cv_ccd.R
# ==============================================================================
#
# Demo of coordinate-descent elastic-net path fitting (ElasticNet) and K-fold
# cross-validation (ElasticNetCV), both glmnet-style pathwise CCD.
#
# Mirrors cpp/ml_methods/model_selection/demo_mlm_ms_02_enet_cv_ccd/
#         demo_mlm_ms_02_enet_cv_ccd.cpp.
#
# Scenarios covered:
#
#   Part A — ElasticNet (path fit):
#     A1. Auto-grid path for alpha = {0.0 (ridge), 0.5 (EN), 1.0 (lasso)},
#         low-dimensional (n=300, p=100, n > p).
#     A2. High-dimensional path (n=200, p=500, p > n) for alpha = 0.0 and 1.0.
#
#   Part B — ElasticNetCV (K-fold CV):
#     B1. CV lambda selection (lambda.min / lambda.1se) for each alpha,
#         low-dimensional (n=300, p=100, 10-fold).
#     B2. CV for pure ridge (alpha=0) as used by TRexGVS; shows the
#         lambda_2_lars = lambda.{min,1se} * p / 2 conversion for both
#         selections.
#
# Algorithm notes:
#   ElasticNet replicates glmnet's pathwise CCD with glmnet's fdev/devmax
#   early termination. ElasticNetCV builds the full-data lambda grid once and
#   reuses it across folds (glmnet semantics). For alpha=0 (pure ridge) the
#   returned lambdas are on glmnet's reported scale; the T-Rex conversion is:
#   lambda_2_lars = lambda.{min,1se} * p / 2.
#
# Notes on the port:
#   - Same n / p / alpha values and data seeds as the C++ demo; R's
#     Mersenne-Twister stream differs from std::mt19937, so results match
#     qualitatively, not bitwise.
#   - Part A adds two checks the C++ demo does not print: the deviance ratio
#     must be non-decreasing along the path, and $fit_grid() evaluated at the
#     auto grid must reproduce the auto path exactly.
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
# Mirrors the C++ generate_data() helper.
generate_data <- function(n, p, sigma, seed, n_active = 10L) {
  set.seed(seed)
  X <- matrix(rnorm(n * p), nrow = n, ncol = p)
  beta_true <- numeric(p)
  stride <- max(1L, p %/% n_active)
  beta_true[1L + (seq_len(n_active) - 1L) * stride] <- 1.0
  y <- as.numeric(X %*% beta_true) + sigma * rnorm(n)
  list(X = X, y = y, beta_true = beta_true)
}


# ==============================================================================
# Print helpers
# ==============================================================================

# Condensed coefficient path from an ElasticNet fit: every step_size-th lambda
# with the non-zero count (l0), the L1-norm of the coefficient vector, and the
# deviance ratio (glmnet's dev.ratio, rising as lambda decreases).
print_path_summary <- function(en, step_size = 10L) {
  lam <- en$get_lambdas()
  B   <- en$get_coef()           # p x K coefficient path
  dev <- en$get_dev_ratio()
  K   <- length(lam)

  cat(sprintf("  Path: %d lambdas fitted%s\n",
              K, if (en$converged()) "" else " [non-converged]"))
  cat(sprintf("  %-12s%-8s%-12s%-10s\n", "lambda", "nnz", "L1(beta)", "dev.ratio"))

  print_row <- function(k) {
    cat(sprintf("  %-12.5f%-8d%-12.5f%-10.5f\n",
                lam[k], sum(abs(B[, k]) > 1e-9), sum(abs(B[, k])), dev[k]))
  }
  for (k in seq(1L, K, by = step_size)) print_row(k)
  if ((K - 1L) %% step_size != 0L) print_row(K)   # always show the last point
  cat("\n")
}

# Condensed summary of a fitted ElasticNetCV object:
#   - lambda.min: grid value minimizing the CV mean-squared error
#   - lambda.1se: largest lambda whose CV-MSE is within one standard error
#     of the minimum (more parsimonious / more heavily regularized choice)
# The lambda grid is stored in descending order (glmnet convention); the curve
# is printed every `step_size` grid points with the min/1se positions marked.
print_cv_summary <- function(cv, step_size = 10L) {
  lambdas <- cv$get_lambdas()
  mse     <- cv$get_cv_errors()
  sem     <- cv$get_cv_std()
  K       <- length(lambdas)
  i_min   <- cv$index_min()
  i_1se   <- cv$index_1se()

  cat(sprintf("  lambda.min : %.5f  (CV-MSE=%.5f +/- %.5f)\n",
              cv$cv_min(), mse[i_min], sem[i_min]))
  cat(sprintf("  lambda.1se : %.5f  (CV-MSE=%.5f +/- %.5f)\n\n",
              cv$cv_1se(), mse[i_1se], sem[i_1se]))

  cat(sprintf("  CV-MSE curve (every %d steps, lambda descending):\n", step_size))
  cat(sprintf("    %-14s%-12s%-12s%s\n", "lambda", "CV-MSE", "SEM", "marker"))

  print_row <- function(k, marker) {
    cat(sprintf("    %-14.5f%-12.5f%-12.5f%s\n",
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

# Fit an auto-grid path per alpha, print the summary, and check that (a) the
# deviance ratio is non-decreasing and (b) $fit_grid() at the auto grid
# reproduces the auto path exactly.
fit_path_suite <- function(dat, alphas) {
  for (alpha in alphas) {
    # alpha=1 is the pure L1 lasso (sparse), alpha=0 the pure L2 ridge (dense),
    # in between the elastic net.
    cat(sprintf("--- alpha=%.1f ---\n", alpha))
    en <- MLElasticNet$new()$fit(dat$X, dat$y, alpha)
    print_path_summary(en, step_size = 10L)

    stopifnot(all(diff(en$get_dev_ratio()) >= -1e-12))
    en_grid  <- MLElasticNet$new()$fit_grid(dat$X, dat$y, en$get_lambdas(), alpha)
    max_diff <- max(abs(en$get_coef() - en_grid$get_coef()))
    stopifnot(max_diff < 1e-12)
    cat(sprintf("  fit_grid() reproduction check (max |coef diff| = %.2e): PASS\n\n",
                max_diff))
  }
}


# ==============================================================================
# Part A — ElasticNet path demos
# ==============================================================================

# A1: Auto-grid paths for alpha = 0.0 / 0.5 / 1.0, low-dim (n=300, p=100).
demo_path_low_dim <- function() {
  print_section_header(
    "Demo MS-02 | enet_gaussian path | Low-Dimensional (n=300, p=100)"
  )

  n <- 300L
  p <- 100L
  sigma <- 1.0

  dat <- generate_data(n, p, sigma, seed = 101L)
  cat(sprintf("Data: n=%d, p=%d, active coefs=%d\n\n",
              n, p, sum(dat$beta_true != 0)))

  fit_path_suite(dat, alphas = c(0.0, 0.5, 1.0))
  invisible(NULL)
}

# A2: Auto-grid path, high-dimensional (n=200, p=500, p > n).
demo_path_high_dim <- function() {
  print_section_header(
    "Demo MS-02 | enet_gaussian path | High-Dimensional (n=200, p=500)"
  )

  n <- 200L
  p <- 500L
  sigma <- 1.0

  dat <- generate_data(n, p, sigma, seed = 202L)
  cat(sprintf("Data: n=%d, p=%d, active coefs=%d\n\n",
              n, p, sum(dat$beta_true != 0)))

  fit_path_suite(dat, alphas = c(0.0, 1.0))
  invisible(NULL)
}


# ==============================================================================
# Part B — ElasticNetCV demos
# ==============================================================================

# B1: CV lambda selection for alpha = 0.0 / 0.5 / 1.0, low-dim (n=300, p=100).
demo_cv_low_dim <- function() {
  print_section_header(
    "Demo MS-02 | enet_cv_ccd | Low-Dimensional CV (n=300, p=100, 10-fold)"
  )

  n <- 300L
  p <- 100L
  sigma <- 1.0
  n_folds <- 10L

  dat <- generate_data(n, p, sigma, seed = 303L)
  cat(sprintf("Data: n=%d, p=%d, folds=%d, active coefs=%d\n\n",
              n, p, n_folds, sum(dat$beta_true != 0)))

  for (alpha in c(0.0, 0.5, 1.0)) {
    cat(sprintf("--- alpha=%.1f ---\n", alpha))
    cv <- MLElasticNetCV$new()
    cv$fit(dat$X, dat$y, alpha = alpha, n_folds = n_folds, seed = 0L)
    print_cv_summary(cv, step_size = 10L)

    # The CV curve is evaluated at every grid point (the grid may hold fewer
    # than n_lambda points when glmnet's early termination kicks in).
    stopifnot(length(cv$get_cv_errors()) == length(cv$get_lambdas()))
    stopifnot(length(cv$get_cv_std())    == length(cv$get_lambdas()))

    order_ok <- cv$cv_1se() >= cv$cv_min()
    cat(sprintf("  Ordering check (lambda.min <= lambda.1se): %s\n\n",
                if (order_ok) "PASS" else "FAIL"))
    stopifnot(order_ok)
  }
  invisible(NULL)
}

# B2: Pure-ridge CV (alpha=0) with the TRexGVS lambda_2 conversion.
#     TRexGVS converts the glmnet-scale lambda via
#       lambda_2_lars = lambda.1se * p / 2,
#     mirroring R's `lm_dummy` and `cv.glmnet(alpha=0)`.
demo_cv_ridge_lars_conversion <- function() {
  print_section_header(
    "Demo MS-02 | enet_cv_ccd | Ridge CV + lambda_2_lars Conversion"
  )

  n <- 300L
  p <- 200L
  sigma <- 1.0
  n_folds <- 10L

  dat <- generate_data(n, p, sigma, seed = 404L)
  cat(sprintf("Data: n=%d, p=%d, folds=%d, active coefs=%d\n\n",
              n, p, n_folds, sum(dat$beta_true != 0)))

  cv <- MLElasticNetCV$new()
  cv$fit(dat$X, dat$y, alpha = 0.0, n_folds = n_folds, seed = 0L)
  print_cv_summary(cv, step_size = 10L)

  stopifnot(cv$cv_1se() >= cv$cv_min())
  lambda2_min <- cv$cv_min() * p / 2
  lambda2_1se <- cv$cv_1se() * p / 2

  cat(sprintf("  lambda_2_lars (min) = cv_min  * p/2 = %.6f * %d/2 = %.6f\n",
              cv$cv_min(), p, lambda2_min))
  cat(sprintf("  lambda_2_lars (1se) = cv_1se  * p/2 = %.6f * %d/2 = %.6f\n\n",
              cv$cv_1se(), p, lambda2_1se))
  cat("  (This is the value TRexGVS uses internally via CV_1SE_CCD.)\n\n")
  invisible(NULL)
}


# ==============================================================================
# Main
# ==============================================================================

print_section_header("enet_cv_ccd Demo Suite  (demo_mlm_ms_02_enet_cv_ccd)")

# Part A: path fitting
demo_path_low_dim()
demo_path_high_dim()

# Part B: K-fold CV
demo_cv_low_dim()
demo_cv_ridge_lars_conversion()

cat("\nAll enet_cv_ccd demos complete.\n\n")
