# ==============================================================================
# demo_mlm_ridge_01.R
# ==============================================================================
#
# Demonstration of RidgeSolver usage.
#
# Mirrors cpp/ml_methods/ridge_regression/demo_mlm_ridge_01/demo_mlm_ridge_01.cpp.
#
# Three demos which illustrate the RidgeSolver interface:
#
#   Demo 1 | Primal path   : n = 500, p = 50, lambda = 1 (n >= p) — dense
#                            ground truth, low noise.
#
#   Demo 2 | Dual path     : n = 80, p = 300, lambda = 10 (n < p) — sparse
#                            ground truth in the high-dimensional regime.
#
#   Demo 3 | Lambda sweep  : n = 200, p = 40 — coefficient shrinkage and
#                            growing training residual as the penalty grows.
#
# Notes on the port:
#   - Ridge regression solves the penalized least-squares problem
#       beta_hat = argmin ||y - X beta||_2^2 + lambda ||beta||_2^2 .
#     The solver picks its computational path automatically:
#       - primal (p x p system) when n >= p,
#       - dual   (n x n system) when n <  p — same solution, cheaper
#         factorization.
#   - Same n / p / lambda values and seeds as the C++ demo; R's
#     Mersenne-Twister stream differs from std::mt19937, so results match
#     qualitatively, not bitwise.
#
# ==============================================================================

library(TRexSelector)


print_section_header <- function(title) {
  cat(strrep("=", 78), "\n", sep = "")
  cat(" ", title, "\n", sep = "")
  cat(strrep("=", 78), "\n\n", sep = "")
}

l2_norm <- function(v) sqrt(sum(v^2))


# ==============================================================================
# Demo 1: Basic Solve (n >= p, primal path)
# ==============================================================================

demo_ridge_primal <- function() {
  print_section_header("RidgeSolver - Primal Path (n >= p)")

  # Step 1. Generate data from a known linear model: y = X * beta_true + noise.
  #         Dense ground truth (all coefficients = 1), low noise: with mild
  #         regularization the ridge estimate should sit close to beta_true.
  n      <- 500L
  p      <- 50L
  lambda <- 1.0
  cat(sprintf("Matrix size : %d x %d\n", n, p))
  cat(sprintf("Lambda      : %g\n\n", lambda))

  set.seed(42)
  X         <- matrix(rnorm(n * p), nrow = n, ncol = p)
  beta_true <- rep(1.0, p)
  y         <- as.numeric(X %*% beta_true) + 0.1 * rnorm(n)

  # Step 2. Solve
  solver   <- MLRidge$new()
  beta_hat <- solver$solve(X, y, lambda)

  # Step 3. Report
  coef_error <- l2_norm(beta_hat - beta_true)
  residual   <- l2_norm(y - X %*% beta_hat)
  cat(sprintf("||beta_hat - beta_true||_2 : %.4e\n", coef_error))
  cat(sprintf("Residual ||y - X*beta||_2  : %.4e\n", residual))

  # Didactic check: with sd(noise) = 0.1 the estimate must be close to truth.
  stopifnot(coef_error < 0.5)
  cat("✓ Ridge estimate close to the true coefficients\n")

  cat("\n\n")
  invisible(NULL)
}


# ==============================================================================
# Demo 2: Dual Path (n < p)
# ==============================================================================

demo_ridge_dual <- function() {
  print_section_header("RidgeSolver - Dual Path (n < p)")

  # Step 1. Generate high-dimensional data. Only the first 10 of 300
  #         coefficients are nonzero. Ridge does not select variables — it
  #         shrinks everything — so the estimate is biased, but the dual
  #         formulation keeps the problem well posed for n < p.
  n      <- 80L
  p      <- 300L
  lambda <- 10.0
  cat(sprintf("Matrix size : %d x %d\n", n, p))
  cat(sprintf("Lambda      : %g\n\n", lambda))

  set.seed(7)
  X <- matrix(rnorm(n * p), nrow = n, ncol = p)

  beta_true <- c(rep(1.0, 10L), rep(0.0, p - 10L))
  y         <- as.numeric(X %*% beta_true) + 0.1 * rnorm(n)

  # Step 2. Solve
  solver   <- MLRidge$new()
  beta_hat <- solver$solve(X, y, lambda)

  # Step 3. Report
  coef_error <- l2_norm(beta_hat - beta_true)
  residual   <- l2_norm(y - X %*% beta_hat)
  cat(sprintf("||beta_hat - beta_true||_2 : %.4e\n", coef_error))
  cat(sprintf("Residual ||y - X*beta||_2  : %.4e\n", residual))

  cat("\n\n")
  invisible(NULL)
}


# ==============================================================================
# Demo 3: Lambda Sweep
# ==============================================================================

demo_ridge_lambda_sweep <- function() {
  print_section_header("RidgeSolver - Lambda Sweep")

  # As lambda grows, ||beta_hat||_2 shrinks monotonically toward 0 while the
  # training residual grows — the classic bias-variance trade-off knob.
  n <- 200L
  p <- 40L
  cat(sprintf("Matrix size : %d x %d\n\n", n, p))

  set.seed(99)
  X         <- matrix(rnorm(n * p), nrow = n, ncol = p)
  beta_true <- rep(1.0, p)
  y         <- as.numeric(X %*% beta_true) + 0.5 * rnorm(n)

  lambdas <- c(1e-4, 1e-2, 0.1, 1.0, 10.0, 100.0, 1000.0)

  solver <- MLRidge$new()

  cat(sprintf("%-12s%-22s%-18s%-18s\n",
              "lambda", "||beta_hat - true||_2", "||beta_hat||_2",
              "||residual||_2"))
  cat(strrep("-", 70), "\n", sep = "")

  beta_norms <- numeric(length(lambdas))
  residuals  <- numeric(length(lambdas))
  for (i in seq_along(lambdas)) {
    beta_hat      <- solver$solve(X, y, lambdas[i])
    beta_norms[i] <- l2_norm(beta_hat)
    residuals[i]  <- l2_norm(y - X %*% beta_hat)
    cat(sprintf("%-12.1e%-22.4e%-18.4e%-18.4e\n",
                lambdas[i], l2_norm(beta_hat - beta_true),
                beta_norms[i], residuals[i]))
  }

  # Didactic checks: for ridge, ||beta_hat(lambda)||_2 is non-increasing and
  # the training residual is non-decreasing in lambda.
  stopifnot(all(diff(beta_norms) <= 1e-10))
  stopifnot(all(diff(residuals) >= -1e-10))
  cat("\n✓ Coefficients shrink and the residual grows with lambda\n")

  cat("\n\n")
  invisible(NULL)
}


# ==============================================================================
# Main
# ==============================================================================

demo_ridge_primal()
demo_ridge_dual()
demo_ridge_lambda_sweep()
