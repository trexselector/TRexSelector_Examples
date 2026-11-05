# ==============================================================================
# demo_mlm_svd_01.R
# ==============================================================================
#
# Demonstration of SVDSolver usage (truncated rank-M SVD).
#
# Mirrors cpp/ml_methods/svd/demo_mlm_svd_01/demo_mlm_svd_01.cpp.
#
# Three demos which illustrate the SVDSolver interface:
#
#   Demo 1 | Direct path   : n = 300, p = 80, M = 10 (n >= p) — truncated and
#                            full-rank reconstruction.
#
#   Demo 2 | Gram path     : n = 100, p = 500, M = 8 (p > 2n) — the solver
#                            works on the n x n Gram matrix X X^T instead of X.
#
#   Demo 3 | Orthogonality : n = 200, p = 150, M = 20 — ||U^T U - I||_F and
#                            ||V^T V - I||_F.
#
# Notes on the port:
#   - The truncated SVD keeps the M leading singular triplets of X (n x p):
#       X_M = U_M diag(S_M) V_M^T,   U_M: n x M,  S_M: M,  V_M: p x M
#     X_M is the best rank-M approximation of X in the Frobenius norm
#     (Eckart-Young). $compute(X, M) returns a list with elements U, S, V.
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


# Compare the truncation error with the Eckart-Young optimum: the best rank-M
# approximation in the Frobenius norm leaves exactly the energy of the
# discarded singular values, which a full LAPACK SVD (base::svd) provides.
print_eckart_young_check <- function(X, M, rel_error) {
  s_all   <- svd(X, nu = 0, nv = 0)$d
  optimum <- sqrt(sum(s_all[-seq_len(M)]^2)) / norm(X, "F")
  diff    <- abs(rel_error - optimum)
  cat(sprintf("Eckart-Young optimum sqrt(sum_{k>M} s_k^2) / ||X||_F  : %.4e\n", optimum))
  cat(sprintf("Eckart-Young check (|difference| = %.4e < 1e-10): %s\n",
              diff, if (diff < 1e-10) "PASS" else "FAIL"))
}


# ==============================================================================
# Demo 1: Direct / Thin-Jacobi Path (n >= p)
# ==============================================================================

demo_svd_direct <- function() {
  print_section_header("SVDSolver - Direct Path (n >= p)")

  # Step 1. Generate random data
  n <- 300L
  p <- 80L
  M <- 10L
  cat(sprintf("Matrix size : %d x %d\n", n, p))
  cat(sprintf("M           : %d\n\n", M))

  set.seed(42)
  X <- matrix(rnorm(n * p), nrow = n, ncol = p)

  # Step 2. Compute SVD (for tall matrices the solver decomposes X directly)
  solver <- MLSVD$new()
  res <- solver$compute(X, M)

  # Step 3. Print dimensions
  cat(sprintf("U dimensions : %d x %d\n", nrow(res$U), ncol(res$U)))
  cat(sprintf("S size       : %d\n", length(res$S)))
  cat(sprintf("V dimensions : %d x %d\n\n", nrow(res$V), ncol(res$V)))

  # Step 4. Reconstruction error: ||X_M - X||_F / ||X||_F  (large here: a
  #         Gaussian matrix has a flat spectrum, so 10 of 80 components
  #         capture little energy).
  X_approx  <- res$U %*% diag(res$S) %*% t(res$V)
  rel_error <- norm(X_approx - X, "F") / norm(X, "F")
  cat(sprintf("Relative reconstruction error ||X_M - X||_F / ||X||_F : %.4e\n",
              rel_error))
  print_eckart_young_check(X, M, rel_error)

  # Step 5. Singular values
  cat(sprintf("\nTop %d singular values:\n", M))
  for (k in seq_len(M)) {
    cat(sprintf("  s%d = %.4f\n", k, res$S[k]))
  }

  # Didactic checks: singular values are sorted decreasingly, and at full rank
  # (M = p) the truncated SVD reproduces X exactly up to floating-point error.
  stopifnot(all(diff(res$S) <= 1e-12))

  res_full       <- solver$compute(X, p)
  X_full         <- res_full$U %*% diag(res_full$S) %*% t(res_full$V)
  rel_error_full <- norm(X_full - X, "F") / norm(X, "F")
  cat(sprintf("\nFull-rank (M = p) relative reconstruction error : %.4e\n",
              rel_error_full))
  stopifnot(rel_error_full < 1e-10)
  cat("✓ Full-rank reconstruction is exact\n")

  cat("\n\n")
  invisible(NULL)
}


# ==============================================================================
# Demo 2: Gram / Kernel Path (p > 2*n)
# ==============================================================================

demo_svd_gram <- function() {
  print_section_header("SVDSolver - Gram Path (p > 2*n)")

  # Step 1. Generate wide matrix
  n <- 100L
  p <- 500L
  M <- 8L
  cat(sprintf("Matrix size : %d x %d  (p > 2*n -> Gram path)\n", n, p))
  cat(sprintf("M           : %d\n\n", M))

  set.seed(7)
  X <- matrix(rnorm(n * p), nrow = n, ncol = p)

  # Step 2. Compute SVD. For wide matrices the solver works on the n x n Gram
  #         matrix X X^T instead of X itself, which is much cheaper when p >> n.
  solver <- MLSVD$new()
  res <- solver$compute(X, M)

  # Step 3. Print dimensions
  cat(sprintf("U dimensions : %d x %d\n", nrow(res$U), ncol(res$U)))
  cat(sprintf("S size       : %d\n", length(res$S)))
  cat(sprintf("V dimensions : %d x %d\n\n", nrow(res$V), ncol(res$V)))

  # Step 4. Reconstruction error
  X_approx  <- res$U %*% diag(res$S) %*% t(res$V)
  rel_error <- norm(X_approx - X, "F") / norm(X, "F")
  cat(sprintf("Relative reconstruction error ||X_M - X||_F / ||X||_F : %.4e\n",
              rel_error))
  print_eckart_young_check(X, M, rel_error)

  cat("\n\n")
  invisible(NULL)
}


# ==============================================================================
# Demo 3: Orthogonality Check
# ==============================================================================

demo_svd_orthogonality <- function() {
  print_section_header("SVDSolver - Orthogonality Check")

  n <- 200L
  p <- 150L
  M <- 20L
  cat(sprintf("Matrix size : %d x %d\n", n, p))
  cat(sprintf("M           : %d\n\n", M))

  set.seed(99)
  X <- matrix(rnorm(n * p), nrow = n, ncol = p)

  solver <- MLSVD$new()
  res <- solver$compute(X, M)

  # The singular vector blocks must be orthonormal: U^T U = I and V^T V = I.
  I_M       <- diag(M)
  UtU_error <- norm(crossprod(res$U) - I_M, "F")
  VtV_error <- norm(crossprod(res$V) - I_M, "F")

  cat(sprintf("||U^T U - I||_F : %.4e\n", UtU_error))
  cat(sprintf("||V^T V - I||_F : %.4e\n\n", VtV_error))

  if (UtU_error < 1e-10 && VtV_error < 1e-10) {
    cat("✓ U and V are orthonormal\n")
  } else {
    cat("✗ Orthonormality check FAILED\n")
  }

  cat("\n\n")
  invisible(NULL)
}


# ==============================================================================
# Main
# ==============================================================================

demo_svd_direct()
demo_svd_gram()
demo_svd_orthogonality()
