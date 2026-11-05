# ==============================================================================
# trex_da_dgps.R
# ==============================================================================
# Consolidated data-generating processes for the T-Rex+DA demonstration suite
# (mirrors cpp dgp_generators.hpp / Python dgp_generators.py). One section per
# DGP family; each demo sources this single file.
#
#   dgp_ar1_snr                 (dgp_ar1)                 -> demos 01, 07
#   dgp_ar1_block_white_snr     (dgp_ar1_block_white)     -> demo 02
#   dgp_ar1_block_snr           (dgp_ar1_block)           -> demo 03
#   dgp_block_toeplitz_hvt_snr  (dgp_block_toeplitz_hvt)  -> demo 04
#   dgp_ht_block_white_snr      (dgp_ht_block_white)      -> demo 05
#   dgp_groups_toeplitz_leaf    (dgp_groups_toeplitz_leaf)-> demo 06
#   dgp_nn_snr                  (dgp_nn)                  -> demo 07
#   dgp_equi_snr                (dgp_equi)                -> demo 08
#
# dgp_bt_snr (two-level factor model) and dgp_nn_block_snr (block-diagonal
# MA(kappa)) are R-only leftovers kept for reference; no demo uses them.
# ==============================================================================


# ==============================================================================
# ==  DGP family: ar1_snr
# ==============================================================================

# ==============================================================================
# dgp_ar1_snr
# ==============================================================================

#' DGP for the AR(1) scenario with SNR-calibrated noise.
#'
#' Predictor rows are generated column-by-column via the stationary AR(1)
#' recursion:
#'
#'   X[i, 1]   = eta[i, 1]
#'   X[i, j]   = rho * X[i, j-1] + sqrt(1 - rho^2) * eta[i, j]
#'
#' with eta[i, j] i.i.d. N(0, 1).
#' This yields the Toeplitz covariance Sigma[j, k] = rho^|j - k|.
#'
#' The response is:
#'
#'   y = X %*% beta + eps,   eps ~ N(0, noise_sigma^2 * I_n)
#'
#' where beta has entries equal to `amplitude` at the positions given by
#' `support` (1-based) and zero elsewhere.
#' The noise standard deviation is chosen according to the target SNR:
#'
#'   noise_sigma = sqrt( sample_var(X %*% beta) / snr )
#'
#' with sample_var() using the n-1 denominator, matching C++ make_snr_response().
#'
#' @param n         Number of observations.
#' @param p         Number of predictors.
#' @param support   1-based integer vector of active predictor indices.
#' @param amplitude Signal coefficient for each active predictor. Default 3.
#' @param rho       AR(1) correlation parameter rho in (-1, 1).
#' @param snr       Target signal-to-noise ratio  Var(signal) / Var(noise).
#' @param seed      Integer seed for the RNG (governs both X and noise).
#'
#' @return A list with components:
#'   \describe{
#'     \item{X}{Predictor matrix (n x p).}
#'     \item{y}{Response vector (length n).}
#'     \item{beta}{Coefficient vector (length p).}
#'     \item{true_support}{1-based integer index vector of active predictors.}
#'     \item{n, p, s, snr}{Dimension and parameter scalars.}
#'   }
#'
#' @examples
#' dat <- dgp_ar1_snr(
#'   n         = 150,
#'   p         = 500,
#'   support   = make_support_random(5, 500, seed = 2026),
#'   amplitude = 3,
#'   rho       = 0.7,
#'   snr       = 2.0,
#'   seed      = 2026
#' )
dgp_ar1_snr <- function(n,
                        p,
                        support,
                        amplitude = 3,
                        rho       = 0.7,
                        snr       = 2.0,
                        seed      = NULL) {

  stopifnot(is.integer(support) || is.numeric(support))
  stopifnot(all(support >= 1L), all(support <= p))
  stopifnot(abs(rho) < 1)
  stopifnot(snr > 0)

  if (!is.null(seed)) set.seed(seed)

  # ------------------------------------------------------------------
  # Generate X: AR(1) recursion row-wise
  # eta[i, j] are drawn in column order: col 1, col 2, ..., col p
  # (each rnorm(n) call advances the RNG by n draws)
  # ------------------------------------------------------------------
  scale_ar <- sqrt(1.0 - rho^2)

  X      <- matrix(0.0, nrow = n, ncol = p)
  X[, 1] <- stats::rnorm(n)
  for (j in seq(2L, p)) {
    X[, j] <- rho * X[, j - 1L] + scale_ar * stats::rnorm(n)
  }

  # ------------------------------------------------------------------
  # Coefficient vector (1-based support)
  # ------------------------------------------------------------------
  beta          <- numeric(p)
  beta[support] <- amplitude

  # ------------------------------------------------------------------
  # SNR-calibrated response:
  # noise_sigma = sqrt(var(signal) / snr)
  # var() in R uses n-1 denominator (Bessel correction factor)
  # ------------------------------------------------------------------

  signal      <- drop(X %*% beta)
  var_sig     <- stats::var(signal)
  noise_sigma <- sqrt(var_sig / snr)
  y           <- signal + stats::rnorm(n, mean = 0, sd = noise_sigma)

  list(
    X            = X,
    y            = y,
    beta         = beta,
    true_support = as.integer(support),
    n            = n,
    p            = p,
    s            = length(support),
    snr          = snr,
    rho          = rho
  )
}

# ==============================================================================
# ==  DGP family: equi_snr
# ==============================================================================

# ==============================================================================
# dgp_equi_snr
# ==============================================================================

#' Equicorrelation DGP with SNR-calibrated noise (cpp: dgp_equi).
#'
#' Compound symmetry via a single latent factor:
#'
#'   X[i, j] = sqrt(rho) * f[i] + sqrt(1 - rho) * eta[i, j]
#'
#' with f and eta i.i.d. N(0, 1), yielding Sigma[j, k] = rho for j != k.
#' The response noise is calibrated to the target SNR exactly as in
#' `dgp_ar1_snr()` (sample variance with the n-1 denominator).
#'
#' @param n         Number of observations.
#' @param p         Number of predictors.
#' @param support   1-based integer vector of active predictor indices.
#' @param amplitude Signal coefficient for each active predictor. Default 3.
#' @param rho       Equicorrelation coefficient in [0, 1).
#' @param snr       Target signal-to-noise ratio Var(signal) / Var(noise).
#' @param seed      Integer seed for the RNG (governs both X and noise).
#'
#' @return A list with components:
#'   X, y, beta, true_support, n, p, s, snr, rho.
dgp_equi_snr <- function(n,
                         p,
                         support,
                         amplitude = 3,
                         rho       = 0.25,
                         snr       = 2.0,
                         seed      = NULL) {

  stopifnot(rho >= 0, rho < 1)
  stopifnot(snr > 0)
  stopifnot(all(support >= 1L), all(support <= p))

  if (!is.null(seed)) set.seed(seed)

  f   <- stats::rnorm(n)
  eta <- matrix(stats::rnorm(n * p), nrow = n, ncol = p)
  X   <- sqrt(rho) * matrix(f, nrow = n, ncol = p) + sqrt(1.0 - rho) * eta

  beta          <- numeric(p)
  beta[support] <- amplitude

  signal      <- drop(X %*% beta)
  noise_sigma <- sqrt(stats::var(signal) / snr)
  y           <- signal + stats::rnorm(n, mean = 0, sd = noise_sigma)

  list(
    X            = X,
    y            = y,
    beta         = beta,
    true_support = as.integer(support),
    n            = n,
    p            = p,
    s            = length(support),
    snr          = snr,
    rho          = rho
  )
}

# ==============================================================================
# ==  DGP family: bt_snr
# ==============================================================================


# ==============================================================================
# dgp_bt_snr
# ==============================================================================

#' DGP for the binary-tree scenario with SNR-calibrated noise.
#'
#' Two-level hierarchical block model.
#' The p predictors are divided into n_blocks equal-sized blocks.
#' Each predictor is generated as:
#'
#'   X[i, j] = sqrt(rho_between)              * f_0[i]
#'           + sqrt(rho_within - rho_between) * f_m[i]   (m = block of j)
#'           + sqrt(1 - rho_within)           * eta[i, j]
#'
#' with f_0, f_m, eta all i.i.d. N(0, 1), yielding:
#'
#'   Sigma[j, k] = rho_within   (same block)
#'   Sigma[j, k] = rho_between  (different block)
#'
#' The response is:
#'
#'   y = X %*% beta + eps,   eps ~ N(0, noise_sigma^2 * I_n)
#'
#' where noise_sigma = sqrt( sample_var(X %*% beta) / snr ).
#'
#' @param n           Number of observations.
#' @param p           Number of predictors (must be divisible by n_blocks).
#' @param support     1-based integer vector of active predictor indices.
#' @param amplitude   Signal coefficient for each active predictor. Default 3.
#' @param n_blocks    Number of hierarchical blocks.
#' @param rho_within  Within-block correlation. Must satisfy rho_within > rho_between.
#' @param rho_between Between-block correlation. Must satisfy rho_between >= 0.
#' @param snr         Target signal-to-noise ratio Var(signal) / Var(noise).
#' @param seed        Integer seed for the RNG (governs X and noise).
#'
#' @return A list with components:
#'   X, y, beta, true_support, n, p, s, n_blocks, rho_within, rho_between, snr.
#'
#' @examples
#' support <- make_support_capped_spread(s = 5L, p = 500L, max_gap = 20L)
#' dat <- dgp_bt_snr(
#'   n           = 150,
#'   p           = 500,
#'   support     = support,
#'   amplitude   = 3,
#'   n_blocks    = 10,
#'   rho_within  = 0.8,
#'   rho_between = 0.1,
#'   snr         = 2.0,
#'   seed        = 2026
#' )
dgp_bt_snr <- function(n,
                       p,
                       support,
                       amplitude   = 3,
                       n_blocks    = 10L,
                       rho_within  = 0.8,
                       rho_between = 0.1,
                       snr         = 2.0,
                       seed        = NULL) {

  stopifnot(p %% n_blocks == 0)
  stopifnot(rho_between >= 0, rho_between <= rho_within, rho_within < 1)
  stopifnot(snr > 0)
  stopifnot(all(support >= 1L), all(support <= p))

  if (!is.null(seed)) set.seed(seed)

  block_size   <- p %/% n_blocks
  block_labels <- rep(seq_len(n_blocks), each = block_size)  # length p

  a  <- sqrt(rho_between)
  b  <- sqrt(rho_within - rho_between)
  c <- sqrt(1.0 - rho_within)

  # Factor draws:
  #   row i: 1 global draw (f0), n_blocks block draws, p idiosyncratic draws.
  f0      <- stats::rnorm(n)
  F_block <- matrix(stats::rnorm(n * n_blocks), nrow = n, ncol = n_blocks)
  eta     <- matrix(stats::rnorm(n * p),        nrow = n, ncol = p)

  X <- a  * matrix(f0, nrow = n, ncol = p) +
    b  * F_block[, block_labels, drop = FALSE] +
    c * eta

  # Coefficient vector
  beta          <- numeric(p)
  beta[support] <- amplitude

  # SNR-calibrated response: noise_sigma = sqrt(var(signal) / snr)
  signal      <- drop(X[, support, drop = FALSE] %*% beta[support])
  var_sig     <- stats::var(signal)
  noise_sigma <- sqrt(var_sig / snr)
  y           <- signal + stats::rnorm(n, mean = 0, sd = noise_sigma)

  list(
    X            = X,
    y            = y,
    beta         = beta,
    true_support = as.integer(support),
    n            = n,
    p            = p,
    s            = length(support),
    n_blocks     = n_blocks,
    rho_within   = rho_within,
    rho_between  = rho_between,
    snr          = snr
  )
}



# ==============================================================================
# dgp_ar1_block_snr
# ==============================================================================

#' DGP for the paper's block-diagonal AR(1)/Toeplitz scenario.
#'
#' The p predictors are divided into n_blocks equal-sized blocks.
#' Within each block the predictors follow an AR(1) process with parameter rho:
#'
#'   X[i, block_start]     ~ N(0, 1)
#'   X[i, block_start + j] = rho * X[i, block_start + j - 1]
#'                         + sqrt(1 - rho^2) * eps[i, j],  eps ~ N(0,1)
#'
#' yielding the Toeplitz covariance Sigma_m[j, k] = rho^|j-k| within each
#' block and zero correlation between blocks (block-diagonal Sigma).
#'
#'
#' @param n         Number of observations.
#' @param p         Number of predictors (must be divisible by n_blocks).
#' @param support   1-based integer vector of active predictor indices.
#' @param amplitude Signal coefficient for each active predictor. Default 3.
#' @param n_blocks  Number of AR(1) blocks.
#' @param rho       AR(1) parameter (within-block). Must satisfy |rho| < 1.
#' @param snr       Target signal-to-noise ratio Var(signal) / Var(noise).
#' @param seed      Integer seed for the RNG.
#'
#' @return A list with components:
#'   X, y, beta, true_support, n, p, s, n_blocks, rho, snr.
dgp_ar1_block_snr <- function(n,
                              p,
                              support,
                              amplitude = 3,
                              n_blocks,
                              rho,
                              snr  = 2.0,
                              seed = NULL) {

  stopifnot(p %% n_blocks == 0)
  stopifnot(abs(rho) < 1)
  stopifnot(snr > 0)
  stopifnot(all(support >= 1L), all(support <= p))

  if (!is.null(seed)) set.seed(seed)

  block_size <- p %/% n_blocks
  innov_sd   <- sqrt(1.0 - rho^2)

  # Build X block by block using the AR(1) column recursion.
  X <- matrix(0.0, nrow = n, ncol = p)
  for (m in seq_len(n_blocks)) {
    col_start <- (m - 1L) * block_size + 1L
    col_end   <- m * block_size

    X[, col_start] <- stats::rnorm(n)
    for (j in seq(col_start + 1L, col_end)) {
      X[, j] <- rho * X[, j - 1L] + innov_sd * stats::rnorm(n)
    }
  }

  # Coefficient vector
  beta          <- numeric(p)
  beta[support] <- amplitude

  # SNR-calibrated response: noise_sigma = sqrt(var(signal) / snr)
  signal      <- drop(X[, support, drop = FALSE] %*% beta[support])
  var_sig     <- stats::var(signal)
  noise_sigma <- sqrt(var_sig / snr)
  y           <- signal + stats::rnorm(n, mean = 0, sd = noise_sigma)

  list(
    X            = X,
    y            = y,
    beta         = beta,
    true_support = as.integer(support),
    n            = n,
    p            = p,
    s            = length(support),
    n_blocks     = n_blocks,
    rho          = rho,
    snr          = snr
  )
}



# ==============================================================================
# dgp_ar1_block_white_snr
# ==============================================================================

#' DGP: M times AR(1) blocks + 1 white-noise block; actives only in the AR blocks.
#'
#' The first p_ar predictors are divided into n_blocks equal-sized AR(1) blocks.
#' The last p_white predictors form one independent white-noise block:
#' each column is i.i.d. N(0, 1), uncorrelated with all other columns.
#'
#' Active predictors (support) must all lie within the AR part (indices 1 ..
#' p_ar).
#' The selector operates on the full design matrix of total width:
#' p = p_ar + p_white.
#'
#' @param n         Number of observations.
#' @param p_ar      Number of AR(1) predictors (must be divisible by n_blocks).
#' @param p_white   Number of white-noise predictors (>= 1).
#' @param support   1-based integer vector of active indices; all <= p_ar.
#' @param amplitude Signal coefficient for each active predictor. Default 3.
#' @param n_blocks  Number of AR(1) blocks.
#' @param rho       AR(1) parameter (within AR blocks). Must satisfy |rho| < 1.
#' @param snr       Target signal-to-noise ratio Var(signal) / Var(noise).
#' @param seed      Integer seed for the RNG.
#'
#' @return A list with components:
#'   X, y, beta, true_support, n, p (= p_ar + p_white), p_ar, p_white,
#'   s, n_blocks, rho, snr.
dgp_ar1_block_white_snr <- function(n,
                                    p_ar,
                                    p_white,
                                    support,
                                    amplitude = 3,
                                    n_blocks,
                                    rho,
                                    snr  = 2.0,
                                    seed = NULL) {

  stopifnot(p_ar %% n_blocks == 0)
  stopifnot(p_white >= 1L)
  stopifnot(abs(rho) < 1)
  stopifnot(snr > 0)
  stopifnot(all(support >= 1L), all(support <= p_ar))

  if (!is.null(seed)) set.seed(seed)

  p          <- p_ar + p_white
  block_size <- p_ar %/% n_blocks
  innov_sd   <- sqrt(1.0 - rho^2)

  # Build AR part block by block.
  X_ar <- matrix(0.0, nrow = n, ncol = p_ar)
  for (m in seq_len(n_blocks)) {
    col_start <- (m - 1L) * block_size + 1L
    col_end   <- m * block_size

    X_ar[, col_start] <- stats::rnorm(n)
    for (j in seq(col_start + 1L, col_end)) {
      X_ar[, j] <- rho * X_ar[, j - 1L] + innov_sd * stats::rnorm(n)
    }
  }

  # White-noise block: i.i.d. N(0, 1) — independent of everything.
  X_white <- matrix(stats::rnorm(n * p_white), nrow = n, ncol = p_white)

  X <- cbind(X_ar, X_white)

  # Coefficient vector (1-based support, restricted to AR part)
  beta          <- numeric(p)
  beta[support] <- amplitude

  # SNR-calibrated response
  signal      <- drop(X %*% beta)
  var_sig     <- stats::var(signal)
  noise_sigma <- sqrt(var_sig / snr)
  y           <- signal + stats::rnorm(n, mean = 0, sd = noise_sigma)

  list(
    X            = X,
    y            = y,
    beta         = beta,
    true_support = as.integer(support),
    n            = n,
    p            = p,
    p_ar         = p_ar,
    p_white      = p_white,
    s            = length(support),
    n_blocks     = n_blocks,
    rho          = rho,
    snr          = snr
  )
}

# ==============================================================================
# ==  DGP family: nn_snr
# ==============================================================================

# ==============================================================================
# dgp_nn_snr
# ==============================================================================

#' Global MA(kappa) DGP with SNR-calibrated noise.
#'
#' Each column of X is the causal MA(kappa) convolution of a shared i.i.d.
#' N(0,1) innovation sheet eta:
#'
#'   X[i, j] = sum_{l=0}^{kappa} theta[l] * eta[i, j - l].
#'
#' In 1-based R indexing we avoid negative column indices by using a padded
#' innovation sheet eta of size n x (p + kappa). The same causal convolution is
#' then implemented as
#'
#'   X[, j] = eta[, j:(j + kappa)] %*% rev(theta),
#'
#' which is algebraically equivalent after reindexing.
#'
#' The coefficients are
#'
#'   theta[l] = c * rho^l,
#'   c = 1 / sqrt(sum_{l=0}^{kappa} rho^(2l)).
#'
#' The resulting covariance between columns j and k (d = |j - k|) is
#'
#'   Sigma[j, k] = sum_{l=0}^{kappa - d} theta[l] * theta[l + d],   d <= kappa,
#'   Sigma[j, k] = 0,                                                d >  kappa.
#'
#' The response is
#'
#'   y = X %*% beta + eps,   eps ~ N(0, noise_sigma^2 * I_n),
#'
#' where
#'
#'   noise_sigma = sqrt(sample_var(X %*% beta) / snr),
#'
#' using the usual sample variance with n - 1 denominator.
#'
#' @param n Number of observations.
#' @param p Number of predictors.
#' @param support 1-based integer vector of active predictor indices.
#' @param amplitude Signal coefficient for each active predictor. Default 3.
#' @param kappa MA order (bandwidth). Default 3.
#' @param rho Decay parameter: theta[l] proportional to rho^l. Default 0.7.
#' @param snr Target signal-to-noise ratio Var(signal) / Var(noise).
#' @param seed Integer seed for the RNG (governs both X and noise).
#'
#' @return A list with components:
#'   X, y, beta, true_support, n, p, s, kappa, rho, snr.
#'
#' @examples
#' support <- make_support_capped_spread(s = 5L, p = 500L, max_gap = 20L)
#' dat <- dgp_nn_snr(
#'   n = 150,
#'   p = 500,
#'   support = support,
#'   kappa = 3L,
#'   rho = 0.7,
#'   snr = 2.0,
#'   seed = 2026
#' )
dgp_nn_snr <- function(n, p, support,
                       amplitude = 3,
                       kappa = 3L,
                       rho = 0.7,
                       snr = 2.0,
                       seed = NULL) {

  stopifnot(kappa >= 1L)
  stopifnot(rho >= 0, rho < 1)
  stopifnot(snr > 0)
  stopifnot(all(support >= 1L), all(support <= p))

  if (!is.null(seed)) set.seed(seed)

  # Normalised MA coefficients:
  # theta[l] = c * rho^l, c = 1 / sqrt(sum_{l=0}^{kappa} rho^{2l})
  l_seq <- 0L:kappa
  theta <- rho ^ l_seq
  theta <- theta / sqrt(sum(theta^2))
  theta_rev <- rev(theta)

  # Innovation sheet: n x (p + kappa)
  eta <- matrix(stats::rnorm(n * (p + kappa)), nrow = n, ncol = p + kappa)

  # Build X via causal MA(kappa) convolution
  # Column j (1-based) = sum_{l=0}^kappa theta[l] * eta[, j + kappa - l]
  # Implemented as eta[, j:(j + kappa)] %*% rev(theta)
  X <- matrix(0.0, nrow = n, ncol = p)
  for (j in seq_len(p)) {
    X[, j] <- drop(eta[, j:(j + kappa)] %*% theta_rev)
  }

  # Coefficient vector (1-based support)
  beta <- numeric(p)
  beta[support] <- amplitude

  # SNR-calibrated response: noise_sigma = sqrt(var(signal) / snr)
  # var() uses the n - 1 denominator.
  signal <- drop(X %*% beta)
  var_sig <- stats::var(signal)
  noise_sigma <- sqrt(var_sig / snr)
  y <- signal + stats::rnorm(n, mean = 0, sd = noise_sigma)

  list(
    X = X,
    y = y,
    beta = beta,
    true_support = as.integer(support),
    n = n,
    p = p,
    s = length(support),
    kappa = kappa,
    rho = rho,
    snr = snr
  )
}

# ==============================================================================
# dgp_nn_block_snr
# ==============================================================================

#' Block-diagonal MA(kappa) DGP with SNR-calibrated noise.
#'
#' The p predictors are divided into n_blocks independent blocks, each of size
#' Q = p / n_blocks. Within block m the covariance is the same causal MA(kappa)
#' banded structure as in dgp_nn_snr(), restricted to Q predictors; between
#' blocks the correlation is exactly zero.
#'
#' This is the nearest-neighbour analogue of dgp_ar1_block_snr():
#' - dgp_ar1_block_snr => blocks with AR(1) / Toeplitz covariance
#' - dgp_nn_block_snr  => blocks with MA(kappa) banded covariance
#'
#' Each block draws its own independent innovation sheet, so block RNG draws
#' are sequential (block 1 first, then block 2, etc.). This differs from the
#' global dgp_nn_snr() where a single sheet covers all p + kappa columns.
#'
#' Active predictors are typically set to one per block via
#' make_support_bt_one_per_block() from dgp_bt_snr.R (deterministic: first
#' element of each block).
#'
#' @param n Number of observations.
#' @param p Number of predictors (must be divisible by n_blocks).
#' @param support 1-based integer vector of active predictor indices.
#' @param amplitude Signal coefficient for each active predictor. Default 3.
#' @param n_blocks Number of MA(kappa) blocks (= number of active predictors
#'   when one active per block).
#' @param kappa MA order (bandwidth) within each block. Default 3.
#'   Must satisfy kappa < Q = p / n_blocks.
#' @param rho Decay parameter: theta[l] proportional to rho^l. Default 0.7.
#' @param snr Target signal-to-noise ratio Var(signal) / Var(noise).
#' @param seed Integer seed for the RNG.
#'
#' @return A list with components:
#'   X, y, beta, true_support, n, p, s, n_blocks, kappa, rho, snr.
#'
#' @examples
#' support <- make_support_bt_one_per_block(p = 25L, n_blocks = 5L)
#' dat <- dgp_nn_block_snr(
#'   n = 150,
#'   p = 25,
#'   support = support,
#'   n_blocks = 5,
#'   kappa = 3L,
#'   rho = 0.7,
#'   snr = 2.0,
#'   seed = 2026
#' )
dgp_nn_block_snr <- function(n, p, support,
                             amplitude = 3,
                             n_blocks,
                             kappa = 3L,
                             rho = 0.7,
                             snr = 2.0,
                             seed = NULL) {

  stopifnot(p %% n_blocks == 0L)
  block_size <- p %/% n_blocks
  stopifnot(kappa < block_size,
            "kappa must be less than block size Q = p / n_blocks")
  stopifnot(kappa >= 1L)
  stopifnot(rho >= 0, rho < 1)
  stopifnot(snr > 0)
  stopifnot(all(support >= 1L), all(support <= p))

  if (!is.null(seed)) set.seed(seed)

  # Shared normalised MA coefficients (identical across all blocks)
  l_seq <- 0L:kappa
  theta <- rho ^ l_seq
  theta <- theta / sqrt(sum(theta^2))
  theta_rev <- rev(theta)

  # Build X block-by-block; each block has its own independent innovation sheet
  X <- matrix(0.0, nrow = n, ncol = p)

  for (m in seq_len(n_blocks)) {
    col_start <- (m - 1L) * block_size + 1L

    # Innovation sheet for this block: n x (block_size + kappa)
    eta_m <- matrix(stats::rnorm(n * (block_size + kappa)),
                    nrow = n, ncol = block_size + kappa)

    for (j in seq_len(block_size)) {
      X[, col_start + j - 1L] <- drop(eta_m[, j:(j + kappa)] %*% theta_rev)
    }
  }

  # Coefficient vector (1-based support)
  beta <- numeric(p)
  beta[support] <- amplitude

  # SNR-calibrated response
  signal <- drop(X %*% beta)
  var_sig <- stats::var(signal)
  noise_sigma <- sqrt(var_sig / snr)
  y <- signal + stats::rnorm(n, mean = 0, sd = noise_sigma)

  list(
    X = X,
    y = y,
    beta = beta,
    true_support = as.integer(support),
    n = n,
    p = p,
    s = length(support),
    n_blocks = n_blocks,
    kappa = kappa,
    rho = rho,
    snr = snr
  )
}

# ==============================================================================
# ==  DGP family: ht_snr
# ==============================================================================


# ==============================================================================
# dgp_block_toeplitz_hvt_snr
# ==============================================================================

#' DGP for heavy-tailed block-Toeplitz data with SNR-calibrated noise
#' (cpp: dgp_block_toeplitz_hvt).
#'
#' All p predictors form n_blocks independent AR(1)-Toeplitz blocks of size
#' p / n_blocks. When `heavy_X`, each block is converted to multivariate
#' Student-t(nu) via a Gaussian scale mixture with an independent chi-squared
#' mixing variable per (row, block) — extreme events are block-local,
#' mirroring cpp dgp_block_toeplitz_hvt. Noise is Gaussian, or
#' variance-matched Student-t(nu) when `heavy_noise`.
#'
#' @param n           Number of observations.
#' @param p           Total number of predictors. Must be divisible by n_blocks.
#' @param support     1-based integer vector of active predictor indices.
#' @param amplitude   Signal coefficient for each active predictor. Default 1.
#' @param n_blocks    Number of AR(1) blocks (= number of active predictors).
#' @param rho         AR(1) within-block correlation parameter.
#' @param snr         Target signal-to-noise ratio Var(signal) / Var(noise).
#' @param nu          Degrees of freedom for the Student-t distribution. Default 3.
#' @param heavy_X     If TRUE (default), predictors are block-local Student-t(nu).
#' @param heavy_noise If TRUE, noise is also Student-t(nu); otherwise Gaussian.
#' @param seed        Integer RNG seed. NULL means do not set seed.
#'
#' @return A list with components: X, y, beta, true_support, n, p, n_blocks,
#'   rho, snr, nu, heavy_noise.
dgp_block_toeplitz_hvt_snr <- function(n,
                                       p,
                                       support,
                                       amplitude   = 1.0,
                                       n_blocks    = 5L,
                                       rho         = 0.8,
                                       snr         = 2.0,
                                       nu          = 3.0,
                                       heavy_X     = TRUE,
                                       heavy_noise = FALSE,
                                       seed        = NULL) {

  stopifnot(p %% n_blocks == 0)
  stopifnot(abs(rho) < 1)
  stopifnot(snr > 0, nu > 2)
  stopifnot(all(support >= 1L), all(support <= p))

  if (!is.null(seed)) set.seed(seed)

  block_size <- p %/% n_blocks
  innov_sd   <- sqrt(1.0 - rho^2)

  X <- matrix(0.0, nrow = n, ncol = p)
  for (m in seq_len(n_blocks)) {
    col_start <- (m - 1L) * block_size + 1L
    col_end   <- m * block_size

    Z_block       <- matrix(0.0, nrow = n, ncol = block_size)
    Z_block[, 1L] <- stats::rnorm(n)
    if (block_size > 1L) {
      for (j in seq(2L, block_size)) {
        Z_block[, j] <- rho * Z_block[, j - 1L] + innov_sd * stats::rnorm(n)
      }
    }
    if (heavy_X) {
      # One mixing variable per (row, block): sqrt(nu / chi2(nu))
      U       <- stats::rchisq(n, df = nu)
      Z_block <- Z_block * sqrt(nu / U)
    }
    X[, col_start:col_end] <- Z_block
  }

  # Response and SNR-calibrated noise
  beta    <- numeric(p)
  beta[support] <- amplitude

  signal      <- drop(X %*% beta)
  noise_sigma <- sqrt(stats::var(signal) / snr)

  if (heavy_noise) {
    raw_noise <- stats::rt(n, df = nu)
    raw_var   <- nu / (nu - 2)
    eps <- raw_noise * (noise_sigma / sqrt(raw_var))
  } else {
    eps <- stats::rnorm(n, mean = 0, sd = noise_sigma)
  }

  y <- signal + eps

  list(
    X           = X,
    y           = y,
    beta        = beta,
    true_support = as.integer(support),
    n           = n,
    p           = p,
    n_blocks    = n_blocks,
    rho         = rho,
    snr         = snr,
    nu          = nu,
    heavy_noise = heavy_noise
  )
}


# ==============================================================================
# dgp_ht_block_white_snr
# ==============================================================================

#' DGP for heavy-tailed AR(1)-block + identity-block data with SNR-calibrated
#' noise (cpp: dgp_ht_block_white).
#'
#' The p_total = p_ar + p_white predictors consist of:
#'   - p_ar columns: M independent AR(1)-Toeplitz t(nu) blocks of size p_ar/M
#'     (one chi-squared mixing variable per (row, block)).
#'   - p_white columns: i.i.d. Student-t(nu) white-noise columns (element-wise
#'     independent, mirroring cpp dgp_ht_block_white).
#'
#' Active predictors lie entirely within the AR blocks (one per block).
#' The white-noise columns allow p_total to stay fixed while M and Q vary.
#'
#' @param n           Number of observations.
#' @param p_ar        Number of AR(1) predictors. Must be divisible by n_blocks.
#' @param p_white     Number of white-noise predictors.
#' @param support     1-based integer vector of active predictor indices
#'                    (into the full p_ar + p_white column space).
#' @param amplitude   Signal coefficient for each active predictor. Default 1.
#' @param n_blocks    Number of AR(1) blocks.
#' @param rho         AR(1) within-block correlation parameter.
#' @param snr         Target signal-to-noise ratio Var(signal) / Var(noise).
#' @param nu          Degrees of freedom for the Student-t distribution. Default 3.
#' @param heavy_noise If TRUE, noise is also Student-t(nu); otherwise Gaussian.
#' @param seed        Integer RNG seed. NULL means do not set seed.
#'
#' @return A list with components: X, y, beta, true_support, n, p (= p_ar + p_white),
#'   rho, snr, nu, heavy_noise.
dgp_ht_block_white_snr <- function(n,
                                   p_ar,
                                   p_white,
                                   support,
                                   amplitude   = 1.0,
                                   n_blocks    = 5L,
                                   rho         = 0.8,
                                   snr         = 2.0,
                                   nu          = 3.0,
                                   heavy_noise = FALSE,
                                   seed        = NULL) {
  if (!is.null(seed)) set.seed(seed)

  p          <- p_ar + p_white
  block_size <- p_ar %/% n_blocks
  innov_sd   <- sqrt(1.0 - rho^2)

  X <- matrix(0.0, nrow = n, ncol = p)

  # AR(1) blocks with heavy-tailed scaling (one chi-sq draw per observation per block)
  for (m in seq_len(n_blocks)) {
    col_start <- (m - 1L) * block_size + 1L
    col_end   <- m * block_size

    Z_block        <- matrix(0.0, nrow = n, ncol = block_size)
    Z_block[, 1L]  <- stats::rnorm(n)
    if (block_size > 1L) {
      for (j in seq(2L, block_size)) {
        Z_block[, j] <- rho * Z_block[, j - 1L] + innov_sd * stats::rnorm(n)
      }
    }

    U_block <- stats::rchisq(n, df = nu)
    X[, col_start:col_end] <- Z_block * sqrt(nu / U_block)
  }

  # White-noise block: element-wise independent Student-t(nu) columns.
  if (p_white > 0L) {
    X[, (p_ar + 1L):p] <- matrix(stats::rt(n * p_white, df = nu),
                                 nrow = n, ncol = p_white)
  }

  beta <- numeric(p)
  beta[support] <- amplitude

  signal      <- drop(X %*% beta)
  noise_sigma <- sqrt(stats::var(signal) / snr)

  if (heavy_noise) {
    raw_noise <- stats::rt(n, df = nu)
    raw_var   <- nu / (nu - 2)
    eps <- raw_noise * (noise_sigma / sqrt(raw_var))
  } else {
    eps <- stats::rnorm(n, mean = 0, sd = noise_sigma)
  }

  y <- signal + eps

  list(
    X           = X,
    y           = y,
    beta        = beta,
    true_support = as.integer(support),
    n           = n,
    p           = p,
    p_ar        = p_ar,
    p_white     = p_white,
    n_blocks    = n_blocks,
    rho         = rho,
    snr         = snr,
    nu          = nu,
    heavy_noise = heavy_noise
  )
}

# ==============================================================================
# ==  DGP family: groups
# ==============================================================================

dgp_groups_toeplitz_leaf <- function(n, p, support, amplitude,
                                     groups, rho_levels, phi,
                                     snr = 2.0, seed = NULL) {
  stopifnot(length(groups) == length(rho_levels), length(groups) >= 2L)
  stopifnot(all(diff(rho_levels) < 0))                 # strictly decreasing
  stopifnot(rho_levels[1L] < 1.0, rho_levels[length(rho_levels)] > 0.0)
  stopifnot(phi >= 0.0, phi < 1.0)

  if (!is.null(seed)) set.seed(seed)

  L       <- length(groups)
  rho_ext <- c(rho_levels, 0.0)
  X       <- matrix(0.0, nrow = n, ncol = p)

  # Coarser latent-factor levels x = 2..L (one shared factor per group).
  for (x in 2:L) {
    loading <- sqrt(rho_ext[x] - rho_ext[x + 1L])
    for (gid in sort(unique(groups[[x]]))) {
      f    <- stats::rnorm(n)
      cols <- which(groups[[x]] == gid)
      X[, cols] <- X[, cols] + loading * f   # f (length n) recycled across cols
    }
  }

  # Finest level: Toeplitz(phi) within each leaf block groups[[1]].
  leaf_scale <- sqrt(rho_levels[1L] - rho_levels[2L])
  for (gid in sort(unique(groups[[1L]]))) {
    cols  <- which(groups[[1L]] == gid)
    g     <- length(cols)
    Sigma <- phi ^ abs(outer(seq_len(g), seq_len(g), "-"))
    Z     <- matrix(stats::rnorm(n * g), nrow = n, ncol = g)
    X[, cols] <- X[, cols] + leaf_scale * (Z %*% chol(Sigma))  # cov(Z %*% R) = R'R = Sigma
  }

  # Idiosyncratic noise.
  X <- X + sqrt(1.0 - rho_levels[1L]) * matrix(stats::rnorm(n * p), nrow = n, ncol = p)

  beta          <- numeric(p)
  beta[support] <- amplitude
  signal        <- drop(X %*% beta)
  noise_sigma   <- sqrt(stats::var(signal) / snr)
  y             <- signal + stats::rnorm(n, mean = 0, sd = noise_sigma)

  list(X = X, y = y, beta = beta, true_support = as.integer(support),
       n = n, p = p, s = length(support), snr = snr)
}
