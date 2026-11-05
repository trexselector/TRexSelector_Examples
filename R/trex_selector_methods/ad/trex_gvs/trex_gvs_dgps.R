# ==============================================================================
# trex_gvs_dgps.R
# ==============================================================================
# Consolidated data-generating processes for the T-Rex+GVS demonstration suite
# (mirrors cpp trex_gvs_dgps.hpp). One section per scenario; each demo sources
# this single file.
#
#   Section 01 — Hastie equicorrelated groups        -> demo_trex_gvs_01
#   Section 02 — Scattered-grouped                   -> demo_trex_gvs_02
#   Section 03 — Mixed blocks (unequal, trap)        -> demo_trex_gvs_03
#   Section 04 — Negative traps                      -> demo_trex_gvs_04
#   Section 05 — Heavy-tailed (t3) equi blocks       -> demo_trex_gvs_05
#   Section 06 — AR(1) blocks                        -> demo_trex_gvs_06
#   Section 07 — ARMA mixed-structure blocks         -> demo_trex_gvs_07
#   Section 08 — Block-equicorrelated benchmark      -> demo_trex_gvs_08
# ==============================================================================


# ==============================================================================
# ==  DGP: hastie
# ==============================================================================


# ------------------------------------------------------------------------------
# dgp_hastie_snr
# ------------------------------------------------------------------------------
# Simulation-ready DGP for the Hastie GVS scenario with SNR-calibrated noise.
#
# The within-group correlation is rho = 1 / (1 + sd_x^2).
# To sweep rho, convert: sd_x = sqrt((1 - rho) / rho).
#
# The noise standard deviation sigma_y is derived from the target SNR:
#
#   signal   = X %*% beta
#   sigma_y  = sqrt( var(signal) / snr )      (n-1 denominator)
#
# @param n     Number of observations.
# @param p     Number of predictors (>= 150). Default 500.
# @param snr   Target signal-to-noise ratio Var(signal) / sigma_y^2.
# @param sd_x  Within-group noise sd. Controls rho = 1/(1+sd_x^2).
#              Default sqrt(0.01) => rho ~ 0.99.
# @param seed  Integer RNG seed. Default NULL.
#
# @return Named list:
#   X        n x p predictor matrix
#   y        length-n response
#   beta     length-p true coefficient vector (field consistent with .run_mc)
#   prior_groups  oracle group labels: blocks 1-3, background columns singletons
#   n, p, s  dimensions
#   snr      target SNR
#   sigma_y  realised noise standard deviation
#   sd_x     within-group noise sd used

dgp_hastie_snr <- function(n, p = 500, snr, sd_x = sqrt(0.01), seed = NULL) {

  if (!is.null(seed)) set.seed(seed)

  if (p < 150) {
    stop("'p' must be at least 150 to accommodate the 3 groups.")
  }

  # True coefficient vector
  beta <- rep(0, p)
  beta[1:150] <- 3

  # Generate predictors via three latent factors
  X <- matrix(0, nrow = n, ncol = p)

  Z1 <- rnorm(n, mean = 0, sd = 1)
  Z2 <- rnorm(n, mean = 0, sd = 1)
  Z3 <- rnorm(n, mean = 0, sd = 1)

  for (i in 1:50)    X[, i] <- Z1 + rnorm(n, mean = 0, sd = sd_x)
  for (i in 51:100)  X[, i] <- Z2 + rnorm(n, mean = 0, sd = sd_x)
  for (i in 101:150) X[, i] <- Z3 + rnorm(n, mean = 0, sd = sd_x)
  for (i in 151:p)   X[, i] <- rnorm(n, mean = 0, sd = 1)

  # SNR-calibrated noise: sigma_y = sqrt(var(signal) / snr)
  signal  <- as.numeric(X %*% beta)
  sigma_y <- sqrt(var(signal) / snr)    # var() uses n-1 denominator

  # Generate response
  y <- signal + rnorm(n, mean = 0, sd = sigma_y)

  list(
    X       = X,
    y       = y,
    beta    = beta,
    # Oracle groups: 1-3 for the three blocks, background columns as singletons
    prior_groups = c(rep(1:3, each = 50L), seq_len(p - 150L) + 3L),
    n       = n,
    p       = p,
    s       = 150L,
    snr     = snr,
    sigma_y = sigma_y,
    sd_x    = sd_x
  )
}

# ==============================================================================
# ==  DGP: scattered_grouped
# ==============================================================================


# ------------------------------------------------------------------------------
# dgp_scattered_grouped_snr
# ------------------------------------------------------------------------------
# Simulation-ready DGP for the scattered-grouped scenario with SNR-calibrated noise.
#
# n_groups groups of group_size variables each, randomly distributed across p
# columns, sharing a latent factor within each group (within-group correlation
# rho = 1 / (1 + sd_x^2)).
#
# The noise standard deviation sigma_y is derived from the target SNR:
#
#   signal   = X %*% beta
#   sigma_y  = sqrt( var(signal) / snr )      (n-1 denominator)
#
# @param n          Number of observations.
# @param p          Number of predictors. Default 500.
# @param snr        Target signal-to-noise ratio Var(signal) / sigma_y^2.
# @param sd_x       Within-group noise sd. Controls rho = 1/(1+sd_x^2).
#                   Default sqrt(0.01) => rho ~ 0.99.
# @param n_groups   Number of active groups. Default 3.
# @param group_size Number of variables per group. Default 50.
# @param seed       Integer RNG seed. Default NULL.
#
# @return Named list:
#   X           n x p predictor matrix
#   y           length-n response
#   beta        length-p true coefficient vector (field consistent with .run_mc)
#   n, p, s     dimensions (s = n_groups * group_size)
#   snr         target SNR
#   sigma_y     realised noise standard deviation
#   sd_x        within-group noise sd used
#   n_groups    number of active groups
#   group_size  variables per group

dgp_scattered_grouped_snr <- function(n, p = 500, snr, sd_x = sqrt(0.01),
                                      n_groups = 3L, group_size = 50L,
                                      seed = NULL) {

  if (!is.null(seed)) set.seed(seed)

  if (n_groups * group_size > p) {
    stop("'n_groups * group_size' must not exceed 'p'.")
  }

  # Randomly partition indices into groups and noise
  all_indices   <- seq_len(p)
  group_indices <- vector("list", n_groups)

  for (k in seq_len(n_groups)) {
    idx              <- sample(all_indices, size = group_size, replace = FALSE)
    group_indices[[k]] <- sort(idx)
    all_indices      <- setdiff(all_indices, idx)
  }

  # True coefficient vector — all group members are active (beta = 3)
  beta <- rep(0, p)
  X    <- matrix(0, nrow = n, ncol = p)

  for (k in seq_len(n_groups)) {
    Z_k <- rnorm(n, mean = 0, sd = 1)
    for (i in group_indices[[k]]) {
      X[, i]   <- Z_k + rnorm(n, mean = 0, sd = sd_x)
      beta[i]  <- 3
    }
  }

  # Remaining columns: independent white noise
  noise_indices <- sort(all_indices)
  if (length(noise_indices) > 0) {
    for (i in noise_indices) X[, i] <- rnorm(n, mean = 0, sd = 1)
  }

  # SNR-calibrated noise
  signal  <- as.numeric(X %*% beta)
  sigma_y <- sqrt(var(signal) / snr)

  y <- signal + rnorm(n, mean = 0, sd = sigma_y)

  list(
    X          = X,
    y          = y,
    beta       = beta,
    n          = n,
    p          = p,
    s          = as.integer(n_groups * group_size),
    snr        = snr,
    sigma_y    = sigma_y,
    sd_x       = sd_x,
    n_groups   = n_groups,
    group_size = group_size
  )
}

# ==============================================================================
# ==  DGP: mixed_blocks
# ==============================================================================


# ------------------------------------------------------------------------------
# dgp_mixed_blocks_snr
# ------------------------------------------------------------------------------
# Simulation-ready DGP: 3 active blocks (20, 50, 80) + 1 inactive block (65),
# randomly ordered and separated by white noise.
# Active set: s = 150 (blocks 1-3 only). Within-block correlation
# rho = 1 / (1 + sd_x^2).  sigma_y derived from target SNR.
#
# @param n     Number of observations.
# @param p     Number of predictors. Default 500.
# @param snr   Target signal-to-noise ratio.
# @param sd_x  Within-block noise sd. Default sqrt(0.01) => rho ~ 0.99.
# @param seed  Integer RNG seed. Default NULL.
#
# @return Named list: X, y, beta, n, p, s=150, snr, sigma_y, sd_x,
#                     block_sizes, is_active, block_order, block_indices.

dgp_mixed_blocks_snr <- function(n, p = 500, snr, sd_x = sqrt(0.01),
                                 seed = NULL) {

  if (!is.null(seed)) set.seed(seed)

  block_sizes   <- c(20L, 50L, 80L, 65L)
  is_active     <- c(TRUE, TRUE, TRUE, FALSE)
  num_blocks    <- length(block_sizes)
  total_grouped <- sum(block_sizes)

  block_order <- sample(seq_len(num_blocks))
  num_noise   <- p - total_grouped
  rem_noise   <- num_noise - (num_blocks - 1L)
  gaps        <- as.numeric(rmultinom(1, size = rem_noise,
                                      prob = rep(1, num_blocks + 1)))
  for (i in 2:num_blocks) gaps[i] <- gaps[i] + 1

  X             <- matrix(0, nrow = n, ncol = p)
  beta          <- rep(0, p)
  latent        <- lapply(seq_len(num_blocks), function(k) rnorm(n, 0, 1))
  current_idx   <- 1L
  block_indices <- vector("list", num_blocks)

  for (i in 1:(num_blocks + 1L)) {
    if (gaps[i] > 0) {
      end_idx <- current_idx + gaps[i] - 1L
      X[, current_idx:end_idx] <- matrix(rnorm(n * gaps[i], 0, 1), n, gaps[i])
      current_idx <- end_idx + 1L
    }
    if (i <= num_blocks) {
      b_id    <- block_order[i]
      b_size  <- block_sizes[b_id]
      end_idx <- current_idx + b_size - 1L
      Z       <- latent[[b_id]]
      for (c in current_idx:end_idx) X[, c] <- Z + rnorm(n, 0, sd_x)
      if (is_active[b_id]) beta[current_idx:end_idx] <- 3
      block_indices[[b_id]] <- current_idx:end_idx
      current_idx <- end_idx + 1L
    }
  }

  signal  <- as.numeric(X %*% beta)
  sigma_y <- sqrt(var(signal) / snr)
  y       <- signal + rnorm(n, 0, sigma_y)

  list(
    X             = X,
    y             = y,
    beta          = beta,
    n             = n,
    p             = p,
    s             = as.integer(sum(block_sizes[is_active])),
    snr           = snr,
    sigma_y       = sigma_y,
    sd_x          = sd_x,
    block_sizes   = block_sizes,
    is_active     = is_active,
    block_order   = block_order,
    block_indices = block_indices
  )
}

# ==============================================================================
# ==  DGP: neg_traps
# ==============================================================================


# ------------------------------------------------------------------------------
# dgp_neg_traps_snr
# ------------------------------------------------------------------------------
# Simulation-ready DGP: active group + two inactive negative-correlation traps.
# Group 1 (1-100):   active,     +Z1/-Z1, beta = +3/-3.  s = 100.
# Group 2 (101-200): Trap 1,     +Z2/-Z2, beta = 0.
# Noise   (201-300): white noise.
# Group 3 (301-360): Trap 2,     +Z3/-Z3, beta = 0.
# Noise   (361-p):   white noise.
# sigma_y derived from target SNR.
#
# @param n     Number of observations.
# @param p     Number of predictors. Default 500.
# @param snr   Target signal-to-noise ratio.
# @param sd_x  Within-group noise sd. Default sqrt(0.01) => rho ~ 0.99.
# @param seed  Integer RNG seed. Default NULL.
#
# @return Named list: X, y, beta, n, p, s=100, snr, sigma_y, sd_x.

dgp_neg_traps_snr <- function(n, p = 500, snr, sd_x = sqrt(0.01),
                              seed = NULL) {

  if (!is.null(seed)) set.seed(seed)

  Z1 <- rnorm(n, 0, 1)
  Z2 <- rnorm(n, 0, 1)
  Z3 <- rnorm(n, 0, 1)

  X    <- matrix(0, nrow = n, ncol = p)
  beta <- rep(0, p)

  # Group 1: active (1-100)
  for (i in 1:50) {
    X[, i] <- Z1 + rnorm(n, 0, sd_x)
    beta[i] <- 3
  }
  for (i in 51:100) {
    X[, i] <- -Z1 + rnorm(n, 0, sd_x)
    beta[i] <- -3
  }

  # Group 2: Trap 1 (101-200)
  for (i in 101:150) X[, i] <- Z2 + rnorm(n, 0, sd_x)
  for (i in 151:200) X[, i] <- -Z2 + rnorm(n, 0, sd_x)

  # White noise (201-300)
  for (i in 201:300) X[, i] <- rnorm(n, 0, 1)

  # Group 3: Trap 2 (301-360)
  for (i in 301:330) X[, i] <- Z3 + rnorm(n, 0, sd_x)
  for (i in 331:360) X[, i] <- -Z3 + rnorm(n, 0, sd_x)

  # White noise (361-p)
  if (p >= 361L) for (i in 361:p) X[, i] <- rnorm(n, 0, 1)

  signal  <- as.numeric(X %*% beta)
  sigma_y <- sqrt(var(signal) / snr)
  y       <- signal + rnorm(n, 0, sigma_y)

  list(
    X       = X,
    y       = y,
    beta    = beta,
    n       = n,
    p       = p,
    s       = 100L,
    snr     = snr,
    sigma_y = sigma_y,
    sd_x    = sd_x
  )
}

# ==============================================================================
# ==  DGP: t3_equi_blocks
# ==============================================================================


# ==============================================================================
# dgp_t3_equi_blocks_snr
# ==============================================================================
#' SNR-calibrated heavy-tailed equi-correlated blocks DGP.
#'
#' Four equicorrelated blocks (20/50/80/65; 3 active, 1 trap) with Student-t(df)
#' distributed latent factors and noise (scaled to variance 1).
#' 4 blocks (sizes 20, 50, 80, 65): 3 active (beta=3), 1 inactive trap.
#' Active set: s = 150.
#'
#' @param n   Number of observations.
#' @param p   Total predictors. Default 500.
#' @param snr Signal-to-noise ratio.
#' @param rho Within-block equi-correlation. Default 0.75.
#' @param df  Degrees of freedom for the t distribution. Default 3.
#' @param seed RNG seed. Default NULL.
#' @return list(X, y, beta, n, p, s, snr, sigma_y, rho, df, block_indices, block_order)
dgp_t3_equi_blocks_snr <- function(n, p = 500, snr, rho = 0.75, df = 3,
                                   seed = NULL) {

  if (!is.null(seed)) set.seed(seed)

  rt_scaled <- function(n_samp) rt(n_samp, df = df) / sqrt(df / (df - 2))

  group_sizes   <- c(20L, 50L, 80L, 65L)
  is_active     <- c(TRUE, TRUE, TRUE, FALSE)
  num_blocks    <- length(group_sizes)
  total_grouped <- sum(group_sizes)

  block_order <- sample(seq_len(num_blocks))

  num_noise <- p - total_grouped
  rem_noise <- num_noise - (num_blocks - 1L)
  gaps      <- as.numeric(rmultinom(1, size = rem_noise,
                                    prob = rep(1, num_blocks + 1)))
  for (i in 2:num_blocks) gaps[i] <- gaps[i] + 1L

  X             <- matrix(0, nrow = n, ncol = p)
  beta          <- rep(0, p)
  latent_factors <- lapply(seq_len(num_blocks), function(i) rt_scaled(n))

  current_idx   <- 1L
  block_indices <- vector("list", num_blocks)
  w_Z           <- sqrt(rho)
  w_E           <- sqrt(1 - rho)

  for (i in seq_len(num_blocks + 1L)) {
    if (gaps[i] > 0) {
      end_idx <- current_idx + gaps[i] - 1L
      X[, current_idx:end_idx] <- rt_scaled(n * gaps[i])
      current_idx <- end_idx + 1L
    }
    if (i <= num_blocks) {
      b_id   <- block_order[i]
      b_size <- group_sizes[b_id]
      end_idx <- current_idx + b_size - 1L
      Z <- latent_factors[[b_id]]
      for (col in current_idx:end_idx) {
        X[, col] <- w_Z * Z + w_E * rt_scaled(n)
      }
      if (is_active[b_id]) beta[current_idx:end_idx] <- 3
      block_indices[[b_id]] <- current_idx:end_idx
      current_idx <- end_idx + 1L
    }
  }

  signal  <- as.numeric(X %*% beta)
  sigma_y <- sqrt(var(signal) / snr)
  y       <- signal + rt_scaled(n) * sigma_y

  list(
    X             = X,
    y             = y,
    beta          = beta,
    n             = n,
    p             = p,
    s             = 150L,
    snr           = snr,
    sigma_y       = sigma_y,
    rho           = rho,
    df            = df,
    block_indices = block_indices,
    block_order   = block_order
  )
}

# ==============================================================================
# ==  DGP: ar1_blocks
# ==============================================================================


# ==============================================================================
# dgp_ar1_blocks_snr
# ==============================================================================
#' SNR-calibrated block-structured AR(1) DGP.
#'
#' 4 blocks (sizes 20, 50, 80, 65): 3 active (beta=3), 1 inactive trap.
#' Within each block: Cor(X_j, X_k) = rho^|j-k|.
#' Active set: s = 150.
#'
#' @param n   Number of observations.
#' @param p   Total predictors. Default 500.
#' @param snr Signal-to-noise ratio.
#' @param rho AR(1) coefficient (lag-1 correlation). Default 0.85.
#' @param seed RNG seed. Default NULL.
#' @return list(X, y, beta, n, p, s, snr, sigma_y, rho, block_indices, block_order)
dgp_ar1_blocks_snr <- function(n, p = 500, snr, rho = 0.85, seed = NULL) {

  if (!is.null(seed)) set.seed(seed)

  group_sizes   <- c(20L, 50L, 80L, 65L)
  is_active     <- c(TRUE, TRUE, TRUE, FALSE)
  num_blocks    <- length(group_sizes)
  total_grouped <- sum(group_sizes)

  block_order <- sample(seq_len(num_blocks))

  num_noise <- p - total_grouped
  rem_noise <- num_noise - (num_blocks - 1L)
  gaps      <- as.numeric(rmultinom(1, size = rem_noise,
                                    prob = rep(1, num_blocks + 1)))
  for (i in 2:num_blocks) gaps[i] <- gaps[i] + 1L

  X             <- matrix(0, nrow = n, ncol = p)
  beta          <- rep(0, p)
  current_idx   <- 1L
  block_indices <- vector("list", num_blocks)

  for (i in seq_len(num_blocks + 1L)) {
    if (gaps[i] > 0) {
      end_idx <- current_idx + gaps[i] - 1L
      X[, current_idx:end_idx] <- rnorm(n * gaps[i])
      current_idx <- end_idx + 1L
    }
    if (i <= num_blocks) {
      b_id   <- block_order[i]
      b_size <- group_sizes[b_id]
      end_idx <- current_idx + b_size - 1L

      # AR(1) covariance: Sigma[r,c] = rho^|r-c|
      r_idx    <- seq_len(b_size)
      Sigma_ar1 <- rho^abs(outer(r_idx, r_idx, `-`))
      Z         <- matrix(rnorm(n * b_size), nrow = n, ncol = b_size)
      X[, current_idx:end_idx] <- Z %*% chol(Sigma_ar1)

      if (is_active[b_id]) beta[current_idx:end_idx] <- 3
      block_indices[[b_id]] <- current_idx:end_idx
      current_idx <- end_idx + 1L
    }
  }

  signal  <- as.numeric(X %*% beta)
  sigma_y <- sqrt(var(signal) / snr)
  y       <- signal + rnorm(n, sd = sigma_y)

  list(
    X             = X,
    y             = y,
    beta          = beta,
    n             = n,
    p             = p,
    s             = 150L,
    snr           = snr,
    sigma_y       = sigma_y,
    rho           = rho,
    block_indices = block_indices,
    block_order   = block_order
  )
}

# ==============================================================================
# ==  DGP: arma_blocks
# ==============================================================================


# ------------------------------------------------------------------------------
# .sim_arma
# ------------------------------------------------------------------------------
#' Simulate one ARMA(p,q) series via a direct recursion with burn-in.
#'
#' Mirrors the inline generator in the cpp DGP: x_t = eps_t + sum_k ar[k] x_{t-k}
#' + sum_k ma[k] eps_{t-k}, with a burn-in of max(100, p + q + 10) discarded.
#' Unlike stats::arima.sim, it does not enforce stationarity, so the callers pass
#' stationary coefficients (the ARMA(2,1) block uses AR terms 5:1 that sum to
#' ar_coef); downstream column standardisation keeps the columns unit-variance.
#'
#' @param model List with optional numeric entries `ar` and `ma`.
#' @param len   Length of the returned series.
#' @return Numeric vector of length `len`.
.sim_arma <- function(model, len) {
  ar   <- if (!is.null(model$ar)) model$ar else numeric(0)
  ma   <- if (!is.null(model$ma)) model$ma else numeric(0)
  p_ar <- length(ar)
  q_ma <- length(ma)
  burn <- max(100L, p_ar + q_ma + 10L)
  total <- burn + len

  eps <- rnorm(total)
  x   <- numeric(total)
  for (t in seq_len(total)) {
    val <- eps[t]
    for (k in seq_len(p_ar)) if (t - k >= 1L) val <- val + ar[k] * x[t - k]
    for (k in seq_len(q_ma)) if (t - k >= 1L) val <- val + ma[k] * eps[t - k]
    x[t] <- val
  }
  x[(burn + 1L):total]
}


# ==============================================================================
# dgp_arma_blocks_snr
# ==============================================================================
#' SNR-calibrated ARMA-structured blocks DGP.
#'
#' 4 blocks governed by different ARMA processes (column-standardized):
#'   Block 1 — AR(1);  Block 2 — MA(3);  Block 3 — ARMA(2,1);  Block 4 (trap) — AR(1).
#' 3 active blocks (sizes 20, 50, 80), 1 inactive trap (size 65).
#' Active set: s = 150.
#'
#' @param n       Number of observations.
#' @param p       Total predictors. Default 500.
#' @param snr     Signal-to-noise ratio.
#' @param ar_coef AR coefficient shared by blocks 1, 3, and the trap. Default 0.8.
#' @param ma_coefs MA coefficients for blocks 2 and 3. Default c(0.5, 0.3, 0.1).
#' @param seed    RNG seed. Default NULL.
#' @return list(X, y, beta, n, p, s, snr, sigma_y, ar_coef, block_indices, block_order)
dgp_arma_blocks_snr <- function(n, p = 500, snr, ar_coef = 0.8,
                                ma_coefs = c(0.5, 0.3, 0.1),
                                seed = NULL) {

  if (!is.null(seed)) set.seed(seed)

  group_sizes   <- c(20L, 50L, 80L, 65L)
  is_active     <- c(TRUE, TRUE, TRUE, FALSE)
  models        <- list(
    list(ar = ar_coef),
    list(ma = ma_coefs),
    # ARMA(2,1): AR terms 5:1 summing to ar_coef, stationary for ar_coef in (0, 1)
    list(ar = c(ar_coef * 5 / 6, ar_coef / 6), ma = ma_coefs[1L]),
    list(ar = ar_coef)
  )
  num_blocks    <- length(group_sizes)
  total_grouped <- sum(group_sizes)

  block_order <- sample(seq_len(num_blocks))

  num_noise <- p - total_grouped
  rem_noise <- num_noise - (num_blocks - 1L)
  gaps      <- as.numeric(rmultinom(1, size = rem_noise,
                                    prob = rep(1, num_blocks + 1)))
  for (i in 2:num_blocks) gaps[i] <- gaps[i] + 1L

  X             <- matrix(0, nrow = n, ncol = p)
  beta          <- rep(0, p)
  current_idx   <- 1L
  block_indices <- vector("list", num_blocks)

  for (i in seq_len(num_blocks + 1L)) {
    if (gaps[i] > 0) {
      end_idx <- current_idx + gaps[i] - 1L
      X[, current_idx:end_idx] <- rnorm(n * gaps[i])
      current_idx <- end_idx + 1L
    }
    if (i <= num_blocks) {
      b_id   <- block_order[i]
      b_size <- group_sizes[b_id]
      end_idx <- current_idx + b_size - 1L
      for (row in seq_len(n)) {
        X[row, current_idx:end_idx] <- .sim_arma(models[[b_id]], b_size)
      }
      # Standardize columns to variance 1
      X[, current_idx:end_idx] <- scale(X[, current_idx:end_idx])
      if (is_active[b_id]) beta[current_idx:end_idx] <- 3
      block_indices[[b_id]] <- current_idx:end_idx
      current_idx <- end_idx + 1L
    }
  }

  signal  <- as.numeric(X %*% beta)
  sigma_y <- sqrt(var(signal) / snr)
  y       <- signal + rnorm(n, sd = sigma_y)

  list(
    X             = X,
    y             = y,
    beta          = beta,
    n             = n,
    p             = p,
    s             = 150L,
    snr           = snr,
    sigma_y       = sigma_y,
    ar_coef       = ar_coef,
    block_indices = block_indices,
    block_order   = block_order
  )
}

# ==============================================================================
# ==  DGP: block_equicorr
# ==============================================================================


#' Block-equicorrelated benchmark DGP.
#'
#' Latent-factor construction with one factor per block, giving exactly
#' unit-variance columns with within-block correlation rho:
#'   X_j = sqrt(rho) * Z_block(j) + sqrt(1 - rho) * E_j,   Z, E ~ N(0, 1).
#' All p = G * block_size columns are tiled by the G blocks; n_active_blocks of
#' them are chosen at random to carry signal (coefficient magnitude b, random
#' sign per block). The within-block active pattern is set by `scenario`.
#'
#' @param n               Number of observations.
#' @param p               Number of predictors (must equal G * block_size).
#' @param G               Number of blocks. Default 100.
#' @param block_size      Variables per block. Default 5.
#' @param n_active_blocks Number of active blocks (<= G). Default 10.
#' @param rho             Within-block correlation in (0, 1).
#' @param snr             Target signal-to-noise ratio.
#' @param scenario        "INDIVIDUAL", "REPRESENTATIVE", or "WHOLE".
#' @param seed            RNG seed. Default NULL.
#' @param b               Nonzero coefficient magnitude. Default 3.
#'
#' @return list with X, y, beta, prior_groups (contiguous 1..G group labels),
#'   active_blocks (1-based block IDs), true_support, n, p, G, block_size,
#'   n_active_blocks, s, snr, sigma_y, rho, scenario.
dgp_block_equicorr <- function(n, p, G = 100L, block_size = 5L,
                               n_active_blocks = 10L, rho, snr,
                               scenario = c("INDIVIDUAL", "REPRESENTATIVE",
                                            "WHOLE"),
                               seed = NULL, b = 3.0) {
  scenario <- match.arg(scenario)
  if (p != G * block_size)
    stop("dgp_block_equicorr: p must equal G * block_size.")
  if (n_active_blocks > G)
    stop("dgp_block_equicorr: n_active_blocks must be <= G.")
  if (rho <= 0 || rho >= 1)
    stop("dgp_block_equicorr: rho must be in (0, 1).")
  if (!is.null(seed)) set.seed(seed)

  # --- Design matrix: one latent factor per block (unit-variance columns) ---
  Z <- matrix(rnorm(n * G), n, G)
  X <- matrix(0, n, p)
  prior_groups <- integer(p)
  sqrt_rho <- sqrt(rho)
  sqrt_omr <- sqrt(1 - rho)
  for (k in seq_len(G)) {
    cols <- ((k - 1L) * block_size + 1L):(k * block_size)
    prior_groups[cols] <- k
    X[, cols] <- sqrt_rho * Z[, k] + sqrt_omr * matrix(rnorm(n * block_size),
                                                       n, block_size)
  }

  # --- Choose active blocks without replacement ---
  active_blocks <- sort(sample.int(G, n_active_blocks))

  # --- Build beta according to the truth scenario ---
  beta <- numeric(p)
  for (blk in active_blocks) {
    base <- (blk - 1L) * block_size          # 0-based offset within column space
    sign <- if (sample.int(2L, 1L) == 1L) 1 else -1
    if (scenario == "INDIVIDUAL") {
      beta[base + 1L] <- sign * b
    } else if (scenario == "REPRESENTATIVE") {
      n_act <- sample(2:3, 1L)
      pos   <- sample.int(block_size, n_act)
      beta[base + pos] <- sign * b
    } else {  # WHOLE
      beta[(base + 1L):(base + block_size)] <- sign * b
    }
  }
  true_support <- which(beta != 0)

  # --- SNR-calibrated response (biased signal variance, as in the cpp DGP) ---
  signal  <- as.numeric(X %*% beta)
  sig_var <- mean((signal - mean(signal))^2)
  sigma_y <- sqrt(sig_var / snr)
  y       <- signal + rnorm(n, sd = sigma_y)

  list(
    X               = X,
    y               = y,
    beta            = beta,
    prior_groups    = prior_groups,
    active_blocks   = active_blocks,
    true_support    = true_support,
    n               = n,
    p               = p,
    G               = G,
    block_size      = block_size,
    n_active_blocks = n_active_blocks,
    s               = length(true_support),
    snr             = snr,
    sigma_y         = sigma_y,
    rho             = rho,
    scenario        = scenario
  )
}
