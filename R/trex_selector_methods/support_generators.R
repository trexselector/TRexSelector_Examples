# ==============================================================================
# support_generators.R
# ==============================================================================
#
# Support-index construction helpers for TRex simulation demos.
# Sourced by each demo file — do not run standalone.
#
# Contents:
#   make_support_capped_spread()    — Deterministic evenly-spaced support.
#   make_support_random()           — Uniformly random support.
#   make_support_bt_one_per_block() — One active predictor per BT block.
# ==============================================================================


# ==============================================================================
# make_support_capped_spread
# ==============================================================================

#' Build an evenly-spaced support with a capped gap (deterministic).
#'
#' Mirrors the C++ `capped_spread_support()`:
#'   gap = min(p %/% s, max_gap)
#'   indices = {1, 1 + gap, 1 + 2*gap, ..., 1 + (s-1)*gap}  (1-based)
#'
#' @param s       Number of active predictors.
#' @param p       Total number of predictors.
#' @param max_gap Maximum gap between consecutive active indices. Default 10.
#'
#' @return Integer vector of length s (1-based R indices).
#'
#' @examples
#' make_support_capped_spread(10, 1000, 20)
#' # -> 1 + c(0,20,40,...,180)  =  {1, 21, 41, ..., 181}
make_support_capped_spread <- function(s, p, max_gap = 10L) {
  gap <- min(p %/% s, max_gap)
  1L + (0L:(s - 1L)) * gap
}


# ==============================================================================
# make_support_random
# ==============================================================================

#' Build a uniformly random support without replacement (stochastic).
#'
#' Mirrors the C++ `random_support()`:
#'   set.seed(seed + 500000L)
#'   sample(p, s) sorted ascending (1-based)
#'
#' @param s    Number of active predictors.
#' @param p    Total number of predictors.
#' @param seed Integer seed.
#'
#' @return Integer vector of length s (1-based R indices), sorted ascending.
#'
#' @examples
#' make_support_random(5, 500, seed = 2026)
make_support_random <- function(s, p, seed) {
  set.seed(seed + 500000L)
  sort(sample.int(p, size = s, replace = FALSE))
}


# ==============================================================================
# make_support_bt_one_per_block
# ==============================================================================

#' Build the natural BT support: exactly one active predictor per block.
#'
#' Blocks are defined the same way as in `dgp_bt_snr()`: block m (1-based)
#' covers 1-based column indices [(m-1)*block_size + 1, m*block_size].
#'
#' @param p        Total number of predictors. Must be divisible by n_blocks.
#' @param n_blocks Number of blocks (equals the number of active predictors).
#' @param seed     If NULL (default), picks the FIRST predictor of each block
#'                 (deterministic).  If an integer, picks a random predictor
#'                 from each block using set.seed(seed).
#'
#' @return Sorted integer vector of length n_blocks with 1-based indices.
make_support_bt_one_per_block <- function(p, n_blocks, seed = NULL) {
  stopifnot(p %% n_blocks == 0L)
  block_size   <- p %/% n_blocks
  block_starts <- (seq_len(n_blocks) - 1L) * block_size  # 0-based block starts

  if (is.null(seed)) {
    # Deterministic: first predictor of each block
    offsets <- rep(0L, n_blocks)
  } else {
    # Random: one predictor drawn uniformly from each block
    set.seed(seed)
    offsets <- sample.int(block_size, n_blocks, replace = TRUE) - 1L
  }

  sort(block_starts + offsets + 1L)  # convert to 1-based
}
