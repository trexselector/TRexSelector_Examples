# ==============================================================================
# demo_trex_scr_01_mc_sim_screen_trex.R
# ==============================================================================
#
# Screen-TRex Selector, demo 01: Monte Carlo baseline on an i.i.d. Gaussian design.
#
# Screen-TRex extends the classical T-Rex Selector with a fast variable-
# *screening* step for ultra-high-dimensional data. It thresholds the dummy-based
# voting proportion Phi_j that T-Rex accumulates internally:
#   * Ordinary    rule: select {j : Phi_j > 0.5}  (a majority vote).
#   * Bootstrap-CI rule: build a bootstrap confidence band around the estimated
#     FDR curve (R_boot replicates) and pick the threshold from that band.
# Unlike the classical T-Rex selector, screening has *no target-FDR parameter*;
# it reports its own estimated FDR alongside the selection.
#
# Mirrors:
#   Python/trex_selector_methods/ad/trex_screening/demo_trex_scr_01_mc_sim_screen_trex/
#       demo_trex_scr_01_mc_sim_screen_trex.py
#   cpp/trex_selector_methods/ad/trex_screening/demo_trex_scr_01_mc_sim_screen_trex/
#       demo_trex_scr_01_mc_sim_screen_trex.cpp
#
# Demo content (single Monte Carlo SNR sweep, in-memory):
#   Ordinary vs. Bootstrap-CI Screen-TRex over an SNR grid, reporting realized
#   FDR, TPR, and the procedure's self-reported estimated FDR.
#
# The Monte Carlo count matches the cpp/Python reference (num_MC = 200).
# Results are statistically comparable, not bit-identical (different RNG
# streams). Support indices are 1-based (R convention).
#
# Usage: Rscript demo_trex_scr_01_mc_sim_screen_trex.R [num_worker_cores]
#   (default TREX_NUM_WORKERS, else 8 worker cores; MC trials run on PSOCK workers via mc_map)
# ==============================================================================

library(TRexSelector)
library(parallel)

# ==============================================================================
# Setup
# ==============================================================================

# Worker processes: first CLI argument, else TREX_NUM_WORKERS, else 8 (0 = all).
num_cores <- local({
  a <- suppressWarnings(as.integer(commandArgs(trailingOnly = TRUE)[1L]))
  if (is.na(a)) a <- suppressWarnings(as.integer(Sys.getenv("TREX_NUM_WORKERS", "8")))
  if (is.na(a)) 8L else if (a < 1L) parallel::detectCores() else a
})

this_dir_ <- tryCatch(
  dirname(normalizePath(sys.frame(1L)$ofile)),
  error = function(e) {
    args <- commandArgs(trailingOnly = FALSE)
    file_arg <- grep("--file=", args, value = TRUE)
    if (length(file_arg) > 0) dirname(normalizePath(sub("--file=", "", file_arg[1]))) else "."
  }
)
source(file.path(this_dir_, "..", "..", "..", "mc_runtime.R"))  # PSOCK workers, cell cache
source(file.path(this_dir_, "..", "trex_scr_sim_utils.R"))

OUT_DIR <- file.path(this_dir_, "simulation_results", "data")

# Monte Carlo count mirrors the cpp/Python reference (num_MC = 200).
NUM_MC <- 200L

N <- 300L
P <- 1000L
SUPPORT_SIZE <- 10L
SNR_GRID <- c(0.01, 0.1, 0.2, 0.5, 0.6, 1.0, 2.0, 5.0)


# ==============================================================================
# Monte Carlo SNR sweep — Ordinary vs. Bootstrap-CI
# ==============================================================================

demo_screen_trex_monte_carlo <- function() {
  cat("\n", strrep("=", 70L), "\n", sep = "")
  cat("Demo: Screen-TRex Monte Carlo Comparison\n")
  cat(strrep("=", 70L), "\n", sep = "")
  cat("High-dimensional (p > n)\n")
  cat(sprintf("n = %d, p = %d, s = %d, num_MC = %d\n\n",
              N, P, SUPPORT_SIZE, NUM_MC))

  methods <- default_scr_methods()
  method_names <- vapply(methods, function(m) m$name, character(1))

  init <- function() {
    setNames(
      lapply(method_names, function(x) numeric(length(SNR_GRID))), method_names
    )
  }
  fdr_map <- init()
  tpr_map <- init()
  est_fdr_map <- init()

  trex_ctrl <- make_scr_trex_ctrl()

  for (m in methods) {
    cat(strrep("=", 70L), "\n", sep = "")
    cat(sprintf("Method: %s\n", m$name))
    cat(strrep("=", 70L), "\n", sep = "")
    for (si in seq_along(SNR_GRID)) {
      snr <- SNR_GRID[si]
      make_data <- function(seed) {
        set.seed(seed + 500000L)                 # isolated support RNG
        sup <- sample.int(P, SUPPORT_SIZE)
        dgp_iid_snr(N, P, sup, rep(1, SUPPORT_SIZE), snr, seed = seed)
      }
      base_seed <- 24L + (si - 1L) * 1000L  # mirrors cpp base_seed = 24 + snr_idx * 1000
      res <- run_mc_screen(
        NUM_MC, sprintf("%s  SNR=%.2f", m$name, snr), make_data, trex_ctrl,
        screen_args = list(trex_method = m$trex_method, use_bootstrap_CI = m$bootstrap),
        base_seed = base_seed, num_cores = num_cores
      )
      fdr_map[[m$name]][si]     <- res$avg_fdr
      tpr_map[[m$name]][si]     <- res$avg_tpr
      est_fdr_map[[m$name]][si] <- res$avg_est_fdr
    }
  }

  file_stem <- sprintf("scr_screen_trex_snr_n%d_p%d", N, P)
  save_and_print_scr_mc(NUM_MC, OUT_DIR, file_stem, SNR_GRID,
                        method_names, fdr_map, tpr_map, est_fdr_map)
}


# ==============================================================================
# Main
# ==============================================================================

demo_screen_trex_monte_carlo()
cat("\ndemo_trex_scr_01_mc_sim_screen_trex.R complete.\n")

# All results are written: the cell cache of a stopped run is no longer needed.
cell_cache_done()
