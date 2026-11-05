# ==============================================================================
# demo_trex_02_mc_sim_fixed_support.R
# ==============================================================================
#
# Classical T-Rex selector demo. Part 02 demonstrates a Monte Carlo (MC)
# simulation for the linear model with Gaussian noise.
# The focus of this demo is on a fixed true support to compare the base
# tsolvers available within the T-Rex framework.
#
# The file mirrors:
# cpp/trex_selector_methods/ad/trex_core/demo_trex_02_mc_sim_fixed_support.cpp
#
# Demo content:
# ----------------------------------------------------------------
#
#  - 14 solvers; SNR sweep {0.1, 0.5, 1.0, 2.0, 5.0}.
#  - Fixed support drawn once with seed 24.
#  - Reports averaged FDR / TPR per solver x SNR.
#
# ==============================================================================

library(TRexSelector)
library(parallel)

# ==============================================================================
# Setup
# ==============================================================================

this_dir_ <- tryCatch(
  dirname(normalizePath(sys.frame(1L)$ofile)),
  error = function(e) {
    args <- commandArgs(trailingOnly = FALSE)
    file_arg <- grep("--file=", args, value = TRUE)
    if (length(file_arg) > 0) dirname(normalizePath(sub("--file=", "", file_arg[1]))) else "."
  }
)
source(file.path(this_dir_, "..", "..", "..", "mc_runtime.R"))  # PSOCK workers, cell cache
source(file.path(this_dir_, "..", "..", "..", "support_generators.R"))
source(file.path(this_dir_, "..", "..", "..", "simulation_utils.R"))
source(file.path(this_dir_, "..", "trex_sim_utils.R"))


# ==============================================================================
# Global Simulation Parameters
# ==============================================================================
OUT_DIR <- file.path(this_dir_, "simulation_results", "data")
dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)

# Worker processes: first CLI argument, else TREX_NUM_WORKERS, else 8 (0 = all).
num_cores <- local({
  a <- suppressWarnings(as.integer(commandArgs(trailingOnly = TRUE)[1L]))
  if (is.na(a)) a <- suppressWarnings(as.integer(Sys.getenv("TREX_NUM_WORKERS", "8")))
  if (is.na(a)) 8L else if (a < 1L) parallel::detectCores() else a
})
NUM_MC <- .resolve_num_mc(200L)  # TREX_NUM_MC for smoke runs

# Run flag
run_trex_02 <- TRUE

# ==============================================================================
# MC simulation — fixed true support, solver comparison
# ==============================================================================

trx_02_mc_sim_fixed_support <- function(num_MC = NUM_MC, high_dim = TRUE) {

  n <- if (high_dim) 300L else 1000L
  p <- if (high_dim) 1000L else 300L
  s <- 10L
  tFDR <- 0.1
  snr_values   <- c(0.1, 0.5, 1.0, 2.0, 5.0)
  stagnant_sw  <- 7L

  dim_label <- if (high_dim) "High-dimensional (p > n)" else "Low-dimensional (n > p)"
  cat("\n", strrep("=", 70L), "\n", sep = "")
  cat("MC Simulation \u2014 Fixed True Support  |  ", dim_label, "\n", sep = "")
  cat(sprintf("n = %d,  p = %d,  s = %d,  num_MC = %d\n", n, p, s, num_MC))

  # Base control (per-solver params added in .make_trex_ctrl)
  base_ctrl <- list(
    K                        = 20L,
    max_dummy_multiplier     = 10L,
    use_max_T_stop           = TRUE,
    lloop_strategy           = "STANDARD",
    tloop_stagnation_stop    = TRUE,
    tloop_max_stagnant_steps = stagnant_sw,
    parallel_rnd_experiments = FALSE,
    opt_threshold            = 0.75,
    use_memory_mapping       = FALSE
  )

  true_support <- make_support_random(s, p, seed = 24L)
  cat(sprintf("True support (cardinality %d): {%s}\n\n",
              s, paste(true_support, collapse = ", ")))
  true_coefs <- rep(1.0, s)

  # TREX_SOLVERS selects a subset (mc_runtime.R), its tables get a suffix
  solvers <- .selected_solvers(SOLVERS_DEFAULT)

  n_sol <- length(solvers)
  n_snr <- length(snr_values)
  fdr_mat <- matrix(0.0, nrow = n_sol, ncol = n_snr)
  tpr_mat <- matrix(0.0, nrow = n_sol, ncol = n_snr)

  rownames(fdr_mat) <- rownames(tpr_mat) <- vapply(solvers, `[[`, "", "name")
  colnames(fdr_mat) <- colnames(tpr_mat) <- as.character(snr_values)

  stem <- sprintf("demo_trex_02_mc_sim_fixed_support_trex_results_n%d_p%d_stagnation_window_%d",
                  n, p, stagnant_sw)
  stem <- paste0(stem, solver_file_suffix())

  # Simulation loop: iterate over solvers, then SNR values.
  # Results are re-saved after each solver (growing file).
  # An abort mid-run keeps the solvers already finished.
  for (si in seq_along(solvers)) {
    solver <- solvers[[si]]
    ctrl   <- .make_trex_ctrl(solver, base_ctrl)
    cat(sprintf("Solver: %s\n", solver$name))
    for (snr_idx in seq_along(snr_values)) {

      snr        <- snr_values[[snr_idx]]
      base_seed  <- 24L + (snr_idx - 1L) * 1000L

      make_data <- local({
        current_snr <- snr
        function(mc, trial_seed) {
          dgp_gauss_snr(n, p, true_support, coefs = true_coefs,
                        snr = current_snr, seed = trial_seed)
        }
      })
      lbl    <- sprintf("SNR=%.2f [%s]", snr, solver$name)
      result <- .run_mc_trex(make_data, ctrl, tFDR, base_seed, num_MC, num_cores,
                             track_L_T = FALSE, label = lbl)
      fdr_mat[si, snr_idx] <- result$mean_FDR
      tpr_mat[si, snr_idx] <- result$mean_TPR
    }
    cat("\n")

    # Incremental save over the solvers completed so far (abort-safe).
    .save_and_print_mc(OUT_DIR, stem, num_MC,
                       rownames(fdr_mat)[seq_len(si)], snr_values,
                       fdr_mat[seq_len(si), , drop = FALSE],
                       tpr_mat[seq_len(si), , drop = FALSE])
  }
  cat("\n")
}


# ==============================================================================
# Main
# --------------
# Run Part 02
# ==============================================================================

if (run_trex_02) trx_02_mc_sim_fixed_support(NUM_MC, high_dim = TRUE)
cat("\ndemo_trex_02_mc_sim_fixed_support.R complete.\n")

# All results are written: the cell cache of a stopped run is no longer needed.
cell_cache_done()
