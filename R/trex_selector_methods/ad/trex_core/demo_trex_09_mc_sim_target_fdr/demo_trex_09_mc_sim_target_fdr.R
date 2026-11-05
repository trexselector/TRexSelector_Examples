# ==============================================================================
# demo_trex_09_mc_sim_target_fdr.R
# ==============================================================================
#
# Classical T-Rex selector demo: achieved FDR and TPR as a function of the
# target FDR, for three base solvers (TLARS, TOMP, TAFS) and four SNRs.
#
# Mirrors:
# cpp/trex_selector_methods/ad/trex_core/demo_trex_09_mc_sim_target_fdr.cpp
#
# The T-Rex selector controls the FDR at the target level tFDR. The demo sweeps
# tFDR and records the achieved FDR (the mean false discovery proportion) and
# TPR, so FDR control shows as the FDR curve on or below the diagonal
# FDR = tFDR, and the TPR curve shows what each level of control costs in power.
#
# Data as in demo 03: n = 300, p = 1,000, s = 10 unit coefficients on a support
# redrawn per trial, Normal predictors and noise.
# Selector as in demo 05: K = 20, max_dummy_multiplier = 10, Normal dummies,
# STANDARD L-loop, use_max_T_stop, stagnation stop AUTO (off for TLARS, on for
# TOMP and TAFS), rho_afs = 0.3 for TAFS.
#
# Grid: SNR in {0.5, 1, 2, 5} x tFDR in {0.01, 0.025, 0.05, 0.075, 0.1, 0.15,
# 0.2, 0.25, 0.3, 0.4, 0.5}, num_MC = 200 trials per cell.
#
# Seeds: trial mc of an SNR uses the data seed 24 + 1000 * (snr_index - 1) +
# mc - 1 for every tFDR and every solver, so the curves along tFDR compare the
# same data sets, and the selector is seeded with the same trial seed
# (reproducible).
#
# Output (simulation_results/data/): one table per solver in
# demo_trex_09_mc_sim_target_fdr.txt and the tidy CSV
# demo_trex_09_mc_sim_target_fdr.csv with the columns
# solver, snr, tfdr, metric, mean, sd (metrics FDR, TPR, AvgL, AvgT; sd for FDR
# and TPR only). Every finished cell is cached, so a stopped run resumes.
#
# Command line: Rscript demo_trex_09_mc_sim_target_fdr.R [workers] [num_MC]
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

OUT_DIR   <- file.path(this_dir_, "simulation_results", "data")
FILE_STEM <- "demo_trex_09_mc_sim_target_fdr"
dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)

# Worker processes: first CLI argument, else TREX_NUM_WORKERS, else 8 (0 = all).
num_cores <- local({
  a <- suppressWarnings(as.integer(commandArgs(trailingOnly = TRUE)[1L]))
  if (is.na(a)) a <- suppressWarnings(as.integer(Sys.getenv("TREX_NUM_WORKERS", "8")))
  if (is.na(a)) 8L else if (a < 1L) parallel::detectCores() else a
})
# Trials per cell: second CLI argument (e.g. 2 for a smoke run), else 200.
NUM_MC <- local({
  a <- suppressWarnings(as.integer(commandArgs(trailingOnly = TRUE)[2L]))
  if (is.na(a) || a < 1L) 200L else a
})

N_OBS       <- 300L
P_VARS      <- 1000L
S_TRUE      <- 10L
SEED_OFFSET <- 24L  # as demos 02-05

SNR_VALUES  <- c(0.5, 1.0, 2.0, 5.0)
TFDR_VALUES <- c(0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5)

# Base solvers (rho_afs only meaningful for TAFS)
SOLVERS <- list(
  list(name = "TLARS", solver = "TLARS", lambda2 = 0.0, rho_afs = 0.0,
       ncgmp_variant = 0L),
  list(name = "TOMP",  solver = "TOMP",  lambda2 = 0.0, rho_afs = 0.0,
       ncgmp_variant = 0L),
  list(name = "TAFS",  solver = "TAFS",  lambda2 = 0.0, rho_afs = 0.3,
       ncgmp_variant = 0L)
)


# ==============================================================================
# Output: per-solver table (console + .txt) and tidy CSV
# ==============================================================================

#' Print the per-solver tables and write the .txt and .csv files.
#'
#' @param results results[[solver_name]][[si]][[ti]] is the .run_mc_trex()
#'                list of one cell.
#' @param num_MC  Trials per cell.
.save_results_09 <- function(results, num_MC) {
  label_w <- 16L
  col_w   <- 8L
  lines <- c(
    "",
    strrep("=", 78L),
    sprintf("=== T-Rex achieved FDR / TPR vs. target FDR (mean over %d trials) ===", num_MC),
    sprintf("    n = %d, p = %d, s = %d, K = 20, STANDARD L-loop, Normal dummies",
            N_OBS, P_VARS, S_TRUE),
    strrep("=", 78L)
  )
  for (solver in SOLVERS) {
    res <- results[[solver$name]]
    lines <- c(lines, "", solver$name,
               paste0(formatC("  tFDR", width = -label_w),
                      paste(formatC(TFDR_VALUES, width = col_w, format = "f", digits = 3L),
                            collapse = "")))
    for (si in seq_along(SNR_VALUES)) {
      for (metric in c("FDR", "TPR")) {
        vals <- vapply(res[[si]], function(cell) cell[[paste0("mean_", metric)]], 0.0)
        label <- sprintf("  %s SNR %g", metric, SNR_VALUES[[si]])
        lines <- c(lines, paste0(formatC(label, width = -label_w),
                                 paste(formatC(vals, width = col_w, format = "f", digits = 3L),
                                       collapse = "")))
      }
    }
  }
  lines <- c(lines, "")
  text <- paste0(paste(lines, collapse = "\n"), "\n")
  cat(text, "\n", sep = "")

  txt_path <- file.path(OUT_DIR, paste0(FILE_STEM, ".txt"))
  writeLines(text, txt_path, sep = "")
  cat(sprintf("[Info] TXT saved to: %s\n", txt_path))

  rows <- character(0)
  for (solver in SOLVERS) {
    res <- results[[solver$name]]
    for (si in seq_along(SNR_VALUES)) {
      for (ti in seq_along(TFDR_VALUES)) {
        cell <- res[[si]][[ti]]
        key <- sprintf("%s,%.6f,%.6f,", solver$name, SNR_VALUES[[si]], TFDR_VALUES[[ti]])
        rows <- c(rows,
                  sprintf("%sFDR,%.6f,%.6f", key, cell$mean_FDR, cell$sd_FDR),
                  sprintf("%sTPR,%.6f,%.6f", key, cell$mean_TPR, cell$sd_TPR),
                  sprintf("%sAvgL,%.6f,", key, cell$mean_L),
                  sprintf("%sAvgT,%.6f,", key, cell$mean_T))
      }
    }
  }
  csv_path <- file.path(OUT_DIR, paste0(FILE_STEM, ".csv"))
  writeLines(c("solver,snr,tfdr,metric,mean,sd", rows), csv_path)
  cat(sprintf("[Info] CSV saved to: %s\n\n", csv_path))
}


# ==============================================================================
# MC simulation — achieved FDR / TPR vs. target FDR
# ==============================================================================

#' Sweep tFDR x SNR for TLARS, TOMP and TAFS.
#'
#' @param num_MC Trials per cell.
trx_09_target_fdr <- function(num_MC = NUM_MC) {

  cat("\n", strrep("=", 70L), "\n", sep = "")
  cat("MC Simulation — achieved FDR / TPR vs. target FDR  |  TLARS / TOMP / TAFS\n")
  cat(sprintf("  n = %d,  p = %d,  s = %d,  num_MC = %d  [support redrawn per trial]\n",
              N_OBS, P_VARS, S_TRUE, num_MC))
  cat(sprintf("  SNR grid: %d values,  tFDR grid: %d values\n\n",
              length(SNR_VALUES), length(TFDR_VALUES)))

  results <- list()
  for (solver_entry in SOLVERS) {
    cat(sprintf("%s\nSolver: %s\n%s\n", strrep("=", 60L), solver_entry$name, strrep("=", 60L)))

    # Demo 05 configuration with Normal dummies, every option explicit (C++
    # reference). tloop_stagnation_stop is left unset (NULL = AUTO, C++ nullopt).
    ctrl <- TRexSelector::trex_control(
      solver                   = solver_entry$solver,
      lambda2                  = solver_entry$lambda2,
      rho_afs                  = solver_entry$rho_afs,
      ncgmp_variant            = solver_entry$ncgmp_variant,
      K                        = 20L,
      max_dummy_multiplier     = 10L,
      use_max_T_stop           = TRUE,
      opt_threshold            = 0.75,
      lloop_strategy           = "STANDARD",
      dummy_distribution       = trex_dummy_distribution("Normal"),
      tloop_max_stagnant_steps = 5L,
      parallel_rnd_experiments = FALSE,
      use_memory_mapping       = FALSE
    )

    res <- vector("list", length(SNR_VALUES))
    for (si in seq_along(SNR_VALUES)) {
      snr       <- SNR_VALUES[[si]]
      base_seed <- SEED_OFFSET + (si - 1L) * 1000L
      make_data <- local({
        current_snr <- snr
        function(mc, trial_seed) {
          trial_support <- make_support_random(S_TRUE, P_VARS, seed = trial_seed)
          dgp_gauss_snr(N_OBS, P_VARS, trial_support, coefs = rep(1.0, S_TRUE),
                        snr = current_snr, seed = trial_seed)
        }
      })
      res[[si]] <- lapply(TFDR_VALUES, function(tfdr) {
        lbl <- sprintf("SNR=%.2f tFDR=%.3f [%s]", snr, tfdr, solver_entry$name)
        .run_mc_trex(make_data, ctrl, tfdr, base_seed, num_MC, num_cores,
                     track_L_T = TRUE, label = lbl, seeded_selector = TRUE)
      })
    }
    results[[solver_entry$name]] <- res
  }

  .save_results_09(results, num_MC)
}


# ==============================================================================
# Main
# ==============================================================================

trx_09_target_fdr(NUM_MC)
cat("\ndemo_trex_09_mc_sim_target_fdr.R complete.\n")

# All results are written: the cell cache of a stopped run is no longer needed.
cell_cache_done()
