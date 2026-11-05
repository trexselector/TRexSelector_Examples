# ==============================================================================
# demo_trex_10_mc_sim_dt_ratio.R
# ==============================================================================
#
# Classical T-Rex selector demo: FDR and TPR as a function of the sample-size
# ratio c = n / (s ln p), for three base solvers (TLARS, TOMP, TAFS).
#
# Mirrors:
# cpp/trex_selector_methods/ad/trex_core/demo_trex_10_mc_sim_dt_ratio.cpp
#
# Sparse recovery needs n to grow like s log p (Donoho-Tanner scaling). The demo
# sets n = round(c * s * ln p) for a grid of ratios c and two p, and records FDR
# and TPR at tFDR = 0.1, so curves of the two p that fall onto each other on the
# c axis confirm that c governs the power of the FDR-controlled selection.
#
# Data as in demo 03 except for n: s = 10 unit coefficients on a support
# redrawn per trial, Normal predictors and noise.
# Selector as in demo 05: K = 20, max_dummy_multiplier = 10, Normal dummies,
# STANDARD L-loop, use_max_T_stop, stagnation stop AUTO (off for TLARS, on for
# TOMP and TAFS), rho_afs = 0.3 for TAFS.
#
# Grid: p in {1,000, 10,000} x c in {1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10} x
# SNR in {0.5, 1, 2}, num_MC = 200 trials per cell.
#
# Seeds: trial mc of a cell uses the data seed 24 + 1000 * (snr_index - 1) +
# 100000 * (c_index - 1) + 10000000 * (p_index - 1) + mc - 1 for every solver,
# and the selector is seeded with the same trial seed (reproducible).
#
# Output (simulation_results/data/): one table per solver and SNR in
# demo_trex_10_mc_sim_dt_ratio.txt and the tidy CSV
# demo_trex_10_mc_sim_dt_ratio.csv with the columns
# solver, snr, p, n, c, metric, mean, sd (metrics FDR, TPR, AvgL, AvgT; sd for
# FDR and TPR only). Every finished cell is cached, so a stopped run resumes.
#
# Command line: Rscript demo_trex_10_mc_sim_dt_ratio.R [workers] [num_MC]
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
FILE_STEM <- "demo_trex_10_mc_sim_dt_ratio"
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

S_TRUE      <- 10L
TFDR        <- 0.1
SEED_OFFSET <- 24L  # as demos 02-05

P_VALUES   <- c(1000L, 10000L)
C_VALUES   <- c(1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0)
SNR_VALUES <- c(0.5, 1.0, 2.0)

# Base solvers (rho_afs only meaningful for TAFS)
SOLVERS <- list(
  list(name = "TLARS", solver = "TLARS", lambda2 = 0.0, rho_afs = 0.0,
       ncgmp_variant = 0L),
  list(name = "TOMP",  solver = "TOMP",  lambda2 = 0.0, rho_afs = 0.0,
       ncgmp_variant = 0L),
  list(name = "TAFS",  solver = "TAFS",  lambda2 = 0.0, rho_afs = 0.3,
       ncgmp_variant = 0L)
)

#' Sample size of a grid point: n = round(c * s * ln p), halves away from zero as C++ std::lround.
#'
#' @param c Ratio n / (s ln p).
#' @param p Number of predictors.
sample_size <- function(c, p) as.integer(floor(c * S_TRUE * log(p) + 0.5))


# ==============================================================================
# Output: per-solver tables (console + .txt) and tidy CSV
# ==============================================================================

#' Print the per-solver tables and write the .txt and .csv files.
#'
#' @param results results[[solver_name]][[si]][[pi]][[ci]] is the
#'                .run_mc_trex() list of one cell.
#' @param num_MC  Trials per cell.
.save_results_10 <- function(results, num_MC) {
  label_w <- 18L
  col_w   <- 8L
  fmt_row <- function(label, cells) {
    paste0(formatC(label, width = -label_w), paste(cells, collapse = ""))
  }
  lines <- c(
    "",
    strrep("=", 78L),
    sprintf("=== T-Rex FDR / TPR vs. c = n / (s ln p) (mean over %d trials) ===", num_MC),
    sprintf("    s = %d, tFDR = %g, K = 20, STANDARD L-loop, Normal dummies", S_TRUE, TFDR),
    strrep("=", 78L)
  )
  for (solver in SOLVERS) {
    res <- results[[solver$name]]
    for (si in seq_along(SNR_VALUES)) {
      lines <- c(lines, "", sprintf("%s  |  SNR = %g", solver$name, SNR_VALUES[[si]]),
                 fmt_row("  c", formatC(C_VALUES, width = col_w, format = "f", digits = 1L)))
      for (pi in seq_along(P_VALUES)) {
        p <- P_VALUES[[pi]]
        lines <- c(lines, fmt_row(sprintf("  n  p=%d", p),
                                  formatC(sample_size(C_VALUES, p), width = col_w)))
        for (metric in c("FDR", "TPR")) {
          vals <- vapply(res[[si]][[pi]], function(cell) cell[[paste0("mean_", metric)]], 0.0)
          lines <- c(lines, fmt_row(sprintf("  %s p=%d", metric, p),
                                    formatC(vals, width = col_w, format = "f", digits = 3L)))
        }
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
      for (pi in seq_along(P_VALUES)) {
        for (ci in seq_along(C_VALUES)) {
          cell <- res[[si]][[pi]][[ci]]
          p <- P_VALUES[[pi]]
          key <- sprintf("%s,%.6f,%d,%d,%.6f,", solver$name, SNR_VALUES[[si]], p,
                         sample_size(C_VALUES[[ci]], p), C_VALUES[[ci]])
          rows <- c(rows,
                    sprintf("%sFDR,%.6f,%.6f", key, cell$mean_FDR, cell$sd_FDR),
                    sprintf("%sTPR,%.6f,%.6f", key, cell$mean_TPR, cell$sd_TPR),
                    sprintf("%sAvgL,%.6f,", key, cell$mean_L),
                    sprintf("%sAvgT,%.6f,", key, cell$mean_T))
        }
      }
    }
  }
  csv_path <- file.path(OUT_DIR, paste0(FILE_STEM, ".csv"))
  writeLines(c("solver,snr,p,n,c,metric,mean,sd", rows), csv_path)
  cat(sprintf("[Info] CSV saved to: %s\n\n", csv_path))
}


# ==============================================================================
# MC simulation — FDR / TPR vs. c = n / (s ln p)
# ==============================================================================

#' Sweep p x c x SNR for TLARS, TOMP and TAFS at tFDR = 0.1.
#'
#' @param num_MC Trials per cell.
trx_10_dt_ratio <- function(num_MC = NUM_MC) {

  cat("\n", strrep("=", 70L), "\n", sep = "")
  cat("MC Simulation — FDR / TPR vs. n / (s ln p)  |  TLARS / TOMP / TAFS\n")
  cat(sprintf("  s = %d,  tFDR = %g,  num_MC = %d  [support redrawn per trial]\n",
              S_TRUE, TFDR, num_MC))
  for (p in P_VALUES)
    cat(sprintf("  p = %d:  n = %s\n", p, paste(sample_size(C_VALUES, p), collapse = " ")))
  cat("\n")

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
      res[[si]] <- lapply(seq_along(P_VALUES), function(pi) {
        lapply(seq_along(C_VALUES), function(ci) {
          snr <- SNR_VALUES[[si]]
          p   <- P_VALUES[[pi]]
          n   <- sample_size(C_VALUES[[ci]], p)
          base_seed <- SEED_OFFSET + (si - 1L) * 1000L + (ci - 1L) * 100000L + (pi - 1L) * 10000000L
          make_data <- function(mc, trial_seed) {
            trial_support <- make_support_random(S_TRUE, p, seed = trial_seed)
            dgp_gauss_snr(n, p, trial_support, coefs = rep(1.0, S_TRUE), snr = snr,
                          seed = trial_seed)
          }
          lbl <- sprintf("SNR=%.2f p=%d c=%.1f n=%d [%s]", snr, p, C_VALUES[[ci]], n,
                         solver_entry$name)
          .run_mc_trex(make_data, ctrl, TFDR, base_seed, num_MC, num_cores,
                       track_L_T = TRUE, label = lbl, seeded_selector = TRUE)
        })
      })
    }
    results[[solver_entry$name]] <- res
  }

  .save_results_10(results, num_MC)
}


# ==============================================================================
# Main
# ==============================================================================

trx_10_dt_ratio(NUM_MC)
cat("\ndemo_trex_10_mc_sim_dt_ratio.R complete.\n")

# All results are written: the cell cache of a stopped run is no longer needed.
cell_cache_done()
