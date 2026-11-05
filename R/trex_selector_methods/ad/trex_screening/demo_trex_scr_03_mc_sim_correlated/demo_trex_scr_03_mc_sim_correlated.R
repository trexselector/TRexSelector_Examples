# ==============================================================================
# demo_trex_scr_03_mc_sim_correlated.R
# ==============================================================================
#
# Screen-TRex Selector, demo 03: correlated designs & dependency-aware screening.
#
# Ordinary screening thresholds each variable's voting proportion Phi_j > 0.5
# independently. Under correlated predictors, redundant correlated variables
# *split* the voting evidence, so Phi drops below 0.5 for all of them and power
# collapses. The dependency-aware (DA) variants pre-group strongly correlated
# variables before voting so the evidence is not split:
#   * TREX_DA_AR1        — for AR(1) (banded) correlation,
#   * TREX_DA_EQUI       — for equicorrelation (single shared factor),
#   * TREX_DA_BLOCK_EQUI — for block-equicorrelation (per-block factors).
#
# NOTE: R's trex_screen_control() exposes only use_bootstrap_CI, R_boot,
# ci_grid_step, and trex_method. The C++/Python make_screen_control()
# additionally takes rho_thr_DA and n_blocks. R uses the library defaults
# (rho_thr_DA = 0.02, n_blocks = 5), the values the C++/Python demos set, so
# Part 5 screens with the DGP's 5 blocks in all three ports.
#
# Mirrors:
#   Python/trex_selector_methods/ad/trex_screening/demo_trex_scr_03_mc_sim_correlated/
#       demo_trex_scr_03_mc_sim_correlated.py
#   cpp/trex_selector_methods/ad/trex_screening/demo_trex_scr_03_mc_sim_correlated/
#       demo_trex_scr_03_mc_sim_correlated.cpp
#
# Demo content (five Monte Carlo studies, matching the cpp Parts 1-5):
#   Part 1: SNR sweep on AR(1) data (rho=0.5): Ordinary/Bootstrap vs. +DA-AR1.
#   Part 2: SNR sweep on equicorrelated data (rho=0.4): vs. +DA-EQUI.
#   Part 3: rho sweep on AR(1) data (SNR=1): vs. +DA-AR1.
#   Part 4: rho sweep on equicorrelated data (SNR=1): vs. +DA-EQUI.
#   Part 5: rho sweep on block-equicorrelated data (SNR=1, 5 blocks): vs. +DA-BLOCK-EQUI.
#
# The Monte Carlo count matches the cpp/Python reference (num_MC = 200).
# Support indices are 1-based (R convention).
#
# Usage: Rscript demo_trex_scr_03_mc_sim_correlated.R [num_worker_cores]
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
P1 <- 10L
SNR_GRID <- c(0.01, 0.1, 0.2, 0.5, 0.6, 1.0, 2.0, 5.0)
RHO_GRID <- c(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


#' Four cpp-style method descriptors: Ordinary/Bootstrap, plain and +DA
#' (mirrors _methods_for() in the Python reference).
methods_for <- function(da_method, da_label) {
  list(
    list(name = "Screen-TRex Ordinary: TLARS",
         trex_method = "TREX", bootstrap = FALSE),
    list(name = "Screen-TRex Bootstrap-CI: TLARS",
         trex_method = "TREX", bootstrap = TRUE),
    list(name = sprintf("Screen-TRex Ordinary+%s: TLARS", da_label),
         trex_method = da_method, bootstrap = FALSE),
    list(name = sprintf("Screen-TRex Bootstrap-CI+%s: TLARS", da_label),
         trex_method = da_method, bootstrap = TRUE)
  )
}

#' Generic MC sweep over one variable (SNR or rho), for a list of methods.
run_sweep <- function(part_title, methods, dgp_fn, sweep_name, sweep_values,
                      file_stem) {
  cat("\n", strrep("=", 70L), "\n", sep = "")
  cat(part_title, "\n")
  cat(strrep("=", 70L), "\n\n", sep = "")
  method_names <- vapply(methods, function(m) m$name, character(1))
  init <- function() {
    setNames(
      lapply(method_names, function(x) numeric(length(sweep_values))), method_names
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
    for (vi in seq_along(sweep_values)) {
      v <- sweep_values[vi]
      make_data <- function(seed) dgp_fn(v, seed)   # v = SNR or rho
      base_seed <- 42L + (vi - 1L) * 1000L  # mirrors cpp base_seed = 42 + idx * 1000
      res <- run_mc_screen(
        NUM_MC, sprintf("%s  %s=%.2f", m$name, sweep_name, v), make_data, trex_ctrl,
        screen_args = list(trex_method = m$trex_method, use_bootstrap_CI = m$bootstrap),
        base_seed = base_seed, num_cores = num_cores
      )
      fdr_map[[m$name]][vi]     <- res$avg_fdr
      tpr_map[[m$name]][vi]     <- res$avg_tpr
      est_fdr_map[[m$name]][vi] <- res$avg_est_fdr
    }
  }
  save_and_print_scr_mc(NUM_MC, OUT_DIR, file_stem, sweep_values,
                        method_names, fdr_map, tpr_map, est_fdr_map,
                        sweep_label = sweep_name)
}


# ==============================================================================
# Main
# ==============================================================================

# Part 1: SNR sweep — AR(1), rho=0.5, beta=1.0
run_sweep("Part 1: DA-AR1 Screen-TRex Monte Carlo on AR(1) Data",
          methods_for("TREX_DA_AR1", "DA-AR1"),
          function(snr, seed) dgp_ar1(N, P, P1, 0.5, snr, 1.0, seed),
          "SNR", SNR_GRID, "scr_ar1_snr_n300_p1000_rho0.50")

# Part 2: SNR sweep — equicorrelated, rho=0.4, beta=3.0
run_sweep("Part 2: DA-EQUI Screen-TRex Monte Carlo on Equicorrelated Data",
          methods_for("TREX_DA_EQUI", "DA-EQUI"),
          function(snr, seed) dgp_equi(N, P, P1, 0.4, snr, 3.0, seed),
          "SNR", SNR_GRID, "scr_equi_snr_n300_p1000_rho0.40")

# Part 3: rho sweep — AR(1), SNR=1.0, beta=1.0
run_sweep("Part 3: DA-AR1 Screen-TRex rho Sweep on AR(1) Data",
          methods_for("TREX_DA_AR1", "DA-AR1"),
          function(rho, seed) dgp_ar1(N, P, P1, rho, 1.0, 1.0, seed),
          "rho", RHO_GRID, "scr_ar1_rho_n300_p1000_snr1.00")

# Part 4: rho sweep — equicorrelated, SNR=1.0, beta=3.0
run_sweep("Part 4: DA-EQUI Screen-TRex rho Sweep on Equicorrelated Data",
          methods_for("TREX_DA_EQUI", "DA-EQUI"),
          function(rho, seed) dgp_equi(N, P, P1, rho, 1.0, 3.0, seed),
          "rho", RHO_GRID, "scr_equi_rho_n300_p1000_snr1.00")

# Part 5: rho sweep — block-equicorrelated, SNR=1.0, beta=3.0, 5 blocks
run_sweep("Part 5: DA-BLOCK-EQUI Screen-TRex rho Sweep on Block-Equicorr Data",
          methods_for("TREX_DA_BLOCK_EQUI", "DA-BLOCK-EQUI"),
          function(rho, seed) dgp_block_equi(N, P, P1, rho, 5L, 1.0, 3.0, seed),
          "rho", RHO_GRID, "scr_block_equi_rho_n300_p1000_blocks5_snr1.00")

cat("\ndemo_trex_scr_03_mc_sim_correlated.R complete.\n")

# All results are written: the cell cache of a stopped run is no longer needed.
cell_cache_done()
