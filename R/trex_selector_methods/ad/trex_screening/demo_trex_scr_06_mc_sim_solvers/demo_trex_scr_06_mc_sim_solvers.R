# ==============================================================================
# demo_trex_scr_06_mc_sim_solvers.R
# ==============================================================================
#
# Screen-TRex Selector, demo 06: solver-backend comparison (TLARS / TAFS / TOMP).
#
# Screening thresholds the dummy-based voting proportion Phi_j, but Phi itself is
# produced by an underlying T-Rex solver. This demo asks whether the choice of
# solver backend changes screening performance:
#   * TLARS    — terminating LARS (the default),
#   * TAFS-0.3 — terminating adaptive forward stepwise (shrinkage rho_afs = 0.3),
#   * TOMP     — terminating orthogonal matching pursuit,
# each combined with Ordinary and Bootstrap-CI screening. The same i.i.d.
# Gaussian design as demo 01 is used, so the numbers are directly comparable.
#
# Mirrors:
#   Python/trex_selector_methods/ad/trex_screening/demo_trex_scr_06_mc_sim_solvers/
#       demo_trex_scr_06_mc_sim_solvers.py
#   cpp/trex_selector_methods/ad/trex_screening/demo_trex_scr_06_mc_sim_solvers/
#       demo_trex_scr_06_mc_sim_solvers.cpp
#
# Demo content (two Monte Carlo SNR sweeps, matching the cpp Demos 01-02):
#   Part 1: three solvers, Ordinary screening only (3 series).
#   Part 2: three solvers x {Ordinary, Bootstrap-CI} (6 series).
#
# The Monte Carlo count matches the cpp/Python reference (num_MC = 200).
# Support indices are 1-based (R convention).
#
# Usage: Rscript demo_trex_scr_06_mc_sim_solvers.R [num_worker_cores]
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


#' A solver x screening-variant descriptor (mirrors cpp SolverVariant).
sv <- function(name, solver, rho_afs = 0.3, bootstrap = FALSE) {
  list(name = name, solver = solver, rho_afs = rho_afs, bootstrap = bootstrap)
}

#' Three solvers, Ordinary screening only (mirrors ordinary_solver_variants()).
ordinary_solver_variants <- function() {
  list(
    sv("Screen-TRex Ordinary: TLARS",    "TLARS"),
    sv("Screen-TRex Ordinary: TAFS-0.3", "TAFS", rho_afs = 0.3),
    sv("Screen-TRex Ordinary: TOMP",     "TOMP")
  )
}

#' Six combined variants — three solvers x {Ordinary, Bootstrap-CI}.
all_solver_variants <- function() {
  variants <- list()
  for (bs in c(FALSE, TRUE)) {
    prefix <- if (bs) "Screen-TRex Bootstrap-CI: " else "Screen-TRex Ordinary: "
    variants <- c(variants, list(
      sv(paste0(prefix, "TLARS"),    "TLARS",                 bootstrap = bs),
      sv(paste0(prefix, "TAFS-0.3"), "TAFS", rho_afs = 0.3,   bootstrap = bs),
      sv(paste0(prefix, "TOMP"),     "TOMP",                  bootstrap = bs)
    ))
  }
  variants
}

#' MC sweep over a list of solver x method variants (SNR grid).
run_solver_sweep <- function(title, variants, file_stem) {
  cat("\n", strrep("=", 70L), "\n", sep = "")
  cat(title, "\n")
  cat(strrep("=", 70L), "\n", sep = "")
  cat("High-dimensional (p > n)\n")
  cat(sprintf("n = %d, p = %d, s = %d, num_MC = %d\n\n",
              N, P, SUPPORT_SIZE, NUM_MC))

  method_names <- vapply(variants, function(v) v$name, character(1))
  init <- function() {
    setNames(
      lapply(method_names, function(x) numeric(length(SNR_GRID))), method_names
    )
  }
  fdr_map <- init()
  tpr_map <- init()
  est_fdr_map <- init()

  for (v in variants) {
    cat(strrep("=", 70L), "\n", sep = "")
    cat(sprintf("Variant: %s\n", v$name))
    cat(strrep("=", 70L), "\n", sep = "")
    # make_trex_ctrl_for(): stagnation stop enabled for every backend.
    trex_ctrl <- make_scr_trex_ctrl(solver = v$solver, rho_afs = v$rho_afs,
                                    tloop_stagnation_stop = TRUE,
                                    tloop_max_stagnant_steps = 5L)
    for (si in seq_along(SNR_GRID)) {
      snr <- SNR_GRID[si]
      make_data <- function(seed) {
        set.seed(seed + 500000L)                 # isolated support RNG
        sup <- sample.int(P, SUPPORT_SIZE)
        dgp_iid_snr(N, P, sup, rep(1, SUPPORT_SIZE), snr, seed = seed)
      }
      base_seed <- 24L + (si - 1L) * 1000L  # mirrors cpp base_seed = 24 + snr_idx * 1000
      res <- run_mc_screen(
        NUM_MC, sprintf("%s  SNR=%.2f", v$name, snr), make_data, trex_ctrl,
        screen_args = list(trex_method = "TREX", use_bootstrap_CI = v$bootstrap),
        base_seed = base_seed, num_cores = num_cores
      )
      fdr_map[[v$name]][si]     <- res$avg_fdr
      tpr_map[[v$name]][si]     <- res$avg_tpr
      est_fdr_map[[v$name]][si] <- res$avg_est_fdr
    }
  }
  save_and_print_scr_mc(NUM_MC, OUT_DIR, file_stem, SNR_GRID,
                        method_names, fdr_map, tpr_map, est_fdr_map)
}


# ==============================================================================
# Main
# ==============================================================================

# Part 1: three solvers, ordinary Screen-TRex
run_solver_sweep(
  "Demo: Screen-TRex MC - Solver Comparison (TLARS / TAFS-0.3 / TOMP)",
  ordinary_solver_variants(),
  sprintf("scr_solvers_snr_n%d_p%d_s%d", N, P, SUPPORT_SIZE)
)

# Part 2: three solvers x {Ordinary, Bootstrap-CI} (6 series)
run_solver_sweep(
  "Demo: Screen-TRex MC - Solver x Method Comparison (6 series)",
  all_solver_variants(),
  sprintf("scr_solver_method_snr_n%d_p%d_s%d", N, P, SUPPORT_SIZE)
)

cat("\ndemo_trex_scr_06_mc_sim_solvers.R complete.\n")

# All results are written: the cell cache of a stopped run is no longer needed.
cell_cache_done()
