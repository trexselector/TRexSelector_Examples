# ==============================================================================
# demo_trex_da_07_mc_sim_nn.R
# ==============================================================================
#
# DA-TRex+NN Monte Carlo simulations for the nearest-neighbours (NN) DA method:
# banded MA(kappa) data, plus an AR(1) method-mismatch study.
#
# Parts 1-2 use the NN DGP, where all p predictors share a banded MA(kappa)
# covariance structure (the correctly-specified case for the NN method). Part 3
# applies the same NN correction to AR(1) data — a deliberate misspecification
# stress test (formerly its own demo, 03b, in the legacy R numbering).
#
#   Part 1: MC SNR sweep on NN data (n=300, p=1000, kappa=3, rho=0.7;
#           CappedSpread and Random support; 3 solvers + base T-Rex comparison).
#   Part 2: 2D kappa x rho sweep on NN data (SNR=2.0; all 3 solvers;
#           CappedSpread and Random sub-sections).
#   Part 3: 2D SNR x rho sweep on AR(1) data with DA+NN (all 3 solvers;
#           CappedSpread and Random sub-sections).
#
# Mirrors:
#  cpp/trex_selector_methods/ad/trex_da/demo_trex_da_07_mc_sim_nn
#  Python/trex_selector_methods/ad/trex_da/demo_trex_da_07_mc_sim_nn
#
# (same DGPs, control parameters, sweep axes, and result files).
# Results are saved to simulation_results/data/ as TXT and CSV files.
# Optional first CLI argument sets the number of parallel worker cores
# (default TREX_NUM_WORKERS, else 8). Indices are 1-based (R convention).
# ==============================================================================

library(TRexSelector)
library(parallel)

# Worker processes: first CLI argument, else TREX_NUM_WORKERS, else 8 (0 = all).
num_cores <- local({
  a <- suppressWarnings(as.integer(commandArgs(trailingOnly = TRUE)[1L]))
  if (is.na(a)) a <- suppressWarnings(as.integer(Sys.getenv("TREX_NUM_WORKERS", "8")))
  if (is.na(a)) 8L else if (a < 1L) parallel::detectCores() else a
})

this_dir_ <- tryCatch(
  dirname(normalizePath(sys.frame(1)$ofile)),
  error = function(e) {
    args <- commandArgs(trailingOnly = FALSE)
    flag <- grep("^--file=", args, value = TRUE)
    if (length(flag) > 0L)
      dirname(normalizePath(sub("^--file=", "", flag[1L]))) else "."
  }
)

source(file.path(this_dir_, "..", "..", "..", "mc_runtime.R"))  # PSOCK workers, cell cache
source(file.path(this_dir_, "..", "..", "..", "support_generators.R"))
source(file.path(this_dir_, "..", "trex_da_dgps.R"))
source(file.path(this_dir_, "..", "da_sim_common.R"))

RUN_PART_1 <- TRUE
RUN_PART_2 <- TRUE
RUN_PART_3 <- TRUE

OUT_DIR <- file.path(this_dir_, "simulation_results", "data")

# Shared MC parameters (mirror cpp)
N <- 300L
P <- 1000L
S <- 10L
KAPPA <- 3L
RHO <- 0.7
SNR <- 2.0
AMPLITUDE <- 3.0
TFDR <- 0.2
K <- 20L
NUM_MC <- 200L
BASE_SEED <- 2026L
MAX_GAP <- 20L

DA_SPEC <- list(method = "NN")


# ==============================================================================
# Part 1: MC SNR sweep — NN DGP
# CappedSpread(max_gap=20) and Random support; base T-Rex comparison rows.
# ==============================================================================

part_1_snr_sweep <- function() {
  snr_grid <- c(0.1, 0.2, 0.5, 1.0, 2.0, 5.0)
  support_fixed <- make_support_capped_spread(S, P, MAX_GAP)
  hdr_base <- sprintf("NN kappa=%d, rho=%s, n=%d, p=%d",
                      KAPPA, fmt_num(RHO), N, P)

  # A — CappedSpread(max_gap=20)
  run_snr_sweep(
    OUT_DIR, "da_nn_snr_capped", snr_grid, NUM_MC, TFDR, BASE_SEED,
    default_solvers("+NN"),
    function(snr) {
      list(n = N, p = P, dgp = "dgp_nn_snr",
           kwargs = list(amplitude = AMPLITUDE, kappa = KAPPA,
                         rho = RHO, snr = snr),
           support = support_fixed, s = S)
    },
    DA_SPEC, K,
    paste0(hdr_base, sprintf(", support=CappedSpread(max_gap=%d)", MAX_GAP)),
    include_base_trex = TRUE
  )

  # B — Random (redrawn each trial)
  run_snr_sweep(
    OUT_DIR, "da_nn_snr_random", snr_grid, NUM_MC, TFDR, BASE_SEED,
    default_solvers("+NN"),
    function(snr) {
      list(n = N, p = P, dgp = "dgp_nn_snr",
           kwargs = list(amplitude = AMPLITUDE, kappa = KAPPA,
                         rho = RHO, snr = snr),
           support = NULL, s = S)
    },
    DA_SPEC, K,
    paste0(hdr_base, ", support=Random (per-trial)"),
    include_base_trex = TRUE
  )
}


# ==============================================================================
# Generic 2D two-support sweep driver (Parts 2 and 3)
#
# Each solver runs with its own exch_tie_alpha, as in the cpp loops. The seed
# offset stays constant across cells (mirrors cpp base_offset = base_seed).
# ==============================================================================

.run_2d_two_support <- function(scenario_tag, row_label, col_label, row_grid,
                                col_grid, make_cell, header_extra, label_fn) {
  solvers <- default_solvers("+NN")
  names_ <- vapply(solvers, `[[`, character(1L), "name")
  n_row <- length(row_grid)
  n_col <- length(col_grid)

  zero_mat <- stats::setNames(
    rep(list(matrix(0.0, n_row, n_col)), length(names_)), names_
  )
  tpp_cs <- zero_mat
  fdp_cs <- zero_mat
  sd_tpp_cs <- zero_mat
  sd_fdp_cs <- zero_mat
  tpp_rand <- zero_mat
  fdp_rand <- zero_mat
  sd_tpp_rand <- zero_mat
  sd_fdp_rand <- zero_mat

  support_cs <- make_support_capped_spread(S, P, MAX_GAP)
  total_cells <- n_row * n_col

  for (sv in solvers) {
    nm <- sv$name
    cat(sprintf("\n  Solver: %s\n\n", nm))

    cat(sprintf("  [A] CappedSpread(max_gap=%d) ...\n\n", MAX_GAP))
    cell_idx <- 0L
    for (ir in seq_len(n_row)) {
      for (ic in seq_len(n_col)) {
        cell_idx <- cell_idx + 1L
        cell <- make_cell(row_grid[ir], col_grid[ic])
        lbl <- sprintf("[%d/%d] %s  %s", cell_idx, total_cells, nm,
                       label_fn(row_grid[ir], col_grid[ic]))
        res <- run_mc_trials(NUM_MC, lbl, N, P, cell$dgp, cell$kwargs,
                             support_cs, S, TFDR, sv, K, DA_SPEC,
                             BASE_SEED)
        tpp_cs[[nm]][ir, ic] <- res$avg_tpr
        fdp_cs[[nm]][ir, ic] <- res$avg_fdr
        sd_tpp_cs[[nm]][ir, ic] <- res$sd_tpr
        sd_fdp_cs[[nm]][ir, ic] <- res$sd_fdr
      }
    }

    cat("\n  [B] Random support ...\n\n")
    cell_idx <- 0L
    for (ir in seq_len(n_row)) {
      for (ic in seq_len(n_col)) {
        cell_idx <- cell_idx + 1L
        cell <- make_cell(row_grid[ir], col_grid[ic])
        lbl <- sprintf("[%d/%d] %s  %s  Random", cell_idx, total_cells, nm,
                       label_fn(row_grid[ir], col_grid[ic]))
        res <- run_mc_trials(NUM_MC, lbl, N, P, cell$dgp, cell$kwargs,
                             NULL, S, TFDR, sv, K, DA_SPEC, BASE_SEED)
        tpp_rand[[nm]][ir, ic] <- res$avg_tpr
        fdp_rand[[nm]][ir, ic] <- res$avg_fdr
        sd_tpp_rand[[nm]][ir, ic] <- res$sd_tpr
        sd_fdp_rand[[nm]][ir, ic] <- res$sd_fdr
      }
    }
  }

  save_and_print_2d_two_support_results(
    OUT_DIR, scenario_tag, row_label, col_label, row_grid, col_grid,
    NUM_MC, names_, tpp_cs, fdp_cs, sd_tpp_cs, sd_fdp_cs,
    tpp_rand, fdp_rand, sd_tpp_rand, sd_fdp_rand, header_extra
  )
}


# ==============================================================================
# Part 2: 2D kappa x rho sweep — NN DGP
# Fixed: SNR=2.0, amplitude=3.0.
# ==============================================================================

part_2_kappa_rho_sweep <- function() {
  kappa_grid <- c(1, 2, 3, 5, 7, 10, 15)
  rho_grid <- c(0.1, 0.3, 0.5, 0.7, 0.8, 0.9)

  print_section_header("NN Part 2: 2D Kappa x Rho sweep")
  cat(sprintf("  n=%d  p=%d  s=%d  SNR=%s  MC=%d  tFDR=%s  amplitude=%s\n",
              N, P, S, fmt_num(SNR), NUM_MC, fmt_num(TFDR),
              fmt_num(AMPLITUDE)))
  cat("  kappa_grid: {", paste(as.integer(kappa_grid), collapse = ", "),
      "}\n", sep = "")
  cat("  rho_grid:   {", paste(sprintf("%.1f", rho_grid), collapse = ", "),
      "}\n\n", sep = "")

  .run_2d_two_support(
    "da_nn_kappa_rho", "Kappa", "Rho", kappa_grid, rho_grid,
    function(kappa, rho) {
      list(dgp = "dgp_nn_snr",
           kwargs = list(amplitude = AMPLITUDE,
                         kappa = as.integer(kappa),
                         rho = rho, snr = SNR))
    },
    sprintf("n=%d, p=%d, s=%d, SNR=%s, tFDR=%s, amplitude=3.0",
            N, P, S, fmt_num(SNR), fmt_num(TFDR)),
    function(kappa, rho) {
      sprintf("kappa=%d  rho=%.1f",
              as.integer(kappa), rho)
    }
  )
}


# ==============================================================================
# Part 3: 2D SNR x rho sweep — AR(1) data with DA+NN (method-mismatch study)
# Uses the AR(1) DGP but keeps the NN method, exploring whether the
# nearest-neighbour aggregation can track the AR(1) correlation structure.
# Fixed: amplitude=3.0.
# ==============================================================================

part_3_nn_ar_snr_rho_sweep <- function() {
  snr_grid <- c(0.1, 0.2, 0.5, 1.0, 2.0, 5.0)
  rho_grid <- seq(0.1, 0.9, by = 0.1)

  print_section_header("AR(1)+NN Part 3: 2D SNR x Rho sweep")
  cat(sprintf("  n=%d  p=%d  s=%d  MC=%d  tFDR=%s  amplitude=%s\n",
              N, P, S, NUM_MC, fmt_num(TFDR), fmt_num(AMPLITUDE)))
  cat("  snr_grid: {", paste(sprintf("%.2f", snr_grid), collapse = ", "),
      "}\n", sep = "")
  cat("  rho_grid: {", paste(sprintf("%.1f", rho_grid), collapse = ", "),
      "}\n\n", sep = "")

  .run_2d_two_support(
    "da_nn_ar_snr_rho", "SNR", "Rho", snr_grid, rho_grid,
    function(snr, rho) {
      list(dgp = "dgp_ar1_snr",
           kwargs = list(amplitude = AMPLITUDE, rho = rho,
                         snr = snr))
    },
    sprintf("n=%d, p=%d, s=%d, tFDR=%s, amplitude=3.0",
            N, P, S, fmt_num(TFDR)),
    function(snr, rho) sprintf("SNR=%.2f  rho=%.1f", snr, rho)
  )
}


# ==============================================================================
# Main
# ==============================================================================

cat(sprintf("Running with %d worker cores\n", num_cores))

if (RUN_PART_1) part_1_snr_sweep()
if (RUN_PART_2) part_2_kappa_rho_sweep()
if (RUN_PART_3) part_3_nn_ar_snr_rho_sweep()
cat("\nNN MC simulation complete.\n")

# All results are written: the cell cache of a stopped run is no longer needed.
cell_cache_done()
