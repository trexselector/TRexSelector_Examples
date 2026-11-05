# ==============================================================================
# demo_trex_da_01_mc_sim_ar1.R
# ==============================================================================
#
# DA-TRex+AR1 Monte Carlo simulations for the AR(1) Toeplitz DGP.
#
#   Part 1: MC SNR sweep (n=300, p=1000, s=10, rho=0.7, 200 MC; CappedSpread
#           and Random support; 3 DA solvers + base T-Rex comparison rows).
#   Part 2: MC rho sweep (SNR=2.0; same layout as Part 1).
#   Part 3: 2D Gap x Rho sweep — probes the kappa boundary where gap < kappa
#           collapses TPP (CappedSpread(gap) matrices + Random column).
#
# Mirrors:
#  cpp/trex_selector_methods/ad/trex_da/demo_trex_da_01_mc_sim_ar1
#  Python/trex_selector_methods/ad/trex_da/demo_trex_da_01_mc_sim_ar1
#
# Results are saved to simulation_results/data/ as TXT and CSV files.
# Every part runs num_MC=200 trials per grid point. Optional first CLI
# argument sets the number of parallel worker cores (default TREX_NUM_WORKERS, else 8).
# Indices are 1-based (R convention); the Python counterpart is 0-based.
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

# Simulation config (mirrors cpp SimConfig + per-part constants)
N <- 300L
P <- 1000L
S <- 10L
AMPLITUDE <- 3.0
TFDR <- 0.2
K <- 20L
NUM_MC <- 200L
BASE_SEED <- 2026L
MAX_GAP <- 20L

DA_SPEC <- list(method = "AR1")


# ==============================================================================
# Part 1: MC simulation — SNR sweep, 2 support types (CappedSpread + Random)
# ==============================================================================

part_1_snr_sweep <- function() {
  rho <- 0.7
  snr_grid <- c(0.1, 0.2, 0.5, 1.0, 2.0, 5.0)
  hdr_base <- sprintf("AR(1) rho=%s, n=%d, p=%d", fmt_num(rho), N, P)
  support_capped <- make_support_capped_spread(S, P, MAX_GAP)

  # --- Run A: CappedSpread(max_gap=20) ---
  run_snr_sweep(
    OUT_DIR, "da_ar1_snr_capped", snr_grid, NUM_MC, TFDR, BASE_SEED,
    default_solvers("+AR1"),
    function(snr) {
      list(n = N, p = P, dgp = "dgp_ar1_snr",
           kwargs = list(amplitude = AMPLITUDE, rho = rho,
                         snr = snr),
           support = support_capped, s = S)
    },
    DA_SPEC, K,
    paste0(hdr_base, sprintf(", support=CappedSpread(max_gap=%d)", MAX_GAP)),
    include_base_trex = TRUE
  )

  # --- Run B: Random support (redrawn per trial) ---
  run_snr_sweep(
    OUT_DIR, "da_ar1_snr_random", snr_grid, NUM_MC, TFDR, BASE_SEED,
    default_solvers("+AR1"),
    function(snr) {
      list(n = N, p = P, dgp = "dgp_ar1_snr",
           kwargs = list(amplitude = AMPLITUDE, rho = rho,
                         snr = snr),
           support = NULL, s = S)
    },
    DA_SPEC, K,
    paste0(hdr_base, ", support=Random (redrawn per trial)"),
    include_base_trex = TRUE
  )
}


# ==============================================================================
# Part 2: MC simulation — Rho sweep, 2 support types (CappedSpread + Random)
# ==============================================================================

part_2_rho_sweep <- function() {
  snr_fixed <- 2.0
  rho_grid <- seq(0.0, 0.9, by = 0.1)
  hdr_base <- sprintf("AR(1) SNR=%s, n=%d, p=%d", fmt_num(snr_fixed), N, P)
  support_capped <- make_support_capped_spread(S, P, MAX_GAP)

  # --- Run A: CappedSpread(max_gap=20) ---
  run_param_sweep(
    OUT_DIR, "da_ar1_rho_capped", "Rho", rho_grid, NUM_MC, TFDR, BASE_SEED,
    default_solvers("+AR1"),
    function(rho) {
      list(n = N, p = P, dgp = "dgp_ar1_snr",
           kwargs = list(amplitude = AMPLITUDE, rho = rho,
                         snr = snr_fixed),
           support = support_capped, s = S)
    },
    DA_SPEC, K,
    paste0(hdr_base, sprintf(", support=CappedSpread(max_gap=%d)", MAX_GAP)),
    include_base_trex = TRUE
  )

  # --- Run B: Random support (redrawn per trial) ---
  run_param_sweep(
    OUT_DIR, "da_ar1_rho_random", "Rho", rho_grid, NUM_MC, TFDR, BASE_SEED,
    default_solvers("+AR1"),
    function(rho) {
      list(n = N, p = P, dgp = "dgp_ar1_snr",
           kwargs = list(amplitude = AMPLITUDE, rho = rho,
                         snr = snr_fixed),
           support = NULL, s = S)
    },
    DA_SPEC, K,
    paste0(hdr_base, ", support=Random (redrawn per trial)"),
    include_base_trex = TRUE
  )
}


# ==============================================================================
# Part 3: 2D Gap x Rho sweep
#
#  Runs num_MC trials per (solver, gap, rho) cell for two support types:
#    A) CappedSpread(gap) — deterministic, evenly-spaced support.
#    B) Random            — support redrawn each trial (not gap-dependent).
#
#  The kappa boundary (kappa = ceiling(log(0.02) / log(rho))) is the DA+AR1
#  correction window half-width. When gap < kappa, active predictors fall
#  inside each other's correction windows and TPP collapses.
# ==============================================================================

part_3_gap_rho_sweep <- function() {
  snr_fixed <- 2.0
  gap_grid <- c(100L, 50L, 20L, 15L, 10L, 5L, 1L)
  rho_grid <- seq(0.0, 0.9, by = 0.1)

  kappa_boundary <- ifelse(rho_grid <= 0.0, 0L,
                           as.integer(ceiling(log(0.02) / log(rho_grid))))

  solvers <- default_solvers("+AR1")
  names_ <- vapply(solvers, `[[`, character(1L), "name")
  n_rho <- length(rho_grid)
  n_gap <- length(gap_grid)

  zero_mat <- stats::setNames(
    rep(list(matrix(0.0, n_rho, n_gap)), length(names_)), names_
  )
  zero_vec <- stats::setNames(
    rep(list(numeric(n_rho)), length(names_)), names_
  )
  tpp_cs <- zero_mat
  fdp_cs <- zero_mat
  sd_tpp_cs <- zero_mat
  sd_fdp_cs <- zero_mat
  tpp_rand <- zero_vec
  fdp_rand <- zero_vec
  sd_tpp_rand <- zero_vec
  sd_fdp_rand <- zero_vec

  print_section_header("AR(1) Part 3: Gap x Rho sweep")
  cat(sprintf("  n=%d  p=%d  s=%d  SNR=%s  MC=%d  tFDR=%s  amplitude=%s\n",
              N, P, S, fmt_num(snr_fixed), NUM_MC, fmt_num(TFDR),
              fmt_num(AMPLITUDE)))
  cat("  gap_grid: {", paste(gap_grid, collapse = ", "), "}\n", sep = "")
  cat("  rho_grid: {", paste(sprintf("%.1f", rho_grid), collapse = ", "),
      "}\n\n", sep = "")

  # Constant seed offset across cells (mirrors cpp base_offset = base_seed)
  base_offset <- BASE_SEED

  for (sv in solvers) {
    nm <- sv$name
    cat(sprintf("\n  Solver: %s\n\n", nm))

    # ---- Section A: CappedSpread(gap) x rho ----
    cat("  [A] CappedSpread(gap) sweep ...\n\n")
    cell_idx <- 0L
    total_cells <- n_rho * n_gap
    for (ir in seq_len(n_rho)) {
      rho <- rho_grid[ir]
      for (ig in seq_len(n_gap)) {
        gap <- gap_grid[ig]
        cell_idx <- cell_idx + 1L
        support <- make_support_capped_spread(S, P, gap)
        lbl <- sprintf("[%d/%d] %s  rho=%.1f  gap=%3d  kappa=%d",
                       cell_idx, total_cells, nm, rho, gap,
                       kappa_boundary[ir])
        res <- run_mc_trials(
          NUM_MC, lbl, N, P, "dgp_ar1_snr",
          list(amplitude = AMPLITUDE, rho = rho, snr = snr_fixed),
          support, S, TFDR, sv, K, DA_SPEC, base_offset
        )
        tpp_cs[[nm]][ir, ig] <- res$avg_tpr
        fdp_cs[[nm]][ir, ig] <- res$avg_fdr
        sd_tpp_cs[[nm]][ir, ig] <- res$sd_tpr
        sd_fdp_cs[[nm]][ir, ig] <- res$sd_fdr
      }
    }

    # ---- Section B: Random support sweep (rho only) ----
    cat("\n  [B] Random support sweep ...\n\n")
    for (ir in seq_len(n_rho)) {
      rho <- rho_grid[ir]
      lbl <- sprintf("[%d/%d] %s  rho=%.1f  kappa=%d  Random",
                     ir, n_rho, nm, rho, kappa_boundary[ir])
      res <- run_mc_trials(
        NUM_MC, lbl, N, P, "dgp_ar1_snr",
        list(amplitude = AMPLITUDE, rho = rho, snr = snr_fixed),
        NULL, S, TFDR, sv, K, DA_SPEC, base_offset
      )
      tpp_rand[[nm]][ir] <- res$avg_tpr
      fdp_rand[[nm]][ir] <- res$avg_fdr
      sd_tpp_rand[[nm]][ir] <- res$sd_tpr
      sd_fdp_rand[[nm]][ir] <- res$sd_fdr
    }
  }

  header_extra <- sprintf("n=%d, p=%d, s=%d, SNR=%s, tFDR=%s, amplitude=%s",
                          N, P, S, fmt_num(snr_fixed), fmt_num(TFDR),
                          fmt_num(AMPLITUDE))
  save_and_print_2d_gap_rho_results(
    OUT_DIR, "da_ar1_gap_rho", gap_grid, rho_grid, kappa_boundary, NUM_MC,
    names_, tpp_cs, fdp_cs, sd_tpp_cs, sd_fdp_cs,
    tpp_rand, fdp_rand, sd_tpp_rand, sd_fdp_rand, header_extra
  )
}


# ==============================================================================
# Main
# ==============================================================================

cat(sprintf("Running with %d worker cores\n", num_cores))

if (RUN_PART_1) part_1_snr_sweep()
if (RUN_PART_2) part_2_rho_sweep()
if (RUN_PART_3) part_3_gap_rho_sweep()
cat("\nAR(1) MC simulation complete.\n")

# All results are written: the cell cache of a stopped run is no longer needed.
cell_cache_done()
