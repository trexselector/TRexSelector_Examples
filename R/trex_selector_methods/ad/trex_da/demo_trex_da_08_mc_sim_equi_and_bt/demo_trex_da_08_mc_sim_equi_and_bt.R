# ==============================================================================
# demo_trex_da_08_mc_sim_equi_and_bt.R
# ==============================================================================
#
# DA-TRex+EQUI Monte Carlo 2D sweep (SNR x rho) for the equicorrelated DGP.
#
# One 2D sweep: for each SNR column in {0.5, 1, 2, 5}, run a full rho sweep
# over {0, 0.025, 0.05, 0.1, 0.2, 0.4, 0.7, 0.9} (dense near 0 to locate the
# DA suppression cliff) with the Equi DGP (n=300, p=1000, Random support),
# 3 DA solvers + the 3 base T-Rex comparison rows.
#
# Rationale: equicorrelation ties every predictor to one shared factor, so
# DA-EQUI suppresses all candidates as soon as rho is noticeably above 0,
# independent of signal strength, while base T-Rex loses FDR control at the
# same tiny rho. The 2D layout shows the cliff location, its SNR independence,
# and the rho=0 working anchor in a single figure.
#
# Earlier revisions of this demo also carried BT (hierarchical-block) SNR and
# linkage sweeps; those were dropped as redundant — the BT method is covered
# on better-suited block designs in Demos 02-05. (The folder keeps the cpp
# name "equi_and_bt" for parity.)
#
# Mirrors:
#  cpp/trex_selector_methods/ad/trex_da/demo_trex_da_08_mc_sim_equi_and_bt
#  Python/trex_selector_methods/ad/trex_da/demo_trex_da_08_mc_sim_equi_and_bt
# (same DGP, control parameters, sweep axes, and result files — one TXT/CSV
# pair per SNR column).
#
# Results are saved to simulation_results/data/.
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

OUT_DIR <- file.path(this_dir_, "simulation_results", "data")

# Simulation config (mirrors cpp)
N <- 300L
P <- 1000L
S <- 10L
AMPLITUDE <- 3.0
TFDR <- 0.2
K <- 20L
NUM_MC <- 200L
BASE_SEED <- 2026L

DA_SPEC <- list(method = "EQUI")


# ==============================================================================
# 2D sweep: SNR (outer, one output file per value) x rho (inner sweep)
# ==============================================================================

part_1_2d_snr_rho_sweep <- function() {
  snr_grid <- c(0.5, 1.0, 2.0, 5.0)
  rho_grid <- c(0.0, 0.025, 0.05, 0.1, 0.2, 0.4, 0.7, 0.9)

  for (si in seq_along(snr_grid)) {
    snr <- snr_grid[si]

    # Filesystem-safe SNR tag: 0.5 -> "0p5", 2.0 -> "2".
    snr_tag <- gsub(".", "p", fmt_num(snr), fixed = TRUE)

    # Disjoint seed blocks per SNR column (inner sweep spans gi * 10000),
    # so the columns are statistically independent.
    col_seed <- BASE_SEED + (si - 1L) * 1000000L

    run_param_sweep(
      OUT_DIR, sprintf("da_equi_rho_snr%s", snr_tag), "Rho", rho_grid,
      NUM_MC, TFDR, col_seed, default_solvers("+EQUI"),
      function(rho) {
        list(n = N, p = P, dgp = "dgp_equi_snr",
             kwargs = list(amplitude = AMPLITUDE, rho = rho,
                           snr = snr),
             support = NULL, s = S)
      },
      DA_SPEC, K,
      sprintf("Equi 2D sweep, SNR=%s, n=%d, p=%d, support=Random",
              fmt_num(snr), N, P),
      include_base_trex = TRUE
    )
  }
}


# ==============================================================================
# Main
# ==============================================================================

cat(sprintf("Running with %d worker cores\n", num_cores))

if (RUN_PART_1) part_1_2d_snr_rho_sweep()
cat("\nEqui 2D (SNR x rho) MC simulation complete.\n")

# All results are written: the cell cache of a stopped run is no longer needed.
cell_cache_done()
