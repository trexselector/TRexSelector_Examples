# ==============================================================================
# demo_trex_da_03_mc_sim_ar1_blocks.R
# ==============================================================================
#
# DA-TRex+BT MC simulations for the AR(1)-block DGP using BT aggregation.
#
# DGP: dgp_ar1_block_snr — block-diagonal AR(1) covariance. One active variable
# per block (OnePerBlock support). All M blocks are active (s = M).
#
# Every part loops over linkage in {Single, Complete, Average}. All 3 solvers
# (TLARS, TAFS, TOMP) run inside each linkage iteration.
#
#   Part 1: SNR sweep    SNR in {0.1, 0.2, 0.5, 0.6, 1.0, 2.0, 5.0}.
#   Part 2: rho sweep    rho in {0.0, 0.1, ..., 0.9}.
#   Part 3: Q sweep      Q in {5, 10, ..., 50}; p = M*Q varies; s = M = 5 fixed.
#   Part 4: M sweep      M in {1, 3, 5, 10, 15, 20, 30}; p = M*Q and s = M vary.
#
# Mirrors:
#  cpp/trex_selector_methods/ad/trex_da/demo_trex_da_03_mc_sim_ar1_blocks
#  Python/trex_selector_methods/ad/trex_da/demo_trex_da_03_mc_sim_ar1_blocks
#
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
RUN_PART_4 <- TRUE

OUT_DIR <- file.path(this_dir_, "simulation_results", "data")

# Shared base parameters (mirror cpp)
N <- 150L
M <- 5L              # number of AR(1) blocks
Q <- 5L              # block size; base p = M * Q = 25
BASE_P <- M * Q      # 25
AMPLITUDE <- 1.0
RHO <- 0.7
SNR <- 2.0
TFDR <- 0.2
K <- 20L
NUM_MC <- 200L
BASE_SEED <- 2026L

LINKAGES <- c("Single", "Complete", "Average")


# ==============================================================================
# Part 1: MC SNR sweep
# Fixed: n=150, M=5, Q=5 (p=25, s=5), rho=0.7.
# ==============================================================================

part_1_snr_sweep <- function() {
  snr_grid <- c(0.1, 0.2, 0.5, 0.6, 1.0, 2.0, 5.0)
  support_base <- make_support_one_per_block(M, BASE_P, Q)

  for (lnk in LINKAGES) {
    run_snr_sweep(
      OUT_DIR, sprintf("da_ar1_blocks_snr_%s", lnk), snr_grid, NUM_MC, TFDR,
      BASE_SEED, default_solvers("+BT"),
      function(snr) {
        list(n = N, p = BASE_P, dgp = "dgp_ar1_block_snr",
             kwargs = list(amplitude = AMPLITUDE, n_blocks = M,
                           rho = RHO, snr = snr),
             support = support_base, s = M)
      },
      list(method = "BT", hc_linkage = lnk), K,
      sprintf(paste0("AR(1)-block SNR sweep  n=%d  M=%d  Q=%d  p=%d  s=%d",
                     "  rho=%s  linkage=%s"),
              N, M, Q, BASE_P, M, fmt_num(RHO), lnk),
      include_base_trex = FALSE
    )
  }
}


# ==============================================================================
# Part 2: MC rho sweep
# Fixed: n=150, M=5, Q=5 (p=25, s=5), SNR=2.
# ==============================================================================

part_2_rho_sweep <- function() {
  rho_grid <- seq(0.0, 0.9, by = 0.1)
  support_base <- make_support_one_per_block(M, BASE_P, Q)

  for (lnk in LINKAGES) {
    run_param_sweep(
      OUT_DIR, sprintf("da_ar1_blocks_rho_%s", lnk), "Rho", rho_grid, NUM_MC,
      TFDR, BASE_SEED, default_solvers("+BT"),
      function(rho) {
        list(n = N, p = BASE_P, dgp = "dgp_ar1_block_snr",
             kwargs = list(amplitude = AMPLITUDE, n_blocks = M,
                           rho = rho, snr = SNR),
             support = support_base, s = M)
      },
      list(method = "BT", hc_linkage = lnk), K,
      sprintf(paste0("AR(1)-block rho sweep  n=%d  M=%d  Q=%d  p=%d  s=%d",
                     "  SNR=%s  linkage=%s"),
              N, M, Q, BASE_P, M, fmt_num(SNR), lnk),
      include_base_trex = FALSE
    )
  }
}


# ==============================================================================
# Part 3: MC Q sweep (block size)
# Fixed: n=150, M=5, rho=0.7, SNR=2.  s = M = 5 throughout.
# Swept: Q in {5, 10, ..., 50}; p = M*Q varies.
# Support recomputed per grid point (OnePerBlock on p_val = M*Q).
# ==============================================================================

part_3_q_sweep <- function() {
  q_grid <- seq(5L, 50L, by = 5L)

  make_dgp <- function(q) {
    q_int <- as.integer(q)
    p_val <- M * q_int
    list(n = N, p = p_val, dgp = "dgp_ar1_block_snr",
         kwargs = list(amplitude = AMPLITUDE, n_blocks = M, rho = RHO,
                       snr = SNR),
         support = make_support_one_per_block(M, p_val, q_int), s = M)
  }

  for (lnk in LINKAGES) {
    run_param_sweep(
      OUT_DIR, sprintf("da_ar1_blocks_Q_%s", lnk), "Q", q_grid, NUM_MC, TFDR,
      BASE_SEED, default_solvers("+BT"), make_dgp,
      list(method = "BT", hc_linkage = lnk), K,
      sprintf(paste0("AR(1)-block Q sweep  n=%d  M=%d  s=%d  rho=%s",
                     "  SNR=%s  p=M*Q varies  linkage=%s"),
              N, M, M, fmt_num(RHO), fmt_num(SNR), lnk),
      include_base_trex = FALSE
    )
  }
}


# ==============================================================================
# Part 4: MC M sweep (number of blocks)
# Fixed: n=150, Q=5, rho=0.7, SNR=2.
# Swept: M in {1, 3, 5, 10, 15, 20, 30}; p = M*Q and s = M both vary.
# Support recomputed per grid point.
# ==============================================================================

part_4_m_sweep <- function() {
  m_grid <- c(1L, 3L, 5L, 10L, 15L, 20L, 30L)

  make_dgp <- function(m) {
    m_int <- as.integer(m)
    p_val <- m_int * Q
    list(n = N, p = p_val, dgp = "dgp_ar1_block_snr",
         kwargs = list(amplitude = AMPLITUDE, n_blocks = m_int, rho = RHO,
                       snr = SNR),
         support = make_support_one_per_block(m_int, p_val, Q), s = m_int)
  }

  for (lnk in LINKAGES) {
    run_param_sweep(
      OUT_DIR, sprintf("da_ar1_blocks_M_%s", lnk), "M", m_grid, NUM_MC, TFDR,
      BASE_SEED, default_solvers("+BT"), make_dgp,
      list(method = "BT", hc_linkage = lnk), K,
      sprintf(paste0("AR(1)-block M sweep  n=%d  Q=%d  rho=%s",
                     "  SNR=%s  p=M*Q and s=M vary  linkage=%s"),
              N, Q, fmt_num(RHO), fmt_num(SNR), lnk),
      include_base_trex = FALSE
    )
  }
}


# ==============================================================================
# Main
# ==============================================================================

cat(sprintf("Running with %d worker cores\n", num_cores))

if (RUN_PART_1) part_1_snr_sweep()
if (RUN_PART_2) part_2_rho_sweep()
if (RUN_PART_3) part_3_q_sweep()
if (RUN_PART_4) part_4_m_sweep()
cat("\nAR(1)-block BT MC simulation complete.\n")

# All results are written: the cell cache of a stopped run is no longer needed.
cell_cache_done()
