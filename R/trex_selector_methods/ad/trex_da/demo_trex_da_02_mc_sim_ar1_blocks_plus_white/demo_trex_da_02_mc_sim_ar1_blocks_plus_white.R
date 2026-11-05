# ==============================================================================
# demo_trex_da_02_mc_sim_ar1_blocks_plus_white.R
# ==============================================================================
#
# DA-TRex+BT MC simulations for the AR(1)-block + white-noise DGP.
#
# DGP: dgp_ar1_block_white_snr — M AR(1) blocks (p_ar columns) plus p_white
# i.i.d. N(0,1) white-noise columns; p_total = p_ar + p_white = 1000 fixed.
# Active variables (s = M, one per AR block) lie in the AR part only.
#
# Every part loops over linkage in {Single, Complete, Average}. All 3 solvers
# (TLARS, TAFS, TOMP) run inside each linkage iteration.
#
#   Part 1: SNR sweep    SNR in {0.1, 0.2, 0.5, 0.6, 1.0, 2.0, 5.0}.
#   Part 2: rho sweep    rho in {0.0, 0.1, ..., 0.9}.
#   Part 3: Q sweep      Q in {5, 10, ..., 50}; p_ar = M*Q; p_white = 1000-p_ar.
#   Part 4: M sweep      M in {1, 3, 5, 10, 15, 20, 30}; p_ar, p_white, s = M vary.
#
# Mirrors:
#  cpp/trex_selector_methods/ad/trex_da/demo_trex_da_02_mc_sim_ar1_blocks_plus_white
#  Python/trex_selector_methods/ad/trex_da/demo_trex_da_02_mc_sim_ar1_blocks_plus_white
#
# Same DGP, control parameters, sweep axes, and result files.
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
N <- 300L
P_TOTAL <- 1000L
M <- 5L              # number of AR(1) blocks
Q <- 5L              # block size; base p_ar = M * Q = 25
P_AR_BASE <- M * Q   # 25
AMPLITUDE <- 1.0
RHO <- 0.7
SNR <- 2.0
TFDR <- 0.1
K <- 20L
NUM_MC <- 200L
BASE_SEED <- 2026L

LINKAGES <- c("Single", "Complete", "Average")


#' OnePerBlock support on the AR part: block size Q, one active per block.
.support_one_per_block <- function(p_ar, m) {
  make_support_one_per_block(m, p_ar, p_ar %/% m)
}


# ==============================================================================
# Part 1: MC SNR sweep
# Fixed: n=300, p_total=1000, M=5, Q=5 (p_ar=25, p_white=975, s=5), rho=0.7.
# ==============================================================================

part_1_snr_sweep <- function() {
  snr_grid <- c(0.1, 0.2, 0.5, 0.6, 1.0, 2.0, 5.0)
  p_white_base <- P_TOTAL - P_AR_BASE
  support_base <- .support_one_per_block(P_AR_BASE, M)

  for (lnk in LINKAGES) {
    run_snr_sweep(
      OUT_DIR, sprintf("da_ar1_blocks_plus_white_snr_%s", lnk), snr_grid,
      NUM_MC, TFDR, BASE_SEED, default_solvers("+BT"),
      function(snr) {
        list(n = N, p = P_TOTAL, dgp = "dgp_ar1_block_white_snr",
             kwargs = list(amplitude = AMPLITUDE, p_ar = P_AR_BASE,
                           p_white = p_white_base, n_blocks = M,
                           rho = RHO, snr = snr),
             support = support_base, s = M)
      },
      list(method = "BT", hc_linkage = lnk), K,
      sprintf(paste0("AR(1)+white SNR sweep  n=%d  p_total=%d  M=%d  Q=%d",
                     "  p_ar=%d  s=%d  rho=%s  linkage=%s"),
              N, P_TOTAL, M, Q, P_AR_BASE, M, fmt_num(RHO), lnk),
      include_base_trex = FALSE
    )
  }
}


# ==============================================================================
# Part 2: MC rho sweep
# Fixed: n=300, p_total=1000, M=5, Q=5, SNR=2.
# ==============================================================================

part_2_rho_sweep <- function() {
  rho_grid <- seq(0.0, 0.9, by = 0.1)
  p_white_base <- P_TOTAL - P_AR_BASE
  support_base <- .support_one_per_block(P_AR_BASE, M)

  for (lnk in LINKAGES) {
    run_param_sweep(
      OUT_DIR, sprintf("da_ar1_blocks_plus_white_rho_%s", lnk), "Rho",
      rho_grid, NUM_MC, TFDR, BASE_SEED, default_solvers("+BT"),
      function(rho) {
        list(n = N, p = P_TOTAL, dgp = "dgp_ar1_block_white_snr",
             kwargs = list(amplitude = AMPLITUDE, p_ar = P_AR_BASE,
                           p_white = p_white_base, n_blocks = M,
                           rho = rho, snr = SNR),
             support = support_base, s = M)
      },
      list(method = "BT", hc_linkage = lnk), K,
      sprintf(paste0("AR(1)+white rho sweep  n=%d  p_total=%d  M=%d  Q=%d",
                     "  s=%d  SNR=%s  linkage=%s"),
              N, P_TOTAL, M, Q, M, fmt_num(SNR), lnk),
      include_base_trex = FALSE
    )
  }
}


# ==============================================================================
# Part 3: MC Q sweep (block size)
# Fixed: n=300, p_total=1000, M=5, rho=0.7, SNR=2.  s = M = 5 throughout.
# Swept: Q in {5, 10, ..., 50}; p_ar = M*Q varies; p_white = 1000-p_ar.
# Support recomputed per grid point (OnePerBlock on p_ar).
# ==============================================================================

part_3_q_sweep <- function() {
  q_grid <- seq(5L, 50L, by = 5L)

  make_dgp <- function(q) {
    q_int <- as.integer(q)
    p_ar <- M * q_int
    p_white <- P_TOTAL - p_ar
    list(n = N, p = P_TOTAL, dgp = "dgp_ar1_block_white_snr",
         kwargs = list(amplitude = AMPLITUDE, p_ar = p_ar,
                       p_white = p_white, n_blocks = M, rho = RHO, snr = SNR),
         support = .support_one_per_block(p_ar, M), s = M)
  }

  for (lnk in LINKAGES) {
    run_param_sweep(
      OUT_DIR, sprintf("da_ar1_blocks_plus_white_Q_%s", lnk), "Q", q_grid,
      NUM_MC, TFDR, BASE_SEED, default_solvers("+BT"), make_dgp,
      list(method = "BT", hc_linkage = lnk), K,
      sprintf(paste0("AR(1)+white Q sweep  n=%d  p_total=%d  M=%d  s=%d",
                     "  rho=%s  SNR=%s  p_ar=M*Q varies  linkage=%s"),
              N, P_TOTAL, M, M, fmt_num(RHO), fmt_num(SNR), lnk),
      include_base_trex = FALSE
    )
  }
}


# ==============================================================================
# Part 4: MC M sweep (number of blocks)
# Fixed: n=300, p_total=1000, Q=5, rho=0.7, SNR=2.
# Swept: M in {1, 3, 5, 10, 15, 20, 30}; p_ar = M*Q, p_white = 1000-p_ar,
# s = M all vary. Support recomputed per grid point.
# ==============================================================================

part_4_m_sweep <- function() {
  m_grid <- c(1L, 3L, 5L, 10L, 15L, 20L, 30L)

  make_dgp <- function(m) {
    m_int <- as.integer(m)
    p_ar <- m_int * Q
    p_white <- P_TOTAL - p_ar
    list(n = N, p = P_TOTAL, dgp = "dgp_ar1_block_white_snr",
         kwargs = list(amplitude = AMPLITUDE, p_ar = p_ar,
                       p_white = p_white, n_blocks = m_int, rho = RHO,
                       snr = SNR),
         support = .support_one_per_block(p_ar, m_int), s = m_int)
  }

  for (lnk in LINKAGES) {
    run_param_sweep(
      OUT_DIR, sprintf("da_ar1_blocks_plus_white_M_%s", lnk), "M", m_grid,
      NUM_MC, TFDR, BASE_SEED, default_solvers("+BT"), make_dgp,
      list(method = "BT", hc_linkage = lnk), K,
      sprintf(paste0("AR(1)+white M sweep  n=%d  p_total=%d  Q=%d",
                     "  rho=%s  SNR=%s  p_ar=M*Q and s=M vary  linkage=%s"),
              N, P_TOTAL, Q, fmt_num(RHO), fmt_num(SNR), lnk),
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
cat("\nAR(1)+white-noise block BT MC simulation complete.\n")

# All results are written: the cell cache of a stopped run is no longer needed.
cell_cache_done()
