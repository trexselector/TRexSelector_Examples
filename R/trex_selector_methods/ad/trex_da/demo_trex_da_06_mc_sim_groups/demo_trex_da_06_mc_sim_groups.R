# ==============================================================================
# demo_trex_da_06_mc_sim_groups.R
# ==============================================================================
#
# DA-TRex MC simulation for the prior-groups DGP.
#
# The predictors carry a known 3-level hierarchical group structure (groups of
# 10 / 50 / 250, fine -> coarse), passed to the selector via
# trex_da_control(prior_groups = ...). The PRIOR_GROUPS method (redefined
# 2026-07-17) treats these labels as merge CONSTRAINTS: the selector
# sub-clusters within their finest common refinement (hc_linkage, correlation
# distance) and calibrates over a data-driven ascending rho grid terminated by
# the conservative rho = 1 singleton anchor. Data is drawn from a multi-level
# latent-factor model with Toeplitz leaf blocks.
#
#   Part 1: SNR sweep (n=300, p=1000, s=10, Random support, 200 MC, 3 solvers).
#
# Mirrors:
#  cpp/trex_selector_methods/ad/trex_da/demo_trex_da_06_mc_sim_groups
#  Python/trex_selector_methods/ad/trex_da/demo_trex_da_06_mc_sim_groups
# (same DGP, control parameters, sweep axis, and result files).
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

OUT_DIR <- file.path(this_dir_, "simulation_results", "data")

# Simulation config (mirrors cpp SimConfig defaults)
N <- 300L
P <- 1000L
S <- 10L
AMPLITUDE <- 3.0
TFDR <- 0.2
K <- 20L
NUM_MC <- 200L
BASE_SEED <- 2026L
MAX_GAP <- 20L   # unused by the Random policy; kept for cpp parity

# 3-level hierarchy: groups of 10, 50, 250 (fine -> coarse) and their
# cumulative variance masses, fine -> coarse.
RHO_LEVELS <- c(0.55, 0.25, 0.10)
PHI_LEAF <- 0.50


#' Level 1: groups of 10; level 2: groups of 50; level 3: groups of 250.
make_group_structure <- function(p) {
  list((seq_len(p) - 1L) %/% 10L,
       (seq_len(p) - 1L) %/% 50L,
       (seq_len(p) - 1L) %/% 250L)
}


# ==============================================================================
# Part 1: MC SNR sweep
# DGP: dgp_groups_toeplitz_leaf with 3-level hierarchy, Random support.
# Fixed: n=300, p=1000, s=10, tFDR=0.2, K=20, num_MC=200.
# Swept: SNR in {0.1, 0.2, 0.5, 1.0, 2.0, 5.0}.
# ==============================================================================

part_1_snr_sweep <- function() {
  groups <- make_group_structure(P)

  # The prior groups act as merge CONSTRAINTS: the selector sub-clusters
  # within their finest common refinement (the groups of 10) and calibrates
  # over a data-driven rho grid ending at the conservative rho = 1 anchor.
  da_spec <- list(prior_groups = groups)

  run_snr_sweep(
    OUT_DIR, "da_groups_toeplitz_leaf",
    c(0.1, 0.2, 0.5, 1.0, 2.0, 5.0), NUM_MC, TFDR, BASE_SEED,
    default_solvers(),
    function(snr) {
      list(n = N, p = P, dgp = "dgp_groups_toeplitz_leaf",
           kwargs = list(amplitude = AMPLITUDE, groups = groups,
                         rho_levels = RHO_LEVELS, phi = PHI_LEAF,
                         snr = snr),
           support = NULL, s = S)
    },
    da_spec, K,
    sprintf(paste0("Prior Groups + Toeplitz leaf, n=%d, p=%d, support=Random",
                   ", phi_leaf=%s, max_gap=%d"),
            N, P, fmt_num(PHI_LEAF), MAX_GAP)
  )
}


# ==============================================================================
# Main
# ==============================================================================

cat(sprintf("Running with %d worker cores\n", num_cores))

if (RUN_PART_1) part_1_snr_sweep()
cat("\nPrior Groups MC simulations complete.\n")

# All results are written: the cell cache of a stopped run is no longer needed.
cell_cache_done()
