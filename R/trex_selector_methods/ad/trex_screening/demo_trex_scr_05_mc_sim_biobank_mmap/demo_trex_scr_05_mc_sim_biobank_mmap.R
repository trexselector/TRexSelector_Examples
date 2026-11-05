# ==============================================================================
# demo_trex_scr_05_mc_sim_biobank_mmap.R
# ==============================================================================
#
# Biobank Screen-TRex Selector ("Algorithm 1"), memory-mapped Monte Carlo variant.
#
# Statistically identical to demo 04, but the internal dummy matrices are backed
# by memory-mapped files (use_memory_mapping = TRUE), the storage mode intended
# for biobank-scale problems where holding the dummy matrices in RAM is
# infeasible. Everything else about Algorithm 1 — the routing between Ordinary,
# Bootstrap-CI and the T-Rex fallback and its per-method Usage % — is
# unchanged. As in the cpp source, the design matrix X still lives in RAM; only
# the dummy matrices move to disk, and the library manages those temporary
# files in its own scratch directories (nothing to clean up in this script).
#
# See demo 04 for the full description of the result accessors and method-used
# labels (they are identical here).
#
# Mirrors:
#   Python/trex_selector_methods/ad/trex_screening/demo_trex_scr_05_mc_sim_biobank_mmap/
#       demo_trex_scr_05_mc_sim_biobank_mmap.py
#   cpp/trex_selector_methods/ad/trex_screening/demo_trex_scr_05_mc_sim_biobank_mmap/
#       demo_trex_scr_05_mc_sim_biobank_mmap.cpp
#
# Demo content (two Monte Carlo SNR sweeps, matching the cpp Parts 1-2):
#   Part 1: single phenotype (s=10): per-method Usage %, FDR, TPR, Est. FDR.
#   Part 2: q=5 phenotypes sharing one X (s=5): same per-method metrics.
#
# The Monte Carlo counts and R_boot match the cpp/Python reference
# (num_MC = 200, num_MC_multi = 200, R_boot = 1000). Support indices are
# 1-based (R convention).
#
# Usage: Rscript demo_trex_scr_05_mc_sim_biobank_mmap.R [num_worker_cores]
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

# Monte Carlo counts mirror the cpp/Python reference (num_MC = num_MC_multi = 200).
NUM_MC       <- 200L
NUM_MC_MULTI <- 200L

N <- 300L
P <- 1000L
SNR_GRID        <- c(0.1, 0.5, 1.0, 2.0, 5.0)
TARGET_FDR_TREX <- 0.10
R_BOOT          <- 1000L   # mirrors the cpp reference (R_boot = 1000)
USE_MMAP        <- TRUE    # dummy (D) matrices backed by on-disk files


#' Empty per-method map: display name -> per-SNR numeric vector.
empty_maps <- function() {
  setNames(
    lapply(BIOBANK_DISPLAY, function(x) numeric(length(SNR_GRID))), BIOBANK_DISPLAY
  )
}

#' Copy one SNR level's aggregate into the display-keyed maps.
store_agg <- function(agg, maps, si) {
  for (j in seq_along(BIOBANK_METHODS)) {
    canon <- BIOBANK_METHODS[j]
    disp <- BIOBANK_DISPLAY[j]
    maps$fdp[[disp]][si]   <- agg[[canon]]$fdp
    maps$tpp[[disp]][si]   <- agg[[canon]]$tpp
    maps$est[[disp]][si]   <- agg[[canon]]$est
    maps$usage[[disp]][si] <- agg[[canon]]$usage
  }
  maps
}


# ==============================================================================
# Part 1 — MC SNR sweep, single phenotype (memory-mapped)
# ==============================================================================

part_1 <- function() {
  cat("\n", strrep("=", 70L), "\n", sep = "")
  cat("Part 1: Biobank Screen-TRex - MC SNR Sweep (Memory-Mapped)\n")
  cat(strrep("=", 70L), "\n", sep = "")
  support_size <- 10L
  cat(sprintf("n = %d, p = %d, s = %d, num_MC = %d\n\n",
              N, P, support_size, NUM_MC))

  maps <- list(fdp = empty_maps(), tpp = empty_maps(),
               est = empty_maps(), usage = empty_maps())

  for (si in seq_along(SNR_GRID)) {
    snr <- SNR_GRID[si]
    trial_fn <- function(mc) {
      # mirrors cpp/Python seed = 1000 * snr_idx + mc
      biobank_single_trial(N, P, support_size, snr,
                           seed = 1000L * (si - 1L) + mc,
                           use_memory_mapping = USE_MMAP, R_boot = R_BOOT,
                           tFDR = TARGET_FDR_TREX)
    }
    agg <- run_biobank_snr_cell(sprintf("part1|snr=%.2f", snr), NUM_MC, trial_fn,
                                num_cores, TARGET_FDR_TREX)
    maps <- store_agg(agg, maps, si)
    cat(sprintf("  SNR %.2f - completed %d runs.\n", snr, NUM_MC))
  }

  save_and_print_biobank_mc(
    NUM_MC, OUT_DIR,
    sprintf("scr_biobank_mmap_snr_n%d_p%d_s%d", N, P, support_size),
    SNR_GRID, BIOBANK_DISPLAY, maps$fdp, maps$tpp, maps$est, maps$usage
  )
}


# ==============================================================================
# Part 2 — MC SNR sweep, multiple phenotypes (memory-mapped)
# ==============================================================================

part_2 <- function() {
  cat("\n", strrep("=", 70L), "\n", sep = "")
  cat("Part 2: Biobank Screen-TRex - MC SNR Sweep, Multiple Phenotypes (Memory-Mapped)\n")
  cat(strrep("=", 70L), "\n", sep = "")
  q <- 5L
  support_size <- 5L
  cat(sprintf("n = %d, p = %d, q = %d, s = %d, num_MC = %d\n\n",
              N, P, q, support_size, NUM_MC_MULTI))

  maps <- list(fdp = empty_maps(), tpp = empty_maps(),
               est = empty_maps(), usage = empty_maps())

  for (si in seq_along(SNR_GRID)) {
    snr <- SNR_GRID[si]
    trial_fn <- function(mc) {
      biobank_multi_trial(
        N, P, q, support_size, snr,
        x_seed        = 5000L * (si - 1L) + mc + 1L,   # shared X per trial
        sup_seed_base = 770000L + q * (NUM_MC_MULTI * (si - 1L) + mc),
        use_memory_mapping = USE_MMAP, R_boot = R_BOOT, tFDR = TARGET_FDR_TREX
      )
    }
    agg <- run_biobank_snr_cell(sprintf("part2|snr=%.2f|q=%d", snr, q), NUM_MC_MULTI,
                                trial_fn, num_cores, TARGET_FDR_TREX)
    maps <- store_agg(agg, maps, si)
    cat(sprintf("  SNR %.2f - completed %d runs (%d phenotypes each).\n",
                snr, NUM_MC_MULTI, q))
  }

  save_and_print_biobank_mc(
    NUM_MC_MULTI, OUT_DIR,
    sprintf("scr_biobank_mmap_multi_n%d_p%d_q%d_s%d", N, P, q, support_size),
    SNR_GRID, BIOBANK_DISPLAY, maps$fdp, maps$tpp, maps$est, maps$usage
  )
}


# ==============================================================================
# Main
# ==============================================================================

part_1()
part_2()
cat("\ndemo_trex_scr_05_mc_sim_biobank_mmap.R complete.\n")

# All results are written: the cell cache of a stopped run is no longer needed.
cell_cache_done()
