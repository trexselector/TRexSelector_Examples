# =================================================================================
# demo_trex_gvs_02_mc_sim_scattered_blocks.R
# =================================================================================
#
# MC simulation: T-Rex+GVS on the scattered-grouped DGP.
#
# This scenario evaluates grouped-signal recovery when active variables are
# randomly scattered across the design rather than arranged in contiguous
# blocks. Within-group correlation is controlled through sd_x via
# rho = 1/(1 + sd_x^2).
#
# Mirrors cpp/Python demo_trex_gvs_02_mc_sim_scattered_blocks.
#
# One part:
#   1. 2-D SNR x rho sweep, using sd_x = sqrt((1 - rho)/rho)
#      (subsumes the former 1-D SNR and rho sweeps as its rho = 0.99 column and
#       SNR = 2 row)
#
# DGP: dgp_scattered_grouped_snr — n_groups = 3, group_size = 50, all grouped
#      variables active (s = 150), p = 500, remaining columns i.i.d. noise.
# Selector: K=20, tFDR=0.1, corr_max=0.98, single-linkage HAC,
#   lambda2 = CV_1SE_CCD (EN rows), CV_1SE_IEN_CCD (IEN rows).
# Methods compared: TENET, TENET_AUG, TIENET_AUG.
# MC: 200 trials per grid point (the full run takes hours — reduce num_MC and
#     the grids for a quick smoke test). Optional first CLI argument sets the
#     number of parallel worker cores (default TREX_NUM_WORKERS, else 8).
# Results are written to simulation_results/data/ (.txt + .csv).
# Indices are 1-based (R convention); the Python counterpart is 0-based.
#
# =================================================================================

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
source(file.path(this_dir_, "..", "..", "..", "simulation_utils.R"))
source(file.path(this_dir_, "..", "trex_gvs_dgps.R"))
source(file.path(this_dir_, "..", "trex_gvs_sim_common.R"))

OUT_DIR <- file.path(this_dir_, "simulation_results", "data")


# ==============================================================================
# Configuration
# ==============================================================================

CFG <- list(
  n        = 200L,
  p        = 500L,
  K        = 20L,
  tFDR     = 0.1,
  seed     = 2026L,
  num_MC   = num_mc(200L),  # TREX_NUM_MC for smoke runs
  corr_max = 0.98
)

# Preset (mirrors cpp make_scattered_grouped_preset()).
PRESET <- list(
  scenario_tag     = "Scattered-Grouped",
  file_stem_prefix = "gvs_scattered_grouped",
  n_groups         = 3L,
  group_size       = 50L,
  snr_grid_2d      = c(0.2, 0.5, 1.0, 2.0, 5.0),
  rho_grid_2d      = c(0.30, 0.50, 0.70, 0.90, 0.95, 0.99)
)

# The TIENET_AUG (IEN) row leaves en_solver at the library default (NULL ->
# "TENET" in trex_gvs_control(), ignored on the IEN path — its solver is
# always TIENET_AUG), mirroring the Python demos' en_solver=None, and names the
# IEN-geometry lambda2 tuner CV_1SE_IEN_CCD.
GVS_METHODS <- list(
  list(label = "TENET",      GVS_type = "EN",  en_solver = "TENET",     lambda2_method = "CV_1SE_CCD"),
  list(label = "TENET_AUG",  GVS_type = "EN",  en_solver = "TENET_AUG", lambda2_method = "CV_1SE_CCD"),
  # support cap 180 > s = 150 (library default 0.8 gives 144)
  list(label = "TIENET_AUG", GVS_type = "IEN", en_solver = NULL,        lambda2_method = "CV_1SE_IEN_CCD",
       cv_dfmax_ratio = 1.0)
)


# ==============================================================================
# Local helpers
# ==============================================================================

#' Run one MC point on the scattered-grouped DGP.
.scattered_point <- function(method, snr, sd_x, preset) {
  .run_mc_scattered(
    n          = CFG$n,
    p          = CFG$p,
    snr        = snr,
    sd_x       = sd_x,
    n_groups   = preset$n_groups,
    group_size = preset$group_size,
    tFDR       = CFG$tFDR,
    K          = CFG$K,
    num_MC     = CFG$num_MC,
    seed       = CFG$seed,
    GVS_type   = method$GVS_type,
    corr_max   = CFG$corr_max,
    hc_dist    = "Single",
    en_solver  = method$en_solver,
    lambda2_method = method$lambda2_method,
    cv_dfmax_ratio = method$cv_dfmax_ratio,
    n_cores    = num_cores
  )
}


# ==============================================================================
# Part 1: 2-D SNR x rho sweep
# ==============================================================================

run_scattered_part1_2d_sweep <- function(cfg, preset) {
  print_section_header(sprintf("Part 1: 2-D SNR x rho Sweep | %s",
                               preset$scenario_tag))
  cat(sprintf("  n=%d  p=%d  s=%d  MC=%d  tFDR=%s  K=%d  corr_max=%s\n",
              cfg$n, cfg$p, preset$n_groups * preset$group_size, cfg$num_MC,
              fmt_num(cfg$tFDR), cfg$K, fmt_num(cfg$corr_max)))
  cat("  methods: TENET / TENET_AUG / TIENET_AUG",
      "  |  lambda2=CV_1SE_CCD (EN), CV_1SE_IEN_CCD (IEN)  linkage=Single\n", sep = "")
  cat("  snr_grid: {", paste(fmt_num(preset$snr_grid_2d), collapse = ", "),
      "}\n", sep = "")
  cat("  rho_grid: {", paste(fmt_num(preset$rho_grid_2d), collapse = ", "),
      "}\n", sep = "")

  sweeps <- lapply(GVS_METHODS, function(m) {
    run_gvs_2d_sweep(
      function(snr, rho) {
        sd_x <- sqrt((1 - rho) / rho)
        .scattered_point(m, snr, sd_x, preset)
      },
      preset$snr_grid_2d, preset$rho_grid_2d, m$label, cfg$num_MC
    )
  })
  en     <- sweeps[[1L]]
  en_aug <- sweeps[[2L]]
  ien    <- sweeps[[3L]]

  snr_labels <- paste0("snr=", fmt_num(preset$snr_grid_2d))
  rho_labels <- paste0("rho=", fmt_num(preset$rho_grid_2d))

  print_mc_matrix("mean_TPP [TENET]", snr_labels, rho_labels, en$mean_TPP)
  print_mc_matrix("mean_FDP [TENET]", snr_labels, rho_labels, en$mean_FDP)
  print_mc_matrix("mean_TPP [TENET_AUG]", snr_labels, rho_labels,
                  en_aug$mean_TPP)
  print_mc_matrix("mean_FDP [TENET_AUG]", snr_labels, rho_labels,
                  en_aug$mean_FDP)
  print_mc_matrix("mean_TPP [TIENET_AUG]", snr_labels, rho_labels,
                  ien$mean_TPP)
  print_mc_matrix("mean_FDP [TIENET_AUG]", snr_labels, rho_labels,
                  ien$mean_FDP)

  save_mc_2d_tables(preset$scenario_tag, snr_labels, rho_labels, en, en_aug,
                    ien, cfg, OUT_DIR,
                    paste0(preset$file_stem_prefix, "_2d"))
}


# ==============================================================================
# Top-level driver
# ==============================================================================

run_scattered_grouped_benchmark <- function(cfg, preset) {  # nolint: object_length_linter. C++ name
  print_section_header(sprintf(
    "Scattered-grouped benchmark | %s\nDGP = dgp_scattered_grouped_snr\nn_groups = %d, group_size = %d, s = %d",
    preset$scenario_tag, preset$n_groups, preset$group_size,
    preset$n_groups * preset$group_size
  ))

  run_scattered_part1_2d_sweep(cfg, preset)

  cat(sprintf("%s GVS MC simulations complete.\n", preset$scenario_tag))
}


# ==============================================================================
# main
# ==============================================================================

cat(sprintf("Running with %d worker cores\n\n", num_cores))

run_scattered_grouped_benchmark(CFG, PRESET)

cat("All scattered-grouped benchmark simulations complete.\n")

# All results are written: the cell cache of a stopped run is no longer needed.
cell_cache_done()
