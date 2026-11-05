# =================================================================================
# demo_trex_gvs_01_mc_sim_hastie_en_blocks.R
# =================================================================================
#
# MC simulation: T-Rex+GVS on the Hastie (Zou & Hastie 2005) DGP.
# This scenario evaluates grouped-signal recovery under strong equicorrelation,
# not sparse within-group support identification. Within-group correlation is
# controlled through sd_x, with rho = 1/(1 + sd_x^2).
#
# Mirrors cpp/Python demo_trex_gvs_01_mc_sim_hastie_en_blocks.
#
# Parts:
#   Part 1: 2-D SNR x rho grid.
#   Part 2: lambda2_method comparison — SVD vs CCD (EN), TIK_SVD vs IEN_CCD (IEN).
#   Part 3: hc_linkage comparison     — Single / Complete / Average / WPGMA.
#   Part 4: clustering cut            — corr_max 0.98 / noise floor / oracle groups.
#
# DGP: three equicorrelated groups of 50 variables each, all active (s = 150),
#      plus 350 null white-noise variables; p = 500 and n = 200.
# Selector: K=20, tFDR=0.1, corr_max=0.98, lambda2=CV_1SE_CCD on the EN track
#   and the IEN-geometry CV_1SE_IEN_CCD on the IEN track (Part 2 compares it
#   with CV_1SE_TIK_SVD).
# Methods compared: TENET, TENET_AUG, TIENET_AUG (Part 4: TENET, TIENET_AUG).
# Part selection: TREX_GVS_PARTS (default "1234"), e.g. TREX_GVS_PARTS=4 runs
#   only the cut comparison.
# MC: 200 trials per grid point (the full run takes hours — reduce num_MC and
#     the grids for a quick smoke test). Optional first CLI argument sets the
#     number of parallel worker cores (default TREX_NUM_WORKERS, else 8).
# Results are written to simulation_results/data/ (.txt + .csv per part).
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

SCENARIO_TAG <- "Hastie"

# Methods compared in the 2-D sweep. The TIENET_AUG (IEN) row leaves en_solver
# at the library default (NULL -> "TENET" in trex_gvs_control(), which the IEN
# path ignores — its solver is always TIENET_AUG), mirroring the Python demos'
# en_solver=None, and names the IEN-geometry lambda2 tuner CV_1SE_IEN_CCD.
GVS_METHODS <- list(
  list(label = "TENET",      GVS_type = "EN",  en_solver = "TENET",     lambda2_method = "CV_1SE_CCD"),
  list(label = "TENET_AUG",  GVS_type = "EN",  en_solver = "TENET_AUG", lambda2_method = "CV_1SE_CCD"),
  # support cap 180 > s = 150 (library default 0.8 gives 144)
  list(label = "TIENET_AUG", GVS_type = "IEN", en_solver = NULL,        lambda2_method = "CV_1SE_IEN_CCD",
       cv_dfmax_ratio = 1.0)
)

# Part selection (default "1234"); not part of the cache fingerprint
PARTS <- Sys.getenv("TREX_GVS_PARTS", "")
if (!nzchar(PARTS)) PARTS <- "1234"
run_part <- function(k) grepl(k, PARTS, fixed = TRUE)


# ==============================================================================
# Local helpers
# ==============================================================================

#' Run one MC point on the Hastie DGP (cv_dfmax_ratio NULL: library default).
.hastie_point <- function(GVS_type, en_solver, snr, sd_x,
                          hc_dist = "Single", lambda2_method = "CV_1SE_CCD",
                          cv_dfmax_ratio = NULL) {
  .run_mc_hastie(
    n        = CFG$n,
    p        = CFG$p,
    snr      = snr,
    sd_x     = sd_x,
    tFDR     = CFG$tFDR,
    K        = CFG$K,
    num_MC   = CFG$num_MC,
    seed     = CFG$seed,
    GVS_type = GVS_type,
    corr_max = CFG$corr_max,
    hc_dist  = hc_dist,
    en_solver      = en_solver,
    lambda2_method = lambda2_method,
    cv_dfmax_ratio = cv_dfmax_ratio,
    n_cores  = num_cores
  )
}

#' Run one fixed-operating-point comparison run with progress lines (all NaN
#' and nothing computed when TREX_SOLVERS does not select the label's method).
.comparison_point <- function(label, GVS_type, en_solver, snr, sd_x,
                              hc_dist = "Single",
                              lambda2_method = "CV_1SE_CCD",
                              cv_dfmax_ratio = NULL) {
  if (!method_selected(label)) return(skipped_point())
  cat(sprintf("  %s — Running %d MC trials ...\n", label, CFG$num_MC))
  r <- .hastie_point(GVS_type, en_solver, snr, sd_x,
                     hc_dist = hc_dist, lambda2_method = lambda2_method,
                     cv_dfmax_ratio = cv_dfmax_ratio)
  cat(sprintf("  %s — done. TPP=%.3f  FDP=%.3f\n\n",
              label, r$mean_TPP, r$mean_FDP))
  r
}


# ==============================================================================
# Part 1: 2-D SNR x rho sweep
# ==============================================================================

sim_hastie_part_1 <- function(cfg, tag) {
  snr_grid_2d <- c(0.2, 0.5, 1.0, 2.0, 5.0)
  rho_grid_2d <- c(0.30, 0.50, 0.70, 0.90, 0.95, 0.99)

  print_section_header(sprintf("Part 1/4: 2-D SNR x rho Sweep  |  %s", tag))
  cat(sprintf("  n=%d  p=%d  s=150  MC=%d  tFDR=%s  K=%d  corr_max=%s\n",
              cfg$n, cfg$p, cfg$num_MC, fmt_num(cfg$tFDR), cfg$K,
              fmt_num(cfg$corr_max)))
  cat("  methods: TENET / TENET_AUG / TIENET_AUG  |  lambda2=CV_1SE_CCD (EN), CV_1SE_IEN_CCD (IEN)",
      "  linkage=Single\n", sep = "")
  cat("  snr_grid: {", paste(fmt_num(snr_grid_2d), collapse = ", "), "}\n",
      sep = "")
  cat("  rho_grid: {", paste(fmt_num(rho_grid_2d), collapse = ", "), "}\n\n",
      sep = "")

  sweeps <- lapply(GVS_METHODS, function(m) {
    run_gvs_2d_sweep(
      function(snr, rho) {
        sd_x <- sqrt((1 - rho) / rho)
        .hastie_point(m$GVS_type, m$en_solver, snr, sd_x,
                      lambda2_method = m$lambda2_method,
                      cv_dfmax_ratio = m$cv_dfmax_ratio)
      },
      snr_grid_2d, rho_grid_2d, m$label, cfg$num_MC
    )
  })
  en_2d     <- sweeps[[1L]]
  en_aug_2d <- sweeps[[2L]]
  ien_2d    <- sweeps[[3L]]

  snr_labels <- paste0("snr=", fmt_num(snr_grid_2d))
  rho_labels <- paste0("rho=", fmt_num(rho_grid_2d))

  print_mc_matrix("mean_TPP [TENET]", snr_labels, rho_labels, en_2d$mean_TPP)
  print_mc_matrix("mean_FDP [TENET]", snr_labels, rho_labels, en_2d$mean_FDP)
  print_mc_matrix("mean_TPP [TENET_AUG]", snr_labels, rho_labels,
                  en_aug_2d$mean_TPP)
  print_mc_matrix("mean_FDP [TENET_AUG]", snr_labels, rho_labels,
                  en_aug_2d$mean_FDP)
  print_mc_matrix("mean_TPP [TIENET_AUG]", snr_labels, rho_labels,
                  ien_2d$mean_TPP)
  print_mc_matrix("mean_FDP [TIENET_AUG]", snr_labels, rho_labels,
                  ien_2d$mean_FDP)
  save_mc_2d_tables(tag, snr_labels, rho_labels, en_2d, en_aug_2d, ien_2d,
                    cfg, OUT_DIR, sprintf("gvs_%s_2d", tag))
}


# ==============================================================================
# Part 2: lambda2_method comparison
# ==============================================================================
# Compare lambda2 selection methods at the fixed operating point SNR = 2.0, rho = 0.7: CV_1SE_SVD vs CV_1SE_CCD for
# TENET, and the two IEN-geometry tuners CV_1SE_TIK_SVD vs CV_1SE_IEN_CCD for TIENET_AUG (the EN track rejects them,
# and an EN tuner on the IEN track would fit the wrong objective). With cv_dfmax_ratio = 1.0, CV_1SE_IEN_CCD caps
# its lambda1 path at n_train = 180 variables, above this design's s = 150 (the library default 0.8 gives 144).
# CV_1SE_TIK_SVD ignores the cap and needs p - M < n_train (M clusters, so p - M within-group contrasts). The
# library refuses it otherwise, e.g. on the trap blocks of demo 04 at rho = 0.99, so the sweeps of demos 01-07 use
# CV_1SE_IEN_CCD.

sim_hastie_part_2 <- function(cfg, tag) {
  fixed_snr <- 2.0
  fixed_rho <- 0.7
  fixed_sdx <- sqrt((1 - fixed_rho) / fixed_rho)
  # Rules per track: the IEN-geometry tuners are valid on the IEN track only.
  en_methods  <- c("CV_1SE_SVD", "CV_1SE_CCD")
  ien_methods <- c("CV_1SE_TIK_SVD", "CV_1SE_IEN_CCD")

  print_section_header(sprintf("Part 2/4: lambda2_method Comparison  |  %s",
                               tag))
  cat(sprintf("  n=%d  p=%d  s=150  MC=%d  tFDR=%s  K=%d\n",
              cfg$n, cfg$p, cfg$num_MC, fmt_num(cfg$tFDR), cfg$K))
  cat(sprintf("  fixed operating point: SNR=%s  rho=%s\n",
              fmt_num(fixed_snr), fmt_num(fixed_rho)))
  cat("  lambda2 methods: TENET CV_1SE_SVD / CV_1SE_CCD,",
      " TIENET_AUG CV_1SE_TIK_SVD / CV_1SE_IEN_CCD  (4 runs)\n\n", sep = "")

  en_results  <- list()
  ien_results <- list()
  for (k in seq_along(en_methods)) {
    en_results[[k]] <- .comparison_point(
      sprintf("TENET+%s", en_methods[k]), "EN", "TENET", fixed_snr,
      fixed_sdx, lambda2_method = en_methods[k]
    )
    ien_results[[k]] <- .comparison_point(
      sprintf("TIENET_AUG+%s", ien_methods[k]), "IEN", NULL, fixed_snr,
      fixed_sdx, lambda2_method = ien_methods[k], cv_dfmax_ratio = 1.0  # TIK_SVD ignores it
    )
  }

  save_mc_1d_method_tables(
    tag, "lambda2_method", "lambda2_method", en_methods, en_results,
    ien_results,
    sprintf("SNR=%s, rho=%s", fmt_num(fixed_snr), fmt_num(fixed_rho)),
    cfg, OUT_DIR, sprintf("gvs_%s_lambda2_method", tag),
    ien_param_labels = ien_methods
  )
}


# ==============================================================================
# Part 3: hc_linkage comparison
# ==============================================================================
# Compare HAC linkage rules (Single / Complete / Average / WPGMA) for TENET
# and TIENET_AUG at the same fixed operating point as Part 2, with lambda2
# fixed at CV_1SE_CCD (EN) and CV_1SE_IEN_CCD (IEN): one rule per track
# throughout, to isolate the linkage effect from lambda selection.

sim_hastie_part_3 <- function(cfg, tag) {
  fixed_snr <- 2.0
  fixed_rho <- 0.7
  fixed_sdx <- sqrt((1 - fixed_rho) / fixed_rho)
  linkages  <- c("Single", "Complete", "Average", "WPGMA")

  print_section_header(sprintf("Part 3/4: hc_linkage Comparison  |  %s", tag))
  cat(sprintf("  n=%d  p=%d  s=150  MC=%d  tFDR=%s  K=%d",
              cfg$n, cfg$p, cfg$num_MC, fmt_num(cfg$tFDR), cfg$K),
      "  lambda2=CV_1SE_CCD (EN), CV_1SE_IEN_CCD (IEN)\n", sep = "")
  cat(sprintf("  fixed operating point: SNR=%s  rho=%s\n",
              fmt_num(fixed_snr), fmt_num(fixed_rho)))
  cat("  linkages: Single / Complete / Average / WPGMA",
      "  |  selectors: TENET, TIENET_AUG  (8 runs)\n\n", sep = "")

  en_results  <- list()
  ien_results <- list()
  for (lk in linkages) {
    en_results[[length(en_results) + 1L]] <- .comparison_point(
      sprintf("TENET+%s", lk), "EN", "TENET", fixed_snr, fixed_sdx,
      hc_dist = lk, lambda2_method = "CV_1SE_CCD"
    )
    ien_results[[length(ien_results) + 1L]] <- .comparison_point(
      sprintf("TIENET_AUG+%s", lk), "IEN", NULL, fixed_snr, fixed_sdx,
      hc_dist = lk, lambda2_method = "CV_1SE_IEN_CCD", cv_dfmax_ratio = 1.0
    )
  }

  save_mc_1d_method_tables(
    tag, "hc_linkage", "hc_linkage", linkages, en_results, ien_results,
    sprintf("SNR=%s, rho=%s", fmt_num(fixed_snr), fmt_num(fixed_rho)),
    cfg, OUT_DIR, sprintf("gvs_%s_hc_linkage", tag)
  )
}


# ==============================================================================
# Part 4: clustering-cut comparison
# ==============================================================================
# Compare the clustering cut at the fixed operating point SNR = 2.0, rho = 0.7:
# the demos' fixed corr_max = 0.98 (no groups below rho = 0.99), the Bonferroni
# noise floor (about 0.35 here) and the DGP's oracle groups, for TENET and
# TIENET_AUG with lambda2 = CV_1SE_CCD (EN), CV_1SE_IEN_CCD (IEN) and single
# linkage. Records variable-level FDP / TPP next to group-level rates.

sim_hastie_part_4 <- function(cfg, tag) {
  fixed_snr <- 2.0
  fixed_rho <- 0.7
  fixed_sdx <- sqrt((1 - fixed_rho) / fixed_rho)
  floor_cut <- noise_floor_corr_max(cfg$n, cfg$p)
  cuts <- list(list(label = "fixed_0.98",  corr_max = cfg$corr_max, oracle = FALSE),
               list(label = "noise_floor", corr_max = floor_cut,    oracle = FALSE),
               list(label = "oracle",      corr_max = cfg$corr_max, oracle = TRUE))

  print_section_header(sprintf("Part 4/4: Clustering-cut Comparison  |  %s",
                               tag))
  cat(sprintf("  n=%d  p=%d  s=150  MC=%d  tFDR=%s  K=%d",
              cfg$n, cfg$p, cfg$num_MC, fmt_num(cfg$tFDR), cfg$K),
      "  lambda2=CV_1SE_CCD (EN), CV_1SE_IEN_CCD (IEN)  linkage=Single\n", sep = "")
  cat(sprintf("  fixed operating point: SNR=%s  rho=%s\n",
              fmt_num(fixed_snr), fmt_num(fixed_rho)))
  cat(sprintf("  cuts: corr_max %s / noise floor %.3f / oracle groups",
              fmt_num(cfg$corr_max), floor_cut),
      "  |  selectors: TENET, TIENET_AUG  (6 runs)\n\n", sep = "")

  # All NaN and nothing computed when TREX_SOLVERS does not select the method
  cut_point <- function(label, GVS_type, en_solver, lambda2_method, cut,
                        cv_dfmax_ratio = NULL) {
    if (!method_selected(label)) return(skipped_cut_point())
    cat(sprintf("  %s — Running %d MC trials ...\n", label, cfg$num_MC))
    r <- .run_mc_hastie_cut(
      n = cfg$n, p = cfg$p, snr = fixed_snr, sd_x = fixed_sdx,
      tFDR = cfg$tFDR, K = cfg$K, num_MC = cfg$num_MC, seed = cfg$seed,
      GVS_type = GVS_type, corr_max = cut$corr_max, en_solver = en_solver,
      lambda2_method = lambda2_method, oracle = cut$oracle,
      cv_dfmax_ratio = cv_dfmax_ratio, n_cores = num_cores
    )
    cat(sprintf("  %s — done. TPP=%.3f  FDP=%.3f  groups=%.3f\n\n",
                label, r$mean_TPP, r$mean_FDP, r$n_groups))
    r
  }

  en_results  <- list()
  ien_results <- list()
  for (cut in cuts) {
    en_results[[length(en_results) + 1L]] <- cut_point(
      sprintf("TENET+%s", cut$label), "EN", "TENET", "CV_1SE_CCD", cut
    )
    ien_results[[length(ien_results) + 1L]] <- cut_point(
      sprintf("TIENET_AUG+%s", cut$label), "IEN", NULL, "CV_1SE_IEN_CCD", cut,
      cv_dfmax_ratio = 1.0
    )
  }

  save_mc_cut_tables(
    tag, vapply(cuts, `[[`, character(1L), "label"), en_results, ien_results,
    fixed_snr, fixed_rho, floor_cut, cfg, OUT_DIR, sprintf("gvs_%s_cut", tag)
  )
}


# ==============================================================================
# main
# ==============================================================================

cat(sprintf("Running with %d worker cores\n\n", num_cores))

if (run_part("1")) sim_hastie_part_1(CFG, SCENARIO_TAG)
if (run_part("2")) sim_hastie_part_2(CFG, SCENARIO_TAG)
if (run_part("3")) sim_hastie_part_3(CFG, SCENARIO_TAG)
if (run_part("4")) sim_hastie_part_4(CFG, SCENARIO_TAG)

cat(sprintf("\n%s GVS MC simulations complete.\n", SCENARIO_TAG))

# All results are written: the cell cache of a stopped run is no longer needed.
cell_cache_done()
