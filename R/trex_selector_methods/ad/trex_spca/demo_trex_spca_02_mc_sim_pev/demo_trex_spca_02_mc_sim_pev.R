# ==============================================================================
# demo_trex_spca_02_mc_sim_pev.R
# ==============================================================================
#
# Cumulative explained variance of T-Rex SPCA under four sweeps.
# R counterpart of cpp demo_trex_spca_02_mc_sim_pev.cpp and Python
# demo_trex_spca_02_mc_sim_pev.py.
#
# DGP: sparse M-factor model — X = Z * V^T + E.
#      n=50, p=100, M=3 latent factors, overlap_pool_size=30 (shared candidate
#      index pool across factors). Unlike demo 01 (which mirrors Fig. 2 of the
#      reference paper at p1=5), this demo mirrors Fig. 3 and therefore uses
#      p1=10 as its default number of true active loadings per factor.
#
# The headline metric is the PEV of Definition 1 of the reference paper: the
# cumulative ADJUSTED EV divided by the Signal + Mixed EV OF THE DATA — the
# total variance carried by the true active variables (union of the factor
# supports), a denominator FIXED per dataset and SHARED by all methods. The
# ratio is deliberately uncapped: a method that spends loadings on null
# variables (ordinary PCA above all) accumulates variance the denominator does
# not contain and climbs past 100 %. The FDR-controlled T-Rex variants stay
# below 100 %, with plateaus of about 80-83 % (ActiveSet) and 86-96 %
# (Thresholded). Alongside it, the total-variance PEV (bounded by 1) and
# PEVsigmix (the method's own Sig + Mix part over the same denominator; for
# OrdPCA the paper's "Ordinary PCA (Sig + Mix)" reference curve) are reported.
#
# Four sweeps, each varying ONE axis with the others held at their defaults
# (# PCs = 3, SNR = 0 dB, p1 = 10, tFDR = 0.10):
#
#   Part 1 — # extracted PCs   {1, 3, 5, 10, 20, 30, 40, 49}
#            The data still carries exactly M = 3 true factors; only the number
#            of components the methods EXTRACT grows. Components beyond the
#            third are null by construction.
#   Part 2 — SNR (dB)          {-10, -7, -5, -3, 0, 3, 5, 7, 10}
#   Part 3 — # true active loadings p1  {1, 5, 10, 15, 20, 25, 30}
#   Part 4 — target FDR        {0.01, 0.05, 0.10, ..., 0.50}
#            Only the T-Rex variants respond to this axis; the two PCA
#            baselines are tFDR-blind and are reported as flat references.
#   Part 5 — FDR/TPR heatmaps over the full (target FDR x SNR) grid, PC1
#            support recovery only, for two selector responses: the plug-in
#            ordinary PC1 (what T-Rex SPCA regresses on) and the true factor
#            scores z_1 (oracle response). Isolates the mechanism behind the
#            realized-FDR overshoot at loose targets; see run_fdr_tpr_heatmap.
#
# Methods compared (identical to demo 01): OrdPCA, OraclePCA and the four
# T-Rex SPCA variants (TENET / TENET_AUG x ActiveSet / Thresholded).
#
# Support indices are 1-based (R convention). The seeds keep the values of the
# C++ and Python demos, while R's generator draws other numbers from them.
#
# Usage: Rscript demo_trex_spca_02_mc_sim_pev.R [num_worker_cores]
#   Env: TREX_SPCA_PARTS  — part selection, default "12345" (e.g. "5")
#        TREX_SPCA_NUM_MC — pilot-run MC override (e.g. 10)
#
# Resumable: every finished cell (method x grid point, or one heatmap cell) is
# cached in simulation_results/cache/, so a stopped run continues where it
# left off.
# ==============================================================================

library(TRexSelector)

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

source(file.path(this_dir_, "..", "..", "..", "mc_runtime.R"))
source(file.path(this_dir_, "..", "trex_spca_sim_utils.R"))

# Ordered method names, shared by every part of this demo.
METHOD_NAMES <- c("OrdPCA", "OraclePCA",
                  "TRexSPCA-EN-Act", "TRexSPCA-ENaug-Act",
                  "TRexSPCA-EN-Thr", "TRexSPCA-ENaug-Thr")


# ==============================================================================
# Shared sweep driver
# ==============================================================================

#' One grid point of a sweep: everything a single run needs to vary.
#'
#' num_comp = 0 means "extract as many components as there are true factors";
#' axis_value is the value plotted on the sweep axis.
make_point <- function(snr = 0.0, p1 = 10L, tFDR = 0.10, num_comp = 0L,
                       axis_value = 0.0) {
  list(snr = snr, p1 = p1, tFDR = tFDR, num_comp = num_comp,
       axis_value = axis_value)
}

#' Run one sweep over the supplied grid points and save table + CSV.
#'
#' @param cfg             Base configuration (n, p, M, num_MC, base_seed, lambda_2).
#' @param points          List of make_point() grid points.
#' @param sweep_col       CSV column name of the sweep axis (e.g. "n_pcs").
#' @param sweep_header    Table header of the sweep axis.
#' @param sweep_precision Decimals of the axis values in the table header.
#' @param banner          Console banner line.
#' @param stem            Output file stem.
#' @param pev_only_table  TRUE shows only the cumulative-PEV rows in the table
#'                        (parts 2-4); FDR/TPR remain in the CSV.
run_sweep <- function(cfg, points, sweep_col, sweep_header, sweep_precision,
                      banner, stem, pev_only_table = FALSE) {
  cat("================================================================\n")
  cat(sprintf("  %s\n", banner))
  cat("================================================================\n")
  cat(sprintf("  n=%d  p=%d  M=%d  num_MC=%d  (%d cores)\n\n",
              cfg$n, cfg$p, cfg$M, cfg$num_MC, num_cores))

  empty_grid <- setNames(
    rep(list(numeric(length(points))), length(METHOD_NAMES)), METHOD_NAMES
  )
  fdr_map <- tpr_map <- pev_map <- pev_sig_map <- pev_sigmix_map <- empty_grid
  acc <- environment()  # the result grids that store() fills
  axis_values <- vapply(points, function(pt) pt$axis_value, numeric(1))

  for (si in seq_along(points)) {
    pt <- points[[si]]
    store <- function(name, r) {
      acc$fdr_map[[name]][si] <- r$avg_fdr
      acc$tpr_map[[name]][si] <- r$avg_tpr
      acc$pev_map[[name]][si] <- r$avg_pev
      acc$pev_sig_map[[name]][si] <- r$avg_pev_sig
      acc$pev_sigmix_map[[name]][si] <- r$avg_pev_sigmix
    }

    # Distinct seed band per grid point, derived from the varied value so that
    # re-running a single point reproduces its own data stream (cpp/Python
    # formula; si is 0-based there). R seeds may be negative, so the unsigned
    # wrap of the cpp offset is not needed.
    offset <- cfg$base_seed + as.integer(round(pt$axis_value * 7.0)) + (si - 1L)

    lbl <- sprintf("%s=%.*f", sweep_header, sweep_precision, pt$axis_value)
    cat("--------------------------------------------------\n")
    cat(lbl, "\n\n")

    make_data <- function(seed) {
      generate_sparse_factor_model(cfg$n, cfg$p, pt$p1, cfg$M, pt$snr,
                                   cfg$overlap_pool_size, seed)
    }
    # One (method x grid point) cell, restored from the cache when finished.
    cell <- function(name, compute) {
      store(name, cell_cached(sprintf("%s=%.4f|%s", sweep_col, pt$axis_value, name),
                              compute))
    }
    trex_spca <- function(name, mode, en_solver) {
      cell(name, function() {
        run_mc_trials_trex_spca(cfg$num_MC, paste0(lbl, " [", name, "]"),
                                make_data, pt$tFDR, mode, offset,
                                lambda_2 = cfg$lambda_2, en_solver = en_solver,
                                num_components = pt$num_comp,
                                num_cores = num_cores)
      })
    }

    # 1. Ordinary PCA
    cell("OrdPCA", function() {
      run_mc_trials_pca(cfg$num_MC, paste0(lbl, " [OrdPCA]"), make_data, offset,
                        num_components = pt$num_comp, num_cores = num_cores)
    })

    # 2. Oracle Thresholded PCA
    cell("OraclePCA", function() {
      run_mc_trials_oracle_pca(cfg$num_MC, paste0(lbl, " [OraclePCA]"),
                               make_data, pt$p1, offset,
                               num_components = pt$num_comp,
                               num_cores = num_cores)
    })

    # 3.-6. T-Rex SPCA — EN (Gram-based TENET) / ENaug (augmented TENET),
    # each with the ActiveSet and the Thresholded loading assembly
    trex_spca("TRexSPCA-EN-Act", "ActiveSet", "TENET")
    trex_spca("TRexSPCA-ENaug-Act", "ActiveSet", "TENET_AUG")
    trex_spca("TRexSPCA-EN-Thr", "Thresholded", "TENET")
    trex_spca("TRexSPCA-ENaug-Thr", "Thresholded", "TENET_AUG")
  }

  save_and_print_spca_sweep_results(
    out_dir = file.path(this_dir_, "simulation_results", "data"),
    file_stem = stem, num_MC = cfg$num_MC, sweep_col = sweep_col,
    sweep_header = sweep_header, sweep_precision = sweep_precision,
    axis_values = axis_values, method_names = METHOD_NAMES,
    fdr_map = fdr_map, tpr_map = tpr_map, pev_map = pev_map,
    pev_sig_map = pev_sig_map, pev_sigmix_map = pev_sigmix_map,
    include_support_metrics = !pev_only_table
  )
}


# ==============================================================================
# Part 5: realized FDR / TPR heatmaps over (target FDR x SNR), PC1 support
# ==============================================================================

#' Per-trial worker: PC1 support recovery for both selector responses.
#'
#' Runs the per-PC T-Rex+GVS(EN/TENET) selector — the configuration
#' TRexSPCASelector uses internally — twice on the same data:
#'
#'   plugin : y = X v_1, the plug-in ordinary PC1, i.e. exactly what T-Rex
#'            SPCA regresses on. The regression's own true support is DENSE
#'            (z_1 is a linear combination of all p variables), while the
#'            scoring truth is the sparse factor-model support of factor 1.
#'   oracle : y = z_1, the true factor scores from the DGP (unobservable in
#'            practice). Here the regression truth IS the sparse factor
#'            support, so selector-level FDR control and factor-model
#'            scoring coincide.
#'
#' @return c(fdr_plugin, tpr_plugin, fdr_oracle, tpr_oracle)
.trial_heatmap <- function(seed, cfg, snr, tfdr) {
  dat <- generate_sparse_factor_model(cfg$n, cfg$p, cfg$p1, cfg$M, snr,
                                      cfg$overlap_pool_size, seed)
  X <- center_columns(dat$X)

  # True support of factor 1 (the scoring truth).
  true_supp <- which(dat$V[, 1L] != 0)

  # Responses: plug-in ordinary PC1 and true factor scores.
  y_plugin <- drop(X %*% svd(X, nu = 0L, nv = 1L)$v)
  y_oracle <- dat$Z[, 1L] - mean(dat$Z[, 1L])

  score <- function(y) {
    # Selector seed -1: hardware-entropy dummies per trial (required for valid
    # MC FDR estimates). Fresh copies: the selector maps X and y zero-copy.
    sel <- TRexGVSSelector$new(
      X + 0, y + 0, tFDR = tfdr, seed = -1L, verbose = FALSE,
      gvs_control = trex_gvs_control(gvs_type = "EN", en_solver = "TENET",
                                     lambda_2 = -1, lambda2_method = "CV_1SE_CCD"),
      control = trex_control(solver = "TENET", scaling_mode = "L2",
                             lloop_strategy = "STANDARD")
    )
    sel$select()
    sel_idx <- sel$selected_indices
    tp <- length(intersect(sel_idx, true_supp))
    k <- length(sel_idx)
    c(fdr = if (k == 0L) 0.0 else (k - tp) / k,
      tpr = if (length(true_supp) == 0L) 0.0 else tp / length(true_supp))
  }

  c(score(y_plugin), score(y_oracle))
}

#' 2D sweep of PC1 support recovery: realized FDR and TPR over the full
#' (target FDR x SNR) grid, for two selector responses (see .trial_heatmap).
#'
#' The pairing isolates the mechanism behind the FDR overshoot at loose
#' targets: if the overshoot comes from scoring a dense-truth regression
#' against a sparse truth, it must vanish under the oracle response.
#'
#' Output: tidy CSV "response,metric,tfdr,snr_db,value" (metrics FDR / TPR),
#' plus compact console matrices.
run_fdr_tpr_heatmap <- function(cfg) {
  tfdr_grid <- c(0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50)
  snr_grid <- c(-10.0, -7.0, -5.0, -3.0, 0.0, 3.0, 5.0, 7.0, 10.0)
  n_tfdr <- length(tfdr_grid)
  n_snr <- length(snr_grid)

  cat("================================================================\n")
  cat("  T-Rex SPCA — Part 5: FDR/TPR heatmaps over (tFDR x SNR), PC1\n")
  cat("================================================================\n")
  cat(sprintf(paste0("  n=%d  p=%d  p1=%d  num_MC=%d  responses: plugin (X v1) / ",
                     "oracle (true z1)  (%d cores)\n\n"),
              cfg$n, cfg$p, cfg$p1, cfg$num_MC, num_cores))

  # res[[r]]: 1/2 = plugin FDR/TPR, 3/4 = oracle FDR/TPR; each (n_snr x n_tfdr).
  res <- rep(list(matrix(0, nrow = n_snr, ncol = n_tfdr)), 4L)

  for (si in seq_len(n_snr)) {
    snr <- snr_grid[si]
    for (ti in seq_len(n_tfdr)) {
      tfdr <- tfdr_grid[ti]
      # Data seed depends on (mc, snr) only — the SAME datasets are reused
      # across the tFDR axis, so the columns of each heatmap row are paired.
      v <- unlist(cell_cached(sprintf("p5|snr=%.1f|tfdr=%.2f", snr, tfdr), function() {
        out <- .mc_map(cfg$num_MC, function(mc) {
          .trial_heatmap(cfg$base_seed + as.integer(round(snr)) + mc * 1000L,
                         cfg, snr, tfdr)
        }, num_cores)
        colMeans(do.call(rbind, out))
      }))
      for (r in 1:4) res[[r]][si, ti] <- v[[r]]

      cat(sprintf("  SNR=%5g  tFDR=%.2f  ->  FDR %.3f (plugin) / %.3f (oracle) | TPR %.3f / %.3f\n",
                  snr, tfdr, res[[1L]][si, ti], res[[3L]][si, ti],
                  res[[2L]][si, ti], res[[4L]][si, ti]))
    }
  }

  # ---- console matrices ----
  resp_names <- c("plugin", "plugin", "oracle", "oracle")
  metric_names <- c("FDR", "TPR", "FDR", "TPR")
  for (r in 1:4) {
    cat(sprintf("\n  Realized %s [%s response]  (rows: SNR dB, cols: tFDR)\n",
                metric_names[r], resp_names[r]))
    cat("      ", sprintf("%7.2f", tfdr_grid), "\n", sep = "")
    for (si in seq_len(n_snr)) {
      cat(sprintf("  %4.0f", snr_grid[si]), sprintf("%7.3f", res[[r]][si, ]), "\n", sep = "")
    }
  }
  cat("\n")

  # ---- tidy CSV ----
  out_dir <- file.path(this_dir_, "simulation_results", "data")
  dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
  rows <- list()
  for (r in 1:4) {
    for (si in seq_len(n_snr)) {
      rows[[length(rows) + 1L]] <- data.frame(
        response = resp_names[r],
        metric   = metric_names[r],
        tfdr     = sprintf("%.6f", tfdr_grid),
        snr_db   = sprintf("%.6f", snr_grid[si]),
        value    = sprintf("%.6f", res[[r]][si, ]),
        stringsAsFactors = FALSE
      )
    }
  }
  csv_path <- file.path(out_dir, "demo_trex_spca_02_mc_sim_pev_fdr_heatmap.csv")
  utils::write.csv(do.call(rbind, rows), csv_path, row.names = FALSE, quote = FALSE)
  cat(sprintf("[Info] Heatmap CSV saved to: %s\n\n", csv_path))
}


# ==============================================================================
# Main
# ==============================================================================

# Defaults held fixed while a single axis is swept. p1 = 10 follows Fig. 3 of
# the reference paper (demo 01 mirrors Fig. 2 and uses p1 = 5).
cfg <- list(
  n                 = 50L,
  p                 = 100L,
  p1                = 10L,
  M                 = 3L,
  overlap_pool_size = 30L,
  tFDR              = 0.10,
  num_MC            = 200L,
  base_seed         = 42L,
  lambda_2          = -1.0  # -1 triggers k-fold CV for lambda_2 selection
)
# Optional pilot-run override, e.g. TREX_SPCA_NUM_MC=10 Rscript demo_...
if (nzchar(Sys.getenv("TREX_SPCA_NUM_MC"))) {
  cfg$num_MC <- as.integer(Sys.getenv("TREX_SPCA_NUM_MC"))
}

# Part selection, e.g. TREX_SPCA_PARTS=5 runs only the FDR/TPR heatmap.
parts <- Sys.getenv("TREX_SPCA_PARTS", "12345")

# =========================================================
# Part 1: number of extracted PCs
# =========================================================
if (grepl("1", parts, fixed = TRUE)) {
  # The DGP keeps its M = 3 true factors throughout; only the number of
  # components the methods extract varies, so every PC beyond the third can
  # contribute Null EV and nothing else.
  #
  # The grid stops at 49, not 50: X is centered, so rank(X) <= n - 1 = 49 and a
  # 50th component carries no variance at all. Asking the per-PC selector to
  # regress on that null direction throws outright, because its response
  # vector is constant. 49 is the largest attainable PC count on this design.
  pc_grid <- c(1L, 3L, 5L, 10L, 20L, 30L, 40L, 49L)
  points <- lapply(pc_grid, function(n_pcs) {
    make_point(snr = 0.0, p1 = cfg$p1, tFDR = cfg$tFDR, num_comp = n_pcs,
               axis_value = as.numeric(n_pcs))
  })
  run_sweep(cfg, points, "n_pcs", "#PCs", 0L,
            "T-Rex SPCA — Cumulative PEV vs. number of extracted PCs",
            "demo_trex_spca_02_mc_sim_pev_pcs")
}

# =========================================================
# Part 2: signal-to-noise ratio
# =========================================================
if (grepl("2", parts, fixed = TRUE)) {
  snr_grid <- c(-10.0, -7.0, -5.0, -3.0, 0.0, 3.0, 5.0, 7.0, 10.0)
  points <- lapply(snr_grid, function(snr) {
    make_point(snr = snr, p1 = cfg$p1, tFDR = cfg$tFDR, num_comp = 0L,
               axis_value = snr)   # 0 = M = 3 factors
  })
  run_sweep(cfg, points, "snr_db", "SNR(dB)", 1L,
            "T-Rex SPCA — Cumulative PEV (PCs 1-3) vs. SNR",
            "demo_trex_spca_02_mc_sim_pev_snr", pev_only_table = TRUE)
}

# =========================================================
# Part 3: number of true active loadings
# =========================================================
if (grepl("3", parts, fixed = TRUE)) {
  # Bounded above by overlap_pool_size = 30: each factor draws its support
  # from that shared pool, so p1 cannot exceed it.
  p1_grid <- c(1L, 5L, 10L, 15L, 20L, 25L, 30L)
  points <- lapply(p1_grid, function(p1) {
    make_point(snr = 0.0, p1 = p1, tFDR = cfg$tFDR, num_comp = 0L,
               axis_value = as.numeric(p1))
  })
  run_sweep(cfg, points, "p1", "#Active", 0L,
            "T-Rex SPCA — Cumulative PEV (PCs 1-3) vs. number of true active loadings",
            "demo_trex_spca_02_mc_sim_pev_loadings", pev_only_table = TRUE)
}

# =========================================================
# Part 4: target FDR
# =========================================================
if (grepl("4", parts, fixed = TRUE)) {
  tfdr_grid <- c(0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50)
  points <- lapply(tfdr_grid, function(tfdr) {
    make_point(snr = 0.0, p1 = cfg$p1, tFDR = tfdr, num_comp = 0L,
               axis_value = tfdr)
  })
  run_sweep(cfg, points, "tfdr", "tFDR", 2L,
            "T-Rex SPCA — Cumulative PEV (PCs 1-3) vs. target FDR",
            "demo_trex_spca_02_mc_sim_pev_tfdr", pev_only_table = TRUE)
}

# =========================================================
# Part 5: FDR/TPR heatmaps over (target FDR x SNR)
# =========================================================
if (grepl("5", parts, fixed = TRUE)) {
  run_fdr_tpr_heatmap(cfg)
}

cell_cache_done()
