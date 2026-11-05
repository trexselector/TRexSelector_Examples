# ==============================================================================
# demo_trex_spca_01_mc_sim.R
# ==============================================================================
#
# Monte Carlo simulation comparing T-Rex SPCA against PCA baselines.
# R counterpart of cpp demo_trex_spca_01_mc_sim.cpp.
#
# DGP: sparse M-factor model — X = Z * V^T + E.
#      n=50, p=100, M=3 latent factors, p1=5 active loadings per factor,
#      overlap_pool_size=30 (shared candidate-index pool across factors).
#
# Section 1: SNR sweep over the legacy CRAN reference grid
#            {-10, -7, -5, -3, 0, 3, 5, 7, 10} dB, num_MC=200.
#   Methods compared:
#     1. OrdPCA             — ordinary PCA (baseline; no sparsity constraint)
#     2. OraclePCA          — thresholded PCA with known support cardinality p1
#     3. TRexSPCA-EN-Act    — T-Rex+GVS(EN/TENET, Gram-based) + ActiveSet
#     4. TRexSPCA-ENaug-Act — T-Rex+GVS(EN/TENETAug, augmented) + ActiveSet
#     5. TRexSPCA-EN-Thr    — T-Rex+GVS(EN/TENET, Gram-based) + Thresholded
#     6. TRexSPCA-ENaug-Thr — T-Rex+GVS(EN/TENETAug, augmented) + Thresholded
#
#   All methods see the same center-only X (covariance-PCA footing, as in the
#   legacy R reference); the per-PC selector uses the default L2 scaling.
#
# Section 2: union-support FDR per PC (ActiveSet/TENET) over the same grid —
#            does the selector start harvesting pure-noise variables on
#            components beyond the first?
#
# Metrics (num_MC-averaged; TPR/FDR on PC1 support only — see evaluate_spca):
#   - FDR  — false discovery rate of PC1's estimated loading support
#   - TPR  — true positive rate  of PC1's estimated loading support
#   - PEV  — cumulative adjusted EV under three normalizations
#            (total variance / Definition 1 / method's own Sig+Mix part)
#
# Usage: Rscript demo_trex_spca_01_mc_sim.R [num_worker_cores]
#   Env: TREX_SPCA_PARTS  — section selection, default "12" (e.g. "2")
#        TREX_SPCA_NUM_MC — pilot-run MC override (e.g. 10)
#
# Resumable: every finished (method x SNR) cell is cached in
# simulation_results/cache/, so a stopped run continues where it left off.
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


# ==============================================================================
# Monte Carlo Simulation — SNR sweep
# ==============================================================================

#' Run the SNR-sweep MC simulation and save results.
#'
#' @param cfg        List of simulation parameters (n, p, p1, M,
#'                   overlap_pool_size, tFDR, lambda_2, num_MC, base_seed).
#' @param snr_values SNR grid in dB.
#' @param stem       Output file stem (no folder, no extension).
demo_trex_spca_mc_snr_sweep <- function(cfg, snr_values, stem) {
  cat("================================================================\n")
  cat("  T-Rex SPCA MC Simulation — SNR Sweep\n")
  cat("================================================================\n")
  cat(sprintf("  n=%d  p=%d  M=%d  p1=%d  tFDR=%g  num_MC=%d  (%d cores)\n\n",
              cfg$n, cfg$p, cfg$M, cfg$p1, cfg$tFDR, cfg$num_MC, num_cores))

  method_names <- c("OrdPCA", "OraclePCA",
                    "TRexSPCA-EN-Act", "TRexSPCA-ENaug-Act",
                    "TRexSPCA-EN-Thr", "TRexSPCA-ENaug-Thr")

  empty_grid <- setNames(
    rep(list(numeric(length(snr_values))), length(method_names)), method_names
  )
  fdr_map <- tpr_map <- pev_map <- pev_sig_map <- pev_sigmix_map <- empty_grid
  acc <- environment()  # the result grids that store() fills

  for (si in seq_along(snr_values)) {
    store <- function(name, r) {
      acc$fdr_map[[name]][si] <- r$avg_fdr
      acc$tpr_map[[name]][si] <- r$avg_tpr
      acc$pev_map[[name]][si] <- r$avg_pev
      acc$pev_sig_map[[name]][si] <- r$avg_pev_sig
      acc$pev_sigmix_map[[name]][si] <- r$avg_pev_sigmix
    }
    snr <- snr_values[si]
    offset <- cfg$base_seed + as.integer(round(snr))
    snr_lbl <- sprintf("SNR=%.1f dB", snr)

    cat("--------------------------------------------------\n")
    cat(snr_lbl, "\n\n")

    make_data <- function(seed) {
      generate_sparse_factor_model(cfg$n, cfg$p, cfg$p1, cfg$M, snr,
                                   cfg$overlap_pool_size, seed)
    }
    # One (method x SNR) cell, restored from the cache when already finished.
    cell <- function(name, compute) {
      store(name, cell_cached(sprintf("s1|snr=%.1f|%s", snr, name), compute))
    }
    trex_spca <- function(name, mode, en_solver) {
      cell(name, function() {
        run_mc_trials_trex_spca(cfg$num_MC, paste0(snr_lbl, " [", name, "]"),
                                make_data, cfg$tFDR, mode, offset,
                                lambda_2 = cfg$lambda_2, en_solver = en_solver,
                                num_cores = num_cores)
      })
    }

    # 1. Ordinary PCA
    cell("OrdPCA", function() {
      run_mc_trials_pca(cfg$num_MC, paste0(snr_lbl, " [OrdPCA]"), make_data,
                        offset, num_cores = num_cores)
    })

    # 2. Oracle Thresholded PCA
    cell("OraclePCA", function() {
      run_mc_trials_oracle_pca(cfg$num_MC, paste0(snr_lbl, " [OraclePCA]"),
                               make_data, cfg$p1, offset, num_cores = num_cores)
    })

    # 3.-6. T-Rex SPCA — EN (Gram-based TENET) / ENaug (augmented TENET),
    # each with the ActiveSet and the Thresholded loading assembly
    trex_spca("TRexSPCA-EN-Act", "ActiveSet", "TENET")
    trex_spca("TRexSPCA-ENaug-Act", "ActiveSet", "TENET_AUG")
    trex_spca("TRexSPCA-EN-Thr", "Thresholded", "TENET")
    trex_spca("TRexSPCA-ENaug-Thr", "Thresholded", "TENET_AUG")
  }

  save_and_print_spca_mc_results(
    out_dir = file.path(this_dir_, "simulation_results", "data"),
    file_stem = stem, num_MC = cfg$num_MC,
    snr_values = snr_values, method_names = method_names,
    fdr_map = fdr_map, tpr_map = tpr_map, pev_map = pev_map,
    pev_sig_map = pev_sig_map, pev_sigmix_map = pev_sigmix_map
  )
}


# ==============================================================================
# Section 2: union-support FDR per PC
# ==============================================================================

#' Realized FDR of each PC's selected support, scored against the UNION of
#' the true factor supports.
#'
#' Beyond PC1 there is no unambiguous per-factor ground truth: the factor
#' supports overlap, and PCA's orthogonality constraint mixes the later
#' components across factors, so a per-factor FDR would penalize legitimately
#' selected leaked variables and measure the ambiguity of the truth rather
#' than the selector. The UNION of the factor supports, however, is well-posed
#' for every component — a variable outside the union is null for EVERY
#' factor, so selecting it is a false discovery no matter how the components
#' mix. This sweep answers the one question about the later PCs that CAN be
#' asked cleanly: does the selector start harvesting pure-noise variables on
#' components beyond the first?
#'
#' One variant is run: ActiveSet + TENET — the selector's own support (the
#' Thresholded mode rebuilds loadings at the same cardinality). Reported per
#' PC m: mean union-FDR and mean support size |A_m|.
#'
#' Output: tidy CSV "pc,metric,snr_db,value" (metrics FDRunion / k).
demo_trex_spca_union_fdr_sweep <- function(cfg, snr_values, stem) {
  cat("================================================================\n")
  cat("  T-Rex SPCA — union-support FDR per PC (ActiveSet/TENET)\n")
  cat("================================================================\n")
  cat(sprintf("  n=%d  p=%d  M=%d  p1=%d  tFDR=%g  num_MC=%d  (%d cores)\n\n",
              cfg$n, cfg$p, cfg$M, cfg$p1, cfg$tFDR, cfg$num_MC, num_cores))

  M <- cfg$M
  n_snr <- length(snr_values)
  fdr_mean <- matrix(0, nrow = M, ncol = n_snr)
  k_mean <- matrix(0, nrow = M, ncol = n_snr)

  one_trial <- function(seed, snr) {
    dat <- generate_sparse_factor_model(cfg$n, cfg$p, cfg$p1, cfg$M, snr,
                                        cfg$overlap_pool_size, seed)
    X <- center_columns(dat$X)

    # Union of the true factor supports (per variable / row of V).
    in_union <- rowSums(dat$V != 0) > 0

    sel <- TRexSPCASelector$new(
      X + 0, M = M, tFDR = cfg$tFDR,
      control = trex_spca_control(mode = "ActiveSet", gvs_type = "EN",
                                  en_solver = "TENET",
                                  lambda_2 = cfg$lambda_2,
                                  lambda2_method = "CV_1SE_CCD"),
      seed = -1L, verbose = FALSE
    )
    sel$select()

    fdr <- numeric(M)
    k <- numeric(M)
    for (m in seq_len(M)) {
      A <- sel$active_sets[[m]]
      fd <- sum(!in_union[A])
      fdr[m] <- if (length(A) == 0L) 0.0 else fd / length(A)
      k[m] <- length(A)
    }
    c(fdr, k)
  }

  for (si in seq_along(snr_values)) {
    snr <- snr_values[si]
    offset <- cfg$base_seed + as.integer(round(snr))

    r <- unlist(cell_cached(sprintf("s2|snr=%.1f", snr), function() {
      res_list <- .mc_map(cfg$num_MC,
                          function(mc) one_trial(offset + mc * 1000L, snr),
                          num_cores)
      res_mat <- do.call(rbind, res_list)       # (num_MC, 2*M)
      c(fdr = colMeans(res_mat[, seq_len(M), drop = FALSE]),
        k = colMeans(res_mat[, M + seq_len(M), drop = FALSE]))
    }))
    fdr_mean[, si] <- r[seq_len(M)]
    k_mean[, si] <- r[M + seq_len(M)]

    cat(sprintf("  SNR=%5g dB — union-FDR:", snr),
        sprintf("  PC%d=%.3f", seq_len(M), fdr_mean[, si]), "\n", sep = "")
  }

  # ---- console table ----
  cat("\n  Union-support FDR per PC (rows) over SNR (cols):\n")
  cat("      ", sprintf("%8.1f", snr_values), "\n", sep = "")
  for (m in seq_len(M)) {
    cat(sprintf("  PC%d ", m), sprintf("%8.3f", fdr_mean[m, ]), "\n", sep = "")
  }
  cat("\n  Mean support size |A_m|:\n")
  for (m in seq_len(M)) {
    cat(sprintf("  PC%d ", m), sprintf("%8.2f", k_mean[m, ]), "\n", sep = "")
  }
  cat("\n")

  # ---- tidy CSV ----
  out_dir <- file.path(this_dir_, "simulation_results", "data")
  dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
  rows <- list()
  for (m in seq_len(M)) {
    for (si in seq_along(snr_values)) {
      rows[[length(rows) + 1L]] <- data.frame(
        pc     = m,
        metric = c("FDRunion", "k"),
        snr_db = sprintf("%.6f", snr_values[si]),
        value  = sprintf("%.6f", c(fdr_mean[m, si], k_mean[m, si])),
        stringsAsFactors = FALSE
      )
    }
  }
  csv_path <- file.path(out_dir, paste0(stem, ".csv"))
  utils::write.csv(do.call(rbind, rows), csv_path,
                   row.names = FALSE, quote = FALSE)
  cat(sprintf("[Info] Union-FDR CSV saved to: %s\n\n", csv_path))
}


# ==============================================================================
# Main
# ==============================================================================

cfg <- list(
  n                 = 50L,
  p                 = 100L,
  p1                = 5L,
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

# Legacy CRAN reference grid, mirroring the cpp demo exactly.
snr_values <- c(-10.0, -7.0, -5.0, -3.0, 0.0, 3.0, 5.0, 7.0, 10.0)

# Part selection, e.g. TREX_SPCA_PARTS=2 runs only the union-FDR sweep.
parts <- Sys.getenv("TREX_SPCA_PARTS", "12")

# =========================================================
# Section 1: SNR sweep
# =========================================================
if (grepl("1", parts, fixed = TRUE)) {
  demo_trex_spca_mc_snr_sweep(cfg, snr_values, "demo_trex_spca_01_mc_sim")
}

# =========================================================
# Section 2: union-support FDR per PC
# =========================================================
if (grepl("2", parts, fixed = TRUE)) {
  demo_trex_spca_union_fdr_sweep(cfg, snr_values,
                                 "demo_trex_spca_01_mc_sim_union_fdr")
}

cell_cache_done()
