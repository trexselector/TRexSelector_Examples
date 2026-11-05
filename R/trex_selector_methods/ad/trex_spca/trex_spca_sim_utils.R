# ==============================================================================
# trex_spca_sim_utils.R
# ==============================================================================
#
# Shared simulation utilities for the T-Rex SPCA demo files. R counterpart of
# cpp/trex_selector_methods/ad/trex_spca/trex_spca_sim_utils.hpp.
#
#   Data generation:
#     generate_sparse_factor_model()  — sparse M-factor model DGP
#                                       (returns list: X, Z, V, E, actual_snr_db)
#
#   Preprocessing:
#     center_columns()                — column centering (covariance-PCA footing)
#
#   Evaluation:
#     evaluate_spca()                 — PC1 TPR/FDR + cumulative PEV (QR-adjusted)
#
#   MC loop functions (PSOCK workers via mc_map() of mc_runtime.R, which the
#   demos source first):
#     run_mc_trials_trex_spca()       — MC loop for T-Rex SPCA
#     run_mc_trials_pca()             — MC loop for the ordinary PCA baseline
#     run_mc_trials_oracle_pca()      — MC loop for oracle thresholded PCA
#
#   Output:
#     save_and_print_spca_sweep_results() — generic sweep table (console + .txt)
#                                           + tidy .csv
#     save_and_print_spca_mc_results()    — SNR-sweep convenience wrapper
#
# Reproducibility pattern (mirrors cpp):
#   - Each trial's DATA is drawn from a deterministic seed
#     (base_seed_offset + mc*1000, disjoint 1000-wide bands per trial).
#   - The T-Rex selector runs with seed = -1 (hardware entropy per trial), so
#     the computer-generated dummies are independent across trials — required
#     for a valid Monte Carlo FDR estimate.
#
# Preprocessing note: X is CENTERED ONLY (covariance PCA, matching the legacy
# R reference and TRexSPCA's own internal convention). The column scales must
# NOT be normalized: the factor amplitude signal lives in the column variances,
# and z-scoring destroys it (measured at -10 dB: T-Rex SPCA FDR degrades from
# ~0.14 to ~0.5 — see cpp validation_trex_spca_06_handrolled_comparison).
# ==============================================================================

library(TRexSelector)


# ==============================================================================
# Data generator (sparse M-factor model)
# ==============================================================================

#' Generate a single dataset from the sparse factor model X = Z V^T + E.
#'
#' Z (n x M) has column-wise N(0, sigma_m^2) factors with sigma = {5, 3, 1}
#' (falling back to 1 for M > 3). V (p x M) has exactly p1 non-zeros (value 0.9)
#' per column, drawn without replacement from a shared candidate pool of size
#' overlap_pool_size — so different factors' active sets can overlap. E is
#' i.i.d. Gaussian noise scaled to match the target SNR in dB.
#'
#' @param n                  Number of samples.
#' @param p                  Number of features.
#' @param p1                 True active loadings per factor.
#' @param M                  Number of latent factors.
#' @param target_snr         Target signal-to-noise ratio in dB.
#' @param overlap_pool_size  Shared candidate-index pool size.
#' @param seed               RNG seed for this dataset.
#'
#' @return list(X, Z, V, E, actual_snr_db); X is raw (uncentered).
generate_sparse_factor_model <- function(n = 50L, p = 100L, p1 = 5L, M = 3L,
                                         target_snr = 0.0,
                                         overlap_pool_size = 30L,
                                         seed = 42L) {
  stopifnot(p1 <= p, n > 0L, p > 0L, p1 > 0L)
  set.seed(seed)

  # 1. True factors Z (n x M) with decreasing signal strength
  factor_stds <- c(5.0, 3.0, 1.0)
  Z <- matrix(0, nrow = n, ncol = M)
  for (m in seq_len(M)) {
    std_dev <- if (m <= length(factor_stds)) factor_stds[m] else 1.0
    Z[, m] <- rnorm(n, mean = 0, sd = std_dev)
  }

  # 2. Sparse loadings V (p x M): each factor draws p1 indices from the pool
  actual_pool <- min(overlap_pool_size, p)
  stopifnot(p1 <= actual_pool)
  V <- matrix(0, nrow = p, ncol = M)
  for (m in seq_len(M)) {
    active_idx <- sample.int(actual_pool, p1, replace = FALSE)
    V[active_idx, m] <- 0.9
  }

  # 3. Signal matrix S = Z V^T
  S <- Z %*% t(V)
  signal_var <- stats::var(as.vector(S))

  # 4. Gaussian noise E scaled to the target SNR
  # SNR_dB = 10 * log10(var_S / var_E)  =>  var_E = var_S / 10^(SNR_dB / 10)
  E_raw <- matrix(rnorm(n * p), nrow = n, ncol = p)
  raw_noise_var <- stats::var(as.vector(E_raw))
  target_noise_var <- signal_var / (10^(target_snr / 10))
  E <- E_raw * sqrt(target_noise_var / raw_noise_var)

  # 5. Assemble X
  X <- S + E
  final_noise_var <- stats::var(as.vector(E))
  actual_snr_db <- 10 * log10(signal_var / final_noise_var)

  list(X = X, Z = Z, V = V, E = E, actual_snr_db = actual_snr_db)
}


# ==============================================================================
# Preprocessing
# ==============================================================================

#' Center columns (mean subtraction only, NO scaling).
#'
#' Covariance-PCA footing shared by every method, matching the legacy R
#' reference pipeline and TRexSPCA's internal convention. Column scales carry
#' the factor amplitude signal and must not be normalized.
center_columns <- function(X) {
  sweep(X, 2L, colMeans(X), "-")
}


# ==============================================================================
# Single-trial evaluator
# ==============================================================================

#' Compute per-trial sparse PCA metrics from estimated loadings and scores.
#'
#' TPR / FDR are evaluated strictly on the FIRST principal component (PC1):
#' PCA's orthogonality constraint mixes the loading supports of PCs 2+ across
#' the true factors, making per-component support recovery metrics ambiguous
#' beyond the first PC. An empty true support yields TPR = 0.0 (legacy R
#' reference convention).
#'
#' The cumulative adjusted EV follows Zou/Hastie/Tibshirani: a QR
#' decomposition of Z_est removes the variance correlated components would
#' double-count, so the diagonal of R carries each component's incremental
#' contribution. It is reported under THREE normalizations (mirrors the cpp
#' evaluate_spca; see the suite README for the full account):
#'
#'   pev        — divided by the total variance of X (bounded by 1).
#'   pev_sig    — Definition 1 of the reference paper: divided by the
#'                Signal + Mixed EV OF THE DATA, i.e. the variance carried by
#'                the true active variables (union of the factor supports),
#'                ||X_A||_F^2 / (n-1) — FIXED per dataset and SHARED by all
#'                methods. Deliberately uncapped: a method that spends
#'                loadings on null variables accumulates variance the
#'                denominator does not contain and climbs past 1.
#'   pev_sigmix — the method's OWN Signal + Mixed part (raw EV minus the Null
#'                EV of its loadings, split by variable) over the same shared
#'                denominator; read for OrdPCA it is the paper's
#'                "Ordinary PCA (Sig + Mix)" reference curve.
#'
#' V_true may have FEWER columns than V_est/Z_est: sweeping the extracted PC
#' count past the number of true factors is how Fig. 3(a) of the reference
#' paper is built — the extra components are null by definition.
#'
#' @param X       Centered observation matrix (same X the methods saw).
#' @param V_est   Estimated sparse loading matrix (p x M).
#' @param Z_est   Estimated score matrix (n x M).
#' @param V_true  True loading matrix; non-zeros mark the true support.
#'
#' @return list(tpr, fdr, pev, pev_sig, pev_sigmix)
evaluate_spca <- function(X, V_est, Z_est, V_true) {
  n <- nrow(X)

  # 1. TPR / FDR strictly on PC1
  true_supp_pc1 <- which(V_true[, 1] != 0)
  est_supp_pc1  <- which(abs(V_est[, 1]) > 1e-12)

  tp <- length(intersect(true_supp_pc1, est_supp_pc1))
  tpr <- if (length(true_supp_pc1) == 0L) 0.0 else tp / length(true_supp_pc1)
  fdr <- if (length(est_supp_pc1) == 0L) 0.0 else
    (length(est_supp_pc1) - tp) / length(est_supp_pc1)

  # 2. Cumulative adjusted EV across ALL components (QR of Z_est)
  R <- qr.R(qr(Z_est))
  cum_ev <- sum(diag(R)^2) / (n - 1)

  # 2a. PEV, normalized by the TOTAL variance of X (bounded by 1)
  total_var <- sum(X^2) / (n - 1)
  pev <- if (total_var > 0) cum_ev / total_var else 0.0

  # 2b. PEV, normalized by the data's SIGNAL + MIXED EV (Definition 1).
  # Membership of A is per VARIABLE (row of V_true): active when it belongs
  # to the true support of at least one factor.
  row_active <- rowSums(V_true != 0) > 0
  sig_mixed_ev <- sum(X[, row_active, drop = FALSE]^2) / (n - 1)
  pev_sig <- if (sig_mixed_ev > 0) cum_ev / sig_mixed_ev else 0.0

  # 2c. The method's own Signal + Mixed share over the same denominator:
  # split V_est by variable into active and null rows and remove the Null EV
  # from its raw (unadjusted) EV.
  V_null <- V_est
  V_null[row_active, ] <- 0
  raw_ev  <- sum((X %*% V_est)^2) / (n - 1)
  null_ev <- sum((X %*% V_null)^2) / (n - 1)
  pev_sigmix <- if (sig_mixed_ev > 0) (raw_ev - null_ev) / sig_mixed_ev else 0.0

  list(tpr = tpr, fdr = fdr, pev = pev,
       pev_sig = pev_sig, pev_sigmix = pev_sigmix)
}


# ==============================================================================
# MC aggregation helper
# ==============================================================================

.aggregate_spca_mc <- function(label, res_mat) {
  res <- list(
    avg_tpr = mean(res_mat[, "tpr"]), avg_fdr = mean(res_mat[, "fdr"]),
    avg_pev = mean(res_mat[, "pev"]),
    avg_pev_sig = mean(res_mat[, "pev_sig"]),
    avg_pev_sigmix = mean(res_mat[, "pev_sigmix"]),
    sd_tpr = if (nrow(res_mat) > 1L) stats::sd(res_mat[, "tpr"]) else 0.0,
    sd_fdr = if (nrow(res_mat) > 1L) stats::sd(res_mat[, "fdr"]) else 0.0
  )
  cat(sprintf("  %s — done. TPR=%.3f  FDR=%.3f\n\n", label,
              res$avg_tpr, res$avg_fdr))
  res
}

.mc_map <- function(num_MC, one_trial, num_cores) {
  # mc index runs 0..num_MC-1 (matches the cpp seed bands)
  mc_map(0:(num_MC - 1L), one_trial, num_cores)
}


# ==============================================================================
# Parallel MC inner loops
# ==============================================================================

#' Run num_MC parallel T-Rex SPCA trials.
#'
#' Per trial: make_data(seed) returns raw data; the pipeline centers X once
#' (covariance PCA); this same X is shared by the selector and by
#' evaluate_spca(). The selector runs with seed = -1 (hardware entropy) so
#' dummies are independent across trials.
#'
#' @param num_MC            Number of Monte Carlo trials.
#' @param progress_label    Label printed in start/done lines.
#' @param make_data         Factory: seed -> data list (called per trial).
#' @param tFDR              Target FDR for T-Rex SPCA.
#' @param mode              "ActiveSet" or "Thresholded".
#' @param base_seed_offset  Base seed; trial mc uses base_seed_offset + mc*1000.
#' @param lambda_2          Ridge penalty (LARS units); < 0 = auto via CV,
#'                          0 = no ridge, > 0 = fixed.
#' @param en_solver         "TENET" or "TENET_AUG".
#' @param num_components    > 0 extracts that many PCs instead of the number
#'                          of true factors (Fig. 3(a) decoupling).
#' @param num_cores         Worker processes for mc_map().
#'
#' @return list(avg_tpr, avg_fdr, avg_pev, avg_pev_sig, avg_pev_sigmix,
#'   sd_tpr, sd_fdr)
run_mc_trials_trex_spca <- function(num_MC, progress_label, make_data,
                                    tFDR, mode, base_seed_offset,
                                    lambda_2 = -1.0, en_solver = "TENET",
                                    num_components = 0L, num_cores = 1L) {
  cat(sprintf("  %s — Running %d MC trials ...\n", progress_label, num_MC))

  one_trial <- function(mc) {
    dat <- make_data(base_seed_offset + mc * 1000L)
    X <- center_columns(dat$X)
    n_comp <- if (num_components > 0L) num_components else ncol(dat$V)

    # X + 0 forces a fresh copy: the selector maps X zero-copy via Eigen and
    # centers it internally; the local X must stay intact for evaluation.
    sel <- TRexSPCASelector$new(
      X + 0, M = n_comp, tFDR = tFDR,
      control = trex_spca_control(mode = mode, gvs_type = "EN",
                                  en_solver = en_solver,
                                  lambda_2 = lambda_2,
                                  lambda2_method = "CV_1SE_CCD"),
      seed = -1L, verbose = FALSE
    )
    sel$select()

    unlist(evaluate_spca(X, sel$V, sel$Z, dat$V))
  }

  res_mat <- do.call(rbind, .mc_map(num_MC, one_trial, num_cores))
  .aggregate_spca_mc(progress_label, res_mat)
}


#' Run num_MC parallel ordinary PCA trials.
#'
#' Ordinary PCA retains all p loadings per component (no sparsity), so
#' FDR ~ (p - p1) / p and TPR = 1 regardless of SNR; the metric of interest
#' is the cumulative PEV.
run_mc_trials_pca <- function(num_MC, progress_label, make_data,
                              base_seed_offset, num_components = 0L,
                              num_cores = 1L) {
  cat(sprintf("  %s — Running %d MC trials ...\n", progress_label, num_MC))

  one_trial <- function(mc) {
    dat <- make_data(base_seed_offset + mc * 1000L)
    X <- center_columns(dat$X)
    n_comp <- if (num_components > 0L) num_components else ncol(dat$V)

    V_ord <- svd(X, nu = 0, nv = n_comp)$v
    Z_ord <- X %*% V_ord

    unlist(evaluate_spca(X, V_ord, Z_ord, dat$V))
  }

  res_mat <- do.call(rbind, .mc_map(num_MC, one_trial, num_cores))
  .aggregate_spca_mc(progress_label, res_mat)
}


#' Run num_MC parallel oracle-thresholded PCA trials.
#'
#' Oracle PCA keeps exactly the top p1 entries (by absolute value) of each
#' ordinary-PCA loading vector — it knows the true support cardinality. This
#' bounds what any data-driven thresholding method can achieve.
run_mc_trials_oracle_pca <- function(num_MC, progress_label, make_data,
                                     p1, base_seed_offset,
                                     num_components = 0L, num_cores = 1L) {
  cat(sprintf("  %s — Running %d MC trials ...\n", progress_label, num_MC))

  one_trial <- function(mc) {
    dat <- make_data(base_seed_offset + mc * 1000L)
    X <- center_columns(dat$X)
    p <- ncol(X)
    n_comp <- if (num_components > 0L) num_components else ncol(dat$V)

    V_ord <- svd(X, nu = 0, nv = n_comp)$v

    # Keep top p1 entries per loading vector, then normalize
    V_oracle <- matrix(0, nrow = p, ncol = n_comp)
    for (comp in seq_len(n_comp)) {
      v_m <- V_ord[, comp]
      thresh <- sort(abs(v_m), decreasing = TRUE)[p1]
      keep <- abs(v_m) >= thresh
      V_oracle[keep, comp] <- v_m[keep]
      V_oracle[, comp] <- V_oracle[, comp] / sqrt(sum(V_oracle[, comp]^2))
    }
    Z_oracle <- X %*% V_oracle

    unlist(evaluate_spca(X, V_oracle, Z_oracle, dat$V))
  }

  res_mat <- do.call(rbind, .mc_map(num_MC, one_trial, num_cores))
  .aggregate_spca_mc(progress_label, res_mat)
}


# ==============================================================================
# Output
# ==============================================================================

#' Print MC-averaged sweep results as an aligned table (console + .txt) and
#' write a tidy long-format CSV (method, metric, <sweep_col>, value).
#'
#' Mirrors cpp/Python save_and_print_spca_sweep_results: the method name gets
#' its own line with the metric rows indented beneath it, and the axis header
#' row is printed at the per-sweep-column precision `sweep_precision`. When
#' `include_support_metrics` is FALSE the displayed table carries only the
#' cumulative-PEV rows; FDR and TPR stay in the tidy CSV either way — they
#' are the data record.
#'
#' @param out_dir         Output directory (created if missing).
#' @param file_stem       Output file base name (no folder, no extension).
#' @param num_MC          Number of Monte Carlo trials.
#' @param sweep_col       CSV column name of the sweep axis (e.g. "snr_db").
#' @param sweep_header    Table header of the sweep axis (e.g. "SNR(dB)").
#' @param sweep_precision Decimals for the axis values in the table header.
#' @param axis_values     Sweep grid used during the simulation.
#' @param method_names    Ordered vector of method names (keys into the maps).
#' @param fdr_map,tpr_map,pev_map,pev_sig_map,pev_sigmix_map Named lists:
#'   method -> numeric vector over the sweep grid.
#' @param include_support_metrics Show FDR/TPR rows in the table (default TRUE).
save_and_print_spca_sweep_results <- function(out_dir, file_stem, num_MC,  # nolint: object_length_linter. C++ name
                                              sweep_col, sweep_header,
                                              sweep_precision, axis_values,
                                              method_names, fdr_map, tpr_map,
                                              pev_map, pev_sig_map,
                                              pev_sigmix_map,
                                              include_support_metrics = TRUE) {
  dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

  indent_w <- 2L
  metric_w <- 23L
  col_w <- 10L

  lines <- character(0)
  acc <- environment()  # the report lines that add() appends to
  add <- function(s) acc$lines[[length(lines) + 1L]] <- s
  add("")
  add(strrep("=", 70L))
  add(sprintf("=== T-Rex SPCA Results (averaged over %d MC trials) ===", num_MC))
  add(strrep("=", 70L))
  add("")

  axis_fmt <- sprintf("%%%d.%df", col_w, sweep_precision)
  add(paste0(strrep(" ", indent_w), sprintf("%-*s", metric_w, sweep_header),
             paste(sprintf(axis_fmt, axis_values), collapse = "")))
  add(strrep("-", indent_w + metric_w + col_w * length(axis_values)))

  row <- function(metric, values) {
    paste0(strrep(" ", indent_w), sprintf("%-*s", metric_w, metric),
           paste(sprintf("%10.4f", values), collapse = ""))
  }

  for (name in method_names) {
    add("")
    add(name)                               # method name on its own line
    if (include_support_metrics) {
      add(row("FDR", fdr_map[[name]]))
      add(row("TPR", tpr_map[[name]]))
    }
    add(row("PEV (total var)", pev_map[[name]]))
    add(row("PEV (Definition 1)", pev_sig_map[[name]]))
    add(row("PEV (sig+mix part)", pev_sigmix_map[[name]]))
  }
  add("")

  text <- paste0(paste(lines, collapse = "\n"), "\n")
  cat(text)
  txt_path <- file.path(out_dir, paste0(file_stem, ".txt"))
  writeLines(text, txt_path, sep = "")

  # Tidy long-format CSV (method, metric, <sweep_col>, value); the axis is
  # written at fixed 6-decimal precision, matching the cpp/Python CSVs.
  metrics <- list(FDR = fdr_map, TPR = tpr_map, PEV = pev_map,
                  PEVsig = pev_sig_map, PEVsigmix = pev_sigmix_map)
  rows <- list()
  for (name in method_names) {
    for (i in seq_along(axis_values)) {
      df <- data.frame(
        method = name,
        metric = names(metrics),
        axis   = sprintf("%.6f", axis_values[i]),
        value  = sprintf("%.6f",
                         vapply(metrics, function(m) m[[name]][i], numeric(1))),
        stringsAsFactors = FALSE
      )
      names(df)[3L] <- sweep_col
      rows[[length(rows) + 1L]] <- df
    }
  }
  csv_path <- file.path(out_dir, paste0(file_stem, ".csv"))
  utils::write.csv(do.call(rbind, rows), csv_path,
                   row.names = FALSE, quote = FALSE)

  cat(sprintf("[Info] CSV results saved to:             %s\n", csv_path))
  cat(sprintf("[Info] Results successfully saved to: %s\n\n", txt_path))
}


#' SNR-sweep convenience wrapper around save_and_print_spca_sweep_results.
#'
#' Fixes the sweep axis to the decibel SNR grid this suite reports on by
#' default: CSV column "snr_db", table header "SNR(dB)", one decimal.
save_and_print_spca_mc_results <- function(out_dir, file_stem, num_MC,
                                           snr_values, method_names,
                                           fdr_map, tpr_map, pev_map,
                                           pev_sig_map, pev_sigmix_map) {
  save_and_print_spca_sweep_results(
    out_dir, file_stem, num_MC, "snr_db", "SNR(dB)", 1L, snr_values,
    method_names, fdr_map, tpr_map, pev_map, pev_sig_map, pev_sigmix_map
  )
}
