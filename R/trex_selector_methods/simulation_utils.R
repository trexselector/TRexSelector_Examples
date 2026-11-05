# ==============================================================================
# simulation_utils.R
# ==============================================================================
#
# Shared utility functions for TRex simulation demos.
# Sourced by each demo file.
#
# Contents:
# ------------------------------------------------------------------------------
#   plot_cormat()              — Plotly heatmap of a correlation matrix.
#   .print_table()             — Print a column-aligned MC results table.
#   .print_matrix()            — Print a labelled 2-D results matrix.
#   .print_table_multi()       — Print aligned table with multiple key columns.
#   .run_mc_hastie()           — Wrapper: dgp_hastie_snr             + trex+GVS (EN or IEN).
#   .run_mc_hastie_cut()       — Same, per clustering cut (corr_max or oracle groups).
#   .run_mc_scattered()        — Wrapper: dgp_scattered_grouped_snr  + trex+GVS (EN or IEN).
#   .run_mc_mixed_blocks()     — Wrapper: dgp_mixed_blocks_snr       + trex+GVS (EN or IEN).
#   .run_mc_neg_traps()        — Wrapper: dgp_neg_traps_snr          + trex+GVS (EN or IEN).
#   .run_mc_t3_equi_blocks()   — Wrapper: dgp_t3_equi_blocks_snr     + trex+GVS (EN or IEN).
#   .run_mc_ar1_blocks()       — Wrapper: dgp_ar1_blocks_snr         + trex+GVS (EN or IEN).
#   .run_mc_arma_blocks()      — Wrapper: dgp_arma_blocks_snr        + trex+GVS (EN or IEN).
#
# The .run_mc_*() runners use PSOCK workers via mc_map() and are resumable grid
# cells (cell_cached()), both from mc_runtime.R, which the demos source first.
# ==============================================================================


# ==============================================================================
# Plotting
# ==============================================================================

#' Plot a correlation matrix as a Plotly heatmap.
plot_cormat <- function(cor_matrix) {
  plotly::plot_ly(
    x          = colnames(cor_matrix),
    y          = rownames(cor_matrix),
    z          = cor_matrix,
    type       = "heatmap",
    colorscale = "RdBu",
    zmin       = -1,
    zmax       = 1
  )
}


# ==============================================================================
# Output helpers
# ==============================================================================

#' Print a column-aligned MC results table.
#'
#' @param results   List of named lists, each containing at least
#'                  `param_col`, `mean_FDP`, `mean_TPP`, `sd_FDP`, `sd_TPP`.
#' @param param_col Name of the swept parameter (string), e.g. "SNR" or "rho".
#' @param param_fmt Printf format for the parameter column, e.g. "%-6.1f".
.print_table <- function(results, param_col, param_fmt = "%-10.4f") {
  # Extract the width length from param_fmt (e.g. "%-6.1f" -> 6)
  fmt_width  <- as.integer(gsub("[^0-9]", "", strsplit(param_fmt, "\\.")[[1]][1]))

  # Ensure param_col header aligns perfectly with the param_fmt width
  header_fmt <- sprintf("%%-%ds", fmt_width)

  cat(sprintf(paste0(" ", header_fmt, " %-10s %-10s %-10s %-10s\n"),
              param_col, "mean_FDP", "mean_TPP", "sd_FDP", "sd_TPP"))
  cat(strrep("-", 58), "\n")
  for (r in results) {
    cat(sprintf(paste0(" ", param_fmt, " %-10.4f %-10.4f %-10.4f %-10.4f\n"),
                r[[param_col]], r$mean_FDP, r$mean_TPP, r$sd_FDP, r$sd_TPP))
  }
  cat("\n")
}


#' Print a column-aligned MC results table with multiple key columns.
#'
#' @param results    List of named lists, each containing entries for all key
#'   columns plus mean_FDP, mean_TPP, sd_FDP, sd_TPP.
#' @param key_cols   Character vector of key column names, e.g. c("Q", "p").
#' @param key_fmts   Named character vector of printf formats per key column,
#'   e.g. c(Q = "%-4d", p = "%-6d"). Must cover all entries in key_cols.
#' @param metric_fmt Printf format for the four metric columns. Default "%-10.4f".
.print_table_multi <- function(results, key_cols, key_fmts,
                               metric_fmt = "%-10.4f") {
  metric_cols <- c("mean_FDP", "mean_TPP", "sd_FDP", "sd_TPP")

  # Extract display width for each key column from its format string
  # e.g. "%-4d" -> 4, "%-6d" -> 6
  key_widths <- vapply(key_cols, function(k) {
    as.integer(gsub("[^0-9]", "", strsplit(key_fmts[[k]], "\\.")[[1]][1]))
  }, integer(1L))

  metric_width <- as.integer(gsub("[^0-9]", "", strsplit(metric_fmt, "\\.")[[1]][1]))

  # Build header
  hdr <- " "
  for (i in seq_along(key_cols)) {
    hdr <- paste0(hdr, sprintf(sprintf("%%-%ds ", key_widths[i]), key_cols[i]))
  }
  for (col in metric_cols) {
    hdr <- paste0(hdr, sprintf(sprintf("%%-%ds ", metric_width), col))
  }
  sep_w <- sum(key_widths + 1L) + (metric_width + 1L) * length(metric_cols) + 1L

  cat(trimws(hdr, "right"), "\n")
  cat(strrep("-", sep_w), "\n")

  for (r in results) {
    row <- " "
    for (i in seq_along(key_cols)) {
      row <- paste0(row, sprintf(key_fmts[[key_cols[i]]], r[[key_cols[i]]]), " ")
    }
    for (col in metric_cols) {
      row <- paste0(row, sprintf(metric_fmt, r[[col]]), " ")
    }
    cat(trimws(row, "right"), "\n")
  }
  cat("\n")
}


#' Print a labelled 2-D results matrix.
#'
#' @param mat   Numeric matrix with named rows and columns.
#' @param label Header label printed above the matrix.
.print_matrix <- function(mat, label) {
  cat(sprintf("\n %s:\n", label))
  col_labels <- colnames(mat)
  row_labels <- rownames(mat)

  header <- sprintf(" %-10s", "")
  for (cl in col_labels) {
    header <- paste0(header, sprintf(" %-8s", cl))
  }

  cat(header, "\n")
  cat(strrep("-", nchar(header) + 2L), "\n")

  for (i in seq_len(nrow(mat))) {
    row_str <- sprintf(" %-10s", row_labels[i])
    for (j in seq_len(ncol(mat))) {
      row_str <- paste0(row_str, sprintf(" %-8.3f", mat[i, j]))
    }
    cat(row_str, "\n")
  }
  cat("\n")
}


# ==============================================================================
# R6 API adapters (legacy TRexSelector -> TRexSelector)
# ==============================================================================

#' Map legacy lowercase HC linkage names to the R6 capitalized names.
#'
#' The R6 API also accepts lowercase, but capitalized names are used for clarity.
#' Unknown / already-capitalized values are passed through unchanged.
cap_hc <- function(hc_dist) {
  switch(hc_dist,
         single   = "Single",
         complete = "Complete",
         average  = "Average",
         wpgma    = "WPGMA",
         hc_dist)
}


# (The DA-TRex MC engine now lives in ad/trex_da/da_sim_common.R, sourced by the
# trex_da demos directly — this file keeps only the GVS wrappers and the
# generic output helpers.)


# ==============================================================================
# GVS selector adapter
# ==============================================================================

#' Build and run a TRexGVSSelector with the given group-selection knobs.
#'
#' Centralizes the selector construction shared by every `.run_mc_*` GVS
#' runner, so that solver / linkage / lambda_2 options are wired consistently.
#'
#' @param X,y            Design matrix and response vector.
#' @param tFDR,K         Target FDR and number of random experiments.
#' @param GVS_type       GVS variant: "EN" or "IEN".
#' @param corr_max       HAC clustering correlation threshold.
#' @param hc_dist        HAC linkage (legacy lowercase; mapped via `cap_hc()`).
#' @param en_solver      EN dummy solver: "TENET" or "TENET_AUG" (EN only;
#'                       ignored by IEN).
#' @param lambda2_method lambda_2 selection rule, e.g. "CV_1SE_CCD" / "CV_1SE_SVD".
#' @param groups         Optional oracle group labels: a length-p integer vector
#'                       of contiguous 1-based cluster IDs (1..M). NULL (default)
#'                       means groups are discovered from the data via HAC.
#' @param seed           Selector seed; the MC loops pass the trial seed
#'                       (mirrors the cpp/Python convention selector seed =
#'                       cv_seed = trial seed). -1 = hardware entropy.
#' @param cv_seed        CV fold-permutation seed; -1 derives it from `seed`.
#' @param cv_dfmax_ratio Support cap of the CV lambda1 path as a share of n_train;
#'                       NULL keeps the trex_gvs_control() default (0.8).
#'
#' @return The TRexGVSSelector object after `$select()` has been called.
.gvs_select <- function(X, y, tFDR, K, GVS_type, corr_max,
                        hc_dist        = "single",
                        en_solver      = "TENET",
                        lambda2_method = "CV_1SE_CCD",
                        groups         = NULL,
                        seed           = -1L,
                        cv_seed        = -1L,
                        cv_dfmax_ratio = NULL) {
  gvs_args <- list(
    gvs_type       = GVS_type,
    corr_max       = corr_max,
    hc_linkage     = cap_hc(hc_dist),
    en_solver      = en_solver,
    lambda2_method = lambda2_method,
    cv_seed        = cv_seed
  )
  if (!is.null(groups)) gvs_args$groups <- groups
  if (!is.null(cv_dfmax_ratio)) gvs_args$cv_dfmax_ratio <- cv_dfmax_ratio

  # The TIENET_AUG-labeled rows run the augmented IEN solver, matching the
  # cpp/Python reference; with the plain "TLARS" base solver the current
  # library would silently route the IEN path to the native TIENET instead.
  base_solver <- if (identical(GVS_type, "IEN")) "TIENET_AUG" else "TLARS"

  # Every field of the cpp make_gvs_trex_control() is set explicitly, so no
  # R default (AUTO stagnation off for TLARS) can diverge from the reference.
  ctrl <- TRexSelector::trex_control(
    solver = base_solver, K = K, max_dummy_multiplier = 10,
    use_max_T_stop = TRUE, lloop_strategy = "HCONCAT",
    tloop_stagnation_stop = TRUE, tloop_max_stagnant_steps = 5,
    use_memory_mapping = FALSE
  )
  sel <- TRexSelector::TRexGVSSelector$new(
    X, y, tFDR = tFDR, seed = as.integer(seed), verbose = FALSE,
    gvs_control = do.call(TRexSelector::trex_gvs_control, gvs_args),
    control     = ctrl
  )
  sel$select()
  sel
}



# ==============================================================================
# GVS wrappers
# ==============================================================================

#' MC runner for the Hastie (Zou & Hastie 2005) DGP with trex+GVS.
#'
#' Each trial generates data via `dgp_hastie_snr(n, p, snr, sd_x, seed)` and
#' applies `TRexSelector::trex()` with `method = "trex+GVS"`.
#'
#' @param n,p         Data dimensions.
#' @param snr         Target signal-to-noise ratio.
#' @param sd_x        Within-group noise sd (rho = 1/(1+sd_x^2)).
#' @param tFDR        Target FDR level.
#' @param K           Number of random experiments per selector run.
#' @param num_MC      Number of Monte Carlo trials.
#' @param seed        Base RNG seed; trial k uses seed + k - 1.
#' @param GVS_type    GVS variant: "EN" or "IEN".
#' @param corr_max    HAC clustering threshold passed to trex.
#' @param hc_dist     HAC linkage type (default "single").
#' @param cv_dfmax_ratio Support cap of the CV lambda1 path (NULL: library default).
#' @param n_cores     Number of parallel cores (default num_cores).
#'
#' @return Named list: mean_FDP, mean_TPP, sd_FDP, sd_TPP.
.run_mc_hastie <- function(
  n,
  p,
  snr,
  sd_x,
  tFDR,
  K,
  num_MC,
  seed,
  GVS_type,
  corr_max,
  hc_dist  = "single",
  en_solver      = "TENET",
  lambda2_method = "CV_1SE_CCD",
  cv_dfmax_ratio = NULL,
  n_cores  = num_cores
) {
  one_trial <- function(mc) {
    trial_seed <- seed + mc - 1L
    dat <- dgp_hastie_snr(n = n, p = p, snr = snr, sd_x = sd_x,
                          seed = trial_seed)
    sel <- .gvs_select(dat$X, dat$y, tFDR, K, GVS_type, corr_max,
                       hc_dist, en_solver, lambda2_method,
                       seed = trial_seed, cv_seed = trial_seed,
                       cv_dfmax_ratio = cv_dfmax_ratio)
    ts <- which(dat$beta != 0)
    c(FDP = TRexSelector::trex_fdp(sel$selected_indices, ts),
      TPP = TRexSelector::trex_tpp(sel$selected_indices, ts))
  }

  # One resumable grid cell: restored from the cell cache after a restart. The
  # arguments (all but the worker count) identify the cell.
  args <- mget(setdiff(names(formals(sys.function())), "n_cores"), envir = environment())
  cell_cached(paste0(".run_mc_hastie|", cell_digest(args)), function() {
    results <- mc_map(seq_len(num_MC), one_trial, n_cores)

    fdp_vec <- sapply(results, `[[`, "FDP")
    tpp_vec <- sapply(results, `[[`, "TPP")

    list(
      mean_FDP = mean(fdp_vec),
      mean_TPP = mean(tpp_vec),
      sd_FDP   = sd(fdp_vec),
      sd_TPP   = sd(tpp_vec)
    )
  })
}


#' MC runner for the clustering-cut comparison on the Hastie DGP.
#'
#' Same trials as `.run_mc_hastie()` (single linkage), with the HAC cut at
#' `corr_max` or, for `oracle = TRUE`, the DGP's prior_groups. Group rates come
#' from `.gvs_group_rates()` in ad/trex_gvs/trex_gvs_sim_common.R.
#'
#' @inheritParams .run_mc_hastie
#' @param oracle Pass the DGP's prior_groups (the cut is then unused).
#'
#' @return Named list: mean / sd of FDP, TPP, grp_hit, grp_full, grp_FDP and
#'   n_groups (clusters with >= 2 members).
.run_mc_hastie_cut <- function(
  n,
  p,
  snr,
  sd_x,
  tFDR,
  K,
  num_MC,
  seed,
  GVS_type,
  corr_max,
  en_solver      = "TENET",
  lambda2_method = "CV_1SE_CCD",
  oracle   = FALSE,
  cv_dfmax_ratio = NULL,
  n_cores  = num_cores
) {
  one_trial <- function(mc) {
    trial_seed <- seed + mc - 1L
    dat <- dgp_hastie_snr(n = n, p = p, snr = snr, sd_x = sd_x,
                          seed = trial_seed)
    sel <- .gvs_select(dat$X, dat$y, tFDR, K, GVS_type, corr_max,
                       "single", en_solver, lambda2_method,
                       groups = if (oracle) dat$prior_groups else NULL,
                       seed = trial_seed, cv_seed = trial_seed,
                       cv_dfmax_ratio = cv_dfmax_ratio)
    si <- sel$selected_indices
    ts <- which(dat$beta != 0)
    c(FDP = TRexSelector::trex_fdp(si, ts),
      TPP = TRexSelector::trex_tpp(si, ts),
      .gvs_group_rates(si, dat$prior_groups, ts),
      n_groups = sum(table(sel$groups) >= 2L))
  }

  args <- mget(setdiff(names(formals(sys.function())), "n_cores"), envir = environment())
  cell_cached(paste0(".run_mc_hastie_cut|", cell_digest(args)), function() {
    res <- do.call(rbind, mc_map(seq_len(num_MC), one_trial, n_cores))
    list(
      mean_FDP = mean(res[, "FDP"]), sd_FDP = sd(res[, "FDP"]),
      mean_TPP = mean(res[, "TPP"]), sd_TPP = sd(res[, "TPP"]),
      grp_hit  = mean(res[, "hit"]),  sd_grp_hit  = sd(res[, "hit"]),
      grp_full = mean(res[, "full"]), sd_grp_full = sd(res[, "full"]),
      grp_FDP  = mean(res[, "fdp"]),  sd_grp_FDP  = sd(res[, "fdp"]),
      n_groups = mean(res[, "n_groups"]), sd_n_groups = sd(res[, "n_groups"])
    )
  })
}


#' MC runner for the scattered-grouped DGP with trex+GVS.
#'
#' Each trial generates data via `dgp_scattered_grouped_snr()` and applies
#' `TRexSelector::trex()` with `method = "trex+GVS"`.
#' Group indices are re-randomised each trial (controlled by trial_seed).
#'
#' @param n,p         Data dimensions.
#' @param snr         Target signal-to-noise ratio.
#' @param sd_x        Within-group noise sd (rho = 1/(1+sd_x^2)).
#' @param n_groups    Number of active groups. Default 3.
#' @param group_size  Variables per group. Default 50.
#' @param tFDR        Target FDR level.
#' @param K           Number of random experiments per selector run.
#' @param num_MC      Number of Monte Carlo trials.
#' @param seed        Base RNG seed; trial k uses seed + k - 1.
#' @param GVS_type    GVS variant: "EN" or "IEN".
#' @param corr_max    HAC clustering threshold passed to trex.
#' @param hc_dist     HAC linkage type (default "single").
#' @param n_cores     Number of parallel cores (default num_cores).
#'
#' @return Named list: mean_FDP, mean_TPP, sd_FDP, sd_TPP.
.run_mc_scattered <- function(
  n,
  p,
  snr,
  sd_x,
  n_groups   = 3L,
  group_size = 50L,
  tFDR,
  K,
  num_MC,
  seed,
  GVS_type,
  corr_max,
  hc_dist  = "single",
  en_solver      = "TENET",
  lambda2_method = "CV_1SE_CCD",
  cv_dfmax_ratio = NULL,
  n_cores  = num_cores
) {
  one_trial <- function(mc) {
    trial_seed <- seed + mc - 1L
    dat <- dgp_scattered_grouped_snr(
      n          = n,
      p          = p,
      snr        = snr,
      sd_x       = sd_x,
      n_groups   = n_groups,
      group_size = group_size,
      seed       = trial_seed
    )
    sel <- .gvs_select(dat$X, dat$y, tFDR, K, GVS_type, corr_max,
                       hc_dist, en_solver, lambda2_method,
                       seed = trial_seed, cv_seed = trial_seed,
                       cv_dfmax_ratio = cv_dfmax_ratio)
    ts <- which(dat$beta != 0)
    c(FDP = TRexSelector::trex_fdp(sel$selected_indices, ts),
      TPP = TRexSelector::trex_tpp(sel$selected_indices, ts))
  }

  # One resumable grid cell: restored from the cell cache after a restart. The
  # arguments (all but the worker count) identify the cell.
  args <- mget(setdiff(names(formals(sys.function())), "n_cores"), envir = environment())
  cell_cached(paste0(".run_mc_scattered|", cell_digest(args)), function() {
    results <- mc_map(seq_len(num_MC), one_trial, n_cores)

    fdp_vec <- sapply(results, `[[`, "FDP")
    tpp_vec <- sapply(results, `[[`, "TPP")

    list(
      mean_FDP = mean(fdp_vec),
      mean_TPP = mean(tpp_vec),
      sd_FDP   = sd(fdp_vec),
      sd_TPP   = sd(tpp_vec)
    )
  })
}


# ==============================================================================
# .run_mc_mixed_blocks
# ==============================================================================

#' MC runner for the mixed-blocks DGP with trex+GVS.
#'
#' 3 active blocks (20, 50, 80) + 1 inactive block (65), randomly ordered.
#' Active set s = 150.
#'
#' @inheritParams .run_mc_hastie
#' @return Named list: mean_FDP, mean_TPP, sd_FDP, sd_TPP.
.run_mc_mixed_blocks <- function(
  n,
  p,
  snr,
  sd_x,
  tFDR,
  K,
  num_MC,
  seed,
  GVS_type,
  corr_max,
  hc_dist  = "single",
  en_solver      = "TENET",
  lambda2_method = "CV_1SE_CCD",
  cv_dfmax_ratio = NULL,
  n_cores  = num_cores
) {
  one_trial <- function(mc) {
    trial_seed <- seed + mc - 1L
    dat <- dgp_mixed_blocks_snr(n = n, p = p, snr = snr, sd_x = sd_x,
                                seed = trial_seed)
    sel <- .gvs_select(dat$X, dat$y, tFDR, K, GVS_type, corr_max,
                       hc_dist, en_solver, lambda2_method,
                       seed = trial_seed, cv_seed = trial_seed,
                       cv_dfmax_ratio = cv_dfmax_ratio)
    ts <- which(dat$beta != 0)
    c(FDP = TRexSelector::trex_fdp(sel$selected_indices, ts),
      TPP = TRexSelector::trex_tpp(sel$selected_indices, ts))
  }

  # One resumable grid cell: restored from the cell cache after a restart. The
  # arguments (all but the worker count) identify the cell.
  args <- mget(setdiff(names(formals(sys.function())), "n_cores"), envir = environment())
  cell_cached(paste0(".run_mc_mixed_blocks|", cell_digest(args)), function() {
    results <- mc_map(seq_len(num_MC), one_trial, n_cores)

    fdp_vec <- sapply(results, `[[`, "FDP")
    tpp_vec <- sapply(results, `[[`, "TPP")

    list(
      mean_FDP = mean(fdp_vec),
      mean_TPP = mean(tpp_vec),
      sd_FDP   = sd(fdp_vec),
      sd_TPP   = sd(tpp_vec)
    )
  })
}


# ==============================================================================
# .run_mc_neg_traps
# ==============================================================================

#' MC runner for the negative-traps DGP with trex+GVS.
#'
#' Active group 1-100 (+Z1/-Z1, beta = +3/-3); two inactive traps (100 + 60).
#' Active set s = 100.
#'
#' @inheritParams .run_mc_hastie
#' @return Named list: mean_FDP, mean_TPP, sd_FDP, sd_TPP.
.run_mc_neg_traps <- function(
  n,
  p,
  snr,
  sd_x,
  tFDR,
  K,
  num_MC,
  seed,
  GVS_type,
  corr_max,
  hc_dist  = "single",
  en_solver      = "TENET",
  lambda2_method = "CV_1SE_CCD",
  cv_dfmax_ratio = NULL,
  n_cores  = num_cores
) {
  one_trial <- function(mc) {
    trial_seed <- seed + mc - 1L
    dat <- dgp_neg_traps_snr(n = n, p = p, snr = snr, sd_x = sd_x,
                             seed = trial_seed)
    sel <- .gvs_select(dat$X, dat$y, tFDR, K, GVS_type, corr_max,
                       hc_dist, en_solver, lambda2_method,
                       seed = trial_seed, cv_seed = trial_seed,
                       cv_dfmax_ratio = cv_dfmax_ratio)
    ts <- which(dat$beta != 0)
    c(FDP = TRexSelector::trex_fdp(sel$selected_indices, ts),
      TPP = TRexSelector::trex_tpp(sel$selected_indices, ts))
  }

  # One resumable grid cell: restored from the cell cache after a restart. The
  # arguments (all but the worker count) identify the cell.
  args <- mget(setdiff(names(formals(sys.function())), "n_cores"), envir = environment())
  cell_cached(paste0(".run_mc_neg_traps|", cell_digest(args)), function() {
    results <- mc_map(seq_len(num_MC), one_trial, n_cores)

    fdp_vec <- sapply(results, `[[`, "FDP")
    tpp_vec <- sapply(results, `[[`, "TPP")

    list(
      mean_FDP = mean(fdp_vec),
      mean_TPP = mean(tpp_vec),
      sd_FDP   = sd(fdp_vec),
      sd_TPP   = sd(tpp_vec)
    )
  })
}


# ==============================================================================
# .run_mc_t3_equi_blocks
# ==============================================================================

#' MC runner for the heavy-tailed equi-correlated blocks DGP with trex+GVS.
#'
#' 4 equi-correlated blocks (sizes 20, 50, 80, 65): 3 active (beta=3), 1 inactive
#' trap; within-block equi-correlation = rho, but with Student-t(df) noise.
#' Active set s = 150.
#'
#' @inheritParams .run_mc_hastie
#' @param rho Within-block equi-correlation.
#' @param df Degrees of freedom for the t distribution. Default 3.
#' @return Named list: mean_FDP, mean_TPP, sd_FDP, sd_TPP.
.run_mc_t3_equi_blocks <- function(
  n,
  p,
  snr,
  rho,
  tFDR,
  K,
  num_MC,
  seed,
  GVS_type,
  corr_max,
  hc_dist  = "single",
  df       = 3L,
  en_solver      = "TENET",
  lambda2_method = "CV_1SE_CCD",
  cv_dfmax_ratio = NULL,
  n_cores  = num_cores
) {
  one_trial <- function(mc) {
    trial_seed <- seed + mc - 1L
    dat <- dgp_t3_equi_blocks_snr(n = n, p = p, snr = snr, rho = rho,
                                  df = df, seed = trial_seed)
    sel <- .gvs_select(dat$X, dat$y, tFDR, K, GVS_type, corr_max,
                       hc_dist, en_solver, lambda2_method,
                       seed = trial_seed, cv_seed = trial_seed,
                       cv_dfmax_ratio = cv_dfmax_ratio)
    ts <- which(dat$beta != 0)
    c(FDP = TRexSelector::trex_fdp(sel$selected_indices, ts),
      TPP = TRexSelector::trex_tpp(sel$selected_indices, ts))
  }

  # One resumable grid cell: restored from the cell cache after a restart. The
  # arguments (all but the worker count) identify the cell.
  args <- mget(setdiff(names(formals(sys.function())), "n_cores"), envir = environment())
  cell_cached(paste0(".run_mc_t3_equi_blocks|", cell_digest(args)), function() {
    results <- mc_map(seq_len(num_MC), one_trial, n_cores)

    fdp_vec <- sapply(results, `[[`, "FDP")
    tpp_vec <- sapply(results, `[[`, "TPP")

    list(
      mean_FDP = mean(fdp_vec),
      mean_TPP = mean(tpp_vec),
      sd_FDP   = sd(fdp_vec),
      sd_TPP   = sd(tpp_vec)
    )
  })
}


# ==============================================================================
# .run_mc_ar1_blocks
# ==============================================================================

#' MC runner for the block-structured AR(1) DGP with trex+GVS.
#'
#' 4 blocks (sizes 20, 50, 80, 65): 3 active (beta=3), 1 inactive trap.
#' Within-block AR(1) correlation: Cor(j,k) = rho^|j-k|. Active set s = 150.
#'
#' @inheritParams .run_mc_hastie
#' @param rho Within-block AR(1) correlation.
#' @return Named list: mean_FDP, mean_TPP, sd_FDP, sd_TPP.
.run_mc_ar1_blocks <- function(
  n,
  p,
  snr,
  rho,
  tFDR,
  K,
  num_MC,
  seed,
  GVS_type,
  corr_max,
  hc_dist  = "single",
  en_solver      = "TENET",
  lambda2_method = "CV_1SE_CCD",
  cv_dfmax_ratio = NULL,
  n_cores  = num_cores
) {
  one_trial <- function(mc) {
    trial_seed <- seed + mc - 1L
    dat <- dgp_ar1_blocks_snr(n = n, p = p, snr = snr, rho = rho,
                              seed = trial_seed)
    sel <- .gvs_select(dat$X, dat$y, tFDR, K, GVS_type, corr_max,
                       hc_dist, en_solver, lambda2_method,
                       seed = trial_seed, cv_seed = trial_seed,
                       cv_dfmax_ratio = cv_dfmax_ratio)
    ts <- which(dat$beta != 0)
    c(FDP = TRexSelector::trex_fdp(sel$selected_indices, ts),
      TPP = TRexSelector::trex_tpp(sel$selected_indices, ts))
  }

  # One resumable grid cell: restored from the cell cache after a restart. The
  # arguments (all but the worker count) identify the cell.
  args <- mget(setdiff(names(formals(sys.function())), "n_cores"), envir = environment())
  cell_cached(paste0(".run_mc_ar1_blocks|", cell_digest(args)), function() {
    results <- mc_map(seq_len(num_MC), one_trial, n_cores)

    fdp_vec <- sapply(results, `[[`, "FDP")
    tpp_vec <- sapply(results, `[[`, "TPP")

    list(
      mean_FDP = mean(fdp_vec),
      mean_TPP = mean(tpp_vec),
      sd_FDP   = sd(fdp_vec),
      sd_TPP   = sd(tpp_vec)
    )
  })
}


# ==============================================================================
# .run_mc_arma_blocks
# ==============================================================================

#' MC runner for the ARMA-structured blocks DGP with trex+GVS.
#'
#' 4 blocks (AR1 / MA3 / ARMA21 / AR1-trap): 3 active (beta=3), 1 inactive.
#' Columns column-standardized. Active set s = 150.
#'
#' @param n,p       Dataset dimensions.
#' @param snr       Signal-to-noise ratio.
#' @param ar_coef   AR coefficient shared by AR(1) and ARMA(2,1) blocks.
#' @param tFDR,K    TRex parameters.
#' @param num_MC,seed Monte Carlo parameters.
#' @param GVS_type  "EN" or "IEN".
#' @param corr_max  Hierarchical clustering threshold.
#' @param hc_dist   HC linkage. Default "single".
#' @param n_cores   Parallel worker count.
#' @return Named list: mean_FDP, mean_TPP, sd_FDP, sd_TPP.
.run_mc_arma_blocks <- function(
  n,
  p,
  snr,
  ar_coef,
  tFDR,
  K,
  num_MC,
  seed,
  GVS_type,
  corr_max,
  hc_dist  = "single",
  en_solver      = "TENET",
  lambda2_method = "CV_1SE_CCD",
  cv_dfmax_ratio = NULL,
  n_cores  = num_cores
) {
  one_trial <- function(mc) {
    trial_seed <- seed + mc - 1L
    dat <- dgp_arma_blocks_snr(n = n, p = p, snr = snr, ar_coef = ar_coef,
                               seed = trial_seed)
    sel <- .gvs_select(dat$X, dat$y, tFDR, K, GVS_type, corr_max,
                       hc_dist, en_solver, lambda2_method,
                       seed = trial_seed, cv_seed = trial_seed,
                       cv_dfmax_ratio = cv_dfmax_ratio)
    ts <- which(dat$beta != 0)
    c(FDP = TRexSelector::trex_fdp(sel$selected_indices, ts),
      TPP = TRexSelector::trex_tpp(sel$selected_indices, ts))
  }

  # One resumable grid cell: restored from the cell cache after a restart. The
  # arguments (all but the worker count) identify the cell.
  args <- mget(setdiff(names(formals(sys.function())), "n_cores"), envir = environment())
  cell_cached(paste0(".run_mc_arma_blocks|", cell_digest(args)), function() {
    results <- mc_map(seq_len(num_MC), one_trial, n_cores)

    fdp_vec <- sapply(results, `[[`, "FDP")
    tpp_vec <- sapply(results, `[[`, "TPP")

    list(
      mean_FDP = mean(fdp_vec),
      mean_TPP = mean(tpp_vec),
      sd_FDP   = sd(fdp_vec),
      sd_TPP   = sd(tpp_vec)
    )
  })
}
