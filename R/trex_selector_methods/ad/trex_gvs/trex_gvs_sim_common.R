# ==============================================================================
# trex_gvs_sim_common.R
# ==============================================================================
#
# Shared infrastructure for the T-Rex+GVS Monte Carlo demos. Mirrors the Python
# Python/trex_selector_methods/ad/trex_gvs/gvs_sim_common.py (which in turn mirrors
# cpp/trex_selector_methods/ad/trex_gvs/trex_gvs_simulation_utils.hpp):
#
#   fmt_num()                    — trailing-zero-trimmed number formatting.
#   print_section_header()       — framed console section header.
#   method_selected() / num_mc() /
#   out_stem() / skipped_point() — method selection TREX_SOLVERS (rows of
#                                  unselected methods are NaN, tables get a
#                                  suffix) and the smoke-run trial count
#                                  TREX_NUM_MC.
#   run_gvs_2d_sweep()           — nested (SNR x rho) grid MC sweep around a
#                                  demo-supplied `.run_mc_*()` point function.
#   render_mc_1d_method_table()  — per-method 1-D table (mean/sd FDP + TPP).
#   render_mc_matrix() /
#   print_mc_matrix()            — 2-D matrix table with "row \ col" corner axis.
#   save_mc_2d_tables()          — .txt (all six matrices) + tidy .csv writer.
#   save_mc_1d_method_tables()   — .txt + .csv writer for the demo-01 method
#                                  comparisons (lambda2_method / hc_linkage).
#   noise_floor_corr_max() /
#   .gvs_group_rates() /
#   save_mc_cut_tables()         — demo-01 clustering-cut comparison (fixed
#                                  cut, Bonferroni noise floor, oracle groups).
#   print_block_grid_table()     — block-benchmark console table + .txt/.csv
#                                  writer for demo 08.
#
# Method labels follow the cpp/Python suites: TENET (EN), TENET_AUG (EN,
# augmented solver), TIENET_AUG (IEN). Support / group indices are 1-based
# (R convention); the Python counterpart is 0-based.
#
# The MC engines themselves live in ../simulation_utils.R (`.run_mc_hastie()`
# etc., each returning mean_FDP / mean_TPP / sd_FDP / sd_TPP); this file only
# adds the sweep driver and the result writers on top of them.
# ==============================================================================


# ==============================================================================
# Number formatting (mirrors Python fmt_num / cpp fmt_num)
# ==============================================================================

#' Fixed-point with at most `max_decimals` decimals, trailing zeros (and a
#' trailing '.') trimmed. E.g. fmt_num(0.05) -> "0.05", fmt_num(2.0) -> "2".
#' Vectorized over `v`.
fmt_num <- function(v, max_decimals = 4L) {
  vapply(as.numeric(v), function(x) {
    s <- sprintf(paste0("%.", max_decimals, "f"), x)
    if (grepl(".", s, fixed = TRUE)) {
      s <- sub("0+$", "", s)
      s <- sub("\\.$", "", s)
    }
    s
  }, character(1L))
}


#' Framed section header (mirrors Python print_section_header).
print_section_header <- function(title) {
  cat(strrep("=", 78), "\n", sep = "")
  for (line in strsplit(as.character(title), "\n", fixed = TRUE)[[1L]]) {
    cat(" ", line, "\n", sep = "")
  }
  cat(strrep("=", 78), "\n\n", sep = "")
}


# ==============================================================================
# Method selection (TREX_SOLVERS; mirrors cpp trex_gvs_simulation_utils.hpp)
# ==============================================================================

#' TRUE when the method of a run label is selected by TREX_SOLVERS
#' (solver_selected() in mc_runtime.R). The method is the label part before
#' '+' or ' ', after an optional "[i/N] " progress counter
#' ("TIENET_AUG+CV_1SE_CCD" -> "TIENET_AUG"). Without a selection every
#' method runs.
method_selected <- function(label) {
  lbl <- sub("^\\[[0-9]+/[0-9]+\\] ", "", label)
  solver_selected(sub("[+ ].*$", "", lbl))
}


#' Monte Carlo trials of a demo: TREX_NUM_MC when a positive integer, else
#' `value`. For smoke runs; TREX_NUM_MC enters the cell-cache fingerprint, so
#' a smoke cache is never reused by a full run.
num_mc <- function(value) {
  n <- suppressWarnings(as.integer(Sys.getenv("TREX_NUM_MC", "")))
  if (is.na(n) || n < 1L) value else n
}


#' Output file stem plus the selection suffix, e.g. "gvs_Hastie_2d_tienet_aug".
out_stem <- function(stem) paste0(stem, solver_file_suffix())


#' Grid point of a skipped (unselected) method: all metrics NaN.
skipped_point <- function() {
  list(mean_FDP = NaN, mean_TPP = NaN, sd_FDP = NaN, sd_TPP = NaN)
}


# ==============================================================================
# 2-D (SNR x rho) sweep (mirrors Python run_gvs_2d_sweep)
# ==============================================================================

#' Run a nested SNR x rho grid MC sweep for one method variant.
#'
#' @param point_fun Function(snr, col_val) running `num_MC` trials at one grid
#'   cell (a demo-local closure around the matching `.run_mc_*()` wrapper);
#'   must return a list with mean_FDP, mean_TPP, sd_FDP, sd_TPP.
#' @param snr_grid,col_grid Row (SNR) and column (rho / ar_coef) grids.
#' @param label  Method label used in progress lines (e.g. "TENET").
#' @param num_MC Number of MC trials per cell (progress lines only).
#'
#' @return list(mean_TPP, mean_FDP, sd_TPP, sd_FDP), each an
#'   `length(snr_grid) x length(col_grid)` matrix; all NaN for a method that
#'   TREX_SOLVERS does not select (nothing is computed then).
run_gvs_2d_sweep <- function(point_fun, snr_grid, col_grid, label, num_MC) {
  n_snr <- length(snr_grid)
  n_col <- length(col_grid)

  if (!method_selected(label)) {
    cat(sprintf("\n  Method: %s  (skipped, not in TREX_SOLVERS)\n\n", label))
    nan_mat <- matrix(NaN, n_snr, n_col)
    return(list(mean_TPP = nan_mat, mean_FDP = nan_mat,
                sd_TPP = nan_mat, sd_FDP = nan_mat))
  }

  mean_TPP <- matrix(NA_real_, n_snr, n_col)
  mean_FDP <- mean_TPP
  sd_TPP   <- mean_TPP
  sd_FDP   <- mean_TPP

  cat(sprintf("\n  Method: %s\n\n", label))

  for (i in seq_len(n_snr)) {
    for (j in seq_len(n_col)) {
      lbl <- sprintf("[%d/%d] %s  SNR=%.2f  rho=%.2f",
                     (i - 1L) * n_col + j, n_snr * n_col, label,
                     snr_grid[i], col_grid[j])
      cat(sprintf("  %s — Running %d MC trials ...\n", lbl, num_MC))
      r <- point_fun(snr_grid[i], col_grid[j])
      mean_TPP[i, j] <- r$mean_TPP
      mean_FDP[i, j] <- r$mean_FDP
      sd_TPP[i, j]   <- r$sd_TPP
      sd_FDP[i, j]   <- r$sd_FDP
      cat(sprintf("  %s — done. TPP=%.3f  FDP=%.3f\n\n",
                  lbl, r$mean_TPP, r$mean_FDP))
    }
  }

  list(mean_TPP = mean_TPP, mean_FDP = mean_FDP,
       sd_TPP = sd_TPP, sd_FDP = sd_FDP)
}


# ==============================================================================
# MC 1-D method table (mirrors Python print_mc_1d_method_table)
# ==============================================================================

#' Render the table for one method variant: rows = parameter labels, columns =
#' mean_FDP, sd_FDP, mean_TPP, sd_TPP. Returns the table as a string.
render_mc_1d_method_table <- function(method_label, param_labels, results) {
  rw <- max(12L, nchar(param_labels) + 2L)  # widened for long labels
  cw <- 10L
  total_w <- rw + 4L * cw

  lines <- c(sprintf("\n  %s", method_label),
             paste0("  ", strrep("-", total_w)),
             paste0("  ", strrep(" ", rw),
                    sprintf("%10s%10s%10s%10s",
                            "mean_FDP", "sd_FDP", "mean_TPP", "sd_TPP")),
             paste0("  ", strrep("-", total_w)))
  for (k in seq_along(param_labels)) {
    r <- results[[k]]
    lines <- c(lines, sprintf("  %-*s%10.4f%10.4f%10.4f%10.4f", rw,
                              param_labels[k], r$mean_FDP, r$sd_FDP,
                              r$mean_TPP, r$sd_TPP))
  }
  paste0(paste(lines, collapse = "\n"), "\n\n")
}


# ==============================================================================
# MC 2-D matrix (mirrors Python render_mc_matrix / print_mc_matrix)
# ==============================================================================

#' Strip a common "axis=value" prefix; returns list(axis, vals).
.strip_axis_prefix <- function(labels) {
  axis <- ""
  vals <- character(length(labels))
  for (k in seq_along(labels)) {
    lbl <- labels[k]
    if (grepl("=", lbl, fixed = TRUE)) {
      parts <- strsplit(lbl, "=", fixed = TRUE)[[1L]]
      if (!nzchar(axis)) axis <- parts[1L]
      vals[k] <- paste(parts[-1L], collapse = "=")
    } else {
      vals[k] <- lbl
    }
  }
  list(axis = axis, vals = vals)
}


#' Render a 2-D sweep matrix (row axis left, column axis named once in the
#' "<row_axis> \ <col_axis>" corner cell; bare swept values as headers).
render_mc_matrix <- function(title, row_labels, col_labels, mat) {
  rl <- .strip_axis_prefix(row_labels)
  cl <- .strip_axis_prefix(col_labels)

  corner <- if (nzchar(rl$axis) || nzchar(cl$axis)) {
    paste0(rl$axis, " \\ ", cl$axis)
  } else {
    ""
  }
  rw    <- max(c(nchar(rl$vals), nchar(corner)))
  rwi   <- max(rw + 2L, 10L)
  cw    <- 10L
  total <- rwi + cw * length(cl$vals)

  # sprintf-based padding (formatC miscounts strings containing backslashes).
  pad_r <- function(s, w) sprintf(paste0("%", w, "s"), s)

  lines <- c(sprintf("\n  %s", title),
             paste0("  ", strrep("-", total)),
             paste0("  ", pad_r(corner, rwi),
                    paste(pad_r(cl$vals, cw), collapse = "")),
             paste0("  ", strrep("-", total)))
  for (i in seq_along(rl$vals)) {
    lines <- c(lines, paste0("  ", pad_r(rl$vals[i], rwi),
                             paste(sprintf("%10.4f", mat[i, ]),
                                   collapse = "")))
  }
  paste0(paste(lines, collapse = "\n"), "\n\n")
}


print_mc_matrix <- function(title, row_labels, col_labels, mat) {
  cat(render_mc_matrix(title, row_labels, col_labels, mat))
}


# ==============================================================================
# Save MC 2-D tables (mirrors Python save_mc_2d_tables)
# ==============================================================================

#' Write the six 2-D matrices to <out_dir>/<file_stem>.txt and the tidy
#' long-format CSV (method,metric,row_label,col_label,mean,sd) to .csv.
#' A TREX_SOLVERS run adds the selection suffix to the stem (out_stem()).
#'
#' @param en_2d,en_aug_2d,ien_2d `run_gvs_2d_sweep()` results for TENET,
#'   TENET_AUG, and TIENET_AUG.
#' @param cfg Named list with num_MC, tFDR, n, p (banner line).
save_mc_2d_tables <- function(scenario_tag, row_labels, col_labels, en_2d,
                              en_aug_2d, ien_2d, cfg, out_dir, file_stem) {
  if (is.null(file_stem) || !nzchar(file_stem)) return(invisible(NULL))
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
  file_stem <- out_stem(file_stem)

  txt_path <- file.path(out_dir, paste0(file_stem, ".txt"))
  txt <- paste0("\n", strrep("=", 78), "\n",
                sprintf("  T-Rex+GVS MC 2D Grid: %s  (MC=%d, tFDR=%s, n=%d, p=%d)\n",
                        scenario_tag, cfg$num_MC, fmt_num(cfg$tFDR),
                        cfg$n, cfg$p),
                strrep("=", 78), "\n")
  for (spec in list(list("mean_TPP [TENET]",      en_2d$mean_TPP),
                    list("mean_FDP [TENET]",      en_2d$mean_FDP),
                    list("mean_TPP [TENET_AUG]",  en_aug_2d$mean_TPP),
                    list("mean_FDP [TENET_AUG]",  en_aug_2d$mean_FDP),
                    list("mean_TPP [TIENET_AUG]", ien_2d$mean_TPP),
                    list("mean_FDP [TIENET_AUG]", ien_2d$mean_FDP))) {
    txt <- paste0(txt, render_mc_matrix(spec[[1L]], row_labels, col_labels,
                                        spec[[2L]]))
  }
  writeLines(txt, txt_path, sep = "")
  cat(sprintf("[Info] TXT saved to: %s\n", txt_path))

  # Row order mirrors the cpp/Python CSV: per grid cell, mean_FDP then
  # mean_TPP for TENET, TENET_AUG, and TIENET_AUG in turn.
  csv_lines <- "method,metric,row_label,col_label,mean,sd"
  for (i in seq_along(row_labels)) {
    for (j in seq_along(col_labels)) {
      for (m in list(list("TENET", en_2d), list("TENET_AUG", en_aug_2d),
                     list("TIENET_AUG", ien_2d))) {
        mat <- m[[2L]]
        csv_lines <- c(csv_lines,
                       sprintf("%s,mean_FDP,%s,%s,%.6f,%.6f", m[[1L]],
                               row_labels[i], col_labels[j],
                               mat$mean_FDP[i, j], mat$sd_FDP[i, j]),
                       sprintf("%s,mean_TPP,%s,%s,%.6f,%.6f", m[[1L]],
                               row_labels[i], col_labels[j],
                               mat$mean_TPP[i, j], mat$sd_TPP[i, j]))
      }
    }
  }
  csv_path <- file.path(out_dir, paste0(file_stem, ".csv"))
  writeLines(csv_lines, csv_path)
  cat(sprintf("[Info] CSV saved to: %s\n\n", csv_path))
  invisible(NULL)
}


# ==============================================================================
# Save MC 1-D method-comparison tables (demo 01 Parts 2-3; mirrors Python)
# ==============================================================================

#' Console + .txt + .csv writer for a fixed-operating-point method comparison
#' (TENET vs TIENET_AUG over lambda2_method / hc_linkage levels).
#' CSV schema mirrors cpp/Python: method,metric,<param_col>,mean,sd.
#' `ien_param_labels` gives the IEN track its own levels (lambda2 rules); the
#' CSV then lists the EN rows before the IEN rows, as the cpp Part 2 does.
#' A TREX_SOLVERS run adds the selection suffix to the stem (out_stem()).
save_mc_1d_method_tables <- function(scenario_tag, comparison, param_col,
                                     param_labels, en_results, ien_results,
                                     fixed_point_note, cfg, out_dir,
                                     file_stem, ien_param_labels = NULL) {
  ien_labels <- if (is.null(ien_param_labels)) param_labels else
    ien_param_labels
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
  file_stem <- out_stem(file_stem)
  txt_path <- file.path(out_dir, paste0(file_stem, ".txt"))

  txt <- paste0("\n", strrep("=", 78), "\n",
                sprintf("  T-Rex+GVS MC: %s — %s comparison  (%s, MC=%d, tFDR=%s)\n",
                        scenario_tag, comparison, fixed_point_note,
                        cfg$num_MC, fmt_num(cfg$tFDR)),
                strrep("=", 78), "\n",
                render_mc_1d_method_table("TENET", param_labels, en_results),
                render_mc_1d_method_table("TIENET_AUG", ien_labels,
                                          ien_results),
                strrep("=", 78), "\n\n")
  cat(txt)
  writeLines(txt, txt_path, sep = "")
  cat(sprintf("[Info] TXT saved to: %s\n", txt_path))

  rows_of <- function(method, lbl, r) {
    c(sprintf("%s,mean_FDP,%s,%.6f,%.6f", method, lbl, r$mean_FDP, r$sd_FDP),
      sprintf("%s,mean_TPP,%s,%.6f,%.6f", method, lbl, r$mean_TPP, r$sd_TPP))
  }
  csv_lines <- sprintf("method,metric,%s,mean,sd", param_col)
  if (is.null(ien_param_labels)) {
    for (k in seq_along(param_labels)) {
      csv_lines <- c(csv_lines,
                     rows_of("TENET", param_labels[k], en_results[[k]]),
                     rows_of("TIENET_AUG", param_labels[k], ien_results[[k]]))
    }
  } else {
    for (k in seq_along(param_labels))
      csv_lines <- c(csv_lines,
                     rows_of("TENET", param_labels[k], en_results[[k]]))
    for (k in seq_along(ien_labels))
      csv_lines <- c(csv_lines,
                     rows_of("TIENET_AUG", ien_labels[k], ien_results[[k]]))
  }
  csv_path <- file.path(out_dir, paste0(file_stem, ".csv"))
  writeLines(csv_lines, csv_path)
  cat(sprintf("[Info] CSV saved to: %s\n\n", csv_path))
  invisible(NULL)
}


# ==============================================================================
# Clustering-cut comparison (demo 01 Part 4; mirrors Python / cpp)
# ==============================================================================

#' Bonferroni noise floor for |corr| between independent columns:
#' corr_max = tanh(z / sqrt(n - 3)) with P(Z > z) = alpha / (2 m) over the
#' m = p (p - 1) / 2 column pairs (Fisher z); 0.346 at n = 200, p = 500.
noise_floor_corr_max <- function(n, p, alpha = 0.05) {
  z <- stats::qnorm(alpha / (p * (p - 1)), lower.tail = FALSE)
  tanh(z / sqrt(n - 3))
}


#' Group-level rates of one selection against the oracle prior_groups (1-based,
#' background columns as singletons): share of active groups hit, share of
#' active groups selected in full, share of touched groups without an active
#' variable.
#'
#' @return Named numeric vector c(hit, full, fdp).
.gvs_group_rates <- function(sel, prior_groups, true_support) {
  m      <- max(prior_groups)
  size   <- tabulate(prior_groups, nbins = m)
  active <- tabulate(prior_groups[true_support], nbins = m) > 0L
  chosen <- tabulate(prior_groups[sel], nbins = m)
  touched <- chosen > 0L
  c(hit  = if (any(active)) sum(active & touched) / sum(active) else 0,
    full = if (any(active)) sum(active & chosen == size) / sum(active) else 0,
    fdp  = if (any(touched)) sum(touched & !active) / sum(touched) else 0)
}


#' Cut-comparison point of a skipped (unselected) method: the fields of
#' `.run_mc_hastie_cut()` (simulation_utils.R), all NaN.
skipped_cut_point <- function() {
  fields <- c("mean_FDP", "sd_FDP", "mean_TPP", "sd_TPP", "grp_hit",
              "sd_grp_hit", "grp_full", "sd_grp_full", "grp_FDP",
              "sd_grp_FDP", "n_groups", "sd_n_groups")
  stats::setNames(as.list(rep(NaN, length(fields))), fields)
}


#' Console + .txt + .csv writer for the clustering-cut comparison (one column
#' per cut, TENET and TIENET_AUG blocks). CSV schema mirrors cpp/Python:
#' method,metric,cut,mean,sd. A TREX_SOLVERS run adds the selection suffix to
#' the stem (out_stem()).
save_mc_cut_tables <- function(scenario_tag, cut_labels, en_results,
                               ien_results, fixed_snr, fixed_rho, floor_cut,
                               cfg, out_dir, file_stem) {
  file_stem <- out_stem(file_stem)
  rows <- list(c("FDR", "mean_FDP", "mean_FDP", "sd_FDP"),
               c("TPR", "mean_TPP", "mean_TPP", "sd_TPP"),
               c("group hit", "grp_hit", "grp_hit", "sd_grp_hit"),
               c("group full", "grp_full", "grp_full", "sd_grp_full"),
               c("group FDR", "grp_FDP", "grp_FDP", "sd_grp_FDP"),
               c("groups", "n_groups", "n_groups", "sd_n_groups"))
  blocks <- list(TENET = en_results, TIENET_AUG = ien_results)
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

  lines <- c("", strrep("=", 78),
             sprintf("  T-Rex+GVS MC: %s — clustering-cut comparison  (SNR=%s, rho=%s, MC=%d, tFDR=%s)",
                     scenario_tag, fmt_num(fixed_snr), fmt_num(fixed_rho),
                     cfg$num_MC, fmt_num(cfg$tFDR)),
             sprintf("  noise floor corr_max = %.4f (Bonferroni, alpha = 0.05)",
                     floor_cut),
             strrep("=", 78))
  for (name in names(blocks)) {
    lines <- c(lines, "", name,
               paste0(sprintf("%-16s", "  metric"),
                      paste(sprintf("%14s", cut_labels), collapse = "")))
    for (r in rows) {
      vals <- vapply(blocks[[name]], function(res) res[[r[2L]]], numeric(1L))
      lines <- c(lines, paste0(sprintf("%-16s", paste0("  ", r[1L])),
                               paste(sprintf("%14.4f", vals), collapse = "")))
    }
  }
  txt <- paste0(paste(c(lines, strrep("=", 78)), collapse = "\n"), "\n\n")
  cat(txt)
  txt_path <- file.path(out_dir, paste0(file_stem, ".txt"))
  writeLines(txt, txt_path, sep = "")
  cat(sprintf("[Info] TXT saved to: %s\n", txt_path))

  csv_lines <- "method,metric,cut,mean,sd"
  for (name in names(blocks)) {
    for (k in seq_along(cut_labels)) {
      res <- blocks[[name]][[k]]
      for (r in rows) {
        csv_lines <- c(csv_lines,
                       sprintf("%s,%s,%s,%.6f,%.6f", name, r[3L], cut_labels[k],
                               res[[r[2L]]], res[[r[4L]]]))
      }
    }
  }
  csv_path <- file.path(out_dir, paste0(file_stem, ".csv"))
  writeLines(csv_lines, csv_path)
  cat(sprintf("[Info] CSV saved to: %s\n\n", csv_path))
  invisible(NULL)
}


# ==============================================================================
# Block-equicorrelated benchmark table writer (demo 08; mirrors Python
# print_block_grid_table)
# ==============================================================================

#' Formatted MC aggregate table for one truth scenario (rows M1-M4 per
#' (rho, SNR) cell). If `file_stem` is non-empty, also writes the table as
#' .txt and all aggregates as a tidy .csv into `out_dir` (a TREX_SOLVERS run
#' adds the selection suffix to the stem, out_stem()).
#'
#' `results[[i_rho]][[i_snr]]` must be a list with entries M1..M4, each a
#' named list of aggregates: mean_coord_tpr, sd_coord_tpr, mean_coord_fdp,
#' sd_coord_fdp, mean_exact_support, mean_block_hit, mean_block_fdp,
#' mean_full_block, mean_null_act, mean_purity, mean_lambda2, mean_T_stop,
#' mean_M_found.
print_block_grid_table <- function(scenario_tag, scenario, rho_grid, snr_grid,
                                   results, cfg, out_dir, file_stem = "") {
  blk_metric_name <- if (scenario == "WHOLE") "full_blk" else "blk_hit"
  blk_key <- if (scenario == "WHOLE") "mean_full_block" else "mean_block_hit"
  method_labels <- c(M1 = "M1(EN-C)", M2 = "M2(EN-O)",
                     M3 = "M3(IE-C)", M4 = "M4(IE-O)")
  cw <- 10L

  lines <- c("", strrep("=", 90),
             sprintf("  Block Bench MC: %s  (MC=%d, tFDR=%s, n=%d, p=%d, G=%d, blk=%d)",
                     scenario_tag, cfg$num_MC, fmt_num(cfg$tFDR), cfg$n,
                     cfg$p, cfg$G, cfg$block_size),
             strrep("=", 90))
  for (i_rho in seq_along(rho_grid)) {
    lines <- c(lines, sprintf("\n  rho = %.2f", rho_grid[i_rho]),
               paste0("  ", sprintf("%-12s", "method"), sprintf("%7s", "SNR"),
                      sprintf("%10s%10s%10s%10s%10s%10s%10s",
                              "TPR", "FDR", "block_FDR", blk_metric_name,
                              "purity", "T_stop", "M_found")),
               paste0("  ", strrep("-", 12L + 7L + 7L * cw)))
    for (i_snr in seq_along(snr_grid)) {
      cell <- results[[i_rho]][[i_snr]]
      for (mk in names(method_labels)) {
        a <- cell[[mk]]
        lines <- c(lines,
                   sprintf("  %-12s%7.2f%10.4f%10.4f%10.4f%10.4f%10.4f%10.1f%10.1f",
                           method_labels[[mk]], snr_grid[i_snr],
                           a$mean_coord_tpr, a$mean_coord_fdp,
                           a$mean_block_fdp, a[[blk_key]], a$mean_purity,
                           a$mean_T_stop, a$mean_M_found))
      }
      lines <- c(lines, "")
    }
  }
  lines <- c(lines, paste0(strrep("=", 90), "\n"))
  table_str <- paste(lines, collapse = "\n")
  cat(table_str, "\n", sep = "")

  if (!nzchar(file_stem)) return(invisible(NULL))
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
  file_stem <- out_stem(file_stem)

  txt_path <- file.path(out_dir, paste0(file_stem, ".txt"))
  writeLines(paste0(table_str, "\n"), txt_path, sep = "")
  cat(sprintf("[Info] TXT saved to: %s\n", txt_path))

  csv_lines <- paste0("scenario,rho,snr,method,",
                      "mean_coord_tpr,sd_coord_tpr,mean_coord_fdp,",
                      "sd_coord_fdp,mean_exact_sup,mean_block_hit,",
                      "mean_block_fdp,mean_full_block,mean_null_act,",
                      "mean_purity,mean_lambda2,mean_T_stop,mean_M_found")
  for (i_rho in seq_along(rho_grid)) {
    for (i_snr in seq_along(snr_grid)) {
      for (mk in names(method_labels)) {
        a <- results[[i_rho]][[i_snr]][[mk]]
        csv_lines <- c(csv_lines,
                       sprintf(paste0("%s,%.6f,%.6f,%s,%.6f,%.6f,%.6f,%.6f,",
                                      "%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,",
                                      "%.6f,%.6f"),
                               scenario_tag, rho_grid[i_rho], snr_grid[i_snr],
                               method_labels[[mk]],
                               a$mean_coord_tpr, a$sd_coord_tpr,
                               a$mean_coord_fdp, a$sd_coord_fdp,
                               a$mean_exact_support, a$mean_block_hit,
                               a$mean_block_fdp, a$mean_full_block,
                               a$mean_null_act, a$mean_purity,
                               a$mean_lambda2, a$mean_T_stop, a$mean_M_found))
      }
    }
  }
  csv_path <- file.path(out_dir, paste0(file_stem, ".csv"))
  writeLines(csv_lines, csv_path)
  cat(sprintf("[Info] CSV saved to: %s\n\n", csv_path))
  invisible(NULL)
}
