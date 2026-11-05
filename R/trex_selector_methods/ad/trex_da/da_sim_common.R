# ==============================================================================
# da_sim_common.R
# ==============================================================================
#
# Shared infrastructure for the DA-TRex demos: support generators, the default
# solver set (TLARS / TAFS / TOMP with per-solver exchangeable tie-breaking),
# parallel Monte Carlo runners for DA-TRex and the base (no-DA) T-Rex, generic
# parameter sweeps, and the console / TXT / CSV result writers.
#
# Mirrors:
#  Python/trex_selector_methods/ad/trex_da/da_sim_common.py
#  cpp/trex_selector_methods/ad/trex_da/trex_da_sim_common.hpp
#
# Content:
#   default_solvers()                       <- default_solvers()
#   base_solver_name()                      <- base_solver_name()
#   make_support_one_per_block()            <- one_per_block_support()
#   run_mc_trials()                         <- run_mc_trials() / run_mc_trials_base()
#   run_param_sweep() / run_snr_sweep()     <- run_param_sweep() / run_snr_sweep()
#   save_and_print_grid_results()           <- save_and_print_grid_results()
#   save_and_print_2d_gap_rho_results()     <- save_and_print_2d_gap_rho_results()
#   save_and_print_2d_two_support_results() <- save_and_print_2d_two_support_results()
#
# (make_support_capped_spread() / make_support_random() come from the shared
# ../support_generators.R, which each demo sources alongside this file.)
#
# Result files use the cpp naming pattern da_trex_mc_{scenario_tag}.{txt,csv}
# and the same tidy CSV schemas; they are written into each demo's
# simulation_results/data/ folder.
#
# Seed spaces (mirror cpp/Python):
#   DGP (X, noise)    : trial_seed = base_seed_offset + mc      (mc = 0-based)
#   Random support    : trial_seed + 500000
#   Selector          : seed = -1 (hardware entropy; required for valid MC FDR)
#
# Support / group indices are 1-based (R convention); the Python counterpart
# is 0-based.
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


# ==============================================================================
# Section header (mirrors Python print_section_header)
# ==============================================================================

print_section_header <- function(title) {
  cat("\n", strrep("=", 60L), "\n", sep = "")
  cat(title, "\n", sep = "")
  cat(strrep("=", 60L), "\n", sep = "")
}


# ==============================================================================
# Default solver set (mirrors Python/cpp default_solvers)
# ==============================================================================

#' Default DA-TRex solver descriptors (plain named lists).
#'
#' exch_tie_alpha = 0.25 for the greedy solvers TAFS/TOMP: exchangeability-
#' calibrated stochastic tie-breaking, required for DA FDR control at high
#' correlation. Path solver TLARS stays 0.
#' LARS-based solvers run with stagnation control OFF; greedy solvers
#' default it ON. TLARS_STAG is TLARS with the stagnation stop ON, the
#' comparison row for TLARS' long T-loops (last, as in cpp). `da_variant` tags
#' the DA method in the row label, e.g. "+AR1" -> "TREX-DA+AR1: TLARS".
default_solvers <- function(da_variant = "") {
  p <- paste0("TREX-DA", da_variant, ": ")
  list(
    list(solver = "TLARS", name = paste0(p, "TLARS"), rho_afs = 0.0,
         use_stagnation = FALSE, exch_tie_alpha = 0.0),
    list(solver = "TAFS", name = paste0(p, "TAFS"), rho_afs = 0.3,
         use_stagnation = TRUE, exch_tie_alpha = 0.25),
    list(solver = "TOMP", name = paste0(p, "TOMP"), rho_afs = 0.0,
         use_stagnation = TRUE, exch_tie_alpha = 0.25),
    list(solver = "TLARS", name = paste0(p, "TLARS_STAG"), rho_afs = 0.0,
         use_stagnation = TRUE, exch_tie_alpha = 0.0)
  )
}


#' Derive the base (no-DA) comparison-row name from a DA row name:
#' 'TREX-DA[+variant]: TLARS' -> 'TREX: TLARS' (mirrors run_param_sweep).
base_solver_name <- function(name) {
  if (startsWith(name, "TREX-DA")) {
    colon <- regexpr(": ", name, fixed = TRUE)
    tail_ <- if (colon != -1L) substring(name, colon + 2L) else name
    return(paste0("TREX: ", tail_))
  }
  paste0("TREX: ", name)
}


# ==============================================================================
# Support generator (1-based; mirrors cpp one_per_block_support)
# ==============================================================================

#' One active predictor per block: the first index of each of the first
#' min(s, p %/% block_size) blocks (block_size = 0 -> auto: p %/% s).
make_support_one_per_block <- function(s, p, block_size = 0L) {
  if (s <= 0L || p <= 0L) return(integer(0L))
  g <- if (block_size > 0L) as.integer(block_size) else max(1L, p %/% s)
  active_blocks <- min(s, p %/% g)
  1L + (0L:(active_blocks - 1L)) * g
}


# ==============================================================================
# Control-parameter builders
# ==============================================================================

#' Build a trex_control() list from a solver descriptor (mirrors Python
#' _make_trex_ctrl / cpp make_trex_control + the per-solver fields set by
#' run_param_sweep).
.make_trex_ctrl <- function(solver_desc, K) {
  TRexSelector::trex_control(
    solver                   = solver_desc$solver,
    K                        = as.integer(K),
    max_dummy_multiplier     = 10,
    use_max_T_stop           = TRUE,
    dummy_distribution       = TRexSelector::trex_dummy_distribution(),
    lloop_strategy           = "STANDARD",
    tloop_stagnation_stop    = isTRUE(solver_desc$use_stagnation),
    tloop_max_stagnant_steps = 7,
    use_memory_mapping       = FALSE,
    rho_afs                  = solver_desc$rho_afs,
    exch_tie_alpha           = solver_desc$exch_tie_alpha
  )
}


#' Build a trex_da_control() list from a spec list (mirrors Python
#' _make_da_ctrl).
#'
#' Keys: method ("AR1" | "EQUI" | "NN" | "BT"), hc_linkage ("Single" |
#' "Complete" | "Average"), prior_groups (list of label vectors; when present
#' the method field is ignored by the selector), rho_thr_DA, cor_coef.
.make_da_ctrl <- function(da_spec) {
  da_args <- list()
  if (!is.null(da_spec$method))       da_args$da_method  <- da_spec$method
  if (!is.null(da_spec$hc_linkage))   da_args$hc_linkage <- da_spec$hc_linkage
  if (!is.null(da_spec$prior_groups)) {
    da_args$prior_groups <- lapply(da_spec$prior_groups, as.integer)
  }
  if (!is.null(da_spec$rho_thr_DA))   da_args$rho_thr_DA <- da_spec$rho_thr_DA
  if (!is.null(da_spec$cor_coef))     da_args$cor_coef   <- da_spec$cor_coef
  do.call(TRexSelector::trex_da_control, da_args)
}


# ==============================================================================
# Standard MC inner loop (mirrors Python run_mc_trials / cpp run_mc_trials)
# ==============================================================================

#' Run num_MC parallel trials of one (DGP, solver, DA) configuration.
#'
#' @param dgp     DGP function name (string) resolved in the sourcing
#'   environment, e.g. "dgp_ar1_snr".
#' @param support Fixed 1-based index vector, or NULL to redraw a random
#'   support per trial (seed = trial_seed + 500000).
#' @param da_spec NULL runs the base (no-DA) T-Rex selector.
#'
#' @return list(avg_fdr, avg_tpr, avg_L, avg_T, sd_fdr, sd_tpr).
run_mc_trials <- function(num_MC, progress_label, n, p, dgp, dgp_kwargs,
                          support, s, tFDR, solver_desc, K, da_spec,
                          base_seed_offset, n_cores = num_cores) {
  dgp_fun <- match.fun(dgp)
  # The white-noise DGPs take p_ar/p_white instead of a single p.
  pass_p <- "p" %in% names(formals(dgp_fun))

  one_trial <- function(mc) {
    trial_seed <- as.integer(base_seed_offset) + mc - 1L

    trial_support <- if (is.null(support)) {
      make_support_random(s, p, trial_seed)
    } else {
      support
    }

    dgp_args <- c(list(n = n, support = trial_support, seed = trial_seed),
                  dgp_kwargs)
    if (pass_p) dgp_args$p <- p
    dat <- do.call(dgp_fun, dgp_args)

    ctrl <- .make_trex_ctrl(solver_desc, K)
    sel <- if (!is.null(da_spec)) {
      TRexSelector::TRexDASelector$new(
        dat$X, dat$y, tFDR = tFDR, seed = -1L, verbose = FALSE,
        da_control = .make_da_ctrl(da_spec), control = ctrl
      )
    } else {
      TRexSelector::TRexSelector$new(
        dat$X, dat$y, tFDR = tFDR, seed = -1L, verbose = FALSE,
        control = ctrl
      )
    }
    sel$select()

    ts <- dat$true_support
    c(FDP = TRexSelector::trex_fdp(sel$selected_indices, ts),
      TPP = TRexSelector::trex_tpp(sel$selected_indices, ts),
      L   = as.numeric(sel$L),
      T   = as.numeric(sel$T_stop))
  }

  # One resumable grid cell (mc_runtime.R): restored from the cell cache after a
  # restart; the trials run on PSOCK workers.
  key <- paste0("da|", progress_label, "|",
                cell_digest(n, p, dgp, dgp_kwargs, support, s, tFDR, solver_desc, K,
                            da_spec, base_seed_offset, num_MC))
  out <- cell_cached(key, function() {
    cat(sprintf("  %s — Running %d MC trials ...\n", progress_label, num_MC))
    res <- do.call(rbind, mc_map(seq_len(num_MC), one_trial, n_cores))
    list(
      avg_fdr = mean(res[, "FDP"]),
      avg_tpr = mean(res[, "TPP"]),
      avg_L   = mean(res[, "L"]),
      avg_T   = mean(res[, "T"]),
      sd_fdr  = if (num_MC > 1L) stats::sd(res[, "FDP"]) else 0.0,
      sd_tpr  = if (num_MC > 1L) stats::sd(res[, "TPP"]) else 0.0
    )
  })

  cat(sprintf("  %s — done. TPP=%.3f  FDP=%.3f\n\n",
              progress_label, out$avg_tpr, out$avg_fdr))
  out
}


# ==============================================================================
# Header-extra line wrapping (mirrors Python/cpp wrap_header_extra)
# ==============================================================================

.HEADER_WIDTH <- 78L

#' Wrap a header info line at runs of two-or-more spaces or at a single
#' space that follows a comma; any other single space is inside a token.
wrap_header_extra <- function(text, width = .HEADER_WIDTH, indent = "  ") {
  n <- nchar(text)
  chars <- strsplit(text, "", fixed = TRUE)[[1L]]
  tokens <- list()   # each: c(token_text, following_separator)
  pos <- 1L
  while (pos <= n) {
    brk <- NA_integer_
    brk_end <- NA_integer_
    i <- pos
    while (i <= n) {
      if (chars[i] != " ") {
        i <- i + 1L
        next
      }
      j <- i
      while (j <= n && chars[j] == " ") j <- j + 1L
      if ((j - i) >= 2L || (i > 1L && chars[i - 1L] == ",")) {
        brk <- i
        brk_end <- j
        break
      }
      i <- j   # lone space inside a token -- keep scanning
    }
    if (is.na(brk)) {
      tokens[[length(tokens) + 1L]] <- c(substr(text, pos, n), "")
      break
    }
    tokens[[length(tokens) + 1L]] <- c(substr(text, pos, brk - 1L),
                                       substr(text, brk, brk_end - 1L))
    pos <- brk_end
  }

  out <- ""
  line <- ""
  for (k in seq_along(tokens)) {
    tok <- tokens[[k]][1L]
    if (!nzchar(tok)) next
    if (!nzchar(line)) {
      line <- paste0(indent, tok)
      next
    }
    sep <- tokens[[k - 1L]][2L]
    if (nchar(line) + nchar(sep) + nchar(tok) <= width) {
      line <- paste0(line, sep, tok)
    } else {
      out <- paste0(out, line, "\n")
      line <- paste0(indent, tok)
    }
  }
  if (nzchar(line)) out <- paste0(out, line, "\n")
  out
}


# ==============================================================================
# Save / print grid results (mirrors Python/cpp save_and_print_grid_results)
# ==============================================================================

#' Write one grid-sweep result table to console, TXT, and tidy CSV.
#'
#' The maps are named lists solver_name -> vector over grid_values. TXT layout
#' and the CSV schema `solver,metric,<GRID>,value` match the cpp/Python suites.
save_and_print_grid_results <- function(out_dir, scenario_tag, grid_label,
                                        grid_values, num_MC, solver_names,
                                        fdr_map, tpr_map, sd_fdr_map,
                                        sd_tpr_map, avg_L_map, avg_T_map,
                                        header_extra = "") {
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
  txt_path <- file.path(out_dir, sprintf("da_trex_mc_%s.txt", scenario_tag))
  csv_path <- file.path(out_dir, sprintf("da_trex_mc_%s.csv", scenario_tag))

  txt <- paste0("\n", strrep("=", 78L), "\n",
                sprintf("  DA-TRex MC: %s  (MC=%d)\n", scenario_tag, num_MC))
  if (nzchar(header_extra)) txt <- paste0(txt, wrap_header_extra(header_extra))
  txt <- paste0(txt, strrep("=", 78L), "\n\n")

  indent_w <- 2L
  metric_w <- 23L
  col_w <- 10L
  ncols <- length(grid_values)
  sep_w <- indent_w + metric_w + col_w * ncols

  integral_grid <- all(grid_values == floor(grid_values) &
                         abs(grid_values) < 1e6)
  th <- paste0(strrep(" ", indent_w), sprintf("%-*s", metric_w, grid_label))
  for (v in grid_values) {
    th <- paste0(th, if (integral_grid) sprintf("%*d", col_w, as.integer(v)) else sprintf("%*.2f", col_w, v))
  }
  txt <- paste0(txt, th, "\n", strrep("-", sep_w), "\n")

  row_ <- function(metric, data) {
    r <- paste0(strrep(" ", indent_w), sprintf("%-*s", metric_w, metric))
    paste0(r, paste(sprintf("%*.4f", col_w, data), collapse = ""), "\n")
  }

  for (nm in solver_names) {
    txt <- paste0(txt, "\n", nm, "\n")
    txt <- paste0(txt, row_("FDR", fdr_map[[nm]]))
    txt <- paste0(txt, row_("sd_FDR", sd_fdr_map[[nm]]))
    txt <- paste0(txt, row_("TPR", tpr_map[[nm]]))
    txt <- paste0(txt, row_("sd_TPR", sd_tpr_map[[nm]]))
    if (!is.null(avg_L_map) && !is.null(avg_L_map[[nm]])) {
      txt <- paste0(txt, row_("Avg L", avg_L_map[[nm]]))
    }
    if (!is.null(avg_T_map) && !is.null(avg_T_map[[nm]])) {
      txt <- paste0(txt, row_("Avg T", avg_T_map[[nm]]))
    }
  }
  txt <- paste0(txt, "\n", strrep("=", 78L), "\n\n")

  cat(txt)
  writeLines(txt, txt_path, sep = "")
  cat(sprintf("[Info] TXT saved to: %s\n", txt_path))

  # CSV -- tidy long format: solver,metric,<GRID>,value
  csv_col <- gsub(" ", "_", grid_label, fixed = TRUE)
  csv_lines <- sprintf("solver,metric,%s,value", csv_col)
  for (nm in solver_names) {
    for (i in seq_along(grid_values)) {
      gv_s <- sprintf("%.6f", grid_values[i])
      csv_lines <- c(csv_lines,
                     sprintf("%s,FDR,%s,%.6f", nm, gv_s, fdr_map[[nm]][i]),
                     sprintf("%s,sd_FDR,%s,%.6f", nm, gv_s, sd_fdr_map[[nm]][i]),
                     sprintf("%s,TPR,%s,%.6f", nm, gv_s, tpr_map[[nm]][i]),
                     sprintf("%s,sd_TPR,%s,%.6f", nm, gv_s, sd_tpr_map[[nm]][i]))
      if (!is.null(avg_L_map) && !is.null(avg_L_map[[nm]])) {
        csv_lines <- c(csv_lines, sprintf("%s,Avg_L,%s,%.6f", nm, gv_s,
                                          avg_L_map[[nm]][i]))
      }
      if (!is.null(avg_T_map) && !is.null(avg_T_map[[nm]])) {
        csv_lines <- c(csv_lines, sprintf("%s,Avg_T,%s,%.6f", nm, gv_s,
                                          avg_T_map[[nm]][i]))
      }
    }
  }
  writeLines(csv_lines, csv_path)
  cat(sprintf("[Info] CSV saved to: %s\n\n", csv_path))
  invisible(NULL)
}


# ==============================================================================
# Generic parameter sweep (mirrors Python/cpp run_param_sweep / run_snr_sweep)
# ==============================================================================

#' Run an MC parameter sweep: for each solver x grid value, run num_MC
#' trials; optionally re-run every solver without the DA correction as a
#' 'TREX: <solver>' comparison row.
#'
#' make_dgp(val) must return a list with entries
#' n, p, dgp (generator name), kwargs (DGP kwargs incl. amplitude / snr /
#' rho / ...), support (fixed 1-based index vector, or NULL for a per-trial
#' random redraw), s (support size, used when support is NULL).
run_param_sweep <- function(out_dir, scenario_tag, param_label, grid_values,
                            num_MC, tFDR, base_seed, solvers, make_dgp,
                            da_spec, K = 20L, header_extra = "",
                            include_base_trex = FALSE, n_cores = num_cores) {
  print_section_header(paste0("MC: ", scenario_tag))

  all_names <- vapply(solvers, `[[`, character(1L), "name")
  zero <- stats::setNames(rep(list(numeric(length(grid_values))), length(all_names)),
                          all_names)
  fdr <- zero
  tpr <- zero
  sd_fdr <- zero
  sd_tpr <- zero
  avg_L <- zero
  avg_T <- zero

  acc <- environment()  # the result lists that sweep_one fills
  sweep_one <- function(solver_desc, name, spec_da) {
    for (gi in seq_along(grid_values)) {
      val <- grid_values[gi]
      d <- make_dgp(val)
      res <- run_mc_trials(
        num_MC, sprintf("%s = %.2f", param_label, val),
        d$n, d$p, d$dgp, d$kwargs, d$support, d$s,
        tFDR, solver_desc, K, spec_da,
        base_seed + (gi - 1L) * 10000L, n_cores
      )
      acc$fdr[[name]][gi] <- res$avg_fdr
      acc$tpr[[name]][gi] <- res$avg_tpr
      acc$sd_fdr[[name]][gi] <- res$sd_fdr
      acc$sd_tpr[[name]][gi] <- res$sd_tpr
      acc$avg_L[[name]][gi] <- res$avg_L
      acc$avg_T[[name]][gi] <- res$avg_T
    }
  }

  for (sv in solvers) {
    cat(sprintf("\n  Solver: %s\n", sv$name))
    sweep_one(sv, sv$name, da_spec)
  }

  if (include_base_trex) {
    for (sv in solvers) {
      nm <- base_solver_name(sv$name)
      all_names <- c(all_names, nm)
      fdr[[nm]] <- numeric(length(grid_values))
      tpr[[nm]] <- numeric(length(grid_values))
      sd_fdr[[nm]] <- numeric(length(grid_values))
      sd_tpr[[nm]] <- numeric(length(grid_values))
      avg_L[[nm]] <- numeric(length(grid_values))
      avg_T[[nm]] <- numeric(length(grid_values))
      cat(sprintf("\n  Solver: %s (no DA)\n", nm))
      sweep_one(sv, nm, NULL)
    }
  }

  save_and_print_grid_results(out_dir, scenario_tag, param_label,
                              grid_values, num_MC, all_names,
                              fdr, tpr, sd_fdr, sd_tpr, avg_L, avg_T,
                              header_extra)
}


#' Convenience wrapper around run_param_sweep with param_label = 'SNR'.
run_snr_sweep <- function(out_dir, scenario_tag, snr_values, num_MC, tFDR,
                          base_seed, solvers, make_dgp, da_spec, K = 20L,
                          header_extra = "", include_base_trex = FALSE,
                          n_cores = num_cores) {
  run_param_sweep(out_dir, scenario_tag, "SNR", snr_values, num_MC, tFDR,
                  base_seed, solvers, make_dgp, da_spec, K, header_extra,
                  include_base_trex, n_cores)
}


# ==============================================================================
# 2D Gap x Rho saving utility (mirrors Python/cpp
# save_and_print_2d_gap_rho_results)
# ==============================================================================

#' Integer string for whole numbers, else two decimals (cpp fmt_val).
.fmt_grid_val <- function(v) {
  if (v == floor(v) && abs(v) < 1e6) return(as.character(as.integer(v)))
  sprintf("%.2f", v)
}


#' Write the 2D Gap x Rho sweep (CappedSpread matrices [n_rho x n_gap] plus
#' Random vectors [n_rho]) to console, TXT, and CSV
#' (`solver,metric,rho,support_type,gap,value`).
save_and_print_2d_gap_rho_results <- function(out_dir, scenario_tag, gap_grid,  # nolint: object_length_linter. C++ name
                                              rho_grid, kappa_boundary, num_MC,
                                              solver_names, tpp_cs, fdp_cs,
                                              sd_tpp_cs, sd_fdp_cs, tpp_rand,
                                              fdp_rand, sd_tpp_rand,
                                              sd_fdp_rand, header_extra = "") {
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
  txt_path <- file.path(out_dir, sprintf("da_trex_mc_%s.txt", scenario_tag))
  csv_path <- file.path(out_dir, sprintf("da_trex_mc_%s.csv", scenario_tag))

  n_gap <- length(gap_grid)
  n_cols <- n_gap + 1L   # gap columns + Random
  rw <- 10L
  cw <- 9L

  col_header <- function() {
    h <- strrep(" ", rw)
    for (g in gap_grid) {
      h <- paste0(h, sprintf("%*s", cw, paste0("gap=", as.integer(g))))
    }
    paste0(h, sprintf("%*s", cw, "Random"), "\n",
           "  ", strrep("-", rw + cw * n_cols), "\n")
  }

  matrix_block <- function(label, cs_mat, rand_vec) {
    block <- paste0("  ", label, "\n", col_header())
    for (ir in seq_along(rho_grid)) {
      block <- paste0(block, "  ",
                      sprintf("%-*s", rw,
                              paste0("rho=", sprintf("%.1f", rho_grid[ir]))))
      block <- paste0(block,
                      paste(sprintf("%*.4f", cw, cs_mat[ir, ]), collapse = ""))
      block <- paste0(block, sprintf("%*.4f", cw, rand_vec[ir]), "\n")
    }
    paste0(block, "\n")
  }

  txt <- paste0("\n", strrep("=", 78L), "\n",
                sprintf("  DA-TRex MC: %s  (MC=%d)\n", scenario_tag, num_MC))
  if (nzchar(header_extra)) txt <- paste0(txt, wrap_header_extra(header_extra))
  txt <- paste0(txt, strrep("=", 78L), "\n\n")

  kb <- "  DA+AR1 correction window: kappa = ceil(log(0.02) / log(rho))\n\n"
  kb <- paste0(kb, "  ", sprintf("%-10s", "rho"),
               paste(sprintf("%6.1f", rho_grid), collapse = ""))
  kb <- paste0(kb, "\n  ", sprintf("%-10s", "kappa"),
               paste(sprintf("%6d", as.integer(kappa_boundary)), collapse = ""))
  txt <- paste0(txt, kb, "\n\n")

  for (nm in solver_names) {
    txt <- paste0(txt, strrep("-", 78L), "\n",
                  sprintf("  Solver: %s\n", nm), strrep("-", 78L), "\n\n")
    txt <- paste0(txt, matrix_block("mean_TPP", tpp_cs[[nm]], tpp_rand[[nm]]))
    txt <- paste0(txt, matrix_block("sd_TPP", sd_tpp_cs[[nm]], sd_tpp_rand[[nm]]))
    txt <- paste0(txt, matrix_block("mean_FDP", fdp_cs[[nm]], fdp_rand[[nm]]))
    txt <- paste0(txt, matrix_block("sd_FDP", sd_fdp_cs[[nm]], sd_fdp_rand[[nm]]))
  }
  txt <- paste0(txt, strrep("=", 78L), "\n\n")

  cat(txt)
  writeLines(txt, txt_path, sep = "")
  cat(sprintf("[Info] TXT saved to: %s\n", txt_path))

  csv_lines <- "solver,metric,rho,support_type,gap,value"
  for (nm in solver_names) {
    for (ir in seq_along(rho_grid)) {
      rho_s <- sprintf("%.6f", rho_grid[ir])
      for (ig in seq_along(gap_grid)) {
        gap <- as.integer(gap_grid[ig])
        csv_lines <- c(csv_lines,
                       sprintf("%s,mean_TPP,%s,CappedSpread,%d,%.6f", nm, rho_s,
                               gap, tpp_cs[[nm]][ir, ig]),
                       sprintf("%s,sd_TPP,%s,CappedSpread,%d,%.6f", nm, rho_s,
                               gap, sd_tpp_cs[[nm]][ir, ig]),
                       sprintf("%s,mean_FDP,%s,CappedSpread,%d,%.6f", nm, rho_s,
                               gap, fdp_cs[[nm]][ir, ig]),
                       sprintf("%s,sd_FDP,%s,CappedSpread,%d,%.6f", nm, rho_s,
                               gap, sd_fdp_cs[[nm]][ir, ig]))
      }
      csv_lines <- c(csv_lines,
                     sprintf("%s,mean_TPP,%s,Random,,%.6f", nm, rho_s,
                             tpp_rand[[nm]][ir]),
                     sprintf("%s,sd_TPP,%s,Random,,%.6f", nm, rho_s,
                             sd_tpp_rand[[nm]][ir]),
                     sprintf("%s,mean_FDP,%s,Random,,%.6f", nm, rho_s,
                             fdp_rand[[nm]][ir]),
                     sprintf("%s,sd_FDP,%s,Random,,%.6f", nm, rho_s,
                             sd_fdp_rand[[nm]][ir]))
    }
  }
  writeLines(csv_lines, csv_path)
  cat(sprintf("[Info] CSV saved to: %s\n\n", csv_path))
  invisible(NULL)
}


# ==============================================================================
# 2D two-support saving utility (mirrors Python/cpp
# save_and_print_2d_two_support_results)
# ==============================================================================

#' Write a 2D parameter sweep with full CappedSpread and Random matrices
#' [n_row x n_col] per solver to console, TXT, and CSV
#' (`solver,support_type,metric,<ROW>,<COL>,value`).
save_and_print_2d_two_support_results <- function(out_dir, scenario_tag,  # nolint: object_length_linter. C++ name
                                                  row_label, col_label,
                                                  row_grid, col_grid, num_MC,
                                                  solver_names, tpp_cs, fdp_cs,
                                                  sd_tpp_cs, sd_fdp_cs,
                                                  tpp_rand, fdp_rand,
                                                  sd_tpp_rand, sd_fdp_rand,
                                                  header_extra = "") {
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
  txt_path <- file.path(out_dir, sprintf("da_trex_mc_%s.txt", scenario_tag))
  csv_path <- file.path(out_dir, sprintf("da_trex_mc_%s.csv", scenario_tag))

  n_row <- length(row_grid)
  n_col <- length(col_grid)
  rlw <- 12L
  cw <- 9L

  col_header <- function() {
    h <- paste0("  ", sprintf("%*s", rlw, col_label))
    for (cv in col_grid) {
      h <- paste0(h, sprintf("%*s", cw, .fmt_grid_val(cv)))
    }
    paste0(h, "\n", "  ", strrep("-", rlw + cw * n_col), "\n")
  }

  matrix_block <- function(label, mat) {
    block <- paste0("  ", label, "\n", col_header())
    for (ir in seq_len(n_row)) {
      rl <- paste0(row_label, "=", .fmt_grid_val(row_grid[ir]))
      block <- paste0(block, "  ", sprintf("%-*s", rlw, rl),
                      paste(sprintf("%*.4f", cw, mat[ir, ]), collapse = ""),
                      "\n")
    }
    paste0(block, "\n")
  }

  txt <- paste0("\n", strrep("=", 78L), "\n",
                sprintf("  DA-TRex MC: %s  (MC=%d)\n", scenario_tag, num_MC))
  if (nzchar(header_extra)) txt <- paste0(txt, wrap_header_extra(header_extra))
  txt <- paste0(txt, strrep("=", 78L), "\n\n")

  for (nm in solver_names) {
    txt <- paste0(txt, strrep("-", 78L), "\n",
                  sprintf("  Solver: %s\n", nm), strrep("-", 78L), "\n\n")
    txt <- paste0(txt, "  [CappedSpread]\n\n")
    txt <- paste0(txt, matrix_block("mean_TPP", tpp_cs[[nm]]))
    txt <- paste0(txt, matrix_block("sd_TPP", sd_tpp_cs[[nm]]))
    txt <- paste0(txt, matrix_block("mean_FDP", fdp_cs[[nm]]))
    txt <- paste0(txt, matrix_block("sd_FDP", sd_fdp_cs[[nm]]))
    txt <- paste0(txt, "  [Random]\n\n")
    txt <- paste0(txt, matrix_block("mean_TPP", tpp_rand[[nm]]))
    txt <- paste0(txt, matrix_block("sd_TPP", sd_tpp_rand[[nm]]))
    txt <- paste0(txt, matrix_block("mean_FDP", fdp_rand[[nm]]))
    txt <- paste0(txt, matrix_block("sd_FDP", sd_fdp_rand[[nm]]))
  }
  txt <- paste0(txt, strrep("=", 78L), "\n\n")

  cat(txt)
  writeLines(txt, txt_path, sep = "")
  cat(sprintf("[Info] TXT saved to: %s\n", txt_path))

  csv_row <- gsub(" ", "_", row_label, fixed = TRUE)
  csv_col <- gsub(" ", "_", col_label, fixed = TRUE)
  csv_lines <- sprintf("solver,support_type,metric,%s,%s,value",
                       csv_row, csv_col)

  write_block <- function(nm, support_type, tpp_m, fdp_m, sd_tpp_m, sd_fdp_m) {
    block <- character(0L)
    for (ir in seq_len(n_row)) {
      for (ic in seq_len(n_col)) {
        rv_s <- sprintf("%.6f", row_grid[ir])
        cv_s <- sprintf("%.6f", col_grid[ic])
        block <- c(block,
                   sprintf("%s,%s,mean_TPP,%s,%s,%.6f", nm, support_type,
                           rv_s, cv_s, tpp_m[[nm]][ir, ic]),
                   sprintf("%s,%s,sd_TPP,%s,%s,%.6f", nm, support_type,
                           rv_s, cv_s, sd_tpp_m[[nm]][ir, ic]),
                   sprintf("%s,%s,mean_FDP,%s,%s,%.6f", nm, support_type,
                           rv_s, cv_s, fdp_m[[nm]][ir, ic]),
                   sprintf("%s,%s,sd_FDP,%s,%s,%.6f", nm, support_type,
                           rv_s, cv_s, sd_fdp_m[[nm]][ir, ic]))
      }
    }
    block
  }

  for (nm in solver_names) {
    csv_lines <- c(csv_lines,
                   write_block(nm, "CappedSpread", tpp_cs, fdp_cs,
                               sd_tpp_cs, sd_fdp_cs),
                   write_block(nm, "Random", tpp_rand, fdp_rand,
                               sd_tpp_rand, sd_fdp_rand))
  }
  writeLines(csv_lines, csv_path)
  cat(sprintf("[Info] CSV saved to: %s\n\n", csv_path))
  invisible(NULL)
}
