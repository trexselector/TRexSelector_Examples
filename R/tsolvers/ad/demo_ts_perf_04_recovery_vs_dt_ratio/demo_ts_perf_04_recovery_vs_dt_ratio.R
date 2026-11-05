# ==============================================================================
# demo_ts_perf_04_recovery_vs_dt_ratio.R
# ==============================================================================
#
# Support recovery of the T-solvers vs. the ratio n/(s log p), and the
# Donoho-Tanner phase transition of T-LARS, T-OMP and T-AFS.
#
# Mirrors cpp/tsolvers/ad/demo_ts_perf_04_recovery_vs_dt_ratio/demo_ts_perf_04_recovery_vs_dt_ratio.cpp.
# Exact recovery: the first s distinct variables that enter the path are the
# true support (a dummy among them counts as a miss). Every solver runs as a
# precentered reader to the first dummy (T = 1), which also gives TPR and FDR
# at T = 1. L = p, coefficients equal to 3 on a fixed random support, noise
# sigma about 1. Indices are 1-based.
#
#   Part 1: p in {250, 500, 1,000, 2,000, 5,000}, s = 10, n = n/(s log p) * s log p
#           for n/(s log p) in {0.25, 0.5, 1, 1.5, 2, 3, 4}, all demo solvers.
#   Part 2: p = 400, 20 x 20 grid of n/p and s/n in {0.05, ..., 1},
#           T-LARS, T-OMP and T-AFS (rho 0.3).
#
# Usage: Rscript demo_ts_perf_04_recovery_vs_dt_ratio.R [num_MC] [workers] [parts...]   (default 500, 6, all)
#
# ==============================================================================

library(TRexSelector)

this_dir_ <- tryCatch(
  dirname(normalizePath(sys.frame(1)$ofile)),
  error = function(e) {
    args <- commandArgs(trailingOnly = FALSE)
    flag <- grep("^--file=", args, value = TRUE)
    if (length(flag) > 0L)
      dirname(normalizePath(sub("^--file=", "", flag[1L])))
    else "."
  }
)

source(file.path(this_dir_, "..", "..", "ts_demo_utils.R"))
source(file.path(this_dir_, "..", "..", "ts_perf_utils.R"))
source(file.path(this_dir_, "..", "..", "ts_perf_solvers.R"))

beta_active <- 3.0
support_seed <- 12345L
data_seed_offset <- 1000L        # Part 1, run mc: seed data_seed_offset + mc
phase_seed_offset <- 3000000L    # Part 2, run mc and cell c: seed offset + mc * cells + c

# Part 1
s_curves <- 10L
p_grid <- c(250L, 500L, 1000L, 2000L, 5000L)
ratio_grid <- c(0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0)

# Part 2
p_phase <- 400L
phase_steps <- 20L  # n/p and s/n in {1, ..., 20} / 20
# Only the ones TREX_SOLVERS selects (all without a selection)
phase_solvers <- Filter(solver_selected, c("T-LARS", "T-OMP", "T-AFS (rho 0.3)"))


#' Number of observations for p and the ratio n/(s log p), natural log.
n_for <- function(p, ratio) as.integer(round(ratio * s_curves * log(p)))

#' TRUE when the first distinct variables entering the path are the support.
first_entries_exact <- function(solver, support) {
  seen <- integer()
  for (a in unlist(solver$get_actions())) {
    if (a <= 0L || a %in% seen) next  # a drop, or a variable that entered before
    seen <- c(seen, a)
    if (!(a %in% support)) return(FALSE)
    if (length(seen) == length(support)) return(TRUE)
  }
  FALSE  # the path ended before s variables entered
}

#' Draw one data set, scale it and run `specs` to the first dummy: one row
#' (exact, tpp, fdp, reached, ok) per spec.
run_problem <- function(n, p, support, seed, specs) {
  coefs <- rep(beta_active, length(support))
  snr <- length(support) * beta_active^2  # noise sigma 1
  data <- reader_data(n, p, p, support, coefs, snr, seed = seed)
  rows <- lapply(specs, function(spec) {
    tryCatch({
      solver <- spec$make(data$X, data$D, data$y)
      solver$set_tie_seed(seed)
      solver$execute_step(1L, early_stop = TRUE)
      selected <- selected_predictors(solver)
      c(as.numeric(first_entries_exact(solver, support)), trex_tpp(selected, support),
        trex_fdp(selected, support), as.numeric(included_dummies(solver) > 0L), 1)
    }, error = function(e) c(0, 0, 0, 0, 0))
  })
  do.call(rbind, rows)
}

#' Results file paths, relative to the repository, after the tables.
print_written <- function(csv_path, txt_path) {
  cat(sprintf("\nResults written to:\n  %s\n  %s\n\n", display_path(csv_path), display_path(txt_path)))
}


# ------------------------------------------------------------------------------
# Part 1
# ------------------------------------------------------------------------------

SUPPORTS <- lapply(p_grid, draw_support, s = s_curves, seed = support_seed)

#' Task t (0-based, configuration-major as in C++): configuration c = t %/% num_MC, run mc = t %% num_MC.
part1_task <- function(t, num_MC) {
  c <- t %/% num_MC
  mc <- t %% num_MC
  i <- c %/% length(ratio_grid) + 1L
  j <- c %% length(ratio_grid) + 1L
  run_problem(n_for(p_grid[i], ratio_grid[j]), p_grid[i], SUPPORTS[[i]], data_seed_offset + mc, demo_solvers())
}

run_part1 <- function(num_MC, workers) {
  print_section_header("Part 1: Recovery vs. n/(s log p)")
  solvers <- demo_solvers()
  S <- length(solvers)
  P <- length(p_grid)
  R <- length(ratio_grid)
  C <- P * R
  cat(sprintf(paste0("p in {250, 500, 1000, 2000, 5000}, L = p, s = %d, n/(s log p) in {0.25, 0.5, 1, 1.5, 2, 3, 4}, ",
                     "T = 1\n%d solvers, %d runs\n\n"), s_curves, S, num_MC))

  t_start <- now()
  flat <- run_parallel(part1_task, seq_len(C * num_MC) - 1L, workers, num_MC = num_MC)
  wall <- now() - t_start
  outcomes <- lapply(seq_len(C), function(c) flat[(c - 1L) * num_MC + seq_len(num_MC)])
  failures <- sum(vapply(flat, function(m) sum(m[, 5L] == 0), numeric(1)))

  column <- function(c, s, field) {
    values <- vapply(outcomes[[c]], function(m) if (m[s, 5L] == 1) m[s, field] else NA_real_, numeric(1))
    values[!is.na(values)]
  }

  out <- character()
  table_block <- function(title, i, field) {
    p <- p_grid[i]
    rows <- vapply(seq_len(S), function(s) {
      values <- vapply(seq_len(R), function(j) 100.0 * mean(column((i - 1L) * R + j, s, field)), numeric(1))
      paste0(sprintf("%-22s", solvers[[s]]$name), paste(fmt_fixed(values, 8L, 1L), collapse = ""))
    }, character(1))
    c(sprintf("%s, p = %d", title, p),
      paste0(sprintf("%-22s", "n/(s log p)"), paste(sprintf("%8g", ratio_grid), collapse = "")),
      paste0(sprintf("%-22s", "n"), paste(sprintf("%8d", n_for(p, ratio_grid)), collapse = "")),
      strrep("-", 22L + 8L * R), rows, "")
  }

  for (i in seq_len(P)) {
    out <- c(out, table_block("Exact recovery by the first s = 10 entries (%)", i, 1L))
    out <- c(out, table_block("TPR at T = 1 (%)", i, 2L))
    out <- c(out, table_block("FDR at T = 1 (%)", i, 3L))
  }
  out <- c(out, sprintf("Failed solver runs: %d", failures), sprintf("Wall time: %.1f s", wall))
  text <- paste0(paste(out, collapse = "\n"), "\n")
  cat(text)

  # Suffixed for a TREX_SOLVERS selection
  stem <- paste0(sprintf("recovery_vs_dt_ratio_mc%d", num_MC), solver_file_suffix())
  data_dir <- output_dir(this_dir_)
  csv_path <- file.path(data_dir, paste0(stem, ".csv"))
  lines <- "solver,family,p,s,n,ratio,exact,exact_se,TPR,TPR_se,FDR,FDR_se,reached"
  for (s in seq_len(S)) {
    for (i in seq_len(P)) {
      for (j in seq_len(R)) {
        c <- (i - 1L) * R + j
        ex <- column(c, s, 1L)
        tp <- column(c, s, 2L)
        fd <- column(c, s, 3L)
        row <- c(mean(ex), std_error(ex), mean(tp), std_error(tp), mean(fd), std_error(fd), mean(column(c, s, 4L)))
        lines <- c(lines, paste(c(solvers[[s]]$name, solvers[[s]]$family, p_grid[i], s_curves,
                                  n_for(p_grid[i], ratio_grid[j]), sprintf("%g", ratio_grid[j]), fmt_g(row)),
                                collapse = ","))
      }
    }
  }
  writeLines(lines, csv_path)
  txt_path <- file.path(data_dir, paste0(stem, ".txt"))
  cat(text, file = txt_path)
  print_written(csv_path, txt_path)
}


# ------------------------------------------------------------------------------
# Part 2
# ------------------------------------------------------------------------------

#' Observations of grid column a (0-based) and true variables of grid cell (a, b).
n_of <- function(a) (a + 1L) * p_phase %/% phase_steps
s_of <- function(a, b) (b + 1L) * n_of(a) %/% phase_steps

CELLS <- phase_steps * phase_steps
PHASE_SUPPORTS <- lapply(seq_len(CELLS) - 1L, function(c) {
  draw_support(p_phase, s_of(c %/% phase_steps, c %% phase_steps), support_seed + c)
})

#' Task t (0-based, run-major as in C++): run mc = t %/% CELLS, cell c = t %% CELLS.
#' Returns 1 (exact), 0 (miss) or -1 (failed) per phase solver.
part2_task <- function(t) {
  mc <- t %/% CELLS
  c <- t %% CELLS
  specs <- Filter(function(spec) spec$name %in% phase_solvers, demo_solvers())
  specs <- specs[match(phase_solvers, vapply(specs, `[[`, character(1), "name"))]
  seed <- phase_seed_offset + mc * CELLS + c
  m <- run_problem(n_of(c %/% phase_steps), p_phase, PHASE_SUPPORTS[[c + 1L]], seed, specs)
  as.integer(ifelse(m[, 5L] == 1, m[, 1L], -1))
}

run_part2 <- function(num_MC, workers) {
  print_section_header("Part 2: Donoho-Tanner phase transition")
  if (length(phase_solvers) == 0L) {
    cat("TREX_SOLVERS selects none of its solvers, Part 2 skipped\n\n")
    return(invisible(NULL))
  }
  S <- length(phase_solvers)
  G <- phase_steps
  cat(sprintf("p = %d, L = p, n/p and s/n in {0.05, 0.10, ..., 1}, T = 1\n", p_phase))
  cat(sprintf("T-LARS, T-OMP and T-AFS (rho 0.3), %d runs per cell\n\n", num_MC))

  t_start <- now()
  flat <- run_parallel(part2_task, seq_len(num_MC * CELLS) - 1L, workers, chunk = 64L)
  wall <- now() - t_start
  exact <- array(unlist(flat), dim = c(S, CELLS, num_MC))  # 1 exact, 0 miss, -1 failed
  failures <- sum(exact < 0L)

  cell_values <- function(c, k) {
    v <- exact[k, c, ]
    as.numeric(v[v >= 0L])
  }

  out <- character()
  for (k in seq_len(S)) {
    rows <- vapply(rev(seq_len(G)), function(b) {
      values <- vapply(seq_len(G), function(a) 100.0 * mean(cell_values((a - 1L) * G + b, k)), numeric(1))
      paste0(sprintf("%10.2f", b / G), paste(fmt_fixed(values, 5L, 0L), collapse = ""))
    }, character(1))
    out <- c(out, sprintf("Exact recovery by the first s entries (%%), %s", phase_solvers[k]),
             paste0(sprintf("%10s", "s/n \\ n/p"), paste(sprintf("%5.2f", seq_len(G) / G), collapse = "")),
             strrep("-", 10L + 5L * G), rows, "")
  }
  out <- c(out, sprintf("Failed solver runs: %d", failures), sprintf("Wall time: %.1f s", wall))
  text <- paste0(paste(out, collapse = "\n"), "\n")
  cat(text)

  families <- vapply(demo_solvers(), `[[`, character(1), "family")
  names(families) <- vapply(demo_solvers(), `[[`, character(1), "name")
  # Suffixed for a TREX_SOLVERS selection
  stem <- paste0(sprintf("phase_transition_mc%d", num_MC), solver_file_suffix())
  data_dir <- output_dir(this_dir_)
  csv_path <- file.path(data_dir, paste0(stem, ".csv"))
  lines <- "solver,family,p,n,s,n_over_p,s_over_n,exact,exact_se"
  for (k in seq_len(S)) {
    for (c in seq_len(CELLS)) {
      a <- (c - 1L) %/% G
      b <- (c - 1L) %% G
      v <- cell_values(c, k)
      lines <- c(lines, paste(c(phase_solvers[k], families[[phase_solvers[k]]], p_phase, n_of(a), s_of(a, b),
                                fmt_g(c((a + 1) / G, (b + 1) / G, mean(v), std_error(v)))), collapse = ","))
    }
  }
  writeLines(lines, csv_path)
  txt_path <- file.path(data_dir, paste0(stem, ".txt"))
  cat(text, file = txt_path)
  print_written(csv_path, txt_path)
}


# ==============================================================================
# Main
# ==============================================================================

args <- commandArgs(trailingOnly = TRUE)
num_MC <- parse_positive(args, 1L, 500L)
workers <- parse_positive(args, 2L, 6L)
run <- part_selection(args[-(1:2)])
cat("\n")
print_section_header("T-Solver Performance: Recovery vs. n/(s log p)")
cat(sprintf("%d runs on %d worker processes\n\n", num_MC, workers))
if (run(1)) run_part1(num_MC, workers)
if (run(2)) run_part2(num_MC, workers)
