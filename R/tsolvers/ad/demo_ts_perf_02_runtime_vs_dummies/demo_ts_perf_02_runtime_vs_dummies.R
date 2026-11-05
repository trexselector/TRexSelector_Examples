# ==============================================================================
# demo_ts_perf_02_runtime_vs_dummies.R
# ==============================================================================
#
# Run time of the terminated solution paths vs. the number of dummies L, for
# all demo T-solvers.
#
# Mirrors cpp/tsolvers/ad/demo_ts_perf_02_runtime_vs_dummies/demo_ts_perf_02_runtime_vs_dummies.cpp:
# n = 500, p = 1,000, 10 true coefficients equal to 3 on a fixed random support,
# noise sigma about 1, T = 10 and L in {1, 2, 5, 10, 20} x p. Every run draws
# 20p dummies once, and each L uses the first L of them (nested dummy sets,
# copied into their own matrix before the clock starts). The runs are spread
# over the worker processes, and each solver runs single-threaded inside its
# run. Per L and solver: the time to T with its steps, the warm start
# T -> T + 1, and a fresh run to T + 1.
#
# Usage: Rscript demo_ts_perf_02_runtime_vs_dummies.R [num_MC] [workers]   (default 500, 6)
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

# Setup
n <- 500L
p <- 1000L
num_active <- 10L
beta_active <- 3.0
snr <- 90.0                   # signal variance 10 * 3^2 over noise variance 1
T_stop <- 10L
support_seed <- 12345L
data_seed_offset <- 1000L     # run mc uses seed data_seed_offset + mc
L_factors <- c(1L, 2L, 5L, 10L, 20L)
FIELDS <- c("ms", "steps", "warm_ms", "fresh_ms")

SUPPORT <- draw_support(p, num_active, support_seed)
COEFS <- rep(beta_active, num_active)


#' One solver at every L: one row of FIELDS per L, NULL when it fails (the whole run is left out).
run_solver <- function(spec, data, D_sets, seed) {
  tryCatch({
    r <- matrix(NA_real_, nrow = length(L_factors), ncol = length(FIELDS), dimnames = list(NULL, FIELDS))
    for (k in seq_along(L_factors)) {
      D <- D_sets[[k]]

      # Path to T, then a warm start to T + 1
      t0 <- now()
      solver <- spec$make(data$X, D, data$y)
      solver$set_tie_seed(seed)
      solver$execute_step(T_stop, early_stop = TRUE)
      to_T <- 1e3 * (now() - t0)
      steps <- solver$get_num_steps()
      t1 <- now()
      solver$execute_step(T_stop + 1L, early_stop = TRUE)
      warm <- 1e3 * (now() - t1)

      # A fresh run to T + 1 for comparison
      t2 <- now()
      fresh <- spec$make(data$X, D, data$y)
      fresh$set_tie_seed(seed)
      fresh$execute_step(T_stop + 1L, early_stop = TRUE)
      fresh_ms <- 1e3 * (now() - t2)

      r[k, ] <- c(to_T, steps, warm, fresh_ms)
    }
    r
  }, error = function(e) NULL)
}

#' One run: every solver at every L on the same nested dummy sets.
run_one <- function(mc) {
  seed <- data_seed_offset + mc
  data <- reader_data(n, p, L_factors[length(L_factors)] * p, SUPPORT, COEFS, snr, seed = seed)
  # The first L dummy columns per L, copied once here and outside the timings
  D_sets <- lapply(L_factors, function(factor) data$D[, seq_len(factor * p), drop = FALSE])
  lapply(demo_solvers(), run_solver, data = data, D_sets = D_sets, seed = seed)
}


# ==============================================================================
# Main
# ==============================================================================

args <- commandArgs(trailingOnly = TRUE)
num_MC <- parse_positive(args, 1L, 500L)
workers <- parse_positive(args, 2L, 6L)
solvers <- demo_solvers()
S <- length(solvers)
K <- length(L_factors)

cat("\n")
print_section_header("T-Solver Performance: Runtime vs. Number of Dummies")
cat(sprintf("n = %d, p = %d, L in {1, 2, 5, 10, 20} x p, T = %d, true coefficients: %d x %s, SNR = %s\n",
            n, p, T_stop, num_active, .num(beta_active), .num(snr)))
cat(sprintf("%d solvers, %d runs on %d worker processes (single-threaded solvers)\n\n", S, num_MC, workers))

t_start <- now()
runs <- run_parallel(run_one, seq_len(num_MC) - 1L, workers)
wall <- now() - t_start
failures <- sum(vapply(runs, function(run) sum(vapply(run, is.null, logical(1))), integer(1)))

#' Values of one field at the k-th L over the runs in which solver s succeeded.
column <- function(s, k, field) {
  as.numeric(unlist(lapply(runs, function(run) if (is.null(run[[s]])) NULL else run[[s]][k, field])))
}

per_step <- function(s, k) 1e3 * column(s, k, "ms") / pmax(1.0, column(s, k, "steps"))
speedup <- function(s, k) column(s, k, "fresh_ms") / pmax(1e-6, column(s, k, "warm_ms"))

out <- character()
table_block <- function(title, values, digits) {
  labels <- paste0("L=", ifelse(L_factors == 1L, "", L_factors), "p")
  header <- paste0(sprintf("%-22s", "solver"), paste(sprintf("%9s", labels), collapse = ""))
  rows <- vapply(seq_len(S), function(s) {
    medians <- vapply(seq_len(K), function(k) stats::median(values(s, k)), numeric(1))
    paste0(sprintf("%-22s", solvers[[s]]$name), paste(fmt_fixed(medians, 9L, digits), collapse = ""))
  }, character(1))
  c(title, header, strrep("-", 22L + 9L * K), rows, "")
}

out <- c(out, table_block("Median time to T = 10 (ms)", function(s, k) column(s, k, "ms"), 2L))
out <- c(out, table_block("Median steps to T = 10", function(s, k) column(s, k, "steps"), 1L))
out <- c(out, table_block("Median time per step (us)", per_step, 1L))
out <- c(out, table_block("Median warm start T = 10 -> 11 (ms)", function(s, k) column(s, k, "warm_ms"), 3L))
out <- c(out, table_block("Median fresh run to T = 11 over warm start 10 -> 11 (ratio)", speedup, 1L))
out <- c(out, sprintf("Failed solver runs: %d", failures), sprintf("Wall time: %.1f s", wall))
text <- paste0(paste(out, collapse = "\n"), "\n")
cat(text)

# Suffixed for a TREX_SOLVERS selection
stem <- paste0(sprintf("runtime_vs_dummies_mc%d", num_MC), solver_file_suffix())
data_dir <- output_dir(this_dir_)
csv_path <- file.path(data_dir, paste0(stem, ".csv"))
lines <- "solver,family,L_factor,L,ms,steps,us_per_step,warm_ms,fresh_ms,fresh_over_warm"
for (s in seq_len(S)) {
  for (k in seq_len(K)) {
    row <- c(stats::median(column(s, k, "ms")), stats::median(column(s, k, "steps")),
             stats::median(per_step(s, k)), stats::median(column(s, k, "warm_ms")),
             stats::median(column(s, k, "fresh_ms")), stats::median(speedup(s, k)))
    lines <- c(lines, paste(c(solvers[[s]]$name, solvers[[s]]$family, L_factors[k], L_factors[k] * p, fmt_g(row)),
                            collapse = ","))
  }
}
writeLines(lines, csv_path)
txt_path <- file.path(data_dir, paste0(stem, ".txt"))
cat(text, file = txt_path)

cat(sprintf("\nResults written to:\n  %s\n  %s\n\n", display_path(csv_path), display_path(txt_path)))
