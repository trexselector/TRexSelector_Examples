# ==============================================================================
# demo_ts_perf_03_memory_vs_np.R
# ==============================================================================
#
# Solver memory vs. the ratio n/p, for all demo T-solvers.
#
# Mirrors cpp/tsolvers/ad/demo_ts_perf_03_memory_vs_np/demo_ts_perf_03_memory_vs_np.cpp:
# p = 1,000, L = p, n in {50, 100, 200, 500, 1,000, 2,000}, 10 true
# coefficients equal to 3 on a fixed random support, noise sigma about 1,
# T = 10. Per run and n the data are drawn and scaled once, and every solver
# runs on them as a precentered reader. The solver memory is the library's
# memory report $get_memory_bytes() after T = 10, the heap bytes the C++ solver
# holds without the borrowed X and D (the binding and the R session are not
# counted). Unlike the peak resident set of a spawned child, the report is
# portable (no fork, no getrusage()) and deterministic for given data, so the
# runs go to worker processes with single-threaded solvers.
#
# Usage: Rscript demo_ts_perf_03_memory_vs_np.R [num_MC] [workers]   (default 500, 6)
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
p <- 1000L
num_dummies <- p
num_active <- 10L
beta_active <- 3.0
snr <- 90.0                   # signal variance 10 * 3^2 over noise variance 1
T_stop <- 10L
support_seed <- 12345L
data_seed_offset <- 1000L     # run mc uses seed data_seed_offset + mc
n_grid <- c(50L, 100L, 200L, 500L, 1000L, 2000L)
BYTES_PER_MIB <- 1024.0 * 1024.0

SUPPORT <- draw_support(p, num_active, support_seed)
COEFS <- rep(beta_active, num_active)


#' One run: every solver on the data of every n. Returns an S x K x 2 array of
#' memory in MiB and steps, NA where the solver fails.
run_one <- function(mc) {
  seed <- data_seed_offset + mc
  solvers <- demo_solvers()
  records <- array(NA_real_, dim = c(length(solvers), length(n_grid), 2L))
  for (k in seq_along(n_grid)) {
    data <- reader_data(n_grid[k], p, num_dummies, SUPPORT, COEFS, snr, seed = seed)
    for (s in seq_along(solvers)) {
      records[s, k, ] <- tryCatch({
        solver <- solvers[[s]]$make(data$X, data$D, data$y)
        solver$set_tie_seed(seed)
        solver$execute_step(T_stop, early_stop = TRUE)
        c(solver$get_memory_bytes() / BYTES_PER_MIB, solver$get_num_steps())
      }, error = function(e) c(NA_real_, NA_real_))  # a failing solver is left out of its medians
    }
  }
  records
}


# ==============================================================================
# Main
# ==============================================================================

args <- commandArgs(trailingOnly = TRUE)
num_MC <- parse_positive(args, 1L, 500L)
workers <- parse_positive(args, 2L, 6L)
solvers <- demo_solvers()
S <- length(solvers)
K <- length(n_grid)

cat("\n")
print_section_header("T-Solver Performance: Memory vs. n/p")
cat(sprintf("p = %d, L = p, n in {50, 100, 200, 500, 1000, 2000}, T = %d, true coefficients: %d x %s, SNR = %s\n",
            p, T_stop, num_active, .num(beta_active), .num(snr)))
cat(sprintf("Solver memory: held heap bytes after T = %d (get_memory_bytes; X, D borrowed, not counted)\n", T_stop))
cat(sprintf("%d solvers, %d runs on %d worker processes (single-threaded solvers)\n\n", S, num_MC, workers))

t_start <- now()
runs <- run_parallel(run_one, seq_len(num_MC) - 1L, workers)
wall <- now() - t_start

#' Values of field f (1 memory, 2 steps) of solver s at the k-th n over the runs in which it finished.
column <- function(s, k, f) {
  values <- vapply(runs, function(run) run[s, k, f], numeric(1))
  values[!is.na(values)]
}
failures <- sum(vapply(runs, function(run) sum(is.na(run[, , 1L])), integer(1)))

# Data size per n: X, D and y as doubles
data_mib <- 8.0 * n_grid * (p + num_dummies + 1) / BYTES_PER_MIB

header <- function(title) {
  c(title, paste0(sprintf("%-22s", "solver"), paste(sprintf("%9s", paste0("n=", n_grid)), collapse = "")),
    strrep("-", 22L + 9L * K))
}

out <- c("", header("Data X, D and y (MiB)"),
         paste0(sprintf("%-22s", "all solvers"), paste(sprintf("%9.2f", data_mib), collapse = "")), "")

table_block <- function(title, value, digits) {
  rows <- vapply(seq_len(S), function(s) {
    values <- vapply(seq_len(K), function(k) value(s, k), numeric(1))
    paste0(sprintf("%-22s", solvers[[s]]$name), paste(fmt_fixed(values, 9L, digits), collapse = ""))
  }, character(1))
  c(header(title), rows, "")
}

median_memory <- function(s, k) stats::median(column(s, k, 1L))
largest <- function(s, k) if (length(column(s, k, 1L)) > 0L) max(column(s, k, 1L)) else 0.0
median_steps <- function(s, k) stats::median(column(s, k, 2L))

out <- c(out, table_block("Median solver memory: held heap after T = 10 (MiB)", median_memory, 2L))
out <- c(out, table_block("Largest solver memory over the runs: held heap after T = 10 (MiB)", largest, 2L))
out <- c(out, table_block("Median solver memory relative to the data X, D and y (%)",
                          function(s, k) 100.0 * median_memory(s, k) / data_mib[k], 1L))
out <- c(out, table_block("Median steps to T = 10", median_steps, 1L))
out <- c(out, sprintf("Failed solver runs: %d", failures), sprintf("Wall time: %.1f s", wall))
text <- paste0(paste(out, collapse = "\n"), "\n")
cat(text)

# Suffixed for a TREX_SOLVERS selection
stem <- paste0(sprintf("memory_vs_np_mc%d", num_MC), solver_file_suffix())
data_dir <- output_dir(this_dir_)
csv_path <- file.path(data_dir, paste0(stem, ".csv"))
lines <- "solver,family,n,n_over_p,data_mib,solver_mib,solver_max_mib,steps"
for (s in seq_len(S)) {
  for (k in seq_len(K)) {
    row <- c(n_grid[k] / p, data_mib[k], median_memory(s, k), largest(s, k), median_steps(s, k))
    lines <- c(lines, paste(c(solvers[[s]]$name, solvers[[s]]$family, n_grid[k], fmt_g(row)), collapse = ","))
  }
}
writeLines(lines, csv_path)
txt_path <- file.path(data_dir, paste0(stem, ".txt"))
cat(text, file = txt_path)

cat(sprintf("\nResults written to:\n  %s\n  %s\n\n", display_path(csv_path), display_path(txt_path)))
