# ==============================================================================
# ts_perf_utils.R
# ==============================================================================
#
# Shared helpers for the performance demos (demo_ts_perf_*), mirroring the
# corresponding parts of cpp/tsolvers/ts_perf_utils.hpp: the fixed random
# support, reader-ready data, the selection read from the coefficients,
# positive integer arguments and the worker processes.
#
# The memory demo reads the solvers' own memory report ($get_memory_bytes())
# instead of a peak resident set, so no helper here builds a getrusage() reader
# with R CMD SHLIB: portable to Windows, and deterministic for given data.
#
# Indices are 1-based. The seeds keep the values of the C++ demos, while R's
# generator draws other numbers from them.
#
# ==============================================================================

#' Fixed random support: s of the p indices, sorted (R draw of the C++ std::sample).
draw_support <- function(p, s, seed) {
  set.seed(seed)
  sort(sample.int(p, s))
}

#' X and D centered and scaled in place to unit L2 norm, for precentered readers.
.center_scale <- function(M) {
  MLLpNormScaler$new(norm_type = 2, center = TRUE)$fit_transform_inplace(M)
}

#' X, D and y drawn once and prepared for precentered readers: X and D
#' centered and scaled to unit L2 norm, y centered.
reader_data <- function(n, p, num_dummies, support, coefs, snr, seed) {
  dat <- gen_synthetic_data(n, p, num_dummies, support, coefs, snr, seed = seed)
  X <- .center_scale(dat$X)
  D <- if (num_dummies > 0L) .center_scale(dat$D) else dat$D
  list(X = X, D = D, y = dat$y - mean(dat$y))
}

#' Selected original variables: nonzero coefficients among the first p, as the
#' T-Rex selector reads them (T-Stagewise keeps the coefficient of a variable
#' it removes from the active set, so the active set can be smaller).
selected_predictors <- function(solver) which(solver$get_beta() != 0)

#' Included dummies: nonzero coefficients beyond the first p, the count that stops a T-solver at T (the dummies
#' of the support, as the T-Rex selector counts them). The active set can hold fewer: T-Stagewise keeps the
#' coefficient of a dummy it drops.
included_dummies <- function(solver) {
  path <- solver$get_beta_path()
  if (ncol(path) == 0L) return(0L)
  sum(path[solver$get_dummy_start_index():nrow(path), ncol(path)] != 0)
}

#' Wall-clock time in seconds, resolved to about a microsecond.
now <- function() as.numeric(Sys.time())

#' Standard error of the mean (Bessel-corrected sd over sqrt(size)).
std_error <- function(v) if (length(v) > 1L) stats::sd(v) / sqrt(length(v)) else 0

#' Numbers as C's %.6g, with "nan" for the mean or median of no values as in the C++ files.
fmt_g <- function(v) ifelse(is.na(v), "nan", sprintf("%.6g", v))

#' Numbers right-aligned in a column of `width` with `digits` decimals, "nan" for no values.
fmt_fixed <- function(v, width, digits) {
  ifelse(is.na(v), formatC("nan", width = width), sprintf(paste0("%", width, ".", digits, "f"), v))
}

#' Optional positive integer argument args[index], or fallback when it is missing.
parse_positive <- function(args, index, fallback) {
  if (length(args) < index) return(fallback)
  text <- args[[index]]
  if (!grepl("^[0-9]+$", text) || as.integer(text) < 1L) stop("expected a positive integer, got: ", text)
  as.integer(text)
}

#' Single-threaded solvers in this process (6-core rule: the parallelism is in the workers).
single_thread <- function() trex_set_num_threads(1L)

#' fun over tasks (and ...) on `workers` spawned R processes (a PSOCK cluster, no
#' fork) with single-threaded solvers, results in task order. Every worker loads
#' the package, receives the globals of the demo script and sets one thread, as
#' the initializer of the Python pool does. `chunk` tasks travel together.
run_parallel <- function(fun, tasks, workers, ..., chunk = 1L) {
  single_thread()
  if (workers == 1L) return(lapply(tasks, fun, ...))
  cl <- parallel::makePSOCKcluster(workers)
  on.exit(parallel::stopCluster(cl), add = TRUE)
  parallel::clusterExport(cl, ls(globalenv(), all.names = TRUE), envir = globalenv())
  parallel::clusterEvalQ(cl, {
    suppressMessages(library(TRexSelector))
    single_thread()
  })
  chunks <- split(tasks, ceiling(seq_along(tasks) / chunk))
  out <- parallel::parLapplyLB(cl, chunks, function(part, ...) lapply(part, fun, ...), ...)
  unlist(out, recursive = FALSE, use.names = FALSE)
}
