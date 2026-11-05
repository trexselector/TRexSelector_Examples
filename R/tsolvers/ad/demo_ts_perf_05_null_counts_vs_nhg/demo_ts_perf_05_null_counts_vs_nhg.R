# ==============================================================================
# demo_ts_perf_05_null_counts_vs_nhg.R
# ==============================================================================
#
# Number of included and selected null variables vs. the negative hypergeometric
# (NHG) distribution, for all demo T-solvers (numerical verification of
# Corollary 2, A-I and A-II of the T-Rex supplement, Figures 10 and 11).
#
# Mirrors cpp/tsolvers/ad/demo_ts_perf_05_null_counts_vs_nhg/demo_ts_perf_05_null_counts_vs_nhg.cpp:
# n = 150, p = 300, 5 true coefficients equal to 1 on a fixed random support,
# SNR = 1. Every run draws one data set and K = 20 random experiments, each with
# L = p fresh standard normal dummies, and runs every solver to T = 20 included
# dummies. The included null variables of one experiment follow
# NHG(p0 + L, p0, T) when nulls and dummies are exchangeable. Their relative
# occurrences over the K experiments give the selected null variables V(v) at
# the voting levels v -> 1, 0.75 and 0.5, which the NHG should dominate.
# Indices are 1-based.
#
# Usage: Rscript demo_ts_perf_05_null_counts_vs_nhg.R [num_MC] [workers]   (default 500, 6)
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
n <- 150L
p <- 300L
num_active <- 5L
p0 <- p - num_active          # null variables
num_dummies <- p              # L = p per random experiment
K <- 20L                      # random experiments
T_stop <- 20L                 # included dummies
beta_active <- 1.0
snr <- 1.0
support_seed <- 12345L
data_seed_offset <- 1000L     # run mc uses seed data_seed_offset + mc

# Voting levels v -> 1 (all K experiments), 0.75 and 0.5 as minimum occurrence counts (Phi > v)
LEVEL_NAMES <- c("v1", "v0.75", "v0.5")
LEVEL_MIN_COUNTS <- c(K, 16L, 11L)

SUPPORT <- draw_support(p, num_active, support_seed)
COEFS <- rep(beta_active, num_active)
NULLS <- setdiff(seq_len(p), SUPPORT)


#' PMF of the number of nulls drawn before the T-th dummy, NHG(p0 + L, p0, T), for 0, ..., p0.
nhg_pmf <- function() {
  v <- 0:p0
  exp(lchoose(v + T_stop - 1, v) + lchoose(p0 + num_dummies - v - T_stop, p0 - v) - lchoose(p0 + num_dummies, p0))
}

#' One solver over the K experiments: included nulls per experiment, selected nulls per level and the
#' share of experiments that reached T dummies, NULL when it fails in one of them.
run_solver <- function(spec, data, seed) {
  tryCatch({
    occurrences <- integer(p)
    included <- integer(K)
    reached <- 0
    for (k in seq_len(K)) {
      D <- data$D_all[, (k - 1L) * num_dummies + seq_len(num_dummies)]
      solver <- spec$make(data$X, D, data$y)
      solver$set_tie_seed(seed * K + k - 1L)
      solver$execute_step(T_stop, early_stop = TRUE)
      selected <- selected_predictors(solver)
      occurrences[selected] <- occurrences[selected] + 1L
      included[k] <- sum(!selected %in% SUPPORT)
      reached <- reached + as.numeric(included_dummies(solver) >= T_stop)
    }
    selected_nulls <- vapply(LEVEL_MIN_COUNTS, function(count) sum(occurrences[NULLS] >= count), integer(1L))
    list(included = included, selected = selected_nulls, reached = reached / K)
  }, error = function(e) NULL)
}

#' One run: one data set with K * L dummies, block k for experiment k, every solver to T.
run_one <- function(mc) {
  seed <- data_seed_offset + mc
  dat <- gen_synthetic_data(n, p, K * num_dummies, SUPPORT, COEFS, snr, seed = seed)
  # Center and scale once in front (column-wise, so the blocks stay independent)
  data <- list(X = .center_scale(dat$X), D_all = .center_scale(dat$D), y = dat$y - mean(dat$y))
  lapply(demo_solvers(), run_solver, data = data, seed = seed)
}

#' Frequencies of the counts 0, ..., p0.
frequencies <- function(values) tabulate(values + 1L, nbins = p0 + 1L)

#' Mean and standard deviation of a histogram over 0, ..., p0.
moments <- function(weights) {
  total <- sum(weights)
  if (total <= 0) return(c(NaN, NaN))
  x <- seq_along(weights) - 1
  mean <- sum(weights * x) / total
  c(mean, sqrt(max(0, sum(weights * x * x) / total - mean * mean)))
}

#' Empirical CDF over 0, ..., p0 from frequencies.
ecdf_of <- function(freq) if (sum(freq) > 0) cumsum(freq) / sum(freq) else rep(NaN, length(freq))


# ==============================================================================
# Main
# ==============================================================================

args <- commandArgs(trailingOnly = TRUE)
num_MC <- parse_positive(args, 1L, 500L)
workers <- parse_positive(args, 2L, 6L)
solvers <- demo_solvers()
S <- length(solvers)
pmf <- nhg_pmf()
nhg_cdf <- cumsum(pmf)
nhg_moments <- moments(pmf)

cat("\n")
print_section_header("T-Solver Performance: Null Counts vs. NHG")
cat(sprintf("n = %d, p = %d, p0 = %d, L = p = %d, true coefficients: %d x %s, SNR = %s\n",
            n, p, p0, num_dummies, num_active, .num(beta_active), .num(snr)))
cat(sprintf("K = %d random experiments, T = %d, voting levels v -> 1, 0.75, 0.5\n", K, T_stop))
cat(sprintf("NHG(p0 + L, p0, T): mean T p0 / (L + 1) = %.2f, sd = %.2f\n", nhg_moments[1L], nhg_moments[2L]))
cat("True support: ", paste(SUPPORT, collapse = " "), "\n", sep = "")
cat(sprintf("%d solvers, %d runs on %d worker processes (single-threaded solvers)\n\n", S, num_MC, workers))

t_start <- now()
runs <- run_parallel(run_one, seq_len(num_MC) - 1L, workers)
wall <- now() - t_start

failures <- sum(vapply(runs, function(results) sum(vapply(results, is.null, logical(1L))), integer(1L)))
freq_included <- vector("list", S)
freq_selected <- vector("list", S)
reached <- numeric(S)
for (s in seq_len(S)) {
  ok <- Filter(Negate(is.null), lapply(runs, `[[`, s))
  freq_included[[s]] <- frequencies(unlist(lapply(ok, `[[`, "included")))
  freq_selected[[s]] <- lapply(seq_along(LEVEL_NAMES), function(lv) {
    frequencies(vapply(ok, function(r) r$selected[lv], integer(1L)))
  })
  reached[s] <- if (length(ok) > 0L) mean(vapply(ok, `[[`, numeric(1L), "reached")) else NaN
}

out <- c(sprintf("Included null variables at T = %d (pooled over runs and experiments)", T_stop),
         sprintf("%-22s%8s%8s%8s%10s", "solver", "mean", "sd", "KS D", "reached"),
         strrep("-", 22L + 8L * 3L + 10L),
         sprintf("%-22s%8.2f%8.2f", "NHG", nhg_moments[1L], nhg_moments[2L]))
for (s in seq_len(S)) {
  m <- moments(freq_included[[s]])
  ks <- max(abs(ecdf_of(freq_included[[s]]) - nhg_cdf))
  out <- c(out, sprintf("%-22s%8.2f%8.2f%8.3f%8.1f %%", solvers[[s]]$name, m[1L], m[2L], ks, 100 * reached[s]))
}
# Stochastic dominance: excess = max over x of F_NHG(x) - F_V(x), positive where the NHG CDF lies above
out <- c(out, "",
         "Selected null variables V(v) over the runs: mean and largest excess of the NHG CDF over the ECDF",
         paste0(sprintf("%-22s", "solver"), paste(sprintf("%13s", paste("mean", LEVEL_NAMES)), collapse = ""),
                paste(sprintf("%13s", paste("excess", LEVEL_NAMES)), collapse = "")),
         strrep("-", 22L + 13L * 2L * length(LEVEL_NAMES)))
for (s in seq_len(S)) {
  means <- vapply(freq_selected[[s]], function(f) moments(f)[1L], numeric(1L))
  excess <- vapply(freq_selected[[s]], function(f) max(0, max(nhg_cdf - ecdf_of(f))), numeric(1L))
  out <- c(out, paste0(sprintf("%-22s", solvers[[s]]$name), paste(sprintf("%13.2f", means), collapse = ""),
                       paste(sprintf("%13.3f", excess), collapse = "")))
}
out <- c(out, "",
         sprintf("Failed solver runs: %d", failures),
         sprintf("Wall time: %.1f s", wall))
text <- paste0(paste(out, collapse = "\n"), "\n")
cat(text)

# Result files: frequencies of the counts, the NHG reference is computed by the plotter
# Suffixed for a TREX_SOLVERS selection
stem <- paste0(sprintf("null_counts_vs_nhg_mc%d", num_MC), solver_file_suffix())
data_dir <- output_dir(this_dir_)
csv_path <- file.path(data_dir, paste0(stem, ".csv"))
lines <- "solver,family,p0,L,T,K,statistic,count,frequency"
rows <- function(spec, statistic, freq) {
  counts <- which(freq > 0L) - 1L
  sprintf("%s,%s,%d,%d,%d,%d,%s,%d,%d", spec$name, spec$family, p0, num_dummies, T_stop, K, statistic, counts,
          freq[counts + 1L])
}
for (s in seq_len(S)) {
  lines <- c(lines, rows(solvers[[s]], "included", freq_included[[s]]))
  for (lv in seq_along(LEVEL_NAMES)) {
    lines <- c(lines, rows(solvers[[s]], paste0("selected_", LEVEL_NAMES[lv]), freq_selected[[s]][[lv]]))
  }
}
writeLines(lines, csv_path)
txt_path <- file.path(data_dir, paste0(stem, ".txt"))
cat(text, file = txt_path)

cat(sprintf("\nResults written to:\n  %s\n  %s\n\n", display_path(csv_path), display_path(txt_path)))
