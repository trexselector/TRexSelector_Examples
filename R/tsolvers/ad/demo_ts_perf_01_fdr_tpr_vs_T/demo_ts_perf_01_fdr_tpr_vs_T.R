# ==============================================================================
# demo_ts_perf_01_fdr_tpr_vs_T.R
# ==============================================================================
#
# FDR and TPR of the terminated solution paths vs. the number of included
# dummies T, for all demo T-solvers (simulation study of the R tlars vignette).
#
# Mirrors cpp/tsolvers/ad/demo_ts_perf_01_fdr_tpr_vs_T/demo_ts_perf_01_fdr_tpr_vs_T.cpp:
# n = 100, p = 300, 10 true coefficients equal to 3 on a fixed random support,
# noise sigma about 1, L = p fresh standard normal dummies per run. Each run
# scales X and D once and hands them to all solvers as readers; every solver
# is warm-started through the T grid {1, 2, 5, 10, 20, 50, 100}. FDR and TPR are
# the mean FDP and TPP over the runs; a variable counts as selected when its
# coefficient is nonzero, as in the T-Rex selector. After each T the selected
# original variables (dummies left out) are refit by least squares, and the
# refit is scored on a fresh test set of n_test rows. Indices are 1-based.
#
# Usage: Rscript demo_ts_perf_01_fdr_tpr_vs_T.R [num_MC] [workers]   (default 500, 6)
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
n <- 100L
p <- 300L
num_dummies <- p
num_active <- 10L
beta_active <- 3.0
snr <- 90.0                   # signal variance 10 * 3^2 over noise variance 1
support_seed <- 12345L
data_seed_offset <- 1000L     # run mc uses seed data_seed_offset + mc
n_test <- 1000L               # rows of the test set of every run
test_seed_offset <- 2000000L  # test set of run mc: seed test_seed_offset + mc
T_grid <- c(1L, 2L, 5L, 10L, 20L, 50L, 100L)
FIELDS <- c("fdp", "tpp", "selected", "steps", "reached", "ms", "rmse", "coef_err")

SUPPORT <- draw_support(p, num_active, support_seed)
COEFS <- rep(beta_active, num_active)
BETA_TRUE <- replace(numeric(p), SUPPORT, beta_active)


#' Minimum-norm least-squares solution (as numpy.linalg.lstsq with its default cutoff).
min_norm_lstsq <- function(A, b) {
  s <- svd(A)
  keep <- s$d > .Machine$double.eps * max(dim(A)) * s$d[1L]
  as.numeric(s$v[, keep, drop = FALSE] %*% (crossprod(s$u[, keep, drop = FALSE], b) / s$d[keep]))
}

#' Errors of the least-squares refit with intercept on the selected original
#' variables: prediction RMSE on the test set and ||beta_hat - beta||_2. The
#' minimum-norm solution also covers selections with as many variables as rows.
refit_errors <- function(X, y, X_test, y_test, selected, beta) {
  y_mean <- mean(y)
  beta_hat <- numeric(length(beta))
  intercept <- y_mean
  if (length(selected) > 0L) {
    Xs <- X[, selected, drop = FALSE]
    x_mean <- colMeans(Xs)
    b <- min_norm_lstsq(sweep(Xs, 2L, x_mean), y - y_mean)
    beta_hat[selected] <- b
    intercept <- y_mean - sum(x_mean * b)
  }
  prediction <- as.numeric(X_test %*% beta_hat) + intercept
  c(sqrt(mean((y_test - prediction)^2)), sqrt(sum((beta_hat - beta)^2)))
}

#' One solver warm-started along the T grid: one row of FIELDS per T, NULL when it fails.
run_solver <- function(spec, data, raw, test, seed) {
  tryCatch({
    r <- matrix(NA_real_, nrow = length(T_grid), ncol = length(FIELDS), dimnames = list(NULL, FIELDS))
    selections <- vector("list", length(T_grid))
    t0 <- now()
    untimed <- 0  # reached count, kept out of the times
    solver <- spec$make(data$X, data$D, data$y)
    solver$set_tie_seed(seed)
    for (k in seq_along(T_grid)) {
      solver$execute_step(T_grid[k], early_stop = TRUE)  # warm start from the previous T
      selected <- selected_predictors(solver)
      r[k, "fdp"] <- trex_fdp(selected, SUPPORT)
      r[k, "tpp"] <- trex_tpp(selected, SUPPORT)
      r[k, "selected"] <- length(selected)
      r[k, "steps"] <- solver$get_num_steps()
      c0 <- now()
      r[k, "reached"] <- as.numeric(included_dummies(solver) >= T_grid[k])
      untimed <- untimed + now() - c0
      r[k, "ms"] <- 1e3 * (now() - t0 - untimed)
      selections[[k]] <- selected
    }
    # Refits after the timed path, so they stay out of the solver times
    for (k in seq_along(T_grid)) {
      r[k, c("rmse", "coef_err")] <- refit_errors(raw$X, raw$y, test$X, test$y, selections[[k]], BETA_TRUE)
    }
    r
  }, error = function(e) NULL)
}

#' One run: all solvers warm-started along the T grid on the same data.
run_one <- function(mc) {
  seed <- data_seed_offset + mc
  dat <- gen_synthetic_data(n, p, num_dummies, SUPPORT, COEFS, snr, seed = seed)
  raw <- list(X = dat$X + 0, y = dat$y)  # deep copy, the scaler below works in place
  test <- gen_synthetic_data(n_test, p, 0L, SUPPORT, COEFS, snr, seed = test_seed_offset + mc)
  oracle <- refit_errors(raw$X, raw$y, test$X, test$y, SUPPORT, BETA_TRUE)

  # Center and scale once in front; every solver below is a precentered reader
  data <- list(X = .center_scale(dat$X), D = .center_scale(dat$D), y = dat$y - mean(dat$y))
  results <- lapply(demo_solvers(), run_solver, data = data, raw = raw, test = test, seed = seed)
  list(results = results, oracle = oracle)
}


# ==============================================================================
# Main
# ==============================================================================

args <- commandArgs(trailingOnly = TRUE)
num_MC <- parse_positive(args, 1L, 500L)
workers <- parse_positive(args, 2L, 6L)
solvers <- demo_solvers()
S <- length(solvers)
K <- length(T_grid)

cat("\n")
print_section_header("T-Solver Performance: FDR and TPR vs. T")
cat(sprintf("n = %d, p = %d, L = p = %d, true coefficients: %d x %s, SNR = %s\n",
            n, p, num_dummies, num_active, .num(beta_active), .num(snr)))
cat("True support: ", paste(SUPPORT, collapse = " "), "\n", sep = "")
cat("T grid: ", paste(T_grid, collapse = " "), "\n", sep = "")
cat(sprintf("%d solvers, %d runs on %d worker processes (single-threaded solvers)\n\n", S, num_MC, workers))

t_start <- now()
runs <- run_parallel(run_one, seq_len(num_MC) - 1L, workers)
wall <- now() - t_start

failures <- sum(vapply(runs, function(run) sum(vapply(run$results, is.null, logical(1))), integer(1)))
oracle_rmse <- mean(vapply(runs, function(run) run$oracle[1L], numeric(1)))
oracle_coef <- mean(vapply(runs, function(run) run$oracle[2L], numeric(1)))

#' Values of one field at the k-th T over the runs in which solver s succeeded.
column <- function(s, k, field) {
  as.numeric(unlist(lapply(runs, function(run) if (is.null(run$results[[s]])) NULL else run$results[[s]][k, field])))
}

out <- character()
table_block <- function(title, field, scale, digits) {
  header <- paste0(sprintf("%-22s", "solver"), paste(sprintf("%8s", paste0("T=", T_grid)), collapse = ""))
  rows <- vapply(seq_len(S), function(s) {
    values <- vapply(seq_len(K), function(k) scale * mean(column(s, k, field)), numeric(1))
    paste0(sprintf("%-22s", solvers[[s]]$name), paste(fmt_fixed(values, 8L, digits), collapse = ""))
  }, character(1))
  c(title, header, strrep("-", 22L + 8L * K), rows, "")
}

out <- c(out, table_block("FDR (%)", "fdp", 100.0, 1L))
out <- c(out, table_block("TPR (%)", "tpp", 100.0, 1L))
out <- c(out, table_block("Runs that reached T dummies (%)", "reached", 100.0, 0L))
out <- c(out, table_block("Average number of selected variables", "selected", 1.0, 1L))
out <- c(out, table_block("Average number of steps", "steps", 1.0, 1L))
out <- c(out, table_block("Average time from construction to T (ms, cumulative)", "ms", 1.0, 2L))
out <- c(out, table_block("Prediction RMSE of the least-squares refit on the selected variables (test set)",
                          "rmse", 1.0, 2L))
out <- c(out, table_block("Coefficient error of the refit, ||beta_hat - beta||_2", "coef_err", 1.0, 2L))
out <- c(out,
         sprintf("Least-squares refit on the true support: prediction RMSE %.2f, coefficient error %.2f",
                 oracle_rmse, oracle_coef),
         "",
         sprintf("Failed solver runs: %d", failures),
         sprintf("Wall time: %.1f s", wall))
text <- paste0(paste(out, collapse = "\n"), "\n")
cat(text)

# Suffixed for a TREX_SOLVERS selection
stem <- paste0(sprintf("fdr_tpr_vs_T_mc%d", num_MC), solver_file_suffix())
data_dir <- output_dir(this_dir_)
csv_path <- file.path(data_dir, paste0(stem, ".csv"))
lines <- paste0("solver,family,T,FDR,FDR_se,TPR,TPR_se,avg_selected,avg_steps,reached,avg_ms,",
                "rmse,rmse_se,coef_err,coef_err_se,oracle_rmse,oracle_coef_err")
for (s in seq_len(S)) {
  for (k in seq_len(K)) {
    fdp <- column(s, k, "fdp")
    tpp <- column(s, k, "tpp")
    rmse <- column(s, k, "rmse")
    coef <- column(s, k, "coef_err")
    row <- c(mean(fdp), std_error(fdp), mean(tpp), std_error(tpp),
             mean(column(s, k, "selected")), mean(column(s, k, "steps")),
             mean(column(s, k, "reached")), mean(column(s, k, "ms")),
             mean(rmse), std_error(rmse), mean(coef), std_error(coef), oracle_rmse, oracle_coef)
    lines <- c(lines, paste(c(solvers[[s]]$name, solvers[[s]]$family, T_grid[k], fmt_g(row)), collapse = ","))
  }
}
writeLines(lines, csv_path)
txt_path <- file.path(data_dir, paste0(stem, ".txt"))
cat(text, file = txt_path)

cat(sprintf("\nResults written to:\n  %s\n  %s\n\n", display_path(csv_path), display_path(txt_path)))
