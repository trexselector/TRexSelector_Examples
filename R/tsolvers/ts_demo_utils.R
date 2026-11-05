# ==============================================================================
# ts_demo_utils.R
# ==============================================================================
#
# Shared helpers for the standalone terminating-solver (tsolvers) demos.
#
# Mirrors the C++ utilities used by cpp/tsolvers/ad/demo_ts_*:
#   - datagen::SyntheticData with predictor_policy::Normal(),
#     dummygen::Distribution::Normal(), noisegen::noise_policy::Normal()
#   - cdiagnostics::print_section_header / print_talgo_demo_config /
#     print_selection / print_selection_quality (same console format)
#   - ts_perf::PartSelection, ts_perf::PathRecorder and ts_perf::display_path
#     (cpp/tsolvers/ts_perf_utils.hpp)
#
# All indices are 1-based (the R convention): variables 1..p, dummies p+1..p+L
# in the combined matrix [X, D], and dummies 1..L on their own in the path.
#
# Public functions:
#   gen_synthetic_data(n, p, num_dummies, support, coefs, snr, seed)
#   print_section_header(title)
#   print_demo_config(n, p, num_dummies, T_stop, support, coefs, snr)
#   selected_variables(solver)
#   print_selection(solver, true_support)
#   print_selection_quality(solver, true_support)
#   display_path(path), relative_path(path), output_dir(demo_dir)
#   part_selection(), path_recorder(stem, out_dir)
#
# ==============================================================================

# Synthetic regression data with explicit dummy matrix, mirroring
# C++ datagen::SyntheticData:
#
#   X[i, j] iid N(0, 1)   (n x p signal predictors)
#   D[i, j] iid N(0, 1)   (n x num_dummies dummies)
#   y = X[, support] %*% coefs + eps,  eps ~ N(0, noise_sigma^2 I_n)
#
# noise_sigma = sqrt(Var(signal) / snr), with the same small-sample rule as
# C++ noisegen::calculate_noise_params(): Bessel-corrected variance (n - 1)
# for n <= 100, population variance (n) otherwise.
#
# `support` is 1-based (R convention); order pairs positionally with `coefs`.
gen_synthetic_data <- function(n, p, num_dummies, support, coefs,
                               snr = 1.0, seed = NULL) {
  stopifnot(length(support) == length(coefs),
            all(support >= 1L), all(support <= p))
  if (!is.null(seed)) set.seed(seed)

  X <- matrix(rnorm(n * p), nrow = n, ncol = p)
  D <- matrix(rnorm(n * num_dummies), nrow = n, ncol = num_dummies)

  signal <- as.numeric(X[, support, drop = FALSE] %*% coefs)
  denom <- if (n <= 100) max(n, 2) - 1 else n
  var_sig <- sum((signal - mean(signal))^2) / denom
  noise_sigma <- if (snr > 0 && var_sig > 0) sqrt(var_sig / snr) else 0
  y <- signal + rnorm(n) * noise_sigma

  list(X = X, D = D, y = y)
}


# Mirrors cdiagnostics::print_section_header.
print_section_header <- function(title) {
  bar <- strrep("=", 60)
  cat("\n", bar, "\n", title, "\n", bar, "\n", sep = "")
}


# A number as the default C++ stream prints it (6 significant digits).
.num <- function(x) sprintf("%g", x)

# A vector in the {a, b, c} form of cdiagnostics::print_vector.
.braced <- function(v) paste0("{", paste(v, collapse = ", "), "}")


# Mirrors cdiagnostics::print_talgo_demo_config.
print_demo_config <- function(n, p, num_dummies, T_stop, support, coefs, snr) {
  cat("Configuration:\n")
  cat(sprintf("  n = %d, p = %d\n", n, p))
  cat(sprintf("  num_dummies = %d\n", num_dummies))
  cat(sprintf("  T_stop = %d\n", T_stop))
  cat(sprintf("  True support size = %d\n", length(support)))
  cat("  True support: ", paste0(support, " ", collapse = ""), "\n", sep = "")
  cat("  True coefficients: ", paste0(.num(coefs), " ", collapse = ""), "\n", sep = "")
  cat(sprintf("  SNR = %s\n\n", .num(snr)))
}


# Active set in the order the solver keeps it: additions append, drops remove.
.active_order <- function(solver) {
  active <- integer()
  for (a in unlist(solver$get_actions())) {
    if (a > 0) {
      if (!(a %in% active)) active <- c(active, a)
    } else {
      active <- active[active != -a]
    }
  }
  active
}


# Mirrors cdiagnostics::selected_variables: the variables with a nonzero
# coefficient on the last path step, as the T-Rex selector reads them.
# Returns predictors, dummies and frozen as 1-based indices into [X, D]:
# active variables first, in the order of the active set, then variables with
# a coefficient outside the active set (T-Stagewise keeps the coefficient of a
# variable it removes), which are also listed in frozen.
selected_variables <- function(solver) {
  path <- solver$get_beta_path()
  none <- list(predictors = integer(), dummies = integer(), frozen = integer())
  if (length(path) == 0L) return(none)
  support <- which(abs(path[, ncol(path)]) > .Machine$double.eps)
  dummy_start <- solver$get_dummy_start_index()
  listed <- intersect(.active_order(solver), support)
  frozen <- sort(setdiff(support, listed))
  all <- c(listed, frozen)
  list(predictors = all[all < dummy_start], dummies = all[all >= dummy_start],
       frozen = frozen)
}


# Mirrors cdiagnostics::print_selection: selected predictors and dummies from
# the coefficient path, then the path of additions and removals with the same
# ANSI colors (true positives green, false positives red, dummies blue,
# removals yellow and magenta).
print_selection <- function(solver, true_support) {
  cat("\nSelection:\n")
  sel <- selected_variables(solver)
  cat("  Predictors: ", .braced(sel$predictors), "\n", sep = "")
  cat("  Dummies: ", .braced(sel$dummies), sep = "")
  if (length(sel$frozen) > 0L)
    cat("\n  Frozen (coefficient kept outside the active set): ", .braced(sel$frozen), sep = "")
  cat("\n")
  p <- solver$get_dummy_start_index() - 1L
  tokens <- vapply(unlist(solver$get_actions()), function(a) {
    idx <- abs(a)
    if (a < 0) {
      if (idx > p) sprintf("\033[35m-D%d\033[0m", idx - p) else sprintf("\033[33m-v%d\033[0m", idx)
    } else if (idx > p) {
      sprintf("\033[34m+D%d\033[0m", idx - p)
    } else if (idx %in% true_support) {
      sprintf("\033[32m+v%d\033[0m", idx)
    } else {
      sprintf("\033[31m+v%d\033[0m", idx)
    }
  }, character(1))
  cat("  Path: [", paste(tokens, collapse = ", "), "]\n", sep = "")
}


# Mirrors cdiagnostics::print_selection_quality on the coefficient-path selection.
print_selection_quality <- function(solver, true_support) {
  selected <- selected_variables(solver)$predictors
  tp <- length(intersect(selected, true_support))
  fdp <- TRexSelector::trex_fdp(selected, true_support)
  tpp <- TRexSelector::trex_tpp(selected, true_support)
  prec <- TRexSelector::trex_precision(selected, true_support)
  rec <- TRexSelector::trex_recall(selected, true_support)
  f1 <- if (prec + rec > 0) 2 * prec * rec / (prec + rec) else 0
  cat("\nSelection Quality:\n")
  cat(sprintf("  True Positives:  %d\n", tp))
  cat(sprintf("  False Positives: %d\n", length(selected) - tp))
  cat(sprintf("  False Negatives: %d\n", length(true_support) - tp))
  cat(sprintf("  FDP:             %s\n", .num(fdp)))
  cat(sprintf("  TPP:             %s\n", .num(tpp)))
  cat(sprintf("  Precision:       %s\n", .num(prec)))
  cat(sprintf("  Recall:          %s\n", .num(rec)))
  cat(sprintf("  F1 Score:        %s\n", .num(f1)))
}


# ==============================================================================
# Part selection, result files and terminated solution paths
# ==============================================================================

# Path for console output without the machine-specific prefix: ".../R/...".
display_path <- function(path) {
  path <- normalizePath(path, mustWork = FALSE)
  pos <- regexpr("/R/", path, fixed = TRUE)
  if (pos > 0) paste0(".../", substring(path, pos + 1)) else basename(path)
}


# Path relative to the working directory when the path lies below it, so the
# logs carry no machine path.
relative_path <- function(path) {
  path <- normalizePath(path, mustWork = FALSE)
  wd <- paste0(normalizePath(getwd()), "/")
  if (startsWith(path, wd)) substring(path, nchar(wd) + 1L) else path
}


# simulation_results/data/ next to the demo script, created on demand.
output_dir <- function(demo_dir) {
  path <- file.path(demo_dir, "simulation_results", "data")
  dir.create(path, recursive = TRUE, showWarnings = FALSE)
  path
}


# Parts of a demo selected on the command line, as ts_perf::PartSelection:
# `Rscript demo_ts_01_tlars.R 5` runs Part 5 only, without arguments every
# part runs. Returns a function run(part) -> TRUE / FALSE.
part_selection <- function(args = commandArgs(trailingOnly = TRUE)) {
  if (length(args) > 0L && !all(grepl("^[1-9][0-9]*$", args)))
    stop("invalid part number: ", paste(args, collapse = " "))
  parts <- as.integer(args)
  function(part) length(parts) == 0L || part %in% parts
}


# Writes the terminated solution paths of one warm-started solver, as
# ts_perf::PathRecorder, into two tidy CSV files of the same columns:
#
#   <stem>_path.csv    : T_stop,step,index,is_dummy,beta
#   <stem>_actions.csv : T_stop,step,index,is_dummy,added
#
# Step 0 is the all-zero start, and actions[[k]] of the solver leads to step
# k. Indices are 1-based into [X, D] (the R convention), and the path lists
# nonzero coefficients only, on the original scale.
path_recorder <- function(stem, out_dir) {
  path_file <- file.path(out_dir, paste0(stem, "_path.csv"))
  actions_file <- file.path(out_dir, paste0(stem, "_actions.csv"))
  writeLines("T_stop,step,index,is_dummy,beta", path_file)
  writeLines("T_stop,step,index,is_dummy,added", actions_file)
  record <- function(solver, T_stop) {
    dummy_start <- solver$get_dummy_start_index()
    path <- solver$get_beta_path()
    lines <- character()
    for (s in seq_len(ncol(path))) {
      idx <- which(path[, s] != 0)
      if (length(idx) > 0L)
        lines <- c(lines, sprintf("%d,%d,%d,%d,%.10g", T_stop, s - 1L, idx,
                                  as.integer(idx >= dummy_start), path[idx, s]))
    }
    if (length(lines) > 0L) cat(paste0(lines, "\n"), file = path_file, sep = "", append = TRUE)
    acts <- solver$get_actions()
    lines <- character()
    for (k in seq_along(acts)) {
      a <- acts[[k]]
      lines <- c(lines, sprintf("%d,%d,%d,%d,%d", T_stop, k, abs(a),
                                as.integer(abs(a) >= dummy_start), as.integer(a > 0)))
    }
    if (length(lines) > 0L) cat(paste0(lines, "\n"), file = actions_file, sep = "", append = TRUE)
  }
  list(record = record, path_file = path_file, actions_file = actions_file)
}
