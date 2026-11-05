# ==============================================================================
# demo_ts_10_tncgmp.R
# ==============================================================================
#
# Demonstration of the T-NCGMP (Terminating Norm-Corrective Generalized Matching Pursuit) solver.
#
# Mirrors cpp/tsolvers/ad/demo_ts_10_tncgmp/demo_ts_10_tncgmp.cpp.
#
# Parts:
#   Demo 1: Basic T-NCGMP with early stopping (internal normalization),
#           high-dimensional (p > n) and low-dimensional (n > p) regimes.
#   Demo 2: External L2 normalization (LpNormScaler) + centered y,
#           solver run with normalize = FALSE, intercept = FALSE.
#   Demo 3: Serialization & warm start — save a partial path, reload,
#           continue, and compare against a reference run.
#   Demo 4: Memory-mapped X via mem_matrix_from(). The R binding
#           accepts an mmap X (D and y stay in memory); cpp maps all three
#           and generates p = 500,000 on disk — scaled to p = 5,000 here.
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


# ==============================================================================
# Demo 1: Basic T-NCGMP with Early Stopping
# ==============================================================================

demo_early_stopping <- function(high_dim, rnd_coef, T_stop) {
  print_section_header("Demo 1: Basic T-NCGMP with Early Stopping")

  n <- if (high_dim) 1000L else 5000L
  p <- if (high_dim) 5000L else 1000L
  num_dummies <- 10L * p

  # cpp true_support is 0-based {27, 149, 398, 420, 4} -> 1-based here
  true_support <- c(28L, 150L, 399L, 421L, 5L)
  true_coefs <- if (rnd_coef) c(-0.4, -0.2, -0.8, 1.1, 2.5) else rep(1, 5)
  snr <- 1.0

  cat(if (high_dim) "High-dimensional (p > n)\n" else "Low-dimensional (n > p)\n")
  print_demo_config(n, p, num_dummies, T_stop, true_support, true_coefs, snr)

  dat <- gen_synthetic_data(n, p, num_dummies, true_support, true_coefs,
                            snr, seed = 42)

  solver <- TNCGMP_Solver$new(
    dat$X, dat$D, dat$y,
    options = tsolver_options(ncgmp_variant = "line_search", normalize = TRUE, intercept = TRUE, verbose = TRUE)
  )
  solver$execute_step(T_stop, early_stop = TRUE)

  print_selection(solver, true_support)
  print_selection_quality(solver, true_support)
  cat("\n\n")
}


# ==============================================================================
# Demo 2: External Normalization (LpNormScaler)
# ==============================================================================

demo_with_external_normalizer <- function(high_dim, rnd_coef, T_stop) {
  print_section_header("=== Demo 2: T-NCGMP with External Normalization ===")

  n <- if (high_dim) 1000L else 5000L
  p <- if (high_dim) 5000L else 1000L
  num_dummies <- 10L * p

  # cpp true_support is 0-based {4, 27, 149, 398, 420} -> 1-based here
  true_support <- c(5L, 28L, 150L, 399L, 421L)
  true_coefs <- if (rnd_coef) c(-0.4, -0.2, -0.8, 1.1, 2.5) else rep(1, 5)
  snr <- 1.0

  cat(if (high_dim) "High-dimensional (p > n)\n" else "Low-dimensional (n > p)\n")
  print_demo_config(n, p, num_dummies, T_stop, true_support, true_coefs, snr)

  dat <- gen_synthetic_data(n, p, num_dummies, true_support, true_coefs,
                            snr, seed = 42)

  # External normalization: center + unit-L2 columns for X and D, center y.
  cat("Applying external L2 normalization...\n")
  l2scaler_X <- MLLpNormScaler$new(norm_type = 2, center = TRUE)
  l2scaler_X$fit(dat$X)
  l2scaler_X$transform_inplace(dat$X)

  l2scaler_D <- MLLpNormScaler$new(norm_type = 2, center = TRUE)
  l2scaler_D$fit(dat$D)
  l2scaler_D$transform_inplace(dat$D)

  dat$y <- dat$y - mean(dat$y)

  # The data was already externally L2-normalized and y centered above, so
  # the solver's own normalization and intercept are disabled here —
  # enabling them would normalize twice and fit an intercept on centered y.
  solver <- TNCGMP_Solver$new(
    dat$X, dat$D, dat$y,
    options = tsolver_options(ncgmp_variant = "line_search", normalize = FALSE, intercept = FALSE, verbose = TRUE)
  )
  solver$execute_step(T_stop, early_stop = TRUE)

  print_selection(solver, true_support)
  print_selection_quality(solver, true_support)
  cat("\n\n")
}


# ==============================================================================
# Demo 3: Serialization & Warm Start
# ==============================================================================

demo_serialization <- function() {
  print_section_header("=== Demo 3: T-NCGMP Serialization & Warm Start ===")

  n <- 100L
  p <- 50L
  num_dummies <- p
  T_stop_final <- 15L
  T_stop_partial <- 7L
  snr <- 1.0

  # cpp true_support is 0-based {10, 25, 40} -> 1-based here
  true_support <- c(11L, 26L, 41L)
  true_coefs <- c(2.5, -1.8, 3.2)

  print_demo_config(n, p, num_dummies, T_stop_final, true_support,
                    true_coefs, snr)

  dat <- gen_synthetic_data(n, p, num_dummies, true_support, true_coefs,
                            snr, seed = 42)

  # Normalize ONCE in front; the three solvers below are READERS.
  # The reference, checkpoint and resumed solvers share dat$X and dat$D (the
  # R6 solvers map the R matrices, no copy). A solver with normalize = TRUE /
  # intercept = TRUE is a WRITER that preprocesses them in place, so three
  # writers would center and scale the data three times, which moves it by
  # rounding and made the paths differ (the 2026-09-25 finding; a second
  # writer on a held matrix errors since the library's in-place contract).
  # So: the statistics live in the scalers, the solvers run as constructed
  # (normalize = FALSE, intercept = FALSE), no copies, bit-identical paths.
  l2scaler_X <- MLLpNormScaler$new(norm_type = 2, center = TRUE)
  l2scaler_X$fit_transform_inplace(dat$X)
  l2scaler_D <- MLLpNormScaler$new(norm_type = 2, center = TRUE)
  l2scaler_D$fit_transform_inplace(dat$D)
  y_mean <- mean(dat$y)
  dat$y <- dat$y - y_mean

  # Reference solver: run until T_stop_final.
  solver_ref <- TNCGMP_Solver$new(
    dat$X, dat$D, dat$y,
    options = tsolver_options(ncgmp_variant = "line_search", normalize = FALSE, intercept = FALSE, verbose = TRUE)
  )
  solver_ref$execute_step(T_stop_final, early_stop = TRUE)
  cat(sprintf("\u2713 Reference completed with %d steps\n\n",
              solver_ref$get_num_steps()))

  # relative to the working directory, so the logs carry no machine path
  filename <- relative_path(file.path(this_dir_, "tncgmp_checkpoint.bin"))

  # STEP 1: Run partial path and save checkpoint.
  solver1 <- TNCGMP_Solver$new(
    dat$X, dat$D, dat$y,
    options = tsolver_options(ncgmp_variant = "line_search", normalize = FALSE, intercept = FALSE, verbose = TRUE)
  )
  solver1$execute_step(T_stop_partial, early_stop = TRUE)
  cat(sprintf("Checkpoint at partial stop: %d steps\n", solver1$get_num_steps()))
  solver1$save(filename)
  cat(sprintf("\u2713 Checkpoint saved at '%s'\n", basename(filename)))

  # STEP 2: Load from checkpoint and continue.
  # The R binding hydrates an existing instance (cpp: static
  # load(filename, X, D)); $new() re-binds the data views. The receiving
  # solver is a reader on the pre-scaled data, the state the checkpoint
  # was saved in.
  solver2 <- TNCGMP_Solver$new(
    dat$X, dat$D, dat$y,
    options = tsolver_options(ncgmp_variant = "line_search", normalize = FALSE, intercept = FALSE, verbose = TRUE)
  )
  solver2$load(filename)
  cat(sprintf("Loaded from checkpoint: %d steps\n", solver2$get_num_steps()))
  solver2$execute_step(T_stop_final, early_stop = TRUE)

  cat("\nCOMPARISON:\n")
  cat(sprintf("Reference steps: %d\n", solver_ref$get_num_steps()))
  cat(sprintf("Reloaded steps:  %d\n", solver2$get_num_steps()))
  cat(sprintf("RSS diff:   %g\n",
              abs(tail(solver_ref$get_rss(), 1) - tail(solver2$get_rss(), 1))))
  cat(sprintf("R2 diff:    %g\n",
              abs(tail(solver_ref$get_r2(), 1) - tail(solver2$get_r2(), 1))))
  paths_match <- identical(unlist(solver_ref$get_actions()),
                           unlist(solver2$get_actions()))
  cat(if (paths_match) "\u2713 Paths match\n" else "\u2717 Paths differ!\n")

  # Give the data back (the statistics are in the scalers)
  l2scaler_X$inverse_transform_inplace(dat$X)
  l2scaler_D$inverse_transform_inplace(dat$D)
  dat$y <- dat$y + y_mean

  file.remove(filename)
  cat("\u2713 Checkpoint file removed\n")
  cat("\n\n")
}


# ==============================================================================
# Demo 4: Memory-Mapped Data
# ==============================================================================

demo_memory_mapped <- function(high_dim, rnd_coef, T_stop) {
  print_section_header("=== Demo 4: T-NCGMP with Memory-Mapped Data ===")

  n <- if (high_dim) 1000L else 5000L
  p <- if (high_dim) 5000L else 1000L  # cpp: 500,000 on disk; scaled here
  num_dummies <- 10L * p
  snr <- 1.0

  # cpp true_support is 0-based {4, 27, 149, 398, 420} -> 1-based here
  true_support <- c(5L, 28L, 150L, 399L, 421L)
  true_coefs <- if (rnd_coef) c(2.5, -0.4, -0.2, -0.8, 1.1) else rep(1, 5)

  print_demo_config(n, p, num_dummies, T_stop, true_support, true_coefs, snr)

  cat("Generating memory-mapped data...\n")
  X_file <- file.path(this_dir_, "demo_tncgmp_X.bin")

  dat <- gen_synthetic_data(n, p, num_dummies, true_support, true_coefs,
                            snr, seed = 42)

  on.exit({
    cat("\nCleaning up files...\n")
    if (file.exists(X_file)) file.remove(X_file)
    cat("\u2713 All files removed\n\n")
  }, add = TRUE)

  # Write X to disk and pass the mmap view to the solver. The R binding
  # accepts an mmap_matrix for X only; D and y remain in memory (cpp maps
  # all three via SyntheticDataMapped).
  X_mmap <- mem_matrix_from(dat$X, X_file)
  dat$X <- NULL
  cat("\u2713 Data generated on disk\n\n")

  cat("Running T-NCGMP on memory-mapped data...\n")
  solver <- TNCGMP_Solver$new(
    X_mmap, dat$D, dat$y,
    options = tsolver_options(ncgmp_variant = "line_search", normalize = TRUE, intercept = TRUE, verbose = TRUE)
  )
  solver$execute_step(T_stop, early_stop = TRUE)
  cat("\u2713 T-NCGMP completed\n\n")

  print_selection(solver, true_support)
  print_selection_quality(solver, true_support)
}



# ==============================================================================
# Demo 5: Terminated Solution Paths
# ==============================================================================

# Record the terminated solution paths of one T-NCGMP object that is
# warm-started T = 1 -> 2 -> 5 (setup of the R tlars vignette). Writes
# tncgmp_<variant>_terminated_paths_{path,actions}.csv to simulation_results/data/;
# generate_plots.sh draws the path figure from them.
demo_terminated_paths <- function(variant) {
  line_search <- variant == "line_search"
  variant_label <- if (line_search) "line search" else "fully corrective"
  variant_stem <- if (line_search) "linesearch" else "fullycorrective"
  print_section_header(paste0("=== Demo 5: T-NCGMP Terminated Solution Paths (", variant_label, ") ==="))

  # Setup of the tlars vignette: 5 actives with beta = 1 and noise sigma about 1
  n <- 150L
  p <- 300L
  num_dummies <- p
  true_support <- 1:5
  true_coefs <- rep(1, 5)
  snr <- 5.0  # signal variance 5 over noise variance 1
  T_stops <- c(1L, 2L, 5L)

  print_demo_config(n, p, num_dummies, max(T_stops), true_support, true_coefs, snr)

  dat <- gen_synthetic_data(n, p, num_dummies, true_support, true_coefs,
                            snr, seed = 42)

  # One solver object; every execute_step() call continues its path
  solver <- TNCGMP_Solver$new(
    dat$X, dat$D, dat$y,
    options = tsolver_options(ncgmp_variant = variant, normalize = TRUE, intercept = TRUE, verbose = FALSE)
  )

  recorder <- path_recorder(paste0("tncgmp_", variant_stem, "_terminated_paths"), output_dir(this_dir_))
  for (T_stop in T_stops) {
    solver$execute_step(T_stop, early_stop = TRUE)
    recorder$record(solver, T_stop)

    selected <- solver$get_active_predictor_indices()
    cat(sprintf("T_stop = %d: steps = %2d, predictors = %2d, dummies = %d, FDP = %.3f, TPP = %.3f\n",
                T_stop, solver$get_num_steps(), length(selected),
                length(solver$get_active_dummy_indices()),
                trex_fdp(selected, true_support), trex_tpp(selected, true_support)))
  }

  print_selection(solver, true_support)

  cat(sprintf("\nPath data written to:\n  %s\n  %s\n\n", display_path(recorder$path_file),
              display_path(recorder$actions_file)))
}

# ==============================================================================
# Main
# ==============================================================================

print_section_header("T-NCGMP Demo Suite")

# Six OpenMP threads, as the C++ demo
trex_set_num_threads(min(6L, trex_get_max_threads()))
cat(sprintf("Running with %d threads\n\n", trex_get_num_threads()))

# Optional arguments: the parts to run (default: all)
run <- part_selection()

# Demo 1: Basic usage with internal normalization
if (run(1)) {
  demo_early_stopping(high_dim = TRUE, rnd_coef = FALSE, T_stop = 10L)
  demo_early_stopping(high_dim = FALSE, rnd_coef = FALSE, T_stop = 10L)
}

# Demo 2: External normalization
if (run(2)) {
  demo_with_external_normalizer(high_dim = FALSE, rnd_coef = FALSE, T_stop = 10L)
  demo_with_external_normalizer(high_dim = TRUE, rnd_coef = FALSE, T_stop = 10L)
}

# Demo 3: Serialization
if (run(3)) {
  demo_serialization()
}

# Demo 4: Memory-mapped
if (run(4)) {
  demo_memory_mapped(high_dim = TRUE, rnd_coef = FALSE, T_stop = 10L)
}

# Demo 5: Terminated solution paths (data for the path figure)
if (run(5)) {
  for (variant in c("line_search", "fully_corrective")) demo_terminated_paths(variant)
}

print_section_header("\u2713 All demos completed successfully")
