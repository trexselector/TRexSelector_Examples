# ==============================================================================
# trex_sim_utils.R
# ==============================================================================
#
# Shared simulation utilities for T-Rex demos.
#
# Mirrors C++ trex_sim_utils.hpp.
#
# Contents:
# -----------------------------------------------------------------------------
#   Section 1 — DGP (in-memory)
#     dgp_gauss_snr()           — i.i.d. Gaussian DGP with SNR-calibrated noise.
#
#   Section 2 — DGP (out-of-core / mmap)
#     dgp_chunked_gauss_snr()   — Chunked out-of-core DGP for massive datasets.
#
#   Section 3 — Single-run print helpers
#     .make_dual_out()           — PrintFn-style dual console + file sink.
#     .print_sparse_vector()     — Nonzero entries of a p-vector.
#     .print_sparse_matrix()     — Nonzero entries of a (T x p) matrix.
#     .print_selection_results() — Selected indices, FDP/TPP, sparse Phi block.
#     .print_single_run()        — Formatted single-run result block.
#
#   Section 4 — MC infrastructure
#     SOLVERS_DEFAULT            — List of 14 solver descriptors (mirrors
#                                  make_default_solvers_to_test() in C++).
#     .selected_solvers()        — The solvers TREX_SOLVERS selects.
#     .resolve_num_mc()          — Trial count, TREX_NUM_MC for smoke runs.
#     make_mmap_trex_ctrl()      — Build trex_control for memory-mapped scenarios.
#     execute_with_temp_mmap()   — RAII-like lifecycle manager for temp mmap files.
#     .print_demo_header()       — Standardised demo section headers.
#     .make_trex_ctrl()          — Build trex_control() from a solver descriptor.
#     .run_mc_trex()             — Parallel MC runner (TRexSelector OOP API).
#     .run_mc_trex_full_mmap()   — MC runner with per-trial mmap-backed X.
#     .print_solver_table()      — Aligned solver x SNR table to console.
#     .save_mc_csv()             — Tidy long-format CSV to disk.
#     .save_and_print_mc()       — Console + .txt + .csv combined output.
#
# ==============================================================================


# ==============================================================================
# Section 1 — DGP (in-memory)
# ==============================================================================


# ==============================================================================
# dgp_gauss_snr
# ==============================================================================

#' DGP for the i.i.d. Gaussian scenario with SNR-calibrated noise.
#'
#' Predictors are drawn i.i.d. from N(0, 1):
#'
#'   X[i, j]  i.i.d. N(0, 1)
#'
#' The response is:
#'
#'   y = X %*% beta + eps,   eps ~ N(0, noise_sigma^2 * I_n)
#'
#' where beta has entries equal to `amplitude` (or `coefs`) at the positions
#' given by `support` (1-based) and zero elsewhere.
#'
#' The noise standard deviation is chosen according to the target SNR:
#'
#'   noise_sigma = sqrt( var(X %*% beta) / snr )
#'
#' with var() using the n-1 denominator (Bessel correction), matching
#' C++ noisegen::calculate_noise_params().
#'
#' @param n         Number of observations.
#' @param p         Number of predictors.
#' @param support   1-based integer vector of active predictor indices.
#' @param amplitude Scalar signal coefficient applied to all active predictors.
#'                  Ignored when `coefs` is supplied. Default 1.
#' @param coefs     Optional numeric vector of length `length(support)` with
#'                  per-predictor active coefficients. Overrides `amplitude`.
#' @param snr       Target signal-to-noise ratio  Var(signal) / Var(noise).
#' @param seed      Integer seed for the RNG (governs both X and noise).
#'
#' @return A list with components:
#'   \describe{
#'     \item{X}{Predictor matrix (n x p).}
#'     \item{y}{Response vector (length n).}
#'     \item{beta}{Coefficient vector (length p).}
#'     \item{true_support}{1-based integer index vector of active predictors.}
#'     \item{n, p, s, snr}{Dimension and parameter scalars.}
#'   }
#'
#' @examples
#' dat <- dgp_gauss_snr(
#'   n       = 300,
#'   p       = 1000,
#'   support = c(1L, 21L, 41L, 61L, 81L, 101L, 121L, 141L, 161L, 181L),
#'   snr     = 1.0,
#'   seed    = 42L
#' )
dgp_gauss_snr <- function(n,
                          p,
                          support,
                          amplitude = 1,
                          coefs     = NULL,
                          snr       = 1.0,
                          seed      = NULL) {

  stopifnot(is.integer(support) || is.numeric(support))
  stopifnot(all(support >= 1L), all(support <= p))
  stopifnot(snr > 0)

  if (!is.null(seed)) set.seed(seed)

  # Generate X: n x p matrix of i.i.d. N(0, 1)
  # Drawn column-major (rnorm(n * p) fills column by column in R).
  X <- matrix(stats::rnorm(n * p), nrow = n, ncol = p)

  # Build coefficient vector
  beta         <- numeric(p)
  active_coefs <- if (!is.null(coefs)) coefs else rep(amplitude, length(support))
  beta[support] <- active_coefs

  # SNR-calibrated response
  signal      <- drop(X[, support] %*% beta[support])
  var_sig     <- stats::var(signal)   # uses n-1 denominator (Bessel correction)
  noise_sigma <- sqrt(var_sig / snr)
  y           <- signal + stats::rnorm(n, mean = 0, sd = noise_sigma)

  list(
    X            = X,
    y            = y,
    beta         = beta,
    true_support = as.integer(support),
    n            = n,
    p            = p,
    s            = length(support),
    snr          = snr
  )
}

# ==============================================================================


# ==============================================================================
# Section 2 — DGP (out-of-core / mmap)
# ==============================================================================

#' Chunked DGP for the i.i.d. Gaussian scenario with SNR-calibrated noise.
#'
#' Generates a massive design matrix X in memory-efficient column chunks,
#' writing it directly to a memory-mapped matrix using TRexSelector's native
#' `mem_matrix` backend. Bypasses R's memory limits for large datasets.
#'
#' @param n            Number of observations.
#' @param p            Number of predictors.
#' @param support      1-based integer vector of active predictor indices.
#' @param x_path       File path where the binary matrix will be written.
#' @param chunk_cols   Number of columns to generate per RAM chunk (default 1000).
#' @param amplitude    Scalar signal coefficient applied to active predictors.
#' @param coefs        Optional numeric vector overriding amplitude.
#' @param snr          Target signal-to-noise ratio.
#' @param seed         Integer seed for the RNG.
#'
#' @return A list with components:
#'   \describe{
#'     \item{X}{Memory-mapped matrix object (TRexSelector mem_matrix).}
#'     \item{y}{Response vector (length n, in memory).}
#'     \item{beta}{Coefficient vector (length p, in memory).}
#'     \item{true_support}{1-based integer index vector of active predictors.}
#'     \item{n, p, s, snr}{Dimension and parameter scalars.}
#'   }
dgp_chunked_gauss_snr <- function(n,
                                  p,
                                  support,
                                  x_path,
                                  chunk_cols = 1000L,
                                  amplitude  = 1,
                                  coefs      = NULL,
                                  snr        = 1.0,
                                  seed       = NULL) {

  stopifnot(is.integer(support) || is.numeric(support))
  stopifnot(all(support >= 1L), all(support <= p))
  stopifnot(snr > 0)

  if (!is.null(seed)) set.seed(seed)

  # Build coefficient vector in memory (p doubles is negligible for RAM)
  beta         <- numeric(p)
  active_coefs <- if (!is.null(coefs)) coefs else rep(amplitude, length(support))
  beta[support] <- active_coefs

  signal <- numeric(n)

  # Instantiate the memory-mapped matrix directly via TRexSelector
  X_mmap <- TRexSelector::mem_matrix(x_path, n, p)

  # Generate and write X in chunks of `chunk_cols`
  start_col <- 1L
  while (start_col <= p) {
    end_col <- min(start_col + chunk_cols - 1L, p)
    ncols   <- end_col - start_col + 1L

    # Generate chunk
    X_chunk <- matrix(stats::rnorm(n * ncols), nrow = n, ncol = ncols)

    # Write chunk to the memory-mapped matrix
    X_mmap[, start_col:end_col] <- X_chunk

    # Accumulate signal chunk-by-chunk
    beta_chunk <- beta[start_col:end_col]
    signal <- signal + as.numeric(X_chunk %*% beta_chunk)

    start_col <- start_col + chunk_cols
  }

  # SNR-calibrated response (matches in-memory Bessel correction)
  var_sig     <- stats::var(signal)
  noise_sigma <- sqrt(var_sig / snr)

  # Note: This is drawn after all X random numbers, identical to the in-memory version.
  y <- signal + stats::rnorm(n, mean = 0, sd = noise_sigma)

  list(X = X_mmap, y = y, beta = beta, true_support = as.integer(support),
       n = n, p = p, s = length(support), snr = snr)
}

# ==============================================================================


# ==============================================================================
# Section 3 — Single-run print helpers
# (mirrors Python trex_helpers.py / C++ trex_sim_utils.hpp sparse printers)
# ==============================================================================
#
# All printers accept an optional `out` callback for dual console + file
# output (mirrors the C++ PrintFn pattern / Python make_dual_out); when
# omitted, output goes to the console only.

#' Console-only output sink (default for the printers below).
.console_out <- function(text) cat(text)


#' Return a PrintFn-style callback writing to both console and `con`.
#'
#' Mirrors the C++ `print_dual` lambda and Python make_dual_out().
#'
#' @param con An open writable connection (text mode).
.make_dual_out <- function(con) {
  function(text) {
    cat(text)
    cat(text, file = con)
  }
}


#' Print only the nonzero entries of a p-vector, with their (1-based) indices.
#'
#' Mirrors Python print_sparse_vector() / C++ print_sparse_vector(): Phi_prime
#' carries one entry per variable, so a dense dump puts p numbers on a single
#' line of which typically ~0.1% are nonzero. Falls back to the dense form when
#' the vector is not actually sparse.
.print_sparse_vector <- function(label, vec, out = .console_out, eps = 1e-12) {
  vec <- as.numeric(vec)
  n   <- length(vec)
  nz  <- which(abs(vec) > eps)

  out(sprintf("\n%s: %d of %d entries nonzero\n", label, length(nz), n))

  if (length(nz) * 4L > n) {   # dense enough: keep the plain dump
    out(paste0("  ", paste(sprintf("%.6f", vec), collapse = "  "), "\n"))
    return(invisible(NULL))
  }

  per_line <- 5L
  cells    <- sprintf("%-20s", sprintf("[%d] %.6f", nz, vec[nz]))
  for (start in seq(1L, length(cells), by = per_line)) {
    stop_i <- min(start + per_line - 1L, length(cells))
    out(paste0("  ", paste(cells[start:stop_i], collapse = "  "), "\n"))
  }
  invisible(NULL)
}


#' Print only the nonzero entries of a matrix, row by row (1-based indices).
#'
#' Mirrors Python print_sparse_matrix(): the Phi matrix is (T x p), so each of
#' its rows is as wide as the variable count.
.print_sparse_matrix <- function(label, mat, out = .console_out, eps = 1e-12) {
  mat   <- as.matrix(mat)
  total <- length(mat)
  nnz   <- sum(abs(mat) > eps)

  out(sprintf("\n%s: %d x %d, %d of %d entries nonzero\n",
              label, nrow(mat), ncol(mat), nnz, total))

  if (nnz * 4L > total) {      # dense enough: keep the plain dump
    out(paste0(paste(utils::capture.output(print(round(mat, 4L))),
                     collapse = "\n"), "\n"))
    return(invisible(NULL))
  }

  for (r in seq_len(nrow(mat))) {
    nz <- which(abs(mat[r, ]) > eps)
    if (length(nz) == 0L) {
      out(sprintf("  row %d:  (all zero)\n", r))
    } else {
      cells <- sprintf("[%d] %.4f", nz, mat[r, nz])
      out(sprintf("  row %d:  %s\n", r, paste(cells, collapse = "  ")))
    }
  }
  invisible(NULL)
}


#' Print all selection outputs (indices, metrics, matrices).
#'
#' Mirrors Python print_selection_results() / C++ trex_sim::print_results():
#' selected indices, FDP/TPP, sparse Phi_prime, sparse Phi, dense FDP_hat and
#' R matrices, and the voting grid. Indices are 1-based (R convention).
.print_selection_results <- function(selected, true_support, selector,
                                     out = .console_out) {

  out(paste0("Selected indices: ", paste(selected, collapse = " "), " \n"))

  fdp <- TRexSelector::trex_fdp(selected, true_support)
  tpp <- TRexSelector::trex_tpp(selected, true_support)
  out(sprintf("False Discovery Proportion (FDP): %.4f\n", fdp))
  out(sprintf("True Positive Proportion (TPP):   %.4f\n", tpp))

  .print_sparse_vector("Adjusted Relative Occurrences (Phi_prime)",
                       selector$phi_prime, out)

  .print_sparse_matrix("Phi Matrix (Phi)", selector$phi_mat, out)

  out("\nEstimated FDP Matrix (FDP_hat):\n")
  out(paste0(paste(utils::capture.output(print(round(selector$fdp_hat_mat, 4L))),
                   collapse = "\n"), "\n"))

  out("\nR Matrix (R):\n")
  out(paste0(paste(utils::capture.output(print(round(selector$r_mat, 4L))),
                   collapse = "\n"), "\n"))

  out("\nVoting Grid:\n")
  out(paste0("  ", paste(sprintf("%.4f", selector$voting_grid),
                         collapse = "  "), "\n"))
  invisible(NULL)
}


#' Print a formatted single-run result block.
#'
#' A data/calibration header followed by the selection-result block with the
#' per-variable Phi_prime / Phi structures printed sparsely (mirrors Python
#' print_single_run_result()).
.print_single_run <- function(scenario_name, dat, selector, tFDR,
                              out = .console_out) {

  true_support <- dat$true_support
  s            <- length(true_support)
  p            <- ncol(dat$X)
  selected     <- selector$selected_indices
  n_sel        <- length(selected)
  n_tp         <- length(intersect(selected, true_support))
  n_fp         <- n_sel - n_tp

  sep_  <- strrep("=", 70L)
  dash_ <- strrep("-", 70L)

  out(paste0(sep_, "\n"))
  out(sprintf("  %s\n", scenario_name))
  out(paste0(dash_, "\n"))
  out(sprintf("  Data:  n = %d,  p = %d,  s = %d,  tFDR = %.2f\n",
              nrow(dat$X), p, s, tFDR))
  out(sprintf("         True support (1-based):   {%s}\n",
              paste(true_support, collapse = ", ")))
  out(sprintf("         Active coefficients:       {%s}\n",
              paste(sprintf("%.3f", dat$beta[true_support]), collapse = ", ")))
  out(paste0(dash_, "\n"))
  out(sprintf("  Calibration:  T_stop = %d,  L = %d\n",
              selector$T_stop, selector$L))
  out(sprintf("  Selection:    %d selected  |  TP = %d  FP = %d\n",
              n_sel, n_tp, n_fp))
  out(paste0(dash_, "\n"))

  .print_selection_results(selected, true_support, selector, out)

  out(paste0(sep_, "\n\n"))
  invisible(NULL)
}


# ==============================================================================
# Section 4 — MC infrastructure
# ==============================================================================
#
# Shared infrastructure for classic T-Rex MC simulations.
# Sourced by demo files — do not run standalone.
#
# Mirrors C++ trex_sim_utils.hpp.


# ==============================================================================
# Default solver list
# (mirrors make_default_solvers_to_test() in C++)
# ==============================================================================

# Values mirror the C++ DemoSolverInfo list exactly: lambda2 and rho_afs default
# to 0.0, overridden only where C++ does — TENET (lambda2 = 0.1), TAFS_rho_0.3 /
# TAFS_rho_1.0 (rho_afs = 0.3 / 1.0), TNCGMP_v1 (ncgmp_variant = 1). lambda2
# only affects the elastic-net solvers (TENET; a nonzero value would turn TLASSO
# into an elastic net), and rho_afs only affects TAFS — so the per-solver
# defaults must be 0.0 to match C++, not a blanket 0.1 / 0.3.
SOLVERS_DEFAULT <- list(
  list(solver = "TLARS",      name = "TLARS",        lambda2 = 0.0, rho_afs = 0.0,
       ncgmp_variant = 0L),
  list(solver = "TLASSO",     name = "TLASSO",       lambda2 = 0.0, rho_afs = 0.0,
       ncgmp_variant = 0L),
  list(solver = "TENET",      name = "TENET",        lambda2 = 0.1, rho_afs = 0.0,
       ncgmp_variant = 0L),
  list(solver = "TSTAGEWISE", name = "TSTAGEWISE",   lambda2 = 0.0, rho_afs = 0.0,
       ncgmp_variant = 0L),
  list(solver = "TSTEPWISE",  name = "TSTEPWISE",    lambda2 = 0.0, rho_afs = 0.0,
       ncgmp_variant = 0L),
  list(solver = "TOMP",       name = "TOMP",         lambda2 = 0.0, rho_afs = 0.0,
       ncgmp_variant = 0L),
  list(solver = "TGP",        name = "TGP",          lambda2 = 0.0, rho_afs = 0.0,
       ncgmp_variant = 0L),
  list(solver = "TACGP",      name = "TACGP",        lambda2 = 0.0, rho_afs = 0.0,
       ncgmp_variant = 0L),
  list(solver = "TMP",        name = "TMP",          lambda2 = 0.0, rho_afs = 0.0,
       ncgmp_variant = 0L),
  list(solver = "TAFS",       name = "TAFS_rho_0.3", lambda2 = 0.0, rho_afs = 0.3,
       ncgmp_variant = 0L),
  list(solver = "TAFS",       name = "TAFS_rho_1.0", lambda2 = 0.0, rho_afs = 1.0,
       ncgmp_variant = 0L),
  list(solver = "TNCGMP",     name = "TNCGMP_v1",    lambda2 = 0.0, rho_afs = 0.0,
       ncgmp_variant = 1L),
  list(solver = "TNCGMP",     name = "TNCGMP_v0",    lambda2 = 0.0, rho_afs = 0.0,
       ncgmp_variant = 0L),
  list(solver = "TOOLS",      name = "TOOLS",        lambda2 = 0.0, rho_afs = 0.0,
       ncgmp_variant = 0L)
)

#' The solvers of `solvers` that TREX_SOLVERS selects (all without a selection).
#' Mirrors selected_solvers() in trex_sim_utils.hpp. A selection run writes its
#' tables with the suffix solver_file_suffix() (mc_runtime.R).
.selected_solvers <- function(solvers) {
  Filter(function(s) solver_selected(s$name), solvers)
}

#' Monte Carlo trials of a demo: TREX_NUM_MC when set (smoke runs), else `default`.
#' Mirrors num_mc() in trex_sim_utils.hpp. TREX_NUM_MC enters the cell-cache
#' fingerprint, so a smoke run never restores the cells of a full run.
.resolve_num_mc <- function(default) {
  n <- suppressWarnings(as.integer(Sys.getenv("TREX_NUM_MC", "")))
  if (is.na(n) || n < 1L) default else n
}


# ==============================================================================
# Shared Helpers for Memory Mapping and Console Output
# ==============================================================================

#' Make a TRex control list for memory-mapping scenarios.
#'
#' Matches C++ make_mmap_trex_ctrl() and Python _MMAP_CTRL_DICT
#' (demo_trex_06_mmap.py): every option is pinned explicitly to the C++
#' reference values — no reliance on trex_control() defaults.
#'
#' @param ... Optional overrides for trex_control parameters.
#' @return A trex_control object.
make_mmap_trex_ctrl <- function(...) {

  base_ctrl <- list(
    K                        = 20L,
    max_dummy_multiplier     = 10L,
    use_max_T_stop           = TRUE,
    opt_threshold            = 0.75,
    lloop_strategy           = "HCONCAT",
    tloop_stagnation_stop    = TRUE,
    tloop_max_stagnant_steps = 5L,
    use_memory_mapping       = TRUE,
    lambda2                  = 0.1,
    rho_afs                  = 0.0,   # non-TAFS solver; rho_afs is unused
    ncgmp_variant            = 0L,
    parallel_rnd_experiments = FALSE
  )

  overrides <- list(...)
  for (nm in names(overrides)) {
    base_ctrl[[nm]] <- overrides[[nm]]
  }
  do.call(TRexSelector::trex_control, base_ctrl)
}


#' Execute a block of code with a memory-mapped temporary file
#'
#' Handles the RAII-like lifecycle of writing an in-memory matrix to a
#' temporary memory-mapped file, yielding it to the provided function,
#' and ensuring the file is unlinked safely.
#'
#' IMPORTANT: everything that touches the memory-mapped matrix — in
#' particular building AND running the TRexSelector — must happen INSIDE
#' `expr_func`. The backing file is unlinked via on.exit() as soon as this
#' function returns, so a mmap view that escapes the callback (e.g.
#' `execute_with_temp_mmap(X, function(x) x)`) points at a deleted file.
#' The callback's return value is checked and an error is raised if the
#' mmap object (or a list containing it) is returned.
#'
#' @param X_matrix The in-memory matrix to convert.
#' @param expr_func A function taking the memory-mapped matrix as its argument.
#' @return The result of `expr_func` (must not contain the mmap view).
execute_with_temp_mmap <- function(X_matrix, expr_func) {

  x_path <- tempfile(fileext = ".dat")
  on.exit(unlink(x_path), add = TRUE)

  X_mmap <- TRexSelector::mem_matrix_from(X_matrix, x_path)
  res    <- expr_func(X_mmap)

  # Guard against the mmap view escaping the callback: once this function
  # returns, the backing file is gone and the view is dangling.
  escapes <- inherits(res, "MemoryMappedMatrix") ||
    (is.list(res) && any(vapply(res, inherits, logical(1L), "MemoryMappedMatrix")))
  if (escapes) {
    stop("execute_with_temp_mmap(): the memory-mapped matrix must not escape ",
         "the callback — run the selector inside `expr_func` and return only ",
         "plain results (the temp backing file is unlinked on return).")
  }
  res
}


#' Print standardized demo section headers
#'
#' @param part_name Section title.
#' @param description Short description.
#' @param high_dim Boolean flag for high vs low dimensional setup.
#' @param n Number of samples.
#' @param p Number of features.
#' @param num_MC Number of Monte Carlo trials (optional).
.print_demo_header <- function(part_name, description, high_dim, n, p, num_MC = NULL) {

  dim_label <- if (high_dim) "High-dimensional (p > n)" else "Low-dimensional (n > p)"

  cat("\n", strrep("=", 70L), "\n", sep = "")
  cat(part_name, "\n")
  cat(sprintf("  %s\n", description))

  if (is.null(num_MC)) {
    cat(sprintf("  %s  (n = %d, p = %d)\n\n", dim_label, n, p))
  } else {
    cat(sprintf("  %s  (n = %d, p = %d,  num_MC = %d)\n\n",
                dim_label, n, p, num_MC))
  }
}


# ==============================================================================
# .make_trex_ctrl
# ==============================================================================

#' Build a trex_control() object from a solver descriptor and a base-params list.
#'
#' @param solver    One element of SOLVERS_DEFAULT (a named list with solver,
#'                  lambda2, rho_afs, ncgmp_variant).
#'
#' @param base      Named list of shared control parameters forwarded verbatim
#'                  to trex_control(). Must NOT include solver, lambda2,
#'                  rho_afs or ncgmp_variant (those come from `solver`).
#'
#' @return A trex_control() object.
.make_trex_ctrl <- function(solver, base) {
  do.call(
    trex_control,
    c(list(solver        = solver$solver,
           lambda2       = solver$lambda2,
           rho_afs       = solver$rho_afs,
           ncgmp_variant = solver$ncgmp_variant),
      base)
  )
}


# ==============================================================================
# .run_mc_trex
# ==============================================================================

#' Parallel MC runner for classical T-Rex (TRexSelector OOP API).
#'
#' Runs `num_MC` independent trials. Each trial calls `make_data_fn(mc,
#' trial_seed)` to obtain the data, builds a TRexSelector with `ctrl`, and
#' records FDP / TPP (and optionally L and T_stop).
#'
#' Parallelism uses spawned PSOCK workers via mc_map() (mc_runtime.R, which the
#' demos source first); each worker is its own R process, so set.seed() inside
#' make_data_fn is safe. The run is a resumable grid cell (cell_cached()).
#'
#' @param make_data_fn  Factory: function(mc, trial_seed) -> list(X, y,
#'                      true_support). Called once per trial.
#' @param ctrl          trex_control() object (same for all trials).
#' @param tFDR          Target FDR level.
#' @param base_seed     Base RNG seed; trial mc uses base_seed + mc - 1.
#' @param num_MC        Number of Monte Carlo trials.
#' @param num_cores     Number of parallel worker cores.
#' @param track_L_T     If TRUE, also record L and T_stop per trial.
#' @param label         Progress label printed before / after the run.
#' @param seeded_selector If TRUE, seed the selector with the trial seed
#'                      (reproducible cells; mirrors the C++ flag).
#'
#' @return Named list: mean_FDR, mean_TPR, sd_FDR, sd_TPR, and (if track_L_T) mean_L, mean_T.
.run_mc_trex <- function(make_data_fn,
                         ctrl,
                         tFDR,
                         base_seed,
                         num_MC,
                         num_cores,
                         track_L_T = FALSE,
                         label     = "",
                         seeded_selector = FALSE) {

  one_trial <- function(mc) {
    trial_seed <- base_seed + mc - 1L
    dat <- make_data_fn(mc, trial_seed)
    sel <- TRexSelector$new(
      X       = dat$X,
      y       = dat$y,
      tFDR    = tFDR,
      # Hardware entropy per trial (independent random experiments), or the
      # trial seed (distinct per trial and reproducible) for seeded_selector
      seed    = if (seeded_selector) as.integer(trial_seed) else -1L,
      verbose = FALSE,
      control = ctrl
    )
    sel$select()
    res <- c(
      FDP = TRexSelector::trex_fdp(sel$selected_indices, dat$true_support),
      TPP = TRexSelector::trex_tpp(sel$selected_indices, dat$true_support)
    )
    if (track_L_T) res <- c(res, L = sel$L, T = sel$T_stop)
    res
  }

  # One resumable grid cell: restored from the cell cache after a restart.
  # The seeded flag joins the key only when set, so the keys of the other demos stay unchanged
  digest_args <- list(ctrl, tFDR, base_seed, num_MC, track_L_T)
  if (seeded_selector) digest_args <- c(digest_args, list("seeded_selector"))
  key <- paste0("trex|", label, "|", do.call(cell_digest, digest_args))
  out <- cell_cached(key, function() {
    if (nchar(label) > 0L)
      cat(sprintf("  %s \u2014 Running %d MC trials ...\n", label, num_MC))
    res_list <- mc_map(seq_len(num_MC), one_trial, num_cores)
    res_mat <- do.call(rbind, res_list)
    fdps <- res_mat[, "FDP"]
    tpps <- res_mat[, "TPP"]

    out <- list(
      mean_FDR = mean(fdps),
      mean_TPR = mean(tpps),
      sd_FDR   = sd(fdps),
      sd_TPR   = sd(tpps)
    )
    if (track_L_T) {
      out$mean_L <- mean(res_mat[, "L"])
      out$mean_T <- mean(res_mat[, "T"])
    }
    out
  })

  if (nchar(label) > 0L)
    cat(sprintf("  %s \u2014 done. mean_TPR=%.3f  mean_FDR=%.3f\n\n",
                label, out$mean_TPR, out$mean_FDR))
  out
}


# ==============================================================================
# .run_mc_trex_full_mmap
# ==============================================================================

#' Parallel MC runner \u2014 fully memory-mapped X per trial.
#'
#' Same interface as .run_mc_trex(), but each trial writes its X to a
#' per-trial temporary mmap file via execute_with_temp_mmap() and runs the
#' selector INSIDE the callback (the backing file is unlinked as soon as the
#' callback returns, so the mmap view must never escape it).
#'
#' Mirrors Python trex_sim_common.run_mc_trex_full_mmap() and C++
#' trx_sim::run_mc_trials_trex_mmap(). The caller should pass a ctrl with
#' use_memory_mapping = TRUE to also enable internal D-mmap and solver
#' serialization on top of the X-mmap layer.
#'
#' @inheritParams .run_mc_trex
#' @return Named list: mean_FDR, mean_TPR, sd_FDR, sd_TPR, and (if track_L_T)
#'   mean_L, mean_T.
.run_mc_trex_full_mmap <- function(make_data_fn,
                                   ctrl,
                                   tFDR,
                                   base_seed,
                                   num_MC,
                                   num_cores,
                                   track_L_T = FALSE,
                                   label     = "") {

  one_trial <- function(mc) {
    trial_seed <- base_seed + mc - 1L
    dat <- make_data_fn(mc, trial_seed)

    # DGP result stays in RAM; X is written to a per-trial temp mmap file and
    # the selector runs on the mmap view inside the callback (exception-safe
    # RAII equivalent \u2014 mirrors C++ MmapFileGuard before SyntheticDataMapped).
    execute_with_temp_mmap(dat$X, function(X_mmap) {
      sel <- TRexSelector$new(
        X       = X_mmap,
        y       = dat$y,
        tFDR    = tFDR,
        seed    = -1L,  # hardware entropy per trial (independent random experiments)
        verbose = FALSE,
        control = ctrl
      )
      sel$select()
      res <- c(
        FDP = TRexSelector::trex_fdp(sel$selected_indices, dat$true_support),
        TPP = TRexSelector::trex_tpp(sel$selected_indices, dat$true_support)
      )
      if (track_L_T) res <- c(res, L = sel$L, T = sel$T_stop)
      res
    })
  }

  # One resumable grid cell: restored from the cell cache after a restart.
  key <- paste0("trex_mmap|", label, "|", cell_digest(ctrl, tFDR, base_seed, num_MC, track_L_T))
  out <- cell_cached(key, function() {
    if (nchar(label) > 0L)
      cat(sprintf("  %s \u2014 Running %d MC trials (full-mmap X) ...\n", label, num_MC))
    res_list <- mc_map(seq_len(num_MC), one_trial, num_cores)
    res_mat <- do.call(rbind, res_list)
    fdps <- res_mat[, "FDP"]
    tpps <- res_mat[, "TPP"]

    out <- list(
      mean_FDR = mean(fdps),
      mean_TPR = mean(tpps),
      sd_FDR   = sd(fdps),
      sd_TPR   = sd(tpps)
    )
    if (track_L_T) {
      out$mean_L <- mean(res_mat[, "L"])
      out$mean_T <- mean(res_mat[, "T"])
    }
    out
  })

  if (nchar(label) > 0L)
    cat(sprintf("  %s \u2014 done. mean_TPR=%.3f  mean_FDR=%.3f\n\n",
                label, out$mean_TPR, out$mean_FDR))
  out
}


# ==============================================================================
# .print_solver_table
# ==============================================================================

#' Print an aligned solver x SNR results table to the console.
#'
#' Columns are SNR values; rows are (solver, metric) pairs.
#'
#' @param solver_names   Character vector of solver names (row order).
#' @param snr_values     Numeric vector of SNR grid values (column order).
#' @param fdr_mat        Matrix (n_solvers x n_snr) of mean FDR values.
#' @param tpr_mat        Matrix (n_solvers x n_snr) of mean TPR values.
#' @param avg_L_mat      Matrix or NULL; if non-NULL, print avg L rows.
#' @param avg_T_mat      Matrix or NULL; if non-NULL, print avg T rows.
#' @param num_MC         Number of MC trials (printed in header).
.print_solver_table <- function(solver_names,
                                snr_values,
                                fdr_mat,
                                tpr_mat,
                                avg_L_mat = NULL,
                                avg_T_mat = NULL,
                                num_MC    = NULL) {

  # Solver column grows to fit the longest name; 15 keeps the classic layout
  # for demos whose names all fit (mirrors trex_sim_utils.hpp).
  sw <- max(15L, max(nchar(solver_names)) + 2L)
  mw <- 8L    # metric column width
  nw <- 5L    # SNR label column width
  cw <- 10L   # data column width

  n_snr <- length(snr_values)

  if (!is.null(num_MC)) {
    cat(sprintf(
      "\n%s\n=== T-Rex Results (averaged over %d Monte Carlo runs) ===\n%s\n\n",
      strrep("=", 70L), num_MC, strrep("=", 70L)
    ))
  }

  # Header
  hdr <- sprintf("%-*s%-*s%*s", sw, "Solver", mw, "Metric", nw, "SNR")
  for (snr in snr_values) {
    hdr <- paste0(hdr, sprintf("%*.1f", cw, snr))
  }

  cat(hdr, "\n")
  cat(strrep("-", sw + mw + nw + cw * n_snr), "\n")

  print_row <- function(sname, metric, vals, first_row) {

    row <- sprintf("%-*s%-*s%*s",
                   sw, if (first_row) sname else "",
                   mw, metric,
                   nw, "")

    for (v in vals) {
      row <- paste0(row, sprintf("%*.4f", cw, v))
    }
    cat(row, "\n")
  }

  for (i in seq_along(solver_names)) {
    nm <- solver_names[[i]]
    print_row(nm, "FDR", fdr_mat[i, ], TRUE)
    print_row(nm, "TPR", tpr_mat[i, ], FALSE)
    if (!is.null(avg_L_mat)) print_row(nm, "Avg L", avg_L_mat[i, ], FALSE)
    if (!is.null(avg_T_mat)) print_row(nm, "Avg T", avg_T_mat[i, ], FALSE)
    cat("\n")
  }
}


# ==============================================================================
# .save_mc_csv
# ==============================================================================

#' Write a tidy long-format CSV (solver, metric, snr, value).
#'
#' @param filepath     Full path to the output CSV file.
#' @param solver_names Character vector of solver names.
#' @param snr_values   Numeric vector of SNR grid values.
#' @param fdr_mat      Matrix (n_solvers x n_snr) of mean FDR values.
#' @param tpr_mat      Matrix (n_solvers x n_snr) of mean TPR values.
#' @param avg_L_mat    Matrix or NULL.
#' @param avg_T_mat    Matrix or NULL.
.save_mc_csv <- function(filepath,
                         solver_names,
                         snr_values,
                         fdr_mat,
                         tpr_mat,
                         avg_L_mat = NULL,
                         avg_T_mat = NULL) {

  rows <- list()
  for (i in seq_along(solver_names)) {
    nm <- solver_names[[i]]
    for (j in seq_along(snr_values)) {

      snr_str <- sprintf("%.6f", snr_values[[j]])
      rows <- c(rows,
        list(data.frame(solver = nm, metric = "FDR", snr = snr_str,
                        value = fdr_mat[i, j], stringsAsFactors = FALSE)),
        list(data.frame(solver = nm, metric = "TPR", snr = snr_str,
                        value = tpr_mat[i, j], stringsAsFactors = FALSE))
      )

      if (!is.null(avg_L_mat))
        rows <- c(rows, list(data.frame(solver = nm, metric = "AvgL",
                                        snr = snr_str, value = avg_L_mat[i, j],
                                        stringsAsFactors = FALSE)))
      if (!is.null(avg_T_mat))
        rows <- c(rows, list(data.frame(solver = nm, metric = "AvgT",
                                        snr = snr_str, value = avg_T_mat[i, j],
                                        stringsAsFactors = FALSE)))
    }
  }

  out_df <- do.call(rbind, rows)
  utils::write.csv(out_df, filepath, row.names = FALSE, quote = FALSE)
  cat(sprintf("[Info] CSV results saved to: %s\n", filepath))
}


# ==============================================================================
# .save_and_print_mc
# ==============================================================================

#' Print results table (console + .txt file) and write tidy CSV.
#'
#' @param out_dir      Output directory (must already exist).
#' @param file_stem    Output file base name (no directory, no extension).
#' @param num_MC       Number of MC trials.
#' @param solver_names Character vector of solver names.
#' @param snr_values   Numeric vector of SNR grid values.
#' @param fdr_mat      Matrix (n_solvers x n_snr) of mean FDR values.
#' @param tpr_mat      Matrix (n_solvers x n_snr) of mean TPR values.
#' @param avg_L_mat    Matrix or NULL.
#' @param avg_T_mat    Matrix or NULL.
.save_and_print_mc <- function(out_dir,
                               file_stem,
                               num_MC,
                               solver_names,
                               snr_values,
                               fdr_mat,
                               tpr_mat,
                               avg_L_mat = NULL,
                               avg_T_mat = NULL) {

  # Write to both console and text file via connection trick
  txt_path <- file.path(out_dir, paste0(file_stem, ".txt"))
  con      <- file(txt_path, open = "wt")
  sink(con, split = TRUE)
  on.exit({
    sink()
    close(con)
  }, add = TRUE)

  .print_solver_table(solver_names, snr_values, fdr_mat, tpr_mat,
                      avg_L_mat, avg_T_mat, num_MC)

  sink()
  close(con)
  on.exit(NULL)
  cat(sprintf("[Info] Results saved to:     %s\n", txt_path))

  .save_mc_csv(
    filepath     = file.path(out_dir, paste0(file_stem, ".csv")),
    solver_names = solver_names,
    snr_values   = snr_values,
    fdr_mat      = fdr_mat,
    tpr_mat      = tpr_mat,
    avg_L_mat    = avg_L_mat,
    avg_T_mat    = avg_T_mat
  )
}

# ==============================================================================
