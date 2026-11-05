# ==============================================================================
# trex_scr_sim_utils.R
# ==============================================================================
#
# Shared Monte Carlo infrastructure for the Screen-TRex R demonstration suite.
#
# Mirrors the Python suite modules
#   Python/trex_selector_methods/ad/trex_screening/trex_scr_common.py
#   Python/trex_selector_methods/ad/trex_screening/trex_scr_bb_common.py
# and, through them, the C++ demo-internal headers
#   cpp/trex_selector_methods/ad/trex_screening/trex_screening_dgps.hpp
#   cpp/trex_selector_methods/ad/trex_screening/trex_screening_simulation_utils.hpp
#
# Provides:
#   Section 1 — data-generating processes (i.i.d., AR(1), equi, block-equi),
#   Section 2 — Screen-TRex control-parameter factories,
#   Section 3 — parallel Monte Carlo runner + dual TXT/CSV table output,
#   Section 4 — Biobank (Algorithm 1) MC helpers + table output.
#
# Parallelism: Monte Carlo trials run on spawned PSOCK workers via mc_map() of
# mc_runtime.R (no fork), which frees the C++ selector state after every trial;
# set.seed() inside a trial is process-local. Every grid point is a resumable
# cell (cell_cached()), restored from the cell cache after a restart.
#
# NOTE on reproducibility: the C++ demos draw with std::mt19937 and the Python
# suite with NumPy's Generator; R uses its own RNG stream. Seeds are mirrored
# as *labels*, but the realised random draws differ, so R results are
# statistically comparable to the C++/Python reference — not bit-identical.
#
# Support sets and selected indices are 1-BASED throughout (R convention); the
# C++/Python sources are 0-based.
# ==============================================================================

library(TRexSelector)


# ==============================================================================
# Section 1 — Data-generating processes
# ==============================================================================

#' Evenly-spaced sparse beta vector (mirrors make_beta() in the cpp DGPs).
#'
#' Active indices are placed at round(i*p/(p1+2)) for i = 1..p1 (0-based in
#' C++), converted to 1-based R indices.
#'
#' @param p        Total number of predictors.
#' @param p1       Number of active predictors.
#' @param beta_val Coefficient value for active predictors.
#' @return Numeric vector of length p with p1 non-zero, evenly spaced entries.
make_beta <- function(p, p1, beta_val) {
  beta <- numeric(p)
  for (i in seq_len(p1)) {
    idx0 <- round(i * p / (p1 + 2))       # 0-based position (C++)
    beta[idx0 + 1L] <- beta_val            # 1-based (R)
  }
  beta
}

#' SNR-calibrated response y = X beta + eps, then mean-centred.
#' Mirrors make_response(): sigma = sqrt(||X beta||^2 / (n * snr)).
.make_response <- function(X, beta, snr) {
  n      <- nrow(X)
  signal <- as.numeric(X %*% beta)
  sigma  <- sqrt(sum(signal^2) / (n * snr))
  y      <- signal + rnorm(n, 0, sigma)
  y - mean(y)
}

#' Standardise columns to mean 0 and unit variance (n - 1 denominator).
.standardise_cols <- function(X) {
  X <- sweep(X, 2L, colMeans(X), "-")
  sds <- sqrt(colSums(X^2) / (nrow(X) - 1L))
  sds[sds < 1e-12] <- 1
  sweep(X, 2L, sds, "/")
}

#' i.i.d. Gaussian DGP (mirrors make_iid_dgp / datagen::SyntheticData).
#'
#' X ~ N(0, 1) i.i.d.; y = X beta + eps with SNR-calibrated noise.
#'
#' @param n,p          Dimensions.
#' @param true_support 1-based indices of active variables.
#' @param coefs        Coefficients for the active variables.
#' @param snr          Signal-to-noise ratio.
#' @param seed         RNG seed.
#' @return list(X, y, true_support).
dgp_iid_snr <- function(n, p, true_support, coefs, snr, seed) {
  set.seed(seed)
  X <- matrix(rnorm(n * p), n, p)
  beta <- numeric(p)
  beta[true_support] <- coefs
  y <- .make_response(X, beta, snr)
  list(X = X, y = y, true_support = sort(true_support))
}

#' AR(1) column-correlated DGP (mirrors make_ar1_dgp).
dgp_ar1 <- function(n, p, p1, rho, snr, beta_val, seed) {
  set.seed(seed)
  innov_sd <- sqrt(1 - rho^2)
  X <- matrix(0, n, p)
  X[, 1L] <- rnorm(n)
  for (j in 2:p) X[, j] <- rho * X[, j - 1L] + innov_sd * rnorm(n)
  X <- .standardise_cols(X)
  beta <- make_beta(p, p1, beta_val)
  y <- .make_response(X, beta, snr)
  list(X = X, y = y, true_support = which(beta != 0))
}

#' Equicorrelated DGP (mirrors make_equi_dgp).
#' x_j = sqrt(rho) * f + sqrt(1 - rho) * w_j, shared factor f per observation.
dgp_equi <- function(n, p, p1, rho, snr, beta_val, seed) {
  set.seed(seed)
  load_f   <- sqrt(rho)
  load_eta <- sqrt(1 - rho)
  f <- rnorm(n)
  X <- load_f * matrix(f, n, p) + load_eta * matrix(rnorm(n * p), n, p)
  X <- .standardise_cols(X)
  beta <- make_beta(p, p1, beta_val)
  y <- .make_response(X, beta, snr)
  list(X = X, y = y, true_support = which(beta != 0))
}

#' Block-equicorrelated DGP (mirrors make_block_equi_dgp).
#' Contiguous blocks, each with its own shared factor.
dgp_block_equi <- function(n, p, p1, rho, n_blocks, snr, beta_val, seed) {
  set.seed(seed)
  load_f    <- sqrt(rho)
  load_eta  <- sqrt(1 - rho)
  base_size <- p %/% n_blocks
  remainder <- p %% n_blocks
  X <- matrix(0, n, p)
  col <- 0L
  for (k in seq_len(n_blocks)) {
    bk <- base_size + if (k <= remainder) 1L else 0L
    z_k <- rnorm(n)                                   # shared factor for block k
    idx <- (col + 1L):(col + bk)
    X[, idx] <- load_f * matrix(z_k, n, bk) +
      load_eta * matrix(rnorm(n * bk), n, bk)
    col <- col + bk
  }
  X <- .standardise_cols(X)
  beta <- make_beta(p, p1, beta_val)
  y <- .make_response(X, beta, snr)
  list(X = X, y = y, true_support = which(beta != 0))
}


# ==============================================================================
# Section 2 — Control-parameter factories
# ==============================================================================

#' Default TRexControlParameter for Screen-TRex (mirrors make_trex_control()).
#'
#' @param solver             T-Rex solver backend (default "TLARS").
#' @param use_memory_mapping Enable D-mmap + solver serialization.
#' @param ...                Extra trex_control() overrides (e.g. rho_afs).
make_scr_trex_ctrl <- function(solver = "TLARS",
                               use_memory_mapping = FALSE, ...) {
  trex_control(
    solver               = solver,
    K                    = 20L,
    max_dummy_multiplier = 10L,
    use_max_T_stop       = TRUE,
    lloop_strategy       = "STANDARD",
    use_memory_mapping   = use_memory_mapping,
    ...
  )
}

#' Screen-TRex method descriptors for the Ordinary-vs-Bootstrap comparison
#' (mirrors default_scr_methods() in trex_scr_common.py).
default_scr_methods <- function() {
  list(
    list(name = "Screen-TRex Ordinary: TLARS",     trex_method = "TREX", bootstrap = FALSE),
    list(name = "Screen-TRex Bootstrap-CI: TLARS", trex_method = "TREX", bootstrap = TRUE)
  )
}


# ==============================================================================
# Section 3 — Parallel Monte Carlo runner + table output (Screen-TRex)
# ==============================================================================

#' Internal: run one_trial over mc = 0..num_MC-1 on PSOCK workers (mc_map() of
#' mc_runtime.R, which the demos source first). Each worker is its own R
#' process, so per-trial set.seed() calls never interleave, and mc_map() frees
#' the selectors' C++ memory after every trial.
.mc_map <- function(num_MC, one_trial, num_cores) {
  # mc index runs 0..num_MC-1 (matches the cpp/Python seed bands)
  mc_map(0:(num_MC - 1L), one_trial, num_cores)
}

#' Run num_MC Screen-TRex trials for one grid point in parallel.
#'
#' Mirrors run_mc_screen() in trex_scr_common.py: trial mc uses seed
#' base_seed + mc (mc = 0..num_MC-1) and a selector seed of -1 (hardware
#' entropy per trial, as in the cpp MC loop).
#'
#' @param num_MC       Monte Carlo trials.
#' @param label        Progress label.
#' @param make_data    function(seed) -> list(X, y, true_support).
#' @param trex_ctrl    trex_control() object (constant across trials).
#' @param screen_args  named list of trex_screen_control() arguments.
#' @param base_seed    Trial seed base.
#' @param num_cores    Worker processes for mc_map().
#' @return list(avg_fdr, avg_tpr, avg_est_fdr, sd_fdr, sd_tpr); a resumable grid
#'   cell (cell_cached()).
run_mc_screen <- function(num_MC, label, make_data, trex_ctrl,
                          screen_args, base_seed, num_cores = 1L) {
  one_trial <- function(mc) {
    dat <- make_data(base_seed + mc)
    sc  <- TRexScreeningSelector$new(
      dat$X, dat$y, seed = -1L, verbose = FALSE,
      screen_control = do.call(trex_screen_control, screen_args),
      control        = trex_ctrl
    )
    sc$select()
    sel <- sc$selected_indices
    c(fdp = trex_fdp(sel, dat$true_support),
      tpp = trex_tpp(sel, dat$true_support),
      est = sc$estimated_FDR)
  }
  key <- paste0("scr|", label, "|", cell_digest(trex_ctrl, screen_args, base_seed, num_MC))
  out <- cell_cached(key, function() {
    res_mat <- do.call(rbind, .mc_map(num_MC, one_trial, num_cores))
    list(
      avg_fdr     = mean(res_mat[, "fdp"]),
      avg_tpr     = mean(res_mat[, "tpp"]),
      avg_est_fdr = mean(res_mat[, "est"]),
      sd_fdr      = if (num_MC > 1L) stats::sd(res_mat[, "fdp"]) else 0.0,
      sd_tpr      = if (num_MC > 1L) stats::sd(res_mat[, "tpp"]) else 0.0
    )
  })
  cat(sprintf("  %s - done.  TPR=%.4f  FDR=%.4f  Est.FDR=%.4f\n",
              label, out$avg_tpr, out$avg_fdr, out$avg_est_fdr))
  out
}

# ------------------------------------------------------------------------------
# Table layout (mirrors trex_scr_common.py / save_and_print_mc_results():
# method name on its own line, metric rows indented beneath it, sweep values
# right-aligned so the columns stay put regardless of name length).
# ------------------------------------------------------------------------------
.SCR_INDENT_W <- 2L
.SCR_METRIC_W <- 23L
.SCR_COL_W    <- 10L

# Internal: one indented, right-aligned numeric row of a sweep table.
.scr_metric_row <- function(label, vec, dp = 4L) {
  paste0(strrep(" ", .SCR_INDENT_W),
         formatC(label, width = -.SCR_METRIC_W, flag = "-"),
         paste(formatC(sprintf(paste0("%.", dp, "f"), vec),
                       width = .SCR_COL_W), collapse = ""))
}

# Internal: sweep-grid header row + separator width.
.scr_header <- function(sweep_label, x_values) {
  paste0(strrep(" ", .SCR_INDENT_W),
         formatC(sweep_label, width = -.SCR_METRIC_W, flag = "-"),
         paste(formatC(sprintf("%.2f", x_values), width = .SCR_COL_W),
               collapse = ""))
}

# Internal: write a tidy long-format CSV (method, metric, <axis>, value) with
# axis and value formatted at %.6f (mirrors the Python float handling).
.scr_write_csv <- function(rows, axis_label, csv_path) {
  csv_df <- do.call(rbind, rows)
  csv_df[[3L]] <- sprintf("%.6f", csv_df[[3L]])
  csv_df$value <- sprintf("%.6f", csv_df$value)
  names(csv_df)[3L] <- axis_label
  utils::write.csv(csv_df, csv_path, row.names = FALSE, quote = FALSE)
}

#' Save + print a sweep-variable Screen-TRex MC table (mirrors
#' save_and_print_scr_mc() in trex_scr_common.py). Writes a human-readable TXT
#' table (also echoed to console) and a tidy long-format CSV.
#'
#' @param num_MC       Monte Carlo trials.
#' @param out_dir      Output directory.
#' @param file_stem    File stem (no extension).
#' @param x_values     Sweep grid (SNR or rho values).
#' @param method_names Ordered method names.
#' @param fdr_map,tpr_map,est_fdr_map Named lists: method -> numeric vector.
#' @param sweep_label  Column label (default "SNR").
save_and_print_scr_mc <- function(num_MC, out_dir, file_stem, x_values,
                                  method_names, fdr_map, tpr_map, est_fdr_map,
                                  sweep_label = "SNR") {
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
  lines <- character(0)
  acc <- environment()  # the report lines that add() appends to
  add <- function(...) acc$lines[[length(lines) + 1L]] <- paste0(...)

  add("")
  add(strrep("=", 70L))
  add(sprintf("=== Screen-TRex Results (averaged over %d Monte Carlo runs) ===", num_MC))
  add(strrep("=", 70L))
  add("")

  add(.scr_header(sweep_label, x_values))
  add(strrep("-", .SCR_INDENT_W + .SCR_METRIC_W + .SCR_COL_W * length(x_values)))

  for (nm in method_names) {
    add("")
    add(nm)                                # method name on its own line
    add(.scr_metric_row("FDR",      fdr_map[[nm]]))
    add(.scr_metric_row("TPR",      tpr_map[[nm]]))
    add(.scr_metric_row("Est. FDR", est_fdr_map[[nm]]))
  }
  add("")

  cat(paste0(paste(lines, collapse = "\n"), "\n"))
  txt_path <- file.path(out_dir, paste0(file_stem, ".txt"))
  writeLines(lines, txt_path)
  cat(sprintf("[Info] TXT results saved to: %s\n", txt_path))

  rows <- list()
  for (nm in method_names) {
    for (i in seq_along(x_values)) {
      rows[[length(rows) + 1L]] <- data.frame(
        method = nm, metric = "FDR",      x = x_values[i], value = fdr_map[[nm]][i]
      )
      rows[[length(rows) + 1L]] <- data.frame(
        method = nm, metric = "TPR",      x = x_values[i], value = tpr_map[[nm]][i]
      )
      rows[[length(rows) + 1L]] <- data.frame(
        method = nm, metric = "Est. FDR", x = x_values[i], value = est_fdr_map[[nm]][i]
      )
    }
  }
  csv_path <- file.path(out_dir, paste0(file_stem, ".csv"))
  .scr_write_csv(rows, sweep_label, csv_path)
  cat(sprintf("[Info] CSV results saved to: %s\n\n", csv_path))
}


# ==============================================================================
# Section 4 — Biobank (Algorithm 1) MC helpers
# ==============================================================================
#
# Mirrors trex_scr_bb_common.py. The biobank workflow ("Algorithm 1") screens
# MANY phenotypes against ONE shared design matrix X. Per phenotype one
# Screen-TRex run yields an Ordinary and a Bootstrap-CI estimated FDR, and the
# selector routes adaptively:
#   1. a rule whose estimate lies in [lower_bound_FDR, upper_bound_FDR] is used,
#   2. if both do, the larger estimate wins (a tie goes to Bootstrap-CI),
#   3. if neither does, the classical T-Rex selector runs at tFDR (fallback).
# The fraction of phenotypes routed to each method is the "Usage %". FDR/TPR
# are accumulated conditionally on which method handled a phenotype; the
# estimated-FDR series are unconditional.
# ==============================================================================

# Canonical method-name keys returned in each record's `method_used` field.
BIOBANK_METHODS <- c("Screen-TRex (ordinary)",
                     "Screen-TRex (bootstrap-CI)",
                     "T-Rex (fallback)")

# Display names used in the saved tables (mirror the Python/cpp display names).
BIOBANK_DISPLAY <- c("Screen-TRex Ordinary: TLARS",
                     "Screen-TRex Bootstrap-CI: TLARS",
                     "T-Rex fallback: TLARS")

#' Internal: build and run a biobank selector (mirrors makeBiobankControl()).
#'
#' @param X,Y                Design matrix and phenotype vector/matrix.
#' @param use_memory_mapping Back the dummy matrices by on-disk files.
#' @param R_boot             Bootstrap replicates for the CI rule.
#' @param tFDR               Target FDR of the classical T-Rex fallback.
#' @param seed               Selector seed (-1 = hardware entropy).
#' @return select() result: one record (vector y) or a list of records (matrix Y).
.run_biobank_selector <- function(X, Y, use_memory_mapping = FALSE,
                                  R_boot = 1000L, tFDR = 0.10, seed = -1L) {
  sel <- TRexBiobankScreeningSelector$new(
    X, Y, tFDR = tFDR, seed = seed, verbose = FALSE,
    biobank_control = trex_biobank_control(lower_bound_FDR = 0.05,
                                           upper_bound_FDR = 0.15),
    screen_control  = trex_screen_control(R_boot = R_boot),
    control         = make_scr_trex_ctrl(use_memory_mapping = use_memory_mapping)
  )
  sel$select()
}

#' Internal: one per-phenotype record (mirrors _record() in trex_scr_bb_common.py).
.biobank_record <- function(rec, true_support) {
  sel <- rec$selected_indices
  list(method_used   = rec$method_used,
       fdp           = trex_fdp(sel, true_support),
       tpp           = trex_tpp(sel, true_support),
       est_ordinary  = rec$estimated_FDR_screen_ordinary,
       est_bootstrap = rec$estimated_FDR_screen_bootstrap)
}

#' One single-phenotype Algorithm-1 trial (mirrors _biobank_single_worker()).
#'
#' Support drawn from an RNG stream offset from the trial seed (by 500000),
#' matching the cpp isolation of the support draw from design/noise.
#'
#' @return List of one per-phenotype record.
biobank_single_trial <- function(n, p, support_size, snr, seed,
                                 use_memory_mapping = FALSE, R_boot = 1000L,
                                 tFDR = 0.10) {
  set.seed(seed + 500000L)                 # isolated support RNG
  sup <- sample.int(p, support_size)
  dat <- dgp_iid_snr(n, p, sup, rep(1, support_size), snr, seed = seed)
  rec <- .run_biobank_selector(dat$X, dat$y, use_memory_mapping, R_boot, tFDR)
  list(.biobank_record(rec, dat$true_support))
}

#' One multi-phenotype Algorithm-1 trial (mirrors _biobank_multi_worker()).
#'
#' All q phenotypes share the design matrix X drawn with x_seed; phenotype k
#' draws its own support and noise from seed sup_seed_base + k - 1 (0-based k
#' in the Python reference).
#'
#' @return List of q per-phenotype records.
biobank_multi_trial <- function(n, p, q, support_size, snr, x_seed,
                                sup_seed_base, use_memory_mapping = FALSE,
                                R_boot = 1000L, tFDR = 0.10) {
  set.seed(x_seed)
  X <- matrix(rnorm(n * p), n, p)          # shared design matrix
  true_supports <- vector("list", q)
  Y <- matrix(0, n, q)
  for (k in seq_len(q)) {
    set.seed(sup_seed_base + (k - 1L))
    sup <- sort(sample.int(p, support_size))
    true_supports[[k]] <- sup
    beta <- numeric(p)
    beta[sup] <- 1
    Y[, k] <- .make_response(X, beta, snr)
  }
  res <- .run_biobank_selector(X, Y, use_memory_mapping, R_boot, tFDR)
  lapply(seq_len(q), function(k) .biobank_record(res[[k]], true_supports[[k]]))
}

#' Run num_MC biobank trials in parallel; return the flat list of records.
#'
#' Mirrors run_mc_biobank() in trex_scr_bb_common.py. trial_fn(mc) (with
#' mc = 0..num_MC-1) returns a list of per-phenotype records (length 1 for a
#' single-phenotype trial, q for a multi-phenotype trial).
#'
#' @param num_MC    Monte Carlo trials.
#' @param trial_fn  function(mc) -> list of per-phenotype records.
#' @param num_cores Worker processes for mc_map().
run_mc_biobank <- function(num_MC, trial_fn, num_cores = 1L) {
  chunks <- .mc_map(num_MC, trial_fn, num_cores)
  do.call(c, chunks)
}

#' Aggregate one SNR level's records into per-method statistics
#' (mirrors accumulate_snr() in trex_scr_bb_common.py).
#'
#' FDR/TPR are conditional on which method a phenotype was routed to; the
#' estimated-FDR series are unconditional. Normalization is over the total
#' number of phenotype screenings (q * num_MC for multi-phenotype trials).
#'
#' @param records         Flat list of per-phenotype records.
#' @param target_FDR_trex Target FDR of the classical T-Rex fallback.
#' @return Named list keyed by BIOBANK_METHODS with fields
#'   usage (fraction in [0,1]), fdp, tpp, est.
accumulate_snr <- function(records, target_FDR_trex) {
  total <- length(records)
  z <- setNames(numeric(length(BIOBANK_METHODS)), BIOBANK_METHODS)
  cnt <- z
  fdp <- z
  tpp <- z
  est <- z

  for (r in records) {
    m <- r$method_used
    cnt[[m]] <- cnt[[m]] + 1
    fdp[[m]] <- fdp[[m]] + r$fdp
    tpp[[m]] <- tpp[[m]] + r$tpp
    est[["Screen-TRex (ordinary)"]]     <- est[["Screen-TRex (ordinary)"]]     + r$est_ordinary
    est[["Screen-TRex (bootstrap-CI)"]] <- est[["Screen-TRex (bootstrap-CI)"]] + r$est_bootstrap
    est[["T-Rex (fallback)"]]           <- est[["T-Rex (fallback)"]]           + target_FDR_trex
  }

  out <- list()
  for (m in BIOBANK_METHODS) {
    out[[m]] <- list(
      usage = if (total > 0) cnt[[m]] / total else 0.0,
      fdp   = if (cnt[[m]] > 0) fdp[[m]] / cnt[[m]] else 0.0,   # conditional mean
      tpp   = if (cnt[[m]] > 0) tpp[[m]] / cnt[[m]] else 0.0,   # conditional mean
      est   = if (total > 0) est[[m]] / total else 0.0          # unconditional mean
    )
  }
  out
}

#' One SNR level of a biobank sweep as a resumable grid cell:
#' run_mc_biobank() + accumulate_snr(), restored from the cell cache after a
#' restart. Same return as accumulate_snr().
#'
#' @param label           Cell label naming the part and SNR level (the trial
#'                        function itself cannot be told apart by its code).
#' @param num_MC          Monte Carlo trials.
#' @param trial_fn        function(mc) -> list of per-phenotype records.
#' @param num_cores       Worker processes for mc_map().
#' @param target_FDR_trex Target FDR of the classical T-Rex fallback.
run_biobank_snr_cell <- function(label, num_MC, trial_fn, num_cores, target_FDR_trex) {
  fields <- c("usage", "fdp", "tpp", "est")
  flat <- cell_cached(paste0("biobank|", label, "|", cell_digest(num_MC, target_FDR_trex)), function() {
    agg <- accumulate_snr(run_mc_biobank(num_MC, trial_fn, num_cores), target_FDR_trex)
    unlist(lapply(seq_along(BIOBANK_METHODS), function(j) {
      setNames(vapply(fields, function(k) agg[[BIOBANK_METHODS[j]]][[k]], numeric(1)),
               paste0("m", j, ".", fields))
    }))
  })
  setNames(lapply(seq_along(BIOBANK_METHODS), function(j) {
    setNames(lapply(fields, function(k) flat[[paste0("m", j, ".", k)]]), fields)
  }), BIOBANK_METHODS)
}

#' Save + print a Biobank MC table with per-method Usage (%) rows
#' (mirrors save_and_print_biobank_mc() in trex_scr_bb_common.py).
#'
#' fdp_map/tpp_map/est_fdr_map/usage_map are keyed by display_names; each
#' value is a per-SNR numeric vector. Usage is printed to 1 dp, the rest to 4.
save_and_print_biobank_mc <- function(num_MC, out_dir, file_stem, snr_values,
                                      display_names, fdp_map, tpp_map,
                                      est_fdr_map, usage_map) {
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
  lines <- character(0)
  acc <- environment()  # the report lines that add() appends to
  add <- function(...) acc$lines[[length(lines) + 1L]] <- paste0(...)

  add("")
  add(strrep("=", 70L))
  add(sprintf("=== Biobank Screen-TRex Results (averaged over %d Monte Carlo runs) ===",
              num_MC))
  add(strrep("=", 70L))
  add("")

  add(.scr_header("SNR", snr_values))
  add(strrep("-", .SCR_INDENT_W + .SCR_METRIC_W + .SCR_COL_W * length(snr_values)))

  for (nm in display_names) {
    add("")
    add(nm)                                # method name on its own line
    add(.scr_metric_row("Usage (%)", usage_map[[nm]] * 100, dp = 1L))
    add(.scr_metric_row("FDR",       fdp_map[[nm]]))
    add(.scr_metric_row("TPR",       tpp_map[[nm]]))
    add(.scr_metric_row("Est. FDR",  est_fdr_map[[nm]]))
  }
  add("")

  cat(paste0(paste(lines, collapse = "\n"), "\n"))
  txt_path <- file.path(out_dir, paste0(file_stem, ".txt"))
  writeLines(lines, txt_path)
  cat(sprintf("[Info] TXT results saved to: %s\n", txt_path))

  rows <- list()
  for (nm in display_names) {
    for (i in seq_along(snr_values)) {
      rows[[length(rows) + 1L]] <- data.frame(method = nm, metric = "Usage",
                                              snr = snr_values[i], value = usage_map[[nm]][i])
      rows[[length(rows) + 1L]] <- data.frame(method = nm, metric = "FDR",
                                              snr = snr_values[i], value = fdp_map[[nm]][i])
      rows[[length(rows) + 1L]] <- data.frame(method = nm, metric = "TPR",
                                              snr = snr_values[i], value = tpp_map[[nm]][i])
      rows[[length(rows) + 1L]] <- data.frame(method = nm, metric = "Est. FDR",
                                              snr = snr_values[i], value = est_fdr_map[[nm]][i])
    }
  }
  csv_path <- file.path(out_dir, paste0(file_stem, ".csv"))
  .scr_write_csv(rows, "snr", csv_path)
  cat(sprintf("[Info] CSV results saved to: %s\n\n", csv_path))
}

# ==============================================================================
# End of trex_scr_sim_utils.R
# ==============================================================================
