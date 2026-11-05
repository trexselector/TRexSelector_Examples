# ==============================================================================
# demo_trex_05_mc_sim_dummy_distributions.R
# ==============================================================================
#
# Classical T-Rex selector demo: a Monte Carlo simulation comparing the dummy
# distributions available for the T-Rex Selector across three base solvers
# (TLARS, TOMP, TAFS), with the STANDARD L-loop strategy held fixed.
#
# Base solvers compared (one result file pair per solver):
#   TLARS — equiangular LARS path (stagnation stop AUTO-resolves to disabled).
#   TOMP  — greedy orthogonal matching pursuit (stagnation stop AUTO-resolves
#           to enabled).
#   TAFS  — greedy adaptive forward selection, rho_afs = 0.3 (stagnation stop
#           AUTO-resolves to enabled).
#
# Mirrors:
# cpp/trex_selector_methods/ad/trex_core/demo_trex_05_mc_sim_dummy_distributions.cpp
#
# The T-Rex FDR calibration rests on exchangeability between real null
# predictors and the injected dummy variables. Since dummies are centered and
# L2-normalized before entering the solver, their scale is immaterial — this
# demo probes whether the SHAPE of the dummy distribution (tails, discreteness,
# skewness, sparsity, cross-column dependence) affects the achieved FDR / TPR
# and the calibrated dummy multiplier L / stopping time T.
#
# Demo content (see trex_dummy_distribution()):
# ----------------------------------------------------------------
#
#    Normal              — N(0, 1); the reference choice.
#    Uniform             — U(-sqrt(3), sqrt(3)); bounded, light tails.
#    Rademacher          — {-1, +1} equiprobable; discrete two-point.
#    StudentT (df 3 / 5) — heavy tails (unit-variance scaled for df > 2).
#    Laplace             — double exponential; heavier-than-normal tails.
#    Gumbel              — extreme-value; skewed (centered at 0).
#    Triangle            — bounded, symmetric; (-sqrt(6), 0, sqrt(6)).
#    Logistic            — between normal and Laplace tails.
#    Mammen              — asymmetric golden-ratio two-point distribution.
#    SparseRademacher    — constrained sparse Rademacher, sparsity s = 0.1.
#    UniformSphere       — uniform on the unit 5-sphere; consecutive 5-column
#                          groups are dependent (norm constraint), a deliberate
#                          probe of the exchangeability axis. The library
#                          requests dummies in multiples of p, so dim must
#                          divide p (the default dim = 3 would throw at
#                          p = 1000, hence dim = 5).
#
#  Scenario:
#  ----------
#  High-dimensional  (n = 300, p = 1000, s = 10).
#  SNR sweep: {0.1, 0.2, 0.5, 0.6, 1.0, 2.0, 5.0}  (7 values, mirrors the
#  current C++ grid and its committed results).
#
#  Two support scenarios:
#    random support  — redrawn per trial;
#    block support   — contiguous block {1, 2, ..., s}.
#  Reports averaged FDR / TPR / Avg L / Avg T per distribution x SNR.
#
# ==============================================================================

library(TRexSelector)
library(parallel)

# ==============================================================================
# Setup
# ==============================================================================

this_dir_ <- tryCatch(
  dirname(normalizePath(sys.frame(1L)$ofile)),
  error = function(e) {
    args <- commandArgs(trailingOnly = FALSE)
    file_arg <- grep("--file=", args, value = TRUE)
    if (length(file_arg) > 0) dirname(normalizePath(sub("--file=", "", file_arg[1]))) else "."
  }
)

source(file.path(this_dir_, "..", "..", "..", "mc_runtime.R"))  # PSOCK workers, cell cache
source(file.path(this_dir_, "..", "..", "..", "support_generators.R"))
source(file.path(this_dir_, "..", "..", "..", "simulation_utils.R"))
source(file.path(this_dir_, "..", "trex_sim_utils.R"))


# ==============================================================================
# Global Simulation Parameters
# ==============================================================================

OUT_DIR <- file.path(this_dir_, "simulation_results", "data")
dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)

# Worker processes: first CLI argument, else TREX_NUM_WORKERS, else 8 (0 = all).
num_cores <- local({
  a <- suppressWarnings(as.integer(commandArgs(trailingOnly = TRUE)[1L]))
  if (is.na(a)) a <- suppressWarnings(as.integer(Sys.getenv("TREX_NUM_WORKERS", "8")))
  if (is.na(a)) 8L else if (a < 1L) parallel::detectCores() else a
})
NUM_MC <- 200L
run_trex_05 <- TRUE

# ==============================================================================
# Base solvers to compare (outer sweep; rho_afs only meaningful for TAFS)
# ==============================================================================

SOLVERS <- list(
  list(name = "TLARS", solver = "TLARS", lambda2 = 0.0, rho_afs = 0.0,
       ncgmp_variant = 0L),
  list(name = "TOMP",  solver = "TOMP",  lambda2 = 0.0, rho_afs = 0.0,
       ncgmp_variant = 0L),
  list(name = "TAFS",  solver = "TAFS",  lambda2 = 0.0, rho_afs = 0.3,
       ncgmp_variant = 0L)
)


# ==============================================================================
# Dummy distributions to compare
# ==============================================================================

# name : row label used in output tables and CSV.
# dist : list built by trex_dummy_distribution(), passed to
#        trex_control(dummy_distribution = ...).

.make_dist_entry <- function(name, dist) {
  list(name = name, dist = dist)
}

DISTRIBUTIONS <- list(
  .make_dist_entry("Normal",         trex_dummy_distribution("Normal")),
  .make_dist_entry("Uniform",        trex_dummy_distribution("Uniform")),
  .make_dist_entry("Rademacher",     trex_dummy_distribution("Rademacher")),
  .make_dist_entry("StudentT_df3",   trex_dummy_distribution("StudentT", df = 3)),
  .make_dist_entry("StudentT_df5",   trex_dummy_distribution("StudentT", df = 5)),
  .make_dist_entry("Laplace",        trex_dummy_distribution("Laplace")),
  .make_dist_entry("Gumbel",         trex_dummy_distribution("Gumbel")),
  .make_dist_entry("Triangle",       trex_dummy_distribution("Triangle")),
  .make_dist_entry("Logistic",       trex_dummy_distribution("Logistic")),
  .make_dist_entry("Mammen",         trex_dummy_distribution("Mammen")),
  .make_dist_entry("SparseRad_s0.1",
                   trex_dummy_distribution("ConstrainedSparseRademacher", s = 0.1)),
  .make_dist_entry("UnifSphere_d5",
                   trex_dummy_distribution("UniformSphere", dim = 5L))
)


# ==============================================================================
# MC simulation — variable true support, dummy distribution comparison
# ==============================================================================

trx_05_dummy_distributions <- function(num_MC = NUM_MC, rnd_coef = FALSE,
                                       block_support = FALSE) {

  n    <- 300L
  p    <- 1000L
  s    <- 10L
  tFDR <- 0.10
  snr_values <- c(0.1, 0.2, 0.5, 0.6, 1.0, 2.0, 5.0)  # mirrors current C++ grid

  cat("\n", strrep("=", 70L), "\n", sep = "")
  cat("MC Simulation \xe2\x80\x94 Dummy Distribution Comparison  |  TLARS / TOMP / TAFS\n")
  cat("  High-dimensional (p > n)\n")
  cat(sprintf("  n = %d,  p = %d,  s = %d,  num_MC = %d  [%s support]\n\n",
              n, p, s, num_MC, if (block_support) "block" else "random"))

  n_dist <- length(DISTRIBUTIONS)
  n_snr  <- length(snr_values)
  dist_names <- vapply(DISTRIBUTIONS, `[[`, "", "name")

  # Solver sweep — one result file pair (txt + csv) per solver
  for (solver_entry in SOLVERS) {

    # Result containers: distribution x SNR (reset per solver)
    fdr_mat   <- matrix(0.0, nrow = n_dist, ncol = n_snr)
    tpr_mat   <- matrix(0.0, nrow = n_dist, ncol = n_snr)
    avg_L_mat <- matrix(0.0, nrow = n_dist, ncol = n_snr)
    avg_T_mat <- matrix(0.0, nrow = n_dist, ncol = n_snr)
    rownames(fdr_mat) <- rownames(tpr_mat) <-
      rownames(avg_L_mat) <- rownames(avg_T_mat) <- dist_names
    colnames(fdr_mat) <- colnames(tpr_mat) <-
      colnames(avg_L_mat) <- colnames(avg_T_mat) <- as.character(snr_values)


    # Simulation sweep: iterate over dummy distributions and SNR values
    for (di in seq_along(DISTRIBUTIONS)) {
      dist_entry <- DISTRIBUTIONS[[di]]

      # Per-distribution ctrl — only dummy_distribution varies. Values taken
      # from the C++ reference (demo_trex_05_mc_sim_dummy_distributions.cpp)
      # — every option explicit. tloop_stagnation_stop is left unset (NULL =
      # AUTO, C++ nullopt): resolved per solver family (disabled for TLARS,
      # enabled for the greedy TOMP/TAFS).
      ctrl <- TRexSelector::trex_control(
        solver                   = solver_entry$solver,
        lambda2                  = solver_entry$lambda2,
        rho_afs                  = solver_entry$rho_afs,
        ncgmp_variant            = solver_entry$ncgmp_variant,
        K                        = 20L,
        max_dummy_multiplier     = 10L,
        use_max_T_stop           = TRUE,
        opt_threshold            = 0.75,
        lloop_strategy           = "STANDARD",
        dummy_distribution       = dist_entry$dist,
        tloop_max_stagnant_steps = 5L,
        parallel_rnd_experiments = FALSE,
        use_memory_mapping       = FALSE
      )

      cat(sprintf("%s\nSolver: %s  |  Dummy distribution: %s\n%s\n\n",
                  strrep("=", 50L), solver_entry$name, dist_entry$name,
                  strrep("=", 50L)))

      for (snr_idx in seq_along(snr_values)) {
        snr       <- snr_values[[snr_idx]]
        base_seed <- 24L + (snr_idx - 1L) * 1000L

        make_data <- local({
          current_snr <- snr
          use_block   <- block_support
          function(mc, trial_seed) {
            trial_support <- if (use_block) seq_len(s) else
              make_support_random(s, p, seed = trial_seed)
            trial_coefs <- rep(1.0, s)
            dgp_gauss_snr(n, p, trial_support, coefs = trial_coefs,
                          snr = current_snr, seed = trial_seed)
          }
        })

        lbl    <- sprintf("SNR=%.2f [%s/%s]", snr, solver_entry$name, dist_entry$name)
        result <- .run_mc_trex(make_data, ctrl, tFDR, base_seed, num_MC, num_cores,
                               track_L_T = TRUE, label = lbl)

        fdr_mat[di, snr_idx]   <- result$mean_FDR
        tpr_mat[di, snr_idx]   <- result$mean_TPR
        avg_L_mat[di, snr_idx] <- result$mean_L
        avg_T_mat[di, snr_idx] <- result$mean_T
      }
      cat("\n")
    }

    stem <- sprintf("demo_trex_05_dummy_distributions_results_n%d_p%d_%s_%s",
                    n, p, tolower(solver_entry$name),
                    if (block_support) "block_support" else "random_support")
    .save_and_print_mc(OUT_DIR, stem, num_MC, dist_names, snr_values,
                       fdr_mat, tpr_mat, avg_L_mat, avg_T_mat)
    cat("\n")

  }  # end solver sweep
}


# ==============================================================================
# Main
# --------------
# Run demo 05
# ==============================================================================
if (run_trex_05) {
  trx_05_dummy_distributions(NUM_MC, rnd_coef = FALSE, block_support = FALSE)
  # trx_05_dummy_distributions(NUM_MC, rnd_coef = FALSE, block_support = TRUE)
}
cat("\ndemo_trex_05_mc_sim_dummy_distributions.R complete.\n")

# All results are written: the cell cache of a stopped run is no longer needed.
cell_cache_done()
