# ==============================================================================
# demo_trex_04_lloop_strategies.R
# ==============================================================================
#
# Classical T-Rex selector demo, where in part 4 a Monte Carlo simulation
# compares the L-loop strategies implemented in the TRexSelector, repeated
# across four base solvers (TLARS / TOMP / TAFS / TLARS_STAG = TLARS with the
# stagnation stop on).
#
# Mirrors:
# cpp/trex_selector_methods/ad/trex_core/demo_trex_04_lloop_strategies.cpp
# and
# Python/trex_selector_methods/ad/trex_core/demo_trex_04_mc_sim_lloop_strategies.py
#
# Demo content (8 strategy rows):
# ----------------------------------------------------------------
#
#    STANDARD             — Fresh i.i.d. dummy matrix at each L-loop iteration
#                           (conservative gold standard; highest memory cost).
#    HCONCAT              — Horizontally expand dummy columns from the same
#                           base draw.
#    PERMUTATION          — Re-use the stored base dummy matrix via
#                           deterministic row permutations per experiment.
#    PERMUTATION_SEEDED   — Seed-derived base + row permutations; nothing
#                           stored. Bit-identical to PERMUTATION per seed.
#    SEEDED               — Seed-derived i.i.d. draws; nothing stored.
#    SKIPL_{5p,10p,20p}   — Skip the L-loop entirely; fixed
#                           L = max_dummy_multiplier in {5, 10, 20} x p.
#
#  Scenario:
#  ----------
#  High-dimensional  (n = 300, p = 1000, s = 10).
#  SNR sweep: {0.1, 0.2, ..., 2.0, 5.0}  (21 values).
#  Base solvers (outer sweep, one result file pair per solver):
#    TLARS, TOMP, TAFS (rho_afs = 0.3), TLARS_STAG (stagnation stop on).
#
#  Two support scenarios:
#    random support  — redrawn per trial;
#    block support   — contiguous block {1, 2, ..., s}.
#  Reports averaged FDR / TPR / Avg L / Avg T per L-strategy x SNR.
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
run_trex_04 <- TRUE

# ==============================================================================
# L-loop strategies to compare in Part 4
# ==============================================================================

# All L-loop strategy variants to compare.
# name           : row label used in output tables and CSV.
# strategy       : string passed to trex_control(lloop_strategy = ...).
# max_dummy_mult : upper bound for the adaptive L-loop (non-SKIPL) or the fixed
#                  L value for SKIPL (as a multiple of p).

.make_strat_entry <- function(name, strategy, max_dummy_mult = 10L) {
  list(name = name, strategy = strategy, max_dummy_mult = max_dummy_mult)
}

L_STRATEGIES <- list(
  .make_strat_entry("STANDARD",           "STANDARD"),
  .make_strat_entry("HCONCAT",            "HCONCAT"),
  .make_strat_entry("PERMUTATION",        "PERMUTATION"),
  .make_strat_entry("PERMUTATION_SEEDED", "PERMUTATION_SEEDED"),
  .make_strat_entry("SEEDED",             "SEEDED"),
  .make_strat_entry("SKIPL_5p",  "SKIPL",  5L),
  .make_strat_entry("SKIPL_10p", "SKIPL", 10L),
  .make_strat_entry("SKIPL_20p", "SKIPL", 20L)
)

# Base solvers for the outer sweep (rho_afs only meaningful for TAFS).
# Deviations between the solvers will be detectable in the results.
# TLARS_STAG is TLARS with the stagnation stop ON (AUTO leaves it off for
# TLARS): does the SKIPL overshoot come from the LARS path or the missing stop?
SOLVERS <- list(
  list(solver = "TLARS", name = "TLARS", lambda2 = 0.0, rho_afs = 0.0,
       ncgmp_variant = 0L),
  list(solver = "TOMP",  name = "TOMP",  lambda2 = 0.0, rho_afs = 0.0,
       ncgmp_variant = 0L),
  list(solver = "TAFS",  name = "TAFS",  lambda2 = 0.0, rho_afs = 0.3,
       ncgmp_variant = 0L),
  list(solver = "TLARS", name = "TLARS_STAG", lambda2 = 0.0, rho_afs = 0.0,
       ncgmp_variant = 0L, stagnation_stop = TRUE)
)


# ==============================================================================
# MC simulation — variable true support, L-strategy comparison
# ==============================================================================

trx_04_lloop_strategies <- function(num_MC = NUM_MC, rnd_coef = FALSE,
                                    block_support = FALSE) {

  n    <- 300L
  p    <- 1000L
  s    <- 10L
  tFDR <- 0.10
  snr_values <- c(seq(0.1, 2.0, by = 0.1), 5.0)

  support_label <- if (block_support) "block" else "random"
  cat("\n", strrep("=", 70L), "\n", sep = "")
  cat("MC Simulation — L-Strategy Comparison  |  TLARS / TOMP / TAFS / TLARS_STAG\n")
  cat("  High-dimensional (p > n)\n")
  cat(sprintf("  n = %d,  p = %d,  s = %d,  num_MC = %d  [%s support]\n\n",
              n, p, s, num_MC, support_label))

  n_strat <- length(L_STRATEGIES)
  n_snr   <- length(snr_values)
  strat_names <- vapply(L_STRATEGIES, `[[`, "", "name")

  # Solver sweep — one result file pair (txt + csv) per base solver
  for (solver_entry in SOLVERS) {

    # Result containers: strategy x SNR (reset per solver)
    fdr_mat   <- matrix(0.0, nrow = n_strat, ncol = n_snr)
    tpr_mat   <- matrix(0.0, nrow = n_strat, ncol = n_snr)
    avg_L_mat <- matrix(0.0, nrow = n_strat, ncol = n_snr)
    avg_T_mat <- matrix(0.0, nrow = n_strat, ncol = n_snr)
    rownames(fdr_mat) <- rownames(tpr_mat) <-
      rownames(avg_L_mat) <- rownames(avg_T_mat) <- strat_names
    colnames(fdr_mat) <- colnames(tpr_mat) <-
      colnames(avg_L_mat) <- colnames(avg_T_mat) <- as.character(snr_values)

    # Strategy sweep: iterate over L-loop strategies and SNR values
    for (si in seq_along(L_STRATEGIES)) {
      strat <- L_STRATEGIES[[si]]

      # Per-strategy ctrl — max_dummy_multiplier and lloop_strategy vary.
      # Values taken from the C++ reference — every option explicit.
      # tloop_stagnation_stop NULL = AUTO (C++ nullopt): it is resolved per
      # solver family (disabled for TLARS, enabled for the greedy TOMP/TAFS).
      # TLARS_STAG sets it to TRUE. tloop_max_stagnant_steps = 5 is the C++
      # struct default (only active when the stop is on).
      ctrl <- TRexSelector::trex_control(
        solver                   = solver_entry$solver,
        lambda2                  = solver_entry$lambda2,
        rho_afs                  = solver_entry$rho_afs,
        ncgmp_variant            = solver_entry$ncgmp_variant,
        K                        = 20L,
        max_dummy_multiplier     = strat$max_dummy_mult,
        use_max_T_stop           = TRUE,
        opt_threshold            = 0.75,
        lloop_strategy           = strat$strategy,
        tloop_stagnation_stop    = solver_entry$stagnation_stop,
        tloop_max_stagnant_steps = 5L,
        parallel_rnd_experiments = FALSE,
        use_memory_mapping       = FALSE
      )

      cat(sprintf("%s\nSolver: %s  |  L-strategy: %s\n%s\n\n",
                  strrep("=", 50L), solver_entry$name, strat$name,
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

        lbl    <- sprintf("SNR=%.2f [%s]", snr, strat$name)
        result <- .run_mc_trex(make_data, ctrl, tFDR, base_seed, num_MC, num_cores,
                               track_L_T = TRUE, label = lbl)

        fdr_mat[si, snr_idx]   <- result$mean_FDR
        tpr_mat[si, snr_idx]   <- result$mean_TPR
        avg_L_mat[si, snr_idx] <- result$mean_L
        avg_T_mat[si, snr_idx] <- result$mean_T
      }
      cat("\n")
    }

    stem <- sprintf("demo_trex_04_lloop_strategies_results_n%d_p%d_%s_%s_support",
                    n, p, tolower(solver_entry$name), support_label)
    .save_and_print_mc(OUT_DIR, stem, num_MC, strat_names, snr_values,
                       fdr_mat, tpr_mat, avg_L_mat, avg_T_mat)
    cat("\n")

  }  # end solver sweep
}


# ==============================================================================
# Main
# --------------
# Run demo 04
# ==============================================================================
if (run_trex_04) {
  trx_04_lloop_strategies(NUM_MC, rnd_coef = FALSE, block_support = FALSE)
  # trx_04_lloop_strategies(NUM_MC, rnd_coef = FALSE, block_support = TRUE)
}
cat("\ndemo_trex_04_lloop_strategies.R complete.\n")

# All results are written: the cell cache of a stopped run is no longer needed.
cell_cache_done()
