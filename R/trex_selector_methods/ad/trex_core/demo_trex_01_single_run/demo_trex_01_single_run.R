# ==============================================================================
# demo_trex_01_single_run.R
# ==============================================================================
#
# Classical T-Rex Selector demo, part 01 is a single-run demonstration of the
# with the TLARS solver.
# A data-generating process functionality `dgp_gauss_snr` supports the demo
# and allows for modifications.
#
# The file mirrors:
# cpp/trex_selector_methods/ad/trex_core/demo_trex_01_single_run.cpp
# and
# Python/trex_selector_methods/ad/trex_core/demo_trex_01_single_run.py
#
# Demo content:
# ----------------------------------------------------------------
#
#  - High- and low-dimensional settings (run in that order, as in C++).
#  - Fixed true support; fixed +1 coefficients; TLARS solver.
#  - Reports selected indices, FDP/TPP, Phi_prime and Phi (printed sparsely),
#    FDP_hat, R, and the voting grid — dual console + file output to
#    simulation_results/data/d01_trex_basic_n{n}_p{p}.txt.
#
# Seed note: data seed 58 and selector seed 42 mirror the C++/Python demos
# structurally; the RNG streams differ across languages, so the drawn data
# (and hence the exact selection) are not bit-identical to the C++ output.
#
# ==============================================================================

library(TRexSelector)

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
source(file.path(this_dir_, "..", "..", "..", "support_generators.R"))
source(file.path(this_dir_, "..", "trex_sim_utils.R"))

# ==============================================================================
# Global Simulation Parameters
# ==============================================================================

OUT_DIR <- file.path(this_dir_, "simulation_results", "data")
dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)

# Fixed true support (0-based in C++, converted to 1-based)
# C++ (0-based): {27, 149, 43, 128, 42, 4}
# R (1-based): c(28, 150, 44, 129, 43, 5)
TRUE_SUPPORT_P1 <- c(28L, 150L, 44L, 129L, 43L, 5L)

run_part_01 <- TRUE


# ==============================================================================
# Part 1: Single-run basic demo
# ==============================================================================

trx_01_part_01 <- function() {

  # Parameters matching C++ demo_TRexSelector()
  tFDR      <- 0.1
  snr       <- 1.0
  data_seed <- 58L
  trex_seed <- 42L

  # Complete control block, values taken from the C++ reference
  # (demo_trex_01_single_run.cpp) — every option explicit, no reliance on
  # trex_control() defaults.
  ctrl <- trex_control(
    solver                   = "TLARS",
    K                        = 20L,
    max_dummy_multiplier     = 10L,
    use_max_T_stop           = TRUE,
    opt_threshold            = 0.75,
    lloop_strategy           = "STANDARD",
    tloop_stagnation_stop    = FALSE,
    tloop_max_stagnant_steps = 5L,
    parallel_rnd_experiments = FALSE,
    use_memory_mapping       = FALSE,
    lambda2                  = 0.1,
    rho_afs                  = 0.0,   # non-TAFS solver; rho_afs is unused
    ncgmp_variant            = 0L
  )


  run_single <- function(high_dim) {

    n <- if (high_dim) 150L else 5000L
    p <- if (high_dim) 300L else 1000L
    dim_label <- if (high_dim) "High-dimensional (p > n)" else "Low-dimensional (n > p)"

    txt_path <- file.path(OUT_DIR, sprintf("d01_trex_basic_n%d_p%d.txt", n, p))
    con      <- file(txt_path, open = "wt")
    on.exit(close(con), add = TRUE)
    out      <- .make_dual_out(con)

    out(sprintf("\n%s\n", strrep("=", 70L)))
    out(sprintf("Part 1: Single-run basic T-Rex demo  |  %s\n", dim_label))
    out(sprintf("n = %d,  p = %d,  tFDR = %.2f,  SNR = %.1f\n\n", n, p, tFDR, snr))

    out("Generating synthetic data ...\n")
    dat <- dgp_gauss_snr(n, p, TRUE_SUPPORT_P1, amplitude = 1, snr = snr,
                         seed = data_seed)

    out("Running T-Rex Selector (TLARS) ...\n\n")
    selector <- TRexSelector$new(
      X       = dat$X,
      y       = dat$y,
      tFDR    = tFDR,
      seed    = trex_seed, # using a != -1 seed for reproducibility in the demo
      verbose = TRUE,
      control = ctrl
    )
    selector$select()

    .print_single_run(
      sprintf("Part 1 — i.i.d. Gaussian  [%s, TLARS]", dim_label),
      dat, selector, tFDR, out = out
    )

    cat(sprintf("[Info] Results saved to: %s\n", txt_path))
    invisible(selector)
  }

  # C++ main() runs the high-dimensional scenario first, then the
  # low-dimensional one; both with fixed +1 coefficients.
  run_single(high_dim = TRUE)
  run_single(high_dim = FALSE)
}


# ==============================================================================
# Main
# ------------
# Run the demo
# ==============================================================================

if (run_part_01) trx_01_part_01()
cat("\ndemo_trex_01_single_run.R complete.\n")
