# ==============================================================================
# demo_trex_06_mmap.R
# ==============================================================================
#
# Classical T-Rex selector demo.
# Part 5 demonstrates memory-mapped usage patterns in the R package TRexSelector.
#
# The file mirrors:
# cpp/trex_selector_methods/ad/trex_core/demo_trex_06_mmap.cpp
#
# Demo content:
# ----------------------------------------------------------------
#
#  Part A: In-memory X + use_memory_mapping = TRUE.
#          Enables internal D-mmap and solver serialization within
#          TRexSelector (via trex_control(use_memory_mapping = TRUE)).
#          Exact mirror of C++ Demo A.
#
#
#  Part B: Fully memory-mapped X + use_memory_mapping = TRUE.
#          In C++ this uses SyntheticDataMapped (X never fully in RAM).
#          The R package TRexSelector supports MemoryMappedMatrix objects
#          for X (via mem_matrix_from / mem_matrix), but creating
#          a memory-mapped version of a generated matrix requires writing it
#          to disk first.
#          This part mirrors the statistical scenario
#          (same n, p, support, snr, ctrl) while demonstrating the R mmap
#          interface for completeness.
#
# Output (mirrors C++/Python): per configuration one .txt (dual console +
# file, with the sparse Phi_prime/Phi block) and one per-variable selection
# .csv, both in simulation_results/data/.
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
source(file.path(this_dir_, "..", "..", "..", "simulation_utils.R"))
source(file.path(this_dir_, "..", "trex_sim_utils.R"))


# ==============================================================================
# Global Simulation Parameters
# ==============================================================================

OUT_DIR <- file.path(this_dir_, "simulation_results", "data")
dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)

# Run flags
run_part_a <- TRUE
run_part_b <- TRUE

# Shared demo config (mirrors make_mmap_demo_config() in C++)
# C++ true_support (0-based): {27, 149, 398, 420, 4}
# R (1-based): {28, 150, 399, 421, 5}
DEMO_CFG <- list(
  true_support = c(28L, 150L, 399L, 421L, 5L),
  true_coefs   = c(1, 1, 1, 1, 1),   # rnd_coef = FALSE
  snr          = 1.0,
  tFDR         = 0.1
)


# ==============================================================================
# Shared helper
# ==============================================================================

#' Print and save single-run results (dual console + file via `out`).
#'
#' Mirrors Python _print_and_save() in demo_trex_06_mmap.py: the C++-style
#' selection-result block (sparse Phi_prime/Phi, dense FDP_hat/R, voting grid)
#' followed by the T_stop / L line, plus the per-variable selection CSV.
.print_and_save_d02 <- function(stem, dat, selector, tFDR, out = .console_out) {
  true_support <- dat$true_support
  selected     <- selector$selected_indices

  .print_selection_results(selected, true_support, selector, out)
  out(sprintf("\nT_stop = %d,  L = %d  (target tFDR <= %.2f)\n\n",
              selector$T_stop, selector$L, tFDR))

  # Save per-variable CSV (mirrors save_selection_csv() in C++)
  p          <- ncol(dat$X)
  phi_prime  <- selector$phi_prime
  is_sel     <- seq_len(p) %in% selected
  is_true    <- seq_len(p) %in% true_support
  csv_df     <- data.frame(
    variable_index = seq_len(p) - 1L,   # 0-based to match C++ output
    phi_prime      = round(phi_prime, 6L),
    selected       = as.integer(is_sel),
    true_positive  = as.integer(is_true)
  )
  csv_path <- file.path(OUT_DIR, paste0(stem, ".csv"))
  utils::write.csv(csv_df, csv_path, row.names = FALSE, quote = FALSE)
  cat(sprintf("[Info] CSV saved to: %s\n\n", csv_path))
}


# ==============================================================================
# Part A: In-memory X + use_memory_mapping = TRUE
# ==============================================================================
# Mirrors demo_TRexSelector_d_mmap_solver_serial() in C++.

trex_05_part_a <- function() {

  run_single <- function(high_dim) {
    cfg       <- DEMO_CFG
    cfg$n     <- if (high_dim) 1000L else 5000L
    cfg$p     <- if (high_dim) 5000L else 1000L
    dim_label <- if (high_dim) "High-dimensional (p > n)" else "Low-dimensional (n > p)"

    stem     <- sprintf("d02_trex_mmap_demo_a_n%d_p%d", cfg$n, cfg$p)
    txt_path <- file.path(OUT_DIR, paste0(stem, ".txt"))
    con      <- file(txt_path, open = "wt")
    on.exit(close(con), add = TRUE)
    out      <- .make_dual_out(con)

    out(sprintf("\n%s\n", strrep("=", 70L)))
    out("Part A: In-Memory X  +  use_memory_mapping = TRUE\n")
    out("  Activates D-mmap and solver serialization inside TRexSelector.\n")
    out(sprintf("  %s  (n = %d, p = %d)\n\n", dim_label, cfg$n, cfg$p))

    ctrl <- make_mmap_trex_ctrl(solver = "TLARS")

    out("Generating synthetic data (in-memory) ...\n")
    dat <- dgp_gauss_snr(cfg$n, cfg$p, cfg$true_support, coefs = cfg$true_coefs,
                         snr = cfg$snr, seed = 58L)

    out("Running T-Rex Selector ...\n\n")
    selector <- TRexSelector$new(
      X       = dat$X,
      y       = dat$y,
      tFDR    = cfg$tFDR,
      seed    = 58L, # deterministic seed for reproducibility
      verbose = TRUE,
      control = ctrl
    )
    selector$select()

    .print_and_save_d02(stem, dat, selector, cfg$tFDR, out = out)
    cat(sprintf("[Info] Results saved to: %s\n", txt_path))
  }

  run_single(high_dim = FALSE)
  run_single(high_dim = TRUE)
}


# ==============================================================================
# Part B: Memory-mapped X + use_memory_mapping = TRUE
# ==============================================================================
# In C++: uses SyntheticDataMapped so X is never fully in RAM.
# In R:   demonstrates the TRexSelector MemoryMappedMatrix interface by
#         writing X to a temporary mmap file before passing it to the selector.
#         use_memory_mapping = TRUE adds D-mmap + solver serialization on top.

trex_05_part_b <- function() {

  run_single <- function(high_dim) {
    cfg       <- DEMO_CFG
    cfg$n     <- if (high_dim) 1000L else 5000L
    cfg$p     <- if (high_dim) 5000L else 1000L
    dim_label <- if (high_dim) "High-dimensional (p > n)" else "Low-dimensional (n > p)"

    stem     <- sprintf("d02_trex_mmap_demo_b_n%d_p%d", cfg$n, cfg$p)
    txt_path <- file.path(OUT_DIR, paste0(stem, ".txt"))
    con      <- file(txt_path, open = "wt")
    on.exit(close(con), add = TRUE)
    out      <- .make_dual_out(con)

    out(sprintf("\n%s\n", strrep("=", 70L)))
    out("Part B: Memory-Mapped X  +  use_memory_mapping = TRUE\n")
    out("  Full mmap pipeline: X mmap + D mmap + solver serialization.\n")
    out(sprintf("  %s  (n = %d, p = %d)\n\n", dim_label, cfg$n, cfg$p))

    ctrl <- make_mmap_trex_ctrl(solver = "TLARS")

    # Generate data in-memory, then convert X to a memory-mapped file.
    out("Generating synthetic data ...\n")
    dat <- dgp_gauss_snr(cfg$n, cfg$p, cfg$true_support, coefs = cfg$true_coefs,
                         snr = cfg$snr, seed = 58L)

    out("Converting X to memory-mapped file and running T-Rex Selector ...\n\n")

    # Selector construction + select() + result printing all run INSIDE the
    # callback — the temp backing file is unlinked on return (see
    # execute_with_temp_mmap()).
    execute_with_temp_mmap(dat$X, function(X_mmap) {
      selector <- TRexSelector$new(
        X       = X_mmap,
        y       = dat$y,
        tFDR    = cfg$tFDR,
        seed    = 58L, # deterministic seed for reproducibility
        verbose = TRUE,
        control = ctrl
      )
      selector$select()
      .print_and_save_d02(stem, dat, selector, cfg$tFDR, out = out)
      invisible(NULL)
    })
    cat(sprintf("[Info] Results saved to: %s\n", txt_path))
  }

  run_single(high_dim = FALSE)
  run_single(high_dim = TRUE)
}


# ==============================================================================
# Main
# -------------------
# Run parts A and B
# ==============================================================================

if (run_part_a) trex_05_part_a()
if (run_part_b) trex_05_part_b()
cat("\ndemo_trex_06_mmap.R complete.\n")
