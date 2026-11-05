# ==============================================================================
# demo_trex_da_04_mc_sim_ht_blocks.R
# ==============================================================================
#
# DA-TRex+BT MC simulations for the heavy-tailed block DGP (BT aggregation).
#
# DGP: dgp_block_toeplitz_hvt_snr — block-diagonal Toeplitz covariance;
# predictors are multivariate t(nu=3) in both scenarios, and the response noise
# is Gaussian ("s1_Gauss") or t(nu=3) ("s2_Heavy"). M blocks of size Q each,
# one active per block (s = M).
#
# Outer loops: h_noise in {FALSE ("s1_Gauss"), TRUE ("s2_Heavy")} and linkage
# in {Single, Complete, Average}. All 3 solvers (TLARS, TAFS, TOMP) run inside
# each linkage iteration.
#
#   Part 1: SNR sweep      SNR in {0.1, ..., 5.0}.
#   Part 2: rho sweep      rho in {0.0, ..., 0.9}.
#   Part 3: Q sweep        Q in {5, ..., 50}.
#   Part 4: M sweep        M in {1, ..., 30}.
#   Part 5: tFDR sweep     tFDR in {0.05, ..., 0.50}.
#   Part 6: linkage sweep  linkage codes {1=Single, 2=Complete, 3=Average}.
#
# Mirrors:
#  cpp/trex_selector_methods/ad/trex_da/demo_trex_da_04_mc_sim_ht_blocks
#  Python/trex_selector_methods/ad/trex_da/demo_trex_da_04_mc_sim_ht_blocks
#
# Results are saved to simulation_results/data/ as TXT and CSV files.
# Optional first CLI argument sets the number of parallel worker cores
# (default TREX_NUM_WORKERS, else 8). Indices are 1-based (R convention).
# ==============================================================================

library(TRexSelector)
library(parallel)

# Worker processes: first CLI argument, else TREX_NUM_WORKERS, else 8 (0 = all).
num_cores <- local({
  a <- suppressWarnings(as.integer(commandArgs(trailingOnly = TRUE)[1L]))
  if (is.na(a)) a <- suppressWarnings(as.integer(Sys.getenv("TREX_NUM_WORKERS", "8")))
  if (is.na(a)) 8L else if (a < 1L) parallel::detectCores() else a
})

this_dir_ <- tryCatch(
  dirname(normalizePath(sys.frame(1)$ofile)),
  error = function(e) {
    args <- commandArgs(trailingOnly = FALSE)
    flag <- grep("^--file=", args, value = TRUE)
    if (length(flag) > 0L)
      dirname(normalizePath(sub("^--file=", "", flag[1L]))) else "."
  }
)

source(file.path(this_dir_, "..", "..", "..", "mc_runtime.R"))  # PSOCK workers, cell cache
source(file.path(this_dir_, "..", "..", "..", "support_generators.R"))
source(file.path(this_dir_, "..", "trex_da_dgps.R"))
source(file.path(this_dir_, "..", "da_sim_common.R"))

RUN_PART_1 <- TRUE
RUN_PART_2 <- TRUE
RUN_PART_3 <- TRUE
RUN_PART_4 <- TRUE
RUN_PART_5 <- TRUE
RUN_PART_6 <- TRUE

OUT_DIR <- file.path(this_dir_, "simulation_results", "data")

# Shared base parameters (mirror cpp)
N <- 150L
M <- 5L
Q <- 5L
BASE_P <- M * Q      # 25
AMPLITUDE <- 1.0
RHO <- 0.8
NU <- 3.0
SNR <- 2.0
TFDR <- 0.2
K <- 20L
NUM_MC <- 200L
BASE_SEED <- 2026L

SCENARIOS <- list(list(tag = "s1_Gauss", h_noise = FALSE),
                  list(tag = "s2_Heavy", h_noise = TRUE))
LINKAGES <- c("Single", "Complete", "Average")

SUPPORT_BASE <- make_support_one_per_block(M, BASE_P, Q)


.kwargs <- function(n_blocks, rho, snr, h_noise) {
  list(amplitude = AMPLITUDE, n_blocks = n_blocks, rho = rho, snr = snr,
       nu = NU, heavy_X = TRUE, heavy_noise = h_noise)
}


# ==============================================================================
# Part 1: MC SNR sweep
# Fixed: n=150, M=5, Q=5 (p=25, s=5), rho=0.8, nu=3.0.
# ==============================================================================

part_1_snr_sweep <- function() {
  snr_grid <- c(0.1, 0.2, 0.5, 0.6, 1.0, 2.0, 5.0)

  for (scen in SCENARIOS) {
    for (lnk in LINKAGES) {
      h_noise <- scen$h_noise
      run_snr_sweep(
        OUT_DIR, sprintf("da_ht_blocks_snr_%s_%s", scen$tag, lnk), snr_grid,
        NUM_MC, TFDR, BASE_SEED, default_solvers("+BT"),
        function(snr) {
          list(n = N, p = BASE_P,
               dgp = "dgp_block_toeplitz_hvt_snr",
               kwargs = .kwargs(M, RHO, snr, h_noise),
               support = SUPPORT_BASE, s = M)
        },
        list(method = "BT", hc_linkage = lnk), K,
        sprintf(paste0("HT-block SNR sweep  n=%d  M=%d  Q=%d  p=%d  s=%d",
                       "  rho=%s  nu=%s  noise=%s  linkage=%s"),
                N, M, Q, BASE_P, M, fmt_num(RHO), fmt_num(NU), scen$tag, lnk),
        include_base_trex = FALSE
      )
    }
  }
}


# ==============================================================================
# Part 2: MC rho sweep
# Fixed: n=150, M=5, Q=5 (p=25, s=5), SNR=2, nu=3.0.
# ==============================================================================

part_2_rho_sweep <- function() {
  rho_grid <- seq(0.0, 0.9, by = 0.1)

  for (scen in SCENARIOS) {
    for (lnk in LINKAGES) {
      h_noise <- scen$h_noise
      run_param_sweep(
        OUT_DIR, sprintf("da_ht_blocks_rho_%s_%s", scen$tag, lnk), "Rho",
        rho_grid, NUM_MC, TFDR, BASE_SEED, default_solvers("+BT"),
        function(rho) {
          list(n = N, p = BASE_P,
               dgp = "dgp_block_toeplitz_hvt_snr",
               kwargs = .kwargs(M, rho, SNR, h_noise),
               support = SUPPORT_BASE, s = M)
        },
        list(method = "BT", hc_linkage = lnk), K,
        sprintf(paste0("HT-block rho sweep  n=%d  M=%d  Q=%d  s=%d",
                       "  SNR=%s  nu=%s  noise=%s  linkage=%s"),
                N, M, Q, M, fmt_num(SNR), fmt_num(NU), scen$tag, lnk),
        include_base_trex = FALSE
      )
    }
  }
}


# ==============================================================================
# Part 3: MC Q sweep (block size)
# Fixed: n=150, M=5, rho=0.8, SNR=2, nu=3.0.  s = M = 5 throughout.
# Swept: Q in {5, 10, ..., 50}; p = M*Q varies.
# ==============================================================================

part_3_q_sweep <- function() {
  q_grid <- seq(5L, 50L, by = 5L)

  for (scen in SCENARIOS) {
    h_noise <- scen$h_noise
    make_dgp <- function(q) {
      q_int <- as.integer(q)
      p_val <- M * q_int
      list(n = N, p = p_val, dgp = "dgp_block_toeplitz_hvt_snr",
           kwargs = .kwargs(M, RHO, SNR, h_noise),
           support = make_support_one_per_block(M, p_val, q_int), s = M)
    }
    for (lnk in LINKAGES) {
      run_param_sweep(
        OUT_DIR, sprintf("da_ht_blocks_Q_%s_%s", scen$tag, lnk), "Q", q_grid,
        NUM_MC, TFDR, BASE_SEED, default_solvers("+BT"), make_dgp,
        list(method = "BT", hc_linkage = lnk), K,
        sprintf(paste0("HT-block Q sweep  n=%d  M=%d  s=%d  rho=%s",
                       "  SNR=%s  nu=%s  p=M*Q varies  noise=%s  linkage=%s"),
                N, M, M, fmt_num(RHO), fmt_num(SNR), fmt_num(NU), scen$tag,
                lnk),
        include_base_trex = FALSE
      )
    }
  }
}


# ==============================================================================
# Part 4: MC M sweep (number of blocks)
# Fixed: n=150, Q=5, rho=0.8, SNR=2, nu=3.0.
# Swept: M in {1, 3, 5, 10, 15, 20, 30}; p = M*Q and s = M both vary.
# ==============================================================================

part_4_m_sweep <- function() {
  m_grid <- c(1L, 3L, 5L, 10L, 15L, 20L, 30L)

  for (scen in SCENARIOS) {
    h_noise <- scen$h_noise
    make_dgp <- function(m) {
      m_int <- as.integer(m)
      p_val <- m_int * Q
      list(n = N, p = p_val, dgp = "dgp_block_toeplitz_hvt_snr",
           kwargs = .kwargs(m_int, RHO, SNR, h_noise),
           support = make_support_one_per_block(m_int, p_val, Q), s = m_int)
    }
    for (lnk in LINKAGES) {
      run_param_sweep(
        OUT_DIR, sprintf("da_ht_blocks_M_%s_%s", scen$tag, lnk), "M", m_grid,
        NUM_MC, TFDR, BASE_SEED, default_solvers("+BT"), make_dgp,
        list(method = "BT", hc_linkage = lnk), K,
        sprintf(paste0("HT-block M sweep  n=%d  Q=%d  rho=%s",
                       "  SNR=%s  nu=%s  p=M*Q and s=M vary  noise=%s",
                       "  linkage=%s"),
                N, Q, fmt_num(RHO), fmt_num(SNR), fmt_num(NU), scen$tag, lnk),
        include_base_trex = FALSE
      )
    }
  }
}


# ==============================================================================
# Part 5: MC tFDR sweep
# Fixed: n=150, M=5, Q=5 (p=25, s=5), rho=0.8, SNR=2, nu=3.0.
# Swept: tFDR in {0.05, 0.10, ..., 0.50}.
# (tFDR is a selector parameter — run_mc_trials called per solver x grid point,
#  each solver with its own exch_tie_alpha as in the cpp section.)
# ==============================================================================

part_5_tfdr_sweep <- function() {
  tfdr_grid <- seq(0.05, 0.50, by = 0.05)

  for (scen in SCENARIOS) {
    h_noise <- scen$h_noise
    for (lnk in LINKAGES) {
      da_spec <- list(method = "BT", hc_linkage = lnk)
      solvers <- default_solvers("+BT")
      names_ <- vapply(solvers, `[[`, character(1L), "name")
      zero <- stats::setNames(
        rep(list(numeric(length(tfdr_grid))), length(names_)), names_
      )
      fdr <- zero
      tpr <- zero
      sd_fdr <- zero
      sd_tpr <- zero
      avg_L <- zero
      avg_T <- zero

      for (sv in solvers) {
        nm <- sv$name
        for (gi in seq_along(tfdr_grid)) {
          tfdr <- tfdr_grid[gi]
          res <- run_mc_trials(
            NUM_MC, sprintf("tFDR = %s", fmt_num(tfdr)), N, BASE_P,
            "dgp_block_toeplitz_hvt_snr",
            .kwargs(M, RHO, SNR, h_noise),
            SUPPORT_BASE, M, tfdr, sv, K, da_spec,
            BASE_SEED + (gi - 1L) * 10000L
          )
          fdr[[nm]][gi] <- res$avg_fdr
          tpr[[nm]][gi] <- res$avg_tpr
          sd_fdr[[nm]][gi] <- res$sd_fdr
          sd_tpr[[nm]][gi] <- res$sd_tpr
          avg_L[[nm]][gi] <- res$avg_L
          avg_T[[nm]][gi] <- res$avg_T
        }
      }

      save_and_print_grid_results(
        OUT_DIR, sprintf("da_ht_blocks_tFDR_%s_%s", scen$tag, lnk), "tFDR",
        tfdr_grid, NUM_MC, names_, fdr, tpr, sd_fdr, sd_tpr, avg_L, avg_T,
        sprintf(paste0("HT-block tFDR sweep  n=%d  M=%d  Q=%d  p=%d  s=%d",
                       "  rho=%s  SNR=%s  nu=%s  noise=%s  linkage=%s"),
                N, M, Q, BASE_P, M, fmt_num(RHO), fmt_num(SNR), fmt_num(NU),
                scen$tag, lnk)
      )
    }
  }
}


# ==============================================================================
# Part 6: Linkage sweep
# Fixed: n=150, M=5, Q=5 (p=25, s=5), rho=0.8, SNR=2, nu=3.0, tFDR=0.2.
# Swept: linkage encoded as {1.0=Single, 2.0=Complete, 3.0=Average}.
# (Each solver runs with its own exch_tie_alpha, as in the cpp section.)
# ==============================================================================

part_6_linkage_sweep <- function() {
  lnk_vals <- c(1.0, 2.0, 3.0)

  for (scen in SCENARIOS) {
    h_noise <- scen$h_noise
    solvers <- default_solvers("+BT")
    names_ <- vapply(solvers, `[[`, character(1L), "name")
    zero <- stats::setNames(
      rep(list(numeric(length(lnk_vals))), length(names_)), names_
    )
    fdr <- zero
    tpr <- zero
    sd_fdr <- zero
    sd_tpr <- zero
    avg_L <- zero
    avg_T <- zero

    for (sv in solvers) {
      nm <- sv$name
      for (gi in seq_along(LINKAGES)) {
        lnk <- LINKAGES[gi]
        res <- run_mc_trials(
          NUM_MC, sprintf("Linkage = %s", lnk), N, BASE_P,
          "dgp_block_toeplitz_hvt_snr",
          .kwargs(M, RHO, SNR, h_noise),
          SUPPORT_BASE, M, TFDR, sv, K,
          list(method = "BT", hc_linkage = lnk),
          BASE_SEED + (gi - 1L) * 10000L
        )
        fdr[[nm]][gi] <- res$avg_fdr
        tpr[[nm]][gi] <- res$avg_tpr
        sd_fdr[[nm]][gi] <- res$sd_fdr
        sd_tpr[[nm]][gi] <- res$sd_tpr
        avg_L[[nm]][gi] <- res$avg_L
        avg_T[[nm]][gi] <- res$avg_T
      }
    }

    save_and_print_grid_results(
      OUT_DIR, sprintf("da_ht_blocks_linkage_%s", scen$tag), "Linkage",
      lnk_vals, NUM_MC, names_, fdr, tpr, sd_fdr, sd_tpr, avg_L, avg_T,
      sprintf(paste0("HT-block Linkage sweep  n=%d  M=%d  Q=%d  p=%d  s=%d",
                     "  rho=%s  SNR=%s  nu=%s  tFDR=%s  noise=%s",
                     "\n  Linkage codes: 1=Single  2=Complete  3=Average"),
              N, M, Q, BASE_P, M, fmt_num(RHO), fmt_num(SNR), fmt_num(NU),
              fmt_num(TFDR), scen$tag)
    )
  }
}


# ==============================================================================
# Main
# ==============================================================================

cat(sprintf("Running with %d worker cores\n", num_cores))

if (RUN_PART_1) part_1_snr_sweep()
if (RUN_PART_2) part_2_rho_sweep()
if (RUN_PART_3) part_3_q_sweep()
if (RUN_PART_4) part_4_m_sweep()
if (RUN_PART_5) part_5_tfdr_sweep()
if (RUN_PART_6) part_6_linkage_sweep()
cat("\nHeavy-tailed block BT MC simulations complete.\n")

# All results are written: the cell cache of a stopped run is no longer needed.
cell_cache_done()
