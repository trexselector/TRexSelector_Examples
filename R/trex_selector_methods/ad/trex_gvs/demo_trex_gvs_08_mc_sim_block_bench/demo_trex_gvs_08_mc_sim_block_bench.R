# =================================================================================
# demo_trex_gvs_08_mc_sim_block_bench.R
# =================================================================================
#
# Monte Carlo benchmark: value of prior group information vs. HAC
# discoverability on the block-equicorrelated DGP.
#
# Mirrors cpp/Python demo_trex_gvs_08_mc_sim_block_bench.
#
# Tests four method variants over a (rho x snr) grid:
#
#   M1 = T-Rex + EN, HAC-clustered groups
#   M2 = T-Rex + EN, oracle groups
#   M3 = T-Rex + IEN, HAC-clustered groups
#   M4 = T-Rex + IEN, oracle groups
#
# Headline scenario (WHOLE truth — active blocks are fully active, matching
# the grouped-selection premise of the T-Rex+GVS papers):
# rho sweeps across the HAC discoverability boundary set by corr_max = 0.5.
# At low rho the clustered methods see near-singletons and the oracle prior
# is the only source of group structure; as rho grows, HAC rediscovers the
# blocks and the oracle-vs-clustered gap closes.
#
# Cautionary appendix (INDIVIDUAL truth — 1 of block_size variables active):
# within a block the null siblings are exchangeable with the active member,
# so coordinate-level FDR control is unattainable by construction; run at a
# low/high rho pair to show the collapse scaling with correlation.
#
# Fixed benchmark constants:
#   n = 200, p = 500, G = 100, block_size = 5, n_active_blocks = 10
#   tFDR = 0.1, K = 20, corr_max = 0.5, b = 3.0
#   snr_grid = {0.1, 0.2, 0.5, 1.0, 2.0, 5.0}
#   rho_grid (WHOLE)      = {0.2, 0.35, 0.5, 0.65, 0.8, 0.9}
#   rho_grid (INDIVIDUAL) = {0.35, 0.9}
#   num_MC = 200 (the full run takes hours — reduce num_MC / the grids for a
#   quick smoke test; the optional first CLI argument sets the number of
#   parallel worker cores, default TREX_NUM_WORKERS, else 8)
#
# Seeds are staggered by grid-cell index so that each (rho, snr) cell uses a
# distinct seed band:
#
#   cell_base_seed = base_seed + cell_index * num_MC
#
# where cell_index = (i_rho - 1) * n_snr + (i_snr - 1) (i_rho indexes the
# scenario's own rho grid). All four methods see identical data within a trial.
# Results (one table per scenario) are written to simulation_results/data/ as
# .txt + .csv (gvs_block_bench_whole / gvs_block_bench_individual).
# Indices are 1-based (R convention); the Python counterpart is 0-based.
#
# =================================================================================

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
source(file.path(this_dir_, "..", "trex_gvs_dgps.R"))
source(file.path(this_dir_, "..", "trex_gvs_sim_common.R"))

OUT_DIR <- file.path(this_dir_, "simulation_results", "data")


# ==============================================================================
# Configuration  (mirrors the cpp/Python BlockBenchPreset)
# ==============================================================================

CFG <- list(
  n               = 200L,
  p               = 500L,
  G               = 100L,
  block_size      = 5L,
  n_active_blocks = 10L,
  tFDR            = 0.1,
  K               = 20L,
  corr_max        = 0.5,      # HAC discoverability boundary (vs 0.98 in 01-07)
  b               = 3.0,
  base_seed       = 2026L,
  num_MC          = 200L,
  en_solver       = "TENET_AUG",  # EN solver variant for the M1/M2 methods
  snr_grid        = c(0.1, 0.2, 0.5, 1.0, 2.0, 5.0),
  # Headline WHOLE-truth sweep: crosses the corr_max = 0.5 discoverability
  # boundary (0.2/0.35 undiscoverable, 0.5 partial, 0.65-0.9 discovered).
  rho_grid_whole      = c(0.2, 0.35, 0.5, 0.65, 0.8, 0.9),
  # Cautionary INDIVIDUAL-truth pair: weak vs. strong sibling correlation.
  rho_grid_individual = c(0.35, 0.9)
)


# ==============================================================================
# Block-level metric helpers  (mirror the cpp/Python benchmark)
# ==============================================================================

#' 1-based column indices of block `blk`.
.block_cols <- function(blk, block_size) {
  ((blk - 1L) * block_size + 1L):(blk * block_size)
}

#' Fraction of active blocks with at least one selected variable.
.block_hit_rate <- function(sel, active, block_size) {
  if (length(active) == 0L) return(0)
  hits <- sum(vapply(active, function(blk) {
    any(.block_cols(blk, block_size) %in% sel)
  }, logical(1L)))
  hits / length(active)
}

#' Block-level FDP = (null blocks hit) / max(total blocks hit, 1).
.block_fdp <- function(sel, active, G, block_size) {
  total_hit <- 0L
  null_hit  <- 0L
  for (blk in seq_len(G)) {
    if (any(.block_cols(blk, block_size) %in% sel)) {
      total_hit <- total_hit + 1L
      if (!(blk %in% active)) null_hit <- null_hit + 1L
    }
  }
  null_hit / max(total_hit, 1L)
}

#' Fraction of active blocks fully contained in the selection.
.full_block_rate <- function(sel, active, block_size) {
  if (length(active) == 0L) return(0)
  fully <- sum(vapply(active, function(blk) {
    all(.block_cols(blk, block_size) %in% sel)
  }, logical(1L)))
  fully / length(active)
}

#' Fraction of null blocks with at least one selected variable.
.null_block_activation <- function(sel, active, G, block_size) {
  null_total <- 0L
  null_hit   <- 0L
  for (blk in seq_len(G)) {
    if (blk %in% active) next
    null_total <- null_total + 1L
    if (any(.block_cols(blk, block_size) %in% sel)) null_hit <- null_hit + 1L
  }
  if (null_total == 0L) 0 else null_hit / null_total
}

#' Fraction of true blocks whose members all share one cluster label
#' (1.0 by construction for oracle runs). `groups_vec` is the selector's
#' per-variable cluster assignment (`$groups`, 1-based).
.block_purity_rate <- function(groups_vec, G, block_size) {
  pure <- sum(vapply(seq_len(G), function(k) {
    idx <- .block_cols(k, block_size)
    all(groups_vec[idx] == groups_vec[idx[1L]])
  }, logical(1L)))
  pure / G
}

#' 1.0 if the selected set equals the true support, else 0.0.
.exact_support <- function(sel, true_support) {
  as.numeric(identical(sort(as.integer(sel)), sort(as.integer(true_support))))
}


# ==============================================================================
# One method on one dataset
# ==============================================================================

#' Run one GVS method variant on a dataset and collect its metrics.
#' Mirrors Python _run_block_method: the selector seed is the trial seed and
#' cv_seed stays at its default -1 (derived from the selector seed).
#'
#' @param dat      A dgp_block_equicorr() result.
#' @param gvs_type "EN" or "IEN".
#' @param oracle   TRUE to supply true block labels (groups); FALSE for HAC.
#' @param cfg      Benchmark configuration list.
#' @param seed     Trial seed (passed as the selector seed).
.run_block_method <- function(dat, gvs_type, oracle, cfg, seed) {
  gvs_args <- list(
    gvs_type       = gvs_type,
    corr_max       = cfg$corr_max,
    hc_linkage     = "Single",
    lambda2_method = if (gvs_type == "IEN") "CV_1SE_IEN_CCD" else "CV_1SE_CCD"
  )
  # en_solver applies to EN only (TENET / TENET_AUG). For IEN the en_solver
  # field is not applicable and is left at its library default, mirroring the
  # Python demos' en_solver=None; the IEN solver itself is pinned to
  # TIENET_AUG below (the solver the cpp/Python benchmarks run).
  if (gvs_type == "EN") gvs_args$en_solver <- cfg$en_solver
  if (oracle)           gvs_args$groups    <- dat$prior_groups

  solver <- if (gvs_type == "IEN") "TIENET_AUG" else "TLARS"
  # All fields of the cpp make_gvs_trex_control() explicit (the R AUTO
  # stagnation default would switch the stop off for the EN rows).
  ctrl <- TRexSelector::trex_control(
    solver = solver, K = cfg$K, max_dummy_multiplier = 10,
    use_max_T_stop = TRUE, lloop_strategy = "HCONCAT",
    tloop_stagnation_stop = TRUE, tloop_max_stagnant_steps = 5,
    use_memory_mapping = FALSE
  )
  sel <- TRexSelector::TRexGVSSelector$new(
    dat$X, dat$y, tFDR = cfg$tFDR, seed = seed, verbose = FALSE,
    gvs_control = do.call(TRexSelector::trex_gvs_control, gvs_args),
    control     = ctrl
  )
  sel$select()

  si <- sel$selected_indices
  ts <- dat$true_support
  list(
    coord_fdp     = TRexSelector::trex_fdp(si, ts),
    coord_tpr     = TRexSelector::trex_tpp(si, ts),
    exact_support = .exact_support(si, ts),
    block_hit     = .block_hit_rate(si, dat$active_blocks, dat$block_size),
    block_fdp     = .block_fdp(si, dat$active_blocks, dat$G, dat$block_size),
    full_block    = .full_block_rate(si, dat$active_blocks, dat$block_size),
    null_act      = .null_block_activation(si, dat$active_blocks, dat$G,
                                           dat$block_size),
    purity        = .block_purity_rate(sel$groups, dat$G, dat$block_size),
    lambda2       = sel$lambda2_used,
    T_stop        = as.numeric(sel$T_stop),
    M_found       = as.numeric(sel$max_clusters)
  )
}

#' Run all four method variants (M1-M4) on one dataset.
.run_block_all_methods <- function(dat, cfg, seed) {
  list(
    M1 = .run_block_method(dat, "EN",  FALSE, cfg, seed),
    M2 = .run_block_method(dat, "EN",  TRUE,  cfg, seed),
    M3 = .run_block_method(dat, "IEN", FALSE, cfg, seed),
    M4 = .run_block_method(dat, "IEN", TRUE,  cfg, seed)
  )
}


# ==============================================================================
# One (rho, snr) cell: num_MC trials, aggregated (mean + sd per metric)
# ==============================================================================

.METRIC_NAMES <- c("coord_tpr", "coord_fdp", "exact_support", "block_hit",
                   "block_fdp", "full_block", "null_act", "purity",
                   "lambda2", "T_stop", "M_found")

.run_block_cell <- function(cfg, scenario, rho, snr, cell_base_seed) {
  one_trial <- function(mc) {
    trial_seed <- cell_base_seed + mc - 1L
    dat <- dgp_block_equicorr(
      n = cfg$n, p = cfg$p, G = cfg$G, block_size = cfg$block_size,
      n_active_blocks = cfg$n_active_blocks, rho = rho, snr = snr,
      scenario = scenario, seed = trial_seed, b = cfg$b
    )
    .run_block_all_methods(dat, cfg, seed = trial_seed)
  }

  # A failed trial is dropped, as before (it returns NULL instead of a list).
  safe_trial <- function(mc) tryCatch(one_trial(mc), error = function(e) NULL)

  # One resumable grid cell (cell_cached() of mc_runtime.R), flattened to
  # "M1.mean_coord_tpr" etc.; the trials run on PSOCK workers.
  key <- paste0("block|", scenario, "|", cell_digest(cfg, rho, snr, cell_base_seed))
  flat <- cell_cached(key, function() {
    trials <- mc_map(seq_len(cfg$num_MC), safe_trial, num_cores)
    ok <- vapply(trials, is.list, logical(1L))
    trials <- trials[ok]

    agg_method <- function(mkey) {
      out <- list()
      for (mt in .METRIC_NAMES) {
        vals <- vapply(trials, function(t) as.numeric(t[[mkey]][[mt]]),
                       numeric(1L))
        out[[paste0("mean_", mt)]] <- mean(vals)
        out[[paste0("sd_", mt)]]   <- if (length(vals) > 1L) sd(vals) else 0.0
      }
      out
    }
    list(M1 = agg_method("M1"), M2 = agg_method("M2"),
         M3 = agg_method("M3"), M4 = agg_method("M4"))
  })
  methods <- c("M1", "M2", "M3", "M4")
  setNames(lapply(methods, function(m) {
    part <- flat[startsWith(names(flat), paste0(m, "."))]
    setNames(part, sub("^M[1-4][.]", "", names(part)))
  }), methods)
}


# ==============================================================================
# One scenario
# ==============================================================================

#' Run the full (rho x snr) grid for one truth scenario.
#'
#' `rho_grid` is scenario-specific: WHOLE sweeps the full discoverability
#' range; INDIVIDUAL uses a low/high pair.
run_block_bench_scenario <- function(cfg, scenario, rho_grid) {
  tag <- scenario

  print_section_header(sprintf(
    "Block Bench MC: %s\nn=%d p=%d G=%d block=%d active=%d tFDR=%s K=%d MC=%d corr_max=%s",
    tag, cfg$n, cfg$p, cfg$G, cfg$block_size, cfg$n_active_blocks,
    fmt_num(cfg$tFDR), cfg$K, cfg$num_MC, fmt_num(cfg$corr_max)
  ))

  n_rho <- length(rho_grid)
  n_snr <- length(cfg$snr_grid)

  results <- vector("list", n_rho)
  for (i_rho in seq_len(n_rho)) {
    results[[i_rho]] <- vector("list", n_snr)
    for (i_snr in seq_len(n_snr)) {
      # Seeds staggered by grid-cell index so each cell uses a distinct band.
      cell_index     <- (i_rho - 1L) * n_snr + (i_snr - 1L)
      cell_base_seed <- cfg$base_seed + cell_index * cfg$num_MC
      cat(sprintf("  %s rho=%.2f snr=%.2f  [%d/%d] — running %d MC trials...\n",
                  tag, rho_grid[i_rho], cfg$snr_grid[i_snr],
                  cell_index + 1L, n_rho * n_snr, cfg$num_MC))
      cell <- .run_block_cell(cfg, scenario, rho_grid[i_rho],
                              cfg$snr_grid[i_snr], cell_base_seed)
      results[[i_rho]][[i_snr]] <- cell
      cat(sprintf("  %s rho=%.2f snr=%.2f — done.  M1 TPR=%.3f  M2 TPR=%.3f  M3 TPR=%.3f  M4 TPR=%.3f\n\n",
                  tag, rho_grid[i_rho], cfg$snr_grid[i_snr],
                  cell$M1$mean_coord_tpr, cell$M2$mean_coord_tpr,
                  cell$M3$mean_coord_tpr, cell$M4$mean_coord_tpr))
    }
  }

  print_block_grid_table(tag, scenario, rho_grid, cfg$snr_grid, results, cfg,
                         OUT_DIR, paste0("gvs_block_bench_", tolower(tag)))
}


# ==============================================================================
# Top-level benchmark
# ==============================================================================

#' Run the full block-equicorrelated benchmark.
#'
#' WHOLE is the headline scenario (prior-information value vs. HAC
#' discoverability); INDIVIDUAL is the cautionary appendix (coordinate-level
#' FDR under exchangeable null siblings).
run_block_benchmark <- function(cfg) {
  print_section_header(paste0(
    "Block-equicorrelated benchmark\n",
    "DGP = dgp_block_equicorr\n",
    "methods = {EN clustered, EN oracle, IEN clustered, IEN oracle}\n",
    "scenarios = {WHOLE (headline), INDIVIDUAL (appendix)}"
  ))

  run_block_bench_scenario(cfg, "WHOLE", cfg$rho_grid_whole)
  run_block_bench_scenario(cfg, "INDIVIDUAL", cfg$rho_grid_individual)

  cat("Block benchmark MC simulations complete.\n")
}


# ==============================================================================
# main
# ==============================================================================

cat(sprintf("Running with %d worker cores\n\n", num_cores))

run_block_benchmark(CFG)

# All results are written: the cell cache of a stopped run is no longer needed.
cell_cache_done()
