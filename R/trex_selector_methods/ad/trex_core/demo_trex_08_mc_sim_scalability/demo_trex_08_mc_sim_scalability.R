# ==============================================================================
# demo_trex_08_mc_sim_scalability.R
# ==============================================================================
#
# Scalability benchmark for the T-Rex Selector: runtime and memory usage as the
# problem scales, across three base solvers (TLARS, TOMP, TAFS).
#
# Mirrors:
# cpp/trex_selector_methods/ad/trex_core/demo_trex_08_mc_sim_scalability.cpp
# and
# Python/trex_selector_methods/ad/trex_core/demo_trex_08_mc_sim_scalability.py
#
# Two aims:
#   (a) high-dimensionality  — sweep p up to 1,000,000;
#   (b) volumes beyond RAM   — the data one T-iteration touches (X plus the K
#                              memory-mapped dummy files) grows past physical
#                              RAM, feasible only because the pipeline is fully
#                              out-of-core (memory-mapped).
#
# Design of the study, two blocks with separate outputs and trial caches:
#   * base block     n in {300,1000,2000,5000} x p in {1k,10k,100k,150k},
#                    max_dummy_multiplier = 10 (log-spaced p for the in-core
#                    scaling, p = 150k for the step into out-of-core)
#   * large-p block  n = 300 x p in {100,000; 1,000,000},
#                    max_dummy_multiplier = 5 (p = 100,000 also runs in the
#                    base block at cap 10, a direct cap comparison)
#   * A file named by TREX_STOP_FILE stops the run before the next trial
#     (exit status 75); finished trials stay cached and the next run resumes.
#   * SNR = 1 fixed (SNR is not a scaling axis; sweeping it only burns time),
#     demo-03 DGP (variable support s=10 redrawn per trial, unit coefficients,
#     Normal predictors/noise).
#   * num_MC = 10 SEQUENTIAL trials per (n, p, solver). Trials are NOT run in
#     parallel: this is a timing/memory benchmark, and concurrent trials would
#     (i) corrupt per-trial wall-clock, (ii) confound RSS, and (iii) multiply
#     the on-disk dummy backing. The K random experiments inside select() also
#     run one after another; OpenMP threads (TREX_NUM_WORKERS, default 8)
#     run the column loops inside each (max_inner_threads).
#
# Fully memory-mapped pipeline in every point: X is generated column-chunk by
# column-chunk into an mmap backing file via dgp_chunked_gauss_snr() (never
# materialized in RAM), and use_memory_mapping = TRUE memory-maps the dummy
# matrices D and serializes solver warm-start state between T-loop iterations.
#
# Disk pre-flight: a point whose worst-case disk need (X, K dummy files at the
# dummy cap, K checkpoints) does not fit the free temp-dir space is skipped
# with a message. The library checks free space too (it throws before creating
# memory-mapped files that may not fit), so the pre-flight is an early,
# per-point check that keeps the run going.
#
# Per grid point we record: timing (DGP, select() mean/median/min/max, per-T-iter),
# memory (post-select RSS, physical footprint, process peak RSS, X+y bytes,
# per T-iteration data volume at the realized L), and FDR / TPR / Avg L / Avg T.
#
# Memory metrics in R (see the profiling section):
#   * rss_mib      — `ps -o rss= -p <pid>` (current resident set size).
#   * peak_rss_mib — Linux: exact VmHWM from /proc/self/status. Elsewhere
#                    (macOS): running maximum of the ps-based RSS SAMPLES taken
#                    at the instrumentation points — a lower bound of the true
#                    kernel high-water mark (base R exposes no getrusage()).
#   * fp_mib       — Linux: VmRSS + VmSwap. macOS: NA (the phys_footprint
#                    counter needs mach task_info(), unreachable from base R;
#                    NA rather than a fake number).
#
# Self-cleanup: the process sweeps its own memory-mapped orphans
# (demo08_scalability_* backing files, trex_memmap_* dummy dirs and
# trex_warmstart_* checkpoint dirs in the temp dir) at startup and on exit, so
# a cancelled run does not strand hundreds of GB. (SIGKILL and SIGBUS cannot be
# handled from R; the next startup sweep reclaims them.)
#
# Expandable by design. Every trial is appended to a raw per-trial cache
# (<stem>_trials.csv, one row per (n, p, solver, mc) keyed by its seed). A run
# computes only the trials it is missing and derives the summary table/CSV from
# the cache, so:
#   * num_MC top-up  — run at num_MC=2 now, later num_MC=10 computes just the 8
#     new trials per cell and re-aggregates (no recompute of the first 2);
#   * grid growth    — append new n/p values later; existing cells are reused;
#   * trial-level resume — a crash/OOM mid-point keeps every finished trial.
# Per-trial seeds are a stable function of (n, p, mc) (grid-position
# independent), so reuse and reproducibility survive reordering or extending
# the grid.
#
# Results go to simulation_results/data/ as an aligned table (.txt) and a tidy
# summary CSV per block, both re-derived from the cache and re-saved after each
# (n, p) point.
#
# Run:
#   Rscript demo_trex_08_mc_sim_scalability.R           # both blocks, num_MC=10
#   Rscript demo_trex_08_mc_sim_scalability.R base      # base block only
#   Rscript demo_trex_08_mc_sim_scalability.R large 2   # large-p block, num_MC=2
#   Rscript demo_trex_08_mc_sim_scalability.R smoke     # tiny end-to-end check
#   Rscript demo_trex_08_mc_sim_scalability.R smoke 3   # smoke grid, num_MC=3
#   TREX_SOLVERS=TLARS limits the computed solvers (cached ones still count).
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
source(file.path(this_dir_, "..", "..", "..", "mc_runtime.R"))  # solver_selected()
source(file.path(this_dir_, "..", "..", "..", "support_generators.R"))
source(file.path(this_dir_, "..", "trex_sim_utils.R"))

# ==============================================================================
# Configuration
# ==============================================================================

OUT_DIR <- file.path(this_dir_, "simulation_results", "data")
dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)

S_TRUE      <- 10L     # cardinality of the true support (redrawn per trial)
TFDR        <- 0.1
SNR         <- 1.0     # fixed; SNR is not a scaling axis
CHUNK_COLS  <- 1000L   # column-chunk size for the out-of-core X generation
NUM_RND_EXP <- 20L     # K random experiments
DUMMY_MULT  <- 10L     # base block max_dummy_multiplier
LARGE_P_DUMMY_MULT <- 5L     # large-p block cap (bounds the dummy disk)
DISK_RESERVE_GIB   <- 50.0   # free disk left untouched by the pre-flight

# OpenMP threads inside select() (max_inner_threads), set like the C++/Python
# demos: TREX_NUM_WORKERS, else 8 (0 = all cores).
NUM_THREADS <- local({
  a <- suppressWarnings(as.integer(Sys.getenv("TREX_NUM_WORKERS", "8")))
  if (is.na(a)) 8L else if (a < 1L) parallel::detectCores() else a
})

# Base block: full n x p cross. Points whose volume X + K*L*X passes physical
# RAM are the out-of-core regime that only the memory-mapped pipeline reaches.
N_VALUES <- c(300L, 1000L, 2000L, 5000L)
P_VALUES <- c(1000L, 10000L, 100000L, 150000L)
# Large-p block: few observations, up to a million predictors.
LARGE_P_N_VALUES <- 300L
LARGE_P_P_VALUES <- c(100000L, 1000000L)
EXIT_STOPPED <- 75L  # exit status after a stop via TREX_STOP_FILE (EX_TEMPFAIL)
NUM_MC   <- 10L

# Tiny end-to-end smoke grid (pipeline verification only); the second point
# exercises the large-p dummy cap.
SMOKE_GRID <- list(c(150L, 500L, DUMMY_MULT), c(200L, 1000L, LARGE_P_DUMMY_MULT))
SMOKE_MC   <- 2L

# Base solvers to compare (rho_afs only used by TAFS, which requires rho in
# (0, 1]; mirrors the C++ DemoSolverInfo list — TOMP default, TAFS rho_afs=0.3).
SOLVERS <- list(
  list(solver = "TLARS", name = "TLARS", lambda2 = 0.0, rho_afs = 0.0),
  list(solver = "TOMP",  name = "TOMP",  lambda2 = 0.0, rho_afs = 0.0),
  list(solver = "TAFS",  name = "TAFS",  lambda2 = 0.0, rho_afs = 0.3)
)

# Temp dir used for the X backing files and searched for mmap orphans.
# Matches Python tempfile.gettempdir() / C++ std::filesystem::temp_directory_path().
TMP_DIR <- {
  td <- Sys.getenv("TMPDIR", unset = "")
  if (!nzchar(td)) td <- "/tmp"
  normalizePath(td, mustWork = FALSE)
}


.volume_gib <- function(n, p, L) {
  # Data volume one T-iteration touches in GiB: X plus the K memory-mapped
  # dummy files of L * p columns each (float64).
  as.numeric(n) * as.numeric(p) * 8.0 * (1 + NUM_RND_EXP * L) / 1024^3
}


.worst_case_disk_gib <- function(n, p, dummy_mult) {
  # Worst-case disk need of one trial in GiB: X, the K dummy files at
  # L = dummy_mult, and K solver checkpoints (at most 32 bytes per column of
  # [X|D]).
  np_ <- as.numeric(n) * as.numeric(p)
  x_bytes     <- np_ * 8.0
  dummies     <- NUM_RND_EXP * np_ * dummy_mult * 8.0
  checkpoints <- NUM_RND_EXP * 32.0 * as.numeric(p) * (1 + dummy_mult)
  (x_bytes + dummies + checkpoints) / 1024^3
}


# Stop flag: TRUE once the file named by TREX_STOP_FILE exists (sticky).
.stop_state <- new.env()
.stop_state$stopped <- FALSE

.stop_requested <- function() {
  # Checked before every trial, so a stop never cuts a trial short.
  if (!.stop_state$stopped) {
    path <- Sys.getenv("TREX_STOP_FILE", "")
    .stop_state$stopped <- nzchar(path) && file.exists(path)
  }
  .stop_state$stopped
}

.free_temp_disk_gib <- function() {
  # Free space of the temp dir in GiB via POSIX `df -Pk` (NA if unavailable).
  out <- tryCatch(suppressWarnings(system2("df", c("-Pk", TMP_DIR), stdout = TRUE)),
                  error = function(e) character(0))
  if (length(out) < 2L) return(NA_real_)
  fields <- strsplit(trimws(out[[2L]]), "\\s+")[[1L]]
  kib <- suppressWarnings(as.numeric(fields[4L]))
  if (is.finite(kib)) kib / 1024^2 else NA_real_
}


.physical_ram_gib <- function() {
  # Physical RAM in GiB (NA if unavailable) — used to flag out-of-core points.
  if (identical(Sys.info()[["sysname"]], "Darwin")) {
    out <- tryCatch(system2("sysctl", c("-n", "hw.memsize"), stdout = TRUE),
                    error = function(e) NA_character_)
    val <- suppressWarnings(as.numeric(out[[1]]))
    if (is.finite(val)) return(val / 1024^3)
  } else if (file.exists("/proc/meminfo")) {
    line <- grep("^MemTotal:", readLines("/proc/meminfo"), value = TRUE)
    val  <- suppressWarnings(as.numeric(sub("^MemTotal:\\s+(\\d+).*$", "\\1",
                                            line[[1]])))
    if (is.finite(val)) return(val / 1024^2)   # kB -> GiB
  }
  NA_real_
}

RAM_GIB <- .physical_ram_gib()


.make_scalability_ctrl <- function(solver_entry, dummy_mult) {
  # T-Rex control (base config of demos 02/03/07) for one grid point.
  # EVERY control option is set explicitly and identically to the C++ demo's
  # make_scalability_trex_ctrl() — no reliance on trex_control() defaults.
  TRexSelector::trex_control(
    solver                   = solver_entry$solver,
    lambda2                  = solver_entry$lambda2,
    rho_afs                  = solver_entry$rho_afs,
    K                        = NUM_RND_EXP,
    max_dummy_multiplier     = dummy_mult,
    use_max_T_stop           = TRUE,
    opt_threshold            = 0.75,
    dummy_distribution       = trex_dummy_distribution("Normal"),
    lloop_strategy           = "STANDARD",
    tloop_stagnation_stop    = TRUE,
    tloop_max_stagnant_steps = 5L,
    parallel_rnd_experiments = FALSE,          # sequential random experiments
    max_inner_threads        = NUM_THREADS,    # column loops inside each experiment
    use_memory_mapping       = TRUE            # X mmap + D mmap + solver serialization
  )
}


.benchmark_grid <- function(num_MC = NUM_MC) {
  # List of (n, p, num_MC, dummy_mult) points of the base block (n x p cross).
  grid <- list()
  for (n in N_VALUES) {
    for (p in P_VALUES) {
      grid[[length(grid) + 1L]] <- list(n = n, p = p, num_MC = num_MC,
                                        dummy_mult = DUMMY_MULT)
    }
  }
  grid
}


.large_p_grid <- function(num_MC = NUM_MC) {
  # Points of the large-p block, ordered by volume (p-outer) so the cheapest
  # points finish first.
  grid <- list()
  for (p in LARGE_P_P_VALUES) {
    for (n in LARGE_P_N_VALUES) {
      grid[[length(grid) + 1L]] <- list(n = n, p = p, num_MC = num_MC,
                                        dummy_mult = LARGE_P_DUMMY_MULT)
    }
  }
  grid
}


.smoke_grid <- function(num_MC = SMOKE_MC) {
  lapply(SMOKE_GRID, function(np) {
    list(n = np[[1L]], p = np[[2L]], num_MC = num_MC, dummy_mult = np[[3L]])
  })
}


# ==============================================================================
# Expandable per-trial cache (top-up num_MC / grid; trial-level resume)
# ==============================================================================

# Deterministic, grid-position-independent, 32-bit-safe seed. Distinct (n, p)
# already yield different data (the X draw has a different shape), so we only
# need determinism in (n, p, mc), not cross-cell seed uniqueness. Matches the
# C++/Python _trial_seed so the languages seed identically.
SEED_MOD <- 2147483647   # 2^31 - 1

.trial_seed <- function(n, p, mc) {
  as.integer((24 + as.numeric(n) * 100003 + as.numeric(p) * 13 + mc) %% SEED_MOD)
}

# Raw per-trial store: one row per (n, p, solver, mc). The summary .txt/.csv
# are derived from it, so a later run tops up missing trials without
# recomputation. mc is 0-based (mirrors the Python cache format).
RAW_FIELDS <- c("n", "p", "solver", "mc", "trial_seed", "dgp_s", "select_s",
                "rss_mib", "fp_mib", "peak_rss_mib", "data_mib",
                "fdp", "tpp", "L", "T")

.raw_cache_path <- function(file_stem, persist_cache = TRUE) {
  # A non-persistent cache lives in the temp dir and is swept with the
  # orphans, so every smoke run computes fresh.
  if (persist_cache) return(file.path(OUT_DIR, paste0(file_stem, "_trials.csv")))
  file.path(TMP_DIR, paste0("demo08_scalability_", file_stem, "_trials.csv"))
}

.cell_key <- function(n, p, solver_name) sprintf("%d|%d|%s", n, p, solver_name)

.load_trial_cache <- function(raw_path) {
  # Load the raw per-trial store into cache[[key]][[as.character(mc)]] = row.
  cache <- list()
  if (!file.exists(raw_path)) return(cache)
  df <- utils::read.csv(raw_path, stringsAsFactors = FALSE)
  for (i in seq_len(nrow(df))) {
    key <- .cell_key(df$n[[i]], df$p[[i]], df$solver[[i]])
    if (is.null(cache[[key]])) cache[[key]] <- list()
    cache[[key]][[as.character(df$mc[[i]])]] <- as.list(df[i, RAW_FIELDS])
  }
  cache
}

.append_trial_row <- function(raw_path, row) {
  # Append one trial to the raw store (write header if the file is new).
  df     <- as.data.frame(row, stringsAsFactors = FALSE)[, RAW_FIELDS]
  is_new <- !file.exists(raw_path)
  utils::write.table(df, raw_path, sep = ",", row.names = FALSE,
                     col.names = is_new, append = !is_new, quote = FALSE)
}

.aggregate_cell <- function(cell_cache, n, p) {
  # Aggregate all cached trials of one (n, p, solver) cell into a summary list.
  # Aggregates over every cached trial (monotone: more trials -> a better
  # estimate), so re-running at a lower num_MC never discards work.
  mcs  <- sort(as.integer(names(cell_cache)))
  rows <- lapply(as.character(mcs), function(k) cell_cache[[k]])
  num  <- function(field) vapply(rows, function(r) as.numeric(r[[field]]), 0.0)

  fdp   <- num("fdp")
  tpp   <- num("tpp")
  L     <- num("L")
  T_    <- num("T")
  dgp   <- num("dgp_s")
  sel   <- num("select_s")
  titer <- sel / pmax(T_, 1.0)
  total <- dgp + sel

  list(
    vol_gib      = .volume_gib(n, p, max(mean(L), 1.0)),
    sel_mean_s   = mean(sel),
    sel_med_s    = stats::median(sel),
    sel_min_s    = min(sel),
    sel_max_s    = max(sel),
    avg_dgp_s    = mean(dgp),
    avg_titer_s  = mean(titer),
    avg_total_s  = mean(total),
    avg_select_s = mean(sel),
    peak_rss_mib = max(num("peak_rss_mib")),
    max_rss_mib  = max(num("rss_mib")),
    max_fp_mib   = if (all(is.na(num("fp_mib")))) NA_real_ else max(num("fp_mib"), na.rm = TRUE),
    data_mib     = as.numeric(rows[[length(rows)]][["data_mib"]]),
    avg_fdr      = mean(fdp),
    avg_tpr      = mean(tpp),
    avg_L        = mean(L),
    avg_T        = mean(T_),
    num_mc       = length(rows)
  )
}


# ==============================================================================
# Self-cleanup of memory-mapped orphans
# ==============================================================================

ORPHAN_GLOBS <- c("demo08_scalability_*", "trex_memmap_*", "trex_warmstart_*")

sweep_mmap_orphans <- function(reason = "") {
  # Remove this demo's stray mmap backing in the temp dir (best-effort).
  # Targets `demo08_scalability_*` X backing files, `trex_memmap_*` dummy dirs
  # and `trex_warmstart_*` checkpoint dirs left behind by a killed/cancelled
  # run. Assumes SOLO execution (this is a sequential benchmark). Returns bytes
  # freed, invisibly.
  freed <- 0
  for (pattern in ORPHAN_GLOBS) {
    for (path in Sys.glob(file.path(TMP_DIR, pattern))) {
      info <- file.info(path)
      if (isTRUE(info$isdir)) {
        files <- list.files(path, recursive = TRUE, full.names = TRUE)
        freed <- freed + sum(file.size(files), na.rm = TRUE)
        unlink(path, recursive = TRUE, force = TRUE)
      } else {
        if (is.finite(info$size)) freed <- freed + info$size
        unlink(path)
      }
    }
  }
  if (freed > 0) {
    cat(sprintf("[cleanup] swept %.1f GiB of mmap orphans%s\n",
                freed / 1024^3,
                if (nzchar(reason)) sprintf(" (%s)", reason) else ""))
  }
  invisible(freed)
}

# Sweep on normal interpreter exit (best-effort; SIGKILL cannot be trapped —
# the next startup sweep reclaims whatever is left).
.cleanup_env_ <- new.env(parent = emptyenv())
invisible(reg.finalizer(.cleanup_env_, function(e) sweep_mmap_orphans("exit"),
                        onexit = TRUE))


# ==============================================================================
# Memory profiling hooks (ps / /proc based — see file header for caveats)
# ==============================================================================

.rss_state_ <- new.env(parent = emptyenv())
.rss_state_$peak_mib <- 0.0

current_rss_mib <- function() {
  # Current resident set size (RSS) of this process in MiB (0.0 if unavailable).
  # Every call also feeds the sampled peak (see sampled_peak_rss_mib()).
  mib <- 0.0
  if (file.exists("/proc/self/statm")) {
    parts <- strsplit(readLines("/proc/self/statm", n = 1L), " ")[[1]]
    pages <- suppressWarnings(as.numeric(parts[[2]]))
    if (is.finite(pages)) mib <- pages * 4096 / 1024^2
  } else {
    out <- tryCatch(
      system2("ps", c("-o", "rss=", "-p", Sys.getpid()), stdout = TRUE),
      error = function(e) NA_character_
    )
    kib <- suppressWarnings(as.numeric(trimws(out[[1]])))
    if (is.finite(kib)) mib <- kib / 1024.0   # ps reports KiB
  }
  .rss_state_$peak_mib <- max(.rss_state_$peak_mib, mib)
  mib
}

peak_rss_mib <- function() {
  # Peak RSS of this process in MiB.
  # Linux: exact VmHWM (kernel high-water mark) from /proc/self/status.
  # Elsewhere (macOS): running max of the ps-based samples taken at the
  # instrumentation points — a LOWER BOUND of the true peak (documented
  # divergence; base R exposes no getrusage()).
  if (file.exists("/proc/self/status")) {
    line <- grep("^VmHWM:", readLines("/proc/self/status"), value = TRUE)
    if (length(line) > 0L) {
      kib <- suppressWarnings(as.numeric(sub("^VmHWM:\\s+(\\d+).*$", "\\1",
                                             line[[1]])))
      if (is.finite(kib)) return(kib / 1024.0)
    }
  }
  current_rss_mib()          # refresh the sample before reporting
  .rss_state_$peak_mib
}

current_footprint_mib <- function() {
  # Physical footprint of this process in MiB.
  # Linux approximation: VmRSS + VmSwap. macOS: NA — the phys_footprint
  # counter (Activity Monitor's "Memory") needs mach task_info(), which base R
  # cannot reach; NA is recorded rather than a fake number.
  if (file.exists("/proc/self/status")) {
    lines <- readLines("/proc/self/status")
    hits  <- grep("^(VmRSS|VmSwap):", lines, value = TRUE)
    kib   <- suppressWarnings(sum(as.numeric(sub("^\\S+\\s+(\\d+).*$", "\\1",
                                                 hits))))
    if (is.finite(kib)) return(kib / 1024.0)
  }
  NA_real_
}


# ==============================================================================
# Single-trial runner (sequential; timing + memory instrumented)
# ==============================================================================

run_scalability_trial <- function(n, p, ctrl, tFDR, snr, seed, support, coefs,
                                  mmap_filestem) {
  # One MC trial on the fully memory-mapped pipeline. The X backing file is
  # removed via on.exit() (exception-safe; mirrors the C++ RAII MmapFileGuard).
  # Returns a list of per-trial measurements.
  m <- list()
  x_path <- paste0(mmap_filestem, "_X.dat")
  on.exit(unlink(x_path), add = TRUE)

  current_rss_mib()   # baseline sample for the sampled-peak tracker

  t0 <- proc.time()[["elapsed"]]
  # Chunked out-of-core DGP: X goes straight to the mmap backing file,
  # working memory is O(n * CHUNK_COLS).
  dat <- dgp_chunked_gauss_snr(n, p, support, x_path,
                               chunk_cols = CHUNK_COLS, coefs = coefs,
                               snr = snr, seed = seed)
  m$dgp_s <- proc.time()[["elapsed"]] - t0

  m$data_mib <- (file.size(x_path) + length(dat$y) * 8) / 1024^2
  current_rss_mib()   # post-DGP sample

  t1 <- proc.time()[["elapsed"]]
  sel <- TRexSelector$new(
    X       = dat$X,
    y       = dat$y,
    tFDR    = tFDR,
    seed    = -1L,     # hardware entropy per trial, as in all MC demos
    verbose = FALSE,
    control = ctrl
  )
  sel$select()
  m$select_s <- proc.time()[["elapsed"]] - t1

  m$rss_mib      <- current_rss_mib()
  m$fp_mib       <- current_footprint_mib()
  m$peak_rss_mib <- peak_rss_mib()   # process-lifetime max up to this trial

  m$fdp <- TRexSelector::trex_fdp(sel$selected_indices, support)
  m$tpp <- TRexSelector::trex_tpp(sel$selected_indices, support)
  m$L   <- as.numeric(sel$L)
  m$T   <- as.numeric(sel$T_stop)

  # Drop the selector and mmap view before the backing file is removed.
  rm(sel, dat)
  m
}


# ==============================================================================
# Grid-point loop (n, p, solver): sequential MC trials
# ==============================================================================

run_scalability_grid_point <- function(n, p, num_MC, dummy_mult, solver_entry,
                                       tFDR, snr, rnd_coef,
                                       cardinality_true_support,
                                       cell_cache, raw_path, disk_ok = TRUE) {
  # Ensure >= num_MC trials exist for this (n, p, solver) cell, then aggregate.
  # Only the missing trial indices [0, num_MC) are computed; each is appended
  # to the raw store immediately (trial-level durability). With
  # disk_ok = FALSE (no disk, or solver not in TREX_SOLVERS) nothing is
  # computed and only cached trials count; agg is NULL when the cell has no
  # trials at all.
  ctrl <- .make_scalability_ctrl(solver_entry, dummy_mult)
  mmap_filestem <- file.path(TMP_DIR, sprintf("demo08_scalability_n%d_p%d", n, p))

  have   <- length(cell_cache)
  todo   <- if (disk_ok) setdiff(0L:(num_MC - 1L), as.integer(names(cell_cache))) else integer(0)
  reuse_note <- if (have > 0L) sprintf("  (reusing %d cached)", have) else ""
  cat(sprintf(
    "  [%s] n=%d, p=%d  (L cap %d) — %d new / %d MC trials (mmap)%s ...\n",
    solver_entry$name, n, p, dummy_mult, length(todo), num_MC, reuse_note
  ))

  for (mc in todo) {
    if (.stop_requested()) {
      cat("  [stop] TREX_STOP_FILE found, no new trial started\n")
      break
    }
    trial_seed <- .trial_seed(n, p, mc)

    # Variable support (demo 03 pattern; make_support_random adds the
    # +500000 offset internally, mirroring C++ draw_support).
    support <- make_support_random(cardinality_true_support, p,
                                   seed = trial_seed)
    coefs <- if (rnd_coef) {
      set.seed(trial_seed + 600000L)     # isolated coefficient seed stream
      stats::rnorm(cardinality_true_support)
    } else {
      rep(1.0, cardinality_true_support)
    }

    m <- run_scalability_trial(n, p, ctrl, tFDR, snr, trial_seed,
                               support, coefs, mmap_filestem)

    row <- list(n = n, p = p, solver = solver_entry$name, mc = mc,
                trial_seed   = trial_seed,
                dgp_s        = round(m$dgp_s, 6L),
                select_s     = round(m$select_s, 6L),
                rss_mib      = round(m$rss_mib, 3L),
                fp_mib       = round(m$fp_mib, 3L),
                peak_rss_mib = round(m$peak_rss_mib, 3L),
                data_mib     = round(m$data_mib, 3L),
                fdp = round(m$fdp, 6L), tpp = round(m$tpp, 6L),
                L = round(m$L, 3L), T = round(m$T, 3L))
    .append_trial_row(raw_path, row)          # durable before we continue
    cell_cache[[as.character(mc)]] <- row

    cat(sprintf(
      "    trial %d/%d: dgp=%.2fs  select=%.2fs  FDP=%.3f  TPP=%.3f  L=%.0f  T=%.0f  RSS=%.1f MiB  FP=%.1f MiB\n",
      mc + 1L, num_MC, m$dgp_s, m$select_s, m$fdp, m$tpp, m$L, m$T,
      m$rss_mib, m$fp_mib
    ))
  }

  if (length(cell_cache) == 0L) return(list(agg = NULL, cell_cache = cell_cache))
  agg <- .aggregate_cell(cell_cache, n, p)
  cat(sprintf(
    "  [%s] n=%d, p=%d done (%d trials). select mean=%.2fs [%.2f, %.2f]  peakRSS=%.0f MiB  FDR=%.3f  TPR=%.3f\n\n",
    solver_entry$name, n, p, agg$num_mc,
    agg$sel_mean_s, agg$sel_min_s, agg$sel_max_s,
    agg$peak_rss_mib, agg$avg_fdr, agg$avg_tpr
  ))
  list(agg = agg, cell_cache = cell_cache)
}


# ==============================================================================
# Output: aligned table (console + .txt) and tidy CSV
# ==============================================================================

# Per-grid-point metric columns (table order and CSV metric set):
# label, decimals, aggregate key.
METRICS <- list(
  list("VolGiB",      1L, "vol_gib"),
  list("SelMean_s",   3L, "sel_mean_s"),
  list("SelMed_s",    3L, "sel_med_s"),
  list("SelMin_s",    3L, "sel_min_s"),
  list("SelMax_s",    3L, "sel_max_s"),
  list("DGP_s",       3L, "avg_dgp_s"),
  list("s/Titer",     4L, "avg_titer_s"),
  list("PeakRSS_MiB", 1L, "peak_rss_mib"),
  list("RSS_MiB",     1L, "max_rss_mib"),
  list("Foot_MiB",    1L, "max_fp_mib"),
  list("X+y_MiB",     1L, "data_mib"),
  list("FDR",         4L, "avg_fdr"),
  list("TPR",         4L, "avg_tpr"),
  list("Avg_L",       2L, "avg_L"),
  list("Avg_T",       2L, "avg_T")
)

save_and_print_scalability_results <- function(file_stem, block_title,  # nolint: object_length_linter. C++ name
                                               tFDR, snr, grid,
                                               solvers, results,
                                               num_MC_target) {
  # Print the scalability results as an aligned per-solver table (rows = (n, p)
  # grid points, columns = metrics) to console + .txt, and write a tidy
  # long-format CSV. `results[[solver_name]][[key]]` holds the agg lists.
  # is_barrier = 1 where the volume at the realized L exceeds RAM.
  # Mirrors the C++/Python function of the same name.
  txt_path <- file.path(OUT_DIR, paste0(file_stem, ".txt"))
  con      <- file(txt_path, open = "wt")
  on.exit(close(con), add = TRUE)
  out      <- .make_dual_out(con)

  fmt_val <- function(v, prec, width) {
    # (R's sprintf allows at most one '*' per conversion — build the format.)
    if (is.na(v)) sprintf("%*s", width, "NA") else sprintf(sprintf("%%%d.%df", width, prec), v)
  }

  out(sprintf("\n%s\n", strrep("=", 70L)))
  out(sprintf("=== T-Rex Scalability Results (tFDR = %.2f, SNR = %.1f, num_MC = %d) ===\n",
              tFDR, snr, num_MC_target))
  out(sprintf("%s\n\n", strrep("=", 70L)))
  out(paste0(block_title, "\n"))
  out("Fully memory-mapped pipeline (X mmap + D mmap + solver serialization).\n")
  out(sprintf(paste0("Physical RAM: %.0f GiB. VolGiB = X + K*L*X, the data one ",
                     "T-iteration touches (K = %d, L = Avg_L).\n\n"),
              RAM_GIB, NUM_RND_EXP))

  n_w   <- 8L
  p_w   <- 9L
  met_w <- 13L
  for (solver_entry in solvers) {
    grid_res <- results[[solver_entry$name]]
    out(paste0(solver_entry$name, "\n"))
    hdr <- sprintf("%-*s%-*s", n_w, "n", p_w, "p")
    for (mspec in METRICS) hdr <- paste0(hdr, sprintf("%*s", met_w, mspec[[1L]]))
    out(paste0(hdr, "\n", strrep("-", nchar(hdr)), "\n"))
    for (gp in grid) {
      agg <- grid_res[[.cell_key(gp$n, gp$p, solver_entry$name)]]
      if (is.null(agg)) next
      lbl_p <- paste0(gp$p, if (isTRUE(agg$vol_gib > RAM_GIB)) "*" else "")
      row   <- sprintf("%-*d%-*s", n_w, gp$n, p_w, lbl_p)
      for (mspec in METRICS) {
        row <- paste0(row, fmt_val(agg[[mspec[[3L]]]], mspec[[2L]], met_w))
      }
      out(paste0(row, "\n"))
    }
    out("\n")
  }
  out("(* = out-of-core point, VolGiB > physical RAM)\n\n")

  close(con)
  on.exit(NULL)

  # Tidy long-format CSV: n, p, is_barrier, solver, num_mc, snr, metric, value
  # (num_mc is the actual cached trial count per cell, not the run target).
  rows <- list()
  for (solver_entry in solvers) {
    grid_res <- results[[solver_entry$name]]
    for (gp in grid) {
      agg <- grid_res[[.cell_key(gp$n, gp$p, solver_entry$name)]]
      if (is.null(agg)) next
      for (mspec in METRICS) {
        rows[[length(rows) + 1L]] <- data.frame(
          n = gp$n, p = gp$p, is_barrier = as.integer(isTRUE(agg$vol_gib > RAM_GIB)),
          solver = solver_entry$name, num_mc = agg$num_mc,
          snr = sprintf("%.6f", snr), metric = mspec[[1L]],
          value = round(agg[[mspec[[3L]]]], 6L),
          stringsAsFactors = FALSE
        )
      }
    }
  }
  csv_path <- file.path(OUT_DIR, paste0(file_stem, ".csv"))
  utils::write.csv(do.call(rbind, rows), csv_path,
                   row.names = FALSE, quote = FALSE)
  cat(sprintf("[Info] CSV results saved to:          %s\n", csv_path))
  cat(sprintf("[Info] Table results saved to:        %s\n\n", txt_path))
}


# ==============================================================================
# Scalability benchmark driver
# ==============================================================================

demo_trex_scalability <- function(grid, file_stem, block_title, num_MC_target,
                                  rnd_coef = FALSE, persist_cache = TRUE) {
  # Clear any orphans stranded by a previous cancelled run before we start.
  sweep_mmap_orphans("startup")

  raw_path <- .raw_cache_path(file_stem, persist_cache)
  cache    <- .load_trial_cache(raw_path)   # cache[[key]][[mc]] = row
  cached_trials <- sum(vapply(cache, length, 0L))

  cat("\n", strrep("=", 70L), "\n", sep = "")
  cat("Demo: T-Rex Selector Scalability Benchmark  |  TLARS / TOMP / TAFS\n")
  cat(strrep("=", 70L), "\n\n", sep = "")

  vols     <- vapply(grid, function(gp) .volume_gib(gp$n, gp$p, 1.0), 0.0)
  max_disk <- max(vapply(grid, function(gp) {
    .worst_case_disk_gib(gp$n, gp$p, gp$dummy_mult)
  }, 0.0))
  n_over   <- sum(vols > RAM_GIB, na.rm = TRUE)
  cat(block_title, "\n", sep = "")
  cat("Fully memory-mapped pipeline (X mmap + D mmap + solver serialization)\n")
  cat(sprintf("Physical RAM: %.0f GiB,  free temp disk: %.0f GiB\n",
              RAM_GIB, .free_temp_disk_gib()))
  cat(sprintf(paste0("Grid: %d points; largest volume X + K*X (L = 1) = %.1f GiB; ",
                     "%d point(s) exceed RAM at L = 1 (out-of-core); ",
                     "largest worst-case disk need = %.1f GiB\n"),
              length(grid), max(vols), n_over, max_disk))
  cat(sprintf("SNR = %.1f (fixed),  s = %d,  tFDR = %.2f,  num_MC = %d sequential per (n, p, solver)\n",
              SNR, S_TRUE, TFDR, num_MC_target))
  cat(sprintf("Trial cache: %s  (%d trial(s) already present — only missing trials are computed)\n\n",
              raw_path, cached_trials))

  # results[[solver_name]][[cell_key]] = agg
  results <- setNames(vector("list", length(SOLVERS)),
                      vapply(SOLVERS, `[[`, "", "name"))
  for (nm in names(results)) results[[nm]] <- list()

  # (n, p)-outer loop; every solver finishes the current point before the
  # next (larger) one starts, and results are re-saved after each point so a
  # crash on a bigger point preserves the finished ones.
  for (gi in seq_along(grid)) {
    gp   <- grid[[gi]]
    vol1 <- .volume_gib(gp$n, gp$p, 1.0)
    cat(strrep("=", 50L), "\n", sep = "")
    cat(sprintf("Grid point: n=%d, p=%d%s\n", gp$n, gp$p,
                if (isTRUE(vol1 > RAM_GIB)) "   [out-of-core]" else ""))
    cat(strrep("=", 50L), "\n", sep = "")

    # Disk pre-flight: skip a point early when its worst case cannot fit.
    need    <- .worst_case_disk_gib(gp$n, gp$p, gp$dummy_mult)
    avail   <- .free_temp_disk_gib()
    disk_ok <- is.na(avail) || need + DISK_RESERVE_GIB <= avail
    cat(sprintf("  volume X + K*X (L = 1) = %.1f GiB  (X backing %.2f GiB, K = %d dummy files)\n",
                vol1, gp$n * as.numeric(gp$p) * 8.0 / 1024^3, NUM_RND_EXP))
    cat(sprintf("  worst-case disk need (L = %d) = %.1f GiB,  free = %.1f GiB,  reserve = %.1f GiB\n",
                gp$dummy_mult, need, avail, DISK_RESERVE_GIB))
    if (!disk_ok) {
      cat("  [skip] not enough free disk — new trials are not computed for this point\n")
    }
    cat("\n")

    # TREX_SOLVERS limits the computed solvers; the others keep their cache.
    for (solver_entry in SOLVERS) {
      key        <- .cell_key(gp$n, gp$p, solver_entry$name)
      cell_cache <- if (is.null(cache[[key]])) list() else cache[[key]]
      compute    <- disk_ok && solver_selected(solver_entry$name)
      point <- run_scalability_grid_point(
        gp$n, gp$p, gp$num_MC, gp$dummy_mult, solver_entry, TFDR, SNR,
        rnd_coef, S_TRUE, cell_cache, raw_path, compute
      )
      cache[[key]] <- point$cell_cache
      if (!is.null(point$agg)) results[[solver_entry$name]][[key]] <- point$agg
    }

    # Incremental save over completed points (derived from the trial cache).
    save_and_print_scalability_results(file_stem, block_title, TFDR, SNR,
                                       grid[seq_len(gi)], SOLVERS, results,
                                       num_MC_target)
    if (.stop_requested()) {
      cat(sprintf(paste0("\n[stop] Stopped on request after point n=%d, ",
                         "p=%d; finished trials are cached, the next run ",
                         "resumes.\n"), gp$n, gp$p))
      return(invisible(NULL))
    }
  }
}


# ==============================================================================
# Main
# ==============================================================================

# Fixed thread count (mirrors the C++/Python demos): the library's default
# budget here, max_inner_threads in the control. Experiments and trials run
# one after another.
TRexSelector::trex_set_num_threads(NUM_THREADS)
cat(sprintf("Running with %d thread(s) (pinned to match the C++ demo)\n\n",
            TRexSelector::trex_get_num_threads()))

# CLI:  [smoke | base | large] [num_MC]  (order-independent; num_MC is any
# integer token; without a block token both blocks run, base first)
cli_args_  <- commandArgs(trailingOnly = TRUE)
smoke_     <- "smoke" %in% cli_args_
run_base_  <- !("large" %in% cli_args_)
run_large_ <- !("base" %in% cli_args_)
mc_tokens_ <- suppressWarnings(
  as.integer(cli_args_[grepl("^\\+?[0-9]+$", cli_args_)])
)
num_MC_cli_ <- if (length(mc_tokens_) > 0L) mc_tokens_[[1L]] else NA_integer_

if (smoke_) {
  mc_ <- if (!is.na(num_MC_cli_)) num_MC_cli_ else SMOKE_MC
  demo_trex_scalability(.smoke_grid(mc_),
                        "demo_trex_08_scalability_smoke",
                        "Smoke grid (L cap 10 and 5)", mc_,
                        persist_cache = FALSE)
} else {
  mc_ <- if (!is.na(num_MC_cli_)) num_MC_cli_ else NUM_MC
  if (run_base_) {
    demo_trex_scalability(
      .benchmark_grid(mc_), "demo_trex_08_scalability_results",
      sprintf("Base block: n x p cross, max_dummy_multiplier = %d", DUMMY_MULT),
      mc_
    )
  }
  if (run_large_ && !.stop_requested()) {
    demo_trex_scalability(
      .large_p_grid(mc_), "demo_trex_08_scalability_large_p",
      sprintf("Large-p block: n = 300, p up to 1e6, max_dummy_multiplier = %d",
              LARGE_P_DUMMY_MULT),
      mc_
    )
  }
}

sweep_mmap_orphans("shutdown")
if (.stop_requested()) quit(save = "no", status = EXIT_STOPPED)
cat("\ndemo_trex_08_mc_sim_scalability.R complete.\n")
