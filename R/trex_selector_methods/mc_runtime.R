# ==============================================================================
# mc_runtime.R
# ==============================================================================
#
# Monte Carlo runtime shared by the R T-Rex demos:
#
#   mc_map()           — runs a function over tasks on a persistent PSOCK
#                        cluster (spawned R processes, no fork), results in
#                        task order. Every worker loads the package and runs
#                        its solvers single-threaded (core budget: the
#                        parallelism lives in the workers).
#   mc_stop_cluster()  — stops the persistent cluster (also done at exit).
#   cell_cached()      — resumable grid cells: every finished cell (method x
#   cell_digest()        grid point) is appended to a cache file, and a
#   cell_cache_done()    restarted demo reads it back instead of recomputing.
#   solver_selected()  — optional solver selection TREX_SOLVERS (mirrors
#   solver_file_suffix() cpp/trex_selector_methods/solver_filter.hpp): a
#                        selection run computes only the named solvers and
#                        writes its tables to files with a suffix.
#
# The cluster is created on the first mc_map() call and receives the globals
# of the demo script at that moment, so the demo sources its helpers and
# defines its functions before the first Monte Carlo loop runs.
#
# The cache is <demo>/simulation_results/cache/cells.tsv (gitignored), the
# same format as in the C++ and Python demos. Its first line holds the
# fingerprint: a hash of the demo script and the R files of its suite and
# family (plotting excluded), the run tag TREX_RUN_TAG (set by the overnight
# runners to the library pin) and the other TREX_* variables that can change
# results (not the worker count, TREX_FRESH, TREX_SOLVERS or a *_PARTS
# selection). A cache with another fingerprint, or any cache under
# TREX_FRESH=1, is set aside as cells.tsv.stale. Keys combine a readable label
# with cell_digest() of the cell's inputs, so two cells with different inputs
# never share a result. A demo that finishes calls cell_cache_done(), which
# removes the cache.
# ==============================================================================

.mc_rt <- new.env(parent = emptyenv())


# ==============================================================================
# Persistent PSOCK cluster
# ==============================================================================

#' Persistent cluster with `workers` processes, created on first use.
.mc_cluster <- function(workers) {
  if (!is.null(.mc_rt$cl) && identical(.mc_rt$workers, workers)) return(.mc_rt$cl)
  mc_stop_cluster()
  cl <- parallel::makePSOCKcluster(workers)
  globals <- setdiff(ls(globalenv(), all.names = TRUE), ".mc_rt")
  parallel::clusterExport(cl, globals, envir = globalenv())
  parallel::clusterEvalQ(cl, {
    suppressMessages(library(TRexSelector))
    TRexSelector::trex_set_num_threads(1L)
  })
  .mc_rt$cl <- cl
  .mc_rt$workers <- workers
  if (is.null(.mc_rt$finalizer)) {
    reg.finalizer(.mc_rt, function(e) mc_stop_cluster(), onexit = TRUE)
    .mc_rt$finalizer <- TRUE
  }
  cl
}

#' Stop the persistent cluster (no-op when none is running).
mc_stop_cluster <- function() {
  if (!is.null(.mc_rt$cl)) {
    try(parallel::stopCluster(.mc_rt$cl), silent = TRUE)
    .mc_rt$cl <- NULL
    .mc_rt$workers <- NULL
    .mc_rt$tasks <- 0L
  }
  invisible(NULL)
}

#' Apply `fun` to every element of `tasks` on `workers` processes.
#'
#' One worker (or a single task) runs in this process. Errors in a worker stop
#' the demo with the worker's message. Load balancing hands out one task at a
#' time, because trial run times vary with the selected support. A garbage
#' collection after each task frees the C++ memory of the selectors at once
#' (their R6 objects reference themselves, so their finalizers wait for one),
#' which the persistent workers would otherwise accumulate.
#'
#' @param tasks   List or vector of task arguments.
#' @param fun     Function of one task.
#' @param workers Number of worker processes.
#' @return List of results in task order.
mc_map <- function(tasks, fun, workers = 1L) {
  TRexSelector::trex_set_num_threads(1L)
  run <- function(task) {
    out <- fun(task)
    invisible(gc(verbose = FALSE))
    out
  }
  if (workers <= 1L || length(tasks) <= 1L) return(lapply(tasks, run))
  # Fresh workers after .MC_RECYCLE_TASKS tasks bound any memory a long demo
  # leaves behind; the restart happens between calls, never inside a cell.
  if (!is.null(.mc_rt$tasks) && .mc_rt$tasks >= .MC_RECYCLE_TASKS) mc_stop_cluster()
  cl <- .mc_cluster(as.integer(workers))
  .mc_rt$tasks <- (if (is.null(.mc_rt$tasks)) 0L else .mc_rt$tasks) + length(tasks)
  parallel::parLapplyLB(cl, tasks, run, chunk.size = 1L)
}

.MC_RECYCLE_TASKS <- 2000L


# ==============================================================================
# Resumable grid cells
# ==============================================================================

#' Path of the running demo script (NULL when not run via Rscript).
.demo_script <- function() {
  flag <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
  if (length(flag) == 0L) return(NULL)
  normalizePath(sub("^--file=", "", flag[1L]))
}

#' Fingerprint: source hash of demo + suite + family files (plotting excluded) and
#' the TREX_* settings that can change results (not the worker count, the fresh
#' switch, a solver or a *_PARTS selection).
.cache_fingerprint <- function(script) {
  demo_dir <- dirname(script)
  suite_dir <- dirname(demo_dir)
  family_dir <- dirname(dirname(suite_dir))
  files <- c(script,
             sort(list.files(suite_dir, "[.]R$", full.names = TRUE)),
             sort(list.files(family_dir, "[.]R$", full.names = TRUE)))
  files <- files[!grepl("plt_utils", basename(files), fixed = TRUE)]
  src <- substr(.digest_md5(files), 1L, 16L)
  env <- Sys.getenv()
  keep <- startsWith(names(env), "TREX_") & !endsWith(names(env), "_PARTS") &
    !names(env) %in% c("TREX_FRESH", "TREX_NUM_WORKERS", "TREX_SOLVERS")
  paste(c(paste0("src=", src), sort(paste0(names(env)[keep], "=", env[keep]))), collapse = " ")
}

#' One md5 over the contents of several files.
.digest_md5 <- function(files) {
  tmp <- tempfile()
  on.exit(unlink(tmp), add = TRUE)
  writeLines(unname(tools::md5sum(files)), tmp)
  unname(tools::md5sum(tmp))
}

#' Open the cache of the running demo on first use.
.cache_open <- function() {
  if (isTRUE(.mc_rt$cache_opened)) return(invisible(NULL))
  .mc_rt$cache_opened <- TRUE
  .mc_rt$cells <- list()
  .mc_rt$used <- character(0)
  script <- .demo_script()
  if (is.null(script)) return(invisible(NULL))
  dir <- file.path(dirname(script), "simulation_results", "cache")
  dir.create(dir, showWarnings = FALSE, recursive = TRUE)
  path <- file.path(dir, "cells.tsv")
  header <- paste0("# ", .cache_fingerprint(script))
  fresh <- identical(Sys.getenv("TREX_FRESH"), "1")
  if (file.exists(path)) {
    lines <- readLines(path, warn = FALSE)
    if (!fresh && length(lines) > 0L && identical(lines[1L], header)) {
      for (ln in lines[-1L]) {
        parts <- strsplit(ln, "\t", fixed = TRUE)[[1L]]
        n <- length(parts)
        # A line cut off by a hard stop lacks the end marker and is recomputed.
        if (n < 3L || parts[n] != ".") next
        kv <- strsplit(parts[2L:(n - 1L)], "=", fixed = TRUE)
        if (any(lengths(kv) != 2L)) next
        vals <- suppressWarnings(as.numeric(vapply(kv, `[`, "", 2L)))
        .mc_rt$cells[[parts[1L]]] <- setNames(vals, vapply(kv, `[`, "", 1L))
      }
      cat(sprintf("[cache] %d finished cell(s) restored from %s\n\n",
                  length(.mc_rt$cells), path))
    } else {
      file.rename(path, paste0(path, ".stale"))
      cat(sprintf("[cache] %s, old cache kept as cells.tsv.stale\n\n",
                  if (fresh) "TREX_FRESH=1" else "sources or settings changed"))
    }
  }
  if (!file.exists(path)) writeLines(header, path)
  .mc_rt$cache_path <- path
  invisible(NULL)
}

#' Short hash of everything a cell depends on (configs, seeds, grids).
cell_digest <- function(...) {
  substr(.digest_text(paste(deparse(list(...), control = "digits17"), collapse = "")), 1L, 12L)
}

#' md5 of a character string.
.digest_text <- function(text) {
  tmp <- tempfile()
  on.exit(unlink(tmp), add = TRUE)
  writeLines(text, tmp)
  unname(tools::md5sum(tmp))
}

#' Result of one grid cell: from the cache when present, else computed and appended.
#'
#' @param key     Cell label without tabs or newlines; build it from a readable
#'                label and cell_digest() of the cell's inputs, so cells with
#'                different inputs never share a key. Repeats of the same key in
#'                one run are told apart by an occurrence index, as an uncached
#'                run would compute each of them anew.
#' @param compute Function without arguments returning a named numeric vector
#'                or a named list of numeric scalars.
#' @return Named list of the cell's numbers.
cell_cached <- function(key, compute) {
  .cache_open()
  n <- sum(.mc_rt$used == key)
  .mc_rt$used <- c(.mc_rt$used, key)
  key <- paste0(key, "#", n)
  if (is.null(.mc_rt$cache_path)) return(as.list(unlist(compute())))
  hit <- .mc_rt$cells[[key]]
  if (!is.null(hit)) {
    cat(sprintf("  [cache] %s\n", key))
    return(as.list(hit))
  }
  vals <- unlist(compute())
  line <- paste(c(key, sprintf("%s=%.17g", names(vals), vals), "."), collapse = "\t")
  cat(line, "\n", sep = "", file = .mc_rt$cache_path, append = TRUE)
  .mc_rt$cells[[key]] <- vals
  as.list(vals)
}

#' Remove the cache after the demo has written all its results.
cell_cache_done <- function() {
  path <- .mc_rt$cache_path
  if (!is.null(path) && file.exists(path)) unlink(path)
  invisible(NULL)
}


# ==============================================================================
# Solver selection (TREX_SOLVERS)
# ==============================================================================

#' Solver names listed in TREX_SOLVERS (comma separated; empty: no selection).
solver_selection <- function() {
  items <- trimws(strsplit(Sys.getenv("TREX_SOLVERS", ""), ",", fixed = TRUE)[[1L]])
  items[nzchar(items)]
}

#' TRUE when a row label's solver is selected (always TRUE without a selection).
#' The name is matched against the part after ": " of the label.
solver_selected <- function(label) {
  wanted <- solver_selection()
  if (length(wanted) == 0L) return(TRUE)
  sub("^.*?: ", "", label, perl = TRUE) %in% wanted
}

#' File-name suffix of a selection run, e.g. "_tstagewise" ("" without a selection).
solver_file_suffix <- function() {
  names <- solver_selection()
  if (length(names) == 0L) "" else paste0("_", tolower(names), collapse = "")
}
