# ==============================================================================
# ts_perf_solvers.R
# ==============================================================================
#
# The T-solvers compared by the performance demos (demo_ts_perf_*), each with
# the settings of its own solver demo. Mirrors cpp/tsolvers/ts_perf_solvers.hpp.
#
# Every solver runs as a reader (normalize = FALSE, intercept = FALSE): the
# caller centers and scales X and D and centers y once in front, and all
# solvers share the same R matrices. The data are marked precentered, so the
# row-bounded active sets stop at the rank n - 1 of the centered design
# (T-ENET with lambda2 > 0 has no such cap). T-AFS appears once per rho.
#
# TREX_SOLVERS (comma separated labels, e.g. "T-Stagewise") restricts
# demo_solvers() to those solvers, and a selection run writes its tables with
# solver_file_suffix(), next to the committed ones of the full list.
# cpp/tsolvers/merge_result_files.py splices them into the full tables.
#
# ==============================================================================

#' Name, family and factory of one compared solver; make(X, D, y) builds it as a reader.
.solver_spec <- function(cls, name, family, ...) {
  extra <- list(...)
  make <- function(X, D, y) {
    options <- do.call(tsolver_options, c(list(normalize = FALSE, intercept = FALSE, precentered = TRUE), extra))
    cls$new(X, D, y, options = options)
  }
  list(name = name, family = family, make = make)
}

#' Solver labels listed in TREX_SOLVERS (comma separated; empty: no selection).
solver_selection <- function() {
  items <- trimws(strsplit(Sys.getenv("TREX_SOLVERS", ""), ",", fixed = TRUE)[[1L]])
  items[nzchar(items)]
}

#' TRUE when TREX_SOLVERS selects the label `name` (always TRUE without a selection).
solver_selected <- function(name) {
  wanted <- solver_selection()
  length(wanted) == 0L || name %in% wanted
}

#' File-name suffix of a selection run, e.g. "_t_stagewise" ("" without a selection).
#' Lower case, every run of other characters than letters and digits becomes "_".
solver_file_suffix <- function() {
  names <- solver_selection()
  if (length(names) == 0L) return("")
  paste0("_", gsub("^_+|_+$", "", gsub("[^a-z0-9]+", "_", tolower(names))), collapse = "")
}

#' The 12 demo solvers with their demo settings, T-AFS for rho in {0.1, 0.3, 0.5, 1}; only those TREX_SOLVERS
#' selects.
demo_solvers <- function() {
  lars <- "LARS-based"
  omp <- "OMP-based"
  afs <- "AFS-based"
  specs <- list(
    .solver_spec(TLARS_Solver, "T-LARS", lars),
    .solver_spec(TLASSO_Solver, "T-LASSO", lars),
    .solver_spec(TSTEPWISE_Solver, "T-Stepwise", lars),
    .solver_spec(TENET_Solver, "T-ENET (lambda2 0.5)", lars, lambda2 = 0.5),
    .solver_spec(TSTAGEWISE_Solver, "T-Stagewise", lars),
    .solver_spec(TOMP_Solver, "T-OMP", omp),
    .solver_spec(TGP_Solver, "T-GP", omp),
    .solver_spec(TACGP_Solver, "T-ACGP", omp),
    .solver_spec(TMP_Solver, "T-MP", omp),
    .solver_spec(TNCGMP_Solver, "T-NCGMP", omp, ncgmp_variant = "line_search"),
    .solver_spec(TOOLS_Solver, "T-OLS", omp),
    .solver_spec(TAFS_Solver, "T-AFS (rho 0.1)", afs, rho_afs = 0.1),
    .solver_spec(TAFS_Solver, "T-AFS (rho 0.3)", afs, rho_afs = 0.3),
    .solver_spec(TAFS_Solver, "T-AFS (rho 0.5)", afs, rho_afs = 0.5),
    .solver_spec(TAFS_Solver, "T-AFS (rho 1)", afs, rho_afs = 1.0)
  )
  Filter(function(spec) solver_selected(spec$name), specs)
}
