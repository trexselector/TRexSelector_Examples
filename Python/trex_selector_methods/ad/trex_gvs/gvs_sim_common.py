# ==============================================================================
# gvs_sim_common.py
# ==============================================================================
#
# Shared infrastructure for the T-Rex+GVS Monte Carlo demos. Mirrors the C++
# cpp/trex_selector_methods/ad/trex_gvs/trex_gvs_simulation_utils.hpp:
#
#   fmt_num()                  — trailing-zero-trimmed number formatting.
#   print_section_header()     — framed console section header.
#   method_selected() /
#   num_mc() / out_stem() /
#   skipped_point()            — method selection (TREX_SOLVERS) and the
#                                smoke-run trial count (TREX_NUM_MC).
#   build_gvs_selector()       — TRexGVSSelector adapter (EN / EN_AUG / IEN,
#                                HAC or oracle prior_groups).
#   run_gvs_mc_trials()        — parallel MC loop at one parameter point
#                                (ProcessPoolExecutor; mirrors the OpenMP loop).
#   run_gvs_2d_sweep()         — nested (SNR x rho) grid MC sweep.
#   print_mc_1d_method_table() — per-method 1-D table (mean/sd FDP + TPP).
#   render_mc_matrix() /
#   print_mc_matrix()          — 2-D matrix table with "row \ col" corner axis.
#   save_mc_2d_tables()        — .txt (all six matrices) + tidy .csv writer.
#   save_mc_1d_method_tables() — .txt + .csv writer for the demo-01 method
#                                comparisons (lambda2_method / hc_linkage).
#   noise_floor_corr_max() /
#   run_gvs_mc_trials_cut() /
#   save_mc_cut_tables()       — demo-01 clustering-cut comparison (fixed cut,
#                                Bonferroni noise floor, oracle groups).
#   Block-benchmark utilities  — block metrics, run_block_mc(), and
#                                print_block_grid_table() for demo 08.
#
# Method labels follow the cpp suite: TENET (EN), TENET_AUG (EN, augmented
# solver), TIENET_AUG (IEN). Support / group indices are 0-based.
#
# Seeding mirrors the cpp MC loops: trial i uses data seed = base_seed + i, and
# the same trial seed is passed as the selector seed and (demos 01-07) as
# cv_seed, so fold assignments are unique per trial.
# ==============================================================================

import atexit
import multiprocessing
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Ensure this directory is on sys.path so that spawned workers (ProcessPool-
# Executor on macOS uses spawn) can import this module and its siblings.
# Python's spawn start method copies sys.path to child processes, so inserting
# here in the main process is sufficient for all workers.
# ---------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)
# Family folder: cell_cache.py (resumable grid cells), solver_filter.py (TREX_SOLVERS).
_FAMILY_DIR = os.path.dirname(os.path.dirname(_THIS_DIR))
if _FAMILY_DIR not in sys.path:
    sys.path.insert(0, _FAMILY_DIR)

import cell_cache
import solver_filter

# ------------------------------------------------------------------------------
# Worker budget  (identical helper in every suite's *_common / *_utils module)
#
# N_CORES MC worker processes, each running the selector serially (the library
# default), so a demo never exceeds N_CORES cores in total. The default
# cap matches the cpp demos' default of 8 threads, keeping runtimes
# comparable across the C++ / Python suites.
#
# Override per run, no code edit needed:
#   TREX_NUM_WORKERS=12   exactly 12 workers (clamped to the usable CPU count)
#   TREX_NUM_WORKERS=0    use every usable CPU
# The gvs demos also accept the count as argv[1], which wins over the env var.
#
# "Usable" honours CPU affinity and cgroup quotas wherever the platform reports
# them, so a 128-core host that granted this process only 4 CPUs yields 4, not
# 128 — relevant under containers, Slurm/PBS, taskset and CI runners.
# ------------------------------------------------------------------------------
_WORKER_CAP = 8


def _usable_cpus():
    """Number of CPUs this process may actually use (>= 1, never None)."""
    n = getattr(os, "process_cpu_count", None)   # Python 3.13+, affinity-aware
    n = n() if n is not None else None
    if not n:
        try:
            n = len(os.sched_getaffinity(0))     # Linux on Python < 3.13
        except AttributeError:                   # macOS / Windows
            n = os.cpu_count()
    return max(1, int(n or 1))


def resolve_workers(requested=None):
    """Resolve the worker count from an explicit request, else env, else cap."""
    if requested is None or str(requested).strip() == "":
        requested = os.environ.get("TREX_NUM_WORKERS", "")
    try:
        n = int(str(requested).strip())
    except (TypeError, ValueError):
        n = None
    usable = _usable_cpus()
    if n is None:
        return max(1, min(_WORKER_CAP, usable))
    if n <= 0:                                   # 0 / negative = "all of them"
        return usable
    return max(1, min(n, usable))


N_CORES = resolve_workers()


# ------------------------------------------------------------------------------
# Persistent worker pool (mirrors trex_da/da_sim_common.py)
#
# The sweeps call run_gvs_mc_trials() once per grid point, so a pool created per
# call re-pays (process spawn + `import trexselector`) on every cell:
# measured at ~0.5 s for 6 workers, i.e. ~0.4% of a ~2 min cell. Modest, but
# free to avoid — one lazily-created pool is reused instead, keyed on
# max_workers so a demo's CLI worker-count override still takes effect
# (rebuilding only when the count actually changes).
#
# Workers are recycled after _RECYCLE_AFTER_TASKS trials. Measured worker RSS
# grows ~0.25 MB/trial on a small design (allocator arenas + extension caches
# that never return to baseline) — negligible per cell, but a demo can push
# thousands of trials through one worker, so the pool is rebuilt periodically to
# keep that bounded. Respawning costs ~0.5 s against several minutes of trials,
# i.e. ~0.2%. Rebuilding also drops a pool broken by a dead worker instead of
# handing out the corpse for the rest of the run.
#
# This is done here rather than with ProcessPoolExecutor's own
# max_tasks_per_child because that argument deadlocks Executor.map on
# CPython 3.14.6 / macOS arm64 (main thread blocks in Future.result while the
# executor management thread waits in wait_result_broken_or_wakeup; reproducible
# with pure stdlib, no project code). Recycling between cells is also safer than
# mid-cell: the pool is only ever replaced when no futures are in flight.
#
# mp_context is pinned to "spawn" for two reasons: forking a parent that has already initialised the
# extension's OpenMP runtime risks deadlocking the child; and spawn is what
# macOS and Windows use anyway, so all three platforms now behave identically.
# (That is also why this module puts its own directory on sys.path above —
# spawned children re-import it instead of inheriting the parent's memory.)
# ------------------------------------------------------------------------------
_RECYCLE_AFTER_TASKS = 500
_TASKS_SINCE_SPAWN = 0

_EXECUTOR = None
_EXECUTOR_WORKERS = None
_ATEXIT_REGISTERED = False


def _shutdown_executor():
    """Shut the persistent pool down (atexit hook; safe to call repeatedly)."""
    global _EXECUTOR, _EXECUTOR_WORKERS, _TASKS_SINCE_SPAWN
    if _EXECUTOR is not None:
        _EXECUTOR.shutdown()
        _EXECUTOR = None
        _EXECUTOR_WORKERS = None
        _TASKS_SINCE_SPAWN = 0


def _get_executor(max_workers, n_tasks=0):
    """Return the shared pool, (re)creating it if the worker count changed."""
    global _EXECUTOR, _EXECUTOR_WORKERS, _ATEXIT_REGISTERED, _TASKS_SINCE_SPAWN
    if _EXECUTOR is not None and (_EXECUTOR_WORKERS != max_workers
                                  or _TASKS_SINCE_SPAWN >= _RECYCLE_AFTER_TASKS):
        _shutdown_executor()
    if _EXECUTOR is None:
        _EXECUTOR = ProcessPoolExecutor(
            max_workers=max_workers,
            mp_context=multiprocessing.get_context("spawn"))
        _EXECUTOR_WORKERS = max_workers
        _TASKS_SINCE_SPAWN = 0
        if not _ATEXIT_REGISTERED:
            atexit.register(_shutdown_executor)
            _ATEXIT_REGISTERED = True
    _TASKS_SINCE_SPAWN += max(0, int(n_tasks))
    return _EXECUTOR


# ==============================================================================
# Number formatting (mirrors cpp fmt_num)
# ==============================================================================

def fmt_num(v, max_decimals=4):
    """Fixed-point with at most max_decimals decimals, trailing zeros (and a
    trailing '.') trimmed. E.g. fmt_num(0.05) -> '0.05', fmt_num(2.0) -> '2'."""
    s = f"{float(v):.{max_decimals}f}"
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s


def print_section_header(title):
    """Framed section header (mirrors cdiag::print_section_header)."""
    print("=" * 78)
    for line in str(title).split("\n"):
        print(f" {line}")
    print("=" * 78 + "\n")


# ==============================================================================
# Method selection (TREX_SOLVERS; mirrors the cpp helpers)
# ==============================================================================

def method_selected(label):
    """True when the method of a run label is selected by TREX_SOLVERS.

    The method is the label part before '+' or ' ', after an optional progress
    counter ("[3/30] TIENET_AUG  SNR=0.20 ..." and "TIENET_AUG+CV_1SE_IEN_CCD"
    -> "TIENET_AUG"). Without a selection every method runs.
    """
    text = re.sub(r"^\[\d+/\d+\]\s*", "", str(label))
    return solver_filter.selected(re.split(r"[+ ]", text, maxsplit=1)[0])


def num_mc(value):
    """Monte Carlo trials of a demo: TREX_NUM_MC when a positive integer, else value.

    For short smoke runs. TREX_NUM_MC enters the cell-cache fingerprint, so a
    smoke cache is never reused by a full run.
    """
    try:
        n = int(os.environ.get("TREX_NUM_MC", "").strip())
    except ValueError:
        return value
    return n if n > 0 else value


def out_stem(stem):
    """Output file stem plus the selection suffix, e.g. "_tienet_aug"."""
    return stem + solver_filter.file_suffix()


def skipped_point():
    """Grid point of a skipped (unselected) method: all metrics NaN."""
    nan = float("nan")
    return dict(mean_fdp=nan, mean_tpp=nan, sd_fdp=nan, sd_tpp=nan)


# ==============================================================================
# GVS selector adapter
# ==============================================================================
#
# String -> enum maps are built lazily inside each process so the module can be
# imported (and pickled args shipped to spawn workers) without trexselector
# being loaded in the parent first.

_GVS_TYPE_MAP = None
_EN_SOLVER_MAP = None
_LAMBDA2_MAP = None
_LINKAGE_MAP = None


def _get_enum_maps():
    global _GVS_TYPE_MAP, _EN_SOLVER_MAP, _LAMBDA2_MAP, _LINKAGE_MAP
    if _GVS_TYPE_MAP is None:
        import trexselector as trex
        from trexselector.ml_methods.clustering import LinkageMethod
        _GVS_TYPE_MAP = {"EN": trex.GVSType.EN, "IEN": trex.GVSType.IEN}
        _EN_SOLVER_MAP = {"TENET": trex.ENSolverType.TENET,
                          "TENET_AUG": trex.ENSolverType.TENET_AUG}
        _LAMBDA2_MAP = {"CV_1SE_CCD": trex.LambdaSelectionMethod.CV_1SE_CCD,
                        "CV_1SE_SVD": trex.LambdaSelectionMethod.CV_1SE_SVD,
                        "CV_MIN_CCD": trex.LambdaSelectionMethod.CV_MIN_CCD,
                        "CV_MIN_SVD": trex.LambdaSelectionMethod.CV_MIN_SVD,
                        "CV_1SE_IEN_CCD":
                            trex.LambdaSelectionMethod.CV_1SE_IEN_CCD,
                        "CV_MIN_IEN_CCD":
                            trex.LambdaSelectionMethod.CV_MIN_IEN_CCD,
                        "CV_1SE_TIK_SVD":
                            trex.LambdaSelectionMethod.CV_1SE_TIK_SVD,
                        "CV_MIN_TIK_SVD":
                            trex.LambdaSelectionMethod.CV_MIN_TIK_SVD}
        _LINKAGE_MAP = {"Single": LinkageMethod.Single,
                        "Complete": LinkageMethod.Complete,
                        "Average": LinkageMethod.Average,
                        "WPGMA": LinkageMethod.WPGMA,
                        "Ward": LinkageMethod.Ward,
                        "Centroid": LinkageMethod.Centroid,
                        "Median": LinkageMethod.Median}
    return _GVS_TYPE_MAP, _EN_SOLVER_MAP, _LAMBDA2_MAP, _LINKAGE_MAP


def make_gvs_trex_control(K=20):
    """TRexControlParameter with the shared GVS defaults (mirrors cpp
    make_gvs_trex_control). build_gvs_selector sets solver_type for IEN
    (TIENET_AUG); for EN the binding derives it from en_solver."""
    import trexselector as trex
    tc = trex.TRexControlParameter()
    tc.K = int(K)
    tc.max_dummy_multiplier = 10
    tc.use_max_T_stop = True
    tc.lloop_strategy = trex.LLoopStrategy.HCONCAT
    tc.tloop_stagnation_stop = True
    tc.tloop_max_stagnant_steps = 5
    tc.use_memory_mapping = False
    return tc


def build_gvs_selector(X, y, tFDR, K, gvs_type, corr_max, hc_linkage="Single",
                       en_solver: "str | None" = "TENET",
                       lambda2_method="CV_1SE_CCD",
                       prior_groups=None, cv_seed=None, seed=-1,
                       cv_dfmax_ratio=None):
    """Build and run a TRexGVSSelector; returns (selector, GVSSelectionResult).

    All method knobs are plain strings (picklable across process boundaries):
    gvs_type in {"EN", "IEN"}, en_solver in {"TENET", "TENET_AUG"} for EN or
    None for IEN (whose solver is set to TIENET_AUG, as in the cpp demos),
    hc_linkage in {"Single", "Complete", "Average", "WPGMA", ...},
    lambda2_method in {"CV_1SE_CCD", "CV_1SE_SVD", "CV_1SE_IEN_CCD",
    "CV_1SE_TIK_SVD", ...}.

    `prior_groups` supplies oracle group labels (length-p int vector of
    contiguous 0-based cluster IDs); None discovers groups from the data via
    HAC at `corr_max`. `cv_seed` fixes the CV fold permutation (None keeps the
    default -1 = derive from the selector seed). `cv_dfmax_ratio` sets the
    support cap of the IEN-geometry CV (floor(ratio * n_train) nonzero
    coefficients; None keeps the library default 0.8). The trex_control is
    passed as the binding's separate argument (it overrides the nested
    trex_ctrl).

    The raw binding is called directly and gets the rule as set. Its C++
    constructor runs CV_1SE_IEN_CCD where the IEN track asks for CV_1SE_CCD
    and refuses the IEN-geometry tuners (CV_*_IEN_CCD, CV_*_TIK_SVD) on the
    EN track, so the IEN callers name their IEN tuner explicitly, as the cpp
    demos do. The returned run object exposes selected_indices like the
    wrapper.
    """
    import trexselector as trex
    gvs_map, en_map, l2_map, hc_map = _get_enum_maps()

    gc = trex.TRexGVSControlParameter()
    gc.gvs_type = gvs_map[gvs_type]
    gc.corr_max = float(corr_max)
    gc.hc_linkage = hc_map[hc_linkage]
    # en_solver applies to EN only (TENET / TENET_AUG). IEN callers pass
    # en_solver=None, and the IEN solver is set on the trex control below.
    if en_solver is not None:
        gc.en_solver = en_map[en_solver]
    gc.lambda2_method = l2_map[lambda2_method]
    if cv_seed is not None:
        gc.cv_seed = int(cv_seed)
    if cv_dfmax_ratio is not None:
        gc.cv_dfmax_ratio = float(cv_dfmax_ratio)
    if prior_groups is not None:
        gc.prior_groups = [int(g) for g in np.asarray(prior_groups)]

    tc = make_gvs_trex_control(K)
    if gvs_type == "IEN":
        # Explicit, as in cpp: the binding would map a non-IEN solver_type
        # to the native TIENET.
        tc.solver_type = trex.SolverTypeForTRex.TIENET_AUG

    run = _GVSRun(X, y)
    run.selector = _raw_gvs_selector()(run.X, run.y, float(tFDR), gc, tc,
                                       int(seed), False)
    run.selector.select()
    return run, run.selector.getGVSResult()


def _raw_gvs_selector():
    """The pybind TRexGVSSelector class (no Python-side lambda2 remap)."""
    from trexselector.trex_selector_methods.ad import TRexGVSSelector
    return TRexGVSSelector


class _GVSRun:
    """Keeps X / y alive for the zero-copy binding and exposes the selection."""

    def __init__(self, X, y):
        self.X = np.asfortranarray(X, dtype=np.float64)
        self.y = np.asfortranarray(y, dtype=np.float64)
        self.selector = None

    @property
    def selected_indices(self):
        return self.selector.getSelectedIndices()


# ==============================================================================
# MC inner loop (mirrors cpp run_gvs_mc_trials; OpenMP -> ProcessPoolExecutor)
# ==============================================================================

def _gvs_trial_worker(args):
    """Top-level worker for ProcessPoolExecutor (must stay module-level so
    pickle can resolve it as ('gvs_sim_common', '_gvs_trial_worker'))."""
    import importlib
    from trexselector.utils import compute_fdp, compute_tpp

    (trial_seed, dgp_name, dgp_kwargs, tFDR, K, gvs_type, corr_max,
     hc_linkage, en_solver, lambda2_method, cv_dfmax_ratio) = args

    dgps = importlib.import_module("trex_gvs_dgps")
    dat = getattr(dgps, dgp_name)(seed=trial_seed, **dgp_kwargs)

    # Mirrors the cpp MC loops: selector seed = cv_seed = trial seed.
    sel, _res = build_gvs_selector(
        dat["X"], dat["y"], tFDR, K, gvs_type, corr_max,
        hc_linkage=hc_linkage, en_solver=en_solver,
        lambda2_method=lambda2_method, cv_seed=trial_seed, seed=trial_seed,
        cv_dfmax_ratio=cv_dfmax_ratio)

    selected = list(sel.selected_indices)
    true_support = np.flatnonzero(dat["beta"]).tolist()
    return compute_fdp(selected, true_support), compute_tpp(selected, true_support)


def run_gvs_mc_trials(dgp_name, dgp_kwargs, num_MC, progress_label, tFDR, K,
                      gvs_type, corr_max, hc_linkage="Single",
                      en_solver: "str | None" = "TENET",
                      lambda2_method="CV_1SE_CCD",
                      base_seed=2026, max_workers=N_CORES,
                      cv_dfmax_ratio=None):
    """Run num_MC parallel trials at a single parameter point.

    Each trial generates a fresh dataset via `dgp_name(seed=base_seed + i,
    **dgp_kwargs)` (a trex_gvs_dgps generator) and runs the selector with
    seed = cv_seed = trial seed (cv_dfmax_ratio: see build_gvs_selector).
    Returns dict(mean_fdp, mean_tpp, sd_fdp,
    sd_tpp), matching the cpp GVSGridPointResult. A method not selected by
    TREX_SOLVERS computes nothing and returns NaN metrics (no cache entry).
    """
    if not method_selected(progress_label):
        return skipped_point()
    tasks = [(base_seed + mc, dgp_name, dgp_kwargs, tFDR, K, gvs_type,
              corr_max, hc_linkage, en_solver, lambda2_method, cv_dfmax_ratio)
             for mc in range(num_MC)]

    # One resumable grid cell (cell_cache): restored after a restart.
    def compute():
        print(f"  {progress_label} — Running {num_MC} MC trials ...", flush=True)
        if max_workers > 1:
            results = list(_get_executor(max_workers, len(tasks)).map(
                _gvs_trial_worker, tasks))
        else:
            results = [_gvs_trial_worker(t) for t in tasks]

        fdp = np.array([r[0] for r in results])
        tpp = np.array([r[1] for r in results])
        return dict(mean_fdp=float(fdp.mean()), mean_tpp=float(tpp.mean()),
                    sd_fdp=float(fdp.std(ddof=1)) if num_MC > 1 else 0.0,
                    sd_tpp=float(tpp.std(ddof=1)) if num_MC > 1 else 0.0)

    out = cell_cache.cached(f"gvs|{progress_label}|{cell_cache.digest(tasks)}", compute)

    print(f"  {progress_label} — done. TPP={out['mean_tpp']:.3f}"
          f"  FDP={out['mean_fdp']:.3f}\n", flush=True)
    return out


# ==============================================================================
# Clustering-cut comparison (mirrors the cpp fixed cut / noise floor / oracle
# section of trex_gvs_simulation_utils.hpp)
# ==============================================================================

def noise_floor_corr_max(n, p, alpha=0.05):
    """Bonferroni noise floor for |corr| between independent columns:
    corr_max = tanh(z / sqrt(n - 3)) with P(Z > z) = alpha / (2 m) over the
    m = p (p - 1) / 2 column pairs (Fisher z); 0.346 at n = 200, p = 500."""
    from statistics import NormalDist
    q = alpha / (2.0 * 0.5 * p * (p - 1))
    z = NormalDist().inv_cdf(1.0 - q)
    return float(np.tanh(z / np.sqrt(n - 3)))


def _gvs_group_rates(sel, prior_groups, true_support):
    """Group-level rates of one selection against the oracle prior_groups
    (background columns are singleton groups): share of active groups hit,
    share of active groups selected in full, share of touched groups without
    an active variable."""
    g = np.asarray(prior_groups, dtype=int)
    size = np.bincount(g)
    active = np.bincount(g[np.asarray(true_support, dtype=int)], minlength=size.size)
    chosen = np.bincount(g[np.asarray(sel, dtype=int)], minlength=size.size) \
        if len(sel) else np.zeros(size.size, dtype=int)
    act, touched = active > 0, chosen > 0
    n_act, n_touched = int(act.sum()), int(touched.sum())
    hit = float((act & touched).sum() / n_act) if n_act else 0.0
    full = float((act & (chosen == size)).sum() / n_act) if n_act else 0.0
    fdp = float((touched & ~act).sum() / n_touched) if n_touched else 0.0
    return hit, full, fdp


def _gvs_cut_trial_worker(args):
    """One trial of the cut comparison (module-level for pickling): FDP, TPP,
    group rates and the number of clusters with >= 2 members."""
    import importlib

    from trexselector.utils import compute_fdp, compute_tpp

    (trial_seed, dgp_name, dgp_kwargs, tFDR, K, gvs_type, corr_max,
     hc_linkage, en_solver, lambda2_method, oracle, cv_dfmax_ratio) = args

    dgps = importlib.import_module("trex_gvs_dgps")
    dat = getattr(dgps, dgp_name)(seed=trial_seed, **dgp_kwargs)

    # Same setup as _gvs_trial_worker; oracle groups bypass the HAC cut
    sel, res = build_gvs_selector(
        dat["X"], dat["y"], tFDR, K, gvs_type, corr_max,
        hc_linkage=hc_linkage, en_solver=en_solver,
        lambda2_method=lambda2_method,
        prior_groups=dat["prior_groups"] if oracle else None,
        cv_seed=trial_seed, seed=trial_seed, cv_dfmax_ratio=cv_dfmax_ratio)

    selected = list(sel.selected_indices)
    true_support = np.flatnonzero(dat["beta"]).tolist()
    sizes = np.bincount(np.asarray(res.groups_vec, dtype=int).ravel(),
                        minlength=int(res.max_clusters))
    hit, full, gfdp = _gvs_group_rates(selected, dat["prior_groups"], true_support)
    return (compute_fdp(selected, true_support), compute_tpp(selected, true_support),
            hit, full, gfdp, float((sizes >= 2).sum()))


def run_gvs_mc_trials_cut(dgp_name, dgp_kwargs, num_MC, progress_label, tFDR,
                          K, gvs_type, corr_max, hc_linkage="Single",
                          en_solver: "str | None" = "TENET",
                          lambda2_method="CV_1SE_CCD", base_seed=2026,
                          oracle=False, max_workers=N_CORES,
                          cv_dfmax_ratio=None):
    """num_MC parallel trials at one fixed operating point under one cut
    (corr_max, or the DGP's prior_groups when oracle=True; cv_dfmax_ratio:
    see build_gvs_selector). Returns mean / sd
    of FDP, TPP, grp_hit, grp_full, grp_fdp and n_groups, matching the cpp
    GVSCutPointResult. A method not selected by TREX_SOLVERS computes nothing
    and returns NaN metrics (no cache entry)."""
    keys = ("fdp", "tpp", "grp_hit", "grp_full", "grp_fdp", "n_groups")
    if not method_selected(progress_label):
        nan = float("nan")
        return {name: nan for k in keys
                for name in (k if k.startswith(("grp", "n_")) else f"mean_{k}", f"sd_{k}")}
    tasks = [(base_seed + mc, dgp_name, dgp_kwargs, tFDR, K, gvs_type,
              corr_max, hc_linkage, en_solver, lambda2_method, bool(oracle),
              cv_dfmax_ratio)
             for mc in range(num_MC)]

    def compute():
        print(f"  {progress_label} — Running {num_MC} MC trials ...", flush=True)
        if max_workers > 1:
            results = list(_get_executor(max_workers, len(tasks)).map(
                _gvs_cut_trial_worker, tasks))
        else:
            results = [_gvs_cut_trial_worker(t) for t in tasks]
        arr = np.array(results, dtype=float)
        out = {}
        for k, col in zip(keys, arr.T):
            name = k if k.startswith(("grp", "n_")) else f"mean_{k}"
            out[name] = float(col.mean())
            out[f"sd_{k}"] = float(col.std(ddof=1)) if num_MC > 1 else 0.0
        return out

    out = cell_cache.cached(f"gvscut|{progress_label}|{cell_cache.digest(tasks)}",
                            compute)

    print(f"  {progress_label} — done. TPP={out['mean_tpp']:.3f}"
          f"  FDP={out['mean_fdp']:.3f}  groups={out['n_groups']:.3f}\n",
          flush=True)
    return out


# ==============================================================================
# 2-D (SNR x rho) sweep (mirrors cpp run_gvs_2d_sweep)
# ==============================================================================

def run_gvs_2d_sweep(dgp_name, cell_kwargs_fn, snr_grid, rho_grid, cfg,
                     gvs_type, en_solver: "str | None", label,
                     hc_linkage="Single",
                     lambda2_method="CV_1SE_CCD", max_workers=N_CORES,
                     cv_dfmax_ratio=None):
    """Run a nested SNR x rho grid MC sweep for one method variant.

    `cell_kwargs_fn(snr, rho)` returns the picklable DGP kwargs for one grid
    cell (the demos map rho to sd_x here where required); cv_dfmax_ratio
    is passed to every cell (see build_gvs_selector). Returns
    results[i_snr][i_rho], each a run_gvs_mc_trials() dict (NaN metrics for a
    method not selected by TREX_SOLVERS).
    """
    n_snr, n_rho = len(snr_grid), len(rho_grid)

    if not method_selected(label):
        print(f"\n  Method: {label}  (skipped, not in TREX_SOLVERS)\n")
        return [[skipped_point() for _ in range(n_rho)] for _ in range(n_snr)]

    results = [[None] * n_rho for _ in range(n_snr)]

    print(f"\n  Method: {label}\n")

    for i_snr, snr in enumerate(snr_grid):
        for i_rho, rho in enumerate(rho_grid):
            lbl = (f"[{i_snr * n_rho + i_rho + 1}/{n_snr * n_rho}] {label}"
                   f"  SNR={snr:.2f}  rho={rho:.2f}")
            results[i_snr][i_rho] = run_gvs_mc_trials(
                dgp_name, cell_kwargs_fn(snr, rho), cfg["num_MC"], lbl,
                cfg["tFDR"], cfg["K"], gvs_type, cfg["corr_max"],
                hc_linkage=hc_linkage, en_solver=en_solver,
                lambda2_method=lambda2_method, base_seed=cfg["base_seed"],
                max_workers=max_workers, cv_dfmax_ratio=cv_dfmax_ratio)
    return results


# ==============================================================================
# Print MC 1-D method table (mirrors cpp print_mc_1d_method_table)
# ==============================================================================

def print_mc_1d_method_table(method_label, param_labels, results, fout=None):
    """Table for one method variant: rows = parameter labels, columns =
    mean_FDP, sd_FDP, mean_TPP, sd_TPP."""
    rw = max([12] + [len(lbl) + 2 for lbl in param_labels])  # widened for long labels
    cw = 10
    total_w = rw + 4 * cw

    lines = [f"\n  {method_label}",
             "  " + "-" * total_w,
             ("  " + " " * rw + f"{'mean_FDP':>{cw}}{'sd_FDP':>{cw}}"
              f"{'mean_TPP':>{cw}}{'sd_TPP':>{cw}}"),
             "  " + "-" * total_w]
    for lbl, r in zip(param_labels, results):
        lines.append(f"  {lbl:<{rw}}{r['mean_fdp']:>{cw}.4f}"
                     f"{r['sd_fdp']:>{cw}.4f}{r['mean_tpp']:>{cw}.4f}"
                     f"{r['sd_tpp']:>{cw}.4f}")
    s = "\n".join(lines) + "\n\n"
    print(s, end="")
    if fout is not None:
        fout.write(s)


# ==============================================================================
# Print MC 2-D matrix (mirrors cpp render_mc_matrix_str / print_mc_matrix)
# ==============================================================================

def _strip_axis_prefix(labels):
    """Strip a common 'axis=value' prefix; returns (axis_name, bare_values)."""
    axis, bare = "", []
    for lbl in labels:
        if "=" in lbl:
            head, val = lbl.split("=", 1)
            if not axis:
                axis = head
            bare.append(val)
        else:
            bare.append(lbl)
    return axis, bare


def render_mc_matrix(title, row_labels, col_labels, values, print_tpp):
    """Render a 2-D sweep matrix (row axis left, column axis named once in the
    '<row_axis> \\ <col_axis>' corner cell; bare swept values as headers)."""
    row_axis, rvals = _strip_axis_prefix(list(row_labels))
    col_axis, cvals = _strip_axis_prefix(list(col_labels))

    corner = f"{row_axis} \\ {col_axis}" if (row_axis or col_axis) else ""
    rw = max(max((len(v) for v in rvals), default=0), len(corner))
    rwi = max(rw + 2, 10)
    cw = 10
    total = rwi + cw * len(cvals)
    key = "mean_tpp" if print_tpp else "mean_fdp"

    lines = [f"\n  {title}", "  " + "-" * total,
             "  " + f"{corner:>{rwi}}" + "".join(f"{c:>{cw}}" for c in cvals),
             "  " + "-" * total]
    for rlbl, row in zip(rvals, values):
        lines.append("  " + f"{rlbl:>{rwi}}"
                     + "".join(f"{cell[key]:>{cw}.4f}" for cell in row))
    return "\n".join(lines) + "\n\n"


def print_mc_matrix(title, row_labels, col_labels, values, print_tpp):
    print(render_mc_matrix(title, row_labels, col_labels, values, print_tpp),
          end="")


# ==============================================================================
# Save MC 2-D tables (mirrors cpp save_mc_2d_tables)
# ==============================================================================

def save_mc_2d_tables(scenario_tag, row_labels, col_labels, en_2d, en_aug_2d,
                      ien_2d, cfg, out_dir, file_stem):
    """Write the six 2-D matrices to <out_dir>/<file_stem>.txt and the tidy
    long-format CSV (method,metric,row_label,col_label,mean,sd) to .csv.
    A TREX_SOLVERS run appends its suffix to file_stem (out_stem)."""
    if not file_stem:
        return
    os.makedirs(out_dir, exist_ok=True)

    txt_path = os.path.join(out_dir, out_stem(file_stem) + ".txt")
    with open(txt_path, "w") as txt:
        txt.write("\n" + "=" * 78 + "\n"
                  f"  T-Rex+GVS MC 2D Grid: {scenario_tag}"
                  f"  (MC={cfg['num_MC']}, tFDR={fmt_num(cfg['tFDR'])},"
                  f" n={cfg['n']}, p={cfg['p']})\n" + "=" * 78 + "\n")
        for title, vals, tpp in (
                ("mean_TPP [TENET]", en_2d, True),
                ("mean_FDP [TENET]", en_2d, False),
                ("mean_TPP [TENET_AUG]", en_aug_2d, True),
                ("mean_FDP [TENET_AUG]", en_aug_2d, False),
                ("mean_TPP [TIENET_AUG]", ien_2d, True),
                ("mean_FDP [TIENET_AUG]", ien_2d, False)):
            txt.write(render_mc_matrix(title, row_labels, col_labels, vals, tpp))
    print(f"[Info] TXT saved to: {txt_path}")

    # Row order mirrors the cpp CSV: per grid cell, mean_FDP then mean_TPP
    # for TENET, TENET_AUG, and TIENET_AUG in turn.
    rows = []
    for i, rlbl in enumerate(row_labels):
        for j, clbl in enumerate(col_labels):
            for method, mat in (("TENET", en_2d), ("TENET_AUG", en_aug_2d),
                                ("TIENET_AUG", ien_2d)):
                cell = mat[i][j]
                rows.append(dict(method=method, metric="mean_FDP",
                                 row_label=rlbl, col_label=clbl,
                                 mean=round(cell["mean_fdp"], 6),
                                 sd=round(cell["sd_fdp"], 6)))
                rows.append(dict(method=method, metric="mean_TPP",
                                 row_label=rlbl, col_label=clbl,
                                 mean=round(cell["mean_tpp"], 6),
                                 sd=round(cell["sd_tpp"], 6)))

    # na_rep: skipped methods write "nan", as the cpp CSV does
    csv_path = os.path.join(out_dir, out_stem(file_stem) + ".csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False,
                              float_format="%.6f", na_rep="nan")
    print(f"[Info] CSV saved to: {csv_path}\n")


# ==============================================================================
# Save MC 1-D method-comparison tables (demo 01 Parts 2-3)
# ==============================================================================

def save_mc_1d_method_tables(scenario_tag, comparison, param_col, param_labels,
                             en_results, ien_results, fixed_point_note, cfg,
                             out_dir, file_stem, ien_param_labels=None):
    """Console + .txt + .csv writer for a fixed-operating-point method
    comparison (TENET vs TIENET_AUG over lambda2_method / hc_linkage levels).
    CSV schema mirrors cpp: method,metric,<param_col>,mean,sd.
    ien_param_labels gives the IEN track its own levels (lambda2 rules); the
    CSV then lists the EN rows before the IEN rows, as the cpp Part 2 does.
    A TREX_SOLVERS run appends its suffix to file_stem (out_stem)."""
    os.makedirs(out_dir, exist_ok=True)
    txt_path = os.path.join(out_dir, out_stem(file_stem) + ".txt")

    with open(txt_path, "w") as txt:
        def write(s):
            print(s, end="")
            txt.write(s)

        write("\n" + "=" * 78 + "\n"
              f"  T-Rex+GVS MC: {scenario_tag} — {comparison} comparison"
              f"  ({fixed_point_note}, MC={cfg['num_MC']},"
              f" tFDR={fmt_num(cfg['tFDR'])})\n" + "=" * 78 + "\n")
        print_mc_1d_method_table("TENET", param_labels, en_results, fout=txt)
        print_mc_1d_method_table("TIENET_AUG",
                                 ien_param_labels or param_labels,
                                 ien_results, fout=txt)
        write("=" * 78 + "\n\n")
    print(f"[Info] TXT saved to: {txt_path}")

    def _rows(method, lbl, r):
        return [dict(method=method, metric="mean_FDP", **{param_col: lbl},
                     mean=round(r["mean_fdp"], 6), sd=round(r["sd_fdp"], 6)),
                dict(method=method, metric="mean_TPP", **{param_col: lbl},
                     mean=round(r["mean_tpp"], 6), sd=round(r["sd_tpp"], 6))]

    rows = []
    if ien_param_labels is None:
        for lbl, en_r, ien_r in zip(param_labels, en_results, ien_results):
            rows += _rows("TENET", lbl, en_r) + _rows("TIENET_AUG", lbl, ien_r)
    else:
        for lbl, en_r in zip(param_labels, en_results):
            rows += _rows("TENET", lbl, en_r)
        for lbl, ien_r in zip(ien_param_labels, ien_results):
            rows += _rows("TIENET_AUG", lbl, ien_r)
    csv_path = os.path.join(out_dir, out_stem(file_stem) + ".csv")
    pd.DataFrame(rows)[["method", "metric", param_col, "mean", "sd"]].to_csv(
        csv_path, index=False, float_format="%.6f", na_rep="nan")
    print(f"[Info] CSV saved to: {csv_path}\n")


# ==============================================================================
# Save the clustering-cut comparison (demo 01 Part 4)
# ==============================================================================

_CUT_ROWS = (("FDR", "mean_fdp", "mean_FDP", "sd_fdp"),
             ("TPR", "mean_tpp", "mean_TPP", "sd_tpp"),
             ("group hit", "grp_hit", "grp_hit", "sd_grp_hit"),
             ("group full", "grp_full", "grp_full", "sd_grp_full"),
             ("group FDR", "grp_fdp", "grp_FDP", "sd_grp_fdp"),
             ("groups", "n_groups", "n_groups", "sd_n_groups"))


def save_mc_cut_tables(scenario_tag, cut_labels, en_results, ien_results,
                       fixed_snr, fixed_rho, floor_cut, cfg, out_dir, file_stem):
    """Console + .txt + .csv writer for the clustering-cut comparison (one
    column per cut, TENET and TIENET_AUG blocks). CSV schema mirrors cpp:
    method,metric,cut,mean,sd. A TREX_SOLVERS run appends its suffix to
    file_stem (out_stem)."""
    os.makedirs(out_dir, exist_ok=True)
    lines = ["", "=" * 78,
             (f"  T-Rex+GVS MC: {scenario_tag} — clustering-cut comparison"
              f"  (SNR={fmt_num(fixed_snr)}, rho={fmt_num(fixed_rho)},"
              f" MC={cfg['num_MC']}, tFDR={fmt_num(cfg['tFDR'])})"),
             f"  noise floor corr_max = {floor_cut:.4f} (Bonferroni, alpha = 0.05)",
             "=" * 78]
    for name, rs in (("TENET", en_results), ("TIENET_AUG", ien_results)):
        lines += ["", name, f"{'  metric':<16}" + "".join(f"{c:>14}" for c in cut_labels)]
        for label, key, _metric, _sd in _CUT_ROWS:
            lines.append(f"{'  ' + label:<16}" + "".join(f"{r[key]:>14.4f}" for r in rs))
    lines.append("=" * 78)
    table = "\n".join(lines) + "\n\n"
    print(table, end="")

    txt_path = os.path.join(out_dir, out_stem(file_stem) + ".txt")
    with open(txt_path, "w") as txt:
        txt.write(table)
    print(f"[Info] TXT saved to: {txt_path}")

    rows = []
    for name, rs in (("TENET", en_results), ("TIENET_AUG", ien_results)):
        for cut, r in zip(cut_labels, rs):
            for _label, key, metric, sd in _CUT_ROWS:
                rows.append(dict(method=name, metric=metric, cut=cut,
                                 mean=round(r[key], 6), sd=round(r[sd], 6)))
    csv_path = os.path.join(out_dir, out_stem(file_stem) + ".csv")
    pd.DataFrame(rows)[["method", "metric", "cut", "mean", "sd"]].to_csv(
        csv_path, index=False, float_format="%.6f", na_rep="nan")
    print(f"[Info] CSV saved to: {csv_path}\n")


# ==============================================================================
# ==============================================================================
# Block-equicorrelated benchmark utilities (demo 08; mirror the cpp
# block-bench section of trex_gvs_simulation_utils.hpp)
# ==============================================================================
# ==============================================================================

def _block_cols(blk, block_size):
    """0-based column indices of block `blk`."""
    return range(blk * block_size, (blk + 1) * block_size)


def compute_block_hit_rate(sel, active, block_size):
    """Fraction of active blocks with at least one selected variable."""
    if len(active) == 0:
        return 0.0
    sel_set = set(int(i) for i in sel)
    hits = sum(any(c in sel_set for c in _block_cols(int(blk), block_size))
               for blk in active)
    return hits / len(active)


def compute_block_fdp(sel, active, G, block_size):
    """Block-level FDP = (null blocks hit) / max(total blocks hit, 1)."""
    sel_set = set(int(i) for i in sel)
    active_set = set(int(a) for a in active)
    total_hit = null_hit = 0
    for blk in range(G):
        if any(c in sel_set for c in _block_cols(blk, block_size)):
            total_hit += 1
            if blk not in active_set:
                null_hit += 1
    return null_hit / max(total_hit, 1)


def compute_full_block_rate(sel, active, block_size):
    """Fraction of active blocks fully contained in the selection."""
    if len(active) == 0:
        return 0.0
    sel_set = set(int(i) for i in sel)
    fully = sum(all(c in sel_set for c in _block_cols(int(blk), block_size))
                for blk in active)
    return fully / len(active)


def compute_null_block_activation(sel, active, G, block_size):
    """Fraction of null blocks with at least one selected variable."""
    sel_set = set(int(i) for i in sel)
    active_set = set(int(a) for a in active)
    null_total = null_hit = 0
    for blk in range(G):
        if blk in active_set:
            continue
        null_total += 1
        if any(c in sel_set for c in _block_cols(blk, block_size)):
            null_hit += 1
    return 0.0 if null_total == 0 else null_hit / null_total


def compute_block_purity_rate(groups_vec, G, block_size):
    """Fraction of true blocks whose members all share one groups_vec label
    (1.0 by construction for oracle runs)."""
    gv = np.asarray(groups_vec).ravel()
    pure = sum(bool(np.all(gv[k * block_size:(k + 1) * block_size]
                           == gv[k * block_size]))
               for k in range(G))
    return pure / G


def compute_exact_support(sel, true_support):
    """1.0 if the selected set equals the true support, else 0.0."""
    return 1.0 if sorted(int(i) for i in sel) == \
        sorted(int(i) for i in true_support) else 0.0


_BLOCK_METRICS = ("coord_fdp", "coord_tpr", "exact_support", "block_hit",
                  "block_fdp", "full_block", "null_act", "purity",
                  "lambda2", "T_stop", "M_found", "elapsed")


def _run_block_method(dat, gvs_type, oracle, cfg, seed):
    """Run one GVS method variant on one dataset; returns a metrics dict.
    Mirrors cpp run_block_method (cv_seed stays at its default -1 = derived
    from the selector seed, as in the cpp block bench)."""
    from trexselector.utils import compute_fdp, compute_tpp

    t0 = time.perf_counter()
    # EN uses the configured solver (TENET / TENET_AUG); IEN runs TIENET_AUG
    # with the IEN-geometry lambda2 rule (the true support lies below its
    # lambda1 support cap here), so en_solver is not applicable.
    en_solver = cfg["en_solver"] if gvs_type == "EN" else None
    lambda2_method = "CV_1SE_CCD" if gvs_type == "EN" else "CV_1SE_IEN_CCD"
    prior_groups = dat["prior_groups"] if oracle else None
    sel, res = build_gvs_selector(
        dat["X"], dat["y"], cfg["tFDR"], cfg["K"], gvs_type, cfg["corr_max"],
        hc_linkage="Single", en_solver=en_solver,
        lambda2_method=lambda2_method, prior_groups=prior_groups, seed=seed)
    elapsed = time.perf_counter() - t0

    si = list(sel.selected_indices)
    ts = list(dat["true_support"])
    G, bs = dat["G"], dat["block_size"]
    active = list(dat["active_blocks"])
    return dict(
        coord_fdp=compute_fdp(si, ts),
        coord_tpr=compute_tpp(si, ts),
        exact_support=compute_exact_support(si, ts),
        block_hit=compute_block_hit_rate(si, active, bs),
        block_fdp=compute_block_fdp(si, active, G, bs),
        full_block=compute_full_block_rate(si, active, bs),
        null_act=compute_null_block_activation(si, active, G, bs),
        purity=compute_block_purity_rate(res.groups_vec, G, bs),
        lambda2=res.lambda2_used,
        T_stop=float(res.T_stop),
        M_found=float(res.max_clusters),
        elapsed=elapsed,
    )


def _block_trial_worker(args):
    """One MC trial of the block benchmark: generate a dataset and run all
    four method variants on it (M1 EN+HAC, M2 EN+oracle, M3 IEN+HAC,
    M4 IEN+oracle). Module-level for pickling."""
    import importlib
    trial_seed, cfg, scenario, rho, snr = args

    dgps = importlib.import_module("trex_gvs_dgps")
    dat = dgps.dgp_block_equicorr(
        n=cfg["n"], p=cfg["p"], G=cfg["G"], block_size=cfg["block_size"],
        n_active_blocks=cfg["n_active_blocks"], rho=rho, snr=snr,
        scenario=scenario, seed=trial_seed, b=cfg["b"])

    return dict(
        M1=_run_block_method(dat, "EN", False, cfg, trial_seed),
        M2=_run_block_method(dat, "EN", True, cfg, trial_seed),
        M3=_run_block_method(dat, "IEN", False, cfg, trial_seed),
        M4=_run_block_method(dat, "IEN", True, cfg, trial_seed),
    )


def run_block_mc(cfg, scenario, rho, snr, cell_base_seed, label,
                 max_workers=N_CORES):
    """Run num_MC parallel trials at one (scenario, rho, snr) cell and
    aggregate all four methods. Trial i uses seed = cell_base_seed + i; all
    four methods see identical data within a trial."""
    tasks = [(cell_base_seed + mc, cfg, scenario, rho, snr)
             for mc in range(cfg["num_MC"])]

    # One resumable grid cell (cell_cache), flattened to "M1.mean_coord_tpr" etc.
    def compute():
        print(f"  {label} — running {cfg['num_MC']} MC trials...", flush=True)
        if max_workers > 1:
            trials = list(_get_executor(max_workers, len(tasks)).map(
                _block_trial_worker, tasks))
        else:
            trials = [_block_trial_worker(t) for t in tasks]

        flat = {}
        for mk in ("M1", "M2", "M3", "M4"):
            for metric in _BLOCK_METRICS:
                vals = np.array([float(t[mk][metric]) for t in trials])
                flat[f"{mk}.mean_{metric}"] = float(vals.mean())
                flat[f"{mk}.sd_{metric}"] = float(vals.std(ddof=1)) \
                    if len(vals) > 1 else 0.0
        return flat

    flat = cell_cache.cached(f"block|{label}|{cell_cache.digest(tasks)}", compute)
    out = {mk: {k.split(".", 1)[1]: v for k, v in flat.items() if k.startswith(mk + ".")}
           for mk in ("M1", "M2", "M3", "M4")}

    print(f"  {label} — done."
          f"  M1 TPR={out['M1']['mean_coord_tpr']:.3f}"
          f"  M2 TPR={out['M2']['mean_coord_tpr']:.3f}"
          f"  M3 TPR={out['M3']['mean_coord_tpr']:.3f}"
          f"  M4 TPR={out['M4']['mean_coord_tpr']:.3f}\n", flush=True)
    return out


def print_block_grid_table(scenario_tag, scenario, rho_grid, snr_grid,
                           results, cfg, out_dir, file_stem=""):
    """Formatted MC aggregate table for one truth scenario (rows M1-M4 per
    (rho, SNR) cell), mirrored from cpp print_block_grid_table. If file_stem
    is non-empty, also writes the table as .txt and all aggregates as a tidy
    .csv into out_dir (a TREX_SOLVERS run appends its suffix, out_stem)."""
    blk_metric_name = "full_blk" if scenario == "WHOLE" else "blk_hit"
    blk_key = "mean_full_block" if scenario == "WHOLE" else "mean_block_hit"
    methods = (("M1(EN-C)", "M1"), ("M2(EN-O)", "M2"),
               ("M3(IE-C)", "M3"), ("M4(IE-O)", "M4"))
    cw = 10

    lines = ["", "=" * 90,
             f"  Block Bench MC: {scenario_tag}  (MC={cfg['num_MC']},"
             f" tFDR={fmt_num(cfg['tFDR'])}, n={cfg['n']}, p={cfg['p']},"
             f" G={cfg['G']}, blk={cfg['block_size']})",
             "=" * 90]
    for i_rho, rho in enumerate(rho_grid):
        lines.append(f"\n  rho = {rho:.2f}")
        lines.append("  " + f"{'method':<12}" + f"{'SNR':>7}"
                     + f"{'TPR':>{cw}}{'FDR':>{cw}}{'block_FDR':>{cw}}"
                     + f"{blk_metric_name:>{cw}}{'purity':>{cw}}"
                     + f"{'T_stop':>{cw}}{'M_found':>{cw}}")
        lines.append("  " + "-" * (12 + 7 + 7 * cw))
        for i_snr, snr in enumerate(snr_grid):
            cell = results[i_rho][i_snr]
            for mlbl, mk in methods:
                a = cell[mk]
                lines.append(
                    f"  {mlbl:<12}{snr:>7.2f}"
                    f"{a['mean_coord_tpr']:>{cw}.4f}"
                    f"{a['mean_coord_fdp']:>{cw}.4f}"
                    f"{a['mean_block_fdp']:>{cw}.4f}"
                    f"{a[blk_key]:>{cw}.4f}"
                    f"{a['mean_purity']:>{cw}.4f}"
                    f"{a['mean_T_stop']:>{cw}.1f}"
                    f"{a['mean_M_found']:>{cw}.1f}")
            lines.append("")
    lines.append("=" * 90 + "\n")
    table = "\n".join(lines)
    print(table)

    if not file_stem:
        return
    os.makedirs(out_dir, exist_ok=True)

    txt_path = os.path.join(out_dir, out_stem(file_stem) + ".txt")
    with open(txt_path, "w") as txt:
        txt.write(table + "\n")
    print(f"[Info] TXT saved to: {txt_path}")

    rows = []
    for i_rho, rho in enumerate(rho_grid):
        for i_snr, snr in enumerate(snr_grid):
            for mlbl, mk in methods:
                a = results[i_rho][i_snr][mk]
                rows.append(dict(
                    scenario=scenario_tag, rho=rho, snr=snr, method=mlbl,
                    mean_coord_tpr=round(a["mean_coord_tpr"], 6),
                    sd_coord_tpr=round(a["sd_coord_tpr"], 6),
                    mean_coord_fdp=round(a["mean_coord_fdp"], 6),
                    sd_coord_fdp=round(a["sd_coord_fdp"], 6),
                    mean_exact_sup=round(a["mean_exact_support"], 6),
                    mean_block_hit=round(a["mean_block_hit"], 6),
                    mean_block_fdp=round(a["mean_block_fdp"], 6),
                    mean_full_block=round(a["mean_full_block"], 6),
                    mean_null_act=round(a["mean_null_act"], 6),
                    mean_purity=round(a["mean_purity"], 6),
                    mean_lambda2=round(a["mean_lambda2"], 6),
                    mean_T_stop=round(a["mean_T_stop"], 6),
                    mean_M_found=round(a["mean_M_found"], 6)))
    csv_path = os.path.join(out_dir, out_stem(file_stem) + ".csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False, float_format="%.6f",
                              na_rep="nan")
    print(f"[Info] CSV saved to: {csv_path}\n")
