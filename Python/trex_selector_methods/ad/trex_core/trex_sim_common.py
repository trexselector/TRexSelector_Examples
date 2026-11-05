# ==============================================================================
# trex_sim_common.py
# ==============================================================================
#
# Shared infrastructure for classical T-Rex MC simulations.
# Mirrors R/trex_selector_methods/ad/trex_core/trex_sim_utils.R.
#
# Contents:
#   SOLVERS_DEFAULT              — List of 14 solver descriptors.
#   selected_solvers()           — The solvers TREX_SOLVERS selects.
#   resolve_num_mc()             — Trial count, TREX_NUM_MC for smoke runs.
#   compute_fdp()                — False discovery proportion.
#   compute_tpp()                — True positive proportion.
#   _make_trex_ctrl_from_dict()  — Build TRexControlParameter from flat dict.
#   _trial_worker()              — Top-level worker for ProcessPoolExecutor.
#   _trial_worker_full_mmap()    — Worker variant: X written to per-trial mmap file.
#   run_mc_trex()                — Parallel MC runner (in-memory X).
#   run_mc_trex_full_mmap()      — Parallel MC runner (fully mmap X per trial).
#   print_solver_table()         — Formatted console output.
#   save_mc_results()            — Pandas CSV + plain-text results.
#
# Seed spaces (no clashes):
#   DGP (X, noise)    : trial_seed              — via dgp_gauss_snr(seed=trial_seed)
#   Support (variable): trial_seed + 500_000    — via make_support_random(seed=trial_seed)
#   Coefficients (rnd): trial_seed + 600_000    — via np.random.default_rng(trial_seed+600_000)
#   TRexSelector      : seed=-1 (HW entropy)    — required for valid MC FDR estimates
#
# ==============================================================================

import atexit
import multiprocessing
import sys
import os
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor

# ---------------------------------------------------------------------------
# Ensure this directory is on sys.path so that spawned workers (ProcessPool-
# Executor on macOS uses spawn) can import this module and its siblings.
# Python's spawn start method copies sys.path to child processes, so inserting
# here in the main process is sufficient for all workers.
# ---------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)
# Family folder: cell_cache.py (resumable grid cells), solver_filter.py
# (TREX_SOLVERS).
_FAMILY_DIR = os.path.dirname(os.path.dirname(_THIS_DIR))
if _FAMILY_DIR not in sys.path:
    sys.path.insert(0, _FAMILY_DIR)

import cell_cache
import solver_filter

from dgp_gauss_snr import dgp_gauss_snr          # noqa: E402 (after sys.path setup)
from support_generators import make_support_random  # noqa: E402


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
# The demos pass num_workers through to the MC runners, which wins over the env.
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
# The demos call the MC runners once per (solver, config) point, so a pool
# created per call re-pays (process spawn + `import trexselector`) every
# time: measured at ~0.5 s for 6 workers. One lazily-created pool is reused
# instead, keyed on max_workers so a demo's num_workers override still applies.
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
    """Return the shared pool, rebuilding it when the worker count changed or
    the current workers have handled _RECYCLE_AFTER_TASKS tasks.

    n_tasks is how many tasks the caller is about to submit; it only feeds the
    recycle counter, so callers that omit it still get a working pool.
    """
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
# SOLVERS_DEFAULT
# (mirrors make_default_solvers_to_test() in demo_trex_02_mc_sim_fixed_support.cpp)
# ==============================================================================
#
# All values are plain Python scalars or strings — unconditionally picklable
# for ProcessPoolExecutor. Solver names are mapped to SolverTypeForTRex enums
# inside the worker, never across the process boundary.
#
# Values mirror the C++ DemoSolverInfo list exactly: lambda2 and rho_afs default
# to 0.0, overridden only where C++ does — TENET (lambda2=0.1), TAFS_rho_0.3 /
# TAFS_rho_1.0 (rho_afs=0.3 / 1.0), TNCGMP_v1 (ncgmp_variant=1). lambda2 only
# affects the elastic-net solvers (TENET; a nonzero value would turn TLASSO into
# an elastic net), and rho_afs only affects TAFS — so the per-solver defaults
# must be 0.0 to match C++, not a blanket 0.1 / 0.3.
SOLVERS_DEFAULT = [
    dict(solver_name="TLARS",      name="TLARS",        lambda2=0.0, rho_afs=0.0, ncgmp_variant=0),
    dict(solver_name="TLASSO",     name="TLASSO",       lambda2=0.0, rho_afs=0.0, ncgmp_variant=0),
    dict(solver_name="TENET",      name="TENET",        lambda2=0.1, rho_afs=0.0, ncgmp_variant=0),
    dict(solver_name="TSTAGEWISE", name="TSTAGEWISE",   lambda2=0.0, rho_afs=0.0, ncgmp_variant=0),
    dict(solver_name="TSTEPWISE",  name="TSTEPWISE",    lambda2=0.0, rho_afs=0.0, ncgmp_variant=0),
    dict(solver_name="TOMP",       name="TOMP",         lambda2=0.0, rho_afs=0.0, ncgmp_variant=0),
    dict(solver_name="TGP",        name="TGP",          lambda2=0.0, rho_afs=0.0, ncgmp_variant=0),
    dict(solver_name="TACGP",      name="TACGP",        lambda2=0.0, rho_afs=0.0, ncgmp_variant=0),
    dict(solver_name="TMP",        name="TMP",          lambda2=0.0, rho_afs=0.0, ncgmp_variant=0),
    dict(solver_name="TAFS",       name="TAFS_rho_0.3", lambda2=0.0, rho_afs=0.3, ncgmp_variant=0),
    dict(solver_name="TAFS",       name="TAFS_rho_1.0", lambda2=0.0, rho_afs=1.0, ncgmp_variant=0),
    dict(solver_name="TNCGMP",     name="TNCGMP_v1",    lambda2=0.0, rho_afs=0.0, ncgmp_variant=1),
    dict(solver_name="TNCGMP",     name="TNCGMP_v0",    lambda2=0.0, rho_afs=0.0, ncgmp_variant=0),
    dict(solver_name="TOOLS",      name="TOOLS",        lambda2=0.0, rho_afs=0.0, ncgmp_variant=0),
]


def selected_solvers(solvers):
    """The solvers that TREX_SOLVERS selects (all without a selection).

    Mirrors selected_solvers() in trex_sim_utils.hpp. A selection run writes
    its tables with the suffix solver_filter.file_suffix().
    """
    return [s for s in solvers if solver_filter.selected(s["name"])]


def resolve_num_mc(default):
    """Monte Carlo trials of a demo: TREX_NUM_MC when set (smoke runs), else default.

    Mirrors num_mc() in trex_sim_utils.hpp. TREX_NUM_MC enters the cell-cache
    fingerprint, so a smoke run never restores the cells of a full run.
    """
    try:
        n = int(os.environ.get("TREX_NUM_MC", ""))
    except ValueError:
        return default
    return n if n > 0 else default


# ==============================================================================
# compute_fdp / compute_tpp
# ==============================================================================

def compute_fdp(selected, true_support):
    """
    False Discovery Proportion: |FP| / max(|selected|, 1).

    Parameters
    ----------
    selected     : iterable of int  0-based selected indices.
    true_support : iterable of int  0-based true active indices.

    Returns
    -------
    float in [0, 1].
    """
    sel_set  = set(selected)
    true_set = set(true_support)
    n_sel = len(sel_set)
    if n_sel == 0:
        return 0.0
    return len(sel_set - true_set) / n_sel


def compute_tpp(selected, true_support):
    """
    True Positive Proportion: |TP| / |true_support|.

    Parameters
    ----------
    selected     : iterable of int  0-based selected indices.
    true_support : iterable of int  0-based true active indices.

    Returns
    -------
    float in [0, 1].
    """
    sel_set  = set(selected)
    true_set = set(true_support)
    if len(true_set) == 0:
        return 0.0
    return len(sel_set & true_set) / len(true_set)


# ==============================================================================
# _make_trex_ctrl_from_dict
# ==============================================================================
#
# Lazy-initialised string → enum maps.  Populated on first use inside a process
# so that the module can be imported without trexselector being available.
#
_SOLVER_MAP = None
_LLOOP_MAP  = None


def _get_solver_map():
    global _SOLVER_MAP
    if _SOLVER_MAP is None:
        import trexselector as trex
        _SOLVER_MAP = {
            "TLARS":      trex.SolverTypeForTRex.TLARS,
            "TLASSO":     trex.SolverTypeForTRex.TLASSO,
            "TENET":      trex.SolverTypeForTRex.TENET,
            "TSTEPWISE":  trex.SolverTypeForTRex.TSTEPWISE,
            "TSTAGEWISE": trex.SolverTypeForTRex.TSTAGEWISE,
            "TOMP":       trex.SolverTypeForTRex.TOMP,
            "TGP":        trex.SolverTypeForTRex.TGP,
            "TACGP":      trex.SolverTypeForTRex.TACGP,
            "TMP":        trex.SolverTypeForTRex.TMP,
            "TNCGMP":     trex.SolverTypeForTRex.TNCGMP,
            "TOOLS":      trex.SolverTypeForTRex.TOOLS,
            "TAFS":       trex.SolverTypeForTRex.TAFS,
        }
    return _SOLVER_MAP


def _get_lloop_map():
    global _LLOOP_MAP
    if _LLOOP_MAP is None:
        import trexselector as trex
        _LLOOP_MAP = {
            "STANDARD":             trex.LLoopStrategy.STANDARD,
            "HCONCAT":              trex.LLoopStrategy.HCONCAT,
            "PERMUTATION":          trex.LLoopStrategy.PERMUTATION,
            "PERMUTATION_SEEDED": trex.LLoopStrategy.PERMUTATION_SEEDED,
            "SEEDED":             trex.LLoopStrategy.SEEDED,
            "SKIPL":                trex.LLoopStrategy.SKIPL,
        }
    return _LLOOP_MAP


def _make_dummy_distribution(spec):
    """
    Build a trex.DummyDistribution from a plain (picklable) spec dict.

    spec: {"type": <name>, <optional distribution-specific parameters>}
    Types and parameters mirror R's trex_dummy_distribution():
        Normal, Rademacher, Mammen                       — no parameters
        Uniform(a), StudentT(df), UniformSphere(dim),
        ConstrainedSparseRademacher(s), Triangle(a, b, c),
        Laplace/Gumbel/Logistic(location, scale)
    Omitted parameters use the binding's unit-variance defaults.
    """
    import trexselector as trex
    D = trex.DummyDistribution
    t = spec["type"]
    if t == "Normal":
        return D.normal()
    if t == "Uniform":
        return D.uniform(**{k: spec[k] for k in ("a",) if k in spec})
    if t == "Rademacher":
        return D.rademacher()
    if t == "StudentT":
        return D.student_t(**{k: spec[k] for k in ("df",) if k in spec})
    if t == "Laplace":
        return D.laplace(**{k: spec[k] for k in ("location", "scale") if k in spec})
    if t == "Gumbel":
        return D.gumbel(**{k: spec[k] for k in ("location", "scale") if k in spec})
    if t == "Triangle":
        return D.triangle(**{k: spec[k] for k in ("a", "b", "c") if k in spec})
    if t == "UniformSphere":
        return D.uniform_sphere(**{k: spec[k] for k in ("dim",) if k in spec})
    if t == "Mammen":
        return D.mammen()
    if t == "ConstrainedSparseRademacher":
        return D.constrained_sparse_rademacher(spec["s"])
    if t == "Logistic":
        return D.logistic(**{k: spec[k] for k in ("location", "scale") if k in spec})
    raise ValueError(f"Unknown dummy distribution type: {t!r}")


def _make_trex_ctrl_from_dict(d):
    """
    Build a TRexControlParameter from a flat dict.

    DIVERGENCE-SAFE by construction: it starts from ``trex.TRexControlParameter()``
    — whose field defaults are *identical* to the C++ ``TRexControlParameter``
    struct (verified field-by-field) — and overrides a field ONLY when the dict
    supplies a non-None value. A missing or None key therefore keeps the
    C++/binding default; the helper never substitutes a Python-specific fallback
    that could silently diverge from the C++ reference.

    In particular, a missing/None ``tloop_stagnation_stop`` preserves the
    tri-state AUTO (``std::nullopt``) that the C++ demos 04/05 rely on (resolved
    per solver family in the selector) — the previous ``bool(...)`` coercion
    could not represent AUTO and forced it to False.

    Called inside worker processes — the object is never sent across a process
    boundary (only the flat dict is pickled).

    Keys recognised (all optional except solver_name):
        solver_name, lambda2, rho_afs, ncgmp_variant,
        K, max_dummy_multiplier, use_max_T_stop, opt_threshold,
        lloop_strategy, tloop_stagnation_stop, tloop_max_stagnant_steps,
        parallel_rnd_experiments, max_inner_threads, use_memory_mapping,
        dummy_distribution (a spec dict for _make_dummy_distribution).
    """
    import trexselector as trex
    solver_map = _get_solver_map()
    lloop_map  = _get_lloop_map()

    ctrl = trex.TRexControlParameter()   # defaults == C++ struct defaults

    # (dict key, ctrl attribute, cast) — applied only when present and not None.
    scalar_fields = (
        ("K",                        "K",                        int),
        ("max_dummy_multiplier",     "max_dummy_multiplier",     int),
        ("use_max_T_stop",           "use_max_T_stop",           bool),
        ("opt_threshold",            "opt_threshold",            float),
        ("tloop_stagnation_stop",    "tloop_stagnation_stop",    bool),
        ("tloop_max_stagnant_steps", "tloop_max_stagnant_steps", int),
        ("parallel_rnd_experiments", "parallel_rnd_experiments", bool),
        ("max_inner_threads",        "max_inner_threads",        int),
        ("use_memory_mapping",       "use_memory_mapping",       bool),
    )
    for key, attr, cast in scalar_fields:
        if d.get(key) is not None:
            setattr(ctrl, attr, cast(d[key]))

    if d.get("lloop_strategy") is not None:
        ctrl.lloop_strategy = lloop_map[d["lloop_strategy"]]
    if d.get("dummy_distribution") is not None:
        ctrl.dummy_distribution = _make_dummy_distribution(d["dummy_distribution"])

    ctrl.solver_type = solver_map[d["solver_name"]]   # required
    for key, cast in (("lambda2", float), ("rho_afs", float),
                      ("ncgmp_variant", int)):
        if d.get(key) is not None:
            setattr(ctrl.solver_params, key, cast(d[key]))
    return ctrl


# ==============================================================================
# _trial_worker
# ==============================================================================

def _trial_worker(args):
    """
    Top-level (module-level) worker for ProcessPoolExecutor.

    Must remain a module-level function — do NOT nest inside another function.
    Pickle resolves it as ('trex_sim_common', '_trial_worker'); spawned workers
    import trex_sim_common to find it (the sys.path insert above ensures this).

    args keys (all plain Python scalars, lists, or None):
        n, p, s, snr, amplitude, fixed_support (list or None), rnd_coef,
        mc, trial_seed, tFDR, track_L_T, seeded_selector,
        solver_name, lambda2, rho_afs, ncgmp_variant,
        K, max_dummy_multiplier, use_max_T_stop, opt_threshold,
        tloop_stagnation_stop, tloop_max_stagnant_steps, lloop_strategy,
        parallel_rnd_experiments, use_memory_mapping.
    """
    trial_seed = args["trial_seed"]
    n, p, s    = args["n"], args["p"], args["s"]
    snr        = args["snr"]
    amplitude  = args.get("amplitude", 1.0)
    fixed_supp = args.get("fixed_support")   # None → variable support per trial
    rnd_coef   = args.get("rnd_coef", False)
    tFDR       = args["tFDR"]
    track_L_T  = args.get("track_L_T", False)
    seeded     = args.get("seeded_selector", False)

    import trexselector as trex

    # Resolve support
    if fixed_supp is None:
        support = make_support_random(s, p, seed=trial_seed)
    else:
        support = np.asarray(fixed_supp, dtype=np.intp)

    # Resolve coefficients
    if rnd_coef:
        # Isolated seed stream (mirrors C++ rng_coef(seed + 600000u) and R's
        # set.seed(trial_seed + 600000L)).  Disjoint from DGP and support streams.
        coef_rng = np.random.default_rng(int(trial_seed) + 600_000)
        coefs    = coef_rng.standard_normal(s)
    else:
        coefs = np.full(s, float(amplitude))

    dat  = dgp_gauss_snr(n, p, support, coefs=coefs, snr=snr, seed=trial_seed)
    ctrl = _make_trex_ctrl_from_dict(args)

    # TRexSelector expects Fortran-contiguous (column-major) X
    sel = trex.TRexSelector(
        np.asfortranarray(dat["X"]),
        dat["y"],
        tFDR=tFDR,
        trex_control=ctrl,
        # Hardware entropy per trial by default; a fixed integer seed shared by all
        # trials would correlate the dummies across trials (mirrors R's seed=-1L).
        # seeded_selector uses the trial seed instead, distinct per trial and
        # reproducible (mirrors the C++ seeded_selector flag).
        seed=int(trial_seed) if seeded else -1,
        verbose=False,
    )
    sel.select()

    support_list = support.tolist()
    fdp = compute_fdp(sel.selected_indices, support_list)
    tpp = compute_tpp(sel.selected_indices, support_list)

    out = {"fdp": fdp, "tpp": tpp}
    if track_L_T:
        out["L"] = sel.L
        out["T"] = sel.T_stop
    return out


# ==============================================================================
# run_mc_trex
# ==============================================================================

def run_mc_trex(
    n,
    p,
    s,
    snr,
    solver_desc,
    base_ctrl_dict,
    tFDR,
    base_seed,
    num_MC,
    max_workers,
    fixed_support=None,
    amplitude=1.0,
    rnd_coef=False,
    track_L_T=False,
    label="",
    seeded_selector=False,
):
    """
    Parallel MC runner for classical T-Rex (ProcessPoolExecutor).

    Runs `num_MC` independent trials. Each trial:
      1. Draws support (fixed_support, or per-trial via make_support_random).
      2. Generates data via dgp_gauss_snr.
      3. Runs TRexSelector with seed=-1 (hardware entropy per trial), or with
         the trial seed when seeded_selector is True.

    Parameters
    ----------
    n, p, s        : int    Dimensions and number of active predictors.
    snr            : float  Signal-to-noise ratio.
    solver_desc    : dict   One entry from SOLVERS_DEFAULT (or compatible dict
                            with keys solver_name, lambda2, rho_afs, ncgmp_variant).
    base_ctrl_dict : dict   Shared TRex control parameters (K, lloop_strategy, ...).
                            Keys are merged with solver_desc into the flat args dict.
    tFDR           : float  Target FDR level.
    base_seed      : int    Base RNG seed; trial mc uses base_seed + mc - 1.
    num_MC         : int    Number of Monte Carlo trials.
    max_workers    : int    Number of parallel worker processes.
    fixed_support  : np.ndarray or None
        0-based fixed support array shared across all trials.
        Pass None to redraw support per trial via make_support_random.
    amplitude      : float  Signal amplitude for constant-coefficient trials.
    rnd_coef       : bool   If True, draw per-trial coefficients from N(0,1).
    track_L_T      : bool   If True, also record L and T_stop per trial.
    label          : str    Progress label printed to console.
    seeded_selector: bool   If True, seed the selector with the trial seed
                            (reproducible cells; mirrors the C++ flag).

    Returns
    -------
    dict with keys: mean_FDR, mean_TPR, sd_FDR, sd_TPR,
    and (if track_L_T) mean_L, mean_T.
    """

    # Convert fixed_support to a plain list so it is unconditionally picklable
    fixed_supp_list = fixed_support.tolist() if fixed_support is not None else None

    # Build one flat args dict per trial (all values are picklable scalars / lists)
    args_list = [
        {
            # Data generation
            "n":             n,
            "p":             p,
            "s":             s,
            "snr":           snr,
            "amplitude":     amplitude,
            "fixed_support": fixed_supp_list,
            "rnd_coef":      rnd_coef,
            # Trial identity
            "mc":            mc,
            "trial_seed":    base_seed + mc - 1,
            # TRex
            "tFDR":          tFDR,
            "track_L_T":     track_L_T,
            # Only when set, so the cache keys of the unseeded demos stay unchanged
            **({"seeded_selector": True} if seeded_selector else {}),
            # Solver-specific params (from solver_desc)
            "solver_name":   solver_desc["solver_name"],
            # Defaults match the C++ DemoSolverInfo struct (0.0 / 0.0 / 0).
            "lambda2":       solver_desc.get("lambda2", 0.0),
            "rho_afs":       solver_desc.get("rho_afs", 0.0),
            "ncgmp_variant": solver_desc.get("ncgmp_variant", 0),
            # Shared control params (from base_ctrl_dict; no key conflicts with above)
            **base_ctrl_dict,
        }
        for mc in range(1, num_MC + 1)
    ]

    # One resumable grid cell (cell_cache): restored after a restart.
    def compute():
        if label:
            print(f"  {label} — Running {num_MC} MC trials ...")
        results = list(_get_executor(max_workers, len(args_list)).map(
            _trial_worker, args_list))

        fdps = np.array([r["fdp"] for r in results])
        tpps = np.array([r["tpp"] for r in results])

        out = {
            "mean_FDR": float(np.mean(fdps)),
            "mean_TPR": float(np.mean(tpps)),
            "sd_FDR":   float(np.std(fdps, ddof=1)),
            "sd_TPR":   float(np.std(tpps, ddof=1)),
        }
        if track_L_T:
            Ls = np.array([r["L"] for r in results], dtype=float)
            Ts = np.array([r["T"] for r in results], dtype=float)
            out["mean_L"] = float(np.mean(Ls))
            out["mean_T"] = float(np.mean(Ts))

        return out

    out = cell_cache.cached(f"trex|{label}|{cell_cache.digest(args_list)}", compute)
    if label:
        print(f"  {label} — done.  mean_TPR={out['mean_TPR']:.3f}  "
              f"mean_FDR={out['mean_FDR']:.3f}\n")
    return out


# ==============================================================================
# print_solver_table
# ==============================================================================

def print_solver_table(
    row_names,
    snr_values,
    fdr_mat,
    tpr_mat,
    avg_L_mat=None,
    avg_T_mat=None,
    num_MC=None,
    file=None,
):
    """
    Print an aligned row × SNR results table.

    Mirrors R .print_solver_table().

    Parameters
    ----------
    row_names  : list of str  Row identifiers (solver names or strategy names).
    snr_values : list of float
    fdr_mat    : np.ndarray, shape (n_rows, n_snr)
    tpr_mat    : np.ndarray, shape (n_rows, n_snr)
    avg_L_mat  : np.ndarray or None
    avg_T_mat  : np.ndarray or None
    num_MC     : int or None
    file       : file-like or None  If given, output is also written to this file.
    """
    # Row-label column grows to fit the longest name; 15 keeps the classic
    # layout for demos whose names all fit (mirrors trex_sim_utils.hpp).
    sw = max(15, max(len(nm) for nm in row_names) + 2)
    mw, nw, cw = 8, 5, 10
    n_snr = len(snr_values)

    def _out(line):
        print(line)
        if file is not None:
            file.write(line + "\n")

    if num_MC is not None:
        sep = "=" * 70
        _out(f"\n{sep}")
        _out(f"=== T-Rex Results (averaged over {num_MC} Monte Carlo runs) ===")
        _out(f"{sep}\n")

    hdr = f"{'Solver':<{sw}}{'Metric':<{mw}}{'SNR':>{nw}}"
    for snr in snr_values:
        hdr += f"{snr:>{cw}.1f}"
    _out(hdr)
    _out("-" * (sw + mw + nw + cw * n_snr))

    def _row(name, metric, vals, first_row):
        label = name if first_row else ""
        r = f"{label:<{sw}}{metric:<{mw}}{'':{nw}}"
        for v in vals:
            r += f"{v:>{cw}.4f}"
        return r

    for i, nm in enumerate(row_names):
        _out(_row(nm, "FDR",   fdr_mat[i], True))
        _out(_row(nm, "TPR",   tpr_mat[i], False))
        if avg_L_mat is not None:
            _out(_row(nm, "Avg L", avg_L_mat[i], False))
        if avg_T_mat is not None:
            _out(_row(nm, "Avg T", avg_T_mat[i], False))
        _out("")


# ==============================================================================
# save_mc_results
# ==============================================================================

def save_mc_results(
    out_dir,
    file_stem,
    num_MC,
    row_names,
    snr_values,
    fdr_mat,
    tpr_mat,
    avg_L_mat=None,
    avg_T_mat=None,
):
    """
    Write results table to a .txt file and a tidy long-format .csv file.

    Mirrors R .save_and_print_mc().

    Parameters
    ----------
    out_dir    : str  Output directory (created if absent).
    file_stem  : str  Base file name without extension.
    num_MC     : int  Number of MC trials.
    row_names  : list of str
    snr_values : list of float
    fdr_mat    : np.ndarray, shape (n_rows, n_snr)
    tpr_mat    : np.ndarray, shape (n_rows, n_snr)
    avg_L_mat  : np.ndarray or None
    avg_T_mat  : np.ndarray or None
    """
    os.makedirs(out_dir, exist_ok=True)

    # 1. Plain-text table (mirrors R sink() trick)
    txt_path = os.path.join(out_dir, f"{file_stem}.txt")
    with open(txt_path, "w") as fh:
        print_solver_table(
            row_names, snr_values, fdr_mat, tpr_mat,
            avg_L_mat, avg_T_mat, num_MC, file=fh,
        )
    print(f"[Info] Results saved to:     {txt_path}")

    # 2. Long-format CSV via pandas
    rows = []
    for i, nm in enumerate(row_names):
        for j, snr in enumerate(snr_values):
            snr_str = f"{snr:.6f}"
            rows.append(dict(solver=nm, metric="FDR",  snr=snr_str, value=fdr_mat[i, j]))
            rows.append(dict(solver=nm, metric="TPR",  snr=snr_str, value=tpr_mat[i, j]))
            if avg_L_mat is not None:
                rows.append(dict(solver=nm, metric="AvgL", snr=snr_str, value=avg_L_mat[i, j]))
            if avg_T_mat is not None:
                rows.append(dict(solver=nm, metric="AvgT", snr=snr_str, value=avg_T_mat[i, j]))

    csv_path = os.path.join(out_dir, f"{file_stem}.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"[Info] CSV results saved to: {csv_path}")


# ==============================================================================
# _trial_worker_full_mmap
# ==============================================================================

def _trial_worker_full_mmap(args):
    """
    Top-level (module-level) worker for ProcessPoolExecutor — fully mmap X variant.

    Mirrors C++ Demo D (demo_trex_07_mc_sim_mmap.cpp):
      - Generates X in memory.
      - Writes X to a per-trial temporary binary file via numpy_to_memmap().
      - Passes X_mmap.to_numpy() (a zero-copy mmap view) to TRexSelector.
      - Deletes the temporary file in a finally block — exception-safe RAII
        equivalent, mirrors C++ MmapFileGuard declared before SyntheticDataMapped.

    The MemoryMappedMatrix object is never pickled — it is created and destroyed
    entirely within this worker process.

    Args dict has the same keys as _trial_worker, with use_memory_mapping
    ignored here (the mmap X path is the meaningful mmap layer; internal D-mmap
    and solver serialization are also active because use_memory_mapping=True
    will be set in the ctrl dict by the caller).
    """
    import tempfile
    import trexselector as trex
    from trexselector.utils import numpy_to_memmap

    trial_seed = args["trial_seed"]
    n, p, s    = args["n"], args["p"], args["s"]
    snr        = args["snr"]
    amplitude  = args.get("amplitude", 1.0)
    fixed_supp = args.get("fixed_support")
    rnd_coef   = args.get("rnd_coef", False)
    tFDR       = args["tFDR"]
    track_L_T  = args.get("track_L_T", False)

    # Resolve support
    if fixed_supp is None:
        support = make_support_random(s, p, seed=trial_seed)
    else:
        support = np.asarray(fixed_supp, dtype=np.intp)

    # Resolve coefficients
    if rnd_coef:
        coef_rng = np.random.default_rng(int(trial_seed) + 600_000)
        coefs    = coef_rng.standard_normal(s)
    else:
        coefs = np.full(s, float(amplitude))

    dat = dgp_gauss_snr(n, p, support, coefs=coefs, snr=snr, seed=trial_seed)
    ctrl = _make_trex_ctrl_from_dict(args)

    # Create a unique per-trial temp file.
    # os.close(fd) releases the OS file descriptor immediately; the file
    # path remains valid and numpy_to_memmap will re-open it for writing.
    fd, mmap_path = tempfile.mkstemp(suffix=".bin")
    os.close(fd)

    try:
        # Write Fortran-order X to the mmap file and get a zero-copy view.
        # numpy_to_memmap calls np.asarray(data, dtype=np.float64) internally
        # so we pass a plain C-order array and let it handle the conversion.
        X_mmap = numpy_to_memmap(mmap_path, dat["X"])

        # X_mmap.to_numpy() returns a Fortran-contiguous float64 view backed
        # by the mmap buffer; TRexSelector.__init__ calls
        # np.asfortranarray(X, dtype=np.float64) which returns it unchanged.
        sel = trex.TRexSelector(
            X_mmap.to_numpy(),
            dat["y"],
            tFDR=tFDR,
            trex_control=ctrl,
            seed=-1,       # hardware entropy per trial — required for valid MC FDR estimates
            verbose=False,
        )
        sel.select()
    finally:
        # Exception-safe cleanup — mirrors C++ RAII MmapFileGuard dtor.
        # The mmap view (X_mmap) is local to this scope; the OS releases the
        # mmap mapping when this worker process exits or when the object is
        # garbage-collected, which always happens before the finally block
        # on CPython (deterministic ref-counting).
        try:
            os.unlink(mmap_path)
        except OSError:
            pass  # already deleted or never created — not a fatal error

    support_list = support.tolist()
    fdp = compute_fdp(sel.selected_indices, support_list)
    tpp = compute_tpp(sel.selected_indices, support_list)

    out = {"fdp": fdp, "tpp": tpp}
    if track_L_T:
        out["L"] = sel.L
        out["T"] = sel.T_stop
    return out


# ==============================================================================
# run_mc_trex_full_mmap
# ==============================================================================

def run_mc_trex_full_mmap(
    n,
    p,
    s,
    snr,
    solver_desc,
    base_ctrl_dict,
    tFDR,
    base_seed,
    num_MC,
    max_workers,
    fixed_support=None,
    amplitude=1.0,
    rnd_coef=False,
    track_L_T=False,
    label="",
):
    """
    Parallel MC runner — fully memory-mapped X per trial.

    Same interface as run_mc_trex(), but dispatches to _trial_worker_full_mmap
    so each worker writes X to a temporary mmap file and reads it back via
    X_mmap.to_numpy().

    Intended for Demo D (demo_trex_07_mc_sim_mmap.py) and mirrors
    trx_sim::run_mc_trials_trex_mmap() in C++.

    The caller should set use_memory_mapping=True in base_ctrl_dict to also
    enable internal D-mmap and solver serialization on top of the X mmap layer.
    """

    fixed_supp_list = fixed_support.tolist() if fixed_support is not None else None

    args_list = [
        {
            "n":             n,
            "p":             p,
            "s":             s,
            "snr":           snr,
            "amplitude":     amplitude,
            "fixed_support": fixed_supp_list,
            "rnd_coef":      rnd_coef,
            "mc":            mc,
            "trial_seed":    base_seed + mc - 1,
            "tFDR":          tFDR,
            "track_L_T":     track_L_T,
            "solver_name":   solver_desc["solver_name"],
            # Defaults match the C++ DemoSolverInfo struct (0.0 / 0.0 / 0).
            "lambda2":       solver_desc.get("lambda2", 0.0),
            "rho_afs":       solver_desc.get("rho_afs", 0.0),
            "ncgmp_variant": solver_desc.get("ncgmp_variant", 0),
            **base_ctrl_dict,
        }
        for mc in range(1, num_MC + 1)
    ]

    # One resumable grid cell (cell_cache): restored after a restart.
    def compute():
        if label:
            print(f"  {label} — Running {num_MC} MC trials (full-mmap X) ...")
        results = list(_get_executor(max_workers, len(args_list)).map(
            _trial_worker_full_mmap, args_list))

        fdps = np.array([r["fdp"] for r in results])
        tpps = np.array([r["tpp"] for r in results])

        out = {
            "mean_FDR": float(np.mean(fdps)),
            "mean_TPR": float(np.mean(tpps)),
            "sd_FDR":   float(np.std(fdps, ddof=1)),
            "sd_TPR":   float(np.std(tpps, ddof=1)),
        }
        if track_L_T:
            Ls = np.array([r["L"] for r in results], dtype=float)
            Ts = np.array([r["T"] for r in results], dtype=float)
            out["mean_L"] = float(np.mean(Ls))
            out["mean_T"] = float(np.mean(Ts))

        return out

    out = cell_cache.cached(f"trex_mmap|{label}|{cell_cache.digest(args_list)}", compute)
    if label:
        print(f"  {label} — done.  mean_TPR={out['mean_TPR']:.3f}  "
              f"mean_FDR={out['mean_FDR']:.3f}\n")
    return out
