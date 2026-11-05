# ==============================================================================
# da_sim_common.py
# ==============================================================================
#
# Shared infrastructure for the DA-TRex demos: support generators, the default
# solver set (TLARS / TAFS / TOMP with per-solver exchangeable tie-breaking),
# parallel Monte Carlo runners for DA-TRex and the base (no-DA) T-Rex, generic
# parameter sweeps, and the console / TXT / CSV result writers.
#
# Mirrors:
#  cpp/trex_selector_methods/ad/trex_da/trex_da_sim_common.hpp:
#
# Content:
#   default_solvers()                       <- default_solvers()
#   make_support_capped_spread()            <- capped_spread_support()
#   make_support_random()                   <- random_support() (seed + 500000)
#   make_support_one_per_block()            <- one_per_block_support()
#   run_mc_trials()                         <- run_mc_trials() / run_mc_trials_base()
#   run_param_sweep() / run_snr_sweep()     <- run_param_sweep() / run_snr_sweep()
#   save_and_print_grid_results()           <- save_and_print_grid_results()
#   save_and_print_2d_gap_rho_results()     <- save_and_print_2d_gap_rho_results()
#   save_and_print_2d_two_support_results() <- save_and_print_2d_two_support_results()
#
# Result files use the cpp naming pattern da_trex_mc_{scenario_tag}.{txt,csv}
# and the same tidy CSV schemas; they are written into each demo's
# simulation_results/data/ folder, alongside the simulation_results/plots/
# figures rendered from those CSVs by ../trex_da_plt_utils.py (same split as
# the cpp suite).
#
# Seed spaces (mirror cpp):
#   DGP (X, noise)    : trial_seed = base_seed_offset + mc
#   Random support    : trial_seed + 500000
#   Selector          : seed = -1 (hardware entropy; required for valid MC FDR)
#
# Support / group indices are 0-based (Python convention).
# ==============================================================================

import atexit
import multiprocessing
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

import trexselector as trex
from trexselector.ml_methods.clustering import LinkageMethod
from trexselector.utils import compute_fdp, compute_tpp

# Family folder: cell_cache.py (resumable grid cells).
_FAMILY_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _FAMILY_DIR not in sys.path:
    sys.path.insert(0, _FAMILY_DIR)
import cell_cache

# ------------------------------------------------------------------------------
# Ensure this directory is on sys.path so that spawned workers (ProcessPool-
# Executor uses spawn on macOS) can import this module and dgp_generators.
# spawn copies sys.path to child processes, so inserting here is sufficient.
# ------------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

# ------------------------------------------------------------------------------
# Worker budget  (identical helper in every suite's *_common / *_utils module)
#
# N_CORES MC worker processes, each running the selector serially (the library
# default), so a demo never exceeds N_CORES cores in total. The default
# cap matches the cpp demos' default of 8 threads + inner set(1), keeping
# runtimes comparable across the C++ / Python suites.
#
# Override per run, no code edit needed:
#   TREX_NUM_WORKERS=12   exactly 12 workers (clamped to the usable CPU count)
#   TREX_NUM_WORKERS=0    use every usable CPU
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

# DA method keys
_DA_METHOD = {"AR1": trex.DAMethod.AR1,
              "NN": trex.DAMethod.NN,
              "EQUI": trex.DAMethod.EQUI,
              "BT": trex.DAMethod.BT,
              "PRIOR_GROUPS": trex.DAMethod.PRIOR_GROUPS}

# Solver Keys
_SOLVER = {"TLARS": trex.SolverTypeForTRex.TLARS,
           "TAFS": trex.SolverTypeForTRex.TAFS,
           "TOMP": trex.SolverTypeForTRex.TOMP}

# Linkage Keys follow the cpp (Single / Complete / Average).
_LINKAGE = {"Single": LinkageMethod.Single,
            "Complete": LinkageMethod.Complete,
            "Average": LinkageMethod.Average}


# ==============================================================================
# Number formatting (mirrors cpp fmt_num)
# ==============================================================================

def fmt_num(v, max_decimals=4):
    """Fixed-point with at most max_decimals decimals, trailing zeros (and a
    trailing '.') trimmed: fmt_num(0.05) -> '0.05', fmt_num(2.0) -> '2'."""
    s = f"{v:.{max_decimals}f}"
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s


# ==============================================================================
# Default solver set (mirrors cpp default_solvers)
# ==============================================================================

def default_solvers(da_variant=""):
    """Default DA-TRex solver descriptors (plain dicts, picklable).

    exch_tie_alpha = 0.25 for the greedy solvers TAFS/TOMP: exchangeability-
    calibrated stochastic tie-breaking, required for DA FDR control at high
    correlation. Path solver TLARS stays 0.
    LARS-based solvers run with  stagnation control OFF; greedy solvers
    default it ON. TLARS_STAG is TLARS with the stagnation stop ON, the
    comparison row for TLARS' long T-loops (last, as in cpp). `da_variant` tags
    the DA method in the row label, e.g. "+AR1" -> "TREX-DA+AR1: TLARS".
    """
    p = "TREX-DA" + da_variant + ": "
    return [
        dict(solver="TLARS", name=p + "TLARS", rho_afs=0.0,
             use_stagnation=False, exch_tie_alpha=0.0),
        dict(solver="TAFS", name=p + "TAFS", rho_afs=0.3,
             use_stagnation=True, exch_tie_alpha=0.25),
        dict(solver="TOMP", name=p + "TOMP", rho_afs=0.0,
             use_stagnation=True, exch_tie_alpha=0.25),
        dict(solver="TLARS", name=p + "TLARS_STAG", rho_afs=0.0,
             use_stagnation=True, exch_tie_alpha=0.0),
    ]


def base_solver_name(name):
    """Derive the base (no-DA) comparison-row name from a DA row name:
    'TREX-DA[+variant]: TLARS' -> 'TREX: TLARS' (mirrors run_param_sweep)."""
    if name.startswith("TREX-DA"):
        colon = name.find(": ")
        return "TREX: " + (name[colon + 2:] if colon != -1 else name)
    return "TREX: " + name


# ==============================================================================
# Support generators (0-based; mirror cpp support policies)
# ==============================================================================

def make_support_capped_spread(s, p, max_gap=10):
    """Evenly spaced support with a capped gap: {0, gap, 2*gap, ...}."""
    gap = min(p // s, max_gap)
    return np.arange(s) * gap


def make_support_random(s, p, seed):
    """Uniformly random support without replacement, sorted ascending.
    Uses seed + 500000 for RNG isolation from the DGP noise seed (cpp parity)."""
    rng = np.random.default_rng(seed + 500000)
    return np.sort(rng.choice(p, size=s, replace=False))


def make_support_one_per_block(s, p, block_size=0):
    """One active predictor per block: the first index of each of the first
    min(s, p // block_size) blocks (block_size = 0 -> auto: p // s)."""
    if s <= 0 or p <= 0:
        return np.empty(0, dtype=np.intp)
    g = block_size if block_size > 0 else max(1, p // s)
    active_blocks = min(s, p // g)
    return np.arange(active_blocks) * g


# ==============================================================================
# Worker-side control-parameter builders
# ==============================================================================

def _make_trex_ctrl(args):
    """Build a TRexControlParameter from the flat args dict (mirrors cpp
    make_trex_control + the per-solver fields set by run_param_sweep)."""
    tc = trex.TRexControlParameter()
    tc.K = int(args["K"])
    tc.max_dummy_multiplier = 10
    tc.use_max_T_stop = True
    tc.dummy_distribution = trex.DummyDistribution.normal()
    tc.lloop_strategy = trex.LLoopStrategy.STANDARD
    tc.tloop_stagnation_stop = bool(args["use_stagnation"])
    tc.tloop_max_stagnant_steps = 7
    tc.use_memory_mapping = False
    tc.solver_type = _SOLVER[args["solver"]]
    tc.solver_params.rho_afs = float(args["rho_afs"])
    tc.solver_params.exch_tie_alpha = float(args["exch_tie_alpha"])
    return tc


def _make_da_ctrl(da_spec):
    """Build a TRexDAControlParameter from a picklable spec dict.

    Keys: method ("AR1" | "EQUI" | "NN" | "BT"), hc_linkage ("Single" |
    "Complete" | "Average"), prior_groups (list of label vectors; when present
    the method field is ignored by the selector), rho_thr_DA, cor_coef.
    """
    da = trex.TRexDAControlParameter()
    if da_spec.get("method") is not None:
        da.method = _DA_METHOD[da_spec["method"]]
    if da_spec.get("hc_linkage") is not None:
        da.hc_linkage = _LINKAGE[da_spec["hc_linkage"]]
    if da_spec.get("prior_groups") is not None:
        da.prior_groups = [list(map(int, g)) for g in da_spec["prior_groups"]]
    if da_spec.get("rho_thr_DA") is not None:
        da.rho_thr_DA = float(da_spec["rho_thr_DA"])
    if da_spec.get("cor_coef") is not None:
        da.cor_coef = float(da_spec["cor_coef"])
    return da


# ==============================================================================
# Monte Carlo worker + persistent executor
# ==============================================================================

def _mc_worker(args):
    """Top-level worker for ProcessPoolExecutor (must stay module-level so
    spawned workers can resolve it as ('da_sim_common', '_mc_worker')).

    args keys: trial_seed, n, p, dgp, dgp_kwargs, support (list | None), s,
    tFDR, K, solver, rho_afs, exch_tie_alpha, use_stagnation,
    da_spec (dict | None -> base T-Rex, no DA).
    """
    import importlib
    dg = importlib.import_module("dgp_generators")

    trial_seed = args["trial_seed"]
    n, p = args["n"], args["p"]

    supp = (np.asarray(args["support"], dtype=np.intp)
            if args["support"] is not None
            else make_support_random(args["s"], p, trial_seed))

    dat = getattr(dg, args["dgp"])(n, p, supp, seed=trial_seed,
                                   **args["dgp_kwargs"])

    tc = _make_trex_ctrl(args)
    if args["da_spec"] is not None:
        sel = trex.TRexDASelector(np.asfortranarray(dat["X"]), dat["y"],
                                 tFDR=args["tFDR"],
                                 da_control=_make_da_ctrl(args["da_spec"]),
                                 trex_control=tc, seed=-1, verbose=False)
    else:
        sel = trex.TRexSelector(np.asfortranarray(dat["X"]), dat["y"],
                               tFDR=args["tFDR"],
                               trex_control=tc, seed=-1, verbose=False)
    sel.select()

    selected = list(sel.selected_indices)
    true_support = dat["true_support"].tolist()
    return (compute_fdp(selected, true_support),
            compute_tpp(selected, true_support),
            float(sel.L), float(sel.T_stop))


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


def _get_executor(n_cores, n_tasks=0):
    """Lazily create one persistent worker pool, reused across grid points so
    the per-pool spawn + import cost is paid only once per demo run.

    Keyed on n_cores: a caller asking for a different worker count rebuilds the
    pool rather than silently getting the previously-created one.
    """
    global _EXECUTOR, _EXECUTOR_WORKERS, _ATEXIT_REGISTERED, _TASKS_SINCE_SPAWN
    if _EXECUTOR is not None and (_EXECUTOR_WORKERS != n_cores
                                  or _TASKS_SINCE_SPAWN >= _RECYCLE_AFTER_TASKS):
        _shutdown_executor()
    if _EXECUTOR is None:
        _EXECUTOR = ProcessPoolExecutor(
            max_workers=n_cores,
            mp_context=multiprocessing.get_context("spawn"))
        _EXECUTOR_WORKERS = n_cores
        _TASKS_SINCE_SPAWN = 0
        if not _ATEXIT_REGISTERED:
            atexit.register(_shutdown_executor)
            _ATEXIT_REGISTERED = True
    _TASKS_SINCE_SPAWN += max(0, int(n_tasks))
    return _EXECUTOR


# ==============================================================================
# Standard MC inner loop (mirrors cpp run_mc_trials / run_mc_trials_base)
# ==============================================================================

def run_mc_trials(num_MC, progress_label, n, p, dgp, dgp_kwargs, support, s,
                  tFDR, solver_desc, K, da_spec, base_seed_offset,
                  n_cores=N_CORES):
    """Run num_MC parallel trials of one (DGP, solver, DA) configuration.

    support: fixed 0-based index list/array, or None to redraw a random
    support per trial (seed = trial_seed + 500000). da_spec None runs the
    base (no-DA) T-Rex selector.

    Returns dict(avg_fdr, avg_tpr, avg_L, avg_T, sd_fdr, sd_tpr).
    """
    supp_list = (list(np.asarray(support).tolist())
                 if support is not None else None)
    tasks = [dict(trial_seed=int(base_seed_offset) + mc, n=n, p=p, dgp=dgp,
                  dgp_kwargs=dgp_kwargs, support=supp_list, s=s, tFDR=tFDR,
                  K=K, solver=solver_desc["solver"],
                  rho_afs=solver_desc["rho_afs"],
                  exch_tie_alpha=solver_desc["exch_tie_alpha"],
                  use_stagnation=solver_desc["use_stagnation"],
                  da_spec=da_spec)
             for mc in range(num_MC)]

    # One resumable grid cell (cell_cache): restored after a restart.
    def compute():
        print(f"  {progress_label} — Running {num_MC} MC trials ...", flush=True)

        if n_cores > 1:
            res = list(_get_executor(n_cores, len(tasks)).map(_mc_worker, tasks))
        else:
            res = [_mc_worker(t) for t in tasks]

        fdp = np.array([r[0] for r in res])
        tpp = np.array([r[1] for r in res])
        L = np.array([r[2] for r in res])
        T = np.array([r[3] for r in res])

        return dict(avg_fdr=float(fdp.mean()), avg_tpr=float(tpp.mean()),
                    avg_L=float(L.mean()), avg_T=float(T.mean()),
                    sd_fdr=float(fdp.std(ddof=1)) if num_MC > 1 else 0.0,
                    sd_tpr=float(tpp.std(ddof=1)) if num_MC > 1 else 0.0)

    out = cell_cache.cached(f"da|{progress_label}|{cell_cache.digest(tasks)}", compute)

    print(f"  {progress_label} — done. TPP={out['avg_tpr']:.3f}"
          f"  FDP={out['avg_fdr']:.3f}\n", flush=True)
    return out


# ==============================================================================
# Section header (mirrors cpp cdiagnostics::print_section_header)
# ==============================================================================

def print_section_header(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


# ==============================================================================
# Header-extra line wrapping (mirrors cpp wrap_header_extra)
# ==============================================================================

_HEADER_WIDTH = 78


def wrap_header_extra(text, width=_HEADER_WIDTH, indent="  "):
    """Wrap a header info line at runs of two-or-more spaces or at a single
    space that follows a comma; any other single space is inside a token."""
    tokens = []   # (token_text, following_separator)
    pos = 0
    while pos < len(text):
        brk = brk_end = None
        i = pos
        while i < len(text):
            if text[i] != " ":
                i += 1
                continue
            j = i
            while j < len(text) and text[j] == " ":
                j += 1
            if (j - i) >= 2 or (i > 0 and text[i - 1] == ","):
                brk, brk_end = i, j
                break
            i = j   # lone space inside a token -- keep scanning
        if brk is None:
            tokens.append((text[pos:], ""))
            break
        tokens.append((text[pos:brk], text[brk:brk_end]))
        pos = brk_end

    out, line = "", ""
    for k, (tok, _) in enumerate(tokens):
        if not tok:
            continue
        if not line:
            line = indent + tok
            continue
        sep = tokens[k - 1][1]
        if len(line) + len(sep) + len(tok) <= width:
            line += sep + tok
        else:
            out += line + "\n"
            line = indent + tok
    if line:
        out += line + "\n"
    return out


# ==============================================================================
# Save / print grid results (mirrors cpp save_and_print_grid_results)
# ==============================================================================

def save_and_print_grid_results(out_dir, scenario_tag, grid_label, grid_values,
                                num_MC, solver_names, fdr_map, tpr_map,
                                sd_fdr_map, sd_tpr_map, avg_L_map, avg_T_map,
                                header_extra=""):
    """Write one grid-sweep result table to console, TXT, and tidy CSV.

    The maps are dicts solver_name -> array over grid_values. TXT layout and
    the CSV schema `solver,metric,<GRID>,value` match the cpp suite.
    """
    os.makedirs(out_dir, exist_ok=True)
    txt_path = os.path.join(out_dir, f"da_trex_mc_{scenario_tag}.txt")
    csv_path = os.path.join(out_dir, f"da_trex_mc_{scenario_tag}.csv")

    lines = []

    def emit(text):
        lines.append(text)

    # Header
    emit("\n" + "=" * 78 + "\n"
         + f"  DA-TRex MC: {scenario_tag}  (MC={num_MC})\n")
    if header_extra:
        emit(wrap_header_extra(header_extra))
    emit("=" * 78 + "\n\n")

    indent_w, metric_w, col_w = 2, 23, 10
    ncols = len(grid_values)
    sep_w = indent_w + metric_w + col_w * ncols

    integral_grid = all(float(v) == np.floor(v) and abs(v) < 1e6
                        for v in grid_values)
    th = " " * indent_w + f"{grid_label:<{metric_w}}"
    for v in grid_values:
        th += f"{int(v):>{col_w}d}" if integral_grid else f"{v:>{col_w}.2f}"
    emit(th + "\n" + "-" * sep_w + "\n")

    def row(metric, data):
        r = " " * indent_w + f"{metric:<{metric_w}}"
        for v in data:
            r += f"{v:>{col_w}.4f}"
        return r + "\n"

    for nm in solver_names:
        emit("\n" + nm + "\n")
        emit(row("FDR", fdr_map[nm]))
        emit(row("sd_FDR", sd_fdr_map[nm]))
        emit(row("TPR", tpr_map[nm]))
        emit(row("sd_TPR", sd_tpr_map[nm]))
        if avg_L_map is not None and nm in avg_L_map:
            emit(row("Avg L", avg_L_map[nm]))
        if avg_T_map is not None and nm in avg_T_map:
            emit(row("Avg T", avg_T_map[nm]))
    emit("\n")
    emit("=" * 78 + "\n\n")

    text = "".join(lines)
    print(text, end="")
    with open(txt_path, "w") as fh:
        fh.write(text)
    print(f"[Info] TXT saved to: {txt_path}")

    # CSV -- tidy long format: solver,metric,<GRID>,value
    csv_col = grid_label.replace(" ", "_")
    rows = []
    for nm in solver_names:
        for i, gv in enumerate(grid_values):
            gv_s = f"{gv:.6f}"
            rows.append((nm, "FDR", gv_s, f"{fdr_map[nm][i]:.6f}"))
            rows.append((nm, "sd_FDR", gv_s, f"{sd_fdr_map[nm][i]:.6f}"))
            rows.append((nm, "TPR", gv_s, f"{tpr_map[nm][i]:.6f}"))
            rows.append((nm, "sd_TPR", gv_s, f"{sd_tpr_map[nm][i]:.6f}"))
            if avg_L_map is not None and nm in avg_L_map:
                rows.append((nm, "Avg_L", gv_s, f"{avg_L_map[nm][i]:.6f}"))
            if avg_T_map is not None and nm in avg_T_map:
                rows.append((nm, "Avg_T", gv_s, f"{avg_T_map[nm][i]:.6f}"))
    pd.DataFrame(rows, columns=["solver", "metric", csv_col, "value"]) \
        .to_csv(csv_path, index=False)
    print(f"[Info] CSV saved to: {csv_path}\n")


# ==============================================================================
# Generic parameter sweep (mirrors cpp run_param_sweep / run_snr_sweep)
# ==============================================================================

def run_param_sweep(out_dir, scenario_tag, param_label, grid_values, num_MC,
                    tFDR, base_seed, solvers, make_dgp, da_spec, K=20,
                    header_extra="", include_base_trex=False, n_cores=N_CORES):
    """Run an MC parameter sweep: for each solver x grid value, run num_MC
    trials; optionally re-run every solver without the DA correction as a
    'TREX: <solver>' comparison row.

    make_dgp(val) must return a picklable spec dict with keys
    n, p, dgp (generator name in dgp_generators), kwargs (DGP kwargs incl.
    amplitude / snr / rho / ...), support (fixed index list, or None for a
    per-trial random redraw), s (support size, used when support is None).
    """
    print_section_header("MC: " + scenario_tag)

    all_names = [sv["name"] for sv in solvers]
    fdr, tpr, sd_fdr, sd_tpr, avg_L, avg_T = ({} for _ in range(6))
    for nm in all_names:
        for m in (fdr, tpr, sd_fdr, sd_tpr, avg_L, avg_T):
            m[nm] = np.zeros(len(grid_values))

    def sweep_one(solver_desc, name, spec_da):
        for gi, val in enumerate(grid_values):
            d = make_dgp(val)
            res = run_mc_trials(
                num_MC, f"{param_label} = {val:.2f}",
                d["n"], d["p"], d["dgp"], d["kwargs"], d["support"], d["s"],
                tFDR, solver_desc, K, spec_da,
                base_seed + gi * 10000, n_cores)
            fdr[name][gi] = res["avg_fdr"]
            tpr[name][gi] = res["avg_tpr"]
            sd_fdr[name][gi] = res["sd_fdr"]
            sd_tpr[name][gi] = res["sd_tpr"]
            avg_L[name][gi] = res["avg_L"]
            avg_T[name][gi] = res["avg_T"]

    for sv in solvers:
        print(f"\n  Solver: {sv['name']}")
        sweep_one(sv, sv["name"], da_spec)

    if include_base_trex:
        for sv in solvers:
            nm = base_solver_name(sv["name"])
            all_names.append(nm)
            for m in (fdr, tpr, sd_fdr, sd_tpr, avg_L, avg_T):
                m[nm] = np.zeros(len(grid_values))
            print(f"\n  Solver: {nm} (no DA)")
            sweep_one(sv, nm, None)

    save_and_print_grid_results(out_dir, scenario_tag, param_label,
                                grid_values, num_MC, all_names,
                                fdr, tpr, sd_fdr, sd_tpr, avg_L, avg_T,
                                header_extra)


def run_snr_sweep(out_dir, scenario_tag, snr_values, num_MC, tFDR, base_seed,
                  solvers, make_dgp, da_spec, K=20, header_extra="",
                  include_base_trex=False, n_cores=N_CORES):
    """Convenience wrapper around run_param_sweep with param_label = 'SNR'."""
    run_param_sweep(out_dir, scenario_tag, "SNR", snr_values, num_MC, tFDR,
                    base_seed, solvers, make_dgp, da_spec, K, header_extra,
                    include_base_trex, n_cores)


# ==============================================================================
# 2D Gap x Rho saving utility (mirrors cpp save_and_print_2d_gap_rho_results)
# ==============================================================================

def _fmt_grid_val(v):
    """Integer string for whole numbers, else two decimals (cpp fmt_val)."""
    if float(v) == np.floor(v) and abs(v) < 1e6:
        return str(int(v))
    return f"{v:.2f}"


def save_and_print_2d_gap_rho_results(out_dir, scenario_tag, gap_grid, rho_grid,
                                      kappa_boundary, num_MC, solver_names,
                                      tpp_cs, fdp_cs, sd_tpp_cs, sd_fdp_cs,
                                      tpp_rand, fdp_rand, sd_tpp_rand,
                                      sd_fdp_rand, header_extra=""):
    """Write the 2D Gap x Rho sweep (CappedSpread matrices [n_rho x n_gap] plus
    Random vectors [n_rho]) to console, TXT, and CSV
    (`solver,metric,rho,support_type,gap,value`)."""
    os.makedirs(out_dir, exist_ok=True)
    txt_path = os.path.join(out_dir, f"da_trex_mc_{scenario_tag}.txt")
    csv_path = os.path.join(out_dir, f"da_trex_mc_{scenario_tag}.csv")

    n_gap = len(gap_grid)
    n_cols = n_gap + 1   # gap columns + Random
    rw, cw = 10, 9

    lines = []

    def emit(text):
        lines.append(text)

    def col_header():
        h = " " * rw
        for g in gap_grid:
            h += f"{'gap=' + str(g):>{cw}}"
        h += f"{'Random':>{cw}}\n"
        h += "  " + "-" * (rw + cw * n_cols) + "\n"
        return h

    def matrix_block(label, cs_mat, rand_vec):
        block = f"  {label}\n" + col_header()
        for ir, rho in enumerate(rho_grid):
            block += "  " + f"{'rho=' + format(rho, '.1f'):<{rw}}"
            for ig in range(n_gap):
                block += f"{cs_mat[ir, ig]:>{cw}.4f}"
            block += f"{rand_vec[ir]:>{cw}.4f}\n"
        return block + "\n"

    emit("\n" + "=" * 78 + "\n"
         + f"  DA-TRex MC: {scenario_tag}  (MC={num_MC})\n")
    if header_extra:
        emit(wrap_header_extra(header_extra))
    emit("=" * 78 + "\n\n")

    kb = "  DA+AR1 correction window: kappa = ceil(log(0.02) / log(rho))\n\n"
    kb += "  " + f"{'rho':<10}"
    for r in rho_grid:
        kb += f"{r:>6.1f}"
    kb += "\n  " + f"{'kappa':<10}"
    for k in kappa_boundary:
        kb += f"{k:>6d}"
    emit(kb + "\n\n")

    for nm in solver_names:
        emit("-" * 78 + "\n" + f"  Solver: {nm}\n" + "-" * 78 + "\n\n")
        emit(matrix_block("mean_TPP", tpp_cs[nm], tpp_rand[nm]))
        emit(matrix_block("sd_TPP", sd_tpp_cs[nm], sd_tpp_rand[nm]))
        emit(matrix_block("mean_FDP", fdp_cs[nm], fdp_rand[nm]))
        emit(matrix_block("sd_FDP", sd_fdp_cs[nm], sd_fdp_rand[nm]))

    emit("=" * 78 + "\n\n")

    text = "".join(lines)
    print(text, end="")
    with open(txt_path, "w") as fh:
        fh.write(text)
    print(f"[Info] TXT saved to: {txt_path}")

    rows = []
    for nm in solver_names:
        for ir, rho in enumerate(rho_grid):
            rho_s = f"{rho:.6f}"
            for ig, gap in enumerate(gap_grid):
                rows.append((nm, "mean_TPP", rho_s, "CappedSpread", gap,
                             f"{tpp_cs[nm][ir, ig]:.6f}"))
                rows.append((nm, "sd_TPP", rho_s, "CappedSpread", gap,
                             f"{sd_tpp_cs[nm][ir, ig]:.6f}"))
                rows.append((nm, "mean_FDP", rho_s, "CappedSpread", gap,
                             f"{fdp_cs[nm][ir, ig]:.6f}"))
                rows.append((nm, "sd_FDP", rho_s, "CappedSpread", gap,
                             f"{sd_fdp_cs[nm][ir, ig]:.6f}"))
            rows.append((nm, "mean_TPP", rho_s, "Random", "",
                         f"{tpp_rand[nm][ir]:.6f}"))
            rows.append((nm, "sd_TPP", rho_s, "Random", "",
                         f"{sd_tpp_rand[nm][ir]:.6f}"))
            rows.append((nm, "mean_FDP", rho_s, "Random", "",
                         f"{fdp_rand[nm][ir]:.6f}"))
            rows.append((nm, "sd_FDP", rho_s, "Random", "",
                         f"{sd_fdp_rand[nm][ir]:.6f}"))
    pd.DataFrame(rows, columns=["solver", "metric", "rho", "support_type",
                                "gap", "value"]).to_csv(csv_path, index=False)
    print(f"[Info] CSV saved to: {csv_path}\n")


# ==============================================================================
# 2D two-support saving utility (mirrors cpp save_and_print_2d_two_support_results)
# ==============================================================================

def save_and_print_2d_two_support_results(out_dir, scenario_tag, row_label,
                                          col_label, row_grid, col_grid, num_MC,
                                          solver_names, tpp_cs, fdp_cs,
                                          sd_tpp_cs, sd_fdp_cs, tpp_rand,
                                          fdp_rand, sd_tpp_rand, sd_fdp_rand,
                                          header_extra=""):
    """Write a 2D parameter sweep with full CappedSpread and Random matrices
    [n_row x n_col] per solver to console, TXT, and CSV
    (`solver,support_type,metric,<ROW>,<COL>,value`)."""
    os.makedirs(out_dir, exist_ok=True)
    txt_path = os.path.join(out_dir, f"da_trex_mc_{scenario_tag}.txt")
    csv_path = os.path.join(out_dir, f"da_trex_mc_{scenario_tag}.csv")

    n_row, n_col = len(row_grid), len(col_grid)
    rlw, cw = 12, 9

    lines = []

    def emit(text):
        lines.append(text)

    def col_header():
        h = "  " + f"{col_label:>{rlw}}"
        for cv in col_grid:
            h += f"{_fmt_grid_val(cv):>{cw}}"
        h += "\n" + "  " + "-" * (rlw + cw * n_col) + "\n"
        return h

    def matrix_block(label, mat):
        block = f"  {label}\n" + col_header()
        for ir in range(n_row):
            rl = f"{row_label}={_fmt_grid_val(row_grid[ir])}"
            block += "  " + f"{rl:<{rlw}}"
            for ic in range(n_col):
                block += f"{mat[ir, ic]:>{cw}.4f}"
            block += "\n"
        return block + "\n"

    emit("\n" + "=" * 78 + "\n"
         + f"  DA-TRex MC: {scenario_tag}  (MC={num_MC})\n")
    if header_extra:
        emit(wrap_header_extra(header_extra))
    emit("=" * 78 + "\n\n")

    for nm in solver_names:
        emit("-" * 78 + "\n" + f"  Solver: {nm}\n" + "-" * 78 + "\n\n")
        emit("  [CappedSpread]\n\n")
        emit(matrix_block("mean_TPP", tpp_cs[nm]))
        emit(matrix_block("sd_TPP", sd_tpp_cs[nm]))
        emit(matrix_block("mean_FDP", fdp_cs[nm]))
        emit(matrix_block("sd_FDP", sd_fdp_cs[nm]))
        emit("  [Random]\n\n")
        emit(matrix_block("mean_TPP", tpp_rand[nm]))
        emit(matrix_block("sd_TPP", sd_tpp_rand[nm]))
        emit(matrix_block("mean_FDP", fdp_rand[nm]))
        emit(matrix_block("sd_FDP", sd_fdp_rand[nm]))

    emit("=" * 78 + "\n\n")

    text = "".join(lines)
    print(text, end="")
    with open(txt_path, "w") as fh:
        fh.write(text)
    print(f"[Info] TXT saved to: {txt_path}")

    csv_row = row_label.replace(" ", "_")
    csv_col = col_label.replace(" ", "_")
    rows = []

    def write_block(nm, support_type, tpp_m, fdp_m, sd_tpp_m, sd_fdp_m):
        for ir, rv in enumerate(row_grid):
            for ic, cv in enumerate(col_grid):
                rv_s, cv_s = f"{rv:.6f}", f"{cv:.6f}"
                rows.append((nm, support_type, "mean_TPP", rv_s, cv_s,
                             f"{tpp_m[nm][ir, ic]:.6f}"))
                rows.append((nm, support_type, "sd_TPP", rv_s, cv_s,
                             f"{sd_tpp_m[nm][ir, ic]:.6f}"))
                rows.append((nm, support_type, "mean_FDP", rv_s, cv_s,
                             f"{fdp_m[nm][ir, ic]:.6f}"))
                rows.append((nm, support_type, "sd_FDP", rv_s, cv_s,
                             f"{sd_fdp_m[nm][ir, ic]:.6f}"))

    for nm in solver_names:
        write_block(nm, "CappedSpread", tpp_cs, fdp_cs, sd_tpp_cs, sd_fdp_cs)
        write_block(nm, "Random", tpp_rand, fdp_rand, sd_tpp_rand, sd_fdp_rand)
    pd.DataFrame(rows, columns=["solver", "support_type", "metric", csv_row,
                                csv_col, "value"]).to_csv(csv_path, index=False)
    print(f"[Info] CSV saved to: {csv_path}\n")
