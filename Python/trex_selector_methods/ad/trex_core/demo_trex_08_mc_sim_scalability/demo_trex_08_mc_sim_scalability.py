# ==============================================================================
# demo_trex_08_mc_sim_scalability.py
# ==============================================================================
#
# Scalability benchmark for the T-Rex Selector: runtime and memory usage as the
# problem scales, across three base solvers (TLARS, TOMP, TAFS).
#
# Mirrors:
#  cpp/trex_selector_methods/ad/trex_core/demo_trex_08_mc_sim_scalability.cpp.
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
#   * per T-iteration volume X + K*L*X: 126 GB at n=5000, p=150k and 50 GB at
#     n=300, p=1e6 (L = 1, 242 GB at L = 5), yet resident RSS stays bounded.
#   * A file named by TREX_STOP_FILE stops the run before the next trial
#     (exit code 75); finished trials stay cached and the next run resumes.
#   * SNR = 1 fixed (SNR is not a scaling axis; sweeping it only burns time),
#     demo-03 DGP (variable support s=10 redrawn per trial, unit coefficients,
#     Normal predictors/noise).
#   * num_MC = 10 SEQUENTIAL trials per (n, p, solver). Trials are NOT run in
#     parallel: this is a timing/memory benchmark, and concurrent trials would
#     (i) corrupt per-trial wall-clock, (ii) confound RSS, and (iii) multiply
#     the on-disk dummy backing. The K random experiments inside select() also
#     run one after another; OpenMP parallelizes the column loops inside each.
#
# Fully memory-mapped pipeline in every point: X is generated column-chunk by
# column-chunk into an mmap backing file (never materialized in RAM), and
# use_memory_mapping=True memory-maps the dummy matrices D and serializes solver
# warm-start state between T-loop iterations. Resident RSS therefore stays
# bounded while the volume runs to hundreds of GB — the out-of-core story.
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
# Self-cleanup: the process sweeps its own memory-mapped orphans
# (demo08_scalability_* backing files, trex_memmap_* dummy dirs and
# trex_warmstart_* checkpoint dirs in the temp dir) at startup and on exit /
# SIGINT / SIGTERM, so a cancelled run does not strand hundreds of GB.
# (SIGKILL and SIGBUS cannot be handled from Python; the next startup sweep
# reclaims them.)
#
# Expandable by design. Every trial is appended to a raw per-trial cache
# (<stem>_trials.csv, one row per (n, p, solver, mc) keyed by its seed). A run
# computes only the trials it is missing and derives the summary table/CSV from
# the cache, so:
#   * num_MC top-up  — run at num_MC=2 now, later num_MC=10 computes just the 8
#     new trials per cell and re-aggregates (no recompute of the first 2);
#   * grid growth    — append new n/p values later; existing cells are reused;
#   * trial-level resume — a crash/OOM mid-point keeps every finished trial.
# Per-trial seeds are a stable function of (n, p, mc) (grid-position independent),
# so reuse and reproducibility survive reordering or extending the grid.
#
# Results go to simulation_results/data/ as an aligned table (.txt) and a tidy
# summary CSV per block, both re-derived from the cache and re-saved after each
# (n, p) point.
#
# Run:
#   python demo_trex_08_mc_sim_scalability.py           # both blocks, num_MC=10
#   python demo_trex_08_mc_sim_scalability.py base      # base block only
#   python demo_trex_08_mc_sim_scalability.py large 2   # large-p block, num_MC=2
#   python demo_trex_08_mc_sim_scalability.py smoke     # tiny end-to-end check
#   python demo_trex_08_mc_sim_scalability.py smoke 3   # smoke grid, num_MC=3
#   TREX_SOLVERS=TLARS limits the computed solvers (cached ones still count).
#
# ==============================================================================

import sys
import os

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.dirname(_THIS_DIR)
for _p in (_THIS_DIR, _PARENT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import atexit  # noqa: E402
import csv  # noqa: E402
import ctypes  # noqa: E402
import glob  # noqa: E402
try:
    import resource  # noqa: E402  (POSIX only — see peak_rss_mib)
except ImportError:  # Windows has no resource module
    resource = None
import shutil  # noqa: E402
import signal  # noqa: E402
import subprocess  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import trexselector as trex  # noqa: E402
from trexselector.utils import (  # noqa: E402
    AccessMode, MemoryMappedMatrix, get_num_threads, set_num_threads,
)

from trex_helpers import make_dual_out  # noqa: E402
from trex_sim_common import _make_trex_ctrl_from_dict, compute_fdp, compute_tpp, resolve_workers  # noqa: E402
import solver_filter  # noqa: E402  (on sys.path through trex_sim_common)
from support_generators import make_support_random  # noqa: E402

# ==============================================================================
# Configuration
# ==============================================================================

_OUT_DIR = os.path.join(_THIS_DIR, "simulation_results", "data")

_S_TRUE      = 10      # cardinality of the true support (redrawn per trial)
_TFDR        = 0.1
_SNR         = 1.0     # fixed; SNR is not a scaling axis
_CHUNK_COLS  = 1000    # column-chunk size for the out-of-core X generation
_NUM_RND_EXP = 20      # K random experiments
_DUMMY_MULT  = 10      # base block max_dummy_multiplier
_LARGE_P_DUMMY_MULT = 5      # large-p block cap (bounds the dummy disk)
_DISK_RESERVE_GIB   = 50.0   # free disk left untouched by the pre-flight
_NUM_THREADS = resolve_workers()  # OpenMP threads inside select(), TREX_NUM_WORKERS
                       # (default 8) as in the C++ demo. Left unset, the library
                       # runs single-threaded, so the count is set explicitly
                       # (max_inner_threads) for a run comparable across languages.

# Base block: full n x p cross. Points whose volume X + K*L*X passes physical
# RAM are the out-of-core regime that only the memory-mapped pipeline reaches.
_N_VALUES   = [300, 1_000, 2_000, 5_000]
_P_VALUES   = [1_000, 10_000, 100_000, 150_000]
# Large-p block: few observations, up to a million predictors.
_LARGE_P_N_VALUES = [300]
_LARGE_P_P_VALUES = [100_000, 1_000_000]
_EXIT_STOPPED = 75   # exit code after a stop via TREX_STOP_FILE (EX_TEMPFAIL)
_NUM_MC     = 10

# Tiny end-to-end smoke grid (pipeline verification only); the second point
# exercises the large-p dummy cap.
_SMOKE_GRID = [(150, 500, _DUMMY_MULT), (200, 1_000, _LARGE_P_DUMMY_MULT)]
_SMOKE_MC   = 2


def _benchmark_grid(num_MC=_NUM_MC):
    """(n, p, num_MC, dummy_mult) points of the base block (full n x p cross)."""
    return [(n, p, num_MC, _DUMMY_MULT) for n in _N_VALUES for p in _P_VALUES]


def _large_p_grid(num_MC=_NUM_MC):
    """(n, p, num_MC, dummy_mult) points of the large-p block, ordered by
    volume (p-outer) so the cheapest points finish first."""
    return [(n, p, num_MC, _LARGE_P_DUMMY_MULT)
            for p in _LARGE_P_P_VALUES for n in _LARGE_P_N_VALUES]


def _smoke_grid(num_MC=_SMOKE_MC):
    return [(n, p, num_MC, dm) for (n, p, dm) in _SMOKE_GRID]


# Base solvers to compare (rho_afs only used by TAFS, which requires rho in
# (0, 1]; mirrors the C++ DemoSolverInfo list — TOMP default, TAFS rho_afs=0.3).
_SOLVERS = [
    dict(solver_name="TLARS", name="TLARS", lambda2=0.0, rho_afs=0.0),
    dict(solver_name="TOMP",  name="TOMP",  lambda2=0.0, rho_afs=0.0),
    dict(solver_name="TAFS",  name="TAFS",  lambda2=0.0, rho_afs=0.3),
]


def _physical_ram_gib():
    """Physical RAM in GiB (nan if unavailable) — used to flag out-of-core points."""
    try:
        if sys.platform == "darwin":
            out = subprocess.run(["sysctl", "-n", "hw.memsize"],
                                 capture_output=True, text=True, check=True)
            return int(out.stdout.strip()) / (1024.0 ** 3)
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemTotal:"):
                    return float(line.split()[1]) / (1024.0 ** 2)   # kB -> GiB
    except Exception:
        pass
    return float("nan")


_RAM_GIB = _physical_ram_gib()


def _make_scalability_ctrl_dict(solver_desc, dummy_mult):
    """T-Rex control (base config of demos 02/03/07) for one grid point.

    EVERY control option is set explicitly and identically to the C++ demo's
    make_scalability_trex_ctrl() — no reliance on the Python helper's defaults
    (which differ from the C++ struct's, e.g. opt_threshold 0.02 vs 0.75 and
    parallel_rnd_experiments True vs false). Leaving those implicit is exactly
    what made the two demos incomparable, so they are pinned here.
    """
    return dict(
        solver_name              = solver_desc["solver_name"],
        lambda2                  = solver_desc["lambda2"],
        rho_afs                  = solver_desc["rho_afs"],
        K                        = _NUM_RND_EXP,
        max_dummy_multiplier     = dummy_mult,
        use_max_T_stop           = True,
        opt_threshold            = 0.75,
        dummy_distribution       = {"type": "Normal"},
        lloop_strategy           = "STANDARD",
        tloop_stagnation_stop    = True,
        tloop_max_stagnant_steps = 5,
        # Sequential random experiments (mirrors C++ + R parallel_rnd_experiments = FALSE): one
        # experiment working set in RAM at a time (OpenMP threads parallelize
        # the column loops inside it), not K concurrent ones — essential
        # out-of-core, and the setting the other trex demos use. Python's
        # default here is True.
        parallel_rnd_experiments = False,
        # TREX_NUM_WORKERS threads run the column loops inside each experiment:
        # normalization, dummy generation and the solver's passes.
        max_inner_threads        = _NUM_THREADS,
        use_memory_mapping       = True,          # X mmap + D mmap + solver serialization
    )


_BYTES_PER_GIB = 1024.0 ** 3


def _volume_gib(n, p, L):
    """Data volume one T-iteration touches in GiB: X plus the K memory-mapped
    dummy files of L * p columns each (float64)."""
    return n * p * 8.0 * (1.0 + _NUM_RND_EXP * L) / _BYTES_PER_GIB


def _worst_case_disk_gib(n, p, dummy_mult):
    """Worst-case disk need of one trial in GiB: X, the K dummy files at
    L = dummy_mult, and K solver checkpoints (at most 32 bytes per column of
    [X|D])."""
    x_bytes = n * p * 8.0
    dummies = _NUM_RND_EXP * n * p * dummy_mult * 8.0
    checkpoints = _NUM_RND_EXP * 32.0 * p * (1 + dummy_mult)
    return (x_bytes + dummies + checkpoints) / _BYTES_PER_GIB


def _free_temp_disk_gib():
    """Free space of the temp dir in GiB (nan if unavailable)."""
    try:
        return shutil.disk_usage(tempfile.gettempdir()).free / _BYTES_PER_GIB
    except OSError:
        return float("nan")


_STOPPED = False


def _stop_requested():
    """True once the file named by TREX_STOP_FILE exists (sticky).

    Checked before every trial, so a stop never cuts a trial short.
    """
    global _STOPPED
    if not _STOPPED:
        path = os.environ.get("TREX_STOP_FILE", "")
        _STOPPED = bool(path) and os.path.exists(path)
    return _STOPPED


# ==============================================================================
# Expandable per-trial cache (top-up num_MC / grid; trial-level resume)
# ==============================================================================

# Deterministic, grid-position-independent, 32-bit-safe seed. Distinct (n, p)
# already yield different data (the X draw has a different shape), so we only need
# determinism in (n, p, mc), not cross-cell seed uniqueness. Matches the C++
# demo's _trial_seed so the two languages seed identically.
_SEED_MOD = 2_147_483_647  # 2**31 - 1


def _trial_seed(n, p, mc):
    return int((24 + n * 100003 + p * 13 + mc) % _SEED_MOD)


# Raw per-trial store: one row per (n, p, solver, mc). The summary .txt/.csv are
# derived from it, so a later run tops up missing trials without recomputation.
_RAW_FIELDS = ["n", "p", "solver", "mc", "trial_seed", "dgp_s", "select_s",
               "rss_mib", "fp_mib", "peak_rss_mib", "data_mib",
               "fdp", "tpp", "L", "T"]


def _raw_cache_path(file_stem, persist_cache=True):
    # A non-persistent cache lives in the temp dir and is swept with the
    # orphans, so every smoke run computes fresh.
    if persist_cache:
        return os.path.join(_OUT_DIR, f"{file_stem}_trials.csv")
    return os.path.join(tempfile.gettempdir(),
                        f"demo08_scalability_{file_stem}_trials.csv")


def _load_trial_cache(raw_path):
    """Load the raw per-trial store into {(n, p, solver): {mc: row-dict}}."""
    cache = {}
    if not os.path.exists(raw_path):
        return cache
    with open(raw_path, newline="") as fh:
        for r in csv.DictReader(fh):
            key = (int(r["n"]), int(r["p"]), r["solver"])
            cache.setdefault(key, {})[int(r["mc"])] = r
    return cache


def _append_trial_row(raw_path, row):
    """Append one trial to the raw store (write header if the file is new)."""
    write_header = not os.path.exists(raw_path)
    with open(raw_path, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=_RAW_FIELDS)
        if write_header:
            w.writeheader()
        w.writerow(row)


def _aggregate_cell(cell_cache, n, p):
    """Aggregate all cached trials of one (n, p, solver) cell into a summary dict.

    Aggregates over every cached trial (monotone: more trials -> a better
    estimate), so re-running at a lower num_MC never discards work.
    """
    rows = [cell_cache[mc] for mc in sorted(cell_cache)]
    fdp    = [float(r["fdp"]) for r in rows]
    tpp    = [float(r["tpp"]) for r in rows]
    L      = [float(r["L"]) for r in rows]
    T      = [float(r["T"]) for r in rows]
    dgp    = [float(r["dgp_s"]) for r in rows]
    sel    = [float(r["select_s"]) for r in rows]
    titer  = [s / max(t, 1.0) for s, t in zip(sel, T)]
    total  = [d + s for d, s in zip(dgp, sel)]
    return dict(
        vol_gib      = _volume_gib(n, p, max(float(np.mean(L)), 1.0)),
        sel_mean_s   = float(np.mean(sel)),
        sel_med_s    = float(np.median(sel)),
        sel_min_s    = float(np.min(sel)),
        sel_max_s    = float(np.max(sel)),
        avg_dgp_s    = float(np.mean(dgp)),
        avg_titer_s  = float(np.mean(titer)),
        avg_total_s  = float(np.mean(total)),
        avg_select_s = float(np.mean(sel)),
        peak_rss_mib = float(np.max([float(r["peak_rss_mib"]) for r in rows])),
        max_rss_mib  = float(np.max([float(r["rss_mib"]) for r in rows])),
        max_fp_mib   = float(np.max([float(r["fp_mib"]) for r in rows])),
        data_mib     = float(rows[-1]["data_mib"]),
        avg_fdr      = float(np.mean(fdp)),
        avg_tpr      = float(np.mean(tpp)),
        avg_L        = float(np.mean(L)),
        avg_T        = float(np.mean(T)),
        num_mc       = len(rows),
    )


# ==============================================================================
# Self-cleanup of memory-mapped orphans
# ==============================================================================

_TMPDIR = tempfile.gettempdir()
_ORPHAN_GLOBS = ("demo08_scalability_*", "trex_memmap_*", "trex_warmstart_*")


def sweep_mmap_orphans(reason=""):
    """Remove this demo's stray mmap backing in the temp dir (best-effort).

    Targets `demo08_scalability_*` X backing files, `trex_memmap_*` dummy dirs
    and `trex_warmstart_*` checkpoint dirs left behind by a killed/cancelled
    run. Assumes SOLO execution (this is
    a sequential benchmark); it does not attempt to spare a concurrently running
    instance's files. Returns bytes freed.
    """
    freed = 0
    for pattern in _ORPHAN_GLOBS:
        for path in glob.glob(os.path.join(_TMPDIR, pattern)):
            try:
                if os.path.isdir(path):
                    for root, _dirs, files in os.walk(path):
                        for fn in files:
                            try:
                                freed += os.path.getsize(os.path.join(root, fn))
                            except OSError:
                                pass
                    shutil.rmtree(path, ignore_errors=True)
                else:
                    try:
                        freed += os.path.getsize(path)
                    except OSError:
                        pass
                    os.unlink(path)
            except OSError:
                pass
    if freed > 0:
        print(f"[cleanup] swept {freed / (1024.0 ** 3):.1f} GiB of mmap orphans"
              f"{f' ({reason})' if reason else ''}", flush=True)
    return freed


def _install_cleanup_handlers():
    """Sweep orphans on normal exit and on SIGINT/SIGTERM (not SIGKILL)."""
    atexit.register(lambda: sweep_mmap_orphans("atexit"))

    def _handler(signum, _frame):
        sweep_mmap_orphans(f"signal {signum}")
        # Restore default and re-raise so the exit status reflects the signal.
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)

    for _sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(_sig, _handler)
        except (ValueError, OSError):
            pass  # e.g. not in the main thread


# ==============================================================================
# Memory profiling hooks
# ==============================================================================

class _PROCESS_MEMORY_COUNTERS(ctypes.Structure):
    """Windows PROCESS_MEMORY_COUNTERS (psapi.h) — working set, current + peak."""
    _fields_ = [
        ("cb",                         ctypes.c_uint32),
        ("PageFaultCount",             ctypes.c_uint32),
        ("PeakWorkingSetSize",         ctypes.c_size_t),
        ("WorkingSetSize",             ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage",    ctypes.c_size_t),
        ("QuotaPagedPoolUsage",        ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage",     ctypes.c_size_t),
        ("PagefileUsage",              ctypes.c_size_t),
        ("PeakPagefileUsage",          ctypes.c_size_t),
    ]


def _win_working_set_mib():
    """(current, peak) working set in MiB on Windows, or None elsewhere/on error.

    Working set is the closest Windows analogue of RSS; PeakWorkingSetSize is
    the monotone high-water mark, matching ru_maxrss on POSIX.
    """
    if sys.platform != "win32":
        return None
    try:
        counters = _PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(_PROCESS_MEMORY_COUNTERS)
        psapi = ctypes.WinDLL("psapi.dll")  # type: ignore[attr-defined]
        kernel32 = ctypes.WinDLL("kernel32.dll")  # type: ignore[attr-defined]
        if psapi.GetProcessMemoryInfo(kernel32.GetCurrentProcess(),
                                      ctypes.byref(counters),
                                      counters.cb):
            mib = 1024.0 * 1024.0
            return (counters.WorkingSetSize / mib,
                    counters.PeakWorkingSetSize / mib)
    except Exception:
        pass
    return None


def current_rss_mib():
    """Current resident set size (RSS) of this process in MiB (0.0 if unavailable)."""
    win = _win_working_set_mib()
    if win is not None:
        return win[0]
    if sys.platform == "darwin":
        try:
            out = subprocess.run(["ps", "-o", "rss=", "-p", str(os.getpid())],
                                 capture_output=True, text=True, check=True)
            return int(out.stdout.strip()) / 1024.0        # ps reports KiB
        except Exception:
            return 0.0
    try:
        with open("/proc/self/statm") as fh:
            resident_pages = int(fh.read().split()[1])
        return resident_pages * os.sysconf("SC_PAGESIZE") / (1024.0 * 1024.0)
    except Exception:
        return 0.0


class _TaskVMInfo(ctypes.Structure):
    """mach task_vm_info through the phys_footprint field (TASK_VM_INFO rev1)."""
    _fields_ = [
        ("virtual_size",                ctypes.c_uint64),
        ("region_count",                ctypes.c_int32),
        ("page_size",                   ctypes.c_int32),
        ("resident_size",               ctypes.c_uint64),
        ("resident_size_peak",          ctypes.c_uint64),
        ("device",                      ctypes.c_uint64),
        ("device_peak",                 ctypes.c_uint64),
        ("internal",                    ctypes.c_uint64),
        ("internal_peak",               ctypes.c_uint64),
        ("external",                    ctypes.c_uint64),
        ("external_peak",               ctypes.c_uint64),
        ("reusable",                    ctypes.c_uint64),
        ("reusable_peak",               ctypes.c_uint64),
        ("purgeable_volatile_pmap",     ctypes.c_uint64),
        ("purgeable_volatile_resident", ctypes.c_uint64),
        ("purgeable_volatile_virtual",  ctypes.c_uint64),
        ("compressed",                  ctypes.c_uint64),
        ("compressed_peak",             ctypes.c_uint64),
        ("compressed_lifetime",         ctypes.c_uint64),
        ("phys_footprint",              ctypes.c_uint64),
    ]


def current_footprint_mib():
    """
    Physical footprint of this process in MiB: anonymous heap including
    compressed / swapped-out pages (Activity Monitor's "Memory" on macOS).
    Linux approximation: VmRSS + VmSwap. Falls back to RSS on failure.
    """
    if sys.platform == "darwin":
        try:
            libc = ctypes.CDLL(None)
            info  = _TaskVMInfo()
            count = ctypes.c_uint32(ctypes.sizeof(_TaskVMInfo) // 4)
            TASK_VM_INFO = 22
            kr = libc.task_info(libc.mach_task_self(), TASK_VM_INFO,
                                ctypes.byref(info), ctypes.byref(count))
            if kr == 0 and info.phys_footprint > 0:
                return info.phys_footprint / (1024.0 * 1024.0)
        except Exception:
            pass
        return current_rss_mib()
    try:
        kib = 0.0
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith(("VmRSS:", "VmSwap:")):
                    kib += float(line.split()[1])
        return kib / 1024.0
    except Exception:
        return current_rss_mib()


def peak_rss_mib():
    """Peak RSS of this process in MiB (monotone over the process lifetime)."""
    win = _win_working_set_mib()
    if win is not None:
        return win[1]
    if resource is None:            # no getrusage: best effort
        return current_rss_mib()
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # ru_maxrss is bytes on macOS, KiB on Linux
    return r / (1024.0 * 1024.0) if sys.platform == "darwin" else r / 1024.0


# ==============================================================================
# Memory-mapped DGP (mirrors SyntheticDataMapped: X generated on disk)
# ==============================================================================

def dgp_gauss_snr_mmap(n, p, support, coefs, snr, seed, x_path,
                       chunk_cols=_CHUNK_COLS):
    """
    Generate X column-chunk by column-chunk into an mmap backing file and an
    SNR-calibrated y (kept in RAM; n doubles are negligible). X never sits fully
    in RAM; working memory is O(n * chunk_cols).

    Returns (X_mmap, y) with X_mmap a MemoryMappedMatrix over x_path.
    """
    rng  = np.random.default_rng(seed)
    beta = np.zeros(p, dtype=np.float64)
    beta[np.asarray(support, dtype=np.intp)] = np.asarray(coefs, dtype=np.float64)

    X = MemoryMappedMatrix(x_path, n, p, AccessMode.ReadWrite)
    V = X.to_numpy()                     # zero-copy F-contiguous view
    signal = np.zeros(n)
    for s0 in range(0, p, chunk_cols):
        e0    = min(s0 + chunk_cols, p)
        chunk = rng.standard_normal((n, e0 - s0))
        V[:, s0:e0] = chunk
        signal += chunk @ beta[s0:e0]

    noise_sigma = np.sqrt(np.var(signal, ddof=1) / snr)
    y = signal + rng.standard_normal(n) * noise_sigma
    return X, y


# ==============================================================================
# Single-trial runner (sequential; timing + memory instrumented)
# ==============================================================================

def run_scalability_trial(n, p, ctrl_dict, tFDR, snr, seed, support, coefs,
                          mmap_filestem):
    """
    One MC trial on the fully memory-mapped pipeline. The X backing file is
    removed in a finally block (exception-safe; mirrors the C++ RAII
    MmapFileGuard). Returns a dict of per-trial measurements.
    """
    m = {}
    x_path = f"{mmap_filestem}_X.dat"

    try:
        t0 = time.perf_counter()
        X_mmap, y = dgp_gauss_snr_mmap(n, p, support, coefs, snr, seed, x_path)
        m["dgp_s"] = time.perf_counter() - t0

        m["data_mib"] = (os.path.getsize(x_path) + y.nbytes) / (1024.0 * 1024.0)

        ctrl = _make_trex_ctrl_from_dict(ctrl_dict)

        t1 = time.perf_counter()
        # X_mmap.to_numpy() → zero-copy F-contiguous float64 mmap view;
        # seed=-1 (hardware entropy) as in all MC demos.
        sel = trex.TRexSelector(X_mmap.to_numpy(), y, tFDR=tFDR,
                               trex_control=ctrl, seed=-1, verbose=False)
        sel.select()
        m["select_s"] = time.perf_counter() - t1

        m["rss_mib"]      = current_rss_mib()
        m["fp_mib"]       = current_footprint_mib()
        m["peak_rss_mib"] = peak_rss_mib()   # process-lifetime max up to this trial

        selected = list(sel.selected_indices)
        support_list = list(support)
        m["fdp"] = compute_fdp(selected, support_list)
        m["tpp"] = compute_tpp(selected, support_list)
        m["L"]   = float(sel.L)
        m["T"]   = float(sel.T_stop)

        # Drop the selector and mmap view before the backing file is removed
        # (CPython ref-counting makes this deterministic).
        del sel, X_mmap
    finally:
        try:
            os.unlink(x_path)
        except OSError:
            pass
    return m


# ==============================================================================
# Grid-point loop (n, p, solver): sequential MC trials
# ==============================================================================

def run_scalability_grid_point(n, p, num_MC, dummy_mult, solver_desc, tFDR, snr,
                               rnd_coef, cardinality_true_support,
                               cell_cache, raw_path, disk_ok=True):
    """Ensure >= num_MC trials exist for this (n, p, solver) cell, then aggregate.

    Only the missing trial indices [0, num_MC) are computed; each is appended to
    the raw store immediately (trial-level durability), so a crash keeps them.
    With disk_ok=False (no disk, or solver not in TREX_SOLVERS) nothing is
    computed and only cached trials count; the return value is None when the
    cell has no trials at all.
    """
    ctrl_dict = _make_scalability_ctrl_dict(solver_desc, dummy_mult)
    mmap_filestem = os.path.join(_TMPDIR, f"demo08_scalability_n{n}_p{p}")

    have = len(cell_cache)
    todo = ([mc for mc in range(num_MC) if mc not in cell_cache]
            if disk_ok else [])
    reuse_note = f"  (reusing {have} cached)" if have else ""
    print(f"  [{solver_desc['name']}] n={n}, p={p}  (L cap {dummy_mult}) — "
          f"{len(todo)} new / {num_MC} MC trials (mmap){reuse_note} ...",
          flush=True)

    for mc in todo:
        if _stop_requested():
            print("  [stop] TREX_STOP_FILE found, no new trial started",
                  flush=True)
            break
        trial_seed = _trial_seed(n, p, mc)

        # Variable support (demo 03 pattern; make_support_random adds the
        # +500_000 offset internally, mirroring C++ draw_support).
        support = make_support_random(cardinality_true_support, p,
                                      seed=trial_seed)
        if rnd_coef:
            coef_rng = np.random.default_rng(int(trial_seed) + 600_000)
            coefs = coef_rng.standard_normal(cardinality_true_support)
        else:
            coefs = np.ones(cardinality_true_support)

        m = run_scalability_trial(n, p, ctrl_dict, tFDR, snr, trial_seed,
                                  support, coefs, mmap_filestem)

        row = dict(n=n, p=p, solver=solver_desc["name"], mc=mc,
                   trial_seed=trial_seed,
                   dgp_s=round(m["dgp_s"], 6), select_s=round(m["select_s"], 6),
                   rss_mib=round(m["rss_mib"], 3), fp_mib=round(m["fp_mib"], 3),
                   peak_rss_mib=round(m["peak_rss_mib"], 3),
                   data_mib=round(m["data_mib"], 3),
                   fdp=round(m["fdp"], 6), tpp=round(m["tpp"], 6),
                   L=round(m["L"], 3), T=round(m["T"], 3))
        _append_trial_row(raw_path, row)          # durable before we continue
        cell_cache[mc] = row

        print(f"    trial {mc + 1}/{num_MC}: dgp={m['dgp_s']:.2f}s  "
              f"select={m['select_s']:.2f}s  FDP={m['fdp']:.3f}  "
              f"TPP={m['tpp']:.3f}  L={m['L']:.0f}  T={m['T']:.0f}  "
              f"RSS={m['rss_mib']:.1f} MiB  FP={m['fp_mib']:.1f} MiB",
              flush=True)

    if not cell_cache:
        return None
    agg = _aggregate_cell(cell_cache, n, p)
    print(f"  [{solver_desc['name']}] n={n}, p={p} done "
          f"({agg['num_mc']} trials). "
          f"select mean={agg['sel_mean_s']:.2f}s "
          f"[{agg['sel_min_s']:.2f}, {agg['sel_max_s']:.2f}]  "
          f"peakRSS={agg['peak_rss_mib']:.0f} MiB  "
          f"FDR={agg['avg_fdr']:.3f}  TPR={agg['avg_tpr']:.3f}\n", flush=True)
    return agg


# ==============================================================================
# Output: aligned table (console + .txt) and tidy CSV
# ==============================================================================

# Per-grid-point metric columns (table order and CSV metric set):
# (label, decimals, aggregate key)
_METRICS = [
    ("VolGiB",     1, "vol_gib"),
    ("SelMean_s",  3, "sel_mean_s"),
    ("SelMed_s",   3, "sel_med_s"),
    ("SelMin_s",   3, "sel_min_s"),
    ("SelMax_s",   3, "sel_max_s"),
    ("DGP_s",      3, "avg_dgp_s"),
    ("s/Titer",    4, "avg_titer_s"),
    ("PeakRSS_MiB", 1, "peak_rss_mib"),
    ("RSS_MiB",    1, "max_rss_mib"),
    ("Foot_MiB",   1, "max_fp_mib"),
    ("X+y_MiB",    1, "data_mib"),
    ("FDR",        4, "avg_fdr"),
    ("TPR",        4, "avg_tpr"),
    ("Avg_L",      2, "avg_L"),
    ("Avg_T",      2, "avg_T"),
]


def save_and_print_scalability_results(file_stem, block_title, tFDR, snr, grid,
                                       solvers, results, num_MC_target):
    """
    Print the scalability results as an aligned per-solver table (rows = (n, p)
    grid points, columns = metrics) to console + .txt, and write a tidy
    long-format CSV. `results` maps solver name -> {(n, p): agg dict}.
    is_barrier = 1 where the volume at the realized L exceeds RAM.
    Mirrors the C++ function of the same name.
    """
    os.makedirs(_OUT_DIR, exist_ok=True)
    txt_path = os.path.join(_OUT_DIR, f"{file_stem}.txt")

    with open(txt_path, "w") as fh:
        out = make_dual_out(fh)

        out("\n" + "=" * 70 + "\n")
        out(f"=== T-Rex Scalability Results (tFDR = {tFDR:.2f}, SNR = {snr:.1f}, "
            f"num_MC = {num_MC_target}) ===\n")
        out("=" * 70 + "\n\n")
        out(block_title + "\n")
        out("Fully memory-mapped pipeline (X mmap + D mmap + solver "
            "serialization).\n")
        out(f"Physical RAM: {_RAM_GIB:.0f} GiB. VolGiB = X + K*L*X, the data one "
            f"T-iteration touches (K = {_NUM_RND_EXP}, L = Avg_L).\n\n")

        n_w, p_w, met_w = 8, 9, 13
        for solver in solvers:
            grid_res = results[solver["name"]]
            out(solver["name"] + "\n")
            hdr = f"{'n':<{n_w}}{'p':<{p_w}}"
            for name, _, _ in _METRICS:
                hdr += f"{name:>{met_w}}"
            out(hdr + "\n" + "-" * len(hdr) + "\n")
            for (n, p, _num_mc, _dummy_mult) in grid:
                agg = grid_res.get((n, p))
                if agg is None:
                    continue
                lbl_p = f"{p}{'*' if agg['vol_gib'] > _RAM_GIB else ''}"
                row = f"{n:<{n_w}}{lbl_p:<{p_w}}"
                for _name, prec, key in _METRICS:
                    row += f"{agg[key]:>{met_w}.{prec}f}"
                out(row + "\n")
            out("\n")
        out("(* = out-of-core point, VolGiB > physical RAM)\n\n")

    # Tidy long-format CSV: n, p, is_barrier, solver, num_mc, snr, metric, value
    # (num_mc is the actual cached trial count per cell, not the run target).
    rows = []
    for solver in solvers:
        grid_res = results[solver["name"]]
        for (n, p, _num_mc, _dummy_mult) in grid:
            agg = grid_res.get((n, p))
            if agg is None:
                continue
            is_barrier = agg["vol_gib"] > _RAM_GIB
            for name, _, key in _METRICS:
                rows.append(dict(n=n, p=p, is_barrier=int(is_barrier),
                                 solver=solver["name"], num_mc=agg["num_mc"],
                                 snr=f"{snr:.6f}", metric=name,
                                 value=round(agg[key], 6)))
    csv_path = os.path.join(_OUT_DIR, f"{file_stem}.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"[Info] CSV results saved to:          {csv_path}")
    print(f"[Info] Table results saved to:        {txt_path}\n")


# ==============================================================================
# Scalability benchmark driver
# ==============================================================================

def demo_trex_scalability(grid, file_stem, block_title, num_MC_target,
                          rnd_coef=False, persist_cache=True):
    # Clear any orphans stranded by a previous cancelled run before we start.
    sweep_mmap_orphans("startup")

    os.makedirs(_OUT_DIR, exist_ok=True)
    raw_path = _raw_cache_path(file_stem, persist_cache)
    cache = _load_trial_cache(raw_path)   # {(n, p, solver): {mc: row}}
    cached_trials = sum(len(v) for v in cache.values())

    print("\n" + "=" * 70)
    print("Demo: T-Rex Selector Scalability Benchmark  |  TLARS / TOMP / TAFS")
    print("=" * 70 + "\n")

    vols = [_volume_gib(n, p, 1.0) for (n, p, _mc, _dm) in grid]
    max_disk = max(_worst_case_disk_gib(n, p, dm) for (n, p, _mc, dm) in grid)
    n_over = sum(1 for v in vols if v > _RAM_GIB)
    print(block_title)
    print("Fully memory-mapped pipeline (X mmap + D mmap + solver serialization)")
    print(f"Physical RAM: {_RAM_GIB:.0f} GiB,  free temp disk: "
          f"{_free_temp_disk_gib():.0f} GiB")
    print(f"Grid: {len(grid)} points; largest volume X + K*X (L = 1) = "
          f"{max(vols):.1f} GiB; {n_over} point(s) exceed RAM at L = 1 "
          f"(out-of-core); largest worst-case disk need = {max_disk:.1f} GiB")
    print(f"SNR = {_SNR:.1f} (fixed),  s = {_S_TRUE},  tFDR = {_TFDR:.2f},  "
          f"num_MC = {num_MC_target} sequential per (n, p, solver)")
    print(f"Trial cache: {raw_path}  ({cached_trials} trial(s) already present "
          f"— only missing trials are computed)\n")

    # results: solver name -> {(n, p): agg}
    results = {sv["name"]: {} for sv in _SOLVERS}

    # (n, p)-outer loop; every solver finishes the current point before the
    # next (larger) one starts, and results are re-saved after each point so a
    # crash on a bigger point preserves the finished ones.
    for gi, (n, p, num_MC, dummy_mult) in enumerate(grid):
        vol1 = _volume_gib(n, p, 1.0)
        print("=" * 50)
        print(f"Grid point: n={n}, p={p}"
              f"{'   [out-of-core]' if vol1 > _RAM_GIB else ''}")
        print("=" * 50)

        # Disk pre-flight: skip a point early when its worst case cannot fit.
        need = _worst_case_disk_gib(n, p, dummy_mult)
        avail = _free_temp_disk_gib()
        disk_ok = np.isnan(avail) or need + _DISK_RESERVE_GIB <= avail
        print(f"  volume X + K*X (L = 1) = {vol1:.1f} GiB  (X backing "
              f"{n * p * 8.0 / _BYTES_PER_GIB:.2f} GiB, K = {_NUM_RND_EXP} "
              f"dummy files)")
        print(f"  worst-case disk need (L = {dummy_mult}) = {need:.1f} GiB,  "
              f"free = {avail:.1f} GiB,  reserve = {_DISK_RESERVE_GIB:.1f} GiB")
        if not disk_ok:
            print("  [skip] not enough free disk — new trials are not computed "
                  "for this point")
        print("", flush=True)

        # TREX_SOLVERS limits the computed solvers; the others keep their cache.
        for solver_desc in _SOLVERS:
            cell_cache = cache.setdefault((n, p, solver_desc["name"]), {})
            compute = disk_ok and solver_filter.selected(solver_desc["name"])
            agg = run_scalability_grid_point(
                n, p, num_MC, dummy_mult, solver_desc, _TFDR, _SNR,
                rnd_coef, _S_TRUE, cell_cache, raw_path, compute)
            if agg is not None:
                results[solver_desc["name"]][(n, p)] = agg

        # Incremental save over completed points (derived from the trial cache).
        save_and_print_scalability_results(file_stem, block_title, _TFDR, _SNR,
                                           grid[:gi + 1], _SOLVERS, results,
                                           num_MC_target)
        if _stop_requested():
            print(f"\n[stop] Stopped on request after point n={n}, p={p}; "
                  "finished trials are cached, the next run resumes.",
                  flush=True)
            return


# ==============================================================================
# Main
# ==============================================================================

if __name__ == "__main__":
    _install_cleanup_handlers()

    # Fixed thread count (mirrors the C++ demo): the library's default budget
    # here, max_inner_threads in the control. Experiments and trials run one
    # after another; the threads run the column loops inside each experiment.
    set_num_threads(_NUM_THREADS)
    print(f"Running with {get_num_threads()} thread(s) "
          f"(pinned to match the C++ demo)\n")

    # CLI:  [smoke | base | large] [num_MC]  (order-independent; num_MC is any
    # integer token; without a block token both blocks run, base first)
    _args = sys.argv[1:]
    _smoke = "smoke" in _args
    _run_base = "large" not in _args
    _run_large = "base" not in _args
    _num_MC = next((int(a) for a in _args if a.lstrip("+").isdigit()), None)

    if _smoke:
        mc = _num_MC if _num_MC is not None else _SMOKE_MC
        demo_trex_scalability(_smoke_grid(mc),
                              "demo_trex_08_scalability_smoke",
                              "Smoke grid (L cap 10 and 5)", mc,
                              persist_cache=False)
    else:
        mc = _num_MC if _num_MC is not None else _NUM_MC
        if _run_base:
            demo_trex_scalability(
                _benchmark_grid(mc), "demo_trex_08_scalability_results",
                f"Base block: n x p cross, max_dummy_multiplier = {_DUMMY_MULT}",
                mc)
        if _run_large and not _stop_requested():
            demo_trex_scalability(
                _large_p_grid(mc), "demo_trex_08_scalability_large_p",
                "Large-p block: n = 300, p up to 1e6, max_dummy_multiplier = "
                f"{_LARGE_P_DUMMY_MULT}", mc)

    if _stop_requested():
        sys.exit(_EXIT_STOPPED)
    print("\ndemo_trex_08_mc_sim_scalability.py complete.")
