# ==============================================================================
# trex_spca_sim_utils.py
# ==============================================================================
#
# Shared simulation utilities for the T-Rex SPCA demo files. Python counterpart
# of cpp/trex_selector_methods/ad/trex_spca/trex_spca_sim_utils.hpp.
#
#   Data generation:
#     generate_sparse_factor_model()  — sparse M-factor model DGP
#                                       (returns dict: X, Z, V, E, actual_snr_db)
#
#   Preprocessing:
#     center_columns()                — column centering (covariance-PCA footing)
#
#   Evaluation:
#     evaluate_spca()                 — PC1 TPR/FDR + the three cumulative-PEV
#                                       readings (total-variance, Definition 1,
#                                       and the method's own Sig+Mix part)
#
#   MC loop functions (parallel via ProcessPoolExecutor):
#     run_mc_trials_trex_spca()       — MC loop for T-Rex SPCA
#     run_mc_trials_pca()             — MC loop for the ordinary PCA baseline
#     run_mc_trials_oracle_pca()      — MC loop for oracle thresholded PCA
#
#   Output:
#     save_and_print_spca_sweep_results() — generic sweep table (console + .txt)
#                                           + tidy .csv (pandas), with the axis
#                                           header printed at a per-sweep-column
#                                           precision (mirrors the cpp helper)
#     save_and_print_spca_mc_results()    — SNR-sweep convenience wrapper
#                                           (CSV column "snr_db", header
#                                           "SNR(dB)", one decimal)
#
# Reproducibility pattern (mirrors cpp):
#   - Each trial's DATA is drawn from a deterministic seed
#     (base_seed_offset + mc*1000, disjoint 1000-wide bands per trial).
#   - The T-Rex selector runs with seed = -1 (hardware entropy per trial), so
#     the computer-generated dummies are independent across trials — required
#     for a valid Monte Carlo FDR estimate.
#
# Support indices are 0-based (Python convention).
# ==============================================================================

import atexit
import multiprocessing
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

import trexselector as trex
from trexselector.ml_methods import PCA

# Family folder: cell_cache.py (resumable grid cells).
_FAMILY_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _FAMILY_DIR not in sys.path:
    sys.path.insert(0, _FAMILY_DIR)
import cell_cache

# ------------------------------------------------------------------------------
# Worker budget  (identical helper in every suite's *_common / *_utils module)
#
# N_CORES MC worker processes, each running the selector serially (the library
# default), so a demo never exceeds N_CORES cores in total. The default cap
# matches the cpp demos' default of 8 threads, keeping runtimes comparable
# across the C++ / Python suites.
#
# Override per run, no code edit needed:
#   TREX_NUM_WORKERS=12   exactly 12 workers (clamped to the usable CPU count)
#   TREX_NUM_WORKERS=0    use every usable CPU
# The spca demos also accept the count as argv[1], which wins over the env var.
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
# The demos call the MC runners once per parameter point, so a pool created per
# call re-pays (process spawn + `import trexselector`) every time: measured
# at ~0.5 s for 6 workers. One lazily-created pool is reused instead, keyed on
# n_cores so a demo's CLI worker-count override still takes effect. Public,
# because the demos drive their own loops for the parts that bypass _run_mc().
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


def get_executor(n_cores, n_tasks=0):
    """Return the shared pool, (re)creating it if the worker count changed."""
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

# String -> enum maps: only plain strings cross the process boundary; workers
# resolve them to trexselector enums on their side.
_SPCA_MODE = {"ActiveSet": trex.SPCAMode.ActiveSet,
              "Thresholded": trex.SPCAMode.Thresholded}
_EN_SOLVER = {"TENET": trex.ENSolverType.TENET,
              "TENET_AUG": trex.ENSolverType.TENET_AUG}
_SCALING = {"L2": trex.ScalingMode.L2,
            "ZSCORE": trex.ScalingMode.ZSCORE}


# ==============================================================================
# Data generator (sparse M-factor model)
# ==============================================================================

def generate_sparse_factor_model(n=50, p=100, p1=5, M=3, target_snr=0.0,
                                 overlap_pool_size=30, seed=42):
    """Generate a single dataset from the sparse factor model X = Z V^T + E.

    Z (n x M) has column-wise N(0, sigma_m^2) factors with sigma = {5, 3, 1}
    (falling back to 1 for M > 3). V (p x M) has exactly p1 non-zeros (value
    0.9) per column, drawn without replacement from a shared candidate pool of
    size overlap_pool_size — so different factors' active sets can overlap.
    E is i.i.d. Gaussian noise scaled to match the target SNR in dB.

    Returns dict(X, Z, V, E, actual_snr_db); X is raw (uncentered).
    """
    if not (0 < p1 <= p) or n <= 0:
        raise ValueError("n, p, p1 must be positive with p1 <= p.")
    rng = np.random.default_rng(seed)

    # 1. True factors Z (n x M) with decreasing signal strength
    factor_stds = [5.0, 3.0, 1.0]
    Z = np.column_stack([
        rng.normal(0.0, factor_stds[m] if m < len(factor_stds) else 1.0, n)
        for m in range(M)])

    # 2. Sparse loadings V (p x M): each factor draws p1 indices from the pool
    actual_pool = min(overlap_pool_size, p)
    if p1 > actual_pool:
        raise ValueError("p1 must be <= the overlap pool size.")
    V = np.zeros((p, M))
    for m in range(M):
        V[rng.choice(actual_pool, p1, replace=False), m] = 0.9

    # 3. Signal matrix S = Z V^T
    S = Z @ V.T
    signal_var = S.var(ddof=1)

    # 4. Gaussian noise E scaled to the target SNR
    # SNR_dB = 10 * log10(var_S / var_E)  =>  var_E = var_S / 10^(SNR_dB / 10)
    E = rng.normal(0.0, 1.0, (n, p))
    target_noise_var = signal_var / 10.0 ** (target_snr / 10.0)
    E *= np.sqrt(target_noise_var / E.var(ddof=1))

    X = S + E
    actual_snr_db = 10.0 * np.log10(signal_var / E.var(ddof=1))
    return dict(X=X, Z=Z, V=V, E=E, actual_snr_db=actual_snr_db)


# ==============================================================================
# Preprocessing
# ==============================================================================

def center_columns(X):
    """Center columns (mean subtraction only, NO scaling).

    Covariance-PCA footing shared by every method, matching the legacy R
    reference pipeline and the selector's internal convention. Column scales
    carry the factor amplitude signal and must not be normalized (z-scoring
    destroys it — see cpp validation_trex_spca_06_handrolled_comparison).
    """
    return X - X.mean(axis=0)


# ==============================================================================
# Single-trial evaluator
# ==============================================================================

def evaluate_spca(X, V_est, Z_est, V_true):
    """Compute per-trial sparse PCA metrics from estimated loadings and scores.

    TPR / FDR are evaluated strictly on the FIRST principal component (PC1):
    PCA's orthogonality constraint mixes the loading supports of PCs 2+ across
    the true factors, making per-component support recovery metrics ambiguous
    beyond the first PC. An empty true support yields TPR = 0.0 (legacy R
    reference convention).

    The cumulative adjusted EV follows Zou/Hastie/Tibshirani: a QR
    decomposition of Z_est removes the variance correlated components would
    double-count, so the diagonal of R carries each component's incremental
    contribution. It is reported under THREE normalizations (mirrors the cpp
    evaluate_spca; see the suite README for the full account):

      pev        — divided by the total variance of X (bounded by 1).
      pev_sig    — Definition 1 of the reference paper: divided by the
                   Signal + Mixed EV OF THE DATA, i.e. the variance carried by
                   the true active variables (union of the factor supports),
                   ||X_A||_F^2 / (n-1) — FIXED per dataset and SHARED by all
                   methods. Deliberately uncapped: a method that spends
                   loadings on null variables accumulates variance the
                   denominator does not contain and climbs past 1. Denominator
                   follows the active-column convention of the legacy R
                   reference (demo_trex_spca_03_fig3.R, mode
                   "active"); it is NOT the method's own Signal + Mixed EV —
                   that self-normalized reading is structurally >= 1 for
                   null-capturing methods and inverts the SNR trend.
      pev_sigmix — the method's OWN Signal + Mixed part (raw EV minus the Null
                   EV of its loadings, split by variable) over the same shared
                   denominator; read for OrdPCA it is the paper's
                   "Ordinary PCA (Sig + Mix)" reference curve.

    V_true may have FEWER columns than V_est/Z_est: sweeping the extracted PC
    count past the number of true factors is how Fig. 3(a) of the reference
    paper is built — the extra components are null by definition.

    Returns dict(tpr, fdr, pev, pev_sig, pev_sigmix).
    """
    n = X.shape[0]

    # -----------------------------------------------------------------
    # 1. TPR / FDR strictly on the FIRST component
    # -----------------------------------------------------------------
    true_supp_pc1 = set(np.flatnonzero(V_true[:, 0] != 0.0))
    est_supp_pc1 = set(np.flatnonzero(np.abs(V_est[:, 0]) > 1e-12))

    tp = len(true_supp_pc1 & est_supp_pc1)
    tpr = 0.0 if not true_supp_pc1 else tp / len(true_supp_pc1)
    fdr = (0.0 if not est_supp_pc1
           else (len(est_supp_pc1) - tp) / len(est_supp_pc1))

    # -----------------------------------------------------------------
    # 2. Cumulative adjusted EV across ALL components (QR of Z_est)
    # -----------------------------------------------------------------
    R = np.linalg.qr(Z_est, mode="r")
    cum_ev = (np.diag(R) ** 2).sum() / (n - 1)

    # 2a. PEV, normalized by the TOTAL variance of X (bounded by 1)
    total_var = (X ** 2).sum() / (n - 1)
    pev = cum_ev / total_var if total_var > 0.0 else 0.0

    # 2b. PEV, normalized by the data's SIGNAL + MIXED EV (Definition 1).
    # Membership of A is per VARIABLE (row of V_true): active when it belongs
    # to the true support of at least one factor.
    row_active = (V_true != 0.0).any(axis=1)
    sig_mixed_ev = (X[:, row_active] ** 2).sum() / (n - 1)
    pev_sig = cum_ev / sig_mixed_ev if sig_mixed_ev > 0.0 else 0.0

    # 2c. The method's own Signal + Mixed share over the same denominator:
    # split V_est by variable into active and null rows and remove the Null EV
    # from its raw (unadjusted) EV.
    V_null = V_est.copy()
    V_null[row_active, :] = 0.0
    raw_ev = ((X @ V_est) ** 2).sum() / (n - 1)
    null_ev = ((X @ V_null) ** 2).sum() / (n - 1)
    pev_sigmix = ((raw_ev - null_ev) / sig_mixed_ev
                  if sig_mixed_ev > 0.0 else 0.0)

    return dict(tpr=tpr, fdr=fdr, pev=pev, pev_sig=pev_sig,
                pev_sigmix=pev_sigmix)


# ==============================================================================
# Per-trial workers (top-level, importable by ProcessPoolExecutor spawn workers)
# ==============================================================================

def _trial_trex_spca(task):
    (seed, dgp_kwargs, tFDR, mode, lambda_2, en_solver, scaling_mode,
     num_components) = task
    dat = generate_sparse_factor_model(seed=seed, **dgp_kwargs)
    X = center_columns(dat["X"])

    ctrl = trex.TRexSPCAControlParameter()
    ctrl.mode = _SPCA_MODE[mode]
    ctrl.en_solver = _EN_SOLVER[en_solver]
    # The base control parameters live under gvs_ctrl.trex_ctrl (GVS
    # sub-selector) — TRexSPCAControlParameter has no top-level trex_ctrl.
    ctrl.gvs_ctrl.trex_ctrl.scaling_mode = _SCALING[scaling_mode]
    ctrl.gvs_ctrl.lambda2_method = trex.LambdaSelectionMethod.CV_1SE_CCD
    ctrl.gvs_ctrl.lambda_2 = lambda_2  # < 0: auto via CV; 0: no ridge; > 0: fixed

    # 0 means "extract as many components as there are true factors"; a
    # positive value decouples the two (Fig. 3(a) sweeps the PC count).
    n_comp = num_components if num_components > 0 else dat["V"].shape[1]

    # Selector seed -1: hardware entropy per trial (dummies independent across
    # trials). The wrapper maps X zero-copy — hand it a Fortran-ordered copy.
    sel = trex.TRexSPCASelector(np.asfortranarray(X), spca_ctrl=ctrl, seed=-1)
    sel.select(n_comp, tFDR)

    m = evaluate_spca(X, np.asarray(sel.V), np.asarray(sel.Z), dat["V"])
    return m["tpr"], m["fdr"], m["pev"], m["pev_sig"], m["pev_sigmix"]


def _trial_pca(task):
    seed, dgp_kwargs, num_components = task
    dat = generate_sparse_factor_model(seed=seed, **dgp_kwargs)
    X = center_columns(dat["X"])
    n_comp = num_components if num_components > 0 else dat["V"].shape[1]

    # X is already centered -> center=False, normalize=False (covariance PCA).
    pca = PCA(center=False, normalize=False)
    res = pca.fit(np.asfortranarray(X), n_comp)

    m = evaluate_spca(X, np.asarray(res.V), np.asarray(res.Z), dat["V"])
    return m["tpr"], m["fdr"], m["pev"], m["pev_sig"], m["pev_sigmix"]


def _trial_oracle_pca(task):
    seed, dgp_kwargs, p1, num_components = task
    dat = generate_sparse_factor_model(seed=seed, **dgp_kwargs)
    X = center_columns(dat["X"])
    p = X.shape[1]
    n_comp = num_components if num_components > 0 else dat["V"].shape[1]

    pca = PCA(center=False, normalize=False)
    ord_res = pca.fit(np.asfortranarray(X), n_comp)
    V_ord = np.asarray(ord_res.V)

    # Keep top p1 entries per loading vector, then normalize
    V_oracle = np.zeros((p, n_comp))
    for comp in range(n_comp):
        v_m = V_ord[:, comp]
        thresh = np.sort(np.abs(v_m))[p - p1]
        keep = np.abs(v_m) >= thresh
        V_oracle[keep, comp] = v_m[keep]
        V_oracle[:, comp] /= np.linalg.norm(V_oracle[:, comp])
    Z_oracle = X @ V_oracle

    m = evaluate_spca(X, V_oracle, Z_oracle, dat["V"])
    return m["tpr"], m["fdr"], m["pev"], m["pev_sig"], m["pev_sigmix"]


# ==============================================================================
# MC aggregation helper
# ==============================================================================

def _run_mc(worker, tasks, label, n_cores):
    # One resumable grid cell (cell_cache): restored after a restart.
    def compute():
        print(f"  {label} — Running {len(tasks)} MC trials ...", flush=True)
        if n_cores > 1:
            res = list(get_executor(n_cores, len(tasks)).map(worker, tasks))
        else:
            res = [worker(t) for t in tasks]

        arr = np.asarray(res)  # (num_MC, 5): tpr, fdr, pev, pev_sig, pev_sigmix
        return dict(avg_tpr=arr[:, 0].mean(), avg_fdr=arr[:, 1].mean(),
                    avg_pev=arr[:, 2].mean(), avg_pev_sig=arr[:, 3].mean(),
                    avg_pev_sigmix=arr[:, 4].mean(),
                    sd_tpr=arr[:, 0].std(ddof=1) if len(arr) > 1 else 0.0,
                    sd_fdr=arr[:, 1].std(ddof=1) if len(arr) > 1 else 0.0)

    out = cell_cache.cached(f"{worker.__name__}|{label}|{cell_cache.digest(tasks)}", compute)
    print(f"  {label} — done. TPR={out['avg_tpr']:.3f}  FDR={out['avg_fdr']:.3f}\n",
          flush=True)
    return out


# ==============================================================================
# Parallel MC inner loops
# ==============================================================================

def run_mc_trials_trex_spca(num_MC, progress_label, dgp_kwargs, tFDR, mode,
                            base_seed_offset, lambda_2=-1.0, en_solver="TENET",
                            scaling_mode="L2", num_components=0,
                            n_cores=N_CORES):
    """Run num_MC parallel T-Rex SPCA trials.

    Per trial: the DGP draws data from seed base_seed_offset + mc*1000; the
    pipeline centers X once (covariance PCA); this same X is shared by the
    selector and by evaluate_spca(). mode is "ActiveSet" or "Thresholded";
    en_solver is "TENET" or "TENET_AUG"; scaling_mode is "L2" or "ZSCORE"
    (per-PC selector column scaling); num_components > 0 extracts that many
    PCs instead of the number of true factors. Returns dict(avg_tpr, avg_fdr,
    avg_pev, avg_pev_sig, avg_pev_sigmix, sd_tpr, sd_fdr).
    """
    tasks = [(base_seed_offset + mc * 1000, dgp_kwargs, tFDR, mode,
              lambda_2, en_solver, scaling_mode, num_components)
             for mc in range(num_MC)]
    return _run_mc(_trial_trex_spca, tasks, progress_label, n_cores)


def run_mc_trials_pca(num_MC, progress_label, dgp_kwargs, base_seed_offset,
                      num_components=0, n_cores=N_CORES):
    """Run num_MC parallel ordinary PCA trials.

    Ordinary PCA retains all p loadings per component (no sparsity), so
    FDR ~ (p - p1) / p and TPR = 1 regardless of SNR; the metrics of interest
    are the cumulative-PEV readings.
    """
    tasks = [(base_seed_offset + mc * 1000, dgp_kwargs, num_components)
             for mc in range(num_MC)]
    return _run_mc(_trial_pca, tasks, progress_label, n_cores)


def run_mc_trials_oracle_pca(num_MC, progress_label, dgp_kwargs, p1,
                             base_seed_offset, num_components=0,
                             n_cores=N_CORES):
    """Run num_MC parallel oracle-thresholded PCA trials.

    Oracle PCA keeps exactly the top p1 entries (by absolute value) of each
    ordinary-PCA loading vector — it knows the true support cardinality. This
    bounds what any data-driven thresholding method can achieve.
    """
    tasks = [(base_seed_offset + mc * 1000, dgp_kwargs, p1, num_components)
             for mc in range(num_MC)]
    return _run_mc(_trial_oracle_pca, tasks, progress_label, n_cores)


# ==============================================================================
# Output
# ==============================================================================

def save_and_print_spca_sweep_results(out_dir, file_stem, num_MC, sweep_col,
                                      sweep_header, sweep_precision,
                                      axis_values, method_names, fdr_map,
                                      tpr_map, pev_map, pev_sig_map,
                                      pev_sigmix_map,
                                      include_support_metrics=True):
    """Print MC-averaged sweep results as an aligned table (console + .txt) and
    write a tidy long-format CSV (method, metric, <sweep_col>, value).

    Mirrors cpp save_and_print_spca_sweep_results: the method name gets its own
    line with the metric rows indented beneath it, and the axis header row is
    printed at the per-sweep-column precision `sweep_precision`. When
    `include_support_metrics` is False the displayed table carries only the
    cumulative-PEV rows; FDR and TPR stay in the tidy CSV either way — they
    are the data record.

    fdr_map / tpr_map / pev_map / pev_sig_map / pev_sigmix_map are dicts:
    method name -> list of values over the sweep grid.
    """
    os.makedirs(out_dir, exist_ok=True)

    indent_w, metric_w, col_w = 2, 23, 10

    lines = []
    lines.append("")
    lines.append("=" * 70)
    lines.append(f"=== T-Rex SPCA Results (averaged over {num_MC} MC trials) ===")
    lines.append("=" * 70)
    lines.append("")

    hdr = " " * indent_w + f"{sweep_header:<{metric_w}}"
    hdr += "".join(f"{v:>{col_w}.{sweep_precision}f}" for v in axis_values)
    lines.append(hdr)
    lines.append("-" * (indent_w + metric_w + col_w * len(axis_values)))

    def row(metric, values):
        r = " " * indent_w + f"{metric:<{metric_w}}"
        r += "".join(f"{v:>{col_w}.4f}" for v in values)
        return r

    for name in method_names:
        lines.append("")
        lines.append(name)                      # method name on its own line
        if include_support_metrics:
            lines.append(row("FDR", fdr_map[name]))
            lines.append(row("TPR", tpr_map[name]))
        lines.append(row("PEV (total var)", pev_map[name]))
        lines.append(row("PEV (Definition 1)", pev_sig_map[name]))
        lines.append(row("PEV (sig+mix part)", pev_sigmix_map[name]))
    lines.append("")

    text = "\n".join(lines) + "\n"
    print(text, end="", flush=True)
    txt_path = os.path.join(out_dir, file_stem + ".txt")
    with open(txt_path, "w") as f:
        f.write(text)

    # Tidy long-format CSV (method, metric, <sweep_col>, value); the axis is
    # written at fixed 6-decimal precision, matching the cpp CSVs.
    rows = []
    for name in method_names:
        for i, v in enumerate(axis_values):
            axis_str = f"{v:.6f}"
            for metric, mmap in (("FDR", fdr_map), ("TPR", tpr_map),
                                 ("PEV", pev_map), ("PEVsig", pev_sig_map),
                                 ("PEVsigmix", pev_sigmix_map)):
                rows.append({"method": name, "metric": metric,
                             sweep_col: axis_str, "value": mmap[name][i]})
    csv_path = os.path.join(out_dir, file_stem + ".csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False, float_format="%.6f")
    print(f"[Info] CSV results saved to:             {csv_path}")
    print(f"[Info] Results successfully saved to: {txt_path}\n")


def save_and_print_spca_mc_results(out_dir, file_stem, num_MC, snr_values,
                                   method_names, fdr_map, tpr_map, pev_map,
                                   pev_sig_map, pev_sigmix_map):
    """SNR-sweep convenience wrapper around save_and_print_spca_sweep_results.

    Fixes the sweep axis to the decibel SNR grid this suite reports on by
    default: CSV column "snr_db", table header "SNR(dB)", one decimal.
    """
    save_and_print_spca_sweep_results(
        out_dir, file_stem, num_MC, "snr_db", "SNR(dB)", 1, snr_values,
        method_names, fdr_map, tpr_map, pev_map, pev_sig_map, pev_sigmix_map)
