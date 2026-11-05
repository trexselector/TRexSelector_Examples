# ==============================================================================
# trex_scr_bb_common.py
# ==============================================================================
#
# Shared Monte Carlo infrastructure for the Biobank Screen-TRex Python demos
# (04-05).
#
# Mirrors the Biobank portion of the C++ demo-internal header
#   cpp/trex_selector_methods/ad/trex_screening/trex_screening_simulation_utils.hpp
#   (makeBiobankControl + save_and_print_biobank_mc_results).
#
# The biobank workflow ("Algorithm 1") screens MANY phenotypes against ONE
# shared design matrix X. Per phenotype, one Screen-TRex run yields both an
# Ordinary and a Bootstrap-CI estimated FDR. A rule whose estimate lies in
# [lower_bound_FDR, upper_bound_FDR] is used (the larger if both, a tie goes
# to Bootstrap-CI), else the classical T-Rex selector runs at target_FDR_trex.
# The fraction of phenotypes routed to each method is the "Usage %".
#
# Support sets and selected indices are 0-BASED (matching the C++ sources).
#
# As in trex_scr_common.py, Monte Carlo trials run in a ProcessPoolExecutor;
# workers are module-level, receive only picklable flat dicts, and rebuild the
# trexselector controls inside the child process.
# ==============================================================================

import os
import sys

import numpy as np
import pandas as pd

import trexselector as trex

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from trex_scr_common import (dgp_iid_snr, _make_response,  # noqa: E402
                             _get_executor)
import cell_cache  # family folder, put on sys.path by trex_scr_common

# Canonical method-name keys returned in each result's `method_used` field.
BIOBANK_METHODS = ["Screen-TRex (ordinary)",
                   "Screen-TRex (bootstrap-CI)",
                   "T-Rex (fallback)"]

# Display names used in the saved tables / figure legends (mirror cpp display_names).
BIOBANK_DISPLAY = ["Screen-TRex Ordinary: TLARS",
                   "Screen-TRex Bootstrap-CI: TLARS",
                   "T-Rex fallback: TLARS"]


# ==============================================================================
# Control factory
# ==============================================================================

def _build_biobank_ctrl(d):
    """Build a trex.BiobankScreenTRexControl from a flat dict (mirrors makeBiobankControl()).

    BiobankScreenTRexControl nests ScreenTRexControlParameter as
    trex_screen_ctrl, which in turn nests its own trex_ctrl. Rebuilt inside
    worker processes.
    """
    ctrl = trex.BiobankScreenTRexControl()
    ctrl.lower_bound_FDR = float(d.get("lower_bound_FDR", 0.05))
    ctrl.upper_bound_FDR = float(d.get("upper_bound_FDR", 0.15))
    ctrl.target_FDR_trex = float(d.get("target_FDR_trex", 0.10))

    ctrl.trex_screen_ctrl.trex_method = trex.ScreenTRexMethod.TREX
    ctrl.trex_screen_ctrl.R_boot = int(d.get("R_boot", 1000))
    ctrl.trex_screen_ctrl.ci_grid_step = float(d.get("ci_grid_step", 0.001))

    tc = ctrl.trex_screen_ctrl.trex_ctrl
    tc.K = int(d.get("K", 20))
    tc.max_dummy_multiplier = 10
    tc.use_max_T_stop = True
    tc.dummy_distribution = trex.DummyDistribution.normal()
    tc.lloop_strategy = trex.LLoopStrategy.STANDARD
    tc.solver_type = trex.SolverTypeForTRex.TLARS
    tc.use_memory_mapping = bool(d.get("use_mmap", False))
    return ctrl


# ==============================================================================
# ProcessPoolExecutor workers (module-level)
# ==============================================================================

def _biobank_single_worker(args):
    """One single-phenotype Algorithm-1 trial. Returns a list of one record.

    Support drawn from an RNG stream offset from the trial seed (by 500000),
    matching the cpp isolation of the support draw from design/noise.
    """
    import trexselector as _tsn
    from trexselector.utils import compute_fdp as _fdp, compute_tpp as _tpp

    n, p = args["n"], args["p"]
    support_size, snr = args["support_size"], args["snr"]
    seed = args["trial_seed"]

    rng_sup = np.random.default_rng(seed + 500000)
    support = rng_sup.choice(p, size=support_size, replace=False)
    dat = dgp_iid_snr(n, p, support, [1.0] * support_size, snr, seed)

    ctrl = _build_biobank_ctrl(args)
    res = _tsn.TRexBiobankScreeningSelector(
        dat["X"], dat["y"], bio_ctrl=ctrl, seed=-1, verbose=False).select()

    tsup = list(dat["true_support"])
    return [_record(res, tsup, _fdp, _tpp)]


def _biobank_multi_worker(args):
    """One multi-phenotype Algorithm-1 trial (q phenotypes sharing one X).

    Returns a list of q records. All phenotypes share the design matrix X drawn
    with x_seed; each phenotype gets its own random support and noise draw.
    """
    import trexselector as _tsn
    from trexselector.utils import compute_fdp as _fdp, compute_tpp as _tpp

    n, p, q = args["n"], args["p"], args["q"]
    support_size, snr = args["support_size"], args["snr"]
    x_seed = args["trial_seed"]
    sup_seed_base = args["sup_seed_base"]

    rng_x = np.random.default_rng(x_seed)
    X = rng_x.standard_normal((n, p))                       # shared design matrix
    true_supports = []
    Y = np.zeros((n, q))
    for k in range(q):
        rng_k = np.random.default_rng(sup_seed_base + k)
        sup = np.sort(rng_k.choice(p, size=support_size, replace=False))
        true_supports.append(sup)
        beta = np.zeros(p)
        beta[sup] = 1.0
        Y[:, k] = _make_response(X, beta, snr, rng_k)

    ctrl = _build_biobank_ctrl(args)
    results = _tsn.TRexBiobankScreeningSelector(
        X, Y, bio_ctrl=ctrl, seed=-1, verbose=False).select()

    return [_record(results[k], list(true_supports[k]), _fdp, _tpp)
            for k in range(q)]


def _record(res, true_support, fdp_fn, tpp_fn):
    """Build one picklable per-phenotype record from a BiobankScreenTRexResult."""
    sel = list(res.selected_indices)
    return {
        "method_used":   res.method_used,
        "fdp":           fdp_fn(sel, true_support),
        "tpp":           tpp_fn(sel, true_support),
        "est_ordinary":  float(res.estimated_FDR_screen_ordinary),
        "est_bootstrap": float(res.estimated_FDR_screen_bootstrap),
    }


def run_mc_biobank(worker, args_list, max_workers):
    """Run a list of biobank trials in parallel; return the flat list of records.

    Each worker returns a list of per-phenotype records (length 1 for the
    single-phenotype worker, q for the multi-phenotype worker).
    """
    chunks = list(_get_executor(max_workers, len(args_list)).map(worker, args_list))
    records = []
    for chunk in chunks:
        records.extend(chunk)
    return records


def accumulate_snr(records, target_FDR_trex):
    """Aggregate one SNR level's records into per-method statistics.

    FDR/TPR are conditional on which method a phenotype was routed to; the
    estimated-FDR series are unconditional (mirrors the cpp accumulation).

    Returns dict keyed by BIOBANK_METHODS with fields
    usage (fraction in [0,1]), fdp, tpp, est.
    """
    total = len(records)
    cnt = {m: 0 for m in BIOBANK_METHODS}
    fdp = {m: 0.0 for m in BIOBANK_METHODS}
    tpp = {m: 0.0 for m in BIOBANK_METHODS}
    est = {m: 0.0 for m in BIOBANK_METHODS}

    for r in records:
        m = r["method_used"]
        cnt[m] += 1
        fdp[m] += r["fdp"]
        tpp[m] += r["tpp"]
        est["Screen-TRex (ordinary)"]     += r["est_ordinary"]
        est["Screen-TRex (bootstrap-CI)"] += r["est_bootstrap"]
        est["T-Rex (fallback)"]           += target_FDR_trex

    out = {}
    for m in BIOBANK_METHODS:
        out[m] = {
            "usage": (cnt[m] / total) if total else 0.0,
            "fdp":   (fdp[m] / cnt[m]) if cnt[m] else 0.0,     # conditional mean
            "tpp":   (tpp[m] / cnt[m]) if cnt[m] else 0.0,     # conditional mean
            "est":   (est[m] / total) if total else 0.0,       # unconditional mean
        }
    return out


def run_biobank_snr_cell(worker, args_list, max_workers, target_FDR_trex):
    """One SNR level as a resumable grid cell: run_mc_biobank + accumulate_snr,
    restored from the cell cache after a restart. Same return as accumulate_snr."""
    fields = ("usage", "fdp", "tpp", "est")

    def compute():
        stats = accumulate_snr(run_mc_biobank(worker, args_list, max_workers), target_FDR_trex)
        return {f"m{i}.{k}": stats[m][k] for i, m in enumerate(BIOBANK_METHODS) for k in fields}

    key = f"biobank|{worker.__name__}|{cell_cache.digest(args_list, target_FDR_trex)}"
    flat = cell_cache.cached(key, compute)
    return {m: {k: flat[f"m{i}.{k}"] for k in fields} for i, m in enumerate(BIOBANK_METHODS)}


# ==============================================================================
# MC results: save + print (TXT + tidy pandas CSV, with Usage % rows)
# ==============================================================================
#
# Layout mirrors save_and_print_biobank_mc_results(): method name on its own
# line, metric rows indented beneath it; Usage printed to 1 dp, the rest to 4.
#
_INDENT_W = 2
_METRIC_W = 23
_COL_W = 10


def save_and_print_biobank_mc(num_MC, out_dir, file_stem, snr_values,
                              display_names, fdp_map, tpp_map, est_fdr_map,
                              usage_map):
    """Save + print a Biobank MC table with per-method Usage (%) rows.

    fdp_map/tpp_map/est_fdr_map/usage_map are keyed by display_names; each value
    is a per-SNR list.
    """
    os.makedirs(out_dir, exist_ok=True)
    lines = []
    lines.append("")
    lines.append("=" * 70)
    lines.append(f"=== Biobank Screen-TRex Results (averaged over {num_MC} "
                 f"Monte Carlo runs) ===")
    lines.append("=" * 70)
    lines.append("")

    hdr = " " * _INDENT_W + f"{'SNR':<{_METRIC_W}}"
    hdr += "".join(f"{s:>{_COL_W}.2f}" for s in snr_values)
    lines.append(hdr)
    sep_w = _INDENT_W + _METRIC_W + _COL_W * len(snr_values)
    lines.append("-" * sep_w)

    def _row(label, vec, dp):
        s = " " * _INDENT_W + f"{label:<{_METRIC_W}}"
        s += "".join(f"{v:>{_COL_W}.{dp}f}" for v in vec)
        return s

    for nm in display_names:
        usage_pct = [u * 100.0 for u in usage_map[nm]]
        lines.append("")
        lines.append(nm)                       # method name on its own line
        lines.append(_row("Usage (%)", usage_pct,       1))
        lines.append(_row("FDR",       fdp_map[nm],      4))
        lines.append(_row("TPR",       tpp_map[nm],      4))
        lines.append(_row("Est. FDR",  est_fdr_map[nm],  4))
    lines.append("")

    print("\n".join(lines))
    txt_path = os.path.join(out_dir, file_stem + ".txt")
    with open(txt_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"[Info] TXT results saved to: {txt_path}")

    rows = []
    for nm in display_names:
        for i, snr in enumerate(snr_values):
            rows.append({"method": nm, "metric": "Usage",    "snr": snr, "value": usage_map[nm][i]})
            rows.append({"method": nm, "metric": "FDR",      "snr": snr, "value": fdp_map[nm][i]})
            rows.append({"method": nm, "metric": "TPR",      "snr": snr, "value": tpp_map[nm][i]})
            rows.append({"method": nm, "metric": "Est. FDR", "snr": snr, "value": est_fdr_map[nm][i]})
    csv_path = os.path.join(out_dir, file_stem + ".csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"[Info] CSV results saved to: {csv_path}\n")

# ==============================================================================
# End of trex_scr_bb_common.py
# ==============================================================================
