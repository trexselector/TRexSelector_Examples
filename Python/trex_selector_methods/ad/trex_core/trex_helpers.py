# ==============================================================================
# trex_helpers.py
# ==============================================================================
#
# Shared helper functions for TRex single-run demos.
# Mirrors the single-run printing/saving helpers in
# cpp/trex_selector_methods/ad/trex_core/trex_sim_utils.hpp (sparse printers,
# print_results, save_selection_csv) and
# R/trex_selector_methods/ad/trex_core/trex_sim_utils.R.
#
# All printers accept an optional `out` callback for dual console + file
# output (mirrors the C++ PrintFn pattern); when None, output goes to the
# console only.
#
# ==============================================================================

import numpy as np


def _console_out(text):
    print(text, end="")


def make_dual_out(file_handle):
    """
    Return a PrintFn-style callback writing to both console and `file_handle`.

    Mirrors the C++ `print_dual` lambda in trex_sim_utils.hpp.
    """
    def _out(text):
        print(text, end="")
        file_handle.write(text)
    return _out


# ==============================================================================
# Sparse printing of the selector's per-variable structures
# (mirrors trex_sim::print_sparse_vector / print_sparse_matrix)
# ==============================================================================

def print_sparse_vector(label, vec, out=None, eps=1e-12):
    """
    Print only the nonzero entries of a p-vector, with their indices.

    Phi_prime carries one entry per variable, so a dense dump puts p numbers
    on a single line of which typically ~0.1% are nonzero. Listing the
    nonzeros keeps every value a reader would use — the zeros are implied by
    absence. Falls back to the dense form when the vector is not actually
    sparse, so the sparse view can never be the longer of the two.
    """
    if out is None:
        out = _console_out
    vec = np.asarray(vec, dtype=np.float64).ravel()
    n   = vec.size
    nz  = np.flatnonzero(np.abs(vec) > eps)

    out(f"\n{label}: {len(nz)} of {n} entries nonzero\n")

    if len(nz) * 4 > n:                      # dense enough: keep the plain dump
        out("  " + "  ".join(f"{x:.6f}" for x in vec) + "\n")
        return

    per_line = 5
    line = []
    for count, i in enumerate(nz, start=1):
        line.append(f"{f'[{i}] {vec[i]:.6f}':<20}")
        if count % per_line == 0:
            out("  " + "  ".join(line) + "\n")
            line = []
    if line:
        out("  " + "  ".join(line) + "\n")


def print_sparse_matrix(label, mat, out=None, eps=1e-12):
    """
    Print only the nonzero entries of a matrix, row by row.

    Same rationale as print_sparse_vector(): the Phi matrix is (T x p), so
    each of its rows is as wide as the variable count.
    """
    if out is None:
        out = _console_out
    mat = np.asarray(mat, dtype=np.float64)
    rows, cols = mat.shape
    total = rows * cols
    nnz   = int(np.count_nonzero(np.abs(mat) > eps))

    out(f"\n{label}: {rows} x {cols}, {nnz} of {total} entries nonzero\n")

    if nnz * 4 > total:                      # dense enough: keep the plain dump
        out(str(np.round(mat, 4)) + "\n")
        return

    for r in range(rows):
        nz = np.flatnonzero(np.abs(mat[r]) > eps)
        if len(nz) == 0:
            out(f"  row {r}:  (all zero)\n")
        else:
            cells = "  ".join(f"[{c}] {mat[r, c]:.4f}" for c in nz)
            out(f"  row {r}:  {cells}\n")


# ==============================================================================
# Selection-result block (mirrors trex_sim::print_results)
# ==============================================================================

def print_selection_results(selected, true_support, sel, out=None):
    """
    Print all selection outputs (indices, metrics, matrices).

    Mirrors trex_sim::print_results() in trex_sim_utils.hpp: selected indices,
    FDP/TPP, sparse Phi_prime, sparse Phi, dense FDP_hat and R matrices, and
    the voting grid.

    Parameters
    ----------
    selected     : iterable of int  0-based selected indices.
    true_support : iterable of int  0-based true active indices.
    sel          : trexselector.TRexSelector after .select().
    out          : callable or None  PrintFn sink (console-only when None).
    """
    from trex_sim_common import compute_fdp, compute_tpp  # local import avoids circular dep

    if out is None:
        out = _console_out

    selected     = list(selected)
    true_support = list(true_support)

    out("Selected indices: " + " ".join(str(x) for x in selected) + " \n")

    fdp_val = compute_fdp(selected, true_support)
    tpp_val = compute_tpp(selected, true_support)
    out(f"False Discovery Proportion (FDP): {fdp_val:.4f}\n")
    out(f"True Positive Proportion (TPP):   {tpp_val:.4f}\n")

    print_sparse_vector("Adjusted Relative Occurrences (Phi_prime)",
                        sel.phi_prime, out)

    print_sparse_matrix("Phi Matrix (Phi)", sel.phi_mat, out)

    out("\nEstimated FDP Matrix (FDP_hat):\n")
    out(str(np.round(np.asarray(sel.fdp_hat_mat), 4)) + "\n")

    out("\nR Matrix (R):\n")
    out(str(np.round(np.asarray(sel.r_mat), 4)) + "\n")

    out("\nVoting Grid:\n")
    out("  " + "  ".join(f"{x:.4f}" for x in np.asarray(sel.voting_grid)) + "\n")


def print_single_run_result(scenario_name, dat, sel, tFDR, out=None):
    """
    Print a formatted single-run result block.

    A data/calibration header (mirrors R .print_single_run()) followed by the
    C++-style selection-result block of trex_sim::print_results() — with the
    per-variable Phi_prime / Phi structures printed sparsely.

    Parameters
    ----------
    scenario_name : str
        Label printed in the header (e.g. "Part 1 — i.i.d. Gaussian [TLARS]").
    dat           : dict
        Output of dgp_gauss_snr() with keys X, y, beta, true_support, n, p, s, snr.
    sel           : trexselector.TRexSelector
        Selector instance after .select() has been called.
    tFDR          : float
        Target FDR level.
    out           : callable or None
        PrintFn sink for dual console + file output (console-only when None).
    """
    if out is None:
        out = _console_out

    true_support = dat["true_support"]          # 0-based np.ndarray
    s            = dat["s"]
    n            = dat["n"]
    p            = dat["p"]

    selected  = list(sel.selected_indices)      # 0-based list
    n_sel     = len(selected)
    true_set  = set(true_support.tolist())
    sel_set   = set(selected)
    n_tp      = len(sel_set & true_set)
    n_fp      = n_sel - n_tp

    sep  = "=" * 70
    dash = "-" * 70

    out(sep + "\n")
    out(f"  {scenario_name}\n")
    out(dash + "\n")
    out(f"  Data:  n = {n},  p = {p},  s = {s},  tFDR = {tFDR:.2f}\n")
    out(f"         True support (0-based):   {{{', '.join(str(x) for x in true_support)}}}\n")
    out(f"         Active coefficients:       "
        f"{{{', '.join(f'{x:.3f}' for x in dat['beta'][true_support])}}}\n")
    out(dash + "\n")
    out(f"  Calibration:  T_stop = {sel.T_stop},  L = {sel.L}\n")
    out(f"  Selection:    {n_sel} selected  |  TP = {n_tp}  FP = {n_fp}\n")
    out(dash + "\n")

    print_selection_results(selected, true_support.tolist(), sel, out)

    out(sep + "\n\n")


def save_selection_csv(out_dir, stem, p, phi_prime, selected_0based, true_support_0based):
    """
    Save a per-variable selection CSV.

    Mirrors C++ trex_sim::save_selection_csv() and the R
    .print_and_save_d02() CSV section.

    One row per predictor with columns:
        variable_index  (0-based int, matches C++ output)
        phi_prime       (float, 6 d.p.)
        selected        (0 or 1)
        true_positive   (0 or 1)

    Parameters
    ----------
    out_dir              : str        Output directory (created if absent).
    stem                 : str        Base file name without extension.
    p                    : int        Total number of predictors.
    phi_prime            : array-like Shape (p,) adjusted relative occurrences.
    selected_0based      : iterable   0-based selected predictor indices.
    true_support_0based  : iterable   0-based true active predictor indices.
    """
    import os
    import pandas as pd

    os.makedirs(out_dir, exist_ok=True)

    phi = np.asarray(phi_prime, dtype=np.float64)
    sel_set  = set(int(x) for x in selected_0based)
    true_set = set(int(x) for x in true_support_0based)

    df = pd.DataFrame({
        "variable_index": np.arange(p, dtype=np.int64),
        "phi_prime":      np.round(phi, 6),
        "selected":       np.array([1 if i in sel_set  else 0 for i in range(p)], dtype=np.int8),
        "true_positive":  np.array([1 if i in true_set else 0 for i in range(p)], dtype=np.int8),
    })

    csv_path = os.path.join(out_dir, f"{stem}.csv")
    df.to_csv(csv_path, index=False)
    print(f"[Info] CSV saved to: {csv_path}\n")
