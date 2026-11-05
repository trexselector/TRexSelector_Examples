# ==============================================================================
# ts_demo_utils.py
# ==============================================================================
#
# Shared helpers for the standalone terminating-solver (tsolvers) demos.
#
# Mirrors the C++ utilities used by cpp/tsolvers/ad/demo_ts_*:
#   - datagen::SyntheticData with predictor_policy::Normal(),
#     dummygen::Distribution::Normal(), noisegen::noise_policy::Normal()
#   - cdiagnostics::print_section_header / print_talgo_demo_config /
#     print_selection / print_selection_quality (same console format)
#   - ts_perf::PartSelection, ts_perf::PathRecorder and ts_perf::display_path
#     (cpp/tsolvers/ts_perf_utils.hpp)
#
# Public functions and classes:
#   gen_synthetic_data(n, p, num_dummies, support, coefs, snr, seed)
#   print_section_header(title)
#   print_demo_config(n, p, num_dummies, T_stop, support, coefs, snr)
#   selected_variables(solver)
#   print_selection(solver, true_support)
#   print_selection_quality(solver, true_support)
#   display_path(path), output_dir(demo_dir)
#   PartSelection(argv), PathRecorder(stem, out_dir)
#
# ==============================================================================

import ctypes
import os
import sys

import numpy as np

from trexselector.utils import (
    compute_fdp,
    compute_precision,
    compute_recall,
    compute_tpp,
    f1_score,
)


def _line_buffered_output():
    """
    Line-buffer Python's and the C library's stdout, so the log lines of the
    C++ backend appear in order when the output goes to a file or a pipe.
    Without it the backend's buffer is written only when the process exits.
    """
    sys.stdout.reconfigure(line_buffering=True)
    try:
        libc = ctypes.CDLL(None)
        name = "__stdoutp" if sys.platform == "darwin" else "stdout"
        c_stdout = ctypes.c_void_p.in_dll(libc, name)
        libc.setvbuf(c_stdout, None, 1, 0)  # 1 = _IOLBF on macOS and Linux
    except (OSError, ValueError, AttributeError):
        pass  # unknown C library: keep its default buffering


_line_buffered_output()


def gen_synthetic_data(n, p, num_dummies, support, coefs, snr=1.0, seed=None):
    """
    Synthetic regression data with explicit dummy matrix, mirroring
    C++ datagen::SyntheticData.

        X[i, j]  i.i.d. N(0, 1)        (n x p signal predictors)
        D[i, j]  i.i.d. N(0, 1)        (n x num_dummies dummies)
        y = X[:, support] @ coefs + eps,   eps ~ N(0, noise_sigma^2 I_n)

    The noise standard deviation is SNR-calibrated:

        noise_sigma = sqrt( Var(signal) / snr )

    with the same small-sample rule as C++ noisegen::calculate_noise_params():
    Bessel-corrected variance (ddof=1) for n <= 100, population variance
    (ddof=0) otherwise.

    Parameters
    ----------
    n, p, num_dummies : int
        Observations, signal predictors, dummy predictors.
    support : array-like of int
        0-based indices of active predictors (order pairs with `coefs`).
    coefs : array-like of float
        Active coefficients, positionally matched to `support`.
    snr : float
        Target signal-to-noise ratio. Default 1.0.
    seed : int or None
        Seed for np.random.default_rng (governs X, D, and noise).

    Returns
    -------
    (X, D, y) : Fortran-ordered float64 arrays of shapes
                (n, p), (n, num_dummies), (n,).
    """
    support = np.asarray(support, dtype=np.intp)
    coefs = np.asarray(coefs, dtype=np.float64)
    assert support.shape == coefs.shape, "support/coefs size mismatch"
    assert np.all(support >= 0) and np.all(support < p), "support out of range"

    rng = np.random.default_rng(seed)
    X = np.asfortranarray(rng.standard_normal((n, p)))
    D = np.asfortranarray(rng.standard_normal((n, num_dummies)))

    signal = X[:, support] @ coefs
    ddof = 1 if n <= 100 else 0
    var_sig = np.var(signal, ddof=ddof)
    noise_sigma = np.sqrt(var_sig / snr) if (snr > 0 and var_sig > 0) else 0.0
    y = signal + rng.standard_normal(n) * noise_sigma

    return X, D, y


def print_section_header(title):
    """Mirrors cdiagnostics::print_section_header."""
    bar = "=" * 60
    print(f"\n{bar}\n{title}\n{bar}")


def _num(value):
    """A number as the default C++ stream prints it (6 significant digits)."""
    return f"{value:g}"


def _braced(values):
    """A list in the {a, b, c} form of cdiagnostics::print_vector."""
    return "{" + ", ".join(str(int(v)) for v in values) + "}"


def print_demo_config(n, p, num_dummies, T_stop, support, coefs, snr):
    """Mirrors cdiagnostics::print_talgo_demo_config."""
    print("Configuration:")
    print(f"  n = {n}, p = {p}")
    print(f"  num_dummies = {num_dummies}")
    print(f"  T_stop = {T_stop}")
    print(f"  True support size = {len(support)}")
    print("  True support: " + "".join(f"{int(i)} " for i in support))
    print("  True coefficients: " + "".join(f"{_num(c)} " for c in coefs))
    print(f"  SNR = {_num(snr)}\n")


def selected_variables(solver):
    """
    Mirrors cdiagnostics::selected_variables: the variables with a nonzero
    coefficient on the last path step, as the T-Rex selector reads them.

    Returns (predictors, dummies, frozen) as 0-based indices into [X, D]. Active
    variables come first in the order of the active set; variables with a
    coefficient outside the active set (T-Stagewise keeps the coefficient of a
    variable it removes) follow in index order and are also listed in frozen.
    """
    path = np.asarray(solver.getBetaPath())
    if path.size == 0:
        return [], [], []
    eps = np.finfo(np.float64).eps
    support = {int(j) for j in np.flatnonzero(np.abs(path[:, -1]) > eps)}
    dummy_start = solver.getDummyStartIndex()
    predictors, dummies, frozen, listed = [], [], [], set()
    for j in solver.getActives():
        j = int(j)
        if j in support:
            (dummies if j >= dummy_start else predictors).append(j)
            listed.add(j)
    for j in sorted(support - listed):
        (dummies if j >= dummy_start else predictors).append(j)
        frozen.append(j)
    return predictors, dummies, frozen


def print_selection(solver, true_support):
    """
    Mirrors cdiagnostics::print_selection: selected predictors and dummies
    from the coefficient path, then the path of additions and removals with
    the same ANSI colors (true positives green, false positives red, dummies
    blue, removals yellow and magenta).
    """
    print("\nSelection:")
    predictors, dummies, frozen = selected_variables(solver)
    print(f"  Predictors: {_braced(predictors)}")
    line = f"  Dummies: {_braced(dummies)}"
    if frozen:
        line += f"\n  Frozen (coefficient kept outside the active set): {_braced(frozen)}"
    print(line)

    truth = {int(i) for i in true_support}
    dummy_start = solver.getDummyStartIndex()
    tokens = []
    for step in solver.getActions():
        for action in step:
            idx = abs(int(action)) - 1
            if action < 0:
                tokens.append(f"\033[35m-D{idx - dummy_start}\033[0m" if idx >= dummy_start
                              else f"\033[33m-v{idx}\033[0m")
            elif idx >= dummy_start:
                tokens.append(f"\033[34m+D{idx - dummy_start}\033[0m")
            elif idx in truth:
                tokens.append(f"\033[32m+v{idx}\033[0m")
            else:
                tokens.append(f"\033[31m+v{idx}\033[0m")
    print("  Path: [" + ", ".join(tokens) + "]")


def print_selection_quality(solver, true_support):
    """Mirrors cdiagnostics::print_selection_quality on the coefficient-path selection."""
    selected = selected_variables(solver)[0]
    true_support = [int(i) for i in true_support]
    true_positives = len(set(selected) & set(true_support))
    fdp = compute_fdp(selected, true_support)
    tpp = compute_tpp(selected, true_support)
    precision = compute_precision(selected, true_support)
    recall = compute_recall(selected, true_support)
    print("\nSelection Quality:")
    print(f"  True Positives:  {true_positives}")
    print(f"  False Positives: {len(selected) - true_positives}")
    print(f"  False Negatives: {len(true_support) - true_positives}")
    print(f"  FDP:             {_num(fdp)}")
    print(f"  TPP:             {_num(tpp)}")
    print(f"  Precision:       {_num(precision)}")
    print(f"  Recall:          {_num(recall)}")
    print(f"  F1 Score:        {_num(f1_score(precision, recall))}")


# ==============================================================================
# Part selection, result files and terminated solution paths
# ==============================================================================

def display_path(path):
    """Path for console output without the machine-specific prefix: '.../Python/...'."""
    path = os.path.abspath(path)
    marker = os.sep + "Python" + os.sep
    pos = path.find(marker)
    return ".../" + path[pos + 1:] if pos >= 0 else os.path.basename(path)


def output_dir(demo_dir):
    """simulation_results/data/ next to the demo script, created on demand."""
    path = os.path.join(demo_dir, "simulation_results", "data")
    os.makedirs(path, exist_ok=True)
    return path


class PartSelection:
    """
    Parts of a demo selected on the command line, as ts_perf::PartSelection:
    `demo_ts_01_tlars.py 5` runs Part 5 only, without arguments every part runs.
    """

    def __init__(self, argv=None):
        self.parts = set()
        for arg in (sys.argv[1:] if argv is None else argv):
            if not arg.isdigit() or int(arg) < 1:
                raise SystemExit(f"invalid part number: {arg}")
            self.parts.add(int(arg))

    def __call__(self, part):
        """True when `part` was selected, or when no part was given at all."""
        return not self.parts or part in self.parts


class PathRecorder:
    """
    Writes the terminated solution paths of one warm-started solver, as
    ts_perf::PathRecorder, into two tidy CSV files of the same format:

        <stem>_path.csv    : T_stop,step,index,is_dummy,beta
        <stem>_actions.csv : T_stop,step,index,is_dummy,added

    Step 0 is the all-zero start, and actions[k] of the solver leads to step
    k + 1. Indices are 0-based, and the path lists nonzero coefficients only,
    on the original scale.
    """

    def __init__(self, stem, out_dir):
        self.path_file = os.path.join(out_dir, f"{stem}_path.csv")
        self.actions_file = os.path.join(out_dir, f"{stem}_actions.csv")
        with open(self.path_file, "w") as f:
            f.write("T_stop,step,index,is_dummy,beta\n")
        with open(self.actions_file, "w") as f:
            f.write("T_stop,step,index,is_dummy,added\n")

    def record(self, solver, T_stop):
        """Append the current path and actions of `solver` after executeStep(T_stop)."""
        p = solver.getDummyStartIndex()
        path = np.asarray(solver.getBetaPath())
        with open(self.path_file, "a") as f:
            for step in range(path.shape[1]):
                f.writelines(f"{T_stop},{step},{index},{int(index >= p)},{path[index, step]:.10g}\n" for index in np.flatnonzero(path[:, step]))
        with open(self.actions_file, "a") as f:
            for k, actions in enumerate(solver.getActions()):
                for a in actions:
                    index = abs(int(a)) - 1
                    f.write(f"{T_stop},{k + 1},{index},{int(index >= p)},{int(a > 0)}\n")
