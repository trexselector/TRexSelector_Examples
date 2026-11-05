# ==============================================================================
# demo_ts_perf_01_fdr_tpr_vs_T.py
# ==============================================================================
#
# FDR and TPR of the terminated solution paths vs. the number of included
# dummies T, for all demo T-solvers (simulation study of the R tlars vignette).
#
# Mirrors cpp/tsolvers/ad/demo_ts_perf_01_fdr_tpr_vs_T/demo_ts_perf_01_fdr_tpr_vs_T.cpp:
# n = 100, p = 300, 10 true coefficients equal to 3 on a fixed random support,
# noise sigma about 1, L = p fresh standard normal dummies per run. Each run
# scales X and D once and hands them to all solvers as readers; every solver
# is warm-started through the T grid {1, 2, 5, 10, 20, 50, 100}. FDR and TPR are
# the mean FDP and TPP over the runs; a variable counts as selected when its
# coefficient is nonzero, as in the T-Rex selector. After each T the selected
# original variables (dummies left out) are refit by least squares, and the
# refit is scored on a fresh test set of n_test rows.
#
# Usage: demo_ts_perf_01_fdr_tpr_vs_T.py [num_MC] [workers]   (default 500, 6)
#
# ==============================================================================

import os
import sys
import time

import numpy as np

from trexselector.utils import compute_fdp, compute_tpp

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.dirname(os.path.dirname(_THIS_DIR))  # tsolvers/, the shared helpers
if _PARENT_DIR not in sys.path:
    sys.path.insert(0, _PARENT_DIR)

from ts_demo_utils import (
    display_path,
    gen_synthetic_data,
    output_dir,
    print_section_header,
)
from ts_perf_solvers import demo_solvers, file_suffix
from ts_perf_utils import (
    draw_support,
    parse_positive,
    run_parallel,
    included_dummies,
    selected_predictors,
)

# Setup
n, p = 100, 300
num_dummies = p
num_active = 10
beta_active = 3.0
snr = 90.0                    # signal variance 10 * 3^2 over noise variance 1
support_seed = 12345
data_seed_offset = 1000       # run mc uses seed data_seed_offset + mc
n_test = 1000                 # rows of the test set of every run
test_seed_offset = 2_000_000  # test set of run mc: seed test_seed_offset + mc
T_grid = [1, 2, 5, 10, 20, 50, 100]
FIELDS = ["fdp", "tpp", "selected", "steps", "reached", "ms", "rmse", "coef_err"]

SUPPORT = draw_support(p, num_active, support_seed)
COEFS = [beta_active] * num_active
BETA_TRUE = np.zeros(p)
BETA_TRUE[SUPPORT] = beta_active


def refit_errors(X, y, X_test, y_test, selected, beta):
    """
    Errors of the least-squares refit with intercept on the selected original
    variables: prediction RMSE on the test set and ||beta_hat - beta||_2. The
    minimum-norm solution also covers selections with as many variables as rows.
    """
    y_mean = y.mean()
    beta_hat = np.zeros(beta.size)
    intercept = y_mean
    if selected:
        Xs = X[:, selected]
        x_mean = Xs.mean(axis=0)
        b = np.linalg.lstsq(Xs - x_mean, y - y_mean, rcond=None)[0]
        beta_hat[selected] = b
        intercept = y_mean - x_mean @ b
    prediction = X_test @ beta_hat + intercept
    return float(np.sqrt(np.mean((y_test - prediction) ** 2))), float(np.linalg.norm(beta_hat - beta))


def run_one(mc):
    """One run: all solvers warm-started along the T grid on the same data."""
    X, D, y = gen_synthetic_data(n, p, num_dummies, SUPPORT, COEFS, snr, seed=data_seed_offset + mc)
    X_raw, y_raw = X.copy(), y.copy()
    X_test, _, y_test = gen_synthetic_data(n_test, p, 0, SUPPORT, COEFS, snr, seed=test_seed_offset + mc)
    oracle = refit_errors(X_raw, y_raw, X_test, y_test, SUPPORT, BETA_TRUE)

    # Center and scale once in front; every solver below is a precentered reader
    from trexselector.ml_methods import LpNormScaler, NormType
    LpNormScaler(NormType.L2, center=True).fit_transform_inplace(X)
    LpNormScaler(NormType.L2, center=True).fit_transform_inplace(D)
    y -= y.mean()

    results = []
    for spec in demo_solvers():
        r = {f: [] for f in FIELDS}
        try:
            t0 = time.perf_counter()
            untimed = 0.0  # reached count, kept out of the times
            solver = spec.make(X, D, y)
            solver.setTieSeed(data_seed_offset + mc)
            selections = []
            for T in T_grid:
                solver.executeStep(T, True)  # warm start from the previous T
                selected = selected_predictors(solver, p)
                r["fdp"].append(compute_fdp(selected, SUPPORT))
                r["tpp"].append(compute_tpp(selected, SUPPORT))
                r["selected"].append(float(len(selected)))
                r["steps"].append(float(solver.getNumSteps()))
                c0 = time.perf_counter()
                r["reached"].append(1.0 if included_dummies(solver, p) >= T else 0.0)
                untimed += time.perf_counter() - c0
                r["ms"].append(1e3 * (time.perf_counter() - t0 - untimed))
                selections.append(selected)
            # Refits after the timed path, so they stay out of the solver times
            for selected in selections:
                rmse, coef_err = refit_errors(X_raw, y_raw, X_test, y_test, selected, BETA_TRUE)
                r["rmse"].append(rmse)
                r["coef_err"].append(coef_err)
        except (RuntimeError, ValueError):  # a failing solver is left out of its means
            r = None
        results.append(r)
    return results, oracle


def std_error(v):
    """Standard error of the mean (Bessel-corrected sd over sqrt(size))."""
    return float(np.std(v, ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0


def main():
    num_MC = parse_positive(sys.argv, 1, 500)
    workers = parse_positive(sys.argv, 2, 6)
    solvers = demo_solvers()
    S, K = len(solvers), len(T_grid)

    print()
    print_section_header("T-Solver Performance: FDR and TPR vs. T")
    print(f"n = {n}, p = {p}, L = p = {num_dummies}, true coefficients: {num_active} x {beta_active:g}, "
          f"SNR = {snr:g}")
    print("True support: " + " ".join(str(j) for j in SUPPORT))
    print("T grid: " + " ".join(str(T) for T in T_grid))
    print(f"{S} solvers, {num_MC} runs on {workers} worker processes (single-threaded solvers)\n")

    t_start = time.perf_counter()
    runs = run_parallel(run_one, range(num_MC), workers)
    wall = time.perf_counter() - t_start

    failures = sum(r is None for results, _ in runs for r in results)
    oracle_rmse = np.mean([o[0] for _, o in runs])
    oracle_coef = np.mean([o[1] for _, o in runs])

    def column(s, k, field):
        return [results[s][field][k] for results, _ in runs if results[s] is not None]

    out = []

    def table(title, field, scale, precision):
        out.append(title)
        out.append(f"{'solver':<22}" + "".join(f"{'T=' + str(T):>8}" for T in T_grid))
        out.append("-" * (22 + 8 * K))
        for s, spec in enumerate(solvers):
            out.append(f"{spec.name:<22}" + "".join(f"{scale * np.mean(column(s, k, field)):>8.{precision}f}"
                                                    for k in range(K)))
        out.append("")

    table("FDR (%)", "fdp", 100.0, 1)
    table("TPR (%)", "tpp", 100.0, 1)
    table("Runs that reached T dummies (%)", "reached", 100.0, 0)
    table("Average number of selected variables", "selected", 1.0, 1)
    table("Average number of steps", "steps", 1.0, 1)
    table("Average time from construction to T (ms, cumulative)", "ms", 1.0, 2)
    table("Prediction RMSE of the least-squares refit on the selected variables (test set)", "rmse", 1.0, 2)
    table("Coefficient error of the refit, ||beta_hat - beta||_2", "coef_err", 1.0, 2)
    out.append(f"Least-squares refit on the true support: prediction RMSE {oracle_rmse:.2f}, "
               f"coefficient error {oracle_coef:.2f}")
    out.append("")
    out.append(f"Failed solver runs: {failures}")
    out.append(f"Wall time: {wall:.1f} s")
    text = "\n".join(out) + "\n"
    print(text, end="")

    stem = f"fdr_tpr_vs_T_mc{num_MC}{file_suffix()}"  # suffixed for a TREX_SOLVERS selection
    data_dir = output_dir(_THIS_DIR)
    csv_path = os.path.join(data_dir, stem + ".csv")
    with open(csv_path, "w") as f:
        f.write("solver,family,T,FDR,FDR_se,TPR,TPR_se,avg_selected,avg_steps,reached,avg_ms,"
                "rmse,rmse_se,coef_err,coef_err_se,oracle_rmse,oracle_coef_err\n")
        for s, spec in enumerate(solvers):
            for k, T in enumerate(T_grid):
                fdp, tpp = column(s, k, "fdp"), column(s, k, "tpp")
                rmse, coef = column(s, k, "rmse"), column(s, k, "coef_err")
                row = [np.mean(fdp), std_error(fdp), np.mean(tpp), std_error(tpp),
                       np.mean(column(s, k, "selected")), np.mean(column(s, k, "steps")),
                       np.mean(column(s, k, "reached")), np.mean(column(s, k, "ms")),
                       np.mean(rmse), std_error(rmse), np.mean(coef), std_error(coef), oracle_rmse, oracle_coef]
                f.write(f"{spec.name},{spec.family},{T}," + ",".join(f"{v:.6g}" for v in row) + "\n")
    txt_path = os.path.join(data_dir, stem + ".txt")
    with open(txt_path, "w") as f:
        f.write(text)

    print(f"\nResults written to:\n  {display_path(csv_path)}\n  {display_path(txt_path)}\n")


if __name__ == "__main__":
    main()
