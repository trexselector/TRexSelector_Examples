# ==============================================================================
# demo_trex_spca_02_mc_sim_pev.py
# ==============================================================================
#
# Cumulative explained variance of T-Rex SPCA under four sweeps.
# Python counterpart of cpp demo_trex_spca_02_mc_sim_pev.cpp.
#
# DGP: sparse M-factor model — X = Z * V^T + E.
#      n=50, p=100, M=3 latent factors, overlap_pool_size=30 (shared candidate
#      index pool across factors). Unlike demo 01 (which mirrors Fig. 2 of the
#      reference paper at p1=5), this demo mirrors Fig. 3 and therefore uses
#      p1=10 as its default number of true active loadings per factor.
#
# The headline metric is the PEV of Definition 1 of the reference paper: the
# cumulative ADJUSTED EV divided by the Signal + Mixed EV OF THE DATA — the
# total variance carried by the true active variables (union of the factor
# supports), a denominator FIXED per dataset and SHARED by all methods. The
# ratio is deliberately uncapped: a method that spends loadings on null
# variables (ordinary PCA above all) accumulates variance the denominator does
# not contain and climbs past 100 %, which the paper reads as inferior
# performance. The FDR-controlled T-Rex variants stay below 100 %, with
# plateaus of about 80-83 % (ActiveSet) and 86-96 % (Thresholded).
#
# NOTE the cpp demo originally normalized by each method's OWN Signal + Mixed
# EV. That self-normalized reading is structurally >= 100 % for null-capturing
# methods, INVERTS the SNR trend, and cannot reproduce the rising curves of
# the published Fig. 3 — it is why the first Fig.-3 reproduction attempt
# failed. The convention used now follows the legacy R reference
# (demo_trex_spca_03_fig3.R, denominator mode "active"). It reproduces the
# Fig. 3(b) rise from about 40 %, but no constructible denominator matches
# all published Fig. 3 anchors. The R script documents the open conventions.
#
# Alongside it, two companions are reported: the conventional total-variance
# PEV (bounded by 1), and PEVsigmix — the method's own Sig+Mix part over the
# same data-level denominator. Read for OrdPCA, the latter is the paper's
# "Ordinary PCA (Sig + Mix)" reference curve, which levels off near 108 %
# while plain OrdPCA keeps climbing on Null EV.
#
# Four sweeps, each varying ONE axis with the others held at their defaults
# (# PCs = 3, SNR = 0 dB, p1 = 10, tFDR = 0.10):
#
#   Part 1 — # extracted PCs   {1, 3, 5, 10, 20, 30, 40, 49}
#            The data still carries exactly M = 3 true factors; only the number
#            of components the methods EXTRACT grows. Components beyond the
#            third are null by construction, so this sweep shows how fast a
#            method starts accumulating variance it should not be claiming.
#   Part 2 — SNR (dB)          {-10, -7, -5, -3, 0, 3, 5, 7, 10}
#   Part 3 — # true active loadings p1  {1, 5, 10, 15, 20, 25, 30}
#   Part 4 — target FDR        {0.01, 0.05, 0.10, ..., 0.50}
#            Only the T-Rex variants respond to this axis; the two PCA
#            baselines are tFDR-blind and are reported as flat references.
#   Part 5 — FDR/TPR heatmaps over the full (target FDR x SNR) grid, PC1
#            support recovery only, for two selector responses: the plug-in
#            ordinary PC1 (what T-Rex SPCA regresses on) and the true factor
#            scores z_1 (oracle response). Isolates the mechanism behind the
#            realized-FDR overshoot at loose targets; see run_fdr_tpr_heatmap.
#
# Part selection: TREX_SPCA_PARTS (default "12345"), e.g. TREX_SPCA_PARTS=5
# reruns only the heatmap sweep. TREX_SPCA_NUM_MC overrides the MC count.
#
# Methods compared (identical to demo 01):
#   1. OrdPCA             — ordinary PCA (baseline; no sparsity constraint)
#   2. OraclePCA          — thresholded PCA with known support cardinality p1
#   3. TRexSPCA-EN-Act    — T-Rex+GVS(EN/TENET, Gram-based) + ActiveSet
#   4. TRexSPCA-ENaug-Act — T-Rex+GVS(EN/TENETAug, augmented) + ActiveSet
#   5. TRexSPCA-EN-Thr    — T-Rex+GVS(EN/TENET, Gram-based) + Thresholded
#   6. TRexSPCA-ENaug-Thr — T-Rex+GVS(EN/TENETAug, augmented) + Thresholded
#
# Metrics (num_MC-averaged; TPR/FDR on PC1 support only — see evaluate_spca):
#   - FDR     — false discovery rate of PC1's estimated loading support
#   - TPR     — true positive rate  of PC1's estimated loading support
#   - PEV     — cumulative adjusted EV / total variance of X (bounded by 1)
#   - PEVsig  — cumulative adjusted EV / (Signal EV + Mixed EV), Definition 1
#
# Support indices are 0-based (Python convention).
#
# Usage: python demo_trex_spca_02_mc_sim_pev.py [num_worker_cores]
# ==============================================================================

import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.dirname(_THIS_DIR)
for _p in (_THIS_DIR, _PARENT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import trexselector as trex  # noqa: E402
from trexselector.ml_methods import PCA  # noqa: E402

from trex_spca_sim_utils import (N_CORES, center_columns,  # noqa: E402
                                 resolve_workers,
                                 generate_sparse_factor_model, get_executor,
                                 run_mc_trials_oracle_pca, run_mc_trials_pca,
                                 run_mc_trials_trex_spca,
                                 save_and_print_spca_sweep_results)
import cell_cache  # family folder, put on sys.path by trex_spca_sim_utils

# Optional CLI worker count, resolved like TREX_NUM_WORKERS (0 = all usable cores).
N_WORKERS = resolve_workers(sys.argv[1]) if len(sys.argv) > 1 else N_CORES

# Ordered method names, shared by every part of this demo.
METHOD_NAMES = ["OrdPCA", "OraclePCA",
                "TRexSPCA-EN-Act", "TRexSPCA-ENaug-Act",
                "TRexSPCA-EN-Thr", "TRexSPCA-ENaug-Thr"]


# ==============================================================================
# Shared sweep driver
# ==============================================================================

def make_point(snr=0.0, p1=10, tFDR=0.10, num_comp=0, axis_value=0.0):
    """One grid point of a sweep: everything a single run needs to vary.

    num_comp = 0 means "extract as many components as there are true factors".
    axis_value is the value plotted on the sweep axis.
    """
    return dict(snr=snr, p1=p1, tFDR=tFDR, num_comp=num_comp,
                axis_value=axis_value)


def run_sweep(cfg, points, sweep_col, sweep_header, sweep_precision, banner,
              stem, pev_only_table=False):
    """Run one sweep over the supplied grid points and save table + CSV.

    cfg carries the base configuration (n, p, M, num_MC, base_seed, lambda_2);
    each point in `points` carries the varied parameter(s). `sweep_col` is the
    CSV column name for the sweep axis (e.g. "n_pcs"), `sweep_header` the
    table header label, and `sweep_precision` the decimals used when printing
    the axis header row. With pev_only_table=True the displayed table carries
    only the cumulative-PEV rows — parts 2-4 are restricted to the cumulative
    PEV of the first three PCs; FDR/TPR remain in the CSV as the data record.
    """
    print("================================================================")
    print(f"  {banner}")
    print("================================================================")
    print(f"  n={cfg['n']}  p={cfg['p']}  M={cfg['M']}"
          f"  num_MC={cfg['num_MC']}  ({N_WORKERS} workers)\n")

    fdr_map = {name: [] for name in METHOD_NAMES}
    tpr_map = {name: [] for name in METHOD_NAMES}
    pev_map = {name: [] for name in METHOD_NAMES}
    pev_sig_map = {name: [] for name in METHOD_NAMES}
    pev_sigmix_map = {name: [] for name in METHOD_NAMES}

    def store(name, r):
        fdr_map[name].append(r["avg_fdr"])
        tpr_map[name].append(r["avg_tpr"])
        pev_map[name].append(r["avg_pev"])
        pev_sig_map[name].append(r["avg_pev_sig"])
        pev_sigmix_map[name].append(r["avg_pev_sigmix"])

    axis_values = [pt["axis_value"] for pt in points]

    for si, pt in enumerate(points):
        # Distinct seed band per grid point, derived from the varied value so
        # that re-running a single point reproduces its own data stream. The
        # cpp demo stores this offset in an unsigned int, which wraps negative
        # values (e.g. the SNR sweep's -10 dB point); mirror that wrap
        # explicitly — numpy rejects negative seeds.
        offset = (cfg["base_seed"] + int(round(pt["axis_value"] * 7.0))
                  + si) % 2**32

        lbl = f"{sweep_header}={pt['axis_value']:.{sweep_precision}f}"
        print("--------------------------------------------------")
        print(f"{lbl}\n")

        dgp_kwargs = dict(n=cfg["n"], p=cfg["p"], p1=pt["p1"], M=cfg["M"],
                          overlap_pool_size=cfg["overlap_pool_size"],
                          target_snr=pt["snr"])

        # 1. Ordinary PCA
        store("OrdPCA", run_mc_trials_pca(
            cfg["num_MC"], f"{lbl} [OrdPCA]", dgp_kwargs, offset,
            num_components=pt["num_comp"], n_cores=N_WORKERS))

        # 2. Oracle Thresholded PCA
        store("OraclePCA", run_mc_trials_oracle_pca(
            cfg["num_MC"], f"{lbl} [OraclePCA]", dgp_kwargs, pt["p1"], offset,
            num_components=pt["num_comp"], n_cores=N_WORKERS))

        # 3. T-Rex SPCA — EN (Gram-based TENET) + ActiveSet
        store("TRexSPCA-EN-Act", run_mc_trials_trex_spca(
            cfg["num_MC"], f"{lbl} [TRexSPCA-EN-Act]", dgp_kwargs, pt["tFDR"],
            "ActiveSet", offset, cfg["lambda_2"], "TENET", "L2",
            num_components=pt["num_comp"], n_cores=N_WORKERS))

        # 4. T-Rex SPCA — ENaug (augmented TENET) + ActiveSet
        store("TRexSPCA-ENaug-Act", run_mc_trials_trex_spca(
            cfg["num_MC"], f"{lbl} [TRexSPCA-ENaug-Act]", dgp_kwargs,
            pt["tFDR"], "ActiveSet", offset, cfg["lambda_2"], "TENET_AUG",
            "L2", num_components=pt["num_comp"], n_cores=N_WORKERS))

        # 5. T-Rex SPCA — EN (Gram-based TENET) + Thresholded
        store("TRexSPCA-EN-Thr", run_mc_trials_trex_spca(
            cfg["num_MC"], f"{lbl} [TRexSPCA-EN-Thr]", dgp_kwargs, pt["tFDR"],
            "Thresholded", offset, cfg["lambda_2"], "TENET", "L2",
            num_components=pt["num_comp"], n_cores=N_WORKERS))

        # 6. T-Rex SPCA — ENaug (augmented TENET) + Thresholded
        store("TRexSPCA-ENaug-Thr", run_mc_trials_trex_spca(
            cfg["num_MC"], f"{lbl} [TRexSPCA-ENaug-Thr]", dgp_kwargs,
            pt["tFDR"], "Thresholded", offset, cfg["lambda_2"], "TENET_AUG",
            "L2", num_components=pt["num_comp"], n_cores=N_WORKERS))

    save_and_print_spca_sweep_results(
        out_dir=os.path.join(_THIS_DIR, "simulation_results", "data"),
        file_stem=stem, num_MC=cfg["num_MC"], sweep_col=sweep_col,
        sweep_header=sweep_header, sweep_precision=sweep_precision,
        axis_values=axis_values, method_names=METHOD_NAMES,
        fdr_map=fdr_map, tpr_map=tpr_map, pev_map=pev_map,
        pev_sig_map=pev_sig_map, pev_sigmix_map=pev_sigmix_map,
        include_support_metrics=not pev_only_table)


# ==============================================================================
# Part 5: realized FDR / TPR heatmaps over (target FDR x SNR), PC1 support
# ==============================================================================

def _trial_heatmap(task):
    """Per-trial worker: PC1 support recovery for both selector responses.

    Runs the per-PC T-Rex+GVS(EN/TENET) selector — the same configuration
    TRexSPCASelector uses internally — twice on the same data:

      "plugin" : y = X v_1, the plug-in ordinary PC1, i.e. exactly what T-Rex
                 SPCA regresses on. The regression's own true support is DENSE
                 (z_1 is a linear combination of all p variables), while the
                 scoring truth is the sparse factor-model support of factor 1.
      "oracle" : y = z_1, the true factor scores from the DGP (unobservable in
                 practice). Here the regression truth IS the sparse factor
                 support, so selector-level FDR control and factor-model
                 scoring coincide.

    Returns (fdr_plugin, tpr_plugin, fdr_oracle, tpr_oracle).
    """
    seed, dgp_kwargs, tfdr = task
    dat = generate_sparse_factor_model(seed=seed, **dgp_kwargs)
    X = center_columns(dat["X"])

    # True support of factor 1 (the scoring truth).
    true_supp = set(np.flatnonzero(dat["V"][:, 0] != 0.0))

    # Responses: plug-in ordinary PC1 and true factor scores.
    pca = PCA(center=False, normalize=False)
    ord_res = pca.fit(np.asfortranarray(X), 1)
    y_plugin = np.asarray(ord_res.Z)[:, 0].copy()
    y_oracle = dat["Z"][:, 0] - dat["Z"][:, 0].mean()

    # Same per-PC selector configuration as TRexSPCASelector internally.
    gvs_ctrl = trex.TRexGVSControlParameter()
    gvs_ctrl.gvs_type = trex.GVSType.EN
    gvs_ctrl.en_solver = trex.ENSolverType.TENET
    gvs_ctrl.lambda2_method = trex.LambdaSelectionMethod.CV_1SE_CCD
    gvs_ctrl.lambda_2 = -1.0   # auto via CV
    trex_ctrl = trex.TRexControlParameter()
    trex_ctrl.scaling_mode = trex.ScalingMode.L2
    trex_ctrl.solver_type = trex.SolverTypeForTRex.TENET

    def score(y):
        # Selector seed -1: hardware-entropy dummies per trial (required for
        # valid MC FDR estimates). Fresh Fortran copies per run — the selector
        # maps X and y zero-copy and works on them in place.
        sel = trex.TRexGVSSelector(np.asfortranarray(X), y.copy(), tFDR=tfdr,
                                  gvs_control=gvs_ctrl,
                                  trex_control=trex_ctrl,
                                  seed=-1, verbose=False)
        sel.select()
        sel_idx = set(int(i) for i in np.asarray(sel.selected_indices).ravel())
        tp = len(sel_idx & true_supp)
        k = len(sel_idx)
        fdr = 0.0 if k == 0 else (k - tp) / k
        tpr = 0.0 if not true_supp else tp / len(true_supp)
        return fdr, tpr

    fdr_p, tpr_p = score(y_plugin)
    fdr_o, tpr_o = score(y_oracle)
    return fdr_p, tpr_p, fdr_o, tpr_o


def run_fdr_tpr_heatmap(cfg):
    """2D sweep of PC1 support recovery: realized FDR and TPR over the full
    (target FDR x SNR) grid, for two selector responses (see _trial_heatmap).

    The pairing isolates the mechanism behind the FDR overshoot at loose
    targets: if the overshoot comes from scoring a dense-truth regression
    against a sparse truth (the README's hypothesis), it must vanish under
    the oracle response.

    Output: tidy CSV "response,metric,tfdr,snr_db,value" (metrics FDR / TPR),
    plus compact console matrices.
    """
    tfdr_grid = [0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40,
                 0.45, 0.50]
    snr_grid = [-10.0, -7.0, -5.0, -3.0, 0.0, 3.0, 5.0, 7.0, 10.0]

    n_tfdr, n_snr = len(tfdr_grid), len(snr_grid)

    print("================================================================")
    print("  T-Rex SPCA — Part 5: FDR/TPR heatmaps over (tFDR x SNR), PC1")
    print("================================================================")
    print(f"  n={cfg['n']}  p={cfg['p']}  p1={cfg['p1']}"
          f"  num_MC={cfg['num_MC']}"
          f"  responses: plugin (X v1) / oracle (true z1)"
          f"  ({N_WORKERS} workers)\n")

    # res[r]: response 0/1 = plugin FDR/TPR, 2/3 = oracle FDR/TPR;
    # each matrix is (n_snr, n_tfdr).
    res = [np.zeros((n_snr, n_tfdr)) for _ in range(4)]

    dgp_base = dict(n=cfg["n"], p=cfg["p"], p1=cfg["p1"], M=cfg["M"],
                    overlap_pool_size=cfg["overlap_pool_size"])

    for si, snr in enumerate(snr_grid):
        dgp_kwargs = dict(dgp_base, target_snr=snr)
        for ti, tfdr in enumerate(tfdr_grid):
            # Data seed depends on (mc, snr) only — the SAME datasets are
            # reused across the tFDR axis, so the columns of each heatmap row
            # are paired comparisons.
            tasks = [(cfg["base_seed"] + int(round(snr)) + mc * 1000,
                      dgp_kwargs, tfdr) for mc in range(cfg["num_MC"])]

            # One heatmap cell, restored from the cell cache after a restart.
            def compute(tasks=tasks):
                if N_WORKERS > 1:
                    out = list(get_executor(N_WORKERS, len(tasks)).map(_trial_heatmap, tasks))
                else:
                    out = [_trial_heatmap(t) for t in tasks]
                arr = np.asarray(out)  # (num_MC, 4)
                return {f"r{r}": arr[:, r].mean() for r in range(4)}

            cell = cell_cache.cached(f"heatmap|snr={snr:g}|tfdr={tfdr:.2f}|"
                                     f"{cell_cache.digest(tasks)}", compute)
            for r in range(4):
                res[r][si, ti] = cell[f"r{r}"]

            print(f"  SNR={snr:>5g}  tFDR={tfdr:.2f}  ->  "
                  f"FDR {res[0][si, ti]:.3f} (plugin) / "
                  f"{res[2][si, ti]:.3f} (oracle) | "
                  f"TPR {res[1][si, ti]:.3f} / {res[3][si, ti]:.3f}",
                  flush=True)

    # ---- console matrices ----
    resp_names = ["plugin", "plugin", "oracle", "oracle"]
    metric_names = ["FDR", "TPR", "FDR", "TPR"]
    for r in range(4):
        print(f"\n  Realized {metric_names[r]} [{resp_names[r]} response]"
              "  (rows: SNR dB, cols: tFDR)")
        print("      " + "".join(f"{t:>7.2f}" for t in tfdr_grid))
        for si, snr in enumerate(snr_grid):
            print(f"  {snr:>4.0f}"
                  + "".join(f"{res[r][si, ti]:>7.3f}" for ti in range(n_tfdr)))
    print("")

    # ---- tidy CSV ----
    out_dir = os.path.join(_THIS_DIR, "simulation_results", "data")
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for r in range(4):
        for si, snr in enumerate(snr_grid):
            for ti, tfdr in enumerate(tfdr_grid):
                rows.append(dict(response=resp_names[r],
                                 metric=metric_names[r],
                                 tfdr=f"{tfdr:.6f}", snr_db=f"{snr:.6f}",
                                 value=res[r][si, ti]))
    csv_path = os.path.join(out_dir,
                            "demo_trex_spca_02_mc_sim_pev_fdr_heatmap.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False, float_format="%.6f")
    print(f"[Info] Heatmap CSV saved to: {csv_path}\n")


# ==============================================================================
# Main
# ==============================================================================

def main():
    # Defaults held fixed while a single axis is swept. p1 = 10 follows Fig. 3
    # of the reference paper (demo 01 mirrors Fig. 2 and uses p1 = 5).
    cfg = dict(
        n=50,
        p=100,
        p1=10,
        M=3,
        overlap_pool_size=30,
        tFDR=0.10,
        num_MC=200,
        base_seed=42,
        lambda_2=-1.0,  # -1 triggers k-fold CV for lambda_2 selection
    )
    # Optional pilot-run override, e.g. TREX_SPCA_NUM_MC=10 python demo_...
    if os.environ.get("TREX_SPCA_NUM_MC"):
        cfg["num_MC"] = int(os.environ["TREX_SPCA_NUM_MC"])

    # Part selection, e.g. TREX_SPCA_PARTS=5 runs only the FDR/TPR heatmap.
    parts = os.environ.get("TREX_SPCA_PARTS", "12345")

    # =========================================================
    # Part 1: number of extracted PCs
    # =========================================================
    if "1" in parts:
        # The DGP keeps its M = 3 true factors throughout; only the number of
        # components the methods extract varies, so every PC beyond the third
        # can contribute Null EV and nothing else.
        #
        # The grid stops at 49, not 50: X is centered, so rank(X) <= n - 1 = 49
        # and a 50th component carries no variance at all. Asking the per-PC
        # selector to regress on that null direction throws outright ("lambda
        # grid anchor is degenerate"), because its response vector is constant.
        # 49 is therefore the largest attainable PC count on this design, not
        # a safety margin.
        pc_grid = [1, 3, 5, 10, 20, 30, 40, 49]
        points = [make_point(snr=0.0, p1=cfg["p1"], tFDR=cfg["tFDR"],
                             num_comp=n_pcs, axis_value=float(n_pcs))
                  for n_pcs in pc_grid]
        run_sweep(cfg, points, "n_pcs", "#PCs", 0,
                  "T-Rex SPCA — Cumulative PEV vs. number of extracted PCs",
                  "demo_trex_spca_02_mc_sim_pev_pcs")

    # =========================================================
    # Part 2: signal-to-noise ratio
    # =========================================================
    if "2" in parts:
        snr_grid = [-10.0, -7.0, -5.0, -3.0, 0.0, 3.0, 5.0, 7.0, 10.0]
        points = [make_point(snr=snr, p1=cfg["p1"], tFDR=cfg["tFDR"],
                             num_comp=0, axis_value=snr)   # 0 = M = 3 factors
                  for snr in snr_grid]
        run_sweep(cfg, points, "snr_db", "SNR(dB)", 1,
                  "T-Rex SPCA — Cumulative PEV (PCs 1-3) vs. SNR",
                  "demo_trex_spca_02_mc_sim_pev_snr",
                  pev_only_table=True)

    # =========================================================
    # Part 3: number of true active loadings
    # =========================================================
    if "3" in parts:
        # Bounded above by overlap_pool_size = 30: each factor draws its
        # support from that shared pool, so p1 cannot exceed it.
        p1_grid = [1, 5, 10, 15, 20, 25, 30]
        points = [make_point(snr=0.0, p1=p1, tFDR=cfg["tFDR"],
                             num_comp=0, axis_value=float(p1))
                  for p1 in p1_grid]
        run_sweep(cfg, points, "p1", "#Active", 0,
                  "T-Rex SPCA — Cumulative PEV (PCs 1-3) vs. number of true "
                  "active loadings",
                  "demo_trex_spca_02_mc_sim_pev_loadings",
                  pev_only_table=True)

    # =========================================================
    # Part 4: target FDR
    # =========================================================
    if "4" in parts:
        tfdr_grid = [0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40,
                     0.45, 0.50]
        points = [make_point(snr=0.0, p1=cfg["p1"], tFDR=tfdr,
                             num_comp=0, axis_value=tfdr)
                  for tfdr in tfdr_grid]
        run_sweep(cfg, points, "tfdr", "tFDR", 2,
                  "T-Rex SPCA — Cumulative PEV (PCs 1-3) vs. target FDR",
                  "demo_trex_spca_02_mc_sim_pev_tfdr",
                  pev_only_table=True)

    # =========================================================
    # Part 5: FDR/TPR heatmaps over (target FDR x SNR)
    # =========================================================
    if "5" in parts:
        run_fdr_tpr_heatmap(cfg)


if __name__ == "__main__":
    main()
