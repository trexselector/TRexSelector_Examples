# ==============================================================================
# demo_trex_spca_01_mc_sim.py
# ==============================================================================
#
# Monte Carlo simulation comparing T-Rex SPCA against PCA baselines.
# Python counterpart of cpp demo_trex_spca_01_mc_sim.cpp.
#
# DGP: sparse M-factor model — X = Z * V^T + E.
#      n=50, p=100, M=3 latent factors, p1=5 active loadings per factor,
#      overlap_pool_size=30 (shared candidate-index pool across factors).
#
# Section 1: SNR sweep over the snr_values grid set in main()
#            (the legacy CRAN reference grid {-10, -7, -5, -3, 0, 3, 5, 7, 10} dB,
#            num_MC=200 — matching the cpp reference exactly).
#   Methods compared:
#     1. OrdPCA             — ordinary PCA (baseline; no sparsity constraint)
#     2. OraclePCA          — thresholded PCA with known support cardinality p1
#     3. TRexSPCA-EN-Act    — T-Rex+GVS(EN/TENET, Gram-based) + ActiveSet
#     4. TRexSPCA-ENaug-Act — T-Rex+GVS(EN/TENETAug, augmented) + ActiveSet
#     5. TRexSPCA-EN-Thr    — T-Rex+GVS(EN/TENET, Gram-based) + Thresholded
#     6. TRexSPCA-ENaug-Thr — T-Rex+GVS(EN/TENETAug, augmented) + Thresholded
#
#   All methods see the same center-only X (covariance-PCA footing, as in the
#   legacy R reference); the per-PC selector uses the default L2 scaling.
#
# Metrics (num_MC-averaged; TPR/FDR on PC1 support only — see evaluate_spca):
#   - FDR  — false discovery rate of PC1's estimated loading support
#   - TPR  — true positive rate  of PC1's estimated loading support
#   - PEV  — cumulative proportion of explained variance (all M components),
#            reported under three normalizations: by the total variance of X
#            (bounded by 1); by the Signal + Mixed EV OF THE DATA — the
#            variance carried by the true active variables — per Definition 1
#            of the reference paper (may exceed 1: the overshoot is null
#            variance being reported as explained variance); and the method's
#            own Sig+Mix part over that same denominator (the paper's
#            "Ordinary PCA (Sig + Mix)" reference when read for OrdPCA).
#
# Section 2: union-support FDR per PC over the same SNR grid — each PC's
#            selected support scored against the UNION of the factor supports,
#            the only well-posed per-PC truth beyond PC1; see
#            demo_trex_spca_union_fdr_sweep.
#
# Section selection: TREX_SPCA_PARTS (default "12"); TREX_SPCA_NUM_MC
# overrides the MC count (e.g. TREX_SPCA_NUM_MC=10 for a pilot run).
#
# Support indices are 0-based (Python convention).
#
# Usage: python demo_trex_spca_01_mc_sim.py [num_worker_cores]
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

from trex_spca_sim_utils import (N_CORES, center_columns,  # noqa: E402
                                 resolve_workers,
                                 generate_sparse_factor_model, get_executor,
                                 run_mc_trials_oracle_pca, run_mc_trials_pca,
                                 run_mc_trials_trex_spca,
                                 save_and_print_spca_mc_results)
import cell_cache  # family folder, put on sys.path by trex_spca_sim_utils

# Optional CLI worker count, resolved like TREX_NUM_WORKERS (0 = all usable cores).
N_WORKERS = resolve_workers(sys.argv[1]) if len(sys.argv) > 1 else N_CORES


# ==============================================================================
# Monte Carlo Simulation — SNR sweep
# ==============================================================================

def demo_trex_spca_mc_snr_sweep(cfg, snr_values, stem):
    """Run the SNR-sweep MC simulation and save results.

    cfg holds the simulation parameters (n, p, p1, M, overlap_pool_size,
    tFDR, lambda_2, num_MC, base_seed); snr_values is the SNR grid in dB;
    stem is the output file base name (no folder, no extension).
    """
    print("================================================================")
    print("  T-Rex SPCA MC Simulation — SNR Sweep")
    print("================================================================")
    print(f"  n={cfg['n']}  p={cfg['p']}  M={cfg['M']}  p1={cfg['p1']}"
          f"  tFDR={cfg['tFDR']}  num_MC={cfg['num_MC']}"
          f"  ({N_WORKERS} workers)\n")

    method_names = ["OrdPCA", "OraclePCA",
                    "TRexSPCA-EN-Act", "TRexSPCA-ENaug-Act",
                    "TRexSPCA-EN-Thr", "TRexSPCA-ENaug-Thr"]

    fdr_map = {name: [] for name in method_names}
    tpr_map = {name: [] for name in method_names}
    pev_map = {name: [] for name in method_names}
    pev_sig_map = {name: [] for name in method_names}
    pev_sigmix_map = {name: [] for name in method_names}

    def store(name, r):
        fdr_map[name].append(r["avg_fdr"])
        tpr_map[name].append(r["avg_tpr"])
        pev_map[name].append(r["avg_pev"])
        pev_sig_map[name].append(r["avg_pev_sig"])
        pev_sigmix_map[name].append(r["avg_pev_sigmix"])

    dgp_base = dict(n=cfg["n"], p=cfg["p"], p1=cfg["p1"], M=cfg["M"],
                    overlap_pool_size=cfg["overlap_pool_size"])

    for snr in snr_values:
        offset = cfg["base_seed"] + int(round(snr))
        snr_lbl = f"SNR={snr:.1f} dB"
        dgp_kwargs = dict(dgp_base, target_snr=snr)

        print("--------------------------------------------------")
        print(f"{snr_lbl}\n")

        # 1. Ordinary PCA
        store("OrdPCA", run_mc_trials_pca(
            cfg["num_MC"], f"{snr_lbl} [OrdPCA]", dgp_kwargs, offset,
            n_cores=N_WORKERS))

        # 2. Oracle Thresholded PCA
        store("OraclePCA", run_mc_trials_oracle_pca(
            cfg["num_MC"], f"{snr_lbl} [OraclePCA]", dgp_kwargs, cfg["p1"],
            offset, n_cores=N_WORKERS))

        # 3. T-Rex SPCA — EN (Gram-based TENET) + ActiveSet
        store("TRexSPCA-EN-Act", run_mc_trials_trex_spca(
            cfg["num_MC"], f"{snr_lbl} [TRexSPCA-EN-Act]", dgp_kwargs,
            cfg["tFDR"], "ActiveSet", offset, cfg["lambda_2"], "TENET",
            n_cores=N_WORKERS))

        # 4. T-Rex SPCA — ENaug (augmented TENET) + ActiveSet
        store("TRexSPCA-ENaug-Act", run_mc_trials_trex_spca(
            cfg["num_MC"], f"{snr_lbl} [TRexSPCA-ENaug-Act]", dgp_kwargs,
            cfg["tFDR"], "ActiveSet", offset, cfg["lambda_2"], "TENET_AUG",
            n_cores=N_WORKERS))

        # 5. T-Rex SPCA — EN (Gram-based TENET) + Thresholded
        store("TRexSPCA-EN-Thr", run_mc_trials_trex_spca(
            cfg["num_MC"], f"{snr_lbl} [TRexSPCA-EN-Thr]", dgp_kwargs,
            cfg["tFDR"], "Thresholded", offset, cfg["lambda_2"], "TENET",
            n_cores=N_WORKERS))

        # 6. T-Rex SPCA — ENaug (augmented TENET) + Thresholded
        store("TRexSPCA-ENaug-Thr", run_mc_trials_trex_spca(
            cfg["num_MC"], f"{snr_lbl} [TRexSPCA-ENaug-Thr]", dgp_kwargs,
            cfg["tFDR"], "Thresholded", offset, cfg["lambda_2"], "TENET_AUG",
            n_cores=N_WORKERS))

    save_and_print_spca_mc_results(
        out_dir=os.path.join(_THIS_DIR, "simulation_results", "data"),
        file_stem=stem, num_MC=cfg["num_MC"],
        snr_values=snr_values, method_names=method_names,
        fdr_map=fdr_map, tpr_map=tpr_map, pev_map=pev_map,
        pev_sig_map=pev_sig_map, pev_sigmix_map=pev_sigmix_map)


# ==============================================================================
# Section 2: union-support FDR per PC
# ==============================================================================

def _trial_union_fdr(task):
    """Per-trial worker: union-support FDR and support size of every PC.

    Module-level so ProcessPoolExecutor spawn workers can resolve it (the
    child re-imports this file as __mp_main__).
    """
    seed, dgp_kwargs, tFDR, lambda_2 = task
    dat = generate_sparse_factor_model(seed=seed, **dgp_kwargs)
    X = center_columns(dat["X"])
    M = dat["V"].shape[1]

    # Union of the true factor supports.
    in_union = (dat["V"] != 0.0).any(axis=1)

    ctrl = trex.TRexSPCAControlParameter()
    ctrl.mode = trex.SPCAMode.ActiveSet
    ctrl.en_solver = trex.ENSolverType.TENET
    ctrl.gvs_ctrl.trex_ctrl.scaling_mode = trex.ScalingMode.L2
    ctrl.gvs_ctrl.lambda2_method = trex.LambdaSelectionMethod.CV_1SE_CCD
    ctrl.gvs_ctrl.lambda_2 = lambda_2

    # Selector seed -1: hardware-entropy dummies per trial (required for valid
    # MC FDR estimates).
    sel = trex.TRexSPCASelector(np.asfortranarray(X), spca_ctrl=ctrl, seed=-1)
    sel.select(M, tFDR)

    fdr, k = [], []
    for m in range(M):
        A = sel.active_sets[m]
        fd = sum(1 for idx in A if not in_union[idx])
        fdr.append(0.0 if not A else fd / len(A))
        k.append(float(len(A)))
    return fdr, k


def demo_trex_spca_union_fdr_sweep(cfg, snr_values, stem):
    """Realized FDR of each PC's selected support, scored against the UNION of
    the true factor supports.

    Beyond PC1 there is no unambiguous per-factor ground truth: the factor
    supports overlap, and PCA's orthogonality constraint mixes the later
    components across factors, so a per-factor FDR would penalize legitimately
    selected leaked variables and measure the ambiguity of the truth rather
    than the selector. The UNION of the factor supports, however, is
    well-posed for every component — a variable outside the union is null for
    EVERY factor, so selecting it is a false discovery no matter how the
    components mix. This sweep answers the one question about the later PCs
    that CAN be asked cleanly: does the selector start harvesting pure-noise
    variables on components beyond the first?

    One variant is run: ActiveSet + TENET — the selector's own support (the
    Thresholded mode rebuilds loadings at the same cardinality). Reported per
    PC m: mean union-FDR and mean support size |A_m|.

    Output: tidy CSV "pc,metric,snr_db,value" (metrics FDRunion / k).
    """
    print("================================================================")
    print("  T-Rex SPCA — union-support FDR per PC (ActiveSet/TENET)")
    print("================================================================")
    print(f"  n={cfg['n']}  p={cfg['p']}  M={cfg['M']}  p1={cfg['p1']}"
          f"  tFDR={cfg['tFDR']}  num_MC={cfg['num_MC']}"
          f"  ({N_WORKERS} workers)\n")

    M = cfg["M"]
    n_snr = len(snr_values)
    fdr_mean = np.zeros((M, n_snr))
    k_mean = np.zeros((M, n_snr))

    dgp_base = dict(n=cfg["n"], p=cfg["p"], p1=cfg["p1"], M=cfg["M"],
                    overlap_pool_size=cfg["overlap_pool_size"])

    for si, snr in enumerate(snr_values):
        offset = cfg["base_seed"] + int(round(snr))
        dgp_kwargs = dict(dgp_base, target_snr=snr)
        tasks = [(offset + mc * 1000, dgp_kwargs, cfg["tFDR"], cfg["lambda_2"])
                 for mc in range(cfg["num_MC"])]

        # One SNR cell (per-PC union-FDR and support size), restored from the
        # cell cache after a restart.
        def compute(tasks=tasks):
            if N_WORKERS > 1:
                res = list(get_executor(N_WORKERS, len(tasks)).map(_trial_union_fdr, tasks))
            else:
                res = [_trial_union_fdr(t) for t in tasks]
            fdr_trial = np.array([r[0] for r in res])   # (num_MC, M)
            k_trial = np.array([r[1] for r in res])     # (num_MC, M)
            return {**{f"fdr{m}": fdr_trial[:, m].mean() for m in range(M)},
                    **{f"k{m}": k_trial[:, m].mean() for m in range(M)}}

        cell = cell_cache.cached(f"union_fdr|snr={snr:g}|{cell_cache.digest(tasks)}", compute)
        fdr_mean[:, si] = [cell[f"fdr{m}"] for m in range(M)]
        k_mean[:, si] = [cell[f"k{m}"] for m in range(M)]

        print(f"  SNR={snr:>5g} dB — union-FDR:"
              + "".join(f"  PC{m + 1}={fdr_mean[m, si]:.3f}" for m in range(M)),
              flush=True)

    # ---- console table ----
    print("\n  Union-support FDR per PC (rows) over SNR (cols):")
    print("      " + "".join(f"{s:>8.1f}" for s in snr_values))
    for m in range(M):
        print(f"  PC{m + 1} "
              + "".join(f"{fdr_mean[m, si]:>8.3f}" for si in range(n_snr)))
    print("\n  Mean support size |A_m|:")
    for m in range(M):
        print(f"  PC{m + 1} "
              + "".join(f"{k_mean[m, si]:>8.2f}" for si in range(n_snr)))
    print("")

    # ---- tidy CSV ----
    out_dir = os.path.join(_THIS_DIR, "simulation_results", "data")
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for m in range(M):
        for si, snr in enumerate(snr_values):
            rows.append(dict(pc=m + 1, metric="FDRunion",
                             snr_db=f"{snr:.6f}", value=fdr_mean[m, si]))
            rows.append(dict(pc=m + 1, metric="k",
                             snr_db=f"{snr:.6f}", value=k_mean[m, si]))
    csv_path = os.path.join(out_dir, stem + ".csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False, float_format="%.6f")
    print(f"[Info] Union-FDR CSV saved to: {csv_path}\n")


# ==============================================================================
# Main
# ==============================================================================

def main():
    cfg = dict(
        n=50,
        p=100,
        p1=5,
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

    # Legacy CRAN reference grid, mirroring the cpp demo exactly.
    snr_values = [-10.0, -7.0, -5.0, -3.0, 0.0, 3.0, 5.0, 7.0, 10.0]

    # Part selection, e.g. TREX_SPCA_PARTS=2 runs only the union-FDR sweep.
    parts = os.environ.get("TREX_SPCA_PARTS", "12")

    # =========================================================
    # Section 1: SNR sweep
    # =========================================================
    if "1" in parts:
        demo_trex_spca_mc_snr_sweep(cfg, snr_values, "demo_trex_spca_01_mc_sim")

    # =========================================================
    # Section 2: union-support FDR per PC
    # =========================================================
    if "2" in parts:
        demo_trex_spca_union_fdr_sweep(cfg, snr_values,
                                       "demo_trex_spca_01_mc_sim_union_fdr")


if __name__ == "__main__":
    main()
