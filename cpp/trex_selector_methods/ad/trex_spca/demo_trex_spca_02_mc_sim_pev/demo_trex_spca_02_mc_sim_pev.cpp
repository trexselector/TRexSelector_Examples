// ==============================================================================
// demo_trex_spca_02_mc_sim_pev.cpp
// ==============================================================================
/**
 * @file demo_trex_spca_02_mc_sim_pev.cpp
 *
 * @brief Cumulative explained variance of T-Rex SPCA under four sweeps.
 *
 * @details
 *  DGP: sparse M-factor model — X = Z * V^T + E.
 *       n=50, p=100, M=3 latent factors, overlap_pool_size=30 (shared candidate
 *       index pool across factors). Unlike demo 01 (which mirrors Fig. 2 of the
 *       reference paper at p1=5), this demo mirrors Fig. 3 and therefore uses
 *       p1=10 as its default number of true active loadings per factor.
 *
 *  The headline metric is the PEV of Definition 1 of the reference paper: the
 *  cumulative ADJUSTED EV divided by the Signal + Mixed EV OF THE DATA — the
 *  total variance carried by the true active variables (union of the factor
 *  supports), a denominator FIXED per dataset and SHARED by all methods. The
 *  ratio is deliberately uncapped: a method that spends loadings on null
 *  variables (ordinary PCA above all) accumulates variance the denominator
 *  does not contain and climbs past 100 %, which the paper reads as inferior
 *  performance. The FDR-controlled T-Rex variants stay below 100 %, with
 *  plateaus of about 80-83 % (ActiveSet) and 86-96 % (Thresholded).
 *
 *  NOTE this demo originally normalized by each method's OWN Signal + Mixed
 *  EV. That self-normalized reading is structurally >= 100 % for
 *  null-capturing methods, INVERTS the SNR trend (OrdPCA fell from 167 % at
 *  -10 dB toward 100 %), and cannot reproduce the rising curves of the
 *  published Fig. 3 — it is why the first Fig.-3 reproduction attempt failed.
 *  The convention used now follows the legacy R reference
 *  (demo_trex_spca_03_fig3.R, denominator mode "active"). It reproduces the
 *  Fig. 3(b) rise from about 40 %, but no constructible denominator matches
 *  all published Fig. 3 anchors. The R script documents the open conventions.
 *
 *  Alongside it, two companions are reported: the conventional total-variance
 *  PEV (bounded by 1), and PEVsigmix — the method's own Sig+Mix part over the
 *  same data-level denominator. Read for OrdPCA, the latter is the paper's
 *  "Ordinary PCA (Sig + Mix)" reference curve, which levels off near 108 %
 *  while plain OrdPCA keeps climbing on Null EV.
 *
 *  Four sweeps, each varying ONE axis with the others held at their defaults
 *  (# PCs = 3, SNR = 0 dB, p1 = 10, tFDR = 0.10):
 *
 *    Part 1 — # extracted PCs   {1, 3, 5, 10, 20, 30, 40, 49}
 *             The data still carries exactly M = 3 true factors; only the number
 *             of components the methods EXTRACT grows. Components beyond the
 *             third are null by construction, so this sweep shows how fast a
 *             method starts accumulating variance it should not be claiming.
 *    Part 2 — SNR (dB)          {-10, -7, -5, -3, 0, 3, 5, 7, 10}
 *    Part 3 — # true active loadings p1  {1, 5, 10, 15, 20, 25, 30}
 *    Part 4 — target FDR        {0.01, 0.05, 0.10, ..., 0.50}
 *             Only the T-Rex variants respond to this axis; the two PCA
 *             baselines are tFDR-blind and are reported as flat references.
 *    Part 5 — FDR/TPR heatmaps over the full (target FDR x SNR) grid, PC1
 *             support recovery only, for two selector responses: the plug-in
 *             ordinary PC1 (what T-Rex SPCA regresses on) and the true factor
 *             scores z_1 (oracle response). Isolates the mechanism behind the
 *             realized-FDR overshoot at loose targets; see run_fdr_tpr_heatmap.
 *
 *  Part selection: TREX_SPCA_PARTS (default "12345"), e.g. TREX_SPCA_PARTS=5
 *  reruns only the heatmap sweep. TREX_SPCA_NUM_MC overrides the MC count.
 *
 *  Methods compared (identical to demo 01):
 *    1. OrdPCA             — ordinary PCA (baseline; no sparsity constraint)
 *    2. OraclePCA          — thresholded PCA with known support cardinality p1
 *    3. TRexSPCA-EN-Act    — T-Rex+GVS(EN/TENET, Gram-based) + ActiveSet
 *    4. TRexSPCA-ENaug-Act — T-Rex+GVS(EN/TENETAug, augmented) + ActiveSet
 *    5. TRexSPCA-EN-Thr    — T-Rex+GVS(EN/TENET, Gram-based) + Thresholded
 *    6. TRexSPCA-ENaug-Thr — T-Rex+GVS(EN/TENETAug, augmented) + Thresholded
 *
 *  Metrics (num_MC-averaged; TPR/FDR on PC1 support only — see evaluate_spca):
 *    - FDR     — false discovery rate of PC1's estimated loading support
 *    - TPR     — true positive rate  of PC1's estimated loading support
 *    - PEV     — cumulative adjusted EV / total variance of X (bounded by 1)
 *    - PEVsig  — cumulative adjusted EV / (Signal EV + Mixed EV), Definition 1
 */
// ==============================================================================

// std includes
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

// Eigen includes
#include <Eigen/Dense>

// OpenMP compatibility layer
#include <utils/openmp/utils_openmp.hpp>

// Demo utilities (transitively includes trex_spca.hpp, pca.hpp, utils_openmp.hpp)
#include "trex_spca_sim_utils.hpp"

// ==============================================================================

// Namespace usage
using namespace spca_sim;
using trex::trex_selector_methods::ad::trex_gvs::TRexGVSSelector;
using trex::trex_selector_methods::ad::trex_gvs::TRexGVSControlParameter;
using trex::trex_selector_methods::ad::trex_gvs::GVSType;
using trex::trex_selector_methods::ad::utils::solver_dispatch::SolverTypeForTRex;

// ==============================================================================
// Shared sweep driver
// ==============================================================================

/** @brief One grid point of a sweep: everything a single run needs to vary. */
struct SweepPoint {
    double       snr          = 0.0;   ///< SNR in dB
    Eigen::Index p1           = 10;    ///< True active loadings per factor
    double       tFDR         = 0.10;  ///< Target FDR handed to the selector
    Eigen::Index num_comp     = 0;     ///< Extracted PCs (0 = as many as true factors)
    double       axis_value   = 0.0;   ///< Value plotted on the sweep axis
};

/** @brief Ordered method names, shared by every part of this demo. */
static const std::vector<std::string> ordered_method_names = {
    "OrdPCA",
    "OraclePCA",
    "TRexSPCA-EN-Act",
    "TRexSPCA-ENaug-Act",
    "TRexSPCA-EN-Thr",
    "TRexSPCA-ENaug-Thr"
};

/**
 * @brief Run one sweep over the supplied grid points and save table + CSV.
 *
 * @param cfg              Base configuration (n, p, M, num_MC, base_seed, lambda_2).
 * @param points           Grid points; each carries the varied parameter(s).
 * @param sweep_col        CSV column name for the sweep axis (e.g. "n_pcs").
 * @param sweep_header     Table header label for the sweep axis.
 * @param sweep_precision  Decimals used when printing the axis header row.
 * @param banner           Human-readable description printed above the run.
 * @param stem             Output file stem (no folder, no extension).
 * @param pev_only_table   Display only the two cumulative-PEV rows in the
 *                         table. Parts 2–4 are restricted to the cumulative
 *                         PEV of the first three PCs; FDR/TPR remain in the
 *                         CSV as the data record.
 */
static void run_sweep(const SimConfig&               cfg,
                      const std::vector<SweepPoint>& points,
                      const std::string&             sweep_col,
                      const std::string&             sweep_header,
                      int                            sweep_precision,
                      const std::string&             banner,
                      const std::string&             stem,
                      bool                           pev_only_table = false)
{
    std::cout << "================================================================\n";
    std::cout << "  " << banner << "\n";
    std::cout << "================================================================\n";
    std::cout << "  n=" << cfg.n << "  p=" << cfg.p << "  M=" << cfg.M
              << "  num_MC=" << cfg.num_MC << "\n\n";

    std::map<std::string, Eigen::VectorXd> fdr_map, tpr_map, pev_map, pev_sig_map,
                                           pev_sigmix_map;
    for (const auto& name : ordered_method_names) {
        fdr_map[name]        = Eigen::VectorXd(points.size());
        tpr_map[name]        = Eigen::VectorXd(points.size());
        pev_map[name]        = Eigen::VectorXd(points.size());
        pev_sig_map[name]    = Eigen::VectorXd(points.size());
        pev_sigmix_map[name] = Eigen::VectorXd(points.size());
    }

    std::vector<double> axis_values;
    axis_values.reserve(points.size());
    for (const auto& pt : points) axis_values.push_back(pt.axis_value);

    for (std::size_t si = 0; si < points.size(); ++si) {
        const SweepPoint& pt = points[si];
        const auto        ei = static_cast<Eigen::Index>(si);

        // Distinct seed band per grid point, derived from the varied value so
        // that re-running a single point reproduces its own data stream.
        const unsigned offset = static_cast<unsigned>(
            cfg.base_seed + static_cast<int>(std::round(pt.axis_value * 7.0))
            + static_cast<int>(si));

        std::ostringstream lbl_ss;
        lbl_ss << std::fixed << std::setprecision(sweep_precision) << pt.axis_value;
        const std::string lbl = sweep_header + "=" + lbl_ss.str();

        std::cout << "--------------------------------------------------\n";
        std::cout << lbl << "\n\n";

        // Capture every DGP parameter by value for OMP thread safety.
        const auto make_data = [cfg, pt](unsigned seed) -> FactorModelData {
            return DataGenerator::generate_sparse_factor_model(
                cfg.n, cfg.p, pt.p1, cfg.M, pt.snr,
                cfg.overlap_pool_size, seed);
        };

        auto store = [&](const std::string& name, const SPCAGridPointResult& r) {
            fdr_map[name](ei)        = r.avg_fdr;
            tpr_map[name](ei)        = r.avg_tpr;
            pev_map[name](ei)        = r.avg_pev;
            pev_sig_map[name](ei)    = r.avg_pev_sig;
            pev_sigmix_map[name](ei) = r.avg_pev_sigmix;
        };

        // 1. Ordinary PCA
        store("OrdPCA", run_mc_trials_pca(
            cfg.num_MC, lbl + " [OrdPCA]", make_data, offset, pt.num_comp));

        // 2. Oracle Thresholded PCA
        store("OraclePCA", run_mc_trials_oracle_pca(
            cfg.num_MC, lbl + " [OraclePCA]", make_data, pt.p1, offset,
            pt.num_comp));

        // 3. T-Rex SPCA — EN (Gram-based TENET) + ActiveSet
        store("TRexSPCA-EN-Act", run_mc_trials_trex_spca(
            cfg.num_MC, lbl + " [TRexSPCA-EN-Act]", make_data, pt.tFDR,
            SPCAMode::ActiveSet, offset, cfg.lambda_2,
            ENSolverType::TENET, ScalingMode::L2, pt.num_comp));

        // 4. T-Rex SPCA — ENaug (augmented TENET) + ActiveSet
        store("TRexSPCA-ENaug-Act", run_mc_trials_trex_spca(
            cfg.num_MC, lbl + " [TRexSPCA-ENaug-Act]", make_data, pt.tFDR,
            SPCAMode::ActiveSet, offset, cfg.lambda_2,
            ENSolverType::TENET_AUG, ScalingMode::L2, pt.num_comp));

        // 5. T-Rex SPCA — EN (Gram-based TENET) + Thresholded
        store("TRexSPCA-EN-Thr", run_mc_trials_trex_spca(
            cfg.num_MC, lbl + " [TRexSPCA-EN-Thr]", make_data, pt.tFDR,
            SPCAMode::Thresholded, offset, cfg.lambda_2,
            ENSolverType::TENET, ScalingMode::L2, pt.num_comp));

        // 6. T-Rex SPCA — ENaug (augmented TENET) + Thresholded
        store("TRexSPCA-ENaug-Thr", run_mc_trials_trex_spca(
            cfg.num_MC, lbl + " [TRexSPCA-ENaug-Thr]", make_data, pt.tFDR,
            SPCAMode::Thresholded, offset, cfg.lambda_2,
            ENSolverType::TENET_AUG, ScalingMode::L2, pt.num_comp));
    }

    save_and_print_spca_sweep_results(cfg.num_MC, stem, sweep_col, sweep_header,
                                      sweep_precision, axis_values, ordered_method_names,
                                      fdr_map, tpr_map, pev_map, pev_sig_map,
                                      pev_sigmix_map,
                                      /*include_support_metrics=*/!pev_only_table);
}


// ==============================================================================
// Part 5: realized FDR / TPR heatmaps over (target FDR x SNR), PC1 support
// ==============================================================================

/**
 * @brief 2D sweep of PC1 support recovery: realized FDR and TPR over the full
 *        (target FDR x SNR) grid, for two selector responses.
 *
 * @details Each trial runs the per-PC T-Rex+GVS(EN/TENET) selector — the same
 *          configuration TRexSPCA uses internally — twice on the same data:
 *
 *   "plugin" : y = X v_1, the plug-in ordinary PC1, i.e. exactly what T-Rex
 *              SPCA regresses on. The regression's own true support is DENSE
 *              (z_1 is a linear combination of all p variables), while the
 *              scoring truth is the sparse factor-model support of factor 1.
 *   "oracle" : y = z_1, the true factor scores from the DGP (unobservable in
 *              practice). Here the regression truth IS the sparse factor
 *              support, so selector-level FDR control and factor-model
 *              scoring coincide.
 *
 *  The pairing isolates the mechanism behind the FDR overshoot at loose
 *  targets: if the overshoot comes from scoring a dense-truth regression
 *  against a sparse truth (the README's hypothesis), it must vanish under
 *  the oracle response.
 *
 *  Output: tidy CSV "response,metric,tfdr,snr_db,value" (metrics FDR / TPR),
 *  plus compact console matrices.
 */
static void run_fdr_tpr_heatmap(const SimConfig& cfg) {
    const std::vector<double> tfdr_grid =
        {0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50};
    const std::vector<double> snr_grid =
        {-10.0, -7.0, -5.0, -3.0, 0.0, 3.0, 5.0, 7.0, 10.0};

    const std::size_t n_tfdr = tfdr_grid.size();
    const std::size_t n_snr  = snr_grid.size();

    std::cout << "================================================================\n";
    std::cout << "  T-Rex SPCA — Part 5: FDR/TPR heatmaps over (tFDR x SNR), PC1\n";
    std::cout << "================================================================\n";
    std::cout << "  n=" << cfg.n << "  p=" << cfg.p << "  p1=" << cfg.p1
              << "  num_MC=" << cfg.num_MC
              << "  responses: plugin (X v1) / oracle (true z1)\n\n";

    // [response][metric][snr][tfdr]; response 0 = plugin, 1 = oracle;
    // metric 0 = FDR, 1 = TPR.
    std::vector<Eigen::MatrixXd> res(4, Eigen::MatrixXd::Zero(
        static_cast<Eigen::Index>(n_snr), static_cast<Eigen::Index>(n_tfdr)));

    const int iMC = static_cast<int>(cfg.num_MC);

    for (std::size_t si = 0; si < n_snr; ++si) {
        const double snr = snr_grid[si];
        for (std::size_t ti = 0; ti < n_tfdr; ++ti) {
            const double tfdr = tfdr_grid[ti];

            // One heatmap cell, restored from the cache after a restart.
            const auto compute_cell = [&]() -> trex_cell_cache::Cell {
                // Per-(mc) sums, accumulated in vectors for OMP safety.
                std::vector<double> fdr_p(cfg.num_MC), tpr_p(cfg.num_MC);
                std::vector<double> fdr_o(cfg.num_MC), tpr_o(cfg.num_MC);

                const auto make_data = [cfg, snr](unsigned seed) -> FactorModelData {
                    return DataGenerator::generate_sparse_factor_model(
                        cfg.n, cfg.p, cfg.p1, cfg.M, snr,
                        cfg.overlap_pool_size, seed);
                };

                #pragma omp parallel for schedule(dynamic)
                for (int mc = 0; mc < iMC; ++mc) {
                    // Data seed depends on (mc, snr) only — the SAME datasets are
                    // reused across the tFDR axis, so the columns of each heatmap
                    // row are paired comparisons.
                    const unsigned seed = static_cast<unsigned>(
                        cfg.base_seed + static_cast<int>(std::round(snr)))
                        + static_cast<unsigned>(mc) * 1000u;
                    auto dat = make_data(seed);
                    center_columns(dat.X);

                    // True support of factor 1 (the scoring truth).
                    std::vector<Eigen::Index> true_supp;
                    for (Eigen::Index j = 0; j < dat.X.cols(); ++j)
                        if (dat.V(j, 0) != 0.0) true_supp.push_back(j);

                    // Responses: plug-in ordinary PC1 and true factor scores.
                    pca_ns::PCA       pca(dat.X, 1, /*center=*/false, /*normalize=*/false);
                    pca_ns::PCAResult ord = pca.fit();
                    Eigen::VectorXd y_plugin = ord.Z.col(0);
                    Eigen::VectorXd y_oracle = dat.Z.col(0);
                    y_oracle.array() -= y_oracle.mean();

                    // Same per-PC selector configuration as TRexSPCA internally.
                    TRexGVSControlParameter gvs_ctrl;
                    gvs_ctrl.gvs_type       = GVSType::EN;
                    gvs_ctrl.en_solver      = ENSolverType::TENET;
                    gvs_ctrl.lambda2_method = LambdaSelectionMethod::CV_1SE_CCD;
                    gvs_ctrl.lambda_2       = -1.0;   // auto via CV
                    gvs_ctrl.trex_ctrl.scaling_mode = ScalingMode::L2;
                    gvs_ctrl.trex_ctrl.solver_type  = SolverTypeForTRex::TENET;

                    Eigen::Map<Eigen::MatrixXd> X_map(dat.X.data(),
                                                      dat.X.rows(), dat.X.cols());

                    const auto score = [&](Eigen::VectorXd& y,
                                           double& fdr_out, double& tpr_out) {
                        Eigen::Map<Eigen::VectorXd> y_map(y.data(), y.size());
                        // Selector seed -1: hardware-entropy dummies per trial
                        // (required for valid MC FDR estimates).
                        TRexGVSSelector sel(X_map, y_map, tfdr, gvs_ctrl,
                                            /*seed=*/-1, /*verbose=*/false);
                        sel.select();
                        const auto& sel_idx = sel.getSelectedIndices();

                        int tp = 0;
                        for (const auto idx : sel_idx)
                            if (std::find(true_supp.begin(), true_supp.end(), idx)
                                != true_supp.end()) ++tp;
                        const auto k = static_cast<int>(sel_idx.size());
                        fdr_out = (k == 0) ? 0.0
                            : static_cast<double>(k - tp) / static_cast<double>(k);
                        tpr_out = true_supp.empty() ? 0.0
                            : static_cast<double>(tp)
                              / static_cast<double>(true_supp.size());
                    };

                    score(y_plugin, fdr_p[mc], tpr_p[mc]);
                    score(y_oracle, fdr_o[mc], tpr_o[mc]);
                }

                const double dMC = static_cast<double>(cfg.num_MC);
                return {{"fdr_plugin", std::accumulate(fdr_p.begin(), fdr_p.end(), 0.0) / dMC},
                        {"tpr_plugin", std::accumulate(tpr_p.begin(), tpr_p.end(), 0.0) / dMC},
                        {"fdr_oracle", std::accumulate(fdr_o.begin(), fdr_o.end(), 0.0) / dMC},
                        {"tpr_oracle", std::accumulate(tpr_o.begin(), tpr_o.end(), 0.0) / dMC}};
            };
            std::ostringstream key;
            key << "heatmap|snr=" << snr << "|tfdr=" << tfdr << "|seed=" << cfg.base_seed
                << "|mc=" << cfg.num_MC;
            const auto cell = TREX_CELL_CACHED(key.str(), compute_cell);
            const auto ei_s = static_cast<Eigen::Index>(si);
            const auto ei_t = static_cast<Eigen::Index>(ti);
            res[0](ei_s, ei_t) = trex_cell_cache::get(cell, "fdr_plugin");
            res[1](ei_s, ei_t) = trex_cell_cache::get(cell, "tpr_plugin");
            res[2](ei_s, ei_t) = trex_cell_cache::get(cell, "fdr_oracle");
            res[3](ei_s, ei_t) = trex_cell_cache::get(cell, "tpr_oracle");

            std::cout << "  SNR=" << std::setw(5) << snr
                      << "  tFDR=" << std::fixed << std::setprecision(2) << tfdr
                      << "  ->  FDR " << std::setprecision(3)
                      << res[0](ei_s, ei_t) << " (plugin) / "
                      << res[2](ei_s, ei_t) << " (oracle)"
                      << " | TPR " << res[1](ei_s, ei_t) << " / "
                      << res[3](ei_s, ei_t) << "\n" << std::flush;
        }
    }

    // ---- console matrices ----
    const std::vector<std::string> resp_names   = {"plugin", "plugin", "oracle", "oracle"};
    const std::vector<std::string> metric_names = {"FDR", "TPR", "FDR", "TPR"};
    for (std::size_t r = 0; r < 4; ++r) {
        std::cout << "\n  Realized " << metric_names[r]
                  << " [" << resp_names[r] << " response]  (rows: SNR dB, cols: tFDR)\n      ";
        for (double t : tfdr_grid)
            std::cout << std::setw(7) << std::fixed << std::setprecision(2) << t;
        std::cout << "\n";
        for (std::size_t si = 0; si < n_snr; ++si) {
            std::cout << "  " << std::setw(4) << std::setprecision(0) << snr_grid[si];
            for (std::size_t ti = 0; ti < n_tfdr; ++ti)
                std::cout << std::setw(7) << std::setprecision(3)
                          << res[r](static_cast<Eigen::Index>(si),
                                    static_cast<Eigen::Index>(ti));
            std::cout << "\n";
        }
    }
    std::cout << "\n";

    // ---- tidy CSV ----
    const std::string folder = DEMO_OUTPUT_DIR;
    std::filesystem::create_directories(folder);
    const std::string csv_path = folder +
        std::string("demo_trex_spca_02_mc_sim_pev_fdr_heatmap.csv");
    std::ofstream csv(csv_path);
    if (csv.is_open()) {
        csv << "response,metric,tfdr,snr_db,value\n"
            << std::fixed << std::setprecision(6);
        for (std::size_t r = 0; r < 4; ++r)
            for (std::size_t si = 0; si < n_snr; ++si)
                for (std::size_t ti = 0; ti < n_tfdr; ++ti)
                    csv << resp_names[r] << "," << metric_names[r] << ","
                        << tfdr_grid[ti] << "," << snr_grid[si] << ","
                        << res[r](static_cast<Eigen::Index>(si),
                                  static_cast<Eigen::Index>(ti)) << "\n";
        std::cout << "[Info] Heatmap CSV saved to: " << csv_path << "\n\n";
    } else {
        std::cout << "[Warning] Could not open CSV file: " << csv_path << "\n\n";
    }
}


// ==============================================================================
// Main
// ==============================================================================

int main() {

    std::cout.setf(std::ios::unitbuf);
    omp_set_num_threads(trex_mc_workers::count());
    std::cout << "Running with " << omp_get_max_threads() << " threads\n\n";

    // Defaults held fixed while a single axis is swept. p1 = 10 follows Fig. 3
    // of the reference paper (demo 01 mirrors Fig. 2 and uses p1 = 5).
    SimConfig cfg;
    cfg.n                 = 50;
    cfg.p                 = 100;
    cfg.p1                = 10;
    cfg.M                 = 3;
    cfg.overlap_pool_size = 30;
    cfg.tFDR              = 0.10;
    cfg.num_MC            = 200;
    // Optional pilot-run override, e.g. TREX_SPCA_NUM_MC=10 ./demo_...
    if (const char* env_mc = std::getenv("TREX_SPCA_NUM_MC"))
        cfg.num_MC = static_cast<std::size_t>(std::stoul(env_mc));
    // Part selection, e.g. TREX_SPCA_PARTS=5 runs only the FDR/TPR heatmap.
    const std::string parts = [] {
        const char* env = std::getenv("TREX_SPCA_PARTS");
        return std::string(env ? env : "12345");
    }();
    const auto run_part = [&parts](char c) {
        return parts.find(c) != std::string::npos;
    };
    cfg.base_seed         = 42;
    cfg.lambda_2          = -1;   // -1 triggers k-fold CV for lambda_2 selection

    // =========================================================
    // Part 1: number of extracted PCs
    // =========================================================
    if (run_part('1'))
    {
        // The DGP keeps its M = 3 true factors throughout; only the number of
        // components the methods extract varies, so every PC beyond the third
        // can contribute Null EV and nothing else.
        //
        // The grid stops at 49, not 50: X is centered, so rank(X) <= n - 1 = 49
        // and a 50th component carries no variance at all. Asking the per-PC
        // selector to regress on that null direction throws outright ("lambda
        // grid anchor is degenerate"), because its response vector is constant.
        // 49 is therefore the largest attainable PC count on this design, not a
        // safety margin.
        const std::vector<Eigen::Index> pc_grid = {1, 3, 5, 10, 20, 30, 40, 49};

        std::vector<SweepPoint> points;
        for (Eigen::Index n_pcs : pc_grid) {
            SweepPoint pt;
            pt.snr        = 0.0;
            pt.p1         = cfg.p1;
            pt.tFDR       = cfg.tFDR;
            pt.num_comp   = n_pcs;
            pt.axis_value = static_cast<double>(n_pcs);
            points.push_back(pt);
        }

        run_sweep(cfg, points, "n_pcs", "#PCs", 0,
                  "T-Rex SPCA — Cumulative PEV vs. number of extracted PCs",
                  "demo_trex_spca_02_mc_sim_pev_pcs");
    }

    // =========================================================
    // Part 2: signal-to-noise ratio
    // =========================================================
    if (run_part('2'))
    {
        const std::vector<double> snr_grid =
            {-10.0, -7.0, -5.0, -3.0, 0.0, 3.0, 5.0, 7.0, 10.0};

        std::vector<SweepPoint> points;
        for (double snr : snr_grid) {
            SweepPoint pt;
            pt.snr        = snr;
            pt.p1         = cfg.p1;
            pt.tFDR       = cfg.tFDR;
            pt.num_comp   = 0;      // = M = 3 true factors
            pt.axis_value = snr;
            points.push_back(pt);
        }

        run_sweep(cfg, points, "snr_db", "SNR(dB)", 1,
                  "T-Rex SPCA — Cumulative PEV (PCs 1-3) vs. SNR",
                  "demo_trex_spca_02_mc_sim_pev_snr",
                  /*pev_only_table=*/true);
    }

    // =========================================================
    // Part 3: number of true active loadings
    // =========================================================
    if (run_part('3'))
    {
        // Bounded above by overlap_pool_size = 30: each factor draws its support
        // from that shared pool, so p1 cannot exceed it.
        const std::vector<Eigen::Index> p1_grid = {1, 5, 10, 15, 20, 25, 30};

        std::vector<SweepPoint> points;
        for (Eigen::Index p1 : p1_grid) {
            SweepPoint pt;
            pt.snr        = 0.0;
            pt.p1         = p1;
            pt.tFDR       = cfg.tFDR;
            pt.num_comp   = 0;
            pt.axis_value = static_cast<double>(p1);
            points.push_back(pt);
        }

        run_sweep(cfg, points, "p1", "#Active", 0,
                  "T-Rex SPCA — Cumulative PEV (PCs 1-3) vs. number of true active loadings",
                  "demo_trex_spca_02_mc_sim_pev_loadings",
                  /*pev_only_table=*/true);
    }

    // =========================================================
    // Part 4: target FDR
    // =========================================================
    if (run_part('4'))
    {
        const std::vector<double> tfdr_grid =
            {0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50};

        std::vector<SweepPoint> points;
        for (double tfdr : tfdr_grid) {
            SweepPoint pt;
            pt.snr        = 0.0;
            pt.p1         = cfg.p1;
            pt.tFDR       = tfdr;
            pt.num_comp   = 0;
            pt.axis_value = tfdr;
            points.push_back(pt);
        }

        run_sweep(cfg, points, "tfdr", "tFDR", 2,
                  "T-Rex SPCA — Cumulative PEV (PCs 1-3) vs. target FDR",
                  "demo_trex_spca_02_mc_sim_pev_tfdr",
                  /*pev_only_table=*/true);
    }

    // =========================================================
    // Part 5: FDR/TPR heatmaps over (target FDR x SNR)
    // =========================================================
    if (run_part('5'))
        run_fdr_tpr_heatmap(cfg);

    return 0;
}
