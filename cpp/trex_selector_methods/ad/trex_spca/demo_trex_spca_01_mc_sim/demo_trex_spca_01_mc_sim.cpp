// ==============================================================================
// demo_trex_spca_01_mc_sim.cpp
// ==============================================================================
/**
 * @file demo_trex_spca_01_mc_sim.cpp
 *
 * @brief Monte Carlo simulation comparing T-Rex SPCA against PCA baselines.
 *
 * @details
 *  DGP: sparse M-factor model — X = Z * V^T + E.
 *       n=50, p=100, M=3 latent factors, p1=5 active loadings per factor,
 *       overlap_pool_size=30 (shared candidate-index pool across factors).
 *
 *  Section 1: SNR sweep over the snr_values grid set in main()
 *             (currently {-10, -7, -5, -3, 0, 3, 5, 7, 10} dB, num_MC=200 —
 *             the legacy CRAN reference grid).
 *    Methods compared:
 *      1. OrdPCA             — ordinary PCA (baseline; no sparsity constraint)
 *      2. OraclePCA          — thresholded PCA with known support cardinality p1
 *      3. TRexSPCA-EN-Act    — T-Rex+GVS(EN/TENET, Gram-based) + ActiveSet
 *      4. TRexSPCA-ENaug-Act — T-Rex+GVS(EN/TENETAug, augmented) + ActiveSet
 *      5. TRexSPCA-EN-Thr    — T-Rex+GVS(EN/TENET, Gram-based) + Thresholded
 *      6. TRexSPCA-ENaug-Thr — T-Rex+GVS(EN/TENETAug, augmented) + Thresholded
 *
 *    All methods see the same center-only X (covariance-PCA footing, as in the
 *    legacy R reference); the per-PC T-Rex selector uses the default L2 scaling.
 *
 *    EN augmentation follows lm_dummy.R exactly:
 *      X_aug = d₂·[X; d₁·Iₚ; 0_L],  D_aug = d₂·[D; 0_p; d₁·I_L],
 *      y_aug = [y; 0_{p+L}],  then column-wise centre + L2-normalise.
 *
 *  Metrics (num_MC-averaged; TPR/FDR on PC1 support only — see evaluate_spca):
 *    - FDR  — false discovery rate of PC1's estimated loading support
 *    - TPR  — true positive rate  of PC1's estimated loading support
 *    - PEV  — cumulative proportion of explained variance (all M components),
 *             reported under three normalizations: by the total variance of X
 *             (bounded by 1); by the Signal + Mixed EV OF THE DATA — the
 *             variance carried by the true active variables — per Definition 1
 *             of the reference paper (may exceed 1: the overshoot is null
 *             variance being reported as explained variance); and the method's
 *             own Sig+Mix part over that same denominator (the paper's
 *             "Ordinary PCA (Sig + Mix)" reference when read for OrdPCA).
 *             The Definition-1 denominator follows the active-column
 *             convention of the legacy R reference
 *             (demo_trex_spca_03_fig3.R, mode "active").
 *
 *  Section 2: union-support FDR per PC over the same SNR grid — each PC's
 *             selected support scored against the UNION of the factor
 *             supports, the only well-posed per-PC truth beyond PC1; see
 *             demo_trex_spca_union_fdr_sweep.
 *
 *  Section selection: TREX_SPCA_PARTS (default "12"); TREX_SPCA_NUM_MC
 *  overrides the MC count.
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

// ==============================================================================
// Monte Carlo Simulation — SNR sweep
// ==============================================================================

/**
 * @brief Run the SNR-sweep MC simulation and save results.
 *
 * @param cfg        Simulation parameters (n, p, p1, M, tFDR, num_MC, seed).
 * @param snr_values SNR grid in dB.
 * @param stem       Output file stem (no folder, no extension).
 */
void demo_trex_spca_mc_snr_sweep(const SimConfig&           cfg,
                                 const std::vector<double>& snr_values,
                                 const std::string&         stem)
{
    std::cout << "================================================================\n";
    std::cout << "  T-Rex SPCA MC Simulation — SNR Sweep\n";
    std::cout << "================================================================\n";
    std::cout << "  n=" << cfg.n << "  p=" << cfg.p << "  M=" << cfg.M
              << "  p1=" << cfg.p1 << "  tFDR=" << cfg.tFDR
              << "  num_MC=" << cfg.num_MC << "\n\n";

    const std::vector<std::string> method_names = {
        "OrdPCA",
        "OraclePCA",
        "TRexSPCA-EN-Act",
        "TRexSPCA-ENaug-Act",
        "TRexSPCA-EN-Thr",
        "TRexSPCA-ENaug-Thr"
    };

    std::map<std::string, Eigen::VectorXd> fdr_map, tpr_map, pev_map, pev_sig_map,
                                           pev_sigmix_map;
    for (const auto& name : method_names) {
        fdr_map[name] = Eigen::VectorXd(snr_values.size());
        tpr_map[name] = Eigen::VectorXd(snr_values.size());
        pev_map[name] = Eigen::VectorXd(snr_values.size());
        pev_sig_map[name] = Eigen::VectorXd(snr_values.size());
        pev_sigmix_map[name] = Eigen::VectorXd(snr_values.size());
    }

    for (std::size_t si = 0; si < snr_values.size(); ++si) {
        const double   snr     = snr_values[si];
        const int      snr_int = static_cast<int>(std::round(snr));
        const unsigned offset  = static_cast<unsigned>(cfg.base_seed + snr_int);

        std::ostringstream snr_ss;
        snr_ss << std::fixed << std::setprecision(1) << snr;
        const std::string snr_lbl = "SNR=" + snr_ss.str() + " dB";

        std::cout << "--------------------------------------------------\n";
        std::cout << snr_lbl << "\n\n";

        // Thread safe parameter capture all DGP parameters by value for OMP thread safety
        const auto make_data = [cfg, snr](unsigned seed) -> FactorModelData {
            return DataGenerator::generate_sparse_factor_model(
                cfg.n, cfg.p, cfg.p1, cfg.M, snr,
                cfg.overlap_pool_size, seed);
        };

        const auto ei = static_cast<Eigen::Index>(si);

        // 1. Ordinary PCA
        auto r1 = run_mc_trials_pca(
            cfg.num_MC, snr_lbl + " [OrdPCA]", make_data, offset);
        fdr_map["OrdPCA"](ei) = r1.avg_fdr;
        tpr_map["OrdPCA"](ei) = r1.avg_tpr;
        pev_map["OrdPCA"](ei) = r1.avg_pev;
        pev_sig_map["OrdPCA"](ei) = r1.avg_pev_sig;
        pev_sigmix_map["OrdPCA"](ei) = r1.avg_pev_sigmix;


        // 2. Oracle Thresholded PCA
        auto r2 = run_mc_trials_oracle_pca(
            cfg.num_MC, snr_lbl +
            " [OraclePCA]", make_data, cfg.p1, offset
        );

        fdr_map["OraclePCA"](ei) = r2.avg_fdr;
        tpr_map["OraclePCA"](ei) = r2.avg_tpr;
        pev_map["OraclePCA"](ei) = r2.avg_pev;
        pev_sig_map["OraclePCA"](ei) = r2.avg_pev_sig;
        pev_sigmix_map["OraclePCA"](ei) = r2.avg_pev_sigmix;


        // 3. T-Rex SPCA — EN (Gram-based TENET) + ActiveSet
        auto r3 = run_mc_trials_trex_spca(
            cfg.num_MC, snr_lbl + " [TRexSPCA-EN-Act]",
            make_data, cfg.tFDR,
            SPCAMode::ActiveSet,
            offset,
            cfg.lambda_2,
            ENSolverType::TENET);
        fdr_map["TRexSPCA-EN-Act"](ei) = r3.avg_fdr;
        tpr_map["TRexSPCA-EN-Act"](ei) = r3.avg_tpr;
        pev_map["TRexSPCA-EN-Act"](ei) = r3.avg_pev;
        pev_sig_map["TRexSPCA-EN-Act"](ei) = r3.avg_pev_sig;
        pev_sigmix_map["TRexSPCA-EN-Act"](ei) = r3.avg_pev_sigmix;


        // 4. T-Rex SPCA — ENaug (augmented TENET) + ActiveSet
        auto r4 = run_mc_trials_trex_spca(
            cfg.num_MC, snr_lbl + " [TRexSPCA-ENaug-Act]",
            make_data, cfg.tFDR,
            SPCAMode::ActiveSet,
            offset,
            cfg.lambda_2,
            ENSolverType::TENET_AUG);
        fdr_map["TRexSPCA-ENaug-Act"](ei) = r4.avg_fdr;
        tpr_map["TRexSPCA-ENaug-Act"](ei) = r4.avg_tpr;
        pev_map["TRexSPCA-ENaug-Act"](ei) = r4.avg_pev;
        pev_sig_map["TRexSPCA-ENaug-Act"](ei) = r4.avg_pev_sig;
        pev_sigmix_map["TRexSPCA-ENaug-Act"](ei) = r4.avg_pev_sigmix;


        // 5. T-Rex SPCA — EN (Gram-based TENET) + Thresholded
        auto r5 = run_mc_trials_trex_spca(
            cfg.num_MC, snr_lbl + " [TRexSPCA-EN-Thr]",
            make_data, cfg.tFDR,
            SPCAMode::Thresholded,
            offset,
            cfg.lambda_2,
            ENSolverType::TENET);
        fdr_map["TRexSPCA-EN-Thr"](ei) = r5.avg_fdr;
        tpr_map["TRexSPCA-EN-Thr"](ei) = r5.avg_tpr;
        pev_map["TRexSPCA-EN-Thr"](ei) = r5.avg_pev;
        pev_sig_map["TRexSPCA-EN-Thr"](ei) = r5.avg_pev_sig;
        pev_sigmix_map["TRexSPCA-EN-Thr"](ei) = r5.avg_pev_sigmix;


        // 6. T-Rex SPCA — ENaug (augmented TENET) + Thresholded
        auto r6 = run_mc_trials_trex_spca(
            cfg.num_MC, snr_lbl + " [TRexSPCA-ENaug-Thr]",
            make_data, cfg.tFDR,
            SPCAMode::Thresholded,
            offset,
            cfg.lambda_2,
            ENSolverType::TENET_AUG);
        fdr_map["TRexSPCA-ENaug-Thr"](ei) = r6.avg_fdr;
        tpr_map["TRexSPCA-ENaug-Thr"](ei) = r6.avg_tpr;
        pev_map["TRexSPCA-ENaug-Thr"](ei) = r6.avg_pev;
        pev_sig_map["TRexSPCA-ENaug-Thr"](ei) = r6.avg_pev_sig;
        pev_sigmix_map["TRexSPCA-ENaug-Thr"](ei) = r6.avg_pev_sigmix;
    }

    save_and_print_spca_mc_results(cfg.num_MC, stem,
                                    snr_values, method_names,
                                    fdr_map, tpr_map, pev_map, pev_sig_map,
                                    pev_sigmix_map);
}


// ==============================================================================
// Section 2: union-support FDR per PC
// ==============================================================================

/**
 * @brief Realized FDR of each PC's selected support, scored against the UNION
 *        of the true factor supports.
 *
 * @details Beyond PC1 there is no unambiguous per-factor ground truth: the
 *  factor supports overlap, and PCA's orthogonality constraint mixes the later
 *  components across factors, so a per-factor FDR would penalize legitimately
 *  selected leaked variables and measure the ambiguity of the truth rather
 *  than the selector. The UNION of the factor supports, however, is well-posed
 *  for every component — a variable outside the union is null for EVERY
 *  factor, so selecting it is a false discovery no matter how the components
 *  mix. This sweep answers the one question about the later PCs that CAN be
 *  asked cleanly: does the selector start harvesting pure-noise variables on
 *  components beyond the first?
 *
 *  One variant is run: ActiveSet + TENET — the selector's own support (the
 *  Thresholded mode rebuilds loadings at the same cardinality). Reported per
 *  PC m: mean union-FDR and mean support size |A_m|.
 *
 *  Output: tidy CSV "pc,metric,snr_db,value" (metrics FDRunion / k).
 */
void demo_trex_spca_union_fdr_sweep(const SimConfig&           cfg,
                                    const std::vector<double>& snr_values,
                                    const std::string&         stem)
{
    std::cout << "================================================================\n";
    std::cout << "  T-Rex SPCA — union-support FDR per PC (ActiveSet/TENET)\n";
    std::cout << "================================================================\n";
    std::cout << "  n=" << cfg.n << "  p=" << cfg.p << "  M=" << cfg.M
              << "  p1=" << cfg.p1 << "  tFDR=" << cfg.tFDR
              << "  num_MC=" << cfg.num_MC << "\n\n";

    const int          iMC = static_cast<int>(cfg.num_MC);
    const Eigen::Index M   = cfg.M;
    const auto         n_snr = static_cast<Eigen::Index>(snr_values.size());

    Eigen::MatrixXd fdr_mean = Eigen::MatrixXd::Zero(M, n_snr);
    Eigen::MatrixXd k_mean   = Eigen::MatrixXd::Zero(M, n_snr);

    for (Eigen::Index si = 0; si < n_snr; ++si) {
        const double   snr    = snr_values[static_cast<std::size_t>(si)];
        const unsigned offset = static_cast<unsigned>(
            cfg.base_seed + static_cast<int>(std::round(snr)));

        // One SNR cell (per-PC union-FDR and support size), restored from the
        // cache after a restart.
        const auto compute_cell = [&]() -> trex_cell_cache::Cell {
            Eigen::MatrixXd fdr_trial = Eigen::MatrixXd::Zero(iMC, M);
            Eigen::MatrixXd k_trial   = Eigen::MatrixXd::Zero(iMC, M);

            const auto make_data = [cfg, snr](unsigned seed) -> FactorModelData {
                return DataGenerator::generate_sparse_factor_model(
                    cfg.n, cfg.p, cfg.p1, cfg.M, snr,
                    cfg.overlap_pool_size, seed);
            };

            #pragma omp parallel for schedule(dynamic)
            for (int mc = 0; mc < iMC; ++mc) {
                const unsigned seed = offset + static_cast<unsigned>(mc) * 1000u;
                auto dat = make_data(seed);
                center_columns(dat.X);

                // Union of the true factor supports.
                std::vector<char> in_union(static_cast<std::size_t>(dat.X.cols()), 0);
                for (Eigen::Index j = 0; j < dat.V.rows(); ++j)
                    for (Eigen::Index m = 0; m < dat.V.cols(); ++m)
                        if (dat.V(j, m) != 0.0) {
                            in_union[static_cast<std::size_t>(j)] = 1;
                            break;
                        }

                TRexSPCAControlParameter ctrl;
                ctrl.mode                    = SPCAMode::ActiveSet;
                ctrl.en_solver               = ENSolverType::TENET;
                ctrl.gvs_ctrl.trex_ctrl.scaling_mode = ScalingMode::L2;
                ctrl.gvs_ctrl.lambda2_method = LambdaSelectionMethod::CV_1SE_CCD;
                ctrl.gvs_ctrl.lambda_2       = cfg.lambda_2;

                Eigen::Map<Eigen::MatrixXd> X_map(dat.X.data(),
                                                  dat.X.rows(), dat.X.cols());
                TRexSPCA       spca(X_map, M, cfg.tFDR, ctrl, -1);
                TRexSPCAResult res = spca.select();

                for (Eigen::Index m = 0; m < M; ++m) {
                    const auto& A = res.active_sets[static_cast<std::size_t>(m)];
                    int fd = 0;
                    for (const auto idx : A)
                        if (!in_union[static_cast<std::size_t>(idx)]) ++fd;
                    fdr_trial(mc, m) = A.empty() ? 0.0
                        : static_cast<double>(fd) / static_cast<double>(A.size());
                    k_trial(mc, m) = static_cast<double>(A.size());
                }
            }

            trex_cell_cache::Cell c;
            for (Eigen::Index m = 0; m < M; ++m) {
                c.emplace_back("fdr" + std::to_string(m), fdr_trial.col(m).mean());
                c.emplace_back("k" + std::to_string(m), k_trial.col(m).mean());
            }
            return c;
        };
        std::ostringstream key;
        key << "union_fdr|snr=" << snr << "|seed=" << offset << "|mc=" << iMC << "|tfdr=" << cfg.tFDR;
        const auto cell = TREX_CELL_CACHED(key.str(), compute_cell);
        for (Eigen::Index m = 0; m < M; ++m) {
            fdr_mean(m, si) = trex_cell_cache::get(cell, "fdr" + std::to_string(m));
            k_mean(m, si)   = trex_cell_cache::get(cell, "k" + std::to_string(m));
        }
        std::cout << "  SNR=" << std::setw(5) << snr << " dB — union-FDR:";
        for (Eigen::Index m = 0; m < M; ++m)
            std::cout << "  PC" << (m + 1) << "=" << std::fixed
                      << std::setprecision(3) << fdr_mean(m, si);
        std::cout << "\n" << std::flush;
    }

    // ---- console table ----
    std::cout << "\n  Union-support FDR per PC (rows) over SNR (cols):\n      ";
    for (double s : snr_values)
        std::cout << std::setw(8) << std::setprecision(1) << s;
    std::cout << "\n";
    for (Eigen::Index m = 0; m < M; ++m) {
        std::cout << "  PC" << (m + 1) << " ";
        for (Eigen::Index si = 0; si < n_snr; ++si)
            std::cout << std::setw(8) << std::setprecision(3) << fdr_mean(m, si);
        std::cout << "\n";
    }
    std::cout << "\n  Mean support size |A_m|:\n";
    for (Eigen::Index m = 0; m < M; ++m) {
        std::cout << "  PC" << (m + 1) << " ";
        for (Eigen::Index si = 0; si < n_snr; ++si)
            std::cout << std::setw(8) << std::setprecision(2) << k_mean(m, si);
        std::cout << "\n";
    }
    std::cout << "\n";

    // ---- tidy CSV ----
    const std::string folder = DEMO_OUTPUT_DIR;
    std::filesystem::create_directories(folder);
    const std::string csv_path = folder + stem + ".csv";
    std::ofstream csv(csv_path);
    if (csv.is_open()) {
        csv << "pc,metric,snr_db,value\n" << std::fixed << std::setprecision(6);
        for (Eigen::Index m = 0; m < M; ++m)
            for (Eigen::Index si = 0; si < n_snr; ++si) {
                csv << (m + 1) << ",FDRunion,"
                    << snr_values[static_cast<std::size_t>(si)] << ","
                    << fdr_mean(m, si) << "\n";
                csv << (m + 1) << ",k,"
                    << snr_values[static_cast<std::size_t>(si)] << ","
                    << k_mean(m, si) << "\n";
            }
        std::cout << "[Info] Union-FDR CSV saved to: " << csv_path << "\n\n";
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


    SimConfig cfg;
    cfg.n                 = 50;
    cfg.p                 = 100;
    cfg.p1                = 5;
    cfg.M                 = 3;
    cfg.overlap_pool_size = 30;
    cfg.tFDR              = 0.10;
    cfg.num_MC            = 200;
    // Optional pilot-run override, e.g. TREX_SPCA_NUM_MC=10 ./demo_...
    if (const char* env_mc = std::getenv("TREX_SPCA_NUM_MC"))
        cfg.num_MC = static_cast<std::size_t>(std::stoul(env_mc));
    cfg.base_seed         = 42;
    cfg.lambda_2          = -1; // -1 triggers k-fold CV for lambda_2 selection

    const std::vector<double> snr_values =
        {-10.0, -7.0, -5, -3.0, 0, 3, 5, 7, 10}; // legacy CRAN reference grid

    // Part selection, e.g. TREX_SPCA_PARTS=2 runs only the union-FDR sweep.
    const std::string parts = [] {
        const char* env = std::getenv("TREX_SPCA_PARTS");
        return std::string(env ? env : "12");
    }();
    const auto run_part = [&parts](char c) {
        return parts.find(c) != std::string::npos;
    };

    // =========================================================
    // Section 1: SNR sweep
    // =========================================================
    if (run_part('1'))
        demo_trex_spca_mc_snr_sweep(cfg, snr_values, "demo_trex_spca_01_mc_sim");

    // =========================================================
    // Section 2: union-support FDR per PC
    // =========================================================
    if (run_part('2'))
        demo_trex_spca_union_fdr_sweep(cfg, snr_values,
                                       "demo_trex_spca_01_mc_sim_union_fdr");

    return 0;
}
