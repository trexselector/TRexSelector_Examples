// ==============================================================================
// trex_spca_sim_utils.hpp
// ==============================================================================
/**
 * @file trex_spca_sim_utils.hpp
 *
 * @brief Shared simulation utilities for T-Rex SPCA demo files.
 *
 * @details Provides all types, data generation, output helpers, and MC loop
 *          infrastructure shared by T-Rex SPCA demo files, all under namespace
 *          spca_sim:
 *
 *   Data generation:
 *    - FactorModelData              — data container (X, Z, V, E, actual_snr_db)
 *    - DataGenerator                — sparse factor model data generator
 *
 *   Configuration:
 *    - SimConfig                    — simulation parameter struct
 *    - SPCADGPFactory               — DGP factory callable type: seed → FactorModelData
 *
 *   Results:
 *    - SPCASingleResult             — per-trial metrics (mean TPR/FDR over M components, PEV)
 *    - SPCAGridPointResult          — MC-aggregated (avg/sd TPR, FDR; avg PEV)
 *
 *   Evaluation:
 *    - evaluate_spca()              — compute SPCASingleResult from V_est, Z_est, V_true
 *
 *   MC loop functions:
 *    - run_mc_trials_trex_spca()    — parallel MC loop for T-Rex SPCA
 *    - run_mc_trials_pca()          — parallel MC loop for ordinary PCA baseline
 *    - run_mc_trials_oracle_pca()   — parallel MC loop for oracle thresholded PCA
 *
 *   Output:
 *    - save_and_print_spca_mc_results() — table + tidy CSV (console + file)
 *
 * Parallelism pattern:
 *   #pragma omp parallel for schedule(dynamic)
 *   for (int mc = 0; mc < iMC; ++mc) { ... }
 *
 * Thread safety:
 *   - Each trial owns its data (returned by-value from the DGP factory).
 *   - DataGenerator::generate_sparse_factor_model() seeds its own RNG from the
 *     trial seed — no shared mutable state.
 *   - PCA and TRexSPCA objects are created per thread (stack-local).
 *   - DGP factory lambdas must capture all parameters by value.
 *
 * @note Demo-internal header — not part of the TRexSelector library interface.
 *       Never include from library code.
 */
// ==============================================================================

#ifndef DEMOS_TREX_SELECTOR_METHODS_TREX_SPCA_SIM_UTILS_HPP
#define DEMOS_TREX_SELECTOR_METHODS_TREX_SPCA_SIM_UTILS_HPP

// std includes
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <trex_selector_methods/ad/trex_gvs/trex_gvs.hpp>
#include <vector>

// Eigen includes
#include <Eigen/Dense>

// OpenMP compatibility layer
#include <utils/openmp/utils_openmp.hpp>

// T-Rex SPCA includes
#include <trex_selector_methods/ad/trex_spca/trex_spca.hpp>

// PCA baseline
#include <ml_methods/pca/pca.hpp>

// Resumable grid cells
#include "../../cell_cache.hpp"
#include "../../mc_workers.hpp"   // thread budget (TREX_NUM_WORKERS)


// ==============================================================================
// Namespace
// ==============================================================================

namespace spca_sim {

// Namespace aliases
namespace pca_ns = trex::ml_methods::pca;
using trex::trex_selector_methods::ad::trex_gvs::LambdaSelectionMethod;
using trex::trex_selector_methods::ad::trex_gvs::ENSolverType;
using trex::trex_selector_methods::ad::utils::data_normalizer::ScalingMode;
using trex::trex_selector_methods::ad::trex_spca::TRexSPCA;
using trex::trex_selector_methods::ad::trex_spca::TRexSPCAControlParameter;
using trex::trex_selector_methods::ad::trex_spca::TRexSPCAResult;
using trex::trex_selector_methods::ad::trex_spca::SPCAMode;


// ==============================================================================
// Data container
// ==============================================================================

/** @brief Container for generated synthetic data and ground-truth support. */
struct FactorModelData {
    /** @brief Centered observation matrix (n x p). */
    Eigen::MatrixXd X;

    /** @brief True factor matrix (n x M). */
    Eigen::MatrixXd Z;

    /** @brief True sparse loadings matrix (p x M). */
    Eigen::MatrixXd V;

    /** @brief Scaled noise matrix (n x p). */
    Eigen::MatrixXd E;

    /** @brief Actual signal-to-noise ratio (dB). */
    double actual_snr_db;
};


// ==============================================================================
// Data generator
// ==============================================================================

/**
 * @brief Synthetic data generator for the sparse factor model.
 *
 * @details Generates data from the model X = Z * V^T + E,
 *  where Z (n x M) has column-wise N(0, sigma_m^2) factors with
 *  sigma = {5, 3, 1} (falling back to 1 for M > 3), V (p x M) has sparse
 *  loadings (exactly p1 non-zeros per column drawn without replacement from a
 *  shared overlap pool of size overlap_pool_size), and E (n x p) is i.i.d.
 *  Gaussian noise scaled to match the target SNR in dB.
 */
class DataGenerator {
public:

    /**
     * @brief Generate a single dataset from the sparse factor model.
     *
     * @param n                  Number of samples (default 50).
     * @param p                  Number of features (default 100).
     * @param p1                 Number of true active loadings per factor (default 5).
     * @param M                  Number of latent factors (default 3).
     * @param target_snr         Target signal-to-noise ratio in dB (default 0.0).
     * @param overlap_pool_size  Size of the shared candidate-index pool from which
     *                           each factor's support is drawn (default 30). Controls
     *                           how much factor supports can overlap.
     * @param seed               Random seed for reproducibility (default 42).
     *
     * @return FactorModelData containing the generated data and true support.
     */
    static FactorModelData generate_sparse_factor_model(
        Eigen::Index n                 = 50,
        Eigen::Index p                 = 100,
        Eigen::Index p1                = 5,
        Eigen::Index M                 = 3,
        double       target_snr        = 0.0,
        Eigen::Index overlap_pool_size = 30,
        unsigned int seed              = 42
    ) {
        if (p1 > p)
            throw std::invalid_argument("p1 must be less than or equal to p.");
        if (n <= 0 || p <= 0 || p1 <= 0)
            throw std::invalid_argument("n, p, and p1 must be positive integers.");

        FactorModelData data;
        std::mt19937 rng(seed);


        // 1. True factors Z (n x M) with decreasing signal strength
        data.Z = Eigen::MatrixXd(n, M);
        const std::vector<double> factor_stds = {5.0, 3.0, 1.0};
        for (Eigen::Index m = 0; m < M; ++m) {
            const double std_dev = (m < static_cast<Eigen::Index>(factor_stds.size()))
                                 ? factor_stds[m] : 1.0;
            std::normal_distribution<double> nd(0.0, std_dev);
            data.Z.col(m) = Eigen::VectorXd::NullaryExpr(n, [&]() { return nd(rng); });
        }


        // 2. Sparse loadings V (p x M): each factor draws p1 indices from pool
        data.V = Eigen::MatrixXd::Zero(p, M);

        const Eigen::Index actual_pool = std::min(overlap_pool_size, p);
        if (p1 > actual_pool)
            throw std::invalid_argument("p1 must be less than or equal to the overlap pool size.");

        std::vector<Eigen::Index> candidates(actual_pool);
        std::iota(candidates.begin(), candidates.end(), 0);

        for (Eigen::Index m = 0; m < M; ++m) {
            std::shuffle(candidates.begin(), candidates.end(), rng);
            for (Eigen::Index k = 0; k < p1; ++k)
                data.V(candidates[k], m) = 0.9;
        }


        // 3. Signal matrix S = Z * V^T
        const Eigen::MatrixXd S = data.Z * data.V.transpose();

        const double signal_mean = S.mean();
        const double signal_var  = (S.array() - signal_mean).square().sum()
                                 / static_cast<double>(S.size() - 1);


        // 4. Gaussian noise E scaled to target SNR
        // SNR_dB = 10 * log10(var_S / var_E)  =>  var_E = var_S / 10^(SNR_dB / 10)
        std::normal_distribution<double> std_normal(0.0, 1.0);
        Eigen::MatrixXd E_raw = Eigen::MatrixXd::NullaryExpr(n, p, [&]() {
            return std_normal(rng);
        });

        const double raw_noise_var       = (E_raw.array() - E_raw.mean()).square().sum()
                                         / static_cast<double>(E_raw.size() - 1);
        const double target_noise_var    = signal_var / std::pow(10.0, target_snr / 10.0);
        const double noise_scaling_factor = std::sqrt(target_noise_var / raw_noise_var);

        data.E = E_raw * noise_scaling_factor;


        // 5. Assemble X
        data.X = S + data.E;

        const double final_noise_var = (data.E.array() - data.E.mean()).square().sum()
                                     / static_cast<double>(data.E.size() - 1);
        data.actual_snr_db = 10.0 * std::log10(signal_var / final_noise_var);

        return data;
    }

private:
    DataGenerator() = delete;
};


// ==============================================================================
// Simulation configuration
// ==============================================================================

/** @brief Simulation parameters shared across T-Rex SPCA demo sections. */
struct SimConfig {
    Eigen::Index n                 = 50;    ///< Number of samples
    Eigen::Index p                 = 100;   ///< Number of features
    Eigen::Index p1                = 5;     ///< True active loadings per factor
    Eigen::Index M                 = 3;     ///< Number of latent factors
    Eigen::Index overlap_pool_size = 30;    ///< Shared candidate-index pool size
    double       tFDR              = 0.10;  ///< Target FDR for T-Rex SPCA
    double       lambda_2          = -1.0;  ///< Ridge penalty (LARS units); < 0 = auto-compute via CV, 0 = no ridge, > 0 = fixed
    std::size_t  num_MC            = 100;   ///< Monte Carlo trials
    int          base_seed         = 42;    ///< Base RNG seed
};

/** @brief DGP factory callable type: seed → FactorModelData. */
using SPCADGPFactory = std::function<FactorModelData(unsigned seed)>;


// ==============================================================================
// Result types
// ==============================================================================

/**
 * @brief Single-trial evaluation result for a sparse PCA method.
 *
 * All per-component metrics are averaged over M components within one MC trial.
 */
struct SPCASingleResult {
    double tpr = 0.0;  ///< TPR of PC1's support only (see evaluate_spca for rationale)
    double fdr = 0.0;  ///< FDR of PC1's support only (see evaluate_spca for rationale)
    double pev = 0.0;  ///< Cumulative adjusted EV / total variance of X — bounded by 1
    double pev_sig = 0.0;  ///< Cumulative adjusted EV / data Signal+Mixed EV (Definition 1) — may exceed 1
    double pev_sigmix = 0.0;  ///< Method's own Signal+Mixed EV share / data Signal+Mixed EV — the "Sig + Mix" reference curve
};

/** @brief MC-aggregated result for one grid point (method × SNR). */
struct SPCAGridPointResult {
    double avg_tpr = 0.0;  ///< Mean TPR across MC trials
    double avg_fdr = 0.0;  ///< Mean FDR across MC trials
    double avg_pev = 0.0;  ///< Mean cumulative PEV across MC trials
    double avg_pev_sig = 0.0;  ///< Mean Definition-1 PEV across MC trials
    double avg_pev_sigmix = 0.0;  ///< Mean Sig+Mix-part PEV across MC trials
    double sd_tpr  = 0.0;  ///< Bessel-corrected SD of TPR
    double sd_fdr  = 0.0;  ///< Bessel-corrected SD of FDR
    double sd_pev  = 0.0;  ///< Bessel-corrected SD of cumulative PEV
};


// ==============================================================================
// Single-trial evaluator
// ==============================================================================

/**
 * @brief Compute per-trial sparse PCA metrics from estimated loadings and scores.
 *
 * @details
 *  - TPR / FDR are evaluated strictly on the **first** principal component (PC1).
 *    PCs 2 and 3 are omitted because PCA's orthogonality constraint mixes their
 *    loading supports across factors, making per-component support recovery
 *    metrics ambiguous beyond the first PC.
 *  - Cumulative EV is computed via QR decomposition of Z_est across all M
 *    components, and is reported under TWO normalizations: by the total
 *    variance of X (bounded by 1) and by the signal + mixed variance of the
 *    truly active columns only (the paper's definition, which may exceed 1
 *    when a method draws variance from null variables).
 *
 * @param X       Centered observation matrix (n x p). Must be the same X that was
 *                used to produce Z_est (no additional preprocessing applied here).
 * @param V_est   Estimated sparse loading matrix (p x M).
 * @param Z_est   Estimated score matrix (n x M).
 * @param V_true  True loading matrix (p x M_true); non-zeros mark the true
 *                active set. M_true may be SMALLER than the number of extracted
 *                components: sweeping the PC count past the number of true
 *                factors is exactly how Fig. 3(a) of the reference paper is
 *                built. Every position in a column beyond M_true is null by
 *                definition, so those components contribute Null EV only.
 *
 * @return SPCASingleResult: PC1 support TPR and FDR, plus both cumulative PEVs.
 */
inline SPCASingleResult evaluate_spca(
    const Eigen::MatrixXd& X,
    const Eigen::MatrixXd& V_est,
    const Eigen::MatrixXd& Z_est,
    const Eigen::MatrixXd& V_true)
{
    const Eigen::Index n = X.rows();
    const Eigen::Index p = X.cols();
    const Eigen::Index M = V_est.cols();
    // Number of TRUE factors, which need not equal the number of extracted
    // components; see the note on V_true above.
    const Eigen::Index M_true = V_true.cols();

    // -----------------------------------------------------------------
    // 1. Evaluate TPR and FDR strictly on the FIRST Component (m = 0)
    // -----------------------------------------------------------------
    // Paper methodology: PCs 2 and 3 are mixed due to the orthogonality
    // constraint of ordinary PCA.
    // True support metrics are only meaningful for the first PC.

    std::vector<Eigen::Index> true_supp_pc1, est_supp_pc1;
    for (Eigen::Index i = 0; i < p; ++i) {
        if (V_true(i, 0) != 0.0) {
            true_supp_pc1.push_back(i);
        }
        if (std::abs(V_est(i, 0)) > 1e-12) {
            est_supp_pc1.push_back(i);
        }
    }

    int tp = 0;
    for (auto idx : est_supp_pc1) {
        if (std::find(true_supp_pc1.begin(), true_supp_pc1.end(), idx)
            != true_supp_pc1.end()) {
            ++tp;
        }
    }

    const int false_disc = static_cast<int>(est_supp_pc1.size()) - tp;

    // Empty true support -> 0.0, matching the legacy R reference convention.
    double pc1_tpr = true_supp_pc1.empty() ? 0.0 : static_cast<double>(tp) / static_cast<double>(true_supp_pc1.size());
    double pc1_fdr = est_supp_pc1.empty()  ? 0.0 : static_cast<double>(false_disc) / static_cast<double>(est_supp_pc1.size());

    // ----------------------------------------------------------------------
    // 2. Cumulative explained variance across ALL components
    // ----------------------------------------------------------------------
    // The adjusted (cumulative) EV follows Zou/Hastie/Tibshirani: a QR
    // decomposition of the estimated score matrix removes the variance that
    // correlated components would otherwise double-count, so the diagonal of R
    // carries each component's *incremental* contribution.
    Eigen::HouseholderQR<Eigen::MatrixXd> qr(Z_est);
    const Eigen::MatrixXd R = qr.matrixQR().triangularView<Eigen::Upper>();

    double cum_ev = 0.0;
    for (Eigen::Index m = 0; m < M; ++m) {
        const double r_mm = R(m, m);
        cum_ev += (r_mm * r_mm) / static_cast<double>(n - 1);
    }

    // ----------------------------------------------------------------------
    // 2a. PEV, normalized by the TOTAL variance of X
    // ----------------------------------------------------------------------
    // The share of everything present in the data that the M components
    // reproduce. Bounded by 1 by construction: no set of M components can
    // explain more variance than X contains. This is the "how much of the data
    // did we capture" reading.
    const double total_var = X.squaredNorm() / static_cast<double>(n - 1);
    const double pev_total = (total_var > 0.0) ? (cum_ev / total_var) : 0.0;

    // ----------------------------------------------------------------------
    // 2b. PEV, normalized by the data's SIGNAL + MIXED EV (Definition 1)
    // ----------------------------------------------------------------------
    // Definition 1 of the reference paper, in the active-column convention of
    // the legacy R reference
    // (TRex_Simulations/TRex Legacy CRAN Simulations/trex_spca/
    //  demo_trex_spca_03_fig3.R, denominator mode "active"):
    //
    //   denominator = the Signal + Mixed EV OF THE DATA — the total variance
    //   carried by the true active variables (union of the factor supports):
    //
    //       denom = || X_A ||_F^2 / (n - 1),
    //
    //   where X_A keeps only the columns of X whose variable is active in ANY
    //   true factor. It is FIXED per dataset and SHARED by all methods.
    //
    //   numerator = the cumulative ADJUSTED EV (QR diagonal) computed above.
    //
    // A method that additionally captures null-variable variance (ordinary
    // PCA above all) pushes its numerator past the denominator, so this ratio
    // is deliberately uncapped — exceeding 100 % is the paper's marker of a
    // method claiming variance it should not. FDR-controlled T-Rex variants
    // plateau below 100 % (about 80-83 % ActiveSet, 86-96 % Thresholded). The
    // gap is not just noise: OrdPCA's Sig + Mix part reaches about 108 %.
    //
    // NOTE the denominator is NOT the method's own Signal + Mixed EV. The
    // self-normalized reading of Definition 1 (each method divided by its own
    // EV minus Null EV) is structurally >= 100 % for null-capturing methods
    // and CANNOT reproduce the rising curves of the published Fig. 3 — that
    // reading was this suite's original implementation and the reason the
    // first Fig.-3 attempt failed. See the R reference script for a full
    // comparison of the candidate denominator conventions.
    //
    // Membership of A is per VARIABLE (row of V_true): a variable is active
    // when it belongs to the true support of at least one factor.
    const double dn = static_cast<double>(n - 1);

    std::vector<bool> row_active(static_cast<std::size_t>(p), false);
    for (Eigen::Index j = 0; j < p; ++j)
        for (Eigen::Index m = 0; m < M_true; ++m)
            if (V_true(j, m) != 0.0) {
                row_active[static_cast<std::size_t>(j)] = true;
                break;
            }

    double sig_mixed_ev = 0.0;   // Signal + Mixed EV of the data
    for (Eigen::Index j = 0; j < p; ++j)
        if (row_active[static_cast<std::size_t>(j)])
            sig_mixed_ev += X.col(j).squaredNorm();
    sig_mixed_ev /= dn;

    const double pev_sig = (sig_mixed_ev > 0.0) ? (cum_ev / sig_mixed_ev) : 0.0;

    // ----------------------------------------------------------------------
    // 2c. The method's own Signal + Mixed share — the "Sig + Mix" reference
    // ----------------------------------------------------------------------
    // Split the method's OWN loading matrix by variable into active rows and
    // null rows, V_hat = V_A + V_AC, and remove the Null EV from its raw
    // (unadjusted) EV:
    //
    //     sig+mix part = ( ||X V_hat||_F^2 - ||X V_AC||_F^2 ) / (n - 1),
    //
    // reported as a share of the same data-level denominator. For ordinary
    // PCA this is the paper's "Ordinary PCA (Sig + Mix)" curve: it levels off
    // near 108 % once all components are extracted, and its distance to the
    // plain PEVsig curve is exactly the Null EV ordinary PCA piles up.
    Eigen::MatrixXd V_null = V_est;
    for (Eigen::Index j = 0; j < p; ++j)
        if (row_active[static_cast<std::size_t>(j)])
            V_null.row(j).setZero();

    const double raw_ev  = (X * V_est).squaredNorm() / dn;
    const double null_ev = (X * V_null).squaredNorm() / dn;
    const double pev_sigmix = (sig_mixed_ev > 0.0)
                            ? (raw_ev - null_ev) / sig_mixed_ev : 0.0;

    // Return the isolated PC1 recovery metrics and the three PEV readings
    return {pc1_tpr, pc1_fdr, pev_total, pev_sig, pev_sigmix};
}


// ==============================================================================
// Preprocessing
// ==============================================================================

/**
 * @brief Center columns of X in place (mean subtraction only, NO scaling).
 *
 * @details Covariance-PCA footing, shared by every method (matching the legacy
 *          R reference pipeline and TRexSPCA's own internal convention). The
 *          column scales must NOT be normalized: in the sparse factor model
 *          the factor amplitude signal lives in the column variances, and
 *          z-scoring the columns (correlation PCA) destroys it — measured
 *          effect at -10 dB: T-Rex SPCA FDR degrades from ~0.14 to ~0.52 and
 *          OraclePCA TPR from ~1.0 to ~0.87 (see
 *          validation_trex_spca_06_handrolled_comparison).
 *
 * @param X Observation matrix (n x p), modified in place.
 */
inline void center_columns(Eigen::MatrixXd& X)
{
    X.rowwise() -= X.colwise().mean();
}


// ==============================================================================
// MC loop helpers
// ==============================================================================

namespace detail {

/** @brief Compute mean/sd from per-trial vectors and print a done line. */
inline SPCAGridPointResult aggregate_mc(
    std::size_t                 num_MC,
    const std::string&          label,
    const std::vector<double>&  tpr_vec,
    const std::vector<double>&  fdr_vec,
    const std::vector<double>&  pev_vec,
    const std::vector<double>&  pev_sig_vec,
    const std::vector<double>&  pev_sigmix_vec)
{
    const int    iMC = static_cast<int>(num_MC);
    const double dMC = static_cast<double>(num_MC);

    double avg_tpr = 0.0, avg_fdr = 0.0, avg_pev = 0.0, avg_pev_sig = 0.0,
           avg_pev_sigmix = 0.0;
    for (int mc = 0; mc < iMC; ++mc) {
        avg_tpr += tpr_vec[mc];
        avg_fdr += fdr_vec[mc];
        avg_pev += pev_vec[mc];
        avg_pev_sig += pev_sig_vec[mc];
        avg_pev_sigmix += pev_sigmix_vec[mc];
    }
    avg_tpr /= dMC;  avg_fdr /= dMC;  avg_pev /= dMC;  avg_pev_sig /= dMC;
    avg_pev_sigmix /= dMC;

    double sd_tpr = 0.0, sd_fdr = 0.0, sd_pev = 0.0;
    if (num_MC > 1) {
        double ss_tpr = 0.0, ss_fdr = 0.0, ss_pev = 0.0;
        for (int mc = 0; mc < iMC; ++mc) {
            ss_tpr += (tpr_vec[mc] - avg_tpr) * (tpr_vec[mc] - avg_tpr);
            ss_fdr += (fdr_vec[mc] - avg_fdr) * (fdr_vec[mc] - avg_fdr);
            ss_pev += (pev_vec[mc] - avg_pev) * (pev_vec[mc] - avg_pev);
        }
        sd_tpr = std::sqrt(ss_tpr / (dMC - 1.0));
        sd_fdr = std::sqrt(ss_fdr / (dMC - 1.0));
        sd_pev = std::sqrt(ss_pev / (dMC - 1.0));
    }

    std::cout << "  " << label
              << " \u2014 done. TPR=" << std::fixed << std::setprecision(3) << avg_tpr
              << "  FDR=" << avg_fdr << "\n\n" << std::flush;

    return {avg_tpr, avg_fdr, avg_pev, avg_pev_sig, avg_pev_sigmix,
            sd_tpr, sd_fdr, sd_pev};
}

/** @brief Cell of one SPCA grid point. */
inline trex_cell_cache::Cell to_cell(const SPCAGridPointResult& r) {
    return {{"avg_tpr", r.avg_tpr}, {"avg_fdr", r.avg_fdr}, {"avg_pev", r.avg_pev},
            {"avg_pev_sig", r.avg_pev_sig}, {"avg_pev_sigmix", r.avg_pev_sigmix},
            {"sd_tpr", r.sd_tpr}, {"sd_fdr", r.sd_fdr}, {"sd_pev", r.sd_pev}};
}

inline SPCAGridPointResult from_cell(const trex_cell_cache::Cell& c) {
    using trex_cell_cache::get;
    return {get(c, "avg_tpr"), get(c, "avg_fdr"), get(c, "avg_pev"), get(c, "avg_pev_sig"),
            get(c, "avg_pev_sigmix"), get(c, "sd_tpr"), get(c, "sd_fdr"), get(c, "sd_pev")};
}

} // namespace detail


// ==============================================================================
// Parallel MC inner loops
// ==============================================================================

/** @brief Uncached body of run_mc_trials_trex_spca(). */
inline SPCAGridPointResult run_mc_trials_trex_spca_impl(
    std::size_t           num_MC,
    const std::string&    progress_label,
    const SPCADGPFactory& make_data,
    double                tFDR,
    SPCAMode              mode,
    unsigned              base_seed_offset,
    double                lambda_2    = -1.0,
    ENSolverType          en_solver   = ENSolverType::TENET,
    ScalingMode           scaling_mode = ScalingMode::L2,
    Eigen::Index          num_components = 0
) {

    const int iMC = static_cast<int>(num_MC);
    std::vector<double> tpr_vec(num_MC, 0.0);
    std::vector<double> fdr_vec(num_MC, 0.0);
    std::vector<double> pev_vec(num_MC, 0.0);
    std::vector<double> pev_sig_vec(num_MC, 0.0);
    std::vector<double> pev_sigmix_vec(num_MC, 0.0);

    std::cout << "  " << progress_label
              << " \u2014 Running " << num_MC << " MC trials ...\n" << std::flush;

    #pragma omp parallel for schedule(dynamic)
    for (int mc = 0; mc < iMC; ++mc) {
        // Deterministic per-trial data seed (reproducible from base_seed_offset).
        // Disjoint 1000-wide bands keep the DGP streams of different trials apart.
        const unsigned seed = base_seed_offset + static_cast<unsigned>(mc) * 1000u;

        auto dat = make_data(seed);

        // Pipeline preprocessing: center X once (covariance-PCA footing, legacy
        // convention). All methods and metrics share this same centered X.
        center_columns(dat.X);

        TRexSPCAControlParameter ctrl;
        ctrl.mode                    = mode;
        ctrl.en_solver               = en_solver;
        // TRexSPCAControlParameter has no top-level trex_ctrl of its own --
        // the base control parameters live under gvs_ctrl.trex_ctrl (GVS
        // sub-selector), avoiding a duplicate/out-of-sync copy.
        ctrl.gvs_ctrl.trex_ctrl.scaling_mode = scaling_mode;
        ctrl.gvs_ctrl.lambda2_method = LambdaSelectionMethod::CV_1SE_CCD;
        ctrl.gvs_ctrl.lambda_2       = lambda_2;   // < 0: auto-compute via CV; 0: no ridge; > 0: fixed

        // 0 means "extract as many components as there are true factors"; a
        // positive value decouples the two (Fig. 3(a) sweeps the PC count).
        const Eigen::Index n_comp = (num_components > 0) ? num_components
                                                         : dat.V.cols();

        Eigen::Map<Eigen::MatrixXd> X_map(dat.X.data(), dat.X.rows(), dat.X.cols());
        // Selector seed -1: hardware entropy per trial, so the computer-generated
        // dummies are independent across trials (required for valid MC FDR
        // estimates). The data DGP above is what the deterministic seed controls.
        TRexSPCA       spca(X_map, n_comp, tFDR, ctrl, -1);
        TRexSPCAResult res = spca.select();

        const auto m   = evaluate_spca(dat.X, res.V, res.Z, dat.V);
        tpr_vec[mc] = m.tpr;
        fdr_vec[mc] = m.fdr;
        pev_vec[mc] = m.pev;
        pev_sig_vec[mc] = m.pev_sig;
        pev_sigmix_vec[mc] = m.pev_sigmix;
    }

    return detail::aggregate_mc(num_MC, progress_label, tpr_vec, fdr_vec,
                                pev_vec, pev_sig_vec, pev_sigmix_vec);
}

/**
 * @brief Run num_MC parallel T-Rex SPCA trials (resumable grid cell).
 *
 * @details Data flow per trial:
 *          - make_data() returns raw FactorModelData (X = Z V^T + E, uncentered).
 *          - The pipeline centers X once (covariance-PCA convention); this same
 *            centered X is shared by the selector and by evaluate_spca().
 *          - TRexSPCA re-centers internally (a no-op on already-centered X) and
 *            restores X before returning, so dat.X stays centered for evaluation.
 *
 * @param num_MC            Number of Monte Carlo trials.
 * @param progress_label    Label printed in start/done lines.
 * @param make_data         Factory: seed → FactorModelData (called per trial).
 *                          Must capture all parameters by value for OMP safety.
 * @param tFDR              Target FDR for T-Rex SPCA.
 * @param mode              Loading assembly mode (ActiveSet or Thresholded).
 * @param base_seed_offset  Base seed; trial mc uses base_seed_offset + mc * 1000.
 * @param lambda_2          Ridge penalty (LARS units); < 0 = auto-compute via CV
 *                          (default), 0 = no ridge (pure T-LASSO), > 0 = fixed.
 * @param en_solver         EN solver variant (TENET or TENET_AUG; default TENET_AUG).
 * @param scaling_mode      Column scaling for X/dummies inside the per-PC
 *                          T-Rex selector (L2 or ZSCORE; default L2).
 *
 * @return SPCAGridPointResult with avg/sd TPR, FDR and avg PEV.
 */
inline SPCAGridPointResult run_mc_trials_trex_spca(
    std::size_t           num_MC,
    const std::string&    progress_label,
    const SPCADGPFactory& make_data,
    double                tFDR,
    SPCAMode              mode,
    unsigned              base_seed_offset,
    double                lambda_2    = -1.0,
    ENSolverType          en_solver   = ENSolverType::TENET,
    ScalingMode           scaling_mode = ScalingMode::L2,
    Eigen::Index          num_components = 0
) {
    std::ostringstream key;
    key << "trex_spca|" << progress_label << "|mode=" << static_cast<int>(mode)
        << "|en=" << static_cast<int>(en_solver) << "|scale=" << static_cast<int>(scaling_mode)
        << "|l2=" << lambda_2 << "|comp=" << num_components << "|seed=" << base_seed_offset
        << "|mc=" << num_MC << "|tfdr=" << tFDR;
    return detail::from_cell(TREX_CELL_CACHED(key.str(), [&] {
        return detail::to_cell(run_mc_trials_trex_spca_impl(num_MC, progress_label, make_data, tFDR, mode,
                                                            base_seed_offset, lambda_2, en_solver,
                                                            scaling_mode, num_components));
    }));
}


/** @brief Uncached body of run_mc_trials_pca(). */
inline SPCAGridPointResult run_mc_trials_pca_impl(
    std::size_t           num_MC,
    const std::string&    progress_label,
    const SPCADGPFactory& make_data,
    unsigned              base_seed_offset,
    Eigen::Index          num_components = 0)
{
    const int iMC = static_cast<int>(num_MC);
    std::vector<double> tpr_vec(num_MC, 0.0);
    std::vector<double> fdr_vec(num_MC, 0.0);
    std::vector<double> pev_vec(num_MC, 0.0);
    std::vector<double> pev_sig_vec(num_MC, 0.0);
    std::vector<double> pev_sigmix_vec(num_MC, 0.0);

    std::cout << "  " << progress_label
              << " — Running " << num_MC << " MC trials ...\n" << std::flush;

    #pragma omp parallel for schedule(dynamic)
    for (int mc = 0; mc < iMC; ++mc) {
        const unsigned seed = base_seed_offset + static_cast<unsigned>(mc) * 1000u;
        auto dat = make_data(seed);

        // Pipeline preprocessing: center X once (covariance PCA), shared by all methods.
        center_columns(dat.X);

        const Eigen::Index n_comp = (num_components > 0) ? num_components
                                                         : dat.V.cols();

        // X is already centered → center=false, normalize=false; covariance PCA.
        pca_ns::PCA       pca(dat.X, n_comp, /*center=*/false, /*normalize=*/false);
        pca_ns::PCAResult res = pca.fit();

        const auto m = evaluate_spca(dat.X, res.V, res.Z, dat.V);
        tpr_vec[mc] = m.tpr;
        fdr_vec[mc] = m.fdr;
        pev_vec[mc] = m.pev;
        pev_sig_vec[mc] = m.pev_sig;
        pev_sigmix_vec[mc] = m.pev_sigmix;
    }

    return detail::aggregate_mc(num_MC, progress_label, tpr_vec, fdr_vec,
                                pev_vec, pev_sig_vec, pev_sigmix_vec);
}

/**
 * @brief Run num_MC parallel ordinary PCA trials (resumable grid cell).
 *
 * @details Ordinary PCA selects all p variables per component (no sparsity),
 *          so FDR ≈ (p - p1) / p and TPR = 1 regardless of SNR.
 *          The metric of interest is the cumulative PEV.
 *
 *          Data flow: make_data() returns raw data; the pipeline centers X once
 *          (covariance-PCA convention). PCA is then constructed with center=false
 *          (X is already centered), and evaluate_spca() shares the same centered X.
 *
 * @param num_MC            Number of Monte Carlo trials.
 * @param progress_label    Label printed in start/done lines.
 * @param make_data         Factory: seed → FactorModelData.
 * @param base_seed_offset  Base seed; trial mc uses base_seed_offset + mc * 1000.
 *
 * @return SPCAGridPointResult with avg/sd TPR, FDR and avg PEV.
 */
inline SPCAGridPointResult run_mc_trials_pca(
    std::size_t           num_MC,
    const std::string&    progress_label,
    const SPCADGPFactory& make_data,
    unsigned              base_seed_offset,
    Eigen::Index          num_components = 0)
{
    std::ostringstream key;
    key << "pca|" << progress_label << "|comp=" << num_components << "|seed=" << base_seed_offset
        << "|mc=" << num_MC;
    return detail::from_cell(TREX_CELL_CACHED(key.str(), [&] {
        return detail::to_cell(run_mc_trials_pca_impl(num_MC, progress_label, make_data,
                                                      base_seed_offset, num_components));
    }));
}


/** @brief Uncached body of run_mc_trials_oracle_pca(). */
inline SPCAGridPointResult run_mc_trials_oracle_pca_impl(
    std::size_t           num_MC,
    const std::string&    progress_label,
    const SPCADGPFactory& make_data,
    Eigen::Index          p1,
    unsigned              base_seed_offset,
    Eigen::Index          num_components = 0)
{
    const int iMC = static_cast<int>(num_MC);
    std::vector<double> tpr_vec(num_MC, 0.0);
    std::vector<double> fdr_vec(num_MC, 0.0);
    std::vector<double> pev_vec(num_MC, 0.0);
    std::vector<double> pev_sig_vec(num_MC, 0.0);
    std::vector<double> pev_sigmix_vec(num_MC, 0.0);

    std::cout << "  " << progress_label
              << " — Running " << num_MC << " MC trials ...\n" << std::flush;

    #pragma omp parallel for schedule(dynamic)
    for (int mc = 0; mc < iMC; ++mc) {
        const unsigned seed = base_seed_offset + static_cast<unsigned>(mc) * 1000u;
        auto dat = make_data(seed);

        // Pipeline preprocessing: center X once (covariance PCA), shared by all methods.
        center_columns(dat.X);

        const Eigen::Index p = dat.X.cols();
        const Eigen::Index M = (num_components > 0) ? num_components
                                                    : dat.V.cols();

        // X is already centered → center=false, normalize=false; covariance PCA.
        pca_ns::PCA       pca(dat.X, M, /*center=*/false, /*normalize=*/false);
        pca_ns::PCAResult ord = pca.fit();

        // Threshold: keep top p1 entries per loading vector, then normalize
        Eigen::MatrixXd V_oracle = Eigen::MatrixXd::Zero(p, M);
        for (Eigen::Index comp = 0; comp < M; ++comp) {
            const Eigen::VectorXd v_m = ord.V.col(comp);

            std::vector<double> abs_vals(static_cast<std::size_t>(p));
            for (Eigen::Index i = 0; i < p; ++i)
                abs_vals[static_cast<std::size_t>(i)] = std::abs(v_m(i));

            std::nth_element(abs_vals.begin(),
                             abs_vals.begin() + (p - p1),
                             abs_vals.end());
            const double thresh = abs_vals[static_cast<std::size_t>(p - p1)];

            for (Eigen::Index i = 0; i < p; ++i)
                if (std::abs(v_m(i)) >= thresh) V_oracle(i, comp) = v_m(i);
            V_oracle.col(comp).normalize();
        }
        const Eigen::MatrixXd Z_oracle = dat.X * V_oracle;

        const auto m   = evaluate_spca(dat.X, V_oracle, Z_oracle, dat.V);
        tpr_vec[mc] = m.tpr;
        fdr_vec[mc] = m.fdr;
        pev_vec[mc] = m.pev;
        pev_sig_vec[mc] = m.pev_sig;
        pev_sigmix_vec[mc] = m.pev_sigmix;
    }

    return detail::aggregate_mc(num_MC, progress_label, tpr_vec, fdr_vec,
                                pev_vec, pev_sig_vec, pev_sigmix_vec);
}

/**
 * @brief Run num_MC parallel oracle-thresholded PCA trials (resumable grid cell).
 *
 * @details Oracle PCA uses the ordinary PCA loadings but retains exactly the
 *          top p1 entries (by absolute value) per component — i.e., it knows
 *          the true support cardinality. This provides an upper bound on what
 *          any data-driven thresholding method can achieve.
 *
 *          Data flow: make_data() returns raw data; the pipeline centers X once
 *          (covariance-PCA convention). PCA is constructed with center=false
 *          (X is already centered). The oracle score matrix Z_oracle = X * V_oracle
 *          and evaluate_spca() share the same centered X.
 *
 * @param num_MC            Number of Monte Carlo trials.
 * @param progress_label    Label printed in start/done lines.
 * @param make_data         Factory: seed → FactorModelData.
 * @param p1                Oracle threshold: true active loadings per factor.
 * @param base_seed_offset  Base seed; trial mc uses base_seed_offset + mc * 1000.
 *
 * @return SPCAGridPointResult with avg/sd TPR, FDR and avg PEV.
 */
inline SPCAGridPointResult run_mc_trials_oracle_pca(
    std::size_t           num_MC,
    const std::string&    progress_label,
    const SPCADGPFactory& make_data,
    Eigen::Index          p1,
    unsigned              base_seed_offset,
    Eigen::Index          num_components = 0)
{
    std::ostringstream key;
    key << "oracle_pca|" << progress_label << "|p1=" << p1 << "|comp=" << num_components
        << "|seed=" << base_seed_offset << "|mc=" << num_MC;
    return detail::from_cell(TREX_CELL_CACHED(key.str(), [&] {
        return detail::to_cell(run_mc_trials_oracle_pca_impl(num_MC, progress_label, make_data, p1,
                                                             base_seed_offset, num_components));
    }));
}


// ==============================================================================
// Output
// ==============================================================================

/**
 * @brief Print MC-averaged T-Rex SPCA results as an aligned table (console + .txt)
 *        and write a tidy long-format CSV.
 *
 * @param num_MC        Number of Monte Carlo trials.
 * @param file_stem     Output file base name (no folder, no extension).
 * @param snr_values    SNR grid (dB) used during the simulation.
 * @param method_names  Ordered list of method names (keys into result maps).
 * @param fdr_map       Averaged FDR per method, indexed by SNR grid position.
 * @param tpr_map       Averaged TPR per method, indexed by SNR grid position.
 * @param pev_map       Averaged cumulative PEV per method (total-variance
 *                      normalization), indexed by SNR grid position.
 * @param pev_sig_map   Averaged Definition-1 PEV per method (data signal +
 *                      mixed EV normalization), indexed by grid position.
 * @param pev_sigmix_map Averaged Sig+Mix-part PEV per method (the method's own
 *                      signal + mixed EV share of the same denominator; the
 *                      "Ordinary PCA (Sig + Mix)" reference of the paper).
 * @param include_support_metrics  When false, the displayed table carries only
 *                      the cumulative-PEV rows. FDR and TPR stay in the
 *                      tidy CSV either way — they are the data record — but
 *                      sweeps whose story is purely the explained variance
 *                      (demo 02 parts 2–4) do not display them.
 */
inline void save_and_print_spca_sweep_results(
    std::size_t                                    num_MC,
    const std::string&                             file_stem,
    const std::string&                             sweep_col,
    const std::string&                             sweep_header,
    int                                            sweep_precision,
    const std::vector<double>&                     snr_values,
    const std::vector<std::string>&                method_names,
    const std::map<std::string, Eigen::VectorXd>&  fdr_map,
    const std::map<std::string, Eigen::VectorXd>&  tpr_map,
    const std::map<std::string, Eigen::VectorXd>&  pev_map,
    const std::map<std::string, Eigen::VectorXd>&  pev_sig_map,
    const std::map<std::string, Eigen::VectorXd>&  pev_sigmix_map,
    bool                                           include_support_metrics = true)
{
    const std::string folder = DEMO_OUTPUT_DIR;
    std::filesystem::create_directories(folder);

    std::ofstream out_file(folder + file_stem + ".txt");
    auto print_dual = [&](const std::string& text) {
        std::cout << text;
        if (out_file.is_open()) out_file << text;
    };

    // Header
    {
        std::ostringstream ss;
        ss << "\n"
           << "======================================================================\n"
           << "=== T-Rex SPCA Results (averaged over " << num_MC << " MC trials) ===\n"
           << "======================================================================\n\n";
        print_dual(ss.str());
    }

    // Table dimensions
    //    The method name gets its own line; metric rows are indented beneath
    //    it, so the value columns stay aligned no matter how long a name is.
    const int indent_w = 2;    // leading indent of a metric row
    const int metric_w = 23;   // left-aligned metric label
    const int col_w    = 10;   // right-aligned values
    const std::size_t sep_w =
        static_cast<std::size_t>(indent_w + metric_w)
        + static_cast<std::size_t>(col_w) * snr_values.size();

    // Column header + dashed separator
    {
        std::ostringstream hdr;
        hdr << std::left << std::string(static_cast<std::size_t>(indent_w), ' ')
            << std::setw(metric_w) << sweep_header;
        for (double snr : snr_values)
            hdr << std::right << std::fixed << std::setprecision(sweep_precision)
                << std::setw(col_w) << snr;
        hdr << "\n" << std::string(sep_w, '-') << "\n";
        print_dual(hdr.str());
    }

    // Data rows
    auto print_row = [&](const std::string& metric,
                         const Eigen::VectorXd& data) {
        std::ostringstream row;
        row << std::left << std::string(static_cast<std::size_t>(indent_w), ' ')
            << std::setw(metric_w) << metric;
        for (Eigen::Index i = 0; i < static_cast<Eigen::Index>(snr_values.size()); ++i)
            row << std::right << std::fixed << std::setprecision(4)
                << std::setw(col_w) << data(i);
        row << "\n";
        print_dual(row.str());
    };

    for (const auto& name : method_names) {
        print_dual("\n" + name + "\n");          // method name on its own line
        if (include_support_metrics) {
            print_row("FDR", fdr_map.at(name));
            print_row("TPR", tpr_map.at(name));
        }
        print_row("PEV (total var)", pev_map.at(name));
        print_row("PEV (Definition 1)", pev_sig_map.at(name));
        print_row("PEV (sig+mix part)", pev_sigmix_map.at(name));
    }
    print_dual("\n");

    // Tidy long-format CSV (method, metric, snr_db, value)
    std::ofstream csv(folder + file_stem + ".csv");
    if (csv.is_open()) {
        csv << "method,metric," << sweep_col << ",value\n" << std::fixed << std::setprecision(6);
        for (const auto& name : method_names) {
            for (std::size_t i = 0; i < snr_values.size(); ++i) {
                const auto ei = static_cast<Eigen::Index>(i);
                std::ostringstream snr_ss;
                snr_ss << std::fixed << std::setprecision(6) << snr_values[i];
                const std::string s = snr_ss.str();
                csv << name << ",FDR," << s << "," << fdr_map.at(name)(ei) << "\n";
                csv << name << ",TPR," << s << "," << tpr_map.at(name)(ei) << "\n";
                csv << name << ",PEV," << s << "," << pev_map.at(name)(ei) << "\n";
                csv << name << ",PEVsig," << s << "," << pev_sig_map.at(name)(ei) << "\n";
                csv << name << ",PEVsigmix," << s << "," << pev_sigmix_map.at(name)(ei) << "\n";
            }
        }
        std::cout << "[Info] CSV results saved to:             "
                  << folder + file_stem + ".csv\n";
    } else {
        std::cout << "[Warning] Could not open CSV file: " << folder + file_stem + ".csv\n";
    }

    if (out_file.is_open()) {
        std::cout << "[Info] Results successfully saved to: " << folder + file_stem + ".txt\n\n";
        out_file.close();
    } else {
        std::cout << "[Warning] Could not open output file: " << folder + file_stem + ".txt\n\n";
    }
}
/**
 * @brief SNR-sweep convenience wrapper around save_and_print_spca_sweep_results.
 *
 * @details Fixes the sweep axis to the decibel SNR grid this suite reports on
 *          by default: CSV column "snr_db", table header "SNR(dB)", one decimal.
 */
inline void save_and_print_spca_mc_results(
    std::size_t                                    num_MC,
    const std::string&                             file_stem,
    const std::vector<double>&                     snr_values,
    const std::vector<std::string>&                method_names,
    const std::map<std::string, Eigen::VectorXd>&  fdr_map,
    const std::map<std::string, Eigen::VectorXd>&  tpr_map,
    const std::map<std::string, Eigen::VectorXd>&  pev_map,
    const std::map<std::string, Eigen::VectorXd>&  pev_sig_map,
    const std::map<std::string, Eigen::VectorXd>&  pev_sigmix_map)
{
    save_and_print_spca_sweep_results(num_MC, file_stem, "snr_db", "SNR(dB)", 1,
                                      snr_values, method_names,
                                      fdr_map, tpr_map, pev_map, pev_sig_map,
                                      pev_sigmix_map);
}

// ==============================================================================
} // namespace spca_sim
// ==============================================================================
#endif /* End of DEMOS_TREX_SELECTOR_METHODS_TREX_SPCA_SIM_UTILS_HPP */
