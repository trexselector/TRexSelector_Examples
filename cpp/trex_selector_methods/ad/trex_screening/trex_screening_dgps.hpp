// ==============================================================================
// trex_screening_dgps.hpp
// ==============================================================================
/**
 * @file trex_screening_dgps.hpp
 *
 * @brief Data-generating processes for the Screen-TRex demonstration suite.
 *
 * @details
 *  All generators return a ScrDGPData bundle (X, y, true_support) and are
 *  self-contained: given a seed they produce one complete Monte Carlo trial,
 *  so they can be called independently from parallel OpenMP threads.
 *
 *   - make_iid_dgp()          — i.i.d. Gaussian design, random sparse support
 *                               (Demos 01, 02, 06)
 *   - make_ar1_dgp()          — AR(1) column correlation (Demo 03)
 *   - make_equi_dgp()         — equicorrelated columns via a shared latent
 *                               factor (Demo 03)
 *   - make_block_equi_dgp()   — contiguous equicorrelated blocks (Demo 03)
 *
 *  Helpers:
 *   - draw_random_support()   — random sparse support without replacement
 *   - make_beta()             — evenly spaced sparse coefficient vector
 *   - make_response()         — y = X*beta + eps, calibrated to a target SNR
 *   - support_from_beta()     — nonzero indices of a coefficient vector
 *
 *  The correlated designs (AR1 / equi / block-equi) standardise every column
 *  to zero mean and unit variance after generation, which leaves the column
 *  correlations unchanged.
 *
 * @note Demo-internal header — not part of the TRexSelector library.
 */
// ==============================================================================

#ifndef TREX_DEMOS_TREX_SCREENING_DGPS_HPP
#define TREX_DEMOS_TREX_SCREENING_DGPS_HPP

// std includes
#include <cmath>
#include <cstddef>
#include <random>
#include <unordered_set>
#include <vector>

// Eigen includes
#include <Eigen/Dense>

// TRex includes
#include <utils/datageneration/utils_datagen.hpp>

// ==============================================================================

namespace scr_demo {

namespace datagen = trex::utils::datageneration::datagen;


// ==============================================================================
// Data container
// ==============================================================================

/** @brief One Monte Carlo trial: design matrix, response, and ground truth. */
struct ScrDGPData {
    /** @brief Design matrix (n x p). */
    Eigen::MatrixXd X;

    /** @brief Response vector (n). */
    Eigen::VectorXd y;

    /** @brief Indices of the truly active variables. */
    std::vector<std::size_t> true_support;
};


// ==============================================================================
// Helpers
// ==============================================================================

/**
 * @brief Draw a random sparse support without replacement.
 *
 * @param p             Total number of predictors.
 * @param support_size  Number of active predictors to draw.
 * @param rng           Random number generator.
 *
 * @return Sorted-by-insertion vector of support_size distinct indices in [0, p).
 */
inline std::vector<std::size_t> draw_random_support(std::size_t p,
                                                    std::size_t support_size,
                                                    std::mt19937& rng)
{
    std::uniform_int_distribution<std::size_t> idx_dist(0, p - 1);
    std::unordered_set<std::size_t> support_set;
    while (support_set.size() < support_size) {
        support_set.insert(idx_dist(rng));
    }
    return {support_set.begin(), support_set.end()};
}


/**
 * @brief Generate an evenly spaced sparse beta vector.
 *
 * @param p         Total number of predictors.
 * @param p1        Number of active predictors (non-zero coefficients).
 * @param beta_val  Coefficient value for active predictors.
 *
 * @return Vector of length p with p1 non-zero entries of value beta_val.
 */
inline Eigen::VectorXd make_beta(Eigen::Index p, Eigen::Index p1, double beta_val)
{
    Eigen::VectorXd beta = Eigen::VectorXd::Zero(p);
    for (Eigen::Index i = 0; i < p1; ++i) {
        // evenly spaced: skip first and last boundary
        const Eigen::Index idx = static_cast<Eigen::Index>(
            std::round(static_cast<double>(i + 1) * static_cast<double>(p)
                       / static_cast<double>(p1 + 2)));
        beta(idx) = beta_val;
    }
    return beta;
}


/**
 * @brief Generate the response vector y = X * beta + eps for a target SNR.
 *
 * @param X     Design matrix.
 * @param beta  Coefficient vector.
 * @param snr   Signal-to-noise ratio.
 * @param rng   Random number generator.
 *
 * @return Centered response vector y.
 */
inline Eigen::VectorXd make_response(const Eigen::MatrixXd& X,
                                     const Eigen::VectorXd& beta,
                                     double snr,
                                     std::mt19937& rng)
{
    const Eigen::VectorXd signal = X * beta;
    const Eigen::Index n = X.rows();
    const double sigma =
        std::sqrt(signal.squaredNorm() / (static_cast<double>(n) * snr));

    std::normal_distribution<double> dist(0.0, sigma);
    Eigen::VectorXd y = signal;
    for (Eigen::Index i = 0; i < n; ++i) { y(i) += dist(rng); }

    y.array() -= y.mean();
    return y;
}


/**
 * @brief Get the support indices of a beta vector.
 *
 * @param beta  Coefficient vector.
 *
 * @return Indices of the non-zero entries.
 */
inline std::vector<std::size_t> support_from_beta(const Eigen::VectorXd& beta)
{
    std::vector<std::size_t> support;
    for (Eigen::Index j = 0; j < beta.size(); ++j) {
        if (std::abs(beta(j)) > 1e-12) {
            support.push_back(static_cast<std::size_t>(j));
        }
    }
    return support;
}


/** @brief Standardise every column of X to zero mean and unit variance. */
inline void standardise_columns(Eigen::MatrixXd& X)
{
    const Eigen::Index n = X.rows();
    for (Eigen::Index j = 0; j < X.cols(); ++j) {
        X.col(j).array() -= X.col(j).mean();
        const double sd =
            std::sqrt(X.col(j).squaredNorm() / static_cast<double>(n - 1));
        if (sd > 1e-12) { X.col(j) /= sd; }
    }
}


// ==============================================================================
// DGP: i.i.d. Gaussian design  (used by Demos 01, 02, 06)
// ==============================================================================

/**
 * @brief i.i.d. Gaussian design with a random sparse support.
 *
 * @details Wraps datagen::SyntheticData. The support-selection and coefficient
 *          RNGs are offset from the trial seed (by 500000 and 600000) so that
 *          they stay independent of the design/noise draws.
 *
 * @param n             Number of observations.
 * @param p             Number of predictors.
 * @param support_size  Number of active predictors.
 * @param snr           Target signal-to-noise ratio.
 * @param rnd_coef      If true, draw coefficients from N(0,1); else all 1.0.
 * @param seed          Trial seed.
 *
 * @return Populated ScrDGPData.
 */
inline ScrDGPData make_iid_dgp(std::size_t n,
                               std::size_t p,
                               std::size_t support_size,
                               double      snr,
                               bool        rnd_coef,
                               unsigned    seed)
{
    // Isolate support-selection RNG to offset 500000 from the trial seed
    std::mt19937 rng_sup(seed + 500000u);
    const std::vector<std::size_t> true_support =
        draw_random_support(p, support_size, rng_sup);

    // Isolate coefficient RNG to offset 600000 from the trial seed
    std::vector<double> true_coefs;
    true_coefs.reserve(support_size);
    if (rnd_coef) {
        std::mt19937 rng_coef(seed + 600000u);
        std::normal_distribution<double> coef_dist(0.0, 1.0);
        for (std::size_t i = 0; i < support_size; ++i) {
            true_coefs.push_back(coef_dist(rng_coef));
        }
    } else {
        true_coefs.assign(support_size, 1.0);
    }

    datagen::SyntheticData data(
        static_cast<Eigen::Index>(n),
        static_cast<Eigen::Index>(p),
        true_support, true_coefs, snr, static_cast<int>(seed));

    return {data.getX(), data.getY(), true_support};
}


// ==============================================================================
// DGP: AR(1) column correlation  (used by Demo 03)
// ==============================================================================

/**
 * @brief AR(1) column-correlation design.
 *
 * @details x_1 ~ N(0, I) and x_j = rho * x_{j-1} + sqrt(1 - rho^2) * w_j, so
 *          columns j and j' correlate as rho^|j - j'|.
 *
 * @param n         Number of observations.
 * @param p         Number of predictors.
 * @param p1        Number of active predictors.
 * @param rho       AR(1) correlation coefficient.
 * @param snr       Target signal-to-noise ratio.
 * @param beta_val  Coefficient value for active predictors.
 * @param seed      Random seed.
 *
 * @return Populated ScrDGPData.
 */
inline ScrDGPData make_ar1_dgp(Eigen::Index n,
                               Eigen::Index p,
                               Eigen::Index p1,
                               double       rho,
                               double       snr,
                               double       beta_val,
                               unsigned     seed)
{
    std::mt19937 rng(seed);
    std::normal_distribution<double> norm(0.0, 1.0);

    Eigen::MatrixXd X(n, p);
    const double innovation_sd = std::sqrt(1.0 - rho * rho);
    for (Eigen::Index i = 0; i < n; ++i) {
        X(i, 0) = norm(rng);
        for (Eigen::Index j = 1; j < p; ++j) {
            X(i, j) = rho * X(i, j - 1) + innovation_sd * norm(rng);
        }
    }
    standardise_columns(X);

    auto beta = make_beta(p, p1, beta_val);
    auto y    = make_response(X, beta, snr, rng);
    return {std::move(X), std::move(y), support_from_beta(beta)};
}


// ==============================================================================
// DGP: Equicorrelated columns  (used by Demo 03)
// ==============================================================================

/**
 * @brief Equicorrelated design via one shared latent factor.
 *
 * @details x_j = sqrt(rho) * z + sqrt(1 - rho) * w_j, so every pair of columns
 *          correlates as rho.
 *
 * @param n         Number of observations.
 * @param p         Number of predictors.
 * @param p1        Number of active predictors.
 * @param rho       Equicorrelation coefficient.
 * @param snr       Target signal-to-noise ratio.
 * @param beta_val  Coefficient value for active predictors.
 * @param seed      Random seed.
 *
 * @return Populated ScrDGPData.
 */
inline ScrDGPData make_equi_dgp(Eigen::Index n,
                                Eigen::Index p,
                                Eigen::Index p1,
                                double       rho,
                                double       snr,
                                double       beta_val,
                                unsigned     seed)
{
    std::mt19937 rng(seed);
    std::normal_distribution<double> norm(0.0, 1.0);

    const double load_f   = std::sqrt(rho);
    const double load_eta = std::sqrt(1.0 - rho);

    Eigen::MatrixXd X(n, p);
    for (Eigen::Index i = 0; i < n; ++i) {
        const double f_i = norm(rng);
        for (Eigen::Index j = 0; j < p; ++j) {
            X(i, j) = load_f * f_i + load_eta * norm(rng);
        }
    }
    standardise_columns(X);

    auto beta = make_beta(p, p1, beta_val);
    auto y    = make_response(X, beta, snr, rng);
    return {std::move(X), std::move(y), support_from_beta(beta)};
}


// ==============================================================================
// DGP: Block-equicorrelated columns  (used by Demo 03)
// ==============================================================================

/**
 * @brief Block-equicorrelated design: n_blocks contiguous equicorrelated blocks.
 *
 * @details Each block k carries its own latent factor z_k, so columns correlate
 *          as rho within a block and are independent across blocks. Block sizes
 *          are p / n_blocks, with the first (p % n_blocks) blocks one larger.
 *
 * @param n         Number of observations.
 * @param p         Number of predictors.
 * @param p1        Number of active predictors.
 * @param rho       Within-block equicorrelation coefficient.
 * @param n_blocks  Number of contiguous blocks.
 * @param snr       Target signal-to-noise ratio.
 * @param beta_val  Coefficient value for active predictors.
 * @param seed      Random seed.
 *
 * @return Populated ScrDGPData.
 */
inline ScrDGPData make_block_equi_dgp(Eigen::Index n,
                                      Eigen::Index p,
                                      Eigen::Index p1,
                                      double       rho,
                                      std::size_t  n_blocks,
                                      double       snr,
                                      double       beta_val,
                                      unsigned     seed)
{
    std::mt19937 rng(seed);
    std::normal_distribution<double> norm(0.0, 1.0);

    // Block sizes: base_size + 1 for the first (p % n_blocks) blocks
    const auto p_sz = static_cast<std::size_t>(p);
    const std::size_t base_size = p_sz / n_blocks;
    const std::size_t remainder = p_sz % n_blocks;

    const double load_f   = std::sqrt(rho);
    const double load_eta = std::sqrt(1.0 - rho);

    Eigen::MatrixXd X(n, p);
    std::size_t col_idx = 0;

    for (std::size_t k = 0; k < n_blocks; ++k) {
        const std::size_t bk = base_size + (k < remainder ? 1 : 0);

        // Shared factor for block k
        Eigen::VectorXd z_k(n);
        for (Eigen::Index i = 0; i < n; ++i) { z_k(i) = norm(rng); }

        // x_j = sqrt(rho) * z_k + sqrt(1 - rho) * w_j
        for (std::size_t j = 0; j < bk; ++j) {
            const auto col = static_cast<Eigen::Index>(col_idx + j);
            for (Eigen::Index i = 0; i < n; ++i) {
                X(i, col) = load_f * z_k(i) + load_eta * norm(rng);
            }
        }

        col_idx += bk;
    }
    standardise_columns(X);

    auto beta = make_beta(p, p1, beta_val);
    auto y    = make_response(X, beta, snr, rng);
    return {std::move(X), std::move(y), support_from_beta(beta)};
}


} // namespace scr_demo

// ==============================================================================
#endif /* End of TREX_DEMOS_TREX_SCREENING_DGPS_HPP */
// ==============================================================================
