// ==============================================================================
// demo_trex_05_mc_sim_dummy_distributions.cpp
// ==============================================================================
/**
 * @file demo_trex_05_mc_sim_dummy_distributions.cpp
 *
 * @brief Monte Carlo simulation comparing the dummy distributions of the
 *        T-Rex Selector across three base solvers (TLARS, TOMP, TAFS), with
 *        the STANDARD L-loop strategy held fixed.
 *
 * @details The T-Rex FDR calibration rests on exchangeability between real null
 *  predictors and the injected dummy variables. Since dummies are centered and
 *  L2-normalized before entering the solver, their scale is immaterial — what
 *  this demo probes is whether the SHAPE of the dummy distribution (tails,
 *  discreteness, skewness, sparsity, cross-column dependence) affects the
 *  achieved FDR / TPR and the calibrated dummy multiplier L / stopping time T.
 *
 *  Base solvers compared (one result file pair per solver):
 *    TLARS — equiangular LARS path; terminates on its own
 *            (stagnation stop AUTO-resolves to disabled).
 *    TOMP  — greedy orthogonal matching pursuit
 *            (stagnation stop AUTO-resolves to enabled).
 *    TAFS  — greedy adaptive forward selection, rho_afs = 0.3
 *            (stagnation stop AUTO-resolves to enabled).
 *
 *  Runs in two modes controlled by the `block_support` parameter:
 *    block_support = true  — contiguous block support {0, 1, ..., s-1}.
 *    block_support = false — scattered support redrawn from uniform per trial.
 *
 *  Dummy distributions compared (see utils_dummygen.hpp):
 *    Normal              — N(0, 1); the reference choice.
 *    Uniform             — U(-sqrt(3), sqrt(3)); bounded, light tails.
 *    Rademacher          — {-1, +1} equiprobable; discrete two-point.
 *    StudentT (df 3 / 5) — heavy tails (unit-variance scaled for df > 2).
 *    Laplace             — double exponential; heavier-than-normal tails.
 *    Gumbel              — extreme-value; skewed (centered at 0).
 *    Triangle            — bounded, symmetric; (-sqrt(6), 0, sqrt(6)).
 *    Logistic            — between normal and Laplace tails.
 *    Mammen              — asymmetric golden-ratio two-point distribution.
 *    SparseRademacher    — constrained sparse Rademacher, sparsity s = 0.1.
 *    UniformSphere       — uniform on the unit 5-sphere; consecutive
 *                          5-column groups are dependent (norm constraint),
 *                          a deliberate probe of the exchangeability axis.
 *
 *    High-dimensional  (n = 300, p = 1000, s = 10).
 *    SNR sweep: {0.1, 0.2, 0.5, 0.6, 1, 2, 5}  (7 values).
 *    Two support scenarios:
 *      random support  — redrawn per trial;
 *      block support   — contiguous block {0, 1, ..., s-1}.
 *    Reports averaged FDR / TPR / Avg L / Avg T per distribution x SNR.
 */
// ==============================================================================

// std includes
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <vector>

// Eigen includes
#include <Eigen/Dense>

// OpenMP compatibility layer
#include <utils/openmp/utils_openmp.hpp>

// T-Rex Selector includes
#include <trex_selector_methods/ad/trex_core/trex.hpp>
#include <utils/datageneration/utils_datagen.hpp>
#include <utils/eval_metrics/utils_eval_cdiagnostics.hpp>
#include <utils/eval_metrics/utils_eval_rates.hpp>

// Demo utilities
#include "trex_sim_utils.hpp"
using namespace trex_sim;


// ==============================================================================
// Namespace aliases
// ==============================================================================

namespace cdianostics = trex::utils::eval::cdiagnostics;
namespace datagen = trex::utils::datageneration::datagen;
namespace dummygen = trex::utils::datageneration::dummygen;

// T-Rex types
using trex::trex_selector_methods::ad::trex_core::TRexControlParameter;
using trex::trex_selector_methods::ad::trex_core::LLoopStrategy;
using trex::trex_selector_methods::ad::utils::solver_dispatch::SolverTypeForTRex;


// ==============================================================================
// Dummy distribution descriptor
// ==============================================================================

/**
 * @brief Descriptor for one dummy distribution variant.
 */
struct DummyDistributionInfo {
    /** @brief Row label used in output table and CSV. */
    std::string name;
    /** @brief Distribution configuration passed to TRexControlParameter. */
    dummygen::Distribution distribution;
};


/** @brief Returns the canonical list of dummy distribution variants to compare.
 *
 *  All distributions use their unit-variance default parameters where
 *  applicable; scale is immaterial anyway since dummies are L2-normalized.
 *  StudentT is compared at two tail weights (df = 3 heavy, df = 5 moderate).
 *  UniformSphere uses dim = 5: the library requests dummies in multiples of
 *  p = 1000, so the sphere dimension must divide p (the default dim = 3
 *  would throw).
 */
static std::vector<DummyDistributionInfo> make_dummy_distributions() {
    return {
        {"Normal",              dummygen::Distribution::Normal()},
        {"Uniform",             dummygen::Distribution::Uniform()},
        {"Rademacher",          dummygen::Distribution::Rademacher()},
        {"StudentT_df3",        dummygen::Distribution::StudentT(3.0)},
        {"StudentT_df5",        dummygen::Distribution::StudentT(5.0)},
        {"Laplace",             dummygen::Distribution::Laplace()},
        {"Gumbel",              dummygen::Distribution::Gumbel()},
        {"Triangle",            dummygen::Distribution::Triangle()},
        {"Logistic",            dummygen::Distribution::Logistic()},
        {"Mammen",              dummygen::Distribution::Mammen()},
        {"SparseRad_s0.1",     dummygen::Distribution::ConstrainedSparseRademacher(0.1)},
        {"UnifSphere_d5",      dummygen::Distribution::UniformSphere(5)}
    };
}


// ==============================================================================
// Monte Carlo Simulation — dummy distribution comparison (TLARS / TOMP / TAFS)
// ==============================================================================

void demo_TRexSelector_dummy_distributions(std::size_t num_MC, bool rnd_coef,
                                           bool block_support) {

    std::cout.setf(std::ios::unitbuf);

    cdianostics::print_section_header(
        "Demo: T-Rex Selector MC — Dummy Distribution Comparison  |  TLARS / TOMP / TAFS");

    // -----------------------------------------------------------------------
    // Scenario parameters (always high-dimensional)
    // -----------------------------------------------------------------------
    const Eigen::Index n = 300;
    const Eigen::Index p = 1000;
    const std::size_t  s = 10;
    const double       tFDR = 0.1;

    std::cout << "High-dimensional (p > n)\n";
    std::cout << "n = " << n << ",  p = " << p
              << ",  s = " << s << ",  num_MC = " << num_MC
              << "  [" << (block_support ? "block" : "random") << " support]\n\n";

    // SNR values (7 values)
    std::vector<double> snr_values = {0.1, 0.2, 0.5, 0.6, 1, 2, 5};

    // -----------------------------------------------------------------------
    // Base solvers and dummy distributions to sweep
    // -----------------------------------------------------------------------
    // TOMP, TAFS with rho_afs = 0.3 with T-loop stagnation stop
    // TLARS (stagnation stop AUTO disabled)
    const std::vector<DemoSolverInfo> solvers = {
        {SolverTypeForTRex::TLARS, "TLARS"},
        {SolverTypeForTRex::TOMP,  "TOMP"},
        {SolverTypeForTRex::TAFS,  "TAFS", 0.0, 0.3}
    };

    const std::vector<DummyDistributionInfo> distributions = make_dummy_distributions();

    std::vector<std::string> distribution_names;
    distribution_names.reserve(distributions.size());
    for (const auto& dist : distributions)
        distribution_names.push_back(dist.name);

    // -----------------------------------------------------------------------
    // Solver loop — one result file pair (txt + csv) per solver
    // -----------------------------------------------------------------------
    for (const auto& solver : solvers) {

        // Result containers: distribution x SNR (reset per solver)
        std::map<std::string, Eigen::VectorXd> fdr_results_map;
        std::map<std::string, Eigen::VectorXd> tpr_results_map;
        std::map<std::string, Eigen::VectorXd> average_L_results_map;
        std::map<std::string, Eigen::VectorXd> average_T_results_map;
        for (const auto& dist : distributions) {
            fdr_results_map[dist.name]       = Eigen::VectorXd(snr_values.size());
            tpr_results_map[dist.name]       = Eigen::VectorXd(snr_values.size());
            average_L_results_map[dist.name] = Eigen::VectorXd(snr_values.size());
            average_T_results_map[dist.name] = Eigen::VectorXd(snr_values.size());
        }

        // -------------------------------------------------------------------
        // Distribution loop
        // -------------------------------------------------------------------
        for (const auto& dist : distributions) {

            cdianostics::print_section_header(
                "Solver: " + solver.solver_name + "  |  Dummy distribution: " + dist.name);

            TRexControlParameter ctrl;
            ctrl.solver_type          = solver.solver_type;
            ctrl.solver_params.rho_afs = solver.rho_afs;
            ctrl.K                    = 20;
            ctrl.max_dummy_multiplier = 10;
            ctrl.use_max_T_stop       = true;
            ctrl.dummy_distribution   = dist.distribution;
            ctrl.lloop_strategy       = LLoopStrategy::STANDARD;

            // Simulation loop: sweep over SNR values
            for (std::size_t snr_idx = 0; snr_idx < snr_values.size(); ++snr_idx) {

                const double   snr       = snr_values[snr_idx];
                const unsigned base_seed = 24u + static_cast<unsigned>(snr_idx) * 1000u;

                std::cout << "SNR = " << std::fixed << std::setprecision(2) << snr << "\n";

                const auto make_data = [=](unsigned seed) -> trex_sim::TrexDGPData {
                    std::vector<std::size_t> pool;
                    if (block_support) {
                        pool.resize(s);
                        std::iota(pool.begin(), pool.end(), std::size_t{0});
                    } else {
                        std::mt19937 rng_sup(seed + 500000u);
                        pool.resize(static_cast<std::size_t>(p));
                        std::iota(pool.begin(), pool.end(), std::size_t{0});
                        std::shuffle(pool.begin(), pool.end(), rng_sup);
                        pool.resize(s);
                        std::sort(pool.begin(), pool.end());
                    }

                    std::vector<double> coefs;
                    coefs.reserve(s);
                    if (rnd_coef) {
                        std::mt19937 rng_coef(seed + 600000u);
                        std::normal_distribution<double> nd(0.0, 1.0);
                        for (std::size_t i = 0; i < s; ++i)
                            coefs.push_back(nd(rng_coef));
                    } else {
                        for (std::size_t i = 0; i < s; ++i)
                            coefs.push_back(1.0);
                    }

                    datagen::SyntheticData data(
                        n, p,
                        pool, coefs, snr,
                        static_cast<int>(seed),
                        datagen::predictor_policy::Normal(),
                        datagen::noisegen::noise_policy::Normal());

                    return {data.getX(), data.getY(), pool};
                };

                std::ostringstream lbl;
                lbl << "SNR=" << std::fixed << std::setprecision(2) << snr
                    << " [" << solver.solver_name << "/" << dist.name << "]";
                const auto result = trex_sim::run_mc_trials_trex(
                    num_MC, lbl.str(), make_data, tFDR, ctrl,
                    base_seed);

                const auto ei = static_cast<Eigen::Index>(snr_idx);
                fdr_results_map[dist.name](ei)        = result.avg_fdr;
                tpr_results_map[dist.name](ei)        = result.avg_tpr;
                average_L_results_map[dist.name](ei)  = result.avg_L;
                average_T_results_map[dist.name](ei)  = result.avg_T;
            }

            std::cout << "\n";
        }

        // -------------------------------------------------------------------
        // Print and save results for this solver
        // -------------------------------------------------------------------
        std::string solver_token = solver.solver_name;
        std::transform(solver_token.begin(), solver_token.end(),
                       solver_token.begin(), ::tolower);

        const std::string stem = "demo_trex_05_dummy_distributions_results_n"
            + std::to_string(static_cast<std::size_t>(n))
            + "_p" + std::to_string(static_cast<std::size_t>(p))
            + "_" + solver_token
            + (block_support ? "_block_support" : "_random_support");

        save_and_print_mc_results(
            num_MC,
            stem,
            snr_values,
            distribution_names,
            fdr_results_map,
            tpr_results_map,
            average_L_results_map,
            average_T_results_map);

        std::cout << "\n\n";
    }
}


// ==============================================================================
// Main
// ==============================================================================

int main() {

    std::cout.setf(std::ios::unitbuf);
    omp_set_num_threads(trex_mc_workers::count());
    std::cout << "Running with " << omp_get_max_threads() << " threads\n\n";

    if (true)
        demo_TRexSelector_dummy_distributions(/*num_MC=*/200,
                                              /*rnd_coef=*/false,
                                              /*block_support=*/false);
    if (false)
        demo_TRexSelector_dummy_distributions(/*num_MC=*/200,
                                              /*rnd_coef=*/false,
                                              /*block_support=*/true);
    return 0;
}
