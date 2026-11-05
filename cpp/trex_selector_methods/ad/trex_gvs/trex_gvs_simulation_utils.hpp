// ==============================================================================
// trex_gvs_simulation_utils.hpp
// ==============================================================================
#ifndef DEMOS_TREX_SELECTOR_METHODS_GVS_SIMULATION_UTILS_HPP
#define DEMOS_TREX_SELECTOR_METHODS_GVS_SIMULATION_UTILS_HPP
// ==============================================================================
/**
 * @file trex_gvs_simulation_utils.hpp
 *
 * @brief Monte Carlo infrastructure and sweep utilities for T-Rex+GVS demos.
 *
 * Provides:
 *   - GVSSimConfig              — simulation parameter struct
 *   - GVSSingleRunResult        — single-run diagnostics (FDP, TPP, T_stop, …)
 *   - GVSGridPointResult        — MC-aggregated (mean/sd FDP, TPP) per grid point
 *   - make_gvs_trex_control()   — default TRexControlParameter factory
 *   - run_gvs_single()          — single selector run, returns GVSSingleRunResult
 *   - print_single_run_result() — R-style formatted output block
 *   - GVSDGPFactory             — type alias for DGP closure
 *   - run_gvs_mc_trials()       — OpenMP-parallel MC inner loop
 *   - run_gvs_snr_sweep()       — outer SNR-grid sweep
 *   - print_mc_1d_method_table() — aligned per-method table for a 1-D sweep
 *   - print_mc_snr_table()      — aligned results table for SNR sweeps
 *   - run_gvs_rho_sweep()       — outer rho-grid sweep
 *   - print_mc_rho_table()      — aligned results table for rho sweeps
 *   - GVS2DDGPFactory           — type alias for 2-D DGP closure (snr, rho, seed)
 *   - run_gvs_2d_sweep()        — nested SNR × rho grid sweep
 *   - print_mc_matrix()         — named matrix table for 2-D sweep results
 *   - print_mc_param_sweep_table() — aligned table for a generic named-parameter sweep
 *   - save_mc_2d_tables()       — file writer (TXT + CSV) for 2-D sweep results
 *
 * Block-benchmark API (Demo 08 — HAC-discovered vs. oracle groups):
 *   - compute_block_hit_rate()        — share of active blocks with >=1 selection
 *   - compute_block_fdp()             — block-level false discovery proportion
 *   - compute_full_block_rate()       — share of active blocks recovered in full
 *   - compute_null_block_activation() — share of null blocks falsely activated
 *   - compute_block_purity_rate()     — block-to-group mapping purity
 *   - BlockMethodResult / run_block_method()  — one group method on one dataset
 *   - BlockTrialResult  / run_block_single()  — all methods on one dataset
 *   - BlockGridResult   / run_block_mc()       — MC aggregation over a grid
 *   - print_block_trial_result() / print_block_grid_table() — console output
 *
 * Includes trex_gvs_dgps.hpp (GVSDGPResult + all DGP generators).
 */
// ==============================================================================

#include "trex_gvs_dgps.hpp"
#include "../../cell_cache.hpp"   // resumable grid cells
#include "../../mc_workers.hpp"   // thread budget (TREX_NUM_WORKERS)
#include "../../solver_filter.hpp" // method selection (TREX_SOLVERS)

// std includes
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <unordered_set>
#include <sstream>
#include <string>
#include <vector>

// Eigen includes
#include <Eigen/Dense>

// OpenMP
#include <utils/openmp/utils_openmp.hpp>

// T-Rex+GVS includes
#include <trex_selector_methods/ad/trex_gvs/trex_gvs.hpp>

// Eval metrics
#include <utils/eval_metrics/utils_eval_cdiagnostics.hpp>
#include <utils/eval_metrics/utils_eval_rates.hpp>


// ==============================================================================
namespace gvs_demo {
// ==============================================================================


// ==============================================================================
// Namespace aliases
// ==============================================================================

namespace rates       = trex::utils::eval::rates;
namespace cdiag       = trex::utils::eval::cdiagnostics;
namespace hac         = trex::ml_methods::clustering::hierarchical::agglomerative;

using trex::trex_selector_methods::ad::trex_gvs::TRexGVSSelector;
using trex::trex_selector_methods::ad::trex_gvs::TRexGVSControlParameter;
using trex::trex_selector_methods::ad::trex_gvs::GVSType;
using trex::trex_selector_methods::ad::trex_gvs::ENSolverType;
using trex::trex_selector_methods::ad::trex_gvs::LambdaSelectionMethod;
using trex::trex_selector_methods::ad::trex_core::TRexSelector;
using trex::trex_selector_methods::ad::trex_core::TRexControlParameter;
using trex::trex_selector_methods::ad::trex_core::LLoopStrategy;
using trex::trex_selector_methods::ad::utils::solver_dispatch::SolverTypeForTRex;
using trex::trex_selector_methods::ad::utils::solver_dispatch::SolverHyperparameters;


// ==============================================================================
// Number formatting
// ==============================================================================

/**
 * @brief Format a double for console/title/label strings: fixed-point with at
 *        most @p max_decimals decimals, trailing zeros (and a trailing '.')
 *        trimmed. E.g. fmt_num(0.05) -> "0.05", fmt_num(2.0) -> "2".
 */
inline std::string fmt_num(double v, int max_decimals = 4) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(max_decimals) << v;
    std::string s = oss.str();
    if (s.find('.') != std::string::npos) {
        s.erase(s.find_last_not_of('0') + 1, std::string::npos);
        if (!s.empty() && s.back() == '.') s.pop_back();
    }
    return s;
}


// ==============================================================================
// Simulation configuration
// ==============================================================================

/** @brief Parameter bundle for MC simulations (SNR sweep, rho sweep, 2-D grid). */
struct GVSSimConfig {
    /** @brief Number of observations. */
    int n = 200;
    /** @brief Number of predictors. */
    int p = 500;
    /** @brief Within-group predictor noise. */
    double sd_x = 0.1;
    /** @brief Target FDR level. */
    double tFDR = 0.1;
    /** @brief Number of random experiments per selector run. */
    std::size_t K = 20;
    /** @brief Number of Monte Carlo trials per grid point. */
    std::size_t num_MC = 200;
    /** @brief Base RNG seed (trial i uses base_seed + i). */
    int base_seed = 2026;
    /** @brief Maximum correlation for auto-clustering HAC. */
    double corr_max = 0.98;
    /** @brief Fixed SNR for rho-sweep simulations. */
    double snr = 1.0;
    /** @brief EN solver variant (only applies to GVSType::EN). */
    ENSolverType en_solver = ENSolverType::TENET;
};


// ==============================================================================
// Single-run result
// ==============================================================================

/** @brief Diagnostics returned by run_gvs_single(). */
struct GVSSingleRunResult {
    /** @brief Selected variable indices (0-based). */
    std::vector<std::size_t> selected;
    /** @brief Realised false discovery proportion. */
    double fdp = 0.0;
    /** @brief Realised true positive proportion. */
    double tpp = 0.0;
    /** @brief lambda_2 resolved by the selector. */
    double lambda2_used = 0.0;
    /** @brief Calibrated stopping time. */
    std::size_t T_stop = 0;
    /** @brief Total dummy variables used. */
    std::size_t num_dummies = 0;
    /** @brief Maximum number of clusters detected. */
    std::size_t M_found = 0;
    /** @brief Wall-clock time in seconds. */
    double elapsed_sec = 0.0;
};


// ==============================================================================
// MC aggregate result
// ==============================================================================

/** @brief Aggregated (mean/sd FDP, TPP) across MC trials for one grid point. */
struct GVSGridPointResult {
    double mean_fdp = 0.0;
    double mean_tpp = 0.0;
    double sd_fdp   = 0.0;
    double sd_tpp   = 0.0;
};


// ==============================================================================
// Method selection (TREX_SOLVERS)
// ==============================================================================

/**
 * @brief True when the method of a run label is selected by TREX_SOLVERS.
 *
 * The method is the label part before '+' or ' ' ("TIENET_AUG+CV_1SE_CCD"
 * -> "TIENET_AUG"). Without a selection every method runs.
 */
inline bool method_selected(const std::string& label) {
    return trex_solver_filter::selected(label.substr(0, label.find_first_of("+ ")));
}

/**
 * @brief Monte Carlo trials of a demo: TREX_NUM_MC when set, else @p value.
 *
 * For short smoke runs. TREX_NUM_MC enters the cell-cache fingerprint, so a
 * smoke cache is never reused by a full run.
 */
inline std::size_t num_mc(std::size_t value) {
    const char* env = std::getenv("TREX_NUM_MC");
    if (env == nullptr || *env == '\0') return value;
    const long n = std::strtol(env, nullptr, 10);
    return n > 0 ? static_cast<std::size_t>(n) : value;
}

/** @brief Output file stem plus the selection suffix, e.g. "_tienet_aug". */
inline std::string out_stem(const std::string& stem) {
    return stem + trex_solver_filter::file_suffix();
}

/** @brief Grid point of a skipped (unselected) method: all metrics NaN. */
inline GVSGridPointResult skipped_point() {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    return {nan, nan, nan, nan};
}


// ==============================================================================
// Default control parameter factory
// ==============================================================================

/**
 * @brief Build a TRexControlParameter with sensible GVS defaults.
 *
 * @param K Number of random experiments (default: 20).
 */
inline TRexControlParameter make_gvs_trex_control(std::size_t K = 20) {
    TRexControlParameter ctrl;
    ctrl.K                        = K;
    ctrl.max_dummy_multiplier     = 10;
    ctrl.use_max_T_stop           = true;
    ctrl.lloop_strategy           = LLoopStrategy::HCONCAT;
    ctrl.tloop_stagnation_stop    = true;
    ctrl.tloop_max_stagnant_steps = 5;
    ctrl.use_memory_mapping       = false;
    return ctrl;
}


// ==============================================================================
// Single-run helper
// ==============================================================================

/**
 * @brief Run TRexGVSSelector once and collect diagnostics.
 *
 * @param X_map      Non-owning Eigen::Map over the design matrix.
 * @param y_map      Non-owning Eigen::Map over the response.
 * @param true_sup   True support for FDP/TPP computation.
 * @param tFDR       Target FDR level.
 * @param gvs_ctrl   GVS-specific control parameters.
 * @param trex_ctrl  Base T-Rex control parameters.
 * @param seed       RNG seed passed to the selector.
 *
 * @return GVSSingleRunResult with realised FDP, TPP, and selector diagnostics.
 */
inline GVSSingleRunResult run_gvs_single(
    Eigen::Map<Eigen::MatrixXd>&        X_map,
    Eigen::Map<Eigen::VectorXd>&        y_map,
    const std::vector<std::size_t>&     true_sup,
    double                              tFDR,
    const TRexGVSControlParameter&      gvs_ctrl,
    const TRexControlParameter&         trex_ctrl,
    int                                 seed)
{
    const auto t0 = std::chrono::steady_clock::now();

    // IEN requires TIENET_AUG (row-augmented system absorbs the group penalty).
    // EN solver variant is controlled by gvs_ctrl.en_solver (default: TENET_AUG).
    // TRexGVSControlParameter nests its own trex_ctrl member.
    TRexGVSControlParameter trex_gvs_ctrl = gvs_ctrl;
    trex_gvs_ctrl.trex_ctrl = trex_ctrl;
    if (trex_gvs_ctrl.gvs_type != GVSType::EN)
        trex_gvs_ctrl.trex_ctrl.solver_type = SolverTypeForTRex::TIENET_AUG;

    TRexGVSSelector selector(X_map, y_map, tFDR,
                             trex_gvs_ctrl,
                             seed, /*verbose=*/false);
    selector.select();

    const auto t1 = std::chrono::steady_clock::now();
    const double elapsed = std::chrono::duration<double>(t1 - t0).count();

    const auto& gvs_res = selector.getGVSResult();
    const auto& sel_idx = selector.getSelectedIndices();

    GVSSingleRunResult out;
    out.selected     = sel_idx;
    out.fdp          = rates::compute_fdp(sel_idx, true_sup);
    out.tpp          = rates::compute_tpp(sel_idx, true_sup);
    out.lambda2_used = gvs_res.lambda2_used;
    out.T_stop       = gvs_res.T_stop;
    out.num_dummies  = gvs_res.num_dummies;
    out.M_found      = gvs_res.max_clusters;
    out.elapsed_sec  = elapsed;

    return out;
}


// ==============================================================================
// Print helpers
// ==============================================================================

/**
 * @brief Print a formatted single-run result block (mirrors R's .print_result()).
 *
 * @param tag    Short scenario / method label (e.g. "Hastie GVS [TENET]").
 * @param dat    DGP metadata.
 * @param res    Single-run result to display.
 * @param tFDR   Target FDR (printed for reference).
 */
inline void print_single_run_result(const std::string&       tag,
                                    const GVSDGPResult&       dat,
                                    const GVSSingleRunResult& res,
                                    double                    tFDR)
{
    const int n_sel = static_cast<int>(res.selected.size());
    const int n_tp  = static_cast<int>(
        std::count_if(res.selected.begin(), res.selected.end(),
                      [&](std::size_t idx) {
                          return std::binary_search(dat.true_support.begin(),
                                                    dat.true_support.end(),
                                                    idx);
                      }));
    const int n_fp = n_sel - n_tp;

    std::cout << std::string(70, '=') << "\n";
    std::cout << "  " << tag << "\n";
    std::cout << std::string(70, '-') << "\n";
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "  Data:  n = " << dat.n << ", p = " << dat.p
              << ", tFDR = " << tFDR << ", s = " << dat.s << "\n";
    std::cout << "         SNR = " << std::setprecision(2) << dat.snr
              << ",  sigma_y = " << std::setprecision(4) << dat.sigma_y
              << ",  sd_x = " << dat.sd_x << "\n";
    std::cout << std::string(70, '-') << "\n";
    std::cout << "  Calibration:  T_stop = " << res.T_stop
              << ",  dummies = " << res.num_dummies
              << ",  M = " << res.M_found
              << ",  lambda2 = " << std::setprecision(4) << res.lambda2_used << "\n";
    std::cout << "  Selection:    " << n_sel << " selected"
              << "  |  TP = " << n_tp << "  FP = " << n_fp << "\n";
    std::cout << "  Rates:        TPP = " << std::setprecision(3) << res.tpp
              << "  |  FDP = " << res.fdp
              << "  (target tFDR <= " << tFDR << ")\n";
    std::cout << "  Wall-clock:   " << std::setprecision(2) << res.elapsed_sec << " s\n";
    std::cout << std::string(70, '=') << "\n\n";
}


// ==============================================================================
// MC inner loop
// ==============================================================================

/**
 * @brief Type alias for a DGP factory function.
 *
 * Signature: (snr_or_rho, seed) → GVSDGPResult.
 * Captures scenario-specific parameters (n, p, sd_x, …) via closure.
 */
using GVSDGPFactory = std::function<GVSDGPResult(double snr, unsigned seed)>;


// ==============================================================================
// Resumable grid cells (../../cell_cache.hpp)
// ==============================================================================

namespace cell_detail {

/** @brief Cell key: label plus the GVS parameters that tell cells apart. */
inline std::string key(const std::string& kind, const std::string& label, std::size_t num_MC,
                       unsigned base_seed, double tFDR, const TRexGVSControlParameter& gvs) {
    std::ostringstream os;
    os << kind << '|' << label << "|gvs=" << static_cast<int>(gvs.gvs_type)
       << "|en=" << static_cast<int>(gvs.en_solver) << "|link=" << static_cast<int>(gvs.hc_linkage)
       << "|l2m=" << static_cast<int>(gvs.lambda2_method) << "|corr=" << gvs.corr_max
       << "|seed=" << base_seed << "|mc=" << num_MC << "|tfdr=" << tFDR;
    return os.str();
}

inline trex_cell_cache::Cell to_cell(const GVSGridPointResult& r) {
    return {{"mean_fdp", r.mean_fdp}, {"mean_tpp", r.mean_tpp},
            {"sd_fdp", r.sd_fdp}, {"sd_tpp", r.sd_tpp}};
}

inline GVSGridPointResult from_cell(const trex_cell_cache::Cell& c) {
    using trex_cell_cache::get;
    return {get(c, "mean_fdp"), get(c, "mean_tpp"), get(c, "sd_fdp"), get(c, "sd_tpp")};
}

}  // namespace cell_detail


/** @brief Uncached body of run_gvs_mc_trials(). */
inline GVSGridPointResult run_gvs_mc_trials_impl(
    const GVSDGPFactory&           dgp_fn,
    double                         param,
    std::size_t                    num_MC,
    const std::string&             progress_label,
    double                         tFDR,
    const TRexGVSControlParameter& gvs_ctrl,
    const TRexControlParameter&    trex_ctrl,
    unsigned                       base_seed)
{
    const int iMC = static_cast<int>(num_MC);
    std::vector<double> fdp_vec(num_MC, 0.0);
    std::vector<double> tpp_vec(num_MC, 0.0);

    std::cout << "  " << progress_label
              << " — Running " << num_MC << " MC trials ...\n" << std::flush;

    #pragma omp parallel for schedule(dynamic)
    for (int mc = 0; mc < iMC; ++mc) {
        const unsigned trial_seed = base_seed + static_cast<unsigned>(mc);
        auto dat = dgp_fn(param, trial_seed);

        Eigen::Map<Eigen::MatrixXd> X_map(dat.X.data(),
                                           dat.X.rows(), dat.X.cols());
        Eigen::Map<Eigen::VectorXd> y_map(dat.y.data(), dat.y.rows());

        // Derive solver_type from gvs_ctrl to satisfy the TRexGVSSelector
        // compatibility check (EN→TENET/TENET_AUG, IEN→TIENET_AUG).
        // Per-trial cv_seed ensures unique fold assignments.
        // TRexGVSControlParameter nests its own trex_ctrl member.
        TRexGVSControlParameter trex_gvs_ctrl = gvs_ctrl;
        trex_gvs_ctrl.trex_ctrl = trex_ctrl;
        trex_gvs_ctrl.cv_seed = trial_seed;
        if (trex_gvs_ctrl.gvs_type == GVSType::EN)
            trex_gvs_ctrl.trex_ctrl.solver_type =
                (trex_gvs_ctrl.en_solver == ENSolverType::TENET_AUG)
                    ? SolverTypeForTRex::TENET_AUG
                    : SolverTypeForTRex::TENET;
        else
            trex_gvs_ctrl.trex_ctrl.solver_type = SolverTypeForTRex::TIENET_AUG;

        TRexGVSSelector selector(X_map, y_map, tFDR,
                                 trex_gvs_ctrl,
                                 static_cast<int>(trial_seed),
                                 /*verbose=*/false);
        selector.select();

        const auto& sel_idx = selector.getSelectedIndices();
        fdp_vec[static_cast<std::size_t>(mc)] =
            rates::compute_fdp(sel_idx, dat.true_support);
        tpp_vec[static_cast<std::size_t>(mc)] =
            rates::compute_tpp(sel_idx, dat.true_support);
    }

    const double dMC = static_cast<double>(num_MC);
    double mean_fdp = 0.0, mean_tpp = 0.0;
    for (int mc = 0; mc < iMC; ++mc) {
        mean_fdp += fdp_vec[static_cast<std::size_t>(mc)];
        mean_tpp += tpp_vec[static_cast<std::size_t>(mc)];
    }
    mean_fdp /= dMC;
    mean_tpp /= dMC;

    double sd_fdp = 0.0, sd_tpp = 0.0;
    if (num_MC > 1) {
        double sq_fdp = 0.0, sq_tpp = 0.0;
        for (int mc = 0; mc < iMC; ++mc) {
            const std::size_t idx = static_cast<std::size_t>(mc);
            sq_fdp += (fdp_vec[idx] - mean_fdp) * (fdp_vec[idx] - mean_fdp);
            sq_tpp += (tpp_vec[idx] - mean_tpp) * (tpp_vec[idx] - mean_tpp);
        }
        sd_fdp = std::sqrt(sq_fdp / (dMC - 1.0));
        sd_tpp = std::sqrt(sq_tpp / (dMC - 1.0));
    }

    std::cout << "  " << progress_label
              << " — done. TPP=" << std::fixed << std::setprecision(3) << mean_tpp
              << "  FDP=" << mean_fdp << "\n\n" << std::flush;

    return {mean_fdp, mean_tpp, sd_fdp, sd_tpp};
}

/**
 * @brief Run num_MC parallel trials at a single parameter point (resumable grid cell).
 *
 * Each trial:
 *  1. Generates a fresh dataset via dgp_fn(param, trial_seed).
 *  2. Runs TRexGVSSelector (verbose=false).
 *  3. Collects FDP and TPP.
 *
 * @param dgp_fn         DGP factory — must be thread-safe (no shared mutable state).
 * @param param          Parameter value passed to dgp_fn (SNR or rho).
 * @param num_MC         Number of trials.
 * @param progress_label Label printed before/after the run.
 * @param tFDR           Target FDR level.
 * @param gvs_ctrl       GVS control parameters.
 * @param trex_ctrl      Base T-Rex control parameters.
 * @param base_seed      Seed for trial 0; trial i uses base_seed + i.
 *
 * @return GVSGridPointResult with mean and sd of FDP and TPP.
 *
 * @note The progress label should carry the parameter reading (e.g.
 *       "SNR = 0.10", as in the trex_da demos); @p param itself is only
 *       forwarded to the DGP factory.
 */
inline GVSGridPointResult run_gvs_mc_trials(
    const GVSDGPFactory&           dgp_fn,
    double                         param,
    std::size_t                    num_MC,
    const std::string&             progress_label,
    double                         tFDR,
    const TRexGVSControlParameter& gvs_ctrl,
    const TRexControlParameter&    trex_ctrl,
    unsigned                       base_seed)
{
    if (!method_selected(progress_label)) return skipped_point();
    std::ostringstream lbl;
    lbl << progress_label << "|param=" << param;
    return cell_detail::from_cell(TREX_CELL_CACHED(
        cell_detail::key("gvs", lbl.str(), num_MC, base_seed, tFDR, gvs_ctrl), [&] {
            return cell_detail::to_cell(run_gvs_mc_trials_impl(dgp_fn, param, num_MC, progress_label,
                                                               tFDR, gvs_ctrl, trex_ctrl, base_seed));
        }));
}


// ==============================================================================
// Clustering-cut comparison (fixed cut, noise floor, oracle groups)
// ==============================================================================

/**
 * @brief Bonferroni noise floor for |corr| between independent columns.
 *
 * @details corr_max = tanh(z / sqrt(n - 3)) with P(Z > z) = alpha / (2 m) and
 *  m = p (p - 1) / 2 column pairs (Fisher z). Correlations above it are not
 *  explained by independence, so a cut there merges only real groups.
 *
 * @param n      Observations.
 * @param p      Variables.
 * @param alpha  Family-wise level (default 0.05).
 * @return The cut as corr_max (0.346 at n = 200, p = 500).
 */
inline double noise_floor_corr_max(int n, int p, double alpha = 0.05) {
    const double m = 0.5 * static_cast<double>(p) * static_cast<double>(p - 1);
    const double q = alpha / (2.0 * m);
    double lo = 0.0, hi = 40.0;  // upper normal quantile by bisection on erfc
    for (int it = 0; it < 200; ++it) {
        const double mid = 0.5 * (lo + hi);
        if (0.5 * std::erfc(mid / std::sqrt(2.0)) > q) lo = mid; else hi = mid;
    }
    return std::tanh(0.5 * (lo + hi) / std::sqrt(static_cast<double>(n - 3)));
}

/** @brief Mean and sd of FDP / TPP plus group-level rates over MC trials. */
struct GVSCutPointResult {
    double mean_fdp = 0.0, sd_fdp = 0.0;    ///< variable-level FDP
    double mean_tpp = 0.0, sd_tpp = 0.0;    ///< variable-level TPP
    double grp_hit  = 0.0, sd_grp_hit  = 0.0;  ///< share of active groups with >= 1 selection
    double grp_full = 0.0, sd_grp_full = 0.0;  ///< share of active groups selected in full
    double grp_fdp  = 0.0, sd_grp_fdp  = 0.0;  ///< share of touched groups without an active variable
    double n_groups = 0.0, sd_n_groups = 0.0;  ///< clusters with >= 2 members formed by the selector
};

namespace cell_detail {

inline trex_cell_cache::Cell cut_to_cell(const GVSCutPointResult& r) {
    return {{"mean_fdp", r.mean_fdp}, {"sd_fdp", r.sd_fdp},
            {"mean_tpp", r.mean_tpp}, {"sd_tpp", r.sd_tpp},
            {"grp_hit", r.grp_hit}, {"sd_grp_hit", r.sd_grp_hit},
            {"grp_full", r.grp_full}, {"sd_grp_full", r.sd_grp_full},
            {"grp_fdp", r.grp_fdp}, {"sd_grp_fdp", r.sd_grp_fdp},
            {"n_groups", r.n_groups}, {"sd_n_groups", r.sd_n_groups}};
}

inline GVSCutPointResult cut_from_cell(const trex_cell_cache::Cell& c) {
    using trex_cell_cache::get;
    GVSCutPointResult r;
    r.mean_fdp = get(c, "mean_fdp"); r.sd_fdp = get(c, "sd_fdp");
    r.mean_tpp = get(c, "mean_tpp"); r.sd_tpp = get(c, "sd_tpp");
    r.grp_hit = get(c, "grp_hit"); r.sd_grp_hit = get(c, "sd_grp_hit");
    r.grp_full = get(c, "grp_full"); r.sd_grp_full = get(c, "sd_grp_full");
    r.grp_fdp = get(c, "grp_fdp"); r.sd_grp_fdp = get(c, "sd_grp_fdp");
    r.n_groups = get(c, "n_groups"); r.sd_n_groups = get(c, "sd_n_groups");
    return r;
}

}  // namespace cell_detail

/**
 * @brief Group-level rates of one selection against the DGP's prior_groups.
 *
 * @details Oracle groups are the DGP labels (background columns are singleton
 *  groups); a group is active when it holds an active variable.
 */
inline void gvs_group_rates(const std::vector<std::size_t>& sel, const GVSDGPResult& dat,
                            double& hit, double& full, double& fdp) {
    const auto& g = dat.prior_groups;
    Eigen::Index n_groups = 0;
    for (const auto l : g) n_groups = std::max(n_groups, l + 1);
    std::vector<int> size(static_cast<std::size_t>(n_groups), 0);
    std::vector<int> active(size.size(), 0), chosen(size.size(), 0);
    for (const auto l : g) ++size[static_cast<std::size_t>(l)];
    for (const auto j : dat.true_support) ++active[static_cast<std::size_t>(g[j])];
    for (const auto j : sel) ++chosen[static_cast<std::size_t>(g[j])];

    int n_active = 0, n_hit = 0, n_full = 0, n_touched = 0, n_false = 0;
    for (std::size_t m = 0; m < size.size(); ++m) {
        if (active[m] > 0) {
            ++n_active;
            if (chosen[m] > 0) ++n_hit;
            if (chosen[m] == size[m]) ++n_full;
        }
        if (chosen[m] > 0) {
            ++n_touched;
            if (active[m] == 0) ++n_false;
        }
    }
    hit  = n_active ? static_cast<double>(n_hit) / n_active : 0.0;
    full = n_active ? static_cast<double>(n_full) / n_active : 0.0;
    fdp  = n_touched ? static_cast<double>(n_false) / n_touched : 0.0;
}

/** @brief Uncached body of run_gvs_mc_trials_cut(). */
inline GVSCutPointResult run_gvs_mc_trials_cut_impl(
    const GVSDGPFactory&           dgp_fn,
    std::size_t                    num_MC,
    const std::string&             progress_label,
    double                         tFDR,
    const TRexGVSControlParameter& gvs_ctrl,
    const TRexControlParameter&    trex_ctrl,
    unsigned                       base_seed,
    bool                           oracle)
{
    const int iMC = static_cast<int>(num_MC);
    std::vector<double> fdp(num_MC), tpp(num_MC), hit(num_MC), full(num_MC),
                        gfdp(num_MC), groups(num_MC);

    std::cout << "  " << progress_label
              << " — Running " << num_MC << " MC trials ...\n" << std::flush;

    #pragma omp parallel for schedule(dynamic)
    for (int mc = 0; mc < iMC; ++mc) {
        const unsigned trial_seed = base_seed + static_cast<unsigned>(mc);
        auto dat = dgp_fn(0.0, trial_seed);
        Eigen::Map<Eigen::MatrixXd> X_map(dat.X.data(), dat.X.rows(), dat.X.cols());
        Eigen::Map<Eigen::VectorXd> y_map(dat.y.data(), dat.y.rows());

        // Same selector setup as run_gvs_mc_trials_impl(); oracle groups bypass the HAC cut
        TRexGVSControlParameter ctrl = gvs_ctrl;
        ctrl.trex_ctrl = trex_ctrl;
        ctrl.cv_seed   = trial_seed;
        if (ctrl.gvs_type == GVSType::EN)
            ctrl.trex_ctrl.solver_type = (ctrl.en_solver == ENSolverType::TENET_AUG)
                                             ? SolverTypeForTRex::TENET_AUG
                                             : SolverTypeForTRex::TENET;
        else
            ctrl.trex_ctrl.solver_type = SolverTypeForTRex::TIENET_AUG;
        if (oracle) ctrl.prior_groups = dat.prior_groups;

        TRexGVSSelector selector(X_map, y_map, tFDR, ctrl,
                                 static_cast<int>(trial_seed), /*verbose=*/false);
        selector.select();

        const auto& sel = selector.getSelectedIndices();
        const auto& res = selector.getGVSResult();
        std::vector<int> size(res.max_clusters, 0);
        for (Eigen::Index j = 0; j < res.groups_vec.size(); ++j)
            ++size[static_cast<std::size_t>(res.groups_vec(j))];
        int multi = 0;
        for (const int s : size) if (s >= 2) ++multi;

        const auto i = static_cast<std::size_t>(mc);
        fdp[i]    = rates::compute_fdp(sel, dat.true_support);
        tpp[i]    = rates::compute_tpp(sel, dat.true_support);
        groups[i] = multi;
        gvs_group_rates(sel, dat, hit[i], full[i], gfdp[i]);
    }

    const auto mean_sd = [](const std::vector<double>& v, double& mean, double& sd) {
        const double m = static_cast<double>(v.size());
        mean = std::accumulate(v.begin(), v.end(), 0.0) / m;
        double sq = 0.0;
        for (const double x : v) sq += (x - mean) * (x - mean);
        sd = v.size() > 1 ? std::sqrt(sq / (m - 1.0)) : 0.0;
    };
    GVSCutPointResult r;
    mean_sd(fdp, r.mean_fdp, r.sd_fdp);
    mean_sd(tpp, r.mean_tpp, r.sd_tpp);
    mean_sd(hit, r.grp_hit, r.sd_grp_hit);
    mean_sd(full, r.grp_full, r.sd_grp_full);
    mean_sd(gfdp, r.grp_fdp, r.sd_grp_fdp);
    mean_sd(groups, r.n_groups, r.sd_n_groups);

    std::cout << "  " << progress_label << " — done. TPP=" << std::fixed << std::setprecision(3)
              << r.mean_tpp << "  FDP=" << r.mean_fdp << "  groups=" << r.n_groups << "\n\n"
              << std::flush;
    return r;
}

/**
 * @brief num_MC parallel trials at one fixed operating point under one cut
 *        (resumable grid cell).
 *
 * @param dgp_fn          DGP factory; its first argument is ignored.
 * @param num_MC          Number of trials.
 * @param progress_label  Label printed before/after the run.
 * @param tFDR            Target FDR level.
 * @param gvs_ctrl        GVS control parameters, including the cut corr_max.
 * @param trex_ctrl       Base T-Rex control parameters.
 * @param base_seed       Seed for trial 0; trial i uses base_seed + i.
 * @param oracle          Pass the DGP's prior_groups (the cut is then unused).
 * @return Variable-level FDP/TPP and group-level rates.
 */
inline GVSCutPointResult run_gvs_mc_trials_cut(
    const GVSDGPFactory&           dgp_fn,
    std::size_t                    num_MC,
    const std::string&             progress_label,
    double                         tFDR,
    const TRexGVSControlParameter& gvs_ctrl,
    const TRexControlParameter&    trex_ctrl,
    unsigned                       base_seed,
    bool                           oracle)
{
    if (!method_selected(progress_label)) {
        const double nan = std::numeric_limits<double>::quiet_NaN();
        return {nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan};
    }
    std::ostringstream lbl;
    lbl << progress_label << "|oracle=" << (oracle ? 1 : 0);
    return cell_detail::cut_from_cell(TREX_CELL_CACHED(
        cell_detail::key("gvscut", lbl.str(), num_MC, base_seed, tFDR, gvs_ctrl), [&] {
            return cell_detail::cut_to_cell(run_gvs_mc_trials_cut_impl(
                dgp_fn, num_MC, progress_label, tFDR, gvs_ctrl, trex_ctrl, base_seed, oracle));
        }));
}


// ==============================================================================
// SNR sweep
// ==============================================================================

/**
 * @brief Run a full SNR-grid MC sweep for one (TENET, TENET_AUG, or TIENET_AUG) method variant.
 *
 * @param dgp_fn     DGP factory — must be thread-safe.
 * @param snr_grid   Vector of SNR values to sweep over.
 * @param cfg        Simulation config (num_MC, base_seed, tFDR, …).
 * @param gvs_ctrl   GVS control parameters (gvs_type is set by caller).
 * @param trex_ctrl  Base T-Rex control parameters.
 * @param label      Short label for progress messages (e.g. "TENET", "TIENET_AUG").
 *
 * @return One GVSGridPointResult per entry in snr_grid.
 */
inline std::vector<GVSGridPointResult> run_gvs_snr_sweep(
    const GVSDGPFactory&           dgp_fn,
    const std::vector<double>&     snr_grid,
    const GVSSimConfig&            cfg,
    const TRexGVSControlParameter& gvs_ctrl,
    const TRexControlParameter&    trex_ctrl,
    const std::string&             label)
{
    std::vector<GVSGridPointResult> results;
    results.reserve(snr_grid.size());

    std::cout << "\n  Method: " << label << "\n";

    for (double snr : snr_grid) {
        std::ostringstream lbl;
        lbl << "SNR = " << std::fixed << std::setprecision(2) << snr;
        results.push_back(
            run_gvs_mc_trials(
                dgp_fn, snr, cfg.num_MC, lbl.str(),
                cfg.tFDR, gvs_ctrl, trex_ctrl,
                static_cast<unsigned>(cfg.base_seed)));
    }

    return results;
}


// ==============================================================================
// Print MC 1-D method table (shared helper for SNR and rho sweeps)
// ==============================================================================

/**
 * @brief Print a table for one method variant with all four metrics as columns.
 *
 * Columns: mean_FDP  sd_FDP  mean_TPP  sd_TPP
 *
 * @param method_label  Method name displayed as table heading (e.g. "TENET").
 * @param param_labels  Labels for each row (e.g. "snr=0.10", "rho=0.30").
 * @param results       One GVSGridPointResult per row.
 */
inline void print_mc_1d_method_table(
    const std::string&                     method_label,
    const std::vector<std::string>&        param_labels,
    const std::vector<GVSGridPointResult>& results,
    std::ofstream*                         fout = nullptr)
{
    int rw = 12;            // row-label width, widened for long labels
    for (const auto& lbl : param_labels) rw = std::max(rw, static_cast<int>(lbl.size()) + 2);
    constexpr int cw = 10;  // value column width
    const int total_w = rw + 4 * cw;

    std::ostringstream oss;
    oss << "\n  " << method_label << "\n"
        << "  " << std::string(total_w, '-') << "\n"
        << "  " << std::left << std::setw(rw) << ""
        << std::right
        << std::setw(cw) << "mean_FDP"
        << std::setw(cw) << "sd_FDP"
        << std::setw(cw) << "mean_TPP"
        << std::setw(cw) << "sd_TPP"
        << "\n"
        << "  " << std::string(total_w, '-') << "\n";

    for (std::size_t r = 0; r < results.size() && r < param_labels.size(); ++r) {
        oss << "  " << std::left  << std::setw(rw) << param_labels[r]
            << std::right << std::fixed << std::setprecision(4)
            << std::setw(cw) << results[r].mean_fdp
            << std::setw(cw) << results[r].sd_fdp
            << std::setw(cw) << results[r].mean_tpp
            << std::setw(cw) << results[r].sd_tpp
            << "\n";
    }
    oss << "\n";

    const std::string s = oss.str();
    std::cout << s;
    if (fout && fout->is_open()) *fout << s;
}


// ==============================================================================
// Print MC SNR table
// ==============================================================================

/**
 * @brief Print three per-method tables for an SNR sweep (TENET, TENET_AUG, TIENET_AUG).
 *
 * Each table shows mean_FDP, sd_FDP, mean_TPP, sd_TPP as columns.
 * If file_stem is non-empty, also writes a .txt and a .csv to DEMO_OUTPUT_DIR.
 *
 * @param scenario_tag    Short label printed in the header.
 * @param snr_grid        The swept SNR values.
 * @param en_results      TENET results per SNR point.
 * @param en_aug_results  TENET_AUG results per SNR point.
 * @param ien_results     TIENET_AUG results per SNR point.
 * @param cfg             Simulation config (num_MC, tFDR, …).
 * @param file_stem       Output file stem (no folder, no extension). Empty = no file output.
 */
inline void print_mc_snr_table(
    const std::string&                     scenario_tag,
    const std::vector<double>&             snr_grid,
    const std::vector<GVSGridPointResult>& en_results,
    const std::vector<GVSGridPointResult>& en_aug_results,
    const std::vector<GVSGridPointResult>& ien_results,
    const GVSSimConfig&                    cfg,
    const std::string&                     file_stem = "")
{
    const std::string dir(DEMO_OUTPUT_DIR);
    std::ofstream txt_file;
    if (!file_stem.empty()) {
        std::filesystem::create_directories(dir);
        txt_file.open(dir + out_stem(file_stem) + ".txt");
    }
    std::ofstream* fout = txt_file.is_open() ? &txt_file : nullptr;

    auto write = [&](const std::string& s) {
        std::cout << s;
        if (fout) *fout << s;
    };

    {
        std::ostringstream hdr;
        hdr << "\n" << std::string(78, '=') << "\n"
            << "  T-Rex+GVS MC: " << scenario_tag
            << "  (MC=" << cfg.num_MC << ", tFDR=" << cfg.tFDR
            << ", n=" << cfg.n << ", p=" << cfg.p << ")\n"
            << std::string(78, '=') << "\n";
        write(hdr.str());
    }

    std::vector<std::string> labels;
    labels.reserve(snr_grid.size());
    for (double s : snr_grid)
        labels.push_back("snr=" + fmt_num(s));

    print_mc_1d_method_table("TENET",     labels, en_results,     fout);
    print_mc_1d_method_table("TENET_AUG", labels, en_aug_results, fout);
    print_mc_1d_method_table("TIENET_AUG",    labels, ien_results,    fout);

    write(std::string(78, '=') + "\n\n");

    if (!file_stem.empty()) {
        if (txt_file.is_open()) {
            txt_file.close();
            std::cout << "[Info] TXT saved to: " << dir + out_stem(file_stem) + ".txt\n";
        }
        // CSV
        std::ofstream csv(dir + out_stem(file_stem) + ".csv");
        if (csv.is_open()) {
            csv << "method,metric,snr,mean,sd\n"
                << std::fixed << std::setprecision(6);
            for (std::size_t i = 0; i < snr_grid.size(); ++i) {
                const double snr = snr_grid[i];
                csv << "TENET,mean_FDP,"     << snr << "," << en_results[i].mean_fdp     << "," << en_results[i].sd_fdp     << "\n";
                csv << "TENET,mean_TPP,"     << snr << "," << en_results[i].mean_tpp     << "," << en_results[i].sd_tpp     << "\n";
                csv << "TENET_AUG,mean_FDP," << snr << "," << en_aug_results[i].mean_fdp << "," << en_aug_results[i].sd_fdp << "\n";
                csv << "TENET_AUG,mean_TPP," << snr << "," << en_aug_results[i].mean_tpp << "," << en_aug_results[i].sd_tpp << "\n";
                csv << "TIENET_AUG,mean_FDP,"    << snr << "," << ien_results[i].mean_fdp    << "," << ien_results[i].sd_fdp    << "\n";
                csv << "TIENET_AUG,mean_TPP,"    << snr << "," << ien_results[i].mean_tpp    << "," << ien_results[i].sd_tpp    << "\n";
            }
            std::cout << "[Info] CSV saved to: " << dir + out_stem(file_stem) + ".csv\n\n";
        }
    }
}


// ==============================================================================
// rho sweep
// ==============================================================================

/**
 * @brief Type alias documenting that the DGP factory's first argument is rho.
 *
 * Functionally identical to GVSDGPFactory — the alias exists to document
 * caller intent.
 */
using GVSRhoDGPFactory = GVSDGPFactory;


/**
 * @brief Run a full rho-grid MC sweep for one (TENET, TENET_AUG, or TIENET_AUG) method variant.
 *
 * @param dgp_fn     DGP factory whose first argument is rho (not SNR).
 * @param rho_grid   Vector of rho values to sweep over.
 * @param cfg        Simulation config (num_MC, base_seed, tFDR, snr, …).
 * @param gvs_ctrl   GVS control parameters.
 * @param trex_ctrl  Base T-Rex control parameters.
 * @param label      Short label for progress messages (e.g. "TENET", "TIENET_AUG").
 *
 * @return One GVSGridPointResult per entry in rho_grid.
 */
inline std::vector<GVSGridPointResult> run_gvs_rho_sweep(
    const GVSRhoDGPFactory&        dgp_fn,
    const std::vector<double>&     rho_grid,
    const GVSSimConfig&            cfg,
    const TRexGVSControlParameter& gvs_ctrl,
    const TRexControlParameter&    trex_ctrl,
    const std::string&             label)
{
    std::vector<GVSGridPointResult> results;
    results.reserve(rho_grid.size());

    std::cout << "\n  Method: " << label << "\n";

    for (double rho : rho_grid) {
        std::ostringstream lbl;
        lbl << "Rho = " << std::fixed << std::setprecision(2) << rho;
        results.push_back(
            run_gvs_mc_trials(
                dgp_fn, rho, cfg.num_MC, lbl.str(),
                cfg.tFDR, gvs_ctrl, trex_ctrl,
                static_cast<unsigned>(cfg.base_seed)));
    }

    return results;
}


// ==============================================================================
// Print MC rho table
// ==============================================================================

/**
 * @brief Print three per-method tables for a rho sweep (TENET, TENET_AUG, TIENET_AUG).
 *
 * Each table shows mean_FDP, sd_FDP, mean_TPP, sd_TPP as columns.
 * If file_stem is non-empty, also writes a .txt and a .csv to DEMO_OUTPUT_DIR.
 *
 * @param scenario_tag    Short label printed in the header.
 * @param rho_grid        The swept rho values.
 * @param en_results      TENET results per rho point.
 * @param en_aug_results  TENET_AUG results per rho point.
 * @param ien_results     TIENET_AUG results per rho point.
 * @param cfg             Simulation config (num_MC, tFDR, snr, …).
 * @param file_stem       Output file stem (no folder, no extension). Empty = no file output.
 */
inline void print_mc_rho_table(
    const std::string&                     scenario_tag,
    const std::vector<double>&             rho_grid,
    const std::vector<GVSGridPointResult>& en_results,
    const std::vector<GVSGridPointResult>& en_aug_results,
    const std::vector<GVSGridPointResult>& ien_results,
    const GVSSimConfig&                    cfg,
    const std::string&                     file_stem = "")
{
    const std::string dir(DEMO_OUTPUT_DIR);
    std::ofstream txt_file;
    if (!file_stem.empty()) {
        std::filesystem::create_directories(dir);
        txt_file.open(dir + out_stem(file_stem) + ".txt");
    }
    std::ofstream* fout = txt_file.is_open() ? &txt_file : nullptr;

    auto write = [&](const std::string& s) {
        std::cout << s;
        if (fout) *fout << s;
    };

    {
        std::ostringstream hdr;
        hdr << "\n" << std::string(78, '=') << "\n"
            << "  T-Rex+GVS MC: " << scenario_tag
            << "  (MC=" << cfg.num_MC << ", tFDR=" << cfg.tFDR
            << ", SNR=" << cfg.snr
            << ", n=" << cfg.n << ", p=" << cfg.p << ")\n"
            << std::string(78, '=') << "\n";
        write(hdr.str());
    }

    std::vector<std::string> labels;
    labels.reserve(rho_grid.size());
    for (double r : rho_grid)
        labels.push_back("rho=" + fmt_num(r));

    print_mc_1d_method_table("TENET",     labels, en_results,     fout);
    print_mc_1d_method_table("TENET_AUG", labels, en_aug_results, fout);
    print_mc_1d_method_table("TIENET_AUG",    labels, ien_results,    fout);

    write(std::string(78, '=') + "\n\n");

    if (!file_stem.empty()) {
        if (txt_file.is_open()) {
            txt_file.close();
            std::cout << "[Info] TXT saved to: " << dir + out_stem(file_stem) + ".txt\n";
        }
        // CSV
        std::ofstream csv(dir + out_stem(file_stem) + ".csv");
        if (csv.is_open()) {
            csv << "method,metric,rho,mean,sd\n"
                << std::fixed << std::setprecision(6);
            for (std::size_t i = 0; i < rho_grid.size(); ++i) {
                const double rho = rho_grid[i];
                csv << "TENET,mean_FDP,"     << rho << "," << en_results[i].mean_fdp     << ","
                    << en_results[i].sd_fdp     << "\n";
                csv << "TENET,mean_TPP,"     << rho << "," << en_results[i].mean_tpp
                    << "," << en_results[i].sd_tpp     << "\n";
                csv << "TENET_AUG,mean_FDP," << rho << "," << en_aug_results[i].mean_fdp << ","
                    << en_aug_results[i].sd_fdp << "\n";
                csv << "TENET_AUG,mean_TPP," << rho << "," << en_aug_results[i].mean_tpp << ","
                    << en_aug_results[i].sd_tpp << "\n";
                csv << "TIENET_AUG,mean_FDP,"    << rho << "," << ien_results[i].mean_fdp    << ","
                    << ien_results[i].sd_fdp    << "\n";
                csv << "TIENET_AUG,mean_TPP,"    << rho << "," << ien_results[i].mean_tpp    << ","
                    << ien_results[i].sd_tpp    << "\n";
            }
            std::cout << "[Info] CSV saved to: " << dir + out_stem(file_stem) + ".csv\n\n";
        }
    }
}



// ==============================================================================
// 2-D (SNR × rho) sweep
// ==============================================================================

/**
 * @brief Type alias for a 2-D DGP factory: (snr, rho, seed) → GVSDGPResult.
 */
using GVS2DDGPFactory =
    std::function<GVSDGPResult(double snr, double rho, unsigned seed)>;


/**
 * @brief Run a nested SNR × rho grid MC sweep for one method variant.
 *
 * @param dgp_fn      2-D DGP factory — must be thread-safe.
 * @param snr_grid    Outer SNR axis.
 * @param rho_grid    Inner rho axis.
 * @param cfg         Simulation config (num_MC, base_seed, tFDR, …).
 * @param gvs_ctrl    GVS control parameters.
 * @param trex_ctrl   Base T-Rex control parameters.
 * @param label       Short label for progress messages.
 *
 * @return results[i_snr][i_rho] — GVSGridPointResult for each cell.
 */
inline std::vector<std::vector<GVSGridPointResult>> run_gvs_2d_sweep(
    const GVS2DDGPFactory&         dgp_fn,
    const std::vector<double>&     snr_grid,
    const std::vector<double>&     rho_grid,
    const GVSSimConfig&            cfg,
    const TRexGVSControlParameter& gvs_ctrl,
    const TRexControlParameter&    trex_ctrl,
    const std::string&             label)
{
    const std::size_t n_snr = snr_grid.size();
    const std::size_t n_rho = rho_grid.size();
    const int         iMC   = static_cast<int>(cfg.num_MC);
    const unsigned base_seed = static_cast<unsigned>(cfg.base_seed);

    std::vector<std::vector<GVSGridPointResult>> results(
        n_snr, std::vector<GVSGridPointResult>(n_rho));

    if (!method_selected(label)) {
        std::cout << "\n  Method: " << label << "  (skipped, not in TREX_SOLVERS)\n\n";
        for (auto& row : results) std::fill(row.begin(), row.end(), skipped_point());
        return results;
    }

    std::cout << "\n  Method: " << label << "\n\n";

    for (std::size_t i_snr = 0; i_snr < n_snr; ++i_snr) {
        for (std::size_t i_rho = 0; i_rho < n_rho; ++i_rho) {
            const double snr = snr_grid[i_snr];
            const double rho = rho_grid[i_rho];

            std::ostringstream lbl;
            lbl << "[" << (i_snr * n_rho + i_rho + 1) << "/"
                << (n_snr * n_rho) << "] " << label
                << "  SNR=" << std::fixed << std::setprecision(2) << snr
                << "  rho=" << rho;

            // One cell: computed unless a restarted run finds it in the cache.
            const auto compute_cell = [&]() -> trex_cell_cache::Cell {
                std::cout << "  " << lbl.str()
                          << " — Running " << cfg.num_MC << " MC trials ...\n"
                          << std::flush;

                std::vector<double> fdp_vec(cfg.num_MC, 0.0);
                std::vector<double> tpp_vec(cfg.num_MC, 0.0);

                #pragma omp parallel for schedule(dynamic)
                for (int mc = 0; mc < iMC; ++mc) {
                    const unsigned trial_seed = base_seed + static_cast<unsigned>(mc);
                    auto dat = dgp_fn(snr, rho, trial_seed);

                    Eigen::Map<Eigen::MatrixXd> X_map(dat.X.data(),
                                                       dat.X.rows(), dat.X.cols());
                    Eigen::Map<Eigen::VectorXd> y_map(dat.y.data(), dat.y.rows());

                    // Derive solver_type from gvs_ctrl (EN→TENET/TENET_AUG, IEN→TIENET_AUG).
                    // Per-trial cv_seed ensures unique fold assignments.
                    // TRexGVSControlParameter nests its own trex_ctrl member.
                    TRexGVSControlParameter trex_gvs_ctrl = gvs_ctrl;
                    trex_gvs_ctrl.trex_ctrl = trex_ctrl;
                    trex_gvs_ctrl.cv_seed = trial_seed;
                    if (trex_gvs_ctrl.gvs_type == GVSType::EN)
                        trex_gvs_ctrl.trex_ctrl.solver_type =
                            (trex_gvs_ctrl.en_solver == ENSolverType::TENET_AUG)
                                ? SolverTypeForTRex::TENET_AUG
                                : SolverTypeForTRex::TENET;
                    else
                        trex_gvs_ctrl.trex_ctrl.solver_type = SolverTypeForTRex::TIENET_AUG;

                    TRexGVSSelector selector(X_map, y_map, cfg.tFDR, trex_gvs_ctrl,
                                             static_cast<int>(trial_seed),
                                             /*verbose=*/false);
                    selector.select();

                    const auto& sel_idx = selector.getSelectedIndices();
                    fdp_vec[static_cast<std::size_t>(mc)] =
                        rates::compute_fdp(sel_idx, dat.true_support);
                    tpp_vec[static_cast<std::size_t>(mc)] =
                        rates::compute_tpp(sel_idx, dat.true_support);
                }

                const double dMC = static_cast<double>(cfg.num_MC);
                double mean_fdp = 0.0, mean_tpp = 0.0;
                for (int mc = 0; mc < iMC; ++mc) {
                    mean_fdp += fdp_vec[static_cast<std::size_t>(mc)];
                    mean_tpp += tpp_vec[static_cast<std::size_t>(mc)];
                }
                mean_fdp /= dMC;
                mean_tpp /= dMC;

                double sd_fdp = 0.0, sd_tpp = 0.0;
                if (cfg.num_MC > 1) {
                    double sq_fdp = 0.0, sq_tpp = 0.0;
                    for (int mc = 0; mc < iMC; ++mc) {
                        const std::size_t idx = static_cast<std::size_t>(mc);
                        sq_fdp += (fdp_vec[idx] - mean_fdp) * (fdp_vec[idx] - mean_fdp);
                        sq_tpp += (tpp_vec[idx] - mean_tpp) * (tpp_vec[idx] - mean_tpp);
                    }
                    sd_fdp = std::sqrt(sq_fdp / (dMC - 1.0));
                    sd_tpp = std::sqrt(sq_tpp / (dMC - 1.0));
                }

                std::cout << "  " << lbl.str()
                          << " — done. TPP=" << std::fixed << std::setprecision(3) << mean_tpp
                          << "  FDP=" << mean_fdp << "\n\n" << std::flush;
                return cell_detail::to_cell({mean_fdp, mean_tpp, sd_fdp, sd_tpp});
            };
            results[i_snr][i_rho] = cell_detail::from_cell(TREX_CELL_CACHED(
                cell_detail::key("gvs2d", lbl.str(), cfg.num_MC, base_seed, cfg.tFDR, gvs_ctrl),
                compute_cell));
        }
    }

    return results;
}


// ==============================================================================
// Print MC 2-D matrix
// ==============================================================================

/**
 * @brief Strip a common "axis=value" prefix from grid labels.
 *
 * Rewrites each label in place to its bare value (e.g. "snr=0.2" -> "0.2") and
 * returns the axis name ("snr"). Labels without a '=' are left untouched and
 * do not contribute an axis name. Used so 2-D tables can name each axis once in
 * a corner cell instead of repeating "rho=" in every column header.
 */
inline std::string strip_axis_prefix(std::vector<std::string>& labels) {
    std::string axis;
    for (auto& lbl : labels) {
        const auto pos = lbl.find('=');
        if (pos != std::string::npos) {
            if (axis.empty()) axis = lbl.substr(0, pos);
            lbl = lbl.substr(pos + 1);
        }
    }
    return axis;
}


/**
 * @brief Render a 2-D sweep matrix to a string.
 *
 * Layout (row axis on the left, column axis named once in the corner):
 *
 *     mean_TPP [TENET]
 *     ----------------------------------------------------------------
 *       snr \ rho      0.30      0.50      0.70      0.90      0.95      0.99
 *     ----------------------------------------------------------------
 *          0.20      0.1355    0.4264    0.6536    0.7940    0.8279    0.8258
 *
 * The redundant per-column "rho=" prefix is dropped; the corner cell
 * "<row_axis> \\ <col_axis>" names both axes once. Column headers are the bare
 * swept values.
 *
 * @param title      Table title string (e.g. "mean_TPP [TENET]").
 * @param row_labels Row labels, typically "snr=<v>" (rewritten internally).
 * @param col_labels Column labels, typically "rho=<v>" (rewritten internally).
 * @param values     Result matrix indexed [row][col].
 * @param print_tpp  If true, render mean_tpp; if false, mean_fdp.
 */
inline std::string render_mc_matrix_str(
    const std::string&                                  title,
    const std::vector<std::string>&                     row_labels,
    const std::vector<std::string>&                     col_labels,
    const std::vector<std::vector<GVSGridPointResult>>& values,
    bool                                                print_tpp)
{
    std::vector<std::string> rvals = row_labels;
    std::vector<std::string> cvals = col_labels;
    const std::string row_axis = strip_axis_prefix(rvals);
    const std::string col_axis = strip_axis_prefix(cvals);

    // Corner cell names both axes once, e.g. "snr \ rho".
    std::string corner;
    if (!row_axis.empty() || !col_axis.empty())
        corner = row_axis + " \\ " + col_axis;

    // Row-label field wide enough for the corner and every bare row value.
    std::size_t rw = corner.size();
    for (const auto& v : rvals) rw = std::max(rw, v.size());
    const int rwi = static_cast<int>(std::max<std::size_t>(rw + 2, 10));
    constexpr int cw = 10;  // value column width (bare header + value both fit)

    const int total = rwi + cw * static_cast<int>(cvals.size());

    std::ostringstream oss;
    oss << "\n  " << title << "\n"
        << "  " << std::string(total, '-') << "\n";

    // Header row: corner label, then bare column values.
    oss << "  " << std::right << std::setw(rwi) << corner;
    for (const auto& c : cvals)
        oss << std::right << std::setw(cw) << c;
    oss << "\n"
        << "  " << std::string(total, '-') << "\n";

    // Data rows: bare row value, then metric values.
    oss << std::fixed << std::setprecision(4);
    for (std::size_t r = 0; r < values.size() && r < rvals.size(); ++r) {
        oss << "  " << std::right << std::setw(rwi) << rvals[r];
        for (std::size_t c = 0; c < values[r].size() && c < cvals.size(); ++c) {
            const double val = print_tpp ? values[r][c].mean_tpp
                                         : values[r][c].mean_fdp;
            oss << std::right << std::setw(cw) << val;
        }
        oss << "\n";
    }
    oss << "\n";
    return oss.str();
}


/**
 * @brief Print a named grid table from a 2-D sweep result to std::cout.
 *
 * @param title      Table title string.
 * @param row_labels Labels for each row (SNR levels).
 * @param col_labels Labels for each column (rho levels).
 * @param values     Result matrix indexed [row][col].
 * @param print_tpp  If true, print mean_tpp; if false, print mean_fdp.
 */
inline void print_mc_matrix(
    const std::string&                                  title,
    const std::vector<std::string>&                     row_labels,
    const std::vector<std::string>&                     col_labels,
    const std::vector<std::vector<GVSGridPointResult>>& values,
    bool                                                print_tpp)
{
    std::cout << render_mc_matrix_str(title, row_labels, col_labels,
                                      values, print_tpp);
}


// ==============================================================================
// Print MC generic parameter sweep table
// ==============================================================================

/**
 * @brief Print two per-method tables for a generic 1-D parameter sweep.
 *
 * Generalises print_mc_rho_table() by accepting a custom parameter name for
 * the row-label prefix (e.g. "ar_coef", "rho_sc").  Row labels are formed as
 * param_name + "=" + fmt_num(v).
 * If file_stem is non-empty, also writes a .txt and a .csv to DEMO_OUTPUT_DIR.
 *
 * @param scenario_tag  Short label printed in the header.
 * @param param_name    Label prefix for each row (max ~7 chars fits rw=12).
 * @param param_grid    The swept parameter values.
 * @param en_results      TENET results per parameter point.
 * @param en_aug_results  TENET_AUG results per parameter point.
 * @param ien_results     TIENET_AUG results per parameter point.
 * @param cfg             Simulation config (num_MC, tFDR, snr, n, p, …).
 * @param file_stem       Output file stem (no folder, no extension). Empty = no file output.
 */
inline void print_mc_param_sweep_table(
    const std::string&                     scenario_tag,
    const std::string&                     param_name,
    const std::vector<double>&             param_grid,
    const std::vector<GVSGridPointResult>& en_results,
    const std::vector<GVSGridPointResult>& en_aug_results,
    const std::vector<GVSGridPointResult>& ien_results,
    const GVSSimConfig&                    cfg,
    const std::string&                     file_stem = "")
{
    const std::string dir(DEMO_OUTPUT_DIR);
    std::ofstream txt_file;
    if (!file_stem.empty()) {
        std::filesystem::create_directories(dir);
        txt_file.open(dir + out_stem(file_stem) + ".txt");
    }
    std::ofstream* fout = txt_file.is_open() ? &txt_file : nullptr;

    auto write = [&](const std::string& s) {
        std::cout << s;
        if (fout) *fout << s;
    };

    {
        std::ostringstream hdr;
        hdr << "\n" << std::string(78, '=') << "\n"
            << "  T-Rex+GVS MC: " << scenario_tag
            << "  (MC=" << cfg.num_MC << ", tFDR=" << cfg.tFDR
            << ", SNR=" << cfg.snr
            << ", n=" << cfg.n << ", p=" << cfg.p << ")\n"
            << std::string(78, '=') << "\n";
        write(hdr.str());
    }

    std::vector<std::string> labels;
    labels.reserve(param_grid.size());
    for (double v : param_grid)
        labels.push_back(param_name + "=" + fmt_num(v));

    print_mc_1d_method_table("TENET",     labels, en_results,     fout);
    print_mc_1d_method_table("TENET_AUG", labels, en_aug_results, fout);
    print_mc_1d_method_table("TIENET_AUG",    labels, ien_results,    fout);

    write(std::string(78, '=') + "\n\n");

    if (!file_stem.empty()) {
        if (txt_file.is_open()) {
            txt_file.close();
            std::cout << "[Info] TXT saved to: " << dir + out_stem(file_stem) + ".txt\n";
        }
        // CSV
        std::ofstream csv(dir + out_stem(file_stem) + ".csv");
        if (csv.is_open()) {
            csv << "method,metric," << param_name << ",mean,sd\n"
                << std::fixed << std::setprecision(6);
            for (std::size_t i = 0; i < param_grid.size(); ++i) {
                const double pv = param_grid[i];
                csv << "TENET,mean_FDP,"     << pv << "," << en_results[i].mean_fdp     << "," << en_results[i].sd_fdp     << "\n";
                csv << "TENET,mean_TPP,"     << pv << "," << en_results[i].mean_tpp     << "," << en_results[i].sd_tpp     << "\n";
                csv << "TENET_AUG,mean_FDP," << pv << "," << en_aug_results[i].mean_fdp << "," << en_aug_results[i].sd_fdp << "\n";
                csv << "TENET_AUG,mean_TPP," << pv << "," << en_aug_results[i].mean_tpp << "," << en_aug_results[i].sd_tpp << "\n";
                csv << "TIENET_AUG,mean_FDP,"    << pv << "," << ien_results[i].mean_fdp    << "," << ien_results[i].sd_fdp    << "\n";
                csv << "TIENET_AUG,mean_TPP,"    << pv << "," << ien_results[i].mean_tpp    << "," << ien_results[i].sd_tpp    << "\n";
            }
            std::cout << "[Info] CSV saved to: " << dir + out_stem(file_stem) + ".csv\n\n";
        }
    }
}

// ==============================================================================
// Save MC 2-D tables (console + optional file/CSV)
// ==============================================================================

/**
 * @brief Save 2-D sweep results to a .txt and .csv file in DEMO_OUTPUT_DIR.
 *
 * This is a pure file writer — the four console matrix prints should be
 * done separately via `print_mc_matrix()` (as in all Part 3 sim functions).
 * If file_stem is empty this function is a no-op.
 *
 * @param scenario_tag  Short label used in the file header.
 * @param row_labels    Labels for the row axis (typically SNR levels).
 * @param col_labels    Labels for the column axis (rho / ar_coef / rho_sc …).
 * @param en_2d         EN (TENET) result matrix indexed [row][col].
 * @param en_aug_2d     EN (TENET_AUG) result matrix indexed [row][col].
 * @param ien_2d        IEN result matrix indexed [row][col].
 * @param cfg           Simulation config (num_MC, tFDR, n, p, …).
 * @param file_stem     Output file stem (no folder, no extension). Empty = no-op.
 */
inline void save_mc_2d_tables(
    const std::string&                                  scenario_tag,
    const std::vector<std::string>&                     row_labels,
    const std::vector<std::string>&                     col_labels,
    const std::vector<std::vector<GVSGridPointResult>>& en_2d,
    const std::vector<std::vector<GVSGridPointResult>>& en_aug_2d,
    const std::vector<std::vector<GVSGridPointResult>>& ien_2d,
    const GVSSimConfig&                                 cfg,
    const std::string&                                  file_stem = "")
{
    if (file_stem.empty()) return;

    const std::string dir(DEMO_OUTPUT_DIR);
    std::filesystem::create_directories(dir);

    // TXT: write all four matrices
    {
        std::ofstream txt(dir + out_stem(file_stem) + ".txt");
        if (txt.is_open()) {
            txt << "\n" << std::string(78, '=') << "\n"
                << "  T-Rex+GVS MC 2D Grid: " << scenario_tag
                << "  (MC=" << cfg.num_MC << ", tFDR=" << cfg.tFDR
                << ", n=" << cfg.n << ", p=" << cfg.p << ")\n"
                << std::string(78, '=') << "\n";

            auto write_matrix = [&](const std::string& title,
                                    const std::vector<std::vector<GVSGridPointResult>>& vals,
                                    bool tpp) {
                txt << render_mc_matrix_str(title, row_labels, col_labels,
                                            vals, tpp);
            };

            write_matrix("mean_TPP [TENET]",     en_2d,     true);
            write_matrix("mean_FDP [TENET]",     en_2d,     false);
            write_matrix("mean_TPP [TENET_AUG]", en_aug_2d, true);
            write_matrix("mean_FDP [TENET_AUG]", en_aug_2d, false);
            write_matrix("mean_TPP [TIENET_AUG]",    ien_2d,    true);
            write_matrix("mean_FDP [TIENET_AUG]",    ien_2d,    false);

            std::cout << "[Info] TXT saved to: " << dir + out_stem(file_stem) + ".txt\n";
        }
    }

    // CSV: tidy long format — method, metric, row_label, col_label, mean, sd
    {
        std::ofstream csv(dir + out_stem(file_stem) + ".csv");
        if (csv.is_open()) {
            csv << "method,metric,row_label,col_label,mean,sd\n"
                << std::fixed << std::setprecision(6);
            for (std::size_t i = 0; i < en_2d.size() && i < row_labels.size(); ++i) {
                for (std::size_t j = 0; j < en_2d[i].size() && j < col_labels.size(); ++j) {
                    csv << "TENET,mean_FDP,"     << row_labels[i] << "," << col_labels[j]
                        << "," << en_2d[i][j].mean_fdp     << "," << en_2d[i][j].sd_fdp     << "\n";
                    csv << "TENET,mean_TPP,"     << row_labels[i] << "," << col_labels[j]
                        << "," << en_2d[i][j].mean_tpp     << "," << en_2d[i][j].sd_tpp     << "\n";
                    csv << "TENET_AUG,mean_FDP," << row_labels[i] << "," << col_labels[j]
                        << "," << en_aug_2d[i][j].mean_fdp << "," << en_aug_2d[i][j].sd_fdp << "\n";
                    csv << "TENET_AUG,mean_TPP," << row_labels[i] << "," << col_labels[j]
                        << "," << en_aug_2d[i][j].mean_tpp << "," << en_aug_2d[i][j].sd_tpp << "\n";
                    csv << "TIENET_AUG,mean_FDP,"    << row_labels[i] << "," << col_labels[j]
                        << "," << ien_2d[i][j].mean_fdp    << "," << ien_2d[i][j].sd_fdp    << "\n";
                    csv << "TIENET_AUG,mean_TPP,"    << row_labels[i] << "," << col_labels[j]
                        << "," << ien_2d[i][j].mean_tpp    << "," << ien_2d[i][j].sd_tpp    << "\n";
                }
            }
            std::cout << "[Info] CSV saved to: " << dir + out_stem(file_stem) + ".csv\n\n";
        }
    }
}


// ==============================================================================


// ==============================================================================
// Block-equicorrelated benchmark utilities
// (consolidated from trex_gvs_block_bench_utils.hpp)
// ==============================================================================


// ==============================================================================
// Simulation configuration
// ==============================================================================

/** @brief Parameter bundle for the block-equicorrelated MC benchmark. */
struct BlockBenchConfig {
    int    n               = 200;  ///< Observations.
    int    p               = 500;  ///< Predictors.
    int    G               = 100;  ///< Number of blocks.
    int    block_size      = 5;    ///< Variables per block.
    int    n_active_blocks = 10;   ///< Active blocks per trial.
    double tFDR            = 0.1;  ///< Target FDR level.
    std::size_t K          = 20;   ///< Random experiments per selector run.
    std::size_t num_MC     = 500;  ///< Monte Carlo trials per grid cell.
    int    base_seed       = 2026; ///< Seed for trial 0; trial i uses base_seed + i.
    double corr_max        = 0.5;  ///< corr_max for HAC auto-clustering.
    double b               = 3.0;  ///< Nonzero coefficient magnitude.
    ENSolverType en_solver = ENSolverType::TENET_AUG; ///< EN solver variant (only applies to GVSType::EN).
};


// ==============================================================================
// Per-method result (one trial)
// ==============================================================================

/**
 * @brief All metrics collected for one method on one MC trial.
 *
 * Coordinate metrics (evaluated against the true variable support):
 *   coord_fdp    = |S_hat \ S_true| / max(|S_hat|, 1)
 *   coord_tpr    = |S_hat ∩ S_true| / |S_true|
 *   exact_support = 1( S_hat == S_true )
 *
 * Block metrics (evaluated against the active block set):
 *   block_hit_rate        = (active blocks with ≥1 selected var) / n_active_blocks
 *   block_fdp             = (null blocks hit) / max(total blocks hit, 1)
 *   full_block_rate       = (active blocks fully in S_hat) / n_active_blocks
 *   null_block_activation = (null blocks with ≥1 selected var) / null_block_count
 *
 * Group-recovery diagnostic:
 *   block_purity_rate = (true blocks where all members share same groups_vec value)
 *                       / G
 *   (= 1.0 by definition for oracle runs)
 *
 * Selector diagnostics:
 *   lambda2_used, T_stop, M_found, elapsed_sec
 */
struct BlockMethodResult {
    // Coordinate-level metrics
    double coord_fdp     = 0.0;
    double coord_tpr     = 0.0;
    double exact_support = 0.0;

    // Block-level metrics
    double block_hit_rate        = 0.0;
    double block_fdp             = 0.0;
    double full_block_rate       = 0.0;
    double null_block_activation = 0.0;

    // Group-recovery diagnostic
    double block_purity_rate = 0.0;

    // Selector diagnostics
    double      lambda2_used = 0.0;
    std::size_t T_stop       = 0;
    std::size_t M_found      = 0;
    double      elapsed_sec  = 0.0;
};


// ==============================================================================
// Trial result (all 4 methods)
// ==============================================================================

/** @brief Results for all four method variants on a single MC trial. */
struct BlockTrialResult {
    BlockMethodResult M1;  ///< EN  + HAC-clustered groups
    BlockMethodResult M2;  ///< EN  + oracle groups
    BlockMethodResult M3;  ///< IEN + HAC-clustered groups
    BlockMethodResult M4;  ///< IEN + oracle groups
};


// ==============================================================================
// MC aggregate result
// ==============================================================================

/** @brief Aggregated statistics for one method over num_MC trials at one grid cell. */
struct BlockMethodAggregate {
    // Coordinate-level metrics
    double mean_coord_fdp = 0.0;   double sd_coord_fdp = 0.0;   double mcse_coord_fdp = 0.0;
    double mean_coord_tpr = 0.0;   double sd_coord_tpr = 0.0;   double mcse_coord_tpr = 0.0;
    double mean_exact_sup = 0.0;   double sd_exact_sup = 0.0;

    // Block-level metrics
    double mean_block_hit  = 0.0;  double sd_block_hit  = 0.0;
    double mean_block_fdp  = 0.0;  double sd_block_fdp  = 0.0;
    double mean_full_block = 0.0;  double sd_full_block = 0.0;
    double mean_null_act   = 0.0;  double sd_null_act   = 0.0;

    // Group-recovery diagnostic
    double mean_purity = 0.0;      double sd_purity = 0.0;

    // Selector diagnostics (means only)
    double      mean_lambda2 = 0.0;
    double      mean_T_stop  = 0.0;
    double      mean_M_found = 0.0;
    double      mean_elapsed = 0.0;
};

/** @brief All four method aggregates for one grid cell. */
struct BlockGridResult {
    BlockMethodAggregate M1;
    BlockMethodAggregate M2;
    BlockMethodAggregate M3;
    BlockMethodAggregate M4;
};


// ==============================================================================
// Block metric helpers
// ==============================================================================

/**
 * @brief Fraction of active blocks that contain at least one selected variable.
 *
 * @param sel      Selected variable indices (0-based).
 * @param active   Active block indices (0-based block IDs).
 * @param block_size  Variables per block.
 * @return block_hit_rate in [0, 1].
 */
inline double compute_block_hit_rate(
    const std::vector<std::size_t>& sel,
    const std::vector<int>&         active,
    int                             block_size)
{
    if (active.empty()) return 0.0;
    const std::unordered_set<std::size_t> sel_set(sel.begin(), sel.end());
    int hits = 0;
    for (int blk : active) {
        const int base = blk * block_size;
        for (int jj = 0; jj < block_size; ++jj) {
            if (sel_set.count(static_cast<std::size_t>(base + jj))) {
                ++hits;
                break;
            }
        }
    }
    return static_cast<double>(hits) / static_cast<double>(active.size());
}


/**
 * @brief False discovery proportion at the block level.
 *
 * block_fdp = (# null blocks with ≥1 selected var) / max(# total blocks hit, 1)
 *
 * @param sel         Selected variable indices (0-based).
 * @param active      Active block indices.
 * @param G           Total number of blocks.
 * @param block_size  Variables per block.
 * @return block_fdp in [0, 1].
 */
inline double compute_block_fdp(
    const std::vector<std::size_t>& sel,
    const std::vector<int>&         active,
    int                             G,
    int                             block_size)
{
    const std::unordered_set<std::size_t> sel_set(sel.begin(), sel.end());
    const std::unordered_set<int>         active_set(active.begin(), active.end());

    int total_hit = 0, null_hit = 0;
    for (int blk = 0; blk < G; ++blk) {
        const int base = blk * block_size;
        bool hit = false;
        for (int jj = 0; jj < block_size; ++jj) {
            if (sel_set.count(static_cast<std::size_t>(base + jj))) {
                hit = true;
                break;
            }
        }
        if (hit) {
            ++total_hit;
            if (!active_set.count(blk)) ++null_hit;
        }
    }
    return static_cast<double>(null_hit) /
           static_cast<double>(std::max(total_hit, 1));
}


/**
 * @brief Fraction of active blocks that are fully contained in the selection.
 *
 * @param sel         Selected variable indices (0-based).
 * @param active      Active block indices.
 * @param block_size  Variables per block.
 * @return full_block_rate in [0, 1].
 */
inline double compute_full_block_rate(
    const std::vector<std::size_t>& sel,
    const std::vector<int>&         active,
    int                             block_size)
{
    if (active.empty()) return 0.0;
    const std::unordered_set<std::size_t> sel_set(sel.begin(), sel.end());
    int fully_hit = 0;
    for (int blk : active) {
        const int base = blk * block_size;
        bool full = true;
        for (int jj = 0; jj < block_size; ++jj) {
            if (!sel_set.count(static_cast<std::size_t>(base + jj))) {
                full = false;
                break;
            }
        }
        if (full) ++fully_hit;
    }
    return static_cast<double>(fully_hit) / static_cast<double>(active.size());
}


/**
 * @brief Fraction of null blocks that contain at least one selected variable.
 *
 * @param sel         Selected variable indices (0-based).
 * @param active      Active block indices.
 * @param G           Total number of blocks.
 * @param block_size  Variables per block.
 * @return null_block_activation in [0, 1].
 */
inline double compute_null_block_activation(
    const std::vector<std::size_t>& sel,
    const std::vector<int>&         active,
    int                             G,
    int                             block_size)
{
    const std::unordered_set<std::size_t> sel_set(sel.begin(), sel.end());
    const std::unordered_set<int>         active_set(active.begin(), active.end());

    int null_total = 0, null_hit = 0;
    for (int blk = 0; blk < G; ++blk) {
        if (active_set.count(blk)) continue;
        ++null_total;
        const int base = blk * block_size;
        for (int jj = 0; jj < block_size; ++jj) {
            if (sel_set.count(static_cast<std::size_t>(base + jj))) {
                ++null_hit;
                break;
            }
        }
    }
    return (null_total == 0) ? 0.0
         : static_cast<double>(null_hit) / static_cast<double>(null_total);
}


/**
 * @brief Fraction of true blocks whose members all share the same groups_vec value.
 *
 * A block is "pure" if groups_vec(base) == groups_vec(base+1) == … == groups_vec(base+block_size-1).
 * This is always 1.0 for oracle runs (prior_groups supplied).
 *
 * @param groups_vec  Cluster assignment per variable (length p, from GVSSelectionResult).
 * @param G           Total number of blocks.
 * @param block_size  Variables per block.
 * @return block_purity_rate in [0, 1].
 */
inline double compute_block_purity_rate(
    const Eigen::VectorXi& groups_vec,
    int                    G,
    int                    block_size)
{
    int pure = 0;
    for (int k = 0; k < G; ++k) {
        const int base  = k * block_size;
        const int first = groups_vec(base);
        bool is_pure    = true;
        for (int jj = 1; jj < block_size; ++jj) {
            if (groups_vec(base + jj) != first) { is_pure = false; break; }
        }
        if (is_pure) ++pure;
    }
    return static_cast<double>(pure) / static_cast<double>(G);
}


/**
 * @brief Exact support recovery: 1.0 if S_hat == S_true, else 0.0.
 *
 * @param sel        Selected variable indices (0-based, need not be sorted).
 * @param true_sup   True support (sorted, 0-based).
 * @return 1.0 or 0.0.
 */
inline double compute_exact_support(
    const std::vector<std::size_t>& sel,
    const std::vector<std::size_t>& true_sup)
{
    if (sel.size() != true_sup.size()) return 0.0;
    std::vector<std::size_t> sorted_sel = sel;
    std::sort(sorted_sel.begin(), sorted_sel.end());
    return (sorted_sel == true_sup) ? 1.0 : 0.0;
}


// ==============================================================================
// Single-method run helper
// ==============================================================================

/**
 * @brief Run TRexGVSSelector once and collect all BlockMethodResult metrics.
 *
 * @param X_map       Non-owning map over the design matrix (n × p).
 * @param y_map       Non-owning map over the response (n).
 * @param dat         Full DGP result (true_support, active_blocks, prior_groups).
 * @param G           Total number of blocks.
 * @param block_size  Variables per block.
 * @param tFDR        Target FDR level.
 * @param gvs_ctrl    GVS control parameters (gvs_type and prior_groups set by caller).
 * @param trex_ctrl   Base T-Rex control parameters.
 * @param seed        RNG seed passed to the selector.
 * @return Populated BlockMethodResult.
 */
inline BlockMethodResult run_block_method(
    Eigen::Map<Eigen::MatrixXd>&   X_map,
    Eigen::Map<Eigen::VectorXd>&   y_map,
    const GVSDGPResult&            dat,
    int                            G,
    int                            block_size,
    double                         tFDR,
    const TRexGVSControlParameter& gvs_ctrl,
    const TRexControlParameter&    trex_ctrl,
    int                            seed)
{
    const auto t0 = std::chrono::steady_clock::now();

    // Derive solver_type from gvs_ctrl (EN→TENET/TENET_AUG, IEN→TIENET_AUG).
    // TRexGVSControlParameter nests its own trex_ctrl member.
    TRexGVSControlParameter trex_gvs_ctrl = gvs_ctrl;
    trex_gvs_ctrl.trex_ctrl = trex_ctrl;
    if (trex_gvs_ctrl.gvs_type == GVSType::EN)
        trex_gvs_ctrl.trex_ctrl.solver_type =
            (trex_gvs_ctrl.en_solver == ENSolverType::TENET_AUG)
                ? SolverTypeForTRex::TENET_AUG
                : SolverTypeForTRex::TENET;
    else
        trex_gvs_ctrl.trex_ctrl.solver_type = SolverTypeForTRex::TIENET_AUG;

    TRexGVSSelector selector(X_map, y_map, tFDR, trex_gvs_ctrl,
                              seed, /*verbose=*/false);
    selector.select();

    const auto t1 = std::chrono::steady_clock::now();

    const auto& sel_idx  = selector.getSelectedIndices();
    const auto& gvs_res  = selector.getGVSResult();

    BlockMethodResult out;
    out.coord_fdp     = rates::compute_fdp(sel_idx, dat.true_support);
    out.coord_tpr     = rates::compute_tpp(sel_idx, dat.true_support);
    out.exact_support = compute_exact_support(sel_idx, dat.true_support);

    out.block_hit_rate        = compute_block_hit_rate(sel_idx, dat.active_blocks,
                                                        block_size);
    out.block_fdp             = compute_block_fdp(sel_idx, dat.active_blocks,
                                                   G, block_size);
    out.full_block_rate       = compute_full_block_rate(sel_idx, dat.active_blocks,
                                                         block_size);
    out.null_block_activation = compute_null_block_activation(sel_idx, dat.active_blocks,
                                                               G, block_size);
    out.block_purity_rate     = compute_block_purity_rate(gvs_res.groups_vec,
                                                           G, block_size);

    out.lambda2_used = gvs_res.lambda2_used;
    out.T_stop       = gvs_res.T_stop;
    out.M_found      = gvs_res.max_clusters;
    out.elapsed_sec  =
        std::chrono::duration<double>(t1 - t0).count();

    return out;
}


/** @brief Result of a skipped (unselected) method: all metrics NaN. */
inline BlockMethodResult skipped_block_method() {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    BlockMethodResult r;
    r.coord_fdp = r.coord_tpr = r.exact_support = nan;
    r.block_hit_rate = r.block_fdp = r.full_block_rate = r.null_block_activation = nan;
    r.block_purity_rate = r.lambda2_used = r.elapsed_sec = nan;
    return r;
}


// ==============================================================================
// Single-trial runner (all 4 methods)
// ==============================================================================

/**
 * @brief Run all four method variants on one generated dataset.
 *
 * M1: EN  + HAC-clustered groups (prior_groups empty)
 * M2: EN  + oracle groups        (prior_groups = dat.prior_groups)
 * M3: IEN + HAC-clustered groups (prior_groups empty)
 * M4: IEN + oracle groups        (prior_groups = dat.prior_groups)
 *
 * @param dat        Generated dataset (from make_block_equicorr_dgp).
 * @param G          Total number of blocks.
 * @param block_size Variables per block.
 * @param tFDR       Target FDR level.
 * @param trex_ctrl  Base T-Rex control parameters.
 * @param corr_max   corr_max passed to HAC auto-clustering methods.
 * @param seed       RNG seed passed to the selector.
 * @return BlockTrialResult with results for all four methods.
 */
inline BlockTrialResult run_block_single(
    const GVSDGPResult&         dat,
    int                         G,
    int                         block_size,
    double                      tFDR,
    const TRexControlParameter& trex_ctrl,
    double                      corr_max,
    int                         seed,
    ENSolverType                en_solver = ENSolverType::TENET_AUG)
{
    Eigen::Map<Eigen::MatrixXd> X_map(
        const_cast<double*>(dat.X.data()), dat.X.rows(), dat.X.cols());
    Eigen::Map<Eigen::VectorXd> y_map(
        const_cast<double*>(dat.y.data()), dat.y.rows());

    // --- Base GVS control parameters ---
    TRexGVSControlParameter gvs_en;
    gvs_en.gvs_type       = GVSType::EN;
    gvs_en.lambda2_method = LambdaSelectionMethod::CV_1SE_CCD;
    gvs_en.corr_max       = corr_max;
    gvs_en.hc_linkage     = hac::LinkageMethod::Single;
    gvs_en.en_solver      = en_solver;

    TRexGVSControlParameter gvs_ien;
    gvs_ien.gvs_type       = GVSType::IEN;
    // IEN-geometry tuner: the true support (10 blocks of 5) lies below its
    // default lambda1 support cap (0.8 * n_train); the s = 150 demos 01-07 raise it.
    gvs_ien.lambda2_method = LambdaSelectionMethod::CV_1SE_IEN_CCD;
    gvs_ien.corr_max       = corr_max;
    gvs_ien.hc_linkage     = hac::LinkageMethod::Single;

    // Oracle variants carry the true prior_groups
    TRexGVSControlParameter gvs_en_oracle  = gvs_en;
    TRexGVSControlParameter gvs_ien_oracle = gvs_ien;
    gvs_en_oracle.prior_groups  = dat.prior_groups;
    gvs_ien_oracle.prior_groups = dat.prior_groups;

    // TREX_SOLVERS: EN methods by their EN solver, IEN methods as TIENET_AUG.
    const bool run_en  = trex_solver_filter::selected(
        en_solver == ENSolverType::TENET_AUG ? "TENET_AUG" : "TENET");
    const bool run_ien = trex_solver_filter::selected("TIENET_AUG");
    const auto run = [&](bool on, const TRexGVSControlParameter& ctrl) {
        return on ? run_block_method(X_map, y_map, dat, G, block_size, tFDR,
                                     ctrl, trex_ctrl, seed)
                  : skipped_block_method();
    };

    BlockTrialResult out;
    out.M1 = run(run_en,  gvs_en);
    out.M2 = run(run_en,  gvs_en_oracle);
    out.M3 = run(run_ien, gvs_ien);
    out.M4 = run(run_ien, gvs_ien_oracle);
    return out;
}


// ==============================================================================
// MC aggregation helper
// ==============================================================================

namespace detail {

/** @brief Compute mean and standard deviation of a vector. */
inline void mean_sd(const std::vector<double>& v, double& mean, double& sd)
{
    if (v.empty()) { mean = sd = 0.0; return; }
    const double dN = static_cast<double>(v.size());
    mean = std::accumulate(v.begin(), v.end(), 0.0) / dN;
    if (v.size() < 2) { sd = 0.0; return; }
    double sq = 0.0;
    for (double x : v) sq += (x - mean) * (x - mean);
    sd = std::sqrt(sq / (dN - 1.0));
}

} // namespace detail


// ==============================================================================
// MC loop
// ==============================================================================

namespace cell_detail {

/** @brief Cell of one block-benchmark grid point: M1-M4, 22 numbers each. */
inline trex_cell_cache::Cell to_cell(const BlockGridResult& r) {
    trex_cell_cache::Cell c;
    const BlockMethodAggregate* ms[4] = {&r.M1, &r.M2, &r.M3, &r.M4};
    for (int m = 0; m < 4; ++m) {
        const BlockMethodAggregate& a = *ms[m];
        const std::string p = "M" + std::to_string(m + 1) + ".";
        c.emplace_back(p + "mean_coord_fdp", a.mean_coord_fdp);
        c.emplace_back(p + "sd_coord_fdp", a.sd_coord_fdp);
        c.emplace_back(p + "mcse_coord_fdp", a.mcse_coord_fdp);
        c.emplace_back(p + "mean_coord_tpr", a.mean_coord_tpr);
        c.emplace_back(p + "sd_coord_tpr", a.sd_coord_tpr);
        c.emplace_back(p + "mcse_coord_tpr", a.mcse_coord_tpr);
        c.emplace_back(p + "mean_exact_sup", a.mean_exact_sup);
        c.emplace_back(p + "sd_exact_sup", a.sd_exact_sup);
        c.emplace_back(p + "mean_block_hit", a.mean_block_hit);
        c.emplace_back(p + "sd_block_hit", a.sd_block_hit);
        c.emplace_back(p + "mean_block_fdp", a.mean_block_fdp);
        c.emplace_back(p + "sd_block_fdp", a.sd_block_fdp);
        c.emplace_back(p + "mean_full_block", a.mean_full_block);
        c.emplace_back(p + "sd_full_block", a.sd_full_block);
        c.emplace_back(p + "mean_null_act", a.mean_null_act);
        c.emplace_back(p + "sd_null_act", a.sd_null_act);
        c.emplace_back(p + "mean_purity", a.mean_purity);
        c.emplace_back(p + "sd_purity", a.sd_purity);
        c.emplace_back(p + "mean_lambda2", a.mean_lambda2);
        c.emplace_back(p + "mean_T_stop", a.mean_T_stop);
        c.emplace_back(p + "mean_M_found", a.mean_M_found);
        c.emplace_back(p + "mean_elapsed", a.mean_elapsed);
    }
    return c;
}

inline BlockGridResult block_from_cell(const trex_cell_cache::Cell& c) {
    using trex_cell_cache::get;
    BlockGridResult r;
    BlockMethodAggregate* ms[4] = {&r.M1, &r.M2, &r.M3, &r.M4};
    for (int m = 0; m < 4; ++m) {
        BlockMethodAggregate& a = *ms[m];
        const std::string p = "M" + std::to_string(m + 1) + ".";
        a.mean_coord_fdp = get(c, p + "mean_coord_fdp");
        a.sd_coord_fdp = get(c, p + "sd_coord_fdp");
        a.mcse_coord_fdp = get(c, p + "mcse_coord_fdp");
        a.mean_coord_tpr = get(c, p + "mean_coord_tpr");
        a.sd_coord_tpr = get(c, p + "sd_coord_tpr");
        a.mcse_coord_tpr = get(c, p + "mcse_coord_tpr");
        a.mean_exact_sup = get(c, p + "mean_exact_sup");
        a.sd_exact_sup = get(c, p + "sd_exact_sup");
        a.mean_block_hit = get(c, p + "mean_block_hit");
        a.sd_block_hit = get(c, p + "sd_block_hit");
        a.mean_block_fdp = get(c, p + "mean_block_fdp");
        a.sd_block_fdp = get(c, p + "sd_block_fdp");
        a.mean_full_block = get(c, p + "mean_full_block");
        a.sd_full_block = get(c, p + "sd_full_block");
        a.mean_null_act = get(c, p + "mean_null_act");
        a.sd_null_act = get(c, p + "sd_null_act");
        a.mean_purity = get(c, p + "mean_purity");
        a.sd_purity = get(c, p + "sd_purity");
        a.mean_lambda2 = get(c, p + "mean_lambda2");
        a.mean_T_stop = get(c, p + "mean_T_stop");
        a.mean_M_found = get(c, p + "mean_M_found");
        a.mean_elapsed = get(c, p + "mean_elapsed");
    }
    return r;
}

}  // namespace cell_detail


/** @brief Uncached body of run_block_mc(). */
inline BlockGridResult run_block_mc_impl(
    const std::function<GVSDGPResult(unsigned)>& dgp_fn,
    const BlockBenchConfig&                      cfg,
    const TRexControlParameter&                  trex_ctrl,
    const std::string&                           label)
{
    const int iMC = static_cast<int>(cfg.num_MC);
    const unsigned base_seed = static_cast<unsigned>(cfg.base_seed);

    // Accumulators for all metrics, all 4 methods
    std::vector<double>
        m1_cfdp(cfg.num_MC), m1_ctpr(cfg.num_MC), m1_esup(cfg.num_MC),
        m1_bhr (cfg.num_MC), m1_bfdp(cfg.num_MC),
        m1_fbr (cfg.num_MC), m1_nba (cfg.num_MC), m1_pur (cfg.num_MC),

        m2_cfdp(cfg.num_MC), m2_ctpr(cfg.num_MC), m2_esup(cfg.num_MC),
        m2_bhr (cfg.num_MC), m2_bfdp(cfg.num_MC),
        m2_fbr (cfg.num_MC), m2_nba (cfg.num_MC), m2_pur (cfg.num_MC),

        m3_cfdp(cfg.num_MC), m3_ctpr(cfg.num_MC), m3_esup(cfg.num_MC),
        m3_bhr (cfg.num_MC), m3_bfdp(cfg.num_MC),
        m3_fbr (cfg.num_MC), m3_nba (cfg.num_MC), m3_pur (cfg.num_MC),

        m4_cfdp(cfg.num_MC), m4_ctpr(cfg.num_MC), m4_esup(cfg.num_MC),
        m4_bhr (cfg.num_MC), m4_bfdp(cfg.num_MC),
        m4_fbr (cfg.num_MC), m4_nba (cfg.num_MC), m4_pur (cfg.num_MC),

        m1_lam(cfg.num_MC), m1_ts(cfg.num_MC), m1_mf(cfg.num_MC), m1_el(cfg.num_MC),
        m2_lam(cfg.num_MC), m2_ts(cfg.num_MC), m2_mf(cfg.num_MC), m2_el(cfg.num_MC),
        m3_lam(cfg.num_MC), m3_ts(cfg.num_MC), m3_mf(cfg.num_MC), m3_el(cfg.num_MC),
        m4_lam(cfg.num_MC), m4_ts(cfg.num_MC), m4_mf(cfg.num_MC), m4_el(cfg.num_MC);

    std::cout << "  " << label << " — running " << cfg.num_MC
              << " MC trials...\n" << std::flush;

    #pragma omp parallel for schedule(dynamic)
    for (int mc = 0; mc < iMC; ++mc) {
        const unsigned trial_seed = base_seed + static_cast<unsigned>(mc);
        auto dat = dgp_fn(trial_seed);

        const auto res = run_block_single(dat, cfg.G, cfg.block_size,
                                          cfg.tFDR, trex_ctrl,
                                          cfg.corr_max,
                                          static_cast<int>(trial_seed),
                                          cfg.en_solver);
        const std::size_t idx = static_cast<std::size_t>(mc);

        m1_cfdp[idx] = res.M1.coord_fdp;    m1_ctpr[idx] = res.M1.coord_tpr;
        m1_esup[idx] = res.M1.exact_support;
        m1_bhr[idx]  = res.M1.block_hit_rate;  m1_bfdp[idx] = res.M1.block_fdp;
        m1_fbr[idx]  = res.M1.full_block_rate; m1_nba[idx]  = res.M1.null_block_activation;
        m1_pur[idx]  = res.M1.block_purity_rate;
        m1_lam[idx]  = res.M1.lambda2_used;
        m1_ts[idx]   = static_cast<double>(res.M1.T_stop);
        m1_mf[idx]   = static_cast<double>(res.M1.M_found);
        m1_el[idx]   = res.M1.elapsed_sec;

        m2_cfdp[idx] = res.M2.coord_fdp;    m2_ctpr[idx] = res.M2.coord_tpr;
        m2_esup[idx] = res.M2.exact_support;
        m2_bhr[idx]  = res.M2.block_hit_rate;  m2_bfdp[idx] = res.M2.block_fdp;
        m2_fbr[idx]  = res.M2.full_block_rate; m2_nba[idx]  = res.M2.null_block_activation;
        m2_pur[idx]  = res.M2.block_purity_rate;
        m2_lam[idx]  = res.M2.lambda2_used;
        m2_ts[idx]   = static_cast<double>(res.M2.T_stop);
        m2_mf[idx]   = static_cast<double>(res.M2.M_found);
        m2_el[idx]   = res.M2.elapsed_sec;

        m3_cfdp[idx] = res.M3.coord_fdp;    m3_ctpr[idx] = res.M3.coord_tpr;
        m3_esup[idx] = res.M3.exact_support;
        m3_bhr[idx]  = res.M3.block_hit_rate;  m3_bfdp[idx] = res.M3.block_fdp;
        m3_fbr[idx]  = res.M3.full_block_rate; m3_nba[idx]  = res.M3.null_block_activation;
        m3_pur[idx]  = res.M3.block_purity_rate;
        m3_lam[idx]  = res.M3.lambda2_used;
        m3_ts[idx]   = static_cast<double>(res.M3.T_stop);
        m3_mf[idx]   = static_cast<double>(res.M3.M_found);
        m3_el[idx]   = res.M3.elapsed_sec;

        m4_cfdp[idx] = res.M4.coord_fdp;    m4_ctpr[idx] = res.M4.coord_tpr;
        m4_esup[idx] = res.M4.exact_support;
        m4_bhr[idx]  = res.M4.block_hit_rate;  m4_bfdp[idx] = res.M4.block_fdp;
        m4_fbr[idx]  = res.M4.full_block_rate; m4_nba[idx]  = res.M4.null_block_activation;
        m4_pur[idx]  = res.M4.block_purity_rate;
        m4_lam[idx]  = res.M4.lambda2_used;
        m4_ts[idx]   = static_cast<double>(res.M4.T_stop);
        m4_mf[idx]   = static_cast<double>(res.M4.M_found);
        m4_el[idx]   = res.M4.elapsed_sec;
    }

    const double dMC = static_cast<double>(cfg.num_MC);

    auto build_agg = [&](
        const std::vector<double>& cfdp, const std::vector<double>& ctpr,
        const std::vector<double>& esup,
        const std::vector<double>& bhr,  const std::vector<double>& bfdp,
        const std::vector<double>& fbr,  const std::vector<double>& nba,
        const std::vector<double>& pur,
        const std::vector<double>& lam,  const std::vector<double>& ts,
        const std::vector<double>& mf,   const std::vector<double>& el)
    {
        BlockMethodAggregate a;
        detail::mean_sd(cfdp, a.mean_coord_fdp, a.sd_coord_fdp);
        detail::mean_sd(ctpr, a.mean_coord_tpr, a.sd_coord_tpr);
        detail::mean_sd(esup, a.mean_exact_sup, a.sd_exact_sup);
        detail::mean_sd(bhr,  a.mean_block_hit, a.sd_block_hit);
        detail::mean_sd(bfdp, a.mean_block_fdp, a.sd_block_fdp);
        detail::mean_sd(fbr,  a.mean_full_block, a.sd_full_block);
        detail::mean_sd(nba,  a.mean_null_act,  a.sd_null_act);
        detail::mean_sd(pur,  a.mean_purity,    a.sd_purity);
        a.mcse_coord_fdp = (cfg.num_MC > 1) ? a.sd_coord_fdp / std::sqrt(dMC) : 0.0;
        a.mcse_coord_tpr = (cfg.num_MC > 1) ? a.sd_coord_tpr / std::sqrt(dMC) : 0.0;
        a.mean_lambda2 = std::accumulate(lam.begin(), lam.end(), 0.0) / dMC;
        a.mean_T_stop  = std::accumulate(ts.begin(),  ts.end(),  0.0) / dMC;
        a.mean_M_found = std::accumulate(mf.begin(),  mf.end(),  0.0) / dMC;
        a.mean_elapsed = std::accumulate(el.begin(),  el.end(),  0.0) / dMC;
        return a;
    };

    BlockGridResult out;
    out.M1 = build_agg(m1_cfdp, m1_ctpr, m1_esup, m1_bhr, m1_bfdp,
                        m1_fbr, m1_nba, m1_pur, m1_lam, m1_ts, m1_mf, m1_el);
    out.M2 = build_agg(m2_cfdp, m2_ctpr, m2_esup, m2_bhr, m2_bfdp,
                        m2_fbr, m2_nba, m2_pur, m2_lam, m2_ts, m2_mf, m2_el);
    out.M3 = build_agg(m3_cfdp, m3_ctpr, m3_esup, m3_bhr, m3_bfdp,
                        m3_fbr, m3_nba, m3_pur, m3_lam, m3_ts, m3_mf, m3_el);
    out.M4 = build_agg(m4_cfdp, m4_ctpr, m4_esup, m4_bhr, m4_bfdp,
                        m4_fbr, m4_nba, m4_pur, m4_lam, m4_ts, m4_mf, m4_el);

    std::cout << "  " << label << " — done."
              << "  M1 TPR=" << std::fixed << std::setprecision(3)
              << out.M1.mean_coord_tpr
              << "  M2 TPR=" << out.M2.mean_coord_tpr
              << "  M3 TPR=" << out.M3.mean_coord_tpr
              << "  M4 TPR=" << out.M4.mean_coord_tpr
              << "\n\n" << std::flush;

    return out;
}

/**
 * @brief Run num_MC parallel trials at one (scenario, snr, rho) point
 *        (resumable grid cell).
 *
 * @param dgp_fn     DGP factory — captures scenario, n, p, G, block_size,
 *                   n_active_blocks, rho, snr, b via closure.
 *                   Signature: (seed) → GVSDGPResult.
 * @param cfg        Benchmark configuration.
 * @param trex_ctrl  Base T-Rex control parameters.
 * @param label      Progress label for console output.
 * @return BlockGridResult aggregating all four methods.
 */
inline BlockGridResult run_block_mc(
    const std::function<GVSDGPResult(unsigned)>& dgp_fn,
    const BlockBenchConfig&                      cfg,
    const TRexControlParameter&                  trex_ctrl,
    const std::string&                           label)
{
    std::ostringstream key;
    key << "block|" << label << "|seed=" << cfg.base_seed << "|mc=" << cfg.num_MC
        << "|tfdr=" << cfg.tFDR << "|corr=" << cfg.corr_max
        << "|sel=" << trex_solver_filter::file_suffix();   // no reuse across selections
    return cell_detail::block_from_cell(TREX_CELL_CACHED(key.str(), [&] {
        return cell_detail::to_cell(run_block_mc_impl(dgp_fn, cfg, trex_ctrl, label));
    }));
}


// ==============================================================================
// Print — single trial (demo use)
// ==============================================================================

/**
 * @brief Print a formatted single-run result block for all four methods.
 *
 * @param scenario_tag  Label for the scenario (e.g. "INDIVIDUAL").
 * @param dat           DGP metadata.
 * @param res           BlockTrialResult for all four methods.
 * @param tFDR          Target FDR (for display).
 * @param G             Total number of blocks.
 * @param block_size    Variables per block.
 */
inline void print_block_trial_result(
    const std::string&    scenario_tag,
    const GVSDGPResult&   dat,
    const BlockTrialResult& res,
    double                tFDR,
    int                   G,
    int                   block_size,
    double                rho)
{
    auto print_method = [&](const std::string& mname,
                             const BlockMethodResult& m) {
        std::cout << std::string(70, '-') << "\n";
        std::cout << "  " << mname << "\n";
        std::cout << std::fixed << std::setprecision(4);
        std::cout << "  Grouping:     purity=" << m.block_purity_rate
                  << "  M=" << m.M_found
                  << "  lambda2=" << m.lambda2_used
                  << "  T_stop=" << m.T_stop << "\n";
        std::cout << "  Coord:        TPR=" << m.coord_tpr
                  << "  FDP=" << m.coord_fdp
                  << "  exact=" << m.exact_support << "\n";
        std::cout << "  Block:        hit_rate=" << m.block_hit_rate
                  << "  block_fdp=" << m.block_fdp
                  << "  full=" << m.full_block_rate
                  << "  null_act=" << m.null_block_activation << "\n";
        std::cout << "  Wall-clock:   " << std::setprecision(2)
                  << m.elapsed_sec << " s\n";
    };

    std::cout << std::string(70, '=') << "\n";
    std::cout << "  Block Bench: " << scenario_tag << "\n";
    std::cout << "  n=" << dat.n << "  p=" << dat.p
              << "  G=" << G << "  block_size=" << block_size
              << "  s=" << dat.s << "\n";
    std::cout << "  rho=" << std::fixed << std::setprecision(2) << rho
              << "  SNR=" << dat.snr
              << "  sigma_y=" << std::setprecision(4) << dat.sigma_y
              << "  tFDR=" << tFDR << "\n";
    std::cout << "  Active blocks: ";
    for (int b : dat.active_blocks) std::cout << b << " ";
    std::cout << "\n";

    print_method("M1  EN  + HAC-clustered", res.M1);
    print_method("M2  EN  + oracle groups", res.M2);
    print_method("M3  IEN + HAC-clustered", res.M3);
    print_method("M4  IEN + oracle groups", res.M4);

    std::cout << std::string(70, '=') << "\n\n";
}


// ==============================================================================
// Print — MC aggregate table (sim use)
// ==============================================================================

/**
 * @brief Print a formatted MC aggregate table for one scenario.
 *
 * Columns: Method | coord_TPR | coord_FDP | block_metric | purity | T_stop | elapsed
 * Rows: M1–M4 for each (rho, snr) cell.
 *
 * @param scenario_tag     Label for the scenario.
 * @param scenario         BlockScenario enum (controls which block metric is shown).
 * @param rho_grid         Rho values swept.
 * @param snr_grid         SNR values swept.
 * @param results          results[i_rho][i_snr] — BlockGridResult.
 * @param cfg              Benchmark configuration.
 * @param file_stem        Output file stem (no folder, no extension). If
 *                         non-empty, the table is also written as .txt and all
 *                         aggregates as a tidy .csv to DEMO_OUTPUT_DIR.
 */
inline void print_block_grid_table(
    const std::string&                                    scenario_tag,
    BlockScenario                                         scenario,
    const std::vector<double>&                            rho_grid,
    const std::vector<double>&                            snr_grid,
    const std::vector<std::vector<BlockGridResult>>&      results,
    const BlockBenchConfig&                               cfg,
    const std::string&                                    file_stem = "")
{
    // Choose the scenario-specific block metric name and accessor
    std::string blk_metric_name;
    switch (scenario) {
        case BlockScenario::INDIVIDUAL:    blk_metric_name = "blk_hit"; break;
        case BlockScenario::REPRESENTATIVE: blk_metric_name = "blk_hit"; break;
        case BlockScenario::WHOLE:          blk_metric_name = "full_blk"; break;
    }

    auto get_blk_metric = [&](const BlockMethodAggregate& a) -> double {
        switch (scenario) {
            case BlockScenario::INDIVIDUAL:    return a.mean_block_hit;
            case BlockScenario::REPRESENTATIVE: return a.mean_block_hit;
            case BlockScenario::WHOLE:          return a.mean_full_block;
        }
        return 0.0;
    };

    constexpr int cw = 10;
    const std::string methods[4] = {"M1(EN-C)", "M2(EN-O)", "M3(IE-C)", "M4(IE-O)"};
    auto get_agg = [](const BlockGridResult& g, int m) -> const BlockMethodAggregate& {
        switch (m) {
            case 0: return g.M1;
            case 1: return g.M2;
            case 2: return g.M3;
            default: return g.M4;
        }
    };

    std::ostringstream oss;
    oss << "\n" << std::string(90, '=') << "\n"
        << "  Block Bench MC: " << scenario_tag
        << "  (MC=" << cfg.num_MC << ", tFDR=" << cfg.tFDR
        << ", n=" << cfg.n << ", p=" << cfg.p
        << ", G=" << cfg.G << ", blk=" << cfg.block_size << ")\n"
        << std::string(90, '=') << "\n";

    for (std::size_t i_rho = 0; i_rho < rho_grid.size(); ++i_rho) {
        oss << "\n  rho = " << std::fixed << std::setprecision(2)
            << rho_grid[i_rho] << "\n";

        // Header
        oss << "  " << std::left << std::setw(12) << "method"
            << std::right
            << std::setw(7) << "SNR"
            << std::setw(cw) << "TPR"
            << std::setw(cw) << "FDR"
            << std::setw(cw) << "block_FDR"
            << std::setw(cw) << blk_metric_name
            << std::setw(cw) << "purity"
            << std::setw(cw) << "T_stop"
            << std::setw(cw) << "M_found"
            << "\n"
            << "  " << std::string(12 + 7 + 7 * cw, '-') << "\n";

        for (std::size_t i_snr = 0; i_snr < snr_grid.size(); ++i_snr) {
            for (int m = 0; m < 4; ++m) {
                const auto& a = get_agg(results[i_rho][i_snr], m);
                oss << "  " << std::left << std::setw(12) << methods[m]
                    << std::right << std::fixed << std::setprecision(2)
                    << std::setw(7) << snr_grid[i_snr]
                    << std::setprecision(4)
                    << std::setw(cw) << a.mean_coord_tpr
                    << std::setw(cw) << a.mean_coord_fdp
                    << std::setw(cw) << a.mean_block_fdp
                    << std::setw(cw) << get_blk_metric(a)
                    << std::setw(cw) << a.mean_purity
                    << std::setprecision(1)
                    << std::setw(cw) << a.mean_T_stop
                    << std::setprecision(1)
                    << std::setw(cw) << a.mean_M_found
                    << "\n";
            }
            oss << "\n";
        }
    }
    oss << std::string(90, '=') << "\n\n";

    const std::string table = oss.str();
    std::cout << table;

    if (file_stem.empty()) return;

    const std::string dir(DEMO_OUTPUT_DIR);
    std::filesystem::create_directories(dir);

    {
        std::ofstream txt(dir + out_stem(file_stem) + ".txt");
        if (txt.is_open()) {
            txt << table;
            std::cout << "[Info] TXT saved to: " << dir + out_stem(file_stem) + ".txt\n";
        }
    }

    // CSV: tidy long format with all aggregates (mean + sd where available)
    {
        std::ofstream csv(dir + out_stem(file_stem) + ".csv");
        if (csv.is_open()) {
            csv << "scenario,rho,snr,method,"
                   "mean_coord_tpr,sd_coord_tpr,mean_coord_fdp,sd_coord_fdp,"
                   "mean_exact_sup,mean_block_hit,mean_block_fdp,"
                   "mean_full_block,mean_null_act,mean_purity,"
                   "mean_lambda2,mean_T_stop,mean_M_found\n"
                << std::fixed << std::setprecision(6);
            for (std::size_t i_rho = 0; i_rho < rho_grid.size(); ++i_rho) {
                for (std::size_t i_snr = 0; i_snr < snr_grid.size(); ++i_snr) {
                    for (int m = 0; m < 4; ++m) {
                        const auto& a = get_agg(results[i_rho][i_snr], m);
                        csv << scenario_tag << ","
                            << rho_grid[i_rho] << ","
                            << snr_grid[i_snr] << ","
                            << methods[m] << ","
                            << a.mean_coord_tpr << "," << a.sd_coord_tpr << ","
                            << a.mean_coord_fdp << "," << a.sd_coord_fdp << ","
                            << a.mean_exact_sup << ","
                            << a.mean_block_hit << ","
                            << a.mean_block_fdp << ","
                            << a.mean_full_block << ","
                            << a.mean_null_act << ","
                            << a.mean_purity << ","
                            << a.mean_lambda2 << ","
                            << a.mean_T_stop << ","
                            << a.mean_M_found << "\n";
                    }
                }
            }
            std::cout << "[Info] CSV saved to: " << dir + out_stem(file_stem) + ".csv\n\n";
        }
    }
}


// ==============================================================================
} /* End of namespace gvs_demo */
// ==============================================================================
#endif /* End of DEMOS_TREX_SELECTOR_METHODS_GVS_SIMULATION_UTILS_HPP */
