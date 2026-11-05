// ==============================================================================
// trex_screening_simulation_utils.hpp
// ==============================================================================
/**
 * @file trex_screening_simulation_utils.hpp
 *
 * @brief Shared simulation infrastructure for the Screen-TRex demo suite.
 *
 * @details
 *  Includes trex_screening_dgps.hpp (ScrDGPData + all DGP generators) and adds
 *  everything the Monte Carlo demos need on top of it:
 *
 *  Types:
 *    - ScreenMethodInfo            method descriptor (name + bootstrap flag)
 *    - ScrGridPointResult          MC aggregate (avg/sd FDR, TPR + avg Est. FDR)
 *    - ScrDGPFactory               factory callable type: seed -> ScrDGPData
 *
 *  Control-parameter factories:
 *    - make_trex_control()         default TRexControlParameter for Screen-TRex
 *    - make_trex_control_mmap()    same, with use_memory_mapping = true
 *    - make_screen_control()       default ScreenTRexControlParameter
 *    - makeBiobankControl()        BiobankScreenTRexControl ("Algorithm 1")
 *
 *  Monte Carlo infrastructure:
 *    - run_mc_trials_screen_trex() parallel MC loop via OpenMP
 *    - print_mc_progress() / print_mc_done()  progress lines
 *
 *  Result output (console + TXT + tidy CSV in DEMO_OUTPUT_DIR):
 *    - save_and_print_mc_results()          FDR / TPR / Est. FDR sweep tables
 *    - save_and_print_biobank_mc_results()  same, plus per-method Usage %
 *    - rekey_for_display()                  library keys -> display names
 *
 * @note Demo-internal header — not part of the TRexSelector library.
 */
// ==============================================================================

#ifndef TREX_DEMOS_TREX_SCREENING_SIMULATION_UTILS_HPP
#define TREX_DEMOS_TREX_SCREENING_SIMULATION_UTILS_HPP

// Screen-TRex demo DGPs (ScrDGPData + generators)
#include "trex_screening_dgps.hpp"
#include "../../cell_cache.hpp"   // resumable grid cells
#include "../../mc_workers.hpp"   // thread budget (TREX_NUM_WORKERS)

// std includes
#include <algorithm>
#include <functional>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

// Eigen includes
#include <Eigen/Dense>

// TRex includes
#include <trex_selector_methods/ad/trex_screening/trex_screening.hpp>
#include <trex_selector_methods/ad/trex_screening/trex_biobank_screening.hpp>
#include <utils/eval_metrics/utils_eval_cdiagnostics.hpp>
#include <utils/eval_metrics/utils_eval_rates.hpp>
#include <utils/openmp/utils_openmp.hpp>

// ==============================================================================
// Namespace aliases
// ==============================================================================

namespace scr_demo {

namespace cdiag    = trex::utils::eval::cdiagnostics;
namespace dummygen = trex::utils::datageneration::dummygen;
namespace rates    = trex::utils::eval::rates;

/** @brief Namespace alias for Biobank Screen-TRex types. */
namespace tbs = trex::trex_selector_methods::ad::trex_biobank_screening;

using trex::trex_selector_methods::ad::trex_core::TRexControlParameter;
using trex::trex_selector_methods::ad::trex_core::LLoopStrategy;
using trex::trex_selector_methods::ad::utils::solver_dispatch::SolverTypeForTRex;
using trex::trex_selector_methods::ad::trex_screening::ScreenTRexControlParameter;
using trex::trex_selector_methods::ad::trex_screening::ScreenTRexMethod;
using trex::trex_selector_methods::ad::trex_screening::ScreenTRexSelector;
using trex::trex_selector_methods::ad::trex_screening::ScreenTRexSelectionResult;


// ==============================================================================
// Types
// ==============================================================================

/** @brief Method descriptor for MC comparisons. */
struct ScreenMethodInfo {
    std::string     name;
    ScreenTRexMethod method    = ScreenTRexMethod::TREX;
    bool            bootstrap = false;
};

/** @brief Default method set for ordinary-vs-bootstrap comparisons. */
inline std::vector<ScreenMethodInfo> default_methods() {
    return {
        {"Screen-TRex Ordinary: TLARS",     ScreenTRexMethod::TREX, false},
        {"Screen-TRex Bootstrap-CI: TLARS", ScreenTRexMethod::TREX, true }
    };
}


// ==============================================================================
// Control-parameter factories
// ==============================================================================

/** @brief Default TRexControlParameter for Screen-TRex (in-memory). */
inline TRexControlParameter make_trex_control(std::size_t K = 20) {
    TRexControlParameter ctrl;
    ctrl.K                    = K;
    ctrl.lloop_strategy       = LLoopStrategy::STANDARD;
    ctrl.solver_type          = SolverTypeForTRex::TLARS;
    ctrl.dummy_distribution   = dummygen::Distribution::Normal();
    ctrl.use_memory_mapping   = false;
    return ctrl;
}

/** @brief Default TRexControlParameter for Screen-TRex (memory-mapped). */
inline TRexControlParameter make_trex_control_mmap(std::size_t K = 20) {
    auto ctrl = make_trex_control(K);
    ctrl.use_memory_mapping = true;
    return ctrl;
}

/** @brief Default ScreenTRexControlParameter. */
inline ScreenTRexControlParameter make_screen_control(
    ScreenTRexMethod method    = ScreenTRexMethod::TREX,
    bool             bootstrap = false,
    std::size_t      n_blocks  = 5)
{
    ScreenTRexControlParameter ctrl;
    ctrl.trex_method      = method;
    ctrl.use_bootstrap_CI = bootstrap;
    ctrl.R_boot           = 1000;
    ctrl.ci_grid_step     = 0.001;
    ctrl.rho_thr_DA       = 0.02;
    ctrl.n_blocks         = n_blocks;
    return ctrl;
}


/**
 * @brief Re-key result maps from the library's method_used strings to display
 *        names for the output tables and figure legends.
 *
 * @details The biobank demos accumulate into maps keyed by
 *          BiobankScreenTRexResult::method_used, so those keys are fixed by the
 *          library. This copies each series under the suite's display name
 *          ("<screening method>: <solver>"), keeping the saved tables and the
 *          plotted legends consistent with the other demos.
 *
 * @param keys      Library method_used strings, in display order.
 * @param display   Display names, same length and order as `keys`.
 * @param src       Map keyed by `keys`.
 * @return          Map keyed by `display`.
 */
inline std::map<std::string, Eigen::VectorXd> rekey_for_display(
    const std::vector<std::string>&               keys,
    const std::vector<std::string>&               display,
    const std::map<std::string, Eigen::VectorXd>& src)
{
    std::map<std::string, Eigen::VectorXd> out;
    for (std::size_t i = 0; i < keys.size(); ++i) {
        out[display[i]] = src.at(keys[i]);
    }
    return out;
}


// ==============================================================================
// MC results: dual console + file output (TXT + CSV)
// ==============================================================================

/**
 * @brief Save and print a sweep-variable MC results table.
 *
 * @details Writes a human-readable table (matching the T-Rex results format)
 *          to **console + TXT file** and a machine-readable tidy-long CSV.
 *          Files are placed in `DEMO_OUTPUT_DIR` under the supplied stem.
 *
 * @param num_MC      Number of Monte Carlo runs.
 * @param file_stem   Output file stem (no folder, no extension).
 * @param x_values    Sweep variable grid (SNR values, rho values, etc.).
 * @param methods     Method descriptors.
 * @param fdr_map     Method name -> FDR vector.
 * @param tpr_map     Method name -> TPR vector.
 * @param est_fdr_map Method name -> estimated FDR vector.
 * @param sweep_label Column label for the sweep variable (default: "SNR").
 */
inline void save_and_print_mc_results(
    std::size_t                                    num_MC,
    const std::string&                             file_stem,
    const std::vector<double>&                     x_values,
    const std::vector<ScreenMethodInfo>&            methods,
    const std::map<std::string, Eigen::VectorXd>&  fdr_map,
    const std::map<std::string, Eigen::VectorXd>&  tpr_map,
    const std::map<std::string, Eigen::VectorXd>&  est_fdr_map,
    const std::string&                             sweep_label = "SNR")
{
    const std::string folder = DEMO_OUTPUT_DIR;
    std::filesystem::create_directories(folder);

    // ── Open TXT file for dual output ────────────────────────────────────
    std::ofstream out_file(folder + file_stem + ".txt");

    auto print_dual = [&](const std::string& text) {
        std::cout  << text;
        if (out_file.is_open()) out_file << text;
    };

    // ── Table dimensions ─────────────────────────────────────────────────
    // The method name gets its own line; metric rows are indented beneath it,
    // so the value columns stay aligned no matter how long a method name is.
    const int indent_w = 2;    // leading indent of a metric row
    const int metric_w = 23;   // left-aligned metric label
    const int col_w    = 10;   // right-aligned values
    const std::size_t sep_w =
        static_cast<std::size_t>(indent_w + metric_w)
        + col_w * x_values.size();

    // ── Header ───────────────────────────────────────────────────────────
    {
        std::ostringstream ss;
        ss << "\n"
           << std::string(70, '=') << "\n"
           << "=== Screen-TRex Results (averaged over "
           << num_MC << " Monte Carlo runs) ===\n"
           << std::string(70, '=') << "\n\n";
        print_dual(ss.str());
    }

    // ── Column header (sweep axis) + separator ───────────────────────────
    {
        std::ostringstream hdr;
        hdr << std::left << std::string(static_cast<std::size_t>(indent_w), ' ')
            << std::setw(metric_w) << sweep_label;
        for (const double x : x_values) {
            hdr << std::right << std::fixed << std::setprecision(2)
                << std::setw(col_w) << x;
        }
        hdr << "\n" << std::string(sep_w, '-') << "\n";
        print_dual(hdr.str());
    }

    // ── Data rows ────────────────────────────────────────────────────────
    auto print_metric = [&](const std::string& label,
                            const Eigen::VectorXd& vec) {
        std::ostringstream row;
        row << std::left << std::string(static_cast<std::size_t>(indent_w), ' ')
            << std::setw(metric_w) << label;
        for (Eigen::Index i = 0;
             i < static_cast<Eigen::Index>(x_values.size()); ++i) {
            row << std::right << std::fixed << std::setprecision(4)
                << std::setw(col_w) << vec(i);
        }
        row << "\n";
        print_dual(row.str());
    };

    for (const auto& m : methods) {
        const std::string& nm = m.name;
        print_dual("\n" + nm + "\n");           // method name on its own line
        print_metric("FDR",      fdr_map.at(nm));
        print_metric("TPR",      tpr_map.at(nm));
        print_metric("Est. FDR", est_fdr_map.at(nm));
    }
    print_dual("\n");

    // ── Close TXT ────────────────────────────────────────────────────────
    if (out_file.is_open()) {
        out_file.close();
        std::cout << "[Info] TXT results saved to: "
                  << folder + file_stem + ".txt\n";
    }

    // ── CSV file: tidy long format ───────────────────────────────────────
    std::ofstream csv(folder + file_stem + ".csv");
    if (csv.is_open()) {
        csv << "method,metric," << sweep_label << ",value\n"
            << std::fixed << std::setprecision(6);
        for (const auto& m : methods) {
            const std::string& nm = m.name;
            for (std::size_t i = 0; i < x_values.size(); ++i) {
                const auto ei = static_cast<Eigen::Index>(i);
                const double x = x_values[i];
                csv << nm << ",FDR,"      << x << "," << fdr_map.at(nm)(ei)     << "\n";
                csv << nm << ",TPR,"      << x << "," << tpr_map.at(nm)(ei)     << "\n";
                csv << nm << ",Est. FDR," << x << "," << est_fdr_map.at(nm)(ei) << "\n";
            }
        }
        csv.close();
        std::cout << "[Info] CSV results saved to: "
                  << folder + file_stem + ".csv\n";
    } else {
        std::cout << "[Warning] Could not open CSV file: "
                  << folder + file_stem + ".csv\n";
    }
    std::cout << "\n";
}


// ==============================================================================
// MC progress helper
// ==============================================================================

/** @brief Print in-place MC progress line (carriage-return overwrite). */
inline void print_mc_progress(double val,
                              std::size_t mc,
                              std::size_t num_MC,
                              const std::string& sweep_label = "SNR") {
    std::cout << "  " << sweep_label << " "
              << std::fixed << std::setprecision(2) << val
              << " \u2014 Progress: ["
              << std::setw(3) << (mc + 1) << "/" << num_MC << "] "
              << std::fixed << std::setprecision(1)
              << (100.0 * static_cast<double>(mc + 1)
                  / static_cast<double>(num_MC))
              << "%    \r" << std::flush;
}

/** @brief Print MC completion line for a given sweep point. */
inline void print_mc_done(double val,
                          std::size_t num_MC,
                          const std::string& sweep_label = "SNR") {
    std::cout << "  " << sweep_label << " "
              << std::fixed << std::setprecision(2) << val
              << " — Completed " << num_MC << " runs.      \n";
}


// ==============================================================================
// Parallel MC infrastructure
// ==============================================================================

/** @brief Aggregate MC result for one Screen-TRex grid point. */
struct ScrGridPointResult {
    double avg_fdr     = 0.0;   ///< Mean FDR (false discovery proportion) across MC trials
    double avg_tpr     = 0.0;   ///< Mean TPR (true positive proportion) across MC trials
    double avg_est_fdr = 0.0;   ///< Mean estimated FDR from Screen-TRex
    double sd_fdr      = 0.0;   ///< Bessel-corrected sd of FDR
    double sd_tpr      = 0.0;   ///< Bessel-corrected sd of TPR
};

/**
 * @brief Factory callable: given a trial seed, produce one MC dataset.
 *
 * Each OpenMP thread calls this independently with its per-trial seed,
 * so the factory must be self-contained and not share mutable state.
 */
using ScrDGPFactory = std::function<ScrDGPData(unsigned seed)>;

namespace cell_detail {

/** @brief Cell key: label plus the parameters that tell cells apart. */
inline std::string key(const std::string& label, std::size_t num_MC, unsigned base_seed_offset,
                       const TRexControlParameter& trex_ctrl, const ScreenTRexControlParameter& screen) {
    std::ostringstream os;
    os << "scr|" << label << "|solver=" << static_cast<int>(trex_ctrl.solver_type)
       << "|method=" << static_cast<int>(screen.trex_method) << "|seed=" << base_seed_offset
       << "|mc=" << num_MC;
    return os.str();
}

inline trex_cell_cache::Cell to_cell(const ScrGridPointResult& r) {
    return {{"avg_fdr", r.avg_fdr}, {"avg_tpr", r.avg_tpr}, {"avg_est_fdr", r.avg_est_fdr},
            {"sd_fdr", r.sd_fdr}, {"sd_tpr", r.sd_tpr}};
}

inline ScrGridPointResult from_cell(const trex_cell_cache::Cell& c) {
    using trex_cell_cache::get;
    return {get(c, "avg_fdr"), get(c, "avg_tpr"), get(c, "avg_est_fdr"),
            get(c, "sd_fdr"), get(c, "sd_tpr")};
}

}  // namespace cell_detail

/**
 * @brief Sums of one biobank SNR cell per library method name: conditional FDP
 *        and TPP sums, unconditional estimated-FDR sum and usage count.
 */
struct BiobankCellSums {
    std::map<std::string, double> fdp, tpp, est_fdr, usage;

    /** @brief Cell with four numbers per method, in the order of @p methods. */
    trex_cell_cache::Cell to_cell(const std::vector<std::string>& methods) const {
        const auto at = [](const std::map<std::string, double>& m, const std::string& k) {
            const auto it = m.find(k);
            return it == m.end() ? 0.0 : it->second;
        };
        trex_cell_cache::Cell c;
        for (std::size_t k = 0; k < methods.size(); ++k) {
            const std::string p = "m" + std::to_string(k) + ".";
            c.emplace_back(p + "fdp", at(fdp, methods[k]));
            c.emplace_back(p + "tpp", at(tpp, methods[k]));
            c.emplace_back(p + "est_fdr", at(est_fdr, methods[k]));
            c.emplace_back(p + "usage", at(usage, methods[k]));
        }
        return c;
    }

    /** @brief Add a cell's sums to the result maps at SNR index @p si. */
    static void add_to(const trex_cell_cache::Cell& c, const std::vector<std::string>& methods,
                       Eigen::Index si, std::map<std::string, Eigen::VectorXd>& fdp_map,
                       std::map<std::string, Eigen::VectorXd>& tpp_map,
                       std::map<std::string, Eigen::VectorXd>& est_fdr_map,
                       std::map<std::string, Eigen::VectorXd>& usage_map) {
        using trex_cell_cache::get;
        for (std::size_t k = 0; k < methods.size(); ++k) {
            const std::string p = "m" + std::to_string(k) + ".";
            fdp_map[methods[k]](si)     += get(c, p + "fdp");
            tpp_map[methods[k]](si)     += get(c, p + "tpp");
            est_fdr_map[methods[k]](si) += get(c, p + "est_fdr");
            usage_map[methods[k]](si)   += get(c, p + "usage");
        }
    }
};

/** @brief Uncached body of run_mc_trials_screen_trex(). */
inline ScrGridPointResult run_mc_trials_screen_trex_impl(
    std::size_t                       num_MC,
    const std::string&                progress_label,
    const ScrDGPFactory&              make_data,
    const TRexControlParameter&       trex_ctrl,
    const ScreenTRexControlParameter& screen_ctrl,
    unsigned                          base_seed_offset)
{
    const int iMC = static_cast<int>(num_MC);
    std::vector<double> fdp_vec    (num_MC, 0.0);
    std::vector<double> tpp_vec    (num_MC, 0.0);
    std::vector<double> est_fdr_vec(num_MC, 0.0);

    std::cout << "  " << progress_label
              << " — Running " << num_MC << " MC trials ...\n" << std::flush;

    #pragma omp parallel for schedule(dynamic)
    for (int mc = 0; mc < iMC; ++mc) {
        const unsigned trial_seed = base_seed_offset + static_cast<unsigned>(mc);
        auto dat = make_data(trial_seed);

        Eigen::Map<Eigen::MatrixXd> X_map(
            dat.X.data(), dat.X.rows(), dat.X.cols());
        Eigen::Map<Eigen::VectorXd> y_map(
            dat.y.data(), dat.y.size());

        // ScreenTRexControlParameter now nests its own trex_ctrl member --
        // fold trex_ctrl into a clean local copy before construction.
        ScreenTRexControlParameter trex_screen_ctrl = screen_ctrl;
        trex_screen_ctrl.trex_ctrl = trex_ctrl;

        ScreenTRexSelector sctrex(
            X_map, y_map, trex_screen_ctrl, /*seed=*/-1, /*verbose=*/false);
        sctrex.select();

        const auto selected    = sctrex.getSelectedIndices();
        fdp_vec    [mc]        = rates::compute_fdp(selected, dat.true_support);
        tpp_vec    [mc]        = rates::compute_tpp(selected, dat.true_support);
        est_fdr_vec[mc]        = sctrex.getScreenResult().estimated_FDR;
    }

    // Compute means
    const double inv  = 1.0 / static_cast<double>(num_MC);
    const double mfdr = std::accumulate(fdp_vec.begin(),     fdp_vec.end(),     0.0) * inv;
    const double mtpr = std::accumulate(tpp_vec.begin(),     tpp_vec.end(),     0.0) * inv;
    const double mest = std::accumulate(est_fdr_vec.begin(), est_fdr_vec.end(), 0.0) * inv;

    // Bessel-corrected standard deviations (only meaningful for num_MC > 1)
    double sfdr = 0.0, stpr = 0.0;
    if (num_MC > 1) {
        for (const double v : fdp_vec) sfdr += (v - mfdr) * (v - mfdr);
        for (const double v : tpp_vec) stpr += (v - mtpr) * (v - mtpr);
        const double bc = 1.0 / static_cast<double>(num_MC - 1);
        sfdr = std::sqrt(sfdr * bc);
        stpr = std::sqrt(stpr * bc);
    }

    std::cout << "  " << progress_label
              << " — done.  TPR=" << std::fixed << std::setprecision(4) << mtpr
              << "  FDR=" << mfdr
              << "  Est.FDR=" << mest << "\n" << std::flush;

    return {mfdr, mtpr, mest, sfdr, stpr};
}

/**
 * @brief Parallel MC loop for Screen-TRex (resumable grid cell).
 *
 * Uses `#pragma omp parallel for schedule(dynamic)`. Each trial calls
 * `make_data(base_seed_offset + mc)` and runs `ScreenTRexSelector`
 * independently.  Collects FDP, TPP, estimated FDR; returns averaged +
 * Bessel-corrected sd result.
 *
 * Thread safety:
 *   - Each trial owns its data (returned by-value from the factory).
 *   - `trex_ctrl` and `screen_ctrl` are passed by const-ref; each
 *     `ScreenTRexSelector` receives them as values / copies internally.
 *   - `TempDirManager::create()` uses `std::random_device` → unique temp
 *     directories per selector instance → safe under OpenMP.
 *
 * @param num_MC           Number of Monte Carlo trials.
 * @param progress_label   Label printed in the "Running … done." lines.
 * @param make_data        DGP factory; called once per trial with the trial seed.
 * @param trex_ctrl        TRex control parameter (constant across trials).
 * @param screen_ctrl      Screen-TRex control parameter (constant across trials).
 * @param base_seed_offset Trial seed = base_seed_offset + mc.
 * @return                 ScrGridPointResult with averaged metrics.
 */
inline ScrGridPointResult run_mc_trials_screen_trex(
    std::size_t                       num_MC,
    const std::string&                progress_label,
    const ScrDGPFactory&              make_data,
    const TRexControlParameter&       trex_ctrl,
    const ScreenTRexControlParameter& screen_ctrl,
    unsigned                          base_seed_offset)
{
    return cell_detail::from_cell(TREX_CELL_CACHED(
        cell_detail::key(progress_label, num_MC, base_seed_offset, trex_ctrl, screen_ctrl), [&] {
            return cell_detail::to_cell(run_mc_trials_screen_trex_impl(
                num_MC, progress_label, make_data, trex_ctrl, screen_ctrl, base_seed_offset));
        }));
}


// ==============================================================================
// Control factory
// ==============================================================================

/**
 * @brief Build a default BiobankScreenTRexControl.
 *
 * @param use_mmap  Use memory-mapped D-file storage (default: false = in-memory).
 * @param K         Number of experiments per phenotype (default: 20).
 */
inline tbs::BiobankScreenTRexControl makeBiobankControl(
    bool use_mmap = false, std::size_t K = 20)
{
    tbs::BiobankScreenTRexControl ctrl;
    ctrl.lower_bound_FDR = 0.05;
    ctrl.upper_bound_FDR = 0.15;
    ctrl.target_FDR_trex = 0.10;

    // BiobankScreenTRexControl nests ScreenTRexControlParameter (which in turn
    // nests its own trex_ctrl) as `trex_screen_ctrl` -- there is no separate
    // top-level screen_ctrl/trex_ctrl anymore.
    ctrl.trex_screen_ctrl.trex_method  = ScreenTRexMethod::TREX;
    ctrl.trex_screen_ctrl.R_boot       = 1000;
    ctrl.trex_screen_ctrl.ci_grid_step = 0.001;

    ctrl.trex_screen_ctrl.trex_ctrl.K                    = K;
    ctrl.trex_screen_ctrl.trex_ctrl.max_dummy_multiplier = 10;
    ctrl.trex_screen_ctrl.trex_ctrl.use_max_T_stop       = true;
    ctrl.trex_screen_ctrl.trex_ctrl.dummy_distribution   = dummygen::Distribution::Normal();
    ctrl.trex_screen_ctrl.trex_ctrl.lloop_strategy       = LLoopStrategy::STANDARD;
    ctrl.trex_screen_ctrl.trex_ctrl.solver_type          = SolverTypeForTRex::TLARS;
    ctrl.trex_screen_ctrl.trex_ctrl.use_memory_mapping   = use_mmap;

    return ctrl;
}


// ==============================================================================
// Single-phenotype result output
// ==============================================================================

// ==============================================================================
// MC results: save + print (TXT + CSV)
// ==============================================================================

/**
 * @brief Save and print a Biobank Screen-TRex SNR-sweep MC results table.
 *
 * @details Writes a human-readable table (console + TXT) and a tidy CSV.
 *          Files are placed in `DEMO_OUTPUT_DIR` (each demo's own
 *          `simulation_results/` folder), which is created if it does not exist.
 *          Includes a "Usage (%)" row per method — the fraction of MC runs
 *          in which Algorithm 1 routed to that method.
 *
 * @param num_MC       Number of Monte Carlo runs.
 * @param file_stem    Output file stem (no folder, no extension).
 * @param snr_values   SNR grid.
 * @param method_names Ordered method names (keys into all result maps).
 * @param fdp_map      Method → conditional-mean FDP vector.
 * @param tpp_map      Method → conditional-mean TPP vector.
 * @param est_fdr_map  Method → mean estimated FDR vector.
 * @param usage_map    Method → usage fraction [0, 1] vector.
 */
inline void save_and_print_biobank_mc_results(
    std::size_t                                    num_MC,
    const std::string&                             file_stem,
    const std::vector<double>&                     snr_values,
    const std::vector<std::string>&                method_names,
    const std::map<std::string, Eigen::VectorXd>&  fdp_map,
    const std::map<std::string, Eigen::VectorXd>&  tpp_map,
    const std::map<std::string, Eigen::VectorXd>&  est_fdr_map,
    const std::map<std::string, Eigen::VectorXd>&  usage_map)
{
    const std::string folder = DEMO_OUTPUT_DIR;
    std::filesystem::create_directories(folder);

    // ── Open TXT file for dual output ─────────────────────────────────────
    std::ofstream out_file(folder + file_stem + ".txt");
    auto print_dual = [&](const std::string& text) {
        std::cout  << text;
        if (out_file.is_open()) out_file << text;
    };

    // ── Table dimensions ──────────────────────────────────────────────────
    // Method name on its own line, metric rows indented beneath it.
    const int indent_w = 2;
    const int metric_w = 23;
    const int col_w    = 10;
    const std::size_t sep_w =
        static_cast<std::size_t>(indent_w + metric_w)
        + col_w * snr_values.size();

    // ── Header ────────────────────────────────────────────────────────────
    {
        std::ostringstream ss;
        ss << "\n"
           << std::string(70, '=') << "\n"
           << "=== Biobank Screen-TRex Results (averaged over "
           << num_MC << " Monte Carlo runs) ===\n"
           << std::string(70, '=') << "\n\n";
        print_dual(ss.str());
    }

    // ── Column header (sweep axis) + separator ────────────────────────────
    {
        std::ostringstream hdr;
        hdr << std::left << std::string(static_cast<std::size_t>(indent_w), ' ')
            << std::setw(metric_w) << "SNR";
        for (const double snr : snr_values)
            hdr << std::right << std::fixed << std::setprecision(2)
                << std::setw(col_w) << snr;
        hdr << "\n" << std::string(sep_w, '-') << "\n";
        print_dual(hdr.str());
    }

    // ── Data rows ─────────────────────────────────────────────────────────
    auto print_metric = [&](const std::string& label,
                            const Eigen::VectorXd& vec, int dp) {
        std::ostringstream row;
        row << std::left << std::string(static_cast<std::size_t>(indent_w), ' ')
            << std::setw(metric_w) << label;
        for (Eigen::Index i = 0; i < static_cast<Eigen::Index>(snr_values.size()); ++i)
            row << std::right << std::fixed << std::setprecision(dp)
                << std::setw(col_w) << vec(i);
        row << "\n";
        print_dual(row.str());
    };

    for (const auto& nm : method_names) {
        const Eigen::VectorXd usage_pct = usage_map.at(nm) * 100.0;
        print_dual("\n" + nm + "\n");            // method name on its own line
        print_metric("Usage (%)", usage_pct,          1);
        print_metric("FDR",       fdp_map.at(nm),     4);
        print_metric("TPR",       tpp_map.at(nm),     4);
        print_metric("Est. FDR",  est_fdr_map.at(nm), 4);
    }
    print_dual("\n");

    // ── Close TXT ─────────────────────────────────────────────────────────
    if (out_file.is_open()) {
        out_file.close();
        std::cout << "[Info] TXT results saved to: " << folder + file_stem + ".txt\n";
    }

    // ── CSV: tidy long format ─────────────────────────────────────────────
    std::ofstream csv(folder + file_stem + ".csv");
    if (csv.is_open()) {
        csv << "method,metric,snr,value\n" << std::fixed << std::setprecision(6);
        for (const auto& nm : method_names) {
            for (std::size_t i = 0; i < snr_values.size(); ++i) {
                const auto   ei  = static_cast<Eigen::Index>(i);
                const double snr = snr_values[i];
                csv << nm << ",Usage,"    << snr << "," << usage_map.at(nm)(ei)   << "\n";
                csv << nm << ",FDR,"      << snr << "," << fdp_map.at(nm)(ei)     << "\n";
                csv << nm << ",TPR,"      << snr << "," << tpp_map.at(nm)(ei)     << "\n";
                csv << nm << ",Est. FDR," << snr << "," << est_fdr_map.at(nm)(ei) << "\n";
            }
        }
        csv.close();
        std::cout << "[Info] CSV results saved to: " << folder + file_stem + ".csv\n";
    }
    std::cout << "\n";
}

} // namespace scr_demo

// ==============================================================================
#endif /* End of TREX_DEMOS_TREX_SCREENING_SIMULATION_UTILS_HPP */
// ==============================================================================
