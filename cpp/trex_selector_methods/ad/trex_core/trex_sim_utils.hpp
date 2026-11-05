// ==============================================================================
// trex_sim_utils.hpp
// ==============================================================================
/**
 * @file trex_sim_utils.hpp
 *
 * @brief Shared simulation utilities for classic T-Rex demo files.
 *
 * @details Provides all types, output helpers, and MC loop infrastructure
 *          shared by the classic T-Rex demo files, all under namespace trex_sim:
 *
 *   Types:
 *    - TrexDGPData               — data container (owned X, y, true_support)
 *    - TrexGridPointResult       — MC aggregate (avg/sd FDR, TPR, L, T)
 *    - TrexDGPFactory            — factory callable type: seed → TrexDGPData
 *    - TrexMmapDGPFactory        — mmap factory type: seed, X_path, y_path → TrexDGPData
 *    - DemoSolverInfo            — solver descriptor (type, name, tuning parameters)
 *    - MmapFileGuard             — RAII guard: deletes mmap backing files on scope exit
 *    - PrintFn                   — dual console+file output callback type
 *
 *  Output functions:
 *    - make_default_solvers_to_test() — canonical list of 14 T-Rex base solvers
 *    - selected_solvers()             — the solvers TREX_SOLVERS selects
 *    - num_mc()                       — trial count, TREX_NUM_MC for smoke runs
 *    - save_selection_csv()          — writes a per-variable tidy CSV
 *    - print_results()                — formats a single-run selection result block
 *    - save_and_print_results()       — MC summary table + tidy CSV (DemoSolverInfo API)
 *    - save_and_print_mc_results()    — MC summary table + tidy CSV (string-name API)
 *
 *  MC loop functions:
 *    - run_mc_trials_trex()      — parallel MC loop using in-memory TRexSelector
 *    - run_mc_trials_trex_mmap() — parallel MC loop using memory-mapped X
 *    Both are resumable grid cells (../../cell_cache.hpp): a finished cell is
 *    cached and restored by a restarted demo.
 *
 * Parallelism pattern:
 *   #pragma omp parallel for schedule(dynamic)
 *   for (int mc = 0; mc < iMC; ++mc) { ... }
 *
 * Thread safety:
 *   - Each trial owns its data (returned by-value from factory).
 *   - TRexControlParameter is passed by const-ref; TRexSelector copies it by value.
 *   - mmap variant uses per-thread file names derived from omp_get_thread_num().
 *   - WarmStartManager / TRexMemMapManager use TempDirManager::create() which
 *     generates unique directories via std::random_device — safe under OpenMP.
 *
 * @note Demo-internal header — not part of the TRexSelector library interface.
 *       Never include from library code.
 */
// ==============================================================================

#ifndef TREX_DEMOS_TREX_SIM_UTILS_HPP
#define TREX_DEMOS_TREX_SIM_UTILS_HPP

// std includes
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

// Eigen includes
#include <Eigen/Dense>

// OpenMP compatibility layer
#include <utils/openmp/utils_openmp.hpp>

// T-Rex Selector includes
#include <trex_selector_methods/ad/trex_core/trex.hpp>
#include <utils/eval_metrics/utils_eval_rates.hpp>

// Resumable grid cells
#include "../../cell_cache.hpp"
#include "../../mc_workers.hpp"   // thread budget (TREX_NUM_WORKERS)
#include "../../solver_filter.hpp"  // optional solver selection (TREX_SOLVERS)


// ==============================================================================
// Namespace
// ==============================================================================

namespace trex_sim {

namespace rates = trex::utils::eval::rates;
using trex::trex_selector_methods::ad::trex_core::TRexControlParameter;
using trex::trex_selector_methods::ad::trex_core::TRexSelector;
using trex::trex_selector_methods::ad::utils::solver_dispatch::SolverTypeForTRex;


// ==============================================================================
// Data and result types
// ==============================================================================

/** @brief Data container returned by a DGP factory for one MC trial. */
struct TrexDGPData {
    Eigen::MatrixXd          X;
    Eigen::VectorXd          y;
    std::vector<std::size_t> true_support;
};

/** @brief Aggregate MC result for one grid point (solver × parameter value). */
struct TrexGridPointResult {
    double avg_fdr = 0.0;   ///< Mean FDR across MC trials
    double avg_tpr = 0.0;   ///< Mean TPR across MC trials
    double avg_L   = 0.0;   ///< Mean dummy multiplier L
    double avg_T   = 0.0;   ///< Mean stopping time T
    double sd_fdr  = 0.0;   ///< Bessel-corrected sd of FDR
    double sd_tpr  = 0.0;   ///< Bessel-corrected sd of TPR
};

/** @brief Factory callable: given a trial seed, produce one MC dataset. */
using TrexDGPFactory = std::function<TrexDGPData(unsigned seed)>;

/**
 * @brief Factory callable for fully memory-mapped X.
 *
 * Given a trial seed and per-thread file paths, produce one MC dataset with
 * data copied into owned matrices. The factory is responsible for the complete
 * lifecycle of the SyntheticDataMapped object and its backing files (via
 * MmapFileGuard declared before SyntheticDataMapped inside the lambda body,
 * ensuring LIFO cleanup: mmap handles closed before files are removed).
 */
using TrexMmapDGPFactory =
    std::function<TrexDGPData(unsigned seed,
                              const std::string& X_path,
                              const std::string& y_path)>;


// ==============================================================================
// Solver descriptor and RAII helpers
// ==============================================================================

/**
 * @brief Solver descriptor used by demo MC simulations.
 */
struct DemoSolverInfo {

    SolverTypeForTRex solver_type;
    std::string       solver_name;
    double            lambda2       = 0.0;
    double            rho_afs       = 0.0;
    int               ncgmp_variant = 0;
    std::optional<bool> stagnation_stop = std::nullopt;  // nullopt: library AUTO
};


/**
 * @brief RAII guard that removes memory-mapped backing files on scope exit.
 * @note  Declare the guard BEFORE the SyntheticDataMapped object so that
 *        C++ reverse-LIFO destruction closes the mmap handles (data dtor)
 *        before the files are removed (guard dtor). Exception-safe.
 */
struct MmapFileGuard {
    std::vector<std::filesystem::path> paths;
    ~MmapFileGuard() {
        for (const auto& p : paths) {
            std::error_code ec;
            std::filesystem::remove(p, ec);
        }
    }
};


/** @brief Dual console + file output callback type. */
using PrintFn = std::function<void(const std::string&)>;


// ==============================================================================
// Shared solver list
// ==============================================================================

/**
 * @brief Returns the canonical list of 14 T-Rex base solvers used across MC demos.
 *
 * Shared by demo_trex_02_mc_sim_fixed_support.cpp and
 * demo_trex_03_mc_sim_variable_support.cpp.
 */
inline std::vector<DemoSolverInfo> make_default_solvers_to_test() {
    return {
        {SolverTypeForTRex::TLARS,      "TLARS"},
        {SolverTypeForTRex::TLASSO,     "TLASSO"},
        {SolverTypeForTRex::TENET,      "TENET",
            0.1},
        {SolverTypeForTRex::TSTAGEWISE, "TSTAGEWISE"},
        {SolverTypeForTRex::TSTEPWISE,  "TSTEPWISE"},
        {SolverTypeForTRex::TOMP,       "TOMP"},
        {SolverTypeForTRex::TGP,        "TGP"},
        {SolverTypeForTRex::TACGP,      "TACGP"},
        {SolverTypeForTRex::TMP,        "TMP"},
        {SolverTypeForTRex::TAFS,       "TAFS_rho_0.3",
            0.0, 0.3},
        {SolverTypeForTRex::TAFS,       "TAFS_rho_1.0",
            0.0, 1.0},
        {SolverTypeForTRex::TNCGMP,     "TNCGMP_v1",
            0.0, 0.0, 1},
        {SolverTypeForTRex::TNCGMP,     "TNCGMP_v0",
            0.0, 0.0, 0},
        {SolverTypeForTRex::TOOLS,      "TOOLS"}
    };
}

/**
 * @brief The solvers of @p solvers that TREX_SOLVERS selects (all without a selection).
 *
 * A selection run writes its tables with the suffix trex_solver_filter::file_suffix(),
 * and merge_result_files.py adds them to the full tables.
 */
inline std::vector<DemoSolverInfo> selected_solvers(const std::vector<DemoSolverInfo>& solvers) {
    std::vector<DemoSolverInfo> out;
    for (const auto& solver : solvers)
        if (trex_solver_filter::selected(solver.solver_name)) out.push_back(solver);
    return out;
}

/**
 * @brief Monte Carlo trials of a demo: TREX_NUM_MC when set, else @p value.
 *
 * For short smoke runs. TREX_NUM_MC enters the cell-cache fingerprint, so a
 * smoke run never restores the cells of a full run.
 */
inline std::size_t num_mc(std::size_t value) {
    const char* env = std::getenv("TREX_NUM_MC");
    if (env == nullptr || *env == '\0') return value;
    const long n = std::strtol(env, nullptr, 10);
    return n > 0 ? static_cast<std::size_t>(n) : value;
}


// ==============================================================================
// Output functions
// ==============================================================================

/**
 * @brief Write per-variable selection results as a tidy CSV for plotting.
 * @details Columns: variable_index, phi_prime, selected, true_positive
 */
inline void save_selection_csv(
    const std::string&              filepath,
    Eigen::Index                    p,
    const Eigen::VectorXd&          phi_prime,
    const std::vector<std::size_t>& selected_indices,
    const std::vector<std::size_t>& true_support)
{
    std::ofstream csv(filepath);
    if (!csv.is_open()) {
        std::cout << "[Warning] Could not open CSV file: " << filepath << "\n";
        return;
    }
    csv << "variable_index,phi_prime,selected,true_positive\n";
    csv << std::fixed << std::setprecision(6);
    for (Eigen::Index i = 0; i < p; ++i) {
        const auto ui      = static_cast<std::size_t>(i);
        const bool is_sel  = std::find(selected_indices.begin(),
                                       selected_indices.end(), ui)
                             != selected_indices.end();
        const bool is_true = std::find(true_support.begin(),
                                       true_support.end(), ui)
                             != true_support.end();
        csv << i << "," << phi_prime(i) << ","
            << (is_sel ? 1 : 0) << "," << (is_true ? 1 : 0) << "\n";
    }
    std::cout << "[Info] CSV saved to:                  " << filepath << "\n";
}


/**
 * @brief Print and save all selection outputs (indices, metrics, matrices).
 *
 * @param selected_indices  Indices selected by T-Rex.
 * @param true_support      Ground-truth active variables.
 * @param selector          Completed TRexSelector instance (for matrix accessors).
 * @param out               Dual output callback (console + file).
 */
// ==============================================================================
// Sparse printing of the selector's per-variable structures
// ==============================================================================

/**
 * @brief Print only the nonzero entries of a p-vector, with their indices.
 *
 * @details Phi_prime carries one entry per variable, so a dense dump puts p
 *          numbers on a single line (45,000 characters at p = 5000) of which
 *          typically ~0.1% are nonzero. Listing the nonzeros keeps every value
 *          a reader would use — the zeros are implied by absence.
 *          Falls back to the dense form when the vector is not actually sparse,
 *          so the sparse view can never be the longer of the two.
 *
 * @param label  Heading printed above the entries.
 * @param vec    Vector to print.
 * @param out    Sink (console and/or file).
 * @param eps    Magnitudes at or below this count as zero.
 */
inline void print_sparse_vector(const std::string& label,
                                const Eigen::VectorXd& vec,
                                const PrintFn& out,
                                double eps = 1e-12)
{
    const Eigen::Index n = vec.size();
    Eigen::Index nnz = 0;
    for (Eigen::Index i = 0; i < n; ++i)
        if (std::abs(vec(i)) > eps) ++nnz;

    std::ostringstream ss;
    ss << "\n" << label << ": " << nnz << " of " << n << " entries nonzero\n";

    if (nnz * 4 > n) {                      // dense enough: keep the plain dump
        ss << vec.transpose() << "\n";
        out(ss.str());
        return;
    }

    constexpr int values_per_line = 5;
    int on_line = 0;
    ss << std::fixed << std::setprecision(6);
    for (Eigen::Index i = 0; i < n; ++i) {
        if (std::abs(vec(i)) <= eps) continue;
        std::ostringstream cell;
        cell << "[" << i << "] " << std::fixed << std::setprecision(6) << vec(i);
        ss << "  " << std::left << std::setw(20) << cell.str();
        if (++on_line == values_per_line) { ss << "\n"; on_line = 0; }
    }
    if (on_line != 0) ss << "\n";
    out(ss.str());
}

/**
 * @brief Print only the nonzero entries of a matrix, row by row.
 *
 * @details Same rationale as print_sparse_vector(): the Phi matrix is
 *          (T x p), so each of its rows is as wide as the variable count.
 *
 * @param label  Heading printed above the rows.
 * @param mat    Matrix to print.
 * @param out    Sink (console and/or file).
 * @param eps    Magnitudes at or below this count as zero.
 */
inline void print_sparse_matrix(const std::string& label,
                                const Eigen::MatrixXd& mat,
                                const PrintFn& out,
                                double eps = 1e-12)
{
    const Eigen::Index rows = mat.rows(), cols = mat.cols();
    const Eigen::Index total = rows * cols;
    Eigen::Index nnz = 0;
    for (Eigen::Index r = 0; r < rows; ++r)
        for (Eigen::Index c = 0; c < cols; ++c)
            if (std::abs(mat(r, c)) > eps) ++nnz;

    std::ostringstream ss;
    ss << "\n" << label << ": " << rows << " x " << cols << ", "
       << nnz << " of " << total << " entries nonzero\n";

    if (nnz * 4 > total) {                  // dense enough: keep the plain dump
        ss << mat << "\n";
        out(ss.str());
        return;
    }

    for (Eigen::Index r = 0; r < rows; ++r) {
        ss << "  row " << r << ":";
        bool any = false;
        for (Eigen::Index c = 0; c < cols; ++c) {
            if (std::abs(mat(r, c)) <= eps) continue;
            std::ostringstream cell;
            cell << "[" << c << "] " << std::fixed << std::setprecision(4)
                 << mat(r, c);
            ss << "  " << cell.str();
            any = true;
        }
        if (!any) ss << "  (all zero)";
        ss << "\n";
    }
    out(ss.str());
}


inline void print_results(
    const std::vector<std::size_t>& selected_indices,
    const std::vector<std::size_t>& true_support,
    TRexSelector&                   selector,
    const PrintFn&                  out)
{
    {
        std::ostringstream ss;
        ss << "Selected indices: ";
        for (const auto& idx : selected_indices) { ss << idx << " "; }
        ss << "\n";
        out(ss.str());
    }

    const double fdp = rates::compute_fdp(selected_indices, true_support);
    const double tpp = rates::compute_tpp(selected_indices, true_support);
    {
        std::ostringstream ss;
        ss << std::fixed << std::setprecision(4);
        ss << "False Discovery Proportion (FDP): " << fdp << "\n";
        ss << "True Positive Proportion (TPP):   " << tpp << "\n";
        out(ss.str());
    }

    print_sparse_vector("Adjusted Relative Occurrences (Phi_prime)",
                        selector.getPhiPrime(), out);

    print_sparse_matrix("Phi Matrix (Phi)", selector.getPhiMat(), out);

    {
        std::ostringstream ss;
        ss << "\nEstimated FDP Matrix (FDP_hat):\n" << selector.getFDPHatMat() << "\n";
        out(ss.str());
    }

    {
        std::ostringstream ss;
        ss << "\nR Matrix (R):\n" << selector.getRMat() << "\n";
        out(ss.str());
    }

    {
        std::ostringstream ss;
        ss << "\nVoting Grid:\n" << selector.getVotingGrid().transpose() << "\n";
        out(ss.str());
    }
}


/**
 * @brief Print averaged MC results as an aligned table (console + .txt) and
 *        write a tidy long-format CSV. Uses DemoSolverInfo for solver metadata.
 *
 * @param num_MC               Number of Monte Carlo repetitions.
 * @param n                    Number of observations.
 * @param p                    Number of predictors.
 * @param stagnation_window    Value of tloop_max_stagnant_steps (used in filename).
 * @param snr_values           SNR grid.
 * @param solvers_to_test      Solver descriptors (determines row order).
 * @param fdr_results_map      Averaged FDR per solver x SNR.
 * @param tpr_results_map      Averaged TPR per solver x SNR.
 * @param avg_L_results_map    Averaged dummy multiplier L (empty = omit).
 * @param avg_T_results_map    Averaged stopping time T   (empty = omit).
 */
inline void save_and_print_results(
    std::size_t num_MC,
    std::size_t n,
    std::size_t p,
    std::size_t stagnation_window,
    const std::vector<double>&                     snr_values,
    const std::vector<DemoSolverInfo>&             solvers_to_test,
    const std::map<std::string, Eigen::VectorXd>&  fdr_results_map,
    const std::map<std::string, Eigen::VectorXd>&  tpr_results_map,
    const std::map<std::string, Eigen::VectorXd>&  avg_L_results_map,
    const std::map<std::string, Eigen::VectorXd>&  avg_T_results_map,
    const std::string&                             file_prefix = "")
{
    // 1. Setup File Output
    // -----------------------------------
    // A TREX_SOLVERS selection writes to suffixed files (solver_filter.hpp)
    std::string folder = DEMO_OUTPUT_DIR;
    std::filesystem::create_directories(folder);
    const std::string stem = file_prefix + "trex_results_n" + std::to_string(n) + "_p" +
                             std::to_string(p) + "_stagnation_window_" +
                             std::to_string(stagnation_window) + trex_solver_filter::file_suffix();
    std::string filename = stem + ".txt";

    std::ofstream out_file(folder + filename);

    auto print_dual = [&](const std::string& text) {
        std::cout << text;
        if (out_file.is_open()) out_file << text;
    };

    // 2. Print Headers
    // -----------------------------------
    std::stringstream ss;
    ss << "\n";
    ss << "======================================================================\n";
    ss << "=== T-Rex Results (averaged over " << num_MC << " Monte Carlo runs) ===\n";
    ss << "======================================================================\n\n";
    print_dual(ss.str());

    // 3. Setup Table Dimensions
    // -----------------------------------
    // Layout: solver name on its own line, metric rows indented beneath it.
    const int indent_w = 2;
    const int metric_w = 23;
    const int col_w    = 10;
    const std::size_t sep_w =
        static_cast<std::size_t>(indent_w + metric_w)
        + col_w * snr_values.size();

    // 4. Print Table Header
    // -----------------------------------
    {
        std::stringstream ss_header;
        ss_header << std::left
                  << std::string(static_cast<std::size_t>(indent_w), ' ')
                  << std::setw(metric_w) << "SNR";
        for (double snr : snr_values) {
            ss_header << std::right << std::fixed << std::setprecision(2)
                      << std::setw(col_w) << snr;
        }
        ss_header << "\n" << std::string(sep_w, '-') << "\n";
        print_dual(ss_header.str());
    }

    auto print_metric_row = [&](const std::string& metric_name,
                                const Eigen::VectorXd& data) {
        std::stringstream ss_row;
        ss_row << std::left
               << std::string(static_cast<std::size_t>(indent_w), ' ')
               << std::setw(metric_w) << metric_name;
        for (Eigen::Index i = 0; i < static_cast<Eigen::Index>(snr_values.size()); ++i) {
            ss_row << std::right << std::fixed << std::setprecision(4)
                   << std::setw(col_w) << data(i);
        }
        ss_row << "\n";
        print_dual(ss_row.str());
    };

    // 5. Print Data Rows
    // -----------------------------------
    for (const auto& solver : solvers_to_test) {
        const std::string& name = solver.solver_name;

        print_dual("\n" + name + "\n");          // solver name on its own line
        print_metric_row("FDR", fdr_results_map.at(name));
        print_metric_row("TPR", tpr_results_map.at(name));

        if (avg_L_results_map.count(name) > 0) {
            print_metric_row("Avg L", avg_L_results_map.at(name));
        }
        if (avg_T_results_map.count(name) > 0) {
            print_metric_row("Avg T", avg_T_results_map.at(name));
        }
    }
    print_dual("\n");

    // 6. Write tidy CSV (long/stacked format) for plotting
    // -----------------------------------
    std::string csv_filename = stem + ".csv";
    std::ofstream csv_file(folder + csv_filename);
    if (csv_file.is_open()) {
        csv_file << "solver,metric,snr,value\n";
        for (const auto& solver : solvers_to_test) {
            const std::string& name = solver.solver_name;
            for (std::size_t i = 0; i < snr_values.size(); ++i) {
                const auto ei = static_cast<Eigen::Index>(i);
                std::ostringstream snr_ss;
                snr_ss << std::fixed << std::setprecision(6) << snr_values[i];
                const std::string snr_str = snr_ss.str();
                csv_file << std::fixed << std::setprecision(6);
                csv_file << name << ",FDR," << snr_str << ","
                         << fdr_results_map.at(name)(ei) << "\n";
                csv_file << name << ",TPR," << snr_str << ","
                         << tpr_results_map.at(name)(ei) << "\n";
                if (avg_L_results_map.count(name) > 0)
                    csv_file << name << ",AvgL," << snr_str << ","
                             << avg_L_results_map.at(name)(ei) << "\n";
                if (avg_T_results_map.count(name) > 0)
                    csv_file << name << ",AvgT," << snr_str << ","
                             << avg_T_results_map.at(name)(ei) << "\n";
            }
        }
        std::cout << "[Info] CSV results saved to:             " << folder + csv_filename << "\n";
    } else {
        std::cout << "[Warning] Could not open CSV file: " << folder + csv_filename << "\n";
    }

    // 7. Footer (text file)
    // -----------------------------------
    if (out_file.is_open()) {
        std::cout << "[Info] Results successfully saved to: " << folder + filename << "\n\n";
        out_file.close();
    } else {
        std::cout << "[Warning] Could not open output file: " << folder + filename << "\n\n";
    }
}


/**
 * @brief Print averaged MC results as an aligned table (console + .txt) and
 *        write a tidy long-format CSV. Uses string names for solvers.
 *
 * @param num_MC             Number of Monte Carlo repetitions used.
 * @param file_stem          Output file base name (no folder, no extension).
 * @param snr_values         SNR grid used during the simulation.
 * @param solver_names       Ordered list of solver names (keys into result maps).
 * @param fdr_results_map    Averaged FDR per solver, indexed by SNR.
 * @param tpr_results_map    Averaged TPR per solver, indexed by SNR.
 * @param avg_L_results_map  Averaged dummy multiplier L (empty map = omit).
 * @param avg_T_results_map  Averaged stopping time T  (empty map = omit).
 */
inline void save_and_print_mc_results(
    std::size_t                                    num_MC,
    const std::string&                             file_stem,
    const std::vector<double>&                     snr_values,
    const std::vector<std::string>&                solver_names,
    const std::map<std::string, Eigen::VectorXd>&  fdr_results_map,
    const std::map<std::string, Eigen::VectorXd>&  tpr_results_map,
    const std::map<std::string, Eigen::VectorXd>&  avg_L_results_map,
    const std::map<std::string, Eigen::VectorXd>&  avg_T_results_map)
{
    const std::string folder = DEMO_OUTPUT_DIR;
    std::filesystem::create_directories(folder);

    // 1. Open text file (dual output)
    std::ofstream out_file(folder + file_stem + ".txt");
    auto print_dual = [&](const std::string& text) {
        std::cout << text;
        if (out_file.is_open()) out_file << text;
    };

    // 2. Header
    {
        std::ostringstream ss;
        ss << "\n"
           << "======================================================================\n"
           << "=== T-Rex Results (averaged over " << num_MC << " Monte Carlo runs) ===\n"
           << "======================================================================\n\n";
        print_dual(ss.str());
    }

    // 3. Table dimensions
    //    The solver name gets its own line; metric rows are indented beneath
    //    it, so the value columns stay aligned no matter how long a name is.
    const int indent_w = 2;    // leading indent of a metric row
    const int metric_w = 23;   // left-aligned metric label
    const int col_w    = 10;   // right-aligned values
    const std::size_t sep_w =
        static_cast<std::size_t>(indent_w + metric_w)
        + col_w * snr_values.size();

    // 4. Column header (sweep axis) + dashed separator
    {
        std::ostringstream hdr;
        hdr << std::left << std::string(static_cast<std::size_t>(indent_w), ' ')
            << std::setw(metric_w) << "SNR";
        for (double snr : snr_values)
            hdr << std::right << std::fixed << std::setprecision(2)
                << std::setw(col_w) << snr;
        hdr << "\n" << std::string(sep_w, '-') << "\n";
        print_dual(hdr.str());
    }

    // 5. Data rows
    auto print_metric = [&](const std::string& metric,
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

    for (const auto& name : solver_names) {
        print_dual("\n" + name + "\n");          // solver name on its own line
        print_metric("FDR", fdr_results_map.at(name));
        print_metric("TPR", tpr_results_map.at(name));
        if (avg_L_results_map.count(name) > 0)
            print_metric("Avg L", avg_L_results_map.at(name));
        if (avg_T_results_map.count(name) > 0)
            print_metric("Avg T", avg_T_results_map.at(name));
    }
    print_dual("\n");

    // 6. Tidy long-format CSV (solver, metric, snr, value)
    std::ofstream csv(folder + file_stem + ".csv");
    if (csv.is_open()) {
        csv << "solver,metric,snr,value\n" << std::fixed << std::setprecision(6);
        for (const auto& name : solver_names) {
            for (std::size_t i = 0; i < snr_values.size(); ++i) {
                const auto ei = static_cast<Eigen::Index>(i);
                std::ostringstream snr_ss;
                snr_ss << std::fixed << std::setprecision(6) << snr_values[i];
                const std::string s = snr_ss.str();
                csv << name << ",FDR,"  << s << ","
                    << fdr_results_map.at(name)(ei) << "\n";
                csv << name << ",TPR,"  << s << ","
                    << tpr_results_map.at(name)(ei) << "\n";
                if (avg_L_results_map.count(name) > 0)
                    csv << name << ",AvgL," << s << ","
                        << avg_L_results_map.at(name)(ei) << "\n";
                if (avg_T_results_map.count(name) > 0)
                    csv << name << ",AvgT," << s << ","
                        << avg_T_results_map.at(name)(ei) << "\n";
            }
        }
        std::cout << "[Info] CSV results saved to:             "
                  << folder + file_stem + ".csv\n";
    } else {
        std::cout << "[Warning] Could not open CSV file: " << folder + file_stem + ".csv\n";
    }

    // 7. Footer
    if (out_file.is_open()) {
        std::cout << "[Info] Results successfully saved to: " << folder + file_stem + ".txt\n\n";
        out_file.close();
    } else {
        std::cout << "[Warning] Could not open output file: " << folder + file_stem + ".txt\n\n";
    }
}


// ==============================================================================
// Grid-cell cache helpers
// ==============================================================================

namespace detail {

/** @brief Cell key: label plus the parameters that tell cells apart. */
inline std::string cell_key(const std::string& kind, const std::string& label, std::size_t num_MC,
                            unsigned base_seed_offset, double tFDR, const TRexControlParameter& ctrl) {
    std::ostringstream os;
    os << kind << '|' << label << "|solver=" << static_cast<int>(ctrl.solver_type)
       << "|stag=" << (ctrl.tloop_stagnation_stop.has_value() ? (*ctrl.tloop_stagnation_stop ? "1" : "0") : "auto")
       << "|l2=" << ctrl.solver_params.lambda2 << "|afs=" << ctrl.solver_params.rho_afs
       << "|ncgmp=" << ctrl.solver_params.ncgmp_variant
       << "|seed=" << base_seed_offset << "|mc=" << num_MC << "|tfdr=" << tFDR;
    return os.str();
}

inline trex_cell_cache::Cell to_cell(const TrexGridPointResult& r) {
    return {{"avg_fdr", r.avg_fdr}, {"avg_tpr", r.avg_tpr}, {"avg_L", r.avg_L},
            {"avg_T", r.avg_T}, {"sd_fdr", r.sd_fdr}, {"sd_tpr", r.sd_tpr}};
}

inline TrexGridPointResult from_cell(const trex_cell_cache::Cell& c) {
    using trex_cell_cache::get;
    return {get(c, "avg_fdr"), get(c, "avg_tpr"), get(c, "avg_L"),
            get(c, "avg_T"), get(c, "sd_fdr"), get(c, "sd_tpr")};
}

}  // namespace detail


// ==============================================================================
// Parallel MC inner loop — in-memory DGP
// ==============================================================================

/** @brief Uncached body of run_mc_trials_trex(). */
inline TrexGridPointResult run_mc_trials_trex_impl(
    std::size_t                  num_MC,
    const std::string&           progress_label,
    const TrexDGPFactory&        make_data,
    double                       tFDR,
    const TRexControlParameter&  trex_ctrl,
    unsigned                     base_seed_offset,
    bool                         seeded_selector = false)
{
    const int iMC = static_cast<int>(num_MC);

    std::vector<double> fdp_vec(num_MC, 0.0);
    std::vector<double> tpp_vec(num_MC, 0.0);
    std::vector<double> L_vec  (num_MC, 0.0);
    std::vector<double> T_vec  (num_MC, 0.0);

    std::cout << "  " << progress_label
              << " \u2014 Running " << num_MC << " MC trials ...\n" << std::flush;

    #pragma omp parallel for schedule(dynamic)
    for (int mc = 0; mc < iMC; ++mc) {
        const unsigned trial_seed = base_seed_offset + static_cast<unsigned>(mc);
        auto dat = make_data(trial_seed);

        Eigen::Map<Eigen::MatrixXd> X_map(dat.X.data(), dat.X.rows(), dat.X.cols());
        Eigen::Map<Eigen::VectorXd> y_map(dat.y.data(), dat.y.rows());

        // Selector seed: the trial seed when seeded, else hardware entropy (-1)
        TRexSelector selector(X_map, y_map, tFDR, trex_ctrl,
                              seeded_selector ? static_cast<int>(trial_seed) : -1, false);
        selector.select();

        const auto sel  = selector.getSelectedIndices();
        fdp_vec[mc] = rates::compute_fdp(sel, dat.true_support);
        tpp_vec[mc] = rates::compute_tpp(sel, dat.true_support);
        L_vec[mc]   = static_cast<double>(selector.getDummyMultiplierL());
        T_vec[mc]   = static_cast<double>(selector.getTStop());
    }

    // Compute means
    const double dMC = static_cast<double>(num_MC);
    double avg_fdr = 0.0, avg_tpr = 0.0, avg_L = 0.0, avg_T = 0.0;
    for (int mc = 0; mc < iMC; ++mc) {
        avg_fdr += fdp_vec[mc];
        avg_tpr += tpp_vec[mc];
        avg_L   += L_vec[mc];
        avg_T   += T_vec[mc];
    }
    avg_fdr /= dMC;  avg_tpr /= dMC;  avg_L /= dMC;  avg_T /= dMC;

    // Bessel-corrected standard deviations
    double sd_fdr = 0.0, sd_tpr = 0.0;
    if (num_MC > 1) {
        double ssq_fdr = 0.0, ssq_tpr = 0.0;
        for (int mc = 0; mc < iMC; ++mc) {
            ssq_fdr += (fdp_vec[mc] - avg_fdr) * (fdp_vec[mc] - avg_fdr);
            ssq_tpr += (tpp_vec[mc] - avg_tpr) * (tpp_vec[mc] - avg_tpr);
        }
        sd_fdr = std::sqrt(ssq_fdr / (dMC - 1.0));
        sd_tpr = std::sqrt(ssq_tpr / (dMC - 1.0));
    }

    std::cout << "  " << progress_label
              << " \u2014 done. TPP=" << std::fixed << std::setprecision(3) << avg_tpr
              << "  FDR=" << avg_fdr << "\n\n" << std::flush;

    return {avg_fdr, avg_tpr, avg_L, avg_T, sd_fdr, sd_tpr};
}

/**
 * @brief Run num_MC parallel MC trials using TRexSelector (resumable grid cell).
 *
 * @param num_MC            Number of Monte Carlo trials.
 * @param progress_label    Label printed in the start / done lines.
 * @param make_data         Factory: seed → TrexDGPData (called per trial).
 * @param tFDR              Target FDR for the selector.
 * @param trex_ctrl         T-Rex control parameters (read-only; TRexSelector copies
 *                          by value so multiple threads are safe).
 * @param base_seed_offset  Base RNG seed offset; trial mc uses base_seed_offset + mc.
 * @param seeded_selector   true: the selector is seeded with the trial seed, so a cell
 *                          is reproducible; false (default): hardware entropy (-1).
 *
 * @return TrexGridPointResult with avg/sd FDR, TPR and avg L, T.
 */
inline TrexGridPointResult run_mc_trials_trex(
    std::size_t                  num_MC,
    const std::string&           progress_label,
    const TrexDGPFactory&        make_data,
    double                       tFDR,
    const TRexControlParameter&  trex_ctrl,
    unsigned                     base_seed_offset,
    bool                         seeded_selector = false)
{
    return detail::from_cell(TREX_CELL_CACHED(
        detail::cell_key(seeded_selector ? "trex_seeded" : "trex", progress_label, num_MC,
                         base_seed_offset, tFDR, trex_ctrl), [&] {
            return detail::to_cell(run_mc_trials_trex_impl(num_MC, progress_label, make_data,
                                                           tFDR, trex_ctrl, base_seed_offset,
                                                           seeded_selector));
        }));
}


// ==============================================================================
// Parallel MC inner loop — fully memory-mapped X
// ==============================================================================

/** @brief Uncached body of run_mc_trials_trex_mmap(). */
inline TrexGridPointResult run_mc_trials_trex_mmap_impl(
    std::size_t                  num_MC,
    const std::string&           progress_label,
    const TrexMmapDGPFactory&    make_data,
    double                       tFDR,
    const TRexControlParameter&  trex_ctrl,
    unsigned                     base_seed_offset,
    const std::string&           mmap_filestem)
{
    const int iMC = static_cast<int>(num_MC);

    std::vector<double> fdp_vec(num_MC, 0.0);
    std::vector<double> tpp_vec(num_MC, 0.0);
    std::vector<double> L_vec  (num_MC, 0.0);
    std::vector<double> T_vec  (num_MC, 0.0);

    std::cout << "  " << progress_label
              << " \u2014 Running " << num_MC << " MC trials (mmap) ...\n" << std::flush;

    #pragma omp parallel for schedule(dynamic)
    for (int mc = 0; mc < iMC; ++mc) {
        const int    tid      = omp_get_thread_num();
        const std::string X_path = mmap_filestem + "_X_tid" + std::to_string(tid) + ".dat";
        const std::string y_path = mmap_filestem + "_y_tid" + std::to_string(tid) + ".dat";

        const unsigned trial_seed = base_seed_offset + static_cast<unsigned>(mc);
        auto dat = make_data(trial_seed, X_path, y_path);

        Eigen::Map<Eigen::MatrixXd> X_map(dat.X.data(), dat.X.rows(), dat.X.cols());
        Eigen::Map<Eigen::VectorXd> y_map(dat.y.data(), dat.y.rows());

        TRexSelector selector(X_map, y_map, tFDR, trex_ctrl, -1, false);
        selector.select();

        const auto sel  = selector.getSelectedIndices();
        fdp_vec[mc] = rates::compute_fdp(sel, dat.true_support);
        tpp_vec[mc] = rates::compute_tpp(sel, dat.true_support);
        L_vec[mc]   = static_cast<double>(selector.getDummyMultiplierL());
        T_vec[mc]   = static_cast<double>(selector.getTStop());
    }

    // Compute means
    const double dMC = static_cast<double>(num_MC);
    double avg_fdr = 0.0, avg_tpr = 0.0, avg_L = 0.0, avg_T = 0.0;
    for (int mc = 0; mc < iMC; ++mc) {
        avg_fdr += fdp_vec[mc];
        avg_tpr += tpp_vec[mc];
        avg_L   += L_vec[mc];
        avg_T   += T_vec[mc];
    }
    avg_fdr /= dMC;  avg_tpr /= dMC;  avg_L /= dMC;  avg_T /= dMC;

    // Bessel-corrected standard deviations
    double sd_fdr = 0.0, sd_tpr = 0.0;
    if (num_MC > 1) {
        double ssq_fdr = 0.0, ssq_tpr = 0.0;
        for (int mc = 0; mc < iMC; ++mc) {
            ssq_fdr += (fdp_vec[mc] - avg_fdr) * (fdp_vec[mc] - avg_fdr);
            ssq_tpr += (tpp_vec[mc] - avg_tpr) * (tpp_vec[mc] - avg_tpr);
        }
        sd_fdr = std::sqrt(ssq_fdr / (dMC - 1.0));
        sd_tpr = std::sqrt(ssq_tpr / (dMC - 1.0));
    }

    std::cout << "  " << progress_label
              << " \u2014 done. TPP=" << std::fixed << std::setprecision(3) << avg_tpr
              << "  FDR=" << avg_fdr << "\n\n" << std::flush;

    return {avg_fdr, avg_tpr, avg_L, avg_T, sd_fdr, sd_tpr};
}

/**
 * @brief Run num_MC parallel MC trials using TRexSelector with memory-mapped X
 *        (resumable grid cell).
 *
 * Each OpenMP thread uses unique backing-file paths derived from mmap_filestem
 * and the thread ID (omp_get_thread_num()), preventing file-name collisions.
 * The factory lambda is responsible for the full lifecycle of SyntheticDataMapped
 * and its files: declare MmapFileGuard BEFORE SyntheticDataMapped inside the
 * lambda body so LIFO destruction closes mmap handles before removing files.
 *
 * @param num_MC            Number of Monte Carlo trials.
 * @param progress_label    Label printed in the start / done lines.
 * @param make_data         Factory: (seed, X_path, y_path) \u2192 TrexDGPData.
 *                          Must copy X and y into owned matrices and clean up files.
 * @param tFDR              Target FDR for the selector.
 * @param trex_ctrl         T-Rex control parameters (read-only; TRexSelector copies
 *                          by value so multiple threads are safe).
 * @param base_seed_offset  Base RNG seed offset; trial mc uses base_seed_offset + mc.
 * @param mmap_filestem     Stem for per-thread backing files.
 *                          Thread t uses: filestem_X_tid{t}.dat / filestem_y_tid{t}.dat
 *
 * @return TrexGridPointResult with avg/sd FDR, TPR and avg L, T.
 */
inline TrexGridPointResult run_mc_trials_trex_mmap(
    std::size_t                  num_MC,
    const std::string&           progress_label,
    const TrexMmapDGPFactory&    make_data,
    double                       tFDR,
    const TRexControlParameter&  trex_ctrl,
    unsigned                     base_seed_offset,
    const std::string&           mmap_filestem)
{
    return detail::from_cell(TREX_CELL_CACHED(
        detail::cell_key("trex_mmap", progress_label, num_MC, base_seed_offset, tFDR, trex_ctrl), [&] {
            return detail::to_cell(run_mc_trials_trex_mmap_impl(num_MC, progress_label, make_data, tFDR,
                                                                trex_ctrl, base_seed_offset, mmap_filestem));
        }));
}

} // namespace trex_sim

// ==============================================================================
#endif /* End of TREX_DEMOS_TREX_SIM_UTILS_HPP */
// ==============================================================================
