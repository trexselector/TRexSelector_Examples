// ============================================================================
// ts_perf_solvers.hpp
// ============================================================================
/**
 * @file ts_perf_solvers.hpp
 *
 * @brief The T-solvers compared by the performance demos (demo_ts_perf_*),
 *        each with the settings of its own solver demo.
 *
 * @details Every solver runs as a reader (normalize = false, intercept =
 *          false): the caller centers and scales X and D and centers y once in
 *          front, and all solvers share the buffers without copies. The data
 *          are marked precentered, so the row-bounded active sets stop at the
 *          rank n - 1 of the centered design (T-ENET with lambda2 > 0 has no
 *          such cap). T-AFS appears once per rho.
 *
 *          TREX_SOLVERS (comma separated labels, e.g. "T-Stagewise") restricts
 *          demo_solvers() to those solvers, and a selection run writes its
 *          tables with file_suffix(), next to the committed ones of the full
 *          list. merge_result_files.py splices them into the full tables.
 */
// ============================================================================

#ifndef TS_PERF_SOLVERS_HPP
#define TS_PERF_SOLVERS_HPP

// std includes
#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <functional>
#include <sstream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

// Eigen includes
#include <Eigen/Dense>

// tsolvers includes
#include <tsolvers/ad/linear_model/afs_based/tafs_solver.hpp>
#include <tsolvers/ad/linear_model/lars_based/tenet_solver.hpp>
#include <tsolvers/ad/linear_model/lars_based/tlars_solver.hpp>
#include <tsolvers/ad/linear_model/lars_based/tlasso_solver.hpp>
#include <tsolvers/ad/linear_model/lars_based/tstagewise_solver.hpp>
#include <tsolvers/ad/linear_model/lars_based/tstepwise_solver.hpp>
#include <tsolvers/ad/linear_model/omp_based/tacgp_solver.hpp>
#include <tsolvers/ad/linear_model/omp_based/tgp_solver.hpp>
#include <tsolvers/ad/linear_model/omp_based/tmp_solver.hpp>
#include <tsolvers/ad/linear_model/omp_based/tncgmp_solver.hpp>
#include <tsolvers/ad/linear_model/omp_based/tomp_solver.hpp>
#include <tsolvers/ad/linear_model/omp_based/tools_solver.hpp>
#include <tsolvers/ad/tsolver_base.hpp>

namespace ts_perf {

/** @brief Builds one solver on the given (already scaled) buffers. */
using SolverFactory = std::function<std::unique_ptr<trex::tsolvers::ad::TSolver_Base>(
    Eigen::Map<Eigen::MatrixXd>&, Eigen::Map<Eigen::MatrixXd>&, Eigen::Map<Eigen::VectorXd>&)>;

/** @brief Name, family and factory of one compared solver. */
struct SolverSpec {
    std::string name;    ///< Label in tables and figures, e.g. "T-LARS".
    std::string family;  ///< "LARS-based", "OMP-based" or "AFS-based" (library namespace).
    SolverFactory make;  ///< Constructs the solver as a reader.
};

/**
 * @brief Spec for solver class @p Solver with the reader options @p options.
 * @param name    Label in tables and figures.
 * @param family  Library family of the solver.
 * @param options SolverOptions; normalize and intercept false, precentered true.
 */
template <class Solver>
SolverSpec make_spec(std::string name, std::string family, trex::tsolvers::ad::SolverOptions options) {
    return {std::move(name), std::move(family),
            [options](Eigen::Map<Eigen::MatrixXd>& X, Eigen::Map<Eigen::MatrixXd>& D,
                      Eigen::Map<Eigen::VectorXd>& y) {
                return std::make_unique<Solver>(X, D, y, options);
            }};
}

/** @brief Solver labels listed in TREX_SOLVERS (empty: no selection). */
inline std::vector<std::string> selection() {
    std::vector<std::string> out;
    const char* env = std::getenv("TREX_SOLVERS");
    if (env == nullptr) return out;
    std::stringstream ss(env);
    std::string item;
    while (std::getline(ss, item, ',')) {
        // Labels hold inner blanks ("T-AFS (rho 0.3)"), so only the ends are trimmed
        const auto first = item.find_first_not_of(" \t");
        if (first == std::string::npos) continue;
        out.push_back(item.substr(first, item.find_last_not_of(" \t") - first + 1));
    }
    return out;
}

/** @brief True when TREX_SOLVERS selects the label @p name (always true without a selection). */
inline bool selected(const std::string& name) {
    const auto wanted = selection();
    return wanted.empty() || std::find(wanted.begin(), wanted.end(), name) != wanted.end();
}

/**
 * @brief File-name suffix of a selection run, e.g. "_t_stagewise" ("" without a selection).
 * @details Lower case, every run of other characters than letters and digits becomes "_".
 */
inline std::string file_suffix() {
    std::string out;
    for (const auto& name : selection()) {
        out += '_';
        for (const unsigned char c : name) {
            if (std::isalnum(c) != 0) out += static_cast<char>(std::tolower(c));
            else if (out.back() != '_') out += '_';
        }
        while (out.back() == '_' && out.size() > 1) out.pop_back();
    }
    return out;
}

/**
 * @brief The compared solvers: the 12 demo solvers with their demo settings,
 *        T-AFS for rho in {0.1, 0.3, 0.5, 1}; only those TREX_SOLVERS selects.
 */
inline std::vector<SolverSpec> demo_solvers() {
    namespace lars = trex::tsolvers::ad::linear_model::lars_based;
    namespace omp = trex::tsolvers::ad::linear_model::omp_based;
    namespace afs = trex::tsolvers::ad::linear_model::afs_based;
    using Options = trex::tsolvers::ad::SolverOptions;

    const std::string LARS = "LARS-based", OMP = "OMP-based", AFS = "AFS-based";
    std::vector<SolverSpec> all = {
        make_spec<lars::TLARS_Solver>("T-LARS", LARS, Options{.normalize = false, .intercept = false, .precentered = true}),
        make_spec<lars::TLASSO_Solver>("T-LASSO", LARS, Options{.normalize = false, .intercept = false, .precentered = true}),
        make_spec<lars::TSTEPWISE_Solver>("T-Stepwise", LARS, Options{.normalize = false, .intercept = false, .precentered = true}),
        make_spec<lars::TENET_Solver>("T-ENET (lambda2 0.5)", LARS,
                                      Options{.normalize = false, .intercept = false, .precentered = true, .lambda2 = 0.5}),
        make_spec<lars::TSTAGEWISE_Solver>("T-Stagewise", LARS, Options{.normalize = false, .intercept = false, .precentered = true}),
        make_spec<omp::TOMP_Solver>("T-OMP", OMP, Options{.normalize = false, .intercept = false, .precentered = true}),
        make_spec<omp::TGP_Solver>("T-GP", OMP, Options{.normalize = false, .intercept = false, .precentered = true}),
        make_spec<omp::TACGP_Solver>("T-ACGP", OMP, Options{.normalize = false, .intercept = false, .precentered = true}),
        make_spec<omp::TMP_Solver>("T-MP", OMP, Options{.normalize = false, .intercept = false, .precentered = true}),
        make_spec<omp::TNCGMP_Solver>("T-NCGMP", OMP,
                                      Options{.normalize = false, .intercept = false, .precentered = true,
                                              .ncgmp_variant = omp::NCGMPVariant::LineSearch}),
        make_spec<omp::TOOLS_Solver>("T-OLS", OMP, Options{.normalize = false, .intercept = false, .precentered = true}),
        make_spec<afs::TAFS_Solver>("T-AFS (rho 0.1)", AFS,
                                    Options{.normalize = false, .intercept = false, .precentered = true, .rho_afs = 0.1}),
        make_spec<afs::TAFS_Solver>("T-AFS (rho 0.3)", AFS,
                                    Options{.normalize = false, .intercept = false, .precentered = true, .rho_afs = 0.3}),
        make_spec<afs::TAFS_Solver>("T-AFS (rho 0.5)", AFS,
                                    Options{.normalize = false, .intercept = false, .precentered = true, .rho_afs = 0.5}),
        make_spec<afs::TAFS_Solver>("T-AFS (rho 1)", AFS,
                                    Options{.normalize = false, .intercept = false, .precentered = true, .rho_afs = 1.0}),
    };
    std::vector<SolverSpec> out;
    for (auto& spec : all)
        if (selected(spec.name)) out.push_back(std::move(spec));
    return out;
}

} // namespace ts_perf

#endif // TS_PERF_SOLVERS_HPP
