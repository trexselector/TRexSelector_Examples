// ==============================================================================
// solver_filter.hpp
// ==============================================================================
/**
 * @file solver_filter.hpp
 * @brief Optional solver selection for the Monte Carlo demos.
 *
 * TREX_SOLVERS (comma separated solver names, e.g. "TLARS_STAG") restricts a
 * demo to those solvers. A name is matched against the part after ": " of a
 * row label, so "TLARS_STAG" selects "TREX-DA+BT: TLARS_STAG" as well. Unset
 * or empty selects every solver.
 *
 * A selection run writes its tables to extra files (file_suffix()), so the
 * committed results of the full solver pool stay untouched. The
 * merge_result_files.py of trex_core, trex_da and trex_gvs adds such extra
 * files to the full tables. Python (solver_filter.py) and R (mc_runtime.R)
 * read TREX_SOLVERS the same way.
 */
// ==============================================================================
#ifndef DEMOS_TREX_SELECTOR_METHODS_SOLVER_FILTER_HPP
#define DEMOS_TREX_SELECTOR_METHODS_SOLVER_FILTER_HPP

// std includes
#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <sstream>
#include <string>
#include <vector>

namespace trex_solver_filter {

// ==============================================================================
// Solver selection (TREX_SOLVERS)
// ==============================================================================

/** @brief Solver names listed in TREX_SOLVERS (empty: no selection). */
inline std::vector<std::string> names() {
    std::vector<std::string> out;
    const char* env = std::getenv("TREX_SOLVERS");
    if (env == nullptr) return out;
    std::stringstream ss(env);
    std::string item;
    while (std::getline(ss, item, ',')) {
        item.erase(std::remove_if(item.begin(), item.end(),
                                  [](unsigned char c) { return std::isspace(c) != 0; }),
                   item.end());
        if (!item.empty()) out.push_back(item);
    }
    return out;
}

/** @brief True when a row label's solver is selected (always true without a selection). */
inline bool selected(const std::string& label) {
    const auto wanted = names();
    if (wanted.empty()) return true;
    const auto colon = label.find(": ");
    const std::string solver = colon == std::string::npos ? label : label.substr(colon + 2);
    return std::find(wanted.begin(), wanted.end(), solver) != wanted.end();
}

/** @brief File-name suffix of a selection run, e.g. "_tlars_stag" ("" without a selection). */
inline std::string file_suffix() {
    std::string out;
    for (const auto& name : names()) {
        out += '_';
        for (const unsigned char c : name) out += static_cast<char>(std::tolower(c));
    }
    return out;
}

}  // namespace trex_solver_filter

#endif /* End of DEMOS_TREX_SELECTOR_METHODS_SOLVER_FILTER_HPP */
// ==============================================================================
