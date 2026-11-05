// ============================================================================
// ts_perf_utils.hpp
// ============================================================================
/**
 * @file ts_perf_utils.hpp
 *
 * @brief Shared helpers for the T-solver demos: result files for the figures
 *        drawn by ts_plt_utils.py.
 *
 * @details Part 5 of every solver demo (terminated solution paths) records
 *          one snapshot per executeStep() call of a warm-started solver into
 *          two tidy CSV files in DEMO_OUTPUT_DIR:
 *
 *            <stem>_path.csv    : T_stop,step,index,is_dummy,beta
 *            <stem>_actions.csv : T_stop,step,index,is_dummy,added
 *
 *          step 0 is the all-zero start, and actions_[k] of the solver leads
 *          to step k + 1. Indices are 0-based as in the console output (the
 *          1-based getActions() entries are converted), the path lists nonzero
 *          coefficients only, on the original scale.
 *
 *          The memory demo reads the solvers' own memory report (memoryBytes())
 *          instead of a peak resident set, so no helper here needs POSIX
 *          getrusage(): portable to Windows, and deterministic for given data.
 */
// ============================================================================

#ifndef TS_PERF_UTILS_HPP
#define TS_PERF_UTILS_HPP

// std includes
#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

// tsolvers includes
#include <tsolvers/ad/tsolver_base.hpp>

namespace ts_perf {

/**
 * @brief Output directory of the demo, created on demand.
 * @return DEMO_OUTPUT_DIR when the build defines it, else simulation_results/data/.
 */
inline std::filesystem::path output_dir() {
#ifdef DEMO_OUTPUT_DIR
    const std::filesystem::path dir(DEMO_OUTPUT_DIR);
#else
    const std::filesystem::path dir("simulation_results/data/");
#endif
    std::filesystem::create_directories(dir);
    return dir;
}

/**
 * @brief Path for console output without the machine-specific prefix.
 * @param path Absolute or relative file path.
 * @return ".../" plus the part from "cpp/" on, else the bare file name.
 */
inline std::string display_path(const std::filesystem::path& path) {
    const std::string s = path.string();
    const auto pos = s.find("/cpp/");
    if (pos != std::string::npos)
        return ".../" + s.substr(pos + 1);
    return path.filename().string();
}

/**
 * @brief Parse an optional positive integer command-line argument.
 * @param text     Argument text, or nullptr when the argument is missing.
 * @param fallback Value when the argument is missing.
 * @throws std::invalid_argument if @p text is not a positive integer.
 */
inline long parse_positive(const char* text, long fallback) {
    if (text == nullptr) return fallback;
    char* end = nullptr;
    const long value = std::strtol(text, &end, 10);
    if (end == text || *end != '\0' || value < 1)
        throw std::invalid_argument(std::string("expected a positive integer, got: ") + text);
    return value;
}

/**
 * @brief Median of @p values (the mean of the two middle values for an even count).
 * @return NaN for an empty input.
 */
inline double median(std::vector<double> values) {
    if (values.empty()) return std::numeric_limits<double>::quiet_NaN();
    const auto mid = values.begin() + static_cast<std::ptrdiff_t>(values.size() / 2);
    std::nth_element(values.begin(), mid, values.end());
    if (values.size() % 2 == 1) return *mid;
    const double upper = *mid;
    const double lower = *std::max_element(values.begin(), mid);
    return 0.5 * (lower + upper);
}

/**
 * @brief Selected original variables: nonzero coefficients among the first p.
 *
 * @details The T-Rex selector and the tlars vignette select this way. The
 *          active set can be smaller: T-Stagewise freezes a variable it drops
 *          from the active set, and the variable keeps its coefficient.
 *
 * @param solver Solver after executeStep().
 * @return 0-based indices of the original variables with a nonzero coefficient.
 */
inline std::vector<std::size_t> selected_predictors(const trex::tsolvers::ad::TSolver_Base& solver) {
    const auto path = solver.getBetaPathSparse();
    std::vector<std::size_t> selected;
    if (path.steps.empty()) return selected;
    const auto& last = path.steps.back();
    const std::size_t p = solver.getDummyStartIndex();
    for (std::size_t k = 0; k < last.idx.size(); ++k)
        if (last.idx[k] < p && last.val[k] != 0.0) selected.push_back(last.idx[k]);
    std::sort(selected.begin(), selected.end());
    return selected;
}

/**
 * @brief Included dummies: nonzero coefficients beyond the first p.
 *
 * @details The count that stops a T-solver at T, the dummies of the support,
 *          as the T-Rex selector counts them. The active set can hold fewer:
 *          T-Stagewise keeps the coefficient of a dummy it drops.
 *
 * @param solver Solver after executeStep().
 * @return Number of dummies with a nonzero coefficient.
 */
inline std::size_t included_dummies(const trex::tsolvers::ad::TSolver_Base& solver) {
    const auto path = solver.getBetaPathSparse();
    if (path.steps.empty()) return 0;
    const auto& last = path.steps.back();
    const std::size_t p = solver.getDummyStartIndex();
    std::size_t count = 0;
    for (std::size_t k = 0; k < last.idx.size(); ++k)
        if (last.idx[k] >= p && last.val[k] != 0.0) ++count;
    return count;
}

/**
 * @brief Parts of a demo selected on the command line.
 *
 * @details `demo_ts_01_tlars 5` runs Part 5 only, `demo_ts_01_tlars 1 3`
 *          Parts 1 and 3; without arguments every part runs.
 */
class PartSelection {
public:
    /**
     * @brief Parse the part numbers from the command line.
     * @throws std::invalid_argument if an argument is not a positive integer.
     */
    PartSelection(int argc, char** argv) {
        for (int i = 1; i < argc; ++i) {
            char* end = nullptr;
            const long part = std::strtol(argv[i], &end, 10);
            if (end == argv[i] || *end != '\0' || part < 1)
                throw std::invalid_argument(std::string("invalid part number: ") + argv[i]);
            parts_.insert(static_cast<int>(part));
        }
    }

    /** @brief True when @p part was selected, or when no part was given at all. */
    bool operator()(int part) const { return parts_.empty() || parts_.count(part) > 0; }

private:
    std::set<int> parts_;
};

/**
 * @brief Writes the terminated solution paths of one warm-started solver.
 *
 * @details Each record() call appends the full path computed so far, labelled
 *          with the T_stop of the preceding executeStep() call.
 */
class PathRecorder {
public:
    /**
     * @brief Open <stem>_path.csv and <stem>_actions.csv and write the headers.
     * @param stem File stem inside output_dir(), e.g. "tlars_terminated_paths".
     */
    explicit PathRecorder(const std::string& stem)
        : path_file_(output_dir() / (stem + "_path.csv")),
          actions_file_(output_dir() / (stem + "_actions.csv")),
          path_out_(path_file_),
          actions_out_(actions_file_) {
        if (!path_out_ || !actions_out_)
            throw std::runtime_error("PathRecorder: cannot open " + path_file_.string());
        path_out_ << "T_stop,step,index,is_dummy,beta\n";
        actions_out_ << "T_stop,step,index,is_dummy,added\n";
        path_out_ << std::setprecision(10);
    }

    /**
     * @brief Append the current path and actions of @p solver.
     * @param solver Solver after executeStep(T_stop).
     * @param T_stop Number of included dummies the path was run to.
     */
    void record(const trex::tsolvers::ad::TSolver_Base& solver, std::size_t T_stop) {
        const std::size_t p = solver.getDummyStartIndex();

        const auto path = solver.getBetaPathSparse();
        for (std::size_t s = 0; s < path.steps.size(); ++s) {
            const auto& step = path.steps[s];
            for (std::size_t k = 0; k < step.idx.size(); ++k) {
                if (step.val[k] == 0.0) continue;
                path_out_ << T_stop << ',' << s << ',' << step.idx[k] << ','
                          << (step.idx[k] >= p ? 1 : 0) << ',' << step.val[k] << '\n';
            }
        }

        const auto& actions = solver.getActions();
        for (std::size_t k = 0; k < actions.size(); ++k) {
            for (const int a : actions[k]) {
                const auto index = static_cast<std::size_t>(std::abs(a)) - 1;
                actions_out_ << T_stop << ',' << k + 1 << ',' << index << ','
                             << (index >= p ? 1 : 0) << ',' << (a > 0 ? 1 : 0) << '\n';
            }
        }
    }

    /** @brief Location of the path CSV. */
    const std::filesystem::path& path_file() const noexcept { return path_file_; }

    /** @brief Location of the actions CSV. */
    const std::filesystem::path& actions_file() const noexcept { return actions_file_; }

private:
    std::filesystem::path path_file_;
    std::filesystem::path actions_file_;
    std::ofstream path_out_;
    std::ofstream actions_out_;
};

} // namespace ts_perf

#endif // TS_PERF_UTILS_HPP
