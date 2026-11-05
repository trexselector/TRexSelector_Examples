// ==============================================================================
// mc_workers.hpp
// ==============================================================================
/**
 * @file mc_workers.hpp
 * @brief Thread budget of the Monte Carlo demos.
 *
 * TREX_NUM_WORKERS sets the number of OpenMP threads a demo's Monte Carlo
 * loops use (0 = every processor). Unset, the demos use default_count = 8, the
 * performance cores of the reference machine. The Python and R demos read the
 * same variable with the same default, and the overnight runners export it.
 * The thread count does not change any result, so the cell cache ignores it.
 */
// ==============================================================================
#ifndef DEMOS_TREX_SELECTOR_METHODS_MC_WORKERS_HPP
#define DEMOS_TREX_SELECTOR_METHODS_MC_WORKERS_HPP

// std includes
#include <algorithm>
#include <cstdlib>
#include <thread>

namespace trex_mc_workers {

// ==============================================================================
// Thread budget (TREX_NUM_WORKERS)
// ==============================================================================

/** @brief Default thread count when TREX_NUM_WORKERS is unset. */
inline constexpr int default_count = 8;

/** @brief Thread count from TREX_NUM_WORKERS (0: every processor), else default_count. */
inline int count() {
    const char* env = std::getenv("TREX_NUM_WORKERS");
    if (env == nullptr || *env == '\0') return default_count;
    char* end = nullptr;
    const long n = std::strtol(env, &end, 10);
    if (end == env) return default_count;
    if (n <= 0) return static_cast<int>(std::max(1U, std::thread::hardware_concurrency()));
    return static_cast<int>(n);
}

}  // namespace trex_mc_workers

#endif /* End of DEMOS_TREX_SELECTOR_METHODS_MC_WORKERS_HPP */
// ==============================================================================
