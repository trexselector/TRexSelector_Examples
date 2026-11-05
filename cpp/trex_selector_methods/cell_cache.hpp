// ==============================================================================
// cell_cache.hpp
// ==============================================================================
/**
 * @file cell_cache.hpp
 * @brief Resumable grid cells for the Monte Carlo demos.
 *
 * Every finished cell (one method at one grid point) is appended to
 *   <DEMO_OUTPUT_DIR>/../cache/cells.tsv
 * and a restarted demo reads it back instead of recomputing the cell. Same
 * file format as the Python and R demos:
 *
 *   # <fingerprint>
 *   <key> \t name=value \t ... \t .
 *
 * The fingerprint holds the build time of the demo, the run tag TREX_RUN_TAG
 * (the library pin, set by the runners) and the other TREX_* variables that
 * can change results (not TREX_FRESH, TREX_NUM_WORKERS, TREX_SOLVERS or a
 * *_PARTS selection). A cache with another fingerprint, or any cache under
 * TREX_FRESH=1, is set aside as cells.tsv.stale. A line without the closing
 * "." marker (cut off by a hard stop) is recomputed. The cache is removed when
 * the demo returns from main(), i.e. once all results are written; a crash or
 * a kill leaves it for the restart.
 *
 * Single-threaded use: call it from the thread that drives the Monte Carlo
 * loops, never from inside an OpenMP region.
 */
// ==============================================================================
#ifndef DEMOS_TREX_SELECTOR_METHODS_CELL_CACHE_HPP
#define DEMOS_TREX_SELECTOR_METHODS_CELL_CACHE_HPP

// std includes
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// Process environment (POSIX environ, Windows _environ)
#if defined(_WIN32)
#define TREX_CELL_CACHE_ENVIRON _environ
#else
extern char** environ;
#define TREX_CELL_CACHE_ENVIRON environ
#endif

namespace trex_cell_cache {

// ==============================================================================
// Cell type
// ==============================================================================

/** @brief One cell: named numbers in a fixed order. */
using Cell = std::vector<std::pair<std::string, double>>;

inline void finish();

// ==============================================================================
// Cache file (internal)
// ==============================================================================

namespace detail {

struct State {
    bool opened = false;
    bool enabled = false;
    std::string path;
    std::map<std::string, Cell> cells;
    std::map<std::string, int> used;
};

inline State& state() {
    static State s;
    return s;
}

/** @brief Fingerprint: build time of this demo plus the TREX_* environment. */
inline std::string fingerprint(const char* build_time) {
    std::vector<std::string> env;
    for (char** e = TREX_CELL_CACHE_ENVIRON; e != nullptr && *e != nullptr; ++e) {
        const std::string kv(*e);
        if (kv.rfind("TREX_", 0) != 0) continue;
        // Worker count, fresh switch, solver and part selection do not change any result.
        const std::string name = kv.substr(0, kv.find('='));
        if (name == "TREX_FRESH" || name == "TREX_NUM_WORKERS" || name == "TREX_SOLVERS") continue;
        if (name.size() >= 6 && name.compare(name.size() - 6, 6, "_PARTS") == 0) continue;
        env.push_back(kv);
    }
    std::sort(env.begin(), env.end());
    std::ostringstream os;
    os << "build=" << build_time;
    for (const auto& kv : env) os << ' ' << kv;
    return os.str();
}

/** @brief Parse "key \t n=v ... \t ." into a cell; false for a cut-off line. */
inline bool parse_line(const std::string& line, std::string& key, Cell& cell) {
    std::vector<std::string> parts;
    std::size_t start = 0;
    for (;;) {
        const std::size_t tab = line.find('\t', start);
        parts.push_back(line.substr(start, tab == std::string::npos ? std::string::npos : tab - start));
        if (tab == std::string::npos) break;
        start = tab + 1;
    }
    if (parts.size() < 3 || parts.back() != ".") return false;
    key = parts.front();
    cell.clear();
    for (std::size_t i = 1; i + 1 < parts.size(); ++i) {
        const std::size_t eq = parts[i].find('=');
        if (eq == std::string::npos) return false;
        cell.emplace_back(parts[i].substr(0, eq), std::strtod(parts[i].c_str() + eq + 1, nullptr));
    }
    return true;
}

/** @brief Open the cache of this demo on first use. */
inline void open(const char* output_dir, const char* build_time) {
    State& s = state();
    if (s.opened) return;
    s.opened = true;
    namespace fs = std::filesystem;
    fs::path dir = fs::path(output_dir).parent_path();          // .../simulation_results/data
    if (dir.filename() == "data") dir = dir.parent_path();      // .../simulation_results
    dir /= "cache";
    std::error_code ec;
    fs::create_directories(dir, ec);
    if (ec) {
        std::cerr << "[cache] cannot create " << dir << ", running without cache\n";
        return;
    }
    s.path = (dir / "cells.tsv").string();
    const std::string header = "# " + fingerprint(build_time);
    const char* fresh_env = std::getenv("TREX_FRESH");
    const bool fresh = fresh_env != nullptr && std::string(fresh_env) == "1";

    if (fs::exists(s.path)) {
        std::ifstream in(s.path);
        std::string line;
        std::getline(in, line);
        if (!fresh && line == header) {
            std::string key;
            Cell cell;
            while (std::getline(in, line)) {
                if (parse_line(line, key, cell)) s.cells[key] = cell;
            }
            std::cout << "[cache] " << s.cells.size() << " finished cell(s) restored from " << s.path
                      << "\n\n";
        } else {
            in.close();
            fs::rename(s.path, s.path + ".stale", ec);
            std::cout << "[cache] " << (fresh ? "TREX_FRESH=1" : "build or settings changed")
                      << ", old cache kept as cells.tsv.stale\n\n";
        }
    }
    if (!fs::exists(s.path)) {
        std::ofstream out(s.path);
        out << header << '\n';
    }
    s.enabled = true;
    // A demo that returns from main() has written all its results.
    std::atexit([] { finish(); });
}

}  // namespace detail

// ==============================================================================
// Cached cells
// ==============================================================================

/**
 * @brief Result of one cell: from the cache when present, else computed and appended.
 *
 * Repeats of the same key in one run are told apart by an occurrence index,
 * as an uncached run computes each of them anew. The demo's order of cells is
 * fixed by its build, so the index maps to the same cell after a restart.
 *
 * @param key     Cell label without tabs or newlines, ideally with the cell's
 *                distinguishing parameters (solver, seed, grid value).
 * @param compute Computes the cell when it is not cached.
 */
inline Cell cached(const std::string& key, const std::function<Cell()>& compute,
                   const char* output_dir, const char* build_time) {
    detail::State& s = detail::state();
    detail::open(output_dir, build_time);
    const std::string full_key = key + "#" + std::to_string(s.used[key]++);
    if (!s.enabled) return compute();
    const auto hit = s.cells.find(full_key);
    if (hit != s.cells.end()) {
        std::cout << "  [cache] " << full_key << '\n';
        return hit->second;
    }
    Cell cell = compute();
    std::ofstream out(s.path, std::ios::app);
    out << full_key;
    char buf[64];
    for (const auto& [name, value] : cell) {
        std::snprintf(buf, sizeof(buf), "%.17g", value);
        out << '\t' << name << '=' << buf;
    }
    out << "\t.\n";
    out.flush();
    s.cells[full_key] = cell;
    return cell;
}

/** @brief Remove the cache once the demo has written all its results. */
inline void finish() {
    detail::State& s = detail::state();
    if (s.enabled && !s.path.empty()) {
        std::error_code ec;
        std::filesystem::remove(s.path, ec);
    }
}

/** @brief Value of `name` in a cell; throws when absent. */
inline double get(const Cell& cell, const std::string& name) {
    for (const auto& [n, v] : cell) {
        if (n == name) return v;
    }
    throw std::out_of_range("cell_cache: no value \"" + name + "\" in cell");
}

}  // namespace trex_cell_cache

// ==============================================================================
// Demo macro
// ==============================================================================

/**
 * @brief Cached cell in a demo translation unit: binds the demo's output folder
 *        (DEMO_OUTPUT_DIR, set by CMake) and its build time to the cache.
 */
#define TREX_CELL_CACHED(key, compute) \
    ::trex_cell_cache::cached((key), (compute), DEMO_OUTPUT_DIR, __DATE__ " " __TIME__)

#endif /* End of DEMOS_TREX_SELECTOR_METHODS_CELL_CACHE_HPP */
// ==============================================================================
