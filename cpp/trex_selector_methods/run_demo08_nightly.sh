#!/usr/bin/env bash
# ==============================================================================
# run_demo08_nightly.sh  —  trex_core demo 08 (scalability) as a resumable job
# ==============================================================================
#
# The scalability benchmark runs for days in total, so it runs in nightly
# portions: start it in the evening, stop it in the morning, start it again the
# next evening. Steps, one after another:
#   C++ base, C++ large, Python base, Python large, R base, R large.
# Every demo keeps a per-trial cache, so a finished step costs seconds and a
# stopped step resumes at its first missing trial.
#
# Usage (from anywhere):
#   cpp/trex_selector_methods/run_demo08_nightly.sh start [cpp|py|r]   # run or continue
#   cpp/trex_selector_methods/run_demo08_nightly.sh stop               # graceful stop
#   cpp/trex_selector_methods/run_demo08_nightly.sh steps [cpp|py|r]   # list, runs nothing
#
#   start  launches the steps in the background (one instance only) and keeps
#          the Mac awake while they run. A language restricts the run to its
#          two steps, e.g. "start cpp" runs C++ base and C++ large only.
#   stop   lets the current trial finish and ends the run there; no finished
#          work is lost. The demos see the stop file through TREX_STOP_FILE.
#
# Logs (gitignored): cpp/trex_selector_methods/logs/demo08_nightly/
#   runner.log  one line per step start and end
#   <step>.log  console output of each step, appended night after night
# ==============================================================================
set -uo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "$here/../.." && pwd)"
logdir="$here/logs/demo08_nightly"
stop_file="$logdir/STOP"
pid_file="$logdir/runner.pid"
lock_dir="$logdir/runner.lock"   # mkdir is atomic: one instance only

cpp_bin="$repo/cpp/build/release/bin/trex_selector_methods/ad/trex_core/demo_trex_08_mc_sim_scalability"
py_demo="$repo/Python/trex_selector_methods/ad/trex_core/demo_trex_08_mc_sim_scalability/demo_trex_08_mc_sim_scalability.py"
r_demo="$repo/R/trex_selector_methods/ad/trex_core/demo_trex_08_mc_sim_scalability/demo_trex_08_mc_sim_scalability.R"

# Exit code of a demo that stopped on request (sysexits EX_TEMPFAIL)
exit_stopped=75

log() { echo "$(date '+%Y-%m-%d %H:%M:%S')  $*" >> "$logdir/runner.log"; }

running() { [[ -f "$pid_file" ]] && kill -0 "$(cat "$pid_file")" 2>/dev/null; }

# Run one step; returns the demo's exit code
run_step() {
    local name="$1"; shift
    echo "===== $(date '+%Y-%m-%d %H:%M:%S')  $name =====" >> "$logdir/$name.log"
    log "start $name"
    TREX_STOP_FILE="$stop_file" "$@" >> "$logdir/$name.log" 2>&1
    local rc=$?
    log "end   $name (exit $rc)"
    return $rc
}

all_steps=(
    "cpp_base|cd '$cpp_bin' && ./demo_trex_08_mc_sim_scalability base"
    "cpp_large|cd '$cpp_bin' && ./demo_trex_08_mc_sim_scalability large"
    "py_base|'$repo/.venv/bin/python' '$py_demo' base"
    "py_large|'$repo/.venv/bin/python' '$py_demo' large"
    "r_base|Rscript '$r_demo' base"
    "r_large|Rscript '$r_demo' large"
)

# Steps of one language ("cpp", "py", "r") or all of them (empty filter)
selected_steps() {
    local filter="$1" s
    for s in "${all_steps[@]}"; do
        [[ -z "$filter" || "${s%%|*}" == "${filter}_"* ]] && echo "$s"
    done
}

valid_filter() { [[ -z "$1" || "$1" == cpp || "$1" == py || "$1" == r ]]; }

chain() {
    trap 'rm -f "$pid_file"; rmdir "$lock_dir" 2>/dev/null' EXIT
    local filter="${1:-}" s
    local steps=()
    while IFS= read -r s; do steps+=("$s"); done < <(selected_steps "$filter")
    log "run: ${filter:-all languages}"
    for s in "${steps[@]}"; do
        local name="${s%%|*}" cmd="${s#*|}"
        run_step "$name" bash -c "$cmd"
        local rc=$?
        if [[ $rc -eq $exit_stopped ]]; then
            log "stopped on request; the next start resumes at $name"
            return 0
        elif [[ $rc -ne 0 ]]; then
            log "ERROR in $name (exit $rc), run ended; see $name.log"
            return "$rc"
        fi
    done
    log "all selected steps complete (${filter:-all languages})"
}

mkdir -p "$logdir"
case "${1:-}" in
    start)
        if ! valid_filter "${2:-}"; then
            echo "unknown language '${2}', use cpp, py or r" >&2
            exit 2
        fi
        if running; then
            echo "demo 08 runner already running (pid $(cat "$pid_file")), nothing started"
            exit 0
        fi
        # A lock left by a crashed runner (no live pid) is stale
        [[ -d "$lock_dir" ]] && ! running && rm -f "$pid_file" && rmdir "$lock_dir" 2>/dev/null
        if ! mkdir "$lock_dir" 2>/dev/null; then
            echo "demo 08 runner is starting in another shell, nothing started"
            exit 0
        fi
        echo $$ > "$pid_file"   # live pid until the runner's own pid replaces it
        rm -f "$stop_file"
        nohup caffeinate -i "${BASH_SOURCE[0]}" _chain "${2:-}" > /dev/null 2>&1 &
        echo $! > "$pid_file"
        echo "demo 08 runner started (${2:-all languages}); log: $logdir/runner.log"
        ;;
    stop)
        if ! running; then
            echo "demo 08 runner is not running"
            exit 0
        fi
        touch "$stop_file"
        echo "stop requested: the current trial finishes, then the run ends (see $logdir/runner.log)"
        ;;
    steps)
        if ! valid_filter "${2:-}"; then
            echo "unknown language '${2}', use cpp, py or r" >&2
            exit 2
        fi
        selected_steps "${2:-}" | sed 's/|.*//'
        ;;
    _chain)
        chain "${2:-}"
        ;;
    *)
        echo "usage: $0 start [cpp|py|r] | stop | steps [cpp|py|r]" >&2
        exit 2
        ;;
esac
