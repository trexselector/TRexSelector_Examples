#!/usr/bin/env bash
# ==============================================================================
# run_all_overnight.sh  —  C++ T-Rex Monte Carlo demos, one after another.
# ==============================================================================
#
# C++ counterpart of Python/ and R/trex_selector_methods/run_all_overnight.sh.
# Runs the Monte Carlo demos of the given suites (default: all, in the order
# below) on the release build. Single runs (trex_core 01, 06) and the
# scalability benchmark (trex_core 08) are not part of the batch.
#
# Usage:
#   ./run_all_overnight.sh [--list] [--build] [--from DEMO] [suite ...]
#     --list       print the demos that would run, then exit
#     --build      rebuild the queued demo targets first (build/release)
#     --from DEMO  skip the demos before the first one whose name starts with
#                  DEMO, e.g. --from demo_trex_04
#   FRESH=1 ./run_all_overnight.sh      rerun everything (see below)
#
# Launch for an overnight run:
#   nohup cpp/trex_selector_methods/run_all_overnight.sh > /dev/null 2>&1 &
#   tail -f cpp/trex_selector_methods/logs/overnight_*/summary.txt
#
# Behaviour:
#   - continue-on-error: a failing demo does not stop the batch
#   - per-demo console logs and a summary in logs/overnight_<timestamp>/
#     (gitignored); the results overwrite the demos' simulation_results/
#   - core budget TREX_NUM_WORKERS (default 8, 0 = every processor), passed
#     on to the demos and to OpenMP; keeps the Mac awake (caffeinate)
#   - stopping the runner (Ctrl-C, kill) also stops the demo in flight
#
# Run tag and resume:
#   TREX_RUN_TAG names the library state the results come from: the pinned
#   commit of LIBRARY_TARGET.txt unless the caller sets it. A demo that exits 0
#   is recorded in logs/done_<tag>.txt, and later runs with the same tag skip
#   it, so a stopped batch restarts at the demo it was in. The demos cache
#   their finished grid cells under the same tag, so the demo in flight
#   continues where it stopped too. FRESH=1 ignores the record and makes the
#   demos set their caches aside, which reruns everything.
#
# Kept 3.2-compatible so it runs under the stock /bin/bash on macOS.
# ==============================================================================
set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CPP_ROOT="$(cd "$HERE/.." && pwd)"
REPO_ROOT="$(cd "$CPP_ROOT/.." && pwd)"
BIN="$CPP_ROOT/build/release/bin/trex_selector_methods/ad"
LOG_ROOT="$HERE/logs"
export TREX_NUM_WORKERS="${TREX_NUM_WORKERS:-8}"
[ "$TREX_NUM_WORKERS" -gt 0 ] 2>/dev/null && export OMP_NUM_THREADS="$TREX_NUM_WORKERS"

declare -a ORDER=(trex_screening trex_spca trex_core trex_da trex_gvs)
suite_demos() {
    case "$1" in
        trex_screening) ls -d "$BIN"/trex_screening/demo_trex_scr_0[1-6]_* ;;
        trex_spca)      ls -d "$BIN"/trex_spca/demo_trex_spca_0[12]_* ;;
        trex_core)      ls -d "$BIN"/trex_core/demo_trex_0[2-579]_* "$BIN"/trex_core/demo_trex_10_* ;;
        trex_da)        ls -d "$BIN"/trex_da/demo_trex_da_0[1-8]_* ;;
        trex_gvs)       ls -d "$BIN"/trex_gvs/demo_trex_gvs_0[1-8]_* ;;
        *)              echo "unknown suite: $1" >&2 ;;
    esac
}

FROM=""
LIST=0
BUILD=0
while [ $# -gt 0 ]; do
    case "$1" in
        --from)  FROM="${2:?--from needs a demo name}"; shift 2 ;;
        --list)  LIST=1; shift ;;
        --build) BUILD=1; shift ;;
        *) break ;;
    esac
done
[ $# -gt 0 ] && ORDER=("$@")

# Run tag: the pinned commit unless set by the caller.
if [ -z "${TREX_RUN_TAG:-}" ]; then
    TREX_RUN_TAG="$(awk '/^Pinned commit:/ { print $3; exit }' "$REPO_ROOT/LIBRARY_TARGET.txt" 2>/dev/null)"
    [ -z "$TREX_RUN_TAG" ] && TREX_RUN_TAG="untagged"
fi
export TREX_RUN_TAG
export TREX_FRESH="${FRESH:-0}"
mkdir -p "$LOG_ROOT"
DONE_FILE="$LOG_ROOT/done_${TREX_RUN_TAG}.txt"

# Run queue as suite|demo-dir pairs, cut at --from
declare -a QUEUE=()
started=1; [ -n "$FROM" ] && started=0
for s in "${ORDER[@]}"; do
    for d in $(suite_demos "$s"); do
        if [ $started -eq 0 ]; then
            case "$(basename "$d")" in "$FROM"*) started=1 ;; *) continue ;; esac
        fi
        QUEUE+=("$s|$d")
    done
done
if [ ${#QUEUE[@]} -eq 0 ]; then
    echo "nothing to run (suites: ${ORDER[*]}${FROM:+, from: $FROM}; build the release targets first?)" >&2
    exit 1
fi
if [ $LIST -eq 1 ]; then
    for q in "${QUEUE[@]}"; do
        n="$(basename "${q#*|}")"
        mark=""
        [ "$TREX_FRESH" != "1" ] && grep -qx "$n" "$DONE_FILE" 2>/dev/null && mark="  (done on $TREX_RUN_TAG)"
        echo "${q%%|*}/$n$mark"
    done
    exit 0
fi

TS="$(date +%Y%m%d_%H%M%S)"
LOG="$LOG_ROOT/overnight_$TS"
mkdir -p "$LOG"
SUMMARY="$LOG/summary.txt"
command -v caffeinate >/dev/null 2>&1 && caffeinate -i -w $$ &

# Stopping the runner stops the demo in flight as well.
DEMO_PID=""
on_signal() {
    [ -n "$DEMO_PID" ] && kill -TERM "$DEMO_PID" 2>/dev/null
    echo "[$(date '+%F %T')] runner stopped ($1)" | tee -a "$SUMMARY"
    exit 130
}
trap 'on_signal INT' INT
trap 'on_signal TERM' TERM
trap 'on_signal HUP' HUP

{
    echo "C++ T-Rex Monte Carlo demos $(date '+%F %T')"
    echo "  suites : ${ORDER[*]}${FROM:+  (from $FROM)}"
    echo "  run tag: $TREX_RUN_TAG   FRESH=$TREX_FRESH   (record: $(basename "$DONE_FILE"))"
    echo "  logs   : $LOG"
} | tee "$SUMMARY"

if [ $BUILD -eq 1 ]; then
    declare -a TARGETS=()
    for q in "${QUEUE[@]}"; do TARGETS+=("$(basename "${q#*|}")"); done
    echo "[$(date '+%F %T')] building ${#TARGETS[@]} target(s)" | tee -a "$SUMMARY"
    if ! cmake --build "$CPP_ROOT/build/release" --target "${TARGETS[@]}" -j 8 > "$LOG/build.log" 2>&1; then
        echo "BUILD FAILED, see $LOG/build.log" | tee -a "$SUMMARY"
        exit 1
    fi
fi

ok=0; fail=0; done_n=0
for q in "${QUEUE[@]}"; do
    s="${q%%|*}"
    d="${q#*|}"
    n="$(basename "$d")"
    if [ "$TREX_FRESH" != "1" ] && grep -qx "$n" "$DONE_FILE" 2>/dev/null; then
        echo "[$(date '+%F %T')] done  $s/$n  (finished earlier on $TREX_RUN_TAG, FRESH=1 reruns it)" \
            | tee -a "$SUMMARY"
        done_n=$((done_n + 1))
        continue
    fi
    t0=$(date +%s)
    echo "[$(date '+%F %T')] start $s/$n" | tee -a "$SUMMARY"
    (cd "$d" && exec "./$n") > "$LOG/$n.log" 2>&1 &
    DEMO_PID=$!
    wait "$DEMO_PID"
    rc=$?
    DEMO_PID=""
    t1=$(date +%s)
    if [ $rc -eq 0 ]; then
        grep -qx "$n" "$DONE_FILE" 2>/dev/null || echo "$n" >> "$DONE_FILE"
        ok=$((ok + 1))
    else
        fail=$((fail + 1))
    fi
    printf '[%s] %s %s/%s  exit=%d  %dm%02ds\n' "$(date '+%F %T')" \
        "$([ $rc -eq 0 ] && echo 'ok   ' || echo 'FAIL ')" "$s" "$n" "$rc" \
        $(((t1 - t0) / 60)) $(((t1 - t0) % 60)) | tee -a "$SUMMARY"
done
echo "[$(date '+%F %T')] all done: $ok ok, $fail failed, $done_n done earlier" | tee -a "$SUMMARY"
