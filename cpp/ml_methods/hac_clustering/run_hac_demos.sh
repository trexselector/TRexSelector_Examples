#!/bin/bash
# ==============================================================================
# run_hac_demos.sh
# ==============================================================================
# Run the hac_clustering demos sequentially and keep their console output.
#
# Default order: 01 02  (pass demo numbers to override, e.g.
#   ./run_hac_demos.sh 01
# runs only the in-memory suite).
#
# Threads: THREADS=8 ./run_hac_demos.sh   (default 6; passed to each demo as
# its first argument, the demos call omp_set_num_threads with it).
#
# Behaviour:
#   - rebuilds the selected demo targets first (cmake build/release)
#   - runs each demo binary in order; a failure does not stop the chain
#   - per-demo console logs in logs/hac_<timestamp>/demo_XX.log
#     (the demos print to the console only, no result files)
#   - keeps the Mac awake while running (caffeinate -i, tied to this script)
#   - prints a summary table with per-demo exit codes and wall-clock times
#
# Launch in the foreground and follow a log in a second terminal:
#   cd cpp/ml_methods/hac_clustering
#   THREADS=8 ./run_hac_demos.sh
#   tail -f logs/hac_*/demo_01.log
# or unattended:  nohup ./run_hac_demos.sh > /dev/null 2>&1 &
# ==============================================================================

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CPP_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BIN_DIR="${CPP_ROOT}/build/release/bin/ml_methods/hac_clustering"
THREADS="${THREADS:-6}"

# Demo number -> full name
demo_name() {
    case "$1" in
        01) echo "demo_mlm_hac_01" ;;
        02) echo "demo_mlm_hac_02_mmap" ;;
        *)  echo "" ;;
    esac
}

DEMOS=("$@")
if [ ${#DEMOS[@]} -eq 0 ]; then
    DEMOS=(01 02)
fi

TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${SCRIPT_DIR}/logs/hac_${TS}"
mkdir -p "${LOG_DIR}"

echo "hac_clustering run — $(date)"
echo "Demos:   ${DEMOS[*]}"
echo "Threads: ${THREADS}"
echo "Logs:    ${LOG_DIR}"
echo

# Keep the machine awake for the lifetime of this script (macOS).
if command -v caffeinate >/dev/null 2>&1; then
    caffeinate -i -w $$ &
fi

# ------------------------------------------------------------------------------
# Rebuild the selected targets so the binaries match the current sources.
# ------------------------------------------------------------------------------
TARGETS=()
for d in "${DEMOS[@]}"; do
    name="$(demo_name "$d")"
    if [ -z "$name" ]; then
        echo "Unknown demo number: $d — skipping."
        continue
    fi
    TARGETS+=("$name")
done

echo "Building: ${TARGETS[*]}"
if ! cmake --build "${CPP_ROOT}/build/release" --target "${TARGETS[@]}" -j 8 \
        > "${LOG_DIR}/build.log" 2>&1; then
    echo "BUILD FAILED — see ${LOG_DIR}/build.log. Aborting."
    exit 1
fi
echo "Build OK."
echo

# ------------------------------------------------------------------------------
# Run demos sequentially.
# ------------------------------------------------------------------------------
declare -a SUMMARY

for d in "${DEMOS[@]}"; do
    name="$(demo_name "$d")"
    [ -z "$name" ] && continue
    bin="${BIN_DIR}/${name}/${name}"
    log="${LOG_DIR}/demo_${d}.log"

    if [ ! -x "$bin" ]; then
        echo "[$(date +%H:%M:%S)] demo ${d}: binary not found (${bin}) — SKIPPED"
        SUMMARY+=("demo ${d}  MISSING")
        continue
    fi

    echo "[$(date +%H:%M:%S)] demo ${d} (${name}) starting with ${THREADS} threads ..."
    t0=$(date +%s)
    "$bin" "${THREADS}" > "$log" 2>&1
    rc=$?
    t1=$(date +%s)
    dt=$((t1 - t0))
    hms=$(printf '%dh %02dm %02ds' $((dt/3600)) $(((dt%3600)/60)) $((dt%60)))

    if [ $rc -eq 0 ]; then
        echo "[$(date +%H:%M:%S)] demo ${d} finished OK in ${hms}"
        SUMMARY+=("demo ${d}  OK      ${hms}")
    else
        echo "[$(date +%H:%M:%S)] demo ${d} FAILED (exit ${rc}) after ${hms} — see ${log}"
        SUMMARY+=("demo ${d}  FAILED(${rc})  ${hms}")
    fi
done

# ------------------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------------------
echo
echo "=============================================================="
echo "  hac_clustering run summary — $(date)"
echo "=============================================================="
for line in "${SUMMARY[@]}"; do
    echo "  $line"
done
echo "  Logs: ${LOG_DIR}"
echo "=============================================================="
