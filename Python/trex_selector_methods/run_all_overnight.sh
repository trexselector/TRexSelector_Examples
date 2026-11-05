#!/usr/bin/env bash
# Overnight master: run every suite's run_overnight.sh sequentially (suites run
# one after another so they don't contend for CPU/RAM). Each suite writes its
# own logs + summary under <suite>/_overnight_logs/.
#
#   ./run_all_overnight.sh                 # all suites, everything
#   SKIP_HEAVY=1 ./run_all_overnight.sh    # all suites, skip heavy demos
#
# To run a subset, pass suite names:
#   ./run_all_overnight.sh trex_da trex_gvs
set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

suites=("$@")
if [[ ${#suites[@]} -eq 0 ]]; then
  suites=(trex_core trex_da trex_gvs trex_screening trex_spca)
fi

echo "### master overnight run: ${suites[*]}  ($(date)) ###"
for s in "${suites[@]}"; do
  script="$here/ad/$s/run_overnight.sh"
  if [[ ! -x "$script" && ! -f "$script" ]]; then
    echo "!!! missing $script — skipping"; continue
  fi
  echo
  echo "######################## $s ########################"
  bash "$script" || echo "!!! $s run_overnight.sh returned non-zero"
done
echo
echo "### master overnight run complete ($(date)) ###"
