#!/usr/bin/env bash
#
# Regenerate this demo's scalability dashboard via the suite plotter's
# 'scalability' mode:
#   ../trex_plt_utils.py scalability <csv> [<csv> ...] ...
# using the repo's local .venv. Figures are written to
# simulation_results/plots/.
#
# Demo 08's CSVs use a wider schema (n,p,is_barrier,solver,num_mc,snr,metric,
# value) than the rest of the suite; the shared module reads them via
# read_scalability() and renders them with plot_scalability_dashboard(). The
# base block and the large-p block are concatenated into one dashboard, so the
# n = 300 line runs from p = 1,000 to 1,000,000. Large-p points are drawn
# white-filled, and its cap-5 point at p = 100,000 sits next to the base one.
#
# Usage:
#   ./generate_plots.sh                          # this demo's default CSVs
#   ./generate_plots.sh <a.csv> [<b.csv> ...]    # explicit CSVs
#   ./generate_plots.sh --formats png            # extra args pass through to
#   ./generate_plots.sh --tfdr 0.05              # the scalability plotter
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Find the repo-root .venv by walking up the directory tree.
dir="$here"
venv_python=""
while [[ "$dir" != "/" ]]; do
  if [[ -x "$dir/.venv/bin/python" ]]; then
    venv_python="$dir/.venv/bin/python"
    break
  fi
  dir="$(dirname "$dir")"
done

if [[ -z "$venv_python" ]]; then
  echo "error: no .venv/bin/python found above $here" >&2
  echo "       create it and 'pip install matplotlib pandas', then re-run." >&2
  exit 1
fi

# Leading non-flag arguments override this demo's default (wide-schema) CSVs.
csvs=()
while [[ $# -gt 0 && "$1" != -* ]]; do
  csvs+=("$1")
  shift
done
if [[ ${#csvs[@]} -eq 0 ]]; then
  data="$here/simulation_results/data"
  for stem in demo_trex_08_scalability_results demo_trex_08_scalability_large_p; do
    [[ -f "$data/$stem.csv" ]] && csvs+=("$data/$stem.csv")
  done
fi
if [[ ${#csvs[@]} -eq 0 ]]; then
  echo "error: no result CSV found in $here/simulation_results/data" >&2
  exit 1
fi

# The scalability dashboard lives in the suite plotter's 'scalability' mode.
exec "$venv_python" "$here/../trex_plt_utils.py" scalability "${csvs[@]}" \
  --title 'T-Rex scalability on the memory-mapped pipeline ($n \times p$ grid, SNR = 1)' \
  --stem demo_trex_08_scalability \
  --outdir "$here/simulation_results/plots" \
  "$@"
