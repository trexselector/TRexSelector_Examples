#!/usr/bin/env bash
#
# Regenerate this demo's FDR/TPR figures via the plotting module:
#   ../trex_plt_utils.py
# using the repo's local .venv. Figures are written to
# simulation_results/plots/.
#
# Usage:
#   ./generate_plots.sh                          # this demo's default CSV
#   ./generate_plots.sh <other.csv>              # explicit CSV
#   ./generate_plots.sh --formats png            # extra args pass through to
#   ./generate_plots.sh --tfdr 0.05              # trex_plt_utils.py
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
  echo "       create it and 'pip install matplotlib pandas plotly', then re-run." >&2
  exit 1
fi

# First non-flag argument overrides this demo's default CSV.
csv="$here/simulation_results/data/demo_trex_02_mc_sim_fixed_support_trex_results_n300_p1000_stagnation_window_7.csv"
if [[ $# -gt 0 && "$1" != -* ]]; then
  csv="$1"
  shift
fi

exec "$venv_python" "$here/../trex_plt_utils.py" "$csv" \
  --title 'T-Rex Monte Carlo simulation ($n=300$, $p=1000$, fixed support)' \
  --legend-title 'Method' \
  "$@"
