#!/usr/bin/env bash
#
# Regenerate this demo's DA-TRex figures via the suite plotting module
#   ../trex_da_plt_utils.py
# using the repo's local .venv. Figures are written to
# simulation_results/plots/.
#
# Demo 06 (multi-level prior-groups DGP) currently exercises a single SNR sweep,
# comparing the three DA-TRex solvers (TLARS, TAFS, TOMP) on a three-level nested
# dependency structure. This script produces its FDR/TPR overview (+ interactive
# html).
#
# Usage:
#   ./generate_plots.sh                 # this demo's default CSV
#   ./generate_plots.sh <other.csv>     # explicit CSV
#   ./generate_plots.sh --formats png   # extra args pass through to the plotter
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

plotter="$here/../trex_da_plt_utils.py"

# First non-flag argument overrides this demo's default CSV.
csv="$here/simulation_results/data/da_trex_mc_da_groups_toeplitz_leaf.csv"
if [[ $# -gt 0 && "$1" != -* ]]; then
  csv="$1"
  shift
fi

exec "$venv_python" "$plotter" "$csv" \
  --title 'DA-TRex+AKG (a priori known groups) on 3-level nested data: SNR sweep' \
  --legend-title 'Method' --relabel 'TREX-DA=TREX-DA+AKG' --tfdr 0.2 "$@"
