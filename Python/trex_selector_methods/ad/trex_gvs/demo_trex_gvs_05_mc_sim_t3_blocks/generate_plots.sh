#!/usr/bin/env bash
#
# Regenerate this demo's T-Rex+GVS figures via the suite plotting module
#   ../trex_gvs_plt_utils.py
# using the repo's local .venv. Figures are written to simulation_results/plots/.
#
# Usage:
#   ./generate_plots.sh                 # this demo's full figure set
#   ./generate_plots.sh --formats png   # extra args pass through to the plotter
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Find the repo-root .venv by walking up the directory tree.
dir="$here"; venv_python=""
while [[ "$dir" != "/" ]]; do
  if [[ -x "$dir/.venv/bin/python" ]]; then venv_python="$dir/.venv/bin/python"; break; fi
  dir="$(dirname "$dir")"
done
if [[ -z "$venv_python" ]]; then
  echo "error: no .venv/bin/python found above $here" >&2
  echo "       create it and 'pip install matplotlib pandas numpy', then re-run." >&2
  exit 1
fi

plotter="$here/../trex_gvs_plt_utils.py"
data="$here/simulation_results/data"
plots="$here/simulation_results/plots"
tfdr=0.1

"$venv_python" "$plotter" "$data/gvs_t3_blocks_2d.csv" \
  --title 'T-Rex+GVS on heavy-tailed t(3) blocks: SNR $\times$ $\rho$ sweep' \
  --tfdr "$tfdr" "$@"
