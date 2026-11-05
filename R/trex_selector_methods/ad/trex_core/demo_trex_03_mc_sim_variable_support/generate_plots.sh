#!/usr/bin/env bash
#
# Regenerate this demo's FDR/TPR figures via the plotting module:
#   ../trex_plt_utils.R  (ggplot2)
# Figures are written to simulation_results/plots/.
#
# Usage:
#   ./generate_plots.sh                          # this demo's default CSV
#   ./generate_plots.sh <other.csv>              # explicit CSV
#   ./generate_plots.sh --formats png            # extra args pass through to
#   ./generate_plots.sh --tfdr 0.05              # trex_plt_utils.R
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

rscript="${RSCRIPT:-$(command -v Rscript || true)}"
if [[ -z "$rscript" ]]; then
  echo "error: no Rscript found on PATH (set RSCRIPT=/path/to/Rscript)" >&2
  exit 1
fi

# First non-flag argument overrides this demo's default CSV.
csv="$here/simulation_results/data/demo_trex_03_mc_sim_variable_support_trex_results_n300_p1000_stagnation_window_5.csv"
if [[ $# -gt 0 && "$1" != -* ]]; then
  csv="$1"
  shift
fi

exec "$rscript" "$here/../trex_plt_utils.R" "$csv" \
  --title 'T-Rex Monte Carlo simulation (n = 300, p = 1000, variable support)' \
  --legend-title 'Method' \
  "$@"
