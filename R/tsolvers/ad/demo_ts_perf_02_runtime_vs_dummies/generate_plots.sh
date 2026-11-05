#!/usr/bin/env bash
#
# Regenerate this demo's runtime figures via the plotting module:
#   ../ts_plt_utils.R  (ggplot2)
# Figures are written to simulation_results/plots/.
#
# Usage:
#   ./generate_plots.sh                          # this demo's default CSV
#   ./generate_plots.sh <other.csv>              # explicit CSV, e.g. a smoke run
#   ./generate_plots.sh --formats png            # extra args pass through to
#                                                # ts_plt_utils.R
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

rscript="${RSCRIPT:-$(command -v Rscript || true)}"
if [[ -z "$rscript" ]]; then
  echo "error: no Rscript found on PATH (set RSCRIPT=/path/to/Rscript)" >&2
  exit 1
fi

# First non-flag argument overrides this demo's default CSV.
csv="$here/simulation_results/data/runtime_vs_dummies_mc500.csv"
if [[ $# -gt 0 && "$1" != -* ]]; then
  csv="$1"
  shift
fi

exec "$rscript" "$here/../../ts_plt_utils.R" runtime "$csv" \
  --title 'T-solvers ($n = 500$, $p = 1{,}000$, $T = 10$, 500 runs)' \
  --p 1000 \
  "$@"
