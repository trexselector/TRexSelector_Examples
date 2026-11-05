#!/usr/bin/env bash
#
# Regenerate this demo's figures via the plotting module:
#   ../ts_plt_utils.R nhg  (ggplot2)
# the histogram and the Q-Q plot of the included null variables against the
# negative hypergeometric distribution, and the ECDFs of the selected null
# variables. Figures are written to simulation_results/plots/.
#
# Usage:
#   ./generate_plots.sh                          # the 500-run result file
#   ./generate_plots.sh 4                        # the file of a 4-run test
#   ./generate_plots.sh --formats png            # extra args pass through to
#                                                # ts_plt_utils.R
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

rscript="${RSCRIPT:-$(command -v Rscript || true)}"
if [[ -z "$rscript" ]]; then
  echo "error: no Rscript found on PATH (set RSCRIPT=/path/to/Rscript)" >&2
  exit 1
fi

# First non-flag argument selects the number of runs in the file name.
mc=500
if [[ $# -gt 0 && "$1" != -* ]]; then
  mc="$1"
  shift
fi

csv="$here/simulation_results/data/null_counts_vs_nhg_mc$mc.csv"
if [[ ! -f "$csv" ]]; then
  echo "error: no result file for $mc runs ($csv)" >&2
  exit 1
fi

exec "$rscript" "$here/../../ts_plt_utils.R" nhg "$csv" \
  --title "T-solvers (\$L = p\$, $mc runs)" \
  "$@"
