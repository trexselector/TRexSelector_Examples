#!/usr/bin/env bash
#
# Regenerate this demo's T-Rex+GVS figures via the suite plotting module
#   ../trex_gvs_plt_utils.R  (ggplot2)
# Figures are written to simulation_results/plots/.
#
# Usage:
#   ./generate_plots.sh                 # this demo's full figure set
#   ./generate_plots.sh --formats png   # extra args pass through to the plotter
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

rscript="${RSCRIPT:-$(command -v Rscript || true)}"
if [[ -z "$rscript" ]]; then
  echo "error: no Rscript found on PATH (set RSCRIPT=/path/to/Rscript)" >&2
  exit 1
fi

plotter="$here/../trex_gvs_plt_utils.R"
data="$here/simulation_results/data"
tfdr=0.1

"$rscript" "$plotter" "$data/gvs_ar1_blocks_2d.csv" \
  --title 'T-Rex+GVS on AR(1) blocks: SNR × ρ sweep' \
  --tfdr "$tfdr" "$@"
