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

"$rscript" "$plotter" "$data/gvs_Hastie_2d.csv" \
  --title 'T-Rex+GVS on Hastie equicorrelated blocks: SNR × ρ sweep' \
  --tfdr "$tfdr" "$@"
"$rscript" "$plotter" "$data/gvs_Hastie_lambda2_method.csv" \
  --title 'T-Rex+GVS on Hastie: λ₂ selection method (SNR=2, ρ=0.7)' \
  --param-label 'λ₂ selection method' --tfdr "$tfdr" "$@"
"$rscript" "$plotter" "$data/gvs_Hastie_hc_linkage.csv" \
  --title 'T-Rex+GVS on Hastie: HAC linkage (SNR=2, ρ=0.7)' \
  --param-label 'HAC linkage' --tfdr "$tfdr" "$@"
"$rscript" "$plotter" "$data/gvs_Hastie_cut.csv" \
  --title 'T-Rex+GVS on Hastie: clustering cut (SNR=2, ρ=0.7)' \
  --param-label 'Clustering cut' --tfdr "$tfdr" "$@"
