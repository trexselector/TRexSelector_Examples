#!/usr/bin/env bash
#
# Regenerate this demo's T-Rex SPCA figures via the suite plotting module
#   ../trex_spca_plt_utils.R  (ggplot2)
# Figures are written to simulation_results/plots/.
#
# Usage:
#   ./generate_plots.sh                     # this demo's full figure set
#   ./generate_plots.sh --formats png       # extra args pass through
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

rscript="${RSCRIPT:-$(command -v Rscript || true)}"
if [[ -z "$rscript" ]]; then
  echo "error: no Rscript found on PATH (set RSCRIPT=/path/to/Rscript)" >&2
  exit 1
fi

plotter="$here/../trex_spca_plt_utils.R"
data="$here/simulation_results/data"

shopt -s nullglob
csvs=("$data"/*.csv)
if [[ ${#csvs[@]} -eq 0 ]]; then
  echo "error: no CSVs in $data — run the demo first." >&2
  exit 1
fi

for csv in "${csvs[@]}"; do
  stem="$(basename "$csv" .csv)"
  if [[ "$stem" == *union_fdr* ]]; then
    title='T-Rex SPCA: union-support FDR per PC (ActiveSet/TENET)'
  else
    title='T-Rex SPCA: TPR/FDR vs SNR (dB)'
  fi
  "$rscript" "$plotter" "$csv" \
    --title "$title" \
    --tfdr 0.1 "$@"
done
