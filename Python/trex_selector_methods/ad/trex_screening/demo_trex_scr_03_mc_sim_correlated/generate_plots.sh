#!/usr/bin/env bash
#
# Regenerate this demo's Screen-TRex figures via the suite plotting module
#   ../trex_scr_plt_utils.py
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

plotter="$here/../trex_scr_plt_utils.py"
data="$here/simulation_results/data"

shopt -s nullglob
csvs=("$data"/*.csv)
if [[ ${#csvs[@]} -eq 0 ]]; then
  echo "error: no CSVs in $data — run the demo first." >&2
  exit 1
fi

for csv in "${csvs[@]}"; do
  stem="$(basename "$csv" .csv)"
  case "$stem" in
    scr_ar1_snr_*)        title='Screen-TRex on an AR(1) design: TPR/FDR vs SNR' ;;
    scr_equi_snr_*)       title='Screen-TRex on an equicorrelated design: TPR/FDR vs SNR' ;;
    scr_ar1_rho_*)        title='Screen-TRex on an AR(1) design: TPR/FDR vs rho' ;;
    scr_equi_rho_*)       title='Screen-TRex on an equicorrelated design: TPR/FDR vs rho' ;;
    scr_block_equi_rho_*) title='Screen-TRex on a block-equicorrelated design: TPR/FDR vs rho' ;;
    *) title="$stem" ;;
  esac
  "$venv_python" "$plotter" "$csv" --title "$title" "$@"
done
