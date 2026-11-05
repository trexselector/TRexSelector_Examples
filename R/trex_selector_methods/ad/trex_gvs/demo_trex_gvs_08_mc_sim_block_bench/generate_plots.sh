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

# Headline: value of prior group knowledge as a function of rho (WHOLE truth).
# Panels at weak / moderate / strong SNR; x-axis = rho crossing the
# corr_max = 0.5 discoverability boundary.
"$rscript" "$plotter" "$data/gvs_block_bench_whole.csv" \
  --vs rho --panels 0.5 2 5 \
  --stem gvs_block_bench_whole_vs_rho \
  --title 'T-Rex+GVS block benchmark (WHOLE truth): prior groups vs HAC discoverability' \
  --tfdr "$tfdr" "$@"

# Supplementary: TPR/FDR vs SNR at three discoverability regimes
# (rho = 0.2 undiscoverable, 0.5 partial, 0.9 fully discovered).
"$rscript" "$plotter" "$data/gvs_block_bench_whole.csv" \
  --vs snr --panels 0.2 0.5 0.9 \
  --stem gvs_block_bench_whole \
  --title 'T-Rex+GVS block benchmark (WHOLE truth): TPR/FDR vs SNR' \
  --tfdr "$tfdr" "$@"

# Cautionary appendix: INDIVIDUAL truth (exchangeable null siblings) —
# coordinate-level FDR is not controllable here; see the README.
"$rscript" "$plotter" "$data/gvs_block_bench_individual.csv" \
  --title 'T-Rex+GVS block benchmark (INDIVIDUAL truth): TPR/FDR vs SNR' \
  --tfdr "$tfdr" "$@"
