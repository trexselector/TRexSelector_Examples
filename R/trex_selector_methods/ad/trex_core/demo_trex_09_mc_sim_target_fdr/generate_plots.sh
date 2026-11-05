#!/usr/bin/env bash
#
# Regenerate this demo's figure via the suite plotter's 'sweep' mode
#   ../trex_plt_utils.R sweep <csv> ...  (ggplot2)
# achieved FDR (top, with the diagonal FDR = tFDR) and TPR (bottom) against
# the target FDR, one column per base solver (TLARS / TOMP / TAFS) and one line
# per SNR. The figure is written to simulation_results/plots/.
#
# Usage:
#   ./generate_plots.sh                    # this demo's CSV
#   ./generate_plots.sh --formats png      # extra args pass through to the plotter
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

rscript="${RSCRIPT:-$(command -v Rscript || true)}"
if [[ -z "$rscript" ]]; then
  echo "error: no Rscript found on PATH (set RSCRIPT=/path/to/Rscript)" >&2
  exit 1
fi

csv="$here/simulation_results/data/demo_trex_09_mc_sim_target_fdr.csv"
if [[ ! -f "$csv" ]]; then
  echo "error: no result CSV found ($csv); run the demo first." >&2
  exit 1
fi

exec "$rscript" "$here/../trex_plt_utils.R" sweep "$csv" \
  --x tfdr --lines snr --diagonal \
  --title 'T-Rex: achieved FDR and TPR vs. target FDR (n = 300, p = 1,000, s = 10)' \
  --stem demo_trex_09_mc_sim_target_fdr \
  --outdir "$here/simulation_results/plots" \
  "$@"
