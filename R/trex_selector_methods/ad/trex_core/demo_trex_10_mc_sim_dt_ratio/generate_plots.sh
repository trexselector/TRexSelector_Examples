#!/usr/bin/env bash
#
# Regenerate this demo's figures via the suite plotter's 'sweep' mode
#   ../trex_plt_utils.R sweep <csv> ...  (ggplot2)
# FDR (top, with the tFDR line) and TPR (bottom) against c = n / (s ln p), one
# column per base solver (TLARS / TOMP / TAFS) and one line per p, one figure
# per SNR. Figures are written to simulation_results/plots/.
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

csv="$here/simulation_results/data/demo_trex_10_mc_sim_dt_ratio.csv"
if [[ ! -f "$csv" ]]; then
  echo "error: no result CSV found ($csv); run the demo first." >&2
  exit 1
fi

for snr in 0.5 1 2; do
  "$rscript" "$here/../trex_plt_utils.R" sweep "$csv" \
    --x c --lines p --filter "snr=$snr" --tfdr 0.1 \
    --title "T-Rex: FDR and TPR vs. n / (s · ln p) (s = 10, SNR = $snr, tFDR = 0.1)" \
    --stem "demo_trex_10_mc_sim_dt_ratio_snr${snr}" \
    --outdir "$here/simulation_results/plots" \
    "$@"
done
