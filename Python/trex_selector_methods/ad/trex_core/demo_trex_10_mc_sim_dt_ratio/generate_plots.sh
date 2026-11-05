#!/usr/bin/env bash
#
# Regenerate this demo's figures via the suite plotter's 'sweep' mode
#   ../trex_plt_utils.py sweep <csv> ...
# using the repo's local .venv: FDR (top, with the tFDR line) and TPR (bottom)
# against c = n / (s ln p), one column per base solver (TLARS / TOMP / TAFS)
# and one line per p, one figure per SNR. Figures are written to
# simulation_results/plots/.
#
# Usage:
#   ./generate_plots.sh                    # this demo's CSV
#   ./generate_plots.sh --formats png      # extra args pass through to the plotter
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Find the repo-root .venv by walking up the directory tree.
dir="$here"
venv_python=""
while [[ "$dir" != "/" ]]; do
  if [[ -x "$dir/.venv/bin/python" ]]; then
    venv_python="$dir/.venv/bin/python"
    break
  fi
  dir="$(dirname "$dir")"
done

if [[ -z "$venv_python" ]]; then
  echo "error: no .venv/bin/python found above $here" >&2
  echo "       create it and 'pip install matplotlib pandas', then re-run." >&2
  exit 1
fi

csv="$here/simulation_results/data/demo_trex_10_mc_sim_dt_ratio.csv"
if [[ ! -f "$csv" ]]; then
  echo "error: no result CSV found ($csv); run the demo first." >&2
  exit 1
fi

for snr in 0.5 1 2; do
  "$venv_python" "$here/../trex_plt_utils.py" sweep "$csv" \
    --x c --lines p --filter "snr=$snr" --tfdr 0.1 \
    --title "T-Rex: FDR and TPR vs. \$n / (s \\cdot \\ln p)\$ (\$s = 10\$, SNR = $snr, tFDR = 0.1)" \
    --stem "demo_trex_10_mc_sim_dt_ratio_snr${snr}" \
    --outdir "$here/simulation_results/plots" \
    "$@"
done
