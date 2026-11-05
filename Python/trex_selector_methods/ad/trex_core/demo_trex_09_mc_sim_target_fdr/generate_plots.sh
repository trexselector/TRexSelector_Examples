#!/usr/bin/env bash
#
# Regenerate this demo's figure via the suite plotter's 'sweep' mode
#   ../trex_plt_utils.py sweep <csv> ...
# using the repo's local .venv: achieved FDR (top, with the diagonal
# FDR = tFDR) and TPR (bottom) against the target FDR, one column per base
# solver (TLARS / TOMP / TAFS) and one line per SNR. The figure is written to
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

csv="$here/simulation_results/data/demo_trex_09_mc_sim_target_fdr.csv"
if [[ ! -f "$csv" ]]; then
  echo "error: no result CSV found ($csv); run the demo first." >&2
  exit 1
fi

exec "$venv_python" "$here/../trex_plt_utils.py" sweep "$csv" \
  --x tfdr --lines snr --diagonal \
  --title 'T-Rex: achieved FDR and TPR vs. target FDR ($n = 300$, $p = 1{,}000$, $s = 10$)' \
  --stem demo_trex_09_mc_sim_target_fdr \
  --outdir "$here/simulation_results/plots" \
  "$@"
