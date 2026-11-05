#!/usr/bin/env bash
#
# Regenerate this demo's T-Rex+GVS figures via the suite plotting module
#   ../trex_gvs_plt_utils.py
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

plotter="$here/../trex_gvs_plt_utils.py"
data="$here/simulation_results/data"
plots="$here/simulation_results/plots"
tfdr=0.1

# Headline: value of prior group knowledge as a function of rho (WHOLE truth).
# Panels at weak / moderate / strong SNR; x-axis = rho crossing the
# corr_max = 0.5 discoverability boundary.
"$venv_python" "$plotter" "$data/gvs_block_bench_whole.csv" \
  --vs rho --panels 0.5 2 5 \
  --stem gvs_block_bench_whole_vs_rho \
  --title 'T-Rex+GVS block benchmark (WHOLE truth): prior groups vs HAC discoverability' \
  --tfdr "$tfdr" "$@"

# Supplementary: TPR/FDR vs SNR at three discoverability regimes
# (rho = 0.2 undiscoverable, 0.5 partial, 0.9 fully discovered).
"$venv_python" "$plotter" "$data/gvs_block_bench_whole.csv" \
  --vs snr --panels 0.2 0.5 0.9 \
  --stem gvs_block_bench_whole \
  --title 'T-Rex+GVS block benchmark (WHOLE truth): TPR/FDR vs SNR' \
  --tfdr "$tfdr" "$@"

# Cautionary appendix: INDIVIDUAL truth (exchangeable null siblings) —
# coordinate-level FDR is not controllable here; see the README.
"$venv_python" "$plotter" "$data/gvs_block_bench_individual.csv" \
  --title 'T-Rex+GVS block benchmark (INDIVIDUAL truth): TPR/FDR vs SNR' \
  --tfdr "$tfdr" "$@"
