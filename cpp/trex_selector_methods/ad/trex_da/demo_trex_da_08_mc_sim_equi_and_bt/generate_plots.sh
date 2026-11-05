#!/usr/bin/env bash
#
# Regenerate this demo's DA-TRex+EQUI figure via the suite plotting module
#   ../trex_da_plt_utils.py
# using the repo's local .venv. The figure is written to
# simulation_results/plots/.
#
# Demo 08 (Equi 2D sweep, SNR x rho) produces one exhibit: a 2x4 comparison
# grid (rows = FDR/TPR, columns = SNR in {0.5, 1, 2, 5}, x-axis =
# equicorrelation rho on a sqrt scale) showing the DA suppression cliff, its
# SNR independence, and the rho=0 anchor.
#
# Usage:
#   ./generate_plots.sh                 # the grid figure
#   ./generate_plots.sh <other.csv>     # one-off FDR/TPR overview for a CSV
#   ./generate_plots.sh --formats png   # extra args pass through to the plotter
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
  echo "       create it and 'pip install matplotlib pandas plotly', then re-run." >&2
  exit 1
fi

plotter="$here/../trex_da_plt_utils.py"
data="$here/simulation_results/data"
plots="$here/simulation_results/plots"
tfdr=0.2
xlabel='Equicorrelation $\rho$'

# First non-flag argument: one-off overview for that CSV instead of the grid.
if [[ $# -gt 0 && "$1" != -* ]]; then
  csv="$1"
  shift
  exec "$venv_python" "$plotter" "$csv" --legend-title 'Method' \
    --xlabel "$xlabel" --xscale sqrt --tfdr "$tfdr" "$@"
fi

# --- Main exhibit: SNR x rho comparison grid (columns = SNR) ----------------
"$venv_python" "$plotter" grid \
  --labels 'SNR=0.5' 'SNR=1' 'SNR=2' 'SNR=5' \
  --csvs "$data/da_trex_mc_da_equi_rho_snr0p5.csv" \
         "$data/da_trex_mc_da_equi_rho_snr1.csv" \
         "$data/da_trex_mc_da_equi_rho_snr2.csv" \
         "$data/da_trex_mc_da_equi_rho_snr5.csv" \
  --title 'DA-TRex+EQUI vs base T-Rex on Equi: $\rho$ sweep by SNR (Random support)' \
  --legend-title 'Method' --xlabel "$xlabel" --xscale sqrt \
  --stem da_trex_mc_da_equi_rho_snr_grid --outdir "$plots" --tfdr "$tfdr" "$@"
