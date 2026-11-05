#!/usr/bin/env bash
#
# Regenerate this demo's DA-TRex+NN figures via the suite plotting module
#   ../trex_da_plt_utils.py
# using the repo's local .venv. Figures are written to
# simulation_results/plots/.
#
# Demo 07 (NN: banded MA(kappa) data + AR(1) method-mismatch study) produces:
#   * per-CSV FDR/TPR overviews (+ interactive html with --plotly) for the SNR sweeps on NN
#     data, each comparing DA-TRex+NN against the classical (no-DA) T-Rex baseline;
#   * a support-placement comparison grid (CappedSpread vs Random columns);
#   * the kappa x rho 2D study on NN data (per-solver TPR/FDR heatmaps, one
#     figure per support placement);
#   * the SNR x rho 2D method-mismatch study (DA+NN on AR(1) data, same
#     heatmap pattern).
#
# Usage:
#   ./generate_plots.sh                 # this demo's full figure set
#   ./generate_plots.sh <other.csv>     # just that CSV (default FDR/TPR mode)
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
ma_rho='MA coefficient $\rho$'

# First non-flag argument: plot just that CSV instead of the full set.
if [[ $# -gt 0 && "$1" != -* ]]; then
  csv="$1"
  shift
  exec "$venv_python" "$plotter" "$csv" --legend-title 'Method' --tfdr "$tfdr" "$@"
fi

# --- Per-CSV overviews: SNR sweeps on NN data, DA-TRex+NN vs base T-Rex -----
"$venv_python" "$plotter" "$data/da_trex_mc_da_nn_snr_capped.csv" \
  --title 'DA-TRex+NN vs base T-Rex on NN ($\kappa=3$, $\rho=0.7$): SNR sweep (CappedSpread support)' \
  --legend-title 'Method' --tfdr "$tfdr" "$@"
"$venv_python" "$plotter" "$data/da_trex_mc_da_nn_snr_random.csv" \
  --title 'DA-TRex+NN vs base T-Rex on NN ($\kappa=3$, $\rho=0.7$): SNR sweep (Random support)' \
  --legend-title 'Method' --tfdr "$tfdr" "$@"

# --- Support-placement comparison grid (CappedSpread vs Random columns) -----
"$venv_python" "$plotter" grid \
  --labels CappedSpread Random \
  --csvs "$data/da_trex_mc_da_nn_snr_capped.csv" "$data/da_trex_mc_da_nn_snr_random.csv" \
  --title 'DA-TRex+NN on NN ($\kappa=3$, $\rho=0.7$): SNR sweep by support placement' \
  --legend-title 'Method' --column-title 'Support' \
  --stem da_trex_mc_da_nn_snr_support_grid --outdir "$plots" --tfdr "$tfdr" "$@"

# --- kappa x rho 2D study on NN data (per-solver TPR/FDR heatmaps) ----------
"$venv_python" "$plotter" sweep2d "$data/da_trex_mc_da_nn_kappa_rho.csv" \
  --title 'DA-TRex+NN on NN data: bandwidth ($\kappa$) vs MA coefficient ($\rho$), SNR=2' \
  --xlabel "$ma_rho" \
  --stem da_trex_mc_da_nn_kappa_rho --outdir "$plots" --tfdr "$tfdr" "$@"

# --- SNR x rho 2D method-mismatch study (DA-TRex+NN on AR(1) data) ----------
"$venv_python" "$plotter" sweep2d "$data/da_trex_mc_da_nn_ar_snr_rho.csv" \
  --title 'DA-TRex+NN on AR(1) data (method mismatch): SNR vs correlation ($\rho$)' \
  --stem da_trex_mc_da_nn_ar_snr_rho --outdir "$plots" --tfdr "$tfdr" "$@"
