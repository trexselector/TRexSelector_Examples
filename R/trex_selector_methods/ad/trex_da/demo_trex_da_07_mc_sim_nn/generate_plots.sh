#!/usr/bin/env bash
#
# Regenerate this demo's DA-TRex+NN figures via the suite plotting module
#   ../trex_da_plt_utils.R  (ggplot2)
# Figures are written to simulation_results/plots/.
#
# Demo 07 (NN: banded MA(kappa) data + AR(1) method-mismatch study) produces:
#   * per-CSV FDR/TPR overviews for the SNR sweeps on NN data, each comparing
#     DA-TRex+NN against the classical (no-DA) T-Rex baseline;
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

rscript="${RSCRIPT:-$(command -v Rscript || true)}"
if [[ -z "$rscript" ]]; then
  echo "error: no Rscript found on PATH (set RSCRIPT=/path/to/Rscript)" >&2
  exit 1
fi

plotter="$here/../trex_da_plt_utils.R"
data="$here/simulation_results/data"
plots="$here/simulation_results/plots"
tfdr=0.2
ma_rho='MA coefficient ρ'

# First non-flag argument: plot just that CSV instead of the full set.
if [[ $# -gt 0 && "$1" != -* ]]; then
  csv="$1"
  shift
  exec "$rscript" "$plotter" "$csv" --legend-title 'Method' --tfdr "$tfdr" "$@"
fi

# --- Per-CSV overviews: SNR sweeps on NN data, DA-TRex+NN vs base T-Rex -----
"$rscript" "$plotter" "$data/da_trex_mc_da_nn_snr_capped.csv" \
  --title 'DA-TRex+NN vs base T-Rex on NN (κ=3, ρ=0.7): SNR sweep (CappedSpread support)' \
  --legend-title 'Method' --tfdr "$tfdr" "$@"
"$rscript" "$plotter" "$data/da_trex_mc_da_nn_snr_random.csv" \
  --title 'DA-TRex+NN vs base T-Rex on NN (κ=3, ρ=0.7): SNR sweep (Random support)' \
  --legend-title 'Method' --tfdr "$tfdr" "$@"

# --- Support-placement comparison grid (CappedSpread vs Random columns) -----
"$rscript" "$plotter" grid \
  --labels CappedSpread Random \
  --csvs "$data/da_trex_mc_da_nn_snr_capped.csv" "$data/da_trex_mc_da_nn_snr_random.csv" \
  --title 'DA-TRex+NN on NN (κ=3, ρ=0.7): SNR sweep by support placement' \
  --legend-title 'Method' --column-title 'Support' \
  --stem da_trex_mc_da_nn_snr_support_grid --outdir "$plots" --tfdr "$tfdr" "$@"

# --- kappa x rho 2D study on NN data (per-solver TPR/FDR heatmaps) ----------
"$rscript" "$plotter" sweep2d "$data/da_trex_mc_da_nn_kappa_rho.csv" \
  --title 'DA-TRex+NN on NN data: bandwidth (κ) vs MA coefficient (ρ), SNR=2' \
  --xlabel "$ma_rho" \
  --stem da_trex_mc_da_nn_kappa_rho --outdir "$plots" --tfdr "$tfdr" "$@"

# --- SNR x rho 2D method-mismatch study (DA-TRex+NN on AR(1) data) ----------
"$rscript" "$plotter" sweep2d "$data/da_trex_mc_da_nn_ar_snr_rho.csv" \
  --title 'DA-TRex+NN on AR(1) data (method mismatch): SNR vs correlation (ρ)' \
  --stem da_trex_mc_da_nn_ar_snr_rho --outdir "$plots" --tfdr "$tfdr" "$@"
