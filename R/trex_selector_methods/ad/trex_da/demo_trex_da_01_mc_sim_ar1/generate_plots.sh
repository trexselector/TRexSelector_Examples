#!/usr/bin/env bash
#
# Regenerate this demo's DA-TRex+AR1 figures via the suite plotting module
#   ../trex_da_plt_utils.R  (ggplot2)
# Figures are written to simulation_results/plots/.
#
# Demo 01 (AR(1) Toeplitz) produces:
#   * per-CSV FDR/TPR overviews for the SNR and rho sweeps, each comparing
#     DA-TRex+AR1 against the classical (no-DA) T-Rex baseline;
#   * two support-placement comparison grids (CappedSpread vs Random columns);
#   * the gap x rho 2D study (kappa-boundary heatmaps) from the wide CSV.
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

# First non-flag argument: plot just that CSV instead of the full set.
if [[ $# -gt 0 && "$1" != -* ]]; then
  csv="$1"
  shift
  exec "$rscript" "$plotter" "$csv" --legend-title 'Method' --tfdr "$tfdr" "$@"
fi

# --- Per-CSV overviews: SNR and rho sweeps, DA-TRex+AR1 vs base T-Rex ---------
"$rscript" "$plotter" "$data/da_trex_mc_da_ar1_snr_capped.csv" \
  --title 'DA-TRex+AR1 vs base T-Rex on AR(1): SNR sweep at ρ=0.7 (CappedSpread support)' \
  --legend-title 'Method' --tfdr "$tfdr" "$@"
"$rscript" "$plotter" "$data/da_trex_mc_da_ar1_snr_random.csv" \
  --title 'DA-TRex+AR1 vs base T-Rex on AR(1): SNR sweep at ρ=0.7 (Random support)' \
  --legend-title 'Method' --tfdr "$tfdr" "$@"
"$rscript" "$plotter" "$data/da_trex_mc_da_ar1_rho_capped.csv" \
  --title 'DA-TRex+AR1 vs base T-Rex on AR(1): ρ sweep (CappedSpread support)' \
  --legend-title 'Method' --tfdr "$tfdr" "$@"
"$rscript" "$plotter" "$data/da_trex_mc_da_ar1_rho_random.csv" \
  --title 'DA-TRex+AR1 vs base T-Rex on AR(1): ρ sweep (Random support)' \
  --legend-title 'Method' --tfdr "$tfdr" "$@"

# --- Support-placement comparison grids (CappedSpread vs Random columns) ----
"$rscript" "$plotter" grid \
  --labels CappedSpread Random \
  --csvs "$data/da_trex_mc_da_ar1_snr_capped.csv" "$data/da_trex_mc_da_ar1_snr_random.csv" \
  --title 'DA-TRex+AR1 on AR(1): SNR sweep at ρ=0.7 by support placement' \
  --legend-title 'Method' --column-title 'Support' \
  --stem da_trex_mc_da_ar1_snr_support_grid --outdir "$plots" --tfdr "$tfdr" "$@"
"$rscript" "$plotter" grid \
  --labels CappedSpread Random \
  --csvs "$data/da_trex_mc_da_ar1_rho_capped.csv" "$data/da_trex_mc_da_ar1_rho_random.csv" \
  --title 'DA-TRex+AR1 on AR(1): ρ sweep by support placement' \
  --legend-title 'Method' --column-title 'Support' \
  --stem da_trex_mc_da_ar1_rho_support_grid --outdir "$plots" --tfdr "$tfdr" "$@"

# --- gap x rho 2D study (kappa-boundary heatmaps + Random baseline) ---------
"$rscript" "$plotter" gaprho "$data/da_trex_mc_da_ar1_gap_rho.csv" \
  --title 'DA-TRex+AR1 on AR(1): support spacing (gap) vs correlation (ρ)' \
  --stem da_trex_mc_da_ar1_gap_rho --outdir "$plots" --tfdr "$tfdr" "$@"
