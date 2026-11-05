#!/usr/bin/env bash
#
# Regenerate this demo's DA-TRex+BT figures via the suite plotting module
#   ../trex_da_plt_utils.R  (ggplot2)
# Figures are written to simulation_results/plots/.
#
# Demo 03 (BT on block-diagonal AR(1)) sweeps SNR, rho, block size Q, and number
# of blocks M, each looped over HAC linkage {Single, Complete, Average}. This
# script produces:
#   * a per-CSV FDR/TPR overview for every sweep x linkage;
#   * one linkage-comparison grid per sweep (Single/Complete/Average columns) --
#     the figure for this demo's core question, whether the choice of HAC linkage
#     materially changes FDR/TPR.
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
prefix="da_trex_mc_da_ar1_blocks"
tfdr=0.2

# First non-flag argument: plot just that CSV instead of the full set.
if [[ $# -gt 0 && "$1" != -* ]]; then
  csv="$1"
  shift
  exec "$rscript" "$plotter" "$csv" --legend-title 'Method' --tfdr "$tfdr" "$@"
fi

# Human-readable sweep labels for the figure titles (a function rather than an
# associative array, so this stays compatible with macOS's stock bash 3.2).
sweep_label() {
  case "$1" in
    snr) echo 'SNR sweep' ;;
    rho) echo 'ρ sweep' ;;
    Q)   echo 'block-size Q sweep' ;;
    M)   echo 'number-of-blocks M sweep' ;;
  esac
}

for sweep in snr rho Q M; do
  label="$(sweep_label "$sweep")"

  # Per-linkage overview.
  for linkage in Single Complete Average; do
    csv="$data/${prefix}_${sweep}_${linkage}.csv"
    "$rscript" "$plotter" "$csv" \
      --title "DA-TRex+BT on block AR(1): ${label}, ${linkage} linkage" \
      --legend-title 'Method' --tfdr "$tfdr" "$@"
  done

  # Linkage-comparison grid (Single/Complete/Average columns) for this sweep.
  "$rscript" "$plotter" grid \
    --labels Single Complete Average \
    --csvs "$data/${prefix}_${sweep}_Single.csv" \
           "$data/${prefix}_${sweep}_Complete.csv" \
           "$data/${prefix}_${sweep}_Average.csv" \
    --title "DA-TRex+BT on block AR(1): ${label} across HAC linkage" \
    --legend-title 'Method' --column-title 'Linkage' \
    --stem "${prefix}_${sweep}_linkage_grid" --outdir "$plots" --tfdr "$tfdr" "$@"
done
