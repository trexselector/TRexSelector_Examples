#!/usr/bin/env bash
#
# Regenerate this demo's DA-TRex figures via the suite plotting module
#   ../trex_da_plt_utils.R  (ggplot2)
# Figures are written to simulation_results/plots/.
#
# Demo 06 (multi-level prior-groups DGP) currently exercises a single SNR sweep,
# comparing the three DA-TRex solvers (TLARS, TAFS, TOMP) on a three-level nested
# dependency structure. This script produces its FDR/TPR overview.
#
# Usage:
#   ./generate_plots.sh                 # this demo's default CSV
#   ./generate_plots.sh <other.csv>     # explicit CSV
#   ./generate_plots.sh --formats png   # extra args pass through to the plotter
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

rscript="${RSCRIPT:-$(command -v Rscript || true)}"
if [[ -z "$rscript" ]]; then
  echo "error: no Rscript found on PATH (set RSCRIPT=/path/to/Rscript)" >&2
  exit 1
fi

plotter="$here/../trex_da_plt_utils.R"

# First non-flag argument overrides this demo's default CSV.
csv="$here/simulation_results/data/da_trex_mc_da_groups_toeplitz_leaf.csv"
if [[ $# -gt 0 && "$1" != -* ]]; then
  csv="$1"
  shift
fi

exec "$rscript" "$plotter" "$csv" \
  --title 'DA-TRex+AKG (a priori known groups) on 3-level nested data: SNR sweep' \
  --legend-title 'Method' --relabel 'TREX-DA=TREX-DA+AKG' --tfdr 0.2 "$@"
