#!/usr/bin/env bash
#
# Render one design-correlation heatmap per DGP in this suite (natural order,
# no clustering) into dgp_corr_plots/, via the shared plotter
# ../corr_plt_utils.R (ggplot2).
#
#   ./generate_corr_plots.sh
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

rscript="${RSCRIPT:-$(command -v Rscript || true)}"
if [[ -z "$rscript" ]]; then
  echo "error: no Rscript found on PATH (set RSCRIPT=/path/to/Rscript)" >&2
  exit 1
fi

plotter="$here/../../corr_plt_utils.R"
outdir="$here/dgp_corr_plots"

# One DGP -> one figure.
keys=(
  da_ar1
  da_equi
  da_nn
  da_ar1_block
  da_ar1_block_white
  da_block_toeplitz_hvt
  da_ht_block_white
  da_groups
)
for k in "${keys[@]}"; do
  "$rscript" "$plotter" "$k" --out "$outdir/$k.png" "$@"
done
