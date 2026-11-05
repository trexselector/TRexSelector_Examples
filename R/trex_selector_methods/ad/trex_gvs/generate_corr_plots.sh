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
  gvs_hastie
  gvs_scattered
  gvs_mixed_blocks
  gvs_neg_traps
  gvs_t3_blocks
  gvs_ar1_blocks
  gvs_arma_blocks
  gvs_block_equicorr
)
for k in "${keys[@]}"; do
  "$rscript" "$plotter" "$k" --out "$outdir/$k.png" "$@"
done
