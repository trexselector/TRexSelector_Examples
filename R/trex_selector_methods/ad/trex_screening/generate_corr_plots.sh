#!/usr/bin/env bash
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

if [[ ! -f "$plotter" ]]; then
  echo "error: shared corr plotter not found: $plotter" >&2
  exit 1
fi

# One DGP -> one figure.
keys=(
  scr_iid
  scr_ar1
  scr_equi
  scr_block_equi
)
for k in "${keys[@]}"; do
  "$rscript" "$plotter" "$k" --out "$outdir/$k.png" "$@"
done
