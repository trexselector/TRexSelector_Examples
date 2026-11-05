#!/usr/bin/env bash
#
# Render this suite's DGP correlation heatmap(s) via the shared R plotter
#   ../corr_plt_utils.R  (ggplot2)
# into dgp_corr_plots/. Invoked by run_overnight.sh after the demos finish;
# safe to run standalone.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

rscript="${RSCRIPT:-$(command -v Rscript || true)}"
if [[ -z "$rscript" ]]; then
  echo "error: no Rscript found on PATH (set RSCRIPT=/path/to/Rscript)" >&2
  exit 1
fi

plotter="$here/../../corr_plt_utils.R"
outdir="$here/dgp_corr_plots"

keys=(
  spca_factor
)

for k in "${keys[@]}"; do
  "$rscript" "$plotter" "$k" --out "$outdir/$k.png" "$@"
done
