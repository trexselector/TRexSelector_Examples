#!/usr/bin/env bash
#
# Render one design-correlation heatmap per DGP in this suite (natural order,
# no clustering) into dgp_corr_plots/, via the shared plotter ../corr_plt_utils.py.
#
#   ./generate_corr_plots.sh
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Find the repo-root .venv by walking up the directory tree.
dir="$here"; venv_python=""
while [[ "$dir" != "/" ]]; do
  if [[ -x "$dir/.venv/bin/python" ]]; then venv_python="$dir/.venv/bin/python"; break; fi
  dir="$(dirname "$dir")"
done
[[ -z "$venv_python" ]] && { echo "error: no .venv/bin/python found above $here" >&2; exit 1; }

plotter="$here/../../corr_plt_utils.py"
outdir="$here/dgp_corr_plots"

# One DGP -> one figure.
keys=(
  trex_gauss
)
for k in "${keys[@]}"; do
  "$venv_python" "$plotter" "$k" --out "$outdir/$k.png" "$@"
done
