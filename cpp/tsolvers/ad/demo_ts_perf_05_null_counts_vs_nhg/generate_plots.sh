#!/usr/bin/env bash
#
# Regenerate this demo's figures via the plotting module:
#   ../ts_plt_utils.py nhg
# using the repo's local .venv: the histogram and the Q-Q plot of the included
# null variables against the negative hypergeometric distribution, and the
# ECDFs of the selected null variables. Figures are written to
# simulation_results/plots/.
#
# Usage:
#   ./generate_plots.sh                          # the 500-run result file
#   ./generate_plots.sh 4                        # the file of a 4-run test
#   ./generate_plots.sh --formats png            # extra args pass through to
#                                                # ts_plt_utils.py
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
  echo "       create it and 'pip install matplotlib pandas', then re-run." >&2
  exit 1
fi

# First non-flag argument selects the number of runs in the file name.
mc=500
if [[ $# -gt 0 && "$1" != -* ]]; then
  mc="$1"
  shift
fi

csv="$here/simulation_results/data/null_counts_vs_nhg_mc$mc.csv"
if [[ ! -f "$csv" ]]; then
  echo "error: no result file for $mc runs ($csv)" >&2
  exit 1
fi

exec "$venv_python" "$here/../../ts_plt_utils.py" nhg "$csv" \
  --title "T-solvers (\$L = p\$, $mc runs)" \
  "$@"
