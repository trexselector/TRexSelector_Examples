#!/usr/bin/env bash
#
# Regenerate this demo's runtime figures via the plotting module:
#   ../ts_plt_utils.py
# using the repo's local .venv. Figures are written to
# simulation_results/plots/.
#
# Usage:
#   ./generate_plots.sh                          # this demo's default CSV
#   ./generate_plots.sh <other.csv>              # explicit CSV, e.g. a smoke run
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

# First non-flag argument overrides this demo's default CSV.
csv="$here/simulation_results/data/runtime_vs_dummies_mc500.csv"
if [[ $# -gt 0 && "$1" != -* ]]; then
  csv="$1"
  shift
fi

exec "$venv_python" "$here/../../ts_plt_utils.py" runtime "$csv" \
  --title 'T-solvers ($n = 500$, $p = 1{,}000$, $T = 10$, 500 runs)' \
  --p 1000 \
  "$@"
