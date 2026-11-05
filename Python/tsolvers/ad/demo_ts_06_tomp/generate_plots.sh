#!/usr/bin/env bash
#
# Regenerate this demo's path figure via the plotting module:
#   ../ts_plt_utils.py
# using the repo's local .venv. Figures are written to
# simulation_results/plots/.
#
# Usage:
#   ./generate_plots.sh                          # this demo's default CSV
#   ./generate_plots.sh <other_path.csv>         # explicit CSV
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
csv="$here/simulation_results/data/tomp_terminated_paths_path.csv"
if [[ $# -gt 0 && "$1" != -* ]]; then
  csv="$1"
  shift
fi

exec "$venv_python" "$here/../../ts_plt_utils.py" paths "$csv" \
  --solver 'T-OMP' \
  --title 'T-OMP: terminated solution paths ($n = 150$, $L = p = 300$, warm start $T = 1 \to 2 \to 5$)' \
  "$@"
