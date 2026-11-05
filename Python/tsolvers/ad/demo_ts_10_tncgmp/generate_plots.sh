#!/usr/bin/env bash
#
# Regenerate this demo's path comparison figure via the plotting module:
#   ../ts_plt_utils.py
# using the repo's local .venv. Figures are written to
# simulation_results/plots/.
#
# Usage:
#   ./generate_plots.sh                          # this demo's CSVs
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

data="$here/simulation_results/data"

exec "$venv_python" "$here/../../ts_plt_utils.py" compare \
  --csvs "$data/tncgmp_linesearch_terminated_paths_path.csv" \
         "$data/tncgmp_fullycorrective_terminated_paths_path.csv" \
  --labels 'Line search (= T-MP)' 'Fully corrective (= T-OMP)' \
  --title 'T-NCGMP: terminated solution paths of both variants ($n = 150$, $L = p = 300$, warm start $T = 1 \to 2 \to 5$)' \
  --stem tncgmp_terminated_paths_variants \
  "$@"
