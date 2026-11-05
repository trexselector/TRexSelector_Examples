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
  --csvs "$data/tafs_rho_0.1_terminated_paths_path.csv" \
         "$data/tafs_rho_0.3_terminated_paths_path.csv" \
         "$data/tafs_rho_0.5_terminated_paths_path.csv" \
         "$data/tafs_rho_1.0_terminated_paths_path.csv" \
  --labels '$\rho = 0.1$' '$\rho = 0.3$' '$\rho = 0.5$' '$\rho = 1$' \
  --title 'T-AFS: terminated solution paths for four $\rho$ ($n = 150$, $L = p = 300$, warm start $T = 1 \to 2 \to 5$)' \
  --stem tafs_terminated_paths_rho \
  "$@"
