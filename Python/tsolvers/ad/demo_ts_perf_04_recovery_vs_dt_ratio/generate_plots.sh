#!/usr/bin/env bash
#
# Regenerate this demo's recovery figures via the plotting module:
#   ../ts_plt_utils.py
# using the repo's local .venv. Figures are written to
# simulation_results/plots/.
#
# Usage:
#   ./generate_plots.sh                          # the 500-run result files
#   ./generate_plots.sh 4                        # the files of a 4-run test
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

# First non-flag argument selects the number of runs in the file names.
mc=500
if [[ $# -gt 0 && "$1" != -* ]]; then
  mc="$1"
  shift
fi

# Plot the parts whose result files exist.
data="$here/simulation_results/data"
inputs=()
[[ -f "$data/recovery_vs_dt_ratio_mc$mc.csv" ]] && inputs+=("$data/recovery_vs_dt_ratio_mc$mc.csv")
[[ -f "$data/phase_transition_mc$mc.csv" ]] && inputs+=(--phase "$data/phase_transition_mc$mc.csv")
if [[ ${#inputs[@]} -eq 0 ]]; then
  echo "error: no result files for $mc runs in $data" >&2
  exit 1
fi

exec "$venv_python" "$here/../../ts_plt_utils.py" recovery "${inputs[@]}" \
  --title "T-solvers (\$L = p\$, $mc runs)" \
  "$@"
