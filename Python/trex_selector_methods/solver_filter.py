# ==============================================================================
# solver_filter.py
# ==============================================================================
#
# Optional solver selection for the Monte Carlo demos. Mirrors
# cpp/trex_selector_methods/solver_filter.hpp.
#
# TREX_SOLVERS (comma separated solver names, e.g. "TSTAGEWISE") restricts a
# demo to those solvers. A name is matched against the part after ": " of a
# row label, so "TLARS_STAG" selects "TREX-DA+BT: TLARS_STAG" as well. Unset
# or empty selects every solver.
#
# A selection run writes its tables to extra files (file_suffix()), so the
# committed results of the full solver pool stay untouched. The suite's
# merge_result_files.py (in the cpp tree) adds such extra files to the full
# tables.
#
# ==============================================================================

import os


def names():
    """Solver names listed in TREX_SOLVERS (empty: no selection)."""
    env = os.environ.get("TREX_SOLVERS", "")
    return [item.strip() for item in env.split(",") if item.strip()]


def selected(label):
    """True when a row label's solver is selected (always true without a selection)."""
    wanted = names()
    if not wanted:
        return True
    solver = label.split(": ", 1)[1] if ": " in label else label
    return solver in wanted


def file_suffix():
    """File-name suffix of a selection run, e.g. "_tstagewise" ("" without a selection)."""
    return "".join("_" + name.lower() for name in names())
