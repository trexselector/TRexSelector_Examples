#!/usr/bin/env python3
"""
merge_result_files.py — add the tables of a solver-selection run to the full tables.

A demo run with TREX_SOLVERS=<names> computes only those solvers and writes its
tables next to the full ones with a suffix, e.g.

    da_trex_mc_da_ar1_snr_capped_tlars_stag.{txt,csv}   (TREX_SOLVERS=TLARS_STAG)
    da_trex_mc_da_ar1_snr_capped.{txt,csv}              (full solver pool)

This script splices the selected solvers' blocks into the full .txt and .csv,
exactly where a full run with the extended pool would write them: each new
solver goes after the last solver of its kind ("TREX-DA+AR1: ...", "TREX: ...").
A solver already in the full table is replaced, so a merge can be repeated.

Both files are written by the same C++ writers, so the merge works on their
text and keeps every number and every column as printed. A header that differs
between the two files (other grid, other MC count) stops the merge.

Usage (from a demo folder or with explicit data folders):
    python merge_result_files.py --suffix _tlars_stag simulation_results/data
    python merge_result_files.py --suffix _tlars_stag --trash demo_*/simulation_results/data
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

RULE = "=" * 78
DASH = "-" * 78


def kind(solver: str) -> str:
    """Row group of a solver label: the part before ": " ("TREX-DA+BT", "TREX")."""
    return solver.split(": ", 1)[0] if ": " in solver else ""


def merged_order(base: list[str], new: list[str]) -> list[str]:
    """Solver order after the merge: a new solver follows the last base solver of its kind."""
    order = list(base)
    for name in new:
        if name in order:
            continue
        same = [i for i, s in enumerate(order) if kind(s) == kind(name)]
        order.insert(same[-1] + 1 if same else len(order), name)
    return order


# ==============================================================================
# CSV
# ==============================================================================

def split_csv(lines: list[str]) -> tuple[str, dict[str, list[str]], list[str]]:
    """Header line, rows per solver and solver order; each solver's rows must be contiguous."""
    header = lines[0]
    blocks: dict[str, list[str]] = {}
    order: list[str] = []
    for line in lines[1:]:
        name = line.split(",", 1)[0]
        if name not in blocks:
            blocks[name] = []
            order.append(name)
        elif order[-1] != name:
            raise ValueError(f"rows of {name!r} are not contiguous")
        blocks[name].append(line)
    return header, blocks, order


def merge_csv(base_text: str, new_text: str) -> str:
    base_h, base_b, base_o = split_csv(base_text.splitlines(keepends=True))
    new_h, new_b, new_o = split_csv(new_text.splitlines(keepends=True))
    if base_h != new_h:
        raise ValueError(f"CSV headers differ: {base_h.strip()!r} vs {new_h.strip()!r}")
    blocks = {**base_b, **new_b}
    return base_h + "".join(line for name in merged_order(base_o, new_o) for line in blocks[name])


# ==============================================================================
# TXT
# ==============================================================================

def split_txt(lines: list[str], solvers: list[str]) -> tuple[list[str], dict[str, list[str]], list[str]]:
    """Header, block per solver and footer of a table written by a trex_da writer.

    1-D tables open a block with a blank line and the solver name. 2-D tables
    open it with a dashed rule and "  Solver: <name>". The footer is the
    closing "=" rule (with the blank line before it in 1-D tables).
    """
    stripped = [line.rstrip("\n") for line in lines]
    two_d = any(line == f"  Solver: {solvers[0]}" for line in stripped)
    starts = []
    for name in solvers:
        marker = f"  Solver: {name}" if two_d else name
        idx = [i for i, line in enumerate(stripped) if line == marker]
        if len(idx) != 1:
            raise ValueError(f"block of {name!r} found {len(idx)} times")
        opener = stripped[idx[0] - 1]
        if opener != (DASH if two_d else ""):
            raise ValueError(f"unexpected line before the block of {name!r}: {opener!r}")
        starts.append(idx[0] - 1)
    if starts != sorted(starts):
        raise ValueError("TXT blocks are not in CSV order")
    rules = [i for i, line in enumerate(stripped) if line == RULE and i > starts[-1]]
    if not rules:
        raise ValueError("no closing rule after the blocks (table from an older writer?)")
    end = rules[-1] if two_d else rules[-1] - 1
    if not two_d and stripped[end] != "":
        raise ValueError("no blank line before the closing rule")
    bounds = [*starts, end]
    blocks = {name: lines[bounds[i]:bounds[i + 1]] for i, name in enumerate(solvers)}
    return lines[:starts[0]], blocks, lines[end:]


def merge_txt(base_text: str, new_text: str, base_solvers: list[str], new_solvers: list[str]) -> str:
    base_h, base_b, base_f = split_txt(base_text.splitlines(keepends=True), base_solvers)
    new_h, new_b, new_f = split_txt(new_text.splitlines(keepends=True), new_solvers)
    if base_h != new_h or base_f != new_f:
        raise ValueError("TXT headers or footers differ (other grid or MC count?)")
    blocks = {**base_b, **new_b}
    order = merged_order(base_solvers, new_solvers)
    return "".join(base_h) + "".join(line for name in order for line in blocks[name]) + "".join(base_f)


# ==============================================================================
# Files
# ==============================================================================

def merge_pair(base_stem: Path, extra_stem: Path) -> str:
    """Merge extra_stem.{csv,txt} into base_stem.{csv,txt}; returns the added solvers."""
    base_csv, new_csv = (Path(f"{s}.csv").read_text() for s in (base_stem, extra_stem))
    base_txt, new_txt = (Path(f"{s}.txt").read_text() for s in (base_stem, extra_stem))
    base_solvers = split_csv(base_csv.splitlines(keepends=True))[2]
    new_solvers = split_csv(new_csv.splitlines(keepends=True))[2]
    csv_out = merge_csv(base_csv, new_csv)
    txt_out = merge_txt(base_txt, new_txt, base_solvers, new_solvers)
    Path(f"{base_stem}.csv").write_text(csv_out)
    Path(f"{base_stem}.txt").write_text(txt_out)
    return ", ".join(new_solvers)


def move_to_trash(path: Path) -> None:
    """Move a file to the user's trash (macOS ~/.Trash), never deleting it."""
    trash = Path.home() / ".Trash"
    target = trash / path.name
    if target.exists():
        target = trash / f"{path.stem}_{time.strftime('%Y%m%d_%H%M%S')}{path.suffix}"
    shutil.move(str(path), str(target))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("data_dirs", nargs="+", type=Path, help="simulation_results/data folders")
    parser.add_argument("--suffix", required=True, help="file suffix of the selection run, e.g. _tlars_stag")
    parser.add_argument("--trash", action="store_true", help="move the merged extra files to ~/.Trash")
    args = parser.parse_args()

    merged = 0
    for data_dir in args.data_dirs:
        for extra_csv in sorted(data_dir.glob(f"*{args.suffix}.csv")):
            extra_stem = extra_csv.with_suffix("")
            base_stem = extra_stem.with_name(extra_stem.name[: -len(args.suffix)])
            if not Path(f"{base_stem}.csv").exists() or not Path(f"{base_stem}.txt").exists():
                print(f"skip  {extra_csv} (no full table {base_stem.name}.csv/.txt)", file=sys.stderr)
                continue
            added = merge_pair(base_stem, extra_stem)
            merged += 1
            print(f"merged {extra_stem.name} -> {base_stem.name}  ({added})")
            if args.trash:
                for ext in (".csv", ".txt"):
                    move_to_trash(Path(f"{extra_stem}{ext}"))
    print(f"{merged} table pair(s) merged")
    return 0


if __name__ == "__main__":
    sys.exit(main())
