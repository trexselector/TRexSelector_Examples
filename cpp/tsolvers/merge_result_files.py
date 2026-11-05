#!/usr/bin/env python3
"""
merge_result_files.py — put the rows of a solver-selection run into the full perf tables.

A demo_ts_perf_* run with TREX_SOLVERS=<labels> computes only those solvers and
writes its tables next to the full ones with a suffix, e.g.

    fdr_tpr_vs_T_mc500_t_stagewise.{txt,csv}   (TREX_SOLVERS=T-Stagewise)
    fdr_tpr_vs_T_mc500.{txt,csv}               (all demo solvers)

This script replaces the rows of the selected solvers in the full .txt and .csv
with the new ones, in place, so every table keeps its solver order and every
number the format of its writer (C++, Python or R, the layouts are the same).

The .txt holds several tables, each opened by a title line. A row belongs to a
solver when its label (the text before the first run of two blanks) is the
solver's name. Every other line of the selection run (titles, column labels,
rules, the NHG or "all solvers" rows, the oracle refit line) must appear in the
same table of the full file, so another grid or another MC count stops the
merge. The full file keeps its wall time. A selection run with failed solver
runs is refused.

The .csv rows of a solver form contiguous blocks. Each block of a selected
solver is replaced by the matching block of the selection run, which may hold
another number of rows (the count histograms of perf 05).

Usage (from a demo folder or with explicit data folders, any language):
    python merge_result_files.py --suffix _t_stagewise simulation_results/data
    python merge_result_files.py --suffix _t_stagewise --trash */simulation_results/data
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROW_LABEL = re.compile(r"^(\S(?:.*?\S)?)(?: {2,}|$)")


class MergeError(RuntimeError):
    """A table pair that cannot be merged safely."""


# ==============================================================================
# CSV
# ==============================================================================

def csv_blocks(lines: list[str]) -> list[tuple[str, list[str]]]:
    """Contiguous blocks of rows with the same solver (first field), in file order."""
    blocks: list[tuple[str, list[str]]] = []
    for line in lines:
        name = line.split(",", 1)[0]
        if blocks and blocks[-1][0] == name:
            blocks[-1][1].append(line)
        else:
            blocks.append((name, [line]))
    return blocks


def merge_csv(base_text: str, new_text: str) -> tuple[str, list[str]]:
    base, new = base_text.splitlines(keepends=True), new_text.splitlines(keepends=True)
    if base[0] != new[0]:
        raise MergeError(f"CSV headers differ: {base[0].strip()!r} vs {new[0].strip()!r}")
    new_blocks = csv_blocks(new[1:])
    solvers = list(dict.fromkeys(name for name, _ in new_blocks))
    queue = {name: [rows for n, rows in new_blocks if n == name] for name in solvers}
    out = [base[0]]
    for name, rows in csv_blocks(base[1:]):
        if name in queue:
            if not queue[name]:
                raise MergeError(f"CSV of the selection has fewer blocks of {name!r} than the full table")
            rows = queue[name].pop(0)
        out.extend(rows)
    for name, left in queue.items():
        if left:
            raise MergeError(f"CSV block of {name!r} has no place in the full table")
    return "".join(out), solvers


# ==============================================================================
# TXT
# ==============================================================================

def paragraphs(lines: list[str]) -> list[tuple[int, int]]:
    """(start, end) line ranges of the runs of non-blank lines."""
    out, i = [], 0
    while i < len(lines):
        if lines[i].strip():
            j = i
            while j < len(lines) and lines[j].strip():
                j += 1
            out.append((i, j))
            i = j
        else:
            i += 1
    return out


def row_label(line: str) -> str:
    m = ROW_LABEL.match(line.rstrip("\n"))
    return m.group(1) if m else ""


def merge_txt(base_text: str, new_text: str, solvers: list[str]) -> str:
    base, new = base_text.splitlines(keepends=True), new_text.splitlines(keepends=True)
    tables = {base[s]: (s, e) for s, e in paragraphs(base)}
    replaced = 0
    for s, e in paragraphs(new):
        title = new[s]
        if title.startswith(("Failed solver runs:", "Wall time:")):
            for line in new[s:e]:
                if line.startswith("Failed solver runs:") and line.split(":", 1)[1].strip() != "0":
                    raise MergeError(f"the selection run reports {line.strip()!r}")
            continue
        if title not in tables:
            raise MergeError(f"table {title.strip()!r} is missing in the full file (other grid?)")
        bs, be = tables[title]
        for line in new[s + 1:e]:
            label = row_label(line)
            if label in solvers:
                hits = [k for k in range(bs + 1, be) if row_label(base[k]) == label]
                if len(hits) != 1:
                    raise MergeError(f"row {label!r} found {len(hits)} times in {title.strip()!r}")
                base[hits[0]] = line
                replaced += 1
            elif line not in base[bs:be]:
                raise MergeError(f"line {line.strip()!r} of {title.strip()!r} differs from the full file "
                                 f"(other grid or MC count?)")
    if replaced == 0:
        raise MergeError("no row of the selected solvers in the TXT")
    return "".join(base)


# ==============================================================================
# Files
# ==============================================================================

def merge_pair(base_stem: Path, extra_stem: Path) -> str:
    """Merge extra_stem.{csv,txt} into base_stem.{csv,txt}; returns the merged solvers."""
    base_csv, new_csv = (Path(f"{s}.csv").read_text() for s in (base_stem, extra_stem))
    base_txt, new_txt = (Path(f"{s}.txt").read_text() for s in (base_stem, extra_stem))
    csv_out, solvers = merge_csv(base_csv, new_csv)
    txt_out = merge_txt(base_txt, new_txt, solvers)
    Path(f"{base_stem}.csv").write_text(csv_out)
    Path(f"{base_stem}.txt").write_text(txt_out)
    return ", ".join(solvers)


def move_to_trash(path: Path) -> None:
    """Move a file to the user's trash (the trash command, else ~/.Trash), never deleting it."""
    if shutil.which("trash"):
        subprocess.run(["trash", str(path)], check=True)
        return
    trash = Path.home() / ".Trash"
    target = trash / path.name
    if target.exists():
        target = trash / f"{path.stem}_{time.strftime('%Y%m%d_%H%M%S')}{path.suffix}"
    shutil.move(str(path), str(target))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("data_dirs", nargs="+", type=Path, help="simulation_results/data folders")
    parser.add_argument("--suffix", required=True, help="file suffix of the selection run, e.g. _t_stagewise")
    parser.add_argument("--trash", action="store_true", help="move the merged extra files to ~/.Trash")
    args = parser.parse_args()

    merged = 0
    failed = 0
    for data_dir in args.data_dirs:
        for extra_csv in sorted(data_dir.glob(f"*{args.suffix}.csv")):
            extra_stem = extra_csv.with_suffix("")
            base_stem = extra_stem.with_name(extra_stem.name[: -len(args.suffix)])
            if not Path(f"{base_stem}.csv").exists() or not Path(f"{base_stem}.txt").exists():
                print(f"skip  {extra_csv} (no full table {base_stem.name}.csv/.txt)", file=sys.stderr)
                continue
            try:
                added = merge_pair(base_stem, extra_stem)
            except MergeError as err:
                print(f"error {extra_stem.name}: {err}", file=sys.stderr)
                failed += 1
                continue
            merged += 1
            print(f"merged {extra_stem.name} -> {base_stem.name}  ({added})")
            if args.trash:
                for ext in (".csv", ".txt"):
                    move_to_trash(Path(f"{extra_stem}{ext}"))
    print(f"{merged} table pair(s) merged" + (f", {failed} refused" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
