#!/usr/bin/env python3
"""
merge_result_files.py — put the tables of a solver-selection run into the full tables.

A trex_core demo run with TREX_SOLVERS=<names> computes only those solvers and
writes its tables next to the full ones with a suffix, e.g.

    demo_trex_02_..._stagnation_window_7_tstagewise.{txt,csv}   (TREX_SOLVERS=TSTAGEWISE)
    demo_trex_02_..._stagnation_window_7.{txt,csv}              (full solver pool)

This script replaces the blocks of the selected solvers in the full .txt and
.csv with the new ones, in place, so the solver order stays. A selected solver
that the full table lacks is appended at its end. The merge works on the text,
so every number keeps the format of its writer. It reads both table layouts:
the C++ one (solver name on its own line, metric rows indented beneath) and the
Python / R one (solver name in front of the first metric row).

A table header that differs between the two files (other SNR grid, other MC
count) or a solver block with another number of metric rows stops the merge.

Usage (from a demo folder or with explicit data folders, any language):
    python merge_result_files.py --suffix _tstagewise simulation_results/data
    python merge_result_files.py --suffix _tstagewise --trash */simulation_results/data
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path


class MergeError(RuntimeError):
    """A table pair that cannot be merged safely."""


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
            raise MergeError(f"CSV rows of {name!r} are not contiguous")
        blocks[name].append(line)
    return header, blocks, order


def merge_csv(base_text: str, new_text: str) -> str:
    base_h, base_b, base_o = split_csv(base_text.splitlines(keepends=True))
    new_h, new_b, new_o = split_csv(new_text.splitlines(keepends=True))
    if base_h != new_h:
        raise MergeError(f"CSV headers differ: {base_h.strip()!r} vs {new_h.strip()!r}")
    for name in new_o:
        if name in base_b and len(base_b[name]) != len(new_b[name]):
            raise MergeError(f"CSV block of {name!r} has {len(new_b[name])} rows, the full table "
                             f"{len(base_b[name])} (other grid?)")
    order = base_o + [name for name in new_o if name not in base_b]
    blocks = {**base_b, **new_b}
    return base_h + "".join(line for name in order for line in blocks[name])


# ==============================================================================
# TXT
# ==============================================================================

def is_blank(line: str) -> bool:
    return line.strip() == ""


def split_txt(lines: list[str]) -> tuple[list[str], list[str], list[list[str]], list[list[str]]]:
    """Header, leading blank lines, and per solver block its rows and trailing blank lines.

    The header ends with the dashed rule under the column labels. A block is a
    run of non-blank lines; its solver is the first word of its first line.
    """
    rule = next((i for i, line in enumerate(lines) if line.strip() and set(line.strip()) == {"-"}), None)
    if rule is None:
        raise MergeError("no dashed rule under the column labels (not a trex_core table?)")
    header, body = lines[:rule + 1], lines[rule + 1:]
    i = 0
    while i < len(body) and is_blank(body[i]):
        i += 1
    leading = body[:i]
    rows: list[list[str]] = []
    trailers: list[list[str]] = []
    while i < len(body):
        start = i
        while i < len(body) and not is_blank(body[i]):
            i += 1
        rows.append(body[start:i])
        start = i
        while i < len(body) and is_blank(body[i]):
            i += 1
        trailers.append(body[start:i])
    return header, leading, rows, trailers


def block_name(rows: list[str]) -> str:
    return rows[0].split()[0]


def merge_txt(base_text: str, new_text: str) -> str:
    base_h, base_lead, base_rows, base_trail = split_txt(base_text.splitlines(keepends=True))
    new_h, _, new_rows, _ = split_txt(new_text.splitlines(keepends=True))
    if base_h != new_h:
        raise MergeError("TXT headers differ (other SNR grid or MC count?)")
    index = {block_name(rows): k for k, rows in enumerate(base_rows)}
    if len(index) != len(base_rows):
        raise MergeError("a solver appears twice in the full TXT table")
    for rows in new_rows:
        name = block_name(rows)
        if name in index:
            k = index[name]
            if len(base_rows[k]) != len(rows):
                raise MergeError(f"TXT block of {name!r} has {len(rows)} rows, the full table "
                                 f"{len(base_rows[k])} (other metrics?)")
            base_rows[k] = rows
        else:
            # A new last block takes the footer, the old last block the separator
            footer = base_trail[-1]
            base_trail[-1] = base_trail[0] if len(base_trail) > 1 else footer
            base_trail.append(footer)
            base_rows.append(rows)
            index[name] = len(base_rows) - 1
    body = "".join("".join(rows) + "".join(trail) for rows, trail in zip(base_rows, base_trail))
    return "".join(base_h) + "".join(base_lead) + body


# ==============================================================================
# Files
# ==============================================================================

def merge_pair(base_stem: Path, extra_stem: Path) -> str:
    """Merge extra_stem.{csv,txt} into base_stem.{csv,txt}; returns the merged solvers."""
    base_csv, new_csv = (Path(f"{s}.csv").read_text() for s in (base_stem, extra_stem))
    base_txt, new_txt = (Path(f"{s}.txt").read_text() for s in (base_stem, extra_stem))
    new_solvers = split_csv(new_csv.splitlines(keepends=True))[2]
    txt_solvers = [block_name(rows) for rows in split_txt(new_txt.splitlines(keepends=True))[2]]
    if txt_solvers != new_solvers:
        raise MergeError(f"the selection's CSV and TXT list other solvers: {new_solvers} vs {txt_solvers}")
    csv_out = merge_csv(base_csv, new_csv)
    txt_out = merge_txt(base_txt, new_txt)
    Path(f"{base_stem}.csv").write_text(csv_out)
    Path(f"{base_stem}.txt").write_text(txt_out)
    return ", ".join(new_solvers)


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
    parser.add_argument("--suffix", required=True, help="file suffix of the selection run, e.g. _tstagewise")
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
