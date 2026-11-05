#!/usr/bin/env python3
"""
merge_result_files.py — complete the tables of a method-selection run with the full tables.

A gvs demo run with TREX_SOLVERS=<names> computes only those methods and writes
its tables next to the full ones with a suffix, e.g.

    gvs_Hastie_2d_tienet_aug.{txt,csv}   (TREX_SOLVERS=TIENET_AUG, other rows NaN)
    gvs_Hastie_2d.{txt,csv}              (full run)

The selection tables have exactly the layout a full run of the current code
writes, so they serve as the template. This script fills every row and block of
a method that was not selected from the full tables and writes the result over
the full tables. The selected methods keep the new numbers, including rows whose
labels changed (demo 01 Part 2 compares other lambda2 rules on the IEN track).

Methods count by their solver: TENET, TENET_AUG, TIENET_AUG, and in the block
bench (demo 08) M1/M2 by the EN solver and M3/M4 ("IE-") as TIENET_AUG.

NaN counts in any spelling (C++ and Python write "nan", R "NaN"). The merge
stops when a filled row has no NaN in the template (it was computed,
so the selection is wrong), when a row is missing in the full table, or when
the table titles differ (other grid, other MC count).

Usage (from a demo folder or with explicit data folders):
    python merge_result_files.py --methods TIENET_AUG simulation_results/data
    python merge_result_files.py --methods TIENET_AUG --trash demo_*/simulation_results/data
"""

from __future__ import annotations

import argparse
import csv
import io
import re
import shutil
import sys
import time
from pathlib import Path

VALUE_COLUMNS = ("mean", "sd", "value")
BLOCK_2D = re.compile(r"^  (?:mean_TPP|mean_FDP) \[(.+)\]\s*$")
BLOCK_1D = re.compile(r"^(?:  )?([A-Z][A-Z_]+)\s*$")  # indented, or flush left in the cut tables
BENCH_RHO = re.compile(r"^  rho = (\S+)\s*$")
BENCH_ROW = re.compile(r"^  (M\d\([A-Z]+-[A-Z]\))\s+(\S+)\s")


class MergeError(RuntimeError):
    """A table pair that cannot be merged safely."""


def solver_of(method: str) -> str:
    """Solver a method row counts as: bench IEN methods are TIENET_AUG, EN ones "EN"."""
    if re.match(r"^M\d\(", method):
        return "TIENET_AUG" if "(IE-" in method else "EN"
    return method.split("+", 1)[0]


def is_selected(method: str, selected: set[str]) -> bool:
    """True when a method row comes from the selection run."""
    solver = solver_of(method)
    if solver == "EN":
        return bool(selected & {"TENET", "TENET_AUG"})
    return solver in selected


# ==============================================================================
# CSV
# ==============================================================================

def csv_rows(text: str) -> tuple[list[str], list[list[str]]]:
    """Header and rows of a CSV text."""
    rows = list(csv.reader(io.StringIO(text)))
    return rows[0], rows[1:]


def merge_csv(template: str, full: str, selected: set[str]) -> str:
    """Template rows of selected methods, all other rows from the full table."""
    header, new_rows = csv_rows(template)
    full_header, full_rows = csv_rows(full)
    if header != full_header:
        raise MergeError(f"CSV headers differ: {header} vs {full_header}")
    m = header.index("method")
    values = [i for i, c in enumerate(header) if c in VALUE_COLUMNS] or list(range(m + 1, len(header)))
    keys = [i for i in range(len(header)) if i not in values]
    lookup = {tuple(r[i] for i in keys): r for r in full_rows}

    out = [header]
    for row in new_rows:
        if is_selected(row[m], selected):
            if any(row[i].lower() == "nan" for i in values):
                raise MergeError(f"selected row has NaN: {row}")
            out.append(row)
            continue
        if not any(row[i].lower() == "nan" for i in values):  # counts stay 0, metrics NaN
            raise MergeError(f"row was computed although not selected: {row}")
        key = tuple(row[i] for i in keys)
        if key not in lookup:
            raise MergeError(f"row missing in the full table: {key}")
        out.append(lookup[key])
    buf = io.StringIO()
    csv.writer(buf, lineterminator="\n").writerows(out)
    return buf.getvalue()


# ==============================================================================
# TXT
# ==============================================================================

def txt_blocks(lines: list[str]) -> list[tuple[str | None, list[str]]]:
    """Split a table text into (method or None, lines) pieces at block headers and rules."""
    pieces: list[tuple[str | None, list[str]]] = [(None, [])]
    for line in lines:
        head = BLOCK_2D.match(line) or BLOCK_1D.match(line)
        if head:
            pieces.append((head.group(1), [line]))
        elif line.startswith("=") and pieces[-1][0] is not None:
            pieces.append((None, [line]))
        else:
            pieces[-1][1].append(line)
    return pieces


def merge_bench_txt(new: list[str], full: list[str], selected: set[str]) -> list[str]:
    """Block bench tables: rows keyed by (rho, method, SNR)."""
    def keyed(lines: list[str]) -> dict[tuple[str, str, str], str]:
        rho, out = "", {}
        for line in lines:
            if (r := BENCH_RHO.match(line)):
                rho = r.group(1)
            elif (row := BENCH_ROW.match(line)):
                out[(rho, row.group(1), row.group(2))] = line
        return out

    lookup = keyed(full)
    out, rho = [], ""
    for line in new:
        if (r := BENCH_RHO.match(line)):
            rho = r.group(1)
        row = BENCH_ROW.match(line)
        if row and not is_selected(row.group(1), selected):
            if "nan" not in line.lower():
                raise MergeError(f"row was computed although not selected: {line.strip()}")
            key = (rho, row.group(1), row.group(2))
            if key not in lookup:
                raise MergeError(f"row missing in the full table: {key}")
            line = lookup[key]
        elif row and "nan" in line.lower():
            raise MergeError(f"selected row has NaN: {line.strip()}")
        out.append(line)
    return out


def merge_txt(template: str, full: str, selected: set[str]) -> str:
    """Template blocks of selected methods, all other blocks from the full table."""
    new, old = template.splitlines(keepends=True), full.splitlines(keepends=True)
    title_new = [ln for ln in new if ln.strip() and not ln.startswith("=")][:1]
    title_old = [ln for ln in old if ln.strip() and not ln.startswith("=")][:1]
    if title_new != title_old:
        raise MergeError(f"titles differ: {title_new} vs {title_old}")
    if any(BENCH_ROW.match(ln) for ln in new):
        return "".join(merge_bench_txt(new, old, selected))

    old_blocks = {(m, lines[0]): lines for m, lines in txt_blocks(old) if m is not None}
    out: list[str] = []
    for method, lines in txt_blocks(new):
        if method is None or is_selected(method, selected):
            out += lines
            continue
        if not any("nan" in ln.lower() for ln in lines[1:]):
            raise MergeError(f"block was computed although not selected: {lines[0].strip()}")
        if (method, lines[0]) not in old_blocks:
            raise MergeError(f"block missing in the full table: {lines[0].strip()}")
        out += old_blocks[(method, lines[0])]
    return "".join(out)


# ==============================================================================
# Files
# ==============================================================================

def merge_pair(base_stem: Path, extra_stem: Path, selected: set[str]) -> None:
    """Write the merged tables over base_stem.{csv,txt}; both are checked before writing."""
    csv_out = merge_csv(Path(f"{extra_stem}.csv").read_text(), Path(f"{base_stem}.csv").read_text(), selected)
    txt_out = merge_txt(Path(f"{extra_stem}.txt").read_text(), Path(f"{base_stem}.txt").read_text(), selected)
    Path(f"{base_stem}.csv").write_text(csv_out)
    Path(f"{base_stem}.txt").write_text(txt_out)


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
    parser.add_argument("--methods", required=True,
                        help="TREX_SOLVERS of the selection run (comma separated), e.g. TIENET_AUG")
    parser.add_argument("--trash", action="store_true", help="move the merged extra files to ~/.Trash")
    args = parser.parse_args()
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    selected = set(methods)
    args.suffix = "".join(f"_{m.lower()}" for m in methods)  # as C++ file_suffix()

    merged = failed = 0
    for data_dir in args.data_dirs:
        for extra_csv in sorted(data_dir.glob(f"*{args.suffix}.csv")):
            extra_stem = extra_csv.with_suffix("")
            base_stem = extra_stem.with_name(extra_stem.name[: -len(args.suffix)])
            if not Path(f"{base_stem}.csv").exists() or not Path(f"{base_stem}.txt").exists():
                print(f"skip   {extra_csv} (no full table {base_stem.name}.csv/.txt)", file=sys.stderr)
                continue
            try:
                merge_pair(base_stem, extra_stem, selected)
            except MergeError as err:
                failed += 1
                print(f"FAILED {extra_stem.name}: {err}", file=sys.stderr)
                continue
            merged += 1
            print(f"merged {extra_stem.name} -> {base_stem.name}  ({', '.join(sorted(selected))})")
            if args.trash:
                for ext in (".csv", ".txt"):
                    move_to_trash(Path(f"{extra_stem}{ext}"))
    print(f"{merged} table pair(s) merged, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
