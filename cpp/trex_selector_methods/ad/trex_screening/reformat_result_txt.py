#!/usr/bin/env python3
"""Rewrite saved Screen-TRex .txt result tables in the current console format.

The demos print (and save) their MC summary tables via
``save_and_print_mc_results()`` / ``save_and_print_biobank_mc_results()`` in
``trex_screening_simulation_utils.hpp``. When that layout changes, the already
saved ``.txt`` files under each demo's ``simulation_results/data/`` go stale.
Rather than re-running hours of Monte Carlo, this script regenerates them from
the sibling ``.csv`` (which carries the same numbers in tidy long format).

It mirrors the C++ renderer: method name on its own line, metric rows indented
beneath it, and the sweep axis as a single header row.

Caveat: the CSV stores 6 decimals while the table shows 4, so a value sitting
exactly on a rounding tie can differ by one unit in the last displayed digit
from a fresh C++ run. Verified against authoritative C++ output on demo 01:
one such digit out of 48. Re-run the demo if bit-exact tables matter.

Usage:
    ./reformat_result_txt.py                # rewrite every *.txt next to a *.csv
    ./reformat_result_txt.py <file.csv> ... # rewrite only these
"""
from __future__ import annotations

import csv
import re
import sys
from collections import OrderedDict
from pathlib import Path

INDENT_W = 2
METRIC_W = 23
COL_W = 10

# Metric row order per table kind (matches the C++ emission order).
SWEEP_METRICS = ["FDR", "TPR", "Est. FDR"]
BIOBANK_METRICS = ["Usage (%)", "FDR", "TPR", "Est. FDR"]


def read_csv(path: Path):
    rows = list(csv.DictReader(open(path)))
    if not rows:
        raise ValueError(f"empty CSV: {path}")
    sweep = [c for c in rows[0] if c not in ("method", "metric", "value")][0]
    xs, methods = [], OrderedDict()
    values = {}
    for r in rows:
        x = float(r[sweep])
        if x not in xs:
            xs.append(x)
        methods.setdefault(r["method"], None)
        values[(r["method"], r["metric"], x)] = float(r["value"])
    return sweep, sorted(xs), list(methods), values


def num_mc(txt_path: Path) -> str:
    """Recover the MC count from the existing TXT banner (it is not in the CSV)."""
    if txt_path.exists():
        m = re.search(r"averaged over (\d+) Monte Carlo runs", txt_path.read_text())
        if m:
            return m.group(1)
    raise ValueError(f"cannot recover num_MC from {txt_path}")


def render(sweep, xs, methods, values, mc, biobank: bool) -> str:
    title = ("Biobank Screen-TRex Results" if biobank else "Screen-TRex Results")
    out = ["", "=" * 70,
           f"=== {title} (averaged over {mc} Monte Carlo runs) ===",
           "=" * 70, ""]

    label = "SNR" if biobank else sweep
    hdr = " " * INDENT_W + label.ljust(METRIC_W)
    hdr += "".join(f"{x:>{COL_W}.2f}" for x in xs)
    out.append(hdr)
    out.append("-" * (INDENT_W + METRIC_W + COL_W * len(xs)))

    metrics = BIOBANK_METRICS if biobank else SWEEP_METRICS
    for m in methods:
        out.append("")
        out.append(m)
        for metric in metrics:
            key = "Usage" if metric == "Usage (%)" else metric
            dp = 1 if metric == "Usage (%)" else 4
            scale = 100.0 if metric == "Usage (%)" else 1.0
            row = " " * INDENT_W + metric.ljust(METRIC_W)
            row += "".join(
                f"{values[(m, key, x)] * scale:>{COL_W}.{dp}f}" for x in xs)
            out.append(row)
    out.append("")
    out.append("=" * 0)  # placeholder removed below
    return "\n".join(out[:-1]) + "\n"


def main(argv: list[str]) -> int:
    here = Path(__file__).resolve().parent
    csvs = ([Path(a) for a in argv[1:]] if len(argv) > 1
            else sorted(here.glob("demo_trex_scr_0*/simulation_results/data/*.csv")))
    if not csvs:
        print("no CSVs found", file=sys.stderr)
        return 1

    for c in csvs:
        sweep, xs, methods, values = read_csv(c)
        biobank = any(k[1] == "Usage" for k in values)
        txt = c.with_suffix(".txt")
        table = render(sweep, xs, methods, values, num_mc(txt), biobank)
        txt.write_text(table)
        print(f"Rewrote: {txt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
