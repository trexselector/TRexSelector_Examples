# ==============================================================================
# cell_cache.py
# ==============================================================================
#
# Resumable grid cells for the Monte Carlo demos. Every finished cell (one
# method at one grid point) is appended to
#   <demo>/simulation_results/cache/cells.tsv
# and a restarted demo reads it back instead of recomputing the cell. Same file
# format as the C++ and R demos:
#
#   # <fingerprint>
#   <key> \t name=value \t ... \t .
#
# The fingerprint hashes the demo script and the shared modules of its suite
# and family (plotting modules excluded), and holds the run tag TREX_RUN_TAG
# (the library pin, set by the runners) and the other TREX_* variables that can
# change results (not the worker count, TREX_FRESH, TREX_SOLVERS or a *_PARTS
# selection). A cache with another fingerprint, or any cache under
# TREX_FRESH=1, is set aside as cells.tsv.stale. A line without the closing "." marker (cut
# off by a hard stop) is recomputed. The cache is removed when the demo exits
# normally, i.e. once all results are written; an uncaught exception or a kill
# leaves it for the restart. Keys combine a readable label with digest() of the
# cell's inputs, so two cells with different inputs never share a result.
#
# The cache lives in the main process only; workers never touch it.
# ==============================================================================

import atexit
import glob
import hashlib
import os
import sys

_STATE: dict = {"opened": False, "enabled": False, "path": None, "cells": {}, "used": {}}


def _demo_dir():
    """Folder of the running demo script (None in an interactive session)."""
    main = sys.modules.get("__main__")
    path = getattr(main, "__file__", None)
    return os.path.dirname(os.path.abspath(path)) if path else None


def _counts(name):
    """True for a TREX_* variable that can change results (not the worker
    count, the fresh switch, a solver or a part selection)."""
    return (name.startswith("TREX_") and name not in ("TREX_FRESH", "TREX_NUM_WORKERS", "TREX_SOLVERS")
            and not name.endswith("_PARTS"))


def _fingerprint(demo_dir):
    """Source hash of demo + suite + family modules (plotting excluded) and the
    TREX_* settings that can change results."""
    h = hashlib.sha256()
    suite_dir = os.path.dirname(demo_dir)
    family_dir = os.path.dirname(os.path.dirname(suite_dir))
    files = (sorted(glob.glob(os.path.join(demo_dir, "*.py")))
             + sorted(glob.glob(os.path.join(suite_dir, "*.py")))
             + sorted(glob.glob(os.path.join(family_dir, "*.py"))))
    for f in files:
        if "plt_utils" in os.path.basename(f):
            continue
        with open(f, "rb") as fh:
            h.update(fh.read())
    env = sorted(f"{k}={v}" for k, v in os.environ.items() if _counts(k))
    return " ".join([f"src={h.hexdigest()[:16]}"] + env)


def _parse_line(line):
    """'key \\t n=v ... \\t .' -> (key, dict), or None for a cut-off line."""
    parts = line.rstrip("\n").split("\t")
    if len(parts) < 3 or parts[-1] != ".":
        return None
    cell = {}
    for kv in parts[1:-1]:
        name, sep, value = kv.partition("=")
        if not sep:
            return None
        cell[name] = float(value)
    return parts[0], cell


def _open():
    """Open the cache of the running demo on first use."""
    if _STATE["opened"]:
        return
    _STATE["opened"] = True
    demo_dir = _demo_dir()
    if demo_dir is None:
        return
    cache_dir = os.path.join(demo_dir, "simulation_results", "cache")
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, "cells.tsv")
    header = "# " + _fingerprint(demo_dir)
    fresh = os.environ.get("TREX_FRESH") == "1"

    if os.path.exists(path):
        with open(path) as fh:
            lines = fh.readlines()
        if not fresh and lines and lines[0].rstrip("\n") == header:
            for line in lines[1:]:
                parsed = _parse_line(line)
                if parsed is not None:
                    _STATE["cells"][parsed[0]] = parsed[1]
            print(f"[cache] {len(_STATE['cells'])} finished cell(s) restored "
                  f"from {path}\n", flush=True)
        else:
            os.replace(path, path + ".stale")
            why = "TREX_FRESH=1" if fresh else "sources or settings changed"
            print(f"[cache] {why}, old cache kept as cells.tsv.stale\n", flush=True)
    if not os.path.exists(path):
        with open(path, "w") as fh:
            fh.write(header + "\n")
    _STATE["path"] = path
    _STATE["enabled"] = True

    # A demo that exits normally has written all its results, so the cache goes.
    # An uncaught exception (incl. Ctrl-C) or a kill leaves it for the restart.
    previous_hook = sys.excepthook

    def _hook(exc_type, exc, tb):
        _STATE["failed"] = True
        previous_hook(exc_type, exc, tb)

    sys.excepthook = _hook
    atexit.register(lambda: None if _STATE.get("failed") else finish())


def _feed(h, obj):
    """Feed a stable byte form of obj into the hash h."""
    if isinstance(obj, dict):
        h.update(b"{")
        for k in sorted(obj, key=repr):
            _feed(h, k)
            _feed(h, obj[k])
        h.update(b"}")
    elif isinstance(obj, (list, tuple)):
        h.update(b"[")
        for v in obj:
            _feed(h, v)
        h.update(b"]")
    elif hasattr(obj, "tobytes") and hasattr(obj, "shape"):      # numpy array
        h.update(repr((obj.shape, str(obj.dtype))).encode())
        h.update(obj.tobytes())
    elif callable(obj):
        h.update(f"{getattr(obj, '__module__', '')}.{getattr(obj, '__qualname__', repr(obj))}".encode())
    else:
        h.update(repr(obj).encode())
    h.update(b";")


def digest(*inputs):
    """Short hash of everything a cell depends on (configs, seeds, grids)."""
    h = hashlib.sha256()
    _feed(h, inputs)
    return h.hexdigest()[:12]


def cached(key, compute):
    """Result of one cell: from the cache when present, else computed and appended.

    key     -- cell label without tabs or newlines; build it from a readable
               label and digest() of the cell's inputs, so cells with different
               inputs never share a key. Repeats of the same key in one run are
               told apart by an occurrence index, as an uncached run would
               compute each of them anew.
    compute -- function without arguments returning a dict name -> number.
    Returns a dict name -> float.
    """
    _open()
    n = _STATE["used"].get(key, 0)
    _STATE["used"][key] = n + 1
    full_key = f"{key}#{n}"
    if not _STATE["enabled"]:
        return {k: float(v) for k, v in compute().items()}
    hit = _STATE["cells"].get(full_key)
    if hit is not None:
        print(f"  [cache] {full_key}", flush=True)
        return dict(hit)
    cell = {k: float(v) for k, v in compute().items()}
    line = "\t".join([full_key] + [f"{k}={v!r}" for k, v in cell.items()] + ["."])
    with open(_STATE["path"], "a") as fh:
        fh.write(line + "\n")
    _STATE["cells"][full_key] = cell
    return dict(cell)


def finish():
    """Remove the cache once the demo has written all its results."""
    if _STATE["enabled"] and _STATE["path"] and os.path.exists(_STATE["path"]):
        os.remove(_STATE["path"])
