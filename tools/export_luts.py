#!/usr/bin/env python3
"""Export marching-cubes look-up tables to little-endian raw binary files
that the C++ side reads with `marching_cubes::load_luts()`.

Pure-stdlib: no numpy required (the Python sources use np.array() for
storage but the data inside is plain int literals, so we extract the
nested lists with `ast.literal_eval`).

Run once from the repo root:

    cd cpp/tools && python3 export_luts.py

Outputs (written to ../data/ relative to this script):

    lut8.bin         uint16[16777216]      (mirror of lut8.npy)
    lut0.bin         uint8[N0 x 24]        (lut0CetredCell)
    lut2.bin         uint8[N2 x 36]        (lut2CetredCell)
    lut2x2.bin       uint8[N2x2 x 42]      (lut2x2CetredCell)
    facet_lut2d.bin  uint8[256 x 8]        (60-config facet table indexed
                                            by the 4-color 4^4 cell code)
"""

from __future__ import annotations

import ast
import os
import re
import struct
import sys

HERE  = os.path.dirname(os.path.abspath(__file__))
PYDIR = os.path.normpath(os.path.join(HERE, "..", "..", "python"))
OUT   = os.path.normpath(os.path.join(HERE, "..", "data"))
os.makedirs(OUT, exist_ok=True)


def extract_array_literal(source: str, fn_name: str) -> list:
    """Extract the list-of-lists literal from the body of ``fn_name``.

    The Python files have the form::

        def fn_name(...):
            ...
            return np.array(
                [[...], [...], ...],
                dtype=np.uint8)

    We locate that function with ``ast``, find its return statement, and
    take the first positional argument of the ``np.array(...)`` call,
    which is the list literal. ``ast.literal_eval`` then gives us a
    pure-Python nested list — no numpy needed.
    """
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == fn_name:
            for stmt in node.body:
                if isinstance(stmt, ast.Return) and isinstance(stmt.value, ast.Call):
                    call = stmt.value
                    if call.args:
                        return ast.literal_eval(call.args[0])
    raise SystemExit(f"could not find list literal in function {fn_name!r}")


def read_python_module(rel_path: str) -> str:
    with open(os.path.join(PYDIR, rel_path), "r") as f:
        return f.read()


# ---------------------------------------------------------------------------
# lut8.npy  →  lut8.bin
# ---------------------------------------------------------------------------
#
# .npy v1.0 format:
#   bytes  0..5   magic     b'\x93NUMPY'
#   bytes  6..7   version   (major, minor)
#   bytes  8..9   header length (little-endian uint16)
#   then          header text of that length
#   then          raw data (little-endian)
#
def export_lut8():
    src = os.path.join(PYDIR, "lut8.npy")
    dst = os.path.join(OUT, "lut8.bin")
    with open(src, "rb") as f:
        magic = f.read(6)
        if magic != b"\x93NUMPY":
            raise SystemExit("lut8.npy: bad magic")
        ver = f.read(2)
        if ver[0] == 1:
            hlen = struct.unpack("<H", f.read(2))[0]
        else:
            hlen = struct.unpack("<I", f.read(4))[0]
        header = f.read(hlen).decode("ascii")
        # Sanity-check the dtype to confirm we're handing C++ what it expects.
        # Python writes "<u2" or "|u2".
        if not re.search(r"'descr'\s*:\s*'[<|]u2'", header):
            raise SystemExit(f"lut8.npy: unexpected dtype in header: {header!r}")
        data = f.read()
    expected = 16_777_216 * 2
    if len(data) != expected:
        raise SystemExit(f"lut8.npy: expected {expected} bytes, got {len(data)}")
    with open(dst, "wb") as f:
        f.write(data)
    print(f"lut8.bin: {len(data)/1024/1024:.1f} MB")


# ---------------------------------------------------------------------------
# cellLUTarray.py  →  lut0.bin / lut2.bin / lut2x2.bin
# ---------------------------------------------------------------------------
def export_cell_luts():
    src = read_python_module("cellLUTarray.py")
    spec = [
        ("lut0CetredCell",   "lut0.bin",   24),
        ("lut2CetredCell",   "lut2.bin",   36),
        ("lut2x2CetredCell", "lut2x2.bin", 42),
    ]
    for fn, fname, ncols in spec:
        arr = extract_array_literal(src, fn)
        if not arr:
            raise SystemExit(f"{fn}: empty array")
        nrows = len(arr)
        for row in arr:
            if len(row) != ncols:
                raise SystemExit(f"{fn}: row width {len(row)} != expected {ncols}")
            for v in row:
                if not (0 <= int(v) <= 19):
                    raise SystemExit(f"{fn}: value {v} out of expected [0,19]")
        flat = bytes(int(v) for row in arr for v in row)
        with open(os.path.join(OUT, fname), "wb") as f:
            f.write(flat)
        print(f"{fname}: {nrows} rows x {ncols}")


# ---------------------------------------------------------------------------
# facetLUTarray.py  →  facet_lut2d.bin
#
# C++ side expects the *resolved* (256 x 8) table indexed directly by the
# 4-color 4^4 cell code. Python does this on the fly as
# ``getLUT2DMMaterials60()[map256to60()]``, so we replicate the same.
# ---------------------------------------------------------------------------
def export_facet_lut2d():
    src = read_python_module("facetLUTarray.py")
    facet60 = extract_array_literal(src, "getLUT2DMMaterials60")
    mp      = extract_array_literal(src, "map256to60")
    if len(facet60) != 60 or any(len(row) != 8 for row in facet60):
        raise SystemExit(f"getLUT2DMMaterials60: expected (60,8), got {len(facet60)}x?")
    if len(mp) != 256:
        raise SystemExit(f"map256to60: expected length 256, got {len(mp)}")
    out = bytearray(256 * 8)
    for i, idx in enumerate(mp):
        row = facet60[int(idx)]
        for j, v in enumerate(row):
            out[i * 8 + j] = int(v) & 0xFF
    with open(os.path.join(OUT, "facet_lut2d.bin"), "wb") as f:
        f.write(out)
    print(f"facet_lut2d.bin: 256 x 8 (resolved through map256to60)")


def main():
    export_lut8()
    export_cell_luts()
    export_facet_lut2d()


if __name__ == "__main__":
    main()
