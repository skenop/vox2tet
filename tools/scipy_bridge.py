#!/usr/bin/env python3
"""Debug-only scipy/numpy reference bridge for the C++ port.

The C++ pipeline writes intermediate arrays as .npy. This script reads
those .npy files alongside the corresponding Python pipeline outputs,
re-runs the same scipy operation that the C++ code re-implemented (EDT,
brep sizing kd-tree query, ...), and reports max / mean numeric diff.

Used for QA — proves where the C++ port diverges from scipy's
"reference" answer for the same input.

Sub-commands:
    edt        — re-run scipy.ndimage.distance_transform_edt and compare
                 indices/distance against the C++ EDT implementation.
                 Inputs: <mask.npy> <my_iz.npy> <my_iy.npy> <my_ix.npy>
    brep_size  — re-run scipy.spatial.cKDTree-based sizing field and
                 compare against a C++-emitted _sz_fld.npy.
                 Inputs: <xyz.npy> <is_brep.npy> <my_sz_fld.npy>
    canonical  — canonicalised diff of two TetGen .smesh files (same as
                 the C++ diff_smesh tool, but uses scipy.spatial.cKDTree
                 for the node-set match which makes tolerant comparison
                 trivially fast on large meshes).
                 Inputs: <a.smesh> <b.smesh> [--coord-tol 1e-9]

Example:
    /tmp/v2t_debug/bin/python tools/scipy_bridge.py edt mask.npy \\
        my_iz.npy my_iy.npy my_ix.npy
"""

from __future__ import annotations

import argparse
import os
import sys

# Soft import — the script may live in a tree without numpy/scipy
# installed in the system Python. The CMake CACHE entry
# VOX2TET_PYTHON_BIN points the C++ caller at a venv that has them.
try:
    import numpy as np
except ImportError:
    print("ERROR: numpy not available. Install in a venv and point "
          "VOX2TET_PYTHON_BIN there.", file=sys.stderr)
    sys.exit(2)
try:
    import scipy.ndimage as ndi
    import scipy.spatial as spatial
except ImportError:
    print("ERROR: scipy not available. Install in a venv and point "
          "VOX2TET_PYTHON_BIN there.", file=sys.stderr)
    sys.exit(2)


def cmd_edt(args: argparse.Namespace) -> int:
    """Run scipy's EDT and compare with the C++ Felzenszwalb output."""
    mask = np.load(args.mask)
    if mask.dtype != bool:
        mask = mask.astype(bool)
    print(f"mask shape={mask.shape}  source voxels (False)={int((~mask).sum())}")
    # scipy returns (dist, indices) when return_distances=True;
    # the C++ port only computes indices, so we ask for indices alone.
    sci_idx = ndi.distance_transform_edt(mask, return_distances=False,
                                         return_indices=True)
    # Compare with our outputs.
    my_iz = np.load(args.iz)
    my_iy = np.load(args.iy)
    my_ix = np.load(args.ix)
    if sci_idx.shape != (3,) + mask.shape:
        print(f"unexpected scipy shape {sci_idx.shape}")
        return 1
    diff = ((sci_idx[0] != my_iz.reshape(mask.shape)) |
            (sci_idx[1] != my_iy.reshape(mask.shape)) |
            (sci_idx[2] != my_ix.reshape(mask.shape)))
    n_diff = int(diff.sum())
    print(f"EDT index disagreement: {n_diff} / {mask.size} voxels ({100*n_diff/mask.size:.3f}%)")
    if n_diff > 0:
        # Show how many of those are at iso-distance (tie cases).
        my_iz = my_iz.reshape(mask.shape)
        my_iy = my_iy.reshape(mask.shape)
        my_ix = my_ix.reshape(mask.shape)
        # squared distance from each voxel to scipy's source
        zz, yy, xx = np.indices(mask.shape)
        d_sci = ((sci_idx[0] - zz)**2 + (sci_idx[1] - yy)**2 +
                 (sci_idx[2] - xx)**2)
        d_my  = ((my_iz - zz)**2 + (my_iy - yy)**2 + (my_ix - xx)**2)
        n_iso = int(((d_sci == d_my) & diff).sum())
        print(f"  of those, {n_iso} are at iso-distance (tie-break choice differs only).")
        print(f"  remaining {n_diff - n_iso} voxels have ACTUAL distance mismatch.")
    return 0 if n_diff == 0 else 1


def cmd_brep_size(args: argparse.Namespace) -> int:
    xyz = np.load(args.xyz)
    is_brep = np.load(args.is_brep).astype(bool)
    print(f"xyz shape={xyz.shape}  brep verts={int(is_brep.sum())}")
    tree = spatial.cKDTree(xyz[is_brep])
    out = np.zeros(xyz.shape[0], dtype=np.float64)
    out[is_brep]  = np.squeeze(tree.query(xyz[is_brep], k=[2])[0])
    out[~is_brep] = tree.query(xyz[~is_brep], k=1)[0]
    my = np.load(args.my)
    if my.shape != out.shape:
        print(f"shape mismatch: mine={my.shape}  scipy={out.shape}")
        return 1
    d = np.abs(my - out)
    print(f"brep sizing max diff = {float(d.max()):.6g}   "
          f"mean diff = {float(d.mean()):.6g}")
    return 0 if d.max() < 1e-6 else 1


def cmd_canonical(args: argparse.Namespace) -> int:
    """Read two .smesh files, canonicalise, and report node/triangle
    set diff using scipy.cKDTree for tolerant coord matching."""
    def read_smesh(path):
        nodes, tris, markers = [], [], []
        with open(path) as f:
            lines = [l for l in f if l.strip() and not l.lstrip().startswith("#")]
        n_nodes = int(lines[0].split()[0])
        first = None
        for i in range(1, n_nodes + 1):
            p = lines[i].split()
            if first is None: first = int(p[0])
            nodes.append((float(p[1]), float(p[2]), float(p[3])))
        idx = 1 + n_nodes
        n_fac, has_marker = int(lines[idx].split()[0]), int(lines[idx].split()[1])
        for i in range(idx + 1, idx + 1 + n_fac):
            p = lines[i].split()
            tris.append((int(p[1]) - first, int(p[2]) - first, int(p[3]) - first))
            markers.append(int(p[4]) if has_marker else 0)
        return np.array(nodes), np.array(tris), np.array(markers)

    A_nodes, A_tris, A_mark = read_smesh(args.a)
    B_nodes, B_tris, B_mark = read_smesh(args.b)
    print(f"A: {len(A_nodes)} nodes / {len(A_tris)} tris")
    print(f"B: {len(B_nodes)} nodes / {len(B_tris)} tris")

    # Match nodes via cKDTree with the given tolerance.
    tree = spatial.cKDTree(B_nodes)
    dist, A_to_B = tree.query(A_nodes, k=1)
    matched = dist < args.coord_tol
    print(f"nodes within tol {args.coord_tol}: {int(matched.sum())} / {len(A_nodes)}")
    # Map A's tris into B's index space; unresolved → -1.
    A_in_B = np.where(matched, A_to_B, -1)
    A_tris_B = A_in_B[A_tris]
    bad_rows = (A_tris_B < 0).any(axis=1)
    print(f"A triangles fully resolvable in B: {int((~bad_rows).sum())} / {len(A_tris)}")

    # Canonicalise: rotate each triangle to (min, mid, max) + sort by marker.
    canon_A = np.sort(A_tris_B[~bad_rows], axis=1)
    canon_A_keys = list(zip(A_mark[~bad_rows].tolist(),
                            canon_A[:, 0].tolist(),
                            canon_A[:, 1].tolist(),
                            canon_A[:, 2].tolist()))
    canon_B = np.sort(B_tris, axis=1)
    canon_B_keys = list(zip(B_mark.tolist(),
                            canon_B[:, 0].tolist(),
                            canon_B[:, 1].tolist(),
                            canon_B[:, 2].tolist()))
    sA, sB = set(canon_A_keys), set(canon_B_keys)
    print(f"canonical tri set: A∩B={len(sA & sB)}  only_A={len(sA - sB)}  only_B={len(sB - sA)}")
    return 0 if (sA == sB and not bad_rows.any()) else 1


def cmd_surface(args: argparse.Namespace) -> int:
    """Canonical comparison of two (xyz, tr, att) surface triples."""
    A_xyz = np.load(args.a_xyz).astype(np.float64)
    A_tr  = np.load(args.a_tr).astype(np.int64)
    A_att = np.load(args.a_att).astype(np.int64)
    B_xyz = np.load(args.b_xyz).astype(np.float64)
    B_tr  = np.load(args.b_tr).astype(np.int64)
    B_att = np.load(args.b_att).astype(np.int64)
    print(f"A: {len(A_xyz)} verts / {len(A_tr)} tris / {len(A_att)} interfaces")
    print(f"B: {len(B_xyz)} verts / {len(B_tr)} tris / {len(B_att)} interfaces")

    # Per-triangle marker: each A_att row is (code, first, count, m1, m2).
    def per_tri_marker(att, n_tri):
        out = np.zeros(n_tri, dtype=np.int64)
        for code, first, count, _, _ in att:
            out[first:first + count] = code
        return out
    A_marker = per_tri_marker(A_att, len(A_tr))
    B_marker = per_tri_marker(B_att, len(B_tr))

    # Match nodes via cKDTree with tolerance.
    tree = spatial.cKDTree(B_xyz)
    dist, A_to_B = tree.query(A_xyz, k=1)
    matched = dist < args.coord_tol
    print(f"nodes within tol {args.coord_tol}: {int(matched.sum())} / {len(A_xyz)}")
    A_in_B = np.where(matched, A_to_B, -1)
    A_tr_B = A_in_B[A_tr]
    bad = (A_tr_B < 0).any(axis=1)

    canon_A = np.sort(A_tr_B[~bad], axis=1)
    sA = set(zip(A_marker[~bad].tolist(),
                 canon_A[:, 0].tolist(),
                 canon_A[:, 1].tolist(),
                 canon_A[:, 2].tolist()))
    canon_B = np.sort(B_tr, axis=1)
    sB = set(zip(B_marker.tolist(),
                 canon_B[:, 0].tolist(),
                 canon_B[:, 1].tolist(),
                 canon_B[:, 2].tolist()))
    print(f"canonical tri set: A∩B={len(sA & sB)}  only_A={len(sA - sB)}  only_B={len(sB - sA)}")
    return 0 if (sA == sB and not bad.any()) else 1


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    p_edt = sub.add_parser("edt", help="compare EDT indices vs scipy")
    p_edt.add_argument("mask")
    p_edt.add_argument("iz")
    p_edt.add_argument("iy")
    p_edt.add_argument("ix")
    p_edt.set_defaults(fn=cmd_edt)

    p_bs = sub.add_parser("brep_size", help="compare brep sizing vs scipy")
    p_bs.add_argument("xyz")
    p_bs.add_argument("is_brep")
    p_bs.add_argument("my")
    p_bs.set_defaults(fn=cmd_brep_size)

    p_c = sub.add_parser("canonical", help="canonical diff of two .smesh files")
    p_c.add_argument("a")
    p_c.add_argument("b")
    p_c.add_argument("--coord-tol", type=float, default=1e-6)
    p_c.set_defaults(fn=cmd_canonical)

    p_s = sub.add_parser("surface", help="canonical diff of .npy surface pairs (xyz + tr + att)")
    p_s.add_argument("a_xyz")
    p_s.add_argument("a_tr")
    p_s.add_argument("a_att")
    p_s.add_argument("b_xyz")
    p_s.add_argument("b_tr")
    p_s.add_argument("b_att")
    p_s.add_argument("--coord-tol", type=float, default=1e-9)
    p_s.set_defaults(fn=cmd_surface)

    args = p.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
