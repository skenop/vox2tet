#!/usr/bin/env python3
"""Compute surface-mesh quality metrics for a vox2tet `.smesh` output,
split by edge category (boundary vs internal).

Edge classification (mirrors the C++ `not_fixed_h` convention):
  * Boundary edges — both endpoints lie on the bounding-box surface
    (at least one coordinate equals the global min or max within tol).
    Guard threshold: `min_dangle_boundary` (typically 20°).
  * Internal edges — all other edges (grain-boundary interior plus
    triple lines / quad points).
    Guard threshold: `min_dangle_internal` (typically 40°).

Metrics reported:
  1. Triangle corner angles (3 per triangle).
  2. Dihedral angles using the geometric cross-section construction
     — 180° = flat, 0° = fully folded; independent of triangle winding
     (required for non-manifold edges where per-triangle normals follow
     each interface's own mat1→mat2 convention rather than a global one).
  3. Distance from each remeshed node to its nearest neighbour in the
     initial multi-material marching-cubes node cloud.

Usage:
  quality_metrics.py --label NAME --smesh path/to/X_RE.smesh \\
                     --mc-xyz path/to/X_xyz.npy \\
                     --thr-boundary 20 --thr-internal 40
"""
from __future__ import annotations
import argparse
import collections
import numpy as np
from scipy.spatial import cKDTree


def read_smesh(path: str):
    nodes, tris, markers = [], [], []
    with open(path) as f:
        lines = [l for l in f if l.strip() and not l.lstrip().startswith("#")]
    n_nodes = int(lines[0].split()[0])
    first = None
    for i in range(1, n_nodes + 1):
        p = lines[i].split()
        if first is None:
            first = int(p[0])
        nodes.append((float(p[1]), float(p[2]), float(p[3])))
    idx = 1 + n_nodes
    head = lines[idx].split()
    n_fac, has_marker = int(head[0]), int(head[1])
    for i in range(idx + 1, idx + 1 + n_fac):
        p = lines[i].split()
        tris.append((int(p[1]) - first, int(p[2]) - first, int(p[3]) - first))
        markers.append(int(p[4]) if has_marker else 0)
    return (np.asarray(nodes, dtype=np.float64),
            np.asarray(tris, dtype=np.int64),
            np.asarray(markers, dtype=np.int64))


def classify_nodes_bbox(xyz: np.ndarray, tol: float = 1e-6) -> np.ndarray:
    on_bbox = np.zeros(len(xyz), dtype=bool)
    for ax in range(3):
        lo = xyz[:, ax].min()
        hi = xyz[:, ax].max()
        on_bbox |= (np.abs(xyz[:, ax] - lo) < tol)
        on_bbox |= (np.abs(xyz[:, ax] - hi) < tol)
    return on_bbox


def classify_edges(xyz, tri):
    on_bbox = classify_nodes_bbox(xyz)
    edges = {}
    M = tri.shape[0]
    for f in range(M):
        for a, b in [(0, 1), (1, 2), (2, 0)]:
            i, j = int(tri[f, a]), int(tri[f, b])
            key = (min(i, j), max(i, j))
            cls = 'boundary' if (on_bbox[i] and on_bbox[j]) else 'internal'
            # Once an edge is seen as 'internal' it stays internal.
            if key not in edges:
                edges[key] = cls
            elif edges[key] == 'boundary' and cls == 'internal':
                edges[key] = 'internal'
    return edges


def corner_angles_deg(xyz, tri):
    v0 = xyz[tri[:, 0]]; v1 = xyz[tri[:, 1]]; v2 = xyz[tri[:, 2]]
    def ang(a, b, c):
        u = b - a; v = c - a
        nu = np.linalg.norm(u, axis=1) + 1e-300
        nv = np.linalg.norm(v, axis=1) + 1e-300
        cos = np.einsum("ij,ij->i", u, v) / (nu * nv)
        return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))
    return np.stack([ang(v0, v1, v2), ang(v1, v2, v0), ang(v2, v0, v1)], axis=1)


def dihedral_angles_split(xyz, tri, edge_class):
    edge_faces = collections.defaultdict(list)
    M = tri.shape[0]
    for f in range(M):
        for a, b in [(0, 1), (1, 2), (2, 0)]:
            i, j = int(tri[f, a]), int(tri[f, b])
            key = (min(i, j), max(i, j))
            apex = int(tri[f, (3 - a - b) % 3])
            edge_faces[key].append(apex)

    boundary_out = []; internal_out = []
    for (va, vb), apexes in edge_faces.items():
        if len(apexes) < 2:
            continue
        A, B = xyz[va], xyz[vb]
        v0 = B - A
        cs = []
        for apex in apexes:
            c = np.cross(xyz[apex] - A, v0)
            n = np.linalg.norm(c)
            cs.append(c / n if n > 1e-18 else None)
        for i in range(len(apexes)):
            for j in range(i + 1, len(apexes)):
                if cs[i] is None or cs[j] is None:
                    continue
                d = float(np.clip(cs[i].dot(cs[j]), -1.0, 1.0))
                ang = np.degrees(np.arccos(d))
                cls = edge_class.get((va, vb), 'internal')
                (boundary_out if cls == 'boundary' else internal_out).append(ang)
    return (np.asarray(boundary_out, dtype=np.float64),
            np.asarray(internal_out, dtype=np.float64))


def summarise(name, arr, hi_is_good, thresholds=None):
    if arr.size == 0:
        print(f"    {name}: (empty)")
        return
    pcts = np.percentile(arr, [5, 50, 95])
    pct_lbl = 'p5' if hi_is_good else 'p95'
    pct_val = pcts[0] if hi_is_good else pcts[2]
    line = (f"    {name:36s}  n={arr.size:7d}  "
            f"min={arr.min():8.3f}  mean={arr.mean():8.3f}  "
            f"median={pcts[1]:8.3f}  {pct_lbl}={pct_val:8.3f}  "
            f"max={arr.max():8.3f}")
    if thresholds:
        for thr in thresholds:
            cnt = int((arr < thr).sum())
            line += f"  <{thr:.0f}°: {cnt}"
    print(line)


def report(label, smesh_path, mc_xyz_path, thr_b=20.0, thr_i=40.0):
    xyz, tri, _ = read_smesh(smesh_path)
    print(f"\n=== {label} ===")
    print(f"  smesh : {smesh_path}")
    print(f"  nodes={len(xyz)}  triangles={len(tri)}")
    print(f"  dihedral thresholds: boundary={thr_b}°  internal={thr_i}°")

    edge_class = classify_edges(xyz, tri)
    n_bnd = sum(1 for v in edge_class.values() if v == 'boundary')
    n_int = sum(1 for v in edge_class.values() if v == 'internal')
    print(f"  edges: {n_bnd} boundary  {n_int} internal\n")

    ca = corner_angles_deg(xyz, tri).ravel()
    print("  [1] Corner angles (all triangles):")
    summarise("corner angles (deg)", ca, hi_is_good=True, thresholds=[20.0, 30.0])

    bnd_di, int_di = dihedral_angles_split(xyz, tri, edge_class)
    print(f"\n  [2a] Dihedral angles — BOUNDARY edges  (threshold {thr_b}°):")
    summarise("dihedral (deg)", bnd_di, hi_is_good=True,
              thresholds=[thr_b, 30.0, 40.0])

    print(f"\n  [2b] Dihedral angles — INTERNAL edges  (threshold {thr_i}°):")
    summarise("dihedral (deg)", int_di, hi_is_good=True,
              thresholds=[thr_i, 30.0, 20.0])

    mc_xyz = np.load(mc_xyz_path).astype(np.float64)
    # x_rotation: y_new = z_old; z_new = ymax - y_old.
    ymax = mc_xyz[:, 1].max()
    rot = np.empty_like(mc_xyz)
    rot[:, 0] = mc_xyz[:, 0]
    rot[:, 1] = mc_xyz[:, 2]
    rot[:, 2] = ymax - mc_xyz[:, 1]
    d, _ = cKDTree(rot).query(xyz, k=1)
    print("\n  [3] Distance to nearest initial-MC node (voxel units):")
    summarise("dist to MC node", d, hi_is_good=False)
    print(f"  (MC node count: {len(mc_xyz)})")


def main():
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--label",        required=True)
    p.add_argument("--smesh",        required=True)
    p.add_argument("--mc-xyz",       required=True)
    p.add_argument("--thr-boundary", type=float, default=20.0)
    p.add_argument("--thr-internal", type=float, default=40.0)
    a = p.parse_args()
    report(a.label, a.smesh, a.mc_xyz, a.thr_boundary, a.thr_internal)


if __name__ == "__main__":
    main()
