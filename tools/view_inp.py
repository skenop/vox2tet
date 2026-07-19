#!/usr/bin/env python3
"""view_inp.py — interactive cut-away viewer for Abaqus .inp tet meshes.

Shows the tetrahedral volume mesh written by vox2tet (`<base>_RE.inp`,
element type C3D4, one `*Elset` per material) with a movable section
plane exposing the interior tets, one colour per material.

Controls
--------
  mouse drag          rotate the model (matplotlib 3D navigation)
  scroll wheel        zoom
  bottom slider       move the section plane along the cut axis
  X / Y / Z buttons   choose the cut-axis
  "flip" checkbox     keep the other half
  S key               save a PNG next to the .inp

Run
---
  python tools/view_inp.py path/to/mesh_RE.inp
  python tools/view_inp.py mesh_RE.inp --save out.png   # headless render

Only numpy + matplotlib required. Sections recompute in milliseconds
even for ~150k-tet meshes: the face table and its unique-face index are
built once, a cut only re-bins face counts over the kept tets.
"""
import argparse
import sys
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# INP parsing (the subset save_inp emits: *NODE, *ELEMENT C3D4, *Elset)
# ---------------------------------------------------------------------------
def load_inp(path):
    node_id, node_xyz, tets = [], [], []
    elsets = {}          # name -> list of (first, last) 1-based inclusive
    mode, cur = None, None
    for raw in open(path):
        ls = raw.strip()
        if not ls:
            continue
        u = ls.upper()
        if u.startswith("*NODE"):
            mode = "n"; continue
        if u.startswith("*ELEMENT"):
            mode = "e"; continue
        if u.startswith("*ELSET"):
            mode = "s"
            cur = [p.split("=")[1] for p in ls.replace(" ", "").split(",")
                   if p.lower().startswith("elset=")][0]
            elsets.setdefault(cur, [])
            continue
        if ls.startswith("*"):
            mode = None; continue
        p = ls.split(",")
        if mode == "n":
            node_id.append(int(p[0]))
            node_xyz.append([float(p[1]), float(p[2]), float(p[3])])
        elif mode == "e":
            tets.append([int(x) for x in p[1:5]])
        elif mode == "s":
            v = [int(x) for x in p if x.strip()]
            if len(v) == 3:                      # generate: first, last, step
                elsets[cur] += [(v[0], v[1])]
            else:                                # plain id list
                elsets[cur] += [(i, i) for i in v]

    node_id = np.asarray(node_id)
    V = np.zeros((node_id.max() + 1, 3))
    V[node_id] = np.asarray(node_xyz)            # INP node ids are arbitrary
    T = np.asarray(tets, dtype=np.int64)

    mat = np.zeros(len(T), dtype=np.int64)       # per-tet material id
    for name, ranges in elsets.items():
        m = int(name.rsplit("-", 1)[-1]) if "-" in name else 0
        for a, b in ranges:
            mat[a - 1:b] = m                     # element ids are 1..N
    return V, T, mat


# ---------------------------------------------------------------------------
# Face tables (built once)
# ---------------------------------------------------------------------------
class FaceTable:
    def __init__(self, T):
        # 4 faces per tet, order chosen so face normals point outward
        # for positively-oriented tets (used for shading only).
        f = np.concatenate([T[:, [1, 2, 3]], T[:, [0, 3, 2]],
                            T[:, [0, 1, 3]], T[:, [0, 2, 1]]], axis=0)
        self.faces = f                            # (4T, 3) vertex ids
        self.tet = np.tile(np.arange(len(T)), 4)  # owner tet of each face
        key = np.sort(f, axis=1)
        self.uniq, self.uid = np.unique(key, axis=0, return_inverse=True)
        cnt = np.bincount(self.uid, minlength=len(self.uniq))
        self.surface = cnt[self.uid] == 1         # faces on the mesh boundary

    # Boundary faces of the subset of tets given by boolean mask `keep`:
    # returns (face rows, owner tets, is_cut) where is_cut marks faces
    # exposed by the section (interior faces of the full mesh).
    def boundary_of(self, keep):
        fk = keep[self.tet]
        cnt = np.bincount(self.uid[fk], minlength=len(self.uniq))
        sel = fk & (cnt[self.uid] == 1)
        return self.faces[sel], self.tet[sel], ~self.surface[sel]


# ---------------------------------------------------------------------------
# Viewer
# ---------------------------------------------------------------------------
def material_colors(mats):
    """Distinct colour per material id (golden-angle hue walk, like the
    pipeline's STL colouring)."""
    import matplotlib.colors as mcolors
    ids = np.unique(mats)
    cols = {}
    for i, m in enumerate(ids):
        h = (0.13 + 0.61803398875 * i) % 1.0
        cols[m] = np.array(mcolors.hsv_to_rgb((h, 0.55, 0.85)))
    return cols


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("inp", help="Abaqus .inp with C3D4 elements")
    ap.add_argument("--axis", choices="xyz", default="x", help="initial cut axis")
    ap.add_argument("--frac", type=float, default=0.5,
                    help="initial cut position, 0..1 across the bbox")
    ap.add_argument("--save", metavar="PNG",
                    help="render once to PNG and exit (headless)")
    args = ap.parse_args()

    if args.save:
        import matplotlib
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider, RadioButtons, CheckButtons
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    V, T, mat = load_inp(args.inp)
    print(f"{args.inp}: {len(V)} nodes, {len(T)} tets, "
          f"materials {sorted(np.unique(mat))}")
    ft = FaceTable(T)
    cent = V[T].mean(axis=1)                     # tet centroids
    cols = material_colors(mat)
    mn, mx = V.min(0), V.max(0)

    state = {"axis": "xyz".index(args.axis), "frac": args.frac,
             "flip": False, "zoom": 1.0}

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")
    fig.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.12)
    LIGHT = np.array([0.35, -0.5, 0.77])

    def redraw():
        a = state["axis"]
        cut = mn[a] + (mx[a] - mn[a]) * state["frac"]
        keep = (cent[:, a] >= cut) if state["flip"] else (cent[:, a] <= cut)
        faces, owner, is_cut = ft.boundary_of(keep)

        tri = V[faces]
        n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        ln = np.linalg.norm(n, axis=1, keepdims=True)
        ln[ln == 0] = 1
        n /= ln
        inten = 0.55 + 0.45 * np.clip((n @ LIGHT), -1, 1) * 0.5 + 0.225
        base = np.array([cols[m] for m in mat[owner]])
        # cut faces: full-strength material colour with visible tet
        # edges; skin faces: same hue, lambert-shaded and dimmed a bit
        fc = np.where(is_cut[:, None], base,
                      np.clip(base * inten[:, None] * 0.85, 0, 1))
        ec = np.where(is_cut[:, None], base * 0.45, base * 0.0)

        ax.clear()
        pc = Poly3DCollection(tri, facecolors=fc, edgecolors=ec,
                              linewidths=0.25)
        pc.set_zsort("average")
        ax.add_collection3d(pc)
        ctr = (mn + mx) / 2
        r = (mx - mn).max() / 2 * 1.05 * state["zoom"]
        ax.set_xlim(ctr[0] - r, ctr[0] + r)
        ax.set_ylim(ctr[1] - r, ctr[1] + r)
        ax.set_zlim(ctr[2] - r, ctr[2] + r)
        ax.set_box_aspect((1, 1, 1))
        ax.set_axis_off()
        if len(cols) <= 15:  # a 50-grain legend would swamp the figure
            handles = [plt.Rectangle((0, 0), 1, 1, fc=cols[m])
                       for m in sorted(cols)]
            ax.legend(handles, [f"material {m}" for m in sorted(cols)],
                      loc="upper right", frameon=False)
        ax.set_title(f"{Path(args.inp).name} — {len(T)} tets, cut "
                     f"{'xyz'[state['axis']]}={cut:.2f}", y=0.98)
        fig.canvas.draw_idle()

    if args.save:
        redraw()
        fig.savefig(args.save, dpi=180, bbox_inches="tight")
        print("wrote", args.save)
        return

    # --- widgets ----------------------------------------------------------
    s_ax = fig.add_axes([0.15, 0.05, 0.55, 0.03])
    slider = Slider(s_ax, "cut", 0.0, 1.0, valinit=state["frac"])

    r_ax = fig.add_axes([0.78, 0.02, 0.08, 0.09])
    radio = RadioButtons(r_ax, ("x", "y", "z"), active=state["axis"])

    c_ax = fig.add_axes([0.88, 0.02, 0.1, 0.09])
    c_ax.set_axis_off()
    check = CheckButtons(c_ax, ["flip"], [state["flip"]])

    def on_slide(v):
        state["frac"] = v
        redraw()

    def on_axis(label):
        state["axis"] = "xyz".index(label)
        redraw()

    def on_flip(_):
        state["flip"] = not state["flip"]
        redraw()

    def on_key(ev):
        if ev.key in ("s", "S"):
            png = str(Path(args.inp).with_suffix(".png"))
            fig.savefig(png, dpi=180, bbox_inches="tight")
            print("wrote", png)

    def on_scroll(ev):
        # Shrink the axis box to zoom in (scroll up), grow to zoom out.
        # Only the limits change — no geometry recompute — and the
        # factor persists in `state` so slider/axis changes keep it.
        state["zoom"] = float(np.clip(
            state["zoom"] * (0.85 if ev.button == "up" else 1 / 0.85),
            0.05, 20.0))
        ctr = (mn + mx) / 2
        r = (mx - mn).max() / 2 * 1.05 * state["zoom"]
        ax.set_xlim(ctr[0] - r, ctr[0] + r)
        ax.set_ylim(ctr[1] - r, ctr[1] + r)
        ax.set_zlim(ctr[2] - r, ctr[2] + r)
        fig.canvas.draw_idle()

    slider.on_changed(on_slide)
    radio.on_clicked(on_axis)
    check.on_clicked(on_flip)
    fig.canvas.mpl_connect("key_press_event", on_key)
    fig.canvas.mpl_connect("scroll_event", on_scroll)

    redraw()
    plt.show()


if __name__ == "__main__":
    sys.exit(main())
