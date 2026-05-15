# vox2tet — Full documentation

## 1. Purpose

**vox2tet** converts a labelled voxel image (3-D TIFF, one integer
material id per voxel) into a high-quality conforming tetrahedral
volume mesh suitable for finite-element computations on polycrystalline
materials, fibre-reinforced composites, X-ray-CT-segmented
microstructures, and any other multi-material 3-D image. Specifically
it produces:

* a single watertight, multi-material **surface mesh** with one
  triangle per pair of adjacent voxel labels (Wu–Sullivan multi-material
  marching cubes), respecting all triple-line and quad-point features
  exactly;
* a clean, FEM-grade **tetrahedral volume mesh** (via TetGen) whose
  element shapes are bounded by user-defined dihedral and corner-angle
  thresholds;
* a per-grain decomposition of the volume mesh for downstream solvers
  (Abaqus `.inp` per material id).

vox2tet implements the method described in:

> Sinchuk Y., et al. *X-ray CT based multi-layer unit cell modeling of
> carbon fiber-reinforced textile composites: segmentation, meshing
> and elastic property homogenization.*
> **Composite Structures**, 298 (2022).

A reference Python implementation of the same method is published
separately at
<https://src.koda.cnrs.fr/pprime-endo/s2m/-/tree/main/src/vox2tet>.

## 2. How to use

After [building](#5-compilation) you get a single executable,
`run_vox2tet`. Pass it a voxel TIFF and the pipeline runs end-to-end:

```bash
run_vox2tet <input.tif | input.json> [out_path_base] [ncpus]
```

| Arg              | Default                                                    |
|------------------|------------------------------------------------------------|
| input            | required — a `.tif` voxel image or a saved settings `.json` |
| `out_path_base`  | for `.tif`: `<input-dir>/vox2tet_out/<input-stem>`<br>for `.json`: value stored in the file |
| `ncpus`          | 4                                                          |

For programmatic use, link against `libvox2tet.a` and call the public
API directly:

```cpp
#include "vox2tet/core/settings.hpp"
#include "vox2tet/pipeline.hpp"

vox2tet::Settings s;
s.input_img_file            = "/data/scan.tif";
s.out_path_base             = "/data/out/scan";
s.min_corner_angle_internal = 25.0;        // tighter sliver guard
s.do_save_remesh_grains_stl = true;        // per-grain colored STLs
vox2tet::generate(s);
```

A `Settings` instance can be saved as JSON for later replay:

```cpp
s.dump_json("/data/out/scan.json");
```

The CLI accepts that JSON in place of a TIFF.

## 3. Example: JMA_30

Test data ships with the repo under `tests/data/JMA_30/JMA_30.tif`.
It is a 30×30×30 voxel image of a polycrystalline microstructure with
~50 distinct grains.

```bash
cd tests/data/JMA_30
run_vox2tet JMA_30.tif
# output: ./vox2tet_out/JMA_30_*
```

The full pipeline (image prep → marching cubes → smoothing → remeshing
→ tetgen → abaqus export) completes in ~3 seconds on a 4-core
workstation for this dataset.

### Resulting mesh views

| Complete model (per-grain colour) | Internal grain interfaces |
|---|---|
| ![JMA_30 full](result_view/JMA_30_result2.png) | ![JMA_30 interfaces](result_view/JMA_30_result2_intrface.png) |

Open `vox2tet_out/JMA_30_RE_ALL.stl` in MeshLab (or the per-grain
`vox2tet_out/JMA_30_RE_G_<m>.stl` files) to see the equivalent views
yourself — the colour palette is deterministic per material id via a
golden-angle hue rotation.

### Resulting mesh quality

**Surface mesh** (`JMA_30_RE.smesh`, 16 208 nodes, 36 653 triangles):

| Metric                              | Value           |
|-------------------------------------|-----------------|
| Min corner angle                    | **17.95°**      |
| Corner angles < 20° / < 30°         | 4 / 228 of 110 k|
| Min dihedral, internal interface    | **40.05°**      |
| Min dihedral, boundary edge         | 42.18°          |
| Pairs below 40° internal dihedral   | **0**           |
| Pairs below 20° boundary dihedral   | **0**           |
| Max node-to-initial-MC distance     | 0.73 voxel      |

**Tetrahedral mesh** (`JMA_30_RE.1.{node,ele,face}`, 23 680 nodes,
125 184 tets — TetGen invoked with `-pYA -q2/15 -o/150 -nn -V`).
Numbers below are copied directly from TetGen's `Mesh quality
statistics` block printed at the end of the run, so they correspond
exactly to what you will see in your own console output:

| TetGen metric                  | Value      |
|--------------------------------|------------|
| Smallest dihedral angle        | **10.17°** |
| Largest dihedral angle         | 162.04°    |
| Smallest face (triangle) angle | 13.10°     |
| Largest face (triangle) angle  | 149.77°    |
| Smallest aspect ratio          | 1.23       |
| Largest aspect ratio           | 11.94      |
| Smallest tet volume            | 0.0041     |
| Largest tet volume             | 4.38       |
| Shortest edge                  | 0.291      |
| Longest edge                   | 4.81       |

Dihedral histogram (TetGen bins, 6 dihedrals per tet, total
6 × 125 184 = 751 104):

| Range                | Count   |    | Range                | Count   |
|----------------------|--------:|----|----------------------|--------:|
| 0–10°                | 0       |    | 80–110°              | 175 688 |
| 10–20°               | 2 839   |    | 110–120°             | 28 330  |
| 20–30°               | 23 656  |    | 120–130°             | 17 047  |
| 30–40°               | 57 658  |    | 130–140°             | 10 393  |
| 40–50°               | 94 800  |    | 140–150°             | 5 835   |
| 50–60°               | 118 133 |    | 150–160°             | 160     |
| 60–70°               | 118 315 |    | 160–170°             | 6       |
| 70–80°               | 98 244  |    | 170–180°             | 0       |

The surface-side dihedral and corner-angle guards (described in
[§9 Key algorithms](#9-key-algorithms)) keep the worst tet shapes in
the 10°–20° band rather than the typical 0°–10° band that uncontrolled
remeshing produces for multi-material data: of 751 104 dihedrals,
none falls below 10° and only 6 exceed 160°.

## 4. Input data format

### Image input

A 3-D TIFF with one integer-typed channel and one slice per Z position
(any of `uint8` / `uint16` / `uint32` / `int*`).

* Each voxel value is interpreted as a **material id**.
* Materials with the same id are grouped into the same grain regardless
  of spatial location (use `do_remove_small` + connected-component
  analysis if you need disjoint grains to get distinct ids).
* Background should be a distinct id (typically 0).
* No restriction on the number of distinct labels — vox2tet auto-detects
  the maximum label and adds 6 boundary "phantom" labels around the
  image for the marching-cubes step.

### Settings input (`.json`)

vox2tet's `Settings` struct can be dumped to / loaded from JSON. The
format is byte-compatible with the Python `Cglobal` config: a 3-element
top-level array `[date_string, "Cglobal", { ...attributes... }]`. The
CLI accepts either form (wrapped array or bare object) and saves
itself in the wrapped form for round-trip compatibility.

## 5. JSON configuration parameters

All `Settings` fields, grouped logically:

### General workflow

| Field                       | Default        | Purpose |
|-----------------------------|----------------|---------|
| `input_img_file`            | `./tests/data/JMA_10/JMA_10.tif` | Path to the input TIFF. |
| `out_path_base`             | `./tests/data/JMA_10/vox2tet_out/JMA_10` | Prefix for all output files. |
| `ncpus`                     | 4              | Worker-thread count (OpenMP). |
| `do_initial_surface`        | true           | Run marching cubes. |
| `do_preremesh_steps`        | true           | Run pre-remesh Laplacian smoothing. |
| `do_remeshing`              | true           | Run split/collapse/flip + dihedral/sliver repair loop. |
| `do_tetgen_meshing`         | true           | Invoke TetGen on the cleaned surface. |
| `do_abaqus_verification`    | true           | Emit per-grain `.inp` files. |
| `do_init_surf_load`         | false          | Skip MC, reload from `_xyz.npy` + `_tr.npy`. |
| `do_preremesh_load`         | false          | Skip pre-remesh, reload from `_xyz_s.npy`. |
| `do_print_redirect`         | false          | Tee log output to `<out_path_base>_OUT.txt`. |
| `do_x_rotation`             | true           | Rotate axes for TetGen (Y↔Z swap and Y flip). |

### Initial-surface (marching cubes)

| Field                       | Default | Purpose |
|-----------------------------|---------|---------|
| `max_remove_size`           | 64      | Remove connected components smaller than this voxel count. |
| `connectivity`              | 3       | Connectivity for component analysis (1/2/3 → 6/18/26-conn.). |
| `do_remove_small`           | true    | Enable the small-component filter. |
| `do_diagonal_connectivity`  | true    | Use diagonal-aware MC lookup for ambiguous 2×2 patterns. |
| `do_2x2patterns`            | true    | Resolve 2×2 face ambiguities via extra disambiguation table. |
| `do_show_hist`              | false   | Print MC histogram of cube cases. |
| `do_save_init_interfaces`   | false   | Save one STL per interface pair (debug). |
| `do_save_init_grains_stl`   | false   | Save per-grain STL for the initial MC mesh. |
| `do_save_init_grains_inp`   | false   | Save per-grain INP for the initial MC mesh. |

### Pre-remesh smoothing

| Field                       | Default | Purpose |
|-----------------------------|---------|---------|
| `n_smooth_steps`            | 7       | Laplacian iterations during pre-remesh. |
| `smooth_alpha`              | 0.5     | Laplacian damping (0 = no smoothing, 1 = pure averaging). |
| `max_D2self`                | 0.8     | Per-vertex displacement cap (voxel units). |
| `min_D2other`               | 0.3     | Minimum distance to any other vertex (collision guard). |
| `brep_smoothness`           | 7.0     | B-spline smoothness weight on grain-edge chains. |
| `brep_degree`               | 3       | B-spline degree for grain-edge smoothing. |
| `do_save_smooth_interfaces` | false   | Save one STL per interface after smoothing (debug). |
| `do_save_smooth_grains_stl` | false   | Per-grain STL after smoothing. |
| `do_save_smooth_grains_inp` | false   | Per-grain INP after smoothing. |

### Dihedral & sliver quality guards

| Field                          | Default | Purpose |
|--------------------------------|---------|---------|
| `min_dangle_boundary`          | 20.0    | Min dihedral (°) at bounding-box edges. |
| `min_dangle_internal`          | 40.0    | Min dihedral (°) at internal interface edges. |
| `min_corner_angle_boundary`    | 20.0    | Min triangle corner angle (°) when all 3 verts are on bbox. |
| `min_corner_angle_internal`    | 30.0    | Min triangle corner angle (°) on interior triangles. |

These four thresholds drive the four-layer guard described in
[§9 Key algorithms](#9-key-algorithms).

### Remeshing

| Field                       | Default | Purpose |
|-----------------------------|---------|---------|
| `n_remesh_itr`              | 7       | Outer iterations of split → collapse → flip → smooth. |
| `Lmax`                      | 40.0    | Maximum edge length (voxel units). |
| `do_save_remesh_interfaces` | false   | Per-interface STL after remesh (debug). |
| `do_save_remesh_grains_stl` | true    | Per-grain STL after remesh — main visualisation output. |
| `do_save_remesh_grains_inp` | true    | Per-grain INP after remesh. |

## 6. Output files

### Main outputs (everything you usually need)

| File                              | Format             | Content |
|-----------------------------------|--------------------|---------|
| `<base>_RE.smesh`                 | TetGen surface     | Watertight multi-material surface mesh, ready for TetGen. |
| `<base>_RE_ALL.stl`               | Binary STL, coloured | The same surface as a viewer-friendly STL with one colour per material id. |
| `<base>_RE_G_<m>.stl`             | Binary STL, coloured | Per-grain surface (one file per material id `m`). |
| `<base>_RE.1.node`                | TetGen node list   | Final tetrahedral-mesh node coordinates. |
| `<base>_RE.1.ele`                 | TetGen element list| Final tetrahedral-mesh elements (4 nodes + region id per row). |
| `<base>_RE.1.face`                | TetGen face list   | Boundary triangle list with interface markers. |
| `<base>_RE.1.edge`, `.1.neigh`    | TetGen aux         | Edge list and tet-adjacency, written by TetGen. |
| `<base>_RE.inp`                   | Abaqus `.inp`      | Whole-volume tet mesh, one element set per grain. |
| `<m>.inp`                         | Abaqus `.inp`      | Per-grain tet mesh (one file per material id `m`). |
| `<base>.json`                     | Settings JSON      | Resolved `Settings` snapshot; re-loadable with the same CLI. |

### Intermediate / debug outputs

These are produced by default but are only needed if you want to inspect
or restart from a specific pipeline stage. They can be ignored for most
workflows.

| File                       | Stage                          | Content |
|----------------------------|--------------------------------|---------|
| `<base>_xyz.npy`           | initial surface                | MC node coordinates (NumPy array). |
| `<base>_tr.npy`            | initial surface                | MC triangle indices. |
| `<base>_att.npy`           | initial surface                | Per-triangle (mat1, mat2) attribute pairs. |
| `<base>_ntp.npy`           | initial surface                | Per-vertex node-type bitmask (corner / edge / face / interior). |
| `<base>_norm.npy`          | initial surface                | Area-weighted vertex normals. |
| `<base>_hedges.npy`        | initial surface                | Half-edge structure. |
| `<base>_ALL0.stl`          | initial surface                | MC surface, no colour. |
| `<base>_ALL.stl`           | initial surface                | MC surface, coloured. |
| `<base>_C3_R64.tif`        | image prep                     | Image after small-component removal + diagonal-conn fix-up. |
| `<base>_L_C3_R64.tif`      | image prep                     | Labelled connected-components volume. |
| `<base>_C3_R64.txt`        | image prep                     | Stats on removed components. |
| `<base>_EXT.xyz`, `_FIX.xyz`, `_MML.xyz` | initial surface     | Boundary / fixed / multi-material line node coordinates (debug). |
| `<base>_E.ply`             | smoothing                      | Initial grain-edge chains. |
| `<base>_S_E.ply`           | smoothing                      | Smoothed grain-edge chains. |
| `<base>_S_ALL.stl`         | smoothing                      | Surface after smoothing. |
| `<base>_xyz_s.npy`         | smoothing                      | Smoothed node coordinates (for re-load). |
| `<base>_sz_fld.npy`        | smoothing                      | Per-vertex sizing field (target edge length). |

## 7. Pipeline overview

```
voxel TIFF (labels)
   │
   │ 1. IMAGE PREPARATION
   │    - small-component removal
   │    - diagonal-connectivity fix-up
   │    - 6-voxel boundary padding
   │
   │ 2. INITIAL SURFACE
   │    - Wu–Sullivan multi-material marching cubes (LUT-driven)
   │    - half-edge construction + manifold/non-manifold link table
   │    - per-vertex node-type classification
   │
   │ 3. PRE-REMESH SMOOTHING
   │    - 1-D B-spline smoothing of grain-edge chains
   │    - Laplacian + tangential surface smoothing
   │    - dihedral & corner-angle reverts (Layer 1 + Layer 2)
   │
   │ 4. REMESHING (n_remesh_itr loops)
   │    - edge split (lengths > 4/3 · sizing)
   │    - interior edge collapse (lengths < 4/5 · sizing)
   │    - quad-diagonal flip on bad shape factor
   │    - tangential smoothing with cycle-min dihedral guard
   │    - post-loop dihedral repair (Layer 3)
   │    - post-loop sliver repair (Layer 4)
   │
   │ 5. TETGEN
   │    - call: tetgen -pYA -q2/15 -o/150 -nn -V <base>_RE.smesh
   │    - re-write per-region tetrahedral mesh
   │
   │ 6. ABAQUS EXPORT
   │    - per-grain .inp + whole-volume .inp
   ▼
TetGen .1.{node,ele,face} + Abaqus .inp files
```

## 8. Compilation

```bash
cd cpp
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

This produces `build/run_vox2tet` and `build/libvox2tet.a`. Optional
install:

```bash
cmake --install build --prefix /usr/local
```

### Build options

| Option                          | Default | Purpose |
|---------------------------------|---------|---------|
| `CMAKE_BUILD_TYPE`              | Release | Use `Debug` for asserts + symbols. |
| `VOX2TET_BUILD_TESTS`           | ON      | Build unit + integration tests. |
| `VOX2TET_USE_OPENMP`            | ON      | Parallelise hot loops via OpenMP. |
| `VOX2TET_DEBUG_BRIDGE`          | OFF     | Enable scipy/numpy reference oracle for debugging. |
| `VOX2TET_PYTHON_BIN`            | empty   | Path to a Python with numpy + scipy (only used by the debug bridge). |

### External library dependencies

| Library          | Purpose                              | How it's obtained |
|------------------|--------------------------------------|-------------------|
| **Eigen3**       | Dense linear algebra                 | Fetched automatically (`FetchContent`). |
| **nlohmann/json**| Settings JSON parsing                | Fetched automatically (`FetchContent`). |
| **libtiff**      | TIFF reading/writing                 | System install — `apt install libtiff-dev` / `brew install libtiff`. |
| **OpenMP**       | Loop parallelism (optional)          | System compiler (gcc/clang/MSVC). |
| **TetGen**       | Tetrahedral mesh generation (runtime)| System install — `apt install tetgen` / `brew install tetgen`. Invoked via `system()`; the binary must be on `$PATH`. The surface pipeline runs fine without it; only the `.1.{node,ele,face}` outputs are skipped. |
| **GoogleTest**   | Test framework (test-only)           | Fetched automatically when `VOX2TET_BUILD_TESTS=ON`. |

C++17-capable toolchain (gcc ≥ 9, clang ≥ 10, MSVC 2019+) and CMake
≥ 3.20.

## 9. Key algorithms

### 9.1 Wu–Sullivan multi-material marching cubes

The classical marching-cubes algorithm handles only binary
inside/outside images. For an image with `k` material labels we need
`k(k-1)/2` interfaces and a guarantee that triple lines (where three
materials meet at an edge) and quad points (where four meet at a
corner) are reproduced exactly without cracks.

vox2tet uses the Wu–Sullivan formulation (2003) with shipped lookup
tables:

* a per-cube "case index" computed from the 8-corner rank pattern
  rather than a binary mask;
* a triangulation table that produces non-manifold triangle fans at
  triple lines (3 triangles meeting on one edge);
* a quad-point disambiguation table that splits four-way junctions
  consistently across adjacent cubes.

The output is stored as a half-edge structure with an extra "blink
table" that walks all triangles around a non-manifold edge in a
canonical cyclic order — essential for the dihedral-guard math
described below.

### 9.2 Laplacian + tangential smoothing with dihedral guards

The MC surface is geometrically jagged. We apply Laplacian smoothing
(α-blended average over the 1-ring) projected onto the per-vertex
normal plane, but each candidate vertex move is **gated by the
dihedral metric**: if any pair-wise dihedral around an incident edge
drops below the threshold AND was already small, the move is reverted.

For non-manifold edges (triple lines, quad points) the canonical
opposite-half-edge `he(h, 3)` covers only one of the 2–3 adjacent
triangle pairs. vox2tet uses a **cycle-min dihedral metric** that
takes the minimum dihedral across all triangles in the blink cycle,
catching folds the canonical pair would miss. This is "Layer 1" of the
quality system.

When a move is rejected, **all** apex vertices of the offending blink
cycle are reverted in addition to the edge endpoints (Layer 2) — this
prevents the third vertex of a triangle from quietly producing a fold
even when the two edge endpoints look fine in isolation.

### 9.3 Remeshing: split / collapse / flip

The remesh inner loop runs `n_remesh_itr` iterations of:

1. **Split** every not-fixed edge whose length exceeds 4/3 × min sizing
   at its two endpoints. New vertices land at edge midpoints.
2. **Collapse** every interior edge shorter than 4/5 × min sizing,
   provided the result is still manifold, no incident dihedral drops
   below threshold, and the new triangle normals stay consistent
   (Hoppe-style adjacency check).
3. **Flip** the diagonal of every degenerate quad whose shape factor
   improves under flipping, with full dihedral-regression checks
   against every cycle-mate (so flips near triple lines see all the
   triangle pairs that matter).
4. **Smooth** (tangential, with the Layer 1+2 guards above).

### 9.4 Dihedral repair (Layer 3)

After the remesh loop converges, some non-manifold edges still have
dihedral folds the regression-only guards can't fix (the smoother saw
each move in isolation and never accumulated enough budget to widen
the offending fold).

`dangle_repair` actively walks every non-manifold edge with min
dihedral below threshold, picks the worst-fold pair, and moves the
apex of one of the two offending triangles in the direction
perpendicular to the edge, away from the plane of the other. The
move is accepted iff (a) the targeted dihedral strictly increases,
(b) no neighbour dihedral drops below threshold and below its prior
value, (c) drift from the original MC position stays within 1 voxel.

Runs for up to 5 passes; early-exits on no progress.

### 9.5 Sliver repair (Layer 4)

Even with clean dihedrals, the remesh occasionally leaves a few needle-
shaped triangles (one interior angle very small, the others near 90°).
`sliver_repair` walks every triangle with min corner angle below
`min_corner_angle_internal` and moves its worst-angle vertex toward
the midpoint of the opposite edge, subject to the same drift and
neighbour-quality guards as `dangle_repair`.

This is geometrically valid only for **needle** slivers. Cap-shaped
slivers (one angle near 180°) can't be removed by vertex relocation
and require topology-changing operations — see "Limitations" in the
PROGRESS notes.

### 9.6 STL colouring

The 16-bit per-facet attribute slot in binary STL is interpreted by
MeshLab / 3DViewer as a 5-5-5 RGB colour (valid-bit + R5 + G5 + B5).
vox2tet derives a colour from each material id by stepping the HSV hue
in golden-angle increments (≈ 137.508° per id), keeping saturation =
0.75 and value = 0.85. Consecutive material ids therefore land far
apart on the colour wheel.

## 10. Reference

* Sinchuk Y., et al. *X-ray CT based multi-layer unit cell modeling of
  carbon fiber-reinforced textile composites: segmentation, meshing
  and elastic property homogenization.*
  **Composite Structures**, 298 (2022).
  <https://doi.org/10.1016/j.compstruct.2022.116003>
* Reference Python implementation of the same method (separate repo):
  <https://src.koda.cnrs.fr/pprime-endo/s2m/-/tree/main/src/vox2tet>
* Wu, Z. & Sullivan, J. M. *Multiple material marching cubes
  algorithm.* International Journal for Numerical Methods in
  Engineering 58:2 (2003), 189-207.
* Si, H. *TetGen, a Delaunay-Based Quality Tetrahedral Mesh
  Generator.* ACM Transactions on Mathematical Software, 41:2 (2015).
