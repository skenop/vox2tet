# vox2tet

Convert a labelled voxel image (3-D TIFF) into a high-quality conforming
tetrahedral mesh for finite-element computation on polycrystalline
materials, fibre-reinforced composites, or any other multi-material
3-D image.

The pipeline runs as a single binary, `run_vox2tet`:

```
voxel TIFF ─► marching cubes ─► smoothing ─► remeshing ─► TetGen ─► .inp
```

Based on the method from
[Sinchuk et al. (Composite Structures, 2022)][paper]. A reference
Python implementation is also available at
<https://src.koda.cnrs.fr/pprime-endo/s2m/-/tree/main/src/vox2tet>.

> **Full documentation:** [`doc/DOCUMENTATION.md`](doc/DOCUMENTATION.md)
> covers every JSON parameter, intermediate file, key algorithm, and
> mesh-quality guarantee.

## Quick start

```bash
# Build
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

# Run on a TIFF
build/run_vox2tet  path/to/image.tif
# → outputs under path/to/vox2tet_out/image_*
```

With no arguments, `run_vox2tet` prints usage.

## Dependencies

| Library          | Source                                          |
|------------------|-------------------------------------------------|
| Eigen3           | FetchContent (automatic)                        |
| nlohmann/json    | FetchContent (automatic)                        |
| libtiff          | `apt install libtiff-dev` / `brew install libtiff` |
| TetGen (runtime) | `apt install tetgen` / `brew install tetgen` — must be on `$PATH` |
| OpenMP           | optional; comes with gcc/clang/MSVC             |

C++17 compiler + CMake ≥ 3.20.

## Input

A 3-D TIFF where each voxel value is the **material id** of that voxel.
Any integer dtype, any number of distinct labels. Background should be a
distinct id (typically 0).

## Pipeline (1-line summary of each stage)

1. **Image preparation** — small-component removal, diagonal-conn fix-up, 6-voxel boundary padding.
2. **Marching cubes** — Wu–Sullivan multi-material LUTs build a watertight non-manifold surface.
3. **Pre-remesh smoothing** — Laplacian + tangential smoothing with dihedral & corner-angle reverts.
4. **Remeshing** — `n_remesh_itr` iterations of split → collapse → flip → smooth, then active dihedral-repair and sliver-repair passes.
5. **TetGen** — shells out to `tetgen -pYA -q2/15 -o/150 -nn -V` to build the volume mesh.
6. **Abaqus export** — whole-volume and per-grain `.inp` files.

## Main outputs

For input `image.tif`, written to `<dir>/vox2tet_out/image_*`:

| File                | Content                                                |
|---------------------|--------------------------------------------------------|
| `image_RE.smesh`    | Watertight multi-material surface mesh (TetGen input). |
| `image_RE_ALL.stl`  | The same surface as a coloured STL (per-material hue). |
| `image_RE_G_<m>.stl`| Per-grain coloured STL (one per material id `m`).      |
| `image_RE.1.node`   | Final tet-mesh node coordinates.                       |
| `image_RE.1.ele`    | Final tet-mesh elements with per-element region id.    |
| `image_RE.1.face`   | Final tet-mesh boundary face list with interface ids.  |
| `image_RE.inp`      | Whole-volume Abaqus mesh, one element set per grain.   |
| `<m>.inp`           | Per-grain Abaqus mesh.                                 |
| `image.json`        | Snapshot of the resolved `Settings` for this run.      |

A long tail of `_xyz.npy`, `_tr.npy`, `_S_ALL.stl`, `_C3_R64.tif`, etc.
is also written for debugging / restarting from a specific stage —
see [`doc/DOCUMENTATION.md`](doc/DOCUMENTATION.md) for the full list.

## Configuration (JSON)

Run with a saved settings file:

```bash
build/run_vox2tet  config.json
```

The most useful fields:

| Field                       | Default | Effect |
|-----------------------------|---------|--------|
| `n_remesh_itr`              | 7       | Outer remesh iterations. |
| `Lmax`                      | 40.0    | Maximum edge length (voxel units). |
| `min_dangle_internal`       | 40.0    | Min dihedral (°) on interior interfaces. |
| `min_dangle_boundary`       | 20.0    | Min dihedral (°) on bounding-box edges. |
| `min_corner_angle_internal` | 30.0    | Min triangle corner angle (°). |
| `min_corner_angle_boundary` | 20.0    | Same, for bbox-only triangles. |
| `do_tetgen_meshing`         | true    | Run TetGen (else stop at the surface). |
| `do_save_remesh_grains_stl` | true    | Per-grain coloured STLs. |

Full list with descriptions: [`doc/DOCUMENTATION.md` §5][doc-cfg].

## Example: JMA_30

```bash
cd tests/data/JMA_30
../../../build/run_vox2tet  JMA_30.tif
```

A small polycrystal (~50 grains, 30³ voxels). Pipeline takes ~3 s on a
4-core workstation.

| Complete model | Internal grain interfaces |
|---|---|
| ![JMA_30 full](doc/result_view/JMA_30_result2.png) | ![JMA_30 interfaces](doc/result_view/JMA_30_result2_intrface.png) |

Resulting mesh quality:

| Metric                           | Surface  | Tet mesh |
|----------------------------------|----------|----------|
| Nodes / elements                 | 16 208 / 36 653 tri | 23 680 / 125 184 tets |
| Min angle                        | **17.95°** (corner) | **17.96°** (dihedral) |
| Min internal dihedral            | **40.05°** | — |
| Min boundary dihedral            | 42.18°   | — |
| Violations of 40° / 20° threshold | 0 / 0    | — |
| Inverted tets (V < 0)            | —        | **0** |
| Tets with dihedral < 20°         | —        | 6 of 125 k |

## Programmatic use

Link against `libvox2tet.a` and call `vox2tet::generate(settings)`. The
CLI is just 15 lines of glue — see `src/app/run_vox2tet.cpp`.

## Reference

[paper]: https://doi.org/10.1016/j.compstruct.2022.116003

* Sinchuk Y., et al. *X-ray CT based multi-layer unit cell modeling of
  carbon fiber-reinforced textile composites: segmentation, meshing
  and elastic property homogenization.* **Composite Structures** 298
  (2022). <https://doi.org/10.1016/j.compstruct.2022.116003>

[doc-cfg]: doc/DOCUMENTATION.md#5-json-configuration-parameters
