#pragma once

// High-level driver of the marching-cubes phase. Reads a (pre-processed)
// image, classifies cells by material count, dispatches each cell to
// the appropriate LUT, assembles a triangle mesh, deduplicates
// vertices, and writes the initial STL/NPY artefacts.

#include <array>
#include <cstdint>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "vox2tet/core/types.hpp"
#include "vox2tet/image/volume.hpp"

namespace vox2tet::marching_cubes {

// Result of createInitialMesh.
struct InitialMesh {
    Coords    xyz;    // double, Nx3 — coordinates in voxel-half units
    Triangles tri;    // u32, Mx3
};

// Interface descriptor: one row per (material_1, material_2) interface.
// Encoded as (interface_code, first_triangle, count, material_1, material_2).
struct Interface {
    std::uint32_t code;
    std::uint32_t first;
    std::uint32_t count;
    std::uint32_t mat1;
    std::uint32_t mat2;
};

// Node-type classification. 8 boolean rows × N vertices. Mirrors the
// `nodes_type` array in extractMaterialInterfaceInfo.
struct NodeTypeMask {
    std::array<std::vector<std::uint8_t>, 8> masks;
};

// Build the initial surface from a *pre-prepared* image (already
// extended-with-boundary). Mirrors `createInitialMesh` in the reference.
InitialMesh create_initial_mesh(const image::Volume& ext_vox,
                                bool do_2x2patterns,
                                const std::string& path_base);

// Classify nodes & group triangles into interfaces. Mirrors
// `extractMaterialInterfaceInfo` and mutates `tri` in place to sort
// triangles by interface code.
std::pair<std::vector<Interface>, NodeTypeMask>
extract_material_interface_info(const image::Volume& ext_vox,
                                const Coords& xyz,
                                Triangles& tri,
                                const std::string& path_base);

// Save STL artefacts. Mirrors `save_surfaces_stl`.
void save_surfaces_stl(const std::string& path_base,
                       const Coords& xyz,
                       const Triangles& tri,
                       const std::vector<Interface>* interfaces,
                       bool do_save_interfaces,
                       bool do_save_grains_stl,
                       bool do_save_grains_inp);

}  // namespace vox2tet::marching_cubes
