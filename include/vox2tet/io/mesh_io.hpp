#pragma once

// Mesh writers for the formats vox2tet emits (binary STL, Abaqus .inp,
// TetGen .smesh, edge PLY). Reading is only implemented where the
// pipeline needs it (TetGen output is parsed by the tetgen runner
// directly).
//
// Conventions:
//   * `xyz` is an Nx3 double matrix (row i = node i coordinates).
//   * `tri` is an Mx3 uint32 matrix (row j = (n0, n1, n2) indices).
//   * INP element types: S3 (3 nodes) for surfaces, C3D4 (4 nodes) for
//     tetrahedra — matching `saveinp`.

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "vox2tet/core/types.hpp"

namespace vox2tet::io {

// Binary STL writer with the same 80-byte header convention used by the reference.
// `attributes` is optional 16-bit-per-triangle metadata (last 2 bytes of
// each 50-byte record). When `attributes_are_colors` is true the writer
// passes them through as-is (already 5-5-5 + valid-bit RGB). When false
// the values are interpreted as material ids and mapped to a high-
// contrast palette so MeshLab / 3DViewer render distinguishable grains
// instead of the green/black/blue clustering that comes from writing
// raw material indices to the colour slot.
void save_stl_blocks(const std::string& path,
                     const Coords& xyz,
                     const Triangles& tri,
                     const std::vector<std::uint16_t>* attributes = nullptr,
                     bool attributes_are_colors = false);

// Encode a material id as a 16-bit STL colour using the Magics / MeshLab
// convention: bit 15 = "colour valid", bits 10-14 = R, 5-9 = G, 0-4 = B
// (each 5-bit channel in 0..31). Hue is generated via a golden-angle
// step in HSV space so consecutive material ids land far apart on the
// colour wheel; saturation/value are clamped well below max so the mesh
// stays readable under MeshLab's default lighting.
std::uint16_t material_to_stl_color(std::uint32_t material_id);

// Abaqus INP. `esets` is parallel to `tri`: when present, elements are
// re-sorted by their element-set id and an *Elset, generate block is
// emitted per unique value (mirrors `saveinp`).
void save_inp(const std::string& path,
              const Coords& xyz,
              const Eigen::Ref<const Eigen::MatrixXi>& elems,
              const std::vector<std::int32_t>* esets = nullptr);

// TetGen surface input (.smesh).
void save_smesh(const std::string& path,
                const Coords& xyz,
                const Triangles& facets,
                const std::vector<std::uint32_t>* attributes = nullptr);

// ASCII PLY writer for an edge list (used by `saveBeges2ply`).
void save_edges_ply(const std::string& path,
                    const Coords& xyz,
                    const Eigen::Ref<const Eigen::Matrix<std::uint32_t,
                                                          Eigen::Dynamic, 2,
                                                          Eigen::RowMajor>>& edges,
                    const std::array<std::uint8_t, 3>* rgb = nullptr);

// Plain xyz text (one node per line, "%1.6f").
void save_xyz_text(const std::string& path,
                   const Coords& xyz,
                   const std::vector<std::uint8_t>& mask);

}  // namespace vox2tet::io
