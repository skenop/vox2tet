#pragma once

// Half-edge data structure used by the remeshing stages. Row layout:
//
//   hedges[i, 0] = end-vertex of half-edge i  (the vertex it points TO)
//   hedges[i, 1] = next half-edge index       (within the same triangle)
//   hedges[i, 2] = previous half-edge index   (within the same triangle)
//   hedges[i, 3] = opposite half-edge         (UINT32_MAX on the boundary)
//   hedges[i, 4] = interface id               (UINT32_MAX if not assigned)
//
// Layout: 3 half-edges per triangle. For triangle i the half-edges live
// at rows i, i+ntr, i+2*ntr.

#include <Eigen/Core>
#include <vector>

#include "vox2tet/core/types.hpp"
#include "vox2tet/marching_cubes/contouring.hpp"

namespace vox2tet::mesh {

using HalfEdges = Eigen::Matrix<u32, Eigen::Dynamic, 5, Eigen::RowMajor>;

// Build the half-edge structure from a triangle list. `itf` (optional)
// assigns each triangle range to an interface id via column 4.
//
// Implementation note: the opposite-edge pairing uses a stable sort on
// the canonical (min(u,v), max(u,v)) edge pair plus a sweep over equal
// runs to pair adjacent half-edges. This handles 2-manifold meshes
// (count == 2 per edge) and yields UINT32_MAX for boundary edges
// (count == 1). Non-manifold edges (count > 2) trigger a runtime error
// — the proper non-manifold link table is built later by
// `create_boundary_hedges_links` (TODO, phase 9).
HalfEdges triangles_to_hedges(const Triangles& tri,
                              const std::vector<marching_cubes::Interface>* itf = nullptr);

// Reconstruct a triangle list from a half-edge structure. Mirrors
// `hedges2triangles2`. When `itf` is provided, triangles are sorted
// per-interface and the `(first, count)` columns of `itf` are updated.
Triangles hedges_to_triangles(const HalfEdges& he,
                              std::vector<marching_cubes::Interface>* itf = nullptr);

// Mask of "not fixed" half-edges + vertices. Mirrors `get_not_fixed`.
// Outputs:
//   not_fixed_v[i] = true when vertex i may move during remeshing
//                   (i.e. it is a regular surface vertex — not boundary,
//                    not edge, not corner; semantic from `is_bnode[0]`).
//   not_fixed_h[i] = true when half-edge i is internal (opposite exists)
//                   AND not part of the bounding-box "frame" (sharp
//                   internal dihedral straddling bb edges).
//
// Requires the NodeTypeMask emitted by extract_material_interface_info.
struct NotFixedMasks {
    std::vector<std::uint8_t> not_fixed_h;
    std::vector<std::uint8_t> not_fixed_v;
};
NotFixedMasks get_not_fixed(const HalfEdges& he, const Coords& xyz,
                            const marching_cubes::NodeTypeMask& is_bnode);

// Save half-edges to .npy with the canonical (N*3, 5) uint32 layout.
void save_hedges_npy(const std::string& path, const HalfEdges& he);

// -- Non-manifold half-edge blink table ----------------------------------
//
// `create_blinks(he, xyz)` returns a per-half-edge array (size = he.rows())
// where `blinks[h]` is the *next* half-edge in the cycle of half-edges
// sharing the same canonical edge:
//   * manifold edge (count == 2): blinks[h] is the original opposite.
//   * non-manifold count == 3: cycle h0 → h1 → h2 → h0, ordered by
//     half-edge index after a canonical-edge sort.
//   * non-manifold count == 4: cycle ordered by directional dihedral
//     angle so adjacent cycle steps are between geometrically adjacent
//     triangles.
//   * boundary edge (count == 1): blinks[h] is kInvalidU32.
//
// Mirrors `mesh_processing.create_boundary_hedges_links`.
std::vector<u32> create_blinks(const HalfEdges& he, const Coords& xyz);

// Two-step composition of `create_blinks`. For each half-edge h that
// has at least one cycle-mate, `blinks2[h]` = blinks[blinks[h]]; if that
// would land back on h (count==2 case), `blinks2[h]` collapses to
// blinks[h] (the only other neighbour). Mirrors
// `mesh_processing.get_second_boundary_hedges_links`.
std::vector<u32> get_second_blinks(const std::vector<u32>& blinks);

// For each vertex v, the list of boundary half-edges starting at v
// (i.e. those whose column-3 opposite is `kInvalidU32`).
// `out.first[v]` = the list; `out.second[v]` = 1 iff v has any
// boundary half-edge.
struct V2HLists {
    std::vector<std::vector<u32>> by_vertex;
    std::vector<std::uint8_t>     is_boundary_vertex;
};
V2HLists get_v2h_lists(const HalfEdges& he);

// `is_edge_exist(he, he_boundary, v)`: does any non-manifold edge with
// vertex `v` and one of the half-edges in `he_boundary` exist? Mirrors
// `mesh_processing.is_edge_exist` — used to gate boundary collapses
// that would otherwise create a duplicated edge between two boundary
// vertices.
bool is_edge_exist(const HalfEdges& he, const std::vector<u32>& he_boundary, u32 v);

}  // namespace vox2tet::mesh
