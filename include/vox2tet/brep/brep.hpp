#pragma once

// Boundary-edge extraction and ordering — selected B-rep primitives:
//
//   * `getBoundaryEdges(triangles, is_bnode, nnodes)` - extract edges
//     whose two endpoints are both boundary/mml/fix nodes.
//   * `orderBedges(bedges, is_bnode)` - order edges into chains: closed
//     loops or open paths between fixed vertices.
//   * `saveBrepPLY(brep, xyz, base, rgb)` - dump chains as a single .ply
//     with optional per-edge RGB.
//
// The graph step ports NetworkX's connected-components + shortest-path
// traversal in pure C++ (BFS) — for our boundary graphs each vertex has
// degree 1 or 2 plus a small number of fixed/junction nodes; the
// ordering algorithm is straightforward.

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "vox2tet/core/types.hpp"
#include "vox2tet/marching_cubes/contouring.hpp"  // NodeTypeMask

namespace vox2tet::brep {

// Ordered chain of vertex indices for one connected component of the
// boundary graph. Open chain when the first and last endpoints are
// "fixed" vertices (is_bnode.masks[2]); otherwise closed (first == last).
using EdgeChain = std::vector<std::uint32_t>;

struct BrepOrdered {
    std::vector<std::array<std::uint32_t, 2>> brepf;   // (first, last) per chain
    std::vector<EdgeChain>                    brep;    // per-chain vertex sequence
};

// Edges (Nx2 u32) with the convention that `edges(:, 0) < edges(:, 1)`.
using BEdges = Eigen::Matrix<std::uint32_t, Eigen::Dynamic, 2, Eigen::RowMajor>;

// Extract boundary edges: keep triangle edges whose both endpoints are in
// `is_bnode.masks[1]` (multi-material lines) OR `is_bnode.masks[2]`
// (fixed cell-centre points). Mirrors `brep_processing.getBoundaryEdges`.
BEdges get_boundary_edges(const Triangles& tri,
                          const marching_cubes::NodeTypeMask& is_bnode);

// Order boundary edges into chains. Mirrors `brep_processing.orderBedges`.
// Each connected component yields one chain:
//   * 2 fixed endpoints → open chain between them.
//   * 1 fixed endpoint  → cycle back to that endpoint (loop).
//   * 0 fixed endpoints → cycle along the loop, anchored at min vertex.
BrepOrdered order_bedges(const BEdges& bedges,
                         const marching_cubes::NodeTypeMask& is_bnode);

// Save all chains as a single .ply edge dump. Mirrors `saveBrepPLY`.
void save_brep_ply(const BrepOrdered& ordered,
                   const Coords& xyz,
                   const std::string& base,
                   const std::array<std::uint8_t, 3>& rgb);

}  // namespace vox2tet::brep
