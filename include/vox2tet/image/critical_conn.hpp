#pragma once

// Critical voxel-connectivity detection & fixing — locates and resolves
// the vertex- and edge-ambiguous voxel patterns that would otherwise
// break marching-cubes topology
// (findVertexConnectivity / findEdgeConnectivity /
//  fixVertexCritical / fixEdgeCritical2M).
//
// All routines operate on the extended (boundary-padded) volume.

#include <array>
#include <cstdint>
#include <vector>

#include "vox2tet/image/volume.hpp"

namespace vox2tet::image {

struct VertexCritResult {
    // Linear indices (into the (sz-1)^3 corner-grid) of cells with the
    // diagonal-only pattern, one vector per diagonal (D0..D3).
    std::array<std::vector<std::size_t>, 4> by_diagonal;
    std::size_t                              n_conn = 0;
};

VertexCritResult find_vertex_connectivity(const Volume& vox);

struct EdgeCritResult {
    // For each axial plane (axis 0/1/2 = the constant axis), two vectors
    // of (z, y, x) cell positions: total critical and 2-material-only.
    std::array<std::vector<std::array<std::size_t, 3>>, 3> total_by_axis;
    std::array<std::vector<std::array<std::size_t, 3>>, 3> m2_by_axis;
    std::size_t m2 = 0;
    std::size_t m3 = 0;
};

EdgeCritResult find_edge_connectivity(const Volume& vox);

// In-place fixers. Both return {before_count, after_count}.
std::pair<std::size_t, std::size_t> fix_vertex_critical(Volume& vox);
std::pair<std::size_t, std::size_t> fix_edge_critical_2m(Volume& vox);

}  // namespace vox2tet::image
