#pragma once

// Multi-material marching cubes core. Combines cell topology constants
// with the loop-building and triangulation helpers used by `contouring`.
//
// Conventions:
//   * Cell corners 0..7 use a (z, y, x) triple with y as the slow axis
//     in each face.
//   * MC nodes live in [0, 19] — 12 edge midpoints, 6 facet centres,
//     1 cell centre, and 19 as a "dummy / no triangle" sentinel.
//   * "Cell code" (uint8) is the 4-color-cell pattern code: 8 corner
//     ranks compressed to a 4^4 facet index per face.

#include <array>
#include <cstdint>
#include <vector>

#include "vox2tet/marching_cubes/lut.hpp"  // View<T>

namespace vox2tet::marching_cubes {

// --- topology ---------------------------------------------------------------
const std::array<std::array<std::uint8_t, 4>, 6>&  cell_facet();
const std::array<std::array<std::uint8_t, 6>, 6>&  edges_map();
const std::array<std::array<std::uint8_t, 4>, 16>& lut2d_2materials();
const std::array<std::array<std::int8_t, 3>, 20>&  cell_mid_edge_coords();
std::array<std::array<std::uint8_t, 3>, 20>        cell_mid_edge_coords1();
const std::array<std::array<std::int8_t, 3>, 8>&   cell_facet_coords();

// "MC node opposite within the cell" table used by triangulateLoops*.
// Values 0..11 → diagonally opposite edge midpoint; 12..19 → dummy (19).
std::uint8_t opposite_edge(std::uint8_t mc_node);

// --- code2DArray / decode helpers -------------------------------------------
//
// code2DArray(arr, base) packs each row of `arr` (length `width`) into a
// scalar using `base` as the digit base, with arr[0] as the most
// significant digit. decode reverses it.
template <typename Out>
inline Out code_row(const std::uint8_t* row, int width, int base) {
    Out v = 0;
    Out mul = 1;
    for (int k = 0; k < width; ++k) mul *= static_cast<Out>(base);
    for (int k = 0; k < width; ++k) {
        mul /= static_cast<Out>(base);
        v += static_cast<Out>(row[k]) * mul;
    }
    return v;
}

// --- edges/triangles in a cell ---------------------------------------------
//
// edges_2materials_cell(id) returns 24 MC node IDs forming 12 directed
// half-edges in a 2-material cell with pattern `id` (0..255). Entries
// (edges[2k], edges[2k+1]) form half-edge k; 19 marks "no edge".
std::array<std::uint8_t, 24> edges_2materials_cell(std::uint8_t id);

// edges2loopsNoCenteredFacet: chain half-edges into closed loops. The
// input is a flat sequence of (u, v) pairs (size = 2*M). Returns a list
// of loops; each loop is the sequence of MC node ids forming a closed
// chain (loop[0] == loop[end] is *not* repeated).
std::vector<std::vector<std::uint8_t>>
edges2loops_no_centered(const std::uint8_t* edges, std::size_t n_edges);

// triangulateLoopsNoCentered: fan-triangulate each loop (3..7 vertices).
// Returns a flat triangle list (3 node-ids per triangle). Empty input
// yields {19,19,19} (the canonical "empty triangle" sentinel).
std::vector<std::uint8_t>
triangulate_loops_no_centered(const std::vector<std::vector<std::uint8_t>>& loops);

// LUT for 2-material cells (256 rows × 18 cols), value 19 = dummy.
// Materialised on first call; subsequent calls return a reference to the
// cached table.
View<std::uint8_t> lut2materials();

}  // namespace vox2tet::marching_cubes
