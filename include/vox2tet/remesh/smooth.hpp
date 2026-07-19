#pragma once

// Surface smoothing primitives. Contains:
//
//   * triangles_to_edges          - Nx2 unique edges from a triangle list
//   * calc_dihedral               - per-half-edge dihedral angle
//   * calc_internal_dihedral      - over `not_fixed_h` half-edges
//   * smooth_surfaces             - Laplacian smoothing of interior verts
//   * calc_initial_vertex_normal  - area-weighted vertex normals
//   * smooth_brep_mean            - 1D smoothing of boundary-edge chains
//   * undo_sharp_bdangle_nodes    - revert near sharp boundary dihedrals
//   * smooth_surfaces_dangle_ctrl - Laplacian with dihedral-angle guard
//   * smooth_laplace              - top-level preremesh smoothing driver

#include <Eigen/Core>
#include <vector>

#include "vox2tet/brep/brep.hpp"
#include "vox2tet/core/settings.hpp"
#include "vox2tet/core/types.hpp"
#include "vox2tet/marching_cubes/contouring.hpp"
#include "vox2tet/mesh/half_edge.hpp"

namespace vox2tet::remesh {

// Nx2 unique edges (with v0 < v1) from a triangle list.
using Edges2 = Eigen::Matrix<std::uint32_t, Eigen::Dynamic, 2, Eigen::RowMajor>;
Edges2 triangles_to_edges(const Triangles& tri);

// Per-half-edge dihedral angle around half-edge `i1` (and its opposite
// `i2`). When `directional` is true, returns angles in [0, 360) using a
// determinant sign; otherwise in [0, 180].
Eigen::VectorXd calc_dihedral(const mesh::HalfEdges& he, const Coords& xyz,
                              const std::vector<std::uint32_t>& i1,
                              const std::vector<std::uint32_t>& i2,
                              bool directional = false);

// Convenience: angles for every half-edge in `not_fixed_h` (paired with
// its column-3 opposite).
struct InternalDihedrals {
    Eigen::VectorXd dan;             // length == count(not_fixed_h)
    std::vector<std::uint32_t> i1;   // half-edge indices that produced dan
};
InternalDihedrals calc_internal_dihedral(const mesh::HalfEdges& he,
                                         const std::vector<std::uint8_t>& not_fixed_h,
                                         const Coords& xyz);

// Pre-compute, for every half-edge, the list of OTHER half-edges sharing
// its canonical edge (walk of the blink cycle). For a manifold edge the
// list has exactly one entry (the opposite); for a triple line two
// entries; for a quad point three. Empty for boundary edges.
//
// Used by the cycle-min dihedral helpers below to enforce the
// `min_dangle_*` thresholds across ALL incident triangle pairs at a
// non-manifold edge — not just the canonical pair.
std::vector<std::vector<std::uint32_t>> build_cycle_mates(
    const mesh::HalfEdges& he,
    const std::vector<std::uint32_t>& blinks);

// For each half-edge in `i1`, the minimum pair-wise dihedral angle
// between its triangle and the triangles of all blink-cycle partners.
// On manifold edges this is identical to `calc_dihedral` against the
// canonical opposite; on non-manifold edges it is the dihedral most
// in need of repair (the "worst fold"). Returns 180° for boundary
// half-edges (no partner) so they never trigger a sharp guard.
Eigen::VectorXd calc_min_dihedral_cycle(
    const mesh::HalfEdges& he, const Coords& xyz,
    const std::vector<std::uint32_t>& i1,
    const std::vector<std::vector<std::uint32_t>>& cycle_mates);

// Convenience that builds `i1 = all not_fixed_h indices` internally and
// uses the cycle-min metric. Returns the InternalDihedrals layout.
InternalDihedrals calc_internal_dihedral_cycle(
    const mesh::HalfEdges& he,
    const std::vector<std::uint8_t>& not_fixed_h,
    const Coords& xyz,
    const std::vector<std::vector<std::uint32_t>>& cycle_mates);

// Laplacian smoothing — uses each half-edge (from `triangles_to_edges`)
// as a single undirected connection, in a sparse coo-matrix layout
// (column-0 = src, column-1 = dst, then sum-by-row to normalise).
//
// `xyz` is modified in place. Only rows where `not_fixed[i] != 0` move.
void smooth_surfaces(const Triangles& tri, Coords& xyz,
                     const std::vector<std::uint8_t>& not_fixed,
                     int n_iter, double alpha);

// Area-weighted vertex normals — sum face normals at each internal vertex
// and re-normalise. Boundary / fixed verts get zero (matches the reference).
NormalsMat calc_initial_vertex_normal(const Coords& xyz,
                                      const Triangles& tri,
                                      const std::vector<std::uint8_t>& is_internal);

// Iterative B-rep chain smoothing. Mirrors `brep_processing.smoothBrepMean`.
// Updates each chain's interior nodes (skipping endpoints) by a 1-2-1
// average; distance from initial position is clamped to settings.max_D2self.
// When `brep_sizing` is non-null, only move nodes whose sizing > min_D2other.
void smooth_brep_mean(const Settings& s,
                      const std::vector<brep::EdgeChain>& chains,
                      Coords& xyz,
                      const Coords& xyz0,
                      int n_iter,
                      const Eigen::VectorXd* brep_sizing = nullptr);

// Per-vertex brep sizing: for each brep vertex, distance to its nearest
// other brep vertex; for non-brep vertices, distance to the nearest brep
// vertex. Brute force O(N * |B|) — fine for current JMA test sizes; can
// upgrade to nanoflann later. Mirrors `brep_processing.calc_brep_sizing`.
Eigen::VectorXd calc_brep_sizing(const Coords& xyz,
                                 const std::vector<std::uint8_t>& is_brep);

// For each interface vertex, find the closest vertex on a *different*
// interface that shares at least one material. Mirrors
// `mesh_processing.calc_closest_to_boundary_h2`.
struct ClosestToBoundary {
    Eigen::VectorXd               distances;
    std::vector<std::uint32_t>    closest_idx;
};
ClosestToBoundary calc_closest_to_boundary(
    const mesh::HalfEdges& he, const Coords& xyz,
    const std::vector<marching_cubes::Interface>& interfaces);

// Full per-vertex sizing field. Mirrors `mesh_processing.calc_sizing_field`.
Eigen::VectorXd calc_sizing_field(const mesh::HalfEdges& he,
                                  const Coords& xyz,
                                  const Eigen::VectorXd& brep_sizing,
                                  const std::vector<std::uint8_t>& not_fixed_v,
                                  const std::vector<marching_cubes::Interface>& interfaces,
                                  double Lmin, double Lmax);

// Grading limit on the sizing field (reseeding Stage C): multi-source
// Dijkstra relaxation over the mesh edge graph enforcing
//     L[v] <= L[u] + (grading - 1) * |uv|
// for every edge (u,v). The result is the pointwise-largest field that
// is <= L everywhere and satisfies the bound, so grading only ever
// LOWERS values (it refines the coarse side of a transition, never
// coarsens the fine side). Returns the number of vertices lowered by
// more than `tol`; no-op when grading <= 1.
std::size_t limit_sizing_gradient(const mesh::HalfEdges& he,
                                  const Coords& xyz,
                                  Eigen::VectorXd& L,
                                  double grading,
                                  double tol = 1e-12);

// Revert smoothing locally where boundary dihedral got sharper than
// `min_dangle_boundary`. `blinks` must give the "opposite" (or pair)
// half-edge for every fixed half-edge. When null, we use he(:,3); for
// our manifold-only build this matches the legacy manifold-only path.
void undo_sharp_bdangle_nodes(Coords& xyz, const Coords& xyz0,
                              const mesh::HalfEdges& he,
                              const std::vector<std::uint8_t>& not_fixed_h,
                              double min_dangle_boundary,
                              const std::vector<std::uint32_t>* blinks = nullptr);

// Per-triangle minimum interior corner angle (degrees). Returns
// length == he.rows()/3 if the half-edges are in stable triangle order,
// otherwise a per-triangle value derived from each half-edge triple.
// Used by the sliver guard / repair passes.
double calc_min_corner_of_triangle(const Coords& xyz,
                                   std::uint32_t va,
                                   std::uint32_t vb,
                                   std::uint32_t vc);

// Layer-2 corner-angle revert. For every triangle whose smallest interior
// corner angle dropped below `min_corner_internal` (or
// `min_corner_boundary` if all 3 vertices are fixed/boundary) AND below
// its prior value, revert the 3 triangle vertices to their `xyz0` state.
// Mirrors `undo_sharp_bdangle_nodes` for corner angles.
void undo_small_corner_nodes(Coords& xyz, const Coords& xyz0,
                             const mesh::HalfEdges& he,
                             const std::vector<std::uint8_t>& not_fixed_v,
                             double min_corner_internal,
                             double min_corner_boundary);

// Laplacian smoothing of interior vertices with dihedral-angle and
// distance-from-original guards. Mirrors `smoothSurfacesDAngleControl`.
// When `blinks` is supplied, the dihedral-acceptance check uses the
// MINIMUM dihedral across all blink-cycle partners of every not_fixed_h
// half-edge — catching folds at triple lines / quad points where the
// canonical column-3 opposite covers only one of the 2-3 pairs.
void smooth_surfaces_dangle_ctrl(const Settings& s,
                                 const mesh::HalfEdges& he,
                                 const std::vector<std::uint8_t>& not_fixed_h,
                                 Coords& xyz,
                                 const Coords& xyz0,
                                 const std::vector<std::uint8_t>& not_fixed_v,
                                 int n_max_iter,
                                 const std::vector<std::uint32_t>* blinks = nullptr);

// Top-level smoothing driver. Returns the per-vertex brep sizing field.
// `path_base` (when non-empty) drives the `_E.ply` / `_S_E.ply` writes.
Eigen::VectorXd smooth_laplace(const Settings& s,
                               const Triangles& tri,
                               const marching_cubes::NodeTypeMask& is_bnode,
                               const mesh::HalfEdges& he,
                               const std::vector<std::uint8_t>& not_fixed_h,
                               Coords& xyz,
                               const std::vector<std::uint8_t>& not_fixed_v,
                               const std::string& path_base);

}  // namespace vox2tet::remesh
