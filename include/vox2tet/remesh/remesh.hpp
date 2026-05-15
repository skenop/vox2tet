#pragma once

// Surface remeshing — split / flip / (collapse) operations and the
// top-level remesh driver. Port of mesh_processing.split_edges,
// flipHalfEdge, smoothSurfacesTangential and remesh.
//
// Limitations vs the reference implementation:
//   * `collapse_edges` is stubbed for now (it relies on the non-manifold
//     boundary-link table from `create_boundary_hedges_links`, which is a
//     follow-up).
//   * `flipHalfEdge` uses the manifold half-edge structure as built by
//     `triangles_to_hedges` — sufficient for the interior; boundary flips
//     are skipped.
//   * `smoothSurfacesTangential` is fully ported.

#include <Eigen/Core>
#include <string>
#include <vector>

#include "vox2tet/core/settings.hpp"
#include "vox2tet/core/types.hpp"
#include "vox2tet/marching_cubes/contouring.hpp"
#include "vox2tet/mesh/half_edge.hpp"

namespace vox2tet::remesh {

// In/out remesh state — bundle of half-edges, masks, coords, normals and
// sizing field. Operations grow / shrink the mesh by re-allocating these.
struct RemeshState {
    mesh::HalfEdges                 hedges;
    std::vector<std::uint8_t>       not_fixed_h;
    Coords                          xyz;
    std::vector<std::uint8_t>       not_fixed_v;
    NormalsMat                      normals;
    Eigen::VectorXd                 sizing;
};

// Tangential Laplacian smoothing — projects each Laplacian step onto the
// tangent plane defined by `normals` and reverts where the internal
// dihedral got sharper than `min_dangle_internal` (or dropped by >2×).
void smooth_surfaces_tangential(const mesh::HalfEdges& he,
                                const std::vector<std::uint8_t>& not_fixed_h,
                                Coords& xyz,
                                const std::vector<std::uint8_t>& not_fixed_v,
                                const NormalsMat& normals,
                                int n_smooth_iter,
                                double min_dangle_internal);

// Split every not-fixed half-edge whose length exceeds 4/3 × min(sizing).
// Each split adds 1 vertex and 6 half-edges. xyz / normals / sizing /
// not_fixed_v / not_fixed_h grow. Mirrors `mesh_processing.split_edges`.
void split_edges(RemeshState& st);

// Edge flips. Iterates not-fixed half-edges sorted by triangle shape
// factor and swaps the underlying quad's diagonal when the shape factor
// improves AND new dihedral angles don't worsen below thresholds.
void flip_half_edge(const Settings& s,
                    mesh::HalfEdges& he,
                    const std::vector<std::uint8_t>& not_fixed_h,
                    Coords& xyz,
                    const std::vector<std::uint8_t>& not_fixed_v,
                    bool do_not_boundary = false,
                    bool do_only_boundary = false);

// Stub for now — full collapse requires the non-manifold blink table.
// Returns the state unchanged. Documented as a TODO in PROGRESS.md.
void collapse_edges(const Settings& s, RemeshState& st,
                    bool do_not_boundary, bool do_only_boundary);

// Top-level driver. Iterates `n_remesh_itr` rounds of split → collapse →
// flip → tangential smoothing. Returns the final triangle list (rebuilt
// from the half-edge structure).
struct RemeshResult {
    std::string  surface_path;
    Triangles    triangles;
    Coords       xyz;
};
RemeshResult remesh(const Settings& s,
                    RemeshState& st,
                    std::vector<marching_cubes::Interface>& interfaces);

// ---------------------------------------------------------------------------
// Dihedral repair (Layer 3 in the dihedral-guard proposal).
//
// Visits every non-manifold edge whose minimum pair-wise dihedral is
// below `min_dangle_threshold` and attempts to widen the fold by moving
// the apex (third vertex) of one of the offending triangles in the
// direction perpendicular to the edge, away from the plane of the other
// triangle. Each move is accepted only if:
//   * the offending dihedral strictly increases;
//   * no dihedral around the moved apex's incident edges drops below
//     the threshold or below its previous value;
//   * the moved apex stays within `max_drift` of its position in
//     `xyz_init` (defaults to the same `s.max_D2self` used by smoothing).
//
// Repeats for at most `max_passes` outer sweeps; stops earlier if a pass
// accepts zero moves. Operates in place on `xyz` and reports how many
// sub-threshold edges remain when it returns.
// ---------------------------------------------------------------------------
struct DangleRepairStats {
    std::size_t bad_edges_before;
    std::size_t bad_edges_after;
    std::size_t attempted_moves;
    std::size_t accepted_moves;
    int         passes_run;
};
DangleRepairStats dangle_repair(const Settings& s,
                                const mesh::HalfEdges& he,
                                Coords& xyz,
                                const Coords& xyz_init,
                                const std::vector<std::uint8_t>& not_fixed_v,
                                double min_dangle_threshold,
                                double max_drift,
                                int max_passes);

// ---------------------------------------------------------------------------
// Sliver repair (Layer 4 — corner-angle remediation).
//
// After the remesh loop, walks every triangle whose smallest interior
// corner angle falls below the matching threshold
// (`s.min_corner_angle_internal`, or `_boundary` for triangles whose
// three vertices are all fixed) and tries to remove the sliver with
// one of two operations applied in order:
//
//   * sliver-targeted edge flip: flip the edge opposite the smallest
//     corner. Accepted iff the worst corner of the two new triangles
//     strictly exceeds the worst corner of the two old triangles, AND
//     every neighbour dihedral around the flipped edge stays above
//     `s.min_dangle_internal`.
//   * sliver-targeted edge collapse: collapse the shortest edge of the
//     remaining sliver. Accepted iff (a) both endpoints are not_fixed_v,
//     (b) no neighbour triangle becomes a new sliver, (c) every
//     neighbour dihedral stays above `s.min_dangle_internal`, (d) the
//     merged point stays within `max_drift` of `xyz_init`.
//
// Topology-changing: each accepted flip swaps the diagonal of a
// manifold triangle pair (no vertex/half-edge count change); each
// accepted collapse removes 2 triangles, 6 half-edges, and 1 vertex.
// The mesh is compacted after the pass.
// ---------------------------------------------------------------------------
struct SliverRepairStats {
    std::size_t bad_triangles_before;
    std::size_t bad_triangles_after;
    std::size_t flip_attempts;
    std::size_t flip_accepted;
    std::size_t collapse_attempts;
    std::size_t collapse_accepted;
    int         passes_run;
};
SliverRepairStats sliver_repair(const Settings& s,
                                RemeshState& st,
                                const Coords& xyz_init,
                                double max_drift,
                                int max_passes);

}  // namespace vox2tet::remesh
