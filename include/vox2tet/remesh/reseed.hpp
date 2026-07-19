#pragma once

// Boundary-edge reseeding — Stages A and A2 of the feature-chain
// coarsening proposal (see doc/RESEEDING.md).
//
// The initial mesh samples every feature chain (triple lines, bbox
// frame, grain-boundary traces on the bbox faces) at voxel spacing and
// the remesh loop never changes that: chain vertices are frozen, so the
// element count near features stays proportional to the voxel size.
//
// This module coarsens chains by *guarded edge collapse*: a chain
// vertex is merged into its along-chain neighbour only when an explicit
// set of topological, geometric, and quality guards all pass. A
// rejected collapse simply leaves the mesh as it was — the worst case
// is less coarsening, never a broken mesh.
//
// Stage A  — the 12 bounding-box frame lines (exactly straight,
//            manifold edges).
// Stage A2 — the curved chains: grain-boundary traces on the bbox
//            faces (edge multiplicity 3) and internal triple / quad
//            lines (multiplicity 3–4). Curvature adaptivity comes from
//            the chordal-deviation guard (`Settings::reseed_eps`), and
//            a global AABB segment-intersection guard rejects any
//            chord that would stab non-incident geometry.
//
// Preserved exactly: the 8 bbox corners and every junction node where
// chains meet (`order_bedges` splits chains at fixed nodes, so
// junctions are chain endpoints and never candidates). Closed-loop
// chains (no fixed node) are coarsened down to at most 4 surviving
// vertices, never further.

#include <array>
#include <cstddef>
#include <vector>

#include <Eigen/Core>

#include "vox2tet/brep/brep.hpp"
#include "vox2tet/core/settings.hpp"
#include "vox2tet/core/types.hpp"
#include "vox2tet/marching_cubes/contouring.hpp"

namespace vox2tet::remesh {

struct ReseedStats {
    std::size_t frame_chains        = 0;  // straight bbox-frame chains
    std::size_t curved_chains       = 0;  // open trace / triple-line chains
    std::size_t loop_chains         = 0;  // closed chains (no fixed node)
    std::size_t verts_before        = 0;
    std::size_t verts_removed       = 0;
    std::size_t tris_removed        = 0;
    std::size_t collapses_attempted = 0;
    std::size_t collapses_accepted  = 0;
    std::size_t rej_topology        = 0;  // non-manifold side edge / link condition
    std::size_t rej_geometry        = 0;  // chordal deviation over reseed_eps
    std::size_t rej_quality         = 0;  // corner angle / normal flip / chord dihedral
    std::size_t rej_intersection    = 0;  // AABB guard: chord stabs distant geometry
    std::size_t lfs_limited         = 0;  // Stage B: spans capped below target by lfs
};

// Reseed the feature chains of a triangle mesh by guarded edge
// collapse. Scope: bbox-frame chains always; trace / triple-line
// chains when `s.do_reseed_triple_lines` is set. All arguments are
// updated in place:
//   * `tri`        — rebuilt (grouped by interface, like the input);
//   * `xyz`        — dead vertex rows removed (surviving rows keep their
//                    coordinates bit-exactly — collapse never moves a vertex);
//   * `node_mask`  — all 8 mask rows compacted to the new vertex set;
//   * `interfaces` — (first, count) ranges updated for the new `tri`.
//
// The caller must rebuild any derived state afterwards (half-edges,
// not-fixed masks, normals, sizing) — see the pipeline integration.
//
// Guards, per candidate collapse of chain vertex v into anchor u
// (w = v's next chain neighbour, so the new chord is u–w):
//   topology     — the chain edge (u,v) has multiplicity 2–4 (one
//                  triangle per incident sheet); every side edge of
//                  every dying triangle is manifold with clean mutual
//                  opposites; link condition: the common neighbours of
//                  u and v are exactly the dying triangles' apexes;
//   geometry     — |u-w| <= target spacing (reseed_target_len, or Lmax
//                  when 0); every vertex removed on the current chord
//                  stays within reseed_eps of segment u–w (this is the
//                  curvature control: tight bends refuse long chords);
//   quality      — every surviving triangle around v, re-evaluated with
//                  v at u's position: positive area, unflipped normal,
//                  min corner angle above the bbox/internal threshold
//                  unless already below; the fold (pairwise dihedral)
//                  across the new chord must not drop below the
//                  matching min_dangle threshold unless already below;
//   intersection — the chord u–w must not strictly cross any live
//                  triangle that does not contain u, v, or w (AABB
//                  tree query; coplanar crossings on the planar bbox
//                  faces are instead prevented by the normal-flip guard).
ReseedStats reseed_feature_chains(const Settings& s,
                                  Triangles& tri,
                                  Coords& xyz,
                                  marching_cubes::NodeTypeMask& node_mask,
                                  std::vector<marching_cubes::Interface>& interfaces);

// ---------------------------------------------------------------------------
// Implementation detail, exposed for unit testing.
namespace reseed_detail {

// Static axis-aligned bounding-box tree over a fixed set of items
// (triangles). Vertex positions never move during reseeding — only
// connectivity changes — so boxes only need *grow-only* refits when a
// triangle is retargeted to a new vertex.
struct AabbTree {
    struct Node {
        Eigen::Vector3d  lo, hi;
        int              left = -1, right = -1, parent = -1;
        std::vector<int> items;   // non-empty on leaves only
    };
    std::vector<Node> nodes;
    std::vector<int>  leaf_of;    // item id → leaf node index
    int               root = -1;

    // Build from per-item boxes (lo, hi). Median split on the longest
    // centroid axis, leaf bucket of 4.
    void build(const std::vector<std::array<Eigen::Vector3d, 2>>& boxes);

    // Grow-only refit: union the item's new box into its leaf and all
    // ancestors. Never shrinks — stale slack is conservative.
    void grow_item(int item, const Eigen::Vector3d& lo,
                   const Eigen::Vector3d& hi);

    // Invoke f(item) for every item whose leaf box the segment [a, b]
    // touches (slab test, small tolerance). Exact intersection is the
    // caller's job.
    template <typename F>
    void for_segment(const Eigen::Vector3d& a, const Eigen::Vector3d& b,
                     F&& f) const {
        if (root < 0) return;
        std::vector<int> stack{root};
        while (!stack.empty()) {
            const int ni = stack.back();
            stack.pop_back();
            const Node& n = nodes[static_cast<std::size_t>(ni)];
            if (!segment_touches_box(a, b, n.lo, n.hi)) continue;
            if (n.left < 0) {
                for (int it : n.items) f(it);
            } else {
                stack.push_back(n.left);
                stack.push_back(n.right);
            }
        }
    }

    static bool segment_touches_box(const Eigen::Vector3d& a,
                                    const Eigen::Vector3d& b,
                                    const Eigen::Vector3d& lo,
                                    const Eigen::Vector3d& hi);

private:
    int build_range(std::vector<int>& items, std::size_t first,
                    std::size_t last,
                    const std::vector<std::array<Eigen::Vector3d, 2>>& boxes,
                    int parent);
};

// Strict-interior segment–triangle intersection (Möller–Trumbore).
// Returns false for parallel / coplanar configurations — on the planar
// bbox faces those are guarded by the normal-flip check instead — and
// for touches at the segment endpoints or the triangle border.
bool segment_hits_triangle(const Eigen::Vector3d& a, const Eigen::Vector3d& b,
                           const Eigen::Vector3d& p0, const Eigen::Vector3d& p1,
                           const Eigen::Vector3d& p2);

// Stage B: per-chain-vertex target spacing from the composite local
// feature size. For every chain (boundary-edge) vertex p:
//
//   lfs(p) = min( d_chains(p), d_self(p), d_surf(p) )
//
// where candidates are excluded by a *geodesic collar*, not by entity
// id: a chain vertex q counts when its distance along the chain graph
// from p exceeds `collar` times the Euclidean distance |p-q| (or q is
// unreachable), and a surface vertex v counts when its graph-geodesic
// distance to the nearest chain vertex exceeds `collar` times |p-v|.
// This way the staircase neighbours and the chain's own surface skirt
// never clamp the spacing, while another arm of the SAME chain — or a
// sheet passing close by without touching — does.
//
// The returned target is h(p) = clamp(beta * lfs(p), h_min, target),
// gradient-limited along the chain graph with slope (grading - 1).
// Non-chain vertices get h = target and lfs = +inf.
struct ChainLfs {
    std::vector<double> h;    // per mesh vertex: allowed chord length
    std::vector<double> lfs;  // raw composite feature size (diagnostic)
};
ChainLfs compute_chain_lfs(const Triangles& tri, const Coords& xyz,
                           const brep::BEdges& bedges,
                           double target, double beta, double grading,
                           double h_min = 1.0, double collar = 2.0);

}  // namespace reseed_detail

}  // namespace vox2tet::remesh
