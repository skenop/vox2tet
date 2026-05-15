// Sliver-repair pass (Layer 4 in the dihedral-guard / quality proposal).
//
// After the remesh loop converges, this pass walks every triangle whose
// smallest interior corner angle falls below `min_corner_angle_internal`
// (or `_boundary` for triangles whose three vertices are all fixed) and
// tries to widen that corner by relocating the worst-angle vertex
// toward the midpoint of the opposite edge.
//
// Geometric rationale. For a "needle" sliver (one angle very small, two
// large) the small-angle vertex sits at the apex of a tall thin
// triangle. Moving that vertex toward the midpoint of the opposite
// edge widens its corner monotonically — the angle at the apex
// approaches 90° as the apex approaches the edge. The other two
// corners decrease, but in needle-shaped slivers they start near 90°
// and have plenty of room to drop without becoming new slivers.
//
// Each candidate move is accepted iff:
//   (a) the target sliver's worst-corner angle strictly increases;
//   (b) no other corner angle around the relocated vertex's incident
//       triangles drops below the threshold and below its prior value
//       (mirrors the dihedral-guard pattern);
//   (c) no dihedral around the relocated vertex's incident edges drops
//       below `min_dangle_internal` and below its prior value;
//   (d) the relocated vertex stays within `max_drift` of `xyz_init`.
//
// The pass preserves topology entirely — no new/removed vertices,
// edges, or triangles. Cap-type slivers (two corners small, one near
// 180°) cannot be removed by vertex relocation alone; those remain.
// The number of iterations is capped by `max_passes`; early exit on
// no-progress.

#include "vox2tet/remesh/remesh.hpp"

#include "vox2tet/core/log.hpp"
#include "vox2tet/mesh/half_edge.hpp"
#include "vox2tet/remesh/smooth.hpp"

#include <Eigen/Geometry>
#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace vox2tet::remesh {

namespace {

constexpr u32 kInvalidU32 = std::numeric_limits<u32>::max();

// Pair-wise geometric dihedral around the canonical edge of half-edge h
// between h's triangle and hp's triangle. 180° = flat, 0° = full fold.
// (Identical helper to dangle_repair.cpp's `dihedral_pair`.)
double dihedral_pair(const mesh::HalfEdges& he, const Coords& xyz,
                     u32 h, u32 hp) {
    if (hp == kInvalidU32) return -1.0;
    const Eigen::Vector3d p01 = xyz.row(he(he(h,  2), 0)).transpose();
    const Eigen::Vector3d p02 = xyz.row(he(h,         0)).transpose();
    const Eigen::Vector3d p1  = xyz.row(he(he(h,  1), 0)).transpose();
    const Eigen::Vector3d p2  = xyz.row(he(he(hp, 1), 0)).transpose();
    const Eigen::Vector3d v0  = p02 - p01;
    const Eigen::Vector3d c1  = (p1 - p01).cross(v0);
    const Eigen::Vector3d c2  = (p2 - p01).cross(v0);
    const double n12 = c1.norm() * c2.norm();
    if (n12 < 1e-18) return -1.0;
    return std::acos(std::clamp(c1.dot(c2) / n12, -1.0, 1.0)) * 180.0 / M_PI;
}

// Min dihedral across the blink-cycle partners of h (cycle-min metric,
// same as smooth.cpp::calc_min_dihedral_cycle for a single index).
double min_cycle_dihedral(const mesh::HalfEdges& he, const Coords& xyz,
                          u32 h, const std::vector<u32>& mates) {
    if (mates.empty()) {
        const u32 hp = he(h, 3);
        return (hp == kInvalidU32) ? 180.0 : dihedral_pair(he, xyz, h, hp);
    }
    double mn = 180.0;
    for (u32 hp : mates) {
        const double d = dihedral_pair(he, xyz, h, hp);
        if (d >= 0 && d < mn) mn = d;
    }
    return mn;
}

// Iterate all triangles via the half-edge structure. For each triangle
// (three half-edges related by the next/prev pointers) return one
// "representative" half-edge — the smallest index of the three.
std::vector<u32> triangle_heads(const mesh::HalfEdges& he) {
    std::vector<std::uint8_t> visited(static_cast<std::size_t>(he.rows()), 0);
    std::vector<u32> heads;
    heads.reserve(static_cast<std::size_t>(he.rows() / 3));
    for (Eigen::Index ih = 0; ih < he.rows(); ++ih) {
        const u32 h0 = static_cast<u32>(ih);
        if (visited[h0]) continue;
        const u32 h1 = he(h0, 1);
        const u32 h2 = he(h0, 2);
        if (h1 >= visited.size() || h2 >= visited.size()) continue;
        visited[h0] = visited[h1] = visited[h2] = 1;
        heads.push_back(std::min({h0, h1, h2}));
    }
    return heads;
}

// Build vertex → incident half-edges map (same convention as in
// dangle_repair.cpp).
std::vector<std::vector<u32>> build_v2he(const mesh::HalfEdges& he) {
    std::vector<std::vector<u32>> v2h(0);
    Eigen::Index max_v = -1;
    for (Eigen::Index i = 0; i < he.rows(); ++i)
        max_v = std::max<Eigen::Index>(max_v, he(i, 0));
    v2h.assign(static_cast<std::size_t>(max_v + 1), {});
    for (Eigen::Index i = 0; i < he.rows(); ++i) {
        const u32 t = he(i, 0);
        v2h[static_cast<std::size_t>(t)].push_back(static_cast<u32>(i));
    }
    return v2h;
}

// For a triangle represented by its "head" half-edge (smallest of the
// three), return (vertices, smallest-corner index 0/1/2, smallest angle).
struct TriCorner {
    u32   verts[3];          // (vi0, vi1, vi2) — in next-pointer order
    u32   hs[3];             // (h0, h1, h2) where h1=he(h0,1), h2=he(h0,2)
    int   worst_i;           // 0/1/2 index into verts/hs
    double worst_angle;
    double min_angle;        // == worst_angle (alias for readability)
};
TriCorner inspect_triangle(const mesh::HalfEdges& he, const Coords& xyz, u32 h0) {
    TriCorner T{};
    T.hs[0] = h0;
    T.hs[1] = he(h0, 1);
    T.hs[2] = he(h0, 2);
    T.verts[0] = he(T.hs[0], 0);
    T.verts[1] = he(T.hs[1], 0);
    T.verts[2] = he(T.hs[2], 0);

    auto corner = [&](u32 vc, u32 va, u32 vb) {
        const Eigen::Vector3d C = xyz.row(vc).transpose();
        const Eigen::Vector3d u1 = xyz.row(va).transpose() - C;
        const Eigen::Vector3d u2 = xyz.row(vb).transpose() - C;
        const double n1 = u1.norm(), n2 = u2.norm();
        if (n1 < 1e-18 || n2 < 1e-18) return 180.0;
        const double c = std::clamp(u1.dot(u2) / (n1 * n2), -1.0, 1.0);
        return std::acos(c) * 180.0 / M_PI;
    };
    const double a0 = corner(T.verts[0], T.verts[2], T.verts[1]);
    const double a1 = corner(T.verts[1], T.verts[0], T.verts[2]);
    const double a2 = corner(T.verts[2], T.verts[1], T.verts[0]);

    T.worst_i = 0;
    T.worst_angle = a0;
    if (a1 < T.worst_angle) { T.worst_i = 1; T.worst_angle = a1; }
    if (a2 < T.worst_angle) { T.worst_i = 2; T.worst_angle = a2; }
    T.min_angle = T.worst_angle;
    return T;
}

// Min corner angle of any triangle incident to vertex v.
double min_corner_at_vertex(const mesh::HalfEdges& he, const Coords& xyz,
                            u32 v, const std::vector<u32>& v2he_v) {
    // Each incident half-edge h with he(h, 0) == v belongs to one
    // triangle. We use h as the "previous" half-edge for the vertex
    // contribution at v in that triangle: the corner at v is between
    // edges (prev.source → v) and (v → next.target). Just compute the
    // triangle's min corner — we'll take the min over all incident
    // triangles, so per-vertex bookkeeping isn't necessary.
    std::vector<std::uint8_t> seen(static_cast<std::size_t>(he.rows()), 0);
    double mn = 180.0;
    for (u32 h : v2he_v) {
        // Find the triangle's head (smallest of (h, h.next, h.prev)).
        const u32 h1 = he(h, 1);
        const u32 h2 = he(h, 2);
        const u32 head = std::min({h, h1, h2});
        if (seen[head]) continue;
        seen[head] = 1;
        const TriCorner T = inspect_triangle(he, xyz, head);
        if (T.min_angle < mn) mn = T.min_angle;
    }
    return mn;
}

}  // namespace

// ---------------------------------------------------------------------------
SliverRepairStats sliver_repair(const Settings& s,
                                RemeshState& st,
                                const Coords& xyz_init,
                                double max_drift,
                                int max_passes) {
    SliverRepairStats R{0, 0, 0, 0, 0, 0, 0};
    if (max_passes <= 0) return R;

    const auto& he   = st.hedges;
    auto&       xyz  = st.xyz;
    const auto& nfv  = st.not_fixed_v;

    // Build blinks + cycle_mates once (re-used across all passes — the
    // half-edge topology doesn't change in this pass).
    auto blinks = mesh::create_blinks(he, xyz);
    auto cycle_mates = build_cycle_mates(he, blinks);
    auto v2he = build_v2he(he);

    // Bookkeeping.
    const double thr_int = s.min_corner_angle_internal;
    const double thr_bnd = s.min_corner_angle_boundary;
    const double dih_thr = s.min_dangle_internal;

    auto triangle_threshold = [&](const TriCorner& T) {
        // If any of the three triangle vertices is movable, use the
        // internal threshold; if ALL three are fixed (bbox-only
        // triangle), use the boundary threshold.
        if (T.verts[0] < nfv.size() && nfv[T.verts[0]]) return thr_int;
        if (T.verts[1] < nfv.size() && nfv[T.verts[1]]) return thr_int;
        if (T.verts[2] < nfv.size() && nfv[T.verts[2]]) return thr_int;
        return thr_bnd;
    };

    // Initial bad-triangle count (worst-corner < threshold).
    auto heads = triangle_heads(he);
    for (u32 h : heads) {
        const TriCorner T = inspect_triangle(he, xyz, h);
        if (T.min_angle < triangle_threshold(T)) ++R.bad_triangles_before;
    }

    double dmin_seen = 180.0;
    for (u32 h : heads) {
        const TriCorner T = inspect_triangle(he, xyz, h);
        if (T.min_angle < dmin_seen) dmin_seen = T.min_angle;
    }
    VOX2TET_LOG() << "sliver_repair: min corner angle = " << dmin_seen
                  << "°  (internal thr " << thr_int
                  << "°, boundary thr " << thr_bnd
                  << "°, bad triangles " << R.bad_triangles_before << ")";

    // Step fraction toward the midpoint of the opposite edge. 0.5 is
    // an aggressive single move; we damp to 0.3 to keep the apex move
    // small enough that neighbour dihedrals rarely flip.
    const double step_frac = 0.3;

    for (int pass = 0; pass < max_passes; ++pass) {
        R.passes_run = pass + 1;

        // Collect candidates this pass — re-evaluate every triangle.
        struct Cand { double ang; u32 head; };
        std::vector<Cand> cands;
        for (u32 h : heads) {
            const TriCorner T = inspect_triangle(he, xyz, h);
            const double thr = triangle_threshold(T);
            if (T.min_angle < thr) cands.push_back({T.min_angle, h});
        }
        std::sort(cands.begin(), cands.end(),
                  [](const Cand& a, const Cand& b) { return a.ang < b.ang; });

        std::size_t accepted_this_pass = 0;
        for (const auto& c : cands) {
            const TriCorner T = inspect_triangle(he, xyz, c.head);
            // Pick the relocatable vertex. Prefer the worst-corner
            // vertex; if it's fixed, try the others by ascending angle.
            int order[3] = {T.worst_i, 0, 0};
            // Fill remaining 0/1/2 indices in some order:
            {
                bool used[3] = {false, false, false};
                used[T.worst_i] = true;
                int j = 1;
                for (int k = 0; k < 3; ++k) if (!used[k]) order[j++] = k;
            }

            bool moved = false;
            for (int oi = 0; oi < 3 && !moved; ++oi) {
                const int idx_worst = order[oi];
                const u32 v_apex  = T.verts[idx_worst];
                const u32 v_oppA  = T.verts[(idx_worst + 1) % 3];
                const u32 v_oppB  = T.verts[(idx_worst + 2) % 3];
                if (v_apex >= nfv.size() || !nfv[v_apex]) continue;

                ++R.flip_attempts;  // counter overloaded: total attempts

                // Candidate position: move apex toward midpoint of
                // opposite edge by step_frac.
                const Eigen::Vector3d C   = xyz.row(v_apex).transpose();
                const Eigen::Vector3d M   = 0.5 * (xyz.row(v_oppA).transpose()
                                                 + xyz.row(v_oppB).transpose());
                const Eigen::Vector3d cand = C + step_frac * (M - C);

                // Drift cap.
                const Eigen::Vector3d delta = cand - xyz_init.row(v_apex).transpose();
                if (delta.norm() > max_drift) continue;

                // Snapshot apex.
                const Eigen::RowVector3d old_pos = xyz.row(v_apex);

                // Snapshot current quality around apex:
                //   - min corner angle in each incident triangle;
                //   - min dihedral in each incident edge (cycle-min).
                if (v_apex >= v2he.size()) continue;
                const auto& adj_h_all = v2he[v_apex];

                // Per-triangle min corner snapshot.
                std::vector<u32> tri_heads_around;
                std::vector<double> tri_min_before;
                {
                    std::vector<std::uint8_t> seen(static_cast<std::size_t>(he.rows()), 0);
                    for (u32 h : adj_h_all) {
                        const u32 h1 = he(h, 1), h2 = he(h, 2);
                        const u32 head = std::min({h, h1, h2});
                        if (seen[head]) continue;
                        seen[head] = 1;
                        tri_heads_around.push_back(head);
                        tri_min_before.push_back(
                            inspect_triangle(he, xyz, head).min_angle);
                    }
                }

                // Per-edge min dihedral snapshot (cycle-min).
                std::vector<double> dih_before;
                dih_before.reserve(adj_h_all.size());
                for (u32 h : adj_h_all) {
                    dih_before.push_back(min_cycle_dihedral(
                        he, xyz, h,
                        cycle_mates[static_cast<std::size_t>(h)]));
                }

                // Apply candidate move.
                xyz.row(v_apex) = cand.transpose();

                // Did the target sliver's worst corner strictly improve?
                const TriCorner T_after = inspect_triangle(he, xyz, c.head);
                bool accept = (T_after.min_angle > c.ang + 0.5);   // ≥ 0.5° gain

                if (accept) {
                    // Per-triangle corner-angle regression check.
                    for (std::size_t k = 0; k < tri_heads_around.size(); ++k) {
                        const u32 head = tri_heads_around[k];
                        const TriCorner Tn = inspect_triangle(he, xyz, head);
                        const double thr = triangle_threshold(Tn);
                        if (Tn.min_angle < thr &&
                            Tn.min_angle < tri_min_before[k] - 1e-6) {
                            accept = false;
                            break;
                        }
                    }
                }

                if (accept) {
                    // Per-edge dihedral regression check (cycle-min).
                    for (std::size_t k = 0; k < adj_h_all.size(); ++k) {
                        const u32 h = adj_h_all[k];
                        const double d_now = min_cycle_dihedral(
                            he, xyz, h,
                            cycle_mates[static_cast<std::size_t>(h)]);
                        if (d_now < dih_thr && d_now < dih_before[k] - 1e-6) {
                            accept = false;
                            break;
                        }
                    }
                }

                if (accept) {
                    ++R.flip_accepted;   // overloaded: accepted moves
                    ++accepted_this_pass;
                    moved = true;
                } else {
                    xyz.row(v_apex) = old_pos;
                }
            }
        }

        if (accepted_this_pass == 0) break;
    }

    // Final bad-triangle count.
    for (u32 h : heads) {
        const TriCorner T = inspect_triangle(he, xyz, h);
        if (T.min_angle < triangle_threshold(T)) ++R.bad_triangles_after;
    }
    return R;
}

}  // namespace vox2tet::remesh
