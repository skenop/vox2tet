// Dihedral repair pass (Layer 3 in the dihedral-guard proposal).
//
// The Wu-Sullivan multi-material marching cubes output, even after
// Laplacian + tangential smoothing and split/collapse/flip remeshing,
// keeps a long tail of small pair-wise dihedrals at triple lines (edges
// shared by 3 grains) and quad points (4 grains). These folds are NOT
// fixable by the standard remesh operations:
//
//   * `flip_half_edge` only operates on manifold edges (valence 2);
//   * `split_edges` cannot help because the new midpoint inherits the
//     same non-manifold edge configuration;
//   * `collapse_edges` could in principle merge the two endpoints of
//     the offending edge, but doing so on a triple line is risky — it
//     re-attaches three or four triangles at the surviving vertex and
//     usually trades the fold for an even worse one elsewhere.
//
// Targeted apex relocation. For each undirected edge with min-pair-wise
// dihedral below the threshold, identify the two triangles forming the
// worst pair and try moving the apex (third vertex) of one of them in
// the direction perpendicular to the edge, off the plane of the other
// triangle. Accept the move iff:
//   (a) the offending dihedral strictly increases;
//   (b) no other dihedral around the apex's incident edges drops below
//       the threshold or below its prior value;
//   (c) the apex stays within `max_drift` of its xyz_init position.
//
// The motion preserves the mesh topology (no new vertices, edges, or
// triangles) and operates only on movable (not_fixed_v) apex vertices.
// A fixed apex (boundary / corner) is skipped — the fold there is
// genuinely irreducible.

#include "vox2tet/remesh/remesh.hpp"

#include "vox2tet/core/log.hpp"
#include "vox2tet/mesh/half_edge.hpp"
#include "vox2tet/mesh/mathx.hpp"
#include "vox2tet/remesh/smooth.hpp"

#include <Eigen/Geometry>
#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace vox2tet::remesh {

namespace {

constexpr u32 kInvalidU32 = std::numeric_limits<u32>::max();

// Pair-wise dihedral around the canonical edge of half-edge h between
// h's triangle and hp's triangle (same convention as smooth.cpp's helper
// — 180° = flat, 0° = full fold).
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

// Compute a candidate move for the apex of h's triangle that increases
// the dihedral between (h's triangle) and (hp's triangle). The apex
// is moved along the unit normal of hp's triangle, with a sign chosen
// to push it AWAY from hp's plane, by a small fraction of the edge
// length (so the geometry can't blow up).
//
// Returns the new apex coordinates. If the move can't be computed
// (degenerate triangles), returns the current apex coordinates.
Eigen::Vector3d compute_apex_lift(const mesh::HalfEdges& he, const Coords& xyz,
                                  u32 h, u32 hp, double step_frac) {
    const u32 v_a   = he(he(h,  2), 0);   // edge endpoint A
    const u32 v_b   = he(h,         0);   // edge endpoint B
    const u32 v_c   = he(he(h,  1), 0);   // apex of h's triangle (C)
    const u32 v_d   = he(he(hp, 1), 0);   // apex of hp's triangle (D)

    const Eigen::Vector3d A = xyz.row(v_a).transpose();
    const Eigen::Vector3d B = xyz.row(v_b).transpose();
    const Eigen::Vector3d C = xyz.row(v_c).transpose();
    const Eigen::Vector3d D = xyz.row(v_d).transpose();

    const Eigen::Vector3d ab = B - A;
    const Eigen::Vector3d ad = D - A;
    Eigen::Vector3d n_d = ab.cross(ad);     // normal of triangle ABD
    const double n_d_norm = n_d.norm();
    if (n_d_norm < 1e-12) return C;          // degenerate triangle ABD
    n_d /= n_d_norm;

    // Signed distance of C from the plane of ABD.
    const double sd = (C - A).dot(n_d);
    // Push away from D's plane: direction = sign(sd) * n_d. If C is on
    // the plane (sd == 0) the fold is exact — pick either side.
    const double sign = (sd >= 0.0) ? 1.0 : -1.0;
    const double step = step_frac * ab.norm();
    return C + sign * step * n_d;
}

// Compute, for a single half-edge h with a given cycle_mates list, the
// minimum dihedral across all partners (== same metric as
// smooth.cpp::calc_min_dihedral_cycle for a single index). Also returns
// which partner achieves the minimum (the worst pair).
struct EdgeDihedral { double min_di; u32 hp_arg; };
EdgeDihedral min_pair_dihedral(const mesh::HalfEdges& he, const Coords& xyz,
                               u32 h, const std::vector<u32>& mates) {
    EdgeDihedral R{180.0, kInvalidU32};
    if (mates.empty()) {
        const u32 hp = he(h, 3);
        if (hp != kInvalidU32) {
            const double d = dihedral_pair(he, xyz, h, hp);
            if (d >= 0) { R.min_di = d; R.hp_arg = hp; }
        }
        return R;
    }
    for (u32 hp : mates) {
        const double d = dihedral_pair(he, xyz, h, hp);
        if (d < 0) continue;
        if (d < R.min_di) { R.min_di = d; R.hp_arg = hp; }
    }
    return R;
}

// Build "half-edges incident to vertex v" map — used to evaluate every
// dihedral around an apex after moving it.
std::vector<std::vector<u32>> build_v2he(const mesh::HalfEdges& he) {
    std::vector<std::vector<u32>> v2h(0);
    Eigen::Index max_v = -1;
    for (Eigen::Index i = 0; i < he.rows(); ++i) max_v = std::max<Eigen::Index>(max_v, he(i, 0));
    v2h.assign(static_cast<std::size_t>(max_v + 1), {});
    for (Eigen::Index i = 0; i < he.rows(); ++i) {
        // The half-edge i is part of a triangle whose three half-edges
        // touch three vertices: target, next.target, prev.target. We
        // want "half-edges whose canonical edge includes v" — that's
        // any half-edge whose source (prev.target) or target equals v.
        // For dihedral evaluation around v's incident edges we only
        // need a representative half-edge per incident edge; using all
        // is fine because dihedral_pair is symmetric.
        const u32 t = he(i, 0);
        v2h[static_cast<std::size_t>(t)].push_back(static_cast<u32>(i));
    }
    return v2h;
}

}  // namespace

// ---------------------------------------------------------------------------
DangleRepairStats dangle_repair(const Settings& /*s*/,
                                const mesh::HalfEdges& he,
                                Coords& xyz,
                                const Coords& xyz_init,
                                const std::vector<std::uint8_t>& not_fixed_v,
                                double min_dangle_threshold,
                                double max_drift,
                                int max_passes) {
    DangleRepairStats R{0, 0, 0, 0, 0};
    if (max_passes <= 0) return R;

    auto v2h = build_v2he(he);

    // Helper: minimum cycle-dihedral around all half-edges incident to v.
    // Returns the worst-fold value seen at v (or +180 if none).
    auto min_dihedral_at_vertex = [&](u32 v,
                                      const std::vector<std::vector<u32>>& mates) {
        if (v >= v2h.size()) return 180.0;
        double mn = 180.0;
        for (u32 h : v2h[v]) {
            const auto& m = mates[static_cast<std::size_t>(h)];
            EdgeDihedral e = min_pair_dihedral(he, xyz, h, m);
            if (e.min_di < mn) mn = e.min_di;
        }
        return mn;
    };

    auto count_bad = [&](const std::vector<std::vector<u32>>& mates) {
        // Count unique non-manifold edges where the cycle-min dihedral
        // is below threshold. We deduplicate by canonical (lo,hi)
        // vertex pair so each edge is counted once.
        std::vector<std::array<u32, 2>> keys;
        for (Eigen::Index i = 0; i < he.rows(); ++i) {
            const auto& m = mates[static_cast<std::size_t>(i)];
            if (m.empty()) continue;        // manifold or boundary — skip
            EdgeDihedral e = min_pair_dihedral(he, xyz, static_cast<u32>(i), m);
            if (e.min_di >= min_dangle_threshold) continue;
            const u32 vp = he(he(i, 2), 0);
            const u32 vt = he(i, 0);
            keys.push_back({std::min(vp, vt), std::max(vp, vt)});
        }
        std::sort(keys.begin(), keys.end());
        keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
        return keys.size();
    };

    // Initial blink cycle & bad count.
    auto blinks0 = mesh::create_blinks(he, xyz);
    auto cycle_mates0 = build_cycle_mates(he, blinks0);
    R.bad_edges_before = count_bad(cycle_mates0);

    // Quick summary of incoming dihedral quality (geometric metric,
    // 180° = flat, 0° = full fold). Reports the smallest seen across
    // all non-manifold pair-occurrences so the user can see what
    // dihedral_repair is starting from.
    double dmin_seen = 180.0;
    for (Eigen::Index i = 0; i < he.rows(); ++i) {
        const auto& m = cycle_mates0[static_cast<std::size_t>(i)];
        if (m.size() < 2) continue;
        for (u32 hp : m) {
            const double d = dihedral_pair(he, xyz, static_cast<u32>(i), hp);
            if (d >= 0 && d < dmin_seen) dmin_seen = d;
        }
    }
    VOX2TET_LOG() << "dihedral_repair: min non-manifold dihedral = "
                  << dmin_seen << "° (threshold "
                  << min_dangle_threshold << "°)";

    // Step fraction for the apex lift. Conservative — push by ~5 % of
    // the edge length so we stay inside max_D2self with comfortable
    // margin and don't overshoot past 90°.
    const double step_frac = 0.05;

    auto cycle_mates = cycle_mates0;
    for (int pass = 0; pass < max_passes; ++pass) {
        R.passes_run = pass + 1;

        // Collect candidate (worst-first) bad edges this pass. Use a
        // canonical (lo,hi) key to dedupe so each undirected edge
        // appears once.
        struct Cand { double di; u32 h; u32 hp; };
        std::vector<Cand> cands;
        std::vector<char> seen_edge(static_cast<std::size_t>(he.rows()), 0);
        for (Eigen::Index i = 0; i < he.rows(); ++i) {
            const auto& m = cycle_mates[static_cast<std::size_t>(i)];
            if (m.empty()) continue;
            EdgeDihedral e = min_pair_dihedral(he, xyz, static_cast<u32>(i), m);
            if (e.min_di >= min_dangle_threshold) continue;
            if (seen_edge[static_cast<std::size_t>(i)]) continue;
            seen_edge[static_cast<std::size_t>(i)] = 1;
            for (u32 hp : m) {
                if (hp < seen_edge.size()) seen_edge[hp] = 1;
            }
            cands.push_back({e.min_di, static_cast<u32>(i), e.hp_arg});
        }
        std::sort(cands.begin(), cands.end(),
                  [](const Cand& a, const Cand& b) { return a.di < b.di; });

        std::size_t accepted_this_pass = 0;
        for (const auto& c : cands) {
            // Try lifting apex of h, then of hp.
            for (int side = 0; side < 2; ++side) {
                const u32 h_try  = (side == 0) ? c.h  : c.hp;
                const u32 h_part = (side == 0) ? c.hp : c.h;
                if (h_part == kInvalidU32) continue;
                const u32 apex = he(he(h_try, 1), 0);
                if (apex >= not_fixed_v.size() || !not_fixed_v[apex]) continue;

                ++R.attempted_moves;

                // Snapshot the apex.
                const Eigen::RowVector3d old_pos = xyz.row(apex);

                // Snapshot every dihedral currently incident to the
                // apex (we'll re-evaluate after the move).
                if (apex >= v2h.size()) continue;
                std::vector<u32>          adj_h     = v2h[apex];
                std::vector<double>       adj_di    = {};
                adj_di.reserve(adj_h.size());
                for (u32 h_adj : adj_h) {
                    EdgeDihedral e = min_pair_dihedral(
                        he, xyz, h_adj, cycle_mates[static_cast<std::size_t>(h_adj)]);
                    adj_di.push_back(e.min_di);
                }
                const double di_before = c.di;

                // Compute candidate position.
                Eigen::Vector3d new_pos = compute_apex_lift(
                    he, xyz, h_try, h_part, step_frac);

                // Drift guard.
                const Eigen::Vector3d delta_from_init =
                    new_pos - xyz_init.row(apex).transpose();
                if (delta_from_init.norm() > max_drift) continue;

                // Apply.
                xyz.row(apex) = new_pos.transpose();

                // Re-evaluate the target dihedral.
                const double di_after =
                    dihedral_pair(he, xyz, h_try, h_part);

                bool accept = (di_after > di_before + 0.5);  // ≥ 0.5° gain

                if (accept) {
                    // Check every neighbour dihedral around the apex —
                    // none may drop below threshold AND below its prior
                    // value (the same guard the smoothing code uses).
                    for (std::size_t j = 0; j < adj_h.size(); ++j) {
                        const u32 h_adj = adj_h[j];
                        EdgeDihedral e = min_pair_dihedral(
                            he, xyz, h_adj,
                            cycle_mates[static_cast<std::size_t>(h_adj)]);
                        if (e.min_di < min_dangle_threshold &&
                            e.min_di < adj_di[j] - 1e-6) {
                            accept = false;
                            break;
                        }
                    }
                }

                if (accept) {
                    ++R.accepted_moves;
                    ++accepted_this_pass;
                    break;  // move was accepted; don't try the other side
                } else {
                    xyz.row(apex) = old_pos;  // revert
                }
            }
        }

        if (accepted_this_pass == 0) break;
    }

    R.bad_edges_after = count_bad(cycle_mates);
    return R;
}

}  // namespace vox2tet::remesh
