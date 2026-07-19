#include "vox2tet/remesh/remesh.hpp"

#include "vox2tet/core/log.hpp"
#include "vox2tet/marching_cubes/contouring.hpp"
#include "vox2tet/mesh/mathx.hpp"
#include "vox2tet/remesh/smooth.hpp"

#include <Eigen/Geometry>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace vox2tet::remesh {

namespace {
constexpr u32 kInvalidU32 = std::numeric_limits<u32>::max();

// Canonical-edge multiplicity (number of half-edges per undirected
// vertex pair). A non-manifold edge (3-4 incident half-edges on a
// multi-material line) has its opposites paired arbitrarily two-by-two,
// so each pair looks interior to the not_fixed_h filter — but a split
// or flip acting on one pair desynchronizes that sheet from the others.
// Interior split/flip must act on genuinely manifold (mult == 2) edges.
inline std::uint64_t edge_key(u32 a, u32 b) {
    if (a > b) std::swap(a, b);
    return (static_cast<std::uint64_t>(a) << 32) | b;
}

std::unordered_map<std::uint64_t, u32> edge_multiplicity(const mesh::HalfEdges& he) {
    std::unordered_map<std::uint64_t, u32> mult;
    mult.reserve(static_cast<std::size_t>(he.rows()));
    for (Eigen::Index i = 0; i < he.rows(); ++i)
        ++mult[edge_key(he(he(i, 2), 0), he(i, 0))];
    return mult;
}
}  // namespace

// ---------------------------------------------------------------------------
// smooth_surfaces_tangential — port of smoothSurfacesTangential.
// Laplacian step, then re-project each vertex onto its initial tangent
// plane: xyz += normal * <normal, (xyz0 - xyz)>. Local revert where the
// dihedral angle around any not-fixed half-edge dropped below threshold.
// ---------------------------------------------------------------------------
void smooth_surfaces_tangential(const mesh::HalfEdges& he,
                                const std::vector<std::uint8_t>& not_fixed_h,
                                Coords& xyz,
                                const std::vector<std::uint8_t>& not_fixed_v,
                                const NormalsMat& normals,
                                int n_smooth_iter,
                                double min_dangle_internal) {
    VOX2TET_PRINT("Start tangential surfaces smoothing...");

    // Build adjacency from hedges' next-prev structure: directed edges
    // (prev.target → target). This matches the reference branch in
    // triangles2edges for hedges input.
    const Eigen::Index n_nodes = xyz.rows();
    std::vector<std::vector<u32>> adj(static_cast<std::size_t>(n_nodes));
    for (Eigen::Index i = 0; i < he.rows(); ++i) {
        const u32 src = he(he(i, 2), 0);  // prev.target
        const u32 dst = he(i, 0);          // target
        if (not_fixed_v[src]) adj[src].push_back(dst);
    }

    // Non-manifold blink cycle + cycle_mates (Layer 1 dihedral guard).
    // For the initial / final dihedral snapshots used by the sharp-edge
    // revert below, we take the MIN dihedral across all triangle pairs
    // around each not_fixed_h edge — covering folds at triple lines
    // and quad points that the canonical column-3 opposite misses.
    auto blinks = mesh::create_blinks(he, xyz);
    auto cycle_mates = build_cycle_mates(he, blinks);

    // Initial dihedral snapshot (cycle-min metric).
    auto initial = calc_internal_dihedral_cycle(he, not_fixed_h, xyz, cycle_mates);
    Coords xyz0 = xyz;

    for (int it = 0; it < n_smooth_iter; ++it) {
        // 1. Laplacian update.
        Coords new_xyz = xyz;
        for (Eigen::Index v = 0; v < n_nodes; ++v) {
            if (!not_fixed_v[static_cast<std::size_t>(v)]) continue;
            const auto& nbrs = adj[static_cast<std::size_t>(v)];
            if (nbrs.empty()) continue;
            Eigen::Vector3d sum = Eigen::Vector3d::Zero();
            for (auto n : nbrs) sum += xyz.row(n).transpose();
            const double alpha = 0.5;
            new_xyz.row(v) = alpha * xyz.row(v)
                           + (1.0 - alpha) * (sum / static_cast<double>(nbrs.size())).transpose();
        }
        xyz = new_xyz;

        // 2. Project onto tangent plane: xyz += normal * <normal, xyz0 - xyz>
        for (Eigen::Index v = 0; v < n_nodes; ++v) {
            if (!not_fixed_v[static_cast<std::size_t>(v)]) continue;
            const Eigen::Vector3d n  = normals.row(v).transpose();
            const Eigen::Vector3d d  = xyz0.row(v).transpose() - xyz.row(v).transpose();
            const double proj = n.dot(d);
            xyz.row(v) += (n * proj).transpose();
        }
    }

    // Final dihedral check — revert vertices on sharp hedges (cycle-min).
    auto final_dih = calc_internal_dihedral_cycle(he, not_fixed_h, xyz, cycle_mates);
    std::size_t n_reverted_edges = 0;
    for (Eigen::Index k = 0; k < final_dih.dan.size(); ++k) {
        const double d0 = initial.dan[k];
        const double d1 = final_dih.dan[k];
        const bool is_sharp = (d1 < 0.5 * d0) ||
                              ((d1 < min_dangle_internal) && (d1 <= d0));
        if (!is_sharp) continue;
        ++n_reverted_edges;
        const u32 h = initial.i1[static_cast<std::size_t>(k)];
        const u32 v1 = he(he(h, 1), 0);  // next.target
        const u32 v2 = he(h, 0);          // target
        xyz.row(v1) = xyz0.row(v1);
        xyz.row(v2) = xyz0.row(v2);

        // --- Layer 2: bidirectional revert at non-manifold edges ------
        // At a triple line / quad point, reverting just the two edge
        // endpoints often leaves the fold in place because the offending
        // motion was the third (apex) vertex of one of the incident
        // triangles tilting toward the plane of the other two. Revert
        // every apex of every triangle in the blink cycle to be safe.
        const auto& m = cycle_mates[static_cast<std::size_t>(h)];
        if (!m.empty()) {
            // Apex of h's own triangle: prev.target.
            const u32 ap_h = he(he(h, 2), 0);
            xyz.row(ap_h) = xyz0.row(ap_h);
            for (auto hp : m) {
                // Apex of triangle hp: prev.target. (Note: edges between
                // hp and v1/v2 might or might not coincide with the
                // canonical edge endpoints — but at any rate reverting
                // the apex is the right move; if it was already reverted
                // above the assignment is a no-op.)
                const u32 ap = he(he(hp, 2), 0);
                xyz.row(ap) = xyz0.row(ap);
            }
        }
    }
    if (n_reverted_edges > 0) {
        VOX2TET_LOG() << "tangential smoothing: reverted around "
                      << n_reverted_edges << " sharp hedges";
    }
    VOX2TET_PRINT("End tangential surfaces smoothing!");
}

// ---------------------------------------------------------------------------
// split_edges — find all not-fixed half-edges whose length exceeds
// 4/3 * min(sizing[both endpoints]), split each at midpoint, and stitch
// the new triangles into the half-edge structure.
//
// Each split: triangle pair (T1, T2) sharing half-edge h becomes four
// triangles. We add 6 new half-edges per split (3 per new sub-triangle,
// the parent edge becomes shared by re-using the slot at h and h.opp).
// Mirrors mesh_processing.split_edges.
// ---------------------------------------------------------------------------
void split_edges(RemeshState& st) {
    auto& he   = st.hedges;
    auto& xyz  = st.xyz;
    auto& nfh  = st.not_fixed_h;
    auto& nfv  = st.not_fixed_v;
    auto& norm = st.normals;
    auto& sz   = st.sizing;

    const Eigen::Index nhe0 = he.rows();
    const Eigen::Index nv0  = xyz.rows();

    // Splitting one pair of a non-manifold edge would subdivide the edge
    // on one sheet only, leaving the other sheet spanning the full chord
    // (a T-junction, χ drops by 1 per split).
    auto edge_mult = edge_multiplicity(he);

    // Filter: not-fixed, manifold, length > 4/3 * min(sz[a], sz[b]), and
    // to avoid double-processing pick orientation with target > source.
    std::vector<u32> itr;
    itr.reserve(static_cast<std::size_t>(nhe0));
    for (Eigen::Index i = 0; i < nhe0; ++i) {
        if (!nfh[static_cast<std::size_t>(i)]) continue;
        const u32 src = he(he(i, 2), 0);  // prev.target == source
        const u32 dst = he(i, 0);          // target
        if (dst <= src) continue;
        if (edge_mult[edge_key(src, dst)] != 2) continue;
        const Eigen::Vector3d a = xyz.row(src).transpose();
        const Eigen::Vector3d b = xyz.row(dst).transpose();
        const double L = (a - b).norm();
        const double th = (4.0 / 3.0) * std::min(sz[src], sz[dst]);
        if (L > th) itr.push_back(static_cast<u32>(i));
    }
    if (itr.empty()) return;

    VOX2TET_LOG() << "split_edges: " << itr.size() << " edges to split";

    // Pre-grow arrays.
    const Eigen::Index nv1 = nv0  + static_cast<Eigen::Index>(itr.size());
    const Eigen::Index nh1 = nhe0 + static_cast<Eigen::Index>(itr.size()) * 6;
    xyz.conservativeResize(nv1, 3);
    norm.conservativeResize(nv1, 3);
    sz.conservativeResize(nv1);
    nfv.resize(static_cast<std::size_t>(nv1), 1);
    he.conservativeResize(nh1, 5);
    nfh.resize(static_cast<std::size_t>(nh1), 1);

    // For each split: compute midpoint, then rewire half-edges. The
    // layout mirrors the reference loop (4 sub-triangles, 6 new half-edges).
    for (std::size_t k = 0; k < itr.size(); ++k) {
        const u32 h = itr[k];
        const Eigen::Index i0 = nhe0 + static_cast<Eigen::Index>(k) * 6;
        const u32 nv_new = static_cast<u32>(nv0 + k);

        // ei = [h, h.next, h.prev, h.opp, h.opp.next, h.opp.prev]
        const u32 e0 = h;
        const u32 e1 = he(e0, 1);
        const u32 e2 = he(e0, 2);
        const u32 e3 = he(e0, 3);
        if (e3 == kInvalidU32) {
            // Boundary edge — the legacy filter `not_fixed_h & (vL > sz)`
            // already excludes them; we double-check.
            continue;
        }
        const u32 e4 = he(e3, 1);
        const u32 e5 = he(e3, 2);

        // vi = [prev.target, target, next.target, e4.target] = quad verts
        const u32 vi0 = he(e2, 0);
        const u32 vi1 = he(e0, 0);
        const u32 vi2 = he(e1, 0);
        const u32 vi3 = he(e4, 0);

        // Midpoint of (vi0, vi1).
        xyz.row(nv_new)  = 0.5 * (xyz.row(vi0) + xyz.row(vi1));
        norm.row(nv_new) = 0.5 * (norm.row(vi0) + norm.row(vi1));
        sz[nv_new]       = 0.5 * (sz[vi0] + sz[vi1]);

        // Interface ids from the original half-edges (column 4 is the
        // per-triangle interface).
        const u32 itf_0 = he(e0, 4);
        const u32 itf_3 = he(e3, 4);

        // New triangle 0 (mid → vi2 part).
        he(i0 + 0, 0) = nv_new;
        he(i0 + 0, 1) = static_cast<u32>(i0 + 1);
        he(i0 + 0, 2) = e2;
        he(i0 + 0, 3) = e3;
        he(i0 + 0, 4) = itf_0;

        he(i0 + 1, 0) = vi2;
        he(i0 + 1, 1) = e2;
        he(i0 + 1, 2) = static_cast<u32>(i0 + 0);
        he(i0 + 1, 3) = static_cast<u32>(i0 + 2);
        he(i0 + 1, 4) = itf_0;

        he(e2, 1) = static_cast<u32>(i0 + 0);
        he(e2, 2) = static_cast<u32>(i0 + 1);

        // New triangle 1 (mid → vi1 → vi2).
        he(i0 + 2, 0) = nv_new;
        he(i0 + 2, 1) = e0;
        he(i0 + 2, 2) = e1;
        he(i0 + 2, 3) = static_cast<u32>(i0 + 1);
        he(i0 + 2, 4) = itf_0;

        he(e0, 2) = static_cast<u32>(i0 + 2);
        he(e0, 3) = static_cast<u32>(i0 + 3);
        he(e1, 1) = static_cast<u32>(i0 + 2);

        // New triangle 2 (other side: mid → vi3 → vi0).
        he(i0 + 3, 0) = nv_new;
        he(i0 + 3, 1) = static_cast<u32>(i0 + 4);
        he(i0 + 3, 2) = e5;
        he(i0 + 3, 3) = e0;
        he(i0 + 3, 4) = itf_3;

        he(i0 + 4, 0) = vi3;
        he(i0 + 4, 1) = e5;
        he(i0 + 4, 2) = static_cast<u32>(i0 + 3);
        he(i0 + 4, 3) = static_cast<u32>(i0 + 5);
        he(i0 + 4, 4) = itf_3;

        he(e5, 1) = static_cast<u32>(i0 + 3);
        he(e5, 2) = static_cast<u32>(i0 + 4);

        // New triangle 3.
        he(i0 + 5, 0) = nv_new;
        he(i0 + 5, 1) = e3;
        he(i0 + 5, 2) = e4;
        he(i0 + 5, 3) = static_cast<u32>(i0 + 4);
        he(i0 + 5, 4) = itf_3;

        he(e3, 2) = static_cast<u32>(i0 + 5);
        he(e3, 3) = static_cast<u32>(i0 + 0);
        he(e4, 1) = static_cast<u32>(i0 + 5);
    }
}

// ---------------------------------------------------------------------------
// flip_half_edge — basic interior edge flips. Iterates not-fixed half-
// edges whose shape factor is < 0.5 and swaps the underlying quad's
// diagonal when both new triangles have a strictly better minimum shape
// factor AND new dihedral angles aren't worse than 120° / current.
//
// This omits the V2H boundary-edge uniqueness check from the reference
// implementation — fine for interior flips, which is what the no-boundary
// path of `flipHalfEdge` does.
// ---------------------------------------------------------------------------
void flip_half_edge(const Settings& s,
                    mesh::HalfEdges& he,
                    const std::vector<std::uint8_t>& not_fixed_h,
                    Coords& xyz,
                    const std::vector<std::uint8_t>& /*not_fixed_v*/,
                    bool do_not_boundary,
                    bool do_only_boundary) {
    if (do_only_boundary) {
        // Boundary-only flips need the V2H matrix from collapse_edges;
        // skip cleanly until that's ported.
        return;
    }

    // Build adjacency once for cheap "is (u,v) already an edge?" queries.
    const Eigen::Index n_nodes = xyz.rows();
    std::vector<std::unordered_map<u32, std::uint8_t>> adj(
        static_cast<std::size_t>(n_nodes));
    for (Eigen::Index i = 0; i < he.rows(); ++i) {
        const u32 src = he(he(i, 2), 0);
        const u32 dst = he(i, 0);
        adj[src][dst] = 1;
    }

    // Flipping one opposite-pair of a non-manifold edge would remove the
    // diagonal from this sheet only, detaching it from the other sheets
    // that still span the edge. Maintained across accepted flips.
    auto edge_mult = edge_multiplicity(he);

    // V2H = boundary half-edges per vertex. Used to gate flips that would
    // duplicate an existing edge across a non-manifold (multi-material)
    // boundary. Mirrors the reference
    //   (is_b_v[vi[2]] and is_b_v[vi[3]]) and is_edge_exist(...).
    auto v2h = mesh::get_v2h_lists(he);
    const auto& is_bv = v2h.is_boundary_vertex;
    auto safe_is_bv = [&](u32 v) -> bool {
        return v < is_bv.size() && is_bv[v] != 0;
    };

    // Non-manifold blink cycle: for each perimeter half-edge, all OTHER
    // half-edges sharing the same canonical edge. the legacy flip uses
    // these as dihedral partners so a flip near a multi-material line
    // doesn't ignore the dihedral against the other triangles meeting
    // along that line.
    auto blinks = mesh::create_blinks(he, xyz);
    // Collect cycle-mates: cycle_mates[h] = list of all other hedges in
    // h's blink cycle.
    std::vector<std::vector<u32>> cycle_mates(static_cast<std::size_t>(he.rows()));
    for (Eigen::Index i = 0; i < he.rows(); ++i) {
        const u32 b = blinks[static_cast<std::size_t>(i)];
        if (b == kInvalidU32) continue;
        u32 cur = b;
        for (int steps = 0; steps < 8 && cur != static_cast<u32>(i); ++steps) {
            cycle_mates[static_cast<std::size_t>(i)].push_back(cur);
            cur = blinks[cur];
            if (cur == kInvalidU32) break;
        }
    }

    // Find candidate half-edges (not fixed + bad shape factor).
    using Vec3 = Eigen::Vector3d;
    auto shape_factor = [](const Vec3& a, const Vec3& b, const Vec3& c) {
        const Vec3 e1 = b - a, e2 = c - a, e3 = c - b;
        const Vec3 cr = e1.cross(e2);
        const double A   = 0.5 * cr.norm();
        const double s1  = e1.squaredNorm(), s2 = e2.squaredNorm(), s3 = e3.squaredNorm();
        const double prod = s1 * s2 * s3;
        if (prod < 1e-30) return 0.0;
        return (64.0 * std::sqrt(3.0) / 9.0) * A * A * A / prod;
    };

    std::vector<u32> candidates;
    for (Eigen::Index i = 0; i < he.rows(); ++i) {
        if (!not_fixed_h[static_cast<std::size_t>(i)]) continue;
        const u32 va = he(i, 0);
        const u32 vb = he(he(i, 1), 0);
        const u32 vc = he(he(i, 2), 0);
        const double sf = shape_factor(xyz.row(va).transpose(),
                                       xyz.row(vb).transpose(),
                                       xyz.row(vc).transpose());
        if (sf < 0.5) candidates.push_back(static_cast<u32>(i));
    }

    std::size_t n_flipped = 0;
    for (u32 h : candidates) {
        // Re-check (could have been modified by an earlier flip).
        if (!not_fixed_h[h]) continue;
        const u32 e0 = h;
        const u32 e1 = he(e0, 1);
        const u32 e2 = he(e0, 2);
        const u32 e3 = he(e0, 3);
        if (e3 == kInvalidU32) continue;
        const u32 e4 = he(e3, 1);
        const u32 e5 = he(e3, 2);

        const u32 vi0 = he(e2, 0);
        const u32 vi1 = he(e0, 0);
        const u32 vi2 = he(e1, 0);
        const u32 vi3 = he(e4, 0);

        const auto em = edge_mult.find(edge_key(vi0, vi1));
        if (em == edge_mult.end() || em->second != 2) continue;

        // Reject if (vi2, vi3) already an edge (the standard adj check).
        if (adj[vi2].count(vi3) > 0) continue;

        // Non-manifold edge guard: if both vi2 and vi3 are boundary
        // vertices, the new edge could duplicate one that already
        // exists across a multi-material line (where the standard adj
        // table doesn't see through the non-manifold pairing). Walk
        // vi2's boundary half-edges and look for vi3 — mirrors the reference
        //   (is_b_v[vi[2]] and is_b_v[vi[3]]) and is_edge_exist(hedges,
        //        V2H.rows[vi[2]], vi[3])
        if (safe_is_bv(vi2) && safe_is_bv(vi3)) {
            if (mesh::is_edge_exist(he, v2h.by_vertex[vi2], vi3)) continue;
        }

        if (do_not_boundary) {
            // Skip when either new vertex is on the boundary.
            // Boundary-vertex test via the V2H map we already built.
            if (safe_is_bv(vi2) || safe_is_bv(vi3)) continue;
        }

        // Save state for rollback.
        std::array<Eigen::Matrix<u32, 1, 5>, 6> tmp;
        const u32 eis[6] = {e0, e1, e2, e3, e4, e5};
        for (int k = 0; k < 6; ++k) tmp[k] = he.row(eis[k]);

        // --- Dihedral-regression check (the legacy quad-perimeter pass) ----
        // For each of the 5 perimeter half-edges (ei[0,1,2,4,5]) check
        // the dihedral against EVERY other half-edge sharing the same
        // canonical edge (manifold = 1 cycle-mate, non-manifold = 2 or
        // 3). Take the minimum (most pessimistic) for regression.
        const u32 perim[5] = {e0, e1, e2, e4, e5};
        auto dihedral_pair = [&](const Eigen::Matrix<u32, Eigen::Dynamic, 5, Eigen::RowMajor>& mat,
                                 u32 h, u32 hp) -> double {
            if (hp == kInvalidU32) return 180.0;
            const Eigen::Vector3d p01 = xyz.row(mat(mat(h,  2), 0)).transpose();
            const Eigen::Vector3d p02 = xyz.row(mat(h,         0)).transpose();
            const Eigen::Vector3d p1  = xyz.row(mat(mat(h,  1), 0)).transpose();
            const Eigen::Vector3d p2  = xyz.row(mat(mat(hp, 1), 0)).transpose();
            const Eigen::Vector3d v0  = p02 - p01;
            const Eigen::Vector3d c1  = (p1 - p01).cross(v0);
            const Eigen::Vector3d c2  = (p2 - p01).cross(v0);
            const double n12 = c1.norm() * c2.norm();
            return (n12 < 1e-18) ? -1.0
                : std::acos(std::clamp(c1.dot(c2) / n12, -1.0, 1.0)) * 180.0 / M_PI;
        };
        std::array<double, 5> da_before;
        for (int k = 0; k < 5; ++k) {
            const u32 h = perim[k];
            const auto& mates = cycle_mates[h];
            double dm = 180.0;
            if (mates.empty()) {
                dm = dihedral_pair(he, h, he(h, 3));
            } else {
                for (auto hp : mates) {
                    const double d = dihedral_pair(he, h, hp);
                    if (d < 0) continue;
                    if (d < dm) dm = d;
                }
            }
            da_before[k] = dm;
        }

        // Apply flip:
        //   e0 / e3 change target vertex to vi2 / vi3
        //   next / prev pointers permute as: 0→2 next, 2→4 prev, etc.
        he(e0, 0) = vi2;
        he(e3, 0) = vi3;
        const u32 next_new[6] = {e2, e3, e4, e5, e0, e1};
        const u32 prev_new[6] = {e4, e5, e0, e1, e2, e3};
        for (int k = 0; k < 6; ++k) {
            he(eis[k], 1) = next_new[k];
            he(eis[k], 2) = prev_new[k];
        }

        // Shape-factor check on the new pair of triangles.
        const Vec3 pa = xyz.row(vi0).transpose();
        const Vec3 pb = xyz.row(vi1).transpose();
        const Vec3 pc = xyz.row(vi2).transpose();
        const Vec3 pd = xyz.row(vi3).transpose();
        const double sf_old1 = shape_factor(pa, pb, pc);
        const double sf_old2 = shape_factor(pa, pd, pb);
        const double sf_new1 = shape_factor(pc, pd, pa);
        const double sf_new2 = shape_factor(pc, pb, pd);
        const double min_old = std::min(sf_old1, sf_old2);
        const double min_new = std::min(sf_new1, sf_new2);

        auto revert = [&]() {
            for (int k = 0; k < 6; ++k) he.row(eis[k]) = tmp[k];
        };

        if (min_new <= min_old) { revert(); continue; }

        // Compute post-flip dihedral angles around the 5 perimeter edges
        // (min over all cycle-mates).
        std::array<double, 5> da_after;
        for (int k = 0; k < 5; ++k) {
            const u32 h = perim[k];
            const auto& mates = cycle_mates[h];
            double dm = 180.0;
            if (mates.empty()) {
                dm = dihedral_pair(he, h, he(h, 3));
            } else {
                for (auto hp : mates) {
                    const double d = dihedral_pair(he, h, hp);
                    if (d < 0) continue;
                    if (d < dm) dm = d;
                }
            }
            da_after[k] = dm;
        }

        // da_after[0] is the dihedral across the new diagonal edge — must
        // stay >= 120° to keep the surface smooth.
        if (da_after[0] < 0 || da_after[0] < 120.0) { revert(); continue; }

        // The two angle gates: the quad must not be too degenerate
        // (angles at vi2 and vi3 not both > 150°). If either inner angle
        // is < 150°, apply the regression thresholds below.
        auto angle_between = [&](const Vec3& a, const Vec3& b) {
            const double na = a.norm(), nb = b.norm();
            if (na < 1e-18 || nb < 1e-18) return -1.0;
            return std::acos(std::clamp(a.dot(b) / (na * nb), -1.0, 1.0)) * 180.0 / M_PI;
        };
        const double a_at_c = angle_between(pa - pc, pb - pc);
        const double a_at_d = angle_between(pa - pd, pb - pd);
        if (a_at_c >= 0 && a_at_c < 150.0 && a_at_d >= 0 && a_at_d < 150.0) {
            // Per-perimeter-edge regression checks. Reject if any flip
            // makes a dihedral measurably sharper than before.
            bool regressed = false;
            for (int k = 0; k < 5; ++k) {
                const double d0 = da_before[k];
                const double d1 = da_after [k];
                if (d0 < 0 || d1 < 0) continue;
                // Threshold rule: (0.7*da0 > da1 AND da1 > 30) OR
                //         (0.9*da0 > da1 AND da1 <= 30)
                if ((0.7 * d0 > d1 && d1 > 30.0) ||
                    (0.9 * d0 > d1 && d1 <= 30.0)) { regressed = true; break; }
            }
            if (regressed) { revert(); continue; }

            // do_not_boundary path also wants min_dangle_internal preserved.
            if (do_not_boundary) {
                bool sharp = false;
                for (int k = 0; k < 5; ++k) {
                    if (da_after[k] >= 0 &&
                        da_after[k] < s.min_dangle_internal &&
                        da_after[k] <= da_before[k]) { sharp = true; break; }
                }
                if (sharp) { revert(); continue; }
            }
        }

        // Accept flip.
        adj[vi0].erase(vi1);
        adj[vi1].erase(vi0);
        adj[vi2][vi3] = 1;
        adj[vi3][vi2] = 1;
        edge_mult.erase(edge_key(vi0, vi1));
        edge_mult[edge_key(vi2, vi3)] = 2;
        ++n_flipped;
    }

    if (n_flipped > 0) VOX2TET_LOG() << "flip_half_edge: flipped " << n_flipped;
}

// ---------------------------------------------------------------------------
// collapse_edges — manifold-only port for the interior case
// (do_not_boundary == true). Mirrors the most common path of
// mesh_processing.collapse_edges:
//
//   For each not-fixed interior half-edge h whose length < 4/5 *
//   min(sizing[src], sizing[dst]):
//     * walk the half-edge ring around src (= vi[0]) and around dst
//       (= vi[1]); reject if any outer vertex appears in both rings
//       (a "fold" that would make the result non-manifold).
//     * compute the new triangle normals and the new dihedral angles
//       around the ring; reject if anyone goes below
//       min_dangle_internal AND worse than before.
//     * collapse: re-target all half-edges pointing to src to point to
//       dst, rewire two opposite-pointer pairs around the removed
//       triangles, mark 6 hedges + 1 vertex as removed.
//
// Then compact: drop removed half-edges, drop removed vertices, remap
// all surviving half-edge fields (next/prev/opposite, target vertex).
// Mirrors the bottom half of mesh_processing.collapse_edges.
// ---------------------------------------------------------------------------
namespace {

// Collect outgoing half-edges from vertex `src` by walking the ring
// hedges[h].next starting at `h_start`, advancing via
// `h = he(he(h, 2), 3)` (prev.opp). Stops when it returns to h_start or
// when prev.opp hits a sentinel.
//
// Returns the list of "next" half-edges in the ring (i.e. for each
// incident triangle, the half-edge that goes from src's neighbour to
// the next-neighbour). Empty list signals a non-manifold ring (open
// boundary or kInvalidU32 hit).
// Mirrors the reference `while h != h_end: append(h.next); h = h.prev.opp`.
// Returns the list of `next` half-edges visited; empty on non-manifold
// (boundary hit before closing the loop).
std::vector<u32> walk_ring(const mesh::HalfEdges& he, u32 h_start, u32 h_end) {
    std::vector<u32> out;
    u32 h = h_start;
    for (std::size_t i = 0; i < 1024; ++i) {  // safety cap
        if (h == h_end) return out;
        out.push_back(he(h, 1));               // store h.next
        const u32 prev = he(h, 2);
        const u32 prev_opp = he(prev, 3);
        if (prev_opp == kInvalidU32) return {};  // boundary
        h = prev_opp;
    }
    return {};
}

inline double squared_dist(const Coords& xyz, u32 a, u32 b) {
    const Eigen::Vector3d d = xyz.row(a).transpose() - xyz.row(b).transpose();
    return d.squaredNorm();
}

}  // namespace

void collapse_edges(const Settings& s, RemeshState& st,
                    bool do_not_boundary, bool do_only_boundary) {
    if (do_only_boundary) {
        // The boundary-only path needs V2H + create_boundary_hedges_links —
        // still deferred.
        return;
    }
    if (!do_not_boundary) {
        // The "mixed" path (collapse boundaries too) also needs V2H; the
        // remesh driver only calls us with do_not_boundary=true.
        return;
    }

    auto& he   = st.hedges;
    auto& xyz  = st.xyz;
    auto& nfh  = st.not_fixed_h;
    auto& nfv  = st.not_fixed_v;
    auto& norm = st.normals;
    auto& sz   = st.sizing;

    const Eigen::Index nhe0 = he.rows();
    const Eigen::Index nv0  = xyz.rows();

    std::vector<std::uint8_t> h_collapsed(static_cast<std::size_t>(nhe0), 0);
    std::vector<std::uint8_t> v_collapsed(static_cast<std::size_t>(nv0), 0);

    const double c45 = 4.0 / 5.0;
    std::size_t n_collapsed = 0;
    std::size_t n_cand = 0, n_skip_len = 0, n_skip_ring = 0, n_skip_dup = 0,
                n_skip_dih = 0, n_skip_orient = 0, n_skip_qual = 0;

    // Iterate half-edges in stable order — newest-added ones may sit at
    // the end; that's fine.
    for (Eigen::Index ih = 0; ih < nhe0; ++ih) {
        const u32 i = static_cast<u32>(ih);
        if (h_collapsed[i] || !nfh[i]) continue;

        // ei = [h, h.next, h.prev, h.opp, h.opp.next, h.opp.prev]
        const u32 e0 = i;
        const u32 e1 = he(e0, 1);
        const u32 e2 = he(e0, 2);
        const u32 e3 = he(e0, 3);
        if (e3 == kInvalidU32) continue;
        const u32 e4 = he(e3, 1);
        const u32 e5 = he(e3, 2);

        const u32 vi0 = he(e2, 0);     // source (the one to be removed)
        const u32 vi1 = he(e0, 0);     // target (kept)
        const u32 vi2 = he(e1, 0);
        const u32 vi3 = he(e4, 0);

        if (v_collapsed[vi0]) continue;
        if (!nfv[vi0] || !nfv[vi1]) continue;     // do_not_boundary

        // Length check.
        const double L = c45 * std::min(sz[vi0], sz[vi1]);
        if (squared_dist(xyz, vi0, vi1) > L * L) { ++n_skip_len; continue; }
        ++n_cand;

        // Rings around vi0 and vi1.
        const u32 h_a_start = he(e2, 3);   // ei[2].opp == o_edges[1]
        const u32 h_a_end   = e4;           // ei[3].next
        const u32 h_b_start = he(e5, 3);   // o_edges[3]
        const u32 h_b_end   = e1;
        auto c_h  = walk_ring(he, h_a_start, h_a_end);
        auto c_hr = walk_ring(he, h_b_start, h_b_end);
        if (c_h.empty() || c_hr.empty()) { ++n_skip_ring; continue; }   // boundary detected

        // Outer-vertex sets for the two rings.
        std::vector<u32> v_ij;
        v_ij.reserve(c_h.size());
        for (std::size_t k = 0; k + 1 < c_h.size(); ++k) v_ij.push_back(he(c_h[k], 0));
        std::vector<u32> v_ji;
        v_ji.reserve(c_hr.size());
        for (std::size_t k = 0; k + 1 < c_hr.size(); ++k) v_ji.push_back(he(c_hr[k], 0));

        // Duplicate check — a third vertex connected to both vi0 and vi1
        // would yield a non-manifold result.
        std::vector<u32> all = v_ij; all.insert(all.end(), v_ji.begin(), v_ji.end());
        std::sort(all.begin(), all.end());
        if (std::adjacent_find(all.begin(), all.end()) != all.end()) { ++n_skip_dup; continue; }

        // Conservative ring guard (the legacy ncpus>=2 path): skip if any
        // half-edge in the c_h ring is itself "fixed" (boundary or near-
        // boundary). This is what keeps the result manifold without the
        // full blink table. The single-cpu path performs a more elaborate
        // boundary-dihedral check; we approximate it with this skip.
        //
        // Cap fix (do_collapse_near_bedges, only with do_reseed_bedges):
        // that skip freezes every vertex one ring from a feature chain,
        // which is exactly what leaves cap triangles — a free vertex
        // nearly collinear with a coarsened chain chord — in the final
        // mesh. The collapse rewiring only ever touches the four side
        // edges of the two dying triangles, so a fixed hedge elsewhere
        // in the ring is safe provided those four (and the collapsing
        // pair itself) are free, manifold and mutual. Retargeted fan
        // triangles that touch the chain are quality-guarded below.
        bool ring_has_fixed = false;
        for (auto h : c_h)  if (!nfh[h]) { ring_has_fixed = true; break; }
        if (!ring_has_fixed) {
            for (auto h : c_hr) if (!nfh[h]) { ring_has_fixed = true; break; }
        }
        if (ring_has_fixed) {
            if (!(s.do_reseed_bedges && s.do_collapse_near_bedges)) {
                ++n_skip_dih;
                continue;
            }
            bool dying_ok = he(e3, 3) == e0;
            for (u32 eS : {e1, e2, e4, e5}) {
                const u32 op = he(eS, 3);
                if (!nfh[eS] || op == kInvalidU32 || he(op, 3) != eS) {
                    dying_ok = false;
                    break;
                }
            }
            if (!dying_ok) { ++n_skip_dih; continue; }
        }

        // Dihedral angle check around the c_h ring.
        // For each h in c_h that is not-fixed, compute the dihedral
        // (h.opp, h)-edge angle to its current opposite triangle's
        // outer vertex against (a) vi0 (current) and (b) vi1 (post-collapse).
        bool dihedral_ok = true;
        for (auto h : c_h) {
            if (!nfh[h]) continue;
            const u32 h_opp = he(h, 3);
            if (h_opp == kInvalidU32) continue;
            const Eigen::Vector3d p0 = xyz.row(he(h, 0)).transpose();
            const Eigen::Vector3d p1 = xyz.row(he(h_opp, 0)).transpose();
            const Eigen::Vector3d p2 = xyz.row(he(he(h_opp, 1), 0)).transpose();
            const Eigen::Vector3d p30 = xyz.row(vi0).transpose();
            const Eigen::Vector3d p31 = xyz.row(vi1).transpose();
            const Eigen::Vector3d v0 = p1 - p0;
            const Eigen::Vector3d v1a = p2 - p0;
            const Eigen::Vector3d v2a = p30 - p0;
            const Eigen::Vector3d v2b = p31 - p0;
            auto angle = [&](const Eigen::Vector3d& a, const Eigen::Vector3d& b) {
                const Eigen::Vector3d ca = a.cross(v0);
                const Eigen::Vector3d cb = b.cross(v0);
                const double na = ca.norm(), nb = cb.norm();
                if (na < 1e-12 || nb < 1e-12) return -1.0;
                const double c = std::clamp(ca.dot(cb) / (na * nb), -1.0, 1.0);
                return std::acos(c) * 180.0 / M_PI;
            };
            const double dA0 = angle(v1a, v2a);
            const double dA1 = angle(v1a, v2b);
            if (dA1 < s.min_dangle_internal && dA1 < dA0) {
                dihedral_ok = false;
                break;
            }
        }
        if (!dihedral_ok) { ++n_skip_dih; continue; }

        // "Is the new edge too long?" check — the reference's
        //   is_long_new_edges = ||xyz[c_h_target] - xyz[vi1]|| >
        //                       4/3 * min(sizing[c_h_target], sizing[vi1])
        //   if any(is_long_new_edges[:-1]) -> skip
        // Excluding the last c_h entry because its target is vi3 (the
        // T2 third vertex), where the "new" edge would coincide with
        // an already-existing one.
        bool any_too_long = false;
        for (std::size_t k = 0; k + 1 < c_h.size(); ++k) {
            const u32 ov = he(c_h[k], 0);                 // outer ring vertex
            const double th = (4.0 / 3.0) * std::min(sz[ov], sz[vi1]);
            if (squared_dist(xyz, ov, vi1) > th * th) { any_too_long = true; break; }
        }
        if (any_too_long) { ++n_skip_orient; continue; }

        // Hoppe-style adjacency check: cross-products of (p1-vi1) ×
        // (p2-vi1) around the new triangle fan must all be non-
        // degenerate and consecutive normals must agree (cos ≥
        // cos(min_dangle_internal)).
        const double cos_thresh = std::cos(s.min_dangle_internal * M_PI / 180.0);
        std::vector<Eigen::Vector3d> v_cr;
        v_cr.reserve(c_h.size());
        bool adjacency_ok = true;
        for (std::size_t k = 0; k + 1 < c_h.size(); ++k) {
            const Eigen::Vector3d p1 = xyz.row(he(c_h[k],     0)).transpose();
            const Eigen::Vector3d p2 = xyz.row(he(c_h[k + 1], 0)).transpose();
            const Eigen::Vector3d b  = xyz.row(vi1).transpose();
            const Eigen::Vector3d a1 = p1 - b, a2 = p2 - b;
            const Eigen::Vector3d cr = a1.cross(a2);
            if (cr.squaredNorm() < L * L * L * L * 1e-6) { adjacency_ok = false; break; }
            v_cr.push_back(cr);
        }
        if (adjacency_ok) {
            for (std::size_t k = 0; k + 1 < v_cr.size(); ++k) {
                const double na = v_cr[k].norm(), nb = v_cr[k + 1].norm();
                if (na < 1e-18 || nb < 1e-18) { adjacency_ok = false; break; }
                const double cos_ab = v_cr[k].dot(v_cr[k + 1]) / (na * nb);
                if (cos_ab < cos_thresh) { adjacency_ok = false; break; }
            }
        }
        if (!adjacency_ok) { ++n_skip_orient; continue; }

        // Near-chain quality guard (relaxed-path collapses only): any
        // retargeted fan triangle that keeps a chain vertex must not
        // flip its normal, and its min corner angle must stay above
        // min_corner_angle_boundary unless it was already below and
        // does not get worse. This is what lets a cap vertex disappear
        // sideways (the ex-cap triangle is re-evaluated with the new
        // apex) while forbidding the collapse from minting new caps.
        if (ring_has_fixed) {
            auto min_corner = [](const Eigen::Vector3d& a,
                                 const Eigen::Vector3d& b,
                                 const Eigen::Vector3d& c) {
                auto ang = [](const Eigen::Vector3d& u,
                              const Eigen::Vector3d& v) {
                    const double nu = u.norm(), nv = v.norm();
                    if (nu < 1e-15 || nv < 1e-15) return 0.0;
                    const double d =
                        std::clamp(u.dot(v) / (nu * nv), -1.0, 1.0);
                    return std::acos(d) * 180.0 / M_PI;
                };
                return std::min({ang(b - a, c - a), ang(a - b, c - b),
                                 ang(a - c, b - c)});
            };
            bool qual_ok = true;
            const Eigen::Vector3d b0 = xyz.row(vi0).transpose();
            const Eigen::Vector3d b1 = xyz.row(vi1).transpose();
            for (std::size_t k = 0; k + 1 < c_h.size(); ++k) {
                const u32 va = he(c_h[k], 0), vb = he(c_h[k + 1], 0);
                if (nfv[va] && nfv[vb]) continue;    // no chain contact
                const Eigen::Vector3d pa = xyz.row(va).transpose();
                const Eigen::Vector3d pb = xyz.row(vb).transpose();
                const Eigen::Vector3d n_old = (pa - b0).cross(pb - b0);
                const Eigen::Vector3d n_new = (pa - b1).cross(pb - b1);
                if (n_new.dot(n_old) <= 0.0) { qual_ok = false; break; }
                const double a_new = min_corner(b1, pa, pb);
                if (a_new < s.min_corner_angle_boundary &&
                    a_new < min_corner(b0, pa, pb)) {
                    qual_ok = false;
                    break;
                }
            }
            if (!qual_ok) { ++n_skip_qual; continue; }
        }

        // ---- Apply collapse -------------------------------------------
        // 1. All half-edges hitting vi0 now hit vi1. the reference does
        //      hedges[hedges[c_h, 1], 0] = vi1
        //    i.e. for each c_h[k] (= h.next walked around vi0), follow
        //    .next once more to land on the hedge that closes the
        //    triangle back to vi0, and retarget that hedge to vi1.
        for (auto h : c_h) he(he(h, 1), 0) = vi1;

        // 2. Re-link opposite pointers of the four edges that used to be
        //    incident to the two collapsed triangles.
        //    reference: hedges[o_edges[not_b], 3] = hedges[ei[[2,1,5,4]][not_b], 3]
        const u32 o1 = he(e1, 3);
        const u32 o2 = he(e2, 3);
        const u32 o4 = he(e4, 3);
        const u32 o5 = he(e5, 3);
        if (o1 != kInvalidU32) he(o1, 3) = he(e2, 3);   // pair (o1,o2)
        if (o2 != kInvalidU32) he(o2, 3) = he(e1, 3);
        if (o4 != kInvalidU32) he(o4, 3) = he(e5, 3);
        if (o5 != kInvalidU32) he(o5, 3) = he(e4, 3);

        // 3. Mark removed.
        for (u32 e : {e0, e1, e2, e3, e4, e5}) h_collapsed[e] = 1;
        v_collapsed[vi0] = 1;
        ++n_collapsed;
    }

    VOX2TET_LOG() << "collapse_edges stats: cand=" << n_cand
                  << " skip_ring=" << n_skip_ring
                  << " skip_dup=" << n_skip_dup
                  << " skip_dih=" << n_skip_dih
                  << " skip_orient=" << n_skip_orient
                  << " skip_qual=" << n_skip_qual
                  << " collapsed=" << n_collapsed;
    if (n_collapsed == 0) return;

    // ---- Compact -------------------------------------------------------
    // 1. Build half-edge remap: kept hedges get new indices 0..n_new-1.
    std::vector<u32> he_remap(static_cast<std::size_t>(nhe0), kInvalidU32);
    Eigen::Index n_new = 0;
    for (Eigen::Index i = 0; i < nhe0; ++i) {
        if (!h_collapsed[static_cast<std::size_t>(i)])
            he_remap[static_cast<std::size_t>(i)] = static_cast<u32>(n_new++);
    }
    // 2. Build vertex remap.
    std::vector<u32> v_remap(static_cast<std::size_t>(nv0), kInvalidU32);
    Eigen::Index n_v_new = 0;
    for (Eigen::Index i = 0; i < nv0; ++i) {
        if (!v_collapsed[static_cast<std::size_t>(i)])
            v_remap[static_cast<std::size_t>(i)] = static_cast<u32>(n_v_new++);
    }
    // 3. Apply to hedges.
    mesh::HalfEdges he2(n_new, 5);
    std::vector<std::uint8_t> nfh2(static_cast<std::size_t>(n_new), 0);
    for (Eigen::Index i = 0; i < nhe0; ++i) {
        if (h_collapsed[static_cast<std::size_t>(i)]) continue;
        const u32 ni = he_remap[static_cast<std::size_t>(i)];
        he2(ni, 0) = v_remap[he(i, 0)];
        he2(ni, 1) = he_remap[he(i, 1)];
        he2(ni, 2) = he_remap[he(i, 2)];
        const u32 op = he(i, 3);
        he2(ni, 3) = (op == kInvalidU32) ? kInvalidU32 : he_remap[op];
        he2(ni, 4) = he(i, 4);
        nfh2[ni] = nfh[static_cast<std::size_t>(i)];
    }
    he  = std::move(he2);
    nfh = std::move(nfh2);

    // 4. Apply to xyz / normals / sizing / not_fixed_v.
    Coords      xyz2(n_v_new, 3);
    NormalsMat  nor2(n_v_new, 3);
    Eigen::VectorXd sz2(n_v_new);
    std::vector<std::uint8_t> nfv2(static_cast<std::size_t>(n_v_new), 0);
    for (Eigen::Index i = 0; i < nv0; ++i) {
        if (v_collapsed[static_cast<std::size_t>(i)]) continue;
        const u32 ni = v_remap[static_cast<std::size_t>(i)];
        xyz2.row(ni)  = xyz.row(i);
        nor2.row(ni)  = norm.row(i);
        sz2[ni]       = sz[i];
        nfv2[ni]      = nfv[static_cast<std::size_t>(i)];
    }
    xyz  = std::move(xyz2);
    norm = std::move(nor2);
    sz   = std::move(sz2);
    nfv  = std::move(nfv2);

    VOX2TET_LOG() << "collapse_edges: collapsed " << n_collapsed
                  << " edges (now " << he.rows() << " hedges, "
                  << xyz.rows() << " verts)";
}

// ---------------------------------------------------------------------------
// remesh — top-level driver. Mirrors mesh_processing.remesh.
// ---------------------------------------------------------------------------
RemeshResult remesh(const Settings& s,
                    RemeshState& st,
                    std::vector<marching_cubes::Interface>& interfaces) {
    const Eigen::Index n_xyz0 = st.xyz.rows();
    VOX2TET_LOG() << "Initial node count: " << n_xyz0;

    // Debug: log Euler characteristic after each phase when
    // V2T_REMESH_DEBUG_CHI is set. χ_all counts every xyz row (the way
    // the watertightness test does); χ_ref counts only vertices that are
    // referenced by some half-edge.
    const bool dbg_chi = std::getenv("V2T_REMESH_DEBUG_CHI") != nullptr;
    auto log_chi = [&st, dbg_chi](const char* phase) {
        if (!dbg_chi) return;
        const auto& he = st.hedges;
        std::set<std::pair<u32, u32>> edges;
        std::vector<std::uint8_t> used(static_cast<std::size_t>(st.xyz.rows()), 0);
        std::size_t n_nonmutual = 0, n_multi = 0;
        std::map<std::pair<u32, u32>, int> edge_mult;
        for (Eigen::Index i = 0; i < he.rows(); ++i) {
            u32 a = he(he(i, 2), 0), b = he(i, 0);
            used[a] = used[b] = 1;
            if (a > b) std::swap(a, b);
            edges.insert({a, b});
            ++edge_mult[{a, b}];
            const u32 op = he(i, 3);
            if (op != kInvalidU32 && he(op, 3) != static_cast<u32>(i)) ++n_nonmutual;
        }
        for (const auto& kv : edge_mult) if (kv.second > 2) ++n_multi;
        const long V = static_cast<long>(st.xyz.rows());
        const long Vr = static_cast<long>(
            std::count(used.begin(), used.end(), std::uint8_t{1}));
        const long E = static_cast<long>(edges.size());
        const long F = static_cast<long>(he.rows() / 3);
        VOX2TET_LOG() << "[chi] " << phase << ": V=" << V << " Vref=" << Vr
                      << " E=" << E << " F=" << F
                      << " chi_all=" << (V - E + F)
                      << " chi_ref=" << (Vr - E + F)
                      << " nonmutual_opp=" << n_nonmutual
                      << " multi_edges=" << n_multi;
    };
    log_chi("initial");

    // Debug gates for isolating remesh sub-steps. Set V2T_REMESH=skip_X
    // (X ∈ split, collapse, flip, tsmooth) to skip individual stages.
    const char* skip = std::getenv("V2T_REMESH_SKIP");
    auto should_skip = [skip](const char* name) {
        return skip != nullptr && std::strstr(skip, name) != nullptr;
    };

    for (int it = 0; it < s.n_remesh_itr; ++it) {
        VOX2TET_LOG() << "\n - \n Remesh iteration " << it;

        if (!should_skip("split")) {
            VOX2TET_PRINT("Start split edges");
            split_edges(st);
            log_chi("split");
        }

        if (!should_skip("collapse")) {
            VOX2TET_PRINT("Start collapse edges");
            collapse_edges(s, st, /*do_not_boundary=*/true, /*do_only_boundary=*/false);
            log_chi("collapse");
        }

        if (!should_skip("flip")) {
            VOX2TET_PRINT("Start flip edges");
            flip_half_edge(s, st.hedges, st.not_fixed_h, st.xyz, st.not_fixed_v,
                           /*do_not_boundary=*/false, /*do_only_boundary=*/false);
            log_chi("flip");
        }

        if (!should_skip("tsmooth")) {
            // Tangential smoothing + sharp-boundary revert. Pass the
            // blink cycle so the boundary-revert sees folds across
            // triple lines (Layer 1).
            Coords xyz_old = st.xyz;
            smooth_surfaces_tangential(st.hedges, st.not_fixed_h, st.xyz,
                                       st.not_fixed_v, st.normals, 1,
                                       s.min_dangle_internal);
            auto blinks = mesh::create_blinks(st.hedges, st.xyz);
            undo_sharp_bdangle_nodes(st.xyz, xyz_old, st.hedges,
                                     st.not_fixed_h, s.min_dangle_boundary,
                                     &blinks);
            // Step 2 (Layer 2 extension): corner-angle guard.
            undo_small_corner_nodes(st.xyz, xyz_old, st.hedges, st.not_fixed_v,
                                    s.min_corner_angle_internal,
                                    s.min_corner_angle_boundary);
        }
    }

    // --- Layer 3: dihedral repair (active fold remediation) -----------
    // After the remesh iterations converge, sweep multi-material edges
    // whose pair-wise minimum dihedral falls below s.min_dangle_internal
    // and try to widen each fold by relocating the apex of the
    // offending triangle. See src/remesh/dangle_repair.cpp.
    //
    // Gate behind V2T_REMESH_SKIP=repair so the pass can be disabled
    // for A/B comparisons; otherwise it runs by default.
    if (!should_skip("repair")) {
        // Use xyz_init = current xyz as the drift baseline. We allow a
        // drift of 1 voxel — generous, since the post-remesh smoothing
        // already enforces the per-iteration max_D2self.
        Coords xyz_pre_repair = st.xyz;
        auto rs = dangle_repair(s, st.hedges, st.xyz, xyz_pre_repair,
                                st.not_fixed_v,
                                /*min_dangle_threshold=*/s.min_dangle_internal,
                                /*max_drift=*/1.0,
                                /*max_passes=*/5);
        VOX2TET_LOG() << "dihedral_repair: bad edges " << rs.bad_edges_before
                      << " -> " << rs.bad_edges_after
                      << "  (passes=" << rs.passes_run
                      << "  attempts=" << rs.attempted_moves
                      << "  accepted=" << rs.accepted_moves << ")";
    }

    // --- Layer 4: sliver repair (active corner-angle remediation) ------
    // Sweep triangles whose smallest interior corner angle falls below
    // s.min_corner_angle_{internal,boundary} and try to widen that
    // corner by relocating the offending vertex toward the midpoint of
    // the opposite edge, subject to dihedral / drift / neighbour
    // corner-angle guards. See src/remesh/sliver_repair.cpp.
    //
    // Gate behind V2T_REMESH_SKIP=sliver to disable for A/B comparisons.
    if (!should_skip("sliver")) {
        Coords xyz_pre_sliver = st.xyz;
        auto rs2 = sliver_repair(s, st, xyz_pre_sliver,
                                 /*max_drift=*/1.0,
                                 /*max_passes=*/5);
        VOX2TET_LOG() << "sliver_repair: bad triangles "
                      << rs2.bad_triangles_before << " -> "
                      << rs2.bad_triangles_after
                      << "  (passes=" << rs2.passes_run
                      << "  attempts=" << rs2.flip_attempts
                      << "  accepted=" << rs2.flip_accepted << ")";
    }

    VOX2TET_LOG() << "Resulting node count: " << st.xyz.rows();

    auto triangles = mesh::hedges_to_triangles(st.hedges, &interfaces);

    // ----- Drop duplicated triangles (both copies) ------------------------
    // Mirrors the reference `removeDuplicatedTriangles(tr, do_remove_both=True)`
    // which is called for the per-grain STL save. We apply it once here
    // so the _RE_ALL.stl and the TetGen .smesh input are also clean.
    //
    // A "duplicate" is two triangles with the same sorted vertex tuple;
    // this happens when remesh produces a back-to-back fold (a triangle
    // and its opposite winding both present). TetGen rejects such inputs
    // with "duplicated triangle ... is ignored" + "self-intersections".
    {
        const Eigen::Index n0 = triangles.rows();
        struct TriRef { std::array<u32, 3> sorted; Eigen::Index idx; };
        std::vector<TriRef> refs(static_cast<std::size_t>(n0));
        for (Eigen::Index i = 0; i < n0; ++i) {
            std::array<u32, 3> a = {triangles(i, 0), triangles(i, 1), triangles(i, 2)};
            std::sort(a.begin(), a.end());
            refs[static_cast<std::size_t>(i)] = {a, i};
        }
        std::sort(refs.begin(), refs.end(),
            [](const TriRef& a, const TriRef& b) { return a.sorted < b.sorted; });

        std::vector<std::uint8_t> keep(static_cast<std::size_t>(n0), 1);
        std::size_t i = 0;
        std::size_t n_dup_pairs = 0;
        while (i < refs.size()) {
            std::size_t j = i + 1;
            while (j < refs.size() && refs[j].sorted == refs[i].sorted) ++j;
            if (j - i > 1) {
                for (std::size_t k = i; k < j; ++k) keep[refs[k].idx] = 0;
                n_dup_pairs += (j - i);
            }
            i = j;
        }
        if (n_dup_pairs > 0) {
            VOX2TET_LOG() << "remesh: removed " << n_dup_pairs
                          << " duplicated triangles (" << (n_dup_pairs / 2)
                          << " duplicate pairs)";

            // Compact triangles + rebuild interfaces' (first, count).
            Triangles tri_kept(n0 - static_cast<Eigen::Index>(n_dup_pairs), 3);
            Eigen::Index w = 0;
            std::vector<u32> tri_old_to_new(static_cast<std::size_t>(n0), kInvalidU32);
            for (Eigen::Index k = 0; k < n0; ++k) {
                if (!keep[static_cast<std::size_t>(k)]) continue;
                tri_kept.row(w) = triangles.row(k);
                tri_old_to_new[static_cast<std::size_t>(k)] = static_cast<u32>(w);
                ++w;
            }
            triangles = std::move(tri_kept);

            // Rebuild interfaces — each interface's [first, first+count)
            // run remains contiguous because removal preserves the
            // sort-by-interface order; we just need to recompute lengths.
            for (auto& it : interfaces) {
                u32 new_first = kInvalidU32;
                u32 new_count = 0;
                for (u32 t = it.first; t < it.first + it.count; ++t) {
                    if (keep[t]) {
                        if (new_first == kInvalidU32)
                            new_first = tri_old_to_new[t];
                        ++new_count;
                    }
                }
                it.first = (new_first == kInvalidU32) ? 0 : new_first;
                it.count = new_count;
            }
        }
    }

    RemeshResult R;
    R.surface_path = s.out_path_base + "_RE";
    R.triangles    = std::move(triangles);
    R.xyz          = st.xyz;
    return R;
}

}  // namespace vox2tet::remesh
