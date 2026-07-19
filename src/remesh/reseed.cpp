// Boundary-edge reseeding — Stages A (bbox frame) and A2 (curved
// chains: bbox-face traces, internal triple / quad lines).
// See include/vox2tet/remesh/reseed.hpp and doc/RESEEDING.md.
//
// Collapse of the chain edge (u = anchor, v), with v merged into u.
// The chain edge is shared by one triangle per incident sheet
// (multiplicity m: 2 on the frame, 3 on face traces and triple lines,
// 4 on quad lines). Per sheet:
//
//        a                     a
//       / \                   /|
//      /   \                 / |
//   --u-----v--w--   →   --u---w--     (chain runs u–v–w, apex a)
//
// All m triangles on (u,v) die; in each, the two side edges (v,a) and
// (a,u) merge, so their outward half-edge partners are stitched
// together. Every other half-edge pointing at v is retargeted to u.
// u never moves — surviving vertex coordinates are preserved
// bit-exactly, and because positions are static the AABB tree over the
// triangles only ever needs grow-only box refits.

#include "vox2tet/remesh/reseed.hpp"

#include "vox2tet/brep/brep.hpp"
#include "vox2tet/core/log.hpp"
#include "vox2tet/mesh/half_edge.hpp"

#include <Eigen/Geometry>
#include <algorithm>
#include <cmath>
#include <limits>
#include <queue>
#include <set>
#include <utility>
#include <vector>

namespace vox2tet::remesh {

// ===========================================================================
// reseed_detail: AABB tree + segment predicates (unit-tested directly).
// ===========================================================================
namespace reseed_detail {

void AabbTree::build(const std::vector<std::array<Eigen::Vector3d, 2>>& boxes) {
    nodes.clear();
    leaf_of.assign(boxes.size(), -1);
    root = -1;
    if (boxes.empty()) return;
    std::vector<int> items(boxes.size());
    for (std::size_t i = 0; i < boxes.size(); ++i) items[i] = static_cast<int>(i);
    root = build_range(items, 0, boxes.size(), boxes, -1);
}

int AabbTree::build_range(std::vector<int>& items, std::size_t first,
                          std::size_t last,
                          const std::vector<std::array<Eigen::Vector3d, 2>>& boxes,
                          int parent) {
    const int ni = static_cast<int>(nodes.size());
    nodes.emplace_back();
    nodes[ni].parent = parent;

    Eigen::Vector3d lo = boxes[static_cast<std::size_t>(items[first])][0];
    Eigen::Vector3d hi = boxes[static_cast<std::size_t>(items[first])][1];
    for (std::size_t i = first + 1; i < last; ++i) {
        lo = lo.cwiseMin(boxes[static_cast<std::size_t>(items[i])][0]);
        hi = hi.cwiseMax(boxes[static_cast<std::size_t>(items[i])][1]);
    }
    nodes[ni].lo = lo;
    nodes[ni].hi = hi;

    const std::size_t count = last - first;
    if (count <= 4) {
        for (std::size_t i = first; i < last; ++i) {
            nodes[ni].items.push_back(items[i]);
            leaf_of[static_cast<std::size_t>(items[i])] = ni;
        }
        return ni;
    }

    // Median split on the longest axis of the centroid spread.
    int axis = 0;
    {
        Eigen::Vector3d clo = Eigen::Vector3d::Constant(
            std::numeric_limits<double>::infinity());
        Eigen::Vector3d chi = -clo;
        for (std::size_t i = first; i < last; ++i) {
            const auto& b = boxes[static_cast<std::size_t>(items[i])];
            const Eigen::Vector3d c = 0.5 * (b[0] + b[1]);
            clo = clo.cwiseMin(c);
            chi = chi.cwiseMax(c);
        }
        const Eigen::Vector3d ext = chi - clo;
        if (ext[1] > ext[axis]) axis = 1;
        if (ext[2] > ext[axis]) axis = 2;
    }
    const std::size_t mid = first + count / 2;
    std::nth_element(items.begin() + static_cast<std::ptrdiff_t>(first),
                     items.begin() + static_cast<std::ptrdiff_t>(mid),
                     items.begin() + static_cast<std::ptrdiff_t>(last),
                     [&](int a, int b) {
                         const auto& ba = boxes[static_cast<std::size_t>(a)];
                         const auto& bb = boxes[static_cast<std::size_t>(b)];
                         return ba[0][axis] + ba[1][axis] <
                                bb[0][axis] + bb[1][axis];
                     });

    // Note: children are created after this node, so indices are stable.
    const int l = build_range(items, first, mid, boxes, ni);
    const int r = build_range(items, mid, last, boxes, ni);
    nodes[ni].left  = l;
    nodes[ni].right = r;
    return ni;
}

void AabbTree::grow_item(int item, const Eigen::Vector3d& lo,
                         const Eigen::Vector3d& hi) {
    int ni = leaf_of[static_cast<std::size_t>(item)];
    while (ni >= 0) {
        Node& n = nodes[static_cast<std::size_t>(ni)];
        const Eigen::Vector3d nlo = n.lo.cwiseMin(lo);
        const Eigen::Vector3d nhi = n.hi.cwiseMax(hi);
        if ((nlo - n.lo).norm() == 0.0 && (nhi - n.hi).norm() == 0.0) break;
        n.lo = nlo;
        n.hi = nhi;
        ni = n.parent;
    }
}

bool AabbTree::segment_touches_box(const Eigen::Vector3d& a,
                                   const Eigen::Vector3d& b,
                                   const Eigen::Vector3d& lo,
                                   const Eigen::Vector3d& hi) {
    constexpr double kPad = 1e-9;
    double t0 = 0.0, t1 = 1.0;
    const Eigen::Vector3d d = b - a;
    for (int k = 0; k < 3; ++k) {
        if (std::abs(d[k]) < 1e-15) {
            if (a[k] < lo[k] - kPad || a[k] > hi[k] + kPad) return false;
        } else {
            double ta = (lo[k] - kPad - a[k]) / d[k];
            double tb = (hi[k] + kPad - a[k]) / d[k];
            if (ta > tb) std::swap(ta, tb);
            t0 = std::max(t0, ta);
            t1 = std::min(t1, tb);
            if (t0 > t1) return false;
        }
    }
    return true;
}

bool segment_hits_triangle(const Eigen::Vector3d& a, const Eigen::Vector3d& b,
                           const Eigen::Vector3d& p0, const Eigen::Vector3d& p1,
                           const Eigen::Vector3d& p2) {
    // Möller–Trumbore, strict interior on both the triangle and the
    // segment. Parallel / coplanar → no hit (the normal-flip guard
    // owns the coplanar case on planar faces).
    constexpr double kBary = 1e-9;
    const Eigen::Vector3d dir = b - a;
    const Eigen::Vector3d e1  = p1 - p0;
    const Eigen::Vector3d e2  = p2 - p0;
    const Eigen::Vector3d pv  = dir.cross(e2);
    const double det = e1.dot(pv);
    // Scale-aware parallelism cut-off.
    const double scale = dir.norm() * e1.norm() * e2.norm();
    if (std::abs(det) < 1e-12 * std::max(scale, 1e-30)) return false;
    const double inv = 1.0 / det;
    const Eigen::Vector3d tv = a - p0;
    const double uu = tv.dot(pv) * inv;
    if (uu < kBary || uu > 1.0 - kBary) return false;
    const Eigen::Vector3d qv = tv.cross(e1);
    const double vv = dir.dot(qv) * inv;
    if (vv < kBary || uu + vv > 1.0 - kBary) return false;
    const double t = e2.dot(qv) * inv;
    return t > 1e-9 && t < 1.0 - 1e-9;
}

}  // namespace reseed_detail

// ===========================================================================
namespace {

using reseed_detail::AabbTree;
using reseed_detail::segment_hits_triangle;

// Distance from point p to segment [a, b].
double point_segment_dist(const Eigen::Vector3d& p,
                          const Eigen::Vector3d& a,
                          const Eigen::Vector3d& b) {
    const Eigen::Vector3d ab = b - a;
    const double L2 = ab.squaredNorm();
    if (L2 < 1e-24) return (p - a).norm();
    const double t = std::clamp((p - a).dot(ab) / L2, 0.0, 1.0);
    return (p - (a + t * ab)).norm();
}

// Min interior corner angle (degrees) of triangle (p0, p1, p2).
double min_corner_angle(const Eigen::Vector3d& p0,
                        const Eigen::Vector3d& p1,
                        const Eigen::Vector3d& p2) {
    auto corner = [](const Eigen::Vector3d& c,
                     const Eigen::Vector3d& x,
                     const Eigen::Vector3d& y) {
        const Eigen::Vector3d u1 = x - c;
        const Eigen::Vector3d u2 = y - c;
        const double n = u1.norm() * u2.norm();
        if (n < 1e-18) return 0.0;
        return std::acos(std::clamp(u1.dot(u2) / n, -1.0, 1.0)) * 180.0 / M_PI;
    };
    return std::min({corner(p0, p1, p2), corner(p1, p2, p0), corner(p2, p0, p1)});
}

// Dihedral (degrees) across edge (e1, e2) between the half-planes of
// apexes a and b: 180° = flat, small = sharp fold. Degenerate → 180°
// (permissive: never fires a guard on zero-area input).
double dihedral_deg(const Eigen::Vector3d& e1, const Eigen::Vector3d& e2,
                    const Eigen::Vector3d& a, const Eigen::Vector3d& b) {
    const Eigen::Vector3d v0 = e2 - e1;
    const Eigen::Vector3d c1 = (a - e1).cross(v0);
    const Eigen::Vector3d c2 = (b - e1).cross(v0);
    const double n = c1.norm() * c2.norm();
    if (n < 1e-18) return 180.0;
    return std::acos(std::clamp(c1.dot(c2) / n, -1.0, 1.0)) * 180.0 / M_PI;
}

// The collapse engine. Owns the mutable half-edge state plus alive
// flags, the vertex → incoming-half-edge lists (updated lazily: dead
// entries are filtered on read), and the AABB tree over triangles.
struct Collapser {
    mesh::HalfEdges&                          he;
    const Coords&                             xyz;
    const marching_cubes::NodeTypeMask&       mask;
    Eigen::Index                              ntr;    // triangle count; rows t, t+ntr, t+2ntr
    std::vector<std::uint8_t>                 alive_h;
    std::vector<std::uint8_t>                 alive_v;
    std::vector<std::vector<u32>>             v2he;   // half-edges whose target is v
    AabbTree                                  tree;

    Collapser(mesh::HalfEdges& he_, const Coords& xyz_,
              const marching_cubes::NodeTypeMask& mask_)
        : he(he_), xyz(xyz_), mask(mask_), ntr(he_.rows() / 3) {
        alive_h.assign(static_cast<std::size_t>(he.rows()), 1);
        alive_v.assign(static_cast<std::size_t>(xyz.rows()), 1);
        v2he.assign(static_cast<std::size_t>(xyz.rows()), {});
        for (Eigen::Index i = 0; i < he.rows(); ++i)
            v2he[he(i, 0)].push_back(static_cast<u32>(i));

        std::vector<std::array<Eigen::Vector3d, 2>> boxes(
            static_cast<std::size_t>(ntr));
        for (Eigen::Index t = 0; t < ntr; ++t)
            boxes[static_cast<std::size_t>(t)] = tri_box(static_cast<int>(t));
        tree.build(boxes);
    }

    u32 src(u32 h) const { return he(he(h, 2), 0); }

    void tri_verts(int t, u32 out[3]) const {
        out[0] = he(t, 0);
        out[1] = he(t + ntr, 0);
        out[2] = he(t + 2 * ntr, 0);
    }

    std::array<Eigen::Vector3d, 2> tri_box(int t) const {
        u32 vv[3];
        tri_verts(t, vv);
        Eigen::Vector3d lo = xyz.row(vv[0]).transpose();
        Eigen::Vector3d hi = lo;
        for (int k = 1; k < 3; ++k) {
            const Eigen::Vector3d p = xyz.row(vv[k]).transpose();
            lo = lo.cwiseMin(p);
            hi = hi.cwiseMax(p);
        }
        return {lo, hi};
    }

    int edge_multiplicity(u32 u, u32 v) const {
        int c = 0;
        for (u32 x : v2he[v]) if (alive_h[x] && src(x) == u) ++c;
        for (u32 x : v2he[u]) if (alive_h[x] && src(x) == v) ++c;
        return c;
    }

    std::set<u32> neighbours(u32 v) const {
        std::set<u32> nb;
        for (u32 x : v2he[v]) if (alive_h[x]) nb.insert(src(x));
        return nb;
    }

    bool tri_on_bbox(u32 a, u32 b, u32 c) const {
        const auto& m3 = mask.masks[3];
        return m3[a] && m3[b] && m3[c];
    }

    // AABB guard: does the chord u–w strictly cross any live triangle
    // not containing u, v, or w?
    bool chord_blocked(u32 u, u32 v, u32 w) const {
        const Eigen::Vector3d a = xyz.row(u).transpose();
        const Eigen::Vector3d b = xyz.row(w).transpose();
        bool hit = false;
        tree.for_segment(a, b, [&](int t) {
            if (hit) return;
            if (!alive_h[static_cast<std::size_t>(t)]) return;
            u32 vv[3];
            tri_verts(t, vv);
            for (u32 x : vv)
                if (x == u || x == v || x == w) return;
            if (segment_hits_triangle(a, b,
                                      xyz.row(vv[0]).transpose(),
                                      xyz.row(vv[1]).transpose(),
                                      xyz.row(vv[2]).transpose()))
                hit = true;
        });
        return hit;
    }

    // Try to collapse v into u. `removed` holds the positions of the
    // vertices already merged onto the chord ending at u; w is v's next
    // chain neighbour, so the prospective chord is u–w. Returns true on
    // an accepted (and applied) collapse.
    bool try_collapse(const Settings& s, u32 u, u32 v, u32 w,
                      const std::vector<Eigen::Vector3d>& removed,
                      ReseedStats& R) {
        // ---- topology guards -----------------------------------------
        // Gather every live half-edge on the chain edge (u, v): one
        // dying triangle per incident sheet.
        std::vector<u32> on_edge;
        for (u32 x : v2he[v]) if (alive_h[x] && src(x) == u) on_edge.push_back(x);
        for (u32 x : v2he[u]) if (alive_h[x] && src(x) == v) on_edge.push_back(x);
        const std::size_t m = on_edge.size();
        if (m < 2 || m > 4) { ++R.rej_topology; return false; }

        struct Dying {
            u32 rows[3];
            u32 apex;
            u32 side_v, side_u;      // rows on edges (v, apex), (u, apex)
        };
        std::vector<Dying> dying(m);
        std::vector<u32>   dying_rows;
        dying_rows.reserve(3 * m);
        for (std::size_t i = 0; i < m; ++i) {
            const u32 h  = on_edge[i];
            const u32 hn = he(h, 1), hp = he(h, 2);
            Dying& D = dying[i];
            D.rows[0] = h; D.rows[1] = hn; D.rows[2] = hp;
            D.apex = he(hn, 0);
            if (he(h, 0) == v) { D.side_v = hn; D.side_u = hp; }
            else               { D.side_u = hn; D.side_v = hp; }
            dying_rows.insert(dying_rows.end(), {h, hn, hp});
        }
        auto is_dying = [&](u32 x) {
            return std::find(dying_rows.begin(), dying_rows.end(), x)
                   != dying_rows.end();
        };

        // Apexes distinct and disjoint from the edge.
        std::vector<u32> apexes;
        for (const auto& D : dying) apexes.push_back(D.apex);
        std::sort(apexes.begin(), apexes.end());
        if (std::adjacent_find(apexes.begin(), apexes.end()) != apexes.end() ||
            std::find(apexes.begin(), apexes.end(), u) != apexes.end() ||
            std::find(apexes.begin(), apexes.end(), v) != apexes.end()) {
            ++R.rej_topology;
            return false;
        }

        // Side edges of every dying triangle must be manifold with
        // clean mutual opposites (a side edge that is itself a chain
        // edge — junction adjacency — rejects the collapse).
        std::vector<std::array<u32, 2>> stitches(m);   // partners to pair up
        for (std::size_t i = 0; i < m; ++i) {
            const Dying& D = dying[i];
            if (edge_multiplicity(v, D.apex) != 2 ||
                edge_multiplicity(u, D.apex) != 2) { ++R.rej_topology; return false; }
            const u32 pv = he(D.side_v, 3);
            const u32 pu = he(D.side_u, 3);
            if (pv == kInvalidU32 || pu == kInvalidU32 ||
                !alive_h[pv] || !alive_h[pu] ||
                he(pv, 3) != D.side_v || he(pu, 3) != D.side_u ||
                is_dying(pv) || is_dying(pu)) { ++R.rej_topology; return false; }
            stitches[i] = {pv, pu};
        }

        // Link condition: the common neighbours of u and v must be
        // exactly the apex set — anything else would pinch the mesh.
        {
            const auto nu = neighbours(u);
            const auto nv = neighbours(v);
            std::vector<u32> common;
            std::set_intersection(nu.begin(), nu.end(), nv.begin(), nv.end(),
                                  std::back_inserter(common));
            if (common != apexes) { ++R.rej_topology; return false; }
        }

        // ---- geometry guard: chordal deviation -----------------------
        const Eigen::Vector3d pu_ = xyz.row(u).transpose();
        const Eigen::Vector3d pw  = xyz.row(w).transpose();
        const Eigen::Vector3d pv_ = xyz.row(v).transpose();
        {
            double dev = point_segment_dist(pv_, pu_, pw);
            for (const auto& p : removed)
                dev = std::max(dev, point_segment_dist(p, pu_, pw));
            if (dev > s.reseed_eps) { ++R.rej_geometry; return false; }
        }

        // ---- quality guard: surviving triangles around v -------------
        for (u32 x : v2he[v]) {
            if (!alive_h[x] || is_dying(x)) continue;
            const u32 va = src(x);              // triangle (va, v, vb)
            const u32 vb = he(he(x, 1), 0);
            const Eigen::Vector3d pa = xyz.row(va).transpose();
            const Eigen::Vector3d pb = xyz.row(vb).transpose();

            const Eigen::Vector3d n_old = (pv_ - pa).cross(pb - pv_);
            const Eigen::Vector3d n_new = (pu_ - pa).cross(pb - pu_);
            if (n_new.norm() < 1e-12) { ++R.rej_quality; return false; }
            if (n_new.dot(n_old) <= 0.0) { ++R.rej_quality; return false; }

            const double thr = tri_on_bbox(va, u, vb)
                                   ? s.min_corner_angle_boundary
                                   : s.min_corner_angle_internal;
            const double ang_old = min_corner_angle(pa, pv_, pb);
            const double ang_new = min_corner_angle(pa, pu_, pb);
            if (ang_new < thr - 1e-9 && ang_new < ang_old - 1e-9) {
                ++R.rej_quality;
                return false;
            }
        }

        // ---- quality guard: fold across the new chord ----------------
        // The triangles sharing the chain edge (v, w) become the
        // triangles around the chord (u, w); their pairwise dihedral
        // must not fold below the threshold unless it already was.
        {
            std::vector<u32> chord_apexes;
            bool on_bbox = mask.masks[3][u] && mask.masks[3][w];
            auto collect = [&](u32 tgt, u32 from) {
                for (u32 x : v2he[tgt]) {
                    if (!alive_h[x] || is_dying(x) || src(x) != from) continue;
                    const u32 c = he(he(x, 1), 0);
                    chord_apexes.push_back(c);
                    on_bbox = on_bbox && mask.masks[3][c];
                }
            };
            collect(w, v);   // halves v → w
            collect(v, w);   // halves w → v
            if (chord_apexes.size() >= 2) {
                const double thr = on_bbox ? s.min_dangle_boundary
                                           : s.min_dangle_internal;
                double before = 180.0, after = 180.0;
                for (std::size_t i = 0; i < chord_apexes.size(); ++i) {
                    for (std::size_t j = i + 1; j < chord_apexes.size(); ++j) {
                        const Eigen::Vector3d ca =
                            xyz.row(chord_apexes[i]).transpose();
                        const Eigen::Vector3d cb =
                            xyz.row(chord_apexes[j]).transpose();
                        before = std::min(before, dihedral_deg(pv_, pw, ca, cb));
                        after  = std::min(after,  dihedral_deg(pu_, pw, ca, cb));
                    }
                }
                if (after < thr - 1e-9 && after < before - 1e-9) {
                    ++R.rej_quality;
                    return false;
                }
            }
        }

        // ---- intersection guard (AABB) -------------------------------
        if (chord_blocked(u, v, w)) { ++R.rej_intersection; return false; }

        // ---- apply ---------------------------------------------------
        // Retarget the surviving half-edges pointing at v; remember the
        // affected triangles for the grow-only box refit.
        std::vector<int> touched;
        for (u32 x : v2he[v]) {
            if (!alive_h[x] || is_dying(x)) continue;
            he(x, 0) = u;
            v2he[u].push_back(x);
            touched.push_back(static_cast<int>(x % static_cast<u32>(ntr)));
        }
        // Stitch the side-edge pairs of every dying triangle.
        for (const auto& st : stitches) {
            he(st[0], 3) = st[1];
            he(st[1], 3) = st[0];
        }
        for (u32 x : dying_rows) alive_h[x] = 0;
        alive_v[v] = 0;
        R.tris_removed += m;

        for (int t : touched) {
            const auto box = tri_box(t);
            tree.grow_item(t, box[0], box[1]);
        }
        return true;
    }
};

// Chain decomposition for reseeding. `order_bedges` splits chains at
// every masks[2] (voxel-corner) node — on curved traces and triple
// lines those appear at every staircase turn, which would fragment the
// chains into unreseedable stubs. For reseeding, only *true* junctions
// matter: nodes whose boundary-graph degree differs from 2 (chain
// crossings, quad points, bbox corners, chain end-points). Degree-2
// voxel-corner nodes are staircase artifacts and become removable
// interior nodes — the chordal-deviation guard decides how far the
// corner may be cut (≤ reseed_eps).
//
// Open chains run junction → junction; components with no junction are
// emitted as closed loops (first == last).
std::vector<std::vector<u32>> build_reseed_chains(const brep::BEdges& bedges) {
    std::vector<std::vector<u32>> chains;
    if (bedges.rows() == 0) return chains;

    u32 max_v = 0;
    for (Eigen::Index i = 0; i < bedges.rows(); ++i)
        max_v = std::max({max_v, bedges(i, 0), bedges(i, 1)});

    // Adjacency as (neighbour, edge id) pairs.
    std::vector<std::vector<std::pair<u32, u32>>> adj(max_v + 1);
    for (Eigen::Index e = 0; e < bedges.rows(); ++e) {
        const u32 a = bedges(e, 0), b = bedges(e, 1);
        adj[a].push_back({b, static_cast<u32>(e)});
        adj[b].push_back({a, static_cast<u32>(e)});
    }

    std::vector<std::uint8_t> edge_used(static_cast<std::size_t>(bedges.rows()), 0);
    auto is_junction = [&](u32 v) { return adj[v].size() != 2; };

    auto walk = [&](u32 start, u32 first_nb, u32 first_edge) {
        std::vector<u32> chain{start};
        u32 cur = first_nb;
        edge_used[first_edge] = 1;
        chain.push_back(cur);
        while (!is_junction(cur) && cur != start) {
            u32 nxt = kInvalidU32, nxt_e = kInvalidU32;
            for (const auto& [nb, e] : adj[cur]) {
                if (edge_used[e]) continue;
                nxt = nb;
                nxt_e = e;
                break;
            }
            if (nxt == kInvalidU32) break;   // dead end (degree-1 node)
            edge_used[nxt_e] = 1;
            chain.push_back(nxt);
            cur = nxt;
        }
        return chain;
    };

    // Junction-anchored open chains first.
    for (u32 v = 0; v <= max_v; ++v) {
        if (adj[v].empty() || !is_junction(v)) continue;
        for (const auto& [nb, e] : adj[v]) {
            if (edge_used[e]) continue;
            chains.push_back(walk(v, nb, e));
        }
    }
    // Remaining unvisited edges belong to pure degree-2 loops.
    for (Eigen::Index e = 0; e < bedges.rows(); ++e) {
        if (edge_used[e]) continue;
        chains.push_back(walk(bedges(e, 0), bedges(e, 1),
                              static_cast<u32>(e)));
    }
    return chains;
}

}  // namespace

// ---------------------------------------------------------------------------
// Stage B: composite local-feature-size targets (see reseed.hpp).
// ---------------------------------------------------------------------------
namespace reseed_detail {

ChainLfs compute_chain_lfs(const Triangles& tri, const Coords& xyz,
                           const brep::BEdges& bedges,
                           double target, double beta, double grading,
                           double h_min, double collar) {
    const double inf = std::numeric_limits<double>::infinity();
    const Eigen::Index N = xyz.rows();
    ChainLfs out;
    out.h.assign(static_cast<std::size_t>(N), target);
    out.lfs.assign(static_cast<std::size_t>(N), inf);
    if (bedges.rows() == 0 || N == 0) return out;

    // Chain flags + chain graph (edge-length weighted).
    std::vector<std::uint8_t> is_chain(static_cast<std::size_t>(N), 0);
    std::vector<std::vector<std::pair<u32, double>>> cadj(
        static_cast<std::size_t>(N));
    for (Eigen::Index e = 0; e < bedges.rows(); ++e) {
        const u32 a = bedges(e, 0), b = bedges(e, 1);
        const double w = (xyz.row(a) - xyz.row(b)).norm();
        is_chain[a] = is_chain[b] = 1;
        cadj[a].emplace_back(b, w);
        cadj[b].emplace_back(a, w);
    }
    std::vector<u32> cverts;
    for (Eigen::Index i = 0; i < N; ++i)
        if (is_chain[static_cast<std::size_t>(i)])
            cverts.push_back(static_cast<u32>(i));

    // Surface graph over all triangle edges; multi-source Dijkstra from
    // the chain set gives every vertex its graph-geodesic distance to
    // the nearest chain vertex — the "skirt depth" used to build the
    // d_surf collar.
    std::vector<std::vector<std::pair<u32, double>>> sadj(
        static_cast<std::size_t>(N));
    for (Eigen::Index t = 0; t < tri.rows(); ++t)
        for (int k = 0; k < 3; ++k) {
            const u32 a = tri(t, k), b = tri(t, (k + 1) % 3);
            const double w = (xyz.row(a) - xyz.row(b)).norm();
            sadj[a].emplace_back(b, w);
            sadj[b].emplace_back(a, w);
        }
    std::vector<double> skirt(static_cast<std::size_t>(N), inf);
    {
        using QE = std::pair<double, u32>;
        std::priority_queue<QE, std::vector<QE>, std::greater<QE>> pq;
        for (u32 c : cverts) { skirt[c] = 0.0; pq.emplace(0.0, c); }
        while (!pq.empty()) {
            const auto [d, u] = pq.top();
            pq.pop();
            if (d > skirt[u]) continue;
            for (const auto& [v, w] : sadj[u]) {
                if (d + w < skirt[v]) { skirt[v] = d + w; pq.emplace(d + w, v); }
            }
        }
    }

    // d_surf per chain vertex: nearest surface vertex whose skirt depth
    // exceeds collar * Euclidean distance (i.e. intrinsically far —
    // a genuinely different sheet approach, not this chain's own skirt).
    // Brute force like calc_brep_sizing; independent per query.
#ifdef VOX2TET_HAS_OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (std::size_t qi = 0; qi < cverts.size(); ++qi) {
        const u32 p = cverts[qi];
        const Eigen::Vector3d pp = xyz.row(p).transpose();
        double best = inf;
        for (Eigen::Index v = 0; v < N; ++v) {
            if (is_chain[static_cast<std::size_t>(v)]) continue;
            const double d = (xyz.row(v).transpose() - pp).norm();
            if (d >= best) continue;
            if (skirt[static_cast<std::size_t>(v)] > collar * d) best = d;
        }
        out.lfs[p] = best;
    }

    // d_chains / d_self per chain vertex: bounded Dijkstra over the
    // chain graph gives the along-chain distance; a chain vertex q
    // counts when it is along-chain far (geo > collar * |p-q|) or in a
    // different component (unreached). The cutoff only needs to cover
    // candidates that could win: geo <= collar * d with d < target.
    {
        const double cutoff = collar * target;
        std::vector<double> geo(static_cast<std::size_t>(N), inf);
        std::vector<u32> touched;
        for (const u32 p : cverts) {
            using QE = std::pair<double, u32>;
            std::priority_queue<QE, std::vector<QE>, std::greater<QE>> pq;
            geo[p] = 0.0;
            touched.push_back(p);
            pq.emplace(0.0, p);
            while (!pq.empty()) {
                const auto [d, u] = pq.top();
                pq.pop();
                if (d > geo[u] || d > cutoff) continue;
                for (const auto& [v, w] : cadj[u]) {
                    if (d + w < geo[v]) {
                        if (geo[v] == inf) touched.push_back(v);
                        geo[v] = d + w;
                        pq.emplace(d + w, v);
                    }
                }
            }
            const Eigen::Vector3d pp = xyz.row(p).transpose();
            double best = out.lfs[p];
            for (const u32 q : cverts) {
                if (q == p) continue;
                const double d = (xyz.row(q).transpose() - pp).norm();
                if (d >= best) continue;
                if (geo[q] > collar * d) best = d;
            }
            out.lfs[p] = best;
            for (const u32 t : touched) geo[t] = inf;
            touched.clear();
        }
    }

    // h = clamp(beta * lfs, h_min, target).
    for (const u32 p : cverts) {
        const double raw = beta * out.lfs[p];
        out.h[p] = std::clamp(std::isfinite(raw) ? raw : target, h_min, target);
    }

    // Gradient limit along the chain graph, slope (grading - 1): small
    // graph, Bellman-Ford sweeps to fixpoint.
    const double slope = grading - 1.0;
    if (slope > 0.0) {
        bool changed = true;
        while (changed) {
            changed = false;
            for (Eigen::Index e = 0; e < bedges.rows(); ++e) {
                const u32 a = bedges(e, 0), b = bedges(e, 1);
                const double w =
                    slope * (xyz.row(a) - xyz.row(b)).norm();
                if (out.h[a] + w < out.h[b] - 1e-12) {
                    out.h[b] = out.h[a] + w;
                    changed = true;
                }
                if (out.h[b] + w < out.h[a] - 1e-12) {
                    out.h[a] = out.h[b] + w;
                    changed = true;
                }
            }
        }
    }
    return out;
}

}  // namespace reseed_detail

// ---------------------------------------------------------------------------
ReseedStats reseed_feature_chains(const Settings& s,
                                  Triangles& tri,
                                  Coords& xyz,
                                  marching_cubes::NodeTypeMask& node_mask,
                                  std::vector<marching_cubes::Interface>& interfaces) {
    ReseedStats R;
    R.verts_before = static_cast<std::size_t>(xyz.rows());

    const double target =
        (s.reseed_target_len > 0.0) ? s.reseed_target_len : s.Lmax;

    auto he     = mesh::triangles_to_hedges(tri, &interfaces);
    auto bedges = brep::get_boundary_edges(tri, node_mask);
    auto chains = build_reseed_chains(bedges);

    // Stage B: per-vertex spacing targets from the composite local
    // feature size (h == target everywhere when disabled).
    reseed_detail::ChainLfs L;
    if (s.do_reseed_lfs) {
        L = reseed_detail::compute_chain_lfs(tri, xyz, bedges, target,
                                             s.reseed_beta,
                                             s.reseed_grading);
    } else {
        L.h.assign(static_cast<std::size_t>(xyz.rows()), target);
    }

    Collapser C(he, xyz, node_mask);

    const auto& on_frame = node_mask.masks[4];
    for (const auto& chain : chains) {
        if (chain.size() < 3) continue;
        const bool is_loop = (chain.front() == chain.back());
        bool frame = true;
        for (u32 v : chain) frame = frame && (v < on_frame.size() && on_frame[v]);
        if (!frame && !s.do_reseed_triple_lines) continue;
        if (is_loop) {
            if (!s.do_reseed_triple_lines) continue;
            ++R.loop_chains;
        } else if (frame) {
            ++R.frame_chains;
        } else {
            ++R.curved_chains;
        }

        // A closed loop repeats its first vertex at the end; keep at
        // least 4 surviving vertices so the ring can never pinch.
        const std::size_t unique_nodes = is_loop ? chain.size() - 1
                                                 : chain.size();
        std::size_t survivors = unique_nodes;

        u32 anchor = chain.front();
        // Stage B running cap: the chord u–w may not exceed the lfs
        // target h of ANY vertex it replaces (anchor, the removed span,
        // and w) — "segment length ≤ local h(t)" from the plan. With
        // do_reseed_lfs off every h is `target` and this reduces to
        // the old global cap.
        double run_h = L.h[anchor];
        std::vector<Eigen::Vector3d> removed;
        for (std::size_t i = 1; i + 1 < chain.size(); ++i) {
            const u32 v = chain[i];
            const u32 w = chain[i + 1];
            if (is_loop && survivors <= 4) break;
            const double cap = std::min({run_h, L.h[v], L.h[w]});
            const double chord = (xyz.row(anchor) - xyz.row(w)).norm();
            if (chord > cap) {
                if (chord <= target) ++R.lfs_limited;
                anchor = v;
                run_h  = L.h[v];
                removed.clear();
                continue;
            }
            ++R.collapses_attempted;
            if (C.try_collapse(s, anchor, v, w, removed, R)) {
                ++R.collapses_accepted;
                ++R.verts_removed;
                --survivors;
                run_h = std::min(run_h, L.h[v]);
                removed.push_back(xyz.row(v).transpose());
            } else {
                anchor = v;
                run_h  = L.h[v];
                removed.clear();
            }
        }
    }

    VOX2TET_LOG() << "reseed_feature_chains: chains frame/curved/loop "
                  << R.frame_chains << "/" << R.curved_chains << "/"
                  << R.loop_chains << ", " << R.collapses_accepted << "/"
                  << R.collapses_attempted << " collapses accepted (rej topo "
                  << R.rej_topology << ", geom " << R.rej_geometry
                  << ", qual " << R.rej_quality << ", isect "
                  << R.rej_intersection << "), " << R.verts_removed << " of "
                  << R.verts_before << " vertices removed"
                  << (s.do_reseed_lfs
                          ? (", lfs capped " + std::to_string(R.lfs_limited)
                             + " spans")
                          : std::string());

    if (R.verts_removed == 0) return R;

    // ---- compact vertices --------------------------------------------
    const Eigen::Index n_v = xyz.rows();
    std::vector<u32> vmap(static_cast<std::size_t>(n_v), kInvalidU32);
    Eigen::Index n_v_new = 0;
    for (Eigen::Index i = 0; i < n_v; ++i)
        if (C.alive_v[static_cast<std::size_t>(i)])
            vmap[static_cast<std::size_t>(i)] = static_cast<u32>(n_v_new++);

    Coords xyz_new(n_v_new, 3);
    marching_cubes::NodeTypeMask mask_new;
    for (auto& m : mask_new.masks) m.assign(static_cast<std::size_t>(n_v_new), 0);
    for (Eigen::Index i = 0; i < n_v; ++i) {
        const u32 ni = vmap[static_cast<std::size_t>(i)];
        if (ni == kInvalidU32) continue;
        xyz_new.row(ni) = xyz.row(i);
        for (int m = 0; m < 8; ++m) {
            if (static_cast<std::size_t>(i) < node_mask.masks[m].size())
                mask_new.masks[m][ni] = node_mask.masks[m][i];
        }
    }

    // ---- compact half-edges ------------------------------------------
    const Eigen::Index n_h = he.rows();
    std::vector<u32> rmap(static_cast<std::size_t>(n_h), kInvalidU32);
    Eigen::Index n_h_new = 0;
    for (Eigen::Index i = 0; i < n_h; ++i)
        if (C.alive_h[static_cast<std::size_t>(i)])
            rmap[static_cast<std::size_t>(i)] = static_cast<u32>(n_h_new++);

    mesh::HalfEdges he_new(n_h_new, 5);
    for (Eigen::Index i = 0; i < n_h; ++i) {
        const u32 ni = rmap[static_cast<std::size_t>(i)];
        if (ni == kInvalidU32) continue;
        he_new(ni, 0) = vmap[he(i, 0)];
        he_new(ni, 1) = rmap[he(i, 1)];
        he_new(ni, 2) = rmap[he(i, 2)];
        he_new(ni, 3) = (he(i, 3) == kInvalidU32) ? kInvalidU32 : rmap[he(i, 3)];
        he_new(ni, 4) = he(i, 4);
    }

    // Rebuild the interface-grouped triangle list; (first, count) ranges
    // in `interfaces` are updated by hedges_to_triangles.
    tri = mesh::hedges_to_triangles(he_new, &interfaces);
    xyz = std::move(xyz_new);
    node_mask = std::move(mask_new);
    return R;
}

}  // namespace vox2tet::remesh
