#include "vox2tet/remesh/smooth.hpp"

#include "vox2tet/core/log.hpp"
#include "vox2tet/io/mesh_io.hpp"
#include "vox2tet/mesh/mathx.hpp"

#include <Eigen/Geometry>  // for Vector3d::cross
#include <algorithm>
#include <cmath>
#include <unordered_map>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace vox2tet::remesh {

namespace {

constexpr std::uint32_t kInvalidU32 = std::numeric_limits<std::uint32_t>::max();

}  // namespace

// ---------------------------------------------------------------------------
Edges2 triangles_to_edges(const Triangles& tri) {
    const Eigen::Index n_tri = tri.rows();
    Edges2 e(n_tri * 3, 2);
    for (Eigen::Index i = 0; i < n_tri; ++i) {
        e(i,            0) = tri(i, 0); e(i,            1) = tri(i, 1);
        e(i + n_tri,    0) = tri(i, 1); e(i + n_tri,    1) = tri(i, 2);
        e(i + 2*n_tri,  0) = tri(i, 2); e(i + 2*n_tri,  1) = tri(i, 0);
    }
    return e;
}

// ---------------------------------------------------------------------------
Eigen::VectorXd calc_dihedral(const mesh::HalfEdges& he, const Coords& xyz,
                              const std::vector<std::uint32_t>& i1,
                              const std::vector<std::uint32_t>& i2,
                              bool directional) {
    const std::size_t N = i1.size();
    if (N != i2.size()) throw std::runtime_error("calc_dihedral: i1/i2 size mismatch");

    MatrixNx3<double> p01(N, 3), p02(N, 3), p1(N, 3), p2(N, 3);
    for (std::size_t k = 0; k < N; ++k) {
        // v1 = [prev.target, target, next.target] for half-edge i1[k]
        // v2 = [prev.target, target, next.target] for half-edge i2[k]
        const std::uint32_t h1 = i1[k];
        const std::uint32_t h2 = i2[k];
        const std::uint32_t v1_0 = he(he(h1, 2), 0);
        const std::uint32_t v1_1 = he(h1, 0);
        const std::uint32_t v1_2 = he(he(h1, 1), 0);
        const std::uint32_t v2_2 = he(he(h2, 1), 0);
        p01.row(static_cast<Eigen::Index>(k)) = xyz.row(v1_0);
        p02.row(static_cast<Eigen::Index>(k)) = xyz.row(v1_1);
        p1.row (static_cast<Eigen::Index>(k)) = xyz.row(v1_2);
        p2.row (static_cast<Eigen::Index>(k)) = xyz.row(v2_2);
    }
    return directional ? mathx::calc_dihedral_angle_v_directional(p01, p02, p1, p2)
                       : mathx::calc_dihedral_angle_v(p01, p02, p1, p2);
}

// ---------------------------------------------------------------------------
InternalDihedrals calc_internal_dihedral(const mesh::HalfEdges& he,
                                         const std::vector<std::uint8_t>& not_fixed_h,
                                         const Coords& xyz) {
    InternalDihedrals R;
    for (Eigen::Index i = 0; i < he.rows(); ++i) {
        if (not_fixed_h[static_cast<std::size_t>(i)]) R.i1.push_back(static_cast<std::uint32_t>(i));
    }
    std::vector<std::uint32_t> i2(R.i1.size());
    for (std::size_t k = 0; k < R.i1.size(); ++k) i2[k] = he(R.i1[k], 3);
    R.dan = calc_dihedral(he, xyz, R.i1, i2);
    return R;
}

// ---------------------------------------------------------------------------
// Cycle-aware dihedral helpers — Layer 1 of the multi-material dihedral
// guard. See include/vox2tet/remesh/smooth.hpp for the rationale.
// ---------------------------------------------------------------------------

std::vector<std::vector<std::uint32_t>> build_cycle_mates(
        const mesh::HalfEdges& he,
        const std::vector<std::uint32_t>& blinks) {
    std::vector<std::vector<std::uint32_t>> mates(
        static_cast<std::size_t>(he.rows()));
    if (static_cast<Eigen::Index>(blinks.size()) != he.rows()) return mates;
    for (Eigen::Index i = 0; i < he.rows(); ++i) {
        const std::uint32_t b = blinks[static_cast<std::size_t>(i)];
        if (b == kInvalidU32) continue;
        std::uint32_t cur = b;
        // valence ≤ 4 in practice (quad point); guard at 8 to be safe.
        for (int steps = 0; steps < 8 && cur != static_cast<std::uint32_t>(i); ++steps) {
            mates[static_cast<std::size_t>(i)].push_back(cur);
            if (cur >= blinks.size()) break;
            cur = blinks[cur];
            if (cur == kInvalidU32) break;
        }
    }
    return mates;
}

namespace {
// Pair-wise dihedral around the canonical edge (he(prev,0) → he(h,0))
// between the triangle of h and the triangle of hp. Returns the same
// non-directional angle as mathx::calc_dihedral_angle_v: 180° = flat,
// 0° = full fold. Returns -1.0 on degenerate (zero-area) triangles.
double pair_dihedral_around_h(const mesh::HalfEdges& he, const Coords& xyz,
                              std::uint32_t h, std::uint32_t hp) {
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
    return std::acos(std::clamp(c1.dot(c2) / n12, -1.0, 1.0))
           * 180.0 / 3.141592653589793238;
}
}  // namespace

Eigen::VectorXd calc_min_dihedral_cycle(
        const mesh::HalfEdges& he, const Coords& xyz,
        const std::vector<std::uint32_t>& i1,
        const std::vector<std::vector<std::uint32_t>>& cycle_mates) {
    const std::size_t N = i1.size();
    Eigen::VectorXd out = Eigen::VectorXd::Constant(
        static_cast<Eigen::Index>(N), 180.0);
    for (std::size_t k = 0; k < N; ++k) {
        const std::uint32_t h = i1[k];
        if (h >= cycle_mates.size()) continue;
        const auto& m = cycle_mates[static_cast<std::size_t>(h)];
        double dm = 180.0;
        bool any = false;
        if (m.empty()) {
            // Manifold / boundary fallback: use canonical column-3 opposite.
            const std::uint32_t hp = he(h, 3);
            if (hp != kInvalidU32) {
                const double d = pair_dihedral_around_h(he, xyz, h, hp);
                if (d >= 0) { dm = d; any = true; }
            }
        } else {
            for (auto hp : m) {
                const double d = pair_dihedral_around_h(he, xyz, h, hp);
                if (d < 0) continue;
                if (d < dm) dm = d;
                any = true;
            }
        }
        out[static_cast<Eigen::Index>(k)] = any ? dm : 180.0;
    }
    return out;
}

InternalDihedrals calc_internal_dihedral_cycle(
        const mesh::HalfEdges& he,
        const std::vector<std::uint8_t>& not_fixed_h,
        const Coords& xyz,
        const std::vector<std::vector<std::uint32_t>>& cycle_mates) {
    InternalDihedrals R;
    R.i1.reserve(static_cast<std::size_t>(he.rows()));
    for (Eigen::Index i = 0; i < he.rows(); ++i) {
        if (not_fixed_h[static_cast<std::size_t>(i)])
            R.i1.push_back(static_cast<std::uint32_t>(i));
    }
    R.dan = calc_min_dihedral_cycle(he, xyz, R.i1, cycle_mates);
    return R;
}

// ---------------------------------------------------------------------------
void smooth_surfaces(const Triangles& tri, Coords& xyz,
                     const std::vector<std::uint8_t>& not_fixed,
                     int n_iter, double alpha) {
    VOX2TET_PRINT("Start surfaces smoothing...");

    auto edges = triangles_to_edges(tri);
    const Eigen::Index n_nodes = xyz.rows();

    // Adjacency: per source vertex, accumulated dst sum and count.
    // We model the legacy `coo_matrix((W[edges[b,1]], (edges[b,0], edges[b,1])))`
    // with W = 1 — so sum-by-row equals number of unique connections.
    // Note this DOES emit duplicates when an edge appears in two
    // triangles, matching the legacy coo behaviour.
    std::vector<std::vector<std::uint32_t>> adj(static_cast<std::size_t>(n_nodes));
    for (Eigen::Index i = 0; i < edges.rows(); ++i) {
        const std::uint32_t src = edges(i, 0);
        const std::uint32_t dst = edges(i, 1);
        if (not_fixed[src]) adj[src].push_back(dst);
    }

    Coords new_xyz = xyz;
    for (int it = 0; it < n_iter; ++it) {
        for (Eigen::Index v = 0; v < n_nodes; ++v) {
            if (!not_fixed[static_cast<std::size_t>(v)]) continue;
            const auto& nbrs = adj[static_cast<std::size_t>(v)];
            if (nbrs.empty()) continue;
            Eigen::Vector3d sum = Eigen::Vector3d::Zero();
            for (auto n : nbrs) sum += xyz.row(n).transpose();
            const double inv = 1.0 / static_cast<double>(nbrs.size());
            new_xyz.row(v) = alpha * xyz.row(v) + (1.0 - alpha) * (sum * inv).transpose();
        }
        // Commit (only not_fixed rows changed but copying all is fine).
        for (Eigen::Index v = 0; v < n_nodes; ++v)
            if (not_fixed[static_cast<std::size_t>(v)]) xyz.row(v) = new_xyz.row(v);
    }
    VOX2TET_PRINT("Smoothing surfaces done!");
}

// ---------------------------------------------------------------------------
NormalsMat calc_initial_vertex_normal(const Coords& xyz, const Triangles& tri,
                                      const std::vector<std::uint8_t>& is_internal) {
    VOX2TET_PRINT("Calc initial vertex normals...");
    const Eigen::Index n_nodes = xyz.rows();
    const Eigen::Index n_tri   = tri.rows();

    // Per-face un-normalised normal = (v1 - v0) × (v2 - v0).
    MatrixNx3<double> fnormals(n_tri, 3);
    for (Eigen::Index t = 0; t < n_tri; ++t) {
        const Eigen::Vector3d p0 = xyz.row(tri(t, 0)).transpose();
        const Eigen::Vector3d p1 = xyz.row(tri(t, 1)).transpose();
        const Eigen::Vector3d p2 = xyz.row(tri(t, 2)).transpose();
        // Materialise the operands so .cross() is the documented
        // Vector3d.cross(Vector3d) overload (Eigen's cross-on-expression
        // overload is declared but not defined in some builds).
        const Eigen::Vector3d a = p1 - p0;
        const Eigen::Vector3d b = p2 - p0;
        fnormals.row(t) = a.cross(b).transpose();
    }

    NormalsMat out(n_nodes, 3);
    out.setZero();

    // Accumulate face normals at each internal vertex.
    for (Eigen::Index t = 0; t < n_tri; ++t) {
        for (int k = 0; k < 3; ++k) {
            const std::uint32_t v = tri(t, k);
            if (!is_internal[v]) continue;
            out.row(v) += fnormals.row(t);
        }
    }
    // Normalise interior rows.
    for (Eigen::Index v = 0; v < n_nodes; ++v) {
        if (!is_internal[static_cast<std::size_t>(v)]) continue;
        const double L = out.row(v).norm();
        if (L > 0) out.row(v) /= L;
    }
    VOX2TET_PRINT("Calc initial vertex normal done!");
    return out;
}

// ---------------------------------------------------------------------------
void smooth_brep_mean(const Settings& s,
                      const std::vector<brep::EdgeChain>& chains,
                      Coords& xyz,
                      const Coords& xyz0,
                      int n_iter,
                      const Eigen::VectorXd* brep_sizing) {
    for (int it = 0; it < n_iter; ++it) {
        for (const auto& ed : chains) {
            if (ed.size() < 2) { VOX2TET_PRINT("ERROR: smoothBrepMean: ed.size < 2"); continue; }
            if (ed.size() < 4) continue;
            // Move interior nodes (skip first and last). For each, compute
            // alpha*xyz_i + (1-alpha) * 0.5 * (xyz_{i-1} + xyz_{i+1}).
            std::vector<Eigen::Vector3d> tmp(ed.size() - 2);
            std::vector<std::uint8_t>    is_close(ed.size() - 2, 1);
            for (std::size_t i = 1; i + 1 < ed.size(); ++i) {
                const std::uint32_t v_im1 = ed[i - 1];
                const std::uint32_t v_i   = ed[i];
                const std::uint32_t v_ip1 = ed[i + 1];
                const Eigen::Vector3d mid = 0.5 * (xyz.row(v_im1).transpose() + xyz.row(v_ip1).transpose());
                const Eigen::Vector3d t   = s.smooth_alpha * xyz.row(v_i).transpose()
                                          + (1.0 - s.smooth_alpha) * mid;
                tmp[i - 1] = t;
                const double d = (xyz0.row(v_i).transpose() - t).norm();
                is_close[i - 1] = (d < s.max_D2self) ? 1 : 0;
                if (brep_sizing) {
                    if ((*brep_sizing)[v_i] <= s.min_D2other) is_close[i - 1] = 0;
                }
            }
            // Commit.
            for (std::size_t i = 1; i + 1 < ed.size(); ++i) {
                if (!is_close[i - 1]) continue;
                xyz.row(ed[i]) = tmp[i - 1].transpose();
            }
        }
    }
}

// ---------------------------------------------------------------------------
Eigen::VectorXd calc_brep_sizing(const Coords& xyz,
                                 const std::vector<std::uint8_t>& is_brep) {
    const Eigen::Index N = xyz.rows();
    Eigen::VectorXd out = Eigen::VectorXd::Zero(N);

    // Indices of brep vertices.
    std::vector<std::uint32_t> brep_idx;
    brep_idx.reserve(static_cast<std::size_t>(N));
    for (Eigen::Index i = 0; i < N; ++i)
        if (is_brep[static_cast<std::size_t>(i)]) brep_idx.push_back(static_cast<std::uint32_t>(i));

    if (brep_idx.empty()) return out;

    // Brute force: for each query, scan all brep vertices. O(N * |B|).
    // For JMA_30 this is ~22k * 4k = 88M, well under a second. Upgrade
    // to nanoflann later if profiling shows a bottleneck. Parallelised
    // with OpenMP — each query is independent.
#ifdef VOX2TET_HAS_OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (Eigen::Index q = 0; q < N; ++q) {
        const Eigen::Vector3d p = xyz.row(q).transpose();
        double best = std::numeric_limits<double>::infinity();
        if (is_brep[static_cast<std::size_t>(q)]) {
            // Self-excluding nearest brep vertex.
            for (auto bi : brep_idx) {
                if (bi == static_cast<std::uint32_t>(q)) continue;
                const double d = (xyz.row(bi).transpose() - p).norm();
                if (d < best) best = d;
            }
        } else {
            for (auto bi : brep_idx) {
                const double d = (xyz.row(bi).transpose() - p).norm();
                if (d < best) best = d;
            }
        }
        out[q] = (std::isfinite(best)) ? best : 0.0;
    }
    return out;
}

// ---------------------------------------------------------------------------
// calc_closest_to_boundary — port of `calc_closest_to_boundary_h2`.
//
// For each interface, build the set of "candidate" other-interface
// vertices (those whose interface shares at least one material with
// this interface, minus the current one — and excluding exterior-only
// interfaces whose materials are > max_material - 6, the boundary
// labels added by ext_volume). Then for each vertex on this interface,
// find the nearest candidate by brute-force.
// ---------------------------------------------------------------------------
ClosestToBoundary calc_closest_to_boundary(
    const mesh::HalfEdges& he, const Coords& xyz,
    const std::vector<marching_cubes::Interface>& interfaces) {
    ClosestToBoundary R;
    const Eigen::Index N = xyz.rows();
    R.distances   = Eigen::VectorXd::Zero(N);
    R.closest_idx.assign(static_cast<std::size_t>(N), 0);

    if (interfaces.empty()) return R;

    // Determine mmax = max material id - 6 (the reference subtracts the 6
    // boundary-face labels added by ext_volume).
    std::uint32_t max_mat = 0;
    for (const auto& it : interfaces) {
        max_mat = std::max({max_mat, it.mat1, it.mat2});
    }
    const std::uint32_t mmax = (max_mat >= 6) ? (max_mat - 6) : 0;

    // For each interface i, vertices_on[i] = list of unique vertex ids
    // touched by any half-edge with interface column == i. This is the
    // the reference `i_v = np.unique(hedges[:,[4,0]], axis=0)` group-by-itf step.
    std::vector<std::vector<std::uint32_t>> vertices_on(interfaces.size());
    for (Eigen::Index k = 0; k < he.rows(); ++k) {
        const std::uint32_t itf_id = he(k, 4);
        if (itf_id == std::numeric_limits<std::uint32_t>::max()) continue;
        if (itf_id >= interfaces.size()) continue;
        vertices_on[itf_id].push_back(he(k, 0));
    }
    for (auto& v : vertices_on) {
        std::sort(v.begin(), v.end());
        v.erase(std::unique(v.begin(), v.end()), v.end());
    }

    // For each material m ≤ mmax, list of interface IDs that include m.
    std::unordered_map<std::uint32_t, std::vector<std::uint32_t>> grains;
    for (std::uint32_t i = 0; i < interfaces.size(); ++i) {
        const std::uint32_t m1 = interfaces[i].mat1;
        const std::uint32_t m2 = interfaces[i].mat2;
        if (m1 <= mmax) grains[m1].push_back(i);
        if (m2 <= mmax) grains[m2].push_back(i);
    }

    // For each interface, compute closest-distance per vertex.
    for (std::uint32_t i = 0; i < interfaces.size(); ++i) {
        const std::uint32_t m1 = interfaces[i].mat1;
        const std::uint32_t m2 = interfaces[i].mat2;

        // Build the candidate vertex pool: vertices on any interface that
        // shares m1 or m2 with this one, excluding interface i itself.
        std::vector<std::uint32_t> v_out;
        auto add_grain = [&](std::uint32_t m) {
            auto it = grains.find(m);
            if (it == grains.end()) return;
            for (auto other_i : it->second) {
                if (other_i == i) continue;
                v_out.insert(v_out.end(),
                              vertices_on[other_i].begin(),
                              vertices_on[other_i].end());
            }
        };
        if (m1 <= mmax) add_grain(m1);
        if (m2 <= mmax) add_grain(m2);
        if (v_out.empty()) continue;
        std::sort(v_out.begin(), v_out.end());
        v_out.erase(std::unique(v_out.begin(), v_out.end()), v_out.end());

        const auto& v_in = vertices_on[i];
#ifdef VOX2TET_HAS_OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for (std::size_t k = 0; k < v_in.size(); ++k) {
            const std::uint32_t q = v_in[k];
            const Eigen::Vector3d p = xyz.row(q).transpose();
            double best = std::numeric_limits<double>::infinity();
            std::uint32_t best_idx = 0;
            for (auto u : v_out) {
                const double d = (xyz.row(u).transpose() - p).norm();
                if (d < best) { best = d; best_idx = u; }
            }
            R.distances[q]   = best;
            R.closest_idx[q] = best_idx;
        }
    }
    return R;
}

// ---------------------------------------------------------------------------
Eigen::VectorXd calc_sizing_field(const mesh::HalfEdges& he,
                                  const Coords& xyz,
                                  const Eigen::VectorXd& brep_sizing,
                                  const std::vector<std::uint8_t>& not_fixed_v,
                                  const std::vector<marching_cubes::Interface>& interfaces,
                                  double Lmin, double Lmax) {
    VOX2TET_PRINT("Start sizing field calculation...");
    auto cb = calc_closest_to_boundary(he, xyz, interfaces);
    Eigen::VectorXd L = cb.distances;
    // Boundary verts (not_fixed_v == 0) take 1.7 * brep_sizing.
    for (Eigen::Index v = 0; v < xyz.rows(); ++v) {
        if (!not_fixed_v[static_cast<std::size_t>(v)]) L[v] = 1.7 * brep_sizing[v];
    }
    // "is_at_bound" half-edges: target is boundary, prev.target is not.
    // For each such hedge, copy L[target] into L[prev.target].
    for (Eigen::Index k = 0; k < he.rows(); ++k) {
        const std::uint32_t tgt  = he(k, 0);
        const std::uint32_t prev = he(he(k, 2), 0);
        if (!not_fixed_v[tgt] && not_fixed_v[prev]) {
            L[prev] = L[tgt];
        }
    }
    // Clip.
    for (Eigen::Index v = 0; v < L.size(); ++v) {
        if (L[v] < Lmin) L[v] = Lmin;
        if (L[v] > Lmax) L[v] = Lmax;
    }
    VOX2TET_PRINT("End sizing field calculation!");
    return L;
}

// ---------------------------------------------------------------------------
void undo_sharp_bdangle_nodes(Coords& xyz, const Coords& xyz0,
                              const mesh::HalfEdges& he,
                              const std::vector<std::uint8_t>& not_fixed_h,
                              double min_dangle_boundary,
                              const std::vector<std::uint32_t>* blinks) {
    // Selection of "boundary" half-edges = !not_fixed_h (the reference's
    // is_sharp_bhedges = ~not_fixed_h).
    std::vector<std::uint32_t> bhe;
    for (Eigen::Index i = 0; i < he.rows(); ++i) {
        if (!not_fixed_h[static_cast<std::size_t>(i)]) bhe.push_back(static_cast<std::uint32_t>(i));
    }
    if (bhe.empty()) return;

    // Two evaluation modes:
    //   (a) blinks == nullptr  : legacy path, uses canonical column-3
    //       opposite. Skips boundary half-edges (where col-3 is sentinel).
    //   (b) blinks != nullptr  : cycle-min path. For every selected
    //       half-edge, take the MIN dihedral across all blink-cycle
    //       partners. This is the only way to see folds across triple
    //       lines (3-way) and quad points (4-way), where the canonical
    //       opposite covers only ONE of the 2-3 adjacent triangle pairs.
    Eigen::VectorXd da0, da;
    std::vector<std::uint32_t> v1, v2;
    if (blinks == nullptr) {
        // Legacy: pair via column 3.
        v1.reserve(bhe.size());
        v2.reserve(bhe.size());
        for (std::size_t k = 0; k < bhe.size(); ++k) {
            const std::uint32_t hp = he(bhe[k], 3);
            if (hp == kInvalidU32 || hp >= static_cast<std::uint32_t>(he.rows())) continue;
            v1.push_back(bhe[k]);
            v2.push_back(hp);
        }
        if (v1.empty()) return;
        da0 = calc_dihedral(he, xyz0, v1, v2);
        da  = calc_dihedral(he, xyz,  v1, v2);
    } else {
        // Cycle-min: build mates once for this call.
        auto mates = build_cycle_mates(he, *blinks);
        v1.assign(bhe.begin(), bhe.end());
        da0 = calc_min_dihedral_cycle(he, xyz0, v1, mates);
        da  = calc_min_dihedral_cycle(he, xyz,  v1, mates);
        // For Layer-2 bidirectional revert (below), we still need a
        // "second vertex" per offending half-edge. Use the partner that
        // achieves the minimum dihedral; fallback to column-3 opposite.
        v2.assign(bhe.size(), kInvalidU32);
        for (std::size_t k = 0; k < bhe.size(); ++k) {
            const std::uint32_t h = bhe[k];
            const auto& m = mates[static_cast<std::size_t>(h)];
            if (m.empty()) {
                v2[k] = he(h, 3);
                continue;
            }
            double mn = 1e9;
            std::uint32_t arg = m.front();
            for (auto hp : m) {
                const double d = pair_dihedral_around_h(he, xyz, h, hp);
                if (d >= 0 && d < mn) { mn = d; arg = hp; }
            }
            v2[k] = arg;
        }
    }

    std::vector<std::uint8_t> is_sharp(static_cast<std::size_t>(he.rows()), 0);
    for (std::size_t k = 0; k < v1.size(); ++k) {
        if (da[k] < min_dangle_boundary && da[k] <= da0[k]) {
            is_sharp[v1[k]] = 1;
            if (v2[k] != kInvalidU32 && v2[k] < static_cast<std::uint32_t>(he.rows()))
                is_sharp[v2[k]] = 1;
        }
    }

    std::vector<std::uint8_t> is_sharp_b(static_cast<std::size_t>(xyz.rows()), 0);
    for (Eigen::Index i = 0; i < he.rows(); ++i) {
        if (!is_sharp[static_cast<std::size_t>(i)]) continue;
        // Nodes "next to" the sharp hedges, plus the hedge's source / target.
        is_sharp_b[he(he(i, 1), 0)] = 1;     // next.target
        is_sharp_b[he(i, 0)]        = 1;     // target
        is_sharp_b[he(he(i, 2), 0)] = 1;     // prev.target (= source)
    }
    for (Eigen::Index v = 0; v < xyz.rows(); ++v) {
        if (is_sharp_b[static_cast<std::size_t>(v)]) xyz.row(v) = xyz0.row(v);
    }
}

// ---------------------------------------------------------------------------
// Min interior corner angle of triangle (va, vb, vc), in degrees.
double calc_min_corner_of_triangle(const Coords& xyz,
                                   std::uint32_t va,
                                   std::uint32_t vb,
                                   std::uint32_t vc) {
    const Eigen::Vector3d A = xyz.row(va).transpose();
    const Eigen::Vector3d B = xyz.row(vb).transpose();
    const Eigen::Vector3d C = xyz.row(vc).transpose();
    auto corner = [](const Eigen::Vector3d& v, const Eigen::Vector3d& a,
                     const Eigen::Vector3d& b) {
        const Eigen::Vector3d u1 = a - v;
        const Eigen::Vector3d u2 = b - v;
        const double n1 = u1.norm(), n2 = u2.norm();
        if (n1 < 1e-18 || n2 < 1e-18) return 180.0;
        const double c = std::clamp(u1.dot(u2) / (n1 * n2), -1.0, 1.0);
        return std::acos(c) * 180.0 / M_PI;
    };
    const double a1 = corner(A, B, C);
    const double a2 = corner(B, C, A);
    const double a3 = corner(C, A, B);
    return std::min(a1, std::min(a2, a3));
}

// ---------------------------------------------------------------------------
// Walk each triangle (3 consecutive half-edges forming a cycle via the
// `next` pointer) and check the min corner angle before vs after a move.
// If the move made the smallest corner drop below threshold AND below
// its prior value, mark the 3 vertices for revert. Threshold is
// `min_corner_internal` for triangles with at least one not_fixed_v
// vertex; `min_corner_boundary` if all 3 are fixed (bbox-only triangle).
void undo_small_corner_nodes(Coords& xyz, const Coords& xyz0,
                             const mesh::HalfEdges& he,
                             const std::vector<std::uint8_t>& not_fixed_v,
                             double min_corner_internal,
                             double min_corner_boundary) {
    std::vector<std::uint8_t> visited(static_cast<std::size_t>(he.rows()), 0);
    std::vector<std::uint8_t> revert(static_cast<std::size_t>(xyz.rows()), 0);

    for (Eigen::Index ih = 0; ih < he.rows(); ++ih) {
        const std::uint32_t h0 = static_cast<std::uint32_t>(ih);
        if (visited[h0]) continue;
        const std::uint32_t h1 = he(h0, 1);
        const std::uint32_t h2 = he(h0, 2);
        if (h1 >= visited.size() || h2 >= visited.size()) continue;
        visited[h0] = visited[h1] = visited[h2] = 1;

        const std::uint32_t va = he(h0, 0);
        const std::uint32_t vb = he(h1, 0);
        const std::uint32_t vc = he(h2, 0);
        if (va >= xyz.rows() || vb >= xyz.rows() || vc >= xyz.rows()) continue;

        const double thr = (not_fixed_v[va] || not_fixed_v[vb] || not_fixed_v[vc])
                         ? min_corner_internal : min_corner_boundary;

        const double c1 = calc_min_corner_of_triangle(xyz,  va, vb, vc);
        if (c1 >= thr) continue;
        const double c0 = calc_min_corner_of_triangle(xyz0, va, vb, vc);
        if (c1 > c0 - 1e-6) continue;     // not made worse — keep
        revert[va] = revert[vb] = revert[vc] = 1;
    }
    for (Eigen::Index v = 0; v < xyz.rows(); ++v) {
        if (revert[static_cast<std::size_t>(v)]) xyz.row(v) = xyz0.row(v);
    }
}

// ---------------------------------------------------------------------------
void smooth_surfaces_dangle_ctrl(const Settings& s,
                                 const mesh::HalfEdges& he,
                                 const std::vector<std::uint8_t>& not_fixed_h,
                                 Coords& xyz,
                                 const Coords& xyz0,
                                 const std::vector<std::uint8_t>& not_fixed_v,
                                 int n_max_iter,
                                 const std::vector<std::uint32_t>* blinks) {
    VOX2TET_PRINT("Start surfaces smoothing with dihedral angle and distance control...");
    const Eigen::Index n_nodes = xyz.rows();

    // Build adjacency from not-fixed half-edges: connections
    // (he(:, 0), he(prev, 0)).
    std::vector<std::vector<std::uint32_t>> adj(static_cast<std::size_t>(n_nodes));
    for (Eigen::Index i = 0; i < he.rows(); ++i) {
        if (!not_fixed_h[static_cast<std::size_t>(i)]) continue;
        const std::uint32_t src = he(i, 0);
        const std::uint32_t dst = he(he(i, 2), 0);
        adj[src].push_back(dst);
    }

    // Collect not_fixed_h indices once; their column-3 partners.
    std::vector<std::uint32_t> nfh, nfh2;
    nfh.reserve(static_cast<std::size_t>(he.rows()));
    for (Eigen::Index i = 0; i < he.rows(); ++i)
        if (not_fixed_h[static_cast<std::size_t>(i)]) nfh.push_back(static_cast<std::uint32_t>(i));
    nfh2.resize(nfh.size());
    for (std::size_t k = 0; k < nfh.size(); ++k) nfh2[k] = he(nfh[k], 3);

    // Cycle-aware dihedral metric (Layer 1). When `blinks` is non-null
    // we evaluate the MIN dihedral across all cycle partners of every
    // not_fixed_h half-edge — needed to catch folds across multi-
    // material lines that the canonical (h, he(h,3)) pair misses.
    std::vector<std::vector<std::uint32_t>> cycle_mates;
    if (blinks) cycle_mates = build_cycle_mates(he, *blinks);

    auto eval_da = [&](const Coords& X) -> Eigen::VectorXd {
        return blinks ? calc_min_dihedral_cycle(he, X, nfh, cycle_mates)
                      : calc_dihedral(he, X, nfh, nfh2);
    };

    auto da0 = eval_da(xyz);

    Coords xyz_tmp = xyz;
    int last_iter = 0;
    for (int it = 0; it < n_max_iter; ++it) {
        last_iter = it;
        // 1. Laplacian update into xyz_tmp.
        for (Eigen::Index v = 0; v < n_nodes; ++v) {
            if (!not_fixed_v[static_cast<std::size_t>(v)]) continue;
            const auto& nbrs = adj[static_cast<std::size_t>(v)];
            if (nbrs.empty()) continue;
            Eigen::Vector3d sum = Eigen::Vector3d::Zero();
            for (auto n : nbrs) sum += xyz.row(n).transpose();
            const double inv = 1.0 / static_cast<double>(nbrs.size());
            xyz_tmp.row(v) = s.smooth_alpha * xyz.row(v)
                           + (1.0 - s.smooth_alpha) * (sum * inv).transpose();
        }
        // 2. Recompute dihedral angles after candidate move (cycle-min
        //    if blinks present, else canonical-pair).
        auto da = eval_da(xyz_tmp);

        if (it > 0) {
            bool no_change = true;
            for (Eigen::Index k = 0; k < da.size(); ++k)
                if (da[k] - da0[k] >= 1.0) { no_change = false; break; }
            if (no_change) break;
        }

        // 3. Find which vertices are "sharp" (i.e. their candidate move
        // produced an unacceptable dihedral). Initially everything is
        // marked "non-sharp" (allowed), then we exclude the sharp ones.
        std::vector<std::uint8_t> is_sharp_i(static_cast<std::size_t>(n_nodes), 1);
        for (std::size_t k = 0; k < nfh.size(); ++k) {
            if (da[k] < s.min_dangle_internal && da[k] <= da0[k]) {
                const std::uint32_t h = nfh[k];
                is_sharp_i[he(h, 0)] = 0;          // target
                is_sharp_i[he(he(h, 1), 0)] = 0;   // next.target
            }
        }
        // 4. Distance-from-original guard: only smooth where moved < max_D2self.
        std::vector<std::uint8_t> is_close(static_cast<std::size_t>(n_nodes), 0);
        for (Eigen::Index v = 0; v < n_nodes; ++v) {
            const double d = (xyz0.row(v).transpose() - xyz_tmp.row(v).transpose()).norm();
            if (d < s.max_D2self) is_close[static_cast<std::size_t>(v)] = 1;
        }
        // 5. Compose the move mask.
        std::vector<std::uint32_t> is_sm;
        is_sm.reserve(static_cast<std::size_t>(n_nodes));
        for (Eigen::Index v = 0; v < n_nodes; ++v) {
            if (not_fixed_v[static_cast<std::size_t>(v)] &&
                is_sharp_i[static_cast<std::size_t>(v)] &&
                is_close[static_cast<std::size_t>(v)]) {
                is_sm.push_back(static_cast<std::uint32_t>(v));
            }
        }
        // 6. Min-distance-to-other guard (kd-tree query in the reference; here we
        // brute-force using `not_fixed_v` vertices as the pool). This is
        // O(|is_sm| * |not_fixed_v|) — fine for the JMA test sizes,
        // parallelised over queries.
        std::vector<std::uint32_t> nfv_idx;
        nfv_idx.reserve(static_cast<std::size_t>(n_nodes));
        for (Eigen::Index v = 0; v < n_nodes; ++v)
            if (not_fixed_v[static_cast<std::size_t>(v)])
                nfv_idx.push_back(static_cast<std::uint32_t>(v));
        std::vector<std::uint8_t> keep(is_sm.size(), 1);
#ifdef VOX2TET_HAS_OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for (std::size_t k = 0; k < is_sm.size(); ++k) {
            const Eigen::Vector3d q = xyz.row(is_sm[k]).transpose();
            // The reference queries xyz_tmp[not_fixed_v]; second-nearest distance
            // (k=[2] = the 2nd-nearest). Here, we look up distance to the
            // nearest *other* candidate point in xyz_tmp.
            double second = std::numeric_limits<double>::infinity();
            double first  = std::numeric_limits<double>::infinity();
            for (auto u : nfv_idx) {
                const double d = (xyz_tmp.row(u).transpose() - q).norm();
                if (d < first)       { second = first; first = d; }
                else if (d < second) { second = d; }
            }
            if (!(second > s.min_D2other)) keep[k] = 0;
        }
        // 7. Commit.
        bool any_moved = false;
        for (std::size_t k = 0; k < is_sm.size(); ++k) {
            if (!keep[k]) continue;
            xyz.row(is_sm[k]) = xyz_tmp.row(is_sm[k]);
            any_moved = true;
        }
        if (!any_moved) break;
        da0 = da;
    }
    VOX2TET_LOG() << "Smoothing surfaces done! N iteration: " << (last_iter + 1);
}

// ---------------------------------------------------------------------------
Eigen::VectorXd smooth_laplace(const Settings& s,
                               const Triangles& tri,
                               const marching_cubes::NodeTypeMask& is_bnode,
                               const mesh::HalfEdges& he,
                               const std::vector<std::uint8_t>& not_fixed_h,
                               Coords& xyz,
                               const std::vector<std::uint8_t>& not_fixed_v,
                               const std::string& path_base) {
    // Save initial coords.
    Coords xyz0 = xyz;

    auto bedges = brep::get_boundary_edges(tri, is_bnode);
    auto brep_  = brep::order_bedges(bedges, is_bnode);

    if (!path_base.empty()) {
        brep::save_brep_ply(brep_, xyz, path_base, {0, 255, 0});
    }

    // First pass: chain smoothing without sizing field.
    smooth_brep_mean(s, brep_.brep, xyz, xyz0, s.n_smooth_steps, /*sizing=*/nullptr);

    // Build sizing field based on the smoothed positions. is_brep = !is_bnode[0].
    std::vector<std::uint8_t> is_brep(static_cast<std::size_t>(xyz.rows()), 0);
    for (Eigen::Index v = 0; v < xyz.rows(); ++v)
        is_brep[static_cast<std::size_t>(v)] = !is_bnode.masks[0][v];
    auto brep_sizing = calc_brep_sizing(xyz, is_brep);

    // Non-manifold blink cycle. Builds the "next half-edge sharing the
    // canonical edge" pointer per half-edge. Manifold edges collapse to
    // the regular column-3 opposite; triple lines / quad points form a
    // 3- or 4-cycle. Threaded into the dihedral-aware smoothers and
    // revert pass so they enforce `min_dangle_*` over every pair of
    // triangles around the edge, not just the canonical one.
    auto blinks = mesh::create_blinks(he, xyz);

    // Restart from original coords, then iterate.
    xyz = xyz0;
    Coords xyz_old = xyz;
    for (int it = 0; it < s.n_smooth_steps; ++it) {
        VOX2TET_LOG() << "\n - \n smooth iteration " << it;
        smooth_brep_mean(s, brep_.brep, xyz, xyz0, 1, &brep_sizing);
        undo_sharp_bdangle_nodes(xyz, xyz_old, he, not_fixed_h, s.min_dangle_boundary, &blinks);
        undo_sharp_bdangle_nodes(xyz, xyz_old, he, not_fixed_h, s.min_dangle_boundary, &blinks);
        // Step 2 (Layer 2 extension): corner-angle guard. Revert any
        // triangle whose smallest interior angle was made smaller than
        // the user threshold by this smoothing step.
        undo_small_corner_nodes(xyz, xyz_old, he, not_fixed_v,
                                s.min_corner_angle_internal,
                                s.min_corner_angle_boundary);
        xyz_old = xyz;
        smooth_surfaces_dangle_ctrl(s, he, not_fixed_h, xyz, xyz0, not_fixed_v, 1, &blinks);
        undo_sharp_bdangle_nodes(xyz, xyz_old, he, not_fixed_h, s.min_dangle_boundary, &blinks);
        undo_small_corner_nodes(xyz, xyz_old, he, not_fixed_v,
                                s.min_corner_angle_internal,
                                s.min_corner_angle_boundary);
        xyz_old = xyz;
    }

    auto final_sizing = calc_brep_sizing(xyz, is_brep);

    // Stats.
    double mn = std::numeric_limits<double>::infinity();
    double mx = -mn;
    double sum = 0;
    std::size_t cnt = 0;
    for (Eigen::Index v = 0; v < xyz.rows(); ++v) {
        if (!is_brep[static_cast<std::size_t>(v)]) continue;
        const double s_ = final_sizing[v];
        mn = std::min(mn, s_);
        mx = std::max(mx, s_);
        sum += s_;
        ++cnt;
    }
    if (cnt > 0) {
        VOX2TET_LOG() << "min brep_sizing: " << mn;
        VOX2TET_LOG() << "max brep_sizing: " << mx;
        VOX2TET_LOG() << "mean brep_sizing: " << (sum / static_cast<double>(cnt));
    }

    if (!path_base.empty()) {
        brep::save_brep_ply(brep_, xyz, path_base + "_S", {0, 255, 0});
    }
    return final_sizing;
}

}  // namespace vox2tet::remesh
