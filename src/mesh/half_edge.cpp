#include "vox2tet/mesh/half_edge.hpp"

#include "vox2tet/core/log.hpp"
#include "vox2tet/io/npy.hpp"
#include "vox2tet/mesh/mathx.hpp"

#include <Eigen/Geometry>
#include <Eigen/LU>
#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace vox2tet::mesh {

namespace {

constexpr u32 kInvalid = std::numeric_limits<u32>::max();

}  // namespace

HalfEdges triangles_to_hedges(const Triangles& tri,
                              const std::vector<marching_cubes::Interface>* itf) {
    VOX2TET_PRINT("Start half-edges creating...");

    const Eigen::Index ntr = tri.rows();
    const Eigen::Index n_he = ntr * 3;

    HalfEdges he(n_he, 5);
    he.setZero();
    // Column 4 (interface id) defaults to sentinel.
    he.col(4).setConstant(kInvalid);

    // Initial fill of column 0 and 1 with directed edge (source, target):
    //   tri i, half-edge 0 → (v0, v1)
    //   tri i, half-edge 1 → (v1, v2)
    //   tri i, half-edge 2 → (v2, v0)
    // Plus column 2 carries the "third corner" (for symmetry); not used
    // beyond this scope but mirrors the reference `hedges[:ntr,:3] = triangles`.
    for (Eigen::Index i = 0; i < ntr; ++i) {
        const u32 v0 = tri(i, 0), v1 = tri(i, 1), v2 = tri(i, 2);
        he(i,            0) = v0; he(i,            1) = v1; he(i,            2) = v2;
        he(i + ntr,      0) = v1; he(i + ntr,      1) = v2; he(i + ntr,      2) = v0;
        he(i + 2 * ntr,  0) = v2; he(i + 2 * ntr,  1) = v0; he(i + 2 * ntr,  2) = v1;
    }

    // --- Pair up opposite half-edges -------------------------------------
    //
    // For each half-edge (u, v), the opposite (if any) has endpoints
    // (v, u). We canonicalise to (min(u,v), max(u,v)), stable-sort by
    // that pair, walk the runs, and link adjacent rows. Boundary edges
    // (run length 1) get `kInvalid`. Non-manifold (run > 2) is reported.
    struct Entry { u32 lo, hi; Eigen::Index idx; };
    std::vector<Entry> entries(static_cast<std::size_t>(n_he));
    for (Eigen::Index i = 0; i < n_he; ++i) {
        const u32 a = he(i, 0), b = he(i, 1);
        entries[static_cast<std::size_t>(i)] = {
            std::min(a, b), std::max(a, b), i,
        };
    }
    std::sort(entries.begin(), entries.end(),
              [](const Entry& a, const Entry& b) {
                  if (a.lo != b.lo) return a.lo < b.lo;
                  return a.hi < b.hi;
              });

    // Initialise column 3 with `kInvalid` (boundary by default).
    he.col(3).setConstant(kInvalid);

    std::size_t i = 0;
    const std::size_t N = entries.size();
    std::size_t n_nonmanifold = 0;
    while (i < N) {
        std::size_t j = i + 1;
        while (j < N && entries[j].lo == entries[i].lo &&
                        entries[j].hi == entries[i].hi) ++j;
        const std::size_t run = j - i;
        if (run == 1) {
            // boundary half-edge — opposite stays kInvalid
        } else if (run == 2) {
            he(entries[i    ].idx, 3) = static_cast<u32>(entries[i + 1].idx);
            he(entries[i + 1].idx, 3) = static_cast<u32>(entries[i    ].idx);
        } else {
            // Non-manifold edge — expected on multi-material lines where
            // 3+ triangles meet. the reference implementation handles count=2..4 properly in
            // create_boundary_hedges_links (built later by remesh code);
            // here we just pair the first two so the basic half-edge
            // structure remains usable and count the occurrences for one
            // summary log line at the end.
            for (std::size_t k = 0; k + 1 < run; k += 2) {
                he(entries[i + k    ].idx, 3) = static_cast<u32>(entries[i + k + 1].idx);
                he(entries[i + k + 1].idx, 3) = static_cast<u32>(entries[i + k    ].idx);
            }
            ++n_nonmanifold;
        }
        i = j;
    }
    if (n_nonmanifold > 0) {
        VOX2TET_LOG() << "triangles_to_hedges: " << n_nonmanifold
                      << " non-manifold edges (3+ incident half-edges) — "
                      << "phase-9 create_boundary_hedges_links will handle them";
    }

    // --- Column 0 (target vertex): set to TARGET (column 1 currently
    //     holds target after the directed-edge fill above).
    for (Eigen::Index k = 0; k < n_he; ++k) he(k, 0) = he(k, 1);

    // --- Column 1 (next half-edge) and column 2 (prev) per triangle.
    for (Eigen::Index k = 0; k < ntr; ++k) {
        he(k,            1) = static_cast<u32>(k + ntr);
        he(k + ntr,      1) = static_cast<u32>(k + 2 * ntr);
        he(k + 2 * ntr,  1) = static_cast<u32>(k);

        he(k,            2) = static_cast<u32>(k + 2 * ntr);
        he(k + ntr,      2) = static_cast<u32>(k);
        he(k + 2 * ntr,  2) = static_cast<u32>(k + ntr);
    }

    // --- Interface ids — broadcast (first, count) ranges to the three
    //     half-edge bands (rows k, k+ntr, k+2*ntr for triangle k).
    if (itf) {
        for (std::size_t a = 0; a < itf->size(); ++a) {
            const auto& I = (*itf)[a];
            const Eigen::Index f = static_cast<Eigen::Index>(I.first);
            const Eigen::Index c = static_cast<Eigen::Index>(I.count);
            for (Eigen::Index t = 0; t < c; ++t) {
                he(f + t,            4) = static_cast<u32>(a);
                he(f + t + ntr,      4) = static_cast<u32>(a);
                he(f + t + 2 * ntr,  4) = static_cast<u32>(a);
            }
        }
    }

    VOX2TET_PRINT("Half-edges created!");
    return he;
}

// ---------------------------------------------------------------------------
Triangles hedges_to_triangles(const HalfEdges& he,
                              std::vector<marching_cubes::Interface>* itf) {
    const Eigen::Index n_he = he.rows();
    if (n_he % 3 != 0) throw std::runtime_error("hedges_to_triangles: n_he not divisible by 3");
    const Eigen::Index ntr = n_he / 3;

    // For each triangle, exactly one half-edge has its target vertex
    // strictly smaller than the targets of `next` and `prev`. That
    // half-edge anchors the triangle output.
    std::vector<u32> tri_anchor;
    tri_anchor.reserve(static_cast<std::size_t>(ntr));
    for (Eigen::Index i = 0; i < n_he; ++i) {
        const u32 v0 = he(i, 0);
        const u32 vn = he(he(i, 1), 0);
        const u32 vp = he(he(i, 2), 0);
        if (v0 < vn && v0 < vp) tri_anchor.push_back(static_cast<u32>(i));
    }
    if (static_cast<Eigen::Index>(tri_anchor.size()) != ntr) {
        throw std::runtime_error("hedges_to_triangles: anchor count mismatch");
    }

    Triangles out(ntr, 3);
    for (Eigen::Index i = 0; i < ntr; ++i) {
        const u32 a = tri_anchor[static_cast<std::size_t>(i)];
        out(i, 0) = he(a, 0);
        out(i, 1) = he(he(a, 1), 0);
        out(i, 2) = he(he(a, 2), 0);
    }

    if (!itf) return out;

    // Sort triangles per-interface (stable, mirrors np.argsort over col 4).
    VOX2TET_PRINT("Triangles from half-edges creating...");
    std::vector<u32> itf_id(static_cast<std::size_t>(ntr));
    for (Eigen::Index i = 0; i < ntr; ++i)
        itf_id[static_cast<std::size_t>(i)] = he(tri_anchor[static_cast<std::size_t>(i)], 4);

    std::vector<Eigen::Index> order(ntr);
    for (Eigen::Index i = 0; i < ntr; ++i) order[i] = i;
    std::stable_sort(order.begin(), order.end(),
        [&](Eigen::Index a, Eigen::Index b) {
            return itf_id[static_cast<std::size_t>(a)] < itf_id[static_cast<std::size_t>(b)];
        });

    Triangles out_sorted(ntr, 3);
    std::vector<u32> itf_id_sorted(static_cast<std::size_t>(ntr));
    for (Eigen::Index i = 0; i < ntr; ++i) {
        out_sorted.row(i)    = out.row(order[i]);
        itf_id_sorted[static_cast<std::size_t>(i)] = itf_id[static_cast<std::size_t>(order[i])];
    }
    // Rebuild itf with (first, count) reflecting the new ordering. We
    // preserve the original `mat1` / `mat2` / `code` lookups by keying on
    // itf_id.
    std::vector<marching_cubes::Interface> new_itf;
    if (!itf_id_sorted.empty()) {
        std::size_t run_start = 0;
        u32 cur = itf_id_sorted[0];
        for (std::size_t i = 1; i <= itf_id_sorted.size(); ++i) {
            if (i == itf_id_sorted.size() || itf_id_sorted[i] != cur) {
                const marching_cubes::Interface& src = (*itf)[cur];
                marching_cubes::Interface nx{src};
                nx.first = static_cast<u32>(run_start);
                nx.count = static_cast<u32>(i - run_start);
                new_itf.push_back(nx);
                if (i < itf_id_sorted.size()) {
                    cur = itf_id_sorted[i];
                    run_start = i;
                }
            }
        }
    }
    *itf = std::move(new_itf);

    VOX2TET_PRINT("Triangles from half-edges created!");
    return out_sorted;
}

// ---------------------------------------------------------------------------
NotFixedMasks get_not_fixed(const HalfEdges& he, const Coords& xyz,
                            const marching_cubes::NodeTypeMask& is_bnode) {
    NotFixedMasks R;
    R.not_fixed_v = is_bnode.masks[0];  // copy of "voxel facets" mask

    const Eigen::Index n_he = he.rows();
    R.not_fixed_h.assign(static_cast<std::size_t>(n_he), 0);

    // Internal = has a real opposite.
    std::vector<std::uint8_t> is_internal(static_cast<std::size_t>(n_he), 0);
    Eigen::Index n_int = 0;
    for (Eigen::Index i = 0; i < n_he; ++i) {
        if (he(i, 3) != kInvalid && he(i, 3) < static_cast<u32>(n_he)) {
            is_internal[static_cast<std::size_t>(i)] = 1;
            ++n_int;
        }
    }

    // For each internal half-edge, compute the (p01, p02, p1, p2)
    // dihedral around it and mark `is_sharp` when <100°.
    MatrixNx3<double> p01(n_int, 3), p02(n_int, 3), p1(n_int, 3), p2(n_int, 3);
    std::vector<Eigen::Index> int_idx(static_cast<std::size_t>(n_int));
    {
        Eigen::Index w = 0;
        for (Eigen::Index i = 0; i < n_he; ++i) {
            if (!is_internal[static_cast<std::size_t>(i)]) continue;
            int_idx[static_cast<std::size_t>(w)] = i;
            // v[0] = hedges[hedges[i, 2], 0]
            // v[1] = hedges[i, 0]
            // v[2] = hedges[hedges[i, 1], 0]
            // v[3] = hedges[hedges[hedges[i, 3], 1], 0]
            const u32 v0 = he(he(i, 2), 0);
            const u32 v1 = he(i, 0);
            const u32 v2 = he(he(i, 1), 0);
            const u32 v3 = he(he(he(i, 3), 1), 0);
            p01.row(w) = xyz.row(v0);
            p02.row(w) = xyz.row(v1);
            p1.row(w)  = xyz.row(v2);
            p2.row(w)  = xyz.row(v3);
            ++w;
        }
    }
    auto dan = mathx::calc_dihedral_angle_v(p01, p02, p1, p2);

    // Sharp = dan < 100 (reference: is_sharp[is_internal] = (dan < 100))
    std::vector<std::uint8_t> is_sharp(static_cast<std::size_t>(n_he), 0);
    for (Eigen::Index w = 0; w < n_int; ++w) {
        if (dan[w] < 100.0) is_sharp[static_cast<std::size_t>(int_idx[static_cast<std::size_t>(w)])] = 1;
    }

    // bb_frame = is_sharp & is_bnode[4][prev.target] & is_bnode[4][target]
    // not_fixed_h = is_internal & ~bb_frame
    for (Eigen::Index i = 0; i < n_he; ++i) {
        const u32 prev_tgt = he(he(i, 2), 0);
        const u32 tgt      = he(i, 0);
        const bool bb_frame = is_sharp[static_cast<std::size_t>(i)] &&
                              is_bnode.masks[4][prev_tgt] &&
                              is_bnode.masks[4][tgt];
        R.not_fixed_h[static_cast<std::size_t>(i)] =
            is_internal[static_cast<std::size_t>(i)] && !bb_frame;
    }
    return R;
}

// ---------------------------------------------------------------------------
void save_hedges_npy(const std::string& path, const HalfEdges& he) {
    std::vector<u32> flat(static_cast<std::size_t>(he.size()));
    for (Eigen::Index i = 0; i < he.size(); ++i)
        flat[static_cast<std::size_t>(i)] = he.data()[i];
    npy::write<u32>(path,
                    {static_cast<std::size_t>(he.rows()), 5},
                    flat.data(), flat.size());
}

// ---------------------------------------------------------------------------
// Helper: canonical-edge sort. Returns:
//   * `sort_order` such that he[sort_order[i]] is in sorted position i
//     (the legacy `ind` from np.lexsort).
//   * `r_ind` is the inverse: r_ind[sort_order[i]] == i.
//   * `unique_first` lists the sorted position of the first half-edge of
//     each unique canonical edge.
//   * `count_per_unique` is parallel to unique_first.
//
// We do NOT modify `he` here — caller can sort a copy if needed.
namespace {

struct SortHedgesResult {
    std::vector<u32> sort_order;           // size = he.rows()
    std::vector<u32> r_ind;                // size = he.rows()
    std::vector<u32> unique_first;
    std::vector<u32> count_per_unique;
};

SortHedgesResult sort_hedges_canonical(const HalfEdges& he) {
    const Eigen::Index n = he.rows();
    // Canonical edge per half-edge: sorted (prev.target, target).
    std::vector<std::array<u32, 2>> edges(static_cast<std::size_t>(n));
    for (Eigen::Index i = 0; i < n; ++i) {
        const u32 a = he(he(i, 2), 0);
        const u32 b = he(i, 0);
        edges[static_cast<std::size_t>(i)] = (a < b) ? std::array<u32, 2>{a, b}
                                                     : std::array<u32, 2>{b, a};
    }

    SortHedgesResult R;
    R.sort_order.resize(static_cast<std::size_t>(n));
    for (Eigen::Index i = 0; i < n; ++i) R.sort_order[static_cast<std::size_t>(i)] = static_cast<u32>(i);
    std::stable_sort(R.sort_order.begin(), R.sort_order.end(),
        [&](u32 a, u32 b) {
            if (edges[a][0] != edges[b][0]) return edges[a][0] < edges[b][0];
            return edges[a][1] < edges[b][1];
        });

    R.r_ind.assign(static_cast<std::size_t>(n), 0);
    for (std::size_t i = 0; i < R.sort_order.size(); ++i)
        R.r_ind[R.sort_order[i]] = static_cast<u32>(i);

    // Group sorted runs into unique canonical edges.
    R.unique_first.reserve(static_cast<std::size_t>(n));
    R.count_per_unique.reserve(static_cast<std::size_t>(n));
    std::size_t i = 0;
    while (i < R.sort_order.size()) {
        std::size_t j = i + 1;
        while (j < R.sort_order.size() &&
               edges[R.sort_order[j]] == edges[R.sort_order[i]]) ++j;
        R.unique_first.push_back(static_cast<u32>(i));
        R.count_per_unique.push_back(static_cast<u32>(j - i));
        i = j;
    }
    return R;
}

// Build the *sorted* half-edge view: new_he[i] = he[sort_order[i]] with
// columns 1/2/3 remapped through r_ind so the new array is internally
// consistent.
HalfEdges apply_sort(const HalfEdges& he,
                     const std::vector<u32>& sort_order,
                     const std::vector<u32>& r_ind) {
    const Eigen::Index n = he.rows();
    HalfEdges out(n, 5);
    for (Eigen::Index i = 0; i < n; ++i) {
        const u32 src = sort_order[static_cast<std::size_t>(i)];
        out(i, 0) = he(src, 0);
        out(i, 1) = r_ind[he(src, 1)];
        out(i, 2) = r_ind[he(src, 2)];
        const u32 op = he(src, 3);
        out(i, 3) = (op == kInvalid || op >= n) ? kInvalid : r_ind[op];
        out(i, 4) = he(src, 4);
    }
    return out;
}

// Directional dihedral between two half-edges that share the same
// canonical edge but live in different triangles. Used to order the
// 4-cycle for count==4 non-manifold edges.
double directional_dihedral(const HalfEdges& he, const Coords& xyz, u32 h1, u32 h2) {
    // Reproduce calcDihedralAngleVdirectional with the same vertex
    // selection as the legacy `calc_dihedral(..., True)`:
    //   v1 = [prev.target, target, next.target]  for h1
    //   v2 = [prev.target, target, next.target]  for h2
    // The two share the (v[0], v[1]) edge — calcDihedralAngleVdirectional
    // wants (p01, p02, p1, p2) = (v1[0], v1[1], v1[2], v2[2]).
    const Eigen::Vector3d p01 = xyz.row(he(he(h1, 2), 0)).transpose();
    const Eigen::Vector3d p02 = xyz.row(he(h1, 0)).transpose();
    const Eigen::Vector3d p1  = xyz.row(he(he(h1, 1), 0)).transpose();
    const Eigen::Vector3d p2  = xyz.row(he(he(h2, 1), 0)).transpose();
    const Eigen::Vector3d v0 = p02 - p01;
    const Eigen::Vector3d v1 = p1  - p01;
    const Eigen::Vector3d v2 = p2  - p01;
    const Eigen::Vector3d cr1 = v1.cross(v0);
    const Eigen::Vector3d cr2 = v2.cross(v0);
    const double n12 = cr1.norm() * cr2.norm();
    double ang = (n12 < 1e-12) ? -1.0
                               : std::acos(std::clamp(cr1.dot(cr2) / n12, -1.0, 1.0))
                                     * 180.0 / 3.14159265358979323846;
    // Determinant sign decides whether to flip into [0, 360).
    Eigen::Matrix3d M;
    M.row(0) = v0;
    M.row(1) = cr1;
    M.row(2) = cr2;
    if (M.determinant() > 0.0 && ang >= 0) ang = 360.0 - ang;
    return ang;
}

}  // namespace

// ---------------------------------------------------------------------------
std::vector<u32> create_blinks(const HalfEdges& he, const Coords& xyz) {
    const Eigen::Index n = he.rows();
    auto sr = sort_hedges_canonical(he);
    HalfEdges he_sorted = apply_sort(he, sr.sort_order, sr.r_ind);

    // b_hedges (sorted space) starts at column 3 of the sorted hedges —
    // matches the reference `b_hedges = hedges[:, 3].copy()`. Boundary edges
    // (count=1) keep their sentinel.
    std::vector<u32> b_sorted(static_cast<std::size_t>(n), kInvalid);
    for (Eigen::Index i = 0; i < n; ++i) b_sorted[static_cast<std::size_t>(i)] = he_sorted(i, 3);

    for (std::size_t k = 0; k < sr.unique_first.size(); ++k) {
        const u32 first = sr.unique_first[k];
        const u32 cnt   = sr.count_per_unique[k];
        if (cnt == 2) {
            b_sorted[first    ] = first + 1;
            b_sorted[first + 1] = first;
        } else if (cnt == 3) {
            b_sorted[first    ] = first + 1;
            b_sorted[first + 1] = first + 2;
            b_sorted[first + 2] = first;
        } else if (cnt == 4) {
            // Order indices 1, 2, 3 by directional dihedral against
            // index 0. i_sort[j] is the rank (0..2) of index (j+1); we
            // add 1 to convert into ordering positions 1..3.
            const double d1 = directional_dihedral(he_sorted, xyz, first, first + 1);
            const double d2 = directional_dihedral(he_sorted, xyz, first, first + 2);
            const double d3 = directional_dihedral(he_sorted, xyz, first, first + 3);
            std::array<std::pair<double, int>, 3> ds = {{
                {d1, 1}, {d2, 2}, {d3, 3}
            }};
            std::stable_sort(ds.begin(), ds.end(),
                [](const auto& a, const auto& b) { return a.first < b.first; });
            const u32 a = first + static_cast<u32>(ds[0].second);
            const u32 b = first + static_cast<u32>(ds[1].second);
            const u32 c = first + static_cast<u32>(ds[2].second);
            b_sorted[first] = a;
            b_sorted[a]     = b;
            b_sorted[b]     = c;
            b_sorted[c]     = first;
        }
        // count==1 leaves the kInvalid sentinel.
    }

    // Map blinks back into the *original* half-edge index space.
    //   reference: b_hedges[:] = b_hedges[r_ind];  b_hedges[b] = rr_ind[b_hedges[b]]
    // where r_ind = sort_order, rr_ind = inverse of r_ind (= sort_order^{-1}).
    // For us r_ind is already the inverse-permutation (r_ind[orig] = sorted),
    // so rr_ind = sort_order.
    std::vector<u32> blinks(static_cast<std::size_t>(n), kInvalid);
    for (Eigen::Index i = 0; i < n; ++i) {
        const u32 sorted_pos = sr.r_ind[static_cast<std::size_t>(i)];
        const u32 b_in_sorted = b_sorted[sorted_pos];
        blinks[static_cast<std::size_t>(i)] =
            (b_in_sorted == kInvalid) ? kInvalid
                                      : sr.sort_order[b_in_sorted];
    }
    return blinks;
}

// ---------------------------------------------------------------------------
std::vector<u32> get_second_blinks(const std::vector<u32>& blinks) {
    const std::size_t n = blinks.size();
    std::vector<u32> b2 = blinks;
    for (std::size_t i = 0; i < n; ++i) {
        const u32 b1 = blinks[i];
        if (b1 == kInvalid) continue;
        u32 b2_step = blinks[b1];
        // If we'd land back on i (count==2 case), fold to b1.
        if (b2_step == static_cast<u32>(i)) b2_step = b1;
        u32 b3_step = blinks[b2_step];
        if (b3_step == static_cast<u32>(i)) b3_step = b2_step;
        b2[i] = b3_step;
    }
    return b2;
}

// ---------------------------------------------------------------------------
V2HLists get_v2h_lists(const HalfEdges& he) {
    V2HLists R;
    u32 max_v = 0;
    for (Eigen::Index i = 0; i < he.rows(); ++i) max_v = std::max(max_v, he(i, 0));
    R.by_vertex.resize(static_cast<std::size_t>(max_v) + 1);
    R.is_boundary_vertex.assign(static_cast<std::size_t>(max_v) + 1, 0);
    for (Eigen::Index i = 0; i < he.rows(); ++i) {
        if (he(i, 3) != kInvalid) continue;  // not a boundary half-edge
        const u32 v = he(i, 0);
        R.by_vertex[v].push_back(static_cast<u32>(i));
        R.is_boundary_vertex[v] = 1;
    }
    return R;
}

// ---------------------------------------------------------------------------
bool is_edge_exist(const HalfEdges& he, const std::vector<u32>& he_boundary, u32 v) {
    const u32 n = static_cast<u32>(he.rows());
    for (u32 h : he_boundary) {
        u32 h0 = he(he(h, 1), 3);
        // Cap traversal to avoid runaway loops on a malformed mesh.
        for (std::size_t step = 0; step < 1024 && h0 < n; ++step) {
            if (he(he(h0, 2), 0) == v) return true;
            h0 = he(he(h0, 1), 3);
        }
    }
    return false;
}

}  // namespace vox2tet::mesh
