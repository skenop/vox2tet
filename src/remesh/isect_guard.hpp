#pragma once

// Self-intersection guard shared by the remesh operations (edge collapse,
// edge flip, dihedral repair).
//
// The near-chain collapse path (`do_collapse_near_bedges`) lets the mesh
// get tight next to feature chains; without a global check, a collapse,
// a flip or a repair move can then fold one sheet through a nearby one.
// Each operation asks this guard whether the triangles it is about to
// create strictly cross any live triangle that shares none of their
// vertices — the same test reseed_feature_chains uses for its chords.
//
// The AABB tree is built once per operation pass over the current
// triangles and refit grow-only after every accepted change, so it stays
// conservative (stale slack only costs extra exact tests).

#include <array>
#include <cstdint>
#include <vector>

#include <Eigen/Core>

#include "vox2tet/core/types.hpp"
#include "vox2tet/mesh/half_edge.hpp"
#include "vox2tet/remesh/reseed.hpp"

namespace vox2tet::remesh::detail {

class IsectGuard {
public:
    // `he` and `xyz` are observed live (the caller keeps mutating them);
    // `dead_h`, when given, flags half-edges of deleted triangles.
    IsectGuard(const mesh::HalfEdges& he, const Coords& xyz,
               const std::vector<std::uint8_t>* dead_h = nullptr)
        : he_(he), xyz_(xyz), dead_h_(dead_h) {
        const Eigen::Index n = he_.rows();
        tri_of_.assign(static_cast<std::size_t>(n), -1);
        std::vector<std::array<Eigen::Vector3d, 2>> boxes;
        for (Eigen::Index i = 0; i < n; ++i) {
            const u32 h = static_cast<u32>(i);
            if (tri_of_[h] >= 0) continue;
            const int t = static_cast<int>(rep_.size());
            rep_.push_back(h);
            assign_rows(t, h);
            boxes.push_back(box(t));
        }
        tree_.build(boxes);
    }

    // Triangle id owning half-edge h.
    int tri_of(u32 h) const { return tri_of_[h]; }

    // Re-assign the three rows of the next-cycle through h0 to triangle t
    // (needed after an edge flip, which permutes rows between the two
    // triangles), then refit its box.
    void reassign(int t, u32 h0) {
        rep_[static_cast<std::size_t>(t)] = h0;
        assign_rows(t, h0);
        refresh(t);
    }

    // Grow-only refit of triangle t to its current geometry.
    void refresh(int t) {
        const auto b = box(t);
        tree_.grow_item(t, b[0], b[1]);
    }

    // True when triangle (a, b, c) strictly crosses a live triangle that
    // shares none of a, b, c, x0, x1 (pass kNone for unused slots).
    bool crosses(u32 a, u32 b, u32 c, u32 x0 = kNone, u32 x1 = kNone) const {
        const Eigen::Vector3d A[3] = {xyz_.row(a).transpose(),
                                      xyz_.row(b).transpose(),
                                      xyz_.row(c).transpose()};
        const Eigen::Vector3d lo = A[0].cwiseMin(A[1]).cwiseMin(A[2]);
        const Eigen::Vector3d hi = A[0].cwiseMax(A[1]).cwiseMax(A[2]);
        bool hit = false;
        tree_.for_box(lo, hi, [&](int t) {
            if (hit) return;
            const u32 r = rep_[static_cast<std::size_t>(t)];
            if (dead_h_ && (*dead_h_)[r]) return;
            u32 v[3];
            verts(t, v);
            for (u32 x : v)
                if (x == a || x == b || x == c || x == x0 || x == x1) return;
            const Eigen::Vector3d B[3] = {xyz_.row(v[0]).transpose(),
                                          xyz_.row(v[1]).transpose(),
                                          xyz_.row(v[2]).transpose()};
            for (int k = 0; k < 3 && !hit; ++k) {
                if (reseed_detail::segment_hits_triangle(A[k], A[(k + 1) % 3],
                                                         B[0], B[1], B[2]) ||
                    reseed_detail::segment_hits_triangle(B[k], B[(k + 1) % 3],
                                                         A[0], A[1], A[2]))
                    hit = true;
            }
        });
        return hit;
    }

    // Live vertex ids of the triangle owning half-edge h.
    void verts_of(u32 h, u32 out[3]) const { verts(tri_of_[h], out); }

    static constexpr u32 kNone = 0xFFFFFFFFu;

private:
    void assign_rows(int t, u32 h0) {
        const u32 h1 = he_(h0, 1), h2 = he_(h1, 1);
        tri_of_[h0] = tri_of_[h1] = tri_of_[h2] = t;
    }
    void verts(int t, u32 out[3]) const {
        const u32 h  = rep_[static_cast<std::size_t>(t)];
        const u32 h1 = he_(h, 1);
        out[0] = he_(h, 0);
        out[1] = he_(h1, 0);
        out[2] = he_(he_(h1, 1), 0);
    }
    std::array<Eigen::Vector3d, 2> box(int t) const {
        u32 v[3];
        verts(t, v);
        Eigen::Vector3d lo = xyz_.row(v[0]).transpose(), hi = lo;
        for (int k = 1; k < 3; ++k) {
            const Eigen::Vector3d p = xyz_.row(v[k]).transpose();
            lo = lo.cwiseMin(p);
            hi = hi.cwiseMax(p);
        }
        return {lo, hi};
    }

    const mesh::HalfEdges&            he_;
    const Coords&                     xyz_;
    const std::vector<std::uint8_t>*  dead_h_;
    std::vector<int>                  tri_of_;
    std::vector<u32>                  rep_;
    reseed_detail::AabbTree           tree_;
};

}  // namespace vox2tet::remesh::detail
