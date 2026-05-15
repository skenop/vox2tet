#include <cstdint>
#include <set>
#include <gtest/gtest.h>

#include "vox2tet/mesh/half_edge.hpp"

namespace v = vox2tet;
using v::mesh::HalfEdges;

namespace {

v::Triangles make_tetrahedron() {
    // 4 verts, 4 triangles — closed (manifold) mesh.
    v::Triangles t(4, 3);
    t << 0, 1, 2,
         0, 2, 3,
         0, 3, 1,
         1, 3, 2;
    return t;
}

v::Triangles make_two_triangles_shared_edge() {
    //   0───1
    //   │ ╲ │
    //   │  ╲│
    //   3───2
    v::Triangles t(2, 3);
    t << 0, 1, 2,
         0, 2, 3;
    return t;
}

}  // namespace

TEST(HalfEdge, BuildClosedManifold) {
    auto tri = make_tetrahedron();
    auto he = v::mesh::triangles_to_hedges(tri);
    EXPECT_EQ(he.rows(), tri.rows() * 3);
    // All 12 half-edges should have a valid opposite (no boundary on a tet).
    for (Eigen::Index i = 0; i < he.rows(); ++i) {
        EXPECT_NE(he(i, 3), static_cast<v::u32>(-1))
            << "half-edge " << i << " has no opposite (expected manifold)";
    }
}

TEST(HalfEdge, BuildBoundary) {
    auto tri = make_two_triangles_shared_edge();
    auto he = v::mesh::triangles_to_hedges(tri);
    EXPECT_EQ(he.rows(), 6);

    // The shared diagonal (0,2) is internal — both halves have opposite.
    // The 4 outer edges are boundary — opposite == sentinel.
    int n_internal = 0, n_boundary = 0;
    for (Eigen::Index i = 0; i < he.rows(); ++i) {
        if (he(i, 3) == static_cast<v::u32>(-1)) ++n_boundary;
        else                                      ++n_internal;
    }
    EXPECT_EQ(n_internal, 2);
    EXPECT_EQ(n_boundary, 4);
}

TEST(HalfEdge, OppositeIsSymmetric) {
    auto tri = make_tetrahedron();
    auto he = v::mesh::triangles_to_hedges(tri);
    for (Eigen::Index i = 0; i < he.rows(); ++i) {
        const v::u32 opp = he(i, 3);
        if (opp == static_cast<v::u32>(-1)) continue;
        EXPECT_EQ(he(static_cast<Eigen::Index>(opp), 3), static_cast<v::u32>(i))
            << "opposite of opposite must be self at i=" << i;
    }
}

TEST(HalfEdge, RoundTrip) {
    auto tri = make_tetrahedron();
    auto he = v::mesh::triangles_to_hedges(tri);
    auto back = v::mesh::hedges_to_triangles(he);
    ASSERT_EQ(back.rows(), tri.rows());

    // Canonicalise each triangle (rotate so min vertex is first) and check
    // set equality.
    auto canon = [](v::Triangles& T) {
        for (Eigen::Index i = 0; i < T.rows(); ++i) {
            int m = 0;
            for (int k = 1; k < 3; ++k) if (T(i, k) < T(i, m)) m = k;
            if (m == 1) { v::u32 a = T(i,1), b = T(i,2), c = T(i,0); T(i,0)=a; T(i,1)=b; T(i,2)=c; }
            else if (m == 2) { v::u32 a = T(i,2), b = T(i,0), c = T(i,1); T(i,0)=a; T(i,1)=b; T(i,2)=c; }
        }
    };
    canon(tri); canon(back);

    std::set<std::array<v::u32, 3>> A, B;
    for (Eigen::Index i = 0; i < tri.rows(); ++i)  A.insert({tri (i,0), tri (i,1), tri (i,2)});
    for (Eigen::Index i = 0; i < back.rows(); ++i) B.insert({back(i,0), back(i,1), back(i,2)});
    EXPECT_EQ(A, B);
}

TEST(HalfEdge, NextPrevConsistency) {
    auto tri = make_tetrahedron();
    auto he = v::mesh::triangles_to_hedges(tri);
    for (Eigen::Index i = 0; i < he.rows(); ++i) {
        EXPECT_EQ(he(he(i, 1), 2), static_cast<v::u32>(i)) << "next.prev != self at " << i;
        EXPECT_EQ(he(he(i, 2), 1), static_cast<v::u32>(i)) << "prev.next != self at " << i;
    }
}

TEST(HalfEdge, BlinksOnManifoldEqualOpposite) {
    // On a closed manifold mesh, every blink should reproduce the
    // canonical opposite stored in column 3.
    auto tri = make_tetrahedron();
    auto he = v::mesh::triangles_to_hedges(tri);
    // Dummy xyz — tetrahedral corners. Coordinates only matter for the
    // count=4 path (not exercised on a tetrahedron).
    v::Coords xyz(4, 3);
    xyz << 0, 0, 0,
           1, 0, 0,
           0, 1, 0,
           0, 0, 1;
    auto blinks = v::mesh::create_blinks(he, xyz);
    ASSERT_EQ(blinks.size(), static_cast<std::size_t>(he.rows()));
    for (Eigen::Index i = 0; i < he.rows(); ++i) {
        EXPECT_EQ(blinks[static_cast<std::size_t>(i)], he(i, 3))
            << "blinks differ from opposite on manifold at h=" << i;
    }
}

TEST(HalfEdge, V2HBoundaryDetection) {
    auto tri = make_two_triangles_shared_edge();
    auto he = v::mesh::triangles_to_hedges(tri);
    auto v2h = v::mesh::get_v2h_lists(he);
    // 4 boundary vertices (0, 1, 2, 3); the diagonal edge endpoints (0, 2)
    // count as boundary because their other edges terminate at sentinels.
    int n_boundary = 0;
    for (auto b : v2h.is_boundary_vertex) if (b) ++n_boundary;
    EXPECT_EQ(n_boundary, 4);
}
