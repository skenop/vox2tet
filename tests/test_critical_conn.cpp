#include <cstdint>
#include <gtest/gtest.h>

#include "vox2tet/image/critical_conn.hpp"
#include "vox2tet/image/volume.hpp"

namespace v = vox2tet;

namespace {

v::image::Volume make_u8(std::array<std::size_t, 3> shape,
                          const std::vector<std::uint8_t>& data) {
    v::image::Volume out;
    out.dtype = v::image::VoxelType::U8;
    out.shape = shape;
    out.bytes = data;
    return out;
}

}  // namespace

TEST(CriticalConn, NoIssueOnSolidBlock) {
    auto vox = make_u8({3, 3, 3}, std::vector<std::uint8_t>(27, 5));
    auto vert = v::image::find_vertex_connectivity(vox);
    auto edge = v::image::find_edge_connectivity(vox);
    EXPECT_EQ(vert.n_conn, 0u);
    EXPECT_EQ(edge.m2, 0u);
    EXPECT_EQ(edge.m3, 0u);
}

TEST(CriticalConn, DetectsDiagonalVertexConnection) {
    // 2x2x2 cell with two diagonally-opposite labels of 1 and six 0s
    // surrounding — classic vertex-only critical pattern (v0=v7=1, rest 0).
    //
    //   z=0 plane:  1 0   z=1 plane: 0 0
    //               0 0              0 1
    //
    std::vector<std::uint8_t> d(8, 0);
    d[0] = 1;  // (z=0,y=0,x=0)
    d[7] = 1;  // (z=1,y=1,x=1)
    auto vox = make_u8({2, 2, 2}, d);
    auto vert = v::image::find_vertex_connectivity(vox);
    EXPECT_EQ(vert.n_conn, 1u);
    EXPECT_EQ(vert.by_diagonal[0].size(), 1u);
}

TEST(CriticalConn, DetectsEdge2MAmbiguity) {
    // In a 2x2 plane: (v1==v4) AND (v2==v3), v1 != v2. Classic 2-material
    // edge ambiguity.
    //   0 1
    //   1 0
    std::vector<std::uint8_t> d = {0, 1, 1, 0};
    auto vox = make_u8({1, 2, 2}, d);
    auto edge = v::image::find_edge_connectivity(vox);
    EXPECT_EQ(edge.m2, 1u);
    EXPECT_EQ(edge.m3, 0u);
}
