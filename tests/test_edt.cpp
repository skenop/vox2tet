#include <cmath>
#include <cstdint>
#include <gtest/gtest.h>

#include "vox2tet/image/edt.hpp"

namespace v = vox2tet;

TEST(Edt, SingleSourceManhattanShape) {
    // 5x5x5 mask, single source at the centre (2,2,2). For every voxel,
    // the EDT should point to that source.
    const std::array<std::size_t, 3> shape{5, 5, 5};
    std::vector<std::uint8_t> mask(125, 1);
    const std::size_t cz = 2, cy = 2, cx = 2;
    const std::size_t src_lin = (cz * 5 + cy) * 5 + cx;
    mask[src_lin] = 0;

    auto idx = v::image::distance_transform_edt_indices(mask, shape);
    for (std::size_t z = 0; z < 5; ++z)
    for (std::size_t y = 0; y < 5; ++y)
    for (std::size_t x = 0; x < 5; ++x) {
        std::size_t lin = (z * 5 + y) * 5 + x;
        EXPECT_EQ(idx.iz[lin], static_cast<int>(cz));
        EXPECT_EQ(idx.iy[lin], static_cast<int>(cy));
        EXPECT_EQ(idx.ix[lin], static_cast<int>(cx));
    }
}

TEST(Edt, TwoSourcesPicksClosest) {
    // 1x1x10 — two sources at x=0 and x=9. The voxel x=4 should pick x=0
    // (distance 4 < 5).
    const std::array<std::size_t, 3> shape{1, 1, 10};
    std::vector<std::uint8_t> mask(10, 1);
    mask[0] = 0;
    mask[9] = 0;
    auto idx = v::image::distance_transform_edt_indices(mask, shape);
    EXPECT_EQ(idx.ix[4], 0);
    EXPECT_EQ(idx.ix[5], 9);
    EXPECT_EQ(idx.ix[0], 0);
    EXPECT_EQ(idx.ix[9], 9);
}

TEST(Edt, DiagonalNearestPicked) {
    // 5x5x1 plane (treat as nz=1, ny=5, nx=5). Two candidate sources:
    //   A at (0, 0, 0) — Manhattan 2 from (0,1,1), Euclidean sqrt(2)
    //   B at (0, 0, 2) — Manhattan 2 from (0,1,1), Euclidean sqrt(2)
    // For (0, 2, 1): A distance sqrt(5), B distance sqrt(5) — tie.
    // For (0, 2, 0): A distance 2  vs B distance sqrt(8) — A wins.
    const std::array<std::size_t, 3> shape{1, 5, 5};
    std::vector<std::uint8_t> mask(25, 1);
    mask[(0 * 5 + 0) * 5 + 0] = 0;  // (y=0, x=0)
    mask[(0 * 5 + 0) * 5 + 2] = 0;  // (y=0, x=2)
    auto idx = v::image::distance_transform_edt_indices(mask, shape);

    const std::size_t q = (0 * 5 + 2) * 5 + 0;  // (y=2, x=0)
    EXPECT_EQ(idx.iy[q], 0);
    EXPECT_EQ(idx.ix[q], 0);
}
