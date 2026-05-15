#include <cstdint>
#include <gtest/gtest.h>

#include "vox2tet/image/label.hpp"

namespace v = vox2tet;

namespace {

v::image::Volume make_volume_u8(const std::vector<std::uint8_t>& d,
                                std::array<std::size_t, 3> shape) {
    v::image::Volume out;
    out.dtype = v::image::VoxelType::U8;
    out.shape = shape;
    out.bytes = d;
    return out;
}

}  // namespace

TEST(Label, TwoIsolatedRegionsSameColor) {
    // 2 1 1
    // 0 0 0
    // 1 1 2
    // Two grains of color 1 (one on top row, one on bottom row) plus
    // background 0 and a couple of color-2 cells.
    std::vector<std::uint8_t> data = {2, 1, 1,  0, 0, 0,  1, 1, 2};
    auto vol = make_volume_u8(data, {1, 3, 3});

    auto r = v::image::label_components(vol, /*conn=*/1);  // 6-neigh
    // Expect 4 components: {top-left "2"}, {top-row "1 1"}, {mid-row "000"},
    // {bottom row "1 1 2"} actually split: "1 1" and "2" because 6-conn.
    EXPECT_GE(r.n_components, 4);
}

TEST(Label, SingleSolidImage) {
    std::vector<std::uint8_t> data(2 * 2 * 2, 7);
    auto vol = make_volume_u8(data, {2, 2, 2});
    auto r = v::image::label_components(vol, 3);
    EXPECT_EQ(r.n_components, 1);
    EXPECT_EQ(r.materials_out.size(), 1u);
    EXPECT_EQ(r.materials_out[0], 7u);
    for (auto l : r.labels) EXPECT_EQ(l, 0);
}
