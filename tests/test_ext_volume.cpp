#include <cstdint>
#include <gtest/gtest.h>

#include "vox2tet/image/ext_volume.hpp"

namespace v = vox2tet;

TEST(ExtVolume, AddsOneVoxelMarginWith26UniqueLabels) {
    // 2x2x2 image, all zeros — max label = 0, so the 26 boundary labels
    // become 1..26 (one per face/edge/corner region).
    v::image::Volume img;
    img.dtype = v::image::VoxelType::U8;
    img.shape = {2, 2, 2};
    img.bytes.assign(8, 0);

    auto [ext, src_shape] = v::image::ext_volume_from_image(img);
    EXPECT_EQ(ext.shape[0], 4u);
    EXPECT_EQ(ext.shape[1], 4u);
    EXPECT_EQ(ext.shape[2], 4u);

    // The 8 corner voxels of the extended volume must all carry one of the
    // 8 unique corner labels (max_label+19..+26).
    const auto* p = reinterpret_cast<const std::uint8_t*>(ext.bytes.data());
    auto at = [&](std::size_t z, std::size_t y, std::size_t x) {
        return p[(z * 4 + y) * 4 + x];
    };
    std::set<int> corners = {at(0,0,0), at(0,0,3), at(0,3,0), at(0,3,3),
                              at(3,0,0), at(3,0,3), at(3,3,0), at(3,3,3)};
    EXPECT_EQ(corners.size(), 8u);

    // Interior is still 0.
    EXPECT_EQ(at(1,1,1), 0);
}
