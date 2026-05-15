#include <cstdint>
#include <filesystem>
#include <gtest/gtest.h>

#include "vox2tet/core/settings.hpp"
#include "vox2tet/image/image_utils.hpp"
#include "vox2tet/image/volume.hpp"

namespace v = vox2tet;
namespace fs = std::filesystem;

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

TEST(RemoveSmall, ReplacesIsolatedDotWithMajority) {
    // 5x5x1 image — background 0 with a single stray "9" at (0, 2, 2).
    // With max_remove_size = 1 the stray should be replaced by 0.
    fs::path tmpdir = fs::temp_directory_path() / "vox2tet_remove_small_test";
    fs::remove_all(tmpdir);
    fs::create_directories(tmpdir);

    v::Settings s;
    s.out_path_base   = (tmpdir / "img").string();
    s.connectivity    = 3;
    s.max_remove_size = 1;

    std::vector<std::uint8_t> data(25, 0);
    data[(0 * 5 + 2) * 5 + 2] = 9;
    auto vox = make_u8({1, 5, 5}, data);

    auto r = v::image::remove_small_regions(s, vox);
    ASSERT_EQ(r.voxels.shape[0], 1u);
    ASSERT_EQ(r.voxels.shape[1], 5u);
    ASSERT_EQ(r.voxels.shape[2], 5u);

    const auto* p = reinterpret_cast<const std::uint8_t*>(r.voxels.bytes.data());
    for (std::size_t i = 0; i < 25; ++i) EXPECT_EQ(p[i], 0u);

    // The expected output artefacts should exist on disk.
    EXPECT_TRUE(fs::is_regular_file(s.out_path_base + "_C3_R1.tif"));
    EXPECT_TRUE(fs::is_regular_file(s.out_path_base + "_L_C3.tif"));
    EXPECT_TRUE(fs::is_regular_file(s.out_path_base + "_L_C3_R1.tif"));
    EXPECT_TRUE(fs::is_regular_file(s.out_path_base + "_L_C3.txt"));
    EXPECT_TRUE(fs::is_regular_file(s.out_path_base + "_L_C3_R1.txt"));
}

TEST(RemoveSmall, KeepsLargeRegion) {
    // Two regions: a 4-cell "9" block (size 4) and surrounding "0"s. With
    // max_remove_size = 2 only true single-stray voxels would be removed;
    // the 4-cell block must stay intact.
    fs::path tmpdir = fs::temp_directory_path() / "vox2tet_remove_small_test_keep";
    fs::remove_all(tmpdir);
    fs::create_directories(tmpdir);

    v::Settings s;
    s.out_path_base   = (tmpdir / "img").string();
    s.connectivity    = 1;     // face-adjacent → the 4-cell block stays one CC
    s.max_remove_size = 2;

    std::vector<std::uint8_t> data(25, 0);
    data[(0 * 5 + 1) * 5 + 1] = 9;
    data[(0 * 5 + 1) * 5 + 2] = 9;
    data[(0 * 5 + 2) * 5 + 1] = 9;
    data[(0 * 5 + 2) * 5 + 2] = 9;
    auto vox = make_u8({1, 5, 5}, data);

    auto r = v::image::remove_small_regions(s, vox);
    const auto* p = reinterpret_cast<const std::uint8_t*>(r.voxels.bytes.data());
    EXPECT_EQ(p[(0 * 5 + 1) * 5 + 1], 9u);
    EXPECT_EQ(p[(0 * 5 + 1) * 5 + 2], 9u);
    EXPECT_EQ(p[(0 * 5 + 2) * 5 + 1], 9u);
    EXPECT_EQ(p[(0 * 5 + 2) * 5 + 2], 9u);
}
