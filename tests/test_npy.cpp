#include <cstdint>
#include <filesystem>
#include <gtest/gtest.h>

#include "vox2tet/io/npy.hpp"

TEST(Npy, RoundTrip1D) {
    namespace fs = std::filesystem;
    fs::path tmp = fs::temp_directory_path() / "vox2tet_npy_1d.npy";
    fs::remove(tmp);

    std::vector<std::uint32_t> data = {0, 1, 2, 3, 4, 5};
    std::vector<std::size_t> shape = {data.size()};
    vox2tet::npy::write<std::uint32_t>(tmp.string(), shape, data.data(), data.size());

    std::vector<std::size_t> rs;
    auto loaded = vox2tet::npy::read<std::uint32_t>(tmp.string(), rs);
    ASSERT_EQ(rs.size(), 1u);
    EXPECT_EQ(rs[0], data.size());
    EXPECT_EQ(loaded, data);
}

TEST(Npy, RoundTrip2D) {
    namespace fs = std::filesystem;
    fs::path tmp = fs::temp_directory_path() / "vox2tet_npy_2d.npy";
    fs::remove(tmp);

    // 3x2 array of doubles
    std::vector<double> data = {0.0, 1.0,
                                2.0, 3.0,
                                4.0, 5.0};
    std::vector<std::size_t> shape = {3, 2};
    vox2tet::npy::write<double>(tmp.string(), shape, data.data(), data.size());

    std::vector<std::size_t> rs;
    auto loaded = vox2tet::npy::read<double>(tmp.string(), rs);
    ASSERT_EQ(rs.size(), 2u);
    EXPECT_EQ(rs[0], 3u);
    EXPECT_EQ(rs[1], 2u);
    EXPECT_EQ(loaded, data);
}
