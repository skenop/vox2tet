#include <cstdint>
#include <filesystem>
#include <gtest/gtest.h>

#include "vox2tet/marching_cubes/lut.hpp"

TEST(Lut, LoadsAndSizesMatch) {
    namespace fs = std::filesystem;
    // Only run if the data dir is present (the binaries are produced by
    // tools/export_luts.py, not by the build).
    if (!fs::exists(fs::path(VOX2TET_DATA_DIR_TEST) / "lut8.bin")) {
        GTEST_SKIP() << "data/lut8.bin missing — run tools/export_luts.py";
    }
    const auto& L = vox2tet::marching_cubes::load_luts(VOX2TET_DATA_DIR_TEST);
    EXPECT_EQ(L.lut8.size, 16'777'216u);
    EXPECT_EQ(L.lut0_rows, 705u);
    EXPECT_EQ(L.lut2_rows, 6582u);
    EXPECT_EQ(L.lut2x2_rows, 3288u);
    EXPECT_EQ(L.facet_lut2d.size, 256u * 8u);
    // First row of lut0 should be (0,1,3,0,3,2,19,19,...) per
    // cellLUTarray.py line 27.
    EXPECT_EQ(L.lut0[0], 0);
    EXPECT_EQ(L.lut0[1], 1);
    EXPECT_EQ(L.lut0[2], 3);
    EXPECT_EQ(L.lut0[5], 2);
    EXPECT_EQ(L.lut0[6], 19);
}
