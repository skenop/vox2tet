#include <gtest/gtest.h>

#include "vox2tet/marching_cubes/m3c.hpp"

namespace mc = vox2tet::marching_cubes;

TEST(M3c, Lut2MaterialsEmptyPattern) {
    // Pattern 0: all 8 corners are colour 0 → no surface.
    auto lut = mc::lut2materials();
    for (int j = 0; j < 18; ++j) EXPECT_EQ(lut[0 * 18 + j], 19);
    // Pattern 255: all 8 corners are colour 1 → also no surface.
    for (int j = 0; j < 18; ++j) EXPECT_EQ(lut[255 * 18 + j], 19);
}

TEST(M3c, Lut2MaterialsSingleCornerHas3Triangle) {
    // Pattern 128 = 10000000: only corner 0 is colour 1, rest are 0.
    // Classic MC corner cut → 1 triangle.
    auto lut = mc::lut2materials();
    int n_set = 0;
    for (int j = 0; j < 18; ++j) {
        if (lut[128 * 18 + j] != 19) ++n_set;
    }
    EXPECT_EQ(n_set, 3);  // one triangle = three node ids
}

TEST(M3c, OppositeEdges) {
    EXPECT_EQ(mc::opposite_edge(0),  3);
    EXPECT_EQ(mc::opposite_edge(7),  4);
    EXPECT_EQ(mc::opposite_edge(8), 11);
    EXPECT_EQ(mc::opposite_edge(12), 19);
    EXPECT_EQ(mc::opposite_edge(19), 19);
}
