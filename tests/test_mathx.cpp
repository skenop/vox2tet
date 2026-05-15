#include <gtest/gtest.h>

#include "vox2tet/mesh/mathx.hpp"

TEST(Mathx, ScalarAngles) {
    Eigen::Vector3d p0(0, 0, 0), p1(1, 0, 0), p2(0, 1, 0);
    EXPECT_NEAR(vox2tet::mathx::calc_angle(p0, p1, p2), 90.0, 1e-9);

    Eigen::Vector3d v1(1, 0, 0), v2(1, 1, 0);
    EXPECT_NEAR(vox2tet::mathx::calc_angle2(v1, v2), 45.0, 1e-9);
}

TEST(Mathx, TriArea) {
    using M = vox2tet::MatrixNx3<double>;
    M p1(1, 3); M p2(1, 3); M p3(1, 3);
    p1.row(0) << 0, 0, 0;
    p2.row(0) << 1, 0, 0;
    p3.row(0) << 0, 1, 0;
    auto a = vox2tet::mathx::calc_tri_area_v(p1, p2, p3);
    EXPECT_NEAR(a[0], 0.5, 1e-9);
}

TEST(Mathx, TriNormUnit) {
    using M = vox2tet::MatrixNx3<double>;
    M p1(1, 3); M p2(1, 3); M p3(1, 3);
    p1.row(0) << 0, 0, 0;
    p2.row(0) << 1, 0, 0;
    p3.row(0) << 0, 1, 0;
    auto n = vox2tet::mathx::calc_tri_norm_v(p1, p2, p3);
    EXPECT_NEAR(n(0, 0), 0.0, 1e-9);
    EXPECT_NEAR(n(0, 1), 0.0, 1e-9);
    EXPECT_NEAR(n(0, 2), 1.0, 1e-9);
}

TEST(Mathx, DihedralAngles) {
    using M = vox2tet::MatrixNx3<double>;
    M p01(4, 3), p02(4, 3), p1(4, 3), p2(4, 3);
    for (int i = 0; i < 4; ++i) {
        p01.row(i) << 0, 0, 0;
        p02.row(i) << 1, 0, 0;
        p1.row(i)  << 0.5, 0, 1;
    }
    p2.row(0) << 0.5, 1, 1;     // 45°
    p2.row(1) << 0.5, 1, 0;     // 90°
    p2.row(2) << 0.5, 1, -1;    // 135°
    p2.row(3) << 0.5, 0, -1;    // 180°
    auto an = vox2tet::mathx::calc_dihedral_angle_v_directional(p01, p02, p1, p2);
    EXPECT_NEAR(an[0],  45.0, 1e-6);
    EXPECT_NEAR(an[1],  90.0, 1e-6);
    EXPECT_NEAR(an[2], 135.0, 1e-6);
    EXPECT_NEAR(an[3], 180.0, 1e-6);
}
