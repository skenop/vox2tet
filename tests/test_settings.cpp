#include <filesystem>
#include <gtest/gtest.h>

#include "vox2tet/core/settings.hpp"

TEST(Settings, RoundTripJson) {
    namespace fs = std::filesystem;
    fs::path tmp = fs::temp_directory_path() / "vox2tet_settings_test.json";
    fs::remove(tmp);

    vox2tet::Settings s1;
    s1.ncpus           = 8;
    s1.out_path_base   = "./tests/out/JMA_X/JMA_X";
    s1.input_img_file  = "./tests/data/JMA_X.tif";
    s1.smooth_alpha    = 0.25;
    s1.do_x_rotation   = false;
    s1.do_2x2patterns  = false;
    s1.n_remesh_itr    = 9;
    s1.Lmax            = 12.5;
    s1.dump_json(tmp.string());

    vox2tet::Settings s2;
    s2.load_json(tmp.string());
    EXPECT_EQ(s2.ncpus, 8);
    EXPECT_EQ(s2.out_path_base, "./tests/out/JMA_X/JMA_X");
    EXPECT_DOUBLE_EQ(s2.smooth_alpha, 0.25);
    EXPECT_FALSE(s2.do_x_rotation);
    EXPECT_FALSE(s2.do_2x2patterns);
    EXPECT_EQ(s2.n_remesh_itr, 9);
    EXPECT_DOUBLE_EQ(s2.Lmax, 12.5);
}
