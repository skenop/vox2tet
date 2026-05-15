// End-to-end "initial-surface" regression: drives `create_initial_mesh`
// and `extract_material_interface_info` against the locked-in golden
// artefacts under tests/data/JMA_{10,30}/out/ and verifies:
//   1. byte-identical _xyz.npy / _att.npy / _ntp.npy
//   2. topological identity of the triangle list (set + canonical form)
//
// Skipped if the prepared TIFF or the golden out/ directory is absent
// (so the test passes cleanly on a fresh check-out without the optional
// JMA fixture data).

#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <set>
#include <string>
#include <vector>

#include "vox2tet/brep/brep.hpp"
#include "vox2tet/image/ext_volume.hpp"
#include "vox2tet/io/image_io.hpp"
#include "vox2tet/io/npy.hpp"
#include "vox2tet/marching_cubes/contouring.hpp"
#include "vox2tet/mesh/half_edge.hpp"

namespace fs = std::filesystem;
namespace v  = vox2tet;

namespace {

bool files_equal(const fs::path& a, const fs::path& b) {
    std::ifstream fa(a, std::ios::binary), fb(b, std::ios::binary);
    if (!fa || !fb) return false;
    constexpr std::size_t kBuf = 64 * 1024;
    std::vector<char> ba(kBuf), bb(kBuf);
    while (fa && fb) {
        fa.read(ba.data(), kBuf);
        fb.read(bb.data(), kBuf);
        const auto na = fa.gcount();
        const auto nb = fb.gcount();
        if (na != nb) return false;
        if (na == 0) return true;
        if (std::memcmp(ba.data(), bb.data(), na) != 0) return false;
    }
    return true;
}

void run_case(const std::string& py_prepared_tif,
              const std::string& py_dir,
              const std::string& base_name) {
    if (!fs::exists(py_prepared_tif) || !fs::exists(py_dir)) {
        GTEST_SKIP() << "missing fixture: " << py_prepared_tif;
    }
    fs::path tmp = fs::temp_directory_path() / ("vox2tet_test_" + base_name);
    fs::remove_all(tmp);
    fs::create_directories(tmp);
    const std::string out_base = (tmp / base_name).string();

    auto img = v::io::read_image(py_prepared_tif, std::nullopt, /*compress=*/true);
    auto [ext_vox, src_shape] = v::image::ext_volume_from_image(img);
    auto initial = v::marching_cubes::create_initial_mesh(
        ext_vox, /*do_2x2=*/true, out_base);
    auto [interfaces, node_mask] = v::marching_cubes::extract_material_interface_info(
        ext_vox, initial.xyz, initial.tri, out_base);

    // Phase 8 — half-edge + brep.
    auto he = v::mesh::triangles_to_hedges(initial.tri, &interfaces);
    v::mesh::save_hedges_npy(out_base + "_hedges.npy", he);

    // Byte-identical artefacts.
    EXPECT_TRUE(files_equal(out_base + "_xyz.npy", fs::path(py_dir) / (base_name + "_xyz.npy")))
        << "_xyz.npy mismatch";
    EXPECT_TRUE(files_equal(out_base + "_att.npy", fs::path(py_dir) / (base_name + "_att.npy")))
        << "_att.npy mismatch";
    EXPECT_TRUE(files_equal(out_base + "_ntp.npy", fs::path(py_dir) / (base_name + "_ntp.npy")))
        << "_ntp.npy mismatch";

    // _hedges.npy: same shape (N*3, 5) uint32; verify by size match.
    EXPECT_EQ(fs::file_size(out_base + "_hedges.npy"),
              fs::file_size(fs::path(py_dir) / (base_name + "_hedges.npy")))
        << "_hedges.npy size differs";

    // Topological identity: same vertex set + same canonical triangle set.
    std::vector<std::size_t> s;
    auto load_verts = [&](const std::string& p) {
        std::vector<std::array<float, 3>> out;
        auto raw = v::npy::read<float>(p, s);
        out.resize(s[0]);
        for (std::size_t i = 0; i < s[0]; ++i)
            out[i] = {raw[i * 3 + 0], raw[i * 3 + 1], raw[i * 3 + 2]};
        return out;
    };
    auto load_tris = [&](const std::string& p) {
        std::vector<std::array<std::uint32_t, 3>> out;
        auto raw = v::npy::read<std::uint32_t>(p, s);
        out.resize(s[0]);
        for (std::size_t i = 0; i < s[0]; ++i)
            out[i] = {raw[i * 3 + 0], raw[i * 3 + 1], raw[i * 3 + 2]};
        return out;
    };
    auto V = load_verts(out_base + "_xyz.npy");
    auto T = load_tris (out_base + "_tr.npy");
    auto Vp = load_verts((fs::path(py_dir) / (base_name + "_xyz.npy")).string());
    auto Tp = load_tris ((fs::path(py_dir) / (base_name + "_tr.npy")).string());

    EXPECT_EQ(V.size(), Vp.size());
    EXPECT_EQ(T.size(), Tp.size());

    auto canon = [&](const std::vector<std::array<std::uint32_t, 3>>& t,
                     const std::vector<std::array<float, 3>>& v) {
        std::vector<std::array<std::array<float, 3>, 3>> out(t.size());
        for (std::size_t i = 0; i < t.size(); ++i) {
            out[i] = {v[t[i][0]], v[t[i][1]], v[t[i][2]]};
            int min_i = 0;
            for (int k = 1; k < 3; ++k) if (out[i][k] < out[i][min_i]) min_i = k;
            if (min_i == 1) out[i] = {out[i][1], out[i][2], out[i][0]};
            else if (min_i == 2) out[i] = {out[i][2], out[i][0], out[i][1]};
        }
        std::sort(out.begin(), out.end());
        return out;
    };
    auto cmp_my = canon(T, V);
    auto cmp_py = canon(Tp, Vp);
    EXPECT_TRUE(cmp_my == cmp_py);
}

}  // namespace

TEST(JmaSurface, Jma10) {
    run_case(
        std::string(VOX2TET_TEST_DATA_DIR) + "/JMA_10/out/JMA_10_C3_R64.tif",
        std::string(VOX2TET_TEST_DATA_DIR) + "/JMA_10/out",
        "JMA_10");
}

TEST(JmaSurface, Jma30) {
    run_case(
        std::string(VOX2TET_TEST_DATA_DIR) + "/JMA_30/out/JMA_30_F.tif",
        std::string(VOX2TET_TEST_DATA_DIR) + "/JMA_30/out",
        "JMA_30");
}
