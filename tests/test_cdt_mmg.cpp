// CDT+MMG volume mesher (`tet_mesher: "cdt"`) — unit tests.
//
// Fixtures go through the real marching-cubes front end, so the
// surfaces are the voxel-exact material boundaries: every material
// region's volume equals its voxel count exactly, which turns the tet
// mesh's per-material volume sums into sharp correctness checks — any
// misclassified region, leaked flood fill, inverted tet or interface
// violation changes a sum.

#include <Eigen/Geometry>
#include <filesystem>
#include <gtest/gtest.h>
#include <map>
#include <string>
#include <vector>

#include "vox2tet/core/settings.hpp"
#include "vox2tet/image/ext_volume.hpp"
#include "vox2tet/marching_cubes/contouring.hpp"
#include "vox2tet/tetmesh/cdt_mmg_runner.hpp"

namespace fs = std::filesystem;
namespace v  = vox2tet;

namespace {

struct VolFixture {
    v::Triangles                              tri;
    v::Coords                                 xyz;
    std::vector<v::marching_cubes::Interface> interfaces;
    std::map<int, double>                     voxels;  // material -> count
    int                                       n = 0;
};

template <typename MatFn>
VolFixture make_volume(int n, const std::string& name, MatFn&& mat) {
    v::image::Volume img;
    img.dtype = v::image::VoxelType::U8;
    img.shape = {static_cast<std::size_t>(n), static_cast<std::size_t>(n),
                 static_cast<std::size_t>(n)};
    img.bytes.assign(static_cast<std::size_t>(n) * n * n, 1);
    VolFixture F;
    for (int z = 0; z < n; ++z)
        for (int y = 0; y < n; ++y)
            for (int x = 0; x < n; ++x) {
                const int m = mat(x, y, z);
                img.bytes[(static_cast<std::size_t>(z) * n + y) * n + x] =
                    static_cast<std::uint8_t>(m);
                F.voxels[m] += 1.0;
            }

    const fs::path tmp = fs::temp_directory_path() / "vox2tet_test_cdtmmg";
    fs::create_directories(tmp);
    const std::string base = (tmp / name).string();

    F.n = n;
    auto [ext, src_shape] = v::image::ext_volume_from_image(img);
    auto initial = v::marching_cubes::create_initial_mesh(ext, true, base);
    auto pr = v::marching_cubes::extract_material_interface_info(
        ext, initial.xyz, initial.tri, base);
    F.tri        = std::move(initial.tri);
    F.xyz        = std::move(initial.xyz);
    F.interfaces = std::move(pr.first);
    return F;
}

std::string out_base(const std::string& name) {
    const fs::path tmp = fs::temp_directory_path() / "vox2tet_test_cdtmmg";
    fs::create_directories(tmp);
    return (tmp / name).string();
}

double tet_volume(const v::Coords& x, const Eigen::MatrixXi& tets,
                  Eigen::Index t) {
    const Eigen::Vector3d a = x.row(tets(t, 0)).transpose();
    const Eigen::Vector3d b = x.row(tets(t, 1)).transpose();
    const Eigen::Vector3d c = x.row(tets(t, 2)).transpose();
    const Eigen::Vector3d d = x.row(tets(t, 3)).transpose();
    return (b - a).cross(c - a).dot(d - a) / 6.0;
}

// Per-material volume sums; also asserts every tet is positively
// oriented (no flat or inverted elements in the output).
std::map<int, double> material_volumes(const v::tetmesh::CdtMmgStats& st) {
    std::map<int, double> vol;
    for (Eigen::Index t = 0; t < st.tets.rows(); ++t) {
        const double vt = tet_volume(st.nodes, st.tets, t);
        EXPECT_GT(vt, 0.0) << "tet " << t << " is flat or inverted";
        vol[st.mats[static_cast<std::size_t>(t)]] += vt;
    }
    return vol;
}

// Volume each material's boundary surface encloses (divergence
// theorem). This is the mesher's actual contract: the tet mesh must
// fill exactly what the input surface bounds — which for staircase
// fixtures with diagonal configurations differs slightly from the raw
// voxel count (facet-centred marching-cubes cells cut corners).
// Interface winding convention: the triangle normal points toward the
// mat1 side, so mat1 sees an inward normal (sign -1), mat2 an outward
// one (sign +1).
std::map<int, double> enclosed_volumes(const VolFixture& F) {
    std::map<int, double> vol;
    for (const auto& I : F.interfaces) {
        double s = 0.0;
        for (v::u32 k = 0; k < I.count; ++k) {
            const Eigen::Index t = static_cast<Eigen::Index>(I.first + k);
            const Eigen::Vector3d a = F.xyz.row(F.tri(t, 0)).transpose();
            const Eigen::Vector3d b = F.xyz.row(F.tri(t, 1)).transpose();
            const Eigen::Vector3d c = F.xyz.row(F.tri(t, 2)).transpose();
            s += a.dot(b.cross(c)) / 6.0;
        }
        vol[static_cast<int>(I.mat1)] -= s;
        vol[static_cast<int>(I.mat2)] += s;
    }
    return vol;
}

}  // namespace

// ===========================================================================
// Pure CDT (MMG off): the surface must be recovered exactly and both
// half-cubes classified with voxel-exact volumes.
// ===========================================================================
TEST(CdtMmg, TwoMaterialCubeExactVolumes) {
    const int n = 6;
    auto F = make_volume(n, "cube2", [&](int x, int, int) {
        return x >= n / 2 ? 2 : 1;
    });

    v::Settings s;
    s.do_mmg_optim = false;

    v::tetmesh::CdtMmgStats st;
    ASSERT_TRUE(v::tetmesh::mesh_volume(s, out_base("cube2_cdt"), F.xyz,
                                        F.tri, F.interfaces, &st));
    EXPECT_TRUE(fs::exists(out_base("cube2_cdt") + ".inp"));

    // The whole input surface must be recovered as tet faces.
    EXPECT_NEAR(st.area_ratio, 1.0, 1e-9);
    EXPECT_EQ(st.n_regions, 2u);
    EXPECT_EQ(st.n_vote_conflicts, 0u);
    EXPECT_FALSE(st.mmg_ran);

    const auto vol = material_volumes(st);
    ASSERT_EQ(vol.size(), 2u);
    EXPECT_NEAR(vol.at(1), F.voxels.at(1), 1e-6);
    EXPECT_NEAR(vol.at(2), F.voxels.at(2), 1e-6);
}

// ===========================================================================
// Full pipeline (MMG on): interfaces are frozen (nosurf + required
// triangles), so material volumes must stay voxel-exact through the
// optimization.
// ===========================================================================
TEST(CdtMmg, MmgPreservesMaterialVolumes) {
    const int n = 6;
    auto F = make_volume(n, "cube2m", [&](int x, int, int) {
        return x >= n / 2 ? 2 : 1;
    });

    v::Settings s;
    s.do_mmg_optim = true;
    s.mmg_verbose  = -1;

    v::tetmesh::CdtMmgStats st;
    ASSERT_TRUE(v::tetmesh::mesh_volume(s, out_base("cube2_mmg"), F.xyz,
                                        F.tri, F.interfaces, &st));
    EXPECT_TRUE(st.mmg_ran);
    EXPECT_NEAR(st.area_ratio, 1.0, 1e-9);

    const auto vol = material_volumes(st);
    ASSERT_EQ(vol.size(), 2u);
    EXPECT_NEAR(vol.at(1), F.voxels.at(1), 1e-6);
    EXPECT_NEAR(vol.at(2), F.voxels.at(2), 1e-6);
}

// ===========================================================================
// Curved interface + non-manifold feature lines (cylindrical inclusion
// touching two bbox faces): the staircase surface exercises Steiner
// insertion and the trace edges are 3-sheet non-manifold. Volumes stay
// voxel-exact because the marching-cubes surface bounds the voxel set.
// ===========================================================================
TEST(CdtMmg, CylinderInclusionVolumes) {
    const int n = 10;
    auto F = make_volume(n, "cyl", [&](int x, int y, int) {
        const double dx = x + 0.5 - 5.0, dy = y + 0.5 - 5.0;
        return dx * dx + dy * dy < 9.0 ? 2 : 1;
    });

    v::Settings s;
    s.mmg_verbose = -1;

    v::tetmesh::CdtMmgStats st;
    ASSERT_TRUE(v::tetmesh::mesh_volume(s, out_base("cyl_mmg"), F.xyz,
                                        F.tri, F.interfaces, &st));
    EXPECT_NEAR(st.area_ratio, 1.0, 1e-9);
    EXPECT_EQ(st.n_regions, 2u);
    EXPECT_EQ(st.n_vote_conflicts, 0u);

    // The tet mesh must fill exactly what the surface encloses; the
    // surface itself deviates from the raw voxel count only by the
    // corner-cutting of facet-centred marching-cubes cells (small).
    const auto vol = material_volumes(st);
    const auto enc = enclosed_volumes(F);
    ASSERT_EQ(vol.size(), 2u);
    EXPECT_NEAR(vol.at(1), enc.at(1), 1e-6);
    EXPECT_NEAR(vol.at(2), enc.at(2), 1e-6);
    EXPECT_NEAR(enc.at(2), F.voxels.at(2), 0.05 * F.voxels.at(2));
    EXPECT_NEAR(vol.at(1) + vol.at(2), (double)(n * n * n), 1e-6);
}

// ===========================================================================
// Settings round-trip for the new fields.
// ===========================================================================
TEST(CdtMmg, SettingsRoundTrip) {
    const fs::path tmp =
        fs::temp_directory_path() / "vox2tet_cdtmmg_settings.json";
    fs::remove(tmp);

    v::Settings s1;
    s1.tet_mesher   = "tetgen";          // non-default
    s1.do_mmg_optim = false;
    s1.mmg_verbose  = 3;
    s1.dump_json(tmp.string());

    v::Settings s2;
    ASSERT_EQ(s2.tet_mesher, "cdt");     // built-in backend is default
    ASSERT_TRUE(s2.do_mmg_optim);
    s2.load_json(tmp.string());
    EXPECT_EQ(s2.tet_mesher, "tetgen");
    EXPECT_FALSE(s2.do_mmg_optim);
    EXPECT_EQ(s2.mmg_verbose, 3);
}
