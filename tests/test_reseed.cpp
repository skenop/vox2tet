// Boundary-edge reseeding (Stages A + A2) — unit tests.
//
// Synthetic fixtures are built through the real marching-cubes front
// end (ext_volume → create_initial_mesh → extract_material_interface_info)
// so the node-type masks and interface grouping match what the pipeline
// hands to `reseed_feature_chains`.
//
// The strongest checks exploit planar geometry: bbox faces and flat
// internal interfaces are exactly planar, so a correct reseed preserves
// their total area to 1e-9 while removing vertices; any leaked, flipped
// or missing triangle changes the sum.

#include <Eigen/Geometry>
#include <array>
#include <filesystem>
#include <gtest/gtest.h>
#include <set>
#include <string>
#include <vector>

#include "vox2tet/brep/brep.hpp"
#include "vox2tet/image/ext_volume.hpp"
#include "vox2tet/marching_cubes/contouring.hpp"
#include "vox2tet/mesh/half_edge.hpp"
#include "vox2tet/remesh/remesh.hpp"
#include "vox2tet/remesh/reseed.hpp"
#include "vox2tet/remesh/smooth.hpp"

namespace fs = std::filesystem;
namespace v  = vox2tet;

namespace {

struct Fixture {
    v::Triangles                                tri;
    v::Coords                                   xyz;
    v::marching_cubes::NodeTypeMask             mask;
    std::vector<v::marching_cubes::Interface>   interfaces;
    int                                         n = 0;   // cube side (voxels)
};

// Build an n³ volume from a per-voxel material callback (x, y, z → id).
template <typename MatFn>
Fixture make_volume(int n, const std::string& name, MatFn&& mat) {
    v::image::Volume img;
    img.dtype = v::image::VoxelType::U8;
    img.shape = {static_cast<std::size_t>(n), static_cast<std::size_t>(n),
                 static_cast<std::size_t>(n)};
    img.bytes.assign(static_cast<std::size_t>(n) * n * n, 1);
    for (int z = 0; z < n; ++z)
        for (int y = 0; y < n; ++y)
            for (int x = 0; x < n; ++x)
                img.bytes[(static_cast<std::size_t>(z) * n + y) * n + x] =
                    static_cast<std::uint8_t>(mat(x, y, z));

    const fs::path tmp = fs::temp_directory_path() / "vox2tet_test_reseed";
    fs::create_directories(tmp);
    const std::string base = (tmp / name).string();

    Fixture F;
    F.n = n;
    auto [ext, src_shape] = v::image::ext_volume_from_image(img);
    auto initial = v::marching_cubes::create_initial_mesh(ext, true, base);
    auto pr = v::marching_cubes::extract_material_interface_info(
        ext, initial.xyz, initial.tri, base);
    F.tri        = std::move(initial.tri);
    F.xyz        = std::move(initial.xyz);
    F.interfaces = std::move(pr.first);
    F.mask       = std::move(pr.second);
    return F;
}

Fixture make_cube(int n, bool two_mat) {
    return make_volume(n, two_mat ? "cube2" : "cube1",
                       [&](int x, int, int) { return two_mat && x >= n / 2 ? 2 : 1; });
}

double total_area(const v::Triangles& tri, const v::Coords& xyz) {
    double A = 0;
    for (Eigen::Index t = 0; t < tri.rows(); ++t) {
        const Eigen::Vector3d a = xyz.row(tri(t, 0)).transpose();
        const Eigen::Vector3d b = xyz.row(tri(t, 1)).transpose();
        const Eigen::Vector3d c = xyz.row(tri(t, 2)).transpose();
        A += 0.5 * (b - a).cross(c - a).norm();
    }
    return A;
}

// Total area of triangles lying entirely in an axis plane coord==value.
double plane_area(const v::Triangles& tri, const v::Coords& xyz,
                  int axis, double value) {
    double A = 0;
    for (Eigen::Index t = 0; t < tri.rows(); ++t) {
        bool in_plane = true;
        for (int k = 0; k < 3; ++k)
            in_plane = in_plane && (xyz(tri(t, k), axis) == value);
        if (!in_plane) continue;
        const Eigen::Vector3d a = xyz.row(tri(t, 0)).transpose();
        const Eigen::Vector3d b = xyz.row(tri(t, 1)).transpose();
        const Eigen::Vector3d c = xyz.row(tri(t, 2)).transpose();
        A += 0.5 * (b - a).cross(c - a).norm();
    }
    return A;
}

// Count half-edges whose opposite is missing or not mutual. On a clean
// closed manifold this is 0; non-manifold feature edges contribute a
// count the reseed must never increase.
int asymmetric_opposites(const v::Triangles& tri,
                         std::vector<v::marching_cubes::Interface> itf) {
    auto he = v::mesh::triangles_to_hedges(tri, &itf);
    int bad = 0;
    for (Eigen::Index i = 0; i < he.rows(); ++i) {
        const v::u32 o = he(i, 3);
        if (o == v::kInvalidU32 || he(o, 3) != static_cast<v::u32>(i)) ++bad;
    }
    return bad;
}

v::Settings reseed_settings() {
    v::Settings s;
    s.Lmax                       = 40.0;
    s.reseed_eps                 = 0.5;
    s.reseed_target_len          = 0.0;    // → Lmax
    s.min_corner_angle_boundary  = 20.0;
    s.min_corner_angle_internal  = 30.0;
    return s;
}

}  // namespace

// ===========================================================================
// Stage A: bbox frame.
// ===========================================================================
TEST(Reseed, CubeFrameCoarsensAndPreservesGeometry) {
    auto F = make_cube(5, /*two_mat=*/false);
    const Eigen::Index V0 = F.xyz.rows();
    const Eigen::Index T0 = F.tri.rows();
    const double A0 = total_area(F.tri, F.xyz);

    const auto s  = reseed_settings();
    const auto st = v::remesh::reseed_feature_chains(s, F.tri, F.xyz, F.mask,
                                                     F.interfaces);

    // All 12 frame chains found; at least one node removed per chain.
    EXPECT_EQ(st.frame_chains, 12u);
    EXPECT_GE(st.verts_removed, 12u);
    EXPECT_EQ(F.xyz.rows(), V0 - static_cast<Eigen::Index>(st.verts_removed));
    // Frame edges are manifold: each collapse removes exactly 2 triangles.
    EXPECT_EQ(st.tris_removed, 2 * st.verts_removed);
    EXPECT_EQ(F.tri.rows(), T0 - static_cast<Eigen::Index>(st.tris_removed));

    // Planar faces + straight frame ⇒ total area must be exact.
    EXPECT_NEAR(total_area(F.tri, F.xyz), A0, 1e-9);

    // Watertight closed manifold: every opposite valid and mutual.
    EXPECT_EQ(asymmetric_opposites(F.tri, F.interfaces), 0);

    // Euler characteristic of a sphere-topology surface: V - E + F = 2
    // with E = 3F/2 on a closed triangle mesh.
    const Eigen::Index Vn = F.xyz.rows(), Fn = F.tri.rows();
    EXPECT_EQ(Fn % 2, 0);
    EXPECT_EQ(Vn - (3 * Fn) / 2 + Fn, 2);

    // The 8 bbox corners survive; every surviving frame vertex still
    // lies exactly on a bbox edge (two coordinates at 0 or n).
    int corners = 0, off_edge = 0;
    for (Eigen::Index i = 0; i < F.xyz.rows(); ++i) {
        if (F.mask.masks[5][static_cast<std::size_t>(i)]) ++corners;
        if (!F.mask.masks[4][static_cast<std::size_t>(i)]) continue;
        int pinned = 0;
        for (int k = 0; k < 3; ++k)
            if (F.xyz(i, k) == 0.0 || F.xyz(i, k) == double(F.n)) ++pinned;
        if (pinned < 2) ++off_edge;
    }
    EXPECT_EQ(corners, 8);
    EXPECT_EQ(off_edge, 0);

    // Mask rows compacted consistently.
    for (int m = 0; m < 8; ++m)
        EXPECT_EQ(F.mask.masks[m].size(),
                  static_cast<std::size_t>(F.xyz.rows()));

    // Interface (first, count) ranges must tile the new triangle list.
    std::size_t covered = 0;
    for (const auto& it : F.interfaces) {
        EXPECT_EQ(it.first, covered);
        covered += it.count;
    }
    EXPECT_EQ(covered, static_cast<std::size_t>(F.tri.rows()));
}

// ---------------------------------------------------------------------------
TEST(Reseed, TargetSpacingCapIsRespected) {
    auto F = make_cube(5, /*two_mat=*/false);
    auto s = reseed_settings();

    // Chain node spacing on a 5-voxel frame edge is 0.5,1,1,1,1,0.5 —
    // every possible merged chord is ≥ 1.5, so a 1.2 target forbids
    // every collapse and the mesh must come back untouched.
    s.reseed_target_len = 1.2;
    const Eigen::Index V0 = F.xyz.rows();
    const auto st = v::remesh::reseed_feature_chains(s, F.tri, F.xyz, F.mask,
                                                     F.interfaces);
    EXPECT_EQ(st.verts_removed, 0u);
    EXPECT_EQ(F.xyz.rows(), V0);

    // With a 2.0 target, chords up to 2 are allowed but not longer:
    // verify on the reseeded mesh that no frame–frame boundary edge
    // exceeds the target.
    s.reseed_target_len = 2.0;
    const auto st2 = v::remesh::reseed_feature_chains(s, F.tri, F.xyz, F.mask,
                                                      F.interfaces);
    EXPECT_GT(st2.verts_removed, 0u);
    auto bedges = v::brep::get_boundary_edges(F.tri, F.mask);
    for (Eigen::Index e = 0; e < bedges.rows(); ++e) {
        const v::u32 a = bedges(e, 0), b = bedges(e, 1);
        if (!F.mask.masks[4][a] || !F.mask.masks[4][b]) continue;
        EXPECT_LE((F.xyz.row(a) - F.xyz.row(b)).norm(), 2.0 + 1e-9);
    }
}

// ---------------------------------------------------------------------------
TEST(Reseed, QualityGuardControlsCoarsening) {
    // A permissive corner-angle threshold must allow at least as much
    // coarsening as the strict default — the guard is what limits the
    // spacing next to the voxel-fine face interior.
    auto Fa = make_cube(5, false);
    auto Fb = make_cube(5, false);
    auto s_strict = reseed_settings();          // 20°
    auto s_loose  = reseed_settings();
    s_loose.min_corner_angle_boundary = 1.0;

    const auto st_strict = v::remesh::reseed_feature_chains(
        s_strict, Fa.tri, Fa.xyz, Fa.mask, Fa.interfaces);
    const auto st_loose = v::remesh::reseed_feature_chains(
        s_loose, Fb.tri, Fb.xyz, Fb.mask, Fb.interfaces);

    EXPECT_GT(st_strict.rej_quality, 0u);
    EXPECT_GT(st_loose.verts_removed, st_strict.verts_removed);
    // Even fully coarsened, the mesh stays valid.
    EXPECT_EQ(asymmetric_opposites(Fb.tri, Fb.interfaces), 0);
    EXPECT_NEAR(total_area(Fb.tri, Fb.xyz), 150.0, 1e-9);
}

// ===========================================================================
// Stage A2 scope switch: frame-only must leave traces alone.
// ===========================================================================
TEST(Reseed, ScopeSwitchPreservesTracesAndJunctions) {
    auto F = make_cube(5, /*two_mat=*/true);
    const double A0 = total_area(F.tri, F.xyz);
    const int asym0 = asymmetric_opposites(F.tri, F.interfaces);

    // Snapshot: internal-interface triangle count and the coordinates
    // of every fixed (masks[2]) node — junctions must survive exactly.
    std::size_t itf_internal_tris0 = 0;
    for (const auto& it : F.interfaces)
        if (it.mat1 == 1 && it.mat2 == 2) itf_internal_tris0 += it.count;
    std::set<std::array<double, 3>> fixed0;
    for (Eigen::Index i = 0; i < F.xyz.rows(); ++i)
        if (F.mask.masks[2][static_cast<std::size_t>(i)])
            fixed0.insert({F.xyz(i, 0), F.xyz(i, 1), F.xyz(i, 2)});

    auto s = reseed_settings();
    s.do_reseed_triple_lines = false;           // Stage A behaviour
    const auto st = v::remesh::reseed_feature_chains(s, F.tri, F.xyz, F.mask,
                                                     F.interfaces);
    EXPECT_GT(st.verts_removed, 0u);
    EXPECT_EQ(st.curved_chains, 0u);
    EXPECT_EQ(st.loop_chains, 0u);

    // Frame chains split at the 4 trace junctions: 8 whole lines plus
    // 4 lines split in two → 16 eligible chains.
    EXPECT_EQ(st.frame_chains, 16u);

    // Geometry preserved (planar faces + planar internal interface).
    EXPECT_NEAR(total_area(F.tri, F.xyz), A0, 1e-9);

    // The internal interface is untouched in frame-only scope.
    std::size_t itf_internal_tris = 0;
    for (const auto& it : F.interfaces)
        if (it.mat1 == 1 && it.mat2 == 2) itf_internal_tris += it.count;
    EXPECT_EQ(itf_internal_tris, itf_internal_tris0);

    // Every fixed node survives at its exact position.
    std::set<std::array<double, 3>> fixed1;
    for (Eigen::Index i = 0; i < F.xyz.rows(); ++i)
        if (F.mask.masks[2][static_cast<std::size_t>(i)])
            fixed1.insert({F.xyz(i, 0), F.xyz(i, 1), F.xyz(i, 2)});
    EXPECT_EQ(fixed1, fixed0);

    // Non-manifoldness must not grow.
    EXPECT_EQ(asymmetric_opposites(F.tri, F.interfaces), asym0);
}

// ===========================================================================
// Stage A2: traces (multiplicity-3 chain edges) reseed.
// ===========================================================================
TEST(Reseed, TracesReseedWithTripleLineScope) {
    auto F = make_cube(5, /*two_mat=*/true);
    const double A0 = total_area(F.tri, F.xyz);
    const int asym0 = asymmetric_opposites(F.tri, F.interfaces);

    // Trace interior vertices: on the interface plane x = 2, on the
    // bbox surface, but not on the frame.
    auto count_trace_verts = [](const Fixture& G) {
        int c = 0;
        for (Eigen::Index i = 0; i < G.xyz.rows(); ++i)
            if (G.xyz(i, 0) == 2.0 &&
                G.mask.masks[3][static_cast<std::size_t>(i)] &&
                !G.mask.masks[4][static_cast<std::size_t>(i)] &&
                (G.mask.masks[1][static_cast<std::size_t>(i)] ||
                 G.mask.masks[2][static_cast<std::size_t>(i)]))
                ++c;
        return c;
    };
    const int trace0 = count_trace_verts(F);
    ASSERT_GT(trace0, 0);

    std::set<std::array<double, 3>> fixed0;
    for (Eigen::Index i = 0; i < F.xyz.rows(); ++i)
        if (F.mask.masks[2][static_cast<std::size_t>(i)])
            fixed0.insert({F.xyz(i, 0), F.xyz(i, 1), F.xyz(i, 2)});

    // Stage B off: on this 5-voxel fixture the traces sit 2 voxels
    // from the frame, so β·lfs (≈1.4) would correctly forbid any
    // coarsening — ChainLfs tests cover that; here we exercise the
    // A2 collapse machinery itself.
    auto s = reseed_settings();                 // triple-line scope on
    s.do_reseed_lfs = false;
    const auto st = v::remesh::reseed_feature_chains(s, F.tri, F.xyz, F.mask,
                                                     F.interfaces);

    EXPECT_EQ(st.frame_chains, 16u);
    EXPECT_EQ(st.curved_chains, 4u);            // 4 straight face traces
    EXPECT_LT(count_trace_verts(F), trace0);    // traces coarsened

    // Straight traces on planar geometry: area is exact.
    EXPECT_NEAR(total_area(F.tri, F.xyz), A0, 1e-9);

    // Surviving trace vertices stay exactly on the x = 2 trace lines.
    for (Eigen::Index i = 0; i < F.xyz.rows(); ++i) {
        if (F.xyz(i, 0) != 2.0) continue;
        if (!F.mask.masks[3][static_cast<std::size_t>(i)]) continue;
        // On a face, so one of y/z is pinned at 0 or 5 — and the trace
        // itself runs along the pinned coordinate's face.
        bool on_face = false;
        for (int k = 1; k < 3; ++k)
            on_face = on_face || F.xyz(i, k) == 0.0 || F.xyz(i, k) == 5.0;
        EXPECT_TRUE(on_face);
    }

    // Junction fixed nodes survive exactly; non-manifoldness shrinks
    // with the trace edge count (each trace collapse removes one
    // unpaired half-edge).
    std::set<std::array<double, 3>> fixed1;
    for (Eigen::Index i = 0; i < F.xyz.rows(); ++i)
        if (F.mask.masks[2][static_cast<std::size_t>(i)])
            fixed1.insert({F.xyz(i, 0), F.xyz(i, 1), F.xyz(i, 2)});
    EXPECT_EQ(fixed1, fixed0);
    EXPECT_LT(asymmetric_opposites(F.tri, F.interfaces), asym0);
}

// ===========================================================================
// Stage A2: internal triple line (three materials).
// ===========================================================================
TEST(Reseed, InternalTripleLineReseeds) {
    // Volume axis mapping: image index 0 → mesh Y, index 1 → mesh Z
    // (flipped), index 2 → mesh X. Splitting on image x and image z
    // gives interface planes mesh-x = 3 and mesh-y = 3, i.e. one
    // straight internal triple line along (3, 3, z) in mesh coords.
    const int n = 6;
    auto F = make_volume(n, "cube3", [&](int x, int, int z) {
        if (x < 3) return 1;
        return z < 3 ? 2 : 3;
    });
    const double A0 = total_area(F.tri, F.xyz);

    auto count_internal_chain = [](const Fixture& G) {
        int c = 0;
        for (Eigen::Index i = 0; i < G.xyz.rows(); ++i)
            if (G.xyz(i, 0) == 3.0 && G.xyz(i, 1) == 3.0 &&
                !G.mask.masks[3][static_cast<std::size_t>(i)])
                ++c;
        return c;
    };
    const int chain0 = count_internal_chain(F);
    ASSERT_GT(chain0, 0);

    auto s = reseed_settings();
    // Permissive corner thresholds: this test is about the topology of
    // multiplicity-3 collapse, not the quality limiter.
    s.min_corner_angle_boundary = 5.0;
    s.min_corner_angle_internal = 5.0;
    const auto st = v::remesh::reseed_feature_chains(s, F.tri, F.xyz, F.mask,
                                                     F.interfaces);

    EXPECT_GE(st.curved_chains, 1u);
    EXPECT_LT(count_internal_chain(F), chain0);

    // Every collapse on the triple line removes 3 triangles; frame
    // collapses remove 2 — so tris_removed must be < 3 * removed and
    // > 2 * removed when both kinds occurred.
    EXPECT_GE(st.tris_removed, 2 * st.verts_removed);
    EXPECT_LE(st.tris_removed, 3 * st.verts_removed);

    // All geometry is planar (box faces + three flat sheets): exact.
    EXPECT_NEAR(total_area(F.tri, F.xyz), A0, 1e-9);

    // Surviving internal chain vertices are still exactly on the line,
    // and its two face junctions survive.
    int endpoints = 0;
    for (Eigen::Index i = 0; i < F.xyz.rows(); ++i) {
        if (F.xyz(i, 0) == 3.0 && F.xyz(i, 1) == 3.0) {
            if (F.xyz(i, 2) == 0.0 || F.xyz(i, 2) == double(n)) ++endpoints;
        }
    }
    EXPECT_EQ(endpoints, 2);
}

// ===========================================================================
// Stage A2: curved trace loops (cylindrical inclusion) — chordal
// accuracy and closed-loop handling.
// ===========================================================================
TEST(Reseed, CylinderTraceLoopsAccuracy) {
    const int n = 12;
    const double r2 = 16.0;                     // radius 4 around (6, 6)
    auto F = make_volume(n, "cyl", [&](int x, int y, int) {
        const double dx = x + 0.5 - 6.0, dy = y + 0.5 - 6.0;
        return dx * dx + dy * dy < r2 ? 2 : 1;
    });

    // The cylinder axis runs along image index 0, which is mesh Y: the
    // two trace circles live on the mesh faces y = 0 and y = n.
    // Trace vertices: on those faces, feature nodes off the frame.
    auto trace_verts = [&](const Fixture& G) {
        std::vector<Eigen::Vector3d> pts;
        for (Eigen::Index i = 0; i < G.xyz.rows(); ++i) {
            if (G.xyz(i, 1) != 0.0 && G.xyz(i, 1) != double(n)) continue;
            if (G.mask.masks[4][static_cast<std::size_t>(i)]) continue;
            if (!G.mask.masks[1][static_cast<std::size_t>(i)] &&
                !G.mask.masks[2][static_cast<std::size_t>(i)]) continue;
            pts.push_back(G.xyz.row(i).transpose());
        }
        return pts;
    };
    const auto pts0 = trace_verts(F);
    ASSERT_GT(pts0.size(), 20u);

    auto s = reseed_settings();
    s.min_corner_angle_boundary = 5.0;
    s.min_corner_angle_internal = 5.0;
    // Stage B off: the trace circles pass within 2 voxels of the face
    // frame, so the lfs cap would (correctly) pin spacing near the
    // current voxel scale and leave nothing to measure here.
    s.do_reseed_lfs = false;
    const auto st = v::remesh::reseed_feature_chains(s, F.tri, F.xyz, F.mask,
                                                     F.interfaces);

    // The two face circles are closed loops with no junction anywhere.
    EXPECT_EQ(st.loop_chains, 2u);
    EXPECT_GT(st.verts_removed, 0u);
    const auto pts1 = trace_verts(F);
    EXPECT_LT(pts1.size(), pts0.size());
    EXPECT_GE(pts1.size(), 8u);                 // ≥ 4 survivors per loop

    // One-sided Hausdorff (original → reseeded): every original trace
    // vertex must lie within reseed_eps of the reseeded trace polyline.
    // This is the geometric-accuracy contract of the deviation guard.
    auto bedges = v::brep::get_boundary_edges(F.tri, F.mask);
    std::vector<std::array<Eigen::Vector3d, 2>> segs;
    for (Eigen::Index e = 0; e < bedges.rows(); ++e) {
        const v::u32 a = bedges(e, 0), b = bedges(e, 1);
        auto is_trace = [&](v::u32 q) {
            return (F.xyz(q, 1) == 0.0 || F.xyz(q, 1) == double(n)) &&
                   !F.mask.masks[4][q];
        };
        if (!is_trace(a) || !is_trace(b)) continue;
        if (F.xyz(a, 1) != F.xyz(b, 1)) continue;
        segs.push_back({F.xyz.row(a).transpose(), F.xyz.row(b).transpose()});
    }
    ASSERT_GT(segs.size(), 0u);
    auto seg_dist = [](const Eigen::Vector3d& p, const Eigen::Vector3d& a,
                       const Eigen::Vector3d& b) {
        const Eigen::Vector3d ab = b - a;
        const double L2 = ab.squaredNorm();
        const double t = L2 < 1e-24
                             ? 0.0
                             : std::clamp((p - a).dot(ab) / L2, 0.0, 1.0);
        return (p - (a + t * ab)).norm();
    };
    for (const auto& p : pts0) {
        double d = std::numeric_limits<double>::infinity();
        for (const auto& sg : segs) d = std::min(d, seg_dist(p, sg[0], sg[1]));
        EXPECT_LE(d, s.reseed_eps + 1e-9);
    }

    // The bbox faces stay exactly covered: retargeting is in-plane and
    // the flip guard forbids folds, so per-face area is preserved even
    // though the trace between the two face regions moved.
    EXPECT_NEAR(plane_area(F.tri, F.xyz, 1, 0.0),       144.0, 1e-9);
    EXPECT_NEAR(plane_area(F.tri, F.xyz, 1, double(n)), 144.0, 1e-9);
}

// ===========================================================================
// AABB tree + segment predicates (reseed_detail).
// ===========================================================================
TEST(ReseedDetail, SegmentTriangleIntersection) {
    using v::remesh::reseed_detail::segment_hits_triangle;
    const Eigen::Vector3d p0(0, 0, 0), p1(4, 0, 0), p2(0, 4, 0);

    // Straight stab through the interior.
    EXPECT_TRUE(segment_hits_triangle({1, 1, -1}, {1, 1, 1}, p0, p1, p2));
    // Stops short of the plane.
    EXPECT_FALSE(segment_hits_triangle({1, 1, -2}, {1, 1, -0.5}, p0, p1, p2));
    // Crosses the plane outside the triangle.
    EXPECT_FALSE(segment_hits_triangle({3.5, 3.5, -1}, {3.5, 3.5, 1}, p0, p1, p2));
    // Coplanar segment → parallel, by design not detected here.
    EXPECT_FALSE(segment_hits_triangle({-1, 1, 0}, {5, 1, 0}, p0, p1, p2));
    // Endpoint exactly on the triangle: strict interior test says no.
    EXPECT_FALSE(segment_hits_triangle({1, 1, 0}, {1, 1, 2}, p0, p1, p2));
}

TEST(ReseedDetail, AabbTreeQueryAndGrowOnlyRefit) {
    using v::remesh::reseed_detail::AabbTree;
    // `for_segment` is a broad phase with a leaf bucket of 4: it must
    // visit every item whose region the segment touches (no misses) and
    // prune far-away subtrees (no full scans). 8 unit boxes spaced
    // along x guarantee at least one split.
    std::vector<std::array<Eigen::Vector3d, 2>> boxes;
    for (int i = 0; i < 8; ++i)
        boxes.push_back({Eigen::Vector3d(10.0 * i, 0, 0),
                         Eigen::Vector3d(10.0 * i + 1, 1, 1)});
    AabbTree tree;
    tree.build(boxes);

    auto hits = [&](const Eigen::Vector3d& a, const Eigen::Vector3d& b) {
        std::set<int> out;
        tree.for_segment(a, b, [&](int item) { out.insert(item); });
        return out;
    };

    // Vertical segment through box 0: must include 0, must have pruned
    // the far half of the tree.
    auto h0 = hits({0.5, 0.5, -1}, {0.5, 0.5, 2});
    EXPECT_TRUE(h0.count(0));
    EXPECT_FALSE(h0.count(7));
    // Segment through box 7: symmetric.
    auto h7 = hits({70.5, 0.5, -1}, {70.5, 0.5, 2});
    EXPECT_TRUE(h7.count(7));
    EXPECT_FALSE(h7.count(0));
    // Long segment along all boxes: everything visited.
    EXPECT_EQ(hits({-1, 0.5, 0.5}, {72, 0.5, 0.5}).size(), 8u);

    // Grow item 0 out to x = 75: a far query must now visit it — the
    // refit has to propagate through every ancestor.
    tree.grow_item(0, {0, 0, 0}, {75, 1, 1});
    EXPECT_TRUE(hits({70.5, 0.5, -1}, {70.5, 0.5, 2}).count(0));
}

// ===========================================================================
// Stage C: graded sizing field (limit_sizing_gradient).
// ===========================================================================

// Exact semantics against an independent reference: the graded field
// must equal  min_u( L0[u] + (g-1) * d(u,v) )  with d the shortest-path
// distance in the mesh edge graph. The reference is a Bellman-Ford
// style relaxation over the raw triangle edge list — a different
// algorithm and a different edge extraction than the implementation.
TEST(SizingGrading, MatchesBruteForceReference) {
    auto F = make_cube(4, /*two_mat=*/true);
    auto he = v::mesh::triangles_to_hedges(F.tri, &F.interfaces);
    const Eigen::Index N = F.xyz.rows();
    const double g = 1.3, slope = g - 1.0;

    Eigen::VectorXd L0 = Eigen::VectorXd::Constant(N, 10.0);
    L0[0] = 1.0;                     // one fine spike in a coarse field
    Eigen::VectorXd L = L0;
    const auto lowered = v::remesh::limit_sizing_gradient(he, F.xyz, L, g);
    EXPECT_GT(lowered, 0u);
    EXPECT_DOUBLE_EQ(L[0], 1.0);     // the minimum is never touched

    // Reference: relax over triangle edges until fixpoint.
    Eigen::VectorXd R = L0;
    bool changed = true;
    while (changed) {
        changed = false;
        for (Eigen::Index t = 0; t < F.tri.rows(); ++t) {
            for (int k = 0; k < 3; ++k) {
                const v::u32 a = F.tri(t, k), b = F.tri(t, (k + 1) % 3);
                const double w =
                    slope * (F.xyz.row(a) - F.xyz.row(b)).norm();
                if (R[a] + w < R[b] - 1e-15) { R[b] = R[a] + w; changed = true; }
                if (R[b] + w < R[a] - 1e-15) { R[a] = R[b] + w; changed = true; }
            }
        }
    }
    for (Eigen::Index i = 0; i < N; ++i) EXPECT_NEAR(L[i], R[i], 1e-9);

    // Idempotent: a field that already satisfies the bound is unchanged.
    Eigen::VectorXd L2 = L;
    EXPECT_EQ(v::remesh::limit_sizing_gradient(he, F.xyz, L2, g), 0u);
    for (Eigen::Index i = 0; i < N; ++i) EXPECT_DOUBLE_EQ(L2[i], L[i]);

    // g <= 1 is a no-op.
    Eigen::VectorXd L3 = L0;
    EXPECT_EQ(v::remesh::limit_sizing_gradient(he, F.xyz, L3, 1.0), 0u);
    for (Eigen::Index i = 0; i < N; ++i) EXPECT_DOUBLE_EQ(L3[i], L0[i]);
}

// End-to-end over the real pre-remesh path: reseed, rebuild derived
// state, compute the sizing field, grade it — then every mesh edge must
// satisfy the Lipschitz bound, grading must only lower values, and the
// field must stay within [Lmin, Lmax].
TEST(SizingGrading, PipelineFieldSatisfiesLipschitzBound) {
    auto F = make_cube(6, /*two_mat=*/true);
    const auto s = reseed_settings();
    v::remesh::reseed_feature_chains(s, F.tri, F.xyz, F.mask, F.interfaces);

    auto he    = v::mesh::triangles_to_hedges(F.tri, &F.interfaces);
    auto masks = v::mesh::get_not_fixed(he, F.xyz, F.mask);
    std::vector<std::uint8_t> is_brep(static_cast<std::size_t>(F.xyz.rows()));
    for (Eigen::Index i = 0; i < F.xyz.rows(); ++i)
        is_brep[static_cast<std::size_t>(i)] = !F.mask.masks[0][i];
    const auto brep_sz = v::remesh::calc_brep_sizing(F.xyz, is_brep);

    double Lmin = std::numeric_limits<double>::infinity();
    for (Eigen::Index i = 0; i < brep_sz.size(); ++i)
        if (brep_sz[i] > 0 && brep_sz[i] < Lmin) Lmin = brep_sz[i];
    ASSERT_TRUE(std::isfinite(Lmin));

    auto L = v::remesh::calc_sizing_field(he, F.xyz, brep_sz,
                                          F.mask.masks[0], F.interfaces,
                                          Lmin, s.Lmax);
    const Eigen::VectorXd pre = L;
    const double g = 1.3;
    v::remesh::limit_sizing_gradient(he, F.xyz, L, g);

    for (Eigen::Index k = 0; k < he.rows(); ++k) {
        const v::u32 a = he(he(k, 2), 0), b = he(k, 0);
        const double w = (F.xyz.row(a) - F.xyz.row(b)).norm();
        EXPECT_LE(L[b], L[a] + (g - 1.0) * w + 1e-9);
    }
    for (Eigen::Index i = 0; i < L.size(); ++i) {
        EXPECT_LE(L[i], pre[i] + 1e-15);         // never raised
        EXPECT_GE(L[i], Lmin - 1e-9);            // never below the floor
        EXPECT_LE(L[i], s.Lmax + 1e-9);
    }
}

// ===========================================================================
// Stage B: composite local-feature-size targets (compute_chain_lfs).
// ===========================================================================
namespace {

// Union-find over the chain graph → component id per chain vertex.
std::vector<int> chain_components(const v::brep::BEdges& bedges,
                                  Eigen::Index nverts) {
    std::vector<int> parent(static_cast<std::size_t>(nverts));
    for (std::size_t i = 0; i < parent.size(); ++i) parent[i] = int(i);
    std::function<int(int)> find = [&](int x) {
        while (parent[x] != x) { parent[x] = parent[parent[x]]; x = parent[x]; }
        return x;
    };
    for (Eigen::Index e = 0; e < bedges.rows(); ++e)
        parent[find(int(bedges(e, 0)))] = find(int(bedges(e, 1)));
    std::vector<int> comp(parent.size(), -1);
    for (std::size_t i = 0; i < parent.size(); ++i) comp[i] = find(int(i));
    return comp;
}

}  // namespace

// Two cylindrical inclusions with a 2-voxel gap: their trace circles
// are separate chain components, so each caps the other's spacing.
// The contract: h(p) ≤ max(h_min, β · distance-to-other-component),
// h never exceeds the target, and h obeys the along-chain grading.
TEST(ChainLfs, TwoCylinderGapCapsSpacing) {
    const int n = 16;
    auto F = make_volume(n, "lfs2cyl", [&](int x, int y, int) {
        const double d1 = std::hypot(x + 0.5 - 5.0, y + 0.5 - 8.0);
        const double d2 = std::hypot(x + 0.5 - 11.0, y + 0.5 - 8.0);
        return (d1 < 2.0 || d2 < 2.0) ? 2 : 1;
    });
    auto bedges = v::brep::get_boundary_edges(F.tri, F.mask);
    ASSERT_GT(bedges.rows(), 0);

    const double target = 40.0, beta = 0.7, grading = 1.3, h_min = 1.0;
    const auto L = v::remesh::reseed_detail::compute_chain_lfs(
        F.tri, F.xyz, bedges, target, beta, grading, h_min);

    const auto comp = chain_components(bedges, F.xyz.rows());
    std::vector<v::u32> cverts;
    for (Eigen::Index i = 0; i < F.xyz.rows(); ++i)
        if (comp[static_cast<std::size_t>(i)] >= 0) {
            bool on_chain = false;
            for (Eigen::Index e = 0; e < bedges.rows() && !on_chain; ++e)
                on_chain = bedges(e, 0) == v::u32(i) || bedges(e, 1) == v::u32(i);
            if (on_chain) cverts.push_back(v::u32(i));
        }
    ASSERT_GT(cverts.size(), 40u);

    double min_trace_h = target;
    for (const v::u32 p : cverts) {
        // Distance to the nearest chain vertex of a DIFFERENT component
        // — always a valid lfs candidate (geodesically unreachable).
        double d_other = std::numeric_limits<double>::infinity();
        for (const v::u32 q : cverts) {
            if (comp[q] == comp[p]) continue;
            d_other = std::min(
                d_other, (F.xyz.row(p) - F.xyz.row(q)).norm());
        }
        if (std::isfinite(d_other))
            EXPECT_LE(L.h[p], std::max(h_min, beta * d_other) + 1e-9)
                << "vertex " << p;
        EXPECT_LE(L.h[p], target + 1e-9);
        EXPECT_GE(L.h[p], h_min - 1e-9);
        if (!F.mask.masks[4][p]) min_trace_h = std::min(min_trace_h, L.h[p]);
    }
    // The 2-voxel gap between the circles must actually bite: some
    // trace vertex sees the other circle at ~2 voxels.
    EXPECT_LE(min_trace_h, beta * 3.0);

    // Along-chain grading bound.
    for (Eigen::Index e = 0; e < bedges.rows(); ++e) {
        const v::u32 a = bedges(e, 0), b = bedges(e, 1);
        const double len = (F.xyz.row(a) - F.xyz.row(b)).norm();
        EXPECT_LE(std::abs(L.h[a] - L.h[b]),
                  (grading - 1.0) * len + 1e-9);
    }
}

// Collar exclusion (the negative test): a single cylinder's trace must
// NOT be clamped by its own staircase neighbours or its own surface
// skirt — without the geodesic collar, adjacent surface vertices at
// ~0.5 voxel would drive every h to h_min. The only genuine cap here
// is the bbox frame (a different component, ≥ 2 voxels away).
TEST(ChainLfs, OwnSkirtExcludedByCollar) {
    const int n = 12;
    auto F = make_volume(n, "lfs1cyl", [&](int x, int y, int) {
        const double dx = x + 0.5 - 6.0, dy = y + 0.5 - 6.0;
        return dx * dx + dy * dy < 16.0 ? 2 : 1;
    });
    auto bedges = v::brep::get_boundary_edges(F.tri, F.mask);
    const double target = 40.0, beta = 0.7;
    const auto L = v::remesh::reseed_detail::compute_chain_lfs(
        F.tri, F.xyz, bedges, target, beta, 1.3, 1.0);

    double max_trace_h = 0.0;
    for (Eigen::Index e = 0; e < bedges.rows(); ++e)
        for (int k = 0; k < 2; ++k) {
            const v::u32 p = bedges(e, k);
            if (F.mask.masks[4][p]) continue;      // skip the frame
            max_trace_h = std::max(max_trace_h, L.h[p]);
        }
    // Circle radius 4 centred in a 12-face: the frame is 2–3.2 voxels
    // away, so β·lfs reaches ≈ 1.4–2.2. If the collar failed, every h
    // would sit at h_min = 1.0.
    EXPECT_GE(max_trace_h, 1.3);
}

// Full reseed with the lfs cap: chords the reseed CREATES must respect
// the per-vertex h field (vertices never move, so surviving vertices
// are matched to their pre-reseed h by exact coordinates).
TEST(ChainLfs, ReseedRespectsLfsCap) {
    const int n = 16;
    auto make = [&]() {
        return make_volume(n, "lfs2cyl_rs", [&](int x, int y, int) {
            const double d1 = std::hypot(x + 0.5 - 5.0, y + 0.5 - 8.0);
            const double d2 = std::hypot(x + 0.5 - 11.0, y + 0.5 - 8.0);
            return (d1 < 2.0 || d2 < 2.0) ? 2 : 1;
        });
    };

    auto F = make();
    auto bedges0 = v::brep::get_boundary_edges(F.tri, F.mask);
    auto s = reseed_settings();
    const double target = s.Lmax;
    const auto L = v::remesh::reseed_detail::compute_chain_lfs(
        F.tri, F.xyz, bedges0, target, s.reseed_beta, s.reseed_grading);
    double max_edge0 = 0.0;
    std::map<std::array<double, 3>, double> h_at;
    for (Eigen::Index e = 0; e < bedges0.rows(); ++e)
        for (int k = 0; k < 2; ++k) {
            const v::u32 p = bedges0(e, k);
            h_at[{F.xyz(p, 0), F.xyz(p, 1), F.xyz(p, 2)}] = L.h[p];
            max_edge0 = std::max(
                max_edge0,
                (F.xyz.row(bedges0(e, 0)) - F.xyz.row(bedges0(e, 1))).norm());
        }

    s.do_reseed_lfs = true;
    const auto st = v::remesh::reseed_feature_chains(s, F.tri, F.xyz,
                                                     F.mask, F.interfaces);
    EXPECT_GT(st.verts_removed, 0u);
    EXPECT_GT(st.lfs_limited, 0u);      // the gap actually bit

    // Every reseed-created chord (longer than any original chain edge)
    // must obey the h of both surviving endpoints.
    auto bedges1 = v::brep::get_boundary_edges(F.tri, F.mask);
    int created = 0;
    for (Eigen::Index e = 0; e < bedges1.rows(); ++e) {
        const v::u32 a = bedges1(e, 0), b = bedges1(e, 1);
        const double len = (F.xyz.row(a) - F.xyz.row(b)).norm();
        if (len <= max_edge0 + 1e-12) continue;   // could be original
        ++created;
        const auto ha = h_at.find({F.xyz(a, 0), F.xyz(a, 1), F.xyz(a, 2)});
        const auto hb = h_at.find({F.xyz(b, 0), F.xyz(b, 1), F.xyz(b, 2)});
        ASSERT_NE(ha, h_at.end());
        ASSERT_NE(hb, h_at.end());
        EXPECT_LE(len, std::min(ha->second, hb->second) + 1e-9);
    }
    EXPECT_GT(created, 0);

    // Control: with the cap off, at least one longer chord appears
    // (otherwise this fixture wouldn't exercise the cap at all).
    auto G = make();
    auto s2 = reseed_settings();
    s2.do_reseed_lfs = false;
    v::remesh::reseed_feature_chains(s2, G.tri, G.xyz, G.mask, G.interfaces);
    auto bedges2 = v::brep::get_boundary_edges(G.tri, G.mask);
    double max_off = 0.0;
    for (Eigen::Index e = 0; e < bedges2.rows(); ++e)
        max_off = std::max(max_off, (G.xyz.row(bedges2(e, 0)) -
                                     G.xyz.row(bedges2(e, 1))).norm());
    double max_on = 0.0;
    for (Eigen::Index e = 0; e < bedges1.rows(); ++e)
        max_on = std::max(max_on, (F.xyz.row(bedges1(e, 0)) -
                                   F.xyz.row(bedges1(e, 1))).norm());
    EXPECT_LT(max_on, max_off + 1e-9);
}

// ===========================================================================
// Cap fix: near-chain collapse in the remesh loop
// (do_collapse_near_bedges) must keep the mesh structurally sound.
// ===========================================================================
TEST(CapFix, RemeshNearChainsStaysWatertight) {
    auto euler = [](const v::Triangles& tri, Eigen::Index nv) {
        std::set<std::pair<v::u32, v::u32>> edges;
        for (Eigen::Index t = 0; t < tri.rows(); ++t)
            for (int k = 0; k < 3; ++k) {
                v::u32 a = tri(t, k), b = tri(t, (k + 1) % 3);
                if (a > b) std::swap(a, b);
                edges.insert({a, b});
            }
        return static_cast<long>(nv) - static_cast<long>(edges.size()) +
               static_cast<long>(tri.rows());
    };

    struct Run {
        long          chi = 0;
        int           nonman = 0;
        Eigen::Index  nverts = 0;
        double        area[3][2] = {};
    };
    Run runs[2];
    Eigen::Index reseeded_verts = 0;
    long chi_reseeded = 0;
    const int n = 12;

    for (const bool cap_fix : {false, true}) {
        SCOPED_TRACE(cap_fix ? "cap fix ON" : "cap fix OFF (control)");
        const double r2 = 16.0;
        auto F = make_volume(n, "capfix", [&](int x, int y, int) {
            const double dx = x + 0.5 - 6.0, dy = y + 0.5 - 6.0;
            return dx * dx + dy * dy < r2 ? 2 : 1;
        });

        auto s = reseed_settings();
        s.n_remesh_itr               = 3;
        s.do_collapse_near_bedges    = cap_fix;
        s.do_save_remesh_interfaces  = false;
        s.do_save_remesh_grains_stl  = false;
        s.do_save_remesh_grains_inp  = false;
        s.out_path_base = (fs::temp_directory_path() / "vox2tet_test_reseed" /
                           "capfix").string();

        v::remesh::reseed_feature_chains(s, F.tri, F.xyz, F.mask,
                                         F.interfaces);
        reseeded_verts     = F.xyz.rows();
        chi_reseeded       = euler(F.tri, F.xyz.rows());
        const int nonman0  = asymmetric_opposites(F.tri, F.interfaces);

        // Build the remesh state the way the pipeline does.
        v::remesh::RemeshState st;
        st.hedges  = v::mesh::triangles_to_hedges(F.tri, &F.interfaces);
        auto masks = v::mesh::get_not_fixed(st.hedges, F.xyz, F.mask);
        st.not_fixed_h = std::move(masks.not_fixed_h);
        st.not_fixed_v = std::move(masks.not_fixed_v);
        st.xyz     = F.xyz;
        st.normals = v::remesh::calc_initial_vertex_normal(F.xyz, F.tri,
                                                           F.mask.masks[0]);
        std::vector<std::uint8_t> is_brep(
            static_cast<std::size_t>(F.xyz.rows()));
        for (Eigen::Index i = 0; i < F.xyz.rows(); ++i)
            is_brep[static_cast<std::size_t>(i)] = !F.mask.masks[0][i];
        const auto brep_sz = v::remesh::calc_brep_sizing(F.xyz, is_brep);
        double Lmin = std::numeric_limits<double>::infinity();
        for (Eigen::Index i = 0; i < brep_sz.size(); ++i)
            if (brep_sz[i] > 0 && brep_sz[i] < Lmin) Lmin = brep_sz[i];
        st.sizing = v::remesh::calc_sizing_field(st.hedges, F.xyz, brep_sz,
                                                 F.mask.masks[0],
                                                 F.interfaces, Lmin, s.Lmax);

        auto R = v::remesh::remesh(s, st, F.interfaces);
        ASSERT_GT(R.triangles.rows(), 0);

        // Watertight with the same feature-chain structure: chain edges
        // are fixed, so the count of non-manifold (3+ sheet) edges is
        // invariant through the whole loop. This is the hard structural
        // requirement for the near-chain collapse path.
        EXPECT_EQ(asymmetric_opposites(R.triangles, F.interfaces), nonman0);

        Run& r   = runs[cap_fix ? 1 : 0];
        r.chi    = euler(R.triangles, R.xyz.rows());
        r.nonman = asymmetric_opposites(R.triangles, F.interfaces);
        r.nverts = R.xyz.rows();
        for (int axis = 0; axis < 3; ++axis) {
            r.area[axis][0] = plane_area(R.triangles, R.xyz, axis, 0.0);
            r.area[axis][1] = plane_area(R.triangles, R.xyz, axis, double(n));
        }
    }

    // Split / collapse / flip are each Euler-characteristic-preserving
    // (interior split skips non-manifold chain edges — splitting one
    // sheet of a triple line would leave a T-junction), so the whole
    // loop must keep χ at the reseeded value, with and without the
    // near-chain collapse path. The bbox faces must stay covered too.
    EXPECT_EQ(runs[0].chi, chi_reseeded);
    EXPECT_EQ(runs[1].chi, chi_reseeded);
    for (int axis = 0; axis < 3; ++axis)
        for (int side = 0; side < 2; ++side) {
            EXPECT_NEAR(runs[0].area[axis][side], double(n) * n,
                        0.015 * n * n);
            EXPECT_NEAR(runs[1].area[axis][side], double(n) * n,
                        0.015 * n * n);
        }
    // The fix must actually coarsen near the chains: strictly fewer
    // final vertices than both the reseeded mesh and the control run.
    EXPECT_LT(runs[1].nverts, reseeded_verts);
    EXPECT_LT(runs[1].nverts, runs[0].nverts);
}

// ---------------------------------------------------------------------------
TEST(Reseed, SettingsRoundTripIncludesReseedFields) {
    const fs::path tmp =
        fs::temp_directory_path() / "vox2tet_reseed_settings.json";
    fs::remove(tmp);

    v::Settings s1;
    s1.do_reseed_bedges         = false;
    s1.do_reseed_triple_lines   = false;
    s1.do_reseed_graded_sizing  = true;          // non-default
    s1.do_collapse_near_bedges  = false;         // non-default
    s1.do_reseed_lfs            = false;         // non-default (Stage B)
    s1.reseed_eps               = 0.7;
    s1.reseed_target_len        = 3.5;
    s1.reseed_grading           = 1.7;
    s1.reseed_beta              = 0.55;
    s1.dump_json(tmp.string());

    v::Settings s2;
    ASSERT_TRUE(s2.do_reseed_bedges);            // defaults: reseeding on,
    ASSERT_TRUE(s2.do_reseed_triple_lines);      // graded sizing off
    ASSERT_FALSE(s2.do_reseed_graded_sizing);    // (experimental)
    ASSERT_TRUE(s2.do_collapse_near_bedges);     // cap fix on by default
    ASSERT_TRUE(s2.do_reseed_lfs);               // Stage B on by default
    s2.load_json(tmp.string());
    EXPECT_FALSE(s2.do_reseed_bedges);
    EXPECT_FALSE(s2.do_reseed_triple_lines);
    EXPECT_TRUE(s2.do_reseed_graded_sizing);
    EXPECT_FALSE(s2.do_collapse_near_bedges);
    EXPECT_DOUBLE_EQ(s2.reseed_eps, 0.7);
    EXPECT_DOUBLE_EQ(s2.reseed_target_len, 3.5);
    EXPECT_DOUBLE_EQ(s2.reseed_grading, 1.7);
    EXPECT_FALSE(s2.do_reseed_lfs);
    EXPECT_DOUBLE_EQ(s2.reseed_beta, 0.55);
}
