#include "vox2tet/image/critical_conn.hpp"

#include "vox2tet/core/log.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <stdexcept>

namespace vox2tet::image {

namespace {

// Read a voxel value from the dtype-erased volume into u64. Float volumes
// are normalised on read so we don't expect them here.
inline std::uint64_t read_v(const Volume& v, std::size_t z, std::size_t y, std::size_t x) {
    const std::size_t lin = (z * v.shape[1] + y) * v.shape[2] + x;
    switch (v.dtype) {
        case VoxelType::U8:  return reinterpret_cast<const std::uint8_t* >(v.bytes.data())[lin];
        case VoxelType::U16: return reinterpret_cast<const std::uint16_t*>(v.bytes.data())[lin];
        case VoxelType::U32: return reinterpret_cast<const std::uint32_t*>(v.bytes.data())[lin];
        case VoxelType::U64: return reinterpret_cast<const std::uint64_t*>(v.bytes.data())[lin];
        case VoxelType::F32:
        case VoxelType::F64: break;
    }
    return 0;
}

inline void write_v(Volume& v, std::size_t z, std::size_t y, std::size_t x, std::uint64_t value) {
    const std::size_t lin = (z * v.shape[1] + y) * v.shape[2] + x;
    switch (v.dtype) {
        case VoxelType::U8:
            reinterpret_cast<std::uint8_t* >(v.bytes.data())[lin] = static_cast<std::uint8_t >(value); break;
        case VoxelType::U16:
            reinterpret_cast<std::uint16_t*>(v.bytes.data())[lin] = static_cast<std::uint16_t>(value); break;
        case VoxelType::U32:
            reinterpret_cast<std::uint32_t*>(v.bytes.data())[lin] = static_cast<std::uint32_t>(value); break;
        case VoxelType::U64:
            reinterpret_cast<std::uint64_t*>(v.bytes.data())[lin] = value; break;
        case VoxelType::F32:
        case VoxelType::F64: break;
    }
}

}  // namespace

// ---------------------------------------------------------------------------
// findVertexConnectivity
// ---------------------------------------------------------------------------
//
// For each cell (corner-grid position 0..NZ-2 x 0..NY-2 x 0..NX-2):
//   * gather the 8 corner labels v0..v7 in the same fixed order as in the reference:
//
//       v0 = vox[z  , y  , x  ]   v1 = vox[z  , y  , x+1]
//       v2 = vox[z  , y+1, x  ]   v3 = vox[z  , y+1, x+1]
//       v4 = vox[z+1, y  , x  ]   v5 = vox[z+1, y  , x+1]
//       v6 = vox[z+1, y+1, x  ]   v7 = vox[z+1, y+1, x+1]
//
//   * compute 4 "equality patterns" d0..d3, one per diagonal pair:
//       d_i bit (7-j) is 1 iff voxel[j] == voxel[i_diag_anchor[i]],
//     with anchors 0,1,2,3 (one per diagonal).
//
// The pattern values that flag a *pure diagonal* critical connectivity:
//     d0 == 129 = 10000001  (v0 alone equals its anti-pole v7)
//     d1 == 66  = 01000010
//     d2 == 36  = 00100100
//     d3 == 24  = 00011000
VertexCritResult find_vertex_connectivity(const Volume& v) {
    VertexCritResult R;
    const std::size_t nz1 = v.shape[0] - 1;
    const std::size_t ny1 = v.shape[1] - 1;
    const std::size_t nx1 = v.shape[2] - 1;

    constexpr std::array<int, 4> kTarget = {129, 66, 36, 24};

    for (std::size_t z = 0; z < nz1; ++z)
    for (std::size_t y = 0; y < ny1; ++y)
    for (std::size_t x = 0; x < nx1; ++x) {
        const std::uint64_t corner[8] = {
            read_v(v, z,     y,     x    ),
            read_v(v, z,     y,     x + 1),
            read_v(v, z,     y + 1, x    ),
            read_v(v, z,     y + 1, x + 1),
            read_v(v, z + 1, y,     x    ),
            read_v(v, z + 1, y,     x + 1),
            read_v(v, z + 1, y + 1, x    ),
            read_v(v, z + 1, y + 1, x + 1),
        };
        // d[i] = pattern relative to anchor corner i.
        std::array<int, 4> d{};
        for (int anchor = 0; anchor < 4; ++anchor) {
            int p = 0;
            for (int j = 0; j < 8; ++j) {
                if (corner[j] == corner[anchor]) p |= (1 << (7 - j));
            }
            d[anchor] = p;
        }
        const std::size_t cell_lin = (z * ny1 + y) * nx1 + x;
        for (int i = 0; i < 4; ++i) {
            if (d[i] == kTarget[i]) {
                R.by_diagonal[i].push_back(cell_lin);
                ++R.n_conn;
            }
        }
    }
    return R;
}

// ---------------------------------------------------------------------------
// findEdgeConnectivity
// ---------------------------------------------------------------------------
//
// Three axial planes (a constant z, y, or x). In each 2x2 square in that
// plane we look for the configuration where two diagonal corners share a
// label, all four are at most three materials, and no edge corners share.
//
//   D = ((v1==v4) | (v2==v3)) & ~(v1==v2) & ~(v2==v4) & ~(v4==v3) & ~(v3==v1)
//   A = ((v1==v4) & (v2==v3)) & ~(v1==v2)            // 2-material case
EdgeCritResult find_edge_connectivity(const Volume& v) {
    EdgeCritResult R;
    const std::size_t nz = v.shape[0];
    const std::size_t ny = v.shape[1];
    const std::size_t nx = v.shape[2];

    auto check2x2 = [&](std::uint64_t a, std::uint64_t b, std::uint64_t c, std::uint64_t d,
                        bool& D, bool& A) {
        const bool e_ad = (a == d);
        const bool e_bc = (b == c);
        const bool e_ab = (a == b);
        const bool e_bd = (b == d);
        const bool e_dc = (d == c);
        const bool e_ca = (c == a);
        D = (e_ad || e_bc) && !e_ab && !e_bd && !e_dc && !e_ca;
        A = (e_ad &&  e_bc) && !e_ab;
    };

    // Plane with normal axis 0 (constant z):
    //   v1=v(z, y,   x), v2=v(z, y,   x+1), v3=v(z, y+1, x), v4=v(z, y+1, x+1)
    for (std::size_t z = 0; z < nz; ++z)
    for (std::size_t y = 0; y + 1 < ny; ++y)
    for (std::size_t x = 0; x + 1 < nx; ++x) {
        bool D, A;
        check2x2(read_v(v, z, y, x), read_v(v, z, y, x + 1),
                 read_v(v, z, y + 1, x), read_v(v, z, y + 1, x + 1), D, A);
        if (D || A) {
            R.total_by_axis[0].push_back({z, y, x});
            if (A) { R.m2_by_axis[0].push_back({z, y, x}); ++R.m2; }
            else                                          { ++R.m3; }
        }
    }
    // Plane with normal axis 1 (constant y):
    for (std::size_t z = 0; z + 1 < nz; ++z)
    for (std::size_t y = 0; y < ny; ++y)
    for (std::size_t x = 0; x + 1 < nx; ++x) {
        bool D, A;
        check2x2(read_v(v, z,     y, x), read_v(v, z,     y, x + 1),
                 read_v(v, z + 1, y, x), read_v(v, z + 1, y, x + 1), D, A);
        if (D || A) {
            R.total_by_axis[1].push_back({z, y, x});
            if (A) { R.m2_by_axis[1].push_back({z, y, x}); ++R.m2; }
            else                                          { ++R.m3; }
        }
    }
    // Plane with normal axis 2 (constant x):
    for (std::size_t z = 0; z + 1 < nz; ++z)
    for (std::size_t y = 0; y + 1 < ny; ++y)
    for (std::size_t x = 0; x < nx; ++x) {
        bool D, A;
        check2x2(read_v(v, z,     y,     x), read_v(v, z,     y + 1, x),
                 read_v(v, z + 1, y,     x), read_v(v, z + 1, y + 1, x), D, A);
        if (D || A) {
            R.total_by_axis[2].push_back({z, y, x});
            if (A) { R.m2_by_axis[2].push_back({z, y, x}); ++R.m2; }
            else                                          { ++R.m3; }
        }
    }
    return R;
}

// ---------------------------------------------------------------------------
// fixVertexCritical — translation of fixVertexCriticalConnectivity.
//
// For each detected critical cell, try flipping one of the 8 surrounding
// voxels to a candidate colour; accept if it reduces the vertex-critical
// count without making edge-critical worse.
// ---------------------------------------------------------------------------

namespace {

// Re-counts vertex- and edge-critical patterns in a 3x3x3 sub-block
// centred at (z, y, x). To match the legacy f333 inspection, the sub-block
// is "voxels[di-1:di+2]" and we only care about the counts.
struct LocalCounts { std::size_t vert = 0; std::size_t edge2m = 0; };

LocalCounts count_local(const Volume& v, std::size_t z, std::size_t y, std::size_t x) {
    // Build a 3x3x3 sub-volume by copying the bytes; then reuse find_*.
    Volume sub;
    sub.dtype = v.dtype;
    sub.shape = {3, 3, 3};
    const std::size_t isz = v.item_size();
    sub.bytes.assign(27 * isz, 0);
    for (int dz = -1; dz <= 1; ++dz)
    for (int dy = -1; dy <= 1; ++dy)
    for (int dx = -1; dx <= 1; ++dx) {
        const std::size_t zz = z + dz, yy = y + dy, xx = x + dx;
        const std::size_t src = ((zz * v.shape[1]) + yy) * v.shape[2] + xx;
        const std::size_t dst = ((std::size_t(dz + 1) * 3) + (dy + 1)) * 3 + (dx + 1);
        std::memcpy(sub.bytes.data() + dst * isz,
                    v.bytes.data() + src * isz, isz);
    }
    LocalCounts r;
    r.vert   = find_vertex_connectivity(sub).n_conn;
    r.edge2m = find_edge_connectivity(sub).m2;
    return r;
}

}  // namespace

std::pair<std::size_t, std::size_t> fix_vertex_critical(Volume& v) {
    auto initial = find_vertex_connectivity(v);
    const std::size_t before = initial.n_conn;
    if (before == 0) return {0, 0};
    VOX2TET_LOG() << "The image has " << before << " vertex critical connectivities";

    const std::size_t ny1 = v.shape[1] - 1;
    const std::size_t nx1 = v.shape[2] - 1;

    // Index-of-8 helper: idx2[k] returns (dz, dy, dx) for k in 0..7.
    constexpr std::array<std::array<int, 3>, 8> idx2 = {{
        {{0, 0, 0}}, {{0, 0, 1}}, {{0, 1, 0}}, {{0, 1, 1}},
        {{1, 0, 0}}, {{1, 0, 1}}, {{1, 1, 0}}, {{1, 1, 1}},
    }};
    constexpr std::array<int, 4> map_d = {3, 2, 1, 0};

    auto unravel = [&](std::size_t lin, std::size_t& z, std::size_t& y, std::size_t& x) {
        z = lin / (ny1 * nx1);
        const std::size_t r = lin % (ny1 * nx1);
        y = r / nx1;
        x = r % nx1;
    };

    for (int i = 0; i < 4; ++i) {
        const auto& cells = initial.by_diagonal[i];
        const std::size_t off_y = static_cast<std::size_t>(i / 2);
        const std::size_t off_x = static_cast<std::size_t>(i % 2);
        const std::size_t mapd_y = static_cast<std::size_t>(map_d[i] / 2);
        const std::size_t mapd_x = static_cast<std::size_t>(map_d[i] % 2);

        for (std::size_t cl : cells) {
            std::size_t cz, cy, cx;
            unravel(cl, cz, cy, cx);
            const std::uint64_t v_main = read_v(v, cz,     cy + off_y, cx + off_x);
            const std::uint64_t v_anti = read_v(v, cz + 1, cy + off_y, cx + off_x);
            const std::uint64_t v_alt  = read_v(v, cz,     cy + mapd_y, cx + mapd_x);

            for (int k = 0; k < 8; ++k) {
                const std::size_t dz = cz + idx2[k][0];
                const std::size_t dy = cy + idx2[k][1];
                const std::size_t dx = cx + idx2[k][2];
                // 3x3x3 must stay inside the volume.
                if (dz < 1 || dy < 1 || dx < 1) continue;
                if (dz + 1 >= v.shape[0] || dy + 1 >= v.shape[1] || dx + 1 >= v.shape[2]) continue;

                LocalCounts before_local = count_local(v, dz, dy, dx);
                if (before_local.vert == 0) break;

                const std::uint64_t old_center = read_v(v, dz, dy, dx);
                std::uint64_t candidate;
                if (old_center == v_main) {
                    candidate = (k < 4) ? v_anti : v_alt;
                } else {
                    candidate = v_main;
                }
                write_v(v, dz, dy, dx, candidate);
                LocalCounts after_local = count_local(v, dz, dy, dx);
                if (after_local.vert  <  before_local.vert &&
                    after_local.edge2m <= before_local.edge2m) {
                    break;  // accept the change
                }
                // Otherwise revert and try the next k.
                write_v(v, dz, dy, dx, old_center);
                if (k == 7) VOX2TET_PRINT("Waring: fixVertexCriticalConnectivity: Not Fixed");
            }
        }
    }

    const std::size_t after = find_vertex_connectivity(v).n_conn;
    if (after == 0) VOX2TET_PRINT("All vertex critical connectivities are fixed");
    else            { VOX2TET_LOG() << after << " vertex critical connectivities are not fixed"; }
    return {before, after};
}

// ---------------------------------------------------------------------------
// fixEdgeCritical2M — port of fixEdgeCriticalConnectivity2M.
// ---------------------------------------------------------------------------
std::pair<std::size_t, std::size_t> fix_edge_critical_2m(Volume& v) {
    auto initial = find_edge_connectivity(v);
    const std::size_t before = initial.m2;
    if (before == 0) return {0, 0};
    VOX2TET_LOG() << "The image has " << before << " edge critical connectivities with 2 materials";

    // For each plane axis i (0/1/2), idx2[i][j] is the (dz, dy, dx) offset
    // of the j-th of the 4 cell corners. Matches the reference `idx2` table.
    constexpr std::array<std::array<std::array<int, 3>, 4>, 3> idx2 = {{
        {{ {{0, 0, 0}}, {{0, 1, 0}}, {{0, 0, 1}}, {{0, 1, 1}} }},
        {{ {{0, 0, 0}}, {{1, 0, 0}}, {{0, 0, 1}}, {{1, 0, 1}} }},
        {{ {{0, 0, 0}}, {{1, 0, 0}}, {{0, 1, 0}}, {{1, 1, 0}} }},
    }};

    for (int axis = 0; axis < 3; ++axis) {
        for (const auto& cell : initial.m2_by_axis[axis]) {
            const std::size_t z = cell[0], y = cell[1], x = cell[2];
            const std::uint64_t v0 = read_v(v, z, y, x);
            const std::size_t z1 = z + idx2[axis][1][0];
            const std::size_t y1 = y + idx2[axis][1][1];
            const std::size_t x1 = x + idx2[axis][1][2];
            const std::uint64_t v1 = read_v(v, z1, y1, x1);
            // candidate colours for each of the 4 corners (reference: v_colors)
            const std::array<std::uint64_t, 4> v_colors = {v1, v0, v0, v1};

            for (int j = 0; j < 4; ++j) {
                const std::size_t pz = z + idx2[axis][j][0];
                const std::size_t py = y + idx2[axis][j][1];
                const std::size_t px = x + idx2[axis][j][2];
                if (pz < 1 || py < 1 || px < 1) continue;
                if (pz + 1 >= v.shape[0] || py + 1 >= v.shape[1] || px + 1 >= v.shape[2]) continue;

                LocalCounts before_local = count_local(v, pz, py, px);
                if (before_local.edge2m == 0) break;

                const std::uint64_t old_center = read_v(v, pz, py, px);
                write_v(v, pz, py, px, v_colors[j]);
                LocalCounts after_local = count_local(v, pz, py, px);
                if (after_local.vert   <= before_local.vert &&
                    after_local.edge2m <  before_local.edge2m) {
                    break;  // accept
                }
                write_v(v, pz, py, px, old_center);
                if (j == 3) VOX2TET_PRINT("Waring: fixEdgeCriticalConnectivity2M: Not Fixed");
            }
        }
    }

    const std::size_t after = find_edge_connectivity(v).m2;
    if (after == 0) VOX2TET_PRINT("All edge 2M critical connectivities are fixed");
    else            { VOX2TET_LOG() << after << " edge 2M critical connectivities are not fixed"; }
    return {before, after};
}

}  // namespace vox2tet::image
