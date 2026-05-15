#include "vox2tet/marching_cubes/contouring.hpp"

#include "vox2tet/core/log.hpp"
#include "vox2tet/core/paths.hpp"
#include "vox2tet/io/mesh_io.hpp"
#include "vox2tet/io/npy.hpp"
#include "vox2tet/marching_cubes/lut.hpp"
#include "vox2tet/marching_cubes/m3c.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <limits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace vox2tet::marching_cubes {

namespace {

// Dtype-erased voxel read in u64.
std::uint64_t read_vox_u64(const image::Volume& v, std::size_t lin) {
    switch (v.dtype) {
        case image::VoxelType::U8:  return reinterpret_cast<const std::uint8_t* >(v.bytes.data())[lin];
        case image::VoxelType::U16: return reinterpret_cast<const std::uint16_t*>(v.bytes.data())[lin];
        case image::VoxelType::U32: return reinterpret_cast<const std::uint32_t*>(v.bytes.data())[lin];
        case image::VoxelType::U64: return reinterpret_cast<const std::uint64_t*>(v.bytes.data())[lin];
        case image::VoxelType::F32:
        case image::VoxelType::F64: break;
    }
    return 0;
}

}  // namespace

// ---------------------------------------------------------------------------
// code_image — for each cell in the (n+1)^3 corner grid compute the 8-digit
// rank-based code, classify by material count, and return the per-cell
// inputs the downstream meshing kernels need.
//
// Output:
//   cell_codes_2m   : 8-bit pattern for cells with exactly 2 materials
//   is_2m           : bool[n_cells] — cell is 2-material
//   cell_ranks_Mm   : (count_Mmaterial, 8) — rank array for >2-material
//                     cells in cell-iteration order (i.e. the rows align
//                     with `is_Mm[true positions]`)
//   is_Mm           : bool[n_cells] — cell is >2-material
//   img_size1       : (nz+1, ny+1, nx+1) corner grid dims
// ---------------------------------------------------------------------------
struct CodeImageResult {
    std::vector<std::uint8_t>              cell_codes_2m;
    std::vector<std::uint8_t>              is_2m;        // 0/1, length n_cells
    std::vector<std::array<std::uint8_t, 8>> cell_ranks_Mm;
    std::vector<std::uint8_t>              is_Mm;
    std::array<std::size_t, 3>             img_size1;    // (nz+1, ny+1, nx+1)
};

static CodeImageResult code_image(const image::Volume& v) {
    // Extended-volume shape (Z, Y, X) = (nz+2, ny+2, nx+2).
    const std::size_t Nz = v.shape[0];
    const std::size_t Ny = v.shape[1];
    const std::size_t Nx = v.shape[2];
    // Corner grid: one less than extended volume in each axis, but the
    // The original comment uses img_size+1 = (extended-2)+1 = extended-1. So the
    // number of cells per axis is `extended - 1`.
    const std::size_t nz1 = Nz - 1;
    const std::size_t ny1 = Ny - 1;
    const std::size_t nx1 = Nx - 1;
    const std::size_t n_cells = nz1 * ny1 * nx1;

    CodeImageResult R;
    R.img_size1 = {nz1, ny1, nx1};
    R.is_2m.assign(n_cells, 0);
    R.is_Mm.assign(n_cells, 0);

    // For each cell we compute 8 ranks (0..7) describing the equality
    // structure of the 8 corner labels — rank j is the number of corners
    // strictly less than corner j (so identical labels share rank).
    //
    // the reference uses base-10 digit accumulation; we just do it in plain
    // arrays since this is in C++.
    std::cout << "Count of cells: " << n_cells << "\n";

    std::size_t n_interface = 0;
    std::size_t n_2m = 0, n_3m = 0, n_4m = 0, n_5m = 0, n_6m = 0, n_7m = 0, n_8m = 0;

    R.cell_codes_2m.reserve(n_cells / 4);
    R.cell_ranks_Mm.reserve(n_cells / 16);

    for (std::size_t z = 0; z < nz1; ++z)
    for (std::size_t y = 0; y < ny1; ++y)
    for (std::size_t x = 0; x < nx1; ++x) {
        const std::size_t cell_lin = (z * ny1 + y) * nx1 + x;

        // The 8 corners in the SAME index order as in the reference:
        //   c[k] = vox[z + dz, y + dy, x + dx]
        //   k=0..7 with (dz, dy, dx) = unravel_index(k, (2,2,2))
        std::array<std::uint64_t, 8> c;
        for (int k = 0; k < 8; ++k) {
            const std::size_t dz = (k >> 2) & 1;
            const std::size_t dy = (k >> 1) & 1;
            const std::size_t dx = (k >> 0) & 1;
            c[k] = read_vox_u64(v, ((z + dz) * Ny + (y + dy)) * Nx + (x + dx));
        }

        // Rank of each corner among the 8.
        std::array<std::uint8_t, 8> rank;
        for (int i = 0; i < 8; ++i) {
            std::uint8_t r = 0;
            for (int j = 0; j < 8; ++j) if (c[j] < c[i]) ++r;
            rank[i] = r;
        }

        // Count distinct values in `c` = count distinct ranks.
        std::array<bool, 8> seen{};
        for (int i = 0; i < 8; ++i) seen[rank[i]] = true;
        int n_dist = 0;
        for (auto s : seen) if (s) ++n_dist;

        if (n_dist == 1) continue;
        ++n_interface;
        switch (n_dist) {
            case 2: ++n_2m; break;
            case 3: ++n_3m; break;
            case 4: ++n_4m; break;
            case 5: ++n_5m; break;
            case 6: ++n_6m; break;
            case 7: ++n_7m; break;
            case 8: ++n_8m; break;
            default: break;
        }

        if (n_dist == 2) {
            // The 8-bit pattern: rank[k] is 0 or one other value; map to
            // 0/1 then concatenate as bits with rank[0] = MSB.
            std::uint8_t pattern = 0;
            // Find the smaller rank; mark corners equal to it as 0, others as 1.
            const std::uint8_t r_min = *std::min_element(rank.begin(), rank.end());
            for (int k = 0; k < 8; ++k) {
                if (rank[k] != r_min) pattern |= (1u << (7 - k));
            }
            R.cell_codes_2m.push_back(pattern);
            R.is_2m[cell_lin] = 1;
        } else {
            R.cell_ranks_Mm.push_back(rank);
            R.is_Mm[cell_lin] = 1;
        }
    }

    std::cout << "Count of interface cells: " << n_interface << "\n"
              << "Count of 2 materials cells: " << n_2m << "\n"
              << "Count of 3 materials cells: " << n_3m << "\n"
              << "Count of 4 materials cells: " << n_4m << "\n"
              << "Count of 5 materials cells: " << n_5m << "\n"
              << "Count of 6 materials cells: " << n_6m << "\n"
              << "Count of 7 materials cells: " << n_7m << "\n"
              << "Count of 8 materials cells: " << n_8m << "\n";

    return R;
}

// ---------------------------------------------------------------------------
// cells2edges — port of m3c.cells2edges. For each cell, for each of its 6
// faces, look up the 8-element MC edges via the facet_lut2d table.
//
// Input:  cell_ranks   shape (N, 8), values 0..7
// Output: edges        shape (N, 48), values 0..18 with 19 = dummy
// ---------------------------------------------------------------------------
static void cells2edges(const std::vector<std::array<std::uint8_t, 8>>& cell_ranks,
                        std::vector<std::array<std::uint8_t, 48>>& out) {
    const auto& L = load_luts();
    const auto& cf = cell_facet();
    const auto& em = edges_map();

    out.assign(cell_ranks.size(), {});
    for (auto& r : out) r.fill(19);

    for (std::size_t i = 0; i < cell_ranks.size(); ++i) {
        const auto& r = cell_ranks[i];
        auto& cell_edges = out[i];

        for (int f = 0; f < 6; ++f) {
            // 4 face-corner ranks → 4-digit base-8 code (0..4095).
            // Then map to 4^4 code (0..255) by re-ranking within the face
            // — exactly what map8to4_60 + getLUT2DMMaterials60 (resolved
            // through map256to60) does. Our facet_lut2d.bin is the
            // already-resolved 256 × 8 table, indexed by the 4^4 code.
            std::array<std::uint8_t, 4> face_ranks = {
                r[cf[f][0]], r[cf[f][1]], r[cf[f][2]], r[cf[f][3]],
            };
            // Re-rank within the face: the four values are mapped to
            // 0..3 preserving order. Identical values share rank.
            std::array<std::uint8_t, 4> face4;
            for (int k = 0; k < 4; ++k) {
                std::uint8_t rk = 0;
                for (int j = 0; j < 4; ++j) if (face_ranks[j] < face_ranks[k]) ++rk;
                face4[k] = rk;
            }
            // 4-digit base-4 code with face4[0] as MSB.
            const std::uint16_t code4 = static_cast<std::uint16_t>(
                64 * face4[0] + 16 * face4[1] + 4 * face4[2] + face4[3]);
            // Look up 8 face-local MC edge ids; translate through em[f].
            for (int k = 0; k < 8; ++k) {
                const std::uint8_t local = L.facet_lut2d[code4 * 8 + k];
                cell_edges[f * 8 + k] = (local < em[f].size()) ? em[f][local] : 19;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// getTrianglesGlobalNodes — flatten per-cell local triangle node ids
// [0..18] into globally unique ids by adding 20*cell_index. Cells whose
// triangles fully fit 19-sentinel padding contribute nothing. Returns an
// Mx3 array.
//
// `tr_per_cell` is a flat (n_cells * row_width) array; we treat each
// cell as a row and read row_width values from it. Each triple (3 values)
// forms one triangle, where 19 marks "no triangle past this slot".
// ---------------------------------------------------------------------------
static Triangles get_triangles_global_nodes(const std::vector<std::uint8_t>& tr_per_cell,
                                            std::size_t n_cells,
                                            std::size_t row_width,
                                            const std::vector<std::size_t>& cell_indices) {
    // First count valid triangles.
    std::size_t n_tri = 0;
    for (std::size_t i = 0; i < n_cells; ++i) {
        for (std::size_t k = 0; k + 2 < row_width; k += 3) {
            const std::size_t base = i * row_width + k;
            if (tr_per_cell[base] != 19 &&
                tr_per_cell[base + 1] != 19 &&
                tr_per_cell[base + 2] != 19) {
                ++n_tri;
            }
        }
    }
    Triangles out(static_cast<Eigen::Index>(n_tri), 3);
    std::size_t w = 0;
    for (std::size_t i = 0; i < n_cells; ++i) {
        const std::uint32_t cell_glob = static_cast<std::uint32_t>(cell_indices[i]);
        const std::uint32_t base_node = cell_glob * 20u;
        for (std::size_t k = 0; k + 2 < row_width; k += 3) {
            const std::size_t base = i * row_width + k;
            const std::uint8_t a = tr_per_cell[base];
            const std::uint8_t b = tr_per_cell[base + 1];
            const std::uint8_t c = tr_per_cell[base + 2];
            if (a == 19 || b == 19 || c == 19) continue;
            out(static_cast<Eigen::Index>(w), 0) = base_node + a;
            out(static_cast<Eigen::Index>(w), 1) = base_node + b;
            out(static_cast<Eigen::Index>(w), 2) = base_node + c;
            ++w;
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// getTrianglesCoords — convert (cell_index, local_node) pairs in
// `triangles` (encoded as cell_glob * 20 + local_node) to integer XYZ
// coordinates in the doubled-and-shifted lattice the reference uses.
//
// The convention (axes 2,0,1 in the volume → x,y,z in the mesh):
//   * iorder = (1, 2, 0)  →  XYZ[:, iorder[k]] = volume_axis_k
//   * Then flip Y:  XYZ[:, 2] = img_size1[1] - (XYZ[:, 2] + 1)
//   * Multiply by 2, then add per-local-node offset from cell_mid_edge_coords1.
// ---------------------------------------------------------------------------
static Eigen::Matrix<std::int32_t, Eigen::Dynamic, 3, Eigen::RowMajor>
get_triangles_coords_int(const std::vector<std::uint32_t>& unique_nodes,
                         std::array<std::size_t, 3> img_size1) {
    const auto coords1 = cell_mid_edge_coords1();
    const std::int64_t ny1 = static_cast<std::int64_t>(img_size1[1]);
    const std::int64_t nx1 = static_cast<std::int64_t>(img_size1[2]);
    (void)img_size1;  // nz1 unused — derivable from (lin, ny1, nx1)

    Eigen::Matrix<std::int32_t, Eigen::Dynamic, 3, Eigen::RowMajor> out(
        static_cast<Eigen::Index>(unique_nodes.size()), 3);

    for (std::size_t i = 0; i < unique_nodes.size(); ++i) {
        const std::uint32_t g = unique_nodes[i];
        const std::uint32_t cell = g / 20u;
        const std::uint32_t local = g % 20u;

        // Recover (z, y, x) of the cell from its linear index in the
        // (nz1, ny1, nx1) grid.
        const std::int64_t lin = static_cast<std::int64_t>(cell);
        const std::int64_t cz = lin / (ny1 * nx1);
        const std::int64_t cy = (lin % (ny1 * nx1)) / nx1;
        const std::int64_t cx = lin % nx1;

        // Apply iorder = (1, 2, 0): XYZ[:, 1] = z, XYZ[:, 2] = y, XYZ[:, 0] = x
        std::int64_t X = cx;
        std::int64_t Y = cz;
        std::int64_t Z = cy;
        // Y-flip: XYZ[:, 2] = img_size1[1] - (XYZ[:, 2] + 1)
        Z = ny1 - (Z + 1);
        // ×2 + per-local-node offset
        X = X * 2 + coords1[local][0];
        Y = Y * 2 + coords1[local][1];
        Z = Z * 2 + coords1[local][2];

        out(static_cast<Eigen::Index>(i), 0) = static_cast<std::int32_t>(X);
        out(static_cast<Eigen::Index>(i), 1) = static_cast<std::int32_t>(Y);
        out(static_cast<Eigen::Index>(i), 2) = static_cast<std::int32_t>(Z);
    }
    return out;
}

// ---------------------------------------------------------------------------
// dedup_by_coordinate — compute integer xyz for every triangle vertex
// (cell_idx * 20 + local_node → 3D coord), then dedup *by coordinate* and
// remap `tri` to indices into the unique-coordinate list. Mirrors
// the legacy two-phase compress: first compressUintArray on global IDs
// (handled implicitly by computing xyz only for the unique ones), then
// compressUintArrayNx3 on the resulting integer xyz to merge vertices
// that occupy the same physical point but live in different cells.
//
// Returns the unique integer xyz matrix; `tri` is rewritten in place.
// ---------------------------------------------------------------------------
static Eigen::Matrix<std::int32_t, Eigen::Dynamic, 3, Eigen::RowMajor>
dedup_by_coordinate(Triangles& tri,
                    std::array<std::size_t, 3> img_size1) {
    // Step 1: gather the unique (cell*20+local) IDs referenced by `tri`.
    std::vector<std::uint32_t> all;
    all.reserve(static_cast<std::size_t>(tri.size()));
    for (Eigen::Index i = 0; i < tri.rows(); ++i)
        for (int k = 0; k < 3; ++k) all.push_back(tri(i, k));
    std::sort(all.begin(), all.end());
    all.erase(std::unique(all.begin(), all.end()), all.end());

    // Step 2: compute integer xyz for each unique global ID.
    auto xyz_per_unique = get_triangles_coords_int(all, img_size1);

    // Step 3: dedup by xyz row, assigning unique IDs in *lex-sort order*
    // of (x, y, z). This matches the reference compressUintArrayNx3 which
    // discovers unique coords via np.where on a 3D mask — i.e. in
    // (axis0, axis1, axis2) lex order.
    std::vector<std::pair<std::array<std::int32_t, 3>, std::uint32_t>> rows_with_uid;
    rows_with_uid.reserve(static_cast<std::size_t>(xyz_per_unique.rows()));
    for (Eigen::Index i = 0; i < xyz_per_unique.rows(); ++i) {
        rows_with_uid.push_back({
            {xyz_per_unique(i, 0), xyz_per_unique(i, 1), xyz_per_unique(i, 2)},
            static_cast<std::uint32_t>(i)
        });
    }
    std::sort(rows_with_uid.begin(), rows_with_uid.end(),
        [](const auto& a, const auto& b) {
            // Lex compare on (x, y, z) — same axis order as np.where on a
            // (axis0, axis1, axis2) mask.
            if (a.first[0] != b.first[0]) return a.first[0] < b.first[0];
            if (a.first[1] != b.first[1]) return a.first[1] < b.first[1];
            return a.first[2] < b.first[2];
        });

    std::vector<std::array<std::int32_t, 3>> unique_coords;
    std::vector<std::uint32_t> uid_to_new(xyz_per_unique.rows(), 0);
    std::array<std::int32_t, 3> last = {std::numeric_limits<std::int32_t>::min(),
                                         std::numeric_limits<std::int32_t>::min(),
                                         std::numeric_limits<std::int32_t>::min()};
    for (const auto& [coord, uid] : rows_with_uid) {
        if (unique_coords.empty() || coord != last) {
            unique_coords.push_back(coord);
            last = coord;
        }
        uid_to_new[uid] = static_cast<std::uint32_t>(unique_coords.size() - 1);
    }

    // Step 4: build the global-id → unique-coord-index map and remap `tri`.
    std::unordered_map<std::uint32_t, std::uint32_t> gid_to_uid;
    gid_to_uid.reserve(all.size() * 2);
    for (std::size_t i = 0; i < all.size(); ++i) gid_to_uid.emplace(all[i], static_cast<std::uint32_t>(i));
    for (Eigen::Index i = 0; i < tri.rows(); ++i) {
        for (int k = 0; k < 3; ++k) {
            const std::uint32_t uid = gid_to_uid.at(tri(i, k));
            tri(i, k) = uid_to_new[uid];
        }
    }

    Eigen::Matrix<std::int32_t, Eigen::Dynamic, 3, Eigen::RowMajor> out(
        static_cast<Eigen::Index>(unique_coords.size()), 3);
    for (std::size_t i = 0; i < unique_coords.size(); ++i) {
        out(static_cast<Eigen::Index>(i), 0) = unique_coords[i][0];
        out(static_cast<Eigen::Index>(i), 1) = unique_coords[i][1];
        out(static_cast<Eigen::Index>(i), 2) = unique_coords[i][2];
    }
    return out;
}

// ---------------------------------------------------------------------------
// crop_mesh_int — drop triangles that include any vertex outside the input
// image bounds. Operates on the *integer doubled-lattice* xyz (before the
// -1 / *0.5 normalisation applied at the end). Bounds:
//   x in [1, 1+2*nx], y in [1, 1+2*nz], z in [1, 1+2*ny]
// where img_size_orig = (nz, ny, nx) of the unextended image.
// Vertices outside the box are dropped; vertex IDs in `tri` are remapped.
// ---------------------------------------------------------------------------
static void crop_mesh_int(Eigen::Matrix<std::int32_t, Eigen::Dynamic, 3, Eigen::RowMajor>& xyz_i,
                          Triangles& tri,
                          std::array<std::size_t, 3> img_size_orig) {
    // xyz2idx = [2, 0, 1]: x→axis 2 (nx), y→axis 0 (nz), z→axis 1 (ny).
    const std::int32_t lim0 = static_cast<std::int32_t>(1 + 2 * img_size_orig[2]);
    const std::int32_t lim1 = static_cast<std::int32_t>(1 + 2 * img_size_orig[0]);
    const std::int32_t lim2 = static_cast<std::int32_t>(1 + 2 * img_size_orig[1]);

    std::vector<std::uint8_t> is_in(static_cast<std::size_t>(xyz_i.rows()), 0);
    for (Eigen::Index i = 0; i < xyz_i.rows(); ++i) {
        const std::int32_t X = xyz_i(i, 0), Y = xyz_i(i, 1), Z = xyz_i(i, 2);
        if (X >= 1 && Y >= 1 && Z >= 1 && X <= lim0 && Y <= lim1 && Z <= lim2)
            is_in[static_cast<std::size_t>(i)] = 1;
    }

    Triangles new_tri(tri.rows(), 3);
    Eigen::Index ntr = 0;
    for (Eigen::Index i = 0; i < tri.rows(); ++i) {
        if (is_in[tri(i, 0)] && is_in[tri(i, 1)] && is_in[tri(i, 2)]) {
            new_tri.row(ntr++) = tri.row(i);
        }
    }
    new_tri.conservativeResize(ntr, 3);

    std::vector<std::uint32_t> remap(static_cast<std::size_t>(xyz_i.rows()),
                                     static_cast<std::uint32_t>(-1));
    std::size_t n_new = 0;
    for (Eigen::Index i = 0; i < xyz_i.rows(); ++i) {
        if (is_in[i]) remap[static_cast<std::size_t>(i)] =
            static_cast<std::uint32_t>(n_new++);
    }
    Eigen::Matrix<std::int32_t, Eigen::Dynamic, 3, Eigen::RowMajor>
        new_xyz(static_cast<Eigen::Index>(n_new), 3);
    for (Eigen::Index i = 0; i < xyz_i.rows(); ++i) {
        if (is_in[i]) new_xyz.row(remap[static_cast<std::size_t>(i)]) = xyz_i.row(i);
    }
    for (Eigen::Index i = 0; i < new_tri.rows(); ++i)
        for (int k = 0; k < 3; ++k) new_tri(i, k) = remap[new_tri(i, k)];

    xyz_i = std::move(new_xyz);
    tri   = std::move(new_tri);
}

// ---------------------------------------------------------------------------
// createInitialMesh — top-level driver. Mirrors contouring.createInitialMesh.
// ---------------------------------------------------------------------------
InitialMesh create_initial_mesh(const image::Volume& ext_vox,
                                bool do_2x2patterns,
                                const std::string& path_base) {
    VOX2TET_PRINT("Creating Initial Mesh...");

    auto CI = code_image(ext_vox);

    // Original image shape (before extension) = ext_vox.shape - 2 each axis.
    const std::array<std::size_t, 3> img_orig = {
        ext_vox.shape[0] - 2, ext_vox.shape[1] - 2, ext_vox.shape[2] - 2,
    };

    // ----- 2-material cells -------------------------------------------------
    auto lut2 = lut2materials();
    std::vector<std::uint8_t> tri2_rows;        // (n_2m * 18) flat
    std::vector<std::size_t>  cell_idx_2m;
    tri2_rows.reserve(CI.cell_codes_2m.size() * 18);
    cell_idx_2m.reserve(CI.cell_codes_2m.size());
    {
        std::size_t cursor = 0;
        for (std::size_t lin = 0; lin < CI.is_2m.size(); ++lin) {
            if (!CI.is_2m[lin]) continue;
            const std::uint8_t code = CI.cell_codes_2m[cursor++];
            for (int k = 0; k < 18; ++k) tri2_rows.push_back(lut2[code * 18 + k]);
            cell_idx_2m.push_back(lin);
        }
    }
    Triangles tri2 = get_triangles_global_nodes(tri2_rows, cell_idx_2m.size(), 18, cell_idx_2m);

    // ----- M-material cells -------------------------------------------------
    // Classify M-material cells by # centred facets, then mesh each group.
    // We re-use the same machinery: per-cell triangle row (variable width),
    // global node remap, coordinate lookup. For now we implement the two
    // simple groups (0 and 2 centred facets) using the precomputed
    // `lut0` / `lut2` tables loaded from disk; the > 2 centred group and
    // the 2x2 group share the same pattern.
    //
    // Centred-facet detection: in face (4 corners), count distinct ranks.
    // If > 2 AND no two opposite corners share a rank, the facet is
    // centred. Mirrors findCenteredFacet in lut8.py.
    auto find_centered = [&](const std::array<std::uint8_t, 8>& r,
                              std::array<bool, 6>& is_c) {
        const auto& cf = cell_facet();
        for (int f = 0; f < 6; ++f) {
            const std::array<std::uint8_t, 4> face = {
                r[cf[f][0]], r[cf[f][1]], r[cf[f][2]], r[cf[f][3]],
            };
            std::array<bool, 8> seen{};
            for (auto v : face) seen[v] = true;
            int n_d = 0; for (auto s : seen) if (s) ++n_d;
            // Note: opposite corners in the face quad are (0,2) and (1,3).
            is_c[f] = (n_d > 2) && (face[0] != face[2]) && (face[1] != face[3]);
        }
    };
    auto is_2x2_pattern = [&](const std::array<std::uint8_t, 8>& r) {
        // Mirrors find2x2CenteredFacet: checks pairs of opposite faces.
        const auto& cf = cell_facet();
        auto check = [&](int f1, int f2) {
            std::array<std::uint8_t, 4> a = {r[cf[f1][0]], r[cf[f1][1]], r[cf[f1][2]], r[cf[f1][3]]};
            std::array<std::uint8_t, 4> b = {r[cf[f2][0]], r[cf[f2][1]], r[cf[f2][2]], r[cf[f2][3]]};
            const bool I1 = (a[0] == a[2]) && ((a[1] != a[3]) || (a[0] >= a[1]));
            const bool I2 = (b[0] == b[2]) && ((b[1] != b[3]) || (b[0] >= b[1]));
            const bool I3 = (a[1] == a[3]) && ((a[0] != a[2]) || (a[1] >= a[0]));
            const bool I4 = (b[1] == b[3]) && ((b[0] != b[2]) || (b[1] >= b[0]));
            return (I1 && I2) || (I3 && I4);
        };
        return check(0, 1) || check(2, 3) || check(4, 5);
    };

    // Bucket cells by centred-facet count + 2x2 flag.
    std::vector<std::size_t> cell_idx_M0, cell_idx_M2, cell_idx_M2x2, cell_idx_MN;
    std::vector<std::array<std::uint8_t, 8>> ranks_M0, ranks_M2, ranks_M2x2, ranks_MN;

    {
        std::size_t cursor = 0;
        for (std::size_t lin = 0; lin < CI.is_Mm.size(); ++lin) {
            if (!CI.is_Mm[lin]) continue;
            const auto& r = CI.cell_ranks_Mm[cursor++];
            std::array<bool, 6> isc{};
            find_centered(r, isc);
            int nc = 0; for (auto b : isc) if (b) ++nc;
            if (nc == 0) {
                cell_idx_M0.push_back(lin); ranks_M0.push_back(r);
            } else if (nc == 2) {
                cell_idx_M2.push_back(lin); ranks_M2.push_back(r);
            } else if (do_2x2patterns && nc == 4 && is_2x2_pattern(r)) {
                cell_idx_M2x2.push_back(lin); ranks_M2x2.push_back(r);
            } else {
                cell_idx_MN.push_back(lin); ranks_MN.push_back(r);
            }
        }
    }
    std::cout << " count 0 facet centered cells " << cell_idx_M0.size() << "\n";
    std::cout << " count 2 facet centered cells " << cell_idx_M2.size() << "\n";
    if (do_2x2patterns)
        std::cout << " count 2x2 facet centered cells " << cell_idx_M2x2.size() << "\n";
    std::cout << " count N>2 facet centered cells " << cell_idx_MN.size() << "\n";

    // lut8 row indexing: 8 base-8 ranks → 0..8^8-1. lut8[id] gives the
    // row within {lut0, lut2, lut2x2}.
    const auto& L = load_luts();
    auto ranks_to_lut8 = [](const std::array<std::uint8_t, 8>& r) {
        std::uint32_t v = 0;
        for (int i = 0; i < 8; ++i) v = v * 8 + r[i];
        return v;
    };

    auto mesh_via_lut = [&](const std::vector<std::array<std::uint8_t, 8>>& ranks,
                            const std::vector<std::size_t>& cell_idx,
                            const std::uint8_t* lut, std::size_t row_width,
                            std::vector<std::size_t>* out_indices,
                            std::vector<std::uint8_t>* out_rows) -> Triangles {
        std::size_t n = ranks.size();
        out_indices->reserve(n);
        out_rows->reserve(n * row_width);
        for (std::size_t i = 0; i < n; ++i) {
            const std::uint32_t id8 = ranks_to_lut8(ranks[i]);
            const std::uint16_t lut_row = L.lut8[id8];
            for (std::size_t k = 0; k < row_width; ++k)
                out_rows->push_back(lut[lut_row * row_width + k]);
            out_indices->push_back(cell_idx[i]);
        }
        return get_triangles_global_nodes(*out_rows, out_indices->size(),
                                          row_width, *out_indices);
    };

    Triangles tri_M0, tri_M2, tri_M2x2, tri_MN;
    {
        std::vector<std::uint8_t> rows; std::vector<std::size_t> idx;
        tri_M0 = mesh_via_lut(ranks_M0, cell_idx_M0,
                              L.lut0.data, 24, &idx, &rows);
    }
    {
        std::vector<std::uint8_t> rows; std::vector<std::size_t> idx;
        tri_M2 = mesh_via_lut(ranks_M2, cell_idx_M2,
                              L.lut2.data, 36, &idx, &rows);
    }
    if (do_2x2patterns) {
        std::vector<std::uint8_t> rows; std::vector<std::size_t> idx;
        tri_M2x2 = mesh_via_lut(ranks_M2x2, cell_idx_M2x2,
                                L.lut2x2.data, 42, &idx, &rows);
    }
    // The N>2 group uses cells2edges (no precomputed LUT). For each cell
    // we build the 48-element edges, then triangulate per-face into a
    // (n_cells × 72) triangle row (6 faces × 4 triangles × 3 nodes each
    // — matching the legacy `n_cells*6*4` allocation in
    // meshCellNcenteredFacet).
    {
        // Per-cell: 6 faces × up to 4 triangles × 3 vertices, but each
        // triangle uses (edge_u, edge_v, central_node=18). So we can
        // directly construct from the 48-element edges array: take pairs
        // (u, v) with u != 19 and emit (u, v, 18).
        std::vector<std::array<std::uint8_t, 48>> edges;
        cells2edges(ranks_MN, edges);

        std::vector<std::uint8_t> rows(ranks_MN.size() * 72, 19);
        for (std::size_t i = 0; i < ranks_MN.size(); ++i) {
            std::size_t out = 0;
            for (std::size_t k = 0; k < 48; k += 2) {
                const std::uint8_t u = edges[i][k];
                const std::uint8_t v = edges[i][k + 1];
                if (u == 19) continue;
                rows[i * 72 + out++] = u;
                rows[i * 72 + out++] = v;
                rows[i * 72 + out++] = 18;
            }
        }
        tri_MN = get_triangles_global_nodes(rows, ranks_MN.size(), 72, cell_idx_MN);
    }

    // ----- Merge all triangle groups ---------------------------------------
    auto append_tris = [](Triangles& a, const Triangles& b) {
        if (b.rows() == 0) return;
        Eigen::Index old = a.rows();
        a.conservativeResize(old + b.rows(), 3);
        a.bottomRows(b.rows()) = b;
    };

    Triangles tri_all = tri2;
    append_tris(tri_all, tri_M0);
    append_tris(tri_all, tri_M2);
    append_tris(tri_all, tri_M2x2);
    append_tris(tri_all, tri_MN);

    // ----- Dedup vertices by *coordinate* (not by global node id) ----------
    auto xyz_i = dedup_by_coordinate(tri_all, CI.img_size1);

    // ----- Crop to original image bounds (in INTEGER lattice space) --------
    crop_mesh_int(xyz_i, tri_all, img_orig);

    // ----- Convert to half-voxel float coords (xyz - 1, * 0.5) -------------
    Coords xyz(xyz_i.rows(), 3);
    for (Eigen::Index i = 0; i < xyz_i.rows(); ++i)
        for (int k = 0; k < 3; ++k)
            xyz(i, k) = (static_cast<double>(xyz_i(i, k)) - 1.0) * 0.5;

    // ----- Persist .npy artefacts ------------------------------------------
    if (!path_base.empty()) {
        paths::create_folder(path_base);
        // _xyz.npy as float32 (matches the reference xyz.astype(np.float32)).
        std::vector<float> xyz_f32(static_cast<std::size_t>(xyz.size()));
        for (Eigen::Index i = 0; i < xyz.size(); ++i)
            xyz_f32[static_cast<std::size_t>(i)] = static_cast<float>(xyz.data()[i]);
        npy::write<float>(path_base + "_xyz.npy",
                          {static_cast<std::size_t>(xyz.rows()), 3},
                          xyz_f32.data(), xyz_f32.size());

        // _tr.npy as uint32.
        std::vector<std::uint32_t> tri_u32(static_cast<std::size_t>(tri_all.size()));
        for (Eigen::Index i = 0; i < tri_all.size(); ++i)
            tri_u32[static_cast<std::size_t>(i)] = tri_all.data()[i];
        npy::write<std::uint32_t>(path_base + "_tr.npy",
                                  {static_cast<std::size_t>(tri_all.rows()), 3},
                                  tri_u32.data(), tri_u32.size());
    }

    // Write _ALL0.stl alongside the .npy files — matches the reference
    // createInitialMesh, which calls save_surfaces_stl(path_base, xyz, tr)
    // (no interfaces argument) before extractMaterialInterfaceInfo.
    if (!path_base.empty()) {
        io::save_stl_blocks(path_base + "_ALL0.stl", xyz, tri_all);
    }

    VOX2TET_PRINT("Initial Mesh Created!");
    return InitialMesh{ std::move(xyz), std::move(tri_all) };
}

// ---------------------------------------------------------------------------
// extract_material_interface_info — port of contouring.extractMaterial-
// InterfaceInfo. Classifies each vertex by type, assigns each triangle to
// a bimaterial interface, sorts triangles by interface, and persists
// _att.npy / _ntp.npy / _EXT.xyz / _FIX.xyz / _MML.xyz.
//
// The xyz coords passed here are in the *un-rotated* half-voxel float
// space (matching the legacy call order).
// ---------------------------------------------------------------------------
std::pair<std::vector<Interface>, NodeTypeMask>
extract_material_interface_info(const image::Volume& ext_vox,
                                const Coords& xyz,
                                Triangles& tri,
                                const std::string& path_base) {
    VOX2TET_PRINT("Collecting Materials Interface Data...");

    // The extended volume is shaped (nz+2, ny+2, nx+2); img_size is the
    // *original* image shape (without the 1-voxel border).
    const std::array<std::size_t, 3> img_size = {
        ext_vox.shape[0] - 2, ext_vox.shape[1] - 2, ext_vox.shape[2] - 2,
    };
    const std::size_t Ny_ext = ext_vox.shape[1];
    const std::size_t Nx_ext = ext_vox.shape[2];

    // --- compressUintArray(voxels): sorted-unique materials + per-voxel
    //     compressed index. We need both the original→index map and the
    //     unique-list (to reconstruct original colours for `interfaces`).
    const std::size_t N_vox = ext_vox.voxel_count();
    std::vector<std::uint64_t> uniq;
    uniq.reserve(64);
    {
        std::vector<std::uint64_t> all(N_vox);
        for (std::size_t i = 0; i < N_vox; ++i) all[i] = read_vox_u64(ext_vox, i);
        std::sort(all.begin(), all.end());
        all.erase(std::unique(all.begin(), all.end()), all.end());
        uniq = std::move(all);
    }
    const std::uint32_t n_materials = static_cast<std::uint32_t>(uniq.size());
    std::unordered_map<std::uint64_t, std::uint32_t> mat_to_idx;
    mat_to_idx.reserve(n_materials * 2);
    for (std::uint32_t i = 0; i < n_materials; ++i) mat_to_idx.emplace(uniq[i], i);

    // Per-voxel compressed index — used by the (xyz_i1, xyz_i2) lookup below.
    std::vector<std::uint32_t> vox_compressed(N_vox);
    for (std::size_t i = 0; i < N_vox; ++i)
        vox_compressed[i] = mat_to_idx.at(read_vox_u64(ext_vox, i));

    // --- xyz_i = ((xyz + 1)*2)  → integer doubled lattice (uint16 in the reference).
    const Eigen::Index N = xyz.rows();
    std::vector<std::array<std::int32_t, 3>> xyz_i(static_cast<std::size_t>(N));
    for (Eigen::Index i = 0; i < N; ++i) {
        xyz_i[static_cast<std::size_t>(i)] = {
            static_cast<std::int32_t>((xyz(i, 0) + 1.0) * 2.0),
            static_cast<std::int32_t>((xyz(i, 1) + 1.0) * 2.0),
            static_cast<std::int32_t>((xyz(i, 2) + 1.0) * 2.0),
        };
    }
    // Per-axis max for the bounding-box detection.
    std::array<std::int32_t, 3> axis_max = {0, 0, 0};
    for (const auto& p : xyz_i) for (int k = 0; k < 3; ++k) axis_max[k] = std::max(axis_max[k], p[k]);

    NodeTypeMask mask;
    for (auto& m : mask.masks) m.assign(static_cast<std::size_t>(N), 0);

    for (Eigen::Index i = 0; i < N; ++i) {
        const auto& p = xyz_i[static_cast<std::size_t>(i)];
        const int n_even = (p[0] % 2 == 0) + (p[1] % 2 == 0) + (p[2] % 2 == 0);
        if (n_even == 1) mask.masks[0][i] = 1;
        if (n_even == 2) mask.masks[1][i] = 1;
        if (n_even == 3) mask.masks[2][i] = 1;

        const bool b0 = (p[0] == 2 || p[0] == axis_max[0]);
        const bool b1 = (p[1] == 2 || p[1] == axis_max[1]);
        const bool b2 = (p[2] == 2 || p[2] == axis_max[2]);
        if (b0 || b1 || b2)                           mask.masks[3][i] = 1;
        if ((b0 && b1) || (b1 && b2) || (b2 && b0))   mask.masks[4][i] = 1;
        if (b0 && b1 && b2)                           mask.masks[5][i] = 1;
    }

    // --- Voxel lookup at the "below" and "above" sides of each vertex.
    //     xyz_i1 = ((xyz+1) - 0.1).astype(uint16)   (cell on the - side)
    //     xyz_i2 = ((xyz+1) + 0.1).astype(uint16)   (cell on the + side)
    // i2c = idx2xyz = [1, 2, 0]:
    //     vox-axis-0 = xyz_i[i2c[0]]   = xyz_i[1]              (Y axis)
    //     vox-axis-1 = img_size[1]+1 - xyz_i[i2c[1]] = - xyz_i[2]  (Z, flipped)
    //     vox-axis-2 = xyz_i[i2c[2]]   = xyz_i[0]              (X axis)
    std::vector<std::uint32_t> vox1(static_cast<std::size_t>(N));
    std::vector<std::uint32_t> vox2(static_cast<std::size_t>(N));
    for (Eigen::Index i = 0; i < N; ++i) {
        const double x = xyz(i, 0) + 1.0;
        const double y = xyz(i, 1) + 1.0;
        const double z = xyz(i, 2) + 1.0;
        const std::int32_t xi1[3] = {
            static_cast<std::int32_t>(x - 0.1),
            static_cast<std::int32_t>(y - 0.1),
            static_cast<std::int32_t>(z - 0.1),
        };
        const std::int32_t xi2[3] = {
            static_cast<std::int32_t>(x + 0.1),
            static_cast<std::int32_t>(y + 0.1),
            static_cast<std::int32_t>(z + 0.1),
        };
        auto lookup = [&](const std::int32_t* xi) -> std::uint32_t {
            const std::int64_t va0 = xi[1];
            const std::int64_t va1 = static_cast<std::int64_t>(img_size[1]) + 1 - xi[2];
            const std::int64_t va2 = xi[0];
            if (va0 < 0 || va1 < 0 || va2 < 0) return 0;
            if (static_cast<std::size_t>(va0) >= ext_vox.shape[0] ||
                static_cast<std::size_t>(va1) >= ext_vox.shape[1] ||
                static_cast<std::size_t>(va2) >= ext_vox.shape[2]) return 0;
            const std::size_t lin = (static_cast<std::size_t>(va0) * Ny_ext +
                                     static_cast<std::size_t>(va1)) * Nx_ext +
                                     static_cast<std::size_t>(va2);
            return vox_compressed[lin];
        };
        vox1[static_cast<std::size_t>(i)] = lookup(xi1);
        vox2[static_cast<std::size_t>(i)] = lookup(xi2);
    }

    // --- node_att[i] = (smaller, larger) packed as smaller*n_materials + larger.
    std::vector<std::uint32_t> node_att(static_cast<std::size_t>(N), 0);
    for (Eigen::Index i = 0; i < N; ++i) {
        if (!mask.masks[0][i]) continue;          // only edge-midpoint vertices
        const std::uint32_t v1 = vox1[i];
        const std::uint32_t v2 = vox2[i];
        if (v1 == v2) continue;
        if (v1 < v2) node_att[i] = v1 * n_materials + v2;
        else          node_att[i] = v2 * n_materials + v1;
    }

    // --- bimaterials[t] = max over t's 3 vertices of node_att.
    std::vector<std::uint32_t> bimat(static_cast<std::size_t>(tri.rows()), 0);
    for (Eigen::Index t = 0; t < tri.rows(); ++t) {
        bimat[static_cast<std::size_t>(t)] = std::max({
            node_att[tri(t, 0)], node_att[tri(t, 1)], node_att[tri(t, 2)],
        });
    }

    // --- Sort triangles by bimaterial (stable, to match the legacy argsort).
    std::vector<Eigen::Index> order(tri.rows());
    for (Eigen::Index i = 0; i < tri.rows(); ++i) order[i] = i;
    std::stable_sort(order.begin(), order.end(),
        [&](Eigen::Index a, Eigen::Index b) {
            return bimat[a] < bimat[b];
        });
    Triangles tri_sorted(tri.rows(), 3);
    std::vector<std::uint32_t> bimat_sorted(static_cast<std::size_t>(tri.rows()));
    for (Eigen::Index i = 0; i < tri.rows(); ++i) {
        tri_sorted.row(i)    = tri.row(order[i]);
        bimat_sorted[i]      = bimat[order[i]];
    }
    tri = std::move(tri_sorted);

    // --- interfaces: unique bimat codes + counts.
    std::vector<Interface> interfaces;
    if (!bimat_sorted.empty()) {
        std::size_t run_start = 0;
        std::uint32_t cur = bimat_sorted[0];
        for (std::size_t i = 1; i <= bimat_sorted.size(); ++i) {
            if (i == bimat_sorted.size() || bimat_sorted[i] != cur) {
                Interface it;
                it.code  = cur;
                it.first = static_cast<std::uint32_t>(run_start);
                it.count = static_cast<std::uint32_t>(i - run_start);
                it.mat1  = static_cast<std::uint32_t>(uniq[cur / n_materials]);
                it.mat2  = static_cast<std::uint32_t>(uniq[cur % n_materials]);
                interfaces.push_back(it);
                if (i < bimat_sorted.size()) {
                    cur = bimat_sorted[i];
                    run_start = i;
                }
            }
        }
    }

    // --- Persist artefacts.
    if (!path_base.empty()) {
        paths::create_folder(path_base);

        // _att.npy: (N_interfaces, 5) uint32
        std::vector<std::uint32_t> att(interfaces.size() * 5);
        for (std::size_t i = 0; i < interfaces.size(); ++i) {
            att[i * 5 + 0] = interfaces[i].code;
            att[i * 5 + 1] = interfaces[i].first;
            att[i * 5 + 2] = interfaces[i].count;
            att[i * 5 + 3] = interfaces[i].mat1;
            att[i * 5 + 4] = interfaces[i].mat2;
        }
        npy::write<std::uint32_t>(path_base + "_att.npy",
                                  {interfaces.size(), 5},
                                  att.data(), att.size());

        // _ntp.npy: (8, N) bool
        std::vector<std::uint8_t> ntp(8 * static_cast<std::size_t>(N));
        for (std::size_t r = 0; r < 8; ++r) {
            for (Eigen::Index i = 0; i < N; ++i)
                ntp[r * static_cast<std::size_t>(N) + i] = mask.masks[r][i];
        }
        npy::write<bool>(path_base + "_ntp.npy",
                        {8, static_cast<std::size_t>(N)},
                        reinterpret_cast<const bool*>(ntp.data()), ntp.size());

        // Re-write _xyz.npy / _tr.npy (the sorted triangle order, as
        // the reference does after extractMaterialInterfaceInfo).
        std::vector<float> xyz_f32(static_cast<std::size_t>(xyz.size()));
        for (Eigen::Index i = 0; i < xyz.size(); ++i)
            xyz_f32[static_cast<std::size_t>(i)] = static_cast<float>(xyz.data()[i]);
        npy::write<float>(path_base + "_xyz.npy",
                          {static_cast<std::size_t>(xyz.rows()), 3},
                          xyz_f32.data(), xyz_f32.size());
        std::vector<std::uint32_t> tri_u32(static_cast<std::size_t>(tri.size()));
        for (Eigen::Index i = 0; i < tri.size(); ++i)
            tri_u32[static_cast<std::size_t>(i)] = tri.data()[i];
        npy::write<std::uint32_t>(path_base + "_tr.npy",
                                  {static_cast<std::size_t>(tri.rows()), 3},
                                  tri_u32.data(), tri_u32.size());

        // _EXT.xyz / _FIX.xyz / _MML.xyz — per-mask vertex dumps.
        io::save_xyz_text(path_base + "_EXT.xyz", xyz, mask.masks[3]);
        io::save_xyz_text(path_base + "_FIX.xyz", xyz, mask.masks[2]);
        io::save_xyz_text(path_base + "_MML.xyz", xyz, mask.masks[1]);
    }

    VOX2TET_PRINT("Materials Interface Data Collected!");
    return {std::move(interfaces), std::move(mask)};
}

// ---------------------------------------------------------------------------
// save_surfaces_stl — writes _ALL.stl (with bimaterial attrs) plus the
// optional per-interface _I_{m1}_{m2}.stl files and per-grain
// _G_{mat}.stl / {mat}.inp dumps. Mirrors save_surfaces_stl in
// contouring.py.
// ---------------------------------------------------------------------------
void save_surfaces_stl(const std::string& path_base,
                       const Coords& xyz,
                       const Triangles& tri,
                       const std::vector<Interface>* interfaces,
                       bool do_save_interfaces,
                       bool do_save_grains_stl,
                       bool do_save_grains_inp) {
    VOX2TET_PRINT("The surface saving...");
    paths::create_folder(path_base);

    if (interfaces == nullptr || interfaces->empty()) {
        io::save_stl_blocks(path_base + "_ALL0.stl", xyz, tri);
        VOX2TET_PRINT("The surface saved!");
        return;
    }

    // Build per-triangle (mat1, mat2) from the run-length interfaces.
    std::vector<std::uint16_t> attr_mat1(static_cast<std::size_t>(tri.rows()), 0);
    std::vector<std::uint16_t> attr_mat2(static_cast<std::size_t>(tri.rows()), 0);
    for (const auto& it : *interfaces) {
        for (std::uint32_t k = 0; k < it.count; ++k) {
            const std::size_t idx = it.first + k;
            if (idx >= attr_mat1.size()) continue;
            attr_mat1[idx] = static_cast<std::uint16_t>(it.mat1);
            attr_mat2[idx] = static_cast<std::uint16_t>(it.mat2);
        }
    }

    // _ALL.stl — one mat1 per triangle (the reference only stores mat1; STL
    // attributes are 16-bit so multi-material info gets truncated).
    io::save_stl_blocks(path_base + "_ALL.stl", xyz, tri, &attr_mat1);

    // Optional per-interface dump.
    if (do_save_interfaces) {
        for (const auto& it : *interfaces) {
            Triangles slice(static_cast<Eigen::Index>(it.count), 3);
            for (std::uint32_t k = 0; k < it.count; ++k)
                slice.row(k) = tri.row(it.first + k);
            const std::string p = path_base + "_I_" + std::to_string(it.mat1)
                                + "_" + std::to_string(it.mat2) + ".stl";
            io::save_stl_blocks(p, xyz, slice);
        }
    }

    // Per-grain dump: for each material id m (skipping the 6 boundary
    // labels = the 6 max materials), gather triangles where it's mat1
    // (kept oriented) or mat2 (flipped to (b, a, c) so normals point
    // outwards), de-duplicate, write _G_{m}.stl + {m}.inp.
    if (!do_save_grains_stl && !do_save_grains_inp) {
        VOX2TET_PRINT("The surface saved!");
        return;
    }

    // Unique materials, sorted.
    std::vector<std::uint32_t> mats;
    for (const auto& it : *interfaces) { mats.push_back(it.mat1); mats.push_back(it.mat2); }
    std::sort(mats.begin(), mats.end());
    mats.erase(std::unique(mats.begin(), mats.end()), mats.end());
    if (mats.empty()) { VOX2TET_PRINT("The surface saved!"); return; }
    const std::uint32_t mmax = mats.back();
    // Folder for the *.inp files: alongside path_base.
    const std::string inp_folder = paths::base_folder(path_base);

    for (std::uint32_t m : mats) {
        if (m + 6 > mmax) continue;          // skip the 6 boundary labels
        std::vector<std::array<std::uint32_t, 3>> tr_m;
        tr_m.reserve(static_cast<std::size_t>(tri.rows() / std::max<std::size_t>(1, mats.size())));
        for (Eigen::Index t = 0; t < tri.rows(); ++t) {
            if (attr_mat1[t] == m) {
                tr_m.push_back({tri(t, 0), tri(t, 1), tri(t, 2)});
            } else if (attr_mat2[t] == m) {
                // Flip orientation so the normal still points "out of m".
                tr_m.push_back({tri(t, 1), tri(t, 0), tri(t, 2)});
            }
        }
        if (tr_m.empty()) continue;

        // Remove duplicated triangles (sorted-tuple uniqueness, both copies removed).
        {
            std::vector<std::array<std::uint32_t, 3>> sorted = tr_m;
            for (auto& t : sorted) std::sort(t.begin(), t.end());
            std::vector<std::pair<std::array<std::uint32_t, 3>, std::size_t>> idx_pairs;
            idx_pairs.reserve(sorted.size());
            for (std::size_t i = 0; i < sorted.size(); ++i)
                idx_pairs.push_back({sorted[i], i});
            std::sort(idx_pairs.begin(), idx_pairs.end());
            std::vector<std::array<std::uint32_t, 3>> keep;
            keep.reserve(tr_m.size());
            std::size_t i = 0;
            while (i < idx_pairs.size()) {
                std::size_t j = i + 1;
                while (j < idx_pairs.size() && idx_pairs[j].first == idx_pairs[i].first) ++j;
                if (j - i == 1) keep.push_back(tr_m[idx_pairs[i].second]);
                i = j;
            }
            tr_m = std::move(keep);
        }
        if (tr_m.empty()) continue;

        // Materialise as Triangles for the STL writer.
        Triangles slice(static_cast<Eigen::Index>(tr_m.size()), 3);
        for (std::size_t i = 0; i < tr_m.size(); ++i) {
            slice(static_cast<Eigen::Index>(i), 0) = tr_m[i][0];
            slice(static_cast<Eigen::Index>(i), 1) = tr_m[i][1];
            slice(static_cast<Eigen::Index>(i), 2) = tr_m[i][2];
        }

        if (do_save_grains_stl) {
            std::vector<std::uint16_t> grain_attr(tr_m.size(), static_cast<std::uint16_t>(m));
            const std::string p = path_base + "_G_" + std::to_string(m) + ".stl";
            io::save_stl_blocks(p, xyz, slice, &grain_attr);
        }
        if (do_save_grains_inp) {
            const std::string p = inp_folder + "/" + std::to_string(m) + ".inp";
            // INP writer needs Eigen::MatrixXi (signed); convert.
            Eigen::MatrixXi as_int(slice.rows(), 3);
            for (Eigen::Index i = 0; i < slice.rows(); ++i)
                for (int k = 0; k < 3; ++k) as_int(i, k) = static_cast<int>(slice(i, k));
            io::save_inp(p, xyz, as_int);
        }
    }
    VOX2TET_PRINT("The surface saved!");
}

}  // namespace vox2tet::marching_cubes
