#include "vox2tet/image/ext_volume.hpp"

#include "vox2tet/io/image_io.hpp"

#include <cstring>
#include <stdexcept>

namespace vox2tet::image {

namespace {

// Set every voxel in the rectangular region [z0..z1)×[y0..y1)×[x0..x1) of
// `v` to `value`. Dtype is u64-erased — we narrow on store.
template <typename T>
void fill_region(Volume& v, std::size_t z0, std::size_t z1,
                 std::size_t y0, std::size_t y1,
                 std::size_t x0, std::size_t x1,
                 std::uint64_t value) {
    T*     p   = reinterpret_cast<T*>(v.bytes.data());
    const auto nx = v.shape[2];
    const auto ny = v.shape[1];
    T val = static_cast<T>(value);
    for (std::size_t z = z0; z < z1; ++z)
    for (std::size_t y = y0; y < y1; ++y) {
        T* row = p + (z * ny + y) * nx;
        std::fill(row + x0, row + x1, val);
    }
}

void fill_region_dispatch(Volume& v,
                          std::size_t z0, std::size_t z1,
                          std::size_t y0, std::size_t y1,
                          std::size_t x0, std::size_t x1,
                          std::uint64_t value) {
    switch (v.dtype) {
        case VoxelType::U8:  fill_region<std::uint8_t >(v, z0, z1, y0, y1, x0, x1, value); break;
        case VoxelType::U16: fill_region<std::uint16_t>(v, z0, z1, y0, y1, x0, x1, value); break;
        case VoxelType::U32: fill_region<std::uint32_t>(v, z0, z1, y0, y1, x0, x1, value); break;
        case VoxelType::U64: fill_region<std::uint64_t>(v, z0, z1, y0, y1, x0, x1, value); break;
        case VoxelType::F32:
        case VoxelType::F64:
            // Float volumes are narrowed to int on read; should never reach here.
            break;
    }
}

}  // namespace

std::pair<Volume, std::array<std::size_t, 3>> ext_volume_from_image(const Volume& img) {
    const std::size_t nz = img.shape[0];
    const std::size_t ny = img.shape[1];
    const std::size_t nx = img.shape[2];

    // Promote dtype if max + 26 would overflow.
    Volume tmp = img;  // copy bytes
    tmp = Volume::compress_to_smallest(std::move(tmp), /*headroom=*/26);
    const std::uint64_t max_label = tmp.max_value();

    Volume out;
    out.shape = {nz + 2, ny + 2, nx + 2};
    out.dtype = tmp.dtype;
    out.bytes.assign(out.voxel_count() * out.item_size(), 0);

    // Copy interior.
    const std::size_t isz = tmp.item_size();
    const std::size_t Nx = out.shape[2];
    const std::size_t Ny = out.shape[1];
    for (std::size_t z = 0; z < nz; ++z)
    for (std::size_t y = 0; y < ny; ++y) {
        const auto* src = tmp.bytes.data() + ((z * ny + y) * nx) * isz;
        auto*       dst = out.bytes.data() +
                          (((z + 1) * Ny + (y + 1)) * Nx + 1) * isz;
        std::memcpy(dst, src, nx * isz);
    }

    // 26 boundary labels in the SAME order as getVoxelsEx.
    const std::size_t Z = out.shape[0], Y = out.shape[1], X = out.shape[2];
    auto fill = [&](std::size_t z0, std::size_t z1,
                    std::size_t y0, std::size_t y1,
                    std::size_t x0, std::size_t x1,
                    std::uint64_t v) {
        fill_region_dispatch(out, z0, z1, y0, y1, x0, x1, v);
    };

    fill(0,     1,     0,     Y,     0,     X,     max_label + 1);
    fill(Z - 1, Z,     0,     Y,     0,     X,     max_label + 2);
    fill(0,     Z,     0,     1,     0,     X,     max_label + 3);
    fill(0,     Z,     Y - 1, Y,     0,     X,     max_label + 4);
    fill(0,     Z,     0,     Y,     0,     1,     max_label + 5);
    fill(0,     Z,     0,     Y,     X - 1, X,     max_label + 6);

    fill(0,     1,     0,     1,     0,     X,     max_label + 7);
    fill(0,     1,     Y - 1, Y,     0,     X,     max_label + 8);
    fill(0,     1,     0,     Y,     0,     1,     max_label + 9);
    fill(0,     1,     0,     Y,     X - 1, X,     max_label + 10);

    fill(Z - 1, Z,     0,     1,     0,     X,     max_label + 11);
    fill(Z - 1, Z,     Y - 1, Y,     0,     X,     max_label + 12);
    fill(Z - 1, Z,     0,     Y,     0,     1,     max_label + 13);
    fill(Z - 1, Z,     0,     Y,     X - 1, X,     max_label + 14);

    fill(0,     Z,     0,     1,     0,     1,     max_label + 15);
    fill(0,     Z,     0,     1,     X - 1, X,     max_label + 16);
    fill(0,     Z,     Y - 1, Y,     0,     1,     max_label + 17);
    fill(0,     Z,     Y - 1, Y,     X - 1, X,     max_label + 18);

    fill(0,     1,     0,     1,     0,     1,     max_label + 19);
    fill(0,     1,     0,     1,     X - 1, X,     max_label + 20);
    fill(0,     1,     Y - 1, Y,     0,     1,     max_label + 21);
    fill(0,     1,     Y - 1, Y,     X - 1, X,     max_label + 22);

    fill(Z - 1, Z,     0,     1,     0,     1,     max_label + 23);
    fill(Z - 1, Z,     0,     1,     X - 1, X,     max_label + 24);
    fill(Z - 1, Z,     Y - 1, Y,     0,     1,     max_label + 25);
    fill(Z - 1, Z,     Y - 1, Y,     X - 1, X,     max_label + 26);

    return {std::move(out), {nz, ny, nx}};
}

std::pair<Volume, std::array<std::size_t, 3>> ext_volume_from_file(const std::string& path) {
    auto img = io::read_image(path, std::nullopt, /*try_compress=*/true);
    return ext_volume_from_image(img);
}

}  // namespace vox2tet::image
