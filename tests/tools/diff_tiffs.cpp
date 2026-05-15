// Tiny diff tool. Reads two TIFF stacks via vox2tet's reader and compares
// them voxel-by-voxel. Used for cross-checking against the Python outputs.
//
//   diff_tiffs <a.tif> <b.tif>

#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>

#include "vox2tet/image/volume.hpp"
#include "vox2tet/io/image_io.hpp"

namespace v = vox2tet;

static std::uint64_t read_voxel_u64(const v::image::Volume& vol, std::size_t lin) {
    switch (vol.dtype) {
        case v::image::VoxelType::U8:
            return reinterpret_cast<const std::uint8_t*>(vol.bytes.data())[lin];
        case v::image::VoxelType::U16:
            return reinterpret_cast<const std::uint16_t*>(vol.bytes.data())[lin];
        case v::image::VoxelType::U32:
            return reinterpret_cast<const std::uint32_t*>(vol.bytes.data())[lin];
        case v::image::VoxelType::U64:
            return reinterpret_cast<const std::uint64_t*>(vol.bytes.data())[lin];
        default:
            return 0;
    }
}

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "usage: diff_tiffs <a.tif> <b.tif>\n";
        return 2;
    }
    auto a = v::io::read_tiff3(argv[1], /*try_compress=*/false);
    auto b = v::io::read_tiff3(argv[2], /*try_compress=*/false);
    if (a.shape != b.shape) {
        std::cerr << "shape mismatch: ("
                  << a.shape[0] << "," << a.shape[1] << "," << a.shape[2] << ") vs ("
                  << b.shape[0] << "," << b.shape[1] << "," << b.shape[2] << ")\n";
        return 1;
    }
    std::size_t n = a.voxel_count();
    std::size_t diff = 0;
    std::size_t first = static_cast<std::size_t>(-1);
    for (std::size_t i = 0; i < n; ++i) {
        if (read_voxel_u64(a, i) != read_voxel_u64(b, i)) {
            if (diff == 0) first = i;
            ++diff;
        }
    }
    std::cout << "shape=(" << a.shape[0] << "," << a.shape[1] << "," << a.shape[2]
              << ") a.dtype=" << static_cast<int>(a.dtype)
              << " b.dtype=" << static_cast<int>(b.dtype)
              << " diff=" << diff << "/" << n;
    if (diff > 0) {
        const std::size_t z = first / (a.shape[1] * a.shape[2]);
        const std::size_t r = first % (a.shape[1] * a.shape[2]);
        const std::size_t y = r / a.shape[2];
        const std::size_t x = r % a.shape[2];
        std::cout << "  first at (z=" << z << ",y=" << y << ",x=" << x
                  << ") a=" << read_voxel_u64(a, first)
                  << " b=" << read_voxel_u64(b, first);
    }
    std::cout << "\n";
    return diff == 0 ? 0 : 1;
}
