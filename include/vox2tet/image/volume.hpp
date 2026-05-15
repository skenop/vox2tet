#pragma once

// 3D voxel volume — type-erased over the integer dtype (u8 / u16 / u32 /
// u64) the same way numpy arrays carry their dtype. Layout mirrors NumPy
// (Z, Y, X) C-order so addressing matches the reference implementation:
//     voxels[z, y, x]   <==>   data[z * ny * nx + y * nx + x]

#include <cstddef>
#include <cstdint>
#include <vector>

#include "vox2tet/core/types.hpp"

namespace vox2tet::image {

enum class VoxelType { U8, U16, U32, U64, F32, F64 };

struct Volume {
    VoxelType                  dtype = VoxelType::U8;
    std::array<std::size_t, 3> shape = {0, 0, 0};   // (nz, ny, nx)
    std::vector<std::uint8_t>  bytes;               // raw storage

    std::size_t voxel_count() const { return shape[0] * shape[1] * shape[2]; }
    std::size_t item_size()   const;
    bool        empty()       const { return bytes.empty(); }

    // Linear index helpers.
    std::size_t lin(std::size_t z, std::size_t y, std::size_t x) const {
        return (z * shape[1] + y) * shape[2] + x;
    }

    // Typed access — call sites must know the dtype (we expose it via
    // dtype()). Returns nullptr if dtype mismatches the request.
    template <typename T> T*       as()       { return as_impl<T>(); }
    template <typename T> const T* as() const { return const_cast<Volume*>(this)->as_impl<T>(); }

    // Compute the max value across all voxels (returned as u64).
    std::uint64_t max_value() const;

    // Promote/demote to fit `min_dtype` headroom (used after extending by
    // the 26 boundary labels). Mirrors `IOimage.compress_img`.
    static Volume compress_to_smallest(Volume&& v, std::uint64_t headroom = 26);

private:
    template <typename T> T* as_impl();
};

}  // namespace vox2tet::image
