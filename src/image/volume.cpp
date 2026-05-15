#include "vox2tet/image/volume.hpp"

#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace vox2tet::image {

std::size_t Volume::item_size() const {
    switch (dtype) {
        case VoxelType::U8:  return 1;
        case VoxelType::U16: return 2;
        case VoxelType::U32: return 4;
        case VoxelType::U64: return 8;
        case VoxelType::F32: return 4;
        case VoxelType::F64: return 8;
    }
    return 0;
}

template <typename T> T* Volume::as_impl() {
    constexpr auto requested =
        sizeof(T) == 1 ? VoxelType::U8  :
        sizeof(T) == 2 ? VoxelType::U16 :
        sizeof(T) == 4 ? VoxelType::U32 :
        sizeof(T) == 8 ? VoxelType::U64 : VoxelType::U8;
    if (requested != dtype) return nullptr;
    return reinterpret_cast<T*>(bytes.data());
}

// Explicit instantiations so users link without pulling in the impl.
template std::uint8_t*  Volume::as_impl<std::uint8_t>();
template std::uint16_t* Volume::as_impl<std::uint16_t>();
template std::uint32_t* Volume::as_impl<std::uint32_t>();
template std::uint64_t* Volume::as_impl<std::uint64_t>();

std::uint64_t Volume::max_value() const {
    if (bytes.empty()) return 0;
    const std::size_t n = voxel_count();
    switch (dtype) {
        case VoxelType::U8: {
            const auto* p = reinterpret_cast<const std::uint8_t*>(bytes.data());
            return *std::max_element(p, p + n);
        }
        case VoxelType::U16: {
            const auto* p = reinterpret_cast<const std::uint16_t*>(bytes.data());
            return *std::max_element(p, p + n);
        }
        case VoxelType::U32: {
            const auto* p = reinterpret_cast<const std::uint32_t*>(bytes.data());
            return *std::max_element(p, p + n);
        }
        case VoxelType::U64: {
            const auto* p = reinterpret_cast<const std::uint64_t*>(bytes.data());
            return *std::max_element(p, p + n);
        }
        case VoxelType::F32: {
            const auto* p = reinterpret_cast<const float*>(bytes.data());
            float m = *std::max_element(p, p + n);
            return static_cast<std::uint64_t>(m < 0 ? 0.0f : m);
        }
        case VoxelType::F64: {
            const auto* p = reinterpret_cast<const double*>(bytes.data());
            double m = *std::max_element(p, p + n);
            return static_cast<std::uint64_t>(m < 0 ? 0.0 : m);
        }
    }
    return 0;
}

Volume Volume::compress_to_smallest(Volume&& v, std::uint64_t headroom) {
    const std::uint64_t m = v.max_value();
    VoxelType target = v.dtype;
    if (m + headroom <= std::numeric_limits<std::uint8_t>::max()) target = VoxelType::U8;
    else if (m + headroom <= std::numeric_limits<std::uint16_t>::max()) target = VoxelType::U16;
    else if (m + headroom <= std::numeric_limits<std::uint32_t>::max()) target = VoxelType::U32;
    else target = VoxelType::U64;

    if (target == v.dtype) return std::move(v);

    Volume out;
    out.shape = v.shape;
    out.dtype = target;
    const std::size_t n = v.voxel_count();
    const std::size_t isz = (target == VoxelType::U8) ? 1 :
                            (target == VoxelType::U16) ? 2 :
                            (target == VoxelType::U32) ? 4 : 8;
    out.bytes.resize(n * isz);

    auto convert = [&](auto src_tag, auto dst_tag) {
        using S = decltype(src_tag);
        using D = decltype(dst_tag);
        const S* sp = reinterpret_cast<const S*>(v.bytes.data());
        D*       dp = reinterpret_cast<D*>(out.bytes.data());
        for (std::size_t i = 0; i < n; ++i) dp[i] = static_cast<D>(sp[i]);
    };

    auto run = [&](auto src_tag) {
        using S = decltype(src_tag);
        switch (target) {
            case VoxelType::U8:  convert(S{}, std::uint8_t{});  break;
            case VoxelType::U16: convert(S{}, std::uint16_t{}); break;
            case VoxelType::U32: convert(S{}, std::uint32_t{}); break;
            case VoxelType::U64: convert(S{}, std::uint64_t{}); break;
            case VoxelType::F32:
            case VoxelType::F64:
                // unreachable — target is always an unsigned int here
                throw std::runtime_error("compress_to_smallest: float target");
        }
    };
    switch (v.dtype) {
        case VoxelType::U8:  run(std::uint8_t{});  break;
        case VoxelType::U16: run(std::uint16_t{}); break;
        case VoxelType::U32: run(std::uint32_t{}); break;
        case VoxelType::U64: run(std::uint64_t{}); break;
        case VoxelType::F32: run(float{});         break;
        case VoxelType::F64: run(double{});        break;
    }
    return out;
}

}  // namespace vox2tet::image
