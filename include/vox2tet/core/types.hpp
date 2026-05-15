#pragma once

// Common numeric aliases used throughout the codebase. Mirror the dtypes
// chosen by the reference implementation so we can byte-compare outputs.

#include <array>
#include <cstdint>
#include <vector>

#include <Eigen/Core>

namespace vox2tet {

using u8  = std::uint8_t;
using u16 = std::uint16_t;
using u32 = std::uint32_t;
using u64 = std::uint64_t;
using i32 = std::int32_t;
using i64 = std::int64_t;
using f32 = float;
using f64 = double;

inline constexpr u32 kInvalidU32 = static_cast<u32>(-1);

// Dense Nx3 matrices, row-major so a row is contiguous (matches NumPy
// (N,3) C-order layout).
template <typename T>
using MatrixNx3 = Eigen::Matrix<T, Eigen::Dynamic, 3, Eigen::RowMajor>;

using Coords      = MatrixNx3<f64>;
using CoordsF32   = MatrixNx3<f32>;
using Triangles   = MatrixNx3<u32>;
using NormalsMat  = MatrixNx3<f64>;

// Image axis ordering used by the reference code:
//   xyz -> image index : np.array([2, 0, 1])
//   image index -> xyz : np.array([1, 2, 0])
inline constexpr std::array<int, 3> kXyzToIdx = {2, 0, 1};
inline constexpr std::array<int, 3> kIdxToXyz = {1, 2, 0};

}  // namespace vox2tet
