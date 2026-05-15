#pragma once

// Minimal reader/writer for the NumPy `.npy` v1.0 format. Only the subset
// used by vox2tet is supported:
//   - little-endian
//   - C-order arrays
//   - dtypes: |b1 (bool), |u1, <u2, <u4, <u8, <i1..<i8, <f4, <f8
//   - up to 4 dimensions
//
// API is dtype-erased on the read side: callers say what they expect and
// the reader complains on mismatch. This is enough for the pipeline's
// needs (xyz, triangles, attributes, etc.) without pulling in a heavy
// dependency.

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace vox2tet::npy {

enum class DType {
    bool_,
    u8, u16, u32, u64,
    i8, i16, i32, i64,
    f32, f64,
};

struct Header {
    DType                 dtype;
    bool                  fortran_order = false;  // we only write false
    std::vector<std::size_t> shape;
};

template <typename T> constexpr DType dtype_of();
template <> constexpr DType dtype_of<bool>()            { return DType::bool_; }
template <> constexpr DType dtype_of<std::uint8_t>()    { return DType::u8;  }
template <> constexpr DType dtype_of<std::uint16_t>()   { return DType::u16; }
template <> constexpr DType dtype_of<std::uint32_t>()   { return DType::u32; }
template <> constexpr DType dtype_of<std::uint64_t>()   { return DType::u64; }
template <> constexpr DType dtype_of<std::int8_t>()     { return DType::i8;  }
template <> constexpr DType dtype_of<std::int16_t>()    { return DType::i16; }
template <> constexpr DType dtype_of<std::int32_t>()    { return DType::i32; }
template <> constexpr DType dtype_of<std::int64_t>()    { return DType::i64; }
template <> constexpr DType dtype_of<float>()           { return DType::f32; }
template <> constexpr DType dtype_of<double>()          { return DType::f64; }

const char* dtype_descr(DType d);   // "<f8" etc.
std::size_t dtype_size(DType d);

Header read_header(std::ifstream& f);

// Read raw bytes (`shape` is filled, caller reshape as needed).
std::vector<std::uint8_t> read_raw(const std::string& path, Header& header);

template <typename T>
std::vector<T> read(const std::string& path, std::vector<std::size_t>& shape) {
    Header h;
    auto bytes = read_raw(path, h);
    if (h.dtype != dtype_of<T>()) {
        throw std::runtime_error("npy::read: dtype mismatch reading " + path);
    }
    shape = h.shape;
    const std::size_t n = bytes.size() / sizeof(T);
    std::vector<T> out(n);
    std::memcpy(out.data(), bytes.data(), n * sizeof(T));
    return out;
}

void write_raw(const std::string& path,
               DType dtype,
               const std::vector<std::size_t>& shape,
               const void* data, std::size_t nbytes);

template <typename T>
void write(const std::string& path,
           const std::vector<std::size_t>& shape,
           const T* data, std::size_t count) {
    write_raw(path, dtype_of<T>(), shape, data, count * sizeof(T));
}

}  // namespace vox2tet::npy
