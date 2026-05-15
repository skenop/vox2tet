#pragma once

// Look-up tables required by the multi-material marching cubes algorithm.
// Five tables are involved:
//
//   * lut8           : uint16[16'777'216] mapping a global 8-color cell
//                      code (0..8^8-1) to a slot inside one of the three
//                      cell LUTs below. Stored as `data/lut8.bin`.
//   * lut_0centered  : uint8[N0 × 24]  cells with no centred facet (N0=705)
//   * lut_2centered  : uint8[N2 × 36]  cells with two centred facets
//   * lut_2x2        : uint8[N2x2 × 42] cells with 2x2 parallel centred lines
//   * facet_lut_2d   : uint8[256 × 8]  60-config facet table (m3c.py)
//
// Each LUT value is a local node id in [0, 18]; the sentinel 19 means
// "dummy" (no triangle past this slot).
//
// The binaries are generated once from the original tables via
// `tools/export_luts.py` and copied to
// `cpp/data/`. They are mmap-ed at startup.

#include <cstddef>
#include <cstdint>
#include <string>

namespace vox2tet::marching_cubes {

// std::span replacement (we're on C++17). Trivially copyable view over
// contiguous read-only memory owned elsewhere.
template <typename T>
struct View {
    const T*    data = nullptr;
    std::size_t size = 0;
    bool empty() const { return size == 0; }
    const T& operator[](std::size_t i) const { return data[i]; }
};

struct LutPack {
    View<std::uint16_t> lut8;          // size 16'777'216
    View<std::uint8_t>  lut0;          // rows × 24, row-major
    View<std::uint8_t>  lut2;          // rows × 36, row-major
    View<std::uint8_t>  lut2x2;        // rows × 42, row-major
    std::size_t         lut0_rows = 0;
    std::size_t         lut2_rows = 0;
    std::size_t         lut2x2_rows = 0;
    View<std::uint8_t>  facet_lut2d;   // 256 × 8
};

// Load all five tables from `data_dir` (defaults to compile-time
// VOX2TET_DATA_DIR).  Throws if a file is missing or has unexpected size.
//
// Two overloads exist instead of a default argument so call sites can
// write `auto& L = load_luts();` without tripping GCC 13's
// -Wdangling-reference check on the implicit temporary string.
const LutPack& load_luts();
const LutPack& load_luts(const std::string& data_dir);

}  // namespace vox2tet::marching_cubes
