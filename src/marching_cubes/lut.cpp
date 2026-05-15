#include "vox2tet/marching_cubes/lut.hpp"

#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <vector>

namespace fs = std::filesystem;

namespace vox2tet::marching_cubes {

namespace {

#ifndef VOX2TET_DATA_DIR
#  define VOX2TET_DATA_DIR "./data"
#endif

std::vector<std::uint8_t> read_all(const fs::path& p) {
    std::ifstream f(p, std::ios::binary);
    if (!f) throw std::runtime_error("lut: cannot open " + p.string());
    f.seekg(0, std::ios::end);
    auto sz = static_cast<std::size_t>(f.tellg());
    f.seekg(0, std::ios::beg);
    std::vector<std::uint8_t> b(sz);
    f.read(reinterpret_cast<char*>(b.data()), sz);
    return b;
}

struct OwnedLuts {
    std::vector<std::uint8_t> lut8_b;
    std::vector<std::uint8_t> lut0_b;
    std::vector<std::uint8_t> lut2_b;
    std::vector<std::uint8_t> lut2x2_b;
    std::vector<std::uint8_t> facet2d_b;
    LutPack                    pack;
};

OwnedLuts& storage() {
    static OwnedLuts S;
    return S;
}

}  // namespace

const LutPack& load_luts() { return load_luts(std::string{}); }

const LutPack& load_luts(const std::string& data_dir) {
    auto& S = storage();
    if (!S.pack.lut8.empty()) return S.pack;

    fs::path root = data_dir.empty() ? fs::path(VOX2TET_DATA_DIR) : fs::path(data_dir);

    S.lut8_b   = read_all(root / "lut8.bin");
    S.lut0_b   = read_all(root / "lut0.bin");
    S.lut2_b   = read_all(root / "lut2.bin");
    S.lut2x2_b = read_all(root / "lut2x2.bin");
    S.facet2d_b= read_all(root / "facet_lut2d.bin");

    if (S.lut8_b.size() != 16'777'216 * sizeof(std::uint16_t)) {
        throw std::runtime_error("lut: lut8.bin has unexpected size");
    }
    if (S.facet2d_b.size() != 256 * 8) {
        throw std::runtime_error("lut: facet_lut2d.bin has unexpected size");
    }
    if (S.lut0_b.size() % 24 != 0)
        throw std::runtime_error("lut: lut0.bin row-size mismatch");
    if (S.lut2_b.size() % 36 != 0)
        throw std::runtime_error("lut: lut2.bin row-size mismatch");
    if (S.lut2x2_b.size() % 42 != 0)
        throw std::runtime_error("lut: lut2x2.bin row-size mismatch");

    S.pack.lut8        = { reinterpret_cast<const std::uint16_t*>(S.lut8_b.data()),
                           S.lut8_b.size() / 2 };
    S.pack.lut0        = { S.lut0_b.data(),   S.lut0_b.size()   };
    S.pack.lut2        = { S.lut2_b.data(),   S.lut2_b.size()   };
    S.pack.lut2x2      = { S.lut2x2_b.data(), S.lut2x2_b.size() };
    S.pack.lut0_rows   = S.lut0_b.size()   / 24;
    S.pack.lut2_rows   = S.lut2_b.size()   / 36;
    S.pack.lut2x2_rows = S.lut2x2_b.size() / 42;
    S.pack.facet_lut2d = { S.facet2d_b.data(), S.facet2d_b.size() };
    return S.pack;
}

}  // namespace vox2tet::marching_cubes
