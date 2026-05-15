#include "vox2tet/io/npy.hpp"

#include "vox2tet/core/paths.hpp"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>
#include <regex>
#include <sstream>

namespace vox2tet::npy {

namespace {

constexpr unsigned char kMagic[6] = {0x93, 'N', 'U', 'M', 'P', 'Y'};

DType parse_descr(const std::string& descr) {
    // Endianness prefix: '<' little, '|' irrelevant. We only support these.
    if (descr.empty()) throw std::runtime_error("npy: empty dtype descr");
    char endian = descr.front();
    if (endian == '>') throw std::runtime_error("npy: big-endian not supported");
    const std::string suffix = descr.substr(1);

    if (suffix == "b1") return DType::bool_;
    if (suffix == "u1") return DType::u8;
    if (suffix == "u2") return DType::u16;
    if (suffix == "u4") return DType::u32;
    if (suffix == "u8") return DType::u64;
    if (suffix == "i1") return DType::i8;
    if (suffix == "i2") return DType::i16;
    if (suffix == "i4") return DType::i32;
    if (suffix == "i8") return DType::i64;
    if (suffix == "f4") return DType::f32;
    if (suffix == "f8") return DType::f64;
    throw std::runtime_error("npy: unsupported dtype: " + descr);
}

}  // namespace

const char* dtype_descr(DType d) {
    switch (d) {
        case DType::bool_: return "|b1";
        case DType::u8:    return "|u1";
        case DType::u16:   return "<u2";
        case DType::u32:   return "<u4";
        case DType::u64:   return "<u8";
        case DType::i8:    return "|i1";
        case DType::i16:   return "<i2";
        case DType::i32:   return "<i4";
        case DType::i64:   return "<i8";
        case DType::f32:   return "<f4";
        case DType::f64:   return "<f8";
    }
    return "?";
}

std::size_t dtype_size(DType d) {
    switch (d) {
        case DType::bool_:
        case DType::u8:
        case DType::i8:    return 1;
        case DType::u16:
        case DType::i16:   return 2;
        case DType::u32:
        case DType::i32:
        case DType::f32:   return 4;
        case DType::u64:
        case DType::i64:
        case DType::f64:   return 8;
    }
    return 0;
}

Header read_header(std::ifstream& f) {
    unsigned char magic[6];
    f.read(reinterpret_cast<char*>(magic), 6);
    if (std::memcmp(magic, kMagic, 6) != 0) {
        throw std::runtime_error("npy: bad magic");
    }
    unsigned char ver[2];
    f.read(reinterpret_cast<char*>(ver), 2);
    std::size_t header_len = 0;
    if (ver[0] == 1) {
        std::uint16_t hl = 0;
        f.read(reinterpret_cast<char*>(&hl), 2);
        header_len = hl;
    } else {
        std::uint32_t hl = 0;
        f.read(reinterpret_cast<char*>(&hl), 4);
        header_len = hl;
    }

    std::string dict(header_len, '\0');
    f.read(dict.data(), static_cast<std::streamsize>(header_len));

    // Format: {'descr': '<f8', 'fortran_order': False, 'shape': (a, b), }
    Header h;
    std::smatch m;
    std::regex re_descr(R"('descr'\s*:\s*'([^']+)')");
    if (!std::regex_search(dict, m, re_descr))
        throw std::runtime_error("npy: cannot parse descr");
    h.dtype = parse_descr(m[1]);

    std::regex re_fo(R"('fortran_order'\s*:\s*(True|False))");
    if (std::regex_search(dict, m, re_fo)) h.fortran_order = (m[1] == "True");

    std::regex re_shape(R"('shape'\s*:\s*\(([^)]*)\))");
    if (!std::regex_search(dict, m, re_shape))
        throw std::runtime_error("npy: cannot parse shape");
    std::string shape_str = m[1];
    std::stringstream ss(shape_str);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        // strip whitespace
        tok.erase(std::remove_if(tok.begin(), tok.end(),
                                 [](unsigned char c) { return std::isspace(c); }),
                  tok.end());
        if (tok.empty()) continue;
        h.shape.push_back(static_cast<std::size_t>(std::stoull(tok)));
    }
    return h;
}

std::vector<std::uint8_t> read_raw(const std::string& path, Header& header) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("npy: cannot open " + path);
    header = read_header(f);
    std::size_t n = 1;
    for (auto d : header.shape) n *= d;
    const std::size_t elem_size = dtype_size(header.dtype);
    std::vector<std::uint8_t> out(n * elem_size);
    f.read(reinterpret_cast<char*>(out.data()),
           static_cast<std::streamsize>(out.size()));
    if (!f) throw std::runtime_error("npy: short read on " + path);

    // Transpose F-order → C-order so downstream code can assume the
    // canonical layout. Only handle 1D / 2D arrays for now (everything
    // we read from numpy is one of those).
    if (header.fortran_order && header.shape.size() >= 2) {
        if (header.shape.size() > 2) {
            throw std::runtime_error("npy: fortran_order with ndim>2 not supported");
        }
        const std::size_t rows = header.shape[0];
        const std::size_t cols = header.shape[1];
        std::vector<std::uint8_t> tx(out.size());
        for (std::size_t r = 0; r < rows; ++r) {
            for (std::size_t c = 0; c < cols; ++c) {
                const std::size_t src = (c * rows + r) * elem_size;     // F-order
                const std::size_t dst = (r * cols + c) * elem_size;     // C-order
                std::memcpy(tx.data() + dst, out.data() + src, elem_size);
            }
        }
        out = std::move(tx);
        header.fortran_order = false;
    }
    return out;
}

void write_raw(const std::string& path, DType dtype,
               const std::vector<std::size_t>& shape,
               const void* data, std::size_t nbytes) {
    paths::create_folder(path);
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("npy: cannot open for write: " + path);

    // Build header dict, then pad to multiple of 64 - 10 (magic+ver+len).
    std::ostringstream dict;
    dict << "{'descr': '" << dtype_descr(dtype) << "', 'fortran_order': False, "
         << "'shape': (";
    for (std::size_t i = 0; i < shape.size(); ++i) {
        dict << shape[i];
        if (i + 1 < shape.size() || shape.size() == 1) dict << ",";
        if (i + 1 < shape.size()) dict << " ";
    }
    dict << "), }";

    std::string h = dict.str();
    // 10 bytes preamble (6 magic + 2 ver + 2 hlen). Pad header so total is
    // multiple of 64; trailing '\n'.
    constexpr std::size_t kAlign = 64;
    std::size_t total = 10 + h.size() + 1;  // +1 for newline
    std::size_t pad = (kAlign - (total % kAlign)) % kAlign;
    h.append(pad, ' ');
    h.push_back('\n');

    f.write(reinterpret_cast<const char*>(kMagic), 6);
    unsigned char ver[2] = {1, 0};
    f.write(reinterpret_cast<const char*>(ver), 2);
    std::uint16_t hl = static_cast<std::uint16_t>(h.size());
    f.write(reinterpret_cast<const char*>(&hl), 2);
    f.write(h.data(), static_cast<std::streamsize>(h.size()));
    f.write(reinterpret_cast<const char*>(data),
            static_cast<std::streamsize>(nbytes));
    if (!f) throw std::runtime_error("npy: short write to " + path);
}

}  // namespace vox2tet::npy
