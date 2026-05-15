#include "vox2tet/io/image_io.hpp"

#include "vox2tet/core/log.hpp"
#include "vox2tet/core/paths.hpp"
#include "vox2tet/io/npy.hpp"

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>

extern "C" {
#include <tiffio.h>
}

namespace fs = std::filesystem;

namespace vox2tet::io {

namespace {

image::VoxelType npy_to_voxel(npy::DType d) {
    switch (d) {
        case npy::DType::u8:  return image::VoxelType::U8;
        case npy::DType::u16: return image::VoxelType::U16;
        case npy::DType::u32: return image::VoxelType::U32;
        case npy::DType::u64: return image::VoxelType::U64;
        case npy::DType::f32: return image::VoxelType::F32;
        case npy::DType::f64: return image::VoxelType::F64;
        default:
            throw std::runtime_error("npy: image must be unsigned int or float");
    }
}

}  // namespace

image::Volume read_npy3(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("read_npy3: cannot open " + path);
    npy::Header h = npy::read_header(f);
    if (h.shape.size() != 3) {
        throw std::runtime_error("read_npy3: expected 3D array in " + path);
    }
    image::Volume v;
    v.shape = {h.shape[0], h.shape[1], h.shape[2]};
    v.dtype = npy_to_voxel(h.dtype);
    std::size_t n = v.voxel_count();
    v.bytes.resize(n * v.item_size());
    f.read(reinterpret_cast<char*>(v.bytes.data()),
           static_cast<std::streamsize>(v.bytes.size()));
    // Match `IOimage.compress_img`: float (and oversized int) inputs get
    // narrowed to the smallest unsigned int dtype that fits the value
    // range with 26 voxels of headroom for ext_volume's boundary labels.
    if (v.dtype == image::VoxelType::F32 || v.dtype == image::VoxelType::F64) {
        v = image::Volume::compress_to_smallest(std::move(v));
    }
    return v;
}

image::Volume read_raw3(const std::string& path,
                        std::array<std::size_t, 3> shape,
                        image::VoxelType dtype) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("read_raw3: cannot open " + path);
    image::Volume v;
    v.shape = shape;
    v.dtype = dtype;
    std::size_t n = v.voxel_count();
    v.bytes.resize(n * v.item_size());
    f.read(reinterpret_cast<char*>(v.bytes.data()),
           static_cast<std::streamsize>(v.bytes.size()));
    return v;
}

image::Volume read_tiff3(const std::string& path, bool try_compress) {
    TIFF* tif = TIFFOpen(path.c_str(), "r");
    if (!tif) throw std::runtime_error("TIFF open failed: " + path);

    std::uint32_t w = 0, h = 0;
    std::uint16_t bps = 0, spp = 1, sf = SAMPLEFORMAT_UINT;
    TIFFGetField(tif, TIFFTAG_IMAGEWIDTH,      &w);
    TIFFGetField(tif, TIFFTAG_IMAGELENGTH,     &h);
    TIFFGetField(tif, TIFFTAG_BITSPERSAMPLE,   &bps);
    TIFFGetFieldDefaulted(tif, TIFFTAG_SAMPLESPERPIXEL, &spp);
    TIFFGetFieldDefaulted(tif, TIFFTAG_SAMPLEFORMAT,    &sf);
    if (spp != 1 ||
        (sf != SAMPLEFORMAT_UINT && sf != SAMPLEFORMAT_INT &&
         sf != SAMPLEFORMAT_IEEEFP)) {
        TIFFClose(tif);
        throw std::runtime_error("TIFF: only single-channel int/float supported: " + path);
    }

    // Count pages (Z-slices) by walking directories.
    std::size_t nz = 0;
    do { ++nz; } while (TIFFReadDirectory(tif));
    TIFFSetDirectory(tif, 0);

    image::Volume v;
    v.shape = { nz, static_cast<std::size_t>(h), static_cast<std::size_t>(w) };
    if (sf == SAMPLEFORMAT_IEEEFP) {
        switch (bps) {
            case 32: v.dtype = image::VoxelType::F32; break;
            case 64: v.dtype = image::VoxelType::F64; break;
            default:
                TIFFClose(tif);
                throw std::runtime_error("TIFF: float must be 32 or 64 bit");
        }
    } else {  // UINT or INT — store both in unsigned slots
        switch (bps) {
            case 8:  v.dtype = image::VoxelType::U8;  break;
            case 16: v.dtype = image::VoxelType::U16; break;
            case 32: v.dtype = image::VoxelType::U32; break;
            case 64: v.dtype = image::VoxelType::U64; break;
            default:
                TIFFClose(tif);
                throw std::runtime_error("TIFF: unsupported BitsPerSample");
        }
    }
    v.bytes.resize(v.voxel_count() * v.item_size());

    const std::size_t slice_bytes = static_cast<std::size_t>(w) * h * (bps / 8);
    std::size_t z = 0;
    do {
        std::size_t row_bytes = TIFFScanlineSize(tif);
        for (std::uint32_t row = 0; row < h; ++row) {
            std::uint8_t* dst = v.bytes.data() + z * slice_bytes + row * row_bytes;
            if (TIFFReadScanline(tif, dst, row) < 0) {
                TIFFClose(tif);
                throw std::runtime_error("TIFF: scanline read failed");
            }
        }
        ++z;
    } while (TIFFReadDirectory(tif));
    TIFFClose(tif);

    // Float inputs must be narrowed to int regardless of try_compress
    // (everything downstream needs integer labels).
    if (try_compress ||
        v.dtype == image::VoxelType::F32 || v.dtype == image::VoxelType::F64) {
        v = image::Volume::compress_to_smallest(std::move(v), /*headroom=*/26);
    }
    return v;
}

void write_tiff3(const std::string& path, const image::Volume& v) {
    paths::create_folder(path);
    TIFF* tif = TIFFOpen(path.c_str(), "w");
    if (!tif) throw std::runtime_error("TIFF write open failed: " + path);

    const std::uint32_t w = static_cast<std::uint32_t>(v.shape[2]);
    const std::uint32_t h = static_cast<std::uint32_t>(v.shape[1]);
    const std::uint32_t nz = static_cast<std::uint32_t>(v.shape[0]);
    const std::uint16_t bps = static_cast<std::uint16_t>(v.item_size() * 8);
    const std::size_t slice_bytes = static_cast<std::size_t>(w) * h * (bps / 8);

    for (std::uint32_t z = 0; z < nz; ++z) {
        TIFFSetField(tif, TIFFTAG_IMAGEWIDTH,      w);
        TIFFSetField(tif, TIFFTAG_IMAGELENGTH,     h);
        TIFFSetField(tif, TIFFTAG_BITSPERSAMPLE,   bps);
        TIFFSetField(tif, TIFFTAG_SAMPLESPERPIXEL, 1);
        TIFFSetField(tif, TIFFTAG_ORIENTATION,     ORIENTATION_TOPLEFT);
        TIFFSetField(tif, TIFFTAG_PLANARCONFIG,    PLANARCONFIG_CONTIG);
        TIFFSetField(tif, TIFFTAG_SAMPLEFORMAT,    SAMPLEFORMAT_UINT);
        TIFFSetField(tif, TIFFTAG_PHOTOMETRIC,     PHOTOMETRIC_MINISBLACK);
        TIFFSetField(tif, TIFFTAG_ROWSPERSTRIP,    TIFFDefaultStripSize(tif, 0));
        TIFFSetField(tif, TIFFTAG_SUBFILETYPE,     (nz > 1) ? FILETYPE_PAGE : 0);
        if (nz > 1) {
            TIFFSetField(tif, TIFFTAG_PAGENUMBER, z, nz);
        }

        const std::uint8_t* src_slice = v.bytes.data() + z * slice_bytes;
        for (std::uint32_t row = 0; row < h; ++row) {
            const std::uint8_t* row_ptr = src_slice + row * (slice_bytes / h);
            if (TIFFWriteScanline(tif, const_cast<std::uint8_t*>(row_ptr),
                                  row, 0) < 0) {
                TIFFClose(tif);
                throw std::runtime_error("TIFF: scanline write failed");
            }
        }
        TIFFWriteDirectory(tif);
    }
    TIFFClose(tif);
}

image::Volume read_image(const std::string& path,
                         std::optional<std::array<std::size_t, 3>> raw_shape,
                         bool try_compress) {
    std::string ext = paths::file_extension(path);
    std::transform(ext.begin(), ext.end(), ext.begin(),
                   [](unsigned char c) { return std::tolower(c); });

    if (ext == ".tif" || ext == ".tiff") {
        return read_tiff3(path, try_compress);
    }
    if (ext == ".npy") {
        return read_npy3(path);
    }
    if (ext == ".raw" || ext.empty()) {
        if (!raw_shape) {
            throw std::runtime_error("read_image: RAW path requires raw_shape");
        }
        // Choose dtype from filesize / voxel-count, like the reference code.
        std::size_t nbytes = fs::file_size(path);
        std::size_t nv = (*raw_shape)[0] * (*raw_shape)[1] * (*raw_shape)[2];
        std::size_t per = nbytes / nv;
        image::VoxelType dt = image::VoxelType::U8;
        if      (per == 1) dt = image::VoxelType::U8;
        else if (per == 2) dt = image::VoxelType::U16;
        else if (per == 4) dt = image::VoxelType::U32;
        else               dt = image::VoxelType::U64;
        return read_raw3(path, *raw_shape, dt);
    }
    throw std::runtime_error("read_image: unknown extension '" + ext + "'");
}

}  // namespace vox2tet::io
