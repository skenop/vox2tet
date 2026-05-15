#pragma once

// Image I/O for the formats vox2tet consumes:
//   - 3D TIFF stack (read & write)  via libtiff
//   - .npy 3D integer arrays        via vox2tet::npy
//   - .raw header-less binary       via shape argument

#include <array>
#include <optional>
#include <string>

#include "vox2tet/image/volume.hpp"

namespace vox2tet::io {

// Auto-detect the format from `path`'s extension (".tif", ".tiff", ".npy",
// ".raw", or empty). When the format is RAW, `raw_shape` must be set.
image::Volume read_image(const std::string& path,
                         std::optional<std::array<std::size_t, 3>> raw_shape = std::nullopt,
                         bool try_compress = true);

image::Volume read_tiff3(const std::string& path, bool try_compress = true);
image::Volume read_raw3(const std::string& path,
                        std::array<std::size_t, 3> shape,
                        image::VoxelType dtype);
image::Volume read_npy3(const std::string& path);

// Multi-page TIFF write — one IFD per Z-slice. Page format is single-channel
// integer at the volume's native dtype.
void write_tiff3(const std::string& path, const image::Volume& v);

}  // namespace vox2tet::io
