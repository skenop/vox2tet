#pragma once

// Extend a 3D image by one voxel layer on every side, then assign 26
// unique boundary labels (one per face/edge/corner region). Matches
// `image_utils.getVoxelsEx`. The result is the volume used by all later
// stages of the pipeline (marching cubes, critical-connectivity fix).

#include <utility>

#include "vox2tet/image/volume.hpp"

namespace vox2tet::image {

// Returns the extended volume and the original (non-padded) shape.
std::pair<Volume, std::array<std::size_t, 3>> ext_volume_from_image(
    const Volume& img);

// Convenience overload that reads + compresses + extends in one step
// (mirrors `getVoxelsEx`).
std::pair<Volume, std::array<std::size_t, 3>> ext_volume_from_file(
    const std::string& path);

}  // namespace vox2tet::image
