#pragma once

// 3D connected-components labelling, equivalent to
// `scipy.ndimage.label(voxels == m, structuring_element)`. We iterate
// per material id and return:
//
//   * a per-voxel int32 label image (0-based; max label = ncomponents-1).
//   * a flat `materials_out` array: materials_out[label] == original voxel
//     color of that component.
//
// `connectivity` ∈ {1, 2, 3} matches the rank-3 variant of
// `generate_binary_structure(rank, connectivity)`:
//   - 1: 6-neighbourhood (face-adjacent)
//   - 2: 18-neighbourhood (face + edge)
//   - 3: 26-neighbourhood (face + edge + vertex)

#include <cstdint>
#include <vector>

#include "vox2tet/image/volume.hpp"

namespace vox2tet::image {

struct LabelResult {
    std::vector<std::int32_t> labels;        // size = nz*ny*nx, 0-based
    std::array<std::size_t, 3> shape{};
    std::vector<std::uint64_t> materials_out; // size = n_components; the
                                              // input color for each label
    std::int32_t n_components = 0;
};

LabelResult label_components(const Volume& vox, int connectivity = 3);

}  // namespace vox2tet::image
