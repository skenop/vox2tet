#pragma once

// Exact 3D Euclidean Distance Transform with nearest-source indices.
// Equivalent to `scipy.ndimage.distance_transform_edt(mask,
// return_distances=False, return_indices=True)` with `mask` being True
// where there is NO source (target voxels to refill) and False at source
// voxels.
//
// Algorithm: Felzenszwalb-Huttenlocher 2004 separable 1D EDT applied
// along X then Y then Z, propagating the per-axis "winner" index so the
// final triple (iz[p], iy[p], ix[p]) gives the coordinates of the source
// voxel closest to p in Euclidean distance.
//
// For voxels where mask[p] == 0 (already a source), the result self-
// references (ix=x, iy=y, iz=z).

#include <array>
#include <cstdint>
#include <vector>

namespace vox2tet::image {

struct EdtIndices {
    std::array<std::size_t, 3>  shape;        // {nz, ny, nx}
    std::vector<std::int32_t>   iz, iy, ix;   // size = nz*ny*nx
};

EdtIndices distance_transform_edt_indices(const std::vector<std::uint8_t>& mask,
                                          std::array<std::size_t, 3> shape);

}  // namespace vox2tet::image
