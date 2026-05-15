#pragma once

// Image preparation step. This module implements:
//   * removeSmallRegions  : drop CC's of size <= max_remove_size, refill
//                           via Euclidean distance transform.
//   * fixCriticalConnectivity : find and resolve vertex/edge ambiguous
//                               voxel patterns that would break marching
//                               cubes topology.
//
// Both operate on the **extended** volume (image::ext_volume_from_image
// result).

#include <string>
#include <utility>

#include "vox2tet/core/settings.hpp"
#include "vox2tet/image/volume.hpp"

namespace vox2tet::image {

struct RemoveSmallResult {
    Volume      voxels;
    std::string img_file_path;   // *_C{c}_R{r}.tif
};

// Removes connected components <= max_remove_size; fills them by nearest-
// neighbour Euclidean distance transform. Persists three TIFF files +
// label-mapping text files alongside `out_path_base`, mirroring the reference.
RemoveSmallResult remove_small_regions(const Settings& s, const Volume& voxels);

// Fixes vertex/edge critical connectivities in-place on `voxels`. Returns
// the path of the resulting TIFF if any fix was applied; otherwise the
// original `in_path` is returned unchanged.
struct FixCritResult {
    Volume      voxels;          // EXTENDED volume after fix
    std::string out_file_path;
};
FixCritResult fix_critical_connectivity(const std::string& in_path,
                                        const std::string& out_path_base);

// TODO[continuation]: the lower-level helpers
//   - findVertexConnectivity
//   - findEdgeConnectivity
//   - fixVertexCriticalConnectivity
//   - fixEdgeCriticalConnectivity2M
// must be ported faithfully. They operate on the extended volume and use
// the bitwise pattern encoding documented in image_utils.py:255-265.

}  // namespace vox2tet::image
