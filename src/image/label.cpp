#include "vox2tet/image/label.hpp"

#include "vox2tet/core/log.hpp"

#include <algorithm>
#include <cstdint>
#include <queue>
#include <set>
#include <stdexcept>

namespace vox2tet::image {

namespace {

struct Neighbour { int dz, dy, dx; };

std::vector<Neighbour> make_neighbours(int connectivity) {
    std::vector<Neighbour> ns;
    for (int dz = -1; dz <= 1; ++dz)
    for (int dy = -1; dy <= 1; ++dy)
    for (int dx = -1; dx <= 1; ++dx) {
        if (!dz && !dy && !dx) continue;
        int sq = dz*dz + dy*dy + dx*dx;
        if (sq <= connectivity) ns.push_back({dz, dy, dx});
    }
    return ns;
}

std::uint64_t get_voxel_u64(const Volume& v, std::size_t lin) {
    switch (v.dtype) {
        case VoxelType::U8:  return reinterpret_cast<const std::uint8_t* >(v.bytes.data())[lin];
        case VoxelType::U16: return reinterpret_cast<const std::uint16_t*>(v.bytes.data())[lin];
        case VoxelType::U32: return reinterpret_cast<const std::uint32_t*>(v.bytes.data())[lin];
        case VoxelType::U64: return reinterpret_cast<const std::uint64_t*>(v.bytes.data())[lin];
        case VoxelType::F32:
        case VoxelType::F64: break;  // narrowed on read
    }
    return 0;
}

}  // namespace

LabelResult label_components(const Volume& v, int connectivity) {
    if (connectivity < 1 || connectivity > 3) {
        throw std::invalid_argument("label_components: connectivity must be 1..3");
    }

    LabelResult r;
    r.shape = v.shape;
    const std::size_t nz = v.shape[0];
    const std::size_t ny = v.shape[1];
    const std::size_t nx = v.shape[2];
    const std::size_t N  = nz * ny * nx;
    r.labels.assign(N, -1);

    // Collect sorted unique materials — matches `np.unique(voxels)` order.
    std::set<std::uint64_t> mat_set;
    for (std::size_t i = 0; i < N; ++i) mat_set.insert(get_voxel_u64(v, i));
    std::vector<std::uint64_t> materials(mat_set.begin(), mat_set.end());

    auto ns = make_neighbours(connectivity);
    std::queue<std::size_t> q;
    std::int32_t next_label = 0;

    // For each material in ascending order, raster-scan; assign a fresh
    // label every time we hit a never-visited voxel of that material.
    // This reproduces `scipy.ndimage.label(voxels == m)` numbering when
    // composed by the legacy outer loop.
    for (std::uint64_t m : materials) {
        for (std::size_t z0 = 0; z0 < nz; ++z0)
        for (std::size_t y0 = 0; y0 < ny; ++y0)
        for (std::size_t x0 = 0; x0 < nx; ++x0) {
            const std::size_t seed = (z0 * ny + y0) * nx + x0;
            if (r.labels[seed] != -1) continue;
            if (get_voxel_u64(v, seed) != m) continue;

            const std::int32_t L = next_label++;
            r.materials_out.push_back(m);
            r.labels[seed] = L;
            q.push(seed);

            while (!q.empty()) {
                std::size_t s = q.front(); q.pop();
                std::size_t sz = s / (ny * nx);
                std::size_t sy = (s % (ny * nx)) / nx;
                std::size_t sx = s % nx;
                for (const auto& n : ns) {
                    long zz = static_cast<long>(sz) + n.dz;
                    long yy = static_cast<long>(sy) + n.dy;
                    long xx = static_cast<long>(sx) + n.dx;
                    if (zz < 0 || zz >= static_cast<long>(nz)) continue;
                    if (yy < 0 || yy >= static_cast<long>(ny)) continue;
                    if (xx < 0 || xx >= static_cast<long>(nx)) continue;
                    std::size_t lin = (static_cast<std::size_t>(zz) * ny +
                                       static_cast<std::size_t>(yy)) * nx +
                                       static_cast<std::size_t>(xx);
                    if (r.labels[lin] != -1) continue;
                    if (get_voxel_u64(v, lin) != m) continue;
                    r.labels[lin] = L;
                    q.push(lin);
                }
            }
        }
    }

    r.n_components = next_label;
    return r;
}

}  // namespace vox2tet::image
