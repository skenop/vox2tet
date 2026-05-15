#include "vox2tet/image/edt.hpp"

#include <cstdint>
#include <limits>
#include <vector>

namespace vox2tet::image {

namespace {

// "infinity" sentinel small enough not to overflow when added to (n-1)^2
// for any 3D image we care about (n^2 << 2^53).
constexpr double kInf = 1e18;

// 1D Felzenszwalb-Huttenlocher EDT.
//
//   D[q]      = min_{q'} (q-q')^2 + f[q']
//   winner[q] = the q' that attains the min
//
// `f` may carry +kInf at non-sources; sources have f = 0 (initial pass) or
// the squared distance accumulated from previous-axis passes.
void felzenszwalb_1d(int n,
                     const double* f,
                     double* D,
                     std::int32_t* winner,
                     std::vector<int>& v,
                     std::vector<double>& z) {
    v.assign(n, 0);
    z.assign(n + 1, 0.0);

    int k = 0;
    v[0] = 0;
    z[0] = -kInf;
    z[1] = +kInf;

    for (int q = 1; q < n; ++q) {
        double s = 0.0;
        while (true) {
            const int    vk  = v[k];
            const double num = (f[q] + static_cast<double>(q) * q) -
                               (f[vk] + static_cast<double>(vk) * vk);
            const double den = 2.0 * static_cast<double>(q - vk);
            // den is always > 0 because q > vk.
            s = num / den;
            if (s <= z[k] && k > 0) {
                --k;
                continue;
            }
            break;
        }
        ++k;
        v[k]   = q;
        z[k]   = s;
        z[k+1] = +kInf;
    }

    k = 0;
    for (int q = 0; q < n; ++q) {
        while (z[k+1] < q) ++k;
        const int    vk = v[k];
        const double d  = static_cast<double>(q - vk);
        D[q]      = d * d + f[vk];
        winner[q] = vk;
    }
}

}  // namespace

EdtIndices distance_transform_edt_indices(const std::vector<std::uint8_t>& mask,
                                          std::array<std::size_t, 3> shape) {
    const std::size_t nz = shape[0];
    const std::size_t ny = shape[1];
    const std::size_t nx = shape[2];
    const std::size_t N  = nz * ny * nx;

    EdtIndices R;
    R.shape = shape;
    R.iz.assign(N, 0);
    R.iy.assign(N, 0);
    R.ix.assign(N, 0);

    std::vector<double> D1(N), D2(N);
    std::vector<std::int32_t> idxX1(N), idxX2(N), idxY2(N);

    // --- Pass along X ------------------------------------------------------
    {
        std::vector<int>    v_buf;
        std::vector<double> z_buf;
        std::vector<double> f_row(nx);
        std::vector<double> D_row(nx);
        std::vector<std::int32_t> w_row(nx);
        for (std::size_t z = 0; z < nz; ++z)
        for (std::size_t y = 0; y < ny; ++y) {
            const std::size_t base = (z * ny + y) * nx;
            for (std::size_t x = 0; x < nx; ++x)
                f_row[x] = mask[base + x] ? kInf : 0.0;
            felzenszwalb_1d(static_cast<int>(nx), f_row.data(), D_row.data(),
                            w_row.data(), v_buf, z_buf);
            for (std::size_t x = 0; x < nx; ++x) {
                D1[base + x]    = D_row[x];
                idxX1[base + x] = w_row[x];
            }
        }
    }

    // --- Pass along Y ------------------------------------------------------
    {
        std::vector<int>    v_buf;
        std::vector<double> z_buf;
        std::vector<double> f_col(ny);
        std::vector<double> D_col(ny);
        std::vector<std::int32_t> w_col(ny);
        for (std::size_t z = 0; z < nz; ++z)
        for (std::size_t x = 0; x < nx; ++x) {
            for (std::size_t y = 0; y < ny; ++y)
                f_col[y] = D1[(z * ny + y) * nx + x];
            felzenszwalb_1d(static_cast<int>(ny), f_col.data(), D_col.data(),
                            w_col.data(), v_buf, z_buf);
            for (std::size_t y = 0; y < ny; ++y) {
                const std::size_t lin = (z * ny + y) * nx + x;
                D2[lin]    = D_col[y];
                idxY2[lin] = w_col[y];
                // Propagate X-index from the winning column.
                const std::size_t lin_win = (z * ny + w_col[y]) * nx + x;
                idxX2[lin] = idxX1[lin_win];
            }
        }
    }

    // --- Pass along Z ------------------------------------------------------
    {
        std::vector<int>    v_buf;
        std::vector<double> z_buf;
        std::vector<double> f_col(nz);
        std::vector<double> D_col(nz);
        std::vector<std::int32_t> w_col(nz);
        for (std::size_t y = 0; y < ny; ++y)
        for (std::size_t x = 0; x < nx; ++x) {
            for (std::size_t z = 0; z < nz; ++z)
                f_col[z] = D2[(z * ny + y) * nx + x];
            felzenszwalb_1d(static_cast<int>(nz), f_col.data(), D_col.data(),
                            w_col.data(), v_buf, z_buf);
            for (std::size_t z = 0; z < nz; ++z) {
                const std::size_t lin     = (z * ny + y) * nx + x;
                const std::size_t lin_win = (static_cast<std::size_t>(w_col[z]) * ny + y) * nx + x;
                R.iz[lin] = w_col[z];
                R.iy[lin] = idxY2[lin_win];
                R.ix[lin] = idxX2[lin_win];
            }
        }
    }

    return R;
}

}  // namespace vox2tet::image
