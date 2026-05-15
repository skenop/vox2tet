// Numeric comparison for paired NPY vertex / vector arrays.
//
//   diff_xyz <a.npy> <b.npy>
//
// Both must be float64 with shape (N, 3) or (N,). Reports max / mean L2
// distance between corresponding rows. Assumes the row ordering already
// matches (no permutation search).

#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

#include "vox2tet/io/npy.hpp"

namespace v = vox2tet;

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "usage: diff_xyz <a.npy> <b.npy>\n";
        return 2;
    }
    std::vector<std::size_t> sa, sb;
    auto a = v::npy::read<double>(argv[1], sa);
    auto b = v::npy::read<double>(argv[2], sb);
    if (sa != sb) {
        std::cerr << "shape mismatch:";
        for (auto v : sa) std::cerr << " " << v;
        std::cerr << " vs";
        for (auto v : sb) std::cerr << " " << v;
        std::cerr << "\n";
        return 1;
    }
    std::size_t cols = sa.size() == 2 ? sa[1] : 1;
    std::size_t rows = sa[0];

    double max_d = 0, sum_d = 0;
    std::size_t i_worst = 0;
    for (std::size_t i = 0; i < rows; ++i) {
        double d2 = 0;
        for (std::size_t k = 0; k < cols; ++k) {
            const double x = a[i * cols + k] - b[i * cols + k];
            d2 += x * x;
        }
        const double d = std::sqrt(d2);
        sum_d += d;
        if (d > max_d) { max_d = d; i_worst = i; }
    }
    std::cout << "rows=" << rows << " cols=" << cols
              << " max_d=" << max_d << " mean_d=" << (sum_d / static_cast<double>(rows))
              << "  worst row=" << i_worst;
    if (max_d > 0 && cols == 3) {
        std::cout << "  a=(" << a[i_worst*3] << ", " << a[i_worst*3+1] << ", " << a[i_worst*3+2]
                  << ")  b=(" << b[i_worst*3] << ", " << b[i_worst*3+1] << ", " << b[i_worst*3+2] << ")";
    }
    std::cout << "\n";
    return 0;
}
