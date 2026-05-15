// Topological mesh diff: read two pairs of (_xyz.npy, _tr.npy) and check
// whether the unordered set of triangles (each rotated/sorted to a
// canonical form) is identical and whether the vertex coordinate sets are
// identical.
//
//   diff_mesh <a_xyz.npy> <a_tr.npy> <b_xyz.npy> <b_tr.npy>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <set>
#include <string>
#include <vector>

#include "vox2tet/io/npy.hpp"

namespace v = vox2tet;

namespace {

struct Mesh {
    std::vector<std::array<float, 3>>   verts;     // float32 (matches .npy)
    std::vector<std::array<std::uint32_t, 3>> tris;
};

Mesh load(const std::string& xyz_path, const std::string& tri_path) {
    Mesh m;
    std::vector<std::size_t> s;
    auto xyz = v::npy::read<float>(xyz_path, s);
    if (s.size() != 2 || s[1] != 3) throw std::runtime_error("xyz shape wrong: " + xyz_path);
    m.verts.resize(s[0]);
    for (std::size_t i = 0; i < s[0]; ++i)
        m.verts[i] = { xyz[i * 3 + 0], xyz[i * 3 + 1], xyz[i * 3 + 2] };

    auto tr = v::npy::read<std::uint32_t>(tri_path, s);
    if (s.size() != 2 || s[1] != 3) throw std::runtime_error("tri shape wrong: " + tri_path);
    m.tris.resize(s[0]);
    for (std::size_t i = 0; i < s[0]; ++i)
        m.tris[i] = { tr[i * 3 + 0], tr[i * 3 + 1], tr[i * 3 + 2] };
    return m;
}

// Canonicalise a triangle: rotate so smallest vertex is first; orientation
// preserved. (Same triangle written in two equivalent cyclic orders.)
std::array<std::uint32_t, 3> canon(const std::array<std::uint32_t, 3>& t) {
    if (t[0] <= t[1] && t[0] <= t[2]) return t;
    if (t[1] <= t[0] && t[1] <= t[2]) return {t[1], t[2], t[0]};
    return {t[2], t[0], t[1]};
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 5) {
        std::cerr << "usage: diff_mesh <a_xyz.npy> <a_tr.npy> <b_xyz.npy> <b_tr.npy>\n";
        return 2;
    }
    auto A = load(argv[1], argv[2]);
    auto B = load(argv[3], argv[4]);
    std::cout << "A: " << A.verts.size() << " verts, " << A.tris.size() << " tris\n";
    std::cout << "B: " << B.verts.size() << " verts, " << B.tris.size() << " tris\n";

    // Vertex set equality.
    auto vkey = [](const std::array<float, 3>& v) {
        return std::array<float, 3>{v[0], v[1], v[2]};
    };
    std::set<std::array<float, 3>> A_set, B_set;
    for (auto& v : A.verts) A_set.insert(vkey(v));
    for (auto& v : B.verts) B_set.insert(vkey(v));
    std::cout << "vertex-set equal: " << (A_set == B_set ? "YES" : "NO")
              << " (|A|=" << A_set.size() << " |B|=" << B_set.size() << ")\n";

    // Translate each triangle's vertex IDs into the vertex's actual
    // coordinate before comparing — handles different ID orderings.
    auto to_coord_tri = [](const Mesh& m) {
        std::vector<std::array<std::array<float, 3>, 3>> out(m.tris.size());
        for (std::size_t i = 0; i < m.tris.size(); ++i) {
            for (int k = 0; k < 3; ++k) out[i][k] = m.verts[m.tris[i][k]];
        }
        return out;
    };
    auto A_tris = to_coord_tri(A);
    auto B_tris = to_coord_tri(B);

    // Canonicalise each triangle by rotating so the lex-smallest vertex
    // is first.
    auto canon_tri = [](std::array<std::array<float, 3>, 3>& t) {
        int min_i = 0;
        for (int k = 1; k < 3; ++k) if (t[k] < t[min_i]) min_i = k;
        if (min_i == 1) t = {t[1], t[2], t[0]};
        else if (min_i == 2) t = {t[2], t[0], t[1]};
    };
    for (auto& t : A_tris) canon_tri(t);
    for (auto& t : B_tris) canon_tri(t);

    auto cmp = [](const auto& a, const auto& b) {
        if (a[0] != b[0]) return a[0] < b[0];
        if (a[1] != b[1]) return a[1] < b[1];
        return a[2] < b[2];
    };
    std::sort(A_tris.begin(), A_tris.end(), cmp);
    std::sort(B_tris.begin(), B_tris.end(), cmp);

    if (A_tris == B_tris) {
        std::cout << "triangle-set equal: YES\n";
        return 0;
    }
    // Count differences for a hint.
    std::size_t both = 0, only_a = 0, only_b = 0;
    std::size_t i = 0, j = 0;
    while (i < A_tris.size() && j < B_tris.size()) {
        if (A_tris[i] == B_tris[j]) { ++both; ++i; ++j; }
        else if (cmp(A_tris[i], B_tris[j])) { ++only_a; ++i; }
        else { ++only_b; ++j; }
    }
    only_a += A_tris.size() - i;
    only_b += B_tris.size() - j;
    std::cout << "triangle-set equal: NO  (both=" << both
              << "  only_A=" << only_a << "  only_B=" << only_b << ")\n";
    return 1;
}
