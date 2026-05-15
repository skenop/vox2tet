#include "vox2tet/marching_cubes/m3c.hpp"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <mutex>

namespace vox2tet::marching_cubes {

const std::array<std::array<std::uint8_t, 4>, 6>& cell_facet() {
    static const std::array<std::array<std::uint8_t, 4>, 6> v = {{
        {{0, 2, 3, 1}}, {{4, 5, 7, 6}}, {{0, 1, 5, 4}},
        {{2, 6, 7, 3}}, {{0, 4, 6, 2}}, {{1, 3, 7, 5}},
    }};
    return v;
}

const std::array<std::array<std::uint8_t, 6>, 6>& edges_map() {
    static const std::array<std::array<std::uint8_t, 6>, 6> v = {{
        {{ 4,  1,  5,  0, 12, 19}},
        {{ 2,  7,  3,  6, 13, 19}},
        {{ 0,  9,  2,  8, 14, 19}},
        {{10,  3, 11,  1, 15, 19}},
        {{ 8,  6, 10,  4, 16, 19}},
        {{ 5, 11,  7,  9, 17, 19}},
    }};
    return v;
}

const std::array<std::array<std::uint8_t, 4>, 16>& lut2d_2materials() {
    static const std::array<std::array<std::uint8_t, 4>, 16> v = {{
        {{5, 5, 5, 5}}, {{2, 3, 5, 5}}, {{1, 2, 5, 5}}, {{1, 3, 5, 5}},
        {{0, 1, 5, 5}}, {{0, 3, 2, 1}}, {{0, 2, 5, 5}}, {{0, 3, 5, 5}},
        {{3, 0, 5, 5}}, {{2, 0, 5, 5}}, {{1, 0, 3, 2}}, {{1, 0, 5, 5}},
        {{3, 1, 5, 5}}, {{2, 1, 5, 5}}, {{3, 2, 5, 5}}, {{5, 5, 5, 5}},
    }};
    return v;
}

const std::array<std::array<std::int8_t, 3>, 20>& cell_mid_edge_coords() {
    static const std::array<std::array<std::int8_t, 3>, 20> v = {{
        {{ 0, -1,  1}}, {{ 0, -1, -1}}, {{ 0,  1,  1}}, {{ 0,  1, -1}},
        {{-1, -1,  0}}, {{ 1, -1,  0}}, {{-1,  1,  0}}, {{ 1,  1,  0}},
        {{-1,  0,  1}}, {{ 1,  0,  1}}, {{-1,  0, -1}}, {{ 1,  0, -1}},
        {{ 0, -1,  0}}, {{ 0,  1,  0}},
        {{ 0,  0,  1}}, {{ 0,  0, -1}},
        {{-1,  0,  0}}, {{ 1,  0,  0}},
        {{ 0,  0,  0}}, {{ 2,  2,  2}},
    }};
    return v;
}

std::array<std::array<std::uint8_t, 3>, 20> cell_mid_edge_coords1() {
    std::array<std::array<std::uint8_t, 3>, 20> out{};
    const auto& src = cell_mid_edge_coords();
    for (std::size_t i = 0; i < 20; ++i)
        for (int k = 0; k < 3; ++k)
            out[i][k] = static_cast<std::uint8_t>(src[i][k] + 1);
    return out;
}

const std::array<std::array<std::int8_t, 3>, 8>& cell_facet_coords() {
    static const std::array<std::array<std::int8_t, 3>, 8> v = {{
        {{-1, -1,  1}}, {{ 1, -1,  1}}, {{-1, -1, -1}}, {{ 1, -1, -1}},
        {{-1,  1,  1}}, {{ 1,  1,  1}}, {{-1,  1, -1}}, {{ 1,  1, -1}},
    }};
    return v;
}

std::uint8_t opposite_edge(std::uint8_t n) {
    static constexpr std::uint8_t kOpp[20] = {
        3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8,
        19, 19, 19, 19, 19, 19, 19, 19
    };
    return kOpp[n];
}

// ---------------------------------------------------------------------------
// edges_2materials_cell — port of getTriangles2MaterialsCell's first half.
// Walk each face, look up the binary 4-color pattern in lut2d_2materials,
// translate face-local edge ids 0..4 through edges_map[face]. Each face
// contributes a fixed 8 entries (4 directed half-edges × 2 endpoints);
// 19 sentinels mark "no edge".
// ---------------------------------------------------------------------------
std::array<std::uint8_t, 24> edges_2materials_cell(std::uint8_t id) {
    std::array<std::uint8_t, 24> edges{};
    edges.fill(19);

    // Recover the 8 corner labels (0/1) from the 8-bit code.
    std::array<std::uint8_t, 8> corner{};
    for (int i = 0; i < 8; ++i) corner[i] = (id >> (7 - i)) & 1;

    const auto& cf   = cell_facet();
    const auto& em   = edges_map();
    const auto& flut = lut2d_2materials();

    // For each face, look up the 4 face-local edge ids, translate to
    // global MC node ids, and write them in groups of 4 (= 2 half-edges)
    // per face. Slot layout:
    //   face f occupies edges[4f .. 4f+3] as (u0, v0, u1, v1).
    for (int f = 0; f < 6; ++f) {
        const std::uint8_t face_code = static_cast<std::uint8_t>(
            (corner[cf[f][0]] << 3) | (corner[cf[f][1]] << 2) |
            (corner[cf[f][2]] << 1) | (corner[cf[f][3]] << 0));
        const auto& face_edges = flut[face_code];   // 4 face-local edge ids
        for (int k = 0; k < 4; ++k) {
            const std::uint8_t local = face_edges[k];
            edges[f * 4 + k] = (local <= 4) ? em[f][local] : 19;
        }
    }
    return edges;
}

// ---------------------------------------------------------------------------
// edges2loops_no_centered — chain half-edges into closed loops. Mirrors the
// Algorithm: build a `next[u] = v` table, then repeatedly walk
// until we return to the starting node. Centred-facet nodes are NOT
// expected in this variant (caller must guarantee).
// ---------------------------------------------------------------------------
std::vector<std::vector<std::uint8_t>>
edges2loops_no_centered(const std::uint8_t* e, std::size_t n_edges) {
    std::vector<std::vector<std::uint8_t>> loops;

    // Strip dummy edges (where source is 19).
    std::vector<std::pair<std::uint8_t, std::uint8_t>> pairs;
    pairs.reserve(n_edges);
    std::uint8_t max_node = 0;
    for (std::size_t i = 0; i < n_edges; ++i) {
        std::uint8_t u = e[2 * i + 0];
        std::uint8_t v = e[2 * i + 1];
        if (u == 19) continue;
        pairs.emplace_back(u, v);
        max_node = std::max({max_node, u, v});
    }
    if (pairs.empty()) return loops;

    // chain[u] = next node (19 if u not a chain head).
    std::vector<std::uint8_t> chain(static_cast<std::size_t>(max_node) + 1, 19);
    for (auto [u, v] : pairs) chain[u] = v;

    while (true) {
        std::size_t start = chain.size();
        for (std::size_t i = chain.size(); i-- > 0;) {
            if (chain[i] != 19) { start = i; break; }
        }
        if (start == chain.size()) break;

        std::vector<std::uint8_t> loop;
        std::uint8_t id = static_cast<std::uint8_t>(start);
        // Centred nodes (12..18) keep an "open" chain marker, but for the
        // no-centered variant they shouldn't occur — defensive copy still
        // matches the reference branch.
        if (id > 11) loop.push_back(id);
        while (true) {
            const std::uint8_t nxt = chain[id];
            loop.push_back(nxt);
            chain[id] = 19;
            id = nxt;
            if (id >= chain.size() || chain[id] == 19) break;
        }
        loops.push_back(std::move(loop));
    }
    return loops;
}

// ---------------------------------------------------------------------------
// triangulate_loops_no_centered — port of triangulateLoopsNoCentered.
//
// Polylines of length 3..7 are triangulated by a fan with an anchor:
//   3, 4 vertices : anchor = argmin
//   5 vertices    : anchor = one of the two endpoints "opposite" to
//                   another vertex (the reference uses np.argmin among them)
//   6 vertices    : two anchor choices (convex vs non-convex)
//   7 vertices    : a unique "doubly-opposite" anchor
//
// Output is a flat list of node ids (3 per triangle).
// ---------------------------------------------------------------------------
std::vector<std::uint8_t>
triangulate_loops_no_centered(const std::vector<std::vector<std::uint8_t>>& polylines) {
    std::vector<std::uint8_t> tri;
    if (polylines.empty()) {
        tri.insert(tri.end(), {19, 19, 19});
        return tri;
    }

    auto push3 = [&](std::uint8_t a, std::uint8_t b, std::uint8_t c) {
        tri.push_back(a); tri.push_back(b); tri.push_back(c);
    };
    auto argmin_loop = [](const std::vector<std::uint8_t>& p) {
        return static_cast<int>(std::distance(p.begin(),
            std::min_element(p.begin(), p.end())));
    };

    auto find_opposite_indices = [&](const std::vector<std::uint8_t>& p) {
        // indices i in [0, p.size()) where opposite_edge(p[i]) is in p.
        std::vector<int> ip;
        for (std::size_t i = 0; i < p.size(); ++i) {
            const std::uint8_t o = opposite_edge(p[i]);
            for (auto v : p) if (v == o) { ip.push_back(static_cast<int>(i)); break; }
        }
        return ip;
    };

    for (const auto& p : polylines) {
        const int n = static_cast<int>(p.size());
        if (n == 3) {
            const int i0 = argmin_loop(p);
            push3(p[i0], p[(i0+1)%3], p[(i0+2)%3]);
        } else if (n == 4) {
            const int i0 = argmin_loop(p);
            push3(p[i0],         p[(i0+1)%4], p[(i0+2)%4]);
            push3(p[i0],         p[(i0+2)%4], p[(i0+3)%4]);
        } else if (n == 5) {
            const auto ip = find_opposite_indices(p);
            int i0 = 0;
            if (ip.size() >= 2) {
                i0 = (p[ip[0]] < p[ip[1]]) ? ip[0] : ip[1];
            }
            push3(p[i0],         p[(i0+1)%5], p[(i0+2)%5]);
            push3(p[i0],         p[(i0+2)%5], p[(i0+3)%5]);
            push3(p[i0],         p[(i0+3)%5], p[(i0+4)%5]);
        } else if (n == 6) {
            const auto ip = find_opposite_indices(p);
            int i0;
            if (ip.size() == 6) {
                i0 = argmin_loop(p);
            } else if (ip.size() >= 2) {
                i0 = (p[ip[0]] < p[ip[1]]) ? ip[0] : ip[1];
            } else {
                i0 = 0;
            }
            push3(p[i0],         p[(i0+1)%6], p[(i0+2)%6]);
            push3(p[i0],         p[(i0+2)%6], p[(i0+3)%6]);
            push3(p[i0],         p[(i0+3)%6], p[(i0+4)%6]);
            push3(p[i0],         p[(i0+4)%6], p[(i0+5)%6]);
        } else if (n == 7) {
            // One edge has both endpoints opposite-paired — its first
            // endpoint anchors the fan.
            int i0 = 0;
            for (int i = 0; i < 7; ++i) {
                const std::uint8_t o1 = opposite_edge(p[i]);
                const std::uint8_t o2 = opposite_edge(p[(i + 1) % 7]);
                bool has_o1 = false, has_o2 = false;
                for (auto v : p) { if (v == o1) has_o1 = true; if (v == o2) has_o2 = true; }
                if (has_o1 && has_o2) { i0 = i; break; }
            }
            push3(p[i0],         p[(i0+1)%7], p[(i0+4)%7]);
            push3(p[i0],         p[(i0+4)%7], p[(i0+5)%7]);
            push3(p[i0],         p[(i0+5)%7], p[(i0+6)%7]);
            push3(p[(i0+1)%7],   p[(i0+2)%7], p[(i0+3)%7]);
            push3(p[(i0+1)%7],   p[(i0+3)%7], p[(i0+4)%7]);
        } else {
            std::cerr << "ERROR: m3c::triangulate_loops_no_centered: polyline length "
                      << n << "\n";
        }
    }
    return tri;
}

// ---------------------------------------------------------------------------
// lut2materials() — materialise the 256×18 2-material triangle LUT once.
// Mirrors generateLUT2.
// ---------------------------------------------------------------------------
View<std::uint8_t> lut2materials() {
    static std::vector<std::uint8_t> table;
    static std::once_flag once;
    std::call_once(once, [] {
        table.assign(256 * 18, 19);
        for (int id = 0; id < 256; ++id) {
            const auto edges = edges_2materials_cell(static_cast<std::uint8_t>(id));
            const auto loops = edges2loops_no_centered(edges.data(), 12);
            const auto t = triangulate_loops_no_centered(loops);
            std::size_t n = std::min(t.size(), static_cast<std::size_t>(18));
            std::copy_n(t.begin(), n, table.begin() + id * 18);
        }
    });
    return {table.data(), table.size()};
}

}  // namespace vox2tet::marching_cubes
