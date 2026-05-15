// diff_smesh — canonicalising comparison of two TetGen .smesh files.
//
// Two .smesh files describe the same TetGen input mesh iff:
//   1. The set of (x, y, z) node coordinates is the same.
//   2. The set of (sorted-vertex-triple, interface_marker) triangles is
//      the same, when each triangle's vertex IDs are mapped through the
//      canonical node permutation.
//
// To check this without being thrown off by node-ordering / triangle-
// storage-order differences (the bulk of the divergence between
// Python's introsort + scipy.cKDTree pipeline and our C++ port), we
// canonicalise both inputs:
//
//   * Sort node coordinates lexicographically; build a remap
//       old_node → rank_in_sorted.
//   * Remap each triangle's vertices through that remap, then rotate
//     the triangle so its smallest vertex is first (preserves winding).
//   * Group triangles by interface_marker, then sort within each group
//     by (v0, v1, v2).
//
// Output:
//   * node set diff (count of common / only-A / only-B coordinates;
//     L2 distance from each "only-A" node to its nearest in B if any).
//   * marker (interface) sets diff.
//   * canonical-triangle set diff.
//
// Usage:
//   diff_smesh <a.smesh> <b.smesh> [--coord-tol 1e-9]
//
// This tool is build-time always available (it lives under tests/),
// but is conceptually a debug/QA helper — not exercised by ctest.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace {

struct Smesh {
    std::vector<std::array<double, 3>>    nodes;
    std::vector<std::array<std::int64_t, 3>> tris;   // vertex IDs (0-based after read)
    std::vector<std::int64_t>             markers;
    std::string                           source_path;
};

std::string next_data_line(std::ifstream& f) {
    std::string s;
    while (std::getline(f, s)) {
        std::size_t i = 0;
        while (i < s.size() && std::isspace(static_cast<unsigned char>(s[i]))) ++i;
        if (i >= s.size() || s[i] == '#') continue;
        return s;
    }
    return {};
}

Smesh read_smesh(const std::string& path) {
    Smesh r; r.source_path = path;
    std::ifstream f(path);
    if (!f) {
        std::cerr << "FATAL: cannot open " << path << "\n";
        std::exit(2);
    }
    // Node block: header "n 3 attr marker", then n lines "id x y z [attr] [marker]"
    auto hdr = next_data_line(f);
    std::istringstream hs(hdr);
    std::size_t n_nodes = 0;
    hs >> n_nodes;
    r.nodes.reserve(n_nodes);
    // Determine node-id firstnumber on the fly (we honour 0- or 1-based).
    std::int64_t first_node_id = -1;
    std::vector<std::array<double, 3>> by_input_order;
    by_input_order.reserve(n_nodes);
    std::vector<std::int64_t> input_ids;
    input_ids.reserve(n_nodes);
    for (std::size_t i = 0; i < n_nodes; ++i) {
        auto ln = next_data_line(f);
        std::istringstream ls(ln);
        std::int64_t id; double x, y, z;
        ls >> id >> x >> y >> z;
        if (first_node_id < 0) first_node_id = id;
        input_ids.push_back(id);
        by_input_order.push_back({x, y, z});
    }
    // Reorder by id so nodes[i] is the i-th node (0-based) in id order.
    r.nodes.resize(n_nodes);
    for (std::size_t i = 0; i < n_nodes; ++i) {
        const std::int64_t idx = input_ids[i] - first_node_id;
        if (idx < 0 || static_cast<std::size_t>(idx) >= n_nodes) {
            std::cerr << "FATAL: bad node id " << input_ids[i] << " in " << path << "\n";
            std::exit(2);
        }
        r.nodes[static_cast<std::size_t>(idx)] = by_input_order[i];
    }

    // Facet block: header "n_fac boundary_marker", then n_fac lines "k v0 v1 v2 [marker]"
    hdr = next_data_line(f);
    std::istringstream fs(hdr);
    std::size_t n_fac = 0;
    int has_marker = 0;
    fs >> n_fac >> has_marker;
    r.tris.reserve(n_fac);
    r.markers.reserve(n_fac);
    for (std::size_t i = 0; i < n_fac; ++i) {
        auto ln = next_data_line(f);
        std::istringstream ls(ln);
        int k = 0; std::int64_t v0, v1, v2;
        ls >> k >> v0 >> v1 >> v2;
        // Map vertex ids back to 0-based row indices using first_node_id.
        v0 -= first_node_id; v1 -= first_node_id; v2 -= first_node_id;
        r.tris.push_back({v0, v1, v2});
        std::int64_t m = 0;
        if (has_marker) ls >> m;
        r.markers.push_back(m);
    }
    return r;
}

// Canonicalise a triangle by rotating so its lex-smallest vertex is first
// (preserves the winding, so we can still distinguish opposite orientations
// if needed). For set-membership we sort the three components afterwards.
std::array<std::int64_t, 3> rotate_min_first(std::array<std::int64_t, 3> t) {
    int min_i = 0;
    for (int k = 1; k < 3; ++k) if (t[k] < t[min_i]) min_i = k;
    if (min_i == 1) return {t[1], t[2], t[0]};
    if (min_i == 2) return {t[2], t[0], t[1]};
    return t;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "usage: diff_smesh <a.smesh> <b.smesh> [--coord-tol 1e-9]\n";
        return 2;
    }
    double tol = 1e-9;
    for (int i = 3; i < argc; ++i) {
        std::string s = argv[i];
        if (s == "--coord-tol" && i + 1 < argc) tol = std::atof(argv[++i]);
    }

    Smesh A = read_smesh(argv[1]);
    Smesh B = read_smesh(argv[2]);

    std::cout << "\n=== input ===\n";
    std::cout << "  A: " << A.source_path << "   nodes=" << A.nodes.size()
              << "  tris=" << A.tris.size() << "\n";
    std::cout << "  B: " << B.source_path << "   nodes=" << B.nodes.size()
              << "  tris=" << B.tris.size() << "\n";

    // ----- Canonicalise: sort nodes lex --------------------------------
    auto quantise = [&](double x) {
        // Round to nearest multiple of tol so float jitter doesn't make
        // semantically-equal nodes look different.
        return std::llround(x / tol);
    };
    auto qcoord = [&](const std::array<double, 3>& p) {
        return std::array<std::int64_t, 3>{quantise(p[0]), quantise(p[1]), quantise(p[2])};
    };

    auto build_node_remap = [&](const Smesh& m) {
        // Pair (quantised coord, original id), then sort lex to assign
        // canonical 0..n-1 rank to each node.
        std::vector<std::pair<std::array<std::int64_t, 3>, std::int64_t>> pairs;
        pairs.reserve(m.nodes.size());
        for (std::size_t i = 0; i < m.nodes.size(); ++i)
            pairs.push_back({qcoord(m.nodes[i]), static_cast<std::int64_t>(i)});
        std::sort(pairs.begin(), pairs.end(),
            [](const auto& x, const auto& y) { return x.first < y.first; });
        std::vector<std::int64_t> remap(m.nodes.size(), -1);
        for (std::size_t r = 0; r < pairs.size(); ++r)
            remap[static_cast<std::size_t>(pairs[r].second)] = static_cast<std::int64_t>(r);
        return std::make_pair(std::move(remap), std::move(pairs));
    };
    auto [remap_A, sorted_A] = build_node_remap(A);
    auto [remap_B, sorted_B] = build_node_remap(B);

    // ----- Node set diff ----------------------------------------------
    {
        std::set<std::array<std::int64_t, 3>> setA, setB;
        for (auto& p : sorted_A) setA.insert(p.first);
        for (auto& p : sorted_B) setB.insert(p.first);
        std::size_t common = 0, onlyA = 0, onlyB = 0;
        for (auto& c : setA) (setB.count(c) ? common : onlyA)++;
        for (auto& c : setB) if (!setA.count(c)) ++onlyB;
        std::cout << "\n=== canonicalised node-coordinate set (tol=" << tol << ") ===\n";
        std::cout << "  |A|=" << setA.size() << "  |B|=" << setB.size()
                  << "  common=" << common << "  only_A=" << onlyA << "  only_B=" << onlyB << "\n";
    }

    // ----- Marker set diff --------------------------------------------
    {
        std::set<std::int64_t> mA(A.markers.begin(), A.markers.end());
        std::set<std::int64_t> mB(B.markers.begin(), B.markers.end());
        std::size_t common = 0, onlyA = 0, onlyB = 0;
        for (auto& m : mA) (mB.count(m) ? common : onlyA)++;
        for (auto& m : mB) if (!mA.count(m)) ++onlyB;
        std::cout << "\n=== interface markers ===\n";
        std::cout << "  |A|=" << mA.size() << "  |B|=" << mB.size()
                  << "  common=" << common << "  only_A=" << onlyA << "  only_B=" << onlyB << "\n";
    }

    // ----- Triangle set diff (canonicalised) --------------------------
    // Each triangle becomes (marker, sorted-vertex-triple).
    auto canonical_tris = [&](const Smesh& m, const std::vector<std::int64_t>& remap) {
        std::vector<std::array<std::int64_t, 4>> ts;
        ts.reserve(m.tris.size());
        for (std::size_t i = 0; i < m.tris.size(); ++i) {
            std::array<std::int64_t, 3> t = {
                remap[static_cast<std::size_t>(m.tris[i][0])],
                remap[static_cast<std::size_t>(m.tris[i][1])],
                remap[static_cast<std::size_t>(m.tris[i][2])],
            };
            std::sort(t.begin(), t.end());
            ts.push_back({m.markers[i], t[0], t[1], t[2]});
        }
        std::sort(ts.begin(), ts.end());
        return ts;
    };
    auto cA = canonical_tris(A, remap_A);
    auto cB = canonical_tris(B, remap_B);

    // To diff fairly, the two remaps must be consistent — i.e. node
    // i in A must rank to the same position as the equivalent node in
    // B. Build a node-coord -> rank map for B and use it to translate
    // A's triangles through B's index space, then diff that against B.
    std::map<std::array<std::int64_t, 3>, std::int64_t> coord_to_rank_B;
    for (auto& p : sorted_B) {
        coord_to_rank_B[p.first] = remap_B[static_cast<std::size_t>(p.second)];
    }
    std::vector<std::array<std::int64_t, 4>> cAB;
    cAB.reserve(A.tris.size());
    bool any_dropped = false;
    for (std::size_t i = 0; i < A.tris.size(); ++i) {
        std::array<std::int64_t, 3> t;
        for (int k = 0; k < 3; ++k) {
            auto q = qcoord(A.nodes[static_cast<std::size_t>(A.tris[i][k])]);
            auto it = coord_to_rank_B.find(q);
            if (it == coord_to_rank_B.end()) { t[k] = -1; any_dropped = true; }
            else                                t[k] = it->second;
        }
        if (t[0] < 0 || t[1] < 0 || t[2] < 0) {
            cAB.push_back({A.markers[i], -1, -1, -1});  // unresolved triangle
            continue;
        }
        std::sort(t.begin(), t.end());
        cAB.push_back({A.markers[i], t[0], t[1], t[2]});
    }
    std::sort(cAB.begin(), cAB.end());

    // Now both cAB and cB index nodes in B's canonical space.
    std::size_t both = 0, onlyA = 0, onlyB = 0;
    {
        std::set<std::array<std::int64_t, 4>> sA(cAB.begin(), cAB.end()),
                                              sB(cB.begin(), cB.end());
        for (auto& t : sA) (sB.count(t) ? both : onlyA)++;
        for (auto& t : sB) if (!sA.count(t)) ++onlyB;
    }
    std::cout << "\n=== canonicalised triangle set (sorted-vertex + marker) ===\n";
    std::cout << "  A∩B=" << both << "  only_A=" << onlyA
              << "  only_B=" << onlyB
              << "  A had " << (any_dropped ? "some nodes outside B's coord set" : "all nodes resolvable") << "\n";

    // ----- Per-marker breakdown of mismatches --------------------------
    std::map<std::int64_t, std::array<std::size_t, 3>> per_marker;  // {only_A, only_B, both}
    {
        std::set<std::array<std::int64_t, 4>> sA(cAB.begin(), cAB.end()),
                                              sB(cB.begin(), cB.end());
        for (auto& t : sA) per_marker[t[0]][sB.count(t) ? 2 : 0]++;
        for (auto& t : sB) if (!sA.count(t)) per_marker[t[0]][1]++;
    }
    std::size_t markers_with_diff = 0;
    for (auto& [m, c] : per_marker) {
        if (c[0] + c[1] > 0) ++markers_with_diff;
    }
    std::cout << "  markers with any mismatch: " << markers_with_diff
              << " (of " << per_marker.size() << " unique markers in A∪B)\n";

    // ----- Summary verdict --------------------------------------------
    std::cout << "\n=== verdict ===\n";
    if (both == cB.size() && both == cAB.size() && onlyA == 0 && onlyB == 0) {
        std::cout << "  meshes are TOPOLOGICALLY IDENTICAL after canonicalisation.\n";
        return 0;
    }
    std::cout << "  meshes DIFFER even after canonicalisation:\n";
    std::cout << "    " << onlyA << " triangles only in A\n";
    std::cout << "    " << onlyB << " triangles only in B\n";
    std::cout << "    " << markers_with_diff << " interface markers affected\n";
    return 1;
}
