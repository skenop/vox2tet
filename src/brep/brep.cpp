#include "vox2tet/brep/brep.hpp"

#include "vox2tet/core/log.hpp"
#include "vox2tet/io/mesh_io.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <queue>
#include <stdexcept>
#include <vector>

namespace vox2tet::brep {

// ---------------------------------------------------------------------------
// get_boundary_edges — port of getBoundaryEdges.
//
// 1. Pull out triangles where at least one vertex is in masks[1]|masks[2].
// 2. Emit the 3 directed edges (v0,v1), (v1,v2), (v2,v0).
// 3. Keep edges with BOTH endpoints in masks[1]|masks[2].
// 4. Deduplicate (we want each undirected edge once, with v0 < v1).
// ---------------------------------------------------------------------------
BEdges get_boundary_edges(const Triangles& tri,
                          const marching_cubes::NodeTypeMask& is_bnode) {
    const auto& m1 = is_bnode.masks[1];
    const auto& m2 = is_bnode.masks[2];
    auto is_b = [&](std::uint32_t v) -> bool {
        return v < m1.size() && (m1[v] != 0 || m2[v] != 0);
    };

    std::vector<std::array<std::uint32_t, 2>> raw;
    raw.reserve(static_cast<std::size_t>(tri.rows() * 3));
    for (Eigen::Index t = 0; t < tri.rows(); ++t) {
        const std::uint32_t a = tri(t, 0), b = tri(t, 1), c = tri(t, 2);
        const bool ba = is_b(a), bb = is_b(b), bc = is_b(c);
        if (!(ba || bb || bc)) continue;
        if (ba && bb) raw.push_back({std::min(a, b), std::max(a, b)});
        if (bb && bc) raw.push_back({std::min(b, c), std::max(b, c)});
        if (bc && ba) raw.push_back({std::min(c, a), std::max(c, a)});
    }

    std::sort(raw.begin(), raw.end());
    raw.erase(std::unique(raw.begin(), raw.end()), raw.end());

    BEdges out(static_cast<Eigen::Index>(raw.size()), 2);
    for (std::size_t i = 0; i < raw.size(); ++i) {
        out(static_cast<Eigen::Index>(i), 0) = raw[i][0];
        out(static_cast<Eigen::Index>(i), 1) = raw[i][1];
    }
    return out;
}

// ---------------------------------------------------------------------------
// order_bedges — connected-component walk over the boundary-edge graph,
// using the legacy fixed-vertex-splitting trick: each occurrence of a fixed
// vertex (masks[2]) in an edge becomes a unique sentinel ID. This breaks
// "junction" topology so every connected component is either a path
// (with sentinel endpoints) or a closed loop.
//
// After path extraction, sentinel IDs are mapped back to their original
// fixed-vertex IDs via the `sentinel_to_original` table.
// ---------------------------------------------------------------------------
BrepOrdered order_bedges(const BEdges& bedges,
                         const marching_cubes::NodeTypeMask& is_bnode) {
    VOX2TET_PRINT("Ordering B edges...");
    BrepOrdered out;
    if (bedges.rows() == 0) return out;

    const auto& fixed_mask = is_bnode.masks[2];
    auto is_fixed = [&](std::uint32_t v) -> bool {
        return v < fixed_mask.size() && fixed_mask[v] != 0;
    };

    // Find maxnnode1 = max vertex id + 1 — base for sentinel allocation.
    std::uint32_t max_v = 0;
    for (Eigen::Index i = 0; i < bedges.rows(); ++i)
        max_v = std::max({max_v, bedges(i, 0), bedges(i, 1)});
    const std::uint32_t sentinel_base = max_v + 1;

    // Allocate a fresh sentinel per fixed-vertex slot in `bedges`.
    // `edges_split[i]` holds the post-substitution endpoints.
    std::vector<std::array<std::uint32_t, 2>> edges_split(static_cast<std::size_t>(bedges.rows()));
    std::vector<std::uint32_t> sentinel_to_original;
    sentinel_to_original.reserve(static_cast<std::size_t>(bedges.rows() * 2));
    for (Eigen::Index i = 0; i < bedges.rows(); ++i) {
        for (int k = 0; k < 2; ++k) {
            const std::uint32_t v = bedges(i, k);
            if (is_fixed(v)) {
                edges_split[static_cast<std::size_t>(i)][k] =
                    sentinel_base + static_cast<std::uint32_t>(sentinel_to_original.size());
                sentinel_to_original.push_back(v);
            } else {
                edges_split[static_cast<std::size_t>(i)][k] = v;
            }
        }
    }

    // Helper to recover the original vertex id from a (possibly sentinel) id.
    auto unsent = [&](std::uint32_t v) -> std::uint32_t {
        if (v >= sentinel_base) return sentinel_to_original[v - sentinel_base];
        return v;
    };

    const std::uint32_t n_nodes = sentinel_base + static_cast<std::uint32_t>(sentinel_to_original.size());

    // Build sorted adjacency lists. Sentinels have degree 1 by construction.
    std::vector<std::vector<std::uint32_t>> adj(n_nodes);
    for (const auto& e : edges_split) {
        adj[e[0]].push_back(e[1]);
        adj[e[1]].push_back(e[0]);
    }
    for (auto& v : adj) std::sort(v.begin(), v.end());

    // BFS shortest path within a given component mask. Returns empty on
    // unreachable.
    auto bfs_path = [&](std::uint32_t u, std::uint32_t v,
                        const std::vector<std::uint8_t>& mask)
        -> std::vector<std::uint32_t>
    {
        std::vector<std::int64_t> parent(adj.size(), -1);
        std::vector<std::uint8_t> seen(adj.size(), 0);
        std::queue<std::uint32_t> q;
        q.push(u);
        seen[u] = 1;
        while (!q.empty()) {
            std::uint32_t x = q.front(); q.pop();
            if (x == v) break;
            for (std::uint32_t y : adj[x]) {
                if (!mask[y]) continue;
                if (seen[y]) continue;
                seen[y] = 1;
                parent[y] = static_cast<std::int64_t>(x);
                q.push(y);
            }
        }
        std::vector<std::uint32_t> path;
        if (!seen[v]) return path;
        for (std::int64_t cur = v; cur != -1; cur = parent[static_cast<std::size_t>(cur)]) {
            path.push_back(static_cast<std::uint32_t>(cur));
            if (cur == static_cast<std::int64_t>(u)) break;
        }
        std::reverse(path.begin(), path.end());
        return path;
    };

    // Walk components.
    std::vector<std::uint8_t> in_graph(adj.size(), 0);
    for (const auto& e : edges_split) {
        in_graph[e[0]] = 1;
        in_graph[e[1]] = 1;
    }
    std::vector<std::uint8_t> visited(adj.size(), 0);

    for (std::uint32_t seed = 0; seed < adj.size(); ++seed) {
        if (!in_graph[seed] || visited[seed]) continue;

        std::vector<std::uint32_t> comp;
        std::vector<std::uint8_t>  mask(adj.size(), 0);
        std::queue<std::uint32_t>  q;
        q.push(seed); visited[seed] = 1; mask[seed] = 1;
        while (!q.empty()) {
            std::uint32_t x = q.front(); q.pop();
            comp.push_back(x);
            for (std::uint32_t y : adj[x]) {
                if (visited[y]) continue;
                visited[y] = 1; mask[y] = 1;
                q.push(y);
            }
        }
        std::sort(comp.begin(), comp.end());

        // Identify sentinel endpoints in this component.
        std::vector<std::uint32_t> sentinels_in_comp;
        for (auto v : comp) if (v >= sentinel_base) sentinels_in_comp.push_back(v);

        EdgeChain chain;
        if (sentinels_in_comp.size() == 2) {
            chain = bfs_path(sentinels_in_comp[0], sentinels_in_comp[1], mask);
            if (chain.empty()) {
                VOX2TET_LOG() << "ERROR: order_bedges: no path between 2 sentinels";
                continue;
            }
        } else if (sentinels_in_comp.size() == 1) {
            // One sentinel — loop that returns to the sentinel itself.
            // The sentinel has degree 1 (its sole neighbour); walk around
            // and close back to it.
            const std::uint32_t s = sentinels_in_comp[0];
            if (adj[s].empty()) continue;
            const std::uint32_t adj0 = adj[s].front();
            chain.push_back(s);
            std::uint32_t prev = s;
            std::uint32_t curr = adj0;
            while (true) {
                chain.push_back(curr);
                if (curr == s) break;
                std::uint32_t nxt = std::numeric_limits<std::uint32_t>::max();
                for (auto n : adj[curr]) if (n != prev) { nxt = n; break; }
                if (nxt == std::numeric_limits<std::uint32_t>::max()) {
                    VOX2TET_LOG() << "ERROR: order_bedges: dead-end at " << curr;
                    chain.clear();
                    break;
                }
                prev = curr;
                curr = nxt;
                if (chain.size() > comp.size() + 1) {
                    VOX2TET_LOG() << "ERROR: order_bedges: loop didn't close (1 sentinel)";
                    chain.clear();
                    break;
                }
            }
        } else if (sentinels_in_comp.empty()) {
            // Closed loop with no fixed nodes — start at the smallest
            // regular vertex.
            const std::uint32_t f = comp.front();
            if (adj[f].empty()) continue;
            const std::uint32_t adj0 = adj[f].front();
            chain.push_back(f);
            std::uint32_t prev = f;
            std::uint32_t curr = adj0;
            while (true) {
                chain.push_back(curr);
                if (curr == f) break;
                std::uint32_t nxt = std::numeric_limits<std::uint32_t>::max();
                for (auto n : adj[curr]) if (n != prev) { nxt = n; break; }
                if (nxt == std::numeric_limits<std::uint32_t>::max()) {
                    chain.clear();
                    break;
                }
                prev = curr;
                curr = nxt;
                if (chain.size() > comp.size() + 1) { chain.clear(); break; }
            }
        } else {
            VOX2TET_LOG() << "ERROR: order_bedges: " << sentinels_in_comp.size()
                          << " sentinels in component (expected 0..2)";
            continue;
        }

        if (chain.empty()) continue;

        // Map sentinel ids back to the original fixed-vertex ids.
        for (auto& v : chain) v = unsent(v);

        out.brepf.push_back({chain.front(), chain.back()});
        out.brep.push_back(std::move(chain));
    }

    VOX2TET_PRINT("Ordering B edges done!");
    return out;
}

// ---------------------------------------------------------------------------
void save_brep_ply(const BrepOrdered& ordered,
                   const Coords& xyz,
                   const std::string& base,
                   const std::array<std::uint8_t, 3>& rgb) {
    std::size_t total = 0;
    for (const auto& c : ordered.brep) if (c.size() >= 2) total += c.size() - 1;
    Eigen::Matrix<std::uint32_t, Eigen::Dynamic, 2, Eigen::RowMajor> edges(
        static_cast<Eigen::Index>(total), 2);
    Eigen::Index w = 0;
    for (const auto& c : ordered.brep) {
        for (std::size_t i = 0; i + 1 < c.size(); ++i) {
            edges(w, 0) = c[i];
            edges(w, 1) = c[i + 1];
            ++w;
        }
    }
    io::save_edges_ply(base + "_E.ply", xyz, edges, &rgb);
}

}  // namespace vox2tet::brep
