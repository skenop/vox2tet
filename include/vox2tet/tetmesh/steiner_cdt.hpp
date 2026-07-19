#pragma once

// Plain-STL interface to the third-party CDT library (Steiner
// constrained Delaunay tetrahedrization, Diazzi et al. 2023). The
// implementation (src/tetmesh/steiner_cdt.cpp) is the only translation
// unit that includes CDT headers; it is compiled into the separate
// `vox2tet_cdt` static library with the C++20 / strict-FP / SIMD flags
// the predicates require, so none of that leaks into libvox2tet.

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace vox2tet::tetmesh {

struct SteinerCdtResult {
    // Snapped floating-point coordinates, 3 per vertex. The first
    // vertices are the (deduplicated, reordered) input vertices,
    // Steiner vertices follow; positions of input vertices are
    // preserved exactly.
    std::vector<double> xyz;
    // Non-ghost tetrahedra, 4 vertex ids each, 0-based.
    std::vector<std::uint32_t> tets;
    std::uint32_t n_steiner   = 0;  // vertices added by recovery
    bool face_recovery_ok     = true;
    bool snap_ok              = true;  // FP-representability achieved
    std::string error;                 // set when the call returns false
};

// Runs the full Steiner-CDT flow (Delaunay, segment recovery, face
// recovery, FP snap) on a triangle surface. `coords` is 3*n_vertices
// doubles, `tris` 3*n_triangles 0-based vertex ids. The input surface
// (a PLC; non-manifold edges are fine) is recovered exactly as a union
// of tetrahedron faces in the output. Returns false on failure.
bool run_steiner_cdt(const double* coords, std::size_t n_vertices,
                     const std::uint32_t* tris, std::size_t n_triangles,
                     bool verbose, SteinerCdtResult& out);

}  // namespace vox2tet::tetmesh
