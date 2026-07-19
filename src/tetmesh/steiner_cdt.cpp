// The only TU that includes CDT headers — compiled into `vox2tet_cdt`
// (C++20, -frounding-math, SIMD flags). Mirrors the library's own
// createSteinerCDT() flow (see third-party/CDT/src/main.cpp), minus
// file IO, plus in-memory extraction of the snapped result.

#include "vox2tet/tetmesh/steiner_cdt.hpp"

#include "delaunay.h"
#include "inputPLC.h"
#include "PLC.h"

#include <memory>

namespace vox2tet::tetmesh {

bool run_steiner_cdt(const double* coords, std::size_t n_vertices,
                     const std::uint32_t* tris, std::size_t n_triangles,
                     bool verbose, SteinerCdtResult& out) {
    out = SteinerCdtResult{};
    if (n_vertices == 0 || n_triangles == 0) {
        out.error = "empty input";
        return false;
    }

    initFPU();  // no-op on 64-bit builds, required on 32-bit x87

    try {
        // initFromVectors deduplicates vertices, drops degenerate
        // triangles and reorders vertices for locality; it copies from
        // the buffers we hand it (which it may permute in place, hence
        // the local copies).
        std::vector<double> c(coords, coords + 3 * n_vertices);
        std::vector<std::uint32_t> t(tris, tris + 3 * n_triangles);
        inputPLC plc;
        plc.initFromVectors(c.data(), (uint32_t)n_vertices, t.data(),
                            (uint32_t)n_triangles, verbose);

        auto tin = std::make_unique<TetMesh>();
        tin->init_vertices(plc.coordinates.data(), plc.numVertices());
        tin->tetrahedrize();

        PLCx steiner_plc(*tin, plc.triangle_vertices.data(),
                         plc.numTriangles());
        steiner_plc.segmentRecovery_HSi(/*quiet=*/!verbose);
        out.face_recovery_ok = steiner_plc.faceRecovery(/*quiet=*/!verbose);

        // Make every coordinate exactly representable in double
        // precision (Steiner points are implicit/rational otherwise).
        // May collapse zero-length edges and swap near-degenerate
        // tets; on failure some tets stay flat/inverted after
        // rounding — the caller decides how much to trust the result.
        out.snap_ok = tin->optimizeNearDegenerateTets(verbose);

        const uint32_t nv = tin->numVertices();
        out.xyz.resize(3 * (std::size_t)nv);
        for (uint32_t i = 0; i < nv; ++i) {
            double x, y, z;
            tin->vertices[i]->getApproxXYZCoordinates(x, y, z, true);
            out.xyz[3 * (std::size_t)i]     = x;
            out.xyz[3 * (std::size_t)i + 1] = y;
            out.xyz[3 * (std::size_t)i + 2] = z;
        }
        out.n_steiner = (nv > plc.numVertices()) ? nv - plc.numVertices() : 0;

        out.tets.reserve(4 * (std::size_t)tin->countNonGhostTets());
        for (uint64_t i = 0; i < tin->numTets(); ++i) {
            if (tin->isGhost(i)) continue;
            const uint32_t* n = tin->getTetNodes(i << 2);
            out.tets.insert(out.tets.end(), {n[0], n[1], n[2], n[3]});
        }
        return true;
    } catch (const std::exception& e) {
        out.error = e.what();
        return false;
    } catch (...) {
        out.error = "unknown CDT failure";
        return false;
    }
}

}  // namespace vox2tet::tetmesh
