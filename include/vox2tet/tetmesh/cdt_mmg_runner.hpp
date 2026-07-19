#pragma once

// Built-in volume mesher: Steiner CDT (exact surface preservation) +
// MMG3D interior quality optimization + material classification.
// In-process alternative to the external-tetgen driver
// (vox2tet/tetgen/tetgen_runner.hpp) — same contract: consumes the
// final remeshed surface, emits `<path_base>.inp` with one element
// set per material.

#include <string>
#include <vector>

#include "vox2tet/core/settings.hpp"
#include "vox2tet/core/types.hpp"
#include "vox2tet/marching_cubes/contouring.hpp"

namespace vox2tet::tetmesh {

// Statistics of one mesh_volume run (exposed for tests/diagnostics).
struct CdtMmgStats {
    std::size_t n_nodes_cdt    = 0;   // after CDT (before MMG)
    std::size_t n_tets_cdt     = 0;
    std::size_t n_steiner      = 0;   // surface Steiner points added
    std::size_t n_iface_faces  = 0;   // constrained faces recovered
    std::size_t n_regions      = 0;   // flood-fill material regions
    std::size_t n_nodes_final  = 0;   // written to the INP
    std::size_t n_tets_final   = 0;
    bool        mmg_ran        = false;
    bool        mmg_ok         = false;
    // Region-classification health: interface area recovered by the
    // constrained faces over the input surface area (should be ~1),
    // and the number of regions with conflicting material votes.
    double      area_ratio     = 0.0;
    std::size_t n_vote_conflicts = 0;
    // The final mesh as written to the INP (filled only when stats are
    // requested — used by tests and diagnostics).
    Coords                    nodes;
    Eigen::MatrixXi           tets;
    std::vector<std::int32_t> mats;   // per-tet material (element set)
};

// Returns true when the INP was written. `stats` is optional.
bool mesh_volume(const Settings& s,
                 const std::string& path_base,
                 const Coords& xyz,
                 const Triangles& tri,
                 const std::vector<marching_cubes::Interface>& itf,
                 CdtMmgStats* stats = nullptr);

}  // namespace vox2tet::tetmesh
