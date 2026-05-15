#pragma once

// TetGen driver — writes the surface to .smesh, calls the `tetgen`
// binary on $PATH, parses its .node/.ele/.face output back into the
// in-memory tetrahedral mesh, then emits an Abaqus INP.

#include <string>
#include <vector>

#include "vox2tet/core/types.hpp"
#include "vox2tet/marching_cubes/contouring.hpp"

namespace vox2tet::tetgen {

// Returns true on success. Standard CLI:
//     tetgen -pYA -q2/15 -o/150 -nn -V <smesh>
//
// TODO[continuation]: parse .node/.ele/.face into Coords + tetra elements
// (Mx4) and translate face attributes (interface ids) to per-element
// material ids via the same volume-sign disambiguation.
bool mesh_volume(const std::string& path_base,
                 const Coords& xyz,
                 const Triangles& tri,
                 const std::vector<marching_cubes::Interface>& itf,
                 bool do_abaqus_verification);

}  // namespace vox2tet::tetgen
