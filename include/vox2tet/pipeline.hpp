#pragma once

// Top-level pipeline orchestrator. `generate(settings)` runs every
// enabled stage (image prep → marching cubes → smoothing → remeshing
// → TetGen → Abaqus export) against the given `Settings`.

#include "vox2tet/core/settings.hpp"

namespace vox2tet {

// Runs the full pipeline against `s`. Stages that are not yet implemented
// in C++ print a clear log line and return early (rather than crash) so
// integration testing of the implemented stages remains possible.
void generate(const Settings& s);

}  // namespace vox2tet
