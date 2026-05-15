#pragma once

// Debug-only bridge to a scipy/numpy reference. Compiled in only when
// the project is built with VOX2TET_DEBUG_BRIDGE=ON (CMake option) AND a
// CMake CACHE entry VOX2TET_PYTHON_BIN points at a Python interpreter
// that has numpy + scipy installed.
//
// Architecturally we keep scipy / numpy out of the runtime library:
//   * the main library is pure C++ — no Python dependency.
//   * `debug_bridge` shells out to `tools/scipy_bridge.py`, passing
//     paths to .npy files written by the C++ pipeline. The bridge
//     loads them, re-runs the same operation with scipy, and prints a
//     numeric diff to stdout.
//
// Used for QA only — never invoked in release builds.

#include <string>
#include <vector>

namespace vox2tet::debug_bridge {

#ifdef VOX2TET_DEBUG_BRIDGE

// Returns true if the bridge is available (the embedded path to a
// Python interpreter with scipy/numpy was discovered at build time).
bool available();

// Invoke `tools/scipy_bridge.py <sub> args...`. Returns the bridge's
// exit code (0 on success).
int run(const std::string& sub_command,
        const std::vector<std::string>& args);

#else   // bridge disabled — provide no-op stubs so callers compile
inline bool available() { return false; }
inline int  run(const std::string&, const std::vector<std::string>&) { return -1; }
#endif

}  // namespace vox2tet::debug_bridge
