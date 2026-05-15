#include "vox2tet/core/debug_bridge.hpp"

#ifdef VOX2TET_DEBUG_BRIDGE

#include "vox2tet/core/log.hpp"

#include <cstdlib>
#include <filesystem>
#include <sstream>

namespace vox2tet::debug_bridge {

namespace {

// CMake -DVOX2TET_DEBUG_PYTHON_BIN=/path/to/python embeds this string.
// Path to the bundled scipy_bridge.py is relative to the source tree.
constexpr const char* kPythonBin   = VOX2TET_DEBUG_PYTHON_BIN;
constexpr const char* kBridgeScript = VOX2TET_DEBUG_BRIDGE_SCRIPT;

}  // namespace

bool available() {
    namespace fs = std::filesystem;
    if (std::string(kPythonBin).empty()) return false;
    return fs::exists(kPythonBin) && fs::exists(kBridgeScript);
}

int run(const std::string& sub_command,
        const std::vector<std::string>& args) {
    if (!available()) {
        VOX2TET_PRINT("debug_bridge: not available (CMake build without VOX2TET_PYTHON_BIN)");
        return -1;
    }
    std::ostringstream cmd;
    cmd << kPythonBin << " " << kBridgeScript << " " << sub_command;
    for (const auto& a : args) cmd << " " << a;
    VOX2TET_LOG() << "debug_bridge: " << cmd.str();
    return std::system(cmd.str().c_str());
}

}  // namespace vox2tet::debug_bridge

#endif  // VOX2TET_DEBUG_BRIDGE
