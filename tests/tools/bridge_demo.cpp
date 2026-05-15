// Demonstrate the debug bridge: compute EDT in C++, dump (mask,iz,iy,ix)
// as .npy, then shell out via vox2tet::debug_bridge to
// `tools/scipy_bridge.py edt ...` so scipy re-runs the same computation
// and reports any disagreement. Only buildable with
// -DVOX2TET_DEBUG_BRIDGE=ON.

#include "vox2tet/core/debug_bridge.hpp"
#include "vox2tet/image/edt.hpp"
#include "vox2tet/io/npy.hpp"

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <random>
#include <vector>

int main(int argc, char** argv) {
    if (!vox2tet::debug_bridge::available()) {
        std::fprintf(stderr,
            "bridge_demo: debug bridge is NOT available — rebuild with\n"
            "  -DVOX2TET_DEBUG_BRIDGE=ON -DVOX2TET_PYTHON_BIN=/path/to/python\n");
        return 2;
    }

    // Either supply a mask.npy on the command line, or synthesise a
    // small random mask so the demo works without input.
    std::array<std::size_t, 3> shape{16, 24, 32};
    std::vector<std::uint8_t>  mask;
    namespace fs = std::filesystem;
    fs::path outdir = fs::temp_directory_path() / "vox2tet_bridge_demo";
    fs::create_directories(outdir);
    fs::path mask_path = outdir / "mask.npy";

    if (argc >= 2) {
        std::vector<std::size_t> s;
        auto raw = vox2tet::npy::read<std::uint8_t>(argv[1], s);
        if (s.size() != 3) { std::fprintf(stderr, "need 3D mask\n"); return 1; }
        shape = {s[0], s[1], s[2]};
        mask  = std::move(raw);
        mask_path = argv[1];
    } else {
        const auto N = shape[0] * shape[1] * shape[2];
        mask.assign(N, 1);                              // all "needs refill"
        std::mt19937 rng(42);
        std::uniform_int_distribution<int> pick(0, int(N) - 1);
        for (int i = 0; i < 20; ++i) mask[pick(rng)] = 0;   // 20 random sources
        vox2tet::npy::write<std::uint8_t>(
            mask_path.string(),
            {shape[0], shape[1], shape[2]}, mask.data(), N);
        std::printf("bridge_demo: synthesised mask written to %s\n",
                    mask_path.string().c_str());
    }

    auto edt = vox2tet::image::distance_transform_edt_indices(mask, shape);
    auto iz_path = outdir / "my_iz.npy";
    auto iy_path = outdir / "my_iy.npy";
    auto ix_path = outdir / "my_ix.npy";
    vox2tet::npy::write<std::int32_t>(iz_path.string(),
        {shape[0], shape[1], shape[2]}, edt.iz.data(), edt.iz.size());
    vox2tet::npy::write<std::int32_t>(iy_path.string(),
        {shape[0], shape[1], shape[2]}, edt.iy.data(), edt.iy.size());
    vox2tet::npy::write<std::int32_t>(ix_path.string(),
        {shape[0], shape[1], shape[2]}, edt.ix.data(), edt.ix.size());

    std::printf("bridge_demo: invoking scipy via debug bridge...\n");
    int rc = vox2tet::debug_bridge::run("edt", {
        mask_path.string(),
        iz_path.string(),
        iy_path.string(),
        ix_path.string(),
    });
    std::printf("bridge_demo: bridge exit code = %d\n", rc);
    // rc == 0  → exact match
    // rc == 1  → some disagreement (the script prints how many are
    //            iso-distance tie-breaks vs true mismatches)
    return rc;
}
