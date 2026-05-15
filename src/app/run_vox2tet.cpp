// CLI entry point.
//
//   run_vox2tet <input.json | input.tif> [out_path_base] [ncpus]

#include <cstdlib>
#include <exception>
#include <iostream>

#include "vox2tet/core/settings.hpp"
#include "vox2tet/pipeline.hpp"

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr <<
            "Usage: run_vox2tet <input.tif | input.json> [out_path_base] [ncpus]\n"
            "  - input.tif       voxel image (one material id per voxel)\n"
            "  - input.json      saved Settings (overrides defaults)\n"
            "  - out_path_base   prefix for output files; defaults to\n"
            "                    <input-dir>/vox2tet_out/<input-stem> when\n"
            "                    input is a .tif, or the value stored in\n"
            "                    the .json otherwise\n"
            "  - ncpus           number of worker threads (default 4)\n";
        return EXIT_FAILURE;
    }
    try {
        vox2tet::Settings s;
        s.apply_cli_args(argc, argv);
        vox2tet::generate(s);
        return EXIT_SUCCESS;
    } catch (const std::exception& e) {
        std::cerr << "FATAL: " << e.what() << '\n';
        return EXIT_FAILURE;
    }
}
