#include "vox2tet/core/settings.hpp"

#include "vox2tet/core/log.hpp"
#include "vox2tet/core/paths.hpp"

#include <chrono>
#include <ctime>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

#include <nlohmann/json.hpp>

using nlohmann::json;

namespace vox2tet {

namespace {

// Macro list of (name, type) pairs to keep dump/load symmetric and
// auditable in one place.
#define VOX2TET_SETTINGS_FIELDS(X)                              \
    X(ncpus)                                                    \
    X(input_img_file)                                           \
    X(out_path_base)                                            \
    X(do_remove_small)                                          \
    X(do_diagonal_connectivity)                                 \
    X(do_initial_surface)                                       \
    X(do_preremesh_steps)                                       \
    X(do_remeshing)                                             \
    X(do_tetgen_meshing)                                        \
    X(do_abaqus_verification)                                   \
    X(do_init_surf_load)                                        \
    X(do_preremesh_load)                                        \
    X(do_print_redirect)                                        \
    X(do_x_rotation)                                            \
    X(max_remove_size)                                          \
    X(connectivity)                                             \
    X(do_show_hist)                                             \
    X(do_2x2patterns)                                           \
    X(do_save_init_interfaces)                                  \
    X(do_save_init_grains_stl)                                  \
    X(do_save_init_grains_inp)                                  \
    X(n_smooth_steps)                                           \
    X(smooth_alpha)                                             \
    X(max_D2self)                                               \
    X(min_D2other)                                              \
    X(min_dangle_boundary)                                      \
    X(min_dangle_internal)                                      \
    X(min_corner_angle_boundary)                                \
    X(min_corner_angle_internal)                                \
    X(do_save_smooth_interfaces)                                \
    X(do_save_smooth_grains_stl)                                \
    X(do_save_smooth_grains_inp)                                \
    X(brep_smoothness)                                          \
    X(brep_degree)                                              \
    X(do_reseed_bedges)                                         \
    X(do_reseed_triple_lines)                                   \
    X(do_reseed_graded_sizing)                                  \
    X(do_collapse_near_bedges)                                  \
    X(do_reseed_lfs)                                            \
    X(reseed_eps)                                               \
    X(reseed_target_len)                                        \
    X(reseed_grading)                                           \
    X(reseed_beta)                                              \
    X(n_remesh_itr)                                             \
    X(Lmax)                                                     \
    X(do_save_remesh_interfaces)                                \
    X(do_save_remesh_grains_stl)                                \
    X(do_save_remesh_grains_inp)                                \
    X(tet_mesher)                                               \
    X(do_mmg_optim)                                             \
    X(mmg_verbose)

std::string now_string() {
    using namespace std::chrono;
    auto t = system_clock::to_time_t(system_clock::now());
    std::tm tm_buf{};
#if defined(_WIN32)
    localtime_s(&tm_buf, &t);
#else
    localtime_r(&t, &tm_buf);
#endif
    char buf[32];
    std::strftime(buf, sizeof(buf), "%d %b %y, %H:%M:%S", &tm_buf);
    return buf;
}

}  // namespace

void Settings::dump_json(const std::string& path) const {
    std::string file = path.empty() ? (out_path_base + ".json") : path;
    paths::create_folder(file);

    json props;
#define X(name) props[#name] = name;
    VOX2TET_SETTINGS_FIELDS(X)
#undef X

    json doc = json::array();
    doc.push_back(now_string());
    doc.push_back("Cglobal");
    doc.push_back(props);

    std::ofstream f(file);
    if (!f) throw std::runtime_error("dump_json: cannot open " + file);
    f << doc.dump(4);
}

void Settings::load_json(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("load_json: cannot open " + path);
    json doc;
    f >> doc;

    // The reference writes [date, "Cglobal", {attrs...}], but we tolerate either
    // the wrapped form or a bare object.
    json props;
    if (doc.is_array() && doc.size() >= 3) {
        props = doc[2];
    } else if (doc.is_object()) {
        props = doc;
    } else {
        throw std::runtime_error("load_json: unexpected layout in " + path);
    }

#define X(name) if (props.contains(#name)) props.at(#name).get_to(name);
    VOX2TET_SETTINGS_FIELDS(X)
#undef X
}

void Settings::apply_cli_args(int argc, char** argv) {
    if (argc <= 1) return;
    std::string input = argv[1];
    if (paths::file_extension(input, /*to_upper=*/true) == ".JSON") {
        load_json(input);
    } else {
        input_img_file = input;
    }
    if (argc > 2) {
        out_path_base = argv[2];
    } else if (paths::file_extension(input, /*to_upper=*/true) != ".JSON") {
        // Default for `run_vox2tet path/to/image.tif`:
        //   out_path_base = path/to/vox2tet_out/image
        // Output files like image_RE.smesh, image_RE_G_*.stl etc. then
        // land next to the input under a dedicated subfolder.
        const std::string folder = paths::base_folder(input);  // "path/to"
        const auto slash = input.find_last_of("/\\");
        const std::string fname = (slash == std::string::npos) ? input
                                                                : input.substr(slash + 1);
        const auto dot = fname.find_last_of('.');
        const std::string stem = (dot == std::string::npos) ? fname : fname.substr(0, dot);
        out_path_base = (folder.empty() ? "vox2tet_out" : folder + "/vox2tet_out")
                      + "/" + stem;
    }
    if (argc > 3) ncpus = std::stoi(argv[3]);
}

}  // namespace vox2tet
