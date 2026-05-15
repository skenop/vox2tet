#pragma once

// Pipeline configuration. Every parameter that can be tweaked from
// the command line or a JSON file lives here as a plain data member
// with a sensible default. JSON I/O stores the object as a 3-element
// list to preserve forward-compatibility with the legacy file format:
//   [date_string, "Cglobal", { attr_name: value, ... }]

#include <string>

namespace vox2tet {

struct Settings {
    // --- main ----------------------------------------------------------
    int         ncpus           = 4;
    std::string input_img_file  = "./tests/data/JMA_10/JMA_10.tif";
    std::string out_path_base   = "./tests/data/JMA_10/vox2tet_out/JMA_10";

    // --- general workflow flags ---------------------------------------
    bool do_remove_small         = true;
    bool do_diagonal_connectivity= true;
    bool do_initial_surface      = true;
    bool do_preremesh_steps      = true;
    bool do_remeshing            = true;
    bool do_tetgen_meshing       = true;
    bool do_abaqus_verification  = true;

    bool do_init_surf_load       = false;
    bool do_preremesh_load       = false;

    bool do_print_redirect       = false;
    bool do_x_rotation           = true;

    // --- initial surface ----------------------------------------------
    int  max_remove_size           = 64;
    int  connectivity              = 3;
    bool do_show_hist              = false;
    bool do_2x2patterns            = true;
    bool do_save_init_interfaces   = false;
    bool do_save_init_grains_stl   = false;
    bool do_save_init_grains_inp   = false;

    // --- smoothing ----------------------------------------------------
    int    n_smooth_steps              = 7;
    double smooth_alpha                = 0.5;
    double max_D2self                  = 0.8;
    double min_D2other                 = 0.3;
    double min_dangle_boundary         = 20.0;
    double min_dangle_internal         = 40.0;
    // Sliver-repair thresholds (Layer 4). Same boundary/internal split
    // as the dihedral guard. Triangles whose smallest interior corner
    // angle falls below the matching threshold are targeted by the
    // sliver-flip and sliver-collapse passes after the remesh loop.
    double min_corner_angle_boundary   = 20.0;
    double min_corner_angle_internal   = 30.0;
    bool   do_save_smooth_interfaces   = false;
    bool   do_save_smooth_grains_stl   = false;
    bool   do_save_smooth_grains_inp   = false;

    // --- b-spline edge smoothing --------------------------------------
    double brep_smoothness = 7.0;
    int    brep_degree     = 3;

    // --- remeshing ----------------------------------------------------
    int    n_remesh_itr                = 7;
    double Lmax                        = 40.0;
    bool   do_save_remesh_interfaces   = false;
    bool   do_save_remesh_grains_stl   = true;
    bool   do_save_remesh_grains_inp   = true;

    // ------------------------------------------------------------------
    // JSON I/O — file layout matches the legacy `obj2json` / `json2obj`
    // format (`[date, "Cglobal", { attrs… }]`).
    void dump_json(const std::string& path = "") const;
    void load_json(const std::string& path);

    // CLI constructor equivalent: positional args
    //   input_image_or_json, output_path_base (optional), ncpus (optional).
    // Mirrors Cglobal.__init__.
    void apply_cli_args(int argc, char** argv);
};

}  // namespace vox2tet
