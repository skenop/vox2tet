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

    // --- boundary-edge reseeding (Stages A + A2) ----------------------
    // When `do_reseed_bedges` is false the pipeline behaves exactly as
    // before (feature-chain nodes stay frozen at voxel spacing).
    // `reseed_eps` is the chordal-deviation budget in voxels: a chain
    // vertex may only be collapsed away when every removed vertex stays
    // within `reseed_eps` of the replacement chord.
    // `reseed_target_len` is the target chain spacing; 0 means "use Lmax".
    // `do_reseed_triple_lines` extends reseeding beyond the straight
    // bbox frame to the curved chains (grain-boundary traces on the
    // bbox faces and internal triple/quad lines); false = Stage A
    // frame-only behaviour.
    // `do_reseed_graded_sizing` (Stage C, EXPERIMENTAL — default off)
    // grades the sizing field after reseeding: a multi-source Dijkstra
    // pass enforces
    //   L[v] <= L[u] + (reseed_grading - 1) * |uv|
    // on every mesh edge. Measured on the JMA fixtures this does NOT
    // improve element quality (the post-reseed slivers are cap
    // triangles pinned to the coarsened chain chords, which no sizing
    // value can remove — see doc/RESEEDING.md) and at g=1.3 it refines
    // the surface next to still-frozen coarse chords, making caps MORE
    // likely. Kept as a switchable knob for future sizing work. Only
    // active together with `do_reseed_bedges`. `reseed_grading` is the
    // allowed size ratio between neighbouring vertices one unit apart;
    // values <= 1 disable grading.
    // `do_collapse_near_bedges` (the cap fix) lets the remesh loop
    // collapse interior edges whose one-ring touches a feature chain —
    // the conservative ring guard otherwise freezes every vertex one
    // ring from a chain, which is what leaves "cap" triangles (a free
    // vertex nearly collinear with a coarsened chain chord) in the
    // final mesh. The relaxed path still requires the two dying
    // triangles' side edges to be free, manifold and mutual, and adds a
    // corner-angle + normal-flip guard on every retargeted fan triangle
    // that touches a chain. Only active together with
    // `do_reseed_bedges` (legacy contract preserved).
    // `do_reseed_lfs` (Stage B) caps the per-chain target spacing by
    // the composite local feature size: for every chain vertex,
    //   lfs = min(d_chains, d_self, d_surf)
    // computed against non-incident geometry only (incidence is decided
    // by a geodesic collar, not by entity id: a candidate counts when
    // its geodesic distance from the query exceeds 2x the Euclidean
    // one, so another arm of the SAME chain still caps the spacing).
    // The chain target is then clamp(reseed_beta * lfs, 1 voxel,
    // reseed_target_len) and gradient-limited along the chain graph
    // with slope (reseed_grading - 1). This closes the chord-vs-chord
    // near-miss hole: without it, only the triangle-quality guard
    // stops a chord from running close past another chain or sheet.
    bool   do_reseed_bedges      = true;
    bool   do_reseed_triple_lines = true;
    bool   do_reseed_graded_sizing = false;
    bool   do_collapse_near_bedges = true;
    bool   do_reseed_lfs         = true;
    double reseed_eps            = 0.4;
    double reseed_target_len     = 0.0;
    double reseed_grading        = 1.3;
    double reseed_beta           = 0.7;

    // --- remeshing ----------------------------------------------------
    int    n_remesh_itr                = 7;
    double Lmax                        = 40.0;
    bool   do_save_remesh_interfaces   = false;
    bool   do_save_remesh_grains_stl   = true;
    bool   do_save_remesh_grains_inp   = true;

    // --- volume (tetrahedral) meshing ---------------------------------
    // Backend selected by `tet_mesher` (gated on `do_tetgen_meshing`,
    // which remains the master "do volume meshing" switch):
    //   "cdt"    — (default) the built-in in-process pipeline: Steiner
    //              constrained Delaunay tetrahedrization
    //              (third-party/CDT) that preserves the remeshed
    //              surface exactly, followed by MMG3D quality
    //              optimization (third-party/mmg) that inserts/moves
    //              interior vertices only (the surface and material
    //              interfaces stay frozen). No external binaries.
    //   "tetgen" — the external `tetgen` binary on $PATH (legacy;
    //              faster and ~40% fewer tets, but worse quality —
    //              see doc/COMPARISON.md).
    // `do_mmg_optim` disables the MMG pass ("cdt" backend only) —
    // the raw CDT mesh is exact but has sliver-quality interior tets.
    // `mmg_verbose` is MMG's own verbosity [-1..10].
    std::string tet_mesher   = "cdt";
    bool        do_mmg_optim = true;
    int         mmg_verbose  = 0;

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
