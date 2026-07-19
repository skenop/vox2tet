#include "vox2tet/pipeline.hpp"

#include "vox2tet/brep/brep.hpp"
#include "vox2tet/core/log.hpp"
#include "vox2tet/core/paths.hpp"
#include "vox2tet/image/ext_volume.hpp"
#include "vox2tet/image/image_utils.hpp"
#include "vox2tet/io/image_io.hpp"
#include "vox2tet/io/mesh_io.hpp"
#include "vox2tet/io/npy.hpp"
#include "vox2tet/marching_cubes/contouring.hpp"
#include "vox2tet/mesh/half_edge.hpp"
#include "vox2tet/remesh/remesh.hpp"
#include "vox2tet/remesh/reseed.hpp"
#include "vox2tet/remesh/smooth.hpp"
#include "vox2tet/tetgen/tetgen_runner.hpp"
#include "vox2tet/tetmesh/cdt_mmg_runner.hpp"

#include <cmath>
#include <iostream>
#include <limits>
#include <vector>

namespace vox2tet {

namespace {

void log_phase(const char* name) {
    std::cout << "\n.\n ------------------   " << name
              << "   ------------------\n";
}

}  // namespace

void generate(const Settings& s) {
    VOX2TET_PRINT("vox2tet start");
    paths::create_folder(s.out_path_base);

    log::PrintRedirect redir;
    if (s.do_print_redirect) {
        redir = log::open_redirect(s.out_path_base + "_OUT.txt");
        VOX2TET_LOG() << "output is redirected in " << s.out_path_base << "_OUT.txt";
    }
    VOX2TET_LOG() << "OUT FOLDER - " << s.out_path_base;

    s.dump_json();

    // ------------------------------------------------------------------
    // Phase: image preparation (remove small regions + critical conn fix)
    // ------------------------------------------------------------------
    std::string img_file = s.input_img_file;
    if (s.do_remove_small || s.do_diagonal_connectivity) {
        log_phase("IMAGE PREPARATION");
        auto vox = io::read_image(s.input_img_file, std::nullopt, /*compress=*/true);
        if (s.do_remove_small) {
            auto rs = image::remove_small_regions(s, vox);
            vox      = std::move(rs.voxels);
            img_file = rs.img_file_path;
        }
        if (s.do_diagonal_connectivity) {
            auto fc = image::fix_critical_connectivity(img_file, s.out_path_base);
            img_file = fc.out_file_path;
        }
    }

    // Aggregate state we need to carry across phases (analogous to the reference
    // returning tuples between initial_surface / preremesh / remesh).
    marching_cubes::InitialMesh                 initial;
    std::vector<marching_cubes::Interface>      interfaces;
    marching_cubes::NodeTypeMask                node_type_mask;
    remesh::RemeshState                         remesh_state;

    if (s.do_initial_surface) {
        log_phase("INITIAL SURFACE STEP");
        auto [ext_vox, src_shape] = image::ext_volume_from_file(img_file);
        // create_initial_mesh writes _xyz.npy, _tr.npy, _ALL0.stl in
        // un-rotated half-voxel float coordinates.
        initial = marching_cubes::create_initial_mesh(
            ext_vox, s.do_2x2patterns, s.out_path_base);

        // extract_material_interface_info: classify nodes, sort triangles
        // by bimaterial, write _att.npy / _ntp.npy / _{EXT,FIX,MML}.xyz.
        auto pair = marching_cubes::extract_material_interface_info(
            ext_vox, initial.xyz, initial.tri, s.out_path_base);
        interfaces     = std::move(pair.first);
        node_type_mask = std::move(pair.second);

        // Now the attributed _ALL.stl (one material id per triangle).
        marching_cubes::save_surfaces_stl(s.out_path_base,
                                          initial.xyz, initial.tri,
                                          &interfaces,
                                          s.do_save_init_interfaces,
                                          s.do_save_init_grains_stl,
                                          s.do_save_init_grains_inp);

        // Optional X-rotation, applied AFTER all initial-surface saves.
        // This mirrors generate_initial_surface() in the reference where the
        // rotation only affects the in-memory mesh passed to subsequent
        // stages (smoothing / remeshing).
        if (s.do_x_rotation && initial.xyz.rows() > 0) {
            const double ymax = initial.xyz.col(1).maxCoeff();
            for (Eigen::Index i = 0; i < initial.xyz.rows(); ++i) {
                const double y_old = initial.xyz(i, 1);
                const double z_old = initial.xyz(i, 2);
                initial.xyz(i, 1) = z_old;          // y_new = z_old
                initial.xyz(i, 2) = ymax - y_old;   // z_new = ymax - y_old
            }
        }
    }

    if (s.do_preremesh_steps) {
        log_phase("PRE REMESH STEPS");
        if (initial.tri.rows() == 0) {
            VOX2TET_PRINT("preremesh: skipped — no surface to process");
        } else {
            // Phase 8: half-edge build.
            auto he = mesh::triangles_to_hedges(initial.tri, &interfaces);
            mesh::save_hedges_npy(s.out_path_base + "_hedges.npy", he);

            // Phase 9a: not-fixed masks → boundary chain smoothing →
            // surface smoothing with dihedral-angle control → vertex
            // normals → sizing field. smooth_laplace itself handles
            // get_boundary_edges + order_bedges + _E.ply / _S_E.ply.
            auto masks   = mesh::get_not_fixed(he, initial.xyz, node_type_mask);
            auto brep_sz = remesh::smooth_laplace(s, initial.tri, node_type_mask,
                                                  he,
                                                  masks.not_fixed_h, initial.xyz,
                                                  masks.not_fixed_v,
                                                  s.out_path_base);
            // Boundary-edge reseeding (Stage A: bbox frame; Stage A2:
            // traces and triple lines when do_reseed_triple_lines).
            // Coarsens the feature chains by guarded edge collapse and
            // compacts the mesh arrays in place; every piece of derived
            // state (half-edges, masks, brep sizing) is then recomputed
            // from the reseeded mesh. With do_reseed_bedges=false this
            // block is skipped and the pipeline behaves exactly as
            // before (frozen chains).
            if (s.do_reseed_bedges) {
                const auto rs = remesh::reseed_feature_chains(
                    s, initial.tri, initial.xyz, node_type_mask, interfaces);
                if (rs.verts_removed > 0) {
                    he    = mesh::triangles_to_hedges(initial.tri, &interfaces);
                    masks = mesh::get_not_fixed(he, initial.xyz, node_type_mask);
                    std::vector<std::uint8_t> is_brep(
                        static_cast<std::size_t>(initial.xyz.rows()), 0);
                    for (Eigen::Index v = 0; v < initial.xyz.rows(); ++v)
                        is_brep[static_cast<std::size_t>(v)] =
                            !node_type_mask.masks[0][v];
                    brep_sz = remesh::calc_brep_sizing(initial.xyz, is_brep);
                    // Debug dump of the reseeded chains (_RS_E.ply).
                    auto bedges2 = brep::get_boundary_edges(initial.tri,
                                                            node_type_mask);
                    auto brep2 = brep::order_bedges(bedges2, node_type_mask);
                    brep::save_brep_ply(brep2, initial.xyz,
                                        s.out_path_base + "_RS",
                                        {255, 128, 0});
                }
            }

            auto normals = remesh::calc_initial_vertex_normal(
                initial.xyz, initial.tri, node_type_mask.masks[0]);
            // Full per-vertex sizing field for the remesh phase.
            // Lmin = min(brep_sz), Lmax = settings.Lmax.
            double Lmin = std::numeric_limits<double>::infinity();
            for (Eigen::Index i = 0; i < brep_sz.size(); ++i)
                if (brep_sz[i] > 0 && brep_sz[i] < Lmin) Lmin = brep_sz[i];
            if (!std::isfinite(Lmin)) Lmin = 1.0;
            auto sizing = remesh::calc_sizing_field(he, initial.xyz, brep_sz,
                                                    node_type_mask.masks[0],
                                                    interfaces,
                                                    Lmin, s.Lmax);
            // Stage C (experimental, default off): grade the field so
            // size transitions are bounded. Gated on do_reseed_bedges
            // so that flag alone still restores the legacy pipeline
            // exactly. See doc/RESEEDING.md for why this is off by
            // default (measured: no quality benefit on JMA fixtures).
            if (s.do_reseed_bedges && s.do_reseed_graded_sizing) {
                const auto lowered = remesh::limit_sizing_gradient(
                    he, initial.xyz, sizing, s.reseed_grading);
                VOX2TET_LOG() << "limit_sizing_gradient: g="
                              << s.reseed_grading << ", lowered " << lowered
                              << " of " << sizing.size() << " vertices";
            }

            // Persist _xyz_s.npy / _norm.npy / _sz_fld.npy as float64 —
            // matches the reference `np.save(... , xyz)` where xyz has been
            // promoted to float64 in generate_initial_surface().
            paths::create_folder(s.out_path_base);
            npy::write<double>(s.out_path_base + "_xyz_s.npy",
                               {static_cast<std::size_t>(initial.xyz.rows()), 3},
                               initial.xyz.data(),
                               static_cast<std::size_t>(initial.xyz.size()));
            npy::write<double>(s.out_path_base + "_norm.npy",
                               {static_cast<std::size_t>(normals.rows()), 3},
                               normals.data(),
                               static_cast<std::size_t>(normals.size()));
            // sz_fld: The reference writes its full-length float64 vector; we
            // just dump the Eigen::VectorXd as a 1D npy.
            npy::write<double>(s.out_path_base + "_sz_fld.npy",
                               {static_cast<std::size_t>(sizing.size())},
                               sizing.data(),
                               static_cast<std::size_t>(sizing.size()));

            // _S_ALL.stl — smoothed surface (with bimaterial attributes).
            marching_cubes::save_surfaces_stl(s.out_path_base + "_S",
                                              initial.xyz, initial.tri,
                                              &interfaces,
                                              s.do_save_smooth_interfaces,
                                              s.do_save_smooth_grains_stl,
                                              s.do_save_smooth_grains_inp);

            // Hand mesh state off to the remesh phase.
            remesh_state.hedges      = std::move(he);
            remesh_state.not_fixed_h = std::move(masks.not_fixed_h);
            remesh_state.xyz         = initial.xyz;          // smoothed coords
            remesh_state.not_fixed_v = std::move(masks.not_fixed_v);
            remesh_state.normals     = std::move(normals);
            remesh_state.sizing      = std::move(sizing);
        }
    }

    if (s.do_remeshing) {
        log_phase("REMESH STEP");
        if (remesh_state.xyz.rows() == 0) {
            VOX2TET_PRINT("remesh: skipped — no preremesh state available");
        } else {
            auto rr = remesh::remesh(s, remesh_state, interfaces);
            // Persist _RE_ALL.stl and friends.
            marching_cubes::save_surfaces_stl(rr.surface_path,
                                              rr.xyz, rr.triangles,
                                              &interfaces,
                                              s.do_save_remesh_interfaces,
                                              s.do_save_remesh_grains_stl,
                                              s.do_save_remesh_grains_inp);
            // Keep the remeshed mesh available for the tetgen phase.
            initial.xyz = std::move(rr.xyz);
            initial.tri = std::move(rr.triangles);
        }
    }

    if (s.do_tetgen_meshing) {
        log_phase("TET MESH STEP");
        if (initial.tri.rows() == 0) {
            VOX2TET_PRINT("tet meshing: skipped — no surface mesh");
        } else {
            const std::string base = s.out_path_base + "_RE";
            if (s.tet_mesher == "cdt") {
                tetmesh::mesh_volume(s, base, initial.xyz, initial.tri,
                                     interfaces);
            } else {
                tetgen::mesh_volume(base, initial.xyz, initial.tri,
                                    interfaces, s.do_abaqus_verification);
            }
        }
    }

    if (redir.active) log::close_redirect(redir);
    VOX2TET_PRINT("DONE!");
}

}  // namespace vox2tet
