// Small diagnostic: read an already-prepared TIFF (i.e. the *_C3_R64.tif
// that follows remove_small + critical_conn) and run *only* the marching-
// cubes step, writing _xyz.npy, _tr.npy, _ALL0.stl. Lets us diff against
// Python's outputs without any upstream EDT tie-break noise.
//
//   mc_only <input.tif> <out_path_base> [no_x_rotation]

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>

#include "vox2tet/brep/brep.hpp"
#include "vox2tet/core/settings.hpp"
#include "vox2tet/image/ext_volume.hpp"
#include "vox2tet/io/image_io.hpp"
#include "vox2tet/io/npy.hpp"
#include "vox2tet/marching_cubes/contouring.hpp"
#include "vox2tet/mesh/half_edge.hpp"
#include "vox2tet/remesh/smooth.hpp"

namespace v = vox2tet;

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "usage: mc_only <in.tif> <out_base> [no_x_rotation]\n";
        return 2;
    }
    const std::string in_path   = argv[1];
    const std::string out_base  = argv[2];
    const bool no_xrot          = (argc > 3) && std::string(argv[3]) == "no_x_rotation";

    auto img = v::io::read_image(in_path, std::nullopt, /*compress=*/true);
    auto [ext_vox, src_shape] = v::image::ext_volume_from_image(img);
    auto initial = v::marching_cubes::create_initial_mesh(ext_vox, /*do_2x2=*/true, out_base);
    auto [interfaces, node_mask] = v::marching_cubes::extract_material_interface_info(
        ext_vox, initial.xyz, initial.tri, out_base);
    v::marching_cubes::save_surfaces_stl(out_base, initial.xyz, initial.tri,
                                         &interfaces, false, false, false);

    // Optional X-rotation — Python applies this AFTER the
    // surface-extraction saves but BEFORE the brep / smoothing pass.
    if (!no_xrot && initial.xyz.rows() > 0) {
        const double ymax = initial.xyz.col(1).maxCoeff();
        for (Eigen::Index i = 0; i < initial.xyz.rows(); ++i) {
            const double y_old = initial.xyz(i, 1);
            const double z_old = initial.xyz(i, 2);
            initial.xyz(i, 1) = z_old;
            initial.xyz(i, 2) = ymax - y_old;
        }
    }

    // Phase 8 — half-edge + brep. Brep .ply uses the (possibly rotated)
    // xyz so it matches Python's saveBrepPLY-inside-smooth_laplace call.
    auto he = v::mesh::triangles_to_hedges(initial.tri, &interfaces);
    v::mesh::save_hedges_npy(out_base + "_hedges.npy", he);

    // Phase 9a — surface smoothing → vertex normals → sizing field.
    v::Settings s;
    auto masks   = v::mesh::get_not_fixed(he, initial.xyz, node_mask);
    auto brep_sz = v::remesh::smooth_laplace(s, initial.tri, node_mask, he,
                                             masks.not_fixed_h, initial.xyz,
                                             masks.not_fixed_v, out_base);
    auto normals = v::remesh::calc_initial_vertex_normal(
        initial.xyz, initial.tri, node_mask.masks[0]);
    double Lmin = std::numeric_limits<double>::infinity();
    for (Eigen::Index i = 0; i < brep_sz.size(); ++i)
        if (brep_sz[i] > 0 && brep_sz[i] < Lmin) Lmin = brep_sz[i];
    if (!std::isfinite(Lmin)) Lmin = 1.0;
    auto sizing = v::remesh::calc_sizing_field(he, initial.xyz, brep_sz,
                                               node_mask.masks[0], interfaces,
                                               Lmin, s.Lmax);

    v::npy::write<double>(out_base + "_xyz_s.npy",
                          {static_cast<std::size_t>(initial.xyz.rows()), 3},
                          initial.xyz.data(),
                          static_cast<std::size_t>(initial.xyz.size()));
    v::npy::write<double>(out_base + "_norm.npy",
                          {static_cast<std::size_t>(normals.rows()), 3},
                          normals.data(),
                          static_cast<std::size_t>(normals.size()));
    v::npy::write<double>(out_base + "_sz_fld.npy",
                          {static_cast<std::size_t>(sizing.size())},
                          sizing.data(),
                          static_cast<std::size_t>(sizing.size()));
    v::marching_cubes::save_surfaces_stl(out_base + "_S",
                                         initial.xyz, initial.tri,
                                         &interfaces, false, false, false);

    return EXIT_SUCCESS;
}
