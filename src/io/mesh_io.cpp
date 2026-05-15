#include "vox2tet/io/mesh_io.hpp"

#include "vox2tet/core/paths.hpp"
#include "vox2tet/mesh/mathx.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>

namespace vox2tet::io {

namespace {

// Map a triangle row to the Nx3 double matrix expected by mathx.
MatrixNx3<double> rows_for(const Coords& xyz, const Triangles& tri, int col) {
    MatrixNx3<double> out(tri.rows(), 3);
    for (Eigen::Index i = 0; i < tri.rows(); ++i) {
        out.row(i) = xyz.row(tri(i, col));
    }
    return out;
}

}  // namespace

// ---------------------------------------------------------------------------
// HSV → packed 16-bit RGB (1 bit valid + 5/5/5). MeshLab and 3DViewer
// pick this up as a per-facet colour. The golden-angle hue stride
// (≈137.508°) ensures consecutive material ids land far apart on the
// colour wheel — much friendlier than the raw-index encoding which
// clusters at green/black/blue because the low bits of small ints fall
// in the "blue + a little green" region of the 5-5-5 colour cube.
std::uint16_t material_to_stl_color(std::uint32_t material_id) {
    constexpr double golden = 0.61803398874989485;   // (1 + √5) / 2 − 1
    const double h = std::fmod(static_cast<double>(material_id) * golden, 1.0);
    // Pleasant pastel: not fully saturated, value below max so light
    // colours don't wash out against MeshLab's white background.
    const double s = 0.75;
    const double v = 0.85;

    // HSV → RGB.
    const double hh = h * 6.0;
    const int    i  = static_cast<int>(std::floor(hh)) % 6;
    const double f  = hh - std::floor(hh);
    const double p  = v * (1.0 - s);
    const double q  = v * (1.0 - s * f);
    const double t  = v * (1.0 - s * (1.0 - f));
    double rr = 0, gg = 0, bb = 0;
    switch (i) {
        case 0: rr = v; gg = t; bb = p; break;
        case 1: rr = q; gg = v; bb = p; break;
        case 2: rr = p; gg = v; bb = t; break;
        case 3: rr = p; gg = q; bb = v; break;
        case 4: rr = t; gg = p; bb = v; break;
        case 5: rr = v; gg = p; bb = q; break;
    }
    auto q5 = [](double c) -> std::uint16_t {
        if (c < 0) c = 0; if (c > 1) c = 1;
        return static_cast<std::uint16_t>(std::lround(c * 31.0));
    };
    const std::uint16_t R = q5(rr);
    const std::uint16_t G = q5(gg);
    const std::uint16_t B = q5(bb);
    // bit 15 = valid; 14-10 = R; 9-5 = G; 4-0 = B.
    return static_cast<std::uint16_t>(0x8000u | (R << 10) | (G << 5) | B);
}

void save_stl_blocks(const std::string& path,
                     const Coords& xyz,
                     const Triangles& tri,
                     const std::vector<std::uint16_t>* attributes,
                     bool attributes_are_colors) {
    paths::create_folder(path);

    // 80-byte header, padded with spaces (matches the reference "Image2Mesh v1.2…").
    std::string header = "Image2Mesh (v1.2) solid " + path;
    if (header.size() > 80) header.resize(80);
    else                    header.resize(80, ' ');

    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("save_stl_blocks: cannot open " + path);
    f.write(header.data(), 80);

    std::uint32_t n_tri = static_cast<std::uint32_t>(tri.rows());
    f.write(reinterpret_cast<const char*>(&n_tri), 4);

    if (n_tri == 0) return;

    // Compute normals in float32 to match the reference float32 buffer cast.
    MatrixNx3<double> p1 = rows_for(xyz, tri, 0);
    MatrixNx3<double> p2 = rows_for(xyz, tri, 1);
    MatrixNx3<double> p3 = rows_for(xyz, tri, 2);
    MatrixNx3<double> n  = mathx::calc_tri_norm_v(p1, p2, p3);

    // 50-byte STL record: float32[3] normal, float32[9] verts, uint16 attr.
    std::array<unsigned char, 50> rec{};
    for (std::uint32_t i = 0; i < n_tri; ++i) {
        std::array<float, 3> nf = {static_cast<float>(n(i, 0)),
                                   static_cast<float>(n(i, 1)),
                                   static_cast<float>(n(i, 2))};
        std::memcpy(rec.data(), nf.data(), 12);
        for (int v = 0; v < 3; ++v) {
            std::array<float, 3> vf = {static_cast<float>(xyz(tri(i, v), 0)),
                                       static_cast<float>(xyz(tri(i, v), 1)),
                                       static_cast<float>(xyz(tri(i, v), 2))};
            std::memcpy(rec.data() + 12 + 12 * v, vf.data(), 12);
        }
        std::uint16_t attr = 0;
        if (attributes) {
            const std::uint16_t raw = (*attributes)[i];
            attr = attributes_are_colors ? raw : material_to_stl_color(raw);
        }
        std::memcpy(rec.data() + 48, &attr, 2);
        f.write(reinterpret_cast<const char*>(rec.data()), 50);
    }
}

void save_inp(const std::string& path,
              const Coords& xyz,
              const Eigen::Ref<const Eigen::MatrixXi>& elems,
              const std::vector<std::int32_t>* esets) {
    paths::create_folder(path);
    std::ofstream f(path);
    if (!f) throw std::runtime_error("save_inp: cannot open " + path);

    // Optionally sort element ids by elset key so the *Elset, generate
    // blocks reference contiguous element ranges. Mirrors `saveinp`'s
    // np.argsort step.
    std::vector<Eigen::Index> order(elems.rows());
    for (Eigen::Index i = 0; i < elems.rows(); ++i) order[i] = i;
    std::vector<std::int32_t> esets_sorted;
    if (esets && static_cast<Eigen::Index>(esets->size()) == elems.rows()) {
        std::sort(order.begin(), order.end(),
                  [&](Eigen::Index a, Eigen::Index b) { return (*esets)[a] < (*esets)[b]; });
        esets_sorted.resize(esets->size());
        for (std::size_t i = 0; i < order.size(); ++i)
            esets_sorted[i] = (*esets)[order[i]];
    }

    // Unique node ids referenced by `elems`.
    std::vector<std::int32_t> nodes;
    nodes.reserve(elems.size());
    for (Eigen::Index i = 0; i < elems.rows(); ++i)
        for (Eigen::Index j = 0; j < elems.cols(); ++j)
            nodes.push_back(elems(i, j));
    std::sort(nodes.begin(), nodes.end());
    nodes.erase(std::unique(nodes.begin(), nodes.end()), nodes.end());

    f << "*HEADING\n*NODE, NSET=ALL\n";
    for (auto nid : nodes) {
        char buf[128];
        std::snprintf(buf, sizeof(buf), "%d, %e, %e, %e\n",
                      nid + 1, xyz(nid, 0), xyz(nid, 1), xyz(nid, 2));
        f << buf;
    }

    const char* eltype = (elems.cols() == 4) ? "C3D4" :
                        (elems.cols() == 3) ? "S3"   : "???";
    f << "*ELEMENT,TYPE=" << eltype << "\n";
    for (std::size_t idx = 0; idx < order.size(); ++idx) {
        Eigen::Index i = order[idx];
        f << (idx + 1);
        for (Eigen::Index j = 0; j < elems.cols(); ++j)
            f << "," << (elems(i, j) + 1);
        f << "\n";
    }

    if (esets && !esets_sorted.empty()) {
        std::size_t i = 0;
        while (i < esets_sorted.size()) {
            std::size_t j = i;
            while (j < esets_sorted.size() && esets_sorted[j] == esets_sorted[i]) ++j;
            f << "*Elset, elset=ESET-" << esets_sorted[i] << ", generate\n"
              << (i + 1) << ", " << j << ", 1\n";
            i = j;
        }
    }
}

void save_smesh(const std::string& path,
                const Coords& xyz,
                const Triangles& facets,
                const std::vector<std::uint32_t>* attributes) {
    paths::create_folder(path);
    std::ofstream f(path);
    if (!f) throw std::runtime_error("save_smesh: cannot open " + path);

    f << "# Part 1 - node list\n";
    f << "# node count, 3 dim, no attribute, no boundary marker\n";
    f << xyz.rows() << " 3 0 0\n";
    for (Eigen::Index i = 0; i < xyz.rows(); ++i) {
        char buf[128];
        std::snprintf(buf, sizeof(buf), "%ld %f %f %f\n",
                      static_cast<long>(i), xyz(i, 0), xyz(i, 1), xyz(i, 2));
        f << buf;
    }
    f << "# Part 2 - facet list\n";
    f << "# facet count, boundary marker\n";
    if (!attributes) {
        f << facets.rows() << " 0\n";
        for (Eigen::Index i = 0; i < facets.rows(); ++i) {
            f << "3 " << facets(i, 0) << " " << facets(i, 1) << " "
              << facets(i, 2) << "\n";
        }
    } else {
        f << facets.rows() << " 1\n";
        for (Eigen::Index i = 0; i < facets.rows(); ++i) {
            f << "3 " << facets(i, 0) << " " << facets(i, 1) << " "
              << facets(i, 2) << " " << (*attributes)[i] << "\n";
        }
    }
}

void save_edges_ply(const std::string& path,
                    const Coords& xyz,
                    const Eigen::Ref<const Eigen::Matrix<std::uint32_t,
                                                          Eigen::Dynamic, 2,
                                                          Eigen::RowMajor>>& edges,
                    const std::array<std::uint8_t, 3>* rgb) {
    paths::create_folder(path);
    std::ofstream f(path);
    if (!f) throw std::runtime_error("save_edges_ply: cannot open " + path);

    // Compress node indices.
    std::vector<std::uint32_t> unique_nodes;
    unique_nodes.reserve(edges.size());
    for (Eigen::Index i = 0; i < edges.rows(); ++i) {
        unique_nodes.push_back(edges(i, 0));
        unique_nodes.push_back(edges(i, 1));
    }
    std::sort(unique_nodes.begin(), unique_nodes.end());
    unique_nodes.erase(std::unique(unique_nodes.begin(), unique_nodes.end()),
                       unique_nodes.end());
    std::vector<std::uint32_t> nodemap(unique_nodes.empty() ? 0 :
                                        unique_nodes.back() + 1,
                                        static_cast<std::uint32_t>(-1));
    for (std::size_t i = 0; i < unique_nodes.size(); ++i)
        nodemap[unique_nodes[i]] = static_cast<std::uint32_t>(i);

    f << "ply\nformat ascii 1.0\n";
    f << "element vertex " << unique_nodes.size() << "\n";
    f << "property float x\nproperty float y\nproperty float z\n";
    f << "element edge " << edges.rows() << "\n";
    f << "property int vertex1\nproperty int vertex2\n";
    if (rgb) {
        f << "property uchar red\nproperty uchar green\nproperty uchar blue\n";
    }
    f << "end_header\n";

    for (auto n : unique_nodes) {
        char buf[64];
        std::snprintf(buf, sizeof(buf), "%f %f %f\n",
                      xyz(n, 0), xyz(n, 1), xyz(n, 2));
        f << buf;
    }
    if (!rgb) {
        for (Eigen::Index i = 0; i < edges.rows(); ++i) {
            f << nodemap[edges(i, 0)] << " " << nodemap[edges(i, 1)] << "\n";
        }
    } else {
        for (Eigen::Index i = 0; i < edges.rows(); ++i) {
            f << nodemap[edges(i, 0)] << " " << nodemap[edges(i, 1)] << " "
              << static_cast<int>((*rgb)[0]) << " "
              << static_cast<int>((*rgb)[1]) << " "
              << static_cast<int>((*rgb)[2]) << "\n";
        }
    }
}

void save_xyz_text(const std::string& path,
                   const Coords& xyz,
                   const std::vector<std::uint8_t>& mask) {
    paths::create_folder(path);
    std::ofstream f(path);
    if (!f) throw std::runtime_error("save_xyz_text: cannot open " + path);
    for (Eigen::Index i = 0; i < xyz.rows(); ++i) {
        if (i < static_cast<Eigen::Index>(mask.size()) && !mask[i]) continue;
        char buf[80];
        std::snprintf(buf, sizeof(buf), "%1.6f %1.6f %1.6f\n",
                      xyz(i, 0), xyz(i, 1), xyz(i, 2));
        f << buf;
    }
}

}  // namespace vox2tet::io
