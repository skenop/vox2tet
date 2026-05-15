#include "vox2tet/tetgen/tetgen_runner.hpp"

#include "vox2tet/core/log.hpp"
#include "vox2tet/core/paths.hpp"
#include "vox2tet/io/mesh_io.hpp"

#include <Eigen/LU>  // for Matrix4d::determinant
#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace vox2tet::tetgen {

namespace {

// Skip TetGen comment ('#') lines and blank lines, returning the next
// real line. Returns false on EOF.
bool next_line(std::istream& in, std::string& out) {
    while (std::getline(in, out)) {
        // Trim leading whitespace + comments.
        const auto pos = out.find_first_not_of(" \t\r");
        if (pos == std::string::npos) continue;
        if (out[pos] == '#') continue;
        return true;
    }
    return false;
}

// Tokenise whitespace-separated fields of a line.
std::vector<std::string> tokenise(const std::string& line) {
    std::vector<std::string> out;
    std::istringstream ss(line);
    std::string tok;
    while (ss >> tok) out.push_back(tok);
    return out;
}

// Parse the TetGen .1.node file: line 0 is "n 3 attr_count marker"; the
// next n lines are "id x y z [attr] [marker]". We only keep id and xyz.
Coords parse_node_file(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("tetgen: cannot open " + path);
    std::string line;
    if (!next_line(f, line)) throw std::runtime_error("tetgen: empty node file");
    const auto header = tokenise(line);
    if (header.size() < 1) throw std::runtime_error("tetgen: bad node header");
    const std::size_t n_nodes = std::stoull(header[0]);

    // Determine the firstnumber convention by peeking at the first node's
    // id: TetGen can emit either 0-based or 1-based depending on the
    // -z flag (default for 1.6.0 is 0-based, but older versions used 1).
    Coords xyz(static_cast<Eigen::Index>(n_nodes), 3);
    std::int64_t firstnumber = -1;
    for (std::size_t i = 0; i < n_nodes; ++i) {
        if (!next_line(f, line))
            throw std::runtime_error("tetgen: short node file at row " + std::to_string(i));
        const auto tok = tokenise(line);
        if (tok.size() < 4) throw std::runtime_error("tetgen: bad node row " + std::to_string(i));
        const std::int64_t id = std::stoll(tok[0]);
        if (firstnumber < 0) firstnumber = id;        // first id seen
        const Eigen::Index r = static_cast<Eigen::Index>(id - firstnumber);
        if (r < 0 || r >= static_cast<Eigen::Index>(n_nodes))
            throw std::runtime_error("tetgen: node id out of range");
        xyz(r, 0) = std::stod(tok[1]);
        xyz(r, 1) = std::stod(tok[2]);
        xyz(r, 2) = std::stod(tok[3]);
    }
    return xyz;
}

// Parse the .1.ele file. Each row: "id v0 v1 v2 v3 [region]". The region
// attribute is the *last* column (mirrors `ele[:, -1]` in the reference). Verts
// are stored 1-based; we convert to 0-based for internal use.
struct EleRecord {
    std::array<std::int64_t, 4> v;     // 0-based
    std::int64_t                region;
};
std::vector<EleRecord> parse_ele_file(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("tetgen: cannot open " + path);
    std::string line;
    if (!next_line(f, line)) throw std::runtime_error("tetgen: empty ele file");
    const auto header = tokenise(line);
    const std::size_t n_ele = std::stoull(header[0]);

    std::vector<EleRecord> out;
    out.reserve(n_ele);
    std::int64_t firstnumber = -1;
    for (std::size_t i = 0; i < n_ele; ++i) {
        if (!next_line(f, line))
            throw std::runtime_error("tetgen: short ele file at row " + std::to_string(i));
        const auto tok = tokenise(line);
        if (tok.size() < 5) throw std::runtime_error("tetgen: bad ele row " + std::to_string(i));
        const std::int64_t id = std::stoll(tok[0]);
        if (firstnumber < 0) firstnumber = id;
        EleRecord r;
        for (int k = 0; k < 4; ++k) r.v[k] = std::stoll(tok[1 + k]) - firstnumber;
        r.region = (tok.size() >= 6) ? std::stoll(tok.back()) : 0;
        out.push_back(r);
    }
    return out;
}

// Parse the .1.face file. Each row: "id v0 v1 v2 boundary_marker e1 e2".
// e1, e2 are the indices of the two tetrahedra incident to this face
// (Tetgen's nn output; -1 means no neighbour). We keep all 7 columns.
struct FaceRecord {
    std::array<std::int64_t, 3> v;          // 0-based vertex ids
    std::int64_t                marker;     // interface id from smesh
    std::array<std::int64_t, 2> e;          // 0-based tet ids (or -1)
};
std::vector<FaceRecord> parse_face_file(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("tetgen: cannot open " + path);
    std::string line;
    if (!next_line(f, line)) throw std::runtime_error("tetgen: empty face file");
    const auto header = tokenise(line);
    const std::size_t n_face = std::stoull(header[0]);

    std::vector<FaceRecord> out;
    out.reserve(n_face);
    std::int64_t firstnumber = -1;
    for (std::size_t i = 0; i < n_face; ++i) {
        if (!next_line(f, line))
            throw std::runtime_error("tetgen: short face file at row " + std::to_string(i));
        const auto tok = tokenise(line);
        if (tok.size() < 7) throw std::runtime_error("tetgen: bad face row " + std::to_string(i));
        const std::int64_t id = std::stoll(tok[0]);
        if (firstnumber < 0) firstnumber = id;
        FaceRecord r;
        for (int k = 0; k < 3; ++k) r.v[k] = std::stoll(tok[1 + k]) - firstnumber;
        r.marker = std::stoll(tok[4]);
        // Tetgen 1.6 emits 0-based neighbour ids with -1 as the "no
        // neighbour" sentinel (matches the reference `face[b, 5] >= 0` test).
        const std::int64_t e1 = std::stoll(tok[5]);
        const std::int64_t e2 = std::stoll(tok[6]);
        r.e[0] = (e1 >= 0) ? (e1 - firstnumber) : -1;
        r.e[1] = (e2 >= 0) ? (e2 - firstnumber) : -1;
        out.push_back(r);
    }
    return out;
}

// Volume of a 4×4 matrix [[1, x, y, z], ...] determinant — used to
// determine whether a tet sits on the m1 or m2 side of an interface
// face. Mirrors the reference `tet[:,0] = 1; np.linalg.det(tet)`.
double signed_vol(const Eigen::Matrix4d& M) {
    return M.determinant();
}

}  // namespace

bool mesh_volume(const std::string& path_base,
                 const Coords& xyz,
                 const Triangles& tri,
                 const std::vector<marching_cubes::Interface>& itf,
                 bool do_abaqus_verification) {
    (void)do_abaqus_verification;

    const std::string smesh_path = path_base + ".smesh";
    const std::string node_path  = path_base + ".1.node";
    const std::string ele_path   = path_base + ".1.ele";
    const std::string face_path  = path_base + ".1.face";
    const std::string inp_path   = path_base + ".inp";

    // Per-triangle interface index used as TetGen facet marker.
    std::vector<std::uint32_t> attr(tri.rows(), 0);
    for (std::size_t i = 0; i < itf.size(); ++i)
        for (std::uint32_t k = 0; k < itf[i].count; ++k)
            attr[itf[i].first + k] = static_cast<std::uint32_t>(i);

    io::save_smesh(smesh_path, xyz, tri, &attr);

    auto exe = paths::which("tetgen");
    if (!exe) {
        VOX2TET_PRINT("Tetgen not found. Tet mesh is not created!");
        return false;
    }
    VOX2TET_LOG() << "Tetgen found: " << *exe;

    const std::string cmd = *exe + " -pYA -q2/15 -o/150 -nn -V " + smesh_path;
    VOX2TET_LOG() << "Tetgen call: " << cmd;
    int rc = std::system(cmd.c_str());
    if (rc != 0) {
        VOX2TET_LOG() << "TetGen returned non-zero status: " << rc;
        return false;
    }

    if (!paths::is_exist_files({node_path, ele_path, face_path})) {
        VOX2TET_PRINT("Tet mesh generation problem. Check TetGen output!");
        return false;
    }

    VOX2TET_PRINT("Reading tetgen output");
    const Coords node = parse_node_file(node_path);
    std::vector<EleRecord>  ele  = parse_ele_file(ele_path);
    std::vector<FaceRecord> face = parse_face_file(face_path);

    // For each tet, ele_f[tet] = first interface face attached to it
    // (or -1 if it's an interior tet).
    std::vector<std::int64_t> ele_f(ele.size(), -1);
    for (std::size_t f = 0; f < face.size(); ++f) {
        for (int k = 0; k < 2; ++k) {
            const std::int64_t e = face[f].e[k];
            if (e < 0) continue;
            if (static_cast<std::size_t>(e) >= ele_f.size()) continue;
            ele_f[static_cast<std::size_t>(e)] = static_cast<std::int64_t>(f);
        }
    }

    // Sort elements (stably) by region attribute. Mirror the reference
    // ele[:,-1].argsort() to keep groups contiguous.
    std::vector<std::size_t> order(ele.size());
    for (std::size_t i = 0; i < ele.size(); ++i) order[i] = i;
    std::stable_sort(order.begin(), order.end(),
        [&](std::size_t a, std::size_t b) { return ele[a].region < ele[b].region; });
    std::vector<EleRecord> ele_sorted(ele.size());
    std::vector<std::int64_t> ele_f_sorted(ele.size());
    for (std::size_t i = 0; i < ele.size(); ++i) {
        ele_sorted[i]   = ele[order[i]];
        ele_f_sorted[i] = ele_f[order[i]];
    }
    ele   = std::move(ele_sorted);
    ele_f = std::move(ele_f_sorted);

    // Walk each region (contiguous run of equal ele[i].region) and pick
    // the material id. The reference algorithm picks the unique material
    // shared by all interface faces incident to this region; if both
    // materials are present, it uses the sign of the first interface
    // tet's volume relative to the face vertex order.
    std::vector<std::int64_t> mat_per_tet(ele.size(), -1);
    std::size_t i = 0;
    while (i < ele.size()) {
        std::size_t j = i + 1;
        while (j < ele.size() && ele[j].region == ele[i].region) ++j;

        // Collect interface face ids and corresponding bimaterials for
        // this region.
        std::vector<std::int64_t> face_ids;
        for (std::size_t k = i; k < j; ++k)
            if (ele_f[k] >= 0) face_ids.push_back(ele_f[k]);

        if (face_ids.empty()) {
            // No interface face — The reference silently leaves -1 here. We
            // keep that, but tag with 0 to keep INP output valid.
            for (std::size_t k = i; k < j; ++k) mat_per_tet[k] = 0;
            i = j;
            continue;
        }

        // Bimat[0] / Bimat[1] from the first interface face.
        const std::int64_t first_face = face_ids.front();
        const std::int64_t first_itf_id = face[first_face].marker;
        if (first_itf_id < 0 ||
            static_cast<std::size_t>(first_itf_id) >= itf.size()) {
            for (std::size_t k = i; k < j; ++k) mat_per_tet[k] = 0;
            i = j;
            continue;
        }
        const std::uint32_t m1 = itf[first_itf_id].mat1;
        const std::uint32_t m2 = itf[first_itf_id].mat2;
        bool is_m1 = true, is_m2 = true;
        for (auto fid : face_ids) {
            const std::int64_t a = face[fid].marker;
            if (a < 0 || static_cast<std::size_t>(a) >= itf.size()) {
                is_m1 = false; is_m2 = false; break;
            }
            const auto& I = itf[a];
            if (I.mat1 != m1 && I.mat2 != m1) is_m1 = false;
            if (I.mat1 != m2 && I.mat2 != m2) is_m2 = false;
        }

        std::int64_t mat;
        if (is_m1 && is_m2) {
            // Volume-sign disambiguation. Find the first interface tet in
            // the region.
            std::size_t k_first = i;
            for (std::size_t k = i; k < j; ++k) {
                if (ele_f[k] == first_face) { k_first = k; break; }
            }
            const auto& f_v = face[first_face].v;
            const auto& e_v = ele[k_first].v;
            std::int64_t n4 = -1;
            for (auto v : e_v) {
                if (v != f_v[0] && v != f_v[1] && v != f_v[2]) { n4 = v; break; }
            }
            if (n4 < 0) {
                mat = m1;
            } else {
                Eigen::Matrix4d M;
                for (int r = 0; r < 4; ++r) {
                    const std::int64_t v = (r < 3) ? f_v[r] : n4;
                    M(r, 0) = 1.0;
                    M(r, 1) = node(v, 0);
                    M(r, 2) = node(v, 1);
                    M(r, 3) = node(v, 2);
                }
                const double vol = signed_vol(M);
                mat = (vol < 0) ? m2 : m1;
            }
        } else if (is_m1) {
            mat = m1;
        } else if (is_m2) {
            mat = m2;
        } else {
            VOX2TET_PRINT("ERROR: tetgen: bimat == bimat[0,:] failed");
            mat = m1;
        }

        for (std::size_t k = i; k < j; ++k) mat_per_tet[k] = mat;
        i = j;
    }

    // Materialise ele as MatrixXi for save_inp.
    Eigen::MatrixXi tets(static_cast<Eigen::Index>(ele.size()), 4);
    for (std::size_t k = 0; k < ele.size(); ++k)
        for (int v = 0; v < 4; ++v)
            tets(static_cast<Eigen::Index>(k), v) = static_cast<int>(ele[k].v[v]);

    std::vector<std::int32_t> esets(ele.size());
    for (std::size_t k = 0; k < ele.size(); ++k)
        esets[k] = static_cast<std::int32_t>(mat_per_tet[k]);

    VOX2TET_PRINT("Saving resulting tetrahedral mesh to abaqus .inp format");
    io::save_inp(inp_path, node, tets, &esets);

    return true;
}

}  // namespace vox2tet::tetgen
