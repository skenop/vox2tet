#include "vox2tet/tetmesh/cdt_mmg_runner.hpp"

#include "vox2tet/core/log.hpp"
#include "vox2tet/io/mesh_io.hpp"
#include "vox2tet/tetmesh/steiner_cdt.hpp"

#include "mmg/mmg3d/libmmg3d.h"

#include <Eigen/Dense>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <unordered_map>
#include <vector>

namespace vox2tet::tetmesh {

namespace {

using Vec3 = Eigen::Vector3d;

// ---------------------------------------------------------------------------
// Uniform grid over the input surface triangles, used to answer "which
// input triangle does this point lie on?" — both to detect the
// constrained (surface) faces of the CDT output and to attribute them
// to interfaces. Cells are keyed by packed integer coordinates.
// ---------------------------------------------------------------------------
class TriGrid {
public:
    TriGrid(const Coords& xyz, const Triangles& tri, double eps)
        : xyz_(xyz), tri_(tri), eps_(eps) {
        double total_len = 0.0;
        for (Eigen::Index t = 0; t < tri.rows(); ++t)
            for (int k = 0; k < 3; ++k)
                total_len += (xyz.row(tri(t, k)) -
                              xyz.row(tri(t, (k + 1) % 3))).norm();
        cell_ = std::max(1e-12, total_len / (3.0 * (double)tri.rows()));

        normals_.resize(tri.rows());
        for (Eigen::Index t = 0; t < tri.rows(); ++t) {
            const Vec3 a = xyz.row(tri(t, 0)).transpose();
            const Vec3 b = xyz.row(tri(t, 1)).transpose();
            const Vec3 c = xyz.row(tri(t, 2)).transpose();
            normals_[(std::size_t)t] = (b - a).cross(c - a);
            const double n2 = normals_[(std::size_t)t].norm();
            if (n2 > 0.0) normals_[(std::size_t)t] /= n2;

            const Vec3 lo =
                a.cwiseMin(b).cwiseMin(c) - Vec3::Constant(eps);
            const Vec3 hi =
                a.cwiseMax(b).cwiseMax(c) + Vec3::Constant(eps);
            const auto ilo = cell_of(lo), ihi = cell_of(hi);
            for (std::int64_t x = ilo[0]; x <= ihi[0]; ++x)
                for (std::int64_t y = ilo[1]; y <= ihi[1]; ++y)
                    for (std::int64_t z = ilo[2]; z <= ihi[2]; ++z)
                        cells_[pack(x, y, z)].push_back((std::uint32_t)t);
        }
    }

    // Input triangle containing point `p` (within the plane/edge
    // tolerance), or -1. The `extra` points, when given, must lie on
    // the matched triangle's plane too.
    std::int64_t find(const Vec3& p, const Vec3* extra = nullptr,
                      int n_extra = 0) const {
        const auto ic = cell_of(p);
        const auto it = cells_.find(pack(ic[0], ic[1], ic[2]));
        if (it == cells_.end()) return -1;
        for (const std::uint32_t t : it->second) {
            if (!on_triangle(p, t)) continue;
            bool ok = true;
            for (int k = 0; k < n_extra && ok; ++k)
                ok = std::abs(plane_dist(extra[k], t)) <= eps_;
            if (ok) return (std::int64_t)t;
        }
        return -1;
    }

    const Vec3& normal(std::int64_t t) const {
        return normals_[(std::size_t)t];
    }
    Vec3 base_point(std::int64_t t) const {
        return xyz_.row(tri_((Eigen::Index)t, 0)).transpose();
    }

private:
    std::array<std::int64_t, 3> cell_of(const Vec3& p) const {
        return {(std::int64_t)std::floor(p[0] / cell_),
                (std::int64_t)std::floor(p[1] / cell_),
                (std::int64_t)std::floor(p[2] / cell_)};
    }
    static std::uint64_t pack(std::int64_t x, std::int64_t y, std::int64_t z) {
        auto u = [](std::int64_t v) {
            return (std::uint64_t)(v + ((std::int64_t)1 << 20)) & 0x1FFFFF;
        };
        return (u(x) << 42) | (u(y) << 21) | u(z);
    }

    double plane_dist(const Vec3& p, std::uint32_t t) const {
        const Vec3 a = xyz_.row(tri_(t, 0)).transpose();
        return (p - a).dot(normals_[t]);
    }

    bool on_triangle(const Vec3& p, std::uint32_t t) const {
        if (std::abs(plane_dist(p, t)) > eps_) return false;
        const Vec3 n = normals_[t];
        for (int k = 0; k < 3; ++k) {
            const Vec3 a = xyz_.row(tri_(t, k)).transpose();
            const Vec3 b = xyz_.row(tri_(t, (k + 1) % 3)).transpose();
            // Signed in-plane distance of p from edge (a,b), positive
            // on the triangle's interior side; tolerance scales with
            // the edge length so eps_ stays a length unit.
            if ((b - a).cross(p - a).dot(n) < -eps_ * (b - a).norm())
                return false;
        }
        return true;
    }

    const Coords& xyz_;
    const Triangles& tri_;
    double eps_;
    double cell_ = 1.0;
    std::vector<Vec3> normals_;
    std::unordered_map<std::uint64_t, std::vector<std::uint32_t>> cells_;
};

struct TripleKey {
    std::array<std::uint32_t, 3> v;
    bool operator==(const TripleKey& o) const { return v == o.v; }
};
struct TripleHash {
    std::size_t operator()(const TripleKey& k) const {
        std::uint64_t h = 1469598103934665603ull;
        for (const std::uint32_t x : k.v) {
            h ^= x;
            h *= 1099511628211ull;
        }
        return (std::size_t)h;
    }
};

TripleKey make_key(std::uint32_t a, std::uint32_t b, std::uint32_t c) {
    TripleKey k{{a, b, c}};
    std::sort(k.v.begin(), k.v.end());
    return k;
}

double tet_signed_volume(const Coords& x, const std::uint32_t* n) {
    const Vec3 a = x.row(n[0]).transpose();
    return (Vec3(x.row(n[1]).transpose()) - a)
        .cross(Vec3(x.row(n[2]).transpose()) - a)
        .dot(Vec3(x.row(n[3]).transpose()) - a);
}

}  // namespace

bool mesh_volume(const Settings& s,
                 const std::string& path_base,
                 const Coords& xyz,
                 const Triangles& tri,
                 const std::vector<marching_cubes::Interface>& itf,
                 CdtMmgStats* stats) {
    CdtMmgStats st;

    // Per input triangle: owning interface index.
    std::vector<std::uint32_t> attr(tri.rows(), 0);
    for (std::size_t i = 0; i < itf.size(); ++i)
        for (std::uint32_t k = 0; k < itf[i].count; ++k)
            attr[itf[i].first + k] = (std::uint32_t)i;

    // ---- 1. Steiner CDT ---------------------------------------------
    std::vector<double> coords((std::size_t)xyz.rows() * 3);
    for (Eigen::Index i = 0; i < xyz.rows(); ++i)
        for (int k = 0; k < 3; ++k)
            coords[(std::size_t)i * 3 + k] = xyz(i, k);
    std::vector<std::uint32_t> tris((std::size_t)tri.rows() * 3);
    for (Eigen::Index i = 0; i < tri.rows(); ++i)
        for (int k = 0; k < 3; ++k)
            tris[(std::size_t)i * 3 + k] = tri(i, k);

    SteinerCdtResult cdt;
    if (!run_steiner_cdt(coords.data(), (std::size_t)xyz.rows(), tris.data(),
                         (std::size_t)tri.rows(), s.mmg_verbose > 0, cdt)) {
        VOX2TET_PRINT("cdt: Steiner CDT failed: " + cdt.error);
        return false;
    }
    if (!cdt.face_recovery_ok)
        VOX2TET_LOG() << "cdt: face recovery reported issues — continuing";
    if (!cdt.snap_ok)
        VOX2TET_LOG() << "cdt: FP snap left near-degenerate tets — continuing";

    const std::size_t nv = cdt.xyz.size() / 3;
    const std::size_t nt = cdt.tets.size() / 4;
    st.n_nodes_cdt = nv;
    st.n_tets_cdt  = nt;
    st.n_steiner   = cdt.n_steiner;

    Coords node((Eigen::Index)nv, 3);
    for (std::size_t i = 0; i < nv; ++i)
        for (int k = 0; k < 3; ++k)
            node((Eigen::Index)i, k) = cdt.xyz[i * 3 + k];

    // Positive orientation for every tet (MMG and INP convention).
    for (std::size_t t = 0; t < nt; ++t) {
        std::uint32_t* n = cdt.tets.data() + t * 4;
        if (tet_signed_volume(node, n) < 0.0) std::swap(n[2], n[3]);
    }

    // ---- 2. Constrained (surface) faces of the tet mesh --------------
    // Shared-face map: sorted vertex triple -> incident tets.
    std::unordered_map<TripleKey, std::array<std::int64_t, 2>, TripleHash>
        face_tets;
    face_tets.reserve(nt * 2);
    for (std::size_t t = 0; t < nt; ++t) {
        const std::uint32_t* n = cdt.tets.data() + t * 4;
        for (int k = 0; k < 4; ++k) {
            const auto key =
                make_key(n[(k + 1) & 3], n[(k + 2) & 3], n[(k + 3) & 3]);
            auto [it, fresh] =
                face_tets.try_emplace(key, std::array<std::int64_t, 2>{-1, -1});
            it->second[fresh ? 0 : 1] = (std::int64_t)t;
        }
    }

    // CDT recovers the input surface exactly as a union of tet faces,
    // so a face is on the surface iff all its corners and its centroid
    // lie on one input triangle (tolerance >> snap error, << mesh
    // scale). The matched triangle also provides the interface id and
    // the orientation reference for material classification.
    const double eps = 1e-6;
    TriGrid grid(xyz, tri, eps);

    struct IfaceFace {
        TripleKey key;
        std::uint32_t iface;
        std::int64_t input_tri;
    };
    std::vector<IfaceFace> iface_faces;
    double area_recovered = 0.0;
    for (const auto& [key, tets2] : face_tets) {
        const Vec3 a = node.row(key.v[0]).transpose();
        const Vec3 b = node.row(key.v[1]).transpose();
        const Vec3 c = node.row(key.v[2]).transpose();
        const Vec3 centroid = (a + b + c) / 3.0;
        const Vec3 extra[3] = {a, b, c};
        const std::int64_t t_in = grid.find(centroid, extra, 3);
        if (t_in < 0) continue;
        iface_faces.push_back({key, attr[(std::size_t)t_in], t_in});
        area_recovered += 0.5 * (b - a).cross(c - a).norm();
    }
    st.n_iface_faces = iface_faces.size();

    double area_input = 0.0;
    for (Eigen::Index t = 0; t < tri.rows(); ++t) {
        const Vec3 a = xyz.row(tri(t, 0)).transpose();
        const Vec3 b = xyz.row(tri(t, 1)).transpose();
        const Vec3 c = xyz.row(tri(t, 2)).transpose();
        area_input += 0.5 * (b - a).cross(c - a).norm();
    }
    st.area_ratio = (area_input > 0.0) ? area_recovered / area_input : 0.0;
    if (std::abs(st.area_ratio - 1.0) > 1e-6)
        VOX2TET_PRINT("cdt: WARNING: recovered interface area is " +
                      std::to_string(st.area_ratio) +
                      " of the input surface area — material "
                      "classification may be unreliable");

    // ---- 3. Material regions (flood fill) ----------------------------
    std::unordered_map<TripleKey, std::size_t, TripleHash> is_iface;
    is_iface.reserve(iface_faces.size());
    for (std::size_t i = 0; i < iface_faces.size(); ++i)
        is_iface.emplace(iface_faces[i].key, i);

    std::vector<std::int64_t> region(nt, -1);
    std::size_t n_regions = 0;
    std::vector<std::size_t> stack;
    for (std::size_t seed = 0; seed < nt; ++seed) {
        if (region[seed] >= 0) continue;
        const auto r = (std::int64_t)n_regions++;
        region[seed] = r;
        stack.assign(1, seed);
        while (!stack.empty()) {
            const std::size_t t = stack.back();
            stack.pop_back();
            const std::uint32_t* n = cdt.tets.data() + t * 4;
            for (int k = 0; k < 4; ++k) {
                const auto key =
                    make_key(n[(k + 1) & 3], n[(k + 2) & 3], n[(k + 3) & 3]);
                if (is_iface.count(key)) continue;  // interface: barrier
                const auto& t2 = face_tets.at(key);
                const std::int64_t o = (t2[0] == (std::int64_t)t) ? t2[1]
                                                                  : t2[0];
                if (o >= 0 && region[(std::size_t)o] < 0) {
                    region[(std::size_t)o] = r;
                    stack.push_back((std::size_t)o);
                }
            }
        }
    }
    st.n_regions = n_regions;

    // Vote materials per region. Convention mirrors the tetgen driver:
    // a tet on the positive side of the input triangle's normal gets
    // the interface's mat1, the other side mat2 (same sign convention
    // as its `det([1,f0;1,f1;1,f2;1,n4]) >= 0 -> m1` rule).
    std::vector<std::unordered_map<std::uint32_t, std::size_t>> votes(
        n_regions);
    for (const auto& f : iface_faces) {
        const auto& t2 = face_tets.at(f.key);
        const Vec3 n_in = grid.normal(f.input_tri);
        const Vec3 p0   = grid.base_point(f.input_tri);
        const auto& I   = itf[f.iface];
        for (int side = 0; side < 2; ++side) {
            const std::int64_t t = t2[side];
            if (t < 0) continue;
            const std::uint32_t* tn = cdt.tets.data() + (std::size_t)t * 4;
            Vec3 ct = Vec3::Zero();
            for (int k = 0; k < 4; ++k)
                ct += node.row(tn[k]).transpose();
            ct /= 4.0;
            const std::uint32_t m =
                ((ct - p0).dot(n_in) >= 0.0) ? I.mat1 : I.mat2;
            ++votes[(std::size_t)region[(std::size_t)t]][m];
        }
    }
    std::vector<std::int32_t> region_mat(n_regions, 0);
    for (std::size_t r = 0; r < n_regions; ++r) {
        std::size_t best = 0, total = 0;
        for (const auto& [m, c] : votes[r]) {
            total += c;
            if (c > best) {
                best = c;
                region_mat[r] = (std::int32_t)m;
            }
        }
        if (total == 0)
            VOX2TET_LOG() << "cdt: region " << r
                          << " has no interface face — material set to 0";
        // A few stray votes are expected from near-degenerate tets
        // whose centroid sits essentially on the interface plane; more
        // than 5% dissent means the classification is suspect.
        else if ((total - best) * 20 > total)
            ++st.n_vote_conflicts;
    }
    if (st.n_vote_conflicts)
        VOX2TET_PRINT("cdt: WARNING: " +
                      std::to_string(st.n_vote_conflicts) +
                      " regions with conflicting material votes");

    // ---- 4. MMG3D interior quality optimization ----------------------
    Eigen::MatrixXi out_tets((Eigen::Index)nt, 4);
    for (std::size_t t = 0; t < nt; ++t)
        for (int k = 0; k < 4; ++k)
            out_tets((Eigen::Index)t, k) = (int)cdt.tets[t * 4 + k];
    std::vector<std::int32_t> esets(nt);
    for (std::size_t t = 0; t < nt; ++t)
        esets[t] = region_mat[(std::size_t)region[t]];
    Coords out_node = node;

    if (s.do_mmg_optim) {
        st.mmg_ran = true;
        MMG5_pMesh mesh = nullptr;
        MMG5_pSol  met  = nullptr;
        MMG3D_Init_mesh(MMG5_ARG_start, MMG5_ARG_ppMesh, &mesh,
                        MMG5_ARG_ppMet, &met, MMG5_ARG_end);

        bool ok =
            MMG3D_Set_meshSize(mesh, (MMG5_int)nv, (MMG5_int)nt, 0,
                               (MMG5_int)iface_faces.size(), 0, 0) == 1;
        for (std::size_t i = 0; ok && i < nv; ++i)
            ok = MMG3D_Set_vertex(mesh, node((Eigen::Index)i, 0),
                                  node((Eigen::Index)i, 1),
                                  node((Eigen::Index)i, 2), 0,
                                  (MMG5_int)(i + 1)) == 1;
        for (std::size_t t = 0; ok && t < nt; ++t) {
            const std::uint32_t* n = cdt.tets.data() + t * 4;
            ok = MMG3D_Set_tetrahedron(
                     mesh, (MMG5_int)(n[0] + 1), (MMG5_int)(n[1] + 1),
                     (MMG5_int)(n[2] + 1), (MMG5_int)(n[3] + 1),
                     esets[t], (MMG5_int)(t + 1)) == 1;
        }
        for (std::size_t i = 0; ok && i < iface_faces.size(); ++i) {
            const auto& v = iface_faces[i].key.v;
            ok = MMG3D_Set_triangle(mesh, (MMG5_int)(v[0] + 1),
                                    (MMG5_int)(v[1] + 1),
                                    (MMG5_int)(v[2] + 1),
                                    (MMG5_int)iface_faces[i].iface,
                                    (MMG5_int)(i + 1)) == 1 &&
                 MMG3D_Set_requiredTriangle(mesh, (MMG5_int)(i + 1)) == 1;
        }

        // The surface (outer boundary and material interfaces) is
        // vox2tet's own remeshed result — MMG must not touch it:
        // nosurf freezes surface points, required triangles keep the
        // triangulation verbatim. optim improves the interior keeping
        // local sizes, so the volume grading follows the surface
        // sizing that the remesh loop established.
        ok = ok &&
             MMG3D_Set_iparameter(mesh, met, MMG3D_IPARAM_verbose,
                                  s.mmg_verbose) == 1 &&
             MMG3D_Set_iparameter(mesh, met, MMG3D_IPARAM_nosurf, 1) == 1 &&
             MMG3D_Set_iparameter(mesh, met, MMG3D_IPARAM_optim, 1) == 1;

        int ret = MMG5_STRONGFAILURE;
        if (ok) ret = MMG3D_mmg3dlib(mesh, met);

        if (ok && ret != MMG5_STRONGFAILURE) {
            st.mmg_ok = (ret == MMG5_SUCCESS);
            MMG5_int np = 0, ne = 0, ntri = 0, na = 0;
            MMG3D_Get_meshSize(mesh, &np, &ne, nullptr, &ntri, nullptr, &na);
            out_node.resize((Eigen::Index)np, 3);
            for (MMG5_int i = 1; i <= np; ++i) {
                double x, y, z;
                MMG5_int ref;
                int corner, required;
                MMG3D_Get_vertex(mesh, &x, &y, &z, &ref, &corner, &required);
                out_node(i - 1, 0) = x;
                out_node(i - 1, 1) = y;
                out_node(i - 1, 2) = z;
            }
            out_tets.resize((Eigen::Index)ne, 4);
            esets.assign((std::size_t)ne, 0);
            for (MMG5_int i = 1; i <= ne; ++i) {
                MMG5_int v0, v1, v2, v3, ref;
                int required;
                MMG3D_Get_tetrahedron(mesh, &v0, &v1, &v2, &v3, &ref,
                                      &required);
                out_tets(i - 1, 0) = (int)(v0 - 1);
                out_tets(i - 1, 1) = (int)(v1 - 1);
                out_tets(i - 1, 2) = (int)(v2 - 1);
                out_tets(i - 1, 3) = (int)(v3 - 1);
                esets[(std::size_t)(i - 1)] = (std::int32_t)ref;
            }
        } else {
            VOX2TET_PRINT("cdt: MMG3D optimization failed — writing the "
                          "unoptimized CDT mesh");
        }
        MMG3D_Free_all(MMG5_ARG_start, MMG5_ARG_ppMesh, &mesh,
                       MMG5_ARG_ppMet, &met, MMG5_ARG_end);
    }

    st.n_nodes_final = (std::size_t)out_node.rows();
    st.n_tets_final  = (std::size_t)out_tets.rows();

    VOX2TET_LOG() << "cdt_mmg: surface " << xyz.rows() << " verts / "
                  << tri.rows() << " tris -> CDT " << st.n_nodes_cdt
                  << " verts (" << st.n_steiner << " Steiner) / "
                  << st.n_tets_cdt << " tets, " << st.n_iface_faces
                  << " interface faces (area ratio " << st.area_ratio
                  << "), " << st.n_regions << " regions"
                  << (st.mmg_ran
                          ? (std::string(" -> MMG ") +
                             (st.mmg_ok ? "ok" : "partial") + " " +
                             std::to_string(st.n_nodes_final) + " verts / " +
                             std::to_string(st.n_tets_final) + " tets")
                          : std::string());

    VOX2TET_PRINT("Saving resulting tetrahedral mesh to abaqus .inp format");
    io::save_inp(path_base + ".inp", out_node, out_tets, &esets);

    if (stats) {
        st.nodes = std::move(out_node);
        st.tets  = std::move(out_tets);
        st.mats  = std::move(esets);
        *stats   = std::move(st);
    }
    return true;
}

}  // namespace vox2tet::tetmesh
