#include "vox2tet/mesh/mathx.hpp"

#include <Eigen/LU>          // for Matrix3d::determinant()
#include <algorithm>
#include <cmath>

namespace vox2tet::mathx {

namespace {
constexpr double kRad2Deg = 180.0 / 3.14159265358979323846;
constexpr double kSmallNorm = 1e-7;
}  // namespace

double calc_angle(const Eigen::Vector3d& p0,
                  const Eigen::Vector3d& p1,
                  const Eigen::Vector3d& p2) {
    return calc_angle2(p1 - p0, p2 - p0);
}

double calc_angle2(const Eigen::Vector3d& v1, const Eigen::Vector3d& v2) {
    double n12 = v1.norm() * v2.norm();
    if (n12 < kSmallNorm) return -1.0;
    double c = std::clamp(v1.dot(v2) / n12, -1.0, 1.0);
    return std::acos(c) * kRad2Deg;
}

Eigen::VectorXd calc_angle_v(const MatrixNx3<double>& p0,
                             const MatrixNx3<double>& p1,
                             const MatrixNx3<double>& p2) {
    return calc_angle_v2(p1 - p0, p2 - p0);
}

Eigen::VectorXd calc_angle_v2(const MatrixNx3<double>& v1,
                              const MatrixNx3<double>& v2) {
    const Eigen::Index n = v1.rows();
    Eigen::VectorXd nv1 = v1.rowwise().norm();
    Eigen::VectorXd nv2 = v2.rowwise().norm();
    Eigen::VectorXd out = Eigen::VectorXd::Constant(n, -1.0);
    for (Eigen::Index i = 0; i < n; ++i) {
        double prod = nv1[i] * nv2[i];
        if (prod < kSmallNorm) continue;
        double c = (v1.row(i).dot(v2.row(i))) / prod;
        c = std::clamp(c, -1.0, 1.0);
        out[i] = std::acos(c) * kRad2Deg;
    }
    return out;
}

namespace {
MatrixNx3<double> cross_rows(const MatrixNx3<double>& a, const MatrixNx3<double>& b) {
    const Eigen::Index n = a.rows();
    MatrixNx3<double> out(n, 3);
    for (Eigen::Index i = 0; i < n; ++i) {
        out(i, 0) = a(i, 1) * b(i, 2) - a(i, 2) * b(i, 1);
        out(i, 1) = a(i, 2) * b(i, 0) - a(i, 0) * b(i, 2);
        out(i, 2) = a(i, 0) * b(i, 1) - a(i, 1) * b(i, 0);
    }
    return out;
}
}  // namespace

Eigen::VectorXd calc_dihedral_angle_v(const MatrixNx3<double>& p01,
                                      const MatrixNx3<double>& p02,
                                      const MatrixNx3<double>& p1,
                                      const MatrixNx3<double>& p2) {
    MatrixNx3<double> v0 = p02 - p01;
    MatrixNx3<double> v1 = p1  - p01;
    MatrixNx3<double> v2 = p2  - p01;
    return calc_angle_v2(cross_rows(v1, v0), cross_rows(v2, v0));
}

Eigen::VectorXd calc_dihedral_angle_v_directional(const MatrixNx3<double>& p01,
                                                  const MatrixNx3<double>& p02,
                                                  const MatrixNx3<double>& p1,
                                                  const MatrixNx3<double>& p2) {
    const Eigen::Index n = p01.rows();
    MatrixNx3<double> v0  = p02 - p01;
    MatrixNx3<double> v1  = p1  - p01;
    MatrixNx3<double> v2  = p2  - p01;
    MatrixNx3<double> c1  = cross_rows(v1, v0);
    MatrixNx3<double> c2  = cross_rows(v2, v0);
    Eigen::VectorXd an    = calc_angle_v2(c1, c2);
    for (Eigen::Index i = 0; i < n; ++i) {
        Eigen::Matrix3d M;
        M.row(0) = v0.row(i);
        M.row(1) = c1.row(i);
        M.row(2) = c2.row(i);
        if (M.determinant() > 0.0 && an[i] >= 0) an[i] = 360.0 - an[i];
    }
    return an;
}

Eigen::VectorXd calc_tri_area_v(const MatrixNx3<double>& p1,
                                const MatrixNx3<double>& p2,
                                const MatrixNx3<double>& p3) {
    MatrixNx3<double> a = p3 - p1;
    MatrixNx3<double> b = p3 - p2;
    return 0.5 * cross_rows(a, b).rowwise().norm();
}

ShapeQuality calc_tri_shape_quality_v(const MatrixNx3<double>& p1,
                                      const MatrixNx3<double>& p2,
                                      const MatrixNx3<double>& p3) {
    ShapeQuality r;
    r.a = calc_tri_area_v(p1, p2, p3);
    MatrixNx3<double> v3 = p2 - p1;
    MatrixNx3<double> v2 = p1 - p3;
    MatrixNx3<double> v1 = p3 - p2;
    Eigen::VectorXd sumL = (v1.array() * v1.array()).rowwise().sum() +
                           (v2.array() * v2.array()).rowwise().sum() +
                           (v3.array() * v3.array()).rowwise().sum();
    r.q = 4.0 * std::sqrt(3.0) * r.a.array() / sumL.array();
    return r;
}

ShapeQuality calc_tri_shape_factor(const MatrixNx3<double>& p1,
                                   const MatrixNx3<double>& p2,
                                   const MatrixNx3<double>& p3) {
    ShapeQuality r;
    r.a = calc_tri_area_v(p1, p2, p3);
    Eigen::VectorXd v1 = ((p2 - p1).array().square().rowwise().sum());
    Eigen::VectorXd v2 = ((p3 - p2).array().square().rowwise().sum());
    Eigen::VectorXd v3 = ((p1 - p3).array().square().rowwise().sum());
    // Avoid -Wfloat-equal: clamp to a tiny eps to prevent divide by zero.
    const double eps = 1e-30;
    Eigen::VectorXd denom = v1.array() * v2.array() * v3.array();
    for (Eigen::Index i = 0; i < denom.size(); ++i)
        if (denom[i] < eps) denom[i] = eps;
    r.q = (64.0 * std::sqrt(3.0) / 9.0) *
          (r.a.array().cube()) / denom.array();
    return r;
}

MatrixNx3<double> calc_tri_norm_v(const MatrixNx3<double>& p1,
                                  const MatrixNx3<double>& p2,
                                  const MatrixNx3<double>& p3) {
    MatrixNx3<double> n = cross_rows(p2 - p1, p3 - p1);
    Eigen::VectorXd  ln = n.rowwise().norm();
    for (Eigen::Index i = 0; i < n.rows(); ++i) {
        if (ln[i] > 0) n.row(i) /= ln[i];
    }
    return n;
}

}  // namespace vox2tet::mathx
