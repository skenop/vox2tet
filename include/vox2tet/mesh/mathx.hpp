#pragma once

// Mesh geometry primitives (triangle normals, dihedral angles, etc.).
// All angle results are in DEGREES. Vectorised variants take 3xN rows
// or Nx3 Eigen blocks (NumPy-style broadcasting).

#include <Eigen/Core>

#include "vox2tet/core/types.hpp"

namespace vox2tet::mathx {

// Angle (deg) at p0 spanned by (p1-p0, p2-p0).
double calc_angle(const Eigen::Vector3d& p0,
                  const Eigen::Vector3d& p1,
                  const Eigen::Vector3d& p2);

// Angle (deg) between two 3-vectors.
double calc_angle2(const Eigen::Vector3d& v1, const Eigen::Vector3d& v2);

// Per-row angle (deg) p0—p1—p2. p0.rows() == p1.rows() == p2.rows().
Eigen::VectorXd calc_angle_v(const MatrixNx3<double>& p0,
                             const MatrixNx3<double>& p1,
                             const MatrixNx3<double>& p2);

// Per-row angle (deg) between rows of v1 and v2. -1 sentinel returned
// where ||v|| < 1e-7 (matches calcAngleV2).
Eigen::VectorXd calc_angle_v2(const MatrixNx3<double>& v1,
                              const MatrixNx3<double>& v2);

// Dihedral angle between (p01,p02,p1) and (p01,p02,p2). Per-row.
Eigen::VectorXd calc_dihedral_angle_v(const MatrixNx3<double>& p01,
                                      const MatrixNx3<double>& p02,
                                      const MatrixNx3<double>& p1,
                                      const MatrixNx3<double>& p2);

// Directional variant — orientation from p01→p02. Angles in [0, 360).
Eigen::VectorXd calc_dihedral_angle_v_directional(const MatrixNx3<double>& p01,
                                                  const MatrixNx3<double>& p02,
                                                  const MatrixNx3<double>& p1,
                                                  const MatrixNx3<double>& p2);

// Triangle area per row.
Eigen::VectorXd calc_tri_area_v(const MatrixNx3<double>& p1,
                                const MatrixNx3<double>& p2,
                                const MatrixNx3<double>& p3);

// Bhatia-Lawrence shape-quality (per row) plus area. Returns
// {quality, area} as two N-vectors.
struct ShapeQuality { Eigen::VectorXd q; Eigen::VectorXd a; };
ShapeQuality calc_tri_shape_quality_v(const MatrixNx3<double>& p1,
                                      const MatrixNx3<double>& p2,
                                      const MatrixNx3<double>& p3);

// Abaqus shape factor.
ShapeQuality calc_tri_shape_factor(const MatrixNx3<double>& p1,
                                   const MatrixNx3<double>& p2,
                                   const MatrixNx3<double>& p3);

// Unit normal of (p1, p2, p3) per row. Mirrors calcTriNormV — direction
// of (p2-p1) × (p3-p1).
MatrixNx3<double> calc_tri_norm_v(const MatrixNx3<double>& p1,
                                  const MatrixNx3<double>& p2,
                                  const MatrixNx3<double>& p3);

}  // namespace vox2tet::mathx
