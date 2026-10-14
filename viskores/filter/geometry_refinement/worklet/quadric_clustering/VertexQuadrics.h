//============================================================================
//  The contents of this file are covered by the Viskores license. See
//  LICENSE.txt for details.
//
//  By contributing to this file, all contributors agree to the Developer
//  Certificate of Origin Version 1.1 (DCO 1.1) as stated in DCO.txt.
//============================================================================
#ifndef viskores_worklet_quadric_clustering_VertexQuadrics_h
#define viskores_worklet_quadric_clustering_VertexQuadrics_h

#include <viskores/filter/geometry_refinement/worklet/quadric_clustering/SurfaceMesh.h>

#include <viskores/cont/Algorithm.h>
#include <viskores/cont/ErrorBadValue.h>

#include <cmath>

VISKORES_QUADRIC_CLUSTERING_STRICT_FP_BEGIN

namespace viskores
{
namespace worklet
{
namespace quadric_clustering
{

/// The per-vertex data of Algorithm 1 of the paper.
struct VertexQuadrics
{
  /// Oriented unit normal of every triangle.
  viskores::cont::ArrayHandle<Vec3> TriangleNormals;
  /// `M_v = A_v' + lambda_v I`, the matrix of the cost `(c - v)^T M_v (c - v)`.
  viskores::cont::ArrayHandle<Mat3> M;
  /// `a_v`, a third of the area of the triangles at `v`.
  viskores::cont::ArrayHandle<Real> Area;
  viskores::cont::ArrayHandle<viskores::UInt8> SharpEdges;
};

namespace detail
{

/// `n_f`, and the contributions `(a_f / 3) n_f n_f^T` and `a_f / 3` of a triangle to `A_v` and
/// `a_v`. A degenerate triangle has no normal and is reported.
struct TriangleQuadric : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn triangle,
                                WholeArrayIn points,
                                FieldOut normal,
                                FieldOut outer,
                                FieldOut third,
                                FieldOut degenerate);
  using ExecutionSignature = void(_1, _2, _3, _4, _5, _6);

  template <typename PointsPortal>
  VISKORES_EXEC void operator()(const viskores::Id3& triangle,
                                const PointsPortal& points,
                                Vec3& normal,
                                Mat3& outer,
                                Real& third,
                                viskores::UInt8& degenerate) const
  {
    const Vec3 a = points.Get(triangle[0]);
    const Vec3 x = Cross(Sub(points.Get(triangle[1]), a), Sub(points.Get(triangle[2]), a));
    const Real area = Norm(x) / 2;
    degenerate = (area > 0) ? 0 : 1;
    const Real twice = 2 * area;
    normal = Vec3(x[0] / twice, x[1] / twice, x[2] / twice);
    third = area / 3;
    for (viskores::IdComponent i = 0; i < 3; ++i)
    {
      for (viskores::IdComponent j = 0; j < 3; ++j)
      {
        outer[3 * i + j] = (third * normal[i]) * normal[j];
      }
    }
  }
};

/// An interior edge is sharp when the oriented normals of its triangles make an angle above
/// the feature angle, compared through cosines.
struct SharpEdge : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn edgeTriangles, WholeArrayIn normals, FieldOut sharp);
  using ExecutionSignature = _3(_1, _2);

  VISKORES_CONT explicit SharpEdge(Real cosFeatureAngle)
    : CosFeatureAngle(cosFeatureAngle)
  {
  }

  Real CosFeatureAngle;

  template <typename NormalsPortal>
  VISKORES_EXEC viskores::UInt8 operator()(const viskores::Id2& edgeTriangles,
                                           const NormalsPortal& normals) const
  {
    if (edgeTriangles[1] < 0)
    {
      return 0;
    }
    const Real c = viskores::Min(
      viskores::Max(Dot(normals.Get(edgeTriangles[0]), normals.Get(edgeTriangles[1])), Real(-1)),
      Real(1));
    return (c < this->CosFeatureAngle) ? 1 : 0;
  }
};

/// Sums the quadric of every vertex: the triangle terms in increasing corner order, then the
/// line quadrics of the sharp edges in increasing edge order, the corner term, and the boundary
/// quadrics in increasing edge order, which is the order of the reference implementation.
struct VertexMatrix : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn cornerRange,
                                FieldIn edgeRange,
                                WholeArrayIn corners,
                                WholeArrayIn vertexEdges,
                                WholeArrayIn outer,
                                WholeArrayIn third,
                                WholeArrayIn edges,
                                WholeArrayIn edgeTriangles,
                                WholeArrayIn sharp,
                                WholeArrayIn normals,
                                WholeArrayIn points,
                                FieldOut M,
                                FieldOut area);
  using ExecutionSignature = void(_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13);

  viskores::Id NumberOfTriangles;
  Real Alpha;
  Real FeatureWeight;
  Real BoundaryWeight;
  bool UseFeatures;
  bool UseBoundary;

  template <typename IdPortal,
            typename MatPortal,
            typename RealPortal,
            typename Id2Portal,
            typename FlagPortal,
            typename VecPortal>
  VISKORES_EXEC void operator()(const viskores::Id2& cornerRange,
                                const viskores::Id2& edgeRange,
                                const IdPortal& corners,
                                const IdPortal& vertexEdges,
                                const MatPortal& outer,
                                const RealPortal& third,
                                const Id2Portal& edges,
                                const Id2Portal& edgeTriangles,
                                const FlagPortal& sharp,
                                const VecPortal& normals,
                                const VecPortal& points,
                                Mat3& m,
                                Real& area) const
  {
    Mat3 a(0);
    area = 0;
    for (viskores::Id k = cornerRange[0]; k < cornerRange[1]; ++k)
    {
      const viskores::Id t = corners.Get(k) % this->NumberOfTriangles;
      a = MatAdd(a, outer.Get(t));
      area = area + third.Get(t);
    }
    auto direction = [&](viskores::Id e)
    {
      const viskores::Id2 edge = edges.Get(e);
      const Vec3 d = Sub(points.Get(edge[1]), points.Get(edge[0]));
      const Real length = Norm(d);
      return Vec3(d[0] / length, d[1] / length, d[2] / length);
    };
    if (this->UseFeatures)
    {
      const Real s = this->FeatureWeight * area;
      viskores::IdComponent numSharp = 0;
      for (viskores::Id k = edgeRange[0]; k < edgeRange[1]; ++k)
      {
        const viskores::Id e = vertexEdges.Get(k);
        if (!sharp.Get(e))
        {
          continue;
        }
        ++numSharp;
        const Vec3 t = direction(e);
        for (viskores::IdComponent i = 0; i < 3; ++i)
        {
          for (viskores::IdComponent j = 0; j < 3; ++j)
          {
            const Real p = ((i == j) ? Real(1) : Real(0)) - t[i] * t[j];
            a[3 * i + j] = a[3 * i + j] + s * p;
          }
        }
      }
      if (numSharp >= 3)
      {
        for (viskores::IdComponent i = 0; i < 9; ++i)
        {
          a[i] = a[i] + ((i % 4 == 0) ? s : s * Real(0));
        }
      }
    }
    if (this->UseBoundary)
    {
      const Real s = this->BoundaryWeight * area;
      for (viskores::Id k = edgeRange[0]; k < edgeRange[1]; ++k)
      {
        const viskores::Id e = vertexEdges.Get(k);
        const viskores::Id2 tris = edgeTriangles.Get(e);
        if (tris[1] >= 0)
        {
          continue;
        }
        Vec3 b = Cross(normals.Get(tris[0]), direction(e));
        const Real length = Norm(b);
        b = Vec3(b[0] / length, b[1] / length, b[2] / length);
        for (viskores::IdComponent i = 0; i < 3; ++i)
        {
          for (viskores::IdComponent j = 0; j < 3; ++j)
          {
            a[3 * i + j] = a[3 * i + j] + s * (b[i] * b[j]);
          }
        }
      }
    }
    const Real lambda = this->Alpha * area;
    for (viskores::IdComponent i = 0; i < 9; ++i)
    {
      m[i] = a[i] + ((i % 4 == 0) ? lambda : lambda * Real(0));
    }
  }
};

struct Ranges : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn index, WholeArrayIn offsets, FieldOut range);
  using ExecutionSignature = _3(_1, _2);

  template <typename OffsetsPortal>
  VISKORES_EXEC viskores::Id2 operator()(viskores::Id index, const OffsetsPortal& offsets) const
  {
    return viskores::Id2(offsets.Get(index), offsets.Get(index + 1));
  }
};

} // namespace detail

/// Parameters of the cost. A weight of zero leaves out its term.
struct QuadricParameters
{
  /// The regularisation factor: the weight of vertex `v` is `Alpha a_v`.
  Real Alpha = 0.02;
  /// In degrees.
  Real FeatureAngle = 45;
  Real FeatureWeight = 1;
  Real BoundaryWeight = 1;
  /// A lower bound on the reach of the surface the mesh samples, or 0. When positive, an edge is
  /// sharp if its dihedral angle exceeds `2 omega` of Theorem "detect", in place of
  /// `FeatureAngle`.
  Real Reach = 0;
};

namespace detail
{

/// The length of an edge.
struct EdgeLength : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn edge, WholeArrayIn points, FieldOut length);
  using ExecutionSignature = _3(_1, _2);

  template <typename PointPortal>
  VISKORES_EXEC Real operator()(const viskores::Id2& edge, const PointPortal& points) const
  {
    return Norm(Sub(points.Get(edge[0]), points.Get(edge[1])));
  }
};

/// The largest `|cos|` of the three angles of a triangle.
struct LargestCornerCosine : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn triangle, WholeArrayIn points, FieldOut cosine);
  using ExecutionSignature = _3(_1, _2);

  template <typename PointPortal>
  VISKORES_EXEC Real operator()(const viskores::Id3& triangle, const PointPortal& points) const
  {
    Real largest = 0;
    for (viskores::IdComponent i = 0; i < 3; ++i)
    {
      const Vec3 p = points.Get(triangle[i]);
      const Vec3 u = Sub(points.Get(triangle[(i + 1) % 3]), p);
      const Vec3 v = Sub(points.Get(triangle[(i + 2) % 3]), p);
      largest = viskores::Max(largest, viskores::Abs(Dot(u, v) / (Norm(u) * Norm(v))));
    }
    return largest;
  }
};

} // namespace detail

/// What a lower bound `R` on the reach of the surface the mesh samples determines, as
/// proofs/lib/vqc.py computes it: the longest edge `h`, and `sin omega = kappa h / R` of Theorem
/// "detect", with `kappa = 1 / sqrt(2 (1 - c))` and `c` the largest `|cos|` of the angles of the
/// triangles.
struct ReachBound
{
  Real H;
  Real SinOmega;
};

/// The `ReachBound` of `reach`. Throws if `kappa h >= reach`, where the mesh is too coarse for
/// that reach.
inline ReachBound ComputeReachBound(const SurfaceMesh& mesh, Real reach)
{
  using Algorithm = viskores::cont::Algorithm;
  viskores::cont::Invoker invoke;
  viskores::cont::ArrayHandle<Real> lengths, cosines;
  invoke(detail::EdgeLength{}, mesh.Edges, mesh.Points, lengths);
  invoke(detail::LargestCornerCosine{}, mesh.Triangles, mesh.Points, cosines);
  const Real h = Algorithm::Reduce(lengths, Real(0), viskores::Maximum());
  const Real c = viskores::Min(Algorithm::Reduce(cosines, Real(0), viskores::Maximum()), Real(1));
  const Real kappa = 1 / viskores::Sqrt(2 * (1 - c));
  if (!(kappa * h < reach))
  {
    throw viskores::cont::ErrorBadValue(
      "QuadricClusteringDecimation: the mesh is too coarse for the reach: kappa h must be below "
      "it, with h the longest edge and kappa from the smallest angle.");
  }
  return ReachBound{ h, kappa * h / reach };
}

namespace double_double
{

/// A double-double number `Hi + Lo`, with `|Lo| <= ulp(Hi) / 2`, and the error-free
/// transformations of Dekker and Knuth that operate on them.
struct DoubleDouble
{
  Real Hi;
  Real Lo;
};

inline DoubleDouble QuickTwoSum(Real a, Real b)
{
  const Real s = a + b;
  return { s, b - (s - a) };
}

inline DoubleDouble TwoSum(Real a, Real b)
{
  const Real s = a + b;
  const Real bb = s - a;
  return { s, (a - (s - bb)) + (b - bb) };
}

inline DoubleDouble Add(const DoubleDouble& a, const DoubleDouble& b)
{
  DoubleDouble s = TwoSum(a.Hi, b.Hi);
  const DoubleDouble t = TwoSum(a.Lo, b.Lo);
  s = QuickTwoSum(s.Hi, s.Lo + t.Hi);
  return QuickTwoSum(s.Hi, s.Lo + t.Lo);
}

inline DoubleDouble Multiply(const DoubleDouble& a, const DoubleDouble& b)
{
  const Real p = a.Hi * b.Hi;
  const Real e = std::fma(a.Hi, b.Hi, -p);
  return QuickTwoSum(p, e + (a.Hi * b.Lo + a.Lo * b.Hi));
}

inline DoubleDouble Divide(const DoubleDouble& a, Real b)
{
  const Real q1 = a.Hi / b;
  const Real p = q1 * b;
  const Real e = std::fma(q1, b, -p);
  const DoubleDouble s = TwoSum(a.Hi, -p);
  const Real q2 = (s.Hi + ((s.Lo - e) + a.Lo)) / b;
  return QuickTwoSum(q1, q2);
}

/// The Taylor series of `cos y` (`odd` false) or `sin y` (`odd` true) for `|y| <= pi / 4`,
/// summed until the terms fall below `2^-110` of the sum: every term is the previous one times
/// `-y^2 / (n (n + 1))`.
inline DoubleDouble Taylor(const DoubleDouble& y, bool odd)
{
  const DoubleDouble minusSquare = Multiply(y, { -y.Hi, -y.Lo });
  DoubleDouble term = odd ? y : DoubleDouble{ 1, 0 };
  DoubleDouble sum = term;
  for (int n = odd ? 2 : 1; std::abs(term.Hi) > std::ldexp(std::abs(sum.Hi), -110); n += 2)
  {
    term = Divide(Divide(Multiply(term, minusSquare), Real(n)), Real(n + 1));
    sum = Add(sum, term);
  }
  return sum;
}

} // namespace double_double

/// The cosine of `x` in `[0, pi]`, correctly rounded: evaluated in double-double arithmetic with
/// an error near `2^-100` relative, then rounded once, as `det.cos` of proofs/lib computes it in
/// 50-digit decimal arithmetic. The platform's `cos` is not guaranteed to be correctly rounded,
/// and the threshold of the sharp edges must not depend on the platform.
inline Real CorrectlyRoundedCos(Real x)
{
  // pi and pi / 2 as double-double numbers.
  const double_double::DoubleDouble pi{ 3.141592653589793116, 1.2246467991473532e-16 };
  const double_double::DoubleDouble halfPi{ 1.5707963267948966, 6.123233995736766e-17 };
  const Real quarter = 0.78539816339744830962;
  if (x <= quarter)
  {
    return double_double::Taylor({ x, 0 }, false).Hi;
  }
  if (x < 3 * quarter)
  {
    // cos x = sin(pi / 2 - x).
    return double_double::Taylor(double_double::Add(halfPi, { -x, 0 }), true).Hi;
  }
  // cos x = -cos(pi - x).
  return -double_double::Taylor(double_double::Add(pi, { -x, 0 }), false).Hi;
}

/// Algorithm 1 of the paper: normals, dihedral angles, and the matrices `M_v`.
inline VertexQuadrics ComputeVertexQuadrics(const SurfaceMesh& mesh,
                                            const QuadricParameters& parameters)
{
  viskores::cont::Invoker invoke;
  VertexQuadrics result;
  viskores::cont::ArrayHandle<Mat3> outer;
  viskores::cont::ArrayHandle<Real> third;
  viskores::cont::ArrayHandle<viskores::UInt8> degenerate;
  invoke(detail::TriangleQuadric{},
         mesh.Triangles,
         mesh.Points,
         result.TriangleNormals,
         outer,
         third,
         degenerate);
  if (viskores::cont::Algorithm::Reduce(degenerate, viskores::Id(0)) > 0)
  {
    throw viskores::cont::ErrorBadValue(
      "QuadricClusteringDecimation: the mesh has a triangle of zero area.");
  }

  const bool useFeatures = parameters.FeatureWeight > 0;
  // The reference evaluates (deg * pi) / 180 and takes its correctly rounded cosine. With a
  // reach, the threshold is cos(2 omega) = 1 - 2 sin^2(omega), without trigonometric functions.
  if (!(parameters.FeatureAngle >= 0 && parameters.FeatureAngle <= 180))
  {
    throw viskores::cont::ErrorBadValue(
      "QuadricClusteringDecimation: the feature angle must be between 0 and 180 degrees.");
  }
  Real cosFeatureAngle = CorrectlyRoundedCos(parameters.FeatureAngle * viskores::Pi() / 180);
  if (parameters.Reach > 0)
  {
    const Real sinOmega = ComputeReachBound(mesh, parameters.Reach).SinOmega;
    cosFeatureAngle = 1 - 2 * (sinOmega * sinOmega);
  }
  invoke(detail::SharpEdge(useFeatures ? cosFeatureAngle : Real(-2)),
         mesh.EdgeTriangles,
         result.TriangleNormals,
         result.SharpEdges);

  viskores::cont::ArrayHandle<viskores::Id2> cornerRange, edgeRange;
  const viskores::cont::ArrayHandleIndex vertices(mesh.NumberOfPoints);
  invoke(detail::Ranges{}, vertices, mesh.VertexCornerOffsets, cornerRange);
  invoke(detail::Ranges{}, vertices, mesh.VertexEdgeOffsets, edgeRange);
  detail::VertexMatrix vertexMatrix;
  vertexMatrix.NumberOfTriangles = mesh.NumberOfTriangles;
  vertexMatrix.Alpha = parameters.Alpha;
  vertexMatrix.FeatureWeight = parameters.FeatureWeight;
  vertexMatrix.BoundaryWeight = parameters.BoundaryWeight;
  vertexMatrix.UseFeatures = useFeatures;
  vertexMatrix.UseBoundary = parameters.BoundaryWeight > 0 && mesh.HasRings && mesh.HasBoundary;
  invoke(vertexMatrix,
         cornerRange,
         edgeRange,
         mesh.VertexCorners,
         mesh.VertexEdges,
         outer,
         third,
         mesh.Edges,
         mesh.EdgeTriangles,
         result.SharpEdges,
         result.TriangleNormals,
         mesh.Points,
         result.M,
         result.Area);
  return result;
}

}
}
} // namespace viskores::worklet::quadric_clustering

VISKORES_QUADRIC_CLUSTERING_STRICT_FP_END

#endif // viskores_worklet_quadric_clustering_VertexQuadrics_h
