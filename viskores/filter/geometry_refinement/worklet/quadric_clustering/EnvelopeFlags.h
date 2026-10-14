//============================================================================
//  The contents of this file are covered by the Viskores license. See
//  LICENSE.txt for details.
//
//  By contributing to this file, all contributors agree to the Developer
//  Certificate of Origin Version 1.1 (DCO 1.1) as stated in DCO.txt.
//============================================================================
#ifndef viskores_worklet_quadric_clustering_EnvelopeFlags_h
#define viskores_worklet_quadric_clustering_EnvelopeFlags_h

#include <viskores/filter/geometry_refinement/worklet/quadric_clustering/SurfaceMesh.h>
#include <viskores/filter/geometry_refinement/worklet/quadric_clustering/TopologyFlags.h>

#include <viskores/cont/Algorithm.h>
#include <viskores/cont/ArrayHandleConstant.h>
#include <viskores/cont/ArrayHandleGroupVec.h>
#include <viskores/cont/ArrayHandleIndex.h>
#include <viskores/cont/ArrayHandlePermutation.h>
#include <viskores/cont/ExecutionObjectBase.h>

#include <viskores/worklet/spatialstructure/BoundingVolumeHierarchy.h>

#include <limits>

VISKORES_QUADRIC_CLUSTERING_STRICT_FP_BEGIN

namespace viskores
{
namespace worklet
{
namespace quadric_clustering
{

/// The unit roundoff of double precision.
constexpr Real UnitRoundoff = 1.1102230246251565404236316680908203125e-16;

/// The data of a triangle mesh queried for distances: its vertices, its triangles and a
/// hierarchy over them.
struct DistanceMesh
{
  viskores::cont::ArrayHandle<Vec3> Points;
  viskores::cont::ArrayHandle<viskores::Id3> Triangles;
  viskores::worklet::spatialstructure::BoundingVolumeHierarchy Hierarchy;
  /// The largest absolute coordinate of the points.
  Real MaxAbsCoordinate = 0;
};

namespace detail
{

/// `nan_to_num` of numpy: NaN to 0, infinities to the largest finite values.
VISKORES_EXEC_CONT inline Real NanToNum(Real x)
{
  if (x != x)
  {
    return 0;
  }
  if (x == std::numeric_limits<Real>::infinity())
  {
    return std::numeric_limits<Real>::max();
  }
  if (x == -std::numeric_limits<Real>::infinity())
  {
    return -std::numeric_limits<Real>::max();
  }
  return x;
}

VISKORES_EXEC_CONT inline Real Clip(Real x, Real lo, Real hi)
{
  return viskores::Min(viskores::Max(x, lo), hi);
}

/// `|p - (base + v (x - base) + w (y - base))|`, evaluated in this order.
VISKORES_EXEC_CONT inline Real Evaluate(const Vec3& p,
                                        const Vec3& base,
                                        const Vec3& x,
                                        const Vec3& y,
                                        Real v,
                                        Real w)
{
  const Vec3 q = Add(Add(base, Scale(v, Sub(x, base))), Scale(w, Sub(y, base)));
  return Norm(Sub(p, q));
}

} // namespace detail

/// The computed distance from `p` to a point of the triangle `abc` of the form of Lemma
/// "One-sided evaluation" (i): the nearest of the clamped projections onto the three edges and
/// onto the plane. Every candidate is a point of the triangle up to the rounding of its
/// evaluation, whatever rounding went into its parameters.
VISKORES_EXEC_CONT inline Real FeasiblePointDistance(const Vec3& p,
                                                     const Vec3& a,
                                                     const Vec3& b,
                                                     const Vec3& c)
{
  Real best = std::numeric_limits<Real>::infinity();
  const Vec3* ends[3][2] = { { &a, &b }, { &b, &c }, { &c, &a } };
  for (int e = 0; e < 3; ++e)
  {
    const Vec3& x = *ends[e][0];
    const Vec3& y = *ends[e][1];
    const Vec3 d = Sub(y, x);
    const Real t = detail::Clip(detail::NanToNum(Dot(Sub(p, x), d) / Dot(d, d)), 0, 1);
    best = viskores::Min(best, detail::Evaluate(p, x, y, x, t, 0));
  }
  const Vec3 ab = Sub(b, a), ac = Sub(c, a), ap = Sub(p, a);
  const Real d00 = Dot(ab, ab), d01 = Dot(ab, ac), d11 = Dot(ac, ac);
  const Real d20 = Dot(ap, ab), d21 = Dot(ap, ac);
  const Real den = d00 * d11 - d01 * d01;
  Real v = detail::NanToNum((d11 * d20 - d01 * d21) / den);
  Real w = detail::NanToNum((d00 * d21 - d01 * d20) / den);
  v = detail::Clip(v, 0, 1);
  w = viskores::Max(w, Real(0));
  if (v + w > 1)
  {
    w = 1 - v;
  }
  return viskores::Min(best, detail::Evaluate(p, a, b, c, v, w));
}

/// An upper bound on the distance from `p` to the triangle `abc`: `(1 + 4u) d + 32 u S` with
/// `d = FeasiblePointDistance` and `S` the largest absolute coordinate involved.
VISKORES_EXEC_CONT inline Real PointTriangleDistanceUpper(const Vec3& p,
                                                          const Vec3& a,
                                                          const Vec3& b,
                                                          const Vec3& c)
{
  Real s = 0;
  for (int i = 0; i < 3; ++i)
  {
    s = viskores::Max(s, viskores::Abs(p[i]));
    s = viskores::Max(s, viskores::Abs(a[i]));
    s = viskores::Max(s, viskores::Abs(b[i]));
    s = viskores::Max(s, viskores::Abs(c[i]));
  }
  return (1 + 4 * UnitRoundoff) * FeasiblePointDistance(p, a, b, c) + 32 * UnitRoundoff * s;
}

/// Execution-side view of a `DistanceMesh`.
template <typename VecPortal, typename Id3Portal, typename HierarchyPortals>
struct DistanceMeshPortals
{
  VecPortal Points;
  Id3Portal Triangles;
  HierarchyPortals Hierarchy;
  Real MaxAbsCoordinate;

  VISKORES_EXEC Real TriangleBound(const Vec3& p, viskores::Id t) const
  {
    const viskores::Id3 tri = this->Triangles.Get(t);
    return PointTriangleDistanceUpper(
      p, this->Points.Get(tri[0]), this->Points.Get(tri[1]), this->Points.Get(tri[2]));
  }

  struct SkipNone
  {
    VISKORES_EXEC bool operator()(viskores::Id) const { return false; }
  };

  /// The smallest upper bound `PointTriangleDistanceUpper` over all the triangles: the
  /// computed distance from `p` to the mesh. A triangle is skipped only if a lower bound on its
  /// distance exceeds the best bound found, so it could not lower the minimum.
  VISKORES_EXEC Real Distance(const Vec3& p, bool& complete) const
  {
    return this->DistanceSkipping(p, SkipNone{}, std::numeric_limits<Real>::infinity(), complete);
  }

  /// `Distance` when it is at most `cutoff`, and otherwise some value above `cutoff`.
  VISKORES_EXEC Real DistanceUpTo(const Vec3& p, Real cutoff, bool& complete) const
  {
    return this->DistanceSkipping(p, SkipNone{}, cutoff, complete);
  }

  /// `DistanceUpTo` over the triangles `t` with `skip(t)` false. The upper bounds are not below
  /// the distance to the boxes, so the query of the hierarchy returns their exact minimum when
  /// that is at most `cutoff`.
  template <typename Skip>
  VISKORES_EXEC Real DistanceSkipping(const Vec3& p,
                                      const Skip& skip,
                                      Real cutoff,
                                      bool& complete) const
  {
    return this->Hierarchy.FindNearest(
      p,
      [&](viskores::Id t)
      { return skip(t) ? std::numeric_limits<Real>::infinity() : this->TriangleBound(p, t); },
      complete,
      cutoff);
  }

  /// Whether some triangle of the mesh has its largest upper bound to the three points `s`,
  /// plus `margin`, at most `bound`: then every point of their convex hull is within `bound`,
  /// since the distance to a convex set is convex.
  VISKORES_EXEC bool WithinOneTriangle(const Vec3* s, Real bound, Real margin, bool& complete) const
  {
    return this->WithinOneTriangleSkipping(s, bound, margin, SkipNone{}, complete);
  }

  /// `WithinOneTriangle` over the triangles `t` with `skip(t)` false.
  template <typename Skip>
  VISKORES_EXEC bool WithinOneTriangleSkipping(const Vec3* s,
                                               Real bound,
                                               Real margin,
                                               const Skip& skip,
                                               bool& complete) const
  {
    // A useful triangle is within the bound of every point, so its box meets the box of every
    // point grown by the bound; grow them a little more, since a superset of candidates changes
    // nothing. A subtree whose box misses one of them holds no useful triangle.
    const Real grow = bound * (1 + Real(1e-9)) + 64 * UnitRoundoff * this->MaxAbsCoordinate;
    Vec3 lo[3], hi[3];
    for (int k = 0; k < 3; ++k)
    {
      for (int i = 0; i < 3; ++i)
      {
        lo[k][i] = s[k][i] - grow;
        hi[k][i] = s[k][i] + grow;
      }
    }
    bool found = false;
    complete = this->Hierarchy.Traverse(
      [&](viskores::Id node)
      {
        const Vec3 l = this->Hierarchy.Lower.Get(node);
        const Vec3 u = this->Hierarchy.Upper.Get(node);
        for (int k = 0; k < 3; ++k)
        {
          for (int i = 0; i < 3; ++i)
          {
            if (l[i] > hi[k][i] || u[i] < lo[k][i])
            {
              return false;
            }
          }
        }
        return true;
      },
      [&](viskores::Id t)
      {
        if (skip(t))
        {
          return true;
        }
        Real worst = 0;
        for (int k = 0; k < 3; ++k)
        {
          worst = viskores::Max(worst, this->TriangleBound(s[k], t));
        }
        if (worst + margin <= bound)
        {
          found = true;
          return false;
        }
        return true;
      });
    return found;
  }
};

/// An execution object giving a worklet a `DistanceMeshPortals`.
struct DistanceMeshObject : viskores::cont::ExecutionObjectBase
{
  explicit DistanceMeshObject(const DistanceMesh& mesh)
    : Mesh(mesh)
  {
  }

  VISKORES_CONT auto PrepareForExecution(viskores::cont::DeviceAdapterId device,
                                         viskores::cont::Token& token) const
  {
    auto hierarchy = this->Mesh.Hierarchy.PrepareForExecution(device, token);
    using Portals =
      DistanceMeshPortals<typename viskores::cont::ArrayHandle<Vec3>::ReadPortalType,
                          typename viskores::cont::ArrayHandle<viskores::Id3>::ReadPortalType,
                          decltype(hierarchy)>;
    return Portals{ this->Mesh.Points.PrepareForInput(device, token),
                    this->Mesh.Triangles.PrepareForInput(device, token),
                    hierarchy,
                    this->Mesh.MaxAbsCoordinate };
  }

  DistanceMesh Mesh;
};

namespace detail
{

struct TriangleBox : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn triangle, WholeArrayIn points, FieldOut lower, FieldOut upper);
  using ExecutionSignature = void(_1, _2, _3, _4);

  template <typename PointsPortal>
  VISKORES_EXEC void operator()(const viskores::Id3& triangle,
                                const PointsPortal& points,
                                Vec3& lower,
                                Vec3& upper) const
  {
    const Vec3 a = points.Get(triangle[0]), b = points.Get(triangle[1]), c = points.Get(triangle[2]);
    for (int i = 0; i < 3; ++i)
    {
      lower[i] = viskores::Min(viskores::Min(a[i], b[i]), c[i]);
      upper[i] = viskores::Max(viskores::Max(a[i], b[i]), c[i]);
    }
  }
};

struct MaxAbs : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn point, FieldOut value);
  using ExecutionSignature = _2(_1);

  VISKORES_EXEC Real operator()(const Vec3& p) const
  {
    return viskores::Max(viskores::Max(viskores::Abs(p[0]), viskores::Abs(p[1])),
                         viskores::Abs(p[2]));
  }
};

} // namespace detail

inline Real MaxAbsCoordinate(const viskores::cont::ArrayHandle<Vec3>& points)
{
  if (points.GetNumberOfValues() == 0)
  {
    return 0;
  }
  viskores::cont::Invoker invoke;
  viskores::cont::ArrayHandle<Real> values;
  invoke(detail::MaxAbs{}, points, values);
  return viskores::cont::Algorithm::Reduce(values, Real(0), viskores::Maximum());
}

inline DistanceMesh BuildDistanceMesh(const viskores::cont::ArrayHandle<Vec3>& points,
                                      const viskores::cont::ArrayHandle<viskores::Id3>& triangles)
{
  viskores::cont::Invoker invoke;
  DistanceMesh mesh;
  mesh.Points = points;
  mesh.Triangles = triangles;
  viskores::cont::ArrayHandle<Vec3> lower, upper;
  invoke(detail::TriangleBox{}, triangles, points, lower, upper);
  mesh.Hierarchy.Build(lower, upper);
  mesh.MaxAbsCoordinate = MaxAbsCoordinate(points);
  return mesh;
}

/// The certified envelope test of Section 8 of the paper, with the one-sided bounds of Lemma
/// "One-sided evaluation": a triangle passes only if every point of it lies within `epsilon`
/// of the mesh. A piece passes if `min_i d_i + D <= epsilon` (distance is 1-Lipschitz), or if
/// its three vertices are within `epsilon` of one triangle of the mesh (distance to a convex set
/// is convex); it fails the triangle if some `d_i > epsilon`; otherwise it is split in four, and
/// a piece still unresolved at the depth cap fails the triangle. A piece at depth `j` must clear
/// `epsilon` by `sqrt(3) j u S`, which covers the rounding of the subdivision points; `S`
/// bounds every coordinate of the triangles tested together and of the mesh.
struct CertifiedWithin
{
  Real Epsilon;
  /// `S (1 + 16 u)`.
  Real MaxCoordinate;
  int MaxDepth = 10;

  template <typename Mesh>
  VISKORES_EXEC bool operator()(const Vec3& a,
                                const Vec3& b,
                                const Vec3& c,
                                const Mesh& mesh,
                                viskores::Id& pieces,
                                bool& complete) const
  {
    return (*this)(a, b, c, nullptr, mesh, pieces, complete);
  }

  /// The test with the distances of `a`, `b` and `c` given, as `mesh.DistanceUpTo(p, Epsilon)`
  /// computes them, when `corners` is not null: triangles that share corners share their
  /// distances.
  template <typename Mesh>
  VISKORES_EXEC bool operator()(const Vec3& a,
                                const Vec3& b,
                                const Vec3& c,
                                const Real* corners,
                                const Mesh& mesh,
                                viskores::Id& pieces,
                                bool& complete) const
  {
    // A piece carries the distances of its vertices, which its children share: each distance
    // is computed once, with the same value it would have if computed again. A distance above
    // epsilon fails the triangle whatever its value, so the queries are exact up to epsilon.
    struct Piece
    {
      Vec3 V[3];
      Real D[3];
      int Depth;
    };
    complete = true;
    Piece stack[4 * 16];
    int top = 0;
    stack[top] = Piece{ { a, b, c }, { 0, 0, 0 }, 0 };
    for (int k = 0; k < 3; ++k)
    {
      bool done = true;
      stack[top].D[k] =
        corners ? corners[k] : mesh.DistanceUpTo(stack[top].V[k], this->Epsilon, done);
      complete = complete && done;
      if (stack[top].D[k] > this->Epsilon)
      {
        ++pieces;
        return false;
      }
    }
    ++top;
    const Real sqrt3 = viskores::Sqrt(Real(3));
    while (top > 0)
    {
      const Piece piece = stack[--top];
      ++pieces;
      const Real margin = ((sqrt3 * Real(piece.Depth)) * UnitRoundoff) * this->MaxCoordinate;
      Real diameter = 0;
      for (int k = 0; k < 3; ++k)
      {
        diameter = viskores::Max(diameter, Norm(Sub(piece.V[k], piece.V[(k + 2) % 3])));
      }
      diameter = (1 + 4 * UnitRoundoff) * diameter;
      const Real nearest = viskores::Min(viskores::Min(piece.D[0], piece.D[1]), piece.D[2]);
      if ((nearest + diameter) + margin <= this->Epsilon)
      {
        continue;
      }
      bool done = true;
      if (mesh.WithinOneTriangle(piece.V, this->Epsilon, margin, done))
      {
        complete = complete && done;
        continue;
      }
      complete = complete && done;
      if (piece.Depth == this->MaxDepth)
      {
        return false;
      }
      const Vec3& p0 = piece.V[0];
      const Vec3& p1 = piece.V[1];
      const Vec3& p2 = piece.V[2];
      const Vec3 m01 = Scale(Real(0.5), Add(p0, p1));
      const Vec3 m12 = Scale(Real(0.5), Add(p1, p2));
      const Vec3 m20 = Scale(Real(0.5), Add(p2, p0));
      Real dm[3];
      const Vec3* mid[3] = { &m01, &m12, &m20 };
      for (int k = 0; k < 3; ++k)
      {
        dm[k] = mesh.DistanceUpTo(*mid[k], this->Epsilon, done);
        complete = complete && done;
        if (dm[k] > this->Epsilon)
        {
          return false;
        }
      }
      const int depth = piece.Depth + 1;
      stack[top++] = Piece{ { m01, m12, m20 }, { dm[0], dm[1], dm[2] }, depth };
      stack[top++] = Piece{ { m20, m12, p2 }, { dm[2], dm[1], piece.D[2] }, depth };
      stack[top++] = Piece{ { m01, p1, m12 }, { dm[0], piece.D[1], dm[1] }, depth };
      stack[top++] = Piece{ { p0, m01, m20 }, { piece.D[0], dm[0], dm[2] }, depth };
    }
    return true;
  }
};

/// The report of `EnvelopeFlags`.
struct EnvelopeReport
{
  bool Valid = false;
  /// 3-label triangles whose output triangle failed the test against the input, part (i).
  viskores::Id FailedOutput = 0;
  /// Input triangles that failed the test against the output, part (ii).
  viskores::Id FailedInput = 0;
  /// Triangles tested, and pieces examined, in both parts.
  viskores::Id Tested = 0;
  viskores::Id Pieces = 0;
  /// `S (1 + 16 u)` of the two parts, over every triangle that needs the test.
  Real ScaleInput = 0;
  Real ScaleOutput = 0;
};

namespace detail
{

/// `D(t)`, the largest displacement of a vertex of `t` to its generator, compared with
/// `epsilon`: the pre-test of Lemma "Displacement pre-test".
struct NeedsTest : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn triangle,
                                WholeArrayIn points,
                                WholeArrayIn labels,
                                WholeArrayIn generators,
                                FieldOut needsTest);
  using ExecutionSignature = _5(_1, _2, _3, _4);

  VISKORES_CONT explicit NeedsTest(Real epsilon)
    : Epsilon(epsilon)
  {
  }

  template <typename PointsPortal, typename LabelsPortal>
  VISKORES_EXEC viskores::UInt8 operator()(const viskores::Id3& triangle,
                                           const PointsPortal& points,
                                           const LabelsPortal& labels,
                                           const PointsPortal& generators) const
  {
    Real displacement = 0;
    for (int k = 0; k < 3; ++k)
    {
      const Vec3 p = points.Get(triangle[k]);
      displacement =
        viskores::Max(displacement, Norm(Sub(generators.Get(labels.Get(triangle[k])), p)));
    }
    return (displacement > this->Epsilon) ? 1 : 0;
  }

  Real Epsilon;
};

/// The largest absolute coordinate of the triangles to be tested, for the margin `S`.
struct TestedMaxAbs : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn triangle,
                                FieldIn needsTest,
                                WholeArrayIn points,
                                WholeArrayIn labels,
                                WholeArrayIn generators,
                                FieldOut value);
  using ExecutionSignature = _6(_1, _2, _3, _4, _5);

  VISKORES_CONT explicit TestedMaxAbs(bool atGenerators)
    : AtGenerators(atGenerators)
  {
  }

  template <typename PointsPortal, typename LabelsPortal>
  VISKORES_EXEC Real operator()(const viskores::Id3& triangle,
                                viskores::UInt8 needsTest,
                                const PointsPortal& points,
                                const LabelsPortal& labels,
                                const PointsPortal& generators) const
  {
    Real s = 0;
    if (needsTest)
    {
      for (int k = 0; k < 3; ++k)
      {
        const Vec3 p = this->AtGenerators ? generators.Get(labels.Get(triangle[k]))
                                          : points.Get(triangle[k]);
        s = viskores::Max(s, viskores::Max(viskores::Max(viskores::Abs(p[0]), viskores::Abs(p[1])),
                                           viskores::Abs(p[2])));
      }
    }
    return s;
  }

  bool AtGenerators;
};

/// Runs the certified test on a triangle, realised either at the generators of its labels or
/// at its own vertices.
struct BothFlags : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn a, FieldIn b, FieldOut both);
  using ExecutionSignature = _3(_1, _2);

  VISKORES_EXEC viskores::UInt8 operator()(viskores::UInt8 a, viskores::UInt8 b) const
  {
    return (a && b) ? 1 : 0;
  }
};

/// Whether a vertex is a corner of a triangle to test.
struct IsTestedCorner : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn vertex,
                                WholeArrayIn cornerOffsets,
                                WholeArrayIn corners,
                                WholeArrayIn needsTest,
                                FieldOut tested);
  using ExecutionSignature = void(_1, _2, _3, _4, _5);

  VISKORES_CONT explicit IsTestedCorner(viskores::Id numberOfTriangles)
    : NumberOfTriangles(numberOfTriangles)
  {
  }

  template <typename IdPortal, typename FlagPortal>
  VISKORES_EXEC void operator()(viskores::Id v,
                                const IdPortal& cornerOffsets,
                                const IdPortal& corners,
                                const FlagPortal& needsTest,
                                viskores::UInt8& tested) const
  {
    tested = 0;
    for (viskores::Id r = cornerOffsets.Get(v); r < cornerOffsets.Get(v + 1) && !tested; ++r)
    {
      tested = needsTest.Get(corners.Get(r) % this->NumberOfTriangles);
    }
  }

private:
  viskores::Id NumberOfTriangles;
};

/// `mesh.DistanceUpTo(p, epsilon)` for the points that are wanted, the first step of the
/// certified test of every triangle with a corner there.
struct CornerDistance : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn point, FieldIn wanted, ExecObject mesh, FieldOut distance);
  using ExecutionSignature = void(_1, _2, _3, _4);

  VISKORES_CONT explicit CornerDistance(Real epsilon)
    : Epsilon(epsilon)
  {
  }

  template <typename Mesh>
  VISKORES_EXEC void operator()(const Vec3& p,
                                viskores::UInt8 wanted,
                                const Mesh& mesh,
                                Real& distance) const
  {
    distance = 0;
    if (wanted)
    {
      bool complete = true;
      distance = mesh.DistanceUpTo(p, this->Epsilon, complete);
      if (!complete)
      {
        this->RaiseError("QuadricClusteringDecimation: bounding volume hierarchy too deep.");
      }
    }
  }

private:
  Real Epsilon;
};

struct TestTriangle : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn triangle,
                                FieldIn needsTest,
                                WholeArrayIn points,
                                WholeArrayIn labels,
                                WholeArrayIn generators,
                                WholeArrayIn cornerDistances,
                                ExecObject distanceMesh,
                                FieldOut failed,
                                FieldOut pieces);
  using ExecutionSignature = void(_1, _2, _3, _4, _5, _6, _7, _8, _9);

  VISKORES_CONT TestTriangle(const CertifiedWithin& test, bool atGenerators)
    : Test(test)
    , AtGenerators(atGenerators)
  {
  }

  template <typename PointsPortal, typename LabelsPortal, typename DistancesPortal, typename Mesh>
  VISKORES_EXEC void operator()(const viskores::Id3& triangle,
                                viskores::UInt8 needsTest,
                                const PointsPortal& points,
                                const LabelsPortal& labels,
                                const PointsPortal& generators,
                                const DistancesPortal& cornerDistances,
                                const Mesh& mesh,
                                viskores::UInt8& failed,
                                viskores::Id& pieces) const
  {
    failed = 0;
    pieces = 0;
    if (!needsTest)
    {
      return;
    }
    Vec3 v[3];
    Real d[3];
    for (int k = 0; k < 3; ++k)
    {
      const viskores::Id corner = this->AtGenerators ? labels.Get(triangle[k]) : triangle[k];
      v[k] = this->AtGenerators ? generators.Get(corner) : points.Get(corner);
      d[k] = cornerDistances.Get(corner);
    }
    bool complete = true;
    const bool within = this->Test(v[0], v[1], v[2], d, mesh, pieces, complete);
    if (!complete)
    {
      this->RaiseError("QuadricClusteringDecimation: bounding volume hierarchy too deep.");
    }
    failed = within ? 0 : 1;
  }

  CertifiedWithin Test;
  bool AtGenerators;
};

struct FailedLabels : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn triangle,
                                FieldIn failedA,
                                FieldIn failedB,
                                WholeArrayIn labels,
                                FieldOut listed);
  using ExecutionSignature = void(_1, _2, _3, _4, _5);

  template <typename LabelsPortal, typename ListedVec>
  VISKORES_EXEC void operator()(const viskores::Id3& triangle,
                                viskores::UInt8 failedA,
                                viskores::UInt8 failedB,
                                const LabelsPortal& labels,
                                ListedVec& listed) const
  {
    for (int k = 0; k < 3; ++k)
    {
      listed[k] = (failedA || failedB) ? labels.Get(triangle[k]) : viskores::Id(-1);
    }
  }
};

struct IsThreeLabelTriangle : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn triangle, WholeArrayIn labels, FieldIn needsTest, FieldOut out);
  using ExecutionSignature = _4(_1, _2, _3);

  template <typename LabelsPortal>
  VISKORES_EXEC viskores::UInt8 operator()(const viskores::Id3& triangle,
                                           const LabelsPortal& labels,
                                           viskores::UInt8 needsTest) const
  {
    const viskores::Id a = labels.Get(triangle[0]);
    const viskores::Id b = labels.Get(triangle[1]);
    const viskores::Id c = labels.Get(triangle[2]);
    return (needsTest && a != b && b != c && a != c) ? 1 : 0;
  }
};

} // namespace detail

/// Which triangles the envelope flag tests, and against what.
enum class EnvelopeMode
{
  /// (F5): the output triangles of the 3-label triangles against the input, and the input
  /// triangles against the realised output.
  Surface,
  /// (F5'), the geometric-only mode: the simplex `g(t)` of every input triangle against the
  /// input, and every input triangle against the union of the simplices.
  Simplices
};

/// The halves of the envelope flag (F5) in use: (i) every output triangle within epsilon of the
/// input, (ii) every input triangle within epsilon of the output.
enum class EnvelopeSides : viskores::UInt8
{
  OutputToInput = 1,
  InputToOutput = 2,
  Both = 3
};

/// The envelope flag of Definition "Envelope flag" and Algorithm 7, for the triangles with
/// `D(t) > epsilon`, with the halves `sides`. `inputDistance` is the distance structure of the
/// input, built once, and `outputTriangles` the realised output, as triples of labels. Flags,
/// among the clusters with at least two vertices, every cluster of a vertex of a triangle that
/// fails a test.
///
/// With `subset`, only the input triangles it marks are tested, with the scales of the whole
/// test: a caller that knows every other triangle passes, and that the scales are those under
/// which it passed, gets the verdict of the whole test.
inline EnvelopeReport EnvelopeFlags(
  const SurfaceMesh& mesh,
  const DistanceMesh& inputDistance,
  const viskores::cont::ArrayHandle<viskores::Id>& labels,
  const viskores::cont::ArrayHandle<Vec3>& generators,
  const viskores::cont::ArrayHandle<viskores::Id3>& outputTriangles,
  const viskores::cont::ArrayHandle<viskores::Id>& sizes,
  Real epsilon,
  EnvelopeMode mode,
  EnvelopeSides sides,
  viskores::cont::ArrayHandle<viskores::UInt8>& flags,
  const viskores::cont::ArrayHandle<viskores::UInt8>* subset = nullptr)
{
  using Algorithm = viskores::cont::Algorithm;
  viskores::cont::Invoker invoke;
  EnvelopeReport report;
  report.Valid = true;
  viskores::cont::ArrayHandle<viskores::UInt8> needsTest;
  invoke(detail::NeedsTest(epsilon), mesh.Triangles, mesh.Points, labels, generators, needsTest);
  const viskores::Id numNeeding = Algorithm::Reduce(needsTest, viskores::Id(0));
  if (numNeeding == 0)
  {
    return report;
  }
  viskores::cont::ArrayHandle<viskores::UInt8> testOutput;
  if (mode == EnvelopeMode::Surface)
  {
    invoke(detail::IsThreeLabelTriangle{}, mesh.Triangles, labels, needsTest, testOutput);
  }
  else
  {
    testOutput = needsTest;
  }
  const Real oneSixteen = 1 + 16 * UnitRoundoff;
  auto restrict = [&](const viskores::cont::ArrayHandle<viskores::UInt8>& all)
  {
    if (!subset)
    {
      return all;
    }
    viskores::cont::ArrayHandle<viskores::UInt8> some;
    invoke(detail::BothFlags{}, all, *subset, some);
    return some;
  };

  // A half not in use tests no triangle.
  const viskores::Id m = mesh.NumberOfTriangles;
  viskores::cont::ArrayHandle<viskores::UInt8> failedA, failedB;
  viskores::cont::ArrayHandle<viskores::Id> piecesA, piecesB;
  if (!(static_cast<viskores::UInt8>(sides) &
        static_cast<viskores::UInt8>(EnvelopeSides::OutputToInput)))
  {
    testOutput.AllocateAndFill(m, 0);
    failedA.AllocateAndFill(m, 0);
    piecesA.AllocateAndFill(m, 0);
  }
  else
  {
    // (i): the realised triangles against the input.
    viskores::cont::ArrayHandle<Real> values;
    invoke(detail::TestedMaxAbs(true), mesh.Triangles, testOutput, mesh.Points, labels, generators, values);
    const Real s = viskores::Max(Algorithm::Reduce(values, Real(0), viskores::Maximum()),
                                 inputDistance.MaxAbsCoordinate);
    CertifiedWithin test{ epsilon, s * oneSixteen };
    report.ScaleInput = test.MaxCoordinate;
    testOutput = restrict(testOutput);
    // The corners are generators, shared by many triangles.
    viskores::cont::ArrayHandle<Real> cornerDistances;
    invoke(detail::CornerDistance(epsilon),
           generators,
           viskores::cont::make_ArrayHandleConstant(viskores::UInt8(1), generators.GetNumberOfValues()),
           DistanceMeshObject(inputDistance),
           cornerDistances);
    invoke(detail::TestTriangle(test, true),
           mesh.Triangles,
           testOutput,
           mesh.Points,
           labels,
           generators,
           cornerDistances,
           DistanceMeshObject(inputDistance),
           failedA,
           piecesA);
  }
  if (!(static_cast<viskores::UInt8>(sides) &
        static_cast<viskores::UInt8>(EnvelopeSides::InputToOutput)))
  {
    needsTest.AllocateAndFill(m, 0);
    failedB.AllocateAndFill(m, 0);
    piecesB.AllocateAndFill(m, 0);
  }
  else
  {
    // (ii): the input triangles against the realised output.
    const DistanceMesh outputDistance = BuildDistanceMesh(generators, outputTriangles);
    viskores::cont::ArrayHandle<Real> values;
    invoke(detail::TestedMaxAbs(false), mesh.Triangles, needsTest, mesh.Points, labels, generators, values);
    const Real s = viskores::Max(Algorithm::Reduce(values, Real(0), viskores::Maximum()),
                                 outputDistance.MaxAbsCoordinate);
    CertifiedWithin test{ epsilon, s * oneSixteen };
    report.ScaleOutput = test.MaxCoordinate;
    needsTest = restrict(needsTest);
    // The corners are input vertices, each shared by several triangles.
    viskores::cont::ArrayHandle<viskores::UInt8> tested;
    invoke(detail::IsTestedCorner(mesh.NumberOfTriangles),
           viskores::cont::ArrayHandleIndex(mesh.NumberOfPoints),
           mesh.VertexCornerOffsets,
           mesh.VertexCorners,
           needsTest,
           tested);
    viskores::cont::ArrayHandle<Real> cornerDistances;
    invoke(detail::CornerDistance(epsilon),
           mesh.Points,
           tested,
           DistanceMeshObject(outputDistance),
           cornerDistances);
    invoke(detail::TestTriangle(test, false),
           mesh.Triangles,
           needsTest,
           mesh.Points,
           labels,
           generators,
           cornerDistances,
           DistanceMeshObject(outputDistance),
           failedB,
           piecesB);
  }
  report.FailedOutput = Algorithm::Reduce(failedA, viskores::Id(0));
  report.FailedInput = Algorithm::Reduce(failedB, viskores::Id(0));
  report.Tested = Algorithm::Reduce(testOutput, viskores::Id(0)) + Algorithm::Reduce(needsTest, viskores::Id(0));
  report.Pieces = Algorithm::Reduce(piecesA, viskores::Id(0)) + Algorithm::Reduce(piecesB, viskores::Id(0));
  if (report.FailedOutput + report.FailedInput == 0)
  {
    return report;
  }
  viskores::cont::ArrayHandle<viskores::Id> listed, valid;
  invoke(detail::FailedLabels{},
         mesh.Triangles,
         failedA,
         failedB,
         labels,
         viskores::cont::make_ArrayHandleGroupVec<3>(listed));
  viskores::cont::ArrayHandle<viskores::UInt8> isValid;
  invoke(detail::IsValidLabel{}, listed, isValid);
  Algorithm::CopyIf(listed, isValid, valid);
  FlagClusters(valid, sizes, flags);
  return report;
}

}
}
} // namespace viskores::worklet::quadric_clustering

VISKORES_QUADRIC_CLUSTERING_STRICT_FP_END

#endif // viskores_worklet_quadric_clustering_EnvelopeFlags_h
