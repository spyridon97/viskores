//============================================================================
//  The contents of this file are covered by the Viskores license. See
//  LICENSE.txt for details.
//
//  By contributing to this file, all contributors agree to the Developer
//  Certificate of Origin Version 1.1 (DCO 1.1) as stated in DCO.txt.
//============================================================================
#ifndef viskores_worklet_quadric_clustering_ThickeningFlags_h
#define viskores_worklet_quadric_clustering_ThickeningFlags_h

#include <viskores/filter/geometry_refinement/worklet/quadric_clustering/EmbeddingFlags.h>
#include <viskores/filter/geometry_refinement/worklet/quadric_clustering/NonManifoldCut.h>
#include <viskores/filter/geometry_refinement/worklet/quadric_clustering/QuadricClusteringCommonWorklets.h>
#include <viskores/filter/geometry_refinement/worklet/quadric_clustering/SurfaceMesh.h>
#include <viskores/filter/geometry_refinement/worklet/quadric_clustering/TopologyFlags.h>

#include <viskores/BinaryOperators.h>
#include <viskores/ExactPredicates.h>
#include <viskores/cont/Algorithm.h>
#include <viskores/cont/ArrayHandleConcatenate.h>
#include <viskores/cont/ArrayHandleConstant.h>
#include <viskores/cont/ArrayHandleGroupVec.h>
#include <viskores/cont/ArrayHandleIndex.h>
#include <viskores/cont/ArrayHandleView.h>
#include <viskores/cont/ErrorBadValue.h>
#include <viskores/cont/ExecutionObjectBase.h>
#include <viskores/worklet/WorkletMapField.h>
#include <viskores/worklet/spatialstructure/BoundingVolumeHierarchy.h>

#include <cstring>
#include <limits>

VISKORES_QUADRIC_CLUSTERING_STRICT_FP_BEGIN

namespace viskores
{
namespace worklet
{
namespace quadric_clustering
{

// Isotopy by a certified thickening T of the input M (Theorem "Isotopy without a reach"), the image of the prism map linear on the tetrahedra of the vertex-ordered split of
// the prisms over the triangles, with the stored sides B^+-_v = fl(v +- delta_v nu_v), and the
// flag (F7) of an output, as proofs/lib/thick.py. Every decision is exact: on the stored doubles
// by the exact orientation, and on the ends of the probes, a third of a sum of three doubles,
// scaled by three in expansion arithmetic.

/// A certified thickening, from `BuildThickening`.
struct Thickening
{
  /// Whether (T1)-(T4) were certified; the other fields are meaningful only then.
  bool Valid = false;
  /// The rounds of halving of the offsets.
  viskores::Id Halvings = 0;
  /// The stored sides `B = [B^+; B^-]`, their triangles `[F; F reversed + n]`, and a hierarchy
  /// over them.
  viskores::cont::ArrayHandle<Vec3> Sides;
  viskores::cont::ArrayHandle<viskores::Id3> SideTriangles;
  viskores::worklet::spatialstructure::BoundingVolumeHierarchy SideHierarchy;
  /// Per component of M, the triangle of its probe and the stored sides at its corners,
  /// `B^-` then `B^+`.
  viskores::cont::ArrayHandle<viskores::Id> ProbeTriangles;
  viskores::cont::ArrayHandle<viskores::Vec<Vec3, 6>> ProbePoints;
  /// The largest side of the bounding box of M, and the largest absolute coordinate involved.
  Real Span = 0;
  Real MaxAbsCoordinate = 0;
};

/// The directions `nu` along which a thickening offsets the vertices (Proposition "When the
/// thickening certifies").
enum class ThickeningDirections
{
  /// The area-weighted normal where its margins at the star are positive, the max-min
  /// direction elsewhere.
  Hybrid,
  /// The area-weighted normal.
  Normal,
  /// The direction that maximises the smallest dot product with the normals of the star.
  MaxMin
};

namespace thickening_detail
{

/// The unnormalised and the unit normal of a triangle, `cross(V1 - V0, V2 - V0)`.
struct FaceNormal : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn triangle,
                                WholeArrayIn points,
                                FieldOut cross,
                                FieldOut unit);
  using ExecutionSignature = void(_1, _2, _3, _4);

  template <typename VecPortal>
  VISKORES_EXEC void operator()(const viskores::Id3& t,
                                const VecPortal& points,
                                Vec3& cross,
                                Vec3& unit) const
  {
    const Vec3 a = points.Get(t[0]);
    cross = Cross(Sub(points.Get(t[1]), a), Sub(points.Get(t[2]), a));
    const Real norm = Norm(cross);
    unit = Vec3(cross[0] / norm, cross[1] / norm, cross[2] / norm);
  }
};

/// The vertex normal: the cross products of its triangles summed in increasing corner order (by
/// corner position, then triangle), as the vertex quadrics are, normalised.
struct VertexNormal : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn vertex,
                                WholeArrayIn cornerOffsets,
                                WholeArrayIn corners,
                                WholeArrayIn crosses,
                                FieldOut normal);
  using ExecutionSignature = _5(_1, _2, _3, _4);

  VISKORES_CONT explicit VertexNormal(viskores::Id numTriangles)
    : NumberOfTriangles(numTriangles)
  {
  }

  template <typename IdPortal, typename VecPortal>
  VISKORES_EXEC Vec3 operator()(viskores::Id v,
                                const IdPortal& cornerOffsets,
                                const IdPortal& corners,
                                const VecPortal& crosses) const
  {
    Vec3 acc(0);
    for (viskores::Id r = cornerOffsets.Get(v); r < cornerOffsets.Get(v + 1); ++r)
    {
      acc = Add(acc, crosses.Get(corners.Get(r) % this->NumberOfTriangles));
    }
    const Real norm = Norm(acc);
    return Vec3(acc[0] / norm, acc[1] / norm, acc[2] / norm);
  }

  viskores::Id NumberOfTriangles;
};

/// The star records `(vertex, triangle)` of the corners of a triangle; sorted, they list the
/// triangles of every vertex in increasing order.
struct StarRecords : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn triangle, FieldOut records);
  using ExecutionSignature = void(_1, _2, WorkIndex);

  template <typename RecordVec>
  VISKORES_EXEC void operator()(const viskores::Id3& tri, RecordVec& records, viskores::Id t) const
  {
    for (viskores::IdComponent s = 0; s < 3; ++s)
    {
      records[s] = viskores::Id2(tri[s], t);
    }
  }
};

/// Whether the triangle of a star record has a finite unit normal: a degenerate one has none.
struct FiniteStarRecord : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn record, WholeArrayIn units, FieldOut finite);
  using ExecutionSignature = _3(_1, _2);

  template <typename VecPortal>
  VISKORES_EXEC viskores::UInt8 operator()(const viskores::Id2& record,
                                           const VecPortal& units) const
  {
    const Vec3 u = units.Get(record[1]);
    return (viskores::IsFinite(u[0]) && viskores::IsFinite(u[1]) && viskores::IsFinite(u[2])) ? 1
                                                                                              : 0;
  }
};

/// The max-min direction of a vertex, as `maxmin_directions` of proofs/lib/thick.py
/// (Proposition "When the thickening certifies" (ii)): the unit direction with the largest smallest dot product
/// with the unit normals of its star, the normalised point nearest to 0 of the convex hull of
/// those normals. It is the nearest point of the affine hull of one, two or three of them with
/// nonnegative barycentric coordinates; the candidates are tried in lexicographic order of the
/// star positions (the star in increasing triangle order, degenerate triangles left out), and
/// the first of largest margin is kept. Where no margin is positive, or the star is all
/// degenerate, the direction is the area-weighted normal. Written at the vertices listed.
struct MaxMinDirection : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn vertex,
                                WholeArrayIn areaNormals,
                                WholeArrayIn starOffsets,
                                WholeArrayIn starTriangles,
                                WholeArrayIn units,
                                WholeArrayInOut directions);
  using ExecutionSignature = void(_1, _2, _3, _4, _5, _6);

  template <typename IdPortal, typename VecPortal, typename OutPortal>
  VISKORES_EXEC void operator()(viskores::Id v,
                                const VecPortal& areaNormals,
                                const IdPortal& starOffsets,
                                const IdPortal& starTriangles,
                                const VecPortal& units,
                                const OutPortal& directions) const
  {
    const viskores::Id begin = starOffsets.Get(v);
    const viskores::Id k = starOffsets.Get(v + 1) - begin;
    Vec3 direction = areaNormals.Get(v);
    if (k == 0)
    {
      directions.Set(v, direction);
      return;
    }
    auto normal = [&](viskores::Id i) { return units.Get(starTriangles.Get(begin + i)); };
    bool found = false;
    Real best = 0;
    // A candidate is kept only if its margin exceeds the best one, and its running minimum only
    // decreases, so its scan stops once that minimum is not above the best.
    auto smallest = [&](const Vec3& q)
    {
      Real low = Dot(normal(0), q);
      for (viskores::Id i = 1; i < k && !(found && !(low > best)); ++i)
      {
        const Real d = Dot(normal(i), q);
        low = (d < low) ? d : low;
      }
      return low;
    };
    Vec3 bestDirection(0);
    auto consider = [&](const Vec3& p)
    {
      const Real pp = Dot(p, p);
      if (!(pp > 0))
      {
        return;
      }
      const Real length = viskores::Sqrt(pp);
      const Vec3 q(p[0] / length, p[1] / length, p[2] / length);
      const Real low = smallest(q);
      if (!found || low > best)
      {
        found = true;
        best = low;
        bestDirection = q;
      }
    };
    for (viskores::Id i = 0; i < k; ++i)
    {
      consider(normal(i));
    }
    for (viskores::Id i = 0; i < k; ++i)
    {
      for (viskores::Id j = i + 1; j < k; ++j)
      {
        const Vec3 p0 = normal(i);
        const Vec3 e0 = Sub(normal(j), p0);
        const Real g00 = Dot(e0, e0);
        const Real r0 = -Dot(e0, p0);
        if (!(g00 > 0))
        {
          continue;
        }
        const Real x0 = r0 / g00;
        if (1 - x0 < 0 || x0 < 0)
        {
          continue;
        }
        // The sum of the reference starts from 0.
        consider(Add(p0, Add(Vec3(0), Scale(x0, e0))));
      }
    }
    for (viskores::Id i = 0; i < k; ++i)
    {
      for (viskores::Id j = i + 1; j < k; ++j)
      {
        for (viskores::Id l = j + 1; l < k; ++l)
        {
          const Vec3 p0 = normal(i);
          const Vec3 e0 = Sub(normal(j), p0);
          const Vec3 e1 = Sub(normal(l), p0);
          const Real g00 = Dot(e0, e0), g01 = Dot(e0, e1), g10 = Dot(e1, e0), g11 = Dot(e1, e1);
          const Real r0 = -Dot(e0, p0), r1 = -Dot(e1, p0);
          const Real dg = g00 * g11 - g01 * g10;
          if (!(dg > 0))
          {
            continue;
          }
          const Real x0 = (r0 * g11 - g01 * r1) / dg;
          const Real x1 = (g00 * r1 - g10 * r0) / dg;
          if (1 - (x0 + x1) < 0 || x0 < 0 || x1 < 0)
          {
            continue;
          }
          consider(Add(p0, Add(Add(Vec3(0), Scale(x0, e0)), Scale(x1, e1))));
        }
      }
    }
    if (found && best > 0)
    {
      direction = bestDirection;
    }
    directions.Set(v, direction);
  }
};

/// The hybrid direction of a vertex, as `hybrid_directions` of proofs/lib/thick.py: the
/// area-weighted normal where its smallest dot product with the unit normals of the star
/// (degenerate triangles ignored) is positive, the max-min direction elsewhere. Writes the
/// area-weighted normal where it holds, and returns whether the max-min direction is needed.
struct HybridDirection : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn vertex,
                                WholeArrayIn areaNormals,
                                WholeArrayIn cornerOffsets,
                                WholeArrayIn corners,
                                WholeArrayIn units,
                                WholeArrayInOut directions,
                                FieldOut needsMaxMin);
  using ExecutionSignature = _7(_1, _2, _3, _4, _5, _6);

  VISKORES_CONT explicit HybridDirection(viskores::Id numTriangles)
    : NumberOfTriangles(numTriangles)
  {
  }

  template <typename IdPortal, typename VecPortal, typename OutPortal>
  VISKORES_EXEC viskores::UInt8 operator()(viskores::Id v,
                                           const VecPortal& areaNormals,
                                           const IdPortal& cornerOffsets,
                                           const IdPortal& corners,
                                           const VecPortal& units,
                                           const OutPortal& directions) const
  {
    const Vec3 areaNormal = areaNormals.Get(v);
    Real low = std::numeric_limits<Real>::infinity();
    for (viskores::Id r = cornerOffsets.Get(v); r < cornerOffsets.Get(v + 1); ++r)
    {
      const Real d = Dot(units.Get(corners.Get(r) % this->NumberOfTriangles), areaNormal);
      low = (d < low) ? d : low;
    }
    if (low > 0)
    {
      directions.Set(v, areaNormal);
      return 0;
    }
    return 1;
  }

  viskores::Id NumberOfTriangles;
};

} // namespace thickening_detail

/// The directions `nu` of the vertices of `mesh`, as `build` of proofs/lib/thick.py takes them:
/// the area-weighted normals (the cross products summed by corner slot, then triangle), or
/// `maxmin_directions`, or `hybrid_directions`.
inline void VertexDirections(const SurfaceMesh& mesh,
                             ThickeningDirections which,
                             viskores::cont::ArrayHandle<Vec3>& directions)
{
  using Algorithm = viskores::cont::Algorithm;
  namespace td = thickening_detail;
  viskores::cont::Invoker invoke;
  const viskores::Id n = mesh.NumberOfPoints, m = mesh.NumberOfTriangles;
  viskores::cont::ArrayHandle<Vec3> crosses, units, areaNormals;
  invoke(td::FaceNormal{}, mesh.Triangles, mesh.Points, crosses, units);
  invoke(td::VertexNormal(m),
         viskores::cont::ArrayHandleIndex(n),
         mesh.VertexCornerOffsets,
         mesh.VertexCorners,
         crosses,
         areaNormals);
  if (which == ThickeningDirections::Normal)
  {
    directions = areaNormals;
    return;
  }
  // The stars in increasing triangle order, without the degenerate triangles.
  viskores::cont::ArrayHandle<viskores::Id2> records, finiteRecords;
  invoke(td::StarRecords{}, mesh.Triangles, viskores::cont::make_ArrayHandleGroupVec<3>(records));
  Algorithm::Sort(records);
  viskores::cont::ArrayHandle<viskores::UInt8> finite;
  invoke(td::FiniteStarRecord{}, records, units, finite);
  Algorithm::CopyIf(records, finite, finiteRecords);
  viskores::cont::ArrayHandle<viskores::Id> starVertices, starTriangles, starOffsets;
  invoke(SplitPair{}, finiteRecords, starVertices, starTriangles);
  Algorithm::LowerBounds(starVertices, viskores::cont::ArrayHandleIndex(n + 1), starOffsets);
  viskores::cont::ArrayHandle<viskores::Id> vertices;
  Algorithm::Copy(viskores::cont::ArrayHandleIndex(n), vertices);
  directions.Allocate(n);
  // The max-min directions, where they are used.
  if (which == ThickeningDirections::Hybrid)
  {
    viskores::cont::ArrayHandle<viskores::UInt8> needsMaxMin;
    invoke(td::HybridDirection(m),
           vertices,
           areaNormals,
           mesh.VertexCornerOffsets,
           mesh.VertexCorners,
           units,
           directions,
           needsMaxMin);
    viskores::cont::ArrayHandle<viskores::Id> maxMinVertices;
    Algorithm::CopyIf(vertices, needsMaxMin, maxMinVertices);
    vertices = maxMinVertices;
  }
  invoke(
    td::MaxMinDirection{}, vertices, areaNormals, starOffsets, starTriangles, units, directions);
}

namespace thickening_detail
{

/// The triangles of the sides: `F` on `B^+`, and `F` reversed on `B^-`.
struct SideTriangle : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn index, WholeArrayIn triangles, FieldOut side);
  using ExecutionSignature = _3(_1, _2);

  VISKORES_CONT explicit SideTriangle(viskores::Id numPoints)
    : NumberOfPoints(numPoints)
  {
  }

  template <typename Id3Portal>
  VISKORES_EXEC viskores::Id3 operator()(viskores::Id i, const Id3Portal& triangles) const
  {
    const viskores::Id m = triangles.GetNumberOfValues();
    if (i < m)
    {
      return triangles.Get(i);
    }
    const viskores::Id3 t = triangles.Get(i - m);
    const viskores::Id n = this->NumberOfPoints;
    return viskores::Id3(t[2] + n, t[1] + n, t[0] + n);
  }

  viskores::Id NumberOfPoints;
};

/// The stored sides of a vertex: `B^+ = v + (delta^+ nu)` and `B^- = v - (delta^- nu)`, the
/// products first.
struct StoredSides : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn point,
                                FieldIn normal,
                                FieldIn deltaPlus,
                                FieldIn deltaMinus,
                                FieldOut plus,
                                FieldOut minus);
  using ExecutionSignature = void(_1, _2, _3, _4, _5, _6);

  VISKORES_EXEC void operator()(const Vec3& v,
                                const Vec3& nu,
                                Real deltaPlus,
                                Real deltaMinus,
                                Vec3& plus,
                                Vec3& minus) const
  {
    plus = Add(v, Vec3(deltaPlus * nu[0], deltaPlus * nu[1], deltaPlus * nu[2]));
    minus = Sub(v, Vec3(deltaMinus * nu[0], deltaMinus * nu[1], deltaMinus * nu[2]));
  }
};

/// The three tetrahedra of the vertex-ordered split of the prism over a triangle, as (vertex,
/// level) pairs, level 0 the lower point of a layer.
VISKORES_EXEC inline void PrismTetrahedra(const viskores::Id3& t,
                                          viskores::Id vertex[3][4],
                                          int level[3][4])
{
  const viskores::Id3 s = SortThree(t);
  const viskores::Id i = s[0], j = s[1], k = s[2];
  const viskores::Id v[3][4] = { { i, j, k, k }, { i, j, j, k }, { i, i, j, k } };
  const int l[3][4] = { { 0, 0, 0, 1 }, { 0, 0, 1, 1 }, { 0, 1, 1, 1 } };
  for (int a = 0; a < 3; ++a)
  {
    for (int b = 0; b < 4; ++b)
    {
      vertex[a][b] = v[a][b];
      level[a][b] = l[a][b];
    }
  }
}

/// (T1) on a triangle: every tetrahedron of either layer is nondegenerate and has the
/// orientation of the flat prism over the triangle, `V` at level 0 and `fl(V + nf)` at level 1;
/// per layer, `[0, 1]` (`badPlus`) and `[-1, 0]` (`badMinus`).
struct BadPrism : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn triangle,
                                FieldIn unit,
                                WholeArrayIn points,
                                WholeArrayIn plus,
                                WholeArrayIn minus,
                                FieldOut badPlus,
                                FieldOut badMinus);
  using ExecutionSignature = void(_1, _2, _3, _4, _5, _6, _7);

  template <typename VecPortal>
  VISKORES_EXEC void operator()(const viskores::Id3& t,
                                const Vec3& nf,
                                const VecPortal& points,
                                const VecPortal& plus,
                                const VecPortal& minus,
                                viskores::UInt8& badPlus,
                                viskores::UInt8& badMinus) const
  {
    badPlus = 0;
    badMinus = 0;
    viskores::Id vertex[3][4];
    int level[3][4];
    PrismTetrahedra(t, vertex, level);
    for (int layer = 0; layer < 2; ++layer)
    {
      for (int a = 0; a < 3; ++a)
      {
        Vec3 p[4], q[4];
        for (int b = 0; b < 4; ++b)
        {
          const viskores::Id v = vertex[a][b];
          const Vec3 x = points.Get(v);
          if (layer == 0)
          {
            p[b] = (level[a][b] == 0) ? minus.Get(v) : x;
          }
          else
          {
            p[b] = (level[a][b] == 0) ? x : plus.Get(v);
          }
          q[b] = (level[a][b] == 0) ? x : Add(x, nf);
        }
        const viskores::IdComponent s = viskores::Orient3D(p[0], p[1], p[2], p[3]);
        const viskores::IdComponent s0 = viskores::Orient3D(q[0], q[1], q[2], q[3]);
        if (s == 0 || (s > 0) != (s0 > 0))
        {
          (layer == 0 ? badMinus : badPlus) = 1;
        }
      }
    }
  }
};

/// Marks the vertices, modulo `n`, of a flagged triangle; every write is a 1.
struct MarkTriangleVertices : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn triangle, FieldIn flagged, WholeArrayOut marks);
  using ExecutionSignature = void(_1, _2, _3);

  VISKORES_CONT explicit MarkTriangleVertices(viskores::Id numPoints)
    : NumberOfPoints(numPoints)
  {
  }

  template <typename FlagPortal>
  VISKORES_EXEC void operator()(const viskores::Id3& t,
                                viskores::UInt8 flagged,
                                FlagPortal& marks) const
  {
    if (flagged)
    {
      for (int k = 0; k < 3; ++k)
      {
        marks.Set(t[k] % this->NumberOfPoints, 1);
      }
    }
  }

  viskores::Id NumberOfPoints;
};

namespace ex = viskores::detail::exact;

/// What a segment meets of a closed triangle, as `segment_hits` of proofs/lib/thick.py: whether
/// it meets it, and whether in a single point interior to the triangle and to the segment.
struct SegmentHit
{
  bool Hit = false;
  bool Generic = false;
};

/// Exact arithmetic on nonoverlapping expansions, for the points that are not doubles.
namespace big
{

/// A nonoverlapping expansion of bounded length, compressed after every operation; `Overflow`
/// marks a result that did not fit.
constexpr int BigCapacity = 64;
struct Big
{
  Real C[BigCapacity];
  int N;
  bool Overflow;
};

VISKORES_EXEC inline Big BigOf(Real x)
{
  Big b;
  b.C[0] = x;
  b.N = 1;
  b.Overflow = false;
  return b;
}

/// Shewchuk's compression of the expansion `e`, increasing in magnitude, into `h`.
VISKORES_EXEC inline int Compress(int eLength, const Real* e, Real* h)
{
  int bottom = eLength - 1;
  Real q = e[bottom];
  for (int i = eLength - 2; i >= 0; --i)
  {
    Real sum, error;
    ex::FastTwoSum(q, e[i], sum, error);
    if (error != 0)
    {
      h[bottom--] = sum;
      q = error;
    }
    else
    {
      q = sum;
    }
  }
  int top = 0;
  for (int i = bottom + 1; i < eLength; ++i)
  {
    Real sum, error;
    ex::FastTwoSum(h[i], q, sum, error);
    if (error != 0)
    {
      h[top++] = error;
    }
    q = sum;
  }
  h[top] = q;
  return top + 1;
}

VISKORES_EXEC inline Big BigFrom(int length, const Real* e, bool overflow)
{
  Real h[4 * BigCapacity];
  const int n = Compress(length, e, h);
  Big b;
  b.Overflow = overflow || n > BigCapacity;
  b.N = viskores::Min(n, BigCapacity);
  for (int i = 0; i < b.N; ++i)
  {
    b.C[i] = h[i];
  }
  return b;
}

VISKORES_EXEC inline Big Add(const Big& a, const Big& b)
{
  Real h[2 * BigCapacity], scratch[2 * BigCapacity];
  const int n = ex::ExpansionSum(a.N, a.C, b.N, b.C, h, scratch);
  return BigFrom(n, h, a.Overflow || b.Overflow);
}

VISKORES_EXEC inline Big Neg(const Big& a)
{
  Big b = a;
  for (int i = 0; i < b.N; ++i)
  {
    b.C[i] = -b.C[i];
  }
  return b;
}

VISKORES_EXEC inline Big Sub(const Big& a, const Big& b)
{
  return Add(a, Neg(b));
}

VISKORES_EXEC inline Big Mul(const Big& a, const Big& b)
{
  Real acc[4 * BigCapacity], part[2 * BigCapacity], sum[6 * BigCapacity], scratch[6 * BigCapacity];
  int length = 0;
  bool overflow = a.Overflow || b.Overflow;
  for (int i = 0; i < b.N; ++i)
  {
    const int partLength = ex::ScaleExpansion(a.N, a.C, b.C[i], part);
    if (length == 0)
    {
      length = Compress(partLength, part, acc);
      continue;
    }
    const int n = ex::ExpansionSum(length, acc, partLength, part, sum, scratch);
    length = Compress(n, sum, acc);
    if (length > 2 * BigCapacity)
    {
      overflow = true;
      length = 2 * BigCapacity;
    }
  }
  return BigFrom(length, acc, overflow);
}

VISKORES_EXEC inline int Sign(const Big& a)
{
  return ex::Sign(a.C[a.N - 1]);
}

struct BigPoint
{
  Big X[3];
};

VISKORES_EXEC inline BigPoint PointOf(const Vec3& p, Real scale)
{
  BigPoint q;
  for (int i = 0; i < 3; ++i)
  {
    q.X[i] = (scale == 1) ? BigOf(p[i]) : Mul(BigOf(p[i]), BigOf(scale));
  }
  return q;
}

VISKORES_EXEC inline BigPoint Sub(const BigPoint& a, const BigPoint& b)
{
  BigPoint c;
  for (int i = 0; i < 3; ++i)
  {
    c.X[i] = Sub(a.X[i], b.X[i]);
  }
  return c;
}

VISKORES_EXEC inline BigPoint Cross(const BigPoint& a, const BigPoint& b)
{
  BigPoint c;
  c.X[0] = Sub(Mul(a.X[1], b.X[2]), Mul(a.X[2], b.X[1]));
  c.X[1] = Sub(Mul(a.X[2], b.X[0]), Mul(a.X[0], b.X[2]));
  c.X[2] = Sub(Mul(a.X[0], b.X[1]), Mul(a.X[1], b.X[0]));
  return c;
}

VISKORES_EXEC inline Big Dot(const BigPoint& a, const BigPoint& b)
{
  return Add(Add(Mul(a.X[0], b.X[0]), Mul(a.X[1], b.X[1])), Mul(a.X[2], b.X[2]));
}

/// `a s + b t`, coordinatewise.
VISKORES_EXEC inline BigPoint Combine(const BigPoint& a,
                                      const Big& s,
                                      const BigPoint& b,
                                      const Big& t)
{
  BigPoint c;
  for (int i = 0; i < 3; ++i)
  {
    c.X[i] = Add(Mul(a.X[i], s), Mul(b.X[i], t));
  }
  return c;
}

/// Whether the result of an exact evaluation fit.
VISKORES_EXEC inline bool Fits(const BigPoint& p)
{
  return !(p.X[0].Overflow || p.X[1].Overflow || p.X[2].Overflow);
}


/// `segment_triangle` of proofs/lib/exact.py with the segment in the plane of the triangle:
/// whether the segment meets the closed triangle, clipped by the three edges.
VISKORES_EXEC inline bool CoplanarSegmentMeets(const BigPoint& a,
                                               const BigPoint& b,
                                               const BigPoint* q,
                                               const BigPoint& n,
                                               bool& fits)
{
  // lo and hi as fractions with positive denominators.
  Big loN = BigOf(0), loD = BigOf(1), hiN = BigOf(1), hiD = BigOf(1);
  for (int e = 0; e < 3; ++e)
  {
    const BigPoint& u = q[e];
    const BigPoint& w = q[(e + 1) % 3];
    const BigPoint wu = Sub(w, u);
    const Big fa = Dot(n, Cross(wu, Sub(a, u)));
    const Big fb = Dot(n, Cross(wu, Sub(b, u)));
    const Big diff = Sub(fa, fb);
    fits = fits && !fa.Overflow && !fb.Overflow && !diff.Overflow;
    const int sd = Sign(diff);
    if (sd == 0)
    {
      if (Sign(fa) < 0)
      {
        return false;
      }
      continue;
    }
    // t = fa / (fa - fb); f(t) = fa + t (fb - fa) >= 0.
    const Big tN = (sd > 0) ? fa : Neg(fa);
    const Big tD = (sd > 0) ? diff : Neg(diff);
    if (sd < 0)
    {
      // fb > fa: lo = max(lo, t).
      if (Sign(Sub(Mul(tN, loD), Mul(loN, tD))) > 0)
      {
        loN = tN;
        loD = tD;
      }
    }
    else if (Sign(Sub(Mul(tN, hiD), Mul(hiN, tD))) < 0)
    {
      hiN = tN;
      hiD = tD;
    }
    if (Sign(Sub(Mul(loN, hiD), Mul(hiN, loD))) > 0)
    {
      return false;
    }
  }
  return true;
}

/// `segment_hits` of proofs/lib/thick.py for one triangle, on exact points.
VISKORES_EXEC inline SegmentHit SegmentTriangleBig(const BigPoint& a,
                                                   const BigPoint& b,
                                                   const BigPoint* q,
                                                   bool& fits)
{
  SegmentHit hit;
  const BigPoint n = Cross(Sub(q[1], q[0]), Sub(q[2], q[0]));
  const Big da = Dot(n, Sub(a, q[0]));
  const Big db = Dot(n, Sub(b, q[0]));
  fits = fits && Fits(n) && !da.Overflow && !db.Overflow;
  const int sa = Sign(da), sb = Sign(db);
  if (sa == 0 && sb == 0)
  {
    hit.Hit = CoplanarSegmentMeets(a, b, q, n, fits);
    return hit;
  }
  if (sa * sb > 0)
  {
    return hit;
  }
  // The crossing x = a + t (b - a), t = da / (da - db): (x - u)(da - db) = (a - u)(da - db)
  // + da (b - a).
  const Big den = Sub(da, db);
  const int sden = Sign(den);
  const BigPoint ba = Sub(b, a);
  bool inner = true;
  for (int e = 0; e < 3; ++e)
  {
    const BigPoint& u = q[e];
    const BigPoint& w = q[(e + 1) % 3];
    const BigPoint x = Combine(Sub(a, u), den, ba, da);
    const Big s = Dot(n, Cross(Sub(w, u), x));
    fits = fits && Fits(x) && !s.Overflow;
    const int side = Sign(s) * sden;
    if (side < 0)
    {
      return hit;
    }
    inner = inner && side > 0;
  }
  hit.Hit = true;
  hit.Generic = sa != 0 && sb != 0 && inner;
  return hit;
}

} // namespace big

/// A value computed in doubles and the same expression on absolute values, which bounds its
/// rounding error.
struct Filtered
{
  Real V;
  Real A;
};

VISKORES_EXEC inline Filtered FAdd(const Filtered& x, const Filtered& y)
{
  return Filtered{ x.V + y.V, x.A + y.A };
}

VISKORES_EXEC inline Filtered FSub(const Filtered& x, const Filtered& y)
{
  return Filtered{ x.V - y.V, x.A + y.A };
}

VISKORES_EXEC inline Filtered FMul(const Filtered& x, const Filtered& y)
{
  return Filtered{ x.V * y.V, x.A * y.A };
}

/// The sign of the exact value of `x`, or 2 if the filter cannot certify it. The absolute
/// term covers what an underflow may lose: at most 2^-1075 per operation, carried through
/// at most fifteen operations by factors below 1e147.
VISKORES_EXEC inline int FSign(const Filtered& x)
{
  return viskores::Abs(x.V) > Real(1e-14) * x.A + Real(1e-150) ? ((x.V > 0) ? 1 : -1) : 2;
}

/// What `SegmentTriangleBig` finds for the probe from `ends[0] + ends[1] + ends[2]` to
/// `ends[3] + ends[4] + ends[5]` and the triangle `3 p`, decided in doubles where every sign it
/// reads is certain: 0 for no hit, 1 for a generic hit, -1 if the exact test must decide. A
/// floating-point filter: every quantity is a polynomial in the doubles given,
/// evaluated at most fifteen operations deep, so its computed value is within `15u / (1 - 15u)`
/// of the same polynomial on absolute values, below `1e-14` times its computed value (Higham,
/// "Accuracy and Stability of Numerical Algorithms", Section 3.1). With every nonzero coordinate
/// between `1e-20` and `1e20`, the polynomials, of degree at most seven, do not overflow;
/// otherwise the exact test decides.
VISKORES_EXEC inline int ProbeHitFilter(const Vec3* ends, const Vec3* p)
{
  auto inRange = [](const Vec3& x)
  {
    for (int c = 0; c < 3; ++c)
    {
      const Real a = viskores::Abs(x[c]);
      if (a != 0 && !(a >= Real(1e-20) && a <= Real(1e20)))
      {
        return false;
      }
    }
    return true;
  };
  for (int k = 0; k < 6; ++k)
  {
    if (!inRange(ends[k]) || (k < 3 && !inRange(p[k])))
    {
      return -1;
    }
  }
  using F3 = Filtered[3];
  auto value = [](Real x) { return Filtered{ x, viskores::Abs(x) }; };
  F3 a, b, q[3];
  for (int c = 0; c < 3; ++c)
  {
    a[c] = FAdd(FAdd(value(ends[0][c]), value(ends[1][c])), value(ends[2][c]));
    b[c] = FAdd(FAdd(value(ends[3][c]), value(ends[4][c])), value(ends[5][c]));
    for (int k = 0; k < 3; ++k)
    {
      q[k][c] = FMul(value(p[k][c]), value(3));
    }
  }
  auto sub = [](const F3& x, const F3& y, F3& out)
  {
    for (int c = 0; c < 3; ++c)
    {
      out[c] = FSub(x[c], y[c]);
    }
  };
  auto cross = [](const F3& x, const F3& y, F3& out)
  {
    out[0] = FSub(FMul(x[1], y[2]), FMul(x[2], y[1]));
    out[1] = FSub(FMul(x[2], y[0]), FMul(x[0], y[2]));
    out[2] = FSub(FMul(x[0], y[1]), FMul(x[1], y[0]));
  };
  auto dot = [](const F3& x, const F3& y)
  { return FAdd(FAdd(FMul(x[0], y[0]), FMul(x[1], y[1])), FMul(x[2], y[2])); };
  F3 e1, e2, n, aq, bq;
  sub(q[1], q[0], e1);
  sub(q[2], q[0], e2);
  cross(e1, e2, n);
  sub(a, q[0], aq);
  sub(b, q[0], bq);
  const Filtered da = dot(n, aq), db = dot(n, bq);
  const int sa = FSign(da), sb = FSign(db);
  if (sa == 2 || sb == 2)
  {
    return -1;
  }
  if (sa == sb)
  {
    return 0;
  }
  // The signs of da and db are opposite, so den = da - db has the sign of da.
  const Filtered den = FSub(da, db);
  const int sden = sa;
  F3 ba;
  sub(b, a, ba);
  bool unknown = false;
  for (int e = 0; e < 3; ++e)
  {
    F3 au, wu, x, c;
    sub(a, q[e], au);
    sub(q[(e + 1) % 3], q[e], wu);
    for (int k = 0; k < 3; ++k)
    {
      x[k] = FAdd(FMul(au[k], den), FMul(ba[k], da));
    }
    cross(wu, x, c);
    const int side = FSign(dot(n, c));
    if (side == 2)
    {
      unknown = true;
    }
    else if (side * sden < 0)
    {
      return 0;
    }
  }
  return unknown ? -1 : 1;
}

/// The offset cap, per side: `|B^+- - v|^2 > eps^2`, exactly.
struct ExceedsCap : viskores::worklet::WorkletMapField
{
  using ControlSignature =
    void(FieldIn point, FieldIn plus, FieldIn minus, FieldInOut badPlus, FieldInOut badMinus);
  using ExecutionSignature = void(_1, _2, _3, _4, _5);

  VISKORES_CONT explicit ExceedsCap(Real epsilon)
    : Epsilon(epsilon)
  {
  }

  VISKORES_EXEC void operator()(const Vec3& v,
                                const Vec3& plus,
                                const Vec3& minus,
                                viskores::UInt8& badPlus,
                                viskores::UInt8& badMinus) const
  {
    const big::Big eps2 = big::Mul(big::BigOf(this->Epsilon), big::BigOf(this->Epsilon));
    for (int side = 0; side < 2; ++side)
    {
      const Vec3& b = (side == 0) ? plus : minus;
      big::Big d2 = big::BigOf(0);
      for (int i = 0; i < 3; ++i)
      {
        const big::Big d = big::Sub(big::BigOf(b[i]), big::BigOf(v[i]));
        d2 = big::Add(d2, big::Mul(d, d));
      }
      if (big::Sign(big::Sub(d2, eps2)) > 0)
      {
        (side == 0 ? badPlus : badMinus) = 1;
      }
    }
  }

  Real Epsilon;
};

/// The union of the failures of the two sides, for the symmetric halving.
struct UniteSides : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldInOut plus, FieldInOut minus);
  using ExecutionSignature = void(_1, _2);

  VISKORES_EXEC void operator()(viskores::UInt8& plus, viskores::UInt8& minus) const
  {
    plus = minus = (plus || minus) ? 1 : 0;
  }
};

/// Halves the offset of a failing vertex.
struct HalveOffset : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn bad, FieldInOut delta);
  using ExecutionSignature = void(_1, _2);

  VISKORES_EXEC void operator()(viskores::UInt8 bad, Real& delta) const
  {
    if (bad)
    {
      delta = delta / 2;
    }
  }
};

/// The smaller of the two offsets of a vertex.
struct SmallerOffset : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn plus, FieldIn minus, FieldOut smaller);
  using ExecutionSignature = _3(_1, _2);

  VISKORES_EXEC Real operator()(Real plus, Real minus) const { return viskores::Min(plus, minus); }
};

/// Whether the box of a segment, padded, meets the box `[lo, hi]`.
VISKORES_EXEC inline bool SegmentBoxMeets(const Vec3& a,
                                          const Vec3& b,
                                          Real pad,
                                          const Vec3& lo,
                                          const Vec3& hi)
{
  for (int i = 0; i < 3; ++i)
  {
    if (viskores::Min(a[i], b[i]) - pad > hi[i] || viskores::Max(a[i], b[i]) + pad < lo[i])
    {
      return false;
    }
  }
  return true;
}

/// Whether the segment `ab`, padded, may meet the box `[lo, hi]`: a slab test with a margin,
/// so that a box it meets is never rejected.
VISKORES_EXEC inline bool SegmentMayMeetBox(const Vec3& a,
                                            const Vec3& b,
                                            Real pad,
                                            const Vec3& lo,
                                            const Vec3& hi)
{
  if (!SegmentBoxMeets(a, b, pad, lo, hi))
  {
    return false;
  }
  Real t0 = 0, t1 = 1;
  for (int i = 0; i < 3; ++i)
  {
    const Real d = b[i] - a[i];
    const Real l = lo[i] - pad, h = hi[i] + pad;
    if (viskores::Abs(d) <= pad)
    {
      continue;
    }
    Real u = (l - a[i]) / d, w = (h - a[i]) / d;
    if (u > w)
    {
      const Real z = u;
      u = w;
      w = z;
    }
    // Widened against the rounding of the quotients.
    u -= Real(1e-9);
    w += Real(1e-9);
    t0 = viskores::Max(t0, u);
    t1 = viskores::Min(t1, w);
    if (t0 > t1)
    {
      return false;
    }
  }
  return true;
}

/// `segment_hits` for one triangle on doubles: by the exact orientation, and in expansion
/// arithmetic for a segment in the plane of the triangle.
VISKORES_EXEC inline SegmentHit SegmentTriangle(const Vec3& a,
                                                const Vec3& b,
                                                const Vec3& q0,
                                                const Vec3& q1,
                                                const Vec3& q2,
                                                bool& fits)
{
  SegmentHit hit;
  const viskores::IdComponent da = viskores::Orient3D(q0, q1, q2, a);
  const viskores::IdComponent db = viskores::Orient3D(q0, q1, q2, b);
  if (da == 0 && db == 0)
  {
    const big::BigPoint q[3] = { big::PointOf(q0, 1), big::PointOf(q1, 1), big::PointOf(q2, 1) };
    const big::BigPoint n = big::Cross(big::Sub(q[1], q[0]), big::Sub(q[2], q[0]));
    hit.Hit = big::CoplanarSegmentMeets(big::PointOf(a, 1), big::PointOf(b, 1), q, n, fits);
    return hit;
  }
  if (da * db > 0)
  {
    return hit;
  }
  // Crossing the plane: inside the closed triangle iff the three orientations of the segment
  // with the edges agree (the triangle is not degenerate, since a and b are not both in its
  // plane).
  const viskores::IdComponent s0 = viskores::Orient3D(a, b, q0, q1);
  const viskores::IdComponent s1 = viskores::Orient3D(a, b, q1, q2);
  const viskores::IdComponent s2 = viskores::Orient3D(a, b, q2, q0);
  const bool positive = s0 >= 0 && s1 >= 0 && s2 >= 0;
  const bool negative = s0 <= 0 && s1 <= 0 && s2 <= 0;
  if (!positive && !negative)
  {
    return hit;
  }
  hit.Hit = true;
  hit.Generic = da != 0 && db != 0 && s0 != 0 && s1 != 0 && s2 != 0;
  return hit;
}

/// The crossings of a segment with the sides, counted, and whether all are generic.
template <typename SidePortals>
VISKORES_EXEC inline void SideCrossings(const Vec3& a,
                                        const Vec3& b,
                                        const SidePortals& sides,
                                        viskores::Id& count,
                                        bool& generic,
                                        bool& complete,
                                        bool& fits)
{
  count = 0;
  generic = true;
  const Real pad = Real(1e-9) *
    viskores::Max(Real(1),
                  viskores::Max(sides.MaxAbsCoordinate, viskores::Max(MaxAbs(a), MaxAbs(b))));
  const bool done = sides.SideHierarchy.Traverse(
    [&](viskores::Id node)
    {
      return SegmentMayMeetBox(
        a, b, pad, sides.SideHierarchy.Lower.Get(node), sides.SideHierarchy.Upper.Get(node));
    },
    [&](viskores::Id f)
    {
      const viskores::Id3 t = sides.SideTriangles.Get(f);
      const SegmentHit hit = SegmentTriangle(
        a, b, sides.Sides.Get(t[0]), sides.Sides.Get(t[1]), sides.Sides.Get(t[2]), fits);
      if (hit.Hit)
      {
        ++count;
        generic = generic && hit.Generic;
      }
      return true;
    });
  complete = complete && done;
}

/// Whether two closed triangles meet, as `_pair_improper` of proofs/lib/exact.py with no shared
/// label: an edge of one meets the other.
VISKORES_EXEC inline bool TrianglesMeet(const Vec3* p, const Vec3* q, bool& fits)
{
  for (int which = 0; which < 2; ++which)
  {
    const Vec3* x = (which == 0) ? p : q;
    const Vec3* y = (which == 0) ? q : p;
    for (int i = 0; i < 3; ++i)
    {
      if (SegmentTriangle(x[i], x[(i + 1) % 3], y[0], y[1], y[2], fits).Hit)
      {
        return true;
      }
    }
  }
  return false;
}

} // namespace thickening_detail

/// The sides of a thickening in the execution environment, and its probes.
struct ThickeningPortals
{
  using VecPortal = viskores::cont::ArrayHandle<Vec3>::ReadPortalType;
  using IdPortal = viskores::cont::ArrayHandle<viskores::Id>::ReadPortalType;
  using Id3Portal = viskores::cont::ArrayHandle<viskores::Id3>::ReadPortalType;
  using ProbePortal = viskores::cont::ArrayHandle<viskores::Vec<Vec3, 6>>::ReadPortalType;
  using HierarchyPortal =
    viskores::worklet::spatialstructure::BoundingVolumeHierarchy::ExecObjectType;

  /// Exact ray parity of `p` against the sides, as `Thickening.inside`: 1 inside T, 0 outside,
  /// and 0 if no tried ray is generic.
  VISKORES_EXEC bool Inside(const Vec3& p, bool& complete, bool& fits) const
  {
    const Real directions[3][3] = { { 0.123, 0.317, 1.0 },
                                    { 0.711, -0.219, 0.5 },
                                    { -0.37, 0.83, 0.29 } };
    const Real rayLength = 10 * this->Span;
    for (int r = 0; r < 3; ++r)
    {
      const Vec3 far(p[0] + rayLength * directions[r][0],
                     p[1] + rayLength * directions[r][1],
                     p[2] + rayLength * directions[r][2]);
      viskores::Id count;
      bool generic;
      thickening_detail::SideCrossings(p, far, *this, count, generic, complete, fits);
      if (generic)
      {
        return count % 2 == 1;
      }
    }
    return false;
  }

  /// The ends of probe `i`, scaled by three: the sums of the stored sides at the corners of its
  /// triangle.
  VISKORES_EXEC void ProbeEnds(viskores::Id i,
                               thickening_detail::big::BigPoint& a,
                               thickening_detail::big::BigPoint& b) const
  {
    namespace big = thickening_detail::big;
    const viskores::Vec<Vec3, 6> q = this->ProbePoints.Get(i);
    for (int c = 0; c < 3; ++c)
    {
      a.X[c] = big::Add(big::Add(big::BigOf(q[0][c]), big::BigOf(q[1][c])), big::BigOf(q[2][c]));
      b.X[c] = big::Add(big::Add(big::BigOf(q[3][c]), big::BigOf(q[4][c])), big::BigOf(q[5][c]));
    }
  }

  /// What probe `i` meets of the closed triangle `p`.
  VISKORES_EXEC thickening_detail::SegmentHit ProbeHit(viskores::Id i,
                                                       const Vec3* p,
                                                       bool& fits) const
  {
    namespace big = thickening_detail::big;
    // A cheap rejection by boxes: the probe lies in the box of the six stored points at its
    // corners.
    const viskores::Vec<Vec3, 6> q = this->ProbePoints.Get(i);
    Vec3 lo = q[0], hi = q[0];
    for (int c = 1; c < 6; ++c)
    {
      for (int a = 0; a < 3; ++a)
      {
        lo[a] = viskores::Min(lo[a], q[c][a]);
        hi[a] = viskores::Max(hi[a], q[c][a]);
      }
    }
    for (int a = 0; a < 3; ++a)
    {
      const Real pmin = viskores::Min(viskores::Min(p[0][a], p[1][a]), p[2][a]);
      const Real pmax = viskores::Max(viskores::Max(p[0][a], p[1][a]), p[2][a]);
      if (pmin > hi[a] || pmax < lo[a])
      {
        return thickening_detail::SegmentHit{};
      }
    }
    const Vec3 ends[6] = { q[0], q[1], q[2], q[3], q[4], q[5] };
    const int filtered = thickening_detail::ProbeHitFilter(ends, p);
    if (filtered >= 0)
    {
      return thickening_detail::SegmentHit{ filtered == 1, filtered == 1 };
    }
    big::BigPoint a, b;
    this->ProbeEnds(i, a, b);
    const big::BigPoint t[3] = { big::PointOf(p[0], 3),
                                 big::PointOf(p[1], 3),
                                 big::PointOf(p[2], 3) };
    return big::SegmentTriangleBig(a, b, t, fits);
  }

  /// The box of probe `i`: its exact ends rounded outward to doubles, as `blame_regions` of
  /// proofs/lib/vqc.py.
  VISKORES_EXEC void ProbeBox(viskores::Id i, Vec3& lo, Vec3& hi) const
  {
    namespace big = thickening_detail::big;
    const viskores::Vec<Vec3, 6> q = this->ProbePoints.Get(i);
    for (int a = 0; a < 3; ++a)
    {
      for (int e = 0; e < 2; ++e)
      {
        // The end is a third of the exact sum of the stored points at the three corners.
        const big::Big sum =
          big::Add(big::Add(big::BigOf(q[3 * e][a]), big::BigOf(q[3 * e + 1][a])),
                   big::BigOf(q[3 * e + 2][a]));
        // The sign of 3 x - sum, exactly.
        auto compare = [&](Real x)
        {
          const big::Big y = big::BigOf(x);
          return big::Sign(big::Sub(big::Add(big::Add(y, y), y), sum));
        };
        Real down = (q[3 * e][a] + q[3 * e + 1][a] + q[3 * e + 2][a]) / 3;
        while (compare(down) > 0)
        {
          down = -NextUp(-down);
        }
        while (compare(NextUp(down)) <= 0)
        {
          down = NextUp(down);
        }
        const Real up = compare(down) == 0 ? down : NextUp(down);
        lo[a] = (e == 0) ? down : viskores::Min(lo[a], down);
        hi[a] = (e == 0) ? up : viskores::Max(hi[a], up);
      }
    }
  }

  /// Whether the closed triangle `p` meets a side, exactly.
  VISKORES_EXEC bool MeetsSides(const Vec3* p, bool& complete, bool& fits) const
  {
    Vec3 lo, hi;
    TriangleBounds(p[0], p[1], p[2], lo, hi);
    bool meets = false;
    const bool done = this->SideHierarchy.ForEachOverlap(
      lo,
      hi,
      [&](viskores::Id f)
      {
        const viskores::Id3 t = this->SideTriangles.Get(f);
        const Vec3 q[3] = { this->Sides.Get(t[0]), this->Sides.Get(t[1]), this->Sides.Get(t[2]) };
        if (thickening_detail::TrianglesMeet(p, q, fits))
        {
          meets = true;
          return false;
        }
        return true;
      });
    complete = complete && done;
    return meets;
  }

  VecPortal Sides;
  Id3Portal SideTriangles;
  HierarchyPortal SideHierarchy;
  IdPortal ProbeTriangles;
  ProbePortal ProbePoints;
  Real Span;
  Real MaxAbsCoordinate;
};

struct ThickeningObject : viskores::cont::ExecutionObjectBase
{
  explicit ThickeningObject(const Thickening& thickening)
    : Source(&thickening)
  {
  }

  VISKORES_CONT ThickeningPortals PrepareForExecution(viskores::cont::DeviceAdapterId device,
                                                      viskores::cont::Token& token) const
  {
    ThickeningPortals portals;
    portals.Sides = this->Source->Sides.PrepareForInput(device, token);
    portals.SideTriangles = this->Source->SideTriangles.PrepareForInput(device, token);
    portals.SideHierarchy = this->Source->SideHierarchy.PrepareForExecution(device, token);
    portals.ProbeTriangles = this->Source->ProbeTriangles.PrepareForInput(device, token);
    portals.ProbePoints = this->Source->ProbePoints.PrepareForInput(device, token);
    portals.Span = this->Source->Span;
    portals.MaxAbsCoordinate = this->Source->MaxAbsCoordinate;
    return portals;
  }

  const Thickening* Source;
};

namespace thickening_detail
{

/// A triangle that `ok` admits as `(component of its first vertex, triangle)`, or `(-1, -1)`:
/// the smallest per component is its first admitted triangle.
struct FirstTriangleCandidate : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn index,
                                FieldIn ok,
                                WholeArrayIn triangles,
                                WholeArrayIn components,
                                FieldOut candidate);
  using ExecutionSignature = _5(_1, _2, _3, _4);

  template <typename Id3Portal, typename IdPortal>
  VISKORES_EXEC viskores::Id2 operator()(viskores::Id f,
                                         viskores::UInt8 ok,
                                         const Id3Portal& triangles,
                                         const IdPortal& components) const
  {
    return ok ? viskores::Id2(components.Get(triangles.Get(f)[0]), f) : viskores::Id2(-1);
  }
};

/// For (T3): the side triangle of a try: the first triangle of its component on `B^+`, or on
/// `B^-` reversed.
struct DegreeTriangle : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn index,
                                WholeArrayIn first,
                                WholeArrayIn sideTriangles,
                                FieldOut triangle);
  using ExecutionSignature = _4(_1, _2, _3);

  template <typename IdPortal, typename Id3Portal>
  VISKORES_EXEC viskores::Id3 operator()(viskores::Id i,
                                         const IdPortal& first,
                                         const Id3Portal& sideTriangles) const
  {
    const viskores::Id numComponents = first.GetNumberOfValues();
    const viskores::Id f = first.Get(i % numComponents);
    const viskores::Id m = sideTriangles.GetNumberOfValues() / 2;
    return sideTriangles.Get((i < numComponents) ? f : f + m);
  }
};

/// For (T3): per component of M and side, a try of the degree test at the first triangle of
/// the component: the inner and outer points `cen -+ eta N`, from `cen` the centroid and `N`
/// the unit normal of the side triangle.
struct DegreePoints : viskores::worklet::WorkletMapField
{
  using ControlSignature =
    void(FieldIn sideTriangle, FieldIn eta, WholeArrayIn sides, FieldOut inner, FieldOut outer);
  using ExecutionSignature = void(_1, _2, _3, _4, _5);

  template <typename VecPortal>
  VISKORES_EXEC void operator()(const viskores::Id3& t,
                                Real eta,
                                const VecPortal& sides,
                                Vec3& inner,
                                Vec3& outer) const
  {
    const Vec3 a = sides.Get(t[0]), b = sides.Get(t[1]), c = sides.Get(t[2]);
    const Vec3 s = Add(Add(a, b), c);
    const Vec3 cen(s[0] / 3, s[1] / 3, s[2] / 3);
    const Vec3 n = Cross(Sub(b, a), Sub(c, a));
    const Real norm = Norm(n);
    const Vec3 unit(n[0] / norm, n[1] / norm, n[2] / norm);
    inner = Sub(cen, Vec3(eta * unit[0], eta * unit[1], eta * unit[2]));
    outer = Add(cen, Vec3(eta * unit[0], eta * unit[1], eta * unit[2]));
  }
};

/// For (T3): the crossings of the segment between the inner and outer points of a try with the
/// sides.
struct DegreeCrossings : viskores::worklet::WorkletMapField
{
  using ControlSignature =
    void(FieldIn inner, FieldIn outer, ExecObject sides, FieldOut count, FieldOut generic);
  using ExecutionSignature = void(_1, _2, _3, _4, _5);

  template <typename Sides>
  VISKORES_EXEC void operator()(const Vec3& inner,
                                const Vec3& outer,
                                const Sides& sides,
                                viskores::Id& count,
                                viskores::UInt8& generic) const
  {
    bool complete = true, fits = true, all = true;
    SideCrossings(inner, outer, sides, count, all, complete, fits);
    generic = all ? 1 : 0;
    if (!complete || !fits)
    {
      this->RaiseError("QuadricClusteringDecimation: exact evaluation out of range.");
    }
  }
};

/// For (T3): per try point and tetrahedron of the thickening (item `point * 6 m + tet`), 2 if
/// the point is on the boundary of the tetrahedron, 1 if inside, 0 otherwise, as
/// `cover_count`.
struct TetrahedronCover : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn item,
                                WholeArrayIn queries,
                                WholeArrayIn triangles,
                                WholeArrayIn points,
                                WholeArrayIn plus,
                                WholeArrayIn minus,
                                FieldOut status);
  using ExecutionSignature = _7(_1, _2, _3, _4, _5, _6);

  template <typename VecPortal, typename Id3Portal>
  VISKORES_EXEC viskores::UInt8 operator()(viskores::Id item,
                                           const VecPortal& queries,
                                           const Id3Portal& triangles,
                                           const VecPortal& points,
                                           const VecPortal& plus,
                                           const VecPortal& minus) const
  {
    const viskores::Id numTets = 6 * triangles.GetNumberOfValues();
    const Vec3 q = queries.Get(item / numTets);
    const viskores::Id tet = item % numTets;
    const viskores::Id f = tet / 6;
    const int layer = static_cast<int>((tet % 6) / 3), a = static_cast<int>(tet % 3);
    viskores::Id vertex[3][4];
    int level[3][4];
    PrismTetrahedra(triangles.Get(f), vertex, level);
    Vec3 p[4];
    for (int b = 0; b < 4; ++b)
    {
      const viskores::Id v = vertex[a][b];
      const Vec3 x = points.Get(v);
      p[b] = (layer == 0) ? ((level[a][b] == 0) ? minus.Get(v) : x)
                          : ((level[a][b] == 0) ? x : plus.Get(v));
    }
    for (int k = 0; k < 3; ++k)
    {
      const Real lo =
        viskores::Min(viskores::Min(p[0][k], p[1][k]), viskores::Min(p[2][k], p[3][k]));
      const Real hi =
        viskores::Max(viskores::Max(p[0][k], p[1][k]), viskores::Max(p[2][k], p[3][k]));
      if (q[k] < lo || q[k] > hi)
      {
        return 0;
      }
    }
    const viskores::IdComponent s = viskores::Orient3D(p[0], p[1], p[2], p[3]);
    const viskores::IdComponent sub[4] = { viskores::Orient3D(q, p[1], p[2], p[3]),
                                           viskores::Orient3D(p[0], q, p[2], p[3]),
                                           viskores::Orient3D(p[0], p[1], q, p[3]),
                                           viskores::Orient3D(p[0], p[1], p[2], q) };
    bool zero = false, agree = true;
    for (int k = 0; k < 4; ++k)
    {
      zero = zero || sub[k] == 0;
      agree = agree && (sub[k] == 0 || (sub[k] > 0) == (s > 0));
    }
    if (zero)
    {
      return agree ? 2 : 0;
    }
    return agree ? 1 : 0;
  }
};

struct StatusCounts : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn status, FieldOut inside, FieldOut boundary);
  using ExecutionSignature = void(_1, _2, _3);

  VISKORES_EXEC void operator()(viskores::UInt8 status,
                                viskores::Id& inside,
                                viskores::Id& boundary) const
  {
    inside = (status == 1) ? 1 : 0;
    boundary = (status == 2) ? 1 : 0;
  }
};

/// The cover of a try point: -1 if it lies on the boundary of a tetrahedron, else the number
/// containing it.
struct CoverOf : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn inside, FieldIn boundary, FieldOut cover);
  using ExecutionSignature = _3(_1, _2);

  VISKORES_EXEC viskores::Id operator()(viskores::Id inside, viskores::Id boundary) const
  {
    return (boundary > 0) ? viskores::Id(-1) : inside;
  }
};

/// For (T3), a try accepted when the segment crosses the sides once, generically, and neither
/// point lies on the boundary of a tetrahedron: it passes if the inner point is covered once
/// and the outer point by none.
struct DegreeVerdict : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn index,
                                FieldIn count,
                                FieldIn generic,
                                WholeArrayIn covers,
                                FieldInOut settled,
                                FieldInOut passed);
  using ExecutionSignature = void(_1, _2, _3, _4, _5, _6);

  template <typename IdPortal>
  VISKORES_EXEC void operator()(viskores::Id i,
                                viskores::Id count,
                                viskores::UInt8 generic,
                                const IdPortal& covers,
                                viskores::UInt8& settled,
                                viskores::UInt8& passed) const
  {
    if (settled)
    {
      return;
    }
    const viskores::Id n = covers.GetNumberOfValues() / 2;
    const viskores::Id inner = covers.Get(i), outer = covers.Get(n + i);
    if (count == 1 && generic && inner >= 0 && outer >= 0)
    {
      settled = 1;
      passed = (inner == 1 && outer == 0) ? 1 : 0;
    }
  }
};

/// (T3): the degree test of every try, a side triangle of `tryTriangles` on the stored points
/// `sides`, against the tetrahedra of the prisms over `triangles` on `points`, `plus` and
/// `minus`. The step `eta` starts at a thousandth of the smallest offset of `delta` and is
/// divided by four until every try is settled, up to 20 times. Whether every try passes.
inline bool DegreeTestPasses(const viskores::cont::ArrayHandle<viskores::Id3>& tryTriangles,
                             const viskores::cont::ArrayHandle<Vec3>& sides,
                             const ThickeningObject& sidesObject,
                             const viskores::cont::ArrayHandle<viskores::Id3>& triangles,
                             const viskores::cont::ArrayHandle<Vec3>& points,
                             const viskores::cont::ArrayHandle<Vec3>& plus,
                             const viskores::cont::ArrayHandle<Vec3>& minus,
                             const viskores::cont::ArrayHandle<Real>& delta)
{
  using Algorithm = viskores::cont::Algorithm;
  viskores::cont::Invoker invoke;
  const viskores::Id numTries = tryTriangles.GetNumberOfValues();
  const viskores::Id m = triangles.GetNumberOfValues();
  Real eta = Real(1e-3) *
    Algorithm::Reduce(delta, std::numeric_limits<Real>::infinity(), viskores::Minimum());
  viskores::cont::ArrayHandle<viskores::UInt8> settled, passed;
  settled.AllocateAndFill(numTries, 0);
  passed.AllocateAndFill(numTries, 0);
  for (int attempt = 0; attempt < 20; ++attempt)
  {
    viskores::cont::ArrayHandle<Vec3> inner, outer, queries;
    invoke(DegreePoints{},
           tryTriangles,
           viskores::cont::make_ArrayHandleConstant(eta, numTries),
           sides,
           inner,
           outer);
    viskores::cont::ArrayHandle<viskores::Id> counts;
    viskores::cont::ArrayHandle<viskores::UInt8> generic;
    invoke(DegreeCrossings{}, inner, outer, sidesObject, counts, generic);
    queries.Allocate(2 * numTries);
    Algorithm::CopySubRange(inner, 0, numTries, queries, 0);
    Algorithm::CopySubRange(outer, 0, numTries, queries, numTries);
    const viskores::Id numItems = 2 * numTries * 6 * m;
    viskores::cont::ArrayHandle<viskores::UInt8> status;
    invoke(TetrahedronCover{},
           viskores::cont::ArrayHandleIndex(numItems),
           queries,
           triangles,
           points,
           plus,
           minus,
           status);
    viskores::cont::ArrayHandle<viskores::Id> insideItems, boundaryItems, owners, keys, inside,
      boundary, covers;
    invoke(StatusCounts{}, status, insideItems, boundaryItems);
    invoke(DivideBy(6 * m), viskores::cont::ArrayHandleIndex(numItems), owners);
    Algorithm::ReduceByKey(owners, insideItems, keys, inside, viskores::Add());
    Algorithm::ReduceByKey(owners, boundaryItems, keys, boundary, viskores::Add());
    invoke(CoverOf{}, inside, boundary, covers);
    // Settle the tries that are accepted: one generic crossing, neither point on a boundary.
    invoke(DegreeVerdict{},
           viskores::cont::ArrayHandleIndex(numTries),
           counts,
           generic,
           covers,
           settled,
           passed);
    if (Algorithm::Reduce(settled, viskores::Id(0)) == numTries)
    {
      break;
    }
    eta /= 4;
  }
  return Algorithm::Reduce(passed, viskores::Id(0)) == numTries;
}

/// For (T4): whether the probe of a triangle of M meets |M| exactly once, in a single point
/// interior to that triangle and to the probe.
struct ProbeCandidate : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn index,
                                WholeArrayIn triangles,
                                WholeArrayIn points,
                                WholeArrayIn plus,
                                WholeArrayIn minus,
                                ExecObject hierarchy,
                                FieldOut ok);
  using ExecutionSignature = _7(_1, _2, _3, _4, _5, _6);

  template <typename Id3Portal, typename VecPortal, typename Hierarchy>
  VISKORES_EXEC viskores::UInt8 operator()(viskores::Id f,
                                           const Id3Portal& triangles,
                                           const VecPortal& points,
                                           const VecPortal& plus,
                                           const VecPortal& minus,
                                           const Hierarchy& hierarchy) const
  {
    const viskores::Id3 t = triangles.Get(f);
    big::BigPoint a, b;
    Vec3 lo, hi;
    for (int c = 0; c < 3; ++c)
    {
      a.X[c] = big::Add(big::Add(big::BigOf(minus.Get(t[0])[c]), big::BigOf(minus.Get(t[1])[c])),
                        big::BigOf(minus.Get(t[2])[c]));
      b.X[c] = big::Add(big::Add(big::BigOf(plus.Get(t[0])[c]), big::BigOf(plus.Get(t[1])[c])),
                        big::BigOf(plus.Get(t[2])[c]));
      lo[c] = viskores::Min(
        viskores::Min(minus.Get(t[0])[c], minus.Get(t[1])[c]),
        viskores::Min(
          minus.Get(t[2])[c],
          viskores::Min(viskores::Min(plus.Get(t[0])[c], plus.Get(t[1])[c]), plus.Get(t[2])[c])));
      hi[c] = viskores::Max(
        viskores::Max(minus.Get(t[0])[c], minus.Get(t[1])[c]),
        viskores::Max(
          minus.Get(t[2])[c],
          viskores::Max(viskores::Max(plus.Get(t[0])[c], plus.Get(t[1])[c]), plus.Get(t[2])[c])));
    }
    const Vec3 ends[6] = { minus.Get(t[0]), minus.Get(t[1]), minus.Get(t[2]),
                           plus.Get(t[0]),  plus.Get(t[1]),  plus.Get(t[2]) };
    viskores::Id count = 0;
    bool onlyHere = true, fits = true;
    const bool complete = hierarchy.ForEachOverlap(
      lo,
      hi,
      [&](viskores::Id g)
      {
        const viskores::Id3 u = triangles.Get(g);
        const Vec3 p[3] = { points.Get(u[0]), points.Get(u[1]), points.Get(u[2]) };
        const int filtered = ProbeHitFilter(ends, p);
        SegmentHit hit{ filtered == 1, filtered == 1 };
        if (filtered < 0)
        {
          const big::BigPoint q[3] = { big::PointOf(points.Get(u[0]), 3),
                                       big::PointOf(points.Get(u[1]), 3),
                                       big::PointOf(points.Get(u[2]), 3) };
          hit = big::SegmentTriangleBig(a, b, q, fits);
        }
        if (hit.Hit)
        {
          ++count;
          onlyHere = onlyHere && g == f && hit.Generic;
        }
        return count <= 1;
      });
    if (!complete || !fits)
    {
      this->RaiseError("QuadricClusteringDecimation: exact evaluation out of range.");
    }
    return (count == 1 && onlyHere) ? 1 : 0;
  }
};

struct ProbeCorners : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn triangle,
                                WholeArrayIn triangles,
                                WholeArrayIn plus,
                                WholeArrayIn minus,
                                FieldOut corners);
  using ExecutionSignature = _5(_1, _2, _3, _4);

  template <typename Id3Portal, typename VecPortal>
  VISKORES_EXEC viskores::Vec<Vec3, 6> operator()(viskores::Id f,
                                                  const Id3Portal& triangles,
                                                  const VecPortal& plus,
                                                  const VecPortal& minus) const
  {
    const viskores::Id3 t = triangles.Get(f);
    return viskores::Vec<Vec3, 6>(minus.Get(t[0]),
                                  minus.Get(t[1]),
                                  minus.Get(t[2]),
                                  plus.Get(t[0]),
                                  plus.Get(t[1]),
                                  plus.Get(t[2]));
  }
};

/// Whether a point differs, bit for bit, from the one of the round before.
struct PointMoved : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn point, FieldIn before, FieldOut moved);
  using ExecutionSignature = _3(_1, _2);

  VISKORES_EXEC viskores::UInt8 operator()(const Vec3& point, const Vec3& before) const
  {
    return std::memcmp(&point, &before, sizeof(Vec3)) != 0 ? 1 : 0;
  }
};

/// Whether a triangle has a moved vertex.
struct TriangleMoved : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn triangle, WholeArrayIn moved, FieldOut triangleMoved);
  using ExecutionSignature = _3(_1, _2);

  template <typename FlagPortal>
  VISKORES_EXEC viskores::UInt8 operator()(const viskores::Id3& t, const FlagPortal& moved) const
  {
    return (moved.Get(t[0]) || moved.Get(t[1]) || moved.Get(t[2])) ? 1 : 0;
  }
};

/// The improper pairs `(i, j)` of `FindImproperPairs` that a triangle `query` takes part in, with
/// the first triangle the one queried, as there: `(query, j)`, and, for a triangle `j` that did
/// not move, `(j, query)` too, whose test reads no moved point but the query's. Without
/// `Fill`, counts them.
template <bool Fill>
struct ImproperPairsOf : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn query,
                                FieldIn offset,
                                WholeArrayIn triangles,
                                WholeArrayIn lower,
                                WholeArrayIn upper,
                                WholeArrayIn degenerates,
                                WholeArrayIn moved,
                                WholeArrayIn points,
                                ExecObject hierarchy,
                                FieldOut count,
                                WholeArrayOut pairs);
  using ExecutionSignature = void(_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11);

  template <typename Id3Portal,
            typename VecPortal,
            typename FlagPortal,
            typename Hierarchy,
            typename PairPortal>
  VISKORES_EXEC void operator()(viskores::Id i,
                                viskores::Id offset,
                                const Id3Portal& triangles,
                                const VecPortal& lower,
                                const VecPortal& upper,
                                const FlagPortal& degenerates,
                                const FlagPortal& moved,
                                const VecPortal& points,
                                const Hierarchy& hierarchy,
                                viskores::Id& count,
                                const PairPortal& pairs) const
  {
    count = 0;
    if (degenerates.Get(i))
    {
      return;
    }
    const viskores::Id3 triangle = triangles.Get(i);
    const Vec3 p[3] = { points.Get(triangle[0]), points.Get(triangle[1]), points.Get(triangle[2]) };
    auto emit = [&](viskores::Id first, viskores::Id second)
    {
      if (Fill)
      {
        pairs.Set(offset + count, viskores::Id2(first, second));
      }
      ++count;
    };
    const bool complete = hierarchy.ForEachOverlap(
      lower.Get(i),
      upper.Get(i),
      [&](viskores::Id j)
      {
        if (j == i || degenerates.Get(j))
        {
          return true;
        }
        const viskores::Id3 other = triangles.Get(j);
        const Vec3 q[3] = { points.Get(other[0]), points.Get(other[1]), points.Get(other[2]) };
        if (exact::ImproperPair(p, triangle, q, other))
        {
          emit(i, j);
        }
        if (!moved.Get(j) && exact::ImproperPair(q, other, p, triangle))
        {
          emit(j, i);
        }
        return true;
      });
    if (!complete)
    {
      this->RaiseError("QuadricClusteringDecimation: bounding volume hierarchy too deep.");
    }
  }
};

/// Marks the first triangle of every pair.
struct MarkFirst : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn pair, WholeArrayInOut marks);
  using ExecutionSignature = void(_1, _2);

  template <typename FlagPortal>
  VISKORES_EXEC void operator()(const viskores::Id2& pair, FlagPortal& marks) const
  {
    marks.Set(pair[0], 1);
  }
};

/// Whether neither triangle of a pair moved.
struct PairKept : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn pair, WholeArrayIn moved, FieldOut kept);
  using ExecutionSignature = _3(_1, _2);

  template <typename FlagPortal>
  VISKORES_EXEC viskores::UInt8 operator()(const viskores::Id2& pair, const FlagPortal& moved) const
  {
    return (moved.Get(pair[0]) || moved.Get(pair[1])) ? 0 : 1;
  }
};

} // namespace thickening_detail

/// The improper pairs of the side triangles of a thickening, kept across its rounds of halving,
/// in which few points move: the pairs of two triangles that did not move are as they were.
struct SidePairs
{
  /// The triangles of the sides, whether each meets another improperly or is degenerate, as
  /// `involved` of `FindImproperPairs`, on the points `sides`.
  viskores::cont::ArrayHandle<viskores::UInt8> Involved(
    const viskores::cont::ArrayHandle<viskores::Id3>& triangles,
    const viskores::cont::ArrayHandle<Vec3>& sides)
  {
    using Algorithm = viskores::cont::Algorithm;
    namespace td = thickening_detail;
    viskores::cont::Invoker invoke;
    const viskores::Id numTriangles = triangles.GetNumberOfValues();
    viskores::cont::ArrayHandle<Vec3> lower, upper;
    viskores::cont::ArrayHandle<viskores::UInt8> degenerate, moved;
    invoke(detail::RealizeTriangle{}, triangles, sides, lower, upper, degenerate);
    viskores::cont::ArrayHandle<viskores::Id> queries;
    if (this->Points.GetNumberOfValues() == 0)
    {
      // The first round: every triangle, and every pair found from its first triangle.
      moved.AllocateAndFill(numTriangles, 1);
      Algorithm::Copy(viskores::cont::ArrayHandleIndex(numTriangles), queries);
    }
    else
    {
      viskores::cont::ArrayHandle<viskores::UInt8> movedPoints, kept;
      invoke(td::PointMoved{}, sides, this->Points, movedPoints);
      invoke(td::TriangleMoved{}, triangles, movedPoints, moved);
      Algorithm::CopyIf(viskores::cont::ArrayHandleIndex(numTriangles), moved, queries);
      invoke(td::PairKept{}, this->Pairs, moved, kept);
      viskores::cont::ArrayHandle<viskores::Id2> pairs;
      Algorithm::CopyIf(this->Pairs, kept, pairs);
      this->Pairs = pairs;
    }
    viskores::cont::ArrayHandle<viskores::Id2> found;
    if (queries.GetNumberOfValues() > 0)
    {
      const viskores::worklet::spatialstructure::BoundingVolumeHierarchy hierarchy(lower, upper);
      viskores::cont::ArrayHandle<viskores::Id> counts, offsets;
      invoke(td::ImproperPairsOf<false>{},
             queries,
             viskores::cont::make_ArrayHandleConstant(viskores::Id(0), queries.GetNumberOfValues()),
             triangles,
             lower,
             upper,
             degenerate,
             moved,
             sides,
             hierarchy,
             counts,
             found);
      // Improper pairs are rare: only the queries that found one look again.
      viskores::cont::ArrayHandle<viskores::Id> finders, finderCounts;
      Algorithm::CopyIf(queries, counts, finders);
      Algorithm::CopyIf(counts, counts, finderCounts);
      found.Allocate(Algorithm::ScanExclusive(finderCounts, offsets));
      invoke(td::ImproperPairsOf<true>{},
             finders,
             offsets,
             triangles,
             lower,
             upper,
             degenerate,
             moved,
             sides,
             hierarchy,
             counts,
             found);
    }
    viskores::cont::ArrayHandle<viskores::Id2> all;
    Algorithm::Copy(viskores::cont::make_ArrayHandleConcatenate(this->Pairs, found), all);
    this->Pairs = all;
    this->Points = sides;
    viskores::cont::ArrayHandle<viskores::UInt8> involved;
    Algorithm::Copy(degenerate, involved);
    invoke(td::MarkFirst{}, this->Pairs, involved);
    return involved;
  }

private:
  viskores::cont::ArrayHandle<viskores::Id2> Pairs;
  viskores::cont::ArrayHandle<Vec3> Points;
};

/// A certified thickening of the closed surface `mesh`, as `build` of proofs/lib/thick.py: the
/// offsets start at `delta0` and are halved at the vertices of every failure of (T1), (T2) and,
/// with `capped`, of `d_v <= epsilon`, up to 40 times; then (T3) and the probes (T4). Both
/// offsets of a failing vertex are halved, or with `asymmetric` only the offset on the side of
/// the failure: the layer of a failing tetrahedron, the side of a failing side triangle, the
/// side beyond the cap. The offsets are along `directions`. `Valid` is false if none is found.
inline Thickening BuildThickening(const SurfaceMesh& mesh,
                                  Real delta0,
                                  Real epsilon,
                                  bool capped,
                                  bool asymmetric = false,
                                  ThickeningDirections directions = ThickeningDirections::Hybrid)
{
  if (mesh.Unoriented)
  {
    throw viskores::cont::ErrorBadValue(
      "QuadricClusteringDecimation: the thickening needs a consistently oriented input; no "
      "result covers a piece that is not orientable.");
  }
  using Algorithm = viskores::cont::Algorithm;
  namespace td = thickening_detail;
  viskores::cont::Invoker invoke;
  Thickening result;
  const viskores::Id n = mesh.NumberOfPoints, m = mesh.NumberOfTriangles;
  viskores::cont::ArrayHandle<Vec3> crosses, faceNormals, normals;
  invoke(td::FaceNormal{}, mesh.Triangles, mesh.Points, crosses, faceNormals);
  VertexDirections(mesh, directions, normals);
  viskores::cont::ArrayHandle<Real> deltaPlus, deltaMinus;
  deltaPlus.AllocateAndFill(n, delta0);
  deltaMinus.AllocateAndFill(n, delta0);
  viskores::cont::ArrayHandle<Vec3> plus, minus;
  viskores::cont::ArrayHandle<viskores::Id3> sideTriangles;
  invoke(
    td::SideTriangle(n), viskores::cont::ArrayHandleIndex(2 * m), mesh.Triangles, sideTriangles);
  bool certified = false;
  viskores::Id round = 0;
  SidePairs sidePairs;
  for (; round < 40; ++round)
  {
    invoke(td::StoredSides{}, mesh.Points, normals, deltaPlus, deltaMinus, plus, minus);
    viskores::cont::ArrayHandle<viskores::UInt8> badPlus, badMinus, prismsPlus, prismsMinus;
    badPlus.AllocateAndFill(n, 0);
    badMinus.AllocateAndFill(n, 0);
    // (T1), per layer.
    invoke(td::BadPrism{},
           mesh.Triangles,
           faceNormals,
           mesh.Points,
           plus,
           minus,
           prismsPlus,
           prismsMinus);
    invoke(td::MarkTriangleVertices(n), mesh.Triangles, prismsMinus, badMinus);
    invoke(td::MarkTriangleVertices(n), mesh.Triangles, prismsPlus, badPlus);
    // (T2): a failing side triangle on its own side, B^+ for the first m, B^- for the others.
    viskores::cont::ArrayHandle<Vec3> sides;
    sides.Allocate(2 * n);
    Algorithm::CopySubRange(plus, 0, n, sides, 0);
    Algorithm::CopySubRange(minus, 0, n, sides, n);
    const viskores::cont::ArrayHandle<viskores::UInt8> involved =
      sidePairs.Involved(sideTriangles, sides);
    invoke(td::MarkTriangleVertices(n),
           viskores::cont::make_ArrayHandleView(sideTriangles, 0, m),
           viskores::cont::make_ArrayHandleView(involved, 0, m),
           badPlus);
    invoke(td::MarkTriangleVertices(n),
           viskores::cont::make_ArrayHandleView(sideTriangles, m, m),
           viskores::cont::make_ArrayHandleView(involved, m, m),
           badMinus);
    // The cap.
    if (capped)
    {
      invoke(td::ExceedsCap(epsilon), mesh.Points, plus, minus, badPlus, badMinus);
    }
    if (Algorithm::Reduce(badPlus, viskores::Id(0)) == 0 &&
        Algorithm::Reduce(badMinus, viskores::Id(0)) == 0)
    {
      certified = true;
      result.Sides = sides;
      break;
    }
    if (!asymmetric)
    {
      invoke(td::UniteSides{}, badPlus, badMinus);
    }
    invoke(td::HalveOffset{}, badPlus, deltaPlus);
    invoke(td::HalveOffset{}, badMinus, deltaMinus);
  }
  if (!certified)
  {
    return result;
  }
  result.Halvings = round;
  result.SideTriangles = sideTriangles;
  viskores::cont::ArrayHandle<Real> delta;
  invoke(td::SmallerOffset{}, deltaPlus, deltaMinus, delta);
  {
    viskores::cont::ArrayHandle<Vec3> lower, upper;
    invoke(TriangleBox{}, sideTriangles, result.Sides, lower, upper);
    result.SideHierarchy.Build(lower, upper);
  }
  result.Span = LargestSide(mesh.Points);
  result.MaxAbsCoordinate = MaxAbsCoordinate(result.Sides);
  const ThickeningObject sidesObject(result);

  // (T3): per component of M and side, at its first triangle.
  viskores::cont::ArrayHandle<viskores::Id> firstTriangles;
  {
    viskores::cont::ArrayHandle<viskores::Id2> candidates;
    invoke(td::FirstTriangleCandidate{},
           viskores::cont::ArrayHandleIndex(m),
           viskores::cont::make_ArrayHandleConstant<viskores::UInt8>(1, m),
           mesh.Triangles,
           mesh.Components,
           candidates);
    viskores::cont::ArrayHandle<viskores::Id> keys, values, uniqueKeys;
    invoke(SplitPair{}, candidates, keys, values);
    // The reference groups the triangles by component, whatever their order; ReduceByKey
    // reduces runs of equal keys, so the keys are sorted first.
    Algorithm::SortByKey(keys, values);
    Algorithm::ReduceByKey(keys, values, uniqueKeys, firstTriangles, viskores::Minimum());
  }
  const viskores::Id numComponents = firstTriangles.GetNumberOfValues();
  const viskores::Id numTries = 2 * numComponents;
  viskores::cont::ArrayHandle<viskores::Id3> tryTriangles;
  invoke(td::DegreeTriangle{},
         viskores::cont::ArrayHandleIndex(numTries),
         firstTriangles,
         sideTriangles,
         tryTriangles);
  if (!td::DegreeTestPasses(
        tryTriangles, result.Sides, sidesObject, mesh.Triangles, mesh.Points, plus, minus, delta))
  {
    return result;
  }

  // (T4): the first triangle of every component whose probe meets |M| once, in its interior.
  {
    viskores::cont::ArrayHandle<Vec3> lower, upper;
    invoke(TriangleBox{}, mesh.Triangles, mesh.Points, lower, upper);
    const viskores::worklet::spatialstructure::BoundingVolumeHierarchy hierarchy(lower, upper);
    viskores::cont::ArrayHandle<viskores::UInt8> ok;
    invoke(td::ProbeCandidate{},
           viskores::cont::ArrayHandleIndex(m),
           mesh.Triangles,
           mesh.Points,
           plus,
           minus,
           hierarchy,
           ok);
    viskores::cont::ArrayHandle<viskores::Id2> candidates, valid;
    invoke(td::FirstTriangleCandidate{},
           viskores::cont::ArrayHandleIndex(m),
           ok,
           mesh.Triangles,
           mesh.Components,
           candidates);
    Algorithm::CopyIf(candidates, candidates, valid, IsNonNegative{});
    viskores::cont::ArrayHandle<viskores::Id> keys, values, uniqueKeys;
    invoke(SplitPair{}, valid, keys, values);
    // Sorted as for (T3).
    Algorithm::SortByKey(keys, values);
    Algorithm::ReduceByKey(keys, values, uniqueKeys, result.ProbeTriangles, viskores::Minimum());
    if (result.ProbeTriangles.GetNumberOfValues() != numComponents)
    {
      return result;
    }
    invoke(
      td::ProbeCorners{}, result.ProbeTriangles, mesh.Triangles, plus, minus, result.ProbePoints);
  }
  result.Valid = true;
  return result;
}

/// What (F7) reads of a labelling besides its 3-label triangles: the component of the output
/// complex of every label (its smallest label, or -1 for a label in no output triangle) and
/// whether its first label's generator is inside T; the second smallest label of every
/// component and whether its generator is inside T, for a merge that removes the first; and per
/// probe, its crossings with the output triangles and how many are not generic.
struct ThickeningRound
{
  viskores::cont::ArrayHandle<viskores::Id> Root;
  viskores::cont::ArrayHandle<viskores::Id> Second;
  viskores::cont::ArrayHandle<viskores::UInt8> InsideFirst;
  viskores::cont::ArrayHandle<viskores::UInt8> InsideSecond;
  viskores::cont::ArrayHandle<viskores::Id> ProbeCrossings;
  viskores::cont::ArrayHandle<viskores::Id> ProbeNongeneric;
};

namespace thickening_detail
{

/// The two edges of an output triangle that join its labels into components, as
/// `components_of` of proofs/lib/thick.py.
struct ComponentEdges : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn triangle, FieldOut edges);
  using ExecutionSignature = void(_1, _2);

  template <typename EdgeVec>
  VISKORES_EXEC void operator()(const viskores::Id3& t, EdgeVec& edges) const
  {
    edges[0] = viskores::Id2(t[0], t[1]);
    edges[1] = viskores::Id2(t[1], t[2]);
  }
};

/// The component of a label in an output triangle, as `(root, label)`; `(-1, -1)` otherwise.
struct LabelRecord : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn label, FieldIn root, FieldIn appears, FieldOut record);
  using ExecutionSignature = _4(_1, _2, _3);

  VISKORES_EXEC viskores::Id2 operator()(viskores::Id label,
                                         viskores::Id root,
                                         viskores::UInt8 appears) const
  {
    return appears ? viskores::Id2(root, label) : viskores::Id2(-1);
  }
};

/// The root of a label in an output triangle, or -1.
struct RootOrNone : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn root, FieldIn appears, FieldOut result);
  using ExecutionSignature = _3(_1, _2);

  VISKORES_EXEC viskores::Id operator()(viskores::Id root, viskores::UInt8 appears) const
  {
    return appears ? root : viskores::Id(-1);
  }
};

/// For every component, sorted records `(root, label)`: the second smallest label, written at
/// the root, or -1.
struct SecondOfComponent : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn index, WholeArrayIn records, WholeArrayOut second);
  using ExecutionSignature = void(_1, _2, _3);

  template <typename RecordPortal, typename IdPortal>
  VISKORES_EXEC void operator()(viskores::Id i, const RecordPortal& records, IdPortal& second) const
  {
    const viskores::Id2 r = records.Get(i);
    if (r[0] >= 0 && i > 0 && records.Get(i - 1)[0] == r[0])
    {
      if (i == 1 || records.Get(i - 2)[0] != r[0])
      {
        second.Set(r[0], r[1]);
      }
    }
  }
};

/// The label itself if it is the root of its component, or -1.
struct ComponentRoot : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn label, FieldIn root, FieldOut result);
  using ExecutionSignature = _3(_1, _2);

  VISKORES_EXEC viskores::Id operator()(viskores::Id label, viskores::Id root) const
  {
    return (root == label) ? label : viskores::Id(-1);
  }
};

/// Whether the generator of a label is inside T, written at the root of its component.
struct InsideAt : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn root,
                                FieldIn label,
                                WholeArrayIn generators,
                                ExecObject thickening,
                                WholeArrayOut inside);
  using ExecutionSignature = void(_1, _2, _3, _4, _5);

  template <typename VecPortal, typename Thick, typename FlagPortal>
  VISKORES_EXEC void operator()(viskores::Id root,
                                viskores::Id label,
                                const VecPortal& generators,
                                const Thick& thickening,
                                FlagPortal& inside) const
  {
    if (label < 0)
    {
      return;
    }
    bool complete = true, fits = true;
    inside.Set(root, thickening.Inside(generators.Get(label), complete, fits) ? 1 : 0);
    if (!complete || !fits)
    {
      this->RaiseError("QuadricClusteringDecimation: exact evaluation out of range.");
    }
  }
};

/// Per item `probe * numTriangles + triangle`: whether the probe crosses the output triangle, 1 if
/// generically, 2 if not.
struct ProbeCrossing : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn item,
                                WholeArrayIn triangles,
                                WholeArrayIn generators,
                                ExecObject thickening,
                                FieldOut crossing,
                                FieldOut nongeneric);
  using ExecutionSignature = void(_1, _2, _3, _4, _5, _6);

  template <typename Id3Portal, typename VecPortal, typename Thick>
  VISKORES_EXEC void operator()(viskores::Id item,
                                const Id3Portal& triangles,
                                const VecPortal& generators,
                                const Thick& thickening,
                                viskores::Id& crossing,
                                viskores::Id& nongeneric) const
  {
    const viskores::Id numTriangles = triangles.GetNumberOfValues();
    const viskores::Id3 t = triangles.Get(item % numTriangles);
    const Vec3 p[3] = { generators.Get(t[0]), generators.Get(t[1]), generators.Get(t[2]) };
    bool fits = true;
    const SegmentHit hit = thickening.ProbeHit(item / numTriangles, p, fits);
    if (!fits)
    {
      this->RaiseError("QuadricClusteringDecimation: exact evaluation out of range.");
    }
    crossing = hit.Hit ? 1 : 0;
    nongeneric = (hit.Hit && !hit.Generic) ? 1 : 0;
  }
};

} // namespace thickening_detail

/// The data (F7) reads of the labelling `labels` with generators `generators` and output
/// triangles `triangles`, as `ThickVQC.iso_pass` of proofs/lib/thick.py computes it.
inline ThickeningRound PrepareThickeningRound(
  const Thickening& thickening,
  const viskores::cont::ArrayHandle<Vec3>& generators,
  const viskores::cont::ArrayHandle<viskores::Id3>& triangles,
  viskores::Id numClusters)
{
  using Algorithm = viskores::cont::Algorithm;
  namespace td = thickening_detail;
  viskores::cont::Invoker invoke;
  ThickeningRound round;
  const ThickeningObject thick(thickening);
  viskores::cont::ArrayHandle<viskores::Id2> edges;
  invoke(td::ComponentEdges{}, triangles, viskores::cont::make_ArrayHandleGroupVec<2>(edges));
  const viskores::cont::ArrayHandle<viskores::Id> roots = SmallestRoots(numClusters, edges);
  viskores::cont::ArrayHandle<viskores::UInt8> appears;
  appears.AllocateAndFill(numClusters, 0);
  invoke(
    td::MarkTriangleVertices(numClusters),
    triangles,
    viskores::cont::make_ArrayHandleConstant<viskores::UInt8>(1, triangles.GetNumberOfValues()),
    appears);
  viskores::cont::ArrayHandle<viskores::Id2> records;
  invoke(td::LabelRecord{}, viskores::cont::ArrayHandleIndex(numClusters), roots, appears, records);
  // A label in no output triangle has no component.
  invoke(td::RootOrNone{}, roots, appears, round.Root);
  Algorithm::Sort(records);
  round.Second.AllocateAndFill(numClusters, -1);
  invoke(td::SecondOfComponent{},
         viskores::cont::ArrayHandleIndex(records.GetNumberOfValues()),
         records,
         round.Second);
  round.InsideFirst.AllocateAndFill(numClusters, 1);
  round.InsideSecond.AllocateAndFill(numClusters, 1);
  {
    viskores::cont::ArrayHandle<viskores::Id> rootLabels;
    invoke(
      td::ComponentRoot{}, viskores::cont::ArrayHandleIndex(numClusters), round.Root, rootLabels);
    invoke(td::InsideAt{},
           viskores::cont::ArrayHandleIndex(numClusters),
           rootLabels,
           generators,
           thick,
           round.InsideFirst);
    invoke(td::InsideAt{},
           viskores::cont::ArrayHandleIndex(numClusters),
           round.Second,
           generators,
           thick,
           round.InsideSecond);
  }
  const viskores::Id numProbes = thickening.ProbeTriangles.GetNumberOfValues();
  const viskores::Id numTriangles = triangles.GetNumberOfValues();
  round.ProbeCrossings.AllocateAndFill(numProbes, 0);
  round.ProbeNongeneric.AllocateAndFill(numProbes, 0);
  if (numTriangles > 0)
  {
    const viskores::Id numItems = numProbes * numTriangles;
    viskores::cont::ArrayHandle<viskores::Id> crossing, nongeneric, owners, keys, crossings,
      nongenerics;
    invoke(td::ProbeCrossing{},
           viskores::cont::ArrayHandleIndex(numItems),
           triangles,
           generators,
           thick,
           crossing,
           nongeneric);
    invoke(DivideBy(numTriangles), viskores::cont::ArrayHandleIndex(numItems), owners);
    Algorithm::ReduceByKey(owners, crossing, keys, crossings, viskores::Add());
    Algorithm::ReduceByKey(owners, nongeneric, keys, nongenerics, viskores::Add());
    round.ProbeCrossings = crossings;
    round.ProbeNongeneric = nongenerics;
  }
  return round;
}

namespace thickening_detail
{

/// (F7) on an input triangle, as `ThickVQC.iso_pass` (`ArcThickVQC.iso_pass` on a cut input,
/// whose output is read on the glued `K` through `glue`): 1 if it is a 3-label triangle that
/// fails, because its output triangle meets a side, its component has its first generator
/// outside T, or a failing probe meets it or has the label of a vertex of its triangle.
struct ThickeningFailed : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn triangle,
                                WholeArrayIn inputTriangles,
                                WholeArrayIn labels,
                                WholeArrayIn glue,
                                WholeArrayIn generators,
                                ExecObject round,
                                FieldOut failed);
  using ExecutionSignature = _7(_1, _2, _3, _4, _5, _6);

  template <typename Id3Portal, typename IdPortal, typename VecPortal, typename Round>
  VISKORES_EXEC viskores::UInt8 operator()(const viskores::Id3& triangle,
                                           const Id3Portal& inputTriangles,
                                           const IdPortal& labels,
                                           const IdPortal& glue,
                                           const VecPortal& generators,
                                           const Round& round) const
  {
    const viskores::Id l[3] = { labels.Get(triangle[0]),
                                labels.Get(triangle[1]),
                                labels.Get(triangle[2]) };
    if (!AllDistinct(l[0], l[1], l[2]))
    {
      return 0;
    }
    const Vec3 p[3] = { generators.Get(l[0]), generators.Get(l[1]), generators.Get(l[2]) };
    bool complete = true, fits = true;
    bool failed = round.Thick.MeetsSides(p, complete, fits);
    const viskores::Id root = round.Root.Get(glue.Get(l[0]));
    failed = failed || (root >= 0 && !round.InsideFirst.Get(root));
    const viskores::Id numProbes = round.ProbeCrossings.GetNumberOfValues();
    for (viskores::Id i = 0; i < numProbes && !failed; ++i)
    {
      if (round.ProbeNongeneric.Get(i) == 0 && round.ProbeCrossings.Get(i) % 2 == 1)
      {
        continue;
      }
      const viskores::Id3 f = inputTriangles.Get(round.Thick.ProbeTriangles.Get(i));
      for (int a = 0; a < 3 && !failed; ++a)
      {
        const viskores::Id pl = glue.Get(labels.Get(f[a]));
        failed = pl == glue.Get(l[0]) || pl == glue.Get(l[1]) || pl == glue.Get(l[2]);
      }
      failed = failed || round.Thick.ProbeHit(i, p, fits).Hit;
    }
    if (!complete || !fits)
    {
      this->RaiseError("QuadricClusteringDecimation: exact evaluation out of range.");
    }
    return failed ? 1 : 0;
  }
};

} // namespace thickening_detail

/// The glue of a labelling, or the identity on its `k` labels if `glue` is empty.
inline viskores::cont::ArrayHandle<viskores::Id> GlueOrIdentity(
  const viskores::cont::ArrayHandle<viskores::Id>& glue,
  viskores::Id k)
{
  if (glue.GetNumberOfValues() > 0)
  {
    return glue;
  }
  viskores::cont::ArrayHandle<viskores::Id> identity;
  viskores::cont::Algorithm::Copy(viskores::cont::ArrayHandleIndex(k), identity);
  return identity;
}

/// The data of a round of (F7) in the execution environment, with the thickening.
struct ThickeningRoundPortals
{
  using IdPortal = viskores::cont::ArrayHandle<viskores::Id>::ReadPortalType;
  using FlagPortal = viskores::cont::ArrayHandle<viskores::UInt8>::ReadPortalType;

  ThickeningPortals Thick;
  IdPortal Root;
  IdPortal Second;
  FlagPortal InsideFirst;
  FlagPortal InsideSecond;
  IdPortal ProbeCrossings;
  IdPortal ProbeNongeneric;
};

struct ThickeningRoundObject : viskores::cont::ExecutionObjectBase
{
  ThickeningRoundObject(const Thickening& thickening, const ThickeningRound& round)
    : Source(&thickening)
    , Round(&round)
  {
  }

  VISKORES_CONT ThickeningRoundPortals PrepareForExecution(viskores::cont::DeviceAdapterId device,
                                                           viskores::cont::Token& token) const
  {
    ThickeningRoundPortals portals;
    portals.Thick = ThickeningObject(*this->Source).PrepareForExecution(device, token);
    portals.Root = this->Round->Root.PrepareForInput(device, token);
    portals.Second = this->Round->Second.PrepareForInput(device, token);
    portals.InsideFirst = this->Round->InsideFirst.PrepareForInput(device, token);
    portals.InsideSecond = this->Round->InsideSecond.PrepareForInput(device, token);
    portals.ProbeCrossings = this->Round->ProbeCrossings.PrepareForInput(device, token);
    portals.ProbeNongeneric = this->Round->ProbeNongeneric.PrepareForInput(device, token);
    return portals;
  }

  const Thickening* Source;
  const ThickeningRound* Round;
};

/// The flag (F7) of a certified thickening: flags the clusters with at least two vertices of
/// every 3-label input triangle that fails (`iso_flags` with `ThickVQC.iso_pass` of
/// proofs/lib/thick.py). With `glue`, for an input cut along its creases, the output is read on
/// the glued `K` (`ArcThickVQC.iso_pass`); the thickening is that of the uncut input, whose
/// triangles the cut keeps in order. With `roundOut`, stores the round, whose probes the region
/// blame reads. Returns the number of triangles that fail.
inline viskores::Id ThickeningFlags(const SurfaceMesh& mesh,
                                    const Thickening& thickening,
                                    const viskores::cont::ArrayHandle<viskores::Id>& labels,
                                    const viskores::cont::ArrayHandle<Vec3>& generators,
                                    const viskores::cont::ArrayHandle<viskores::Id>& sizes,
                                    viskores::cont::ArrayHandle<viskores::UInt8>& flags,
                                    const viskores::cont::ArrayHandle<viskores::Id>& glue = {},
                                    ThickeningRound* roundOut = nullptr)
{
  viskores::cont::Invoker invoke;
  const viskores::Id k = sizes.GetNumberOfValues();
  viskores::cont::ArrayHandle<viskores::Id3> triangles = OutputTriangles(mesh, labels);
  if (glue.GetNumberOfValues() > 0)
  {
    triangles = GlueTriangles(triangles, glue);
  }
  ThickeningRound local;
  ThickeningRound& round = roundOut ? *roundOut : local;
  round = PrepareThickeningRound(thickening, generators, triangles, k);
  viskores::cont::ArrayHandle<viskores::UInt8> failed;
  invoke(thickening_detail::ThickeningFailed{},
         mesh.Triangles,
         mesh.Triangles,
         labels,
         GlueOrIdentity(glue, k),
         generators,
         ThickeningRoundObject(thickening, round),
         failed);
  return FlagClustersOfTriangles(mesh, labels, failed, sizes, flags);
}

}
}
} // namespace viskores::worklet::quadric_clustering

VISKORES_QUADRIC_CLUSTERING_STRICT_FP_END

#endif // viskores_worklet_quadric_clustering_ThickeningFlags_h
