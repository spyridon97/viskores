//============================================================================
//  The contents of this file are covered by the Viskores license. See
//  LICENSE.txt for details.
//
//  By contributing to this file, all contributors agree to the Developer
//  Certificate of Origin Version 1.1 (DCO 1.1) as stated in DCO.txt.
//============================================================================
#ifndef viskores_worklet_quadric_clustering_IsotopyFlags_h
#define viskores_worklet_quadric_clustering_IsotopyFlags_h

#include <viskores/filter/geometry_refinement/worklet/quadric_clustering/EnvelopeFlags.h>
#include <viskores/filter/geometry_refinement/worklet/quadric_clustering/TopologyFlags.h>
#include <viskores/filter/geometry_refinement/worklet/quadric_clustering/VertexQuadrics.h>

#include <viskores/cont/Algorithm.h>
#include <viskores/cont/ArrayHandleGroupVec.h>
#include <viskores/cont/ArrayHandleIndex.h>
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

/// The input surface as the isotopy flag (F6) reads it: its triangles with their unit normals,
/// the corners of every vertex, hierarchies over its vertices and over the centroids of its
/// triangles, and what the reach `R` determines.
struct IsotopyMesh
{
  viskores::cont::ArrayHandle<Vec3> Points;
  viskores::cont::ArrayHandle<viskores::Id3> Triangles;
  viskores::cont::ArrayHandle<Vec3> Normals;
  viskores::cont::ArrayHandle<viskores::Id> CornerOffsets;
  viskores::cont::ArrayHandle<viskores::Id> Corners;
  viskores::cont::ArrayHandle<Vec3> Centroids;
  viskores::worklet::spatialstructure::BoundingVolumeHierarchy VertexHierarchy;
  viskores::worklet::spatialstructure::BoundingVolumeHierarchy CentroidHierarchy;
  Real Reach = 0;
  /// The longest edge `h`, and `sin omega` and `cos omega` of Theorem "detect".
  Real H = 0;
  Real SinOmega = 0;
  Real CosOmega = 1;
  /// The largest absolute coordinate of the points.
  Real MaxAbsCoordinate = 0;
};

namespace detail
{

/// The point divided by `d`, coordinate by coordinate.
VISKORES_EXEC_CONT inline Vec3 Divide(const Vec3& a, Real d)
{
  return Vec3(a[0] / d, a[1] / d, a[2] / d);
}

/// The centroid `((a + b) + c) / 3` of a triangle, in this order.
VISKORES_EXEC_CONT inline Vec3 Centroid(const Vec3& a, const Vec3& b, const Vec3& c)
{
  return Divide(Add(Add(a, b), c), 3);
}

struct TriangleCentroid : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn triangle, WholeArrayIn points, FieldOut centroid);
  using ExecutionSignature = _3(_1, _2);

  template <typename PointPortal>
  VISKORES_EXEC Vec3 operator()(const viskores::Id3& triangle, const PointPortal& points) const
  {
    return Centroid(points.Get(triangle[0]), points.Get(triangle[1]), points.Get(triangle[2]));
  }
};

} // namespace detail

/// The `IsotopyMesh` of a closed surface with triangle normals `normals` and a lower bound
/// `reach` on the reach of the surface it samples.
inline IsotopyMesh BuildIsotopyMesh(const SurfaceMesh& mesh,
                                    const viskores::cont::ArrayHandle<Vec3>& normals,
                                    Real reach)
{
  viskores::cont::Invoker invoke;
  IsotopyMesh result;
  result.Points = mesh.Points;
  result.Triangles = mesh.Triangles;
  result.Normals = normals;
  result.CornerOffsets = mesh.VertexCornerOffsets;
  result.Corners = mesh.VertexCorners;
  invoke(detail::TriangleCentroid{}, mesh.Triangles, mesh.Points, result.Centroids);
  result.VertexHierarchy.Build(mesh.Points, mesh.Points);
  result.CentroidHierarchy.Build(result.Centroids, result.Centroids);
  const ReachBound bound = ComputeReachBound(mesh, reach);
  result.Reach = reach;
  result.H = bound.H;
  result.SinOmega = bound.SinOmega;
  result.CosOmega = viskores::Sqrt(1 - bound.SinOmega * bound.SinOmega);
  result.MaxAbsCoordinate = MaxAbsCoordinate(mesh.Points);
  return result;
}

/// The isotopy test of Definition "isoflag", in the execution environment, as `iso_pass` of
/// proofs/lib/vqc.py: an output triangle passes if the angle of Lemma "graph" stays below a
/// right angle with the displacement of its corners, or otherwise if every piece of a cover of
/// it by subdivision passes with one of the input vertices nearest the piece. Every comparison
/// is made without trigonometric functions.
struct IsotopyPortals
{
  using VecPortal = viskores::cont::ArrayHandle<Vec3>::ReadPortalType;
  using IdPortal = viskores::cont::ArrayHandle<viskores::Id>::ReadPortalType;
  using Id3Portal = viskores::cont::ArrayHandle<viskores::Id3>::ReadPortalType;
  using Hierarchy = viskores::worklet::spatialstructure::BoundingVolumeHierarchy::ExecObjectType;

  /// The candidates of a piece: its nearest input vertices and triangles.
  static constexpr int NumberOfNearest = 8;
  /// The depth of the subdivision of a triangle that fails with its corners.
  static constexpr int MaxDepth = 4;

  VecPortal Points;
  Id3Portal Triangles;
  VecPortal Normals;
  IdPortal CornerOffsets;
  IdPortal Corners;
  VecPortal Centroids;
  Hierarchy VertexHierarchy;
  Hierarchy CentroidHierarchy;
  Real Reach;
  Real H;
  Real SinOmega;
  Real CosOmega;
  Real MaxAbsCoordinate;

  /// Whether the output triangle `p`, realised from the input triangle `t`, passes the test.
  /// `complete` is false if a query overflowed its stack.
  VISKORES_EXEC bool Passes(const Vec3* p, viskores::Id t, bool& complete) const
  {
    const viskores::Id3 input = this->Triangles.Get(t);
    const Vec3 ns = Cross(Sub(p[1], p[0]), Sub(p[2], p[0]));
    const Real nn = Norm(ns);
    if (!(nn > 0))
    {
      return false;
    }
    Real displacement = 0;
    for (int k = 0; k < 3; ++k)
    {
      displacement = viskores::Max(displacement, Norm(Sub(p[k], this->Points.Get(input[k]))));
    }
    const Real cosAngle = Dot(ns, this->Normals.Get(t)) / nn;
    const Real a =
      (this->H + 2 * displacement + this->H * this->H / this->Reach) / (2 * this->Reach);
    if (this->AngleBelowRightAt(cosAngle, a))
    {
      return true;
    }
    return this->PassesInPieces(p, detail::Divide(ns, nn), complete);
  }

  /// Whether the angle of cosine `cosAngle`, plus omega, plus `2 arcsin(a)`, is below a right
  /// angle: `a <= 1`, `cos(gamma) > 0` and `cosAngle > sin(gamma)` for
  /// `gamma = omega + 2 arcsin(a)`, whose sine and cosine are algebraic in `a` and `sin omega`.
  VISKORES_EXEC bool AngleBelowRightAt(Real cosAngle, Real a) const
  {
    const Real r = viskores::Sqrt(viskores::Max(1 - a * a, Real(0)));
    const Real sinGamma = this->SinOmega * (1 - 2 * a * a) + this->CosOmega * 2 * a * r;
    const Real cosGamma = this->CosOmega * (1 - 2 * a * a) - this->SinOmega * 2 * a * r;
    return a <= 1 && cosGamma > 0 && cosAngle > sinGamma;
  }

  /// The local test (`iso_local`): every piece of the subdivision of `p`, down to `MaxDepth`,
  /// must pass with one of its nearest input vertices `v`, with the largest cosine between the
  /// unit normal `ns` and the normal of a triangle at `v`, and with `2 D` bounded by the
  /// distance from the piece to `v` plus a bound on its distance to the surface.
  VISKORES_EXEC bool PassesInPieces(const Vec3* p, const Vec3& ns, bool& complete) const
  {
    Real s = this->MaxAbsCoordinate;
    for (int k = 0; k < 3; ++k)
    {
      for (int i = 0; i < 3; ++i)
      {
        s = viskores::Max(s, viskores::Abs(p[k][i]));
      }
    }
    s = s * (1 + 16 * UnitRoundoff);
    // Depth first; the pieces left to test, each with its depth.
    struct Piece
    {
      Vec3 V[3];
      int Depth;
    };
    Piece stack[3 * MaxDepth + 1];
    int top = 0;
    stack[top++] = Piece{ { p[0], p[1], p[2] }, 0 };
    while (top > 0)
    {
      const Piece piece = stack[--top];
      if (this->PiecePasses(piece.V, piece.Depth, ns, s, complete))
      {
        continue;
      }
      if (piece.Depth == MaxDepth)
      {
        return false;
      }
      const Vec3 a = piece.V[0], b = piece.V[1], c = piece.V[2];
      const Vec3 ab = detail::Divide(Add(a, b), 2);
      const Vec3 bc = detail::Divide(Add(b, c), 2);
      const Vec3 ca = detail::Divide(Add(c, a), 2);
      const int depth = piece.Depth + 1;
      stack[top++] = Piece{ { ab, bc, ca }, depth };
      stack[top++] = Piece{ { ca, bc, c }, depth };
      stack[top++] = Piece{ { ab, b, bc }, depth };
      stack[top++] = Piece{ { a, ab, ca }, depth };
    }
    return true;
  }

  VISKORES_EXEC bool PiecePasses(const Vec3* piece,
                                 int depth,
                                 const Vec3& ns,
                                 Real s,
                                 bool& complete) const
  {
    const Real margin =
      viskores::Sqrt(Real(3)) * Real(depth) * UnitRoundoff * s + 8 * UnitRoundoff * s;
    const Vec3 centroid = detail::Centroid(piece[0], piece[1], piece[2]);
    viskores::Id vertices[NumberOfNearest], triangles[NumberOfNearest];
    const int numVertices =
      this->Nearest(this->VertexHierarchy, this->Points, centroid, vertices, complete);
    const int numTriangles =
      this->Nearest(this->CentroidHierarchy, this->Centroids, centroid, triangles, complete);
    // |x - v| over the piece, for every candidate vertex: distance is convex.
    Real dv[NumberOfNearest];
    Real ds = std::numeric_limits<Real>::infinity();
    for (int c = 0; c < numVertices; ++c)
    {
      const Vec3 v = this->Points.Get(vertices[c]);
      Real largest = 0;
      for (int k = 0; k < 3; ++k)
      {
        largest = viskores::Max(largest, Norm(Sub(piece[k], v)));
      }
      dv[c] = largest * (1 + 8 * UnitRoundoff) + margin;
      ds = viskores::Min(ds, dv[c]);
    }
    // The distance to the surface, also through any input triangle (Theorem "faces").
    for (int c = 0; c < numTriangles; ++c)
    {
      const viskores::Id3 tri = this->Triangles.Get(triangles[c]);
      const Vec3 ta = this->Points.Get(tri[0]), tb = this->Points.Get(tri[1]),
                 tc = this->Points.Get(tri[2]);
      Real largest = 0;
      for (int k = 0; k < 3; ++k)
      {
        largest = viskores::Max(largest, PointTriangleDistanceUpper(piece[k], ta, tb, tc));
      }
      ds = viskores::Min(ds, largest + margin + this->H * this->H / this->Reach);
    }
    for (int c = 0; c < numVertices; ++c)
    {
      if (this->AngleBelowRight(this->BestCosine(vertices[c], ns), dv[c] + ds))
      {
        return true;
      }
    }
    return false;
  }

  /// The `NumberOfNearest` points of `points` nearest `q` (or all, if fewer): the smallest
  /// pairs `(|x - q|, index)`, so that ties go to the smaller index, whatever the hierarchy
  /// (`VQC.nearest`). Returns how many.
  VISKORES_EXEC int Nearest(const Hierarchy& hierarchy,
                            const VecPortal& points,
                            const Vec3& q,
                            viskores::Id* nearest,
                            bool& complete) const
  {
    Real distances[NumberOfNearest];
    int count = 0;
    const bool done = hierarchy.Traverse(
      [&](viskores::Id node)
      {
        return count < NumberOfNearest ||
          !(hierarchy.BoxDistanceLowerBound(q, node) > distances[NumberOfNearest - 1]);
      },
      [&](viskores::Id index)
      {
        const Real d = Norm(Sub(points.Get(index), q));
        if (count == NumberOfNearest &&
            !(d < distances[count - 1] ||
              (d == distances[count - 1] && index < nearest[count - 1])))
        {
          return true;
        }
        int slot = (count < NumberOfNearest) ? count++ : count - 1;
        while (slot > 0 &&
               (d < distances[slot - 1] || (d == distances[slot - 1] && index < nearest[slot - 1])))
        {
          distances[slot] = distances[slot - 1];
          nearest[slot] = nearest[slot - 1];
          --slot;
        }
        distances[slot] = d;
        nearest[slot] = index;
        return true;
      });
    complete = complete && done;
    return count;
  }

  /// The largest cosine between `ns` and the normal of a triangle at vertex `v`.
  VISKORES_EXEC Real BestCosine(viskores::Id v, const Vec3& ns) const
  {
    const viskores::Id numTriangles = this->Normals.GetNumberOfValues();
    Real best = -std::numeric_limits<Real>::infinity();
    for (viskores::Id c = this->CornerOffsets.Get(v); c < this->CornerOffsets.Get(v + 1); ++c)
    {
      best = viskores::Max(best, Dot(ns, this->Normals.Get(this->Corners.Get(c) % numTriangles)));
    }
    return best;
  }

  /// `AngleBelowRightAt` with `a = distance / 2R` (`iso_ok`).
  VISKORES_EXEC bool AngleBelowRight(Real cosAngle, Real distance) const
  {
    return this->AngleBelowRightAt(cosAngle, distance / (2 * this->Reach));
  }
};

struct IsotopyMeshObject : viskores::cont::ExecutionObjectBase
{
  explicit IsotopyMeshObject(const IsotopyMesh& mesh)
    : Mesh(mesh)
  {
  }

  VISKORES_CONT IsotopyPortals PrepareForExecution(viskores::cont::DeviceAdapterId device,
                                                   viskores::cont::Token& token) const
  {
    IsotopyPortals portals;
    portals.Points = this->Mesh.Points.PrepareForInput(device, token);
    portals.Triangles = this->Mesh.Triangles.PrepareForInput(device, token);
    portals.Normals = this->Mesh.Normals.PrepareForInput(device, token);
    portals.CornerOffsets = this->Mesh.CornerOffsets.PrepareForInput(device, token);
    portals.Corners = this->Mesh.Corners.PrepareForInput(device, token);
    portals.Centroids = this->Mesh.Centroids.PrepareForInput(device, token);
    portals.VertexHierarchy = this->Mesh.VertexHierarchy.PrepareForExecution(device, token);
    portals.CentroidHierarchy = this->Mesh.CentroidHierarchy.PrepareForExecution(device, token);
    portals.Reach = this->Mesh.Reach;
    portals.H = this->Mesh.H;
    portals.SinOmega = this->Mesh.SinOmega;
    portals.CosOmega = this->Mesh.CosOmega;
    portals.MaxAbsCoordinate = this->Mesh.MaxAbsCoordinate;
    return portals;
  }

  IsotopyMesh Mesh;
};

namespace detail
{

/// Whether a 3-label input triangle fails the isotopy test, realised at the generators. With
/// `UseChanged`, only the triangles at a changed cluster are tested; the others passed before.
struct IsotopyFailed : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn triangle,
                                WholeArrayIn labels,
                                WholeArrayIn generators,
                                WholeArrayIn changed,
                                ExecObject isotopy,
                                FieldOut failed);
  using ExecutionSignature = void(_1, _2, _3, _4, _5, _6, WorkIndex);

  bool UseChanged = false;

  template <typename IdPortal, typename VecPortal, typename FlagPortal>
  VISKORES_EXEC void operator()(const viskores::Id3& triangle,
                                const IdPortal& labels,
                                const VecPortal& generators,
                                const FlagPortal& changed,
                                const IsotopyPortals& isotopy,
                                viskores::UInt8& failed,
                                viskores::Id t) const
  {
    failed = 0;
    const viskores::Id a = labels.Get(triangle[0]), b = labels.Get(triangle[1]),
                       c = labels.Get(triangle[2]);
    if (a == b || b == c || a == c)
    {
      return;
    }
    if (this->UseChanged && !changed.Get(a) && !changed.Get(b) && !changed.Get(c))
    {
      return;
    }
    const Vec3 p[3] = { generators.Get(a), generators.Get(b), generators.Get(c) };
    bool complete = true;
    failed = isotopy.Passes(p, t, complete) ? 0 : 1;
    if (!complete)
    {
      this->RaiseError("QuadricClusteringDecimation: bounding volume hierarchy too deep.");
    }
  }
};
} // namespace detail

/// The isotopy flag (F6) of Definition "isoflag": flags the clusters with at least two vertices
/// of every 3-label input triangle whose output triangle fails the isotopy test (`iso_flags`).
/// With `changed`, the clusters merges made from a labelling that passed, only the triangles at
/// them are tested. Returns the number of triangles that fail.
inline viskores::Id IsotopyFlags(
  const SurfaceMesh& mesh,
  const IsotopyMesh& isotopy,
  const viskores::cont::ArrayHandle<viskores::Id>& labels,
  const viskores::cont::ArrayHandle<Vec3>& generators,
  const viskores::cont::ArrayHandle<viskores::Id>& sizes,
  viskores::cont::ArrayHandle<viskores::UInt8>& flags,
  const viskores::cont::ArrayHandle<viskores::UInt8>* changed = nullptr)
{
  using Algorithm = viskores::cont::Algorithm;
  viskores::cont::Invoker invoke;
  detail::IsotopyFailed failedTest;
  failedTest.UseChanged = changed != nullptr;
  viskores::cont::ArrayHandle<viskores::UInt8> none;
  viskores::cont::ArrayHandle<viskores::UInt8> failed;
  invoke(failedTest,
         mesh.Triangles,
         labels,
         generators,
         changed ? *changed : none,
         IsotopyMeshObject(isotopy),
         failed);
  viskores::cont::ArrayHandle<viskores::Id> failing;
  Algorithm::CopyIf(viskores::cont::ArrayHandleIndex(mesh.NumberOfTriangles), failed, failing);
  const viskores::Id numFailing = failing.GetNumberOfValues();
  if (numFailing > 0)
  {
    viskores::cont::ArrayHandle<viskores::Id3> triples;
    invoke(detail::TriangleLabels{}, failing, mesh.Triangles, labels, triples);
    viskores::cont::ArrayHandle<viskores::Id> listed;
    auto grouped = viskores::cont::make_ArrayHandleGroupVec<3>(listed);
    Algorithm::Copy(triples, grouped);
    FlagClusters(listed, sizes, flags);
  }
  return numFailing;
}

}
}
} // namespace viskores::worklet::quadric_clustering

VISKORES_QUADRIC_CLUSTERING_STRICT_FP_END

#endif // viskores_worklet_quadric_clustering_IsotopyFlags_h
