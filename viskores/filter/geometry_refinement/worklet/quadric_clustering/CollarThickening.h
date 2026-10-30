//============================================================================
//  The contents of this file are covered by the Viskores license. See
//  LICENSE.txt for details.
//
//  By contributing to this file, all contributors agree to the Developer
//  Certificate of Origin Version 1.1 (DCO 1.1) as stated in DCO.txt.
//============================================================================
#ifndef viskores_worklet_quadric_clustering_CollarThickening_h
#define viskores_worklet_quadric_clustering_CollarThickening_h

#include <viskores/filter/geometry_refinement/worklet/quadric_clustering/Clustering.h>
#include <viskores/filter/geometry_refinement/worklet/quadric_clustering/QuadricClusteringCommonWorklets.h>
#include <viskores/filter/geometry_refinement/worklet/quadric_clustering/SingularCurves.h>
#include <viskores/filter/geometry_refinement/worklet/quadric_clustering/ThickeningFlags.h>
#include <viskores/filter/geometry_refinement/worklet/quadric_clustering/TopologyFlags.h>

#include <viskores/cont/Algorithm.h>
#include <viskores/cont/ArrayHandleGroupVec.h>
#include <viskores/cont/ArrayHandleIndex.h>
#include <viskores/cont/ArrayHandleView.h>
#include <viskores/cont/ErrorBadValue.h>
#include <viskores/cont/ExecutionObjectBase.h>
#include <viskores/worklet/WorkletMapField.h>

#include <limits>

// Isotopy and a map along the fibres for inputs with boundary (Theorem "A homeomorphism along
// the fibres, with boundary"), as
// proofs/lib/collar.py: the thickening of M extended by a collar beyond every boundary loop,
// with the walls over the boundary of the collar added to the boundary surface; annuli with annulus
// angle coordinate theta around the loops; and the flag (F9) of annulus output, whose loop over
// every input loop must lie over the interior of its annulus and turn once around it. Every
// decision is exact: barycentric coordinates in the tetrahedra by the exact orientation in
// expansion arithmetic, and the coordinates of the annuli, dyadic or of small denominators,
// as integers.

VISKORES_QUADRIC_CLUSTERING_STRICT_FP_BEGIN

namespace viskores
{
namespace worklet
{
namespace quadric_clustering
{

/// The boundary loops of annulus input with boundary, their collar and their annuli, from
/// `BuildCollaredThickening`.
struct Collar
{
  viskores::Id NumberOfLoops = 0;
  /// The boundary vertices loop by loop, in the boundary orientation (M on the left) from the
  /// smallest vertex of every loop, the loops in the order of that vertex: position
  /// `LoopOffsets[j] + k` holds the k-th vertex of loop j.
  viskores::cont::ArrayHandle<viskores::Id> LoopVertices;
  /// The triangle of M that traverses the edge from every position to the next.
  viskores::cont::ArrayHandle<viskores::Id> LoopTriangles;
  /// The loop of every position.
  viskores::cont::ArrayHandle<viskores::Id> LoopOf;
  /// The first position of every loop, then the number of positions.
  viskores::cont::ArrayHandle<viskores::Id> LoopOffsets;
  /// The triangles of the collared extension Mh: those of M, then two per boundary edge.
  viskores::cont::ArrayHandle<viskores::Id3> Triangles;
  /// The triangles of the annuli, loop by loop as `annulus` of collar.py lists them: its loop,
  /// the barycentric coordinates of its corners with respect to the vertices of the triangle of
  /// Mh it lies in (row by row), theta at its corners as three numerators and three
  /// denominators, and rho at its corners.
  viskores::cont::ArrayHandle<viskores::Id> AnnulusLoops;
  viskores::cont::ArrayHandle<viskores::Vec<Real, 9>> AnnulusBarycentrics;
  viskores::cont::ArrayHandle<viskores::Vec<Real, 6>> AnnulusTheta;
  viskores::cont::ArrayHandle<Vec3> AnnulusRho;
  /// The triangles of the annuli over every triangle of Mh: offsets and indices.
  viskores::cont::ArrayHandle<viskores::Id> AnnulusOffsets;
  viskores::cont::ArrayHandle<viskores::Id> AnnulusOrder;
};

namespace collar_detail
{

/// At the tail `a` of every edge `a -> b` of a triangle that is in no other triangle: `b`, and
/// the triangle (`boundary_loops` and `edge_triangle` of collar.py).
struct BoundarySuccessor : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn triangle,
                                FieldIn index,
                                WholeArrayIn edges,
                                WholeArrayIn edgeCounts,
                                WholeArrayOut successors,
                                WholeArrayOut successorTriangles,
                                WholeArrayOut pinched);
  using ExecutionSignature = void(_1, _2, _3, _4, _5, _6, _7);

  template <typename EdgePortal,
            typename CountPortal,
            typename IdPortal,
            typename IdOutPortal,
            typename FlagPortal>
  VISKORES_EXEC void operator()(const viskores::Id3& t,
                                viskores::Id f,
                                const EdgePortal& edges,
                                const CountPortal& edgeCounts,
                                IdPortal& successors,
                                IdOutPortal& successorTriangles,
                                FlagPortal& pinched) const
  {
    for (int s = 0; s < 3; ++s)
    {
      const viskores::Id a = t[s], b = t[(s + 1) % 3];
      const viskores::Id e = FindSorted(edges, SortedEdge(a, b));
      if (edgeCounts.Get(e) == 1)
      {
        // A pinched boundary vertex has two boundary edges from it; the surface mesh has none,
        // but check.
        if (successors.Get(a) >= 0)
        {
          pinched.Set(0, 1);
        }
        successors.Set(a, b);
        successorTriangles.Set(a, f);
      }
    }
  }
};

/// The edge `v -> successor` of a boundary vertex, or `(-1, -1)`.
struct SuccessorEdge : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn index, FieldIn successor, FieldOut edge);
  using ExecutionSignature = _3(_1, _2);

  VISKORES_EXEC viskores::Id2 operator()(viskores::Id v, viskores::Id successor) const
  {
    return (successor >= 0) ? viskores::Id2(v, successor) : viskores::Id2(-1);
  }
};

struct Predecessor : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn index, FieldIn successor, WholeArrayOut predecessors);
  using ExecutionSignature = void(_1, _2, _3);

  template <typename IdPortal>
  VISKORES_EXEC void operator()(viskores::Id v,
                                viskores::Id successor,
                                IdPortal& predecessors) const
  {
    if (successor >= 0)
    {
      predecessors.Set(successor, v);
    }
  }
};

/// List ranking towards the smallest vertex of every loop: every other boundary vertex points
/// to its predecessor.
struct RankStart : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn index,
                                FieldIn successor,
                                FieldIn root,
                                FieldIn predecessor,
                                FieldOut next,
                                FieldOut distance);
  using ExecutionSignature = void(_1, _2, _3, _4, _5, _6);

  VISKORES_EXEC void operator()(viskores::Id v,
                                viskores::Id successor,
                                viskores::Id root,
                                viskores::Id predecessor,
                                viskores::Id& next,
                                viskores::Id& distance) const
  {
    const bool start = successor < 0 || root == v;
    next = start ? v : predecessor;
    distance = start ? 0 : 1;
  }
};

/// `(root, position, vertex)` of a boundary vertex, or invalid.
struct LoopRecord : viskores::worklet::WorkletMapField
{
  using ControlSignature =
    void(FieldIn index, FieldIn successor, FieldIn root, FieldIn distance, FieldOut record);
  using ExecutionSignature = _5(_1, _2, _3, _4);

  VISKORES_EXEC viskores::Id3 operator()(viskores::Id v,
                                         viskores::Id successor,
                                         viskores::Id root,
                                         viskores::Id distance) const
  {
    return (successor >= 0) ? viskores::Id3(root, distance, v) : viskores::Id3(-1);
  }
};

struct IsLoopRoot : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn index, FieldIn successor, FieldIn root, FieldOut flag);
  using ExecutionSignature = _4(_1, _2, _3);

  VISKORES_EXEC viskores::UInt8 operator()(viskores::Id v,
                                           viskores::Id successor,
                                           viskores::Id root) const
  {
    return (successor >= 0 && root == v) ? 1 : 0;
  }
};

struct RecordLoop : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn record, WholeArrayIn roots, FieldOut loop, FieldOut vertex);
  using ExecutionSignature = void(_1, _2, _3, _4);

  template <typename IdPortal>
  VISKORES_EXEC void operator()(const viskores::Id3& record,
                                const IdPortal& roots,
                                viskores::Id& loop,
                                viskores::Id& vertex) const
  {
    loop = FindSorted(roots, record[0]);
    vertex = record[2];
  }
};

} // namespace collar_detail

/// The boundary loops of the surface `mesh`, as `boundary_loops` of collar.py. Throws on a
/// pinched boundary vertex.
inline void FindBoundaryLoops(const SurfaceMesh& mesh, Collar& collar)
{
  using Algorithm = viskores::cont::Algorithm;
  namespace cd = collar_detail;
  viskores::cont::Invoker invoke;
  const viskores::Id n = mesh.NumberOfPoints, m = mesh.NumberOfTriangles;
  viskores::cont::ArrayHandle<viskores::Id> successors, successorTriangles;
  successors.AllocateAndFill(n, -1);
  successorTriangles.AllocateAndFill(n, -1);
  viskores::cont::ArrayHandle<viskores::UInt8> pinched;
  pinched.AllocateAndFill(1, 0);
  invoke(cd::BoundarySuccessor{},
         mesh.Triangles,
         viskores::cont::ArrayHandleIndex(m),
         mesh.Edges,
         mesh.EdgeTriangleCounts,
         successors,
         successorTriangles,
         pinched);
  if (viskores::cont::ArrayGetValue(0, pinched) != 0)
  {
    throw viskores::cont::ErrorBadValue(
      "QuadricClusteringDecimation: the fibre flag needs a boundary without pinched vertices.");
  }
  viskores::cont::ArrayHandle<viskores::Id2> allEdges, edges;
  invoke(cd::SuccessorEdge{}, viskores::cont::ArrayHandleIndex(n), successors, allEdges);
  Algorithm::CopyIf(allEdges, allEdges, edges, IsNonNegative{});
  const viskores::cont::ArrayHandle<viskores::Id> roots = SmallestRoots(n, edges);
  viskores::cont::ArrayHandle<viskores::Id> predecessors, next, distance;
  predecessors.AllocateAndFill(n, -1);
  invoke(cd::Predecessor{}, viskores::cont::ArrayHandleIndex(n), successors, predecessors);
  invoke(cd::RankStart{},
         viskores::cont::ArrayHandleIndex(n),
         successors,
         roots,
         predecessors,
         next,
         distance);
  RankLists(next, distance);
  viskores::cont::ArrayHandle<viskores::Id3> allRecords, records;
  invoke(
    cd::LoopRecord{}, viskores::cont::ArrayHandleIndex(n), successors, roots, distance, allRecords);
  Algorithm::CopyIf(allRecords, allRecords, records, IsNonNegative{});
  Algorithm::Sort(records);
  viskores::cont::ArrayHandle<viskores::UInt8> isRoot;
  invoke(cd::IsLoopRoot{}, viskores::cont::ArrayHandleIndex(n), successors, roots, isRoot);
  viskores::cont::ArrayHandle<viskores::Id> loopRoots;
  Algorithm::CopyIf(viskores::cont::ArrayHandleIndex(n), isRoot, loopRoots);
  collar.NumberOfLoops = loopRoots.GetNumberOfValues();
  invoke(cd::RecordLoop{}, records, loopRoots, collar.LoopOf, collar.LoopVertices);
  Algorithm::LowerBounds(
    collar.LoopOf, viskores::cont::ArrayHandleIndex(collar.NumberOfLoops + 1), collar.LoopOffsets);
  invoke(GatherValue{}, collar.LoopVertices, successorTriangles, collar.LoopTriangles);
}

namespace collar_detail
{

/// The next and the previous position on the loop of position `q`.
template <typename IdPortal>
VISKORES_EXEC_CONT inline viskores::Id NextOnLoop(viskores::Id q,
                                                  const IdPortal& loopOf,
                                                  const IdPortal& offsets)
{
  const viskores::Id j = loopOf.Get(q);
  return (q + 1 < offsets.Get(j + 1)) ? q + 1 : offsets.Get(j);
}

/// The two collar triangles of the boundary edge from position `q`: `(v_{k+1}, v_k, o_{k+1})`
/// and `(v_k, o_k, o_{k+1})`.
struct CollarTriangles : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn index,
                                WholeArrayIn loopVertices,
                                WholeArrayIn loopOf,
                                WholeArrayIn offsets,
                                FieldOut triangles);
  using ExecutionSignature = void(_1, _2, _3, _4, _5);

  VISKORES_CONT explicit CollarTriangles(viskores::Id numPoints)
    : NumberOfPoints(numPoints)
  {
  }

  template <typename IdPortal, typename PairVec>
  VISKORES_EXEC void operator()(viskores::Id q,
                                const IdPortal& loopVertices,
                                const IdPortal& loopOf,
                                const IdPortal& offsets,
                                PairVec& triangles) const
  {
    const viskores::Id next = NextOnLoop(q, loopOf, offsets);
    const viskores::Id a = loopVertices.Get(q), b = loopVertices.Get(next);
    const viskores::Id oa = this->NumberOfPoints + q, ob = this->NumberOfPoints + next;
    triangles[0] = viskores::Id3(b, a, ob);
    triangles[1] = viskores::Id3(a, oa, ob);
  }

  viskores::Id NumberOfPoints;
};

/// The walls over the outer collar edge from position `q`, two per layer, split as the prisms
/// are: from the smaller index at the bottom to the larger at the top. The stored points are
/// `[B^+; B^-; Mh]`, so the layers are `[-1, 0]` from `B^-` to Mh and `[0, 1]` from Mh to `B^+`.
struct Walls : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn index,
                                WholeArrayIn loopOf,
                                WholeArrayIn offsets,
                                FieldOut walls);
  using ExecutionSignature = void(_1, _2, _3, _4);

  VISKORES_CONT Walls(viskores::Id numInputPoints, viskores::Id numPoints)
    : NumberOfInputPoints(numInputPoints)
    , NumberOfPoints(numPoints)
  {
  }

  template <typename IdPortal, typename WallVec>
  VISKORES_EXEC void operator()(viskores::Id q,
                                const IdPortal& loopOf,
                                const IdPortal& offsets,
                                WallVec& walls) const
  {
    const viskores::Id2 e = SortedEdge(this->NumberOfInputPoints + q,
                                       this->NumberOfInputPoints + NextOnLoop(q, loopOf, offsets));
    const viskores::Id i = e[0], j = e[1], n = this->NumberOfPoints;
    const viskores::Id layers[2][2] = { { n, 2 * n }, { 2 * n, 0 } };
    for (int l = 0; l < 2; ++l)
    {
      const viskores::Id lo = layers[l][0], hi = layers[l][1];
      walls[2 * l] = viskores::Id3(i + lo, j + lo, j + hi);
      walls[2 * l + 1] = viskores::Id3(i + lo, j + hi, i + hi);
    }
  }

  viskores::Id NumberOfInputPoints;
  viskores::Id NumberOfPoints;
};

/// The unit outward normal in the plane of its triangle of the boundary edge from position
/// `q`: `(b - a) x n_t`, normalised, as `collar_mesh`.
template <typename IdPortal, typename VecPortal>
VISKORES_EXEC inline Vec3 OutwardNormal(viskores::Id q,
                                        const IdPortal& loopVertices,
                                        const IdPortal& loopTriangles,
                                        const IdPortal& loopOf,
                                        const IdPortal& offsets,
                                        const VecPortal& points,
                                        const VecPortal& faceUnits)
{
  const viskores::Id a = loopVertices.Get(q);
  const viskores::Id b = loopVertices.Get(NextOnLoop(q, loopOf, offsets));
  const Vec3 u = Cross(Sub(points.Get(b), points.Get(a)), faceUnits.Get(loopTriangles.Get(q)));
  const Real norm = Norm(u);
  return Vec3(u[0] / norm, u[1] / norm, u[2] / norm);
}

template <typename IdPortal>
VISKORES_EXEC_CONT inline viskores::Id PreviousOnLoop(viskores::Id q,
                                                      const IdPortal& loopOf,
                                                      const IdPortal& offsets)
{
  const viskores::Id j = loopOf.Get(q);
  return (q > offsets.Get(j)) ? q - 1 : offsets.Get(j + 1) - 1;
}

/// The collar point of a boundary vertex: `o_k = v_k + width_k mu_k`, `mu_k` the normalised sum
/// of the outward normals of the two boundary edges at `v_k`, or that of the edge from it when
/// the sum vanishes.
struct CollarPoint : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn index,
                                FieldIn width,
                                WholeArrayIn loopVertices,
                                WholeArrayIn loopTriangles,
                                WholeArrayIn loopOf,
                                WholeArrayIn offsets,
                                WholeArrayIn points,
                                WholeArrayIn faceUnits,
                                FieldOut point);
  using ExecutionSignature = _9(_1, _2, _3, _4, _5, _6, _7, _8);

  template <typename IdPortal, typename VecPortal>
  VISKORES_EXEC Vec3 operator()(viskores::Id q,
                                Real width,
                                const IdPortal& loopVertices,
                                const IdPortal& loopTriangles,
                                const IdPortal& loopOf,
                                const IdPortal& offsets,
                                const VecPortal& points,
                                const VecPortal& faceUnits) const
  {
    const Vec3 previous = OutwardNormal(PreviousOnLoop(q, loopOf, offsets),
                                        loopVertices,
                                        loopTriangles,
                                        loopOf,
                                        offsets,
                                        points,
                                        faceUnits);
    const Vec3 here =
      OutwardNormal(q, loopVertices, loopTriangles, loopOf, offsets, points, faceUnits);
    Vec3 mu = Add(previous, here);
    const Real norm = Norm(mu);
    mu = (norm < Real(1e-12)) ? here : Vec3(mu[0] / norm, mu[1] / norm, mu[2] / norm);
    return Add(points.Get(loopVertices.Get(q)), Scale(width, mu));
  }
};

/// Whether the unit normal of a triangle is not finite: a degenerate triangle.
struct NotFinite : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn unit, FieldOut flag);
  using ExecutionSignature = _2(_1);

  VISKORES_EXEC viskores::UInt8 operator()(const Vec3& u) const
  {
    return (viskores::IsFinite(u[0]) && viskores::IsFinite(u[1]) && viskores::IsFinite(u[2])) ? 0
                                                                                              : 1;
  }
};

/// The component of the triangle of the boundary surface: that of its first vertex modulo `n`.
struct SideComponent : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn triangle, WholeArrayIn components, FieldOut component);
  using ExecutionSignature = _3(_1, _2);

  VISKORES_CONT explicit SideComponent(viskores::Id n)
    : N(n)
  {
  }

  template <typename IdPortal>
  VISKORES_EXEC viskores::Id operator()(const viskores::Id3& t, const IdPortal& components) const
  {
    return components.Get(t[0] % this->N);
  }

  viskores::Id N;
};

/// What the annulus construction reads of M and of the loops.
struct FanPortals
{
  using IdPortal = viskores::cont::ArrayHandle<viskores::Id>::ReadPortalType;
  IdPortal LoopVertices;
  IdPortal LoopOf;
  IdPortal LoopOffsets;
  viskores::cont::ArrayHandle<viskores::Id2>::ReadPortalType Edges;
  viskores::cont::ArrayHandle<viskores::Id2>::ReadPortalType EdgeTriangles;
  viskores::cont::ArrayHandle<viskores::Id3>::ReadPortalType Triangles;
  viskores::cont::ArrayHandle<viskores::Id3>::ReadPortalType CollaredTriangles;
};

struct FanObject : viskores::cont::ExecutionObjectBase
{
  VISKORES_CONT FanPortals PrepareForExecution(viskores::cont::DeviceAdapterId device,
                                               viskores::cont::Token& token) const
  {
    FanPortals portals;
    portals.LoopVertices = this->Loops->LoopVertices.PrepareForInput(device, token);
    portals.LoopOf = this->Loops->LoopOf.PrepareForInput(device, token);
    portals.LoopOffsets = this->Loops->LoopOffsets.PrepareForInput(device, token);
    portals.Edges = this->Mesh->Edges.PrepareForInput(device, token);
    portals.EdgeTriangles = this->Mesh->EdgeTriangles.PrepareForInput(device, token);
    portals.Triangles = this->Mesh->Triangles.PrepareForInput(device, token);
    portals.CollaredTriangles = this->Loops->Triangles.PrepareForInput(device, token);
    return portals;
  }

  const SurfaceMesh* Mesh;
  const Collar* Loops;
};

/// The triangle of M that traverses the directed edge `x -> y`, or -1.
template <typename EdgePortal, typename PairPortal, typename Id3Portal>
VISKORES_EXEC inline viskores::Id TraversingTriangle(viskores::Id x,
                                                     viskores::Id y,
                                                     const EdgePortal& edges,
                                                     const PairPortal& edgeTriangles,
                                                     const Id3Portal& triangles)
{
  const viskores::Id e = FindSorted(edges, SortedEdge(x, y));
  if (e < 0)
  {
    return -1;
  }
  const viskores::Id2 candidates = edgeTriangles.Get(e);
  for (int c = 0; c < 2; ++c)
  {
    const viskores::Id f = candidates[c];
    if (f < 0)
    {
      continue;
    }
    const viskores::Id3 t = triangles.Get(f);
    for (int s = 0; s < 3; ++s)
    {
      if (t[s] == x && t[(s + 1) % 3] == y)
      {
        return f;
      }
    }
  }
  return -1;
}

/// Walks the fan of triangles of M at the boundary vertex of position `q`, as `fan_path`: from
/// the triangle on `v_{k-1} v_k` to the one on `v_k v_{k+1}`, calling `visit(c, w, f)` for the
/// third vertex `w` of every triangle but the last, the c-th, with the triangle `f` reached
/// across `v w`; returns the last triangle, and the number of such vertices in `count`.
template <typename Portals, typename Visit>
VISKORES_EXEC inline viskores::Id WalkFan(viskores::Id q,
                                          const Portals& portals,
                                          viskores::Id& count,
                                          Visit&& visit)
{
  const viskores::Id v = portals.LoopVertices.Get(q);
  const viskores::Id a =
    portals.LoopVertices.Get(PreviousOnLoop(q, portals.LoopOf, portals.LoopOffsets));
  const viskores::Id b =
    portals.LoopVertices.Get(NextOnLoop(q, portals.LoopOf, portals.LoopOffsets));
  viskores::Id f =
    TraversingTriangle(a, v, portals.Edges, portals.EdgeTriangles, portals.Triangles);
  count = 0;
  // A fan has fewer triangles than the mesh.
  for (viskores::Id step = 0; step <= portals.Triangles.GetNumberOfValues(); ++step)
  {
    const viskores::Id3 t = portals.Triangles.Get(f);
    int i = 0;
    while (t[i] != v)
    {
      ++i;
    }
    const viskores::Id w = t[(i + 1) % 3];
    if (w == b)
    {
      return f;
    }
    f = TraversingTriangle(w, v, portals.Edges, portals.EdgeTriangles, portals.Triangles);
    visit(count, w, f);
    ++count;
  }
  return -1;
}

/// The number of m-points of a boundary vertex (at least one) and of its annulus triangles:
/// the fan, two in the bridge and two in the collar.
struct FanCount : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn index, ExecObject fans, FieldOut points, FieldOut count);
  using ExecutionSignature = void(_1, _2, _3, _4);

  template <typename Portals>
  VISKORES_EXEC void operator()(viskores::Id q,
                                const Portals& fans,
                                viskores::Id& points,
                                viskores::Id& count) const
  {
    viskores::Id numFanVertices = 0;
    if (WalkFan(q, fans, numFanVertices, [](viskores::Id, viskores::Id, viskores::Id) {}) < 0)
    {
      this->RaiseError("QuadricClusteringDecimation: a boundary fan is not a path.");
    }
    points = viskores::Max(numFanVertices, viskores::Id(1));
    count = points + 3;
  }
};

/// A point of annulus annulus: at most three input vertices with weights.
struct WeightedPoint
{
  viskores::Id Vertex[3];
  Real Weight[3];
  int Count;
};

/// The barycentric coordinates of a point with respect to the vertices of a triangle.
VISKORES_EXEC inline Vec3 RowOf(const WeightedPoint& p, const viskores::Id3& t)
{
  Vec3 row(0);
  for (int c = 0; c < 3; ++c)
  {
    for (int k = 0; k < p.Count; ++k)
    {
      if (p.Vertex[k] == t[c])
      {
        row[c] = p.Weight[k];
      }
    }
  }
  return row;
}

VISKORES_EXEC inline WeightedPoint VertexPoint(viskores::Id v)
{
  WeightedPoint p;
  p.Vertex[0] = v;
  p.Weight[0] = 1;
  p.Count = 1;
  return p;
}

/// The point `(1 - lambda) v + lambda w` of the non-boundary edge `v w`, lambda = 1/4.
VISKORES_EXEC inline WeightedPoint EdgePoint(viskores::Id v, viskores::Id w)
{
  WeightedPoint p;
  p.Vertex[0] = v;
  p.Weight[0] = Real(0.75);
  p.Vertex[1] = w;
  p.Weight[1] = Real(0.25);
  p.Count = 2;
  return p;
}

/// theta of the i-th of the `r` m-points of position `kx`: `kx - 1/2 + (i + 1) / (r + 1)`, as a
/// numerator and a denominator.
VISKORES_EXEC inline viskores::Vec<Real, 2> ThetaOf(viskores::Id kx, viskores::Id i, viskores::Id r)
{
  return viskores::Vec<Real, 2>(static_cast<Real>((2 * kx - 1) * (r + 1) + 2 * (i + 1)),
                                static_cast<Real>(2 * (r + 1)));
}

/// theta at the boundary vertex of position k.
VISKORES_EXEC inline viskores::Vec<Real, 2> ThetaOf(viskores::Id k)
{
  return viskores::Vec<Real, 2>(static_cast<Real>(k), 1);
}

/// The point of annulus ear or a one-triangle component: `1 - 2 lambda` at `v`, lambda at the others.
VISKORES_EXEC inline WeightedPoint EarPoint(viskores::Id v, const viskores::Id3& t)
{
  WeightedPoint p;
  for (int c = 0; c < 3; ++c)
  {
    p.Vertex[c] = t[c];
    p.Weight[c] = (t[c] == v) ? Real(0.5) : Real(0.25);
  }
  p.Count = 3;
  return p;
}

/// The triangles of the annulus at a boundary vertex, as `annulus` of collar.py.
struct FanAnnulus : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn index,
                                FieldIn offset,
                                WholeArrayIn points,
                                ExecObject fans,
                                WholeArrayOut triangles,
                                WholeArrayOut loops,
                                WholeArrayOut barycentrics,
                                WholeArrayOut theta,
                                WholeArrayOut rho);
  using ExecutionSignature = void(_1, _2, _3, _4, _5, _6, _7, _8, _9);

  VISKORES_CONT FanAnnulus(viskores::Id numInputPoints, viskores::Id numInputTriangles)
    : NumberOfInputPoints(numInputPoints)
    , NumberOfInputTriangles(numInputTriangles)
  {
  }

  template <typename IdPortal,
            typename Portals,
            typename IdOut,
            typename BaryOut,
            typename ThetaOut,
            typename RhoOut>
  VISKORES_EXEC void operator()(viskores::Id q,
                                viskores::Id offset,
                                const IdPortal& numPoints,
                                const Portals& fans,
                                IdOut& triangles,
                                IdOut& loops,
                                BaryOut& barycentrics,
                                ThetaOut& theta,
                                RhoOut& rho) const
  {
    const viskores::Id j = fans.LoopOf.Get(q);
    const viskores::Id k = q - fans.LoopOffsets.Get(j);
    const viskores::Id next = NextOnLoop(q, fans.LoopOf, fans.LoopOffsets);
    const viskores::Id afterNext = NextOnLoop(next, fans.LoopOf, fans.LoopOffsets);
    const viskores::Id v = fans.LoopVertices.Get(q), b = fans.LoopVertices.Get(next);
    const viskores::Id r = numPoints.Get(q), r1 = numPoints.Get(next);
    viskores::Id out = offset;
    auto add =
      [&](
        viskores::Id d, const WeightedPoint* p, const viskores::Vec<Real, 2>* th, const Vec3& rhos)
    {
      const viskores::Id3 tv = fans.CollaredTriangles.Get(d);
      viskores::Vec<Real, 9> rows;
      viskores::Vec<Real, 6> thetas;
      for (int c = 0; c < 3; ++c)
      {
        const Vec3 row = RowOf(p[c], tv);
        for (int x = 0; x < 3; ++x)
        {
          rows[3 * c + x] = row[x];
        }
        thetas[c] = th[c][0];
        thetas[3 + c] = th[c][1];
      }
      triangles.Set(out, d);
      loops.Set(out, j);
      barycentrics.Set(out, rows);
      theta.Set(out, thetas);
      rho.Set(out, rhos);
      ++out;
    };
    const Real half = Real(0.5);
    // The fan: the triangle reached across v w_{c-1} holds the m-points c - 1 and c.
    viskores::Id previousW = -1, previousF = -1, numFanVertices = 0;
    const viskores::Id d = WalkFan(
      q,
      fans,
      numFanVertices,
      [&](viskores::Id c, viskores::Id w, viskores::Id f)
      {
        if (c >= 1)
        {
          const WeightedPoint p[3] = { VertexPoint(v), EdgePoint(v, previousW), EdgePoint(v, w) };
          const viskores::Vec<Real, 2> th[3] = { ThetaOf(k),
                                                 ThetaOf(k, c - 1, r),
                                                 ThetaOf(k, c, r) };
          add(previousF, p, th, Vec3(half, 0, 0));
        }
        previousW = w;
        previousF = f;
      });
    // The bridge, in the triangle d on v b: the last m-point of v, and the first of the next
    // vertex, from the third vertex x of d.
    const viskores::Id3 dt = fans.Triangles.Get(d);
    const WeightedPoint lastPoint =
      (numFanVertices > 0) ? EdgePoint(v, previousW) : EarPoint(v, dt);
    viskores::Id x = -1;
    for (int c = 0; c < 3; ++c)
    {
      if (dt[c] != v && dt[c] != b)
      {
        x = dt[c];
      }
    }
    const WeightedPoint firstNext =
      (x == fans.LoopVertices.Get(afterNext)) ? EarPoint(b, dt) : EdgePoint(b, x);
    {
      const WeightedPoint p[3] = { VertexPoint(v), lastPoint, firstNext };
      const viskores::Vec<Real, 2> th[3] = { ThetaOf(k),
                                             ThetaOf(k, r - 1, r),
                                             ThetaOf(k + 1, 0, r1) };
      add(d, p, th, Vec3(half, 0, 0));
    }
    {
      const WeightedPoint p[3] = { VertexPoint(v), firstNext, VertexPoint(b) };
      const viskores::Vec<Real, 2> th[3] = { ThetaOf(k), ThetaOf(k + 1, 0, r1), ThetaOf(k + 1) };
      add(d, p, th, Vec3(half, 0, half));
    }
    // The collar.
    const viskores::Id ct = this->NumberOfInputTriangles + 2 * q;
    const viskores::Id oa = this->NumberOfInputPoints + q, ob = this->NumberOfInputPoints + next;
    {
      const WeightedPoint p[3] = { VertexPoint(v), VertexPoint(b), VertexPoint(ob) };
      const viskores::Vec<Real, 2> th[3] = { ThetaOf(k), ThetaOf(k + 1), ThetaOf(k + 1) };
      add(ct, p, th, Vec3(half, half, 1));
    }
    {
      const WeightedPoint p[3] = { VertexPoint(v), VertexPoint(ob), VertexPoint(oa) };
      const viskores::Vec<Real, 2> th[3] = { ThetaOf(k), ThetaOf(k + 1), ThetaOf(k) };
      add(ct + 1, p, th, Vec3(half, 1, 1));
    }
  }

  viskores::Id NumberOfInputPoints;
  viskores::Id NumberOfInputTriangles;
};

} // namespace collar_detail

/// The annuli of the loops, as `annulus` of collar.py with lambda = 1/4, and their triangles
/// listed by the triangle of Mh they lie in.
inline void BuildAnnuli(const SurfaceMesh& mesh, Collar& collar)
{
  using Algorithm = viskores::cont::Algorithm;
  namespace cd = collar_detail;
  viskores::cont::Invoker invoke;
  const viskores::Id numBoundary = collar.LoopVertices.GetNumberOfValues();
  cd::FanObject fans;
  fans.Mesh = &mesh;
  fans.Loops = &collar;
  viskores::cont::ArrayHandle<viskores::Id> numPoints, counts, offsets;
  invoke(cd::FanCount{}, viskores::cont::ArrayHandleIndex(numBoundary), fans, numPoints, counts);
  const viskores::Id total = Algorithm::ScanExclusive(counts, offsets);
  // The triangle of Mh every annulus triangle lies in, the key that lists them by it.
  viskores::cont::ArrayHandle<viskores::Id> keys;
  keys.Allocate(total);
  collar.AnnulusLoops.Allocate(total);
  collar.AnnulusBarycentrics.Allocate(total);
  collar.AnnulusTheta.Allocate(total);
  collar.AnnulusRho.Allocate(total);
  invoke(cd::FanAnnulus(mesh.NumberOfPoints, mesh.NumberOfTriangles),
         viskores::cont::ArrayHandleIndex(numBoundary),
         offsets,
         numPoints,
         fans,
         keys,
         collar.AnnulusLoops,
         collar.AnnulusBarycentrics,
         collar.AnnulusTheta,
         collar.AnnulusRho);
  Algorithm::Copy(viskores::cont::ArrayHandleIndex(total), collar.AnnulusOrder);
  Algorithm::SortByKey(keys, collar.AnnulusOrder);
  Algorithm::LowerBounds(keys,
                         viskores::cont::ArrayHandleIndex(collar.Triangles.GetNumberOfValues() + 1),
                         collar.AnnulusOffsets);
}

/// The collared thickening of Theorem "A homeomorphism along the fibres, with boundary", as
/// `build_collared` of proofs/lib/collar.py: offsets
/// and collar widths start at `delta0`; a degenerate collar triangle halves the widths of its
/// collar points and the round is tried again; otherwise the (symmetric) offset is halved at the
/// vertices of every failure of (T1), (T2b) over the sides and the walls and, with `capped`, of
/// `d_v <= epsilon`, and the width too at a collar point, up to 40 rounds; then (T3b) per
/// component of the boundary surface. No probes (T4). The thickening carries the fibres of (F8)
/// over Mh; `collar` receives the loops and the annuli. `Valid` is false if none is found.
inline Thickening BuildCollaredThickening(
  const SurfaceMesh& mesh,
  Real delta0,
  Real epsilon,
  bool capped,
  Collar& collar,
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
  namespace cd = collar_detail;
  viskores::cont::Invoker invoke;
  Thickening result;
  FindBoundaryLoops(mesh, collar);
  const viskores::Id n = mesh.NumberOfPoints, m = mesh.NumberOfTriangles;
  const viskores::Id numBoundary = collar.LoopVertices.GetNumberOfValues();
  const viskores::Id nh = n + numBoundary, mh = m + 2 * numBoundary;

  // The collar reads the unit normals of M; Mh has the triangles of M, then of the collar.
  viskores::cont::ArrayHandle<Vec3> crossesM, unitsM;
  invoke(td::FaceNormal{}, mesh.Triangles, mesh.Points, crossesM, unitsM);
  {
    viskores::cont::ArrayHandle<viskores::Id3> collarTriangles;
    invoke(cd::CollarTriangles(n),
           viskores::cont::ArrayHandleIndex(numBoundary),
           collar.LoopVertices,
           collar.LoopOf,
           collar.LoopOffsets,
           viskores::cont::make_ArrayHandleGroupVec<2>(collarTriangles));
    collar.Triangles.Allocate(mh);
    Algorithm::CopySubRange(mesh.Triangles, 0, m, collar.Triangles, 0);
    Algorithm::CopySubRange(collarTriangles, 0, 2 * numBoundary, collar.Triangles, m);
  }
  // The boundary surface: the sides of Mh, then the walls.
  viskores::cont::ArrayHandle<viskores::Id3> boundaryTriangles;
  {
    viskores::cont::ArrayHandle<viskores::Id3> sideTriangles, walls;
    invoke(td::SideTriangle(nh),
           viskores::cont::ArrayHandleIndex(2 * mh),
           collar.Triangles,
           sideTriangles);
    invoke(cd::Walls(n, nh),
           viskores::cont::ArrayHandleIndex(numBoundary),
           collar.LoopOf,
           collar.LoopOffsets,
           viskores::cont::make_ArrayHandleGroupVec<4>(walls));
    boundaryTriangles.Allocate(2 * mh + 4 * numBoundary);
    Algorithm::CopySubRange(sideTriangles, 0, 2 * mh, boundaryTriangles, 0);
    Algorithm::CopySubRange(walls, 0, 4 * numBoundary, boundaryTriangles, 2 * mh);
  }
  const viskores::Id numBoundaryTriangles = boundaryTriangles.GetNumberOfValues();

  viskores::cont::ArrayHandle<Real> widths, delta;
  widths.AllocateAndFill(numBoundary, delta0);
  delta.AllocateAndFill(nh, delta0);
  SurfaceMesh collared;
  bool built = false, certified = false;
  viskores::cont::ArrayHandle<Vec3> points, plus, minus, sides;
  // The directions, and the points they were computed at: a halving moves only collar points,
  // so most directions are kept.
  viskores::cont::ArrayHandle<Vec3> normals, normalPoints;
  SidePairs sidePairs;
  viskores::Id round = 0;
  for (; round < 40; ++round)
  {
    viskores::cont::ArrayHandle<Vec3> outer;
    invoke(cd::CollarPoint{},
           viskores::cont::ArrayHandleIndex(numBoundary),
           widths,
           collar.LoopVertices,
           collar.LoopTriangles,
           collar.LoopOf,
           collar.LoopOffsets,
           mesh.Points,
           unitsM,
           outer);
    points = viskores::cont::ArrayHandle<Vec3>{};
    points.Allocate(nh);
    Algorithm::CopySubRange(mesh.Points, 0, n, points, 0);
    Algorithm::CopySubRange(outer, 0, numBoundary, points, n);
    viskores::cont::ArrayHandle<Vec3> crosses, faceNormals;
    invoke(td::FaceNormal{}, collar.Triangles, points, crosses, faceNormals);
    // A degenerate triangle of Mh: halve the widths at its collar points, and try again.
    {
      viskores::cont::ArrayHandle<viskores::UInt8> degenerate, marks;
      invoke(cd::NotFinite{}, faceNormals, degenerate);
      if (Algorithm::Reduce(degenerate, viskores::Id(0)) > 0)
      {
        marks.AllocateAndFill(nh, 0);
        invoke(td::MarkTriangleVertices(nh), collar.Triangles, degenerate, marks);
        invoke(
          td::HalveOffset{}, viskores::cont::make_ArrayHandleView(marks, n, numBoundary), widths);
        continue;
      }
    }
    if (!built)
    {
      BuildSurfaceMesh(points, collar.Triangles, true, collared);
      built = true;
    }
    collared.Points = points;
    if (normalPoints.GetNumberOfValues() == 0)
    {
      VertexDirections(collared, directions, normals);
    }
    else
    {
      viskores::cont::ArrayHandle<viskores::UInt8> moved;
      invoke(td::PointMoved{}, points, normalPoints, moved);
      VertexDirections(collared, directions, normals, &moved);
    }
    normalPoints = points;
    invoke(td::StoredSides{}, points, normals, delta, delta, plus, minus);
    viskores::cont::ArrayHandle<viskores::UInt8> bad, prismsPlus, prismsMinus;
    bad.AllocateAndFill(nh, 0);
    // (T1), both layers.
    invoke(
      td::BadPrism{}, collar.Triangles, faceNormals, points, plus, minus, prismsPlus, prismsMinus);
    invoke(td::MarkTriangleVertices(nh), collar.Triangles, prismsMinus, bad);
    invoke(td::MarkTriangleVertices(nh), collar.Triangles, prismsPlus, bad);
    // (T2b): the sides and the walls, on the stored points [B^+; B^-; Mh].
    sides = viskores::cont::ArrayHandle<Vec3>{};
    sides.Allocate(3 * nh);
    Algorithm::CopySubRange(plus, 0, nh, sides, 0);
    Algorithm::CopySubRange(minus, 0, nh, sides, nh);
    Algorithm::CopySubRange(points, 0, nh, sides, 2 * nh);
    invoke(td::MarkTriangleVertices(nh),
           boundaryTriangles,
           sidePairs.Involved(boundaryTriangles, sides),
           bad);
    if (capped)
    {
      viskores::cont::ArrayHandle<viskores::UInt8> capPlus, capMinus;
      capPlus.AllocateAndFill(nh, 0);
      capMinus.AllocateAndFill(nh, 0);
      invoke(td::ExceedsCap(epsilon), points, plus, minus, capPlus, capMinus);
      invoke(td::UniteSides{}, capPlus, capMinus);
      OrFlags(bad, capPlus);
    }
    if (Algorithm::Reduce(bad, viskores::Id(0)) == 0)
    {
      certified = true;
      break;
    }
    invoke(td::HalveOffset{}, bad, delta);
    invoke(td::HalveOffset{}, viskores::cont::make_ArrayHandleView(bad, n, numBoundary), widths);
  }
  if (!certified)
  {
    return result;
  }
  result.Halvings = round;
  result.Sides = sides;
  result.SideTriangles = boundaryTriangles;
  {
    viskores::cont::ArrayHandle<Vec3> lower, upper;
    invoke(TriangleBox{}, boundaryTriangles, sides, lower, upper);
    result.SideHierarchy.Build(lower, upper);
  }
  result.Span = LargestSide(points);
  result.MaxAbsCoordinate = MaxAbsCoordinate(sides);
  // The largest third coordinate of the stored points.
  {
    const Vec3 infinity(std::numeric_limits<Real>::infinity());
    result.Top = Algorithm::Reduce(sides, Vec3(-infinity[0]), ComponentMax{})[2];
  }
  invoke(cd::SideComponent(nh), boundaryTriangles, collared.Components, result.SideComponents);
  result.VertexComponents = mesh.Components;
  const ThickeningObject sidesObject(result);

  // (T3b): per component of the boundary surface, at its first triangle.
  viskores::cont::ArrayHandle<viskores::Id3> tryTriangles;
  {
    viskores::cont::ArrayHandle<viskores::Id2> edges;
    invoke(
      td::ComponentEdges{}, boundaryTriangles, viskores::cont::make_ArrayHandleGroupVec<2>(edges));
    const viskores::cont::ArrayHandle<viskores::Id> roots = SmallestRoots(3 * nh, edges);
    viskores::cont::ArrayHandle<viskores::Id2> records;
    invoke(td::FirstTriangleCandidate{},
           viskores::cont::ArrayHandleIndex(numBoundaryTriangles),
           viskores::cont::make_ArrayHandleConstant<viskores::UInt8>(1, numBoundaryTriangles),
           boundaryTriangles,
           roots,
           records);
    viskores::cont::ArrayHandle<viskores::Id> keys, values, uniqueKeys, firstTriangles;
    invoke(SplitPair{}, records, keys, values);
    Algorithm::SortByKey(keys, values);
    Algorithm::ReduceByKey(keys, values, uniqueKeys, firstTriangles, viskores::Minimum());
    invoke(GatherValue{}, firstTriangles, boundaryTriangles, tryTriangles);
  }
  if (!td::DegreeTestPasses(
        tryTriangles, sides, sidesObject, collar.Triangles, points, plus, minus, delta))
  {
    return result;
  }
  // The fibres of (F8), over Mh, and the annuli.
  AddFibres(collared, result);
  BuildAnnuli(mesh, collar);
  result.Valid = true;
  return result;
}

/// What (F9) reads of a labelling: the loop `gamma_j` of K over every boundary loop, as
/// `output_loop` of collar.py (the labels of its vertices in order, cyclic repeats removed),
/// one entry per edge `gamma_j[e] -> gamma_j[e + 1]`, loop by loop; per entry, whether its
/// segment passes `edge_check`, whether the edge is traversed forward by annulus output triangle,
/// the increase of theta, and whether it is bad (fails, or its loop does not turn once); per
/// loop, the number of its failing edges and the increase of its passing ones; the entries by unordered
/// label pair and by label; and the directed edges of the output triangles with their third
/// label.
struct CollarRound
{
  viskores::cont::ArrayHandle<viskores::Id> EntryLoop;
  viskores::cont::ArrayHandle<viskores::Id> EntryLabel;
  viskores::cont::ArrayHandle<viskores::Id> EntryOffsets;
  viskores::cont::ArrayHandle<viskores::UInt8> EntryCheck;
  viskores::cont::ArrayHandle<viskores::UInt8> EntryBad;
  viskores::cont::ArrayHandle<Real> EntryIncrease;
  viskores::cont::ArrayHandle<viskores::Id> LoopFailures;
  viskores::cont::ArrayHandle<Real> LoopIncrease;
  viskores::cont::ArrayHandle<viskores::Id2> EdgeKeys;
  viskores::cont::ArrayHandle<viskores::Id> EdgeEntries;
  viskores::cont::ArrayHandle<viskores::Id> LabelEntryOffsets;
  viskores::cont::ArrayHandle<viskores::Id> LabelEntries;
  viskores::cont::ArrayHandle<viskores::Id3> Forward;
  bool AnyBad = false;
};

namespace collar_detail
{

/// 1 where the label differs from that of the previous position: cyclic repeats are dropped
/// (`output_loop`).
struct KeepLabel : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn index,
                                WholeArrayIn labels,
                                WholeArrayIn loopOf,
                                WholeArrayIn offsets,
                                FieldOut keep);
  using ExecutionSignature = _5(_1, _2, _3, _4);

  template <typename IdPortal>
  VISKORES_EXEC viskores::Id operator()(viskores::Id q,
                                        const IdPortal& labels,
                                        const IdPortal& loopOf,
                                        const IdPortal& offsets) const
  {
    return (labels.Get(q) != labels.Get(PreviousOnLoop(q, loopOf, offsets))) ? 1 : 0;
  }
};

/// A loop whose vertices all carry one label keeps its first.
struct KeepFirst : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn index,
                                FieldIn keep,
                                WholeArrayIn loopOf,
                                WholeArrayIn offsets,
                                WholeArrayIn keptCounts,
                                FieldOut kept);
  using ExecutionSignature = _6(_1, _2, _3, _4, _5);

  template <typename IdPortal>
  VISKORES_EXEC viskores::UInt8 operator()(viskores::Id q,
                                           viskores::Id keep,
                                           const IdPortal& loopOf,
                                           const IdPortal& offsets,
                                           const IdPortal& keptCounts) const
  {
    const viskores::Id j = loopOf.Get(q);
    return (keep != 0 || (q == offsets.Get(j) && keptCounts.Get(j) == 0)) ? 1 : 0;
  }
};

/// The edge of annulus entry, from its label to the label of the next entry of its loop, with its
/// generators.
struct EntryEdge : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn index,
                                WholeArrayIn entryLabel,
                                WholeArrayIn entryLoop,
                                WholeArrayIn entryOffsets,
                                WholeArrayIn generators,
                                FieldOut edge,
                                FieldOut start,
                                FieldOut end);
  using ExecutionSignature = void(_1, _2, _3, _4, _5, _6, _7, _8);

  template <typename IdPortal, typename VecPortal>
  VISKORES_EXEC void operator()(viskores::Id e,
                                const IdPortal& entryLabel,
                                const IdPortal& entryLoop,
                                const IdPortal& entryOffsets,
                                const VecPortal& generators,
                                viskores::Id2& edge,
                                Vec3& start,
                                Vec3& end) const
  {
    const viskores::Id a = entryLabel.Get(e);
    const viskores::Id b = entryLabel.Get(NextOnLoop(e, entryLoop, entryOffsets));
    edge = viskores::Id2(a, b);
    start = generators.Get(a);
    end = generators.Get(b);
  }
};

/// What the loop test reads of the collared thickening: the fibre tetrahedra and the annuli.
struct AnnulusPortals
{
  using IdPortal = viskores::cont::ArrayHandle<viskores::Id>::ReadPortalType;

  /// Calls `visit(tau, annulus)` for every tetrahedron `tau` whose box meets that of the
  /// segment `a b`, and every annulus triangle `annulus` of loop `loop` over the triangle of
  /// tau, as the candidates of `edge_pieces`; any superset of them gives the same pieces. Returns
  /// false if the hierarchy is too deep.
  template <typename Visit>
  VISKORES_EXEC bool ForEachCandidate(viskores::Id loop,
                                      const Vec3& a,
                                      const Vec3& b,
                                      Visit&& visit) const
  {
    Vec3 lo, hi;
    const Real pad = Real(1e-9) * viskores::Max(Real(1), viskores::Max(MaxAbs(a), MaxAbs(b)));
    for (int k = 0; k < 3; ++k)
    {
      lo[k] = viskores::Min(a[k], b[k]) - pad;
      hi[k] = viskores::Max(a[k], b[k]) + pad;
    }
    return this->Hierarchy.ForEachOverlap(
      lo,
      hi,
      [&](viskores::Id tau)
      {
        const viskores::Id d = tau / 6;
        for (viskores::Id x = this->AnnulusOffsets.Get(d); x < this->AnnulusOffsets.Get(d + 1); ++x)
        {
          const viskores::Id annulus = this->AnnulusOrder.Get(x);
          if (this->AnnulusLoops.Get(annulus) == loop)
          {
            visit(tau, annulus);
          }
        }
        return true;
      });
  }

  viskores::cont::ArrayHandle<viskores::Vec<Vec3, 4>>::ReadPortalType Tetrahedra;
  viskores::worklet::spatialstructure::BoundingVolumeHierarchy::ExecObjectType Hierarchy;
  viskores::cont::ArrayHandle<viskores::Id3>::ReadPortalType Triangles;
  IdPortal AnnulusLoops;
  IdPortal AnnulusOffsets;
  IdPortal AnnulusOrder;
  viskores::cont::ArrayHandle<viskores::Vec<Real, 9>>::ReadPortalType Barycentrics;
  viskores::cont::ArrayHandle<viskores::Vec<Real, 6>>::ReadPortalType Theta;
  viskores::cont::ArrayHandle<Vec3>::ReadPortalType Rho;
};

struct AnnulusObject : viskores::cont::ExecutionObjectBase
{
  VISKORES_CONT AnnulusPortals PrepareForExecution(viskores::cont::DeviceAdapterId device,
                                                   viskores::cont::Token& token) const
  {
    AnnulusPortals portals;
    portals.Tetrahedra = this->Thick->FibreTetrahedra.PrepareForInput(device, token);
    portals.Hierarchy = this->Thick->FibreHierarchy.PrepareForExecution(device, token);
    portals.Triangles = this->Loops->Triangles.PrepareForInput(device, token);
    portals.AnnulusLoops = this->Loops->AnnulusLoops.PrepareForInput(device, token);
    portals.AnnulusOffsets = this->Loops->AnnulusOffsets.PrepareForInput(device, token);
    portals.AnnulusOrder = this->Loops->AnnulusOrder.PrepareForInput(device, token);
    portals.Barycentrics = this->Loops->AnnulusBarycentrics.PrepareForInput(device, token);
    portals.Theta = this->Loops->AnnulusTheta.PrepareForInput(device, token);
    portals.Rho = this->Loops->AnnulusRho.PrepareForInput(device, token);
    return portals;
  }

  const Thickening* Thick;
  const Collar* Loops;
};

/// The number of candidate pieces of every segment.
struct CountPieces : viskores::worklet::WorkletMapField
{
  using ControlSignature =
    void(FieldIn loop, FieldIn start, FieldIn end, ExecObject loops, FieldOut count);
  using ExecutionSignature = _5(_1, _2, _3, _4);

  template <typename Loops>
  VISKORES_EXEC viskores::Id operator()(viskores::Id loop,
                                        const Vec3& start,
                                        const Vec3& end,
                                        const Loops& loops) const
  {
    viskores::Id count = 0;
    if (!loops.ForEachCandidate(loop, start, end, [&](viskores::Id, viskores::Id) { ++count; }))
    {
      this->RaiseError("QuadricClusteringDecimation: bounding volume hierarchy too deep.");
    }
    return count;
  }
};

/// The candidate pieces of every segment, `(tau, annulus triangle)`, with the segment.
struct FillPieces : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn index,
                                FieldIn loop,
                                FieldIn start,
                                FieldIn end,
                                FieldIn offset,
                                ExecObject loops,
                                WholeArrayOut items,
                                WholeArrayOut owners);
  using ExecutionSignature = void(_1, _2, _3, _4, _5, _6, _7, _8);

  template <typename Loops, typename ItemPortal, typename IdPortal>
  VISKORES_EXEC void operator()(viskores::Id edge,
                                viskores::Id loop,
                                const Vec3& start,
                                const Vec3& end,
                                viskores::Id offset,
                                const Loops& loops,
                                ItemPortal& items,
                                IdPortal& owners) const
  {
    viskores::Id out = offset;
    loops.ForEachCandidate(loop,
                           start,
                           end,
                           [&](viskores::Id tau, viskores::Id annulus)
                           {
                             items.Set(out, viskores::Id2(tau, annulus));
                             owners.Set(out, edge);
                             ++out;
                           });
  }
};

/// A piece of a segment over annulus annulus, stored as `[s0 numerator, s0 denominator, s1
/// numerator, s1 denominator]`, each annulus expansion of at most `StoredCapacity` components after
/// its length, then the slope of theta along the segment (approximate), whether the piece has
/// positive length, and whether it fails (theta not increasing, or rho not positive at annulus end).
constexpr int StoredCapacity = 15;
constexpr int StoredSize = StoredCapacity + 1;
using PieceRecord = viskores::Vec<Real, 4 * StoredSize + 3>;
constexpr int SlopeAt = 4 * StoredSize;
constexpr int ValidAt = 4 * StoredSize + 1;
constexpr int FailsAt = 4 * StoredSize + 2;

namespace big = thickening_detail::big;

VISKORES_EXEC inline big::Big Orient(const big::BigPoint& a,
                                     const big::BigPoint& b,
                                     const big::BigPoint& c,
                                     const big::BigPoint& d)
{
  return big::Dot(big::Cross(big::Sub(b, a), big::Sub(c, a)), big::Sub(d, a));
}

/// An exact rational `Num / Den` with `Den > 0`.
struct Ratio
{
  big::Big Num;
  big::Big Den;
};

VISKORES_EXEC inline Ratio RatioOf(const big::Big& num, const big::Big& den)
{
  Ratio r;
  if (big::Sign(den) < 0)
  {
    r.Num = big::Neg(num);
    r.Den = big::Neg(den);
  }
  else
  {
    r.Num = num;
    r.Den = den;
  }
  return r;
}

VISKORES_EXEC inline Ratio RatioOf(Real x)
{
  return RatioOf(big::BigOf(x), big::BigOf(1));
}

/// The sign of `x - y`.
VISKORES_EXEC inline int Compare(const Ratio& x, const Ratio& y, bool& fits)
{
  const big::Big d = big::Sub(big::Mul(x.Num, y.Den), big::Mul(y.Num, x.Den));
  fits = fits && !d.Overflow;
  return big::Sign(d);
}

/// `_affine_roots` of collar.py: clips `[lo, hi]` by `(c0 + t (c1 - c0)) / g >= 0`, `g` of sign
/// `sign`.
VISKORES_EXEC inline void Clip(const big::Big& c0,
                               const big::Big& c1,
                               int sign,
                               Ratio& lo,
                               Ratio& hi,
                               bool& empty,
                               bool& fits)
{
  const big::Big s = big::Sub(c1, c0);
  fits = fits && !s.Overflow;
  const int slope = big::Sign(s) * sign;
  if (slope == 0)
  {
    empty = empty || big::Sign(c0) * sign < 0;
    return;
  }
  const Ratio r = RatioOf(c0, big::Neg(s));
  if (slope > 0)
  {
    if (Compare(r, lo, fits) > 0)
    {
      lo = r;
    }
  }
  else if (Compare(r, hi, fits) < 0)
  {
    hi = r;
  }
}

/// An approximation of annulus expansion.
VISKORES_EXEC inline Real Approximate(const big::Big& b)
{
  Real s = 0;
  for (int i = 0; i < b.N; ++i)
  {
    s += b.C[i];
  }
  return s;
}

VISKORES_EXEC inline Real Approximate(const Ratio& r)
{
  return Approximate(r.Num) / Approximate(r.Den);
}

VISKORES_EXEC inline void Store(const big::Big& b, PieceRecord& record, int at, bool& fits)
{
  fits = fits && !b.Overflow && b.N <= StoredCapacity;
  const int n = viskores::Min(b.N, StoredCapacity);
  record[at] = static_cast<Real>(n);
  for (int i = 0; i < n; ++i)
  {
    record[at + 1 + i] = b.C[i];
  }
}

/// A candidate piece, as `edge_pieces` of collar.py: the part of the segment in the
/// tetrahedron, by its barycentric coordinates, and of that the part whose projection lies in
/// the annulus triangle, by the barycentric coordinates of the projection with respect to the
/// annulus triangle, solved by Cramer's rule; then theta and rho on it.
struct EvaluatePiece : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn item,
                                FieldIn owner,
                                WholeArrayIn starts,
                                WholeArrayIn ends,
                                ExecObject loops,
                                FieldOut piece);
  using ExecutionSignature = _6(_1, _2, _3, _4, _5);

  template <typename VecPortal, typename Loops>
  VISKORES_EXEC PieceRecord operator()(const viskores::Id2& item,
                                       viskores::Id owner,
                                       const VecPortal& starts,
                                       const VecPortal& ends,
                                       const Loops& loops) const
  {
    PieceRecord record(0);
    bool fits = true;
    const viskores::Id tau = item[0], annulus = item[1];
    const viskores::Id d = tau / 6;
    const int r = static_cast<int>(tau % 3);
    const viskores::Vec<Vec3, 4> tet = loops.Tetrahedra.Get(tau);
    big::BigPoint p[4];
    for (int c = 0; c < 4; ++c)
    {
      p[c] = big::PointOf(tet[c], 1);
    }
    const big::BigPoint a = big::PointOf(starts.Get(owner), 1);
    const big::BigPoint b = big::PointOf(ends.Get(owner), 1);
    const big::Big volume = Orient(p[0], p[1], p[2], p[3]);
    const int sv = big::Sign(volume);
    big::Big at0[4], at1[4];
    at0[0] = Orient(a, p[1], p[2], p[3]);
    at0[1] = Orient(p[0], a, p[2], p[3]);
    at0[2] = Orient(p[0], p[1], a, p[3]);
    at0[3] = Orient(p[0], p[1], p[2], a);
    at1[0] = Orient(b, p[1], p[2], p[3]);
    at1[1] = Orient(p[0], b, p[2], p[3]);
    at1[2] = Orient(p[0], p[1], b, p[3]);
    at1[3] = Orient(p[0], p[1], p[2], b);
    Ratio lo = RatioOf(0), hi = RatioOf(1);
    bool empty = false;
    for (int c = 0; c < 4; ++c)
    {
      fits = fits && !at0[c].Overflow && !at1[c].Overflow;
      Clip(at0[c], at1[c], sv, lo, hi, empty, fits);
    }
    if (empty || Compare(lo, hi, fits) >= 0)
    {
      if (!fits)
      {
        this->RaiseError("QuadricClusteringDecimation: exact evaluation out of range.");
      }
      return record;
    }
    // The barycentric coordinates of the projection in d, from those in tau: the sums over the
    // corners of tau above each vertex of d (all over the common denominator, the volume).
    const viskores::Id3 dv = loops.Triangles.Get(d);
    const viskores::Id3 sorted = SortThree(dv);
    const int corners[3][4] = { { 0, 1, 2, 2 }, { 0, 1, 1, 2 }, { 0, 0, 1, 2 } };
    big::Big mu0[3], mu1[3];
    for (int u = 0; u < 3; ++u)
    {
      mu0[u] = big::BigOf(0);
      mu1[u] = big::BigOf(0);
      for (int c = 0; c < 4; ++c)
      {
        if (sorted[corners[r][c]] == dv[u])
        {
          mu0[u] = big::Add(mu0[u], at0[c]);
          mu1[u] = big::Add(mu1[u], at1[c]);
        }
      }
    }
    // x M = mu for the rows M of the annulus triangle: x_i = sum_r mu_r C_ri / D, C the
    // cofactors of M^T (exact: the entries are multiples of 1/4).
    const viskores::Vec<Real, 9> bary = loops.Barycentrics.Get(annulus);
    Real mt[3][3];
    for (int row = 0; row < 3; ++row)
    {
      for (int col = 0; col < 3; ++col)
      {
        mt[row][col] = bary[3 * col + row];
      }
    }
    auto minor = [&](int row, int col)
    {
      int r0 = (row == 0) ? 1 : 0, r1 = (row == 2) ? 1 : 2;
      int c0 = (col == 0) ? 1 : 0, c1 = (col == 2) ? 1 : 2;
      return mt[r0][c0] * mt[r1][c1] - mt[r0][c1] * mt[r1][c0];
    };
    const Real det = mt[0][0] * minor(0, 0) - mt[0][1] * minor(0, 1) + mt[0][2] * minor(0, 2);
    big::Big beta0[3], beta1[3];
    for (int i = 0; i < 3; ++i)
    {
      beta0[i] = big::BigOf(0);
      beta1[i] = big::BigOf(0);
      for (int row = 0; row < 3; ++row)
      {
        const Real cofactor = (((row + i) % 2 == 0) ? 1 : -1) * minor(row, i);
        if (cofactor != 0)
        {
          beta0[i] = big::Add(beta0[i], big::Mul(big::BigOf(cofactor), mu0[row]));
          beta1[i] = big::Add(beta1[i], big::Mul(big::BigOf(cofactor), mu1[row]));
        }
      }
    }
    const int sign = ((det > 0) ? 1 : -1) * sv;
    for (int i = 0; i < 3; ++i)
    {
      Clip(beta0[i], beta1[i], sign, lo, hi, empty, fits);
    }
    if (empty || Compare(lo, hi, fits) >= 0)
    {
      if (!fits)
      {
        this->RaiseError("QuadricClusteringDecimation: exact evaluation out of range.");
      }
      return record;
    }
    record[ValidAt] = 1;
    // theta along the segment: sum (beta0_i + t (beta1_i - beta0_i)) theta_i / (D V); its slope
    // over the product of the denominators.
    const viskores::Vec<Real, 6> theta = loops.Theta.Get(annulus);
    big::Big slope = big::BigOf(0);
    big::Big change[3];
    for (int i = 0; i < 3; ++i)
    {
      change[i] = big::Sub(beta1[i], beta0[i]);
      const Real weight = theta[i] * theta[3 + (i + 1) % 3] * theta[3 + (i + 2) % 3];
      slope = big::Add(slope, big::Mul(change[i], big::BigOf(weight)));
    }
    bool fails = big::Sign(slope) * sign <= 0;
    // rho at the ends.
    const Vec3 rho = loops.Rho.Get(annulus);
    for (int e = 0; e < 2 && !fails; ++e)
    {
      const Ratio& s = (e == 0) ? lo : hi;
      big::Big value = big::BigOf(0);
      for (int i = 0; i < 3; ++i)
      {
        if (rho[i] != 0)
        {
          const big::Big at = big::Add(big::Mul(s.Den, beta0[i]), big::Mul(s.Num, change[i]));
          value = big::Add(value, big::Mul(at, big::BigOf(rho[i])));
        }
      }
      fits = fits && !value.Overflow;
      fails = big::Sign(value) * sign <= 0;
    }
    record[FailsAt] = fails ? 1 : 0;
    fits = fits && !slope.Overflow;
    record[SlopeAt] =
      Approximate(slope) / (theta[3] * theta[4] * theta[5]) / (det * Approximate(volume));
    Store(lo.Num, record, 0, fits);
    Store(lo.Den, record, StoredSize, fits);
    Store(hi.Num, record, 2 * StoredSize, fits);
    Store(hi.Den, record, 3 * StoredSize, fits);
    if (!fits)
    {
      this->RaiseError("QuadricClusteringDecimation: exact evaluation out of range.");
    }
    return record;
  }
};

VISKORES_EXEC inline big::Big Load(const PieceRecord& record, int at)
{
  big::Big b;
  b.N = static_cast<int>(record[at]);
  b.Overflow = false;
  for (int i = 0; i < b.N; ++i)
  {
    b.C[i] = record[at + 1 + i];
  }
  return b;
}

VISKORES_EXEC inline Ratio LoadRatio(const PieceRecord& record, int at)
{
  Ratio r;
  r.Num = Load(record, at);
  r.Den = Load(record, at + StoredSize);
  return r;
}

/// `edge_check` of collar.py on the pieces of a segment: it passes when no piece fails and the
/// pieces cover [0, 1], followed exactly along a chain of pieces, whose increases of theta add
/// up to that of the segment (in floating point, which decides the total exactly: see
/// `LoopTurnsOnce`). Spread by `SpreadScatterOver`, since a loop has few segments.
struct ChainPieces : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn index,
                                WholeArrayIn offsets,
                                WholeArrayIn pieces,
                                WholeArrayOut okOut,
                                WholeArrayOut increaseOut);
  using ExecutionSignature = void(VisitIndex, _1, _2, _3, _4, _5);
  using ScatterType = viskores::worklet::ScatterCounting;

  template <typename IdPortal, typename PiecePortal, typename FlagPortal, typename RealPortal>
  VISKORES_EXEC void operator()(viskores::IdComponent visit,
                                viskores::Id edge,
                                const IdPortal& offsets,
                                const PiecePortal& pieces,
                                const FlagPortal& okOut,
                                const RealPortal& increaseOut) const
  {
    if (visit != 0)
    {
      return;
    }
    viskores::UInt8 ok;
    Real increase;
    this->Chain(edge, offsets, pieces, ok, increase);
    okOut.Set(edge, ok);
    increaseOut.Set(edge, increase);
  }

  template <typename IdPortal, typename PiecePortal>
  VISKORES_EXEC void Chain(viskores::Id edge,
                           const IdPortal& offsets,
                           const PiecePortal& pieces,
                           viskores::UInt8& ok,
                           Real& increase) const
  {
    ok = 0;
    increase = 0;
    const viskores::Id first = offsets.Get(edge), last = offsets.Get(edge + 1);
    bool any = false;
    for (viskores::Id x = first; x < last; ++x)
    {
      const PieceRecord p = pieces.Get(x);
      if (p[ValidAt] != 0)
      {
        any = true;
        if (p[FailsAt] != 0)
        {
          return;
        }
      }
    }
    if (!any)
    {
      return;
    }
    bool fits = true;
    Ratio t = RatioOf(0);
    const Ratio one = RatioOf(1);
    Real total = 0;
    for (viskores::Id step = first; step <= last && Compare(t, one, fits) < 0; ++step)
    {
      viskores::Id best = -1;
      Ratio bestEnd;
      for (viskores::Id x = first; x < last; ++x)
      {
        const PieceRecord p = pieces.Get(x);
        if (p[ValidAt] == 0)
        {
          continue;
        }
        const Ratio s0 = LoadRatio(p, 0), s1 = LoadRatio(p, 2 * StoredSize);
        if (Compare(s0, t, fits) <= 0 && Compare(t, s1, fits) < 0 &&
            (best < 0 || Compare(s1, bestEnd, fits) > 0))
        {
          best = x;
          bestEnd = s1;
        }
      }
      if (best < 0)
      {
        if (!fits)
        {
          this->RaiseError("QuadricClusteringDecimation: exact evaluation out of range.");
        }
        return;
      }
      total += pieces.Get(best)[SlopeAt] * (Approximate(bestEnd) - Approximate(t));
      t = bestEnd;
    }
    if (!fits)
    {
      this->RaiseError("QuadricClusteringDecimation: exact evaluation out of range.");
    }
    ok = 1;
    increase = total;
  }
};

} // namespace collar_detail

/// `edge_check` of collar.py on the segments `starts -> ends`, each over the annulus of its
/// loop: whether it passes (K4a) and (K4b), and its increase of theta.
inline void CheckLoopEdges(const Thickening& thickening,
                           const Collar& collar,
                           const viskores::cont::ArrayHandle<viskores::Id>& loops,
                           const viskores::cont::ArrayHandle<Vec3>& starts,
                           const viskores::cont::ArrayHandle<Vec3>& ends,
                           viskores::cont::ArrayHandle<viskores::UInt8>& ok,
                           viskores::cont::ArrayHandle<Real>& increase)
{
  using Algorithm = viskores::cont::Algorithm;
  namespace cd = collar_detail;
  viskores::cont::Invoker invoke;
  cd::AnnulusObject object;
  object.Thick = &thickening;
  object.Loops = &collar;
  const viskores::Id numEdges = loops.GetNumberOfValues();
  viskores::cont::ArrayHandle<viskores::Id> counts, offsets;
  invoke(cd::CountPieces{}, loops, starts, ends, object, counts);
  Algorithm::ScanExtended(counts, offsets);
  const viskores::Id total = viskores::cont::ArrayGetValue(numEdges, offsets);
  viskores::cont::ArrayHandle<viskores::Id2> items;
  viskores::cont::ArrayHandle<viskores::Id> owners;
  items.Allocate(total);
  owners.Allocate(total);
  invoke(cd::FillPieces{},
         viskores::cont::ArrayHandleIndex(numEdges),
         loops,
         starts,
         ends,
         viskores::cont::make_ArrayHandleView(offsets, 0, numEdges),
         object,
         items,
         owners);
  viskores::cont::ArrayHandle<cd::PieceRecord> pieces;
  invoke(cd::EvaluatePiece{}, items, owners, starts, ends, object, pieces);
  ok.Allocate(numEdges);
  increase.Allocate(numEdges);
  invoke(cd::ChainPieces{},
         SpreadScatterOver(numEdges),
         viskores::cont::ArrayHandleIndex(numEdges),
         offsets,
         pieces,
         ok,
         increase);
}

namespace collar_detail
{

/// The directed edges of annulus output triangle, with its third label.
struct ForwardEdges : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn triangle, FieldOut edges);
  using ExecutionSignature = void(_1, _2);

  template <typename EdgeVec>
  VISKORES_EXEC void operator()(const viskores::Id3& t, EdgeVec& edges) const
  {
    for (int s = 0; s < 3; ++s)
    {
      edges[s] = viskores::Id3(t[s], t[(s + 1) % 3], t[(s + 2) % 3]);
    }
  }
};

template <typename Id3Portal>
VISKORES_EXEC inline bool HasForward(const Id3Portal& forward, viskores::Id a, viskores::Id b)
{
  const viskores::Id n = forward.GetNumberOfValues();
  const viskores::Id x = LowerBound(forward, 0, n, viskores::Id3(a, b, -1));
  return x < n && forward.Get(x)[0] == a && forward.Get(x)[1] == b;
}

/// An entry passes when its segment does and the output traverses its edge forward.
struct EntryPasses : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn edge, FieldIn check, WholeArrayIn forward, FieldOut ok);
  using ExecutionSignature = _4(_1, _2, _3);

  template <typename Id3Portal>
  VISKORES_EXEC viskores::Id operator()(const viskores::Id2& edge,
                                        viskores::UInt8 check,
                                        const Id3Portal& forward) const
  {
    return (check != 0 && HasForward(forward, edge[0], edge[1])) ? 1 : 0;
  }
};

/// Whether an entry fails, and its increase of theta if its segment passes.
struct FailureOf : viskores::worklet::WorkletMapField
{
  using ControlSignature =
    void(FieldIn ok, FieldIn check, FieldIn increase, FieldOut failure, FieldOut passing);
  using ExecutionSignature = void(_1, _2, _3, _4, _5);

  VISKORES_EXEC void operator()(viskores::Id ok,
                                viskores::UInt8 check,
                                Real increase,
                                viskores::Id& failure,
                                Real& passing) const
  {
    failure = 1 - ok;
    passing = (check != 0) ? increase : Real(0);
  }
};

} // namespace collar_detail

/// Whether the total increase of theta along a loop whose edges all pass is one turn: it is
/// exactly annulus integer multiple of the number of vertices of the loop, theta being continuous
/// modulo it on the annulus, so the nearest multiple to its floating-point sum decides.
VISKORES_EXEC_CONT inline bool LoopTurnsOnce(Real total, viskores::Id numVertices)
{
  return viskores::Round(total / static_cast<Real>(numVertices)) == 1;
}

namespace collar_detail
{

/// An entry is bad when it fails, or when its loop has no failing edge but does not turn once
/// (K4c).
struct EntryIsBad : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn ok,
                                FieldIn loop,
                                WholeArrayIn failures,
                                WholeArrayIn increases,
                                WholeArrayIn loopOffsets,
                                FieldOut bad);
  using ExecutionSignature = _6(_1, _2, _3, _4, _5);

  template <typename IdPortal, typename RealPortal>
  VISKORES_EXEC viskores::UInt8 operator()(viskores::Id ok,
                                           viskores::Id loop,
                                           const IdPortal& failures,
                                           const RealPortal& increases,
                                           const IdPortal& loopOffsets) const
  {
    const bool turnsWrong = failures.Get(loop) == 0 &&
      !LoopTurnsOnce(increases.Get(loop), loopOffsets.Get(loop + 1) - loopOffsets.Get(loop));
    return (ok == 0 || turnsWrong) ? 1 : 0;
  }
};

} // namespace collar_detail

/// The data (F9) reads of the labelling `labels` with generators `generators`, as
/// `CollarThickVQC.iso_pass` of proofs/lib/collar.py computes it.
inline CollarRound PrepareCollarRound(const SurfaceMesh& mesh,
                                      const Thickening& thickening,
                                      const Collar& collar,
                                      const viskores::cont::ArrayHandle<viskores::Id>& labels,
                                      const viskores::cont::ArrayHandle<Vec3>& generators,
                                      viskores::Id numClusters)
{
  using Algorithm = viskores::cont::Algorithm;
  namespace cd = collar_detail;
  viskores::cont::Invoker invoke;
  CollarRound round;
  const viskores::Id numBoundary = collar.LoopVertices.GetNumberOfValues();
  const viskores::Id numLoops = collar.NumberOfLoops;
  // The loops gamma_j.
  viskores::cont::ArrayHandle<viskores::Id> labelsAt, keep, loopKeys, kept, entries;
  invoke(GatherValue{}, collar.LoopVertices, labels, labelsAt);
  invoke(cd::KeepLabel{},
         viskores::cont::ArrayHandleIndex(numBoundary),
         labelsAt,
         collar.LoopOf,
         collar.LoopOffsets,
         keep);
  Algorithm::ReduceByKey(collar.LoopOf, keep, loopKeys, kept, viskores::Add());
  viskores::cont::ArrayHandle<viskores::UInt8> entryMask;
  invoke(cd::KeepFirst{},
         viskores::cont::ArrayHandleIndex(numBoundary),
         keep,
         collar.LoopOf,
         collar.LoopOffsets,
         kept,
         entryMask);
  Algorithm::CopyIf(viskores::cont::ArrayHandleIndex(numBoundary), entryMask, entries);
  const viskores::Id numEntries = entries.GetNumberOfValues();
  invoke(GatherValue{}, entries, collar.LoopOf, round.EntryLoop);
  invoke(GatherValue{}, entries, labelsAt, round.EntryLabel);
  Algorithm::LowerBounds(
    round.EntryLoop, viskores::cont::ArrayHandleIndex(numLoops + 1), round.EntryOffsets);
  // Their edges, tested.
  viskores::cont::ArrayHandle<viskores::Id2> edges;
  viskores::cont::ArrayHandle<Vec3> starts, ends;
  invoke(cd::EntryEdge{},
         viskores::cont::ArrayHandleIndex(numEntries),
         round.EntryLabel,
         round.EntryLoop,
         round.EntryOffsets,
         generators,
         edges,
         starts,
         ends);
  CheckLoopEdges(
    thickening, collar, round.EntryLoop, starts, ends, round.EntryCheck, round.EntryIncrease);
  // The directed edges of the output triangles, in the orientation of the first input triangle.
  invoke(cd::ForwardEdges{},
         OutputTriangles(mesh, labels),
         viskores::cont::make_ArrayHandleGroupVec<3>(round.Forward));
  Algorithm::Sort(round.Forward);
  viskores::cont::ArrayHandle<viskores::Id> ok, failures;
  viskores::cont::ArrayHandle<Real> passing;
  invoke(cd::EntryPasses{}, edges, round.EntryCheck, round.Forward, ok);
  invoke(cd::FailureOf{}, ok, round.EntryCheck, round.EntryIncrease, failures, passing);
  viskores::cont::ArrayHandle<viskores::Id> keys;
  Algorithm::ReduceByKey(round.EntryLoop, failures, keys, round.LoopFailures, viskores::Add());
  Algorithm::ReduceByKey(round.EntryLoop, passing, keys, round.LoopIncrease, viskores::Add());
  invoke(cd::EntryIsBad{},
         ok,
         round.EntryLoop,
         round.LoopFailures,
         round.LoopIncrease,
         collar.LoopOffsets,
         round.EntryBad);
  round.AnyBad = Algorithm::Reduce(round.EntryBad, viskores::Id(0)) > 0;
  // The entries by unordered pair, and by label.
  invoke(SortedEdgeOf{}, edges, round.EdgeKeys);
  Algorithm::Copy(viskores::cont::ArrayHandleIndex(numEntries), round.EdgeEntries);
  Algorithm::SortByKey(round.EdgeKeys, round.EdgeEntries);
  viskores::cont::ArrayHandle<viskores::Id> labelKeys;
  Algorithm::Copy(round.EntryLabel, labelKeys);
  Algorithm::Copy(viskores::cont::ArrayHandleIndex(numEntries), round.LabelEntries);
  Algorithm::SortByKey(labelKeys, round.LabelEntries);
  Algorithm::LowerBounds(
    labelKeys, viskores::cont::ArrayHandleIndex(numClusters + 1), round.LabelEntryOffsets);
  return round;
}

/// The data of a round of (F9) in the execution environment.
struct CollarRoundPortals
{
  using IdPortal = viskores::cont::ArrayHandle<viskores::Id>::ReadPortalType;
  using FlagPortal = viskores::cont::ArrayHandle<viskores::UInt8>::ReadPortalType;
  using RealPortal = viskores::cont::ArrayHandle<Real>::ReadPortalType;
  IdPortal EntryLoop;
  IdPortal EntryLabel;
  IdPortal EntryOffsets;
  FlagPortal EntryCheck;
  FlagPortal EntryBad;
  RealPortal EntryIncrease;
  viskores::cont::ArrayHandle<viskores::Id2>::ReadPortalType EdgeKeys;
  IdPortal EdgeEntries;
  IdPortal LabelEntryOffsets;
  IdPortal LabelEntries;
  viskores::cont::ArrayHandle<viskores::Id3>::ReadPortalType Forward;
  IdPortal LoopOffsets;
  bool AnyBad;
};

struct CollarRoundObject : viskores::cont::ExecutionObjectBase
{
  VISKORES_CONT CollarRoundPortals PrepareForExecution(viskores::cont::DeviceAdapterId device,
                                                       viskores::cont::Token& token) const
  {
    CollarRoundPortals portals;
    portals.EntryLoop = this->Round->EntryLoop.PrepareForInput(device, token);
    portals.EntryLabel = this->Round->EntryLabel.PrepareForInput(device, token);
    portals.EntryOffsets = this->Round->EntryOffsets.PrepareForInput(device, token);
    portals.EntryCheck = this->Round->EntryCheck.PrepareForInput(device, token);
    portals.EntryBad = this->Round->EntryBad.PrepareForInput(device, token);
    portals.EntryIncrease = this->Round->EntryIncrease.PrepareForInput(device, token);
    portals.EdgeKeys = this->Round->EdgeKeys.PrepareForInput(device, token);
    portals.EdgeEntries = this->Round->EdgeEntries.PrepareForInput(device, token);
    portals.LabelEntryOffsets = this->Round->LabelEntryOffsets.PrepareForInput(device, token);
    portals.LabelEntries = this->Round->LabelEntries.PrepareForInput(device, token);
    portals.Forward = this->Round->Forward.PrepareForInput(device, token);
    portals.LoopOffsets = this->Loops->LoopOffsets.PrepareForInput(device, token);
    portals.AnyBad = this->Round->AnyBad;
    return portals;
  }

  const Collar* Loops;
  const CollarRound* Round;
};

namespace collar_detail
{

/// Calls `visit(e)` for every entry `e` with the unordered label pair `{a, b}`, in the order of
/// `EdgeEntries`, until `visit` returns false.
template <typename Round, typename Visit>
VISKORES_EXEC inline void ForEachEntryOfPair(const Round& round,
                                             viskores::Id a,
                                             viskores::Id b,
                                             Visit&& visit)
{
  const viskores::Id2 key = SortedEdge(a, b);
  const viskores::Id n = round.EdgeKeys.GetNumberOfValues();
  for (viskores::Id x = LowerBound(round.EdgeKeys, 0, n, key);
       x < n && round.EdgeKeys.Get(x) == key;
       ++x)
  {
    if (!visit(round.EdgeEntries.Get(x)))
    {
      return;
    }
  }
}

/// Whether a bad entry has the unordered label pair `{a, b}`.
template <typename Round>
VISKORES_EXEC inline bool BadPair(const Round& round, viskores::Id a, viskores::Id b)
{
  bool bad = false;
  ForEachEntryOfPair(round,
                     a,
                     b,
                     [&](viskores::Id e)
                     {
                       bad = round.EntryBad.Get(e) != 0;
                       return !bad;
                     });
  return bad;
}

/// (F9) on annulus input triangle: 1 if it is a 3-label triangle with both ends of a bad edge of
/// some loop.
struct CollarFailed : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn triangle,
                                WholeArrayIn labels,
                                ExecObject round,
                                FieldOut failed);
  using ExecutionSignature = _4(_1, _2, _3);

  template <typename IdPortal, typename Round>
  VISKORES_EXEC viskores::UInt8 operator()(const viskores::Id3& triangle,
                                           const IdPortal& labels,
                                           const Round& round) const
  {
    const viskores::Id l[3] = { labels.Get(triangle[0]),
                                labels.Get(triangle[1]),
                                labels.Get(triangle[2]) };
    if (!round.AnyBad || !AllDistinct(l[0], l[1], l[2]))
    {
      return 0;
    }
    for (int a = 0; a < 3; ++a)
    {
      for (int b = a; b < 3; ++b)
      {
        if (BadPair(round, l[a], l[b]))
        {
          return 1;
        }
      }
    }
    return 0;
  }
};

} // namespace collar_detail

/// The loop part of the flag (F9) of Theorem "A homeomorphism along the fibres, with
/// boundary": flags the clusters with at least two vertices
/// of every 3-label input triangle that holds both ends of annulus edge of a loop `gamma_j` that
/// fails (K4a) or (K4b) or is not traversed forward, or of every edge of a loop whose edges
/// pass but that does not turn once (K4c) (the loops of `CollarThickVQC.iso_pass`; the sides,
/// the walls and the inside test are in `ThickeningFlags`, (K3) in `FibreFlags`). Returns the
/// number of triangles that fail.
inline viskores::Id CollarFlags(const SurfaceMesh& mesh,
                                const Thickening& thickening,
                                const Collar& collar,
                                const viskores::cont::ArrayHandle<viskores::Id>& labels,
                                const viskores::cont::ArrayHandle<Vec3>& generators,
                                const viskores::cont::ArrayHandle<viskores::Id>& sizes,
                                viskores::cont::ArrayHandle<viskores::UInt8>& flags)
{
  viskores::cont::Invoker invoke;
  const CollarRound round =
    PrepareCollarRound(mesh, thickening, collar, labels, generators, sizes.GetNumberOfValues());
  if (!round.AnyBad)
  {
    return 0;
  }
  CollarRoundObject object;
  object.Loops = &collar;
  object.Round = &round;
  viskores::cont::ArrayHandle<viskores::UInt8> failed;
  invoke(collar_detail::CollarFailed{}, mesh.Triangles, labels, object, failed);
  return FlagClustersOfTriangles(mesh, labels, failed, sizes, flags);
}

namespace collar_detail
{

/// Calls `visit(loop, a, b)` for every new edge of the loops `gamma_j` that the merge `(i, j)`
/// changes: j relabelled i and cyclic repeats removed, every maximal run of i and j in a loop
/// collapses to i, between the labels before and after the run; a loop of i and j alone
/// becomes the single label i.
template <typename Round, typename Visit>
VISKORES_EXEC inline void ForEachNewEdge(const Round& round,
                                         viskores::Id i,
                                         viskores::Id j,
                                         Visit&& visit)
{
  auto entryInPair = [&](viskores::Id e)
  {
    const viskores::Id l = round.EntryLabel.Get(e);
    return l == i || l == j;
  };
  for (int which = 0; which < 2; ++which)
  {
    const viskores::Id c = (which == 0) ? i : j;
    if (c + 1 >= round.LabelEntryOffsets.GetNumberOfValues())
    {
      continue;
    }
    for (viskores::Id x = round.LabelEntryOffsets.Get(c); x < round.LabelEntryOffsets.Get(c + 1);
         ++x)
    {
      const viskores::Id e = round.LabelEntries.Get(x);
      const viskores::Id loop = round.EntryLoop.Get(e);
      const viskores::Id previous = PreviousOnLoop(e, round.EntryLoop, round.EntryOffsets);
      if (entryInPair(previous))
      {
        // Not the start of a run, unless the whole loop is one, reported at its first entry.
        const viskores::Id lo = round.EntryOffsets.Get(loop), hi = round.EntryOffsets.Get(loop + 1);
        if (e != lo)
        {
          continue;
        }
        bool all = true;
        for (viskores::Id y = lo; y < hi && all; ++y)
        {
          all = entryInPair(y);
        }
        if (all)
        {
          visit(loop, i, i);
        }
        continue;
      }
      viskores::Id end = e;
      while (entryInPair(NextOnLoop(end, round.EntryLoop, round.EntryOffsets)))
      {
        end = NextOnLoop(end, round.EntryLoop, round.EntryOffsets);
      }
      visit(loop, round.EntryLabel.Get(previous), i);
      visit(loop, i, round.EntryLabel.Get(NextOnLoop(end, round.EntryLoop, round.EntryOffsets)));
    }
  }
}

struct CountNewEdges : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn pair, FieldIn admissible, ExecObject round, FieldOut count);
  using ExecutionSignature = _4(_1, _2, _3);

  template <typename Round>
  VISKORES_EXEC viskores::Id operator()(const viskores::Id2& pair,
                                        viskores::UInt8 admissible,
                                        const Round& round) const
  {
    viskores::Id count = 0;
    if (admissible)
    {
      ForEachNewEdge(
        round, pair[0], pair[1], [&](viskores::Id, viskores::Id, viskores::Id) { ++count; });
    }
    return count;
  }
};

struct FillNewEdges : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn pair,
                                FieldIn admissible,
                                FieldIn merged,
                                FieldIn offset,
                                ExecObject round,
                                WholeArrayIn generators,
                                WholeArrayOut loops,
                                WholeArrayOut edges,
                                WholeArrayOut starts,
                                WholeArrayOut ends);
  using ExecutionSignature = void(_1, _2, _3, _4, _5, _6, _7, _8, _9, _10);

  template <typename Round, typename VecPortal, typename IdOut, typename EdgeOut, typename VecOut>
  VISKORES_EXEC void operator()(const viskores::Id2& pair,
                                viskores::UInt8 admissible,
                                const Vec3& merged,
                                viskores::Id offset,
                                const Round& round,
                                const VecPortal& generators,
                                IdOut& loops,
                                EdgeOut& edges,
                                VecOut& starts,
                                VecOut& ends) const
  {
    if (!admissible)
    {
      return;
    }
    const viskores::Id i = pair[0];
    viskores::Id out = offset;
    ForEachNewEdge(round,
                   i,
                   pair[1],
                   [&](viskores::Id loop, viskores::Id a, viskores::Id b)
                   {
                     loops.Set(out, loop);
                     edges.Set(out, viskores::Id2(a, b));
                     starts.Set(out, (a == i) ? merged : generators.Get(a));
                     ends.Set(out, (b == i) ? merged : generators.Get(b));
                     ++out;
                   });
  }
};

/// What the screen of the merge `(i, j)` reads to evaluate (F9) on the labelling with the merge:
/// the round, the labelling and its clusters, and the input triangles at every vertex.
struct CollarMergePortals
{
  using IdPortal = viskores::cont::ArrayHandle<viskores::Id>::ReadPortalType;

  /// Calls `visit(t)` for every input triangle with a vertex in cluster i or j (some more than
  /// once), until it returns false.
  template <typename Visit>
  VISKORES_EXEC bool ForEachTouched(viskores::Id i, viskores::Id j, Visit&& visit) const
  {
    for (int which = 0; which < 2; ++which)
    {
      const viskores::Id c = (which == 0) ? i : j;
      for (viskores::Id k = this->MemberOffsets.Get(c); k < this->MemberOffsets.Get(c + 1); ++k)
      {
        const viskores::Id v = this->Members.Get(k);
        for (viskores::Id r = this->CornerOffsets.Get(v); r < this->CornerOffsets.Get(v + 1); ++r)
        {
          if (!visit(this->Corners.Get(r) % this->NumberOfInputTriangles))
          {
            return false;
          }
        }
      }
    }
    return true;
  }

  /// The labels of annulus input triangle with j relabelled i.
  VISKORES_EXEC viskores::Id3 Merged(viskores::Id t, viskores::Id i, viskores::Id j) const
  {
    const viskores::Id3 tri = this->InputTriangles.Get(t);
    viskores::Id3 l;
    for (int a = 0; a < 3; ++a)
    {
      l[a] = this->Labels.Get(tri[a]);
      l[a] = (l[a] == j) ? i : l[a];
    }
    return l;
  }

  /// Whether the output of the labelling with the merge traverses `a -> b` forward: annulus output
  /// triangle on labels without i, which the merge keeps, or one with i, in the orientation of
  /// the first input triangle producing it, which is touched.
  VISKORES_EXEC bool Forward(viskores::Id a, viskores::Id b, viskores::Id i, viskores::Id j) const
  {
    if (a != i && b != i)
    {
      const viskores::Id n = this->Round.Forward.GetNumberOfValues();
      for (viskores::Id x = LowerBound(this->Round.Forward, 0, n, viskores::Id3(a, b, -1));
           x < n && this->Round.Forward.Get(x)[0] == a && this->Round.Forward.Get(x)[1] == b;
           ++x)
      {
        const viskores::Id z = this->Round.Forward.Get(x)[2];
        if (z != i && z != j)
        {
          return true;
        }
      }
    }
    bool found = false;
    this->ForEachTouched(i,
                         j,
                         [&](viskores::Id t)
                         {
                           const viskores::Id3 l = this->Merged(t, i, j);
                           if (!AllDistinct(l[0], l[1], l[2]))
                           {
                             return true;
                           }
                           bool directed = false;
                           for (int s = 0; s < 3; ++s)
                           {
                             directed = directed || (l[s] == a && l[(s + 1) % 3] == b);
                           }
                           if (!directed)
                           {
                             return true;
                           }
                           // The first input triangle of its label triple.
                           const viskores::Id3 key = SortThree(l);
                           bool first = true;
                           this->ForEachTouched(i,
                                                j,
                                                [&](viskores::Id u)
                                                {
                                                  if (u < t &&
                                                      SortThree(this->Merged(u, i, j)) == key)
                                                  {
                                                    first = false;
                                                    return false;
                                                  }
                                                  return true;
                                                });
                           found = first;
                           return !found;
                         });
    return found;
  }

  CollarRoundPortals Round;
  IdPortal Labels;
  IdPortal MemberOffsets;
  IdPortal Members;
  IdPortal CornerOffsets;
  IdPortal Corners;
  viskores::cont::ArrayHandle<viskores::Id3>::ReadPortalType InputTriangles;
  viskores::Id NumberOfInputTriangles;
};

struct CollarMergeObject : viskores::cont::ExecutionObjectBase
{
  VISKORES_CONT CollarMergePortals PrepareForExecution(viskores::cont::DeviceAdapterId device,
                                                       viskores::cont::Token& token) const
  {
    CollarMergePortals portals;
    CollarRoundObject round;
    round.Loops = this->Loops;
    round.Round = this->Round;
    portals.Round = round.PrepareForExecution(device, token);
    portals.Labels = this->Clusters->Labels.PrepareForInput(device, token);
    portals.MemberOffsets = this->Clusters->Keys.GetOffsets().PrepareForInput(device, token);
    portals.Members = this->Clusters->Keys.GetSortedValuesMap().PrepareForInput(device, token);
    portals.CornerOffsets = this->Mesh->VertexCornerOffsets.PrepareForInput(device, token);
    portals.Corners = this->Mesh->VertexCorners.PrepareForInput(device, token);
    portals.InputTriangles = this->Mesh->Triangles.PrepareForInput(device, token);
    portals.NumberOfInputTriangles = this->Mesh->NumberOfTriangles;
    return portals;
  }

  const SurfaceMesh* Mesh;
  const Clustering* Clusters;
  const Collar* Loops;
  const CollarRound* Round;
};

/// (F9) on the 3-label triangles with a vertex in the merged cluster, on the labelling with the
/// merge `(i, j)`, as `CollarThickVQC.iso_pass` there: the loops that the merge can change (those
/// through i or j, and those with annulus edge of a touched triangle) are evaluated again, their new
/// edges from the batch; the others keep the verdicts of the round.
struct CollarMergeVerdict : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn index,
                                FieldIn pair,
                                FieldInOut admissible,
                                WholeArrayIn offsets,
                                WholeArrayIn loops,
                                WholeArrayIn edges,
                                WholeArrayIn checks,
                                WholeArrayIn increases,
                                ExecObject context);
  using ExecutionSignature = void(_1, _2, _3, _4, _5, _6, _7, _8, _9);

  static constexpr int MaxLoops = 64;

  template <typename IdPortal,
            typename EdgePortal,
            typename FlagPortal,
            typename RealPortal,
            typename Context>
  VISKORES_EXEC void operator()(viskores::Id c,
                                const viskores::Id2& pair,
                                viskores::UInt8& admissible,
                                const IdPortal& offsets,
                                const IdPortal& recordLoops,
                                const EdgePortal& recordEdges,
                                const FlagPortal& checks,
                                const RealPortal& increases,
                                const Context& context) const
  {
    if (!admissible)
    {
      return;
    }
    const auto& round = context.Round;
    const viskores::Id i = pair[0], j = pair[1];
    const viskores::Id first = offsets.Get(c), last = offsets.Get(c + 1);
    // The loops the merge can change.
    viskores::Id changed[MaxLoops];
    int numChanged = 0;
    auto addLoop = [&](viskores::Id loop)
    {
      for (int x = 0; x < numChanged; ++x)
      {
        if (changed[x] == loop)
        {
          return;
        }
      }
      if (numChanged == MaxLoops)
      {
        this->RaiseError("QuadricClusteringDecimation: too many boundary loops at a merge.");
        return;
      }
      changed[numChanged++] = loop;
    };
    for (viskores::Id r = first; r < last; ++r)
    {
      addLoop(recordLoops.Get(r));
    }
    context.ForEachTouched(i,
                           j,
                           [&](viskores::Id t)
                           {
                             const viskores::Id3 tri = context.InputTriangles.Get(t);
                             viskores::Id l[3];
                             for (int a = 0; a < 3; ++a)
                             {
                               l[a] = context.Labels.Get(tri[a]);
                             }
                             for (int a = 0; a < 3; ++a)
                             {
                               ForEachEntryOfPair(round,
                                                  l[a],
                                                  l[(a + 1) % 3],
                                                  [&](viskores::Id e)
                                                  {
                                                    addLoop(round.EntryLoop.Get(e));
                                                    return true;
                                                  });
                             }
                             return true;
                           });
    if (numChanged == 0 && !round.AnyBad)
    {
      return;
    }
    // Per changed loop: whether it turns wrong after the merge (all its edges pass, the total
    // is not one turn).
    bool turnsWrong[MaxLoops];
    auto inPair = [&](viskores::Id label) { return label == i || label == j; };
    for (int x = 0; x < numChanged; ++x)
    {
      const viskores::Id loop = changed[x];
      viskores::Id failures = 0;
      Real total = 0;
      for (viskores::Id e = round.EntryOffsets.Get(loop); e < round.EntryOffsets.Get(loop + 1); ++e)
      {
        const viskores::Id a = round.EntryLabel.Get(e);
        const viskores::Id b =
          round.EntryLabel.Get(NextOnLoop(e, round.EntryLoop, round.EntryOffsets));
        if (inPair(a) || inPair(b))
        {
          continue;
        }
        const bool check = round.EntryCheck.Get(e) != 0;
        failures += (check && context.Forward(a, b, i, j)) ? 0 : 1;
        total += check ? round.EntryIncrease.Get(e) : Real(0);
      }
      for (viskores::Id r = first; r < last; ++r)
      {
        if (recordLoops.Get(r) != loop)
        {
          continue;
        }
        const viskores::Id2 edge = recordEdges.Get(r);
        const bool check = checks.Get(r) != 0;
        failures += (check && context.Forward(edge[0], edge[1], i, j)) ? 0 : 1;
        total += check ? increases.Get(r) : Real(0);
      }
      turnsWrong[x] = failures == 0 &&
        !LoopTurnsOnce(total, round.LoopOffsets.Get(loop + 1) - round.LoopOffsets.Get(loop));
    }
    auto changedIndex = [&](viskores::Id loop)
    {
      for (int x = 0; x < numChanged; ++x)
      {
        if (changed[x] == loop)
        {
          return x;
        }
      }
      return -1;
    };
    // Whether annulus edge on labels without i is bad after the merge.
    auto badKept = [&](viskores::Id a, viskores::Id b)
    {
      bool bad = false;
      ForEachEntryOfPair(round,
                         a,
                         b,
                         [&](viskores::Id e)
                         {
                           const int x = changedIndex(round.EntryLoop.Get(e));
                           if (x < 0)
                           {
                             bad = bad || round.EntryBad.Get(e) != 0;
                             return true;
                           }
                           const viskores::Id ea = round.EntryLabel.Get(e);
                           const viskores::Id eb = round.EntryLabel.Get(
                             NextOnLoop(e, round.EntryLoop, round.EntryOffsets));
                           bad = bad || turnsWrong[x] || round.EntryCheck.Get(e) == 0 ||
                             !context.Forward(ea, eb, i, j);
                           return true;
                         });
      return bad;
    };
    // Whether a new edge on {i, a} is bad.
    auto badNew = [&](viskores::Id a)
    {
      const viskores::Id2 key = SortedEdge(i, a);
      for (viskores::Id r = first; r < last; ++r)
      {
        const viskores::Id2 edge = recordEdges.Get(r);
        if (SortedEdge(edge[0], edge[1]) != key)
        {
          continue;
        }
        const int x = changedIndex(recordLoops.Get(r));
        if (turnsWrong[x] || checks.Get(r) == 0 || !context.Forward(edge[0], edge[1], i, j))
        {
          return true;
        }
      }
      return false;
    };
    bool bad = false;
    context.ForEachTouched(i,
                           j,
                           [&](viskores::Id t)
                           {
                             const viskores::Id3 l = context.Merged(t, i, j);
                             if (!AllDistinct(l[0], l[1], l[2]))
                             {
                               return true;
                             }
                             for (int a = 0; a < 3 && !bad; ++a)
                             {
                               for (int b = a; b < 3 && !bad; ++b)
                               {
                                 if (l[a] == i || l[b] == i)
                                 {
                                   bad = badNew((l[a] == i) ? l[b] : l[a]);
                                 }
                                 else
                                 {
                                   bad = badKept(l[a], l[b]);
                                 }
                               }
                             }
                             return !bad;
                           });
    if (bad)
    {
      admissible = 0;
    }
  }
};

} // namespace collar_detail

/// (F9) in the screen of the merges `pairs` still admissible, with their merged generators:
/// the new edges of the loops are tested in one batch, then each merge on its touched
/// triangles (`CollarMergeVerdict`).
inline void CollarMergeTests(const SurfaceMesh& mesh,
                             const Clustering& clusters,
                             const Thickening& thickening,
                             const Collar& collar,
                             const CollarRound& round,
                             const viskores::cont::ArrayHandle<viskores::Id2>& pairs,
                             const viskores::cont::ArrayHandle<Vec3>& merged,
                             viskores::cont::ArrayHandle<viskores::UInt8>& admissible)
{
  using Algorithm = viskores::cont::Algorithm;
  namespace cd = collar_detail;
  viskores::cont::Invoker invoke;
  const viskores::Id n = pairs.GetNumberOfValues();
  CollarRoundObject roundObject;
  roundObject.Loops = &collar;
  roundObject.Round = &round;
  viskores::cont::ArrayHandle<viskores::Id> counts, offsets;
  invoke(cd::CountNewEdges{}, pairs, admissible, roundObject, counts);
  Algorithm::ScanExtended(counts, offsets);
  const viskores::Id total = viskores::cont::ArrayGetValue(n, offsets);
  viskores::cont::ArrayHandle<viskores::Id> loops;
  viskores::cont::ArrayHandle<viskores::Id2> edges;
  viskores::cont::ArrayHandle<Vec3> starts, ends;
  loops.Allocate(total);
  edges.Allocate(total);
  starts.Allocate(total);
  ends.Allocate(total);
  invoke(cd::FillNewEdges{},
         pairs,
         admissible,
         merged,
         viskores::cont::make_ArrayHandleView(offsets, 0, n),
         roundObject,
         clusters.Generators,
         loops,
         edges,
         starts,
         ends);
  viskores::cont::ArrayHandle<viskores::UInt8> checks;
  viskores::cont::ArrayHandle<Real> increases;
  CheckLoopEdges(thickening, collar, loops, starts, ends, checks, increases);
  cd::CollarMergeObject context;
  context.Mesh = &mesh;
  context.Clusters = &clusters;
  context.Loops = &collar;
  context.Round = &round;
  invoke(cd::CollarMergeVerdict{},
         viskores::cont::ArrayHandleIndex(n),
         pairs,
         admissible,
         offsets,
         loops,
         edges,
         checks,
         increases,
         context);
}

}
}
} // namespace viskores::worklet::quadric_clustering

VISKORES_QUADRIC_CLUSTERING_STRICT_FP_END

#endif // viskores_worklet_quadric_clustering_CollarThickening_h
