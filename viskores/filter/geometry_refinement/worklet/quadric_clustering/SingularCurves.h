//============================================================================
//  The contents of this file are covered by the Viskores license. See
//  LICENSE.txt for details.
//
//  By contributing to this file, all contributors agree to the Developer
//  Certificate of Origin Version 1.1 (DCO 1.1) as stated in DCO.txt.
//============================================================================
#ifndef viskores_worklet_quadric_clustering_SingularCurves_h
#define viskores_worklet_quadric_clustering_SingularCurves_h

#include <viskores/filter/geometry_refinement/worklet/quadric_clustering/Clustering.h>
#include <viskores/filter/geometry_refinement/worklet/quadric_clustering/NonManifoldCut.h>
#include <viskores/filter/geometry_refinement/worklet/quadric_clustering/QuadricClusteringCommonWorklets.h>
#include <viskores/filter/geometry_refinement/worklet/quadric_clustering/TopologyFlags.h>

#include <viskores/cont/Algorithm.h>
#include <viskores/cont/ArrayGetValues.h>
#include <viskores/cont/ArrayHandleConcatenate.h>
#include <viskores/cont/ArrayHandleGroupVec.h>
#include <viskores/cont/ArrayHandleIndex.h>
#include <viskores/cont/ArrayHandlePermutation.h>
#include <viskores/cont/ArrayHandleView.h>
#include <viskores/worklet/Keys.h>
#include <viskores/worklet/WorkletMapField.h>
#include <viskores/worklet/WorkletReduceByKey.h>

VISKORES_QUADRIC_CLUSTERING_STRICT_FP_BEGIN

namespace viskores
{
namespace worklet
{
namespace quadric_clustering
{

namespace curves_detail
{

/// Whether an edge in exactly two triangles has unit normals with a dot product, times the sign
/// of the edge, below `CosAngle`.
struct IsCrease : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn edgeTriangles,
                                FieldIn count,
                                FieldIn sign,
                                WholeArrayIn triangles,
                                WholeArrayIn points,
                                FieldOut crease);
  using ExecutionSignature = _6(_1, _2, _3, _4, _5);

  VISKORES_CONT explicit IsCrease(Real cosAngle)
    : CosAngle(cosAngle)
  {
  }

  template <typename TrianglePortal, typename PointPortal>
  VISKORES_EXEC Vec3 Normal(viskores::Id t,
                            const TrianglePortal& triangles,
                            const PointPortal& points) const
  {
    const viskores::Id3 tri = triangles.Get(t);
    const Vec3 a = points.Get(tri[0]);
    const Vec3 x = Cross(Sub(points.Get(tri[1]), a), Sub(points.Get(tri[2]), a));
    const Real twice = 2 * (Norm(x) / 2);
    return Vec3(x[0] / twice, x[1] / twice, x[2] / twice);
  }

  template <typename TrianglePortal, typename PointPortal>
  VISKORES_EXEC viskores::UInt8 operator()(const viskores::Id2& edgeTriangles,
                                           viskores::IdComponent count,
                                           viskores::Int8 sign,
                                           const TrianglePortal& triangles,
                                           const PointPortal& points) const
  {
    if (count != 2)
    {
      return 0;
    }
    return (Real(sign) *
              Dot(this->Normal(edgeTriangles[0], triangles, points),
                  this->Normal(edgeTriangles[1], triangles, points)) <
            this->CosAngle)
      ? 1
      : 0;
  }

  Real CosAngle;
};

} // namespace curves_detail

/// The creases of a mesh: its edges in exactly two triangles whose unit normals have a dot
/// product, times the sign of the edge, below `cosAngle`, as sorted vertex pairs in increasing order. The normals are
/// `x / (2 a)` with `x` the cross product of the edges at the first corner and `a = |x| / 2`, as
/// `Mesh.nf` of proofs/lib/vqc.py, so the test is exact against the reference.
inline viskores::cont::ArrayHandle<viskores::Id2> CreaseEdges(const SurfaceMesh& mesh,
                                                              Real cosAngle)
{
  viskores::cont::Invoker invoke;
  viskores::cont::ArrayHandle<viskores::UInt8> isCrease;
  invoke(curves_detail::IsCrease(cosAngle),
         mesh.EdgeTriangles,
         mesh.EdgeTriangleCounts,
         mesh.EdgeSigns,
         mesh.Triangles,
         mesh.Points,
         isCrease);
  viskores::cont::ArrayHandle<viskores::Id2> creases;
  viskores::cont::Algorithm::CopyIf(mesh.Edges, isCrease, creases);
  viskores::cont::Algorithm::Sort(creases);
  return creases;
}

/// The singular curves of an input cut into surfaces (Proposition "Decimating the singular
/// curves"), on the device. A frozen input vertex is regular if it lies on exactly two edges of
/// the cut `X` and every copy of it lies on one copy of each; the others are junctions. An arc
/// is a path of edges of `X` whose interior vertices are regular and whose ends are junctions,
/// or a cycle of regular vertices. The interior vertices of every arc are grouped into segments,
/// runs of consecutive vertices, initially one per vertex; in every sheet the copies of a
/// segment form one cluster, a piece, and the pieces of a segment are one vertex of the glued
/// output. With junction merges, the end segments of an open arc may join its junctions.
struct SingularCurves
{
  viskores::Id NumberOfArcs = 0;
  /// The interior vertices of every arc in order, arc after arc: the open arcs by their first
  /// junction and its neighbour, then the closed arcs by their smallest vertex, which they start
  /// with, followed by its smaller neighbour.
  viskores::cont::ArrayHandle<viskores::Id> ArcOffsets;
  viskores::cont::ArrayHandle<viskores::Id> ArcVertices;
  /// The junctions at the two ends of every arc, or `(-1, -1)` for a closed arc.
  viskores::cont::ArrayHandle<viskores::Id2> ArcEnds;
  /// Per position of `ArcVertices`: whether a segment starts there.
  viskores::cont::ArrayHandle<viskores::UInt8> SegmentHead;
  /// Per arc: the position of the head of its first segment. A closed arc whose last segment
  /// absorbed its first starts later.
  viskores::cont::ArrayHandle<viskores::Id> FirstHead;
  /// Per arc: how many of its vertices joined the junction at each end.
  viskores::cont::ArrayHandle<viskores::Id2> Absorbed;
  /// Per input vertex: the first vertex of its segment, its junction for a junction or a vertex
  /// that joined one, or -1 for a vertex that is not frozen.
  viskores::cont::ArrayHandle<viskores::Id> Segment;
  /// Per input vertex: whether it is a regular arc vertex.
  viskores::cont::ArrayHandle<viskores::UInt8> Regular;
  /// Per edge of the cut mesh: whether it is a copy of an edge of `X`.
  viskores::cont::ArrayHandle<viskores::UInt8> CutEdge;
  /// The copies of every edge of `X`: the edges of the cut mesh that copy one, sorted by the
  /// edge of `X` they copy, then by index.
  viskores::cont::ArrayHandle<viskores::Id2> CopyKeys;
  viskores::cont::ArrayHandle<viskores::Id2> CopyEdges;
};

namespace curves_detail
{

/// Whether an edge of the cut mesh is a copy of an edge of `X`, and the edge it copies.
struct CopiedEdge : viskores::worklet::WorkletMapField
{
  using ControlSignature =
    void(FieldIn edge, WholeArrayIn original, WholeArrayIn cutEdges, FieldOut inX, FieldOut record);
  using ExecutionSignature = void(_1, _2, _3, _4, _5, WorkIndex);

  template <typename IdPortal, typename EdgePortal>
  VISKORES_EXEC void operator()(const viskores::Id2& edge,
                                const IdPortal& original,
                                const EdgePortal& cutEdges,
                                viskores::UInt8& inX,
                                viskores::Id3& record,
                                viskores::Id h) const
  {
    const viskores::Id a = original.Get(edge[0]), b = original.Get(edge[1]);
    const viskores::Id2 key(viskores::Min(a, b), viskores::Max(a, b));
    inX = (FindSorted(cutEdges, key) >= 0) ? 1 : 0;
    record = viskores::Id3(key[0], key[1], h);
  }
};

struct RecordKey : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn record, WholeArrayIn edges, FieldOut key, FieldOut edge);
  using ExecutionSignature = void(_1, _2, _3, _4);

  template <typename EdgePortal>
  VISKORES_EXEC void operator()(const viskores::Id3& record,
                                const EdgePortal& edges,
                                viskores::Id2& key,
                                viskores::Id2& edge) const
  {
    key = viskores::Id2(record[0], record[1]);
    edge = edges.Get(record[2]);
  }
};

/// For a copy of an edge of `X`: the input vertex that each end reaches through it.
struct CopyReach : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn edge, FieldIn inX, WholeArrayIn original, FieldOut reach);
  using ExecutionSignature = void(_1, _2, _3, _4);

  template <typename IdPortal, typename OutVec>
  VISKORES_EXEC void operator()(const viskores::Id2& e,
                                viskores::UInt8 inX,
                                const IdPortal& original,
                                OutVec& out) const
  {
    out[0] = inX ? viskores::Id2(e[0], original.Get(e[1])) : viskores::Id2(-1, -1);
    out[1] = inX ? viskores::Id2(e[1], original.Get(e[0])) : viskores::Id2(-1, -1);
  }
};

/// Whether an input vertex is a regular arc vertex: frozen, on exactly two edges of `X`, and
/// every copy reaching exactly those two neighbours through copies of edges of `X`; and its
/// initial segment.
struct RegularVertex : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn vertex,
                                WholeArrayIn isFrozen,
                                WholeArrayIn adjacencyOffsets,
                                WholeArrayIn adjacency,
                                WholeArrayIn copyOffsets,
                                WholeArrayIn reachOffsets,
                                WholeArrayIn reach,
                                FieldOut regular,
                                FieldOut segment);
  using ExecutionSignature = void(_1, _2, _3, _4, _5, _6, _7, _8, _9);

  template <typename FlagPortal, typename IdPortal, typename PairPortal>
  VISKORES_EXEC void operator()(viskores::Id v,
                                const FlagPortal& isFrozen,
                                const IdPortal& adjacencyOffsets,
                                const PairPortal& adjacency,
                                const IdPortal& copyOffsets,
                                const IdPortal& reachOffsets,
                                const PairPortal& reach,
                                viskores::UInt8& regular,
                                viskores::Id& segment) const
  {
    const bool frozen = isFrozen.Get(v) != 0;
    segment = frozen ? v : -1;
    const viskores::Id a = adjacencyOffsets.Get(v);
    bool ok = frozen && adjacencyOffsets.Get(v + 1) - a == 2;
    if (ok)
    {
      const viskores::Id n1 = adjacency.Get(a)[1], n2 = adjacency.Get(a + 1)[1];
      for (viskores::Id c = copyOffsets.Get(v); ok && c < copyOffsets.Get(v + 1); ++c)
      {
        const viskores::Id r = reachOffsets.Get(c);
        ok = reachOffsets.Get(c + 1) - r == 2 && reach.Get(r)[1] == n1 && reach.Get(r + 1)[1] == n2;
      }
    }
    regular = ok ? 1 : 0;
  }
};

/// The ends of half-edge `h` of `X`: edge `h / 2`, from its smaller vertex if `h` is even.
template <typename EdgePortal>
VISKORES_EXEC viskores::Id2 HalfEdgeEnds(const EdgePortal& cutEdges, viskores::Id h)
{
  const viskores::Id2 e = cutEdges.Get(h / 2);
  return (h % 2 == 0) ? e : viskores::Id2(e[1], e[0]);
}

/// The half-edge from `a` to `b`.
template <typename EdgePortal>
VISKORES_EXEC viskores::Id HalfEdge(const EdgePortal& cutEdges, viskores::Id a, viskores::Id b)
{
  const viskores::Id e =
    FindSorted(cutEdges, viskores::Id2(viskores::Min(a, b), viskores::Max(a, b)));
  return 2 * e + ((a < b) ? 0 : 1);
}

/// The successor of a half-edge along the curve: through its head when the head is regular and
/// not a root, to the other neighbour; otherwise the half-edge is terminal.
struct Successor : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn halfEdge,
                                WholeArrayIn cutEdges,
                                WholeArrayIn regular,
                                WholeArrayIn roots,
                                WholeArrayIn adjacencyOffsets,
                                WholeArrayIn adjacency,
                                FieldOut next,
                                FieldOut distance);
  using ExecutionSignature = void(_1, _2, _3, _4, _5, _6, _7, _8);

  template <typename EdgePortal, typename FlagPortal, typename IdPortal>
  VISKORES_EXEC void operator()(viskores::Id h,
                                const EdgePortal& cutEdges,
                                const FlagPortal& regular,
                                const FlagPortal& roots,
                                const IdPortal& adjacencyOffsets,
                                const EdgePortal& adjacency,
                                viskores::Id& next,
                                viskores::Id& distance) const
  {
    const viskores::Id2 ends = HalfEdgeEnds(cutEdges, h);
    const viskores::Id head = ends[1];
    if (!regular.Get(head) || roots.Get(head))
    {
      next = h;
      distance = 0;
      return;
    }
    const viskores::Id a = adjacencyOffsets.Get(head);
    const viskores::Id n1 = adjacency.Get(a)[1], n2 = adjacency.Get(a + 1)[1];
    next = HalfEdge(cutEdges, head, (n1 == ends[0]) ? n2 : n1);
    distance = 1;
  }
};

/// Marks the head of the terminal of a half-edge when it is regular, which happens only on a
/// cycle of regular vertices.
struct OnCycle : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn next,
                                WholeArrayIn cutEdges,
                                WholeArrayIn regular,
                                WholeArrayOut cycleVertex);
  using ExecutionSignature = void(_1, _2, _3, _4);

  template <typename EdgePortal, typename FlagPortal, typename OutPortal>
  VISKORES_EXEC void operator()(viskores::Id next,
                                const EdgePortal& cutEdges,
                                const FlagPortal& regular,
                                OutPortal& cycleVertex) const
  {
    // A terminal has a head that is not regular; the head of a cycle half-edge is.
    const viskores::Id head = HalfEdgeEnds(cutEdges, next)[1];
    if (regular.Get(head))
    {
      cycleVertex.Set(head, 1);
    }
  }
};

/// The edges of `X` between two cycle vertices; loops at 0 otherwise.
struct CycleEdge : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn edge, WholeArrayIn cycleVertex, FieldOut graphEdge);
  using ExecutionSignature = _3(_1, _2);

  template <typename FlagPortal>
  VISKORES_EXEC viskores::Id2 operator()(const viskores::Id2& e,
                                         const FlagPortal& cycleVertex) const
  {
    return (cycleVertex.Get(e[0]) && cycleVertex.Get(e[1])) ? e : viskores::Id2(0, 0);
  }
};

struct IsCycleRoot : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn root, FieldIn cycleVertex, FieldOut isRoot);
  using ExecutionSignature = _3(_1, _2, WorkIndex);

  VISKORES_EXEC viskores::UInt8 operator()(viskores::Id root,
                                           viskores::UInt8 cycle,
                                           viskores::Id v) const
  {
    return (cycle && root == v) ? 1 : 0;
  }
};

/// For a half-edge from a junction to a regular vertex: the arc it starts, `(j, w, terminal,
/// length)`, if it is the smaller of the two starts of its arc; `(-1, ...)` otherwise.
struct OpenStart : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn halfEdge,
                                FieldIn next,
                                FieldIn distance,
                                WholeArrayIn cutEdges,
                                WholeArrayIn isFrozen,
                                WholeArrayIn regular,
                                FieldOut start);
  using ExecutionSignature = void(_1, _2, _3, _4, _5, _6, _7);

  template <typename EdgePortal, typename FlagPortal>
  VISKORES_EXEC void operator()(viskores::Id h,
                                viskores::Id next,
                                viskores::Id distance,
                                const EdgePortal& cutEdges,
                                const FlagPortal& isFrozen,
                                const FlagPortal& regular,
                                viskores::Vec<viskores::Id, 4>& start) const
  {
    start = viskores::Vec<viskores::Id, 4>(-1);
    const viskores::Id2 ends = HalfEdgeEnds(cutEdges, h);
    if (!isFrozen.Get(ends[0]) || regular.Get(ends[0]) || !regular.Get(ends[1]))
    {
      return;
    }
    // The terminal enters the other junction from the last interior vertex.
    const viskores::Id2 last = HalfEdgeEnds(cutEdges, next);
    if (ends < viskores::Id2(last[1], last[0]))
    {
      start = viskores::Vec<viskores::Id, 4>(ends[0], ends[1], next, distance);
    }
  }
};

/// For the root of a cycle: the arc it starts, `(r, -1, terminal, length)`, from the half-edge to
/// its smaller neighbour.
struct ClosedStart : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn root,
                                WholeArrayIn cutEdges,
                                WholeArrayIn adjacencyOffsets,
                                WholeArrayIn adjacency,
                                WholeArrayIn next,
                                WholeArrayIn distance,
                                FieldOut start);
  using ExecutionSignature = void(_1, _2, _3, _4, _5, _6, _7);

  template <typename EdgePortal, typename IdPortal>
  VISKORES_EXEC void operator()(viskores::Id r,
                                const EdgePortal& cutEdges,
                                const IdPortal& adjacencyOffsets,
                                const EdgePortal& adjacency,
                                const IdPortal& next,
                                const IdPortal& distance,
                                viskores::Vec<viskores::Id, 4>& start) const
  {
    const viskores::Id h = HalfEdge(cutEdges, r, adjacency.Get(adjacencyOffsets.Get(r))[1]);
    start = viskores::Vec<viskores::Id, 4>(r, -1, next.Get(h), distance.Get(h) + 1);
  }
};

/// Per arc: its length, its ends, and its terminal paired with it, from its start; open arcs
/// come first.
struct ArcTable : viskores::worklet::WorkletMapField
{
  using ControlSignature =
    void(FieldIn start, WholeArrayIn cutEdges, FieldOut length, FieldOut ends, FieldOut terminal);
  using ExecutionSignature = void(_1, _2, _3, _4, _5, WorkIndex);

  template <typename EdgePortal>
  VISKORES_EXEC void operator()(const viskores::Vec<viskores::Id, 4>& start,
                                const EdgePortal& cutEdges,
                                viskores::Id& length,
                                viskores::Id2& ends,
                                viskores::Id2& terminal,
                                viskores::Id arc) const
  {
    length = start[3];
    const bool closed = start[1] < 0;
    ends =
      closed ? viskores::Id2(-1, -1) : viskores::Id2(start[0], HalfEdgeEnds(cutEdges, start[2])[1]);
    terminal = viskores::Id2(start[2], arc);
  }
};

/// The root of a closed arc at its first position.
struct PlaceRoot : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn start, FieldIn offset, WholeArrayOut vertices);
  using ExecutionSignature = void(_1, _2, _3);

  template <typename OutPortal>
  VISKORES_EXEC void operator()(const viskores::Vec<viskores::Id, 4>& start,
                                viskores::Id offset,
                                OutPortal& vertices) const
  {
    if (start[1] < 0)
    {
      vertices.Set(offset, start[0]);
    }
  }
};

/// The head of a half-edge, at its position in the arc whose terminal it reaches, if any:
/// `length - distance`, where the length of a closed arc counts its root, placed first by
/// `PlaceRoot`.
struct PlaceHead : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn halfEdge,
                                FieldIn next,
                                FieldIn distance,
                                WholeArrayIn cutEdges,
                                WholeArrayIn terminals,
                                WholeArrayIn lengths,
                                WholeArrayIn offsets,
                                WholeArrayOut vertices);
  using ExecutionSignature = void(_1, _2, _3, _4, _5, _6, _7, _8);

  template <typename EdgePortal, typename IdPortal, typename OutPortal>
  VISKORES_EXEC void operator()(viskores::Id h,
                                viskores::Id next,
                                viskores::Id distance,
                                const EdgePortal& cutEdges,
                                const EdgePortal& terminals,
                                const IdPortal& lengths,
                                const IdPortal& offsets,
                                OutPortal& vertices) const
  {
    if (distance == 0)
    {
      return;
    }
    // The terminals are sorted by half-edge: (terminal, arc).
    viskores::Id lo =
      PartitionPoint(0,
                     terminals.GetNumberOfValues(),
                     [&](viskores::Id mid) { return terminals.Get(mid)[0] < next; });
    if (lo == terminals.GetNumberOfValues() || terminals.Get(lo)[0] != next)
    {
      return;
    }
    const viskores::Id arc = terminals.Get(lo)[1];
    const viskores::Id position = lengths.Get(arc) - distance;
    vertices.Set(offsets.Get(arc) + position, HalfEdgeEnds(cutEdges, h)[1]);
  }
};

} // namespace curves_detail

/// The arcs of the singular graph, the frozen input vertices with the edges of `X` (sorted),
/// as `arcs_of` of proofs/lib/arcs.py, by list ranking: every directed edge of `X` points to the
/// next along the curve through a regular vertex, and pointer jumping gives the end and the
/// distance to it of every one. An open arc is kept from the smaller of its two starts (a
/// junction and its regular neighbour), and the arcs are ordered by it; the cycles of regular
/// vertices follow, by their smallest vertex, from which they continue to its smaller
/// neighbour.
inline SingularCurves FindSingularCurves(const viskores::cont::ArrayHandle<viskores::Id2>& cutEdges,
                                         const SurfaceMesh& mesh,
                                         const viskores::cont::ArrayHandle<viskores::Id>& original,
                                         const viskores::cont::ArrayHandle<viskores::UInt8>& frozen,
                                         viskores::Id numInputVertices)
{
  using Algorithm = viskores::cont::Algorithm;
  namespace cd = curves_detail;
  viskores::cont::Invoker invoke;
  SingularCurves curves;
  const viskores::Id n = numInputVertices;
  const viskores::Id numCopies = original.GetNumberOfValues();

  // The copies of the edges of X, sorted by the edge they copy.
  viskores::cont::ArrayHandle<viskores::Id3> copyRecords, copies;
  invoke(cd::CopiedEdge{}, mesh.Edges, original, cutEdges, curves.CutEdge, copyRecords);
  Algorithm::CopyIf(copyRecords, curves.CutEdge, copies);
  Algorithm::Sort(copies);
  invoke(cd::RecordKey{}, copies, mesh.Edges, curves.CopyKeys, curves.CopyEdges);

  // Regular vertices: frozen, two neighbours in X, every copy reaching both.
  viskores::cont::ArrayHandle<viskores::UInt8> isFrozen;
  isFrozen.AllocateAndFill(n, 0);
  invoke(MarkIfFlagged{}, frozen, original, isFrozen);
  viskores::cont::ArrayHandle<viskores::Id2> adjacency;
  invoke(BothDirections{}, cutEdges, viskores::cont::make_ArrayHandleGroupVec<2>(adjacency));
  Algorithm::Sort(adjacency);
  viskores::cont::ArrayHandle<viskores::Id> adjacencyFirst, adjacencyOffsets;
  invoke(FirstComponent{}, adjacency, adjacencyFirst);
  Algorithm::LowerBounds(adjacencyFirst, viskores::cont::ArrayHandleIndex(n + 1), adjacencyOffsets);
  viskores::cont::ArrayHandle<viskores::Id2> reachAll, reach;
  invoke(cd::CopyReach{},
         mesh.Edges,
         curves.CutEdge,
         original,
         viskores::cont::make_ArrayHandleGroupVec<2>(reachAll));
  viskores::cont::ArrayHandle<viskores::UInt8> validReach;
  invoke(IsValidRecord{}, reachAll, validReach);
  Algorithm::CopyIf(reachAll, validReach, reach);
  Algorithm::Sort(reach);
  viskores::cont::ArrayHandle<viskores::Id> reachFirst, reachOffsets, copyOffsets;
  invoke(FirstComponent{}, reach, reachFirst);
  Algorithm::LowerBounds(reachFirst, viskores::cont::ArrayHandleIndex(numCopies + 1), reachOffsets);
  Algorithm::LowerBounds(original, viskores::cont::ArrayHandleIndex(n + 1), copyOffsets);
  invoke(cd::RegularVertex{},
         viskores::cont::ArrayHandleIndex(n),
         isFrozen,
         adjacencyOffsets,
         adjacency,
         copyOffsets,
         reachOffsets,
         reach,
         curves.Regular,
         curves.Segment);

  // The half-edges of X ranked along the curves; first without roots, which leaves the cycles
  // unended, then with the smallest vertex of every cycle as a root.
  const viskores::Id numHalfEdges = 2 * cutEdges.GetNumberOfValues();
  viskores::cont::ArrayHandle<viskores::UInt8> roots;
  roots.AllocateAndFill(n, 0);
  viskores::cont::ArrayHandle<viskores::Id> next, distance;
  invoke(cd::Successor{},
         viskores::cont::ArrayHandleIndex(numHalfEdges),
         cutEdges,
         curves.Regular,
         roots,
         adjacencyOffsets,
         adjacency,
         next,
         distance);
  RankLists(next, distance);
  viskores::cont::ArrayHandle<viskores::UInt8> cycleVertex;
  cycleVertex.AllocateAndFill(n, 0);
  invoke(cd::OnCycle{}, next, cutEdges, curves.Regular, cycleVertex);
  viskores::cont::ArrayHandle<viskores::Id2> cycleEdges;
  invoke(cd::CycleEdge{}, cutEdges, cycleVertex, cycleEdges);
  const viskores::cont::ArrayHandle<viskores::Id> cycleRoots = SmallestRoots(n, cycleEdges);
  invoke(cd::IsCycleRoot{}, cycleRoots, cycleVertex, roots);
  invoke(cd::Successor{},
         viskores::cont::ArrayHandleIndex(numHalfEdges),
         cutEdges,
         curves.Regular,
         roots,
         adjacencyOffsets,
         adjacency,
         next,
         distance);
  RankLists(next, distance);

  // The arcs: open ones by their smaller start, then closed ones by their root.
  viskores::cont::ArrayHandle<viskores::Vec<viskores::Id, 4>> openAll, openStarts, closedStarts;
  invoke(cd::OpenStart{},
         viskores::cont::ArrayHandleIndex(numHalfEdges),
         next,
         distance,
         cutEdges,
         isFrozen,
         curves.Regular,
         openAll);
  viskores::cont::ArrayHandle<viskores::UInt8> validStart;
  invoke(IsValidRecord{}, openAll, validStart);
  Algorithm::CopyIf(openAll, validStart, openStarts);
  Algorithm::Sort(openStarts);
  viskores::cont::ArrayHandle<viskores::Id> rootVertices;
  Algorithm::CopyIf(viskores::cont::ArrayHandleIndex(n), roots, rootVertices);
  invoke(cd::ClosedStart{},
         rootVertices,
         cutEdges,
         adjacencyOffsets,
         adjacency,
         next,
         distance,
         closedStarts);
  viskores::cont::ArrayHandle<viskores::Vec<viskores::Id, 4>> starts;
  Algorithm::Copy(viskores::cont::make_ArrayHandleConcatenate(openStarts, closedStarts), starts);
  curves.NumberOfArcs = starts.GetNumberOfValues();
  viskores::cont::ArrayHandle<viskores::Id> lengths;
  viskores::cont::ArrayHandle<viskores::Id2> terminals;
  invoke(cd::ArcTable{}, starts, cutEdges, lengths, curves.ArcEnds, terminals);
  Algorithm::ScanExtended(lengths, curves.ArcOffsets);
  const viskores::Id total = viskores::cont::ArrayGetValue(curves.NumberOfArcs, curves.ArcOffsets);
  curves.ArcVertices.AllocateAndFill(total, -1);
  invoke(cd::PlaceRoot{},
         starts,
         viskores::cont::make_ArrayHandleView(curves.ArcOffsets, 0, curves.NumberOfArcs),
         curves.ArcVertices);
  Algorithm::Sort(terminals);
  invoke(cd::PlaceHead{},
         viskores::cont::ArrayHandleIndex(numHalfEdges),
         next,
         distance,
         cutEdges,
         terminals,
         lengths,
         curves.ArcOffsets,
         curves.ArcVertices);

  // One segment per vertex.
  curves.SegmentHead.AllocateAndFill(total, 1);
  Algorithm::Copy(viskores::cont::make_ArrayHandleView(curves.ArcOffsets, 0, curves.NumberOfArcs),
                  curves.FirstHead);
  curves.Absorbed.AllocateAndFill(curves.NumberOfArcs, viskores::Id2(0, 0));
  return curves;
}

namespace curves_detail
{

/// Whether an input vertex is a junction: frozen, and not a regular arc vertex.
struct IsJunction : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn segment, FieldIn regular, FieldOut junction);
  using ExecutionSignature = _3(_1, _2);

  VISKORES_EXEC viskores::Id operator()(viskores::Id segment, viskores::UInt8 regular) const
  {
    return (segment >= 0 && !regular) ? 1 : 0;
  }
};

} // namespace curves_detail

/// The number of junctions of the curves as found.
inline viskores::Id NumberOfJunctions(const SingularCurves& curves)
{
  viskores::cont::ArrayHandle<viskores::Id> junction;
  viskores::cont::Invoker invoke;
  invoke(curves_detail::IsJunction{}, curves.Segment, curves.Regular, junction);
  return viskores::cont::Algorithm::Reduce(junction, viskores::Id(0));
}

/// The smallest frozen member of every label, or -1 for a label without one, as
/// `frozen_first` of proofs/lib/arcs.py. A cluster of a sheet merged into a piece or a junction
/// copy keeps the frozen label, and its anchor may then not be frozen.
inline viskores::cont::ArrayHandle<viskores::Id> FrozenFirst(
  const viskores::cont::ArrayHandle<viskores::UInt8>& frozenVertices,
  const Clustering& clustering)
{
  using Algorithm = viskores::cont::Algorithm;
  viskores::cont::Invoker invoke;
  viskores::cont::ArrayHandle<viskores::Id> frozenVertexIds, labels, uniqueLabels, smallest, first;
  Algorithm::CopyIf(viskores::cont::ArrayHandleIndex(frozenVertices.GetNumberOfValues()),
                    frozenVertices,
                    frozenVertexIds);
  invoke(GatherValue{}, frozenVertexIds, clustering.Labels, labels);
  Algorithm::SortByKey(labels, frozenVertexIds);
  Algorithm::ReduceByKey(labels, frozenVertexIds, uniqueLabels, smallest, viskores::Minimum());
  first.AllocateAndFill(clustering.NumberOfClusters, -1);
  invoke(ScatterValue{}, uniqueLabels, smallest, first);
  return first;
}

namespace curves_detail
{

/// A piece holding sheet vertices, placed from its frozen members alone: its generator and
/// quadric replace those of `Place`. Every other label keeps them.
struct ScatterPiece : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn label,
                                FieldIn generator,
                                FieldIn q,
                                FieldIn frozenSize,
                                WholeArrayIn sizes,
                                WholeArrayInOut generators,
                                WholeArrayInOut quadrics);
  using ExecutionSignature = void(_1, _2, _3, _4, _5, _6, _7);

  template <typename IdPortal, typename VecPortal, typename MatPortal>
  VISKORES_EXEC void operator()(viskores::Id label,
                                const Vec3& generator,
                                const Mat3& q,
                                viskores::Id frozenSize,
                                const IdPortal& sizes,
                                VecPortal& generators,
                                MatPortal& quadrics) const
  {
    if (sizes.Get(label) > frozenSize)
    {
      generators.Set(label, generator);
      quadrics.Set(label, q);
    }
  }
};

} // namespace curves_detail

/// The generators and quadrics of the labels with every piece placed as if no sheet vertex had
/// joined it (`piece_place="frozen"` of proofs/lib/arcs.py): a label holding frozen and other
/// vertices gets those of `Place` over its frozen members alone, every other label keeps
/// `generators` and `quadrics`, those of `Place`.
inline void FrozenPiecePlacement(const SurfaceMesh& mesh,
                                 const viskores::cont::ArrayHandle<Mat3>& vertexMatrices,
                                 const viskores::cont::ArrayHandle<viskores::UInt8>& frozenVertices,
                                 const viskores::cont::ArrayHandle<viskores::Id>& labels,
                                 const viskores::cont::ArrayHandle<viskores::Id>& sizes,
                                 const viskores::cont::ArrayHandle<Vec3>& generators,
                                 const viskores::cont::ArrayHandle<Mat3>& quadrics,
                                 viskores::cont::ArrayHandle<Vec3>& pieceGenerators,
                                 viskores::cont::ArrayHandle<Mat3>& pieceQuadrics)
{
  using Algorithm = viskores::cont::Algorithm;
  viskores::cont::Invoker invoke;
  Algorithm::Copy(generators, pieceGenerators);
  Algorithm::Copy(quadrics, pieceQuadrics);
  viskores::cont::ArrayHandle<viskores::Id> frozenVertexIds, frozenLabels;
  Algorithm::CopyIf(viskores::cont::ArrayHandleIndex(frozenVertices.GetNumberOfValues()),
                    frozenVertices,
                    frozenVertexIds);
  if (frozenVertexIds.GetNumberOfValues() == 0)
  {
    return;
  }
  invoke(GatherValue{}, frozenVertexIds, labels, frozenLabels);
  viskores::worklet::Keys<viskores::Id> pieces;
  pieces.BuildArrays(frozenLabels, viskores::worklet::KeysSortType::Stable);
  viskores::cont::ArrayHandle<viskores::Id> anchors, frozenSizes;
  viskores::cont::ArrayHandle<Mat3> q;
  viskores::cont::ArrayHandle<Vec3> b, generator;
  viskores::cont::ArrayHandle<Real> s;
  invoke(detail::Place{},
         pieces,
         frozenVertexIds,
         mesh.Points,
         vertexMatrices,
         anchors,
         q,
         b,
         s,
         generator,
         frozenSizes);
  invoke(curves_detail::ScatterPiece{},
         pieces.GetUniqueKeys(),
         generator,
         q,
         frozenSizes,
         sizes,
         pieceGenerators,
         pieceQuadrics);
}

/// The first copy of every input vertex, for an input cut into surfaces whose copies are sorted
/// by input vertex.
inline viskores::cont::ArrayHandle<viskores::Id> FirstCopies(
  const viskores::cont::ArrayHandle<viskores::Id>& original,
  viskores::Id numInputVertices)
{
  viskores::cont::ArrayHandle<viskores::Id> offsets;
  viskores::cont::Algorithm::LowerBounds(
    original, viskores::cont::ArrayHandleIndex(numInputVertices), offsets);
  return offsets;
}

namespace curves_detail
{

/// How a glue group is placed, by `GroupPlacement` and `CandidateGroupPlacement`. Consumers test
/// only whether the generators are kept and whether the group is at a junction.
enum : viskores::IdComponent
{
  KeepGenerators = 0,
  AtJunction = 1,
  AtMinimiser = 3
};

/// For a glue group, its labels in increasing order, as `place` of proofs/lib/arcs.py with the
/// pieces placed as if no sheet vertex had joined them (`pieceGenerators`, `pieceQuadrics`):
/// `KeepGenerators` keeps the generator of a label with no frozen member, `AtJunction` places a
/// junction at itself, `AtMinimiser` places the group at the minimiser of the sum of the costs of
/// its labels, `pieceGenerators` for a single label.
struct GroupPlacement : viskores::worklet::WorkletReduceByKey
{
  using ControlSignature = void(KeysIn group,
                                ValuesIn label,
                                WholeArrayIn frozenFirst,
                                WholeArrayIn original,
                                WholeArrayIn segment,
                                WholeArrayIn regular,
                                WholeArrayIn copyOffsets,
                                WholeArrayIn points,
                                WholeArrayIn pieceQuadrics,
                                WholeArrayIn pieceGenerators,
                                ReducedValuesOut kind,
                                ReducedValuesOut generator);
  using ExecutionSignature = void(_2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12);

  template <typename LabelVec,
            typename IdPortal,
            typename FlagPortal,
            typename VecPortal,
            typename MatPortal>
  VISKORES_EXEC void operator()(const LabelVec& labels,
                                const IdPortal& frozenFirst,
                                const IdPortal& original,
                                const IdPortal& segment,
                                const FlagPortal& regular,
                                const IdPortal& copyOffsets,
                                const VecPortal& points,
                                const MatPortal& pieceQuadrics,
                                const VecPortal& pieceGenerators,
                                viskores::IdComponent& kind,
                                Vec3& generator) const
  {
    const viskores::IdComponent count = labels.GetNumberOfComponents();
    const viskores::Id first = frozenFirst.Get(labels[0]);
    kind = KeepGenerators;
    generator = pieceGenerators.Get(labels[0]);
    if (first < 0)
    {
      return;
    }
    const viskores::Id s = segment.Get(original.Get(first));
    if (!regular.Get(s))
    {
      // A junction stays at itself: at any of its copies.
      kind = AtJunction;
      generator = points.Get(copyOffsets.Get(s));
      return;
    }
    kind = AtMinimiser;
    if (count < 2)
    {
      return;
    }
    const Vec3 o = pieceGenerators.Get(labels[0]);
    TreeSum<Mat3> qs;
    TreeSum<Vec3> bs;
    for (viskores::IdComponent i = 0; i < count; ++i)
    {
      qs.Push(pieceQuadrics.Get(labels[i]));
      bs.Push(MatVec(pieceQuadrics.Get(labels[i]), Sub(pieceGenerators.Get(labels[i]), o)));
    }
    generator = Add(o, SolveSPD(qs.Result(), bs.Result()));
  }
};

/// The generator of a label from its group: kept, or the group's.
struct GroupGenerator : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn glue,
                                WholeArrayIn groups,
                                WholeArrayIn kinds,
                                WholeArrayIn groupGenerators,
                                FieldInOut generator);
  using ExecutionSignature = void(_1, _2, _3, _4, _5);

  template <typename IdPortal, typename KindPortal, typename VecPortal>
  VISKORES_EXEC void operator()(viskores::Id glue,
                                const IdPortal& groups,
                                const KindPortal& kinds,
                                const VecPortal& groupGenerators,
                                Vec3& generator) const
  {
    const viskores::Id g = FindSorted(groups, glue);
    if (kinds.Get(g) != KeepGenerators)
    {
      generator = groupGenerators.Get(g);
    }
  }
};

} // namespace curves_detail

/// The generators of the vertices of `K` on the curves, as `place` of proofs/lib/arcs.py
/// (Proposition "Decimating the singular curves", (e)). The labels that `glue` joins share one
/// generator: at the input position of the junction, for the copies of a junction and the
/// segments merged into it; and otherwise the minimiser of the sum of their costs,
/// `c = o + (sum Q_l)^-1 sum Q_l (c_l - o)` with `o` the generator of the smallest label, summed
/// in increasing label order, with every piece placed as if no sheet vertex had joined it
/// (`FrozenPiecePlacement`), so that a sheet vertex joining a piece moves no generator. The
/// copies of one vertex, at its position, stay there.
inline void PlaceCurveGenerators(const SurfaceMesh& mesh,
                                 const viskores::cont::ArrayHandle<Mat3>& vertexMatrices,
                                 const SingularCurves& curves,
                                 const viskores::cont::ArrayHandle<viskores::UInt8>& frozenVertices,
                                 const viskores::cont::ArrayHandle<viskores::Id>& original,
                                 const viskores::cont::ArrayHandle<viskores::Id>& glue,
                                 Clustering& clustering)
{
  namespace cd = curves_detail;
  viskores::cont::Invoker invoke;
  const viskores::Id k = clustering.NumberOfClusters;
  const viskores::Id numInput = curves.Segment.GetNumberOfValues();
  const viskores::cont::ArrayHandle<viskores::Id> frozenFirst =
    FrozenFirst(frozenVertices, clustering);
  viskores::cont::ArrayHandle<Vec3> pieceGenerators;
  viskores::cont::ArrayHandle<Mat3> pieceQuadrics;
  FrozenPiecePlacement(mesh,
                       vertexMatrices,
                       frozenVertices,
                       clustering.Labels,
                       clustering.Sizes,
                       clustering.Generators,
                       clustering.Q,
                       pieceGenerators,
                       pieceQuadrics);
  const auto copyOffsets = FirstCopies(original, numInput);

  // Per group, in increasing label order.
  viskores::worklet::Keys<viskores::Id> groups;
  groups.BuildArrays(glue, viskores::worklet::KeysSortType::Stable);
  viskores::cont::ArrayHandle<viskores::IdComponent> kinds;
  viskores::cont::ArrayHandle<Vec3> groupGenerators;
  invoke(cd::GroupPlacement{},
         groups,
         viskores::cont::ArrayHandleIndex(k),
         frozenFirst,
         original,
         curves.Segment,
         curves.Regular,
         copyOffsets,
         mesh.Points,
         pieceQuadrics,
         pieceGenerators,
         kinds,
         groupGenerators);
  invoke(cd::GroupGenerator{},
         glue,
         groups.GetUniqueKeys(),
         kinds,
         groupGenerators,
         clustering.Generators);
}

namespace curves_detail
{

/// The segment of a label through its smallest frozen member, or -1.
struct LabelSegment : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn frozenFirst,
                                WholeArrayIn original,
                                WholeArrayIn segment,
                                FieldOut labelSegment);
  using ExecutionSignature = _4(_1, _2, _3);

  template <typename IdPortal>
  VISKORES_EXEC viskores::Id operator()(viskores::Id first,
                                        const IdPortal& original,
                                        const IdPortal& segment) const
  {
    return (first >= 0) ? segment.Get(original.Get(first)) : viskores::Id(-1);
  }
};

/// The glued label: the smallest label of the segment, or the label itself.
struct GlueOfLabel : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn labelSegment,
                                WholeArrayIn segments,
                                WholeArrayIn smallest,
                                FieldOut glue);
  using ExecutionSignature = _4(_1, _2, _3, WorkIndex);

  template <typename IdPortal>
  VISKORES_EXEC viskores::Id operator()(viskores::Id s,
                                        const IdPortal& segments,
                                        const IdPortal& smallest,
                                        viskores::Id label) const
  {
    if (s < 0)
    {
      return label;
    }
    return smallest.Get(FindSorted(segments, s));
  }
};

} // namespace curves_detail

/// The vertex of `K` of every label, as `glue_ids` of proofs/lib/arcs.py: for a label with a
/// frozen member, the smallest label whose smallest frozen member is in the same segment, which
/// joins the pieces of a segment and the copies of a junction; every other label stays.
inline viskores::cont::ArrayHandle<viskores::Id> SegmentGlueMap(
  const SingularCurves& curves,
  const viskores::cont::ArrayHandle<viskores::UInt8>& frozenVertices,
  const viskores::cont::ArrayHandle<viskores::Id>& original,
  const Clustering& clustering)
{
  using Algorithm = viskores::cont::Algorithm;
  viskores::cont::Invoker invoke;
  viskores::cont::ArrayHandle<viskores::Id> labelSegment, keys, labels, segments, smallest, glue;
  invoke(curves_detail::LabelSegment{},
         FrozenFirst(frozenVertices, clustering),
         original,
         curves.Segment,
         labelSegment);
  Algorithm::CopyIf(labelSegment, labelSegment, keys, IsNonNegative{});
  Algorithm::CopyIf(viskores::cont::ArrayHandleIndex(clustering.NumberOfClusters),
                    labelSegment,
                    labels,
                    IsNonNegative{});
  Algorithm::SortByKey(keys, labels);
  Algorithm::ReduceByKey(keys, labels, segments, smallest, viskores::Minimum());
  invoke(curves_detail::GlueOfLabel{}, labelSegment, segments, smallest, glue);
  return glue;
}

namespace curves_detail
{

/// For an edge of the cut mesh between two clusters that is a copy of an edge of `X`: the two
/// labels and the two input vertices, each pair sorted; otherwise `(-1, -1, -1, -1)`.
struct CutEdgeStep : viskores::worklet::WorkletMapField
{
  using ControlSignature =
    void(FieldIn edge, FieldIn cutEdge, WholeArrayIn labels, WholeArrayIn original, FieldOut step);
  using ExecutionSignature = _5(_1, _2, _3, _4);

  template <typename IdPortal>
  VISKORES_EXEC viskores::Vec<viskores::Id, 4> operator()(const viskores::Id2& edge,
                                                          viskores::UInt8 cutEdge,
                                                          const IdPortal& labels,
                                                          const IdPortal& original) const
  {
    const viskores::Id la = labels.Get(edge[0]), lb = labels.Get(edge[1]);
    if (!cutEdge || la == lb)
    {
      return viskores::Vec<viskores::Id, 4>(-1);
    }
    const viskores::Id oa = original.Get(edge[0]), ob = original.Get(edge[1]);
    return viskores::Vec<viskores::Id, 4>(
      viskores::Min(la, lb), viskores::Max(la, lb), viskores::Min(oa, ob), viskores::Max(oa, ob));
  }
};

struct StepKey : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn step, FieldOut key, FieldOut realisedBy);
  using ExecutionSignature = void(_1, _2, _3);

  VISKORES_EXEC void operator()(const viskores::Vec<viskores::Id, 4>& step,
                                viskores::Id2& key,
                                viskores::Id2& realisedBy) const
  {
    key = viskores::Id2(step[0], step[1]);
    realisedBy = viskores::Id2(step[2], step[3]);
  }
};

/// For an edge of `K_X`: its glued edge, and the edges of `X` that realise it, as a count
/// (two meaning two or more) and the one edge when there is exactly one.
struct GlueEdge : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn edge,
                                WholeArrayIn glue,
                                WholeArrayIn stepKeys,
                                WholeArrayIn stepRealised,
                                FieldOut glued,
                                FieldOut count,
                                FieldOut realisedBy);
  using ExecutionSignature = void(_1, _2, _3, _4, _5, _6, _7);

  template <typename IdPortal, typename Id2Portal>
  VISKORES_EXEC void operator()(const viskores::Id2& edge,
                                const IdPortal& glue,
                                const Id2Portal& stepKeys,
                                const Id2Portal& stepRealised,
                                viskores::Id2& glued,
                                viskores::IdComponent& count,
                                viskores::Id2& realisedBy) const
  {
    const viskores::Id ga = glue.Get(edge[0]), gb = glue.Get(edge[1]);
    glued = viskores::Id2(viskores::Min(ga, gb), viskores::Max(ga, gb));
    // The steps are sorted: find the first with this key.
    viskores::Id lo = LowerBound(stepKeys, 0, stepKeys.GetNumberOfValues(), edge);
    count = 0;
    realisedBy = viskores::Id2(-1, -1);
    for (viskores::Id s = lo;
         s < stepKeys.GetNumberOfValues() && stepKeys.Get(s) == edge && count < 2;
         ++s)
    {
      ++count;
      realisedBy = stepRealised.Get(s);
    }
  }
};

/// (d') on the edges of `K_X` sorted by glued edge: an edge joining two identified vertices,
/// and every edge of a glued edge with two or more edges of `K_X` unless all are realised by one
/// and the same edge of `X` and by no other, list their labels.
struct GlueEdgeViolation : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn index,
                                WholeArrayIn edges,
                                WholeArrayIn glued,
                                WholeArrayIn counts,
                                WholeArrayIn realised,
                                FieldOut listed);
  using ExecutionSignature = void(_1, _2, _3, _4, _5, _6);

  template <typename Id2Portal, typename CountPortal, typename ListedVec>
  VISKORES_EXEC void operator()(viskores::Id i,
                                const Id2Portal& edges,
                                const Id2Portal& glued,
                                const CountPortal& counts,
                                const Id2Portal& realised,
                                ListedVec& listed) const
  {
    const viskores::Id2 e = edges.Get(i);
    const viskores::Id2 g = glued.Get(i);
    bool bad = g[0] == g[1];
    if (!bad)
    {
      viskores::Id first = i, last = i + 1;
      while (first > 0 && glued.Get(first - 1) == g)
      {
        --first;
      }
      while (last < glued.GetNumberOfValues() && glued.Get(last) == g)
      {
        ++last;
      }
      if (last - first > 1)
      {
        for (viskores::Id m = first; m < last && !bad; ++m)
        {
          bad = counts.Get(m) != 1 || realised.Get(m) != realised.Get(first);
        }
      }
    }
    listed[0] = bad ? e[0] : viskores::Id(-1);
    listed[1] = bad ? e[1] : viskores::Id(-1);
  }
};

/// (d') on the triangles of `K_X` sorted by glued triangle: two with one image list the labels
/// of both.
struct GlueTriangleViolation : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn index,
                                WholeArrayIn glued,
                                WholeArrayIn order,
                                WholeArrayIn triangles,
                                FieldOut listed);
  using ExecutionSignature = void(_1, _2, _3, _4, _5);

  template <typename Id3Portal, typename IdPortal, typename ListedVec>
  VISKORES_EXEC void operator()(viskores::Id i,
                                const Id3Portal& glued,
                                const IdPortal& order,
                                const Id3Portal& triangles,
                                ListedVec& listed) const
  {
    const viskores::Id3 g = glued.Get(i);
    const bool bad = (i > 0 && glued.Get(i - 1) == g) ||
      (i + 1 < glued.GetNumberOfValues() && glued.Get(i + 1) == g);
    const viskores::Id3 t = triangles.Get(order.Get(i));
    for (viskores::IdComponent k = 0; k < 3; ++k)
    {
      listed[k] = bad ? t[k] : viskores::Id(-1);
    }
  }
};

} // namespace curves_detail

/// The labels involved in a violation of (d') of Proposition "Decimating the singular curves",
/// as `glue_violations` of proofs/lib/arcs.py, sorted and without repetition: the
/// identification of `K_X` onto `K` given by `glue` may identify two vertices only if they are
/// pieces of one segment or copies of one junction, two edges only if both are realised by one
/// edge of `X` and by no other, and no two triangles, and no edge of `K_X` may join two vertices
/// it identifies.
inline viskores::cont::ArrayHandle<viskores::Id> GlueViolations(
  const SurfaceMesh& mesh,
  const viskores::cont::ArrayHandle<viskores::Id>& labels,
  const viskores::cont::ArrayHandle<viskores::Id>& original,
  const SingularCurves& curves,
  const viskores::cont::ArrayHandle<viskores::Id>& glue)
{
  using Algorithm = viskores::cont::Algorithm;
  viskores::cont::Invoker invoke;
  // The edges of X that realise every edge of K_X, from the cut edges between clusters.
  viskores::cont::ArrayHandle<viskores::Vec<viskores::Id, 4>> steps, valid;
  invoke(curves_detail::CutEdgeStep{}, mesh.Edges, curves.CutEdge, labels, original, steps);
  viskores::cont::ArrayHandle<viskores::UInt8> isStep;
  invoke(IsValidRecord{}, steps, isStep);
  Algorithm::CopyIf(steps, isStep, valid);
  Algorithm::Sort(valid);
  Algorithm::Unique(valid);
  viskores::cont::ArrayHandle<viskores::Id2> stepKeys, stepRealised;
  invoke(curves_detail::StepKey{}, valid, stepKeys, stepRealised);

  const viskores::cont::ArrayHandle<viskores::Id3> triangles = OutputTriangles(mesh, labels);
  viskores::cont::ArrayHandle<viskores::Id> listed;
  if (triangles.GetNumberOfValues() == 0)
  {
    return listed;
  }
  viskores::cont::ArrayHandle<viskores::Id2> edges;
  invoke(SortedTriangleEdges{}, triangles, viskores::cont::make_ArrayHandleGroupVec<3>(edges));
  Algorithm::Sort(edges);
  Algorithm::Unique(edges);
  viskores::cont::ArrayHandle<viskores::Id2> glued, realised;
  viskores::cont::ArrayHandle<viskores::IdComponent> counts;
  invoke(curves_detail::GlueEdge{}, edges, glue, stepKeys, stepRealised, glued, counts, realised);
  // Sorted by glued edge, the edges of K_X with one image are runs.
  viskores::cont::ArrayHandle<viskores::Id> order;
  Algorithm::Copy(viskores::cont::ArrayHandleIndex(edges.GetNumberOfValues()), order);
  Algorithm::SortByKey(glued, order);
  viskores::cont::ArrayHandle<viskores::Id2> sortedEdges, sortedRealised;
  viskores::cont::ArrayHandle<viskores::IdComponent> sortedCounts;
  Algorithm::Copy(viskores::cont::make_ArrayHandlePermutation(order, edges), sortedEdges);
  Algorithm::Copy(viskores::cont::make_ArrayHandlePermutation(order, counts), sortedCounts);
  Algorithm::Copy(viskores::cont::make_ArrayHandlePermutation(order, realised), sortedRealised);
  viskores::cont::ArrayHandle<viskores::Id> edgeListed;
  invoke(curves_detail::GlueEdgeViolation{},
         viskores::cont::ArrayHandleIndex(edges.GetNumberOfValues()),
         sortedEdges,
         glued,
         sortedCounts,
         sortedRealised,
         viskores::cont::make_ArrayHandleGroupVec<2>(edgeListed));
  // Two triangles of K_X with one image.
  viskores::cont::ArrayHandle<viskores::Id3> gluedTriangles;
  invoke(GatherSortedTriple{}, triangles, glue, gluedTriangles);
  viskores::cont::ArrayHandle<viskores::Id> triangleOrder, triangleListed;
  Algorithm::Copy(viskores::cont::ArrayHandleIndex(triangles.GetNumberOfValues()), triangleOrder);
  Algorithm::SortByKey(gluedTriangles, triangleOrder);
  invoke(curves_detail::GlueTriangleViolation{},
         viskores::cont::ArrayHandleIndex(triangles.GetNumberOfValues()),
         gluedTriangles,
         triangleOrder,
         triangles,
         viskores::cont::make_ArrayHandleGroupVec<3>(triangleListed));
  viskores::cont::ArrayHandle<viskores::Id> all;
  Algorithm::Copy(viskores::cont::make_ArrayHandleConcatenate(edgeListed, triangleListed), all);
  Algorithm::CopyIf(all, all, listed, IsNonNegative{});
  Algorithm::Sort(listed);
  Algorithm::Unique(listed);
  return listed;
}

}
}
} // namespace viskores::worklet::quadric_clustering

VISKORES_QUADRIC_CLUSTERING_STRICT_FP_END

#endif // viskores_worklet_quadric_clustering_SingularCurves_h
