//============================================================================
//  The contents of this file are covered by the Viskores license. See
//  LICENSE.txt for details.
//
//  By contributing to this file, all contributors agree to the Developer
//  Certificate of Origin Version 1.1 (DCO 1.1) as stated in DCO.txt.
//============================================================================
#ifndef viskores_worklet_quadric_clustering_SurfaceMesh_h
#define viskores_worklet_quadric_clustering_SurfaceMesh_h

#include <viskores/filter/geometry_refinement/worklet/quadric_clustering/Arithmetic.h>
#include <viskores/filter/geometry_refinement/worklet/quadric_clustering/QuadricClusteringCommonWorklets.h>

#include <viskores/BinaryOperators.h>
#include <viskores/cont/Algorithm.h>
#include <viskores/cont/ArrayCopy.h>
#include <viskores/cont/ArrayGetValues.h>
#include <viskores/cont/ArrayHandle.h>
#include <viskores/cont/ArrayHandleConstant.h>
#include <viskores/cont/ArrayHandleGroupVec.h>
#include <viskores/cont/ArrayHandleIndex.h>
#include <viskores/cont/ArrayHandlePermutation.h>
#include <viskores/cont/ErrorBadValue.h>
#include <viskores/cont/Invoker.h>

#include <viskores/filter/connected_components/worklet/UnionFind.h>
#include <viskores/worklet/Keys.h>
#include <viskores/worklet/WorkletMapField.h>
#include <viskores/worklet/WorkletReduceByKey.h>

#include <string>

VISKORES_QUADRIC_CLUSTERING_STRICT_FP_BEGIN

namespace viskores
{
namespace worklet
{
namespace quadric_clustering
{

/// The connectivity of a triangle complex that every step of the algorithm reads.
///
/// Edges are sorted vertex pairs in lexicographic order, so the edges of a vertex in increasing
/// edge order are its neighbours in increasing order. Corner `c` of triangle `t` is numbered
/// `c * NumberOfTriangles + t`, so the corners of a vertex in increasing order are those of
/// position 0 in triangle order, then of position 1, then of position 2. The reference
/// implementation accumulates the vertex quadrics in that order.
struct SurfaceMesh
{
  viskores::Id NumberOfPoints = 0;
  viskores::Id NumberOfTriangles = 0;
  viskores::cont::ArrayHandle<Vec3> Points;
  viskores::cont::ArrayHandle<viskores::Id3> Triangles;

  viskores::cont::ArrayHandle<viskores::Id2> Edges;
  /// The two smallest triangle indices of every edge, in increasing order; the second is -1 on
  /// a boundary edge.
  viskores::cont::ArrayHandle<viskores::Id2> EdgeTriangles;
  viskores::cont::ArrayHandle<viskores::IdComponent> EdgeTriangleCounts;

  viskores::cont::ArrayHandle<viskores::Id> VertexEdgeOffsets;
  viskores::cont::ArrayHandle<viskores::Id> VertexEdges;
  viskores::cont::ArrayHandle<viskores::Id> VertexCornerOffsets;
  viskores::cont::ArrayHandle<viskores::Id> VertexCorners;

  /// The one-ring of every vertex in fan order: a cycle at an interior vertex, a path at a
  /// boundary vertex. Only built for surfaces.
  viskores::cont::ArrayHandle<viskores::Id> RingOffsets;
  viskores::cont::ArrayHandle<viskores::Id> Rings;
  viskores::cont::ArrayHandle<viskores::UInt8> IsBoundaryVertex;
  bool HasRings = false;
  bool HasBoundary = false;
  /// The connected component of every vertex, numbered in the order of their smallest vertex,
  /// and the number of boundary vertices of every component. Only built for surfaces.
  viskores::Id NumberOfComponents = 0;
  viskores::cont::ArrayHandle<viskores::Id> Components;
  viskores::cont::ArrayHandle<viskores::Id> ComponentBoundaryVertices;
};

namespace detail
{

/// The three edges of a triangle as sorted vertex pairs, and their half-edge ids `3 t + s`.
struct HalfEdges : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn triangle, FieldOut keys, FieldOut halfEdges);
  using ExecutionSignature = void(_1, _2, _3, WorkIndex);

  template <typename KeysVec, typename HalfEdgesVec>
  VISKORES_EXEC void operator()(const viskores::Id3& triangle,
                                KeysVec& keys,
                                HalfEdgesVec& halfEdges,
                                viskores::Id t) const
  {
    for (viskores::IdComponent s = 0; s < 3; ++s)
    {
      const viskores::Id x = triangle[s];
      const viskores::Id y = triangle[(s + 1) % 3];
      keys[s] = SortedEdge(x, y);
      halfEdges[s] = 3 * t + s;
    }
  }
};

/// Per edge: its two smallest triangles, its number of triangles, and its sign: -1 if those two
/// triangles traverse it in the same direction, +1 otherwise.
struct EdgeIncidence : viskores::worklet::WorkletReduceByKey
{
  using ControlSignature = void(KeysIn keys,
                                ValuesIn halfEdges,
                                WholeArrayIn triangles,
                                ReducedValuesOut edgeTriangles,
                                ReducedValuesOut count,
                                ReducedValuesOut sign);
  using ExecutionSignature = void(_2, _3, _4, _5, _6);
  using InputDomain = _1;

  template <typename HalfEdgesVec, typename TrianglesPortal>
  VISKORES_EXEC void operator()(const HalfEdgesVec& halfEdges,
                                const TrianglesPortal& triangles,
                                viskores::Id2& edgeTriangles,
                                viskores::IdComponent& count,
                                viskores::Int8& sign) const
  {
    count = halfEdges.GetNumberOfComponents();
    viskores::Id first = -1;
    viskores::Id second = -1;
    for (viskores::IdComponent i = 0; i < count; ++i)
    {
      const viskores::Id halfEdge = halfEdges[i];
      if (first < 0 || halfEdge / 3 < first / 3)
      {
        second = first;
        first = halfEdge;
      }
      else if (second < 0 || halfEdge / 3 < second / 3)
      {
        second = halfEdge;
      }
    }
    edgeTriangles = viskores::Id2(first / 3, (second < 0) ? -1 : second / 3);
    sign = 1;
    if (second >= 0)
    {
      auto forward = [&](viskores::Id halfEdge)
      {
        const viskores::Id3 tri = triangles.Get(halfEdge / 3);
        const viskores::IdComponent s = static_cast<viskores::IdComponent>(halfEdge % 3);
        return tri[s] < tri[(s + 1) % 3];
      };
      sign = (forward(first) == forward(second)) ? -1 : 1;
    }
  }
};

struct NegativeSign : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn sign, FieldOut negative);
  using ExecutionSignature = _2(_1);

  VISKORES_EXEC viskores::Id operator()(viskores::Int8 sign) const { return (sign < 0) ? 1 : 0; }
};

/// The two ends of every edge, as (vertex, edge) records `2 e` and `2 e + 1`.
struct EdgeEnds : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn edge, FieldOut vertices, FieldOut edgeIds);
  using ExecutionSignature = void(_1, _2, _3, WorkIndex);

  template <typename VertexVec, typename EdgeVec>
  VISKORES_EXEC void operator()(const viskores::Id2& edge,
                                VertexVec& vertices,
                                EdgeVec& edgeIds,
                                viskores::Id e) const
  {
    vertices[0] = edge[0];
    vertices[1] = edge[1];
    edgeIds[0] = e;
    edgeIds[1] = e;
  }
};

/// At every corner of a triangle `(a, b, c)`, the directed pair of the opposite edge: at `a`
/// the pair `b -> c`, at `b` the pair `c -> a`, at `c` the pair `a -> b`. Corner `s` of
/// triangle `t` is stored at `s * numTriangles + t`.
struct CornerPairs : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn triangle,
                                WholeArrayOut vertex,
                                WholeArrayOut from,
                                WholeArrayOut to);
  using ExecutionSignature = void(_1, _2, _3, _4, WorkIndex);

  VISKORES_CONT explicit CornerPairs(viskores::Id numTriangles)
    : NumberOfTriangles(numTriangles)
  {
  }

  template <typename IdPortal>
  VISKORES_EXEC void operator()(const viskores::Id3& triangle,
                                IdPortal& vertex,
                                IdPortal& from,
                                IdPortal& to,
                                viskores::Id t) const
  {
    for (viskores::IdComponent s = 0; s < 3; ++s)
    {
      const viskores::Id k = s * this->NumberOfTriangles + t;
      vertex.Set(k, triangle[s]);
      from.Set(k, triangle[(s + 1) % 3]);
      to.Set(k, triangle[(s + 2) % 3]);
    }
  }

  viskores::Id NumberOfTriangles;
};

/// Walks the one-ring of a vertex. Its star must be consistently oriented (no source and no
/// target repeats among its successor pairs `b -> c` of the triangles `(v, b, c)`), and the pairs
/// must form a single cycle, or a single path that starts at the one neighbour that is no
/// pair's target. The ring has as many vertices as the link. With `Write` set, it is stored.
template <bool Write>
struct RingWalk : viskores::worklet::WorkletReduceByKey
{
  using ControlSignature = void(KeysIn keys,
                                ValuesIn from,
                                ValuesIn to,
                                WholeArrayIn ringOffsets,
                                WholeArrayOut rings,
                                ReducedValuesOut ringSize,
                                ReducedValuesOut boundary,
                                ReducedValuesOut valid);
  using ExecutionSignature = void(_1, _2, _3, _4, _5, _6, _7, _8);
  using InputDomain = _1;

  template <typename IdVec, typename OffsetsPortal, typename RingsPortal>
  VISKORES_EXEC void operator()(viskores::Id vertex,
                                const IdVec& from,
                                const IdVec& to,
                                const OffsetsPortal& ringOffsets,
                                RingsPortal& rings,
                                viskores::Id& ringSize,
                                viskores::UInt8& boundary,
                                viskores::UInt8& valid) const
  {
    const viskores::IdComponent degree = from.GetNumberOfComponents();
    valid = 1;
    boundary = 0;
    ringSize = degree;
    bool repeated = false;
    for (viskores::IdComponent i = 0; i < degree; ++i)
    {
      for (viskores::IdComponent j = i + 1; j < degree; ++j)
      {
        if (from[j] == from[i] || to[j] == to[i])
        {
          repeated = true;
        }
      }
    }
    viskores::Id base = 0;
    if (Write)
    {
      base = ringOffsets.Get(vertex);
    }
    if (repeated)
    {
      valid = 0; // two triangles with the same pair: not a surface around v
      return;
    }
    viskores::IdComponent head = -1;
    for (viskores::IdComponent i = 0; i < degree; ++i)
    {
      bool isTarget = false;
      for (viskores::IdComponent j = 0; j < degree; ++j)
      {
        if (to[j] == from[i])
        {
          isTarget = true;
        }
      }
      if (!isTarget)
      {
        if (head >= 0)
        {
          valid = 0; // several fans: a pinched vertex
        }
        head = i;
      }
    }
    if (!valid)
    {
      return;
    }
    viskores::IdComponent count = 0;
    viskores::IdComponent current = (head >= 0) ? head : 0;
    const viskores::Id start = from[current];
    while (true)
    {
      if (count >= degree)
      {
        valid = 0;
        return;
      }
      if (Write)
      {
        rings.Set(base + count, from[current]);
      }
      ++count;
      const viskores::Id next = to[current];
      if (next == start)
      {
        break;
      }
      viskores::IdComponent successor = -1;
      for (viskores::IdComponent j = 0; j < degree; ++j)
      {
        if (from[j] == next)
        {
          successor = j;
          break;
        }
      }
      if (successor < 0)
      {
        // The end of a path, which is only allowed when it started at the head.
        if (head < 0)
        {
          valid = 0;
          return;
        }
        if (Write)
        {
          rings.Set(base + count, next);
        }
        boundary = 1;
        break;
      }
      current = successor;
    }
    if (count != degree)
    {
      valid = 0; // more than one cycle
      return;
    }
    ringSize = degree + boundary;
  }
};

/// Unites the ends of every edge whose ends carry the same label.
struct UniteEdges : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn edge, WholeArrayIn labels, AtomicArrayInOut parents);
  using ExecutionSignature = void(_1, _2, _3);

  template <typename LabelsPortal, typename Parents>
  VISKORES_EXEC void operator()(const viskores::Id2& edge,
                                const LabelsPortal& labels,
                                Parents& parents) const
  {
    if (labels.Get(edge[0]) == labels.Get(edge[1]))
    {
      viskores::worklet::connectivity::UnionFind::Unite(parents, edge[0], edge[1]);
    }
  }
};

} // namespace detail

/// The connected components of the graph of the edges whose ends carry the same label, numbered
/// in the order of their smallest vertex: the union-find of the connected-components filters
/// links by index, so the root of every component is its smallest vertex, whatever the order of
/// the unions. Returns the number of components.
/// `roots`, when given, receives the smallest vertex of the component of every vertex.
template <typename LabelsArray>
viskores::Id LabelComponents(viskores::Id numVertices,
                             const viskores::cont::ArrayHandle<viskores::Id2>& edges,
                             const LabelsArray& labels,
                             viskores::cont::ArrayHandle<viskores::Id>& components,
                             viskores::cont::ArrayHandle<viskores::Id>* roots = nullptr)
{
  viskores::cont::Invoker invoke;
  viskores::cont::ArrayHandle<viskores::Id> parents;
  viskores::cont::ArrayCopy(viskores::cont::ArrayHandleIndex(numVertices), parents);
  invoke(detail::UniteEdges{}, edges, labels, parents);
  invoke(viskores::worklet::connectivity::PointerJumping{}, parents);
  viskores::cont::ArrayHandle<viskores::Id> isRoot;
  invoke(IsEqual{}, viskores::cont::ArrayHandleIndex(numVertices), parents, isRoot);
  viskores::cont::ArrayHandle<viskores::Id> rank;
  const viskores::Id count = viskores::cont::Algorithm::ScanExclusive(isRoot, rank);
  invoke(GatherValue{}, parents, rank, components);
  if (roots)
  {
    *roots = parents;
  }
  return count;
}

namespace detail
{

struct CountBoundaryVertices : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn boundary, FieldIn label, AtomicArrayInOut count);
  using ExecutionSignature = void(_1, _2, _3);

  template <typename Atomic>
  VISKORES_EXEC void operator()(viskores::UInt8 boundary, viskores::Id label, Atomic& count) const
  {
    if (boundary)
    {
      count.Add(label, 1);
    }
  }
};

} // namespace detail

/// Reads a triangle complex and builds its edges and incidences, and, if it is a surface, its
/// one-rings. Throws `ErrorBadValue` if a point is in no triangle, and if the complex is not a
/// consistently oriented surface (closed or with boundary).
inline void BuildSurfaceMesh(const viskores::cont::ArrayHandle<Vec3>& points,
                             const viskores::cont::ArrayHandle<viskores::Id3>& triangles,
                             SurfaceMesh& mesh)
{
  using Algorithm = viskores::cont::Algorithm;
  viskores::cont::Invoker invoke;
  const viskores::Id n = points.GetNumberOfValues();
  const viskores::Id m = triangles.GetNumberOfValues();
  mesh = SurfaceMesh{};
  mesh.NumberOfPoints = n;
  mesh.NumberOfTriangles = m;
  mesh.Points = points;
  mesh.Triangles = triangles;
  if (m == 0)
  {
    throw viskores::cont::ErrorBadValue("QuadricClusteringDecimation: the mesh has no triangles.");
  }

  // Edges.
  viskores::cont::ArrayHandle<viskores::Id2> halfEdgeKeys;
  viskores::cont::ArrayHandle<viskores::Id> halfEdgeIds;
  invoke(detail::HalfEdges{},
         triangles,
         viskores::cont::make_ArrayHandleGroupVec<3>(halfEdgeKeys),
         viskores::cont::make_ArrayHandleGroupVec<3>(halfEdgeIds));
  viskores::worklet::Keys<viskores::Id2> edgeKeys;
  edgeKeys.BuildArrays(halfEdgeKeys, viskores::worklet::KeysSortType::Unstable);
  mesh.Edges = edgeKeys.GetUniqueKeys();
  viskores::cont::ArrayHandle<viskores::Int8> edgeSigns;
  invoke(detail::EdgeIncidence{},
         edgeKeys,
         halfEdgeIds,
         triangles,
         mesh.EdgeTriangles,
         mesh.EdgeTriangleCounts,
         edgeSigns);
  bool unoriented = false;
  {
    viskores::cont::ArrayHandle<viskores::Id> negative;
    invoke(detail::NegativeSign{}, edgeSigns, negative);
    unoriented = Algorithm::Reduce(negative, viskores::Id(0)) > 0;
  }

  // Edges of every vertex, in increasing edge order.
  viskores::cont::ArrayHandle<viskores::Id> endVertex, endEdge;
  invoke(detail::EdgeEnds{},
         mesh.Edges,
         viskores::cont::make_ArrayHandleGroupVec<2>(endVertex),
         viskores::cont::make_ArrayHandleGroupVec<2>(endEdge));
  viskores::worklet::Keys<viskores::Id> vertexKeys;
  vertexKeys.BuildArrays(endVertex, viskores::worklet::KeysSortType::Stable);
  if (vertexKeys.GetInputRange() != n)
  {
    throw viskores::cont::ErrorBadValue(
      "QuadricClusteringDecimation: the mesh has points that no triangle uses.");
  }
  mesh.VertexEdgeOffsets = vertexKeys.GetOffsets();
  Algorithm::Copy(
    viskores::cont::make_ArrayHandlePermutation(vertexKeys.GetSortedValuesMap(), endEdge),
    mesh.VertexEdges);

  // Corners of every vertex, in increasing corner order.
  viskores::cont::ArrayHandle<viskores::Id> cornerVertex, cornerFrom, cornerTo;
  cornerVertex.Allocate(3 * m);
  cornerFrom.Allocate(3 * m);
  cornerTo.Allocate(3 * m);
  invoke(detail::CornerPairs(m), triangles, cornerVertex, cornerFrom, cornerTo);
  viskores::worklet::Keys<viskores::Id> cornerKeys;
  cornerKeys.BuildArrays(cornerVertex, viskores::worklet::KeysSortType::Stable);
  mesh.VertexCornerOffsets = cornerKeys.GetOffsets();
  mesh.VertexCorners = cornerKeys.GetSortedValuesMap();

  // Is it a surface?
  std::string problem;
  const viskores::IdComponent maxCount =
    Algorithm::Reduce(mesh.EdgeTriangleCounts, viskores::IdComponent(0), viskores::Maximum());
  if (maxCount > 2)
  {
    problem = "an edge lies in more than two triangles";
  }
  else if (unoriented)
  {
    problem = "the triangles are not consistently oriented";
  }
  else
  {
    viskores::cont::ArrayHandle<viskores::Id> ringSize;
    viskores::cont::ArrayHandle<viskores::UInt8> valid;
    viskores::cont::ArrayHandle<viskores::Id> noOffsets;
    viskores::cont::ArrayHandle<viskores::Id> noRings;
    invoke(detail::RingWalk<false>{},
           cornerKeys,
           cornerFrom,
           cornerTo,
           noOffsets,
           noRings,
           ringSize,
           mesh.IsBoundaryVertex,
           valid);
    if (Algorithm::Reduce(valid, viskores::Id(0)) != n)
    {
      problem = "the triangles around a vertex do not form a single fan";
    }
    else
    {
      Algorithm::ScanExtended(ringSize, mesh.RingOffsets);
      mesh.Rings.Allocate(viskores::cont::ArrayGetValue(n, mesh.RingOffsets));
      invoke(detail::RingWalk<true>{},
             cornerKeys,
             cornerFrom,
             cornerTo,
             mesh.RingOffsets,
             mesh.Rings,
             ringSize,
             mesh.IsBoundaryVertex,
             valid);
      mesh.HasRings = true;
      mesh.HasBoundary = Algorithm::Reduce(mesh.IsBoundaryVertex, viskores::Id(0)) > 0;
      mesh.NumberOfComponents =
        LabelComponents(n,
                        mesh.Edges,
                        viskores::cont::make_ArrayHandleConstant(viskores::Id(0), n),
                        mesh.Components);
      mesh.ComponentBoundaryVertices.AllocateAndFill(mesh.NumberOfComponents, 0);
      invoke(detail::CountBoundaryVertices{},
             mesh.IsBoundaryVertex,
             mesh.Components,
             mesh.ComponentBoundaryVertices);
    }
  }
  if (!problem.empty())
  {
    throw viskores::cont::ErrorBadValue(
      "QuadricClusteringDecimation: the mesh is not a surface: " + problem + ".");
  }
}

}
}
} // namespace viskores::worklet::quadric_clustering

VISKORES_QUADRIC_CLUSTERING_STRICT_FP_END

#endif // viskores_worklet_quadric_clustering_SurfaceMesh_h
