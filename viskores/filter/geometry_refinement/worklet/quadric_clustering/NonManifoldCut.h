//============================================================================
//  The contents of this file are covered by the Viskores license. See
//  LICENSE.txt for details.
//
//  By contributing to this file, all contributors agree to the Developer
//  Certificate of Origin Version 1.1 (DCO 1.1) as stated in DCO.txt.
//============================================================================
#ifndef viskores_worklet_quadric_clustering_NonManifoldCut_h
#define viskores_worklet_quadric_clustering_NonManifoldCut_h

#include <viskores/filter/geometry_refinement/worklet/quadric_clustering/QuadricClusteringCommonWorklets.h>

#include <viskores/Types.h>
#include <viskores/cont/Algorithm.h>
#include <viskores/cont/ArrayHandle.h>
#include <viskores/cont/ArrayHandleGroupVec.h>
#include <viskores/cont/ArrayHandleIndex.h>
#include <viskores/cont/Invoker.h>
#include <viskores/worklet/WorkletMapField.h>

namespace viskores
{
namespace worklet
{
namespace quadric_clustering
{

/// A triangle complex cut into surfaces (Theorem "Inputs that are not surfaces" of the paper,
/// after Gueziec et al., "Cutting and stitching"): every vertex is split into one copy per fan of
/// triangles connected across edges not in the cut set `X`: the singular edges (in three or more
/// triangles), and with `seams` a seam that makes every piece orientable.
struct CutComplex
{
  /// The triangles on the copies, reoriented with `seams` so that every piece is consistently
  /// oriented.
  viskores::cont::ArrayHandle<viskores::Id3> Triangles;
  /// The input vertex of every copy.
  viskores::cont::ArrayHandle<viskores::Id> Original;
  /// The copies of the vertices on an edge of `X` and of the vertices with several fans, which
  /// stay singleton clusters.
  viskores::cont::ArrayHandle<viskores::UInt8> Frozen;
  /// The edges of `X`, as sorted vertex pairs in lexicographic order.
  viskores::cont::ArrayHandle<viskores::Id2> CutEdges;
};

namespace cut_detail
{

/// The three edge records `(min, max, t)` of triangle `t`.
struct EdgeRecords : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn triangle, FieldOut records);
  using ExecutionSignature = void(_1, _2, WorkIndex);

  template <typename RecordVec>
  VISKORES_EXEC void operator()(const viskores::Id3& tri, RecordVec& records, viskores::Id t) const
  {
    for (viskores::IdComponent s = 0; s < 3; ++s)
    {
      const viskores::Id x = tri[s], y = tri[(s + 1) % 3];
      records[s] = viskores::Id3(viskores::Min(x, y), viskores::Max(x, y), t);
    }
  }
};

/// Whether a sorted edge record is the first of its edge, and its edge.
struct FirstOfEdge : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn index, WholeArrayIn records, FieldOut first, FieldOut edge);
  using ExecutionSignature = void(_1, _2, _3, _4);

  template <typename RecordPortal>
  VISKORES_EXEC void operator()(viskores::Id i,
                                const RecordPortal& records,
                                viskores::UInt8& first,
                                viskores::Id2& edge) const
  {
    const viskores::Id3 r = records.Get(i);
    edge = viskores::Id2(r[0], r[1]);
    first = (i == 0 || records.Get(i - 1)[0] != r[0] || records.Get(i - 1)[1] != r[1]) ? 1 : 0;
  }
};

/// For every edge: its triangles when it has exactly two, whether it is in `X` (singular), and
/// whether the orientation crosses it.
struct EdgeIncidence : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn start,
                                WholeArrayIn starts,
                                WholeArrayIn records,
                                FieldOut triangles,
                                FieldOut inX,
                                FieldOut usable);
  using ExecutionSignature = void(_1, _2, _3, _4, _5, _6, WorkIndex);

  template <typename IdPortal, typename RecordPortal>
  VISKORES_EXEC void operator()(viskores::Id start,
                                const IdPortal& starts,
                                const RecordPortal& records,
                                viskores::Id2& triangles,
                                viskores::UInt8& inX,
                                viskores::UInt8& usable,
                                viskores::Id e) const
  {
    const viskores::Id end =
      (e + 1 < starts.GetNumberOfValues()) ? starts.Get(e + 1) : records.GetNumberOfValues();
    const viskores::Id count = end - start;
    triangles = (count == 2) ? viskores::Id2(records.Get(start)[2], records.Get(start + 1)[2])
                             : viskores::Id2(-1);
    inX = (count >= 3) ? 1 : 0;
    usable = (count == 2 && !inX) ? 1 : 0;
  }
};

/// The two triangles of a usable edge, as an edge of the dual graph; otherwise a loop at 0,
/// which unites nothing.
struct DualEdge : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn triangles, FieldIn usable, FieldOut dual);
  using ExecutionSignature = _3(_1, _2);

  VISKORES_EXEC viskores::Id2 operator()(const viskores::Id2& triangles,
                                         viskores::UInt8 usable) const
  {
    return usable ? triangles : viskores::Id2(0, 0);
  }
};

/// The root of every component starts the orientation with sign +1.
struct RootSign : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn root, FieldOut sign, FieldOut isRoot);
  using ExecutionSignature = void(_1, _2, _3, WorkIndex);

  VISKORES_EXEC void operator()(viskores::Id root,
                                viskores::Int8& sign,
                                viskores::UInt8& isRoot,
                                viskores::Id t) const
  {
    isRoot = (root == t) ? 1 : 0;
    sign = isRoot ? 1 : 0;
  }
};

/// +1 if the triangle traverses `x -> y`, -1 if `y -> x`.
VISKORES_EXEC inline viskores::Id Direction(const viskores::Id3& t, viskores::Id x, viskores::Id y)
{
  for (viskores::IdComponent s = 0; s < 3; ++s)
  {
    if (t[s] == x && t[(s + 1) % 3] == y)
    {
      return 1;
    }
  }
  return -1;
}

/// For the triangle at position `p` of the frontier, through its side `s`: the unsigned
/// neighbour it reaches, with the key `3 p + s` of the order of a queue and the sign it would
/// give; `(-1, -1, 0)` where there is none.
struct Discover : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn triangle,
                                WholeArrayIn triangles,
                                WholeArrayIn sideEdges,
                                WholeArrayIn edgeTriangles,
                                WholeArrayIn usable,
                                WholeArrayIn signs,
                                FieldOut candidates);
  using ExecutionSignature = void(_1, _2, _3, _4, _5, _6, _7, WorkIndex);

  template <typename TrianglePortal,
            typename IdPortal,
            typename PairPortal,
            typename FlagPortal,
            typename SignPortal,
            typename CandidateVec>
  VISKORES_EXEC void operator()(viskores::Id t,
                                const TrianglePortal& triangles,
                                const IdPortal& sideEdges,
                                const PairPortal& edgeTriangles,
                                const FlagPortal& usable,
                                const SignPortal& signs,
                                CandidateVec& candidates,
                                viskores::Id p) const
  {
    const viskores::Id3 tri = triangles.Get(t);
    for (viskores::IdComponent s = 0; s < 3; ++s)
    {
      candidates[s] = viskores::Id3(-1, -1, 0);
      const viskores::Id e = sideEdges.Get(3 * t + s);
      if (!usable.Get(e))
      {
        continue;
      }
      const viskores::Id2 pair = edgeTriangles.Get(e);
      const viskores::Id u = (pair[0] == t) ? pair[1] : pair[0];
      if (signs.Get(u) != 0)
      {
        continue;
      }
      const viskores::Id x = tri[s], y = tri[(s + 1) % 3];
      const viskores::Id want =
        -viskores::Id(signs.Get(t)) * Direction(tri, x, y) * Direction(triangles.Get(u), x, y);
      candidates[s] = viskores::Id3(u, 3 * p + s, want);
    }
  }
};

/// Whether a candidate sorted by (triangle, key) is the first of its triangle, the one that a
/// queue processed in order reaches first; with its (key, triangle, sign) for the next frontier.
struct FirstDiscoverer : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn index,
                                WholeArrayIn candidates,
                                FieldOut first,
                                FieldOut order);
  using ExecutionSignature = void(_1, _2, _3, _4);

  template <typename CandidatePortal>
  VISKORES_EXEC void operator()(viskores::Id i,
                                const CandidatePortal& candidates,
                                viskores::UInt8& first,
                                viskores::Id3& order) const
  {
    const viskores::Id3 c = candidates.Get(i);
    first = (i == 0 || candidates.Get(i - 1)[0] != c[0]) ? 1 : 0;
    order = viskores::Id3(c[1], c[0], c[2]);
  }
};

/// Gives a newly reached triangle its sign, and returns it for the next frontier.
struct SignReached : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn order, WholeArrayInOut signs, FieldOut triangle);
  using ExecutionSignature = _3(_1, _2);

  template <typename SignPortal>
  VISKORES_EXEC viskores::Id operator()(const viskores::Id3& order, SignPortal& signs) const
  {
    signs.Set(order[1], static_cast<viskores::Int8>(order[2]));
    return order[1];
  }
};

/// Whether an edge is in `X`: in it already, or a usable edge that joins its triangles against
/// their final orientations, the seam.
struct InCutSet : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn edge,
                                FieldIn edgeTriangles,
                                FieldIn usable,
                                FieldIn inX,
                                WholeArrayIn triangles,
                                WholeArrayIn signs,
                                FieldOut finalInX);
  using ExecutionSignature = _7(_1, _2, _3, _4, _5, _6);

  template <typename TrianglePortal, typename SignPortal>
  VISKORES_EXEC viskores::UInt8 operator()(const viskores::Id2& edge,
                                           const viskores::Id2& pair,
                                           viskores::UInt8 usable,
                                           viskores::UInt8 inX,
                                           const TrianglePortal& triangles,
                                           const SignPortal& signs) const
  {
    if (!usable)
    {
      return inX;
    }
    const viskores::Id want = -viskores::Id(signs.Get(pair[0])) *
      Direction(triangles.Get(pair[0]), edge[0], edge[1]) *
      Direction(triangles.Get(pair[1]), edge[0], edge[1]);
    return (viskores::Id(signs.Get(pair[1])) != want) ? 1 : 0;
  }
};

/// The triangle reversed where its sign is negative.
struct Orient : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn triangle, FieldIn sign, FieldOut oriented);
  using ExecutionSignature = _3(_1, _2);

  VISKORES_EXEC viskores::Id3 operator()(const viskores::Id3& t, viskores::Int8 sign) const
  {
    return (sign < 0) ? viskores::Id3(t[2], t[1], t[0]) : t;
  }
};

/// The corner `3 t + c` of vertex `v` in triangle `t`.
template <typename TrianglePortal>
VISKORES_EXEC viskores::Id Corner(const TrianglePortal& triangles, viskores::Id v, viskores::Id t)
{
  const viskores::Id3 tri = triangles.Get(t);
  for (viskores::IdComponent c = 0; c < 3; ++c)
  {
    if (tri[c] == v)
    {
      return 3 * t + c;
    }
  }
  return 3 * t;
}

/// The pairs of corners that an edge in two triangles and not in `X` joins, one at each end;
/// loops at 0, which unite nothing, otherwise.
struct CornerEdges : viskores::worklet::WorkletMapField
{
  using ControlSignature =
    void(FieldIn edge, FieldIn edgeTriangles, FieldIn inX, WholeArrayIn oriented, FieldOut pairs);
  using ExecutionSignature = void(_1, _2, _3, _4, _5);

  template <typename TrianglePortal, typename PairVec>
  VISKORES_EXEC void operator()(const viskores::Id2& edge,
                                const viskores::Id2& pair,
                                viskores::UInt8 inX,
                                const TrianglePortal& oriented,
                                PairVec& pairs) const
  {
    for (viskores::IdComponent end = 0; end < 2; ++end)
    {
      if (pair[0] < 0 || inX)
      {
        pairs[end] = viskores::Id2(0, 0);
        continue;
      }
      const viskores::Id v = edge[end];
      pairs[end] = viskores::Id2(Corner(oriented, v, pair[0]), Corner(oriented, v, pair[1]));
    }
  }
};

/// For a root corner: its vertex and triangle, which order the copies, and the corner.
struct RootRecord : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn corner, WholeArrayIn oriented, FieldOut record);
  using ExecutionSignature = _3(_1, _2);

  template <typename TrianglePortal>
  VISKORES_EXEC viskores::Id3 operator()(viskores::Id z, const TrianglePortal& oriented) const
  {
    return viskores::Id3(oriented.Get(z / 3)[static_cast<viskores::IdComponent>(z % 3)], z / 3, z);
  }
};

/// Copy `r` is the root corner of rank `r`: its input vertex, and the copy of the corner.
struct NumberCopy : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn record, FieldOut original, WholeArrayOut copyOfRoot);
  using ExecutionSignature = void(_1, _2, _3, WorkIndex);

  template <typename IdPortal>
  VISKORES_EXEC void operator()(const viskores::Id3& record,
                                viskores::Id& original,
                                IdPortal& copyOfRoot,
                                viskores::Id r) const
  {
    original = record[0];
    copyOfRoot.Set(record[2], r);
  }
};

/// The copies of the three corners of a triangle.
struct CopyTriangle : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn index,
                                WholeArrayIn roots,
                                WholeArrayIn copyOfRoot,
                                FieldOut triangle);
  using ExecutionSignature = _4(_1, _2, _3);

  template <typename IdPortal>
  VISKORES_EXEC viskores::Id3 operator()(viskores::Id t,
                                         const IdPortal& roots,
                                         const IdPortal& copyOfRoot) const
  {
    return viskores::Id3(copyOfRoot.Get(roots.Get(3 * t)),
                         copyOfRoot.Get(roots.Get(3 * t + 1)),
                         copyOfRoot.Get(roots.Get(3 * t + 2)));
  }
};

/// Marks the ends of an edge of `X` on the cut; every write is a 1.
struct MarkOnCut : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn edge, WholeArrayOut onCut);
  using ExecutionSignature = void(_1, _2);

  template <typename FlagPortal>
  VISKORES_EXEC void operator()(const viskores::Id2& edge, FlagPortal& onCut) const
  {
    onCut.Set(edge[0], 1);
    onCut.Set(edge[1], 1);
  }
};

/// A copy is frozen if its vertex has several copies, which are numbered consecutively, or
/// lies on the cut.
struct IsFrozenCopy : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn index,
                                WholeArrayIn original,
                                WholeArrayIn onCut,
                                FieldOut frozen);
  using ExecutionSignature = _4(_1, _2, _3);

  template <typename IdPortal, typename FlagPortal>
  VISKORES_EXEC viskores::UInt8 operator()(viskores::Id r,
                                           const IdPortal& original,
                                           const FlagPortal& onCut) const
  {
    const viskores::Id v = original.Get(r);
    const bool several = (r > 0 && original.Get(r - 1) == v) ||
      (r + 1 < original.GetNumberOfValues() && original.Get(r + 1) == v);
    return (several || onCut.Get(v)) ? 1 : 0;
  }
};

} // namespace cut_detail

/// Cuts the triangles `triangles` on `numVertices` vertices as `cut` of
/// proofs/lib/nonmanifold.py, deterministically: the orientation is propagated breadth first
/// from the smallest triangle of every component, across edges in exactly two triangles that
/// are not in `X`, and an edge whose two triangles cannot be made to agree joins the seam. The
/// search runs a level at a time, and a triangle takes its sign from the first triangle of the
/// level, in queue order, that reaches it, as a queue would. Copies are numbered by vertex, then
/// by the smallest triangle of their fan. Without `seams`, only the singular edges are cut and
/// nothing is reoriented, so a piece may not be orientable (Corollary "Cuts that are not orientable").
inline CutComplex CutAlongSingularEdges(const viskores::cont::ArrayHandle<viskores::Id3>& triangles,
                                        viskores::Id numVertices,
                                        bool seams = true)
{
  using Algorithm = viskores::cont::Algorithm;
  namespace cd = cut_detail;
  viskores::cont::Invoker invoke;
  CutComplex cut;
  const viskores::Id m = triangles.GetNumberOfValues();

  // The edges, with their triangles in increasing order.
  viskores::cont::ArrayHandle<viskores::Id3> records;
  invoke(cd::EdgeRecords{}, triangles, viskores::cont::make_ArrayHandleGroupVec<3>(records));
  Algorithm::Sort(records);
  viskores::cont::ArrayHandle<viskores::UInt8> first;
  viskores::cont::ArrayHandle<viskores::Id2> recordEdges, edges;
  invoke(cd::FirstOfEdge{}, viskores::cont::ArrayHandleIndex(3 * m), records, first, recordEdges);
  Algorithm::CopyIf(recordEdges, first, edges);
  viskores::cont::ArrayHandle<viskores::Id> starts;
  Algorithm::CopyIf(viskores::cont::ArrayHandleIndex(3 * m), first, starts);
  viskores::cont::ArrayHandle<viskores::Id2> edgeTriangles;
  viskores::cont::ArrayHandle<viskores::UInt8> inX, usable;
  invoke(cd::EdgeIncidence{}, starts, starts, records, edgeTriangles, inX, usable);
  viskores::cont::ArrayHandle<viskores::UInt8> finalInX;
  viskores::cont::ArrayHandle<viskores::Id3> oriented;
  if (!seams)
  {
    finalInX = inX;
    oriented = triangles;
  }
  else
  {
    viskores::cont::ArrayHandle<viskores::Id2> sideKeys;
    invoke(SortedTriangleEdges{}, triangles, viskores::cont::make_ArrayHandleGroupVec<3>(sideKeys));
    viskores::cont::ArrayHandle<viskores::Id> sideEdges;
    Algorithm::LowerBounds(edges, sideKeys, sideEdges);

    // Orientation by breadth-first search from the smallest triangle of every component of the
    // graph of the usable edges.
    viskores::cont::ArrayHandle<viskores::Id2> dual;
    invoke(cd::DualEdge{}, edgeTriangles, usable, dual);
    const viskores::cont::ArrayHandle<viskores::Id> triangleRoots = SmallestRoots(m, dual);
    viskores::cont::ArrayHandle<viskores::Int8> signs;
    viskores::cont::ArrayHandle<viskores::UInt8> isRoot;
    invoke(cd::RootSign{}, triangleRoots, signs, isRoot);
    viskores::cont::ArrayHandle<viskores::Id> frontier;
    Algorithm::CopyIf(viskores::cont::ArrayHandleIndex(m), isRoot, frontier);
    while (frontier.GetNumberOfValues() > 0)
    {
      viskores::cont::ArrayHandle<viskores::Id3> slots, candidates;
      invoke(cd::Discover{},
             frontier,
             triangles,
             sideEdges,
             edgeTriangles,
             usable,
             signs,
             viskores::cont::make_ArrayHandleGroupVec<3>(slots));
      viskores::cont::ArrayHandle<viskores::UInt8> valid;
      invoke(IsValidRecord{}, slots, valid);
      Algorithm::CopyIf(slots, valid, candidates);
      Algorithm::Sort(candidates);
      viskores::cont::ArrayHandle<viskores::UInt8> firsts;
      viskores::cont::ArrayHandle<viskores::Id3> orders, reached;
      invoke(cd::FirstDiscoverer{},
             viskores::cont::ArrayHandleIndex(candidates.GetNumberOfValues()),
             candidates,
             firsts,
             orders);
      Algorithm::CopyIf(orders, firsts, reached);
      Algorithm::Sort(reached);
      invoke(cd::SignReached{}, reached, signs, frontier);
    }
    invoke(cd::InCutSet{}, edges, edgeTriangles, usable, inX, triangles, signs, finalInX);
    invoke(cd::Orient{}, triangles, signs, oriented);
  }

  // Fans: the corners joined across the edges in two triangles and not in X; the root of a fan
  // is its corner of smallest triangle.
  viskores::cont::ArrayHandle<viskores::Id2> cornerEdges;
  invoke(cd::CornerEdges{},
         edges,
         edgeTriangles,
         finalInX,
         oriented,
         viskores::cont::make_ArrayHandleGroupVec<2>(cornerEdges));
  const viskores::cont::ArrayHandle<viskores::Id> cornerRoots = SmallestRoots(3 * m, cornerEdges);
  // Copies numbered in the order of their roots (v, t).
  viskores::cont::ArrayHandle<viskores::Id> isOwnRoot;
  invoke(IsEqual{}, viskores::cont::ArrayHandleIndex(3 * m), cornerRoots, isOwnRoot);
  viskores::cont::ArrayHandle<viskores::Id> rootCorners;
  Algorithm::CopyIf(viskores::cont::ArrayHandleIndex(3 * m), isOwnRoot, rootCorners);
  viskores::cont::ArrayHandle<viskores::Id3> rootRecords;
  invoke(cd::RootRecord{}, rootCorners, oriented, rootRecords);
  Algorithm::Sort(rootRecords);
  viskores::cont::ArrayHandle<viskores::Id> copyOfRoot;
  copyOfRoot.AllocateAndFill(3 * m, -1);
  invoke(cd::NumberCopy{}, rootRecords, cut.Original, copyOfRoot);
  invoke(cd::CopyTriangle{},
         viskores::cont::ArrayHandleIndex(m),
         cornerRoots,
         copyOfRoot,
         cut.Triangles);
  Algorithm::CopyIf(edges, finalInX, cut.CutEdges);
  viskores::cont::ArrayHandle<viskores::UInt8> onCut;
  onCut.AllocateAndFill(numVertices, 0);
  invoke(cd::MarkOnCut{}, cut.CutEdges, onCut);
  invoke(cd::IsFrozenCopy{},
         viskores::cont::ArrayHandleIndex(cut.Original.GetNumberOfValues()),
         cut.Original,
         onCut,
         cut.Frozen);
  return cut;
}

}
}
} // namespace viskores::worklet::quadric_clustering

#endif // viskores_worklet_quadric_clustering_NonManifoldCut_h
