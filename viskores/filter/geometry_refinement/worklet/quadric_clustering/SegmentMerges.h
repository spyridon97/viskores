//============================================================================
//  The contents of this file are covered by the Viskores license. See
//  LICENSE.txt for details.
//
//  By contributing to this file, all contributors agree to the Developer
//  Certificate of Origin Version 1.1 (DCO 1.1) as stated in DCO.txt.
//============================================================================
#ifndef viskores_worklet_quadric_clustering_SegmentMerges_h
#define viskores_worklet_quadric_clustering_SegmentMerges_h

#include <viskores/filter/geometry_refinement/worklet/quadric_clustering/MergePhase.h>
#include <viskores/filter/geometry_refinement/worklet/quadric_clustering/QuadricClusteringCommonWorklets.h>
#include <viskores/filter/geometry_refinement/worklet/quadric_clustering/SegmentFlags.h>
#include <viskores/filter/geometry_refinement/worklet/quadric_clustering/SingularCurves.h>


#include <viskores/cont/Algorithm.h>
#include <viskores/cont/ArrayGetValues.h>
#include <viskores/cont/ArrayHandle.h>
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

/// The segments of the arcs in a round of segment merges, in the order of every arc's list:
/// along an open arc, and around a closed arc from the head of its first segment.
struct SegmentList
{
  /// Per segment: its arc, the position of its head, its first and last vertices and its
  /// length.
  viskores::cont::ArrayHandle<viskores::Id> Arc;
  viskores::cont::ArrayHandle<viskores::Id> Head;
  viskores::cont::ArrayHandle<viskores::Id> First;
  viskores::cont::ArrayHandle<viskores::Id> Last;
  viskores::cont::ArrayHandle<viskores::Id> Length;
  /// Per arc: the index of its first segment; one more entry at the end.
  viskores::cont::ArrayHandle<viskores::Id> Offsets;
};

namespace segment_detail
{

/// The head of a segment, alive (not merged into a junction), as `(arc, order, position)`, where
/// the order runs along an open arc and around a closed arc from its first head; `(-1, ...)`
/// otherwise.
struct HeadRecord : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn position,
                                FieldIn arc,
                                FieldIn isHead,
                                WholeArrayIn arcOffsets,
                                WholeArrayIn arcEnds,
                                WholeArrayIn firstHead,
                                WholeArrayIn absorbed,
                                FieldOut record);
  using ExecutionSignature = _8(_1, _2, _3, _4, _5, _6, _7);

  template <typename IdPortal, typename PairPortal>
  VISKORES_EXEC viskores::Id3 operator()(viskores::Id p,
                                         viskores::Id a,
                                         viskores::UInt8 isHead,
                                         const IdPortal& arcOffsets,
                                         const PairPortal& arcEnds,
                                         const IdPortal& firstHead,
                                         const PairPortal& absorbed) const
  {
    const viskores::Id start = arcOffsets.Get(a), end = arcOffsets.Get(a + 1);
    const viskores::Id2 absorbedCounts = absorbed.Get(a);
    if (!isHead || p < start + absorbedCounts[0] || p >= end - absorbedCounts[1])
    {
      return viskores::Id3(-1);
    }
    const viskores::Id length = end - start;
    const bool closed = arcEnds.Get(a)[0] < 0;
    const viskores::Id order = closed ? (p - firstHead.Get(a) + length) % length : p;
    return viskores::Id3(a, order, p);
  }
};

struct HeadOfRecord : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn record, FieldOut arc, FieldOut head);
  using ExecutionSignature = void(_1, _2, _3);

  VISKORES_EXEC void operator()(const viskores::Id3& r, viskores::Id& arc, viskores::Id& head) const
  {
    arc = r[0];
    head = r[2];
  }
};

/// The extent of a segment: from its head to the next head, around a closed arc, or to the end
/// of the alive part of an open arc.
struct SegmentExtent : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn arc,
                                FieldIn head,
                                WholeArrayIn heads,
                                WholeArrayIn segmentOffsets,
                                WholeArrayIn arcOffsets,
                                WholeArrayIn arcEnds,
                                WholeArrayIn absorbed,
                                WholeArrayIn arcVertices,
                                FieldOut first,
                                FieldOut last,
                                FieldOut length);
  using ExecutionSignature = void(_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, WorkIndex);

  template <typename IdPortal, typename PairPortal>
  VISKORES_EXEC void operator()(viskores::Id a,
                                viskores::Id head,
                                const IdPortal& heads,
                                const IdPortal& segmentOffsets,
                                const IdPortal& arcOffsets,
                                const PairPortal& arcEnds,
                                const PairPortal& absorbed,
                                const IdPortal& arcVertices,
                                viskores::Id& first,
                                viskores::Id& last,
                                viskores::Id& length,
                                viskores::Id s) const
  {
    const viskores::Id start = arcOffsets.Get(a), end = arcOffsets.Get(a + 1);
    const viskores::Id arcLength = end - start;
    const bool closed = arcEnds.Get(a)[0] < 0;
    const viskores::Id fs = segmentOffsets.Get(a), ls = segmentOffsets.Get(a + 1) - 1;
    first = arcVertices.Get(head);
    if (closed)
    {
      const viskores::Id next = (s < ls) ? heads.Get(s + 1) : heads.Get(fs);
      length = (next - head + arcLength) % arcLength;
      length = (length == 0) ? arcLength : length;
      last = arcVertices.Get(start + (head - start + length - 1) % arcLength);
    }
    else
    {
      const viskores::Id next = (s < ls) ? heads.Get(s + 1) : end - absorbed.Get(a)[1];
      length = next - head;
      last = arcVertices.Get(next - 1);
    }
  }
};

} // namespace segment_detail

/// The candidate merges of a round. A segment merge (`SegmentMergeKind`, position `si`) joins
/// segment `si` of the arc and the next; a junction merge (`JunctionMergeKind`, position `end`)
/// joins the end segment at `end` to its junction. In every sheet, the piece of the source joins
/// the cluster of the copy of the target that a copy of the edge `(target, source)` of `X` meets.
struct CandidateList
{
  viskores::cont::ArrayHandle<viskores::Id> Arc;
  viskores::cont::ArrayHandle<viskores::IdComponent> Kind;
  viskores::cont::ArrayHandle<viskores::Id> Position;
  viskores::cont::ArrayHandle<viskores::Id2> Edge;
  /// The segment whose vertices change segment, and the segment (its first vertex) or junction
  /// they join.
  viskores::cont::ArrayHandle<viskores::Id> Moved;
  viskores::cont::ArrayHandle<viskores::Id> MovedTo;
  /// The two segments (first vertices) or junction whose glue groups merge.
  viskores::cont::ArrayHandle<viskores::Id2> Groups;
  /// The keys the candidate touches and those it takes, for the greedy commit: segments by
  /// index, junctions after them; -1 where none.
  viskores::cont::ArrayHandle<viskores::Vec<viskores::Id, 5>> Touch;
  viskores::cont::ArrayHandle<viskores::Id2> Take;
};

namespace segment_detail
{

/// The number of junction merges of an arc: two when allowed.
VISKORES_EXEC inline viskores::Id JunctionCandidates(bool junctionMerges,
                                                     viskores::Id ns,
                                                     const viskores::Id2& ends)
{
  return (junctionMerges && ends[0] >= 0 && ns >= 1 && !(ns == 1 && ends[0] == ends[1])) ? 2 : 0;
}

/// The number of candidates of an arc: its junction merges, and one segment merge per pair of
/// consecutive segments.
struct CandidateCount : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn arc,
                                WholeArrayIn segmentOffsets,
                                WholeArrayIn arcEnds,
                                FieldOut count);
  using ExecutionSignature = _4(_1, _2, _3);

  VISKORES_CONT explicit CandidateCount(bool junctionMerges)
    : JunctionMerges(junctionMerges)
  {
  }

  template <typename IdPortal, typename PairPortal>
  VISKORES_EXEC viskores::Id operator()(viskores::Id a,
                                        const IdPortal& segmentOffsets,
                                        const PairPortal& arcEnds) const
  {
    const viskores::Id ns = segmentOffsets.Get(a + 1) - segmentOffsets.Get(a);
    const viskores::Id2 ends = arcEnds.Get(a);
    const viskores::Id segment = (ns >= 2) ? ((ends[0] >= 0) ? ns - 1 : ns) : 0;
    return JunctionCandidates(this->JunctionMerges, ns, ends) + segment;
  }

  bool JunctionMerges;
};

} // namespace segment_detail

/// The kinds of candidate merges; the kind orders the candidates after their arc.
enum : viskores::IdComponent
{
  SegmentMergeKind = 0,
  JunctionMergeKind = 1
};

namespace segment_detail
{

/// Candidate `c`: its arc, kind and position, the edge of `X` it merges across, the segment it
/// moves, the groups it merges, and its keys.
struct CandidateFields : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn candidate,
                                WholeArrayIn candidateOffsets,
                                WholeArrayIn segmentOffsets,
                                WholeArrayIn arcOffsets,
                                WholeArrayIn arcEnds,
                                WholeArrayIn absorbed,
                                WholeArrayIn arcVertices,
                                WholeArrayIn first,
                                WholeArrayIn last,
                                FieldOut arc,
                                FieldOut kind,
                                FieldOut position,
                                FieldOut edge,
                                FieldOut moved,
                                FieldOut movedTo,
                                FieldOut groups,
                                FieldOut touch,
                                FieldOut take);
  using ExecutionSignature =
    void(_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18);

  VISKORES_CONT CandidateFields(bool junctionMerges, viskores::Id numSegments)
    : JunctionMerges(junctionMerges)
    , NumberOfSegments(numSegments)
  {
  }

  template <typename IdPortal, typename PairPortal>
  VISKORES_EXEC void operator()(viskores::Id c,
                                const IdPortal& candidateOffsets,
                                const IdPortal& segmentOffsets,
                                const IdPortal& arcOffsets,
                                const PairPortal& arcEnds,
                                const PairPortal& absorbed,
                                const IdPortal& arcVertices,
                                const IdPortal& first,
                                const IdPortal& last,
                                viskores::Id& arc,
                                viskores::IdComponent& kind,
                                viskores::Id& position,
                                viskores::Id2& edge,
                                viskores::Id& moved,
                                viskores::Id& movedTo,
                                viskores::Id2& groups,
                                viskores::Vec<viskores::Id, 5>& touch,
                                viskores::Id2& take) const
  {
    // The offsets of the arcs with no candidate repeat: the owner is the last arc starting at
    // or before c.
    const viskores::Id a = OwnerOf(candidateOffsets, c);
    arc = a;
    const viskores::Id i = c - candidateOffsets.Get(a);
    const viskores::Id fs = segmentOffsets.Get(a);
    const viskores::Id ns = segmentOffsets.Get(a + 1) - fs;
    const viskores::Id2 ends = arcEnds.Get(a);
    const viskores::Id junctions = JunctionCandidates(this->JunctionMerges, ns, ends);
    touch = viskores::Vec<viskores::Id, 5>(-1);
    if (i < junctions)
    {
      // The end segment joins its junction across the edge of X from its outer vertex to the
      // junction, or to the vertex absorbed last at that end.
      kind = JunctionMergeKind;
      position = i;
      const viskores::IdComponent end = static_cast<viskores::IdComponent>(i);
      const viskores::Id index = (end == 0) ? 0 : ns - 1;
      const viskores::Id s = fs + index;
      const viskores::Id u = (end == 0) ? first.Get(s) : last.Get(s);
      const viskores::Id2 absorbedCounts = absorbed.Get(a);
      const viskores::Id j = ends[end];
      viskores::Id t = j;
      if (end == 0 && absorbedCounts[0] > 0)
      {
        t = arcVertices.Get(arcOffsets.Get(a) + absorbedCounts[0] - 1);
      }
      else if (end == 1 && absorbedCounts[1] > 0)
      {
        t = arcVertices.Get(arcOffsets.Get(a + 1) - absorbedCounts[1]);
      }
      edge = viskores::Id2(t, u);
      moved = s;
      movedTo = j;
      groups = viskores::Id2(j, first.Get(s));
      touch[0] = s;
      const viskores::Id neighbor = index + ((end == 0) ? 1 : -1);
      touch[1] = (neighbor >= 0 && neighbor < ns) ? fs + neighbor : viskores::Id(-1);
      touch[2] = this->NumberOfSegments + j;
      take = viskores::Id2(s, this->NumberOfSegments + j);
      return;
    }
    // Segments si and si + 1 (around a closed arc): the piece of the second joins the first.
    kind = SegmentMergeKind;
    const viskores::Id si = i - junctions;
    position = si;
    const viskores::Id s1 = fs + si, s2 = fs + (si + 1) % ns;
    edge = viskores::Id2(last.Get(s1), first.Get(s2));
    moved = s2;
    movedTo = first.Get(s1);
    groups = viskores::Id2(first.Get(s1), first.Get(s2));
    // As the reference, the neighbours wrap around also on an open arc.
    touch[0] = s1;
    touch[1] = s2;
    touch[2] = fs + (si + ns - 1) % ns;
    touch[3] = fs + (si + 2) % ns;
    take = viskores::Id2(s1, s2);
  }

  bool JunctionMerges;
  viskores::Id NumberOfSegments;
};

/// The range of the copies of the edge of `X` of a candidate.
struct CopyRange : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn edge,
                                WholeArrayIn copyKeys,
                                FieldOut first,
                                FieldOut count);
  using ExecutionSignature = void(_1, _2, _3, _4);

  template <typename KeyPortal>
  VISKORES_EXEC void operator()(const viskores::Id2& edge,
                                const KeyPortal& copyKeys,
                                viskores::Id& first,
                                viskores::Id& count) const
  {
    const viskores::Id2 key(viskores::Min(edge[0], edge[1]), viskores::Max(edge[0], edge[1]));
    viskores::Id lo = LowerBound(copyKeys, 0, copyKeys.GetNumberOfValues(), key);
    first = lo;
    count = 0;
    while (lo + count < copyKeys.GetNumberOfValues() && copyKeys.Get(lo + count) == key)
    {
      ++count;
    }
  }
};

/// A relabel of a candidate, from a copy of its edge of `X` in one sheet: the cluster of the copy
/// of the source joins that of the copy of the target. `(candidate, from, copy, to)`.
struct RelabelItem : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn item,
                                WholeArrayIn itemOffsets,
                                WholeArrayIn copyFirst,
                                WholeArrayIn edges,
                                WholeArrayIn copyEdges,
                                WholeArrayIn original,
                                WholeArrayIn labels,
                                FieldOut record);
  using ExecutionSignature = _8(_1, _2, _3, _4, _5, _6, _7);

  template <typename IdPortal, typename PairPortal>
  VISKORES_EXEC viskores::Vec<viskores::Id, 4> operator()(viskores::Id item,
                                                          const IdPortal& itemOffsets,
                                                          const IdPortal& copyFirst,
                                                          const PairPortal& edges,
                                                          const PairPortal& copyEdges,
                                                          const IdPortal& original,
                                                          const IdPortal& labels) const
  {
    const viskores::Id c = OwnerOf(itemOffsets, item);
    const viskores::Id k = item - itemOffsets.Get(c);
    viskores::Id2 e = copyEdges.Get(copyFirst.Get(c) + k);
    const viskores::Id target = edges.Get(c)[0];
    if (original.Get(e[0]) != target)
    {
      e = viskores::Id2(e[1], e[0]);
    }
    return viskores::Vec<viskores::Id, 4>(c, labels.Get(e[1]), k, labels.Get(e[0]));
  }
};

/// Whether a relabel sorted by (candidate, from, copy) is the last of its source, which wins.
struct LastRelabel : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn index,
                                WholeArrayIn records,
                                FieldOut last,
                                FieldOut relabel);
  using ExecutionSignature = void(_1, _2, _3, _4);

  template <typename RecordPortal>
  VISKORES_EXEC void operator()(viskores::Id i,
                                const RecordPortal& records,
                                viskores::UInt8& last,
                                viskores::Id3& relabel) const
  {
    const auto r = records.Get(i);
    last = (i + 1 == records.GetNumberOfValues() || records.Get(i + 1)[0] != r[0] ||
            records.Get(i + 1)[1] != r[1])
      ? 1
      : 0;
    relabel = viskores::Id3(r[0], r[1], r[3]);
  }
};

/// The glue groups of a candidate, as indices of the unique groups and as glue ids, and the
/// number of their labels.
struct AffectedGroups : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn groups,
                                WholeArrayIn copyOffsets,
                                WholeArrayIn labels,
                                WholeArrayIn glue,
                                WholeArrayIn uniqueGroups,
                                WholeArrayIn groupOffsets,
                                FieldOut indices,
                                FieldOut glueIds,
                                FieldOut count);
  using ExecutionSignature = void(_1, _2, _3, _4, _5, _6, _7, _8, _9);

  template <typename IdPortal>
  VISKORES_EXEC void operator()(const viskores::Id2& groups,
                                const IdPortal& copyOffsets,
                                const IdPortal& labels,
                                const IdPortal& glue,
                                const IdPortal& uniqueGroups,
                                const IdPortal& groupOffsets,
                                viskores::Id2& indices,
                                viskores::Id2& glueIds,
                                viskores::Id& count) const
  {
    count = 0;
    for (viskores::IdComponent i = 0; i < 2; ++i)
    {
      const viskores::Id g = glue.Get(labels.Get(copyOffsets.Get(groups[i])));
      glueIds[i] = g;
      indices[i] = FindSorted(uniqueGroups, g);
      count += groupOffsets.Get(indices[i] + 1) - groupOffsets.Get(indices[i]);
    }
  }
};

/// An affected label of a candidate: `(candidate, label, new label)`.
struct AffectedLabel : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn item,
                                WholeArrayIn itemOffsets,
                                WholeArrayIn indices,
                                WholeArrayIn groupOffsets,
                                WholeArrayIn groupLabels,
                                WholeArrayIn relabels,
                                WholeArrayIn relabelOffsets,
                                FieldOut record);
  using ExecutionSignature = _8(_1, _2, _3, _4, _5, _6, _7);

  template <typename IdPortal, typename PairPortal, typename RelabelPortal>
  VISKORES_EXEC viskores::Id3 operator()(viskores::Id item,
                                         const IdPortal& itemOffsets,
                                         const PairPortal& indices,
                                         const IdPortal& groupOffsets,
                                         const IdPortal& groupLabels,
                                         const RelabelPortal& relabels,
                                         const IdPortal& relabelOffsets) const
  {
    const viskores::Id c = OwnerOf(itemOffsets, item);
    viskores::Id k = item - itemOffsets.Get(c);
    const viskores::Id2 g = indices.Get(c);
    const viskores::Id n1 = groupOffsets.Get(g[0] + 1) - groupOffsets.Get(g[0]);
    const viskores::Id label = (k < n1) ? groupLabels.Get(groupOffsets.Get(g[0]) + k)
                                        : groupLabels.Get(groupOffsets.Get(g[1]) + k - n1);
    return viskores::Id3(c, label, region_detail::NewLabelOf(relabels, relabelOffsets, c, label));
  }
};

/// The record `(candidate, cluster, vertex)` of a vertex in a cluster the candidate changes, -1
/// elsewhere.
struct MergedMember : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn record,
                                WholeArrayIn relabels,
                                WholeArrayIn relabelOffsets,
                                FieldOut member);
  using ExecutionSignature = _4(_1, _2, _3);

  template <typename RelabelPortal, typename IdPortal>
  VISKORES_EXEC viskores::Id3 operator()(const viskores::Vec<viskores::Id, 4>& r,
                                         const RelabelPortal& relabels,
                                         const IdPortal& relabelOffsets) const
  {
    return region_detail::IsTargetOf(relabels, relabelOffsets, r[0], r[3])
      ? viskores::Id3(r[0], r[3], r[1])
      : viskores::Id3(-1);
  }
};

struct MemberKey : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn member, FieldOut key, FieldOut vertex);
  using ExecutionSignature = void(_1, _2, _3);

  VISKORES_EXEC void operator()(const viskores::Id3& m,
                                viskores::Id2& key,
                                viskores::Id& vertex) const
  {
    key = viskores::Id2(m[0], m[1]);
    vertex = m[2];
  }
};

/// The first frozen member of a changed cluster, or -1.
struct MergedFrozen : viskores::worklet::WorkletReduceByKey
{
  using ControlSignature = void(KeysIn key,
                                ValuesIn vertex,
                                WholeArrayIn frozen,
                                ReducedValuesOut first);
  using ExecutionSignature = void(_2, _3, _4);

  template <typename VertexVec, typename FlagPortal>
  VISKORES_EXEC void operator()(const VertexVec& vertices,
                                const FlagPortal& frozen,
                                viskores::Id& first) const
  {
    first = -1;
    for (viskores::IdComponent i = 0; i < vertices.GetNumberOfComponents(); ++i)
    {
      if (frozen.Get(vertices[i]) && first < 0)
      {
        first = vertices[i];
      }
    }
  }
};

/// The generator `Place` gives a cluster, `o + Q^-1 b`, before the curves move it.
struct RawGenerator : viskores::worklet::WorkletMapField
{
  using ControlSignature =
    void(FieldIn anchor, FieldIn q, FieldIn b, WholeArrayIn points, FieldOut generator);
  using ExecutionSignature = _5(_1, _2, _3, _4);

  template <typename VecPortal>
  VISKORES_EXEC Vec3
  operator()(viskores::Id anchor, const Mat3& q, const Vec3& b, const VecPortal& points) const
  {
    return Add(points.Get(anchor), SolveSPD(q, b));
  }
};

/// A changed cluster as if no sheet vertex had joined it: placed from its frozen members alone
/// if it holds other vertices too, by `Place` otherwise.
struct MergedPiece : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn key,
                                FieldIn generator,
                                FieldIn q,
                                FieldIn size,
                                WholeArrayIn frozenKeys,
                                WholeArrayIn frozenGenerators,
                                WholeArrayIn frozenQ,
                                WholeArrayIn frozenSizes,
                                FieldOut cp,
                                FieldOut qp);
  using ExecutionSignature = void(_1, _2, _3, _4, _5, _6, _7, _8, _9, _10);

  template <typename KeyPortal, typename VecPortal, typename MatPortal, typename IdPortal>
  VISKORES_EXEC void operator()(const viskores::Id2& key,
                                const Vec3& generator,
                                const Mat3& q,
                                viskores::Id size,
                                const KeyPortal& frozenKeys,
                                const VecPortal& frozenGenerators,
                                const MatPortal& frozenQ,
                                const IdPortal& frozenSizes,
                                Vec3& cp,
                                Mat3& qp) const
  {
    const viskores::Id i = FindSorted(frozenKeys, key);
    if (i >= 0 && size > frozenSizes.Get(i))
    {
      cp = frozenGenerators.Get(i);
      qp = frozenQ.Get(i);
    }
    else
    {
      cp = generator;
      qp = q;
    }
  }
};

/// The values of an affected label of a candidate on the candidate's labelling: from its merged
/// cluster if the candidate changes it, its own otherwise, with the pieces placed as if no sheet
/// vertex had joined them (`cp`, `qp`); its first frozen member and its size.
struct LabelValues : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn record,
                                WholeArrayIn mergedKeys,
                                WholeArrayIn mergedCp,
                                WholeArrayIn mergedQp,
                                WholeArrayIn mergedSizes,
                                WholeArrayIn mergedFrozenFirst,
                                WholeArrayIn cp,
                                WholeArrayIn qp,
                                WholeArrayIn sizes,
                                WholeArrayIn frozenFirst,
                                FieldOut generator,
                                FieldOut matrix,
                                FieldOut firstFrozen,
                                FieldOut size);
  using ExecutionSignature = void(_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14);

  template <typename KeyPortal, typename VecPortal, typename MatPortal, typename IdPortal>
  VISKORES_EXEC void operator()(const viskores::Id3& r,
                                const KeyPortal& mergedKeys,
                                const VecPortal& mergedCp,
                                const MatPortal& mergedQp,
                                const IdPortal& mergedSizes,
                                const IdPortal& mergedFrozenFirst,
                                const VecPortal& cp,
                                const MatPortal& qp,
                                const IdPortal& sizes,
                                const IdPortal& frozenFirst,
                                Vec3& generator,
                                Mat3& matrix,
                                viskores::Id& firstFrozen,
                                viskores::Id& size) const
  {
    const viskores::Id label = r[1];
    const viskores::Id i = FindSorted(mergedKeys, viskores::Id2(r[0], label));
    if (i >= 0)
    {
      generator = mergedCp.Get(i);
      matrix = mergedQp.Get(i);
      size = mergedSizes.Get(i);
      firstFrozen = mergedFrozenFirst.Get(i);
    }
    else
    {
      generator = cp.Get(label);
      matrix = qp.Get(label);
      size = sizes.Get(label);
      firstFrozen = frozenFirst.Get(label);
    }
  }
};

/// The generator of the merged glue group of every candidate, as `GroupPlacement` on the
/// candidate's labelling: `KeepGenerators` keeps the clusters' generators, `AtJunction` places
/// it at a junction, `AtMinimiser` at the shared minimiser of the pieces placed as if no sheet
/// vertex had joined them.
struct CandidateGroupPlacement : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn candidate,
                                WholeArrayIn labelRecords,
                                WholeArrayIn labelOffsets,
                                WholeArrayIn labelGenerators,
                                WholeArrayIn labelQ,
                                WholeArrayIn labelFrozenFirst,
                                WholeArrayIn original,
                                WholeArrayIn segment,
                                WholeArrayIn regular,
                                WholeArrayIn copyOffsets,
                                WholeArrayIn points,
                                WholeArrayIn moved,
                                WholeArrayIn movedTo,
                                WholeArrayIn segmentFirst,
                                FieldOut kind,
                                FieldOut generator);
  using ExecutionSignature =
    void(_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16);

  template <typename RecordPortal,
            typename IdPortal,
            typename VecPortal,
            typename MatPortal,
            typename FlagPortal>
  VISKORES_EXEC void operator()(viskores::Id c,
                                const RecordPortal& labelRecords,
                                const IdPortal& labelOffsets,
                                const VecPortal& labelGenerators,
                                const MatPortal& labelQ,
                                const IdPortal& labelFrozenFirst,
                                const IdPortal& original,
                                const IdPortal& segment,
                                const FlagPortal& regular,
                                const IdPortal& copyOffsets,
                                const VecPortal& points,
                                const IdPortal& moved,
                                const IdPortal& movedTo,
                                const IdPortal& segmentFirst,
                                viskores::IdComponent& kind,
                                Vec3& generator) const
  {
    // The labels of the merged group after the candidate, in increasing order, are the affected
    // labels that are not sources; the first of them leads.
    const viskores::Id begin = labelOffsets.Get(c), stop = labelOffsets.Get(c + 1);
    viskores::Id first = -1, count = 0;
    for (viskores::Id r = begin; r < stop; ++r)
    {
      const viskores::Id3 record = labelRecords.Get(r);
      if (record[1] == record[2])
      {
        first = (first < 0) ? r : first;
        ++count;
      }
    }
    kind = curves_detail::KeepGenerators;
    generator = labelGenerators.Get(first);
    const viskores::Id frozenMember = labelFrozenFirst.Get(first);
    if (frozenMember >= 0)
    {
      // The segment after the candidate: the moved segment's vertices have joined theirs.
      const viskores::Id x = original.Get(frozenMember);
      const viskores::Id s =
        (segment.Get(x) == segmentFirst.Get(moved.Get(c))) ? movedTo.Get(c) : segment.Get(x);
      if (!regular.Get(s))
      {
        kind = curves_detail::AtJunction;
        generator = points.Get(copyOffsets.Get(s));
        return;
      }
      kind = curves_detail::AtMinimiser;
    }
    if (count < 2)
    {
      return;
    }
    const Vec3 o = labelGenerators.Get(first);
    TreeSum<Mat3> qs;
    TreeSum<Vec3> bs;
    for (viskores::Id r = begin; r < stop; ++r)
    {
      const viskores::Id3 record = labelRecords.Get(r);
      if (record[1] != record[2])
      {
        continue;
      }
      qs.Push(labelQ.Get(r));
      bs.Push(MatVec(labelQ.Get(r), Sub(labelGenerators.Get(r), o)));
    }
    kind = curves_detail::AtMinimiser;
    generator = Add(o, SolveSPD(qs.Result(), bs.Result()));
  }
};

/// For an affected vertex: whether the candidate moves its point, and its costs before and
/// after.
struct VertexCosts : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn record,
                                WholeArrayIn kinds,
                                WholeArrayIn groupGenerators,
                                WholeArrayIn mergedKeys,
                                WholeArrayIn mergedGenerators,
                                WholeArrayIn generators,
                                WholeArrayIn rawGenerators,
                                WholeArrayIn points,
                                WholeArrayIn m,
                                FieldOut changed,
                                FieldOut before,
                                FieldOut after);
  using ExecutionSignature = void(_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12);

  template <typename KindPortal, typename VecPortal, typename KeyPortal, typename MatPortal>
  VISKORES_EXEC void operator()(const viskores::Vec<viskores::Id, 4>& r,
                                const KindPortal& kinds,
                                const VecPortal& groupGenerators,
                                const KeyPortal& mergedKeys,
                                const VecPortal& mergedGenerators,
                                const VecPortal& generators,
                                const VecPortal& rawGenerators,
                                const VecPortal& points,
                                const MatPortal& m,
                                viskores::UInt8& changed,
                                Real& before,
                                Real& after) const
  {
    const viskores::Id c = r[0], v = r[1];
    const Vec3 p = generators.Get(r[2]);
    Vec3 p2;
    if (kinds.Get(c) != curves_detail::KeepGenerators)
    {
      p2 = groupGenerators.Get(c);
    }
    else
    {
      const viskores::Id i = FindSorted(mergedKeys, viskores::Id2(c, r[3]));
      p2 = (i >= 0) ? mergedGenerators.Get(i) : rawGenerators.Get(r[3]);
    }
    // As numpy compares, by value.
    changed = (p[0] != p2[0] || p[1] != p2[1] || p[2] != p2[2]) ? 1 : 0;
    const Vec3 point = points.Get(v);
    const Mat3 mv = m.Get(v);
    before = Quad(Sub(p, point), mv);
    after = Quad(Sub(p2, point), mv);
  }
};

/// The generator of an affected label on the candidate's labelling: the generator of the merged
/// group, or its own from `Place`.
struct CandidateLabelGenerator : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn record,
                                FieldIn generator,
                                WholeArrayIn kinds,
                                WholeArrayIn groupGenerators,
                                FieldOut candidateGenerator);
  using ExecutionSignature = _5(_1, _2, _3, _4);

  template <typename KindPortal, typename VecPortal>
  VISKORES_EXEC Vec3 operator()(const viskores::Id3& r,
                                const Vec3& generator,
                                const KindPortal& kinds,
                                const VecPortal& groupGenerators) const
  {
    return (kinds.Get(r[0]) != curves_detail::KeepGenerators) ? groupGenerators.Get(r[0])
                                                              : generator;
  }
};

/// The increase of the objective of a candidate, clamped at zero.
struct Increase : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn after, FieldIn before, FieldOut increase);
  using ExecutionSignature = _3(_1, _2);

  VISKORES_EXEC Real operator()(Real after, Real before) const
  {
    return viskores::Max(after - before, Real(0));
  }
};

struct HalfCount : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn index, WholeArrayIn offsets, FieldOut half, FieldOut count);
  using ExecutionSignature = void(_1, _2, _3, _4);

  template <typename IdPortal>
  VISKORES_EXEC void operator()(viskores::Id c,
                                const IdPortal& offsets,
                                viskores::Id& half,
                                viskores::Id& count) const
  {
    count = offsets.Get(c + 1) - offsets.Get(c);
    half = (count + 1) / 2;
  }
};

/// The tree sum of the reference, one level: adjacent pairs of every candidate's values are
/// added, and an odd last value carried.
struct TreeLevel : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn index,
                                WholeArrayIn values,
                                WholeArrayIn offsets,
                                WholeArrayIn newOffsets,
                                FieldOut value);
  using ExecutionSignature = _5(_1, _2, _3, _4);

  template <typename RealPortal, typename IdPortal>
  VISKORES_EXEC Real operator()(viskores::Id i,
                                const RealPortal& values,
                                const IdPortal& offsets,
                                const IdPortal& newOffsets) const
  {
    const viskores::Id owner = OwnerOf(newOffsets, i);
    const viskores::Id j = i - newOffsets.Get(owner);
    const viskores::Id start = offsets.Get(owner), count = offsets.Get(owner + 1) - start;
    const viskores::Id a = start + 2 * j;
    return (2 * j + 1 < count) ? values.Get(a) + values.Get(a + 1) : values.Get(a);
  }
};

/// The result of every candidate's tree sum, 0 for a candidate without values.
struct TreeResult : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn candidate,
                                WholeArrayIn values,
                                WholeArrayIn offsets,
                                FieldOut sum);
  using ExecutionSignature = _4(_1, _2, _3);

  template <typename RealPortal, typename IdPortal>
  VISKORES_EXEC Real operator()(viskores::Id c,
                                const RealPortal& values,
                                const IdPortal& offsets) const
  {
    return (offsets.Get(c + 1) > offsets.Get(c)) ? values.Get(offsets.Get(c)) : Real(0);
  }
};

/// What candidate `c` changes in the output polyline of its arc: the head of the segment whose
/// point it removes, and that of the segment whose point moves, or -1. A segment merge removes
/// the point of the second segment and moves that of the first; a junction merge removes the
/// point of the end segment.
struct CurveChange : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn arc,
                                FieldIn kind,
                                FieldIn position,
                                WholeArrayIn segmentOffsets,
                                WholeArrayIn heads,
                                FieldOut removed,
                                FieldOut moved);
  using ExecutionSignature = void(_1, _2, _3, _4, _5, _6, _7);

  template <typename IdPortal>
  VISKORES_EXEC void operator()(viskores::Id a,
                                viskores::IdComponent kind,
                                viskores::Id position,
                                const IdPortal& segmentOffsets,
                                const IdPortal& heads,
                                viskores::Id& removed,
                                viskores::Id& moved) const
  {
    const viskores::Id fs = segmentOffsets.Get(a);
    const viskores::Id ns = segmentOffsets.Get(a + 1) - fs;
    if (kind == JunctionMergeKind)
    {
      removed = heads.Get(position == 0 ? fs : fs + ns - 1);
      moved = -1;
      return;
    }
    removed = heads.Get(fs + (position + 1) % ns);
    moved = heads.Get(fs + position);
  }
};

/// The order of the passing candidates: increase, arc, kind and position.
struct CandidateKey : viskores::worklet::WorkletMapField
{
  using ControlSignature =
    void(FieldIn increase, FieldIn arc, FieldIn kind, FieldIn position, FieldOut key);
  using ExecutionSignature = _5(_1, _2, _3, _4, WorkIndex);

  VISKORES_EXEC viskores::Pair<Real, viskores::Vec<viskores::Id, 4>> operator()(
    Real increase,
    viskores::Id arc,
    viskores::IdComponent kind,
    viskores::Id position,
    viskores::Id c) const
  {
    return viskores::Pair<Real, viskores::Vec<viskores::Id, 4>>(
      increase, viskores::Vec<viskores::Id, 4>(arc, kind, position, c));
  }
};

struct KeyCandidate : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn key, FieldOut candidate);
  using ExecutionSignature = _2(_1);

  VISKORES_EXEC viskores::Id operator()(
    const viskores::Pair<Real, viskores::Vec<viskores::Id, 4>>& key) const
  {
    return key.second[3];
  }
};

/// The incidences of a ranked candidate: `(key, rank, takes)` for every key it touches.
struct KeyIncidence : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn candidate,
                                WholeArrayIn touch,
                                WholeArrayIn take,
                                FieldOut records);
  using ExecutionSignature = void(_1, _2, _3, _4, WorkIndex);

  template <typename TouchPortal, typename TakePortal, typename RecordVec>
  VISKORES_EXEC void operator()(viskores::Id c,
                                const TouchPortal& touch,
                                const TakePortal& take,
                                RecordVec& records,
                                viskores::Id rank) const
  {
    const auto keys = touch.Get(c);
    const viskores::Id2 taken = take.Get(c);
    for (viskores::IdComponent i = 0; i < 5; ++i)
    {
      const viskores::Id key = keys[i];
      records[i] =
        viskores::Id3(key, rank, (key >= 0 && (key == taken[0] || key == taken[1])) ? 1 : 0);
    }
  }
};

struct IncidenceFields : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn record, FieldOut key, FieldOut rank, FieldOut takes);
  using ExecutionSignature = void(_1, _2, _3, _4);

  VISKORES_EXEC void operator()(const viskores::Id3& r,
                                viskores::Id& key,
                                viskores::Id& rank,
                                viskores::UInt8& takes) const
  {
    key = r[0];
    rank = r[1];
    takes = static_cast<viskores::UInt8>(r[2]);
  }
};

/// The states of a candidate in the greedy commit.
enum : viskores::UInt8
{
  Undecided = 0,
  Accepted = 1,
  Rejected = 2
};

/// One pass of the greedy commit in rank order: a candidate is rejected if an earlier accepted
/// candidate takes a key it touches, and accepted once every earlier candidate taking such a
/// key is rejected.
struct GreedyPass : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn candidate,
                                FieldIn state,
                                WholeArrayIn touch,
                                WholeArrayIn keys,
                                WholeArrayIn ranks,
                                WholeArrayIn takes,
                                WholeArrayIn states,
                                FieldOut newState);
  using ExecutionSignature = void(_1, _2, _3, _4, _5, _6, _7, _8, WorkIndex);

  template <typename TouchPortal, typename IdPortal, typename FlagPortal, typename StatePortal>
  VISKORES_EXEC void operator()(viskores::Id c,
                                viskores::UInt8 state,
                                const TouchPortal& touch,
                                const IdPortal& keys,
                                const IdPortal& ranks,
                                const FlagPortal& takes,
                                const StatePortal& states,
                                viskores::UInt8& newState,
                                viskores::Id rank) const
  {
    newState = state;
    if (state != Undecided)
    {
      return;
    }
    bool waiting = false;
    const auto touched = touch.Get(c);
    for (viskores::IdComponent i = 0; i < 5; ++i)
    {
      const viskores::Id key = touched[i];
      if (key < 0)
      {
        continue;
      }
      viskores::Id lo = LowerBound(keys, 0, keys.GetNumberOfValues(), key);
      for (viskores::Id r = lo; r < keys.GetNumberOfValues() && keys.Get(r) == key; ++r)
      {
        if (ranks.Get(r) >= rank || !takes.Get(r))
        {
          continue;
        }
        const viskores::UInt8 other = states.Get(ranks.Get(r));
        if (other == Accepted)
        {
          newState = Rejected;
          return;
        }
        waiting = waiting || other == Undecided;
      }
    }
    newState = waiting ? Undecided : Accepted;
  }
};

/// Whether a relabel belongs to a selected candidate, as a pair `(to, from)` of the merge phase.
struct SelectedRelabel : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn relabel,
                                WholeArrayIn selected,
                                FieldOut keep,
                                FieldOut pair);
  using ExecutionSignature = void(_1, _2, _3, _4);

  template <typename FlagPortal>
  VISKORES_EXEC void operator()(const viskores::Id3& r,
                                const FlagPortal& selected,
                                viskores::UInt8& keep,
                                viskores::Id2& pair) const
  {
    keep = selected.Get(r[0]);
    pair = viskores::Id2(r[2], r[1]);
  }
};

/// A selected segment merge removes the head of the merged segment; a closed arc whose first
/// segment merged into its last starts at its next head. A selected junction merge adds the
/// length of its segment at its end.
struct ApplySegmentChange : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn candidate,
                                WholeArrayIn arcs,
                                WholeArrayIn kinds,
                                WholeArrayIn positions,
                                WholeArrayIn moved,
                                WholeArrayIn heads,
                                WholeArrayIn lengths,
                                WholeArrayIn segmentOffsets,
                                WholeArrayInOut segmentHead,
                                WholeArrayInOut firstHead,
                                WholeArrayInOut absorbedFront,
                                WholeArrayInOut absorbedBack);
  using ExecutionSignature = void(_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12);

  template <typename IdPortal, typename KindPortal, typename HeadFlagPortal, typename OutPortal>
  VISKORES_EXEC void operator()(viskores::Id c,
                                const IdPortal& arcs,
                                const KindPortal& kinds,
                                const IdPortal& positions,
                                const IdPortal& moved,
                                const IdPortal& heads,
                                const IdPortal& lengths,
                                const IdPortal& segmentOffsets,
                                HeadFlagPortal& segmentHead,
                                OutPortal& firstHead,
                                OutPortal& absorbedFront,
                                OutPortal& absorbedBack) const
  {
    const viskores::Id a = arcs.Get(c);
    const viskores::Id s = moved.Get(c);
    if (kinds.Get(c) == JunctionMergeKind)
    {
      // At most one junction merge per end of an arc.
      if (positions.Get(c) == 0)
      {
        absorbedFront.Set(a, lengths.Get(s));
      }
      else
      {
        absorbedBack.Set(a, lengths.Get(s));
      }
      return;
    }
    segmentHead.Set(heads.Get(s), 0);
    const viskores::Id fs = segmentOffsets.Get(a);
    if (s == fs)
    {
      // The first segment joined the last: the list starts at the next.
      firstHead.Set(a, heads.Get(fs + 1));
    }
  }
};

/// The absorbed counts of an arc, increased by those of its selected junction merges.
struct AddAbsorbed : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn front, FieldIn back, FieldInOut absorbed);
  using ExecutionSignature = void(_1, _2, _3);

  VISKORES_EXEC void operator()(viskores::Id front,
                                viskores::Id back,
                                viskores::Id2& absorbed) const
  {
    absorbed[0] += front;
    absorbed[1] += back;
  }
};

/// The vertices of a moved segment join their new segment or junction.
struct MoveSegmentVertex : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn item,
                                WholeArrayIn itemOffsets,
                                WholeArrayIn selected,
                                WholeArrayIn moved,
                                WholeArrayIn movedTo,
                                WholeArrayIn arcs,
                                WholeArrayIn heads,
                                WholeArrayIn arcOffsets,
                                WholeArrayIn arcVertices,
                                WholeArrayOut segment);
  using ExecutionSignature = void(_1, _2, _3, _4, _5, _6, _7, _8, _9, _10);

  template <typename IdPortal, typename OutPortal>
  VISKORES_EXEC void operator()(viskores::Id item,
                                const IdPortal& itemOffsets,
                                const IdPortal& selected,
                                const IdPortal& moved,
                                const IdPortal& movedTo,
                                const IdPortal& arcs,
                                const IdPortal& heads,
                                const IdPortal& arcOffsets,
                                const IdPortal& arcVertices,
                                OutPortal& segment) const
  {
    const viskores::Id i = OwnerOf(itemOffsets, item);
    const viskores::Id k = item - itemOffsets.Get(i);
    const viskores::Id c = selected.Get(i);
    const viskores::Id s = moved.Get(c);
    const viskores::Id a = arcs.Get(c);
    const viskores::Id start = arcOffsets.Get(a), length = arcOffsets.Get(a + 1) - start;
    const viskores::Id p = start + (heads.Get(s) - start + k) % length;
    segment.Set(arcVertices.Get(p), movedTo.Get(c));
  }
};

/// The key vertex of a candidate: for a segment merge, the first vertex of the segment that
/// stays, whose copies are in the merged pieces; for a junction merge, the first vertex of the
/// absorbed segment.
template <typename KindPortal, typename IdPortal>
VISKORES_EXEC inline viskores::Id KeyVertex(viskores::Id c,
                                            const KindPortal& kinds,
                                            const IdPortal& moved,
                                            const IdPortal& movedTo,
                                            const IdPortal& segmentFirst)
{
  return (kinds.Get(c) == JunctionMergeKind) ? segmentFirst.Get(moved.Get(c)) : movedTo.Get(c);
}

/// Whether a candidate of a batch that failed the full test is a culprit: a frozen copy of its
/// key vertex, in the labelling of the batch, is in a flagged cluster.
struct SegmentCulprit : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn candidate,
                                WholeArrayIn kinds,
                                WholeArrayIn moved,
                                WholeArrayIn movedTo,
                                WholeArrayIn segmentFirst,
                                WholeArrayIn copyOffsets,
                                WholeArrayIn frozen,
                                WholeArrayIn labels,
                                WholeArrayIn flags,
                                FieldOut culprit);
  using ExecutionSignature = _10(_1, _2, _3, _4, _5, _6, _7, _8, _9);

  template <typename KindPortal, typename IdPortal, typename FlagPortal>
  VISKORES_EXEC viskores::UInt8 operator()(viskores::Id c,
                                           const KindPortal& kinds,
                                           const IdPortal& moved,
                                           const IdPortal& movedTo,
                                           const IdPortal& segmentFirst,
                                           const IdPortal& copyOffsets,
                                           const FlagPortal& frozen,
                                           const IdPortal& labels,
                                           const FlagPortal& flags) const
  {
    const viskores::Id key = KeyVertex(c, kinds, moved, movedTo, segmentFirst);
    const viskores::Id end = (key + 1 < copyOffsets.GetNumberOfValues())
      ? copyOffsets.Get(key + 1)
      : labels.GetNumberOfValues();
    for (viskores::Id x = copyOffsets.Get(key); x < end; ++x)
    {
      if (frozen.Get(x) && flags.Get(labels.Get(x)))
      {
        return 1;
      }
    }
    return 0;
  }
};

/// Marks the clusters a candidate changes, those of the frozen copies of its key vertex in the
/// labelling of the batch, which `SegmentCulprit` reads.
struct MarkKeyClusters : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn candidate,
                                WholeArrayIn kinds,
                                WholeArrayIn moved,
                                WholeArrayIn movedTo,
                                WholeArrayIn segmentFirst,
                                WholeArrayIn copyOffsets,
                                WholeArrayIn frozen,
                                WholeArrayIn labels,
                                AtomicArrayInOut changed);
  using ExecutionSignature = void(_1, _2, _3, _4, _5, _6, _7, _8, _9);

  template <typename KindPortal, typename IdPortal, typename FlagPortal, typename AtomicPortal>
  VISKORES_EXEC void operator()(viskores::Id c,
                                const KindPortal& kinds,
                                const IdPortal& moved,
                                const IdPortal& movedTo,
                                const IdPortal& segmentFirst,
                                const IdPortal& copyOffsets,
                                const FlagPortal& frozen,
                                const IdPortal& labels,
                                const AtomicPortal& changed) const
  {
    const viskores::Id key = KeyVertex(c, kinds, moved, movedTo, segmentFirst);
    const viskores::Id end = (key + 1 < copyOffsets.GetNumberOfValues())
      ? copyOffsets.Get(key + 1)
      : labels.GetNumberOfValues();
    for (viskores::Id x = copyOffsets.Get(key); x < end; ++x)
    {
      if (frozen.Get(x))
      {
        changed.Set(labels.Get(x), 1);
      }
    }
  }
};

/// 1 where a mark is set.
struct MarkedFlag : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn mark, FieldOut flag);
  using ExecutionSignature = _2(_1);

  VISKORES_EXEC viskores::UInt8 operator()(viskores::UInt32 mark) const
  {
    return mark != 0 ? 1 : 0;
  }
};

} // namespace segment_detail

/// The merges of consecutive segments of the singular curves, (f) of Proposition "Decimating
/// the singular curves", and with junction merges of end segments into their junctions, as
/// `ArcVQC` of proofs/lib/arcs.py.
class SegmentMerger
{
public:
  SegmentMerger(const SurfaceMesh& mesh,
                const VertexQuadrics& quadrics,
                const DistanceMesh& inputDistance,
                const MergeParameters& parameters,
                SingularCurves& curves)
    : Mesh(mesh)
    , Quadrics(quadrics)
    , InputDistance(inputDistance)
    , Parameters(parameters)
    , Curves(curves)
  {
    this->Parameters.Curves = &curves;
    this->CopyOffsets = FirstCopies(parameters.Original, curves.Segment.GetNumberOfValues());
  }

  /// The merge phase of the proposition: the merge phase of the theorem, alternating with rounds
  /// of merges of consecutive segments until neither changes the labelling, as
  /// `ArcVQC.merge_phase`.
  template <typename RoundObserver>
  void Run(Clustering& clusters, MergeDiagnostics& diagnostics, RoundObserver&& observer)
  {
    auto sheets = [&]
    {
      MergePhase(this->Mesh,
                 this->Quadrics,
                 this->InputDistance,
                 this->Parameters,
                 clusters,
                 diagnostics,
                 observer);
    };
    sheets();
    while (true)
    {
      bool segmentsMerged = false;
      while (this->Round(clusters, diagnostics))
      {
        segmentsMerged = true;
      }
      const viskores::Id before = clusters.NumberOfClusters;
      sheets();
      if (clusters.NumberOfClusters != before)
      {
        continue;
      }
      // The sheet merges changed nothing: one more round of segment merges decides.
      if (!segmentsMerged || !this->Round(clusters, diagnostics))
      {
        break;
      }
    }
  }

private:
  /// One round, as `arc_merge_round`: every candidate that keeps every flag quiet is ranked by
  /// the increase of the objective, then by arc, kind and position; the greedy set of those that
  /// share no segment or junction and are not consecutive is applied and checked, and if a flag
  /// fires, chosen again without the deferred candidates. Returns whether the labelling changed.
  bool Round(Clustering& clusters, MergeDiagnostics& diagnostics)
  {
    using Algorithm = viskores::cont::Algorithm;
    namespace sd = segment_detail;
    viskores::cont::Invoker invoke;
    const SegmentList segments = this->BuildSegments();
    const CandidateList candidates = this->BuildCandidates(segments);
    const viskores::Id numCandidates = candidates.Arc.GetNumberOfValues();
    if (numCandidates == 0)
    {
      return false;
    }
    const viskores::cont::ArrayHandle<viskores::Id3> relabels =
      this->BuildRelabels(candidates, clusters);
    viskores::cont::ArrayHandle<viskores::UInt8> fires;
    viskores::cont::ArrayHandle<Real> increases;
    this->Evaluate(candidates, segments, relabels, clusters, fires, increases);
    diagnostics.SegmentCandidates += numCandidates;

    // The passing candidates, ranked.
    viskores::cont::ArrayHandle<viskores::Pair<Real, viskores::Vec<viskores::Id, 4>>> keys, passing;
    invoke(
      sd::CandidateKey{}, increases, candidates.Arc, candidates.Kind, candidates.Position, keys);
    Algorithm::CopyIf(keys, fires, passing, IsZero{});
    if (passing.GetNumberOfValues() == 0)
    {
      return false;
    }
    Algorithm::Sort(passing);
    viskores::cont::ArrayHandle<viskores::Id> ranked;
    invoke(sd::KeyCandidate{}, passing, ranked);
    // A batch that fails the full test defers its culprits (`SegmentCulprit`, with the region
    // blame those `RegionFlags` names too) and the greedy set is chosen again without them; when
    // none or all are culprits, its first candidate is tried alone, and with `DeferFlagged`
    // deferred if it fails. `trimmed` marks the candidates deferred only as culprits of a batch.
    viskores::cont::ArrayHandle<viskores::UInt8> deferred, trimmed;
    deferred.AllocateAndFill(numCandidates, 0);
    trimmed.AllocateAndFill(numCandidates, 0);
    while (true)
    {
      viskores::cont::ArrayHandle<viskores::Id> live;
      Algorithm::CopyIf(
        ranked, viskores::cont::make_ArrayHandlePermutation(ranked, deferred), live, IsZero{});
      const viskores::cont::ArrayHandle<viskores::Id> accepted = this->Greedy(candidates, live);
      const viskores::Id numAccepted = accepted.GetNumberOfValues();
      if (numAccepted == 0)
      {
        // Before the round merges nothing, the candidates deferred only as culprits of a batch
        // are tried alone, in priority order, as in the merge phase.
        viskores::cont::ArrayHandle<viskores::Id> retry;
        Algorithm::CopyIf(
          ranked, viskores::cont::make_ArrayHandlePermutation(ranked, trimmed), retry);
        for (viskores::Id t = 0; t < retry.GetNumberOfValues(); ++t)
        {
          viskores::cont::ArrayHandle<viskores::Id> one;
          Algorithm::Copy(viskores::cont::make_ArrayHandleView(retry, t, 1), one);
          if (this->Apply(candidates, segments, relabels, one, clusters))
          {
            ++diagnostics.SegmentRounds;
            ++diagnostics.SegmentRetried;
            return true;
          }
        }
        return false;
      }
      viskores::cont::ArrayHandle<viskores::UInt8> culprits, blamed;
      const bool blame = this->Parameters.DeferFlagged && numAccepted > 1;
      if (this->Apply(candidates,
                      segments,
                      relabels,
                      accepted,
                      clusters,
                      blame ? &culprits : nullptr,
                      blame ? &blamed : nullptr))
      {
        ++diagnostics.SegmentRounds;
        return true;
      }
      if (blame)
      {
        const viskores::Id numCulprits = Algorithm::Reduce(culprits, viskores::Id(0));
        if (numCulprits > 0 && numCulprits < numAccepted)
        {
          viskores::cont::ArrayHandle<viskores::Id> named;
          Algorithm::CopyIf(accepted, culprits, named);
          invoke(MarkIndex{}, named, deferred);
          invoke(MarkIndex{}, named, trimmed);
          diagnostics.BlameDeferred += Algorithm::Reduce(blamed, viskores::Id(0));
          continue;
        }
      }
      viskores::cont::ArrayHandle<viskores::Id> first;
      Algorithm::Copy(viskores::cont::make_ArrayHandleView(accepted, 0, 1), first);
      // A batch of one has just failed alone.
      if (numAccepted > 1 && this->Apply(candidates, segments, relabels, first, clusters))
      {
        ++diagnostics.SegmentRounds;
        return true;
      }
      if (!this->Parameters.DeferFlagged)
      {
        return false;
      }
      invoke(MarkIndex{}, first, deferred);
      invoke(MarkIndex(0), first, trimmed);
    }
  }

  /// The segments of the arcs, in the order of their lists.
  SegmentList BuildSegments() const
  {
    using Algorithm = viskores::cont::Algorithm;
    namespace sd = segment_detail;
    viskores::cont::Invoker invoke;
    const SingularCurves& curves = this->Curves;
    const viskores::Id numPositions = curves.ArcVertices.GetNumberOfValues();
    SegmentList list;
    viskores::cont::ArrayHandle<viskores::Id> arcOfPosition;
    Algorithm::UpperBounds(
      viskores::cont::make_ArrayHandleView(curves.ArcOffsets, 1, curves.NumberOfArcs),
      viskores::cont::ArrayHandleIndex(numPositions),
      arcOfPosition);
    viskores::cont::ArrayHandle<viskores::Id3> records, heads;
    invoke(sd::HeadRecord{},
           viskores::cont::ArrayHandleIndex(numPositions),
           arcOfPosition,
           curves.SegmentHead,
           curves.ArcOffsets,
           curves.ArcEnds,
           curves.FirstHead,
           curves.Absorbed,
           records);
    viskores::cont::ArrayHandle<viskores::UInt8> valid;
    invoke(IsValidRecord{}, records, valid);
    Algorithm::CopyIf(records, valid, heads);
    Algorithm::Sort(heads);
    invoke(sd::HeadOfRecord{}, heads, list.Arc, list.Head);
    Algorithm::LowerBounds(
      list.Arc, viskores::cont::ArrayHandleIndex(curves.NumberOfArcs + 1), list.Offsets);
    invoke(sd::SegmentExtent{},
           list.Arc,
           list.Head,
           list.Head,
           list.Offsets,
           curves.ArcOffsets,
           curves.ArcEnds,
           curves.Absorbed,
           curves.ArcVertices,
           list.First,
           list.Last,
           list.Length);
    return list;
  }

  /// The candidates of a round, from the segment lists.
  CandidateList BuildCandidates(const SegmentList& segments) const
  {
    using Algorithm = viskores::cont::Algorithm;
    namespace sd = segment_detail;
    viskores::cont::Invoker invoke;
    const SingularCurves& curves = this->Curves;
    const bool junctionMerges = this->Parameters.JunctionMerges;
    viskores::cont::ArrayHandle<viskores::Id> counts, offsets;
    invoke(sd::CandidateCount(junctionMerges),
           viskores::cont::ArrayHandleIndex(curves.NumberOfArcs),
           segments.Offsets,
           curves.ArcEnds,
           counts);
    Algorithm::ScanExtended(counts, offsets);
    const viskores::Id total = viskores::cont::ArrayGetValue(curves.NumberOfArcs, offsets);
    CandidateList candidates;
    invoke(sd::CandidateFields(junctionMerges, segments.Head.GetNumberOfValues()),
           viskores::cont::ArrayHandleIndex(total),
           offsets,
           segments.Offsets,
           curves.ArcOffsets,
           curves.ArcEnds,
           curves.Absorbed,
           curves.ArcVertices,
           segments.First,
           segments.Last,
           candidates.Arc,
           candidates.Kind,
           candidates.Position,
           candidates.Edge,
           candidates.Moved,
           candidates.MovedTo,
           candidates.Groups,
           candidates.Touch,
           candidates.Take);
    return candidates;
  }

  /// The relabels of every candidate, `(candidate, from, to)`, sorted by candidate and source:
  /// in every sheet, the cluster of the copy of the source joins that of the copy of the target.
  viskores::cont::ArrayHandle<viskores::Id3> BuildRelabels(const CandidateList& candidates,
                                                           const Clustering& clusters) const
  {
    using Algorithm = viskores::cont::Algorithm;
    namespace sd = segment_detail;
    viskores::cont::Invoker invoke;
    viskores::cont::ArrayHandle<viskores::Id> copyFirst, copyCount, itemOffsets;
    invoke(sd::CopyRange{}, candidates.Edge, this->Curves.CopyKeys, copyFirst, copyCount);
    const viskores::Id numItems = Algorithm::ScanExclusive(copyCount, itemOffsets);
    viskores::cont::ArrayHandle<viskores::Vec<viskores::Id, 4>> records;
    invoke(sd::RelabelItem{},
           viskores::cont::ArrayHandleIndex(numItems),
           itemOffsets,
           copyFirst,
           candidates.Edge,
           this->Curves.CopyEdges,
           this->Parameters.Original,
           clusters.Labels,
           records);
    Algorithm::Sort(records);
    viskores::cont::ArrayHandle<viskores::UInt8> last;
    viskores::cont::ArrayHandle<viskores::Id3> all, relabels;
    invoke(sd::LastRelabel{}, viskores::cont::ArrayHandleIndex(numItems), records, last, all);
    Algorithm::CopyIf(all, last, relabels);
    return relabels;
  }

  /// The verdict (1 if a flag fires) and the increase of the objective of every candidate.
  void Evaluate(const CandidateList& candidates,
                const SegmentList& segments,
                const viskores::cont::ArrayHandle<viskores::Id3>& relabels,
                const Clustering& clusters,
                viskores::cont::ArrayHandle<viskores::UInt8>& fires,
                viskores::cont::ArrayHandle<Real>& increases) const
  {
    const viskores::cont::ArrayHandle<viskores::Id> glue = OutputGlue(this->Parameters, clusters);
    CandidateRegions regions;
    increases = this->Increases(candidates, segments, relabels, clusters, glue, regions);
    CurveCandidates curve;
    if (this->Parameters.CurveTolerance > 0)
    {
      viskores::cont::Invoker invoke;
      curve.Arc = candidates.Arc;
      invoke(segment_detail::CurveChange{},
             candidates.Arc,
             candidates.Kind,
             candidates.Position,
             segments.Offsets,
             segments.Head,
             curve.Removed,
             curve.Moved);
    }
    fires = CandidateFlags(this->Mesh,
                           this->InputDistance,
                           this->Parameters,
                           this->Curves,
                           clusters,
                           glue,
                           regions,
                           this->Parameters.CurveTolerance > 0 ? &curve : nullptr);
  }

  /// The labellings of the candidates, as what each changes, and the increase of the objective
  /// of every candidate, as `ArcVQC.increase`: over the vertices whose point the candidate
  /// moves, in increasing order, the tree sum of their costs after minus the tree sum before,
  /// clamped at zero.
  viskores::cont::ArrayHandle<Real> Increases(
    const CandidateList& candidates,
    const SegmentList& segments,
    const viskores::cont::ArrayHandle<viskores::Id3>& relabels,
    const Clustering& clusters,
    const viskores::cont::ArrayHandle<viskores::Id>& glue,
    CandidateRegions& regions) const
  {
    using Algorithm = viskores::cont::Algorithm;
    namespace sd = segment_detail;
    viskores::cont::Invoker invoke;
    const viskores::Id numCandidates = candidates.Arc.GetNumberOfValues();
    const viskores::cont::ArrayHandle<viskores::Id> relabelOffsets =
      RecordOffsets(relabels, numCandidates);
    viskores::worklet::Keys<viskores::Id> groups;
    groups.BuildArrays(glue, viskores::worklet::KeysSortType::Stable);

    // The labels of the two glue groups of every candidate.
    viskores::cont::ArrayHandle<viskores::Id2> groupIndices;
    viskores::cont::ArrayHandle<viskores::Id> labelCounts, labelItemOffsets;
    invoke(sd::AffectedGroups{},
           candidates.Groups,
           this->CopyOffsets,
           clusters.Labels,
           glue,
           groups.GetUniqueKeys(),
           groups.GetOffsets(),
           groupIndices,
           regions.GroupGlue,
           labelCounts);
    const viskores::Id numLabelItems = Algorithm::ScanExclusive(labelCounts, labelItemOffsets);
    viskores::cont::ArrayHandle<viskores::Id3> labelRecords;
    invoke(sd::AffectedLabel{},
           viskores::cont::ArrayHandleIndex(numLabelItems),
           labelItemOffsets,
           groupIndices,
           groups.GetOffsets(),
           groups.GetSortedValuesMap(),
           relabels,
           relabelOffsets,
           labelRecords);
    Algorithm::Sort(labelRecords);
    const viskores::cont::ArrayHandle<viskores::Id> labelOffsets =
      RecordOffsets(labelRecords, numCandidates);

    // Their vertices, by candidate and vertex.
    viskores::cont::ArrayHandle<viskores::Id> vertexCounts, vertexItemOffsets;
    invoke(GatherComponent<1>{}, labelRecords, clusters.Sizes, vertexCounts);
    const viskores::Id numVertexItems = Algorithm::ScanExclusive(vertexCounts, vertexItemOffsets);
    viskores::cont::ArrayHandle<viskores::Vec<viskores::Id, 4>> vertexRecords;
    invoke(detail::RegionVertex{},
           viskores::cont::ArrayHandleIndex(numVertexItems),
           vertexItemOffsets,
           labelRecords,
           clusters.Keys.GetOffsets(),
           clusters.Keys.GetSortedValuesMap(),
           vertexRecords);
    Algorithm::Sort(vertexRecords);

    // The clusters the candidates change, placed as Place would.
    viskores::cont::ArrayHandle<viskores::Id3> memberRecords, members;
    invoke(sd::MergedMember{}, vertexRecords, relabels, relabelOffsets, memberRecords);
    viskores::cont::ArrayHandle<viskores::UInt8> isMember;
    invoke(IsValidRecord{}, memberRecords, isMember);
    Algorithm::CopyIf(memberRecords, isMember, members);
    Algorithm::Sort(members);
    viskores::cont::ArrayHandle<viskores::Id2> memberKeys;
    viskores::cont::ArrayHandle<viskores::Id> memberVertices;
    invoke(sd::MemberKey{}, members, memberKeys, memberVertices);
    viskores::worklet::Keys<viskores::Id2> merged;
    merged.BuildArrays(memberKeys, viskores::worklet::KeysSortType::Stable);
    viskores::cont::ArrayHandle<viskores::Id> mergedAnchors, mergedSizes, mergedFrozenFirst;
    viskores::cont::ArrayHandle<Mat3> mergedQ;
    viskores::cont::ArrayHandle<Vec3> mergedB, mergedGenerators;
    viskores::cont::ArrayHandle<Real> mergedS;
    invoke(detail::Place{},
           merged,
           memberVertices,
           this->Mesh.Points,
           this->Quadrics.M,
           mergedAnchors,
           mergedQ,
           mergedB,
           mergedS,
           mergedGenerators,
           mergedSizes);
    invoke(sd::MergedFrozen{},
           merged,
           memberVertices,
           this->Parameters.FrozenVertices,
           mergedFrozenFirst);

    // The generators of `Place`, before the curves move them.
    viskores::cont::ArrayHandle<Vec3> rawGenerators;
    invoke(sd::RawGenerator{},
           clusters.Anchors,
           clusters.Q,
           clusters.B,
           this->Mesh.Points,
           rawGenerators);
    const viskores::cont::ArrayHandle<viskores::Id> frozenFirst =
      FrozenFirst(this->Parameters.FrozenVertices, clusters);
    // The pieces as if no sheet vertex had joined them: of the round, and the changed clusters.
    viskores::cont::ArrayHandle<Vec3> cp, mergedCp;
    viskores::cont::ArrayHandle<Mat3> qp, mergedQp;
    FrozenPiecePlacement(this->Mesh,
                         this->Quadrics.M,
                         this->Parameters.FrozenVertices,
                         clusters.Labels,
                         clusters.Sizes,
                         rawGenerators,
                         clusters.Q,
                         cp,
                         qp);
    {
      viskores::cont::ArrayHandle<viskores::UInt8> isFrozen;
      viskores::cont::ArrayHandle<viskores::Id3> frozenMembers;
      invoke(GatherComponent<2>{}, members, this->Parameters.FrozenVertices, isFrozen);
      Algorithm::CopyIf(members, isFrozen, frozenMembers);
      viskores::cont::ArrayHandle<viskores::Id2> frozenKeys, uniqueFrozenKeys;
      viskores::cont::ArrayHandle<viskores::Id> frozenMemberVertices, frozenAnchors, frozenSizes;
      viskores::cont::ArrayHandle<Mat3> frozenQ;
      viskores::cont::ArrayHandle<Vec3> frozenB, frozenGenerators;
      viskores::cont::ArrayHandle<Real> frozenS;
      if (frozenMembers.GetNumberOfValues() > 0)
      {
        invoke(sd::MemberKey{}, frozenMembers, frozenKeys, frozenMemberVertices);
        viskores::worklet::Keys<viskores::Id2> pieces;
        pieces.BuildArrays(frozenKeys, viskores::worklet::KeysSortType::Stable);
        invoke(detail::Place{},
               pieces,
               frozenMemberVertices,
               this->Mesh.Points,
               this->Quadrics.M,
               frozenAnchors,
               frozenQ,
               frozenB,
               frozenS,
               frozenGenerators,
               frozenSizes);
        uniqueFrozenKeys = pieces.GetUniqueKeys();
      }
      invoke(sd::MergedPiece{},
             merged.GetUniqueKeys(),
             mergedGenerators,
             mergedQ,
             mergedSizes,
             uniqueFrozenKeys,
             frozenGenerators,
             frozenQ,
             frozenSizes,
             mergedCp,
             mergedQp);
    }
    viskores::cont::ArrayHandle<Vec3> labelGenerators;
    viskores::cont::ArrayHandle<Mat3> labelQ;
    viskores::cont::ArrayHandle<viskores::Id> labelFrozenFirst, labelSizes;
    invoke(sd::LabelValues{},
           labelRecords,
           merged.GetUniqueKeys(),
           mergedCp,
           mergedQp,
           mergedSizes,
           mergedFrozenFirst,
           cp,
           qp,
           clusters.Sizes,
           frozenFirst,
           labelGenerators,
           labelQ,
           labelFrozenFirst,
           labelSizes);
    // The generator of every candidate's merged glue group.
    viskores::cont::ArrayHandle<viskores::IdComponent> kinds;
    viskores::cont::ArrayHandle<Vec3> groupGenerators;
    invoke(sd::CandidateGroupPlacement{},
           viskores::cont::ArrayHandleIndex(numCandidates),
           labelRecords,
           labelOffsets,
           labelGenerators,
           labelQ,
           labelFrozenFirst,
           this->Parameters.Original,
           this->Curves.Segment,
           this->Curves.Regular,
           this->CopyOffsets,
           this->Mesh.Points,
           candidates.Moved,
           candidates.MovedTo,
           segments.First,
           kinds,
           groupGenerators);

    // The costs of the moved vertices before and after, and their tree sums.
    viskores::cont::ArrayHandle<viskores::UInt8> changed;
    viskores::cont::ArrayHandle<Real> before, after;
    invoke(sd::VertexCosts{},
           vertexRecords,
           kinds,
           groupGenerators,
           merged.GetUniqueKeys(),
           mergedGenerators,
           clusters.Generators,
           rawGenerators,
           this->Mesh.Points,
           this->Quadrics.M,
           changed,
           before,
           after);
    viskores::cont::ArrayHandle<Vec3> candidateGenerators;
    invoke(sd::CandidateLabelGenerator{},
           labelRecords,
           labelGenerators,
           kinds,
           groupGenerators,
           candidateGenerators);
    regions.NumberOfCandidates = numCandidates;
    regions.Relabels = relabels;
    regions.RelabelOffsets = relabelOffsets;
    regions.Labels = labelRecords;
    regions.LabelOffsets = labelOffsets;
    regions.LabelGenerators = candidateGenerators;
    regions.LabelSizes = labelSizes;
    regions.Vertices = vertexRecords;
    regions.MemberKeys = memberKeys;
    regions.MemberVertices = memberVertices;
    regions.MergedSizes = mergedSizes;

    viskores::cont::ArrayHandle<viskores::Vec<viskores::Id, 4>> movedRecords;
    viskores::cont::ArrayHandle<Real> movedBefore, movedAfter;
    Algorithm::CopyIf(vertexRecords, changed, movedRecords);
    Algorithm::CopyIf(before, changed, movedBefore);
    Algorithm::CopyIf(after, changed, movedAfter);
    const viskores::cont::ArrayHandle<viskores::Id> movedOffsets =
      RecordOffsets(movedRecords, numCandidates);
    viskores::cont::ArrayHandle<Real> increases;
    invoke(sd::Increase{},
           this->TreeSums(movedAfter, movedOffsets, numCandidates),
           this->TreeSums(movedBefore, movedOffsets, numCandidates),
           increases);
    return increases;
  }

  /// The tree sum of the reference for every candidate, of `values` sorted by candidate with
  /// the offsets `offsets` (one more entry at the end).
  viskores::cont::ArrayHandle<Real> TreeSums(viskores::cont::ArrayHandle<Real> values,
                                             viskores::cont::ArrayHandle<viskores::Id> offsets,
                                             viskores::Id numCandidates) const
  {
    using Algorithm = viskores::cont::Algorithm;
    namespace sd = segment_detail;
    viskores::cont::Invoker invoke;
    while (true)
    {
      viskores::cont::ArrayHandle<viskores::Id> halves, counts, newOffsets;
      invoke(
        sd::HalfCount{}, viskores::cont::ArrayHandleIndex(numCandidates), offsets, halves, counts);
      if (Algorithm::Reduce(counts, viskores::Id(0), viskores::Maximum()) <= 1)
      {
        break;
      }
      Algorithm::ScanExtended(halves, newOffsets);
      const viskores::Id size = viskores::cont::ArrayGetValue(numCandidates, newOffsets);
      viskores::cont::ArrayHandle<Real> next;
      invoke(
        sd::TreeLevel{}, viskores::cont::ArrayHandleIndex(size), values, offsets, newOffsets, next);
      values = next;
      offsets = newOffsets;
    }
    viskores::cont::ArrayHandle<Real> sums;
    invoke(
      sd::TreeResult{}, viskores::cont::ArrayHandleIndex(numCandidates), values, offsets, sums);
    return sums;
  }

  /// The greedy commit in rank order: the ranked candidates accepted, in rank order.
  viskores::cont::ArrayHandle<viskores::Id> Greedy(
    const CandidateList& candidates,
    const viskores::cont::ArrayHandle<viskores::Id>& ranked) const
  {
    using Algorithm = viskores::cont::Algorithm;
    namespace sd = segment_detail;
    viskores::cont::Invoker invoke;
    const viskores::Id n = ranked.GetNumberOfValues();
    viskores::cont::ArrayHandle<viskores::Id3> incidence;
    invoke(sd::KeyIncidence{},
           ranked,
           candidates.Touch,
           candidates.Take,
           viskores::cont::make_ArrayHandleGroupVec<5>(incidence));
    Algorithm::Sort(incidence);
    viskores::cont::ArrayHandle<viskores::Id> keys, ranks;
    viskores::cont::ArrayHandle<viskores::UInt8> takes;
    invoke(sd::IncidenceFields{}, incidence, keys, ranks, takes);
    viskores::cont::ArrayHandle<viskores::UInt8> states;
    states.AllocateAndFill(n, sd::Undecided);
    while (true)
    {
      viskores::cont::ArrayHandle<viskores::UInt8> next;
      invoke(sd::GreedyPass{}, ranked, states, candidates.Touch, keys, ranks, takes, states, next);
      states = next;
      viskores::cont::ArrayHandle<viskores::UInt8> undecided;
      invoke(IsState(sd::Undecided), states, undecided);
      if (Algorithm::Reduce(undecided, viskores::Id(0)) == 0)
      {
        break;
      }
    }
    viskores::cont::ArrayHandle<viskores::UInt8> accepted;
    invoke(IsState(sd::Accepted), states, accepted);
    viskores::cont::ArrayHandle<viskores::Id> result;
    Algorithm::CopyIf(ranked, accepted, result);
    return result;
  }

  /// Applies the candidates `selected` and keeps them if no flag fires, as `apply` in
  /// `arc_merge_round`; if one fires, sets `culprits`, when given, per selected candidate.
  bool Apply(const CandidateList& candidates,
             const SegmentList& segments,
             const viskores::cont::ArrayHandle<viskores::Id3>& relabels,
             const viskores::cont::ArrayHandle<viskores::Id>& selected,
             Clustering& clusters,
             viskores::cont::ArrayHandle<viskores::UInt8>* culprits = nullptr,
             viskores::cont::ArrayHandle<viskores::UInt8>* blamed = nullptr)
  {
    using Algorithm = viskores::cont::Algorithm;
    namespace sd = segment_detail;
    viskores::cont::Invoker invoke;
    SingularCurves& curves = this->Curves;
    const viskores::Id numCandidates = candidates.Arc.GetNumberOfValues();
    // The state to restore.
    viskores::cont::ArrayHandle<viskores::Id> savedSegment, savedFirstHead;
    viskores::cont::ArrayHandle<viskores::UInt8> savedHead;
    viskores::cont::ArrayHandle<viskores::Id2> savedAbsorbed;
    Algorithm::Copy(curves.Segment, savedSegment);
    Algorithm::Copy(curves.FirstHead, savedFirstHead);
    Algorithm::Copy(curves.SegmentHead, savedHead);
    Algorithm::Copy(curves.Absorbed, savedAbsorbed);

    // The relabels of the selected candidates, as merges (to, from).
    viskores::cont::ArrayHandle<viskores::UInt8> isSelected, keep;
    isSelected.AllocateAndFill(numCandidates, 0);
    invoke(MarkIndex{}, selected, isSelected);
    viskores::cont::ArrayHandle<viskores::Id2> allPairs, pairs;
    invoke(sd::SelectedRelabel{}, relabels, isSelected, keep, allPairs);
    Algorithm::CopyIf(allPairs, keep, pairs);

    // Merge the segments, and absorb the end segments of junction merges.
    viskores::cont::ArrayHandle<viskores::Id> front, back;
    front.AllocateAndFill(curves.NumberOfArcs, 0);
    back.AllocateAndFill(curves.NumberOfArcs, 0);
    invoke(sd::ApplySegmentChange{},
           selected,
           candidates.Arc,
           candidates.Kind,
           candidates.Position,
           candidates.Moved,
           segments.Head,
           segments.Length,
           segments.Offsets,
           curves.SegmentHead,
           curves.FirstHead,
           front,
           back);
    invoke(sd::AddAbsorbed{}, front, back, curves.Absorbed);
    viskores::cont::ArrayHandle<viskores::Id> lengths, itemOffsets;
    invoke(GatherTwice{}, selected, candidates.Moved, segments.Length, lengths);
    const viskores::Id numItems = Algorithm::ScanExclusive(lengths, itemOffsets);
    invoke(sd::MoveSegmentVertex{},
           viskores::cont::ArrayHandleIndex(numItems),
           itemOffsets,
           selected,
           candidates.Moved,
           candidates.MovedTo,
           candidates.Arc,
           segments.Head,
           curves.ArcOffsets,
           curves.ArcVertices,
           curves.Segment);

    Clustering merged = ApplyMerges(this->Mesh, this->Quadrics, this->Parameters, clusters, pairs);
    const bool fires =
      FlagsFire(this->Mesh, this->Quadrics, this->InputDistance, merged, this->Parameters);
    if (fires)
    {
      if (culprits)
      {
        // The clusters the flags blame, those a violation of (D0)-(D2) names included, read
        // before the curves are restored: a candidate is a culprit if a frozen copy of its key
        // vertex is in one.
        const bool region = this->Parameters.Blame == MergeBlame::Region;
        viskores::cont::ArrayHandle<viskores::UInt8> flags;
        FailingItems items;
        blamed->AllocateAndFill(selected.GetNumberOfValues(), 0);
        if (FlaggedClusters(this->Mesh,
                            this->Quadrics,
                            this->InputDistance,
                            merged,
                            this->Parameters,
                            flags,
                            true,
                            nullptr,
                            nullptr,
                            region ? &items : nullptr))
        {
          invoke(sd::SegmentCulprit{},
                 selected,
                 candidates.Kind,
                 candidates.Moved,
                 candidates.MovedTo,
                 segments.First,
                 this->CopyOffsets,
                 this->Parameters.FrozenVertices,
                 merged.Labels,
                 flags,
                 *culprits);
          if (region)
          {
            // The region blame, with the clusters each candidate changes.
            viskores::cont::ArrayHandle<viskores::UInt32> marks;
            marks.AllocateAndFill(merged.NumberOfClusters, 0);
            invoke(sd::MarkKeyClusters{},
                   selected,
                   candidates.Kind,
                   candidates.Moved,
                   candidates.MovedTo,
                   segments.First,
                   this->CopyOffsets,
                   this->Parameters.FrozenVertices,
                   merged.Labels,
                   marks);
            viskores::cont::ArrayHandle<viskores::UInt8> changed, regionCulprit;
            invoke(sd::MarkedFlag{}, marks, changed);
            invoke(sd::SegmentCulprit{},
                   selected,
                   candidates.Kind,
                   candidates.Moved,
                   candidates.MovedTo,
                   segments.First,
                   this->CopyOffsets,
                   this->Parameters.FrozenVertices,
                   merged.Labels,
                   RegionFlags(this->Mesh, this->Parameters, clusters, merged, changed, items),
                   regionCulprit);
            invoke(detail::BlamedOnly{}, *culprits, regionCulprit, *blamed);
          }
        }
        else
        {
          culprits->AllocateAndFill(selected.GetNumberOfValues(), 0);
        }
      }
      curves.Segment = savedSegment;
      curves.FirstHead = savedFirstHead;
      curves.SegmentHead = savedHead;
      curves.Absorbed = savedAbsorbed;
      return false;
    }
    clusters = merged;
    return true;
  }

  const SurfaceMesh& Mesh;
  const VertexQuadrics& Quadrics;
  const DistanceMesh& InputDistance;
  MergeParameters Parameters;
  SingularCurves& Curves;
  /// The first copy of every input vertex.
  viskores::cont::ArrayHandle<viskores::Id> CopyOffsets;
};

}
}
} // namespace viskores::worklet::quadric_clustering

VISKORES_QUADRIC_CLUSTERING_STRICT_FP_END

#endif // viskores_worklet_quadric_clustering_SegmentMerges_h
