//============================================================================
//  The contents of this file are covered by the Viskores license. See
//  LICENSE.txt for details.
//
//  By contributing to this file, all contributors agree to the Developer
//  Certificate of Origin Version 1.1 (DCO 1.1) as stated in DCO.txt.
//============================================================================
#ifndef viskores_worklet_quadric_clustering_MergePhase_h
#define viskores_worklet_quadric_clustering_MergePhase_h

#include <viskores/filter/geometry_refinement/worklet/quadric_clustering/Clustering.h>
#include <viskores/filter/geometry_refinement/worklet/quadric_clustering/EmbeddingFlags.h>
#include <viskores/filter/geometry_refinement/worklet/quadric_clustering/EnvelopeFlags.h>
#include <viskores/filter/geometry_refinement/worklet/quadric_clustering/IsotopyFlags.h>
#include <viskores/filter/geometry_refinement/worklet/quadric_clustering/SingularCurves.h>
#include <viskores/filter/geometry_refinement/worklet/quadric_clustering/TopologyFlags.h>
#include <viskores/filter/geometry_refinement/worklet/quadric_clustering/VertexQuadrics.h>

#include <viskores/Pair.h>
#include <viskores/cont/Algorithm.h>
#include <viskores/cont/ArrayHandleGroupVec.h>
#include <viskores/cont/ArrayHandleIndex.h>
#include <viskores/cont/ArrayHandleView.h>
#include <viskores/cont/Timer.h>

#include <array>
#include <iterator>
#include <limits>
#include <map>
#include <set>
#include <vector>

VISKORES_QUADRIC_CLUSTERING_STRICT_FP_BEGIN

namespace viskores
{
namespace worklet
{
namespace quadric_clustering
{

/// The output complex `K` of a labelling, with the incidences the merge phase reads.
struct OutputComplex
{
  /// The output triangles, as label triples in the order and orientation of Algorithm 5.
  viskores::cont::ArrayHandle<viskores::Id3> Triangles;
  /// Its edges, sorted pairs in lexicographic order, and the triangles of each.
  viskores::cont::ArrayHandle<viskores::Id2> Edges;
  viskores::cont::ArrayHandle<viskores::IdComponent> EdgeCounts;
  /// Whether every vertex of `K` lies on its boundary.
  viskores::cont::ArrayHandle<viskores::UInt8> BoundaryVertex;
  bool HasBoundary = false;
  /// The neighbours of every vertex of `K`, in increasing order.
  viskores::cont::ArrayHandle<viskores::Id> NeighborOffsets;
  viskores::cont::ArrayHandle<viskores::Id> Neighbors;
  /// The sorted label triples of the triangles, in lexicographic order.
  viskores::cont::ArrayHandle<viskores::Id3> Faces;
  /// The triangles at every vertex of `K`, in increasing order.
  viskores::cont::ArrayHandle<viskores::Id> FanOffsets;
  viskores::cont::ArrayHandle<viskores::Id> Fans;
};

namespace detail
{

struct CountValues : viskores::worklet::WorkletReduceByKey
{
  using ControlSignature = void(KeysIn keys, ReducedValuesOut count);
  using ExecutionSignature = _2(ValueCount);
  using InputDomain = _1;

  VISKORES_EXEC viskores::IdComponent operator()(viskores::IdComponent count) const { return count; }
};

struct BoundaryEnds : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn edge, FieldIn count, FieldOut ends);
  using ExecutionSignature = void(_1, _2, _3);

  template <typename EndsVec>
  VISKORES_EXEC void operator()(const viskores::Id2& edge,
                                viskores::IdComponent count,
                                EndsVec& ends) const
  {
    ends[0] = (count == 1) ? edge[0] : viskores::Id(-1);
    ends[1] = (count == 1) ? edge[1] : viskores::Id(-1);
  }
};

struct SetListed : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn index, WholeArrayInOut flags);
  using ExecutionSignature = void(_1, _2);

  template <typename FlagsPortal>
  VISKORES_EXEC void operator()(viskores::Id index, FlagsPortal& flags) const
  {
    flags.Set(index, 1);
  }
};

struct EdgeRecords : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn edge, FieldOut records);
  using ExecutionSignature = void(_1, _2);

  template <typename RecordsVec>
  VISKORES_EXEC void operator()(const viskores::Id2& edge, RecordsVec& records) const
  {
    records[0] = viskores::Id2(edge[0], edge[1]);
    records[1] = viskores::Id2(edge[1], edge[0]);
  }
};

struct CornerRecords : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn triangle, FieldOut records);
  using ExecutionSignature = void(_1, _2, WorkIndex);

  template <typename RecordsVec>
  VISKORES_EXEC void operator()(const viskores::Id3& t, RecordsVec& records, viskores::Id index) const
  {
    for (int c = 0; c < 3; ++c)
    {
      records[c] = viskores::Id2(t[c], index);
    }
  }
};

VISKORES_EXEC_CONT inline viskores::Id3 SortedTriple(const viskores::Id3& t)
{
  viskores::Id a = t[0], b = t[1], c = t[2];
  if (a > b)
  {
    const viskores::Id x = a;
    a = b;
    b = x;
  }
  if (b > c)
  {
    const viskores::Id x = b;
    b = c;
    c = x;
  }
  if (a > b)
  {
    const viskores::Id x = a;
    a = b;
    b = x;
  }
  return viskores::Id3(a, b, c);
}

struct SortTriple : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn triangle, FieldOut sorted);
  using ExecutionSignature = _2(_1);

  VISKORES_EXEC viskores::Id3 operator()(const viskores::Id3& t) const { return SortedTriple(t); }
};

struct SplitTriple : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn triple, FieldOut first);
  using ExecutionSignature = _2(_1);

  VISKORES_EXEC viskores::Id operator()(const viskores::Id3& triple) const { return triple[0]; }
};

struct SplitRecord : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn record, FieldOut first, FieldOut second);
  using ExecutionSignature = void(_1, _2, _3);

  VISKORES_EXEC void operator()(const viskores::Id2& record,
                                viskores::Id& first,
                                viskores::Id& second) const
  {
    first = record[0];
    second = record[1];
  }
};

} // namespace detail

/// Sorts records `(key, value)` and returns, for keys `0, ..., numKeys - 1`, the offsets of their
/// runs and the values in order: a compressed adjacency list.
inline void BuildAdjacency(viskores::cont::ArrayHandle<viskores::Id2>& records,
                           viskores::Id numKeys,
                           viskores::cont::ArrayHandle<viskores::Id>& offsets,
                           viskores::cont::ArrayHandle<viskores::Id>& values)
{
  using Algorithm = viskores::cont::Algorithm;
  viskores::cont::Invoker invoke;
  Algorithm::Sort(records);
  viskores::cont::ArrayHandle<viskores::Id> keys;
  invoke(detail::SplitRecord{}, records, keys, values);
  Algorithm::LowerBounds(keys, viskores::cont::ArrayHandleIndex(numKeys + 1), offsets);
}

/// Builds the output complex of the output triangles `triangles` on `k` vertices.
inline OutputComplex BuildOutputComplex(const viskores::cont::ArrayHandle<viskores::Id3>& triangles,
                                        viskores::Id k)
{
  using Algorithm = viskores::cont::Algorithm;
  viskores::cont::Invoker invoke;
  OutputComplex complex;
  complex.Triangles = triangles;
  complex.BoundaryVertex.AllocateAndFill(k, 0);
  if (triangles.GetNumberOfValues() == 0)
  {
    complex.NeighborOffsets.AllocateAndFill(k + 1, 0);
    complex.FanOffsets.AllocateAndFill(k + 1, 0);
    return complex;
  }
  viskores::cont::ArrayHandle<viskores::Id2> sides;
  invoke(detail::TriangleEdges{}, triangles, viskores::cont::make_ArrayHandleGroupVec<3>(sides));
  viskores::worklet::Keys<viskores::Id2> edgeKeys;
  edgeKeys.BuildArrays(sides, viskores::worklet::KeysSortType::Unstable);
  complex.Edges = edgeKeys.GetUniqueKeys();
  invoke(detail::CountValues{}, edgeKeys, complex.EdgeCounts);

  viskores::cont::ArrayHandle<viskores::Id> ends, listed;
  invoke(detail::BoundaryEnds{},
         complex.Edges,
         complex.EdgeCounts,
         viskores::cont::make_ArrayHandleGroupVec<2>(ends));
  viskores::cont::ArrayHandle<viskores::UInt8> valid;
  invoke(detail::IsValidLabel{}, ends, valid);
  Algorithm::CopyIf(ends, valid, listed);
  complex.HasBoundary = listed.GetNumberOfValues() > 0;
  invoke(detail::SetListed{}, listed, complex.BoundaryVertex);

  viskores::cont::ArrayHandle<viskores::Id2> edgeRecords;
  invoke(detail::EdgeRecords{},
         complex.Edges,
         viskores::cont::make_ArrayHandleGroupVec<2>(edgeRecords));
  BuildAdjacency(edgeRecords, k, complex.NeighborOffsets, complex.Neighbors);

  viskores::cont::ArrayHandle<viskores::Id2> cornerRecords;
  invoke(detail::CornerRecords{},
         triangles,
         viskores::cont::make_ArrayHandleGroupVec<3>(cornerRecords));
  BuildAdjacency(cornerRecords, k, complex.FanOffsets, complex.Fans);

  invoke(detail::SortTriple{}, triangles, complex.Faces);
  Algorithm::Sort(complex.Faces);
  return complex;
}

/// The priority of a candidate merge `(i, j)`, smaller first: `max(D_0, deg i + deg j)`, the
/// degree sum of the two clusters with a floor (0 without the degree key), then the increase of
/// the objective, then the labels.
using MergePriority = viskores::Pair<viskores::Id, viskores::Pair<Real, viskores::Id2>>;

/// `Phi_C(c_C) = s_C - b_C^T Q_C^{-1} b_C`.
VISKORES_EXEC_CONT inline Real Phi(const Mat3& q, const Vec3& b, Real s)
{
  return s - Dot(b, SolveSPD(q, b));
}

/// The sums of `C_i` and `C_j` together, both moved to the anchor `o` of the union, the member of
/// smallest index: moving from `o_c` to `o` adds `Q_c (o_c - o)` to `b_c` and
/// `2 (o_c - o) . b_c + (o_c - o)^T Q_c (o_c - o)` to `s_c`.
template <typename IdPortal, typename VecPortal, typename MatPortal, typename RealPortal>
VISKORES_EXEC_CONT void MergedSums(viskores::Id i,
                                   viskores::Id j,
                                   const IdPortal& anchors,
                                   const VecPortal& points,
                                   const MatPortal& q,
                                   const VecPortal& b,
                                   const RealPortal& s,
                                   Vec3& o,
                                   Mat3& qm,
                                   Vec3& bm,
                                   Real& sm)
{
  const viskores::Id a = (anchors.Get(i) < anchors.Get(j)) ? i : j;
  o = points.Get(anchors.Get(a));
  qm = Mat3(0);
  bm = Vec3(0);
  sm = 0;
  for (int which = 0; which < 2; ++which)
  {
    const viskores::Id c = (which == 0) ? i : j;
    const Vec3 d = Sub(points.Get(anchors.Get(c)), o);
    const Mat3 qc = q.Get(c);
    const Vec3 bc = b.Get(c);
    qm = MatAdd(qm, qc);
    bm = Add(Add(bm, bc), MatVec(qc, d));
    sm = sm + ((s.Get(c) + 2 * Dot(d, bc)) + Quad(d, qc));
  }
}

namespace detail
{

/// Whether the edge `ij` of `K` is a candidate merge, by the local test of Proposition "A merge
/// is an edge contraction" (ii): not a linking edge, and in no non-facial 3-cycle; and its
/// priority, the increase of the objective, then the labels.
struct MergeCandidate : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn edge,
                                FieldIn count,
                                WholeArrayIn boundary,
                                WholeArrayIn neighborOffsets,
                                WholeArrayIn neighbors,
                                WholeArrayIn degreeOffsets,
                                WholeArrayIn faces,
                                WholeArrayIn frozenLabels,
                                WholeArrayIn contactOffsets,
                                WholeArrayIn contacts,
                                WholeArrayIn anchors,
                                WholeArrayIn points,
                                WholeArrayIn q,
                                WholeArrayIn b,
                                WholeArrayIn s,
                                FieldOut isCandidate,
                                FieldOut record);
  using ExecutionSignature =
    void(_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17);

  VISKORES_CONT MergeCandidate(bool topological, bool hasFrozen, viskores::Id degreeFloor)
    : Topological(topological)
    , HasFrozen(hasFrozen)
    , DegreeFloor(degreeFloor)
  {
  }

  template <typename FlagPortal,
            typename IdPortal,
            typename Id3Portal,
            typename VecPortal,
            typename MatPortal,
            typename RealPortal>
  VISKORES_EXEC void operator()(const viskores::Id2& edge,
                                viskores::IdComponent count,
                                const FlagPortal& boundary,
                                const IdPortal& neighborOffsets,
                                const IdPortal& neighbors,
                                const IdPortal& degreeOffsets,
                                const Id3Portal& faces,
                                const FlagPortal& frozenLabels,
                                const IdPortal& contactOffsets,
                                const Id3Portal& contacts,
                                const IdPortal& anchors,
                                const VecPortal& points,
                                const MatPortal& q,
                                const VecPortal& b,
                                const RealPortal& s,
                                viskores::UInt8& isCandidate,
                                MergePriority& record) const
  {
    const viskores::Id i = edge[0], j = edge[1];
    isCandidate = 0;
    record = MergePriority(0, viskores::Pair<Real, viskores::Id2>(0, edge));
    if (this->HasFrozen)
    {
      if (frozenLabels.Get(i) || frozenLabels.Get(j))
      {
        return;
      }
      // The glue condition: the merged cluster touches at most one copy of every frozen input
      // vertex. The contacts of each cluster are sorted by input vertex, then copy.
      viskores::Id a = contactOffsets.Get(i), aEnd = contactOffsets.Get(i + 1);
      viskores::Id c = contactOffsets.Get(j), cEnd = contactOffsets.Get(j + 1);
      viskores::Id vertex = -1, copy = -1;
      while (a < aEnd || c < cEnd)
      {
        viskores::Id3 next;
        if (c >= cEnd || (a < aEnd && contacts.Get(a)[1] <= contacts.Get(c)[1]))
        {
          next = contacts.Get(a++);
        }
        else
        {
          next = contacts.Get(c++);
        }
        if (next[1] == vertex && next[2] != copy)
        {
          return;
        }
        vertex = next[1];
        copy = next[2];
      }
    }
    if (this->Topological)
    {
      if (count == 2 && boundary.Get(i) && boundary.Get(j))
      {
        return; // a linking edge
      }
      // Every common neighbour x must make ijx a triangle of K.
      viskores::Id a = neighborOffsets.Get(i), aEnd = neighborOffsets.Get(i + 1);
      viskores::Id c = neighborOffsets.Get(j), cEnd = neighborOffsets.Get(j + 1);
      while (a < aEnd && c < cEnd)
      {
        const viskores::Id x = neighbors.Get(a), y = neighbors.Get(c);
        if (x < y)
        {
          ++a;
        }
        else if (y < x)
        {
          ++c;
        }
        else
        {
          viskores::Id3 triple(i, j, x);
          if (x < i)
          {
            triple = viskores::Id3(x, i, j);
          }
          else if (x < j)
          {
            triple = viskores::Id3(i, x, j);
          }
          viskores::Id lo = 0, hi = faces.GetNumberOfValues();
          while (lo < hi)
          {
            const viskores::Id mid = (lo + hi) / 2;
            if (faces.Get(mid) < triple)
            {
              lo = mid + 1;
            }
            else
            {
              hi = mid;
            }
          }
          if (lo == faces.GetNumberOfValues() || faces.Get(lo) != triple)
          {
            return; // a non-facial 3-cycle
          }
          ++a;
          ++c;
        }
      }
    }
    Vec3 o, bm;
    Mat3 qm;
    Real sm;
    MergedSums(i, j, anchors, points, q, b, s, o, qm, bm, sm);
    const Real d = Phi(qm, bm, sm) - Phi(q.Get(i), b.Get(i), s.Get(i)) - Phi(q.Get(j), b.Get(j), s.Get(j));
    record.second.first = viskores::Max(d, Real(0));
    if (this->DegreeFloor >= 0)
    {
      const viskores::Id degrees = (degreeOffsets.Get(i + 1) - degreeOffsets.Get(i)) +
        (degreeOffsets.Get(j + 1) - degreeOffsets.Get(j));
      record.first = viskores::Max(this->DegreeFloor, degrees);
    }
    isCandidate = 1;
  }

  bool Topological;
  bool HasFrozen;
  viskores::Id DegreeFloor;
};

struct CandidatePair : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn record, FieldOut pair, FieldOut ends);
  using ExecutionSignature = void(_1, _2, _3, WorkIndex);

  template <typename EndsVec>
  VISKORES_EXEC void operator()(const MergePriority& record,
                                viskores::Id2& pair,
                                EndsVec& ends,
                                viskores::Id rank) const
  {
    pair = record.second.second;
    ends[0] = viskores::Id2(pair[0], rank);
    ends[1] = viskores::Id2(pair[1], rank);
  }
};

/// The candidates that conflict with candidate `r`: those touching one of its two clusters or a
/// neighbour of one in `K`.
template <typename IdPortal, typename Visit>
VISKORES_EXEC bool ForEachConflict(const viskores::Id2& pair,
                                   const IdPortal& neighborOffsets,
                                   const IdPortal& neighbors,
                                   const IdPortal& touchingOffsets,
                                   const IdPortal& touching,
                                   Visit&& visit)
{
  auto visitCluster = [&](viskores::Id x)
  {
    for (viskores::Id t = touchingOffsets.Get(x); t < touchingOffsets.Get(x + 1); ++t)
    {
      if (!visit(touching.Get(t)))
      {
        return false;
      }
    }
    return true;
  };
  for (int e = 0; e < 2; ++e)
  {
    const viskores::Id c = pair[e];
    if (!visitCluster(c))
    {
      return false;
    }
    for (viskores::Id n = neighborOffsets.Get(c); n < neighborOffsets.Get(c + 1); ++n)
    {
      if (!visitCluster(neighbors.Get(n)))
      {
        return false;
      }
    }
  }
  return true;
}

enum : viskores::UInt8
{
  Alive = 0,
  Committed = 1,
  Dead = 2,
  Rejected = 3
};

/// A live candidate is ready when no live candidate of higher priority conflicts with it; then
/// every candidate of higher priority it conflicts with has been rejected or has died, and the
/// sequential greedy rule would examine it.
struct ReadyCandidates : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn pair,
                                FieldIn state,
                                WholeArrayIn states,
                                WholeArrayIn neighborOffsets,
                                WholeArrayIn neighbors,
                                WholeArrayIn touchingOffsets,
                                WholeArrayIn touching,
                                FieldOut ready);
  using ExecutionSignature = void(WorkIndex, _1, _2, _3, _4, _5, _6, _7, _8);

  template <typename StatePortal, typename IdPortal>
  VISKORES_EXEC void operator()(viskores::Id r,
                                const viskores::Id2& pair,
                                viskores::UInt8 state,
                                const StatePortal& states,
                                const IdPortal& neighborOffsets,
                                const IdPortal& neighbors,
                                const IdPortal& touchingOffsets,
                                const IdPortal& touching,
                                viskores::UInt8& ready) const
  {
    ready = 0;
    if (state != Alive)
    {
      return;
    }
    bool blocked = false;
    ForEachConflict(pair,
                    neighborOffsets,
                    neighbors,
                    touchingOffsets,
                    touching,
                    [&](viskores::Id q)
                    {
                      if (q < r && states.Get(q) == Alive)
                      {
                        blocked = true;
                        return false;
                      }
                      return true;
                    });
    ready = blocked ? 0 : 1;
  }
};

/// A ready candidate is committed if admissible and rejected otherwise.
struct DecideReady : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn ready, FieldIn admissible, FieldInOut state);
  using ExecutionSignature = void(_1, _2, _3);

  VISKORES_EXEC void operator()(viskores::UInt8 ready,
                                viskores::UInt8 admissible,
                                viskores::UInt8& state) const
  {
    if (ready)
    {
      state = (admissible == 1) ? Committed : Rejected;
    }
  }
};

/// A live candidate conflicting with a committed one dies.
struct KillConflicting : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn pair,
                                FieldInOut state,
                                WholeArrayIn states,
                                WholeArrayIn neighborOffsets,
                                WholeArrayIn neighbors,
                                WholeArrayIn touchingOffsets,
                                WholeArrayIn touching);
  using ExecutionSignature = void(_1, _2, _3, _4, _5, _6, _7);

  template <typename StatePortal, typename IdPortal>
  VISKORES_EXEC void operator()(const viskores::Id2& pair,
                                viskores::UInt8& state,
                                const StatePortal& states,
                                const IdPortal& neighborOffsets,
                                const IdPortal& neighbors,
                                const IdPortal& touchingOffsets,
                                const IdPortal& touching) const
  {
    if (state != Alive)
    {
      return;
    }
    ForEachConflict(pair,
                    neighborOffsets,
                    neighbors,
                    touchingOffsets,
                    touching,
                    [&](viskores::Id q)
                    {
                      if (states.Get(q) == Committed)
                      {
                        state = Dead;
                        return false;
                      }
                      return true;
                    });
  }
};

struct IsState : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn state, FieldOut match);
  using ExecutionSignature = _2(_1);

  VISKORES_CONT explicit IsState(viskores::UInt8 state)
    : State(state)
  {
  }

  VISKORES_EXEC viskores::UInt8 operator()(viskores::UInt8 state) const
  {
    return (state == this->State) ? 1 : 0;
  }

  viskores::UInt8 State;
};

struct NeedsEvaluation : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn ready, FieldIn admissible, FieldOut needed);
  using ExecutionSignature = _3(_1, _2);

  VISKORES_EXEC viskores::UInt8 operator()(viskores::UInt8 ready, viskores::UInt8 admissible) const
  {
    return (ready && admissible == 0) ? 1 : 0;
  }
};

struct ScatterAdmissible : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn rank, FieldIn admissible, WholeArrayInOut memo);
  using ExecutionSignature = void(_1, _2, _3);

  template <typename MemoPortal>
  VISKORES_EXEC void operator()(viskores::Id rank, viskores::UInt8 admissible, MemoPortal& memo) const
  {
    memo.Set(rank, admissible ? 1 : 2);
  }
};

struct IsCutRecord : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn record, FieldOut cut);
  using ExecutionSignature = _2(_1);

  VISKORES_EXEC viskores::UInt8 operator()(const viskores::Id3& record) const
  {
    return (record[0] != 0) ? 1 : 0;
  }
};

struct IdentityMap : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn index, FieldOut map);
  using ExecutionSignature = _2(_1);

  VISKORES_EXEC viskores::Id operator()(viskores::Id index) const { return index; }
};

struct MergeInto : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn pair, WholeArrayInOut map);
  using ExecutionSignature = void(_1, _2);

  template <typename MapPortal>
  VISKORES_EXEC void operator()(const viskores::Id2& pair, MapPortal& map) const
  {
    map.Set(pair[1], pair[0]);
  }
};

struct ApplyMap : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn label, WholeArrayIn map, FieldOut newLabel);
  using ExecutionSignature = _3(_1, _2);

  template <typename MapPortal>
  VISKORES_EXEC viskores::Id operator()(viskores::Id label, const MapPortal& map) const
  {
    return map.Get(label);
  }
};

struct ExactlyDegenerate : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn triangle, WholeArrayIn generators, FieldOut degenerate);
  using ExecutionSignature = _3(_1, _2);

  template <typename GeneratorsPortal>
  VISKORES_EXEC viskores::UInt8 operator()(const viskores::Id3& t,
                                           const GeneratorsPortal& generators) const
  {
    return viskores::TriangleIsDegenerate(generators.Get(t[0]), generators.Get(t[1]), generators.Get(t[2]))
      ? 1
      : 0;
  }
};

} // namespace detail

/// What the merge test and the full test of a merge check, and the frozen labels.
struct MergeParameters
{
  /// `k*`, or 0 in error mode.
  viskores::Id Budget = 0;
  Real MaxClusterRadius = std::numeric_limits<Real>::infinity();
  bool Embedding = true;
  Real ErrorBound = 0;
  /// The halves of the envelope flag (F5) in use.
  EnvelopeSides Sides = EnvelopeSides::Both;
  /// The input for the isotopy flag (F6), or null without it.
  const IsotopyMesh* Isotopy = nullptr;
  /// Whether candidates must keep the discrete closed-ball property; off in the geometric-only
  /// mode, where every pair of adjacent clusters is a candidate and the output is the set of
  /// simplices.
  bool Topological = true;
  /// (D2) by count of the clusters per closed component instead of by sort of the label triples.
  bool D2ByCount = false;
  /// The floor `D_0` of the degree key of the priority, or a negative value to order the
  /// candidates by the increase of the objective alone.
  viskores::Id DegreeFloor = 12;
  /// When the committed set fails the full test: defer the merges whose merged cluster is
  /// flagged and test the others again, or, if false, merge only the candidate of highest
  /// priority.
  bool DeferFlagged = true;
  /// For an input cut into surfaces, the first copy of the input vertex of every copy, which
  /// glues the output for (F4), and the frozen copies; empty otherwise.
  viskores::cont::ArrayHandle<viskores::Id> Canonical;
  viskores::cont::ArrayHandle<viskores::UInt8> FrozenVertices;
  /// With the singular curves decimated (Proposition "Decimating the singular curves"): their
  /// segments, which glue the output in place of `Canonical`, with (d') in place of (d) in the
  /// full test, and the input vertex of every copy; null otherwise.
  const SingularCurves* Curves = nullptr;
  viskores::cont::ArrayHandle<viskores::Id> Original;

  bool GeometricFlags() const
  {
    return this->Embedding || this->ErrorBound > 0 || this->Isotopy != nullptr ||
      this->MaxClusterRadius < std::numeric_limits<Real>::infinity();
  }
};

struct MergeDiagnostics
{
  viskores::Id Rounds = 0;
  /// Merges that passed the local test but whose full test failed.
  viskores::Id Dropped = 0;
  /// Rounds in which the committed set failed the full test and one merge was made instead.
  viskores::Id Fallbacks = 0;
  /// Merges deferred to a later round because their merged cluster was flagged.
  viskores::Id Deferred = 0;
  /// Local tests evaluated, and verdicts of earlier rounds reused instead.
  viskores::Id Evaluated = 0;
  viskores::Id Reused = 0;
  /// With the singular curves decimated: the rounds of segment merges, and the candidates
  /// tested.
  viskores::Id SegmentRounds = 0;
  viskores::Id SegmentCandidates = 0;
  viskores::Id EnvelopePieces = 0;
  /// Wall-clock seconds: building `K` and the candidates, the passes of the greedy set without
  /// the local tests, the local tests, and the full tests of the merged labellings.
  double SecondsCandidates = 0;
  double SecondsPasses = 0;
  double SecondsLocalTests = 0;
  double SecondsFullTests = 0;
  /// Per round: the merges committed, the merges deferred, and the wall-clock seconds.
  std::vector<viskores::Id> RoundMerges;
  std::vector<viskores::Id> RoundDeferred;
  std::vector<double> RoundSeconds;
};

/// The vertex of the glued output `K` of every label: the copies of every frozen vertex
/// identified, or with the singular curves decimated, the pieces of every segment; empty for an
/// input that is a surface.
inline viskores::cont::ArrayHandle<viskores::Id> OutputGlue(const MergeParameters& parameters,
                                                            const Clustering& clusters)
{
  if (parameters.Curves)
  {
    return SegmentGlueMap(*parameters.Curves, parameters.FrozenVertices, parameters.Original, clusters);
  }
  return GlueMap(parameters.Canonical, clusters);
}

/// The placement step, and with the singular curves decimated, one generator for the pieces of
/// every segment.
inline void PlaceGlued(const SurfaceMesh& mesh,
                       const VertexQuadrics& quadrics,
                       const MergeParameters& parameters,
                       Clustering& clusters)
{
  Place(mesh, quadrics.M, clusters);
  if (parameters.Curves)
  {
    ShareGenerators(OutputGlue(parameters, clusters), clusters);
  }
}

/// The data the local test of a merge reads, in the execution environment.
struct MergeContextPortals
{
  using VecPortal = viskores::cont::ArrayHandle<Vec3>::ReadPortalType;
  using IdPortal = viskores::cont::ArrayHandle<viskores::Id>::ReadPortalType;
  using Id2Portal = viskores::cont::ArrayHandle<viskores::Id2>::ReadPortalType;
  using Id3Portal = viskores::cont::ArrayHandle<viskores::Id3>::ReadPortalType;
  using MatPortal = viskores::cont::ArrayHandle<Mat3>::ReadPortalType;
  using FlagPortal = viskores::cont::ArrayHandle<viskores::UInt8>::ReadPortalType;
  using Distance =
    DistanceMeshPortals<VecPortal,
                        Id3Portal,
                        viskores::worklet::spatialstructure::BoundingVolumeHierarchy::ExecObjectType>;

  VecPortal Points;
  Id3Portal InputTriangles;
  IdPortal CornerOffsets;
  IdPortal Corners;
  viskores::Id NumberOfInputTriangles;
  IdPortal Labels;
  IdPortal MemberOffsets;
  IdPortal Members;
  IdPortal Anchors;
  VecPortal Generators;
  MatPortal Q;
  VecPortal B;
  Id3Portal OutputTriangles;
  IdPortal FanOffsets;
  IdPortal Fans;
  FlagPortal OutputDegenerate;
  Distance Input;
  Distance Output;
  IdPortal Glue;
  bool HasGlue;
  bool Geometric;
  Real MaxRadius;
  Real Epsilon;
  bool Embedding;
  /// The halves of (F5) in use, as the bits of the tests of `ForEachEnvelopeItem`.
  viskores::UInt8 EnvelopeSides;
  bool Isotopy;
  IsotopyPortals IsotopyTest;
  /// The three largest absolute coordinates of the generators, with their clusters, so that the
  /// largest with two clusters replaced is known.
  Real TopAbs[3];
  viskores::Id TopLabel[3];

  VISKORES_EXEC bool Incident(viskores::Id t, viskores::Id i, viskores::Id j) const
  {
    const viskores::Id3 tri = this->OutputTriangles.Get(t);
    for (int c = 0; c < 3; ++c)
    {
      if (tri[c] == i || tri[c] == j)
      {
        return true;
      }
    }
    return false;
  }

  VISKORES_EXEC viskores::Id3 Glued(const viskores::Id3& t) const
  {
    if (!this->HasGlue)
    {
      return t;
    }
    return viskores::Id3(this->Glue.Get(t[0]), this->Glue.Get(t[1]), this->Glue.Get(t[2]));
  }

  /// Calls `visit(P, labels)` for every triangle of `K/ij` at the merged vertex `i`, realised
  /// with `i` at `merged`: the triangles at `i` or `j`, relabelled, but not those at both, which
  /// degenerate. They are visited in the order of the fans of `i`, then of `j`. In the
  /// geometric-only mode the output is the set of simplices, sorted triples with repetition, and
  /// those at both are kept as lower-dimensional simplices.
  template <typename Visit>
  VISKORES_EXEC bool ForEachNewTriangle(viskores::Id i,
                                        viskores::Id j,
                                        const Vec3& merged,
                                        Visit&& visit) const
  {
    for (int which = 0; which < 2; ++which)
    {
      const viskores::Id c = (which == 0) ? i : j;
      for (viskores::Id f = this->FanOffsets.Get(c); f < this->FanOffsets.Get(c + 1); ++f)
      {
        const viskores::Id t = this->Fans.Get(f);
        viskores::Id3 labels = this->OutputTriangles.Get(t);
        bool hasI = false, hasJ = false;
        for (int k = 0; k < 3; ++k)
        {
          hasI = hasI || labels[k] == i;
          hasJ = hasJ || labels[k] == j;
        }
        if (hasI && hasJ && !this->Geometric)
        {
          continue;
        }
        for (int k = 0; k < 3; ++k)
        {
          if (labels[k] == j)
          {
            labels[k] = i;
          }
        }
        if (this->Geometric)
        {
          labels = detail::SortedTriple(labels);
        }
        Vec3 p[3];
        for (int k = 0; k < 3; ++k)
        {
          p[k] = (labels[k] == i) ? merged : this->Generators.Get(labels[k]);
        }
        if (!visit(p, labels, t))
        {
          return false;
        }
      }
    }
    return true;
  }

  /// The largest absolute coordinate of the generators with clusters `i` and `j` at `merged`.
  VISKORES_EXEC Real MergedGeneratorsMaxAbs(viskores::Id i, viskores::Id j, const Vec3& merged) const
  {
    Real s = viskores::Max(viskores::Max(viskores::Abs(merged[0]), viskores::Abs(merged[1])),
                           viskores::Abs(merged[2]));
    for (int k = 0; k < 3; ++k)
    {
      if (this->TopLabel[k] >= 0 && this->TopLabel[k] != i && this->TopLabel[k] != j)
      {
        return viskores::Max(s, this->TopAbs[k]);
      }
    }
    return s;
  }

  /// The local test of a merge of Section 11.5 but for the envelope: (F0) on the merged
  /// cluster, (F4) on the pairs with a triangle at the merged vertex, and (F6) on the 3-label
  /// input triangles with a vertex in the merged cluster. Sets `merged` to the position of the
  /// merged generator.
  VISKORES_EXEC bool Admissible(viskores::Id i,
                                viskores::Id j,
                                Vec3& merged,
                                bool& complete) const;

  /// Visits the input triangles the envelope part of the local test reads, as `visit(t, tri,
  /// tests)`, with the tests each needs: bit 0, its simplex in the merged labelling against the
  /// input ((F5)(i), the 3-label triangles with a vertex in the merged cluster); bit 1, the
  /// triangle against `K/ij` ((F5)(ii), the triangles with a vertex in the merged cluster and
  /// those whose box meets the box of the removed output triangles grown by epsilon). Only the
  /// triangles with `D(t) > epsilon` are visited. In the geometric-only mode, both tests of
  /// (F5') on the triangles with a vertex in the merged cluster, the second against all the
  /// simplices of the merged labelling. `parts` selects the triangles with a vertex in the
  /// merged cluster (bit 0) and the others (bit 1).
  template <typename Visit>
  VISKORES_EXEC void ForEachEnvelopeItem(viskores::Id i,
                                         viskores::Id j,
                                         const Vec3& merged,
                                         viskores::UInt8 parts,
                                         bool& complete,
                                         Visit&& visit) const;

  /// The tests `tests` of `ForEachEnvelopeItem` on input triangle `t`. `scaleInput` and
  /// `scaleOutput` are `S (1 + 16 u)` of the two batches of tests of the merge.
  VISKORES_EXEC bool TestEnvelopeItem(viskores::Id i,
                                      viskores::Id j,
                                      const Vec3& merged,
                                      viskores::Id t,
                                      viskores::UInt8 tests,
                                      Real scaleInput,
                                      Real scaleOutput,
                                      viskores::Id& pieces,
                                      bool& complete) const;
};

/// The distance to `K/ij`: the triangles of `K` not at `i` or `j`, and the new triangles at the
/// merged vertex.
struct MergedOutputDistance
{
  const MergeContextPortals& Context;
  viskores::Id I;
  viskores::Id J;
  Vec3 Merged;

  /// The distance when it is at most `cutoff`, and otherwise some value above `cutoff`.
  VISKORES_EXEC Real DistanceUpTo(const Vec3& p, Real cutoff, bool& complete) const
  {
    Real best = this->Context.Output.DistanceSkipping(
      p,
      [&](viskores::Id t) { return this->Context.Incident(t, this->I, this->J); },
      cutoff,
      complete);
    this->Context.ForEachNewTriangle(
      this->I,
      this->J,
      this->Merged,
      [&](const Vec3* q, const viskores::Id3&, viskores::Id)
      {
        best = viskores::Min(best, PointTriangleDistanceUpper(p, q[0], q[1], q[2]));
        return true;
      });
    return best;
  }

  VISKORES_EXEC bool WithinOneTriangle(const Vec3* s, Real bound, Real margin, bool& complete) const
  {
    if (this->Context.Output.WithinOneTriangleSkipping(
          s,
          bound,
          margin,
          [&](viskores::Id t) { return this->Context.Incident(t, this->I, this->J); },
          complete))
    {
      return true;
    }
    bool found = false;
    this->Context.ForEachNewTriangle(this->I,
                                     this->J,
                                     this->Merged,
                                     [&](const Vec3* q, const viskores::Id3&, viskores::Id)
                                     {
                                       Real worst = 0;
                                       for (int k = 0; k < 3; ++k)
                                       {
                                         worst = viskores::Max(
                                           worst, PointTriangleDistanceUpper(s[k], q[0], q[1], q[2]));
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

VISKORES_EXEC inline bool MergeContextPortals::Admissible(viskores::Id i,
                                                          viskores::Id j,
                                                          Vec3& merged,
                                                          bool& complete) const
{
  Vec3 o, bm;
  Mat3 qm;
  {
    // Only Q and b are needed for the position.
    const viskores::Id a = (this->Anchors.Get(i) < this->Anchors.Get(j)) ? i : j;
    o = this->Points.Get(this->Anchors.Get(a));
    qm = Mat3(0);
    bm = Vec3(0);
    for (int which = 0; which < 2; ++which)
    {
      const viskores::Id c = (which == 0) ? i : j;
      const Vec3 d = Sub(this->Points.Get(this->Anchors.Get(c)), o);
      const Mat3 qc = this->Q.Get(c);
      qm = MatAdd(qm, qc);
      bm = Add(Add(bm, this->B.Get(c)), MatVec(qc, d));
    }
  }
  merged = Add(o, SolveSPD(qm, bm));
  auto forEachMember = [&](auto&& visit)
  {
    for (int which = 0; which < 2; ++which)
    {
      const viskores::Id c = (which == 0) ? i : j;
      for (viskores::Id k = this->MemberOffsets.Get(c); k < this->MemberOffsets.Get(c + 1); ++k)
      {
        visit(this->Members.Get(k));
      }
    }
  };
  // (F0)
  if (this->MaxRadius < std::numeric_limits<Real>::infinity())
  {
    Real radius = 0;
    forEachMember([&](viskores::Id v)
                  { radius = viskores::Max(radius, Norm(Sub(this->Points.Get(v), merged))); });
    if (radius > this->MaxRadius)
    {
      return false;
    }
  }
  if (this->Geometric)
  {
    return true;
  }
  // (F4): the new triangles must be nondegenerate and meet no other triangle improperly.
  if (this->Embedding)
  {
    bool bad = false;
    this->ForEachNewTriangle(
      i,
      j,
      merged,
      [&](const Vec3* p, const viskores::Id3& labels, viskores::Id t)
      {
        if (viskores::TriangleIsDegenerate(p[0], p[1], p[2]))
        {
          bad = true;
          return false;
        }
        // Against the other new triangles, each pair once.
        bool after = false;
        this->ForEachNewTriangle(i,
                                 j,
                                 merged,
                                 [&](const Vec3* q, const viskores::Id3& otherLabels, viskores::Id u)
                                 {
                                   if (u == t)
                                   {
                                     after = true;
                                     return true;
                                   }
                                   if (after &&
                                       exact::ImproperPair(
                                         p, this->Glued(labels), q, this->Glued(otherLabels)))
                                   {
                                     bad = true;
                                     return false;
                                   }
                                   return true;
                                 });
        if (bad)
        {
          return false;
        }
        // Against the triangles of K away from i and j, which keep their positions.
        Vec3 lo = p[0], hi = p[0];
        for (int k = 1; k < 3; ++k)
        {
          for (int a = 0; a < 3; ++a)
          {
            lo[a] = viskores::Min(lo[a], p[k][a]);
            hi[a] = viskores::Max(hi[a], p[k][a]);
          }
        }
        const bool done = this->Output.Hierarchy.ForEachOverlap(
          lo,
          hi,
          [&](viskores::Id u)
          {
            if (this->Incident(u, i, j) || this->OutputDegenerate.Get(u))
            {
              return true;
            }
            const viskores::Id3 other = this->OutputTriangles.Get(u);
            const Vec3 q[3] = { this->Generators.Get(other[0]),
                                this->Generators.Get(other[1]),
                                this->Generators.Get(other[2]) };
            if (exact::ImproperPair(p, this->Glued(labels), q, this->Glued(other)))
            {
              bad = true;
              return false;
            }
            return true;
          });
        complete = complete && done;
        return !bad;
      });
    if (bad)
    {
      return false;
    }
  }
  // (F6): a 3-label triangle has exactly one corner in the merged cluster, so each is tested
  // once.
  if (this->Isotopy)
  {
    bool bad = false;
    forEachMember(
      [&](viskores::Id v)
      {
        for (viskores::Id c = this->CornerOffsets.Get(v);
             !bad && c < this->CornerOffsets.Get(v + 1);
             ++c)
        {
          const viskores::Id t = this->Corners.Get(c) % this->NumberOfInputTriangles;
          const viskores::Id3 tri = this->InputTriangles.Get(t);
          viskores::Id labels[3];
          for (int k = 0; k < 3; ++k)
          {
            labels[k] = this->Labels.Get(tri[k]);
            labels[k] = (labels[k] == j) ? i : labels[k];
          }
          if (labels[0] == labels[1] || labels[1] == labels[2] || labels[0] == labels[2])
          {
            continue;
          }
          Vec3 p[3];
          for (int k = 0; k < 3; ++k)
          {
            p[k] = (labels[k] == i) ? merged : this->Generators.Get(labels[k]);
          }
          bad = !this->IsotopyTest.Passes(p, t, complete);
        }
      });
    if (bad)
    {
      return false;
    }
  }
  return true;
}

template <typename Visit>
VISKORES_EXEC inline void MergeContextPortals::ForEachEnvelopeItem(viskores::Id i,
                                                                   viskores::Id j,
                                                                   const Vec3& merged,
                                                                   viskores::UInt8 parts,
                                                                   bool& complete,
                                                                   Visit&& visit) const
{
  const Real epsilon = this->Epsilon;
  // Only the tests of the halves in use.
  auto emit = [&](viskores::Id t, const viskores::Id3& tri, viskores::UInt8 tests)
  {
    tests = static_cast<viskores::UInt8>(tests & this->EnvelopeSides);
    if (tests != 0)
    {
      visit(t, tri, tests);
    }
  };
  auto inMerged = [&](viskores::Id v)
  {
    const viskores::Id l = this->Labels.Get(v);
    return l == i || l == j;
  };
  auto generator = [&](viskores::Id l) { return (l == i || l == j) ? merged : this->Generators.Get(l); };
  auto newLabel = [&](viskores::Id l) { return (l == j) ? i : l; };
  auto displacementExceeds = [&](const viskores::Id3& tri)
  {
    Real d = 0;
    for (int k = 0; k < 3; ++k)
    {
      const Vec3 p = this->Points.Get(tri[k]);
      d = viskores::Max(d, Norm(Sub(generator(this->Labels.Get(tri[k])), p)));
    }
    return d > epsilon;
  };
  // The input triangles with a vertex in the merged cluster, each from its smallest vertex there.
  for (int which = 0; which < 2 && (parts & 1) != 0; ++which)
  {
    const viskores::Id c = (which == 0) ? i : j;
    for (viskores::Id k = this->MemberOffsets.Get(c); k < this->MemberOffsets.Get(c + 1); ++k)
    {
      const viskores::Id v = this->Members.Get(k);
      for (viskores::Id r = this->CornerOffsets.Get(v); r < this->CornerOffsets.Get(v + 1); ++r)
      {
        const viskores::Id t = this->Corners.Get(r) % this->NumberOfInputTriangles;
        const viskores::Id3 tri = this->InputTriangles.Get(t);
        viskores::Id first = -1;
        for (int a = 0; a < 3; ++a)
        {
          if (inMerged(tri[a]) && (first < 0 || tri[a] < first))
          {
            first = tri[a];
          }
        }
        if (first != v || !displacementExceeds(tri))
        {
          continue;
        }
        if (this->Geometric)
        {
          emit(t, tri, viskores::UInt8(3));
          continue;
        }
        const viskores::Id a = newLabel(this->Labels.Get(tri[0]));
        const viskores::Id b = newLabel(this->Labels.Get(tri[1]));
        const viskores::Id d = newLabel(this->Labels.Get(tri[2]));
        emit(t, tri, viskores::UInt8((a != b && b != d && a != d) ? 3 : 2));
      }
    }
  }
  if (this->Geometric || (parts & 2) == 0)
  {
    return;
  }
  // The input triangles away from the merged cluster near a removed output triangle.
  Vec3 lo(std::numeric_limits<Real>::infinity()), hi(-std::numeric_limits<Real>::infinity());
  for (int which = 0; which < 2; ++which)
  {
    const viskores::Id c = (which == 0) ? i : j;
    for (viskores::Id f = this->FanOffsets.Get(c); f < this->FanOffsets.Get(c + 1); ++f)
    {
      const viskores::Id3 tri = this->OutputTriangles.Get(this->Fans.Get(f));
      for (int k = 0; k < 3; ++k)
      {
        const Vec3 p = this->Generators.Get(tri[k]);
        for (int a = 0; a < 3; ++a)
        {
          lo[a] = viskores::Min(lo[a], p[a]);
          hi[a] = viskores::Max(hi[a], p[a]);
        }
      }
    }
  }
  for (int a = 0; a < 3; ++a)
  {
    lo[a] = lo[a] - epsilon;
    hi[a] = hi[a] + epsilon;
  }
  const bool done = this->Input.Hierarchy.ForEachOverlap(
    lo,
    hi,
    [&](viskores::Id t)
    {
      const viskores::Id3 tri = this->InputTriangles.Get(t);
      if (!inMerged(tri[0]) && !inMerged(tri[1]) && !inMerged(tri[2]) && displacementExceeds(tri))
      {
        emit(t, tri, viskores::UInt8(2));
      }
      return true;
    });
  complete = complete && done;
}

VISKORES_EXEC inline bool MergeContextPortals::TestEnvelopeItem(viskores::Id i,
                                                                viskores::Id j,
                                                                const Vec3& merged,
                                                                viskores::Id t,
                                                                viskores::UInt8 tests,
                                                                Real scaleInput,
                                                                Real scaleOutput,
                                                                viskores::Id& pieces,
                                                                bool& complete) const
{
  const viskores::Id3 tri = this->InputTriangles.Get(t);
  auto generator = [&](viskores::Id l) { return (l == i || l == j) ? merged : this->Generators.Get(l); };
  bool done = true;
  if ((tests & 1) != 0)
  {
    const CertifiedWithin test{ this->Epsilon, scaleInput };
    const bool within = test(generator(this->Labels.Get(tri[0])),
                             generator(this->Labels.Get(tri[1])),
                             generator(this->Labels.Get(tri[2])),
                             this->Input,
                             pieces,
                             done);
    complete = complete && done;
    if (!within)
    {
      return false;
    }
  }
  if ((tests & 2) != 0)
  {
    const CertifiedWithin test{ this->Epsilon, scaleOutput };
    const MergedOutputDistance distance{ *this, i, j, merged };
    const bool within = test(this->Points.Get(tri[0]),
                             this->Points.Get(tri[1]),
                             this->Points.Get(tri[2]),
                             distance,
                             pieces,
                             done);
    complete = complete && done;
    if (!within)
    {
      return false;
    }
  }
  return true;
}

/// An execution object giving a worklet a `MergeContextPortals`.
struct MergeContextObject : viskores::cont::ExecutionObjectBase
{
  const SurfaceMesh* Mesh;
  const Clustering* Clusters;
  const OutputComplex* Complex;
  viskores::cont::ArrayHandle<viskores::UInt8> OutputDegenerate;
  viskores::cont::ArrayHandle<viskores::Id> Glue;
  bool Geometric = false;
  DistanceMesh Input;
  DistanceMesh Output;
  Real MaxRadius;
  Real Epsilon;
  bool Embedding;
  quadric_clustering::EnvelopeSides Sides = quadric_clustering::EnvelopeSides::Both;
  const IsotopyMesh* Isotopy = nullptr;
  std::array<Real, 3> TopAbs;
  std::array<viskores::Id, 3> TopLabel;

  VISKORES_CONT MergeContextPortals PrepareForExecution(viskores::cont::DeviceAdapterId device,
                                                        viskores::cont::Token& token) const
  {
    MergeContextPortals portals;
    portals.Points = this->Mesh->Points.PrepareForInput(device, token);
    portals.InputTriangles = this->Mesh->Triangles.PrepareForInput(device, token);
    portals.CornerOffsets = this->Mesh->VertexCornerOffsets.PrepareForInput(device, token);
    portals.Corners = this->Mesh->VertexCorners.PrepareForInput(device, token);
    portals.NumberOfInputTriangles = this->Mesh->NumberOfTriangles;
    portals.Labels = this->Clusters->Labels.PrepareForInput(device, token);
    portals.MemberOffsets = this->Clusters->Keys.GetOffsets().PrepareForInput(device, token);
    portals.Members = this->Clusters->Keys.GetSortedValuesMap().PrepareForInput(device, token);
    portals.Anchors = this->Clusters->Anchors.PrepareForInput(device, token);
    portals.Generators = this->Clusters->Generators.PrepareForInput(device, token);
    portals.Q = this->Clusters->Q.PrepareForInput(device, token);
    portals.B = this->Clusters->B.PrepareForInput(device, token);
    portals.OutputTriangles = this->Complex->Triangles.PrepareForInput(device, token);
    portals.FanOffsets = this->Complex->FanOffsets.PrepareForInput(device, token);
    portals.Fans = this->Complex->Fans.PrepareForInput(device, token);
    portals.OutputDegenerate = this->OutputDegenerate.PrepareForInput(device, token);
    portals.Glue = this->Glue.PrepareForInput(device, token);
    portals.HasGlue = this->Glue.GetNumberOfValues() > 0;
    portals.Geometric = this->Geometric;
    portals.Input = DistanceMeshObject(this->Input).PrepareForExecution(device, token);
    portals.Output = DistanceMeshObject(this->Output).PrepareForExecution(device, token);
    portals.MaxRadius = this->MaxRadius;
    portals.Epsilon = this->Epsilon;
    portals.Embedding = this->Embedding;
    portals.EnvelopeSides = static_cast<viskores::UInt8>(this->Sides);
    portals.Isotopy = this->Isotopy != nullptr;
    if (this->Isotopy)
    {
      portals.IsotopyTest = IsotopyMeshObject(*this->Isotopy).PrepareForExecution(device, token);
    }
    for (int k = 0; k < 3; ++k)
    {
      portals.TopAbs[k] = this->TopAbs[static_cast<std::size_t>(k)];
      portals.TopLabel[k] = this->TopLabel[static_cast<std::size_t>(k)];
    }
    return portals;
  }
};

namespace detail
{

/// The box of what a merge changes: the members of the two clusters, the generators of the
/// vertices of their simplices in `K`, and the merged generator, which with them spans the new
/// simplices.
struct ChangedRegion : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn pair,
                                WholeArrayIn memberOffsets,
                                WholeArrayIn members,
                                WholeArrayIn points,
                                WholeArrayIn generators,
                                WholeArrayIn fanOffsets,
                                WholeArrayIn fans,
                                WholeArrayIn simplices,
                                WholeArrayIn anchors,
                                WholeArrayIn nextLabels,
                                WholeArrayIn nextGenerators,
                                FieldOut lower,
                                FieldOut upper);
  using ExecutionSignature = void(_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13);

  template <typename IdPortal, typename VecPortal, typename Id3Portal>
  VISKORES_EXEC void operator()(const viskores::Id2& pair,
                                const IdPortal& memberOffsets,
                                const IdPortal& members,
                                const VecPortal& points,
                                const VecPortal& generators,
                                const IdPortal& fanOffsets,
                                const IdPortal& fans,
                                const Id3Portal& simplices,
                                const IdPortal& anchors,
                                const IdPortal& nextLabels,
                                const VecPortal& nextGenerators,
                                Vec3& lower,
                                Vec3& upper) const
  {
    lower = nextGenerators.Get(nextLabels.Get(anchors.Get(pair[0])));
    upper = lower;
    auto extend = [&](const Vec3& p)
    {
      for (int a = 0; a < 3; ++a)
      {
        lower[a] = viskores::Min(lower[a], p[a]);
        upper[a] = viskores::Max(upper[a], p[a]);
      }
    };
    for (int which = 0; which < 2; ++which)
    {
      const viskores::Id c = pair[which];
      for (viskores::Id k = memberOffsets.Get(c); k < memberOffsets.Get(c + 1); ++k)
      {
        extend(points.Get(members.Get(k)));
      }
      for (viskores::Id f = fanOffsets.Get(c); f < fanOffsets.Get(c + 1); ++f)
      {
        const viskores::Id3 simplex = simplices.Get(fans.Get(f));
        for (int k = 0; k < 3; ++k)
        {
          extend(generators.Get(simplex[k]));
        }
      }
    }
  }
};

} // namespace detail

/// `MergedGeneratorsMaxAbs` of the merge of `i` and `j`, with `mergedAbs` the largest absolute
/// coordinate of the merged generator: the one term of a local test that reads the whole
/// labelling.
inline Real MergedScale(const MergeContextObject& context, viskores::Id i, viskores::Id j, Real mergedAbs)
{
  for (std::size_t slot = 0; slot < 3; ++slot)
  {
    if (context.TopLabel[slot] >= 0 && context.TopLabel[slot] != i && context.TopLabel[slot] != j)
    {
      return viskores::Max(mergedAbs, context.TopAbs[slot]);
    }
  }
  return mergedAbs;
}

/// The verdicts of the local tests, kept across the rounds of the merge phase. A local test
/// reads only what lies in the box of `EvaluateMerge`, and the largest coordinate of the
/// generators away from the two clusters; its verdict holds while neither cluster merges, no
/// merge changes anything in the box, and that coordinate stays the same. The clusters are
/// known by their anchors, which the renumbering of the labels keeps.
class VerdictCache
{
public:
  /// Sets `memo` of the candidates `pairs` whose verdicts hold. Returns how many.
  viskores::Id Reuse(const MergeContextObject& context,
                     const Clustering& clusters,
                     const viskores::cont::ArrayHandle<viskores::Id2>& pairs,
                     viskores::cont::ArrayHandle<viskores::UInt8>& memo) const
  {
    if (this->Entries.empty())
    {
      return 0;
    }
    auto pairPortal = pairs.ReadPortal();
    auto anchors = clusters.Anchors.ReadPortal();
    auto memoPortal = memo.WritePortal();
    viskores::Id reused = 0;
    for (viskores::Id c = 0; c < pairPortal.GetNumberOfValues(); ++c)
    {
      const viskores::Id2 pair = pairPortal.Get(c);
      const auto found = this->Entries.find(std::make_pair(anchors.Get(pair[0]), anchors.Get(pair[1])));
      if (found != this->Entries.end() &&
          MergedScale(context, pair[0], pair[1], found->second.MergedAbs) == found->second.Scale)
      {
        memoPortal.Set(c, found->second.Admissible ? 1 : 2);
        ++reused;
      }
    }
    return reused;
  }

  /// Keeps the verdicts of the local tests of `pairs`.
  void Remember(const MergeContextObject& context,
                const Clustering& clusters,
                const viskores::cont::ArrayHandle<viskores::Id2>& pairs,
                const viskores::cont::ArrayHandle<viskores::UInt8>& admissible,
                const viskores::cont::ArrayHandle<Vec3>& merged,
                const viskores::cont::ArrayHandle<Vec3>& lower,
                const viskores::cont::ArrayHandle<Vec3>& upper)
  {
    auto pairPortal = pairs.ReadPortal();
    auto anchors = clusters.Anchors.ReadPortal();
    auto admissiblePortal = admissible.ReadPortal();
    auto mergedPortal = merged.ReadPortal();
    auto lowerPortal = lower.ReadPortal();
    auto upperPortal = upper.ReadPortal();
    for (viskores::Id c = 0; c < pairPortal.GetNumberOfValues(); ++c)
    {
      const viskores::Id2 pair = pairPortal.Get(c);
      const Vec3 p = mergedPortal.Get(c);
      Entry entry;
      entry.Lower = lowerPortal.Get(c);
      entry.Upper = upperPortal.Get(c);
      entry.MergedAbs = viskores::Max(viskores::Max(viskores::Abs(p[0]), viskores::Abs(p[1])), viskores::Abs(p[2]));
      entry.Scale = MergedScale(context, pair[0], pair[1], entry.MergedAbs);
      entry.Admissible = admissiblePortal.Get(c) != 0;
      this->Entries[std::make_pair(anchors.Get(pair[0]), anchors.Get(pair[1]))] = entry;
    }
  }

  /// Forgets the verdicts that the merges `pairs` of the labelling `previous`, with output
  /// complex `complex`, into `next` may change.
  void Forget(const SurfaceMesh& mesh,
              const OutputComplex& complex,
              const Clustering& previous,
              const Clustering& next,
              const viskores::cont::ArrayHandle<viskores::Id2>& pairs)
  {
    if (this->Entries.empty())
    {
      return;
    }
    viskores::cont::Invoker invoke;
    viskores::cont::ArrayHandle<Vec3> lower, upper;
    invoke(detail::ChangedRegion{},
           pairs,
           previous.Keys.GetOffsets(),
           previous.Keys.GetSortedValuesMap(),
           mesh.Points,
           previous.Generators,
           complex.FanOffsets,
           complex.Fans,
           complex.Triangles,
           previous.Anchors,
           next.Labels,
           next.Generators,
           lower,
           upper);
    std::set<viskores::Id> merged;
    {
      auto pairPortal = pairs.ReadPortal();
      auto anchors = previous.Anchors.ReadPortal();
      for (viskores::Id c = 0; c < pairPortal.GetNumberOfValues(); ++c)
      {
        merged.insert(anchors.Get(pairPortal.Get(c)[0]));
        merged.insert(anchors.Get(pairPortal.Get(c)[1]));
      }
    }
    auto lowerPortal = lower.ReadPortal();
    auto upperPortal = upper.ReadPortal();
    for (auto entry = this->Entries.begin(); entry != this->Entries.end();)
    {
      bool changed = merged.count(entry->first.first) > 0 || merged.count(entry->first.second) > 0;
      for (viskores::Id c = 0; !changed && c < lowerPortal.GetNumberOfValues(); ++c)
      {
        const Vec3 l = lowerPortal.Get(c), u = upperPortal.Get(c);
        changed = l[0] <= entry->second.Upper[0] && u[0] >= entry->second.Lower[0] &&
          l[1] <= entry->second.Upper[1] && u[1] >= entry->second.Lower[1] &&
          l[2] <= entry->second.Upper[2] && u[2] >= entry->second.Lower[2];
      }
      entry = changed ? this->Entries.erase(entry) : std::next(entry);
    }
  }

private:
  struct Entry
  {
    Vec3 Lower;
    Vec3 Upper;
    Real MergedAbs;
    Real Scale;
    bool Admissible;
  };
  std::map<std::pair<viskores::Id, viskores::Id>, Entry> Entries;
};

namespace detail
{

/// The local test of a merge but for the envelope, and the work of its envelope part: the
/// numbers of input triangles to test with a vertex in the merged cluster and away from it, and
/// `S (1 + 16 u)` of the two batches of tests. Also a box that holds everything the whole local
/// test reads: the members of the two clusters, the generators of their triangles of `K`, the
/// merged generator and the input triangles of the envelope test, grown by epsilon, since the
/// distances it computes are exact only up to epsilon. Nothing else of `K` or of the labelling
/// is read, but the largest coordinate of the generators.
struct EvaluateMerge : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn pair,
                                ExecObject context,
                                FieldOut admissible,
                                FieldOut merged,
                                FieldOut numItems,
                                FieldOut scales,
                                FieldOut lower,
                                FieldOut upper);
  using ExecutionSignature = void(_1, _2, _3, _4, _5, _6, _7, _8);

  template <typename Context>
  VISKORES_EXEC void operator()(const viskores::Id2& pair,
                                const Context& context,
                                viskores::UInt8& admissible,
                                Vec3& merged,
                                viskores::Id2& numItems,
                                viskores::Vec<Real, 2>& scales,
                                Vec3& lower,
                                Vec3& upper) const
  {
    const viskores::Id i = pair[0], j = pair[1];
    bool complete = true;
    admissible = context.Admissible(i, j, merged, complete) ? 1 : 0;
    numItems = viskores::Id2(0, 0);
    scales = viskores::Vec<Real, 2>(0, 0);
    lower = merged;
    upper = merged;
    auto extend = [&](const Vec3& p)
    {
      for (int a = 0; a < 3; ++a)
      {
        lower[a] = viskores::Min(lower[a], p[a]);
        upper[a] = viskores::Max(upper[a], p[a]);
      }
    };
    for (int which = 0; which < 2; ++which)
    {
      const viskores::Id c = (which == 0) ? i : j;
      for (viskores::Id k = context.MemberOffsets.Get(c); k < context.MemberOffsets.Get(c + 1); ++k)
      {
        extend(context.Points.Get(context.Members.Get(k)));
      }
      for (viskores::Id f = context.FanOffsets.Get(c); f < context.FanOffsets.Get(c + 1); ++f)
      {
        const viskores::Id3 tri = context.OutputTriangles.Get(context.Fans.Get(f));
        for (int k = 0; k < 3; ++k)
        {
          extend(context.Generators.Get(tri[k]));
        }
      }
    }
    if (admissible != 0 && context.Epsilon > 0)
    {
      auto absMax = [](const Vec3& p)
      {
        return viskores::Max(viskores::Max(viskores::Abs(p[0]), viskores::Abs(p[1])),
                             viskores::Abs(p[2]));
      };
      Real sInput = context.Input.MaxAbsCoordinate;
      Real sOutput = context.MergedGeneratorsMaxAbs(i, j, merged);
      for (viskores::IdComponent part = 0; part < 2; ++part)
      {
        context.ForEachEnvelopeItem(
          i,
          j,
          merged,
          viskores::UInt8(1 << part),
          complete,
          [&](viskores::Id, const viskores::Id3& tri, viskores::UInt8 tests)
          {
            ++numItems[part];
            for (int k = 0; k < 3; ++k)
            {
              extend(context.Points.Get(tri[k]));
              if ((tests & 1) != 0)
              {
                const viskores::Id l = context.Labels.Get(tri[k]);
                sInput = viskores::Max(
                  sInput, absMax((l == i || l == j) ? merged : context.Generators.Get(l)));
              }
              if ((tests & 2) != 0)
              {
                sOutput = viskores::Max(sOutput, absMax(context.Points.Get(tri[k])));
              }
            }
          });
      }
      const Real oneSixteen = 1 + 16 * UnitRoundoff;
      scales = viskores::Vec<Real, 2>(sInput * oneSixteen, sOutput * oneSixteen);
    }
    Real size = 0;
    for (int a = 0; a < 3; ++a)
    {
      size = viskores::Max(size, viskores::Max(viskores::Abs(lower[a]), viskores::Abs(upper[a])));
    }
    const Real grow = context.Epsilon * (1 + Real(1e-9)) + 64 * UnitRoundoff * size;
    for (int a = 0; a < 3; ++a)
    {
      lower[a] = lower[a] - grow;
      upper[a] = upper[a] + grow;
    }
    if (!complete)
    {
      this->RaiseError("QuadricClusteringDecimation: bounding volume hierarchy too deep.");
    }
  }
};

/// The number of input triangles of part `Part` to test for a merge, none once it has failed.
struct PartCount : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn numItems, FieldIn failed, FieldOut count);
  using ExecutionSignature = _3(_1, _2);

  VISKORES_CONT explicit PartCount(viskores::IdComponent part)
    : Part(part)
  {
  }

  VISKORES_EXEC viskores::Id operator()(const viskores::Id2& numItems, viskores::UInt32 failed) const
  {
    return failed != 0 ? 0 : numItems[this->Part];
  }

private:
  viskores::IdComponent Part;
};

/// Lists the input triangles of part `Part` of the envelope tests of the merges `First` to
/// `Last`, from position `offset - Base` on.
struct FillEnvelopeItems : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn pair,
                                FieldIn merged,
                                FieldIn numItems,
                                FieldIn offset,
                                ExecObject context,
                                WholeArrayOut candidates,
                                WholeArrayOut triangles,
                                WholeArrayOut tests);
  using ExecutionSignature = void(WorkIndex, _1, _2, _3, _4, _5, _6, _7, _8);

  VISKORES_CONT FillEnvelopeItems(viskores::UInt8 part,
                                  viskores::Id first,
                                  viskores::Id last,
                                  viskores::Id base)
    : Part(part)
    , First(first)
    , Last(last)
    , Base(base)
  {
  }

  template <typename Context, typename IdPortal, typename FlagPortal>
  VISKORES_EXEC void operator()(viskores::Id index,
                                const viskores::Id2& pair,
                                const Vec3& merged,
                                viskores::Id numItems,
                                viskores::Id offset,
                                const Context& context,
                                const IdPortal& candidates,
                                const IdPortal& triangles,
                                const FlagPortal& tests) const
  {
    if (index < this->First || index >= this->Last || numItems == 0)
    {
      return;
    }
    viskores::Id k = offset - this->Base;
    bool complete = true;
    context.ForEachEnvelopeItem(pair[0],
                                pair[1],
                                merged,
                                this->Part,
                                complete,
                                [&](viskores::Id t, const viskores::Id3&, viskores::UInt8 flags)
                                {
                                  candidates.Set(k, index);
                                  triangles.Set(k, t);
                                  tests.Set(k, flags);
                                  ++k;
                                });
  }

private:
  viskores::UInt8 Part;
  viskores::Id First;
  viskores::Id Last;
  viskores::Id Base;
};

/// One input triangle of the envelope test of a merge. Marks the merge failed if a test fails,
/// and skips the triangle if the merge has already failed: the outcome is the conjunction over
/// the triangles, whatever the order.
struct TestEnvelopeItems : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn candidate,
                                FieldIn triangle,
                                FieldIn tests,
                                WholeArrayIn pairs,
                                WholeArrayIn merged,
                                WholeArrayIn scales,
                                ExecObject context,
                                AtomicArrayInOut failed,
                                FieldOut pieces);
  using ExecutionSignature = void(_1, _2, _3, _4, _5, _6, _7, _8, _9);

  template <typename PairPortal,
            typename VecPortal,
            typename ScalePortal,
            typename Context,
            typename AtomicPortal>
  VISKORES_EXEC void operator()(viskores::Id c,
                                viskores::Id t,
                                viskores::UInt8 tests,
                                const PairPortal& pairs,
                                const VecPortal& merged,
                                const ScalePortal& scales,
                                const Context& context,
                                const AtomicPortal& failed,
                                viskores::Id& pieces) const
  {
    pieces = 0;
    if (failed.Get(c) != 0)
    {
      return;
    }
    const viskores::Id2 pair = pairs.Get(c);
    const viskores::Vec<Real, 2> scale = scales.Get(c);
    bool complete = true;
    if (!context.TestEnvelopeItem(
          pair[0], pair[1], merged.Get(c), t, tests, scale[0], scale[1], pieces, complete))
    {
      failed.Set(c, 1);
    }
    if (!complete)
    {
      this->RaiseError("QuadricClusteringDecimation: bounding volume hierarchy too deep.");
    }
  }
};

struct ClearFailed : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn failed, FieldInOut admissible);
  using ExecutionSignature = void(_1, _2);

  VISKORES_EXEC void operator()(viskores::UInt32 failed, viskores::UInt8& admissible) const
  {
    if (failed != 0)
    {
      admissible = 0;
    }
  }
};

} // namespace detail

/// The local tests of the merges `pairs`, with the merged generators and the boxes of what every
/// test reads (`EvaluateMerge`). The test but for the envelope runs per merge; the
/// envelope tests run per input triangle, in batches of at most `maxItems` triangles, so that a
/// few merges of large clusters keep every thread busy: first the triangles with a vertex in
/// the merged cluster, where a merge that fails mostly fails, then the others, for the merges
/// still standing.
inline void EvaluateMerges(const viskores::cont::ArrayHandle<viskores::Id2>& pairs,
                           const MergeContextObject& context,
                           viskores::cont::ArrayHandle<viskores::UInt8>& admissible,
                           viskores::cont::ArrayHandle<Vec3>& merged,
                           viskores::cont::ArrayHandle<Vec3>& lower,
                           viskores::cont::ArrayHandle<Vec3>& upper,
                           MergeDiagnostics& diagnostics,
                           viskores::Id maxItems = viskores::Id(1) << 24)
{
  using Algorithm = viskores::cont::Algorithm;
  viskores::cont::Invoker invoke;
  viskores::cont::ArrayHandle<viskores::Id2> numItems;
  viskores::cont::ArrayHandle<viskores::Vec<Real, 2>> scales;
  invoke(detail::EvaluateMerge{}, pairs, context, admissible, merged, numItems, scales, lower, upper);
  const viskores::Id n = pairs.GetNumberOfValues();
  viskores::cont::ArrayHandle<viskores::UInt32> failed;
  failed.AllocateAndFill(n, 0);
  viskores::cont::ArrayHandle<viskores::Id> counts, offsets, candidates, triangles, pieces;
  viskores::cont::ArrayHandle<viskores::UInt8> tests;
  for (viskores::IdComponent part = 0; part < 2; ++part)
  {
    invoke(detail::PartCount(part), numItems, failed, counts);
    const viskores::Id total = Algorithm::ScanExclusive(counts, offsets);
    if (total == 0)
    {
      continue;
    }
    std::vector<viskores::Id> start(static_cast<std::size_t>(n) + 1);
    {
      auto portal = offsets.ReadPortal();
      for (viskores::Id c = 0; c < n; ++c)
      {
        start[static_cast<std::size_t>(c)] = portal.Get(c);
      }
      start[static_cast<std::size_t>(n)] = total;
    }
    for (viskores::Id first = 0; first < n;)
    {
      viskores::Id last = first + 1;
      while (last < n &&
             start[static_cast<std::size_t>(last) + 1] - start[static_cast<std::size_t>(first)] <=
               maxItems)
      {
        ++last;
      }
      const viskores::Id base = start[static_cast<std::size_t>(first)];
      const viskores::Id size = start[static_cast<std::size_t>(last)] - base;
      if (size > 0)
      {
        candidates.Allocate(size);
        triangles.Allocate(size);
        tests.Allocate(size);
        invoke(detail::FillEnvelopeItems(viskores::UInt8(1 << part), first, last, base),
               pairs,
               merged,
               counts,
               offsets,
               context,
               candidates,
               triangles,
               tests);
        invoke(detail::TestEnvelopeItems{},
               candidates,
               triangles,
               tests,
               pairs,
               merged,
               scales,
               context,
               failed,
               pieces);
        diagnostics.EnvelopePieces += Algorithm::Reduce(pieces, viskores::Id(0));
      }
      first = last;
    }
  }
  invoke(detail::ClearFailed{}, failed, admissible);
}


namespace detail
{

/// Marks the cluster that a merge `(i, j)` became, by the label its anchor has after the merge.
struct MarkMerged : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn pair, WholeArrayIn anchors, WholeArrayIn nextLabels, WholeArrayOut changed);
  using ExecutionSignature = void(_1, _2, _3, _4);

  template <typename IdPortal, typename FlagPortal>
  VISKORES_EXEC void operator()(const viskores::Id2& pair,
                                const IdPortal& anchors,
                                const IdPortal& nextLabels,
                                const FlagPortal& changed) const
  {
    changed.Set(nextLabels.Get(anchors.Get(pair[0])), 1);
  }
};

struct HasChangedLabel : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn triangle, WholeArrayIn changed, FieldOut isNew);
  using ExecutionSignature = _3(_1, _2);

  template <typename FlagPortal>
  VISKORES_EXEC viskores::UInt8 operator()(const viskores::Id3& t, const FlagPortal& changed) const
  {
    return (changed.Get(t[0]) || changed.Get(t[1]) || changed.Get(t[2])) ? 1 : 0;
  }
};

/// For a new triangle `t`: whether it is degenerate or meets another nondegenerate triangle
/// improperly, as `FindImproperPairs` decides.
struct NewTriangleFlagged : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn index,
                                WholeArrayIn triangles,
                                WholeArrayIn lower,
                                WholeArrayIn upper,
                                WholeArrayIn degenerates,
                                WholeArrayIn generators,
                                ExecObject hierarchy,
                                FieldOut flagged);
  using ExecutionSignature = void(_1, _2, _3, _4, _5, _6, _7, _8);

  template <typename Id3Portal, typename VecPortal, typename FlagPortal, typename Hierarchy>
  VISKORES_EXEC void operator()(viskores::Id t,
                                const Id3Portal& triangles,
                                const VecPortal& lower,
                                const VecPortal& upper,
                                const FlagPortal& degenerates,
                                const VecPortal& generators,
                                const Hierarchy& hierarchy,
                                viskores::UInt8& flagged) const
  {
    flagged = degenerates.Get(t);
    if (flagged)
    {
      return;
    }
    const viskores::Id3 triangle = triangles.Get(t);
    const Vec3 p[3] = { generators.Get(triangle[0]), generators.Get(triangle[1]), generators.Get(triangle[2]) };
    const bool complete = hierarchy.ForEachOverlap(lower.Get(t),
                                                   upper.Get(t),
                                                   [&](viskores::Id j)
                                                   {
                                                     if (j == t || degenerates.Get(j))
                                                     {
                                                       return true;
                                                     }
                                                     const viskores::Id3 other = triangles.Get(j);
                                                     const Vec3 q[3] = { generators.Get(other[0]),
                                                                         generators.Get(other[1]),
                                                                         generators.Get(other[2]) };
                                                     if (exact::ImproperPair(p, triangle, q, other))
                                                     {
                                                       flagged = 1;
                                                       return false;
                                                     }
                                                     return true;
                                                   });
    if (!complete)
    {
      this->RaiseError("QuadricClusteringDecimation: bounding volume hierarchy too deep.");
    }
  }
};

} // namespace detail

/// The clusters of `next` that the merges `pairs` of `previous` made.
inline viskores::cont::ArrayHandle<viskores::UInt8> MergedClusters(const Clustering& previous,
                                                                    const Clustering& next,
                                                                    const viskores::cont::ArrayHandle<viskores::Id2>& pairs)
{
  viskores::cont::ArrayHandle<viskores::UInt8> changed;
  changed.AllocateAndFill(next.NumberOfClusters, 0);
  viskores::cont::Invoker invoke;
  invoke(detail::MarkMerged{}, pairs, previous.Anchors, next.Labels, changed);
  return changed;
}

/// Merges the pairs `pairs` (each `(i, j)` merging `j` into `i`) and places the generators.
inline Clustering ApplyMerges(const SurfaceMesh& mesh,
                              const VertexQuadrics& quadrics,
                              const MergeParameters& parameters,
                              const Clustering& clusters,
                              const viskores::cont::ArrayHandle<viskores::Id2>& pairs)
{
  viskores::cont::Invoker invoke;
  viskores::cont::ArrayHandle<viskores::Id> map;
  invoke(detail::IdentityMap{}, viskores::cont::ArrayHandleIndex(clusters.NumberOfClusters), map);
  invoke(detail::MergeInto{}, pairs, map);
  Clustering merged;
  invoke(detail::ApplyMap{}, clusters.Labels, map, merged.Labels);
  PlaceGlued(mesh, quadrics, parameters, merged);
  return merged;
}

/// The input triangles whose envelope test a change of the labelling can change, for a caller
/// that knows the labelling before it passed the test with scales `ScaleInput` and
/// `ScaleOutput` (`EnvelopeReport`).
struct EnvelopeScope
{
  viskores::cont::ArrayHandle<viskores::UInt8> Triangles;
  Real ScaleInput;
  Real ScaleOutput;
};

/// Whether any flag in use fires on a placed labelling, evaluated in full: the discrete
/// closed-ball property, (F0), (F4), (F5) and (F6). With `scope`, (F5) tests only its triangles while
/// the scales of the whole test are those of `scope`, and in full otherwise.
///
/// With `changed`, the clusters that merges made from a labelling on which no flag fired, (F4)
/// and (F6) test only the triangles at them: every other triangle, and every pair of them, is as
/// it was, and a failing triangle or pair flags a cluster only if one of its clusters has two or
/// more vertices, which a merged cluster has.
inline bool FlagsFire(const SurfaceMesh& mesh,
                      const VertexQuadrics& quadrics,
                      const DistanceMesh& inputDistance,
                      const Clustering& clusters,
                      const MergeParameters& parameters,
                      MergeDiagnostics& diagnostics,
                      const EnvelopeScope* scope = nullptr,
                      const viskores::cont::ArrayHandle<viskores::UInt8>* changed = nullptr)
{
  const viskores::Id k = clusters.NumberOfClusters;
  viskores::cont::ArrayHandle<viskores::UInt8> flags;
  if (parameters.Topological)
  {
    flags.AllocateAndFill(k, 0);
    const TopologyReport report =
      TopologyFlags(mesh, clusters.Labels, clusters.Sizes, flags, parameters.D2ByCount);
    if (report.NumberOfD0 + report.NumberOfD1 + report.NumberOfD2 > 0)
    {
      return true;
    }
    if (parameters.Curves)
    {
      // (d'): any violation fires.
      if (GlueViolations(mesh,
                         clusters.Labels,
                         parameters.Original,
                         *parameters.Curves,
                         OutputGlue(parameters, clusters))
            .GetNumberOfValues() > 0)
      {
        return true;
      }
    }
    else if (GlueFlags(mesh,
                       clusters.Labels,
                       parameters.FrozenVertices,
                       parameters.Canonical,
                       clusters.Sizes,
                       flags) > 0)
    {
      return true;
    }
  }
  if (parameters.MaxClusterRadius < std::numeric_limits<Real>::infinity())
  {
    flags.AllocateAndFill(k, 0);
    if (RadiusFlags(Scan(mesh, quadrics.M, clusters), parameters.MaxClusterRadius, flags) > 0)
    {
      return true;
    }
  }
  const viskores::cont::ArrayHandle<viskores::Id3> triangles = parameters.Topological
    ? OutputTriangles(mesh, clusters.Labels)
    : Simplices(mesh, clusters.Labels);
  if (parameters.Embedding && parameters.Topological)
  {
    const viskores::cont::ArrayHandle<viskores::Id3> glued =
      GlueTriangles(triangles, OutputGlue(parameters, clusters));
    if (changed)
    {
      viskores::cont::Invoker invoke;
      viskores::cont::ArrayHandle<Vec3> lower, upper;
      viskores::cont::ArrayHandle<viskores::UInt8> degenerate, isNew, flagged;
      invoke(detail::RealizeTriangle{}, glued, clusters.Generators, lower, upper, degenerate);
      invoke(detail::HasChangedLabel{}, glued, *changed, isNew);
      viskores::cont::ArrayHandle<viskores::Id> newTriangles;
      viskores::cont::Algorithm::CopyIf(viskores::cont::ArrayHandleIndex(glued.GetNumberOfValues()), isNew, newTriangles);
      const viskores::worklet::spatialstructure::BoundingVolumeHierarchy hierarchy(lower, upper);
      invoke(detail::NewTriangleFlagged{},
             newTriangles,
             glued,
             lower,
             upper,
             degenerate,
             clusters.Generators,
             hierarchy,
             flagged);
      if (viskores::cont::Algorithm::Reduce(flagged, viskores::Id(0)) > 0)
      {
        return true;
      }
    }
    else
    {
      flags.AllocateAndFill(k, 0);
      EmbeddingFlags(glued, clusters.Generators, clusters.Sizes, flags);
      if (CountFlags(flags) > 0)
      {
        return true;
      }
    }
  }
  if (parameters.ErrorBound > 0)
  {
    auto envelope = [&](const viskores::cont::ArrayHandle<viskores::UInt8>* subset)
    {
      flags.AllocateAndFill(k, 0);
      const EnvelopeReport report =
        EnvelopeFlags(mesh,
                      inputDistance,
                      clusters.Labels,
                      clusters.Generators,
                      triangles,
                      clusters.Sizes,
                      parameters.ErrorBound,
                      parameters.Topological ? EnvelopeMode::Surface : EnvelopeMode::Simplices,
                      parameters.Sides,
                      flags,
                      subset);
      diagnostics.EnvelopePieces += report.Pieces;
      return report;
    };
    const EnvelopeReport report = envelope(scope ? &scope->Triangles : nullptr);
    if (scope && (report.ScaleInput != scope->ScaleInput || report.ScaleOutput != scope->ScaleOutput))
    {
      // Other margins: the triangles outside the scope must be tested again.
      envelope(nullptr);
    }
    if (CountFlags(flags) > 0)
    {
      return true;
    }
  }
  if (parameters.Isotopy)
  {
    flags.AllocateAndFill(k, 0);
    IsotopyFlags(mesh,
                 *parameters.Isotopy,
                 clusters.Labels,
                 clusters.Generators,
                 clusters.Sizes,
                 flags,
                 changed);
    if (CountFlags(flags) > 0)
    {
      return true;
    }
  }
  return false;
}

/// The clusters that the flags in use flag on a placed labelling, as `flagged_clusters` of
/// proofs/lib/vqc.py: (F0), the glue condition, (F4), (F5) and (F6). Returns false, with no flags, if
/// the discrete closed-ball property fails, which Lemma "mergeindependent" excludes for a
/// committed set.
inline bool FlaggedClusters(const SurfaceMesh& mesh,
                            const VertexQuadrics& quadrics,
                            const DistanceMesh& inputDistance,
                            const Clustering& clusters,
                            const MergeParameters& parameters,
                            MergeDiagnostics& diagnostics,
                            viskores::cont::ArrayHandle<viskores::UInt8>& flags)
{
  const viskores::Id k = clusters.NumberOfClusters;
  flags.AllocateAndFill(k, 0);
  viskores::cont::ArrayHandle<viskores::UInt8> other;
  if (parameters.Topological)
  {
    other.AllocateAndFill(k, 0);
    const TopologyReport report =
      TopologyFlags(mesh, clusters.Labels, clusters.Sizes, other, parameters.D2ByCount);
    if (report.NumberOfD0 + report.NumberOfD1 + report.NumberOfD2 > 0)
    {
      return false;
    }
    other.AllocateAndFill(k, 0);
    if (parameters.Curves)
    {
      viskores::cont::ArrayHandle<viskores::Id> listed =
        GlueViolations(mesh,
                       clusters.Labels,
                       parameters.Original,
                       *parameters.Curves,
                       OutputGlue(parameters, clusters));
      FlagClusters(listed, clusters.Sizes, other);
    }
    else
    {
      GlueFlags(mesh,
                clusters.Labels,
                parameters.FrozenVertices,
                parameters.Canonical,
                clusters.Sizes,
                other);
    }
    OrFlags(flags, other);
  }
  if (parameters.MaxClusterRadius < std::numeric_limits<Real>::infinity())
  {
    other.AllocateAndFill(k, 0);
    RadiusFlags(Scan(mesh, quadrics.M, clusters), parameters.MaxClusterRadius, other);
    OrFlags(flags, other);
  }
  const viskores::cont::ArrayHandle<viskores::Id3> triangles = parameters.Topological
    ? OutputTriangles(mesh, clusters.Labels)
    : Simplices(mesh, clusters.Labels);
  if (parameters.Embedding && parameters.Topological)
  {
    other.AllocateAndFill(k, 0);
    EmbeddingFlags(GlueTriangles(triangles, OutputGlue(parameters, clusters)),
                   clusters.Generators,
                   clusters.Sizes,
                   other);
    OrFlags(flags, other);
  }
  if (parameters.ErrorBound > 0)
  {
    other.AllocateAndFill(k, 0);
    const EnvelopeReport report =
      EnvelopeFlags(mesh,
                    inputDistance,
                    clusters.Labels,
                    clusters.Generators,
                    triangles,
                    clusters.Sizes,
                    parameters.ErrorBound,
                    parameters.Topological ? EnvelopeMode::Surface : EnvelopeMode::Simplices,
                    parameters.Sides,
                    other);
    diagnostics.EnvelopePieces += report.Pieces;
    OrFlags(flags, other);
  }
  if (parameters.Isotopy)
  {
    other.AllocateAndFill(k, 0);
    IsotopyFlags(
      mesh, *parameters.Isotopy, clusters.Labels, clusters.Generators, clusters.Sizes, other);
    OrFlags(flags, other);
  }
  return true;
}

namespace detail
{

/// Whether the cluster that a merge `(i, j)` became is flagged, by the label its anchor has after
/// the merge.
struct MergedFlagged : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn pair,
                                WholeArrayIn anchors,
                                WholeArrayIn nextLabels,
                                WholeArrayIn flags,
                                FieldOut flagged);
  using ExecutionSignature = _5(_1, _2, _3, _4);

  template <typename IdPortal, typename FlagPortal>
  VISKORES_EXEC viskores::UInt8 operator()(const viskores::Id2& pair,
                                           const IdPortal& anchors,
                                           const IdPortal& nextLabels,
                                           const FlagPortal& flags) const
  {
    return flags.Get(nextLabels.Get(anchors.Get(pair[0])));
  }
};

/// Marks the deferred candidates not admissible for the rest of the round.
struct Defer : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn rank, WholeArrayOut memo);
  using ExecutionSignature = void(_1, _2);

  template <typename MemoPortal>
  VISKORES_EXEC void operator()(viskores::Id rank, MemoPortal& memo) const
  {
    memo.Set(rank, 2);
  }
};

/// Per component of the mesh, which is a component of `K`: its clusters, and those on the
/// boundary. Integer atomics give the same counts in any order.
struct CountComponentClusters : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn anchor,
                                FieldIn boundary,
                                WholeArrayIn vertexComponents,
                                AtomicArrayInOut sizes,
                                AtomicArrayInOut boundaries);
  using ExecutionSignature = void(_1, _2, _3, _4, _5);

  template <typename IdPortal, typename Atomic>
  VISKORES_EXEC void operator()(viskores::Id anchor,
                                viskores::UInt8 boundary,
                                const IdPortal& vertexComponents,
                                Atomic& sizes,
                                Atomic& boundaries) const
  {
    const viskores::Id component = vertexComponents.Get(anchor);
    sizes.Add(component, 1);
    if (boundary)
    {
      boundaries.Add(component, 1);
    }
  }
};

/// Whether an edge of `K` lies in a component that is neither a triangle nor the boundary of a
/// tetrahedron.
struct InReducibleComponent : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn edge,
                                WholeArrayIn anchors,
                                WholeArrayIn vertexComponents,
                                WholeArrayIn sizes,
                                WholeArrayIn boundaries,
                                FieldOut reducible);
  using ExecutionSignature = _6(_1, _2, _3, _4, _5);

  template <typename IdPortal>
  VISKORES_EXEC viskores::UInt8 operator()(const viskores::Id2& edge,
                                           const IdPortal& anchors,
                                           const IdPortal& vertexComponents,
                                           const IdPortal& sizes,
                                           const IdPortal& boundaries) const
  {
    const viskores::Id component = vertexComponents.Get(anchors.Get(edge[0]));
    const viskores::Id size = sizes.Get(component);
    return (size > 4 || (size == 4 && boundaries.Get(component) > 0)) ? 1 : 0;
  }
};

} // namespace detail

/// The merge phase of Algorithm 8: merges adjacent clusters in rounds while `k > k*` and some
/// merge is admissible. A round commits the greedy maximal set of admissible candidates in
/// priority order, each conflicting with no other, keeps the `k - k*` of highest priority, and
/// checks the merged labelling with every flag. If one fires, it defers the merges whose merged
/// cluster is flagged and chooses again without them, as deterministic reservations retry the
/// items that fail (Blelloch, Fineman, Gibbons and Shun, PPoPP 2012); if none or all of them are,
/// it merges only the candidate of highest priority, or, if that fails too, rejects it and
/// chooses again.
template <typename RoundObserver>
void MergePhase(const SurfaceMesh& mesh,
                const VertexQuadrics& quadrics,
                const DistanceMesh& inputDistance,
                const viskores::cont::ArrayHandle<viskores::UInt8>& frozenVertices,
                const MergeParameters& parameters,
                Clustering& clusters,
                MergeDiagnostics& diagnostics,
                RoundObserver&& observer)
{
  using Algorithm = viskores::cont::Algorithm;
  viskores::cont::Invoker invoke;
  const bool geometric = parameters.GeometricFlags();
  // With the singular curves decimated, the local test checks (d) and not (d'), so the full test
  // runs even with no geometric flag in use.
  const bool fullTest = geometric || parameters.Curves != nullptr;
  viskores::cont::Timer timer, roundTimer;
  VerdictCache verdicts;
  while (true)
  {
    timer.Start();
    roundTimer.Start();
    const viskores::Id deferredBefore = diagnostics.Deferred;
    const Clustering previous = clusters;
    const viskores::Id k = clusters.NumberOfClusters;
    if (k <= parameters.Budget)
    {
      break;
    }
    const OutputComplex complex = BuildOutputComplex(
      parameters.Topological ? OutputTriangles(mesh, clusters.Labels) : Simplices(mesh, clusters.Labels),
      k);
    if (parameters.Topological && (k <= 3 || (k == 4 && !complex.HasBoundary)))
    {
      break; // a triangle or a tetrahedron: irreducible
    }

    // Candidates in priority order.
    viskores::cont::ArrayHandle<viskores::Id2> candidateEdges = complex.Edges;
    viskores::cont::ArrayHandle<viskores::IdComponent> candidateCounts = complex.EdgeCounts;
    if (!parameters.Topological)
    {
      // Every pair of adjacent clusters: the label pairs of the cut edges of the input.
      viskores::cont::ArrayHandle<viskores::Id2> pairs;
      viskores::cont::ArrayHandle<viskores::Id3> records;
      invoke(detail::CutEdgeRecord{}, mesh.Edges, mesh.EdgeTriangles, clusters.Labels, pairs, records);
      viskores::cont::ArrayHandle<viskores::UInt8> isCut;
      invoke(detail::IsCutRecord{}, records, isCut);
      Algorithm::CopyIf(pairs, isCut, candidateEdges);
      Algorithm::Sort(candidateEdges);
      Algorithm::Unique(candidateEdges);
      candidateCounts.AllocateAndFill(candidateEdges.GetNumberOfValues(), 0);
    }
    else if (mesh.NumberOfComponents > 1)
    {
      // The local test decides the discrete closed-ball property only in a component of K with
      // at least four vertices that is not the boundary of a tetrahedron (Proposition "A merge
      // is an edge contraction"): a component that is a triangle or a tetrahedron is
      // irreducible, and none of its edges is a candidate. The components of K are those of
      // the mesh, which no cluster crosses.
      viskores::cont::ArrayHandle<viskores::Id> sizes, boundaries;
      sizes.AllocateAndFill(mesh.NumberOfComponents, 0);
      boundaries.AllocateAndFill(mesh.NumberOfComponents, 0);
      invoke(detail::CountComponentClusters{},
             clusters.Anchors,
             complex.BoundaryVertex,
             mesh.Components,
             sizes,
             boundaries);
      viskores::cont::ArrayHandle<viskores::UInt8> reducible;
      invoke(detail::InReducibleComponent{},
             candidateEdges,
             clusters.Anchors,
             mesh.Components,
             sizes,
             boundaries,
             reducible);
      viskores::cont::ArrayHandle<viskores::Id2> edges;
      viskores::cont::ArrayHandle<viskores::IdComponent> counts;
      Algorithm::CopyIf(candidateEdges, reducible, edges);
      Algorithm::CopyIf(candidateCounts, reducible, counts);
      candidateEdges = edges;
      candidateCounts = counts;
    }
    // Two candidates conflict if they share a cluster or an edge of K joins them. Without the
    // topological test, K is replaced by the adjacency of the clusters. The degrees of the
    // priority are taken in the same graph.
    viskores::cont::ArrayHandle<viskores::Id> conflictOffsets = complex.NeighborOffsets;
    viskores::cont::ArrayHandle<viskores::Id> conflictNeighbors = complex.Neighbors;
    if (!parameters.Topological)
    {
      viskores::cont::ArrayHandle<viskores::Id2> adjacency;
      invoke(detail::EdgeRecords{}, candidateEdges, viskores::cont::make_ArrayHandleGroupVec<2>(adjacency));
      BuildAdjacency(adjacency, k, conflictOffsets, conflictNeighbors);
    }
    // No merge involves a frozen cluster, or joins two copies of one frozen vertex.
    const viskores::cont::ArrayHandle<viskores::UInt8> frozenLabels =
      FrozenLabels(frozenVertices, clusters);
    viskores::cont::ArrayHandle<viskores::Id> contactOffsets;
    viskores::cont::ArrayHandle<viskores::Id3> contacts;
    if (frozenVertices.GetNumberOfValues() > 0)
    {
      contacts = FrozenContacts(mesh, clusters.Labels, frozenVertices, parameters.Canonical);
      viskores::cont::ArrayHandle<viskores::Id> contactClusters, unused;
      invoke(detail::SplitTriple{}, contacts, contactClusters);
      Algorithm::LowerBounds(contactClusters, viskores::cont::ArrayHandleIndex(k + 1), contactOffsets);
    }
    viskores::cont::ArrayHandle<viskores::UInt8> isCandidate;
    viskores::cont::ArrayHandle<MergePriority> records, candidates;
    invoke(detail::MergeCandidate(
             parameters.Topological, frozenLabels.GetNumberOfValues() > 0, parameters.DegreeFloor),
           candidateEdges,
           candidateCounts,
           complex.BoundaryVertex,
           complex.NeighborOffsets,
           complex.Neighbors,
           conflictOffsets,
           complex.Faces,
           frozenLabels,
           contactOffsets,
           contacts,
           clusters.Anchors,
           mesh.Points,
           clusters.Q,
           clusters.B,
           clusters.S,
           isCandidate,
           records);
    Algorithm::CopyIf(records, isCandidate, candidates);
    Algorithm::Sort(candidates);
    const viskores::Id numCandidates = candidates.GetNumberOfValues();
    viskores::cont::ArrayHandle<viskores::Id2> pairs, ends;
    invoke(detail::CandidatePair{}, candidates, pairs, viskores::cont::make_ArrayHandleGroupVec<2>(ends));
    viskores::cont::ArrayHandle<viskores::Id> touchingOffsets, touching;
    BuildAdjacency(ends, k, touchingOffsets, touching);

    // The local test reads the current labelling and K.
    MergeContextObject context;
    context.Mesh = &mesh;
    context.Clusters = &clusters;
    context.Complex = &complex;
    context.Input = inputDistance;
    context.MaxRadius = parameters.MaxClusterRadius;
    context.Epsilon = parameters.ErrorBound;
    context.Embedding = parameters.Embedding && parameters.Topological;
    context.Isotopy = parameters.Isotopy;
    context.Sides = parameters.Sides;
    context.Geometric = !parameters.Topological;
    context.Glue = OutputGlue(parameters, clusters);
    if (geometric)
    {
      invoke(detail::ExactlyDegenerate{}, complex.Triangles, clusters.Generators, context.OutputDegenerate);
      context.Output = BuildDistanceMesh(clusters.Generators, complex.Triangles);
      context.TopAbs = { -1, -1, -1 };
      context.TopLabel = { -1, -1, -1 };
      auto generators = clusters.Generators.ReadPortal();
      for (viskores::Id c = 0; c < k; ++c)
      {
        const Vec3 p = generators.Get(c);
        Real a = viskores::Max(viskores::Max(viskores::Abs(p[0]), viskores::Abs(p[1])), viskores::Abs(p[2]));
        viskores::Id label = c;
        for (std::size_t slot = 0; slot < 3; ++slot)
        {
          if (a > context.TopAbs[slot])
          {
            std::swap(a, context.TopAbs[slot]);
            std::swap(label, context.TopLabel[slot]);
          }
        }
      }
    }

    timer.Stop();
    diagnostics.SecondsCandidates += timer.GetElapsedTime();
    timer.Start();
    viskores::cont::ArrayHandle<viskores::UInt8> memo; // 0 unknown, 1 admissible, 2 not
    memo.AllocateAndFill(numCandidates, geometric ? 0 : 1);
    if (geometric)
    {
      diagnostics.Reused += verdicts.Reuse(context, clusters, pairs, memo);
    }
    viskores::cont::ArrayHandle<viskores::Id2> committedPairs;
    while (true)
    {
      // The greedy maximal set in priority order, built in passes.
      viskores::cont::ArrayHandle<viskores::UInt8> states;
      states.AllocateAndFill(numCandidates, detail::Alive);
      viskores::cont::ArrayHandle<viskores::UInt8> ready, needed, alive;
      while (true)
      {
        invoke(detail::IsState(detail::Alive), states, alive);
        if (Algorithm::Reduce(alive, viskores::Id(0)) == 0)
        {
          break;
        }
        invoke(detail::ReadyCandidates{},
               pairs,
               states,
               states,
               conflictOffsets,
               conflictNeighbors,
               touchingOffsets,
               touching,
               ready);
        invoke(detail::NeedsEvaluation{}, ready, memo, needed);
        viskores::cont::ArrayHandle<viskores::Id> toEvaluate;
        Algorithm::CopyIf(viskores::cont::ArrayHandleIndex(numCandidates), needed, toEvaluate);
        if (toEvaluate.GetNumberOfValues() > 0)
        {
          timer.Stop();
          diagnostics.SecondsPasses += timer.GetElapsedTime();
          timer.Start();
          viskores::cont::ArrayHandle<viskores::Id2> evaluated;
          Algorithm::Copy(viskores::cont::make_ArrayHandlePermutation(toEvaluate, pairs), evaluated);
          viskores::cont::ArrayHandle<viskores::UInt8> admissible;
          viskores::cont::ArrayHandle<Vec3> mergedPositions, lower, upper;
          EvaluateMerges(evaluated, context, admissible, mergedPositions, lower, upper, diagnostics);
          invoke(detail::ScatterAdmissible{}, toEvaluate, admissible, memo);
          verdicts.Remember(context, clusters, evaluated, admissible, mergedPositions, lower, upper);
          diagnostics.Evaluated += toEvaluate.GetNumberOfValues();
          timer.Stop();
          diagnostics.SecondsLocalTests += timer.GetElapsedTime();
          timer.Start();
        }
        invoke(detail::DecideReady{}, ready, memo, states);
        viskores::cont::ArrayHandle<viskores::UInt8> decided;
        Algorithm::Copy(states, decided);
        invoke(detail::KillConflicting{},
               pairs,
               states,
               decided,
               conflictOffsets,
               conflictNeighbors,
               touchingOffsets,
               touching);
      }
      viskores::cont::ArrayHandle<viskores::UInt8> committed;
      invoke(detail::IsState(detail::Committed), states, committed);
      viskores::cont::ArrayHandle<viskores::Id> committedRanks;
      Algorithm::CopyIf(viskores::cont::ArrayHandleIndex(numCandidates), committed, committedRanks);
      const viskores::Id numCommitted =
        viskores::Min(committedRanks.GetNumberOfValues(),
                      parameters.Budget > 0 ? k - parameters.Budget : k);
      if (numCommitted == 0)
      {
        committedPairs.ReleaseResources();
        break;
      }
      Algorithm::Copy(viskores::cont::make_ArrayHandlePermutation(
                        viskores::cont::make_ArrayHandleView(committedRanks, 0, numCommitted), pairs),
                      committedPairs);
      timer.Stop();
      diagnostics.SecondsPasses += timer.GetElapsedTime();
      timer.Start();
      Clustering merged = ApplyMerges(mesh, quadrics, parameters, clusters, committedPairs);
      const viskores::cont::ArrayHandle<viskores::UInt8> mergedClusters =
        MergedClusters(clusters, merged, committedPairs);
      const bool fires =
        fullTest &&
        FlagsFire(
          mesh, quadrics, inputDistance, merged, parameters, diagnostics, nullptr, &mergedClusters);
      timer.Stop();
      diagnostics.SecondsFullTests += timer.GetElapsedTime();
      timer.Start();
      if (!fires)
      {
        clusters = merged;
        break;
      }
      // The set fired a flag. A cluster that no merge made keeps its members and generator, so
      // the flags fire at a merged cluster, unless (F5) flags an unmerged one from the input side:
      // defer the merges whose merged cluster is flagged, and choose again without them.
      if (parameters.DeferFlagged && numCommitted > 1)
      {
        viskores::cont::ArrayHandle<viskores::UInt8> flags, culprit;
        if (FlaggedClusters(mesh, quadrics, inputDistance, merged, parameters, diagnostics, flags))
        {
          invoke(detail::MergedFlagged{},
                 committedPairs,
                 clusters.Anchors,
                 merged.Labels,
                 flags,
                 culprit);
          viskores::cont::ArrayHandle<viskores::Id> culprits;
          Algorithm::CopyIf(viskores::cont::make_ArrayHandleView(committedRanks, 0, numCommitted),
                            culprit,
                            culprits);
          const viskores::Id numCulprits = culprits.GetNumberOfValues();
          if (numCulprits > 0 && numCulprits < numCommitted)
          {
            invoke(detail::Defer{}, culprits, memo);
            diagnostics.Deferred += numCulprits;
            timer.Stop();
            diagnostics.SecondsFullTests += timer.GetElapsedTime();
            timer.Start();
            continue;
          }
        }
        timer.Stop();
        diagnostics.SecondsFullTests += timer.GetElapsedTime();
        timer.Start();
      }
      // Merge only the candidate of highest priority, if that passes.
      const viskores::Id first = viskores::cont::ArrayGetValue(0, committedRanks);
      viskores::cont::ArrayHandle<viskores::Id2> single;
      single.Allocate(1);
      single.WritePortal().Set(0, viskores::cont::ArrayGetValue(first, pairs));
      Clustering one = ApplyMerges(mesh, quadrics, parameters, clusters, single);
      const viskores::cont::ArrayHandle<viskores::UInt8> oneCluster = MergedClusters(clusters, one, single);
      const bool oneFires =
        FlagsFire(mesh, quadrics, inputDistance, one, parameters, diagnostics, nullptr, &oneCluster);
      timer.Stop();
      diagnostics.SecondsFullTests += timer.GetElapsedTime();
      timer.Start();
      if (!oneFires)
      {
        diagnostics.Fallbacks += (numCommitted > 1) ? 1 : 0;
        clusters = one;
        committedPairs = single;
        break;
      }
      ++diagnostics.Dropped;
      memo.WritePortal().Set(first, 2);
    }
    if (committedPairs.GetNumberOfValues() == 0)
    {
      timer.Stop();
      diagnostics.SecondsPasses += timer.GetElapsedTime();
      break;
    }
    verdicts.Forget(mesh, complex, previous, clusters, committedPairs);
    timer.Stop();
    diagnostics.SecondsPasses += timer.GetElapsedTime();
    ++diagnostics.Rounds;
    observer(clusters, committedPairs);
    roundTimer.Stop();
    diagnostics.RoundMerges.push_back(committedPairs.GetNumberOfValues());
    diagnostics.RoundDeferred.push_back(diagnostics.Deferred - deferredBefore);
    diagnostics.RoundSeconds.push_back(roundTimer.GetElapsedTime());
  }
}

}
}
} // namespace viskores::worklet::quadric_clustering

VISKORES_QUADRIC_CLUSTERING_STRICT_FP_END

#endif // viskores_worklet_quadric_clustering_MergePhase_h
