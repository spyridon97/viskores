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
#include <viskores/filter/geometry_refinement/worklet/quadric_clustering/QuadricClusteringCommonWorklets.h>
#include <viskores/filter/geometry_refinement/worklet/quadric_clustering/TopologyFlags.h>
#include <viskores/filter/geometry_refinement/worklet/quadric_clustering/VertexQuadrics.h>

#include <viskores/Pair.h>
#include <viskores/cont/Algorithm.h>
#include <viskores/cont/ArrayHandleConcatenate.h>
#include <viskores/cont/ArrayHandleConstant.h>
#include <viskores/cont/ArrayHandleGroupVec.h>
#include <viskores/cont/ArrayHandleIndex.h>
#include <viskores/cont/ArrayHandlePermutation.h>
#include <viskores/cont/ArrayHandleView.h>
#include <viskores/cont/ExecutionObjectBase.h>
#include <viskores/cont/Timer.h>
#include <viskores/worklet/Keys.h>
#include <viskores/worklet/WorkletReduceByKey.h>

#include <limits>
#include <vector>

VISKORES_QUADRIC_CLUSTERING_STRICT_FP_BEGIN

namespace viskores
{
namespace worklet
{
namespace quadric_clustering
{

/// The key of the merge priority after the degree.
enum class MergeOrder
{
  /// The increase of the objective.
  Cost,
  /// `HashPair` of the anchors of the two clusters: a pseudo-random order within each degree
  /// class, which takes fewer passes per round.
  Hash
};

/// What the merge test and the full test of a merge check.
struct MergeParameters
{
  bool GeometricFlags() const
  {
    return this->MaxClusterRadius < std::numeric_limits<Real>::infinity();
  }

  /// `k*`, or 0 without a target.
  viskores::Id Budget = 0;
  Real MaxClusterRadius = std::numeric_limits<Real>::infinity();
  /// (D2) by count of the clusters per closed component instead of by sort of the label triples.
  bool D2ByCount = false;
  /// The floor `D_0` of the degree key of the priority, or a negative value to order the
  /// candidates by the increase of the objective alone.
  viskores::Id DegreeFloor = 12;
  /// The key after the degree.
  MergeOrder Order = MergeOrder::Cost;
  /// When the committed set fails the full test: defer the merges whose merged cluster is
  /// flagged and test the others again, or, if false, merge only the candidate of highest
  /// priority.
  bool DeferFlagged = true;
};

struct MergeDiagnostics
{
  viskores::Id Rounds = 0;
  /// Rounds in which the committed set failed the full test and one merge was made instead.
  viskores::Id Fallbacks = 0;
  /// Merges deferred to a later round because their merged cluster was flagged.
  viskores::Id Deferred = 0;
  /// Local tests evaluated, and verdicts of earlier rounds reused instead.
  viskores::Id Evaluated = 0;
  viskores::Id Reused = 0;
  /// Wall-clock seconds: the local tests, and the full tests of the merged labellings.
  double SecondsLocalTests = 0;
  double SecondsFullTests = 0;
  /// Per round: the merges committed, the merges deferred, and the wall-clock seconds.
  std::vector<viskores::Id> RoundMerges;
  std::vector<viskores::Id> RoundDeferred;
  std::vector<double> RoundSeconds;
};

/// The data the local test of a merge reads, in the execution environment.
struct MergeContextPortals
{
  using VecPortal = viskores::cont::ArrayHandle<Vec3>::ReadPortalType;
  using IdPortal = viskores::cont::ArrayHandle<viskores::Id>::ReadPortalType;
  using Id3Portal = viskores::cont::ArrayHandle<viskores::Id3>::ReadPortalType;
  using MatPortal = viskores::cont::ArrayHandle<Mat3>::ReadPortalType;

  /// The local test of a merge of Section 11.5: (F0) on the merged
  /// cluster. Sets `merged` to the position of the merged generator.
  VISKORES_EXEC bool Admissible(viskores::Id i, viskores::Id j, Vec3& merged) const;

  VecPortal Points;
  IdPortal MemberOffsets;
  IdPortal Members;
  IdPortal Anchors;
  VecPortal Generators;
  /// The matrices `M_v` of the vertices, for the generator of a merged cluster.
  MatPortal VertexMatrices;
  Id3Portal OutputTriangles;
  IdPortal FanOffsets;
  IdPortal Fans;
  Real MaxRadius;
};

VISKORES_EXEC inline bool MergeContextPortals::Admissible(viskores::Id i,
                                                          viskores::Id j,
                                                          Vec3& merged) const
{
  Vec3 o, bm;
  Mat3 qm;
  // The generator step (b) gives the merged cluster, by the same sums: tree sums over its
  // members in increasing order, from the smallest (Theorem "budget" (v)).
  viskores::Id a = this->MemberOffsets.Get(i), aEnd = this->MemberOffsets.Get(i + 1);
  viskores::Id c = this->MemberOffsets.Get(j), cEnd = this->MemberOffsets.Get(j + 1);
  o = this->Points.Get(viskores::Min(this->Anchors.Get(i), this->Anchors.Get(j)));
  TreeSum<Mat3> qSum;
  TreeSum<Vec3> bSum;
  while (a < aEnd || c < cEnd)
  {
    const bool fromI = c >= cEnd || (a < aEnd && this->Members.Get(a) < this->Members.Get(c));
    const viskores::Id v = fromI ? this->Members.Get(a++) : this->Members.Get(c++);
    const Mat3 mv = this->VertexMatrices.Get(v);
    qSum.Push(mv);
    bSum.Push(MatVec(mv, Sub(this->Points.Get(v), o)));
  }
  qm = qSum.Result();
  bm = bSum.Result();
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
    if (LengthUp(radius) > this->MaxRadius)
    {
      return false;
    }
  }
  return true;
}

/// The output complex `K` of a labelling, with the incidences the merge phase reads.
struct OutputComplex
{
  /// The output triangles, as label triples in the order and orientation of Algorithm 5.
  viskores::cont::ArrayHandle<viskores::Id3> Triangles;
  /// Its edges, sorted pairs in lexicographic order, and the number of triangles at each.
  viskores::cont::ArrayHandle<viskores::Id2> Edges;
  viskores::cont::ArrayHandle<viskores::IdComponent> EdgeCounts;
  /// Whether each vertex of `K` lies on its boundary.
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

/// An execution object giving a worklet a `MergeContextPortals`.
struct MergeContextObject : viskores::cont::ExecutionObjectBase
{
  VISKORES_CONT MergeContextPortals PrepareForExecution(viskores::cont::DeviceAdapterId device,
                                                        viskores::cont::Token& token) const
  {
    MergeContextPortals portals;
    portals.Points = this->Mesh->Points.PrepareForInput(device, token);
    portals.MemberOffsets = this->Clusters->Keys.GetOffsets().PrepareForInput(device, token);
    portals.Members = this->Clusters->Keys.GetSortedValuesMap().PrepareForInput(device, token);
    portals.Anchors = this->Clusters->Anchors.PrepareForInput(device, token);
    portals.Generators = this->Clusters->Generators.PrepareForInput(device, token);
    portals.VertexMatrices = this->VertexMatrices.PrepareForInput(device, token);
    portals.OutputTriangles = this->Complex->Triangles.PrepareForInput(device, token);
    portals.FanOffsets = this->Complex->FanOffsets.PrepareForInput(device, token);
    portals.Fans = this->Complex->Fans.PrepareForInput(device, token);
    portals.MaxRadius = this->MaxRadius;
    return portals;
  }

  const SurfaceMesh* Mesh;
  const Clustering* Clusters;
  const OutputComplex* Complex;
  viskores::cont::ArrayHandle<Mat3> VertexMatrices;
  Real MaxRadius;
};

namespace detail
{

/// The verdict of the local test of a candidate, kept in `memo` across the passes of a round.
enum : viskores::UInt8
{
  Unknown = 0,
  Admissible = 1,
  Inadmissible = 2
};

/// A verdict that holds for candidate `pair`: sets `memo` and counts it.
struct ReuseVerdict : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn pair,
                                WholeArrayIn anchors,
                                WholeArrayIn keys,
                                WholeArrayIn admissible,
                                FieldInOut memo,
                                FieldOut reused);
  using ExecutionSignature = void(_1, _2, _3, _4, _5, _6);

  template <typename IdPortal, typename KeyPortal, typename FlagPortal>
  VISKORES_EXEC void operator()(const viskores::Id2& pair,
                                const IdPortal& anchors,
                                const KeyPortal& keys,
                                const FlagPortal& admissible,
                                viskores::UInt8& memo,
                                viskores::Id& reused) const
  {
    reused = 0;
    const viskores::Id e =
      FindSorted(keys, viskores::Id2(anchors.Get(pair[0]), anchors.Get(pair[1])));
    if (e >= 0)
    {
      memo = admissible.Get(e) ? Admissible : Inadmissible;
      reused = 1;
    }
  }
};

/// The key of an evaluated candidate.
struct VerdictEntry : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn pair, WholeArrayIn anchors, FieldOut key);
  using ExecutionSignature = void(_1, _2, _3);

  template <typename IdPortal>
  VISKORES_EXEC void operator()(const viskores::Id2& pair,
                                const IdPortal& anchors,
                                viskores::Id2& key) const
  {
    key = viskores::Id2(anchors.Get(pair[0]), anchors.Get(pair[1]));
  }
};

/// The key of an entry with its position, so that a sort keeps insertion order among equal keys.
struct OrderedKey : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn key, FieldOut ordered);
  using ExecutionSignature = _2(_1, WorkIndex);

  VISKORES_EXEC viskores::Id3 operator()(const viskores::Id2& key, viskores::Id position) const
  {
    return viskores::Id3(key[0], key[1], position);
  }
};

/// Whether a sorted entry is the last of its key, the one inserted last.
struct LastOfKey : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn index,
                                WholeArrayIn sorted,
                                FieldOut last,
                                FieldOut position);
  using ExecutionSignature = void(_1, _2, _3, _4);

  template <typename KeyPortal>
  VISKORES_EXEC void operator()(viskores::Id i,
                                const KeyPortal& sorted,
                                viskores::UInt8& last,
                                viskores::Id& position) const
  {
    const viskores::Id3 k = sorted.Get(i);
    position = k[2];
    last = (i + 1 == sorted.GetNumberOfValues() || sorted.Get(i + 1)[0] != k[0] ||
            sorted.Get(i + 1)[1] != k[1])
      ? 1
      : 0;
  }
};

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

/// The anchors, in the labelling before the merges, of the two clusters of a committed merge.
struct PairAnchors : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn pair, WholeArrayIn anchors, FieldOut pairAnchors);
  using ExecutionSignature = void(_1, _2, _3);

  template <typename IdPortal, typename OutVec>
  VISKORES_EXEC void operator()(const viskores::Id2& pair,
                                const IdPortal& anchors,
                                OutVec& out) const
  {
    out[0] = anchors.Get(pair[0]);
    out[1] = anchors.Get(pair[1]);
  }
};

/// Whether an entry still holds: neither of its clusters merged, and no box of a change meets
/// its box.
struct VerdictHolds : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn key,
                                FieldIn lower,
                                FieldIn upper,
                                WholeArrayIn mergedAnchors,
                                WholeArrayIn changeLower,
                                WholeArrayIn changeUpper,
                                FieldOut holds);
  using ExecutionSignature = _7(_1, _2, _3, _4, _5, _6);

  template <typename IdPortal, typename VecPortal>
  VISKORES_EXEC viskores::UInt8 operator()(const viskores::Id2& key,
                                           const Vec3& lower,
                                           const Vec3& upper,
                                           const IdPortal& mergedAnchors,
                                           const VecPortal& changeLower,
                                           const VecPortal& changeUpper) const
  {
    if (FindSorted(mergedAnchors, key[0]) >= 0 || FindSorted(mergedAnchors, key[1]) >= 0)
    {
      return 0;
    }
    for (viskores::Id c = 0; c < changeLower.GetNumberOfValues(); ++c)
    {
      const Vec3 l = changeLower.Get(c), u = changeUpper.Get(c);
      if (l[0] <= upper[0] && u[0] >= lower[0] && l[1] <= upper[1] && u[1] >= lower[1] &&
          l[2] <= upper[2] && u[2] >= lower[2])
      {
        return 0;
      }
    }
    return 1;
  }
};

} // namespace detail

/// The verdicts of the local tests, kept across the rounds of the merge phase. A local test
/// reads only what lies in the box of `EvaluateMerge`; its verdict holds while neither cluster
/// merges and no merge changes anything in the box. The clusters are
/// known by their anchors, which the renumbering of the labels keeps. The entries are kept
/// sorted by the anchors of the two clusters.
class VerdictCache
{
public:
  /// Sets `memo` of the candidates `pairs` whose verdicts hold. Returns how many.
  viskores::Id Reuse(const Clustering& clusters,
                     const viskores::cont::ArrayHandle<viskores::Id2>& pairs,
                     viskores::cont::ArrayHandle<viskores::UInt8>& memo) const
  {
    if (this->Keys.GetNumberOfValues() == 0)
    {
      return 0;
    }
    viskores::cont::Invoker invoke;
    viskores::cont::ArrayHandle<viskores::Id> reused;
    invoke(
      detail::ReuseVerdict{}, pairs, clusters.Anchors, this->Keys, this->Admissible, memo, reused);
    return viskores::cont::Algorithm::Reduce(reused, viskores::Id(0));
  }

  /// Keeps the verdicts of the local tests of `pairs`, in place of earlier ones of the same
  /// clusters.
  void Remember(const Clustering& clusters,
                const viskores::cont::ArrayHandle<viskores::Id2>& pairs,
                const viskores::cont::ArrayHandle<viskores::UInt8>& admissible,
                const viskores::cont::ArrayHandle<Vec3>& lower,
                const viskores::cont::ArrayHandle<Vec3>& upper)
  {
    using Algorithm = viskores::cont::Algorithm;
    viskores::cont::Invoker invoke;
    viskores::cont::ArrayHandle<viskores::Id2> keys;
    invoke(detail::VerdictEntry{}, pairs, clusters.Anchors, keys);
    // The old entries, then the new; the last entry of every key wins.
    auto allKeys = viskores::cont::make_ArrayHandleConcatenate(this->Keys, keys);
    viskores::cont::ArrayHandle<viskores::Id3> ordered;
    invoke(detail::OrderedKey{}, allKeys, ordered);
    Algorithm::Sort(ordered);
    viskores::cont::ArrayHandle<viskores::UInt8> last;
    viskores::cont::ArrayHandle<viskores::Id> positions, kept;
    invoke(detail::LastOfKey{},
           viskores::cont::ArrayHandleIndex(ordered.GetNumberOfValues()),
           ordered,
           last,
           positions);
    Algorithm::CopyIf(positions, last, kept);
    this->Gather(kept,
                 allKeys,
                 viskores::cont::make_ArrayHandleConcatenate(this->Lower, lower),
                 viskores::cont::make_ArrayHandleConcatenate(this->Upper, upper),
                 viskores::cont::make_ArrayHandleConcatenate(this->Admissible, admissible));
  }

  /// Forgets the verdicts that the merges `pairs` of the labelling `previous`, with output
  /// complex `complex`, into `next` may change.
  void Forget(const SurfaceMesh& mesh,
              const OutputComplex& complex,
              const Clustering& previous,
              const Clustering& next,
              const viskores::cont::ArrayHandle<viskores::Id2>& pairs)
  {
    using Algorithm = viskores::cont::Algorithm;
    if (this->Keys.GetNumberOfValues() == 0)
    {
      return;
    }
    viskores::cont::Invoker invoke;
    viskores::cont::ArrayHandle<Vec3> changeLower, changeUpper;
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
           changeLower,
           changeUpper);
    viskores::cont::ArrayHandle<viskores::Id> mergedAnchors;
    invoke(detail::PairAnchors{},
           pairs,
           previous.Anchors,
           viskores::cont::make_ArrayHandleGroupVec<2>(mergedAnchors));
    Algorithm::Sort(mergedAnchors);
    Algorithm::Unique(mergedAnchors);
    viskores::cont::ArrayHandle<viskores::UInt8> holds;
    invoke(detail::VerdictHolds{},
           this->Keys,
           this->Lower,
           this->Upper,
           mergedAnchors,
           changeLower,
           changeUpper,
           holds);
    viskores::cont::ArrayHandle<viskores::Id> kept;
    Algorithm::CopyIf(
      viskores::cont::ArrayHandleIndex(this->Keys.GetNumberOfValues()), holds, kept);
    this->Gather(kept, this->Keys, this->Lower, this->Upper, this->Admissible);
  }

private:
  /// Replaces the entries by the entries `kept` of the given arrays, in order.
  template <typename KeyArray, typename LowerArray, typename UpperArray, typename FlagArray>
  void Gather(const viskores::cont::ArrayHandle<viskores::Id>& kept,
              const KeyArray& keys,
              const LowerArray& lower,
              const UpperArray& upper,
              const FlagArray& admissible)
  {
    using Algorithm = viskores::cont::Algorithm;
    viskores::cont::ArrayHandle<viskores::Id2> newKeys;
    viskores::cont::ArrayHandle<Vec3> newLower, newUpper;
    viskores::cont::ArrayHandle<viskores::UInt8> newAdmissible;
    Algorithm::Copy(viskores::cont::make_ArrayHandlePermutation(kept, keys), newKeys);
    Algorithm::Copy(viskores::cont::make_ArrayHandlePermutation(kept, lower), newLower);
    Algorithm::Copy(viskores::cont::make_ArrayHandlePermutation(kept, upper), newUpper);
    Algorithm::Copy(viskores::cont::make_ArrayHandlePermutation(kept, admissible), newAdmissible);
    this->Keys = newKeys;
    this->Lower = newLower;
    this->Upper = newUpper;
    this->Admissible = newAdmissible;
  }

  viskores::cont::ArrayHandle<viskores::Id2> Keys;
  viskores::cont::ArrayHandle<Vec3> Lower;
  viskores::cont::ArrayHandle<Vec3> Upper;
  viskores::cont::ArrayHandle<viskores::UInt8> Admissible;
};

namespace detail
{

struct CountValues : viskores::worklet::WorkletReduceByKey
{
  using ControlSignature = void(KeysIn keys, ReducedValuesOut count);
  using ExecutionSignature = _2(ValueCount);
  using InputDomain = _1;

  VISKORES_EXEC viskores::IdComponent operator()(viskores::IdComponent count) const
  {
    return count;
  }
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
  invoke(SplitPair{}, records, keys, values);
  Algorithm::LowerBounds(keys, viskores::cont::ArrayHandleIndex(numKeys + 1), offsets);
}

namespace detail
{

struct CornerRecords : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn triangle, FieldOut records);
  using ExecutionSignature = void(_1, _2, WorkIndex);

  template <typename RecordsVec>
  VISKORES_EXEC void operator()(const viskores::Id3& t,
                                RecordsVec& records,
                                viskores::Id index) const
  {
    for (int c = 0; c < 3; ++c)
    {
      records[c] = viskores::Id2(t[c], index);
    }
  }
};

struct SortTriple : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn triangle, FieldOut sorted);
  using ExecutionSignature = _2(_1);

  VISKORES_EXEC viskores::Id3 operator()(const viskores::Id3& t) const { return SortThree(t); }
};

} // namespace detail

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
  invoke(SortedTriangleEdges{}, triangles, viskores::cont::make_ArrayHandleGroupVec<3>(sides));
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
  invoke(IsValidLabel{}, ends, valid);
  Algorithm::CopyIf(ends, valid, listed);
  complex.HasBoundary = listed.GetNumberOfValues() > 0;
  invoke(MarkIndex{}, listed, complex.BoundaryVertex);

  viskores::cont::ArrayHandle<viskores::Id2> edgeRecords;
  invoke(BothDirections{}, complex.Edges, viskores::cont::make_ArrayHandleGroupVec<2>(edgeRecords));
  BuildAdjacency(edgeRecords, k, complex.NeighborOffsets, complex.Neighbors);

  viskores::cont::ArrayHandle<viskores::Id2> cornerRecords;
  invoke(
    detail::CornerRecords{}, triangles, viskores::cont::make_ArrayHandleGroupVec<3>(cornerRecords));
  BuildAdjacency(cornerRecords, k, complex.FanOffsets, complex.Fans);

  invoke(detail::SortTriple{}, triangles, complex.Faces);
  Algorithm::Sort(complex.Faces);
  return complex;
}

namespace detail
{

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

/// The priority of a candidate merge `(i, j)`, smaller first: `max(D_0, deg i + deg j)`, the
/// degree sum of the two clusters with a floor (0 without the degree key), then the increase of
/// the objective (0 in hash order), then the hash of the anchors (0 in cost order), then the
/// labels.
using MergePriority =
  viskores::Pair<viskores::Id,
                 viskores::Pair<Real, viskores::Pair<viskores::UInt64, viskores::Id2>>>;

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

/// `Phi_C(c_C) = s_C - b_C^T Q_C^{-1} b_C`.
VISKORES_EXEC_CONT inline Real Phi(const Mat3& q, const Vec3& b, Real s)
{
  return s - Dot(b, SolveSPD(q, b));
}

namespace detail
{

/// Whether the edge `ij` of `K` is a candidate merge: the local test of Proposition "A merge is
/// an edge contraction" (ii) (not a linking edge, and in no non-facial 3-cycle); and its
/// `MergePriority`.
struct MergeCandidate : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn edge,
                                FieldIn count,
                                WholeArrayIn boundary,
                                WholeArrayIn neighborOffsets,
                                WholeArrayIn neighbors,
                                WholeArrayIn degreeOffsets,
                                WholeArrayIn faces,
                                WholeArrayIn anchors,
                                WholeArrayIn points,
                                WholeArrayIn q,
                                WholeArrayIn b,
                                WholeArrayIn s,
                                FieldOut isCandidate,
                                FieldOut record);
  using ExecutionSignature = void(_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14);

  VISKORES_CONT MergeCandidate(viskores::Id degreeFloor, MergeOrder order)
    : DegreeFloor(degreeFloor)
    , Order(order)
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
    record = MergePriority(0, { 0, { 0, edge } });
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
        viskores::Id lo = LowerBound(faces, 0, faces.GetNumberOfValues(), triple);
        if (lo == faces.GetNumberOfValues() || faces.Get(lo) != triple)
        {
          return; // a non-facial 3-cycle
        }
        ++a;
        ++c;
      }
    }
    Vec3 o, bm;
    Mat3 qm;
    Real sm;
    MergedSums(i, j, anchors, points, q, b, s, o, qm, bm, sm);
    const Real d =
      Phi(qm, bm, sm) - Phi(q.Get(i), b.Get(i), s.Get(i)) - Phi(q.Get(j), b.Get(j), s.Get(j));
    if (this->Order == MergeOrder::Hash)
    {
      record.second.second.first = HashPair(anchors.Get(i), anchors.Get(j));
    }
    else
    {
      record.second.first = viskores::Max(d, Real(0));
    }
    if (this->DegreeFloor >= 0)
    {
      const viskores::Id degrees = (degreeOffsets.Get(i + 1) - degreeOffsets.Get(i)) +
        (degreeOffsets.Get(j + 1) - degreeOffsets.Get(j));
      record.first = viskores::Max(this->DegreeFloor, degrees);
    }
    isCandidate = 1;
  }

  viskores::Id DegreeFloor;
  MergeOrder Order;
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
    pair = record.second.second.second;
    ends[0] = viskores::Id2(pair[0], rank);
    ends[1] = viskores::Id2(pair[1], rank);
  }
};


enum : viskores::UInt8
{
  Alive = 0,
  Committed = 1,
  Dead = 2,
  Rejected = 3
};

/// Visits, while `visit` returns true, the candidates that conflict with the merge `pair`: those
/// touching one of its two clusters or a neighbour of one in `K`.
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

struct NeedsEvaluation : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn ready, FieldIn memo, FieldOut needed);
  using ExecutionSignature = _3(_1, _2);

  VISKORES_EXEC viskores::UInt8 operator()(viskores::UInt8 ready, viskores::UInt8 memo) const
  {
    return (ready && memo == Unknown) ? 1 : 0;
  }
};

/// Grows the box `[lower, upper]` by a margin for the rounding of the bounds, `size` the largest
/// coordinate they come from.
VISKORES_EXEC_CONT inline void GrowBox(Vec3& lower, Vec3& upper, Real size)
{
  const Real grow = 64 * UnitRoundoff * size;
  for (int a = 0; a < 3; ++a)
  {
    lower[a] = lower[a] - grow;
    upper[a] = upper[a] + grow;
  }
}

/// The local test of a merge. Also a box that holds everything the local test reads: the
/// members of the two clusters, the generators of their triangles of `K` and the merged
/// generator. Nothing else of `K` or of the labelling is read.
struct EvaluateMerge : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn pair,
                                ExecObject context,
                                WholeArrayOut admissibleOut,
                                WholeArrayOut mergedOut,
                                WholeArrayOut lowerOut,
                                WholeArrayOut upperOut);
  using ExecutionSignature = void(InputIndex, VisitIndex, _1, _2, _3, _4, _5, _6);
  using ScatterType = viskores::worklet::ScatterCounting;

  template <typename Context, typename FlagPortal, typename VecPortal>
  VISKORES_EXEC void operator()(viskores::Id index,
                                viskores::IdComponent visit,
                                const viskores::Id2& pair,
                                const Context& context,
                                const FlagPortal& admissibleOut,
                                const VecPortal& mergedOut,
                                const VecPortal& lowerOut,
                                const VecPortal& upperOut) const
  {
    if (visit != 0)
    {
      return;
    }
    viskores::UInt8 admissible;
    Vec3 merged, lower, upper;
    this->Evaluate(pair, context, admissible, merged, lower, upper);
    admissibleOut.Set(index, admissible);
    mergedOut.Set(index, merged);
    lowerOut.Set(index, lower);
    upperOut.Set(index, upper);
  }

  template <typename Context>
  VISKORES_EXEC void Evaluate(const viskores::Id2& pair,
                              const Context& context,
                              viskores::UInt8& admissible,
                              Vec3& merged,
                              Vec3& lower,
                              Vec3& upper) const
  {
    const viskores::Id i = pair[0], j = pair[1];
    admissible = context.Admissible(i, j, merged) ? 1 : 0;
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
    Real size = 0;
    for (int a = 0; a < 3; ++a)
    {
      size = viskores::Max(size, viskores::Max(viskores::Abs(lower[a]), viskores::Abs(upper[a])));
    }
    GrowBox(lower, upper, size);
  }
};

} // namespace detail

/// The local tests of the merges `pairs`, with the merged generators and the boxes of what every
/// test reads (`EvaluateMerge`).
inline void EvaluateMerges(const viskores::cont::ArrayHandle<viskores::Id2>& pairs,
                           const MergeContextObject& context,
                           viskores::cont::ArrayHandle<viskores::UInt8>& admissible,
                           viskores::cont::ArrayHandle<Vec3>& merged,
                           viskores::cont::ArrayHandle<Vec3>& lower,
                           viskores::cont::ArrayHandle<Vec3>& upper)
{
  viskores::cont::Invoker invoke;
  const viskores::Id n = pairs.GetNumberOfValues();
  admissible.Allocate(n);
  merged.Allocate(n);
  lower.Allocate(n);
  upper.Allocate(n);
  invoke(detail::EvaluateMerge{},
         SpreadScatterOver(n),
         pairs,
         context,
         admissible,
         merged,
         lower,
         upper);
}

namespace detail
{

struct ScatterAdmissible : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn rank, FieldIn admissible, WholeArrayInOut memo);
  using ExecutionSignature = void(_1, _2, _3);

  template <typename MemoPortal>
  VISKORES_EXEC void operator()(viskores::Id rank,
                                viskores::UInt8 admissible,
                                MemoPortal& memo) const
  {
    memo.Set(rank, admissible ? Admissible : Inadmissible);
  }
};

/// A ready candidate is committed if admissible and rejected otherwise.
struct DecideReady : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn ready, FieldIn memo, FieldInOut state);
  using ExecutionSignature = void(_1, _2, _3);

  VISKORES_EXEC void operator()(viskores::UInt8 ready,
                                viskores::UInt8 memo,
                                viskores::UInt8& state) const
  {
    if (ready)
    {
      state = (memo == Admissible) ? Committed : Rejected;
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

} // namespace detail

/// Merges the pairs `pairs` (each `(i, j)` merging `j` into `i`) and places the generators.
inline Clustering ApplyMerges(const SurfaceMesh& mesh,
                              const VertexQuadrics& quadrics,
                              const Clustering& clusters,
                              const viskores::cont::ArrayHandle<viskores::Id2>& pairs)
{
  viskores::cont::Invoker invoke;
  viskores::cont::ArrayHandle<viskores::Id> map;
  viskores::cont::Algorithm::Copy(viskores::cont::ArrayHandleIndex(clusters.NumberOfClusters), map);
  invoke(detail::MergeInto{}, pairs, map);
  Clustering merged;
  invoke(GatherValue{}, clusters.Labels, map, merged.Labels);
  Place(mesh, quadrics.M, merged);
  return merged;
}

/// Whether any flag in use fires on a placed labelling, evaluated in full: the discrete
/// closed-ball property and (F0).
inline bool FlagsFire(const SurfaceMesh& mesh,
                      const VertexQuadrics& quadrics,
                      const Clustering& clusters,
                      const MergeParameters& parameters)
{
  const viskores::Id k = clusters.NumberOfClusters;
  viskores::cont::ArrayHandle<viskores::UInt8> flags;
  flags.AllocateAndFill(k, 0);
  const TopologyReport report =
    TopologyFlags(mesh, clusters.Labels, clusters.Sizes, flags, parameters.D2ByCount);
  if (report.NumberOfD0 + report.NumberOfD1 + report.NumberOfD2 > 0)
  {
    return true;
  }
  if (parameters.MaxClusterRadius < std::numeric_limits<Real>::infinity())
  {
    flags.AllocateAndFill(k, 0);
    if (RadiusFlags(Scan(mesh, quadrics.M, clusters), parameters.MaxClusterRadius, flags) > 0)
    {
      return true;
    }
  }
  return false;
}

/// The clusters that the flags in use flag on a placed labelling, as `flagged_clusters` of
/// proofs/lib/vqc.py: (F0). Returns
/// false, with no flags, if the discrete closed-ball property fails, which Lemma
/// "mergeindependent" excludes for a committed set of cluster merges.
inline bool FlaggedClusters(const SurfaceMesh& mesh,
                            const VertexQuadrics& quadrics,
                            const Clustering& clusters,
                            const MergeParameters& parameters,
                            viskores::cont::ArrayHandle<viskores::UInt8>& flags)
{
  const viskores::Id k = clusters.NumberOfClusters;
  flags.AllocateAndFill(k, 0);
  viskores::cont::ArrayHandle<viskores::UInt8> other;
  other.AllocateAndFill(k, 0);
  const TopologyReport report =
    TopologyFlags(mesh, clusters.Labels, clusters.Sizes, other, parameters.D2ByCount);
  if (report.NumberOfD0 + report.NumberOfD1 + report.NumberOfD2 > 0)
  {
    return false;
  }
  if (parameters.MaxClusterRadius < std::numeric_limits<Real>::infinity())
  {
    other.AllocateAndFill(k, 0);
    RadiusFlags(Scan(mesh, quadrics.M, clusters), parameters.MaxClusterRadius, other);
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

} // namespace detail

/// The merge phase of Algorithm 8: merges adjacent clusters in rounds while `k > k*` and some
/// merge is admissible. A round commits the greedy maximal set of admissible candidates in
/// priority order, each conflicting with no other, keeps the `k - k*` of highest priority, and
/// checks the merged labelling with every flag. If one fires, it defers the merges whose merged
/// cluster is flagged and chooses again without them, as
/// deterministic reservations retry the
/// items that fail (Blelloch, Fineman, Gibbons and Shun, PPoPP 2012); if none or all of them are,
/// it merges only the candidate of highest priority, or, if that fails too, rejects it and
/// chooses again.
template <typename RoundObserver>
void MergePhase(const SurfaceMesh& mesh,
                const VertexQuadrics& quadrics,
                const MergeParameters& parameters,
                Clustering& clusters,
                MergeDiagnostics& diagnostics,
                RoundObserver&& observer)
{
  using Algorithm = viskores::cont::Algorithm;
  viskores::cont::Invoker invoke;
  const bool geometric = parameters.GeometricFlags();
  viskores::cont::Timer timer, roundTimer;
  VerdictCache verdicts;
  while (true)
  {
    roundTimer.Start();
    const viskores::Id deferredBefore = diagnostics.Deferred;
    const Clustering previous = clusters;
    const viskores::Id k = clusters.NumberOfClusters;
    if (k <= parameters.Budget)
    {
      break;
    }
    const OutputComplex complex = BuildOutputComplex(OutputTriangles(mesh, clusters.Labels), k);
    if (k <= 3 || (k == 4 && !complex.HasBoundary))
    {
      break; // a triangle or a tetrahedron: irreducible
    }

    // Candidates in priority order.
    viskores::cont::ArrayHandle<viskores::Id2> candidateEdges = complex.Edges;
    viskores::cont::ArrayHandle<viskores::IdComponent> candidateCounts = complex.EdgeCounts;
    if (mesh.NumberOfComponents > 1)
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
    // Two candidates conflict if they share a cluster or an edge of K joins them. The degrees of
    // the priority are taken in the same graph.
    viskores::cont::ArrayHandle<viskores::Id> conflictOffsets = complex.NeighborOffsets;
    viskores::cont::ArrayHandle<viskores::Id> conflictNeighbors = complex.Neighbors;
    viskores::cont::ArrayHandle<viskores::UInt8> isCandidate;
    viskores::cont::ArrayHandle<MergePriority> records, candidates;
    invoke(detail::MergeCandidate(parameters.DegreeFloor, parameters.Order),
           candidateEdges,
           candidateCounts,
           complex.BoundaryVertex,
           complex.NeighborOffsets,
           complex.Neighbors,
           conflictOffsets,
           complex.Faces,
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
    invoke(detail::CandidatePair{},
           candidates,
           pairs,
           viskores::cont::make_ArrayHandleGroupVec<2>(ends));
    viskores::cont::ArrayHandle<viskores::Id> touchingOffsets, touching;
    BuildAdjacency(ends, k, touchingOffsets, touching);

    // The local test reads the current labelling and K.
    MergeContextObject context;
    context.Mesh = &mesh;
    context.Clusters = &clusters;
    context.Complex = &complex;
    context.VertexMatrices = quadrics.M;
    context.MaxRadius = parameters.MaxClusterRadius;

    viskores::cont::ArrayHandle<viskores::UInt8> memo;
    memo.AllocateAndFill(numCandidates, geometric ? detail::Unknown : detail::Admissible);
    if (geometric)
    {
      diagnostics.Reused += verdicts.Reuse(clusters, pairs, memo);
    }
    viskores::cont::ArrayHandle<viskores::Id2> committedPairs;
    // The merges deferred as culprits of a batch in this round, never tested alone.
    viskores::cont::ArrayHandle<viskores::UInt8> deferred;
    deferred.AllocateAndFill(numCandidates, 0);
    while (true)
    {
      // The greedy maximal set in priority order, built in passes.
      viskores::cont::ArrayHandle<viskores::UInt8> states;
      states.AllocateAndFill(numCandidates, detail::Alive);
      viskores::cont::ArrayHandle<viskores::UInt8> ready, needed, alive;
      while (true)
      {
        invoke(IsState(detail::Alive), states, alive);
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
          timer.Start();
          viskores::cont::ArrayHandle<viskores::Id2> evaluated;
          Algorithm::Copy(viskores::cont::make_ArrayHandlePermutation(toEvaluate, pairs),
                          evaluated);
          viskores::cont::ArrayHandle<viskores::UInt8> admissible;
          viskores::cont::ArrayHandle<Vec3> mergedPositions, lower, upper;
          EvaluateMerges(evaluated, context, admissible, mergedPositions, lower, upper);
          invoke(detail::ScatterAdmissible{}, toEvaluate, admissible, memo);
          verdicts.Remember(clusters, evaluated, admissible, lower, upper);
          diagnostics.Evaluated += toEvaluate.GetNumberOfValues();
          timer.Stop();
          diagnostics.SecondsLocalTests += timer.GetElapsedTime();
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
      invoke(IsState(detail::Committed), states, committed);
      viskores::cont::ArrayHandle<viskores::Id> committedRanks;
      Algorithm::CopyIf(viskores::cont::ArrayHandleIndex(numCandidates), committed, committedRanks);
      const viskores::Id numCommitted = viskores::Min(
        committedRanks.GetNumberOfValues(), parameters.Budget > 0 ? k - parameters.Budget : k);
      if (numCommitted == 0)
      {
        committedPairs.ReleaseResources();
        // Before the phase stops, the merges only deferred from a batch are tried alone, in
        // priority order (Theorem "budget" (v)).
        viskores::cont::ArrayHandle<viskores::Id> deferredRanks;
        Algorithm::CopyIf(viskores::cont::ArrayHandleIndex(numCandidates), deferred, deferredRanks);
        for (viskores::Id t = 0; t < deferredRanks.GetNumberOfValues(); ++t)
        {
          const viskores::Id rank = viskores::cont::ArrayGetValue(t, deferredRanks);
          viskores::cont::ArrayHandle<viskores::Id2> single;
          Algorithm::Copy(viskores::cont::make_ArrayHandleView(pairs, rank, 1), single);
          Clustering one = ApplyMerges(mesh, quadrics, clusters, single);
          if (!FlagsFire(mesh, quadrics, one, parameters))
          {
            clusters = one;
            committedPairs = single;
            break;
          }
        }
        break;
      }
      Algorithm::Copy(
        viskores::cont::make_ArrayHandlePermutation(
          viskores::cont::make_ArrayHandleView(committedRanks, 0, numCommitted), pairs),
        committedPairs);
      timer.Start();
      Clustering merged = ApplyMerges(mesh, quadrics, clusters, committedPairs);
      const bool fires = geometric && FlagsFire(mesh, quadrics, merged, parameters);
      timer.Stop();
      diagnostics.SecondsFullTests += timer.GetElapsedTime();
      timer.Start();
      if (!fires)
      {
        clusters = merged;
        break;
      }
      // The set fired a flag. A cluster that no merge made keeps its members and generator, so
      // the flags fire at a merged cluster: defer the merges whose merged cluster is flagged, and
      // choose again without them.
      if (parameters.DeferFlagged && numCommitted > 1)
      {
        viskores::cont::ArrayHandle<viskores::UInt8> flags, culprit;
        if (FlaggedClusters(mesh, quadrics, merged, parameters, flags))
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
            invoke(MarkIndex(detail::Inadmissible), culprits, memo);
            invoke(MarkIndex{}, culprits, deferred);
            diagnostics.Deferred += numCulprits;
            timer.Stop();
            diagnostics.SecondsFullTests += timer.GetElapsedTime();
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
      Algorithm::Copy(viskores::cont::make_ArrayHandleView(pairs, first, 1), single);
      Clustering one = ApplyMerges(mesh, quadrics, clusters, single);
      const bool oneFires = FlagsFire(mesh, quadrics, one, parameters);
      timer.Stop();
      diagnostics.SecondsFullTests += timer.GetElapsedTime();
      if (!oneFires)
      {
        diagnostics.Fallbacks += (numCommitted > 1) ? 1 : 0;
        clusters = one;
        committedPairs = single;
        break;
      }
      invoke(
        MarkIndex(detail::Inadmissible), viskores::cont::make_ArrayHandleConstant(first, 1), memo);
    }
    if (committedPairs.GetNumberOfValues() == 0)
    {
      break;
    }
    verdicts.Forget(mesh, complex, previous, clusters, committedPairs);
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
