//============================================================================
//  The contents of this file are covered by the Viskores license. See
//  LICENSE.txt for details.
//
//  By contributing to this file, all contributors agree to the Developer
//  Certificate of Origin Version 1.1 (DCO 1.1) as stated in DCO.txt.
//============================================================================
#ifndef viskores_worklet_quadric_clustering_Assignment_h
#define viskores_worklet_quadric_clustering_Assignment_h

#include <viskores/filter/geometry_refinement/worklet/quadric_clustering/QuadricClusteringCommonWorklets.h>
#include <viskores/filter/geometry_refinement/worklet/quadric_clustering/SurfaceMesh.h>

#include <viskores/cont/Algorithm.h>
#include <viskores/cont/ArrayHandleIndex.h>
#include <viskores/cont/ArrayHandlePermutation.h>

#include <viskores/worklet/WorkletMapField.h>

VISKORES_QUADRIC_CLUSTERING_STRICT_FP_BEGIN

namespace viskores
{
namespace worklet
{
namespace quadric_clustering
{

// The assignment step of Section 11 (Algorithm 3): guarded propose and commit sweeps to a fixed
// point, in a fixed commit order (by default a hash of the vertex index).

namespace detail
{

struct InitialPriority : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn index, FieldOut key);
  using ExecutionSignature = _2(_1);

  VISKORES_EXEC viskores::UInt64 operator()(viskores::Id index) const { return HashIndex(index); }
};

} // namespace detail

/// The default order of the commit step: a hash of the vertex index, ties broken by the index.
inline viskores::cont::ArrayHandle<viskores::UInt64> DefaultPriority(viskores::Id numVertices)
{
  viskores::cont::ArrayHandle<viskores::UInt64> keys;
  viskores::cont::Invoker invoke;
  invoke(detail::InitialPriority{}, viskores::cont::ArrayHandleIndex(numVertices), keys);
  return keys;
}

/// The ring of every vertex as a range of `SurfaceMesh::Rings`.
inline viskores::cont::ArrayHandle<viskores::Id2> MakeRingRanges(
  const viskores::cont::ArrayHandle<viskores::Id>& offsets,
  viskores::Id numVertices)
{
  viskores::cont::ArrayHandle<viskores::Id2> ranges;
  viskores::cont::Invoker invoke;
  invoke(OffsetRange{}, viskores::cont::ArrayHandleIndex(numVertices), offsets, ranges);
  return ranges;
}

namespace detail
{

/// The propose pass of Algorithm 3: the cheapest label among the vertex's own and those of its
/// neighbours, ties going to its own and then to the smallest, and whether the vertex may move
/// to it. A move must strictly decrease the cost and pass the guard: the neighbours in the
/// vertex's cluster form one run of the ring, and there is at least one, so the last vertex of a
/// cluster stays.
struct Propose : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn vertex,
                                FieldIn ringRange,
                                FieldIn boundary,
                                FieldIn point,
                                FieldIn m,
                                WholeArrayIn rings,
                                WholeArrayIn labels,
                                WholeArrayIn generators,
                                FieldOut propose,
                                FieldOut target);
  using ExecutionSignature = void(_1, _2, _3, _4, _5, _6, _7, _8, _9, _10);

  template <typename IdPortal, typename VecPortal>
  VISKORES_EXEC void operator()(viskores::Id v,
                                const viskores::Id2& ringRange,
                                viskores::UInt8 boundary,
                                const Vec3& point,
                                const Mat3& m,
                                const IdPortal& rings,
                                const IdPortal& labels,
                                const VecPortal& generators,
                                viskores::UInt8& propose,
                                viskores::Id& target) const
  {
    const viskores::Id own = labels.Get(v);
    propose = 0;
    target = own;
    const Real ownCost = Quad(Sub(generators.Get(own), point), m);
    Real best = ownCost;
    viskores::Id runs = 0;
    bool anyIn = false;
    const viskores::Id degree = ringRange[1] - ringRange[0];
    bool previousIn = false;
    if (!boundary && degree > 0)
    {
      previousIn = labels.Get(rings.Get(ringRange[1] - 1)) == own;
    }
    for (viskores::Id i = ringRange[0]; i < ringRange[1]; ++i)
    {
      const viskores::Id label = labels.Get(rings.Get(i));
      const bool in = (label == own);
      if (in && !previousIn)
      {
        ++runs;
      }
      anyIn = anyIn || in;
      previousIn = in;
      const Real cost = Quad(Sub(generators.Get(label), point), m);
      if (cost < best)
      {
        best = cost;
        target = label;
      }
      else if (cost == best && label < target)
      {
        target = label;
      }
    }
    if (!(best < ownCost))
    {
      return;
    }
    propose = (runs <= 1 && anyIn) ? 1 : 0;
  }
};

/// `u` beats `v` in the commit step if its key is larger, or equal with a larger index.
VISKORES_EXEC inline bool Beats(viskores::UInt64 keyU,
                                viskores::Id u,
                                viskores::UInt64 keyV,
                                viskores::Id v)
{
  return (keyU > keyV) || (keyU == keyV && u > v);
}

/// The commit pass of Algorithm 3: a proposing vertex moves unless a proposing neighbour beats
/// it.
struct Commit : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn vertex,
                                FieldIn ringRange,
                                FieldIn label,
                                FieldIn target,
                                WholeArrayIn rings,
                                WholeArrayIn propose,
                                WholeArrayIn priority,
                                FieldOut newLabel,
                                FieldOut changed);
  using ExecutionSignature = void(_1, _2, _3, _4, _5, _6, _7, _8, _9);

  template <typename IdPortal, typename FlagPortal, typename KeyPortal>
  VISKORES_EXEC void operator()(viskores::Id v,
                                const viskores::Id2& ringRange,
                                viskores::Id label,
                                viskores::Id target,
                                const IdPortal& rings,
                                const FlagPortal& propose,
                                const KeyPortal& priority,
                                viskores::Id& newLabel,
                                viskores::Id& changed) const
  {
    newLabel = label;
    changed = 0;
    if (!propose.Get(v))
    {
      return;
    }
    const viskores::UInt64 key = priority.Get(v);
    for (viskores::Id i = ringRange[0]; i < ringRange[1]; ++i)
    {
      const viskores::Id u = rings.Get(i);
      if (propose.Get(u) && Beats(priority.Get(u), u, key, v))
      {
        return;
      }
    }
    newLabel = target;
    changed = 1;
  }
};

struct CountChange : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn vertex, WholeArrayInOut changes);
  using ExecutionSignature = void(_1, _2);

  template <typename ChangesPortal>
  VISKORES_EXEC void operator()(viskores::Id v, const ChangesPortal& changes) const
  {
    changes.Set(v, changes.Get(v) + 1);
  }
};

/// The number of vertices of the closed one-ring of a vertex.
struct ClosedRingSize : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn ringRange, FieldOut size);
  using ExecutionSignature = _2(_1);

  VISKORES_EXEC viskores::Id operator()(const viskores::Id2& ringRange) const
  {
    return 1 + ringRange[1] - ringRange[0];
  }
};

/// Lists a vertex and its ring from position `offset` on.
struct ListClosedRing : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn vertex,
                                FieldIn ringRange,
                                FieldIn offset,
                                WholeArrayIn rings,
                                WholeArrayOut list);
  using ExecutionSignature = void(_1, _2, _3, _4, _5);

  template <typename IdPortal, typename OutPortal>
  VISKORES_EXEC void operator()(viskores::Id v,
                                const viskores::Id2& ringRange,
                                viskores::Id offset,
                                const IdPortal& rings,
                                const OutPortal& list) const
  {
    list.Set(offset, v);
    for (viskores::Id i = ringRange[0]; i < ringRange[1]; ++i)
    {
      list.Set(offset + 1 + i - ringRange[0], rings.Get(i));
    }
  }
};

} // namespace detail

/// The vertices of `vertices` and their rings, sorted and without repetition.
inline viskores::cont::ArrayHandle<viskores::Id> ClosedNeighborhood(
  const viskores::cont::ArrayHandle<viskores::Id>& vertices,
  const viskores::cont::ArrayHandle<viskores::Id2>& ringRanges,
  const viskores::cont::ArrayHandle<viskores::Id>& rings)
{
  using Algorithm = viskores::cont::Algorithm;
  viskores::cont::Invoker invoke;
  const auto vertexRanges = viskores::cont::make_ArrayHandlePermutation(vertices, ringRanges);
  viskores::cont::ArrayHandle<viskores::Id> sizes, offsets, list;
  invoke(detail::ClosedRingSize{}, vertexRanges, sizes);
  list.Allocate(Algorithm::ScanExclusive(sizes, offsets));
  invoke(detail::ListClosedRing{}, vertices, vertexRanges, offsets, rings, list);
  Algorithm::Sort(list);
  Algorithm::Unique(list);
  return list;
}

/// The guarded assignment of Algorithm 3, from the current labels to a fixed point or for at most
/// `maxSweeps` sweeps (0 for no cap). Returns the number of sweeps. `changes`, when given,
/// accumulates the number of label changes of every vertex.
///
/// A proposal reads only the labels of the vertex and its ring, and a commit only the
/// proposals of the vertex and its ring, so after the first sweep only the vertices next to a
/// change propose again, and only those next to them commit again; the others would compute
/// what they computed before. The labels are those of full sweeps.
template <typename SweepObserver>
viskores::Id Assign(const SurfaceMesh& mesh,
                    const viskores::cont::ArrayHandle<viskores::Id2>& ringRanges,
                    const viskores::cont::ArrayHandle<viskores::Id>& rings,
                    const viskores::cont::ArrayHandle<viskores::UInt8>& isBoundary,
                    const viskores::cont::ArrayHandle<Mat3>& m,
                    const viskores::cont::ArrayHandle<viskores::UInt64>& priority,
                    viskores::Id maxSweeps,
                    viskores::cont::ArrayHandle<viskores::Id>& labels,
                    const viskores::cont::ArrayHandle<Vec3>& generators,
                    viskores::cont::ArrayHandle<viskores::Id>* changes,
                    SweepObserver&& observer)
{
  using Algorithm = viskores::cont::Algorithm;
  using viskores::cont::make_ArrayHandlePermutation;
  viskores::cont::Invoker invoke;
  detail::Propose propose;
  viskores::cont::ArrayHandle<viskores::UInt8> proposes;
  viskores::cont::ArrayHandle<viskores::Id> targets;
  const viskores::Id n = mesh.NumberOfPoints;
  invoke(propose,
         viskores::cont::ArrayHandleIndex(n),
         ringRanges,
         isBoundary,
         mesh.Points,
         m,
         rings,
         labels,
         generators,
         proposes,
         targets);
  // The vertices to commit: all of them at first.
  viskores::cont::ArrayHandle<viskores::Id> active;
  Algorithm::Copy(viskores::cont::ArrayHandleIndex(n), active);
  viskores::cont::ArrayHandle<viskores::Id> newLabels, changed, moved, newMoved, newTargets;
  viskores::cont::ArrayHandle<viskores::UInt8> newProposes;
  viskores::Id sweeps = 0;
  while (maxSweeps <= 0 || sweeps < maxSweeps)
  {
    if (Algorithm::Reduce(proposes, viskores::Id(0)) == 0)
    {
      break;
    }
    invoke(detail::Commit{},
           active,
           make_ArrayHandlePermutation(active, ringRanges),
           make_ArrayHandlePermutation(active, labels),
           make_ArrayHandlePermutation(active, targets),
           rings,
           proposes,
           priority,
           newLabels,
           changed);
    // Every commit reads its own label only, so the moves can be written in place.
    Algorithm::CopyIf(active, changed, moved);
    Algorithm::CopyIf(newLabels, changed, newMoved);
    invoke(ScatterValue{}, moved, newMoved, labels);
    if (changes)
    {
      invoke(detail::CountChange{}, moved, *changes);
    }
    ++sweeps;
    observer(labels);
    const viskores::cont::ArrayHandle<viskores::Id> proposing =
      ClosedNeighborhood(moved, ringRanges, rings);
    invoke(propose,
           proposing,
           make_ArrayHandlePermutation(proposing, ringRanges),
           make_ArrayHandlePermutation(proposing, isBoundary),
           make_ArrayHandlePermutation(proposing, mesh.Points),
           make_ArrayHandlePermutation(proposing, m),
           rings,
           labels,
           generators,
           newProposes,
           newTargets);
    invoke(ScatterValue{}, proposing, newProposes, proposes);
    invoke(ScatterValue{}, proposing, newTargets, targets);
    active = ClosedNeighborhood(proposing, ringRanges, rings);
  }
  return sweeps;
}
}
}
} // namespace viskores::worklet::quadric_clustering

VISKORES_QUADRIC_CLUSTERING_STRICT_FP_END

#endif // viskores_worklet_quadric_clustering_Assignment_h
