//============================================================================
//  The contents of this file are covered by the Viskores license. See
//  LICENSE.txt for details.
//
//  By contributing to this file, all contributors agree to the Developer
//  Certificate of Origin Version 1.1 (DCO 1.1) as stated in DCO.txt.
//============================================================================
#ifndef viskores_worklet_quadric_clustering_QuadricClusteringCommonWorklets_h
#define viskores_worklet_quadric_clustering_QuadricClusteringCommonWorklets_h

#include <viskores/filter/geometry_refinement/worklet/quadric_clustering/Arithmetic.h>

#include <viskores/BinaryOperators.h>
#include <viskores/Swap.h>
#include <viskores/Types.h>
#include <viskores/cont/Algorithm.h>
#include <viskores/cont/ArrayHandle.h>
#include <viskores/cont/Invoker.h>
#include <viskores/worklet/ScatterCounting.h>
#include <viskores/worklet/WorkletMapField.h>


VISKORES_QUADRIC_CLUSTERING_STRICT_FP_BEGIN

namespace viskores
{
namespace worklet
{
namespace quadric_clustering
{

/// The edge `{a, b}` as `(min, max)`.
VISKORES_EXEC_CONT inline viskores::Id2 SortedEdge(viskores::Id a, viskores::Id b)
{
  return viskores::Id2(viskores::Min(a, b), viskores::Max(a, b));
}

namespace detail
{

struct OrFlag : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldInOut flags, FieldIn other, FieldOut added);
  using ExecutionSignature = void(_1, _2, _3);

  VISKORES_EXEC void operator()(viskores::UInt8& flags,
                                viskores::UInt8 other,
                                viskores::Id& added) const
  {
    added = (other && !flags) ? 1 : 0;
    flags = (flags || other) ? 1 : 0;
  }
};

} // namespace detail

/// Adds the flags `other` to `flags`, and returns how many clusters were not flagged before.
inline viskores::Id OrFlags(viskores::cont::ArrayHandle<viskores::UInt8>& flags,
                            const viskores::cont::ArrayHandle<viskores::UInt8>& other)
{
  viskores::cont::Invoker invoke;
  viskores::cont::ArrayHandle<viskores::Id> added;
  invoke(detail::OrFlag{}, flags, other, added);
  return viskores::cont::Algorithm::Reduce(added, viskores::Id(0));
}

/// The number of flagged clusters.
inline viskores::Id CountFlags(const viskores::cont::ArrayHandle<viskores::UInt8>& flags)
{
  return viskores::cont::Algorithm::Reduce(flags, viskores::Id(0));
}

/// 1 where two ids are equal: an index that is its own root or representative.
struct IsEqual : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn a, FieldIn b, FieldOut equal);
  using ExecutionSignature = _3(_1, _2);

  VISKORES_EXEC viskores::Id operator()(viskores::Id a, viskores::Id b) const
  {
    return (a == b) ? 1 : 0;
  }
};

/// `table[index]`.
struct GatherValue : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn index, WholeArrayIn table, FieldOut value);
  using ExecutionSignature = _3(_1, _2);

  template <typename Portal>
  VISKORES_EXEC typename Portal::ValueType operator()(viskores::Id index, const Portal& table) const
  {
    return table.Get(index);
  }
};

/// The entries of `table` at the three corners of a triangle.
struct GatherTriple : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn triangle, WholeArrayIn table, FieldOut gathered);
  using ExecutionSignature = _3(_1, _2);

  template <typename IdPortal>
  VISKORES_EXEC viskores::Id3 operator()(const viskores::Id3& t, const IdPortal& table) const
  {
    return viskores::Id3(table.Get(t[0]), table.Get(t[1]), table.Get(t[2]));
  }
};

/// 1 for a label, 0 for a slot marked invalid by a negative value.
struct IsValidLabel : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn label, FieldOut valid);
  using ExecutionSignature = _2(_1);

  VISKORES_EXEC viskores::UInt8 operator()(viskores::Id label) const { return label >= 0 ? 1 : 0; }
};

/// Component-wise minimum and maximum of two points, for `Reduce`.
struct ComponentMin
{
  VISKORES_EXEC_CONT Vec3 operator()(const Vec3& a, const Vec3& b) const
  {
    return Vec3(viskores::Min(a[0], b[0]), viskores::Min(a[1], b[1]), viskores::Min(a[2], b[2]));
  }
};

struct ComponentMax
{
  VISKORES_EXEC_CONT Vec3 operator()(const Vec3& a, const Vec3& b) const
  {
    return Vec3(viskores::Max(a[0], b[0]), viskores::Max(a[1], b[1]), viskores::Max(a[2], b[2]));
  }
};

/// The two components of a pair.
struct SplitPair : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn pair, FieldOut first, FieldOut second);
  using ExecutionSignature = void(_1, _2, _3);

  VISKORES_EXEC void operator()(const viskores::Id2& pair,
                                viskores::Id& first,
                                viskores::Id& second) const
  {
    first = pair[0];
    second = pair[1];
  }
};

/// The three edges of a triangle, each as `SortedEdge`.
struct SortedTriangleEdges : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn triangle, FieldOut edges);
  using ExecutionSignature = void(_1, _2);

  template <typename EdgeVec>
  VISKORES_EXEC void operator()(const viskores::Id3& t, EdgeVec& edges) const
  {
    for (viskores::IdComponent s = 0; s < 3; ++s)
    {
      edges[s] = SortedEdge(t[s], t[(s + 1) % 3]);
    }
  }
};

/// 1 for a record whose first component is not negative, 0 for one marked invalid.
struct IsValidRecord : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn record, FieldOut valid);
  using ExecutionSignature = _2(_1);

  template <typename Record>
  VISKORES_EXEC viskores::UInt8 operator()(const Record& r) const
  {
    return r[0] >= 0 ? 1 : 0;
  }
};

/// The first index in `[lo, hi)` at which `before(index)` is false, for a `before` that is true
/// on a prefix of the range: a binary search.
template <typename Before>
VISKORES_EXEC_CONT viskores::Id PartitionPoint(viskores::Id lo, viskores::Id hi, Before&& before)
{
  while (lo < hi)
  {
    const viskores::Id mid = (lo + hi) / 2;
    if (before(mid))
    {
      lo = mid + 1;
    }
    else
    {
      hi = mid;
    }
  }
  return lo;
}

/// The first index in `[lo, hi)` of the sorted `sorted` whose value is not below `value`.
template <typename Portal, typename Value>
VISKORES_EXEC_CONT viskores::Id LowerBound(const Portal& sorted,
                                           viskores::Id lo,
                                           viskores::Id hi,
                                           const Value& value)
{
  return PartitionPoint(lo, hi, [&](viskores::Id i) { return sorted.Get(i) < value; });
}

/// The position of `value` in the sorted `sorted`, or -1.
template <typename Portal, typename Value>
VISKORES_EXEC_CONT viskores::Id FindSorted(const Portal& sorted, const Value& value)
{
  const viskores::Id n = sorted.GetNumberOfValues();
  const viskores::Id lo = LowerBound(sorted, 0, n, value);
  return (lo < n && sorted.Get(lo) == value) ? lo : -1;
}

/// Both directions of an edge.
struct BothDirections : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn edge, FieldOut directed);
  using ExecutionSignature = void(_1, _2);

  template <typename OutVec>
  VISKORES_EXEC void operator()(const viskores::Id2& e, OutVec& out) const
  {
    out[0] = e;
    out[1] = viskores::Id2(e[1], e[0]);
  }
};

/// The range `[offsets[i], offsets[i + 1])` of item `i`.
struct OffsetRange : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn index, WholeArrayIn offsets, FieldOut range);
  using ExecutionSignature = _3(_1, _2);

  template <typename OffsetsPortal>
  VISKORES_EXEC viskores::Id2 operator()(viskores::Id index, const OffsetsPortal& offsets) const
  {
    return viskores::Id2(offsets.Get(index), offsets.Get(index + 1));
  }
};

/// Three indices in increasing order.
VISKORES_EXEC_CONT inline viskores::Id3 SortThree(viskores::Id a, viskores::Id b, viskores::Id c)
{
  if (a > b)
  {
    viskores::Swap(a, b);
  }
  if (b > c)
  {
    viskores::Swap(b, c);
  }
  if (a > b)
  {
    viskores::Swap(a, b);
  }
  return viskores::Id3(a, b, c);
}

VISKORES_EXEC_CONT inline viskores::Id3 SortThree(const viskores::Id3& t)
{
  return SortThree(t[0], t[1], t[2]);
}

/// Writes `value` at `index`.
struct ScatterValue : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn index, FieldIn value, WholeArrayOut values);
  using ExecutionSignature = void(_1, _2, _3);

  template <typename T, typename OutPortal>
  VISKORES_EXEC void operator()(viskores::Id index, const T& value, OutPortal& values) const
  {
    values.Set(index, value);
  }
};

/// Writes `Value` (1 by default) at `index`; every write is the same.
struct MarkIndex : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn index, WholeArrayInOut marks);
  using ExecutionSignature = void(_1, _2);

  VISKORES_CONT explicit MarkIndex(viskores::UInt8 value = 1)
    : Value(value)
  {
  }

  template <typename MarkPortal>
  VISKORES_EXEC void operator()(viskores::Id index, MarkPortal& marks) const
  {
    marks.Set(index, this->Value);
  }

  viskores::UInt8 Value;
};

/// 1 where a state equals `State`.
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

/// The scatter of a heavy worklet over `n` items: every item is visited `k` times, of which
/// only the first works, with `k` (1 to 1024) such that the launch has about 64 Ki visits. The
/// TBB device schedules at least 1024 visits per task, so a launch of a few hundred items would
/// otherwise run on one thread.
inline viskores::worklet::ScatterCounting SpreadScatterOver(viskores::Id n)
{
  constexpr viskores::Id visits = viskores::Id(1) << 16;
  const viskores::Id k = viskores::Max(
    viskores::Id(1), viskores::Min(viskores::Id(1024), visits / viskores::Max(n, viskores::Id(1))));
  viskores::cont::ArrayHandle<viskores::Int32> counts;
  counts.AllocateAndFill(n, static_cast<viskores::Int32>(k));
  return viskores::worklet::ScatterCounting(counts);
}

}
}
} // namespace viskores::worklet::quadric_clustering

VISKORES_QUADRIC_CLUSTERING_STRICT_FP_END

#endif // viskores_worklet_quadric_clustering_QuadricClusteringCommonWorklets_h
