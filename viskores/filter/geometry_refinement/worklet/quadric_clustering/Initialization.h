//============================================================================
//  The contents of this file are covered by the Viskores license. See
//  LICENSE.txt for details.
//
//  By contributing to this file, all contributors agree to the Developer
//  Certificate of Origin Version 1.1 (DCO 1.1) as stated in DCO.txt.
//============================================================================
#ifndef viskores_worklet_quadric_clustering_Initialization_h
#define viskores_worklet_quadric_clustering_Initialization_h

#include <viskores/filter/geometry_refinement/worklet/quadric_clustering/QuadricClusteringCommonWorklets.h>
#include <viskores/filter/geometry_refinement/worklet/quadric_clustering/SurfaceMesh.h>

#include <viskores/Pair.h>
#include <viskores/cont/Algorithm.h>
#include <viskores/cont/ArrayGetValues.h>
#include <viskores/cont/ArrayHandleConcatenate.h>
#include <viskores/cont/ArrayHandleConstant.h>
#include <viskores/cont/ArrayHandleIndex.h>
#include <viskores/cont/ArrayHandlePermutation.h>
#include <viskores/cont/ErrorBadValue.h>

#include <viskores/worklet/WorkletMapField.h>

#include <algorithm>

VISKORES_QUADRIC_CLUSTERING_STRICT_FP_BEGIN

namespace viskores
{
namespace worklet
{
namespace quadric_clustering
{

// The initialisation of Section 11: seeds at evenly spaced positions of the Morton order of the
// vertices, grown into connected clusters.

namespace detail
{

/// The Morton code of a vertex, from its coordinates quantised to 20 bits per axis as in the
/// reference implementation (bit 20 is always 0).
struct MortonCode : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn point, FieldOut code);
  using ExecutionSignature = _2(_1, WorkIndex);

  VISKORES_CONT MortonCode(const Vec3& minimum, Real extent)
    : Minimum(minimum)
    , Extent(extent)
  {
  }

  VISKORES_EXEC viskores::Pair<viskores::UInt64, viskores::Id> operator()(const Vec3& point,
                                                                          viskores::Id index) const
  {
    viskores::UInt64 code = 0;
    for (viskores::IdComponent axis = 0; axis < 3; ++axis)
    {
      const Real scaled = (point[axis] - this->Minimum[axis]) / this->Extent * Real(1048575);
      const viskores::UInt64 q = static_cast<viskores::UInt64>(viskores::Floor(scaled));
      for (viskores::IdComponent bit = 0; bit < 21; ++bit)
      {
        code |= ((q >> bit) & 1ULL) << (3 * bit + axis);
      }
    }
    return viskores::Pair<viskores::UInt64, viskores::Id>(code, index);
  }

  Vec3 Minimum;
  Real Extent;
};

} // namespace detail

/// The vertices in Morton order, ties broken by index, and the largest side of the bounding box.
inline viskores::cont::ArrayHandle<viskores::Pair<viskores::UInt64, viskores::Id>> MortonOrder(
  const SurfaceMesh& mesh,
  Real& size)
{
  // Bounds, exactly: min and max are exact in any order.
  const Vec3 first = viskores::cont::ArrayGetValue(0, mesh.Points);
  const Vec3 lo = viskores::cont::Algorithm::Reduce(mesh.Points, first, ComponentMin{});
  const Vec3 hi = viskores::cont::Algorithm::Reduce(mesh.Points, first, ComponentMax{});
  size = viskores::Max(viskores::Max(hi[0] - lo[0], hi[1] - lo[1]), hi[2] - lo[2]);
  const Real extent = viskores::Max(size, Real(1e-300));
  viskores::cont::ArrayHandle<viskores::Pair<viskores::UInt64, viskores::Id>> codes;
  viskores::cont::Invoker invoke;
  invoke(detail::MortonCode(lo, extent), mesh.Points, codes);
  viskores::cont::Algorithm::Sort(codes);
  return codes;
}

namespace detail
{

/// `x` rounded to the nearest integer, half to even, as `std::nearbyint` in the default
/// rounding mode: `x - floor(x)` is exact.
VISKORES_EXEC_CONT inline viskores::Id RoundHalfEven(Real x)
{
  const Real f = viskores::Floor(x);
  const Real d = x - f;
  const viskores::Id i = static_cast<viskores::Id>(f);
  return (d > Real(0.5) || (d == Real(0.5) && (i % 2) != 0)) ? i + 1 : i;
}

/// Position `i` of `k` evenly spaced positions of `n`: `round(i (n - 1) / (k - 1))`, half to
/// even, with the last one exactly `n - 1`.
struct EvenPosition : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn i, FieldOut position);
  using ExecutionSignature = _2(_1);

  VISKORES_CONT EvenPosition(viskores::Id n, viskores::Id k)
    : N(n)
    , K(k)
    , Step((k > 1) ? Real(n - 1) / Real(k - 1) : Real(0))
  {
  }

  VISKORES_EXEC viskores::Id operator()(viskores::Id i) const
  {
    const Real y = (this->K > 1 && i == this->K - 1) ? Real(this->N - 1) : Real(i) * this->Step;
    return RoundHalfEven(y);
  }

  viskores::Id N;
  viskores::Id K;
  Real Step;
};

/// The vertex at a position of the Morton order.
struct OrderedVertex : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn position, WholeArrayIn order, FieldOut vertex);
  using ExecutionSignature = _3(_1, _2);

  template <typename OrderPortal>
  VISKORES_EXEC viskores::Id operator()(viskores::Id position, const OrderPortal& order) const
  {
    return order.Get(position).second;
  }
};

/// Whether a vertex is the smallest of a component without a seed.
struct IsUnseededRoot : viskores::worklet::WorkletMapField
{
  using ControlSignature =
    void(FieldIn vertex, FieldIn root, FieldIn component, WholeArrayIn seeded, FieldOut extra);
  using ExecutionSignature = _5(_1, _2, _3, _4);

  template <typename FlagPortal>
  VISKORES_EXEC viskores::UInt8 operator()(viskores::Id v,
                                           viskores::Id root,
                                           viskores::Id component,
                                           const FlagPortal& seeded) const
  {
    return (root == v && !seeded.Get(component)) ? 1 : 0;
  }
};

/// One Jacobi sweep of the seed growth: an unlabelled vertex adopts the smallest label among its
/// labelled neighbours.
struct GrowSeeds : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn ringRange,
                                FieldIn label,
                                WholeArrayIn rings,
                                WholeArrayIn labels,
                                FieldOut newLabel);
  using ExecutionSignature = _5(_1, _2, _3, _4);

  template <typename IdPortal>
  VISKORES_EXEC viskores::Id operator()(const viskores::Id2& ringRange,
                                        viskores::Id label,
                                        const IdPortal& rings,
                                        const IdPortal& labels) const
  {
    if (label >= 0)
    {
      return label;
    }
    viskores::Id best = -1;
    for (viskores::Id i = ringRange[0]; i < ringRange[1]; ++i)
    {
      const viskores::Id l = labels.Get(rings.Get(i));
      if (l >= 0 && (best < 0 || l < best))
      {
        best = l;
      }
    }
    return best;
  }
};

struct IsUnlabelled : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn label, FieldOut unlabelled);
  using ExecutionSignature = _2(_1);

  VISKORES_EXEC viskores::Id operator()(viskores::Id label) const { return (label < 0) ? 1 : 0; }
};

} // namespace detail

/// Grows seeds into connected clusters: adds the smallest vertex of every component without a
/// seed, numbers the seeds in increasing vertex order, and runs Jacobi sweeps in which an
/// unlabelled vertex adopts the smallest label among its labelled neighbours. `seeds` may
/// repeat.
inline viskores::cont::ArrayHandle<viskores::Id> GrowSeedLabels(
  const SurfaceMesh& mesh,
  const viskores::cont::ArrayHandle<viskores::Id2>& ringRanges,
  const viskores::cont::ArrayHandle<viskores::Id>& rings,
  const viskores::cont::ArrayHandle<viskores::Id>& seeds)
{
  using Algorithm = viskores::cont::Algorithm;
  viskores::cont::Invoker invoke;
  const viskores::Id n = mesh.NumberOfPoints;
  // One seed for every component that has none: its smallest vertex.
  viskores::cont::ArrayHandle<viskores::Id> components, roots;
  const viskores::Id numComponents = LabelComponents(
    n, mesh.Edges, viskores::cont::ArrayHandleConstant<viskores::Id>(0, n), components, &roots);
  viskores::cont::ArrayHandle<viskores::UInt8> seeded, isExtra;
  seeded.AllocateAndFill(numComponents, 0);
  // Marks the component of every seed.
  invoke(MarkIndex{}, viskores::cont::make_ArrayHandlePermutation(seeds, components), seeded);
  invoke(detail::IsUnseededRoot{},
         viskores::cont::ArrayHandleIndex(n),
         roots,
         components,
         seeded,
         isExtra);
  viskores::cont::ArrayHandle<viskores::Id> extra, allSeeds;
  Algorithm::CopyIf(viskores::cont::ArrayHandleIndex(n), isExtra, extra);
  Algorithm::Copy(viskores::cont::make_ArrayHandleConcatenate(seeds, extra), allSeeds);
  Algorithm::Sort(allSeeds);
  Algorithm::Unique(allSeeds);

  viskores::cont::ArrayHandle<viskores::Id> labels;
  labels.AllocateAndFill(n, -1);
  // Seed `i`, in increasing vertex order, gets label `i`.
  invoke(ScatterValue{},
         allSeeds,
         viskores::cont::ArrayHandleIndex(allSeeds.GetNumberOfValues()),
         labels);
  viskores::cont::ArrayHandle<viskores::Id> next, unlabelledFlags;
  viskores::Id unlabelled = n - allSeeds.GetNumberOfValues();
  while (unlabelled > 0)
  {
    invoke(detail::GrowSeeds{}, ringRanges, labels, rings, labels, next);
    std::swap(labels, next);
    const viskores::Id before = unlabelled;
    invoke(detail::IsUnlabelled{}, labels, unlabelledFlags);
    unlabelled = Algorithm::Reduce(unlabelledFlags, viskores::Id(0));
    if (unlabelled == before)
    {
      throw viskores::cont::ErrorBadValue("QuadricClusteringDecimation: seed growth stalled.");
    }
  }
  return labels;
}

/// Seeds of the Morton initialisation: `k` vertices at evenly spaced positions of the Morton
/// order, grown into connected clusters by `GrowSeedLabels`.
inline viskores::cont::ArrayHandle<viskores::Id> SeedLabels(
  const SurfaceMesh& mesh,
  const viskores::cont::ArrayHandle<viskores::Id2>& ringRanges,
  const viskores::cont::ArrayHandle<viskores::Id>& rings,
  viskores::Id k)
{
  using Algorithm = viskores::cont::Algorithm;
  const viskores::Id n = mesh.NumberOfPoints;
  Real size;
  const auto codes = MortonOrder(mesh, size);
  // Positions round(i (n - 1) / (k - 1)), half to even, with the last one exactly n - 1; they
  // do not decrease, so the distinct ones are those that differ from the previous.
  viskores::cont::ArrayHandle<viskores::Id> positions, seeds;
  viskores::cont::Invoker invoke;
  invoke(detail::EvenPosition(n, k), viskores::cont::ArrayHandleIndex(k), positions);
  Algorithm::Unique(positions);
  invoke(detail::OrderedVertex{}, positions, codes, seeds);
  return GrowSeedLabels(mesh, ringRanges, rings, seeds);
}

}
}
} // namespace viskores::worklet::quadric_clustering

VISKORES_QUADRIC_CLUSTERING_STRICT_FP_END

#endif // viskores_worklet_quadric_clustering_Initialization_h
