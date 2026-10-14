//============================================================================
//  The contents of this file are covered by the Viskores license. See
//  LICENSE.txt for details.
//
//  By contributing to this file, all contributors agree to the Developer
//  Certificate of Origin Version 1.1 (DCO 1.1) as stated in DCO.txt.
//============================================================================
#ifndef viskores_worklet_quadric_clustering_Initialization_h
#define viskores_worklet_quadric_clustering_Initialization_h

#include <viskores/filter/geometry_refinement/worklet/quadric_clustering/SurfaceMesh.h>

#include <viskores/Pair.h>
#include <viskores/cont/Algorithm.h>
#include <viskores/cont/ArrayHandleConstant.h>
#include <viskores/cont/ErrorBadValue.h>

#include <viskores/worklet/WorkletMapField.h>

#include <algorithm>
#include <cmath>
#include <vector>

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
/// reference implementation.
struct MortonCode : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn point, FieldOut code);
  using ExecutionSignature = _2(_1, WorkIndex);

  VISKORES_CONT MortonCode(const Vec3& minimum, Real extent)
    : Minimum(minimum)
    , Extent(extent)
  {
  }

  Vec3 Minimum;
  Real Extent;

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
};

struct IsUnlabelled : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn label, FieldOut unlabelled);
  using ExecutionSignature = _2(_1);

  VISKORES_EXEC viskores::Id operator()(viskores::Id label) const { return (label < 0) ? 1 : 0; }
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

} // namespace detail

/// The vertices in Morton order, ties broken by index, and the largest side of the bounding box.
inline viskores::cont::ArrayHandle<viskores::Pair<viskores::UInt64, viskores::Id>> MortonOrder(
  const SurfaceMesh& mesh,
  Real& size)
{
  const viskores::Id n = mesh.NumberOfPoints;
  // Bounds, exactly: min and max are exact in any order.
  Vec3 lo, hi;
  {
    auto portal = mesh.Points.ReadPortal();
    lo = hi = portal.Get(0);
    for (viskores::Id v = 1; v < n; ++v)
    {
      const Vec3 p = portal.Get(v);
      for (viskores::IdComponent a = 0; a < 3; ++a)
      {
        lo[a] = viskores::Min(lo[a], p[a]);
        hi[a] = viskores::Max(hi[a], p[a]);
      }
    }
  }
  size = viskores::Max(viskores::Max(hi[0] - lo[0], hi[1] - lo[1]), hi[2] - lo[2]);
  const Real extent = viskores::Max(size, Real(1e-300));
  viskores::cont::ArrayHandle<viskores::Pair<viskores::UInt64, viskores::Id>> codes;
  viskores::cont::Invoker invoke;
  invoke(detail::MortonCode(lo, extent), mesh.Points, codes);
  viskores::cont::Algorithm::Sort(codes);
  return codes;
}

/// Grows seeds into connected clusters: adds the smallest vertex of every component without a
/// seed, numbers the seeds in increasing vertex order, and runs Jacobi sweeps in which an
/// unlabelled vertex adopts the smallest label among its labelled neighbours.
inline viskores::cont::ArrayHandle<viskores::Id> GrowSeedLabels(
  const SurfaceMesh& mesh,
  const viskores::cont::ArrayHandle<viskores::Id2>& ringRanges,
  const viskores::cont::ArrayHandle<viskores::Id>& rings,
  std::vector<viskores::Id> seeds)
{
  using Algorithm = viskores::cont::Algorithm;
  viskores::cont::Invoker invoke;
  const viskores::Id n = mesh.NumberOfPoints;
  // One seed for every component that has none.
  viskores::cont::ArrayHandle<viskores::Id> components, roots;
  const viskores::Id numComponents = LabelComponents(
    n, mesh.Edges, viskores::cont::ArrayHandleConstant<viskores::Id>(0, n), components, &roots);
  {
    std::vector<bool> seeded(static_cast<std::size_t>(numComponents), false);
    auto componentPortal = components.ReadPortal();
    for (viskores::Id s : seeds)
    {
      seeded[static_cast<std::size_t>(componentPortal.Get(s))] = true;
    }
    auto rootPortal = roots.ReadPortal();
    for (viskores::Id v = 0; v < n; ++v)
    {
      const viskores::Id c = componentPortal.Get(v);
      if (rootPortal.Get(v) == v && !seeded[static_cast<std::size_t>(c)])
      {
        seeds.push_back(v);
        seeded[static_cast<std::size_t>(c)] = true;
      }
    }
  }
  std::sort(seeds.begin(), seeds.end());
  seeds.erase(std::unique(seeds.begin(), seeds.end()), seeds.end());

  viskores::cont::ArrayHandle<viskores::Id> labels;
  labels.AllocateAndFill(n, -1);
  {
    auto portal = labels.WritePortal();
    for (std::size_t i = 0; i < seeds.size(); ++i)
    {
      portal.Set(seeds[i], static_cast<viskores::Id>(i));
    }
  }
  viskores::cont::ArrayHandle<viskores::Id> next, unlabelledFlags;
  viskores::Id unlabelled = n - static_cast<viskores::Id>(seeds.size());
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
  const viskores::Id n = mesh.NumberOfPoints;
  Real size;
  const auto codes = MortonOrder(mesh, size);

  // Positions round(i (n - 1) / (k - 1)), half to even, with the last one exactly n - 1.
  std::vector<viskores::Id> seeds;
  {
    auto order = codes.ReadPortal();
    const Real step = (k > 1) ? Real(n - 1) / Real(k - 1) : Real(0);
    viskores::Id previous = -1;
    for (viskores::Id i = 0; i < k; ++i)
    {
      const Real y = (k > 1 && i == k - 1) ? Real(n - 1) : Real(i) * step;
      const viskores::Id position = static_cast<viskores::Id>(std::nearbyint(y));
      if (position != previous)
      {
        seeds.push_back(order.Get(position).second);
        previous = position;
      }
    }
  }
  return GrowSeedLabels(mesh, ringRanges, rings, seeds);
}
}
}
} // namespace viskores::worklet::quadric_clustering

VISKORES_QUADRIC_CLUSTERING_STRICT_FP_END

#endif // viskores_worklet_quadric_clustering_Initialization_h
