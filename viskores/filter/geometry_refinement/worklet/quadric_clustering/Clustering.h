//============================================================================
//  The contents of this file are covered by the Viskores license. See
//  LICENSE.txt for details.
//
//  By contributing to this file, all contributors agree to the Developer
//  Certificate of Origin Version 1.1 (DCO 1.1) as stated in DCO.txt.
//============================================================================
#ifndef viskores_worklet_quadric_clustering_Clustering_h
#define viskores_worklet_quadric_clustering_Clustering_h

#include <viskores/filter/geometry_refinement/worklet/quadric_clustering/SurfaceMesh.h>

#include <viskores/Pair.h>
#include <viskores/cont/Algorithm.h>
#include <viskores/cont/ArrayHandleCast.h>
#include <viskores/cont/ArrayHandleIndex.h>
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

// The steps of the main loop of Section 11 (Algorithm 4): (b) placement, (c) scan with the
// radius and budget flags, (d) insertion and (e) split into connected components.

/// A labelling of the vertices and the generators of its clusters, with the data of the last
/// placement step.
struct Clustering
{
  viskores::cont::ArrayHandle<viskores::Id> Labels;
  viskores::cont::ArrayHandle<Vec3> Generators;
  viskores::Id NumberOfClusters = 0;

  /// Groups the vertices by label, each group in increasing vertex order.
  viskores::worklet::Keys<viskores::Id> Keys;
  /// The member of smallest index of every cluster, the anchor of its sums.
  viskores::cont::ArrayHandle<viskores::Id> Anchors;
  /// `Q_C = sum M_v`, `b_C = sum M_v (v - o_C)` and `s_C = sum (v - o_C)^T M_v (v - o_C)`, so
  /// that `Phi_C(c_C) = s_C - b_C^T Q_C^{-1} b_C`.
  viskores::cont::ArrayHandle<Mat3> Q;
  viskores::cont::ArrayHandle<Vec3> B;
  viskores::cont::ArrayHandle<Real> S;
  viskores::cont::ArrayHandle<viskores::Id> Sizes;
};

/// The result of the scan of step (c).
struct ClusterScan
{
  /// `(c_{L(v)} - v)^T M_v (c_{L(v)} - v)` for every vertex.
  viskores::cont::ArrayHandle<Real> Costs;
  /// `r_C = max_{v in C} |c_C - v|`.
  viskores::cont::ArrayHandle<Real> Radii;
  /// The member of largest cost, the smallest index among equal costs.
  viskores::cont::ArrayHandle<viskores::Id> Worst;
};

namespace detail
{

/// The compact label of every vertex: the index of its key among the sorted distinct labels.
struct CompactLabels : viskores::worklet::WorkletReduceByKey
{
  using ControlSignature = void(KeysIn keys, ValuesOut labels);
  using ExecutionSignature = void(WorkIndex, _2);
  using InputDomain = _1;

  template <typename LabelsVec>
  VISKORES_EXEC void operator()(viskores::Id index, LabelsVec& labels) const
  {
    for (viskores::IdComponent i = 0; i < labels.GetNumberOfComponents(); ++i)
    {
      labels[i] = index;
    }
  }
};

/// Step (b): `c_C = o + Q_C^{-1} b_C` with `Q_C = sum M_v` and `b_C = sum M_v (v - o)`, summed in
/// increasing vertex order from the anchor `o`, the member of smallest index.
struct Place : viskores::worklet::WorkletReduceByKey
{
  using ControlSignature = void(KeysIn keys,
                                ValuesIn vertices,
                                WholeArrayIn points,
                                WholeArrayIn m,
                                ReducedValuesOut anchor,
                                ReducedValuesOut q,
                                ReducedValuesOut b,
                                ReducedValuesOut s,
                                ReducedValuesOut generator,
                                ReducedValuesOut size);
  using ExecutionSignature = void(_2, _3, _4, _5, _6, _7, _8, _9, _10);
  using InputDomain = _1;

  template <typename VerticesVec, typename PointsPortal, typename MatPortal>
  VISKORES_EXEC void operator()(const VerticesVec& vertices,
                                const PointsPortal& points,
                                const MatPortal& m,
                                viskores::Id& anchor,
                                Mat3& q,
                                Vec3& b,
                                Real& s,
                                Vec3& generator,
                                viskores::Id& size) const
  {
    size = vertices.GetNumberOfComponents();
    anchor = vertices[0];
    const Vec3 o = points.Get(anchor);
    q = Mat3(0);
    b = Vec3(0);
    s = 0;
    for (viskores::IdComponent i = 0; i < size; ++i)
    {
      const viskores::Id v = vertices[i];
      const Mat3 mv = m.Get(v);
      const Vec3 w = Sub(points.Get(v), o);
      q = MatAdd(q, mv);
      b = Add(b, MatVec(mv, w));
      s = s + Quad(w, mv);
    }
    generator = Add(o, SolveSPD(q, b));
  }
};

/// Step (c), per vertex: the cost at the generator of its cluster and the distance to it.
struct VertexScan : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn point,
                                FieldIn m,
                                FieldIn label,
                                WholeArrayIn generators,
                                FieldOut cost,
                                FieldOut distance);
  using ExecutionSignature = void(_1, _2, _3, _4, _5, _6);

  template <typename VecPortal>
  VISKORES_EXEC void operator()(const Vec3& point,
                                const Mat3& m,
                                viskores::Id label,
                                const VecPortal& generators,
                                Real& cost,
                                Real& distance) const
  {
    const Vec3 w = Sub(generators.Get(label), point);
    cost = Quad(w, m);
    distance = Norm(w);
  }
};

/// Step (c), per cluster: the radius and the worst member, the first of largest cost in
/// increasing vertex order.
struct ClusterReduce : viskores::worklet::WorkletReduceByKey
{
  using ControlSignature = void(KeysIn keys,
                                ValuesIn vertices,
                                ValuesIn costs,
                                ValuesIn distances,
                                ReducedValuesOut radius,
                                ReducedValuesOut worst);
  using ExecutionSignature = void(_2, _3, _4, _5, _6);
  using InputDomain = _1;

  template <typename IdVec, typename RealVec>
  VISKORES_EXEC void operator()(const IdVec& vertices,
                                const RealVec& costs,
                                const RealVec& distances,
                                Real& radius,
                                viskores::Id& worst) const
  {
    radius = distances[0];
    worst = vertices[0];
    Real worstCost = costs[0];
    for (viskores::IdComponent i = 1; i < vertices.GetNumberOfComponents(); ++i)
    {
      radius = viskores::Max(radius, distances[i]);
      if (costs[i] > worstCost)
      {
        worstCost = costs[i];
        worst = vertices[i];
      }
    }
  }
};

/// Step (d): the worst member of every flagged cluster gets the fresh label `k + rank`, and its
/// own position as generator.
struct Insert : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn flag,
                                FieldIn rank,
                                FieldIn worst,
                                WholeArrayInOut labels,
                                WholeArrayIn points,
                                WholeArrayInOut generators);
  using ExecutionSignature = void(_1, _2, _3, _4, _5, _6);

  VISKORES_CONT explicit Insert(viskores::Id numClusters)
    : NumberOfClusters(numClusters)
  {
  }

  viskores::Id NumberOfClusters;

  template <typename LabelsPortal, typename PointsPortal, typename GeneratorsPortal>
  VISKORES_EXEC void operator()(viskores::UInt8 flag,
                                viskores::Id rank,
                                viskores::Id worst,
                                LabelsPortal& labels,
                                const PointsPortal& points,
                                GeneratorsPortal& generators) const
  {
    if (flag)
    {
      labels.Set(worst, this->NumberOfClusters + rank);
      generators.Set(this->NumberOfClusters + rank, points.Get(worst));
    }
  }
};

/// Step (e): every component keeps the generator of the cluster its vertices came from.
struct ComponentGenerator : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn root,
                                FieldIn component,
                                FieldIn label,
                                WholeArrayIn generators,
                                WholeArrayOut componentGenerators);
  using ExecutionSignature = void(WorkIndex, _1, _2, _3, _4, _5);

  template <typename InPortal, typename OutPortal>
  VISKORES_EXEC void operator()(viskores::Id v,
                                viskores::Id root,
                                viskores::Id component,
                                viskores::Id label,
                                const InPortal& generators,
                                OutPortal& componentGenerators) const
  {
    if (root == v)
    {
      componentGenerators.Set(component, generators.Get(label));
    }
  }
};

struct ExceedsRadius : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn radius, FieldOut flag);
  using ExecutionSignature = _2(_1);

  VISKORES_CONT explicit ExceedsRadius(Real maxRadius)
    : MaxRadius(maxRadius)
  {
  }

  VISKORES_EXEC viskores::UInt8 operator()(Real radius) const
  {
    return (radius > this->MaxRadius) ? 1 : 0;
  }

  Real MaxRadius;
};

/// Orders budget candidates by decreasing cost of their worst member, then increasing label.
struct BudgetOrder
{
  VISKORES_EXEC_CONT bool operator()(const viskores::Pair<Real, viskores::Id>& a,
                                     const viskores::Pair<Real, viskores::Id>& b) const
  {
    return (a.first > b.first) || (a.first == b.first && a.second < b.second);
  }
};

struct BudgetCandidate : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn size,
                                FieldIn flag,
                                FieldIn worst,
                                WholeArrayIn costs,
                                FieldOut candidate,
                                FieldOut record);
  using ExecutionSignature = void(WorkIndex, _1, _2, _3, _4, _5, _6);

  template <typename CostsPortal>
  VISKORES_EXEC void operator()(viskores::Id cluster,
                                viskores::Id size,
                                viskores::UInt8 flag,
                                viskores::Id worst,
                                const CostsPortal& costs,
                                viskores::UInt8& candidate,
                                viskores::Pair<Real, viskores::Id>& record) const
  {
    candidate = (size > 1 && !flag) ? 1 : 0;
    record = viskores::Pair<Real, viskores::Id>(costs.Get(worst), cluster);
  }
};

struct FlagRecords : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn record, WholeArrayInOut flags);
  using ExecutionSignature = void(_1, _2);

  template <typename FlagsPortal>
  VISKORES_EXEC void operator()(const viskores::Pair<Real, viskores::Id>& record,
                                FlagsPortal& flags) const
  {
    flags.Set(record.second, 1);
  }
};

} // namespace detail

/// Step (b): compacts the labels to `0, ..., k-1` in increasing order and places every
/// generator.
inline void Place(const SurfaceMesh& mesh,
                  const viskores::cont::ArrayHandle<Mat3>& m,
                  Clustering& clustering)
{
  viskores::cont::Invoker invoke;
  clustering.Keys.BuildArrays(clustering.Labels, viskores::worklet::KeysSortType::Stable);
  clustering.NumberOfClusters = clustering.Keys.GetInputRange();
  invoke(detail::CompactLabels{}, clustering.Keys, clustering.Labels);
  invoke(detail::Place{},
         clustering.Keys,
         viskores::cont::ArrayHandleIndex(mesh.NumberOfPoints),
         mesh.Points,
         m,
         clustering.Anchors,
         clustering.Q,
         clustering.B,
         clustering.S,
         clustering.Generators,
         clustering.Sizes);
}

/// Step (c): costs, radii and worst members.
inline ClusterScan Scan(const SurfaceMesh& mesh,
                        const viskores::cont::ArrayHandle<Mat3>& m,
                        const Clustering& clustering)
{
  viskores::cont::Invoker invoke;
  ClusterScan scan;
  viskores::cont::ArrayHandle<Real> distances;
  invoke(detail::VertexScan{},
         mesh.Points,
         m,
         clustering.Labels,
         clustering.Generators,
         scan.Costs,
         distances);
  invoke(detail::ClusterReduce{},
         clustering.Keys,
         viskores::cont::ArrayHandleIndex(mesh.NumberOfPoints),
         scan.Costs,
         distances,
         scan.Radii,
         scan.Worst);
  return scan;
}

namespace detail
{
struct MarkFrozenLabel : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn frozen, FieldIn label, WholeArrayInOut frozenLabels);
  using ExecutionSignature = void(_1, _2, _3);

  template <typename FlagsPortal>
  VISKORES_EXEC void operator()(viskores::UInt8 frozen,
                                viskores::Id label,
                                FlagsPortal& frozenLabels) const
  {
    if (frozen)
    {
      frozenLabels.Set(label, 1); // frozen vertices are singletons: one writer per label
    }
  }
};
} // namespace detail

/// The clusters whose vertex is frozen, or an empty array without frozen vertices.
inline viskores::cont::ArrayHandle<viskores::UInt8> FrozenLabels(
  const viskores::cont::ArrayHandle<viskores::UInt8>& frozenVertices,
  const Clustering& clustering)
{
  viskores::cont::ArrayHandle<viskores::UInt8> labels;
  if (frozenVertices.GetNumberOfValues() == 0)
  {
    return labels;
  }
  labels.AllocateAndFill(clustering.NumberOfClusters, 0);
  viskores::cont::Invoker invoke;
  invoke(detail::MarkFrozenLabel{}, frozenVertices, clustering.Labels, labels);
  return labels;
}

namespace detail
{
struct GlueLabel : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn anchor, WholeArrayIn canonical, WholeArrayIn labels, FieldOut glued);
  using ExecutionSignature = _4(_1, _2, _3);

  template <typename IdPortal>
  VISKORES_EXEC viskores::Id operator()(viskores::Id anchor,
                                        const IdPortal& canonical,
                                        const IdPortal& labels) const
  {
    return labels.Get(canonical.Get(anchor));
  }
};
} // namespace detail

/// For an input cut into surfaces: every label mapped to the label of the first copy of the
/// input vertex of its anchor, which identifies the frozen copies of one vertex (Theorem
/// "Inputs that are not surfaces", (c)). Empty when `canonical` is.
inline viskores::cont::ArrayHandle<viskores::Id> GlueMap(
  const viskores::cont::ArrayHandle<viskores::Id>& canonical,
  const Clustering& clustering)
{
  viskores::cont::ArrayHandle<viskores::Id> glue;
  if (canonical.GetNumberOfValues() > 0)
  {
    viskores::cont::Invoker invoke;
    invoke(detail::GlueLabel{}, clustering.Anchors, canonical, clustering.Labels, glue);
  }
  return glue;
}

/// The radius flag (F0): `r_C > r_max`. Returns how many it flags.
inline viskores::Id RadiusFlags(const ClusterScan& scan,
                                Real maxRadius,
                                viskores::cont::ArrayHandle<viskores::UInt8>& flags)
{
  viskores::cont::Invoker invoke;
  invoke(detail::ExceedsRadius(maxRadius), scan.Radii, flags);
  return viskores::cont::Algorithm::Reduce(flags, viskores::Id(0));
}

/// The budget flag (Fb): if `k < k*`, the `k* - k` unflagged clusters with at least two vertices
/// whose worst members have the largest costs, ties broken by label. Returns how many it flags.
inline viskores::Id BudgetFlags(const Clustering& clustering,
                                const ClusterScan& scan,
                                viskores::Id budget,
                                viskores::cont::ArrayHandle<viskores::UInt8>& flags)
{
  using Algorithm = viskores::cont::Algorithm;
  const viskores::Id k = clustering.NumberOfClusters;
  if (budget <= 0 || k >= budget)
  {
    return 0;
  }
  viskores::cont::Invoker invoke;
  viskores::cont::ArrayHandle<viskores::UInt8> isCandidate;
  viskores::cont::ArrayHandle<viskores::Pair<Real, viskores::Id>> records, candidates;
  invoke(detail::BudgetCandidate{},
         clustering.Sizes,
         flags,
         scan.Worst,
         scan.Costs,
         isCandidate,
         records);
  Algorithm::CopyIf(records, isCandidate, candidates);
  Algorithm::Sort(candidates, detail::BudgetOrder{});
  const viskores::Id count = viskores::Min(budget - k, candidates.GetNumberOfValues());
  invoke(detail::FlagRecords{},
         viskores::cont::make_ArrayHandleView(candidates, 0, count),
         flags);
  return count;
}

namespace detail
{

/// Whether a vertex lies in a flagged cluster.
struct InFlagged : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn label, WholeArrayIn flags, FieldOut flagged);
  using ExecutionSignature = _3(_1, _2);

  template <typename FlagPortal>
  VISKORES_EXEC viskores::UInt8 operator()(viskores::Id label, const FlagPortal& flags) const
  {
    return flags.Get(label);
  }
};

/// The two centres of a flagged cluster: its generator and its worst member.
struct SplitCentres : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn cluster,
                                WholeArrayIn generators,
                                WholeArrayIn worst,
                                WholeArrayIn points,
                                FieldOut centre0,
                                FieldOut centre1);
  using ExecutionSignature = void(_1, _2, _3, _4, _5, _6);

  template <typename VecPortal, typename IdPortal>
  VISKORES_EXEC void operator()(viskores::Id cluster,
                                const VecPortal& generators,
                                const IdPortal& worst,
                                const VecPortal& points,
                                Vec3& centre0,
                                Vec3& centre1) const
  {
    centre0 = generators.Get(cluster);
    centre1 = points.Get(worst.Get(cluster));
  }
};

/// The key `2 slot + side` of a vertex of a flagged cluster: side 1 when the second centre costs
/// strictly less, so that ties stay on side 0.
struct SplitSide : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn vertex,
                                WholeArrayIn labels,
                                WholeArrayIn slots,
                                WholeArrayIn points,
                                WholeArrayIn m,
                                WholeArrayIn centres0,
                                WholeArrayIn centres1,
                                FieldOut key);
  using ExecutionSignature = _8(_1, _2, _3, _4, _5, _6, _7);

  template <typename IdPortal, typename VecPortal, typename MatPortal>
  VISKORES_EXEC viskores::Id operator()(viskores::Id v,
                                        const IdPortal& labels,
                                        const IdPortal& slots,
                                        const VecPortal& points,
                                        const MatPortal& m,
                                        const VecPortal& centres0,
                                        const VecPortal& centres1) const
  {
    const viskores::Id slot = slots.Get(labels.Get(v));
    const Vec3 p = points.Get(v);
    const Mat3 mv = m.Get(v);
    const bool second = Quad(Sub(centres1.Get(slot), p), mv) < Quad(Sub(centres0.Get(slot), p), mv);
    return 2 * slot + (second ? 1 : 0);
  }
};

/// Sets the centre of every side that has members, and marks the sides present.
struct ScatterSide : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn key,
                                FieldIn generator,
                                WholeArrayOut centres0,
                                WholeArrayOut centres1,
                                WholeArrayOut present);
  using ExecutionSignature = void(_1, _2, _3, _4, _5);

  template <typename VecPortal, typename FlagPortal>
  VISKORES_EXEC void operator()(viskores::Id key,
                                const Vec3& generator,
                                VecPortal& centres0,
                                VecPortal& centres1,
                                FlagPortal& present) const
  {
    if (key % 2 == 0)
    {
      centres0.Set(key / 2, generator);
    }
    else
    {
      centres1.Set(key / 2, generator);
    }
    present.Set(key, 1);
  }
};

/// The new label `k + slot` of a vertex on side 1 of a cluster split in two.
struct RelabelSide : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn vertex,
                                FieldIn key,
                                WholeArrayIn present,
                                WholeArrayInOut labels);
  using ExecutionSignature = void(_1, _2, _3, _4);

  VISKORES_CONT explicit RelabelSide(viskores::Id numClusters)
    : NumberOfClusters(numClusters)
  {
  }

  viskores::Id NumberOfClusters;

  template <typename FlagPortal, typename IdPortal>
  VISKORES_EXEC void operator()(viskores::Id v,
                                viskores::Id key,
                                const FlagPortal& present,
                                IdPortal& labels) const
  {
    const viskores::Id slot = key / 2;
    if (key % 2 == 1 && present.Get(2 * slot) && present.Get(2 * slot + 1))
    {
      labels.Set(v, this->NumberOfClusters + slot);
    }
  }
};

/// The generators of a flagged cluster after the split: its two centres, or, with an empty
/// side, the insertion at the worst member.
struct SplitGenerators : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn cluster,
                                FieldIn slot,
                                FieldIn centre0,
                                FieldIn centre1,
                                WholeArrayIn present,
                                WholeArrayIn worst,
                                WholeArrayIn points,
                                WholeArrayInOut labels,
                                WholeArrayInOut generators);
  using ExecutionSignature = void(_1, _2, _3, _4, _5, _6, _7, _8, _9);

  VISKORES_CONT explicit SplitGenerators(viskores::Id numClusters)
    : NumberOfClusters(numClusters)
  {
  }

  viskores::Id NumberOfClusters;

  template <typename FlagPortal,
            typename WorstPortal,
            typename PointPortal,
            typename LabelPortal,
            typename GeneratorPortal>
  VISKORES_EXEC void operator()(viskores::Id cluster,
                                viskores::Id slot,
                                const Vec3& centre0,
                                const Vec3& centre1,
                                const FlagPortal& present,
                                const WorstPortal& worst,
                                const PointPortal& points,
                                LabelPortal& labels,
                                GeneratorPortal& generators) const
  {
    if (present.Get(2 * slot) && present.Get(2 * slot + 1))
    {
      generators.Set(cluster, centre0);
      generators.Set(this->NumberOfClusters + slot, centre1);
    }
    else
    {
      const viskores::Id w = worst.Get(cluster);
      labels.Set(w, this->NumberOfClusters + slot);
      generators.Set(this->NumberOfClusters + slot, points.Get(w));
    }
  }
};

} // namespace detail

/// Step (d) in two: every flagged cluster is split by 4 Lloyd iterations restricted to its
/// members, from its generator and its worst member. A member goes to the second side when the
/// second centre costs strictly less, and each side with members is placed as in step (b): from
/// its member of smallest index, summed in increasing vertex order. The second side gets the
/// fresh label `k + rank`. A cluster with an empty side after the iterations gets the insertion
/// at its worst member instead. Starting from the generator and the worst member decreases the
/// objective by more than the insertion threshold, as the insertion does.
inline void SplitInTwo(const SurfaceMesh& mesh,
                       const viskores::cont::ArrayHandle<Mat3>& m,
                       const viskores::cont::ArrayHandle<viskores::UInt8>& flags,
                       const viskores::cont::ArrayHandle<viskores::Id>& rank,
                       viskores::Id numFlagged,
                       const ClusterScan& scan,
                       Clustering& clustering,
                       viskores::cont::ArrayHandle<Vec3>& generators)
{
  using Algorithm = viskores::cont::Algorithm;
  viskores::cont::Invoker invoke;
  const viskores::Id k = clustering.NumberOfClusters;
  viskores::cont::ArrayHandle<viskores::Id> flaggedClusters, vertices;
  Algorithm::CopyIf(viskores::cont::ArrayHandleIndex(k), flags, flaggedClusters);
  viskores::cont::ArrayHandle<viskores::UInt8> inFlagged;
  invoke(detail::InFlagged{}, clustering.Labels, flags, inFlagged);
  Algorithm::CopyIf(viskores::cont::ArrayHandleIndex(mesh.NumberOfPoints), inFlagged, vertices);
  viskores::cont::ArrayHandle<Vec3> centres0, centres1;
  invoke(detail::SplitCentres{},
         flaggedClusters,
         clustering.Generators,
         scan.Worst,
         mesh.Points,
         centres0,
         centres1);
  viskores::cont::ArrayHandle<viskores::Id> keys;
  viskores::cont::ArrayHandle<viskores::UInt8> present;
  for (int iteration = 0; iteration < 4; ++iteration)
  {
    invoke(detail::SplitSide{},
           vertices,
           clustering.Labels,
           rank,
           mesh.Points,
           m,
           centres0,
           centres1,
           keys);
    // Stable: every side in increasing vertex order, as step (b) sums.
    viskores::worklet::Keys<viskores::Id> sides;
    sides.BuildArrays(keys, viskores::worklet::KeysSortType::Stable);
    viskores::cont::ArrayHandle<viskores::Id> anchors, sizes;
    viskores::cont::ArrayHandle<Mat3> q;
    viskores::cont::ArrayHandle<Vec3> b, placed;
    viskores::cont::ArrayHandle<Real> s;
    invoke(detail::Place{}, sides, vertices, mesh.Points, m, anchors, q, b, s, placed, sizes);
    present.AllocateAndFill(2 * numFlagged, 0);
    invoke(detail::ScatterSide{}, sides.GetUniqueKeys(), placed, centres0, centres1, present);
  }
  // The sides of the last iteration, whose placements the centres are.
  invoke(detail::RelabelSide(k), vertices, keys, present, clustering.Labels);
  invoke(detail::SplitGenerators(k),
         flaggedClusters,
         viskores::cont::ArrayHandleIndex(numFlagged),
         centres0,
         centres1,
         present,
         scan.Worst,
         mesh.Points,
         clustering.Labels,
         generators);
}

/// Steps (d) and (e): inserts a generator at the worst member of every flagged cluster, or with
/// `splitInTwo` splits it in two by `SplitInTwo`, then splits every cluster into its connected
/// components, each keeping its parent's generator.
inline void InsertAndSplit(const SurfaceMesh& mesh,
                           const viskores::cont::ArrayHandle<Mat3>& m,
                           const viskores::cont::ArrayHandle<viskores::UInt8>& flags,
                           const ClusterScan& scan,
                           bool splitInTwo,
                           Clustering& clustering)
{
  using Algorithm = viskores::cont::Algorithm;
  viskores::cont::Invoker invoke;
  const viskores::Id k = clustering.NumberOfClusters;
  viskores::cont::ArrayHandle<viskores::Id> rank;
  const viskores::Id numFlagged =
    Algorithm::ScanExclusive(viskores::cont::make_ArrayHandleCast<viskores::Id>(flags), rank);
  viskores::cont::ArrayHandle<Vec3> generators;
  generators.Allocate(k + numFlagged);
  Algorithm::CopySubRange(clustering.Generators, 0, k, generators, 0);
  if (splitInTwo && numFlagged > 0)
  {
    SplitInTwo(mesh, m, flags, rank, numFlagged, scan, clustering, generators);
  }
  else
  {
    invoke(detail::Insert(k), flags, rank, scan.Worst, clustering.Labels, mesh.Points, generators);
  }

  viskores::cont::ArrayHandle<viskores::Id> components, roots;
  const viskores::Id numComponents =
    LabelComponents(mesh.NumberOfPoints, mesh.Edges, clustering.Labels, components, &roots);
  clustering.Generators.Allocate(numComponents);
  invoke(detail::ComponentGenerator{},
         roots,
         components,
         clustering.Labels,
         generators,
         clustering.Generators);
  clustering.Labels = components;
  clustering.NumberOfClusters = numComponents;
}
}
}
} // namespace viskores::worklet::quadric_clustering

VISKORES_QUADRIC_CLUSTERING_STRICT_FP_END

#endif // viskores_worklet_quadric_clustering_Clustering_h
