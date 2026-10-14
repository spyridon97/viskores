//============================================================================
//  The contents of this file are covered by the Viskores license. See
//  LICENSE.txt for details.
//
//  By contributing to this file, all contributors agree to the Developer
//  Certificate of Origin Version 1.1 (DCO 1.1) as stated in DCO.txt.
//============================================================================
#ifndef viskores_worklet_quadric_clustering_TopologyFlags_h
#define viskores_worklet_quadric_clustering_TopologyFlags_h

#include <viskores/filter/geometry_refinement/worklet/quadric_clustering/SurfaceMesh.h>

#include <viskores/BinaryOperators.h>
#include <viskores/cont/Algorithm.h>
#include <viskores/cont/ArrayHandleGroupVec.h>
#include <viskores/cont/ArrayHandleIndex.h>

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

/// The violations of the discrete closed-ball property found by `TopologyFlags`, and the counts
/// of Lemma "Euler" of the paper, which tests can check against the Euler characteristic.
struct TopologyReport
{
  /// Whether the flags were evaluated.
  bool Valid = false;
  viskores::Id NumberOfD0 = 0;
  viskores::Id NumberOfD1 = 0;
  viskores::Id NumberOfD2 = 0;
  /// `sum_C chi(M[C])`.
  viskores::Id SumOfClusterEuler = 0;
  /// The number of 3-label triangles.
  viskores::Id NumberOfThreeLabelTriangles = 0;
  /// The number of boundary edges whose ends carry different labels.
  viskores::Id NumberOfBoundaryCutEdges = 0;
};

namespace detail
{

/// The sorted label triple of a 3-label triangle, or `(-1, -1, -1)`.
struct LabelTriple : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn triangle, WholeArrayIn labels, FieldOut triple);
  using ExecutionSignature = _3(_1, _2);

  template <typename LabelsPortal>
  VISKORES_EXEC viskores::Id3 operator()(const viskores::Id3& triangle,
                                         const LabelsPortal& labels) const
  {
    viskores::Id a = labels.Get(triangle[0]);
    viskores::Id b = labels.Get(triangle[1]);
    viskores::Id c = labels.Get(triangle[2]);
    if (a == b || b == c || a == c)
    {
      return viskores::Id3(-1);
    }
    auto order = [](viskores::Id& x, viskores::Id& y)
    {
      if (x > y)
      {
        const viskores::Id z = x;
        x = y;
        y = z;
      }
    };
    order(a, b);
    order(b, c);
    order(a, b);
    return viskores::Id3(a, b, c);
  }
};

struct IsThreeLabel : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn triple, FieldOut flag);
  using ExecutionSignature = _2(_1);

  VISKORES_EXEC viskores::UInt8 operator()(const viskores::Id3& triple) const
  {
    return (triple[0] >= 0) ? 1 : 0;
  }
};

/// Per-cluster counts: edges inside, triangles inside, boundary vertices, boundary edges inside.
/// Integer atomics give the same counts in any order.
struct CountEdges : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn edge,
                                FieldIn edgeTriangles,
                                WholeArrayIn labels,
                                AtomicArrayInOut inside,
                                AtomicArrayInOut boundaryInside);
  using ExecutionSignature = void(_1, _2, _3, _4, _5);

  template <typename LabelsPortal, typename Atomic>
  VISKORES_EXEC void operator()(const viskores::Id2& edge,
                                const viskores::Id2& edgeTriangles,
                                const LabelsPortal& labels,
                                Atomic& inside,
                                Atomic& boundaryInside) const
  {
    const viskores::Id label = labels.Get(edge[0]);
    if (label == labels.Get(edge[1]))
    {
      inside.Add(label, 1);
      if (edgeTriangles[1] < 0)
      {
        boundaryInside.Add(label, 1);
      }
    }
  }
};

struct CountTriangles : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn triangle, WholeArrayIn labels, AtomicArrayInOut inside);
  using ExecutionSignature = void(_1, _2, _3);

  template <typename LabelsPortal, typename Atomic>
  VISKORES_EXEC void operator()(const viskores::Id3& triangle,
                                const LabelsPortal& labels,
                                Atomic& inside) const
  {
    const viskores::Id label = labels.Get(triangle[0]);
    if (label == labels.Get(triangle[1]) && label == labels.Get(triangle[2]))
    {
      inside.Add(label, 1);
    }
  }
};

/// (D0): `chi(M[C]) = 1`, and the boundary vertices of `C` form at most one run:
/// `beta_C - gamma_C = [beta_C > 0]`. Clusters are connected, which is not tested here.
struct D0 : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn size,
                                FieldIn edgesInside,
                                FieldIn trianglesInside,
                                FieldIn boundaryVertices,
                                FieldIn boundaryEdgesInside,
                                FieldOut euler,
                                FieldOut violates);
  using ExecutionSignature = void(_1, _2, _3, _4, _5, _6, _7);

  VISKORES_EXEC void operator()(viskores::Id size,
                                viskores::Id edgesInside,
                                viskores::Id trianglesInside,
                                viskores::Id boundaryVertices,
                                viskores::Id boundaryEdgesInside,
                                viskores::Id& euler,
                                viskores::UInt8& violates) const
  {
    euler = size - edgesInside + trianglesInside;
    const viskores::Id runs = boundaryVertices - boundaryEdgesInside;
    violates = (euler != 1 || runs != ((boundaryVertices > 0) ? 1 : 0)) ? 1 : 0;
  }
};

/// The component of every cluster, which all its vertices share.
struct ClusterComponent : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn label, FieldIn component, AtomicArrayInOut components);
  using ExecutionSignature = void(_1, _2, _3);

  template <typename Atomic>
  VISKORES_EXEC void operator()(viskores::Id label,
                                viskores::Id component,
                                Atomic& components) const
  {
    components.Set(label, component);
  }
};

/// The clusters of every component. Integer atomics give the same counts in any order.
struct CountClusters : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn component, AtomicArrayInOut count);
  using ExecutionSignature = void(_1, _2);

  template <typename Atomic>
  VISKORES_EXEC void operator()(viskores::Id component, Atomic& count) const
  {
    count.Add(component, 1);
  }
};

/// (D2) by count, per component: a closed component with exactly three clusters.
struct D2ByCount : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn clusters, FieldIn boundaryVertices, FieldOut violates);
  using ExecutionSignature = _3(_1, _2);

  VISKORES_EXEC viskores::UInt8 operator()(viskores::Id clusters,
                                           viskores::Id boundaryVertices) const
  {
    return (clusters == 3 && boundaryVertices == 0) ? 1 : 0;
  }
};

/// Flags a cluster with at least two vertices of a component violating (D2).
struct FlagD2 : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn size, FieldIn component, WholeArrayIn d2, FieldInOut flag);
  using ExecutionSignature = void(_1, _2, _3, _4);

  template <typename FlagPortal>
  VISKORES_EXEC void operator()(viskores::Id size,
                                viskores::Id component,
                                const FlagPortal& d2,
                                viskores::UInt8& flag) const
  {
    if (size > 1 && d2.Get(component))
    {
      flag = 1;
    }
  }
};

/// For every label pair: a record `(cut, d, e)` from a cut edge (`e` = 1 on the boundary), or
/// `(0, 1, 0)` from each pair of a 3-label triangle.
struct CutEdgeRecord : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn edge,
                                FieldIn edgeTriangles,
                                WholeArrayIn labels,
                                FieldOut pair,
                                FieldOut record);
  using ExecutionSignature = void(_1, _2, _3, _4, _5);

  template <typename LabelsPortal>
  VISKORES_EXEC void operator()(const viskores::Id2& edge,
                                const viskores::Id2& edgeTriangles,
                                const LabelsPortal& labels,
                                viskores::Id2& pair,
                                viskores::Id3& record) const
  {
    const viskores::Id a = labels.Get(edge[0]);
    const viskores::Id b = labels.Get(edge[1]);
    pair = viskores::Id2(viskores::Min(a, b), viskores::Max(a, b));
    record = (a != b) ? viskores::Id3(1, 0, (edgeTriangles[1] < 0) ? 1 : 0) : viskores::Id3(0);
  }
};

struct TriplePairs : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn triple, FieldOut pairs, FieldOut records);
  using ExecutionSignature = void(_1, _2, _3);

  template <typename PairsVec, typename RecordsVec>
  VISKORES_EXEC void operator()(const viskores::Id3& triple,
                                PairsVec& pairs,
                                RecordsVec& records) const
  {
    pairs[0] = viskores::Id2(triple[0], triple[1]);
    pairs[1] = viskores::Id2(triple[1], triple[2]);
    pairs[2] = viskores::Id2(triple[0], triple[2]);
    for (viskores::IdComponent i = 0; i < 3; ++i)
    {
      records[i] = viskores::Id3(0, 1, 0);
    }
  }
};

/// (D1): a cut pair must lie in at least one 3-label triangle, and in exactly two counting the
/// boundary cut edges.
struct D1 : viskores::worklet::WorkletReduceByKey
{
  using ControlSignature = void(KeysIn pairs, ValuesIn records, ReducedValuesOut violates);
  using ExecutionSignature = _3(_2);
  using InputDomain = _1;

  template <typename RecordsVec>
  VISKORES_EXEC viskores::UInt8 operator()(const RecordsVec& records) const
  {
    viskores::Id cut = 0, d = 0, e = 0;
    for (viskores::IdComponent i = 0; i < records.GetNumberOfComponents(); ++i)
    {
      cut += records[i][0];
      d += records[i][1];
      e += records[i][2];
    }
    return (cut > 0 && (d < 1 || d + e != 2)) ? 1 : 0;
  }
};

/// (D2): a label triple must come from a single triangle.
struct D2 : viskores::worklet::WorkletReduceByKey
{
  using ControlSignature = void(KeysIn triples, ReducedValuesOut violates);
  using ExecutionSignature = _2(_1, ValueCount);
  using InputDomain = _1;

  VISKORES_EXEC viskores::UInt8 operator()(const viskores::Id3&, viskores::IdComponent count) const
  {
    return (count > 1) ? 1 : 0;
  }
};

struct PairLabels : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn pair, FieldOut labels);
  using ExecutionSignature = void(_1, _2);

  template <typename LabelsVec>
  VISKORES_EXEC void operator()(const viskores::Id2& pair, LabelsVec& labels) const
  {
    labels[0] = pair[0];
    labels[1] = pair[1];
  }
};

struct TripleLabels : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn triple, FieldOut labels);
  using ExecutionSignature = void(_1, _2);

  template <typename LabelsVec>
  VISKORES_EXEC void operator()(const viskores::Id3& triple, LabelsVec& labels) const
  {
    labels[0] = triple[0];
    labels[1] = triple[1];
    labels[2] = triple[2];
  }
};

/// Flags every listed cluster with at least two vertices. The list has no duplicates.
struct FlagListed : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn label, WholeArrayIn sizes, WholeArrayInOut flags);
  using ExecutionSignature = void(_1, _2, _3);

  template <typename SizesPortal, typename FlagsPortal>
  VISKORES_EXEC void operator()(viskores::Id label,
                                const SizesPortal& sizes,
                                FlagsPortal& flags) const
  {
    if (sizes.Get(label) > 1)
    {
      flags.Set(label, 1);
    }
  }
};

struct IsValidLabel : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn label, FieldOut valid);
  using ExecutionSignature = _2(_1);

  VISKORES_EXEC viskores::UInt8 operator()(viskores::Id label) const { return label >= 0 ? 1 : 0; }
};

struct FirstTriangle : viskores::worklet::WorkletReduceByKey
{
  using ControlSignature = void(KeysIn triples, ValuesIn triangles, ReducedValuesOut first);
  using ExecutionSignature = _3(_2);
  using InputDomain = _1;

  template <typename IdVec>
  VISKORES_EXEC viskores::Id operator()(const IdVec& triangles) const
  {
    viskores::Id first = triangles[0];
    for (viskores::IdComponent i = 1; i < triangles.GetNumberOfComponents(); ++i)
    {
      const viskores::Id t = triangles[i];
      first = viskores::Min(first, t);
    }
    return first;
  }
};

struct TriangleLabels : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn triangleId,
                                WholeArrayIn triangles,
                                WholeArrayIn labels,
                                FieldOut outputTriangle);
  using ExecutionSignature = _4(_1, _2, _3);

  template <typename TrianglesPortal, typename LabelsPortal>
  VISKORES_EXEC viskores::Id3 operator()(viskores::Id t,
                                         const TrianglesPortal& triangles,
                                         const LabelsPortal& labels) const
  {
    const viskores::Id3 tri = triangles.Get(t);
    return viskores::Id3(labels.Get(tri[0]), labels.Get(tri[1]), labels.Get(tri[2]));
  }
};

} // namespace detail

/// Flags every listed cluster that has at least two vertices.
inline void FlagClusters(viskores::cont::ArrayHandle<viskores::Id>& listed,
                         const viskores::cont::ArrayHandle<viskores::Id>& sizes,
                         viskores::cont::ArrayHandle<viskores::UInt8>& flags)
{
  viskores::cont::Algorithm::Sort(listed);
  viskores::cont::Algorithm::Unique(listed);
  viskores::cont::Invoker invoke;
  invoke(detail::FlagListed{}, listed, sizes, flags);
}

/// The 3-label triangles of a labelling: their indices, and their sorted label triples.
inline void ThreeLabelTriangles(const SurfaceMesh& mesh,
                                const viskores::cont::ArrayHandle<viskores::Id>& labels,
                                viskores::cont::ArrayHandle<viskores::Id>& indices,
                                viskores::cont::ArrayHandle<viskores::Id3>& triples)
{
  viskores::cont::Invoker invoke;
  viskores::cont::ArrayHandle<viskores::Id3> all;
  invoke(detail::LabelTriple{}, mesh.Triangles, labels, all);
  viskores::cont::ArrayHandle<viskores::UInt8> isThree;
  invoke(detail::IsThreeLabel{}, all, isThree);
  viskores::cont::Algorithm::CopyIf(
    viskores::cont::ArrayHandleIndex(mesh.NumberOfTriangles), isThree, indices);
  viskores::cont::Algorithm::CopyIf(all, isThree, triples);
}

/// The topological flags (F2) and (F3) of Algorithm 4: (D0) per cluster, and the clusters with
/// at least two vertices of the pairs violating (D1) and of the triples violating (D2), or with
/// `d2ByCount` of the components violating it (Section 11). Sets the flags in `flags` and
/// returns the violations found: for (D2), triples or components.
inline TopologyReport TopologyFlags(const SurfaceMesh& mesh,
                                    const viskores::cont::ArrayHandle<viskores::Id>& labels,
                                    const viskores::cont::ArrayHandle<viskores::Id>& sizes,
                                    viskores::cont::ArrayHandle<viskores::UInt8>& flags,
                                    bool d2ByCount = false)
{
  using Algorithm = viskores::cont::Algorithm;
  viskores::cont::Invoker invoke;
  TopologyReport report;
  report.Valid = true;
  const viskores::Id k = sizes.GetNumberOfValues();

  // (D0)
  viskores::cont::ArrayHandle<viskores::Id> edgesInside, trianglesInside, boundaryVertices,
    boundaryEdgesInside;
  edgesInside.AllocateAndFill(k, 0);
  trianglesInside.AllocateAndFill(k, 0);
  boundaryVertices.AllocateAndFill(k, 0);
  boundaryEdgesInside.AllocateAndFill(k, 0);
  invoke(detail::CountEdges{},
         mesh.Edges,
         mesh.EdgeTriangles,
         labels,
         edgesInside,
         boundaryEdgesInside);
  invoke(detail::CountTriangles{}, mesh.Triangles, labels, trianglesInside);
  invoke(detail::CountBoundaryVertices{}, mesh.IsBoundaryVertex, labels, boundaryVertices);
  viskores::cont::ArrayHandle<viskores::Id> euler;
  viskores::cont::ArrayHandle<viskores::UInt8> d0;
  invoke(detail::D0{},
         sizes,
         edgesInside,
         trianglesInside,
         boundaryVertices,
         boundaryEdgesInside,
         euler,
         d0);
  report.NumberOfD0 = Algorithm::Reduce(d0, viskores::Id(0));
  report.SumOfClusterEuler = Algorithm::Reduce(euler, viskores::Id(0));
  // A cluster violating (D0) has at least two vertices: a singleton never does.
  OrFlags(flags, d0);

  // (D2)
  viskores::cont::ArrayHandle<viskores::Id> threeIndices;
  viskores::cont::ArrayHandle<viskores::Id3> triples;
  ThreeLabelTriangles(mesh, labels, threeIndices, triples);
  report.NumberOfThreeLabelTriangles = triples.GetNumberOfValues();
  viskores::cont::ArrayHandle<viskores::Id> listed;
  if (d2ByCount)
  {
    // With (D0) and (D1), (D2) fails only on a closed component with exactly three clusters
    // (Proposition "Condition (D2) is almost redundant"), so the clusters with at least two
    // vertices of such a component are flagged in place of a sort of the label triples. One has
    // two, as a closed component has at least four vertices.
    viskores::cont::ArrayHandle<viskores::Id> clusterComponents, componentClusters;
    clusterComponents.AllocateAndFill(k, 0);
    invoke(detail::ClusterComponent{}, labels, mesh.Components, clusterComponents);
    componentClusters.AllocateAndFill(mesh.NumberOfComponents, 0);
    invoke(detail::CountClusters{}, clusterComponents, componentClusters);
    viskores::cont::ArrayHandle<viskores::UInt8> d2;
    invoke(detail::D2ByCount{}, componentClusters, mesh.ComponentBoundaryVertices, d2);
    report.NumberOfD2 = Algorithm::Reduce(d2, viskores::Id(0));
    if (report.NumberOfD2 > 0)
    {
      invoke(detail::FlagD2{}, sizes, clusterComponents, d2, flags);
    }
  }
  else if (triples.GetNumberOfValues() > 0)
  {
    viskores::worklet::Keys<viskores::Id3> tripleKeys;
    tripleKeys.BuildArrays(triples, viskores::worklet::KeysSortType::Unstable);
    viskores::cont::ArrayHandle<viskores::UInt8> d2;
    invoke(detail::D2{}, tripleKeys, d2);
    viskores::cont::ArrayHandle<viskores::Id3> bad;
    Algorithm::CopyIf(tripleKeys.GetUniqueKeys(), d2, bad);
    report.NumberOfD2 = bad.GetNumberOfValues();
    viskores::cont::ArrayHandle<viskores::Id> badLabels;
    invoke(detail::TripleLabels{}, bad, viskores::cont::make_ArrayHandleGroupVec<3>(badLabels));
    listed = badLabels;
  }

  // (D1)
  viskores::cont::ArrayHandle<viskores::Id2> edgePairs;
  viskores::cont::ArrayHandle<viskores::Id3> edgeRecords;
  invoke(detail::CutEdgeRecord{}, mesh.Edges, mesh.EdgeTriangles, labels, edgePairs, edgeRecords);
  viskores::cont::ArrayHandle<viskores::Id2> triplePairs;
  viskores::cont::ArrayHandle<viskores::Id3> tripleRecords;
  invoke(detail::TriplePairs{},
         triples,
         viskores::cont::make_ArrayHandleGroupVec<3>(triplePairs),
         viskores::cont::make_ArrayHandleGroupVec<3>(tripleRecords));
  const viskores::Id numEdges = edgePairs.GetNumberOfValues();
  const viskores::Id numTriplePairs = triplePairs.GetNumberOfValues();
  viskores::cont::ArrayHandle<viskores::Id2> pairs;
  viskores::cont::ArrayHandle<viskores::Id3> records;
  pairs.Allocate(numEdges + numTriplePairs);
  records.Allocate(numEdges + numTriplePairs);
  Algorithm::CopySubRange(edgePairs, 0, numEdges, pairs, 0);
  Algorithm::CopySubRange(edgeRecords, 0, numEdges, records, 0);
  Algorithm::CopySubRange(triplePairs, 0, numTriplePairs, pairs, numEdges);
  Algorithm::CopySubRange(tripleRecords, 0, numTriplePairs, records, numEdges);
  viskores::worklet::Keys<viskores::Id2> pairKeys;
  pairKeys.BuildArrays(pairs, viskores::worklet::KeysSortType::Unstable);
  viskores::cont::ArrayHandle<viskores::UInt8> d1;
  invoke(detail::D1{}, pairKeys, records, d1);
  viskores::cont::ArrayHandle<viskores::Id2> badPairs;
  Algorithm::CopyIf(pairKeys.GetUniqueKeys(), d1, badPairs);
  report.NumberOfD1 = badPairs.GetNumberOfValues();
  {
    viskores::cont::ArrayHandle<viskores::Id> badLabels;
    invoke(detail::PairLabels{}, badPairs, viskores::cont::make_ArrayHandleGroupVec<2>(badLabels));
    const viskores::Id before = listed.GetNumberOfValues();
    viskores::cont::ArrayHandle<viskores::Id> all;
    all.Allocate(before + badLabels.GetNumberOfValues());
    Algorithm::CopySubRange(listed, 0, before, all, 0);
    Algorithm::CopySubRange(badLabels, 0, badLabels.GetNumberOfValues(), all, before);
    listed = all;
  }
  FlagClusters(listed, sizes, flags);

  // Boundary cut edges, for the Euler identity.
  report.NumberOfBoundaryCutEdges =
    Algorithm::Reduce(edgeRecords, viskores::Id3(0), viskores::Sum())[2];
  return report;
}

namespace detail
{
struct SortedLabels : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn triangle, WholeArrayIn labels, FieldOut sorted);
  using ExecutionSignature = _3(_1, _2);

  template <typename LabelsPortal>
  VISKORES_EXEC viskores::Id3 operator()(const viskores::Id3& t, const LabelsPortal& labels) const
  {
    viskores::Id a = labels.Get(t[0]), b = labels.Get(t[1]), c = labels.Get(t[2]);
    auto order = [](viskores::Id& x, viskores::Id& y)
    {
      if (x > y)
      {
        const viskores::Id z = x;
        x = y;
        y = z;
      }
    };
    order(a, b);
    order(b, c);
    order(a, b);
    return viskores::Id3(a, b, c);
  }
};

struct GlueTriangle : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn triangle, WholeArrayIn glue, FieldOut glued);
  using ExecutionSignature = _3(_1, _2);

  template <typename GluePortal>
  VISKORES_EXEC viskores::Id3 operator()(const viskores::Id3& t, const GluePortal& glue) const
  {
    return viskores::Id3(glue.Get(t[0]), glue.Get(t[1]), glue.Get(t[2]));
  }
};
} // namespace detail

namespace detail
{
/// For an edge with exactly one frozen end: the cluster of the other end, the first copy of the
/// input vertex of the frozen end, and the frozen copy; otherwise `(-1, -1, -1)`.
struct FrozenContact : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn edge,
                                WholeArrayIn frozen,
                                WholeArrayIn canonical,
                                WholeArrayIn labels,
                                FieldOut contact);
  using ExecutionSignature = _5(_1, _2, _3, _4);

  template <typename FlagPortal, typename IdPortal>
  VISKORES_EXEC viskores::Id3 operator()(const viskores::Id2& edge,
                                         const FlagPortal& frozen,
                                         const IdPortal& canonical,
                                         const IdPortal& labels) const
  {
    const bool a = frozen.Get(edge[0]) != 0, b = frozen.Get(edge[1]) != 0;
    if (a == b)
    {
      return viskores::Id3(-1);
    }
    const viskores::Id u = a ? edge[1] : edge[0];
    const viskores::Id x = a ? edge[0] : edge[1];
    return viskores::Id3(labels.Get(u), canonical.Get(x), x);
  }
};

struct IsContact : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn contact, FieldOut valid);
  using ExecutionSignature = _2(_1);

  VISKORES_EXEC viskores::UInt8 operator()(const viskores::Id3& contact) const
  {
    return contact[0] >= 0 ? 1 : 0;
  }
};

/// A cluster with two contacts to different copies of one input vertex.
struct DoubleContact : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn index, WholeArrayIn contacts, FieldOut cluster);
  using ExecutionSignature = _3(_1, _2);

  template <typename ContactPortal>
  VISKORES_EXEC viskores::Id operator()(viskores::Id index, const ContactPortal& contacts) const
  {
    if (index == 0)
    {
      return -1;
    }
    const viskores::Id3 a = contacts.Get(index - 1), b = contacts.Get(index);
    return (a[0] == b[0] && a[1] == b[1]) ? b[0] : viskores::Id(-1);
  }
};
} // namespace detail

/// For an input cut into surfaces: the frozen copies every cluster is adjacent to in `K_X`, as
/// sorted triples (cluster, first copy of the input vertex, copy), without repetition.
inline viskores::cont::ArrayHandle<viskores::Id3> FrozenContacts(
  const SurfaceMesh& mesh,
  const viskores::cont::ArrayHandle<viskores::Id>& labels,
  const viskores::cont::ArrayHandle<viskores::UInt8>& frozenVertices,
  const viskores::cont::ArrayHandle<viskores::Id>& canonical)
{
  viskores::cont::Invoker invoke;
  viskores::cont::ArrayHandle<viskores::Id3> all, contacts;
  invoke(detail::FrozenContact{}, mesh.Edges, frozenVertices, canonical, labels, all);
  viskores::cont::ArrayHandle<viskores::UInt8> valid;
  invoke(detail::IsContact{}, all, valid);
  viskores::cont::Algorithm::CopyIf(all, valid, contacts);
  viskores::cont::Algorithm::Sort(contacts);
  viskores::cont::Algorithm::Unique(contacts);
  return contacts;
}

/// The glue condition for an input cut into surfaces (Theorem "Inputs that are not surfaces",
/// (d)): flags every cluster adjacent in `K_X` to two copies of one frozen input vertex, whose
/// gluing would identify two of its edges or triangles. Returns how many clusters it finds.
inline viskores::Id GlueFlags(const SurfaceMesh& mesh,
                              const viskores::cont::ArrayHandle<viskores::Id>& labels,
                              const viskores::cont::ArrayHandle<viskores::UInt8>& frozenVertices,
                              const viskores::cont::ArrayHandle<viskores::Id>& canonical,
                              const viskores::cont::ArrayHandle<viskores::Id>& sizes,
                              viskores::cont::ArrayHandle<viskores::UInt8>& flags)
{
  if (frozenVertices.GetNumberOfValues() == 0)
  {
    return 0;
  }
  viskores::cont::Invoker invoke;
  const viskores::cont::ArrayHandle<viskores::Id3> contacts =
    FrozenContacts(mesh, labels, frozenVertices, canonical);
  viskores::cont::ArrayHandle<viskores::Id> doubled, listed;
  invoke(detail::DoubleContact{},
         viskores::cont::ArrayHandleIndex(contacts.GetNumberOfValues()),
         contacts,
         doubled);
  viskores::cont::ArrayHandle<viskores::UInt8> valid;
  invoke(detail::IsValidLabel{}, doubled, valid);
  viskores::cont::Algorithm::CopyIf(doubled, valid, listed);
  viskores::cont::Algorithm::Sort(listed);
  viskores::cont::Algorithm::Unique(listed);
  const viskores::Id count = listed.GetNumberOfValues();
  FlagClusters(listed, sizes, flags);
  return count;
}

/// The simplices `g(t)` of the geometric-only mode: the sorted label triple of every input
/// triangle, once each, with labels repeated for a segment or a point.
inline viskores::cont::ArrayHandle<viskores::Id3> Simplices(
  const SurfaceMesh& mesh,
  const viskores::cont::ArrayHandle<viskores::Id>& labels)
{
  viskores::cont::Invoker invoke;
  viskores::cont::ArrayHandle<viskores::Id3> simplices;
  invoke(detail::SortedLabels{}, mesh.Triangles, labels, simplices);
  viskores::cont::Algorithm::Sort(simplices);
  viskores::cont::Algorithm::Unique(simplices);
  return simplices;
}

/// The triangles with every label `l` replaced by `glue[l]`.
inline viskores::cont::ArrayHandle<viskores::Id3> GlueTriangles(
  const viskores::cont::ArrayHandle<viskores::Id3>& triangles,
  const viskores::cont::ArrayHandle<viskores::Id>& glue)
{
  if (glue.GetNumberOfValues() == 0)
  {
    return triangles;
  }
  viskores::cont::Invoker invoke;
  viskores::cont::ArrayHandle<viskores::Id3> glued;
  invoke(detail::GlueTriangle{}, triangles, glue, glued);
  return glued;
}

/// The output triangles of Algorithm 5: one per distinct label triple of the 3-label triangles,
/// in the orientation and order of the first input triangle producing it, whose index is
/// stored in `sources` when given.
inline viskores::cont::ArrayHandle<viskores::Id3> OutputTriangles(
  const SurfaceMesh& mesh,
  const viskores::cont::ArrayHandle<viskores::Id>& labels,
  viskores::cont::ArrayHandle<viskores::Id>* sources = nullptr)
{
  viskores::cont::Invoker invoke;
  viskores::cont::ArrayHandle<viskores::Id> indices;
  viskores::cont::ArrayHandle<viskores::Id3> triples;
  ThreeLabelTriangles(mesh, labels, indices, triples);
  viskores::cont::ArrayHandle<viskores::Id3> result;
  viskores::cont::ArrayHandle<viskores::Id> first;
  if (triples.GetNumberOfValues() > 0)
  {
    viskores::worklet::Keys<viskores::Id3> keys;
    keys.BuildArrays(triples, viskores::worklet::KeysSortType::Unstable);
    invoke(detail::FirstTriangle{}, keys, indices, first);
    viskores::cont::Algorithm::Sort(first);
    invoke(detail::TriangleLabels{}, first, mesh.Triangles, labels, result);
  }
  if (sources)
  {
    *sources = first;
  }
  return result;
}

}
}
} // namespace viskores::worklet::quadric_clustering

VISKORES_QUADRIC_CLUSTERING_STRICT_FP_END

#endif // viskores_worklet_quadric_clustering_TopologyFlags_h
