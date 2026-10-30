//============================================================================
//  The contents of this file are covered by the Viskores license. See
//  LICENSE.txt for details.
//
//  By contributing to this file, all contributors agree to the Developer
//  Certificate of Origin Version 1.1 (DCO 1.1) as stated in DCO.txt.
//============================================================================
#ifndef viskores_worklet_quadric_clustering_EmbeddingFlags_h
#define viskores_worklet_quadric_clustering_EmbeddingFlags_h

#include <viskores/filter/geometry_refinement/worklet/quadric_clustering/ImproperPairs.h>
#include <viskores/filter/geometry_refinement/worklet/quadric_clustering/QuadricClusteringCommonWorklets.h>
#include <viskores/filter/geometry_refinement/worklet/quadric_clustering/TopologyFlags.h>

#include <viskores/cont/Algorithm.h>
#include <viskores/cont/ArrayHandleGroupVec.h>

#include <viskores/worklet/spatialstructure/BoundingVolumeHierarchy.h>

VISKORES_QUADRIC_CLUSTERING_STRICT_FP_BEGIN

namespace viskores
{
namespace worklet
{
namespace quadric_clustering
{

struct EmbeddingReport
{
  /// Pairs of output triangles that meet other than in their common vertex or edge.
  viskores::Id NumberOfImproperPairs = 0;
  /// Degenerate output triangles.
  viskores::Id NumberOfDegenerateTriangles = 0;
};

namespace detail
{

/// The realised output triangle: its box, and whether it is degenerate, exactly.
struct RealizeTriangle : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn triangle,
                                WholeArrayIn generators,
                                FieldOut lower,
                                FieldOut upper,
                                FieldOut degenerate);
  using ExecutionSignature = void(_1, _2, _3, _4, _5);

  template <typename GeneratorsPortal>
  VISKORES_EXEC void operator()(const viskores::Id3& triangle,
                                const GeneratorsPortal& generators,
                                Vec3& lower,
                                Vec3& upper,
                                viskores::UInt8& degenerate) const
  {
    const Vec3 a = generators.Get(triangle[0]);
    const Vec3 b = generators.Get(triangle[1]);
    const Vec3 c = generators.Get(triangle[2]);
    TriangleBounds(a, b, c, lower, upper);
    degenerate = viskores::TriangleIsDegenerate(a, b, c) ? 1 : 0;
  }
};

/// For every nondegenerate output triangle: whether it meets another nondegenerate one
/// improperly, and how many such partners have a larger index, so that every pair is counted
/// once. Degenerate triangles are flagged on their own and take part in no pair.
struct FindImproperPairs : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn triangle,
                                FieldIn lower,
                                FieldIn upper,
                                FieldIn degenerate,
                                WholeArrayIn triangles,
                                WholeArrayIn degenerates,
                                WholeArrayIn generators,
                                ExecObject hierarchy,
                                FieldOut involved,
                                FieldOut pairsAbove);
  using ExecutionSignature = void(WorkIndex, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10);

  template <typename TrianglesPortal,
            typename FlagPortal,
            typename GeneratorsPortal,
            typename Hierarchy>
  VISKORES_EXEC void operator()(viskores::Id i,
                                const viskores::Id3& triangle,
                                const Vec3& lower,
                                const Vec3& upper,
                                viskores::UInt8 degenerate,
                                const TrianglesPortal& triangles,
                                const FlagPortal& degenerates,
                                const GeneratorsPortal& generators,
                                const Hierarchy& hierarchy,
                                viskores::UInt8& involved,
                                viskores::Id& pairsAbove) const
  {
    involved = degenerate;
    pairsAbove = 0;
    if (degenerate)
    {
      return;
    }
    const Vec3 p[3] = { generators.Get(triangle[0]),
                        generators.Get(triangle[1]),
                        generators.Get(triangle[2]) };
    const bool complete = hierarchy.ForEachOverlap(
      lower,
      upper,
      [&](viskores::Id j)
      {
        if (j == i || degenerates.Get(j))
        {
          return true;
        }
        const viskores::Id3 other = triangles.Get(j);
        const Vec3 q[3] = { generators.Get(other[0]),
                            generators.Get(other[1]),
                            generators.Get(other[2]) };
        if (exact::ImproperPair(p, triangle, q, other))
        {
          involved = 1;
          if (j > i)
          {
            ++pairsAbove;
          }
        }
        return true;
      });
    if (!complete)
    {
      this->RaiseError("QuadricClusteringDecimation: bounding volume hierarchy too deep.");
    }
  }
};

struct InvolvedLabels : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn triangle, FieldIn involved, FieldOut labels);
  using ExecutionSignature = void(_1, _2, _3);

  template <typename LabelsVec>
  VISKORES_EXEC void operator()(const viskores::Id3& triangle,
                                viskores::UInt8 involved,
                                LabelsVec& labels) const
  {
    for (int c = 0; c < 3; ++c)
    {
      labels[c] = involved ? triangle[c] : viskores::Id(-1);
    }
  }
};

} // namespace detail

/// The embedding flag (F4) of Algorithm 6: realises the output triangles at the generators,
/// and flags, among the clusters with at least two vertices, those of every degenerate triangle
/// and of every pair of triangles that meet other than in their common vertex or edge. Every
/// predicate is evaluated exactly.
inline EmbeddingReport EmbeddingFlags(const viskores::cont::ArrayHandle<viskores::Id3>& triangles,
                                      const viskores::cont::ArrayHandle<Vec3>& generators,
                                      const viskores::cont::ArrayHandle<viskores::Id>& sizes,
                                      viskores::cont::ArrayHandle<viskores::UInt8>& flags)
{
  using Algorithm = viskores::cont::Algorithm;
  viskores::cont::Invoker invoke;
  EmbeddingReport report;
  if (triangles.GetNumberOfValues() == 0)
  {
    return report;
  }
  viskores::cont::ArrayHandle<Vec3> lower, upper;
  viskores::cont::ArrayHandle<viskores::UInt8> degenerate;
  invoke(detail::RealizeTriangle{}, triangles, generators, lower, upper, degenerate);
  const viskores::worklet::spatialstructure::BoundingVolumeHierarchy bvh(lower, upper);
  viskores::cont::ArrayHandle<viskores::UInt8> involved;
  viskores::cont::ArrayHandle<viskores::Id> pairsAbove;
  invoke(detail::FindImproperPairs{},
         triangles,
         lower,
         upper,
         degenerate,
         triangles,
         degenerate,
         generators,
         bvh,
         involved,
         pairsAbove);
  report.NumberOfDegenerateTriangles = Algorithm::Reduce(degenerate, viskores::Id(0));
  report.NumberOfImproperPairs = Algorithm::Reduce(pairsAbove, viskores::Id(0));
  if (report.NumberOfDegenerateTriangles + report.NumberOfImproperPairs == 0)
  {
    return report;
  }
  viskores::cont::ArrayHandle<viskores::Id> labels, listed;
  invoke(detail::InvolvedLabels{},
         triangles,
         involved,
         viskores::cont::make_ArrayHandleGroupVec<3>(labels));
  viskores::cont::ArrayHandle<viskores::UInt8> valid;
  invoke(IsValidLabel{}, labels, valid);
  Algorithm::CopyIf(labels, valid, listed);
  FlagClusters(listed, sizes, flags);
  return report;
}

}
}
} // namespace viskores::worklet::quadric_clustering

VISKORES_QUADRIC_CLUSTERING_STRICT_FP_END

#endif // viskores_worklet_quadric_clustering_EmbeddingFlags_h
