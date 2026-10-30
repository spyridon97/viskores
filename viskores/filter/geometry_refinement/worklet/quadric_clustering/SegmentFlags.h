//============================================================================
//  The contents of this file are covered by the Viskores license. See
//  LICENSE.txt for details.
//
//  By contributing to this file, all contributors agree to the Developer
//  Certificate of Origin Version 1.1 (DCO 1.1) as stated in DCO.txt.
//============================================================================
#ifndef viskores_worklet_quadric_clustering_SegmentFlags_h
#define viskores_worklet_quadric_clustering_SegmentFlags_h

#include <viskores/filter/geometry_refinement/worklet/quadric_clustering/CurveFlags.h>
#include <viskores/filter/geometry_refinement/worklet/quadric_clustering/GlueItems.h>
#include <viskores/filter/geometry_refinement/worklet/quadric_clustering/MergePhase.h>
#include <viskores/filter/geometry_refinement/worklet/quadric_clustering/QuadricClusteringCommonWorklets.h>
#include <viskores/filter/geometry_refinement/worklet/quadric_clustering/SingularCurves.h>

#include <viskores/BinaryOperators.h>
#include <viskores/cont/Algorithm.h>
#include <viskores/cont/ArrayHandle.h>
#include <viskores/cont/ArrayHandleCast.h>
#include <viskores/cont/ArrayHandleConcatenate.h>
#include <viskores/cont/ArrayHandleConstant.h>
#include <viskores/cont/ArrayHandleGroupVec.h>
#include <viskores/cont/ArrayHandleIndex.h>
#include <viskores/cont/ArrayHandleZip.h>
#include <viskores/worklet/WorkletMapField.h>

VISKORES_QUADRIC_CLUSTERING_STRICT_FP_BEGIN

namespace viskores
{
namespace worklet
{
namespace quadric_clustering
{

namespace segment_flags_detail
{

//---------------------------------------------------------------------------------------------
// (D0), (D1) and (D2) on the clusters the candidates change.

/// For a vertex of a changed cluster: the edges inside the cluster from it to a larger vertex,
/// the triangles inside whose smallest vertex it is, whether it is on the boundary, the
/// boundary edges inside from it to a larger vertex, and the cut edges from it.
struct D0Counts : viskores::worklet::WorkletMapField
{
  using ControlSignature =
    void(FieldIn key, FieldIn vertex, ExecObject mesh, ExecObject region, FieldOut counts);
  using ExecutionSignature = _5(_1, _2, _3, _4);

  template <typename Mesh, typename Region>
  VISKORES_EXEC viskores::Vec<viskores::Id, 5> operator()(const viskores::Id2& key,
                                                          viskores::Id v,
                                                          const Mesh& mesh,
                                                          const Region& region) const
  {
    const viskores::Id c = key[0], label = key[1];
    viskores::Vec<viskores::Id, 5> counts(0);
    for (viskores::Id r = mesh.VertexEdgeOffsets.Get(v); r < mesh.VertexEdgeOffsets.Get(v + 1); ++r)
    {
      const viskores::Id e = mesh.VertexEdges.Get(r);
      const viskores::Id w = mesh.OtherEnd(e, v);
      const bool inside = region.VertexLabel(c, w) == label;
      if (w > v && inside)
      {
        ++counts[0];
        counts[3] += (mesh.EdgeTriangles.Get(e)[1] < 0) ? 1 : 0;
      }
      counts[4] += inside ? 0 : 1;
    }
    for (viskores::Id r = mesh.VertexCornerOffsets.Get(v); r < mesh.VertexCornerOffsets.Get(v + 1);
         ++r)
    {
      const viskores::Id3 tri = mesh.Triangles.Get(mesh.CornerTriangle(r));
      if (viskores::Min(viskores::Min(tri[0], tri[1]), tri[2]) == v &&
          region.VertexLabel(c, tri[0]) == label && region.VertexLabel(c, tri[1]) == label &&
          region.VertexLabel(c, tri[2]) == label)
      {
        ++counts[1];
      }
    }
    counts[2] = mesh.IsBoundaryVertex.Get(v) ? 1 : 0;
    return counts;
  }
};

using region_detail::Fire;

/// (D0) on a changed cluster, as `detail::D0`.
struct D0Fires : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn key, FieldIn counts, FieldIn size, WholeArrayInOut fires);
  using ExecutionSignature = void(_1, _2, _3, _4);

  template <typename FirePortal>
  VISKORES_EXEC void operator()(const viskores::Id2& key,
                                const viskores::Vec<viskores::Id, 5>& counts,
                                viskores::Id size,
                                FirePortal& fires) const
  {
    if (detail::D0Violates(size - counts[0] + counts[1], counts[2], counts[3], counts[4]))
    {
      Fire(fires, key[0]);
    }
  }
};

/// The number of record slots of a vertex: one per incident edge and three per incident
/// triangle for (D1), one per incident triangle for (D2).
struct TopologySlots : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn vertex,
                                ExecObject mesh,
                                FieldOut pairSlots,
                                FieldOut tripleSlots);
  using ExecutionSignature = void(_1, _2, _3, _4);

  template <typename Mesh>
  VISKORES_EXEC void operator()(viskores::Id v,
                                const Mesh& mesh,
                                viskores::Id& pairSlots,
                                viskores::Id& tripleSlots) const
  {
    const viskores::Id edges = mesh.VertexEdgeOffsets.Get(v + 1) - mesh.VertexEdgeOffsets.Get(v);
    tripleSlots = mesh.VertexCornerOffsets.Get(v + 1) - mesh.VertexCornerOffsets.Get(v);
    pairSlots = edges + 3 * tripleSlots;
  }
};

/// The (D1) records `((candidate, a, b), (cut, d, e))` and the (D2) triples of a vertex of a
/// changed cluster: its cut edges to other clusters, each once, and the 3-label triangles whose
/// smallest vertex in a changed cluster it is, with their pairs at a changed cluster.
struct TopologyRecords : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn key,
                                FieldIn vertex,
                                FieldIn pairOffset,
                                FieldIn tripleOffset,
                                ExecObject mesh,
                                ExecObject region,
                                WholeArrayOut pairKeys,
                                WholeArrayOut pairRecords,
                                WholeArrayOut triples);
  using ExecutionSignature = void(_1, _2, _3, _4, _5, _6, _7, _8, _9);

  template <typename Mesh,
            typename Region,
            typename PairKeyPortal,
            typename RecordPortal,
            typename TriplePortal>
  VISKORES_EXEC void operator()(const viskores::Id2& key,
                                viskores::Id v,
                                viskores::Id pairOffset,
                                viskores::Id tripleOffset,
                                const Mesh& mesh,
                                const Region& region,
                                PairKeyPortal& pairKeys,
                                RecordPortal& pairRecords,
                                TriplePortal& triples) const
  {
    const viskores::Id c = key[0], label = key[1];
    const viskores::Vec<viskores::Id, 3> none(-1);
    viskores::Id p = pairOffset, q = tripleOffset;
    for (viskores::Id r = mesh.VertexEdgeOffsets.Get(v); r < mesh.VertexEdgeOffsets.Get(v + 1); ++r)
    {
      const viskores::Id e = mesh.VertexEdges.Get(r);
      const viskores::Id w = mesh.OtherEnd(e, v);
      const viskores::Id other = region.VertexLabel(c, w);
      const bool once = !(region.IsTarget(c, other) && w < v);
      if (other != label && once)
      {
        pairKeys.Set(p,
                     viskores::Vec<viskores::Id, 3>(
                       c, viskores::Min(label, other), viskores::Max(label, other)));
        pairRecords.Set(p, viskores::Id3(1, 0, (mesh.EdgeTriangles.Get(e)[1] < 0) ? 1 : 0));
      }
      else
      {
        pairKeys.Set(p, none);
        pairRecords.Set(p, viskores::Id3(0));
      }
      ++p;
    }
    for (viskores::Id r = mesh.VertexCornerOffsets.Get(v); r < mesh.VertexCornerOffsets.Get(v + 1);
         ++r)
    {
      const viskores::Id3 tri = mesh.Triangles.Get(mesh.CornerTriangle(r));
      viskores::Id l[3];
      viskores::Id first = -1;
      for (int k = 0; k < 3; ++k)
      {
        l[k] = region.VertexLabel(c, tri[k]);
        if (region.IsTarget(c, l[k]) && (first < 0 || tri[k] < first))
        {
          first = tri[k];
        }
      }
      const bool three = AllDistinct(l[0], l[1], l[2]);
      const bool emit = three && first == v;
      const viskores::Id3 sorted = SortThree(l[0], l[1], l[2]);
      triples.Set(q++,
                  emit ? viskores::Vec<viskores::Id, 4>(c, sorted[0], sorted[1], sorted[2])
                       : viskores::Vec<viskores::Id, 4>(-1));
      for (int k = 0; k < 3; ++k)
      {
        const viskores::Id a = l[k], b = l[(k + 1) % 3];
        if (emit && (region.IsTarget(c, a) || region.IsTarget(c, b)))
        {
          pairKeys.Set(p,
                       viskores::Vec<viskores::Id, 3>(c, viskores::Min(a, b), viskores::Max(a, b)));
          pairRecords.Set(p, viskores::Id3(0, 1, 0));
        }
        else
        {
          pairKeys.Set(p, none);
          pairRecords.Set(p, viskores::Id3(0));
        }
        ++p;
      }
    }
  }
};

/// (D1) on a pair at a changed cluster, as `detail::D1`.
struct D1Fires : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn key, FieldIn sums, WholeArrayInOut fires);
  using ExecutionSignature = void(_1, _2, _3);

  template <typename FirePortal>
  VISKORES_EXEC void operator()(const viskores::Vec<viskores::Id, 3>& key,
                                const viskores::Id3& sums,
                                FirePortal& fires) const
  {
    if (sums[0] > 0 && (sums[1] < 1 || sums[1] + sums[2] != 2))
    {
      Fire(fires, key[0]);
    }
  }
};

/// The component of a source cluster of a candidate, from its anchor.
struct SourceComponent : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn relabel,
                                WholeArrayIn anchors,
                                WholeArrayIn components,
                                FieldOut key);
  using ExecutionSignature = _4(_1, _2, _3);

  template <typename Portal>
  VISKORES_EXEC viskores::Id2 operator()(const viskores::Id3& relabel,
                                         const Portal& anchors,
                                         const Portal& components) const
  {
    return viskores::Id2(relabel[0], components.Get(anchors.Get(relabel[1])));
  }
};

/// (D2) by count: a closed component left with exactly three clusters.
struct D2CountFires : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn key,
                                FieldIn sources,
                                WholeArrayIn componentClusters,
                                WholeArrayIn componentBoundary,
                                WholeArrayInOut fires);
  using ExecutionSignature = void(_1, _2, _3, _4, _5);

  template <typename Portal, typename FirePortal>
  VISKORES_EXEC void operator()(const viskores::Id2& key,
                                viskores::Id sources,
                                const Portal& componentClusters,
                                const Portal& componentBoundary,
                                FirePortal& fires) const
  {
    if (componentClusters.Get(key[1]) - sources == 3 && componentBoundary.Get(key[1]) == 0)
    {
      Fire(fires, key[0]);
    }
  }
};

//---------------------------------------------------------------------------------------------
// (F0)

struct RadiusFires : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn record,
                                ExecObject mesh,
                                ExecObject region,
                                WholeArrayInOut fires);
  using ExecutionSignature = void(_1, _2, _3, _4);

  VISKORES_CONT explicit RadiusFires(Real maxRadius)
    : MaxRadius(maxRadius)
  {
  }

  template <typename Mesh, typename Region, typename FirePortal>
  VISKORES_EXEC void operator()(const viskores::Vec<viskores::Id, 4>& record,
                                const Mesh& mesh,
                                const Region& region,
                                FirePortal& fires) const
  {
    const Vec3 g = region.Generator(record[0], record[3]);
    if (LengthUp(Norm(Sub(g, mesh.Points.Get(record[1])))) > this->MaxRadius)
    {
      Fire(fires, record[0]);
    }
  }

  Real MaxRadius;
};

/// The largest absolute coordinate of a new generator of a label that stays.
struct GroupGeneratorScale : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn record,
                                FieldIn generator,
                                FieldOut candidate,
                                FieldOut value);
  using ExecutionSignature = void(_1, _2, _3, _4);

  VISKORES_EXEC void operator()(const viskores::Id3& record,
                                const Vec3& generator,
                                viskores::Id& candidate,
                                Real& value) const
  {
    candidate = record[0];
    value = (record[1] == record[2]) ? MaxAbs(generator) : Real(0);
  }
};

/// The largest absolute coordinate of the generators of a candidate's labelling: of the labels
/// it leaves, by a walk down their values in decreasing order, and of those it changes.
struct CandidateGeneratorScale : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn candidate,
                                FieldIn changedGenerators,
                                WholeArrayIn generatorOrder,
                                ExecObject region,
                                FieldOut generatorScale);
  using ExecutionSignature = _5(_1, _2, _3, _4);

  template <typename OrderPortal, typename Region>
  VISKORES_EXEC Real operator()(viskores::Id c,
                                Real changedGenerators,
                                const OrderPortal& generatorOrder,
                                const Region& region) const
  {
    Real unchangedGenerators = 0;
    for (viskores::Id k = 0; k < generatorOrder.GetNumberOfValues(); ++k)
    {
      const auto entry = generatorOrder.Get(k);
      if (!region.Affected(c, entry.second))
      {
        unchangedGenerators = entry.first;
        break;
      }
    }
    return viskores::Max(unchangedGenerators, changedGenerators);
  }
};

struct MaxWith : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn value, FieldOut result);
  using ExecutionSignature = _2(_1);

  VISKORES_CONT explicit MaxWith(Real floor)
    : Floor(floor)
  {
  }

  VISKORES_EXEC Real operator()(Real value) const { return viskores::Max(value, this->Floor); }

  Real Floor;
};

/// The new point of the segment a candidate moves in its output polyline: the generator of its
/// head's first copy on the candidate's labelling.
struct MovedPoint : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn candidate,
                                FieldIn moved,
                                WholeArrayIn arcVertices,
                                WholeArrayIn copyOffsets,
                                ExecObject region,
                                FieldOut point);
  using ExecutionSignature = _6(_1, _2, _3, _4, _5);

  template <typename IdPortal, typename Region>
  VISKORES_EXEC Vec3 operator()(viskores::Id c,
                                viskores::Id moved,
                                const IdPortal& arcVertices,
                                const IdPortal& copyOffsets,
                                const Region& region) const
  {
    if (moved < 0)
    {
      return Vec3(0);
    }
    const viskores::Id label = region.Labels.Get(copyOffsets.Get(arcVertices.Get(moved)));
    return region.Generator(c, region.NewLabel(c, label));
  }
};

//---------------------------------------------------------------------------------------------
// (F4), as `NewImproper`: a triangle at the merged group is never inherited, so the candidate
// fires if one of them is degenerate or meets another nondegenerate triangle improperly.

struct GroupTriangleImproper : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn index,
                                WholeArrayIn candidates,
                                WholeArrayIn triangles,
                                WholeArrayIn offsets,
                                WholeArrayIn baseTriangles,
                                WholeArrayIn baseDegenerate,
                                ExecObject hierarchy,
                                ExecObject region,
                                WholeArrayInOut fires);
  using ExecutionSignature = void(_1, _2, _3, _4, _5, _6, _7, _8, _9);

  template <typename IdPortal,
            typename Id3Portal,
            typename FlagPortal,
            typename Hierarchy,
            typename Region,
            typename FirePortal>
  VISKORES_EXEC void operator()(viskores::Id i,
                                const IdPortal& candidates,
                                const Id3Portal& triangles,
                                const IdPortal& offsets,
                                const Id3Portal& baseTriangles,
                                const FlagPortal& baseDegenerate,
                                const Hierarchy& hierarchy,
                                const Region& region,
                                FirePortal& fires) const
  {
    const viskores::Id c = candidates.Get(i);
    auto realize = [&](viskores::Id j, Vec3* p, viskores::Id3& glued)
    {
      const viskores::Id3 t = triangles.Get(j);
      for (int k = 0; k < 3; ++k)
      {
        p[k] = region.Generator(c, t[k]);
        glued[k] = region.GlueOf(c, t[k]);
      }
    };
    Vec3 p[3];
    viskores::Id3 glued;
    realize(i, p, glued);
    if (viskores::TriangleIsDegenerate(p[0], p[1], p[2]))
    {
      Fire(fires, c);
      return;
    }
    // Against the other triangles at the merged group.
    for (viskores::Id j = offsets.Get(c); j < offsets.Get(c + 1); ++j)
    {
      if (j == i)
      {
        continue;
      }
      Vec3 q[3];
      viskores::Id3 other;
      realize(j, q, other);
      if (!viskores::TriangleIsDegenerate(q[0], q[1], q[2]) &&
          exact::ImproperPair(p, glued, q, other))
      {
        Fire(fires, c);
        return;
      }
    }
    // Against the glued triangles of the round away from the two groups, which stay.
    const viskores::Id2 groups = region.GroupGlue.Get(c);
    Vec3 lo, hi;
    TriangleBounds(p[0], p[1], p[2], lo, hi);
    bool bad = false;
    const bool complete =
      hierarchy.ForEachOverlap(lo,
                               hi,
                               [&](viskores::Id j)
                               {
                                 const viskores::Id3 other = baseTriangles.Get(j);
                                 if (baseDegenerate.Get(j))
                                 {
                                   return true;
                                 }
                                 for (int k = 0; k < 3; ++k)
                                 {
                                   if (other[k] == groups[0] || other[k] == groups[1])
                                   {
                                     return true;
                                   }
                                 }
                                 const Vec3 q[3] = { region.Generators.Get(other[0]),
                                                     region.Generators.Get(other[1]),
                                                     region.Generators.Get(other[2]) };
                                 if (exact::ImproperPair(p, glued, q, other))
                                 {
                                   bad = true;
                                   return false;
                                 }
                                 return true;
                               });
    if (!complete)
    {
      this->RaiseError("QuadricClusteringDecimation: bounding volume hierarchy too deep.");
    }
    if (bad)
    {
      Fire(fires, c);
    }
  }
};

/// The corners of the glued output triangles, one vertex per corner.
struct GluedCorners : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn triangle, FieldOut corners);
  using ExecutionSignature = void(_1, _2);

  template <typename CornerVec>
  VISKORES_EXEC void operator()(const viskores::Id3& t, CornerVec& corners) const
  {
    for (int a = 0; a < 3; ++a)
    {
      corners[a] = t[a];
    }
  }
};

/// (F7) on a candidate's labelling, given that no 3-label triangle fails on the labelling of
/// the round, as `ArcThickVQC.iso_pass`: a 3-label triangle at neither glue group keeps its
/// output triangle, and fails only through its component or a probe. The component of the
/// merged group keeps its first vertex unless that is the removed group, and that first vertex
/// moves only if it is the merged group. A probe's crossings change by those of the output
/// triangles at the two groups and at the merged group; a failing probe fails a triangle that
/// meets it, which exists if it has a crossing, or one with a vertex of its triangle's labels.
/// Sides are tested by `IncidentThickeningFires`.
struct CandidateThickening : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn candidate,
                                ExecObject region,
                                ExecObject round,
                                WholeArrayIn outputTriangles,
                                WholeArrayIn gluedTriangles,
                                WholeArrayIn fanOffsets,
                                WholeArrayIn fans,
                                WholeArrayIn groupOffsets,
                                WholeArrayIn groupTriangles,
                                WholeArrayIn inputTriangles,
                                WholeArrayIn appear,
                                WholeArrayInOut fires);
  using ExecutionSignature = void(_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12);

  template <typename Region,
            typename Round,
            typename Id3Portal,
            typename IdPortal,
            typename FirePortal>
  VISKORES_EXEC void operator()(viskores::Id c,
                                const Region& region,
                                const Round& round,
                                const Id3Portal& outputTriangles,
                                const Id3Portal& gluedTriangles,
                                const IdPortal& fanOffsets,
                                const IdPortal& fans,
                                const IdPortal& groupOffsets,
                                const Id3Portal& groupTriangles,
                                const Id3Portal& inputTriangles,
                                const IdPortal& appear,
                                FirePortal& fires) const
  {
    // The glue group that stays (the target) and the one that joins it.
    const viskores::Id2 groups = region.GroupGlue.Get(c);
    const viskores::Id kept = groups[0], removed = groups[1];
    const viskores::Id newBegin = groupOffsets.Get(c), newEnd = groupOffsets.Get(c + 1);
    const bool hasNew = newEnd > newBegin;
    bool complete = true, fits = true;
    bool fire = false;

    // The component: its first vertex without the removed group.
    viskores::Id first = -1;
    bool firstInside = true;
    for (int which = 0; which < 2; ++which)
    {
      const viskores::Id root = round.Root.Get(which == 0 ? kept : removed);
      if (root < 0)
      {
        continue;
      }
      viskores::Id f = root;
      bool inside = round.InsideFirst.Get(root) != 0;
      if (f == removed)
      {
        f = round.Second.Get(root);
        inside = round.InsideSecond.Get(root) != 0;
      }
      if (f >= 0 && (first < 0 || f < first))
      {
        first = f;
        firstInside = inside;
      }
    }
    if (first >= 0 && hasNew)
    {
      const bool inside = (first == kept)
        ? round.Thick.Inside(region.Generator(c, kept), complete, fits)
        : firstInside;
      fire = !inside;
    }

    // The output triangles of the round at the two groups, each once: from its smallest label
    // in them.
    auto forEachRemoved = [&](auto&& visit)
    {
      for (viskores::Id r = region.LabelOffsets.Get(c); r < region.LabelOffsets.Get(c + 1); ++r)
      {
        const viskores::Id x = region.LabelRecords.Get(r)[1];
        for (viskores::Id k = fanOffsets.Get(x); k < fanOffsets.Get(x + 1); ++k)
        {
          const viskores::Id row = fans.Get(k);
          const viskores::Id3 tri = outputTriangles.Get(row);
          bool firstHere = true;
          for (int a = 0; a < 3; ++a)
          {
            if (tri[a] < x && region.Affected(c, tri[a]))
            {
              firstHere = false;
            }
          }
          if (firstHere)
          {
            visit(row, tri);
          }
        }
      }
    };
    // The glued vertex of a label of the candidate's labelling, -1 for the merged group.
    auto gluedNew = [&](viskores::Id label)
    { return region.Affected(c, label) ? viskores::Id(-1) : region.Glue.Get(label); };

    const viskores::Id numProbes = round.ProbeCrossings.GetNumberOfValues();
    for (viskores::Id q = 0; q < numProbes && !fire; ++q)
    {
      viskores::Id crossings = round.ProbeCrossings.Get(q);
      viskores::Id nongeneric = round.ProbeNongeneric.Get(q);
      forEachRemoved(
        [&](viskores::Id, const viskores::Id3& tri)
        {
          const Vec3 p[3] = { region.Generators.Get(tri[0]),
                              region.Generators.Get(tri[1]),
                              region.Generators.Get(tri[2]) };
          const thickening_detail::SegmentHit hit = round.Thick.ProbeHit(q, p, fits);
          if (hit.Hit)
          {
            --crossings;
            nongeneric -= hit.Generic ? 0 : 1;
          }
        });
      for (viskores::Id g = newBegin; g < newEnd; ++g)
      {
        const viskores::Id3 tri = groupTriangles.Get(g);
        const Vec3 p[3] = { region.Generator(c, tri[0]),
                            region.Generator(c, tri[1]),
                            region.Generator(c, tri[2]) };
        const thickening_detail::SegmentHit hit = round.Thick.ProbeHit(q, p, fits);
        if (hit.Hit)
        {
          ++crossings;
          nongeneric += hit.Generic ? 0 : 1;
        }
      }
      if (nongeneric == 0 && crossings % 2 == 1)
      {
        continue;
      }
      if (crossings > 0)
      {
        fire = true;
        break;
      }
      // A 3-label triangle with a vertex at a label of the probe's triangle.
      const viskores::Id3 f = inputTriangles.Get(round.Thick.ProbeTriangles.Get(q));
      for (int a = 0; a < 3 && !fire; ++a)
      {
        const viskores::Id y = region.Labels.Get(f[a]);
        const viskores::Id x = gluedNew(y);
        if (x < 0)
        {
          fire = hasNew;
          continue;
        }
        viskores::Id count = appear.Get(x);
        forEachRemoved(
          [&](viskores::Id row, const viskores::Id3&)
          {
            const viskores::Id3 gt = gluedTriangles.Get(row);
            count -= (gt[0] == x || gt[1] == x || gt[2] == x) ? 1 : 0;
          });
        for (viskores::Id g = newBegin; g < newEnd; ++g)
        {
          const viskores::Id3 tri = groupTriangles.Get(g);
          count +=
            (gluedNew(tri[0]) == x || gluedNew(tri[1]) == x || gluedNew(tri[2]) == x) ? 1 : 0;
        }
        fire = count > 0;
      }
    }
    if (!complete || !fits)
    {
      this->RaiseError("QuadricClusteringDecimation: exact evaluation out of range.");
    }
    if (fire)
    {
      Fire(fires, c);
    }
  }
};

/// (F7) on the input triangles at the two glue groups of a candidate: whether one is a 3-label
/// triangle of its labelling whose output triangle meets a side of the thickening; these are the
/// only triangles whose output triangles change.
struct IncidentThickeningFires : viskores::worklet::WorkletMapField
{
  using ControlSignature =
    void(FieldIn item, ExecObject mesh, ExecObject region, ExecObject round, WholeArrayInOut fires);
  using ExecutionSignature = void(_1, _2, _3, _4, _5);

  template <typename Mesh, typename Region, typename Round, typename FirePortal>
  VISKORES_EXEC void operator()(const viskores::Id2& item,
                                const Mesh& mesh,
                                const Region& region,
                                const Round& round,
                                FirePortal& fires) const
  {
    const viskores::Id c = item[0];
    const viskores::Id3 tri = mesh.Triangles.Get(item[1]);
    viskores::Id l[3];
    for (int a = 0; a < 3; ++a)
    {
      l[a] = region.VertexLabel(c, tri[a]);
    }
    if (!AllDistinct(l[0], l[1], l[2]))
    {
      return;
    }
    const Vec3 p[3] = { region.Generator(c, l[0]),
                        region.Generator(c, l[1]),
                        region.Generator(c, l[2]) };
    bool complete = true, fits = true;
    const bool meets = round.Thick.MeetsSides(p, complete, fits);
    if (!complete || !fits)
    {
      this->RaiseError("QuadricClusteringDecimation: exact evaluation out of range.");
    }
    if (meets)
    {
      Fire(fires, c);
    }
  }
};

//---------------------------------------------------------------------------------------------
// (F5)

/// The per-triangle values of the scales of the round: (i) on 3-label triangles that need the
/// test, at the generators; (ii) on triangles that need it, at the vertices.
struct BaseScales : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn triangle,
                                WholeArrayIn points,
                                WholeArrayIn labels,
                                WholeArrayIn generators,
                                FieldOut needsTest,
                                FieldOut input,
                                FieldOut output);
  using ExecutionSignature = void(_1, _2, _3, _4, _5, _6, _7);

  VISKORES_CONT explicit BaseScales(Real epsilon)
    : Epsilon(epsilon)
  {
  }

  template <typename VecPortal, typename IdPortal>
  VISKORES_EXEC void operator()(const viskores::Id3& tri,
                                const VecPortal& points,
                                const IdPortal& labels,
                                const VecPortal& generators,
                                viskores::UInt8& needsTest,
                                Real& input,
                                Real& output) const
  {
    Real displacement = 0;
    viskores::Id l[3];
    for (int k = 0; k < 3; ++k)
    {
      l[k] = labels.Get(tri[k]);
      displacement =
        viskores::Max(displacement, Norm(Sub(generators.Get(l[k]), points.Get(tri[k]))));
    }
    needsTest = DisplacementExceeds(displacement, this->Epsilon) ? 1 : 0;
    input = 0;
    output = 0;
    if (needsTest)
    {
      if (AllDistinct(l[0], l[1], l[2]))
      {
        input =
          viskores::Max(viskores::Max(MaxAbs(generators.Get(l[0])), MaxAbs(generators.Get(l[1]))),
                        MaxAbs(generators.Get(l[2])));
      }
      output = viskores::Max(viskores::Max(MaxAbs(points.Get(tri[0])), MaxAbs(points.Get(tri[1]))),
                             MaxAbs(points.Get(tri[2])));
    }
  }

  Real Epsilon;
};

/// The values of the scales of `EnvelopeFlags` on the triangles at the two glue groups, on the
/// candidate's labelling: for (i), the largest absolute coordinate of the generators of a
/// 3-label triangle that needs the test; for (ii), of the vertices of a triangle that needs it.
struct IncidentScales : viskores::worklet::WorkletMapField
{
  using ControlSignature =
    void(FieldIn item, ExecObject mesh, ExecObject region, FieldOut input, FieldOut output);
  using ExecutionSignature = void(_1, _2, _3, _4, _5);

  VISKORES_CONT explicit IncidentScales(Real epsilon)
    : Epsilon(epsilon)
  {
  }

  template <typename Mesh, typename Region>
  VISKORES_EXEC void operator()(const viskores::Id2& item,
                                const Mesh& mesh,
                                const Region& region,
                                Real& input,
                                Real& output) const
  {
    const viskores::Id c = item[0];
    const viskores::Id3 tri = mesh.Triangles.Get(item[1]);
    viskores::Id l[3];
    Vec3 g[3];
    Real displacement = 0;
    for (int k = 0; k < 3; ++k)
    {
      l[k] = region.VertexLabel(c, tri[k]);
      g[k] = region.Generator(c, l[k]);
      displacement = viskores::Max(displacement, Norm(Sub(g[k], mesh.Points.Get(tri[k]))));
    }
    input = 0;
    output = 0;
    if (DisplacementExceeds(displacement, this->Epsilon))
    {
      if (AllDistinct(l[0], l[1], l[2]))
      {
        input = viskores::Max(viskores::Max(MaxAbs(g[0]), MaxAbs(g[1])), MaxAbs(g[2]));
      }
      output = viskores::Max(
        viskores::Max(MaxAbs(mesh.Points.Get(tri[0])), MaxAbs(mesh.Points.Get(tri[1]))),
        MaxAbs(mesh.Points.Get(tri[2])));
    }
  }

  Real Epsilon;
};

/// The scales of every candidate: the largest value of the round's triangles it leaves, by a
/// walk down their values in decreasing order, with the values it changes, and for (ii) the
/// scale of its generators (`CandidateGeneratorScale`).
struct CandidateScales : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn candidate,
                                FieldIn changedInput,
                                FieldIn changedOutput,
                                FieldIn generatorScale,
                                WholeArrayIn inputOrder,
                                WholeArrayIn outputOrder,
                                ExecObject mesh,
                                ExecObject region,
                                FieldOut scaleInput,
                                FieldOut scaleOutput);
  using ExecutionSignature = void(_1, _2, _3, _4, _5, _6, _7, _8, _9, _10);

  VISKORES_CONT explicit CandidateScales(Real inputMaxAbs)
    : InputMaxAbs(inputMaxAbs)
  {
  }

  template <typename OrderPortal, typename Mesh, typename Region>
  VISKORES_EXEC void operator()(viskores::Id c,
                                Real changedInput,
                                Real changedOutput,
                                Real generatorScale,
                                const OrderPortal& inputOrder,
                                const OrderPortal& outputOrder,
                                const Mesh& mesh,
                                const Region& region,
                                Real& scaleInput,
                                Real& scaleOutput) const
  {
    auto walkTriangles = [&](const OrderPortal& order)
    {
      for (viskores::Id k = 0; k < order.GetNumberOfValues(); ++k)
      {
        const auto entry = order.Get(k);
        if (entry.first <= 0)
        {
          return Real(0);
        }
        if (!region.Incident(c, mesh, entry.second))
        {
          return entry.first;
        }
      }
      return Real(0);
    };
    scaleInput =
      viskores::Max(viskores::Max(walkTriangles(inputOrder), changedInput), this->InputMaxAbs);
    scaleOutput =
      viskores::Max(viskores::Max(walkTriangles(outputOrder), changedOutput), generatorScale);
  }

  Real InputMaxAbs;
};

/// Whether a candidate changes a scale of a side in use: then every triangle is tested again.
struct ScalesDiffer : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn scaleInput, FieldIn scaleOutput, FieldOut differ);
  using ExecutionSignature = _3(_1, _2);

  VISKORES_CONT ScalesDiffer(Real baseInput, Real baseOutput, viskores::UInt8 sides)
    : BaseInput(baseInput)
    , BaseOutput(baseOutput)
    , Sides(sides)
  {
  }

  VISKORES_EXEC viskores::UInt8 operator()(Real scaleInput, Real scaleOutput) const
  {
    const bool input = (this->Sides & 1) != 0 && scaleInput != this->BaseInput;
    const bool output = (this->Sides & 2) != 0 && scaleOutput != this->BaseOutput;
    return (input || output) ? 1 : 0;
  }

  Real BaseInput;
  Real BaseOutput;
  viskores::UInt8 Sides;
};

/// Whether a candidate changes a scale and has not fired yet.
struct RescaledNotFired : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn differ, FieldIn fires, FieldOut keep);
  using ExecutionSignature = _3(_1, _2);

  VISKORES_EXEC viskores::UInt8 operator()(viskores::UInt8 differ, viskores::UInt8 fires) const
  {
    return (differ && fires == 0) ? 1 : 0;
  }
};

/// The number of output triangles of the round at a label of a candidate's two groups.
struct RemovedSlots : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn record, WholeArrayIn fanOffsets, FieldOut slots);
  using ExecutionSignature = _3(_1, _2);

  template <typename IdPortal>
  VISKORES_EXEC viskores::Id operator()(const viskores::Id3& record,
                                        const IdPortal& fanOffsets) const
  {
    return fanOffsets.Get(record[1] + 1) - fanOffsets.Get(record[1]);
  }
};

/// The removed output triangles of a candidate: the triangles of the round at a label of its
/// two groups, `(candidate, triangle)`.
struct RemovedTriangles : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn record,
                                FieldIn offset,
                                WholeArrayIn testsAll,
                                WholeArrayIn fanOffsets,
                                WholeArrayIn fans,
                                WholeArrayOut removed);
  using ExecutionSignature = void(_1, _2, _3, _4, _5, _6);

  template <typename FlagPortal, typename IdPortal, typename OutPortal>
  VISKORES_EXEC void operator()(const viskores::Id3& record,
                                viskores::Id offset,
                                const FlagPortal& testsAll,
                                const IdPortal& fanOffsets,
                                const IdPortal& fans,
                                OutPortal& removed) const
  {
    viskores::Id q = offset;
    for (viskores::Id f = fanOffsets.Get(record[1]); f < fanOffsets.Get(record[1] + 1); ++f, ++q)
    {
      // A candidate that tests every triangle needs none near them.
      removed.Set(
        q, testsAll.Get(record[0]) ? viskores::Id2(-1) : viskores::Id2(record[0], fans.Get(f)));
    }
  }
};

/// The input triangles near a removed output triangle, whose test (ii) may use it: those whose
/// box meets its box grown by epsilon, with a margin, that need the test and are away from the
/// two groups. Counts them, or with `Fill`, lists them as `(candidate, t)`.
template <bool Fill>
struct NearRemoved : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn removed,
                                FieldIn offset,
                                WholeArrayIn outputTriangles,
                                WholeArrayIn needsTest,
                                ExecObject inputHierarchy,
                                ExecObject mesh,
                                ExecObject region,
                                FieldOut count,
                                WholeArrayOut near);
  using ExecutionSignature = void(_1, _2, _3, _4, _5, _6, _7, _8, _9);

  VISKORES_CONT explicit NearRemoved(Real grow)
    : Grow(grow)
  {
  }

  template <typename Id3Portal,
            typename FlagPortal,
            typename Hierarchy,
            typename Mesh,
            typename Region,
            typename NearPortal>
  VISKORES_EXEC void operator()(const viskores::Id2& removed,
                                viskores::Id offset,
                                const Id3Portal& outputTriangles,
                                const FlagPortal& needsTest,
                                const Hierarchy& hierarchy,
                                const Mesh& mesh,
                                const Region& region,
                                viskores::Id& count,
                                NearPortal& near) const
  {
    const viskores::Id c = removed[0];
    const viskores::Id3 tri = outputTriangles.Get(removed[1]);
    Vec3 lo(std::numeric_limits<Real>::infinity()), hi(-std::numeric_limits<Real>::infinity());
    for (int k = 0; k < 3; ++k)
    {
      const Vec3 p = region.Generators.Get(tri[k]);
      for (int a = 0; a < 3; ++a)
      {
        lo[a] = viskores::Min(lo[a], p[a] - this->Grow);
        hi[a] = viskores::Max(hi[a], p[a] + this->Grow);
      }
    }
    count = 0;
    const bool complete =
      hierarchy.ForEachOverlap(lo,
                               hi,
                               [&](viskores::Id t)
                               {
                                 if (needsTest.Get(t) && !region.Incident(c, mesh, t))
                                 {
                                   if (Fill)
                                   {
                                     near.Set(offset + count, viskores::Id2(c, t));
                                   }
                                   ++count;
                                 }
                                 return true;
                               });
    if (!complete)
    {
      this->RaiseError("QuadricClusteringDecimation: bounding volume hierarchy too deep.");
    }
  }

  Real Grow;
};

/// Appends `values` to `all`.
template <typename T>
inline void Append(viskores::cont::ArrayHandle<T>& all,
                   const viskores::cont::ArrayHandle<T>& values)
{
  viskores::cont::ArrayHandle<T> result;
  viskores::cont::Algorithm::Copy(viskores::cont::make_ArrayHandleConcatenate(all, values), result);
  all = result;
}

/// Whether an item belongs to a candidate that has not fired yet; the others need no test.
struct NotFired : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn item, WholeArrayIn fires, FieldOut keep);
  using ExecutionSignature = _3(_1, _2);

  template <typename FlagPortal>
  VISKORES_EXEC viskores::UInt8 operator()(const viskores::Id2& item, const FlagPortal& fires) const
  {
    return fires.Get(item[0]) == 0 ? 1 : 0;
  }
};

/// Every triangle of a candidate that tests them all.
struct AllTriangles : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn index, WholeArrayIn rescaled, FieldOut item);
  using ExecutionSignature = _3(_1, _2);

  VISKORES_CONT explicit AllTriangles(viskores::Id numTriangles)
    : NumberOfTriangles(numTriangles)
  {
  }

  template <typename IdPortal>
  VISKORES_EXEC viskores::Id2 operator()(viskores::Id i, const IdPortal& rescaled) const
  {
    return viskores::Id2(rescaled.Get(i / this->NumberOfTriangles), i % this->NumberOfTriangles);
  }

  viskores::Id NumberOfTriangles;
};

/// The distance to the output of a candidate: the output triangles of the round away from its
/// two groups, and its output triangles at the merged group.
template <typename OutputMesh, typename RegionType, typename Id3Portal, typename IdPortal>
struct CandidateOutputDistance
{
  VISKORES_EXEC bool Removed(viskores::Id t) const
  {
    const viskores::Id3 tri = this->Output.Triangles.Get(t);
    return this->Region.Affected(this->Candidate, tri[0]) ||
      this->Region.Affected(this->Candidate, tri[1]) ||
      this->Region.Affected(this->Candidate, tri[2]);
  }

  template <typename Visit>
  VISKORES_EXEC void ForEachNew(Visit&& visit) const
  {
    for (viskores::Id j = this->Offsets.Get(this->Candidate);
         j < this->Offsets.Get(this->Candidate + 1);
         ++j)
    {
      const viskores::Id3 t = this->Triangles.Get(j);
      const Vec3 q[3] = { this->Region.Generator(this->Candidate, t[0]),
                          this->Region.Generator(this->Candidate, t[1]),
                          this->Region.Generator(this->Candidate, t[2]) };
      if (!visit(q))
      {
        return;
      }
    }
  }

  VISKORES_EXEC Real DistanceUpTo(const Vec3& p,
                                  Real cutoff,
                                  bool& complete,
                                  const CentroidSeeds* seeds = nullptr) const
  {
    Real best = this->Output.DistanceSkipping(
      p, [&](viskores::Id t) { return this->Removed(t); }, cutoff, complete, seeds);
    this->ForEachNew(
      [&](const Vec3* q)
      {
        best = viskores::Min(best, PointTriangleDistanceUpper(p, q[0], q[1], q[2]));
        return true;
      });
    return best;
  }

  using Seeds = CentroidSeeds;

  VISKORES_EXEC bool WithinOneTriangle(const Vec3* s,
                                       Real bound,
                                       Real margin,
                                       bool& complete,
                                       CentroidSeeds& seeds) const
  {
    // The candidates among the output triangles of the candidate's labelling, ties broken by
    // their first input triangles, as their order there. The new triangles are offered after
    // the others, with or without seeds.
    const Vec3 q = TriangleCentroid(s[0], s[1], s[2]);
    CentroidCandidates candidates;
    this->Output.NearestCentroids(
      q, [&](viskores::Id t) { return this->Removed(t); }, candidates, complete, &seeds);
    for (viskores::Id j = this->Offsets.Get(this->Candidate);
         j < this->Offsets.Get(this->Candidate + 1);
         ++j)
    {
      const viskores::Id3 t = this->Triangles.Get(j);
      const Vec3 p[3] = { this->Region.Generator(this->Candidate, t[0]),
                          this->Region.Generator(this->Candidate, t[1]),
                          this->Region.Generator(this->Candidate, t[2]) };
      candidates.Offer(Norm(Sub(TriangleCentroid(p[0], p[1], p[2]), q)),
                       viskores::Id3(this->Sources.Get(j), 0, 0),
                       p[0],
                       p[1],
                       p[2]);
    }
    seeds.Keep(candidates);
    return candidates.Passes(s, bound, margin);
  }

  const OutputMesh& Output;
  const RegionType& Region;
  const Id3Portal& Triangles;
  const IdPortal& Offsets;
  /// The first input triangle of every new triangle, which breaks ties as the index of an
  /// output triangle of the candidate's labelling does.
  const IdPortal& Sources;
  viskores::Id Candidate;
};

/// The envelope tests of a triangle on the labelling of a candidate, which fires if the
/// triangle fails and one of its clusters has two or more vertices.
struct EnvelopeItemFires : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn item,
                                WholeArrayIn scaleInput,
                                WholeArrayIn scaleOutput,
                                WholeArrayIn generatorScale,
                                WholeArrayIn triangles,
                                WholeArrayIn offsets,
                                WholeArrayIn sources,
                                ExecObject inputDistance,
                                ExecObject outputDistance,
                                ExecObject mesh,
                                ExecObject region,
                                AtomicArrayInOut fires);
  using ExecutionSignature = void(_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12);

  VISKORES_CONT EnvelopeItemFires(Real epsilon, viskores::UInt8 sides)
    : Epsilon(epsilon)
    , Sides(sides)
  {
  }

  template <typename RealPortal,
            typename Id3Portal,
            typename IdPortal,
            typename InputMesh,
            typename OutputMesh,
            typename Mesh,
            typename Region,
            typename FirePortal>
  VISKORES_EXEC void operator()(const viskores::Id2& item,
                                const RealPortal& scaleInput,
                                const RealPortal& scaleOutput,
                                const RealPortal& generatorScale,
                                const Id3Portal& triangles,
                                const IdPortal& offsets,
                                const IdPortal& sources,
                                const InputMesh& inputDistance,
                                const OutputMesh& outputDistance,
                                const Mesh& mesh,
                                const Region& region,
                                const FirePortal& fires) const
  {
    const viskores::Id c = item[0];
    // The flag of a candidate only goes from 0 to 1, and the test only sets it: once another
    // triangle has fired it, this one cannot change it.
    if (fires.Get(c) != 0)
    {
      return;
    }
    const viskores::Id3 tri = mesh.Triangles.Get(item[1]);
    viskores::Id l[3];
    Vec3 g[3], v[3];
    Real displacement = 0;
    for (int k = 0; k < 3; ++k)
    {
      l[k] = region.VertexLabel(c, tri[k]);
      g[k] = region.Generator(c, l[k]);
      v[k] = mesh.Points.Get(tri[k]);
      displacement = viskores::Max(displacement, Norm(Sub(g[k], v[k])));
    }
    if (!DisplacementExceeds(displacement, this->Epsilon))
    {
      return;
    }
    const Real oneSixteen = 1 + 16 * UnitRoundoff;
    bool failed = false;
    viskores::Id pieces = 0;
    if ((this->Sides & 1) != 0 && AllDistinct(l[0], l[1], l[2]))
    {
      // (i): the realised triangle against the input.
      bool complete = true;
      const CertifiedWithin test{ this->Epsilon, scaleInput.Get(c) * oneSixteen };
      failed = !test(g[0], g[1], g[2], inputDistance, pieces, complete);
      if (!complete)
      {
        this->RaiseError("QuadricClusteringDecimation: bounding volume hierarchy too deep.");
      }
    }
    if (!failed && (this->Sides & 2) != 0)
    {
      // (ii): the input triangle against the candidate's output, whose largest coordinate is
      // that of its generators.
      auto output = outputDistance;
      output.MaxAbsCoordinate = generatorScale.Get(c);
      const CandidateOutputDistance<decltype(output), Region, Id3Portal, IdPortal> distance{
        output, region, triangles, offsets, sources, c
      };
      bool complete = true;
      const CertifiedWithin test{ this->Epsilon, scaleOutput.Get(c) * oneSixteen };
      failed = !test(v[0], v[1], v[2], distance, pieces, complete);
      if (!complete)
      {
        this->RaiseError("QuadricClusteringDecimation: bounding volume hierarchy too deep.");
      }
    }
    if (failed &&
        (region.Size(c, l[0]) > 1 || region.Size(c, l[1]) > 1 || region.Size(c, l[2]) > 1))
    {
      fires.Set(c, 1);
    }
  }

  Real Epsilon;
  viskores::UInt8 Sides;
};

/// The flag of a candidate from its atomic counterpart.
struct FiredFlag : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn fired, FieldOut fires);
  using ExecutionSignature = _2(_1);

  VISKORES_EXEC viskores::UInt8 operator()(viskores::UInt32 fired) const
  {
    return fired != 0 ? 1 : 0;
  }
};

} // namespace segment_flags_detail

/// Whether a flag of the full test fires on the labelling of every candidate of a round of
/// segment merges, as `FlagsFire` on that labelling decides, evaluated for all candidates at
/// once from what each changes, given that no flag fires on the labelling of the round: the
/// discrete closed-ball property on the clusters it changes and the pairs and triples at them,
/// (d') at its merged group, (F0) on its moved generators, (F4) on its output triangles at the
/// merged group, which are never inherited, (F5) on the triangles whose tests it can change, or
/// on every triangle if it changes a scale of the test, (Fc) on the output polyline of its arc,
/// and (F7) on its glued output.
inline viskores::cont::ArrayHandle<viskores::UInt8> CandidateFlags(
  const SurfaceMesh& mesh,
  const DistanceMesh& inputDistance,
  const MergeParameters& parameters,
  const SingularCurves& curves,
  const Clustering& clusters,
  const viskores::cont::ArrayHandle<viskores::Id>& glue,
  const CandidateRegions& regions,
  const CurveCandidates* curveCandidates = nullptr)
{
  using Algorithm = viskores::cont::Algorithm;
  namespace sf = segment_flags_detail;
  viskores::cont::Invoker invoke;
  const viskores::Id n = regions.NumberOfCandidates;
  const region_detail::MeshObject meshObject(mesh);
  const region_detail::RegionObject region(regions, clusters, glue);
  viskores::cont::ArrayHandle<viskores::UInt8> fires;
  fires.AllocateAndFill(n, 0);

  // (D0), (D1) and (D2) on the clusters the candidates change.
  if (regions.MemberVertices.GetNumberOfValues() > 0)
  {
    viskores::cont::ArrayHandle<viskores::Vec<viskores::Id, 5>> counts, sums;
    viskores::cont::ArrayHandle<viskores::Id2> keys;
    invoke(sf::D0Counts{}, regions.MemberKeys, regions.MemberVertices, meshObject, region, counts);
    Algorithm::ReduceByKey(regions.MemberKeys, counts, keys, sums, viskores::Add());
    invoke(sf::D0Fires{}, keys, sums, regions.MergedSizes, fires);

    viskores::cont::ArrayHandle<viskores::Id> pairSlots, tripleSlots, pairOffsets, tripleOffsets;
    invoke(sf::TopologySlots{}, regions.MemberVertices, meshObject, pairSlots, tripleSlots);
    const viskores::Id numPairs = Algorithm::ScanExclusive(pairSlots, pairOffsets);
    const viskores::Id numTriples = Algorithm::ScanExclusive(tripleSlots, tripleOffsets);
    viskores::cont::ArrayHandle<viskores::Vec<viskores::Id, 3>> pairKeys, validKeys, uniqueKeys;
    viskores::cont::ArrayHandle<viskores::Id3> pairRecords, validRecords, pairSums;
    viskores::cont::ArrayHandle<viskores::Vec<viskores::Id, 4>> triples, validTriples;
    pairKeys.Allocate(numPairs);
    pairRecords.Allocate(numPairs);
    triples.Allocate(numTriples);
    invoke(sf::TopologyRecords{},
           regions.MemberKeys,
           regions.MemberVertices,
           pairOffsets,
           tripleOffsets,
           meshObject,
           region,
           pairKeys,
           pairRecords,
           triples);
    viskores::cont::ArrayHandle<viskores::UInt8> valid;
    invoke(IsValidRecord{}, pairKeys, valid);
    Algorithm::CopyIf(pairKeys, valid, validKeys);
    Algorithm::CopyIf(pairRecords, valid, validRecords);
    if (validKeys.GetNumberOfValues() > 0)
    {
      Algorithm::SortByKey(validKeys, validRecords);
      Algorithm::ReduceByKey(validKeys, validRecords, uniqueKeys, pairSums, viskores::Add());
      invoke(sf::D1Fires{}, uniqueKeys, pairSums, fires);
    }
    if (parameters.D2ByCount)
    {
      // The clusters of every component in the round, less the sources of the candidate.
      viskores::cont::ArrayHandle<viskores::Id> clusterComponents, componentClusters, sourceCounts;
      clusterComponents.AllocateAndFill(clusters.NumberOfClusters, 0);
      invoke(detail::ClusterComponent{}, clusters.Labels, mesh.Components, clusterComponents);
      componentClusters.AllocateAndFill(mesh.NumberOfComponents, 0);
      invoke(detail::CountClusters{}, clusterComponents, componentClusters);
      viskores::cont::ArrayHandle<viskores::Id2> sourceKeys, uniqueSources;
      invoke(
        sf::SourceComponent{}, regions.Relabels, clusters.Anchors, mesh.Components, sourceKeys);
      Algorithm::Sort(sourceKeys);
      Algorithm::ReduceByKey(
        sourceKeys,
        viskores::cont::make_ArrayHandleConstant(viskores::Id(1), sourceKeys.GetNumberOfValues()),
        uniqueSources,
        sourceCounts,
        viskores::Add());
      invoke(sf::D2CountFires{},
             uniqueSources,
             sourceCounts,
             componentClusters,
             mesh.ComponentBoundaryVertices,
             fires);
    }
    else
    {
      invoke(IsValidRecord{}, triples, valid);
      Algorithm::CopyIf(triples, valid, validTriples);
      Algorithm::Sort(validTriples);
      invoke(region_detail::DuplicateFires{},
             viskores::cont::ArrayHandleIndex(validTriples.GetNumberOfValues()),
             validTriples,
             fires);
    }
  }

  // The input triangles at the two glue groups, and the output triangles at the merged group.
  viskores::cont::ArrayHandle<viskores::Id2> incident;
  viskores::cont::ArrayHandle<viskores::Id> groupCandidates, groupOffsets, groupSources;
  viskores::cont::ArrayHandle<viskores::Id3> groupTriangles;
  GroupTriangles(mesh,
                 clusters,
                 glue,
                 regions,
                 incident,
                 groupCandidates,
                 groupTriangles,
                 groupOffsets,
                 groupSources);
  if (parameters.Topological)
  {
    GroupGlueFires(mesh,
                   parameters.Original,
                   curves,
                   clusters,
                   glue,
                   regions,
                   groupCandidates,
                   groupTriangles,
                   fires);
  }

  // (F0)
  if (parameters.MaxClusterRadius < std::numeric_limits<Real>::infinity())
  {
    invoke(
      sf::RadiusFires(parameters.MaxClusterRadius), regions.Vertices, meshObject, region, fires);
  }

  // Values in decreasing order with their indices, for the walks of the candidates.
  using Entry = viskores::Pair<Real, viskores::Id>;
  auto order = [&](const viskores::cont::ArrayHandle<Real>& values)
  {
    viskores::cont::ArrayHandle<Entry> entries;
    Algorithm::Copy(viskores::cont::make_ArrayHandleZip(
                      values, viskores::cont::ArrayHandleIndex(values.GetNumberOfValues())),
                    entries);
    Algorithm::Sort(entries, viskores::SortGreater());
    return entries;
  };

  // The largest absolute coordinate of the generators of the round and of every candidate's
  // labelling, for (Fc) and (F5).
  const bool curveTest =
    parameters.CurveTolerance > 0 && curveCandidates && curves.NumberOfArcs > 0;
  const bool envelopeTest = parameters.ErrorBound > 0;
  viskores::cont::ArrayHandle<Real> generatorValues, generatorScale;
  if (curveTest || envelopeTest)
  {
    invoke(PointMaxAbs{}, clusters.Generators, generatorValues);
    const viskores::cont::ArrayHandle<Entry> generatorOrder = order(generatorValues);
    viskores::cont::ArrayHandle<Real> changedGenerators;
    changedGenerators.AllocateAndFill(n, 0);
    {
      viskores::cont::ArrayHandle<viskores::Id> owners, keys;
      viskores::cont::ArrayHandle<Real> values, maxValues;
      invoke(sf::GroupGeneratorScale{}, regions.Labels, regions.LabelGenerators, owners, values);
      Algorithm::ReduceByKey(owners, values, keys, maxValues, viskores::Maximum());
      invoke(ScatterValue{}, keys, maxValues, changedGenerators);
    }
    invoke(sf::CandidateGeneratorScale{},
           viskores::cont::ArrayHandleIndex(n),
           changedGenerators,
           generatorOrder,
           region,
           generatorScale);
  }

  // (Fc), on the output polyline of every candidate's arc.
  if (curveTest)
  {
    const viskores::cont::ArrayHandle<viskores::Id> copyOffsets =
      FirstCopies(parameters.Original, curves.Segment.GetNumberOfValues());
    const CurvePolylines polylines = BuildCurvePolylines(mesh, curves, copyOffsets, clusters);
    const CurveTestResult base =
      CurveTest(polylines, parameters.CurveTolerance, parameters.CurveDepth);
    // The scale of every candidate's labelling.
    viskores::cont::ArrayHandle<Real> scales;
    invoke(sf::MaxWith(MaxAbsCoordinate(mesh.Points)), generatorScale, scales);
    viskores::cont::ArrayHandle<Vec3> movedPoints;
    invoke(sf::MovedPoint{},
           viskores::cont::ArrayHandleIndex(n),
           curveCandidates->Moved,
           curves.ArcVertices,
           copyOffsets,
           region,
           movedPoints);
    CandidateCurveFires(polylines,
                        base,
                        *curveCandidates,
                        movedPoints,
                        scales,
                        parameters.CurveTolerance,
                        parameters.CurveDepth,
                        fires);
  }

  // The output triangles of the round, with their first input triangles, which order the output
  // triangles of a candidate's labelling.
  viskores::cont::ArrayHandle<viskores::Id> outputSources;
  const viskores::cont::ArrayHandle<viskores::Id3> outputTriangles =
    OutputTriangles(mesh, clusters.Labels, &outputSources);
  viskores::cont::ArrayHandle<viskores::Id3> sourceKeys;
  invoke(detail::IndexKey{}, outputSources, sourceKeys);

  // The glued output triangles of the round, for (F4) and (F7).
  const bool improperTest =
    parameters.Embedding && parameters.Topological && groupTriangles.GetNumberOfValues() > 0;
  const bool thickeningTest = parameters.CertifiedThickening && parameters.Topological;
  viskores::cont::ArrayHandle<viskores::Id3> glued;
  if (improperTest || thickeningTest)
  {
    glued = GlueTriangles(outputTriangles, glue);
  }
  // The output triangles of the round at every label, for (F7) and (ii) of (F5).
  const viskores::UInt8 sides = static_cast<viskores::UInt8>(parameters.Sides);
  viskores::cont::ArrayHandle<viskores::Id> fanOffsets, fans;
  if (thickeningTest || (envelopeTest && (sides & 2) != 0))
  {
    viskores::cont::ArrayHandle<viskores::Id2> cornerRecords;
    invoke(detail::CornerRecords{},
           outputTriangles,
           viskores::cont::make_ArrayHandleGroupVec<3>(cornerRecords));
    BuildAdjacency(cornerRecords, clusters.NumberOfClusters, fanOffsets, fans);
  }

  // (F4)
  if (improperTest)
  {
    viskores::cont::ArrayHandle<Vec3> lower, upper;
    viskores::cont::ArrayHandle<viskores::UInt8> degenerate;
    invoke(detail::RealizeTriangle{}, glued, clusters.Generators, lower, upper, degenerate);
    const viskores::worklet::spatialstructure::BoundingVolumeHierarchy hierarchy(lower, upper);
    invoke(sf::GroupTriangleImproper{},
           viskores::cont::ArrayHandleIndex(groupTriangles.GetNumberOfValues()),
           groupCandidates,
           groupTriangles,
           groupOffsets,
           glued,
           degenerate,
           hierarchy,
           region,
           fires);
  }

  // (F7), on the glued output of every candidate's labelling.
  if (thickeningTest)
  {
    const ThickeningRound round = PrepareThickeningRound(
      *parameters.CertifiedThickening, clusters.Generators, glued, clusters.NumberOfClusters);
    const ThickeningRoundObject roundObject(*parameters.CertifiedThickening, round);
    // The glued output triangles at every vertex of K.
    viskores::cont::ArrayHandle<viskores::Id> appear;
    appear.AllocateAndFill(clusters.NumberOfClusters, 0);
    if (glued.GetNumberOfValues() > 0)
    {
      viskores::cont::ArrayHandle<viskores::Id> corners, keys, counts;
      corners.Allocate(3 * glued.GetNumberOfValues());
      invoke(sf::GluedCorners{}, glued, viskores::cont::make_ArrayHandleGroupVec<3>(corners));
      Algorithm::Sort(corners);
      Algorithm::ReduceByKey(
        corners,
        viskores::cont::make_ArrayHandleConstant(viskores::Id(1), corners.GetNumberOfValues()),
        keys,
        counts,
        viskores::Add());
      invoke(ScatterValue{}, keys, counts, appear);
    }
    invoke(sf::CandidateThickening{},
           viskores::cont::ArrayHandleIndex(n),
           region,
           roundObject,
           outputTriangles,
           glued,
           fanOffsets,
           fans,
           groupOffsets,
           groupTriangles,
           mesh.Triangles,
           appear,
           fires);
    if (incident.GetNumberOfValues() > 0)
    {
      invoke(sf::IncidentThickeningFires{}, incident, meshObject, region, roundObject, fires);
    }
  }

  // (F5)
  if (envelopeTest)
  {
    const Real epsilon = parameters.ErrorBound;
    // The scales of the round.
    viskores::cont::ArrayHandle<viskores::UInt8> needsTest;
    viskores::cont::ArrayHandle<Real> inputValues, outputValues;
    invoke(sf::BaseScales(epsilon),
           mesh.Triangles,
           mesh.Points,
           clusters.Labels,
           clusters.Generators,
           needsTest,
           inputValues,
           outputValues);
    const Real generatorMax = Algorithm::Reduce(generatorValues, Real(0), viskores::Maximum());
    const Real baseInput = viskores::Max(
      Algorithm::Reduce(inputValues, Real(0), viskores::Maximum()), inputDistance.MaxAbsCoordinate);
    const Real baseOutput =
      viskores::Max(Algorithm::Reduce(outputValues, Real(0), viskores::Maximum()), generatorMax);
    const viskores::cont::ArrayHandle<Entry> inputOrder = order(inputValues);
    const viskores::cont::ArrayHandle<Entry> outputOrder = order(outputValues);

    // The values each candidate changes.
    viskores::cont::ArrayHandle<Real> changedInput, changedOutput;
    changedInput.AllocateAndFill(n, 0);
    changedOutput.AllocateAndFill(n, 0);
    if (incident.GetNumberOfValues() > 0)
    {
      viskores::cont::ArrayHandle<Real> input, output, maxInput, maxOutput;
      viskores::cont::ArrayHandle<viskores::Id> owners, keys;
      invoke(sf::IncidentScales(epsilon), incident, meshObject, region, input, output);
      invoke(FirstComponent{}, incident, owners);
      Algorithm::ReduceByKey(owners, input, keys, maxInput, viskores::Maximum());
      invoke(ScatterValue{}, keys, maxInput, changedInput);
      Algorithm::ReduceByKey(owners, output, keys, maxOutput, viskores::Maximum());
      invoke(ScatterValue{}, keys, maxOutput, changedOutput);
    }
    viskores::cont::ArrayHandle<Real> scaleInput, scaleOutput;
    invoke(sf::CandidateScales(inputDistance.MaxAbsCoordinate),
           viskores::cont::ArrayHandleIndex(n),
           changedInput,
           changedOutput,
           generatorScale,
           inputOrder,
           outputOrder,
           meshObject,
           region,
           scaleInput,
           scaleOutput);
    viskores::cont::ArrayHandle<viskores::UInt8> differ;
    invoke(sf::ScalesDiffer(baseInput, baseOutput, sides), scaleInput, scaleOutput, differ);
    // A candidate that has fired stays fired, whatever (F5) finds: its triangles are not tested.
    viskores::cont::ArrayHandle<viskores::UInt8> testRescaled;
    invoke(sf::RescaledNotFired{}, differ, fires, testRescaled);
    viskores::cont::ArrayHandle<viskores::Id> rescaled;
    Algorithm::CopyIf(viskores::cont::ArrayHandleIndex(n), testRescaled, rescaled);

    // The triangles to test: at the two groups; near the removed output triangles, for (ii);
    // all of them for a candidate that changes a scale.
    viskores::cont::ArrayHandle<viskores::Id2> items;
    Algorithm::Copy(incident, items);
    if ((sides & 2) != 0)
    {
      viskores::cont::ArrayHandle<viskores::Id> slots, offsets;
      invoke(sf::RemovedSlots{}, regions.Labels, fanOffsets, slots);
      const viskores::Id total = Algorithm::ScanExclusive(slots, offsets);
      viskores::cont::ArrayHandle<viskores::Id2> allRemoved, removed;
      allRemoved.Allocate(total);
      invoke(sf::RemovedTriangles{}, regions.Labels, offsets, differ, fanOffsets, fans, allRemoved);
      viskores::cont::ArrayHandle<viskores::UInt8> valid;
      invoke(IsValidRecord{}, allRemoved, valid);
      Algorithm::CopyIf(allRemoved, valid, removed);
      Algorithm::Sort(removed);
      Algorithm::Unique(removed);
      const Real grow = epsilon * (1 + Real(1e-9)) +
        64 * UnitRoundoff * viskores::Max(inputDistance.MaxAbsCoordinate, generatorMax);
      viskores::cont::ArrayHandle<viskores::Id> counts, nearOffsets;
      viskores::cont::ArrayHandle<viskores::Id2> near;
      invoke(sf::NearRemoved<false>(grow),
             removed,
             viskores::cont::make_ArrayHandleConstant(viskores::Id(0), removed.GetNumberOfValues()),
             outputTriangles,
             needsTest,
             inputDistance.Hierarchy,
             meshObject,
             region,
             counts,
             near);
      const viskores::Id numNear = Algorithm::ScanExclusive(counts, nearOffsets);
      near.Allocate(numNear);
      invoke(sf::NearRemoved<true>(grow),
             removed,
             nearOffsets,
             outputTriangles,
             needsTest,
             inputDistance.Hierarchy,
             meshObject,
             region,
             counts,
             near);
      sf::Append(items, near);
    }
    {
      viskores::cont::ArrayHandle<viskores::UInt8> keep;
      viskores::cont::ArrayHandle<viskores::Id2> kept;
      invoke(sf::NotFired{}, items, fires, keep);
      Algorithm::CopyIf(items, keep, kept);
      items = kept;
    }
    if (rescaled.GetNumberOfValues() > 0)
    {
      viskores::cont::ArrayHandle<viskores::Id2> all;
      invoke(
        sf::AllTriangles(mesh.NumberOfTriangles),
        viskores::cont::ArrayHandleIndex(rescaled.GetNumberOfValues() * mesh.NumberOfTriangles),
        rescaled,
        all);
      sf::Append(items, all);
    }
    Algorithm::Sort(items);
    Algorithm::Unique(items);
    if (items.GetNumberOfValues() > 0)
    {
      const DistanceMesh outputDistance =
        BuildDistanceMesh(clusters.Generators, outputTriangles, sourceKeys);
      viskores::cont::ArrayHandle<viskores::UInt32> fired;
      Algorithm::Copy(viskores::cont::make_ArrayHandleCast<viskores::UInt32>(fires), fired);
      invoke(sf::EnvelopeItemFires(epsilon, sides),
             items,
             scaleInput,
             scaleOutput,
             generatorScale,
             groupTriangles,
             groupOffsets,
             groupSources,
             DistanceMeshObject(inputDistance),
             DistanceMeshObject(outputDistance),
             meshObject,
             region,
             fired);
      invoke(sf::FiredFlag{}, fired, fires);
    }
  }
  return fires;
}

}
}
} // namespace viskores::worklet::quadric_clustering

VISKORES_QUADRIC_CLUSTERING_STRICT_FP_END

#endif // viskores_worklet_quadric_clustering_SegmentFlags_h
