//============================================================================
//  The contents of this file are covered by the Viskores license. See
//  LICENSE.txt for details.
//
//  By contributing to this file, all contributors agree to the Developer
//  Certificate of Origin Version 1.1 (DCO 1.1) as stated in DCO.txt.
//============================================================================
#ifndef viskores_worklet_quadric_clustering_GlueItems_h
#define viskores_worklet_quadric_clustering_GlueItems_h

#include <viskores/filter/geometry_refinement/worklet/quadric_clustering/Clustering.h>
#include <viskores/filter/geometry_refinement/worklet/quadric_clustering/QuadricClusteringCommonWorklets.h>
#include <viskores/filter/geometry_refinement/worklet/quadric_clustering/SingularCurves.h>
#include <viskores/filter/geometry_refinement/worklet/quadric_clustering/SurfaceMesh.h>

#include <viskores/cont/Algorithm.h>
#include <viskores/cont/ArrayHandle.h>
#include <viskores/cont/ArrayHandleGroupVec.h>
#include <viskores/cont/ArrayHandleIndex.h>
#include <viskores/cont/ExecutionObjectBase.h>
#include <viskores/worklet/WorkletMapField.h>

VISKORES_QUADRIC_CLUSTERING_STRICT_FP_BEGIN

namespace viskores
{
namespace worklet
{
namespace quadric_clustering
{

// Local changes of a labelling with the singular curves decimated: candidate merges that change
// a few clusters and glue groups, the region of the mesh each changes, and (d') on it.

/// The labellings of the candidates of a round of segment merges, each given by what it changes
/// in the labelling of the round: the clusters of its sources join its targets, and the labels
/// of its two glue groups, which become one, move to new generators.
struct CandidateRegions
{
  viskores::Id NumberOfCandidates = 0;
  /// `(candidate, from, to)`, sorted, and their offsets per candidate.
  viskores::cont::ArrayHandle<viskores::Id3> Relabels;
  viskores::cont::ArrayHandle<viskores::Id> RelabelOffsets;
  /// `(candidate, label, new label)` for the labels of the two glue groups, sorted, and their
  /// offsets per candidate; per record, the generator and the size of a label that stays.
  viskores::cont::ArrayHandle<viskores::Id3> Labels;
  viskores::cont::ArrayHandle<viskores::Id> LabelOffsets;
  viskores::cont::ArrayHandle<Vec3> LabelGenerators;
  viskores::cont::ArrayHandle<viskores::Id> LabelSizes;
  /// `(candidate, vertex, label, new label)` for their vertices, sorted.
  viskores::cont::ArrayHandle<viskores::Vec<viskores::Id, 4>> Vertices;
  /// The vertices of the clusters a candidate changes, by `(candidate, cluster)` then vertex;
  /// per changed cluster, in the order of the distinct keys, its size.
  viskores::cont::ArrayHandle<viskores::Id2> MemberKeys;
  viskores::cont::ArrayHandle<viskores::Id> MemberVertices;
  viskores::cont::ArrayHandle<viskores::Id> MergedSizes;
  /// Per candidate: the glue ids of its two groups.
  viskores::cont::ArrayHandle<viskores::Id2> GroupGlue;
};

namespace region_detail
{

using IdPortal = viskores::cont::ArrayHandle<viskores::Id>::ReadPortalType;
using Id2Portal = viskores::cont::ArrayHandle<viskores::Id2>::ReadPortalType;
using Id3Portal = viskores::cont::ArrayHandle<viskores::Id3>::ReadPortalType;
using VecPortal = viskores::cont::ArrayHandle<Vec3>::ReadPortalType;
using FlagPortal = viskores::cont::ArrayHandle<viskores::UInt8>::ReadPortalType;

/// The input mesh and its adjacency, in the execution environment.
struct MeshPortals
{
  VISKORES_EXEC viskores::Id CornerTriangle(viskores::Id r) const
  {
    return this->VertexCorners.Get(r) % this->NumberOfTriangles;
  }

  VISKORES_EXEC viskores::Id OtherEnd(viskores::Id e, viskores::Id v) const
  {
    const viskores::Id2 edge = this->Edges.Get(e);
    return (edge[0] == v) ? edge[1] : edge[0];
  }

  VecPortal Points;
  Id3Portal Triangles;
  Id2Portal Edges;
  Id2Portal EdgeTriangles;
  IdPortal VertexEdgeOffsets;
  IdPortal VertexEdges;
  IdPortal VertexCornerOffsets;
  IdPortal VertexCorners;
  FlagPortal IsBoundaryVertex;
  viskores::Id NumberOfTriangles;
};

struct MeshObject : viskores::cont::ExecutionObjectBase
{
  explicit MeshObject(const SurfaceMesh& mesh)
    : Mesh(&mesh)
  {
  }

  VISKORES_CONT MeshPortals PrepareForExecution(viskores::cont::DeviceAdapterId device,
                                                viskores::cont::Token& token) const
  {
    const SurfaceMesh& m = *this->Mesh;
    return MeshPortals{ m.Points.PrepareForInput(device, token),
                        m.Triangles.PrepareForInput(device, token),
                        m.Edges.PrepareForInput(device, token),
                        m.EdgeTriangles.PrepareForInput(device, token),
                        m.VertexEdgeOffsets.PrepareForInput(device, token),
                        m.VertexEdges.PrepareForInput(device, token),
                        m.VertexCornerOffsets.PrepareForInput(device, token),
                        m.VertexCorners.PrepareForInput(device, token),
                        m.IsBoundaryVertex.PrepareForInput(device, token),
                        m.NumberOfTriangles };
  }

  const SurfaceMesh* Mesh;
};

/// The label candidate `c` gives `label`: its target if it is a source of the candidate, from
/// the relabels `(candidate, from, to)` sorted by candidate.
template <typename RelabelPortal, typename OffsetPortal>
VISKORES_EXEC viskores::Id NewLabelOf(const RelabelPortal& relabels,
                                      const OffsetPortal& relabelOffsets,
                                      viskores::Id c,
                                      viskores::Id label)
{
  for (viskores::Id r = relabelOffsets.Get(c); r < relabelOffsets.Get(c + 1); ++r)
  {
    const viskores::Id3 relabel = relabels.Get(r);
    if (relabel[1] == label)
    {
      return relabel[2];
    }
  }
  return label;
}

/// Whether `label` is a target of candidate `c`: a cluster the candidate changes.
template <typename RelabelPortal, typename OffsetPortal>
VISKORES_EXEC bool IsTargetOf(const RelabelPortal& relabels,
                              const OffsetPortal& relabelOffsets,
                              viskores::Id c,
                              viskores::Id label)
{
  for (viskores::Id r = relabelOffsets.Get(c); r < relabelOffsets.Get(c + 1); ++r)
  {
    if (relabels.Get(r)[2] == label)
    {
      return true;
    }
  }
  return false;
}

/// The labelling of every candidate, in the execution environment: a label of the labelling of
/// the round, or a new label, which is a label of the round that stays.
struct RegionPortals
{
  /// The new label of a label of the round.
  VISKORES_EXEC viskores::Id NewLabel(viskores::Id c, viskores::Id label) const
  {
    return NewLabelOf(this->Relabels, this->RelabelOffsets, c, label);
  }

  VISKORES_EXEC viskores::Id VertexLabel(viskores::Id c, viskores::Id v) const
  {
    return this->NewLabel(c, this->Labels.Get(v));
  }

  /// Whether a new label is a cluster the candidate changes.
  VISKORES_EXEC bool IsTarget(viskores::Id c, viskores::Id label) const
  {
    return IsTargetOf(this->Relabels, this->RelabelOffsets, c, label);
  }

  /// The record of a label of the two glue groups, or -1.
  VISKORES_EXEC viskores::Id Record(viskores::Id c, viskores::Id label) const
  {
    viskores::Id lo =
      PartitionPoint(this->LabelOffsets.Get(c),
                     this->LabelOffsets.Get(c + 1),
                     [&](viskores::Id mid) { return this->LabelRecords.Get(mid)[1] < label; });
    return (lo < this->LabelOffsets.Get(c + 1) && this->LabelRecords.Get(lo)[1] == label) ? lo : -1;
  }

  VISKORES_EXEC Vec3 Generator(viskores::Id c, viskores::Id label) const
  {
    const viskores::Id r = this->Record(c, label);
    return (r >= 0) ? this->LabelGenerators.Get(r) : this->Generators.Get(label);
  }

  VISKORES_EXEC viskores::Id Size(viskores::Id c, viskores::Id label) const
  {
    const viskores::Id r = this->Record(c, label);
    return (r >= 0) ? this->LabelSizes.Get(r) : this->Sizes.Get(label);
  }

  /// The glue id of a new label: the merged group takes the smaller id of the two.
  VISKORES_EXEC viskores::Id GlueOf(viskores::Id c, viskores::Id label) const
  {
    if (this->Record(c, label) >= 0)
    {
      const viskores::Id2 g = this->GroupGlue.Get(c);
      return viskores::Min(g[0], g[1]);
    }
    return this->Glue.Get(label);
  }

  /// Whether a label of the round is in the two glue groups.
  VISKORES_EXEC bool Affected(viskores::Id c, viskores::Id label) const
  {
    return this->Record(c, label) >= 0;
  }

  /// Whether an input triangle has a vertex in the two glue groups.
  template <typename Mesh>
  VISKORES_EXEC bool Incident(viskores::Id c, const Mesh& mesh, viskores::Id t) const
  {
    const viskores::Id3 tri = mesh.Triangles.Get(t);
    return this->Affected(c, this->Labels.Get(tri[0])) ||
      this->Affected(c, this->Labels.Get(tri[1])) || this->Affected(c, this->Labels.Get(tri[2]));
  }

  IdPortal Labels;
  VecPortal Generators;
  IdPortal Sizes;
  IdPortal Glue;
  Id3Portal Relabels;
  IdPortal RelabelOffsets;
  Id3Portal LabelRecords;
  IdPortal LabelOffsets;
  VecPortal LabelGenerators;
  IdPortal LabelSizes;
  Id2Portal GroupGlue;
};

struct RegionObject : viskores::cont::ExecutionObjectBase
{
  RegionObject(const CandidateRegions& regions,
               const Clustering& clusters,
               const viskores::cont::ArrayHandle<viskores::Id>& glue)
    : Regions(&regions)
    , Clusters(&clusters)
    , Glue(glue)
  {
  }

  VISKORES_CONT RegionPortals PrepareForExecution(viskores::cont::DeviceAdapterId device,
                                                  viskores::cont::Token& token) const
  {
    const CandidateRegions& r = *this->Regions;
    return RegionPortals{ this->Clusters->Labels.PrepareForInput(device, token),
                          this->Clusters->Generators.PrepareForInput(device, token),
                          this->Clusters->Sizes.PrepareForInput(device, token),
                          this->Glue.PrepareForInput(device, token),
                          r.Relabels.PrepareForInput(device, token),
                          r.RelabelOffsets.PrepareForInput(device, token),
                          r.Labels.PrepareForInput(device, token),
                          r.LabelOffsets.PrepareForInput(device, token),
                          r.LabelGenerators.PrepareForInput(device, token),
                          r.LabelSizes.PrepareForInput(device, token),
                          r.GroupGlue.PrepareForInput(device, token) };
  }

  const CandidateRegions* Regions;
  const Clustering* Clusters;
  viskores::cont::ArrayHandle<viskores::Id> Glue;
};

//---------------------------------------------------------------------------------------------
// The output triangles of the candidates at their merged group, for (d'), (F4) and (F5).

struct CornerSlots : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn record, ExecObject mesh, FieldOut slots);
  using ExecutionSignature = _3(_1, _2);

  template <typename Mesh>
  VISKORES_EXEC viskores::Id operator()(const viskores::Vec<viskores::Id, 4>& r,
                                        const Mesh& mesh) const
  {
    return mesh.VertexCornerOffsets.Get(r[1] + 1) - mesh.VertexCornerOffsets.Get(r[1]);
  }
};

/// The input triangles at the vertices of a candidate's two glue groups, each from its smallest
/// vertex there: `(candidate, t)`, and in `output`, as `(candidate, sorted new label triple, t)`
/// for the 3-label ones, the triangles of `K_X` at the merged group.
struct IncidentTriangles : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn record,
                                FieldIn offset,
                                ExecObject mesh,
                                ExecObject region,
                                WholeArrayOut incident,
                                WholeArrayOut output);
  using ExecutionSignature = void(_1, _2, _3, _4, _5, _6);

  template <typename Mesh, typename Region, typename IncidentPortal, typename OutputPortal>
  VISKORES_EXEC void operator()(const viskores::Vec<viskores::Id, 4>& record,
                                viskores::Id offset,
                                const Mesh& mesh,
                                const Region& region,
                                IncidentPortal& incident,
                                OutputPortal& output) const
  {
    const viskores::Id c = record[0], v = record[1];
    viskores::Id q = offset;
    for (viskores::Id r = mesh.VertexCornerOffsets.Get(v); r < mesh.VertexCornerOffsets.Get(v + 1);
         ++r, ++q)
    {
      const viskores::Id t = mesh.CornerTriangle(r);
      const viskores::Id3 tri = mesh.Triangles.Get(t);
      viskores::Id first = -1;
      viskores::Id l[3];
      for (int k = 0; k < 3; ++k)
      {
        if (region.Affected(c, region.Labels.Get(tri[k])) && (first < 0 || tri[k] < first))
        {
          first = tri[k];
        }
        l[k] = region.VertexLabel(c, tri[k]);
      }
      const bool emit = first == v;
      incident.Set(q, emit ? viskores::Id2(c, t) : viskores::Id2(-1));
      if (emit && l[0] != l[1] && l[1] != l[2] && l[0] != l[2])
      {
        const viskores::Id3 s = SortThree(l[0], l[1], l[2]);
        output.Set(q, viskores::Vec<viskores::Id, 5>(c, s[0], s[1], s[2], t));
      }
      else
      {
        output.Set(q, viskores::Vec<viskores::Id, 5>(-1));
      }
    }
  }
};

/// An output triangle of a candidate at its merged group: its new labels, in the order of its
/// first input triangle.
struct OrientedTriangle : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn record,
                                ExecObject mesh,
                                ExecObject region,
                                FieldOut candidate,
                                FieldOut labels,
                                FieldOut first);
  using ExecutionSignature = void(_1, _2, _3, _4, _5, _6);

  template <typename Mesh, typename Region>
  VISKORES_EXEC void operator()(const viskores::Vec<viskores::Id, 5>& record,
                                const Mesh& mesh,
                                const Region& region,
                                viskores::Id& candidate,
                                viskores::Id3& labels,
                                viskores::Id& first) const
  {
    candidate = record[0];
    first = record[4];
    const viskores::Id3 tri = mesh.Triangles.Get(record[4]);
    for (int k = 0; k < 3; ++k)
    {
      labels[k] = region.VertexLabel(candidate, tri[k]);
    }
  }
};

} // namespace region_detail

/// The input triangles at the two glue groups of every candidate, `(candidate, t)`, each from
/// its smallest vertex there, and the output triangles of the candidate's labelling at its
/// merged group: per candidate, their new labels in the order of their first input triangle,
/// and the index of that triangle.
inline void GroupTriangles(const SurfaceMesh& mesh,
                           const Clustering& clusters,
                           const viskores::cont::ArrayHandle<viskores::Id>& glue,
                           const CandidateRegions& regions,
                           viskores::cont::ArrayHandle<viskores::Id2>& incident,
                           viskores::cont::ArrayHandle<viskores::Id>& groupCandidates,
                           viskores::cont::ArrayHandle<viskores::Id3>& groupTriangles,
                           viskores::cont::ArrayHandle<viskores::Id>& groupOffsets,
                           viskores::cont::ArrayHandle<viskores::Id>& groupSources)
{
  using Algorithm = viskores::cont::Algorithm;
  namespace rd = region_detail;
  viskores::cont::Invoker invoke;
  const rd::MeshObject meshObject(mesh);
  const rd::RegionObject region(regions, clusters, glue);
  viskores::cont::ArrayHandle<viskores::Id> slots, offsets;
  invoke(rd::CornerSlots{}, regions.Vertices, meshObject, slots);
  const viskores::Id total = Algorithm::ScanExclusive(slots, offsets);
  viskores::cont::ArrayHandle<viskores::Id2> allIncident;
  viskores::cont::ArrayHandle<viskores::Vec<viskores::Id, 5>> allOutput, output, first;
  allIncident.Allocate(total);
  allOutput.Allocate(total);
  invoke(
    rd::IncidentTriangles{}, regions.Vertices, offsets, meshObject, region, allIncident, allOutput);
  viskores::cont::ArrayHandle<viskores::UInt8> valid;
  invoke(IsValidRecord{}, allIncident, valid);
  Algorithm::CopyIf(allIncident, valid, incident);
  invoke(IsValidRecord{}, allOutput, valid);
  Algorithm::CopyIf(allOutput, valid, output);
  Algorithm::Sort(output);
  invoke(
    FirstOfRun<4>{}, viskores::cont::ArrayHandleIndex(output.GetNumberOfValues()), output, valid);
  Algorithm::CopyIf(output, valid, first);
  invoke(rd::OrientedTriangle{},
         first,
         meshObject,
         region,
         groupCandidates,
         groupTriangles,
         groupSources);
  Algorithm::LowerBounds(groupCandidates,
                         viskores::cont::ArrayHandleIndex(regions.NumberOfCandidates + 1),
                         groupOffsets);
}

namespace region_detail
{

//---------------------------------------------------------------------------------------------
// (d')

/// The edges of `K_X` at the merged group of a candidate, from its output triangles there.
struct GroupEdges : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn candidate,
                                FieldIn triangle,
                                ExecObject region,
                                FieldOut edges);
  using ExecutionSignature = void(_1, _2, _3, _4);

  template <typename Region, typename EdgeVec>
  VISKORES_EXEC void operator()(viskores::Id c,
                                const viskores::Id3& t,
                                const Region& region,
                                EdgeVec& edges) const
  {
    for (viskores::IdComponent s = 0; s < 3; ++s)
    {
      const viskores::Id a = t[s], b = t[(s + 1) % 3];
      edges[s] = (region.Affected(c, a) || region.Affected(c, b))
        ? viskores::Vec<viskores::Id, 3>(c, viskores::Min(a, b), viskores::Max(a, b))
        : viskores::Vec<viskores::Id, 3>(-1);
    }
  }
};

struct EdgeSlots : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn record, ExecObject mesh, FieldOut slots);
  using ExecutionSignature = _3(_1, _2);

  template <typename Mesh>
  VISKORES_EXEC viskores::Id operator()(const viskores::Vec<viskores::Id, 4>& r,
                                        const Mesh& mesh) const
  {
    return mesh.VertexEdgeOffsets.Get(r[1] + 1) - mesh.VertexEdgeOffsets.Get(r[1]);
  }
};

/// The steps of a vertex of a candidate's glue groups: its cut edges that are copies of edges
/// of `X` between two new clusters, as `(candidate, labels, input vertices)`, each pair sorted.
struct GroupSteps : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn record,
                                FieldIn offset,
                                ExecObject mesh,
                                ExecObject region,
                                WholeArrayIn cutEdge,
                                WholeArrayIn original,
                                WholeArrayOut steps);
  using ExecutionSignature = void(_1, _2, _3, _4, _5, _6, _7);

  template <typename Mesh,
            typename Region,
            typename FlagPortal,
            typename IdPortal,
            typename StepPortal>
  VISKORES_EXEC void operator()(const viskores::Vec<viskores::Id, 4>& record,
                                viskores::Id offset,
                                const Mesh& mesh,
                                const Region& region,
                                const FlagPortal& cutEdge,
                                const IdPortal& original,
                                StepPortal& steps) const
  {
    const viskores::Id c = record[0], v = record[1];
    viskores::Id q = offset;
    for (viskores::Id r = mesh.VertexEdgeOffsets.Get(v); r < mesh.VertexEdgeOffsets.Get(v + 1);
         ++r, ++q)
    {
      const viskores::Id e = mesh.VertexEdges.Get(r);
      const viskores::Id w = mesh.OtherEnd(e, v);
      const viskores::Id la = record[3], lb = region.VertexLabel(c, w);
      if (!cutEdge.Get(e) || la == lb)
      {
        steps.Set(q, viskores::Vec<viskores::Id, 5>(-1));
        continue;
      }
      const viskores::Id oa = original.Get(v), ob = original.Get(w);
      steps.Set(q,
                viskores::Vec<viskores::Id, 5>(c,
                                               viskores::Min(la, lb),
                                               viskores::Max(la, lb),
                                               viskores::Min(oa, ob),
                                               viskores::Max(oa, ob)));
    }
  }
};

/// Marks the candidate of a record that fires; every write is a 1.
template <typename FlagPortal>
VISKORES_EXEC void Fire(FlagPortal& fires, viskores::Id c)
{
  fires.Set(c, 1);
}

/// An edge of `K_X` at a merged group: its glued edge, firing if it joins two vertices the
/// gluing identifies, and the edges of `X` that realise it, as `GlueEdge`.
struct GroupEdgeImage : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn edge,
                                WholeArrayIn steps,
                                ExecObject region,
                                FieldOut image,
                                WholeArrayInOut fires);
  using ExecutionSignature = void(_1, _2, _3, _4, _5);

  template <typename StepPortal, typename Region, typename FirePortal>
  VISKORES_EXEC void operator()(const viskores::Vec<viskores::Id, 3>& edge,
                                const StepPortal& steps,
                                const Region& region,
                                viskores::Vec<viskores::Id, 6>& image,
                                FirePortal& fires) const
  {
    const viskores::Id c = edge[0];
    const viskores::Id ga = region.GlueOf(c, edge[1]), gb = region.GlueOf(c, edge[2]);
    if (ga == gb)
    {
      Fire(fires, c);
    }
    auto before = [&](const viskores::Vec<viskores::Id, 5>& s)
    {
      for (int k = 0; k < 3; ++k)
      {
        if (s[k] != edge[k])
        {
          return s[k] < edge[k];
        }
      }
      return false;
    };
    auto equal = [&](const viskores::Vec<viskores::Id, 5>& s)
    { return s[0] == edge[0] && s[1] == edge[1] && s[2] == edge[2]; };
    viskores::Id lo = PartitionPoint(
      0, steps.GetNumberOfValues(), [&](viskores::Id mid) { return before(steps.Get(mid)); });
    viskores::Id count = 0;
    viskores::Id2 realised(-1, -1);
    for (viskores::Id s = lo; s < steps.GetNumberOfValues() && equal(steps.Get(s)) && count < 2;
         ++s)
    {
      ++count;
      realised = viskores::Id2(steps.Get(s)[3], steps.Get(s)[4]);
    }
    image = viskores::Vec<viskores::Id, 6>(
      c, viskores::Min(ga, gb), viskores::Max(ga, gb), count, realised[0], realised[1]);
  }
};

/// (d') on the edges of `K_X` at the merged groups sorted by glued edge, as `GlueEdgeViolation`.
struct GroupEdgeFires : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn index, WholeArrayIn images, WholeArrayInOut fires);
  using ExecutionSignature = void(_1, _2, _3);

  template <typename ImagePortal, typename FirePortal>
  VISKORES_EXEC void operator()(viskores::Id i, const ImagePortal& images, FirePortal& fires) const
  {
    auto same = [&](viskores::Id a, viskores::Id b)
    {
      const auto x = images.Get(a), y = images.Get(b);
      return x[0] == y[0] && x[1] == y[1] && x[2] == y[2];
    };
    // From the first of every run.
    if (i > 0 && same(i, i - 1))
    {
      return;
    }
    viskores::Id last = i + 1;
    while (last < images.GetNumberOfValues() && same(last, i))
    {
      ++last;
    }
    if (last - i < 2)
    {
      return;
    }
    const auto first = images.Get(i);
    for (viskores::Id m = i; m < last; ++m)
    {
      const auto x = images.Get(m);
      if (x[3] != 1 || x[4] != first[4] || x[5] != first[5])
      {
        Fire(fires, x[0]);
        return;
      }
    }
  }
};

/// The glued triangle of an output triangle at a merged group, sorted.
struct GroupGluedTriangle : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn candidate,
                                FieldIn triangle,
                                ExecObject region,
                                FieldOut glued);
  using ExecutionSignature = _4(_1, _2, _3);

  template <typename Region>
  VISKORES_EXEC viskores::Vec<viskores::Id, 4> operator()(viskores::Id c,
                                                          const viskores::Id3& t,
                                                          const Region& region) const
  {
    const viskores::Id3 s =
      SortThree(region.GlueOf(c, t[0]), region.GlueOf(c, t[1]), region.GlueOf(c, t[2]));
    return viskores::Vec<viskores::Id, 4>(c, s[0], s[1], s[2]);
  }
};

/// A record equal to the one before it: a duplicate triple, (D2) or a (d') triangle.
struct DuplicateFires : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn index, WholeArrayIn records, WholeArrayInOut fires);
  using ExecutionSignature = void(_1, _2, _3);

  template <typename RecordPortal, typename FirePortal>
  VISKORES_EXEC void operator()(viskores::Id i,
                                const RecordPortal& records,
                                FirePortal& fires) const
  {
    if (i > 0 && records.Get(i) == records.Get(i - 1))
    {
      Fire(fires, records.Get(i)[0]);
    }
  }
};

} // namespace region_detail

/// (d') of Proposition "Decimating the singular curves" at the merged group of every candidate,
/// given that it holds on the labelling of the round: the edges of `K_X` with an end at the
/// merged group and its triangles there, from `GroupTriangles`. Sets `fires` for a candidate
/// that violates it.
inline void GroupGlueFires(const SurfaceMesh& mesh,
                           const viskores::cont::ArrayHandle<viskores::Id>& original,
                           const SingularCurves& curves,
                           const Clustering& clusters,
                           const viskores::cont::ArrayHandle<viskores::Id>& glue,
                           const CandidateRegions& regions,
                           const viskores::cont::ArrayHandle<viskores::Id>& groupCandidates,
                           const viskores::cont::ArrayHandle<viskores::Id3>& groupTriangles,
                           viskores::cont::ArrayHandle<viskores::UInt8>& fires)
{
  using Algorithm = viskores::cont::Algorithm;
  namespace rd = region_detail;
  viskores::cont::Invoker invoke;
  if (groupTriangles.GetNumberOfValues() == 0)
  {
    return;
  }
  const rd::MeshObject meshObject(mesh);
  const rd::RegionObject region(regions, clusters, glue);
  viskores::cont::ArrayHandle<viskores::Vec<viskores::Id, 3>> edges, validEdges;
  invoke(rd::GroupEdges{},
         groupCandidates,
         groupTriangles,
         region,
         viskores::cont::make_ArrayHandleGroupVec<3>(edges));
  viskores::cont::ArrayHandle<viskores::UInt8> valid;
  invoke(IsValidRecord{}, edges, valid);
  Algorithm::CopyIf(edges, valid, validEdges);
  Algorithm::Sort(validEdges);
  Algorithm::Unique(validEdges);
  viskores::cont::ArrayHandle<viskores::Id> slots, offsets;
  invoke(rd::EdgeSlots{}, regions.Vertices, meshObject, slots);
  const viskores::Id total = Algorithm::ScanExclusive(slots, offsets);
  viskores::cont::ArrayHandle<viskores::Vec<viskores::Id, 5>> allSteps, steps;
  allSteps.Allocate(total);
  invoke(rd::GroupSteps{},
         regions.Vertices,
         offsets,
         meshObject,
         region,
         curves.CutEdge,
         original,
         allSteps);
  invoke(IsValidRecord{}, allSteps, valid);
  Algorithm::CopyIf(allSteps, valid, steps);
  Algorithm::Sort(steps);
  Algorithm::Unique(steps);
  viskores::cont::ArrayHandle<viskores::Vec<viskores::Id, 6>> images;
  invoke(rd::GroupEdgeImage{}, validEdges, steps, region, images, fires);
  Algorithm::Sort(images);
  invoke(rd::GroupEdgeFires{},
         viskores::cont::ArrayHandleIndex(images.GetNumberOfValues()),
         images,
         fires);
  viskores::cont::ArrayHandle<viskores::Vec<viskores::Id, 4>> glued;
  invoke(rd::GroupGluedTriangle{}, groupCandidates, groupTriangles, region, glued);
  Algorithm::Sort(glued);
  invoke(rd::DuplicateFires{},
         viskores::cont::ArrayHandleIndex(glued.GetNumberOfValues()),
         glued,
         fires);
}

}
}
} // namespace viskores::worklet::quadric_clustering

VISKORES_QUADRIC_CLUSTERING_STRICT_FP_END

#endif // viskores_worklet_quadric_clustering_GlueItems_h
