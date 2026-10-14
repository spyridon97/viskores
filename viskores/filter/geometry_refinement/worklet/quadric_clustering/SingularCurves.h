//============================================================================
//  The contents of this file are covered by the Viskores license. See
//  LICENSE.txt for details.
//
//  By contributing to this file, all contributors agree to the Developer
//  Certificate of Origin Version 1.1 (DCO 1.1) as stated in DCO.txt.
//============================================================================
#ifndef viskores_worklet_quadric_clustering_SingularCurves_h
#define viskores_worklet_quadric_clustering_SingularCurves_h

#include <viskores/filter/geometry_refinement/worklet/quadric_clustering/Clustering.h>
#include <viskores/filter/geometry_refinement/worklet/quadric_clustering/TopologyFlags.h>

#include <viskores/Swap.h>
#include <viskores/cont/Algorithm.h>
#include <viskores/cont/ArrayHandleIndex.h>
#include <viskores/worklet/WorkletMapField.h>

#include <algorithm>
#include <map>
#include <set>
#include <vector>

VISKORES_QUADRIC_CLUSTERING_STRICT_FP_BEGIN

namespace viskores
{
namespace worklet
{
namespace quadric_clustering
{

/// The singular curves of an input cut into surfaces (Proposition "Decimating the singular
/// curves"). A frozen input vertex is regular if it lies on exactly two edges of the cut `X` and
/// every copy of it lies on one copy of each; the others are junctions. An arc is a path of
/// edges of `X` whose interior vertices are regular and whose ends are junctions, or a cycle of
/// regular vertices. The interior vertices of every arc are grouped into segments, runs of
/// consecutive vertices, initially one per vertex; in every sheet the copies of a segment form
/// one cluster, a piece, and the pieces of a segment are one vertex of the glued output.
struct SingularCurves
{
  struct Arc
  {
    /// The interior vertices, in order.
    std::vector<viskores::Id> Vertices;
    /// The junctions at the two ends, for an arc that is not closed.
    viskores::Id2 Ends;
    bool Closed;
  };

  std::vector<Arc> Arcs;
  /// The segments of every arc, in order, each the list of its vertices.
  std::vector<std::vector<std::vector<viskores::Id>>> Segments;
  /// For every input vertex: the first vertex of its segment, itself for a junction, or -1 for
  /// a vertex that is not frozen.
  std::vector<viskores::Id> Segment;
  /// Whether every input vertex is a regular arc vertex.
  std::vector<bool> Regular;
  /// For every edge of the cut mesh, whether it is a copy of an edge of `X`.
  viskores::cont::ArrayHandle<viskores::UInt8> CutEdge;
};

/// The arcs of the singular graph, the frozen input vertices with the edges of `X`, as
/// `arcs_of` of proofs/lib/arcs.py: junctions in increasing order, their neighbours in
/// increasing order, then the cycles of regular vertices from their smallest vertex.
inline SingularCurves FindSingularCurves(const std::vector<viskores::Id2>& cutEdges,
                                         const SurfaceMesh& mesh,
                                         const std::vector<viskores::Id>& original,
                                         const std::vector<viskores::UInt8>& frozen,
                                         viskores::Id numInputVertices)
{
  SingularCurves curves;
  const std::size_t n = static_cast<std::size_t>(numInputVertices);
  const std::set<std::pair<viskores::Id, viskores::Id>> inX = [&]
  {
    std::set<std::pair<viskores::Id, viskores::Id>> s;
    for (const auto& e : cutEdges)
    {
      s.insert(std::make_pair(std::min(e[0], e[1]), std::max(e[0], e[1])));
    }
    return s;
  }();
  std::vector<std::set<viskores::Id>> neighbors(n);
  for (const auto& e : inX)
  {
    neighbors[static_cast<std::size_t>(e.first)].insert(e.second);
    neighbors[static_cast<std::size_t>(e.second)].insert(e.first);
  }
  // The input vertices that the copies of the edges of X at every copy reach.
  std::vector<std::vector<viskores::Id>> reach(original.size());
  std::vector<viskores::UInt8> cutEdge(static_cast<std::size_t>(mesh.Edges.GetNumberOfValues()), 0);
  {
    auto edges = mesh.Edges.ReadPortal();
    for (viskores::Id h = 0; h < edges.GetNumberOfValues(); ++h)
    {
      const viskores::Id2 e = edges.Get(h);
      const viskores::Id oa = original[static_cast<std::size_t>(e[0])];
      const viskores::Id ob = original[static_cast<std::size_t>(e[1])];
      if (inX.count(std::make_pair(std::min(oa, ob), std::max(oa, ob))))
      {
        cutEdge[static_cast<std::size_t>(h)] = 1;
        reach[static_cast<std::size_t>(e[0])].push_back(ob);
        reach[static_cast<std::size_t>(e[1])].push_back(oa);
      }
    }
  }
  curves.CutEdge = viskores::cont::make_ArrayHandle(cutEdge, viskores::CopyFlag::On);
  std::vector<std::vector<viskores::Id>> copies(n);
  for (std::size_t c = 0; c < original.size(); ++c)
  {
    copies[static_cast<std::size_t>(original[c])].push_back(static_cast<viskores::Id>(c));
  }
  std::vector<bool> isFrozen(n, false);
  for (std::size_t c = 0; c < original.size(); ++c)
  {
    if (frozen[c])
    {
      isFrozen[static_cast<std::size_t>(original[c])] = true;
    }
  }
  curves.Regular.assign(n, false);
  curves.Segment.assign(n, -1);
  for (std::size_t v = 0; v < n; ++v)
  {
    if (!isFrozen[v])
    {
      continue;
    }
    curves.Segment[v] = static_cast<viskores::Id>(v);
    const std::vector<viskores::Id> around(neighbors[v].begin(), neighbors[v].end());
    bool regular = around.size() == 2;
    for (viskores::Id c : copies[v])
    {
      std::vector<viskores::Id> r = reach[static_cast<std::size_t>(c)];
      std::sort(r.begin(), r.end());
      regular = regular && r == around;
    }
    curves.Regular[v] = regular;
  }
  auto other = [&](viskores::Id cur, viskores::Id prev)
  {
    const auto& around = neighbors[static_cast<std::size_t>(cur)];
    return *around.begin() != prev ? *around.begin() : *std::next(around.begin());
  };
  std::set<std::pair<viskores::Id, viskores::Id>> seen;
  for (std::size_t j = 0; j < n; ++j)
  {
    if (!isFrozen[j] || curves.Regular[j])
    {
      continue;
    }
    for (viskores::Id w : neighbors[j])
    {
      const viskores::Id jj = static_cast<viskores::Id>(j);
      if (!curves.Regular[static_cast<std::size_t>(w)] || seen.count(std::make_pair(jj, w)))
      {
        continue;
      }
      SingularCurves::Arc arc;
      viskores::Id prev = jj, cur = w;
      while (curves.Regular[static_cast<std::size_t>(cur)])
      {
        arc.Vertices.push_back(cur);
        seen.insert(std::make_pair(prev, cur));
        seen.insert(std::make_pair(cur, prev));
        const viskores::Id next = other(cur, prev);
        prev = cur;
        cur = next;
      }
      seen.insert(std::make_pair(prev, cur));
      seen.insert(std::make_pair(cur, prev));
      arc.Ends = viskores::Id2(jj, cur);
      arc.Closed = false;
      curves.Arcs.push_back(arc);
    }
  }
  std::vector<bool> done(n, false);
  for (const auto& arc : curves.Arcs)
  {
    for (viskores::Id v : arc.Vertices)
    {
      done[static_cast<std::size_t>(v)] = true;
    }
  }
  for (std::size_t v = 0; v < n; ++v)
  {
    if (!curves.Regular[v] || done[v])
    {
      continue;
    }
    SingularCurves::Arc arc;
    const viskores::Id start = static_cast<viskores::Id>(v);
    arc.Vertices.push_back(start);
    viskores::Id prev = start, cur = *neighbors[v].begin();
    while (cur != start)
    {
      arc.Vertices.push_back(cur);
      const viskores::Id next = other(cur, prev);
      prev = cur;
      cur = next;
    }
    for (viskores::Id x : arc.Vertices)
    {
      done[static_cast<std::size_t>(x)] = true;
    }
    arc.Ends = viskores::Id2(-1, -1);
    arc.Closed = true;
    curves.Arcs.push_back(arc);
  }
  for (const auto& arc : curves.Arcs)
  {
    std::vector<std::vector<viskores::Id>> segments;
    for (viskores::Id v : arc.Vertices)
    {
      segments.push_back({ v });
    }
    curves.Segments.push_back(segments);
  }
  return curves;
}

/// The vertex of `K` of every label, as `glue_ids` of proofs/lib/arcs.py: for a label whose
/// anchor is a frozen copy, the smallest label with an anchor in the same segment, which joins
/// the pieces of a segment and the copies of a junction; every other label stays.
inline viskores::cont::ArrayHandle<viskores::Id> SegmentGlueMap(
  const SingularCurves& curves,
  const viskores::cont::ArrayHandle<viskores::UInt8>& frozenVertices,
  const viskores::cont::ArrayHandle<viskores::Id>& original,
  const Clustering& clustering)
{
  const viskores::Id k = clustering.NumberOfClusters;
  std::vector<viskores::Id> glue(static_cast<std::size_t>(k));
  auto anchors = clustering.Anchors.ReadPortal();
  auto frozen = frozenVertices.ReadPortal();
  auto copyOf = original.ReadPortal();
  std::map<viskores::Id, viskores::Id> first;
  for (viskores::Id label = 0; label < k; ++label)
  {
    const viskores::Id c = anchors.Get(label);
    glue[static_cast<std::size_t>(label)] = label;
    if (frozen.Get(c))
    {
      const viskores::Id s = curves.Segment[static_cast<std::size_t>(copyOf.Get(c))];
      glue[static_cast<std::size_t>(label)] = first.emplace(s, label).first->second;
    }
  }
  return viskores::cont::make_ArrayHandle(glue, viskores::CopyFlag::On);
}

/// Gives the clusters that `glue` joins one generator, the minimiser of the sum of their costs
/// (Proposition "Decimating the singular curves", (e)): `c = o + (sum Q_l)^-1 sum Q_l (c_l - o)`
/// with `o` the generator of the smallest label, summed in increasing label order, as `place`
/// of proofs/lib/arcs.py. The copies of one vertex, at its position, stay there.
inline void ShareGenerators(const viskores::cont::ArrayHandle<viskores::Id>& glue,
                            Clustering& clustering)
{
  const viskores::Id k = clustering.NumberOfClusters;
  std::map<viskores::Id, std::vector<viskores::Id>> groups;
  {
    auto portal = glue.ReadPortal();
    for (viskores::Id label = 0; label < k; ++label)
    {
      groups[portal.Get(label)].push_back(label);
    }
  }
  auto q = clustering.Q.ReadPortal();
  auto generators = clustering.Generators.WritePortal();
  for (const auto& group : groups)
  {
    const std::vector<viskores::Id>& labels = group.second;
    if (labels.size() < 2)
    {
      continue;
    }
    const Vec3 o = generators.Get(labels.front());
    Mat3 qs(0);
    Vec3 bs(0);
    for (viskores::Id l : labels)
    {
      qs = MatAdd(qs, q.Get(l));
      bs = Add(bs, MatVec(q.Get(l), Sub(generators.Get(l), o)));
    }
    const Vec3 c = Add(o, SolveSPD(qs, bs));
    for (viskores::Id l : labels)
    {
      generators.Set(l, c);
    }
  }
}

namespace detail
{

/// For an edge of the cut mesh between two clusters that is a copy of an edge of `X`: the two
/// labels and the two input vertices, each pair sorted; otherwise `(-1, -1, -1, -1)`.
struct CutEdgeStep : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn edge,
                                FieldIn cutEdge,
                                WholeArrayIn labels,
                                WholeArrayIn original,
                                FieldOut step);
  using ExecutionSignature = _5(_1, _2, _3, _4);

  template <typename IdPortal>
  VISKORES_EXEC viskores::Vec<viskores::Id, 4> operator()(const viskores::Id2& edge,
                                                          viskores::UInt8 cutEdge,
                                                          const IdPortal& labels,
                                                          const IdPortal& original) const
  {
    const viskores::Id la = labels.Get(edge[0]), lb = labels.Get(edge[1]);
    if (!cutEdge || la == lb)
    {
      return viskores::Vec<viskores::Id, 4>(-1);
    }
    const viskores::Id oa = original.Get(edge[0]), ob = original.Get(edge[1]);
    return viskores::Vec<viskores::Id, 4>(
      viskores::Min(la, lb), viskores::Max(la, lb), viskores::Min(oa, ob), viskores::Max(oa, ob));
  }
};

struct IsStep : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn step, FieldOut valid);
  using ExecutionSignature = _2(_1);

  VISKORES_EXEC viskores::UInt8 operator()(const viskores::Vec<viskores::Id, 4>& step) const
  {
    return step[0] >= 0 ? 1 : 0;
  }
};

struct StepKey : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn step, FieldOut key, FieldOut realisedBy);
  using ExecutionSignature = void(_1, _2, _3);

  VISKORES_EXEC void operator()(const viskores::Vec<viskores::Id, 4>& step,
                                viskores::Id2& key,
                                viskores::Id2& realisedBy) const
  {
    key = viskores::Id2(step[0], step[1]);
    realisedBy = viskores::Id2(step[2], step[3]);
  }
};

/// The three sorted edges of an output triangle.
struct TriangleEdges : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn triangle, FieldOut edges);
  using ExecutionSignature = void(_1, _2);

  template <typename EdgesVec>
  VISKORES_EXEC void operator()(const viskores::Id3& t, EdgesVec& edges) const
  {
    for (viskores::IdComponent s = 0; s < 3; ++s)
    {
      const viskores::Id a = t[s], b = t[(s + 1) % 3];
      edges[s] = viskores::Id2(viskores::Min(a, b), viskores::Max(a, b));
    }
  }
};

/// For an edge of `K_X`: its glued edge, and the edges of `X` that realise it, as a count
/// (two meaning two or more) and the one edge when there is exactly one.
struct GlueEdge : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn edge,
                                WholeArrayIn glue,
                                WholeArrayIn stepKeys,
                                WholeArrayIn stepRealised,
                                FieldOut glued,
                                FieldOut count,
                                FieldOut realisedBy);
  using ExecutionSignature = void(_1, _2, _3, _4, _5, _6, _7);

  template <typename IdPortal, typename Id2Portal>
  VISKORES_EXEC void operator()(const viskores::Id2& edge,
                                const IdPortal& glue,
                                const Id2Portal& stepKeys,
                                const Id2Portal& stepRealised,
                                viskores::Id2& glued,
                                viskores::IdComponent& count,
                                viskores::Id2& realisedBy) const
  {
    const viskores::Id ga = glue.Get(edge[0]), gb = glue.Get(edge[1]);
    glued = viskores::Id2(viskores::Min(ga, gb), viskores::Max(ga, gb));
    // The steps are sorted: find the first with this key.
    viskores::Id lo = 0, hi = stepKeys.GetNumberOfValues();
    while (lo < hi)
    {
      const viskores::Id mid = (lo + hi) / 2;
      if (stepKeys.Get(mid) < edge)
      {
        lo = mid + 1;
      }
      else
      {
        hi = mid;
      }
    }
    count = 0;
    realisedBy = viskores::Id2(-1, -1);
    for (viskores::Id s = lo; s < stepKeys.GetNumberOfValues() && stepKeys.Get(s) == edge && count < 2;
         ++s)
    {
      ++count;
      realisedBy = stepRealised.Get(s);
    }
  }
};

/// (d') on the edges of `K_X` sorted by glued edge: an edge joining two identified vertices,
/// and every edge of a glued edge with two or more edges of `K_X` unless all are realised by one
/// and the same edge of `X` and by no other, list their labels.
struct GlueEdgeViolation : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn index,
                                WholeArrayIn edges,
                                WholeArrayIn glued,
                                WholeArrayIn counts,
                                WholeArrayIn realised,
                                FieldOut listed);
  using ExecutionSignature = void(_1, _2, _3, _4, _5, _6);

  template <typename Id2Portal, typename CountPortal, typename ListedVec>
  VISKORES_EXEC void operator()(viskores::Id i,
                                const Id2Portal& edges,
                                const Id2Portal& glued,
                                const CountPortal& counts,
                                const Id2Portal& realised,
                                ListedVec& listed) const
  {
    const viskores::Id2 e = edges.Get(i);
    const viskores::Id2 g = glued.Get(i);
    bool bad = g[0] == g[1];
    if (!bad)
    {
      viskores::Id first = i, last = i + 1;
      while (first > 0 && glued.Get(first - 1) == g)
      {
        --first;
      }
      while (last < glued.GetNumberOfValues() && glued.Get(last) == g)
      {
        ++last;
      }
      if (last - first > 1)
      {
        for (viskores::Id m = first; m < last && !bad; ++m)
        {
          bad = counts.Get(m) != 1 || realised.Get(m) != realised.Get(first);
        }
      }
    }
    listed[0] = bad ? e[0] : viskores::Id(-1);
    listed[1] = bad ? e[1] : viskores::Id(-1);
  }
};

struct GlueSortedTriangle : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn triangle, WholeArrayIn glue, FieldOut glued);
  using ExecutionSignature = _3(_1, _2);

  template <typename GluePortal>
  VISKORES_EXEC viskores::Id3 operator()(const viskores::Id3& t, const GluePortal& glue) const
  {
    viskores::Id a = glue.Get(t[0]), b = glue.Get(t[1]), c = glue.Get(t[2]);
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
};

/// (d') on the triangles of `K_X` sorted by glued triangle: two with one image list the labels
/// of both.
struct GlueTriangleViolation : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn index,
                                WholeArrayIn glued,
                                WholeArrayIn order,
                                WholeArrayIn triangles,
                                FieldOut listed);
  using ExecutionSignature = void(_1, _2, _3, _4, _5);

  template <typename Id3Portal, typename IdPortal, typename ListedVec>
  VISKORES_EXEC void operator()(viskores::Id i,
                                const Id3Portal& glued,
                                const IdPortal& order,
                                const Id3Portal& triangles,
                                ListedVec& listed) const
  {
    const viskores::Id3 g = glued.Get(i);
    const bool bad = (i > 0 && glued.Get(i - 1) == g) ||
      (i + 1 < glued.GetNumberOfValues() && glued.Get(i + 1) == g);
    const viskores::Id3 t = triangles.Get(order.Get(i));
    for (viskores::IdComponent k = 0; k < 3; ++k)
    {
      listed[k] = bad ? t[k] : viskores::Id(-1);
    }
  }
};

} // namespace detail

/// The labels involved in a violation of (d') of Proposition "Decimating the singular curves",
/// as `glue_violations` of proofs/lib/arcs.py, sorted and without repetition: the
/// identification of `K_X` onto `K` given by `glue` may identify two vertices only if they are
/// pieces of one segment or copies of one junction, two edges only if both are realised by one
/// edge of `X` and by no other, and no two triangles, and no edge of `K_X` may join two vertices
/// it identifies.
inline viskores::cont::ArrayHandle<viskores::Id> GlueViolations(
  const SurfaceMesh& mesh,
  const viskores::cont::ArrayHandle<viskores::Id>& labels,
  const viskores::cont::ArrayHandle<viskores::Id>& original,
  const SingularCurves& curves,
  const viskores::cont::ArrayHandle<viskores::Id>& glue)
{
  using Algorithm = viskores::cont::Algorithm;
  viskores::cont::Invoker invoke;
  // The edges of X that realise every edge of K_X, from the cut edges between clusters.
  viskores::cont::ArrayHandle<viskores::Vec<viskores::Id, 4>> steps, valid;
  invoke(detail::CutEdgeStep{}, mesh.Edges, curves.CutEdge, labels, original, steps);
  viskores::cont::ArrayHandle<viskores::UInt8> isStep;
  invoke(detail::IsStep{}, steps, isStep);
  Algorithm::CopyIf(steps, isStep, valid);
  Algorithm::Sort(valid);
  Algorithm::Unique(valid);
  viskores::cont::ArrayHandle<viskores::Id2> stepKeys, stepRealised;
  invoke(detail::StepKey{}, valid, stepKeys, stepRealised);

  const viskores::cont::ArrayHandle<viskores::Id3> triangles = OutputTriangles(mesh, labels);
  viskores::cont::ArrayHandle<viskores::Id> listed;
  if (triangles.GetNumberOfValues() == 0)
  {
    return listed;
  }
  viskores::cont::ArrayHandle<viskores::Id2> edges;
  invoke(detail::TriangleEdges{}, triangles, viskores::cont::make_ArrayHandleGroupVec<3>(edges));
  Algorithm::Sort(edges);
  Algorithm::Unique(edges);
  viskores::cont::ArrayHandle<viskores::Id2> glued, realised;
  viskores::cont::ArrayHandle<viskores::IdComponent> counts;
  invoke(detail::GlueEdge{}, edges, glue, stepKeys, stepRealised, glued, counts, realised);
  // Sorted by glued edge, the edges of K_X with one image are runs.
  viskores::cont::ArrayHandle<viskores::Id> order;
  Algorithm::Copy(viskores::cont::ArrayHandleIndex(edges.GetNumberOfValues()), order);
  Algorithm::SortByKey(glued, order);
  viskores::cont::ArrayHandle<viskores::Id2> sortedEdges, sortedRealised;
  viskores::cont::ArrayHandle<viskores::IdComponent> sortedCounts;
  Algorithm::Copy(viskores::cont::make_ArrayHandlePermutation(order, edges), sortedEdges);
  Algorithm::Copy(viskores::cont::make_ArrayHandlePermutation(order, counts), sortedCounts);
  Algorithm::Copy(viskores::cont::make_ArrayHandlePermutation(order, realised), sortedRealised);
  viskores::cont::ArrayHandle<viskores::Id> edgeListed;
  invoke(detail::GlueEdgeViolation{},
         viskores::cont::ArrayHandleIndex(edges.GetNumberOfValues()),
         sortedEdges,
         glued,
         sortedCounts,
         sortedRealised,
         viskores::cont::make_ArrayHandleGroupVec<2>(edgeListed));
  // Two triangles of K_X with one image.
  viskores::cont::ArrayHandle<viskores::Id3> gluedTriangles;
  invoke(detail::GlueSortedTriangle{}, triangles, glue, gluedTriangles);
  viskores::cont::ArrayHandle<viskores::Id> triangleOrder, triangleListed;
  Algorithm::Copy(viskores::cont::ArrayHandleIndex(triangles.GetNumberOfValues()), triangleOrder);
  Algorithm::SortByKey(gluedTriangles, triangleOrder);
  invoke(detail::GlueTriangleViolation{},
         viskores::cont::ArrayHandleIndex(triangles.GetNumberOfValues()),
         gluedTriangles,
         triangleOrder,
         triangles,
         viskores::cont::make_ArrayHandleGroupVec<3>(triangleListed));
  viskores::cont::ArrayHandle<viskores::Id> all;
  all.Allocate(edgeListed.GetNumberOfValues() + triangleListed.GetNumberOfValues());
  Algorithm::CopySubRange(edgeListed, 0, edgeListed.GetNumberOfValues(), all, 0);
  Algorithm::CopySubRange(
    triangleListed, 0, triangleListed.GetNumberOfValues(), all, edgeListed.GetNumberOfValues());
  viskores::cont::ArrayHandle<viskores::UInt8> isListed;
  invoke(detail::IsValidLabel{}, all, isListed);
  Algorithm::CopyIf(all, isListed, listed);
  Algorithm::Sort(listed);
  Algorithm::Unique(listed);
  return listed;
}

/// The sum of `values` as numpy sums a float64 array, `0 + pairwise(values)`: blocks of fewer
/// than eight values left to right, blocks of up to 128 in eight interleaved partial sums, and
/// larger ranges split in two at a multiple of eight. A fixed order, for sums that feed
/// decisions.
inline Real PairwiseSum(const std::vector<Real>& values, std::size_t first, std::size_t count)
{
  if (count < 8)
  {
    Real result = 0;
    for (std::size_t i = 0; i < count; ++i)
    {
      result += values[first + i];
    }
    return result;
  }
  if (count <= 128)
  {
    Real r[8];
    for (std::size_t j = 0; j < 8; ++j)
    {
      r[j] = values[first + j];
    }
    std::size_t i = 8;
    for (; i < count - (count % 8); i += 8)
    {
      for (std::size_t j = 0; j < 8; ++j)
      {
        r[j] += values[first + i + j];
      }
    }
    Real result = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
    for (; i < count; ++i)
    {
      result += values[first + i];
    }
    return result;
  }
  std::size_t half = count / 2;
  half -= half % 8;
  return PairwiseSum(values, first, half) + PairwiseSum(values, first + half, count - half);
}

inline Real NumpySum(const viskores::cont::ArrayHandle<Real>& array)
{
  std::vector<Real> values(static_cast<std::size_t>(array.GetNumberOfValues()));
  auto portal = array.ReadPortal();
  for (std::size_t i = 0; i < values.size(); ++i)
  {
    values[i] = portal.Get(static_cast<viskores::Id>(i));
  }
  return Real(0) + PairwiseSum(values, 0, values.size());
}

}
}
} // namespace viskores::worklet::quadric_clustering

VISKORES_QUADRIC_CLUSTERING_STRICT_FP_END

#endif // viskores_worklet_quadric_clustering_SingularCurves_h
