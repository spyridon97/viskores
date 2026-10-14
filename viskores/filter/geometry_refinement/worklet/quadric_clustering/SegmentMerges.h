//============================================================================
//  The contents of this file are covered by the Viskores license. See
//  LICENSE.txt for details.
//
//  By contributing to this file, all contributors agree to the Developer
//  Certificate of Origin Version 1.1 (DCO 1.1) as stated in DCO.txt.
//============================================================================
#ifndef viskores_worklet_quadric_clustering_SegmentMerges_h
#define viskores_worklet_quadric_clustering_SegmentMerges_h

#include <viskores/filter/geometry_refinement/worklet/quadric_clustering/MergePhase.h>
#include <viskores/filter/geometry_refinement/worklet/quadric_clustering/SingularCurves.h>

#include <viskores/cont/ArrayHandle.h>
#include <viskores/worklet/WorkletMapField.h>

#include <algorithm>
#include <limits>
#include <map>
#include <set>
#include <tuple>
#include <vector>

VISKORES_QUADRIC_CLUSTERING_STRICT_FP_BEGIN

namespace viskores
{
namespace worklet
{
namespace quadric_clustering
{

namespace detail
{

/// Replaces the labels `from[i]` by `to[i]`.
struct Relabel : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn label, WholeArrayIn from, WholeArrayIn to, FieldOut relabelled);
  using ExecutionSignature = _4(_1, _2, _3);

  template <typename IdPortal>
  VISKORES_EXEC viskores::Id operator()(viskores::Id label,
                                        const IdPortal& from,
                                        const IdPortal& to) const
  {
    for (viskores::Id i = 0; i < from.GetNumberOfValues(); ++i)
    {
      if (from.Get(i) == label)
      {
        return to.Get(i);
      }
    }
    return label;
  }
};

/// Whether an input triangle has a corner among the changed copies, or a box that meets the box
/// `[Lower, Upper]`.
struct ScopeTriangle : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn triangle, WholeArrayIn points, WholeArrayIn changed, FieldOut inScope);
  using ExecutionSignature = _4(_1, _2, _3);

  VISKORES_CONT ScopeTriangle(const Vec3& lower, const Vec3& upper)
    : Lower(lower)
    , Upper(upper)
  {
  }

  template <typename PointPortal, typename FlagPortal>
  VISKORES_EXEC viskores::UInt8 operator()(const viskores::Id3& t,
                                           const PointPortal& points,
                                           const FlagPortal& changed) const
  {
    Vec3 lo = points.Get(t[0]), hi = lo;
    bool touched = changed.Get(t[0]) != 0;
    for (viskores::IdComponent k = 1; k < 3; ++k)
    {
      const Vec3 p = points.Get(t[k]);
      touched = touched || changed.Get(t[k]) != 0;
      for (viskores::IdComponent a = 0; a < 3; ++a)
      {
        lo[a] = viskores::Min(lo[a], p[a]);
        hi[a] = viskores::Max(hi[a], p[a]);
      }
    }
    bool meets = true;
    for (viskores::IdComponent a = 0; a < 3; ++a)
    {
      meets = meets && lo[a] <= this->Upper[a] && hi[a] >= this->Lower[a];
    }
    return (touched || meets) ? 1 : 0;
  }

private:
  Vec3 Lower;
  Vec3 Upper;
};

} // namespace detail

/// The merges of consecutive segments of the singular curves, (f) of Proposition "Decimating
/// the singular curves", as `ArcVQC` of proofs/lib/arcs.py.
class SegmentMerger
{
public:
  SegmentMerger(const SurfaceMesh& mesh,
                const VertexQuadrics& quadrics,
                const DistanceMesh& inputDistance,
                const MergeParameters& parameters,
                SingularCurves& curves)
    : Mesh(mesh)
    , Quadrics(quadrics)
    , InputDistance(inputDistance)
    , Parameters(parameters)
    , Curves(curves)
  {
    this->Parameters.Curves = &curves;
    this->CopiesOf.resize(curves.Segment.size());
    {
      auto copyOf = parameters.Original.ReadPortal();
      for (viskores::Id c = 0; c < copyOf.GetNumberOfValues(); ++c)
      {
        this->CopiesOf[static_cast<std::size_t>(copyOf.Get(c))].push_back(c);
      }
    }
    // The copies of every edge of X, as edges of the cut mesh.
    auto edges = mesh.Edges.ReadPortal();
    auto cutEdge = curves.CutEdge.ReadPortal();
    auto original = parameters.Original.ReadPortal();
    for (viskores::Id h = 0; h < edges.GetNumberOfValues(); ++h)
    {
      if (cutEdge.Get(h))
      {
        const viskores::Id2 e = edges.Get(h);
        const viskores::Id oa = original.Get(e[0]), ob = original.Get(e[1]);
        this->Copies[std::make_pair(std::min(oa, ob), std::max(oa, ob))].push_back(e);
      }
    }
  }

  /// The merge phase of the proposition: the merge phase of the theorem, alternating with rounds
  /// of merges of consecutive segments until neither changes the labelling, as
  /// `ArcVQC.merge_phase`.
  template <typename RoundObserver>
  void Run(const viskores::cont::ArrayHandle<viskores::UInt8>& frozenVertices,
           Clustering& clusters,
           MergeDiagnostics& diagnostics,
           RoundObserver&& observer)
  {
    auto sheets = [&]
    {
      MergePhase(this->Mesh,
                 this->Quadrics,
                 this->InputDistance,
                 frozenVertices,
                 this->Parameters,
                 clusters,
                 diagnostics,
                 observer);
    };
    sheets();
    while (true)
    {
      bool segmentsMerged = false;
      while (this->Round(clusters, diagnostics))
      {
        segmentsMerged = true;
      }
      const viskores::Id before = clusters.NumberOfClusters;
      sheets();
      if (clusters.NumberOfClusters != before)
      {
        continue;
      }
      // The sheet merges changed nothing: one more round of segment merges decides.
      if (!segmentsMerged || !this->Round(clusters, diagnostics))
      {
        break;
      }
    }
  }

private:
  /// One round, as `arc_merge_round`: every merge of two consecutive segments of an arc that
  /// keeps every flag quiet is a candidate; the greedy set of candidates in increasing order of
  /// the increase of the objective, then of arc and position, that share no segment and are not
  /// consecutive, is applied and checked; if a flag fires, only the first is applied. Returns
  /// whether the labelling changed.
  bool Round(Clustering& clusters, MergeDiagnostics& diagnostics)
  {
    const Real energy = this->Energy(clusters);
    // The labelling passes (F5); a candidate changes its verdict only on the triangles near the
    // two segments, under the same scales.
    bool scoped = false;
    EnvelopeScope scope;
    viskores::cont::ArrayHandle<viskores::Id3> output;
    std::vector<std::vector<viskores::Id>> fans;
    if (this->Parameters.ErrorBound > 0)
    {
      output = OutputTriangles(this->Mesh, clusters.Labels);
      viskores::cont::ArrayHandle<viskores::UInt8> flags;
      flags.AllocateAndFill(clusters.NumberOfClusters, 0);
      const EnvelopeReport report = EnvelopeFlags(this->Mesh,
                                                  this->InputDistance,
                                                  clusters.Labels,
                                                  clusters.Generators,
                                                  output,
                                                  clusters.Sizes,
                                                  this->Parameters.ErrorBound,
                                                  EnvelopeMode::Surface,
                                                  this->Parameters.Sides,
                                                  flags);
      diagnostics.EnvelopePieces += report.Pieces;
      scoped = report.FailedInput + report.FailedOutput == 0;
      scope.ScaleInput = report.ScaleInput;
      scope.ScaleOutput = report.ScaleOutput;
      fans.resize(static_cast<std::size_t>(clusters.NumberOfClusters));
      auto triangles = output.ReadPortal();
      for (viskores::Id t = 0; t < triangles.GetNumberOfValues(); ++t)
      {
        for (viskores::IdComponent k = 0; k < 3; ++k)
        {
          fans[static_cast<std::size_t>(triangles.Get(t)[k])].push_back(t);
        }
      }
    }
    std::vector<std::tuple<Real, std::size_t, std::size_t>> candidates;
    for (std::size_t a = 0; a < this->Curves.Segments.size(); ++a)
    {
      const std::size_t ns = this->Curves.Segments[a].size();
      if (ns < 2)
      {
        continue;
      }
      const std::size_t count = this->Curves.Arcs[a].Closed ? ns : ns - 1;
      for (std::size_t s = 0; s < count; ++s)
      {
        const std::vector<viskores::Id> saved = this->Curves.Segment;
        const auto& segments = this->Curves.Segments[a];
        const auto& next = segments[(s + 1) % ns];
        viskores::cont::ArrayHandle<viskores::Id> labels = this->MergePieces(clusters.Labels, a, s);
        for (viskores::Id v : next)
        {
          this->Curves.Segment[static_cast<std::size_t>(v)] =
            this->Curves.Segment[static_cast<std::size_t>(segments[s].front())];
        }
        Clustering merged;
        merged.Labels = labels;
        PlaceGlued(this->Mesh, this->Quadrics, this->Parameters, merged);
        if (scoped)
        {
          scope.Triangles = this->Scope(clusters, output, fans, a, s);
        }
        const bool fires = FlagsFire(this->Mesh,
                                     this->Quadrics,
                                     this->InputDistance,
                                     merged,
                                     this->Parameters,
                                     diagnostics,
                                     scoped ? &scope : nullptr);
        ++diagnostics.SegmentCandidates;
        if (!fires)
        {
          const Real increase = this->Energy(merged) - energy;
          candidates.emplace_back(increase < 0 ? Real(0) : increase, a, s);
        }
        this->Curves.Segment = saved;
      }
    }
    if (candidates.empty())
    {
      return false;
    }
    std::sort(candidates.begin(), candidates.end());
    std::set<std::pair<std::size_t, std::size_t>> taken;
    std::vector<std::tuple<std::size_t, viskores::Id, viskores::Id>> commit;
    for (const auto& candidate : candidates)
    {
      const std::size_t a = std::get<1>(candidate), s = std::get<2>(candidate);
      const std::size_t ns = this->Curves.Segments[a].size();
      const std::size_t touch[4] = { s, (s + 1) % ns, (s + ns - 1) % ns, (s + 2) % ns };
      bool touches = false;
      for (std::size_t t : touch)
      {
        touches = touches || taken.count(std::make_pair(a, t)) > 0;
      }
      if (touches)
      {
        continue;
      }
      taken.insert(std::make_pair(a, s));
      taken.insert(std::make_pair(a, (s + 1) % ns));
      const auto& segments = this->Curves.Segments[a];
      commit.emplace_back(a, segments[s].front(), segments[(s + 1) % ns].front());
    }
    if (!this->Apply(commit, clusters, diagnostics))
    {
      commit.resize(1);
      if (!this->Apply(commit, clusters, diagnostics))
      {
        return false;
      }
    }
    ++diagnostics.SegmentRounds;
    return true;
  }

  /// `E = sum_v (c_L(v) - v)^T M_v (c_L(v) - v)`, summed as numpy sums.
  Real Energy(const Clustering& clusters) const
  {
    return NumpySum(Scan(this->Mesh, this->Quadrics.M, clusters).Costs);
  }

  /// The labels with segment `s` of arc `a` and the next one merged: in every sheet, the piece
  /// of the next segment joins the piece of `s` that a copy of the edge of `X` between them
  /// meets, as `_merge_segments`.
  viskores::cont::ArrayHandle<viskores::Id> MergePieces(const viskores::cont::ArrayHandle<viskores::Id>& labels,
                                                        std::size_t a,
                                                        std::size_t s) const
  {
    const auto& segments = this->Curves.Segments[a];
    const viskores::Id u = segments[s].back();
    const viskores::Id w = segments[(s + 1) % segments.size()].front();
    auto labelPortal = labels.ReadPortal();
    auto original = this->Parameters.Original.ReadPortal();
    std::map<viskores::Id, viskores::Id> relabel;
    for (viskores::Id2 e : this->Copies.at(std::make_pair(std::min(u, w), std::max(u, w))))
    {
      if (original.Get(e[0]) == w)
      {
        std::swap(e[0], e[1]);
      }
      relabel[labelPortal.Get(e[1])] = labelPortal.Get(e[0]);
    }
    std::vector<viskores::Id> from, to;
    for (const auto& r : relabel)
    {
      from.push_back(r.first);
      to.push_back(r.second);
    }
    viskores::cont::ArrayHandle<viskores::Id> result;
    viskores::cont::Invoker invoke;
    invoke(detail::Relabel{},
           labels,
           viskores::cont::make_ArrayHandle(from, viskores::CopyFlag::On),
           viskores::cont::make_ArrayHandle(to, viskores::CopyFlag::On),
           result);
    return result;
  }

  /// The input triangles whose envelope test the merge of segment `s` of arc `a` and the next
  /// one can change: those with a corner in the two segments, and those whose box meets the box
  /// of the output triangles at their pieces grown by epsilon, since the test of a triangle reads
  /// only the output triangles within epsilon of it and passes at least as well with more.
  viskores::cont::ArrayHandle<viskores::UInt8> Scope(const Clustering& clusters,
                                                     const viskores::cont::ArrayHandle<viskores::Id3>& output,
                                                     const std::vector<std::vector<viskores::Id>>& fans,
                                                     std::size_t a,
                                                     std::size_t s) const
  {
    const auto& segments = this->Curves.Segments[a];
    std::vector<viskores::Id> copies;
    for (const auto* segment : { &segments[s], &segments[(s + 1) % segments.size()] })
    {
      for (viskores::Id v : *segment)
      {
        const auto& c = this->CopiesOf[static_cast<std::size_t>(v)];
        copies.insert(copies.end(), c.begin(), c.end());
      }
    }
    viskores::cont::ArrayHandle<viskores::UInt8> changed;
    changed.AllocateAndFill(this->Mesh.NumberOfPoints, 0);
    Vec3 lo(std::numeric_limits<Real>::infinity()), hi(-std::numeric_limits<Real>::infinity());
    {
      auto changedPortal = changed.WritePortal();
      auto labels = clusters.Labels.ReadPortal();
      auto generators = clusters.Generators.ReadPortal();
      auto triangles = output.ReadPortal();
      std::set<viskores::Id> pieces;
      for (viskores::Id c : copies)
      {
        changedPortal.Set(c, 1);
        pieces.insert(labels.Get(c));
      }
      for (viskores::Id piece : pieces)
      {
        for (viskores::Id t : fans[static_cast<std::size_t>(piece)])
        {
          for (viskores::IdComponent k = 0; k < 3; ++k)
          {
            const Vec3 p = generators.Get(triangles.Get(t)[k]);
            for (viskores::IdComponent i = 0; i < 3; ++i)
            {
              lo[i] = viskores::Min(lo[i], p[i]);
              hi[i] = viskores::Max(hi[i], p[i]);
            }
          }
        }
      }
    }
    Real size = 0;
    for (viskores::IdComponent i = 0; i < 3; ++i)
    {
      size = viskores::Max(size, viskores::Max(viskores::Abs(lo[i]), viskores::Abs(hi[i])));
    }
    if (lo[0] <= hi[0])
    {
      const Real grow = this->Parameters.ErrorBound * (1 + Real(1e-9)) + 64 * UnitRoundoff * size;
      for (viskores::IdComponent i = 0; i < 3; ++i)
      {
        lo[i] = lo[i] - grow;
        hi[i] = hi[i] + grow;
      }
    }
    viskores::cont::ArrayHandle<viskores::UInt8> inScope;
    viskores::cont::Invoker invoke;
    invoke(detail::ScopeTriangle(lo, hi), this->Mesh.Triangles, this->Mesh.Points, changed, inScope);
    return inScope;
  }

  /// Applies the merges `batch`, each given by its arc and the first vertices of its two
  /// segments, and keeps them if no flag fires, as `apply` in `arc_merge_round`.
  bool Apply(const std::vector<std::tuple<std::size_t, viskores::Id, viskores::Id>>& batch,
             Clustering& clusters,
             MergeDiagnostics& diagnostics)
  {
    const std::vector<viskores::Id> savedSegment = this->Curves.Segment;
    const auto savedSegments = this->Curves.Segments;
    viskores::cont::ArrayHandle<viskores::Id> labels = clusters.Labels;
    for (const auto& merge : batch)
    {
      const std::size_t a = std::get<0>(merge);
      auto& segments = this->Curves.Segments[a];
      const std::size_t ns = segments.size();
      std::size_t s = 0;
      while (!(segments[s].front() == std::get<1>(merge) &&
               segments[(s + 1) % ns].front() == std::get<2>(merge)))
      {
        ++s;
      }
      labels = this->MergePieces(labels, a, s);
      const std::size_t j = (s + 1) % ns;
      for (viskores::Id v : segments[j])
      {
        this->Curves.Segment[static_cast<std::size_t>(v)] =
          this->Curves.Segment[static_cast<std::size_t>(segments[s].front())];
      }
      segments[s].insert(segments[s].end(), segments[j].begin(), segments[j].end());
      segments.erase(segments.begin() + static_cast<std::ptrdiff_t>(j));
    }
    Clustering merged;
    merged.Labels = labels;
    PlaceGlued(this->Mesh, this->Quadrics, this->Parameters, merged);
    if (FlagsFire(this->Mesh, this->Quadrics, this->InputDistance, merged, this->Parameters, diagnostics))
    {
      this->Curves.Segment = savedSegment;
      this->Curves.Segments = savedSegments;
      return false;
    }
    clusters = merged;
    return true;
  }

  const SurfaceMesh& Mesh;
  const VertexQuadrics& Quadrics;
  const DistanceMesh& InputDistance;
  MergeParameters Parameters;
  SingularCurves& Curves;
  std::map<std::pair<viskores::Id, viskores::Id>, std::vector<viskores::Id2>> Copies;
  std::vector<std::vector<viskores::Id>> CopiesOf;
};

}
}
} // namespace viskores::worklet::quadric_clustering

VISKORES_QUADRIC_CLUSTERING_STRICT_FP_END

#endif // viskores_worklet_quadric_clustering_SegmentMerges_h
