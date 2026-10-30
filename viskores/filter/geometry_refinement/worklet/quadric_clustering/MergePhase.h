//============================================================================
//  The contents of this file are covered by the Viskores license. See
//  LICENSE.txt for details.
//
//  By contributing to this file, all contributors agree to the Developer
//  Certificate of Origin Version 1.1 (DCO 1.1) as stated in DCO.txt.
//============================================================================
#ifndef viskores_worklet_quadric_clustering_MergePhase_h
#define viskores_worklet_quadric_clustering_MergePhase_h

#include <viskores/filter/geometry_refinement/worklet/quadric_clustering/Clustering.h>
#include <viskores/filter/geometry_refinement/worklet/quadric_clustering/CurveFlags.h>
#include <viskores/filter/geometry_refinement/worklet/quadric_clustering/EmbeddingFlags.h>
#include <viskores/filter/geometry_refinement/worklet/quadric_clustering/EnvelopeFlags.h>
#include <viskores/filter/geometry_refinement/worklet/quadric_clustering/GlueItems.h>
#include <viskores/filter/geometry_refinement/worklet/quadric_clustering/QuadricClusteringCommonWorklets.h>
#include <viskores/filter/geometry_refinement/worklet/quadric_clustering/SingularCurves.h>
#include <viskores/filter/geometry_refinement/worklet/quadric_clustering/ThickeningFlags.h>
#include <viskores/filter/geometry_refinement/worklet/quadric_clustering/TopologyFlags.h>
#include <viskores/filter/geometry_refinement/worklet/quadric_clustering/VertexQuadrics.h>

#include <viskores/Pair.h>
#include <viskores/cont/Algorithm.h>
#include <viskores/cont/ArrayHandleConcatenate.h>
#include <viskores/cont/ArrayHandleConstant.h>
#include <viskores/cont/ArrayHandleGroupVec.h>
#include <viskores/cont/ArrayHandleIndex.h>
#include <viskores/cont/ArrayHandlePermutation.h>
#include <viskores/cont/ArrayHandleView.h>
#include <viskores/cont/Timer.h>
#include <viskores/worklet/Keys.h>
#include <viskores/worklet/ScatterUniform.h>
#include <viskores/worklet/WorkletReduceByKey.h>

#include <limits>
#include <vector>

VISKORES_QUADRIC_CLUSTERING_STRICT_FP_BEGIN

namespace viskores
{
namespace worklet
{
namespace quadric_clustering
{

/// The key of the merge priority after the degree.
enum class MergeOrder
{
  /// The increase of the objective.
  Cost,
  /// `HashPair` of the anchors of the two clusters: a pseudo-random order within each degree
  /// class, which takes fewer passes per round.
  Hash
};

/// Which merges of a batch that fails the full test are deferred.
enum class MergeBlame
{
  /// Those whose merged cluster is flagged.
  Merged,
  /// Those, and those with a changed output triangle that meets the region read by a failing item
  /// that flags no merged cluster (`merge_blame="region"` of proofs/lib/vqc.py).
  Region
};

/// What the merge test and the full test of a merge check, and the frozen labels.
struct MergeParameters
{
  bool GeometricFlags() const
  {
    return this->Embedding || this->ErrorBound > 0 || this->CertifiedThickening != nullptr ||
      this->MaxClusterRadius < std::numeric_limits<Real>::infinity();
  }

  /// `k*`, or 0 in error mode.
  viskores::Id Budget = 0;
  Real MaxClusterRadius = std::numeric_limits<Real>::infinity();
  bool Embedding = true;
  Real ErrorBound = 0;
  /// The halves of the envelope flag (F5) in use.
  EnvelopeSides Sides = EnvelopeSides::Both;
  /// The certified thickening of the flag (F7), or null without it.
  const Thickening* CertifiedThickening = nullptr;
  /// Whether candidates must keep the discrete closed-ball property; off in the geometric-only
  /// mode, where every pair of adjacent clusters is a candidate and the output is the set of
  /// simplices.
  bool Topological = true;
  /// (D2) by count of the clusters per closed component instead of by sort of the label triples.
  bool D2ByCount = false;
  /// The floor `D_0` of the degree key of the priority, or a negative value to order the
  /// candidates by the increase of the objective alone.
  viskores::Id DegreeFloor = 12;
  /// The key after the degree.
  MergeOrder Order = MergeOrder::Cost;
  /// Experimental: a cluster of a sheet may merge into an adjacent piece of a segment or copy of a
  /// junction on the boundary of the cut, which keeps its label and its generator. Such merges
  /// take the local test with that generator fixed, and (d') at the glued vertex.
  bool FaceMerges = false;
  /// Experimental: the end segment of an open arc may merge into the junction at that end, whose
  /// generator stays at the junction.
  bool JunctionMerges = false;
  /// When the committed set fails the full test: defer the merges whose merged cluster is
  /// flagged and test the others again, or, if false, merge only the candidate of highest
  /// priority.
  bool DeferFlagged = true;
  /// Which merges `DeferFlagged` defers.
  MergeBlame Blame = MergeBlame::Region;
  /// For an input cut into surfaces, the first copy of the input vertex of every copy, which
  /// glues the output for (F4), and the frozen copies; empty otherwise.
  viskores::cont::ArrayHandle<viskores::Id> Canonical;
  viskores::cont::ArrayHandle<viskores::UInt8> FrozenVertices;
  /// With the singular curves decimated (Proposition "Decimating the singular curves"): their
  /// segments, which glue the output in place of `Canonical`, with (d') in place of (d) in the
  /// full test, and the input vertex of every copy; null otherwise.
  const SingularCurves* Curves = nullptr;
  viskores::cont::ArrayHandle<viskores::Id> Original;
  /// With the curves decimated: the tolerance `eps_c` of the curve test (Fc) of Proposition
  /// "curvetol", or 0 for none, and its subdivision depth.
  Real CurveTolerance = 0;
  viskores::IdComponent CurveDepth = 10;
};

struct MergeDiagnostics
{
  viskores::Id Rounds = 0;
  /// Rounds in which the committed set failed the full test and one merge was made instead.
  viskores::Id Fallbacks = 0;
  /// Merges deferred to a later round because their merged cluster was flagged.
  viskores::Id Deferred = 0;
  /// Of those, the merges deferred by the region blame alone.
  viskores::Id BlameDeferred = 0;
  /// Local tests evaluated, and verdicts of earlier rounds reused instead.
  viskores::Id Evaluated = 0;
  viskores::Id Reused = 0;
  /// With the singular curves decimated: the rounds of segment merges, and the candidates
  /// tested.
  viskores::Id SegmentRounds = 0;
  viskores::Id SegmentCandidates = 0;
  /// The rounds of segment merges that merged a candidate deferred as a culprit, tried alone.
  viskores::Id SegmentRetried = 0;
  /// Wall-clock seconds: the local tests, and the full tests of the merged labellings.
  double SecondsLocalTests = 0;
  double SecondsFullTests = 0;
  /// Per round: the merges committed, the merges deferred, and the wall-clock seconds.
  std::vector<viskores::Id> RoundMerges;
  std::vector<viskores::Id> RoundDeferred;
  std::vector<double> RoundSeconds;
};

/// The largest absolute coordinate of the generators with clusters `i` and `j` replaced by one
/// at `mergedAbs`: the one term of a local test that reads the whole labelling. It keeps the
/// three largest absolute coordinates of the generators, with their clusters.
struct MergedScale
{
  VISKORES_EXEC_CONT Real operator()(viskores::Id i, viskores::Id j, Real mergedAbs) const
  {
    for (int slot = 0; slot < 3; ++slot)
    {
      if (this->TopLabel[slot] >= 0 && this->TopLabel[slot] != i && this->TopLabel[slot] != j)
      {
        return viskores::Max(mergedAbs, this->TopAbs[slot]);
      }
    }
    return mergedAbs;
  }

  viskores::Vec<Real, 3> TopAbs;
  viskores::Vec<viskores::Id, 3> TopLabel;
};

/// The data the local test of a merge reads, in the execution environment.
struct MergeContextPortals
{
  using VecPortal = viskores::cont::ArrayHandle<Vec3>::ReadPortalType;
  using IdPortal = viskores::cont::ArrayHandle<viskores::Id>::ReadPortalType;
  using Id3Portal = viskores::cont::ArrayHandle<viskores::Id3>::ReadPortalType;
  using MatPortal = viskores::cont::ArrayHandle<Mat3>::ReadPortalType;
  using FlagPortal = viskores::cont::ArrayHandle<viskores::UInt8>::ReadPortalType;
  using Distance = DistanceMeshPortals<
    VecPortal,
    Id3Portal,
    viskores::worklet::spatialstructure::BoundingVolumeHierarchy::ExecObjectType>;

  /// Calls `visit(P, labels, t)` for every triangle of `K/ij` at the glued vertex of a frozen
  /// cluster `i` that the sheet cluster `j` joins, with `j` at the generator `merged` of `i`,
  /// which does not move: the triangles at a label of the glue group of `i` or at `j`,
  /// relabelled, each once, but not those at both `i` and `j`, which degenerate.
  template <typename Visit>
  VISKORES_EXEC bool ForEachSheetMergeTriangle(viskores::Id i,
                                               viskores::Id j,
                                               const Vec3& merged,
                                               Visit&& visit) const
  {
    const viskores::Id group = this->Glue.Get(i);
    const viskores::Id g =
      LowerBound(this->GroupKeys, 0, this->GroupKeys.GetNumberOfValues(), group);
    const viskores::Id begin = this->GroupOffsets.Get(g), end = this->GroupOffsets.Get(g + 1);
    // The group's labels in increasing order, then j; a triangle is visited from the first.
    for (viskores::Id r = begin; r <= end; ++r)
    {
      const viskores::Id c = (r < end) ? this->GroupLabels.Get(r) : j;
      for (viskores::Id f = this->FanOffsets.Get(c); f < this->FanOffsets.Get(c + 1); ++f)
      {
        const viskores::Id t = this->Fans.Get(f);
        viskores::Id3 labels = this->OutputTriangles.Get(t);
        bool earlier = false, hasI = false, hasJ = false;
        for (int k = 0; k < 3; ++k)
        {
          const bool inGroup = this->Glue.Get(labels[k]) == group;
          earlier = earlier || (inGroup && (c == j || labels[k] < c));
          hasI = hasI || labels[k] == i;
          hasJ = hasJ || labels[k] == j;
        }
        if (earlier || (hasI && hasJ))
        {
          continue;
        }
        for (int k = 0; k < 3; ++k)
        {
          labels[k] = (labels[k] == j) ? i : labels[k];
        }
        Vec3 p[3];
        for (int k = 0; k < 3; ++k)
        {
          p[k] = (labels[k] == i) ? merged : this->Generators.Get(labels[k]);
        }
        if (!visit(p, labels, t))
        {
          return false;
        }
      }
    }
    return true;
  }

  VISKORES_EXEC viskores::Id3 Glued(const viskores::Id3& t) const
  {
    if (!this->HasGlue)
    {
      return t;
    }
    return viskores::Id3(this->Glue.Get(t[0]), this->Glue.Get(t[1]), this->Glue.Get(t[2]));
  }

  /// (F4) for a sheet cluster `j` joining a frozen cluster `i`, as `merge_ok(fixed=True)` of
  /// proofs/lib/vqc.py: every triangle at the glued vertex of `i` must be nondegenerate and
  /// meet no other nondegenerate triangle improperly.
  VISKORES_EXEC bool SheetMergeEmbedded(viskores::Id i,
                                        viskores::Id j,
                                        const Vec3& merged,
                                        bool& complete) const
  {
    const viskores::Id group = this->Glue.Get(i);
    bool bad = false;
    this->ForEachSheetMergeTriangle(
      i,
      j,
      merged,
      [&](const Vec3* p, const viskores::Id3& labels, viskores::Id t)
      {
        if (viskores::TriangleIsDegenerate(p[0], p[1], p[2]))
        {
          bad = true;
          return false;
        }
        // Against the other triangles at the glued vertex, each pair once.
        bool after = false;
        this->ForEachSheetMergeTriangle(
          i,
          j,
          merged,
          [&](const Vec3* q, const viskores::Id3& otherLabels, viskores::Id u)
          {
            if (u == t)
            {
              after = true;
              return true;
            }
            if (after && !viskores::TriangleIsDegenerate(q[0], q[1], q[2]) &&
                exact::ImproperPair(p, this->Glued(labels), q, this->Glued(otherLabels)))
            {
              bad = true;
              return false;
            }
            return true;
          });
        if (bad)
        {
          return false;
        }
        // Against the triangles of K away from the glued vertex and from j, which stay.
        Vec3 lo, hi;
        TriangleBounds(p[0], p[1], p[2], lo, hi);
        const bool done = this->Output.Hierarchy.ForEachOverlap(
          lo,
          hi,
          [&](viskores::Id u)
          {
            const viskores::Id3 other = this->OutputTriangles.Get(u);
            for (int k = 0; k < 3; ++k)
            {
              if (other[k] == j || this->Glue.Get(other[k]) == group)
              {
                return true;
              }
            }
            if (this->OutputDegenerate.Get(u))
            {
              return true;
            }
            const Vec3 q[3] = { this->Generators.Get(other[0]),
                                this->Generators.Get(other[1]),
                                this->Generators.Get(other[2]) };
            if (exact::ImproperPair(p, this->Glued(labels), q, this->Glued(other)))
            {
              bad = true;
              return false;
            }
            return true;
          });
        complete = complete && done;
        return !bad;
      });
    return !bad;
  }

  /// Calls `visit(P, labels, t)` for every triangle of `K/ij` at the merged vertex `i`, realised
  /// with `i` at `merged`: the triangles at `i` or `j`, relabelled, but not those at both, which
  /// degenerate. They are visited in the order of the fans of `i`, then of `j`. In the
  /// geometric-only mode the output is the set of simplices, sorted triples with repetition, and
  /// those at both are kept as lower-dimensional simplices.
  template <typename Visit>
  VISKORES_EXEC bool ForEachNewTriangle(viskores::Id i,
                                        viskores::Id j,
                                        const Vec3& merged,
                                        Visit&& visit) const
  {
    for (int which = 0; which < 2; ++which)
    {
      const viskores::Id c = (which == 0) ? i : j;
      for (viskores::Id f = this->FanOffsets.Get(c); f < this->FanOffsets.Get(c + 1); ++f)
      {
        const viskores::Id t = this->Fans.Get(f);
        viskores::Id3 labels = this->OutputTriangles.Get(t);
        bool hasI = false, hasJ = false;
        for (int k = 0; k < 3; ++k)
        {
          hasI = hasI || labels[k] == i;
          hasJ = hasJ || labels[k] == j;
        }
        if (hasI && hasJ && !this->GeometricOnly)
        {
          continue;
        }
        for (int k = 0; k < 3; ++k)
        {
          if (labels[k] == j)
          {
            labels[k] = i;
          }
        }
        if (this->GeometricOnly)
        {
          labels = SortThree(labels);
        }
        Vec3 p[3];
        for (int k = 0; k < 3; ++k)
        {
          p[k] = (labels[k] == i) ? merged : this->Generators.Get(labels[k]);
        }
        if (!visit(p, labels, t))
        {
          return false;
        }
      }
    }
    return true;
  }

  VISKORES_EXEC bool Incident(viskores::Id t, viskores::Id i, viskores::Id j) const
  {
    const viskores::Id3 tri = this->OutputTriangles.Get(t);
    for (int c = 0; c < 3; ++c)
    {
      if (tri[c] == i || tri[c] == j)
      {
        return true;
      }
    }
    return false;
  }

  /// (F7) for a merge `(i, j)` with the merged generator at `merged`, on the 3-label input
  /// triangles with a vertex in the merged cluster, as `ThickVQC.iso_pass` of
  /// proofs/lib/thick.py on the labelling with the merge: the output triangle misses the sides,
  /// the component of the merged cluster has its first generator inside T, and no probe that
  /// fails on that labelling meets the triangle or has the label of a vertex of its triangle.
  VISKORES_EXEC bool ThickeningAdmissible(viskores::Id i,
                                          viskores::Id j,
                                          const Vec3& merged,
                                          bool& complete) const
  {
    const ThickeningRoundPortals& round = this->RoundThickening;
    bool fits = true;
    // The vertex of the glued output of a label; j is never glued, and i only when it is frozen.
    auto glued = [&](viskores::Id label) { return this->HasGlue ? this->Glue.Get(label) : label; };
    const viskores::Id gi = glued(i);
    // The component, without j: its first vertex and whether its generator is inside T.
    bool inside = true;
    const viskores::Id root = round.Root.Get(gi);
    if (root >= 0)
    {
      viskores::Id first = root;
      bool firstInside = round.InsideFirst.Get(root) != 0;
      if (first == j)
      {
        first = round.Second.Get(root);
        firstInside = round.InsideSecond.Get(root) != 0;
      }
      inside = (first == gi) ? round.Thick.Inside(merged, complete, fits) : firstInside;
    }
    auto atI = [&](const viskores::Id3& tri) { return tri[0] == i || tri[1] == i || tri[2] == i; };
    // Whether probe p fails on the labelling with the merge: its crossings with the output
    // triangles at i or j are replaced by those with the triangles at the merged vertex, once
    // each.
    auto probeFails = [&](viskores::Id p)
    {
      viskores::Id crossings = round.ProbeCrossings.Get(p);
      viskores::Id nongeneric = round.ProbeNongeneric.Get(p);
      auto count = [&](const Vec3* q, viskores::Id sign)
      {
        const thickening_detail::SegmentHit hit = round.Thick.ProbeHit(p, q, fits);
        if (hit.Hit)
        {
          crossings += sign;
          nongeneric += hit.Generic ? 0 : sign;
        }
      };
      for (int which = 0; which < 2; ++which)
      {
        const viskores::Id c = (which == 0) ? i : j;
        for (viskores::Id f = this->FanOffsets.Get(c); f < this->FanOffsets.Get(c + 1); ++f)
        {
          const viskores::Id3 tri = this->OutputTriangles.Get(this->Fans.Get(f));
          if (which == 1 && atI(tri))
          {
            continue;
          }
          const Vec3 q[3] = { this->Generators.Get(tri[0]),
                              this->Generators.Get(tri[1]),
                              this->Generators.Get(tri[2]) };
          count(q, -1);
        }
      }
      // The triangles at the merged vertex, a triangle of j dropped when i has one on the same
      // labels.
      this->ForEachNewTriangle(
        i,
        j,
        merged,
        [&](const Vec3* q, const viskores::Id3& labels, viskores::Id t)
        {
          if (!atI(this->OutputTriangles.Get(t)))
          {
            for (viskores::Id f = this->FanOffsets.Get(i); f < this->FanOffsets.Get(i + 1); ++f)
            {
              const viskores::Id3 other = this->OutputTriangles.Get(this->Fans.Get(f));
              int same = 0;
              for (int a = 0; a < 3; ++a)
              {
                same +=
                  (other[a] == labels[0] || other[a] == labels[1] || other[a] == labels[2]) ? 1 : 0;
              }
              if (same == 3)
              {
                return true;
              }
            }
          }
          count(q, 1);
          return true;
        });
      return nongeneric > 0 || crossings % 2 == 0;
    };
    const viskores::Id numProbes = round.ProbeCrossings.GetNumberOfValues();
    // The verdicts of the first probes: 0 unknown, 1 passing, 2 failing.
    viskores::UInt8 verdicts[64];
    for (int p = 0; p < 64; ++p)
    {
      verdicts[p] = 0;
    }
    auto fails = [&](viskores::Id p)
    {
      if (p >= 64)
      {
        return probeFails(p);
      }
      if (verdicts[p] == 0)
      {
        verdicts[p] = probeFails(p) ? 2 : 1;
      }
      return verdicts[p] == 2;
    };
    bool bad = false;
    for (int which = 0; which < 2 && !bad; ++which)
    {
      const viskores::Id c = (which == 0) ? i : j;
      for (viskores::Id k = this->MemberOffsets.Get(c); !bad && k < this->MemberOffsets.Get(c + 1);
           ++k)
      {
        const viskores::Id v = this->Members.Get(k);
        for (viskores::Id r = this->CornerOffsets.Get(v);
             !bad && r < this->CornerOffsets.Get(v + 1);
             ++r)
        {
          const viskores::Id t = this->Corners.Get(r) % this->NumberOfInputTriangles;
          const viskores::Id3 tri = this->InputTriangles.Get(t);
          viskores::Id labels[3];
          for (int a = 0; a < 3; ++a)
          {
            labels[a] = this->Labels.Get(tri[a]);
            labels[a] = (labels[a] == j) ? i : labels[a];
          }
          if (!AllDistinct(labels[0], labels[1], labels[2]))
          {
            continue;
          }
          Vec3 p[3];
          for (int a = 0; a < 3; ++a)
          {
            p[a] = (labels[a] == i) ? merged : this->Generators.Get(labels[a]);
          }
          bad = !inside || round.Thick.MeetsSides(p, complete, fits) ||
            (round.Thick.Fibres && round.Thick.FibresFail(p, complete, fits));
          for (viskores::Id q = 0; q < numProbes && !bad; ++q)
          {
            if (!fails(q))
            {
              continue;
            }
            const viskores::Id3 f = this->InputTriangles.Get(round.Thick.ProbeTriangles.Get(q));
            for (int a = 0; a < 3 && !bad; ++a)
            {
              viskores::Id pl = this->Labels.Get(f[a]);
              pl = glued((pl == j) ? i : pl);
              bad = pl == glued(labels[0]) || pl == glued(labels[1]) || pl == glued(labels[2]);
            }
            bad = bad || round.Thick.ProbeHit(q, p, fits).Hit;
          }
        }
      }
    }
    complete = complete && fits;
    return !bad;
  }

  /// The local test of a merge of Section 11.5 but for the envelope: (F0) on the merged
  /// cluster, (F4) on the pairs with a triangle at the merged vertex, and (F7), with (F8) when
  /// the fibres are on, on the 3-label input triangles with a vertex in the merged cluster. Sets `merged` to the position of the
  /// merged generator; a sheet cluster joining a frozen cluster leaves its generator in place.
  /// (d') of such a merge is tested by `SheetMergeGlueTest`.
  VISKORES_EXEC bool Admissible(viskores::Id i, viskores::Id j, Vec3& merged, bool& complete) const;

  /// Visits the input triangles the envelope part of the local test reads, as `visit(t, tri,
  /// tests)`, with the tests each needs: bit 0, its simplex in the merged labelling against the
  /// input ((F5)(i), the 3-label triangles with a vertex in the merged cluster); bit 1, the
  /// triangle against `K/ij` ((F5)(ii), the triangles with a vertex in the merged cluster and
  /// those whose box meets the box of the removed output triangles grown by epsilon). Only the
  /// triangles with `D(t) > epsilon` are visited. In the geometric-only mode, both tests of
  /// (F5') on the triangles with a vertex in the merged cluster, the second against all the
  /// simplices of the merged labelling. `parts` selects the triangles with a vertex in the
  /// merged cluster (bit 0) and the others (bit 1).
  template <typename Visit>
  VISKORES_EXEC void ForEachEnvelopeItem(viskores::Id i,
                                         viskores::Id j,
                                         const Vec3& merged,
                                         viskores::UInt8 parts,
                                         bool& complete,
                                         Visit&& visit) const;

  /// The tests `tests` of `ForEachEnvelopeItem` on input triangle `t`. `scaleInput` and
  /// `scaleOutput` are `S (1 + 16 u)` of the two batches of tests of the merge.
  VISKORES_EXEC bool TestEnvelopeItem(viskores::Id i,
                                      viskores::Id j,
                                      const Vec3& merged,
                                      viskores::Id t,
                                      viskores::UInt8 tests,
                                      Real scaleInput,
                                      Real scaleOutput,
                                      bool& complete) const;

  VecPortal Points;
  Id3Portal InputTriangles;
  IdPortal CornerOffsets;
  IdPortal Corners;
  viskores::Id NumberOfInputTriangles;
  IdPortal Labels;
  IdPortal MemberOffsets;
  IdPortal Members;
  IdPortal Anchors;
  VecPortal Generators;
  /// The matrices `M_v` of the vertices, for the generator of a merged cluster.
  MatPortal VertexMatrices;
  Id3Portal OutputTriangles;
  IdPortal FanOffsets;
  IdPortal Fans;
  FlagPortal OutputDegenerate;
  Distance Input;
  Distance Output;
  IdPortal Glue;
  bool HasGlue;
  /// With sheet merges: the frozen clusters, which a cluster of a sheet joins without moving
  /// their generator, and the labels of every glue group, sorted by glue id.
  FlagPortal FrozenLabels;
  bool HasFrozen;
  IdPortal GroupKeys;
  IdPortal GroupOffsets;
  IdPortal GroupLabels;
  bool GeometricOnly;
  Real MaxRadius;
  Real Epsilon;
  bool Embedding;
  /// The halves of (F5) in use, as the bits of the tests of `ForEachEnvelopeItem`.
  viskores::UInt8 SideTests;
  /// With (F7): the thickening and what the round's labelling reads of it.
  bool HasThickening;
  ThickeningRoundPortals RoundThickening;
  /// The largest absolute coordinate of the generators with two clusters replaced.
  MergedScale Scale;
};

VISKORES_EXEC inline bool MergeContextPortals::Admissible(viskores::Id i,
                                                          viskores::Id j,
                                                          Vec3& merged,
                                                          bool& complete) const
{
  const bool fixed = this->HasFrozen && this->FrozenLabels.Get(i) != 0;
  Vec3 o, bm;
  Mat3 qm;
  if (fixed)
  {
    merged = this->Generators.Get(i);
  }
  else
  {
    // The generator step (b) gives the merged cluster, by the same sums: tree sums over its
    // members in increasing order, from the smallest (Theorem "budget" (v)).
    viskores::Id a = this->MemberOffsets.Get(i), aEnd = this->MemberOffsets.Get(i + 1);
    viskores::Id c = this->MemberOffsets.Get(j), cEnd = this->MemberOffsets.Get(j + 1);
    o = this->Points.Get(viskores::Min(this->Anchors.Get(i), this->Anchors.Get(j)));
    TreeSum<Mat3> qSum;
    TreeSum<Vec3> bSum;
    while (a < aEnd || c < cEnd)
    {
      const bool fromI = c >= cEnd || (a < aEnd && this->Members.Get(a) < this->Members.Get(c));
      const viskores::Id v = fromI ? this->Members.Get(a++) : this->Members.Get(c++);
      const Mat3 mv = this->VertexMatrices.Get(v);
      qSum.Push(mv);
      bSum.Push(MatVec(mv, Sub(this->Points.Get(v), o)));
    }
    qm = qSum.Result();
    bm = bSum.Result();
    merged = Add(o, SolveSPD(qm, bm));
  }
  auto forEachMember = [&](auto&& visit)
  {
    for (int which = 0; which < 2; ++which)
    {
      const viskores::Id c = (which == 0) ? i : j;
      for (viskores::Id k = this->MemberOffsets.Get(c); k < this->MemberOffsets.Get(c + 1); ++k)
      {
        visit(this->Members.Get(k));
      }
    }
  };
  // (F0)
  if (this->MaxRadius < std::numeric_limits<Real>::infinity())
  {
    Real radius = 0;
    forEachMember([&](viskores::Id v)
                  { radius = viskores::Max(radius, Norm(Sub(this->Points.Get(v), merged))); });
    if (LengthUp(radius) > this->MaxRadius)
    {
      return false;
    }
  }
  if (this->GeometricOnly)
  {
    return true;
  }
  if (this->Embedding && fixed && !this->SheetMergeEmbedded(i, j, merged, complete))
  {
    return false;
  }
  // (F4): the new triangles must be nondegenerate and meet no other triangle improperly.
  if (this->Embedding && !fixed)
  {
    bool bad = false;
    this->ForEachNewTriangle(
      i,
      j,
      merged,
      [&](const Vec3* p, const viskores::Id3& labels, viskores::Id t)
      {
        if (viskores::TriangleIsDegenerate(p[0], p[1], p[2]))
        {
          bad = true;
          return false;
        }
        // Against the other new triangles, each pair once.
        bool after = false;
        this->ForEachNewTriangle(
          i,
          j,
          merged,
          [&](const Vec3* q, const viskores::Id3& otherLabels, viskores::Id u)
          {
            if (u == t)
            {
              after = true;
              return true;
            }
            if (after && exact::ImproperPair(p, this->Glued(labels), q, this->Glued(otherLabels)))
            {
              bad = true;
              return false;
            }
            return true;
          });
        if (bad)
        {
          return false;
        }
        // Against the triangles of K away from i and j, which keep their positions.
        Vec3 lo, hi;
        TriangleBounds(p[0], p[1], p[2], lo, hi);
        const bool done = this->Output.Hierarchy.ForEachOverlap(
          lo,
          hi,
          [&](viskores::Id u)
          {
            if (this->Incident(u, i, j) || this->OutputDegenerate.Get(u))
            {
              return true;
            }
            const viskores::Id3 other = this->OutputTriangles.Get(u);
            const Vec3 q[3] = { this->Generators.Get(other[0]),
                                this->Generators.Get(other[1]),
                                this->Generators.Get(other[2]) };
            if (exact::ImproperPair(p, this->Glued(labels), q, this->Glued(other)))
            {
              bad = true;
              return false;
            }
            return true;
          });
        complete = complete && done;
        return !bad;
      });
    if (bad)
    {
      return false;
    }
  }
  // (F7), and (F8) with the fibres, on the 3-label triangles with a vertex in the merged
  // cluster, on the labelling with the merge (`ThickVQC.iso_pass`).
  if (this->HasThickening && !this->ThickeningAdmissible(i, j, merged, complete))
  {
    return false;
  }
  return true;
}

/// Grows the box `[lower, upper]` by `epsilon`, with a margin for the rounding of the bounds,
/// `size` the largest coordinate they come from.
VISKORES_EXEC_CONT inline void GrowBox(Vec3& lower, Vec3& upper, Real epsilon, Real size)
{
  const Real grow = epsilon * (1 + Real(1e-9)) + 64 * UnitRoundoff * size;
  for (int a = 0; a < 3; ++a)
  {
    lower[a] = lower[a] - grow;
    upper[a] = upper[a] + grow;
  }
}

template <typename Visit>
VISKORES_EXEC inline void MergeContextPortals::ForEachEnvelopeItem(viskores::Id i,
                                                                   viskores::Id j,
                                                                   const Vec3& merged,
                                                                   viskores::UInt8 parts,
                                                                   bool& complete,
                                                                   Visit&& visit) const
{
  const Real epsilon = this->Epsilon;
  // Only the tests of the halves in use.
  auto emit = [&](viskores::Id t, const viskores::Id3& tri, viskores::UInt8 tests)
  {
    tests = static_cast<viskores::UInt8>(tests & this->SideTests);
    if (tests != 0)
    {
      visit(t, tri, tests);
    }
  };
  auto inMerged = [&](viskores::Id v)
  {
    const viskores::Id l = this->Labels.Get(v);
    return l == i || l == j;
  };
  auto generator = [&](viskores::Id l)
  { return (l == i || l == j) ? merged : this->Generators.Get(l); };
  auto newLabel = [&](viskores::Id l) { return (l == j) ? i : l; };
  auto displacementExceeds = [&](const viskores::Id3& tri)
  {
    Real d = 0;
    for (int k = 0; k < 3; ++k)
    {
      const Vec3 p = this->Points.Get(tri[k]);
      d = viskores::Max(d, Norm(Sub(generator(this->Labels.Get(tri[k])), p)));
    }
    return DisplacementExceeds(d, epsilon);
  };
  // The input triangles with a vertex in the merged cluster, each from its smallest vertex there.
  for (int which = 0; which < 2 && (parts & 1) != 0; ++which)
  {
    const viskores::Id c = (which == 0) ? i : j;
    for (viskores::Id k = this->MemberOffsets.Get(c); k < this->MemberOffsets.Get(c + 1); ++k)
    {
      const viskores::Id v = this->Members.Get(k);
      for (viskores::Id r = this->CornerOffsets.Get(v); r < this->CornerOffsets.Get(v + 1); ++r)
      {
        const viskores::Id t = this->Corners.Get(r) % this->NumberOfInputTriangles;
        const viskores::Id3 tri = this->InputTriangles.Get(t);
        viskores::Id first = -1;
        for (int a = 0; a < 3; ++a)
        {
          if (inMerged(tri[a]) && (first < 0 || tri[a] < first))
          {
            first = tri[a];
          }
        }
        if (first != v || !displacementExceeds(tri))
        {
          continue;
        }
        if (this->GeometricOnly)
        {
          emit(t, tri, viskores::UInt8(3));
          continue;
        }
        const viskores::Id a = newLabel(this->Labels.Get(tri[0]));
        const viskores::Id b = newLabel(this->Labels.Get(tri[1]));
        const viskores::Id d = newLabel(this->Labels.Get(tri[2]));
        emit(t, tri, viskores::UInt8((a != b && b != d && a != d) ? 3 : 2));
      }
    }
  }
  if (this->GeometricOnly || (parts & 2) == 0)
  {
    return;
  }
  // The input triangles away from the merged cluster near a removed output triangle.
  Vec3 lo(std::numeric_limits<Real>::infinity()), hi(-std::numeric_limits<Real>::infinity());
  for (int which = 0; which < 2; ++which)
  {
    const viskores::Id c = (which == 0) ? i : j;
    for (viskores::Id f = this->FanOffsets.Get(c); f < this->FanOffsets.Get(c + 1); ++f)
    {
      const viskores::Id3 tri = this->OutputTriangles.Get(this->Fans.Get(f));
      for (int k = 0; k < 3; ++k)
      {
        const Vec3 p = this->Generators.Get(tri[k]);
        for (int a = 0; a < 3; ++a)
        {
          lo[a] = viskores::Min(lo[a], p[a]);
          hi[a] = viskores::Max(hi[a], p[a]);
        }
      }
    }
  }
  // Grown by epsilon with a margin for the rounding of the bounds, as proofs/lib/vqc.py: the
  // largest coordinate of the removed triangles and of the input.
  Real size = this->Input.MaxAbsCoordinate;
  for (int a = 0; a < 3; ++a)
  {
    size = viskores::Max(size, viskores::Max(viskores::Abs(lo[a]), viskores::Abs(hi[a])));
  }
  GrowBox(lo, hi, epsilon, size);
  const bool done = this->Input.Hierarchy.ForEachOverlap(
    lo,
    hi,
    [&](viskores::Id t)
    {
      const viskores::Id3 tri = this->InputTriangles.Get(t);
      if (!inMerged(tri[0]) && !inMerged(tri[1]) && !inMerged(tri[2]) && displacementExceeds(tri))
      {
        emit(t, tri, viskores::UInt8(2));
      }
      return true;
    });
  complete = complete && done;
}

/// The distance to `K/ij`: the triangles of `K` not at `i` or `j`, and the new triangles at the
/// merged vertex.
struct MergedOutputDistance
{
  /// The distance when it is at most `cutoff`, and otherwise some value above `cutoff`.
  VISKORES_EXEC Real DistanceUpTo(const Vec3& p,
                                  Real cutoff,
                                  bool& complete,
                                  const CentroidSeeds* seeds = nullptr) const
  {
    Real best = this->Context.Output.DistanceSkipping(
      p,
      [&](viskores::Id t) { return this->Context.Incident(t, this->I, this->J); },
      cutoff,
      complete,
      seeds);
    this->Context.ForEachNewTriangle(this->I,
                                     this->J,
                                     this->Merged,
                                     [&](const Vec3* q, const viskores::Id3&, viskores::Id)
                                     {
                                       best = viskores::Min(
                                         best, PointTriangleDistanceUpper(p, q[0], q[1], q[2]));
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
    // The candidates among the triangles of K/ij, ties broken by their index in K.
    const Vec3 q = TriangleCentroid(s[0], s[1], s[2]);
    CentroidCandidates candidates;
    this->Context.Output.NearestCentroids(
      q,
      [&](viskores::Id t) { return this->Context.Incident(t, this->I, this->J); },
      candidates,
      complete,
      &seeds);
    this->Context.ForEachNewTriangle(
      this->I,
      this->J,
      this->Merged,
      [&](const Vec3* p, const viskores::Id3& labels, viskores::Id t)
      {
        // In the geometric-only mode, the simplex's sorted
        // labels: its position in the sorted simplices of the
        // merged labelling.
        candidates.Offer(Norm(Sub(TriangleCentroid(p[0], p[1], p[2]), q)),
                         this->Context.GeometricOnly ? labels : viskores::Id3(t, 0, 0),
                         p[0],
                         p[1],
                         p[2]);
        return true;
      });
    seeds.Keep(candidates);
    return candidates.Passes(s, bound, margin);
  }

  const MergeContextPortals& Context;
  viskores::Id I;
  viskores::Id J;
  Vec3 Merged;
};

VISKORES_EXEC inline bool MergeContextPortals::TestEnvelopeItem(viskores::Id i,
                                                                viskores::Id j,
                                                                const Vec3& merged,
                                                                viskores::Id t,
                                                                viskores::UInt8 tests,
                                                                Real scaleInput,
                                                                Real scaleOutput,
                                                                bool& complete) const
{
  const viskores::Id3 tri = this->InputTriangles.Get(t);
  viskores::Id pieces = 0;
  auto generator = [&](viskores::Id l)
  { return (l == i || l == j) ? merged : this->Generators.Get(l); };
  bool done = true;
  if ((tests & 1) != 0)
  {
    const CertifiedWithin test{ this->Epsilon, scaleInput };
    const bool within = test(generator(this->Labels.Get(tri[0])),
                             generator(this->Labels.Get(tri[1])),
                             generator(this->Labels.Get(tri[2])),
                             this->Input,
                             pieces,
                             done);
    complete = complete && done;
    if (!within)
    {
      return false;
    }
  }
  if ((tests & 2) != 0)
  {
    const CertifiedWithin test{ this->Epsilon, scaleOutput };
    const MergedOutputDistance distance{ *this, i, j, merged };
    const bool within = test(this->Points.Get(tri[0]),
                             this->Points.Get(tri[1]),
                             this->Points.Get(tri[2]),
                             distance,
                             pieces,
                             done);
    complete = complete && done;
    if (!within)
    {
      return false;
    }
  }
  return true;
}

/// The output complex `K` of a labelling, with the incidences the merge phase reads.
struct OutputComplex
{
  /// The output triangles, as label triples in the order and orientation of Algorithm 5.
  viskores::cont::ArrayHandle<viskores::Id3> Triangles;
  /// Its edges, sorted pairs in lexicographic order, and the number of triangles at each.
  viskores::cont::ArrayHandle<viskores::Id2> Edges;
  viskores::cont::ArrayHandle<viskores::IdComponent> EdgeCounts;
  /// Whether each vertex of `K` lies on its boundary.
  viskores::cont::ArrayHandle<viskores::UInt8> BoundaryVertex;
  bool HasBoundary = false;
  /// The neighbours of every vertex of `K`, in increasing order.
  viskores::cont::ArrayHandle<viskores::Id> NeighborOffsets;
  viskores::cont::ArrayHandle<viskores::Id> Neighbors;
  /// The sorted label triples of the triangles, in lexicographic order.
  viskores::cont::ArrayHandle<viskores::Id3> Faces;
  /// The triangles at every vertex of `K`, in increasing order.
  viskores::cont::ArrayHandle<viskores::Id> FanOffsets;
  viskores::cont::ArrayHandle<viskores::Id> Fans;
};

/// An execution object giving a worklet a `MergeContextPortals`.
struct MergeContextObject : viskores::cont::ExecutionObjectBase
{
  VISKORES_CONT MergeContextPortals PrepareForExecution(viskores::cont::DeviceAdapterId device,
                                                        viskores::cont::Token& token) const
  {
    MergeContextPortals portals;
    portals.Points = this->Mesh->Points.PrepareForInput(device, token);
    portals.InputTriangles = this->Mesh->Triangles.PrepareForInput(device, token);
    portals.CornerOffsets = this->Mesh->VertexCornerOffsets.PrepareForInput(device, token);
    portals.Corners = this->Mesh->VertexCorners.PrepareForInput(device, token);
    portals.NumberOfInputTriangles = this->Mesh->NumberOfTriangles;
    portals.Labels = this->Clusters->Labels.PrepareForInput(device, token);
    portals.MemberOffsets = this->Clusters->Keys.GetOffsets().PrepareForInput(device, token);
    portals.Members = this->Clusters->Keys.GetSortedValuesMap().PrepareForInput(device, token);
    portals.Anchors = this->Clusters->Anchors.PrepareForInput(device, token);
    portals.Generators = this->Clusters->Generators.PrepareForInput(device, token);
    portals.VertexMatrices = this->VertexMatrices.PrepareForInput(device, token);
    portals.OutputTriangles = this->Complex->Triangles.PrepareForInput(device, token);
    portals.FanOffsets = this->Complex->FanOffsets.PrepareForInput(device, token);
    portals.Fans = this->Complex->Fans.PrepareForInput(device, token);
    portals.OutputDegenerate = this->OutputDegenerate.PrepareForInput(device, token);
    portals.Glue = this->Glue.PrepareForInput(device, token);
    portals.HasGlue = this->Glue.GetNumberOfValues() > 0;
    portals.FrozenLabels = this->FrozenLabels.PrepareForInput(device, token);
    portals.HasFrozen = this->FrozenLabels.GetNumberOfValues() > 0;
    portals.GroupKeys = this->GroupKeys.PrepareForInput(device, token);
    portals.GroupOffsets = this->GroupOffsets.PrepareForInput(device, token);
    portals.GroupLabels = this->GroupLabels.PrepareForInput(device, token);
    portals.GeometricOnly = this->GeometricOnly;
    portals.Input = DistanceMeshObject(this->Input).PrepareForExecution(device, token);
    portals.Output = DistanceMeshObject(this->Output).PrepareForExecution(device, token);
    portals.MaxRadius = this->MaxRadius;
    portals.Epsilon = this->Epsilon;
    portals.Embedding = this->Embedding;
    portals.SideTests = static_cast<viskores::UInt8>(this->Sides);
    portals.HasThickening = this->CertifiedThickening != nullptr;
    if (this->CertifiedThickening)
    {
      portals.RoundThickening =
        ThickeningRoundObject(*this->CertifiedThickening, *this->RoundThickening)
          .PrepareForExecution(device, token);
    }
    portals.Scale = this->Scale;
    return portals;
  }

  const SurfaceMesh* Mesh;
  const Clustering* Clusters;
  const OutputComplex* Complex;
  viskores::cont::ArrayHandle<Mat3> VertexMatrices;
  viskores::cont::ArrayHandle<viskores::UInt8> OutputDegenerate;
  viskores::cont::ArrayHandle<viskores::Id> Glue;
  viskores::cont::ArrayHandle<viskores::UInt8> FrozenLabels;
  viskores::cont::ArrayHandle<viskores::Id> GroupKeys;
  viskores::cont::ArrayHandle<viskores::Id> GroupOffsets;
  viskores::cont::ArrayHandle<viskores::Id> GroupLabels;
  bool GeometricOnly = false;
  DistanceMesh Input;
  DistanceMesh Output;
  Real MaxRadius;
  Real Epsilon;
  bool Embedding;
  EnvelopeSides Sides = EnvelopeSides::Both;
  const Thickening* CertifiedThickening = nullptr;
  const ThickeningRound* RoundThickening = nullptr;
  MergedScale Scale;
};

namespace detail
{

/// The verdict of the local test of a candidate, kept in `memo` across the passes of a round.
enum : viskores::UInt8
{
  Unknown = 0,
  Admissible = 1,
  Inadmissible = 2
};

/// A verdict that holds for candidate `pair`: sets `memo` and counts it.
struct ReuseVerdict : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn pair,
                                WholeArrayIn anchors,
                                WholeArrayIn keys,
                                WholeArrayIn mergedAbs,
                                WholeArrayIn scales,
                                WholeArrayIn admissible,
                                FieldInOut memo,
                                FieldOut reused);
  using ExecutionSignature = void(_1, _2, _3, _4, _5, _6, _7, _8);

  template <typename IdPortal, typename KeyPortal, typename RealPortal, typename FlagPortal>
  VISKORES_EXEC void operator()(const viskores::Id2& pair,
                                const IdPortal& anchors,
                                const KeyPortal& keys,
                                const RealPortal& mergedAbs,
                                const RealPortal& scales,
                                const FlagPortal& admissible,
                                viskores::UInt8& memo,
                                viskores::Id& reused) const
  {
    reused = 0;
    const viskores::Id e =
      FindSorted(keys, viskores::Id2(anchors.Get(pair[0]), anchors.Get(pair[1])));
    if (e >= 0 && this->Scale(pair[0], pair[1], mergedAbs.Get(e)) == scales.Get(e))
    {
      memo = admissible.Get(e) ? Admissible : Inadmissible;
      reused = 1;
    }
  }

  MergedScale Scale;
};

/// The key, the largest absolute coordinate of the merged generator and the scale of an
/// evaluated candidate.
struct VerdictEntry : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn pair,
                                FieldIn merged,
                                WholeArrayIn anchors,
                                FieldOut key,
                                FieldOut mergedAbs,
                                FieldOut scale);
  using ExecutionSignature = void(_1, _2, _3, _4, _5, _6);

  template <typename IdPortal>
  VISKORES_EXEC void operator()(const viskores::Id2& pair,
                                const Vec3& merged,
                                const IdPortal& anchors,
                                viskores::Id2& key,
                                Real& mergedAbs,
                                Real& scale) const
  {
    key = viskores::Id2(anchors.Get(pair[0]), anchors.Get(pair[1]));
    mergedAbs = MaxAbs(merged);
    scale = this->Scale(pair[0], pair[1], mergedAbs);
  }

  MergedScale Scale;
};

/// The key of an entry with its position, so that a sort keeps insertion order among equal keys.
struct OrderedKey : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn key, FieldOut ordered);
  using ExecutionSignature = _2(_1, WorkIndex);

  VISKORES_EXEC viskores::Id3 operator()(const viskores::Id2& key, viskores::Id position) const
  {
    return viskores::Id3(key[0], key[1], position);
  }
};

/// Whether a sorted entry is the last of its key, the one inserted last.
struct LastOfKey : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn index,
                                WholeArrayIn sorted,
                                FieldOut last,
                                FieldOut position);
  using ExecutionSignature = void(_1, _2, _3, _4);

  template <typename KeyPortal>
  VISKORES_EXEC void operator()(viskores::Id i,
                                const KeyPortal& sorted,
                                viskores::UInt8& last,
                                viskores::Id& position) const
  {
    const viskores::Id3 k = sorted.Get(i);
    position = k[2];
    last = (i + 1 == sorted.GetNumberOfValues() || sorted.Get(i + 1)[0] != k[0] ||
            sorted.Get(i + 1)[1] != k[1])
      ? 1
      : 0;
  }
};

/// The box of what a merge changes: the members of the two clusters, the generators of the
/// vertices of their simplices in `K`, and the merged generator, which with them spans the new
/// simplices.
struct ChangedRegion : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn pair,
                                WholeArrayIn memberOffsets,
                                WholeArrayIn members,
                                WholeArrayIn points,
                                WholeArrayIn generators,
                                WholeArrayIn fanOffsets,
                                WholeArrayIn fans,
                                WholeArrayIn simplices,
                                WholeArrayIn anchors,
                                WholeArrayIn nextLabels,
                                WholeArrayIn nextGenerators,
                                FieldOut lower,
                                FieldOut upper);
  using ExecutionSignature = void(_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13);

  template <typename IdPortal, typename VecPortal, typename Id3Portal>
  VISKORES_EXEC void operator()(const viskores::Id2& pair,
                                const IdPortal& memberOffsets,
                                const IdPortal& members,
                                const VecPortal& points,
                                const VecPortal& generators,
                                const IdPortal& fanOffsets,
                                const IdPortal& fans,
                                const Id3Portal& simplices,
                                const IdPortal& anchors,
                                const IdPortal& nextLabels,
                                const VecPortal& nextGenerators,
                                Vec3& lower,
                                Vec3& upper) const
  {
    lower = nextGenerators.Get(nextLabels.Get(anchors.Get(pair[0])));
    upper = lower;
    auto extend = [&](const Vec3& p)
    {
      for (int a = 0; a < 3; ++a)
      {
        lower[a] = viskores::Min(lower[a], p[a]);
        upper[a] = viskores::Max(upper[a], p[a]);
      }
    };
    for (int which = 0; which < 2; ++which)
    {
      const viskores::Id c = pair[which];
      for (viskores::Id k = memberOffsets.Get(c); k < memberOffsets.Get(c + 1); ++k)
      {
        extend(points.Get(members.Get(k)));
      }
      for (viskores::Id f = fanOffsets.Get(c); f < fanOffsets.Get(c + 1); ++f)
      {
        const viskores::Id3 simplex = simplices.Get(fans.Get(f));
        for (int k = 0; k < 3; ++k)
        {
          extend(generators.Get(simplex[k]));
        }
      }
    }
  }
};

/// The anchors, in the labelling before the merges, of the two clusters of a committed merge.
struct PairAnchors : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn pair, WholeArrayIn anchors, FieldOut pairAnchors);
  using ExecutionSignature = void(_1, _2, _3);

  template <typename IdPortal, typename OutVec>
  VISKORES_EXEC void operator()(const viskores::Id2& pair,
                                const IdPortal& anchors,
                                OutVec& out) const
  {
    out[0] = anchors.Get(pair[0]);
    out[1] = anchors.Get(pair[1]);
  }
};

/// Whether an entry still holds: neither of its clusters merged, and no box of a change meets
/// its box.
struct VerdictHolds : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn key,
                                FieldIn lower,
                                FieldIn upper,
                                WholeArrayIn mergedAnchors,
                                WholeArrayIn changeLower,
                                WholeArrayIn changeUpper,
                                FieldOut holds);
  using ExecutionSignature = _7(_1, _2, _3, _4, _5, _6);

  template <typename IdPortal, typename VecPortal>
  VISKORES_EXEC viskores::UInt8 operator()(const viskores::Id2& key,
                                           const Vec3& lower,
                                           const Vec3& upper,
                                           const IdPortal& mergedAnchors,
                                           const VecPortal& changeLower,
                                           const VecPortal& changeUpper) const
  {
    if (FindSorted(mergedAnchors, key[0]) >= 0 || FindSorted(mergedAnchors, key[1]) >= 0)
    {
      return 0;
    }
    for (viskores::Id c = 0; c < changeLower.GetNumberOfValues(); ++c)
    {
      const Vec3 l = changeLower.Get(c), u = changeUpper.Get(c);
      if (l[0] <= upper[0] && u[0] >= lower[0] && l[1] <= upper[1] && u[1] >= lower[1] &&
          l[2] <= upper[2] && u[2] >= lower[2])
      {
        return 0;
      }
    }
    return 1;
  }
};

} // namespace detail

/// The verdicts of the local tests, kept across the rounds of the merge phase. A local test
/// reads only what lies in the box of `EvaluateMerge`, and the largest coordinate of the
/// generators away from the two clusters; its verdict holds while neither cluster merges, no
/// merge changes anything in the box, and that coordinate stays the same. The clusters are
/// known by their anchors, which the renumbering of the labels keeps. The entries are kept
/// sorted by the anchors of the two clusters.
class VerdictCache
{
public:
  /// Sets `memo` of the candidates `pairs` whose verdicts hold. Returns how many.
  viskores::Id Reuse(const MergeContextObject& context,
                     const Clustering& clusters,
                     const viskores::cont::ArrayHandle<viskores::Id2>& pairs,
                     viskores::cont::ArrayHandle<viskores::UInt8>& memo) const
  {
    if (this->Keys.GetNumberOfValues() == 0)
    {
      return 0;
    }
    viskores::cont::Invoker invoke;
    detail::ReuseVerdict reuse;
    reuse.Scale = context.Scale;
    viskores::cont::ArrayHandle<viskores::Id> reused;
    invoke(reuse,
           pairs,
           clusters.Anchors,
           this->Keys,
           this->MergedAbs,
           this->Scales,
           this->Admissible,
           memo,
           reused);
    return viskores::cont::Algorithm::Reduce(reused, viskores::Id(0));
  }

  /// Keeps the verdicts of the local tests of `pairs`, in place of earlier ones of the same
  /// clusters.
  void Remember(const MergeContextObject& context,
                const Clustering& clusters,
                const viskores::cont::ArrayHandle<viskores::Id2>& pairs,
                const viskores::cont::ArrayHandle<viskores::UInt8>& admissible,
                const viskores::cont::ArrayHandle<Vec3>& merged,
                const viskores::cont::ArrayHandle<Vec3>& lower,
                const viskores::cont::ArrayHandle<Vec3>& upper)
  {
    using Algorithm = viskores::cont::Algorithm;
    viskores::cont::Invoker invoke;
    detail::VerdictEntry entry;
    entry.Scale = context.Scale;
    viskores::cont::ArrayHandle<viskores::Id2> keys;
    viskores::cont::ArrayHandle<Real> mergedAbs, scales;
    invoke(entry, pairs, merged, clusters.Anchors, keys, mergedAbs, scales);
    // The old entries, then the new; the last entry of every key wins.
    auto allKeys = viskores::cont::make_ArrayHandleConcatenate(this->Keys, keys);
    viskores::cont::ArrayHandle<viskores::Id3> ordered;
    invoke(detail::OrderedKey{}, allKeys, ordered);
    Algorithm::Sort(ordered);
    viskores::cont::ArrayHandle<viskores::UInt8> last;
    viskores::cont::ArrayHandle<viskores::Id> positions, kept;
    invoke(detail::LastOfKey{},
           viskores::cont::ArrayHandleIndex(ordered.GetNumberOfValues()),
           ordered,
           last,
           positions);
    Algorithm::CopyIf(positions, last, kept);
    this->Gather(kept,
                 allKeys,
                 viskores::cont::make_ArrayHandleConcatenate(this->Lower, lower),
                 viskores::cont::make_ArrayHandleConcatenate(this->Upper, upper),
                 viskores::cont::make_ArrayHandleConcatenate(this->MergedAbs, mergedAbs),
                 viskores::cont::make_ArrayHandleConcatenate(this->Scales, scales),
                 viskores::cont::make_ArrayHandleConcatenate(this->Admissible, admissible));
  }

  /// Forgets the verdicts that the merges `pairs` of the labelling `previous`, with output
  /// complex `complex`, into `next` may change.
  void Forget(const SurfaceMesh& mesh,
              const OutputComplex& complex,
              const Clustering& previous,
              const Clustering& next,
              const viskores::cont::ArrayHandle<viskores::Id2>& pairs)
  {
    using Algorithm = viskores::cont::Algorithm;
    if (this->Keys.GetNumberOfValues() == 0)
    {
      return;
    }
    viskores::cont::Invoker invoke;
    viskores::cont::ArrayHandle<Vec3> changeLower, changeUpper;
    invoke(detail::ChangedRegion{},
           pairs,
           previous.Keys.GetOffsets(),
           previous.Keys.GetSortedValuesMap(),
           mesh.Points,
           previous.Generators,
           complex.FanOffsets,
           complex.Fans,
           complex.Triangles,
           previous.Anchors,
           next.Labels,
           next.Generators,
           changeLower,
           changeUpper);
    viskores::cont::ArrayHandle<viskores::Id> mergedAnchors;
    invoke(detail::PairAnchors{},
           pairs,
           previous.Anchors,
           viskores::cont::make_ArrayHandleGroupVec<2>(mergedAnchors));
    Algorithm::Sort(mergedAnchors);
    Algorithm::Unique(mergedAnchors);
    viskores::cont::ArrayHandle<viskores::UInt8> holds;
    invoke(detail::VerdictHolds{},
           this->Keys,
           this->Lower,
           this->Upper,
           mergedAnchors,
           changeLower,
           changeUpper,
           holds);
    viskores::cont::ArrayHandle<viskores::Id> kept;
    Algorithm::CopyIf(
      viskores::cont::ArrayHandleIndex(this->Keys.GetNumberOfValues()), holds, kept);
    this->Gather(
      kept, this->Keys, this->Lower, this->Upper, this->MergedAbs, this->Scales, this->Admissible);
  }

private:
  /// Replaces the entries by the entries `kept` of the given arrays, in order.
  template <typename KeyArray,
            typename LowerArray,
            typename UpperArray,
            typename AbsArray,
            typename ScaleArray,
            typename FlagArray>
  void Gather(const viskores::cont::ArrayHandle<viskores::Id>& kept,
              const KeyArray& keys,
              const LowerArray& lower,
              const UpperArray& upper,
              const AbsArray& mergedAbs,
              const ScaleArray& scales,
              const FlagArray& admissible)
  {
    using Algorithm = viskores::cont::Algorithm;
    viskores::cont::ArrayHandle<viskores::Id2> newKeys;
    viskores::cont::ArrayHandle<Vec3> newLower, newUpper;
    viskores::cont::ArrayHandle<Real> newAbs, newScales;
    viskores::cont::ArrayHandle<viskores::UInt8> newAdmissible;
    Algorithm::Copy(viskores::cont::make_ArrayHandlePermutation(kept, keys), newKeys);
    Algorithm::Copy(viskores::cont::make_ArrayHandlePermutation(kept, lower), newLower);
    Algorithm::Copy(viskores::cont::make_ArrayHandlePermutation(kept, upper), newUpper);
    Algorithm::Copy(viskores::cont::make_ArrayHandlePermutation(kept, mergedAbs), newAbs);
    Algorithm::Copy(viskores::cont::make_ArrayHandlePermutation(kept, scales), newScales);
    Algorithm::Copy(viskores::cont::make_ArrayHandlePermutation(kept, admissible), newAdmissible);
    this->Keys = newKeys;
    this->Lower = newLower;
    this->Upper = newUpper;
    this->MergedAbs = newAbs;
    this->Scales = newScales;
    this->Admissible = newAdmissible;
  }

  viskores::cont::ArrayHandle<viskores::Id2> Keys;
  viskores::cont::ArrayHandle<Vec3> Lower;
  viskores::cont::ArrayHandle<Vec3> Upper;
  viskores::cont::ArrayHandle<Real> MergedAbs;
  viskores::cont::ArrayHandle<Real> Scales;
  viskores::cont::ArrayHandle<viskores::UInt8> Admissible;
};

/// A labelling whose envelope verdicts are known, against which the merge phase tests the
/// labellings it tries: the clusters, their output triangles and the verdicts.
struct EnvelopeBase
{
  Clustering Clusters;
  viskores::cont::ArrayHandle<viskores::Id3> Triangles;
  EnvelopeVerdicts Verdicts;
};

/// The bases of the merge phase: the last labelling committed, and the last whose full test
/// fired with every envelope verdict known. In a run of rounds that fall back to one merge, the
/// set tried next differs from the one that fired by a few merges, and from the committed
/// labelling by all of them.
struct EnvelopeBases
{
  EnvelopeBase Committed;
  EnvelopeBase Rejected;
};

namespace detail
{

/// Whether the cluster with `anchor` is not the base cluster of its anchor, with the same
/// members and the bit-identical generator, as far as the size and the generator tell;
/// `LeavesBaseCluster` checks the members.
struct ChangedCluster : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn anchor,
                                FieldIn size,
                                FieldIn generator,
                                WholeArrayIn baseLabels,
                                WholeArrayIn baseSizes,
                                WholeArrayIn baseGenerators,
                                FieldOut changed);
  using ExecutionSignature = _7(_1, _2, _3, _4, _5, _6);

  template <typename IdPortal, typename VecPortal>
  VISKORES_EXEC viskores::UInt8 operator()(viskores::Id anchor,
                                           viskores::Id size,
                                           const Vec3& generator,
                                           const IdPortal& baseLabels,
                                           const IdPortal& baseSizes,
                                           const VecPortal& baseGenerators) const
  {
    const viskores::Id b = baseLabels.Get(anchor);
    const Vec3 g = baseGenerators.Get(b);
    return (baseSizes.Get(b) != size || std::memcmp(&g, &generator, sizeof(Vec3)) != 0) ? 1 : 0;
  }
};

/// Marks the cluster of a vertex changed if the vertex is in another base cluster than the
/// anchor of its cluster. A cluster whose members all lie in one base cluster, of its size, is
/// that cluster. The committed base is refined by every labelling tried, so this marks nothing
/// against it; a rejected one is not, and two clusters of the same anchor and size may differ.
struct LeavesBaseCluster : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn label,
                                FieldIn baseLabel,
                                WholeArrayIn anchors,
                                WholeArrayIn baseLabels,
                                WholeArrayInOut changed);
  using ExecutionSignature = void(_1, _2, _3, _4, _5);

  template <typename IdPortal, typename FlagPortal>
  VISKORES_EXEC void operator()(viskores::Id label,
                                viskores::Id baseLabel,
                                const IdPortal& anchors,
                                const IdPortal& baseLabels,
                                FlagPortal& changed) const
  {
    if (baseLabel != baseLabels.Get(anchors.Get(label)))
    {
      changed.Set(label, 1);
    }
  }
};

/// Whether the cluster of a base anchor is changed in the labelling tried.
struct ChangedBaseCluster : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn anchor,
                                WholeArrayIn labels,
                                WholeArrayIn changed,
                                FieldOut out);
  using ExecutionSignature = _4(_1, _2, _3);

  template <typename IdPortal, typename FlagPortal>
  VISKORES_EXEC viskores::UInt8 operator()(viskores::Id anchor,
                                           const IdPortal& labels,
                                           const FlagPortal& changed) const
  {
    return changed.Get(labels.Get(anchor));
  }
};

/// The box of an output triangle with a changed label, and whether it has one.
struct ChangedTriangleBox : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn triangle,
                                WholeArrayIn changed,
                                WholeArrayIn generators,
                                FieldOut isChanged,
                                FieldOut lower,
                                FieldOut upper);
  using ExecutionSignature = void(_1, _2, _3, _4, _5, _6);

  template <typename FlagPortal, typename VecPortal>
  VISKORES_EXEC void operator()(const viskores::Id3& t,
                                const FlagPortal& changed,
                                const VecPortal& generators,
                                viskores::UInt8& isChanged,
                                Vec3& lower,
                                Vec3& upper) const
  {
    isChanged = (changed.Get(t[0]) || changed.Get(t[1]) || changed.Get(t[2])) ? 1 : 0;
    TriangleBounds(generators.Get(t[0]), generators.Get(t[1]), generators.Get(t[2]), lower, upper);
  }
};

/// Whether the envelope test of an input triangle may read something that differs from the base:
/// a vertex in a changed cluster, or a changed output triangle whose box meets, closed, the
/// region of the output that its test read.
struct AffectedTriangle : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn triangle,
                                FieldIn lower,
                                FieldIn upper,
                                WholeArrayIn labels,
                                WholeArrayIn changed,
                                ExecObject changedTriangles,
                                FieldOut affected);
  using ExecutionSignature = void(_1, _2, _3, _4, _5, _6, _7);

  template <typename IdPortal, typename FlagPortal, typename Hierarchy>
  VISKORES_EXEC void operator()(const viskores::Id3& t,
                                const Vec3& lower,
                                const Vec3& upper,
                                const IdPortal& labels,
                                const FlagPortal& changed,
                                const Hierarchy& changedTriangles,
                                viskores::UInt8& affected) const
  {
    affected = 0;
    for (int k = 0; k < 3; ++k)
    {
      if (changed.Get(labels.Get(t[k])))
      {
        affected = 1;
        return;
      }
    }
    changedTriangles.ForEachOverlap(lower,
                                    upper,
                                    [&](viskores::Id)
                                    {
                                      affected = 1;
                                      return false;
                                    });
  }
};

} // namespace detail

/// The input triangles whose envelope test on `clusters`, with output triangles `triangles`, may
/// read something that differs from its test on `base`. A cluster is changed unless it has the
/// members and the bit-identical generator of a base cluster; an output triangle is changed if
/// one of its labels is. Every other output triangle is in both outputs with the same points, and
/// its relative order, which breaks the ties of the single-triangle candidates, is the same: the
/// output triangles are ordered by their first input triangle, and the labels are renumbered in
/// order. A test reads the output only through its distance queries, which reach the triangles
/// whose boxes meet the ball of radius epsilon around the query, and its single-triangle
/// candidates, which a centroid farther than the last candidate kept cannot enter or leave.
inline viskores::cont::ArrayHandle<viskores::UInt8> AffectedTriangles(
  const SurfaceMesh& mesh,
  const EnvelopeBase& base,
  const Clustering& clusters,
  const viskores::cont::ArrayHandle<viskores::Id3>& triangles)
{
  using Algorithm = viskores::cont::Algorithm;
  viskores::cont::Invoker invoke;
  const Clustering& previous = base.Clusters;
  viskores::cont::ArrayHandle<viskores::UInt8> changed, changedBefore;
  invoke(detail::ChangedCluster{},
         clusters.Anchors,
         clusters.Sizes,
         clusters.Generators,
         previous.Labels,
         previous.Sizes,
         previous.Generators,
         changed);
  invoke(detail::LeavesBaseCluster{},
         clusters.Labels,
         previous.Labels,
         clusters.Anchors,
         previous.Labels,
         changed);
  invoke(detail::ChangedBaseCluster{}, previous.Anchors, clusters.Labels, changed, changedBefore);
  // The boxes of the changed output triangles, removed and new.
  viskores::cont::ArrayHandle<viskores::UInt8> isChanged;
  viskores::cont::ArrayHandle<Vec3> lower, upper, removedLower, removedUpper, newLower, newUpper;
  invoke(detail::ChangedTriangleBox{},
         base.Triangles,
         changedBefore,
         previous.Generators,
         isChanged,
         lower,
         upper);
  Algorithm::CopyIf(lower, isChanged, removedLower);
  Algorithm::CopyIf(upper, isChanged, removedUpper);
  invoke(
    detail::ChangedTriangleBox{}, triangles, changed, clusters.Generators, isChanged, lower, upper);
  Algorithm::CopyIf(lower, isChanged, newLower);
  Algorithm::CopyIf(upper, isChanged, newUpper);
  Algorithm::Copy(viskores::cont::make_ArrayHandleConcatenate(removedLower, newLower), lower);
  Algorithm::Copy(viskores::cont::make_ArrayHandleConcatenate(removedUpper, newUpper), upper);
  const viskores::worklet::spatialstructure::BoundingVolumeHierarchy changedTriangles(lower, upper);
  viskores::cont::ArrayHandle<viskores::UInt8> affected;
  invoke(detail::AffectedTriangle{},
         mesh.Triangles,
         base.Verdicts.Lower,
         base.Verdicts.Upper,
         clusters.Labels,
         changed,
         changedTriangles,
         affected);
  return affected;
}

/// (F5) on a labelling of the merge phase: in full, or with `bases` only on the triangles whose
/// verdicts can differ from those of the base that leaves the fewest (`AffectedTriangles`), with
/// the verdicts of the whole test. Stores the verdicts in `verdicts`.
inline void MergeEnvelopeFlags(const SurfaceMesh& mesh,
                               const DistanceMesh& inputDistance,
                               const Clustering& clusters,
                               const viskores::cont::ArrayHandle<viskores::Id3>& triangles,
                               const MergeParameters& parameters,
                               viskores::cont::ArrayHandle<viskores::UInt8>& flags,
                               const EnvelopeBases* bases,
                               EnvelopeVerdicts* verdicts)
{
  const EnvelopeMode mode =
    parameters.Topological ? EnvelopeMode::Surface : EnvelopeMode::Simplices;
  const EnvelopeBase* base = bases ? &bases->Committed : nullptr;
  viskores::cont::ArrayHandle<viskores::UInt8> affected;
  if (base)
  {
    affected = AffectedTriangles(mesh, *base, clusters, triangles);
    if (bases->Rejected.Verdicts.FailedOutput.GetNumberOfValues() > 0)
    {
      viskores::cont::ArrayHandle<viskores::UInt8> affectedRejected =
        AffectedTriangles(mesh, bases->Rejected, clusters, triangles);
      if (viskores::cont::Algorithm::Reduce(affectedRejected, viskores::Id(0)) <
          viskores::cont::Algorithm::Reduce(affected, viskores::Id(0)))
      {
        base = &bases->Rejected;
        affected = affectedRejected;
      }
    }
  }
  EnvelopeVerdicts local;
  EnvelopeVerdicts& out = verdicts ? *verdicts : local;
  EnvelopeFlags(mesh,
                inputDistance,
                clusters.Labels,
                clusters.Generators,
                triangles,
                clusters.Sizes,
                parameters.ErrorBound,
                mode,
                parameters.Sides,
                flags,
                &out,
                base ? &base->Verdicts : nullptr,
                base ? &affected : nullptr);
}

namespace detail
{

struct CountValues : viskores::worklet::WorkletReduceByKey
{
  using ControlSignature = void(KeysIn keys, ReducedValuesOut count);
  using ExecutionSignature = _2(ValueCount);
  using InputDomain = _1;

  VISKORES_EXEC viskores::IdComponent operator()(viskores::IdComponent count) const
  {
    return count;
  }
};

struct BoundaryEnds : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn edge, FieldIn count, FieldOut ends);
  using ExecutionSignature = void(_1, _2, _3);

  template <typename EndsVec>
  VISKORES_EXEC void operator()(const viskores::Id2& edge,
                                viskores::IdComponent count,
                                EndsVec& ends) const
  {
    ends[0] = (count == 1) ? edge[0] : viskores::Id(-1);
    ends[1] = (count == 1) ? edge[1] : viskores::Id(-1);
  }
};

} // namespace detail

/// Sorts records `(key, value)` and returns, for keys `0, ..., numKeys - 1`, the offsets of their
/// runs and the values in order: a compressed adjacency list.
inline void BuildAdjacency(viskores::cont::ArrayHandle<viskores::Id2>& records,
                           viskores::Id numKeys,
                           viskores::cont::ArrayHandle<viskores::Id>& offsets,
                           viskores::cont::ArrayHandle<viskores::Id>& values)
{
  using Algorithm = viskores::cont::Algorithm;
  viskores::cont::Invoker invoke;
  Algorithm::Sort(records);
  viskores::cont::ArrayHandle<viskores::Id> keys;
  invoke(SplitPair{}, records, keys, values);
  Algorithm::LowerBounds(keys, viskores::cont::ArrayHandleIndex(numKeys + 1), offsets);
}

namespace detail
{

struct CornerRecords : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn triangle, FieldOut records);
  using ExecutionSignature = void(_1, _2, WorkIndex);

  template <typename RecordsVec>
  VISKORES_EXEC void operator()(const viskores::Id3& t,
                                RecordsVec& records,
                                viskores::Id index) const
  {
    for (int c = 0; c < 3; ++c)
    {
      records[c] = viskores::Id2(t[c], index);
    }
  }
};

struct SortTriple : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn triangle, FieldOut sorted);
  using ExecutionSignature = _2(_1);

  VISKORES_EXEC viskores::Id3 operator()(const viskores::Id3& t) const { return SortThree(t); }
};

} // namespace detail

/// Builds the output complex of the output triangles `triangles` on `k` vertices.
inline OutputComplex BuildOutputComplex(const viskores::cont::ArrayHandle<viskores::Id3>& triangles,
                                        viskores::Id k)
{
  using Algorithm = viskores::cont::Algorithm;
  viskores::cont::Invoker invoke;
  OutputComplex complex;
  complex.Triangles = triangles;
  complex.BoundaryVertex.AllocateAndFill(k, 0);
  if (triangles.GetNumberOfValues() == 0)
  {
    complex.NeighborOffsets.AllocateAndFill(k + 1, 0);
    complex.FanOffsets.AllocateAndFill(k + 1, 0);
    return complex;
  }
  viskores::cont::ArrayHandle<viskores::Id2> sides;
  invoke(SortedTriangleEdges{}, triangles, viskores::cont::make_ArrayHandleGroupVec<3>(sides));
  viskores::worklet::Keys<viskores::Id2> edgeKeys;
  edgeKeys.BuildArrays(sides, viskores::worklet::KeysSortType::Unstable);
  complex.Edges = edgeKeys.GetUniqueKeys();
  invoke(detail::CountValues{}, edgeKeys, complex.EdgeCounts);

  viskores::cont::ArrayHandle<viskores::Id> ends, listed;
  invoke(detail::BoundaryEnds{},
         complex.Edges,
         complex.EdgeCounts,
         viskores::cont::make_ArrayHandleGroupVec<2>(ends));
  viskores::cont::ArrayHandle<viskores::UInt8> valid;
  invoke(IsValidLabel{}, ends, valid);
  Algorithm::CopyIf(ends, valid, listed);
  complex.HasBoundary = listed.GetNumberOfValues() > 0;
  invoke(MarkIndex{}, listed, complex.BoundaryVertex);

  viskores::cont::ArrayHandle<viskores::Id2> edgeRecords;
  invoke(BothDirections{}, complex.Edges, viskores::cont::make_ArrayHandleGroupVec<2>(edgeRecords));
  BuildAdjacency(edgeRecords, k, complex.NeighborOffsets, complex.Neighbors);

  viskores::cont::ArrayHandle<viskores::Id2> cornerRecords;
  invoke(
    detail::CornerRecords{}, triangles, viskores::cont::make_ArrayHandleGroupVec<3>(cornerRecords));
  BuildAdjacency(cornerRecords, k, complex.FanOffsets, complex.Fans);

  invoke(detail::SortTriple{}, triangles, complex.Faces);
  Algorithm::Sort(complex.Faces);
  return complex;
}

namespace detail
{

struct IsCutRecord : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn record, FieldOut cut);
  using ExecutionSignature = _2(_1);

  VISKORES_EXEC viskores::UInt8 operator()(const viskores::Id3& record) const
  {
    return (record[0] != 0) ? 1 : 0;
  }
};

/// Per component of the mesh, which is a component of `K`: its clusters, and those on the
/// boundary. Integer atomics give the same counts in any order.
struct CountComponentClusters : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn anchor,
                                FieldIn boundary,
                                WholeArrayIn vertexComponents,
                                AtomicArrayInOut sizes,
                                AtomicArrayInOut boundaries);
  using ExecutionSignature = void(_1, _2, _3, _4, _5);

  template <typename IdPortal, typename Atomic>
  VISKORES_EXEC void operator()(viskores::Id anchor,
                                viskores::UInt8 boundary,
                                const IdPortal& vertexComponents,
                                Atomic& sizes,
                                Atomic& boundaries) const
  {
    const viskores::Id component = vertexComponents.Get(anchor);
    sizes.Add(component, 1);
    if (boundary)
    {
      boundaries.Add(component, 1);
    }
  }
};

/// Whether an edge of `K` lies in a component that is neither a triangle nor the boundary of a
/// tetrahedron.
struct InReducibleComponent : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn edge,
                                WholeArrayIn anchors,
                                WholeArrayIn vertexComponents,
                                WholeArrayIn sizes,
                                WholeArrayIn boundaries,
                                FieldOut reducible);
  using ExecutionSignature = _6(_1, _2, _3, _4, _5);

  template <typename IdPortal>
  VISKORES_EXEC viskores::UInt8 operator()(const viskores::Id2& edge,
                                           const IdPortal& anchors,
                                           const IdPortal& vertexComponents,
                                           const IdPortal& sizes,
                                           const IdPortal& boundaries) const
  {
    const viskores::Id component = vertexComponents.Get(anchors.Get(edge[0]));
    const viskores::Id size = sizes.Get(component);
    return (size > 4 || (size == 4 && boundaries.Get(component) > 0)) ? 1 : 0;
  }
};

/// Marks the label of a frozen vertex on the boundary of the cut; every write is a 1.
struct MarkFrozenBoundary : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn frozen,
                                FieldIn boundary,
                                FieldIn label,
                                WholeArrayOut labels);
  using ExecutionSignature = void(_1, _2, _3, _4);

  template <typename FlagsPortal>
  VISKORES_EXEC void operator()(viskores::UInt8 frozen,
                                viskores::UInt8 boundary,
                                viskores::Id label,
                                FlagsPortal& labels) const
  {
    if (frozen && boundary)
    {
      labels.Set(label, 1);
    }
  }
};

} // namespace detail

/// The priority of a candidate merge `(i, j)`, smaller first: `max(D_0, deg i + deg j)`, the
/// degree sum of the two clusters with a floor (0 without the degree key), then the increase of
/// the objective (0 in hash order), then the hash of the anchors (0 in cost order), then the
/// labels.
using MergePriority =
  viskores::Pair<viskores::Id,
                 viskores::Pair<Real, viskores::Pair<viskores::UInt64, viskores::Id2>>>;

/// The sums of `C_i` and `C_j` together, both moved to the anchor `o` of the union, the member of
/// smallest index: moving from `o_c` to `o` adds `Q_c (o_c - o)` to `b_c` and
/// `2 (o_c - o) . b_c + (o_c - o)^T Q_c (o_c - o)` to `s_c`.
template <typename IdPortal, typename VecPortal, typename MatPortal, typename RealPortal>
VISKORES_EXEC_CONT void MergedSums(viskores::Id i,
                                   viskores::Id j,
                                   const IdPortal& anchors,
                                   const VecPortal& points,
                                   const MatPortal& q,
                                   const VecPortal& b,
                                   const RealPortal& s,
                                   Vec3& o,
                                   Mat3& qm,
                                   Vec3& bm,
                                   Real& sm)
{
  const viskores::Id a = (anchors.Get(i) < anchors.Get(j)) ? i : j;
  o = points.Get(anchors.Get(a));
  qm = Mat3(0);
  bm = Vec3(0);
  sm = 0;
  for (int which = 0; which < 2; ++which)
  {
    const viskores::Id c = (which == 0) ? i : j;
    const Vec3 d = Sub(points.Get(anchors.Get(c)), o);
    const Mat3 qc = q.Get(c);
    const Vec3 bc = b.Get(c);
    qm = MatAdd(qm, qc);
    bm = Add(Add(bm, bc), MatVec(qc, d));
    sm = sm + ((s.Get(c) + 2 * Dot(d, bc)) + Quad(d, qc));
  }
}

/// `Phi_C(c_C) = s_C - b_C^T Q_C^{-1} b_C`.
VISKORES_EXEC_CONT inline Real Phi(const Mat3& q, const Vec3& b, Real s)
{
  return s - Dot(b, SolveSPD(q, b));
}

namespace detail
{

/// Whether the edge `ij` of `K` is a candidate merge: the frozen-cluster rule and the glue
/// condition, then the local test of Proposition "A merge is an edge contraction" (ii) (not a
/// linking edge, and in no non-facial 3-cycle); and its `MergePriority`.
struct MergeCandidate : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn edge,
                                FieldIn count,
                                WholeArrayIn boundary,
                                WholeArrayIn neighborOffsets,
                                WholeArrayIn neighbors,
                                WholeArrayIn degreeOffsets,
                                WholeArrayIn faces,
                                WholeArrayIn frozenLabels,
                                WholeArrayIn frozenBoundary,
                                WholeArrayIn contactOffsets,
                                WholeArrayIn contacts,
                                WholeArrayIn anchors,
                                WholeArrayIn points,
                                WholeArrayIn q,
                                WholeArrayIn b,
                                WholeArrayIn s,
                                FieldOut isCandidate,
                                FieldOut record);
  using ExecutionSignature =
    void(_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18);

  VISKORES_CONT MergeCandidate(bool topological,
                               bool hasFrozen,
                               bool faceMerges,
                               viskores::Id degreeFloor,
                               MergeOrder order)
    : Topological(topological)
    , HasFrozen(hasFrozen)
    , FaceMerges(faceMerges)
    , DegreeFloor(degreeFloor)
    , Order(order)
  {
  }

  template <typename FlagPortal,
            typename IdPortal,
            typename Id3Portal,
            typename VecPortal,
            typename MatPortal,
            typename RealPortal>
  VISKORES_EXEC void operator()(const viskores::Id2& edge,
                                viskores::IdComponent count,
                                const FlagPortal& boundary,
                                const IdPortal& neighborOffsets,
                                const IdPortal& neighbors,
                                const IdPortal& degreeOffsets,
                                const Id3Portal& faces,
                                const FlagPortal& frozenLabels,
                                const FlagPortal& frozenBoundary,
                                const IdPortal& contactOffsets,
                                const Id3Portal& contacts,
                                const IdPortal& anchors,
                                const VecPortal& points,
                                const MatPortal& q,
                                const VecPortal& b,
                                const RealPortal& s,
                                viskores::UInt8& isCandidate,
                                MergePriority& record) const
  {
    const viskores::Id i = edge[0], j = edge[1];
    isCandidate = 0;
    record = MergePriority(0, { 0, { 0, edge } });
    bool frozenJ = false;
    if (this->HasFrozen)
    {
      // With face merges, a cluster of a sheet may join a frozen one, which keeps its label,
      // if a frozen copy in it is on the boundary of the cut: a copy of a junction inside it (a
      // pinch, or an edge of X that does not separate its sheet) never grows.
      const bool fi = frozenLabels.Get(i) != 0, fj = frozenLabels.Get(j) != 0;
      if ((fi || fj) && !(this->FaceMerges && fi != fj && frozenBoundary.Get(fi ? i : j)))
      {
        return;
      }
      frozenJ = fj;
      // The glue condition: the merged cluster touches at most one copy of every frozen input
      // vertex. The contacts of each cluster are sorted by input vertex, then copy.
      viskores::Id a = contactOffsets.Get(i), aEnd = contactOffsets.Get(i + 1);
      viskores::Id c = contactOffsets.Get(j), cEnd = contactOffsets.Get(j + 1);
      viskores::Id vertex = -1, copy = -1;
      while (a < aEnd || c < cEnd)
      {
        viskores::Id3 next;
        if (c >= cEnd || (a < aEnd && contacts.Get(a)[1] <= contacts.Get(c)[1]))
        {
          next = contacts.Get(a++);
        }
        else
        {
          next = contacts.Get(c++);
        }
        if (next[1] == vertex && next[2] != copy)
        {
          return;
        }
        vertex = next[1];
        copy = next[2];
      }
    }
    if (this->Topological)
    {
      if (count == 2 && boundary.Get(i) && boundary.Get(j))
      {
        return; // a linking edge
      }
      // Every common neighbour x must make ijx a triangle of K.
      viskores::Id a = neighborOffsets.Get(i), aEnd = neighborOffsets.Get(i + 1);
      viskores::Id c = neighborOffsets.Get(j), cEnd = neighborOffsets.Get(j + 1);
      while (a < aEnd && c < cEnd)
      {
        const viskores::Id x = neighbors.Get(a), y = neighbors.Get(c);
        if (x < y)
        {
          ++a;
        }
        else if (y < x)
        {
          ++c;
        }
        else
        {
          viskores::Id3 triple(i, j, x);
          if (x < i)
          {
            triple = viskores::Id3(x, i, j);
          }
          else if (x < j)
          {
            triple = viskores::Id3(i, x, j);
          }
          viskores::Id lo = LowerBound(faces, 0, faces.GetNumberOfValues(), triple);
          if (lo == faces.GetNumberOfValues() || faces.Get(lo) != triple)
          {
            return; // a non-facial 3-cycle
          }
          ++a;
          ++c;
        }
      }
    }
    Vec3 o, bm;
    Mat3 qm;
    Real sm;
    MergedSums(i, j, anchors, points, q, b, s, o, qm, bm, sm);
    const Real d =
      Phi(qm, bm, sm) - Phi(q.Get(i), b.Get(i), s.Get(i)) - Phi(q.Get(j), b.Get(j), s.Get(j));
    if (this->Order == MergeOrder::Hash)
    {
      record.second.second.first = HashPair(anchors.Get(i), anchors.Get(j));
    }
    else
    {
      record.second.first = viskores::Max(d, Real(0));
    }
    // The pair merges its second cluster into its first, which a frozen cluster must be.
    record.second.second.second = frozenJ ? viskores::Id2(j, i) : edge;
    if (this->DegreeFloor >= 0)
    {
      const viskores::Id degrees = (degreeOffsets.Get(i + 1) - degreeOffsets.Get(i)) +
        (degreeOffsets.Get(j + 1) - degreeOffsets.Get(j));
      record.first = viskores::Max(this->DegreeFloor, degrees);
    }
    isCandidate = 1;
  }

  bool Topological;
  bool HasFrozen;
  bool FaceMerges;
  viskores::Id DegreeFloor;
  MergeOrder Order;
};

struct CandidatePair : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn record, FieldOut pair, FieldOut ends);
  using ExecutionSignature = void(_1, _2, _3, WorkIndex);

  template <typename EndsVec>
  VISKORES_EXEC void operator()(const MergePriority& record,
                                viskores::Id2& pair,
                                EndsVec& ends,
                                viskores::Id rank) const
  {
    pair = record.second.second.second;
    ends[0] = viskores::Id2(pair[0], rank);
    ends[1] = viskores::Id2(pair[1], rank);
  }
};

} // namespace detail

/// The vertex of the glued output `K` of every label: the copies of every frozen vertex
/// identified, or with the singular curves decimated, the pieces of every segment; empty for an
/// input that is a surface.
inline viskores::cont::ArrayHandle<viskores::Id> OutputGlue(const MergeParameters& parameters,
                                                            const Clustering& clusters)
{
  if (parameters.Curves)
  {
    return SegmentGlueMap(
      *parameters.Curves, parameters.FrozenVertices, parameters.Original, clusters);
  }
  return GlueMap(parameters.Canonical, clusters);
}

namespace detail
{

struct ExactlyDegenerate : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn triangle, WholeArrayIn generators, FieldOut degenerate);
  using ExecutionSignature = _3(_1, _2);

  template <typename GeneratorsPortal>
  VISKORES_EXEC viskores::UInt8 operator()(const viskores::Id3& t,
                                           const GeneratorsPortal& generators) const
  {
    return viskores::TriangleIsDegenerate(
             generators.Get(t[0]), generators.Get(t[1]), generators.Get(t[2]))
      ? 1
      : 0;
  }
};

/// Up to three largest absolute coordinates with their labels, by value descending and then
/// label ascending; empty slots hold -1.
using TopThree = viskores::Pair<viskores::Vec<Real, 3>, viskores::Vec<viskores::Id, 3>>;

VISKORES_EXEC_CONT inline TopThree NoTop()
{
  return TopThree(viskores::Vec<Real, 3>(-1), viskores::Vec<viskores::Id, 3>(-1));
}

/// The largest absolute coordinate of a generator, as a list of one.
struct GeneratorTop : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn generator, FieldOut top);
  using ExecutionSignature = _2(_1, WorkIndex);

  VISKORES_EXEC TopThree operator()(const Vec3& p, viskores::Id label) const
  {
    TopThree top = NoTop();
    top.first[0] = MaxAbs(p);
    top.second[0] = label;
    return top;
  }
};

/// Merges two lists of `TopThree`: associative and commutative, so any order of reduction gives
/// the list of the sequential scan with strict comparisons.
struct MergeTops
{
  VISKORES_EXEC_CONT TopThree operator()(const TopThree& a, const TopThree& b) const
  {
    auto before = [](Real va, viskores::Id la, Real vb, viskores::Id lb)
    { return va > vb || (va == vb && la >= 0 && (lb < 0 || la < lb)); };
    TopThree merged = NoTop();
    viskores::IdComponent i = 0, j = 0;
    for (viskores::IdComponent slot = 0; slot < 3; ++slot)
    {
      if (j >= 3 || (i < 3 && before(a.first[i], a.second[i], b.first[j], b.second[j])))
      {
        merged.first[slot] = a.first[i];
        merged.second[slot] = a.second[i];
        ++i;
      }
      else
      {
        merged.first[slot] = b.first[j];
        merged.second[slot] = b.second[j];
        ++j;
      }
    }
    return merged;
  }
};

/// Tests again the merges into a frozen cluster, the first of their pair, whose verdicts are not
/// kept between rounds.
struct RetestSheetMerge : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn pair, WholeArrayIn frozenLabels, FieldInOut memo);
  using ExecutionSignature = void(_1, _2, _3);

  template <typename FlagPortal>
  VISKORES_EXEC void operator()(const viskores::Id2& pair,
                                const FlagPortal& frozenLabels,
                                viskores::UInt8& memo) const
  {
    if (frozenLabels.Get(pair[0]))
    {
      memo = Unknown;
    }
  }
};

enum : viskores::UInt8
{
  Alive = 0,
  Committed = 1,
  Dead = 2,
  Rejected = 3
};

/// Visits, while `visit` returns true, the candidates that conflict with the merge `pair`: those
/// touching one of its two clusters or a neighbour of one in `K`.
template <typename IdPortal, typename Visit>
VISKORES_EXEC bool ForEachConflict(const viskores::Id2& pair,
                                   const IdPortal& neighborOffsets,
                                   const IdPortal& neighbors,
                                   const IdPortal& touchingOffsets,
                                   const IdPortal& touching,
                                   Visit&& visit)
{
  auto visitCluster = [&](viskores::Id x)
  {
    for (viskores::Id t = touchingOffsets.Get(x); t < touchingOffsets.Get(x + 1); ++t)
    {
      if (!visit(touching.Get(t)))
      {
        return false;
      }
    }
    return true;
  };
  for (int e = 0; e < 2; ++e)
  {
    const viskores::Id c = pair[e];
    if (!visitCluster(c))
    {
      return false;
    }
    for (viskores::Id n = neighborOffsets.Get(c); n < neighborOffsets.Get(c + 1); ++n)
    {
      if (!visitCluster(neighbors.Get(n)))
      {
        return false;
      }
    }
  }
  return true;
}

/// A live candidate is ready when no live candidate of higher priority conflicts with it; then
/// every candidate of higher priority it conflicts with has been rejected or has died, and the
/// sequential greedy rule would examine it.
struct ReadyCandidates : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn pair,
                                FieldIn state,
                                WholeArrayIn states,
                                WholeArrayIn neighborOffsets,
                                WholeArrayIn neighbors,
                                WholeArrayIn touchingOffsets,
                                WholeArrayIn touching,
                                FieldOut ready);
  using ExecutionSignature = void(WorkIndex, _1, _2, _3, _4, _5, _6, _7, _8);

  template <typename StatePortal, typename IdPortal>
  VISKORES_EXEC void operator()(viskores::Id r,
                                const viskores::Id2& pair,
                                viskores::UInt8 state,
                                const StatePortal& states,
                                const IdPortal& neighborOffsets,
                                const IdPortal& neighbors,
                                const IdPortal& touchingOffsets,
                                const IdPortal& touching,
                                viskores::UInt8& ready) const
  {
    ready = 0;
    if (state != Alive)
    {
      return;
    }
    bool blocked = false;
    ForEachConflict(pair,
                    neighborOffsets,
                    neighbors,
                    touchingOffsets,
                    touching,
                    [&](viskores::Id q)
                    {
                      if (q < r && states.Get(q) == Alive)
                      {
                        blocked = true;
                        return false;
                      }
                      return true;
                    });
    ready = blocked ? 0 : 1;
  }
};

struct NeedsEvaluation : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn ready, FieldIn memo, FieldOut needed);
  using ExecutionSignature = _3(_1, _2);

  VISKORES_EXEC viskores::UInt8 operator()(viskores::UInt8 ready, viskores::UInt8 memo) const
  {
    return (ready && memo == Unknown) ? 1 : 0;
  }
};

/// The scatter of the envelope items of the merge phase: 16 visits per item, of which only the
/// first works. The TBB device schedules at least 1024 visits per task, so a launch of a few
/// thousand envelope items would otherwise run on one or a few threads; the idle visits cost
/// about 5% of the local tests on the Serial device.
using SpreadScatter = viskores::worklet::ScatterUniform<16>;

/// The local test of a merge but for the envelope, and the work of its envelope part: the
/// numbers of input triangles to test with a vertex in the merged cluster and away from it, and
/// `S (1 + 16 u)` of the two batches of tests. Also a box that holds everything the whole local
/// test reads: the members of the two clusters, the generators of their triangles of `K`, the
/// merged generator and the input triangles of the envelope test, grown by epsilon, since the
/// distances it computes are exact only up to epsilon. Nothing else of `K` or of the labelling
/// is read, but the largest coordinate of the generators.
struct EvaluateMerge : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn pair,
                                ExecObject context,
                                WholeArrayOut admissibleOut,
                                WholeArrayOut mergedOut,
                                WholeArrayOut numItemsOut,
                                WholeArrayOut scalesOut,
                                WholeArrayOut lowerOut,
                                WholeArrayOut upperOut);
  using ExecutionSignature = void(InputIndex, VisitIndex, _1, _2, _3, _4, _5, _6, _7, _8);
  using ScatterType = viskores::worklet::ScatterCounting;

  template <typename Context,
            typename FlagPortal,
            typename VecPortal,
            typename Id2Portal,
            typename ScalePortal>
  VISKORES_EXEC void operator()(viskores::Id index,
                                viskores::IdComponent visit,
                                const viskores::Id2& pair,
                                const Context& context,
                                const FlagPortal& admissibleOut,
                                const VecPortal& mergedOut,
                                const Id2Portal& numItemsOut,
                                const ScalePortal& scalesOut,
                                const VecPortal& lowerOut,
                                const VecPortal& upperOut) const
  {
    if (visit != 0)
    {
      return;
    }
    viskores::UInt8 admissible;
    Vec3 merged, lower, upper;
    viskores::Id2 numItems;
    viskores::Vec<Real, 2> scales;
    this->Evaluate(pair, context, admissible, merged, numItems, scales, lower, upper);
    admissibleOut.Set(index, admissible);
    mergedOut.Set(index, merged);
    numItemsOut.Set(index, numItems);
    scalesOut.Set(index, scales);
    lowerOut.Set(index, lower);
    upperOut.Set(index, upper);
  }

  template <typename Context>
  VISKORES_EXEC void Evaluate(const viskores::Id2& pair,
                              const Context& context,
                              viskores::UInt8& admissible,
                              Vec3& merged,
                              viskores::Id2& numItems,
                              viskores::Vec<Real, 2>& scales,
                              Vec3& lower,
                              Vec3& upper) const
  {
    const viskores::Id i = pair[0], j = pair[1];
    bool complete = true;
    admissible = context.Admissible(i, j, merged, complete) ? 1 : 0;
    numItems = viskores::Id2(0, 0);
    scales = viskores::Vec<Real, 2>(0, 0);
    lower = merged;
    upper = merged;
    auto extend = [&](const Vec3& p)
    {
      for (int a = 0; a < 3; ++a)
      {
        lower[a] = viskores::Min(lower[a], p[a]);
        upper[a] = viskores::Max(upper[a], p[a]);
      }
    };
    for (int which = 0; which < 2; ++which)
    {
      const viskores::Id c = (which == 0) ? i : j;
      for (viskores::Id k = context.MemberOffsets.Get(c); k < context.MemberOffsets.Get(c + 1); ++k)
      {
        extend(context.Points.Get(context.Members.Get(k)));
      }
      for (viskores::Id f = context.FanOffsets.Get(c); f < context.FanOffsets.Get(c + 1); ++f)
      {
        const viskores::Id3 tri = context.OutputTriangles.Get(context.Fans.Get(f));
        for (int k = 0; k < 3; ++k)
        {
          extend(context.Generators.Get(tri[k]));
        }
      }
    }
    if (admissible != 0 && context.Epsilon > 0)
    {
      Real sInput = context.Input.MaxAbsCoordinate;
      Real sOutput = context.Scale(i, j, MaxAbs(merged));
      for (viskores::IdComponent part = 0; part < 2; ++part)
      {
        context.ForEachEnvelopeItem(
          i,
          j,
          merged,
          viskores::UInt8(1 << part),
          complete,
          [&](viskores::Id, const viskores::Id3& tri, viskores::UInt8 tests)
          {
            ++numItems[part];
            for (int k = 0; k < 3; ++k)
            {
              extend(context.Points.Get(tri[k]));
              if ((tests & 1) != 0)
              {
                const viskores::Id l = context.Labels.Get(tri[k]);
                sInput = viskores::Max(
                  sInput, MaxAbs((l == i || l == j) ? merged : context.Generators.Get(l)));
              }
              if ((tests & 2) != 0)
              {
                sOutput = viskores::Max(sOutput, MaxAbs(context.Points.Get(tri[k])));
              }
            }
          });
      }
      const Real scaleFactor = 1 + 16 * UnitRoundoff;
      scales = viskores::Vec<Real, 2>(sInput * scaleFactor, sOutput * scaleFactor);
    }
    Real size = 0;
    for (int a = 0; a < 3; ++a)
    {
      size = viskores::Max(size, viskores::Max(viskores::Abs(lower[a]), viskores::Abs(upper[a])));
    }
    GrowBox(lower, upper, context.Epsilon, size);
    if (!complete)
    {
      this->RaiseError("QuadricClusteringDecimation: bounding volume hierarchy too deep.");
    }
  }
};

/// The number of input triangles of part `Part` to test for a merge, none once it has failed.
struct PartCount : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn numItems, FieldIn failed, FieldOut count);
  using ExecutionSignature = _3(_1, _2);

  VISKORES_CONT explicit PartCount(viskores::IdComponent part)
    : Part(part)
  {
  }

  VISKORES_EXEC viskores::Id operator()(const viskores::Id2& numItems,
                                        viskores::UInt32 failed) const
  {
    return failed != 0 ? 0 : numItems[this->Part];
  }

private:
  viskores::IdComponent Part;
};

/// Lists the input triangles of part `Part` of the envelope tests of the merges `First` to
/// `Last`, from position `offset - Base` on.
struct FillEnvelopeItems : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn pair,
                                FieldIn merged,
                                FieldIn numItems,
                                FieldIn offset,
                                ExecObject context,
                                WholeArrayOut candidates,
                                WholeArrayOut triangles,
                                WholeArrayOut tests);
  using ExecutionSignature = void(WorkIndex, _1, _2, _3, _4, _5, _6, _7, _8);

  VISKORES_CONT FillEnvelopeItems(viskores::UInt8 part,
                                  viskores::Id first,
                                  viskores::Id last,
                                  viskores::Id base)
    : Part(part)
    , First(first)
    , Last(last)
    , Base(base)
  {
  }

  template <typename Context, typename IdPortal, typename FlagPortal>
  VISKORES_EXEC void operator()(viskores::Id index,
                                const viskores::Id2& pair,
                                const Vec3& merged,
                                viskores::Id numItems,
                                viskores::Id offset,
                                const Context& context,
                                const IdPortal& candidates,
                                const IdPortal& triangles,
                                const FlagPortal& tests) const
  {
    if (index < this->First || index >= this->Last || numItems == 0)
    {
      return;
    }
    viskores::Id k = offset - this->Base;
    bool complete = true;
    context.ForEachEnvelopeItem(pair[0],
                                pair[1],
                                merged,
                                this->Part,
                                complete,
                                [&](viskores::Id t, const viskores::Id3&, viskores::UInt8 flags)
                                {
                                  candidates.Set(k, index);
                                  triangles.Set(k, t);
                                  tests.Set(k, flags);
                                  ++k;
                                });
  }

private:
  viskores::UInt8 Part;
  viskores::Id First;
  viskores::Id Last;
  viskores::Id Base;
};

/// One input triangle of the envelope test of a merge. Marks the merge failed if a test fails,
/// and skips the triangle if the merge has already failed: the outcome is the conjunction over
/// the triangles, whatever the order.
struct TestEnvelopeItems : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn candidate,
                                FieldIn triangle,
                                FieldIn tests,
                                WholeArrayIn pairs,
                                WholeArrayIn merged,
                                WholeArrayIn scales,
                                ExecObject context,
                                AtomicArrayInOut failed);
  using ExecutionSignature = void(VisitIndex, _1, _2, _3, _4, _5, _6, _7, _8);
  using ScatterType = SpreadScatter;

  template <typename PairPortal,
            typename VecPortal,
            typename ScalePortal,
            typename Context,
            typename AtomicPortal>
  VISKORES_EXEC void operator()(viskores::IdComponent visit,
                                viskores::Id c,
                                viskores::Id t,
                                viskores::UInt8 tests,
                                const PairPortal& pairs,
                                const VecPortal& merged,
                                const ScalePortal& scales,
                                const Context& context,
                                const AtomicPortal& failed) const
  {
    if (visit != 0)
    {
      return;
    }
    if (failed.Get(c) == 0)
    {
      const viskores::Id2 pair = pairs.Get(c);
      const viskores::Vec<Real, 2> scale = scales.Get(c);
      bool complete = true;
      if (!context.TestEnvelopeItem(
            pair[0], pair[1], merged.Get(c), t, tests, scale[0], scale[1], complete))
      {
        failed.Set(c, 1);
      }
      if (!complete)
      {
        this->RaiseError("QuadricClusteringDecimation: bounding volume hierarchy too deep.");
      }
    }
  }
};

struct ClearFailed : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn failed, FieldInOut admissible);
  using ExecutionSignature = void(_1, _2);

  VISKORES_EXEC void operator()(viskores::UInt32 failed, viskores::UInt8& admissible) const
  {
    if (failed != 0)
    {
      admissible = 0;
    }
  }
};

} // namespace detail

/// The local tests of the merges `pairs`, with the merged generators and the boxes of what every
/// test reads (`EvaluateMerge`). The test but for the envelope runs per merge; the
/// envelope tests run per input triangle, in batches of at most `maxItems` triangles, so that a
/// few merges of large clusters keep every thread busy: first the triangles with a vertex in
/// the merged cluster, where a merge that fails mostly fails, then the others, for the merges
/// still standing.
inline void EvaluateMerges(const viskores::cont::ArrayHandle<viskores::Id2>& pairs,
                           const MergeContextObject& context,
                           viskores::cont::ArrayHandle<viskores::UInt8>& admissible,
                           viskores::cont::ArrayHandle<Vec3>& merged,
                           viskores::cont::ArrayHandle<Vec3>& lower,
                           viskores::cont::ArrayHandle<Vec3>& upper)
{
  constexpr viskores::Id maxItems = viskores::Id(1) << 24;
  using Algorithm = viskores::cont::Algorithm;
  viskores::cont::Invoker invoke;
  viskores::cont::ArrayHandle<viskores::Id2> numItems;
  viskores::cont::ArrayHandle<viskores::Vec<Real, 2>> scales;
  const viskores::Id n = pairs.GetNumberOfValues();
  admissible.Allocate(n);
  merged.Allocate(n);
  numItems.Allocate(n);
  scales.Allocate(n);
  lower.Allocate(n);
  upper.Allocate(n);
  invoke(detail::EvaluateMerge{},
         SpreadScatterOver(n),
         pairs,
         context,
         admissible,
         merged,
         numItems,
         scales,
         lower,
         upper);
  viskores::cont::ArrayHandle<viskores::UInt32> failed;
  failed.AllocateAndFill(n, 0);
  viskores::cont::ArrayHandle<viskores::Id> counts, offsets, candidates, triangles;
  viskores::cont::ArrayHandle<viskores::UInt8> tests;
  for (viskores::IdComponent part = 0; part < 2; ++part)
  {
    invoke(detail::PartCount(part), numItems, failed, counts);
    const viskores::Id total = Algorithm::ScanExclusive(counts, offsets);
    if (total == 0)
    {
      continue;
    }
    // Batches of candidates by the offset of their first item; the items of one test are
    // independent, so the batches change nothing but the memory in use.
    viskores::cont::ArrayHandle<viskores::Id> batchOf, batchStarts;
    invoke(DivideBy(maxItems), offsets, batchOf);
    const viskores::Id numBatches = (total - 1) / maxItems + 1;
    Algorithm::LowerBounds(batchOf, viskores::cont::ArrayHandleIndex(numBatches + 1), batchStarts);
    for (viskores::Id batch = 0; batch < numBatches; ++batch)
    {
      const viskores::Id first = viskores::cont::ArrayGetValue(batch, batchStarts);
      const viskores::Id last = viskores::cont::ArrayGetValue(batch + 1, batchStarts);
      if (first == last)
      {
        continue;
      }
      const viskores::Id base = viskores::cont::ArrayGetValue(first, offsets);
      const viskores::Id size =
        ((last < n) ? viskores::cont::ArrayGetValue(last, offsets) : total) - base;
      if (size > 0)
      {
        candidates.Allocate(size);
        triangles.Allocate(size);
        tests.Allocate(size);
        invoke(detail::FillEnvelopeItems(viskores::UInt8(1 << part), first, last, base),
               pairs,
               merged,
               counts,
               offsets,
               context,
               candidates,
               triangles,
               tests);
        invoke(detail::TestEnvelopeItems{},
               candidates,
               triangles,
               tests,
               pairs,
               merged,
               scales,
               context,
               failed);
      }
    }
  }
  invoke(detail::ClearFailed{}, failed, admissible);
}

namespace detail
{

/// Whether an evaluated merge is a sheet merge that passed its local test.
struct IsPassingSheetMerge : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn pair,
                                FieldIn admissible,
                                WholeArrayIn frozenLabels,
                                FieldOut fixed);
  using ExecutionSignature = _4(_1, _2, _3);

  template <typename FlagPortal>
  VISKORES_EXEC viskores::UInt8 operator()(const viskores::Id2& pair,
                                           viskores::UInt8 admissible,
                                           const FlagPortal& frozenLabels) const
  {
    return (admissible && frozenLabels.Get(pair[0])) ? 1 : 0;
  }
};

/// The glue group of the frozen cluster of a sheet merge `(i, j)`, as an index of the groups,
/// and the number of labels of its region: those of the group, and `j`.
struct SheetRegionCount : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn pair,
                                WholeArrayIn glue,
                                WholeArrayIn groupKeys,
                                WholeArrayIn groupOffsets,
                                FieldOut group,
                                FieldOut count);
  using ExecutionSignature = void(_1, _2, _3, _4, _5, _6);

  template <typename IdPortal>
  VISKORES_EXEC void operator()(const viskores::Id2& pair,
                                const IdPortal& glue,
                                const IdPortal& groupKeys,
                                const IdPortal& groupOffsets,
                                viskores::Id& group,
                                viskores::Id& count) const
  {
    const viskores::Id key = glue.Get(pair[0]);
    viskores::Id lo = LowerBound(groupKeys, 0, groupKeys.GetNumberOfValues(), key);
    group = lo;
    count = groupOffsets.Get(lo + 1) - groupOffsets.Get(lo) + 1;
  }
};

/// A label of the region of a sheet merge: `(candidate, label, new label)`, the labels of the
/// group staying, `j` joining `i`.
struct SheetRegionLabel : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn item,
                                WholeArrayIn itemOffsets,
                                WholeArrayIn pairs,
                                WholeArrayIn groups,
                                WholeArrayIn groupOffsets,
                                WholeArrayIn groupLabels,
                                FieldOut record);
  using ExecutionSignature = _7(_1, _2, _3, _4, _5, _6);

  template <typename IdPortal, typename PairPortal>
  VISKORES_EXEC viskores::Id3 operator()(viskores::Id item,
                                         const IdPortal& itemOffsets,
                                         const PairPortal& pairs,
                                         const IdPortal& groups,
                                         const IdPortal& groupOffsets,
                                         const IdPortal& groupLabels) const
  {
    const viskores::Id c = OwnerOf(itemOffsets, item);
    const viskores::Id k = item - itemOffsets.Get(c);
    const viskores::Id g = groups.Get(c);
    const viskores::Id size = groupOffsets.Get(g + 1) - groupOffsets.Get(g);
    if (k < size)
    {
      const viskores::Id label = groupLabels.Get(groupOffsets.Get(g) + k);
      return viskores::Id3(c, label, label);
    }
    return viskores::Id3(c, pairs.Get(c)[1], pairs.Get(c)[0]);
  }
};

/// The relabel of a sheet merge, `(candidate, j, i)`, and its glue group, twice.
struct SheetRelabel : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn pair,
                                WholeArrayIn glue,
                                FieldOut relabel,
                                FieldOut groupGlue);
  using ExecutionSignature = void(_1, _2, _3, _4, WorkIndex);

  template <typename IdPortal>
  VISKORES_EXEC void operator()(const viskores::Id2& pair,
                                const IdPortal& glue,
                                viskores::Id3& relabel,
                                viskores::Id2& groupGlue,
                                viskores::Id c) const
  {
    relabel = viskores::Id3(c, pair[1], pair[0]);
    groupGlue = viskores::Id2(glue.Get(pair[0]), glue.Get(pair[0]));
  }
};

/// A vertex of a label of a region: `(candidate, vertex, label, new label)`.
struct RegionVertex : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn item,
                                WholeArrayIn itemOffsets,
                                WholeArrayIn records,
                                WholeArrayIn memberOffsets,
                                WholeArrayIn members,
                                FieldOut vertex);
  using ExecutionSignature = _6(_1, _2, _3, _4, _5);

  template <typename IdPortal, typename RecordPortal>
  VISKORES_EXEC viskores::Vec<viskores::Id, 4> operator()(viskores::Id item,
                                                          const IdPortal& itemOffsets,
                                                          const RecordPortal& records,
                                                          const IdPortal& memberOffsets,
                                                          const IdPortal& members) const
  {
    const viskores::Id r = OwnerOf(itemOffsets, item);
    const viskores::Id3 record = records.Get(r);
    const viskores::Id v = members.Get(memberOffsets.Get(record[1]) + item - itemOffsets.Get(r));
    return viskores::Vec<viskores::Id, 4>(record[0], v, record[1], record[2]);
  }
};

/// Clears the verdict of a sheet merge that violates (d').
struct ClearViolating : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn index, FieldIn fires, WholeArrayInOut admissible);
  using ExecutionSignature = void(_1, _2, _3);

  template <typename FlagPortal>
  VISKORES_EXEC void operator()(viskores::Id index,
                                viskores::UInt8 fires,
                                FlagPortal& admissible) const
  {
    if (fires)
    {
      admissible.Set(index, 0);
    }
  }
};

} // namespace detail

/// (d') for the sheet merges among `evaluated` that passed their local test, as
/// `merge_ok(fixed=True)` of proofs/lib/vqc.py, tested at the glued vertex of the frozen
/// cluster in every sheet (`GroupGlueFires`); clears `admissible` for those that violate it.
inline void SheetMergeGlueTest(const SurfaceMesh& mesh,
                               const MergeParameters& parameters,
                               const Clustering& clusters,
                               const MergeContextObject& context,
                               const viskores::cont::ArrayHandle<viskores::Id2>& evaluated,
                               viskores::cont::ArrayHandle<viskores::UInt8>& admissible)
{
  using Algorithm = viskores::cont::Algorithm;
  viskores::cont::Invoker invoke;
  viskores::cont::ArrayHandle<viskores::UInt8> isFixed;
  invoke(detail::IsPassingSheetMerge{}, evaluated, admissible, context.FrozenLabels, isFixed);
  viskores::cont::ArrayHandle<viskores::Id> indices;
  Algorithm::CopyIf(
    viskores::cont::ArrayHandleIndex(evaluated.GetNumberOfValues()), isFixed, indices);
  const viskores::Id n = indices.GetNumberOfValues();
  if (n == 0)
  {
    return;
  }
  viskores::cont::ArrayHandle<viskores::Id2> pairs;
  Algorithm::Copy(viskores::cont::make_ArrayHandlePermutation(indices, evaluated), pairs);
  CandidateRegions regions;
  regions.NumberOfCandidates = n;
  viskores::cont::ArrayHandle<viskores::Id> groups, counts, itemOffsets;
  invoke(detail::SheetRegionCount{},
         pairs,
         context.Glue,
         context.GroupKeys,
         context.GroupOffsets,
         groups,
         counts);
  const viskores::Id numItems = Algorithm::ScanExclusive(counts, itemOffsets);
  invoke(detail::SheetRegionLabel{},
         viskores::cont::ArrayHandleIndex(numItems),
         itemOffsets,
         pairs,
         groups,
         context.GroupOffsets,
         context.GroupLabels,
         regions.Labels);
  Algorithm::Sort(regions.Labels);
  regions.LabelOffsets = RecordOffsets(regions.Labels, n);
  regions.LabelGenerators.AllocateAndFill(numItems, Vec3(0));
  regions.LabelSizes.AllocateAndFill(numItems, 0);
  invoke(detail::SheetRelabel{}, pairs, context.Glue, regions.Relabels, regions.GroupGlue);
  regions.RelabelOffsets = RecordOffsets(regions.Relabels, n);
  viskores::cont::ArrayHandle<viskores::Id> vertexCounts, vertexOffsets;
  invoke(GatherComponent<1>{}, regions.Labels, clusters.Sizes, vertexCounts);
  const viskores::Id numVertices = Algorithm::ScanExclusive(vertexCounts, vertexOffsets);
  invoke(detail::RegionVertex{},
         viskores::cont::ArrayHandleIndex(numVertices),
         vertexOffsets,
         regions.Labels,
         clusters.Keys.GetOffsets(),
         clusters.Keys.GetSortedValuesMap(),
         regions.Vertices);
  Algorithm::Sort(regions.Vertices);

  viskores::cont::ArrayHandle<viskores::Id2> incident;
  viskores::cont::ArrayHandle<viskores::Id> groupCandidates, groupOffsets, groupSources;
  viskores::cont::ArrayHandle<viskores::Id3> groupTriangles;
  GroupTriangles(mesh,
                 clusters,
                 context.Glue,
                 regions,
                 incident,
                 groupCandidates,
                 groupTriangles,
                 groupOffsets,
                 groupSources);
  viskores::cont::ArrayHandle<viskores::UInt8> fires;
  fires.AllocateAndFill(n, 0);
  GroupGlueFires(mesh,
                 parameters.Original,
                 *parameters.Curves,
                 clusters,
                 context.Glue,
                 regions,
                 groupCandidates,
                 groupTriangles,
                 fires);
  invoke(detail::ClearViolating{}, indices, fires, admissible);
}

namespace detail
{

struct ScatterAdmissible : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn rank, FieldIn admissible, WholeArrayInOut memo);
  using ExecutionSignature = void(_1, _2, _3);

  template <typename MemoPortal>
  VISKORES_EXEC void operator()(viskores::Id rank,
                                viskores::UInt8 admissible,
                                MemoPortal& memo) const
  {
    memo.Set(rank, admissible ? Admissible : Inadmissible);
  }
};

/// A ready candidate is committed if admissible and rejected otherwise.
struct DecideReady : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn ready, FieldIn memo, FieldInOut state);
  using ExecutionSignature = void(_1, _2, _3);

  VISKORES_EXEC void operator()(viskores::UInt8 ready,
                                viskores::UInt8 memo,
                                viskores::UInt8& state) const
  {
    if (ready)
    {
      state = (memo == Admissible) ? Committed : Rejected;
    }
  }
};

/// A live candidate conflicting with a committed one dies.
struct KillConflicting : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn pair,
                                FieldInOut state,
                                WholeArrayIn states,
                                WholeArrayIn neighborOffsets,
                                WholeArrayIn neighbors,
                                WholeArrayIn touchingOffsets,
                                WholeArrayIn touching);
  using ExecutionSignature = void(_1, _2, _3, _4, _5, _6, _7);

  template <typename StatePortal, typename IdPortal>
  VISKORES_EXEC void operator()(const viskores::Id2& pair,
                                viskores::UInt8& state,
                                const StatePortal& states,
                                const IdPortal& neighborOffsets,
                                const IdPortal& neighbors,
                                const IdPortal& touchingOffsets,
                                const IdPortal& touching) const
  {
    if (state != Alive)
    {
      return;
    }
    ForEachConflict(pair,
                    neighborOffsets,
                    neighbors,
                    touchingOffsets,
                    touching,
                    [&](viskores::Id q)
                    {
                      if (states.Get(q) == Committed)
                      {
                        state = Dead;
                        return false;
                      }
                      return true;
                    });
  }
};

struct MergeInto : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn pair, WholeArrayInOut map);
  using ExecutionSignature = void(_1, _2);

  template <typename MapPortal>
  VISKORES_EXEC void operator()(const viskores::Id2& pair, MapPortal& map) const
  {
    map.Set(pair[1], pair[0]);
  }
};

} // namespace detail

/// The placement step, and with the singular curves decimated, the generators of the vertices
/// of `K` on the curves.
inline void PlaceGlued(const SurfaceMesh& mesh,
                       const VertexQuadrics& quadrics,
                       const MergeParameters& parameters,
                       Clustering& clusters)
{
  Place(mesh, quadrics.M, clusters);
  if (parameters.Curves)
  {
    PlaceCurveGenerators(mesh,
                         quadrics.M,
                         *parameters.Curves,
                         parameters.FrozenVertices,
                         parameters.Original,
                         OutputGlue(parameters, clusters),
                         clusters);
  }
}

/// Merges the pairs `pairs` (each `(i, j)` merging `j` into `i`) and places the generators.
inline Clustering ApplyMerges(const SurfaceMesh& mesh,
                              const VertexQuadrics& quadrics,
                              const MergeParameters& parameters,
                              const Clustering& clusters,
                              const viskores::cont::ArrayHandle<viskores::Id2>& pairs)
{
  viskores::cont::Invoker invoke;
  viskores::cont::ArrayHandle<viskores::Id> map;
  viskores::cont::Algorithm::Copy(viskores::cont::ArrayHandleIndex(clusters.NumberOfClusters), map);
  invoke(detail::MergeInto{}, pairs, map);
  Clustering merged;
  invoke(GatherValue{}, clusters.Labels, map, merged.Labels);
  PlaceGlued(mesh, quadrics, parameters, merged);
  return merged;
}

namespace detail
{

/// Marks the cluster that a merge `(i, j)` became, by the label its anchor has after the merge.
struct MarkMerged : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn pair,
                                WholeArrayIn anchors,
                                WholeArrayIn nextLabels,
                                WholeArrayOut changed);
  using ExecutionSignature = void(_1, _2, _3, _4);

  template <typename IdPortal, typename FlagPortal>
  VISKORES_EXEC void operator()(const viskores::Id2& pair,
                                const IdPortal& anchors,
                                const IdPortal& nextLabels,
                                const FlagPortal& changed) const
  {
    changed.Set(nextLabels.Get(anchors.Get(pair[0])), 1);
  }
};

} // namespace detail

/// The clusters of `next` that the merges `pairs` of `previous` made.
inline viskores::cont::ArrayHandle<viskores::UInt8> MergedClusters(
  const Clustering& previous,
  const Clustering& next,
  const viskores::cont::ArrayHandle<viskores::Id2>& pairs)
{
  viskores::cont::ArrayHandle<viskores::UInt8> changed;
  changed.AllocateAndFill(next.NumberOfClusters, 0);
  viskores::cont::Invoker invoke;
  invoke(detail::MarkMerged{}, pairs, previous.Anchors, next.Labels, changed);
  return changed;
}

namespace detail
{

struct HasChangedLabel : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn triangle, WholeArrayIn changed, FieldOut isNew);
  using ExecutionSignature = _3(_1, _2);

  template <typename FlagPortal>
  VISKORES_EXEC viskores::UInt8 operator()(const viskores::Id3& t, const FlagPortal& changed) const
  {
    return (changed.Get(t[0]) || changed.Get(t[1]) || changed.Get(t[2])) ? 1 : 0;
  }
};

/// For a new triangle `t`: whether it is degenerate or meets another nondegenerate triangle
/// improperly, as `FindImproperPairs` decides.
struct NewTriangleFlagged : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn index,
                                WholeArrayIn triangles,
                                WholeArrayIn lower,
                                WholeArrayIn upper,
                                WholeArrayIn degenerates,
                                WholeArrayIn generators,
                                ExecObject hierarchy,
                                FieldOut flagged);
  using ExecutionSignature = void(_1, _2, _3, _4, _5, _6, _7, _8);

  template <typename Id3Portal, typename VecPortal, typename FlagPortal, typename Hierarchy>
  VISKORES_EXEC void operator()(viskores::Id t,
                                const Id3Portal& triangles,
                                const VecPortal& lower,
                                const VecPortal& upper,
                                const FlagPortal& degenerates,
                                const VecPortal& generators,
                                const Hierarchy& hierarchy,
                                viskores::UInt8& flagged) const
  {
    flagged = degenerates.Get(t);
    if (flagged)
    {
      return;
    }
    const viskores::Id3 triangle = triangles.Get(t);
    const Vec3 p[3] = { generators.Get(triangle[0]),
                        generators.Get(triangle[1]),
                        generators.Get(triangle[2]) };
    const bool complete = hierarchy.ForEachOverlap(lower.Get(t),
                                                   upper.Get(t),
                                                   [&](viskores::Id j)
                                                   {
                                                     if (j == t || degenerates.Get(j))
                                                     {
                                                       return true;
                                                     }
                                                     const viskores::Id3 other = triangles.Get(j);
                                                     const Vec3 q[3] = { generators.Get(other[0]),
                                                                         generators.Get(other[1]),
                                                                         generators.Get(other[2]) };
                                                     if (exact::ImproperPair(p, triangle, q, other))
                                                     {
                                                       flagged = 1;
                                                       return false;
                                                     }
                                                     return true;
                                                   });
    if (!complete)
    {
      this->RaiseError("QuadricClusteringDecimation: bounding volume hierarchy too deep.");
    }
  }
};

/// The group of a vertex, and whether the group's generator is at the vertex.
struct VertexGroup : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn label,
                                WholeArrayIn glue,
                                WholeArrayIn generators,
                                FieldIn point,
                                FieldOut group,
                                FieldOut atPoint);
  using ExecutionSignature = void(_1, _2, _3, _4, _5, _6);

  template <typename IdPortal, typename VecPortal>
  VISKORES_EXEC void operator()(viskores::Id label,
                                const IdPortal& glue,
                                const VecPortal& generators,
                                const Vec3& point,
                                viskores::Id& group,
                                viskores::UInt8& atPoint) const
  {
    group = glue.Get(label);
    atPoint = (generators.Get(group) == point) ? 1 : 0;
  }
};

/// Whether the members of a group are copies of one input vertex, every one at the generator.
struct GroupIsOneVertex : viskores::worklet::WorkletReduceByKey
{
  using ControlSignature = void(KeysIn group,
                                ValuesIn original,
                                ValuesIn atPoint,
                                ReducedValuesOut one);
  using ExecutionSignature = _4(_2, _3);

  template <typename OriginalVec, typename FlagVec>
  VISKORES_EXEC viskores::UInt8 operator()(const OriginalVec& original,
                                           const FlagVec& atPoint) const
  {
    for (viskores::IdComponent i = 0; i < original.GetNumberOfComponents(); ++i)
    {
      if (original[i] != original[0] || !atPoint[i])
      {
        return 0;
      }
    }
    return 1;
  }
};

/// Whether a triangle in a violation has a corner that is not one input vertex at itself.
struct NotInherited : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn triangle, FieldIn involved, WholeArrayIn one, FieldOut bad);
  using ExecutionSignature = _4(_1, _2, _3);

  template <typename FlagPortal>
  VISKORES_EXEC viskores::Id operator()(const viskores::Id3& t,
                                        viskores::UInt8 involved,
                                        const FlagPortal& one) const
  {
    return (involved && !(one.Get(t[0]) && one.Get(t[1]) && one.Get(t[2]))) ? 1 : 0;
  }
};

} // namespace detail

/// For an input cut into surfaces: whether the glued output `glued` has a degenerate triangle,
/// or a triangle in an improper pair, that is not inherited from the input, as `new_improper` of
/// proofs/lib/arcs.py. A triangle is inherited when every corner is a vertex of `K` made of the
/// copies of one input vertex and realised at that vertex. (F4) flags only the clusters with two
/// or more vertices, and misses a violation whose group is represented by a singleton label
/// although the group is not one vertex.
inline bool NewImproper(const SurfaceMesh& mesh,
                        const MergeParameters& parameters,
                        const Clustering& clusters,
                        const viskores::cont::ArrayHandle<viskores::Id>& glue,
                        const viskores::cont::ArrayHandle<viskores::Id3>& glued)
{
  using Algorithm = viskores::cont::Algorithm;
  viskores::cont::Invoker invoke;
  if (glued.GetNumberOfValues() == 0)
  {
    return false;
  }
  viskores::cont::ArrayHandle<Vec3> lower, upper;
  viskores::cont::ArrayHandle<viskores::UInt8> degenerate, involved;
  viskores::cont::ArrayHandle<viskores::Id> pairsAbove;
  invoke(detail::RealizeTriangle{}, glued, clusters.Generators, lower, upper, degenerate);
  const viskores::worklet::spatialstructure::BoundingVolumeHierarchy hierarchy(lower, upper);
  invoke(detail::FindImproperPairs{},
         glued,
         lower,
         upper,
         degenerate,
         glued,
         degenerate,
         clusters.Generators,
         hierarchy,
         involved,
         pairsAbove);
  if (Algorithm::Reduce(involved, viskores::Id(0)) == 0)
  {
    return false;
  }
  // Every group made of the copies of one input vertex, realised at it.
  viskores::cont::ArrayHandle<viskores::Id> groupOfVertex;
  viskores::cont::ArrayHandle<viskores::UInt8> atPoint;
  invoke(detail::VertexGroup{},
         clusters.Labels,
         glue,
         clusters.Generators,
         mesh.Points,
         groupOfVertex,
         atPoint);
  const viskores::worklet::Keys<viskores::Id> groups(groupOfVertex);
  viskores::cont::ArrayHandle<viskores::UInt8> groupIsOne, one;
  invoke(detail::GroupIsOneVertex{}, groups, parameters.Original, atPoint, groupIsOne);
  one.AllocateAndFill(clusters.NumberOfClusters, 1);
  invoke(ScatterValue{}, groups.GetUniqueKeys(), groupIsOne, one);
  viskores::cont::ArrayHandle<viskores::UInt8> bad;
  invoke(detail::NotInherited{}, glued, involved, one, bad);
  return Algorithm::Reduce(bad, viskores::Id(0)) > 0;
}

/// Whether any flag in use fires on a placed labelling, evaluated in full: the discrete
/// closed-ball property, the glue condition, (Fc), (F0), (F4), (F5), (F7) and (F8). With
/// `bases`, (F5) tests only the triangles whose verdicts can differ from a base's
/// (`MergeEnvelopeFlags`); with `verdicts`, its verdicts are stored there, for a caller that
/// commits the labelling or keeps it as a base.
///
/// With `changed`, the clusters that merges made from a labelling on which no flag fired, (F4)
/// tests only the triangles at them: every other triangle, and every pair of them, is as
/// it was, and a failing triangle or pair flags a cluster only if one of its clusters has two or
/// more vertices, which a merged cluster has.
inline bool FlagsFire(const SurfaceMesh& mesh,
                      const VertexQuadrics& quadrics,
                      const DistanceMesh& inputDistance,
                      const Clustering& clusters,
                      const MergeParameters& parameters,
                      const viskores::cont::ArrayHandle<viskores::UInt8>* changed = nullptr,
                      const EnvelopeBases* bases = nullptr,
                      EnvelopeVerdicts* verdicts = nullptr)
{
  const viskores::Id k = clusters.NumberOfClusters;
  viskores::cont::ArrayHandle<viskores::UInt8> flags;
  if (parameters.Topological)
  {
    flags.AllocateAndFill(k, 0);
    const TopologyReport report =
      TopologyFlags(mesh, clusters.Labels, clusters.Sizes, flags, parameters.D2ByCount);
    if (report.NumberOfD0 + report.NumberOfD1 + report.NumberOfD2 > 0)
    {
      return true;
    }
    if (parameters.Curves)
    {
      // (d'): any violation fires.
      if (GlueViolations(mesh,
                         clusters.Labels,
                         parameters.Original,
                         *parameters.Curves,
                         OutputGlue(parameters, clusters))
            .GetNumberOfValues() > 0)
      {
        return true;
      }
      // (Fc)
      if (parameters.CurveTolerance > 0 &&
          CurveTestFails(mesh,
                         *parameters.Curves,
                         parameters.Original,
                         clusters,
                         parameters.CurveTolerance,
                         parameters.CurveDepth))
      {
        return true;
      }
    }
    else if (GlueFlags(mesh,
                       clusters.Labels,
                       parameters.FrozenVertices,
                       parameters.Canonical,
                       clusters.Sizes,
                       flags) > 0)
    {
      return true;
    }
  }
  if (parameters.MaxClusterRadius < std::numeric_limits<Real>::infinity())
  {
    flags.AllocateAndFill(k, 0);
    if (RadiusFlags(Scan(mesh, quadrics.M, clusters), parameters.MaxClusterRadius, flags) > 0)
    {
      return true;
    }
  }
  const viskores::cont::ArrayHandle<viskores::Id3> triangles = parameters.Topological
    ? OutputTriangles(mesh, clusters.Labels)
    : Simplices(mesh, clusters.Labels);
  if (parameters.Embedding && parameters.Topological)
  {
    const viskores::cont::ArrayHandle<viskores::Id> glue = OutputGlue(parameters, clusters);
    const viskores::cont::ArrayHandle<viskores::Id3> glued = GlueTriangles(triangles, glue);
    if (changed)
    {
      viskores::cont::Invoker invoke;
      viskores::cont::ArrayHandle<Vec3> lower, upper;
      viskores::cont::ArrayHandle<viskores::UInt8> degenerate, isNew, flagged;
      invoke(detail::RealizeTriangle{}, glued, clusters.Generators, lower, upper, degenerate);
      // The glued triangles carry the label of their group: a group changes with any member.
      viskores::cont::ArrayHandle<viskores::UInt8> changedGroups = *changed;
      if (glue.GetNumberOfValues() > 0)
      {
        viskores::cont::ArrayHandle<viskores::UInt8> groups;
        viskores::cont::Algorithm::Copy(*changed, groups);
        invoke(MarkIfFlagged{}, *changed, glue, groups);
        changedGroups = groups;
      }
      invoke(detail::HasChangedLabel{}, glued, changedGroups, isNew);
      viskores::cont::ArrayHandle<viskores::Id> newTriangles;
      viskores::cont::Algorithm::CopyIf(
        viskores::cont::ArrayHandleIndex(glued.GetNumberOfValues()), isNew, newTriangles);
      const viskores::worklet::spatialstructure::BoundingVolumeHierarchy hierarchy(lower, upper);
      invoke(detail::NewTriangleFlagged{},
             newTriangles,
             glued,
             lower,
             upper,
             degenerate,
             clusters.Generators,
             hierarchy,
             flagged);
      if (viskores::cont::Algorithm::Reduce(flagged, viskores::Id(0)) > 0)
      {
        return true;
      }
    }
    else if (glue.GetNumberOfValues() > 0)
    {
      if (NewImproper(mesh, parameters, clusters, glue, glued))
      {
        return true;
      }
    }
    else
    {
      flags.AllocateAndFill(k, 0);
      EmbeddingFlags(glued, clusters.Generators, clusters.Sizes, flags);
      if (CountFlags(flags) > 0)
      {
        return true;
      }
    }
  }
  if (parameters.ErrorBound > 0)
  {
    flags.AllocateAndFill(k, 0);
    MergeEnvelopeFlags(
      mesh, inputDistance, clusters, triangles, parameters, flags, bases, verdicts);
    if (CountFlags(flags) > 0)
    {
      return true;
    }
  }
  if (parameters.CertifiedThickening)
  {
    flags.AllocateAndFill(k, 0);
    ThickeningFlags(mesh,
                    *parameters.CertifiedThickening,
                    clusters.Labels,
                    clusters.Generators,
                    clusters.Sizes,
                    flags,
                    OutputGlue(parameters, clusters));
    if (CountFlags(flags) > 0)
    {
      return true;
    }
    // (F8)
    if (parameters.CertifiedThickening->Fibres &&
        FibreFlags(mesh,
                   *parameters.CertifiedThickening,
                   clusters.Labels,
                   clusters.Generators,
                   clusters.Sizes,
                   flags) > 0)
    {
      return true;
    }
  }
  return false;
}

/// The failing items of the full test on a labelling that the region blame reads: the input
/// triangles that fail (F5) from the input side (`input_fails` of proofs/lib/vqc.py), empty
/// without (F5), and the round of (F7), whose probes may fail, with the glue it read.
struct FailingItems
{
  viskores::cont::ArrayHandle<viskores::UInt8> FailedInput;
  bool HasThickeningRound = false;
  ThickeningRound Round;
  viskores::cont::ArrayHandle<viskores::Id> Glue;
};

/// The clusters that the flags in use flag on a placed labelling, as `flagged_clusters` of
/// proofs/lib/vqc.py: (F0), the glue condition, (Fc), (F4), (F5), (F7) and (F8). Returns
/// false, with no flags, if the discrete closed-ball property fails, which Lemma
/// "mergeindependent" excludes for a committed set of cluster merges; with `blameTopology`, for
/// a batch of segment merges, which can break it, flags every cluster a violation names, of any
/// size, and the others as well (`blame_cbp`). With `items`, stores the failing items that the
/// region blame reads.
inline bool FlaggedClusters(const SurfaceMesh& mesh,
                            const VertexQuadrics& quadrics,
                            const DistanceMesh& inputDistance,
                            const Clustering& clusters,
                            const MergeParameters& parameters,
                            viskores::cont::ArrayHandle<viskores::UInt8>& flags,
                            bool blameTopology = false,
                            const EnvelopeBases* bases = nullptr,
                            const EnvelopeVerdicts* known = nullptr,
                            FailingItems* items = nullptr)
{
  const viskores::Id k = clusters.NumberOfClusters;
  flags.AllocateAndFill(k, 0);
  viskores::cont::ArrayHandle<viskores::UInt8> other;
  if (parameters.Topological)
  {
    other.AllocateAndFill(k, 0);
    viskores::cont::ArrayHandle<viskores::Id> named;
    const TopologyReport report = TopologyFlags(mesh,
                                                clusters.Labels,
                                                clusters.Sizes,
                                                other,
                                                parameters.D2ByCount,
                                                blameTopology ? &named : nullptr);
    if (report.NumberOfD0 + report.NumberOfD1 + report.NumberOfD2 > 0)
    {
      if (!blameTopology)
      {
        return false;
      }
      viskores::cont::Invoker invoke;
      invoke(MarkIndex{}, named, flags);
    }
    other.AllocateAndFill(k, 0);
    if (parameters.Curves)
    {
      viskores::cont::ArrayHandle<viskores::Id> listed =
        GlueViolations(mesh,
                       clusters.Labels,
                       parameters.Original,
                       *parameters.Curves,
                       OutputGlue(parameters, clusters));
      FlagClusters(listed, clusters.Sizes, other);
      if (parameters.CurveTolerance > 0)
      {
        CurveFlags(mesh,
                   *parameters.Curves,
                   parameters.Original,
                   clusters,
                   parameters.CurveTolerance,
                   parameters.CurveDepth,
                   other);
      }
    }
    else
    {
      GlueFlags(mesh,
                clusters.Labels,
                parameters.FrozenVertices,
                parameters.Canonical,
                clusters.Sizes,
                other);
    }
    OrFlags(flags, other);
  }
  if (parameters.MaxClusterRadius < std::numeric_limits<Real>::infinity())
  {
    other.AllocateAndFill(k, 0);
    RadiusFlags(Scan(mesh, quadrics.M, clusters), parameters.MaxClusterRadius, other);
    OrFlags(flags, other);
  }
  const viskores::cont::ArrayHandle<viskores::Id3> triangles = parameters.Topological
    ? OutputTriangles(mesh, clusters.Labels)
    : Simplices(mesh, clusters.Labels);
  if (parameters.Embedding && parameters.Topological)
  {
    other.AllocateAndFill(k, 0);
    EmbeddingFlags(GlueTriangles(triangles, OutputGlue(parameters, clusters)),
                   clusters.Generators,
                   clusters.Sizes,
                   other);
    OrFlags(flags, other);
  }
  if (parameters.ErrorBound > 0)
  {
    other.AllocateAndFill(k, 0);
    if (known && known->FailedOutput.GetNumberOfValues() > 0)
    {
      // The verdicts of `FlagsFire` on this labelling.
      FlagFailedTriangles(mesh, clusters.Labels, clusters.Sizes, *known, other);
      if (items)
      {
        items->FailedInput = known->FailedInput;
      }
    }
    else
    {
      EnvelopeVerdicts verdicts;
      MergeEnvelopeFlags(mesh,
                         inputDistance,
                         clusters,
                         triangles,
                         parameters,
                         other,
                         bases,
                         items ? &verdicts : nullptr);
      if (items)
      {
        items->FailedInput = verdicts.FailedInput;
      }
    }
    OrFlags(flags, other);
  }
  if (parameters.CertifiedThickening)
  {
    other.AllocateAndFill(k, 0);
    const viskores::cont::ArrayHandle<viskores::Id> glue = OutputGlue(parameters, clusters);
    ThickeningFlags(mesh,
                    *parameters.CertifiedThickening,
                    clusters.Labels,
                    clusters.Generators,
                    clusters.Sizes,
                    other,
                    glue,
                    items ? &items->Round : nullptr);
    if (items)
    {
      items->HasThickeningRound = true;
      items->Glue = GlueOrIdentity(glue, k);
    }
    OrFlags(flags, other);
    if (parameters.CertifiedThickening->Fibres)
    {
      other.AllocateAndFill(k, 0);
      FibreFlags(mesh,
                 *parameters.CertifiedThickening,
                 clusters.Labels,
                 clusters.Generators,
                 clusters.Sizes,
                 other);
      OrFlags(flags, other);
    }
  }
  return true;
}

namespace detail
{

/// Whether the cluster that a merge `(i, j)` became is flagged, by the label its anchor has after
/// the merge.
struct MergedFlagged : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn pair,
                                WholeArrayIn anchors,
                                WholeArrayIn nextLabels,
                                WholeArrayIn flags,
                                FieldOut flagged);
  using ExecutionSignature = _5(_1, _2, _3, _4);

  template <typename IdPortal, typename FlagPortal>
  VISKORES_EXEC viskores::UInt8 operator()(const viskores::Id2& pair,
                                           const IdPortal& anchors,
                                           const IdPortal& nextLabels,
                                           const FlagPortal& flags) const
  {
    return flags.Get(nextLabels.Get(anchors.Get(pair[0])));
  }
};

/// The cluster that a merge `(i, j)` became, by the label its anchor has after the merge.
struct MergedLabel : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn pair,
                                WholeArrayIn anchors,
                                WholeArrayIn nextLabels,
                                FieldOut label);
  using ExecutionSignature = _4(_1, _2, _3);

  template <typename IdPortal>
  VISKORES_EXEC viskores::Id operator()(const viskores::Id2& pair,
                                        const IdPortal& anchors,
                                        const IdPortal& nextLabels) const
  {
    return nextLabels.Get(anchors.Get(pair[0]));
  }
};

/// The region that an input triangle failing (F5) from the input side reads, if it flags no
/// changed cluster: its box grown by `epsilon`, rounded to nearest (`blame_regions` of
/// proofs/lib/vqc.py).
struct InputRegion : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn triangle,
                                FieldIn failed,
                                WholeArrayIn labels,
                                WholeArrayIn sizes,
                                WholeArrayIn changed,
                                WholeArrayIn points,
                                FieldOut keep,
                                FieldOut lower,
                                FieldOut upper);
  using ExecutionSignature = void(_1, _2, _3, _4, _5, _6, _7, _8, _9);

  VISKORES_CONT explicit InputRegion(Real epsilon)
    : Epsilon(epsilon)
  {
  }

  template <typename IdPortal, typename FlagPortal, typename VecPortal>
  VISKORES_EXEC void operator()(const viskores::Id3& t,
                                viskores::UInt8 failed,
                                const IdPortal& labels,
                                const IdPortal& sizes,
                                const FlagPortal& changed,
                                const VecPortal& points,
                                viskores::UInt8& keep,
                                Vec3& lower,
                                Vec3& upper) const
  {
    keep = 0;
    lower = upper = Vec3(0);
    if (!failed)
    {
      return;
    }
    for (int a = 0; a < 3; ++a)
    {
      const viskores::Id l = labels.Get(t[a]);
      if (sizes.Get(l) > 1 && changed.Get(l))
      {
        return;
      }
    }
    TriangleBounds(points.Get(t[0]), points.Get(t[1]), points.Get(t[2]), lower, upper);
    for (int a = 0; a < 3; ++a)
    {
      lower[a] = lower[a] - this->Epsilon;
      upper[a] = upper[a] + this->Epsilon;
    }
    keep = 1;
  }

  Real Epsilon;
};

/// 1 for a probe that fails (K2): a crossing is not generic, or their number is even.
struct ProbeFails : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn crossings, FieldIn nongeneric, FieldOut fails);
  using ExecutionSignature = _3(_1, _2);

  VISKORES_EXEC viskores::UInt8 operator()(viskores::Id crossings, viskores::Id nongeneric) const
  {
    return (nongeneric == 0 && crossings % 2 == 1) ? 0 : 1;
  }
};

/// Marks every failing probe that fails a 3-label input triangle with a changed cluster of at
/// least two vertices: the probe meets its output triangle, or has the label of a vertex of the
/// probe's triangle, read through `glue` (`probe_fails` of `ThickVQC.iso_pass` of
/// proofs/lib/thick.py).
struct ProbeFlagsChanged : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn triangle,
                                WholeArrayIn inputTriangles,
                                WholeArrayIn labels,
                                WholeArrayIn glue,
                                WholeArrayIn generators,
                                WholeArrayIn sizes,
                                WholeArrayIn changed,
                                WholeArrayIn fails,
                                ExecObject thickening,
                                AtomicArrayInOut flagsChanged);
  using ExecutionSignature = void(_1, _2, _3, _4, _5, _6, _7, _8, _9, _10);

  template <typename Id3Portal,
            typename IdPortal,
            typename VecPortal,
            typename FlagPortal,
            typename Thick,
            typename AtomicPortal>
  VISKORES_EXEC void operator()(const viskores::Id3& triangle,
                                const Id3Portal& inputTriangles,
                                const IdPortal& labels,
                                const IdPortal& glue,
                                const VecPortal& generators,
                                const IdPortal& sizes,
                                const FlagPortal& changed,
                                const FlagPortal& fails,
                                const Thick& thickening,
                                const AtomicPortal& flagsChanged) const
  {
    const viskores::Id l[3] = { labels.Get(triangle[0]),
                                labels.Get(triangle[1]),
                                labels.Get(triangle[2]) };
    if (!AllDistinct(l[0], l[1], l[2]))
    {
      return;
    }
    bool touches = false;
    for (int a = 0; a < 3; ++a)
    {
      touches = touches || (sizes.Get(l[a]) > 1 && changed.Get(l[a]));
    }
    if (!touches)
    {
      return;
    }
    const Vec3 p[3] = { generators.Get(l[0]), generators.Get(l[1]), generators.Get(l[2]) };
    bool fits = true;
    for (viskores::Id i = 0; i < fails.GetNumberOfValues(); ++i)
    {
      if (!fails.Get(i) || flagsChanged.Get(i) != 0)
      {
        continue;
      }
      const viskores::Id3 f = inputTriangles.Get(thickening.ProbeTriangles.Get(i));
      bool marked = false;
      for (int a = 0; a < 3 && !marked; ++a)
      {
        const viskores::Id pl = glue.Get(labels.Get(f[a]));
        marked = pl == glue.Get(l[0]) || pl == glue.Get(l[1]) || pl == glue.Get(l[2]);
      }
      marked = marked || thickening.ProbeHit(i, p, fits).Hit;
      if (marked)
      {
        flagsChanged.Set(i, 1);
      }
    }
    if (!fits)
    {
      this->RaiseError("QuadricClusteringDecimation: exact evaluation out of range.");
    }
  }
};

/// The region a failing probe that flags no changed cluster reads: the box of its segment.
struct ProbeRegion : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn probe,
                                FieldIn fails,
                                FieldIn flagsChanged,
                                ExecObject thickening,
                                FieldOut keep,
                                FieldOut lower,
                                FieldOut upper);
  using ExecutionSignature = void(_1, _2, _3, _4, _5, _6, _7);

  template <typename Thick>
  VISKORES_EXEC void operator()(viskores::Id i,
                                viskores::UInt8 fails,
                                viskores::UInt32 flagsChanged,
                                const Thick& thickening,
                                viskores::UInt8& keep,
                                Vec3& lower,
                                Vec3& upper) const
  {
    keep = (fails && flagsChanged == 0) ? 1 : 0;
    lower = upper = Vec3(0);
    if (keep)
    {
      thickening.ProbeBox(i, lower, upper);
    }
  }
};

/// Marks the labels of every output triangle whose box meets a region.
struct MarkRegionHits : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn triangle,
                                WholeArrayIn generators,
                                ExecObject regions,
                                WholeArrayIn lower,
                                WholeArrayIn upper,
                                AtomicArrayInOut hits);
  using ExecutionSignature = void(_1, _2, _3, _4, _5, _6);

  template <typename VecPortal, typename Hierarchy, typename AtomicPortal>
  VISKORES_EXEC void operator()(const viskores::Id3& t,
                                const VecPortal& generators,
                                const Hierarchy& regions,
                                const VecPortal& lower,
                                const VecPortal& upper,
                                const AtomicPortal& hits) const
  {
    Vec3 lo, hi;
    TriangleBounds(generators.Get(t[0]), generators.Get(t[1]), generators.Get(t[2]), lo, hi);
    bool hit = false;
    const bool done = regions.ForEachOverlap(lo,
                                             hi,
                                             [&](viskores::Id r)
                                             {
                                               const Vec3 rl = lower.Get(r), rh = upper.Get(r);
                                               hit = lo[0] <= rh[0] && hi[0] >= rl[0] &&
                                                 lo[1] <= rh[1] && hi[1] >= rl[1] &&
                                                 lo[2] <= rh[2] && hi[2] >= rl[2];
                                               return !hit;
                                             });
    if (!done)
    {
      this->RaiseError("QuadricClusteringDecimation: hierarchy traversal overflowed.");
    }
    if (hit)
    {
      for (int a = 0; a < 3; ++a)
      {
        hits.Set(t[a], 1);
      }
    }
  }
};

/// Marks the cluster of a vertex in the batch labelling when its cluster in the base labelling
/// is marked.
struct SpreadBaseHits : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn baseLabel,
                                FieldIn batchLabel,
                                WholeArrayIn baseHits,
                                AtomicArrayInOut marks);
  using ExecutionSignature = void(_1, _2, _3, _4);

  template <typename HitPortal, typename AtomicPortal>
  VISKORES_EXEC void operator()(viskores::Id baseLabel,
                                viskores::Id batchLabel,
                                const HitPortal& baseHits,
                                const AtomicPortal& marks) const
  {
    if (baseHits.Get(baseLabel) != 0)
    {
      marks.Set(batchLabel, 1);
    }
  }
};

/// 1 where a mark is set.
struct NonzeroMark : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn mark, FieldOut flag);
  using ExecutionSignature = _2(_1);

  VISKORES_EXEC viskores::UInt8 operator()(viskores::UInt32 mark) const
  {
    return mark != 0 ? 1 : 0;
  }
};

} // namespace detail

/// The region blame of a batch `batch` of merges of `base` that failed the full test, with
/// `changed` its changed clusters and `items` its failing items (`region_culprits` of
/// proofs/lib/vqc.py): per cluster of `batch`, 1 if an output triangle of `base` at the cluster
/// of one of its vertices has a box that meets a region read by a failing item that flags no
/// changed cluster. Such an item passed on `base`, and reads only output within its region: the
/// regions of the input triangles failing (F5) from the input side, their boxes grown by
/// `epsilon`, and the boxes of the failing probes of (F7). A new triangle that meets a probe is at
/// a changed cluster, which the probe then flags, so the triangles of `batch` are not read.
inline viskores::cont::ArrayHandle<viskores::UInt8> RegionFlags(
  const SurfaceMesh& mesh,
  const MergeParameters& parameters,
  const Clustering& base,
  const Clustering& batch,
  const viskores::cont::ArrayHandle<viskores::UInt8>& changed,
  const FailingItems& items)
{
  using Algorithm = viskores::cont::Algorithm;
  viskores::cont::Invoker invoke;
  const viskores::Id k = batch.NumberOfClusters;
  viskores::cont::ArrayHandle<viskores::UInt8> flags;
  viskores::cont::ArrayHandle<Vec3> lower, upper;
  if (items.FailedInput.GetNumberOfValues() > 0)
  {
    viskores::cont::ArrayHandle<viskores::UInt8> keep;
    viskores::cont::ArrayHandle<Vec3> lo, hi;
    invoke(detail::InputRegion(parameters.ErrorBound),
           mesh.Triangles,
           items.FailedInput,
           batch.Labels,
           batch.Sizes,
           changed,
           mesh.Points,
           keep,
           lo,
           hi);
    Algorithm::CopyIf(lo, keep, lower);
    Algorithm::CopyIf(hi, keep, upper);
  }
  const viskores::Id numProbes = items.Round.ProbeCrossings.GetNumberOfValues();
  if (items.HasThickeningRound && numProbes > 0)
  {
    const ThickeningObject thickening(*parameters.CertifiedThickening);
    viskores::cont::ArrayHandle<viskores::UInt8> fails;
    invoke(detail::ProbeFails{}, items.Round.ProbeCrossings, items.Round.ProbeNongeneric, fails);
    if (Algorithm::Reduce(fails, viskores::Id(0)) > 0)
    {
      viskores::cont::ArrayHandle<viskores::UInt32> flagsChanged;
      flagsChanged.AllocateAndFill(numProbes, 0);
      invoke(detail::ProbeFlagsChanged{},
             mesh.Triangles,
             mesh.Triangles,
             batch.Labels,
             items.Glue,
             batch.Generators,
             batch.Sizes,
             changed,
             fails,
             thickening,
             flagsChanged);
      viskores::cont::ArrayHandle<viskores::UInt8> keep;
      viskores::cont::ArrayHandle<Vec3> lo, hi, probeLower, probeUpper;
      invoke(detail::ProbeRegion{},
             viskores::cont::ArrayHandleIndex(numProbes),
             fails,
             flagsChanged,
             thickening,
             keep,
             lo,
             hi);
      Algorithm::CopyIf(lo, keep, probeLower);
      Algorithm::CopyIf(hi, keep, probeUpper);
      viskores::cont::ArrayHandle<Vec3> allLower, allUpper;
      Algorithm::Copy(viskores::cont::make_ArrayHandleConcatenate(lower, probeLower), allLower);
      Algorithm::Copy(viskores::cont::make_ArrayHandleConcatenate(upper, probeUpper), allUpper);
      lower = allLower;
      upper = allUpper;
    }
  }
  if (lower.GetNumberOfValues() == 0)
  {
    flags.AllocateAndFill(k, 0);
    return flags;
  }
  const viskores::worklet::spatialstructure::BoundingVolumeHierarchy regions(lower, upper);
  viskores::cont::ArrayHandle<viskores::UInt32> baseHits, marks;
  baseHits.AllocateAndFill(base.NumberOfClusters, 0);
  marks.AllocateAndFill(k, 0);
  invoke(detail::MarkRegionHits{},
         parameters.Topological ? OutputTriangles(mesh, base.Labels) : Simplices(mesh, base.Labels),
         base.Generators,
         regions,
         lower,
         upper,
         baseHits);
  invoke(detail::SpreadBaseHits{}, base.Labels, batch.Labels, baseHits, marks);
  invoke(detail::NonzeroMark{}, marks, flags);
  return flags;
}

namespace detail
{

/// 1 for a merge that the region blame names and the merged-cluster rule does not, which joins
/// the culprits.
struct BlamedOnly : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldInOut culprit, FieldIn region, FieldOut blamed);
  using ExecutionSignature = void(_1, _2, _3);

  VISKORES_EXEC void operator()(viskores::UInt8& culprit,
                                viskores::UInt8 region,
                                viskores::UInt8& blamed) const
  {
    blamed = (region && !culprit) ? 1 : 0;
    culprit = (culprit || region) ? 1 : 0;
  }
};

} // namespace detail

/// The merge phase of Algorithm 8: merges adjacent clusters in rounds while `k > k*` and some
/// merge is admissible. A round commits the greedy maximal set of admissible candidates in
/// priority order, each conflicting with no other, keeps the `k - k*` of highest priority, and
/// checks the merged labelling with every flag. If one fires, it defers the merges whose merged
/// cluster is flagged (and those `RegionFlags` names) and chooses again without them, as
/// deterministic reservations retry the
/// items that fail (Blelloch, Fineman, Gibbons and Shun, PPoPP 2012); if none or all of them are,
/// it merges only the candidate of highest priority, or, if that fails too, rejects it and
/// chooses again.
template <typename RoundObserver>
void MergePhase(const SurfaceMesh& mesh,
                const VertexQuadrics& quadrics,
                const DistanceMesh& inputDistance,
                const MergeParameters& parameters,
                Clustering& clusters,
                MergeDiagnostics& diagnostics,
                RoundObserver&& observer)
{
  using Algorithm = viskores::cont::Algorithm;
  viskores::cont::Invoker invoke;
  const bool geometric = parameters.GeometricFlags();
  // With the singular curves decimated, the local test checks (d) and not (d'), so the full test
  // runs even with no geometric flag in use.
  const bool fullTest = geometric || parameters.Curves != nullptr;
  viskores::cont::Timer timer, roundTimer;
  VerdictCache verdicts;
  // The labellings every full test compares with: the last one committed and the last one
  // rejected, whose (F5) verdicts are known, so that a full test retests only the triangles a
  // change can reach.
  EnvelopeBases envelopeBases;
  const bool incremental = fullTest && parameters.ErrorBound > 0;
  auto setEnvelopeBase =
    [&](EnvelopeBase& base, const Clustering& labelling, const EnvelopeVerdicts& verdictsOf)
  {
    base.Clusters = labelling;
    base.Triangles = parameters.Topological ? OutputTriangles(mesh, labelling.Labels)
                                            : Simplices(mesh, labelling.Labels);
    base.Verdicts = verdictsOf;
  };
  auto commitEnvelopeBase = [&](const Clustering& committed, const EnvelopeVerdicts& verdictsOf)
  { setEnvelopeBase(envelopeBases.Committed, committed, verdictsOf); };
  if (incremental)
  {
    EnvelopeVerdicts initial;
    viskores::cont::ArrayHandle<viskores::UInt8> unused;
    unused.AllocateAndFill(clusters.NumberOfClusters, 0);
    MergeEnvelopeFlags(mesh,
                       inputDistance,
                       clusters,
                       parameters.Topological ? OutputTriangles(mesh, clusters.Labels)
                                              : Simplices(mesh, clusters.Labels),
                       parameters,
                       unused,
                       nullptr,
                       &initial);
    commitEnvelopeBase(clusters, initial);
  }
  const EnvelopeBases* bases = incremental ? &envelopeBases : nullptr;
  while (true)
  {
    roundTimer.Start();
    const viskores::Id deferredBefore = diagnostics.Deferred;
    const Clustering previous = clusters;
    const viskores::Id k = clusters.NumberOfClusters;
    if (k <= parameters.Budget)
    {
      break;
    }
    const OutputComplex complex =
      BuildOutputComplex(parameters.Topological ? OutputTriangles(mesh, clusters.Labels)
                                                : Simplices(mesh, clusters.Labels),
                         k);
    if (parameters.Topological && (k <= 3 || (k == 4 && !complex.HasBoundary)))
    {
      break; // a triangle or a tetrahedron: irreducible
    }

    // Candidates in priority order.
    viskores::cont::ArrayHandle<viskores::Id2> candidateEdges = complex.Edges;
    viskores::cont::ArrayHandle<viskores::IdComponent> candidateCounts = complex.EdgeCounts;
    if (!parameters.Topological)
    {
      // Every pair of adjacent clusters: the label pairs of the cut edges of the input.
      viskores::cont::ArrayHandle<viskores::Id2> pairs;
      viskores::cont::ArrayHandle<viskores::Id3> records;
      invoke(
        detail::CutEdgeRecord{}, mesh.Edges, mesh.EdgeTriangles, clusters.Labels, pairs, records);
      viskores::cont::ArrayHandle<viskores::UInt8> isCut;
      invoke(detail::IsCutRecord{}, records, isCut);
      Algorithm::CopyIf(pairs, isCut, candidateEdges);
      Algorithm::Sort(candidateEdges);
      Algorithm::Unique(candidateEdges);
      candidateCounts.AllocateAndFill(candidateEdges.GetNumberOfValues(), 0);
    }
    else if (mesh.NumberOfComponents > 1)
    {
      // The local test decides the discrete closed-ball property only in a component of K with
      // at least four vertices that is not the boundary of a tetrahedron (Proposition "A merge
      // is an edge contraction"): a component that is a triangle or a tetrahedron is
      // irreducible, and none of its edges is a candidate. The components of K are those of
      // the mesh, which no cluster crosses.
      viskores::cont::ArrayHandle<viskores::Id> sizes, boundaries;
      sizes.AllocateAndFill(mesh.NumberOfComponents, 0);
      boundaries.AllocateAndFill(mesh.NumberOfComponents, 0);
      invoke(detail::CountComponentClusters{},
             clusters.Anchors,
             complex.BoundaryVertex,
             mesh.Components,
             sizes,
             boundaries);
      viskores::cont::ArrayHandle<viskores::UInt8> reducible;
      invoke(detail::InReducibleComponent{},
             candidateEdges,
             clusters.Anchors,
             mesh.Components,
             sizes,
             boundaries,
             reducible);
      viskores::cont::ArrayHandle<viskores::Id2> edges;
      viskores::cont::ArrayHandle<viskores::IdComponent> counts;
      Algorithm::CopyIf(candidateEdges, reducible, edges);
      Algorithm::CopyIf(candidateCounts, reducible, counts);
      candidateEdges = edges;
      candidateCounts = counts;
    }
    // Two candidates conflict if they share a cluster or an edge of K joins them. Without the
    // topological test, K is replaced by the adjacency of the clusters. The degrees of the
    // priority are taken in the same graph.
    viskores::cont::ArrayHandle<viskores::Id> conflictOffsets = complex.NeighborOffsets;
    viskores::cont::ArrayHandle<viskores::Id> conflictNeighbors = complex.Neighbors;
    if (!parameters.Topological)
    {
      viskores::cont::ArrayHandle<viskores::Id2> adjacency;
      invoke(
        BothDirections{}, candidateEdges, viskores::cont::make_ArrayHandleGroupVec<2>(adjacency));
      BuildAdjacency(adjacency, k, conflictOffsets, conflictNeighbors);
    }
    // No merge involves a frozen cluster but a sheet merge, and none joins two copies of one
    // frozen vertex.
    const viskores::cont::ArrayHandle<viskores::UInt8> frozenLabels =
      FrozenLabels(parameters.FrozenVertices, clusters);
    // The frozen clusters a cluster of a sheet may join.
    viskores::cont::ArrayHandle<viskores::UInt8> frozenBoundary;
    if (parameters.FaceMerges && frozenLabels.GetNumberOfValues() > 0)
    {
      frozenBoundary.AllocateAndFill(k, 0);
      invoke(detail::MarkFrozenBoundary{},
             parameters.FrozenVertices,
             mesh.IsBoundaryVertex,
             clusters.Labels,
             frozenBoundary);
    }
    viskores::cont::ArrayHandle<viskores::Id> contactOffsets;
    viskores::cont::ArrayHandle<viskores::Id3> contacts;
    if (parameters.FrozenVertices.GetNumberOfValues() > 0)
    {
      contacts =
        FrozenContacts(mesh, clusters.Labels, parameters.FrozenVertices, parameters.Canonical);
      viskores::cont::ArrayHandle<viskores::Id> contactClusters;
      invoke(FirstComponent{}, contacts, contactClusters);
      Algorithm::LowerBounds(
        contactClusters, viskores::cont::ArrayHandleIndex(k + 1), contactOffsets);
    }
    viskores::cont::ArrayHandle<viskores::UInt8> isCandidate;
    viskores::cont::ArrayHandle<MergePriority> records, candidates;
    invoke(detail::MergeCandidate(parameters.Topological,
                                  frozenLabels.GetNumberOfValues() > 0,
                                  parameters.FaceMerges,
                                  parameters.DegreeFloor,
                                  parameters.Order),
           candidateEdges,
           candidateCounts,
           complex.BoundaryVertex,
           complex.NeighborOffsets,
           complex.Neighbors,
           conflictOffsets,
           complex.Faces,
           frozenLabels,
           frozenBoundary,
           contactOffsets,
           contacts,
           clusters.Anchors,
           mesh.Points,
           clusters.Q,
           clusters.B,
           clusters.S,
           isCandidate,
           records);
    Algorithm::CopyIf(records, isCandidate, candidates);
    Algorithm::Sort(candidates);
    const viskores::Id numCandidates = candidates.GetNumberOfValues();
    viskores::cont::ArrayHandle<viskores::Id2> pairs, ends;
    invoke(detail::CandidatePair{},
           candidates,
           pairs,
           viskores::cont::make_ArrayHandleGroupVec<2>(ends));
    viskores::cont::ArrayHandle<viskores::Id> touchingOffsets, touching;
    BuildAdjacency(ends, k, touchingOffsets, touching);

    // The local test reads the current labelling and K.
    MergeContextObject context;
    context.Mesh = &mesh;
    context.Clusters = &clusters;
    context.Complex = &complex;
    context.VertexMatrices = quadrics.M;
    context.Input = inputDistance;
    context.MaxRadius = parameters.MaxClusterRadius;
    context.Epsilon = parameters.ErrorBound;
    context.Embedding = parameters.Embedding && parameters.Topological;
    context.Glue = OutputGlue(parameters, clusters);
    // (F7) reads the whole labelling: the components of the glued output and the crossings of
    // the probes.
    ThickeningRound thickeningRound;
    if (parameters.CertifiedThickening)
    {
      thickeningRound = PrepareThickeningRound(*parameters.CertifiedThickening,
                                               clusters.Generators,
                                               GlueTriangles(complex.Triangles, context.Glue),
                                               k);
      context.CertifiedThickening = parameters.CertifiedThickening;
      context.RoundThickening = &thickeningRound;
    }
    context.Sides = parameters.Sides;
    context.GeometricOnly = !parameters.Topological;
    if (parameters.FaceMerges && frozenLabels.GetNumberOfValues() > 0)
    {
      context.FrozenLabels = frozenLabels;
      viskores::worklet::Keys<viskores::Id> groups;
      groups.BuildArrays(context.Glue, viskores::worklet::KeysSortType::Stable);
      context.GroupKeys = groups.GetUniqueKeys();
      context.GroupOffsets = groups.GetOffsets();
      context.GroupLabels = groups.GetSortedValuesMap();
    }
    if (geometric)
    {
      invoke(detail::ExactlyDegenerate{},
             complex.Triangles,
             clusters.Generators,
             context.OutputDegenerate);
      // The tie keys of the single-triangle candidates: the index in K, or for the simplices of
      // the geometric-only mode, sorted and distinct, their labels.
      context.Output = context.GeometricOnly
        ? BuildDistanceMesh(clusters.Generators, complex.Triangles, complex.Triangles)
        : BuildDistanceMesh(clusters.Generators, complex.Triangles);
      // The three largest absolute coordinates, by value and then label.
      viskores::cont::ArrayHandle<detail::TopThree> tops;
      invoke(detail::GeneratorTop{}, clusters.Generators, tops);
      const detail::TopThree top = Algorithm::Reduce(tops, detail::NoTop(), detail::MergeTops{});
      for (viskores::IdComponent slot = 0; slot < 3; ++slot)
      {
        context.Scale.TopAbs[slot] = top.first[slot];
        context.Scale.TopLabel[slot] = top.second[slot];
      }
    }

    viskores::cont::ArrayHandle<viskores::UInt8> memo;
    memo.AllocateAndFill(numCandidates, geometric ? detail::Unknown : detail::Admissible);
    // A verdict of (F7) depends on the whole labelling, so none is reused.
    if (geometric && !parameters.CertifiedThickening)
    {
      diagnostics.Reused += verdicts.Reuse(context, clusters, pairs, memo);
    }
    if (geometric && parameters.FaceMerges && frozenLabels.GetNumberOfValues() > 0)
    {
      invoke(detail::RetestSheetMerge{}, pairs, frozenLabels, memo);
    }
    viskores::cont::ArrayHandle<viskores::Id2> committedPairs;
    // The merges deferred as culprits of a batch in this round, never tested alone.
    viskores::cont::ArrayHandle<viskores::UInt8> deferred;
    deferred.AllocateAndFill(numCandidates, 0);
    while (true)
    {
      // The greedy maximal set in priority order, built in passes.
      viskores::cont::ArrayHandle<viskores::UInt8> states;
      states.AllocateAndFill(numCandidates, detail::Alive);
      viskores::cont::ArrayHandle<viskores::UInt8> ready, needed, alive;
      while (true)
      {
        invoke(IsState(detail::Alive), states, alive);
        if (Algorithm::Reduce(alive, viskores::Id(0)) == 0)
        {
          break;
        }
        invoke(detail::ReadyCandidates{},
               pairs,
               states,
               states,
               conflictOffsets,
               conflictNeighbors,
               touchingOffsets,
               touching,
               ready);
        invoke(detail::NeedsEvaluation{}, ready, memo, needed);
        viskores::cont::ArrayHandle<viskores::Id> toEvaluate;
        Algorithm::CopyIf(viskores::cont::ArrayHandleIndex(numCandidates), needed, toEvaluate);
        if (toEvaluate.GetNumberOfValues() > 0)
        {
          timer.Start();
          viskores::cont::ArrayHandle<viskores::Id2> evaluated;
          Algorithm::Copy(viskores::cont::make_ArrayHandlePermutation(toEvaluate, pairs),
                          evaluated);
          viskores::cont::ArrayHandle<viskores::UInt8> admissible;
          viskores::cont::ArrayHandle<Vec3> mergedPositions, lower, upper;
          EvaluateMerges(evaluated, context, admissible, mergedPositions, lower, upper);
          if (parameters.Curves && context.FrozenLabels.GetNumberOfValues() > 0)
          {
            SheetMergeGlueTest(mesh, parameters, clusters, context, evaluated, admissible);
          }
          invoke(detail::ScatterAdmissible{}, toEvaluate, admissible, memo);
          verdicts.Remember(
            context, clusters, evaluated, admissible, mergedPositions, lower, upper);
          diagnostics.Evaluated += toEvaluate.GetNumberOfValues();
          timer.Stop();
          diagnostics.SecondsLocalTests += timer.GetElapsedTime();
        }
        invoke(detail::DecideReady{}, ready, memo, states);
        viskores::cont::ArrayHandle<viskores::UInt8> decided;
        Algorithm::Copy(states, decided);
        invoke(detail::KillConflicting{},
               pairs,
               states,
               decided,
               conflictOffsets,
               conflictNeighbors,
               touchingOffsets,
               touching);
      }
      viskores::cont::ArrayHandle<viskores::UInt8> committed;
      invoke(IsState(detail::Committed), states, committed);
      viskores::cont::ArrayHandle<viskores::Id> committedRanks;
      Algorithm::CopyIf(viskores::cont::ArrayHandleIndex(numCandidates), committed, committedRanks);
      const viskores::Id numCommitted = viskores::Min(
        committedRanks.GetNumberOfValues(), parameters.Budget > 0 ? k - parameters.Budget : k);
      if (numCommitted == 0)
      {
        committedPairs.ReleaseResources();
        // Before the phase stops, the merges only deferred from a batch are tried alone, in
        // priority order (Theorem "budget" (v)).
        viskores::cont::ArrayHandle<viskores::Id> deferredRanks;
        Algorithm::CopyIf(viskores::cont::ArrayHandleIndex(numCandidates), deferred, deferredRanks);
        for (viskores::Id t = 0; t < deferredRanks.GetNumberOfValues(); ++t)
        {
          const viskores::Id rank = viskores::cont::ArrayGetValue(t, deferredRanks);
          viskores::cont::ArrayHandle<viskores::Id2> single;
          Algorithm::Copy(viskores::cont::make_ArrayHandleView(pairs, rank, 1), single);
          Clustering one = ApplyMerges(mesh, quadrics, parameters, clusters, single);
          const viskores::cont::ArrayHandle<viskores::UInt8> oneChanged =
            MergedClusters(clusters, one, single);
          EnvelopeVerdicts oneVerdicts;
          if (!FlagsFire(
                mesh, quadrics, inputDistance, one, parameters, &oneChanged, bases, &oneVerdicts))
          {
            clusters = one;
            committedPairs = single;
            if (incremental)
            {
              commitEnvelopeBase(clusters, oneVerdicts);
            }
            break;
          }
        }
        break;
      }
      Algorithm::Copy(
        viskores::cont::make_ArrayHandlePermutation(
          viskores::cont::make_ArrayHandleView(committedRanks, 0, numCommitted), pairs),
        committedPairs);
      timer.Start();
      Clustering merged = ApplyMerges(mesh, quadrics, parameters, clusters, committedPairs);
      const viskores::cont::ArrayHandle<viskores::UInt8> mergedChanged =
        MergedClusters(clusters, merged, committedPairs);
      EnvelopeVerdicts mergedVerdicts;
      const bool fires = fullTest &&
        FlagsFire(mesh,
                  quadrics,
                  inputDistance,
                  merged,
                  parameters,
                  &mergedChanged,
                  bases,
                  &mergedVerdicts);
      timer.Stop();
      diagnostics.SecondsFullTests += timer.GetElapsedTime();
      timer.Start();
      if (!fires)
      {
        clusters = merged;
        if (incremental)
        {
          commitEnvelopeBase(clusters, mergedVerdicts);
        }
        break;
      }
      if (incremental && mergedVerdicts.FailedOutput.GetNumberOfValues() > 0)
      {
        setEnvelopeBase(envelopeBases.Rejected, merged, mergedVerdicts);
      }
      // The set fired a flag. A cluster that no merge made keeps its members and generator, so
      // the flags fire at a merged cluster, unless (F5) from the input side or a probe of (F7)
      // flags an unmerged one: defer the merges whose merged cluster is flagged, and with the
      // region blame those whose changed triangles meet the region such an item reads, and
      // choose again without them.
      if (parameters.DeferFlagged && numCommitted > 1)
      {
        const bool region = parameters.Blame == MergeBlame::Region;
        viskores::cont::ArrayHandle<viskores::UInt8> flags, culprit, blamed;
        FailingItems items;
        if (FlaggedClusters(mesh,
                            quadrics,
                            inputDistance,
                            merged,
                            parameters,
                            flags,
                            false,
                            bases,
                            &mergedVerdicts,
                            region ? &items : nullptr))
        {
          invoke(detail::MergedFlagged{},
                 committedPairs,
                 clusters.Anchors,
                 merged.Labels,
                 flags,
                 culprit);
          if (region)
          {
            viskores::cont::ArrayHandle<viskores::Id> mergedLabels;
            invoke(
              detail::MergedLabel{}, committedPairs, clusters.Anchors, merged.Labels, mergedLabels);
            viskores::cont::ArrayHandle<viskores::UInt8> changed, regionCulprit;
            changed.AllocateAndFill(merged.NumberOfClusters, 0);
            invoke(MarkIndex{}, mergedLabels, changed);
            invoke(detail::MergedFlagged{},
                   committedPairs,
                   clusters.Anchors,
                   merged.Labels,
                   RegionFlags(mesh, parameters, clusters, merged, changed, items),
                   regionCulprit);
            invoke(detail::BlamedOnly{}, culprit, regionCulprit, blamed);
          }
          viskores::cont::ArrayHandle<viskores::Id> culprits;
          Algorithm::CopyIf(viskores::cont::make_ArrayHandleView(committedRanks, 0, numCommitted),
                            culprit,
                            culprits);
          const viskores::Id numCulprits = culprits.GetNumberOfValues();
          if (numCulprits > 0 && numCulprits < numCommitted)
          {
            invoke(MarkIndex(detail::Inadmissible), culprits, memo);
            invoke(MarkIndex{}, culprits, deferred);
            diagnostics.Deferred += numCulprits;
            if (region)
            {
              diagnostics.BlameDeferred += Algorithm::Reduce(blamed, viskores::Id(0));
            }
            timer.Stop();
            diagnostics.SecondsFullTests += timer.GetElapsedTime();
            continue;
          }
        }
        timer.Stop();
        diagnostics.SecondsFullTests += timer.GetElapsedTime();
        timer.Start();
      }
      // Merge only the candidate of highest priority, if that passes.
      const viskores::Id first = viskores::cont::ArrayGetValue(0, committedRanks);
      viskores::cont::ArrayHandle<viskores::Id2> single;
      Algorithm::Copy(viskores::cont::make_ArrayHandleView(pairs, first, 1), single);
      Clustering one = ApplyMerges(mesh, quadrics, parameters, clusters, single);
      const viskores::cont::ArrayHandle<viskores::UInt8> oneChanged =
        MergedClusters(clusters, one, single);
      EnvelopeVerdicts oneVerdicts;
      const bool oneFires =
        FlagsFire(mesh, quadrics, inputDistance, one, parameters, &oneChanged, bases, &oneVerdicts);
      timer.Stop();
      diagnostics.SecondsFullTests += timer.GetElapsedTime();
      if (!oneFires)
      {
        diagnostics.Fallbacks += (numCommitted > 1) ? 1 : 0;
        clusters = one;
        committedPairs = single;
        if (incremental)
        {
          commitEnvelopeBase(clusters, oneVerdicts);
        }
        break;
      }
      invoke(
        MarkIndex(detail::Inadmissible), viskores::cont::make_ArrayHandleConstant(first, 1), memo);
    }
    if (committedPairs.GetNumberOfValues() == 0)
    {
      break;
    }
    verdicts.Forget(mesh, complex, previous, clusters, committedPairs);
    ++diagnostics.Rounds;
    observer(clusters, committedPairs);
    roundTimer.Stop();
    diagnostics.RoundMerges.push_back(committedPairs.GetNumberOfValues());
    diagnostics.RoundDeferred.push_back(diagnostics.Deferred - deferredBefore);
    diagnostics.RoundSeconds.push_back(roundTimer.GetElapsedTime());
  }
}

}
}
} // namespace viskores::worklet::quadric_clustering

VISKORES_QUADRIC_CLUSTERING_STRICT_FP_END

#endif // viskores_worklet_quadric_clustering_MergePhase_h
