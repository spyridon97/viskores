//============================================================================
//  The contents of this file are covered by the Viskores license. See
//  LICENSE.txt for details.
//
//  By contributing to this file, all contributors agree to the Developer
//  Certificate of Origin Version 1.1 (DCO 1.1) as stated in DCO.txt.
//============================================================================
#ifndef viskores_filter_geometry_refinement_worklet_QuadricClusteringDecimation_h
#define viskores_filter_geometry_refinement_worklet_QuadricClusteringDecimation_h

#include <viskores/filter/geometry_refinement/worklet/quadric_clustering/Assignment.h>
#include <viskores/filter/geometry_refinement/worklet/quadric_clustering/Clustering.h>
#include <viskores/filter/geometry_refinement/worklet/quadric_clustering/CurvatureSeeds.h>
#include <viskores/filter/geometry_refinement/worklet/quadric_clustering/EmbeddingFlags.h>
#include <viskores/filter/geometry_refinement/worklet/quadric_clustering/EnvelopeFlags.h>
#include <viskores/filter/geometry_refinement/worklet/quadric_clustering/Initialization.h>
#include <viskores/filter/geometry_refinement/worklet/quadric_clustering/MergePhase.h>
#include <viskores/filter/geometry_refinement/worklet/quadric_clustering/NonManifoldCut.h>
#include <viskores/filter/geometry_refinement/worklet/quadric_clustering/QuadricClusteringCommonWorklets.h>
#include <viskores/filter/geometry_refinement/worklet/quadric_clustering/SegmentMerges.h>
#include <viskores/filter/geometry_refinement/worklet/quadric_clustering/SurfaceMesh.h>
#include <viskores/filter/geometry_refinement/worklet/quadric_clustering/ThickeningFlags.h>
#include <viskores/filter/geometry_refinement/worklet/quadric_clustering/TopologyFlags.h>
#include <viskores/filter/geometry_refinement/worklet/quadric_clustering/VertexQuadrics.h>

#include <viskores/BinaryPredicates.h>
#include <viskores/CellShape.h>
#include <viskores/cont/Algorithm.h>
#include <viskores/cont/ArrayGetValues.h>
#include <viskores/cont/ArrayHandle.h>
#include <viskores/cont/ArrayHandleCast.h>
#include <viskores/cont/ArrayHandleConstant.h>
#include <viskores/cont/ArrayHandleCounting.h>
#include <viskores/cont/ArrayHandleGroupVec.h>
#include <viskores/cont/ArrayHandleIndex.h>
#include <viskores/cont/ArrayHandlePermutation.h>
#include <viskores/cont/ArrayHandleView.h>
#include <viskores/cont/ErrorBadValue.h>
#include <viskores/cont/ErrorExecution.h>
#include <viskores/cont/Timer.h>

#include <cmath>
#include <functional>
#include <limits>
#include <string>

namespace viskores
{
namespace worklet
{

/// Worklets of the initial components of `QuadricClusteringDecimation`.
namespace component_detail
{

/// The key of a vertex for the initial components: frozen vertices apart.
struct ComponentKey : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn frozen, FieldOut key);
  using ExecutionSignature = _2(_1, WorkIndex);

  VISKORES_EXEC viskores::Id operator()(viskores::UInt8 frozen, viskores::Id v) const
  {
    return frozen ? -1 - v : 0;
  }
};

} // namespace component_detail

/// Worklets of the extraction of the output of `QuadricClusteringDecimation`.
namespace output_detail
{

/// The sorted labels of a triangle, with the triangle.
struct SortedLabelRecord : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn triangle, WholeArrayIn labels, FieldOut record);
  using ExecutionSignature = _3(_1, _2, WorkIndex);

  template <typename IdPortal>
  VISKORES_EXEC viskores::Vec<viskores::Id, 4> operator()(const viskores::Id3& t,
                                                          const IdPortal& labels,
                                                          viskores::Id index) const
  {
    const viskores::Id3 s =
      quadric_clustering::SortThree(labels.Get(t[0]), labels.Get(t[1]), labels.Get(t[2]));
    return viskores::Vec<viskores::Id, 4>(s[0], s[1], s[2], index);
  }
};

/// The edges of a triangle simplex, and the labels of a simplex that is not a point; -1 where
/// there is none.
struct SimplexFaces : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn record, FieldOut edges, FieldOut labels);
  using ExecutionSignature = void(_1, _2, _3);

  template <typename EdgeVec, typename LabelVec>
  VISKORES_EXEC void operator()(const viskores::Vec<viskores::Id, 4>& r,
                                EdgeVec& edges,
                                LabelVec& labels) const
  {
    const bool triangle = r[0] != r[1] && r[1] != r[2];
    edges[0] = triangle ? viskores::Id2(r[0], r[1]) : viskores::Id2(-1);
    edges[1] = triangle ? viskores::Id2(r[1], r[2]) : viskores::Id2(-1);
    edges[2] = triangle ? viskores::Id2(r[0], r[2]) : viskores::Id2(-1);
    const bool point = r[0] == r[2];
    for (viskores::IdComponent i = 0; i < 3; ++i)
    {
      labels[i] = point ? viskores::Id(-1) : r[i];
    }
  }
};

/// Whether a simplex is not contained in another: a segment not an edge of a triangle, a point
/// not a vertex of a segment or triangle.
struct NotContained : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn record,
                                WholeArrayIn triangleEdges,
                                WholeArrayIn covered,
                                FieldOut kept);
  using ExecutionSignature = _4(_1, _2, _3);

  template <typename EdgePortal, typename LabelPortal>
  VISKORES_EXEC viskores::UInt8 operator()(const viskores::Vec<viskores::Id, 4>& r,
                                           const EdgePortal& triangleEdges,
                                           const LabelPortal& covered) const
  {
    if (r[0] != r[1] && r[1] != r[2])
    {
      return 1;
    }
    if (r[0] == r[2])
    {
      return quadric_clustering::ContainsSorted(covered, r[0]) ? 0 : 1;
    }
    return quadric_clustering::ContainsSorted(triangleEdges, viskores::Id2(r[0], r[2])) ? 0 : 1;
  }
};

struct RecordTriangle : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn record, FieldOut triangle);
  using ExecutionSignature = _2(_1);

  VISKORES_EXEC viskores::Id operator()(const viskores::Vec<viskores::Id, 4>& r) const
  {
    return r[3];
  }
};

/// The labels of triangle `t`, without repetition, in the order of its corners.
template <typename TrianglePortal, typename IdPortal>
VISKORES_EXEC viskores::IdComponent DistinctLabels(viskores::Id t,
                                                   const TrianglePortal& triangles,
                                                   const IdPortal& labels,
                                                   viskores::Id* ids)
{
  const viskores::Id3 tri = triangles.Get(t);
  viskores::IdComponent count = 0;
  for (viskores::IdComponent c = 0; c < 3; ++c)
  {
    const viskores::Id l = labels.Get(tri[c]);
    bool seen = false;
    for (viskores::IdComponent i = 0; i < count; ++i)
    {
      seen = seen || ids[i] == l;
    }
    if (!seen)
    {
      ids[count++] = l;
    }
  }
  return count;
}

struct DistinctLabelCount : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn triangle,
                                WholeArrayIn triangles,
                                WholeArrayIn labels,
                                FieldOut count);
  using ExecutionSignature = _4(_1, _2, _3);

  template <typename TrianglePortal, typename IdPortal>
  VISKORES_EXEC viskores::IdComponent operator()(viskores::Id t,
                                                 const TrianglePortal& triangles,
                                                 const IdPortal& labels) const
  {
    viskores::Id ids[3];
    return DistinctLabels(t, triangles, labels, ids);
  }
};

/// The simplex of a triangle: its distinct labels at its offset, and its shape.
struct WriteSimplex : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn triangle,
                                FieldIn offset,
                                WholeArrayIn triangles,
                                WholeArrayIn labels,
                                WholeArrayOut connectivity,
                                FieldOut shape);
  using ExecutionSignature = void(_1, _2, _3, _4, _5, _6);

  template <typename TrianglePortal, typename IdPortal, typename OutPortal>
  VISKORES_EXEC void operator()(viskores::Id t,
                                viskores::Id offset,
                                const TrianglePortal& triangles,
                                const IdPortal& labels,
                                OutPortal& connectivity,
                                viskores::UInt8& shape) const
  {
    viskores::Id ids[3];
    const viskores::IdComponent count = DistinctLabels(t, triangles, labels, ids);
    for (viskores::IdComponent i = 0; i < count; ++i)
    {
      connectivity.Set(offset + i, ids[i]);
    }
    shape = (count == 3) ? viskores::CELL_SHAPE_TRIANGLE
                         : ((count == 2) ? viskores::CELL_SHAPE_LINE : viskores::CELL_SHAPE_VERTEX);
  }
};

/// The output point of an input point, through one of its copies; the copies of a point are
/// glued, so every write is the same.
struct MapInputPoint : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn input,
                                FieldIn label,
                                WholeArrayIn pointOfLabel,
                                WholeArrayOut map);
  using ExecutionSignature = void(_1, _2, _3, _4);

  template <typename IdPortal, typename OutPortal>
  VISKORES_EXEC void operator()(viskores::Id input,
                                viskores::Id label,
                                const IdPortal& pointOfLabel,
                                OutPortal& map) const
  {
    map.Set(input, pointOfLabel.Get(label));
  }
};

} // namespace output_detail

/// @brief Decimates a triangle mesh by guaranteed variational quadric clustering.
///
/// The algorithm is that of Section 11 of "Guaranteed Parallel Mesh Decimation by Variational
/// Quadric Clustering": vertices are partitioned into connected clusters, each of which becomes
/// an output vertex placed at the minimiser of its quadric cost. A refinement loop alternates
/// a guarded assignment, the placement of the generators and a scan that flags clusters that
/// break a requirement; a flagged cluster receives a new generator at its worst member. The loop
/// stops when nothing is flagged.
///
/// Every quantity that feeds a decision is computed in double precision with a fixed order of
/// operations, so the output is a function of the input, as numbered, and of the parameters,
/// independent of the device and of the number of threads.
class QuadricClusteringDecimation
{
public:
  using Real = quadric_clustering::Real;
  using Vec3 = quadric_clustering::Vec3;

  /// The initial clusters, before the refinement loop.
  enum class Initialization
  {
    /// One cluster per connected component.
    Single,
    /// Seeds evenly spaced by vertex count along the Morton order: `k*` in budget mode, and in
    /// error mode as many as the curvature initialisation would place.
    Morton,
    /// Seeds spread along the Morton order by a density that follows the curvature against the
    /// error bound: `k*` in budget mode, and in error mode as many as the density asks for,
    /// within the topological floor and a quarter of the vertices.
    Curvature
  };

  /// The insertion of step (d) for a flagged cluster.
  enum class Insertion
  {
    /// A new cluster at the worst member.
    Worst,
    /// The cluster split in two by 4 Lloyd iterations from its generator and its worst member.
    Split2
  };

  struct Parameters
  {
    /// The regularisation factor and the feature and boundary terms of the cost.
    quadric_clustering::QuadricParameters Quadric;
    /// The target vertex count `k*` of budget mode, or 0.
    viskores::Id TargetNumberOfPoints = 0;
    /// The bound `epsilon_H` on the Hausdorff distance between input and output of error mode,
    /// enforced by the envelope flag (F5), or by (F7) with the fast thickening, or 0.
    Real ErrorBound = 0;
    /// A cap on the cluster radius, flag (F0).
    Real MaxClusterRadius = std::numeric_limits<Real>::infinity();
    /// The topological flags (F2) and (F3). Turning them off voids every topological guarantee;
    /// it exists to measure what they do.
    bool Topology = true;
    /// (D2) by count of the clusters per closed component (Section 11) instead of by sort of
    /// the label triples. The guarantees are the same; rounds with other violations flag fewer
    /// clusters.
    bool D2ByCount = false;
    /// The embedding flag (F4): the realised output has no self-intersections.
    bool Embedding = true;
    /// The halves of the envelope flag (F5) with an error bound. The geometric-only mode needs
    /// both.
    quadric_clustering::EnvelopeSides ErrorSides = quadric_clustering::EnvelopeSides::Both;
    /// Isotopy, an option: the flag (F7) of Theorem I, the output isotopic to the input through a
    /// certified thickening of it (proofs/lib/thick.py). Needs a closed surface,
    /// the topological flags, (F4) and an error bound. With `Creases`, the thickening is that of
    /// the input before the cut, and (F7) reads the glued output (Corollary "creasesiso").
    ///
    /// - `Standard`: a wide thickening, offsets from three times the error bound, not capped,
    ///   and (F5) as `ErrorSides` says, so the output is within the error bound by (F5).
    /// - `Fast`: offsets from just below the error bound and capped by it, and (F5) off: (F7)
    ///   alone certifies the error bound (Proposition "thickhausdorff"). Faster, but it costs
    ///   output size on boxy parts.
    ///
    /// If no certified thickening is found, the run takes the default path, with (F5) as
    /// `ErrorSides` says, and `Diagnostics::ThickeningCertified` is false.
    ///
    /// Measured cost: free on smooth and boxy inputs (identical outputs on 16 of 18 synthetic
    /// inputs and the roller), about twice the output size on CAD parts with thin features
    /// (gears 515-524 vertices against 255-260, Link31 162 against 120): the certificate needs
    /// the output inside a thickening, which thin features make thin. A better thickening is an
    /// open problem.
    enum class ThickeningMode
    {
      Off,
      Standard,
      Fast
    };
    ThickeningMode Thickening = ThickeningMode::Off;
    /// Experimental: the initial offsets of the thickening as a multiple of the error bound,
    /// in place of those of the mode, or 0 for those.
    Real ThickeningOffset = 0;
    /// With the thickening, halve only the offset on the side of a failure while building it,
    /// not both.
    bool ThickeningAsymmetric = true;
    /// With the thickening, the directions of its offsets (Proposition "thicklocal"): by
    /// default the area-weighted normal where it makes a positive angle margin with every
    /// triangle at the vertex, and the max-min direction elsewhere.
    quadric_clustering::ThickeningDirections ThickeningDirections =
      quadric_clustering::ThickeningDirections::Hybrid;
    /// With the thickening, the fibre flag (F8) of Theorem "fibres": every output triangle
    /// meets the fibres of the thickening transversally, in its orientation (K3). Closed inputs,
    /// also with `Creases` (Corollary "creasesfibres").
    /// Off with the thickening when none is certified.
    bool Fibres = false;
    /// The cap `T` on the propose/commit sweeps of each assignment step, or 0 for none.
    viskores::Id MaxSweeps = 0;
    /// The initial clusters. Seeds are followed by a Lloyd phase. An input cut into surfaces and
    /// the geometric-only mode always start from one cluster per component (Theorem "Inputs
    /// that are not surfaces", (a)).
    Initialization InitialClusters = Initialization::Curvature;
    /// The cap on the alternations of assignment and placement after seeding.
    viskores::Id MaxLloydIterations = 100;
    /// The insertion of step (d).
    Insertion InsertionRule = Insertion::Worst;
    /// The merge phase after the loop, down to `k*` in budget mode and as far as the flags allow
    /// in error mode.
    bool Merge = true;
    /// The geometric-only mode of Proposition "Geometric-only mode": any complex of triangles,
    /// no topological or embedding flag, the graph form of the guard, and (F5') for the error
    /// bound. The output is the set of simplices `g(t)`.
    bool GeometricOnly = false;
    /// The floor `D_0` of the degree key of the merge priority, or a negative value to order
    /// merges by the increase of the objective alone.
    viskores::Id MergeDegreeFloor = 12;
    /// The key of the merge priority after the degree: the increase of the objective, or a hash
    /// of the anchors of the two clusters.
    quadric_clustering::MergeOrder MergeOrder = quadric_clustering::MergeOrder::Cost;
    /// When the merges of a round fail the full test together: defer those whose merged cluster
    /// is flagged and test the others again, or, if false, merge only the one of highest
    /// priority.
    bool DeferFlaggedMerges = true;
    /// Which merges are deferred: those whose merged cluster is flagged, or also those whose
    /// changed output triangles meet the region read by a failing item that flags no merged
    /// cluster (the default).
    quadric_clustering::MergeBlame MergeBlame = quadric_clustering::MergeBlame::Region;
    /// For an input that is not a consistently oriented surface: cut a seam too, so that every
    /// piece is orientable. Otherwise only the singular edges (and the creases) are cut and
    /// nothing is reoriented, so a piece may not be orientable (Corollary "nonorientable"); a
    /// surface that is not orientable, such as a Moebius band, takes the cut with no edge cut
    /// and no frozen copy. The thickening needs a consistently oriented input either way.
    bool Seams = true;
    /// For an input cut into surfaces: decimate its singular curves too, by merging consecutive
    /// segments of them in the merge phase (Proposition "Decimating the singular curves");
    /// otherwise they keep their input resolution (Theorem "Inputs that are not surfaces").
    bool DecimateSingularCurves = true;
    /// Experimental: cut the creases too, the edges in two triangles whose normals are farther
    /// apart than the feature angle, so that every crease curve
    /// comes out as a path of output edges through its ends, decimated as the singular curves
    /// are. Error mode only.
    bool Creases = false;
    /// With `Creases`: the cut set `X`, as pairs of input vertices, in place of the creases
    /// found by the feature angle; empty for those.
    viskores::cont::ArrayHandle<viskores::Id2> CutEdges;
    /// With the curves decimated: the tolerance `eps_c` of the curve test (Fc) of Proposition
    /// "curvetol", a length, or 0 for none. Every output curve then stays within `eps_c` of its
    /// input curve, and every input curve within `eps_c` of its output curve.
    Real CurveTolerance = 0;
    /// Experimental, with the curves decimated: the end segment of an open curve may merge into
    /// the junction at that end.
    bool JunctionMerges = false;
    /// Experimental, with the curves decimated: a cluster of a sheet may merge into an adjacent
    /// piece of a segment or copy of a junction.
    bool FaceMerges = false;
  };

  /// The points of the run at which an observer is called.
  enum class Stage
  {
    /// After a sweep of the assignment; only the labels are current.
    Sweep,
    /// After the placement step.
    Placed,
    /// After the scan, with the flags of the round.
    Scanned,
    /// After the insertion and split steps.
    Split,
    /// After a round of the merge phase, with the merged pairs.
    Merged
  };

  struct State
  {
    const quadric_clustering::SurfaceMesh& Mesh;
    const quadric_clustering::VertexQuadrics& Quadrics;
    const quadric_clustering::Clustering& Clusters;
    /// Valid from `Scanned` on.
    const quadric_clustering::ClusterScan* Scan = nullptr;
    const viskores::cont::ArrayHandle<viskores::UInt8>* Flags = nullptr;
    const quadric_clustering::TopologyReport* Topology = nullptr;
    /// In `Sweep`: the labels after the sweep, and the changes of every vertex so far in this
    /// assignment step.
    const viskores::cont::ArrayHandle<viskores::Id>* SweepLabels = nullptr;
    const viskores::cont::ArrayHandle<viskores::Id>* Changes = nullptr;
    /// In `Merged`: the pairs `(i, j)` of the round, in the labels before it.
    const viskores::cont::ArrayHandle<viskores::Id2>* MergedPairs = nullptr;
  };

  using Observer = std::function<void(Stage, const State&)>;

  struct Diagnostics
  {
    /// Rounds of the refinement loop.
    viskores::Id Rounds = 0;
    /// Clusters after the initialisation, and alternations of assignment and placement in it.
    viskores::Id InitialClusters = 0;
    viskores::Id LloydIterations = 0;
    viskores::Id Sweeps = 0;
    /// Clusters flagged by each test over the whole loop, each counted once per round by the
    /// first test that flagged it, in the order (F0), (Fb), (F2)-(F3), (F4), (F5), (F7), (F8).
    viskores::Id FlaggedRadius = 0;
    viskores::Id FlaggedBudget = 0;
    viskores::Id FlaggedTopology = 0;
    viskores::Id FlaggedEmbedding = 0;
    viskores::Id FlaggedEnvelope = 0;
    viskores::Id FlaggedThickening = 0;
    viskores::Id FlaggedFibres = 0;
    /// With the thickening: whether a certified one was found, so that (F7) ran and the output
    /// is isotopic to the input, and the rounds of halving of its offsets.
    bool ThickeningCertified = false;
    viskores::Id ThickeningHalvings = 0;
    /// Clusters found adjacent to two copies of a frozen vertex of a cut input, over the loop;
    /// they are flagged with (F2) and (F3).
    viskores::Id GlueViolations = 0;
    /// With creases: the crease edges; with the curves decimated, the arcs and the junctions.
    viskores::Id CreaseEdges = 0;
    viskores::Id Arcs = 0;
    viskores::Id Junctions = 0;
    /// Output triangles given to (F4), and pieces examined by the envelope tests, over the run:
    /// machine-independent measures of the two most expensive steps.
    viskores::Id EmbeddingTriangles = 0;
    viskores::Id EnvelopePieces = 0;
    /// Wall-clock seconds spent in each step, the device synchronised.
    struct
    {
      double Setup = 0;
      double Quadrics = 0;
      double Initialization = 0;
      double Assignment = 0;
      double Placement = 0;
      double Scan = 0;
      double Topology = 0;
      double Embedding = 0;
      double Envelope = 0;
      double Thickening = 0;
      double InsertSplit = 0;
      double Merge = 0;
      double Output = 0;
    } Seconds;
  };

  VISKORES_CONT void SetParameters(const Parameters& parameters) { this->Params = parameters; }
  VISKORES_CONT const Parameters& GetParameters() const { return this->Params; }

  /// Sets the fixed order of the commit step: vertex `u` beats `v` if its key is larger, or equal
  /// with a larger index. By default the key is a hash of the vertex index. Implementations that
  /// are to agree must use the same order. For an input that is cut into surfaces, the keys are
  /// those of the copies, numbered as by `GetCopyOriginals()`.
  VISKORES_CONT void SetPriority(const viskores::cont::ArrayHandle<viskores::UInt64>& priority)
  {
    this->Priority = priority;
  }

  VISKORES_CONT void SetObserver(const Observer& observer) { this->ObserverFunction = observer; }

  /// Decimates the triangles `triangles` on the points `points`. A complex that is not a
  /// consistently oriented surface is cut along its singular edges and, with `Seams`, a seam
  /// into surfaces whose copies of the cut vertices stay fixed, and glued back (Theorem "Inputs
  /// that are not surfaces"), unless the geometric-only mode is on. Throws `ErrorBadValue` for degenerate
  /// triangles, points in no triangle, budget mode on an input that needs the cut, and, with the
  /// embedding flag, an input that is not embedded (Hypothesis "embedded").
  VISKORES_CONT void Run(const viskores::cont::ArrayHandle<Vec3>& points,
                         const viskores::cont::ArrayHandle<viskores::Id3>& triangles)
  {
    namespace qc = quadric_clustering;
    using Algorithm = viskores::cont::Algorithm;
    if (this->Params.TargetNumberOfPoints < 0)
    {
      throw viskores::cont::ErrorBadValue(
        "QuadricClusteringDecimation: the target number of points is negative.");
    }
    if (this->Params.ErrorBound < 0)
    {
      throw viskores::cont::ErrorBadValue(
        "QuadricClusteringDecimation: the error bound is negative.");
    }
    this->Diag = Diagnostics{};
    auto& seconds = this->Diag.Seconds;
    viskores::cont::Timer timer;
    auto timed = [&timer](double& slot, auto&& step)
    {
      timer.Start();
      step();
      timer.Stop();
      slot += timer.GetElapsedTime();
    };
    timer.Start();
    const bool geometric = this->Params.GeometricOnly;
    const viskores::Id budget = this->Params.TargetNumberOfPoints;
    const viskores::Id numInputPoints = points.GetNumberOfValues();

    // The input: a surface, a complex cut into surfaces, or any complex.
    const std::string problem =
      qc::BuildSurfaceMesh(points, triangles, false, this->Mesh, !this->Params.Seams);
    this->Canonical.ReleaseResources();
    this->FrozenCopies.ReleaseResources();
    this->CopyOriginals.ReleaseResources();
    this->HasCurves = false;
    this->Curves = quadric_clustering::SingularCurves{};
    if (this->Params.Embedding && !geometric)
    {
      this->CheckEmbedded(points, triangles);
    }
    if (geometric && this->Params.ErrorBound > 0 &&
        this->Params.ErrorSides != qc::EnvelopeSides::Both)
    {
      throw viskores::cont::ErrorBadValue("QuadricClusteringDecimation: the geometric-only mode "
                                          "checks both sides of the error bound.");
    }
    const bool thickeningRequested = this->Params.Thickening != Parameters::ThickeningMode::Off;
    if (thickeningRequested &&
        (geometric || !this->Params.Topology || !this->Params.Embedding ||
         !(this->Params.ErrorBound > 0) || !problem.empty() || this->Mesh.HasBoundary))
    {
      throw viskores::cont::ErrorBadValue(
        "QuadricClusteringDecimation: the thickening flag needs a closed surface, the "
        "topological and embedding flags and an error bound.");
    }
    if (thickeningRequested && this->Mesh.Unoriented)
    {
      throw viskores::cont::ErrorBadValue(
        "QuadricClusteringDecimation: the thickening flag needs a consistently oriented input; "
        "no result covers a piece that is not orientable.");
    }
    if (this->Params.Fibres && !thickeningRequested)
    {
      throw viskores::cont::ErrorBadValue(
        "QuadricClusteringDecimation: the fibre flag needs the thickening flag.");
    }
    // With (F7), the certified thickening of the input, before a cut along its creases, which
    // keeps its triangles in order (Corollary "creasesiso"); without one, the default path. The
    // fast setting turns (F5) off, which (F7) then replaces.
    qc::Thickening thick;
    qc::EnvelopeSides sides = this->Params.ErrorSides;
    if (thickeningRequested)
    {
      const Real bound = this->Params.ErrorBound;
      const bool asymmetric = this->Params.ThickeningAsymmetric;
      const bool fast = this->Params.Thickening == Parameters::ThickeningMode::Fast;
      Real delta0 = fast ? bound * (1 - std::ldexp(Real(1), -20)) : 3 * bound;
      if (this->Params.ThickeningOffset > 0)
      {
        delta0 = this->Params.ThickeningOffset * bound;
      }
      thick = qc::BuildThickening(
        this->Mesh, delta0, bound, fast, asymmetric, this->Params.ThickeningDirections);
      this->Diag.ThickeningCertified = thick.Valid;
      this->Diag.ThickeningHalvings = thick.Halvings;
      if (thick.Valid && this->Params.Fibres)
      {
        qc::AddFibres(this->Mesh, thick);
      }
      if (thick.Valid && fast)
      {
        sides = qc::EnvelopeSides::None;
      }
    }
    const bool thickening = thick.Valid;
    // The creases, cut as the singular edges are: the given cut set, or the sharp edges of the
    // feature angle.
    viskores::cont::ArrayHandle<viskores::Id2> creases;
    if (this->Params.Creases && !geometric)
    {
      if (this->Params.CutEdges.GetNumberOfValues() > 0)
      {
        viskores::cont::Invoker invoke;
        invoke(qc::SortedEdgeOf{}, this->Params.CutEdges, creases);
        Algorithm::Sort(creases);
        Algorithm::Unique(creases);
      }
      else
      {
        const qc::QuadricParameters& q = this->Params.Quadric;
        const Real cosAngle = qc::FeatureAngleCosine(q.FeatureAngle);
        creases = qc::CreaseEdges(this->Mesh, cosAngle);
      }
      this->Diag.CreaseEdges = creases.GetNumberOfValues();
    }
    // Without seams, a surface that is not consistently oriented is cut too, along nothing but
    // its creases, as the reference cuts it.
    const bool unoriented = !this->Params.Seams && this->Mesh.Unoriented;
    if (!geometric && (!problem.empty() || creases.GetNumberOfValues() > 0 || unoriented))
    {
      if (budget > 0)
      {
        throw viskores::cont::ErrorBadValue(
          "QuadricClusteringDecimation: budget mode needs a surface without cut, and the input " +
          (!problem.empty() ? "is not one: " + problem
             : unoriented   ? std::string("is not consistently oriented")
                            : std::string("has creases")) +
          ".");
      }
      this->CutInput(points, triangles, creases);
      if (this->HasCurves)
      {
        this->Diag.Arcs = this->Curves.NumberOfArcs;
        this->Diag.Junctions = qc::NumberOfJunctions(this->Curves);
      }
    }
    const qc::SurfaceMesh& mesh = this->Mesh;
    const viskores::Id n = mesh.NumberOfPoints;
    qc::QuadricParameters quadric = this->Params.Quadric;
    if (geometric)
    {
      // Without a surface there are no dihedral angles or boundary edges.
      quadric.FeatureWeight = 0;
      quadric.BoundaryWeight = 0;
    }
    timer.Stop();
    seconds.Setup += timer.GetElapsedTime();
    timed(seconds.Quadrics, [&] { this->Quadrics = qc::ComputeVertexQuadrics(mesh, quadric); });
    timer.Start();
    const Real maxRadius = this->Params.MaxClusterRadius;
    viskores::cont::ArrayHandle<viskores::UInt64> priority = this->Priority;
    if (priority.GetNumberOfValues() == 0)
    {
      priority = qc::DefaultPriority(n);
    }
    else if (priority.GetNumberOfValues() != n)
    {
      throw viskores::cont::ErrorBadValue("QuadricClusteringDecimation: the priority needs one key "
                                          "per vertex of the mesh decimated, the copies of a cut "
                                          "input included.");
    }
    // One-rings for the guard of surfaces, sorted neighbours for its graph form.
    const viskores::cont::ArrayHandle<viskores::Id>& ringOffsets =
      geometric ? mesh.NeighborOffsets : mesh.RingOffsets;
    const viskores::cont::ArrayHandle<viskores::Id>& rings =
      geometric ? mesh.Neighbors : mesh.Rings;
    viskores::cont::ArrayHandle<viskores::UInt8> isBoundary = mesh.IsBoundaryVertex;
    if (geometric)
    {
      isBoundary.AllocateAndFill(n, 0);
    }
    const viskores::cont::ArrayHandle<viskores::Id2> ringRanges =
      qc::MakeRingRanges(ringOffsets, n);
    const Real errorBound = this->Params.ErrorBound;
    qc::DistanceMesh inputDistance;
    if (errorBound > 0)
    {
      inputDistance = qc::BuildDistanceMesh(mesh.Points, mesh.Triangles);
    }
    const bool embedding = this->Params.Embedding && !geometric;
    const bool topology = this->Params.Topology && !geometric;
    const qc::EnvelopeMode envelopeMode =
      geometric ? qc::EnvelopeMode::Simplices : qc::EnvelopeMode::Surface;
    qc::Clustering& clusters = this->Clusters;
    clusters = qc::Clustering{};
    viskores::cont::ArrayHandle<viskores::UInt8> frozenLabels;

    auto observe = [&](Stage stage, const State& state)
    {
      if (this->ObserverFunction)
      {
        this->ObserverFunction(stage, state);
      }
    };
    viskores::cont::ArrayHandle<viskores::Id> changes;
    auto assign = [&](bool inLoop)
    {
      if (this->ObserverFunction || !inLoop)
      {
        changes.AllocateAndFill(n, 0);
      }
      auto sweepObserver = [&](const viskores::cont::ArrayHandle<viskores::Id>& labels)
      {
        if (inLoop)
        {
          State state{ mesh, this->Quadrics, clusters };
          state.SweepLabels = &labels;
          state.Changes = &changes;
          observe(Stage::Sweep, state);
        }
      };
      const viskores::Id sweeps =
        qc::Assign(mesh,
                   ringRanges,
                   rings,
                   ringOffsets,
                   geometric,
                   isBoundary,
                   this->Quadrics.M,
                   priority,
                   frozenLabels,
                   this->Params.MaxSweeps,
                   clusters.Labels,
                   clusters.Generators,
                   (this->ObserverFunction || !inLoop) ? &changes : nullptr,
                   sweepObserver);
      this->Diag.Sweeps += sweeps;
    };
    auto place = [&]()
    {
      qc::Place(mesh, this->Quadrics.M, clusters);
      if (this->HasCurves)
      {
        qc::PlaceCurveGenerators(mesh,
                                 this->Quadrics.M,
                                 this->Curves,
                                 this->FrozenCopies,
                                 this->CopyOriginals,
                                 this->Glue(clusters),
                                 clusters);
      }
      frozenLabels = qc::FrozenLabels(this->FrozenCopies, clusters);
      observe(Stage::Placed, State{ mesh, this->Quadrics, clusters });
    };

    timer.Stop();
    seconds.Setup += timer.GetElapsedTime();
    timer.Start();
    // Initialisation: one cluster per component, frozen vertices apart, or seeds and a Lloyd
    // phase. An input cut into surfaces and the geometric-only mode start from the components.
    Initialization initialization = this->Params.InitialClusters;
    if (geometric || this->FrozenCopies.GetNumberOfValues() > 0)
    {
      initialization = Initialization::Single;
    }
    if (initialization != Initialization::Single)
    {
      if (initialization == Initialization::Morton && budget > 0)
      {
        clusters.Labels = qc::SeedLabels(mesh, ringRanges, rings, budget);
      }
      else
      {
        const qc::CurvatureDensity density =
          qc::ComputeCurvatureDensity(mesh, (budget > 0) ? Real(0) : errorBound, maxRadius);
        const viskores::Id k = (budget > 0) ? budget : qc::CurvatureClusterCount(mesh, density);
        clusters.Labels = (initialization == Initialization::Morton)
          ? qc::SeedLabels(mesh, ringRanges, rings, k)
          : qc::CurvatureSeedLabels(mesh, ringRanges, rings, density, k);
      }
      place();
      for (viskores::Id i = 0; i < this->Params.MaxLloydIterations; ++i)
      {
        viskores::cont::ArrayHandle<viskores::Id> before;
        Algorithm::Copy(clusters.Labels, before);
        const viskores::Id k = clusters.NumberOfClusters;
        assign(false);
        place();
        ++this->Diag.LloydIterations;
        const bool noChange = Algorithm::Reduce(changes, viskores::Id(0)) == 0;
        const bool same = clusters.NumberOfClusters == k && ArraysEqual(before, clusters.Labels);
        if (same || noChange)
        {
          break;
        }
      }
    }
    else
    {
      qc::LabelComponents(n, mesh.Edges, this->InitialComponentKeys(n), clusters.Labels);
      place();
    }

    timer.Stop();
    this->Diag.InitialClusters = clusters.NumberOfClusters;
    seconds.Initialization += timer.GetElapsedTime();
    // The refinement loop. By Theorem "Number of rounds" it stops within n + 1 rounds.
    while (true)
    {
      ++this->Diag.Rounds;
      if (this->Diag.Rounds > n + 1)
      {
        throw viskores::cont::ErrorExecution(
          "QuadricClusteringDecimation: the refinement loop did not stop.");
      }
      timed(seconds.Assignment, [&] { assign(true); });
      timed(seconds.Placement, place);
      qc::ClusterScan scan;
      timed(seconds.Scan, [&] { scan = qc::Scan(mesh, this->Quadrics.M, clusters); });
      const viskores::Id k = clusters.NumberOfClusters;
      timer.Start();

      viskores::cont::ArrayHandle<viskores::UInt8> flags;
      flags.AllocateAndFill(k, 0);
      if (maxRadius < std::numeric_limits<Real>::infinity())
      {
        this->Diag.FlaggedRadius += qc::RadiusFlags(scan, maxRadius, flags);
      }
      this->Diag.FlaggedBudget += qc::BudgetFlags(clusters, scan, budget, flags);
      timer.Stop();
      seconds.Scan += timer.GetElapsedTime();
      qc::TopologyReport topologyReport;
      if (topology)
      {
        timed(seconds.Topology,
              [&]
              {
                viskores::cont::ArrayHandle<viskores::UInt8> topologyFlags;
                topologyFlags.AllocateAndFill(k, 0);
                topologyReport = qc::TopologyFlags(
                  mesh, clusters.Labels, clusters.Sizes, topologyFlags, this->Params.D2ByCount);
                if (this->HasCurves)
                {
                  // (d') of Proposition "Decimating the singular curves".
                  viskores::cont::ArrayHandle<viskores::Id> listed = qc::GlueViolations(
                    mesh, clusters.Labels, this->CopyOriginals, this->Curves, this->Glue(clusters));
                  this->Diag.GlueViolations += listed.GetNumberOfValues();
                  qc::FlagClusters(listed, clusters.Sizes, topologyFlags);
                }
                else
                {
                  this->Diag.GlueViolations += qc::GlueFlags(mesh,
                                                             clusters.Labels,
                                                             this->FrozenCopies,
                                                             this->Canonical,
                                                             clusters.Sizes,
                                                             topologyFlags);
                }
                this->Diag.FlaggedTopology += qc::OrFlags(flags, topologyFlags);
              });
      }
      // The embedding test is expensive, so it runs only when no cheaper flag fired.
      if (embedding && qc::CountFlags(flags) == 0)
      {
        timer.Start();
        viskores::cont::ArrayHandle<viskores::UInt8> embeddingFlags;
        embeddingFlags.AllocateAndFill(k, 0);
        const auto outputTriangles = qc::OutputTriangles(mesh, clusters.Labels);
        this->Diag.EmbeddingTriangles += outputTriangles.GetNumberOfValues();
        qc::EmbeddingFlags(qc::GlueTriangles(outputTriangles, this->Glue(clusters)),
                           clusters.Generators,
                           clusters.Sizes,
                           embeddingFlags);
        this->Diag.FlaggedEmbedding += qc::OrFlags(flags, embeddingFlags);
        timer.Stop();
        seconds.Embedding += timer.GetElapsedTime();
      }
      // The envelope test needs the discrete closed-ball property, so it runs only when nothing
      // else is flagged, the topological flags in particular.
      if (errorBound > 0 && qc::CountFlags(flags) == 0)
      {
        timer.Start();
        viskores::cont::ArrayHandle<viskores::UInt8> envelopeFlags;
        envelopeFlags.AllocateAndFill(k, 0);
        const qc::EnvelopeReport envelopeReport =
          qc::EnvelopeFlags(mesh,
                            inputDistance,
                            clusters.Labels,
                            clusters.Generators,
                            geometric ? qc::Simplices(mesh, clusters.Labels)
                                      : qc::OutputTriangles(mesh, clusters.Labels),
                            clusters.Sizes,
                            errorBound,
                            envelopeMode,
                            sides,
                            envelopeFlags);
        this->Diag.EnvelopePieces += envelopeReport.Pieces;
        this->Diag.FlaggedEnvelope += qc::OrFlags(flags, envelopeFlags);
        timer.Stop();
        seconds.Envelope += timer.GetElapsedTime();
      }
      // (F7), last, likewise.
      if (thickening && qc::CountFlags(flags) == 0)
      {
        timer.Start();
        viskores::cont::ArrayHandle<viskores::UInt8> thickeningFlags;
        thickeningFlags.AllocateAndFill(k, 0);
        qc::ThickeningFlags(mesh,
                            thick,
                            clusters.Labels,
                            clusters.Generators,
                            clusters.Sizes,
                            thickeningFlags,
                            this->Glue(clusters));
        // (F8) with (F7), as one test of each triangle.
        viskores::cont::ArrayHandle<viskores::UInt8> fibreFlags;
        if (thick.Fibres)
        {
          fibreFlags.AllocateAndFill(k, 0);
          qc::FibreFlags(
            mesh, thick, clusters.Labels, clusters.Generators, clusters.Sizes, fibreFlags);
        }
        this->Diag.FlaggedThickening += qc::OrFlags(flags, thickeningFlags);
        if (thick.Fibres)
        {
          this->Diag.FlaggedFibres += qc::OrFlags(flags, fibreFlags);
        }
        timer.Stop();
        seconds.Thickening += timer.GetElapsedTime();
      }

      State state{ mesh, this->Quadrics, clusters };
      state.Scan = &scan;
      state.Flags = &flags;
      state.Topology = &topologyReport;
      observe(Stage::Scanned, state);
      if (qc::CountFlags(flags) == 0)
      {
        break;
      }
      timed(seconds.InsertSplit,
            [&]
            {
              qc::InsertAndSplit(mesh,
                                 this->Quadrics.M,
                                 flags,
                                 scan,
                                 this->Params.InsertionRule == Insertion::Split2,
                                 clusters);
              frozenLabels = qc::FrozenLabels(this->FrozenCopies, clusters);
            });
      observe(Stage::Split, State{ mesh, this->Quadrics, clusters });
    }
    this->ClustersAfterLoop = clusters.NumberOfClusters;

    // The merge phase.
    this->MergeDiag = qc::MergeDiagnostics{};
    timer.Start();
    if (this->Params.Merge && clusters.NumberOfClusters > budget)
    {
      qc::MergeParameters merge;
      merge.Budget = budget;
      merge.MaxClusterRadius = maxRadius;
      merge.CertifiedThickening = thickening ? &thick : nullptr;
      merge.Embedding = embedding;
      merge.ErrorBound = errorBound;
      merge.Sides = sides;
      merge.Topological = !geometric;
      merge.D2ByCount = this->Params.D2ByCount;
      merge.Canonical = this->Canonical;
      merge.FrozenVertices = this->FrozenCopies;
      merge.Original = this->CopyOriginals;
      merge.DegreeFloor = this->Params.MergeDegreeFloor;
      merge.Order = this->Params.MergeOrder;
      merge.DeferFlagged = this->Params.DeferFlaggedMerges;
      merge.Blame = this->Params.MergeBlame;
      merge.FaceMerges = this->HasCurves && this->Params.FaceMerges;
      merge.JunctionMerges = this->HasCurves && this->Params.JunctionMerges;
      merge.CurveTolerance = this->HasCurves ? this->Params.CurveTolerance : 0;
      auto merged =
        [&](const qc::Clustering&, const viskores::cont::ArrayHandle<viskores::Id2>& pairs)
      {
        State state{ mesh, this->Quadrics, clusters };
        state.MergedPairs = &pairs;
        observe(Stage::Merged, state);
      };
      if (this->HasCurves)
      {
        qc::SegmentMerger(mesh, this->Quadrics, inputDistance, merge, this->Curves)
          .Run(clusters, this->MergeDiag, merged);
      }
      else
      {
        qc::MergePhase(
          mesh, this->Quadrics, inputDistance, merge, clusters, this->MergeDiag, merged);
      }
    }
    timer.Stop();
    seconds.Merge += timer.GetElapsedTime();
    timed(seconds.Output, [&] { this->ExtractOutput(numInputPoints, geometric); });
  }

  /// The cluster of every input vertex.
  VISKORES_CONT const viskores::cont::ArrayHandle<viskores::Id>& GetLabels() const
  {
    return this->Clusters.Labels;
  }
  /// The output vertices: the generator of every cluster.
  VISKORES_CONT const viskores::cont::ArrayHandle<Vec3>& GetGenerators() const
  {
    return this->Clusters.Generators;
  }
  /// The output triangles, as triples of cluster labels.
  VISKORES_CONT const viskores::cont::ArrayHandle<viskores::Id3>& GetOutputTriangles() const
  {
    return this->OutputTriangles;
  }
  /// The input triangle that produced every output triangle.
  VISKORES_CONT const viskores::cont::ArrayHandle<viskores::Id>& GetOutputTriangleSources() const
  {
    return this->OutputTriangleSources;
  }
  /// Groups the input vertices by cluster, for averaging point fields.
  VISKORES_CONT const viskores::worklet::Keys<viskores::Id>& GetClusterKeys() const
  {
    return this->Clusters.Keys;
  }
  VISKORES_CONT viskores::Id GetNumberOfClusters() const { return this->Clusters.NumberOfClusters; }
  VISKORES_CONT viskores::Id GetNumberOfClustersAfterLoop() const
  {
    return this->ClustersAfterLoop;
  }
  VISKORES_CONT const Diagnostics& GetDiagnostics() const { return this->Diag; }
  VISKORES_CONT const quadric_clustering::MergeDiagnostics& GetMergeDiagnostics() const
  {
    return this->MergeDiag;
  }
  VISKORES_CONT const quadric_clustering::SurfaceMesh& GetMesh() const { return this->Mesh; }
  VISKORES_CONT const quadric_clustering::VertexQuadrics& GetQuadrics() const
  {
    return this->Quadrics;
  }

  /// The output cells: triangles, and in the geometric-only mode also segments and points that
  /// lie in no other output cell. `GetOutputCellOffsets()` has one more entry than there are
  /// cells.
  VISKORES_CONT const viskores::cont::ArrayHandle<viskores::UInt8>& GetOutputCellShapes() const
  {
    return this->OutputCellShapes;
  }
  VISKORES_CONT const viskores::cont::ArrayHandle<viskores::Id>& GetOutputCellOffsets() const
  {
    return this->OutputCellOffsets;
  }
  VISKORES_CONT const viskores::cont::ArrayHandle<viskores::Id>& GetOutputCellConnectivity() const
  {
    return this->OutputCellConnectivity;
  }
  /// The input triangle that produced every output cell.
  VISKORES_CONT const viskores::cont::ArrayHandle<viskores::Id>& GetOutputCellSources() const
  {
    return this->OutputCellSources;
  }
  /// The output points.
  VISKORES_CONT const viskores::cont::ArrayHandle<Vec3>& GetOutputPoints() const
  {
    return this->OutputPoints;
  }
  /// The output point of every input point.
  VISKORES_CONT const viskores::cont::ArrayHandle<viskores::Id>& GetOutputPointMap() const
  {
    return this->OutputPointMap;
  }
  /// For an input that was cut into surfaces: the input point of every copy, and which copies
  /// were frozen. Empty otherwise.
  VISKORES_CONT const viskores::cont::ArrayHandle<viskores::Id>& GetCopyOriginals() const
  {
    return this->CopyOriginals;
  }
  VISKORES_CONT const viskores::cont::ArrayHandle<viskores::UInt8>& GetFrozenCopies() const
  {
    return this->FrozenCopies;
  }
  /// For a cut input with its singular curves decimated: the arcs and their segments.
  VISKORES_CONT const quadric_clustering::SingularCurves& GetSingularCurves() const
  {
    return this->Curves;
  }

private:
  /// Hypothesis "embedded", checked as Section 11 says: the test of the embedding flag applied
  /// to the input, every triangle a singleton cluster.
  VISKORES_CONT void CheckEmbedded(const viskores::cont::ArrayHandle<Vec3>& points,
                                   const viskores::cont::ArrayHandle<viskores::Id3>& triangles)
  {
    namespace qc = quadric_clustering;
    viskores::cont::ArrayHandle<viskores::Id> sizes;
    sizes.AllocateAndFill(points.GetNumberOfValues(), 1);
    viskores::cont::ArrayHandle<viskores::UInt8> flags;
    flags.AllocateAndFill(points.GetNumberOfValues(), 0);
    const qc::EmbeddingReport report = qc::EmbeddingFlags(triangles, points, sizes, flags);
    if (report.NumberOfDegenerateTriangles + report.NumberOfImproperPairs > 0)
    {
      throw viskores::cont::ErrorBadValue(
        "QuadricClusteringDecimation: the input is not embedded (" +
        std::to_string(report.NumberOfDegenerateTriangles) + " degenerate triangles, " +
        std::to_string(report.NumberOfImproperPairs) +
        " pairs of triangles that intersect improperly), so the embedding of the output "
        "cannot be guaranteed; turn the embedding flag off, or use the geometric-only mode.");
    }
  }

  /// Cuts the input along its singular edges and, with `Seams`, a seam into surfaces, and builds
  /// the mesh of the copies, with the frozen copies and the first copy of every input point.
  VISKORES_CONT void CutInput(const viskores::cont::ArrayHandle<Vec3>& points,
                              const viskores::cont::ArrayHandle<viskores::Id3>& triangles,
                              const viskores::cont::ArrayHandle<viskores::Id2>& creases)
  {
    namespace qc = quadric_clustering;
    using Algorithm = viskores::cont::Algorithm;
    const viskores::Id numPoints = points.GetNumberOfValues();
    const qc::CutComplex cut =
      qc::CutAlongSingularEdges(triangles, numPoints, creases, this->Params.Seams);
    // The copies of a vertex are numbered consecutively; the first is canonical.
    viskores::cont::ArrayHandle<Vec3> copyPoints;
    Algorithm::Copy(viskores::cont::make_ArrayHandlePermutation(cut.Original, points), copyPoints);
    Algorithm::LowerBounds(cut.Original, cut.Original, this->Canonical);
    qc::BuildSurfaceMesh(copyPoints, cut.Triangles, true, this->Mesh, !this->Params.Seams);
    this->FrozenCopies = cut.Frozen;
    this->CopyOriginals = cut.Original;
    if (this->Params.DecimateSingularCurves)
    {
      this->HasCurves = true;
      this->Curves =
        qc::FindSingularCurves(cut.CutEdges, this->Mesh, cut.Original, cut.Frozen, numPoints);
    }
  }

  /// The vertex of the glued output of every label of a cut input: the copies of every frozen
  /// vertex identified, and with the singular curves decimated, the pieces of every segment.
  VISKORES_CONT viskores::cont::ArrayHandle<viskores::Id> Glue(
    const quadric_clustering::Clustering& clusters) const
  {
    if (this->HasCurves)
    {
      return quadric_clustering::SegmentGlueMap(
        this->Curves, this->FrozenCopies, this->CopyOriginals, clusters);
    }
    return quadric_clustering::GlueMap(this->Canonical, clusters);
  }

  static bool ArraysEqual(const viskores::cont::ArrayHandle<viskores::Id>& a,
                          const viskores::cont::ArrayHandle<viskores::Id>& b)
  {
    if (a.GetNumberOfValues() != b.GetNumberOfValues())
    {
      return false;
    }
    viskores::cont::ArrayHandle<viskores::UInt8> differ;
    viskores::cont::Algorithm::Transform(a, b, differ, viskores::NotEqual{});
    return viskores::cont::Algorithm::Reduce(differ, viskores::Id(0)) == 0;
  }

  /// Keys whose equal neighbours are joined into the initial clusters: all vertices alike, but
  /// every frozen vertex apart.
  VISKORES_CONT viskores::cont::ArrayHandle<viskores::Id> InitialComponentKeys(viskores::Id n) const
  {
    viskores::cont::ArrayHandle<viskores::Id> keys;
    if (this->FrozenCopies.GetNumberOfValues() > 0)
    {
      viskores::cont::Invoker invoke;
      invoke(component_detail::ComponentKey{}, this->FrozenCopies, keys);
    }
    else
    {
      keys.AllocateAndFill(n, 0);
    }
    return keys;
  }

  /// The output of Algorithm 5. For a surface, the output triangles; for a cut input, glued at
  /// the copies of every frozen vertex and compacted; in the geometric-only mode, the simplices
  /// `g(t)`, each a triangle, a segment or a point, without those contained in another.
  VISKORES_CONT void ExtractOutput(viskores::Id numInputPoints, bool geometric)
  {
    namespace qc = quadric_clustering;
    namespace od = output_detail;
    using Algorithm = viskores::cont::Algorithm;
    viskores::cont::Invoker invoke;
    const qc::SurfaceMesh& mesh = this->Mesh;
    const qc::Clustering& clusters = this->Clusters;
    this->OutputTriangles =
      qc::OutputTriangles(mesh, clusters.Labels, &this->OutputTriangleSources);
    const viskores::Id k = clusters.NumberOfClusters;
    viskores::cont::ArrayHandle<viskores::Id> pointOfLabel;
    if (geometric)
    {
      // Every distinct label set, from its first input triangle, with repeated labels removed.
      viskores::cont::ArrayHandle<viskores::Vec<viskores::Id, 4>> records;
      invoke(od::SortedLabelRecord{}, mesh.Triangles, clusters.Labels, records);
      Algorithm::Sort(records);
      viskores::cont::ArrayHandle<viskores::UInt8> firstOfKey;
      invoke(qc::FirstOfRun<3>{},
             viskores::cont::ArrayHandleIndex(records.GetNumberOfValues()),
             records,
             firstOfKey);
      viskores::cont::ArrayHandle<viskores::Vec<viskores::Id, 4>> distinct;
      Algorithm::CopyIf(records, firstOfKey, distinct);
      // The edges of the triangles, and the labels in a simplex that is not a point.
      viskores::cont::ArrayHandle<viskores::Id2> triangleEdges;
      viskores::cont::ArrayHandle<viskores::Id> covered;
      invoke(od::SimplexFaces{},
             distinct,
             viskores::cont::make_ArrayHandleGroupVec<3>(triangleEdges),
             viskores::cont::make_ArrayHandleGroupVec<3>(covered));
      Algorithm::Sort(triangleEdges);
      Algorithm::Unique(triangleEdges);
      Algorithm::Sort(covered);
      Algorithm::Unique(covered);
      viskores::cont::ArrayHandle<viskores::UInt8> kept;
      invoke(od::NotContained{}, distinct, triangleEdges, covered, kept);
      // The kept simplices in the order of their first triangles.
      viskores::cont::ArrayHandle<viskores::Id> firstTriangles;
      invoke(od::RecordTriangle{}, distinct, firstTriangles);
      viskores::cont::ArrayHandle<viskores::Id> cellTriangles;
      Algorithm::CopyIf(firstTriangles, kept, cellTriangles);
      Algorithm::Sort(cellTriangles);
      viskores::cont::ArrayHandle<viskores::IdComponent> sizes;
      invoke(od::DistinctLabelCount{}, cellTriangles, mesh.Triangles, clusters.Labels, sizes);
      viskores::cont::ArrayHandle<viskores::Id> offsets;
      Algorithm::ScanExtended(viskores::cont::make_ArrayHandleCast<viskores::Id>(sizes), offsets);
      const viskores::Id total =
        viskores::cont::ArrayGetValue(cellTriangles.GetNumberOfValues(), offsets);
      this->OutputCellConnectivity.Allocate(total);
      invoke(od::WriteSimplex{},
             cellTriangles,
             viskores::cont::make_ArrayHandleView(offsets, 0, cellTriangles.GetNumberOfValues()),
             mesh.Triangles,
             clusters.Labels,
             this->OutputCellConnectivity,
             this->OutputCellShapes);
      this->OutputCellOffsets = offsets;
      this->OutputCellSources = cellTriangles;
      this->OutputPoints = clusters.Generators;
      Algorithm::Copy(viskores::cont::ArrayHandleIndex(k), pointOfLabel);
    }
    else
    {
      // Only the labels that are their own glued label remain, numbered in increasing order.
      viskores::cont::ArrayHandle<viskores::Id> glue;
      if (this->Canonical.GetNumberOfValues() > 0)
      {
        glue = this->Glue(clusters);
      }
      else
      {
        Algorithm::Copy(viskores::cont::ArrayHandleIndex(k), glue);
      }
      viskores::cont::ArrayHandle<viskores::Id> isOwnGlue, outputPoint;
      invoke(qc::IsEqual{}, viskores::cont::ArrayHandleIndex(k), glue, isOwnGlue);
      Algorithm::ScanExclusive(isOwnGlue, outputPoint);
      invoke(qc::GatherValue{}, glue, outputPoint, pointOfLabel);
      Algorithm::CopyIf(clusters.Generators, isOwnGlue, this->OutputPoints);
      const viskores::Id m = this->OutputTriangles.GetNumberOfValues();
      invoke(qc::GatherTriple{},
             this->OutputTriangles,
             pointOfLabel,
             viskores::cont::make_ArrayHandleGroupVec<3>(this->OutputCellConnectivity));
      Algorithm::Copy(viskores::cont::make_ArrayHandleConstant(
                        static_cast<viskores::UInt8>(viskores::CELL_SHAPE_TRIANGLE), m),
                      this->OutputCellShapes);
      Algorithm::Copy(viskores::cont::make_ArrayHandleCounting<viskores::Id>(0, 3, m + 1),
                      this->OutputCellOffsets);
      Algorithm::Copy(this->OutputTriangleSources, this->OutputCellSources);
    }
    // The output point of every input point, through any of its copies, all glued together.
    this->OutputPointMap.AllocateAndFill(numInputPoints, -1);
    if (this->CopyOriginals.GetNumberOfValues() > 0)
    {
      invoke(od::MapInputPoint{},
             this->CopyOriginals,
             clusters.Labels,
             pointOfLabel,
             this->OutputPointMap);
    }
    else
    {
      invoke(od::MapInputPoint{},
             viskores::cont::ArrayHandleIndex(numInputPoints),
             clusters.Labels,
             pointOfLabel,
             this->OutputPointMap);
    }
  }

  Parameters Params;
  viskores::cont::ArrayHandle<viskores::UInt64> Priority;
  Observer ObserverFunction;

  quadric_clustering::SurfaceMesh Mesh;
  quadric_clustering::VertexQuadrics Quadrics;
  quadric_clustering::Clustering Clusters;
  viskores::cont::ArrayHandle<viskores::Id3> OutputTriangles;
  viskores::cont::ArrayHandle<viskores::Id> OutputTriangleSources;
  viskores::cont::ArrayHandle<viskores::UInt8> OutputCellShapes;
  viskores::cont::ArrayHandle<viskores::Id> OutputCellOffsets;
  viskores::cont::ArrayHandle<viskores::Id> OutputCellConnectivity;
  viskores::cont::ArrayHandle<viskores::Id> OutputCellSources;
  viskores::cont::ArrayHandle<Vec3> OutputPoints;
  viskores::cont::ArrayHandle<viskores::Id> OutputPointMap;
  viskores::cont::ArrayHandle<viskores::UInt8> FrozenCopies;
  viskores::cont::ArrayHandle<viskores::Id> CopyOriginals;
  viskores::cont::ArrayHandle<viskores::Id> Canonical;
  bool HasCurves = false;
  quadric_clustering::SingularCurves Curves;
  viskores::Id ClustersAfterLoop = 0;
  Diagnostics Diag;
  quadric_clustering::MergeDiagnostics MergeDiag;
};

} // namespace worklet
} // namespace viskores

#endif // viskores_filter_geometry_refinement_worklet_QuadricClusteringDecimation_h
