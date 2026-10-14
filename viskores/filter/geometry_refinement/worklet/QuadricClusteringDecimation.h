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
#include <viskores/filter/geometry_refinement/worklet/quadric_clustering/IsotopyFlags.h>
#include <viskores/filter/geometry_refinement/worklet/quadric_clustering/MergePhase.h>
#include <viskores/filter/geometry_refinement/worklet/quadric_clustering/NonManifoldCut.h>
#include <viskores/filter/geometry_refinement/worklet/quadric_clustering/SegmentMerges.h>
#include <viskores/filter/geometry_refinement/worklet/quadric_clustering/SurfaceMesh.h>
#include <viskores/filter/geometry_refinement/worklet/quadric_clustering/TopologyFlags.h>
#include <viskores/filter/geometry_refinement/worklet/quadric_clustering/VertexQuadrics.h>

#include <viskores/BinaryPredicates.h>
#include <viskores/cont/Algorithm.h>
#include <viskores/cont/ArrayHandle.h>
#include <viskores/cont/ArrayHandleConstant.h>
#include <viskores/cont/ErrorBadValue.h>
#include <viskores/cont/ErrorExecution.h>
#include <viskores/cont/Timer.h>

#include <algorithm>
#include <array>
#include <functional>
#include <limits>
#include <map>
#include <numeric>
#include <set>
#include <string>
#include <vector>

namespace viskores
{
namespace worklet
{

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
    /// enforced by the envelope flag (F5), or 0.
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
    /// The isotopy flag (F6) of Theorem "isotopy": the output is isotopic to the input, a closed
    /// surface sampling a smooth one whose reach is at least `Quadric.Reach`. Needs the reach,
    /// the topological flags and (F4), and caps the cluster radius at
    /// `(1 - 2^-10) (R - h^2 / R)`, with `h` the longest edge.
    bool Isotopy = false;
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
    /// When the merges of a round fail the full test together: defer those whose merged cluster
    /// is flagged and test the others again, or, if false, merge only the one of highest
    /// priority.
    bool DeferFlaggedMerges = true;
    /// For an input cut into surfaces: decimate its singular curves too, by merging consecutive
    /// segments of them in the merge phase (Proposition "Decimating the singular curves");
    /// otherwise they keep their input resolution (Theorem "Inputs that are not surfaces").
    bool DecimateSingularCurves = true;
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
    const quadric_clustering::EmbeddingReport* Embedding = nullptr;
    const quadric_clustering::EnvelopeReport* Envelope = nullptr;
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
    /// first test that flagged it, in the order (F0), (Fb), (F2)-(F3), (F4), (F5).
    viskores::Id FlaggedRadius = 0;
    viskores::Id FlaggedBudget = 0;
    viskores::Id FlaggedTopology = 0;
    viskores::Id FlaggedEmbedding = 0;
    viskores::Id FlaggedEnvelope = 0;
    viskores::Id FlaggedIsotopy = 0;
    /// Clusters found adjacent to two copies of a frozen vertex of a cut input, over the loop;
    /// they are flagged with (F2) and (F3).
    viskores::Id GlueViolations = 0;
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
      double Isotopy = 0;
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
  /// consistently oriented surface is cut along its singular edges and a seam into surfaces
  /// whose copies of the cut vertices stay fixed, and glued back (Theorem "Inputs that are not
  /// surfaces"), unless the geometric-only mode is on. Throws `ErrorBadValue` for degenerate
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
    const std::string problem = qc::BuildSurfaceMesh(points, triangles, false, this->Mesh);
    this->Canonical.ReleaseResources();
    this->FrozenVertices.ReleaseResources();
    this->CopyOriginal.ReleaseResources();
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
    const bool isotopy = this->Params.Isotopy;
    if (isotopy &&
        (geometric || !this->Params.Topology || !this->Params.Embedding ||
         !(this->Params.Quadric.Reach > 0)))
    {
      throw viskores::cont::ErrorBadValue("QuadricClusteringDecimation: the isotopy flag needs "
                                          "the topological and embedding flags and a reach.");
    }
    if (isotopy && (!problem.empty() || this->Mesh.HasBoundary))
    {
      throw viskores::cont::ErrorBadValue(
        "QuadricClusteringDecimation: the isotopy flag needs a closed surface.");
    }
    if (!geometric && !problem.empty())
    {
      if (budget > 0)
      {
        throw viskores::cont::ErrorBadValue("QuadricClusteringDecimation: budget mode needs a "
                                            "surface, and the input is not one: " +
                                            problem + ".");
      }
      this->CutInput(points, triangles);
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
    // With the isotopy flag, the radius cap of Theorem "isotopy", below R - h^2 / R.
    Real maxRadius = this->Params.MaxClusterRadius;
    qc::IsotopyMesh isotopyMesh;
    if (isotopy)
    {
      isotopyMesh =
        qc::BuildIsotopyMesh(mesh, this->Quadrics.TriangleNormals, this->Params.Quadric.Reach);
      const Real reach = isotopyMesh.Reach, h = isotopyMesh.H;
      maxRadius = viskores::Min(maxRadius, (1 - Real(1) / 1024) * (reach - h * h / reach));
    }
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
    auto assign = [&](bool observed)
    {
      if (this->ObserverFunction || !observed)
      {
        changes.AllocateAndFill(n, 0);
      }
      auto sweepObserver = [&](const viskores::cont::ArrayHandle<viskores::Id>& labels)
      {
        if (observed)
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
                   (this->ObserverFunction || !observed) ? &changes : nullptr,
                   sweepObserver);
      this->Diag.Sweeps += sweeps;
    };
    auto place = [&]()
    {
      qc::Place(mesh, this->Quadrics.M, clusters);
      if (this->HasCurves)
      {
        qc::ShareGenerators(this->Glue(clusters), clusters);
      }
      frozenLabels = qc::FrozenLabels(this->FrozenVertices, clusters);
      observe(Stage::Placed, State{ mesh, this->Quadrics, clusters });
    };

    timer.Stop();
    seconds.Setup += timer.GetElapsedTime();
    timer.Start();
    // Initialisation: one cluster per component, frozen vertices apart, or seeds and a Lloyd
    // phase. An input cut into surfaces and the geometric-only mode start from the components.
    Initialization initialization = this->Params.InitialClusters;
    if (geometric || this->FrozenVertices.GetNumberOfValues() > 0)
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
                    mesh, clusters.Labels, this->CopyOriginal, this->Curves, this->Glue(clusters));
                  this->Diag.GlueViolations += listed.GetNumberOfValues();
                  qc::FlagClusters(listed, clusters.Sizes, topologyFlags);
                }
                else
                {
                  this->Diag.GlueViolations += qc::GlueFlags(mesh,
                                                             clusters.Labels,
                                                             this->FrozenVertices,
                                                             this->Canonical,
                                                             clusters.Sizes,
                                                             topologyFlags);
                }
                this->Diag.FlaggedTopology += qc::OrFlags(flags, topologyFlags);
              });
      }
      // The embedding test is the most expensive, and only the stopping round needs it.
      qc::EmbeddingReport embeddingReport;
      if (embedding && qc::CountFlags(flags) == 0)
      {
        timer.Start();
        viskores::cont::ArrayHandle<viskores::UInt8> embeddingFlags;
        embeddingFlags.AllocateAndFill(k, 0);
        const auto outputTriangles = qc::OutputTriangles(mesh, clusters.Labels);
        this->Diag.EmbeddingTriangles += outputTriangles.GetNumberOfValues();
        embeddingReport = qc::EmbeddingFlags(
          qc::GlueTriangles(outputTriangles, this->Glue(clusters)),
          clusters.Generators,
          clusters.Sizes,
          embeddingFlags);
        this->Diag.FlaggedEmbedding += qc::OrFlags(flags, embeddingFlags);
        timer.Stop();
        seconds.Embedding += timer.GetElapsedTime();
      }
      // The envelope test needs the discrete closed-ball property, so it runs only when nothing
      // else is flagged, the topological flags in particular.
      qc::EnvelopeReport envelopeReport;
      if (errorBound > 0 && qc::CountFlags(flags) == 0)
      {
        timer.Start();
        viskores::cont::ArrayHandle<viskores::UInt8> envelopeFlags;
        envelopeFlags.AllocateAndFill(k, 0);
        envelopeReport = qc::EnvelopeFlags(mesh,
                                           inputDistance,
                                           clusters.Labels,
                                           clusters.Generators,
                                           geometric ? qc::Simplices(mesh, clusters.Labels)
                                                     : qc::OutputTriangles(mesh, clusters.Labels),
                                           clusters.Sizes,
                                           errorBound,
                                           envelopeMode,
                                           this->Params.ErrorSides,
                                           envelopeFlags);
        this->Diag.EnvelopePieces += envelopeReport.Pieces;
        this->Diag.FlaggedEnvelope += qc::OrFlags(flags, envelopeFlags);
        timer.Stop();
        seconds.Envelope += timer.GetElapsedTime();
      }
      // The isotopy test, last, likewise.
      if (isotopy && qc::CountFlags(flags) == 0)
      {
        timer.Start();
        viskores::cont::ArrayHandle<viskores::UInt8> isotopyFlags;
        isotopyFlags.AllocateAndFill(k, 0);
        qc::IsotopyFlags(
          mesh, isotopyMesh, clusters.Labels, clusters.Generators, clusters.Sizes, isotopyFlags);
        this->Diag.FlaggedIsotopy += qc::OrFlags(flags, isotopyFlags);
        timer.Stop();
        seconds.Isotopy += timer.GetElapsedTime();
      }

      State state{ mesh, this->Quadrics, clusters };
      state.Scan = &scan;
      state.Flags = &flags;
      state.Topology = &topologyReport;
      state.Embedding = &embeddingReport;
      state.Envelope = &envelopeReport;
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
              frozenLabels = qc::FrozenLabels(this->FrozenVertices, clusters);
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
      merge.Isotopy = isotopy ? &isotopyMesh : nullptr;
      merge.Embedding = embedding;
      merge.ErrorBound = errorBound;
      merge.Sides = this->Params.ErrorSides;
      merge.Topological = !geometric;
      merge.D2ByCount = this->Params.D2ByCount;
      merge.Canonical = this->Canonical;
      merge.FrozenVertices = this->FrozenVertices;
      merge.Original = this->CopyOriginal;
      merge.DegreeFloor = this->Params.MergeDegreeFloor;
      merge.DeferFlagged = this->Params.DeferFlaggedMerges;
      auto merged = [&](const qc::Clustering&, const viskores::cont::ArrayHandle<viskores::Id2>& pairs)
      {
        State state{ mesh, this->Quadrics, clusters };
        state.MergedPairs = &pairs;
        observe(Stage::Merged, state);
      };
      if (this->HasCurves)
      {
        qc::SegmentMerger(mesh, this->Quadrics, inputDistance, merge, this->Curves)
          .Run(this->FrozenVertices, clusters, this->MergeDiag, merged);
      }
      else
      {
        qc::MergePhase(
          mesh, this->Quadrics, inputDistance, this->FrozenVertices, merge, clusters, this->MergeDiag, merged);
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
  VISKORES_CONT viskores::Id GetNumberOfClusters() const
  {
    return this->Clusters.NumberOfClusters;
  }
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
    return this->CopyOriginal;
  }
  VISKORES_CONT const viskores::cont::ArrayHandle<viskores::UInt8>& GetFrozenCopies() const
  {
    return this->FrozenVertices;
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

  /// Cuts the input along its singular edges and a seam into surfaces, and builds the mesh of
  /// the copies, with the frozen copies and the first copy of every input point.
  VISKORES_CONT void CutInput(const viskores::cont::ArrayHandle<Vec3>& points,
                              const viskores::cont::ArrayHandle<viskores::Id3>& triangles)
  {
    namespace qc = quadric_clustering;
    std::vector<viskores::Id3> hostTriangles(static_cast<std::size_t>(triangles.GetNumberOfValues()));
    {
      auto portal = triangles.ReadPortal();
      for (std::size_t t = 0; t < hostTriangles.size(); ++t)
      {
        hostTriangles[t] = portal.Get(static_cast<viskores::Id>(t));
      }
    }
    const viskores::Id numPoints = points.GetNumberOfValues();
    const qc::CutComplex cut = qc::CutAlongSingularEdges(hostTriangles, numPoints);
    const std::size_t numCopies = cut.Original.size();
    std::vector<Vec3> copyPoints(numCopies);
    std::vector<viskores::Id> canonical(numCopies);
    std::vector<viskores::Id> firstCopy(static_cast<std::size_t>(numPoints), -1);
    {
      auto portal = points.ReadPortal();
      for (std::size_t c = 0; c < numCopies; ++c)
      {
        const viskores::Id v = cut.Original[c];
        copyPoints[c] = portal.Get(v);
        if (firstCopy[static_cast<std::size_t>(v)] < 0)
        {
          firstCopy[static_cast<std::size_t>(v)] = static_cast<viskores::Id>(c);
        }
        canonical[c] = firstCopy[static_cast<std::size_t>(v)];
      }
    }
    qc::BuildSurfaceMesh(viskores::cont::make_ArrayHandle(copyPoints, viskores::CopyFlag::On),
                         viskores::cont::make_ArrayHandle(cut.Triangles, viskores::CopyFlag::On),
                         true,
                         this->Mesh);
    this->FrozenVertices = viskores::cont::make_ArrayHandle(cut.Frozen, viskores::CopyFlag::On);
    this->CopyOriginal = viskores::cont::make_ArrayHandle(cut.Original, viskores::CopyFlag::On);
    this->Canonical = viskores::cont::make_ArrayHandle(canonical, viskores::CopyFlag::On);
    if (this->Params.DecimateSingularCurves)
    {
      this->HasCurves = true;
      this->Curves =
        qc::FindSingularCurves(cut.CutEdges, this->Mesh, cut.Original, cut.Frozen, numPoints);
    }
  }

  /// Keys whose equal neighbours are joined into the initial clusters: all vertices alike, but
  /// every frozen vertex apart.
  VISKORES_CONT viskores::cont::ArrayHandle<viskores::Id> InitialComponentKeys(viskores::Id n) const
  {
    viskores::cont::ArrayHandle<viskores::Id> keys;
    keys.AllocateAndFill(n, 0);
    if (this->FrozenVertices.GetNumberOfValues() > 0)
    {
      auto frozen = this->FrozenVertices.ReadPortal();
      auto portal = keys.WritePortal();
      for (viskores::Id v = 0; v < n; ++v)
      {
        if (frozen.Get(v))
        {
          portal.Set(v, -1 - v);
        }
      }
    }
    return keys;
  }

  /// The vertex of the glued output of every label of a cut input: the copies of every frozen
  /// vertex identified, and with the singular curves decimated, the pieces of every segment.
  VISKORES_CONT viskores::cont::ArrayHandle<viskores::Id> Glue(
    const quadric_clustering::Clustering& clusters) const
  {
    if (this->HasCurves)
    {
      return quadric_clustering::SegmentGlueMap(
        this->Curves, this->FrozenVertices, this->CopyOriginal, clusters);
    }
    return quadric_clustering::GlueMap(this->Canonical, clusters);
  }

  /// The output of Algorithm 5. For a surface, the output triangles; for a cut input, glued at
  /// the copies of every frozen vertex and compacted; in the geometric-only mode, the simplices
  /// `g(t)`, each a triangle, a segment or a point, without those contained in another.
  VISKORES_CONT void ExtractOutput(viskores::Id numInputPoints, bool geometric)
  {
    namespace qc = quadric_clustering;
    const qc::SurfaceMesh& mesh = this->Mesh;
    const qc::Clustering& clusters = this->Clusters;
    this->OutputTriangles =
      qc::OutputTriangles(mesh, clusters.Labels, &this->OutputTriangleSources);
    const viskores::Id k = clusters.NumberOfClusters;
    std::vector<viskores::Id> pointOfLabel(static_cast<std::size_t>(k));
    std::iota(pointOfLabel.begin(), pointOfLabel.end(), 0);
    std::vector<viskores::UInt8> shapes;
    std::vector<viskores::Id> offsets = { 0 };
    std::vector<viskores::Id> connectivity;
    std::vector<viskores::Id> sources;
    auto addCell = [&](const std::vector<viskores::Id>& ids, viskores::Id source)
    {
      shapes.push_back(ids.size() == 3 ? viskores::CELL_SHAPE_TRIANGLE
                                       : (ids.size() == 2 ? viskores::CELL_SHAPE_LINE
                                                          : viskores::CELL_SHAPE_VERTEX));
      connectivity.insert(connectivity.end(), ids.begin(), ids.end());
      offsets.push_back(static_cast<viskores::Id>(connectivity.size()));
      sources.push_back(source);
    };
    if (geometric)
    {
      // Every distinct label set, from its first input triangle, with repeated labels removed.
      auto labels = clusters.Labels.ReadPortal();
      auto tris = mesh.Triangles.ReadPortal();
      std::map<std::array<viskores::Id, 3>, viskores::Id> first;
      for (viskores::Id t = 0; t < mesh.NumberOfTriangles; ++t)
      {
        const viskores::Id3 tri = tris.Get(t);
        std::array<viskores::Id, 3> key = { labels.Get(tri[0]), labels.Get(tri[1]), labels.Get(tri[2]) };
        std::sort(key.begin(), key.end());
        first.emplace(key, t);
      }
      std::set<std::pair<viskores::Id, viskores::Id>> triangleEdges;
      std::set<viskores::Id> covered;
      for (const auto& entry : first)
      {
        const auto& key = entry.first;
        if (key[0] != key[1] && key[1] != key[2])
        {
          triangleEdges.insert({ key[0], key[1] });
          triangleEdges.insert({ key[1], key[2] });
          triangleEdges.insert({ key[0], key[2] });
        }
        if (key[0] != key[2])
        {
          covered.insert(key.begin(), key.end());
        }
      }
      std::vector<std::pair<viskores::Id, std::vector<viskores::Id>>> cells;
      for (const auto& entry : first)
      {
        const auto& key = entry.first;
        const viskores::Id3 tri = tris.Get(entry.second);
        std::vector<viskores::Id> ids;
        for (int c = 0; c < 3; ++c)
        {
          const viskores::Id l = labels.Get(tri[c]);
          if (std::find(ids.begin(), ids.end(), l) == ids.end())
          {
            ids.push_back(l);
          }
        }
        if ((ids.size() == 2 && triangleEdges.count({ key[0], key[2] })) ||
            (ids.size() == 1 && covered.count(key[0])))
        {
          continue; // contained in another simplex
        }
        cells.emplace_back(entry.second, ids);
      }
      std::sort(cells.begin(), cells.end());
      for (const auto& cell : cells)
      {
        addCell(cell.second, cell.first);
      }
      this->OutputPoints = clusters.Generators;
    }
    else
    {
      std::vector<viskores::Id> glue(static_cast<std::size_t>(k));
      std::iota(glue.begin(), glue.end(), 0);
      if (this->Canonical.GetNumberOfValues() > 0)
      {
        const viskores::cont::ArrayHandle<viskores::Id> glueMap = this->Glue(clusters);
        auto map = glueMap.ReadPortal();
        for (viskores::Id c = 0; c < k; ++c)
        {
          glue[static_cast<std::size_t>(c)] = map.Get(c);
        }
      }
      // Only the labels that are their own glued label remain.
      std::vector<Vec3> points;
      auto generators = clusters.Generators.ReadPortal();
      for (viskores::Id c = 0; c < k; ++c)
      {
        if (glue[static_cast<std::size_t>(c)] == c)
        {
          pointOfLabel[static_cast<std::size_t>(c)] = static_cast<viskores::Id>(points.size());
          points.push_back(generators.Get(c));
        }
      }
      for (viskores::Id c = 0; c < k; ++c)
      {
        pointOfLabel[static_cast<std::size_t>(c)] =
          pointOfLabel[static_cast<std::size_t>(glue[static_cast<std::size_t>(c)])];
      }
      auto triangles = this->OutputTriangles.ReadPortal();
      auto source = this->OutputTriangleSources.ReadPortal();
      for (viskores::Id t = 0; t < triangles.GetNumberOfValues(); ++t)
      {
        const viskores::Id3 tri = triangles.Get(t);
        addCell({ pointOfLabel[static_cast<std::size_t>(tri[0])],
                  pointOfLabel[static_cast<std::size_t>(tri[1])],
                  pointOfLabel[static_cast<std::size_t>(tri[2])] },
                source.Get(t));
      }
      this->OutputPoints = viskores::cont::make_ArrayHandle(points, viskores::CopyFlag::On);
    }
    // The output point of every input point, through any of its copies.
    std::vector<viskores::Id> pointMap(static_cast<std::size_t>(numInputPoints), -1);
    auto labels = clusters.Labels.ReadPortal();
    if (this->CopyOriginal.GetNumberOfValues() > 0)
    {
      auto original = this->CopyOriginal.ReadPortal();
      for (viskores::Id c = 0; c < mesh.NumberOfPoints; ++c)
      {
        pointMap[static_cast<std::size_t>(original.Get(c))] =
          pointOfLabel[static_cast<std::size_t>(labels.Get(c))];
      }
    }
    else
    {
      for (viskores::Id v = 0; v < numInputPoints; ++v)
      {
        pointMap[static_cast<std::size_t>(v)] = pointOfLabel[static_cast<std::size_t>(labels.Get(v))];
      }
    }
    this->OutputPointMap = viskores::cont::make_ArrayHandle(pointMap, viskores::CopyFlag::On);
    this->OutputCellShapes = viskores::cont::make_ArrayHandle(shapes, viskores::CopyFlag::On);
    this->OutputCellOffsets = viskores::cont::make_ArrayHandle(offsets, viskores::CopyFlag::On);
    this->OutputCellConnectivity = viskores::cont::make_ArrayHandle(connectivity, viskores::CopyFlag::On);
    this->OutputCellSources = viskores::cont::make_ArrayHandle(sources, viskores::CopyFlag::On);
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
  viskores::cont::ArrayHandle<viskores::UInt8> FrozenVertices;
  viskores::cont::ArrayHandle<viskores::Id> CopyOriginal;
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
