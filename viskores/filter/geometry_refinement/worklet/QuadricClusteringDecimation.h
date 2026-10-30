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
#include <viskores/filter/geometry_refinement/worklet/quadric_clustering/EnvelopeFlags.h>
#include <viskores/filter/geometry_refinement/worklet/quadric_clustering/Initialization.h>
#include <viskores/filter/geometry_refinement/worklet/quadric_clustering/MergePhase.h>
#include <viskores/filter/geometry_refinement/worklet/quadric_clustering/QuadricClusteringCommonWorklets.h>
#include <viskores/filter/geometry_refinement/worklet/quadric_clustering/SurfaceMesh.h>
#include <viskores/filter/geometry_refinement/worklet/quadric_clustering/TopologyFlags.h>
#include <viskores/filter/geometry_refinement/worklet/quadric_clustering/VertexQuadrics.h>

#include <viskores/BinaryPredicates.h>
#include <viskores/cont/Algorithm.h>
#include <viskores/cont/ArrayHandle.h>
#include <viskores/cont/ArrayHandleGroupVec.h>
#include <viskores/cont/ArrayHandleIndex.h>
#include <viskores/cont/ErrorBadValue.h>
#include <viskores/cont/ErrorExecution.h>
#include <viskores/cont/Timer.h>

#include <cmath>
#include <functional>
#include <limits>

namespace viskores
{
namespace worklet
{

/// Worklets of the extraction of the output of `QuadricClusteringDecimation`.
namespace output_detail
{

/// The output point of an input point.
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
    /// The halves of the envelope flag (F5) with an error bound.
    quadric_clustering::EnvelopeSides ErrorSides = quadric_clustering::EnvelopeSides::Both;
    /// The cap `T` on the propose/commit sweeps of each assignment step, or 0 for none.
    viskores::Id MaxSweeps = 0;
    /// The initial clusters. Seeds are followed by a Lloyd phase.
    Initialization InitialClusters = Initialization::Curvature;
    /// The cap on the alternations of assignment and placement after seeding.
    viskores::Id MaxLloydIterations = 100;
    /// The insertion of step (d).
    Insertion InsertionRule = Insertion::Worst;
    /// The merge phase after the loop, down to `k*` in budget mode and as far as the flags allow
    /// in error mode.
    bool Merge = true;
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
    /// first test that flagged it, in the order (F0), (Fb), (F2)-(F3), (F5).
    viskores::Id FlaggedRadius = 0;
    viskores::Id FlaggedBudget = 0;
    viskores::Id FlaggedTopology = 0;
    viskores::Id FlaggedEnvelope = 0;
    /// Pieces examined by the envelope tests, over the run: a machine-independent measure of
    /// the most expensive step.
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
      double Envelope = 0;
      double InsertSplit = 0;
      double Merge = 0;
      double Output = 0;
    } Seconds;
  };

  VISKORES_CONT void SetParameters(const Parameters& parameters) { this->Params = parameters; }
  VISKORES_CONT const Parameters& GetParameters() const { return this->Params; }

  /// Sets the fixed order of the commit step: vertex `u` beats `v` if its key is larger, or equal
  /// with a larger index. By default the key is a hash of the vertex index. Implementations that
  /// are to agree must use the same order.
  VISKORES_CONT void SetPriority(const viskores::cont::ArrayHandle<viskores::UInt64>& priority)
  {
    this->Priority = priority;
  }

  VISKORES_CONT void SetObserver(const Observer& observer) { this->ObserverFunction = observer; }

  /// Decimates the triangles `triangles` on the points `points`. Throws `ErrorBadValue` for
  /// degenerate triangles, points in no triangle, and an input that is not a consistently
  /// oriented surface.
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
    const viskores::Id budget = this->Params.TargetNumberOfPoints;
    const viskores::Id numInputPoints = points.GetNumberOfValues();

    // The input: a surface.
    qc::BuildSurfaceMesh(points, triangles, this->Mesh);
    const qc::EnvelopeSides sides = this->Params.ErrorSides;
    const qc::SurfaceMesh& mesh = this->Mesh;
    const viskores::Id n = mesh.NumberOfPoints;
    const qc::QuadricParameters quadric = this->Params.Quadric;
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
                                          "per vertex of the mesh decimated.");
    }
    // One-rings for the guard of surfaces.
    const viskores::cont::ArrayHandle<viskores::Id>& ringOffsets = mesh.RingOffsets;
    const viskores::cont::ArrayHandle<viskores::Id>& rings = mesh.Rings;
    const viskores::cont::ArrayHandle<viskores::UInt8>& isBoundary = mesh.IsBoundaryVertex;
    const viskores::cont::ArrayHandle<viskores::Id2> ringRanges =
      qc::MakeRingRanges(ringOffsets, n);
    const Real errorBound = this->Params.ErrorBound;
    qc::DistanceMesh inputDistance;
    if (errorBound > 0)
    {
      inputDistance = qc::BuildDistanceMesh(mesh.Points, mesh.Triangles);
    }
    const bool topology = this->Params.Topology;
    qc::Clustering& clusters = this->Clusters;
    clusters = qc::Clustering{};

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
                   isBoundary,
                   this->Quadrics.M,
                   priority,
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
      observe(Stage::Placed, State{ mesh, this->Quadrics, clusters });
    };

    timer.Stop();
    seconds.Setup += timer.GetElapsedTime();
    timer.Start();
    // Initialisation: one cluster per component, or seeds and a Lloyd phase.
    const Initialization initialization = this->Params.InitialClusters;
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
                this->Diag.FlaggedTopology += qc::OrFlags(flags, topologyFlags);
              });
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
                            qc::OutputTriangles(mesh, clusters.Labels),
                            clusters.Sizes,
                            errorBound,
                            sides,
                            envelopeFlags);
        this->Diag.EnvelopePieces += envelopeReport.Pieces;
        this->Diag.FlaggedEnvelope += qc::OrFlags(flags, envelopeFlags);
        timer.Stop();
        seconds.Envelope += timer.GetElapsedTime();
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
      merge.ErrorBound = errorBound;
      merge.Sides = sides;
      merge.D2ByCount = this->Params.D2ByCount;
      merge.DegreeFloor = this->Params.MergeDegreeFloor;
      merge.Order = this->Params.MergeOrder;
      merge.DeferFlagged = this->Params.DeferFlaggedMerges;
      merge.Blame = this->Params.MergeBlame;
      auto merged =
        [&](const qc::Clustering&, const viskores::cont::ArrayHandle<viskores::Id2>& pairs)
      {
        State state{ mesh, this->Quadrics, clusters };
        state.MergedPairs = &pairs;
        observe(Stage::Merged, state);
      };
      qc::MergePhase(mesh, this->Quadrics, inputDistance, merge, clusters, this->MergeDiag, merged);
    }
    timer.Stop();
    seconds.Merge += timer.GetElapsedTime();
    timed(seconds.Output, [&] { this->ExtractOutput(numInputPoints); });
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

  /// The output cells: triangles.
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

private:
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

  /// Keys whose equal neighbours are joined into the initial clusters: all vertices alike.
  VISKORES_CONT viskores::cont::ArrayHandle<viskores::Id> InitialComponentKeys(viskores::Id n) const
  {
    viskores::cont::ArrayHandle<viskores::Id> keys;
    keys.AllocateAndFill(n, 0);
    return keys;
  }

  /// The output of Algorithm 5: the output triangles.
  VISKORES_CONT void ExtractOutput(viskores::Id numInputPoints)
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
    Algorithm::Copy(viskores::cont::ArrayHandleIndex(k), pointOfLabel);
    Algorithm::Copy(clusters.Generators, this->OutputPoints);
    invoke(qc::GatherTriple{},
           this->OutputTriangles,
           pointOfLabel,
           viskores::cont::make_ArrayHandleGroupVec<3>(this->OutputCellConnectivity));
    Algorithm::Copy(this->OutputTriangleSources, this->OutputCellSources);
    // The output point of every input point.
    this->OutputPointMap.AllocateAndFill(numInputPoints, -1);
    invoke(od::MapInputPoint{},
           viskores::cont::ArrayHandleIndex(numInputPoints),
           clusters.Labels,
           pointOfLabel,
           this->OutputPointMap);
  }

  Parameters Params;
  viskores::cont::ArrayHandle<viskores::UInt64> Priority;
  Observer ObserverFunction;

  quadric_clustering::SurfaceMesh Mesh;
  quadric_clustering::VertexQuadrics Quadrics;
  quadric_clustering::Clustering Clusters;
  viskores::cont::ArrayHandle<viskores::Id3> OutputTriangles;
  viskores::cont::ArrayHandle<viskores::Id> OutputTriangleSources;
  viskores::cont::ArrayHandle<viskores::Id> OutputCellConnectivity;
  viskores::cont::ArrayHandle<viskores::Id> OutputCellSources;
  viskores::cont::ArrayHandle<Vec3> OutputPoints;
  viskores::cont::ArrayHandle<viskores::Id> OutputPointMap;
  viskores::Id ClustersAfterLoop = 0;
  Diagnostics Diag;
  quadric_clustering::MergeDiagnostics MergeDiag;
};

} // namespace worklet
} // namespace viskores

#endif // viskores_filter_geometry_refinement_worklet_QuadricClusteringDecimation_h
