//============================================================================
//  The contents of this file are covered by the Viskores license. See
//  LICENSE.txt for details.
//
//  By contributing to this file, all contributors agree to the Developer
//  Certificate of Origin Version 1.1 (DCO 1.1) as stated in DCO.txt.
//============================================================================

// Tests of the guaranteed variational quadric clustering of the paper "Guaranteed Parallel Mesh
// Decimation by Variational Quadric Clustering". The checks are the invariants of its Section
// 11.13, asserted after every step of every round through the observer of the worklet, and the
// output sizes of the reference implementation (proofs/lib/vqc.py) on meshes that both build bit
// for bit, run with the hash order of this implementation.

#include <viskores/cont/ErrorBadValue.h>
#include <viskores/cont/RuntimeDeviceInformation.h>
#include <viskores/cont/RuntimeDeviceTracker.h>
#include <viskores/cont/testing/Testing.h>

#include <viskores/filter/geometry_refinement/testing/QuadricClusteringTestMeshes.h>
#include <viskores/filter/geometry_refinement/worklet/QuadricClusteringDecimation.h>
#include <viskores/filter/geometry_refinement/worklet/quadric_clustering/ImproperPairs.h>

#include <viskores/cont/ArrayHandleIndex.h>

#include <cstring>
#include <limits>
#include <numeric>
#include <random>
#include <tuple>

namespace
{
namespace qc = viskores::worklet::quadric_clustering;
using Decimation = viskores::worklet::QuadricClusteringDecimation;
using Stage = Decimation::Stage;
using qcd_test::ComponentsPerLabel;
using qcd_test::Mesh;
using qcd_test::ToVector;
using Vec3 = viskores::Vec3f_64;

long double Cost(const qc::Mat3& m, const Vec3& w)
{
  long double result = 0;
  for (int i = 0; i < 3; ++i)
  {
    for (int j = 0; j < 3; ++j)
    {
      result += static_cast<long double>(w[i]) * m[3 * i + j] * w[j];
    }
  }
  return result;
}

/// `E = sum_v cost(v, c_{L(v)})`.
long double Energy(const Mesh& mesh,
                   const std::vector<qc::Mat3>& m,
                   const std::vector<viskores::Id>& labels,
                   const std::vector<Vec3>& generators)
{
  long double e = 0;
  for (std::size_t v = 0; v < mesh.Points.size(); ++v)
  {
    e += Cost(m[v], generators[static_cast<std::size_t>(labels[v])] - mesh.Points[v]);
  }
  return e;
}

/// Asserts the invariants of Section 11.13 at every stage of a run.
struct InvariantChecker
{
  const Mesh& Input;
  viskores::Id EulerM;
  double Quantum; // lambda_low zeta^2
  std::vector<qc::Mat3> M;
  long double LastEnergy = std::numeric_limits<long double>::infinity();
  long double PlacedEnergy = 0;
  long double FlaggedCost = 0;
  viskores::Id NumFlagged = 0;
  viskores::Id PreviousClusters = 0;
  int Sweeps = 0, Rounds = 0, PlacementChecks = 0, MergeRounds = 0;

  InvariantChecker(const Mesh& input, viskores::Id eulerM, double quantum)
    : Input(input)
    , EulerM(eulerM)
    , Quantum(quantum)
  {
  }

  static bool NotAbove(long double a, long double b)
  {
    return a <= b + 1e-9L * std::max(1.0L, std::abs(b));
  }

  void CheckConnected(const std::vector<viskores::Id>& labels, viskores::Id k) const
  {
    const std::vector<int> components = ComponentsPerLabel(this->Input, labels, k);
    for (viskores::Id c = 0; c < k; ++c)
    {
      // Invariant 1, and Theorem "Number of rounds": no cluster is empty.
      VISKORES_TEST_ASSERT(components[static_cast<std::size_t>(c)] == 1,
                           "A cluster is disconnected or empty.");
    }
  }

  void operator()(Stage stage, const Decimation::State& state)
  {
    if (this->M.empty())
    {
      this->M = ToVector(state.Quadrics.M);
    }
    const viskores::Id k = state.Clusters.NumberOfClusters;
    if (stage != Stage::Merged && stage != Stage::Sweep)
    {
      this->PreviousClusters = k;
    }
    if (stage == Stage::Sweep)
    {
      ++this->Sweeps;
      const auto labels = ToVector(*state.SweepLabels);
      this->CheckConnected(labels, k);
      // Invariant 3: within one assignment step no vertex changes label more than k - 1 times.
      const auto changes = ToVector(*state.Changes);
      VISKORES_TEST_ASSERT(*std::max_element(changes.begin(), changes.end()) <= k - 1,
                           "A vertex changed label more than k - 1 times.");
      // Invariant 2: step (a) does not increase E.
      const long double e =
        Energy(this->Input, this->M, labels, ToVector(state.Clusters.Generators));
      VISKORES_TEST_ASSERT(NotAbove(e, this->LastEnergy), "A sweep increased E.");
      this->LastEnergy = e;
      return;
    }
    const auto labels = ToVector(state.Clusters.Labels);
    const auto generators = ToVector(state.Clusters.Generators);
    if (stage == Stage::Placed)
    {
      this->CheckConnected(labels, k);
      const long double e = Energy(this->Input, this->M, labels, generators);
      // Invariant 2: step (b) does not increase E.
      VISKORES_TEST_ASSERT(NotAbove(e, this->LastEnergy), "The placement increased E.");
      this->LastEnergy = this->PlacedEnergy = e;
      this->CheckPlacement(state, labels, generators);
    }
    else if (stage == Stage::Scanned)
    {
      ++this->Rounds;
      const auto flags = ToVector(*state.Flags);
      const auto sizes = ToVector(state.Clusters.Sizes);
      const auto worst = ToVector(state.Scan->Worst);
      const auto costs = ToVector(state.Scan->Costs);
      this->NumFlagged = 0;
      this->FlaggedCost = 0;
      for (viskores::Id c = 0; c < k; ++c)
      {
        if (flags[static_cast<std::size_t>(c)])
        {
          // Invariant 4: a flagged singleton is a bug.
          VISKORES_TEST_ASSERT(sizes[static_cast<std::size_t>(c)] >= 2, "A singleton was flagged.");
          const double cost =
            costs[static_cast<std::size_t>(worst[static_cast<std::size_t>(c)])];
          VISKORES_TEST_ASSERT(cost >= this->Quantum * (1 - 1e-9),
                               "A flagged cluster pays less than lambda zeta^2.");
          ++this->NumFlagged;
          this->FlaggedCost += cost;
        }
      }
      // Invariant 5: sum_C chi(M[C]) - T3 / 2 - Bcut / 2 = chi(M).
      if (state.Topology && state.Topology->Valid)
      {
        const auto& t = *state.Topology;
        VISKORES_TEST_ASSERT(2 * t.SumOfClusterEuler - t.NumberOfThreeLabelTriangles -
                                 t.NumberOfBoundaryCutEdges ==
                               2 * this->EulerM,
                             "The Euler identity of Lemma Euler fails.");
      }
    }
    else if (stage == Stage::Merged)
    {
      // Invariant 10: after every round of the merge phase the labelling has the discrete
      // closed-ball property, and k has decreased by at least one.
      ++this->MergeRounds;
      this->CheckConnected(labels, k);
      viskores::cont::ArrayHandle<viskores::UInt8> flags;
      flags.AllocateAndFill(k, 0);
      const qc::TopologyReport report =
        qc::TopologyFlags(state.Mesh, state.Clusters.Labels, state.Clusters.Sizes, flags);
      VISKORES_TEST_ASSERT(report.NumberOfD0 + report.NumberOfD1 + report.NumberOfD2 == 0,
                           "A merge round broke the discrete closed-ball property.");
      VISKORES_TEST_ASSERT(k <= this->PreviousClusters - 1, "A merge round did not merge.");
      VISKORES_TEST_ASSERT(k == this->PreviousClusters - state.MergedPairs->GetNumberOfValues(),
                           "A merge round lost track of its merges.");
      this->PreviousClusters = k;
    }
    else if (stage == Stage::Split)
    {
      this->CheckConnected(labels, k);
      // Invariant 2: step (d) decreases E by the costs of the worst members, at least
      // lambda zeta^2 per flagged cluster, and step (e) keeps it.
      const long double e = Energy(this->Input, this->M, labels, generators);
      VISKORES_TEST_ASSERT(NotAbove(e, this->PlacedEnergy - this->FlaggedCost),
                           "The insertion did not decrease E by the worst costs.");
      VISKORES_TEST_ASSERT(NotAbove(e, this->PlacedEnergy - this->Quantum * this->NumFlagged),
                           "The insertion decreased E by less than lambda zeta^2 per cluster.");
      this->LastEnergy = e;
    }
  }

  /// Invariant 11: `Phi_C(p) - Phi_C(c_C) = (p - c_C)^T Q_C (p - c_C)` for random `p`.
  void CheckPlacement(const Decimation::State& state,
                      const std::vector<viskores::Id>& labels,
                      const std::vector<Vec3>& generators)
  {
    if (this->PlacementChecks++ % 4 != 0)
    {
      return;
    }
    const auto q = ToVector(state.Clusters.Q);
    std::vector<std::vector<viskores::Id>> members(generators.size());
    for (std::size_t v = 0; v < labels.size(); ++v)
    {
      members[static_cast<std::size_t>(labels[v])].push_back(static_cast<viskores::Id>(v));
    }
    std::mt19937 rng(7);
    std::uniform_real_distribution<double> offset(-0.3, 0.3);
    for (std::size_t c = 0; c < generators.size(); c += 1 + generators.size() / 16)
    {
      const Vec3 p = generators[c] + Vec3(offset(rng), offset(rng), offset(rng));
      auto phi = [&](const Vec3& x)
      {
        long double s = 0;
        for (viskores::Id v : members[c])
        {
          s += Cost(this->M[static_cast<std::size_t>(v)], x - this->Input.Points[static_cast<std::size_t>(v)]);
        }
        return s;
      };
      const long double lhs = phi(p) - phi(generators[c]);
      const long double rhs = Cost(q[c], p - generators[c]);
      VISKORES_TEST_ASSERT(std::abs(lhs - rhs) <= 1e-9L * std::max(1.0L, phi(p)),
                           "The generator does not minimise the cluster's cost.");
    }
  }
};

double ShortestEdge(const Mesh& mesh)
{
  double shortest = std::numeric_limits<double>::infinity();
  for (const auto& e : qcd_test::EdgeCounts(mesh.Triangles))
  {
    shortest = std::min(shortest,
                        qcd_test::Norm(mesh.Points[static_cast<std::size_t>(e.first.first)] -
                                       mesh.Points[static_cast<std::size_t>(e.first.second)]));
  }
  return shortest;
}

std::vector<Vec3> GeneratorVector(const Decimation& decimation)
{
  return ToVector(decimation.GetGenerators());
}

/// Runs the decimation with the invariant checker, and checks the output on termination:
/// invariants 6 and 7.
Decimation RunChecked(const Mesh& mesh, const Decimation::Parameters& parameters)
{
  const viskores::Id eulerM =
    qcd_test::EulerCharacteristic(static_cast<viskores::Id>(mesh.Points.size()), mesh.Triangles);
  const int loopsM = qcd_test::BoundaryLoops(mesh.Triangles);

  Decimation decimation;
  decimation.SetParameters(parameters);
  // lambda_low, from a first run of the precomputation alone.
  double lambdaLow = std::numeric_limits<double>::infinity();
  {
    qc::SurfaceMesh surface;
    qc::BuildSurfaceMesh(mesh.PointArray(), mesh.TriangleArray(), true, surface);
    const auto quadrics = qc::ComputeVertexQuadrics(surface, parameters.Quadric);
    for (double a : ToVector(quadrics.Area))
    {
      lambdaLow = std::min(lambdaLow, parameters.Quadric.Alpha * a);
    }
  }
  const double zeta = std::min(parameters.MaxClusterRadius, ShortestEdge(mesh) / 2);
  InvariantChecker checker(mesh, eulerM, lambdaLow * zeta * zeta);
  decimation.SetObserver(std::ref(checker));
  decimation.Run(mesh.PointArray(), mesh.TriangleArray());
  VISKORES_TEST_ASSERT(checker.Rounds == decimation.GetDiagnostics().Rounds, "Missed rounds.");
  std::cout << "  n = " << mesh.Points.size() << ", k = " << decimation.GetNumberOfClusters()
            << ", rounds checked " << checker.Rounds << ", sweeps checked " << checker.Sweeps
            << std::endl;

  // Invariant 6: the output is a surface of the input's Euler characteristic and boundary loops.
  const viskores::Id k = decimation.GetNumberOfClusters();
  const auto triangles = ToVector(decimation.GetOutputTriangles());
  const std::string problem = qcd_test::CheckSurface(k, triangles);
  VISKORES_TEST_ASSERT(problem.empty(), "The output is not a surface: ", problem);
  VISKORES_TEST_ASSERT(qcd_test::EulerCharacteristic(k, triangles) == eulerM,
                       "The output has another Euler characteristic.");
  VISKORES_TEST_ASSERT(qcd_test::BoundaryLoops(triangles) == loopsM,
                       "The output has another number of boundary loops.");

  // Invariant 7: sampled points of each lie within max r_C of the other.
  const auto labels = ToVector(decimation.GetLabels());
  const auto generators = GeneratorVector(decimation);
  double maxRadius = 0;
  for (std::size_t v = 0; v < labels.size(); ++v)
  {
    maxRadius = std::max(
      maxRadius, qcd_test::Norm(generators[static_cast<std::size_t>(labels[v])] - mesh.Points[v]));
  }
  const double spacing = ShortestEdge(mesh) / 2;
  const double hausdorff =
    qcd_test::SampledHausdorff(mesh.Points, mesh.Triangles, generators, triangles, spacing);
  VISKORES_TEST_ASSERT(hausdorff <= maxRadius * (1 + 1e-12),
                       "The output is farther from the input than the largest cluster radius.");
  if (parameters.ErrorBound > 0)
  {
    // Invariant 7, in error mode: within epsilon_H on the sides in use, sampled more finely.
    const double eps = parameters.ErrorBound;
    const double outToIn =
      qcd_test::SampledDistance(generators, triangles, mesh.Points, mesh.Triangles, eps / 4);
    const double inToOut =
      qcd_test::SampledDistance(mesh.Points, mesh.Triangles, generators, triangles, eps / 4);
    std::cout << "  sampled distances " << outToIn / eps << " (output to input) and "
              << inToOut / eps << " (input to output) epsilon_H" << std::endl;
    const auto sides = static_cast<viskores::UInt8>(parameters.ErrorSides);
    VISKORES_TEST_ASSERT(
      !(sides & static_cast<viskores::UInt8>(qc::EnvelopeSides::OutputToInput)) || outToIn <= eps,
      "The output is not within epsilon_H of the input.");
    VISKORES_TEST_ASSERT(
      !(sides & static_cast<viskores::UInt8>(qc::EnvelopeSides::InputToOutput)) || inToOut <= eps,
      "The input is not within epsilon_H of the output.");
  }

  // Invariant 8: with (F4), an exact all-pairs test finds no improper pair of output triangles.
  if (parameters.Embedding)
  {
    viskores::Id improper = 0;
    for (std::size_t i = 0; i < triangles.size(); ++i)
    {
      const Vec3 p[3] = { generators[static_cast<std::size_t>(triangles[i][0])],
                          generators[static_cast<std::size_t>(triangles[i][1])],
                          generators[static_cast<std::size_t>(triangles[i][2])] };
      VISKORES_TEST_ASSERT(!viskores::TriangleIsDegenerate(p[0], p[1], p[2]),
                           "A degenerate output triangle.");
      for (std::size_t j = i + 1; j < triangles.size(); ++j)
      {
        const Vec3 q[3] = { generators[static_cast<std::size_t>(triangles[j][0])],
                            generators[static_cast<std::size_t>(triangles[j][1])],
                            generators[static_cast<std::size_t>(triangles[j][2])] };
        improper += qc::exact::ImproperPair(p, triangles[i], q, triangles[j]) ? 1 : 0;
      }
    }
    VISKORES_TEST_ASSERT(improper == 0, "The output intersects itself.");
  }
  return decimation;
}

Decimation::Parameters Plain(viskores::Id budget = 0,
                             double maxRadius = std::numeric_limits<double>::infinity())
{
  Decimation::Parameters p;
  p.Quadric.Alpha = 0.05;
  p.Quadric.FeatureWeight = 0;
  p.Quadric.BoundaryWeight = 0;
  p.TargetNumberOfPoints = budget;
  p.MaxClusterRadius = maxRadius;
  p.Embedding = false;
  p.Merge = false;
  return p;
}

Decimation::Parameters Augmented(viskores::Id budget = 0,
                                 double maxRadius = std::numeric_limits<double>::infinity())
{
  Decimation::Parameters p = Plain(budget, maxRadius);
  p.Quadric.FeatureWeight = 1;
  p.Quadric.BoundaryWeight = 1;
  return p;
}

double LargestSide(const Mesh& mesh)
{
  Vec3 lo(1e300), hi(-1e300);
  for (const auto& p : mesh.Points)
  {
    for (int a = 0; a < 3; ++a)
    {
      lo[a] = std::min(lo[a], p[a]);
      hi[a] = std::max(hi[a], p[a]);
    }
  }
  return std::max({ hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2] });
}

struct Expected
{
  viskores::Id Clusters, Rounds, Lloyd, Radius, Topology, Embedding = 0, Envelope = 0;
};

void CheckExpected(const Decimation& decimation, const Expected& expected)
{
  const auto& d = decimation.GetDiagnostics();
  VISKORES_TEST_ASSERT(decimation.GetNumberOfClusters() == expected.Clusters,
                       "Output size differs from the reference: ",
                       decimation.GetNumberOfClusters(),
                       " vs ",
                       expected.Clusters);
  VISKORES_TEST_ASSERT(d.Rounds == expected.Rounds, "Rounds differ from the reference.");
  VISKORES_TEST_ASSERT(d.LloydIterations == expected.Lloyd, "Lloyd phase differs.");
  VISKORES_TEST_ASSERT(d.FlaggedRadius == expected.Radius, "(F0) counts differ.");
  VISKORES_TEST_ASSERT(d.FlaggedTopology == expected.Topology, "(F2)-(F3) counts differ.");
  VISKORES_TEST_ASSERT(d.FlaggedEmbedding == expected.Embedding, "(F4) counts differ.");
  VISKORES_TEST_ASSERT(d.FlaggedEnvelope == expected.Envelope, "(F5) counts differ.");
}

Decimation::Parameters WithEmbedding(Decimation::Parameters p)
{
  p.Embedding = true;
  return p;
}

Decimation::Parameters DefaultPath(const Mesh& mesh, bool augmented)
{
  Decimation::Parameters p = augmented ? Augmented() : Plain();
  p.Embedding = true;
  p.ErrorBound = 0.01 * LargestSide(mesh);
  return p;
}

void TestEnvelope()
{
  // The default path of Section 11: (F2) to (F5), epsilon_H 1% of the largest side. The
  // expected numbers come from proofs/lib/vqc.py with the curvature initialisation, the hash
  // order, exact embedding predicates, and the second condition of its envelope test applied to
  // every input triangle instead of the eight with the nearest centroids.
  std::cout << "Envelope: sphere, augmented quadric" << std::endl;
  const Mesh sphere = qcd_test::Icosphere(3);
  CheckExpected(RunChecked(sphere, DefaultPath(sphere, true)), { 195, 3, 4, 0, 0, 0, 75 });
  std::cout << "Envelope: plate with two holes, plain quadric" << std::endl;
  const Mesh plate = qcd_test::VoxelSolid(qcd_test::PlateWithHoles(2), 0.25);
  CheckExpected(RunChecked(plate, DefaultPath(plate, false)), { 147, 7, 5, 0, 0, 0, 114 });
  std::cout << "Envelope: annulus, augmented quadric" << std::endl;
  const Mesh annulus = qcd_test::Annulus();
  CheckExpected(RunChecked(annulus, DefaultPath(annulus, true)), { 193, 5, 6, 0, 0, 0, 123 });
  std::cout << "Envelope: genus-2 plate with an opening, augmented quadric" << std::endl;
  const Mesh opening = qcd_test::Genus2PlateWithOpening();
  CheckExpected(RunChecked(opening, DefaultPath(opening, true)), { 66, 6, 5, 0, 4, 0, 31 });
}

void TestErrorSides()
{
  // The default path of proofs/cor-skip.py (vii) with the merge phase, alpha 0.02 and
  // epsilon_H 1% of the largest side, checking both sides of the error bound, the output to
  // the input, or the input to the output. RunChecked samples the sides in use. The expected
  // sizes come from proofs/lib/vqc.py with error_sides both, output and input.
  auto withSides = [](const Mesh& mesh, qc::EnvelopeSides sides)
  {
    Decimation::Parameters p;
    p.Quadric.Alpha = 0.02;
    p.ErrorBound = 0.01 * LargestSide(mesh);
    p.Embedding = true;
    p.ErrorSides = sides;
    return p;
  };
  const struct
  {
    const char* Name;
    Mesh Input;
    viskores::Id Both, OutputOnly, InputOnly;
  } cases[] = {
    { "sphere", qcd_test::Icosphere(3), 114, 112, 114 },
    { "disk", qcd_test::Disk(), 109, 107, 108 },
    { "annulus", qcd_test::Annulus(), 107, 103, 102 },
    { "torus with one hole", qcd_test::TorusWithHoles(1), 130, 125, 126 },
    { "torus with two holes", qcd_test::TorusWithHoles(2), 131, 131, 128 },
    { "genus-2 plate with an opening", qcd_test::Genus2PlateWithOpening(), 31, 26, 31 }
  };
  for (const auto& c : cases)
  {
    const std::pair<qc::EnvelopeSides, viskores::Id> runs[] = {
      { qc::EnvelopeSides::Both, c.Both },
      { qc::EnvelopeSides::OutputToInput, c.OutputOnly },
      { qc::EnvelopeSides::InputToOutput, c.InputOnly }
    };
    for (const auto& run : runs)
    {
      std::cout << "Error sides " << static_cast<int>(run.first) << ": " << c.Name << std::endl;
      const Decimation decimation = RunChecked(c.Input, withSides(c.Input, run.first));
      VISKORES_TEST_ASSERT(decimation.GetNumberOfClusters() == run.second,
                           "Output size differs from the reference: ",
                           decimation.GetNumberOfClusters(),
                           " vs ",
                           run.second);
    }
  }
}

/// Checks the output size and merge rounds of a run of the merge phase against the reference,
/// and, without geometric flags, that it stopped at `k*` or at an irreducible complex: contracting
/// any edge of the output gives a complex of another type (Theorem "Output size", (iii), (iv)).
void CheckMerged(const Mesh& mesh,
                 const Decimation::Parameters& parameters,
                 viskores::Id loop,
                 viskores::Id merged,
                 viskores::Id mergeRounds)
{
  const Decimation decimation = RunChecked(mesh, parameters);
  VISKORES_TEST_ASSERT(decimation.GetNumberOfClustersAfterLoop() == loop,
                       "The loop stopped elsewhere than the reference.");
  VISKORES_TEST_ASSERT(decimation.GetNumberOfClusters() == merged,
                       "The merge phase stopped elsewhere than the reference: ",
                       decimation.GetNumberOfClusters(),
                       " vs ",
                       merged);
  VISKORES_TEST_ASSERT(decimation.GetMergeDiagnostics().Rounds == mergeRounds,
                       "The merge phase took another number of rounds.");
  const viskores::Id k = decimation.GetNumberOfClusters();
  const bool geometric = parameters.Embedding || parameters.ErrorBound > 0 ||
    parameters.MaxClusterRadius < std::numeric_limits<double>::infinity();
  if (!geometric && k > parameters.TargetNumberOfPoints)
  {
    const auto triangles = ToVector(decimation.GetOutputTriangles());
    const auto type = qcd_test::SurfaceType(triangles);
    for (const auto& e : qcd_test::EdgeCounts(triangles))
    {
      VISKORES_TEST_ASSERT(
        qcd_test::SurfaceType(qcd_test::Contract(triangles, e.first.first, e.first.second)) != type,
        "The merge phase stopped at a complex with a contractible edge.");
    }
  }
}

Decimation::Parameters Merging(Decimation::Parameters p, viskores::Id degreeFloor)
{
  p.Merge = true;
  p.MergeDegreeFloor = degreeFloor;
  return p;
}

void TestMerge()
{
  // The expected numbers come from proofs/lib/vqc.py with the curvature initialisation and the
  // hash order (exact embedding predicates, and the envelope rule of TestEnvelope), with
  // merge_priority=None (the priority the increase of the objective, then the labels) and
  // merge_priority=12 (the degree key), and coarsen=True in error mode.
  const Mesh plate2 = qcd_test::VoxelSolid(qcd_test::PlateWithHoles(2), 1.0);
  const Mesh plate3 = qcd_test::VoxelSolid(qcd_test::PlateWithHoles(3), 1.0);
  const Mesh opening = qcd_test::Genus2PlateWithOpening();
  const Mesh sphere = qcd_test::Icosphere(3);
  std::cout << "Merge by cost: plate with two holes, budget 2" << std::endl;
  CheckMerged(plate2, Merging(Plain(2), -1), 22, 10, 6);
  std::cout << "Merge by cost: plate with two holes, budget 2, with (F4)" << std::endl;
  CheckMerged(plate2, Merging(WithEmbedding(Plain(2)), -1), 22, 12, 6);
  std::cout << "Merge by cost: plate with three holes, budget 2" << std::endl;
  CheckMerged(plate3, Merging(Plain(2), -1), 38, 15, 7);
  std::cout << "Merge by cost: disk, budget 2" << std::endl;
  CheckMerged(qcd_test::Disk(), Merging(Plain(2), -1), 4, 3, 1);
  std::cout << "Merge by cost: annulus, budget 2" << std::endl;
  CheckMerged(qcd_test::Annulus(), Merging(Plain(2), -1), 16, 6, 5);
  std::cout << "Merge by cost: genus-2 plate with an opening, budget 2, with (F4)" << std::endl;
  CheckMerged(opening, Merging(WithEmbedding(Plain(2)), -1), 17, 13, 3);
  std::cout << "Merge by cost: sphere, radius cap, error mode" << std::endl;
  CheckMerged(sphere, Merging(Plain(0, 0.08 * LargestSide(sphere)), -1), 207, 179, 3);

  std::cout << "Merge with the degree key: plate with two holes, budget 2" << std::endl;
  CheckMerged(plate2, Merging(Plain(2), 12), 22, 11, 6);
  std::cout << "Merge with the degree key: plate with two holes, budget 2, with (F4)" << std::endl;
  CheckMerged(plate2, Merging(WithEmbedding(Plain(2)), 12), 22, 13, 5);
  std::cout << "Merge with the degree key: plate with three holes, budget 2, with (F4)" << std::endl;
  CheckMerged(plate3, Merging(WithEmbedding(Plain(2)), 12), 47, 17, 9);
  std::cout << "Merge with the degree key: genus-2 plate with an opening, budget 2" << std::endl;
  CheckMerged(opening, Merging(Plain(2), 12), 17, 11, 3);
  std::cout << "Merge with the degree key: annulus, radius cap, error mode" << std::endl;
  const Mesh annulus = qcd_test::Annulus();
  CheckMerged(annulus, Merging(Plain(0, 0.08 * LargestSide(annulus)), 12), 157, 128, 4);

  // A component that is a triangle or a tetrahedron is irreducible while other components still
  // merge; without geometric flags only the local test guards the merges, so it must leave them
  // alone (Proposition "A merge is an edge contraction"). The icosphere stops at a tetrahedron.
  std::cout << "Merge with the degree key: icosphere, tetrahedron and triangle, budget 6"
            << std::endl;
  const Decimation components =
    RunChecked(qcd_test::IcosphereWithSmallComponents(2), Merging(Plain(6), 12));
  VISKORES_TEST_ASSERT(components.GetNumberOfClusters() == 11,
                       "The merge phase stopped elsewhere than at three irreducible components.");
}

void TestEmbedding()
{
  // The expected numbers come from proofs/lib/vqc.py with the curvature initialisation, the hash
  // order and its embedding test evaluated by the exact rational predicates of
  // proofs/lib/exact.py.
  std::cout << "Embedding: plate with three holes, topology only, augmented quadric" << std::endl;
  CheckExpected(RunChecked(qcd_test::VoxelSolid(qcd_test::PlateWithHoles(3), 1.0),
                           WithEmbedding(Augmented())),
                { 46, 7, 5, 0, 7, 0 });
  std::cout << "Embedding: genus-2 plate with an opening, topology only, augmented quadric"
            << std::endl;
  CheckExpected(RunChecked(qcd_test::Genus2PlateWithOpening(), WithEmbedding(Augmented())),
                { 31, 2, 5, 0, 2, 0 });
  std::cout << "Embedding: sphere, budget 2" << std::endl;
  CheckExpected(RunChecked(qcd_test::Icosphere(3), WithEmbedding(Plain(2))), { 8, 3, 4, 0, 6, 0 });
}

void TestLoop()
{
  // The expected numbers come from proofs/lib/vqc.py run with the curvature initialisation and
  // the hash order of this implementation, with merge=False.
  std::cout << "Loop: sphere, budget 2" << std::endl;
  const Mesh sphere = qcd_test::Icosphere(3);
  CheckExpected(RunChecked(sphere, Plain(2)), { 8, 3, 4, 0, 6 });

  std::cout << "Loop: sphere, radius cap, augmented quadric" << std::endl;
  CheckExpected(RunChecked(sphere, Augmented(0, 0.08 * LargestSide(sphere))), { 207, 3, 4, 47, 0 });

  std::cout << "Loop: plate with two holes, budget 2" << std::endl;
  const Mesh plate2 = qcd_test::VoxelSolid(qcd_test::PlateWithHoles(2), 1.0);
  CheckExpected(RunChecked(plate2, Plain(2)), { 22, 8, 5, 0, 20 });

  std::cout << "Loop: plate with three holes, budget 60" << std::endl;
  const Mesh plate3 = qcd_test::VoxelSolid(qcd_test::PlateWithHoles(3), 1.0);
  CheckExpected(RunChecked(plate3, Plain(60)), { 60, 2, 7, 0, 0 });

  std::cout << "Loop: disk, topology only, augmented quadric" << std::endl;
  CheckExpected(RunChecked(qcd_test::Disk(), Augmented()), { 95, 1, 5, 0, 0 });

  std::cout << "Loop: annulus, radius cap, augmented quadric" << std::endl;
  const Mesh annulus = qcd_test::Annulus();
  CheckExpected(RunChecked(annulus, Augmented(0, 0.08 * LargestSide(annulus))),
                { 155, 4, 6, 49, 0 });

  std::cout << "Loop: genus-2 plate with an opening, budget 60, one sweep per step" << std::endl;
  Decimation::Parameters oneSweep = Plain(60);
  oneSweep.MaxSweeps = 1;
  CheckExpected(RunChecked(qcd_test::Genus2PlateWithOpening(), oneSweep), { 63, 3, 7, 0, 3 });

  // Meshes with trigonometric coordinates: invariants only.
  std::cout << "Loop: thin torus, radius cap" << std::endl;
  const Mesh thinTorus = qcd_test::Torus(1.0, 0.1, 40, 6);
  RunChecked(thinTorus, Plain(0, 0.08 * LargestSide(thinTorus)));
  std::cout << "Loop: torus with a hole, topology only" << std::endl;
  const Mesh torus = qcd_test::Torus(1.0, 0.35, 32, 14);
  const Mesh holed = qcd_test::RemoveTriangles(
    torus,
    [&](const viskores::Id3& t)
    {
      for (int c = 0; c < 3; ++c)
      {
        const Vec3 p = torus.Points[static_cast<std::size_t>(t[c])];
        if (std::abs(std::atan2(p[1], p[0])) < 0.3 && p[2] > 0.15)
        {
          return true;
        }
      }
      return false;
    });
  RunChecked(holed, Augmented());
}

/// Invariant 9: the output is bitwise identical across runs and devices.
void CheckDeterministic(const Mesh& mesh, const Decimation::Parameters& parameters)
{
  auto run = [&](viskores::cont::DeviceAdapterId device)
  {
    viskores::cont::ScopedRuntimeDeviceTracker tracker(device);
    Decimation decimation;
    decimation.SetParameters(parameters);
    decimation.Run(mesh.PointArray(), mesh.TriangleArray());
    return std::make_tuple(ToVector(decimation.GetLabels()),
                           ToVector(decimation.GetOutputPoints()),
                           ToVector(decimation.GetOutputCellConnectivity()));
  };
  const auto reference = run(viskores::cont::DeviceAdapterTagSerial{});
  std::vector<viskores::cont::DeviceAdapterId> devices = { viskores::cont::DeviceAdapterTagSerial{} };
  viskores::cont::RuntimeDeviceInformation info;
  for (viskores::Int8 i = 1; i < VISKORES_MAX_DEVICE_ADAPTER_ID; ++i)
  {
    const auto device = viskores::cont::make_DeviceAdapterId(i);
    if (device != viskores::cont::DeviceAdapterTagSerial{} && info.Exists(device) &&
        viskores::cont::GetRuntimeDeviceTracker().CanRunOn(device))
    {
      devices.push_back(device);
    }
  }
  for (const auto& device : devices)
  {
    const auto result = run(device);
    VISKORES_TEST_ASSERT(std::get<0>(result) == std::get<0>(reference),
                         "Labels differ on ",
                         device.GetName());
    const auto& points = std::get<1>(result);
    const auto& expected = std::get<1>(reference);
    VISKORES_TEST_ASSERT(points.size() == expected.size() &&
                           std::memcmp(points.data(), expected.data(), points.size() * sizeof(Vec3)) == 0,
                         "Output points differ on ",
                         device.GetName());
    VISKORES_TEST_ASSERT(std::get<2>(result) == std::get<2>(reference),
                         "Output cells differ on ",
                         device.GetName());
  }
}

void TestDeterminism()
{
  std::cout << "Determinism across runs and devices: annulus, budget mode" << std::endl;
  CheckDeterministic(qcd_test::Annulus(), Augmented(40));
}

void TestInvalidInput()
{
  std::cout << "Invalid inputs" << std::endl;
  auto rejects = [](const Mesh& mesh)
  {
    try
    {
      Decimation decimation;
      decimation.SetParameters(Plain(4));
      decimation.Run(mesh.PointArray(), mesh.TriangleArray());
    }
    catch (const viskores::cont::ErrorBadValue&)
    {
      return true;
    }
    return false;
  };
  const Mesh sphere = qcd_test::Icosphere(1);
  Mesh unused = sphere;
  unused.Points.push_back(Vec3(5, 5, 5));
  VISKORES_TEST_ASSERT(rejects(unused), "A point in no triangle should be rejected.");
  Mesh flipped = sphere;
  std::swap(flipped.Triangles[0][1], flipped.Triangles[0][2]);
  VISKORES_TEST_ASSERT(rejects(flipped), "Inconsistent orientation should be rejected.");
  Mesh degenerate = sphere;
  degenerate.Points[static_cast<std::size_t>(degenerate.Triangles[0][0])] =
    degenerate.Points[static_cast<std::size_t>(degenerate.Triangles[0][1])];
  VISKORES_TEST_ASSERT(rejects(degenerate), "A zero-area triangle should be rejected.");
  // Two tetrahedra sharing a vertex: a pinched vertex.
  Mesh pinched;
  pinched.Points = { { 0, 0, 0 }, { 1, 0, 0 }, { 0, 1, 0 }, { 0, 0, 1 },
                     { -1, 0, 0 }, { 0, -1, 0 }, { 0, 0, -1 } };
  pinched.Triangles = { { 0, 2, 1 }, { 0, 1, 3 }, { 0, 3, 2 }, { 1, 2, 3 },
                        { 0, 4, 5 }, { 0, 6, 4 }, { 0, 5, 6 }, { 4, 6, 5 } };
  VISKORES_TEST_ASSERT(rejects(pinched), "A pinched vertex should be rejected.");
  // A fin: an edge in three triangles.
  Mesh fin = sphere;
  fin.Points.push_back(Vec3(3, 3, 3));
  const viskores::Id3 t = fin.Triangles[0];
  fin.Triangles.push_back({ t[0], t[1], static_cast<viskores::Id>(fin.Points.size()) - 1 });
  VISKORES_TEST_ASSERT(rejects(fin), "An edge in three triangles should be rejected.");
  // Hypothesis "embedded": with (F4), an input that intersects itself is rejected; without, it
  // is decimated.
  const Mesh overlapping = qcd_test::OverlappingIcospheres(2);
  auto run = [&](bool embedding)
  {
    Decimation decimation;
    Decimation::Parameters p = Plain();
    p.Embedding = embedding;
    p.ErrorBound = 0.02;
    decimation.SetParameters(p);
    decimation.Run(overlapping.PointArray(), overlapping.TriangleArray());
  };
  bool rejected = false;
  try
  {
    run(true);
  }
  catch (const viskores::cont::ErrorBadValue&)
  {
    rejected = true;
  }
  VISKORES_TEST_ASSERT(rejected, "An input that intersects itself should be rejected with (F4).");
  run(false);
}

void TestQuadricClusteringDecimationWorklet()
{
  TestLoop();
  TestEmbedding();
  TestEnvelope();
  TestErrorSides();
  TestMerge();
  TestDeterminism();
  TestInvalidInput();
}
} // anonymous namespace

int UnitTestQuadricClusteringDecimationWorklet(int argc, char* argv[])
{
  return viskores::cont::testing::Testing::Run(TestQuadricClusteringDecimationWorklet, argc, argv);
}
