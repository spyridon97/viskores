//============================================================================
//  The contents of this file are covered by the Viskores license. See
//  LICENSE.txt for details.
//
//  By contributing to this file, all contributors agree to the Developer
//  Certificate of Origin Version 1.1 (DCO 1.1) as stated in DCO.txt.
//============================================================================

// Tests of the assignment step of the quadric clustering decimation (Section 11, Algorithm 3):
// the guard and the commit rule keep every cluster connected and nonempty.

#include <viskores/cont/ArrayHandleIndex.h>
#include <viskores/cont/Invoker.h>
#include <viskores/cont/testing/Testing.h>

#include <viskores/filter/geometry_refinement/testing/QuadricClusteringTestMeshes.h>
#include <viskores/filter/geometry_refinement/worklet/quadric_clustering/Assignment.h>
#include <viskores/filter/geometry_refinement/worklet/quadric_clustering/VertexQuadrics.h>

#include <random>

namespace
{
namespace qc = viskores::worklet::quadric_clustering;
using qcd_test::ComponentsPerLabel;
using qcd_test::Mesh;
using qcd_test::ToVector;
using Vec3 = viskores::Vec3f_64;

void TestGuardGrid()
{
  std::cout << "Guard: the 5 x 4 grid of Proposition guard (iii)" << std::endl;
  // Nodes (i, j) at (i, j, 0), with diagonals from (i, j) to (i + 1, j + 1).
  Mesh grid;
  auto id = [](int i, int j) { return static_cast<viskores::Id>(i * 4 + j); };
  for (int i = 0; i < 5; ++i)
  {
    for (int j = 0; j < 4; ++j)
    {
      grid.Points.push_back(Vec3(i, j, 0));
    }
  }
  for (int i = 0; i < 4; ++i)
  {
    for (int j = 0; j < 3; ++j)
    {
      grid.Triangles.push_back({ id(i, j), id(i + 1, j), id(i + 1, j + 1) });
      grid.Triangles.push_back({ id(i, j), id(i + 1, j + 1), id(i, j + 1) });
    }
  }
  // A is the band of the proposition; B its complement at the corner; C = {(4, 3)}. u = (2, 2)
  // and v = (3, 3) are adjacent, each passes the guard, and leaving together disconnects A.
  std::vector<viskores::Id> labels(20, 0);
  for (auto node : { id(0, 0), id(0, 1), id(0, 2), id(0, 3), id(1, 0), id(1, 1) })
  {
    labels[static_cast<std::size_t>(node)] = 1;
  }
  labels[static_cast<std::size_t>(id(4, 3))] = 2;
  const viskores::Id u = id(2, 2), v = id(3, 3);
  std::vector<Vec3> generators = { Vec3(2, 1.5, 10), grid.Points[static_cast<std::size_t>(u)],
                                   grid.Points[static_cast<std::size_t>(v)] };

  qc::SurfaceMesh surface;
  qc::BuildSurfaceMesh(grid.PointArray(), grid.TriangleArray(), true, surface);
  qc::QuadricParameters parameters;
  const auto quadrics = qc::ComputeVertexQuadrics(surface, parameters);
  const auto ranges = qc::MakeRingRanges(surface.RingOffsets, 20);
  {
    // Both u and v propose to leave A, so only the commit rule keeps A connected.
    viskores::cont::ArrayHandle<viskores::UInt8> proposes;
    viskores::cont::ArrayHandle<viskores::Id> targets;
    viskores::cont::Invoker invoke;
    invoke(qc::detail::Propose{},
           viskores::cont::ArrayHandleIndex(surface.NumberOfPoints),
           ranges,
           surface.IsBoundaryVertex,
           surface.Points,
           quadrics.M,
           surface.Rings,
           surface.RingOffsets,
           viskores::cont::make_ArrayHandle(labels, viskores::CopyFlag::On),
           viskores::cont::make_ArrayHandle(generators, viskores::CopyFlag::On),
           viskores::cont::ArrayHandle<viskores::UInt8>(),
           proposes,
           targets);
    const auto p = ToVector(proposes);
    VISKORES_TEST_ASSERT(p[static_cast<std::size_t>(u)] && p[static_cast<std::size_t>(v)],
                         "The grid configuration should make u and v both propose.");
  }
  // Every order of the commit step must keep A connected: try many.
  for (viskores::UInt64 seed = 0; seed < 64; ++seed)
  {
    std::vector<viskores::UInt64> keys(20);
    std::mt19937_64 rng(seed);
    for (auto& key : keys)
    {
      key = rng();
    }
    auto labelArray = viskores::cont::make_ArrayHandle(labels, viskores::CopyFlag::On);
    bool uvBothProposed = false;
    qc::Assign(surface,
               ranges,
               surface.Rings,
               surface.RingOffsets,
               false,
               surface.IsBoundaryVertex,
               quadrics.M,
               viskores::cont::make_ArrayHandle(keys, viskores::CopyFlag::On),
               viskores::cont::ArrayHandle<viskores::UInt8>(),
               1,
               labelArray,
               viskores::cont::make_ArrayHandle(generators, viskores::CopyFlag::On),
               nullptr,
               [](const viskores::cont::ArrayHandle<viskores::Id>&) {});
    const auto after = ToVector(labelArray);
    uvBothProposed = after[static_cast<std::size_t>(u)] != 0 && after[static_cast<std::size_t>(v)] != 0;
    VISKORES_TEST_ASSERT(!uvBothProposed, "Adjacent vertices committed in the same sweep.");
    const auto components = ComponentsPerLabel(grid, after, 3);
    VISKORES_TEST_ASSERT(components[0] == 1 && components[1] == 1 && components[2] == 1,
                         "A sweep disconnected a cluster of the grid.");
  }
}

void TestGuardBoundaryPath()
{
  std::cout << "Guard: neighbours at the two ends of a boundary path" << std::endl;
  // A strip of triangles; A is three vertices of its bottom row. The middle one, v = 2, prefers
  // the other cluster, but its neighbours in A are the two ends of its ring, which is a path.
  Mesh strip;
  const int w = 5;
  for (int j = 0; j < 3; ++j)
  {
    for (int i = 0; i < w; ++i)
    {
      strip.Points.push_back(Vec3(i, j, 0));
    }
  }
  for (int j = 0; j < 2; ++j)
  {
    for (int i = 0; i < w - 1; ++i)
    {
      const viskores::Id a = j * w + i, b = a + 1, c = (j + 1) * w + i, d = c + 1;
      strip.Triangles.push_back({ a, b, d });
      strip.Triangles.push_back({ a, d, c });
    }
  }
  std::vector<viskores::Id> labels(15, 1);
  labels[1] = labels[2] = labels[3] = 0;
  std::vector<Vec3> generators = { Vec3(2, 50, 0), Vec3(2, 0.5, 0) };
  qc::SurfaceMesh surface;
  qc::BuildSurfaceMesh(strip.PointArray(), strip.TriangleArray(), true, surface);
  const auto quadrics = qc::ComputeVertexQuadrics(surface, qc::QuadricParameters{});
  auto labelArray = viskores::cont::make_ArrayHandle(labels, viskores::CopyFlag::On);
  qc::Assign(surface,
             qc::MakeRingRanges(surface.RingOffsets, 15),
             surface.Rings,
             surface.RingOffsets,
             false,
             surface.IsBoundaryVertex,
             quadrics.M,
             qc::DefaultPriority(15),
             viskores::cont::ArrayHandle<viskores::UInt8>(),
             0,
             labelArray,
             viskores::cont::make_ArrayHandle(generators, viskores::CopyFlag::On),
             nullptr,
             [&](const viskores::cont::ArrayHandle<viskores::Id>& current)
             {
               const auto l = ToVector(current);
               const auto components = ComponentsPerLabel(strip, l, 2);
               VISKORES_TEST_ASSERT(components[0] == 1 && components[1] == 1,
                                    "A sweep disconnected or emptied a cluster of the strip.");
             });
  const auto after = ToVector(labelArray);
  // Vertex 2 never leaves: first the guard blocks it, then it is the last vertex of A.
  VISKORES_TEST_ASSERT(after[2] == 0, "The guard let a vertex disconnect or empty its cluster.");
}

void TestQuadricClusteringDecimationAssignment()
{
  TestGuardGrid();
  TestGuardBoundaryPath();
}
} // anonymous namespace

int UnitTestQuadricClusteringDecimationAssignment(int argc, char* argv[])
{
  return viskores::cont::testing::Testing::Run(
    TestQuadricClusteringDecimationAssignment, argc, argv);
}
