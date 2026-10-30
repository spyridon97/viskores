//============================================================================
//  The contents of this file are covered by the Viskores license. See
//  LICENSE.txt for details.
//
//  By contributing to this file, all contributors agree to the Developer
//  Certificate of Origin Version 1.1 (DCO 1.1) as stated in DCO.txt.
//============================================================================

// Tests of the precomputation of the quadric clustering decimation (Section 11, Algorithm 1):
// the vertex areas and the matrices M_v of the cost.

#include <viskores/cont/testing/Testing.h>

#include <viskores/filter/geometry_refinement/testing/QuadricClusteringTestMeshes.h>
#include <viskores/filter/geometry_refinement/worklet/quadric_clustering/SurfaceMesh.h>
#include <viskores/filter/geometry_refinement/worklet/quadric_clustering/VertexQuadrics.h>

#include <cmath>

namespace
{
namespace qc = viskores::worklet::quadric_clustering;
using qcd_test::Mesh;
using qcd_test::ToVector;

void TestQuadrics()
{
  std::cout << "Vertex quadrics of an octahedron" << std::endl;
  Mesh octahedron;
  octahedron.Points = { { 1, 0, 0 }, { -1, 0, 0 }, { 0, 1, 0 }, { 0, -1, 0 }, { 0, 0, 1 }, { 0, 0, -1 } };
  for (viskores::Id x : { 0, 1 })
  {
    for (viskores::Id y : { 2, 3 })
    {
      for (viskores::Id z : { 4, 5 })
      {
        // Oriented outward: the normal of (x, y, z) has the signs of the three coordinates.
        const bool flip = ((x == 1) + (y == 3) + (z == 5)) % 2 == 1;
        octahedron.Triangles.push_back(flip ? viskores::Id3(x, z, y) : viskores::Id3(x, y, z));
      }
    }
  }
  qc::SurfaceMesh surface;
  qc::BuildSurfaceMesh(octahedron.PointArray(), octahedron.TriangleArray(), surface);
  qc::QuadricParameters parameters;
  parameters.Alpha = 0.1;
  parameters.FeatureWeight = 0;
  const auto quadrics = qc::ComputeVertexQuadrics(surface, parameters);
  const double area = 4.0 * (std::sqrt(3.0) / 2.0) / 3.0;
  const auto m = ToVector(quadrics.M);
  const auto a = ToVector(quadrics.Area);
  for (std::size_t v = 0; v < 6; ++v)
  {
    VISKORES_TEST_ASSERT(test_equal(a[v], area), "Wrong vertex area.");
    // The four normals around a vertex are symmetric, so A_v = (a_v / 3) I.
    for (int i = 0; i < 9; ++i)
    {
      const double expected = (i % 4 == 0) ? area / 3 + 0.1 * area : 0.0;
      VISKORES_TEST_ASSERT(test_equal(m[v][i], expected), "Wrong M_v.");
    }
  }
}

void TestFeatureCosine()
{
  // The threshold of the sharp edges is the correctly rounded cosine of (deg * pi) / 180, as
  // det.cos of proofs/lib computes it in decimal arithmetic. The platform's cos differs at 44.9
  // and 120 degrees on some systems.
  std::cout << "Cosine of the feature angle" << std::endl;
  const struct
  {
    double Degrees;
    double Cosine;
  } cases[] = { { 0, 0x1.0000000000000p+0 },           { 1e-06, 0x1.fffffffffffffp-1 },
                { 0.5, 0x1.fffb02599c9cdp-1 },         { 10, 0x1.f838b8c811c17p-1 },
                { 30, 0x1.bb67ae8584cabp-1 },          { 44.9, 0x1.6aab84eaf62b4p-1 },
                { 45, 0x1.6a09e667f3bcdp-1 },          { 45.1, 0x1.6967ff9e7b6fdp-1 },
                { 60, 0x1.0000000000001p-1 },          { 89.999, 0x1.24d15211548ffp-16 },
                { 90, 0x1.1a62633145c07p-54 },         { 90.001, -0x1.24d152115bbcep-16 },
                { 120, -0x1.ffffffffffffcp-2 },        { 134.9, -0x1.6967ff9e7b6fdp-1 },
                { 135, -0x1.6a09e667f3bccp-1 },        { 150, -0x1.bb67ae8584cabp-1 },
                { 179.5, -0x1.fffb02599c9cdp-1 },      { 180, -0x1.0000000000000p+0 },
                { 73.123456789, 0x1.2947249b512bdp-2 } };
  for (const auto& c : cases)
  {
    const double cosine = qc::CorrectlyRoundedCos(c.Degrees * viskores::Pi() / 180);
    VISKORES_TEST_ASSERT(cosine == c.Cosine,
                         "The cosine of ",
                         c.Degrees,
                         " degrees is not correctly rounded: ",
                         cosine,
                         " vs ",
                         c.Cosine);
  }
}

void TestQuadricClusteringDecimationPrecomputation()
{
  TestQuadrics();
  TestFeatureCosine();
}
} // anonymous namespace

int UnitTestQuadricClusteringDecimationPrecomputation(int argc, char* argv[])
{
  return viskores::cont::testing::Testing::Run(
    TestQuadricClusteringDecimationPrecomputation, argc, argv);
}
