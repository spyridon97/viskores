//============================================================================
//  The contents of this file are covered by the Viskores license. See
//  LICENSE.txt for details.
//
//  By contributing to this file, all contributors agree to the Developer
//  Certificate of Origin Version 1.1 (DCO 1.1) as stated in DCO.txt.
//============================================================================

// Tests of the predicates behind the envelope flag (F5) of the quadric clustering decimation:
// the one-sided distance bound and the certified envelope test. The exact predicates and the
// bounding volume hierarchy they use have their own tests.

#include <viskores/cont/Invoker.h>
#include <viskores/cont/testing/Testing.h>

#include <viskores/filter/geometry_refinement/testing/QuadricClusteringTestMeshes.h>
#include <viskores/filter/geometry_refinement/worklet/quadric_clustering/EnvelopeFlags.h>

#include <array>
#include <limits>
#include <random>
#include <set>
#include <vector>

namespace
{
namespace qc = viskores::worklet::quadric_clustering;
using Vec3 = viskores::Vec3f_64;

/// Distance from `p` to the triangle `abc` in long double: the projection onto the plane if it
/// lies in the triangle, else the nearest of the three edges.
long double ReferenceDistance(const Vec3& p, const Vec3& a, const Vec3& b, const Vec3& c)
{
  using L = long double;
  auto sub = [](const Vec3& x, const Vec3& y)
  { return std::array<L, 3>{ L(x[0]) - y[0], L(x[1]) - y[1], L(x[2]) - y[2] }; };
  auto dot = [](const std::array<L, 3>& x, const std::array<L, 3>& y)
  { return x[0] * y[0] + x[1] * y[1] + x[2] * y[2]; };
  const auto ab = sub(b, a), ac = sub(c, a), ap = sub(p, a);
  const L d00 = dot(ab, ab), d01 = dot(ab, ac), d11 = dot(ac, ac), d20 = dot(ap, ab), d21 = dot(ap, ac);
  const L den = d00 * d11 - d01 * d01;
  L best = std::numeric_limits<L>::infinity();
  if (den > 0)
  {
    const L v = (d11 * d20 - d01 * d21) / den, w = (d00 * d21 - d01 * d20) / den;
    if (v >= 0 && w >= 0 && v + w <= 1)
    {
      std::array<L, 3> r;
      for (int i = 0; i < 3; ++i)
      {
        r[i] = ap[i] - v * ab[i] - w * ac[i];
      }
      best = std::sqrt(dot(r, r));
    }
  }
  const Vec3* ends[3][2] = { { &a, &b }, { &b, &c }, { &c, &a } };
  for (const auto& e : ends)
  {
    const auto d = sub(*e[1], *e[0]), q = sub(p, *e[0]);
    const L t = std::min(L(1), std::max(L(0), dot(q, d) / dot(d, d)));
    std::array<L, 3> r;
    for (int i = 0; i < 3; ++i)
    {
      r[i] = q[i] - t * d[i];
    }
    best = std::min(best, std::sqrt(dot(r, r)));
  }
  return best;
}

void TestDistanceBound()
{
  std::cout << "One-sided distance bound" << std::endl;
  std::mt19937 rng(5);
  std::uniform_real_distribution<double> coordinate(-3, 3), small(-1e-7, 1e-7);
  double worstSlack = 0;
  for (int i = 0; i < 20000; ++i)
  {
    const Vec3 a(coordinate(rng), coordinate(rng), coordinate(rng));
    Vec3 b(coordinate(rng), coordinate(rng), coordinate(rng));
    Vec3 c(coordinate(rng), coordinate(rng), coordinate(rng));
    if (i % 3 == 1)
    {
      c = a + 0.5 * (b - a) + Vec3(small(rng), small(rng), small(rng)); // a thin triangle
    }
    if (i % 5 == 2)
    {
      b = a; // a segment
    }
    const Vec3 p(coordinate(rng), coordinate(rng), coordinate(rng));
    const long double truth = ReferenceDistance(p, a, b, c);
    const double bound = qc::PointTriangleDistanceUpper(p, a, b, c);
    VISKORES_TEST_ASSERT(bound >= truth * (1 - 1e-15L), "The upper bound is below the distance.");
    worstSlack = std::max(worstSlack, static_cast<double>(bound - truth));
  }
  std::cout << "  largest slack " << worstSlack << std::endl;
  VISKORES_TEST_ASSERT(worstSlack < 1e-6, "The upper bound is far from the distance.");
}

void TestCertifiedTest()
{
  std::cout << "Certified envelope test" << std::endl;
  const qcd_test::Mesh sphere = qcd_test::Icosphere(2);
  const qc::DistanceMesh mesh =
    qc::BuildDistanceMesh(sphere.PointArray(), sphere.TriangleArray());
  viskores::cont::Token token;
  const auto portals =
    qc::DistanceMeshObject(mesh).PrepareForExecution(viskores::cont::DeviceAdapterTagSerial{}, token);
  // Distances to the mesh agree with brute force over its triangles.
  std::mt19937 rng(9);
  std::uniform_real_distribution<double> coordinate(-1.5, 1.5);
  for (int i = 0; i < 500; ++i)
  {
    const Vec3 p(coordinate(rng), coordinate(rng), coordinate(rng));
    double brute = std::numeric_limits<double>::infinity();
    for (const auto& t : sphere.Triangles)
    {
      brute = std::min(brute,
                       qc::PointTriangleDistanceUpper(p,
                                                      sphere.Points[static_cast<std::size_t>(t[0])],
                                                      sphere.Points[static_cast<std::size_t>(t[1])],
                                                      sphere.Points[static_cast<std::size_t>(t[2])]));
    }
    bool complete = true;
    VISKORES_TEST_ASSERT(portals.Distance(p, complete) == brute && complete,
                         "The hierarchy query is not the minimum over all triangles.");
  }
  const qc::CertifiedWithin test{ 0.05, 2.0 };
  auto passes = [&](const Vec3& a, const Vec3& b, const Vec3& c)
  {
    viskores::Id pieces = 0;
    bool complete = true;
    return test(a, b, c, portals, pieces, complete);
  };
  const auto& t0 = sphere.Triangles[0];
  const Vec3 a = sphere.Points[static_cast<std::size_t>(t0[0])];
  const Vec3 b = sphere.Points[static_cast<std::size_t>(t0[1])];
  const Vec3 c = sphere.Points[static_cast<std::size_t>(t0[2])];
  VISKORES_TEST_ASSERT(passes(a, b, c), "A triangle of the mesh lies in its envelope.");
  const Vec3 n = qcd_test::Normalized(a + b + c);
  VISKORES_TEST_ASSERT(passes(a + 0.03 * n, b + 0.03 * n, c + 0.03 * n),
                       "A triangle 0.03 off the mesh lies in its 0.05 envelope.");
  VISKORES_TEST_ASSERT(!passes(a + 0.08 * n, b + 0.08 * n, c + 0.08 * n),
                       "A triangle 0.08 off the mesh is outside its 0.05 envelope.");
  // A chord across the sphere leaves the envelope in its middle, though its ends are on it.
  VISKORES_TEST_ASSERT(!passes(sphere.Points[0], sphere.Points[3], sphere.Points[5]),
                       "A large chord triangle is outside the envelope.");
}

void TestQuadricClusteringDecimationPredicates()
{
  TestDistanceBound();
  TestCertifiedTest();
}
} // anonymous namespace

int UnitTestQuadricClusteringDecimationPredicates(int argc, char* argv[])
{
  return viskores::cont::testing::Testing::Run(TestQuadricClusteringDecimationPredicates, argc, argv);
}
