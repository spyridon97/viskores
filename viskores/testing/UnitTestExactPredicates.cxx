//============================================================================
//  The contents of this file are covered by the Viskores license. See
//  LICENSE.txt for details.
//
//  By contributing to this file, all contributors agree to the Developer
//  Certificate of Origin Version 1.1 (DCO 1.1) as stated in DCO.txt.
//============================================================================

#include <viskores/ExactPredicates.h>

#include <viskores/cont/ArrayHandle.h>
#include <viskores/cont/Invoker.h>
#include <viskores/cont/testing/Testing.h>
#include <viskores/worklet/WorkletMapField.h>

#include <cmath>

namespace
{
using Vec2 = viskores::Vec<viskores::Float64, 2>;
using Vec3 = viskores::Vec3f_64;

void TestOrient2D()
{
  std::cout << "Orient2D near a line" << std::endl;
  // After Kettner et al., "Classroom examples of robustness problems in geometric computations":
  // q and r lie on the line y = x, so orient2d(p, q, r) = 12 (p_y - p_x) for p near (0.5, 0.5),
  // whose sign plain floating point gets wrong for many of these points.
  const double u = std::ldexp(1.0, -53);
  int wrongInFloatingPoint = 0;
  for (int i = 0; i < 64; ++i)
  {
    for (int j = 0; j < 64; ++j)
    {
      const Vec2 p(0.5 + i * u, 0.5 + j * u);
      const viskores::IdComponent expected = (j > i) - (j < i);
      VISKORES_TEST_ASSERT(viskores::Orient2D(p, Vec2(12, 12), Vec2(24, 24)) == expected,
                           "Orient2D has the wrong sign.");
      const double naive = (p[0] - 24) * (12 - 24) - (p[1] - 24) * (12 - 24);
      wrongInFloatingPoint += ((naive > 0) - (naive < 0)) != expected;
    }
  }
  std::cout << "  plain floating point was wrong on " << wrongInFloatingPoint << " of 4096 points"
            << std::endl;
  VISKORES_TEST_ASSERT(wrongInFloatingPoint > 0, "The test points should be hard.");
}

void TestOrient3D()
{
  std::cout << "Orient3D near a plane" << std::endl;
  // q, r, s lie on the plane z = x, so orient3d(p, q, r, s) has the sign of +-(p_z - p_x).
  const Vec3 q(12, 0, 12), r(24, 0, 24), s(12, 12, 12);
  const viskores::IdComponent far = viskores::Orient3D(Vec3(0, 0, 1), q, r, s);
  VISKORES_TEST_ASSERT(far != 0, "A point off the plane is not coplanar.");
  const double u = std::ldexp(1.0, -53);
  for (int i = 0; i < 48; ++i)
  {
    for (int j = 0; j < 48; ++j)
    {
      const Vec3 p(0.5 + i * u, 0.25 + (i + j) * u, 0.5 + j * u);
      VISKORES_TEST_ASSERT(viskores::Orient3D(p, q, r, s) == far * ((j > i) - (j < i)),
                           "Orient3D has the wrong sign.");
    }
  }
  // Exactly coplanar and collinear points whose differences are not representable.
  const Vec3 a(0.1, 0.2, 0.3), b(0.7, 0.2, 0.3), c(0.1, 0.9, 0.3), d(0.4, 0.55, 0.3);
  VISKORES_TEST_ASSERT(viskores::Orient3D(a, b, c, d) == 0, "Coplanar points give zero.");
  VISKORES_TEST_ASSERT(viskores::TriangleIsDegenerate(a, Vec3(0.3, 0.2, 0.3), b),
                       "Collinear points make a degenerate triangle.");
  VISKORES_TEST_ASSERT(!viskores::TriangleIsDegenerate(a, b, c), "A proper triangle.");
}

void TestIntersections()
{
  std::cout << "Segment-triangle and triangle-triangle intersections" << std::endl;
  const Vec3 t0(0, 0, 0), t1(1, 0, 0), t2(0, 1, 0);
  VISKORES_TEST_ASSERT(viskores::SegmentIntersectsTriangle(Vec3(0.2, 0.2, -1), Vec3(0.2, 0.2, 1), t0, t1, t2),
                       "A segment through the triangle.");
  VISKORES_TEST_ASSERT(!viskores::SegmentIntersectsTriangle(Vec3(0.8, 0.8, -1), Vec3(0.8, 0.8, 1), t0, t1, t2),
                       "A segment beside the triangle.");
  VISKORES_TEST_ASSERT(viskores::SegmentIntersectsTriangle(Vec3(0.5, 0.5, -1), Vec3(0.5, 0.5, 1), t0, t1, t2),
                       "A segment through the hypotenuse touches the closed triangle.");
  VISKORES_TEST_ASSERT(viskores::SegmentIntersectsTriangle(Vec3(-1, 0.5, 0), Vec3(2, 0.5, 0), t0, t1, t2),
                       "A coplanar segment across the triangle.");
  VISKORES_TEST_ASSERT(!viskores::SegmentIntersectsTriangle(Vec3(-1, 2, 0), Vec3(2, 2, 0), t0, t1, t2),
                       "A coplanar segment beside the triangle.");
  VISKORES_TEST_ASSERT(viskores::SegmentIntersectsTriangle(Vec3(1, 0, 0), Vec3(2, 0, 0), t0, t1, t2),
                       "A coplanar segment touching a vertex.");
  VISKORES_TEST_ASSERT(viskores::SegmentIntersectsTriangle(Vec3(1, 0, 0), Vec3(1, 0, 1), t0, t1, t2),
                       "A segment from a vertex out of the plane touches it.");
  VISKORES_TEST_ASSERT(
    !viskores::TrianglesIntersect(t0, t1, t2, Vec3(0, 0, 1), Vec3(1, 0, 1), Vec3(0, 1, 1)),
    "Parallel triangles do not meet.");
  VISKORES_TEST_ASSERT(
    viskores::TrianglesIntersect(t0, t1, t2, Vec3(0.2, 0.2, -1), Vec3(0.2, 0.2, 1), Vec3(3, 3, 0)),
    "Crossing triangles meet.");
  VISKORES_TEST_ASSERT(
    viskores::TrianglesIntersect(t0, t1, t2, Vec3(1, 0, 0), Vec3(2, 0, 1), Vec3(2, 0, -1)),
    "Triangles touching at a point meet.");
  VISKORES_TEST_ASSERT(
    viskores::TrianglesIntersect(t0, t1, t2, Vec3(0.25, 0.25, 0), Vec3(2, 0.25, 0), Vec3(0.25, 2, 0)),
    "Overlapping coplanar triangles meet.");
  const double u = std::ldexp(1.0, -52);
  VISKORES_TEST_ASSERT(
    !viskores::TrianglesIntersect(t0, t1, t2, Vec3(1 + u, 0, 0), Vec3(2, 0, 1), Vec3(2, 0, -1)),
    "Triangles an ulp apart do not meet.");
}

struct Orient3DWorklet : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn p, FieldOut sign);
  using ExecutionSignature = _2(_1);
  VISKORES_EXEC viskores::IdComponent operator()(const Vec3& p) const
  {
    return viskores::Orient3D(p, Vec3(12, 0, 12), Vec3(24, 0, 24), Vec3(12, 12, 12));
  }
};

void TestOnDevice()
{
  std::cout << "Exact predicates in a worklet" << std::endl;
  const double u = std::ldexp(1.0, -53);
  std::vector<Vec3> points;
  for (int i = 0; i < 16; ++i)
  {
    for (int j = 0; j < 16; ++j)
    {
      points.emplace_back(0.5 + i * u, 0.25 + (i + j) * u, 0.5 + j * u);
    }
  }
  viskores::cont::ArrayHandle<viskores::IdComponent> signs;
  viskores::cont::Invoker invoke;
  invoke(Orient3DWorklet{}, viskores::cont::make_ArrayHandle(points, viskores::CopyFlag::On), signs);
  auto portal = signs.ReadPortal();
  for (std::size_t k = 0; k < points.size(); ++k)
  {
    VISKORES_TEST_ASSERT(portal.Get(static_cast<viskores::Id>(k)) ==
                           viskores::Orient3D(points[k], Vec3(12, 0, 12), Vec3(24, 0, 24), Vec3(12, 12, 12)),
                         "Exact predicates differ on the device.");
  }
}

void TestExactPredicates()
{
  TestOrient2D();
  TestOrient3D();
  TestIntersections();
  TestOnDevice();
}
} // anonymous namespace

int UnitTestExactPredicates(int argc, char* argv[])
{
  return viskores::cont::testing::Testing::Run(TestExactPredicates, argc, argv);
}
