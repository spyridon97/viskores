//============================================================================
//  The contents of this file are covered by the Viskores license. See
//  LICENSE.txt for details.
//
//  By contributing to this file, all contributors agree to the Developer
//  Certificate of Origin Version 1.1 (DCO 1.1) as stated in DCO.txt.
//============================================================================

// Tests of the predicates behind the embedding flag (F4) of the quadric clustering
// decimation: the classification of triangle pairs by their shared labels. The exact
// predicates and the bounding volume hierarchy they use have their own tests.

#include <viskores/cont/Invoker.h>
#include <viskores/cont/testing/Testing.h>

#include <viskores/filter/geometry_refinement/worklet/quadric_clustering/ImproperPairs.h>

#include <array>
#include <set>
#include <vector>

namespace
{
namespace exact = viskores::worklet::quadric_clustering::exact;
using Vec3 = viskores::Vec3f_64;

void TestTrianglePairs()
{
  std::cout << "Classification of triangle pairs" << std::endl;
  auto improper = [](const std::array<Vec3, 3>& p,
                     const viskores::Id3& pl,
                     const std::array<Vec3, 3>& q,
                     const viskores::Id3& ql)
  { return exact::ImproperPair(p.data(), pl, q.data(), ql); };
  const viskores::Id3 abc(0, 1, 2), def(3, 4, 5);
  // No shared label.
  VISKORES_TEST_ASSERT(!improper({ Vec3(0, 0, 0), Vec3(1, 0, 0), Vec3(0, 1, 0) },
                                 abc,
                                 { Vec3(0, 0, 1), Vec3(1, 0, 1), Vec3(0, 1, 1) },
                                 def),
                       "Parallel triangles do not meet.");
  VISKORES_TEST_ASSERT(improper({ Vec3(0, 0, 0), Vec3(1, 0, 0), Vec3(0, 1, 0) },
                                abc,
                                { Vec3(0.2, 0.2, -1), Vec3(0.2, 0.2, 1), Vec3(3, 3, 0) },
                                def),
                       "Crossing triangles meet.");
  VISKORES_TEST_ASSERT(improper({ Vec3(0, 0, 0), Vec3(1, 0, 0), Vec3(0, 1, 0) },
                                abc,
                                { Vec3(1, 0, 0), Vec3(2, 0, 1), Vec3(2, 0, -1) },
                                def),
                       "Triangles touching at a point without sharing it meet improperly.");
  VISKORES_TEST_ASSERT(improper({ Vec3(0, 0, 0), Vec3(1, 0, 0), Vec3(0, 1, 0) },
                                abc,
                                { Vec3(0.25, 0.25, 0), Vec3(2, 0.25, 0), Vec3(0.25, 2, 0) },
                                def),
                       "Overlapping coplanar triangles meet.");
  // One shared label, at the origin.
  const viskores::Id3 shared1(0, 4, 5);
  VISKORES_TEST_ASSERT(!improper({ Vec3(0, 0, 0), Vec3(1, 0, 0), Vec3(0, 1, 0) },
                                 abc,
                                 { Vec3(0, 0, 0), Vec3(-1, 0, 0), Vec3(0, -1, 0) },
                                 shared1),
                       "Coplanar triangles meeting only at their shared vertex are proper.");
  VISKORES_TEST_ASSERT(improper({ Vec3(0, 0, 0), Vec3(1, 0, 0), Vec3(0, 1, 0) },
                                abc,
                                { Vec3(0, 0, 0), Vec3(1, 1, 0), Vec3(-1, 2, 0) },
                                shared1),
                       "Coplanar triangles overlapping beyond their shared vertex meet improperly.");
  VISKORES_TEST_ASSERT(improper({ Vec3(0, 0, 0), Vec3(1, 0, 0), Vec3(0, 1, 0) },
                                abc,
                                { Vec3(0, 0, 0), Vec3(2, 0, 0), Vec3(1, -1, 0) },
                                shared1),
                       "Triangles sharing a vertex and a stretch of an edge meet improperly.");
  VISKORES_TEST_ASSERT(!improper({ Vec3(0, 0, 0), Vec3(1, 0, 0), Vec3(0, 1, 0) },
                                 abc,
                                 { Vec3(0, 0, 0), Vec3(1, 1, 1), Vec3(-1, 1, 1) },
                                 shared1),
                       "Triangles sharing a vertex and leaving the plane are proper.");
  VISKORES_TEST_ASSERT(improper({ Vec3(0, 0, 0), Vec3(1, 0, 0), Vec3(0, 1, 0) },
                                abc,
                                { Vec3(0, 0, 0), Vec3(0.3, 0.3, 1), Vec3(0.3, 0.3, -1) },
                                shared1),
                       "A triangle through another one from a shared vertex meets it improperly.");
  // Two shared labels: the edge from the origin to (1, 0, 0).
  const viskores::Id3 shared2(0, 1, 5);
  VISKORES_TEST_ASSERT(!improper({ Vec3(0, 0, 0), Vec3(1, 0, 0), Vec3(0, 1, 0) },
                                 abc,
                                 { Vec3(0, 0, 0), Vec3(1, 0, 0), Vec3(0, -1, 0) },
                                 shared2),
                       "An unfolded pair across an edge is proper.");
  VISKORES_TEST_ASSERT(improper({ Vec3(0, 0, 0), Vec3(1, 0, 0), Vec3(0, 1, 0) },
                                abc,
                                { Vec3(0, 0, 0), Vec3(1, 0, 0), Vec3(0.5, 2, 0) },
                                shared2),
                       "A folded coplanar pair across an edge is improper.");
  const double u = std::ldexp(1.0, -52);
  VISKORES_TEST_ASSERT(!improper({ Vec3(0, 0, 0), Vec3(1, 0, 0), Vec3(0, 1, 0) },
                                 abc,
                                 { Vec3(0, 0, 0), Vec3(1, 0, 0), Vec3(0.5, 2, u) },
                                 shared2),
                       "A fold a few ulps out of the plane meets only in the edge.");
}

void TestQuadricClusteringDecimationPredicates()
{
  TestTrianglePairs();
}
} // anonymous namespace

int UnitTestQuadricClusteringDecimationPredicates(int argc, char* argv[])
{
  return viskores::cont::testing::Testing::Run(TestQuadricClusteringDecimationPredicates, argc, argv);
}
