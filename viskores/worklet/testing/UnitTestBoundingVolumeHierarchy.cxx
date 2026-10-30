//============================================================================
//  The contents of this file are covered by the Viskores license. See
//  LICENSE.txt for details.
//
//  By contributing to this file, all contributors agree to the Developer
//  Certificate of Origin Version 1.1 (DCO 1.1) as stated in DCO.txt.
//============================================================================

#include <viskores/worklet/spatialstructure/BoundingVolumeHierarchy.h>

#include <viskores/cont/Invoker.h>
#include <viskores/cont/testing/Testing.h>
#include <viskores/worklet/WorkletMapField.h>

#include <random>
#include <set>
#include <vector>

namespace
{
using Vec3 = viskores::Vec3f_64;
using BVH = viskores::worklet::spatialstructure::BoundingVolumeHierarchy;

/// For every box, the number of boxes it overlaps, and the distance from its corner `lower` to
/// the nearest centre of another box, also with a cutoff of one.
struct QueryWorklet : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn lower,
                                FieldIn upper,
                                WholeArrayIn centres,
                                ExecObject bvh,
                                FieldOut overlaps,
                                FieldOut nearest,
                                FieldOut nearestUpToOne);
  using ExecutionSignature = void(WorkIndex, _1, _2, _3, _4, _5, _6, _7);

  template <typename CentresPortal, typename Hierarchy>
  VISKORES_EXEC void operator()(viskores::Id index,
                                const Vec3& lower,
                                const Vec3& upper,
                                const CentresPortal& centres,
                                const Hierarchy& bvh,
                                viskores::Id& overlaps,
                                viskores::Float64& nearest,
                                viskores::Float64& nearestUpToOne) const
  {
    overlaps = 0;
    bool complete = bvh.ForEachOverlap(lower,
                                       upper,
                                       [&](viskores::Id)
                                       {
                                         ++overlaps;
                                         return true;
                                       });
    // The distance to a centre is not below the distance to its box: a valid query.
    auto distance = [&](viskores::Id j)
    {
      return j == index ? std::numeric_limits<viskores::Float64>::infinity()
                        : viskores::Magnitude(centres.Get(j) - lower);
    };
    nearest = bvh.FindNearest(lower, distance, complete);
    bool done = true;
    nearestUpToOne = bvh.FindNearest(lower, distance, done, 1.0);
    complete = complete && done;
    if (!complete)
    {
      this->RaiseError("Stack overflow.");
    }
  }
};

void TestBoundingVolumeHierarchy()
{
  std::mt19937 rng(11);
  std::uniform_real_distribution<double> position(0, 10), size(0, 1.5);
  for (int trial = 0; trial < 6; ++trial)
  {
    const int n = 1 + trial * trial * 40;
    std::cout << "Hierarchy over " << n << " boxes and some copies" << std::endl;
    std::vector<Vec3> lower, upper, centres;
    for (int i = 0; i < n; ++i)
    {
      // Degenerate boxes, flat boxes and duplicated boxes too.
      const Vec3 lo(position(rng), position(rng), (trial % 2) ? 0.0 : position(rng));
      const Vec3 hi = (i % 7 == 0) ? lo : lo + Vec3(size(rng), size(rng), size(rng));
      for (int copy = 0; copy < ((i % 11 == 0) ? 2 : 1); ++copy)
      {
        lower.push_back(lo);
        upper.push_back(hi);
        centres.push_back((lo + hi) * 0.5);
      }
    }
    const BVH bvh(viskores::cont::make_ArrayHandle(lower, viskores::CopyFlag::On),
                  viskores::cont::make_ArrayHandle(upper, viskores::CopyFlag::On));
    VISKORES_TEST_ASSERT(bvh.GetNumberOfPrimitives() == static_cast<viskores::Id>(lower.size()),
                         "Wrong number of primitives.");
    viskores::cont::ArrayHandle<viskores::Id> overlaps;
    viskores::cont::ArrayHandle<viskores::Float64> nearest, nearestUpToOne;
    viskores::cont::Invoker invoke;
    invoke(QueryWorklet{},
           viskores::cont::make_ArrayHandle(lower, viskores::CopyFlag::On),
           viskores::cont::make_ArrayHandle(upper, viskores::CopyFlag::On),
           viskores::cont::make_ArrayHandle(centres, viskores::CopyFlag::On),
           bvh,
           overlaps,
           nearest,
           nearestUpToOne);
    auto overlapPortal = overlaps.ReadPortal();
    auto nearestPortal = nearest.ReadPortal();
    auto upToOnePortal = nearestUpToOne.ReadPortal();
    for (std::size_t i = 0; i < lower.size(); ++i)
    {
      viskores::Id expected = 0;
      viskores::Float64 closest = std::numeric_limits<viskores::Float64>::infinity();
      for (std::size_t j = 0; j < lower.size(); ++j)
      {
        bool overlap = true;
        for (int a = 0; a < 3; ++a)
        {
          overlap = overlap && lower[j][a] <= upper[i][a] && upper[j][a] >= lower[i][a];
        }
        expected += overlap ? 1 : 0;
        if (j != i)
        {
          closest = std::min(closest, viskores::Magnitude(centres[j] - lower[i]));
        }
      }
      VISKORES_TEST_ASSERT(overlapPortal.Get(static_cast<viskores::Id>(i)) == expected,
                           "The hierarchy missed or invented an overlap.");
      VISKORES_TEST_ASSERT(nearestPortal.Get(static_cast<viskores::Id>(i)) == closest,
                           "The nearest query is not the exact minimum.");
      const viskores::Float64 upToOne = upToOnePortal.Get(static_cast<viskores::Id>(i));
      VISKORES_TEST_ASSERT(closest <= 1 ? upToOne == closest : upToOne > 1,
                           "The nearest query with a cutoff is wrong.");
    }
  }
}
} // anonymous namespace

int UnitTestBoundingVolumeHierarchy(int argc, char* argv[])
{
  return viskores::cont::testing::Testing::Run(TestBoundingVolumeHierarchy, argc, argv);
}
