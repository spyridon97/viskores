//============================================================================
//  The contents of this file are covered by the Viskores license. See
//  LICENSE.txt for details.
//
//  By contributing to this file, all contributors agree to the Developer
//  Certificate of Origin Version 1.1 (DCO 1.1) as stated in DCO.txt.
//============================================================================
#ifndef viskores_filter_geometry_refinement_testing_QuadricClusteringTestMeshes_h
#define viskores_filter_geometry_refinement_testing_QuadricClusteringTestMeshes_h

// Test meshes for the quadric clustering tests, ported from the meshes of the reference
// implementation of the paper (proofs/lib/meshes.py), and checks of outputs written
// independently of the algorithm. The icosphere and the voxel solids are built with the same
// operations as the reference, so they agree with it bit for bit.

#include <viskores/Types.h>
#include <viskores/cont/ArrayHandle.h>
#include <viskores/cont/DataSetBuilderExplicit.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <map>
#include <numeric>
#include <set>
#include <string>
#include <vector>

#if defined(__clang__)
#pragma float_control(push)
#pragma clang fp contract(off)
#endif

namespace qcd_test
{
using Vec3 = viskores::Vec3f_64;

struct Mesh
{
  std::vector<Vec3> Points;
  std::vector<viskores::Id3> Triangles;

  viskores::cont::ArrayHandle<Vec3> PointArray() const
  {
    return viskores::cont::make_ArrayHandle(this->Points, viskores::CopyFlag::On);
  }
  viskores::cont::ArrayHandle<viskores::Id3> TriangleArray() const
  {
    return viskores::cont::make_ArrayHandle(this->Triangles, viskores::CopyFlag::On);
  }
  viskores::cont::DataSet ToDataSet() const
  {
    std::vector<viskores::UInt8> shapes(this->Triangles.size(), viskores::CELL_SHAPE_TRIANGLE);
    std::vector<viskores::IdComponent> numIndices(this->Triangles.size(), 3);
    std::vector<viskores::Id> connectivity;
    for (const auto& t : this->Triangles)
    {
      connectivity.insert(connectivity.end(), { t[0], t[1], t[2] });
    }
    return viskores::cont::DataSetBuilderExplicit::Create(
      this->Points, shapes, numIndices, connectivity, "coordinates");
  }
};

/// The values of an array handle.
template <typename T>
inline std::vector<T> ToVector(const viskores::cont::ArrayHandle<T>& array)
{
  std::vector<T> result(static_cast<std::size_t>(array.GetNumberOfValues()));
  auto portal = array.ReadPortal();
  for (viskores::Id i = 0; i < array.GetNumberOfValues(); ++i)
  {
    result[static_cast<std::size_t>(i)] = portal.Get(i);
  }
  return result;
}

} // namespace qcd_test

#if defined(__clang__)
#pragma float_control(pop)
#endif

#endif // viskores_filter_geometry_refinement_testing_QuadricClusteringTestMeshes_h
