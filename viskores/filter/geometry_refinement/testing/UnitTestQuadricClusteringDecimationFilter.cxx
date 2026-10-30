//============================================================================
//  The contents of this file are covered by the Viskores license. See
//  LICENSE.txt for details.
//
//  By contributing to this file, all contributors agree to the Developer
//  Certificate of Origin Version 1.1 (DCO 1.1) as stated in DCO.txt.
//============================================================================

#include <viskores/cont/ErrorBadValue.h>
#include <viskores/cont/ErrorExecution.h>
#include <viskores/cont/ErrorFilterExecution.h>
#include <viskores/cont/testing/MakeTestDataSet.h>
#include <viskores/cont/testing/Testing.h>

#include <viskores/filter/geometry_refinement/QuadricClusteringDecimation.h>
#include <viskores/filter/geometry_refinement/testing/QuadricClusteringTestMeshes.h>

namespace
{
using Filter = viskores::filter::geometry_refinement::QuadricClusteringDecimation;

void TestParameters()
{
  std::cout << "Parameters" << std::endl;
  Filter filter;
  VISKORES_TEST_ASSERT(filter.GetRegularization() == 0.02, "Bad default regularization.");
  VISKORES_TEST_ASSERT(filter.GetTargetNumberOfPoints() == 0, "Bad default target.");
  VISKORES_TEST_ASSERT(filter.GetFeatureAngle() == 45, "Bad default feature angle.");
  VISKORES_TEST_ASSERT(filter.GetFeatureWeight() == 1 && filter.GetBoundaryWeight() == 1,
                       "Bad default weights.");
  VISKORES_TEST_ASSERT(filter.GetMaxSweeps() == 0, "Bad default sweep cap.");
  VISKORES_TEST_ASSERT(filter.GetMerge(), "The merge phase should run by default.");
  VISKORES_TEST_ASSERT(filter.GetInitialization() == Filter::Initialization::Curvature,
                       "Bad default initialization.");
  VISKORES_TEST_ASSERT(filter.GetGuarantees() == Filter::Topology, "Bad default guarantees.");

  const viskores::cont::DataSet sphere = qcd_test::Icosphere(1).ToDataSet();
  auto rejects = [&](const std::function<void(Filter&)>& set)
  {
    Filter f;
    f.SetTargetNumberOfPoints(4);
    set(f);
    try
    {
      f.Execute(sphere);
    }
    catch (const viskores::cont::ErrorFilterExecution&)
    {
      return true;
    }
    return false;
  };
  VISKORES_TEST_ASSERT(rejects([](Filter& f) { f.SetRegularization(0); }),
                       "A regularization of zero should be rejected.");
  VISKORES_TEST_ASSERT(rejects([](Filter& f) { f.SetTargetNumberOfPoints(-1); }),
                       "A negative target should be rejected.");
  VISKORES_TEST_ASSERT(rejects([](Filter& f) { f.SetFeatureWeight(-1); }),
                       "A negative weight should be rejected.");
  VISKORES_TEST_ASSERT(rejects([](Filter& f) { f.SetGuarantees(1 << 7); }),
                       "Unknown guarantee bits should be rejected.");
}

void TestNonTriangles()
{
  std::cout << "Cells that are not triangles" << std::endl;
  viskores::cont::testing::MakeTestDataSet maker;
  Filter filter;
  filter.SetTargetNumberOfPoints(4);
  bool threw = false;
  try
  {
    filter.Execute(maker.Make3DExplicitDataSet0());
  }
  catch (const viskores::cont::ErrorExecution&)
  {
    threw = true;
  }
  VISKORES_TEST_ASSERT(threw, "Cells other than triangles should be rejected.");
}

void TestDecimation()
{
  std::cout << "Decimating a sphere with fields" << std::endl;
  const qcd_test::Mesh mesh = qcd_test::Icosphere(3);
  viskores::cont::DataSet input = mesh.ToDataSet();
  std::vector<viskores::Float64> height;
  for (const auto& p : mesh.Points)
  {
    height.push_back(p[2]);
  }
  input.AddPointField("height", height);
  std::vector<viskores::Id> cellIds(mesh.Triangles.size());
  std::iota(cellIds.begin(), cellIds.end(), 0);
  input.AddCellField("cellId", cellIds);

  Filter filter;
  filter.SetTargetNumberOfPoints(60);
  const viskores::cont::DataSet output = filter.Execute(input);

  const viskores::Id k = output.GetNumberOfPoints();
  VISKORES_TEST_ASSERT(k >= 60, "The output has fewer points than the target.");
  viskores::cont::CellSetSingleType<> cells;
  output.GetCellSet().AsCellSet(cells);
  std::vector<viskores::Id3> triangles;
  for (viskores::Id c = 0; c < cells.GetNumberOfCells(); ++c)
  {
    viskores::Id3 ids;
    cells.GetIndices(c, ids);
    triangles.push_back(ids);
  }
  VISKORES_TEST_ASSERT(qcd_test::CheckSurface(k, triangles).empty(), "The output is not a surface.");
  VISKORES_TEST_ASSERT(qcd_test::EulerCharacteristic(k, triangles) == 2,
                       "The output is not a sphere.");
  VISKORES_TEST_ASSERT(
    output.GetCoordinateSystem().GetData().IsType<viskores::cont::ArrayHandle<viskores::Vec3f_64>>(),
    "The output coordinates should be in double precision.");

  // Point fields are averaged over clusters, so they stay within the range of the input.
  viskores::cont::ArrayHandle<viskores::Float64> outHeight;
  output.GetPointField("height").GetData().AsArrayHandle(outHeight);
  VISKORES_TEST_ASSERT(outHeight.GetNumberOfValues() == k, "Wrong size of the point field.");
  auto heights = outHeight.ReadPortal();
  for (viskores::Id i = 0; i < k; ++i)
  {
    VISKORES_TEST_ASSERT(heights.Get(i) >= -1 && heights.Get(i) <= 1, "Bad averaged point field.");
  }
  // Cell fields come from the input triangle that produced every output triangle.
  viskores::cont::ArrayHandle<viskores::Id> outCellIds;
  output.GetCellField("cellId").GetData().AsArrayHandle(outCellIds);
  VISKORES_TEST_ASSERT(outCellIds.GetNumberOfValues() == cells.GetNumberOfCells(),
                       "Wrong size of the cell field.");
  auto ids = outCellIds.ReadPortal();
  for (viskores::Id c = 1; c < outCellIds.GetNumberOfValues(); ++c)
  {
    VISKORES_TEST_ASSERT(ids.Get(c - 1) < ids.Get(c), "Output triangles should keep input order.");
  }
}

void TestQuadricClusteringDecimation()
{
  TestParameters();
  TestNonTriangles();
  TestDecimation();
}
} // anonymous namespace

int UnitTestQuadricClusteringDecimationFilter(int argc, char* argv[])
{
  return viskores::cont::testing::Testing::Run(TestQuadricClusteringDecimation, argc, argv);
}
