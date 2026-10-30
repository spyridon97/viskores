//============================================================================
//  The contents of this file are covered by the Viskores license. See
//  LICENSE.txt for details.
//
//  By contributing to this file, all contributors agree to the Developer
//  Certificate of Origin Version 1.1 (DCO 1.1) as stated in DCO.txt.
//============================================================================

#include <viskores/cont/ArrayCopy.h>
#include <viskores/cont/CellSetSingleType.h>
#include <viskores/cont/ErrorFilterExecution.h>
#include <viskores/filter/MapFieldMergeAverage.h>
#include <viskores/filter/MapFieldPermutation.h>
#include <viskores/filter/geometry_refinement/QuadricClusteringDecimation.h>
#include <viskores/filter/geometry_refinement/worklet/QuadricClusteringDecimation.h>
#include <viskores/worklet/WorkletMapTopology.h>

#include <string>

namespace
{
/// The point ids of every cell, which must be a triangle.
struct ExtractTriangles : viskores::worklet::WorkletVisitCellsWithPoints
{
  using ControlSignature = void(CellSetIn cellSet, FieldOutCell triangle);
  using ExecutionSignature = void(CellShape, PointCount, PointIndices, _2);
  using InputDomain = _1;

  template <typename CellShapeTag, typename PointIndices>
  VISKORES_EXEC void operator()(CellShapeTag shape,
                                viskores::IdComponent numPoints,
                                const PointIndices& ids,
                                viskores::Id3& triangle) const
  {
    if (numPoints != 3 ||
        (shape.Id != viskores::CELL_SHAPE_TRIANGLE && shape.Id != viskores::CELL_SHAPE_POLYGON))
    {
      this->RaiseError("QuadricClusteringDecimation requires a mesh of triangles.");
      return;
    }
    triangle = viskores::Id3(ids[0], ids[1], ids[2]);
  }
};

VISKORES_CONT bool DoMapField(viskores::cont::DataSet& result,
                              const viskores::cont::Field& field,
                              const viskores::worklet::Keys<viskores::Id>& pointKeys,
                              const viskores::worklet::QuadricClusteringDecimation& worklet)
{
  if (field.IsPointField())
  {
    return viskores::filter::MapFieldMergeAverage(field, pointKeys, result);
  }
  else if (field.IsCellField())
  {
    return viskores::filter::MapFieldPermutation(field, worklet.GetOutputCellSources(), result);
  }
  else if (field.IsWholeDataSetField())
  {
    result.AddField(field);
    return true;
  }
  return false;
}
} // anonymous namespace

namespace viskores
{
namespace filter
{
namespace geometry_refinement
{
VISKORES_CONT viskores::cont::DataSet QuadricClusteringDecimation::DoExecute(
  const viskores::cont::DataSet& input)
{
  auto reject = [](const char* message)
  {
    throw viskores::cont::ErrorFilterExecution(std::string("QuadricClusteringDecimation: ") +
                                               message);
  };
  if (!(this->ErrorBound >= 0))
  {
    reject("the error bound must not be negative.");
  }
  if (this->TargetNumberOfPoints < 0)
  {
    reject("the target number of points must not be negative.");
  }
  const viskores::UInt32 guarantees = this->Guarantees;
  const viskores::UInt32 all = Topology | TwoSidedError | NoSelfIntersections;
  if ((guarantees & ~all) != 0)
  {
    reject("unknown guarantee bits.");
  }
  if (!(guarantees & Topology))
  {
    reject("the Topology guarantee cannot be turned off.");
  }
  if (this->ErrorBound > 0 && !(guarantees & TwoSidedError))
  {
    reject("an error bound needs ErrorOutputToInput, ErrorInputToOutput or both.");
  }
  if (!(this->Regularization > 0))
  {
    reject("the regularization must be positive.");
  }
  if (!(this->FeatureAngle >= 0 && this->FeatureAngle <= 180))
  {
    reject("the feature angle must be between 0 and 180 degrees.");
  }
  if (!(this->FeatureWeight >= 0) || !(this->BoundaryWeight >= 0))
  {
    reject("the feature and boundary weights must not be negative.");
  }

  viskores::cont::ArrayHandle<viskores::Id3> triangles;
  this->Invoke(ExtractTriangles{},
               input.GetCellSet().ResetCellSetList<VISKORES_DEFAULT_CELL_SET_LIST_UNSTRUCTURED>(),
               triangles);
  viskores::cont::ArrayHandle<viskores::Vec3f_64> points;
  viskores::cont::ArrayCopy(input.GetCoordinateSystem().GetData(), points);

  viskores::worklet::QuadricClusteringDecimation worklet;
  viskores::worklet::QuadricClusteringDecimation::Parameters parameters;
  parameters.ErrorBound = this->ErrorBound;
  parameters.TargetNumberOfPoints = this->TargetNumberOfPoints;
  parameters.Embedding = (guarantees & NoSelfIntersections) != 0;
  using Sides = viskores::worklet::quadric_clustering::EnvelopeSides;
  const viskores::UInt32 sides = guarantees & TwoSidedError;
  parameters.ErrorSides = (sides == ErrorOutputToInput) ? Sides::OutputToInput
    : (sides == ErrorInputToOutput)                     ? Sides::InputToOutput
                                                        : Sides::Both;
  using WorkletInitialization = viskores::worklet::QuadricClusteringDecimation::Initialization;
  switch (this->InitialClusters)
  {
    case Initialization::Single:
      parameters.InitialClusters = WorkletInitialization::Single;
      break;
    case Initialization::Morton:
      parameters.InitialClusters = WorkletInitialization::Morton;
      break;
    case Initialization::Curvature:
      parameters.InitialClusters = WorkletInitialization::Curvature;
      break;
  }
  parameters.Quadric.Alpha = this->Regularization;
  parameters.Quadric.FeatureAngle = this->FeatureAngle;
  parameters.Quadric.FeatureWeight = this->FeatureWeight;
  parameters.Quadric.BoundaryWeight = this->BoundaryWeight;
  parameters.Merge = this->Merge;
  parameters.MaxSweeps = this->MaxSweeps;
  worklet.SetParameters(parameters);
  worklet.Run(points, triangles);

  const viskores::Id numPoints = worklet.GetOutputPoints().GetNumberOfValues();
  viskores::cont::UnknownCellSet outCellSet;
  viskores::cont::CellSetSingleType<> triangleCells;
  triangleCells.Fill(
    numPoints, viskores::CELL_SHAPE_TRIANGLE, 3, worklet.GetOutputCellConnectivity());
  outCellSet = triangleCells;
  // Every output point averages the input points mapped to it.
  viskores::worklet::Keys<viskores::Id> pointKeys;
  pointKeys.BuildArrays(worklet.GetOutputPointMap(), viskores::worklet::KeysSortType::Stable);

  auto mapper = [&](auto& result, const auto& field)
  { DoMapField(result, field, pointKeys, worklet); };
  return this->CreateResultCoordinateSystem(input,
                                            outCellSet,
                                            input.GetCoordinateSystem().GetName(),
                                            worklet.GetOutputPoints(),
                                            mapper);
}
} // namespace geometry_refinement
} // namespace filter
} // namespace viskores
