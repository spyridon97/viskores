//============================================================================
//  The contents of this file are covered by the Viskores license. See
//  LICENSE.txt for details.
//
//  By contributing to this file, all contributors agree to the Developer
//  Certificate of Origin Version 1.1 (DCO 1.1) as stated in DCO.txt.
//============================================================================

#ifndef viskores_filter_geometry_refinement_QuadricClusteringDecimation_h
#define viskores_filter_geometry_refinement_QuadricClusteringDecimation_h

#include <viskores/filter/Filter.h>
#include <viskores/filter/geometry_refinement/viskores_filter_geometry_refinement_export.h>

namespace viskores
{
namespace filter
{
namespace geometry_refinement
{
/// @brief Decimates a triangle mesh by guaranteed variational quadric clustering.
///
/// The input points are partitioned into connected clusters, and every cluster becomes one
/// output point, placed where it minimises the cluster's quadric error plus a regularisation
/// term. A refinement loop splits clusters until none breaks a requirement, and a merge phase
/// then merges adjacent clusters as long as every requirement still holds. The output is a
/// deterministic function of the input as numbered: it does not depend on the device or on the
/// number of threads.
///
/// The options come in four groups. What the user asks for: an error bound, a target number of
/// points, or both. The guarantees, as bits of
/// `SetGuarantees`: by default the output is a surface of the input's type. Choices that
/// change only the output size: the initialisation and the terms of the cost. How the work is
/// done: the merge phase and the sweep cap.
///
/// Point fields are averaged over every cluster, and every output cell takes the cell fields
/// of the first input triangle that produced it. The output coordinates are in double
/// precision.
///
/// Glossary. A *sweep* is one parallel pass in which every point may move to a neighbouring
/// cluster. An *assignment step* repeats sweeps until nothing moves, or `MaxSweeps` times. A
/// *Lloyd iteration*, up to 100 right after seeding, is an assignment step and the placement of
/// the output points. A *round* is an assignment step, the placement, the checks of the
/// guarantees, and the splitting of the clusters that fail them. A *merge round* merges one
/// parallel batch of adjacent clusters.
class VISKORES_FILTER_GEOMETRY_REFINEMENT_EXPORT QuadricClusteringDecimation
  : public viskores::filter::Filter
{
public:
  /// @brief The bound on the Hausdorff distance between input and output, or 0 for none.
  ///
  /// Certifies that every point of the output is within this distance of the input and every
  /// point of the input within this distance of the output, or one of the two with the error
  /// bits of `SetGuarantees`. Setting it turns on the envelope checks of those bits, both by
  /// default, which are the most expensive checks. Use it whenever the error must be
  /// guaranteed. The default is 0.
  VISKORES_CONT void SetErrorBound(viskores::Float64 bound) { this->ErrorBound = bound; }
  /// @copydoc SetErrorBound
  VISKORES_CONT viskores::Float64 GetErrorBound() const { return this->ErrorBound; }

  /// @brief The number of output points to aim for, or 0 for none.
  ///
  /// The output has at least as many points as the target and as the topology of the input
  /// needs, and exactly the target when the merges allow. If an error bound is set too, the
  /// error bound wins: the output stops above the target rather than exceed the error. Use it
  /// for a fixed size, such as levels of detail or transmission. The default is 0.
  VISKORES_CONT void SetTargetNumberOfPoints(viskores::Id target)
  {
    this->TargetNumberOfPoints = target;
  }
  /// @copydoc SetTargetNumberOfPoints
  VISKORES_CONT viskores::Id GetTargetNumberOfPoints() const { return this->TargetNumberOfPoints; }

  /// The guarantees, as bits of `SetGuarantees`.
  enum Guarantee : viskores::UInt32
  {
    /// The output is a surface of the input's type: the same Euler characteristic and boundary
    /// loops, consistently oriented. The input must be a consistently oriented surface. It
    /// cannot be turned off.
    Topology = 1 << 0,
    /// Every point of the output is within the error bound of the input. It takes effect only
    /// with an error bound. Alone, the side that WMTK and PQ-Free HD certify, it was about twice
    /// as fast in the measurements of the paper, but the input then strayed from the output by
    /// up to ten times the bound, at boundaries.
    ErrorOutputToInput = 1 << 1,
    /// Every point of the input is within the error bound of the output. It takes effect only
    /// with an error bound. Alone it was 2 to 4 times slower than both sides, and the output
    /// strayed from the input by up to 1.7 times the bound.
    ErrorInputToOutput = 1 << 2,
    /// Both sides of the error bound: the Hausdorff distance between input and output is at
    /// most the bound.
    TwoSidedError = ErrorOutputToInput | ErrorInputToOutput
  };

  /// @brief The guarantees of the output, as a combination of `Guarantee` bits.
  ///
  /// The default is `Topology | TwoSidedError`. A
  /// combination that breaks a requirement of a bit is rejected when the filter runs.
  VISKORES_CONT void SetGuarantees(viskores::UInt32 guarantees) { this->Guarantees = guarantees; }
  /// @copydoc SetGuarantees
  VISKORES_CONT viskores::UInt32 GetGuarantees() const { return this->Guarantees; }

  /// The initial clusters, before the refinement loop.
  enum class Initialization
  {
    /// One cluster per connected component.
    Single,
    /// Seeds evenly spaced by point count along the Morton order.
    Morton,
    /// Seeds spread along the Morton order by a density that follows the curvature of the
    /// surface against the error bound.
    Curvature
  };

  /// @brief The initial clusters, before the refinement loop.
  ///
  /// `Curvature` places more seeds where the surface bends relative to the error bound, so the
  /// loop starts near its final size: as many as the target, or in error mode as many as the
  /// curvature asks for. Seeds are followed by Lloyd iterations. Use the others for comparison.
  /// No guarantee depends on it. The default is `Curvature`.
  VISKORES_CONT void SetInitialization(Initialization initialization)
  {
    this->InitialClusters = initialization;
  }
  /// @copydoc SetInitialization
  VISKORES_CONT Initialization GetInitialization() const { return this->InitialClusters; }

  /// @brief The regularisation factor `alpha`; the weight of a point is `alpha` times its area.
  ///
  /// It keeps the output points near their clusters on flat regions. Must be positive. Values
  /// from 0.005 to 0.02 gave the smallest outputs. No guarantee depends on it. The default is
  /// 0.02.
  VISKORES_CONT void SetRegularization(viskores::Float64 alpha) { this->Regularization = alpha; }
  /// @copydoc SetRegularization
  VISKORES_CONT viskores::Float64 GetRegularization() const { return this->Regularization; }

  /// @brief The angle, in degrees, between the normals of two triangles above which their
  /// common edge is sharp.
  ///
  /// The sharp-edge and corner terms of the cost pull output points onto creases. No guarantee
  /// depends on it. The default is 45.
  VISKORES_CONT void SetFeatureAngle(viskores::Float64 degrees) { this->FeatureAngle = degrees; }
  /// @copydoc SetFeatureAngle
  VISKORES_CONT viskores::Float64 GetFeatureAngle() const { return this->FeatureAngle; }

  /// @brief The weight of the sharp-edge and corner terms of the cost; 0 leaves them out.
  ///
  /// Use it for inputs with sharp edges and corners. No guarantee depends on it. The default
  /// is 1.
  VISKORES_CONT void SetFeatureWeight(viskores::Float64 weight) { this->FeatureWeight = weight; }
  /// @copydoc SetFeatureWeight
  VISKORES_CONT viskores::Float64 GetFeatureWeight() const { return this->FeatureWeight; }

  /// @brief The weight of the boundary terms of the cost; 0 leaves them out.
  ///
  /// They pull output points onto the boundary of an input with boundary. No guarantee depends
  /// on it. The default is 1.
  VISKORES_CONT void SetBoundaryWeight(viskores::Float64 weight) { this->BoundaryWeight = weight; }
  /// @copydoc SetBoundaryWeight
  VISKORES_CONT viskores::Float64 GetBoundaryWeight() const { return this->BoundaryWeight; }

  /// @brief Whether adjacent clusters are merged after the refinement while every guarantee
  /// still holds.
  ///
  /// Needed to reach the target number of points; with an error bound alone it removed 24 to
  /// 53% of the points in the measurements of the paper. Turn it off only to save time. The
  /// default is true.
  VISKORES_CONT void SetMerge(bool merge) { this->Merge = merge; }
  /// @copydoc SetMerge
  VISKORES_CONT bool GetMerge() const { return this->Merge; }

  /// @brief A cap on the sweeps of every assignment step, or 0 for none.
  ///
  /// It bounds the depth of a round, for example on a GPU. Every guarantee holds for any cap,
  /// but a small cap can stop at a larger output. The default is 0.
  VISKORES_CONT void SetMaxSweeps(viskores::Id sweeps) { this->MaxSweeps = sweeps; }
  /// @copydoc SetMaxSweeps
  VISKORES_CONT viskores::Id GetMaxSweeps() const { return this->MaxSweeps; }

private:
  VISKORES_CONT viskores::cont::DataSet DoExecute(const viskores::cont::DataSet& input) override;

  viskores::Float64 ErrorBound = 0;
  viskores::Id TargetNumberOfPoints = 0;
  viskores::UInt32 Guarantees = Topology | TwoSidedError;
  Initialization InitialClusters = Initialization::Curvature;
  viskores::Float64 Regularization = 0.02;
  viskores::Float64 FeatureAngle = 45;
  viskores::Float64 FeatureWeight = 1;
  viskores::Float64 BoundaryWeight = 1;
  bool Merge = true;
  viskores::Id MaxSweeps = 0;
};
} // namespace geometry_refinement
} // namespace filter
} // namespace viskores

#endif // viskores_filter_geometry_refinement_QuadricClusteringDecimation_h
