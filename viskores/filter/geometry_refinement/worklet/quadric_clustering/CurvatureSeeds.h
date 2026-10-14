//============================================================================
//  The contents of this file are covered by the Viskores license. See
//  LICENSE.txt for details.
//
//  By contributing to this file, all contributors agree to the Developer
//  Certificate of Origin Version 1.1 (DCO 1.1) as stated in DCO.txt.
//============================================================================
#ifndef viskores_worklet_quadric_clustering_CurvatureSeeds_h
#define viskores_worklet_quadric_clustering_CurvatureSeeds_h

#include <viskores/filter/geometry_refinement/worklet/quadric_clustering/Initialization.h>
#include <viskores/filter/geometry_refinement/worklet/quadric_clustering/VertexQuadrics.h>

#include <viskores/cont/Algorithm.h>
#include <viskores/cont/ArrayHandle.h>
#include <viskores/cont/ArrayHandleConstant.h>
#include <viskores/cont/ArrayHandleIndex.h>

#include <viskores/worklet/WorkletMapField.h>

#include <cmath>
#include <limits>
#include <vector>

VISKORES_QUADRIC_CLUSTERING_STRICT_FP_BEGIN

namespace viskores
{
namespace worklet
{
namespace quadric_clustering
{

// The curvature initialisation: seeds spread along the Morton order by a density that follows
// the curvature of the surface against the error bound, so that a flat region gets few clusters
// and a curved one many. The weights are integers, and every quantity that decides a seed is an
// integer or a fixed sequence of IEEE operations, so the seeds are the same on every device and
// thread count.

namespace detail
{

/// The eigenvalues of a symmetric 3x3 matrix in ascending order, by 12 cyclic Jacobi sweeps over
/// the pairs (0, 1), (0, 2), (1, 2), with the operations of the reference implementation
/// (`det.eigh`): only additions, multiplications, divisions and square roots.
VISKORES_EXEC_CONT inline Vec3 JacobiEigenvalues(const Mat3& m)
{
  Real a[3][3] = { { m[0], m[1], m[2] }, { m[3], m[4], m[5] }, { m[6], m[7], m[8] } };
  const int pairs[3][2] = { { 0, 1 }, { 0, 2 }, { 1, 2 } };
  for (int sweep = 0; sweep < 12; ++sweep)
  {
    for (int k = 0; k < 3; ++k)
    {
      const int p = pairs[k][0], q = pairs[k][1], r = 3 - p - q;
      const Real app = a[p][p], aqq = a[q][q], apq = a[p][q], arp = a[r][p], arq = a[r][q];
      Real t = 0;
      if (apq != 0)
      {
        const Real theta = (aqq - app) / (2 * apq);
        if (viskores::Abs(theta) > Real(1e150))
        {
          t = Real(0.5) / theta; // 1 / (2 theta) to full precision, as theta^2 overflows
        }
        else
        {
          t = ((theta >= 0) ? Real(1) : Real(-1)) /
            (viskores::Abs(theta) + viskores::Sqrt(theta * theta + 1));
        }
      }
      const Real c = 1 / viskores::Sqrt(t * t + 1);
      const Real s = t * c;
      a[p][p] = app - t * apq;
      a[q][q] = aqq + t * apq;
      a[p][q] = a[q][p] = 0;
      a[r][p] = a[p][r] = c * arp - s * arq;
      a[r][q] = a[q][r] = s * arp + c * arq;
    }
  }
  // A stable sort of the diagonal.
  Real w[3] = { a[0][0], a[1][1], a[2][2] };
  for (int i = 1; i < 3; ++i)
  {
    const Real x = w[i];
    int j = i;
    while (j > 0 && w[j - 1] > x)
    {
      w[j] = w[j - 1];
      --j;
    }
    w[j] = x;
  }
  return Vec3(w[0], w[1], w[2]);
}

/// The weight `q_v = floor(min(w_v, 1) 2^20)` of a vertex, with `w_v = a_v / (2.6 r_1 r_2)` the
/// share of a cluster it asks for. `r_i = min(sqrt(2 epsilon / kappa_i), size, r_max)` is the
/// radius within which the surface stays within epsilon of its tangent plane along the principal
/// direction `i`, capped by the largest cluster radius, with
/// `kappa_i = sqrt(max(2 lambda_i / a_v, 0)) sqrt(3) / h_v` from the two smallest eigenvalues
/// `lambda_i` of the plain quadric `A_v` and the mean length `h_v` of the edges at `v`, summed in
/// increasing edge order.
struct CurvatureWeight : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn quadric,
                                FieldIn area,
                                FieldIn edgeRange,
                                WholeArrayIn vertexEdges,
                                WholeArrayIn edges,
                                WholeArrayIn points,
                                FieldOut weight);
  using ExecutionSignature = _7(_1, _2, _3, _4, _5, _6);

  VISKORES_CONT CurvatureWeight(Real epsilon, Real cap)
    : Epsilon(epsilon)
    , Cap(cap)
  {
  }

  template <typename IdPortal, typename EdgePortal, typename PointPortal>
  VISKORES_EXEC viskores::UInt64 operator()(const Mat3& quadric,
                                            Real area,
                                            const viskores::Id2& edgeRange,
                                            const IdPortal& vertexEdges,
                                            const EdgePortal& edges,
                                            const PointPortal& points) const
  {
    Real sum = 0;
    for (viskores::Id e = edgeRange[0]; e < edgeRange[1]; ++e)
    {
      const viskores::Id2 edge = edges.Get(vertexEdges.Get(e));
      sum = sum + Norm(Sub(points.Get(edge[1]), points.Get(edge[0])));
    }
    const Real h = sum / Real(edgeRange[1] - edgeRange[0]);
    const Vec3 lambda = JacobiEigenvalues(quadric);
    const Real w =
      area / (Real(2.6) * this->Radius(lambda[0], area, h) * this->Radius(lambda[1], area, h));
    return static_cast<viskores::UInt64>(
      viskores::Floor(viskores::Min(w, Real(1)) * Real(1048576)));
  }

  VISKORES_EXEC Real Radius(Real lambda, Real area, Real h) const
  {
    const Real kappa =
      viskores::Sqrt(viskores::Max(2 * lambda / area, Real(0))) * viskores::Sqrt(Real(3)) / h;
    return viskores::Min(viskores::Sqrt(2 * this->Epsilon / viskores::Max(kappa, Real(1e-300))),
                         this->Cap);
  }

  Real Epsilon;
  Real Cap;
};

} // namespace detail

/// The vertices in Morton order with their curvature weights, and the total weight.
struct CurvatureDensity
{
  viskores::cont::ArrayHandle<viskores::Pair<viskores::UInt64, viskores::Id>> Order;
  viskores::cont::ArrayHandle<viskores::UInt64> Weights;
  viskores::UInt64 Total = 0;
};

/// The curvature weights of the vertices for the error bound `epsilon`, or, if it is 0, for 1%
/// of the largest side of the bounding box: in budget mode only their ratios matter. The radii
/// are capped by the largest cluster radius `maxRadius`.
inline CurvatureDensity ComputeCurvatureDensity(
  const SurfaceMesh& mesh,
  Real epsilon,
  Real maxRadius = std::numeric_limits<Real>::infinity())
{
  viskores::cont::Invoker invoke;
  CurvatureDensity density;
  Real size;
  density.Order = MortonOrder(mesh, size);
  if (epsilon <= 0)
  {
    epsilon = Real(0.01) * size;
  }
  // The plain quadric A_v: without regularisation, feature or boundary terms, whose sums only
  // add zeros.
  QuadricParameters plain;
  plain.Alpha = 0;
  plain.FeatureWeight = 0;
  plain.BoundaryWeight = 0;
  const VertexQuadrics quadrics = ComputeVertexQuadrics(mesh, plain);
  viskores::cont::ArrayHandle<viskores::Id2> edgeRange;
  invoke(detail::Ranges{},
         viskores::cont::ArrayHandleIndex(mesh.NumberOfPoints),
         mesh.VertexEdgeOffsets,
         edgeRange);
  invoke(detail::CurvatureWeight(epsilon, viskores::Min(size, maxRadius)),
         quadrics.M,
         quadrics.Area,
         edgeRange,
         mesh.VertexEdges,
         mesh.Edges,
         mesh.Points,
         density.Weights);
  // Integers: the same sum in any order.
  density.Total = viskores::cont::Algorithm::Reduce(density.Weights, viskores::UInt64(0));
  return density;
}

namespace detail
{

struct IsBoundaryEdge : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn edgeTriangles, FieldOut boundary);
  using ExecutionSignature = _2(_1);

  VISKORES_EXEC viskores::UInt8 operator()(const viskores::Id2& edgeTriangles) const
  {
    return (edgeTriangles[1] < 0) ? 1 : 0;
  }
};

} // namespace detail

/// The topological floor `k_lo` of the output size: the sum over the components of `k_min` of
/// Lemma "Heawood", `ceil((7 + sqrt(49 - 24 chi)) / 2)` for a closed component and
/// `max(3 b, ceil((5 + sqrt(25 - 24 chi)) / 2))` for one with `b` boundary loops.
inline viskores::Id TopologicalFloor(const SurfaceMesh& mesh)
{
  viskores::cont::Invoker invoke;
  const viskores::Id n = mesh.NumberOfPoints;
  const std::size_t numComponents = static_cast<std::size_t>(mesh.NumberOfComponents);
  std::vector<viskores::Id> chi(numComponents, 0), loops(numComponents, 0);
  auto components = mesh.Components.ReadPortal();
  for (viskores::Id v = 0; v < n; ++v)
  {
    ++chi[static_cast<std::size_t>(components.Get(v))];
  }
  auto edges = mesh.Edges.ReadPortal();
  for (viskores::Id e = 0; e < edges.GetNumberOfValues(); ++e)
  {
    --chi[static_cast<std::size_t>(components.Get(edges.Get(e)[0]))];
  }
  auto triangles = mesh.Triangles.ReadPortal();
  for (viskores::Id t = 0; t < mesh.NumberOfTriangles; ++t)
  {
    ++chi[static_cast<std::size_t>(components.Get(triangles.Get(t)[0]))];
  }
  // The boundary loops: the components of the graph of the boundary edges, at boundary
  // vertices.
  viskores::cont::ArrayHandle<viskores::UInt8> isBoundary;
  invoke(detail::IsBoundaryEdge{}, mesh.EdgeTriangles, isBoundary);
  viskores::cont::ArrayHandle<viskores::Id2> boundaryEdges;
  viskores::cont::Algorithm::CopyIf(mesh.Edges, isBoundary, boundaryEdges);
  viskores::cont::ArrayHandle<viskores::Id> loopIds, roots;
  LabelComponents(
    n, boundaryEdges, viskores::cont::ArrayHandleConstant<viskores::Id>(0, n), loopIds, &roots);
  auto rootPortal = roots.ReadPortal();
  auto boundaryVertices = mesh.IsBoundaryVertex.ReadPortal();
  for (viskores::Id v = 0; v < n; ++v)
  {
    if (boundaryVertices.Get(v) && rootPortal.Get(v) == v)
    {
      ++loops[static_cast<std::size_t>(components.Get(v))];
    }
  }
  viskores::Id sum = 0;
  for (std::size_t c = 0; c < numComponents; ++c)
  {
    const Real x = Real(chi[c]);
    if (loops[c] == 0)
    {
      sum += static_cast<viskores::Id>(std::ceil((7 + std::sqrt(49 - 24 * x)) / 2));
    }
    else
    {
      sum += viskores::Max(3 * loops[c],
                           static_cast<viskores::Id>(std::ceil((5 + std::sqrt(25 - 24 * x)) / 2)));
    }
  }
  return sum;
}

/// The number of seeds of error mode: `round(total / 2^20)`, half to even, clamped to
/// `[k_lo, max(k_lo, n / 4)]`.
inline viskores::Id CurvatureClusterCount(const SurfaceMesh& mesh, const CurvatureDensity& density)
{
  viskores::UInt64 k = density.Total >> 20;
  const viskores::UInt64 remainder = density.Total & 0xFFFFFULL;
  if (remainder > 0x80000ULL || (remainder == 0x80000ULL && (k & 1) != 0))
  {
    ++k;
  }
  const viskores::Id low = TopologicalFloor(mesh);
  const viskores::Id high = viskores::Max(low, mesh.NumberOfPoints / 4);
  return viskores::Min(viskores::Max(static_cast<viskores::Id>(k), low), high);
}

namespace detail
{

/// `a b` as a 128-bit integer, high and low words.
inline void Multiply128(viskores::UInt64 a,
                        viskores::UInt64 b,
                        viskores::UInt64& hi,
                        viskores::UInt64& lo)
{
  const viskores::UInt64 mask = 0xFFFFFFFFULL;
  const viskores::UInt64 a0 = a & mask, a1 = a >> 32, b0 = b & mask, b1 = b >> 32;
  const viskores::UInt64 p00 = a0 * b0, p01 = a0 * b1, p10 = a1 * b0, p11 = a1 * b1;
  const viskores::UInt64 middle = (p00 >> 32) + (p01 & mask) + (p10 & mask);
  lo = (middle << 32) | (p00 & mask);
  hi = p11 + (p01 >> 32) + (p10 >> 32) + (middle >> 32);
}

/// Whether `a b >= c d`, exactly.
inline bool ProductAtLeast(viskores::UInt64 a,
                           viskores::UInt64 b,
                           viskores::UInt64 c,
                           viskores::UInt64 d)
{
  viskores::UInt64 hi1, lo1, hi2, lo2;
  Multiply128(a, b, hi1, lo1);
  Multiply128(c, d, hi2, lo2);
  return (hi1 != hi2) ? (hi1 > hi2) : (lo1 >= lo2);
}

} // namespace detail

/// Seeds of the curvature initialisation: seed `j` of `k` is the first vertex in Morton order
/// whose cumulative weight `C` satisfies `2 k C >= (2 j + 1) total`. Seeds can repeat.
inline std::vector<viskores::Id> CurvatureSeeds(const CurvatureDensity& density, viskores::Id k)
{
  auto order = density.Order.ReadPortal();
  auto weights = density.Weights.ReadPortal();
  const viskores::UInt64 twiceK = 2 * static_cast<viskores::UInt64>(k);
  std::vector<viskores::Id> seeds;
  viskores::Id position = 0;
  viskores::UInt64 cumulative = weights.Get(order.Get(0).second);
  for (viskores::Id j = 0; j < k; ++j)
  {
    const viskores::UInt64 odd = 2 * static_cast<viskores::UInt64>(j) + 1;
    // Ends before the last position: (2 j + 1) total < 2 k total.
    while (!detail::ProductAtLeast(twiceK, cumulative, odd, density.Total))
    {
      ++position;
      cumulative += weights.Get(order.Get(position).second);
    }
    seeds.push_back(order.Get(position).second);
  }
  return seeds;
}

/// The curvature seeds grown into connected clusters by `GrowSeedLabels`.
inline viskores::cont::ArrayHandle<viskores::Id> CurvatureSeedLabels(
  const SurfaceMesh& mesh,
  const viskores::cont::ArrayHandle<viskores::Id2>& ringRanges,
  const viskores::cont::ArrayHandle<viskores::Id>& rings,
  const CurvatureDensity& density,
  viskores::Id k)
{
  return GrowSeedLabels(mesh, ringRanges, rings, CurvatureSeeds(density, k));
}

}
}
} // namespace viskores::worklet::quadric_clustering

VISKORES_QUADRIC_CLUSTERING_STRICT_FP_END

#endif // viskores_worklet_quadric_clustering_CurvatureSeeds_h
