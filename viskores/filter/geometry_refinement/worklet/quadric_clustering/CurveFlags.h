//============================================================================
//  The contents of this file are covered by the Viskores license. See
//  LICENSE.txt for details.
//
//  By contributing to this file, all contributors agree to the Developer
//  Certificate of Origin Version 1.1 (DCO 1.1) as stated in DCO.txt.
//============================================================================
#ifndef viskores_worklet_quadric_clustering_CurveFlags_h
#define viskores_worklet_quadric_clustering_CurveFlags_h

#include <viskores/filter/geometry_refinement/worklet/quadric_clustering/Clustering.h>
#include <viskores/filter/geometry_refinement/worklet/quadric_clustering/QuadricClusteringCommonWorklets.h>
#include <viskores/filter/geometry_refinement/worklet/quadric_clustering/SingularCurves.h>

#include <viskores/BinaryOperators.h>
#include <viskores/cont/Algorithm.h>
#include <viskores/cont/ArrayGetValues.h>
#include <viskores/cont/ArrayHandleGroupVec.h>
#include <viskores/cont/ArrayHandleIndex.h>
#include <viskores/cont/ArrayHandleView.h>
#include <viskores/worklet/WorkletMapField.h>

VISKORES_QUADRIC_CLUSTERING_STRICT_FP_BEGIN

namespace viskores
{
namespace worklet
{
namespace quadric_clustering
{

/// The curve test (Fc) of Definition "Curve test" (Proposition "Curves within a tolerance"), as
/// proofs/lib/curves.py: for every arc of a cut input, every edge of its output polyline lies
/// within `eps_c` of the input arc, and every edge of the input arc within `eps_c` of the output
/// polyline, decided by one-sided bounds.
///
/// The input polyline of an arc runs through its end junctions and its vertices; the output
/// polyline through its ends, realised at their junction clusters, and one point per segment
/// (the alive heads, in order), at the generator of its pieces; a closed arc closes back to its
/// first point. Every point has an owner, the segment (its first vertex) or the junction it
/// belongs to; a vertex absorbed into a junction belongs to the junction.
struct CurvePolylines
{
  viskores::Id NumberOfArcs = 0;
  /// Per arc: whether it is closed, and the offsets of its points in each polyline.
  viskores::cont::ArrayHandle<viskores::UInt8> Closed;
  viskores::cont::ArrayHandle<viskores::Id> InputOffsets;
  viskores::cont::ArrayHandle<viskores::Id> OutputOffsets;
  /// The points (a closed polyline without its repeated first point), their owners, and per
  /// point the local index of a point of the other polyline near it, where the search for a
  /// covering edge starts.
  viskores::cont::ArrayHandle<Vec3> InputPoints;
  viskores::cont::ArrayHandle<viskores::Id> InputOwners;
  viskores::cont::ArrayHandle<viskores::Id> InputHints;
  viskores::cont::ArrayHandle<Vec3> OutputPoints;
  viskores::cont::ArrayHandle<viskores::Id> OutputOwners;
  viskores::cont::ArrayHandle<viskores::Id> OutputHints;
  /// Per position of `ArcVertices`: the local index of its output point if it is the head of an
  /// alive segment, or -1.
  viskores::cont::ArrayHandle<viskores::Id> OutputOfPosition;
  /// Per arc: the offsets of its edge tests, the edges of its input polyline first, then those
  /// of its output polyline.
  viskores::cont::ArrayHandle<viskores::Id> ItemOffsets;
  /// The largest absolute coordinate of the input points and the generators, `S`.
  Real Scale = 0;
};

namespace curve_test_detail
{

/// Whether a position of an arc is the head of an alive segment.
struct AliveHead : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn position,
                                FieldIn arc,
                                WholeArrayIn segmentHead,
                                WholeArrayIn arcOffsets,
                                WholeArrayIn absorbed,
                                FieldOut alive);
  using ExecutionSignature = _6(_1, _2, _3, _4, _5);

  template <typename FlagPortal, typename IdPortal, typename PairPortal>
  VISKORES_EXEC viskores::Id operator()(viskores::Id p,
                                        viskores::Id a,
                                        const FlagPortal& segmentHead,
                                        const IdPortal& arcOffsets,
                                        const PairPortal& absorbed) const
  {
    const viskores::Id2 abs = absorbed.Get(a);
    return (segmentHead.Get(p) && p >= arcOffsets.Get(a) + abs[0] &&
            p < arcOffsets.Get(a + 1) - abs[1])
      ? 1
      : 0;
  }
};

/// The number of points of the two polylines of an arc and of their edges.
struct PolylineCounts : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn arc,
                                WholeArrayIn arcOffsets,
                                WholeArrayIn arcEnds,
                                WholeArrayIn headOffsets,
                                FieldOut closed,
                                FieldOut inputCount,
                                FieldOut outputCount,
                                FieldOut items);
  using ExecutionSignature = void(_1, _2, _3, _4, _5, _6, _7, _8);

  template <typename IdPortal, typename PairPortal>
  VISKORES_EXEC void operator()(viskores::Id a,
                                const IdPortal& arcOffsets,
                                const PairPortal& arcEnds,
                                const IdPortal& headOffsets,
                                viskores::UInt8& closed,
                                viskores::Id& inputCount,
                                viskores::Id& outputCount,
                                viskores::Id& items) const
  {
    const bool isClosed = arcEnds.Get(a)[0] < 0;
    closed = isClosed ? 1 : 0;
    const viskores::Id m = arcOffsets.Get(a + 1) - arcOffsets.Get(a);
    const viskores::Id heads =
      headOffsets.Get(arcOffsets.Get(a + 1)) - headOffsets.Get(arcOffsets.Get(a));
    inputCount = isClosed ? m : m + 2;
    outputCount = isClosed ? heads : heads + 2;
    items = isClosed ? inputCount + outputCount : inputCount + outputCount - 2;
  }
};

/// The input point of a position of an arc, with its owner and the output point of its segment
/// (the junction's for an absorbed vertex); the output point of an alive head.
struct PositionPoints : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn position,
                                FieldIn arc,
                                FieldIn alive,
                                FieldIn headOffset,
                                ExecObject curves,
                                WholeArrayIn points,
                                WholeArrayIn labels,
                                WholeArrayIn generators,
                                WholeArrayIn copyOffsets,
                                WholeArrayIn closed,
                                WholeArrayIn inputOffsets,
                                WholeArrayIn outputOffsets,
                                WholeArrayOut inputPoints,
                                WholeArrayOut inputOwners,
                                WholeArrayOut inputHints,
                                WholeArrayOut outputPoints,
                                WholeArrayOut outputOwners,
                                WholeArrayOut outputHints,
                                WholeArrayOut outputOfPosition);
  using ExecutionSignature =
    void(_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19);

  template <typename Curves,
            typename VecPortal,
            typename IdPortal,
            typename FlagPortal,
            typename VecOut,
            typename IdOut>
  VISKORES_EXEC void operator()(viskores::Id p,
                                viskores::Id a,
                                viskores::Id alive,
                                viskores::Id headOffset,
                                const Curves& curves,
                                const VecPortal& points,
                                const IdPortal& labels,
                                const VecPortal& generators,
                                const IdPortal& copyOffsets,
                                const FlagPortal& closed,
                                const IdPortal& inputOffsets,
                                const IdPortal& outputOffsets,
                                VecOut& inputPoints,
                                IdOut& inputOwners,
                                IdOut& inputHints,
                                VecOut& outputPoints,
                                IdOut& outputOwners,
                                IdOut& outputHints,
                                IdOut& outputOfPosition) const
  {
    const bool isClosed = closed.Get(a) != 0;
    const viskores::Id shift = isClosed ? 0 : 1;
    const viskores::Id start = curves.ArcOffsets.Get(a), end = curves.ArcOffsets.Get(a + 1);
    const viskores::Id2 abs = curves.Absorbed.Get(a);
    const viskores::Id v = curves.ArcVertices.Get(p);
    const viskores::Id outputCount = outputOffsets.Get(a + 1) - outputOffsets.Get(a);
    // The alive heads before p, and the output point of p's segment.
    const viskores::Id before = headOffset - curves.HeadOffsets.Get(start);
    viskores::Id hint;
    if (p < start + abs[0])
    {
      hint = 0;
    }
    else if (p >= end - abs[1])
    {
      hint = outputCount - 1;
    }
    else if (before + alive == 0)
    {
      hint = outputCount - 1; // a closed arc's last segment, which wraps around
    }
    else
    {
      hint = shift + before + alive - 1;
    }
    const viskores::Id in = inputOffsets.Get(a) + shift + (p - start);
    inputPoints.Set(in, points.Get(copyOffsets.Get(v)));
    inputOwners.Set(in, curves.Segment.Get(v));
    inputHints.Set(in, hint);
    if (alive)
    {
      const viskores::Id local = shift + before;
      const viskores::Id out = outputOffsets.Get(a) + local;
      outputPoints.Set(out, generators.Get(labels.Get(copyOffsets.Get(v))));
      outputOwners.Set(out, curves.Segment.Get(v));
      outputHints.Set(out, shift + (p - start));
      outputOfPosition.Set(p, local);
    }
  }
};

/// The curves in the execution environment, with the alive-head offsets.
struct CurvesPortals
{
  using IdPortal = viskores::cont::ArrayHandle<viskores::Id>::ReadPortalType;
  using PairPortal = viskores::cont::ArrayHandle<viskores::Id2>::ReadPortalType;
  IdPortal ArcOffsets;
  IdPortal ArcVertices;
  PairPortal Absorbed;
  IdPortal Segment;
  IdPortal HeadOffsets;
};

struct CurvesObject : viskores::cont::ExecutionObjectBase
{
  CurvesObject(const SingularCurves& curves, const viskores::cont::ArrayHandle<viskores::Id>& heads)
    : Curves(&curves)
    , HeadOffsets(heads)
  {
  }

  VISKORES_CONT CurvesPortals PrepareForExecution(viskores::cont::DeviceAdapterId device,
                                                  viskores::cont::Token& token) const
  {
    return CurvesPortals{ this->Curves->ArcOffsets.PrepareForInput(device, token),
                          this->Curves->ArcVertices.PrepareForInput(device, token),
                          this->Curves->Absorbed.PrepareForInput(device, token),
                          this->Curves->Segment.PrepareForInput(device, token),
                          this->HeadOffsets.PrepareForInput(device, token) };
  }

  const SingularCurves* Curves;
  viskores::cont::ArrayHandle<viskores::Id> HeadOffsets;
};

/// The two ends of an open arc in both polylines: its junctions, realised at their clusters.
struct EndPoints : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn arc,
                                WholeArrayIn arcEnds,
                                WholeArrayIn points,
                                WholeArrayIn labels,
                                WholeArrayIn generators,
                                WholeArrayIn copyOffsets,
                                WholeArrayIn inputOffsets,
                                WholeArrayIn outputOffsets,
                                WholeArrayOut inputPoints,
                                WholeArrayOut inputOwners,
                                WholeArrayOut inputHints,
                                WholeArrayOut outputPoints,
                                WholeArrayOut outputOwners,
                                WholeArrayOut outputHints);
  using ExecutionSignature = void(_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14);

  template <typename PairPortal,
            typename VecPortal,
            typename IdPortal,
            typename VecOut,
            typename IdOut>
  VISKORES_EXEC void operator()(viskores::Id a,
                                const PairPortal& arcEnds,
                                const VecPortal& points,
                                const IdPortal& labels,
                                const VecPortal& generators,
                                const IdPortal& copyOffsets,
                                const IdPortal& inputOffsets,
                                const IdPortal& outputOffsets,
                                VecOut& inputPoints,
                                IdOut& inputOwners,
                                IdOut& inputHints,
                                VecOut& outputPoints,
                                IdOut& outputOwners,
                                IdOut& outputHints) const
  {
    const viskores::Id2 ends = arcEnds.Get(a);
    if (ends[0] < 0)
    {
      return;
    }
    const viskores::Id inputLast = inputOffsets.Get(a + 1) - inputOffsets.Get(a) - 1;
    const viskores::Id outputLast = outputOffsets.Get(a + 1) - outputOffsets.Get(a) - 1;
    for (int end = 0; end < 2; ++end)
    {
      const viskores::Id j = ends[end];
      const viskores::Id copy = copyOffsets.Get(j);
      const viskores::Id in = inputOffsets.Get(a) + (end == 0 ? 0 : inputLast);
      const viskores::Id out = outputOffsets.Get(a) + (end == 0 ? 0 : outputLast);
      inputPoints.Set(in, points.Get(copy));
      inputOwners.Set(in, j);
      inputHints.Set(in, end == 0 ? 0 : outputLast);
      outputPoints.Set(out, generators.Get(labels.Get(copy)));
      outputOwners.Set(out, j);
      outputHints.Set(out, end == 0 ? 0 : inputLast);
    }
  }
};

} // namespace curve_test_detail

/// The polylines of every arc on a placed labelling of the cut mesh, as `arc_polylines` of
/// proofs/lib/curves.py. `copyOffsets` gives the first copy of every input vertex.
inline CurvePolylines BuildCurvePolylines(
  const SurfaceMesh& mesh,
  const SingularCurves& curves,
  const viskores::cont::ArrayHandle<viskores::Id>& copyOffsets,
  const Clustering& clusters)
{
  using Algorithm = viskores::cont::Algorithm;
  namespace cd = curve_test_detail;
  viskores::cont::Invoker invoke;
  CurvePolylines polylines;
  const viskores::Id numArcs = curves.NumberOfArcs;
  polylines.NumberOfArcs = numArcs;
  polylines.Scale =
    viskores::Max(MaxAbsCoordinate(mesh.Points), MaxAbsCoordinate(clusters.Generators));
  const viskores::Id numPositions = curves.ArcVertices.GetNumberOfValues();
  viskores::cont::ArrayHandle<viskores::Id> arcOfPosition, alive, headOffsets;
  Algorithm::UpperBounds(viskores::cont::make_ArrayHandleView(curves.ArcOffsets, 1, numArcs),
                         viskores::cont::ArrayHandleIndex(numPositions),
                         arcOfPosition);
  invoke(cd::AliveHead{},
         viskores::cont::ArrayHandleIndex(numPositions),
         arcOfPosition,
         curves.SegmentHead,
         curves.ArcOffsets,
         curves.Absorbed,
         alive);
  Algorithm::ScanExtended(alive, headOffsets);
  viskores::cont::ArrayHandle<viskores::Id> inputCounts, outputCounts, itemCounts;
  invoke(cd::PolylineCounts{},
         viskores::cont::ArrayHandleIndex(numArcs),
         curves.ArcOffsets,
         curves.ArcEnds,
         headOffsets,
         polylines.Closed,
         inputCounts,
         outputCounts,
         itemCounts);
  Algorithm::ScanExtended(inputCounts, polylines.InputOffsets);
  Algorithm::ScanExtended(outputCounts, polylines.OutputOffsets);
  Algorithm::ScanExtended(itemCounts, polylines.ItemOffsets);
  const viskores::Id numInput = viskores::cont::ArrayGetValue(numArcs, polylines.InputOffsets);
  const viskores::Id numOutput = viskores::cont::ArrayGetValue(numArcs, polylines.OutputOffsets);
  polylines.InputPoints.Allocate(numInput);
  polylines.InputOwners.Allocate(numInput);
  polylines.InputHints.Allocate(numInput);
  polylines.OutputPoints.Allocate(numOutput);
  polylines.OutputOwners.Allocate(numOutput);
  polylines.OutputHints.Allocate(numOutput);
  polylines.OutputOfPosition.AllocateAndFill(numPositions, -1);
  invoke(cd::PositionPoints{},
         viskores::cont::ArrayHandleIndex(numPositions),
         arcOfPosition,
         alive,
         viskores::cont::make_ArrayHandleView(headOffsets, 0, numPositions),
         cd::CurvesObject(curves, headOffsets),
         mesh.Points,
         clusters.Labels,
         clusters.Generators,
         copyOffsets,
         polylines.Closed,
         polylines.InputOffsets,
         polylines.OutputOffsets,
         polylines.InputPoints,
         polylines.InputOwners,
         polylines.InputHints,
         polylines.OutputPoints,
         polylines.OutputOwners,
         polylines.OutputHints,
         polylines.OutputOfPosition);
  invoke(cd::EndPoints{},
         viskores::cont::ArrayHandleIndex(numArcs),
         curves.ArcEnds,
         mesh.Points,
         clusters.Labels,
         clusters.Generators,
         copyOffsets,
         polylines.InputOffsets,
         polylines.OutputOffsets,
         polylines.InputPoints,
         polylines.InputOwners,
         polylines.InputHints,
         polylines.OutputPoints,
         polylines.OutputOwners,
         polylines.OutputHints);
  return polylines;
}

/// The results of the curve test on the polylines as they are.
struct CurveTestResult
{
  /// Per edge test: whether its edge is not covered, the owners of its ends if so, and the
  /// edges of the other polyline that accepted its pieces.
  viskores::cont::ArrayHandle<viskores::UInt8> Uncovered;
  viskores::cont::ArrayHandle<viskores::Id2> Owners;
  viskores::cont::ArrayHandle<viskores::Id2> Witnesses;
  viskores::Id NumberOfUncovered = 0;
};

namespace curve_test_detail
{

/// The drift of the computed midpoints at depth `j`, `sqrt3 j u S`, rounded up.
VISKORES_EXEC_CONT inline Real Drift(viskores::IdComponent j, Real scale)
{
  const Real sqrt3Up = NextUp(viskores::Sqrt(Real(3)));
  return NextUp(NextUp(sqrt3Up * Real(j)) * UnitRoundoff * scale);
}

/// The upper bound on the distance from a point to an edge whose computed value is `d` (Lemma
/// "One-sided evaluation" (i)): `(1 + 4u) d + 32 u S`, rounded up.
VISKORES_EXEC_CONT inline Real DistanceUp(Real d, Real scale)
{
  return NextUp(NextUp((1 + 4 * UnitRoundoff) * d) + 32 * UnitRoundoff * scale);
}

/// The computed distance from `p` to the point `a + v (b - a)` of the edge `ab`, with `v` the
/// parameter of the nearest point clamped to [0, 1], and 0 on an edge of length 0.
VISKORES_EXEC_CONT inline Real PointEdgeDistance(const Vec3& p, const Vec3& a, const Vec3& b)
{
  const Vec3 d = Sub(b, a);
  const Real dd = Dot(d, d);
  const Real t = Dot(Sub(p, a), d);
  Real v = (dd > 0) ? t / dd : Real(0);
  v = viskores::Min(viskores::Max(v, Real(0)), Real(1));
  const Vec3 q(a[0] + v * d[0], a[1] + v * d[1], a[2] + v * d[2]);
  return Norm(Sub(p, q));
}

/// Whether the edge `xy` is covered by the polyline `u`, as `edge_covered` of
/// proofs/lib/curves.py: split at computed midpoints to depth `maxDepth`, every piece accepted by
/// the convexity rule (both ends within the bound of one edge) or the midpoint rule. Both rules
/// are monotone in the distance, so a piece is accepted when one edge accepts it, and the edges
/// are searched outward from `hint`. Records the smallest and largest edge that accepted a piece.
template <typename View>
VISKORES_EXEC bool EdgeCovered(const Vec3& x,
                               const Vec3& y,
                               const View& u,
                               viskores::Id hint,
                               Real epsilon,
                               Real scale,
                               viskores::IdComponent maxDepth,
                               viskores::Id& lowWitness,
                               viskores::Id& highWitness)
{
  // Depth-first: at most maxDepth + 1 pieces wait; enough for maxDepth <= 16.
  constexpr viskores::IdComponent StackSize = 2 * 16 + 2;
  const viskores::Id numEdges = u.NumberOfEdges();
  lowWitness = numEdges;
  highWitness = -1;
  if (numEdges <= 0)
  {
    return false;
  }
  hint = viskores::Min(viskores::Max(hint, viskores::Id(0)), numEdges - 1);
  Vec3 from[StackSize], to[StackSize];
  viskores::IdComponent depth[StackSize];
  from[0] = x;
  to[0] = y;
  depth[0] = 0;
  viskores::IdComponent top = 1;
  while (top > 0)
  {
    --top;
    const Vec3 p0 = from[top], p1 = to[top];
    const viskores::IdComponent j = depth[top];
    const Real drift = Drift(j, scale);
    const Real driftMid = Drift(j + 1, scale);
    const Vec3 c(
      (p0[0] + p1[0]) * Real(0.5), (p0[1] + p1[1]) * Real(0.5), (p0[2] + p1[2]) * Real(0.5));
    const Real half = LengthUp(Norm(Sub(p1, p0))) * Real(0.5);
    bool accepted = false;
    // The edges outward from the hint: hint, hint + 1, hint - 1, hint + 2, ...
    for (viskores::Id k = 0; !accepted && k < 2 * numEdges; ++k)
    {
      const viskores::Id e = (k % 2 == 0) ? hint + k / 2 : hint - (k + 1) / 2;
      if (e < 0 || e >= numEdges)
      {
        continue;
      }
      const Vec3 a = u.Point(e), b = u.Point(e + 1);
      const Real d0 = DistanceUp(PointEdgeDistance(p0, a, b), scale);
      const Real d1 = DistanceUp(PointEdgeDistance(p1, a, b), scale);
      accepted = NextUp(viskores::Max(d0, d1) + drift) <= epsilon;
      if (!accepted)
      {
        const Real dc = DistanceUp(PointEdgeDistance(c, a, b), scale);
        accepted = NextUp(NextUp(dc + half) + driftMid) <= epsilon;
      }
      if (accepted)
      {
        lowWitness = viskores::Min(lowWitness, e);
        highWitness = viskores::Max(highWitness, e);
      }
    }
    if (accepted)
    {
      continue;
    }
    if (j >= maxDepth)
    {
      return false;
    }
    from[top] = c;
    to[top] = p1;
    depth[top] = j + 1;
    from[top + 1] = p0;
    to[top + 1] = c;
    depth[top + 1] = j + 1;
    top += 2;
  }
  return true;
}

/// One edge test of the polylines as they are: a local item below the number of input edges
/// tests an input edge against the output polyline, the others an output edge against the
/// input arc. Writes whether the edge is not covered, the owners of its ends, and the smallest
/// and largest edge of the other polyline that accepted a piece.
struct BaseEdgeTest : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn item,
                                ExecObject polylines,
                                FieldOut uncovered,
                                FieldOut owners,
                                FieldOut witnesses);
  using ExecutionSignature = void(_1, _2, _3, _4, _5);

  VISKORES_CONT BaseEdgeTest(Real epsilon, Real scale, viskores::IdComponent depth)
    : Epsilon(epsilon)
    , Scale(scale)
    , Depth(depth)
  {
  }

  template <typename Polylines>
  VISKORES_EXEC void operator()(viskores::Id item,
                                const Polylines& polylines,
                                viskores::UInt8& uncovered,
                                viskores::Id2& owners,
                                viskores::Id2& witnesses) const
  {
    viskores::Id local;
    const viskores::Id a = polylines.ArcOfItem(item, local);
    const auto input = polylines.Input(a);
    const auto output = polylines.Output(a);
    const viskores::Id inputEdges = input.NumberOfEdges();
    const bool fromInput = local < inputEdges;
    const auto& x = fromInput ? input : output;
    const viskores::Id e = fromInput ? local : local - inputEdges;
    const viskores::Id b0 = x.Base(e), b1 = x.Base(e + 1);
    const auto& hints = fromInput ? polylines.InputHints : polylines.OutputHints;
    const auto& owned = fromInput ? polylines.InputOwners : polylines.OutputOwners;
    const viskores::Id hint = hints.Get(x.Start + b0);
    const bool covered = EdgeCovered(x.Point(e),
                                     x.Point(e + 1),
                                     fromInput ? output : input,
                                     hint,
                                     this->Epsilon,
                                     this->Scale,
                                     this->Depth,
                                     witnesses[0],
                                     witnesses[1]);
    uncovered = covered ? 0 : 1;
    owners =
      covered ? viskores::Id2(-1) : viskores::Id2(owned.Get(x.Start + b0), owned.Get(x.Start + b1));
  }

  Real Epsilon;
  Real Scale;
  viskores::IdComponent Depth;
};

/// A polyline of one arc, possibly with one point removed and one moved: the polyline of a
/// candidate merge. Indices are local to the arc.
template <typename VecPortal>
struct PolylineView
{
  VISKORES_EXEC viskores::Id Size() const { return this->Count - (this->Removed >= 0 ? 1 : 0); }

  VISKORES_EXEC viskores::Id NumberOfEdges() const
  {
    return this->Closed ? this->Size() : this->Size() - 1;
  }

  /// The index in the view of a point of the polyline; the removed point gives its successor.
  VISKORES_EXEC viskores::Id View(viskores::Id base) const
  {
    return (this->Removed >= 0 && base > this->Removed) ? base - 1 : base;
  }

  /// The local index of a point of the view in the polyline.
  VISKORES_EXEC viskores::Id Base(viskores::Id i) const
  {
    if (this->Closed && i == this->Size())
    {
      i = 0;
    }
    return (this->Removed >= 0 && i >= this->Removed) ? i + 1 : i;
  }

  VISKORES_EXEC Vec3 Point(viskores::Id i) const
  {
    const viskores::Id b = this->Base(i);
    return (b == this->Moved) ? this->MovedPoint : this->Points.Get(this->Start + b);
  }

  VecPortal Points;
  viskores::Id Start = 0;
  viskores::Id Count = 0;
  bool Closed = false;
  viskores::Id Removed = -1;
  viskores::Id Moved = -1;
  Vec3 MovedPoint;
};

/// The polylines in the execution environment.
struct PolylinesPortals
{
  using IdPortal = viskores::cont::ArrayHandle<viskores::Id>::ReadPortalType;
  using VecPortal = viskores::cont::ArrayHandle<Vec3>::ReadPortalType;
  using FlagPortal = viskores::cont::ArrayHandle<viskores::UInt8>::ReadPortalType;

  /// The arc of a global item and its local item.
  VISKORES_EXEC viskores::Id ArcOfItem(viskores::Id item, viskores::Id& local) const
  {
    const viskores::Id lo = OwnerOf(this->ItemOffsets, item);
    local = item - this->ItemOffsets.Get(lo);
    return lo;
  }

  VISKORES_EXEC PolylineView<VecPortal> Input(viskores::Id a) const
  {
    PolylineView<VecPortal> view;
    view.Points = this->InputPoints;
    view.Start = this->InputOffsets.Get(a);
    view.Count = this->InputOffsets.Get(a + 1) - view.Start;
    view.Closed = this->Closed.Get(a) != 0;
    return view;
  }

  VISKORES_EXEC PolylineView<VecPortal> Output(viskores::Id a) const
  {
    PolylineView<VecPortal> view;
    view.Points = this->OutputPoints;
    view.Start = this->OutputOffsets.Get(a);
    view.Count = this->OutputOffsets.Get(a + 1) - view.Start;
    view.Closed = this->Closed.Get(a) != 0;
    return view;
  }

  FlagPortal Closed;
  IdPortal InputOffsets;
  IdPortal OutputOffsets;
  VecPortal InputPoints;
  IdPortal InputOwners;
  IdPortal InputHints;
  VecPortal OutputPoints;
  IdPortal OutputOwners;
  IdPortal OutputHints;
  IdPortal OutputOfPosition;
  IdPortal ItemOffsets;
};

struct PolylinesObject : viskores::cont::ExecutionObjectBase
{
  explicit PolylinesObject(const CurvePolylines& polylines)
    : Polylines(&polylines)
  {
  }

  VISKORES_CONT PolylinesPortals PrepareForExecution(viskores::cont::DeviceAdapterId device,
                                                     viskores::cont::Token& token) const
  {
    const CurvePolylines& p = *this->Polylines;
    return PolylinesPortals{ p.Closed.PrepareForInput(device, token),
                             p.InputOffsets.PrepareForInput(device, token),
                             p.OutputOffsets.PrepareForInput(device, token),
                             p.InputPoints.PrepareForInput(device, token),
                             p.InputOwners.PrepareForInput(device, token),
                             p.InputHints.PrepareForInput(device, token),
                             p.OutputPoints.PrepareForInput(device, token),
                             p.OutputOwners.PrepareForInput(device, token),
                             p.OutputHints.PrepareForInput(device, token),
                             p.OutputOfPosition.PrepareForInput(device, token),
                             p.ItemOffsets.PrepareForInput(device, token) };
  }

  const CurvePolylines* Polylines;
};

} // namespace curve_test_detail

/// The curve test on the polylines as they are, with tolerance `epsilon` and subdivision depth
/// `depth`.
inline CurveTestResult CurveTest(const CurvePolylines& polylines,
                                 Real epsilon,
                                 viskores::IdComponent depth)
{
  viskores::cont::Invoker invoke;
  CurveTestResult result;
  const viskores::Id numItems =
    viskores::cont::ArrayGetValue(polylines.NumberOfArcs, polylines.ItemOffsets);
  invoke(curve_test_detail::BaseEdgeTest(epsilon, polylines.Scale, depth),
         viskores::cont::ArrayHandleIndex(numItems),
         curve_test_detail::PolylinesObject(polylines),
         result.Uncovered,
         result.Owners,
         result.Witnesses);
  result.NumberOfUncovered = viskores::cont::Algorithm::Reduce(result.Uncovered, viskores::Id(0));
  return result;
}

/// Whether the curve test fails on a placed labelling.
inline bool CurveTestFails(const SurfaceMesh& mesh,
                           const SingularCurves& curves,
                           const viskores::cont::ArrayHandle<viskores::Id>& original,
                           const Clustering& clusters,
                           Real epsilon,
                           viskores::IdComponent depth)
{
  if (curves.NumberOfArcs == 0)
  {
    return false;
  }
  const CurvePolylines polylines = BuildCurvePolylines(
    mesh, curves, FirstCopies(original, curves.Segment.GetNumberOfValues()), clusters);
  return CurveTest(polylines, epsilon, depth).NumberOfUncovered > 0;
}

namespace curve_test_detail
{

/// Marks the clusters with at least two vertices that hold a frozen copy of a vertex whose
/// owner is named; every write is a 1.
struct FlagOwned : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn original,
                                FieldIn label,
                                WholeArrayIn segment,
                                WholeArrayIn named,
                                WholeArrayIn sizes,
                                WholeArrayOut flags);
  using ExecutionSignature = void(_1, _2, _3, _4, _5, _6);

  template <typename IdPortal, typename FlagPortal>
  VISKORES_EXEC void operator()(viskores::Id original,
                                viskores::Id label,
                                const IdPortal& segment,
                                const IdPortal& named,
                                const IdPortal& sizes,
                                FlagPortal& flags) const
  {
    const viskores::Id owner = segment.Get(original);
    if (owner < 0 || sizes.Get(label) < 2 || FindSorted(named, owner) < 0)
    {
      return;
    }
    flags.Set(label, 1);
  }
};

} // namespace curve_test_detail

/// The clusters the curve test names on a placed labelling, as `ArcVQC.flagged_clusters`: for
/// every edge it does not cover, the owners of its two ends; flags the clusters with at least
/// two vertices that hold a frozen copy of a vertex of one, the pieces of a segment or the
/// junction clusters of a junction. Returns the number of edges not covered.
inline viskores::Id CurveFlags(const SurfaceMesh& mesh,
                               const SingularCurves& curves,
                               const viskores::cont::ArrayHandle<viskores::Id>& original,
                               const Clustering& clusters,
                               Real epsilon,
                               viskores::IdComponent depth,
                               viskores::cont::ArrayHandle<viskores::UInt8>& flags)
{
  using Algorithm = viskores::cont::Algorithm;
  if (curves.NumberOfArcs == 0)
  {
    return 0;
  }
  const CurvePolylines polylines = BuildCurvePolylines(
    mesh, curves, FirstCopies(original, curves.Segment.GetNumberOfValues()), clusters);
  const CurveTestResult result = CurveTest(polylines, epsilon, depth);
  if (result.NumberOfUncovered == 0)
  {
    return 0;
  }
  viskores::cont::ArrayHandle<viskores::Id2> owners;
  Algorithm::CopyIf(result.Owners, result.Uncovered, owners);
  viskores::cont::ArrayHandle<viskores::Id> named;
  named.Allocate(2 * owners.GetNumberOfValues());
  auto grouped = viskores::cont::make_ArrayHandleGroupVec<2>(named);
  Algorithm::Copy(owners, grouped);
  Algorithm::Sort(named);
  Algorithm::Unique(named);
  viskores::cont::Invoker invoke;
  invoke(curve_test_detail::FlagOwned{},
         original,
         clusters.Labels,
         curves.Segment,
         named,
         clusters.Sizes,
         flags);
  return result.NumberOfUncovered;
}

/// Per candidate merge of segments: its arc, the position of the head of the segment whose
/// point it removes from the output polyline, and that of the segment whose point it moves, or
/// -1.
struct CurveCandidates
{
  viskores::cont::ArrayHandle<viskores::Id> Arc;
  viskores::cont::ArrayHandle<viskores::Id> Removed;
  viskores::cont::ArrayHandle<viskores::Id> Moved;
};

namespace curve_test_detail
{

struct ItemArc : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn item, ExecObject polylines, FieldOut arc);
  using ExecutionSignature = _3(_1, _2);

  template <typename Polylines>
  VISKORES_EXEC viskores::Id operator()(viskores::Id item, const Polylines& polylines) const
  {
    viskores::Id local;
    return polylines.ArcOfItem(item, local);
  }
};

/// A candidate that leaves the scale fires if an arc other than its own fails as it is.
struct OtherArcsFail : viskores::worklet::WorkletMapField
{
  using ControlSignature =
    void(FieldIn candidate, FieldIn arc, FieldIn scale, WholeArrayIn arcFails, WholeArrayOut fires);
  using ExecutionSignature = void(_1, _2, _3, _4, _5);

  VISKORES_CONT OtherArcsFail(Real baseScale, viskores::Id failingArcs)
    : BaseScale(baseScale)
    , FailingArcs(failingArcs)
  {
  }

  template <typename IdPortal, typename FirePortal>
  VISKORES_EXEC void operator()(viskores::Id c,
                                viskores::Id a,
                                Real scale,
                                const IdPortal& arcFails,
                                FirePortal& fires) const
  {
    if (scale == this->BaseScale && this->FailingArcs - (arcFails.Get(a) > 0 ? 1 : 0) > 0)
    {
      fires.Set(c, 1);
    }
  }

  Real BaseScale;
  viskores::Id FailingArcs;
};

/// The number of edge tests of a candidate: those of its arc, or of every arc if it changes
/// the scale.
struct CandidateCurveCount : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn arc, FieldIn scale, ExecObject polylines, FieldOut count);
  using ExecutionSignature = _4(_1, _2, _3);

  VISKORES_CONT explicit CandidateCurveCount(Real baseScale)
    : BaseScale(baseScale)
  {
  }

  template <typename Polylines>
  VISKORES_EXEC viskores::Id operator()(viskores::Id a,
                                        Real scale,
                                        const Polylines& polylines) const
  {
    if (scale != this->BaseScale)
    {
      return polylines.ItemOffsets.Get(polylines.ItemOffsets.GetNumberOfValues() - 1);
    }
    return polylines.ItemOffsets.Get(a + 1) - polylines.ItemOffsets.Get(a);
  }

  Real BaseScale;
};

/// The polylines of a candidate merge of segments: its arc's output polyline without the point
/// of the segment it removes, and with the point of the segment it moves at its new generator.
struct CandidateChange
{
  viskores::Id Arc = -1;
  viskores::Id Removed = -1;
  viskores::Id Moved = -1;
  Vec3 MovedPoint;
  /// Whether the candidate changes the scale `S`: then every edge is tested again.
  bool ChangesScale = false;
};

template <typename Polylines, typename IdPortal, typename VecPortal, typename RealPortal>
VISKORES_EXEC CandidateChange ChangeOf(viskores::Id c,
                                       const Polylines& polylines,
                                       const IdPortal& arcs,
                                       const IdPortal& removed,
                                       const IdPortal& moved,
                                       const VecPortal& movedPoints,
                                       const RealPortal& scales,
                                       Real baseScale)
{
  CandidateChange change;
  change.Arc = arcs.Get(c);
  change.Removed = polylines.OutputOfPosition.Get(removed.Get(c));
  const viskores::Id m = moved.Get(c);
  change.Moved = (m >= 0) ? polylines.OutputOfPosition.Get(m) : viskores::Id(-1);
  change.MovedPoint = movedPoints.Get(c);
  change.ChangesScale = scales.Get(c) != baseScale;
  return change;
}

/// Whether an edge at `point` of a polyline with `count` points is in the range of witnesses
/// `w`, the smallest and largest edge that accepted a piece.
VISKORES_EXEC inline bool WitnessAtPoint(const viskores::Id2& w,
                                         viskores::Id point,
                                         viskores::Id count,
                                         bool closed)
{
  if (point < 0)
  {
    return false;
  }
  const viskores::Id before = closed ? (point + count - 1) % count : point - 1;
  return (point >= w[0] && point <= w[1]) || (before >= w[0] && before <= w[1]);
}

/// One edge test of a candidate. A test the change cannot affect, with the scale unchanged,
/// takes the verdict of the polylines as they are: an output edge between two unchanged
/// consecutive points, or an input edge none of whose pieces an edge at a changed point
/// accepted.
struct CandidateEdgeTest : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn item,
                                WholeArrayIn itemOffsets,
                                WholeArrayIn arcs,
                                WholeArrayIn removed,
                                WholeArrayIn moved,
                                WholeArrayIn movedPoints,
                                WholeArrayIn scales,
                                ExecObject polylines,
                                WholeArrayIn baseUncovered,
                                WholeArrayIn baseWitnesses,
                                WholeArrayOut fires);
  using ExecutionSignature = void(_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11);

  VISKORES_CONT CandidateEdgeTest(Real epsilon, Real baseScale, viskores::IdComponent depth)
    : Epsilon(epsilon)
    , BaseScale(baseScale)
    , Depth(depth)
  {
  }

  template <typename IdPortal,
            typename VecPortal,
            typename RealPortal,
            typename Polylines,
            typename FlagPortal,
            typename PairPortal,
            typename FirePortal>
  VISKORES_EXEC void operator()(viskores::Id item,
                                const IdPortal& itemOffsets,
                                const IdPortal& arcs,
                                const IdPortal& removed,
                                const IdPortal& moved,
                                const VecPortal& movedPoints,
                                const RealPortal& scales,
                                const Polylines& polylines,
                                const FlagPortal& baseUncovered,
                                const PairPortal& baseWitnesses,
                                FirePortal& fires) const
  {
    const viskores::Id c = OwnerOf(itemOffsets, item);
    const CandidateChange change =
      ChangeOf(c, polylines, arcs, removed, moved, movedPoints, scales, this->BaseScale);
    const viskores::Id first = change.ChangesScale ? 0 : polylines.ItemOffsets.Get(change.Arc);
    const viskores::Id global = first + (item - itemOffsets.Get(c));
    viskores::Id local;
    const viskores::Id b = polylines.ArcOfItem(global, local);
    const Real scale = scales.Get(c);
    const auto input = polylines.Input(b);
    auto output = polylines.Output(b);
    const viskores::Id outputCount = output.Count;
    if (b == change.Arc)
    {
      output.Removed = change.Removed;
      output.Moved = change.Moved;
      output.MovedPoint = change.MovedPoint;
    }
    const viskores::Id inputEdges = input.NumberOfEdges();
    viskores::Id low, high;
    bool covered;
    if (local < inputEdges)
    {
      const viskores::Id e = local;
      if (b == change.Arc && !change.ChangesScale && !baseUncovered.Get(global))
      {
        const viskores::Id2 w = baseWitnesses.Get(global);
        if (!WitnessAtPoint(w, change.Removed, outputCount, output.Closed) &&
            !WitnessAtPoint(w, change.Moved, outputCount, output.Closed))
        {
          return;
        }
      }
      const viskores::Id hint = output.View(polylines.InputHints.Get(input.Start + input.Base(e)));
      covered = EdgeCovered(input.Point(e),
                            input.Point(e + 1),
                            output,
                            hint,
                            this->Epsilon,
                            scale,
                            this->Depth,
                            low,
                            high);
    }
    else
    {
      const viskores::Id e = local - inputEdges;
      if (e >= output.NumberOfEdges())
      {
        return;
      }
      const viskores::Id b0 = output.Base(e), b1 = output.Base(e + 1);
      if (b == change.Arc && !change.ChangesScale && b0 != change.Moved && b1 != change.Moved &&
          (b1 == b0 + 1 || (output.Closed && b1 == 0 && b0 == outputCount - 1)))
      {
        // An edge of the polyline as it is.
        if (baseUncovered.Get(polylines.ItemOffsets.Get(b) + inputEdges + b0))
        {
          fires.Set(c, 1);
        }
        return;
      }
      const viskores::Id hint = polylines.OutputHints.Get(output.Start + b0);
      covered = EdgeCovered(output.Point(e),
                            output.Point(e + 1),
                            input,
                            hint,
                            this->Epsilon,
                            scale,
                            this->Depth,
                            low,
                            high);
    }
    if (!covered)
    {
      fires.Set(c, 1);
    }
  }

  Real Epsilon;
  Real BaseScale;
  viskores::IdComponent Depth;
};

} // namespace curve_test_detail

/// The curve test on the labelling of every candidate merge of segments, given the test on the
/// labelling of the round: per candidate, `candidates`, the new point, and its scale `S`. Sets
/// `fires` (one per candidate) where the test fails.
inline void CandidateCurveFires(const CurvePolylines& polylines,
                                const CurveTestResult& base,
                                const CurveCandidates& candidates,
                                const viskores::cont::ArrayHandle<Vec3>& movedPoints,
                                const viskores::cont::ArrayHandle<Real>& scales,
                                Real epsilon,
                                viskores::IdComponent depth,
                                viskores::cont::ArrayHandle<viskores::UInt8>& fires)
{
  using Algorithm = viskores::cont::Algorithm;
  namespace cd = curve_test_detail;
  viskores::cont::Invoker invoke;
  const viskores::Id n = candidates.Arc.GetNumberOfValues();
  if (n == 0)
  {
    return;
  }
  const cd::PolylinesObject object(polylines);
  // The arcs that fail as they are.
  const viskores::Id numItems = base.Uncovered.GetNumberOfValues();
  viskores::cont::ArrayHandle<viskores::Id> arcFails;
  arcFails.AllocateAndFill(polylines.NumberOfArcs, 0);
  viskores::Id failingArcs = 0;
  if (base.NumberOfUncovered > 0)
  {
    viskores::cont::ArrayHandle<viskores::Id> itemArcs, keys, counts, uncovered;
    invoke(cd::ItemArc{}, viskores::cont::ArrayHandleIndex(numItems), object, itemArcs);
    Algorithm::Copy(base.Uncovered, uncovered);
    Algorithm::ReduceByKey(itemArcs, uncovered, keys, counts, viskores::Add());
    invoke(ScatterValue{}, keys, counts, arcFails);
    viskores::cont::ArrayHandle<viskores::Id> failing;
    Algorithm::CopyIf(counts, counts, failing, IsPositive{});
    failingArcs = failing.GetNumberOfValues();
  }
  invoke(cd::OtherArcsFail(polylines.Scale, failingArcs),
         viskores::cont::ArrayHandleIndex(n),
         candidates.Arc,
         scales,
         arcFails,
         fires);
  viskores::cont::ArrayHandle<viskores::Id> counts, offsets;
  invoke(cd::CandidateCurveCount(polylines.Scale), candidates.Arc, scales, object, counts);
  Algorithm::ScanExtended(counts, offsets);
  const viskores::Id total = viskores::cont::ArrayGetValue(n, offsets);
  invoke(cd::CandidateEdgeTest(epsilon, polylines.Scale, depth),
         viskores::cont::ArrayHandleIndex(total),
         offsets,
         candidates.Arc,
         candidates.Removed,
         candidates.Moved,
         movedPoints,
         scales,
         object,
         base.Uncovered,
         base.Witnesses,
         fires);
}

}
}
} // namespace viskores::worklet::quadric_clustering

VISKORES_QUADRIC_CLUSTERING_STRICT_FP_END

#endif // viskores_worklet_quadric_clustering_CurveFlags_h
