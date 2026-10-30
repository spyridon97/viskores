//============================================================================
//  The contents of this file are covered by the Viskores license. See
//  LICENSE.txt for details.
//
//  By contributing to this file, all contributors agree to the Developer
//  Certificate of Origin Version 1.1 (DCO 1.1) as stated in DCO.txt.
//============================================================================
#ifndef viskores_worklet_quadric_clustering_EnvelopeFlags_h
#define viskores_worklet_quadric_clustering_EnvelopeFlags_h

#include <viskores/filter/geometry_refinement/worklet/quadric_clustering/QuadricClusteringCommonWorklets.h>
#include <viskores/filter/geometry_refinement/worklet/quadric_clustering/SurfaceMesh.h>
#include <viskores/filter/geometry_refinement/worklet/quadric_clustering/TopologyFlags.h>

#include <viskores/cont/Algorithm.h>
#include <viskores/cont/ArrayHandleIndex.h>
#include <viskores/cont/ExecutionObjectBase.h>

#include <viskores/worklet/spatialstructure/BoundingVolumeHierarchy.h>

#include <limits>

VISKORES_QUADRIC_CLUSTERING_STRICT_FP_BEGIN

namespace viskores
{
namespace worklet
{
namespace quadric_clustering
{

/// The report of `EnvelopeFlags`.
struct EnvelopeReport
{
  /// Pieces examined, in both parts.
  viskores::Id Pieces = 0;
};

/// The data of a triangle mesh queried for distances: its vertices, its triangles and a
/// hierarchy over them; the centroids of the triangles, a hierarchy over them and the keys that
/// break ties between equal centroid distances, for the candidates of the single-triangle
/// condition.
struct DistanceMesh
{
  viskores::cont::ArrayHandle<Vec3> Points;
  viskores::cont::ArrayHandle<viskores::Id3> Triangles;
  viskores::worklet::spatialstructure::BoundingVolumeHierarchy Hierarchy;
  /// The largest absolute coordinate of the points.
  Real MaxAbsCoordinate = 0;
  viskores::cont::ArrayHandle<Vec3> Centroids;
  viskores::worklet::spatialstructure::BoundingVolumeHierarchy CentroidHierarchy;
  viskores::cont::ArrayHandle<viskores::Id3> Keys;
};

/// Which triangles the envelope flag tests, and against what.
enum class EnvelopeMode
{
  /// (F5): the output triangles of the 3-label triangles against the input, and the input
  /// triangles against the realised output.
  Surface,
  /// (F5'), the geometric-only mode: the simplex `g(t)` of every input triangle against the
  /// input, and every input triangle against the union of the simplices.
  Simplices
};

/// The halves of the envelope flag (F5) in use: (i) every output triangle within epsilon of the
/// input, (ii) every input triangle within epsilon of the output.
enum class EnvelopeSides : viskores::UInt8
{
  /// Neither: (F5) is off; with the fast thickening, (F7) certifies the error bound.
  None = 0,
  OutputToInput = 1,
  InputToOutput = 2,
  Both = 3
};

/// The verdicts of the envelope test on a labelling, per input triangle, with what an incremental
/// test on a later labelling needs: for part (ii), the box of the output that each test read
/// (`RecordingMesh`), and the scales.
struct EnvelopeVerdicts
{
  /// Part (i), the realised triangle against the input, and part (ii), the input triangle against
  /// the output; 1 where the test failed.
  viskores::cont::ArrayHandle<viskores::UInt8> FailedOutput;
  viskores::cont::ArrayHandle<viskores::UInt8> FailedInput;
  /// The region of the output that part (ii) read, empty where it did not run.
  viskores::cont::ArrayHandle<Vec3> Lower;
  viskores::cont::ArrayHandle<Vec3> Upper;
  Real ScaleInput = 0;
  Real ScaleOutput = 0;
};

/// The pre-test of Lemma "Displacement pre-test", one-sided: a triangle whose computed
/// displacement is `displacement` needs the certified test unless the upper bound
/// `(1 + 4u) D`, rounded up, on its exact displacement is at most `epsilon` (Lemma "One-sided
/// evaluation" (ii)).
VISKORES_EXEC_CONT inline bool DisplacementExceeds(Real displacement, Real epsilon)
{
  return LengthUp(displacement) > epsilon;
}

namespace detail
{

/// `D(t)`, the largest displacement of a vertex of `t` to its generator, compared with
/// `epsilon`: the pre-test of Lemma "Displacement pre-test".
struct NeedsTest : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn triangle,
                                WholeArrayIn points,
                                WholeArrayIn labels,
                                WholeArrayIn generators,
                                FieldOut needsTest);
  using ExecutionSignature = _5(_1, _2, _3, _4);

  VISKORES_CONT explicit NeedsTest(Real epsilon)
    : Epsilon(epsilon)
  {
  }

  template <typename PointsPortal, typename LabelsPortal>
  VISKORES_EXEC viskores::UInt8 operator()(const viskores::Id3& triangle,
                                           const PointsPortal& points,
                                           const LabelsPortal& labels,
                                           const PointsPortal& generators) const
  {
    Real displacement = 0;
    for (int k = 0; k < 3; ++k)
    {
      const Vec3 p = points.Get(triangle[k]);
      displacement =
        viskores::Max(displacement, Norm(Sub(generators.Get(labels.Get(triangle[k])), p)));
    }
    return DisplacementExceeds(displacement, this->Epsilon) ? 1 : 0;
  }

  Real Epsilon;
};

struct IsThreeLabelTriangle : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn triangle,
                                WholeArrayIn labels,
                                FieldIn needsTest,
                                FieldOut out);
  using ExecutionSignature = _4(_1, _2, _3);

  template <typename LabelsPortal>
  VISKORES_EXEC viskores::UInt8 operator()(const viskores::Id3& triangle,
                                           const LabelsPortal& labels,
                                           viskores::UInt8 needsTest) const
  {
    const viskores::Id a = labels.Get(triangle[0]);
    const viskores::Id b = labels.Get(triangle[1]);
    const viskores::Id c = labels.Get(triangle[2]);
    return (needsTest && AllDistinct(a, b, c)) ? 1 : 0;
  }
};

/// 1 where both flags are.
struct BothFlags : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn a, FieldIn b, FieldOut both);
  using ExecutionSignature = _3(_1, _2);

  VISKORES_EXEC viskores::UInt8 operator()(viskores::UInt8 a, viskores::UInt8 b) const
  {
    return (a && b) ? 1 : 0;
  }
};

/// The largest absolute coordinate of the triangles to be tested, for the margin `S`.
struct TestedMaxAbs : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn triangle,
                                FieldIn needsTest,
                                WholeArrayIn points,
                                WholeArrayIn labels,
                                WholeArrayIn generators,
                                FieldOut value);
  using ExecutionSignature = _6(_1, _2, _3, _4, _5);

  VISKORES_CONT explicit TestedMaxAbs(bool atGenerators)
    : AtGenerators(atGenerators)
  {
  }

  template <typename PointsPortal, typename LabelsPortal>
  VISKORES_EXEC Real operator()(const viskores::Id3& triangle,
                                viskores::UInt8 needsTest,
                                const PointsPortal& points,
                                const LabelsPortal& labels,
                                const PointsPortal& generators) const
  {
    Real s = 0;
    if (needsTest)
    {
      for (int k = 0; k < 3; ++k)
      {
        const Vec3 p =
          this->AtGenerators ? generators.Get(labels.Get(triangle[k])) : points.Get(triangle[k]);
        s = viskores::Max(s, MaxAbs(p));
      }
    }
    return s;
  }

  bool AtGenerators;
};

} // namespace detail

/// The certified envelope test of Section 8 of the paper, with the one-sided bounds of Lemma
/// "One-sided evaluation": a triangle passes only if every point of it lies within `epsilon`
/// of the mesh. A piece passes if `min_i d_i + D <= epsilon` (distance is 1-Lipschitz), or if
/// its three vertices are within `epsilon` of one triangle of the mesh (distance to a convex set
/// is convex); it fails the triangle if some `d_i > epsilon`; otherwise it is split in four, and
/// a piece still unresolved at the depth cap fails the triangle. A piece at depth `j` must clear
/// `epsilon` by `sqrt(3) j u S`, which covers the rounding of the subdivision points; `S`
/// bounds every coordinate of the triangles tested together and of the mesh.
struct CertifiedWithin
{
  template <typename Mesh>
  VISKORES_EXEC bool operator()(const Vec3& a,
                                const Vec3& b,
                                const Vec3& c,
                                const Mesh& mesh,
                                viskores::Id& pieces,
                                bool& complete) const
  {
    return (*this)(a, b, c, nullptr, mesh, pieces, complete);
  }

  /// The test with the distances of `a`, `b` and `c` given, as `mesh.DistanceUpTo(p, Epsilon)`
  /// computes them, when `corners` is not null: triangles that share corners share their
  /// distances.
  template <typename Mesh>
  VISKORES_EXEC bool operator()(const Vec3& a,
                                const Vec3& b,
                                const Vec3& c,
                                const Real* corners,
                                const Mesh& mesh,
                                viskores::Id& pieces,
                                bool& complete) const
  {
    // A piece carries the distances of its vertices, which its children share: each distance
    // is computed once, with the same value it would have if computed again. A distance above
    // epsilon fails the triangle whatever its value, so the queries are exact up to epsilon.
    struct Piece
    {
      Vec3 V[3];
      Real D[3];
      int Depth;
    };
    complete = true;
    Piece stack[4 * 16];
    int top = 0;
    stack[top] = Piece{ { a, b, c }, { 0, 0, 0 }, 0 };
    for (int k = 0; k < 3; ++k)
    {
      bool done = true;
      stack[top].D[k] =
        corners ? corners[k] : mesh.DistanceUpTo(stack[top].V[k], this->Epsilon, done);
      complete = complete && done;
      if (stack[top].D[k] > this->Epsilon)
      {
        ++pieces;
        return false;
      }
    }
    ++top;
    // Above sqrt(3), which the correctly rounded square root may not be.
    const Real sqrt3 = NextUp(viskores::Sqrt(Real(3)));
    typename Mesh::Seeds seeds;
    while (top > 0)
    {
      const Piece piece = stack[--top];
      ++pieces;
      const Real margin =
        NextUp(NextUp(sqrt3 * Real(piece.Depth)) * UnitRoundoff * this->MaxCoordinate);
      Real diameter = 0;
      for (int k = 0; k < 3; ++k)
      {
        diameter = viskores::Max(diameter, Norm(Sub(piece.V[k], piece.V[(k + 2) % 3])));
      }
      diameter = NextUp((1 + 4 * UnitRoundoff) * diameter);
      const Real nearest = viskores::Min(viskores::Min(piece.D[0], piece.D[1]), piece.D[2]);
      if (NextUp(NextUp(nearest + diameter) + margin) <= this->Epsilon)
      {
        continue;
      }
      bool done = true;
      if (mesh.WithinOneTriangle(piece.V, this->Epsilon, margin, done, seeds))
      {
        complete = complete && done;
        continue;
      }
      complete = complete && done;
      if (piece.Depth == this->MaxDepth)
      {
        return false;
      }
      const Vec3& p0 = piece.V[0];
      const Vec3& p1 = piece.V[1];
      const Vec3& p2 = piece.V[2];
      const Vec3 m01 = Scale(Real(0.5), Add(p0, p1));
      const Vec3 m12 = Scale(Real(0.5), Add(p1, p2));
      const Vec3 m20 = Scale(Real(0.5), Add(p2, p0));
      Real dm[3];
      const Vec3* mid[3] = { &m01, &m12, &m20 };
      for (int k = 0; k < 3; ++k)
      {
        dm[k] = mesh.DistanceUpTo(*mid[k], this->Epsilon, done, &seeds);
        complete = complete && done;
        if (dm[k] > this->Epsilon)
        {
          return false;
        }
      }
      const int depth = piece.Depth + 1;
      stack[top++] = Piece{ { m01, m12, m20 }, { dm[0], dm[1], dm[2] }, depth };
      stack[top++] = Piece{ { m20, m12, p2 }, { dm[2], dm[1], piece.D[2] }, depth };
      stack[top++] = Piece{ { m01, p1, m12 }, { dm[0], piece.D[1], dm[1] }, depth };
      stack[top++] = Piece{ { p0, m01, m20 }, { piece.D[0], dm[0], dm[2] }, depth };
    }
    return true;
  }

  Real Epsilon;
  /// `S (1 + 16 u)`.
  Real MaxCoordinate;
  int MaxDepth = 10;
};

namespace detail
{

/// Marks the labels of the triangles to test, whose generators are their corners in part (i);
/// every write is a 1.
struct MarkTestedLabels : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn triangle,
                                FieldIn tested,
                                WholeArrayIn labels,
                                WholeArrayOut wanted);
  using ExecutionSignature = void(_1, _2, _3, _4);

  template <typename LabelsPortal, typename FlagPortal>
  VISKORES_EXEC void operator()(const viskores::Id3& triangle,
                                viskores::UInt8 tested,
                                const LabelsPortal& labels,
                                FlagPortal& wanted) const
  {
    if (tested)
    {
      for (int k = 0; k < 3; ++k)
      {
        wanted.Set(labels.Get(triangle[k]), 1);
      }
    }
  }
};

/// `mesh.DistanceUpTo(p, epsilon)` for the points that are wanted, the first step of the
/// certified test of every triangle with a corner there.
struct CornerDistance : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn point, FieldIn wanted, ExecObject mesh, FieldOut distance);
  using ExecutionSignature = void(_1, _2, _3, _4);

  VISKORES_CONT explicit CornerDistance(Real epsilon)
    : Epsilon(epsilon)
  {
  }

  template <typename Mesh>
  VISKORES_EXEC void operator()(const Vec3& p,
                                viskores::UInt8 wanted,
                                const Mesh& mesh,
                                Real& distance) const
  {
    distance = 0;
    if (wanted)
    {
      bool complete = true;
      distance = mesh.DistanceUpTo(p, this->Epsilon, complete);
      if (!complete)
      {
        this->RaiseError("QuadricClusteringDecimation: bounding volume hierarchy too deep.");
      }
    }
  }

private:
  Real Epsilon;
};

VISKORES_EXEC_CONT inline Real Clip(Real x, Real lo, Real hi)
{
  return viskores::Min(viskores::Max(x, lo), hi);
}

/// `nan_to_num` of numpy: NaN to 0, infinities to the largest finite values.
VISKORES_EXEC_CONT inline Real NanToNum(Real x)
{
  if (x != x)
  {
    return 0;
  }
  if (x == std::numeric_limits<Real>::infinity())
  {
    return std::numeric_limits<Real>::max();
  }
  if (x == -std::numeric_limits<Real>::infinity())
  {
    return -std::numeric_limits<Real>::max();
  }
  return x;
}

/// `|p - (base + v (x - base) + w (y - base))|`, evaluated in this order.
VISKORES_EXEC_CONT inline Real Evaluate(const Vec3& p,
                                        const Vec3& base,
                                        const Vec3& x,
                                        const Vec3& y,
                                        Real v,
                                        Real w)
{
  const Vec3 q = Add(Add(base, Scale(v, Sub(x, base))), Scale(w, Sub(y, base)));
  return Norm(Sub(p, q));
}

} // namespace detail

/// The computed distance from `p` to a point of the triangle `abc` of the form of Lemma
/// "One-sided evaluation" (i): the nearest of the clamped projections onto the three edges and
/// onto the plane. Every candidate is a point of the triangle up to the rounding of its
/// evaluation, whatever rounding went into its parameters.
VISKORES_EXEC_CONT inline Real FeasiblePointDistance(const Vec3& p,
                                                     const Vec3& a,
                                                     const Vec3& b,
                                                     const Vec3& c)
{
  Real best = std::numeric_limits<Real>::infinity();
  const Vec3* ends[3][2] = { { &a, &b }, { &b, &c }, { &c, &a } };
  for (int e = 0; e < 3; ++e)
  {
    const Vec3& x = *ends[e][0];
    const Vec3& y = *ends[e][1];
    const Vec3 d = Sub(y, x);
    const Real t = detail::Clip(detail::NanToNum(Dot(Sub(p, x), d) / Dot(d, d)), 0, 1);
    best = viskores::Min(best, detail::Evaluate(p, x, y, x, t, 0));
  }
  const Vec3 ab = Sub(b, a), ac = Sub(c, a), ap = Sub(p, a);
  const Real d00 = Dot(ab, ab), d01 = Dot(ab, ac), d11 = Dot(ac, ac);
  const Real d20 = Dot(ap, ab), d21 = Dot(ap, ac);
  const Real den = d00 * d11 - d01 * d01;
  Real v = detail::NanToNum((d11 * d20 - d01 * d21) / den);
  Real w = detail::NanToNum((d00 * d21 - d01 * d20) / den);
  v = detail::Clip(v, 0, 1);
  w = viskores::Max(w, Real(0));
  if (v + w > 1)
  {
    w = 1 - v;
  }
  return viskores::Min(best, detail::Evaluate(p, a, b, c, v, w));
}

/// An upper bound on the distance from `p` to the triangle `abc`: `(1 + 4u) d + 32 u S` with
/// `d = FeasiblePointDistance` and `S` the largest absolute coordinate involved, rounded up.
VISKORES_EXEC_CONT inline Real PointTriangleDistanceUpper(const Vec3& p,
                                                          const Vec3& a,
                                                          const Vec3& b,
                                                          const Vec3& c)
{
  Real s = 0;
  for (int i = 0; i < 3; ++i)
  {
    s = viskores::Max(s, viskores::Abs(p[i]));
    s = viskores::Max(s, viskores::Abs(a[i]));
    s = viskores::Max(s, viskores::Abs(b[i]));
    s = viskores::Max(s, viskores::Abs(c[i]));
  }
  return NextUp(NextUp((1 + 4 * UnitRoundoff) * FeasiblePointDistance(p, a, b, c)) +
                32 * UnitRoundoff * s);
}

/// The candidates of the single-triangle condition, as `single_triangle_bound` of
/// proofs/lib/common.py: the triangles whose centroids are nearest the centroid of a piece, at
/// most eight, by (distance, key), whatever the spatial structure. A key is the position of
/// the triangle in the mesh the reference builds: `(index, 0, 0)`, or the sorted label triple
/// of a simplex of the geometric-only mode, whose mesh is the sorted list of distinct simplices;
/// a key offered twice is the same simplex, kept once.
struct CentroidCandidates
{
  static constexpr int Capacity = 8;

  /// Whether the three points `s` plus `margin` are within `bound` of one candidate: the
  /// smallest over them of the largest upper bound from a point, plus `margin`, rounded up, at
  /// most `bound`.
  VISKORES_EXEC bool Passes(const Vec3* s, Real bound, Real margin) const
  {
    if (this->Count == 0)
    {
      return false;
    }
    Real best = std::numeric_limits<Real>::infinity();
    for (int c = 0; c < this->Count; ++c)
    {
      Real worst = 0;
      for (int k = 0; k < 3; ++k)
      {
        worst = viskores::Max(
          worst, PointTriangleDistanceUpper(s[k], this->P[c][0], this->P[c][1], this->P[c][2]));
      }
      best = viskores::Min(best, worst);
    }
    return NextUp(best + margin) <= bound;
  }

  VISKORES_EXEC static bool Less(const viskores::Id3& a, const viskores::Id3& b)
  {
    return a[0] != b[0] ? a[0] < b[0] : (a[1] != b[1] ? a[1] < b[1] : a[2] < b[2]);
  }

  /// Offers the triangle `abc`, at centroid distance `d`, with key `key`, and its index in the
  /// mesh searched, if it has one, to seed the next search.
  VISKORES_EXEC void Offer(Real d,
                           const viskores::Id3& key,
                           const Vec3& a,
                           const Vec3& b,
                           const Vec3& c,
                           viskores::Id triangle = -1)
  {
    for (int i = 0; i < this->Count; ++i)
    {
      if (this->Key[i] == key)
      {
        return;
      }
    }
    if (this->Count == Capacity &&
        !(d < this->Distance[Capacity - 1] ||
          (d == this->Distance[Capacity - 1] && Less(key, this->Key[Capacity - 1]))))
    {
      return;
    }
    int slot = (this->Count < Capacity) ? this->Count++ : Capacity - 1;
    while (slot > 0 &&
           (d < this->Distance[slot - 1] ||
            (d == this->Distance[slot - 1] && Less(key, this->Key[slot - 1]))))
    {
      this->Distance[slot] = this->Distance[slot - 1];
      this->Key[slot] = this->Key[slot - 1];
      this->Triangle[slot] = this->Triangle[slot - 1];
      for (int k = 0; k < 3; ++k)
      {
        this->P[slot][k] = this->P[slot - 1][k];
      }
      --slot;
    }
    this->Distance[slot] = d;
    this->Key[slot] = key;
    this->Triangle[slot] = triangle;
    this->P[slot][0] = a;
    this->P[slot][1] = b;
    this->P[slot][2] = c;
  }

  int Count = 0;
  Real Distance[Capacity];
  viskores::Id3 Key[Capacity];
  viskores::Id Triangle[Capacity];
  Vec3 P[Capacity][3];
};

/// The triangles kept by the last search of the single-triangle candidates of a test, offered
/// first to the next one, whose query is nearby: the eight nearest centroids are the first by
/// (distance, key) whatever the order of the offers, since no two triangles of a mesh share a
/// key, and a bound found sooner prunes the traversal sooner.
struct CentroidSeeds
{
  VISKORES_EXEC void Keep(const CentroidCandidates& candidates)
  {
    this->Count = 0;
    for (int c = 0; c < candidates.Count; ++c)
    {
      if (candidates.Triangle[c] >= 0)
      {
        this->Triangle[this->Count++] = candidates.Triangle[c];
      }
    }
  }

  int Count = 0;
  viskores::Id Triangle[CentroidCandidates::Capacity];
};

/// The centroid `((a + b) + c) / 3` of a triangle, in this order.
VISKORES_EXEC_CONT inline Vec3 TriangleCentroid(const Vec3& a, const Vec3& b, const Vec3& c)
{
  const Vec3 s = Add(Add(a, b), c);
  return Vec3(s[0] / 3, s[1] / 3, s[2] / 3);
}

/// Execution-side view of a `DistanceMesh`.
template <typename VecPortal,
          typename Id3Portal,
          typename HierarchyPortals,
          typename KeyPortal = typename viskores::cont::ArrayHandle<viskores::Id3>::ReadPortalType>
struct DistanceMeshPortals
{
  /// The smallest upper bound `PointTriangleDistanceUpper` over all the triangles: the
  /// computed distance from `p` to the mesh. A triangle is skipped only if a lower bound on its
  /// distance exceeds the best bound found, so it could not lower the minimum.
  VISKORES_EXEC Real Distance(const Vec3& p, bool& complete) const
  {
    return this->DistanceSkipping(p, SkipNone{}, std::numeric_limits<Real>::infinity(), complete);
  }

  /// `DistanceUpTo` over the triangles `t` with `skip(t)` false. The upper bounds are not below
  /// the distance to the boxes, so the query of the hierarchy returns their exact minimum when
  /// that is at most `cutoff`. The bound of the first of `seeds`, a triangle near `p`, starts the
  /// search.
  template <typename Skip>
  VISKORES_EXEC Real DistanceSkipping(const Vec3& p,
                                      const Skip& skip,
                                      Real cutoff,
                                      bool& complete,
                                      const CentroidSeeds* seeds = nullptr) const
  {
    auto bound = [&](viskores::Id t)
    { return skip(t) ? std::numeric_limits<Real>::infinity() : this->TriangleBound(p, t); };
    const Real initial = (seeds && seeds->Count > 0) ? bound(seeds->Triangle[0])
                                                     : std::numeric_limits<Real>::infinity();
    return this->Hierarchy.FindNearest(p, bound, complete, cutoff, initial);
  }

  struct SkipNone
  {
    VISKORES_EXEC bool operator()(viskores::Id) const { return false; }
  };

  VISKORES_EXEC Real TriangleBound(const Vec3& p, viskores::Id t) const
  {
    const viskores::Id3 tri = this->Triangles.Get(t);
    return PointTriangleDistanceUpper(
      p, this->Points.Get(tri[0]), this->Points.Get(tri[1]), this->Points.Get(tri[2]));
  }

  /// `Distance` when it is at most `cutoff`, and otherwise some value above `cutoff`.
  VISKORES_EXEC Real DistanceUpTo(const Vec3& p,
                                  Real cutoff,
                                  bool& complete,
                                  const CentroidSeeds* seeds = nullptr) const
  {
    return this->DistanceSkipping(p, SkipNone{}, cutoff, complete, seeds);
  }

  /// Whether the three points `s` are within `bound`, with `margin`, of one triangle among
  /// the candidates of the single-triangle condition: then every point of their convex hull is,
  /// since the distance to a convex set is convex.
  using Seeds = CentroidSeeds;

  VISKORES_EXEC bool WithinOneTriangle(const Vec3* s,
                                       Real bound,
                                       Real margin,
                                       bool& complete,
                                       CentroidSeeds& seeds) const
  {
    CentroidCandidates candidates;
    this->NearestCentroids(
      TriangleCentroid(s[0], s[1], s[2]), SkipNone{}, candidates, complete, &seeds);
    seeds.Keep(candidates);
    return candidates.Passes(s, bound, margin);
  }

  /// Offers the triangles whose centroids are nearest `q` to `candidates`, but those with
  /// `skip(t)`, the triangles `seeds` first. The candidates kept are the first by (distance, key)
  /// whatever the order of the offers, so the seeds and the nearer subtrees go first only to
  /// prune sooner.
  template <typename Skip>
  VISKORES_EXEC void NearestCentroids(const Vec3& q,
                                      const Skip& skip,
                                      CentroidCandidates& candidates,
                                      bool& complete,
                                      const CentroidSeeds* seeds = nullptr) const
  {
    auto offer = [&](viskores::Id t)
    {
      const viskores::Id3 tri = this->Triangles.Get(t);
      candidates.Offer(Norm(Sub(this->Centroids.Get(t), q)),
                       this->Keys.Get(t),
                       this->Points.Get(tri[0]),
                       this->Points.Get(tri[1]),
                       this->Points.Get(tri[2]),
                       t);
    };
    for (int s = 0; seeds && s < seeds->Count; ++s)
    {
      if (!skip(seeds->Triangle[s]))
      {
        offer(seeds->Triangle[s]);
      }
    }
    const bool done = this->CentroidHierarchy.TraverseNearestFirst(
      q,
      [&]
      {
        return candidates.Count < CentroidCandidates::Capacity
          ? std::numeric_limits<Real>::infinity()
          : candidates.Distance[CentroidCandidates::Capacity - 1];
      },
      [&](viskores::Id t)
      {
        if (!skip(t))
        {
          offer(t);
        }
        return true;
      });
    complete = complete && done;
  }

  VecPortal Points;
  Id3Portal Triangles;
  HierarchyPortals Hierarchy;
  Real MaxAbsCoordinate;
  VecPortal Centroids;
  HierarchyPortals CentroidHierarchy;
  KeyPortal Keys;
};

/// An execution object giving a worklet a `DistanceMeshPortals`.
struct DistanceMeshObject : viskores::cont::ExecutionObjectBase
{
  explicit DistanceMeshObject(const DistanceMesh& mesh)
    : Mesh(mesh)
  {
  }

  VISKORES_CONT auto PrepareForExecution(viskores::cont::DeviceAdapterId device,
                                         viskores::cont::Token& token) const
  {
    auto hierarchy = this->Mesh.Hierarchy.PrepareForExecution(device, token);
    using Portals =
      DistanceMeshPortals<typename viskores::cont::ArrayHandle<Vec3>::ReadPortalType,
                          typename viskores::cont::ArrayHandle<viskores::Id3>::ReadPortalType,
                          decltype(hierarchy)>;
    return Portals{ this->Mesh.Points.PrepareForInput(device, token),
                    this->Mesh.Triangles.PrepareForInput(device, token),
                    hierarchy,
                    this->Mesh.MaxAbsCoordinate,
                    this->Mesh.Centroids.PrepareForInput(device, token),
                    this->Mesh.CentroidHierarchy.PrepareForExecution(device, token),
                    this->Mesh.Keys.PrepareForInput(device, token) };
  }

  DistanceMesh Mesh;
};

/// The box `[lower, upper]` grown to cover the ball of radius `radius` around `p`, with the margin
/// `radius (1 + 1e-9) + 64 u size` for the rounding of the queries and of the box; the whole space
/// if `radius` is infinite.
VISKORES_EXEC inline void ExtendByBall(const Vec3& p,
                                       Real radius,
                                       Real size,
                                       Vec3& lower,
                                       Vec3& upper)
{
  const Real grow = radius * (1 + Real(1e-9)) + 64 * UnitRoundoff * size;
  for (int a = 0; a < 3; ++a)
  {
    lower[a] = viskores::Min(lower[a], p[a] - grow);
    upper[a] = viskores::Max(upper[a], p[a] + grow);
  }
}

/// A distance mesh that records, in `[Lower, Upper]`, the region of the mesh that its queries
/// read: the result of `DistanceUpTo(p, cutoff)`, when at most `cutoff`, depends only on the
/// triangles whose boxes meet the ball of radius `cutoff` around `p`, and the candidates of
/// `WithinOneTriangle` only on the centroids in the ball around the query whose radius is the
/// distance of the last candidate kept (the whole space with fewer than eight).
template <typename Mesh>
struct RecordingMesh
{
  VISKORES_EXEC Real DistanceUpTo(const Vec3& p,
                                  Real cutoff,
                                  bool& complete,
                                  const CentroidSeeds* seeds = nullptr) const
  {
    ExtendByBall(p, cutoff, this->Size, this->Lower, this->Upper);
    return this->Inner.DistanceUpTo(p, cutoff, complete, seeds);
  }

  using Seeds = CentroidSeeds;

  VISKORES_EXEC bool WithinOneTriangle(const Vec3* s,
                                       Real bound,
                                       Real margin,
                                       bool& complete,
                                       CentroidSeeds& seeds) const
  {
    const Vec3 q = TriangleCentroid(s[0], s[1], s[2]);
    CentroidCandidates candidates;
    this->Inner.NearestCentroids(q, typename Mesh::SkipNone{}, candidates, complete, &seeds);
    seeds.Keep(candidates);
    ExtendByBall(q,
                 candidates.Count < CentroidCandidates::Capacity
                   ? std::numeric_limits<Real>::infinity()
                   : candidates.Distance[CentroidCandidates::Capacity - 1],
                 this->Size,
                 this->Lower,
                 this->Upper);
    return candidates.Passes(s, bound, margin);
  }

  const Mesh& Inner;
  Real Size;
  Vec3& Lower;
  Vec3& Upper;
};

namespace detail
{

struct TestTriangle : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn triangle,
                                FieldIn needsTest,
                                WholeArrayIn points,
                                WholeArrayIn labels,
                                WholeArrayIn generators,
                                WholeArrayIn cornerDistances,
                                ExecObject distanceMesh,
                                FieldOut failed,
                                FieldOut pieces,
                                FieldOut lower,
                                FieldOut upper);
  using ExecutionSignature = void(_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11);

  /// `atGenerators` for part (i); part (ii) records the region of the output it reads.
  VISKORES_CONT TestTriangle(const CertifiedWithin& test, bool atGenerators)
    : Test(test)
    , AtGenerators(atGenerators)
  {
  }

  template <typename PointsPortal, typename LabelsPortal, typename DistancesPortal, typename Mesh>
  VISKORES_EXEC void operator()(const viskores::Id3& triangle,
                                viskores::UInt8 needsTest,
                                const PointsPortal& points,
                                const LabelsPortal& labels,
                                const PointsPortal& generators,
                                const DistancesPortal& cornerDistances,
                                const Mesh& mesh,
                                viskores::UInt8& failed,
                                viskores::Id& pieces,
                                Vec3& lower,
                                Vec3& upper) const
  {
    failed = 0;
    pieces = 0;
    lower = Vec3(std::numeric_limits<Real>::infinity());
    upper = Vec3(-std::numeric_limits<Real>::infinity());
    if (!needsTest)
    {
      return;
    }
    Vec3 v[3];
    Real d[3];
    for (int k = 0; k < 3; ++k)
    {
      const viskores::Id corner = this->AtGenerators ? labels.Get(triangle[k]) : triangle[k];
      v[k] = this->AtGenerators ? generators.Get(corner) : points.Get(corner);
      d[k] = cornerDistances.Get(corner);
    }
    bool complete = true;
    bool within;
    if (this->AtGenerators)
    {
      within = this->Test(v[0], v[1], v[2], d, mesh, pieces, complete);
    }
    else
    {
      // The corner distances were queries too.
      for (int k = 0; k < 3; ++k)
      {
        ExtendByBall(v[k], this->Test.Epsilon, this->Test.MaxCoordinate, lower, upper);
      }
      const RecordingMesh<Mesh> recording{ mesh, this->Test.MaxCoordinate, lower, upper };
      within = this->Test(v[0], v[1], v[2], d, recording, pieces, complete);
    }
    if (!complete)
    {
      this->RaiseError("QuadricClusteringDecimation: bounding volume hierarchy too deep.");
    }
    failed = within ? 0 : 1;
  }

  CertifiedWithin Test;
  bool AtGenerators;
};

/// The value computed where `affected`, and otherwise that of the base.
struct KeepBase : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn affected, FieldIn base, FieldInOut value);
  using ExecutionSignature = void(_1, _2, _3);

  template <typename T>
  VISKORES_EXEC void operator()(viskores::UInt8 affected, const T& base, T& value) const
  {
    if (!affected)
    {
      value = base;
    }
  }
};

struct CentroidOf : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn triangle, WholeArrayIn points, FieldOut centroid);
  using ExecutionSignature = _3(_1, _2);

  template <typename PointsPortal>
  VISKORES_EXEC Vec3 operator()(const viskores::Id3& t, const PointsPortal& points) const
  {
    return TriangleCentroid(points.Get(t[0]), points.Get(t[1]), points.Get(t[2]));
  }
};

/// The key `(i, 0, 0)` of an index `i`.
struct IndexKey : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn index, FieldOut key);
  using ExecutionSignature = _2(_1);

  VISKORES_EXEC viskores::Id3 operator()(viskores::Id i) const { return viskores::Id3(i, 0, 0); }
};

} // namespace detail

/// The `DistanceMesh` of `triangles` on `points`; `keys`, by default `(index, 0, 0)`, break
/// ties between equal centroid distances (`CentroidCandidates`).
inline DistanceMesh BuildDistanceMesh(const viskores::cont::ArrayHandle<Vec3>& points,
                                      const viskores::cont::ArrayHandle<viskores::Id3>& triangles,
                                      const viskores::cont::ArrayHandle<viskores::Id3>& keys =
                                        viskores::cont::ArrayHandle<viskores::Id3>{})
{
  viskores::cont::Invoker invoke;
  DistanceMesh mesh;
  mesh.Points = points;
  mesh.Triangles = triangles;
  viskores::cont::ArrayHandle<Vec3> lower, upper;
  invoke(TriangleBox{}, triangles, points, lower, upper);
  mesh.Hierarchy.Build(lower, upper);
  mesh.MaxAbsCoordinate = MaxAbsCoordinate(points);
  invoke(detail::CentroidOf{}, triangles, points, mesh.Centroids);
  mesh.CentroidHierarchy.Build(mesh.Centroids, mesh.Centroids);
  if (keys.GetNumberOfValues() == triangles.GetNumberOfValues())
  {
    mesh.Keys = keys;
  }
  else
  {
    invoke(detail::IndexKey{},
           viskores::cont::ArrayHandleIndex(triangles.GetNumberOfValues()),
           mesh.Keys);
  }
  return mesh;
}

namespace detail
{

/// Whether a vertex is a corner of a triangle to test.
struct IsTestedCorner : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn vertex,
                                WholeArrayIn cornerOffsets,
                                WholeArrayIn corners,
                                WholeArrayIn needsTest,
                                FieldOut tested);
  using ExecutionSignature = void(_1, _2, _3, _4, _5);

  VISKORES_CONT explicit IsTestedCorner(viskores::Id numberOfTriangles)
    : NumberOfTriangles(numberOfTriangles)
  {
  }

  template <typename IdPortal, typename FlagPortal>
  VISKORES_EXEC void operator()(viskores::Id v,
                                const IdPortal& cornerOffsets,
                                const IdPortal& corners,
                                const FlagPortal& needsTest,
                                viskores::UInt8& tested) const
  {
    tested = 0;
    for (viskores::Id r = cornerOffsets.Get(v); r < cornerOffsets.Get(v + 1) && !tested; ++r)
    {
      tested = needsTest.Get(corners.Get(r) % this->NumberOfTriangles);
    }
  }

private:
  viskores::Id NumberOfTriangles;
};

} // namespace detail

/// Flags, among the clusters with at least two vertices, every cluster of a vertex of a triangle
/// that `verdicts` says failed.
inline void FlagFailedTriangles(const SurfaceMesh& mesh,
                                const viskores::cont::ArrayHandle<viskores::Id>& labels,
                                const viskores::cont::ArrayHandle<viskores::Id>& sizes,
                                const EnvelopeVerdicts& verdicts,
                                viskores::cont::ArrayHandle<viskores::UInt8>& flags)
{
  viskores::cont::ArrayHandle<viskores::UInt8> failed;
  viskores::cont::Algorithm::Copy(verdicts.FailedOutput, failed);
  OrFlags(failed, verdicts.FailedInput);
  FlagClustersOfTriangles(mesh, labels, failed, sizes, flags);
}

/// The envelope flag of Definition "Envelope flag" and Algorithm 7, for the triangles with
/// `D(t) > epsilon`, with the halves `sides`. `inputDistance` is the distance structure of the
/// input, built once, and `outputTriangles` the realised output, as triples of labels. Flags,
/// among the clusters with at least two vertices, every cluster of a vertex of a triangle that
/// fails a test.
///
/// With `verdicts`, stores the verdict of every triangle and what part (ii) read. With `base`, the
/// verdicts of an earlier labelling, and `affected`, the triangles whose test may read something
/// that differs from it, tests only those and takes the other verdicts from `base`. A test that
/// reads the same values computes the same values, so these are the verdicts of the whole test,
/// as long as the scales are those of `base`; a half whose scale differs is tested in full.
inline EnvelopeReport EnvelopeFlags(
  const SurfaceMesh& mesh,
  const DistanceMesh& inputDistance,
  const viskores::cont::ArrayHandle<viskores::Id>& labels,
  const viskores::cont::ArrayHandle<Vec3>& generators,
  const viskores::cont::ArrayHandle<viskores::Id3>& outputTriangles,
  const viskores::cont::ArrayHandle<viskores::Id>& sizes,
  Real epsilon,
  EnvelopeMode mode,
  EnvelopeSides sides,
  viskores::cont::ArrayHandle<viskores::UInt8>& flags,
  EnvelopeVerdicts* verdicts = nullptr,
  const EnvelopeVerdicts* base = nullptr,
  const viskores::cont::ArrayHandle<viskores::UInt8>* affected = nullptr)
{
  using Algorithm = viskores::cont::Algorithm;
  viskores::cont::Invoker invoke;
  EnvelopeReport report;
  const viskores::Id m = mesh.NumberOfTriangles;
  EnvelopeVerdicts local;
  EnvelopeVerdicts& out = verdicts ? *verdicts : local;
  out.FailedOutput.AllocateAndFill(m, 0);
  out.FailedInput.AllocateAndFill(m, 0);
  out.Lower.AllocateAndFill(m, Vec3(std::numeric_limits<Real>::infinity()));
  out.Upper.AllocateAndFill(m, Vec3(-std::numeric_limits<Real>::infinity()));
  out.ScaleInput = 0;
  out.ScaleOutput = 0;
  viskores::cont::ArrayHandle<viskores::UInt8> needsTest;
  invoke(detail::NeedsTest(epsilon), mesh.Triangles, mesh.Points, labels, generators, needsTest);
  const viskores::Id numNeeding = Algorithm::Reduce(needsTest, viskores::Id(0));
  if (numNeeding == 0)
  {
    return report;
  }
  viskores::cont::ArrayHandle<viskores::UInt8> testOutput;
  if (mode == EnvelopeMode::Surface)
  {
    invoke(detail::IsThreeLabelTriangle{}, mesh.Triangles, labels, needsTest, testOutput);
  }
  else
  {
    testOutput = needsTest;
  }
  const Real scaleFactor = 1 + 16 * UnitRoundoff;
  // The triangles a half tests: all, or with a base of the same scale, the affected ones.
  auto restrict = [&](const viskores::cont::ArrayHandle<viskores::UInt8>& all, bool sameScale)
  {
    if (!sameScale)
    {
      return all;
    }
    viskores::cont::ArrayHandle<viskores::UInt8> some;
    invoke(detail::BothFlags{}, all, *affected, some);
    return some;
  };
  const bool incremental = base && affected;

  // A half not in use tests no triangle.
  viskores::cont::ArrayHandle<viskores::Id> piecesOutput, piecesInput;
  // Part (i) reads the input only, and records nothing.
  viskores::cont::ArrayHandle<Vec3> unusedLower, unusedUpper;
  if (!(static_cast<viskores::UInt8>(sides) &
        static_cast<viskores::UInt8>(EnvelopeSides::OutputToInput)))
  {
    testOutput.AllocateAndFill(m, 0);
    piecesOutput.AllocateAndFill(m, 0);
  }
  else
  {
    // (i): the realised triangles against the input.
    viskores::cont::ArrayHandle<Real> values;
    invoke(detail::TestedMaxAbs(true),
           mesh.Triangles,
           testOutput,
           mesh.Points,
           labels,
           generators,
           values);
    const Real s = viskores::Max(Algorithm::Reduce(values, Real(0), viskores::Maximum()),
                                 inputDistance.MaxAbsCoordinate);
    CertifiedWithin test{ epsilon, s * scaleFactor };
    out.ScaleInput = test.MaxCoordinate;
    const bool sameScale = incremental && base->ScaleInput == test.MaxCoordinate;
    testOutput = restrict(testOutput, sameScale);
    // The corners are generators, shared by many triangles.
    viskores::cont::ArrayHandle<viskores::UInt8> wanted;
    wanted.AllocateAndFill(generators.GetNumberOfValues(), 0);
    invoke(detail::MarkTestedLabels{}, mesh.Triangles, testOutput, labels, wanted);
    viskores::cont::ArrayHandle<Real> cornerDistances;
    invoke(detail::CornerDistance(epsilon),
           generators,
           wanted,
           DistanceMeshObject(inputDistance),
           cornerDistances);
    invoke(detail::TestTriangle(test, true),
           mesh.Triangles,
           testOutput,
           mesh.Points,
           labels,
           generators,
           cornerDistances,
           DistanceMeshObject(inputDistance),
           out.FailedOutput,
           piecesOutput,
           unusedLower,
           unusedUpper);
    if (sameScale)
    {
      invoke(detail::KeepBase{}, *affected, base->FailedOutput, out.FailedOutput);
    }
  }
  // Part (ii) tests the triangles that need a test.
  viskores::cont::ArrayHandle<viskores::UInt8> testInput = needsTest;
  if (!(static_cast<viskores::UInt8>(sides) &
        static_cast<viskores::UInt8>(EnvelopeSides::InputToOutput)))
  {
    testInput.AllocateAndFill(m, 0);
    piecesInput.AllocateAndFill(m, 0);
  }
  else
  {
    // (ii): the input triangles against the realised output.
    const DistanceMesh outputDistance = BuildDistanceMesh(generators, outputTriangles);
    viskores::cont::ArrayHandle<Real> values;
    invoke(detail::TestedMaxAbs(false),
           mesh.Triangles,
           testInput,
           mesh.Points,
           labels,
           generators,
           values);
    const Real s = viskores::Max(Algorithm::Reduce(values, Real(0), viskores::Maximum()),
                                 outputDistance.MaxAbsCoordinate);
    CertifiedWithin test{ epsilon, s * scaleFactor };
    out.ScaleOutput = test.MaxCoordinate;
    const bool sameScale = incremental && base->ScaleOutput == test.MaxCoordinate;
    testInput = restrict(testInput, sameScale);
    // The corners are input vertices, each shared by several triangles.
    viskores::cont::ArrayHandle<viskores::UInt8> tested;
    invoke(detail::IsTestedCorner(mesh.NumberOfTriangles),
           viskores::cont::ArrayHandleIndex(mesh.NumberOfPoints),
           mesh.VertexCornerOffsets,
           mesh.VertexCorners,
           testInput,
           tested);
    viskores::cont::ArrayHandle<Real> cornerDistances;
    invoke(detail::CornerDistance(epsilon),
           mesh.Points,
           tested,
           DistanceMeshObject(outputDistance),
           cornerDistances);
    invoke(detail::TestTriangle(test, false),
           mesh.Triangles,
           testInput,
           mesh.Points,
           labels,
           generators,
           cornerDistances,
           DistanceMeshObject(outputDistance),
           out.FailedInput,
           piecesInput,
           out.Lower,
           out.Upper);
    if (sameScale)
    {
      invoke(detail::KeepBase{}, *affected, base->FailedInput, out.FailedInput);
      invoke(detail::KeepBase{}, *affected, base->Lower, out.Lower);
      invoke(detail::KeepBase{}, *affected, base->Upper, out.Upper);
    }
  }
  const viskores::Id failedOutput = Algorithm::Reduce(out.FailedOutput, viskores::Id(0));
  const viskores::Id failedInput = Algorithm::Reduce(out.FailedInput, viskores::Id(0));
  report.Pieces = Algorithm::Reduce(piecesOutput, viskores::Id(0)) +
    Algorithm::Reduce(piecesInput, viskores::Id(0));
  if (failedOutput + failedInput > 0)
  {
    FlagFailedTriangles(mesh, labels, sizes, out, flags);
  }
  return report;
}

}
}
} // namespace viskores::worklet::quadric_clustering

VISKORES_QUADRIC_CLUSTERING_STRICT_FP_END

#endif // viskores_worklet_quadric_clustering_EnvelopeFlags_h
