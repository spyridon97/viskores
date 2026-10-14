//============================================================================
//  The contents of this file are covered by the Viskores license. See
//  LICENSE.txt for details.
//
//  By contributing to this file, all contributors agree to the Developer
//  Certificate of Origin Version 1.1 (DCO 1.1) as stated in DCO.txt.
//============================================================================
#ifndef viskores_ExactPredicates_h
#define viskores_ExactPredicates_h

#include <viskores/Math.h>
#include <viskores/Types.h>

// Exact orientation predicates on double-precision coordinates, after Shewchuk, "Adaptive
// Precision Floating-Point Arithmetic and Fast Robust Geometric Predicates" (1997): a
// floating-point evaluation with an error bound, and an exact evaluation in expansion arithmetic
// when the bound does not settle the sign. Both need every operation rounded separately, so
// fused multiply-add contraction is turned off for this header. With CUDA, compile with
// --fmad=false. Underflow and overflow are not handled.
#if defined(__clang__)
#pragma float_control(push)
#pragma clang fp contract(off)
#elif defined(_MSC_VER)
#pragma float_control(push)
#pragma fp_contract(off)
#endif

namespace viskores
{
namespace detail
{
namespace exact
{

using Real = viskores::Float64;

/// `x + y = a + b` exactly, with `x` the rounded sum.
VISKORES_EXEC_CONT inline void TwoSum(Real a, Real b, Real& x, Real& y)
{
  x = a + b;
  const Real bVirtual = x - a;
  const Real aVirtual = x - bVirtual;
  y = (a - aVirtual) + (b - bVirtual);
}

/// `x + y = a + b` exactly, assuming `|a| >= |b|`.
VISKORES_EXEC_CONT inline void FastTwoSum(Real a, Real b, Real& x, Real& y)
{
  x = a + b;
  y = b - (x - a);
}

/// `x + y = a - b` exactly.
VISKORES_EXEC_CONT inline void TwoDiff(Real a, Real b, Real& x, Real& y)
{
  x = a - b;
  const Real bVirtual = a - x;
  const Real aVirtual = x + bVirtual;
  y = (a - aVirtual) + (bVirtual - b);
}

/// Splits `a` into two halves of 26 significant bits each.
VISKORES_EXEC_CONT inline void Split(Real a, Real& hi, Real& lo)
{
  const Real c = Real(134217729.0) * a; // 2^27 + 1
  hi = c - (c - a);
  lo = a - hi;
}

/// `x + y = a b` exactly, with `b` already split.
VISKORES_EXEC_CONT inline void TwoProductPresplit(Real a, Real b, Real bHi, Real bLo, Real& x, Real& y)
{
  x = a * b;
  Real aHi, aLo;
  Split(a, aHi, aLo);
  const Real err1 = x - aHi * bHi;
  const Real err2 = err1 - aLo * bHi;
  const Real err3 = err2 - aHi * bLo;
  y = aLo * bLo - err3;
}

/// `h = e + b` for a nonoverlapping expansion `e` in increasing order of magnitude; zero
/// components are dropped. Returns the length of `h`, at most `eLength + 1`.
VISKORES_EXEC_CONT inline int GrowExpansion(int eLength, const Real* e, Real b, Real* h)
{
  Real q = b;
  int hLength = 0;
  for (int i = 0; i < eLength; ++i)
  {
    Real sum, error;
    TwoSum(q, e[i], sum, error);
    q = sum;
    if (error != 0)
    {
      h[hLength++] = error;
    }
  }
  if (q != 0 || hLength == 0)
  {
    h[hLength++] = q;
  }
  return hLength;
}

/// `h = e + f`, growing `e` by every component of `f`. `h` and `scratch` must hold
/// `eLength + fLength` components and may not alias `e` or `f`.
VISKORES_EXEC_CONT inline int ExpansionSum(int eLength,
                                           const Real* e,
                                           int fLength,
                                           const Real* f,
                                           Real* h,
                                           Real* scratch)
{
  int length = eLength;
  for (int i = 0; i < eLength; ++i)
  {
    h[i] = e[i];
  }
  for (int i = 0; i < fLength; ++i)
  {
    length = GrowExpansion(length, h, f[i], scratch);
    for (int j = 0; j < length; ++j)
    {
      h[j] = scratch[j];
    }
  }
  return length;
}

/// `h = b e`. `h` must hold `2 eLength` components.
VISKORES_EXEC_CONT inline int ScaleExpansion(int eLength, const Real* e, Real b, Real* h)
{
  Real bHi, bLo;
  Split(b, bHi, bLo);
  Real q, error;
  TwoProductPresplit(e[0], b, bHi, bLo, q, error);
  int hLength = 0;
  if (error != 0)
  {
    h[hLength++] = error;
  }
  for (int i = 1; i < eLength; ++i)
  {
    Real product1, product0, sum;
    TwoProductPresplit(e[i], b, bHi, bLo, product1, product0);
    TwoSum(q, product0, sum, error);
    if (error != 0)
    {
      h[hLength++] = error;
    }
    FastTwoSum(product1, sum, q, error);
    if (error != 0)
    {
      h[hLength++] = error;
    }
  }
  if (q != 0 || hLength == 0)
  {
    h[hLength++] = q;
  }
  return hLength;
}

/// `h = e f`. `h` must hold `2 eLength fLength` components, at most `Capacity`.
template <int Capacity>
VISKORES_EXEC_CONT inline int MultiplyExpansions(int eLength,
                                                 const Real* e,
                                                 int fLength,
                                                 const Real* f,
                                                 Real* h)
{
  Real part[Capacity], sum[Capacity], scratch[Capacity];
  int length = 0;
  for (int i = 0; i < fLength; ++i)
  {
    const int partLength = ScaleExpansion(eLength, e, f[i], part);
    if (length == 0)
    {
      for (int j = 0; j < partLength; ++j)
      {
        h[j] = part[j];
      }
      length = partLength;
    }
    else
    {
      length = ExpansionSum(length, h, partLength, part, sum, scratch);
      for (int j = 0; j < length; ++j)
      {
        h[j] = sum[j];
      }
    }
  }
  return length;
}

VISKORES_EXEC_CONT inline viskores::IdComponent Sign(Real x)
{
  return (x > 0) ? 1 : ((x < 0) ? -1 : 0);
}

/// The exact difference `a - b` as an expansion of at most two components.
VISKORES_EXEC_CONT inline int Difference(Real a, Real b, Real* h)
{
  Real x, y;
  TwoDiff(a, b, x, y);
  int length = 0;
  if (y != 0)
  {
    h[length++] = y;
  }
  if (x != 0 || length == 0)
  {
    h[length++] = x;
  }
  return length;
}

/// `h = a b - c d` for expansions of at most two components; at most 16 components.
VISKORES_EXEC_CONT inline int CrossTerm(int aLength,
                                        const Real* a,
                                        int bLength,
                                        const Real* b,
                                        int cLength,
                                        const Real* c,
                                        int dLength,
                                        const Real* d,
                                        Real* h)
{
  Real ab[8], cd[8], scratch[16];
  const int abLength = MultiplyExpansions<8>(aLength, a, bLength, b, ab);
  const int cdLength = MultiplyExpansions<8>(cLength, c, dLength, d, cd);
  for (int i = 0; i < cdLength; ++i)
  {
    cd[i] = -cd[i];
  }
  return ExpansionSum(abLength, ab, cdLength, cd, h, scratch);
}

constexpr Real UnitRoundoff = 1.1102230246251565404236316680908203125e-16; // 2^-53

} // namespace exact
} // namespace detail

/// @brief The sign of the orientation of three points in the plane, evaluated exactly.
///
/// Returns 1 if `a`, `b`, `c` turn counterclockwise, -1 if clockwise and 0 if they are
/// collinear: the sign of `(a - c) x (b - c)`.
VISKORES_EXEC_CONT inline viskores::IdComponent Orient2D(const viskores::Vec<viskores::Float64, 2>& a,
                                                         const viskores::Vec<viskores::Float64, 2>& b,
                                                         const viskores::Vec<viskores::Float64, 2>& c)
{
  using namespace viskores::detail::exact;
  const Real detLeft = (a[0] - c[0]) * (b[1] - c[1]);
  const Real detRight = (a[1] - c[1]) * (b[0] - c[0]);
  const Real det = detLeft - detRight;
  Real detSum;
  if (detLeft > 0)
  {
    if (detRight <= 0)
    {
      return Sign(det);
    }
    detSum = detLeft + detRight;
  }
  else if (detLeft < 0)
  {
    if (detRight >= 0)
    {
      return Sign(det);
    }
    detSum = -detLeft - detRight;
  }
  else
  {
    return Sign(det);
  }
  const Real bound = (Real(3) + Real(16) * UnitRoundoff) * UnitRoundoff * detSum;
  if (det >= bound || -det >= bound)
  {
    return Sign(det);
  }
  Real acx[2], bcy[2], acy[2], bcx[2], h[16];
  const int acxLength = Difference(a[0], c[0], acx);
  const int bcyLength = Difference(b[1], c[1], bcy);
  const int acyLength = Difference(a[1], c[1], acy);
  const int bcxLength = Difference(b[0], c[0], bcx);
  const int length = CrossTerm(acxLength, acx, bcyLength, bcy, acyLength, acy, bcxLength, bcx, h);
  return Sign(h[length - 1]);
}

/// @brief The sign of the orientation of four points in space, evaluated exactly.
///
/// Returns the sign of the determinant of the rows `a - d`, `b - d`, `c - d`: positive if `d`
/// lies below the plane through `a`, `b`, `c` seen from the side where they turn
/// counterclockwise, and 0 if and only if the four points are coplanar.
VISKORES_EXEC_CONT inline viskores::IdComponent Orient3D(const viskores::Vec3f_64& a,
                                                         const viskores::Vec3f_64& b,
                                                         const viskores::Vec3f_64& c,
                                                         const viskores::Vec3f_64& d)
{
  using namespace viskores::detail::exact;
  const Real adx = a[0] - d[0], bdx = b[0] - d[0], cdx = c[0] - d[0];
  const Real ady = a[1] - d[1], bdy = b[1] - d[1], cdy = c[1] - d[1];
  const Real adz = a[2] - d[2], bdz = b[2] - d[2], cdz = c[2] - d[2];
  const Real bdxcdy = bdx * cdy, cdxbdy = cdx * bdy;
  const Real cdxady = cdx * ady, adxcdy = adx * cdy;
  const Real adxbdy = adx * bdy, bdxady = bdx * ady;
  const Real det = adz * (bdxcdy - cdxbdy) + bdz * (cdxady - adxcdy) + cdz * (adxbdy - bdxady);
  const Real permanent = (viskores::Abs(bdxcdy) + viskores::Abs(cdxbdy)) * viskores::Abs(adz) +
    (viskores::Abs(cdxady) + viskores::Abs(adxcdy)) * viskores::Abs(bdz) +
    (viskores::Abs(adxbdy) + viskores::Abs(bdxady)) * viskores::Abs(cdz);
  const Real bound = (Real(7) + Real(56) * UnitRoundoff) * UnitRoundoff * permanent;
  if (det > bound || -det > bound)
  {
    return Sign(det);
  }
  // det = adz (bdx cdy - cdx bdy) + bdz (cdx ady - adx cdy) + cdz (adx bdy - bdx ady), exactly.
  Real exactAdx[2], exactBdx[2], exactCdx[2], exactAdy[2], exactBdy[2], exactCdy[2];
  Real exactAdz[2], exactBdz[2], exactCdz[2];
  const int adxLength = Difference(a[0], d[0], exactAdx);
  const int bdxLength = Difference(b[0], d[0], exactBdx);
  const int cdxLength = Difference(c[0], d[0], exactCdx);
  const int adyLength = Difference(a[1], d[1], exactAdy);
  const int bdyLength = Difference(b[1], d[1], exactBdy);
  const int cdyLength = Difference(c[1], d[1], exactCdy);
  const int adzLength = Difference(a[2], d[2], exactAdz);
  const int bdzLength = Difference(b[2], d[2], exactBdz);
  const int cdzLength = Difference(c[2], d[2], exactCdz);
  Real bc[16], ca[16], ab[16];
  const int bcLength = CrossTerm(
    bdxLength, exactBdx, cdyLength, exactCdy, cdxLength, exactCdx, bdyLength, exactBdy, bc);
  const int caLength = CrossTerm(
    cdxLength, exactCdx, adyLength, exactAdy, adxLength, exactAdx, cdyLength, exactCdy, ca);
  const int abLength = CrossTerm(
    adxLength, exactAdx, bdyLength, exactBdy, bdxLength, exactBdx, adyLength, exactAdy, ab);
  Real termA[64], termB[64], termC[64];
  const int aLength = MultiplyExpansions<64>(bcLength, bc, adzLength, exactAdz, termA);
  const int bLength = MultiplyExpansions<64>(caLength, ca, bdzLength, exactBdz, termB);
  const int cLength = MultiplyExpansions<64>(abLength, ab, cdzLength, exactCdz, termC);
  Real abSum[128], all[192], scratch[192];
  const int abSumLength = ExpansionSum(aLength, termA, bLength, termB, abSum, scratch);
  const int length = ExpansionSum(abSumLength, abSum, cLength, termC, all, scratch);
  return Sign(all[length - 1]);
}

/// `Orient2D` of the projections of three points onto the coordinate plane that leaves out
/// `axis`.
VISKORES_EXEC_CONT inline viskores::IdComponent Orient2DProjected(const viskores::Vec3f_64& a,
                                                                  const viskores::Vec3f_64& b,
                                                                  const viskores::Vec3f_64& c,
                                                                  viskores::IdComponent axis)
{
  const viskores::IdComponent u = (axis + 1) % 3;
  const viskores::IdComponent v = (axis + 2) % 3;
  return Orient2D(viskores::Vec<viskores::Float64, 2>(a[u], a[v]),
                  viskores::Vec<viskores::Float64, 2>(b[u], b[v]),
                  viskores::Vec<viskores::Float64, 2>(c[u], c[v]));
}

/// @brief Whether the triangle `abc` is degenerate, exactly: its three vertices are collinear.
VISKORES_EXEC_CONT inline bool TriangleIsDegenerate(const viskores::Vec3f_64& a,
                                                    const viskores::Vec3f_64& b,
                                                    const viskores::Vec3f_64& c)
{
  return Orient2DProjected(a, b, c, 0) == 0 && Orient2DProjected(a, b, c, 1) == 0 &&
    Orient2DProjected(a, b, c, 2) == 0;
}

/// @brief A coordinate axis along which the projection of the nondegenerate triangle `abc` is
/// nondegenerate, trying the largest component of its normal first.
VISKORES_EXEC_CONT inline viskores::IdComponent TriangleProjectionAxis(const viskores::Vec3f_64& a,
                                                                       const viskores::Vec3f_64& b,
                                                                       const viskores::Vec3f_64& c)
{
  const viskores::Vec3f_64 ab = b - a, ac = c - a;
  const viskores::Vec3f_64 normal(ab[1] * ac[2] - ab[2] * ac[1],
                                  ab[2] * ac[0] - ab[0] * ac[2],
                                  ab[0] * ac[1] - ab[1] * ac[0]);
  viskores::IdComponent axis = 0;
  for (viskores::IdComponent i = 1; i < 3; ++i)
  {
    if (viskores::Abs(normal[i]) > viskores::Abs(normal[axis]))
    {
      axis = i;
    }
  }
  for (viskores::IdComponent i = 0; i < 3; ++i)
  {
    const viskores::IdComponent candidate = (axis + i) % 3;
    if (Orient2DProjected(a, b, c, candidate) != 0)
    {
      return candidate;
    }
  }
  return axis;
}

namespace detail
{
namespace exact
{
/// Whether `p`, collinear with `a` and `b` in the projection leaving out `axis`, lies on the
/// closed segment `ab`.
VISKORES_EXEC_CONT inline bool OnSegment(const viskores::Vec3f_64& a,
                                         const viskores::Vec3f_64& b,
                                         const viskores::Vec3f_64& p,
                                         viskores::IdComponent axis)
{
  for (viskores::IdComponent i = 1; i <= 2; ++i)
  {
    const viskores::IdComponent k = (axis + i) % 3;
    if (p[k] < viskores::Min(a[k], b[k]) || p[k] > viskores::Max(a[k], b[k]))
    {
      return false;
    }
  }
  return true;
}

/// Whether the closed segments `ab` and `cd` of one plane meet, in the projection leaving out
/// `axis`.
VISKORES_EXEC_CONT inline bool SegmentsMeet2D(const viskores::Vec3f_64& a,
                                              const viskores::Vec3f_64& b,
                                              const viskores::Vec3f_64& c,
                                              const viskores::Vec3f_64& d,
                                              viskores::IdComponent axis)
{
  const viskores::IdComponent d1 = Orient2DProjected(a, b, c, axis);
  const viskores::IdComponent d2 = Orient2DProjected(a, b, d, axis);
  const viskores::IdComponent d3 = Orient2DProjected(c, d, a, axis);
  const viskores::IdComponent d4 = Orient2DProjected(c, d, b, axis);
  if (d1 * d2 < 0 && d3 * d4 < 0)
  {
    return true;
  }
  return (d1 == 0 && OnSegment(a, b, c, axis)) || (d2 == 0 && OnSegment(a, b, d, axis)) ||
    (d3 == 0 && OnSegment(c, d, a, axis)) || (d4 == 0 && OnSegment(c, d, b, axis));
}
} // namespace exact
} // namespace detail

/// @brief Whether the closed segment `xy` meets the closed nondegenerate triangle `q0 q1 q2`,
/// evaluated exactly.
VISKORES_EXEC_CONT inline bool SegmentIntersectsTriangle(const viskores::Vec3f_64& x,
                                                         const viskores::Vec3f_64& y,
                                                         const viskores::Vec3f_64& q0,
                                                         const viskores::Vec3f_64& q1,
                                                         const viskores::Vec3f_64& q2)
{
  const viskores::IdComponent ox = Orient3D(q0, q1, q2, x);
  const viskores::IdComponent oy = Orient3D(q0, q1, q2, y);
  if (ox * oy > 0)
  {
    return false;
  }
  if (ox == 0 && oy == 0)
  {
    // Coplanar: in a projection where the triangle keeps its area.
    const viskores::IdComponent axis = TriangleProjectionAxis(q0, q1, q2);
    const viskores::IdComponent s = Orient2DProjected(q0, q1, q2, axis);
    auto inside = [&](const viskores::Vec3f_64& p)
    {
      return Orient2DProjected(q0, q1, p, axis) * s >= 0 &&
        Orient2DProjected(q1, q2, p, axis) * s >= 0 && Orient2DProjected(q2, q0, p, axis) * s >= 0;
    };
    return inside(x) || inside(y) || detail::exact::SegmentsMeet2D(x, y, q0, q1, axis) ||
      detail::exact::SegmentsMeet2D(x, y, q1, q2, axis) ||
      detail::exact::SegmentsMeet2D(x, y, q2, q0, axis);
  }
  // The line xy crosses the plane at one point of the segment; it lies in the triangle if the
  // line passes on the same side of the three edges.
  const viskores::IdComponent o0 = Orient3D(x, y, q0, q1);
  const viskores::IdComponent o1 = Orient3D(x, y, q1, q2);
  const viskores::IdComponent o2 = Orient3D(x, y, q2, q0);
  return (o0 >= 0 && o1 >= 0 && o2 >= 0) || (o0 <= 0 && o1 <= 0 && o2 <= 0);
}

/// @brief Whether two closed nondegenerate triangles meet, evaluated exactly: whether an edge of
/// one meets the other.
VISKORES_EXEC_CONT inline bool TrianglesIntersect(const viskores::Vec3f_64& p0,
                                                  const viskores::Vec3f_64& p1,
                                                  const viskores::Vec3f_64& p2,
                                                  const viskores::Vec3f_64& q0,
                                                  const viskores::Vec3f_64& q1,
                                                  const viskores::Vec3f_64& q2)
{
  return SegmentIntersectsTriangle(p0, p1, q0, q1, q2) ||
    SegmentIntersectsTriangle(p1, p2, q0, q1, q2) || SegmentIntersectsTriangle(p2, p0, q0, q1, q2) ||
    SegmentIntersectsTriangle(q0, q1, p0, p1, p2) || SegmentIntersectsTriangle(q1, q2, p0, p1, p2) ||
    SegmentIntersectsTriangle(q2, q0, p0, p1, p2);
}

} // namespace viskores

#if defined(__clang__) || defined(_MSC_VER)
#pragma float_control(pop)
#endif

#endif // viskores_ExactPredicates_h
