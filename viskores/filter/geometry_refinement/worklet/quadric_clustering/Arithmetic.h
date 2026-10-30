//============================================================================
//  The contents of this file are covered by the Viskores license. See
//  LICENSE.txt for details.
//
//  By contributing to this file, all contributors agree to the Developer
//  Certificate of Origin Version 1.1 (DCO 1.1) as stated in DCO.txt.
//============================================================================
#ifndef viskores_worklet_quadric_clustering_Arithmetic_h
#define viskores_worklet_quadric_clustering_Arithmetic_h

#include <viskores/Math.h>
#include <viskores/Types.h>

#include <cstring>
#include <limits>

// Every decision of the algorithm compares floating-point values, and the output depends on
// their rounding (Proposition "Determinism" of the paper), so every quantity that feeds a
// decision is evaluated with a fixed sequence of IEEE operations. Fused multiply-add contraction
// changes the rounding, so the headers of this algorithm turn it off between these two macros.
// GCC contracts only in its GNU dialects, and Viskores compiles in ISO mode. CUDA device code
// needs --fmad=false for the same effect.
#if defined(__clang__)
#define VISKORES_QUADRIC_CLUSTERING_STRICT_FP_BEGIN \
  _Pragma("float_control(push)") _Pragma("clang fp contract(off)")
#define VISKORES_QUADRIC_CLUSTERING_STRICT_FP_END _Pragma("float_control(pop)")
#elif defined(_MSC_VER)
#define VISKORES_QUADRIC_CLUSTERING_STRICT_FP_BEGIN \
  __pragma(float_control(push)) __pragma(fp_contract(off))
#define VISKORES_QUADRIC_CLUSTERING_STRICT_FP_END __pragma(float_control(pop))
#else
#define VISKORES_QUADRIC_CLUSTERING_STRICT_FP_BEGIN
#define VISKORES_QUADRIC_CLUSTERING_STRICT_FP_END
#endif

VISKORES_QUADRIC_CLUSTERING_STRICT_FP_BEGIN

namespace viskores
{
namespace worklet
{
namespace quadric_clustering
{

/// The scalar type of every computation, whatever the type of the input coordinates.
using Real = viskores::Float64;
using Vec3 = viskores::Vec<Real, 3>;
/// A 3x3 matrix in row-major order. The matrices are stored in full: their two triangles are
/// accumulated separately and need not agree in the last bit.
using Mat3 = viskores::Vec<Real, 9>;

/// The functions below evaluate their operations in the order written, summing left to right,
/// so that results are reproducible bit for bit. `viskores::Dot` and friends make no such promise.

VISKORES_EXEC_CONT inline Vec3 Cross(const Vec3& a, const Vec3& b)
{
  return Vec3(a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]);
}

VISKORES_EXEC_CONT inline Vec3 Sub(const Vec3& a, const Vec3& b)
{
  return Vec3(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
}

VISKORES_EXEC_CONT inline Real Dot(const Vec3& a, const Vec3& b)
{
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

VISKORES_EXEC_CONT inline Real Norm(const Vec3& a)
{
  return viskores::Sqrt(Dot(a, a));
}

VISKORES_EXEC_CONT inline Vec3 Add(const Vec3& a, const Vec3& b)
{
  return Vec3(a[0] + b[0], a[1] + b[1], a[2] + b[2]);
}

VISKORES_EXEC_CONT inline Mat3 MatAdd(const Mat3& a, const Mat3& b)
{
  Mat3 c;
  for (viskores::IdComponent i = 0; i < 9; ++i)
  {
    c[i] = a[i] + b[i];
  }
  return c;
}

/// `M x`, each row summed left to right.
VISKORES_EXEC_CONT inline Vec3 MatVec(const Mat3& m, const Vec3& x)
{
  return Vec3(m[0] * x[0] + m[1] * x[1] + m[2] * x[2],
              m[3] * x[0] + m[4] * x[1] + m[5] * x[2],
              m[6] * x[0] + m[7] * x[1] + m[8] * x[2]);
}

/// `x^T M x`.
VISKORES_EXEC_CONT inline Real Quad(const Vec3& x, const Mat3& m)
{
  return Dot(x, MatVec(m, x));
}

/// The sum of a sequence by a fixed tree: each level adds adjacent pairs `(x0 + x1, x2 + x3,
/// ...)` and carries an odd last term up, until one value is left; zero when empty. Computed as
/// a binary counter: equal blocks are added as they complete, and `Result` adds the remaining
/// blocks from the right, which gives the same tree for every length.
template <typename T>
class TreeSum
{
public:
  VISKORES_EXEC_CONT void Push(const T& x)
  {
    this->Blocks[this->Top] = x;
    this->Levels[this->Top] = 0;
    ++this->Top;
    while (this->Top > 1 && this->Levels[this->Top - 1] == this->Levels[this->Top - 2])
    {
      this->Blocks[this->Top - 2] = this->Blocks[this->Top - 2] + this->Blocks[this->Top - 1];
      ++this->Levels[this->Top - 2];
      --this->Top;
    }
  }

  VISKORES_EXEC_CONT T Result() const
  {
    if (this->Top == 0)
    {
      return T(0);
    }
    T sum = this->Blocks[this->Top - 1];
    for (viskores::IdComponent i = this->Top - 2; i >= 0; --i)
    {
      sum = this->Blocks[i] + sum;
    }
    return sum;
  }

private:
  // Enough for 2^32 - 1 terms.
  static constexpr viskores::IdComponent Depth = 32;
  T Blocks[Depth];
  viskores::IdComponent Levels[Depth];
  viskores::IdComponent Top = 0;
};

/// Solves `Q x = b` for a symmetric positive definite `Q` by an explicit Cholesky factorisation
/// of its lower triangle and two triangular solves.
VISKORES_EXEC_CONT inline Vec3 SolveSPD(const Mat3& q, const Vec3& b)
{
  const Real l00 = viskores::Sqrt(q[0]);
  const Real l10 = q[3] / l00;
  const Real l20 = q[6] / l00;
  const Real l11 = viskores::Sqrt(q[4] - l10 * l10);
  const Real l21 = (q[7] - l20 * l10) / l11;
  const Real l22 = viskores::Sqrt(q[8] - l20 * l20 - l21 * l21);
  const Real y0 = b[0] / l00;
  const Real y1 = (b[1] - l10 * y0) / l11;
  const Real y2 = (b[2] - l20 * y0 - l21 * y1) / l22;
  const Real x2 = y2 / l22;
  const Real x1 = (y1 - l21 * x2) / l11;
  const Real x0 = (y0 - l10 * x1 - l20 * x2) / l00;
  return Vec3(x0, x1, x2);
}

/// The smallest double above a finite `x`: a result computed to nearest, rounded up, bounds
/// the exact result from above.
VISKORES_EXEC_CONT inline Real NextUp(Real x)
{
  if (!(x < std::numeric_limits<Real>::infinity()))
  {
    return x;
  }
  if (x == 0)
  {
    return std::numeric_limits<Real>::denorm_min();
  }
  viskores::UInt64 bits;
  std::memcpy(&bits, &x, sizeof(bits));
  bits = (x > 0) ? bits + 1 : bits - 1;
  std::memcpy(&x, &bits, sizeof(bits));
  return x;
}

/// The unit roundoff of double precision.
constexpr Real UnitRoundoff = 1.1102230246251565404236316680908203125e-16;

/// An upper bound `(1 + 4u) l`, rounded up, on the exact length whose computed value is `l`
/// (Lemma "One-sided evaluation" (ii)).
VISKORES_EXEC_CONT inline Real LengthUp(Real length)
{
  return NextUp((1 + 4 * UnitRoundoff) * length);
}

/// The finaliser of SplitMix64, a bijective hash of a 64-bit integer: the default key of the
/// fixed order on vertices used by the commit step of the assignment.
VISKORES_EXEC_CONT inline viskores::UInt64 HashIndex(viskores::Id index)
{
  viskores::UInt64 z = static_cast<viskores::UInt64>(index) + 0x9E3779B97F4A7C15ULL;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  return z ^ (z >> 31);
}

}
}
} // namespace viskores::worklet::quadric_clustering

VISKORES_QUADRIC_CLUSTERING_STRICT_FP_END

#endif // viskores_worklet_quadric_clustering_Arithmetic_h
