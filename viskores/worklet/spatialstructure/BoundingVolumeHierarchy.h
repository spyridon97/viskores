//============================================================================
//  The contents of this file are covered by the Viskores license. See
//  LICENSE.txt for details.
//
//  By contributing to this file, all contributors agree to the Developer
//  Certificate of Origin Version 1.1 (DCO 1.1) as stated in DCO.txt.
//============================================================================
#ifndef viskores_worklet_spatialstructure_BoundingVolumeHierarchy_h
#define viskores_worklet_spatialstructure_BoundingVolumeHierarchy_h

#include <viskores/BinaryOperators.h>
#include <viskores/Math.h>
#include <viskores/Pair.h>
#include <viskores/Types.h>
#include <viskores/cont/Algorithm.h>
#include <viskores/cont/ArrayHandle.h>
#include <viskores/cont/ArrayHandleIndex.h>
#include <viskores/cont/ExecutionObjectBase.h>
#include <viskores/cont/Invoker.h>
#include <viskores/worklet/WorkletMapField.h>

#include <limits>

namespace viskores
{
namespace worklet
{
namespace spatialstructure
{

/// @brief The execution-side view of a `BoundingVolumeHierarchy`, for queries inside worklets.
///
/// Nodes `0, ..., n - 2` are internal, with node 0 the root, and nodes `n - 1, ..., 2 n - 2`
/// are the leaves in Morton order.
template <typename VecPortal, typename Id2Portal, typename IdPortal>
struct BoundingVolumeHierarchyPortals
{
  using Vec3 = viskores::Vec3f_64;

  /// Traversals use a stack of this depth, which bounds the depth of the tree: 63 bits of
  /// Morton code and 64 bits of index.
  static constexpr int StackSize = 160;

  viskores::Id NumberOfLeaves;
  VecPortal Lower;
  VecPortal Upper;
  Id2Portal Children;
  IdPortal LeafPrimitives;

  VISKORES_EXEC bool IsLeaf(viskores::Id node) const { return node >= this->NumberOfLeaves - 1; }

  VISKORES_EXEC viskores::Id LeafPrimitive(viskores::Id node) const
  {
    return this->LeafPrimitives.Get(node - (this->NumberOfLeaves - 1));
  }

  /// Depth-first traversal of the nodes accepted by `accept(node)`, calling `visit(primitive)`
  /// at the accepted leaves until it returns false. Returns false if the stack overflowed.
  template <typename Accept, typename Visit>
  VISKORES_EXEC bool Traverse(Accept&& accept, Visit&& visit) const
  {
    if (this->NumberOfLeaves == 0)
    {
      return true;
    }
    viskores::Id stack[StackSize];
    int top = 0;
    stack[top++] = 0;
    while (top > 0)
    {
      const viskores::Id node = stack[--top];
      if (!accept(node))
      {
        continue;
      }
      if (this->IsLeaf(node))
      {
        if (!visit(this->LeafPrimitive(node)))
        {
          return true;
        }
        continue;
      }
      if (top + 2 > StackSize)
      {
        return false;
      }
      const viskores::Id2 c = this->Children.Get(node);
      stack[top++] = c[1];
      stack[top++] = c[0];
    }
    return true;
  }

  /// Calls `visit(primitive)` for every primitive whose box meets the closed box `[lo, hi]`,
  /// until it returns false. Returns false if the stack overflowed.
  template <typename Visit>
  VISKORES_EXEC bool ForEachOverlap(const Vec3& lo, const Vec3& hi, Visit&& visit) const
  {
    return this->Traverse(
      [&](viskores::Id node)
      {
        const Vec3 l = this->Lower.Get(node);
        const Vec3 u = this->Upper.Get(node);
        return l[0] <= hi[0] && u[0] >= lo[0] && l[1] <= hi[1] && u[1] >= lo[1] &&
          l[2] <= hi[2] && u[2] >= lo[2];
      },
      visit);
  }

  /// A lower bound on the distance from `p` to the box of `node`, safe against the rounding of
  /// its evaluation.
  VISKORES_EXEC viskores::Float64 BoxDistanceLowerBound(const Vec3& p, viskores::Id node) const
  {
    const Vec3 lo = this->Lower.Get(node);
    const Vec3 hi = this->Upper.Get(node);
    viskores::Float64 sum = 0;
    for (int i = 0; i < 3; ++i)
    {
      const viskores::Float64 d =
        viskores::Max(viskores::Max(lo[i] - p[i], p[i] - hi[i]), viskores::Float64(0));
      sum = sum + d * d;
    }
    return viskores::Sqrt(sum) * (1 - 16 * std::numeric_limits<viskores::Float64>::epsilon() / 2);
  }

  /// @brief The smallest `distance(primitive)` over all primitives.
  ///
  /// `distance(primitive)` must not be below the Euclidean distance from `p` to the box of the
  /// primitive. A subtree is skipped only when the distance to its box exceeds the best value
  /// found, so the result is the exact minimum over all primitives, independent of the shape of
  /// the tree. Nearer children are visited first. `complete` is false if the stack overflowed.
  ///
  /// With a finite `cutoff`, subtrees farther than it are skipped too: the result is the exact
  /// minimum when that is at most `cutoff`, and otherwise some value above `cutoff`, possibly
  /// infinity. Queries that only compare the distance with a threshold run faster this way.
  template <typename Distance>
  VISKORES_EXEC viskores::Float64 FindNearest(
    const Vec3& p,
    Distance&& distance,
    bool& complete,
    viskores::Float64 cutoff = std::numeric_limits<viskores::Float64>::infinity()) const
  {
    viskores::Float64 best = std::numeric_limits<viskores::Float64>::infinity();
    complete = true;
    if (this->NumberOfLeaves == 0)
    {
      return best;
    }
    viskores::Id stack[StackSize];
    int top = 0;
    stack[top++] = 0;
    while (top > 0)
    {
      const viskores::Id node = stack[--top];
      if (this->BoxDistanceLowerBound(p, node) > viskores::Min(best, cutoff))
      {
        continue;
      }
      if (this->IsLeaf(node))
      {
        best = viskores::Min(best, distance(this->LeafPrimitive(node)));
        continue;
      }
      if (top + 2 > StackSize)
      {
        complete = false;
        return best;
      }
      const viskores::Id2 c = this->Children.Get(node);
      if (this->BoxDistanceLowerBound(p, c[0]) <= this->BoxDistanceLowerBound(p, c[1]))
      {
        stack[top++] = c[1];
        stack[top++] = c[0];
      }
      else
      {
        stack[top++] = c[0];
        stack[top++] = c[1];
      }
    }
    return best;
  }
};

namespace detail
{

VISKORES_EXEC_CONT inline int LeadingZeros64(viskores::UInt64 x)
{
  if (x == 0)
  {
    return 64;
  }
  int n = 0;
  for (int shift = 32; shift > 0; shift /= 2)
  {
    if ((x >> (64 - shift)) == 0)
    {
      n += shift;
      x <<= shift;
    }
  }
  return n;
}

/// The 63-bit Morton code of the centre of a box, with the box index.
struct BoxMortonCode : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn lower, FieldIn upper, FieldOut code);
  using ExecutionSignature = _3(_1, _2, WorkIndex);

  VISKORES_CONT BoxMortonCode(const viskores::Vec3f_64& minimum, const viskores::Vec3f_64& scale)
    : Minimum(minimum)
    , Scale(scale)
  {
  }

  VISKORES_EXEC viskores::Pair<viskores::UInt64, viskores::Id> operator()(
    const viskores::Vec3f_64& lower,
    const viskores::Vec3f_64& upper,
    viskores::Id index) const
  {
    viskores::UInt64 code = 0;
    for (viskores::IdComponent axis = 0; axis < 3; ++axis)
    {
      const viskores::Float64 centre = (lower[axis] + upper[axis]) / 2;
      viskores::Float64 scaled = (centre - this->Minimum[axis]) * this->Scale[axis];
      scaled = viskores::Max(viskores::Float64(0), viskores::Min(scaled, viskores::Float64(2097151)));
      const viskores::UInt64 q = static_cast<viskores::UInt64>(scaled);
      for (viskores::IdComponent bit = 0; bit < 21; ++bit)
      {
        code |= ((q >> bit) & 1ULL) << (3 * bit + (2 - axis));
      }
    }
    return viskores::Pair<viskores::UInt64, viskores::Id>(code, index);
  }

  viskores::Vec3f_64 Minimum;
  viskores::Vec3f_64 Scale;
};

struct SplitCode : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn pair, FieldOut code, FieldOut primitive);
  using ExecutionSignature = void(_1, _2, _3);

  VISKORES_EXEC void operator()(const viskores::Pair<viskores::UInt64, viskores::Id>& pair,
                                viskores::UInt64& code,
                                viskores::Id& primitive) const
  {
    code = pair.first;
    primitive = pair.second;
  }
};

/// Karras's construction of internal node `i`: its range of leaves, its split, its children
/// and their parents. Equal codes are ordered by leaf index.
struct BuildInternalNodes : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn index, WholeArrayIn codes, FieldOut children, WholeArrayOut parents);
  using ExecutionSignature = void(_1, _2, _3, _4);

  VISKORES_CONT explicit BuildInternalNodes(viskores::Id numLeaves)
    : NumberOfLeaves(numLeaves)
  {
  }

  template <typename CodesPortal>
  VISKORES_EXEC int Delta(const CodesPortal& codes, viskores::Id i, viskores::Id j) const
  {
    if (j < 0 || j >= this->NumberOfLeaves)
    {
      return -1;
    }
    const viskores::UInt64 a = codes.Get(i);
    const viskores::UInt64 b = codes.Get(j);
    if (a == b)
    {
      return 64 + LeadingZeros64(static_cast<viskores::UInt64>(i) ^ static_cast<viskores::UInt64>(j));
    }
    return LeadingZeros64(a ^ b);
  }

  template <typename CodesPortal, typename ParentsPortal>
  VISKORES_EXEC void operator()(viskores::Id i,
                                const CodesPortal& codes,
                                viskores::Id2& children,
                                ParentsPortal& parents) const
  {
    const viskores::Id d =
      (this->Delta(codes, i, i + 1) - this->Delta(codes, i, i - 1)) >= 0 ? 1 : -1;
    const int deltaMin = this->Delta(codes, i, i - d);
    viskores::Id lengthMax = 2;
    while (this->Delta(codes, i, i + lengthMax * d) > deltaMin)
    {
      lengthMax *= 2;
    }
    viskores::Id length = 0;
    for (viskores::Id t = lengthMax / 2; t >= 1; t /= 2)
    {
      if (this->Delta(codes, i, i + (length + t) * d) > deltaMin)
      {
        length += t;
      }
    }
    const viskores::Id j = i + length * d;
    const int deltaNode = this->Delta(codes, i, j);
    viskores::Id split = 0;
    for (viskores::Id divisor = 2;; divisor *= 2)
    {
      const viskores::Id t = (length + divisor - 1) / divisor;
      if (this->Delta(codes, i, i + (split + t) * d) > deltaNode)
      {
        split += t;
      }
      if (t <= 1)
      {
        break;
      }
    }
    const viskores::Id gamma = i + split * d + viskores::Min(d, viskores::Id(0));
    const viskores::Id leafBase = this->NumberOfLeaves - 1;
    const viskores::Id left = (viskores::Min(i, j) == gamma) ? leafBase + gamma : gamma;
    const viskores::Id right =
      (viskores::Max(i, j) == gamma + 1) ? leafBase + gamma + 1 : gamma + 1;
    children = viskores::Id2(left, right);
    parents.Set(left, i);
    parents.Set(right, i);
  }

  viskores::Id NumberOfLeaves;
};

/// Sets the box of every leaf, then walks to the root: the second child to arrive at a node
/// computes its box. Unions of boxes are exact, so the boxes do not depend on the order.
struct PropagateBoxes : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn leaf,
                                FieldIn primitive,
                                WholeArrayIn primitiveLower,
                                WholeArrayIn primitiveUpper,
                                WholeArrayIn parents,
                                WholeArrayIn children,
                                WholeArrayInOut lower,
                                WholeArrayInOut upper,
                                AtomicArrayInOut visits);
  using ExecutionSignature = void(_1, _2, _3, _4, _5, _6, _7, _8, _9);

  template <typename VecPortal, typename IdPortal, typename Id2Portal, typename OutPortal, typename Atomic>
  VISKORES_EXEC void operator()(viskores::Id leaf,
                                viskores::Id primitive,
                                const VecPortal& primitiveLower,
                                const VecPortal& primitiveUpper,
                                const IdPortal& parents,
                                const Id2Portal& children,
                                OutPortal& lower,
                                OutPortal& upper,
                                Atomic& visits) const
  {
    lower.Set(leaf, primitiveLower.Get(primitive));
    upper.Set(leaf, primitiveUpper.Get(primitive));
    viskores::Id node = parents.Get(leaf);
    while (node >= 0)
    {
      if (visits.Add(node, 1) == 0)
      {
        return; // the other child is not done yet
      }
      const viskores::Id2 c = children.Get(node);
      const viskores::Vec3f_64 lo0 = lower.Get(c[0]), lo1 = lower.Get(c[1]);
      const viskores::Vec3f_64 hi0 = upper.Get(c[0]), hi1 = upper.Get(c[1]);
      lower.Set(node,
                viskores::Vec3f_64(viskores::Min(lo0[0], lo1[0]),
                                   viskores::Min(lo0[1], lo1[1]),
                                   viskores::Min(lo0[2], lo1[2])));
      upper.Set(node,
                viskores::Vec3f_64(viskores::Max(hi0[0], hi1[0]),
                                   viskores::Max(hi0[1], hi1[1]),
                                   viskores::Max(hi0[2], hi1[2])));
      node = parents.Get(node);
    }
  }
};

struct LeafNode : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn index, FieldOut leaf);
  using ExecutionSignature = _2(_1);

  VISKORES_CONT explicit LeafNode(viskores::Id base)
    : Base(base)
  {
  }

  VISKORES_EXEC viskores::Id operator()(viskores::Id index) const { return this->Base + index; }

  viskores::Id Base;
};

} // namespace detail

/// @brief A linear bounding volume hierarchy over axis-aligned boxes in double precision.
///
/// Built in parallel after Karras, "Maximizing Parallelism in the Construction of BVHs,
/// Octrees, and k-d Trees" (2012): the boxes are sorted by the Morton codes of their centres,
/// every internal node is built independently, and the node boxes are propagated to the root.
/// The node boxes are exact unions of the input boxes, so the tree is a deterministic function
/// of them. Unlike the cell locators of Viskores, which find the cell containing a point, it
/// answers box-overlap and nearest-primitive queries: pass it to a worklet as an `ExecObject`,
/// whose execution object is a `BoundingVolumeHierarchyPortals`.
class BoundingVolumeHierarchy : public viskores::cont::ExecutionObjectBase
{
public:
  using Vec3 = viskores::Vec3f_64;

  BoundingVolumeHierarchy() = default;

  /// Builds the hierarchy over the boxes `[lower[i], upper[i]]`.
  VISKORES_CONT BoundingVolumeHierarchy(const viskores::cont::ArrayHandle<Vec3>& lower,
                                        const viskores::cont::ArrayHandle<Vec3>& upper)
  {
    this->Build(lower, upper);
  }

  VISKORES_CONT void Build(const viskores::cont::ArrayHandle<Vec3>& lower,
                           const viskores::cont::ArrayHandle<Vec3>& upper)
  {
    using Algorithm = viskores::cont::Algorithm;
    viskores::cont::Invoker invoke;
    const viskores::Id n = lower.GetNumberOfValues();
    this->NumberOfLeaves = n;
    if (n == 0)
    {
      this->Lower.ReleaseResources();
      this->Upper.ReleaseResources();
      this->Children.ReleaseResources();
      this->LeafPrimitives.ReleaseResources();
      return;
    }
    const Vec3 big(std::numeric_limits<viskores::Float64>::infinity());
    const Vec3 minimum = Algorithm::Reduce(lower, big, viskores::Minimum());
    const Vec3 maximum = Algorithm::Reduce(upper, -big, viskores::Maximum());
    Vec3 scale;
    for (int a = 0; a < 3; ++a)
    {
      const viskores::Float64 extent = maximum[a] - minimum[a];
      scale[a] = (extent > 0) ? viskores::Float64(2097151) / extent : viskores::Float64(0);
    }
    viskores::cont::ArrayHandle<viskores::Pair<viskores::UInt64, viskores::Id>> keyed;
    invoke(detail::BoxMortonCode(minimum, scale), lower, upper, keyed);
    Algorithm::Sort(keyed);
    viskores::cont::ArrayHandle<viskores::UInt64> codes;
    invoke(detail::SplitCode{}, keyed, codes, this->LeafPrimitives);

    const viskores::Id numNodes = 2 * n - 1;
    viskores::cont::ArrayHandle<viskores::Id> parents;
    parents.AllocateAndFill(numNodes, -1);
    if (n > 1)
    {
      invoke(detail::BuildInternalNodes(n),
             viskores::cont::ArrayHandleIndex(n - 1),
             codes,
             this->Children,
             parents);
    }
    else
    {
      this->Children.ReleaseResources();
    }
    this->Lower.Allocate(numNodes);
    this->Upper.Allocate(numNodes);
    viskores::cont::ArrayHandle<viskores::Id> visits;
    visits.AllocateAndFill(numNodes, 0);
    viskores::cont::ArrayHandle<viskores::Id> leaves;
    invoke(detail::LeafNode(n - 1), viskores::cont::ArrayHandleIndex(n), leaves);
    invoke(detail::PropagateBoxes{},
           leaves,
           this->LeafPrimitives,
           lower,
           upper,
           parents,
           this->Children,
           this->Lower,
           this->Upper,
           visits);
  }

  VISKORES_CONT viskores::Id GetNumberOfPrimitives() const { return this->NumberOfLeaves; }

  using ExecObjectType =
    BoundingVolumeHierarchyPortals<typename viskores::cont::ArrayHandle<Vec3>::ReadPortalType,
                                   typename viskores::cont::ArrayHandle<viskores::Id2>::ReadPortalType,
                                   typename viskores::cont::ArrayHandle<viskores::Id>::ReadPortalType>;

  VISKORES_CONT ExecObjectType PrepareForExecution(viskores::cont::DeviceAdapterId device,
                                                   viskores::cont::Token& token) const
  {
    return ExecObjectType{ this->NumberOfLeaves,
                           this->Lower.PrepareForInput(device, token),
                           this->Upper.PrepareForInput(device, token),
                           this->Children.PrepareForInput(device, token),
                           this->LeafPrimitives.PrepareForInput(device, token) };
  }

private:
  viskores::Id NumberOfLeaves = 0;
  viskores::cont::ArrayHandle<Vec3> Lower;
  viskores::cont::ArrayHandle<Vec3> Upper;
  viskores::cont::ArrayHandle<viskores::Id2> Children;
  viskores::cont::ArrayHandle<viskores::Id> LeafPrimitives;
};

}
}
} // namespace viskores::worklet::spatialstructure

#endif // viskores_worklet_spatialstructure_BoundingVolumeHierarchy_h
