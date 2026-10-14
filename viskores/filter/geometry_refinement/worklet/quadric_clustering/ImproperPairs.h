//============================================================================
//  The contents of this file are covered by the Viskores license. See
//  LICENSE.txt for details.
//
//  By contributing to this file, all contributors agree to the Developer
//  Certificate of Origin Version 1.1 (DCO 1.1) as stated in DCO.txt.
//============================================================================
#ifndef viskores_worklet_quadric_clustering_ImproperPairs_h
#define viskores_worklet_quadric_clustering_ImproperPairs_h

#include <viskores/ExactPredicates.h>
#include <viskores/Types.h>

// The classification of a pair of realised output triangles by the labels they share, for the
// embedding flag (F4) of Algorithm 6. Every predicate is exact.

namespace viskores
{
namespace worklet
{
namespace quadric_clustering
{
namespace exact
{

using Vec3 = viskores::Vec3f_64;

/// For a triangle `p c d` and a segment `pw` from its vertex `p`: whether they meet in more
/// than `p`. That happens only if `w` lies in the plane of the triangle and the direction
/// `w - p` points into its closed angle at `p`.
VISKORES_EXEC_CONT inline bool SegmentLeavesVertexInto(const Vec3& p,
                                                       const Vec3& w,
                                                       const Vec3& c,
                                                       const Vec3& d)
{
  if (viskores::Orient3D(p, c, d, w) != 0)
  {
    return false;
  }
  const int axis = viskores::TriangleProjectionAxis(p, c, d);
  const int s = viskores::Orient2DProjected(p, c, d, axis);
  const int toC = viskores::Orient2DProjected(p, c, w, axis) * s;
  const int toD = viskores::Orient2DProjected(p, w, d, axis) * s;
  return toC >= 0 && toD >= 0 && (toC > 0 || toD > 0);
}

/// Whether two nondegenerate realised output triangles, with their labels, meet improperly:
/// with no shared label, if they meet at all; with one, `p`, if they meet in more than `p`;
/// with two, if they are coplanar and their third vertices lie on the same side of the shared
/// edge. Two noncoplanar triangles sharing an edge meet exactly in it.
VISKORES_EXEC_CONT inline bool ImproperPair(const Vec3* p,
                                            const viskores::Id3& pLabels,
                                            const Vec3* q,
                                            const viskores::Id3& qLabels)
{
  int pShared[3] = { -1, -1, -1 }; // for every vertex of p, the matching vertex of q
  int numShared = 0;
  for (int i = 0; i < 3; ++i)
  {
    for (int j = 0; j < 3; ++j)
    {
      if (pLabels[i] == qLabels[j])
      {
        pShared[i] = j;
        ++numShared;
      }
    }
  }
  if (numShared == 3)
  {
    return false;
  }
  if (numShared == 2)
  {
    int a = 0;
    while (pShared[a] >= 0)
    {
      ++a;
    }
    int b = 0;
    while (b == pShared[(a + 1) % 3] || b == pShared[(a + 2) % 3])
    {
      ++b;
    }
    const Vec3& s0 = p[(a + 1) % 3];
    const Vec3& s1 = p[(a + 2) % 3];
    if (viskores::Orient3D(s0, s1, p[a], q[b]) != 0)
    {
      return false;
    }
    const int axis = viskores::TriangleProjectionAxis(s0, s1, p[a]);
    return viskores::Orient2DProjected(s0, s1, p[a], axis) == viskores::Orient2DProjected(s0, s1, q[b], axis);
  }
  if (numShared == 1)
  {
    int i = 0;
    while (pShared[i] < 0)
    {
      ++i;
    }
    const int j = pShared[i];
    const Vec3& shared = p[i];
    const Vec3& a = p[(i + 1) % 3];
    const Vec3& b = p[(i + 2) % 3];
    const Vec3& c = q[(j + 1) % 3];
    const Vec3& d = q[(j + 2) % 3];
    return viskores::SegmentIntersectsTriangle(a, b, shared, c, d) || viskores::SegmentIntersectsTriangle(c, d, shared, a, b) ||
      SegmentLeavesVertexInto(shared, a, c, d) || SegmentLeavesVertexInto(shared, b, c, d) ||
      SegmentLeavesVertexInto(shared, c, a, b) || SegmentLeavesVertexInto(shared, d, a, b);
  }
  for (int e = 0; e < 3; ++e)
  {
    if (viskores::SegmentIntersectsTriangle(p[e], p[(e + 1) % 3], q[0], q[1], q[2]) ||
        viskores::SegmentIntersectsTriangle(q[e], q[(e + 1) % 3], p[0], p[1], p[2]))
    {
      return true;
    }
  }
  return false;
}

}
}
}
} // namespace viskores::worklet::quadric_clustering::exact

#endif // viskores_worklet_quadric_clustering_ImproperPairs_h
