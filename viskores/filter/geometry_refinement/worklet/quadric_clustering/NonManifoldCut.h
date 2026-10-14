//============================================================================
//  The contents of this file are covered by the Viskores license. See
//  LICENSE.txt for details.
//
//  By contributing to this file, all contributors agree to the Developer
//  Certificate of Origin Version 1.1 (DCO 1.1) as stated in DCO.txt.
//============================================================================
#ifndef viskores_worklet_quadric_clustering_NonManifoldCut_h
#define viskores_worklet_quadric_clustering_NonManifoldCut_h

#include <viskores/Types.h>
#include <viskores/cont/ArrayHandle.h>

#include <algorithm>
#include <deque>
#include <numeric>
#include <tuple>
#include <vector>

namespace viskores
{
namespace worklet
{
namespace quadric_clustering
{

/// A triangle complex cut into surfaces (Section 9.9 of the paper, after Gueziec et al.,
/// "Cutting and stitching"): every vertex is split into one copy per fan of triangles connected
/// across edges not in the cut set `X`, which contains the singular edges, those in three or
/// more triangles, and a seam that makes every piece orientable.
struct CutComplex
{
  /// The triangles on the copies, reoriented so that every piece is consistently oriented.
  std::vector<viskores::Id3> Triangles;
  /// The input vertex of every copy.
  std::vector<viskores::Id> Original;
  /// The copies of the vertices on an edge of `X` and of the vertices with several fans, which
  /// stay singleton clusters.
  std::vector<viskores::UInt8> Frozen;
  /// The edges of `X`, as sorted vertex pairs in lexicographic order.
  std::vector<viskores::Id2> CutEdges;
};

/// Cuts the triangles `triangles` on `numVertices` vertices as lib/nonmanifold.cut of the
/// reference implementation, sequentially and deterministically: the orientation is propagated
/// breadth first from the triangles in index order, across edges in exactly two triangles that
/// are not singular, and an edge whose two triangles cannot be made to agree joins the seam.
/// Copies are numbered by vertex, then by the smallest triangle of their fan.
inline CutComplex CutAlongSingularEdges(const std::vector<viskores::Id3>& triangles,
                                        viskores::Id numVertices)
{
  CutComplex cut;
  const std::size_t m = triangles.size();
  // The triangles of every edge, in increasing order.
  std::vector<std::tuple<viskores::Id, viskores::Id, std::size_t>> records;
  records.reserve(3 * m);
  for (std::size_t t = 0; t < m; ++t)
  {
    for (int s = 0; s < 3; ++s)
    {
      const viskores::Id x = triangles[t][s], y = triangles[t][(s + 1) % 3];
      records.emplace_back(std::min(x, y), std::max(x, y), t);
    }
  }
  std::sort(records.begin(), records.end());
  std::vector<viskores::Id2> edges;
  std::vector<std::size_t> edgeStart;
  for (std::size_t r = 0; r < records.size(); ++r)
  {
    const viskores::Id2 e(std::get<0>(records[r]), std::get<1>(records[r]));
    if (edges.empty() || edges.back() != e)
    {
      edges.push_back(e);
      edgeStart.push_back(r);
    }
  }
  edgeStart.push_back(records.size());
  auto edgeIndex = [&](viskores::Id x, viskores::Id y)
  {
    const viskores::Id2 e(std::min(x, y), std::max(x, y));
    return static_cast<std::size_t>(std::lower_bound(edges.begin(), edges.end(), e) - edges.begin());
  };
  auto count = [&](std::size_t e) { return edgeStart[e + 1] - edgeStart[e]; };
  auto triangleOf = [&](std::size_t e, std::size_t k) { return std::get<2>(records[edgeStart[e] + k]); };
  std::vector<bool> inX(edges.size(), false);
  for (std::size_t e = 0; e < edges.size(); ++e)
  {
    inX[e] = count(e) >= 3;
  }
  // +1 if the triangle traverses x -> y, -1 if y -> x.
  auto direction = [](const viskores::Id3& t, viskores::Id x, viskores::Id y)
  {
    for (int s = 0; s < 3; ++s)
    {
      if (t[s] == x && t[(s + 1) % 3] == y)
      {
        return 1;
      }
    }
    return -1;
  };

  // Orientation by breadth-first search; conflicts make the seam.
  std::vector<int> sign(m, 0);
  for (std::size_t t0 = 0; t0 < m; ++t0)
  {
    if (sign[t0] != 0)
    {
      continue;
    }
    sign[t0] = 1;
    std::deque<std::size_t> queue = { t0 };
    while (!queue.empty())
    {
      const std::size_t t = queue.front();
      queue.pop_front();
      const viskores::Id3 tri = triangles[t];
      for (int s = 0; s < 3; ++s)
      {
        const viskores::Id x = tri[s], y = tri[(s + 1) % 3];
        const std::size_t e = edgeIndex(x, y);
        if (inX[e] || count(e) != 2)
        {
          continue;
        }
        const std::size_t u = (triangleOf(e, 0) == t) ? triangleOf(e, 1) : triangleOf(e, 0);
        const int want = -sign[t] * direction(tri, x, y) * direction(triangles[u], x, y);
        if (sign[u] == 0)
        {
          sign[u] = want;
          queue.push_back(u);
        }
        else if (sign[u] != want)
        {
          inX[e] = true; // the seam
        }
      }
    }
  }
  std::vector<viskores::Id3> oriented = triangles;
  for (std::size_t t = 0; t < m; ++t)
  {
    if (sign[t] < 0)
    {
      oriented[t] = viskores::Id3(triangles[t][2], triangles[t][1], triangles[t][0]);
    }
  }

  // Fans: the corners (v, t) joined across the edges at v in two triangles and not in X; the
  // root of a fan is its corner of smallest triangle.
  std::vector<std::size_t> parent(3 * m);
  std::iota(parent.begin(), parent.end(), 0);
  auto corner = [&](viskores::Id v, std::size_t t)
  {
    for (int c = 0; c < 3; ++c)
    {
      if (oriented[t][c] == v)
      {
        return 3 * t + static_cast<std::size_t>(c);
      }
    }
    return std::size_t(0);
  };
  auto find = [&](std::size_t z)
  {
    while (parent[z] != z)
    {
      parent[z] = parent[parent[z]];
      z = parent[z];
    }
    return z;
  };
  for (std::size_t e = 0; e < edges.size(); ++e)
  {
    if (inX[e] || count(e) != 2)
    {
      continue;
    }
    for (int end = 0; end < 2; ++end)
    {
      const viskores::Id v = edges[e][end];
      const std::size_t ra = find(corner(v, triangleOf(e, 0)));
      const std::size_t rb = find(corner(v, triangleOf(e, 1)));
      if (ra != rb)
      {
        // The corner of the smaller triangle stays the root.
        if (ra / 3 < rb / 3)
        {
          parent[rb] = ra;
        }
        else
        {
          parent[ra] = rb;
        }
      }
    }
  }
  // Copies numbered in the order of their roots (v, t).
  std::vector<std::pair<viskores::Id, std::size_t>> roots;
  for (std::size_t t = 0; t < m; ++t)
  {
    for (int c = 0; c < 3; ++c)
    {
      const std::size_t z = 3 * t + static_cast<std::size_t>(c);
      if (find(z) == z)
      {
        roots.emplace_back(oriented[t][c], t);
      }
    }
  }
  std::sort(roots.begin(), roots.end());
  std::vector<viskores::Id> copyOfRoot(3 * m, -1);
  for (std::size_t r = 0; r < roots.size(); ++r)
  {
    copyOfRoot[corner(roots[r].first, roots[r].second)] = static_cast<viskores::Id>(r);
    cut.Original.push_back(roots[r].first);
  }
  cut.Triangles.resize(m);
  for (std::size_t t = 0; t < m; ++t)
  {
    for (int c = 0; c < 3; ++c)
    {
      cut.Triangles[t][c] = copyOfRoot[find(3 * t + static_cast<std::size_t>(c))];
    }
  }
  std::vector<viskores::Id> numCopies(static_cast<std::size_t>(numVertices), 0);
  for (viskores::Id v : cut.Original)
  {
    ++numCopies[static_cast<std::size_t>(v)];
  }
  std::vector<bool> onCut(static_cast<std::size_t>(numVertices), false);
  for (std::size_t e = 0; e < edges.size(); ++e)
  {
    if (inX[e])
    {
      onCut[static_cast<std::size_t>(edges[e][0])] = true;
      onCut[static_cast<std::size_t>(edges[e][1])] = true;
      cut.CutEdges.push_back(edges[e]);
    }
  }
  for (viskores::Id v : cut.Original)
  {
    const std::size_t i = static_cast<std::size_t>(v);
    cut.Frozen.push_back((numCopies[i] > 1 || onCut[i]) ? 1 : 0);
  }
  return cut;
}

}
}
} // namespace viskores::worklet::quadric_clustering

#endif // viskores_worklet_quadric_clustering_NonManifoldCut_h
