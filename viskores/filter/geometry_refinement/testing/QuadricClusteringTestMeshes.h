//============================================================================
//  The contents of this file are covered by the Viskores license. See
//  LICENSE.txt for details.
//
//  By contributing to this file, all contributors agree to the Developer
//  Certificate of Origin Version 1.1 (DCO 1.1) as stated in DCO.txt.
//============================================================================
#ifndef viskores_filter_geometry_refinement_testing_QuadricClusteringTestMeshes_h
#define viskores_filter_geometry_refinement_testing_QuadricClusteringTestMeshes_h

// Test meshes for the quadric clustering tests, ported from the meshes of the reference
// implementation of the paper (proofs/lib/meshes.py), and checks of outputs written
// independently of the algorithm. The icosphere and the voxel solids are built with the same
// operations as the reference, so they agree with it bit for bit.

#include <viskores/Types.h>
#include <viskores/cont/ArrayHandle.h>
#include <viskores/cont/DataSetBuilderExplicit.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <map>
#include <numeric>
#include <set>
#include <string>
#include <vector>

#if defined(__clang__)
#pragma float_control(push)
#pragma clang fp contract(off)
#endif

namespace qcd_test
{
using Vec3 = viskores::Vec3f_64;

struct Mesh
{
  std::vector<Vec3> Points;
  std::vector<viskores::Id3> Triangles;

  viskores::cont::ArrayHandle<Vec3> PointArray() const
  {
    return viskores::cont::make_ArrayHandle(this->Points, viskores::CopyFlag::On);
  }
  viskores::cont::ArrayHandle<viskores::Id3> TriangleArray() const
  {
    return viskores::cont::make_ArrayHandle(this->Triangles, viskores::CopyFlag::On);
  }
  viskores::cont::DataSet ToDataSet() const
  {
    std::vector<viskores::UInt8> shapes(this->Triangles.size(), viskores::CELL_SHAPE_TRIANGLE);
    std::vector<viskores::IdComponent> numIndices(this->Triangles.size(), 3);
    std::vector<viskores::Id> connectivity;
    for (const auto& t : this->Triangles)
    {
      connectivity.insert(connectivity.end(), { t[0], t[1], t[2] });
    }
    return viskores::cont::DataSetBuilderExplicit::Create(
      this->Points, shapes, numIndices, connectivity, "coordinates");
  }
};

/// The values of an array handle.
template <typename T>
inline std::vector<T> ToVector(const viskores::cont::ArrayHandle<T>& array)
{
  std::vector<T> result(static_cast<std::size_t>(array.GetNumberOfValues()));
  auto portal = array.ReadPortal();
  for (viskores::Id i = 0; i < array.GetNumberOfValues(); ++i)
  {
    result[static_cast<std::size_t>(i)] = portal.Get(i);
  }
  return result;
}

inline double Norm(const Vec3& v)
{
  return std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
}

inline Vec3 Normalized(const Vec3& v)
{
  const double n = Norm(v);
  return Vec3(v[0] / n, v[1] / n, v[2] / n);
}

/// The icosahedron subdivided `level` times, projected onto the unit sphere.
inline Mesh Icosphere(int level)
{
  const double t = (1 + std::sqrt(5.0)) / 2;
  Mesh mesh;
  for (const Vec3& v : { Vec3(-1, t, 0),
                         Vec3(1, t, 0),
                         Vec3(-1, -t, 0),
                         Vec3(1, -t, 0),
                         Vec3(0, -1, t),
                         Vec3(0, 1, t),
                         Vec3(0, -1, -t),
                         Vec3(0, 1, -t),
                         Vec3(t, 0, -1),
                         Vec3(t, 0, 1),
                         Vec3(-t, 0, -1),
                         Vec3(-t, 0, 1) })
  {
    mesh.Points.push_back(Normalized(v));
  }
  mesh.Triangles = { { 0, 11, 5 }, { 0, 5, 1 },  { 0, 1, 7 },   { 0, 7, 10 }, { 0, 10, 11 },
                     { 1, 5, 9 },  { 5, 11, 4 }, { 11, 10, 2 }, { 10, 7, 6 }, { 7, 1, 8 },
                     { 3, 9, 4 },  { 3, 4, 2 },  { 3, 2, 6 },   { 3, 6, 8 },  { 3, 8, 9 },
                     { 4, 9, 5 },  { 2, 4, 11 }, { 6, 2, 10 },  { 8, 6, 7 },  { 9, 8, 1 } };
  for (int l = 0; l < level; ++l)
  {
    std::map<std::pair<viskores::Id, viskores::Id>, viskores::Id> mid;
    auto m = [&](viskores::Id a, viskores::Id b)
    {
      const auto key = std::make_pair(std::min(a, b), std::max(a, b));
      auto it = mid.find(key);
      if (it != mid.end())
      {
        return it->second;
      }
      const Vec3 p = mesh.Points[static_cast<std::size_t>(a)] + mesh.Points[static_cast<std::size_t>(b)];
      mesh.Points.push_back(Normalized(p));
      const viskores::Id id = static_cast<viskores::Id>(mesh.Points.size()) - 1;
      mid[key] = id;
      return id;
    };
    std::vector<viskores::Id3> next;
    for (const auto& f : mesh.Triangles)
    {
      const viskores::Id a = f[0], b = f[1], c = f[2];
      const viskores::Id ab = m(a, b);
      const viskores::Id ca = m(c, a);
      const viskores::Id bc = m(b, c);
      next.push_back({ a, ab, ca });
      next.push_back({ b, bc, ab });
      next.push_back({ c, ca, bc });
      next.push_back({ ab, bc, ca });
    }
    mesh.Triangles = next;
  }
  return mesh;
}

/// The boundary of a union of unit voxels, every square face split into two triangles.
inline Mesh VoxelSolid(const std::vector<std::vector<std::vector<bool>>>& occupied, double scale)
{
  const int nx = static_cast<int>(occupied.size()) + 2;
  const int ny = static_cast<int>(occupied[0].size()) + 2;
  const int nz = static_cast<int>(occupied[0][0].size()) + 2;
  auto occ = [&](int x, int y, int z)
  {
    if (x < 1 || y < 1 || z < 1 || x > nx - 2 || y > ny - 2 || z > nz - 2)
    {
      return false;
    }
    return static_cast<bool>(occupied[static_cast<std::size_t>(x - 1)][static_cast<std::size_t>(
      y - 1)][static_cast<std::size_t>(z - 1)]);
  };
  std::map<std::array<int, 3>, viskores::Id> ids;
  std::vector<std::array<int, 3>> points;
  Mesh mesh;
  auto vertex = [&](const std::array<int, 3>& p)
  {
    auto it = ids.find(p);
    if (it != ids.end())
    {
      return it->second;
    }
    const viskores::Id id = static_cast<viskores::Id>(points.size());
    ids[p] = id;
    points.push_back(p);
    return id;
  };
  for (int x = 0; x < nx; ++x)
  {
    for (int y = 0; y < ny; ++y)
    {
      for (int z = 0; z < nz; ++z)
      {
        if (!occ(x, y, z))
        {
          continue;
        }
        for (int axis = 0; axis < 3; ++axis)
        {
          for (int sign : { -1, 1 })
          {
            std::array<int, 3> nb = { x, y, z };
            nb[static_cast<std::size_t>(axis)] += sign;
            if (occ(nb[0], nb[1], nb[2]))
            {
              continue;
            }
            std::array<int, 3> base = { x, y, z };
            if (sign == 1)
            {
              base[static_cast<std::size_t>(axis)] += 1;
            }
            const std::size_t a1 = static_cast<std::size_t>((axis + 1) % 3);
            const std::size_t a2 = static_cast<std::size_t>((axis + 2) % 3);
            std::vector<viskores::Id> corners;
            for (const auto& d : { std::array<int, 2>{ 0, 0 },
                                   std::array<int, 2>{ 1, 0 },
                                   std::array<int, 2>{ 1, 1 },
                                   std::array<int, 2>{ 0, 1 } })
            {
              std::array<int, 3> p = base;
              p[a1] += d[0];
              p[a2] += d[1];
              corners.push_back(vertex(p));
            }
            if (sign == -1)
            {
              std::reverse(corners.begin(), corners.end());
            }
            mesh.Triangles.push_back({ corners[0], corners[1], corners[2] });
            mesh.Triangles.push_back({ corners[0], corners[2], corners[3] });
          }
        }
      }
    }
  }
  // As in the reference, coordinates are indices into the grid padded by one voxel.
  for (const auto& p : points)
  {
    mesh.Points.push_back(Vec3(static_cast<double>(p[0]) * scale,
                               static_cast<double>(p[1]) * scale,
                               static_cast<double>(p[2]) * scale));
  }
  return mesh;
}

/// A voxel plate with `numHoles` square holes through it.
inline std::vector<std::vector<std::vector<bool>>> PlateWithHoles(int numHoles,
                                                                  int hole = 2,
                                                                  int gap = 2,
                                                                  int thick = 2,
                                                                  int border = 2)
{
  const int nx = border * 2 + numHoles * hole + (numHoles - 1) * gap;
  const int ny = border * 2 + hole;
  std::vector<std::vector<std::vector<bool>>> occ(
    static_cast<std::size_t>(nx),
    std::vector<std::vector<bool>>(static_cast<std::size_t>(ny),
                                   std::vector<bool>(static_cast<std::size_t>(thick), true)));
  for (int h = 0; h < numHoles; ++h)
  {
    const int x0 = border + h * (hole + gap);
    for (int x = x0; x < x0 + hole; ++x)
    {
      for (int y = border; y < border + hole; ++y)
      {
        for (int z = 0; z < thick; ++z)
        {
          occ[static_cast<std::size_t>(x)][static_cast<std::size_t>(y)][static_cast<std::size_t>(z)] =
            false;
        }
      }
    }
  }
  return occ;
}

/// A torus of major radius `major` and minor radius `minor`, `nu` by `nv` quads split in two.
/// Its coordinates use the C library's trigonometry and need not match the reference bit for
/// bit.
inline Mesh Torus(double major, double minor, int nu, int nv)
{
  Mesh mesh;
  const double pi = 3.14159265358979323846;
  for (int i = 0; i < nu; ++i)
  {
    for (int j = 0; j < nv; ++j)
    {
      const double u = 2 * pi * i / nu;
      const double v = 2 * pi * j / nv;
      mesh.Points.push_back(Vec3((major + minor * std::cos(v)) * std::cos(u),
                                 (major + minor * std::cos(v)) * std::sin(u),
                                 minor * std::sin(v)));
    }
  }
  auto idx = [&](int a, int b) { return static_cast<viskores::Id>((a % nu) * nv + (b % nv)); };
  for (int a = 0; a < nu; ++a)
  {
    for (int b = 0; b < nv; ++b)
    {
      const viskores::Id p = idx(a, b), q = idx(a + 1, b), r = idx(a + 1, b + 1), s = idx(a, b + 1);
      mesh.Triangles.push_back({ p, q, r });
      mesh.Triangles.push_back({ p, r, s });
    }
  }
  return mesh;
}

inline Vec3 Centroid(const Mesh& mesh, const viskores::Id3& t)
{
  const Vec3 s = (mesh.Points[static_cast<std::size_t>(t[0])] + mesh.Points[static_cast<std::size_t>(t[1])]) +
    mesh.Points[static_cast<std::size_t>(t[2])];
  return Vec3(s[0] / 3, s[1] / 3, s[2] / 3);
}

/// Removes the triangles for which `remove` holds, and the points no triangle uses.
inline Mesh RemoveTriangles(const Mesh& mesh, const std::function<bool(const viskores::Id3&)>& remove)
{
  std::vector<viskores::Id3> kept;
  for (const auto& t : mesh.Triangles)
  {
    if (!remove(t))
    {
      kept.push_back(t);
    }
  }
  std::vector<viskores::Id> remap(mesh.Points.size(), -1);
  for (const auto& t : kept)
  {
    for (int c = 0; c < 3; ++c)
    {
      remap[static_cast<std::size_t>(t[c])] = 0;
    }
  }
  Mesh result;
  for (std::size_t v = 0; v < mesh.Points.size(); ++v)
  {
    if (remap[v] == 0)
    {
      remap[v] = static_cast<viskores::Id>(result.Points.size());
      result.Points.push_back(mesh.Points[v]);
    }
  }
  for (const auto& t : kept)
  {
    result.Triangles.push_back({ remap[static_cast<std::size_t>(t[0])],
                                 remap[static_cast<std::size_t>(t[1])],
                                 remap[static_cast<std::size_t>(t[2])] });
  }
  return result;
}

/// The open meshes of the reference: a sphere without a cap (a disk), without two caps (an
/// annulus), and a voxel plate of genus two with an opening in its top face.
inline Mesh Disk()
{
  const Mesh sphere = Icosphere(3);
  return RemoveTriangles(sphere, [&](const viskores::Id3& t) { return Centroid(sphere, t)[2] > 0.6; });
}

inline Mesh Annulus()
{
  const Mesh sphere = Icosphere(3);
  return RemoveTriangles(sphere,
                         [&](const viskores::Id3& t)
                         { return std::abs(Centroid(sphere, t)[2]) > 0.6; });
}

inline Mesh Genus2PlateWithOpening()
{
  const Mesh plate = VoxelSolid(PlateWithHoles(2), 1.0);
  double zMax = -1e300, xMin = 1e300;
  for (const auto& p : plate.Points)
  {
    zMax = std::max(zMax, p[2]);
    xMin = std::min(xMin, p[0]);
  }
  return RemoveTriangles(plate,
                         [&](const viskores::Id3& t)
                         {
                           const Vec3 c = Centroid(plate, t);
                           return std::abs(c[2] - zMax) < 1e-9 && c[0] < xMin + 3.5;
                         });
}

/// The number of connected components of every cluster, by union-find over the input edges.
inline std::vector<int> ComponentsPerLabel(const Mesh& mesh,
                                           const std::vector<viskores::Id>& labels,
                                           viskores::Id k)
{
  std::vector<viskores::Id> parent(mesh.Points.size());
  std::iota(parent.begin(), parent.end(), 0);
  std::function<viskores::Id(viskores::Id)> find = [&](viskores::Id x)
  {
    while (parent[static_cast<std::size_t>(x)] != x)
    {
      x = parent[static_cast<std::size_t>(x)] =
        parent[static_cast<std::size_t>(parent[static_cast<std::size_t>(x)])];
    }
    return x;
  };
  for (const auto& t : mesh.Triangles)
  {
    for (int s = 0; s < 3; ++s)
    {
      const viskores::Id a = t[s], b = t[(s + 1) % 3];
      if (labels[static_cast<std::size_t>(a)] == labels[static_cast<std::size_t>(b)])
      {
        parent[static_cast<std::size_t>(find(a))] = find(b);
      }
    }
  }
  std::vector<std::set<viskores::Id>> roots(static_cast<std::size_t>(k));
  for (std::size_t v = 0; v < mesh.Points.size(); ++v)
  {
    roots[static_cast<std::size_t>(labels[v])].insert(find(static_cast<viskores::Id>(v)));
  }
  std::vector<int> counts;
  for (const auto& r : roots)
  {
    counts.push_back(static_cast<int>(r.size()));
  }
  return counts;
}

/// Undirected edges of a triangle list, with the number of triangles of each.
inline std::map<std::pair<viskores::Id, viskores::Id>, int> EdgeCounts(
  const std::vector<viskores::Id3>& triangles)
{
  std::map<std::pair<viskores::Id, viskores::Id>, int> counts;
  for (const auto& t : triangles)
  {
    for (int s = 0; s < 3; ++s)
    {
      const viskores::Id a = t[s], b = t[(s + 1) % 3];
      ++counts[std::make_pair(std::min(a, b), std::max(a, b))];
    }
  }
  return counts;
}

/// The number of boundary loops: connected components of the graph of boundary edges.
inline int BoundaryLoops(const std::vector<viskores::Id3>& triangles)
{
  std::map<viskores::Id, std::vector<viskores::Id>> adjacency;
  for (const auto& e : EdgeCounts(triangles))
  {
    if (e.second == 1)
    {
      adjacency[e.first.first].push_back(e.first.second);
      adjacency[e.first.second].push_back(e.first.first);
    }
  }
  std::set<viskores::Id> seen;
  int loops = 0;
  for (const auto& entry : adjacency)
  {
    if (seen.count(entry.first))
    {
      continue;
    }
    ++loops;
    std::vector<viskores::Id> stack = { entry.first };
    seen.insert(entry.first);
    while (!stack.empty())
    {
      const viskores::Id x = stack.back();
      stack.pop_back();
      for (viskores::Id y : adjacency[x])
      {
        if (seen.insert(y).second)
        {
          stack.push_back(y);
        }
      }
    }
  }
  return loops;
}

inline viskores::Id EulerCharacteristic(viskores::Id numVertices,
                                        const std::vector<viskores::Id3>& triangles)
{
  return numVertices - static_cast<viskores::Id>(EdgeCounts(triangles).size()) +
    static_cast<viskores::Id>(triangles.size());
}

/// Checks that `triangles` on `numVertices` vertices form a compact surface, consistently
/// oriented: every edge in one or two triangles, in two once in each direction, and the
/// triangles around every vertex a single cycle or a single path. Returns an empty string or
/// the first problem found.
inline std::string CheckSurface(viskores::Id numVertices, const std::vector<viskores::Id3>& triangles)
{
  std::set<std::pair<viskores::Id, viskores::Id>> directed;
  std::vector<std::map<viskores::Id, viskores::Id>> successor(static_cast<std::size_t>(numVertices));
  for (const auto& t : triangles)
  {
    for (int s = 0; s < 3; ++s)
    {
      if (t[s] < 0 || t[s] >= numVertices)
      {
        return "a triangle has an invalid vertex";
      }
      if (!directed.insert(std::make_pair(t[s], t[(s + 1) % 3])).second)
      {
        return "a directed edge appears twice";
      }
      auto& succ = successor[static_cast<std::size_t>(t[s])];
      if (succ.count(t[(s + 1) % 3]))
      {
        return "a vertex link branches";
      }
      succ[t[(s + 1) % 3]] = t[(s + 2) % 3];
    }
  }
  for (const auto& e : EdgeCounts(triangles))
  {
    if (e.second > 2)
    {
      return "an edge lies in more than two triangles";
    }
  }
  for (viskores::Id v = 0; v < numVertices; ++v)
  {
    const auto& succ = successor[static_cast<std::size_t>(v)];
    if (succ.empty())
    {
      return "a vertex lies in no triangle";
    }
    std::set<viskores::Id> targets;
    for (const auto& p : succ)
    {
      targets.insert(p.second);
    }
    std::vector<viskores::Id> heads;
    for (const auto& p : succ)
    {
      if (!targets.count(p.first))
      {
        heads.push_back(p.first);
      }
    }
    if (heads.size() > 1)
    {
      return "a vertex is pinched";
    }
    viskores::Id current = heads.empty() ? succ.begin()->first : heads[0];
    const viskores::Id start = current;
    std::size_t steps = 0;
    while (succ.count(current))
    {
      current = succ.at(current);
      ++steps;
      if (current == start || steps > succ.size())
      {
        break;
      }
    }
    if (steps != succ.size())
    {
      return "a vertex link is not a single cycle or path";
    }
  }
  return "";
}

/// Distance from `p` to the triangle `abc`, after Ericson's closest point on a triangle.
inline double PointTriangleDistance(const Vec3& p, const Vec3& a, const Vec3& b, const Vec3& c)
{
  auto dot = [](const Vec3& x, const Vec3& y) { return x[0] * y[0] + x[1] * y[1] + x[2] * y[2]; };
  const Vec3 ab = b - a, ac = c - a, ap = p - a;
  const double d1 = dot(ab, ap), d2 = dot(ac, ap);
  if (d1 <= 0 && d2 <= 0)
  {
    return Norm(p - a);
  }
  const Vec3 bp = p - b;
  const double d3 = dot(ab, bp), d4 = dot(ac, bp);
  if (d3 >= 0 && d4 <= d3)
  {
    return Norm(p - b);
  }
  const double vc = d1 * d4 - d3 * d2;
  if (vc <= 0 && d1 >= 0 && d3 <= 0)
  {
    return Norm(p - (a + (d1 / (d1 - d3)) * ab));
  }
  const Vec3 cp = p - c;
  const double d5 = dot(ab, cp), d6 = dot(ac, cp);
  if (d6 >= 0 && d5 <= d6)
  {
    return Norm(p - c);
  }
  const double vb = d5 * d2 - d1 * d6;
  if (vb <= 0 && d2 >= 0 && d6 <= 0)
  {
    return Norm(p - (a + (d2 / (d2 - d6)) * ac));
  }
  const double va = d3 * d6 - d5 * d4;
  if (va <= 0 && (d4 - d3) >= 0 && (d5 - d6) >= 0)
  {
    return Norm(p - (b + ((d4 - d3) / ((d4 - d3) + (d5 - d6))) * (c - b)));
  }
  const double denom = 1 / (va + vb + vc);
  return Norm(p - (a + (vb * denom) * ab + (vc * denom) * ac));
}

/// Points on every triangle such that every point of it is within `spacing` of one of them.
inline std::vector<Vec3> SampleTriangles(const std::vector<Vec3>& points,
                                         const std::vector<viskores::Id3>& triangles,
                                         double spacing)
{
  std::vector<Vec3> samples;
  for (const auto& t : triangles)
  {
    const Vec3 a = points[static_cast<std::size_t>(t[0])];
    const Vec3 b = points[static_cast<std::size_t>(t[1])];
    const Vec3 c = points[static_cast<std::size_t>(t[2])];
    const double longest = std::max({ Norm(b - a), Norm(c - b), Norm(a - c) });
    const int m = std::max(1, static_cast<int>(std::ceil(longest / spacing)));
    for (int i = 0; i <= m; ++i)
    {
      for (int j = 0; i + j <= m; ++j)
      {
        samples.push_back(a + (static_cast<double>(i) / m) * (b - a) +
                          (static_cast<double>(j) / m) * (c - a));
      }
    }
  }
  return samples;
}

/// The sampled distance from the triangle mesh A to the triangle mesh B, by brute force: the
/// largest distance from a sample of A to B.
inline double SampledDistance(const std::vector<Vec3>& pointsA,
                              const std::vector<viskores::Id3>& trianglesA,
                              const std::vector<Vec3>& pointsB,
                              const std::vector<viskores::Id3>& trianglesB,
                              double spacing)
{
  double worst = 0;
  for (const Vec3& p : SampleTriangles(pointsA, trianglesA, spacing))
  {
    double best = 1e300;
    for (const auto& t : trianglesB)
    {
      best = std::min(best,
                      PointTriangleDistance(p,
                                            pointsB[static_cast<std::size_t>(t[0])],
                                            pointsB[static_cast<std::size_t>(t[1])],
                                            pointsB[static_cast<std::size_t>(t[2])]));
    }
    worst = std::max(worst, best);
  }
  return worst;
}

/// The sampled two-sided Hausdorff distance between two triangle meshes, by brute force.
inline double SampledHausdorff(const std::vector<Vec3>& pointsA,
                               const std::vector<viskores::Id3>& trianglesA,
                               const std::vector<Vec3>& pointsB,
                               const std::vector<viskores::Id3>& trianglesB,
                               double spacing)
{
  return std::max(SampledDistance(pointsA, trianglesA, pointsB, trianglesB, spacing),
                  SampledDistance(pointsB, trianglesB, pointsA, trianglesA, spacing));
}

} // namespace qcd_test

#if defined(__clang__)
#pragma float_control(pop)
#endif

#endif // viskores_filter_geometry_refinement_testing_QuadricClusteringTestMeshes_h
