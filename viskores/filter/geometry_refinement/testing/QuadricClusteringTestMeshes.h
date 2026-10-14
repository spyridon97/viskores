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

} // namespace qcd_test

#if defined(__clang__)
#pragma float_control(pop)
#endif

#endif // viskores_filter_geometry_refinement_testing_QuadricClusteringTestMeshes_h
