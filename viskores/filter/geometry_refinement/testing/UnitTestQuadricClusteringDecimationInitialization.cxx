//============================================================================
//  The contents of this file are covered by the Viskores license. See
//  LICENSE.txt for details.
//
//  By contributing to this file, all contributors agree to the Developer
//  Certificate of Origin Version 1.1 (DCO 1.1) as stated in DCO.txt.
//============================================================================

// Tests of the initialisation of the quadric clustering decimation (Section 11): the curvature
// seeds against the values pinned by the reference implementation (proofs/prop-determinism.py),
// the topological floor, the total weight, the number of seeds and the seeds themselves, which
// are integers and so must agree exactly.

#include <viskores/cont/testing/Testing.h>

#include <viskores/filter/geometry_refinement/testing/QuadricClusteringTestMeshes.h>
#include <viskores/filter/geometry_refinement/worklet/quadric_clustering/CurvatureSeeds.h>
#include <viskores/filter/geometry_refinement/worklet/quadric_clustering/SurfaceMesh.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

namespace
{
namespace qc = viskores::worklet::quadric_clustering;
using qcd_test::Mesh;

/// SHA-256 of a byte string, as lowercase hexadecimal (FIPS 180-4).
std::string Sha256(const std::string& message)
{
  static const std::uint32_t k[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
  };
  std::uint32_t h[8] = { 0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                         0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19 };
  auto rotr = [](std::uint32_t x, int n) { return (x >> n) | (x << (32 - n)); };
  std::vector<std::uint8_t> data(message.begin(), message.end());
  const std::uint64_t bits = static_cast<std::uint64_t>(data.size()) * 8;
  data.push_back(0x80);
  while (data.size() % 64 != 56)
  {
    data.push_back(0);
  }
  for (int i = 7; i >= 0; --i)
  {
    data.push_back(static_cast<std::uint8_t>(bits >> (8 * i)));
  }
  for (std::size_t chunk = 0; chunk < data.size(); chunk += 64)
  {
    std::uint32_t w[64];
    for (int i = 0; i < 16; ++i)
    {
      w[i] = (std::uint32_t(data[chunk + 4 * i]) << 24) |
        (std::uint32_t(data[chunk + 4 * i + 1]) << 16) |
        (std::uint32_t(data[chunk + 4 * i + 2]) << 8) | std::uint32_t(data[chunk + 4 * i + 3]);
    }
    for (int i = 16; i < 64; ++i)
    {
      const std::uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
      const std::uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
      w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    std::uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], hh = h[7];
    for (int i = 0; i < 64; ++i)
    {
      const std::uint32_t t1 =
        hh + (rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25)) + ((e & f) ^ (~e & g)) + k[i] + w[i];
      const std::uint32_t t2 =
        (rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22)) + ((a & b) ^ (a & c) ^ (b & c));
      hh = g;
      g = f;
      f = e;
      e = d + t1;
      d = c;
      c = b;
      b = a;
      a = t1 + t2;
    }
    h[0] += a;
    h[1] += b;
    h[2] += c;
    h[3] += d;
    h[4] += e;
    h[5] += f;
    h[6] += g;
    h[7] += hh;
  }
  std::ostringstream hex;
  for (std::uint32_t word : h)
  {
    hex << std::hex << std::setw(8) << std::setfill('0') << word;
  }
  return hex.str();
}

struct Pinned
{
  viskores::Id Floor;
  viskores::UInt64 Total;
  viskores::Id Seeds;
  std::string Digest;
  std::array<viskores::Id, 6> First;
};

double LargestSide(const Mesh& mesh)
{
  viskores::Vec3f_64 lo = mesh.Points[0], hi = mesh.Points[0];
  for (const auto& p : mesh.Points)
  {
    for (int a = 0; a < 3; ++a)
    {
      lo[a] = std::min(lo[a], p[a]);
      hi[a] = std::max(hi[a], p[a]);
    }
  }
  return std::max({ hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2] });
}

/// The curvature seeds with `epsilon` 1% of the largest side, against the
/// reference. The digest is the first 16 hexadecimal digits of the SHA-256 of the sorted distinct
/// seeds, joined by commas.
void CheckCurvatureSeeds(const std::string& name, const Mesh& mesh, const Pinned& pinned)
{
  std::cout << "Curvature seeds of the " << name << std::endl;
  qc::SurfaceMesh surface;
  qc::BuildSurfaceMesh(mesh.PointArray(), mesh.TriangleArray(), surface);
  const qc::CurvatureDensity density =
    qc::ComputeCurvatureDensity(surface, 0.01 * LargestSide(mesh));
  const viskores::Id floor = qc::TopologicalFloor(surface);
  const viskores::Id k = qc::CurvatureClusterCount(surface, density);
  std::vector<viskores::Id> seeds = qcd_test::ToVector(qc::CurvatureSeeds(density, k));
  std::sort(seeds.begin(), seeds.end());
  seeds.erase(std::unique(seeds.begin(), seeds.end()), seeds.end());
  std::ostringstream joined;
  for (std::size_t i = 0; i < seeds.size(); ++i)
  {
    joined << (i > 0 ? "," : "") << seeds[i];
  }
  const std::string digest = Sha256(joined.str()).substr(0, 16);
  std::cout << "  k_floor " << floor << ", total " << density.Total << ", k0 " << k << ", "
            << seeds.size() << " distinct seeds, digest " << digest << std::endl;
  VISKORES_TEST_ASSERT(floor == pinned.Floor, "The topological floor differs from the reference.");
  VISKORES_TEST_ASSERT(density.Total == pinned.Total,
                       "The total weight differs from the reference.");
  VISKORES_TEST_ASSERT(k == pinned.Seeds, "k0 differs from the reference.");
  VISKORES_TEST_ASSERT(seeds.size() >= pinned.First.size() &&
                         std::equal(pinned.First.begin(), pinned.First.end(), seeds.begin()),
                       "The first seeds differ from the reference.");
  VISKORES_TEST_ASSERT(digest == pinned.Digest, "The seeds differ from the reference.");
}

void TestCurvatureSeeds()
{
  CheckCurvatureSeeds("icosphere(3)",
                      qcd_test::Icosphere(3),
                      { 4, 126038616, 120, "c8a33d47668aeff0", { 12, 13, 24, 33, 36, 37 } });
  CheckCurvatureSeeds("torus(1, 0.12, 60, 8)",
                      qcd_test::Torus(1.0, 0.12, 60, 8),
                      { 7, 101580600, 97, "f7825786650a5786", { 8, 16, 18, 24, 28, 39 } });
  CheckCurvatureSeeds("voxel plate with 2 holes",
                      qcd_test::VoxelSolid(qcd_test::PlateWithHoles(2), 0.25),
                      { 9, 34873788, 33, "0c3e6ab433014e2a", { 0, 8, 11, 31, 35, 42 } });
}

void TestQuadricClusteringDecimationInitialization()
{
  // FIPS 180-4, example "abc".
  VISKORES_TEST_ASSERT(Sha256("abc") ==
                         "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
                       "SHA-256 is wrong.");
  TestCurvatureSeeds();
}
} // anonymous namespace

int UnitTestQuadricClusteringDecimationInitialization(int argc, char* argv[])
{
  return viskores::cont::testing::Testing::Run(
    TestQuadricClusteringDecimationInitialization, argc, argv);
}
