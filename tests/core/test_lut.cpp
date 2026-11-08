// Tests for the 3D LUT (tetrahedral interpolation) used to bake the film model.
#include <doctest/doctest.h>

#include <cmath>
#include <random>

#include "core/lut.hpp"

using namespace openfilm;

TEST_CASE("identity LUT is exact at nodes and close between") {
  Lut3D lut(17, {0, 0, 0}, {1, 1, 1});
  lut.bake([](const Vec3& v) { return v; });
  for (int i = 0; i < 17; i += 4) {
    const float c = float(i) / 16.f;
    const auto o = lut.lookup(c, 1.f - c, c);
    CHECK(o[0] == doctest::Approx(c).epsilon(1e-6));
    CHECK(o[1] == doctest::Approx(1.f - c).epsilon(1e-6));
  }
  std::mt19937 rng(3);
  std::uniform_real_distribution<float> u(0.f, 1.f);
  for (int i = 0; i < 500; ++i) {
    const float x = u(rng), y = u(rng), z = u(rng);
    const auto o = lut.lookup(x, y, z);
    CHECK(std::abs(o[0] - x) < 1e-6);
    CHECK(std::abs(o[1] - y) < 1e-6);
    CHECK(std::abs(o[2] - z) < 1e-6);
  }
}

TEST_CASE("tetrahedral reproduces linear functions exactly") {
  const Mat3 A{{{0.5, -0.2, 0.1}, {0.3, 0.9, -0.4}, {-0.1, 0.2, 0.7}}};
  const Vec3 b{0.1, -0.3, 0.2};
  Lut3D lut(9, {-1, 0, 2}, {1, 4, 3});  // non-unit domain too
  lut.bake([&](const Vec3& x) {
    Vec3 y = mul(A, x);
    for (int i = 0; i < 3; ++i) y[i] += b[i];
    return y;
  });
  std::mt19937 rng(5);
  std::uniform_real_distribution<float> u(0.f, 1.f);
  for (int i = 0; i < 500; ++i) {
    const float x = -1.f + 2.f * u(rng), y = 4.f * u(rng), z = 2.f + u(rng);
    Vec3 ref = mul(A, Vec3{x, y, z});
    const auto o = lut.lookup(x, y, z);
    for (int c = 0; c < 3; ++c) CHECK(std::abs(o[c] - (ref[c] + b[c])) < 1e-5);
  }
}

TEST_CASE("inputs clamp to domain") {
  Lut3D lut(5, {0, 0, 0}, {1, 1, 1});
  lut.bake([](const Vec3& v) { return Vec3{v[0] * v[0], v[1] + v[2], v[2]}; });
  CHECK(lut.lookup(-1.f, 2.f, 0.5f) == lut.lookup(0.f, 1.f, 0.5f));
}

TEST_CASE("data layout is RGB interleaved, red index fastest") {
  Lut3D lut(3, {0, 0, 0}, {1, 1, 1});
  lut.bake([](const Vec3& v) { return v; });
  const auto& d = lut.data();
  REQUIRE(d.size() == 3u * 27u);
  // Node (r=1,g=2,b=0): index ((0*3+2)*3+1)*3
  const size_t i = ((0 * 3 + 2) * 3 + 1) * 3;
  CHECK(d[i] == doctest::Approx(0.5f));
  CHECK(d[i + 1] == doctest::Approx(1.f));
  CHECK(d[i + 2] == doctest::Approx(0.f));
}
