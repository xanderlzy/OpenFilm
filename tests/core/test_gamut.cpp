// Tests for gamut compression into Rec.2020.
#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <random>

#include "core/color.hpp"
#include "core/gamut.hpp"

using namespace openfilm;

namespace {
Mat3 dwgToRec2020() { return mul(inverse(rgbToXyz(kRec2020)), rgbToXyz(kDaVinciWideGamut)); }

// How far a channel sits below the brightest channel, as a fraction of the brightest.
double distance(const Vec3& c, int ch) {
  const double ach = std::max({c[0], c[1], c[2]});
  return (ach - c[ch]) / ach;
}
}  // namespace

TEST_CASE("in-threshold colours unchanged") {
  std::mt19937 rng(1);
  std::uniform_real_distribution<double> u(0.05, 1.0);
  const Vec3 lim = dwgLimitsInRec2020();
  int checked = 0;
  while (checked < 500) {
    const Vec3 c{u(rng), u(rng), u(rng)};
    const double ach = std::max({c[0], c[1], c[2]});
    if (distance(c, 0) > 0.8 || distance(c, 1) > 0.8 || distance(c, 2) > 0.8) continue;
    const Vec3 o = gamutCompress(c, 0.5, lim);
    for (int i = 0; i < 3; ++i) CHECK(o[i] == doctest::Approx(c[i]).epsilon(1e-12).scale(ach));
    ++checked;
  }
}

TEST_CASE("output always inside Rec.2020") {
  std::mt19937 rng(2);
  std::uniform_real_distribution<double> u(0.0, 1.0);
  const Mat3 m = dwgToRec2020();
  const Vec3 lim = dwgLimitsInRec2020();
  std::vector<Vec3> inputs{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
  for (int i = 0; i < 2000; ++i) inputs.push_back({u(rng), u(rng), u(rng)});
  for (const Vec3& dwg : inputs)
    for (double amount : {0.0, 0.5, 1.0}) {
      const Vec3 o = gamutCompress(mul(m, dwg), amount, lim);
      for (int i = 0; i < 3; ++i) CHECK(o[i] >= -1e-12);
    }
}

TEST_CASE("monotonic along a saturation ramp") {
  const Mat3 m = dwgToRec2020();
  const Vec3 lim = dwgLimitsInRec2020();
  for (const Vec3 primary : {Vec3{1, 0, 0}, Vec3{0, 1, 0}, Vec3{0, 0, 1}}) {
    double prev[3] = {0, 0, 0};
    for (int s = 0; s <= 100; ++s) {
      const double t = s / 100.0;
      const Vec3 dwg{0.2 * (1 - t) + primary[0] * t, 0.2 * (1 - t) + primary[1] * t,
                     0.2 * (1 - t) + primary[2] * t};
      const Vec3 o = gamutCompress(mul(m, dwg), 0.5, lim);
      for (int c = 0; c < 3; ++c) {
        const double d = distance(o, c);
        CHECK(d >= prev[c] - 1e-12);
        prev[c] = d;
      }
    }
  }
}

TEST_CASE("amount 0 is hard clip") {
  const Vec3 o = gamutCompress({1.0, -0.5, 0.2}, 0.0, dwgLimitsInRec2020());
  CHECK(o[0] == doctest::Approx(1.0));
  CHECK(o[1] == doctest::Approx(0.0).epsilon(1e-12));
  CHECK(o[2] == doctest::Approx(0.2));
}

TEST_CASE("amount 1 compresses more than 0.5") {
  const Vec3 lim = dwgLimitsInRec2020();
  const Vec3 c{1.0, 0.1, -0.2};  // out of gamut: distance on blue is 1.2
  const Vec3 o1 = gamutCompress(c, 1.0, lim);
  const Vec3 o5 = gamutCompress(c, 0.5, lim);
  for (int ch = 0; ch < 3; ++ch) CHECK(distance(o1, ch) <= distance(o5, ch) + 1e-12);
}

TEST_CASE("black stays black") {
  const Vec3 o = gamutCompress({0, 0, 0}, 0.5, dwgLimitsInRec2020());
  CHECK(o == Vec3{0, 0, 0});
}

TEST_CASE("DWG limits are at least 1.0001 and cover the DWG primaries") {
  const Vec3 lim = dwgLimitsInRec2020();
  for (int c = 0; c < 3; ++c) CHECK(lim[c] >= 1.0001);
}

TEST_CASE("DWG limits cover the whole DWG cube surface") {
  // Mixtures of two primaries reach further from the achromatic axis than the primaries do.
  const Vec3 lim = dwgLimitsInRec2020();
  const Mat3 m = dwgToRec2020();
  double worst[3] = {0, 0, 0};
  for (int f = 0; f < 3; ++f)
    for (int i = 0; i <= 200; ++i)
      for (int j = 0; j <= 200; ++j) {
        Vec3 dwg;
        dwg[size_t(f)] = 1.0;
        dwg[size_t((f + 1) % 3)] = i / 200.0;
        dwg[size_t((f + 2) % 3)] = j / 200.0;
        const Vec3 c = mul(m, dwg);
        if (std::max({c[0], c[1], c[2]}) <= 0) continue;
        for (int ch = 0; ch < 3; ++ch) worst[ch] = std::max(worst[ch], distance(c, ch));
      }
  for (int ch = 0; ch < 3; ++ch) CHECK(lim[size_t(ch)] >= worst[ch]);
}
