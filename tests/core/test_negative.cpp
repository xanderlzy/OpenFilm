// Tests for the spectral camera-negative model.
#include <doctest/doctest.h>

#include <cmath>

#include "core/negative.hpp"

using namespace openfilm;

namespace {
const std::string kData = OPENFILM_SOURCE_DATA_DIR;

struct Fixture {
  SpectralTables tables = SpectralTables::load(kData + "/spectral");
  Rgb2Spec rgb2spec = Rgb2Spec::load(OPENFILM_COEFF_PATH);
  Stock stock = loadStock(kData + "/stocks/kodak-5219-type.json");
  Densitometer statusM = Densitometer::load(kData + "/spectral", Densitometry::StatusM);
  NegativeModel model(double amount) const { return NegativeModel(stock, tables, rgb2spec, statusM, amount); }
};
const Fixture& fx() { static const Fixture f; return f; }
Vec3 grey(double linear) { const double c = diEncode(linear); return {c, c, c}; }
}  // namespace

TEST_CASE("18% grey hits mid-grey log E on all layers") {
  const Vec3 e = fx().model(0.5).logExposure(grey(0.18));
  for (double v : e) CHECK(std::abs(v - fx().stock.midGreyLogExposure) < 1e-4);
}

TEST_CASE("+1 stop is +0.30103") {
  const NegativeModel m = fx().model(0.5);
  const Vec3 a = m.logExposure(grey(0.18)), b = m.logExposure(grey(0.36));
  for (int i = 0; i < 3; ++i) CHECK(std::abs((b[i] - a[i]) - 0.30103) < 1e-4);
}

TEST_CASE("black is finite and at toe") {
  const NegativeModel m = fx().model(0.5);
  const Vec3 e = m.logExposure({0, 0, 0});
  const Vec3 d = m.density(e);
  for (int i = 0; i < 3; ++i) {
    CHECK(std::isfinite(e[i]));
    CHECK(std::abs(d[i] - fx().stock.curve[i].density.front()) < 1e-9);
  }
}

TEST_CASE("density monotonic in exposure") {
  const NegativeModel m = fx().model(0.5);
  Vec3 prev{-1e9, -1e9, -1e9};
  for (int s = -80; s <= 80; ++s) {
    const Vec3 d = m.density(m.logExposure(grey(0.18 * std::pow(2.0, s / 10.0))));
    for (int i = 0; i < 3; ++i) { CHECK(d[i] >= prev[i] - 1e-12); prev[i] = d[i]; }
  }
}

TEST_CASE("transmittance reproduces density") {
  const NegativeModel m = fx().model(0.5);
  const Vec3 d = m.density(m.logExposure(grey(0.18)));
  const Vec3 r = fx().statusM.density(m.transmittanceDensity(d));
  for (int i = 0; i < 3; ++i) CHECK(std::abs(r[i] - d[i]) < 1e-5);
}

TEST_CASE("unreachable density gives non-negative dye") {
  const NegativeModel m = fx().model(0.5);
  const Spectrum s = m.transmittanceDensity({4.0, 0.0, 0.0});
  for (int i = 0; i < kSpectralSamples; ++i) {
    CHECK(std::isfinite(s[i]));
    CHECK(s[i] >= fx().stock.base[i] - 1e-12);
  }
}

TEST_CASE("extreme DWG primary is finite") {
  const Vec3 primaries[] = {{0, 0, 1}, {1, 0, 0}};
  for (double amount : {0.0, 1.0})
    for (const Vec3& p : primaries) {
      const NegativeModel m = fx().model(amount);
      const Vec3 e = m.logExposure({diEncode(p[0]), diEncode(p[1]), diEncode(p[2])});
      for (double v : e) CHECK(std::isfinite(v));
      for (double v : m.density(e)) CHECK(std::isfinite(v));
    }
}
