// Tests for Status A/M densitometry and the density -> dye amount solve.
#include <doctest/doctest.h>

#include <cmath>

#include "core/densitometry.hpp"

using namespace openfilm;

namespace {
const std::string kData = OPENFILM_SOURCE_DATA_DIR;
Spectrum flat(double v) { Spectrum s; s.fill(v); return s; }

void checkReproduces(const char* stockFile, Densitometry kind, const std::vector<Vec3>& targets) {
  const Stock s = loadStock(kData + "/stocks/" + stockFile);
  const Densitometer dm = Densitometer::load(kData + "/spectral", kind);
  for (const Vec3& t : targets) {
    const Vec3 c = dyeAmountsForDensity(dm, s.dye, s.base, t);
    const Vec3 r = dm.density(spectralDensity(s.dye, s.base, c));
    for (int i = 0; i < 3; ++i) CHECK(std::abs(r[i] - t[i]) < 1e-5);
  }
}
}  // namespace

TEST_CASE("zero density is zero") {
  for (auto k : {Densitometry::StatusA, Densitometry::StatusM}) {
    const Vec3 d = Densitometer::load(kData + "/spectral", k).density(flat(0.0));
    for (double v : d) CHECK(std::abs(v) < 1e-9);
  }
}

TEST_CASE("flat density reads flat") {
  for (auto k : {Densitometry::StatusA, Densitometry::StatusM}) {
    const Vec3 d = Densitometer::load(kData + "/spectral", k).density(flat(1.0));
    for (double v : d) CHECK(std::abs(v - 1.0) < 1e-9);
  }
}

TEST_CASE("dye amounts reproduce target") {
  checkReproduces("kodak-5219-type.json", Densitometry::StatusM,
                  {{0.5, 0.8, 1.0}, {1.2, 1.6, 2.0}, {2.5, 2.8, 3.0}});
}

TEST_CASE("dye amounts reproduce target for 2383 dyes (Status A)") {
  // 2383 dyes sit at the datasheet's printed scale, not peak 1; the solve must not care.
  checkReproduces("kodak-2383-type.json", Densitometry::StatusA, {{1.0, 1.0, 1.0}, {2.0, 2.0, 2.0}});
}
