// Tests for the print stage on its own (printer lamp -> 2383 -> projector).
#include <doctest/doctest.h>

#include <cmath>

#include "core/negative.hpp"
#include "core/print.hpp"

using namespace openfilm;

namespace {
const std::string kData = OPENFILM_SOURCE_DATA_DIR;
struct Fixture {
  SpectralTables tables = SpectralTables::load(kData + "/spectral");
  Stock print = loadStock(kData + "/stocks/kodak-2383-type.json");
  Densitometer statusA = Densitometer::load(kData + "/spectral", Densitometry::StatusA);
  PrintModel model() const { return PrintModel(print, tables, statusA); }
};
const Fixture& fx() { static const Fixture f; return f; }
Spectrum flat(double d) { Spectrum s; s.fill(d); return s; }
}  // namespace

TEST_CASE("clear negative prints at zero log E plus offsets") {
  const Vec3 e = fx().model().printLogExposure(flat(0.0), {0.1, -0.2, 0.0});
  CHECK(std::abs(e[0] - 0.1) < 1e-12);
  CHECK(std::abs(e[1] + 0.2) < 1e-12);
  CHECK(std::abs(e[2]) < 1e-12);
}

TEST_CASE("neutral density 0.3 on the negative costs 0.3 log E") {
  const Vec3 e = fx().model().printLogExposure(flat(0.3), {0, 0, 0});
  for (double v : e) CHECK(std::abs(v + 0.3) < 1e-9);
}

TEST_CASE("more print exposure gives a darker projection") {
  const PrintModel m = fx().model();
  // Print is positive-working: more exposure -> more density -> less projected light.
  const double yLow = m.projectedXyz({-2, -2, -2}).Y, yHigh = m.projectedXyz({0, 0, 0}).Y;
  CHECK(yHigh < yLow);
  CHECK(yLow <= 1.0);
  CHECK(yHigh > 0.0);
}

TEST_CASE("print density reads back from the curves") {
  const PrintModel m = fx().model();
  const Vec3 d = m.printDensity({-1, -1, -1});
  for (int j = 0; j < 3; ++j) CHECK(d[j] == doctest::Approx(fx().print.curve[j].eval(-1)));
}

TEST_CASE("printer lamp has a Wratten 2B-style UV cut") {
  // Kodak's 2383 data sheet calls for a UV-absorbing filter (Wratten 2B) in the printer light.
  const Spectrum raw = blackbody(3200.0);
  const Spectrum lamp = fx().model().lamp();
  for (int i = 0; i < kSpectralSamples; ++i) {
    const double nm = wavelength(i);
    if (nm <= 385) CHECK(lamp[i] <= 0.02 * raw[i]);
    if (nm >= 420) CHECK(lamp[i] == doctest::Approx(raw[i]));
  }
}
