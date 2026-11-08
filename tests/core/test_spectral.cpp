// Tests for the spectral foundation: the 380-780 nm grid, CIE tables and illuminants.
#include <doctest/doctest.h>

#include <cmath>
#include <stdexcept>
#include <string>

#include "core/spectral.hpp"

using namespace openfilm;

namespace {
const SpectralTables& tables() {
  static const SpectralTables t = SpectralTables::load(OPENFILM_SOURCE_DATA_DIR "/spectral");
  return t;
}
}  // namespace

TEST_CASE("equal-energy spectrum integrates to Y=1 and x≈y≈1/3") {
  Spectrum eq;
  eq.fill(1.0);
  const Xyz xyz = integrate(eq, tables());
  CHECK(xyz.Y == doctest::Approx(1.0).epsilon(1e-9));
  const Xy xy = chromaticity(xyz);
  CHECK(std::abs(xy.x - 0.33333) < 0.0005);
  CHECK(std::abs(xy.y - 0.33333) < 0.0005);
}

TEST_CASE("daylight 6504 K is D65") {
  const Xy xy = chromaticity(integrate(tables().daylight(6504.0), tables()));
  CHECK(std::abs(xy.x - 0.3127) < 0.0001);
  CHECK(std::abs(xy.y - 0.3290) < 0.0001);
}

TEST_CASE("daylight 5503 K is D55") {
  const Xy xy = chromaticity(integrate(tables().daylight(5503.0), tables()));
  CHECK(std::abs(xy.x - 0.3324) < 0.0002);
  CHECK(std::abs(xy.y - 0.3474) < 0.0002);
}

TEST_CASE("daylight is normalised to 100 at 560 nm") {
  CHECK(tables().daylight(6504.0)[(560 - 380) / 5] == doctest::Approx(100.0));
}

TEST_CASE("blackbody 2856 K is Illuminant A") {
  const Xy xy = chromaticity(integrate(blackbody(2856.0), tables()));
  CHECK(std::abs(xy.x - 0.44757) < 0.0005);
  CHECK(std::abs(xy.y - 0.40745) < 0.0005);
  CHECK(blackbody(2856.0)[(560 - 380) / 5] == doctest::Approx(1.0));
}

TEST_CASE("resample interpolates and fills outside") {
  const Spectrum s = resample({{400.0, 1.0}, {410.0, 3.0}}, -7.0);
  CHECK(s[(405 - 380) / 5] == doctest::Approx(2.0));
  CHECK(s[0] == -7.0);
  CHECK(s[kSpectralSamples - 1] == -7.0);
}

TEST_CASE("grid helpers") {
  CHECK(wavelength(0) == 380.0);
  CHECK(wavelength(kSpectralSamples - 1) == 780.0);
  Spectrum a;
  a.fill(2.0);
  CHECK(sum(a) == doctest::Approx(2.0 * kSpectralSamples));
  CHECK(sum(a * a) == doctest::Approx(4.0 * kSpectralSamples));
  CHECK(sum(a * 0.5) == doctest::Approx(1.0 * kSpectralSamples));
}

TEST_CASE("uvPrime of equal energy white") {
  const auto uv = uvPrime({1.0, 1.0, 1.0});
  CHECK(uv[0] == doctest::Approx(4.0 / 19.0));
  CHECK(uv[1] == doctest::Approx(9.0 / 19.0));
}

TEST_CASE("load reports missing file") {
  try {
    SpectralTables::load("/nonexistent");
    FAIL("expected throw");
  } catch (const std::runtime_error& e) {
    CHECK(std::string(e.what()).find("cie_1931_2deg_cmf.csv") != std::string::npos);
  }
}
