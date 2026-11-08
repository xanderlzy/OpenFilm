// Tests for the colour-space maths: DaVinci Intermediate, primaries->XYZ, Bradford, BT.1886.
#include <doctest/doctest.h>

#include <cmath>

#include "core/color.hpp"

using namespace openfilm;

namespace {
Xyz xyzFromXy(const Xy& c) { return {c.x / c.y, 1.0, (1.0 - c.x - c.y) / c.y}; }
}  // namespace

TEST_CASE("DI round trip") {
  // Log-spaced highlights, the linear toe near zero, and negatives (DWG allows them).
  for (int i = 0; i < 1000; ++i) {
    const double t = i / 999.0;
    const double x = (i % 10 == 9) ? -0.01 * t : 1e-4 * std::pow(1e6, t);
    CHECK(std::abs(diDecode(diEncode(x)) - x) <= 1e-6 * std::max(1.0, std::abs(x)));
  }
}

TEST_CASE("DI reference points") {
  CHECK(diEncode(0.18) == doctest::Approx(0.3360).epsilon(0.001 / 0.336));
  CHECK(diEncode(0.0) == 0.0);
}

TEST_CASE("rgbToXyz maps white to D65") {
  for (const Primaries* p : {&kDaVinciWideGamut, &kRec2020, &kRec709}) {
    const Vec3 w = mul(rgbToXyz(*p), Vec3{1, 1, 1});
    const Xy xy = chromaticity({w[0], w[1], w[2]});
    CHECK(std::abs(xy.x - 0.3127) < 1e-6);
    CHECK(std::abs(xy.y - 0.3290) < 1e-6);
    CHECK(w[1] == doctest::Approx(1.0).epsilon(1e-9));
  }
}

TEST_CASE("Rec709 matrix matches the published one") {
  const Mat3 m = rgbToXyz(kRec709);
  CHECK(std::abs(m[0][0] - 0.4124) < 1e-4);
  CHECK(std::abs(m[0][1] - 0.3576) < 1e-4);
  CHECK(std::abs(m[0][2] - 0.1805) < 1e-4);
}

TEST_CASE("bradford identity and white mapping") {
  const Xyz d65 = xyzFromXy(kD65);
  const Xyz d55 = xyzFromXy({0.33242, 0.34743});
  const Mat3 id = bradford(d65, d65);
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) CHECK(std::abs(id[i][j] - (i == j ? 1.0 : 0.0)) < 1e-9);
  const Vec3 out = mul(bradford(d55, d65), Vec3{d55.X, d55.Y, d55.Z});
  CHECK(std::abs(out[0] - d65.X) < 1e-9);
  CHECK(std::abs(out[1] - d65.Y) < 1e-9);
  CHECK(std::abs(out[2] - d65.Z) < 1e-9);
}

TEST_CASE("bt1886") {
  CHECK(std::abs(bt1886Encode(0.11706) - 0.409) < 0.0005);
  CHECK(bt1886Encode(-1.0) == 0.0);
}
