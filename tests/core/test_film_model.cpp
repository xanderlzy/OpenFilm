// Tests for the whole film chain and its auto-balanced printer lights.
#include <doctest/doctest.h>

#include <cmath>
#include <cstdio>
#include <limits>
#include <random>

#include "core/film_model.hpp"

using namespace openfilm;

namespace {
const std::string kData = OPENFILM_SOURCE_DATA_DIR;
std::shared_ptr<const FilmResources> res() {
  static const auto r = FilmResources::load(kData, OPENFILM_COEFF_PATH);
  return r;
}
FilmParams params() {
  FilmParams p;
  p.negativeId = "kodak-5219-type";
  p.printId = "kodak-2383-type";
  return p;
}
LookControls look(double stops = 0) { LookControls c; c.exposureStops = stops; return c; }
Vec3 grey(double linear) { const double c = diEncode(linear); return {c, c, c}; }
double dUv(const Vec3& code) {
  // Display code -> linear Rec.709 -> XYZ -> u'v' distance from D65.
  Vec3 lin;
  for (int i = 0; i < 3; ++i) lin[i] = std::pow(std::max(code[i], 1e-9), 2.4);
  const Vec3 x = mul(rgbToXyz(kRec709), lin);
  const auto uv = uvPrime({x[0], x[1], x[2]});
  const auto w = uvPrime({kD65.x / kD65.y, 1.0, (1 - kD65.x - kD65.y) / kD65.y});
  return std::hypot(uv[0] - w[0], uv[1] - w[1]);
}
}  // namespace

TEST_CASE("grey prints neutral at code 0.409") {
  const FilmModel m(res(), params());
  const Vec3 out = m.evaluate(grey(0.18), look());
  for (double v : out) CHECK(std::abs(v - 0.409) < 0.005);
  CHECK(dUv(out) < 0.001);
  std::printf("balance %.6f %.6f %.6f anchorScale %.6f out %.5f %.5f %.5f\n", m.balance()[0], m.balance()[1],
              m.balance()[2], m.anchorScale(), out[0], out[1], out[2]);
}

TEST_CASE("printer point adds 0.025 log E") {
  const FilmModel m(res(), params());
  LookControls c; c.printerPoints = {1, 0, 0};
  CHECK(std::abs((m.printerOffsets(c)[0] - m.printerOffsets(LookControls{})[0]) - 0.025) < 1e-9);
  CHECK(std::abs(m.printerOffsets(c)[1] - m.printerOffsets(LookControls{})[1]) < 1e-12);
}

TEST_CASE("print density moves all three") {
  const FilmModel m(res(), params());
  LookControls a, b;
  a.printDensity = 1;
  b.printerPoints = {1, 1, 1};
  for (int j = 0; j < 3; ++j) CHECK(std::abs(m.printerOffsets(a)[j] - m.printerOffsets(b)[j]) < 1e-12);
  const Vec3 oa = m.evaluate(grey(0.18), a), ob = m.evaluate(grey(0.18), b);
  for (int j = 0; j < 3; ++j) CHECK(oa[j] == doctest::Approx(ob[j]));
}

TEST_CASE("grey ramp neutral and monotonic") {
  // Controller ruling (replaces the spec's 0.003): u'v' distance < 0.008 within +-1 stop and
  // < 0.012 within +-2 stops. What is left after the UV and edge-held density fixes is attributed
  // to unmodelled printer optics (heat-absorbing glass) and datasheet digitising tolerance
  // (~0.02 D per channel); to be revisited in the M7 fidelity report.
  const FilmModel m(res(), params());
  double worst = 0;
  for (int q = -8; q <= 8; ++q) {
    const double stops = q * 0.25;
    const double d = dUv(m.evaluate(grey(0.18), look(stops)));
    worst = std::max(worst, d);
    if (q % 4 == 0) std::printf("stops %+.0f du'v' %.5f\n", stops, d);
    CHECK(d < (std::abs(stops) <= 1.0 ? 0.008 : 0.012));
  }
  std::printf("worst du'v' within +-2 stops: %.5f\n", worst);
  Vec3 prev{-1, -1, -1};
  for (int q = -24; q <= 24; ++q) {
    const Vec3 o = m.evaluate(grey(0.18), look(q * 0.25));
    for (int j = 0; j < 3; ++j) { CHECK(o[j] >= prev[j] - 1e-9); prev[j] = o[j]; }
  }
}

TEST_CASE("black is finite") {
  const FilmModel m(res(), params());
  const Vec3 black = m.evaluate({0, 0, 0}, look()), dark = m.evaluate(grey(0.18), look(-6));
  for (int j = 0; j < 3; ++j) {
    CHECK(std::isfinite(black[j]));
    CHECK(black[j] >= 0);
    CHECK(black[j] < dark[j]);
  }
}

TEST_CASE("DWG primaries finite") {
  for (double amount : {0.0, 1.0}) {
    FilmParams p = params();
    p.gamutCompression = amount;
    const FilmModel m(res(), p);
    const Vec3 prim[] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
    for (const Vec3& v : prim) {
      const Vec3 o = m.evaluate({diEncode(v[0]), diEncode(v[1]), diEncode(v[2])}, look());
      for (double c : o) { CHECK(std::isfinite(c)); CHECK(c >= 0); }
    }
  }
}

TEST_CASE("unknown stock id throws") {
  CHECK_THROWS(res()->stock("nope"));
  FilmParams p = params();
  p.printId = "nope";
  CHECK_THROWS(FilmModel(res(), p));
}

TEST_CASE("print output factors through print log E and print density") {
  // printOutput must equal the split chain exactly: this is what lets the bake split the table.
  const FilmModel m(res(), params());
  std::mt19937 rng(5);
  std::uniform_real_distribution<double> ud(0.1, 2.5), up(-20, 20);
  for (int i = 0; i < 200; ++i) {
    const Vec3 d{ud(rng), ud(rng), ud(rng)};
    LookControls c; c.printerPoints = {up(rng), up(rng), up(rng)}; c.printDensity = up(rng);
    const Vec3 off = m.printerOffsets(c);
    Vec3 e = m.printLogExposureBase(d);
    for (int k = 0; k < 3; ++k) e[k] += off[k];
    const Vec3 split = m.displayFromPrintDensity(m.printDensity(e));
    const Vec3 direct = m.printOutput(d, c);
    for (int k = 0; k < 3; ++k) CHECK(split[k] == doctest::Approx(direct[k]).epsilon(1e-12));
  }
}

TEST_CASE("print offsets: balance plus 0.025 per point, non-finite counts as zero") {
  LookControls c; c.printerPoints = {4, 0, -2}; c.printDensity = 1;
  const Vec3 o = printOffsetsLogE({0.1, 0.2, 0.3}, c);
  CHECK(o[0] == doctest::Approx(0.1 + 0.025 * 5));
  CHECK(o[1] == doctest::Approx(0.2 + 0.025 * 1));
  CHECK(o[2] == doctest::Approx(0.3 + 0.025 * -1));
  c.printerPoints[1] = std::numeric_limits<double>::quiet_NaN(); c.printDensity = INFINITY;
  const Vec3 n = printOffsetsLogE({0, 0, 0}, c);
  CHECK(n[1] == 0.0);
  CHECK(n[0] == doctest::Approx(0.1));
}

TEST_CASE("params hash holds only bake inputs") {
  FilmParams a = params(), b = params();
  CHECK(a.hash() == b.hash());
  b.gamutCompression = 0.6; CHECK(a.hash() != b.hash());
  b = params(); b.printId = "x"; CHECK(a.hash() != b.hash());
  b = params(); b.negativeId = "x"; CHECK(a.hash() != b.hash());
}

TEST_CASE("green-layer exposure is smooth for saturated reds and magentas") {
  // Regression for the jagged spectra near the Rec.2020 edge: a fine sweep of red must move
  // green-layer log E in one direction only (up to a tiny epsilon), not wobble.
  const FilmModel m(res(), params());
  int reversals = 0;
  double prev = m.negativeLogExposure({0.80, 0.0834, 0.7028})[1], prevDelta = 0;
  for (int i = 1; i <= 491; ++i) {  // 0.80 .. 0.92 in 1/4096 steps
    const double g = m.negativeLogExposure({0.80 + i / 4096.0, 0.0834, 0.7028})[1];
    const double delta = g - prev;
    if (std::abs(delta) > 1e-6 && std::abs(prevDelta) > 1e-6 && (delta > 0) != (prevDelta > 0)) ++reversals;
    if (std::abs(delta) > 1e-6) prevDelta = delta;
    prev = g;
  }
  CHECK(reversals == 0);
}
