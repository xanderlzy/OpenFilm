// Tests for the per-layer sharpness fit, the halation/bloom lobes and the film-format gate widths.
#include <doctest/doctest.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>

#include "core/lut.hpp"
#include "core/optics.hpp"

using namespace openfilm;

namespace {
const std::string kData = OPENFILM_SOURCE_DATA_DIR;
std::shared_ptr<const FilmResources> res() {
  static const auto r = FilmResources::load(kData, OPENFILM_COEFF_PATH);
  return r;
}
constexpr double kPi = 3.14159265358979323846;

void checkFit(const MtfCurve& mtf, bool withEdge, int lobes, double bar, const char* label) {
  const LayerSharpness s = fitSharpness(mtf, withEdge, lobes);
  double worst = 0, worstF = 0;
  for (const auto& p : mtf) {
    const double e = std::abs(sharpnessResponse(s, p[0]) - p[1]);
    if (e > worst) { worst = e; worstF = p[0]; }
  }
  INFO(std::string(label) << ": worst |error| " << worst << " at " << worstF << " c/mm");
  CHECK(worst <= bar);
  REQUIRE(s.scatter.size() == size_t(lobes));
  double sum = 0;
  for (const auto& l : s.scatter) {
    sum += l.weight;
    CHECK(l.sigmaMm > 0);
  }
  for (size_t i = 1; i < s.scatter.size(); ++i) CHECK(s.scatter[i - 1].sigmaMm <= s.scatter[i].sigmaMm);
  CHECK(std::abs(sum - 1) < 1e-9);
  CHECK(s.edgeK >= 0);
  if (withEdge) {
    CHECK(s.edgeSigmaMm >= 0.005);
    CHECK(s.edgeSigmaMm <= 0.5);
  }
  else CHECK(s.edgeK == 0);
}
}  // namespace

TEST_CASE("fit matches the 5219 datasheet") {
  const Stock& neg = res()->stock("kodak-5219-type");
  const char* names[3] = {"5219 red", "5219 green", "5219 blue"};
  for (int c = 0; c < 3; ++c) checkFit(neg.mtf[size_t(c)], true, 2, 0.05, names[c]);
}

TEST_CASE("fit matches the 2383 datasheet") {
  const Stock& print = res()->stock("kodak-2383-type");
  const char* names[3] = {"2383 red", "2383 green", "2383 blue"};
  for (int c = 0; c < 3; ++c) checkFit(print.mtf[size_t(c)], true, 3, 0.05, names[c]);
}

TEST_CASE("fit is deterministic") {
  const auto same = [](const LayerSharpness& a, const LayerSharpness& b) {
    REQUIRE(a.scatter.size() == b.scatter.size());
    for (size_t i = 0; i < a.scatter.size(); ++i) {
      CHECK(std::memcmp(&a.scatter[i].weight, &b.scatter[i].weight, sizeof(double)) == 0);
      CHECK(std::memcmp(&a.scatter[i].sigmaMm, &b.scatter[i].sigmaMm, sizeof(double)) == 0);
    }
    CHECK(std::memcmp(&a.edgeK, &b.edgeK, sizeof(double)) == 0);
    CHECK(std::memcmp(&a.edgeSigmaMm, &b.edgeSigmaMm, sizeof(double)) == 0);
  };
  const Stock& neg = res()->stock("kodak-5219-type");
  same(fitSharpness(neg.mtf[1], true), fitSharpness(neg.mtf[1], true));
  const Stock& print = res()->stock("kodak-2383-type");
  same(fitSharpness(print.mtf[2], true, 3), fitSharpness(print.mtf[2], true, 3));
}

TEST_CASE("gaussian mtf") {
  for (double s : {0.001, 0.01, 0.2}) {
    CHECK(gaussianMtf(s, 0) == 1.0);
    const double f = std::sqrt(std::log(2.0) / (2 * kPi * kPi)) / s;
    CHECK(std::abs(gaussianMtf(s, f) - 0.5) < 1e-12);
  }
}

TEST_CASE("lobes are normalised") {
  const auto sum = [](const std::vector<Lobe>& v) {
    double s = 0;
    for (const auto& l : v) s += l.weight;
    return s;
  };
  CHECK(std::abs(sum(halationLobes(0.127, 1)) - 1) < 1e-12);
  CHECK(std::abs(sum(bloomLobes(0.5)) - 1) < 1e-12);
  const auto a = halationLobes(0.127, 1), b = halationLobes(0.127, 2);
  REQUIRE(a.size() == b.size());
  for (size_t i = 0; i < a.size(); ++i) {
    CHECK(b[i].sigmaMm == 2 * a[i].sigmaMm);
    CHECK(b[i].weight == a[i].weight);
  }
}

TEST_CASE("formats") {
  CHECK(gateWidthMm(FilmFormat::Super35, 0) == 24.89);
  CHECK(gateWidthMm(FilmFormat::Academy35, 0) == 21.95);
  CHECK(gateWidthMm(FilmFormat::Super16, 0) == 12.52);
  CHECK(gateWidthMm(FilmFormat::Std16, 0) == 10.26);
  CHECK(gateWidthMm(FilmFormat::Mm65, 0) == 52.48);
  CHECK(gateWidthMm(FilmFormat::Custom, 3) == 5);
  CHECK(gateWidthMm(FilmFormat::Custom, 100) == 80);
  CHECK(gateWidthMm(FilmFormat::Custom, 30) == 30);
  CHECK(gateWidthMm(FilmFormat::Custom, std::numeric_limits<double>::quiet_NaN()) == 24.89);
  CHECK(gateWidthMm(FilmFormat::Custom, std::numeric_limits<double>::infinity()) == 24.89);
  CHECK(pixelsPerMm(3840, 24.89) == 3840 / 24.89);
}

TEST_CASE("bake carries optics") {
  FilmParams p;
  p.negativeId = "kodak-5219-type";
  p.printId = "kodak-2383-type";
  const auto film = bakeFilm(FilmModel(res(), p), 17, 17, 17);
  CHECK(!film->optics.negative[0].scatter.empty());
  CHECK(film->optics.negative[0].scatter.size() == 2);
  CHECK(film->optics.print[0].scatter.size() == 3);
  CHECK(film->optics.halation.baseThicknessMm == 0.127);
  for (int c = 0; c < 3; ++c) CHECK(film->optics.print[size_t(c)].edgeK >= 0);
}
