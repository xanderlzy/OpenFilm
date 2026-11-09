// Tests for the additive dye-cloud grain model (ruling R3): strength against the datasheet, the
// analytic aperture and pixel rms against brute-force simulations, Selwyn's law, the Size gain and
// the no-data case.
#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <string>
#include <vector>

#include "core/film_model.hpp"
#include "core/grain_model.hpp"

using namespace openfilm;

namespace {
const std::string kData = OPENFILM_SOURCE_DATA_DIR;
std::shared_ptr<const FilmResources> res() {
  static const auto r = FilmResources::load(kData, OPENFILM_COEFF_PATH);
  return r;
}

GrainLayer testLayer(double rMm = 0.004) {
  GrainLayer g;
  g.rMm = rMm;
  g.mu = 4;
  return g;
}

// One realisation of N (clouds covering each cell) on an n x n torus of cells of side pxMm,
// discs rasterised at cell centres.
std::vector<uint8_t> simulateCounts(const GrainLayer& g, int n, double pxMm, uint32_t seed) {
  const double rPx = g.rMm / pxMm, side = n * pxMm;
  std::mt19937 rng(seed);
  std::poisson_distribution<int> count(g.lambdaPerMm2() * side * side);
  std::uniform_real_distribution<double> pos(0.0, double(n));
  std::vector<uint8_t> N(size_t(n) * size_t(n), 0);
  const int discs = count(rng), reach = int(std::ceil(rPx));
  for (int k = 0; k < discs; ++k) {
    const double cx = pos(rng), cy = pos(rng);
    const int x0 = int(std::floor(cx)), y0 = int(std::floor(cy));
    for (int y = y0 - reach; y <= y0 + reach; ++y) {
      const double dy = y + 0.5 - cy;
      const size_t row = size_t((y % n + n) % n) * size_t(n);
      for (int x = x0 - reach; x <= x0 + reach; ++x) {
        const double dx = x + 0.5 - cx;
        if (dx * dx + dy * dy <= rPx * rPx) ++N[row + size_t((x % n + n) % n)];
      }
    }
  }
  return N;
}

double stdOf(const std::vector<double>& v) {
  double s = 0, s2 = 0;
  for (double x : v) { s += x; s2 += x * x; }
  const double m = s / double(v.size());
  return std::sqrt(std::max(0.0, s2 / double(v.size()) - m * m));
}

// Std of the mean of u = N - mu through a circular aperture of `apPx` cells' diameter, on a `stridePx` lattice.
double simulateApertureRms(const GrainLayer& g, int n, double pxMm, int apPx, int stridePx, uint32_t seed) {
  const std::vector<uint8_t> N = simulateCounts(g, n, pxMm, seed);
  std::vector<int32_t> prefix(size_t(n) * size_t(n + 1), 0);  // row prefix sums
  for (int y = 0; y < n; ++y)
    for (int x = 0; x < n; ++x)
      prefix[size_t(y) * size_t(n + 1) + size_t(x + 1)] = prefix[size_t(y) * size_t(n + 1) + size_t(x)] + N[size_t(y) * size_t(n) + size_t(x)];
  const double R = apPx / 2.0;
  std::vector<int> half(static_cast<size_t>(apPx));
  long area = 0;
  for (int j = 0; j < apPx; ++j) {
    const double dy = j + 0.5 - R;
    half[size_t(j)] = int(std::floor(std::sqrt(std::max(0.0, R * R - dy * dy)) + 0.5));
    area += 2 * half[size_t(j)];
  }
  std::vector<double> means;
  for (int y0 = 0; y0 + apPx <= n; y0 += stridePx)
    for (int x0 = 0; x0 + apPx <= n; x0 += stridePx) {
      const int cx = x0 + apPx / 2;
      long sum = 0;
      for (int j = 0; j < apPx; ++j) {
        const size_t row = size_t(y0 + j) * size_t(n + 1);
        sum += prefix[row + size_t(cx + half[size_t(j)])] - prefix[row + size_t(cx - half[size_t(j)])];
      }
      means.push_back(double(sum) / double(area) - g.mu);
    }
  return stdOf(means);
}

// Std of the mean of u over square pixels of `b` x `b` cells.
double simulatePixelRms(const GrainLayer& g, int n, double pxMm, int b, uint32_t seed) {
  const std::vector<uint8_t> N = simulateCounts(g, n, pxMm, seed);
  std::vector<double> means;
  for (int y0 = 0; y0 + b <= n; y0 += b)
    for (int x0 = 0; x0 + b <= n; x0 += b) {
      long sum = 0;
      for (int y = y0; y < y0 + b; ++y)
        for (int x = x0; x < x0 + b; ++x) sum += N[size_t(y) * size_t(n) + size_t(x)];
      means.push_back(double(sum) / double(b * b) - g.mu);
    }
  return stdOf(means);
}

void checkStock(const Stock& s, const std::array<GrainLayer, 3>& layers) {
  REQUIRE(s.granularity);
  const char* names[3] = {"red", "green", "blue"};
  for (size_t c = 0; c < 3; ++c) {
    const GranularityCurve& curve = s.granularity->layers[c];
    const GrainLayer& g = layers[c];
    CHECK(g.rMm == doctest::Approx(s.granularity->cloudRadiusUm / 1000));
    double worst = 0;
    for (size_t i = 0; i < curve.density.size(); ++i) {
      const double e = std::abs(g.amplitude(curve.density[i]) * g.unitApertureRms(0.048) / curve.sigma[i] - 1);
      worst = std::max(worst, e);
    }
    MESSAGE(s.id << " " << std::string(names[c]) << ": r " << g.rMm * 1000 << " um, amplitude " << g.amplitude(curve.density.front())
                 << " to " << g.amplitude(curve.density.back()) << ", max error " << worst * 100 << " %");
    CHECK(worst <= 0.01);
  }
}
}  // namespace

TEST_CASE("model matches the datasheet") {
  const FilmGrain grain = makeGrain(res()->stock("kodak-5219-type"), res()->stock("kodak-2383-type"));
  REQUIRE(grain.valid);
  checkStock(res()->stock("kodak-5219-type"), grain.negative);
  checkStock(res()->stock("kodak-2383-type"), grain.print);
  // Held at the ends, interpolated in log sigma between points, nothing for a non-finite density.
  const GrainLayer& g = grain.negative[0];
  CHECK(g.sigmaAt(-1) == g.sigma.front());
  CHECK(g.sigmaAt(9) == g.sigma.back());
  const double mid = 0.5 * (g.density[0] + g.density[1]);
  CHECK(g.sigmaAt(mid) == doctest::Approx(std::sqrt(g.sigma[0] * g.sigma[1])).epsilon(1e-12));
  CHECK(g.sigmaAt(std::nan("")) == 0);
}

TEST_CASE("aperture rms matches a brute-force simulation") {
  // r = 4 um, mu = 4; 2 mm square at 0.5 um; 48 um apertures (96 cells) on a 24 um lattice.
  const GrainLayer g = testLayer();
  const double sim = simulateApertureRms(g, 4000, 0.0005, 96, 48, 12345u);
  const double model = g.unitApertureRms(0.048);
  INFO("simulated " << sim << ", model " << model);
  CHECK(std::abs(sim / model - 1) <= 0.05);
}

TEST_CASE("pixel rms matches a brute-force simulation") {
  // r = 4 um, 4 um pixels (r = 1 px); 1 mm square at 0.25 um, 16 x 16 cells per pixel.
  const GrainLayer g = testLayer();
  const double sim = simulatePixelRms(g, 4000, 0.00025, 16, 777u);
  const double model = g.unitPixelRms(0.004);
  INFO("simulated " << sim << ", model " << model);
  CHECK(std::abs(sim / model - 1) <= 0.05);
}

TEST_CASE("selwyn") {
  // r = 4 um. Selwyn's law (rms x diameter constant) holds once the aperture is much larger than a
  // cloud; the exact model keeps an edge term of order r/R, so the reference is taken mid-range (40 r).
  const GrainLayer g = testLayer();
  const double ref = g.unitApertureRms(0.16) * 0.16;
  for (double D : {0.08, 0.12, 0.24, 0.32, 0.64, 4.0}) {
    INFO("aperture " << D);
    CHECK(std::abs(g.unitApertureRms(D) * D / ref - 1) <= 0.02);
  }
}

TEST_CASE("size gain keeps strength") {
  const GrainLayer g = makeGrain(res()->stock("kodak-5219-type"), res()->stock("kodak-2383-type")).negative[1];
  CHECK(g.unitApertureRms(0.048, 2) > g.unitApertureRms(0.048));  // larger clouds: coarser unit field
  for (double D : {0.5, 1.2, 2.0, 3.0}) {
    const double s1 = g.amplitude(D, 1) * g.unitApertureRms(0.048, 1), s2 = g.amplitude(D, 2) * g.unitApertureRms(0.048, 2);
    CHECK(std::abs(s2 / s1 - 1) <= 1e-6);
  }
}

TEST_CASE("no granularity → invalid") {
  Stock neg = res()->stock("kodak-5219-type"), print = res()->stock("kodak-2383-type");
  CHECK(makeGrain(neg, print).valid);
  neg.granularity.reset();
  CHECK_FALSE(makeGrain(neg, print).valid);
  neg = res()->stock("kodak-5219-type");
  print.granularity.reset();
  CHECK_FALSE(makeGrain(neg, print).valid);
}

TEST_CASE("unit field basics") {
  const GrainLayer g = testLayer();
  // Point variance of a Poisson count is its mean; the covariance vanishes beyond 2 r.
  CHECK(g.unitCovariance(0) == doctest::Approx(4).epsilon(1e-12));
  CHECK(g.unitCovariance(0.008) == doctest::Approx(0).epsilon(1e-12));
  CHECK(g.lambdaPerMm2(2) == doctest::Approx(g.lambdaPerMm2() / 4).epsilon(1e-12));
  // A pixel much smaller than a cloud sees nearly the point variance.
  CHECK(g.unitPixelRms(0.0001) == doctest::Approx(2).epsilon(0.03));
  CHECK(g.unitPixelRms(0.006) < g.unitPixelRms(0.003));
}
