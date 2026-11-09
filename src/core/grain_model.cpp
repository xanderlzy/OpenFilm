#include "core/grain_model.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <string>

namespace openfilm {

namespace {
constexpr double kPi = 3.14159265358979323846;
constexpr int kQuadIntervals = 128;  // Simpson intervals over h in [0, 2r]; even

// Area common to two discs of radius r whose centres are h apart (also the disc's covariogram).
double discOverlap(double r, double h) {
  if (h >= 2 * r) return 0;
  return 2 * r * r * std::acos(h / (2 * r)) - 0.5 * h * std::sqrt(4 * r * r - h * h);
}

// Covariogram of a square of side a averaged over directions: the mean over theta of
// (a - h|cos|)+ (a - h|sin|)+, integrated in closed form over the angles where both factors are positive.
double squareCovariogram(double a, double h) {
  const double q = std::min(1.0, a / h);
  const double t1 = h <= a ? 0.0 : std::acos(q), t2 = h <= a ? kPi / 2 : std::asin(q);
  if (t2 <= t1) return 0;
  const auto F = [&](double t) {
    const double s = std::sin(t);
    return a * a * t - a * h * s + a * h * std::cos(t) + 0.5 * h * h * s * s;
  };
  return (2 / kPi) * (F(t2) - F(t1));
}

// Variance of the mean of u over a window with isotropic covariogram `cov` and area `area`:
// (1/area^2) * integral over h in [0, 2r] of C(h) cov(h) 2 pi h dh.
template <class Covariogram>
double windowVariance(const GrainLayer& g, double sizeMul, double area, Covariogram cov) {
  const double r = g.rMm * sizeMul;
  if (!(r > 0) || !(area > 0)) return 0;
  const double step = 2 * r / kQuadIntervals;
  double sum = 0;
  for (int i = 0; i <= kQuadIntervals; ++i) {
    const double h = i * step;
    const double w = (i == 0 || i == kQuadIntervals) ? 1 : (i % 2 ? 4 : 2);
    sum += w * g.unitCovariance(h, sizeMul) * cov(h) * h;
  }
  return 2 * kPi * sum * step / 3 / (area * area);
}

GrainLayer layerOf(const GranularityCurve& c, double cloudRadiusMm) {
  if (!(cloudRadiusMm > 0)) throw std::invalid_argument("granularity: cloud radius must be positive");
  std::vector<size_t> idx(std::min(c.density.size(), c.sigma.size()));
  std::iota(idx.begin(), idx.end(), size_t(0));
  std::stable_sort(idx.begin(), idx.end(), [&](size_t a, size_t b) { return c.density[a] < c.density[b]; });
  GrainLayer g;
  g.rMm = cloudRadiusMm;
  for (size_t i : idx) {
    if (!(c.sigma[i] > 0) || !std::isfinite(c.density[i])) continue;  // log sigma needs sigma > 0
    g.density.push_back(c.density[i]);
    g.sigma.push_back(c.sigma[i]);
  }
  if (g.density.empty()) throw std::invalid_argument("granularity: no usable points");
  if (g.density.size() > size_t(kGrainMaxNodes))
    throw std::runtime_error("granularity: more than " + std::to_string(kGrainMaxNodes) + " points in one layer");
  return g;
}

std::array<GrainLayer, 3> layersOf(const Stock& s) {
  std::array<GrainLayer, 3> layers;
  for (size_t c = 0; c < 3; ++c) layers[c] = layerOf(s.granularity->layers[c], s.granularity->cloudRadiusUm / 1000);
  return layers;
}
}  // namespace

double GrainLayer::sigmaAt(double D) const {
  if (!std::isfinite(D) || density.empty()) return 0;
  if (D <= density.front()) return sigma.front();
  if (D >= density.back()) return sigma.back();
  const size_t i = size_t(std::upper_bound(density.begin(), density.end(), D) - density.begin());  // density[i-1] <= D < density[i]
  const double span = density[i] - density[i - 1];
  const double t = span > 0 ? (D - density[i - 1]) / span : 0.0;
  return std::exp(std::log(sigma[i - 1]) + t * (std::log(sigma[i]) - std::log(sigma[i - 1])));
}

double GrainLayer::lambdaPerMm2(double sizeMul) const {
  const double r = rMm * sizeMul;
  return r > 0 ? mu / (kPi * r * r) : 0.0;
}

double GrainLayer::unitCovariance(double hMm, double sizeMul) const {
  return lambdaPerMm2(sizeMul) * discOverlap(rMm * sizeMul, std::abs(hMm));
}

double GrainLayer::unitApertureRms(double apertureDiameterMm, double sizeMul) const {
  const double R = apertureDiameterMm / 2;
  return std::sqrt(std::max(0.0, windowVariance(*this, sizeMul, kPi * R * R, [R](double h) { return discOverlap(R, h); })));
}

double GrainLayer::unitPixelRms(double pixelMm, double sizeMul) const {
  return std::sqrt(std::max(0.0, windowVariance(*this, sizeMul, pixelMm * pixelMm,
                                                [pixelMm](double h) { return squareCovariogram(pixelMm, h); })));
}

double GrainLayer::amplitude(double D, double sizeMul) const {
  const double unit = unitApertureRms(kGrainApertureMm, sizeMul);
  return unit > 0 ? sigmaAt(D) / unit : 0.0;
}

FilmGrain makeGrain(const Stock& negative, const Stock& print) {
  FilmGrain out;
  if (!negative.granularity || !print.granularity) return out;
  // Only stocks with granularity are stored, so a stock without it never hits the cache.
  static std::mutex mutex;
  static std::map<std::string, std::array<GrainLayer, 3>> cache;
  const auto get = [](const Stock& s) {
    std::lock_guard<std::mutex> lock(mutex);
    auto it = cache.find(s.id);
    if (it == cache.end()) it = cache.emplace(s.id, layersOf(s)).first;
    return it->second;
  };
  out.negative = get(negative);
  out.print = get(print);
  out.valid = true;
  return out;
}

}  // namespace openfilm
