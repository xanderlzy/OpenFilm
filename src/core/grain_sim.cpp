#include "core/grain_sim.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "metal/grain_common.h"

namespace openfilm {

namespace {
constexpr double kPi = 3.14159265358979323846;

GrainAmplitude amplitudeOf(const GrainLayer& g, double sizeMul) {
  GrainAmplitude a;
  if (g.density.size() > size_t(kGrainMaxNodes)) throw std::runtime_error("grain: more than kGrainMaxNodes nodes");
  a.count = int(g.density.size());
  for (size_t i = 0; i < g.density.size(); ++i) {
    a.density[i] = g.density[i];
    a.lnSigma[i] = std::log(g.sigma[i]);
  }
  const double unit = g.unitApertureRms(kGrainApertureMm, sizeMul);
  a.k = unit > 0 ? 1 / unit : 0.0;
  return a;
}
}  // namespace

uint32_t grainSeed(uint32_t frameSeed, GrainStage st, int layer) {
  return of_hash2(frameSeed, of_hash2(OF_U32(st), OF_U32(layer)));
}

GrainUnitParams grainUnitParams(const GrainLayer& g, double pxPerMm, double sizeMul) {
  GrainUnitParams p;
  const double r = g.rMm * sizeMul * pxPerMm;
  if (!(r > 0) || !std::isfinite(r)) return p;  // no clouds: u = 0
  p.rPx = float(r);
  p.lambdaPx = float(g.mu / (kPi * r * r));
  p.mu = float(g.mu);
  if (p.rPx < OF_GRAIN_MIN_RADIUS_PX) p.gaussRms = float(g.unitPixelRms(1 / pxPerMm, sizeMul));
  return p;
}

float grainUnit(const GrainUnitParams& p, int px, int py, uint32_t seed, int periodCells) {
  if (!(p.rPx > 0)) return 0;
  return of_grain_unit(px, py, p.rPx, p.lambdaPx, p.mu, p.gaussRms, seed, periodCells);
}

float grainUnit(const GrainLayer& g, double pxPerMm, double sizeMul, int px, int py, uint32_t seed, int periodCells) {
  return grainUnit(grainUnitParams(g, pxPerMm, sizeMul), px, py, seed, periodCells);
}

double GrainAmplitude::at(double D) const {
  if (!std::isfinite(D) || count <= 0) return 0;
  const auto lnS = [&]() {
    if (D <= density[0]) return lnSigma[0];
    const size_t n = size_t(count);
    if (D >= density[n - 1]) return lnSigma[n - 1];
    const size_t i = size_t(std::upper_bound(density.begin(), density.begin() + count, D) - density.begin());
    const double span = density[i] - density[i - 1];
    const double t = span > 0 ? (D - density[i - 1]) / span : 0.0;
    return lnSigma[i - 1] + t * (lnSigma[i] - lnSigma[i - 1]);
  };
  return std::exp(lnS()) * k;
}

GrainAmplitudes grainAmplitudes(const FilmGrain& grain, double sizeMul) {
  GrainAmplitudes a;
  for (size_t c = 0; c < 3; ++c) {
    a.negative[c] = amplitudeOf(grain.negative[c], sizeMul);
    a.print[c] = amplitudeOf(grain.print[c], sizeMul);
  }
  return a;
}

}  // namespace openfilm
