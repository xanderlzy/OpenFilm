#include "core/grain_sim.hpp"

#include <algorithm>
#include <cmath>

#include "metal/grain_common.h"

namespace openfilm {

namespace {
constexpr double kPi = 3.14159265358979323846;

GrainAmplitudeTable tableOf(const GrainLayer& g, double sizeMul) {
  GrainAmplitudeTable t;
  if (g.density.empty()) return t;
  const double lo = g.density.front(), hi = g.density.back();
  t.lo = float(lo);
  t.hi = float(hi);
  for (int i = 0; i < kGrainTableSize; ++i)
    t.a[size_t(i)] = float(g.amplitude(lo + (hi - lo) * i / (kGrainTableSize - 1), sizeMul));
  return t;
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

float GrainAmplitudeTable::at(float D) const {
  if (!std::isfinite(D)) return 0;
  const float span = hi - lo;
  if (!(span > 0) || D <= lo) return a.front();
  if (D >= hi) return a.back();
  const float t = (D - lo) / span * float(kGrainTableSize - 1);
  const int i = std::min(int(t), kGrainTableSize - 2);
  const float f = t - float(i);
  return a[size_t(i)] + f * (a[size_t(i) + 1] - a[size_t(i)]);
}

GrainTables grainTables(const FilmGrain& grain, double sizeMul) {
  GrainTables t;
  for (size_t c = 0; c < 3; ++c) {
    t.negative[c] = tableOf(grain.negative[c], sizeMul);
    t.print[c] = tableOf(grain.print[c], sizeMul);
  }
  return t;
}

}  // namespace openfilm
