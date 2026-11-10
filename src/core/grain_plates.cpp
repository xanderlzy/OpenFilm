#include "core/grain_plates.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <deque>
#include <mutex>

#include "core/pixel_ops.hpp"
#include "metal/grain_common.h"

namespace openfilm {

uint16_t floatToHalf(float f) {
  uint32_t x;
  std::memcpy(&x, &f, 4);
  const uint32_t sign = (x >> 16) & 0x8000u;
  const uint32_t abs = x & 0x7fffffffu;
  if (abs >= 0x7f800000u) return uint16_t(sign | (abs > 0x7f800000u ? 0x7e00u : 0x7c00u));  // NaN / inf
  if (abs >= 0x477ff000u) return uint16_t(sign | 0x7c00u);  // rounds to >= 65520: inf
  if (abs < 0x33000001u) return uint16_t(sign);             // <= 2^-25: rounds to zero
  uint32_t mant = (abs & 0x7fffffu), exp = abs >> 23;
  if (exp < 113) {  // half subnormal
    mant |= 0x800000u;
    const uint32_t shift = 126 - exp;  // 14..24
    uint32_t r = mant >> shift;
    const uint32_t rem = mant & ((1u << shift) - 1), halfway = 1u << (shift - 1);
    if (rem > halfway || (rem == halfway && (r & 1))) ++r;
    return uint16_t(sign | r);
  }
  uint32_t h = ((exp - 112) << 10) | (mant >> 13);
  const uint32_t rem = mant & 0x1fffu;
  if (rem > 0x1000u || (rem == 0x1000u && (h & 1))) ++h;  // carry into the exponent is correct
  return uint16_t(sign | h);
}

float halfToFloat(uint16_t h) {
  const uint32_t sign = uint32_t(h & 0x8000u) << 16;
  uint32_t exp = (h >> 10) & 0x1f, mant = h & 0x3ffu, bits;
  if (exp == 0x1f) {
    bits = sign | 0x7f800000u | (mant << 13);
  } else if (exp != 0) {
    bits = sign | ((exp + 112) << 23) | (mant << 13);
  } else if (mant == 0) {
    bits = sign;
  } else {  // subnormal
    exp = 113;
    while (!(mant & 0x400u)) { mant <<= 1; --exp; }
    bits = sign | (exp << 23) | ((mant & 0x3ffu) << 13);
  }
  float f;
  std::memcpy(&f, &bits, 4);
  return f;
}

float GrainPlates::sample(GrainStage st, int layer, int px, int py, uint32_t seed) const {
  const OF_U32 c = of_plate_coord(px, py, seed, size);
  return halfToFloat(plates[size_t(st)][size_t(layer)][size_t(c >> 16) * size_t(size) + size_t(c & 0xffffu)]);
}

namespace {
std::vector<uint16_t> makePlate(const GrainLayer& g, double pxPerMm, double sizeMul, int size, uint32_t seed) {
  std::vector<uint16_t> out(size_t(size) * size_t(size), 0);
  const GrainUnitParams p = grainUnitParams(g, pxPerMm, sizeMul);
  if (!(p.rPx > 0)) return out;  // no clouds: u = 0
  const bool sim = p.rPx >= OF_GRAIN_MIN_RADIUS_PX;
  const int periodCells = std::max(1, int(std::lround(double(size) / double(p.rPx))));
  const float cellPx = float(size) / float(periodCells);
  detail::parallelRows(size, [&](int y0, int y1) {
    for (int y = y0; y < y1; ++y)
      for (int x = 0; x < size; ++x) {
        const float u = sim ? of_overlap_mean(x, y, p.rPx, p.lambdaPx, cellPx, seed, periodCells) - p.mu
                            : p.gaussRms * of_gauss(x, y, seed);
        out[size_t(y) * size_t(size) + size_t(x)] = floatToHalf(u);
      }
  });
  return out;
}

struct Key {
  std::array<double, 12> layers;  // rMm, mu of each of the 6 layers
  double pxPerMm, sizeMul;
  int size;
  bool operator==(const Key& o) const {
    return layers == o.layers && pxPerMm == o.pxPerMm && sizeMul == o.sizeMul && size == o.size;
  }
};
Key keyOf(const FilmGrain& g, double pxPerMm, double sizeMul, int size) {
  Key k{};
  size_t i = 0;
  for (const auto* st : {&g.negative, &g.print})
    for (const GrainLayer& l : *st) {
      k.layers[i++] = l.rMm;
      k.layers[i++] = l.mu;
    }
  k.pxPerMm = pxPerMm;
  k.sizeMul = sizeMul;
  k.size = size;
  return k;
}

std::atomic<int> gPlateSize{kPlateSize};
}  // namespace

std::shared_ptr<const GrainPlates> makePlates(const FilmGrain& g, double pxPerMm, double sizeMul, int size) {
  auto pl = std::make_shared<GrainPlates>();
  pl->size = size;
  for (int st = 0; st < 2; ++st)
    for (int c = 0; c < 3; ++c) {
      const auto& layers = st == 0 ? g.negative : g.print;
      pl->plates[size_t(st)][size_t(c)] =
          makePlate(layers[size_t(c)], pxPerMm, sizeMul, size, grainSeed(0, GrainStage(st), c));
    }
  return pl;
}

std::shared_ptr<const GrainPlates> cpuPlates(const FilmGrain& g, double pxPerMm, double sizeMul, int size) {
  static std::mutex mu;
  static std::deque<std::pair<Key, std::shared_ptr<const GrainPlates>>> cache;
  const Key key = keyOf(g, pxPerMm, sizeMul, size);
  std::lock_guard<std::mutex> lock(mu);  // held while building: concurrent frames share one build
  for (const auto& e : cache)
    if (e.first == key) return e.second;
  auto made = makePlates(g, pxPerMm, sizeMul, size);
  cache.emplace_front(key, made);
  if (cache.size() > 2) cache.pop_back();
  return made;
}

void setCpuPlateSizeForTests(int size) { gPlateSize = size > 0 ? size : kPlateSize; }
int cpuPlateSize() { return gPlateSize; }

}  // namespace openfilm
