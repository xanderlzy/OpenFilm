#include "core/grain_plates.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <deque>
#include <functional>
#include <future>
#include <mutex>
#include <stdexcept>

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

PlateLayerParams plateLayerParams(const GrainLayer& g, double pxPerMm, double sizeMul, int size, GrainStage st,
                                  int layer) {
  PlateLayerParams p;
  p.unit = grainUnitParams(g, pxPerMm, sizeMul);
  p.seed = grainSeed(0, st, layer);
  if (p.unit.rPx > 0) {
    p.periodCells = std::max(1, int(std::lround(double(size) / double(p.unit.rPx))));
    p.cellPx = float(size) / float(p.periodCells);
  }
  return p;
}

PlateKey plateKey(const FilmGrain& g, double pxPerMm, double sizeMul, int size) {
  PlateKey k;
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

namespace {
std::vector<uint16_t> makePlate(const PlateLayerParams& p, int size) {
  std::vector<uint16_t> out(size_t(size) * size_t(size), 0);
  if (!(p.unit.rPx > 0)) return out;  // no clouds: u = 0
  const bool sim = p.unit.rPx >= OF_GRAIN_MIN_RADIUS_PX;
  detail::parallelRows(size, [&](int y0, int y1) {
    for (int y = y0; y < y1; ++y)
      for (int x = 0; x < size; ++x) {
        const float u = sim ? of_overlap_mean(x, y, p.unit.rPx, p.unit.lambdaPx, p.cellPx, p.seed, p.periodCells) -
                                  p.unit.mu
                            : p.unit.gaussRms * of_gauss(x, y, p.seed);
        out[size_t(y) * size_t(size) + size_t(x)] = floatToHalf(u);
      }
  });
  return out;
}

std::atomic<int> gPlateSize{kPlateSize};

// The cpuPlates cache. `ready` holds the last two sets used, most recent first; `building` the keys
// being built right now, each with the future its builder fulfils.
using PlatesPtr = std::shared_ptr<const GrainPlates>;
struct PlateCache {
  std::mutex mu;
  std::deque<std::pair<PlateKey, PlatesPtr>> ready;
  std::vector<std::pair<PlateKey, std::shared_future<PlatesPtr>>> building;
  size_t builds = 0;
};
PlateCache& plateCache() {
  static PlateCache c;
  return c;
}

// Tests only: runs in cpuPlates after a set is built and before it is published.
std::mutex gHookMu;
std::function<void()> gBuildHook;
void runBuildHook() {
  std::function<void()> hook;
  {
    std::lock_guard<std::mutex> lock(gHookMu);
    hook = gBuildHook;
  }
  if (hook) hook();
}
}  // namespace

std::shared_ptr<const GrainPlates> makePlates(const FilmGrain& g, double pxPerMm, double sizeMul, int size) {
  auto pl = std::make_shared<GrainPlates>();
  pl->size = size;
  for (int st = 0; st < 2; ++st)
    for (int c = 0; c < 3; ++c) {
      const auto& layers = st == 0 ? g.negative : g.print;
      pl->plates[size_t(st)][size_t(c)] =
          makePlate(plateLayerParams(layers[size_t(c)], pxPerMm, sizeMul, size, GrainStage(st), c), size);
    }
  return pl;
}

std::shared_ptr<const GrainPlates> cpuPlates(const FilmGrain& g, double pxPerMm, double sizeMul, int size) {
  PlateCache& c = plateCache();
  const PlateKey key = plateKey(g, pxPerMm, sizeMul, size);
  std::promise<PlatesPtr> promise;
  std::shared_future<PlatesPtr> pending;  // another caller's build of this key
  {
    std::lock_guard<std::mutex> lock(c.mu);
    for (auto it = c.ready.begin(); it != c.ready.end(); ++it)
      if (it->first == key) {
        PlatesPtr hit = it->second;
        c.ready.erase(it);
        c.ready.emplace_front(key, hit);  // most recently used first
        return hit;
      }
    for (const auto& b : c.building)
      if (b.first == key) pending = b.second;
    if (!pending.valid()) {
      c.building.emplace_back(key, promise.get_future().share());
      ++c.builds;
    }
  }
  if (pending.valid()) return pending.get();  // waits with no lock held; rethrows the builder's failure
  // This caller builds; the set is published (and the waiters woken) only once it is complete. Whatever
  // throws on the way (the build, the hook, an allocation while publishing), the in-progress entry is
  // removed and the waiters get the failure, so the next request for the key builds again.
  try {
    PlatesPtr made = makePlates(g, pxPerMm, sizeMul, size);
    runBuildHook();
    PlatesPtr released;  // freed after the lock is dropped
    {
      std::lock_guard<std::mutex> lock(c.mu);
      c.ready.emplace_front(key, made);
      if (c.ready.size() > 2) {
        released = std::move(c.ready.back().second);
        c.ready.pop_back();
      }
      c.building.erase(
          std::find_if(c.building.begin(), c.building.end(), [&](const auto& b) { return b.first == key; }));
    }
    promise.set_value(made);
    return made;
  } catch (...) {
    const std::exception_ptr failure = std::current_exception();
    try {
      std::lock_guard<std::mutex> lock(c.mu);
      auto it = std::find_if(c.building.begin(), c.building.end(), [&](const auto& b) { return b.first == key; });
      if (it != c.building.end()) c.building.erase(it);
    } catch (...) {
    }
    promise.set_exception(failure);
    throw;
  }
}

size_t cpuPlateSetCount() {
  std::lock_guard<std::mutex> lock(plateCache().mu);
  return plateCache().ready.size();
}

size_t cpuPlateBuilds() {
  std::lock_guard<std::mutex> lock(plateCache().mu);
  return plateCache().builds;
}

void setCpuPlateBuildHookForTests(std::function<void()> hook) {
  std::lock_guard<std::mutex> lock(gHookMu);
  gBuildHook = std::move(hook);
}

void setCpuPlateSizeForTests(int size) { gPlateSize = size > 0 ? size : kPlateSize; }
int cpuPlateSize() { return gPlateSize; }

}  // namespace openfilm
