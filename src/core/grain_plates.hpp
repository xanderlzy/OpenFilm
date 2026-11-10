// Preview grain on the CPU: precomputed tiling plates of the dye-cloud unit field u (ruling R3).
// u does not depend on density, so one plate per stage and layer suffices; the grain stage adds
// Amount * a(D) * plate sample, exactly the amplitude Full uses. A frame reads each plate through a
// per-frame offset and orientation (of_plate_coord, src/metal/grain_common.h, shared with the GPU).
#pragma once

#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

#include "core/grain_sim.hpp"

namespace openfilm {

constexpr int kPlateSize = 2048;

// IEEE 754 half <-> float, round-to-nearest-even on store. The GPU reads the same bits.
uint16_t floatToHalf(float f);
float halfToFloat(uint16_t h);

struct GrainPlates {
  int size = 0;                                                // plate side, px
  std::array<std::array<std::vector<uint16_t>, 3>, 2> plates;  // [stage][layer] -> size*size half u values
  // Unit-field sample (u, before a(D)) at frame pixel (px, py); `seed` = grainSeed(frameSeed, st, layer).
  float sample(GrainStage st, int layer, int px, int py, uint32_t seed) const;
};

// How one plate is generated: the layer's cloud parameters at this scale and Size, its cell grid
// (periodCells cells of cellPx per side, wrapping so the plate tiles; 0 cells when there are no
// clouds) and its frame-independent seed grainSeed(0, stage, layer). makePlates and the GPU's
// plate generator both start from it.
struct PlateLayerParams {
  GrainUnitParams unit;
  int periodCells = 0;
  float cellPx = 0;
  uint32_t seed = 0;
};
PlateLayerParams plateLayerParams(const GrainLayer& g, double pxPerMm, double sizeMul, int size, GrainStage st,
                                  int layer);

// What a plate set depends on: every layer's (rMm, mu), pxPerMm, sizeMul and size (FilmGrain has no
// stock id; these are the plates' only inputs). The CPU and GPU caches are keyed by it.
struct PlateKey {
  std::array<double, 12> layers{};  // rMm, mu of each of the 6 layers
  double pxPerMm = 0, sizeMul = 0;
  int size = 0;
  bool operator==(const PlateKey& o) const {
    return layers == o.layers && pxPerMm == o.pxPerMm && sizeMul == o.sizeMul && size == o.size;
  }
};
PlateKey plateKey(const FilmGrain& g, double pxPerMm, double sizeMul, int size);

// Plates for a stock pair at a scale (px/mm) and Size. Plate seeds are frame-independent
// (grainSeed(0, stage, layer)); the cell grid wraps so each plate tiles. Multithreaded.
std::shared_ptr<const GrainPlates> makePlates(const FilmGrain& g, double pxPerMm, double sizeMul, int size = kPlateSize);
// Process-wide cache of the last 2 sets used, keyed by plateKey. Thread-safe: a key not yet cached is
// built once, by the first caller, with no lock held; other callers asking for that key wait for it,
// callers asking for other keys do not.
std::shared_ptr<const GrainPlates> cpuPlates(const FilmGrain& g, double pxPerMm, double sizeMul, int size = kPlateSize);
// The cache's bookkeeping, for tests: sets it holds, and sets it has built since the process started.
size_t cpuPlateSetCount();
size_t cpuPlateBuilds();
// Plate size grainStage uses (default kPlateSize); tests only.
void setCpuPlateSizeForTests(int size);
// Tests only: `hook` runs in cpuPlates once a set is built, before it is published; it may throw to
// simulate a failure at that point (an empty function removes it).
void setCpuPlateBuildHookForTests(std::function<void()> hook);
int cpuPlateSize();

}  // namespace openfilm
