// Preview grain on the CPU: precomputed tiling plates of the dye-cloud unit field u (ruling R3).
// u does not depend on density, so one plate per stage and layer suffices; the grain stage adds
// Amount * a(D) * plate sample, exactly the amplitude Full uses. A frame reads each plate through a
// per-frame offset and orientation (of_plate_coord, src/metal/grain_common.h, shared with the GPU).
#pragma once

#include <array>
#include <cstdint>
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

// Plates for a stock pair at a scale (px/mm) and Size. Plate seeds are frame-independent
// (grainSeed(0, stage, layer)); the cell grid wraps so each plate tiles. Multithreaded.
std::shared_ptr<const GrainPlates> makePlates(const FilmGrain& g, double pxPerMm, double sizeMul, int size = kPlateSize);
// Process-wide cache of the last 2 sets, keyed by the layers' cloud parameters (the only inputs of a
// plate), pxPerMm, sizeMul and size. Thread-safe.
std::shared_ptr<const GrainPlates> cpuPlates(const FilmGrain& g, double pxPerMm, double sizeMul, int size = kPlateSize);
// Plate size grainStage uses (default kPlateSize); tests only.
void setCpuPlateSizeForTests(int size);
int cpuPlateSize();

}  // namespace openfilm
