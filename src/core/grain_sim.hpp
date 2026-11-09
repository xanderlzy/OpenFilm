// Full grain on the CPU (ruling R3): the per-pixel dye-cloud unit field of one layer, the seeds
// that make every frame, stage and layer draw its own clouds, and the amplitude a(D) the grain
// stage scales it by. The sampler itself lives in src/metal/grain_common.h, shared with the
// GPU, so both run the same operations.
#pragma once

#include <array>
#include <cstdint>

#include "core/grain_model.hpp"

namespace openfilm {

enum class GrainStage { Negative = 0, Print = 1 };

// Seed of one layer's clouds: hash of (frameSeed, stage, layer).
uint32_t grainSeed(uint32_t frameSeed, GrainStage st, int layer);

// One layer's sampler constants at one scale (px/mm) and Size, in pixels. Building them runs a
// quadrature (gaussRms), so build once per frame, never per pixel.
struct GrainUnitParams {
  float rPx = 0;       // cloud radius rMm * sizeMul * pxPerMm
  float lambdaPx = 0;  // cloud centres per px^2
  float mu = 0;        // mean overlap count
  float gaussRms = 0;  // pixel-area rms of u; used when rPx < 0.25 (OF_GRAIN_MIN_RADIUS_PX)
};
GrainUnitParams grainUnitParams(const GrainLayer& g, double pxPerMm, double sizeMul);

// u_pixel of the pixel (px, py) in absolute frame coordinates: the mean overlap count over its 4 x 4
// sub-samples minus mu, or gaussRms times a standard normal for sub-pixel clouds. Zero mean; the
// grain stage multiplies it by Amount * a(D). periodCells > 0 wraps the cell grid (plates).
float grainUnit(const GrainUnitParams& p, int px, int py, uint32_t seed, int periodCells = 0);
// Convenience form building the params on every call (a quadrature): for tests, not per pixel.
float grainUnit(const GrainLayer& g, double pxPerMm, double sizeMul, int px, int py, uint32_t seed,
                int periodCells = 0);

// Ruling R4: the amplitude a(D) = sigmaAt(D) * k, k = 1 / unitApertureRms(0.048 mm, Size), exact at
// every density. Held as plain data for the GPU: the layer's datasheet nodes (density ascending, and
// the NATURAL log of sigma), at most kGrainMaxNodes (makeGrain enforces it), and k.
struct GrainAmplitude {
  int count = 0;
  std::array<double, kGrainMaxNodes> density{}, lnSigma{};
  double k = 0;
  // GrainLayer::sigmaAt(D) * k, with sigmaAt's arithmetic: linear in (density, ln sigma) between nodes,
  // held at the end nodes outside. 0 for a non-finite D or no nodes.
  double at(double D) const;
};
struct GrainAmplitudes {
  std::array<GrainAmplitude, 3> negative, print;  // layer order as FilmGrain
  const std::array<GrainAmplitude, 3>& stage(GrainStage st) const { return st == GrainStage::Negative ? negative : print; }
};
// Builds k with a quadrature per layer: once per frame, never per pixel.
GrainAmplitudes grainAmplitudes(const FilmGrain& grain, double sizeMul);

}  // namespace openfilm
