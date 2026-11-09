// Film optics in millimetres on the film plane: per-layer sharpness (light scatter in the emulsion,
// plus the development edge effect on the negative) fitted to the datasheet MTF curves, the halation and bloom
// lobes, and the gate widths that turn millimetres into pixels.
#pragma once

#include <array>
#include <vector>

#include "core/stock.hpp"

namespace openfilm {

// A normalised Gaussian carrying `weight` of the light; sigma in mm on the film.
struct Lobe {
  double weight;
  double sigmaMm;
};

// Scatter is a sum of lobes (weights sum to 1). The edge effect (adjacency) sharpens by adding back
// k times the detail lost to a wider Gaussian of sigma edgeSigmaMm: an unsharp mask.
struct LayerSharpness {
  std::vector<Lobe> scatter;
  double edgeK = 0;
  double edgeSigmaMm = 0;
};

// MTF of a normalised Gaussian: exp(-2 pi^2 sigma^2 f^2).
double gaussianMtf(double sigmaMm, double cyclesPerMm);

// MTF_s(f) * (1 + k (1 - MTF_A(f))), MTF_s the scatter lobes, MTF_A the edge Gaussian.
double sharpnessResponse(const LayerSharpness& s, double cyclesPerMm);

// Least-squares fit of `scatterLobes` (2 or 3) scatter lobes (and, with withEdgeEffect, the edge gain
// bounded to [0, 0.3] and a width bounded to [0.005, 0.5] mm) to a datasheet curve. Lobes come back sorted by sigma,
// ascending. Deterministic: Nelder-Mead from fixed starts, best kept.
LayerSharpness fitSharpness(const MtfCurve& mtf, bool withEdgeEffect, int scatterLobes = 2);

struct FilmOptics {
  std::array<LayerSharpness, 3> negative;  // 2 scatter lobes + edge effect
  // 3 scatter lobes, no edge term (edgeK == 0). Ruling R14: the fitted print edge effect (sigma ~0.5 mm)
  // tinted every mid-contrast edge with a wide green halo, so the print is scatter-only.
  std::array<LayerSharpness, 3> print;
  Halation halation;                       // from the negative stock
};

// Throws std::runtime_error when the negative has no halation data.
FilmOptics makeOptics(const Stock& negative, const Stock& print);

// Two lobes bracketing the total-internal-reflection ring (radius about 1.83 t for base thickness t).
std::vector<Lobe> halationLobes(double baseThicknessMm, double radiusMultiplier);
std::vector<Lobe> bloomLobes(double radiusMm);

enum class FilmFormat { Super35, Academy35, Super16, Std16, Mm65, Custom };

// Camera-gate width in mm. Custom is clamped to [5, 80]; a non-finite custom width gives Super 35.
double gateWidthMm(FilmFormat format, double customMm);
double pixelsPerMm(double imageWidthPx, double gateMm);

}  // namespace openfilm
