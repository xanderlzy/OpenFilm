// Camera-negative model: a DaVinci Intermediate pixel becomes the exposure each film layer
// receives, then the dye density the developed negative holds at that point.
#pragma once

#include "core/color.hpp"
#include "core/densitometry.hpp"
#include "core/rgb2spec.hpp"
#include "core/spectral.hpp"
#include "core/stock.hpp"

namespace openfilm {

class NegativeModel {
 public:
  // The tables, Rgb2Spec and Densitometer must outlive the model (held by reference).
  NegativeModel(const Stock& stock, const SpectralTables& tables, const Rgb2Spec& rgb2spec,
                const Densitometer& statusM, double gamutAmount);

  // DI code -> log10 exposure on the red, green and blue layers. A flat 18 % reflectance under
  // the stock's balance light lands exactly on the stock's mid-grey exposure on every layer.
  Vec3 logExposure(const Vec3& diCode) const;

  // Log exposures -> Status M density (R, G, B) read off each layer's H&D curve.
  Vec3 density(const Vec3& logE) const;

  // Spectral density of the dye stack (plus base) that reads as `statusMDensity`.
  Spectrum transmittanceDensity(const Vec3& statusMDensity) const;

  // The stock's characteristic curves (used to size baked 1D curves).
  const std::array<Curve, 3>& curves() const { return stock_.curve; }

 private:
  Stock stock_;
  const SpectralTables& tables_;
  const Rgb2Spec& rgb2spec_;
  const Densitometer& statusM_;
  double gamutAmount_;
  Vec3 limits_;
  Mat3 dwgToRec2020_;
  Mat3 interlayer_;         // layer cross-talk; identity in M2
  Spectrum illuminant_;     // balance light
  Vec3 whiteExposure_;      // sum(illuminant * sensitivity) per layer
};

}  // namespace openfilm
