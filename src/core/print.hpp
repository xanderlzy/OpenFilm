// Print stage: the developed negative is printed onto 2383-type print film and projected.
// This is the half of the look that turns negative density into the colours a viewer sees.
#pragma once

#include "core/color.hpp"
#include "core/densitometry.hpp"
#include "core/spectral.hpp"
#include "core/stock.hpp"

namespace openfilm {

class PrintModel {
 public:
  // The tables and Densitometer must outlive the model (held by reference).
  PrintModel(const Stock& print, const SpectralTables& tables, const Densitometer& statusA);

  // Spectral density of the negative -> log10 exposure on each print layer, as seen under the
  // 3200 K printer lamp, plus the printer-light offsets (log E; one printer point is 0.025).
  Vec3 printLogExposure(const Spectrum& negSpectralDensity, const Vec3& offsetsLogE) const;

  // Colour of the projected print: D55 light through the print's dyes. Y = 1 would be an
  // open gate (no film at all).
  Xyz projectedXyz(const Vec3& printLogE) const;

  // Status A density of the print for these exposures.
  Vec3 printDensity(const Vec3& printLogE) const;

  const Stock& stock() const { return stock_; }
  // The filtered printer lamp (3200 K behind a UV cut); exposed for tests and diagnostics.
  const Spectrum& lamp() const { return lamp_; }

 private:
  Stock stock_;
  const SpectralTables& tables_;
  const Densitometer& statusA_;
  Spectrum lamp_;            // printer lamp
  Spectrum projector_;       // projector light, D55
  Vec3 lampWhite_;           // sum(lamp * sensitivity) per print layer
  double projectorWhiteY_;   // Y of the open-gate projector light, used to normalise
};

}  // namespace openfilm
