// Status A / Status M densitometry: how a densitometer "sees" a spectral density, and the
// inverse problem of finding dye amounts that read as a given RGB density.
#pragma once

#include <array>
#include <string>

#include "core/color.hpp"
#include "core/spectral.hpp"
#include "core/stock.hpp"

namespace openfilm {

class Densitometer {
 public:
  // Reads iso5_3_status_{a,m}.csv from spectralDir.
  static Densitometer load(const std::string& spectralDir, Densitometry kind);

  // Density as read through the R, G, B status filters (output order R, G, B).
  Vec3 density(const Spectrum& spectralDensity) const;

  // Linear filter weights, index 0 = R; exposed for the Newton solve's Jacobian.
  const std::array<Spectrum, 3>& weights() const { return w_; }

 private:
  std::array<Spectrum, 3> w_{};
  std::array<double, 3> wSum_{};
};

// Spectral density of a dye layer stack: base + sum of amount * unit dye.
Spectrum spectralDensity(const std::array<Spectrum, 3>& dye, const Spectrum& base, const Vec3& amounts);

// Dye amounts (C, M, Y) whose spectral density reads as targetRGB through the densitometer.
// Dyes need not peak at 1; amounts absorb whatever scale the dye data uses.
Vec3 dyeAmountsForDensity(const Densitometer& dm, const std::array<Spectrum, 3>& dye,
                          const Spectrum& base, const Vec3& targetRGB);

}  // namespace openfilm
