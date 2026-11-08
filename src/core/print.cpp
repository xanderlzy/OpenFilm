#include "core/print.hpp"

#include <algorithm>
#include <cmath>

namespace openfilm {

PrintModel::PrintModel(const Stock& print, const SpectralTables& tables, const Densitometer& statusA)
    : stock_(print), tables_(tables), statusA_(statusA), lamp_(blackbody(3200.0)),
      projector_(tables.daylight(5503.0)) {
  // Kodak's 2383 data sheet asks for a UV-absorbing Wratten 2B in the printer light. Without it
  // the print's blue layer (which is sensitive below 400 nm) sees light that never passed the
  // negative's yellow dye, which flattens blue and tints the grey ramp. The transmission below is
  // an approximation of the 2B's cut-on edge (about 50 % near 398 nm, clear by 410 nm), not
  // digitised from a published curve.
  static const double kTwoB[7][2] = {{380, 0.0},  {385, 0.01}, {390, 0.08}, {395, 0.35},
                                     {400, 0.70}, {405, 0.90}, {410, 1.0}};
  for (int i = 0; i < kSpectralSamples; ++i) {
    const double nm = wavelength(i);
    if (nm >= 410.0) continue;
    for (int k = 0; k < 6; ++k) {
      if (nm >= kTwoB[k][0] && nm <= kTwoB[k + 1][0]) {
        lamp_[i] *= kTwoB[k][1] + (nm - kTwoB[k][0]) / 5.0 * (kTwoB[k + 1][1] - kTwoB[k][1]);
        break;
      }
    }
  }
  for (int j = 0; j < 3; ++j) lampWhite_[j] = sum(lamp_ * stock_.sensitivity[j]);
  projectorWhiteY_ = integrate(projector_, tables_).Y;
}

Vec3 PrintModel::printLogExposure(const Spectrum& negSpectralDensity, const Vec3& offsetsLogE) const {
  // Light through the negative: the lamp dimmed by the negative's dyes at each wavelength.
  Spectrum lit;
  for (int i = 0; i < kSpectralSamples; ++i) lit[i] = lamp_[i] * std::pow(10.0, -negSpectralDensity[i]);
  Vec3 out;
  for (int j = 0; j < 3; ++j) {
    const double e = sum(lit * stock_.sensitivity[j]) / lampWhite_[j];
    out[j] = std::log10(std::max(e, 1e-10)) + offsetsLogE[j];
  }
  return out;
}

Vec3 PrintModel::printDensity(const Vec3& printLogE) const {
  Vec3 d;
  for (int j = 0; j < 3; ++j) d[j] = stock_.curve[j].eval(printLogE[j]);
  return d;
}

Xyz PrintModel::projectedXyz(const Vec3& printLogE) const {
  Vec3 amounts = dyeAmountsForDensity(statusA_, stock_.dye, stock_.base, printDensity(printLogE));
  // Negative dye is unphysical; unreachable density asks for it, so clip.
  for (double& a : amounts) a = std::max(a, 0.0);
  const Spectrum d = spectralDensity(stock_.dye, stock_.base, amounts);
  Spectrum seen;
  for (int i = 0; i < kSpectralSamples; ++i) seen[i] = projector_[i] * std::pow(10.0, -d[i]);
  const Xyz c = integrate(seen, tables_);
  return {c.X / projectorWhiteY_, c.Y / projectorWhiteY_, c.Z / projectorWhiteY_};
}

}  // namespace openfilm
