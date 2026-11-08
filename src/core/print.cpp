#include "core/print.hpp"

#include <algorithm>
#include <cmath>

namespace openfilm {

PrintModel::PrintModel(const Stock& print, const SpectralTables& tables, const Densitometer& statusA)
    : stock_(print), tables_(tables), statusA_(statusA), lamp_(blackbody(3200.0)),
      projector_(tables.daylight(5503.0)) {
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
