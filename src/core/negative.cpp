#include "core/negative.hpp"

#include <algorithm>
#include <cmath>

#include "core/gamut.hpp"

namespace openfilm {

NegativeModel::NegativeModel(const Stock& stock, const SpectralTables& tables, const Rgb2Spec& rgb2spec,
                             const Densitometer& statusM, double gamutAmount)
    : stock_(stock), tables_(tables), rgb2spec_(rgb2spec), statusM_(statusM), gamutAmount_(gamutAmount),
      limits_(dwgLimitsInRec2020()),
      dwgToRec2020_(mul(inverse(rgbToXyz(kRec2020)), rgbToXyz(kDaVinciWideGamut))),
      interlayer_{{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}},
      illuminant_(blackbody(stock.balanceIlluminantK)) {
  for (int i = 0; i < 3; ++i) whiteExposure_[i] = sum(illuminant_ * stock_.sensitivity[i]);
}

Vec3 NegativeModel::logExposure(const Vec3& diCode) const {
  // DWG and Rec.2020 share the D65 white, so a plain matrix (no adaptation) converts between them.
  Vec3 dwg;
  for (int i = 0; i < 3; ++i) dwg[i] = diDecode(diCode[i]);
  const Vec3 rec2020 = gamutCompress(mul(dwgToRec2020_, dwg), gamutAmount_, limits_);

  // Light reaching the film = scene reflectance x the light the stock is balanced for.
  const Spectrum light = rgb2spec_.toSpectrum(rec2020) * illuminant_;

  // Dividing by the exposure of a perfect white makes 18 % grey land on the mid-grey anchor.
  Vec3 out;
  for (int i = 0; i < 3; ++i) {
    const double e = sum(light * stock_.sensitivity[i]) / whiteExposure_[i];
    out[i] = std::log10(std::max(e, 1e-10)) + stock_.midGreyLogExposure - std::log10(0.18);
  }
  return out;
}

Vec3 NegativeModel::density(const Vec3& logE) const {
  Vec3 d, dMin;
  for (int i = 0; i < 3; ++i) {
    d[i] = stock_.curve[i].eval(logE[i]);
    dMin[i] = stock_.curve[i].density.front();
  }
  // Cross-talk acts on density above each layer's fog level, which stays put.
  const Vec3 above{d[0] - dMin[0], d[1] - dMin[1], d[2] - dMin[2]};
  const Vec3 mixed = mul(interlayer_, above);
  return {mixed[0] + dMin[0], mixed[1] + dMin[1], mixed[2] + dMin[2]};
}

Spectrum NegativeModel::transmittanceDensity(const Vec3& statusMDensity) const {
  Vec3 amounts = dyeAmountsForDensity(statusM_, stock_.dye, stock_.base, statusMDensity);
  // Unreachable density combinations make the solver ask for negative dye, which cannot exist.
  for (double& a : amounts) a = std::max(a, 0.0);
  return spectralDensity(stock_.dye, stock_.base, amounts);
}

}  // namespace openfilm
