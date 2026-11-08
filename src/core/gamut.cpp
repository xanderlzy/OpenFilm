#include "core/gamut.hpp"

#include <algorithm>
#include <cmath>

namespace openfilm {

Vec3 dwgLimitsInRec2020() {
  // Express each DWG primary in Rec.2020: these are the most saturated colours the input can
  // hold, so their distances are the worst case the compression must map to the gamut edge.
  const Mat3 m = mul(inverse(rgbToXyz(kRec2020)), rgbToXyz(kDaVinciWideGamut));
  Vec3 lim{1.0001, 1.0001, 1.0001};
  for (int p = 0; p < 3; ++p) {
    Vec3 dwg{0, 0, 0};
    dwg[p] = 1.0;
    const Vec3 c = mul(m, dwg);
    const double ach = std::max({c[0], c[1], c[2]});
    for (int ch = 0; ch < 3; ++ch) lim[ch] = std::max(lim[ch], (ach - c[ch]) / ach);
  }
  return lim;
}

Vec3 gamutCompress(const Vec3& c, double amount, const Vec3& limits) {
  constexpr double p = 1.2;  // how sharply the curve bends at the threshold
  const double ach = std::max({c[0], c[1], c[2]});
  if (ach <= 0.0) return {0, 0, 0};

  // Colours closer to neutral than the threshold are left exactly as they were.
  const double t = 1.0 - 0.35 * amount;
  Vec3 out;
  for (int i = 0; i < 3; ++i) {
    const double d = (ach - c[i]) / ach;
    double dn = d;
    if (d > t) {
      if (t >= 1.0) {
        dn = 1.0;  // amount 0: plain clip at the gamut edge
      } else {
        // Scale chosen so the widest distance the input can reach lands exactly on 1.0.
        const double lim = limits[i];
        const double s = (lim - t) / std::pow(std::pow((1.0 - t) / (lim - t), -p) - 1.0, 1.0 / p);
        const double x = (d - t) / s;
        dn = t + s * x / std::pow(1.0 + std::pow(x, p), 1.0 / p);
      }
    }
    dn = std::min(dn, 1.0);
    out[i] = ach - dn * ach;
  }
  return out;
}

}  // namespace openfilm
