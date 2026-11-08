#include "core/gamut.hpp"

#include <algorithm>
#include <cmath>

namespace openfilm {

Vec3 dwgLimitsInRec2020() {
  // Walk the three faces of the DWG cube that touch white (one channel at 1, the others 0..1) in
  // Rec.2020. Mixtures of two primaries reach further out than the primaries themselves, so
  // scanning only the corners under-estimates the limit and hard-clips real colours at the edge.
  const Mat3 m = mul(inverse(rgbToXyz(kRec2020)), rgbToXyz(kDaVinciWideGamut));
  Vec3 lim{1.0001, 1.0001, 1.0001};
  constexpr int n = 400;
  for (int f = 0; f < 3; ++f)
    for (int i = 0; i <= n; ++i)
      for (int j = 0; j <= n; ++j) {
        Vec3 dwg;
        dwg[size_t(f)] = 1.0;
        dwg[size_t((f + 1) % 3)] = double(i) / n;
        dwg[size_t((f + 2) % 3)] = double(j) / n;
        const Vec3 c = mul(m, dwg);
        const double ach = std::max({c[0], c[1], c[2]});
        if (ach <= 0) continue;
        // Small margin: grid sampling can miss the exact peak between nodes.
        for (int ch = 0; ch < 3; ++ch) lim[size_t(ch)] = std::max(lim[size_t(ch)], (ach - c[size_t(ch)]) / ach * 1.0001);
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
