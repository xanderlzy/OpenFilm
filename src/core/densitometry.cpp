#include "core/densitometry.hpp"

#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace openfilm {

Densitometer Densitometer::load(const std::string& spectralDir, Densitometry kind) {
  const std::string path =
      spectralDir + (kind == Densitometry::StatusA ? "/iso5_3_status_a.csv" : "/iso5_3_status_m.csv");
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot open densitometry file: " + path);

  // CSV columns are nm,blue,green,red; the densitometer reports R,G,B, so reorder on read.
  // An empty cell means the standard tabulates nothing there: weight 0.
  std::vector<std::array<double, 2>> pts[3];
  std::string line;
  bool headerSeen = false;
  while (std::getline(in, line)) {
    if (line.empty() || line[0] == '#') continue;
    if (!headerSeen) { headerSeen = true; continue; }
    std::vector<std::string> cells;
    std::stringstream ss(line);
    std::string cell;
    while (std::getline(ss, cell, ',')) cells.push_back(cell);
    if (line.back() == ',') cells.push_back("");
    if (cells.size() != 4) throw std::runtime_error(path + ": wrong column count in line: " + line);
    try {
      const double nm = std::stod(cells[0]);
      for (int ch = 0; ch < 3; ++ch) {
        const std::string& c = cells[3 - ch];  // R=col 3, G=col 2, B=col 1
        pts[ch].push_back({nm, c.empty() ? 0.0 : std::pow(10.0, std::stod(c))});
      }
    } catch (const std::exception&) {
      throw std::runtime_error(path + ": bad number in line: " + line);
    }
  }
  Densitometer d;
  for (int ch = 0; ch < 3; ++ch) {
    if (pts[ch].size() < 2) throw std::runtime_error(path + ": too few rows");
    // Weights are interpolated in linear space so a missing cell fades to 0 rather than to -inf.
    d.w_[ch] = resample(pts[ch], 0.0);
    d.wSum_[ch] = sum(d.w_[ch]);
    if (!(d.wSum_[ch] > 0.0)) throw std::runtime_error(path + ": filter has no weight");
  }
  return d;
}

Vec3 Densitometer::density(const Spectrum& D) const {
  Vec3 out{};
  for (int ch = 0; ch < 3; ++ch) {
    double t = 0.0;
    for (int i = 0; i < kSpectralSamples; ++i) t += w_[ch][i] * std::pow(10.0, -D[i]);
    out[ch] = -std::log10(t / wSum_[ch]);
  }
  return out;
}

Spectrum spectralDensity(const std::array<Spectrum, 3>& dye, const Spectrum& base, const Vec3& c) {
  Spectrum s = base;
  for (int i = 0; i < kSpectralSamples; ++i)
    for (int j = 0; j < 3; ++j) s[i] += c[j] * dye[j][i];
  return s;
}

Vec3 dyeAmountsForDensity(const Densitometer& dm, const std::array<Spectrum, 3>& dye,
                          const Spectrum& base, const Vec3& target) {
  auto err = [&](const Vec3& c) {
    const Vec3 r = dm.density(spectralDensity(dye, base, c));
    return Vec3{r[0] - target[0], r[1] - target[1], r[2] - target[2]};
  };
  auto norm = [](const Vec3& e) { return std::max({std::abs(e[0]), std::abs(e[1]), std::abs(e[2])}); };

  // Start: each unit dye's status density as a matrix (rows R,G,B; columns C,M,Y), solved
  // against the target minus the base. Exact if density were additive; close enough to seed Newton.
  Mat3 M{};
  const Spectrum zero{};
  const Vec3 baseRead = dm.density(base);
  for (int j = 0; j < 3; ++j) {
    Vec3 unit{};
    unit[j] = 1.0;
    const Vec3 r = dm.density(spectralDensity(dye, zero, unit));
    for (int i = 0; i < 3; ++i) M[i][j] = r[i];
  }
  Vec3 c = mul(inverse(M), Vec3{target[0] - baseRead[0], target[1] - baseRead[1], target[2] - baseRead[2]});

  Vec3 best = c;
  double bestErr = norm(err(c));
  for (int iter = 0; iter < 20 && bestErr >= 1e-6; ++iter) {
    // Jacobian of the reading: dD_i/dc_j = sum(w_i T dye_j) / sum(w_i T), T = 10^-density.
    const Spectrum S = spectralDensity(dye, base, c);
    Mat3 J{};
    for (int i = 0; i < 3; ++i) {
      double den = 0.0;
      double num[3] = {0, 0, 0};
      for (int k = 0; k < kSpectralSamples; ++k) {
        const double wt = dm.weights()[i][k] * std::pow(10.0, -S[k]);
        den += wt;
        for (int j = 0; j < 3; ++j) num[j] += wt * dye[j][k];
      }
      for (int j = 0; j < 3; ++j) J[i][j] = num[j] / den;
    }
    const Vec3 e = err(c);
    const Vec3 step = mul(inverse(J), e);
    // Halve the step until the error actually shrinks, so a bad linearisation cannot diverge.
    double scale = 1.0;
    bool improved = false;
    for (int tries = 0; tries < 8; ++tries, scale *= 0.5) {
      const Vec3 cn{c[0] - scale * step[0], c[1] - scale * step[1], c[2] - scale * step[2]};
      const double en = norm(err(cn));
      if (en < bestErr) { c = cn; best = cn; bestErr = en; improved = true; break; }
    }
    if (!improved) break;
  }
  return best;
}

}  // namespace openfilm
