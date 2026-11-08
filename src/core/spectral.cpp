#include "core/spectral.hpp"

#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace openfilm {

namespace {

// Reads a CSV with `#` comment lines, one header row, then nm,v1..vN rows. Requires exactly the
// 81 grid wavelengths so a damaged or re-sampled data file is caught at load time.
std::vector<std::vector<double>> readGridCsv(const std::string& path, int valueColumns) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot open spectral data file: " + path);
  std::vector<std::vector<double>> rows;
  std::string line;
  bool headerSeen = false;
  while (std::getline(in, line)) {
    if (line.empty() || line[0] == '#') continue;
    if (!headerSeen) { headerSeen = true; continue; }
    std::vector<double> row;
    std::stringstream ss(line);
    std::string cell;
    try {
      while (std::getline(ss, cell, ',')) row.push_back(std::stod(cell));
    } catch (const std::exception&) {
      throw std::runtime_error("bad number in spectral data file: " + path + " line: " + line);
    }
    if (static_cast<int>(row.size()) != valueColumns + 1)
      throw std::runtime_error("wrong column count in spectral data file: " + path);
    rows.push_back(std::move(row));
  }
  if (static_cast<int>(rows.size()) != kSpectralSamples)
    throw std::runtime_error("spectral data file needs 81 rows (380-780 nm, 5 nm): " + path);
  for (int i = 0; i < kSpectralSamples; ++i)
    if (std::abs(rows[i][0] - wavelength(i)) > 1e-6)
      throw std::runtime_error("spectral data file off the 5 nm grid: " + path);
  return rows;
}

}  // namespace

Spectrum resample(const std::vector<std::array<double, 2>>& pts, double outside) {
  Spectrum out;
  for (int i = 0; i < kSpectralSamples; ++i) {
    const double nm = wavelength(i);
    out[i] = outside;
    for (size_t k = 0; k + 1 < pts.size(); ++k) {
      if (nm >= pts[k][0] && nm <= pts[k + 1][0]) {
        const double span = pts[k + 1][0] - pts[k][0];
        const double t = span > 0.0 ? (nm - pts[k][0]) / span : 0.0;
        out[i] = pts[k][1] + t * (pts[k + 1][1] - pts[k][1]);
        break;
      }
    }
  }
  return out;
}

SpectralTables SpectralTables::load(const std::string& dataDir) {
  SpectralTables t;
  const auto cmf = readGridCsv(dataDir + "/cie_1931_2deg_cmf.csv", 3);
  const auto day = readGridCsv(dataDir + "/cie_daylight_components.csv", 3);
  for (int i = 0; i < kSpectralSamples; ++i) {
    t.x_[i] = cmf[i][1]; t.y_[i] = cmf[i][2]; t.z_[i] = cmf[i][3];
    t.s0_[i] = day[i][1]; t.s1_[i] = day[i][2]; t.s2_[i] = day[i][3];
  }
  return t;
}

Spectrum SpectralTables::daylight(double T) const {
  // CIE 15: daylight chromaticity follows a smooth curve in colour temperature; the spectrum is
  // the mean daylight S0 plus two correction curves weighted to hit that chromaticity.
  const double x = T <= 7000.0
      ? 0.244063 + 0.09911e3 / T + 2.9678e6 / (T * T) - 4.6070e9 / (T * T * T)
      : 0.237040 + 0.24748e3 / T + 1.9018e6 / (T * T) - 2.0064e9 / (T * T * T);
  const double y = -3.0 * x * x + 2.870 * x - 0.275;
  const double d = 0.0241 + 0.2562 * x - 0.7341 * y;
  const double m1 = (-1.3515 - 1.7703 * x + 5.9114 * y) / d;
  const double m2 = (0.0300 - 31.4424 * x + 30.0717 * y) / d;
  Spectrum s;
  for (int i = 0; i < kSpectralSamples; ++i) s[i] = s0_[i] + m1 * s1_[i] + m2 * s2_[i];
  const double at560 = s[(560 - 380) / 5];
  return s * (100.0 / at560);
}

Spectrum blackbody(double kelvin) {
  const double c2 = 1.4388e-2;  // second radiation constant, m*K
  auto planck = [&](double nm) {
    const double m = nm * 1e-9;
    return 1.0 / (std::pow(m, 5.0) * (std::exp(c2 / (m * kelvin)) - 1.0));
  };
  Spectrum s;
  const double ref = planck(560.0);
  for (int i = 0; i < kSpectralSamples; ++i) s[i] = planck(wavelength(i)) / ref;
  return s;
}

Xyz integrate(const Spectrum& s, const SpectralTables& t) {
  double X = 0, Y = 0, Z = 0;
  for (int i = 0; i < kSpectralSamples; ++i) {
    X += s[i] * t.xBar()[i];
    Y += s[i] * t.yBar()[i];
    Z += s[i] * t.zBar()[i];
  }
  const double k = 1.0 / sum(t.yBar());
  return {X * k, Y * k, Z * k};
}

Xy chromaticity(const Xyz& c) {
  const double n = c.X + c.Y + c.Z;
  return {c.X / n, c.Y / n};
}

std::array<double, 2> uvPrime(const Xyz& c) {
  const double d = c.X + 15.0 * c.Y + 3.0 * c.Z;
  return {4.0 * c.X / d, 9.0 * c.Y / d};
}

double sum(const Spectrum& s) {
  double a = 0;
  for (double v : s) a += v;
  return a;
}

Spectrum operator*(const Spectrum& a, const Spectrum& b) {
  Spectrum r;
  for (int i = 0; i < kSpectralSamples; ++i) r[i] = a[i] * b[i];
  return r;
}

Spectrum operator*(const Spectrum& a, double k) {
  Spectrum r;
  for (int i = 0; i < kSpectralSamples; ++i) r[i] = a[i] * k;
  return r;
}

}  // namespace openfilm
