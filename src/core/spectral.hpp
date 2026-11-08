// Spectral foundation: one shared wavelength grid, the CIE observer and light-source models.
// Every spectrum in the film model (light, dye, sensitivity) lives on this grid so they can be
// multiplied together sample by sample.
#pragma once

#include <array>
#include <string>
#include <vector>

namespace openfilm {

constexpr int kSpectralSamples = 81;  // 380..780 nm
constexpr double kLambdaMin = 380.0;
constexpr double kLambdaStep = 5.0;

// Wavelength in nanometres of grid sample i.
inline double wavelength(int i) { return kLambdaMin + kLambdaStep * i; }

using Spectrum = std::array<double, kSpectralSamples>;
struct Xyz { double X, Y, Z; };
struct Xy { double x, y; };

// Puts scattered (nm, value) points onto the grid by straight-line interpolation. Grid samples
// outside the data's wavelength range get `outside` (e.g. 0 for a filter that stops at 700 nm).
Spectrum resample(const std::vector<std::array<double, 2>>& nmValue, double outside);

class SpectralTables {
 public:
  // Reads cie_1931_2deg_cmf.csv and cie_daylight_components.csv from dataDir.
  // Throws std::runtime_error naming the file if one is missing or malformed.
  static SpectralTables load(const std::string& dataDir);

  const Spectrum& xBar() const { return x_; }
  const Spectrum& yBar() const { return y_; }
  const Spectrum& zBar() const { return z_; }

  // CIE daylight (D-series) at a correlated colour temperature, scaled to 100 at 560 nm.
  Spectrum daylight(double cctKelvin) const;

 private:
  Spectrum x_{}, y_{}, z_{};
  Spectrum s0_{}, s1_{}, s2_{};
};

// Thermal (tungsten-like) light of a given temperature, scaled to 1 at 560 nm.
Spectrum blackbody(double kelvin);

// Colour of a spectrum as seen by the standard observer; an all-ones spectrum gives Y = 1.
Xyz integrate(const Spectrum& s, const SpectralTables& t);

Xy chromaticity(const Xyz& c);
std::array<double, 2> uvPrime(const Xyz& c);

double sum(const Spectrum& s);
Spectrum operator*(const Spectrum& a, const Spectrum& b);
Spectrum operator*(const Spectrum& a, double k);
inline Spectrum operator*(double k, const Spectrum& a) { return a * k; }

}  // namespace openfilm
