#include "core/rgb2spec.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>

namespace openfilm {

namespace {
struct FileCloser { void operator()(std::FILE* f) const { if (f) std::fclose(f); } };

double sigmoid(double x) { return 0.5 + x / (2.0 * std::sqrt(1.0 + x * x)); }
}  // namespace

Rgb2Spec Rgb2Spec::load(const std::string& coeffPath) {
  std::unique_ptr<std::FILE, FileCloser> f(std::fopen(coeffPath.c_str(), "rb"));
  if (!f) throw std::runtime_error("cannot open RGB-to-spectrum table: " + coeffPath);

  // Measure the file first: a corrupt header must not be able to trigger a huge allocation.
  if (std::fseek(f.get(), 0, SEEK_END) != 0) throw std::runtime_error("cannot read RGB-to-spectrum table: " + coeffPath);
  const long fileLen = std::ftell(f.get());
  std::rewind(f.get());

  char magic[4];
  std::uint32_t res = 0;
  if (fileLen < 8 || std::fread(magic, 4, 1, f.get()) != 1 || std::memcmp(magic, "SPEC", 4) != 0 ||
      std::fread(&res, sizeof res, 1, f.get()) != 1)
    throw std::runtime_error("not an RGB-to-spectrum table: " + coeffPath);
  if (res < 2 || res > 256)
    throw std::runtime_error("RGB-to-spectrum table has unsupported resolution: " + coeffPath);
  const std::uint64_t r = res;
  if (std::uint64_t(fileLen) != 8 + 4 * r + 4 * 9 * r * r * r)
    throw std::runtime_error("RGB-to-spectrum table has the wrong size: " + coeffPath);

  Rgb2Spec m;
  m.res_ = static_cast<int>(res);
  m.scale_.resize(res);
  m.data_.resize(std::size_t(3) * res * res * res * 3);
  if (std::fread(m.scale_.data(), sizeof(float), m.scale_.size(), f.get()) != m.scale_.size() ||
      std::fread(m.data_.data(), sizeof(float), m.data_.size(), f.get()) != m.data_.size())
    throw std::runtime_error("truncated RGB-to-spectrum table: " + coeffPath);
  // coefficients() binary-searches the brightness slices, so they must be strictly increasing.
  for (std::size_t i = 1; i < m.scale_.size(); ++i)
    if (!(m.scale_[i] > m.scale_[i - 1]))
      throw std::runtime_error("RGB-to-spectrum table has a bad brightness scale: " + coeffPath);
  return m;
}

std::array<double, 3> Rgb2Spec::coefficients(const Vec3& rgbIn) const {
  // The table is indexed by the largest channel (z) and the other two relative to it (x, y).
  Vec3 rgb;
  // Non-finite channels count as 0 so a bad pixel can never index outside the table.
  for (int j = 0; j < 3; ++j) rgb[j] = std::isfinite(rgbIn[j]) ? std::clamp(rgbIn[j], 0.0, 1.0) : 0.0;
  int i = 0;
  for (int j = 1; j < 3; ++j)
    if (rgb[j] >= rgb[i]) i = j;
  const double z = rgb[i];
  if (z <= 0.0) {
    return {double(data_[0]), double(data_[1]), double(data_[2])};
  }

  // A grey must give a perfectly flat curve; the interpolated table is only close to flat, which
  // would tint neutrals. For equal channels the exact answer is a constant reflectance z
  // (closed form from Mitsuba's rgb2spec).
  // "Equal" allows for rounding noise from upstream matrix maths (a DWG grey is not bit-exact).
  const double lo = std::min({rgb[0], rgb[1], rgb[2]});
  if (z - lo <= 1e-9 * z) {
    const double g = std::clamp(z, 1e-9, 1.0 - 1e-9);
    return {0.0, 0.0, (g - 0.5) / std::sqrt(g * (1.0 - g))};
  }

  const int res = res_;
  const double s = (res - 1) / z;
  const double x = rgb[(i + 1) % 3] * s, y = rgb[(i + 2) % 3] * s;
  const int xi = std::min(int(x), res - 2), yi = std::min(int(y), res - 2);
  // The brightness slices are spaced by a double smoothstep, so find the interval by search.
  const int zi = int(std::clamp<std::ptrdiff_t>(
      std::upper_bound(scale_.begin(), scale_.end(), float(z)) - scale_.begin() - 1, 0, res - 2));

  const double x1 = x - xi, x0 = 1.0 - x1, y1 = y - yi, y0 = 1.0 - y1;
  const double z1 = (z - scale_[zi]) / (double(scale_[zi + 1]) - scale_[zi]), z0 = 1.0 - z1;

  const std::size_t offset = ((std::size_t(i) * res + zi) * res + yi) * res + xi;
  const std::size_t dx = 1, dy = res, dz = std::size_t(res) * res;
  std::array<double, 3> out;
  for (int k = 0; k < 3; ++k) {
    auto at = [&](std::size_t d) { return double(data_[(offset + d) * 3 + k]); };
    out[k] = ((at(0) * x0 + at(dx) * x1) * y0 + (at(dy) * x0 + at(dy + dx) * x1) * y1) * z0 +
             ((at(dz) * x0 + at(dz + dx) * x1) * y0 + (at(dz + dy) * x0 + at(dz + dy + dx) * x1) * y1) * z1;
  }
  return out;
}

Spectrum Rgb2Spec::toSpectrum(const Vec3& rgb) const {
  Spectrum out{};
  // A NaN/Inf pixel (e.g. from a bad upstream node) is treated as 0 rather than poisoning the spectrum.
  Vec3 v = rgb;
  for (double& c : v)
    if (!std::isfinite(c)) c = 0.0;
  const double m = std::max({v[0], v[1], v[2]});
  if (m <= 0.0) return out;
  // Fit rgb/k on a mid-brightness slice of the table, then multiply by k to restore brightness.
  // Ordinary colours use brightness 0.5 (k = 2m) so the curve sits mid-range. Very saturated
  // colours (near the Rec.2020 edge) have no valid 0..1 curve at that brightness: the table cells
  // there hold arbitrary, mutually unrelated fits, which shows up as jagged film exposure.
  // On a darker slice every hue is reachable, so from saturation 0.85 the brightness eases
  // (smoothstep) down to 0.2 at the gamut edge. Colours below 0.85 are unchanged.
  const double mn = std::max(std::min({v[0], v[1], v[2]}), 0.0);
  const double sat = std::clamp(1.0 - mn / m, 0.0, 1.0);
  double t = std::clamp((sat - 0.85) / (1.0 - 0.85), 0.0, 1.0);
  t = t * t * (3.0 - 2.0 * t);
  const double brightness = 0.5 + (0.2 - 0.5) * t;
  const double k = m / brightness;
  const auto c = coefficients({v[0] / k, v[1] / k, v[2] / k});
  for (int i = 0; i < kSpectralSamples; ++i) {
    const double l = wavelength(i);
    out[i] = k * sigmoid((c[0] * l + c[1]) * l + c[2]);
  }
  return out;
}

}  // namespace openfilm
