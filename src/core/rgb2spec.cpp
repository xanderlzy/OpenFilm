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

  char magic[4];
  std::uint32_t res = 0;
  if (std::fread(magic, 4, 1, f.get()) != 1 || std::memcmp(magic, "SPEC", 4) != 0 ||
      std::fread(&res, sizeof res, 1, f.get()) != 1 || res < 2 || res > 1024)
    throw std::runtime_error("not an RGB-to-spectrum table: " + coeffPath);

  Rgb2Spec m;
  m.res_ = static_cast<int>(res);
  m.scale_.resize(res);
  m.data_.resize(std::size_t(3) * res * res * res * 3);
  if (std::fread(m.scale_.data(), sizeof(float), m.scale_.size(), f.get()) != m.scale_.size() ||
      std::fread(m.data_.data(), sizeof(float), m.data_.size(), f.get()) != m.data_.size())
    throw std::runtime_error("truncated RGB-to-spectrum table: " + coeffPath);
  // Anything left over means the file was written by a different tool version.
  char extra;
  if (std::fread(&extra, 1, 1, f.get()) != 0)
    throw std::runtime_error("RGB-to-spectrum table has unexpected size: " + coeffPath);
  return m;
}

std::array<double, 3> Rgb2Spec::coefficients(const Vec3& rgbIn) const {
  // The table is indexed by the largest channel (z) and the other two relative to it (x, y).
  Vec3 rgb;
  for (int j = 0; j < 3; ++j) rgb[j] = std::clamp(rgbIn[j], 0.0, 1.0);
  int i = 0;
  for (int j = 1; j < 3; ++j)
    if (rgb[j] >= rgb[i]) i = j;
  const double z = rgb[i];
  if (z <= 0.0) {
    return {double(data_[0]), double(data_[1]), double(data_[2])};
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
  const double m = std::max({rgb[0], rgb[1], rgb[2]});
  if (m <= 0.0) return out;
  // Fit rgb/(2m) so the curve sits mid-range (fully saturated colours have no 0..1 curve),
  // then multiply by 2m to restore the brightness.
  const double k = 2.0 * m;
  const auto c = coefficients({rgb[0] / k, rgb[1] / k, rgb[2] / k});
  for (int i = 0; i < kSpectralSamples; ++i) {
    const double l = wavelength(i);
    out[i] = k * sigmoid((c[0] * l + c[1]) * l + c[2]);
  }
  return out;
}

}  // namespace openfilm
