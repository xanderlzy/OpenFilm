// RGB -> spectrum upsampling (Jakob & Hanika 2019). Each Rec.2020 colour becomes a smooth
// reflectance curve whose colour under D65 is that same RGB, so the film model can work with
// real spectra even though the input image only carries three numbers per pixel.
#pragma once

#include <array>
#include <string>
#include <vector>

#include "core/color.hpp"
#include "core/spectral.hpp"

namespace openfilm {

class Rgb2Spec {
 public:
  // Reads a coefficient table made by tools/rgb2spec (Mitsuba layout). Throws std::runtime_error
  // if the file is missing, is not a table, or has the wrong size.
  static Rgb2Spec load(const std::string& coeffPath);

  // Curve coefficients (c0,c1,c2) for an RGB in [0,1]^3: reflectance(l nm) = sigmoid(c0 l^2 + c1 l + c2).
  std::array<double, 3> coefficients(const Vec3& rgb01) const;

  // Spectrum for a scene-linear Rec.2020 colour. Brightness above 1 is carried by scaling a
  // mid-range reflectance, since a reflectance itself cannot exceed 1.
  Spectrum toSpectrum(const Vec3& rec2020Linear) const;

 private:
  int res_ = 0;
  std::vector<float> scale_;  // non-uniform brightness slices of the table
  std::vector<float> data_;   // [3][res][res][res][3]
};

}  // namespace openfilm
