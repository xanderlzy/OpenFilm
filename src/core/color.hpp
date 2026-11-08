// Colour-space maths: the working gamut (DaVinci Wide Gamut), the display gamut (Rec.709),
// chromatic adaptation and the transfer curves used at the plugin's input and output.
#pragma once

#include <array>

#include "core/spectral.hpp"

namespace openfilm {

using Vec3 = std::array<double, 3>;
using Mat3 = std::array<Vec3, 3>;

Vec3 mul(const Mat3& m, const Vec3& v);
Mat3 mul(const Mat3& a, const Mat3& b);
Mat3 inverse(const Mat3& m);

// A colour space's red, green and blue primaries and white point, as xy chromaticities.
struct Primaries { Xy r, g, b, white; };

inline constexpr Xy kD65{0.3127, 0.3290};
inline constexpr Primaries kDaVinciWideGamut{{0.8000, 0.3130}, {0.1682, 0.9877}, {0.0790, -0.1155}, kD65};
inline constexpr Primaries kRec2020{{0.708, 0.292}, {0.170, 0.797}, {0.131, 0.046}, kD65};
inline constexpr Primaries kRec709{{0.640, 0.330}, {0.300, 0.600}, {0.150, 0.060}, kD65};

// Linear RGB -> CIE XYZ for these primaries, scaled so RGB (1,1,1) is white with Y = 1.
Mat3 rgbToXyz(const Primaries& p);

// Bradford matrix that re-expresses colours seen under srcWhite as they look under dstWhite.
Mat3 bradford(const Xyz& srcWhite, const Xyz& dstWhite);

// DaVinci Intermediate: scene-linear <-> the log code values Resolve works in.
double diDecode(double code);
double diEncode(double linear);

// Display encoding for Rec.709 output (BT.1886, gamma 2.4). Negative light is clipped to black;
// values above 1 are kept so later stages decide how to handle over-range.
double bt1886Encode(double displayLinear);

}  // namespace openfilm
