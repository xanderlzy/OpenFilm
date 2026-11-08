#include "core/color.hpp"

#include <algorithm>
#include <cmath>

namespace openfilm {

Vec3 mul(const Mat3& m, const Vec3& v) {
  Vec3 o{};
  for (int i = 0; i < 3; ++i) o[i] = m[i][0] * v[0] + m[i][1] * v[1] + m[i][2] * v[2];
  return o;
}

Mat3 mul(const Mat3& a, const Mat3& b) {
  Mat3 o{};
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) o[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
  return o;
}

Mat3 inverse(const Mat3& m) {
  // Adjugate / determinant; the matrices here are always well-conditioned 3x3s.
  const double c00 = m[1][1] * m[2][2] - m[1][2] * m[2][1];
  const double c01 = m[1][2] * m[2][0] - m[1][0] * m[2][2];
  const double c02 = m[1][0] * m[2][1] - m[1][1] * m[2][0];
  const double det = m[0][0] * c00 + m[0][1] * c01 + m[0][2] * c02;
  const double d = 1.0 / det;
  return {{{c00 * d, (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * d, (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * d},
           {c01 * d, (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * d, (m[0][2] * m[1][0] - m[0][0] * m[1][2]) * d},
           {c02 * d, (m[0][1] * m[2][0] - m[0][0] * m[2][1]) * d, (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * d}}};
}

namespace {
Vec3 xyzOf(const Xy& c) { return {c.x / c.y, 1.0, (1.0 - c.x - c.y) / c.y}; }
}  // namespace

Mat3 rgbToXyz(const Primaries& p) {
  const Vec3 r = xyzOf(p.r), g = xyzOf(p.g), b = xyzOf(p.b);
  const Mat3 cols{{{r[0], g[0], b[0]}, {r[1], g[1], b[1]}, {r[2], g[2], b[2]}}};
  // Solve for the channel gains that make RGB (1,1,1) land exactly on the white point.
  const Vec3 s = mul(inverse(cols), xyzOf(p.white));
  Mat3 m = cols;
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) m[i][j] *= s[j];
  return m;
}

Mat3 bradford(const Xyz& srcWhite, const Xyz& dstWhite) {
  // Bradford cone-response space: scale each cone channel by dst/src white, then return to XYZ.
  static const Mat3 kBfd{{{0.8951, 0.2664, -0.1614}, {-0.7502, 1.7135, 0.0367}, {0.0389, -0.0685, 1.0296}}};
  const Vec3 s = mul(kBfd, Vec3{srcWhite.X, srcWhite.Y, srcWhite.Z});
  const Vec3 d = mul(kBfd, Vec3{dstWhite.X, dstWhite.Y, dstWhite.Z});
  const Mat3 scale{{{d[0] / s[0], 0, 0}, {0, d[1] / s[1], 0}, {0, 0, d[2] / s[2]}}};
  return mul(inverse(kBfd), mul(scale, kBfd));
}

namespace {
// DaVinci Intermediate constants (Blackmagic's published definition).
constexpr double kA = 0.0075, kB = 7.0, kC = 0.07329248, kM = 10.44426855;
constexpr double kLinCut = 0.00262409, kLogCut = 0.02740668;
}  // namespace

double diEncode(double lin) { return lin <= kLinCut ? lin * kM : (std::log2(lin + kA) + kB) * kC; }

double diDecode(double code) { return code <= kLogCut ? code / kM : std::exp2(code / kC - kB) - kA; }

double bt1886Encode(double l) { return std::pow(std::max(l, 0.0), 1.0 / 2.4); }

}  // namespace openfilm
