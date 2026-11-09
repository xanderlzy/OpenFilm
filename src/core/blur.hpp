// Gaussian spread: the blur engine behind sharpness, edge effect, halation and bloom.
// The Metal port must follow the algorithm pinned below literally.
#pragma once

#include <array>
#include <vector>

namespace openfilm {

// RGBA interleaved float image.
struct Image4f {
  int width = 0, height = 0;
  std::vector<float> px;
  float* at(int x, int y) { return px.data() + (size_t(y) * size_t(width) + size_t(x)) * 4; }
  const float* at(int x, int y) const { return px.data() + (size_t(y) * size_t(width) + size_t(x)) * 4; }
};

// One Gaussian lobe, per-channel sigma in pixels.
struct PxLobe {
  float weight;
  std::array<float, 3> sigmaPx;
};

// out[c] = sum over lobes of weight * (G_{sigmaPx[c]} * in)[c] for c < 3; alpha copied. Clamp-to-edge at borders.
// Lobes with sigma > 4 px are computed at reduced resolution (factor from the lobe's smallest sigma). `out` must not alias `in`.
void spread(const Image4f& in, Image4f& out, const std::vector<PxLobe>& lobes);

// One lobe's blur of RGB at full resolution (alpha copied). Taps exp(-i^2/(2 s^2)), |i| <= blurRadiusPx(s), normalised,
// separable (horizontal then vertical), clamp-to-edge. A channel with sigma < 0.25 px is copied. `out` must not alias `in`.
void gaussianBlur(const Image4f& in, Image4f& out, const std::array<float, 3>& sigmaPx);

// How far (full-res px) spread(in, out, lobes) reads from an output pixel: blurRadiusPx for full-res lobes,
// (coarse radius + 2) x f for reduced-resolution ones. Pixels farther than this from the image edge
// are unaffected by the clamp-to-edge border.
int spreadReachPx(const std::vector<PxLobe>& lobes);

int blurRadiusPx(float sigmaPx);      // ceil(3 sigma); 0 when sigma < 0.25 (identity)
// The normalised taps of one channel's blur, 2 * blurRadiusPx(sigmaPx) + 1 of them (computed in
// double, stored as float). The GPU uploads exactly these.
std::vector<float> blurTaps(float sigmaPx);
// Sigma (coarse px) that, after the f x f box down and the bilinear up, gives sigmaPx at full resolution.
float reducedSigmaPx(float sigmaPx, int f);
int downsampleFactor(float sigmaPx);  // 1 if sigma <= 4, else largest power of two f <= 32 with sigma / f >= 2

}  // namespace openfilm
