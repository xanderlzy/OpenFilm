#include "core/blur.hpp"

#include "core/pixel_ops.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <thread>

namespace openfilm {

namespace {

using detail::parallelRows;

std::vector<float> makeTaps(float sigma) {
  const int r = blurRadiusPx(sigma);
  std::vector<float> taps(size_t(2 * r + 1));
  double sum = 0;
  std::vector<double> t(taps.size());
  for (int i = -r; i <= r; ++i) {
    t[size_t(i + r)] = std::exp(-double(i) * i / (2.0 * double(sigma) * double(sigma)));
    sum += t[size_t(i + r)];
  }
  for (size_t i = 0; i < t.size(); ++i) taps[i] = float(t[i] / sum);
  return taps;
}

void resizeTo(Image4f& img, int w, int h) {
  img.width = w;
  img.height = h;
  img.px.assign(size_t(w) * size_t(h) * 4, 0.f);
}

// Sigma (in coarse px) that, after the f x f box down and the bilinear up, gives sigma sg at full resolution.
float reducedSigma(float sg, int f) {
  const float ff = float(f);
  return std::sqrt(std::max(sg * sg - (ff * ff - 1.f) / 12.f - ff * ff / 6.f, 0.0625f)) / ff;
}

inline int clampi(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }

// Separable blur of RGB; alpha copied.
void blurImpl(const Image4f& in, Image4f& out, const std::array<float, 3>& sigma) {
  const int w = in.width, h = in.height;
  resizeTo(out, w, h);
  std::vector<float> taps[3];
  int radius[3];
  for (int c = 0; c < 3; ++c) {
    radius[c] = blurRadiusPx(sigma[size_t(c)]);
    if (radius[c] > 0) taps[c] = makeTaps(sigma[size_t(c)]);
  }
  Image4f tmp;
  resizeTo(tmp, w, h);
  parallelRows(h, [&](int y0, int y1) {
    for (int y = y0; y < y1; ++y)
      for (int x = 0; x < w; ++x) {
        const float* src = in.at(x, y);
        float* dst = tmp.at(x, y);
        for (int c = 0; c < 3; ++c) {
          const int r = radius[c];
          if (r == 0) {
            dst[c] = src[c];
            continue;
          }
          float acc = 0.f;
          for (int i = -r; i <= r; ++i) acc += taps[c][size_t(i + r)] * in.at(clampi(x + i, 0, w - 1), y)[c];
          dst[c] = acc;
        }
        dst[3] = src[3];
      }
  });
  parallelRows(h, [&](int y0, int y1) {
    for (int y = y0; y < y1; ++y)
      for (int x = 0; x < w; ++x) {
        float* dst = out.at(x, y);
        for (int c = 0; c < 3; ++c) {
          const int r = radius[c];
          if (r == 0) {
            dst[c] = tmp.at(x, y)[c];
            continue;
          }
          float acc = 0.f;
          for (int i = -r; i <= r; ++i) acc += taps[c][size_t(i + r)] * tmp.at(x, clampi(y + i, 0, h - 1))[c];
          dst[c] = acc;
        }
        dst[3] = tmp.at(x, y)[3];
      }
  });
}

// f x f box average; partial edge blocks average only their real pixels.
void boxDown(const Image4f& in, Image4f& out, int f) {
  const int w = (in.width + f - 1) / f, h = (in.height + f - 1) / f;
  resizeTo(out, w, h);
  parallelRows(h, [&](int y0, int y1) {
    for (int y = y0; y < y1; ++y)
      for (int x = 0; x < w; ++x) {
        const int xe = std::min((x + 1) * f, in.width), ye = std::min((y + 1) * f, in.height);
        float s[4] = {0, 0, 0, 0};
        for (int yy = y * f; yy < ye; ++yy)
          for (int xx = x * f; xx < xe; ++xx)
            for (int c = 0; c < 4; ++c) s[c] += in.at(xx, yy)[c];
        const float inv = 1.f / float((xe - x * f) * (ye - y * f));
        for (int c = 0; c < 4; ++c) out.at(x, y)[c] = s[c] * inv;
      }
  });
}

// Bilinear upsample at fine-pixel centres ((x + 0.5)/f - 0.5), clamp-to-edge; adds weight * result to the
// channels of acc in `mask` (bit c = channel c).
void upAccumulate(const Image4f& small, Image4f& acc, int f, float weight, int mask) {
  const int sw = small.width, sh = small.height;
  parallelRows(acc.height, [&](int y0, int y1) {
    for (int y = y0; y < y1; ++y) {
      const float fy = (float(y) + 0.5f) / float(f) - 0.5f;
      const float flY = std::floor(fy);
      const float ty = fy - flY;
      const int ya = clampi(int(flY), 0, sh - 1), yb = clampi(int(flY) + 1, 0, sh - 1);
      for (int x = 0; x < acc.width; ++x) {
        const float fx = (float(x) + 0.5f) / float(f) - 0.5f;
        const float flX = std::floor(fx);
        const float tx = fx - flX;
        const int xa = clampi(int(flX), 0, sw - 1), xb = clampi(int(flX) + 1, 0, sw - 1);
        const float* p00 = small.at(xa, ya);
        const float* p10 = small.at(xb, ya);
        const float* p01 = small.at(xa, yb);
        const float* p11 = small.at(xb, yb);
        float* dst = acc.at(x, y);
        for (int c = 0; c < 3; ++c) {
          if (!(mask >> c & 1)) continue;
          const float top = p00[c] + (p10[c] - p00[c]) * tx;
          const float bot = p01[c] + (p11[c] - p01[c]) * tx;
          dst[c] += weight * (top + (bot - top) * ty);
        }
      }
    }
  });
}

}  // namespace

std::vector<float> blurTaps(float sigmaPx) { return makeTaps(sigmaPx); }

float reducedSigmaPx(float sigmaPx, int f) { return reducedSigma(sigmaPx, f); }

int blurRadiusPx(float sigmaPx) { return sigmaPx < 0.25f ? 0 : int(std::ceil(3.f * sigmaPx)); }

std::vector<LobeGroup> lobeGroups(const PxLobe& lobe) {
  std::vector<LobeGroup> groups;
  for (int c = 0; c < 3; ++c) {
    const int f = downsampleFactor(lobe.sigmaPx[size_t(c)]);
    auto it = std::find_if(groups.begin(), groups.end(), [&](const LobeGroup& g) { return g.factor == f; });
    if (it == groups.end()) groups.push_back({f, 1 << c});
    else it->mask |= 1 << c;
  }
  return groups;
}

int downsampleFactor(float sigmaPx) {
  if (sigmaPx <= 4.f) return 1;
  int f = 32;
  while (f > 1 && sigmaPx / float(f) < 2.f) f /= 2;
  return f;
}

int spreadReachPx(const std::vector<PxLobe>& lobes) {
  int reach = 0;
  for (const PxLobe& lobe : lobes) {
    for (size_t c = 0; c < 3; ++c) {
      // A coarse pixel depends on its f x f block; the bilinear up reads one coarse pixel either side.
      const int f = downsampleFactor(lobe.sigmaPx[c]);
      const int r = f == 1 ? blurRadiusPx(lobe.sigmaPx[c]) : (blurRadiusPx(reducedSigma(lobe.sigmaPx[c], f)) + 2) * f;
      reach = std::max(reach, r);
    }
  }
  return reach;
}

void gaussianBlur(const Image4f& in, Image4f& out, const std::array<float, 3>& sigmaPx) { blurImpl(in, out, sigmaPx); }

void spread(const Image4f& in, Image4f& out, const std::vector<PxLobe>& lobes) {
  resizeTo(out, in.width, in.height);
  parallelRows(in.height, [&](int y0, int y1) {
    for (int y = y0; y < y1; ++y)
      for (int x = 0; x < in.width; ++x) out.at(x, y)[3] = in.at(x, y)[3];
  });
  if (in.width <= 0 || in.height <= 0) return;
  for (const PxLobe& lobe : lobes) {
    // Each channel runs at its own downsampleFactor; channels sharing one are blurred together.
    for (const LobeGroup& g : lobeGroups(lobe)) {
      Image4f blurred;
      std::array<float, 3> s{0.f, 0.f, 0.f};  // channels outside the group: radius 0, not used
      if (g.factor == 1) {
        for (size_t c = 0; c < 3; ++c)
          if (g.mask >> c & 1) s[c] = lobe.sigmaPx[c];
        blurImpl(in, blurred, s);
        parallelRows(in.height, [&](int y0, int y1) {
          for (int y = y0; y < y1; ++y)
            for (int x = 0; x < in.width; ++x)
              for (int c = 0; c < 3; ++c)
                if (g.mask >> c & 1) out.at(x, y)[c] += lobe.weight * blurred.at(x, y)[c];
        });
      } else {
        Image4f small;
        boxDown(in, small, g.factor);
        for (size_t c = 0; c < 3; ++c)
          if (g.mask >> c & 1) s[c] = reducedSigma(lobe.sigmaPx[c], g.factor);
        blurImpl(small, blurred, s);
        upAccumulate(blurred, out, g.factor, lobe.weight, g.mask);
      }
    }
  }
}

}  // namespace openfilm
