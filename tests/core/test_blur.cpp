#include <doctest/doctest.h>

#include <cmath>

#include "core/blur.hpp"

using namespace openfilm;

namespace {

Image4f makeImage(int w, int h) {
  Image4f im;
  im.width = w;
  im.height = h;
  im.px.assign(size_t(w) * size_t(h) * 4, 0.f);
  return im;
}

double channelSum(const Image4f& im, int c) {
  double s = 0;
  for (size_t i = 0; i < im.px.size(); i += 4) s += im.px[i + size_t(c)];
  return s;
}

}  // namespace

TEST_CASE("light is conserved") {
  struct Case { int size; float sigma; };
  for (Case cs : {Case{512, 0.3f}, Case{512, 3.f}, Case{512, 20.f}, Case{1024, 90.f}}) {
    Image4f in = makeImage(cs.size, cs.size), out;
    float* p = in.at(cs.size / 2, cs.size / 2);
    p[0] = p[1] = p[2] = 1.f;
    p[3] = 1.f;
    spread(in, out, {{1.f, {cs.sigma, cs.sigma, cs.sigma}}});
    for (int c = 0; c < 3; ++c) CHECK(std::abs(channelSum(out, c) - 1.0) < 1e-4);
  }
}

TEST_CASE("matches a continuous Gaussian") {
  for (float sigma : {6.f, 60.f}) {
    const int w = 1024, h = 8, edge = 512;
    Image4f in = makeImage(w, h), out;
    for (int y = 0; y < h; ++y)
      for (int x = edge; x < w; ++x) in.at(x, y)[0] = in.at(x, y)[1] = in.at(x, y)[2] = 1.f;
    spread(in, out, {{1.f, {sigma, sigma, sigma}}});
    double worst = 0;
    for (int x = 0; x < w; ++x) {
      const double d = (x + 0.5) - edge;
      const double ref = 0.5 * std::erfc(-d / (double(sigma) * std::sqrt(2.0)));
      worst = std::max(worst, std::abs(double(out.at(x, h / 2)[1]) - ref));
    }
    CHECK(worst < 0.01);
  }
}

TEST_CASE("tiny sigma is identity") {
  Image4f in = makeImage(16, 9), out;
  for (size_t i = 0; i < in.px.size(); ++i) in.px[i] = float((i * 37) % 101) / 17.f;
  spread(in, out, {{1.f, {0.2f, 0.2f, 0.2f}}});
  CHECK(out.px == in.px);
}

TEST_CASE("per-channel sigmas") {
  const int n = 257;
  Image4f in = makeImage(n, n), out;
  float* p = in.at(n / 2, n / 2);
  p[0] = p[1] = p[2] = 1.f;
  spread(in, out, {{1.f, {1.f, 4.f, 16.f}}});
  const float sig[3] = {1.f, 4.f, 16.f};
  for (int c = 0; c < 3; ++c) {
    double m0 = 0, m2 = 0;
    for (int y = 0; y < n; ++y)
      for (int x = 0; x < n; ++x) {
        const double v = out.at(x, y)[c];
        m0 += v;
        m2 += v * (double(x - n / 2) * (x - n / 2));
      }
    CHECK(std::abs(m2 / m0 / (double(sig[c]) * sig[c]) - 1.0) < 0.03);
  }
}

// Second moment along x of channel c's response to a centred point, over sigma^2.
double secondMomentRatio(const Image4f& out, int c, float sigma) {
  const int n = out.width;
  double m0 = 0, m2 = 0;
  for (int y = 0; y < n; ++y)
    for (int x = 0; x < n; ++x) {
      const double v = out.at(x, y)[c];
      m0 += v;
      m2 += v * (double(x - n / 2) * (x - n / 2));
    }
  return m2 / m0 / (double(sigma) * sigma);
}

TEST_CASE("each channel runs at its own downsample factor") {
  SUBCASE("all three reduced, each at its own factor") {
    const PxLobe lobe{1.f, {10.f, 20.f, 40.f}};
    const auto g = lobeGroups(lobe);
    REQUIRE(g.size() == 3);
    CHECK((g[0].factor == 4 && g[0].mask == 1));
    CHECK((g[1].factor == 8 && g[1].mask == 2));
    CHECK((g[2].factor == 16 && g[2].mask == 4));
    const int n = 513;
    Image4f in = makeImage(n, n), out;
    float* p = in.at(n / 2, n / 2);
    p[0] = p[1] = p[2] = 1.f;
    spread(in, out, {lobe});
    for (int c = 0; c < 3; ++c) CHECK(std::abs(secondMomentRatio(out, c, lobe.sigmaPx[size_t(c)]) - 1.0) < 0.03);
  }
  SUBCASE("mixed: only the wide channel is reduced") {
    const PxLobe lobe{1.f, {1.f, 4.f, 40.f}};
    const auto g = lobeGroups(lobe);
    REQUIRE(g.size() == 2);
    CHECK((g[0].factor == 1 && g[0].mask == 3));
    CHECK((g[1].factor == 16 && g[1].mask == 4));
    const int n = 513;
    Image4f in = makeImage(n, n), out, full;
    float* p = in.at(n / 2, n / 2);
    p[0] = p[1] = p[2] = 1.f;
    spread(in, out, {lobe});
    for (int c = 0; c < 3; ++c) CHECK(std::abs(secondMomentRatio(out, c, lobe.sigmaPx[size_t(c)]) - 1.0) < 0.03);
    // The narrow channels are exactly the full-resolution blur.
    gaussianBlur(in, full, {1.f, 4.f, 0.f});
    bool same = true;
    for (int y = 0; y < n; ++y)
      for (int x = 0; x < n; ++x)
        for (int c = 0; c < 2; ++c) same = same && out.at(x, y)[c] == full.at(x, y)[c];
    CHECK(same);
  }
}

TEST_CASE("kernel larger than image") {
  for (auto dims : {std::pair<int, int>{64, 36}, std::pair<int, int>{1, 1}}) {
    Image4f in = makeImage(dims.first, dims.second), out;
    for (size_t i = 0; i < in.px.size(); ++i) in.px[i] = 0.1f + 0.01f * float(i % 7);
    spread(in, out, {{1.f, {500.f, 500.f, 500.f}}});
    for (float v : out.px) CHECK(std::isfinite(v));
    for (int c = 0; c < 3; ++c) CHECK(channelSum(out, c) <= channelSum(in, c) * (1.0 + 1e-3));

    Image4f flat = makeImage(dims.first, dims.second);
    for (size_t i = 0; i < flat.px.size(); ++i) flat.px[i] = (i % 4 == 3) ? 1.f : 0.37f;
    spread(flat, out, {{1.f, {500.f, 500.f, 500.f}}});
    for (size_t i = 0; i < out.px.size(); ++i) CHECK(std::abs(out.px[i] - flat.px[i]) < 1e-5);
  }
}

TEST_CASE("downsample factor") {
  CHECK(downsampleFactor(4.f) == 1);
  CHECK(downsampleFactor(4.1f) == 2);
  CHECK(downsampleFactor(16.f) == 8);
  CHECK(downsampleFactor(1000.f) == 32);
}
