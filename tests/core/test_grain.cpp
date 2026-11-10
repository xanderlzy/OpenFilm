// Tests for Full grain on the CPU (ruling R3, additive dye clouds): grain off is M4, the measured
// 48 um rms against the datasheet, mean density, per-frame seeds, format and Size scaling of the
// texture, sub-windows and non-finite densities.
#include <doctest/doctest.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <thread>

#include "core/color.hpp"
#include "core/film_renderer.hpp"
#include "core/film_spatial.hpp"
#include "core/grain_plates.hpp"
#include "core/grain_sim.hpp"
#include "metal/grain_common.h"

using namespace openfilm;

namespace {
const std::string kData = OPENFILM_SOURCE_DATA_DIR;
constexpr double kUhdS35 = 3840 / 24.89;  // UHD Super 35 px/mm
constexpr double kApertureMm = 0.048;     // the datasheet's rms aperture diameter

std::shared_ptr<const FilmResources> res() {
  static const auto r = FilmResources::load(kData, OPENFILM_COEFF_PATH);
  return r;
}
const BakedFilm& film() {
  static const auto f = [] {
    FilmParams p;
    p.negativeId = "kodak-5219-type";
    p.printId = "kodak-2383-type";
    return bakeFilm(FilmModel(res(), p));
  }();
  return *f;
}

struct TestImage {
  ImageGeometry geom;
  std::vector<unsigned char> bytes;
  TestImage(ImageGeometry g, float fill) : geom(g), bytes(size_t(g.rowBytes) * size_t(g.y2 - g.y1)) {
    float* f = reinterpret_cast<float*>(bytes.data());
    for (size_t i = 0; i < bytes.size() / sizeof(float); ++i) f[i] = fill;
  }
  ImageView view() { return {reinterpret_cast<float*>(bytes.data()), geom}; }
  ConstImageView cview() const { return {reinterpret_cast<const float*>(bytes.data()), geom}; }
  float* at(int x, int y) { return pixelAddress(view(), x, y); }
};
ImageGeometry tight(int x1, int y1, int x2, int y2) {
  return {x1, y1, x2, y2, std::ptrdiff_t(x2 - x1) * 4 * std::ptrdiff_t(sizeof(float))};
}

void fillRandom(TestImage& img, unsigned seed, bool specials) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> u(-0.1f, 1.1f);
  const float nan = std::numeric_limits<float>::quiet_NaN();
  const float inf = std::numeric_limits<float>::infinity();
  for (int y = img.geom.y1; y < img.geom.y2; ++y)
    for (int x = img.geom.x1; x < img.geom.x2; ++x) {
      float* p = img.at(x, y);
      for (int c = 0; c < 3; ++c) p[c] = u(rng);
      p[3] = float(rng() % 1000) / 999.f;
      if (specials && rng() % 23 == 0) p[rng() % 3] = (rng() % 2) ? nan : -inf;
    }
}

SpatialControls effects(bool on) {
  SpatialControls s;
  s.sharpness = s.halation = s.bloom = on;
  s.remjet = false;
  s.pxPerMm = kUhdS35;
  return s;
}
SpatialControls grainOnly(double pxPerMm = kUhdS35, uint32_t frameSeed = 1) {
  SpatialControls s = effects(false);
  s.grain = true;
  s.grainQuality = GrainQuality::Full;  // ruling R2
  s.pxPerMm = pxPerMm;
  s.frameSeed = frameSeed;
  return s;
}

Image4f uniform(int w, int h, std::array<float, 3> d) {
  Image4f img;
  img.width = w;
  img.height = h;
  img.px.resize(size_t(w) * size_t(h) * 4);
  for (size_t i = 0; i < img.px.size(); i += 4) {
    for (size_t c = 0; c < 3; ++c) img.px[i + c] = d[c];
    img.px[i + 3] = 1.f;
  }
  return img;
}

const std::array<GrainLayer, 3>& layersOf(GrainStage st) {
  return st == GrainStage::Negative ? film().grain.negative : film().grain.print;
}

// grainStage on a uniform patch; returns the deviation field of channel c (output minus input).
std::vector<double> deviation(const Image4f& in, const SpatialControls& s, GrainStage st, int c) {
  Image4f out = in;
  grainStage(out, {0, 0, in.width, in.height}, film(), s, st);
  std::vector<double> v(size_t(in.width) * size_t(in.height));
  for (int y = 0; y < in.height; ++y)
    for (int x = 0; x < in.width; ++x)
      v[size_t(y) * size_t(in.width) + size_t(x)] = double(out.at(x, y)[c]) - double(in.at(x, y)[c]);
  return v;
}

// Std of the means over discs of diameter `diamPx` (pixels whose centres are inside), the discs on a
// grid one diameter (+1 px) apart so their means are independent.
double discMeanStd(const std::vector<double>& v, int w, int h, double diamPx) {
  const double r = diamPx / 2;
  const int reach = int(std::ceil(r)), step = int(std::ceil(diamPx)) + 1;
  std::vector<double> means;
  for (int cy = reach; cy + reach < h; cy += step)
    for (int cx = reach; cx + reach < w; cx += step) {
      double sum = 0;
      int n = 0;
      for (int y = cy - reach; y <= cy + reach; ++y)
        for (int x = cx - reach; x <= cx + reach; ++x)
          if (double(x - cx) * (x - cx) + double(y - cy) * (y - cy) <= r * r) {
            sum += v[size_t(y) * size_t(w) + size_t(x)];
            ++n;
          }
      means.push_back(sum / n);
    }
  double s = 0, s2 = 0;
  for (double m : means) { s += m; s2 += m * m; }
  const double mean = s / double(means.size());
  return std::sqrt(std::max(0.0, s2 / double(means.size()) - mean * mean));
}

// Lag (px) where a correlation curve corr(lag), lag = 0, 1, ..., falls to half of corr(0), linearly
// interpolated between integer lags.
template <class Corr>
double halfMaxLag(Corr corr, int maxLag) {
  const double c0 = corr(0);
  double prev = 1;
  for (int lag = 1; lag < maxLag; ++lag) {
    const double c = corr(lag) / c0;
    if (c <= 0.5) return lag - 1 + (prev - 0.5) / (prev - c);
    prev = c;
  }
  return std::numeric_limits<double>::infinity();
}

// Half-max radius of the field's measured autocorrelation (x and y lags averaged).
double halfMaxRadius(const std::vector<double>& v, int w, int h) {
  double mean = 0;
  for (double x : v) mean += x;
  mean /= double(v.size());
  return halfMaxLag(
      [&](int lag) {
        double s = 0;
        long long n = 0;
        for (int y = 0; y + lag < h; ++y)
          for (int x = 0; x + lag < w; ++x) {
            const double a = v[size_t(y) * size_t(w) + size_t(x)] - mean;
            s += a * (v[size_t(y) * size_t(w) + size_t(x + lag)] - mean) + a * (v[size_t(y + lag) * size_t(w) + size_t(x)] - mean);
            n += 2;
          }
        return s / double(n);
      },
      std::min(w, h) / 2);
}

// The same half-max radius predicted exactly by the model: a pixel value is the mean of u over its
// 4 x 4 sub-samples and u's covariance between two points is unitCovariance(distance), so two pixels
// `lag` apart have covariance (1/256) sum over sub-sample pairs. The pixel footprint is fixed while
// the clouds scale, so this is not exactly proportional to the cloud radius at a few px.
double predictedHalfMaxRadius(const GrainLayer& g, double pxPerMm, double sizeMul) {
  return halfMaxLag(
      [&](int lag) {
        double s = 0;
        for (int di = -3; di <= 3; ++di)
          for (int dj = -3; dj <= 3; ++dj)
            s += (4 - std::abs(di)) * (4 - std::abs(dj)) *
                 g.unitCovariance(std::hypot(lag + di / 4.0, dj / 4.0) / pxPerMm, sizeMul);
        return s / 256;
      },
      100000);
}

double correlation(const std::vector<double>& a, const std::vector<double>& b) {
  double ma = 0, mb = 0;
  for (size_t i = 0; i < a.size(); ++i) { ma += a[i]; mb += b[i]; }
  ma /= double(a.size());
  mb /= double(b.size());
  double sab = 0, saa = 0, sbb = 0;
  for (size_t i = 0; i < a.size(); ++i) {
    sab += (a[i] - ma) * (b[i] - mb);
    saa += (a[i] - ma) * (a[i] - ma);
    sbb += (b[i] - mb) * (b[i] - mb);
  }
  return sab / std::sqrt(saa * sbb);
}
}  // namespace

TEST_CASE("grain off or amount 0 is M4") {
  REQUIRE(film().grain.valid);
  const int w = 97, h = 61;
  TestImage src(tight(0, 0, w, h), 0.f);
  fillRandom(src, 7, true);
  const RenderWindow win{5, 7, 80, 50};
  LookControls look;
  look.exposureStops = 0.4;
  for (bool on : {true, false}) {
    const SpatialControls m4 = effects(on);
    TestImage ref(tight(0, 0, w, h), 7.f);
    renderFilm(src.cview(), ref.view(), win, film(), look, m4);
    SpatialControls off = grainOnly();
    off.sharpness = off.halation = off.bloom = on;
    off.grain = false;
    SpatialControls zero = off;
    zero.grain = true;
    zero.grainAmount = 0;
    for (const SpatialControls& s : {off, zero}) {
      TestImage out(tight(0, 0, w, h), 7.f);
      renderFilm(src.cview(), out.view(), win, film(), look, s);
      CHECK(std::memcmp(out.bytes.data(), ref.bytes.data(), out.bytes.size()) == 0);
    }
  }
  // And grain on really changes the render.
  TestImage ref(tight(0, 0, w, h), 7.f), grainy(tight(0, 0, w, h), 7.f);
  renderFilm(src.cview(), ref.view(), win, film(), look, effects(false));
  renderFilm(src.cview(), grainy.view(), win, film(), look, grainOnly());
  CHECK(std::memcmp(grainy.bytes.data(), ref.bytes.data(), ref.bytes.size()) != 0);
}

TEST_CASE("measured grain matches the datasheet") {
  const int n = 512;
  const double diamPx = kApertureMm * kUhdS35;
  for (GrainStage st : {GrainStage::Negative, GrainStage::Print}) {
    const auto& layers = layersOf(st);
    size_t points = 0;
    for (const auto& g : layers) points = std::max(points, g.density.size());
    std::array<double, 3> worst{}, worstD{};
    for (size_t k = 0; k < points; ++k) {
      // Each channel at its own layer's k-th digitised density (the last one again once a layer runs out).
      std::array<float, 3> d{};
      for (size_t c = 0; c < 3; ++c) d[c] = float(layers[c].density[std::min(k, layers[c].density.size() - 1)]);
      const Image4f in = uniform(n, n, d);
      Image4f out = in;
      grainStage(out, {0, 0, n, n}, film(), grainOnly(kUhdS35, uint32_t(k + 1)), st);
      for (int c = 0; c < 3; ++c) {
        if (k >= layers[size_t(c)].density.size()) continue;
        std::vector<double> v(size_t(n) * size_t(n));
        for (int y = 0; y < n; ++y)
          for (int x = 0; x < n; ++x) v[size_t(y) * n + size_t(x)] = double(out.at(x, y)[c]) - double(d[size_t(c)]);
        const double measured = discMeanStd(v, n, n, diamPx), want = layers[size_t(c)].sigma[k];
        const double e = std::abs(measured / want - 1);
        if (!(e <= worst[size_t(c)])) { worst[size_t(c)] = std::isfinite(e) ? e : 1e9; worstD[size_t(c)] = d[size_t(c)]; }
      }
    }
    const char* names[3] = {"red", "green", "blue"};
    for (int c = 0; c < 3; ++c) {
      const std::string label = std::string(st == GrainStage::Negative ? "negative " : "print ") + names[c];
      MESSAGE(label << ": worst |measured/datasheet - 1| " << worst[size_t(c)] << " at D " << worstD[size_t(c)]);
      CHECK(worst[size_t(c)] <= 0.15);
    }
  }
}

TEST_CASE("grain keeps mean density") {
  const int n = 1024;
  for (GrainStage st : {GrainStage::Negative, GrainStage::Print}) {
    const auto& layers = layersOf(st);
    for (double t : {0.25, 0.75}) {
      std::array<float, 3> d{};
      for (size_t c = 0; c < 3; ++c)
        d[c] = float(layers[c].density.front() + t * (layers[c].density.back() - layers[c].density.front()));
      const Image4f in = uniform(n, n, d);
      Image4f out = in;
      grainStage(out, {0, 0, n, n}, film(), grainOnly(kUhdS35, 5), st);
      for (int c = 0; c < 3; ++c) {
        double sum = 0;
        for (int y = 0; y < n; ++y)
          for (int x = 0; x < n; ++x) sum += double(out.at(x, y)[c]) - double(d[size_t(c)]);
        const double shift = sum / (double(n) * n);
        INFO("stage " << int(st) << " layer " << c << " D " << d[size_t(c)] << " mean shift " << shift);
        CHECK(std::abs(shift) <= 0.005);
      }
    }
  }
}

TEST_CASE("grain changes every frame and repeats per frame") {
  const int n = 256;
  const Image4f in = uniform(n, n, {1.f, 1.2f, 1.4f});
  const double t = 3600.5;
  const uint32_t a = frameSeedFromTime(t), b = frameSeedFromTime(t + 1.0 / 24);
  CHECK(a != b);
  CHECK(frameSeedFromTime(t) == a);
  for (GrainStage st : {GrainStage::Negative, GrainStage::Print})
    for (int c = 0; c < 3; ++c) {
      const auto va = deviation(in, grainOnly(kUhdS35, a), st, c);
      CHECK(va == deviation(in, grainOnly(kUhdS35, a), st, c));
      const double r = correlation(va, deviation(in, grainOnly(kUhdS35, b), st, c));
      INFO("stage " << int(st) << " layer " << c << " frame-to-frame correlation " << r);
      CHECK(std::abs(r) < 0.05);
    }
  // The three layers and the two stages draw different clouds.
  CHECK(grainSeed(a, GrainStage::Negative, 0) != grainSeed(a, GrainStage::Negative, 1));
  CHECK(grainSeed(a, GrainStage::Negative, 0) != grainSeed(a, GrainStage::Print, 0));
}

// The brief's bar (half-max radius ratio 2 +-5 % between 600 and 1200 px/mm) cannot hold as written:
// the measured autocorrelation includes the pixel footprint, which does not scale with the format
// (about +0.25 px on the radius at any scale; ratio ~1.88 between r = 3 and 6 px, exactly as the
// model predicts). So: at 600 and 1200 px/mm the measured radius matches the model's exact pixel-
// sampled prediction within 3 %, and the 2x ratio is checked at 1200 / 2400 px/mm (r = 6 / 12 px),
// where the footprint is small. The patch grows with the clouds (512 px at r = 3 px) so every
// estimate sees as many clouds.
TEST_CASE("grain format scaling") {
  for (int c = 0; c < 3; ++c) {
    const GrainLayer& g = film().grain.negative[size_t(c)];
    double r[3];
    const double px[3] = {600, 1200, 2400};
    for (int k = 0; k < 3; ++k) {
      const int n = 512 << k;
      r[k] = halfMaxRadius(deviation(uniform(n, n, {1.f, 1.2f, 1.4f}), grainOnly(px[k]), GrainStage::Negative, c), n, n);
      const double want = predictedHalfMaxRadius(g, px[k], 1);
      MESSAGE("negative layer " << c << " at " << px[k] << " px/mm: autocorrelation half-max radius " << r[k]
                                << " px, model " << want << " px");
      if (k < 2) CHECK(std::abs(r[k] / want - 1) <= 0.03);
    }
    MESSAGE("negative layer " << c << ": ratio 1200/600 " << r[1] / r[0] << ", 2400/1200 " << r[2] / r[1]);
    CHECK(std::abs(r[2] / r[1] / 2 - 1) <= 0.05);
  }
}

// Radius at 1200 px/mm (clouds 6 -> 12 px, patch 1024 -> 2048 px; see the note on format scaling),
// the 48 um rms at UHD Super 35, the production scale.
TEST_CASE("grain size") {
  for (int c = 0; c < 3; ++c) {
    const GrainLayer& g = film().grain.negative[size_t(c)];
    SpatialControls s1 = grainOnly(1200, 3), s2 = s1;
    s2.grainSize = 2;
    const double r1 = halfMaxRadius(deviation(uniform(1024, 1024, {1.f, 1.2f, 1.4f}), s1, GrainStage::Negative, c), 1024, 1024);
    const double r2 = halfMaxRadius(deviation(uniform(2048, 2048, {1.f, 1.2f, 1.4f}), s2, GrainStage::Negative, c), 2048, 2048);
    const double want2 = predictedHalfMaxRadius(g, 1200, 2);
    s1.pxPerMm = s2.pxPerMm = kUhdS35;
    const int n = 1024;
    const double diamPx = kApertureMm * kUhdS35;
    const Image4f big = uniform(n, n, {1.f, 1.2f, 1.4f});
    const double rms1 = discMeanStd(deviation(big, s1, GrainStage::Negative, c), n, n, diamPx);
    const double rms2 = discMeanStd(deviation(big, s2, GrainStage::Negative, c), n, n, diamPx);
    MESSAGE("negative layer " << c << ": half-max radius " << r1 << " -> " << r2 << " px (ratio " << r2 / r1
                              << ", model at Size 2 " << want2 << "), 48 um rms " << rms1 << " -> " << rms2
                              << " (ratio " << rms2 / rms1 << ")");
    CHECK(std::abs(r2 / r1 / 2 - 1) <= 0.05);
    CHECK(std::abs(r2 / want2 - 1) <= 0.03);
    CHECK(std::abs(rms2 / rms1 - 1) <= 0.10);
  }
  // Print at Size 0.5: 2 um clouds become 0.15 px, the sub-pixel Gaussian path; the rms must hold too.
  const int n = 1024;
  const double diamPx = kApertureMm * kUhdS35;
  const Image4f big = uniform(n, n, {1.f, 1.2f, 1.4f});
  for (int c = 0; c < 3; ++c) {
    SpatialControls s1 = grainOnly(kUhdS35, 3), s05 = s1;
    s05.grainSize = 0.5;
    REQUIRE(grainUnitParams(film().grain.print[size_t(c)], kUhdS35, 0.5).rPx < 0.25f);
    const double rms1 = discMeanStd(deviation(big, s1, GrainStage::Print, c), n, n, diamPx);
    const double rms05 = discMeanStd(deviation(big, s05, GrainStage::Print, c), n, n, diamPx);
    MESSAGE("print layer " << c << ": 48 um rms Size 1 " << rms1 << ", Size 0.5 (Gaussian) " << rms05 << " (ratio "
                           << rms05 / rms1 << ")");
    CHECK(std::abs(rms05 / rms1 - 1) <= 0.10);
  }
}

TEST_CASE("grain sub-window") {
  const int w = 300, h = 200;
  TestImage src(tight(0, 0, w, h), 0.f);
  fillRandom(src, 5, false);
  const RenderWindow sub{37, 53, 37 + 150, 53 + 90};
  for (bool fx : {false, true}) {
    SpatialControls s = grainOnly(kUhdS35, 11);
    s.sharpness = s.halation = s.bloom = fx;
    TestImage a(tight(0, 0, w, h), 7.f), b(tight(0, 0, w, h), 7.f);
    renderFilm(src.cview(), a.view(), {0, 0, w, h}, film(), LookControls{}, s);
    renderFilm(src.cview(), b.view(), sub, film(), LookControls{}, s);
    double worst = 0;
    for (int y = sub.y1; y < sub.y2; ++y)
      for (int x = sub.x1; x < sub.x2; ++x)
        for (int c = 0; c < 4; ++c) worst = std::max(worst, double(std::abs(a.at(x, y)[c] - b.at(x, y)[c])));
    INFO("effects " << fx << ": max |diff| " << worst);
    if (fx) CHECK(worst <= 1e-6);  // same bar as the M4 sub-window test
    else CHECK(worst == 0);
    CHECK(b.at(sub.x1 - 1, sub.y1)[0] == 7.f);
  }
}

TEST_CASE("grain extreme densities") {
  const float nan = std::numeric_limits<float>::quiet_NaN(), inf = std::numeric_limits<float>::infinity();
  for (GrainStage st : {GrainStage::Negative, GrainStage::Print}) {
    const auto& layers = layersOf(st);
    for (int c = 0; c < 3; ++c) {
      const auto& g = layers[size_t(c)];
      const float values[] = {-1.f, float(g.density.front() - 0.5), float(g.density.back() + 1.0), 10.f, nan, inf, -inf};
      for (float v : values) {
        Image4f img = uniform(64, 64, {1.f, 1.f, 1.f});
        for (size_t i = size_t(c); i < img.px.size(); i += 4) img.px[i] = v;
        SpatialControls s = grainOnly(kUhdS35, 9);
        s.grainAmount = 3;
        for (double size : {0.5, 2.0}) {  // at 0.5 the print clouds are sub-pixel: the Gaussian path
          s.grainSize = size;
          Image4f out = img;
          grainStage(out, {0, 0, 64, 64}, film(), s, st);
          bool ok = true;
          for (size_t i = size_t(c); i < out.px.size(); i += 4) {
            if (std::isfinite(v)) ok = ok && std::isfinite(out.px[i]);
            else ok = ok && std::memcmp(&out.px[i], &v, sizeof v) == 0;
          }
          INFO("stage " << int(st) << " layer " << c << " D " << v << " size " << size);
          CHECK(ok);
        }
      }
    }
  }
  // Through the whole render: non-finite source pixels give the same finite output rules as M4.
  const int w = 64, h = 36;
  TestImage src(tight(0, 0, w, h), 0.f);
  fillRandom(src, 9, true);
  SpatialControls s = grainOnly(kUhdS35, 4);
  s.sharpness = s.halation = s.bloom = true;
  s.grainAmount = 3;
  TestImage out(tight(0, 0, w, h), 7.f);
  renderFilm(src.cview(), out.view(), {0, 0, w, h}, film(), LookControls{}, s);
  bool finite = true;
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x)
      for (int c = 0; c < 4; ++c) finite = finite && std::isfinite(out.at(x, y)[c]);
  CHECK(finite);
}

TEST_CASE("grain controls are sanitised") {
  const double nan = std::numeric_limits<double>::quiet_NaN();
  SpatialControls s;
  s.grainAmount = nan;
  s.grainSize = nan;
  SpatialControls d = sanitize(s);
  CHECK(d.grainAmount == 1);
  CHECK(d.grainSize == 1);
  s.grainAmount = 9;
  s.grainSize = 9;
  d = sanitize(s);
  CHECK(d.grainAmount == 3);
  CHECK(d.grainSize == 2);
  s.grainAmount = -1;
  s.grainSize = 0.1;
  d = sanitize(s);
  CHECK(d.grainAmount == 0);
  CHECK(d.grainSize == 0.5);
  CHECK_FALSE(SpatialControls{}.grain);  // core default off: every M4 test stays on its path
  SpatialControls on;
  on.sharpness = on.halation = on.bloom = false;
  on.grain = true;
  CHECK(on.grainOn());
  CHECK(on.anyOn());
  on.grainAmount = 0;
  CHECK_FALSE(on.grainOn());
  CHECK_FALSE(on.anyOn());
}

// Ruling R4: the stage's a(D) is sigmaAt(D) * k exactly, k = 1 / unitApertureRms(48 um, Size), also
// between nodes where the 2383 curve bends hardest (where R1's 64-entry table was 7.5 % off).
TEST_CASE("grain amplitude is exact") {
  const double size = 1.3;
  const GrainAmplitudes amps = grainAmplitudes(film().grain, size);
  for (GrainStage st : {GrainStage::Negative, GrainStage::Print})
    for (int c = 0; c < 3; ++c) {
      const GrainLayer& g = layersOf(st)[size_t(c)];
      const GrainAmplitude& a = amps.stage(st)[size_t(c)];
      const double k = 1 / g.unitApertureRms(0.048, size);
      REQUIRE(a.count == int(g.density.size()));
      CHECK(a.k == doctest::Approx(k).epsilon(1e-12));
      // The sharpest bend: the node whose neighbouring slopes of ln sigma differ most.
      size_t bend = 1;
      double worstTurn = -1;
      for (size_t i = 1; i + 1 < g.density.size(); ++i) {
        const double s0 = std::log(g.sigma[i] / g.sigma[i - 1]) / (g.density[i] - g.density[i - 1]);
        const double s1 = std::log(g.sigma[i + 1] / g.sigma[i]) / (g.density[i + 1] - g.density[i]);
        if (std::abs(s1 - s0) > worstTurn) { worstTurn = std::abs(s1 - s0); bend = i; }
      }
      double worst = 0;
      for (double D : {0.5 * (g.density[bend - 1] + g.density[bend]), 0.5 * (g.density[bend] + g.density[bend + 1]),
                       g.density[bend], g.density.front() - 1, g.density.back() + 1})
        worst = std::max(worst, std::abs(a.at(D) / (g.sigmaAt(D) * k) - 1));
      MESSAGE("stage " << int(st) << " layer " << c << ": sharpest bend at D " << g.density[bend] << ", worst |a/(sigmaAt k) - 1| " << worst);
      CHECK(worst <= 1e-6);
      CHECK(a.at(std::numeric_limits<double>::quiet_NaN()) == 0);
    }
}

TEST_CASE("grain node limit") {
  Stock neg = res()->stock("kodak-5219-type"), print = res()->stock("kodak-2383-type");
  CHECK_NOTHROW(makeGrain(neg, print));
  neg.id = "test-too-many-grain-nodes";  // makeGrain memoises by id
  GranularityCurve& red = neg.granularity->layers[0];
  red.density.clear();
  red.sigma.clear();
  for (int i = 0; i <= kGrainMaxNodes; ++i) {
    red.density.push_back(0.1 + 0.02 * i);
    red.sigma.push_back(0.01);
  }
  CHECK_THROWS_AS(makeGrain(neg, print), std::runtime_error);
}

TEST_CASE("UHD grain timing" * doctest::skip()) {
  const int w = 3840, h = 2160;
  const Image4f in = uniform(w, h, {1.f, 1.2f, 1.4f});
  for (GrainStage st : {GrainStage::Negative, GrainStage::Print}) {
    Image4f img = in;
    const auto t0 = std::chrono::steady_clock::now();
    grainStage(img, {0, 0, w, h}, film(), grainOnly(kUhdS35, 1), st);
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    const std::string label = st == GrainStage::Negative ? "negative" : "print";
    MESSAGE(label << " grainStage Full, UHD: " << ms << " ms");
  }
}

// ---- Preview: precomputed tiling plates (one per stage and layer) ----
namespace {
constexpr int kTestPlate = 512;
struct PlateSizeGuard {
  PlateSizeGuard() { setCpuPlateSizeForTests(kTestPlate); }
  ~PlateSizeGuard() { setCpuPlateSizeForTests(0); }
};
SpatialControls previewOnly(double pxPerMm = kUhdS35, uint32_t frameSeed = 1) {
  SpatialControls s = grainOnly(pxPerMm, frameSeed);
  s.grainQuality = GrainQuality::Preview;
  return s;
}
double rmsOf(const std::vector<double>& v) {
  double s = 0;
  for (double x : v) s += x * x;
  return std::sqrt(s / double(v.size()));
}
}  // namespace

TEST_CASE("preview matches full") {
  PlateSizeGuard guard;
  const int n = kTestPlate;
  // The negative at UHD S35 (clouds of about 0.5 px: the simulated path), the print at twice that
  // scale. Each layer at its lowest, a middle and its highest digitised density.
  for (GrainStage st : {GrainStage::Negative, GrainStage::Print}) {
    const double scale = st == GrainStage::Negative ? kUhdS35 : 2 * kUhdS35;
    const auto& layers = layersOf(st);
    for (int c = 0; c < 3; ++c) {
      const GrainLayer& g = layers[size_t(c)];
      const double rPx = g.rMm * scale;
      for (double t : {0.0, 0.5, 1.0}) {
        std::array<float, 3> d{};
        for (size_t k = 0; k < 3; ++k)
          d[k] = float(layers[k].density.front() + t * (layers[k].density.back() - layers[k].density.front()));
        const Image4f in = uniform(n, n, d);
        const auto full = deviation(in, grainOnly(scale, 9), st, c);
        const auto prev = deviation(in, previewOnly(scale, 9), st, c);
        const double rf = rmsOf(full), rp = rmsOf(prev);
        const double hf = halfMaxRadius(full, n, n), hp = halfMaxRadius(prev, n, n);
        INFO("stage " << int(st) << " layer " << c << " t " << t << " r " << rPx << " px: rms " << rp << " vs " << rf
                      << ", half-max " << hp << " vs " << hf);
        CHECK(std::abs(rp / rf - 1) <= 0.10);
        CHECK(std::abs(hp / hf - 1) <= 0.10);
      }
    }
  }
}

TEST_CASE("plates tile") {
  const int n = 256;
  const auto plates = makePlates(film().grain, kUhdS35, 1.0, n);
  for (int st = 0; st < 2; ++st)
    for (int c = 0; c < 3; ++c) {
      const auto& p = plates->plates[size_t(st)][size_t(c)];
      // Std of the horizontal differences between columns x and x + 1 (wrapping for x = n - 1).
      auto diffStd = [&](int x) {
        double s2 = 0;
        for (int y = 0; y < n; ++y) {
          const double d = double(halfToFloat(p[size_t(y) * n + size_t((x + 1) % n)])) -
                           double(halfToFloat(p[size_t(y) * n + size_t(x)]));
          s2 += d * d;
        }
        return std::sqrt(s2 / n);
      };
      double interior = 0;
      int count = 0;
      for (int x = 8; x < n - 8; x += 4) { interior += diffStd(x); ++count; }
      interior /= count;
      INFO("stage " << st << " layer " << c << " seam " << diffStd(n - 1) << " interior " << interior);
      CHECK(std::abs(diffStd(n - 1) / interior - 1) <= 0.10);
    }
}

TEST_CASE("plates tile for clouds below the simulation radius") {
  const auto plates = makePlates(film().grain, kUhdS35 / 8, 1.0, 128);  // white-noise path
  const auto& p = plates->plates[0][0];
  double s = 0, s2 = 0;
  for (uint16_t h : p) { const double v = halfToFloat(h); s += v; s2 += v * v; }
  CHECK(s2 > 0);
  CHECK(std::abs(s / double(p.size())) < 0.1 * std::sqrt(s2 / double(p.size())));
}

TEST_CASE("preview changes every frame and repeats per frame") {
  PlateSizeGuard guard;
  const int n = 256;
  const Image4f in = uniform(n, n, {1.f, 1.2f, 1.4f});
  for (GrainStage st : {GrainStage::Negative, GrainStage::Print})
    for (int c = 0; c < 3; ++c) {
      const uint32_t a = frameSeedFromTime(3600.5), b = frameSeedFromTime(3600.5 + 1.0 / 24);
      const auto va = deviation(in, previewOnly(kUhdS35, a), st, c);
      CHECK(va == deviation(in, previewOnly(kUhdS35, a), st, c));
      const double r = correlation(va, deviation(in, previewOnly(kUhdS35, b), st, c));
      INFO("stage " << int(st) << " layer " << c << " frame-to-frame correlation " << r);
      CHECK(std::abs(r) < 0.05);
    }
}

TEST_CASE("plate view mapping") {
  // Every orientation is a bijection of the plate and the offset moves the view.
  const int size = 16;
  std::array<int, 8> seen{};
  for (uint32_t seed = 1; seed < 400; ++seed) {
    std::vector<int> hit(size * size, 0);
    for (int y = 0; y < size; ++y)
      for (int x = 0; x < size; ++x) {
        const uint32_t c = of_plate_coord(x - 5, y + 40, seed, size);
        REQUIRE((c & 0xffff) < uint32_t(size));
        REQUIRE((c >> 16) < uint32_t(size));
        hit[size_t(c >> 16) * size + (c & 0xffff)]++;
      }
    CHECK(std::all_of(hit.begin(), hit.end(), [](int h) { return h == 1; }));
  }
}

TEST_CASE("half conversion") {
  for (float f : {0.f, 1.f, -1.5f, 0.333251953125f, 65504.f, 6.1035156e-5f, 5.9604645e-8f}) CHECK(halfToFloat(floatToHalf(f)) == f);
  CHECK(floatToHalf(1.0f + 1.0f / 2048) == 0x3c00);  // tie rounds to even
  CHECK(floatToHalf(1.0f + 3.0f / 2048) == 0x3c02);
  CHECK(floatToHalf(1e6f) == 0x7c00);
  CHECK(floatToHalf(1e-9f) == 0);
  for (int i = 0; i < 2000; ++i) {
    const float f = (float(i) - 1000) * 0.00137f;
    CHECK(std::abs(halfToFloat(floatToHalf(f)) - f) <= std::abs(f) * 0.00049f + 1e-7f);
  }
}

TEST_CASE("preview grain off or amount 0 is M4") {
  PlateSizeGuard guard;
  const int w = 97, h = 61;
  TestImage src(tight(0, 0, w, h), 0.f);
  fillRandom(src, 7, true);
  const RenderWindow win{5, 7, 80, 50};
  LookControls look;
  look.exposureStops = 0.4;
  TestImage ref(tight(0, 0, w, h), 7.f);
  renderFilm(src.cview(), ref.view(), win, film(), look, effects(true));
  SpatialControls off = previewOnly();
  off.sharpness = off.halation = off.bloom = true;
  SpatialControls zero = off;
  zero.grain = true;
  zero.grainAmount = 0;
  off.grain = false;
  for (const SpatialControls& s : {off, zero}) {
    TestImage out(tight(0, 0, w, h), 7.f);
    renderFilm(src.cview(), out.view(), win, film(), look, s);
    CHECK(std::memcmp(out.bytes.data(), ref.bytes.data(), out.bytes.size()) == 0);
  }
  TestImage grainy(tight(0, 0, w, h), 7.f);
  SpatialControls on = previewOnly();
  on.sharpness = on.halation = on.bloom = true;
  renderFilm(src.cview(), grainy.view(), win, film(), look, on);
  CHECK(std::memcmp(grainy.bytes.data(), ref.bytes.data(), ref.bytes.size()) != 0);
}

// ---- The cpuPlates cache (ruling R8): one build per key however many renders ask at once; the last
// two sets kept ----
TEST_CASE("cpu: concurrent renders share one plate set") {
  PlateSizeGuard guard;
  setCpuPlateSizeForTests(64);
  const double scale = 97.3125;  // a key no other test uses
  const size_t builds = cpuPlateBuilds();
  const Image4f in = uniform(48, 40, {1.f, 1.2f, 1.4f});
  std::vector<Image4f> out(4, in);
  std::vector<std::shared_ptr<const GrainPlates>> got(4);
  std::vector<std::thread> threads;
  for (size_t t = 0; t < 4; ++t)
    threads.emplace_back([&, t] {
      got[t] = cpuPlates(film().grain, scale, 1.0, 64);
      grainStage(out[t], {0, 0, in.width, in.height}, film(), previewOnly(scale, 3), GrainStage::Negative);
    });
  for (auto& th : threads) th.join();
  CHECK(cpuPlateBuilds() == builds + 1);
  for (size_t t = 1; t < 4; ++t) {
    CHECK(got[t] == got[0]);
    CHECK(out[t].px == out[0].px);
  }
  CHECK(out[0].px != in.px);
}

TEST_CASE("cpu: plate cache keeps two sets") {
  const double scales[] = {61.0625, 62.0625, 63.0625};  // keys no other test uses
  const size_t builds = cpuPlateBuilds();
  for (double s : scales) cpuPlates(film().grain, s, 1.0, 32);
  CHECK(cpuPlateBuilds() == builds + 3);
  CHECK(cpuPlateSetCount() == 2);
  cpuPlates(film().grain, scales[2], 1.0, 32);  // newest: kept
  cpuPlates(film().grain, scales[1], 1.0, 32);  // kept
  CHECK(cpuPlateBuilds() == builds + 3);
  cpuPlates(film().grain, scales[0], 1.0, 32);  // released: built again
  CHECK(cpuPlateBuilds() == builds + 4);
  CHECK(cpuPlateSetCount() == 2);
}

// The first build of a full-size (kPlateSize) set at UHD Super 35, for the record (run with --no-skip).
TEST_CASE("plate build timing" * doctest::skip()) {
  const auto t0 = std::chrono::steady_clock::now();
  const auto plates = makePlates(film().grain, kUhdS35, 1.0, kPlateSize);
  const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  CHECK(plates->size == kPlateSize);
  MESSAGE("makePlates " << kPlateSize << ", UHD S35: " << ms << " ms");
}
