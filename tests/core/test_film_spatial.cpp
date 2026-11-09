// Tests for the CPU spatial pipeline: sharpness, edge effect, halation and bloom chained into the
// colour path.
#include <doctest/doctest.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>

#include "core/color.hpp"
#include "core/film_renderer.hpp"
#include "core/film_spatial.hpp"

using namespace openfilm;

namespace {
const std::string kData = OPENFILM_SOURCE_DATA_DIR;
constexpr double kPi = 3.14159265358979323846;

std::shared_ptr<const FilmResources> res() {
  static const auto r = FilmResources::load(kData, OPENFILM_COEFF_PATH);
  return r;
}
// One 5219 / 2383 bake shared by every test case (a bake takes ~1.5 s).
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
  const float* at(int x, int y) const { return pixelAddress(cview(), x, y); }
};
ImageGeometry tight(int x1, int y1, int x2, int y2) {
  return {x1, y1, x2, y2, std::ptrdiff_t(x2 - x1) * 4 * std::ptrdiff_t(sizeof(float))};
}

SpatialControls allOff() {
  SpatialControls s;
  s.sharpness = s.halation = s.bloom = false;
  return s;
}
SpatialControls allOn(double pxPerMm) {
  SpatialControls s;
  s.sharpness = s.halation = s.bloom = true;
  s.remjet = false;
  s.pxPerMm = pxPerMm;
  return s;
}

Image4f blank(int w, int h, float rgb = 0, float a = 1) {
  Image4f img;
  img.width = w;
  img.height = h;
  img.px.assign(size_t(w) * size_t(h) * 4, rgb);
  for (size_t i = 3; i < img.px.size(); i += 4) img.px[i] = a;
  return img;
}

// Mid-grey (scene 0.18) exposure per layer, through the exposure table.
std::array<float, 3> midGreyExposure() {
  const float g = float(diEncode(0.18));
  const auto e = film().lutE.lookup(g, g, g);
  return {std::pow(10.f, e[0]) - float(kExposureLutEps), std::pow(10.f, e[1]) - float(kExposureLutEps),
          std::pow(10.f, e[2]) - float(kExposureLutEps)};
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

// Full stage chain, written out as the GPU will run it.
void runStages(const ConstImageView& src, const ImageView& dst, const RenderWindow& win, const LookControls& look,
               const SpatialControls& s) {
  const BakedFilm& f = film();
  const WorkRect rect = workRect(src, win, spatialMarginPx(f.optics, s));
  Image4f img;
  exposeStage(src, rect, f, exposureGain(look.exposureStops), img);
  exposureEffects(img, f.optics, s);
  developStage(img, f);
  edgeEffect(img, f.optics, s);
  printExposeStage(img, f);
  printSharpness(img, f.optics, s);
  printDevelopStage(img, f, printOffsetsLogE(f.balance, look));
  printEdgeEffect(img, f.optics, s);
  finishStage(img, rect, f, dst, win);
}

// Least-squares fit of a + b sin(wx) + c cos(wx) over [x0, x1); returns hypot(b, c).
double sineAmplitude(const std::vector<double>& v, double w, int x0, int x1) {
  double m[3][4] = {};
  for (int x = x0; x < x1; ++x) {
    const double basis[3] = {1, std::sin(w * x), std::cos(w * x)};
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) m[i][j] += basis[i] * basis[j];
      m[i][3] += basis[i] * v[size_t(x)];
    }
  }
  for (int i = 0; i < 3; ++i) {  // Gauss-Jordan; the system is well conditioned
    for (int r = 0; r < 3; ++r) {
      if (r == i) continue;
      const double k = m[r][i] / m[i][i];
      for (int j = 0; j < 4; ++j) m[r][j] -= k * m[i][j];
    }
  }
  return std::hypot(m[1][3] / m[1][1], m[2][3] / m[2][2]);
}

double secondMomentRadius(const Image4f& img, const Image4f* minus, int channel) {
  const double cx = (img.width - 1) / 2.0, cy = (img.height - 1) / 2.0;
  double s = 0, s2 = 0;
  for (int y = 0; y < img.height; ++y)
    for (int x = 0; x < img.width; ++x) {
      double v = img.at(x, y)[channel];
      if (minus) v -= minus->at(x, y)[channel];
      const double r2 = (x - cx) * (x - cx) + (y - cy) * (y - cy);
      s += v;
      s2 += v * r2;
    }
  return std::sqrt(s2 / s);
}

// The brightest code the print can show: its clear base (the lutD domain's low corner). M3 itself
// reaches it (red 1.0165 for this bake: BT.1886 code is not clipped at 1), so it, not 1, is the top.
std::array<float, 3> printWhite() {
  Image4f dp = blank(1, 1);
  for (int c = 0; c < 3; ++c) dp.px[size_t(c)] = float(film().lutD.domainMin()[size_t(c)]);
  float out[4] = {0, 0, 0, 0};
  finishStage(dp, {0, 0, 1, 1}, film(), {out, tight(0, 0, 1, 1)}, {0, 0, 1, 1});
  return {out[0], out[1], out[2]};
}

bool allFiniteInRange(const TestImage& img, const RenderWindow& win) {
  const auto top = printWhite();
  for (int y = win.y1; y < win.y2; ++y)
    for (int x = win.x1; x < win.x2; ++x)
      for (int c = 0; c < 3; ++c) {
        const float v = img.at(x, y)[c];
        if (!std::isfinite(v) || v < 0.f || v > top[size_t(c)] + 1e-6f) {
          std::printf("out of range at %d,%d c%d: %g\n", x, y, c, double(v));
          return false;
        }
      }
  return true;
}
}  // namespace

TEST_CASE("all off is M3") {
  TestImage src(tight(0, 0, 97, 61), 0.f);
  fillRandom(src, 7, true);
  TestImage a(tight(0, 0, 97, 61), 7.f), b(tight(0, 0, 97, 61), 7.f);
  const RenderWindow win{5, 7, 80, 50};
  LookControls look;
  look.exposureStops = 0.6;
  look.printerPoints = {2, -1, 3};
  renderFilm(src.cview(), a.view(), win, film(), look, allOff());
  renderFilmCpu(src.cview(), b.view(), win, film(), look);
  CHECK(std::memcmp(a.bytes.data(), b.bytes.data(), a.bytes.size()) == 0);
}

TEST_CASE("staged chain without effects matches M3") {
  TestImage src(tight(0, 0, 97, 61), 0.f);
  fillRandom(src, 11, true);
  // dst and window reach past the source: those pixels must become (0,0,0,0) as in M3.
  TestImage a(tight(0, 0, 100, 64), 7.f), b(tight(0, 0, 100, 64), 7.f);
  const RenderWindow win{3, 2, 100, 64};
  LookControls look;
  look.exposureStops = -0.7;
  look.printerPoints = {-3, 2, 1};
  look.printDensity = 1;
  runStages(src.cview(), a.view(), win, look, allOff());
  renderFilmCpu(src.cview(), b.view(), win, film(), look);
  double worst = 0;
  bool alphaExact = true;
  for (int y = 0; y < 64; ++y)
    for (int x = 0; x < 100; ++x) {
      for (int c = 0; c < 3; ++c) worst = std::max(worst, double(std::abs(a.at(x, y)[c] - b.at(x, y)[c])));
      if (std::memcmp(a.at(x, y) + 3, b.at(x, y) + 3, sizeof(float)) != 0) alphaExact = false;
    }
  INFO("max |diff| " << worst);
  CHECK(worst <= 1e-5);
  CHECK(alphaExact);
}

TEST_CASE("measured sharpness matches the datasheet") {
  const Stock& neg = res()->stock("kodak-5219-type");
  SpatialControls on = allOff();
  on.sharpness = true;
  on.pxPerMm = 4096 / 5.0;  // gate Custom 5 mm at 4096 px
  const SpatialControls off = [&] { SpatialControls s = on; s.sharpness = false; return s; }();
  const int w = 4096, h = 4;
  const double mid = 0.18;
  // Density modulation of a horizontal 2 % exposure sine, all three layers at once.
  auto modulation = [&](double cyclesPerMm, const SpatialControls& s) {
    const double wpx = 2 * kPi * cyclesPerMm / s.pxPerMm;
    TestImage src(tight(0, 0, w, h), 1.f);
    for (int y = 0; y < h; ++y)
      for (int x = 0; x < w; ++x) {
        const float di = float(diEncode(mid * (1 + 0.02 * std::sin(wpx * x))));
        src.at(x, y)[0] = src.at(x, y)[1] = src.at(x, y)[2] = di;
      }
    const WorkRect rect{0, 0, w, h};
    Image4f img;
    exposeStage(src.cview(), rect, film(), 1.f, img);
    exposureEffects(img, film().optics, s);
    developStage(img, film());
    edgeEffect(img, film().optics, s);
    std::array<double, 3> amp{};
    for (int c = 0; c < 3; ++c) {
      std::vector<double> row(static_cast<size_t>(w));
      for (int x = 0; x < w; ++x) row[size_t(x)] = img.at(x, h / 2)[c];
      amp[size_t(c)] = sineAmplitude(row, wpx, w / 4, 3 * w / 4);
    }
    return amp;
  };
  const char* names[3] = {"red", "green", "blue"};
  for (int c = 0; c < 3; ++c) {
    double worst = 0, worstF = 0;
    for (const auto& p : neg.mtf[size_t(c)]) {
      if (p[0] > 200) continue;
      const double resp = modulation(p[0], on)[size_t(c)] / modulation(p[0], off)[size_t(c)];
      const double e = std::isfinite(resp) ? std::abs(resp - p[1]) : 1e9;  // NaN must fail
      if (std::getenv("OPENFILM_PRINT_MTF")) std::printf("mtf %s %.3f datasheet %.4f measured %.4f\n", names[c], p[0], p[1], resp);
      if (e > worst) { worst = e; worstF = p[0]; }
    }
    MESSAGE("5219 " << std::string(names[c]) << ": worst |measured - datasheet| " << worst << " at " << worstF << " c/mm");
    CHECK(worst <= 0.07);
  }
}

TEST_CASE("halation physics") {
  const auto mid = midGreyExposure();
  const int n = 301;
  Image4f in = blank(n, n);
  for (int c = 0; c < 3; ++c) in.at(n / 2, n / 2)[c] = 1000.f * mid[size_t(c)];
  SpatialControls s = allOff();
  s.halation = true;
  auto energy = [&](const SpatialControls& sc) {
    Image4f e = in;
    exposureEffects(e, film().optics, sc);
    std::array<double, 3> out{};
    for (int c = 0; c < 3; ++c) {
      double sum = 0;
      for (int y = 0; y < n; ++y)
        for (int x = 0; x < n; ++x) sum += double(e.at(x, y)[c]) - double(in.at(x, y)[c]);
      out[size_t(c)] = sum;
    }
    return out;
  };
  const Halation& hal = film().optics.halation;
  const auto on = energy(s);
  s.remjet = false;
  const auto off = energy(s);
  MESSAGE("halo energy rem-jet on  r " << on[0] << " g " << on[1] << " b " << on[2]);
  MESSAGE("halo energy rem-jet off r " << off[0] << " g " << off[1] << " b " << off[2]);
  // Red: all the returned light, h times the red exposure (the halo kernel is normalised).
  CHECK(on[0] == doctest::Approx(hal.reflectanceOn * 1000.0 * mid[0]).epsilon(0.01));
  CHECK(on[0] > on[1]);
  CHECK(on[1] > 0);
  CHECK(on[2] >= 0);
  // Ruling R7: the split decides green and blue; 5219's blue share is exactly 0.
  CHECK(on[1] == doctest::Approx(hal.layerSplit[1] / hal.layerSplit[0] * on[0]).epsilon(0.05));
  if (hal.layerSplit[2] == 0) CHECK(on[2] == 0);
  else CHECK(on[2] == doctest::Approx(hal.layerSplit[2] / hal.layerSplit[0] * on[0]).epsilon(0.05));
  CHECK(off[0] / on[0] == doctest::Approx(hal.reflectanceOff / hal.reflectanceOn).epsilon(0.05));
  CHECK(off[1] / on[1] == doctest::Approx(hal.reflectanceOff / hal.reflectanceOn).epsilon(0.05));
  // Halation off adds exactly nothing.
  Image4f same = in;
  exposureEffects(same, film().optics, allOff());
  CHECK(same.px == in.px);
  SpatialControls zero = allOff();
  zero.halation = false;
  zero.remjet = false;
  zero.halationStrength = 5;
  Image4f same2 = in;
  exposureEffects(same2, film().optics, zero);
  CHECK(same2.px == in.px);
}

TEST_CASE("edge effect keeps mean density") {
  std::mt19937 rng(3);
  std::uniform_real_distribution<float> u(0.2f, 2.2f);
  const int n = 256, border = 64;
  Image4f d = blank(n, n, 1.2f);
  for (int y = border; y < n - border; ++y)
    for (int x = border; x < n - border; ++x)
      for (int c = 0; c < 3; ++c) d.at(x, y)[c] = u(rng);
  SpatialControls s = allOff();
  s.sharpness = true;
  s.pxPerMm = 30;  // the widest edge kernel (print, sigma 0.5 mm = 15 px) then reaches < 64 px
  auto mean = [&](const Image4f& img, int c) {
    double sum = 0;
    for (int y = 0; y < n; ++y)
      for (int x = 0; x < n; ++x) sum += img.at(x, y)[c];
    return sum / (double(n) * n);
  };
  Image4f neg = d, print = d;
  edgeEffect(neg, film().optics, s);
  printEdgeEffect(print, film().optics, s);
  for (int c = 0; c < 3; ++c) {
    CHECK(std::abs(mean(neg, c) - mean(d, c)) <= 1e-4);
    CHECK(std::abs(mean(print, c) - mean(d, c)) <= 1e-4);
  }
  CHECK(neg.px != d.px);  // it did something
  CHECK(print.px != d.px);
  Image4f none = d;
  edgeEffect(none, film().optics, allOff());
  printEdgeEffect(none, film().optics, allOff());
  CHECK(none.px == d.px);
}

TEST_CASE("format scaling") {
  const double pA = 3840 / 24.89, pB = 3840 / 49.78;
  SUBCASE("halation") {
    const int n = 401;
    Image4f in = blank(n, n);
    for (int c = 0; c < 3; ++c) in.at(n / 2, n / 2)[c] = 1000.f;
    auto radius = [&](double px) {
      SpatialControls s = allOff();
      s.halation = true;
      s.pxPerMm = px;
      Image4f e = in;
      exposureEffects(e, film().optics, s);
      return secondMomentRadius(e, &in, 0);
    };
    const double rA = radius(pA), rB = radius(pB);
    MESSAGE("halo rms radius " << rA << " px vs " << rB << " px");
    CHECK(std::abs(rA - 2 * rB) <= 1);
  }
  SUBCASE("bloom") {
    const int n = 801;
    Image4f in = blank(n, n);
    for (int c = 0; c < 3; ++c) in.at(n / 2, n / 2)[c] = 1000.f;
    auto radius = [&](double px) {
      SpatialControls s = allOff();
      s.bloom = true;
      s.bloomAmount = 1;
      s.pxPerMm = px;
      Image4f e = in;
      exposureEffects(e, film().optics, s);
      return secondMomentRadius(e, nullptr, 0);
    };
    const double rA = radius(pA), rB = radius(pB);
    MESSAGE("bloom rms radius " << rA << " px vs " << rB << " px");
    CHECK(std::abs(rA - 2 * rB) <= 1);
  }
}

TEST_CASE("render scale matches the downsampled render") {
  const int w = 512, h = 288;
  TestImage full(tight(0, 0, w, h), 1.f);
  struct Blob { double x, y, sigma, amp; };
  const Blob blobs[] = {{120, 90, 12, 0.45}, {300, 160, 16, 0.35}, {420, 70, 10, 0.5}, {200, 220, 20, 0.3}};
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) {
      double v[3] = {0.30 + 0.05 * std::sin(x / 60.0), 0.33, 0.36 - 0.04 * std::cos(y / 50.0)};
      for (const Blob& b : blobs) {
        const double g = b.amp * std::exp(-((x - b.x) * (x - b.x) + (y - b.y) * (y - b.y)) / (2 * b.sigma * b.sigma));
        for (double& c : v) c += g;
      }
      for (int c = 0; c < 3; ++c) full.at(x, y)[c] = float(v[c]);
    }
  TestImage half(tight(0, 0, w / 2, h / 2), 1.f);
  for (int y = 0; y < h / 2; ++y)
    for (int x = 0; x < w / 2; ++x)
      for (int c = 0; c < 3; ++c)
        half.at(x, y)[c] = 0.25f * (full.at(2 * x, 2 * y)[c] + full.at(2 * x + 1, 2 * y)[c] +
                                    full.at(2 * x, 2 * y + 1)[c] + full.at(2 * x + 1, 2 * y + 1)[c]);
  const double p = w / 24.89;
  TestImage outFull(tight(0, 0, w, h), 0.f), outHalf(tight(0, 0, w / 2, h / 2), 0.f);
  renderFilm(full.cview(), outFull.view(), {0, 0, w, h}, film(), LookControls{}, allOn(p));
  renderFilm(half.cview(), outHalf.view(), {0, 0, w / 2, h / 2}, film(), LookControls{}, allOn(p / 2));
  double sum = 0;
  int count = 0;
  const int b = 16;
  for (int y = b; y < h / 2 - b; ++y)
    for (int x = b; x < w / 2 - b; ++x)
      for (int c = 0; c < 3; ++c) {
        const double down = 0.25 * (outFull.at(2 * x, 2 * y)[c] + outFull.at(2 * x + 1, 2 * y)[c] +
                                    outFull.at(2 * x, 2 * y + 1)[c] + outFull.at(2 * x + 1, 2 * y + 1)[c]);
        sum += std::abs(down - outHalf.at(x, y)[c]);
        ++count;
      }
  MESSAGE("render-scale mean |diff| " << sum / count);
  CHECK(sum / count <= 2e-3);
}

TEST_CASE("sub-window equals full frame") {
  const int w = 960, h = 540;
  TestImage src(tight(0, 0, w, h), 0.f);
  fillRandom(src, 5, false);
  for (int y = 200; y < 260; ++y)  // a hard bright patch for the halo and bloom to spread
    for (int x = 400; x < 470; ++x)
      for (int c = 0; c < 3; ++c) src.at(x, y)[c] = 0.95f;
  const SpatialControls s = allOn(8.0);
  const RenderWindow full{0, 0, w, h};
  const RenderWindow sub{int(w * 0.3), int(h * 0.3), int(w * 0.7), int(h * 0.7)};
  const int margin = spatialMarginPx(film().optics, s);
  const WorkRect r = workRect(src.cview(), sub, margin);
  INFO("margin " << margin << " work rect " << r.x1 << "," << r.y1 << " - " << r.x2 << "," << r.y2);
  REQUIRE((r.x1 > 0 || r.y1 > 0 || r.x2 < w || r.y2 < h));  // the tile really is cut
  TestImage a(tight(0, 0, w, h), 7.f), b(tight(0, 0, w, h), 7.f);
  renderFilm(src.cview(), a.view(), full, film(), LookControls{}, s);
  renderFilm(src.cview(), b.view(), sub, film(), LookControls{}, s);
  double worst = 0;
  for (int y = sub.y1; y < sub.y2; ++y)
    for (int x = sub.x1; x < sub.x2; ++x)
      for (int c = 0; c < 4; ++c) worst = std::max(worst, double(std::abs(a.at(x, y)[c] - b.at(x, y)[c])));
  INFO("max |diff| " << worst);
  CHECK(worst <= 1e-6);
  CHECK(b.at(sub.x1 - 1, sub.y1)[0] == 7.f);  // nothing written outside the window

  // Host ignored the region of interest: the source is only the window.
  TestImage cut(tight(sub.x1, sub.y1, sub.x2, sub.y2), 0.f);
  for (int y = sub.y1; y < sub.y2; ++y)
    for (int x = sub.x1; x < sub.x2; ++x) std::memcpy(cut.at(x, y), src.at(x, y), 4 * sizeof(float));
  TestImage c(tight(0, 0, w, h), 7.f);
  renderFilm(cut.cview(), c.view(), sub, film(), LookControls{}, s);
  CHECK(allFiniteInRange(c, sub));
}

TEST_CASE("extreme highlights stay finite") {
  const int w = 96, h = 54;
  TestImage src(tight(0, 0, w, h), 1.f);
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w / 3; ++x) src.at(x, y)[0] = src.at(x, y)[1] = src.at(x, y)[2] = 0.f;
  SpatialControls s = allOn(3840 / 24.89);
  s.halationStrength = 5;
  s.halationRadius = 4;
  s.bloomAmount = 1;
  s.bloomRadiusMm = 5;
  LookControls look;
  look.exposureStops = 10;
  TestImage out(tight(0, 0, w, h), 7.f);
  renderFilm(src.cview(), out.view(), {0, 0, w, h}, film(), look, s);
  CHECK(allFiniteInRange(out, {0, 0, w, h}));
}

TEST_CASE("tiny image and huge kernels") {
  for (const auto& size : {std::array<int, 2>{64, 36}, std::array<int, 2>{1, 1}}) {
    const int w = size[0], h = size[1];
    TestImage src(tight(0, 0, w, h), 0.f);
    fillRandom(src, 9, true);
    SpatialControls s = allOn(pixelsPerMm(w, gateWidthMm(FilmFormat::Std16, 0)));
    s.bloomRadiusMm = 5;
    s.halationRadius = 4;
    TestImage out(tight(0, 0, w, h), 7.f);
    renderFilm(src.cview(), out.view(), {0, 0, w, h}, film(), LookControls{}, s);
    CHECK(allFiniteInRange(out, {0, 0, w, h}));
  }
}

TEST_CASE("sanitize") {
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const SpatialControls d;
  SpatialControls s;
  s.halationStrength = nan;
  s.halationRadius = -1;
  s.bloomAmount = 7;
  s.bloomRadiusMm = std::numeric_limits<double>::infinity();
  s.pxPerMm = nan;
  SpatialControls t = sanitize(s);
  CHECK(t.halationStrength == 1);
  CHECK(t.halationRadius == 0.25);
  CHECK(t.bloomAmount == 1);
  CHECK(t.bloomRadiusMm == d.bloomRadiusMm);
  CHECK(t.pxPerMm == d.pxPerMm);
  s.pxPerMm = 0;
  CHECK(sanitize(s).pxPerMm == d.pxPerMm);
  s.pxPerMm = -5;
  CHECK(sanitize(s).pxPerMm == d.pxPerMm);
  s.halationStrength = 9;
  s.bloomRadiusMm = 0.001;
  s.halationRadius = nan;
  t = sanitize(s);
  CHECK(t.halationStrength == 5);
  CHECK(t.bloomRadiusMm == 0.05);
  CHECK(t.halationRadius == 1);
  // The renderer sanitises too: NaN controls give finite output.
  TestImage src(tight(0, 0, 32, 18), 0.5f), out(tight(0, 0, 32, 18), 7.f);
  SpatialControls bad = allOn(nan);
  bad.halationStrength = nan;
  bad.bloomAmount = nan;
  bad.bloomRadiusMm = nan;
  bad.halationRadius = nan;
  renderFilm(src.cview(), out.view(), {0, 0, 32, 18}, film(), LookControls{}, bad);
  CHECK(allFiniteInRange(out, {0, 0, 32, 18}));
}

// Not part of the suite: `test_core -tc="UHD timing" --no-skip`.
TEST_CASE("UHD timing" * doctest::skip()) {
  const int w = 3840, h = 2160;
  TestImage src(tight(0, 0, w, h), 0.f);
  fillRandom(src, 1, false);
  TestImage out(tight(0, 0, w, h), 0.f);
  SpatialControls s = allOn(w / 24.89);
  for (int i = 0; i < 3; ++i) {
    const auto t0 = std::chrono::steady_clock::now();
    renderFilm(src.cview(), out.view(), {0, 0, w, h}, film(), LookControls{}, s);
    const auto t1 = std::chrono::steady_clock::now();
    MESSAGE("UHD all on: " << std::chrono::duration<double, std::milli>(t1 - t0).count() << " ms");
  }
}
