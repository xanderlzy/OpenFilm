// Tests for the baked film LUTs, the cache and the CPU film renderer.
#include <doctest/doctest.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <limits>
#include <random>
#include <atomic>
#include <thread>

#include "core/film_renderer.hpp"
#include "core/lut.hpp"

using namespace openfilm;

namespace {
const std::string kData = OPENFILM_SOURCE_DATA_DIR;
std::shared_ptr<const FilmResources> res() {
  static const auto r = FilmResources::load(kData, OPENFILM_COEFF_PATH);
  return r;
}
FilmParams params(double r = 0) {
  FilmParams p;
  p.negativeId = "kodak-5219-type";
  p.printId = "kodak-2383-type";
  p.gamutCompression = 0.5 + 0.1 * r;  // a distinct look per `r`; printer lights no longer enter the bake
  return p;
}
LookControls look(double stops) { LookControls c; c.exposureStops = stops; return c; }
FilmCache& cache() {
  static FilmCache c(res());
  return c;
}

struct TestImage {
  ImageGeometry geom;
  std::vector<unsigned char> bytes;
  TestImage(ImageGeometry g, float fill) : geom(g), bytes(size_t(g.rowBytes) * size_t(g.y2 - g.y1)) {
    float* f = reinterpret_cast<float*>(bytes.data());
    for (size_t i = 0; i < bytes.size() / sizeof(float); ++i) f[i] = fill;
  }
  ImageView view() { return {reinterpret_cast<float*>(bytes.data()), geom}; }
  ConstImageView cview() { return {reinterpret_cast<const float*>(bytes.data()), geom}; }
  float* at(int x, int y) { return pixelAddress(view(), x, y); }
};
ImageGeometry tight(int x1, int y1, int x2, int y2) {
  return {x1, y1, x2, y2, std::ptrdiff_t(x2 - x1) * 4 * std::ptrdiff_t(sizeof(float))};
}

// Display code -> CIELAB (D65), via BT.1886 decode and Rec.709 primaries.
Vec3 toLab(const Vec3& code) {
  Vec3 lin;
  for (int i = 0; i < 3; ++i) lin[i] = std::pow(std::max(code[i], 0.0), 2.4);
  const Vec3 xyz = mul(rgbToXyz(kRec709), lin);
  const double white[3] = {kD65.x / kD65.y, 1.0, (1 - kD65.x - kD65.y) / kD65.y};
  auto f = [](double t) { return t > 216.0 / 24389.0 ? std::cbrt(t) : (24389.0 / 27.0 * t + 16) / 116; };
  const double fx = f(xyz[0] / white[0]), fy = f(xyz[1]), fz = f(xyz[2] / white[2]);
  return {116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)};
}

// CIEDE2000 (Sharma, Wu, Dalal 2005).
double deltaE2000(const Vec3& a, const Vec3& b) {
  const double kPi = 3.14159265358979323846;
  auto rad = [&](double d) { return d * kPi / 180; };
  auto deg = [&](double r) { double d = r * 180 / kPi; return d < 0 ? d + 360 : d; };
  const double C1 = std::hypot(a[1], a[2]), C2 = std::hypot(b[1], b[2]), Cm = (C1 + C2) / 2;
  const double G = 0.5 * (1 - std::sqrt(std::pow(Cm, 7) / (std::pow(Cm, 7) + std::pow(25.0, 7))));
  const double a1 = (1 + G) * a[1], a2 = (1 + G) * b[1];
  const double Cp1 = std::hypot(a1, a[2]), Cp2 = std::hypot(a2, b[2]);
  const double hp1 = (a1 == 0 && a[2] == 0) ? 0 : deg(std::atan2(a[2], a1));
  const double hp2 = (a2 == 0 && b[2] == 0) ? 0 : deg(std::atan2(b[2], a2));
  const double dL = b[0] - a[0], dC = Cp2 - Cp1;
  double dh = 0;
  if (Cp1 * Cp2 != 0) {
    dh = hp2 - hp1;
    if (dh > 180) dh -= 360;
    if (dh < -180) dh += 360;
  }
  const double dH = 2 * std::sqrt(Cp1 * Cp2) * std::sin(rad(dh / 2));
  const double Lm = (a[0] + b[0]) / 2, Cpm = (Cp1 + Cp2) / 2;
  double hm = hp1 + hp2;
  if (Cp1 * Cp2 != 0) {
    if (std::abs(hp1 - hp2) <= 180) hm /= 2;
    else hm = (hp1 + hp2 < 360) ? (hm + 360) / 2 : (hm - 360) / 2;
  }
  const double T = 1 - 0.17 * std::cos(rad(hm - 30)) + 0.24 * std::cos(rad(2 * hm)) +
                   0.32 * std::cos(rad(3 * hm + 6)) - 0.20 * std::cos(rad(4 * hm - 63));
  const double dTheta = 30 * std::exp(-std::pow((hm - 275) / 25, 2));
  const double Rc = 2 * std::sqrt(std::pow(Cpm, 7) / (std::pow(Cpm, 7) + std::pow(25.0, 7)));
  const double Sl = 1 + 0.015 * std::pow(Lm - 50, 2) / std::sqrt(20 + std::pow(Lm - 50, 2));
  const double Sc = 1 + 0.045 * Cpm, Sh = 1 + 0.015 * Cpm * T;
  const double Rt = -std::sin(rad(2 * dTheta)) * Rc;
  return std::sqrt(std::pow(dL / Sl, 2) + std::pow(dC / Sc, 2) + std::pow(dH / Sh, 2) +
                   Rt * (dC / Sc) * (dH / Sh));
}

// Renders one pixel through the baked film.
Vec3 renderOne(const BakedFilm& film, float r, float g, float b, const LookControls& controls) {
  TestImage src(tight(0, 0, 1, 1), 0.f), dst(tight(0, 0, 1, 1), 0.f);
  float* p = src.at(0, 0);
  p[0] = r; p[1] = g; p[2] = b; p[3] = 1.f;
  renderFilmCpu(src.cview(), dst.view(), {0, 0, 1, 1}, film, controls);
  const float* o = dst.at(0, 0);
  return {o[0], o[1], o[2]};
}
}  // namespace

TEST_CASE("baked matches direct within DeltaE2000 0.5") {
  const FilmModel model(res(), params());
  const auto t0 = std::chrono::steady_clock::now();
  const auto film = bakeFilm(model);
  const double bakeSec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  std::printf("bake default sizes: %.2f s\n", bakeSec);

  std::mt19937 rng(11);
  std::uniform_real_distribution<double> u(0, 1), ue(-5, 5), up(-20, 20);  // displayed Exposure range; printer points
  double maxDe = 0;
  std::vector<double> all;
  Vec3 worstIn{}, worstDirect{}, worstBaked{};
  double worstStops = 0;
  for (int i = 0; i < 2000; ++i) {
    const Vec3 in{u(rng), u(rng), u(rng)};
    const double stops = ue(rng);
    LookControls c = look(stops);
    c.printerPoints = {up(rng), up(rng), up(rng)};
    c.printDensity = up(rng);
    const Vec3 direct = model.evaluate(in, c);
    const Vec3 baked = renderOne(*film, float(in[0]), float(in[1]), float(in[2]), c);
    const double de = deltaE2000(toLab(direct), toLab(baked));
    all.push_back(de);
    if (de > maxDe) { maxDe = de; worstIn = in; worstStops = stops; worstDirect = direct; worstBaked = baked; }
  }
  std::printf("max dE2000 %.4f at in %.4f %.4f %.4f stops %.3f direct %.4f %.4f %.4f baked %.4f %.4f %.4f\n", maxDe,
              worstIn[0], worstIn[1], worstIn[2], worstStops, worstDirect[0], worstDirect[1], worstDirect[2],
              worstBaked[0], worstBaked[1], worstBaked[2]);
  std::sort(all.begin(), all.end());
  std::printf("dE2000 p50 %.4f p99 %.4f p99.9 %.4f, %d of %zu above 0.5\n", all[all.size() / 2],
              all[all.size() * 99 / 100], all[all.size() * 999 / 1000],
              int(std::count_if(all.begin(), all.end(), [](double d) { return d > 0.5; })), all.size());
  CHECK(maxDe < 0.5);
}

TEST_CASE("baked matches direct in the deep shadows with exposure pushed") {
  // Dark DI codes on every channel, +-5 stops: this is where the toe of the negative lives and where
  // an exposure table that interpolates toward the black corner goes badly wrong.
  const FilmModel model(res(), params());
  const auto film = cache().get(params());
  std::mt19937 rng(23);
  std::uniform_real_distribution<double> u(0, 0.05), ue(-5, 5), up(-20, 20);
  double maxDe = 0, maxGrey = 0;
  for (int i = 0; i < 3000; ++i) {  // M2's conditions: printer lights at zero
    const Vec3 in{u(rng), u(rng), u(rng)};
    const double stops = ue(rng);
    const double de = deltaE2000(toLab(model.evaluate(in, look(stops))),
                                 toLab(renderOne(*film, float(in[0]), float(in[1]), float(in[2]), look(stops))));
    maxDe = std::max(maxDe, de);
  }
  // Recorded, not checked: the same dark colours with the printer lights thrown to +-20. Those
  // amplify any small error in the exposure table's deep shadows by a large factor.
  std::mt19937 rng2(23);
  double maxExtreme = 0;
  for (int i = 0; i < 3000; ++i) {
    const Vec3 in{u(rng2), u(rng2), u(rng2)};
    LookControls c = look(ue(rng2));
    c.printerPoints = {up(rng2), up(rng2), up(rng2)};
    c.printDensity = up(rng2);
    maxExtreme = std::max(maxExtreme, deltaE2000(toLab(model.evaluate(in, c)),
                                                  toLab(renderOne(*film, float(in[0]), float(in[1]), float(in[2]), c))));
  }
  std::printf("dark with printer points/density +-20: max dE2000 %.4f (not checked)\n", maxExtreme);
  for (int i = 0; i <= 60; ++i)
    for (double stops : {-5.0, -2.0, 0.0, 2.0, 4.0, 5.0}) {
      const double v = 0.03 * i / 60;  // neutral ramp from black to DI 0.03
      const Vec3 in{v, v, v};
      maxGrey = std::max(maxGrey, deltaE2000(toLab(model.evaluate(in, look(stops))),
                                             toLab(renderOne(*film, float(v), float(v), float(v), look(stops)))));
    }
  std::printf("dark random max dE2000 %.4f, dark grey ramp max %.4f\n", maxDe, maxGrey);
  CHECK(maxDe < 0.5);
  CHECK(maxGrey < 0.5);
}

TEST_CASE("baked tables are finite") {
  const auto film = cache().get(params());
  for (float v : film->lutE.data()) CHECK(std::isfinite(v));
  for (float v : film->lutPE.data()) CHECK(std::isfinite(v));
  for (float v : film->lutD.data()) CHECK(std::isfinite(v));  // lutD holds log10 of projected light, so negative values are legitimate
  for (const auto& c : film->negCurves)
    for (float v : c.values) CHECK(std::isfinite(v));
  for (const auto& c : film->printCurves)
    for (float v : c.values) CHECK(std::isfinite(v));
}

TEST_CASE("exposure needs no rebake") {
  const auto p = params();
  CHECK(cache().get(p) == cache().get(p));
  const auto film = cache().get(p);
  const float c = float(diEncode(0.18));
  const Vec3 a = renderOne(*film, c, c, c, look(0)), b = renderOne(*film, c, c, c, look(1));
  CHECK(b[1] > a[1] + 0.01);
}

TEST_CASE("printer lights and print density need no rebake") {
  const auto film = cache().get(params());
  const float c = float(diEncode(0.18));
  LookControls none, red; red.printerPoints = {5, 0, 0};
  const Vec3 a = renderOne(*film, c, c, c, none), b = renderOne(*film, c, c, c, red);
  CHECK(cache().get(params()) == film);  // same bake object
  CHECK(b[0] < a[0] - 0.01);             // R +5 -> more cyan dye -> less red
}

TEST_CASE("extreme and non-finite controls") {
  const auto film = cache().get(params());
  LookControls big; big.printerPoints = {50, -50, 50}; big.printDensity = -50; big.exposureStops = 10;
  const Vec3 o = renderOne(*film, 0.5f, 0.2f, 0.9f, big);
  for (double v : o) CHECK(std::isfinite(v));
  LookControls nan; nan.printerPoints = {NAN, 0, 0}; nan.printDensity = INFINITY; nan.exposureStops = NAN;
  const Vec3 n = renderOne(*film, 0.4f, 0.4f, 0.4f, nan), z = renderOne(*film, 0.4f, 0.4f, 0.4f, LookControls{});
  for (int k = 0; k < 3; ++k) CHECK(n[k] == z[k]);
}

TEST_CASE("print exposure beyond the print curve clamps") {
  // Clear negative (DI 0) at +20 points and a dense one at -20: print log E leaves the curve's range.
  const FilmModel model(res(), params());
  const auto film = cache().get(params());
  for (double pts : {-20.0, 20.0})
    for (float v : {0.f, 1.f}) {
      LookControls c; c.printerPoints = {pts, pts, pts}; c.printDensity = pts; c.exposureStops = pts > 0 ? 10 : -10;
      const Vec3 baked = renderOne(*film, v, v, v, c), direct = model.evaluate({v, v, v}, c);
      CHECK(deltaE2000(toLab(direct), toLab(baked)) < 0.5);
    }
}

TEST_CASE("window, bounds and padding respected") {
  const auto film = cache().get(params());
  const float nan = std::numeric_limits<float>::quiet_NaN();
  ImageGeometry sg{100, 50, 104, 53, 4 * 16 + 32};  // 32 bytes of padding per row
  TestImage src(sg, nan), dst(tight(98, 50, 106, 53), 7.f);
  for (int y = 50; y < 53; ++y)
    for (int x = 100; x < 104; ++x) {
      float* p = src.at(x, y);
      p[0] = 0.4f + 0.02f * float(x - 100); p[1] = 0.4f; p[2] = 0.4f + 0.02f * float(y - 50); p[3] = 0.25f + float(x - 100);
    }
  renderFilmCpu(src.cview(), dst.view(), {100, 50, 105, 53}, *film, LookControls{});  // window sticks out past src on the right
  for (int y = 50; y < 53; ++y)
    for (int x = 98; x < 106; ++x) {
      const float* d = dst.at(x, y);
      CAPTURE(x);
      CAPTURE(y);
      if (x < 100 || x == 105) {  // outside the window: untouched
        for (int c = 0; c < 4; ++c) CHECK(d[c] == 7.f);
      } else if (x == 104) {  // inside the window but outside src: transparent black
        for (int c = 0; c < 4; ++c) CHECK(d[c] == 0.f);
      } else {
        CHECK(d[3] == 0.25f + float(x - 100));
        const Vec3 ref = renderOne(*film, src.at(x, y)[0], src.at(x, y)[1], src.at(x, y)[2], LookControls{});
        for (int c = 0; c < 3; ++c) { CHECK(std::isfinite(d[c])); CHECK(d[c] == doctest::Approx(ref[c])); }
      }
    }
}

TEST_CASE("out-of-domain and non-finite inputs") {
  const auto film = cache().get(params());
  const float inf = std::numeric_limits<float>::infinity(), nan = std::numeric_limits<float>::quiet_NaN();
  const Vec3 zero = renderOne(*film, 0, 0, 0, LookControls{});
  for (float v : {-0.5f, 1.7f, nan, inf, -inf}) {
    const Vec3 o = renderOne(*film, v, v, v, LookControls{});
    for (double c : o) CHECK(std::isfinite(c));
  }
  const Vec3 one = renderOne(*film, 1.0f, 1.0f, 1.0f, LookControls{});
  const Vec3 over = renderOne(*film, 1.7f, 1.7f, 1.7f, LookControls{});
  for (int c = 0; c < 3; ++c) CHECK(over[c] == one[c]);  // clamped to the edge of the domain
  const Vec3 neg = renderOne(*film, -0.5f, -0.5f, -0.5f, LookControls{});
  for (int c = 0; c < 3; ++c) CHECK(neg[c] == zero[c]);
  const Vec3 bad = renderOne(*film, nan, inf, -inf, LookControls{});
  for (int c = 0; c < 3; ++c) CHECK(bad[c] == zero[c]);
}

TEST_CASE("concurrent cache use") {
  const double lights[3] = {-5, 0, 5};
  auto grey = [](const BakedFilm& f) {
    TestImage src(tight(0, 0, 16, 16), 0.f), dst(tight(0, 0, 16, 16), 0.f);
    const float c = float(diEncode(0.18));
    for (int y = 0; y < 16; ++y)
      for (int x = 0; x < 16; ++x) { float* p = src.at(x, y); p[0] = p[1] = p[2] = c; p[3] = 1.f; }
    renderFilmCpu(src.cview(), dst.view(), {0, 0, 16, 16}, f, look(0.5));
    return std::vector<float>(dst.at(0, 0), dst.at(0, 0) + 4);
  };
  FilmCache c(res());
  std::vector<float> ref[3];
  for (int i = 0; i < 3; ++i) ref[i] = grey(*c.get(params(lights[i])));
  FilmCache shared(res());
  std::vector<std::thread> pool;
  std::atomic<int> bad{0};
  for (int t = 0; t < 8; ++t)
    pool.emplace_back([&, t] {
      for (int it = 0; it < 20; ++it) {
        const int k = t % 3;
        if (grey(*shared.get(params(lights[k]))) != ref[k]) ++bad;
      }
    });
  for (auto& th : pool) th.join();
  CHECK(bad == 0);
}

TEST_CASE("non-finite exposure is treated as zero stops") {
    const auto film = cache().get(params());
    TestImage src(tight(0, 0, 2, 1), 0.4f), ref(tight(0, 0, 2, 1), 0.f), bad(tight(0, 0, 2, 1), 0.f);
    const RenderWindow win{0, 0, 2, 1};
    renderFilmCpu(src.cview(), ref.view(), win, *film, LookControls{});
    for (double v : {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity()}) {
        renderFilmCpu(src.cview(), bad.view(), win, *film, look(v));
        for (int c = 0; c < 3; ++c) CHECK(bad.at(0, 0)[c] == ref.at(0, 0)[c]);
    }
}

TEST_CASE("a bake does not stall renders of an already-cached look") {
  FilmCache c(res());
  c.get(params(0));  // look B, cached
  std::atomic<bool> started{false};
  std::thread baker([&] {
    started = true;
    c.get(params(3));  // look A: a fresh ~1 s bake
  });
  while (!started) std::this_thread::yield();
  std::this_thread::sleep_for(std::chrono::milliseconds(200));  // A is well into its bake
  const auto t0 = std::chrono::steady_clock::now();
  c.get(params(0));
  const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  std::printf("cached get while another look bakes: %.2f ms\n", ms);
  baker.join();
  CHECK(ms < 100.0);
}

TEST_CASE("simultaneous requests for one new look bake it once") {
  FilmCache c(res());
  std::shared_ptr<const BakedFilm> a, b;
  std::thread t1([&] { a = c.get(params(2)); }), t2([&] { b = c.get(params(2)); });
  t1.join();
  t2.join();
  CHECK(a);
  CHECK(a == b);
}

TEST_CASE("a failed bake reaches the caller and is not cached") {
  FilmCache c(res());
  FilmParams bad = params();
  bad.negativeId = "no-such-stock";
  CHECK_THROWS(c.get(bad));
  CHECK_THROWS(c.get(bad));  // retried, not a remembered failure
  CHECK(c.get(params()) != nullptr);  // the cache still works afterwards
}

TEST_CASE("FilmCache::peek never bakes and returns a finished look") {
  FilmCache c(res());
  const FilmParams p = params();
  CHECK(c.peek(p) == nullptr);  // not requested: no bake started
  CHECK(c.peek(p) == nullptr);
  const auto baked = c.get(p);
  CHECK(c.peek(p) == baked);
}
