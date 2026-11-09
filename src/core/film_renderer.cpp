#include "core/film_renderer.hpp"

#include <algorithm>
#include <cmath>
#include <thread>
#include <vector>

namespace openfilm {

namespace {

// Black for anything that is not a real number, so no NaN can reach the lookups.
inline float sane(float v) { return std::isfinite(v) ? v : 0.f; }

// The table holds projected light; the screen's gamma (BT.1886 ~ 1/2.4) and the clip of colours
// the display cannot show (negative light -> black) happen here, per pixel.
inline float encodeDisplay(float linear) { return std::pow(std::max(linear, 0.f), 1.f / 2.4f); }

void renderRows(const ConstImageView& src, const ImageView& dst, const RenderWindow& win, const BakedFilm& film,
                float gain, const float off[3], int yBegin, int yEnd) {
  for (int y = yBegin; y < yEnd; ++y) {
    for (int x = win.x1; x < win.x2; ++x) {
      float* out = pixelAddress(dst, x, y);
      if (!out) continue;  // window pixel the host gave us no room for
      const float* in = pixelAddress(src, x, y);
      if (!in) {
        out[0] = out[1] = out[2] = out[3] = 0.f;
        continue;
      }
      // DI code -> table value -> linear exposure; the Exposure control multiplies that by 2^stops,
      // then log10 gives the log E the negative curves expect. The floor only guards log10(0).
      const auto e = film.lutE.lookup(sane(in[0]), sane(in[1]), sane(in[2]));
      float le[3];
      for (int c = 0; c < 3; ++c) le[c] = std::log10(std::max((std::pow(10.f, e[c]) - float(kExposureLutEps)) * gain, 1e-30f));
      const float d[3] = {film.negCurves[0].eval(le[0]), film.negCurves[1].eval(le[1]),
                          film.negCurves[2].eval(le[2])};
      // Negative density -> exposure on the print; the lab's balance, printer lights and Print
      // Density shift it (a printer point is a fixed step of print exposure); the print's curves give
      // dye density; the projected dyes give the display colour.
      const auto pe = film.lutPE.lookup(d[0], d[1], d[2]);
      const float pd[3] = {film.printCurves[0].eval(pe[0] + off[0]), film.printCurves[1].eval(pe[1] + off[1]),
                           film.printCurves[2].eval(pe[2] + off[2])};
      const auto o = film.lutD.lookup(pd[0], pd[1], pd[2]);
      out[0] = encodeDisplay(o[0]);
      out[1] = encodeDisplay(o[1]);
      out[2] = encodeDisplay(o[2]);
      out[3] = in[3];
    }
  }
}

}  // namespace

float exposureGain(double stops) {
  if (!std::isfinite(stops)) return 1.f;
  return float(std::exp2(std::clamp(stops, -64.0, 64.0)));
}

void renderFilmCpu(const ConstImageView& src, const ImageView& dst, const RenderWindow& win, const BakedFilm& film,
                   const LookControls& controls) {
  const int rows = win.y2 - win.y1;
  if (rows <= 0 || win.x2 <= win.x1) return;
  const float gain = exposureGain(controls.exposureStops);
  const Vec3 o = printOffsetsLogE(film.balance, controls);  // once per render, not per pixel
  const float off[3] = {float(o[0]), float(o[1]), float(o[2])};
  const int threads = std::clamp(int(std::thread::hardware_concurrency()), 1, rows);
  std::vector<std::thread> pool;
  pool.reserve(size_t(threads));
  for (int t = 0; t < threads; ++t)
    pool.emplace_back(renderRows, src, dst, win, std::cref(film), gain, off, win.y1 + rows * t / threads,
                      win.y1 + rows * (t + 1) / threads);
  for (auto& th : pool) th.join();
}

}  // namespace openfilm
