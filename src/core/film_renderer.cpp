#include "core/film_renderer.hpp"

#include <algorithm>
#include <cmath>
#include <thread>
#include <vector>

namespace openfilm {

namespace {

// Black for anything that is not a real number, so no NaN can reach the lookups.
inline float sane(float v) { return std::isfinite(v) ? v : 0.f; }

void renderRows(const ConstImageView& src, const ImageView& dst, const RenderWindow& win, const BakedFilm& film,
                float gain, int yBegin, int yEnd) {
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
      const auto o = film.lutP.lookup(d[0], d[1], d[2]);
      out[0] = o[0];
      out[1] = o[1];
      out[2] = o[2];
      out[3] = in[3];
    }
  }
}

}  // namespace

void renderFilmCpu(const ConstImageView& src, const ImageView& dst, const RenderWindow& win, const BakedFilm& film,
                   double exposureStops) {
  const int rows = win.y2 - win.y1;
  if (rows <= 0 || win.x2 <= win.x1) return;
  // A host can hand us NaN/inf for a keyframed value; treat it as 'no exposure change'.
  if (!std::isfinite(exposureStops)) exposureStops = 0.0;
  // Far beyond any real control range; keeps the gain finite so 0 x gain can never be NaN.
  const float gain = float(std::exp2(std::clamp(exposureStops, -64.0, 64.0)));
  const int threads = std::clamp(int(std::thread::hardware_concurrency()), 1, rows);
  std::vector<std::thread> pool;
  pool.reserve(size_t(threads));
  for (int t = 0; t < threads; ++t)
    pool.emplace_back(renderRows, src, dst, win, std::cref(film), gain, win.y1 + rows * t / threads,
                      win.y1 + rows * (t + 1) / threads);
  for (auto& th : pool) th.join();
}

}  // namespace openfilm
