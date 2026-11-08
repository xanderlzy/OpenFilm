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
                float stopsLog10, int yBegin, int yEnd) {
  for (int y = yBegin; y < yEnd; ++y) {
    for (int x = win.x1; x < win.x2; ++x) {
      float* out = pixelAddress(dst, x, y);
      if (!out) continue;  // window pixel the host gave us no room for
      const float* in = pixelAddress(src, x, y);
      if (!in) {
        out[0] = out[1] = out[2] = out[3] = 0.f;
        continue;
      }
      // DI code -> exposure; the Exposure control is a plain shift in log E (stops x log10 2).
      const auto e = film.lutE.lookup(sane(in[0]), sane(in[1]), sane(in[2]));
      const float d[3] = {film.negCurves[0].eval(e[0] + stopsLog10), film.negCurves[1].eval(e[1] + stopsLog10),
                          film.negCurves[2].eval(e[2] + stopsLog10)};
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
  const float stopsLog10 = float(exposureStops * std::log10(2.0));
  const int threads = std::clamp(int(std::thread::hardware_concurrency()), 1, rows);
  std::vector<std::thread> pool;
  pool.reserve(size_t(threads));
  for (int t = 0; t < threads; ++t)
    pool.emplace_back(renderRows, src, dst, win, std::cref(film), stopsLog10, win.y1 + rows * t / threads,
                      win.y1 + rows * (t + 1) / threads);
  for (auto& th : pool) th.join();
}

}  // namespace openfilm
