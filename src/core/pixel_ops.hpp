// Small helpers shared by the CPU renderers and the blur engine (internal; not part of the API).
#pragma once

#include <algorithm>
#include <cmath>
#include <functional>
#include <thread>
#include <vector>

namespace openfilm::detail {

// Black for anything that is not a real number, so no NaN can reach the lookups.
inline float sane(float v) { return std::isfinite(v) ? v : 0.f; }

// The table holds the log of the projected light; the matrix has turned it into linear Rec.709; the screen's gamma (BT.1886 ~ 1/2.4) and the clip of colours
// the display cannot show (negative light -> black) happen here, per pixel.
inline float encodeDisplay(float linear) { return std::pow(std::max(linear, 0.f), 1.f / 2.4f); }

// Runs fn(yBegin, yEnd) over [0, rows) on a pool sized from the hardware.
inline void parallelRows(int rows, const std::function<void(int, int)>& fn) {
  if (rows <= 0) return;
  const int threads = std::clamp(int(std::thread::hardware_concurrency()), 1, rows);
  if (threads == 1) {
    fn(0, rows);
    return;
  }
  std::vector<std::thread> pool;
  pool.reserve(size_t(threads));
  for (int t = 0; t < threads; ++t) pool.emplace_back(fn, rows * t / threads, rows * (t + 1) / threads);
  for (auto& th : pool) th.join();
}

}  // namespace openfilm::detail
