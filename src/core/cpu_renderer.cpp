#include "core/cpu_renderer.hpp"

#include <algorithm>
#include <cmath>
#include <thread>
#include <vector>

namespace openfilm {

namespace {

void renderRows(const ConstImageView& src, const ImageView& dst, const RenderWindow& win, float gain, int yBegin,
                int yEnd) {
    for (int y = yBegin; y < yEnd; ++y) {
        for (int x = win.x1; x < win.x2; ++x) {
            float* out = pixelAddress(dst, x, y);
            if (!out) continue;  // window pixel the host gave us no room for
            const float* in = pixelAddress(src, x, y);
            if (!in) {
                out[0] = out[1] = out[2] = out[3] = 0.f;
                continue;
            }
            out[0] = in[0] * gain;
            out[1] = in[1] * gain;
            out[2] = in[2] * gain;
            out[3] = in[3];  // alpha is never touched by the film model
        }
    }
}

}  // namespace

void renderExposureCpu(const ConstImageView& src, const ImageView& dst, const RenderWindow& win, const RenderParams& p) {
    const int rows = win.y2 - win.y1;
    if (rows <= 0 || win.x2 <= win.x1) return;

    // One stop = a factor of two in light.
    const float gain = float(std::exp2(p.exposureStops));

    // Split the rows into one band per CPU core.
    const int threads = std::clamp(int(std::thread::hardware_concurrency()), 1, rows);
    std::vector<std::thread> pool;
    pool.reserve(size_t(threads));
    for (int t = 0; t < threads; ++t) {
        const int y0 = win.y1 + rows * t / threads;
        const int y1 = win.y1 + rows * (t + 1) / threads;
        pool.emplace_back(renderRows, src, dst, win, gain, y0, y1);
    }
    for (auto& th : pool) th.join();
}

}  // namespace openfilm
