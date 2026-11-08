// GPU-vs-CPU harness: runs the Metal path and the CPU reference on the same input and checks they
// agree. Any Metal kernel added later gets a case here.
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#import <Metal/Metal.h>

#include "core/cpu_renderer.hpp"
#include "metal/metal_renderer.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <vector>

using namespace openfilm;

namespace {

ImageGeometry tight(int x1, int y1, int x2, int y2) {
    return {x1, y1, x2, y2, std::ptrdiff_t(x2 - x1) * 16};
}

size_t byteSize(const ImageGeometry& g) { return size_t(g.rowBytes) * size_t(g.y2 - g.y1); }

MetalRenderer& renderer() {
    static MetalRenderer r(OPENFILM_METALLIB_PATH);
    return r;
}

struct Result {
    double maxDiff = 0;
    double maxAbs = 0;
};

// Runs both paths and returns the max abs difference over every dst pixel.
Result runBoth(const ImageGeometry& sg, const ImageGeometry& dg, const RenderWindow& win, double stops) {
    REQUIRE(renderer().isValid());

    // Source: deterministic pattern in [-1, 100]; NaN in row padding.
    std::vector<float> src(byteSize(sg) / 4, std::numeric_limits<float>::quiet_NaN());
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.f, 100.f);
    ConstImageView srcView{src.data(), sg};
    for (int y = sg.y1; y < sg.y2; ++y)
        for (int x = sg.x1; x < sg.x2; ++x) {
            float* p = const_cast<float*>(pixelAddress(srcView, x, y));
            for (int c = 0; c < 4; ++c) p[c] = dist(rng);
        }

    // CPU reference.
    std::vector<float> cpuDst(byteSize(dg) / 4, 7.f);
    RenderParams params{stops};
    renderExposureCpu(srcView, ImageView{cpuDst.data(), dg}, win, params);

    // GPU.
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    const bool haveDevice = device != nil;
    REQUIRE(haveDevice);
    id<MTLCommandQueue> queue = [device newCommandQueue];
    id<MTLBuffer> srcBuf = [device newBufferWithBytes:src.data() length:byteSize(sg) options:MTLResourceStorageModeShared];
    std::vector<float> sevens(byteSize(dg) / 4, 7.f);
    id<MTLBuffer> dstBuf = [device newBufferWithBytes:sevens.data() length:byteSize(dg) options:MTLResourceStorageModeShared];

    REQUIRE(renderer().render((__bridge void*)queue, (__bridge void*)srcBuf, sg, (__bridge void*)dstBuf, dg, win, params));
    // The queue runs command buffers in order, so waiting on a later one waits for ours too.
    id<MTLCommandBuffer> fence = [queue commandBuffer];
    [fence commit];
    [fence waitUntilCompleted];

    ConstImageView cpuView{cpuDst.data(), dg};
    ConstImageView gpuView{static_cast<const float*>(dstBuf.contents), dg};
    Result r;
    for (int y = dg.y1; y < dg.y2; ++y)
        for (int x = dg.x1; x < dg.x2; ++x) {
            const float* a = pixelAddress(cpuView, x, y);
            const float* b = pixelAddress(gpuView, x, y);
            for (int c = 0; c < 4; ++c) {
                double d = std::isnan(a[c]) || std::isnan(b[c]) ? INFINITY : std::fabs(double(a[c]) - double(b[c]));
                r.maxDiff = std::max(r.maxDiff, d);
                r.maxAbs = std::max(r.maxAbs, std::fabs(double(a[c])));
            }
        }
    return r;
}

void checkMatch(const ImageGeometry& sg, const ImageGeometry& dg, const RenderWindow& win, double stops) {
    Result r = runBoth(sg, dg, win, stops);
    CAPTURE(r.maxDiff);
    CAPTURE(r.maxAbs);
    CHECK(r.maxDiff <= 1e-5 * std::max(1.0, r.maxAbs));
}

}  // namespace

TEST_CASE("gpu matches cpu: 1920x1080, +1.5 stops") {
    auto g = tight(0, 0, 1920, 1080);
    checkMatch(g, g, {0, 0, 1920, 1080}, 1.5);
}

TEST_CASE("gpu matches cpu: odd sizes") {
    auto one = tight(0, 0, 1, 1);
    checkMatch(one, one, {0, 0, 1, 1}, -0.7);
    auto odd = tight(0, 0, 1921, 1081);
    checkMatch(odd, odd, {0, 0, 1921, 1081}, 2.0);
}

TEST_CASE("gpu matches cpu: offset bounds, padded rows, sub-window") {
    ImageGeometry sg{10, 20, 330, 260, 320 * 16 + 64};
    ImageGeometry dg = tight(10, 20, 330, 260);
    checkMatch(sg, dg, {50, 40, 300, 250}, 1.0);
}

TEST_CASE("invalid metallib path") {
    MetalRenderer bad("/nonexistent.metallib");
    CHECK_FALSE(bad.isValid());
    CHECK_FALSE(bad.error().empty());
}
