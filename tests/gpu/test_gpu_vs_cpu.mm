// GPU-vs-CPU harness: renders the same frame through renderFilmCpu() and the Metal film kernel and
// checks they agree (RGB within 1e-3 of display code, alpha bit for bit). Any change to the film
// chain on one side shows up here as a mismatch.
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "core/film_renderer.hpp"
#include "core/lut.hpp"
#include "metal/metal_renderer.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <limits>
#include <random>
#include <thread>
#include <vector>

using namespace openfilm;

// Stands in for Resolve's command queue so the benchmark can read the GPU timestamps of the command
// buffer render() creates. Everything else is forwarded to the real queue.
@interface OFRecordingQueue : NSObject
@property(nonatomic, strong) id<MTLCommandQueue> real;
@property(nonatomic, strong) id<MTLCommandBuffer> last;
@end

@implementation OFRecordingQueue
- (id<MTLCommandBuffer>)commandBuffer {
    self.last = [self.real commandBuffer];
    return self.last;
}
- (id<MTLDevice>)device {
    return self.real.device;
}
- (id)forwardingTargetForSelector:(SEL)sel {
    return self.real;
}
@end

namespace {

constexpr float kSentinel = 7.f;  // dst starts as this; pixels outside the window must keep it

ImageGeometry tight(int x1, int y1, int x2, int y2) {
    return {x1, y1, x2, y2, std::ptrdiff_t(x2 - x1) * 16};
}

size_t byteSize(const ImageGeometry& g) { return size_t(g.rowBytes) * size_t(std::max(g.y2 - g.y1, 0)); }

MetalRenderer& renderer() {
    static MetalRenderer r(OPENFILM_METALLIB_PATH);
    return r;
}

std::shared_ptr<const FilmResources> res() {
    static const auto r = FilmResources::load(OPENFILM_SOURCE_DATA_DIR, OPENFILM_COEFF_PATH);
    return r;
}

FilmParams params(double gamutCompression = 0.5) {
    FilmParams p;
    p.negativeId = "kodak-5219-type";
    p.printId = "kodak-2383-type";
    p.gamutCompression = gamutCompression;
    return p;
}

std::shared_ptr<const BakedFilm> film() {
    static FilmCache cache(res());
    return cache.get(params());
}

id<MTLDevice> device() {
    static id<MTLDevice> d = MTLCreateSystemDefaultDevice();
    return d;
}

using Fill = std::function<void(int x, int y, float* px)>;

struct Frame {
    ImageGeometry sg, dg;
    std::vector<float> src;
    std::vector<float> cpu;
};

// Source filled by `fill` inside its bounds, NaN in any row padding; dst prefilled with kSentinel.
Frame makeFrame(const ImageGeometry& sg, const ImageGeometry& dg, const Fill& fill) {
    Frame f{sg, dg, std::vector<float>(std::max<size_t>(byteSize(sg) / 4, 4), std::numeric_limits<float>::quiet_NaN()),
            std::vector<float>(std::max<size_t>(byteSize(dg) / 4, 4), kSentinel)};
    ConstImageView v{f.src.data(), sg};
    for (int y = sg.y1; y < sg.y2; ++y)
        for (int x = sg.x1; x < sg.x2; ++x) fill(x, y, const_cast<float*>(pixelAddress(v, x, y)));
    return f;
}

struct Result {
    double maxDiff = 0;          // max |GPU - CPU| over the RGB of every dst pixel
    bool alphaExact = true;      // alpha identical bit for bit
    bool outsideUntouched = true;  // GPU left every dst pixel outside the window at kSentinel
};

Result compare(const Frame& f, const float* gpu, const RenderWindow& win) {
    ConstImageView cpuView{f.cpu.data(), f.dg}, gpuView{gpu, f.dg};
    Result r;
    for (int y = f.dg.y1; y < f.dg.y2; ++y)
        for (int x = f.dg.x1; x < f.dg.x2; ++x) {
            const float* a = pixelAddress(cpuView, x, y);
            const float* b = pixelAddress(gpuView, x, y);
            for (int c = 0; c < 3; ++c) {
                const double d = std::isnan(a[c]) || std::isnan(b[c]) ? INFINITY : std::fabs(double(a[c]) - double(b[c]));
                r.maxDiff = std::max(r.maxDiff, d);
            }
            if (std::memcmp(&a[3], &b[3], sizeof(float)) != 0) r.alphaExact = false;
            const bool inWin = x >= win.x1 && x < win.x2 && y >= win.y1 && y < win.y2;
            if (!inWin)
                for (int c = 0; c < 4; ++c)
                    if (b[c] != kSentinel) r.outsideUntouched = false;
        }
    return r;
}

// Encodes one GPU render of `f` on `queue` and waits for it. Returns the dst buffer's contents.
std::vector<float> renderGpu(id<MTLCommandQueue> queue, const Frame& f, const RenderWindow& win,
                             const std::shared_ptr<const BakedFilm>& baked, const LookControls& controls,
                             bool* ok) {
    id<MTLDevice> dev = queue.device;
    id<MTLBuffer> srcBuf = [dev newBufferWithBytes:f.src.data() length:f.src.size() * 4 options:MTLResourceStorageModeShared];
    std::vector<float> fill(f.cpu.size(), kSentinel);
    id<MTLBuffer> dstBuf = [dev newBufferWithBytes:fill.data() length:fill.size() * 4 options:MTLResourceStorageModeShared];
    *ok = renderer().render((__bridge void*)queue, (__bridge void*)srcBuf, f.sg, (__bridge void*)dstBuf, f.dg, win,
                            baked, controls);
    // The queue runs command buffers in order, so waiting on a later one waits for ours too.
    id<MTLCommandBuffer> fence = [queue commandBuffer];
    [fence commit];
    [fence waitUntilCompleted];
    const float* p = static_cast<const float*>(dstBuf.contents);
    return std::vector<float>(p, p + fill.size());
}

// Renders the same source through renderFilmCpu and MetalRenderer::render and compares them.
Result runBoth(const ImageGeometry& sg, const ImageGeometry& dg, const RenderWindow& win, const LookControls& controls,
               const Fill& fill, const std::shared_ptr<const BakedFilm>& baked = film()) {
    REQUIRE(renderer().isValid());
    Frame f = makeFrame(sg, dg, fill);
    renderFilmCpu(ConstImageView{f.src.data(), sg}, ImageView{f.cpu.data(), dg}, win, *baked, controls);
    id<MTLCommandQueue> queue = [device() newCommandQueue];
    bool ok = false;
    const std::vector<float> gpu = renderGpu(queue, f, win, baked, controls, &ok);
    CHECK(ok);
    return compare(f, gpu.data(), win);
}

void checkBoth(const ImageGeometry& sg, const ImageGeometry& dg, const RenderWindow& win, const LookControls& controls,
               const Fill& fill) {
    const Result r = runBoth(sg, dg, win, controls, fill);
    CAPTURE(r.maxDiff);
    MESSAGE("max GPU-vs-CPU difference: " << r.maxDiff);
    CHECK(r.maxDiff <= 1e-3);
    CHECK(r.alphaExact);
    CHECK(r.outsideUntouched);
}

Fill randomFill(unsigned seed, float lo, float hi, float alpha = -1) {
    auto rng = std::make_shared<std::mt19937>(seed);
    return [rng, lo, hi, alpha](int, int, float* p) {
        std::uniform_real_distribution<float> d(lo, hi);
        for (int c = 0; c < 3; ++c) p[c] = d(*rng);
        p[3] = alpha >= 0 ? alpha : d(*rng);
    };
}

}  // namespace

TEST_CASE("grey ramp") {
    const auto g = tight(0, 0, 256, 16);
    checkBoth(g, g, {0, 0, 256, 16}, LookControls{}, [](int x, int, float* p) {
        p[0] = p[1] = p[2] = float(x) / 255.f;
        p[3] = 1.f;
    });
}

TEST_CASE("saturated primaries and DWG edges") {
    const auto g = tight(0, 0, 512, 512);
    checkBoth(g, g, {0, 0, 512, 512}, LookControls{}, randomFill(7, 0.f, 1.f));
}

TEST_CASE("exposure +-8 and points +-20") {
    const auto g = tight(0, 0, 256, 256);
    LookControls up;
    up.exposureStops = 8;
    up.printerPoints = {20, -20, 20};
    up.printDensity = -20;
    LookControls down;
    down.exposureStops = -8;
    down.printerPoints = {-20, 20, -20};
    down.printDensity = 20;
    checkBoth(g, g, {0, 0, 256, 256}, up, randomFill(11, 0.f, 1.f));
    checkBoth(g, g, {0, 0, 256, 256}, down, randomFill(11, 0.f, 1.f));
}

TEST_CASE("non-finite and negative pixels") {
    const float nan = std::numeric_limits<float>::quiet_NaN(), inf = std::numeric_limits<float>::infinity();
    const float values[] = {nan, inf, -inf, -1.f, 2.f, 0.5f};
    const auto g = tight(0, 0, 6, 36);
    checkBoth(g, g, {0, 0, 6, 36}, LookControls{}, [&](int x, int y, float* p) {
        p[0] = values[x];
        p[1] = values[y % 6];
        p[2] = values[(y / 6) % 6];
        p[3] = 0.25f;
    });
}

TEST_CASE("offset bounds, padded rows, sub-window") {
    const ImageGeometry sg{10, 20, 330, 260, 320 * 16 + 64};
    const ImageGeometry dg = tight(10, 20, 330, 260);
    checkBoth(sg, dg, {50, 40, 300, 250}, LookControls{}, randomFill(3, 0.f, 1.f));
}

TEST_CASE("odd sizes") {
    for (auto wh : {std::pair<int, int>{1, 1}, {3, 7}, {1921, 1081}}) {
        CAPTURE(wh.first);
        CAPTURE(wh.second);
        const auto g = tight(0, 0, wh.first, wh.second);
        checkBoth(g, g, {0, 0, wh.first, wh.second}, LookControls{}, randomFill(5, -0.1f, 1.2f));
    }
}

TEST_CASE("extreme controls on the GPU") {
    const auto g = tight(0, 0, 128, 128);
    for (double pts : {50.0, -50.0}) {
        LookControls c;
        c.printerPoints = {pts, -pts, pts};
        c.printDensity = pts;
        c.exposureStops = std::numeric_limits<double>::quiet_NaN();
        CAPTURE(pts);
        checkBoth(g, g, {0, 0, 128, 128}, c, randomFill(13, 0.f, 1.f));
    }
}

TEST_CASE("degenerate windows") {
    const auto g = tight(0, 0, 64, 32);
    SUBCASE("empty window") { checkBoth(g, g, {20, 0, 20, 32}, LookControls{}, randomFill(17, 0.f, 1.f)); }
    SUBCASE("window entirely outside src bounds") {
        const auto sg = tight(100, 100, 164, 132);  // dst is at the origin, src far away
        checkBoth(sg, g, {0, 0, 64, 32}, LookControls{}, randomFill(17, 0.f, 1.f));
    }
    SUBCASE("1-pixel-wide image") {
        const auto one = tight(0, 0, 1, 50);
        checkBoth(one, one, {0, 0, 1, 50}, LookControls{}, randomFill(17, 0.f, 1.f));
    }
}

TEST_CASE("concurrent renders share one upload") {
    REQUIRE(renderer().isValid());
    const auto fresh = bakeFilm(FilmModel(res(), params(0.71)));
    const size_t before = renderer().cachedFilmCount();
    const auto g = tight(0, 0, 320, 180);
    const RenderWindow win{0, 0, 320, 180};
    Frame f = makeFrame(g, g, randomFill(19, 0.f, 1.f));
    renderFilmCpu(ConstImageView{f.src.data(), g}, ImageView{f.cpu.data(), g}, win, *fresh, LookControls{});

    std::vector<Result> results(4);
    std::vector<int> ok(4, 0);
    std::vector<std::thread> threads;
    for (int t = 0; t < 4; ++t)
        threads.emplace_back([&, t] {
            @autoreleasepool {
                id<MTLCommandQueue> queue = [device() newCommandQueue];
                bool rendered = false;
                const std::vector<float> gpu = renderGpu(queue, f, win, fresh, LookControls{}, &rendered);
                ok[size_t(t)] = rendered;
                results[size_t(t)] = compare(f, gpu.data(), win);
            }
        });
    for (auto& th : threads) th.join();
    for (int t = 0; t < 4; ++t) {
        CAPTURE(t);
        CHECK(ok[size_t(t)]);
        CHECK(results[size_t(t)].maxDiff <= 1e-3);
        CHECK(results[size_t(t)].alphaExact);
    }
    CHECK(renderer().cachedFilmCount() == before + 1);
}

TEST_CASE("gpu tables are released with their bake") {
    REQUIRE(renderer().isValid());
    const auto g = tight(0, 0, 8, 8);
    const auto anyFill = randomFill(23, 0.f, 1.f);
    runBoth(g, g, {0, 0, 8, 8}, LookControls{}, anyFill);  // sweeps entries left over by earlier tests
    const size_t before = renderer().cachedFilmCount();
    {
        auto doomed = bakeFilm(FilmModel(res(), params(0.83)));
        runBoth(g, g, {0, 0, 8, 8}, LookControls{}, anyFill, doomed);
        CHECK(renderer().cachedFilmCount() == before + 1);
    }  // the only reference to the bake goes here
    const auto other = bakeFilm(FilmModel(res(), params(0.29)));
    const Result r = runBoth(g, g, {0, 0, 8, 8}, LookControls{}, anyFill, other);
    CHECK(r.maxDiff <= 1e-3);
    CHECK(renderer().cachedFilmCount() == before + 1);  // `other` in, the dropped bake out
}

TEST_CASE("invalid metallib path") {
    MetalRenderer bad("/nonexistent.metallib");
    CHECK_FALSE(bad.isValid());
    CHECK_FALSE(bad.error().empty());
}

TEST_CASE("benchmark: UHD under 5 ms") {
    REQUIRE(renderer().isValid());
    const auto g = tight(0, 0, 3840, 2160);
    // A smooth frame with every hue and level in it, like real footage (neighbouring pixels hit
    // neighbouring table cells).
    Frame f = makeFrame(g, g, [](int x, int y, float* p) {
        const float u = float(x) / 3839.f, v = float(y) / 2159.f;
        p[0] = u;
        p[1] = v;
        p[2] = 0.5f + 0.5f * std::sin(6.2831853f * (u + v));
        p[3] = 1.f;
    });
    id<MTLDevice> dev = device();
    OFRecordingQueue* queue = [OFRecordingQueue new];
    queue.real = [dev newCommandQueue];
    id<MTLBuffer> srcBuf = [dev newBufferWithBytes:f.src.data() length:f.src.size() * 4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> dstBuf = [dev newBufferWithLength:f.cpu.size() * 4 options:MTLResourceStorageModeShared];
    const auto baked = film();
    LookControls c;
    c.exposureStops = 0.5;
    c.printerPoints = {2, -1, 0};

    std::vector<double> times;
    for (int i = 0; i < 51; ++i) {  // frame 0 warms up (uploads the tables)
        REQUIRE(renderer().render((__bridge void*)queue, (__bridge void*)srcBuf, g, (__bridge void*)dstBuf, g,
                                  {0, 0, 3840, 2160}, baked, c));
        id<MTLCommandBuffer> cmd = queue.last;
        [cmd waitUntilCompleted];
        if (i > 0) times.push_back(cmd.GPUEndTime - cmd.GPUStartTime);
    }
    std::sort(times.begin(), times.end());
    const double median = times[times.size() / 2];
    std::printf("UHD GPU median: %.3f ms (min %.3f, max %.3f)\n", median * 1e3, times.front() * 1e3,
                times.back() * 1e3);
    CHECK(median <= 5e-3);
}
