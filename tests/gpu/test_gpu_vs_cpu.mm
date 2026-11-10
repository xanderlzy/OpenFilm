// GPU-vs-CPU harness: renders the same frame through renderFilmCpu() and the Metal film kernel and
// checks they agree (RGB within 1e-3 of display code, alpha bit for bit). Any change to the film
// chain on one side shows up here as a mismatch.
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "core/film_renderer.hpp"
#include "core/film_spatial.hpp"
#include "core/lut.hpp"
#include "metal/metal_renderer.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
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
                             bool* ok, const SpatialControls* spatial = nullptr) {
    id<MTLDevice> dev = queue.device;
    id<MTLBuffer> srcBuf = [dev newBufferWithBytes:f.src.data() length:f.src.size() * 4 options:MTLResourceStorageModeShared];
    std::vector<float> fill(f.cpu.size(), kSentinel);
    id<MTLBuffer> dstBuf = [dev newBufferWithBytes:fill.data() length:fill.size() * 4 options:MTLResourceStorageModeShared];
    *ok = spatial ? renderer().render((__bridge void*)queue, (__bridge void*)srcBuf, f.sg, (__bridge void*)dstBuf, f.dg,
                                      win, baked, controls, *spatial)
                  : renderer().render((__bridge void*)queue, (__bridge void*)srcBuf, f.sg, (__bridge void*)dstBuf, f.dg,
                                      win, baked, controls);
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


// ---- Spatial (M4) helpers ----

SpatialControls spatialOff() {
    SpatialControls s;
    s.sharpness = s.halation = s.bloom = false;
    return s;
}

SpatialControls spatialAllOn(double pxPerMm) {
    SpatialControls s;
    s.sharpness = s.halation = s.bloom = true;
    s.pxPerMm = pxPerMm;
    return s;
}

// Every effect on, then each alone (halation with the rem-jet both on and off).
std::vector<std::pair<const char*, SpatialControls>> spatialCases(double pxPerMm) {
    std::vector<std::pair<const char*, SpatialControls>> v;
    v.push_back({"all on", spatialAllOn(pxPerMm)});
    SpatialControls s = spatialOff();
    s.pxPerMm = pxPerMm;
    s.sharpness = true;
    v.push_back({"sharpness", s});
    s = spatialOff();
    s.pxPerMm = pxPerMm;
    s.halation = true;
    v.push_back({"halation, rem-jet on", s});
    s.remjet = false;
    s.halationStrength = 2.5;
    s.halationRadius = 1.7;
    v.push_back({"halation, rem-jet off", s});
    s = spatialOff();
    s.pxPerMm = pxPerMm;
    s.bloom = true;
    s.bloomAmount = 0.6;
    s.bloomRadiusMm = 0.8;
    v.push_back({"bloom", s});
    return v;
}

// Point lights on a dark noisy field with a hard vertical edge: what halation, bloom and the edge
// effect act on.
Fill pointsEdgesNoise(unsigned seed, int width, int height) {
    auto rng = std::make_shared<std::mt19937>(seed);
    return [rng, width, height](int x, int y, float* p) {
        std::uniform_real_distribution<float> n(0.f, 0.08f);
        const bool bright = x >= width / 2;
        for (int c = 0; c < 3; ++c) p[c] = (bright ? 0.62f : 0.12f) + n(*rng);
        const int cx = x % 37, cy = y % 29;
        if (cx == 11 && cy == 7) p[0] = p[1] = p[2] = 1.4f;  // super highlights
        if (cx == 23 && cy == 19) {
            p[0] = 1.2f;
            p[1] = 0.9f;
            p[2] = 0.2f;
        }
        if ((x * 7 + y * 3) % 101 == 0) p[(x + y) % 3] = 1.0f;
        (void)height;
        p[3] = float((x * 31 + y * 17) % 1000) / 999.f;
    };
}

// Renders the same source through renderFilm (CPU) and the spatial MetalRenderer::render overload.
Result runSpatial(const ImageGeometry& sg, const ImageGeometry& dg, const RenderWindow& win, const LookControls& look,
                  const SpatialControls& spatial, const Fill& fill, std::vector<float>* gpuOut = nullptr) {
    REQUIRE(renderer().isValid());
    const auto baked = film();
    Frame f = makeFrame(sg, dg, fill);
    renderFilm(ConstImageView{f.src.data(), sg}, ImageView{f.cpu.data(), dg}, win, *baked, look, spatial);
    id<MTLCommandQueue> queue = [device() newCommandQueue];
    bool ok = false;
    std::vector<float> gpu = renderGpu(queue, f, win, baked, look, &ok, &spatial);
    CHECK(ok);
    const Result r = compare(f, gpu.data(), win);
    if (gpuOut) *gpuOut = std::move(gpu);
    return r;
}

void checkSpatial(const ImageGeometry& sg, const ImageGeometry& dg, const RenderWindow& win, const LookControls& look,
                  const SpatialControls& spatial, const Fill& fill) {
    const Result r = runSpatial(sg, dg, win, look, spatial, fill);
    CAPTURE(r.maxDiff);
    MESSAGE("max GPU-vs-CPU difference: " << r.maxDiff);
    CHECK(r.maxDiff <= 1e-3);
    CHECK(r.alphaExact);
    CHECK(r.outsideUntouched);
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

// ---- M4: spatial passes ----

TEST_CASE("spatial: points, edges, noise") {
    const int w = 384, h = 216;
    const auto g = tight(0, 0, w, h);
    // Two scales: a small frame at its own scale (kernels of a few px, mostly full resolution) and
    // the same frame at UHD Super 35 scale (wide lobes at reduced resolution, up to f = 32).
    for (double pxPerMm : {w / 24.89, 3840 / 24.89}) {
        for (const auto& c : spatialCases(pxPerMm)) {
            CAPTURE(pxPerMm);
            CAPTURE(c.first);
            checkSpatial(g, g, {0, 0, w, h}, LookControls{}, c.second, pointsEdgesNoise(29, w, h));
        }
    }
}

// The GPU picks how each full-resolution lobe runs (its horizontal pass inside the fused pass, a
// separate horizontal pass, or summed by separate passes before the fused one). The default picks
// leave some of these unused at a given scale; forcing each way at both scales covers them all.
TEST_CASE("spatial: every blur path") {
    const int w = 384, h = 216;
    const auto g = tight(0, 0, w, h);
    for (int limit : {-1, 1000}) {  // nothing computed in the fused pass / everything
        CAPTURE(limit);
        renderer().setDirectBlurLimit(limit);
        for (double pxPerMm : {w / 24.89, 3840 / 24.89}) {
            CAPTURE(pxPerMm);
            checkSpatial(g, g, {0, 0, w, h}, LookControls{}, spatialAllOn(pxPerMm), pointsEdgesNoise(31, w, h));
        }
        const auto sg = tight(40, 30, 200, 130), dg = tight(0, 0, 240, 160);  // window leaves the source
        checkSpatial(sg, dg, {20, 10, 230, 150}, LookControls{}, spatialAllOn(3840 / 24.89),
                     pointsEdgesNoise(41, 240, 160));
    }
    renderer().setDirectBlurLimit(24);
}

TEST_CASE("spatial: odd sizes, padded rows, sub-windows") {
    const SpatialControls on = spatialAllOn(3840 / 24.89);
    const SpatialControls small = spatialAllOn(1000 / 24.89);
    for (auto wh : {std::pair<int, int>{1, 1}, {3, 97}, {1001, 17}}) {
        CAPTURE(wh.first);
        CAPTURE(wh.second);
        const auto g = tight(0, 0, wh.first, wh.second);
        checkSpatial(g, g, {0, 0, wh.first, wh.second}, LookControls{}, on, pointsEdgesNoise(31, wh.first, wh.second));
        checkSpatial(g, g, {0, 0, wh.first, wh.second}, LookControls{}, small, randomFill(5, -0.1f, 1.2f));
    }
    SUBCASE("padded rows, offset bounds, window inside the image") {
        const ImageGeometry sg{10, 20, 650, 380, 640 * 16 + 80};
        const ImageGeometry dg{10, 20, 650, 380, 640 * 16 + 48};
        LookControls look;
        look.exposureStops = 1.5;
        look.printerPoints = {3, -2, 1};
        for (const RenderWindow win : {RenderWindow{150, 90, 420, 300}, RenderWindow{10, 20, 77, 41},
                                       RenderWindow{600, 330, 650, 380}}) {
            CAPTURE(win.x1);
            CAPTURE(win.y1);
            checkSpatial(sg, dg, win, look, on, pointsEdgesNoise(37, 640, 360));
            checkSpatial(sg, dg, win, look, small, pointsEdgesNoise(37, 640, 360));
        }
    }
    SUBCASE("window partly outside the source") {
        const auto sg = tight(40, 30, 200, 130);
        const auto dg = tight(0, 0, 240, 160);
        checkSpatial(sg, dg, {20, 10, 230, 150}, LookControls{}, on, pointsEdgesNoise(41, 240, 160));
    }
    SUBCASE("window entirely outside the source") {
        const auto sg = tight(300, 300, 364, 332);
        const auto dg = tight(0, 0, 64, 32);
        checkSpatial(sg, dg, {0, 0, 64, 32}, LookControls{}, on, randomFill(43, 0.f, 1.f));
    }
    SUBCASE("non-finite and negative pixels") {
        const float nan = std::numeric_limits<float>::quiet_NaN(), inf = std::numeric_limits<float>::infinity();
        const float values[] = {nan, inf, -inf, -1.f, 2.f, 0.5f};
        const auto g = tight(0, 0, 6, 36);
        checkSpatial(g, g, {0, 0, 6, 36}, LookControls{}, small, [&](int x, int y, float* p) {
            p[0] = values[x];
            p[1] = values[y % 6];
            p[2] = values[(y / 6) % 6];
            p[3] = 0.25f;
        });
    }
}

TEST_CASE("spatial off runs the M3 kernel") {
    REQUIRE(renderer().isValid());
    const ImageGeometry sg{0, 0, 300, 170, 300 * 16 + 32};
    const auto dg = tight(0, 0, 300, 170);
    const RenderWindow win{13, 7, 290, 160};
    Frame f = makeFrame(sg, dg, randomFill(47, -0.1f, 1.2f));
    id<MTLCommandQueue> queue = [device() newCommandQueue];
    LookControls look;
    look.exposureStops = -1.25;
    bool ok1 = false, ok2 = false;
    const std::vector<float> m3 = renderGpu(queue, f, win, film(), look, &ok1);
    SpatialControls off = spatialOff();
    off.pxPerMm = 3840 / 24.89;
    off.bloomAmount = 0.9;  // ignored when everything is off
    const std::vector<float> m4 = renderGpu(queue, f, win, film(), look, &ok2, &off);
    CHECK(ok1);
    CHECK(ok2);
    REQUIRE(m3.size() == m4.size());
    CHECK(std::memcmp(m3.data(), m4.data(), m3.size() * sizeof(float)) == 0);
}

TEST_CASE("spatial: extreme highlights finite") {
    const int w = 192, h = 108;
    const auto g = tight(0, 0, w, h);
    SpatialControls s = spatialAllOn(3840 / 24.89);
    s.remjet = false;
    s.halationStrength = 5;
    s.halationRadius = 4;
    s.bloomAmount = 1;
    s.bloomRadiusMm = 5;
    LookControls look;
    look.exposureStops = 10;
    const Fill fill = [](int x, int, float* p) {
        const float v = x < 64 ? 0.f : 1.f;
        p[0] = p[1] = p[2] = v;
        p[3] = 1.f;
    };
    std::vector<float> gpu;
    const Result r = runSpatial(g, g, {0, 0, w, h}, look, s, fill, &gpu);
    CAPTURE(r.maxDiff);
    CHECK(r.maxDiff <= 1e-3);
    CHECK(r.alphaExact);
    bool finite = true;
    for (float v : gpu) finite = finite && std::isfinite(v);
    CHECK(finite);
    // NaN controls are sanitised on both sides.
    SpatialControls bad = spatialAllOn(std::numeric_limits<double>::quiet_NaN());
    bad.halationStrength = bad.bloomAmount = bad.bloomRadiusMm = bad.halationRadius =
        std::numeric_limits<double>::quiet_NaN();
    checkSpatial(g, g, {0, 0, w, h}, LookControls{}, bad, pointsEdgesNoise(53, w, h));
}

TEST_CASE("spatial: concurrent renders of different sizes") {
    REQUIRE(renderer().isValid());
    const std::pair<int, int> sizes[8] = {{64, 64}, {1920, 1080}, {320, 180}, {1280, 720},
                                          {97, 61}, {640, 360},   {1024, 576}, {200, 113}};
    const auto baked = film();
    std::vector<Frame> frames;
    std::vector<SpatialControls> controls;
    for (int t = 0; t < 8; ++t) {
        const int w = sizes[t].first, h = sizes[t].second;
        const auto g = tight(0, 0, w, h);
        frames.push_back(makeFrame(g, g, pointsEdgesNoise(60 + unsigned(t), w, h)));
        controls.push_back(spatialAllOn(w / 24.89));
        Frame& f = frames.back();
        renderFilm(ConstImageView{f.src.data(), g}, ImageView{f.cpu.data(), g}, {0, 0, w, h}, *baked, LookControls{},
                   controls.back());
    }
    std::vector<Result> results(8);
    std::vector<int> ok(8, 0);
    id<MTLCommandQueue> shared = [device() newCommandQueue];
    std::vector<std::thread> threads;
    for (int t = 0; t < 8; ++t)
        threads.emplace_back([&, t] {
            @autoreleasepool {
                // Half the threads share one queue (as Resolve's render threads may), half have their own.
                id<MTLCommandQueue> queue = t % 2 ? shared : [device() newCommandQueue];
                const Frame& f = frames[size_t(t)];
                const RenderWindow win{0, 0, sizes[t].first, sizes[t].second};
                for (int rep = 0; rep < 2; ++rep) {
                    bool rendered = false;
                    const std::vector<float> gpu =
                        renderGpu(queue, f, win, baked, LookControls{}, &rendered, &controls[size_t(t)]);
                    const Result r = compare(f, gpu.data(), win);
                    ok[size_t(t)] += rendered;
                    // Combine the reps so no rep's failure is masked: worst diff, every check held.
                    Result& acc = results[size_t(t)];
                    if (rep == 0) acc = r;
                    else {
                        acc.maxDiff = std::max(acc.maxDiff, r.maxDiff);
                        acc.alphaExact = acc.alphaExact && r.alphaExact;
                        acc.outsideUntouched = acc.outsideUntouched && r.outsideUntouched;
                    }
                }
            }
        });
    for (auto& th : threads) th.join();
    for (int t = 0; t < 8; ++t) {
        CAPTURE(sizes[t].first);
        CAPTURE(sizes[t].second);
        MESSAGE(sizes[t].first << "x" << sizes[t].second << " max GPU-vs-CPU difference: " << results[size_t(t)].maxDiff);
        CHECK(ok[size_t(t)] == 2);
        CHECK(results[size_t(t)].maxDiff <= 1e-3);
        CHECK(results[size_t(t)].alphaExact);
        CHECK(results[size_t(t)].outsideUntouched);
    }
}

TEST_CASE("spatial: the work pool keeps at most three idle sets, one once they expire") {
    // Runs after the concurrent test above, which had up to 3 sets out at once (the cap). Every
    // completion handler must have returned its set before we look.
    const auto settle = [] {
        for (int i = 0; i < 200 && renderer().poolStats().inFlight > 0; ++i)
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
    };
    settle();
    auto st = renderer().poolStats();
    CHECK(st.inFlight == 0);
    CHECK(st.idleMaxPerDevice <= 3);
    // A set left idle (volatile) is taken again and still renders correctly.
    const auto g = tight(0, 0, 160, 90);
    checkSpatial(g, g, {0, 0, 160, 90}, LookControls{}, spatialAllOn(160 / 24.89), pointsEdgesNoise(73, 160, 90));
    // Idle sets beyond the most recent one are released once idle a while (checked on take/give).
    std::this_thread::sleep_for(std::chrono::milliseconds(1100));
    checkSpatial(g, g, {0, 0, 160, 90}, LookControls{}, spatialAllOn(160 / 24.89), pointsEdgesNoise(74, 160, 90));
    settle();
    st = renderer().poolStats();
    CHECK(st.inFlight == 0);
    CHECK(st.idleMaxPerDevice <= 1);
}

// Renders issued back to back without waiting (a host queueing frames ahead) must not each get
// their own work set: at most three are out per device, and a render over the cap waits for one to
// come back and reuses it. Every frame must still match its CPU render.
TEST_CASE("spatial: at most three work sets in flight") {
    REQUIRE(renderer().isValid());
    const int w = 1280, h = 720, n = 12;
    const auto g = tight(0, 0, w, h);
    const auto baked = film();
    const SpatialControls s = spatialAllOn(3840 / 24.89);  // UHD-sized kernels on a smaller frame
    const RenderWindow win{0, 0, w, h};
    id<MTLDevice> dev = device();
    id<MTLCommandQueue> queue = [dev newCommandQueue];
    std::vector<Frame> frames;
    std::vector<id<MTLBuffer>> dsts;
    std::vector<id<MTLBuffer>> srcs;
    for (int i = 0; i < n; ++i) {
        frames.push_back(makeFrame(g, g, pointsEdgesNoise(80 + unsigned(i), w, h)));
        Frame& f = frames.back();
        renderFilm(ConstImageView{f.src.data(), g}, ImageView{f.cpu.data(), g}, win, *baked, LookControls{}, s);
        srcs.push_back([dev newBufferWithBytes:f.src.data() length:f.src.size() * 4 options:MTLResourceStorageModeShared]);
        std::vector<float> fill(f.cpu.size(), kSentinel);
        dsts.push_back([dev newBufferWithBytes:fill.data() length:fill.size() * 4 options:MTLResourceStorageModeShared]);
    }
    for (int i = 0; i < 200 && renderer().poolStats().inFlight > 0; ++i)
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    renderer().resetPoolPeak();
    for (int i = 0; i < n; ++i)
        CHECK(renderer().render((__bridge void*)queue, (__bridge void*)srcs[size_t(i)], g,
                                (__bridge void*)dsts[size_t(i)], g, win, baked, LookControls{}, s));
    id<MTLCommandBuffer> fence = [queue commandBuffer];
    [fence commit];
    [fence waitUntilCompleted];
    const size_t peak = renderer().poolStats().peakInFlight;
    MESSAGE("peak work sets in flight: " << peak);
    CHECK(peak <= 3);
    double worst = 0;
    bool alphaExact = true, outside = true;
    for (int i = 0; i < n; ++i) {
        const Result r = compare(frames[size_t(i)], static_cast<const float*>(dsts[size_t(i)].contents), win);
        worst = std::max(worst, r.maxDiff);
        alphaExact = alphaExact && r.alphaExact;
        outside = outside && r.outsideUntouched;
    }
    MESSAGE("worst GPU-vs-CPU difference over " << n << " renders: " << worst);
    CHECK(worst <= 1e-3);
    CHECK(alphaExact);
    CHECK(outside);
}

// ---- M5: Full grain on the GPU ----
// The grain is a per-pixel simulation of random dye clouds (grain_common.h, shared with the CPU). The
// GPU computes in float what the CPU computes partly in double (the datasheet amplitude), and a cloud
// that touches a sub-sample by a hair's width can land differently, so the bar is statistical: at
// least 99.9 % of window pixels within 1e-3 of the CPU, the grain's rms (render minus the same render
// with grain off, each side against itself) within 2 %, alpha bit for bit.

namespace {

SpatialControls withGrain(SpatialControls s, uint32_t frameSeed = 7, double size = 1) {
    s.grain = true;
    s.grainAmount = 1;
    s.grainSize = size;
    s.grainQuality = GrainQuality::Full;
    s.frameSeed = frameSeed;
    return s;
}

struct GrainResult {
    double within = 0;            // fraction of window pixels whose RGB is within 1e-3 of the CPU's
    double rmsCpu = 0, rmsGpu = 0;  // rms of (grain on - grain off) over the window's RGB, each side
    long pixels = 0;              // window pixels compared
    double maxDiff = 0;           // largest |GPU - CPU| over the window's RGB (for the log)
    bool alphaExact = true;
    bool outsideUntouched = true;
};

GrainResult runGrain(const ImageGeometry& sg, const ImageGeometry& dg, const RenderWindow& win, const LookControls& look,
                     const SpatialControls& spatial, const Fill& fill) {
    REQUIRE(renderer().isValid());
    const auto baked = film();
    Frame f = makeFrame(sg, dg, fill);
    SpatialControls off = spatial;
    off.grain = false;
    std::vector<float> cpuOff(f.cpu.size(), kSentinel);
    renderFilm(ConstImageView{f.src.data(), sg}, ImageView{f.cpu.data(), dg}, win, *baked, look, spatial);
    renderFilm(ConstImageView{f.src.data(), sg}, ImageView{cpuOff.data(), dg}, win, *baked, look, off);
    id<MTLCommandQueue> queue = [device() newCommandQueue];
    bool ok1 = false, ok2 = false;
    const std::vector<float> gpu = renderGpu(queue, f, win, baked, look, &ok1, &spatial);
    const std::vector<float> gpuOff = renderGpu(queue, f, win, baked, look, &ok2, &off);
    CHECK(ok1);
    CHECK(ok2);
    ConstImageView cOn{f.cpu.data(), dg}, cOff{cpuOff.data(), dg}, gOn{gpu.data(), dg}, gOff{gpuOff.data(), dg};
    GrainResult r;
    long good = 0;
    double sc = 0, sg2 = 0;
    for (int y = dg.y1; y < dg.y2; ++y)
        for (int x = dg.x1; x < dg.x2; ++x) {
            const float *a = pixelAddress(cOn, x, y), *b = pixelAddress(gOn, x, y);
            const bool inWin = x >= win.x1 && x < win.x2 && y >= win.y1 && y < win.y2;
            if (!inWin) {
                for (int c = 0; c < 4; ++c)
                    if (b[c] != kSentinel) r.outsideUntouched = false;
                continue;
            }
            const float *a0 = pixelAddress(cOff, x, y), *b0 = pixelAddress(gOff, x, y);
            bool close = true;
            for (int c = 0; c < 3; ++c) {
                const double d = std::fabs(double(a[c]) - double(b[c]));
                close = close && d <= 1e-3;  // NaN: not close
                r.maxDiff = std::max(r.maxDiff, std::isnan(d) ? INFINITY : d);
                sc += (double(a[c]) - a0[c]) * (double(a[c]) - a0[c]);
                sg2 += (double(b[c]) - b0[c]) * (double(b[c]) - b0[c]);
            }
            if (std::memcmp(&a[3], &b[3], sizeof(float)) != 0) r.alphaExact = false;
            good += close ? 1 : 0;
            ++r.pixels;
        }
    if (r.pixels > 0) {
        r.within = double(good) / double(r.pixels);
        r.rmsCpu = std::sqrt(sc / double(3 * r.pixels));
        r.rmsGpu = std::sqrt(sg2 / double(3 * r.pixels));
    }
    return r;
}

// The bars. The rms is compared only over windows big enough to hold a measurable grain (1000 px);
// a window with no source under it (`grainy` false) has none on either side.
void checkGrain(const ImageGeometry& sg, const ImageGeometry& dg, const RenderWindow& win, const LookControls& look,
                const SpatialControls& spatial, const Fill& fill, bool grainy = true) {
    const GrainResult r = runGrain(sg, dg, win, look, spatial, fill);
    CAPTURE(r.within);
    CAPTURE(r.rmsCpu);
    CAPTURE(r.rmsGpu);
    MESSAGE("grain: " << r.within * 100 << " % within 1e-3 (max " << r.maxDiff << "), rms cpu " << r.rmsCpu
                      << " gpu " << r.rmsGpu);
    CHECK(r.within >= 0.999);
    if (!grainy) {
        CHECK(r.rmsCpu == 0);
        CHECK(r.rmsGpu == 0);
    } else if (r.pixels >= 1000) {
        CHECK(r.rmsCpu > 1e-3);  // there is grain to compare
        CHECK(std::fabs(r.rmsGpu / r.rmsCpu - 1) <= 0.02);
    }
    CHECK(r.alphaExact);
    CHECK(r.outsideUntouched);
}

// Pixel scales: clouds below a quarter pixel on both stages (Gaussian), the UHD Super 35 scale (the
// negative simulated, the print just above the threshold), and 8K (both simulated, larger clouds).
const double kGrainScales[] = {1000 / 24.89, 3840 / 24.89, 8192 / 24.89};

}  // namespace

TEST_CASE("grain full: gpu vs cpu") {
    const int w = 384, h = 216;
    const auto g = tight(0, 0, w, h);
    SUBCASE("points, edges, noise at three scales") {
        for (double pxPerMm : kGrainScales) {
            CAPTURE(pxPerMm);
            const SpatialControls s = withGrain(spatialAllOn(pxPerMm));
            checkGrain(g, g, {0, 0, w, h}, LookControls{}, s, pointsEdgesNoise(29, w, h));  // points, edges
            checkGrain(g, g, {0, 0, w, h}, LookControls{}, s, randomFill(61, 0.f, 1.f, 1.f));  // noise
        }
    }
    SUBCASE("Size, Amount and another frame") {
        SpatialControls s = withGrain(spatialAllOn(3840 / 24.89), 0xC0FFEE, 2.0);
        s.grainAmount = 3;
        checkGrain(g, g, {0, 0, w, h}, LookControls{}, s, randomFill(67, 0.f, 1.f, 1.f));
        s = withGrain(spatialAllOn(3840 / 24.89), 12345, 0.5);
        checkGrain(g, g, {0, 0, w, h}, LookControls{}, s, randomFill(67, 0.f, 1.f, 1.f));
    }
    SUBCASE("odd sizes") {
        for (auto wh : {std::pair<int, int>{1, 1}, {3, 97}, {1001, 17}}) {
            CAPTURE(wh.first);
            CAPTURE(wh.second);
            const auto og = tight(0, 0, wh.first, wh.second);
            checkGrain(og, og, {0, 0, wh.first, wh.second}, LookControls{}, withGrain(spatialAllOn(3840 / 24.89)),
                       randomFill(5, -0.1f, 1.2f));
        }
    }
    SUBCASE("padded rows, offset bounds, sub-windows") {
        const ImageGeometry sg{10, 20, 650, 380, 640 * 16 + 80};
        const ImageGeometry dg{10, 20, 650, 380, 640 * 16 + 48};
        LookControls look;
        look.exposureStops = 1.5;
        look.printerPoints = {3, -2, 1};
        for (const RenderWindow win : {RenderWindow{150, 90, 420, 300}, RenderWindow{10, 20, 77, 41},
                                       RenderWindow{600, 330, 650, 380}}) {
            CAPTURE(win.x1);
            CAPTURE(win.y1);
            checkGrain(sg, dg, win, look, withGrain(spatialAllOn(3840 / 24.89)), pointsEdgesNoise(37, 640, 360));
        }
    }
    SUBCASE("window partly outside the source; non-finite pixels") {
        checkGrain(tight(40, 30, 200, 130), tight(0, 0, 240, 160), {20, 10, 230, 150}, LookControls{},
                   withGrain(spatialAllOn(3840 / 24.89)), pointsEdgesNoise(41, 240, 160));
        const float nan = std::numeric_limits<float>::quiet_NaN(), inf = std::numeric_limits<float>::infinity();
        const float values[] = {nan, inf, -inf, -1.f, 2.f, 0.5f};
        const auto ng = tight(0, 0, 6, 36);
        checkGrain(ng, ng, {0, 0, 6, 36}, LookControls{}, withGrain(spatialAllOn(1000 / 24.89)),
                   [&](int x, int y, float* p) {
                       p[0] = values[x];
                       p[1] = values[y % 6];
                       p[2] = values[(y / 6) % 6];
                       p[3] = 0.25f;
                   });
    }
}

TEST_CASE("grain full: only effect on") {
    const int w = 384, h = 216;
    const auto g = tight(0, 0, w, h);
    for (double pxPerMm : kGrainScales) {
        CAPTURE(pxPerMm);
        SpatialControls s = spatialOff();
        s.pxPerMm = pxPerMm;
        s = withGrain(s, 99);
        checkGrain(g, g, {0, 0, w, h}, LookControls{}, s, pointsEdgesNoise(31, w, h));
        checkGrain(g, g, {0, 0, w, h}, LookControls{}, s, randomFill(73, 0.f, 1.f, 1.f));
    }
    SpatialControls s = spatialOff();
    s = withGrain(s, 5);
    checkGrain(tight(10, 20, 650, 380), tight(10, 20, 650, 380), {150, 90, 420, 300}, LookControls{}, s,
               pointsEdgesNoise(37, 640, 360));
    checkGrain(tight(300, 300, 364, 332), tight(0, 0, 64, 32), {0, 0, 64, 32}, LookControls{}, s,
               randomFill(43, 0.f, 1.f), false);  // window entirely outside the source
}

TEST_CASE("grain off on the gpu is M4") {
    REQUIRE(renderer().isValid());
    const auto g = tight(0, 0, 320, 180);
    Frame f = makeFrame(g, g, pointsEdgesNoise(79, 320, 180));
    id<MTLCommandQueue> queue = [device() newCommandQueue];
    const RenderWindow win{0, 0, 320, 180};
    for (bool allOn : {true, false}) {
        CAPTURE(allOn);
        SpatialControls m4 = allOn ? spatialAllOn(3840 / 24.89) : spatialOff();
        SpatialControls off = withGrain(m4);
        off.grain = false;
        SpatialControls zero = withGrain(m4);
        zero.grainAmount = 0;
        bool ok[3] = {false, false, false};
        const std::vector<float> ref = renderGpu(queue, f, win, film(), LookControls{}, &ok[0], &m4);
        const std::vector<float> a = renderGpu(queue, f, win, film(), LookControls{}, &ok[1], &off);
        const std::vector<float> b = renderGpu(queue, f, win, film(), LookControls{}, &ok[2], &zero);
        CHECK((ok[0] && ok[1] && ok[2]));
        double d = 0;
        for (size_t i = 0; i < ref.size(); ++i)
            d = std::max({d, std::fabs(double(a[i]) - ref[i]), std::fabs(double(b[i]) - ref[i])});
        CAPTURE(d);
        CHECK(d <= 1e-3);
    }
}

// Full grain at UHD Super 35 on the worst-case noise frame, every M4 control at its default: the
// spec's ceiling is 1 s per frame. Median of 3 after a warm-up frame.
TEST_CASE("benchmark: UHD Full grain (<= 1 s)") {
    REQUIRE(renderer().isValid());
    const auto g = tight(0, 0, 3840, 2160);
    Frame f = makeFrame(g, g, randomFill(71, 0.f, 1.f, 1.f));
    id<MTLDevice> dev = device();
    OFRecordingQueue* queue = [OFRecordingQueue new];
    queue.real = [dev newCommandQueue];
    id<MTLBuffer> srcBuf = [dev newBufferWithBytes:f.src.data() length:f.src.size() * 4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> dstBuf = [dev newBufferWithLength:f.cpu.size() * 4 options:MTLResourceStorageModeShared];
    SpatialControls s = withGrain(SpatialControls{});
    s.pxPerMm = 3840 / 24.89;
    std::vector<double> times;
    for (int i = 0; i < 4; ++i) {  // frame 0 warms up
        s.frameSeed = uint32_t(100 + i);
        const auto t0 = std::chrono::steady_clock::now();
        REQUIRE(renderer().render((__bridge void*)queue, (__bridge void*)srcBuf, g, (__bridge void*)dstBuf, g,
                                  {0, 0, 3840, 2160}, film(), LookControls{}, s));
        id<MTLCommandBuffer> cmd = queue.last;
        [cmd waitUntilCompleted];
        const double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        CHECK(cmd.status == MTLCommandBufferStatusCompleted);
        if (i > 0) times.push_back(wall);
        std::printf("UHD Full grain frame %d: GPU %.1f ms, wall %.1f ms\n", i, (cmd.GPUEndTime - cmd.GPUStartTime) * 1e3,
                    wall * 1e3);
    }
    std::sort(times.begin(), times.end());
    const double median = times[1];
    std::printf("UHD Full grain median (wall, host + GPU): %.1f ms\n", median * 1e3);
    CHECK(median <= 1.0);
}

// The M4 speed budget: a UHD Super 35 frame of uniform noise (the M3 worst case: neighbouring
// pixels land in unrelated table cells), every control at its default with Bloom on, tables already
// uploaded. Median GPU time of 50 frames: target 20 ms, asserted <= 22 ms (see the check). A
// profiling run first prints each pass's time.
TEST_CASE("benchmark: UHD all M4 effects (target 20 ms, ceiling 22 ms)") {
    REQUIRE(renderer().isValid());
    const auto g = tight(0, 0, 3840, 2160);
    Frame f = makeFrame(g, g, randomFill(71, 0.f, 1.f, 1.f));
    id<MTLDevice> dev = device();
    OFRecordingQueue* queue = [OFRecordingQueue new];
    queue.real = [dev newCommandQueue];
    id<MTLBuffer> srcBuf = [dev newBufferWithBytes:f.src.data() length:f.src.size() * 4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> dstBuf = [dev newBufferWithLength:f.cpu.size() * 4 options:MTLResourceStorageModeShared];
    SpatialControls s;  // every control at its default...
    s.bloom = true;     // ...with Bloom enabled
    s.pxPerMm = 3840 / 24.89;
    const RenderWindow win{0, 0, 3840, 2160};
    const auto render = [&] {
        return renderer().render((__bridge void*)queue, (__bridge void*)srcBuf, g, (__bridge void*)dstBuf, g, win,
                                 film(), LookControls{}, s);
    };

    // Profiling run: one command buffer per pass.
    renderer().setPassProfiling(true);
    REQUIRE(render());
    REQUIRE(render());  // the second, warm, is the one printed
    renderer().setPassProfiling(false);
    double sum = 0;
    for (const auto& p : renderer().lastPassTimes()) {
        std::printf("  %7.3f ms  %s\n", p.ms, p.name.c_str());
        sum += p.ms;
    }
    std::printf("  %7.3f ms  sum of passes\n", sum);

    std::vector<double> times;
    for (int i = 0; i < 51; ++i) {  // frame 0 warms up
        REQUIRE(render());
        id<MTLCommandBuffer> cmd = queue.last;
        [cmd waitUntilCompleted];
        CHECK(cmd.status == MTLCommandBufferStatusCompleted);
        if (i > 0) times.push_back(cmd.GPUEndTime - cmd.GPUStartTime);
    }
    std::sort(times.begin(), times.end());
    const double median = times[times.size() / 2] * 1e3;
    std::printf("UHD all M4 effects GPU median: %.3f ms (min %.3f, max %.3f); work set %.1f MB\n", median,
                times.front() * 1e3, times.back() * 1e3, double(renderer().poolStats().lastSetBytes) / 1e6);
    // The spec target is 20 ms. On the M4 Pro, three full ctest runs (print scatter-only, R14)
    // measured medians of 19.05, 19.86 and 19.13 ms: about 19.0-19.9 ms depending on load. The 22 ms
    // ceiling absorbs GPU contention from the rest of the desktop.
    CHECK(median <= 22.0);
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
