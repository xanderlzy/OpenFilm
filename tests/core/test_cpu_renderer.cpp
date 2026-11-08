// Unit tests for the CPU reference renderer (src/core/cpu_renderer.*).
#include <doctest/doctest.h>

#include "core/cpu_renderer.hpp"
#include "core/image.hpp"
#include "core/render_params.hpp"

#include <cmath>
#include <functional>
#include <limits>
#include <vector>

using namespace openfilm;

namespace {

// Owns the memory for a test image and hands out views onto it.
struct TestImage {
    ImageGeometry geom;
    std::vector<unsigned char> bytes;

    TestImage(ImageGeometry g, float fill) : geom(g), bytes(size_t(g.rowBytes) * size_t(g.y2 - g.y1)) {
        // Fill everything, padding included, with `fill`.
        float* f = reinterpret_cast<float*>(bytes.data());
        for (size_t i = 0; i < bytes.size() / sizeof(float); ++i) f[i] = fill;
    }
    void set(const std::function<void(int x, int y, float* px)>& fn) {
        for (int y = geom.y1; y < geom.y2; ++y)
            for (int x = geom.x1; x < geom.x2; ++x) fn(x, y, pixelAddress(view(), x, y));
    }
    ImageView view() { return {reinterpret_cast<float*>(bytes.data()), geom}; }
    ConstImageView cview() { return {reinterpret_cast<const float*>(bytes.data()), geom}; }
    const float* at(int x, int y) { return pixelAddress(cview(), x, y); }
};

ImageGeometry tight(int x1, int y1, int x2, int y2) {
    return {x1, y1, x2, y2, std::ptrdiff_t(x2 - x1) * 4 * std::ptrdiff_t(sizeof(float))};
}

RenderWindow fullWindow(const ImageGeometry& g) { return {g.x1, g.y1, g.x2, g.y2}; }

void fillPattern(TestImage& img) {
    img.set([](int x, int y, float* p) {
        p[0] = 0.1f * float(x) - 0.5f;             // includes negative values
        p[1] = float(y) * 0.25f;
        p[2] = (x == 3 && y == 2) ? 1e4f : 0.5f;   // includes a very large value
        p[3] = 0.75f;
    });
}

}  // namespace

TEST_CASE("zero stops is identity") {
    TestImage src(tight(0, 0, 4, 3), 0.f), dst(tight(0, 0, 4, 3), 0.f);
    fillPattern(src);
    renderExposureCpu(src.cview(), dst.view(), fullWindow(src.geom), RenderParams{0.0});
    for (int y = 0; y < 3; ++y)
        for (int x = 0; x < 4; ++x)
            for (int c = 0; c < 4; ++c) CHECK(dst.at(x, y)[c] == src.at(x, y)[c]);
}

TEST_CASE("plus one stop doubles rgb, keeps alpha") {
    TestImage src(tight(0, 0, 1, 1), 0.f), dst(tight(0, 0, 1, 1), 0.f);
    src.set([](int, int, float* p) { p[0] = 0.25f; p[1] = 1.0f; p[2] = 3.0f; p[3] = 0.5f; });
    renderExposureCpu(src.cview(), dst.view(), fullWindow(src.geom), RenderParams{1.0});
    CHECK(dst.at(0, 0)[0] == 0.5f);
    CHECK(dst.at(0, 0)[1] == 2.0f);
    CHECK(dst.at(0, 0)[2] == 6.0f);
    CHECK(dst.at(0, 0)[3] == 0.5f);
}

TEST_CASE("minus two stops quarters rgb") {
    TestImage src(tight(0, 0, 1, 1), 1.f), dst(tight(0, 0, 1, 1), 0.f);
    renderExposureCpu(src.cview(), dst.view(), fullWindow(src.geom), RenderParams{-2.0});
    CHECK(dst.at(0, 0)[0] == 0.25f);
    CHECK(dst.at(0, 0)[1] == 0.25f);
    CHECK(dst.at(0, 0)[2] == 0.25f);
    CHECK(dst.at(0, 0)[3] == 1.0f);
}

TEST_CASE("only render window is written") {
    TestImage src(tight(0, 0, 4, 3), 1.f), dst(tight(0, 0, 4, 3), 7.f);
    RenderWindow win{1, 1, 3, 2};
    renderExposureCpu(src.cview(), dst.view(), win, RenderParams{1.0});
    for (int y = 0; y < 3; ++y)
        for (int x = 0; x < 4; ++x) {
            bool inside = x >= 1 && x < 3 && y >= 1 && y < 2;
            CAPTURE(x);
            CAPTURE(y);
            CHECK(dst.at(x, y)[0] == (inside ? 2.f : 7.f));
            CHECK(dst.at(x, y)[3] == (inside ? 1.f : 7.f));
        }
}

TEST_CASE("non-zero bounds origin and padded rows") {
    const float nan = std::numeric_limits<float>::quiet_NaN();
    ImageGeometry sg{100, 50, 104, 53, 4 * 16 + 32};  // 32 bytes of padding per row
    TestImage src(sg, nan), dst(tight(100, 50, 104, 53), 0.f);
    src.set([](int x, int y, float* p) {
        p[0] = float(x);
        p[1] = float(y);
        p[2] = float(x + y);
        p[3] = 0.5f;
    });
    renderExposureCpu(src.cview(), dst.view(), fullWindow(sg), RenderParams{1.0});
    for (int y = 50; y < 53; ++y)
        for (int x = 100; x < 104; ++x) {
            const float* d = dst.at(x, y);
            CAPTURE(x);
            CAPTURE(y);
            CHECK(d[0] == 2.f * float(x));
            CHECK(d[1] == 2.f * float(y));
            CHECK(d[2] == 2.f * float(x + y));
            CHECK(d[3] == 0.5f);
            for (int c = 0; c < 4; ++c) CHECK_FALSE(std::isnan(d[c]));
        }
}

TEST_CASE("pixelAddress returns nullptr out of bounds") {
    TestImage img(tight(10, 20, 14, 23), 0.f);
    CHECK(pixelAddress(img.cview(), 9, 20) == nullptr);
    CHECK(pixelAddress(img.cview(), 14, 20) == nullptr);
    CHECK(pixelAddress(img.cview(), 10, 23) == nullptr);
    CHECK(pixelAddress(img.cview(), 10, 20) != nullptr);
}
