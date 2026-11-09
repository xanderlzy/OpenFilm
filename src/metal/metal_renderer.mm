#include "metal/metal_renderer.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "core/film_renderer.hpp"
#include "metal/shader_types.h"

#include <map>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>

namespace openfilm {

static_assert(float(kExposureLutEps) == OF_EXPOSURE_LUT_EPS, "film.metal's exposure-table offset must match the CPU's");

namespace {

OFImageGeom toShader(const ImageGeometry& g) {
    return {g.x1, g.y1, g.x2, g.y2, int(g.rowBytes / 4)};
}

// The domain goes over as float exactly as Lut3D::lookup rounds it (min, and max - min taken in
// double), so the GPU normalises a pixel with the very same numbers.
OFLut3D describe(const Lut3D& lut) {
    OFLut3D d{};
    d.size = lut.size();
    d.denseDarks = lut.denseDarks() ? 1 : 0;
    for (int k = 0; k < 3; ++k) {
        d.lo[k] = float(lut.domainMin()[size_t(k)]);
        d.span[k] = float(lut.domainMax()[size_t(k)] - lut.domainMin()[size_t(k)]);
    }
    return d;
}

id<MTLBuffer> upload(id<MTLDevice> device, const std::vector<float>& v) {
    return [device newBufferWithBytes:v.data() length:v.size() * sizeof(float) options:MTLResourceStorageModeShared];
}

// One bake's tables in GPU memory: the three 3D tables, and the six H&D curves (three negative,
// then three print) packed one after another in a single buffer.
struct GpuFilm {
    std::weak_ptr<const BakedFilm> owner;  // expires when the bake is dropped
    id<MTLBuffer> lutE, lutPE, lutD, curves;
    OFLut3D descE, descPE, descD;
    OFCurve neg[3], print[3];
};

std::shared_ptr<GpuFilm> uploadFilm(id<MTLDevice> device, const std::shared_ptr<const BakedFilm>& film) {
    auto g = std::make_shared<GpuFilm>();
    g->owner = film;
    g->lutE = upload(device, film->lutE.data());
    g->lutPE = upload(device, film->lutPE.data());
    g->lutD = upload(device, film->lutD.data());
    g->descE = describe(film->lutE);
    g->descPE = describe(film->lutPE);
    g->descD = describe(film->lutD);
    std::vector<float> packed;
    const auto add = [&](const Curve1D& c) {
        const OFCurve d{c.min, c.max, int(packed.size()), int(c.values.size())};
        packed.insert(packed.end(), c.values.begin(), c.values.end());
        return d;
    };
    for (int c = 0; c < 3; ++c) g->neg[c] = add(film->negCurves[size_t(c)]);
    for (int c = 0; c < 3; ++c) g->print[c] = add(film->printCurves[size_t(c)]);
    g->curves = upload(device, packed);
    if (!g->lutE || !g->lutPE || !g->lutD || !g->curves) return nullptr;
    return g;
}

}  // namespace

struct MetalRenderer::Impl {
    std::string path;
    std::string error;
    bool valid = false;

    // A Mac can have several GPUs, and compiled kernels ("pipeline states") and buffers belong to
    // one GPU. We build them lazily for whichever device Resolve's queue lives on, and cache them.
    mutable std::mutex mutex;
    std::unordered_map<uintptr_t, id<MTLComputePipelineState>> pipelines;
    // Uploaded film tables by (device, bake). The weak_ptr inside tells us when the bake is gone;
    // such entries are dropped on every lookup, so a new bake that happens to reuse a freed
    // address is never mistaken for the old one.
    std::map<std::pair<uintptr_t, const BakedFilm*>, std::shared_ptr<GpuFilm>> films;

    id<MTLComputePipelineState> pipelineFor(id<MTLDevice> device, std::string* err) {
        std::lock_guard<std::mutex> lock(mutex);
        auto key = reinterpret_cast<uintptr_t>((__bridge void*)device);
        auto it = pipelines.find(key);
        if (it != pipelines.end()) return it->second;

        NSError* nsErr = nil;
        NSURL* url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:path.c_str()]];
        id<MTLLibrary> library = [device newLibraryWithURL:url error:&nsErr];
        if (!library) {
            *err = "cannot load " + path + ": " +
                   (nsErr ? std::string(nsErr.localizedDescription.UTF8String) : "unknown error");
            return nil;
        }
        id<MTLFunction> fn = [library newFunctionWithName:@"film_kernel"];
        if (!fn) {
            *err = "film_kernel missing from " + path;
            return nil;
        }
        id<MTLComputePipelineState> pso = [device newComputePipelineStateWithFunction:fn error:&nsErr];
        if (!pso) {
            *err = "cannot build film_kernel: " +
                   (nsErr ? std::string(nsErr.localizedDescription.UTF8String) : "unknown error");
            return nil;
        }
        pipelines[key] = pso;
        return pso;
    }

    // The film's GPU tables, uploaded on first use. Uploading happens under the lock, so several
    // render threads meeting a new bake at once upload it only once.
    std::shared_ptr<GpuFilm> filmFor(id<MTLDevice> device, const std::shared_ptr<const BakedFilm>& film) {
        std::lock_guard<std::mutex> lock(mutex);
        for (auto it = films.begin(); it != films.end();)
            it = it->second->owner.expired() ? films.erase(it) : std::next(it);
        const auto key = std::make_pair(reinterpret_cast<uintptr_t>((__bridge void*)device), film.get());
        auto it = films.find(key);
        if (it != films.end()) return it->second;
        auto g = uploadFilm(device, film);
        if (g) films[key] = g;
        return g;
    }
};

MetalRenderer::MetalRenderer(const std::string& metallibPath) : impl_(std::make_unique<Impl>()) {
    impl_->path = metallibPath;
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            impl_->error = "no Metal device available";
            return;
        }
        // Validate the library up front so a broken install is reported immediately.
        impl_->valid = impl_->pipelineFor(device, &impl_->error) != nil;
    }
}

MetalRenderer::~MetalRenderer() = default;

bool MetalRenderer::isValid() const { return impl_->valid; }

const std::string& MetalRenderer::error() const { return impl_->error; }

size_t MetalRenderer::cachedFilmCount() const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    return impl_->films.size();
}

bool MetalRenderer::render(void* commandQueue, void* srcBuffer, const ImageGeometry& srcGeom, void* dstBuffer,
                           const ImageGeometry& dstGeom, const RenderWindow& win,
                           const std::shared_ptr<const BakedFilm>& film, const LookControls& controls) {
    if (!impl_->valid || !commandQueue || !srcBuffer || !dstBuffer || !film) return false;
    const int w = win.x2 - win.x1;
    const int h = win.y2 - win.y1;
    if (w <= 0 || h <= 0) return true;  // nothing to draw

    @autoreleasepool {
        id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)commandQueue;
        id<MTLBuffer> src = (__bridge id<MTLBuffer>)srcBuffer;
        id<MTLBuffer> dst = (__bridge id<MTLBuffer>)dstBuffer;
        id<MTLDevice> device = queue.device;

        std::string err;
        id<MTLComputePipelineState> pso = impl_->pipelineFor(device, &err);
        if (!pso) return false;
        const std::shared_ptr<GpuFilm> g = impl_->filmFor(device, film);
        if (!g) return false;

        // Per-render controls, computed exactly as renderFilmCpu computes them.
        OFFilmUniforms u{};
        u.src = toShader(srcGeom);
        u.dst = toShader(dstGeom);
        u.winX1 = win.x1;
        u.winY1 = win.y1;
        u.winX2 = win.x2;
        u.winY2 = win.y2;
        u.gain = exposureGain(controls.exposureStops);
        const Vec3 off = printOffsetsLogE(film->balance, controls);
        for (int k = 0; k < 3; ++k) u.offsets[k] = float(off[size_t(k)]);
        u.lutE = g->descE;
        u.lutPE = g->descPE;
        u.lutD = g->descD;
        for (int c = 0; c < 3; ++c) {
            u.neg[c] = g->neg[c];
            u.print[c] = g->print[c];
        }
        for (int k = 0; k < 9; ++k) u.displayMatrix[k] = film->displayMatrix[size_t(k)];

        // The command buffer keeps the table buffers alive until the GPU is done, even if the bake
        // is dropped meanwhile.
        id<MTLCommandBuffer> cmd = [queue commandBuffer];
        cmd.label = @"OpenFilm";
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:src offset:0 atIndex:0];
        [enc setBuffer:dst offset:0 atIndex:1];
        [enc setBytes:&u length:sizeof(u) atIndex:2];
        [enc setBuffer:g->lutE offset:0 atIndex:3];
        [enc setBuffer:g->lutPE offset:0 atIndex:4];
        [enc setBuffer:g->lutD offset:0 atIndex:5];
        [enc setBuffer:g->curves offset:0 atIndex:6];
        // One GPU thread per window pixel, in 16x16 tiles.
        [enc dispatchThreads:MTLSizeMake(NSUInteger(w), NSUInteger(h), 1)
            threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
        [enc endEncoding];
        [cmd commit];
    }
    return true;
}

}  // namespace openfilm
