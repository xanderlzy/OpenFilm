#include "metal/metal_renderer.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "core/blur.hpp"
#include "core/film_renderer.hpp"
#include "core/film_spatial.hpp"
#include "metal/shader_types.h"

#include <algorithm>
#include <array>
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

// Every kernel of OpenFilm.metallib, compiled for one device.
struct Pipelines {
    id<MTLComputePipelineState> film;  // M3 colour kernel
    id<MTLComputePipelineState> expose, blurH, blurV, boxDown, up, clear, bloomMix, halationAdd, develop, edge,
        printExpose, finish;
};

// One thread per pixel in tiles shaped from what the pipeline allows: a SIMD-group wide, as many
// rows as fit.
MTLSize threadgroupFor(id<MTLComputePipelineState> pso) {
    const NSUInteger w = std::max<NSUInteger>(pso.threadExecutionWidth, 1);
    const NSUInteger h = std::max<NSUInteger>(pso.maxTotalThreadsPerThreadgroup / w, 1);
    return MTLSizeMake(w, h, 1);
}

void dispatch(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pso, int w, int h) {
    [enc setComputePipelineState:pso];
    [enc dispatchThreads:MTLSizeMake(NSUInteger(w), NSUInteger(h), 1) threadsPerThreadgroup:threadgroupFor(pso)];
}

// The intermediate images of one spatial render: three full-size work images (two the stages
// ping-pong between, one for the horizontal blur pass) and two at half size or less for the
// reduced-resolution lobes (box down -> horizontal -> vertical back into the first -> up).
struct WorkSet {
    uintptr_t device = 0;
    size_t fullPx = 0, smallPx = 0;  // capacity in float4 pixels
    id<MTLBuffer> full[3];
    id<MTLBuffer> small[2];
};

// Per-device pool of WorkSets, shared by every render thread. A render takes a set big enough for
// its work rect (the smallest that fits, else a new one) and gives it back from its command
// buffer's completion handler, so a set is never reused while the GPU still works in it. At most
// kMaxIdle sets per device are kept between renders; the smallest go first.
class WorkPool {
public:
    std::shared_ptr<WorkSet> take(id<MTLDevice> device, size_t fullPx, size_t smallPx) {
        const auto key = reinterpret_cast<uintptr_t>((__bridge void*)device);
        {
            std::lock_guard<std::mutex> lock(mutex_);
            auto& idle = idle_[key];
            auto best = idle.end();
            for (auto it = idle.begin(); it != idle.end(); ++it)
                if ((*it)->fullPx >= fullPx && (*it)->smallPx >= smallPx &&
                    (best == idle.end() || (*it)->fullPx < (*best)->fullPx))
                    best = it;
            if (best != idle.end()) {
                auto set = *best;
                idle.erase(best);
                return set;
            }
        }
        // Allocate outside the lock: large buffers take a while.
        auto set = std::make_shared<WorkSet>();
        set->device = key;
        set->fullPx = fullPx;
        set->smallPx = smallPx;
        const auto bytes = [](size_t px) { return std::max<size_t>(px, 1) * 4 * sizeof(float); };
        for (auto& b : set->full) {
            b = [device newBufferWithLength:bytes(fullPx) options:MTLResourceStorageModePrivate];
            if (!b) return nullptr;
            b.label = @"OpenFilm work";
        }
        for (auto& b : set->small) {
            b = [device newBufferWithLength:bytes(smallPx) options:MTLResourceStorageModePrivate];
            if (!b) return nullptr;
            b.label = @"OpenFilm work (reduced)";
        }
        return set;
    }

    void give(std::shared_ptr<WorkSet> set) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto& idle = idle_[set->device];
        idle.push_back(std::move(set));
        if (idle.size() > kMaxIdle) {
            auto smallest = std::min_element(idle.begin(), idle.end(),
                                             [](const auto& a, const auto& b) { return a->fullPx < b->fullPx; });
            idle.erase(smallest);
        }
    }

    size_t idleCount() const {
        std::lock_guard<std::mutex> lock(mutex_);
        size_t n = 0;
        for (const auto& kv : idle_) n += kv.second.size();
        return n;
    }

private:
    static constexpr size_t kMaxIdle = 2;
    mutable std::mutex mutex_;
    std::unordered_map<uintptr_t, std::vector<std::shared_ptr<WorkSet>>> idle_;
};

// The spatial chain as a list of GPU passes, built on the host from the CPU's own lobes, sigmas and
// taps (film_spatial.cpp, blur.cpp), then encoded in order. Buffers are named by slot.
enum Slot { kFull0, kFull1, kFull2, kSmall0, kSmall1 };

struct Pass {
    enum Kind { BlurH, BlurV, BoxDown, Up, Clear, BloomMix, HalationAdd, Develop, Edge, PrintExpose } kind;
    int in = -1, out = -1;  // slots (in-place passes use `out` only)
    OFPassUniforms u{};
};

class SpatialPlan {
public:
    SpatialPlan(int w, int h) : w_(w), h_(h) {}

    std::vector<float> taps;
    std::vector<Pass> passes;
    bool usesReduced = false;
    size_t smallPx = 0;

    // out.rgb = sum_i weights[i] * G_{lobes[i]}(in)  (spread(), with per-channel weights).
    void spread(int in, int out, const std::vector<PxLobe>& lobes, const std::vector<std::array<float, 3>>& weights) {
        if (lobes.empty()) {
            add(Pass::Clear, -1, out, base());
            return;
        }
        for (size_t i = 0; i < lobes.size(); ++i) {
            const PxLobe& lobe = lobes[i];
            const int mode = i == 0 ? OF_STORE_WEIGHTED : OF_ACCUMULATE;
            const float smin = std::min({lobe.sigmaPx[0], lobe.sigmaPx[1], lobe.sigmaPx[2]});
            const int f = downsampleFactor(smin);
            if (f == 1) {
                OFPassUniforms b = blurUniforms(w_, h_, lobe.sigmaPx);
                add(Pass::BlurH, in, kFull2, b);
                b.mode = mode;
                setWeight(b, weights[i]);
                add(Pass::BlurV, kFull2, out, b);
            } else {
                const int sw = (w_ + f - 1) / f, sh = (h_ + f - 1) / f;
                usesReduced = true;
                smallPx = std::max(smallPx, size_t(sw) * size_t(sh));
                OFPassUniforms box = base();
                box.width = sw;
                box.height = sh;
                box.factor = f;
                add(Pass::BoxDown, in, kSmall0, box);
                std::array<float, 3> s;
                for (size_t c = 0; c < 3; ++c) s[c] = reducedSigmaPx(lobe.sigmaPx[c], f);
                OFPassUniforms b = blurUniforms(sw, sh, s);
                add(Pass::BlurH, kSmall0, kSmall1, b);
                b.mode = OF_STORE;
                add(Pass::BlurV, kSmall1, kSmall0, b);
                OFPassUniforms up = base();
                up.inWidth = sw;
                up.inHeight = sh;
                up.factor = f;
                up.mode = mode;
                setWeight(up, weights[i]);
                add(Pass::Up, kSmall0, out, up);
            }
        }
    }

    // Per-pixel pass over the full work image, with per-channel weights.
    void pointwise(Pass::Kind kind, int img, int other, std::array<float, 3> weight = {0, 0, 0}, int mode = 0) {
        OFPassUniforms u = base();
        u.mode = mode;
        setWeight(u, weight);
        add(kind, other, img, u);
    }

private:
    int w_, h_;

    OFPassUniforms base() const {
        OFPassUniforms u{};
        u.width = u.inWidth = w_;
        u.height = u.inHeight = h_;
        u.factor = 1;
        return u;
    }

    OFPassUniforms blurUniforms(int w, int h, const std::array<float, 3>& sigma) {
        OFPassUniforms u{};
        u.width = u.inWidth = w;
        u.height = u.inHeight = h;
        u.factor = 1;
        u.mode = OF_STORE;
        for (int c = 0; c < 3; ++c) {
            u.radius[c] = blurRadiusPx(sigma[size_t(c)]);
            u.tapOffset[c] = int(taps.size());
            if (u.radius[c] > 0) {
                const std::vector<float> t = blurTaps(sigma[size_t(c)]);
                taps.insert(taps.end(), t.begin(), t.end());
            }
        }
        return u;
    }

    static void setWeight(OFPassUniforms& u, const std::array<float, 3>& w) {
        for (int c = 0; c < 3; ++c) u.weight[c] = w[size_t(c)];
    }

    void add(Pass::Kind kind, int in, int out, const OFPassUniforms& u) {
        Pass p;
        p.kind = kind;
        p.in = in;
        p.out = out;
        p.u = u;
        passes.push_back(p);
    }
};

std::vector<std::array<float, 3>> lobeWeights(const std::vector<PxLobe>& lobes) {
    std::vector<std::array<float, 3>> w;
    for (const PxLobe& l : lobes) w.push_back({l.weight, l.weight, l.weight});
    return w;
}

std::array<float, 3> edgeGains(const std::array<LayerSharpness, 3>& layers) {
    return {float(layers[0].edgeK), float(layers[1].edgeK), float(layers[2].edgeK)};
}

// The stages of renderFilm() after exposeStage, in its order; the working image starts in kFull0.
// Returns the slot holding the final print density.
int planStages(SpatialPlan& plan, const FilmOptics& optics, const SpatialControls& s) {
    int img = kFull0, other = kFull1;
    // exposureEffects: bloom, scatter, halation.
    if (s.bloom) {
        const auto lobes = bloomPxLobes(s);
        plan.spread(img, other, lobes, lobeWeights(lobes));
        const float a = float(s.bloomAmount);
        plan.pointwise(Pass::BloomMix, img, other, {a, a, a});
    }
    if (s.sharpness) {
        const LayerPasses lp = layerPasses(optics.negative, s.pxPerMm);
        plan.spread(img, other, lp.passes, lp.weights);
        std::swap(img, other);
    }
    if (s.halation) {
        const auto lobes = haloLobes(optics, s);
        plan.spread(img, other, lobes, lobeWeights(lobes));
        plan.pointwise(Pass::HalationAdd, img, other, halationWeights(optics, s));
    }
    plan.pointwise(Pass::Develop, img, -1, {0, 0, 0}, 0);
    if (s.sharpness) {
        plan.spread(img, other, edgeLobe(optics.negative, s.pxPerMm), {{1.f, 1.f, 1.f}});
        plan.pointwise(Pass::Edge, img, other, edgeGains(optics.negative));
    }
    plan.pointwise(Pass::PrintExpose, img, -1);
    if (s.sharpness) {
        const LayerPasses lp = layerPasses(optics.print, s.pxPerMm);
        plan.spread(img, other, lp.passes, lp.weights);
        std::swap(img, other);
    }
    plan.pointwise(Pass::Develop, img, -1, {0, 0, 0}, 1);
    if (s.sharpness) {
        plan.spread(img, other, edgeLobe(optics.print, s.pxPerMm), {{1.f, 1.f, 1.f}});
        plan.pointwise(Pass::Edge, img, other, edgeGains(optics.print));
    }
    return img;
}

}  // namespace

struct MetalRenderer::Impl {
    std::string path;
    std::string error;
    bool valid = false;

    // A Mac can have several GPUs, and compiled kernels ("pipeline states") and buffers belong to
    // one GPU. We build them lazily for whichever device Resolve's queue lives on, and cache them.
    mutable std::mutex mutex;
    std::unordered_map<uintptr_t, Pipelines> pipelines;
    // Uploaded film tables by (device, bake). The weak_ptr inside tells us when the bake is gone;
    // such entries are dropped on every lookup, so a new bake that happens to reuse a freed
    // address is never mistaken for the old one.
    std::map<std::pair<uintptr_t, const BakedFilm*>, std::shared_ptr<GpuFilm>> films;
    // Shared with in-flight command buffers' completion handlers, which may outlive the renderer.
    std::shared_ptr<WorkPool> pool = std::make_shared<WorkPool>();

    bool pipelinesFor(id<MTLDevice> device, Pipelines* out, std::string* err) {
        std::lock_guard<std::mutex> lock(mutex);
        auto key = reinterpret_cast<uintptr_t>((__bridge void*)device);
        auto it = pipelines.find(key);
        if (it != pipelines.end()) {
            *out = it->second;
            return true;
        }

        NSError* nsErr = nil;
        NSURL* url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:path.c_str()]];
        id<MTLLibrary> library = [device newLibraryWithURL:url error:&nsErr];
        if (!library) {
            *err = "cannot load " + path + ": " +
                   (nsErr ? std::string(nsErr.localizedDescription.UTF8String) : "unknown error");
            return false;
        }
        const auto build = [&](const char* name, __strong id<MTLComputePipelineState>* pso) {
            id<MTLFunction> fn = [library newFunctionWithName:[NSString stringWithUTF8String:name]];
            if (!fn) {
                *err = std::string(name) + " missing from " + path;
                return false;
            }
            NSError* e = nil;
            *pso = [device newComputePipelineStateWithFunction:fn error:&e];
            if (!*pso) {
                *err = std::string("cannot build ") + name + ": " +
                       (e ? std::string(e.localizedDescription.UTF8String) : "unknown error");
                return false;
            }
            return true;
        };
        Pipelines p;
        if (!build("film_kernel", &p.film) || !build("sp_expose", &p.expose) || !build("sp_blur_h", &p.blurH) ||
            !build("sp_blur_v", &p.blurV) || !build("sp_box_down", &p.boxDown) || !build("sp_up", &p.up) ||
            !build("sp_clear", &p.clear) || !build("sp_bloom_mix", &p.bloomMix) ||
            !build("sp_halation_add", &p.halationAdd) || !build("sp_develop", &p.develop) ||
            !build("sp_edge", &p.edge) || !build("sp_print_expose", &p.printExpose) ||
            !build("sp_finish", &p.finish))
            return false;
        pipelines[key] = p;
        *out = p;
        return true;
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
        Pipelines p;
        impl_->valid = impl_->pipelinesFor(device, &p, &impl_->error);
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
        Pipelines pl;
        if (!impl_->pipelinesFor(device, &pl, &err)) return false;
        id<MTLComputePipelineState> pso = pl.film;
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
        if (!cmd) return false;  // report failure rather than leave the frame unwritten
        cmd.label = @"OpenFilm";
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        if (!enc) return false;
        [enc setComputePipelineState:pso];
        [enc setBuffer:src offset:0 atIndex:0];
        [enc setBuffer:dst offset:0 atIndex:1];
        [enc setBytes:&u length:sizeof(u) atIndex:2];
        [enc setBuffer:g->lutE offset:0 atIndex:3];
        [enc setBuffer:g->lutPE offset:0 atIndex:4];
        [enc setBuffer:g->lutD offset:0 atIndex:5];
        [enc setBuffer:g->curves offset:0 atIndex:6];
        // One GPU thread per window pixel.
        [enc dispatchThreads:MTLSizeMake(NSUInteger(w), NSUInteger(h), 1) threadsPerThreadgroup:threadgroupFor(pso)];
        [enc endEncoding];
        [cmd commit];
    }
    return true;
}

bool MetalRenderer::render(void* commandQueue, void* srcBuffer, const ImageGeometry& srcGeom, void* dstBuffer,
                           const ImageGeometry& dstGeom, const RenderWindow& win,
                           const std::shared_ptr<const BakedFilm>& film, const LookControls& look,
                           const SpatialControls& spatial) {
    if (!spatial.anyOn()) return render(commandQueue, srcBuffer, srcGeom, dstBuffer, dstGeom, win, film, look);
    if (!impl_->valid || !commandQueue || !srcBuffer || !dstBuffer || !film) return false;
    const int ww = win.x2 - win.x1;
    const int wh = win.y2 - win.y1;
    if (ww <= 0 || wh <= 0) return true;  // nothing to draw

    // The same work rect, lobes and taps as renderFilm(), from the CPU's own functions.
    const SpatialControls s = sanitize(spatial);
    static const float kAnyPixel = 0.f;  // workRect only needs a non-null source to look at its bounds
    const WorkRect rect = workRect(ConstImageView{&kAnyPixel, srcGeom}, win, spatialMarginPx(film->optics, s));
    const int W = rect.width(), H = rect.height();

    @autoreleasepool {
        id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)commandQueue;
        id<MTLBuffer> src = (__bridge id<MTLBuffer>)srcBuffer;
        id<MTLBuffer> dst = (__bridge id<MTLBuffer>)dstBuffer;
        id<MTLDevice> device = queue.device;

        std::string err;
        Pipelines pl;
        if (!impl_->pipelinesFor(device, &pl, &err)) return false;
        const std::shared_ptr<GpuFilm> g = impl_->filmFor(device, film);
        if (!g) return false;

        OFSpatialUniforms u{};
        u.src = toShader(srcGeom);
        u.dst = toShader(dstGeom);
        u.winX1 = win.x1;
        u.winY1 = win.y1;
        u.winX2 = win.x2;
        u.winY2 = win.y2;
        u.rectX1 = rect.x1;
        u.rectY1 = rect.y1;
        u.width = W;
        u.height = H;
        u.gain = exposureGain(look.exposureStops);
        const Vec3 off = printOffsetsLogE(film->balance, look);
        for (int k = 0; k < 3; ++k) u.offsets[k] = float(off[size_t(k)]);
        u.lutE = g->descE;
        u.lutPE = g->descPE;
        u.lutD = g->descD;
        for (int c = 0; c < 3; ++c) {
            u.neg[c] = g->neg[c];
            u.print[c] = g->print[c];
        }
        for (int k = 0; k < 9; ++k) u.displayMatrix[k] = film->displayMatrix[size_t(k)];

        SpatialPlan plan(W, H);
        int finalSlot = kFull0;
        std::shared_ptr<WorkSet> set;
        id<MTLBuffer> tapsBuf = nil;
        if (W > 0 && H > 0) {
            finalSlot = planStages(plan, film->optics, s);
            const size_t fullPx = size_t(W) * size_t(H);
            if (fullPx * 16 > device.maxBufferLength) return false;
            set = impl_->pool->take(device, fullPx, plan.smallPx);
            if (!set) return false;
            if (plan.taps.empty()) plan.taps.push_back(0.f);
            tapsBuf = upload(device, plan.taps);
            if (!tapsBuf) {
                impl_->pool->give(set);
                return false;
            }
        }
        const auto slot = [&](int i) -> id<MTLBuffer> {
            return i < kSmall0 ? set->full[i] : set->small[i - kSmall0];
        };

        id<MTLCommandBuffer> cmd = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = cmd ? [cmd computeCommandEncoder] : nil;
        if (!enc) {
            if (set) impl_->pool->give(set);
            return false;  // report failure rather than leave the frame unwritten
        }
        cmd.label = @"OpenFilm spatial";

        if (set) {
            // exposeStage
            [enc setBuffer:src offset:0 atIndex:0];
            [enc setBuffer:slot(kFull0) offset:0 atIndex:1];
            [enc setBytes:&u length:sizeof(u) atIndex:2];
            [enc setBuffer:g->lutE offset:0 atIndex:3];
            dispatch(enc, pl.expose, W, H);

            for (const Pass& p : plan.passes) {
                id<MTLComputePipelineState> pso = nil;
                switch (p.kind) {
                    case Pass::BlurH: pso = pl.blurH; break;
                    case Pass::BlurV: pso = pl.blurV; break;
                    case Pass::BoxDown: pso = pl.boxDown; break;
                    case Pass::Up: pso = pl.up; break;
                    case Pass::Clear: pso = pl.clear; break;
                    case Pass::BloomMix: pso = pl.bloomMix; break;
                    case Pass::HalationAdd: pso = pl.halationAdd; break;
                    case Pass::Develop: pso = pl.develop; break;
                    case Pass::Edge: pso = pl.edge; break;
                    case Pass::PrintExpose: pso = pl.printExpose; break;
                }
                // Two-image passes read `in` at 0 and write `out` at 1; combine passes update `out`
                // (bound at 0) from `in` (at 1); per-pixel stages work on `out` in place at 0.
                const bool combine = p.kind == Pass::BloomMix || p.kind == Pass::HalationAdd || p.kind == Pass::Edge;
                const bool inPlace = p.kind == Pass::Develop || p.kind == Pass::PrintExpose;
                if (combine) {
                    [enc setBuffer:slot(p.out) offset:0 atIndex:0];
                    [enc setBuffer:slot(p.in) offset:0 atIndex:1];
                } else if (inPlace) {
                    [enc setBuffer:slot(p.out) offset:0 atIndex:0];
                } else {
                    if (p.in >= 0) [enc setBuffer:slot(p.in) offset:0 atIndex:0];
                    [enc setBuffer:slot(p.out) offset:0 atIndex:1];
                }
                [enc setBytes:&p.u length:sizeof(p.u) atIndex:2];
                if (p.kind == Pass::BlurH || p.kind == Pass::BlurV) [enc setBuffer:tapsBuf offset:0 atIndex:3];
                if (inPlace) {
                    [enc setBytes:&u length:sizeof(u) atIndex:3];
                    [enc setBuffer:p.kind == Pass::Develop ? g->curves : g->lutPE offset:0 atIndex:4];
                }
                dispatch(enc, pso, p.u.width, p.u.height);
            }
        }

        // finishStage: every window pixel (zeros where the window leaves the source).
        [enc setBuffer:set ? slot(finalSlot) : dst offset:0 atIndex:0];
        [enc setBuffer:dst offset:0 atIndex:1];
        [enc setBytes:&u length:sizeof(u) atIndex:2];
        [enc setBuffer:g->lutD offset:0 atIndex:3];
        [enc setBuffer:src offset:0 atIndex:4];
        dispatch(enc, pl.finish, ww, wh);
        [enc endEncoding];

        // The command buffer keeps every bound buffer alive until the GPU is done; the work set
        // goes back to the pool only then.
        if (set) {
            std::shared_ptr<WorkPool> pool = impl_->pool;
            std::shared_ptr<WorkSet> held = set;
            [cmd addCompletedHandler:^(id<MTLCommandBuffer>) {
                pool->give(held);
            }];
        }
        [cmd commit];
    }
    return true;
}

}  // namespace openfilm
