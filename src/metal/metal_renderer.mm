#include "metal/metal_renderer.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "core/blur.hpp"
#include "core/film_renderer.hpp"
#include "core/film_spatial.hpp"
#include "core/grain_plates.hpp"
#include "metal/grain_common.h"
#include "metal/shader_types.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <functional>
#include <future>
#include <string>
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
    // The spatial path's copies, one 4-lane node per table entry (the 4th lane unused) so a corner
    // is one load: lutE as half (its values are exact in half), lutPE and lutD as float.
    id<MTLBuffer> lutE4h, lutPE4, lutD4;
    OFLut3D descE, descPE, descD;
    OFCurve neg[3], print[3];
    // Grain: every layer's datasheet nodes (density, ln sigma) as float pairs, negative R, G, B then
    // print R, G, B; where each layer's start and how many (0 when the film has no grain data).
    id<MTLBuffer> grainNodes;
    int grainOffset[6] = {}, grainCount[6] = {};
    // Each layer's node index (OFGrainLayer::binScale): OF_GRAIN_BINS node numbers per layer.
    id<MTLBuffer> grainBins;
    float grainBinScale[6] = {};
};

std::shared_ptr<GpuFilm> uploadFilm(id<MTLDevice> device, const std::shared_ptr<const BakedFilm>& film) {
    auto g = std::make_shared<GpuFilm>();
    g->owner = film;
    g->lutE = upload(device, film->lutE.data());
    {
        const auto& e = film->lutE.data();
        std::vector<_Float16> h(e.size() / 3 * 4, _Float16(0.0f));
        for (size_t i = 0; i < e.size(); ++i) h[i / 3 * 4 + i % 3] = _Float16(e[i]);
        g->lutE4h = [device newBufferWithBytes:h.data() length:h.size() * sizeof(_Float16)
                                       options:MTLResourceStorageModeShared];
        const auto pad = [&](const std::vector<float>& v) {
            std::vector<float> p(v.size() / 3 * 4, 0.0f);
            for (size_t i = 0; i < v.size(); ++i) p[i / 3 * 4 + i % 3] = v[i];
            return upload(device, p);
        };
        g->lutPE4 = pad(film->lutPE.data());
        g->lutD4 = pad(film->lutD.data());
    }
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
    // GrainAmplitude's nodes (amplitudeOf: density, natural log of sigma), rounded to float.
    std::vector<float> nodes;
    if (film->grain.valid)
        for (int l = 0; l < 6; ++l) {
            const GrainLayer& gl = l < 3 ? film->grain.negative[size_t(l)] : film->grain.print[size_t(l - 3)];
            const size_t count = std::min(gl.density.size(), gl.sigma.size());
            g->grainOffset[l] = int(nodes.size() / 2);
            g->grainCount[l] = int(std::min<size_t>(count, size_t(kGrainMaxNodes)));
            for (size_t i = 0; i < size_t(g->grainCount[l]); ++i) {
                nodes.push_back(float(gl.density[i]));
                nodes.push_back(float(std::log(gl.sigma[i])));
            }
        }
    // Bin j of a layer starts at density first + j / binScale and holds upper_bound's answer there,
    // kept within [1, n - 1]; the shader walks from it to the answer for its own D.
    std::vector<uint8_t> bins(6 * OF_GRAIN_BINS, 1);
    static_assert(kGrainMaxNodes <= 255, "node numbers are stored as bytes");
    for (int l = 0; l < 6; ++l) {
        const int n = g->grainCount[l];
        if (n < 2) continue;
        const float* nd = nodes.data() + 2 * g->grainOffset[l];
        const float span = nd[2 * (n - 1)] - nd[0];
        if (!(span > 0)) continue;
        g->grainBinScale[l] = float(OF_GRAIN_BINS) / span;
        for (int j = 0; j < OF_GRAIN_BINS; ++j) {
            const float start = nd[0] + float(j) / g->grainBinScale[l];
            int lo = 1;
            while (lo < n - 1 && !(start < nd[2 * lo])) ++lo;
            bins[size_t(l * OF_GRAIN_BINS + j)] = uint8_t(lo);
        }
    }
    if (nodes.empty()) nodes.assign(2, 0.f);  // a buffer to bind
    g->grainNodes = upload(device, nodes);
    g->grainBins = [device newBufferWithBytes:bins.data() length:bins.size() options:MTLResourceStorageModeShared];
    if (!g->lutE || !g->lutPE || !g->lutD || !g->curves || !g->lutE4h || !g->lutPE4 || !g->lutD4 || !g->grainNodes ||
        !g->grainBins)
        return nullptr;
    return g;
}

// One Preview plate set in GPU memory: the six plates (negative R, G, B, then print R, G, B), each
// size x size halves, one after another in one buffer.
struct GpuPlates {
    uintptr_t device = 0;
    PlateKey key;
    int size = 0;
    id<MTLBuffer> plates;
};

// Every kernel of OpenFilm.metallib, compiled for one device.
struct Pipelines {
    id<MTLComputePipelineState> film;  // M3 colour kernel
    id<MTLComputePipelineState> expose, boxDown, boxDown2, up, blurHTiled, blurHBlock, blurV, finish;
    id<MTLComputePipelineState> makePlate;  // Preview grain plates
    id<MTLLibrary> library;  // for the specialised sp_fused pipelines
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

// The intermediate images of one spatial render: four full-size work images (two the stages
// ping-pong between, two for separate horizontal blur passes, T and T2), and at reduced resolution three
// scratch images (box down, box down by twice the factor, horizontal blur) plus one that holds every
// reduced lobe's result for the fused pass that reads them.
struct WorkSet {
    uintptr_t device = 0;
    // Capacity: full and scratch images in pixels (three float planes each), the results in floats.
    size_t fullPx = 0, smallPx = 0, resPx = 0;
    id<MTLBuffer> full[4];
    id<MTLBuffer> small[4];  // scratch x 3, results

    size_t bytes() const { return (3 * (4 * fullPx + 3 * smallPx) + resPx) * sizeof(float); }
};

// Per-device pool of WorkSets, shared by every render thread. A render takes a set big enough for
// its work rect (the smallest idle one that fits, else a new one) and gives it back from its command
// buffer's completion handler, so a set is never reused while the GPU still works in it.
//
// Memory is bounded: at most kMaxSets sets exist per device, in flight and idle together (a UHD set
// is ~500 MB). A render that would go over waits in take() until a set comes back, then reuses it
// (or replaces it, when it is too small). take() is called with no renderer lock held, and give()
// never waits, so completion handlers cannot stall. Idle sets are marked volatile so the OS may
// reclaim their memory under pressure; take() makes a set non-volatile again and drops it for a
// fresh one if the OS did. Besides the most recently returned one, idle sets are released once they
// have been idle for kIdleExpiry (checked on every take and give; there is no timer).
class WorkPool {
public:
    static constexpr size_t kMaxSets = 3;
    static constexpr std::chrono::milliseconds kIdleExpiry{1000};

    std::shared_ptr<WorkSet> take(id<MTLDevice> device, size_t fullPx, size_t smallPx, size_t resPx) {
        const auto key = reinterpret_cast<uintptr_t>((__bridge void*)device);
        std::shared_ptr<WorkSet> reused;
        std::vector<std::shared_ptr<WorkSet>> released;  // freed after the lock is dropped
        {
            std::unique_lock<std::mutex> lock(mutex_);
            Device& d = devices_[key];
            returned_.wait(lock, [&] { return d.inFlight < kMaxSets; });
            ++d.inFlight;
            ++inFlight_;
            peak_ = std::max(peak_, d.inFlight);
            expire(d, released);
            auto best = d.idle.end();
            for (auto it = d.idle.begin(); it != d.idle.end(); ++it)
                if (it->set->fullPx >= fullPx && it->set->smallPx >= smallPx && it->set->resPx >= resPx &&
                    (best == d.idle.end() || it->set->fullPx < best->set->fullPx))
                    best = it;
            if (best != d.idle.end()) {
                reused = best->set;
                d.idle.erase(best);
            } else {
                // A new set is coming: drop idle ones (smallest first) so the total stays within the cap.
                while (!d.idle.empty() && d.inFlight + d.idle.size() > kMaxSets) {
                    auto smallest = std::min_element(d.idle.begin(), d.idle.end(), [](const Idle& a, const Idle& b) {
                        return a.set->fullPx < b.set->fullPx;
                    });
                    released.push_back(std::move(smallest->set));
                    d.idle.erase(smallest);
                }
            }
        }
        released.clear();
        if (reused && makeNonVolatile(*reused)) return reused;
        reused.reset();  // purged while idle (or never there): allocate afresh
        // Allocate outside the lock: large buffers take a while.
        auto set = std::make_shared<WorkSet>();
        set->device = key;
        set->fullPx = fullPx;
        set->smallPx = smallPx;
        set->resPx = resPx;
        const auto bytes = [](size_t floats) { return std::max<size_t>(floats, 1) * sizeof(float); };
        for (auto& b : set->full) {
            b = [device newBufferWithLength:bytes(3 * fullPx) options:MTLResourceStorageModePrivate];
            if (!b) return failed(key);
            b.label = @"OpenFilm work";
        }
        for (int k = 0; k < 4; ++k) {
            id<MTLBuffer> b = [device newBufferWithLength:bytes(k < 3 ? 3 * smallPx : resPx)
                                                  options:MTLResourceStorageModePrivate];
            if (!b) return failed(key);
            b.label = @"OpenFilm work (reduced)";
            set->small[k] = b;
        }
        return set;
    }

    // Never blocks beyond the pool's own short lock: called from completion handlers.
    void give(std::shared_ptr<WorkSet> set) {
        for (auto& b : set->full) [b setPurgeableState:MTLPurgeableStateVolatile];
        for (auto& b : set->small) [b setPurgeableState:MTLPurgeableStateVolatile];
        std::vector<std::shared_ptr<WorkSet>> released;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            Device& d = devices_[set->device];
            --d.inFlight;
            --inFlight_;
            d.idle.push_back({std::move(set), Clock::now()});
            expire(d, released);
        }
        returned_.notify_all();
    }

    size_t idleMaxPerDevice() const {
        std::lock_guard<std::mutex> lock(mutex_);
        size_t n = 0;
        for (const auto& kv : devices_) n = std::max(n, kv.second.idle.size());
        return n;
    }

    size_t inFlight() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return inFlight_;
    }

    size_t peakInFlight() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return peak_;
    }

    void resetPeak() {
        std::lock_guard<std::mutex> lock(mutex_);
        peak_ = 0;
        for (const auto& kv : devices_) peak_ = std::max(peak_, kv.second.inFlight);
    }

private:
    using Clock = std::chrono::steady_clock;
    struct Idle {
        std::shared_ptr<WorkSet> set;
        Clock::time_point since;
    };
    struct Device {
        std::vector<Idle> idle;  // in the order they came back
        size_t inFlight = 0;
    };
    mutable std::mutex mutex_;
    std::condition_variable returned_;
    std::unordered_map<uintptr_t, Device> devices_;
    size_t inFlight_ = 0;  // all devices
    size_t peak_ = 0;      // most in flight on one device since resetPeak()

    // Moves every idle set but the most recently returned one that has been idle for kIdleExpiry
    // into `out` (freed by the caller once the lock is dropped).
    static void expire(Device& d, std::vector<std::shared_ptr<WorkSet>>& out) {
        if (d.idle.size() <= 1) return;
        const auto now = Clock::now();
        for (size_t i = d.idle.size() - 1; i-- > 0;)
            if (now - d.idle[i].since >= kIdleExpiry) {
                out.push_back(std::move(d.idle[i].set));
                d.idle.erase(d.idle.begin() + std::ptrdiff_t(i));
            }
    }

    // Every buffer back to non-volatile; false if the OS discarded any of them while idle.
    static bool makeNonVolatile(WorkSet& set) {
        bool intact = true;
        for (auto& b : set.full)
            intact = [b setPurgeableState:MTLPurgeableStateNonVolatile] != MTLPurgeableStateEmpty && intact;
        for (auto& b : set.small)
            intact = [b setPurgeableState:MTLPurgeableStateNonVolatile] != MTLPurgeableStateEmpty && intact;
        return intact;
    }

    std::shared_ptr<WorkSet> failed(uintptr_t key) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            --devices_[key].inFlight;
            --inFlight_;
        }
        returned_.notify_all();
        return nullptr;
    }
};

// The spatial chain as a list of GPU passes, built on the host from the CPU's own lobes, sigmas and
// taps (film_spatial.cpp, blur.cpp), then encoded in order. Buffers are named by slot.
enum Slot { kFull0, kFull1, kFull2, kFull3, kSmall0, kSmall1, kSmall2, kSmallRes, kDst };

struct Pass {
    enum Kind { BoxDown, BoxDown2, BlurHTiled, BlurHBlock, BlurV, Up, Fused } kind;
    int in = -1, out = -1, out2 = -1;  // slots (out2: BoxDown2's 2f result)
    size_t inOffset = 0, outOffset = 0;  // floats into `in` / `out` (reduced results)
    OFPassUniforms u{};
    OFPassUniforms u2{};               // BlurHTiled: a second blur from the same tile (radius -1: none)
    size_t tileBytes = 0;              // BlurHTiled: threadgroup memory
    const char* stage = "";            // for profiling
    // Fused: uniforms, shape (function constants) and its pipeline.
    OFFuseUniforms fu{};
    int combine = 0, post = 0, channels = 3, nLobes = 0, kinds = 0;
    int sum = -1;                      // slot holding the sum of the lobes before the fused ones, if any
    id<MTLComputePipelineState> pso = nil;
};

// One spatial stage of renderFilm(): a spread of the working image, the combine that uses it, and
// which pointwise stages run straight after it.
struct Stage {
    const char* name;
    std::vector<PxLobe> lobes;
    std::vector<std::array<float, 3>> weights;  // per lobe, per channel
    int combine;
    std::array<float, 3> cw;
    int channels;  // 1 when only red of the sum is used
    int post = 0;
};

// Default for MetalRenderer::setDirectBlurLimit: at UHD Super 35 the negative's finest scatter lobe
// (about 19 reads) is computed in the fused pass, its wider one (54) and the edge blurs are not.
constexpr int kDefaultDirectLimit = 24;
// A DIRECT lobe at least this costly moves to the second T buffer when the fused pass has one PRE_H.
constexpr int kPromoteCost = 12;

class SpatialPlan {
public:
    SpatialPlan(int w, int h, int ww, int wh, size_t tgLimitBytes, int directLimit)
        : w_(w), h_(h), ww_(ww), wh_(wh), tgLimit_(tgLimitBytes), directLimit_(directLimit) {}

    std::vector<float> taps;
    std::vector<Pass> passes;
    size_t smallPx = 0, resPx = 0;

    // Encodes `st` reading slot `img`; the result goes to `other` (or dst when it finishes).
    void stage(const Stage& st, int img, int other) {
        stage_ = st.name;
        // spread() runs each lobe's channels in groups that share a downsample factor (lobeGroups);
        // each group is one lobe here, holding only its channels (and only those the stage uses).
        struct Sub {
            PxLobe lobe;
            std::array<float, 3> w;
            int mask;
        };
        std::vector<Sub> sub;
        std::vector<int> factor;
        for (size_t i = 0; i < st.lobes.size(); ++i)
            for (const LobeGroup& g : lobeGroups(st.lobes[i])) {
                const int mask = g.mask & ((1 << st.channels) - 1);
                if (mask == 0) continue;  // channels the stage never reads
                sub.push_back({st.lobes[i], st.weights[i], mask});
                factor.push_back(g.factor);
            }
        const size_t n = sub.size();
        // How each lobe runs: REDUCED when downsampled; at full resolution DIRECT while its
        // horizontal pass is cheap to redo per column block (directCost), else PRE_H.
        std::vector<int> kind(n);
        for (size_t i = 0; i < n; ++i)
            kind[i] = factor[i] > 1 ? OF_LOBE_REDUCED
                      : directCost(sub[i].lobe, sub[i].mask) <= directLimit_ ? OF_LOBE_DIRECT
                                                                             : OF_LOBE_PRE_H;
        // The fused pass takes the longest tail of lobes it can (OF_MAX_LOBES, two PRE_H: there are
        // two T buffers); the lobes before it are summed into `other` by separate passes, as spread()
        // does, and the fused pass continues that sum and overwrites it with its result.
        size_t first = n;
        int preH = 0;
        while (first > 0 && n - first < OF_MAX_LOBES) {
            if (kind[first - 1] == OF_LOBE_PRE_H && preH++ == 2) break;
            --first;
        }
        // A tail with one PRE_H and a DIRECT lobe that is not nearly free: its horizontal pass is
        // cheaper done once, in the same pass as the PRE_H one's (one read of the image), than per
        // column block (measured at UHD: negative scatter's r = 1,2,2 lobe, cost 19, gains ~1 ms;
        // print scatter's r = 0,0,1, cost 6, loses ~0.4 ms).
        if (preH == 1)
            for (size_t i = first; i < n; ++i)
                if (kind[i] == OF_LOBE_DIRECT && directCost(sub[i].lobe, sub[i].mask) >= kPromoteCost) {
                    kind[i] = OF_LOBE_PRE_H;
                    break;
                }
        // The tail's first PRE_H lobe reads T (kFull2), its second T2 (kFull3).
        for (size_t i = first, k = 0; i < n; ++i)
            if (kind[i] == OF_LOBE_PRE_H && k++ == 1) kind[i] = OF_LOBE_PRE_H2;

        // Every reduced lobe first, each blurred into its own place in kSmallRes. One pair of
        // factors f and 2f is boxed down in one read (bloom's and halation's two lobes are).
        std::vector<size_t> resAt(n, 0);
        size_t resUsed = 0;
        size_t pa = n, pb = n;
        for (size_t i = 0; i < n && pa == n; ++i)
            for (size_t j = i + 1; j < n && pa == n; ++j)
                if (factor[i] > 1 && (factor[j] == 2 * factor[i] || 2 * factor[j] == factor[i])) {
                    pa = factor[i] < factor[j] ? i : j;
                    pb = pa == i ? j : i;
                }
        if (pa < n) {
            int mask = 0;  // the channels any lobe at either factor needs
            for (size_t i = 0; i < n; ++i)
                if (factor[i] == factor[pa] || factor[i] == factor[pb]) mask |= sub[i].mask;
            box2(img, factor[pa], mask);
        }
        for (int pass = 0; pass < 2; ++pass)  // the paired factors' lobes, then the rest
            for (size_t i = 0; i < n; ++i) {
                if (factor[i] == 1) continue;
                const bool paired = pa < n && (factor[i] == factor[pa] || factor[i] == factor[pb]);
                if (paired != (pass == 0)) continue;
                resAt[i] = resUsed;
                reduce(img, paired ? (factor[i] == factor[pa] ? kSmall0 : kSmall2) : -1, sub[i].lobe, factor[i],
                       resUsed, sub[i].mask);
                resUsed += 3 * size_t((w_ + factor[i] - 1) / factor[i]) * size_t((h_ + factor[i] - 1) / factor[i]);
            }
        resPx = std::max(resPx, resUsed);

        // Prefix: spread() pass by pass, summed into `other`.
        for (size_t i = 0; i < first; ++i) {
            const PxLobe& lobe = sub[i].lobe;
            const int mode = i == 0 ? OF_STORE_WEIGHTED : OF_ACCUMULATE;
            if (factor[i] == 1) {
                OFPassUniforms b = blurUniforms(w_, h_, lobe.sigmaPx, sub[i].mask);
                addBlurH(img, kFull2, b);
                b.mode = mode;
                setWeight(b, sub[i].w);
                add(Pass::BlurV, kFull2, other, b);
            } else {
                OFPassUniforms up = base();
                up.inWidth = (w_ + factor[i] - 1) / factor[i];
                up.inHeight = (h_ + factor[i] - 1) / factor[i];
                up.factor = factor[i];
                up.mode = mode;
                setWeight(up, sub[i].w);
                for (int c = 0; c < 3; ++c) up.radius[c] = sub[i].mask >> c & 1 ? 0 : -1;
                add(Pass::Up, kSmallRes, other, up);
                passes.back().inOffset = resAt[i];
            }
        }

        // The fused pass (and the horizontal pass of its PRE_H lobe).
        Pass f;
        f.kind = Pass::Fused;
        f.in = img;
        f.out = st.post & OF_POST_FINISH ? kDst : other;
        f.sum = first > 0 ? other : -1;
        f.combine = st.combine;
        f.post = st.post;
        f.channels = st.channels;
        OFFuseUniforms& fu = f.fu;
        fu.width = w_;
        fu.height = h_;
        for (int c = 0; c < 3; ++c) fu.cw[c] = st.cw[size_t(c)];
        std::vector<OFPassUniforms> preHs;  // the tail's PRE_H lobes' horizontal passes, T then T2
        for (size_t i = first; i < n; ++i) {
            const PxLobe& lobe = sub[i].lobe;
            f.kinds |= kind[i] << (2 * f.nLobes);
            OFLobe& L = fu.lobes[f.nLobes++];
            for (int c = 0; c < 3; ++c) L.weight[c] = sub[i].w[size_t(c)];
            if (kind[i] == OF_LOBE_REDUCED) {
                const int fct = factor[i];
                L.factor = fct;
                L.sw = (w_ + fct - 1) / fct;
                L.sh = (h_ + fct - 1) / fct;
                L.smallOffset = int(resAt[i]);
                for (int c = 0; c < 3; ++c) L.radius[c] = sub[i].mask >> c & 1 ? 0 : -1;
                continue;
            }
            const OFPassUniforms b = blurUniforms(w_, h_, lobe.sigmaPx, sub[i].mask);
            for (int c = 0; c < 3; ++c) {
                L.radius[c] = b.radius[c];
                L.tapOffset[c] = b.tapOffset[c];
            }
            if (kind[i] != OF_LOBE_DIRECT) preHs.push_back(b);
        }
        if (preHs.size() == 2 && tiledFits(preHs[0], preHs[1])) {
            addBlurH(img, kFull2, preHs[0], &preHs[1], kFull3);  // both from one tile
        } else {
            for (size_t k = 0; k < preHs.size(); ++k) addBlurH(img, k == 0 ? kFull2 : kFull3, preHs[k]);
        }
        f.stage = stage_;
        passes.push_back(f);
    }

    // The fused pass's grid: the window when it finishes, else the work rect.
    int gridW(const Pass& p) const { return p.post & OF_POST_FINISH ? ww_ : w_; }
    int gridH(const Pass& p) const { return p.post & OF_POST_FINISH ? wh_ : h_; }

private:
    int w_, h_, ww_, wh_;
    size_t tgLimit_;
    int directLimit_;
    const char* stage_ = "";

    // Image reads per output pixel of a DIRECT lobe: each column block of OF_BLOCK pixels computes
    // OF_BLOCK + 2r horizontal sums of 2r + 1 taps per channel.
    static int directCost(const PxLobe& lobe, int mask) {
        int n = 0;
        for (int c = 0; c < 3; ++c) {
            if (!(mask >> c & 1)) continue;
            const int r = blurRadiusPx(lobe.sigmaPx[size_t(c)]);
            n += (OF_BLOCK + 2 * r) * (2 * r + 1);
        }
        return (n + OF_BLOCK - 1) / OF_BLOCK;
    }

    OFPassUniforms base() const {
        OFPassUniforms u{};
        u.width = u.inWidth = w_;
        u.height = u.inHeight = h_;
        u.factor = 1;
        return u;
    }

    // Taps of the channels in `mask` (the others get radius -1: not computed).
    OFPassUniforms blurUniforms(int w, int h, const std::array<float, 3>& sigma, int mask) {
        OFPassUniforms u{};
        u.width = u.inWidth = w;
        u.height = u.inHeight = h;
        u.factor = 1;
        u.mode = OF_STORE;
        for (int c = 0; c < 3; ++c) {
            u.radius[c] = mask >> c & 1 ? blurRadiusPx(sigma[size_t(c)]) : -1;
            u.tapOffset[c] = int(taps.size());
            if (u.radius[c] > 0) {
                const std::vector<float> t = blurTaps(sigma[size_t(c)]);
                taps.insert(taps.end(), t.begin(), t.end());
            }
        }
        return u;
    }

    // Box down of `img` by f into kSmall0 and by 2f into kSmall2, one read.
    void box2(int img, int f, int mask) {
        const int sw = (w_ + 2 * f - 1) / (2 * f), sh = (h_ + 2 * f - 1) / (2 * f);
        smallPx = std::max(smallPx, size_t((w_ + f - 1) / f) * size_t((h_ + f - 1) / f));
        OFPassUniforms box = base();
        box.width = sw;
        box.height = sh;
        box.factor = f;
        for (int c = 0; c < 3; ++c) box.radius[c] = mask >> c & 1 ? 0 : -1;  // channels to box
        add(Pass::BoxDown2, img, kSmall0, box);
        passes.back().out2 = kSmall2;
    }

    // Box down `img` by f (unless already boxed into `boxed`), then blur with the lobe's reduced
    // sigmas into kSmallRes at `offset`.
    void reduce(int img, int boxed, const PxLobe& lobe, int f, size_t offset, int mask) {
        const int sw = (w_ + f - 1) / f, sh = (h_ + f - 1) / f;
        smallPx = std::max(smallPx, size_t(sw) * size_t(sh));
        if (boxed < 0) {
            OFPassUniforms box = base();
            box.width = sw;
            box.height = sh;
            box.factor = f;
            for (int c = 0; c < 3; ++c) box.radius[c] = mask >> c & 1 ? 0 : -1;  // channels to box
            add(Pass::BoxDown, img, kSmall0, box);
            boxed = kSmall0;
        }
        std::array<float, 3> s;
        for (size_t c = 0; c < 3; ++c) s[c] = reducedSigmaPx(lobe.sigmaPx[c], f);
        OFPassUniforms b = blurUniforms(sw, sh, s, mask);
        addBlurH(boxed, kSmall1, b);
        b.mode = OF_STORE;
        add(Pass::BlurV, kSmall1, kSmallRes, b);
        passes.back().outOffset = offset;
    }

    // Threadgroup memory of a tiled horizontal pass of u (and u2), or 0 when it should not be tiled.
    size_t tileBytes(const OFPassUniforms& u, const OFPassUniforms& u2) const {
        size_t n = 0;
        int rmax = 0;
        for (int c = 0; c < 3; ++c) {
            const int R = std::max(u.radius[c], u2.radius[c]);
            if (R >= 0) n += size_t(OF_HTILE + 2 * R);
            rmax = std::max(rmax, R);
        }
        return rmax < kTiledHMaxRadius && n * sizeof(float) <= tgLimit_ ? n * sizeof(float) : 0;
    }
    bool tiledFits(const OFPassUniforms& u, const OFPassUniforms& u2) const { return tileBytes(u, u2) > 0; }

    // Horizontal blur: from threadgroup memory for small radii, else register-blocked. With u2, a
    // second blur of the same image into out2 from the same tile (the caller checked tiledFits).
    void addBlurH(int in, int out, const OFPassUniforms& u, const OFPassUniforms* u2 = nullptr, int out2 = -1) {
        const size_t bytes = tileBytes(u, u2 ? *u2 : kNoChannels);
        add(bytes ? Pass::BlurHTiled : Pass::BlurHBlock, in, out, u);
        passes.back().tileBytes = bytes;
        passes.back().u2 = u2 ? *u2 : kNoChannels;
        passes.back().out2 = u2 ? out2 : -1;
    }
    static constexpr OFPassUniforms kNoChannels = {0, 0, 0, 0, 1, 0, {-1, -1, -1}, {0, 0, 0}, {0.f, 0.f, 0.f}};
    static constexpr int kTiledHMaxRadius = 16;  // measured crossover at UHD

    static void setWeight(OFPassUniforms& u, const std::array<float, 3>& w) {
        for (int c = 0; c < 3; ++c) u.weight[c] = w[size_t(c)];
    }

    void add(Pass::Kind kind, int in, int out, const OFPassUniforms& u) {
        Pass p;
        p.kind = kind;
        p.in = in;
        p.out = out;
        p.u = u;
        p.stage = stage_;
        passes.push_back(p);
    }
};

// "blurV 3840x2160 r=4,4,9", "fused 3840x2160 lobes: direct preH reduced" and the like, for profiles.
std::string passName(const Pass& p, int gw, int gh) {
    static const char* kNames[] = {"boxDown", "boxDown2", "blurH(tg)", "blurH(block)", "blurV", "up", "fused"};
    std::string n = std::string(kNames[p.kind]) + " " + std::to_string(gw) + "x" + std::to_string(gh);
    if (p.kind == Pass::BlurHTiled || p.kind == Pass::BlurHBlock || p.kind == Pass::BlurV)
        n += " r=" + std::to_string(p.u.radius[0]) + "," + std::to_string(p.u.radius[1]) + "," +
             std::to_string(p.u.radius[2]);
    if (p.kind == Pass::BoxDown || p.kind == Pass::Up) n += " f=" + std::to_string(p.u.factor);
    if (p.kind == Pass::BoxDown2) n += " f=" + std::to_string(p.u.factor) + "," + std::to_string(2 * p.u.factor);
    if (p.kind == Pass::Fused) {
        static const char* kKinds[] = {"direct", "preH", "reduced", "preH2"};
        n += " lobes:";
        for (int l = 0; l < p.nLobes; ++l) n += std::string(" ") + kKinds[(p.kinds >> (2 * l)) & 3];
        if (p.sum >= 0) n += " +sum";
    }
    return n;
}

std::vector<std::array<float, 3>> lobeWeights(const std::vector<PxLobe>& lobes) {
    std::vector<std::array<float, 3>> w;
    for (const PxLobe& l : lobes) w.push_back({l.weight, l.weight, l.weight});
    return w;
}

std::array<float, 3> edgeGains(const std::array<LayerSharpness, 3>& layers) {
    return {float(layers[0].edgeK), float(layers[1].edgeK), float(layers[2].edgeK)};
}

// The stages of renderFilm() after exposeStage, in its order, each spread fused with its combine and
// the pointwise stages that follow it; the last one writes the window of dst. The working image
// starts in kFull0. With `grain`, grainStage runs on the negative density after the edge effect (or
// after development when there is none) and on the print density after print development; with
// Quality Preview both read u from the plates (OF_POST_GRAIN_PLATES).
void planStages(SpatialPlan& plan, const FilmOptics& optics, const SpatialControls& s, bool grain) {
    const int plates = s.grainQuality == GrainQuality::Preview ? OF_POST_GRAIN_PLATES : 0;
    std::vector<Stage> stages;
    // A pointwise stage goes on the last spatial stage before it; with none at all (grain alone) the
    // pointwise stages make one pass of their own, with no lobes.
    const auto post = [&](int bit) {
        if (stages.empty()) stages.push_back({"pointwise", {}, {}, OF_COMBINE_NONE, {0, 0, 0}, 3});
        stages.back().post |= bit;
    };
    if (s.bloom) {
        const auto lobes = bloomPxLobes(s);
        const float a = float(s.bloomAmount);
        stages.push_back({"bloom", lobes, lobeWeights(lobes), OF_COMBINE_BLOOM, {a, 0, 0}, 3});
    }
    if (s.sharpness) {
        const LayerPasses lp = layerPasses(optics.negative, s.pxPerMm);
        stages.push_back({"neg scatter", lp.passes, lp.weights, OF_COMBINE_STORE, {0, 0, 0}, 3});
    }
    if (s.halation) {
        // Only the red of the halo is used (the red layer's light reaches the base), so only red is
        // blurred: the same red as the CPU's three-channel spread, at the same downsample factor.
        const auto lobes = haloLobes(optics, s);
        stages.push_back({"halation", lobes, lobeWeights(lobes), OF_COMBINE_HALATION, halationWeights(optics, s), 1});
    }
    post(OF_POST_DEVELOP_NEG);
    if (s.sharpness)
        stages.push_back({"neg edge", edgeLobe(optics.negative, s.pxPerMm), {{1.f, 1.f, 1.f}}, OF_COMBINE_EDGE,
                          edgeGains(optics.negative), 3});
    if (grain) post(OF_POST_NEG_GRAIN | plates);
    post(OF_POST_PRINT_EXPOSE);
    if (s.sharpness) {
        const LayerPasses lp = layerPasses(optics.print, s.pxPerMm);
        stages.push_back({"print scatter", lp.passes, lp.weights, OF_COMBINE_STORE, {0, 0, 0}, 3});
    }
    post(OF_POST_DEVELOP_PRINT);  // no print edge effect (ruling R14): print density goes straight to finish
    if (grain) post(OF_POST_PRINT_GRAIN | plates);
    post(OF_POST_FINISH);
    int img = kFull0, other = kFull1;
    for (const Stage& st : stages) {
        plan.stage(st, img, other);
        std::swap(img, other);
    }
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
    std::atomic<bool> profiling{false};
    std::atomic<int> directLimit{kDefaultDirectLimit};
    std::atomic<size_t> lastSetBytes{0};
    std::vector<MetalRenderer::PassTime> passTimes;  // guarded by mutex

    // Preview plate sets: the last two used per device, most recent first (`plateSets`), and the sets
    // being generated right now, each with the future its generating render fulfils (`plateBuilding`).
    // Guarded by plateMutex, never held while the GPU works.
    struct PlateBuild {
        uintptr_t device;
        PlateKey key;
        std::shared_future<std::shared_ptr<GpuPlates>> done;
    };
    mutable std::mutex plateMutex;
    std::vector<std::shared_ptr<GpuPlates>> plateSets;
    std::vector<PlateBuild> plateBuilding;
    size_t plateGenerations = 0;  // generations started (tests)
    std::atomic<int> plateSize{kPlateSize};
    static constexpr size_t kPlateSetsPerDevice = 2;

    // The plate set for `key` on the queue's device. A set not yet held is generated on `queue` by the
    // first render to ask for it, which waits for the GPU once; renders asking for the same key
    // meanwhile wait for that generation instead of starting their own, renders asking for others do
    // not. The set is published only after its command buffer has completed. nullptr on failure.
    std::shared_ptr<GpuPlates> platesFor(id<MTLCommandQueue> queue, id<MTLComputePipelineState> makePlate,
                                         const FilmGrain& grain, double pxPerMm, double sizeMul, int size) {
        const uintptr_t dev = reinterpret_cast<uintptr_t>((__bridge void*)queue.device);
        const PlateKey key = plateKey(grain, pxPerMm, sizeMul, size);
        std::promise<std::shared_ptr<GpuPlates>> promise;
        std::shared_future<std::shared_ptr<GpuPlates>> pending;
        {
            std::lock_guard<std::mutex> lock(plateMutex);
            for (auto it = plateSets.begin(); it != plateSets.end(); ++it)
                if ((*it)->device == dev && (*it)->key == key) {
                    std::shared_ptr<GpuPlates> hit = *it;
                    plateSets.erase(it);
                    plateSets.insert(plateSets.begin(), hit);  // most recently used first
                    return hit;
                }
            for (const PlateBuild& b : plateBuilding)
                if (b.device == dev && b.key == key) pending = b.done;
            if (!pending.valid()) {
                plateBuilding.push_back({dev, key, promise.get_future().share()});
                ++plateGenerations;
            }
        }
        if (pending.valid()) return pending.get();  // no lock held while waiting

        std::shared_ptr<GpuPlates> made;
        try {
            made = generatePlates(queue, makePlate, grain, pxPerMm, sizeMul, size);
            if (made) {
                made->device = dev;
                made->key = key;
            }
        } catch (...) {
            made.reset();
        }
        std::vector<std::shared_ptr<GpuPlates>> released;  // freed after the lock is dropped
        {
            std::lock_guard<std::mutex> lock(plateMutex);
            if (made) publish(made, released);
            plateBuilding.erase(std::find_if(plateBuilding.begin(), plateBuilding.end(), [&](const PlateBuild& b) {
                return b.device == dev && b.key == key;
            }));
        }
        promise.set_value(made);
        return made;
    }

    // Puts `set` first among the held sets, dropping any other for its key and its device's sets
    // beyond kPlateSetsPerDevice into `released`. Called with plateMutex held.
    void publish(const std::shared_ptr<GpuPlates>& set, std::vector<std::shared_ptr<GpuPlates>>& released) {
        size_t mine = 0;
        for (auto it = plateSets.begin(); it != plateSets.end();)
            if ((*it)->device == set->device && ((*it)->key == set->key || ++mine >= kPlateSetsPerDevice)) {
                released.push_back(std::move(*it));
                it = plateSets.erase(it);
            } else {
                ++it;
            }
        plateSets.insert(plateSets.begin(), set);
    }

    // makePlates on the GPU: one sp_make_plate dispatch per stage and layer, committed on `queue` and
    // waited for.
    static std::shared_ptr<GpuPlates> generatePlates(id<MTLCommandQueue> queue, id<MTLComputePipelineState> makePlate,
                                                     const FilmGrain& grain, double pxPerMm, double sizeMul, int size) {
        const size_t plateBytes = size_t(size) * size_t(size) * sizeof(uint16_t);
        auto set = std::make_shared<GpuPlates>();
        set->size = size;
        set->plates = [queue.device newBufferWithLength:6 * plateBytes options:MTLResourceStorageModePrivate];
        if (!set->plates) return nullptr;
        set->plates.label = @"OpenFilm grain plates";
        id<MTLCommandBuffer> cmd = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = cmd ? [cmd computeCommandEncoder] : nil;
        if (!enc) return nullptr;
        cmd.label = @"OpenFilm grain plates";
        for (int l = 0; l < 6; ++l) {
            const GrainStage st = l < 3 ? GrainStage::Negative : GrainStage::Print;
            const int c = l % 3;
            const GrainLayer& g = l < 3 ? grain.negative[size_t(c)] : grain.print[size_t(c)];
            const PlateLayerParams p = plateLayerParams(g, pxPerMm, sizeMul, size, st, c);
            const OFPlateUniforms u{p.unit.rPx, p.unit.lambdaPx, p.unit.mu,  p.unit.gaussRms,
                                    p.cellPx,   p.periodCells,   size,       p.seed};
            [enc setBuffer:set->plates offset:size_t(l) * plateBytes atIndex:0];
            [enc setBytes:&u length:sizeof(u) atIndex:1];
            dispatch(enc, makePlate, size, size);
        }
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
        return cmd.status == MTLCommandBufferStatusCompleted ? set : nullptr;
    }

    // sp_fused specialised for one stage shape, built on first use, by (device, shape).
    std::map<std::pair<uintptr_t, std::array<int, 6>>, id<MTLComputePipelineState>> fused;

    id<MTLComputePipelineState> fusedFor(id<MTLDevice> device, id<MTLLibrary> library, const Pass& p) {
        const std::array<int, 6> shape{p.combine, p.post, p.channels, p.sum >= 0 ? 1 : 0, p.nLobes, p.kinds};
        const auto key = std::make_pair(reinterpret_cast<uintptr_t>((__bridge void*)device), shape);
        std::lock_guard<std::mutex> lock(mutex);
        auto it = fused.find(key);
        if (it != fused.end()) return it->second;
        MTLFunctionConstantValues* fc = [MTLFunctionConstantValues new];
        for (int k = 0; k < 6; ++k) {
            if (k == 3) {
                const bool b = p.sum >= 0;
                [fc setConstantValue:&b type:MTLDataTypeBool atIndex:NSUInteger(k)];
            } else {
                [fc setConstantValue:&shape[size_t(k)] type:MTLDataTypeInt atIndex:NSUInteger(k)];
            }
        }
        NSError* e = nil;
        id<MTLFunction> fn = [library newFunctionWithName:@"sp_fused" constantValues:fc error:&e];
        id<MTLComputePipelineState> pso = fn ? [device newComputePipelineStateWithFunction:fn error:&e] : nil;
        if (pso) fused[key] = pso;
        return pso;
    }

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
        p.library = library;
        if (!build("film_kernel", &p.film) || !build("sp_expose", &p.expose) || !build("sp_box_down", &p.boxDown) ||
            !build("sp_box_down2", &p.boxDown2) || !build("sp_up", &p.up) ||
            !build("sp_blur_h_tiled", &p.blurHTiled) || !build("sp_blur_h_block", &p.blurHBlock) ||
            !build("sp_blur_v_block", &p.blurV) || !build("sp_finish_empty", &p.finish) ||
            !build("sp_make_plate", &p.makePlate))
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

MetalRenderer::PoolStats MetalRenderer::poolStats() const {
    PoolStats st;
    st.idleMaxPerDevice = impl_->pool->idleMaxPerDevice();
    st.inFlight = impl_->pool->inFlight();
    st.peakInFlight = impl_->pool->peakInFlight();
    st.lastSetBytes = impl_->lastSetBytes.load();
    return st;
}

void MetalRenderer::resetPoolPeak() { impl_->pool->resetPeak(); }

void MetalRenderer::setPassProfiling(bool on) { impl_->profiling = on; }

void MetalRenderer::setDirectBlurLimit(int reads) { impl_->directLimit = reads; }

std::vector<MetalRenderer::PassTime> MetalRenderer::lastPassTimes() const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    return impl_->passTimes;
}

size_t MetalRenderer::cachedPlateSetCount() const {
    std::lock_guard<std::mutex> lock(impl_->plateMutex);
    return impl_->plateSets.size();
}

size_t MetalRenderer::plateGenerations() const {
    std::lock_guard<std::mutex> lock(impl_->plateMutex);
    return impl_->plateGenerations;
}

void MetalRenderer::setPlateSizeForTests(int size) { impl_->plateSize = size > 0 ? size : kPlateSize; }

std::vector<float> MetalRenderer::debugPlate(void* commandQueue, const BakedFilm& film, double pxPerMm, double sizeMul,
                                             int stage, int layer) {
    if (!impl_->valid || !commandQueue || stage < 0 || stage > 1 || layer < 0 || layer > 2) return {};
    @autoreleasepool {
        id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)commandQueue;
        std::string err;
        Pipelines pl;
        if (!impl_->pipelinesFor(queue.device, &pl, &err)) return {};
        const auto set = impl_->platesFor(queue, pl.makePlate, film.grain, pxPerMm, sizeMul, impl_->plateSize.load());
        if (!set) return {};
        const size_t n = size_t(set->size) * size_t(set->size), bytes = n * sizeof(uint16_t);
        id<MTLBuffer> out = [queue.device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        id<MTLCommandBuffer> cmd = [queue commandBuffer];
        id<MTLBlitCommandEncoder> blit = [cmd blitCommandEncoder];
        if (!out || !blit) return {};
        [blit copyFromBuffer:set->plates sourceOffset:size_t(stage * 3 + layer) * bytes toBuffer:out destinationOffset:0
                        size:bytes];
        [blit endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
        const _Float16* h = static_cast<const _Float16*>(out.contents);
        return std::vector<float>(h, h + n);
    }
}

bool MetalRenderer::setPlatesForTests(void* commandQueue, const BakedFilm& film, double pxPerMm, double sizeMul,
                                      const GrainPlates& plates) {
    if (!commandQueue || plates.size <= 0) return false;
    @autoreleasepool {
        id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)commandQueue;
        const size_t n = size_t(plates.size) * size_t(plates.size);
        std::vector<uint16_t> all;
        all.reserve(6 * n);
        for (const auto& stage : plates.plates)
            for (const auto& p : stage) {
                if (p.size() != n) return false;
                all.insert(all.end(), p.begin(), p.end());
            }
        auto set = std::make_shared<GpuPlates>();
        set->device = reinterpret_cast<uintptr_t>((__bridge void*)queue.device);
        set->key = plateKey(film.grain, pxPerMm, sizeMul, plates.size);
        set->size = plates.size;
        set->plates = [queue.device newBufferWithBytes:all.data() length:all.size() * sizeof(uint16_t)
                                               options:MTLResourceStorageModeShared];
        if (!set->plates) return false;
        std::vector<std::shared_ptr<GpuPlates>> released;
        std::lock_guard<std::mutex> lock(impl_->plateMutex);
        impl_->publish(set, released);
    }
    return true;
}

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
    // Decided on the sanitised controls, as renderFilm decides (a NaN Grain Amount means the default, 1).
    if (!sanitize(spatial).anyOn()) return render(commandQueue, srcBuffer, srcGeom, dstBuffer, dstGeom, win, film, look);
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

        // Grain (grainStage's per-frame constants, from the CPU's own functions). Preview reads the
        // plate set of this scale and Size, generated here first if no render has made it yet.
        const bool grain = s.grainOn() && film->grain.valid;
        OFGrainUniforms gu{};
        std::shared_ptr<GpuPlates> plates;
        if (grain) {
            const GrainAmplitudes amps = grainAmplitudes(film->grain, s.grainSize);
            gu.amount = float(s.grainAmount);
            if (s.grainQuality == GrainQuality::Preview && W > 0 && H > 0) {
                plates = impl_->platesFor(queue, pl.makePlate, film->grain, s.pxPerMm, s.grainSize,
                                          impl_->plateSize.load());
                if (!plates) return false;
                gu.plateSize = plates->size;
            }
            for (int l = 0; l < 6; ++l) {
                const GrainStage st = l < 3 ? GrainStage::Negative : GrainStage::Print;
                const int c = l % 3;
                const GrainLayer& gl = l < 3 ? film->grain.negative[size_t(c)] : film->grain.print[size_t(c)];
                const GrainUnitParams p = grainUnitParams(gl, s.pxPerMm, s.grainSize);
                OFGrainLayer& L = gu.layer[l];
                L.rPx = p.rPx;
                L.lambdaPx = p.lambdaPx;
                L.mu = p.mu;
                L.gaussRms = p.gaussRms;
                L.k = float(amps.stage(st)[size_t(c)].k);
                L.nodeOffset = g->grainOffset[l];
                L.nodeCount = g->grainCount[l];
                L.binScale = g->grainBinScale[l];
                L.seed = grainSeed(s.frameSeed, st, c);
                if (plates) {  // this frame's view of the plates, as GrainPlates::sample takes it
                    const OFPlateView v = of_plate_view(L.seed, plates->size);
                    L.plateOx = v.ox;
                    L.plateOy = v.oy;
                    L.plateOrient = v.orient;
                }
            }
        }

        SpatialPlan plan(W, H, ww, wh, std::min<size_t>(device.maxThreadgroupMemoryLength, 32768),
                         impl_->directLimit.load());
        std::shared_ptr<WorkSet> set;
        id<MTLBuffer> tapsBuf = nil;
        if (W > 0 && H > 0) {
            planStages(plan, film->optics, s, grain);
            const size_t fullPx = size_t(W) * size_t(H);
            if (fullPx * 3 * sizeof(float) > device.maxBufferLength) return false;
            set = impl_->pool->take(device, fullPx, plan.smallPx, plan.resPx);
            if (!set) return false;
            impl_->lastSetBytes = set->bytes();
            for (Pass& p : plan.passes)
                if (p.kind == Pass::Fused && !(p.pso = impl_->fusedFor(device, pl.library, p))) {
                    impl_->pool->give(set);
                    return false;
                }
            if (plan.taps.empty()) plan.taps.push_back(0.f);
            tapsBuf = upload(device, plan.taps);
            if (!tapsBuf) {
                impl_->pool->give(set);
                return false;
            }
        }
        const auto slot = [&](int i) -> id<MTLBuffer> {
            if (i == kDst) return dst;
            return i < kSmall0 ? set->full[i] : set->small[i - kSmall0];
        };

        std::function<void(id<MTLComputeCommandEncoder>, const Pass&)> encodePass;
        // Every GPU step of the render, in order: expose, the plan's passes, finish.
        struct Step {
            std::string name;
            std::function<void(id<MTLComputeCommandEncoder>)> encode;
        };
        std::vector<Step> steps;
        if (set) {
            steps.push_back({"expose", [&](id<MTLComputeCommandEncoder> enc) {
                                 [enc setBuffer:src offset:0 atIndex:0];
                                 [enc setBuffer:slot(kFull0) offset:0 atIndex:1];
                                 [enc setBytes:&u length:sizeof(u) atIndex:2];
                                 [enc setBuffer:g->lutE4h offset:0 atIndex:3];
                                 dispatch(enc, pl.expose, W, H);
                             }});
            for (const Pass& p : plan.passes)
                steps.push_back({std::string(p.stage) + " " +
                                     passName(p, p.kind == Pass::Fused ? plan.gridW(p) : p.u.width,
                                              p.kind == Pass::Fused ? plan.gridH(p) : p.u.height),
                                 [&, pp = &p](id<MTLComputeCommandEncoder> enc) { encodePass(enc, *pp); }});
        }
        if (!set)  // empty work rect: finishStage alone (every window pixel becomes 0)
            steps.push_back({"finish", [&](id<MTLComputeCommandEncoder> enc) {
                             [enc setBuffer:dst offset:0 atIndex:1];
                             [enc setBytes:&u length:sizeof(u) atIndex:2];
                             dispatch(enc, pl.finish, ww, wh);
                         }});
        // Binds and dispatches one plan pass.
        encodePass = [&](id<MTLComputeCommandEncoder> enc, const Pass& p) {
            switch (p.kind) {
                case Pass::Fused: {
                    [enc setComputePipelineState:p.pso];
                    [enc setBuffer:slot(p.in) offset:0 atIndex:0];
                    [enc setBuffer:slot(p.out) offset:0 atIndex:1];
                    [enc setBytes:&p.fu length:sizeof(p.fu) atIndex:2];
                    [enc setBuffer:tapsBuf offset:0 atIndex:3];
                    [enc setBuffer:slot(kFull2) offset:0 atIndex:4];
                    [enc setBuffer:slot(kSmallRes) offset:0 atIndex:5];
                    [enc setBuffer:slot(p.sum >= 0 ? p.sum : kFull2) offset:0 atIndex:6];
                    [enc setBuffer:slot(kFull3) offset:0 atIndex:13];
                    [enc setBytes:&u length:sizeof(u) atIndex:7];
                    [enc setBuffer:g->curves offset:0 atIndex:8];
                    [enc setBuffer:g->lutPE4 offset:0 atIndex:9];
                    [enc setBuffer:g->lutD4 offset:0 atIndex:10];
                    [enc setBuffer:src offset:0 atIndex:11];
                    [enc setBuffer:dst offset:0 atIndex:12];
                    [enc setBytes:&gu length:sizeof(gu) atIndex:14];
                    [enc setBuffer:g->grainNodes offset:0 atIndex:15];
                    [enc setBuffer:plates ? plates->plates : g->grainNodes offset:0 atIndex:16];
                    [enc setBuffer:g->grainBins offset:0 atIndex:17];
                    // One thread per OF_BLOCK pixels of a column.
                    [enc dispatchThreads:MTLSizeMake(NSUInteger(plan.gridW(p)),
                                                     NSUInteger((plan.gridH(p) + OF_BLOCK - 1) / OF_BLOCK), 1)
                        threadsPerThreadgroup:threadgroupFor(p.pso)];
                    return;
                }
                case Pass::BoxDown:
                case Pass::BoxDown2:
                case Pass::Up:
                    [enc setBuffer:slot(p.in) offset:p.inOffset * sizeof(float) atIndex:0];
                    [enc setBuffer:slot(p.out) offset:0 atIndex:1];
                    [enc setBytes:&p.u length:sizeof(p.u) atIndex:2];
                    if (p.kind == Pass::BoxDown2) [enc setBuffer:slot(p.out2) offset:0 atIndex:3];
                    dispatch(enc, p.kind == Pass::Up ? pl.up : p.kind == Pass::BoxDown ? pl.boxDown : pl.boxDown2,
                             p.u.width, p.kind == Pass::Up ? p.u.height : 3 * p.u.height);  // box: per channel
                    return;
                case Pass::BlurHTiled:
                case Pass::BlurHBlock:
                case Pass::BlurV:
                    [enc setBuffer:slot(p.in) offset:0 atIndex:0];
                    [enc setBuffer:slot(p.out) offset:p.outOffset * sizeof(float) atIndex:1];
                    [enc setBytes:&p.u length:sizeof(p.u) atIndex:2];
                    [enc setBuffer:tapsBuf offset:0 atIndex:3];
                    if (p.kind == Pass::BlurHTiled) {
                        [enc setBuffer:slot(p.out2 >= 0 ? p.out2 : p.out) offset:0 atIndex:4];
                        [enc setBytes:&p.u2 length:sizeof(p.u2) atIndex:5];
                        [enc setComputePipelineState:pl.blurHTiled];
                        [enc setThreadgroupMemoryLength:(p.tileBytes + 15) / 16 * 16 atIndex:0];
                        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger(p.u.width) + OF_HTILE - 1) / OF_HTILE,
                                                              NSUInteger(p.u.height), 1)
                            threadsPerThreadgroup:MTLSizeMake(OF_HTILE, 1, 1)];
                    } else if (p.kind == Pass::BlurHBlock) {  // x: rows, y: blocks of OF_HBLOCK pixels
                        dispatch(enc, pl.blurHBlock, p.u.height, (p.u.width + OF_HBLOCK - 1) / OF_HBLOCK);
                    } else {
                        dispatch(enc, pl.blurV, p.u.width, (p.u.height + OF_BLOCK - 1) / OF_BLOCK);
                    }
                    return;
            }
        };

        bool profiling = impl_->profiling.load();
        if (profiling) {
            // One command buffer per step, each waited for and timed (diagnostics only).
            std::vector<PassTime> times;
            bool good = true;
            for (const Step& st : steps) {
                id<MTLCommandBuffer> cmd = [queue commandBuffer];
                id<MTLComputeCommandEncoder> enc = cmd ? [cmd computeCommandEncoder] : nil;
                if (!enc) {
                    good = false;
                    break;
                }
                st.encode(enc);
                [enc endEncoding];
                [cmd commit];
                [cmd waitUntilCompleted];
                times.push_back({st.name, (cmd.GPUEndTime - cmd.GPUStartTime) * 1e3});
            }
            if (set) impl_->pool->give(set);
            std::lock_guard<std::mutex> lock(impl_->mutex);
            impl_->passTimes = std::move(times);
            return good;
        }

        id<MTLCommandBuffer> cmd = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = cmd ? [cmd computeCommandEncoder] : nil;
        if (!enc) {
            if (set) impl_->pool->give(set);
            return false;  // report failure rather than leave the frame unwritten
        }
        cmd.label = @"OpenFilm spatial";
        for (const Step& st : steps) st.encode(enc);
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
