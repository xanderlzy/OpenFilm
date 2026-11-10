// MetalRenderer: runs OpenFilm's GPU kernels on Resolve's Metal command queue.
//
// When Metal rendering is on, Resolve does not give us CPU memory. It gives us an
// id<MTLCommandQueue> (the GPU's to-do list) and id<MTLBuffer>s (frames living in GPU memory).
// We add our kernels to that queue and return; Resolve waits for the GPU itself.
//
// This header is plain C++ so the OFX layer can include it. The Objective-C objects are passed as
// void* and the Metal details stay hidden in metal_renderer.mm.
#pragma once

#include "core/film_spatial.hpp"
#include "core/grain_plates.hpp"
#include "core/image.hpp"
#include "core/lut.hpp"

#include <cstddef>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace openfilm {

class MetalRenderer {
public:
    // Loads the compiled shader library (OpenFilm.metallib). Check isValid() afterwards.
    explicit MetalRenderer(const std::string& metallibPath);
    ~MetalRenderer();
    MetalRenderer(const MetalRenderer&) = delete;
    MetalRenderer& operator=(const MetalRenderer&) = delete;

    bool isValid() const;
    const std::string& error() const;  // why isValid() is false

    // Encodes the film kernel (the GPU twin of renderFilmCpu) onto `commandQueue`
    // (id<MTLCommandQueue>) reading `srcBuffer` and writing `dstBuffer` (both id<MTLBuffer>), with
    // the same window, bounds and alpha rules as the CPU. Commits without waiting. The film's tables
    // are uploaded to the GPU on first use and kept while the bake is alive. Thread-safe.
    bool render(void* commandQueue, void* srcBuffer, const ImageGeometry& srcGeom, void* dstBuffer,
                const ImageGeometry& dstGeom, const RenderWindow& win, const std::shared_ptr<const BakedFilm>& film,
                const LookControls& controls);

    // The M4 path: the GPU twin of renderFilm() (film_spatial.hpp), pass for pass. Calls the overload
    // above unchanged when !sanitize(spatial).anyOn(). Otherwise works on the window grown by
    // spatialMarginPx (cut to the source, see workRect) in pooled GPU buffers that go back to the
    // pool when the command buffer completes. Commits without waiting. Thread-safe. Preview grain
    // at a scale and Size with no plate set yet first generates one on `commandQueue` and waits for
    // it (about 0.2 s at kPlateSize); renders asking for that set meanwhile wait for the same one.
    bool render(void* commandQueue, void* srcBuffer, const ImageGeometry& srcGeom, void* dstBuffer,
                const ImageGeometry& dstGeom, const RenderWindow& win, const std::shared_ptr<const BakedFilm>& film,
                const LookControls& look, const SpatialControls& spatial);

    // Number of film table sets currently held on the GPU (for tests).
    size_t cachedFilmCount() const;

    // Preview grain plate sets held on the GPU (all devices), and how many generations this renderer
    // has started so far (tests).
    size_t cachedPlateSetCount() const;
    size_t plateGenerations() const;
    // Plate side for Preview grain (default kPlateSize; 0 or less restores it). Tests only.
    void setPlateSizeForTests(int size);
    // Tests only: `hook` runs once a plate set is generated, before it is published; it may throw to
    // simulate a failure there (an empty function removes it).
    void setPlateBuildHookForTests(std::function<void()> hook);
    // One plate of the set for (film's grain, pxPerMm, sizeMul) at the current plate size, generated on
    // `commandQueue` if not cached, read back (tests): stage 0 negative / 1 print, layer 0..2.
    std::vector<float> debugPlate(void* commandQueue, const BakedFilm& film, double pxPerMm, double sizeMul, int stage,
                                  int layer);
    // Tests: hold `plates` (e.g. the CPU's own, cpuPlates) as the set for (film's grain, pxPerMm, sizeMul,
    // plates.size) on the queue's device, in place of any set held for that key, so a render reads
    // exactly the plates the CPU reads.
    bool setPlatesForTests(void* commandQueue, const BakedFilm& film, double pxPerMm, double sizeMul,
                           const GrainPlates& plates);

    // ---- Diagnostics (tests and benchmarks) ----
    // The intermediate-buffer pool: the most idle work sets any one device holds, how many sets are
    // out with renders whose command buffers have not completed yet, and the bytes of the set the
    // last spatial render used.
    struct PoolStats {
        size_t idleMaxPerDevice = 0;
        size_t inFlight = 0;
        size_t peakInFlight = 0;  // most sets out on any one device at once since resetPoolPeak()
        size_t lastSetBytes = 0;
    };
    PoolStats poolStats() const;
    void resetPoolPeak();

    // When on, the spatial overload commits every pass as its own command buffer, waits for each, and
    // records its GPU time (so render() blocks). Off by default; for profiling only.
    struct PassTime {
        std::string name;
        double ms;
    };
    void setPassProfiling(bool on);
    // Largest cost (image reads per output pixel) of a full-resolution lobe whose horizontal pass the
    // fused pass computes itself; costlier ones get their own horizontal pass, and a stage with more
    // than one of those sums the rest in separate passes first. Tests lower it to cover those paths.
    void setDirectBlurLimit(int reads);
    std::vector<PassTime> lastPassTimes() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace openfilm
