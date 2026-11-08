// MetalRenderer: runs OpenFilm's GPU kernels on Resolve's Metal command queue.
//
// When Metal rendering is on, Resolve does not give us CPU memory. It gives us an
// id<MTLCommandQueue> (the GPU's to-do list) and id<MTLBuffer>s (frames living in GPU memory).
// We add our kernels to that queue and return; Resolve waits for the GPU itself.
//
// This header is plain C++ so the OFX layer can include it. The Objective-C objects are passed as
// void* and the Metal details stay hidden in metal_renderer.mm.
#pragma once

#include "core/image.hpp"
#include "core/render_params.hpp"

#include <memory>
#include <string>

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

    // Encodes the pipeline onto `commandQueue` (id<MTLCommandQueue>) reading `srcBuffer` and
    // writing `dstBuffer` (both id<MTLBuffer>). Commits without waiting. Thread-safe.
    bool render(void* commandQueue, void* srcBuffer, const ImageGeometry& srcGeom, void* dstBuffer,
                const ImageGeometry& dstGeom, const RenderWindow& win, const RenderParams& p);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace openfilm
