#include "metal/metal_renderer.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "metal/shader_types.h"

#include <cmath>
#include <mutex>
#include <unordered_map>

namespace openfilm {

namespace {

OFImageGeom toShader(const ImageGeometry& g) {
    return {g.x1, g.y1, g.x2, g.y2, int(g.rowBytes / 4)};
}

}  // namespace

struct MetalRenderer::Impl {
    std::string path;
    std::string error;
    bool valid = false;

    // A Mac can have several GPUs, and compiled kernels ("pipeline states") belong to one GPU.
    // We build them lazily for whichever device Resolve's queue lives on, and cache them.
    std::mutex mutex;
    std::unordered_map<uintptr_t, id<MTLComputePipelineState>> exposurePipelines;

    id<MTLComputePipelineState> pipelineFor(id<MTLDevice> device, std::string* err) {
        std::lock_guard<std::mutex> lock(mutex);
        auto key = reinterpret_cast<uintptr_t>((__bridge void*)device);
        auto it = exposurePipelines.find(key);
        if (it != exposurePipelines.end()) return it->second;

        NSError* nsErr = nil;
        NSURL* url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:path.c_str()]];
        id<MTLLibrary> library = [device newLibraryWithURL:url error:&nsErr];
        if (!library) {
            *err = "cannot load " + path + ": " +
                   (nsErr ? std::string(nsErr.localizedDescription.UTF8String) : "unknown error");
            return nil;
        }
        id<MTLFunction> fn = [library newFunctionWithName:@"exposure_kernel"];
        if (!fn) {
            *err = "exposure_kernel missing from " + path;
            return nil;
        }
        id<MTLComputePipelineState> pso = [device newComputePipelineStateWithFunction:fn error:&nsErr];
        if (!pso) {
            *err = "cannot build exposure_kernel: " +
                   (nsErr ? std::string(nsErr.localizedDescription.UTF8String) : "unknown error");
            return nil;
        }
        exposurePipelines[key] = pso;
        return pso;
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

bool MetalRenderer::render(void* commandQueue, void* srcBuffer, const ImageGeometry& srcGeom, void* dstBuffer,
                           const ImageGeometry& dstGeom, const RenderWindow& win, const RenderParams& p) {
    if (!impl_->valid || !commandQueue || !srcBuffer || !dstBuffer) return false;
    const int w = win.x2 - win.x1;
    const int h = win.y2 - win.y1;
    if (w <= 0 || h <= 0) return true;

    @autoreleasepool {
        id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)commandQueue;
        id<MTLBuffer> src = (__bridge id<MTLBuffer>)srcBuffer;
        id<MTLBuffer> dst = (__bridge id<MTLBuffer>)dstBuffer;

        std::string err;
        id<MTLComputePipelineState> pso = impl_->pipelineFor(queue.device, &err);
        if (!pso) return false;

        OFExposureUniforms u{};
        u.src = toShader(srcGeom);
        u.dst = toShader(dstGeom);
        u.winX1 = win.x1;
        u.winY1 = win.y1;
        u.winX2 = win.x2;
        u.winY2 = win.y2;
        u.gain = float(std::exp2(p.exposureStops));  // same float as the CPU path

        id<MTLCommandBuffer> cmd = [queue commandBuffer];
        cmd.label = @"OpenFilm";
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:src offset:0 atIndex:0];
        [enc setBuffer:dst offset:0 atIndex:1];
        [enc setBytes:&u length:sizeof(u) atIndex:2];
        // One GPU thread per window pixel, in 16x16 tiles.
        [enc dispatchThreads:MTLSizeMake(NSUInteger(w), NSUInteger(h), 1)
            threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
        [enc endEncoding];
        [cmd commit];
    }
    return true;
}

}  // namespace openfilm
