// GPU version of renderExposureCpu() (src/core/cpu_renderer.cpp). Must stay numerically identical;
// tests/gpu/test_gpu_vs_cpu.mm checks this.
//
// A Metal "kernel" runs once per GPU thread. We launch one thread per pixel of the render
// window; gid is that thread's (column, row) inside the window.
#include <metal_stdlib>
#include "metal/shader_types.h"

using namespace metal;

static inline bool inside(OFImageGeom g, int x, int y) {
    return x >= g.x1 && x < g.x2 && y >= g.y1 && y < g.y2;
}

kernel void exposure_kernel(device const float* src [[buffer(0)]],
                            device float* dst [[buffer(1)]],
                            constant OFExposureUniforms& u [[buffer(2)]],
                            uint2 gid [[thread_position_in_grid]]) {
    const int x = u.winX1 + int(gid.x);
    const int y = u.winY1 + int(gid.y);
    // Edge guard: the last threadgroups can overhang odd-sized windows.
    if (x >= u.winX2 || y >= u.winY2) return;
    if (!inside(u.dst, x, y)) return;

    device float* out = dst + (y - u.dst.y1) * u.dst.rowFloats + (x - u.dst.x1) * 4;
    if (!inside(u.src, x, y)) {
        out[0] = 0.0f; out[1] = 0.0f; out[2] = 0.0f; out[3] = 0.0f;
        return;
    }
    device const float* in = src + (y - u.src.y1) * u.src.rowFloats + (x - u.src.x1) * 4;
    out[0] = in[0] * u.gain;
    out[1] = in[1] * u.gain;
    out[2] = in[2] * u.gain;
    out[3] = in[3];  // alpha untouched
}
