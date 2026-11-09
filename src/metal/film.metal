// GPU twin of renderFilmCpu(); any change there must be made here too.
// (renderRows in src/core/film_renderer.cpp; the table lookups are in film_common.h.)
// tests/gpu/test_gpu_vs_cpu.mm renders both and checks they agree.
//
// A Metal "kernel" runs once per GPU thread. We launch one thread per pixel of the render window;
// gid is that thread's (column, row) inside the window. Everything is float, operation for
// operation the same as the CPU, including how out-of-range and non-finite values are handled.
// pow/log10 are the precise versions (and the library is compiled with -fno-fast-math) so they
// track the CPU's std::pow/std::log10.
#include <metal_stdlib>
#include "metal/film_common.h"

using namespace metal;

kernel void film_kernel(device const float* src [[buffer(0)]],
                        device float* dst [[buffer(1)]],
                        constant OFFilmUniforms& u [[buffer(2)]],
                        device const float* lutE [[buffer(3)]],
                        device const float* lutPE [[buffer(4)]],
                        device const float* lutD [[buffer(5)]],
                        device const float* curves [[buffer(6)]],
                        uint2 gid [[thread_position_in_grid]]) {
    const int x = u.winX1 + int(gid.x);
    const int y = u.winY1 + int(gid.y);
    // Edge guard: the last threadgroups can overhang odd-sized windows.
    if (x >= u.winX2 || y >= u.winY2) return;
    if (!inside(u.dst, x, y)) return;  // window pixel the host gave us no room for

    device float* out = dst + (y - u.dst.y1) * u.dst.rowFloats + (x - u.dst.x1) * 4;
    if (!inside(u.src, x, y)) {
        out[0] = 0.0f; out[1] = 0.0f; out[2] = 0.0f; out[3] = 0.0f;
        return;
    }
    device const float* in = src + (y - u.src.y1) * u.src.rowFloats + (x - u.src.x1) * 4;

    // DI code -> table value -> linear exposure on the negative; the Exposure control multiplies
    // that by 2^stops, then log10 gives the log E the negative's curves expect.
    const float3 e = lookup(lutE, u.lutE, float3(sane(in[0]), sane(in[1]), sane(in[2])));
    float le[3];
    for (int c = 0; c < 3; ++c)
        le[c] = precise::log10(stdMax((precise::pow(10.0f, e[c]) - OF_EXPOSURE_LUT_EPS) * u.gain, 1e-30f));
    // Negative's H&D curves: log exposure -> Status M density per layer.
    const float3 d = float3(evalCurve(curves, u.neg[0], le[0]), evalCurve(curves, u.neg[1], le[1]),
                            evalCurve(curves, u.neg[2], le[2]));

    // Negative density -> exposure on the print; the lab's balance, printer lights and Print Density
    // shift it; the print's curves give dye density.
    const float3 pe = lookup(lutPE, u.lutPE, d);
    const float3 pd = float3(evalCurve(curves, u.print[0], pe[0] + u.offsets[0]),
                             evalCurve(curves, u.print[1], pe[1] + u.offsets[1]),
                             evalCurve(curves, u.print[2], pe[2] + u.offsets[2]));

    // Print density -> log of the projected light; undo the log, mix to Rec.709 primaries.
    const float3 q = lookup(lutD, u.lutD, pd);
    const float xyz[3] = {precise::pow(10.0f, q[0]), precise::pow(10.0f, q[1]), precise::pow(10.0f, q[2])};
    float o[3];
    for (int k = 0; k < 3; ++k)
        o[k] = u.displayMatrix[k * 3] * xyz[0] + u.displayMatrix[k * 3 + 1] * xyz[1] + u.displayMatrix[k * 3 + 2] * xyz[2];
    out[0] = encodeDisplay(o[0]);
    out[1] = encodeDisplay(o[1]);
    out[2] = encodeDisplay(o[2]);
    out[3] = in[3];  // alpha passes through untouched
}
