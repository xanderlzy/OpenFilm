// GPU twin of renderFilmCpu(); any change there must be made here too.
// (renderRows in src/core/film_renderer.cpp, Lut3D::lookup and Curve1D::eval in src/core/lut.cpp.)
// tests/gpu/test_gpu_vs_cpu.mm renders both and checks they agree.
//
// A Metal "kernel" runs once per GPU thread. We launch one thread per pixel of the render window;
// gid is that thread's (column, row) inside the window. Everything is float, operation for
// operation the same as the CPU, including how out-of-range and non-finite values are handled.
// pow/log10 are the precise versions (and the library is compiled with -fno-fast-math) so they
// track the CPU's std::pow/std::log10.
#include <metal_stdlib>
#include "metal/shader_types.h"

using namespace metal;

namespace {

// std::clamp / std::max / std::min exactly, including what they do with NaN (Metal's own
// clamp/fmax would quietly swap a NaN for a bound).
inline float stdClamp(float v, float lo, float hi) { return v < lo ? lo : (hi < v ? hi : v); }
inline float stdMax(float a, float b) { return a < b ? b : a; }
inline int stdMinI(int a, int b) { return b < a ? b : a; }

// Black for anything that is not a real number, so no NaN can reach the lookups.
inline float sane(float v) { return isfinite(v) ? v : 0.0f; }

inline bool inside(OFImageGeom g, int x, int y) { return x >= g.x1 && x < g.x2 && y >= g.y1 && y < g.y2; }

// Lut3D::lookup: clamp into the table's domain, find the cell, tetrahedral interpolation.
float3 lookup(device const float* data, constant OFLut3D& L, float3 in) {
    const int n = L.size;
    int i0[3];
    float f[3];
    for (int k = 0; k < 3; ++k) {
        float t = stdClamp((in[k] - L.lo[k]) / L.span[k], 0.0f, 1.0f);
        if (!(t == t)) t = 0.0f;
        if (L.denseDarks) t = precise::sqrt(t);  // inverse of the t^2 node spacing (dense shadows)
        const float u = t * float(n - 1);
        const int i = stdMinI(int(u), n - 2);
        i0[k] = i;
        f[k] = u - float(i);
    }
    const uint sr = 3, sg = uint(n) * 3, sb = uint(n) * uint(n) * 3;
    const uint stride[3] = {sr, sg, sb};
    const uint base = uint(i0[2]) * sb + uint(i0[1]) * sg + uint(i0[0]) * sr;

    // Tetrahedral: order the fractional offsets largest first, then walk the cube corner by corner
    // from the cell origin along that order. Ties give a zero weight, so their order never matters.
    int o0 = 0, o1 = 1, o2 = 2, tmp;
    if (f[o1] > f[o0]) { tmp = o0; o0 = o1; o1 = tmp; }
    if (f[o2] > f[o1]) { tmp = o1; o1 = o2; o2 = tmp; }
    if (f[o1] > f[o0]) { tmp = o0; o0 = o1; o1 = tmp; }
    const uint p0 = base;
    const uint p1 = p0 + stride[o0];
    const uint p2 = p1 + stride[o1];
    const uint p3 = p2 + stride[o2];
    const float w0 = 1.0f - f[o0], w1 = f[o0] - f[o1], w2 = f[o1] - f[o2], w3 = f[o2];
    float3 out;
    for (int k = 0; k < 3; ++k)
        out[k] = w0 * data[p0 + k] + w1 * data[p1 + k] + w2 * data[p2 + k] + w3 * data[p3 + k];
    return out;
}

// Curve1D::eval: an H&D curve sampled evenly, linear between samples, flat past either end.
float evalCurve(device const float* curves, constant OFCurve& c, float x) {
    const float u = stdClamp((x - c.lo) / (c.hi - c.lo), 0.0f, 1.0f) * float(c.count - 1);
    const int i = stdMinI(int(u), c.count - 2);
    const float t = u - float(i);
    device const float* v = curves + c.offset;
    return v[i] + (v[i + 1] - v[i]) * t;
}

// The screen's gamma (BT.1886 ~ 1/2.4); light the display cannot show (negative) clips to black.
inline float encodeDisplay(float linear) { return precise::pow(stdMax(linear, 0.0f), 1.0f / 2.4f); }

}  // namespace

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
