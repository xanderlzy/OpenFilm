// Table lookups and small helpers shared by film.metal (the M3 colour kernel) and spatial.metal
// (the M4 passes). GPU twins of Lut3D::lookup and Curve1D::eval (src/core/lut.cpp) and of the
// helpers in src/core/pixel_ops.hpp; operation for operation the same as the CPU.
#ifndef OPENFILM_FILM_COMMON_H
#define OPENFILM_FILM_COMMON_H

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

#endif
