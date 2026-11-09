// GPU twin of renderFilm() with spatial effects on (src/core/film_spatial.cpp and the blur engine
// in src/core/blur.cpp); any change there must be made here too. One kernel per CPU stage or blur
// pass; metal_renderer.mm encodes them in the CPU's order. tests/gpu/test_gpu_vs_cpu.mm renders
// both and checks they agree.
//
// Every per-pixel kernel runs one thread per pixel of the grid it writes (gid = column, row) and
// does the CPU's float operations in the CPU's order: blur taps are summed from -r to r, box
// averages row by row, lobes accumulate in lobe order. The blur taps themselves are computed on
// the host by the CPU's own blurTaps() and uploaded, so both sides weight pixels identically.
// Alpha is not carried through the work images: the finish pass copies it from the source.
#include <metal_stdlib>
#include "metal/film_common.h"

using namespace metal;

namespace {

inline int clampI(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }

inline bool outside(uint2 gid, int w, int h) { return int(gid.x) >= w || int(gid.y) >= h; }

// One channel of a 1D blur along x (dir 0) or y (dir 1), clamp-to-edge; radius 0 copies.
inline float blur1(device const float4* in, device const float* taps, int w, int h, int x, int y, int c, int r,
                   int off, int dir) {
    if (r == 0) return in[y * w + x][c];
    float acc = 0.0f;
    for (int i = -r; i <= r; ++i) {
        const int xx = dir == 0 ? clampI(x + i, 0, w - 1) : x;
        const int yy = dir == 0 ? y : clampI(y + i, 0, h - 1);
        acc += taps[off + i + r] * in[yy * w + xx][c];
    }
    return acc;
}

// Writes v by the pass mode (OF_STORE / OF_STORE_WEIGHTED / OF_ACCUMULATE); alpha from `a`.
inline void put(device float4* out, int i, float3 v, constant OFPassUniforms& p, float a) {
    if (p.mode == OF_STORE) {
        out[i] = float4(v, a);
        return;
    }
    float4 o = p.mode == OF_ACCUMULATE ? out[i] : float4(0.0f, 0.0f, 0.0f, a);
    for (int c = 0; c < 3; ++c) {
        const float wv = p.weight[c] * v[c];
        o[c] = p.mode == OF_ACCUMULATE ? o[c] + wv : wv;
    }
    out[i] = o;
}

}  // namespace

// exposeStage: DI -> linear exposure (10^lutE - eps) * gain over the work rect.
kernel void sp_expose(device const float* src [[buffer(0)]],
                      device float4* E [[buffer(1)]],
                      constant OFSpatialUniforms& u [[buffer(2)]],
                      device const float* lutE [[buffer(3)]],
                      uint2 gid [[thread_position_in_grid]]) {
    if (outside(gid, u.width, u.height)) return;
    const int x = u.rectX1 + int(gid.x), y = u.rectY1 + int(gid.y);
    device const float* in = src + (y - u.src.y1) * u.src.rowFloats + (x - u.src.x1) * 4;
    const float3 e = lookup(lutE, u.lutE, float3(sane(in[0]), sane(in[1]), sane(in[2])));
    float4 o;
    for (int c = 0; c < 3; ++c) o[c] = (precise::pow(10.0f, e[c]) - OF_EXPOSURE_LUT_EPS) * u.gain;
    o[3] = in[3];
    E[int(gid.y) * u.width + int(gid.x)] = o;
}

// blurImpl, horizontal half: out = taps (x) in along x, per channel.
kernel void sp_blur_h(device const float4* in [[buffer(0)]],
                      device float4* out [[buffer(1)]],
                      constant OFPassUniforms& p [[buffer(2)]],
                      device const float* taps [[buffer(3)]],
                      uint2 gid [[thread_position_in_grid]]) {
    if (outside(gid, p.width, p.height)) return;
    const int x = int(gid.x), y = int(gid.y), i = y * p.width + x;
    float4 o;
    for (int c = 0; c < 3; ++c) o[c] = blur1(in, taps, p.width, p.height, x, y, c, p.radius[c], p.tapOffset[c], 0);
    o[3] = in[i][3];
    out[i] = o;
}

// blurImpl, vertical half, then stored / weighted / accumulated into out (spread's lobe sum).
kernel void sp_blur_v(device const float4* in [[buffer(0)]],
                      device float4* out [[buffer(1)]],
                      constant OFPassUniforms& p [[buffer(2)]],
                      device const float* taps [[buffer(3)]],
                      uint2 gid [[thread_position_in_grid]]) {
    if (outside(gid, p.width, p.height)) return;
    const int x = int(gid.x), y = int(gid.y), i = y * p.width + x;
    float3 v;
    for (int c = 0; c < 3; ++c) v[c] = blur1(in, taps, p.width, p.height, x, y, c, p.radius[c], p.tapOffset[c], 1);
    put(out, i, v, p, in[i][3]);
}

// boxDown: f x f average; partial edge blocks average only their real pixels.
kernel void sp_box_down(device const float4* in [[buffer(0)]],
                        device float4* out [[buffer(1)]],
                        constant OFPassUniforms& p [[buffer(2)]],
                        uint2 gid [[thread_position_in_grid]]) {
    if (outside(gid, p.width, p.height)) return;
    const int x = int(gid.x), y = int(gid.y), f = p.factor;
    const int xe = min((x + 1) * f, p.inWidth), ye = min((y + 1) * f, p.inHeight);
    float4 s = float4(0.0f);
    for (int yy = y * f; yy < ye; ++yy)
        for (int xx = x * f; xx < xe; ++xx) s += in[yy * p.inWidth + xx];
    const float inv = 1.0f / float((xe - x * f) * (ye - y * f));
    out[y * p.width + x] = s * inv;
}

// upAccumulate: bilinear at ((x + 0.5) / f - 0.5), clamp-to-edge, weighted into out.
kernel void sp_up(device const float4* small [[buffer(0)]],
                  device float4* out [[buffer(1)]],
                  constant OFPassUniforms& p [[buffer(2)]],
                  uint2 gid [[thread_position_in_grid]]) {
    if (outside(gid, p.width, p.height)) return;
    const int x = int(gid.x), y = int(gid.y), sw = p.inWidth, sh = p.inHeight;
    const float ff = float(p.factor);
    const float fy = (float(y) + 0.5f) / ff - 0.5f;
    const float flY = floor(fy);
    const float ty = fy - flY;
    const int ya = clampI(int(flY), 0, sh - 1), yb = clampI(int(flY) + 1, 0, sh - 1);
    const float fx = (float(x) + 0.5f) / ff - 0.5f;
    const float flX = floor(fx);
    const float tx = fx - flX;
    const int xa = clampI(int(flX), 0, sw - 1), xb = clampI(int(flX) + 1, 0, sw - 1);
    const float4 p00 = small[ya * sw + xa], p10 = small[ya * sw + xb];
    const float4 p01 = small[yb * sw + xa], p11 = small[yb * sw + xb];
    float3 v;
    for (int c = 0; c < 3; ++c) {
        const float top = p00[c] + (p10[c] - p00[c]) * tx;
        const float bot = p01[c] + (p11[c] - p01[c]) * tx;
        v[c] = top + (bot - top) * ty;
    }
    const int i = y * p.width + x;
    put(out, i, v, p, out[i][3]);
}

// A spread with no lobes: RGB 0.
kernel void sp_clear(device float4* out [[buffer(1)]],
                     constant OFPassUniforms& p [[buffer(2)]],
                     uint2 gid [[thread_position_in_grid]]) {
    if (outside(gid, p.width, p.height)) return;
    const int i = int(gid.y) * p.width + int(gid.x);
    out[i] = float4(0.0f, 0.0f, 0.0f, out[i][3]);
}

// Bloom: E <- (1 - a) E + a B, a = weight[0].
kernel void sp_bloom_mix(device float4* E [[buffer(0)]],
                         device const float4* B [[buffer(1)]],
                         constant OFPassUniforms& p [[buffer(2)]],
                         uint2 gid [[thread_position_in_grid]]) {
    if (outside(gid, p.width, p.height)) return;
    const int i = int(gid.y) * p.width + int(gid.x);
    const float a = p.weight[0];
    float4 e = E[i];
    const float4 b = B[i];
    for (int c = 0; c < 3; ++c) e[c] = (1.0f - a) * e[c] + a * b[c];
    E[i] = e;
}

// Halation: E_c <- E_c + hw_c * H_r.
kernel void sp_halation_add(device float4* E [[buffer(0)]],
                            device const float4* H [[buffer(1)]],
                            constant OFPassUniforms& p [[buffer(2)]],
                            uint2 gid [[thread_position_in_grid]]) {
    if (outside(gid, p.width, p.height)) return;
    const int i = int(gid.y) * p.width + int(gid.x);
    const float r = H[i][0];
    float4 e = E[i];
    for (int c = 0; c < 3; ++c) e[c] += p.weight[c] * r;
    E[i] = e;
}

// developStage (mode 0): negCurves(log10(max(E, 1e-30))).
// printDevelopStage (mode 1): printCurves(log10(max(P, 1e-30)) + offsets).
kernel void sp_develop(device float4* img [[buffer(0)]],
                       constant OFPassUniforms& p [[buffer(2)]],
                       constant OFSpatialUniforms& u [[buffer(3)]],
                       device const float* curves [[buffer(4)]],
                       uint2 gid [[thread_position_in_grid]]) {
    if (outside(gid, p.width, p.height)) return;
    const int i = int(gid.y) * p.width + int(gid.x);
    float4 v = img[i];
    for (int c = 0; c < 3; ++c) {
        const float le = precise::log10(stdMax(v[c], 1e-30f));
        v[c] = p.mode == 0 ? evalCurve(curves, u.neg[c], le) : evalCurve(curves, u.print[c], le + u.offsets[c]);
    }
    img[i] = v;
}

// Edge effect (negative or print density): D_c <- D_c + k_c (D_c - B_c).
kernel void sp_edge(device float4* D [[buffer(0)]],
                    device const float4* B [[buffer(1)]],
                    constant OFPassUniforms& p [[buffer(2)]],
                    uint2 gid [[thread_position_in_grid]]) {
    if (outside(gid, p.width, p.height)) return;
    const int i = int(gid.y) * p.width + int(gid.x);
    float4 d = D[i];
    const float4 b = B[i];
    for (int c = 0; c < 3; ++c) d[c] += p.weight[c] * (d[c] - b[c]);
    D[i] = d;
}

// printExposeStage: negative density -> P = 10^lutPE(D).
kernel void sp_print_expose(device float4* img [[buffer(0)]],
                            constant OFPassUniforms& p [[buffer(2)]],
                            constant OFSpatialUniforms& u [[buffer(3)]],
                            device const float* lutPE [[buffer(4)]],
                            uint2 gid [[thread_position_in_grid]]) {
    if (outside(gid, p.width, p.height)) return;
    const int i = int(gid.y) * p.width + int(gid.x);
    float4 v = img[i];
    const float3 pe = lookup(lutPE, u.lutPE, v.rgb);
    for (int c = 0; c < 3; ++c) v[c] = precise::pow(10.0f, pe[c]);
    img[i] = v;
}

// finishStage: one thread per window pixel. Print density -> lutD -> 10^q -> display matrix ->
// encode; alpha from the source. Window pixels outside the work rect (outside the source) -> 0.
kernel void sp_finish(device const float4* Dp [[buffer(0)]],
                      device float* dst [[buffer(1)]],
                      constant OFSpatialUniforms& u [[buffer(2)]],
                      device const float* lutD [[buffer(3)]],
                      device const float* src [[buffer(4)]],
                      uint2 gid [[thread_position_in_grid]]) {
    const int x = u.winX1 + int(gid.x);
    const int y = u.winY1 + int(gid.y);
    if (x >= u.winX2 || y >= u.winY2) return;
    if (!inside(u.dst, x, y)) return;  // window pixel the host gave us no room for
    device float* out = dst + (y - u.dst.y1) * u.dst.rowFloats + (x - u.dst.x1) * 4;
    const int wx = x - u.rectX1, wy = y - u.rectY1;
    if (wx < 0 || wy < 0 || wx >= u.width || wy >= u.height || !inside(u.src, x, y)) {
        out[0] = 0.0f; out[1] = 0.0f; out[2] = 0.0f; out[3] = 0.0f;
        return;
    }
    const float4 pd = Dp[wy * u.width + wx];
    const float3 q = lookup(lutD, u.lutD, pd.rgb);
    const float xyz[3] = {precise::pow(10.0f, q[0]), precise::pow(10.0f, q[1]), precise::pow(10.0f, q[2])};
    for (int k = 0; k < 3; ++k)
        out[k] = encodeDisplay(u.displayMatrix[k * 3] * xyz[0] + u.displayMatrix[k * 3 + 1] * xyz[1] +
                               u.displayMatrix[k * 3 + 2] * xyz[2]);
    out[3] = src[(y - u.src.y1) * u.src.rowFloats + (x - u.src.x1) * 4 + 3];
}
