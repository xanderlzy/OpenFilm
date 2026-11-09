// GPU twin of renderFilm() with spatial effects on (src/core/film_spatial.cpp and the blur engine
// in src/core/blur.cpp); any change there must be made here too. metal_renderer.mm plans the passes
// from the CPU's own lobes and taps and encodes them in the CPU's order; tests/gpu/test_gpu_vs_cpu.mm
// renders both and checks they agree.
//
// Every kernel does the CPU's float operations in the CPU's order: blur taps are summed from -r to
// r (one channel at a time), box averages row by row, lobes accumulate in lobe order. The taps are
// computed on the host by the CPU's own blurTaps() and uploaded. What the GPU changes is only where
// the work happens: each spatial stage's spread, its combine and the per-pixel stages after it run
// in one pass (sp_fused), and blurs keep inputs in registers or threadgroup memory instead of
// re-reading them. Alpha is not carried through the work images: the last pass takes it from the
// source.
#include <metal_stdlib>
#include "metal/film_common.h"

using namespace metal;

namespace {

inline int clampI(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }

inline bool outside(uint2 gid, int w, int h) { return int(gid.x) >= w || int(gid.y) >= h; }

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

// upAccumulate's bilinear sample of a coarse image at fine pixel (x, y).
inline float3 upSample(device const float4* small, int sw, int sh, int f, int x, int y, int channels) {
    const float ff = float(f);
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
    float3 v = float3(0.0f);
    for (int c = 0; c < channels; ++c) {
        const float top = p00[c] + (p10[c] - p00[c]) * tx;
        const float bot = p01[c] + (p11[c] - p01[c]) * tx;
        v[c] = top + (bot - top) * ty;
    }
    return v;
}

// One input of a blocked blur at offset o along it: DIR 0 along x, DIR 1 along y, DIR 2 along y of
// the horizontal blur (radius r, taps tp) of rows computed here and now (blurImpl's first half).
template <int DIR>
inline float blockAt(device const float* inF, int w, int h, int x, int y, int c, int o, constant float* tp, int r) {
    if (DIR == 0) return inF[(y * w + clampI(o, 0, w - 1)) * 4 + c];
    device const float* row = inF + clampI(o, 0, h - 1) * w * 4 + c;
    if (DIR == 1 || r == 0) return row[x * 4];
    float acc = 0.0f;
    for (int t = -r; t <= r; ++t) acc += tp[t] * row[clampI(x + t, 0, w - 1) * 4];
    return acc;
}

template <int DIR>
inline void blurBlock(device const float* inF, constant float* tp, int w, int h, int x, int y, int c, int r,
                      thread float* acc) {
    // Output k sits at (x + k, y) (DIR 0) or (x, y + k) (DIR 1, 2).
    const int p0 = DIR == 0 ? x : y;
    float win[OF_BLOCK];
    for (int k = 0; k < OF_BLOCK - 1; ++k) win[k] = blockAt<DIR>(inF, w, h, x, y, c, p0 - r + k, tp, r);
    for (int k = 0; k < OF_BLOCK; ++k) acc[k] = 0.0f;
    for (int t = -r; t <= r; ++t) {
        win[OF_BLOCK - 1] = blockAt<DIR>(inF, w, h, x, y, c, p0 + t + OF_BLOCK - 1, tp, r);
        const float wt = tp[t];
        for (int k = 0; k < OF_BLOCK; ++k) acc[k] += wt * win[k];
        for (int k = 0; k < OF_BLOCK - 1; ++k) win[k] = win[k + 1];
    }
}
}  // namespace

// exposeStage: DI -> linear exposure (10^lutE - eps) * gain over the work rect.
kernel void sp_expose(device const float* src [[buffer(0)]],
                      device float4* E [[buffer(1)]],
                      constant OFSpatialUniforms& u [[buffer(2)]],
                      device const half* lutE [[buffer(3)]],
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

// Two box-downs of one image in one read, by f (out) and 2f (out2): one thread per 2f block walks
// it row by row, so each f block and the 2f block are each summed in their own row-major order,
// exactly as sp_box_down sums them. p.factor = f; p.width/height = the 2f grid.
kernel void sp_box_down2(device const float4* in [[buffer(0)]],
                         device float4* out [[buffer(1)]],
                         constant OFPassUniforms& p [[buffer(2)]],
                         device float4* out2 [[buffer(3)]],
                         uint2 gid [[thread_position_in_grid]]) {
    if (outside(gid, p.width, p.height)) return;
    const int x = int(gid.x), y = int(gid.y), f = p.factor, f2 = 2 * f;
    const int xe = min((x + 1) * f2, p.inWidth), ye = min((y + 1) * f2, p.inHeight);
    const int w1 = (p.inWidth + f - 1) / f;
    float4 s[4] = {float4(0.0f), float4(0.0f), float4(0.0f), float4(0.0f)};
    float4 s2 = float4(0.0f);
    for (int yy = y * f2; yy < ye; ++yy) {
        const int sy = yy - y * f2 < f ? 0 : 2;
        for (int xx = x * f2; xx < xe; ++xx) {
            const float4 v = in[yy * p.inWidth + xx];
            s[sy + (xx - x * f2 < f ? 0 : 1)] += v;
            s2 += v;
        }
    }
    out2[y * p.width + x] = s2 * (1.0f / float((xe - x * f2) * (ye - y * f2)));
    for (int k = 0; k < 4; ++k) {
        const int bx = 2 * x + (k & 1), by = 2 * y + (k >> 1);
        const int xs = bx * f, ys = by * f;
        if (xs >= p.inWidth || ys >= p.inHeight) continue;
        const int bxe = min(xs + f, p.inWidth), bye = min(ys + f, p.inHeight);
        out[by * w1 + bx] = s[k] * (1.0f / float((bxe - xs) * (bye - ys)));
    }
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

// ---- Separable blur passes (blurImpl's two halves), same sums in the same order ----

// Horizontal, small radii: OF_HTILE threads per threadgroup, one row segment each. The pixels the
// taps reach are copied into threadgroup memory once (one plane per channel); radius 0 copies.
kernel void sp_blur_h_tiled(device const float4* in [[buffer(0)]],
                            device float4* out [[buffer(1)]],
                            constant OFPassUniforms& p [[buffer(2)]],
                            constant float* taps [[buffer(3)]],
                            threadgroup float* buf [[threadgroup(0)]],
                            uint2 tid [[thread_position_in_threadgroup]],
                            uint2 tgid [[threadgroup_position_in_grid]]) {
    const int w = p.width, y = int(tgid.y), x0 = int(tgid.x) * OF_HTILE, tx = int(tid.x);
    if (y >= p.height) return;  // whole threadgroup
    device const float* inF = (device const float*)in + y * w * 4;
    int base[3];
    int total = 0;
    for (int c = 0; c < 3; ++c) {
        base[c] = total;
        const int R = p.radius[c], n = OF_HTILE + 2 * R;
        for (int k = tx; k < n; k += OF_HTILE) buf[total + k] = inF[clampI(x0 - R + k, 0, w - 1) * 4 + c];
        total += n;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const int x = x0 + tx;
    if (x >= w) return;
    float4 o;
    for (int c = 0; c < 3; ++c) {
        const int r = p.radius[c];
        threadgroup const float* row = buf + base[c] + tx + r;
        if (r == 0) {
            o[c] = row[0];
            continue;
        }
        constant float* tp = taps + p.tapOffset[c] + r;
        float acc = 0.0f;
        _Pragma("unroll 8") for (int t = -r; t <= r; ++t) acc += tp[t] * row[t];
        o[c] = acc;
    }
    o[3] = in[y * w + x][3];
    out[y * w + x] = o;
}

// Horizontal, wide radii: lanes of a SIMD group take consecutive rows; each thread slides along
// OF_BLOCK pixels of its row (registers hold the window, each input is read once per tap step).
kernel void sp_blur_h_block(device const float4* in [[buffer(0)]],
                            device float4* out [[buffer(1)]],
                            constant OFPassUniforms& p [[buffer(2)]],
                            constant float* taps [[buffer(3)]],
                            uint2 gid [[thread_position_in_grid]]) {
    const int w = p.width, h = p.height;
    // Lanes of a SIMD group take consecutive rows; each thread walks OF_BLOCK pixels along its row.
    const int x0 = int(gid.y) * OF_BLOCK, y = int(gid.x);
    if (x0 >= w || y >= h) return;
    device const float* inF = (device const float*)in;
    float res[3][OF_BLOCK];
    for (int c = 0; c < 3; ++c) {
        const int r = p.radius[c];
        if (r == 0) {
            for (int k = 0; k < OF_BLOCK; ++k) res[c][k] = inF[(y * w + min(x0 + k, w - 1)) * 4 + c];
            continue;
        }
        blurBlock<0>(inF, taps + p.tapOffset[c] + r, w, h, x0, y, c, r, res[c]);
    }
    for (int k = 0; k < OF_BLOCK && x0 + k < w; ++k) {
        const int i = y * w + x0 + k;
        out[i] = float4(res[0][k], res[1][k], res[2][k], in[i][3]);
    }
}

// Vertical (any radius), stored / weighted / accumulated like the CPU's lobe sum: each thread
// slides down OF_BLOCK pixels of one column.
kernel void sp_blur_v_block(device const float4* in [[buffer(0)]],
                            device float4* out [[buffer(1)]],
                            constant OFPassUniforms& p [[buffer(2)]],
                            constant float* taps [[buffer(3)]],
                            uint2 gid [[thread_position_in_grid]]) {
    const int w = p.width, h = p.height;
    const int x = int(gid.x), y0 = int(gid.y) * OF_BLOCK;
    if (x >= w || y0 >= h) return;
    device const float* inF = (device const float*)in;
    float res[3][OF_BLOCK];
    for (int c = 0; c < 3; ++c) {
        const int r = p.radius[c];
        if (r == 0) {
            for (int k = 0; k < OF_BLOCK; ++k) res[c][k] = inF[(min(y0 + k, h - 1) * w + x) * 4 + c];
            continue;
        }
        blurBlock<1>(inF, taps + p.tapOffset[c] + r, w, h, x, y0, c, r, res[c]);
    }
    for (int k = 0; k < OF_BLOCK && y0 + k < h; ++k) {
        const int i = (y0 + k) * w + x;
        put(out, i, float3(res[0][k], res[1][k], res[2][k]), p, in[i][3]);
    }
}

// ---- Fused stage: spread + combine + the pointwise stages after it (OFFuseUniforms) ----
// The stage's shape (see OFFuseUniforms): each stage gets a pipeline holding only the code it runs.
constant int fcCombine [[function_constant(0)]];
constant int fcPost [[function_constant(1)]];
constant int fcChannels [[function_constant(2)]];
constant bool fcAccum [[function_constant(3)]];
constant int fcLobes [[function_constant(4)]];
constant int fcKinds [[function_constant(5)]];  // 2 bits per lobe, lobe 0 lowest

// The stage's combine of S with its input pixel X, then the pointwise stages after it; writes the
// work image at i, or with finish the display pixel (ix, iy) of dst.
inline void emitPixel(float4 X, float3 S, constant OFFuseUniforms& fu, constant OFSpatialUniforms& u,
                      device const float* curves, device const float* lutPE, device const float* lutD,
                      device const float* src, device float* dst, device float4* out, int i, int ix, int iy) {
    switch (fcCombine) {
        case OF_COMBINE_STORE:
            X.rgb = S;
            break;
        case OF_COMBINE_BLOOM:
            for (int c = 0; c < 3; ++c) X[c] = (1.0f - fu.cw[0]) * X[c] + fu.cw[0] * S[c];
            break;
        case OF_COMBINE_HALATION:
            for (int c = 0; c < 3; ++c) X[c] += fu.cw[c] * S[0];
            break;
        default:  // OF_COMBINE_EDGE
            for (int c = 0; c < 3; ++c) X[c] += fu.cw[c] * (X[c] - S[c]);
            break;
    }
    if (fcPost & OF_POST_DEVELOP_NEG)
        for (int c = 0; c < 3; ++c) X[c] = evalCurve(curves, u.neg[c], precise::log10(stdMax(X[c], 1e-30f)));
    if (fcPost & OF_POST_PRINT_EXPOSE) {
        const float3 pe = lookup(lutPE, u.lutPE, X.rgb);
        for (int c = 0; c < 3; ++c) X[c] = precise::pow(10.0f, pe[c]);
    }
    if (fcPost & OF_POST_DEVELOP_PRINT)
        for (int c = 0; c < 3; ++c)
            X[c] = evalCurve(curves, u.print[c], precise::log10(stdMax(X[c], 1e-30f)) + u.offsets[c]);
    if (fcPost & OF_POST_FINISH) {
        device float* o = dst + (iy - u.dst.y1) * u.dst.rowFloats + (ix - u.dst.x1) * 4;
        const float3 q = lookup(lutD, u.lutD, X.rgb);
        const float xyz[3] = {precise::pow(10.0f, q[0]), precise::pow(10.0f, q[1]), precise::pow(10.0f, q[2])};
        for (int k = 0; k < 3; ++k)
            o[k] = encodeDisplay(u.displayMatrix[k * 3] * xyz[0] + u.displayMatrix[k * 3 + 1] * xyz[1] +
                                 u.displayMatrix[k * 3 + 2] * xyz[2]);
        o[3] = src[(iy - u.src.y1) * u.src.rowFloats + (ix - u.src.x1) * 4 + 3];
        return;
    }
    out[i] = X;
}

// Each thread takes OF_BLOCK consecutive pixels of one column. DIRECT and PRE_H lobes slide a window
// of horizontally blurred values down the column (each read or computed once for all OF_BLOCK
// outputs): DIRECT computes them from the stage input row by row, PRE_H reads them from T. REDUCED
// lobes are sampled per pixel. B holds the sum of lobes summed by separate passes (fcAccum).
kernel void sp_fused(device const float4* img [[buffer(0)]],
                         device float4* out [[buffer(1)]],
                         constant OFFuseUniforms& fu [[buffer(2)]],
                         constant float* taps [[buffer(3)]],
                         device const float4* T [[buffer(4)]],
                         device const float4* small [[buffer(5)]],
                         device const float4* B [[buffer(6)]],
                         constant OFSpatialUniforms& u [[buffer(7)]],
                         device const float* curves [[buffer(8)]],
                         device const float* lutPE [[buffer(9)]],
                         device const float* lutD [[buffer(10)]],
                         device const float* src [[buffer(11)]],
                         device float* dst [[buffer(12)]],
                         uint2 gid [[thread_position_in_grid]]) {
    const int W = fu.width, H = fu.height, K = OF_BLOCK;
    const bool finish = (fcPost & OF_POST_FINISH) != 0;
    const int wx = finish ? u.winX1 + int(gid.x) - u.rectX1 : int(gid.x);
    const int wy0 = finish ? u.winY1 + int(gid.y) * K - u.rectY1 : int(gid.y) * K;
    const int ix = u.rectX1 + wx, iy0 = u.rectY1 + wy0;
    if (finish ? ix >= u.winX2 : wx >= W) return;
    const bool colIn = wx >= 0 && wx < W;
    const int ch = fcChannels;

    float S[3][OF_BLOCK];
    for (int c = 0; c < 3; ++c)
        for (int k = 0; k < K; ++k) S[c][k] = 0.0f;
    if (colIn) {
        if (fcAccum)
            for (int k = 0; k < K; ++k) {
                const int y = clampI(wy0 + k, 0, H - 1);
                const float4 b = B[y * W + wx];
                for (int c = 0; c < 3; ++c) S[c][k] = b[c];
            }
        for (int l = 0; l < fcLobes; ++l) {
            constant OFLobe& L = fu.lobes[l];
            const int kind = (fcKinds >> (2 * l)) & 3;
            const bool first = l == 0 && !fcAccum;
            if (kind != OF_LOBE_REDUCED) {
                device const float* Tf = (device const float*)(kind == OF_LOBE_PRE_H ? T : img);
                for (int c = 0; c < ch; ++c) {
                    const int r = L.radius[c];
                    float v[OF_BLOCK];
                    if (r == 0) {
                        for (int k = 0; k < K; ++k) v[k] = Tf[(clampI(wy0 + k, 0, H - 1) * W + wx) * 4 + c];
                    } else if (kind == OF_LOBE_PRE_H) {
                        blurBlock<1>(Tf, taps + L.tapOffset[c] + r, W, H, wx, wy0, c, r, v);
                    } else {
                        blurBlock<2>(Tf, taps + L.tapOffset[c] + r, W, H, wx, wy0, c, r, v);
                    }
                    for (int k = 0; k < K; ++k) S[c][k] = first ? L.weight[c] * v[k] : S[c][k] + L.weight[c] * v[k];
                }
            } else {
                for (int k = 0; k < K; ++k) {
                    const float3 v = upSample(small + L.smallOffset, L.sw, L.sh, L.factor, wx, wy0 + k, ch);
                    for (int c = 0; c < ch; ++c) S[c][k] = first ? L.weight[c] * v[c] : S[c][k] + L.weight[c] * v[c];
                }
            }
        }
    }
    for (int k = 0; k < K; ++k) {
        const int wy = wy0 + k, iy = iy0 + k;
        if (finish) {
            if (iy >= u.winY2 || !inside(u.dst, ix, iy)) continue;
            if (!colIn || wy < 0 || wy >= H || !inside(u.src, ix, iy)) {
                device float* o = dst + (iy - u.dst.y1) * u.dst.rowFloats + (ix - u.dst.x1) * 4;
                o[0] = 0.0f; o[1] = 0.0f; o[2] = 0.0f; o[3] = 0.0f;
                continue;
            }
        } else if (wy >= H) {
            break;
        }
        const int i = wy * W + wx;
        emitPixel(img[i], float3(S[0][k], S[1][k], S[2][k]), fu, u, curves, lutPE, lutD, src, dst, out, i, ix, iy);
    }
}
