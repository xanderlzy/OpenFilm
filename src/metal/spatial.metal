// GPU twin of renderFilm() with spatial effects on (src/core/film_spatial.cpp and the blur engine
// in src/core/blur.cpp); any change there must be made here too. metal_renderer.mm plans the passes
// from the CPU's own lobes and taps and encodes them in the CPU's order; tests/gpu/test_gpu_vs_cpu.mm
// renders both and checks they agree. Grain (M5, grainStage) runs in the fused pass as two pointwise
// stages on density, with the CPU's sampler itself (grain_common.h): per pixel for Quality Full, from
// plates that sp_make_plate generates with that sampler for Preview.
//
// Every kernel does the CPU's float operations in the CPU's order: blur taps are summed from -r to
// r (one channel at a time), box averages row by row, lobes accumulate in lobe order. The taps are
// computed on the host by the CPU's own blurTaps() and uploaded. What the GPU changes is only where
// the work happens: each spatial stage's spread, its combine and the per-pixel stages after it run
// in one pass (sp_fused), and blurs keep their inputs in registers or threadgroup memory instead of
// re-reading them.
//
// Work images are planar: three float planes (R, G, B) of w x h, pixel (x, y) of channel c at
// c * w * h + y * w + x. A blur reads one channel at a time, so neighbouring threads read
// neighbouring floats. Alpha is not carried: the last pass takes it from the source.
#include <metal_stdlib>
#include "metal/film_common.h"
#include "metal/grain_common.h"

using namespace metal;

namespace {

inline int clampI(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }

inline bool outside(uint2 gid, int w, int h) { return int(gid.x) >= w || int(gid.y) >= h; }

// Writes v at pixel i of the planar image `out` (plane size n) by the pass mode (OF_STORE /
// OF_STORE_WEIGHTED / OF_ACCUMULATE). A channel the pass leaves out (radius -1: another of the lobe's
// downsample groups computes it) is left alone, except that the first lobe of a sum zeroes it, as the
// CPU's sum starts at 0.
inline void put(device float* out, int n, int i, float3 v, constant OFPassUniforms& p) {
    for (int c = 0; c < 3; ++c) {
        device float* o = out + c * n + i;
        if (p.radius[c] < 0) {
            if (p.mode == OF_STORE_WEIGHTED) *o = 0.0f;
        } else if (p.mode == OF_STORE) *o = v[c];
        else if (p.mode == OF_STORE_WEIGHTED) *o = p.weight[c] * v[c];
        else *o = *o + p.weight[c] * v[c];
    }
}

// upAccumulate's bilinear sample of a planar coarse image (sw x sh) at fine pixel (x, y), for the
// channels in `mask`.
inline float3 upSample(device const float* small, int sw, int sh, int f, int x, int y, int mask) {
    const float ff = float(f);
    const float fy = (float(y) + 0.5f) / ff - 0.5f;
    const float flY = floor(fy);
    const float ty = fy - flY;
    const int ya = clampI(int(flY), 0, sh - 1), yb = clampI(int(flY) + 1, 0, sh - 1);
    const float fx = (float(x) + 0.5f) / ff - 0.5f;
    const float flX = floor(fx);
    const float tx = fx - flX;
    const int xa = clampI(int(flX), 0, sw - 1), xb = clampI(int(flX) + 1, 0, sw - 1);
    const int n = sw * sh;
    float3 v = float3(0.0f);
    for (int c = 0; c < 3; ++c) {
        if (!(mask >> c & 1)) continue;
        device const float* s = small + c * n;
        const float p00 = s[ya * sw + xa], p10 = s[ya * sw + xb], p01 = s[yb * sw + xa], p11 = s[yb * sw + xb];
        const float top = p00 + (p10 - p00) * tx;
        const float bot = p01 + (p11 - p01) * tx;
        v[c] = top + (bot - top) * ty;
    }
    return v;
}

// One input of a blocked blur of the plane `pl` (w x h) at offset o along the blur: DIR 0 along x
// (row y), DIR 1 along y (column x), DIR 2 along y of the horizontal blur (radius r, taps tp) of the
// rows, computed here and now (blurImpl's first half).
template <int DIR>
inline float blockAt(device const float* pl, int w, int h, int x, int y, int o, constant float* tp, int r) {
    if (DIR == 0) return pl[y * w + clampI(o, 0, w - 1)];
    device const float* row = pl + clampI(o, 0, h - 1) * w;
    if (DIR == 1 || r == 0) return row[x];
    float acc = 0.0f;
    for (int t = -r; t <= r; ++t) acc += tp[t] * row[clampI(x + t, 0, w - 1)];
    return acc;
}

// OF_BLOCK outputs of a 1D blur (radius r > 0, taps tp centred) at (x + k, y) (DIR 0) or (x, y + k)
// (DIR 1, 2). A window of OF_BLOCK inputs slides along in registers, so each input is read (or, for
// DIR 2, computed) once for all OF_BLOCK outputs; each output still adds its taps from -r to r.
template <int DIR, int B = OF_BLOCK>
inline void blurBlock(device const float* pl, constant float* tp, int w, int h, int x, int y, int r,
                      thread float* acc) {
    const int p0 = DIR == 0 ? x : y;
    float win[B];
    for (int k = 0; k < B - 1; ++k) win[k] = blockAt<DIR>(pl, w, h, x, y, p0 - r + k, tp, r);
    for (int k = 0; k < B; ++k) acc[k] = 0.0f;
    for (int t = -r; t <= r; ++t) {
        win[B - 1] = blockAt<DIR>(pl, w, h, x, y, p0 + t + B - 1, tp, r);
        const float wt = tp[t];
        for (int k = 0; k < B; ++k) acc[k] += wt * win[k];
        for (int k = 0; k < B - 1; ++k) win[k] = win[k + 1];
    }
}

// GrainAmplitude::at (ruling R4): ln sigma linear in density between the layer's datasheet nodes
// (density ascending), held at the end nodes outside; exp of it times k. 0 for a non-finite D or no
// nodes. The pair of nodes is std::upper_bound's (the first node denser than D, and the one before),
// found by walking from the node the layer's bin index gives for D: the same pair, in a step or two
// instead of a binary search's seven dependent loads.
inline float grainAmplitude(device const float2* nodes, device const uchar* bins, int layer, constant OFGrainLayer& L,
                            float D) {
    if (!isfinite(D) || L.nodeCount <= 0) return 0.0f;
    device const float2* nd = nodes + L.nodeOffset;
    const int n = L.nodeCount;
    float lnS;
    if (D <= nd[0].x) {
        lnS = nd[0].y;
    } else if (D >= nd[n - 1].x) {
        lnS = nd[n - 1].y;
    } else {
        // nd[0].x < D < nd[n - 1].x here, so both walks stop inside [1, n - 1].
        int lo = int(bins[layer * OF_GRAIN_BINS + clamp(int((D - nd[0].x) * L.binScale), 0, OF_GRAIN_BINS - 1)]);
        while (D < nd[lo - 1].x) --lo;
        while (!(D < nd[lo].x)) ++lo;
        const float span = nd[lo].x - nd[lo - 1].x;
        const float t = span > 0.0f ? (D - nd[lo - 1].x) / span : 0.0f;
        lnS = nd[lo - 1].y + t * (nd[lo].y - nd[lo - 1].y);
    }
    return precise::exp(lnS) * L.k;
}

// grainStage on one pixel's three densities X at stage st (0 negative, 1 print), frame pixel (ix, iy):
// D_c += amount * a_c(D_c) * u_c; non-finite D_c unchanged, as are layers with no clouds. u_c is the
// simulation (Full) or, with fromPlates (a pipeline constant), the layer's plate read through this
// frame's view of it (Preview, GrainPlates::sample); `plates` holds the six one after another,
// negative R, G, B, then print R, G, B.
inline void addGrain(thread float3& X, int st, bool fromPlates, constant OFGrainUniforms& gu,
                     device const float2* nodes, device const uchar* bins, device const half* plates, int ix, int iy) {
    for (int c = 0; c < 3; ++c) {
        constant OFGrainLayer& L = gu.layer[st * 3 + c];
        if (!isfinite(X[c]) || !(L.rPx > 0.0f)) continue;
        float u;
        if (fromPlates) {
            const int n = gu.plateSize;
            const uint pc = of_plate_map(ix, iy, L.plateOx, L.plateOy, L.plateOrient, n);  // of_plate_coord
            u = float(plates[(st * 3 + c) * n * n + int(pc >> 16) * n + int(pc & 0xffffu)]);
        } else {
            u = of_grain_unit(ix, iy, L.rPx, L.lambdaPx, L.mu, L.gaussRms, L.seed, 0);
        }
        X[c] += gu.amount * grainAmplitude(nodes, bins, st * 3 + c, L, X[c]) * u;
    }
}

}  // namespace

static_assert(sizeof(OFGrainLayer) == 48 && sizeof(OFGrainUniforms) == 296 &&
                  sizeof(OFPlateUniforms) == 32,
              "grain uniforms layout (see shader_types.h)");

// exposeStage: DI -> linear exposure (10^lutE - eps) * gain over the work rect.
kernel void sp_expose(device const float* src [[buffer(0)]],
                      device float* E [[buffer(1)]],
                      constant OFSpatialUniforms& u [[buffer(2)]],
                      device const half4* lutE [[buffer(3)]],
                      uint2 gid [[thread_position_in_grid]]) {
    if (outside(gid, u.width, u.height)) return;
    const int x = u.rectX1 + int(gid.x), y = u.rectY1 + int(gid.y);
    device const float* in = src + (y - u.src.y1) * u.src.rowFloats + (x - u.src.x1) * 4;
    const float3 e = lookup(lutE, u.lutE, float3(sane(in[0]), sane(in[1]), sane(in[2])));
    const int n = u.width * u.height, i = int(gid.y) * u.width + int(gid.x);
    for (int c = 0; c < 3; ++c) E[c * n + i] = (precise::pow(10.0f, e[c]) - OF_EXPOSURE_LUT_EPS) * u.gain;
}

// boxDown: f x f average; partial edge blocks average only their real pixels.
kernel void sp_box_down(device const float* in [[buffer(0)]],
                        device float* out [[buffer(1)]],
                        constant OFPassUniforms& p [[buffer(2)]],
                        uint2 gid [[thread_position_in_grid]]) {
    const int c = int(gid.y) / p.height;  // one thread per channel: the grid is 3 planes tall
    const int x = int(gid.x), y = int(gid.y) - c * p.height, f = p.factor, iw = p.inWidth;
    if (x >= p.width || c >= 3 || p.radius[c] < 0) return;  // radius -1: channel not needed
    const int xe = min((x + 1) * f, iw), ye = min((y + 1) * f, p.inHeight);
    device const float* pl = in + c * iw * p.inHeight;
    float s = 0.0f;
    for (int yy = y * f; yy < ye; ++yy)
        for (int xx = x * f; xx < xe; ++xx) s += pl[yy * iw + xx];
    const float inv = 1.0f / float((xe - x * f) * (ye - y * f));
    out[c * p.width * p.height + y * p.width + x] = s * inv;
}

// Two box downs of one image in one read, by f (out) and 2f (out2): one thread per 2f block walks
// it row by row, so each f block and the 2f block are each summed in their own row-major order,
// exactly as sp_box_down sums them. p.factor = f; p.width/height = the 2f grid.
kernel void sp_box_down2(device const float* in [[buffer(0)]],
                         device float* out [[buffer(1)]],
                         constant OFPassUniforms& p [[buffer(2)]],
                         device float* out2 [[buffer(3)]],
                         uint2 gid [[thread_position_in_grid]]) {
    const int c = int(gid.y) / p.height;  // one thread per channel: the grid is 3 planes tall
    const int x = int(gid.x), y = int(gid.y) - c * p.height, f = p.factor, f2 = 2 * f, iw = p.inWidth,
              ih = p.inHeight;
    if (x >= p.width || c >= 3 || p.radius[c] < 0) return;  // radius -1: channel not needed
    const int xe = min((x + 1) * f2, iw), ye = min((y + 1) * f2, ih);
    const int w1 = (iw + f - 1) / f, h1 = (ih + f - 1) / f;
    device const float* pl = in + c * iw * ih;
    float s[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float s2 = 0.0f;
    for (int yy = y * f2; yy < ye; ++yy) {
        const int sy = yy - y * f2 < f ? 0 : 2;
        for (int xx = x * f2; xx < xe; ++xx) {
            const float v = pl[yy * iw + xx];
            s[sy + (xx - x * f2 < f ? 0 : 1)] += v;
            s2 += v;
        }
    }
    const float inv2 = 1.0f / float((xe - x * f2) * (ye - y * f2));
    out2[c * p.width * p.height + y * p.width + x] = s2 * inv2;
    for (int k = 0; k < 4; ++k) {
        const int bx = 2 * x + (k & 1), by = 2 * y + (k >> 1);
        const int xs = bx * f, ys = by * f;
        if (xs >= iw || ys >= ih) continue;
        const float inv = 1.0f / float((min(xs + f, iw) - xs) * (min(ys + f, ih) - ys));
        out[c * w1 * h1 + by * w1 + bx] = s[k] * inv;
    }
}

// upAccumulate: bilinear at ((x + 0.5) / f - 0.5), clamp-to-edge, weighted into out (summing the
// lobes before a fused pass's own).
kernel void sp_up(device const float* small [[buffer(0)]],
                  device float* out [[buffer(1)]],
                  constant OFPassUniforms& p [[buffer(2)]],
                  uint2 gid [[thread_position_in_grid]]) {
    if (outside(gid, p.width, p.height)) return;
    const int x = int(gid.x), y = int(gid.y);
    const int mask = (p.radius[0] >= 0 ? 1 : 0) | (p.radius[1] >= 0 ? 2 : 0) | (p.radius[2] >= 0 ? 4 : 0);
    const float3 v = upSample(small, p.inWidth, p.inHeight, p.factor, x, y, mask);
    put(out, p.width * p.height, y * p.width + x, v, p);
}

// finishStage when the work rect is empty: every window pixel lies outside the source -> (0,0,0,0).
kernel void sp_finish_empty(device float* dst [[buffer(1)]],
                            constant OFSpatialUniforms& u [[buffer(2)]],
                            uint2 gid [[thread_position_in_grid]]) {
    const int x = u.winX1 + int(gid.x);
    const int y = u.winY1 + int(gid.y);
    if (x >= u.winX2 || y >= u.winY2) return;
    if (!inside(u.dst, x, y)) return;  // window pixel the host gave us no room for
    device float* out = dst + (y - u.dst.y1) * u.dst.rowFloats + (x - u.dst.x1) * 4;
    out[0] = 0.0f; out[1] = 0.0f; out[2] = 0.0f; out[3] = 0.0f;
}

// ---- Separable blur passes (blurImpl's two halves), same sums in the same order ----

// Horizontal, small radii: OF_HTILE threads per threadgroup, one row segment each. The pixels the
// taps reach are copied into threadgroup memory once (one plane per channel); radius 0 copies. A
// second blur of the same image (p2, out2: another lobe's) can run from the same tile; a pass
// without one gives p2 radius -1 everywhere.
inline void tiledRow(threadgroup const float* row, constant float* taps, constant OFPassUniforms& p, int c,
                     device float* out, int i) {
    const int r = p.radius[c];
    if (r < 0) return;
    float acc;
    if (r == 0) {
        acc = row[0];
    } else {
        constant float* tp = taps + p.tapOffset[c] + r;
        acc = 0.0f;
        _Pragma("unroll 8") for (int t = -r; t <= r; ++t) acc += tp[t] * row[t];
    }
    out[i] = acc;
}

kernel void sp_blur_h_tiled(device const float* in [[buffer(0)]],
                            device float* out [[buffer(1)]],
                            constant OFPassUniforms& p [[buffer(2)]],
                            constant float* taps [[buffer(3)]],
                            device float* out2 [[buffer(4)]],
                            constant OFPassUniforms& p2 [[buffer(5)]],
                            threadgroup float* buf [[threadgroup(0)]],
                            uint2 tid [[thread_position_in_threadgroup]],
                            uint2 tgid [[threadgroup_position_in_grid]]) {
    const int w = p.width, n = w * p.height, y = int(tgid.y), x0 = int(tgid.x) * OF_HTILE, tx = int(tid.x);
    if (y >= p.height) return;  // whole threadgroup
    int base[3], reach[3];
    int total = 0;
    for (int c = 0; c < 3; ++c) {
        const int R = max(p.radius[c], p2.radius[c]);
        base[c] = total;
        reach[c] = R;
        if (R < 0) continue;  // channel not needed
        const int m = OF_HTILE + 2 * R;
        device const float* row = in + c * n + y * w;
        for (int k = tx; k < m; k += OF_HTILE) buf[total + k] = row[clampI(x0 - R + k, 0, w - 1)];
        total += m;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const int x = x0 + tx;
    if (x >= w) return;
    for (int c = 0; c < 3; ++c) {
        if (reach[c] < 0) continue;
        threadgroup const float* row = buf + base[c] + tx + reach[c];  // this pixel
        tiledRow(row, taps, p, c, out, c * n + y * w + x);
        tiledRow(row, taps, p2, c, out2, c * n + y * w + x);
    }
}

// Horizontal, wide radii: lanes of a SIMD group take consecutive rows; each thread slides along
// OF_BLOCK pixels of its row (blurBlock).
kernel void sp_blur_h_block(device const float* in [[buffer(0)]],
                            device float* out [[buffer(1)]],
                            constant OFPassUniforms& p [[buffer(2)]],
                            constant float* taps [[buffer(3)]],
                            uint2 gid [[thread_position_in_grid]]) {
    const int w = p.width, h = p.height, n = w * h;
    const int x0 = int(gid.y) * OF_HBLOCK, y = int(gid.x);
    if (x0 >= w || y >= h) return;
    for (int c = 0; c < 3; ++c) {
        const int r = p.radius[c];
        device const float* pl = in + c * n;
        device float* o = out + c * n + y * w;
        if (r < 0) continue;  // channel not needed
        if (r == 0) {
            for (int k = 0; k < OF_HBLOCK && x0 + k < w; ++k) o[x0 + k] = pl[y * w + x0 + k];
            continue;
        }
        float res[OF_HBLOCK];
        blurBlock<0, OF_HBLOCK>(pl, taps + p.tapOffset[c] + r, w, h, x0, y, r, res);
        for (int k = 0; k < OF_HBLOCK && x0 + k < w; ++k) o[x0 + k] = res[k];
    }
}

// Vertical (any radius), stored / weighted / accumulated like the CPU's lobe sum: each thread
// slides down OF_BLOCK pixels of one column.
kernel void sp_blur_v_block(device const float* in [[buffer(0)]],
                            device float* out [[buffer(1)]],
                            constant OFPassUniforms& p [[buffer(2)]],
                            constant float* taps [[buffer(3)]],
                            uint2 gid [[thread_position_in_grid]]) {
    const int w = p.width, h = p.height, n = w * h;
    const int x = int(gid.x), y0 = int(gid.y) * OF_BLOCK;
    if (x >= w || y0 >= h) return;
    float res[3][OF_BLOCK];
    for (int c = 0; c < 3; ++c) {
        const int r = p.radius[c];
        device const float* pl = in + c * n;
        if (r < 0) continue;  // channel not needed (put leaves it)
        if (r == 0) {
            for (int k = 0; k < OF_BLOCK; ++k) res[c][k] = pl[min(y0 + k, h - 1) * w + x];
            continue;
        }
        blurBlock<1>(pl, taps + p.tapOffset[c] + r, w, h, x, y0, r, res[c]);
    }
    for (int k = 0; k < OF_BLOCK && y0 + k < h; ++k)
        put(out, n, (y0 + k) * w + x, float3(res[0][k], res[1][k], res[2][k]), p);
}

// ---- Fused stage: spread + combine + the pointwise stages after it (OFFuseUniforms) ----
// The stage's shape (see OFFuseUniforms): each stage gets a pipeline holding only the code it runs.
constant int fcCombine [[function_constant(0)]];
constant int fcPost [[function_constant(1)]];
constant int fcChannels [[function_constant(2)]];
constant bool fcAccum [[function_constant(3)]];
constant int fcLobes [[function_constant(4)]];
constant int fcKinds [[function_constant(5)]];  // 2 bits per lobe, lobe 0 lowest

// The stage's combine of the lobe sum S with its input pixel X, then the pointwise stages after it;
// writes pixel i of the planar work image `out` (plane size n), or with finish the display pixel
// (ix, iy) of dst. (ix, iy) is the pixel's frame position either way (grain is seeded by it).
inline void emitPixel(float3 X, float3 S, constant OFFuseUniforms& fu, constant OFSpatialUniforms& u,
                      device const float* curves, device const float4* lutPE, device const float4* lutD,
                      device const float* src, device float* dst, device float* out, int n, int i, int ix,
                      int iy, constant OFGrainUniforms& gu, device const float2* grainNodes,
                      device const uchar* grainBins, device const half* plates) {
    switch (fcCombine) {
        case OF_COMBINE_STORE:
            X = S;
            break;
        case OF_COMBINE_BLOOM:
            for (int c = 0; c < 3; ++c) X[c] = (1.0f - fu.cw[0]) * X[c] + fu.cw[0] * S[c];
            break;
        case OF_COMBINE_HALATION:
            for (int c = 0; c < 3; ++c) X[c] += fu.cw[c] * S[0];
            break;
        case OF_COMBINE_EDGE:
            for (int c = 0; c < 3; ++c) X[c] += fu.cw[c] * (X[c] - S[c]);
            break;
        default:  // OF_COMBINE_NONE
            break;
    }
    if (fcPost & OF_POST_DEVELOP_NEG)
        for (int c = 0; c < 3; ++c) X[c] = evalCurve(curves, u.neg[c], precise::log10(stdMax(X[c], 1e-30f)));
    const bool fromPlates = (fcPost & OF_POST_GRAIN_PLATES) != 0;
    if (fcPost & OF_POST_NEG_GRAIN) addGrain(X, 0, fromPlates, gu, grainNodes, grainBins, plates, ix, iy);
    if (fcPost & OF_POST_PRINT_EXPOSE) {
        const float3 pe = lookup(lutPE, u.lutPE, X);
        for (int c = 0; c < 3; ++c) X[c] = precise::pow(10.0f, pe[c]);
    }
    if (fcPost & OF_POST_DEVELOP_PRINT)
        for (int c = 0; c < 3; ++c)
            X[c] = evalCurve(curves, u.print[c], precise::log10(stdMax(X[c], 1e-30f)) + u.offsets[c]);
    if (fcPost & OF_POST_PRINT_GRAIN) addGrain(X, 1, fromPlates, gu, grainNodes, grainBins, plates, ix, iy);
    if (fcPost & OF_POST_FINISH) {
        device float* o = dst + (iy - u.dst.y1) * u.dst.rowFloats + (ix - u.dst.x1) * 4;
        const float3 q = lookup(lutD, u.lutD, X);
        const float xyz[3] = {precise::pow(10.0f, q[0]), precise::pow(10.0f, q[1]), precise::pow(10.0f, q[2])};
        for (int k = 0; k < 3; ++k)
            o[k] = encodeDisplay(u.displayMatrix[k * 3] * xyz[0] + u.displayMatrix[k * 3 + 1] * xyz[1] +
                                 u.displayMatrix[k * 3 + 2] * xyz[2]);
        o[3] = src[(iy - u.src.y1) * u.src.rowFloats + (ix - u.src.x1) * 4 + 3];
        return;
    }
    for (int c = 0; c < 3; ++c) out[c * n + i] = X[c];
}

// Each thread takes OF_BLOCK consecutive pixels of one column. DIRECT and PRE_H lobes slide a window
// of horizontally blurred values down the column (each read or computed once for all OF_BLOCK
// outputs): DIRECT computes them from the stage input row by row, PRE_H reads them from T. REDUCED
// lobes are sampled per pixel from the small-results buffer. B holds the sum of the lobes summed by
// separate passes before this one (fcAccum).
kernel void sp_fused(device const float* img [[buffer(0)]],
                     device float* out [[buffer(1)]],
                     constant OFFuseUniforms& fu [[buffer(2)]],
                     constant float* taps [[buffer(3)]],
                     device const float* T [[buffer(4)]],
                     device const float* small [[buffer(5)]],
                     device const float* B [[buffer(6)]],
                     constant OFSpatialUniforms& u [[buffer(7)]],
                     device const float* curves [[buffer(8)]],
                     device const float4* lutPE [[buffer(9)]],
                     device const float4* lutD [[buffer(10)]],
                     device const float* src [[buffer(11)]],
                     device float* dst [[buffer(12)]],
                     device const float* T2 [[buffer(13)]],
                     constant OFGrainUniforms& gu [[buffer(14)]],
                     device const float2* grainNodes [[buffer(15)]],
                     device const half* plates [[buffer(16)]],
                     device const uchar* grainBins [[buffer(17)]],
                     uint2 gid [[thread_position_in_grid]]) {
    const int W = fu.width, H = fu.height, n = W * H, K = OF_BLOCK;
    const bool finish = (fcPost & OF_POST_FINISH) != 0;
    // This thread's column and first row, in work coordinates; with finish the grid is the window.
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
            for (int c = 0; c < 3; ++c)
                for (int k = 0; k < K; ++k) S[c][k] = B[c * n + clampI(wy0 + k, 0, H - 1) * W + wx];
        for (int l = 0; l < fcLobes; ++l) {
            constant OFLobe& L = fu.lobes[l];
            const int kind = (fcKinds >> (2 * l)) & 3;
            const bool first = l == 0 && !fcAccum;
            if (kind != OF_LOBE_REDUCED) {
                for (int c = 0; c < ch; ++c) {
                    const int r = L.radius[c];
                    if (r < 0) continue;  // another downsample group of the lobe computes this channel
                    device const float* pl =
                        (kind == OF_LOBE_PRE_H ? T : kind == OF_LOBE_PRE_H2 ? T2 : img) + c * n;
                    float v[OF_BLOCK];
                    if (r == 0) {
                        for (int k = 0; k < K; ++k) v[k] = pl[clampI(wy0 + k, 0, H - 1) * W + wx];
                    } else if (kind != OF_LOBE_DIRECT) {
                        blurBlock<1>(pl, taps + L.tapOffset[c] + r, W, H, wx, wy0, r, v);
                    } else {
                        blurBlock<2>(pl, taps + L.tapOffset[c] + r, W, H, wx, wy0, r, v);
                    }
                    for (int k = 0; k < K; ++k) S[c][k] = first ? L.weight[c] * v[k] : S[c][k] + L.weight[c] * v[k];
                }
            } else {
                int mask = 0;
                for (int c = 0; c < ch; ++c) mask |= L.radius[c] >= 0 ? 1 << c : 0;
                for (int k = 0; k < K; ++k) {
                    const float3 v = upSample(small + L.smallOffset, L.sw, L.sh, L.factor, wx, wy0 + k, mask);
                    for (int c = 0; c < ch; ++c)
                        if (mask >> c & 1) S[c][k] = first ? L.weight[c] * v[c] : S[c][k] + L.weight[c] * v[c];
                }
            }
        }
    }
    for (int k = 0; k < K; ++k) {
        const int wy = wy0 + k, iy = iy0 + k;
        if (finish) {
            if (iy >= u.winY2 || !inside(u.dst, ix, iy)) continue;
            if (!colIn || wy < 0 || wy >= H || !inside(u.src, ix, iy)) {  // outside the source
                device float* o = dst + (iy - u.dst.y1) * u.dst.rowFloats + (ix - u.dst.x1) * 4;
                o[0] = 0.0f; o[1] = 0.0f; o[2] = 0.0f; o[3] = 0.0f;
                continue;
            }
        } else if (wy >= H) {
            break;
        }
        const int i = wy * W + wx;
        emitPixel(float3(img[i], img[n + i], img[2 * n + i]), float3(S[0][k], S[1][k], S[2][k]), fu, u, curves,
                  lutPE, lutD, src, dst, out, n, i, ix, iy, gu, grainNodes, grainBins, plates);
    }
}

// One Preview plate (makePlates' makePlate): u at plate pixel gid, rounded to half (round to nearest
// even, as floatToHalf).
kernel void sp_make_plate(device half* plate [[buffer(0)]],
                          constant OFPlateUniforms& p [[buffer(1)]],
                          uint2 gid [[thread_position_in_grid]]) {
    const int x = int(gid.x), y = int(gid.y);
    if (x >= p.size || y >= p.size) return;
    float u = 0.0f;
    if (p.rPx > 0.0f)
        u = p.rPx >= OF_GRAIN_MIN_RADIUS_PX
                ? of_overlap_mean(x, y, p.rPx, p.lambdaPx, p.cellPx, p.seed, p.periodCells) - p.mu
                : p.gaussRms * of_gauss(x, y, p.seed);
    plate[y * p.size + x] = half(u);
}
