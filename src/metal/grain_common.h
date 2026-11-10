// The grain sampler shared by the CPU (C++) and the GPU (Metal Shading Language): hashing, the
// Poisson cloud count per cell and the dye-cloud unit field u = N - mu of one pixel (ruling R3).
//
// Like shader_types.h this header is compiled twice, by clang++ (src/core/grain_sim.cpp) and by
// Apple's `metal`, so it keeps to their common subset: plain functions on int, float and OF_U32,
// no references, no templates, no std::. The CPU and the GPU then run the same operations on the
// same numbers.
//
// The field: clouds are discs of radius rPx (pixels) whose centres form a Poisson process of
// intensity lambdaPx per px^2. The plane is cut into square cells of side cellPx; a cell's cloud
// count and cloud positions come from hashes of (seed, cell), so any pixel can be evaluated alone
// and every window of a frame sees the same clouds.
#ifndef OPENFILM_GRAIN_COMMON_H
#define OPENFILM_GRAIN_COMMON_H

#ifdef __METAL_VERSION__
#include <metal_stdlib>
typedef uint OF_U32;
#define OF_INLINE inline
#define OF_FLOOR(x) metal::floor(x)
#define OF_EXP(x) metal::exp(x)
#define OF_LOG(x) metal::log(x)
#define OF_SQRT(x) metal::sqrt(x)
#define OF_COS(x) metal::cos(x)
#else
#include <math.h>
#include <stdint.h>
typedef uint32_t OF_U32;
#define OF_INLINE static inline
#define OF_FLOOR(x) floorf(x)
#define OF_EXP(x) expf(x)
#define OF_LOG(x) logf(x)
#define OF_SQRT(x) sqrtf(x)
#define OF_COS(x) cosf(x)
#endif

// Sub-samples per pixel side: a pixel's value is the mean over (i + 0.5) / 4, i = 0..3, in x and y.
#define OF_GRAIN_SUBSAMPLES 4
// Below this cloud radius (px) the pixel value is drawn as a Gaussian of the exact pixel rms.
#define OF_GRAIN_MIN_RADIUS_PX 0.25f
// Largest cloud count of one cell (its Poisson mean is mu / pi, about 1.3).
#define OF_GRAIN_MAX_CLOUDS 64

// lowbias32 (Chris Wellons): a full-avalanche 32-bit integer hash.
OF_INLINE OF_U32 of_hash(OF_U32 x) {
    x ^= x >> 16;
    x *= 0x7feb352dU;
    x ^= x >> 15;
    x *= 0x846ca68bU;
    x ^= x >> 16;
    return x;
}

OF_INLINE OF_U32 of_hash2(OF_U32 a, OF_U32 b) { return of_hash(a ^ of_hash(b)); }

// The hash's top 24 bits as a float in [0, 1).
OF_INLINE float of_unit(OF_U32 h) { return (float)(h >> 8) * (1.0f / 16777216.0f); }

// Poisson count of the given mean by inverse CDF at u in [0, 1): the smallest k with CDF(k) >= u.
// Stops at OF_GRAIN_MAX_CLOUDS, and where the float CDF stops growing (so u just below 1 cannot run
// to the cap on rounding). For a fixed u the count rises with the mean.
OF_INLINE int of_poisson(float mean, float u) {
    float p = OF_EXP(-mean);
    float cdf = p;
    int k = 0;
    while (u > cdf && k < OF_GRAIN_MAX_CLOUDS) {
        k += 1;
        p *= mean / (float)k;
        const float next = cdf + p;
        if (!(next > cdf)) break;
        cdf = next;
    }
    return k;
}

// Mean, over the pixel (px, py)'s 4 x 4 stratified sub-samples, of the number of clouds covering
// each sub-sample. Cells of side cellPx hold of_poisson(lambdaPx cellPx^2, ...) clouds; cell (cx, cy)
// covers [cx cellPx, (cx + 1) cellPx) x [cy cellPx, (cy + 1) cellPx) and draws from the hash of
// (seed, its coordinates wrapped modulo periodCells when periodCells > 0). Every cell that a cloud
// reaching the pixel can sit in is visited. Positions are taken relative to the pixel's corner so
// float keeps its precision far from the origin.
OF_INLINE float of_overlap_mean(int px, int py, float rPx, float lambdaPx, float cellPx, OF_U32 seed, int periodCells) {
    const float meanCount = lambdaPx * cellPx * cellPx;
    const float r2 = rPx * rPx;
    const float fx = (float)px, fy = (float)py;
    const int cx0 = (int)OF_FLOOR((fx - rPx) / cellPx), cx1 = (int)OF_FLOOR((fx + 1.0f + rPx) / cellPx);
    const int cy0 = (int)OF_FLOOR((fy - rPx) / cellPx), cy1 = (int)OF_FLOOR((fy + 1.0f + rPx) / cellPx);
    const float step = 1.0f / (float)OF_GRAIN_SUBSAMPLES;
    int hits = 0;
    for (int cy = cy0; cy <= cy1; ++cy) {
        int wy = cy;
        if (periodCells > 0) {
            wy %= periodCells;
            if (wy < 0) wy += periodCells;
        }
        const float oy = (float)cy * cellPx - fy;
        for (int cx = cx0; cx <= cx1; ++cx) {
            int wx = cx;
            if (periodCells > 0) {
                wx %= periodCells;
                if (wx < 0) wx += periodCells;
            }
            const float ox = (float)cx * cellPx - fx;
            const OF_U32 cellHash = of_hash2(seed, of_hash2((OF_U32)wx, (OF_U32)wy));
            const int n = of_poisson(meanCount, of_unit(cellHash));
            for (int i = 0; i < n; ++i) {
                const float x = ox + cellPx * of_unit(of_hash2(cellHash, (OF_U32)(2 * i)));
                const float y = oy + cellPx * of_unit(of_hash2(cellHash, (OF_U32)(2 * i + 1)));
                if (x <= -rPx || x >= 1.0f + rPx || y <= -rPx || y >= 1.0f + rPx) continue;
                for (int sy = 0; sy < OF_GRAIN_SUBSAMPLES; ++sy) {
                    const float dy = ((float)sy + 0.5f) * step - y;
                    const float rest = r2 - dy * dy;
                    if (rest <= 0.0f) continue;
                    for (int sx = 0; sx < OF_GRAIN_SUBSAMPLES; ++sx) {
                        const float dx = ((float)sx + 0.5f) * step - x;
                        if (dx * dx < rest) hits += 1;
                    }
                }
            }
        }
    }
    return (float)hits * (1.0f / (float)(OF_GRAIN_SUBSAMPLES * OF_GRAIN_SUBSAMPLES));
}

// A standard normal for the pixel (px, py): Box-Muller from two hashes of (seed, px, py).
OF_INLINE float of_gauss(int px, int py, OF_U32 seed) {
    const OF_U32 h = of_hash2(seed, of_hash2((OF_U32)px, (OF_U32)py));
    const float u1 = ((float)(of_hash2(h, 0U) >> 8) + 0.5f) * (1.0f / 16777216.0f);  // (0, 1): log is finite
    const float u2 = of_unit(of_hash2(h, 1U));
    return OF_SQRT(-2.0f * OF_LOG(u1)) * OF_COS(6.28318530718f * u2);
}

// The pixel's unit grain u_pixel (mean overlap count minus mu): the cloud simulation for clouds of
// at least OF_GRAIN_MIN_RADIUS_PX, below it gaussRms (the exact pixel-area rms of u) times a normal.
// Cells are one cloud radius wide.
OF_INLINE float of_grain_unit(int px, int py, float rPx, float lambdaPx, float mu, float gaussRms, OF_U32 seed,
                              int periodCells) {
    if (rPx < OF_GRAIN_MIN_RADIUS_PX) return gaussRms * of_gauss(px, py, seed);
    return of_overlap_mean(px, py, rPx, lambdaPx, rPx, seed, periodCells) - mu;
}

// Preview plates: where frame pixel (px, py) reads a size x size tiling plate for the frame whose
// layer seed is `seed`. A per-frame offset in [0, size)^2 and one of the 8 symmetries of the square
// (swap x/y, flip x, flip y) come from hashes of the seed, so every frame, stage and layer reads a
// different view of the same plate. Returns x | (y << 16); size must be at most 65535.
OF_INLINE OF_U32 of_plate_coord(int px, int py, OF_U32 seed, int size) {
    const OF_U32 h = of_hash2(seed, 0x706c6174U);
    const int ox = (int)(of_hash2(h, 1U) % (OF_U32)size), oy = (int)(of_hash2(h, 2U) % (OF_U32)size);
    const OF_U32 o = of_hash2(h, 3U);
    int x = (px + ox) % size, y = (py + oy) % size;
    if (x < 0) x += size;
    if (y < 0) y += size;
    if (o & 1U) {
        const int t = x;
        x = y;
        y = t;
    }
    if (o & 2U) x = size - 1 - x;
    if (o & 4U) y = size - 1 - y;
    return (OF_U32)x | ((OF_U32)y << 16);
}

#endif
