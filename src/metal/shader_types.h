// Data shared between the CPU (C++) and the GPU (Metal Shading Language).
//
// This header is compiled twice: once by clang++ for metal_renderer.mm and once by Apple's
// `metal` compiler for the .metal kernels. Both must see byte-identical struct layouts, so it
// uses only plain C types (int, float) and no C++ features.
#ifndef OPENFILM_SHADER_TYPES_H
#define OPENFILM_SHADER_TYPES_H

// One image's layout, mirroring openfilm::ImageGeometry. rowFloats = rowBytes / 4.
typedef struct {
    int x1, y1, x2, y2;
    int rowFloats;
} OFImageGeom;

// openfilm::kExposureLutEps as a float: the exposure table stores log10(E + eps). metal_renderer.mm
// static_asserts that the two agree.
#define OF_EXPOSURE_LUT_EPS 1e-3f

// One baked 3D table (openfilm::Lut3D): `size` nodes per axis, RGB interleaved, red index fastest.
// The domain is given as lo and span = (max - min), each rounded to float exactly the way
// Lut3D::lookup rounds them, so both sides normalise a pixel with the same numbers.
typedef struct {
    int size;
    int denseDarks;  // 1: nodes at t^2 (packed toward the domain minimum)
    float lo[3];
    float span[3];
} OFLut3D;

// One 1D curve (openfilm::Curve1D): `count` evenly spaced samples over [lo, hi], starting at
// element `offset` of the packed curve buffer.
typedef struct {
    float lo, hi;
    int offset, count;
} OFCurve;

// Inputs to film_kernel.
typedef struct {
    OFImageGeom src;
    OFImageGeom dst;
    int winX1, winY1, winX2, winY2;
    float gain;               // Exposure control as a linear multiplier, 2^stops
    float offsets[3];         // lab balance + printer lights + Print Density, in print log E
    OFLut3D lutE, lutPE, lutD;
    OFCurve neg[3], print[3];
    float displayMatrix[9];   // XYZ -> linear Rec.709, row-major
} OFFilmUniforms;

// ---- M4 spatial passes (spatial.metal) ----
// Work images are float4 per pixel, rows packed (pixel (x, y) at index y * width + x). Work pixel
// (0, 0) is image pixel (rectX1, rectY1); the rect is openfilm::workRect().

// Per-render inputs to the per-pixel stages (expose, develop, print expose, finish).
typedef struct {
    OFImageGeom src;
    OFImageGeom dst;
    int winX1, winY1, winX2, winY2;
    int rectX1, rectY1, width, height;  // work rect origin (image coords) and size
    float gain;                         // Exposure control, 2^stops
    float offsets[3];                   // print log E offsets (balance + printer lights + Print Density)
    OFLut3D lutE, lutPE, lutD;
    OFCurve neg[3], print[3];
    float displayMatrix[9];             // XYZ -> linear Rec.709, row-major
} OFSpatialUniforms;

// How a pass writes its result (blur V and bilinear up).
#define OF_STORE 0           // out = v
#define OF_STORE_WEIGHTED 1  // out.rgb = weight * v   (first lobe of a spread: 0 + w v)
#define OF_ACCUMULATE 2      // out.rgb += weight * v  (later lobes)

// Inputs to one blur / box-down / bilinear-up / combine pass.
typedef struct {
    int width, height;      // the grid the pass writes
    int inWidth, inHeight;  // the grid it reads (differs for box down and bilinear up)
    int factor;             // box / bilinear factor f
    int mode;               // OF_STORE..OF_ACCUMULATE; develop: 0 negative, 1 print
    int radius[3];          // per-channel blur radius (0 = copy)
    int tapOffset[3];       // where each channel's 2r+1 taps start in the taps buffer
    float weight[3];        // per-channel weight (lobe weight, layer weight, k, h w_c, bloom amount)
} OFPassUniforms;

// ---- Fused spread + combine + pointwise stages (sp_fused) ----
// One pass per spatial stage: it sums the stage's lobes (in lobe order, as spread() does), combines
// the sum with the stage's input pixel, then runs the pointwise stages that follow, all in registers.
#define OF_MAX_LOBES 8
#define OF_BLOCK 8     // pixels per thread along the blur in the blocked blurs and sp_fused
#define OF_HBLOCK 16   // the same for sp_blur_h_block (wide horizontal blurs)
#define OF_HTILE 256   // sp_blur_h_tiled: threads (pixels) per row segment

// How sp_fused gets one full-resolution or reduced lobe.
#define OF_LOBE_DIRECT 0   // full resolution, small kernel: horizontal pass computed in the fused pass
#define OF_LOBE_PRE_H 1    // full resolution: horizontal pass already in the T buffer; vertical in the fused pass
#define OF_LOBE_REDUCED 2  // reduced resolution, already blurred into the small-results buffer; bilinear up here

typedef struct {
    int radius[3];     // per-channel blur radius (0 = copy); DIRECT / PRE_H
    int tapOffset[3];  // where each channel's taps start
    float weight[3];   // per-channel weight of this lobe in the sum
    int factor;        // REDUCED: f
    int sw, sh;        // REDUCED: coarse grid size
    int smallOffset;   // REDUCED: first float of its result (three planes) in the small-results buffer
} OFLobe;

// Combine of the lobe sum S with the stage's input pixel X.
#define OF_COMBINE_STORE 0     // S (per-layer scatter)
#define OF_COMBINE_BLOOM 1     // (1 - a) X + a S, a = cw[0]
#define OF_COMBINE_HALATION 2  // X_c + cw_c S_r
#define OF_COMBINE_EDGE 3      // X_c + cw_c (X_c - S_c)

// Pointwise stages run after the combine, in this order (bit mask).
#define OF_POST_DEVELOP_NEG 1
#define OF_POST_PRINT_EXPOSE 2
#define OF_POST_DEVELOP_PRINT 4
#define OF_POST_FINISH 8  // grid is the render window; writes dst (as sp_finish)

// The stage's shape (combine, post, channels, hasAccum, lobe count and kinds) is also passed as
// function constants, so each stage gets its own specialised pipeline.
typedef struct {
    int width, height;  // work rect size
    float cw[3];        // combine weights
    OFLobe lobes[OF_MAX_LOBES];
} OFFuseUniforms;

#if defined(__cplusplus) && !defined(__METAL_VERSION__)
// The Metal compiler lays these out with the same rules (4-byte scalars, no padding); pin the
// layout on the C++ side so an edit that would desynchronise the two fails to compile.
#include <stddef.h>
static_assert(sizeof(OFImageGeom) == 20, "OFImageGeom layout");
static_assert(sizeof(OFLut3D) == 32 && offsetof(OFLut3D, lo) == 8 && offsetof(OFLut3D, span) == 20, "OFLut3D layout");
static_assert(sizeof(OFCurve) == 16 && offsetof(OFCurve, offset) == 8, "OFCurve layout");
static_assert(offsetof(OFFilmUniforms, dst) == 20 && offsetof(OFFilmUniforms, winX1) == 40 &&
                  offsetof(OFFilmUniforms, gain) == 56 && offsetof(OFFilmUniforms, offsets) == 60 &&
                  offsetof(OFFilmUniforms, lutE) == 72 && offsetof(OFFilmUniforms, lutPE) == 104 &&
                  offsetof(OFFilmUniforms, lutD) == 136 && offsetof(OFFilmUniforms, neg) == 168 &&
                  offsetof(OFFilmUniforms, print) == 216 && offsetof(OFFilmUniforms, displayMatrix) == 264 &&
                  sizeof(OFFilmUniforms) == 300,
              "OFFilmUniforms layout");
static_assert(offsetof(OFSpatialUniforms, dst) == 20 && offsetof(OFSpatialUniforms, winX1) == 40 &&
                  offsetof(OFSpatialUniforms, rectX1) == 56 && offsetof(OFSpatialUniforms, gain) == 72 &&
                  offsetof(OFSpatialUniforms, offsets) == 76 && offsetof(OFSpatialUniforms, lutE) == 88 &&
                  offsetof(OFSpatialUniforms, lutPE) == 120 && offsetof(OFSpatialUniforms, lutD) == 152 &&
                  offsetof(OFSpatialUniforms, neg) == 184 && offsetof(OFSpatialUniforms, print) == 232 &&
                  offsetof(OFSpatialUniforms, displayMatrix) == 280 && sizeof(OFSpatialUniforms) == 316,
              "OFSpatialUniforms layout");
static_assert(offsetof(OFPassUniforms, factor) == 16 && offsetof(OFPassUniforms, radius) == 24 &&
                  offsetof(OFPassUniforms, tapOffset) == 36 && offsetof(OFPassUniforms, weight) == 48 &&
                  sizeof(OFPassUniforms) == 60,
              "OFPassUniforms layout");
static_assert(sizeof(OFLobe) == 52 && offsetof(OFLobe, weight) == 24 && offsetof(OFLobe, smallOffset) == 48,
              "OFLobe layout");
static_assert(offsetof(OFFuseUniforms, cw) == 8 && offsetof(OFFuseUniforms, lobes) == 20 &&
                  sizeof(OFFuseUniforms) == 20 + 52 * OF_MAX_LOBES,
              "OFFuseUniforms layout");
#endif

#endif
