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
#endif

#endif
