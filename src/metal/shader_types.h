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

#endif
