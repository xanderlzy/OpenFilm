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

// Inputs to exposure_kernel.
typedef struct {
    OFImageGeom src;
    OFImageGeom dst;
    int winX1, winY1, winX2, winY2;
    float gain;
} OFExposureUniforms;

#endif
