// Image views: how OpenFilm looks at a block of pixels without owning it.
//
// Resolve (the OFX "host") owns every frame's memory. It hands the plugin a pointer plus a
// description of the layout. We copy that description into ImageGeometry so the film maths
// never has to know about OFX.
//
// Layout facts worth knowing:
//  * Pixels are 4 floats: R, G, B, A (32-bit float, scene-linear or log, never clamped).
//  * Bounds (x1,y1)-(x2,y2) are in frame pixel coordinates, x2/y2 exclusive. They do NOT have to
//    start at (0,0): a host may give us just a region of the frame.
//  * rowBytes is the distance in bytes from one row to the next. It can be larger than
//    width * 16 (padding), so always step rows with rowBytes, never with width.
#pragma once

#include <cstddef>

namespace openfilm {

struct ImageGeometry {
    int x1, y1, x2, y2;          // bounds, x2/y2 exclusive
    std::ptrdiff_t rowBytes;     // bytes between the starts of consecutive rows
};

struct ImageView {
    float* data;                 // first pixel of row y1
    ImageGeometry geom;
};

struct ConstImageView {
    const float* data;
    ImageGeometry geom;
};

// The rectangle the host asked us to produce this time (x2/y2 exclusive).
struct RenderWindow {
    int x1, y1, x2, y2;
};

inline bool contains(const ImageGeometry& g, int x, int y) {
    return x >= g.x1 && x < g.x2 && y >= g.y1 && y < g.y2;
}

// Address of pixel (x,y) in frame coordinates, or nullptr if it lies outside the image.
inline const float* pixelAddress(const ConstImageView& img, int x, int y) {
    if (!img.data || !contains(img.geom, x, y)) return nullptr;
    const char* row = reinterpret_cast<const char*>(img.data) + std::ptrdiff_t(y - img.geom.y1) * img.geom.rowBytes;
    return reinterpret_cast<const float*>(row) + std::ptrdiff_t(x - img.geom.x1) * 4;
}

inline float* pixelAddress(const ImageView& img, int x, int y) {
    return const_cast<float*>(pixelAddress(ConstImageView{img.data, img.geom}, x, y));
}

}  // namespace openfilm
