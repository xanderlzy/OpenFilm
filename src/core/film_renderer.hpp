// CPU film renderer: DI pixels -> baked LUTs -> Rec.709 display code.
#pragma once

#include "core/image.hpp"
#include "core/lut.hpp"

namespace openfilm {

// Same window, bounds and padding rules as renderExposureCpu: only pixels inside `win` are
// written, alpha is copied, and source pixels outside `src` become (0,0,0,0). Non-finite RGB is
// treated as 0, and non-finite controls count as 0 (the pipeline's NaN guards).
void renderFilmCpu(const ConstImageView& src, const ImageView& dst, const RenderWindow& win, const BakedFilm& film,
                   const LookControls& controls);

// Exposure control -> linear gain, 2^clamp(stops, -64, 64); non-finite -> 1. The clamp keeps 0 x gain
// from ever being NaN. Shared with the GPU host code.
float exposureGain(double stops);

}  // namespace openfilm
