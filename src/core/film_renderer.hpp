// CPU film renderer: DI pixels -> baked LUTs -> Rec.709 display code.
#pragma once

#include "core/image.hpp"
#include "core/lut.hpp"

namespace openfilm {

// Same window, bounds and padding rules as renderExposureCpu: only pixels inside `win` are
// written, alpha is copied, and source pixels outside `src` become (0,0,0,0). Non-finite RGB is
// treated as 0 (this is the pipeline's single NaN guard).
void renderFilmCpu(const ConstImageView& src, const ImageView& dst, const RenderWindow& win, const BakedFilm& film,
                   double exposureStops);

}  // namespace openfilm
