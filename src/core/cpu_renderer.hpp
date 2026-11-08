// CPU reference renderer.
//
// This is the "ground truth" implementation of the pipeline. The Metal (GPU) path must produce
// the same numbers; tests/gpu compares the two. Resolve also uses it when GPU rendering is off.
#pragma once

#include "core/image.hpp"
#include "core/render_params.hpp"

namespace openfilm {

// Processes the pixels inside `win` from `src` into `dst`. Pixels of `dst` outside `win` are left
// untouched. Where `win` reaches outside `src`, the output is transparent black (0,0,0,0).
void renderCpu(const ConstImageView& src, const ImageView& dst, const RenderWindow& win, const RenderParams& p);

}  // namespace openfilm
