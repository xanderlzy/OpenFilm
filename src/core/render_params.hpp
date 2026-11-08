// Everything the film pipeline needs to know for one frame, independent of OFX.
// The OFX layer reads the user's knobs and fills this struct; core code only ever sees this.
#pragma once

namespace openfilm {

struct RenderParams {
    // Negative exposure in photographic stops. +1 doubles the light reaching the film.
    double exposureStops = 0.0;
};

}  // namespace openfilm
