// Gamut compression: pulls colours that fall outside Rec.2020 (blue LEDs, neon) smoothly back
// inside, so the RGB-to-spectrum step only ever sees colours a real light could produce.
#pragma once

#include "core/color.hpp"

namespace openfilm {

// Per channel, the largest "distance" (how far the channel sits below the brightest channel,
// as a fraction of it) reached by any colour inside the DaVinci Wide Gamut once expressed in
// Rec.2020. Always at least 1.0001. This is the distance that gets mapped exactly to 1.0.
Vec3 dwgLimitsInRec2020();

// ACES-style per-channel distance compression in linear Rec.2020. amount 0 hard-clips at the
// gamut edge; amount 1 starts compressing at 65 % distance. Never returns a negative channel.
Vec3 gamutCompress(const Vec3& rec2020Linear, double amount, const Vec3& limits);

}  // namespace openfilm
