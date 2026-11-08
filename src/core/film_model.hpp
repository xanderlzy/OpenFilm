// The complete film look: DI pixel -> negative -> printer lights -> 2383 print -> projected
// light -> Rec.709 display code. Printer lights are auto-balanced so 18 % grey prints neutral.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "core/color.hpp"
#include "core/densitometry.hpp"
#include "core/negative.hpp"
#include "core/print.hpp"
#include "core/rgb2spec.hpp"
#include "core/spectral.hpp"
#include "core/stock.hpp"

namespace openfilm {

struct FilmParams {
  std::string negativeId, printId;
  double gamutCompression = 0.5;
  Vec3 printerLights{0, 0, 0};  // printer points per channel; one point = 0.025 log E
  double printDensity = 0;      // printer points added to all three channels

  uint64_t hash() const;
};

// Everything loaded from disk once and shared by every FilmModel.
struct FilmResources {
  SpectralTables tables;
  Rgb2Spec rgb2spec;
  Densitometer statusM, statusA;
  std::vector<Stock> stocks;

  // Layout: <dir>/data/spectral, <dir>/data/stocks, <dir>/rec2020.coeff.
  static std::shared_ptr<const FilmResources> load(const std::string& resourcesDir);
  static std::shared_ptr<const FilmResources> load(const std::string& dataDir, const std::string& coeffPath);

  // Throws std::runtime_error on an unknown id.
  const Stock& stock(const std::string& id) const;
};

class FilmModel {
 public:
  FilmModel(std::shared_ptr<const FilmResources> resources, const FilmParams& params);

  Vec3 negativeLogExposure(const Vec3& diCode) const;
  Vec3 negativeDensity(const Vec3& logE) const;

  // Log E span of the negative's digitised H&D curve for layer `ch` (0 = red).
  std::array<double, 2> negativeLogERange(int ch) const;

  // Negative Status M density -> Rec.709 display code.
  Vec3 printOutput(const Vec3& negStatusMDensity) const;

  // DI pixel pushed `exposureStops` over/under, through the whole chain.
  Vec3 evaluate(const Vec3& diCode, double exposureStops) const;

  // Log E offsets applied to the print layers: auto-balance plus the user's printer points.
  // Exposed for tests and diagnostics.
  Vec3 printerOffsets() const;
  const Vec3& balance() const { return balance_; }
  double anchorScale() const { return anchorScale_; }

 private:
  Xyz adaptedXyz(const Vec3& negStatusMDensity, const Vec3& offsets) const;

  std::shared_ptr<const FilmResources> res_;  // declared first: the models below reference it
  FilmParams params_;
  NegativeModel negative_;
  PrintModel print_;
  Mat3 adapt_;     // D55 -> D65 Bradford
  Mat3 toRec709_;  // XYZ -> linear Rec.709
  Vec3 balance_{0, 0, 0};
  double anchorScale_ = 1;
};

}  // namespace openfilm
