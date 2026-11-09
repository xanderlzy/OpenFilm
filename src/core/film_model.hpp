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

  uint64_t hash() const;  // bake inputs only; the controls below never enter it
};

struct LookControls {            // per-render; never part of the bake
  double exposureStops = 0;
  Vec3 printerPoints{0, 0, 0};   // R, G, B points
  double printDensity = 0;       // points added to all three
};

// The lab's auto-balance plus the printer lights: balance + 0.025 log E per point (printer
// points and Print Density add), per channel. Non-finite control values count as 0.
Vec3 printOffsetsLogE(const Vec3& balance, const LookControls& c);

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

  // Negative Status M density -> exposure on each print layer, with no printer-light offsets.
  Vec3 printLogExposureBase(const Vec3& negStatusMDensity) const;
  // Print Status A density from the print's H&D curves.
  Vec3 printDensity(const Vec3& printLogE) const;
  // Projector light through the print dyes -> D65 -> Rec.709 -> grey anchor: linear display light,
  // unclamped (a saturated dye can push a channel below 0, outside the Rec.709 gamut).
  Vec3 displayLinearFromPrintDensity(const Vec3& printStatusADensity) const;
  // log10 of the projected XYZ (floor 1e-10 per component). Nearly linear in dye density
  // (Beer-Lambert), so a table of it interpolates far better than one of linear light.
  Vec3 printLogXyzFromPrintDensity(const Vec3& printStatusADensity) const;
  // Projected XYZ -> linear display light in one matrix: Rec.709 . Bradford D55->D65 . grey anchor.
  Mat3 displayMatrix() const;
  // The same, clipped at black and encoded as BT.1886 code.
  Vec3 displayFromPrintDensity(const Vec3& printStatusADensity) const;

  // Digitised log E span and min..max density of the print's curve for layer `ch` (0 = red).
  std::array<double, 2> printLogERange(int ch) const;
  std::array<double, 2> printDensityRange(int ch) const;

  // Log E offsets on the print layers: auto-balance plus the printer controls.
  Vec3 printerOffsets(const LookControls& c) const;

  // Negative Status M density -> Rec.709 display code.
  Vec3 printOutput(const Vec3& negStatusMDensity, const LookControls& c) const;

  // DI pixel pushed `c.exposureStops` over/under, through the whole chain.
  Vec3 evaluate(const Vec3& diCode, const LookControls& c) const;
  const Vec3& balance() const { return balance_; }
  const Stock& negativeStock() const { return res_->stock(params_.negativeId); }
  const Stock& printStock() const { return res_->stock(params_.printId); }
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
