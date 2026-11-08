// Film-stock data as loaded from data/stocks/*.json: spectral sensitivities, characteristic
// curves and dye spectra, all on the shared 380-780 nm grid.
#pragma once

#include <array>
#include <string>
#include <vector>

#include "core/color.hpp"
#include "core/spectral.hpp"

namespace openfilm {

enum class StockType { Negative, Print };
enum class Densitometry { StatusM, StatusA };

// H&D curve: density versus log10 exposure, as a polyline. Outside the digitised range it holds
// the end values, which is how the toe and shoulder plateaus behave.
struct Curve {
  std::vector<double> logE, density;
  double eval(double logExposure) const;
  // Exposure that gives `d`; monotonic inverse, clamped to the curve's end exposures.
  double inverse(double d) const;
  double minLogE() const { return logE.front(); }
  double maxLogE() const { return logE.back(); }
};

struct Stock {
  std::string id, displayName;
  StockType type = StockType::Negative;
  Densitometry densitometry = Densitometry::StatusM;
  std::array<Spectrum, 3> sensitivity{};  // linear (10^log), 0 outside data; 0 = red layer
  std::array<Curve, 3> curve;             // same layer order as sensitivity
  std::array<Spectrum, 3> dye{};          // 0 = cyan, 1 = magenta, 2 = yellow
  Spectrum base{};                        // film base density; zeros if absent
  double balanceIlluminantK = 0, midGreyLogExposure = 0;
  Vec3 labAimDensity{};
};

// Throws std::runtime_error("<path>: <what>") when a field is missing, mistyped, unsorted, or a
// curve loses density as exposure rises.
Stock loadStock(const std::string& jsonPath);

// Every *.json in the directory except schema.json, sorted by id.
std::vector<Stock> loadStocks(const std::string& stocksDir);

}  // namespace openfilm
