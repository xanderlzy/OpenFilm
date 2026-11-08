// Tests for the stock loader: real stock files load, malformed ones are rejected with the field name.
#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>

#include "core/stock.hpp"

using namespace openfilm;

namespace {
const std::string kStocks = OPENFILM_SOURCE_DATA_DIR "/stocks";

// Index of the wavelength (nm) where a spectrum is largest.
double peakNm(const Spectrum& s) {
  return wavelength(static_cast<int>(std::max_element(s.begin(), s.end()) - s.begin()));
}

// Writes `body` to a temp file and returns the exception message loadStock throws ("" if none).
std::string loadError(const std::string& name, const std::string& body) {
  const auto path = std::filesystem::temp_directory_path() / ("openfilm_" + name + ".json");
  { std::ofstream(path) << body; }
  std::string msg;
  try { loadStock(path.string()); } catch (const std::runtime_error& e) { msg = e.what(); }
  std::filesystem::remove(path);
  return msg;
}

const char* kDye = R"([[500,0.1],[505,0.2]])";
std::string stockJson(const std::string& layers, const std::string& curve, const std::string& extra,
                      const std::string& type = "negative") {
  std::string layer = R"({"logSensitivity":[[500,-1],[505,-1]],"characteristicCurve":)" + curve + "}";
  std::string ls = layers.empty() ? R"("layers":{"red":)" + layer + ",\"green\":" + layer + ",\"blue\":" + layer + "}" : layers;
  return R"({"id":"t","displayName":"T","type":")" + type + R"(","densitometry":"statusM",)" + extra + ls +
         R"(,"dyeDensity":{"cyan":)" + kDye + R"(,"magenta":)" + kDye + R"(,"yellow":)" + kDye + "}}";
}
const char* kNegExtra = R"("balanceIlluminantK":3200,"midGreyLogExposure":-1.5,)";
const char* kGoodCurve = "[[-2,0.1],[-1,0.5],[0,1.0]]";
}  // namespace

TEST_CASE("loads 5219-type") {
  const Stock s = loadStock(kStocks + "/kodak-5219-type.json");
  CHECK(s.type == StockType::Negative);
  CHECK(s.densitometry == Densitometry::StatusM);
  CHECK(std::isfinite(s.midGreyLogExposure));
  CHECK(s.balanceIlluminantK == 3200.0);
  for (int i = 0; i < 3; ++i) {
    CHECK(s.curve[i].logE.size() >= 20);
    CHECK(std::is_sorted(s.curve[i].density.begin(), s.curve[i].density.end()));
  }
  CHECK(peakNm(s.sensitivity[2]) >= 400.0);  CHECK(peakNm(s.sensitivity[2]) <= 500.0);
  CHECK(peakNm(s.sensitivity[1]) >= 500.0);  CHECK(peakNm(s.sensitivity[1]) <= 600.0);
  CHECK(peakNm(s.sensitivity[0]) >= 600.0);  CHECK(peakNm(s.sensitivity[0]) <= 700.0);
}

TEST_CASE("loads 2383-type") {
  const Stock s = loadStock(kStocks + "/kodak-2383-type.json");
  CHECK(s.type == StockType::Print);
  CHECK(s.densitometry == Densitometry::StatusA);
  for (double v : s.labAimDensity) { CHECK(v >= 0.8); CHECK(v <= 1.3); }
}

TEST_CASE("rejects malformed") {
  CHECK(!loadError("good", stockJson("", kGoodCurve, kNegExtra)).size());  // control: the fixture is valid
  CHECK(loadError("nolayers", R"({"id":"t","displayName":"T","type":"negative","densitometry":"statusM",)"
                              R"("balanceIlluminantK":3200,"midGreyLogExposure":-1,)"
                              R"("dyeDensity":{"cyan":[[500,1],[505,1]],"magenta":[[500,1],[505,1]],"yellow":[[500,1],[505,1]]}})")
            .find("layers") != std::string::npos);
  CHECK(loadError("unsorted", stockJson("", "[[-1,0.5],[-2,0.1],[0,1.0]]", kNegExtra))
            .find("characteristicCurve") != std::string::npos);
  CHECK(loadError("decreasing", stockJson("", "[[-2,0.5],[-1,0.1],[0,1.0]]", kNegExtra))
            .find("characteristicCurve") != std::string::npos);
  CHECK(loadError("noaim", stockJson("", kGoodCurve, "", "print")).find("labAimDensity") != std::string::npos);
  CHECK(loadError("nobalance", stockJson("", kGoodCurve, "")).find("balanceIlluminantK") != std::string::npos);
}

TEST_CASE("loadStocks lists both, sorted") {
  const auto all = loadStocks(kStocks);
  REQUIRE(all.size() == 2);
  CHECK(all[0].id == "kodak-2383-type");
  CHECK(all[1].id == "kodak-5219-type");
}

TEST_CASE("curve eval and inverse") {
  Curve c;
  c.logE = {0.0, 1.0, 2.0, 3.0};
  c.density = {0.2, 0.2, 1.2, 1.5};  // flat toe, then a rise, then a shoulder
  CHECK(c.eval(-5.0) == 0.2);
  CHECK(c.eval(10.0) == 1.5);
  CHECK(c.eval(1.5) == doctest::Approx(0.7));
  CHECK(c.inverse(0.0) == 0.0);
  CHECK(c.inverse(9.0) == 3.0);
  for (double x : {1.1, 1.5, 1.9, 2.4}) CHECK(c.inverse(c.eval(x)) == doctest::Approx(x));
}
