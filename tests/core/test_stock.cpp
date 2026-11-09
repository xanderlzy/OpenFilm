// Tests for the stock loader: real stock files load, malformed ones are rejected with the field name.
#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

#include <nlohmann/json.hpp>

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
  const std::string m = R"([[1,1.0],[10,0.8]])";
  const std::string mtf = R"("mtf":{"red":)" + m + R"(,"green":)" + m + R"(,"blue":)" + m + "},";
  const std::string hal = type == "negative"
      ? R"("halation":{"baseThicknessMm":0.127,"remjetReflectance":{"on":0.01,"off":0.3},"layerSplit":[1,0.1,0]},)"
      : "";
  return R"({"id":"t","displayName":"T","type":")" + type + R"(","densitometry":"statusM",)" + extra + mtf + hal + ls +
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
                              R"("balanceIlluminantK":3200,"midGreyLogExposure":-1,"mtf":{},)"
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

TEST_CASE("density spectra are held at their edge values outside the digitised range") {
  // A density of 0 outside the chart would mean perfectly clear film, which lets UV through a
  // negative that is really dyed there. So dyes and base keep their last digitised value.
  const Stock neg = loadStock(kStocks + "/kodak-5219-type.json");
  const Stock prt = loadStock(kStocks + "/kodak-2383-type.json");
  CHECK(neg.dye[2][0] > 0.0);                 // 380 nm, below the chart's 400 nm start
  CHECK(neg.dye[2][0] == neg.dye[2][4]);      // equals the 400 nm value
  CHECK(neg.base[0] == neg.base[4]);
  for (int i = 0; i < 3; ++i) CHECK(prt.dye[i][80] == prt.dye[i][74]);  // 780 nm == 750 nm
}

TEST_CASE("sensitivities stay zero outside their digitised range") {
  // Unlike density, no data means no response.
  const Stock neg = loadStock(kStocks + "/kodak-5219-type.json");
  CHECK(neg.sensitivity[0][64] == 0.0);  // red layer ends at 680 nm
  CHECK(neg.sensitivity[0][24] == 0.0);  // and starts at 510 nm
}

namespace {
void checkMtf(const Stock& s) {
  for (int i = 0; i < 3; ++i) {
    const MtfCurve& m = s.mtf[i];
    REQUIRE(m.size() >= 8);
    CHECK(m.front()[1] >= 0.9);
    CHECK(m.front()[1] <= 1.3);
    for (size_t k = 0; k < m.size(); ++k) {
      CHECK(m[k][1] > 0.0);
      if (k > 0) CHECK(m[k][0] > m[k - 1][0]);
    }
  }
}
}  // namespace

TEST_CASE("5219 carries MTF and halation data") {
  const Stock s = loadStock(kStocks + "/kodak-5219-type.json");
  checkMtf(s);
  REQUIRE(s.halation.has_value());
  const Halation& h = *s.halation;
  CHECK(h.baseThicknessMm > 0.05);
  CHECK(h.baseThicknessMm < 0.3);
  CHECK(h.reflectanceOff > h.reflectanceOn);
  CHECK(h.reflectanceOn > 0.0);
  CHECK(h.layerSplit[0] == 1.0);
  CHECK(h.layerSplit[0] > h.layerSplit[1]);
  CHECK(h.layerSplit[1] > h.layerSplit[2]);
  CHECK(h.layerSplit[2] >= 0.0);
}

TEST_CASE("2383 carries MTF, no halation") {
  const Stock s = loadStock(kStocks + "/kodak-2383-type.json");
  checkMtf(s);
  CHECK(!s.halation.has_value());
}

TEST_CASE("malformed MTF is rejected") {
  std::ifstream in(kStocks + "/kodak-5219-type.json");
  std::stringstream ss;
  ss << in.rdbuf();
  const nlohmann::json good = nlohmann::json::parse(ss.str());
  CHECK(!loadError("mtfgood", good.dump()).size());  // control
  nlohmann::json rev = good;
  std::reverse(rev["mtf"]["green"].begin(), rev["mtf"]["green"].end());
  CHECK(loadError("mtfrev", rev.dump()).find("mtf") != std::string::npos);
  nlohmann::json nohal = good;
  nohal.erase("halation");
  CHECK(loadError("nohal", nohal.dump()).find("halation") != std::string::npos);
  nlohmann::json nomtf = good;
  nomtf.erase("mtf");
  CHECK(loadError("nomtf", nomtf.dump()).find("mtf") != std::string::npos);
}
