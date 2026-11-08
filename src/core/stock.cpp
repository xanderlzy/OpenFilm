#include "core/stock.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <stdexcept>

#include <nlohmann/json.hpp>

namespace openfilm {

using json = nlohmann::json;

double Curve::eval(double x) const {
  if (x <= logE.front()) return density.front();
  if (x >= logE.back()) return density.back();
  const size_t k = std::upper_bound(logE.begin(), logE.end(), x) - logE.begin() - 1;
  const double t = (x - logE[k]) / (logE[k + 1] - logE[k]);
  return density[k] + t * (density[k + 1] - density[k]);
}

double Curve::inverse(double d) const {
  if (d <= density.front()) return logE.front();
  if (d >= density.back()) return logE.back();
  // First segment that reaches d; flat segments (plateaus) are skipped since d is strictly inside.
  for (size_t k = 0; k + 1 < density.size(); ++k) {
    if (d >= density[k] && d <= density[k + 1] && density[k + 1] > density[k]) {
      const double t = (d - density[k]) / (density[k + 1] - density[k]);
      return logE[k] + t * (logE[k + 1] - logE[k]);
    }
  }
  return logE.back();
}

namespace {

struct Ctx {
  const std::string& path;
  [[noreturn]] void fail(const std::string& what) const { throw std::runtime_error(path + ": " + what); }

  const json& field(const json& o, const char* name, const std::string& where = "") const {
    if (!o.is_object() || !o.contains(name)) fail("missing field '" + where + name + "'");
    return o[name];
  }
  std::string str(const json& o, const char* name) const {
    const json& v = field(o, name);
    if (!v.is_string()) fail(std::string("field '") + name + "' must be a string");
    return v.get<std::string>();
  }
  double num(const json& o, const char* name) const {
    const json& v = field(o, name);
    if (!v.is_number()) fail(std::string("field '") + name + "' must be a number");
    return v.get<double>();
  }
  // [[x, y], ...] with strictly ascending x.
  std::vector<std::array<double, 2>> pairs(const json& v, const std::string& name) const {
    if (!v.is_array() || v.size() < 2) fail("field '" + name + "' must be an array of at least 2 [x, y] pairs");
    std::vector<std::array<double, 2>> out;
    for (const json& p : v) {
      if (!p.is_array() || p.size() != 2 || !p[0].is_number() || !p[1].is_number())
        fail("field '" + name + "' must contain [x, y] number pairs");
      out.push_back({p[0].get<double>(), p[1].get<double>()});
      if (out.size() > 1 && !(out[out.size() - 1][0] > out[out.size() - 2][0]))
        fail("field '" + name + "' is not sorted ascending");
    }
    return out;
  }
};

}  // namespace

Stock loadStock(const std::string& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error(path + ": cannot open stock file");
  json j;
  try {
    j = json::parse(in);
  } catch (const std::exception& e) {
    throw std::runtime_error(path + ": invalid JSON (" + e.what() + ")");
  }
  const Ctx c{path};
  Stock s;
  s.id = c.str(j, "id");
  s.displayName = c.str(j, "displayName");

  const std::string type = c.str(j, "type");
  if (type == "negative") s.type = StockType::Negative;
  else if (type == "print") s.type = StockType::Print;
  else c.fail("field 'type' must be 'negative' or 'print'");

  const std::string dens = c.str(j, "densitometry");
  if (dens == "statusM") s.densitometry = Densitometry::StatusM;
  else if (dens == "statusA") s.densitometry = Densitometry::StatusA;
  else c.fail("field 'densitometry' must be 'statusM' or 'statusA'");

  // A negative is exposed by a scene (needs its balance light and grey anchor); a print is judged
  // against the lab's aim density.
  if (s.type == StockType::Negative) {
    s.balanceIlluminantK = c.num(j, "balanceIlluminantK");
    s.midGreyLogExposure = c.num(j, "midGreyLogExposure");
  } else {
    const json& a = c.field(j, "labAimDensity");
    if (!a.is_array() || a.size() != 3 || !a[0].is_number() || !a[1].is_number() || !a[2].is_number())
      c.fail("field 'labAimDensity' must be an array of 3 numbers");
    s.labAimDensity = {a[0].get<double>(), a[1].get<double>(), a[2].get<double>()};
  }

  const json& layers = c.field(j, "layers");
  const char* layerNames[3] = {"red", "green", "blue"};
  for (int i = 0; i < 3; ++i) {
    const std::string where = std::string("layers.") + layerNames[i] + ".";
    const json& layer = c.field(layers, layerNames[i], "layers.");
    auto logSens = c.pairs(c.field(layer, "logSensitivity", where), where + "logSensitivity");
    // Sensitivity is stored as log10; the model multiplies it as a linear quantity.
    for (auto& p : logSens) p[1] = std::pow(10.0, p[1]);
    s.sensitivity[i] = resample(logSens, 0.0);
    const auto cur = c.pairs(c.field(layer, "characteristicCurve", where), where + "characteristicCurve");
    for (size_t k = 0; k < cur.size(); ++k) {
      s.curve[i].logE.push_back(cur[k][0]);
      s.curve[i].density.push_back(cur[k][1]);
      if (k > 0 && cur[k][1] < cur[k - 1][1])
        c.fail("field '" + where + "characteristicCurve' density decreases with exposure");
    }
  }

  const json& dyes = c.field(j, "dyeDensity");
  const char* dyeNames[3] = {"cyan", "magenta", "yellow"};
  for (int i = 0; i < 3; ++i) {
    const std::string where = std::string("dyeDensity.") + dyeNames[i];
    s.dye[i] = resample(c.pairs(c.field(dyes, dyeNames[i], "dyeDensity."), where), 0.0);
  }
  s.base.fill(0.0);
  if (j.contains("baseDensity")) s.base = resample(c.pairs(j["baseDensity"], "baseDensity"), 0.0);
  return s;
}

std::vector<Stock> loadStocks(const std::string& dir) {
  std::vector<Stock> out;
  for (const auto& e : std::filesystem::directory_iterator(dir)) {
    if (e.path().extension() == ".json" && e.path().filename() != "schema.json")
      out.push_back(loadStock(e.path().string()));
  }
  std::sort(out.begin(), out.end(), [](const Stock& a, const Stock& b) { return a.id < b.id; });
  return out;
}

}  // namespace openfilm
