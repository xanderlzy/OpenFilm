#include "core/film_model.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace openfilm {

namespace {
constexpr double kPrinterPoint = 0.025;  // log E per printer point
constexpr double kGreyCode = 0.409;      // output code for 18 % grey
const Stock& findStock(const std::vector<Stock>& stocks, const std::string& id) {
  for (const Stock& s : stocks) if (s.id == id) return s;
  throw std::runtime_error("unknown stock id: " + id);
}
Xyz xyzOfXy(const Xy& c) { return {c.x / c.y, 1.0, (1.0 - c.x - c.y) / c.y}; }

// The projector white (D55) is re-expressed as D65 so a neutral print reads neutral on a 709 display.
Mat3 d55ToD65(const SpectralTables& t) {
  const Xyz w = integrate(t.daylight(5503.0), t);
  return bradford({w.X / w.Y, 1.0, w.Z / w.Y}, xyzOfXy(kD65));
}

struct Fnv {
  uint64_t h = 1469598103934665603ull;
  void bytes(const void* p, size_t n) {
    const unsigned char* b = static_cast<const unsigned char*>(p);
    for (size_t i = 0; i < n; ++i) { h ^= b[i]; h *= 1099511628211ull; }
  }
  void str(const std::string& s) { const uint64_t n = s.size(); bytes(&n, sizeof n); bytes(s.data(), s.size()); }
  void num(double v) { if (v == 0) v = 0; bytes(&v, sizeof v); }  // folds -0 into +0
};
}  // namespace

uint64_t FilmParams::hash() const {
  Fnv f;
  f.str(negativeId); f.str(printId);
  f.num(gamutCompression);
  return f.h;
}

std::shared_ptr<const FilmResources> FilmResources::load(const std::string& dir) {
  return load(dir + "/data", dir + "/rec2020.coeff");
}

std::shared_ptr<const FilmResources> FilmResources::load(const std::string& dataDir, const std::string& coeffPath) {
  const std::string spectral = dataDir + "/spectral";
  return std::make_shared<const FilmResources>(FilmResources{
      SpectralTables::load(spectral), Rgb2Spec::load(coeffPath),
      Densitometer::load(spectral, Densitometry::StatusM),
      Densitometer::load(spectral, Densitometry::StatusA), loadStocks(dataDir + "/stocks")});
}

const Stock& FilmResources::stock(const std::string& id) const { return findStock(stocks, id); }

FilmModel::FilmModel(std::shared_ptr<const FilmResources> resources, const FilmParams& params)
    : res_(std::move(resources)), params_(params),
      negative_(res_->stock(params.negativeId), res_->tables, res_->rgb2spec, res_->statusM, params.gamutCompression),
      print_(res_->stock(params.printId), res_->tables, res_->statusA),
      adapt_(d55ToD65(res_->tables)),
      toRec709_(inverse(rgbToXyz(kRec709))) {
  // Balance: find printer-light offsets that make 18 % grey print to the lab-aim density on
  // green and to a D65-neutral chromaticity, the way a lab timer sets the printer lights.
  const double g = diEncode(0.18);
  const Vec3 dg = negative_.density(negative_.logExposure({g, g, g}));
  const Spectrum negSpec = negative_.transmittanceDensity(dg);
  const Vec3 e0 = print_.printLogExposure(negSpec, {0, 0, 0});
  const Stock& ps = print_.stock();
  const auto d65 = uvPrime(xyzOfXy(kD65));

  auto residual = [&](const Vec3& b) -> Vec3 {
    const Vec3 e{e0[0] + b[0], e0[1] + b[1], e0[2] + b[2]};
    const Xyz c = adaptedXyz(dg, b);
    const auto uv = uvPrime(c);
    return {uv[0] - d65[0], uv[1] - d65[1], (print_.printDensity(e)[1] - ps.labAimDensity[1]) / 10.0};
  };
  auto norm = [](const Vec3& r) { return std::sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2]); };

  Vec3 b;
  for (int j = 0; j < 3; ++j) b[j] = ps.curve[j].inverse(ps.labAimDensity[j]) - e0[j];
  Vec3 r = residual(b);
  bool ok = norm(r) < 1e-7;
  for (int it = 0; it < 50 && !ok; ++it) {
    const double h = 1e-4;
    Mat3 jac;  // jac[row][col]
    for (int c = 0; c < 3; ++c) {
      Vec3 bp = b; bp[c] += h;
      const Vec3 rp = residual(bp);
      for (int row = 0; row < 3; ++row) jac[row][c] = (rp[row] - r[row]) / h;
    }
    const Vec3 step = mul(inverse(jac), r);
    // Damp (halve) the step while it makes things worse.
    double t = 1.0;
    Vec3 nb = b, nr = r;
    for (int k = 0; k < 20; ++k) {
      for (int c = 0; c < 3; ++c) nb[c] = b[c] - t * step[c];
      nr = residual(nb);
      if (std::isfinite(norm(nr)) && norm(nr) < norm(r)) break;
      t *= 0.5;
    }
    b = nb; r = nr;
    ok = norm(r) < 1e-7;
  }
  if (!ok) throw std::runtime_error("print balance did not converge");
  balance_ = b;

  const Xyz grey = adaptedXyz(dg, balance_);
  anchorScale_ = std::pow(kGreyCode, 2.4) / grey.Y;
}

Xyz FilmModel::adaptedXyz(const Vec3& negStatusMDensity, const Vec3& offsets) const {
  const Spectrum negSpec = negative_.transmittanceDensity(negStatusMDensity);
  const Xyz c = print_.projectedXyz(print_.printLogExposure(negSpec, offsets));
  const Vec3 a = mul(adapt_, Vec3{c.X, c.Y, c.Z});
  return {a[0], a[1], a[2]};
}

Vec3 FilmModel::negativeLogExposure(const Vec3& diCode) const { return negative_.logExposure(diCode); }
Vec3 FilmModel::negativeDensity(const Vec3& logE) const { return negative_.density(logE); }

std::array<double, 2> FilmModel::negativeLogERange(int ch) const {
  const Curve& c = negative_.curves()[size_t(ch)];
  return {c.minLogE(), c.maxLogE()};
}

Vec3 printOffsetsLogE(const Vec3& balance, const LookControls& c) {
  auto finite = [](double v) { return std::isfinite(v) ? v : 0.0; };
  Vec3 off;
  for (int j = 0; j < 3; ++j)
    off[j] = balance[j] + kPrinterPoint * (finite(c.printerPoints[j]) + finite(c.printDensity));
  return off;
}

std::array<double, 2> FilmModel::printLogERange(int ch) const {
  const Curve& c = print_.stock().curve[size_t(ch)];
  return {c.minLogE(), c.maxLogE()};
}

std::array<double, 2> FilmModel::printDensityRange(int ch) const {
  const auto& d = print_.stock().curve[size_t(ch)].density;
  return {*std::min_element(d.begin(), d.end()), *std::max_element(d.begin(), d.end())};
}

Vec3 FilmModel::printerOffsets(const LookControls& c) const { return printOffsetsLogE(balance_, c); }

Vec3 FilmModel::printLogExposureBase(const Vec3& negStatusMDensity) const {
  return print_.printLogExposure(negative_.transmittanceDensity(negStatusMDensity), {0, 0, 0});
}

Vec3 FilmModel::printDensity(const Vec3& printLogE) const { return print_.printDensity(printLogE); }

Vec3 FilmModel::displayFromPrintDensity(const Vec3& printStatusADensity) const {
  const Xyz p = print_.projectedXyzFromDensity(printStatusADensity);
  const Vec3 a = mul(adapt_, Vec3{p.X, p.Y, p.Z});
  const Vec3 rgb = mul(toRec709_, a);
  return {bt1886Encode(rgb[0] * anchorScale_), bt1886Encode(rgb[1] * anchorScale_),
          bt1886Encode(rgb[2] * anchorScale_)};
}

Vec3 FilmModel::printOutput(const Vec3& negStatusMDensity, const LookControls& c) const {
  Vec3 e = printLogExposureBase(negStatusMDensity);
  const Vec3 off = printerOffsets(c);
  for (int j = 0; j < 3; ++j) e[j] += off[j];
  return displayFromPrintDensity(printDensity(e));
}

Vec3 FilmModel::evaluate(const Vec3& diCode, const LookControls& c) const {
  Vec3 e = negative_.logExposure(diCode);
  const double stops = std::isfinite(c.exposureStops) ? c.exposureStops : 0.0;
  for (double& v : e) v += stops * std::log10(2.0);
  return printOutput(negative_.density(e), c);
}

}  // namespace openfilm
