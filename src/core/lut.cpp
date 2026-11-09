#include "core/lut.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <future>
#include <thread>

namespace openfilm {

Lut3D::Lut3D(int size, Vec3 domainMin, Vec3 domainMax, bool denseDarks)
    : size_(size), min_(domainMin), max_(domainMax), denseDarks_(denseDarks), data_(size_t(size) * size * size * 3, 0.f) {}

void Lut3D::bake(const std::function<Vec3(const Vec3&)>& f) {
  const int n = size_;
  auto slices = [&](int b0, int b1) {
    for (int b = b0; b < b1; ++b)
      for (int g = 0; g < n; ++g)
        for (int r = 0; r < n; ++r) {
          const int idx[3] = {r, g, b};
          Vec3 in;
          for (int k = 0; k < 3; ++k) {
            double t = double(idx[k]) / double(n - 1);
            if (denseDarks_) t *= t;  // nodes bunch up near the domain minimum
            in[k] = min_[k] + (max_[k] - min_[k]) * t;
          }
          const Vec3 out = f(in);
          float* dst = &data_[((size_t(b) * n + g) * n + r) * 3];
          for (int k = 0; k < 3; ++k) dst[k] = float(out[k]);
        }
  };
  const int threads = std::clamp(int(std::thread::hardware_concurrency()), 1, n);
  std::vector<std::thread> pool;
  for (int t = 0; t < threads; ++t) pool.emplace_back(slices, n * t / threads, n * (t + 1) / threads);
  for (auto& th : pool) th.join();
}

std::array<float, 3> Lut3D::lookup(float a, float b, float c) const {
  const int n = size_;
  const float in[3] = {a, b, c};
  int i0[3];
  float f[3];
  for (int k = 0; k < 3; ++k) {
    float t = std::clamp((in[k] - float(min_[k])) / float(max_[k] - min_[k]), 0.f, 1.f);  // NaN fails both compares
    if (!(t == t)) t = 0.f;
    if (denseDarks_) t = std::sqrt(t);  // inverse of the node spacing used in bake()
    const float u = t * float(n - 1);
    int i = std::min(int(u), n - 2);
    i0[k] = i;
    f[k] = u - float(i);
  }
  const size_t sr = 3, sg = size_t(n) * 3, sb = size_t(n) * n * 3;
  const float* base = &data_[size_t(i0[2]) * sb + size_t(i0[1]) * sg + size_t(i0[0]) * sr];

  // Tetrahedral: sort the fractional offsets, then walk the cube corner by corner from the
  // origin along the largest axis first. Four samples instead of eight, and no colour bleeding
  // across the grey diagonal the way trilinear has.
  const size_t stride[3] = {sr, sg, sb};
  int order[3] = {0, 1, 2};
  std::sort(order, order + 3, [&](int x, int y) { return f[x] > f[y]; });
  const float* p0 = base;
  const float* p1 = p0 + stride[order[0]];
  const float* p2 = p1 + stride[order[1]];
  const float* p3 = p2 + stride[order[2]];
  const float w0 = 1.f - f[order[0]], w1 = f[order[0]] - f[order[1]], w2 = f[order[1]] - f[order[2]], w3 = f[order[2]];
  std::array<float, 3> out;
  for (int k = 0; k < 3; ++k) out[k] = w0 * p0[k] + w1 * p1[k] + w2 * p2[k] + w3 * p3[k];
  return out;
}

float Curve1D::eval(float x) const {
  const float u = std::clamp((x - min) / (max - min), 0.f, 1.f) * float(values.size() - 1);
  const size_t i = std::min(size_t(u), values.size() - 2);
  const float t = u - float(i);
  return values[i] + (values[i + 1] - values[i]) * t;
}

void Lut3D::roundToHalf() {
  for (float& v : data_) v = float(_Float16(v));
}

std::shared_ptr<const BakedFilm> bakeFilm(const FilmModel& model, int lutSizeE, int lutSizePE, int lutSizeD) {
  Lut3D lutE(lutSizeE, {0, 0, 0}, {1, 1, 1}, true);
  // What the table stores matters. Raw log E is wrong near black: the black corner sits at the
  // 1e-10 floor (log E -10), and interpolating toward it pulls dark pixels far below the toe once
  // exposure is added afterwards. Raw linear E is exact in the shadows but too coarse in the
  // highlights. So store log10(E + eps): it is linear in E below eps (smooth into black, floor
  // fades away) and logarithmic above. The renderer undoes it with 10^q - eps. The nodes are also
  // packed toward black (dense darks), because saturated dark colours bend fastest there.
  lutE.bake([&](const Vec3& di) {
    const Vec3 logE = model.negativeLogExposure(di);
    Vec3 q;
    for (size_t c = 0; c < 3; ++c) q[c] = std::log10(std::pow(10.0, logE[c]) + kExposureLutEps);
    return q;
  });
  // Rounded to half precision (a log10 step of at most 2^-10 near the top of the range, 0.23 % in E),
  // which stays within the 0.5 dE2000 budget, so the GPU's spatial path can read a half copy padded
  // to four channels: one 8-byte load per node, for footage that reads the table from all over
  // memory. This does not save GPU memory: the GPU keeps the float table for the colour-only kernel
  // as well as the padded half copy, so table memory rose.
  lutE.roundToHalf();

  std::array<Curve1D, 3> curves;
  constexpr int kCurveSamples = 4096;
  for (int c = 0; c < 3; ++c) {
    // One log E (about 3.3 stops) of headroom each side so exposure shifts reach the toe and shoulder plateaus.
    const auto range = model.negativeLogERange(c);
    Curve1D& cv = curves[size_t(c)];
    cv.min = float(range[0] - 1);
    cv.max = float(range[1] + 1);
    cv.values.resize(kCurveSamples);
    for (int i = 0; i < kCurveSamples; ++i) {
      const double e = double(cv.min) + (double(cv.max) - double(cv.min)) * i / (kCurveSamples - 1);
      // Layers are independent (interlayer is identity in M2), so feed the same e to all three.
      cv.values[size_t(i)] = float(model.negativeDensity({e, e, e})[size_t(c)]);
    }
  }

  // LUT_PE only needs the densities the negative can actually produce (its curves' min..max per
  // layer, plus a sliver of margin). Spending the nodes there instead of on the whole 0..4 cube
  // makes the table several times denser where it matters. Output: exposure on each print layer
  // with the printer lights at zero; the lab's balance and the user's lights are added per pixel.
  Vec3 dLo, dHi;
  for (size_t c = 0; c < 3; ++c) {
    const auto mm = std::minmax_element(curves[c].values.begin(), curves[c].values.end());
    dLo[c] = std::max(0.0, double(*mm.first) - 0.02);
    dHi[c] = double(*mm.second) + 0.02;
  }
  Lut3D lutPE(lutSizePE, dLo, dHi);
  lutPE.bake([&](const Vec3& d) { return model.printLogExposureBase(d); });

  // The print's H&D curves, one per layer, over the digitised range plus one log E of headroom so
  // heavy printer-light offsets still land on the toe and shoulder plateaus (where they clamp).
  std::array<Curve1D, 3> printCurves;
  for (int c = 0; c < 3; ++c) {
    const auto range = model.printLogERange(c);
    Curve1D& cv = printCurves[size_t(c)];
    cv.min = float(range[0] - 1);
    cv.max = float(range[1] + 1);
    cv.values.resize(kCurveSamples);
    for (int i = 0; i < kCurveSamples; ++i) {
      const double e = double(cv.min) + (double(cv.max) - double(cv.min)) * i / (kCurveSamples - 1);
      cv.values[size_t(i)] = float(model.printDensity({e, e, e})[size_t(c)]);
    }
  }

  // LUT_D: print dye density -> log10 of the projected light (XYZ). Light falls off exponentially
  // with dye density, so the log is nearly a straight line in density and a table follows it well.
  // The matrix to Rec.709, the screen's gamma and the black clip are applied per pixel.
  // The domain starts exactly at the print's clear-film density (it cannot go lower), and the nodes
  // bunch up there: clear highlights are where dye amounts hit zero and the response bends sharply.
  Vec3 pLo, pHi;
  for (int c = 0; c < 3; ++c) {
    const auto range = model.printDensityRange(c);
    pLo[size_t(c)] = range[0];
    pHi[size_t(c)] = range[1] + 0.02;
  }
  Lut3D lutD(lutSizeD, pLo, pHi, true);
  lutD.bake([&](const Vec3& d) { return model.printLogXyzFromPrintDensity(d); });
  const Mat3 dm = model.displayMatrix();
  BakedFilm film{std::move(lutE), std::move(curves), std::move(lutPE), std::move(printCurves), std::move(lutD), {}, model.balance(),
                 makeOptics(model.negativeStock(), model.printStock()), makeGrain(model.negativeStock(), model.printStock())};
  for (int r = 0; r < 3; ++r)
    for (int c = 0; c < 3; ++c) film.displayMatrix[size_t(r * 3 + c)] = float(dm[size_t(r)][size_t(c)]);
  return std::make_shared<const BakedFilm>(std::move(film));
}

FilmCache::FilmCache(std::shared_ptr<const FilmResources> resources) : res_(std::move(resources)) {}

std::shared_ptr<const BakedFilm> FilmCache::get(const FilmParams& params) {
  const uint64_t key = params.hash();
  std::promise<std::shared_ptr<const BakedFilm>> promise;
  Future future;
  bool isBaker = false;
  {
    // The lock only guards the bookkeeping; the ~1 s bake below runs without it, so renders of
    // looks that are already cached never wait on someone else's bake.
    std::lock_guard<std::mutex> lock(mutex_);
    for (size_t i = 0; i < entries_.size(); ++i) {
      if (entries_[i].first != key) continue;
      std::rotate(entries_.begin(), entries_.begin() + long(i), entries_.begin() + long(i) + 1);  // now most recent
      future = entries_.front().second;
      break;
    }
    if (!future.valid()) {
      // First request for this look: publish a future so a second request for the same look waits
      // on this bake instead of starting its own.
      future = promise.get_future().share();
      entries_.insert(entries_.begin(), {key, future});
      if (entries_.size() > 4) entries_.pop_back();
      isBaker = true;
    }
  }
  if (!isBaker) return future.get();
  try {
    promise.set_value(bakeFilm(FilmModel(res_, params)));
  } catch (...) {
    // Waiters get the exception; forget the look so the next request tries again.
    promise.set_exception(std::current_exception());
    std::lock_guard<std::mutex> lock(mutex_);
    entries_.erase(std::remove_if(entries_.begin(), entries_.end(), [&](const auto& e) { return e.first == key; }),
                   entries_.end());
  }
  return future.get();
}

std::shared_ptr<const BakedFilm> FilmCache::peek(const FilmParams& params) {
  const uint64_t key = params.hash();
  Future future;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    for (const auto& e : entries_)
      if (e.first == key) { future = e.second; break; }
  }
  if (!future.valid() || future.wait_for(std::chrono::seconds(0)) != std::future_status::ready) return nullptr;
  try {
    return future.get();
  } catch (...) {
    return nullptr;
  }
}

}  // namespace openfilm
