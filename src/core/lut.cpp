#include "core/lut.hpp"

#include <algorithm>
#include <cmath>
#include <thread>

namespace openfilm {

Lut3D::Lut3D(int size, Vec3 domainMin, Vec3 domainMax)
    : size_(size), min_(domainMin), max_(domainMax), data_(size_t(size) * size * size * 3, 0.f) {}

void Lut3D::bake(const std::function<Vec3(const Vec3&)>& f) {
  const int n = size_;
  auto slices = [&](int b0, int b1) {
    for (int b = b0; b < b1; ++b)
      for (int g = 0; g < n; ++g)
        for (int r = 0; r < n; ++r) {
          const int idx[3] = {r, g, b};
          Vec3 in;
          for (int k = 0; k < 3; ++k) in[k] = min_[k] + (max_[k] - min_[k]) * double(idx[k]) / double(n - 1);
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
    float u = (in[k] - float(min_[k])) / float(max_[k] - min_[k]) * float(n - 1);
    u = std::clamp(u, 0.f, float(n - 1));  // also maps NaN to a valid cell (comparisons are false)
    if (!(u == u)) u = 0.f;
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

std::shared_ptr<const BakedFilm> bakeFilm(const FilmModel& model, int lutSizeE, int lutSizeP) {
  Lut3D lutE(lutSizeE, {0, 0, 0}, {1, 1, 1});
  lutE.bake([&](const Vec3& di) { return model.negativeLogExposure(di); });

  std::array<Curve1D, 3> curves;
  constexpr int kCurveSamples = 4096;
  for (int c = 0; c < 3; ++c) {
    // One stop of headroom each side so exposure shifts reach the toe and shoulder plateaus.
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

  // The print LUT only needs the densities the negative can actually produce (its curves'
  // min..max per layer, plus a sliver of margin). Spending the nodes there instead of on the
  // whole 0..4 cube makes the print LUT several times denser where it matters.
  Vec3 dLo, dHi;
  for (size_t c = 0; c < 3; ++c) {
    const auto mm = std::minmax_element(curves[c].values.begin(), curves[c].values.end());
    dLo[c] = std::max(0.0, double(*mm.first) - 0.02);
    dHi[c] = double(*mm.second) + 0.02;
  }
  Lut3D lutP(lutSizeP, dLo, dHi);
  lutP.bake([&](const Vec3& d) { return model.printOutput(d); });
  return std::make_shared<const BakedFilm>(BakedFilm{std::move(lutE), std::move(curves), std::move(lutP)});
}

FilmCache::FilmCache(std::shared_ptr<const FilmResources> resources) : res_(std::move(resources)) {}

std::shared_ptr<const BakedFilm> FilmCache::get(const FilmParams& params) {
  const uint64_t key = params.hash();
  std::lock_guard<std::mutex> lock(mutex_);
  for (size_t i = 0; i < entries_.size(); ++i) {
    if (entries_[i].first != key) continue;
    std::rotate(entries_.begin(), entries_.begin() + long(i), entries_.begin() + long(i) + 1);  // now most recent
    return entries_.front().second;
  }
  auto baked = bakeFilm(FilmModel(res_, params));
  entries_.insert(entries_.begin(), {key, baked});
  if (entries_.size() > 4) entries_.pop_back();
  return baked;
}

}  // namespace openfilm
