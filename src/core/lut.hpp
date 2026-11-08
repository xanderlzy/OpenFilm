// Baked film: the slow spectral model is sampled once into small tables so the per-pixel work is
// a few table lookups. Three stages: LUT_E (DI code -> log exposure), three 1D curves (log
// exposure -> negative density, where exposure in stops is applied), LUT_P (negative density ->
// display code, where the printer lights live).
#pragma once

#include <array>
#include <functional>
#include <memory>
#include <mutex>
#include <vector>

#include "core/film_model.hpp"

namespace openfilm {

// A cube of RGB samples with tetrahedral interpolation (the same scheme Resolve's own LUTs use).
class Lut3D {
 public:
  Lut3D(int size, Vec3 domainMin, Vec3 domainMax);

  // Samples f at every node; slices are spread over the CPU cores, so f must be thread-safe.
  void bake(const std::function<Vec3(const Vec3&)>& f);

  // Inputs outside the domain are clamped to its edge.
  std::array<float, 3> lookup(float a, float b, float c) const;

  int size() const { return size_; }
  // RGB interleaved, red index fastest: ((b * N + g) * N + r) * 3. This is the layout M3 uploads.
  const std::vector<float>& data() const { return data_; }

 private:
  int size_;
  Vec3 min_, max_;
  std::vector<float> data_;
};

// 4096 evenly spaced samples over [min, max]; linear interpolation, clamped at both ends.
struct Curve1D {
  float min = 0, max = 1;
  std::vector<float> values;
  float eval(float x) const;
};

struct BakedFilm {
  Lut3D lutE;                       // DI [0,1]^3 -> log E
  std::array<Curve1D, 3> negCurves;  // log E over [curve min - 1, curve max + 1] -> Status M density
  Lut3D lutP;                       // reachable negative density (per-channel domain) -> display code
};

// Default sizes (113^3 exposure, 97^3 print) were chosen to keep baked-vs-direct error under
// 0.4 dE2000 on 2000 random look-ups; 65^3 reached 0.65. Holds no reference to `model` afterwards.
std::shared_ptr<const BakedFilm> bakeFilm(const FilmModel& model, int lutSizeE = 113, int lutSizeP = 97);

// Keeps the last 4 looks so dragging a slider back and forth does not rebake. Exposure is not
// part of FilmParams (it is applied between the LUTs), so changing it never misses the cache.
class FilmCache {
 public:
  explicit FilmCache(std::shared_ptr<const FilmResources> resources);
  std::shared_ptr<const BakedFilm> get(const FilmParams& params);

 private:
  std::shared_ptr<const FilmResources> res_;
  std::mutex mutex_;
  std::vector<std::pair<uint64_t, std::shared_ptr<const BakedFilm>>> entries_;  // most recent first
};

}  // namespace openfilm
