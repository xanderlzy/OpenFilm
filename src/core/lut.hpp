// Baked film: the slow spectral model is sampled once into small tables so the per-pixel work is
// a few table lookups. Five stages: LUT_E (DI code -> exposure, stored as log10(E + eps)), three 1D
// negative curves (log exposure -> negative density; the Exposure control is applied just before),
// LUT_PE (negative density -> exposure on the print, no printer lights), three 1D print curves
// (print log exposure -> print density; the printer lights and Print Density are added just before),
// LUT_D (print density -> log10 projected XYZ).
#pragma once

#include <array>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <vector>

#include "core/film_model.hpp"

namespace openfilm {

// A cube of RGB samples with tetrahedral interpolation (the same scheme Resolve's own LUTs use).
class Lut3D {
 public:
  // denseDarks spaces the nodes as t^2 (t = 0..1 across the domain) so the cells next to the domain
  // minimum are narrow; the exposure table needs that because deep shadows are where the colour
  // model bends fastest, and exposure can push them up the toe of the negative.
  Lut3D(int size, Vec3 domainMin, Vec3 domainMax, bool denseDarks = false);

  // Samples f at every node; slices are spread over the CPU cores, so f must be thread-safe.
  void bake(const std::function<Vec3(const Vec3&)>& f);

  // Inputs outside the domain are clamped to its edge.
  std::array<float, 3> lookup(float a, float b, float c) const;

  int size() const { return size_; }
  // RGB interleaved, red index fastest: ((b * N + g) * N + r) * 3. This is the layout M3 uploads.
  // Node i sits at domain position (i/(N-1))^2 when denseDarks is set, else i/(N-1).
  const std::vector<float>& data() const { return data_; }

 private:
  int size_;
  Vec3 min_, max_;
  bool denseDarks_;
  std::vector<float> data_;
};

// 4096 evenly spaced samples over [min, max]; linear interpolation, clamped at both ends.
struct Curve1D {
  float min = 0, max = 1;
  std::vector<float> values;
  float eval(float x) const;
};

// Offset inside the exposure table's log; see bakeFilm().
constexpr double kExposureLutEps = 1e-3;

struct BakedFilm {
  Lut3D lutE;                         // DI [0,1]^3 -> log10(E + kExposureLutEps), E = 10^logE
  std::array<Curve1D, 3> negCurves;   // log E over [curve min - 1, curve max + 1] -> Status M density
  Lut3D lutPE;                        // neg Status M density (reachable domain) -> print log E, zero offsets
  std::array<Curve1D, 3> printCurves; // print log E over [print curve min - 1, max + 1] -> Status A density
  Lut3D lutD;                         // print density over [curve min, curve max + 0.02], nodes t^2 (dense at low density) -> log10 of projected XYZ
  std::array<float, 9> displayMatrix; // XYZ -> linear display light, row-major
  Vec3 balance;                       // auto-balance offsets (log E)
};

// Default sizes were chosen to keep baked-vs-direct error under 0.5 dE2000 (see the accuracy tests).
// Holds no reference to `model` afterwards.
std::shared_ptr<const BakedFilm> bakeFilm(const FilmModel& model, int lutSizeE = 129, int lutSizePE = 97,
                                          int lutSizeD = 97);

// Keeps the last 4 looks so returning to a recent look does not rebake. A bake runs outside the
// lock and is shared: other threads asking for the same new look wait for it, while threads asking
// for an already-baked look are never blocked. A failed bake is thrown to all its waiters, then forgotten. Exposure, printer
// lights and Print Density are not part of FilmParams (applied per pixel between the tables), so changing them never misses the cache.
class FilmCache {
 public:
  explicit FilmCache(std::shared_ptr<const FilmResources> resources);
  std::shared_ptr<const BakedFilm> get(const FilmParams& params);

 private:
  std::shared_ptr<const FilmResources> res_;
  using Future = std::shared_future<std::shared_ptr<const BakedFilm>>;
  std::mutex mutex_;
  std::vector<std::pair<uint64_t, Future>> entries_;  // most recent first; a look may still be baking
};

}  // namespace openfilm
