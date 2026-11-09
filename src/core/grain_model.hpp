// Film grain as additive dye clouds (shot noise; ruling R3): each layer's dye image is a Poisson
// scatter of discs (clouds) of radius r whose densities add where they overlap. The unit field
// u = N - mu (N the number of clouds covering a point, mu its mean) has zero mean, so grain keeps the
// mean density exactly. Its covariance is closed form, so the rms through any aperture is a
// one-dimensional integral. The grain added to density is a(D) u, with the amplitude a(D) chosen so
// the rms through the datasheet's 48 um aperture equals the datasheet's rms granularity at D.
// Lengths are in mm on the film.
#pragma once

#include <array>
#include <vector>

#include "core/stock.hpp"

namespace openfilm {

// Mean number of clouds covering a point. An assumption: the datasheet's single aperture cannot pin it.
constexpr double kGrainMeanOverlap = 4;

struct GrainLayer {
  double rMm = 0;                     // cloud radius at Size 1 (the stock's published/assumed radius)
  double mu = kGrainMeanOverlap;      // mean overlap count; lambda = mu / (pi r^2)
  std::vector<double> density, sigma; // datasheet rms granularity at absolute density, density ascending

  // Datasheet rms at absolute density D: linear in (density, log sigma) between points, held at the
  // end values outside them. 0 for a non-finite D or a layer without points.
  double sigmaAt(double D) const;
  // `sizeMul` multiplies the cloud radius in every function below.
  double lambdaPerMm2(double sizeMul = 1) const;
  // Covariance of u at distance h (mm): lambda * A(h), A the overlap area of two discs of radius r.
  double unitCovariance(double hMm, double sizeMul = 1) const;
  double unitApertureRms(double apertureDiameterMm, double sizeMul = 1) const;  // circular aperture
  double unitPixelRms(double pixelMm, double sizeMul = 1) const;                // square pixel
  // sigmaAt(D) / unitApertureRms(0.048 mm, sizeMul): the 48 um rms of a(D) u is the datasheet's at every Size.
  double amplitude(double D, double sizeMul = 1) const;
};

struct FilmGrain {
  std::array<GrainLayer, 3> negative, print;  // layer order as Stock::curve
  bool valid = false;
};

// Memoised by stock id (mutex-guarded); valid == false when either stock has no granularity.
FilmGrain makeGrain(const Stock& negative, const Stock& print);

}  // namespace openfilm
