#include "core/optics.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace openfilm {

namespace {
constexpr double kPi = 3.14159265358979323846;
constexpr double kEdgeSigmaMin = 0.005, kEdgeSigmaMax = 0.5;  // mm; adjacency acts well under 1 mm
// Edge gain bound: adjacency gains in normal processing are modest, and a large gain acting in print
// density would not cancel against wide scatter on a real high-contrast edge (it would fringe).
constexpr double kEdgeKMax = 0.3;

// What is being fitted: `lobes` scatter Gaussians (2 or 3) and, optionally, the edge term.
struct Shape {
  int lobes;
  bool edge;
  int dims() const { return (lobes - 1) + lobes + (edge ? 2 : 0); }
};

// Unconstrained fit parameters, in order: lobes-1 weight logits (softmax, the last logit fixed at 0;
// with 2 lobes that is a logistic), lobes log-sigmas, then with the edge term the logits of k in
// [0, kEdgeKMax] and of sigma_A between kEdgeSigmaMin and kEdgeSigmaMax.
using Params = std::array<double, 8>;

double logistic(double a) { return 1.0 / (1.0 + std::exp(-a)); }
double logit(double p) { return std::log(p / (1.0 - p)); }

LayerSharpness decode(const Params& x, Shape sh) {
  const int L = sh.lobes;
  double logits[3] = {0, 0, 0}, mx = 0;
  for (int i = 0; i < L - 1; ++i) {
    logits[i] = x[size_t(i)];
    mx = std::max(mx, logits[i]);
  }
  double w[3], sum = 0;
  for (int i = 0; i < L; ++i) sum += (w[i] = std::exp(logits[i] - mx));
  LayerSharpness s;
  for (int i = 0; i < L; ++i) s.scatter.push_back({w[i] / sum, std::exp(x[size_t(L - 1 + i)])});
  if (sh.edge) {
    const int e = 2 * L - 1;
    s.edgeK = kEdgeKMax * logistic(x[size_t(e)]);
    s.edgeSigmaMm = kEdgeSigmaMin + (kEdgeSigmaMax - kEdgeSigmaMin) * logistic(x[size_t(e + 1)]);
  }
  return s;
}

double sumSquaredError(const MtfCurve& mtf, const Params& x, Shape sh) {
  const LayerSharpness s = decode(x, sh);
  double e = 0;
  for (const auto& p : mtf) {
    const double d = sharpnessResponse(s, p[0]) - p[1];
    e += d * d;
  }
  return std::isfinite(e) ? e : std::numeric_limits<double>::max();
}

// Plain Nelder-Mead (reflection 1, expansion 2, contraction 0.5, shrink 0.5) over the shape's
// coordinates; ties are broken by index so the result is reproducible.
Params nelderMead(const MtfCurve& mtf, const Params& start, Shape sh) {
  constexpr int kMaxIter = 4000;
  constexpr double kTol = 1e-12, kStep = 0.5;
  const int n = sh.dims();
  std::vector<Params> pts(size_t(n + 1), start);
  std::vector<double> val(size_t(n + 1));
  for (int i = 0; i < n; ++i) pts[size_t(i + 1)][size_t(i)] += kStep;
  for (int i = 0; i <= n; ++i) val[size_t(i)] = sumSquaredError(mtf, pts[size_t(i)], sh);

  std::vector<int> order(size_t(n + 1));
  for (int iter = 0; iter < kMaxIter; ++iter) {
    for (int i = 0; i <= n; ++i) order[size_t(i)] = i;
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return val[size_t(a)] < val[size_t(b)]; });
    const int best = order.front(), worst = order.back(), second = order[size_t(n - 1)];
    if (val[size_t(worst)] - val[size_t(best)] <= kTol) break;

    Params centroid{};
    for (int i = 0; i <= n; ++i) {
      if (i == worst) continue;
      for (int d = 0; d < n; ++d) centroid[size_t(d)] += pts[size_t(i)][size_t(d)] / n;
    }
    const auto along = [&](double t) {
      Params p = start;
      for (int d = 0; d < n; ++d)
        p[size_t(d)] = centroid[size_t(d)] + t * (pts[size_t(worst)][size_t(d)] - centroid[size_t(d)]);
      return p;
    };
    const Params r = along(-1.0);
    const double fr = sumSquaredError(mtf, r, sh);
    if (fr < val[size_t(best)]) {
      const Params e = along(-2.0);
      const double fe = sumSquaredError(mtf, e, sh);
      if (fe < fr) { pts[size_t(worst)] = e; val[size_t(worst)] = fe; }
      else { pts[size_t(worst)] = r; val[size_t(worst)] = fr; }
      continue;
    }
    if (fr < val[size_t(second)]) { pts[size_t(worst)] = r; val[size_t(worst)] = fr; continue; }
    const bool outside = fr < val[size_t(worst)];
    const Params c = along(outside ? -0.5 : 0.5);
    const double fc = sumSquaredError(mtf, c, sh);
    if (fc < (outside ? fr : val[size_t(worst)])) { pts[size_t(worst)] = c; val[size_t(worst)] = fc; continue; }
    for (int i = 0; i <= n; ++i) {  // shrink towards the best point
      if (i == best) continue;
      for (int d = 0; d < n; ++d)
        pts[size_t(i)][size_t(d)] = pts[size_t(best)][size_t(d)] + 0.5 * (pts[size_t(i)][size_t(d)] - pts[size_t(best)][size_t(d)]);
      val[size_t(i)] = sumSquaredError(mtf, pts[size_t(i)], sh);
    }
  }
  int best = 0;
  for (int i = 1; i <= n; ++i)
    if (val[size_t(i)] < val[size_t(best)]) best = i;
  return pts[size_t(best)];
}
}  // namespace

double gaussianMtf(double sigmaMm, double cyclesPerMm) {
  const double a = kPi * sigmaMm * cyclesPerMm;
  return std::exp(-2.0 * a * a);
}

double sharpnessResponse(const LayerSharpness& s, double cyclesPerMm) {
  double scatter = 0;
  for (const auto& l : s.scatter) scatter += l.weight * gaussianMtf(l.sigmaMm, cyclesPerMm);
  return scatter * (1.0 + s.edgeK * (1.0 - gaussianMtf(s.edgeSigmaMm, cyclesPerMm)));
}

LayerSharpness fitSharpness(const MtfCurve& mtf, bool withEdgeEffect, int scatterLobes) {
  if (mtf.empty()) throw std::invalid_argument("fitSharpness: empty MTF curve");
  if (scatterLobes != 2 && scatterLobes != 3) throw std::invalid_argument("fitSharpness: 2 or 3 scatter lobes");
  double f50 = mtf.back()[0];
  for (const auto& p : mtf)
    if (p[1] < 0.5) { f50 = p[0]; break; }
  const double s50 = std::sqrt(std::log(2.0) / (2 * kPi * kPi)) / f50;

  // k = 0 and k = kEdgeKMax are the logistic's asymptotes, so the "no edge effect" start is a negligible
  // gain of 1e-6 and the top start is pulled just inside the bound (0.299).
  // Edge-width starts span 0.02-0.2 mm (bounded to 0.005-0.5): a wide edge Gaussian is what lifts the response above 100 %
  // at the lowest datasheet frequencies (the 2383 hump peaks at or below 2.5 c/mm).
  const Shape sh{scatterLobes, withEdgeEffect};
  const std::vector<double> kStarts = withEdgeEffect ? std::vector<double>{1e-6, 0.1, 0.3} : std::vector<double>{1e-6};
  const std::vector<double> edgeStarts = withEdgeEffect ? std::vector<double>{0.02, 0.05, 0.1, 0.2} : std::vector<double>{0.02};
  Params bestX{};
  double bestErr = std::numeric_limits<double>::infinity();
  for (double m : {0.5, 1.0, 2.0}) {
    for (double k : kStarts) {
      for (double sA : edgeStarts) {
        const double s1 = m * s50;
        Params start{};
        size_t i = 0;
        if (scatterLobes == 2) {
          start[i++] = logit(0.8);  // weights 0.8 / 0.2
        } else {
          start[i++] = std::log(0.6 / 0.1);  // weights 0.6 / 0.3 / 0.1
          start[i++] = std::log(0.3 / 0.1);
        }
        for (int l = 0; l < scatterLobes; ++l) start[i++] = std::log(s1 * std::pow(4.0, l));  // sigma_{n+1} = 4 sigma_n
        start[i++] = logit(std::min(k, kEdgeKMax - 1e-3) / kEdgeKMax);
        start[i++] = logit((sA - kEdgeSigmaMin) / (kEdgeSigmaMax - kEdgeSigmaMin));
        const Params x = nelderMead(mtf, start, sh);
        const double err = sumSquaredError(mtf, x, sh);
        if (err < bestErr) { bestErr = err; bestX = x; }
      }
    }
  }
  LayerSharpness s = decode(bestX, sh);
  std::stable_sort(s.scatter.begin(), s.scatter.end(), [](const Lobe& a, const Lobe& b) { return a.sigmaMm < b.sigmaMm; });
  return s;
}

FilmOptics makeOptics(const Stock& negative, const Stock& print) {
  if (!negative.halation) throw std::runtime_error(negative.id + ": no halation data");
  FilmOptics o;
  for (size_t c = 0; c < 3; ++c) {
    o.negative[c] = fitSharpness(negative.mtf[c], true);
    o.print[c] = fitSharpness(print.mtf[c], true, 3);
  }
  o.halation = *negative.halation;
  return o;
}

std::vector<Lobe> halationLobes(double baseThicknessMm, double radiusMultiplier) {
  const double t = baseThicknessMm * radiusMultiplier;
  return {{0.5, 1.0 * t}, {0.5, 2.0 * t}};
}

std::vector<Lobe> bloomLobes(double radiusMm) { return {{0.7, 0.5 * radiusMm}, {0.3, 1.5 * radiusMm}}; }

double gateWidthMm(FilmFormat format, double customMm) {
  switch (format) {
    case FilmFormat::Super35: return 24.89;
    case FilmFormat::Academy35: return 21.95;
    case FilmFormat::Super16: return 12.52;
    case FilmFormat::Std16: return 10.26;
    case FilmFormat::Mm65: return 52.48;
    case FilmFormat::Custom: break;
  }
  if (!std::isfinite(customMm)) return 24.89;
  return std::clamp(customMm, 5.0, 80.0);
}

double pixelsPerMm(double imageWidthPx, double gateMm) { return imageWidthPx / gateMm; }

}  // namespace openfilm
