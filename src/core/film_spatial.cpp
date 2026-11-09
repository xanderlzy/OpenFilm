#include "core/film_spatial.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

#include "core/film_renderer.hpp"
#include "core/pixel_ops.hpp"
#include "metal/grain_common.h"

namespace openfilm {

namespace {

// Lobe width in px; anything not a positive finite number is "no spread" (0, copied by the blur).
float pxSigma(double sigmaMm, double pxPerMm) {
  const double s = sigmaMm * pxPerMm;
  return std::isfinite(s) && s > 0 ? float(std::min(s, 1e6)) : 0.f;
}

std::vector<PxLobe> toPx(const std::vector<Lobe>& lobes, double pxPerMm) {
  std::vector<PxLobe> out;
  for (const Lobe& l : lobes) {
    const float s = pxSigma(l.sigmaMm, pxPerMm);
    out.push_back({float(l.weight), {s, s, s}});
  }
  return out;
}

}  // namespace

LayerPasses layerPasses(const std::array<LayerSharpness, 3>& layers, double pxPerMm) {
  size_t n = 0;
  for (const auto& l : layers) n = std::max(n, l.scatter.size());
  LayerPasses lp;
  for (size_t i = 0; i < n; ++i) {
    PxLobe pass{1.f, {0.f, 0.f, 0.f}};
    std::array<float, 3> w{0.f, 0.f, 0.f};
    for (size_t c = 0; c < 3; ++c) {
      if (i >= layers[c].scatter.size()) continue;  // a layer with fewer lobes takes nothing from this pass
      pass.sigmaPx[c] = pxSigma(layers[c].scatter[i].sigmaMm, pxPerMm);
      w[c] = float(layers[c].scatter[i].weight);
    }
    lp.passes.push_back(pass);
    lp.weights.push_back(w);
  }
  return lp;
}

std::vector<PxLobe> edgeLobe(const std::array<LayerSharpness, 3>& layers, double pxPerMm) {
  return {{1.f,
           {pxSigma(layers[0].edgeSigmaMm, pxPerMm), pxSigma(layers[1].edgeSigmaMm, pxPerMm),
            pxSigma(layers[2].edgeSigmaMm, pxPerMm)}}};
}

std::vector<PxLobe> haloLobes(const FilmOptics& optics, const SpatialControls& s) {
  return toPx(halationLobes(optics.halation.baseThicknessMm, s.halationRadius), s.pxPerMm);
}

std::vector<PxLobe> bloomPxLobes(const SpatialControls& s) { return toPx(bloomLobes(s.bloomRadiusMm), s.pxPerMm); }

std::array<float, 3> halationWeights(const FilmOptics& optics, const SpatialControls& s) {
  const Halation& hal = optics.halation;
  const double h = (s.remjet ? hal.reflectanceOn : hal.reflectanceOff) * s.halationStrength;
  return {float(h * hal.layerSplit[0]), float(h * hal.layerSplit[1]), float(h * hal.layerSplit[2])};
}

namespace {

using detail::encodeDisplay;
using detail::parallelRows;
using detail::sane;

constexpr int kBlockAlign = 32;  // largest downsampleFactor: wide lobes' block grid starts on it

double fixed(double v, double def, double lo, double hi) { return std::isfinite(v) ? std::clamp(v, lo, hi) : def; }

int layerReach(const LayerPasses& lp) {
  int r = 0;
  for (const PxLobe& p : lp.passes) r = std::max(r, spreadReachPx({p}));
  return r;
}

// img_c <- sum_i weights[i][c] * (pass i * img)_c; alpha kept.
void spreadLayers(Image4f& img, const std::array<LayerSharpness, 3>& layers, double pxPerMm) {
  const LayerPasses lp = layerPasses(layers, pxPerMm);
  Image4f out = img;
  parallelRows(out.height, [&](int y0, int y1) {
    for (int y = y0; y < y1; ++y)
      for (int x = 0; x < out.width; ++x) out.at(x, y)[0] = out.at(x, y)[1] = out.at(x, y)[2] = 0.f;
  });
  for (size_t i = 0; i < lp.passes.size(); ++i) {
    Image4f blurred;
    spread(img, blurred, {lp.passes[i]});
    const auto& w = lp.weights[i];
    parallelRows(out.height, [&](int y0, int y1) {
      for (int y = y0; y < y1; ++y)
        for (int x = 0; x < out.width; ++x)
          for (int c = 0; c < 3; ++c) out.at(x, y)[c] += w[size_t(c)] * blurred.at(x, y)[c];
    });
  }
  img = std::move(out);
}

// D_c <- D_c + k_c (D_c - G_{sigmaA,c} * D_c).
void unsharpDensity(Image4f& d, const std::array<LayerSharpness, 3>& layers, double pxPerMm) {
  Image4f blurred;
  spread(d, blurred, edgeLobe(layers, pxPerMm));
  const float k[3] = {float(layers[0].edgeK), float(layers[1].edgeK), float(layers[2].edgeK)};
  parallelRows(d.height, [&](int y0, int y1) {
    for (int y = y0; y < y1; ++y)
      for (int x = 0; x < d.width; ++x) {
        float* p = d.at(x, y);
        const float* b = blurred.at(x, y);
        for (int c = 0; c < 3; ++c) p[c] += k[c] * (p[c] - b[c]);
      }
  });
}

// Applies fn(pixel) to every pixel of img, rows in parallel.
template <class Fn>
void forEachPixel(Image4f& img, Fn fn) {
  parallelRows(img.height, [&](int y0, int y1) {
    for (int y = y0; y < y1; ++y)
      for (int x = 0; x < img.width; ++x) fn(img.at(x, y));
  });
}

}  // namespace

SpatialControls sanitize(SpatialControls s) {
  const SpatialControls d;
  s.halationStrength = fixed(s.halationStrength, d.halationStrength, 0, 5);
  s.halationRadius = fixed(s.halationRadius, d.halationRadius, 0.25, 4);
  s.bloomAmount = fixed(s.bloomAmount, d.bloomAmount, 0, 1);
  s.bloomRadiusMm = fixed(s.bloomRadiusMm, d.bloomRadiusMm, 0.05, 5);
  s.pxPerMm = std::isfinite(s.pxPerMm) && s.pxPerMm > 0 ? std::min(s.pxPerMm, 1e4) : d.pxPerMm;
  s.grainAmount = fixed(s.grainAmount, d.grainAmount, 0, 3);
  s.grainSize = fixed(s.grainSize, d.grainSize, 0.5, 2);
  return s;
}

uint32_t frameSeedFromTime(double time) {
  uint64_t bits = 0;
  std::memcpy(&bits, &time, sizeof bits);
  return of_hash2(uint32_t(bits), uint32_t(bits >> 32));
}

int spatialMarginPx(const FilmOptics& optics, const SpatialControls& controls) {
  const SpatialControls s = sanitize(controls);
  long long m = 0;
  if (s.bloom) m += spreadReachPx(bloomPxLobes(s));
  if (s.sharpness) m += layerReach(layerPasses(optics.negative, s.pxPerMm));
  if (s.halation) m += spreadReachPx(haloLobes(optics, s));
  if (s.sharpness) {
    m += spreadReachPx(edgeLobe(optics.negative, s.pxPerMm));
    m += layerReach(layerPasses(optics.print, s.pxPerMm));
  }
  return int(std::min(m, 1LL << 28));
}

WorkRect workRect(const ConstImageView& src, const RenderWindow& win, int marginPx) {
  if (!src.data) return {0, 0, 0, 0};
  const long long m = std::max(marginPx, 0);
  auto alignDown = [](long long v) { return (v >= 0 ? v / kBlockAlign : -((-v + kBlockAlign - 1) / kBlockAlign)) * kBlockAlign; };
  const long long x1 = std::max<long long>(alignDown(win.x1 - m), src.geom.x1);
  const long long y1 = std::max<long long>(alignDown(win.y1 - m), src.geom.y1);
  const long long x2 = std::min<long long>(win.x2 + m, src.geom.x2);
  const long long y2 = std::min<long long>(win.y2 + m, src.geom.y2);
  if (x2 <= x1 || y2 <= y1) return {0, 0, 0, 0};
  return {int(x1), int(y1), int(x2), int(y2)};
}

void exposeStage(const ConstImageView& src, const WorkRect& rect, const BakedFilm& film, float gain, Image4f& E) {
  E.width = rect.width();
  E.height = rect.height();
  E.px.assign(size_t(E.width) * size_t(E.height) * 4, 0.f);
  parallelRows(E.height, [&](int y0, int y1) {
    for (int y = y0; y < y1; ++y)
      for (int x = 0; x < E.width; ++x) {
        const float* in = pixelAddress(src, rect.x1 + x, rect.y1 + y);
        float* out = E.at(x, y);
        if (!in) continue;  // cannot happen for a rect from workRect(src, ...)
        const auto e = film.lutE.lookup(sane(in[0]), sane(in[1]), sane(in[2]));
        for (int c = 0; c < 3; ++c) out[c] = (std::pow(10.f, e[size_t(c)]) - float(kExposureLutEps)) * gain;
        out[3] = in[3];
      }
  });
}

void exposureEffects(Image4f& E, const FilmOptics& optics, const SpatialControls& controls) {
  const SpatialControls s = sanitize(controls);
  if (s.bloom) {
    Image4f blurred;
    spread(E, blurred, bloomPxLobes(s));
    const float a = float(s.bloomAmount);
    parallelRows(E.height, [&](int y0, int y1) {
      for (int y = y0; y < y1; ++y)
        for (int x = 0; x < E.width; ++x)
          for (int c = 0; c < 3; ++c) E.at(x, y)[c] = (1.f - a) * E.at(x, y)[c] + a * blurred.at(x, y)[c];
    });
  }
  if (s.sharpness) spreadLayers(E, optics.negative, s.pxPerMm);
  if (s.halation) {
    const std::array<float, 3> hw = halationWeights(optics, s);
    Image4f halo;
    spread(E, halo, haloLobes(optics, s));  // only its red channel is used: the red layer's light reaches the base
    parallelRows(E.height, [&](int y0, int y1) {
      for (int y = y0; y < y1; ++y)
        for (int x = 0; x < E.width; ++x) {
          const float r = halo.at(x, y)[0];
          for (int c = 0; c < 3; ++c) E.at(x, y)[c] += hw[size_t(c)] * r;
        }
    });
  }
}

void developStage(Image4f& EtoD, const BakedFilm& film) {
  forEachPixel(EtoD, [&](float* p) {
    for (int c = 0; c < 3; ++c) p[c] = film.negCurves[size_t(c)].eval(std::log10(std::max(p[c], 1e-30f)));
  });
}

void edgeEffect(Image4f& D, const FilmOptics& optics, const SpatialControls& controls) {
  const SpatialControls s = sanitize(controls);
  if (s.sharpness) unsharpDensity(D, optics.negative, s.pxPerMm);
}

void grainStage(Image4f& D, const WorkRect& rect, const BakedFilm& film, const SpatialControls& controls, GrainStage st) {
  const SpatialControls s = sanitize(controls);
  if (!s.grainOn() || !film.grain.valid) return;
  const auto& layers = st == GrainStage::Negative ? film.grain.negative : film.grain.print;
  const GrainTables tables = grainTables(film.grain, s.grainSize);
  const auto& amp = tables.stage(st);
  std::array<GrainUnitParams, 3> unit;
  std::array<uint32_t, 3> seed{};
  for (size_t c = 0; c < 3; ++c) {
    unit[c] = grainUnitParams(layers[c], s.pxPerMm, s.grainSize);
    seed[c] = grainSeed(s.frameSeed, st, int(c));
  }
  const float amount = float(s.grainAmount);
  parallelRows(D.height, [&](int y0, int y1) {
    for (int y = y0; y < y1; ++y)
      for (int x = 0; x < D.width; ++x) {
        float* p = D.at(x, y);
        for (size_t c = 0; c < 3; ++c) {
          if (!std::isfinite(p[c])) continue;
          p[c] += amount * amp[c].at(p[c]) * grainUnit(unit[c], rect.x1 + x, rect.y1 + y, seed[c]);
        }
      }
  });
}

void printExposeStage(Image4f& DtoP, const BakedFilm& film) {
  forEachPixel(DtoP, [&](float* p) {
    const auto pe = film.lutPE.lookup(p[0], p[1], p[2]);
    for (int c = 0; c < 3; ++c) p[c] = std::pow(10.f, pe[size_t(c)]);
  });
}

void printSharpness(Image4f& P, const FilmOptics& optics, const SpatialControls& controls) {
  const SpatialControls s = sanitize(controls);
  if (s.sharpness) spreadLayers(P, optics.print, s.pxPerMm);
}

void printDevelopStage(Image4f& PtoDp, const BakedFilm& film, const Vec3& offsets) {
  const float off[3] = {float(offsets[0]), float(offsets[1]), float(offsets[2])};
  forEachPixel(PtoDp, [&](float* p) {
    for (int c = 0; c < 3; ++c)
      p[c] = film.printCurves[size_t(c)].eval(std::log10(std::max(p[c], 1e-30f)) + off[c]);
  });
}

void finishStage(const Image4f& Dp, const WorkRect& rect, const BakedFilm& film, const ImageView& dst,
                 const RenderWindow& win) {
  parallelRows(win.y2 - win.y1, [&](int r0, int r1) {
    for (int y = win.y1 + r0; y < win.y1 + r1; ++y)
      for (int x = win.x1; x < win.x2; ++x) {
        float* out = pixelAddress(dst, x, y);
        if (!out) continue;  // window pixel the host gave us no room for
        const int wx = x - rect.x1, wy = y - rect.y1;
        if (wx < 0 || wy < 0 || wx >= rect.width() || wy >= rect.height() || wx >= Dp.width || wy >= Dp.height) {
          out[0] = out[1] = out[2] = out[3] = 0.f;  // outside the source
          continue;
        }
        const float* pd = Dp.at(wx, wy);
        const auto q = film.lutD.lookup(pd[0], pd[1], pd[2]);
        const float xyz[3] = {std::pow(10.f, q[0]), std::pow(10.f, q[1]), std::pow(10.f, q[2])};
        const float* m = film.displayMatrix.data();
        for (int k = 0; k < 3; ++k) out[k] = encodeDisplay(m[k * 3] * xyz[0] + m[k * 3 + 1] * xyz[1] + m[k * 3 + 2] * xyz[2]);
        out[3] = pd[3];
      }
  });
}

void renderFilm(const ConstImageView& src, const ImageView& dst, const RenderWindow& win, const BakedFilm& film,
                const LookControls& look, const SpatialControls& spatial) {
  if (win.x2 <= win.x1 || win.y2 <= win.y1) return;
  const SpatialControls s = sanitize(spatial);
  if (!s.anyOn()) {
    renderFilmCpu(src, dst, win, film, look);
    return;
  }
  const WorkRect rect = workRect(src, win, spatialMarginPx(film.optics, s));
  Image4f img;
  exposeStage(src, rect, film, exposureGain(look.exposureStops), img);
  exposureEffects(img, film.optics, s);
  developStage(img, film);
  edgeEffect(img, film.optics, s);
  grainStage(img, rect, film, s, GrainStage::Negative);
  printExposeStage(img, film);
  printSharpness(img, film.optics, s);
  printDevelopStage(img, film, printOffsetsLogE(film.balance, look));
  grainStage(img, rect, film, s, GrainStage::Print);
  finishStage(img, rect, film, dst, win);
}

}  // namespace openfilm
