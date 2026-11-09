// The film's spatial behaviour on the CPU: bloom, per-layer scatter and halation on the exposure,
// the development edge effect on the negative density, scatter on the print (no print edge effect:
// ruling R14). The
// M3 colour chain is split into stages so each effect acts in the domain where it happens. This is
// the reference the Metal path mirrors pass for pass. Grain (M5) adds dye-cloud noise to the
// negative and the print density.
#pragma once

#include <cstdint>

#include "core/blur.hpp"
#include "core/grain_sim.hpp"
#include "core/image.hpp"
#include "core/lut.hpp"

namespace openfilm {

enum class GrainQuality { Preview, Full };

struct SpatialControls {
  bool sharpness = true;  // scatter on negative and print, edge effect on the negative
  bool halation = true;
  bool remjet = true;
  double halationStrength = 1;  // 0..5, 1 = physical
  double halationRadius = 1;    // 0.25..4, multiplier on the physical radius
  bool bloom = false;
  double bloomAmount = 0.25;   // 0..1
  double bloomRadiusMm = 0.5;  // 0.05..5 mm on the film
  double pxPerMm = 3840 / 24.89;  // image width (px, at the current render scale) / gate width (mm)
  bool grain = false;  // core default off (M4 path); the OFX default is on
  double grainAmount = 1;  // 0..3, multiplies the datasheet rms
  double grainSize = 1;    // 0.5..2, multiplies the cloud radius
  GrainQuality grainQuality = GrainQuality::Preview;
  uint32_t frameSeed = 0;  // frameSeedFromTime(time): new clouds every frame, the same ones on a re-render
  bool grainOn() const { return grain && grainAmount > 0; }
  bool anyOn() const { return sharpness || halation || bloom || grainOn(); }
};

// Seed of a frame's grain: a hash of the bits of the host's time.
uint32_t frameSeedFromTime(double time);

// Non-finite values -> default; values clamped to the control ranges above; pxPerMm <= 0 or
// non-finite -> default, and capped at 1e4 px/mm so kernel sizes stay representable.
SpatialControls sanitize(SpatialControls s);

// How far (full-res px) the active stages read beyond a pixel: the sum over stages of each stage's
// widest reach (blurRadiusPx, plus the block size of reduced-resolution lobes; see spreadReachPx).
// A render window grown by this needs nothing outside it, so a tile equals the full-frame render.
int spatialMarginPx(const FilmOptics& optics, const SpatialControls& s);

// The window grown by the margin and cut to the source bounds. Working pixel (0,0) is image pixel
// (x1, y1). The near edges are rounded down to a multiple of 32 (the largest reduced-resolution
// block) in frame coordinates, so the block grid of wide lobes is the same for every window.
struct WorkRect {
  int x1, y1, x2, y2;
  int width() const { return x2 > x1 ? x2 - x1 : 0; }
  int height() const { return y2 > y1 ? y2 - y1 : 0; }
};
WorkRect workRect(const ConstImageView& src, const RenderWindow& win, int marginPx);

// Calls renderFilmCpu when !sanitize(spatial).anyOn() (bit-identical to M3); otherwise runs the stages below.
// Same window, alpha and NaN rules as renderFilmCpu.
void renderFilm(const ConstImageView& src, const ImageView& dst, const RenderWindow& win, const BakedFilm& film,
                const LookControls& look, const SpatialControls& spatial);

// ---- Lobes in pixels, exactly as the stages use them (the GPU host builds its passes from these).
// `s` must already be sanitised. ----
// Per-layer scatter as passes: pass i blurs every channel with that layer's i-th lobe (weight 1);
// layer c then takes weights[i][c] of pass i. Layers' lobes are sorted by sigma, so pass i pairs
// like-sized lobes.
struct LayerPasses {
  std::vector<PxLobe> passes;
  std::vector<std::array<float, 3>> weights;
};
LayerPasses layerPasses(const std::array<LayerSharpness, 3>& layers, double pxPerMm);
std::vector<PxLobe> edgeLobe(const std::array<LayerSharpness, 3>& layers, double pxPerMm);  // one lobe, weight 1
std::vector<PxLobe> haloLobes(const FilmOptics& optics, const SpatialControls& s);
std::vector<PxLobe> bloomPxLobes(const SpatialControls& s);
// h * layerSplit_c, h = rem-jet reflectance x strength: what each layer gains from the halo of red.
std::array<float, 3> halationWeights(const FilmOptics& optics, const SpatialControls& s);

// ---- Stages (public for the tests and as the GPU's reference) ----
// DI -> linear exposure: RGB = (10^lutE - eps) * gain, A = source alpha. Non-finite DI counts as 0.
void exposeStage(const ConstImageView& src, const WorkRect& rect, const BakedFilm& film, float gain, Image4f& E);
// Bloom E <- (1-a)E + a*spread(E, bloomLobes); scatter E_c <- spread_c(E, negative[c].scatter);
// halation E_c <- E_c + h w_c (H * E_r). Each only when its switch is on, in that order.
void exposureEffects(Image4f& E, const FilmOptics& optics, const SpatialControls& s);
// E -> negative density: negCurves(log10(max(E, 1e-30))).
void developStage(Image4f& EtoD, const BakedFilm& film);
// Negative edge effect (if sharpness): D_c <- D_c + k_c (D_c - G_{sigmaA,c} * D_c).
void edgeEffect(Image4f& D, const FilmOptics& optics, const SpatialControls& s);
// Grain on the negative (st = Negative, after the edge effect) or on the print density (st = Print,
// after print development): D_c += Amount * a_c(D_c) * u_c, a from grainTables(film.grain, Size), u the
// layer's grainUnit at frame pixel (rect.x1 + i, rect.y1 + j) for work pixel (i, j), seeded by
// grainSeed(frameSeed, st, c). Non-finite D_c unchanged. No-op when !s.grainOn() or !film.grain.valid.
// Reads no neighbours (adds nothing to spatialMarginPx). Full quality for either GrainQuality (ruling R2).
void grainStage(Image4f& D, const WorkRect& rect, const BakedFilm& film, const SpatialControls& s, GrainStage st);
// Negative density -> linear print exposure P = 10^lutPE(D).
void printExposeStage(Image4f& DtoP, const BakedFilm& film);
// Print scatter (if sharpness): P_c <- spread_c(P, print[c].scatter).
void printSharpness(Image4f& P, const FilmOptics& optics, const SpatialControls& s);
// P -> print density: printCurves(log10(max(P, 1e-30)) + offsets).
void printDevelopStage(Image4f& PtoDp, const BakedFilm& film, const Vec3& offsets);
// Print density -> lutD -> 10^q -> display matrix -> encode; writes only window pixels. Window pixels
// outside the work rect (i.e. outside the source) become (0,0,0,0).
void finishStage(const Image4f& Dp, const WorkRect& rect, const BakedFilm& film, const ImageView& dst,
                 const RenderWindow& win);

}  // namespace openfilm
