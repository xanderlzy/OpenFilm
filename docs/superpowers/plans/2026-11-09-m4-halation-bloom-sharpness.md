# M4 — Halation, bloom and sharpness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the film's spatial behaviour (per-layer sharpness with edge effects, halation, optional
bloom, print sharpness), scaled by film format, on CPU and Metal, with UHD real time kept.

**Architecture:** The M3 fused kernel stays and runs whenever every M4 effect is off. With any effect
on, the frame runs as a chain of passes over float RGBA working images: expose → (bloom, scatter,
halation on linear exposure) → develop → (edge effect on density) → print-expose → (print sharpness on
linear print exposure) → finish. Every spread is a sum of Gaussian lobes in millimetres, fitted at
stock-load time from digitised datasheet MTF curves. The CPU chain (`src/core`) is the reference; the
Metal chain mirrors it pass for pass.

**Tech Stack:** C++20, CMake, doctest, Metal (Objective-C++ host), OpenFX Support library, Python
(PyMuPDF) for digitising.

**Spec:** [docs/superpowers/specs/2026-11-09-m4-halation-bloom-sharpness-design.md](../specs/2026-11-09-m4-halation-bloom-sharpness-design.md)

## Global Constraints

- Branch `m4-halation`. Never merge or push to `master` without asking the user.
- Commits use the user's GitHub noreply email (already configured) and end with
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Datasheet PDFs are never committed. Run digitising scripts with `python -I`. PDFs and the venv:
  `/private/tmp/claude-501/-Users-xanderlzy-Documents-Projects-Plugin-DVR-Film-Emulator/24b3310b-7fc8-41af-90f3-3cd9b2c8b9fb/scratchpad/{datasheets,venv}`.
- Off = M3: with Sharpness, Halation and Bloom all off, output equals M3's exactly (CPU bit-exact; GPU
  runs M3's kernel unchanged).
- Every spreading kernel is normalised to sum 1; the edge effect keeps mean density.
- Kernels are in mm on the film; `px_per_mm = image width in pixels (render scale applied) / gate width (mm)`.
- `src/metal/*.metal` mirrors `src/core` exactly: any change to a CPU pass is made to its GPU twin.
  Kernels are compiled with `-fno-fast-math`; use `precise::` maths as `film.metal` does.
- Any new OFX host-suite call is wrapped (Resolve may not implement it), like `showError`/`clearError`.
- Non-finite control values are treated as the control's default; ranges are clamped to the spec table.
- No Quality control (that arrives with M5). None of the M4 controls re-bake the colour tables.
- Build: `cmake --build build -j` then `ctest --test-dir build --output-on-failure`.

## Review Focus

1. **Extreme highlights** (DI code 1.0 at +10 stops, Halation Strength 5, Radius 4, Bloom Amount 1):
   output must stay finite and in [0, 1] after the display clip — no NaN/inf from huge exposures being
   spread. Test in Task 4 (CPU) and Task 5 (GPU).
2. **Kernels larger than the image** (Bloom Radius 5 mm with a 16mm gate on a 64×36 image; a 1×1
   image): no crash or out-of-bounds read, finite output, light conserved within 1e-3. Test in Task 3
   (blur library) and Task 4.
3. **Sub-windows and window edges**: a render window inside the image, with the full source available,
   must equal the same pixels of a full-frame render (≤ 1e-6 CPU) — no seam at the window edge. When the
   host gives less source than requested, output is still finite and edge-clamped. Test in Task 4 and Task 5.
4. **Concurrent GPU renders of different sizes** share the intermediate-image pool without one
   overwriting another. Test in Task 5.
5. **Non-finite or out-of-range spatial controls** (NaN Strength, gate 0 or 1000 mm, Radius −1): treated
   as default / clamped, never a crash or black frame. Test in Task 4 (`sanitize`) and Task 7 (reader).

## Decisions this plan adds to the spec

- The print is fitted with scatter only (no print edge effect): the 2383 curves show no hump above
  100 %, and the spec's pipeline has no print-density pass.
- Halation kernel `H` = two lobes at 1.0× and 2.0× base thickness (× Radius); bloom kernel = lobes at
  0.5× and 1.5× Radius. Wide lobes (σ > 4 px) run at reduced resolution.
- `layerSplit` is derived from the stock's own spectral sensitivities over 600–700 nm, not typed in.
- The "measured sharpness" test measures the negative (up to developed density), with print sharpness
  excluded, since the spec compares it with the negative's datasheet curve.
- The spec asks for an early Resolve check of regions of interest. The plan makes the code correct
  either way (it edge-clamps whatever source the host gives) and checks Resolve's behaviour at the
  final Resolve check, avoiding a mid-run install.

---

### Task 1: MTF and halation stock data

**Files:**
- Create: `tools/digitize/figures/7219_mtf.json`, `tools/digitize/figures/2383_mtf.json`,
  `tools/digitize/points/7219_mtf_{red,green,blue}.csv`, `tools/digitize/points/2383_mtf_{red,green,blue}.csv`
- Modify: `tools/digitize/build_stock.py`, `tools/digitize/meta/kodak-5219-type.json`,
  `tools/digitize/meta/kodak-2383-type.json`, `tools/digitize/README.md`, `data/stocks/schema.json`,
  `data/stocks/kodak-5219-type.json`, `data/stocks/kodak-2383-type.json` (regenerated),
  `src/core/stock.hpp`, `src/core/stock.cpp`
- Test: `tests/core/test_stock.cpp`

**Interfaces:**
- Produces (in `stock.hpp`):
  ```cpp
  using MtfCurve = std::vector<std::array<double, 2>>;  // {cycles/mm, response (1.0 = 100 %)}, f ascending
  struct Halation {
    double baseThicknessMm;
    double reflectanceOn, reflectanceOff;  // fraction of red-layer exposure returned, rem-jet on / off
    Vec3 layerSplit;                       // r, g, b; r == 1
  };
  // added to Stock:
  std::array<MtfCurve, 3> mtf;             // layer order as `curve`
  std::optional<Halation> halation;        // negatives only
  ```

- [ ] **Step 1: Digitise the 7219 MTF graph (vector, page 3)** — `figures/7219_mtf.json`, columns
  `["cyclesPerMm", "responsePercent"]`, both axes `"log": true` (x 1–600, y 1–200), curves red/green/blue
  told apart by colour/dash/bbox as in `7219_sensitometry.json`. Run `pdf_vectors.py`, then `overlay.py`;
  look at `build/digitize-review/7219_mtf.png` and fix until the magenta sits on each line.
- [ ] **Step 2: Hand-read the 2383 MTF graph (raster, page 4)** — `figures/2383_mtf.json` with
  `"hand": true` curves and explicit `p0`/`p1` from the tick marks; at least 10 points per curve from the
  curve start (~3 c/mm) to its end, including the G curve's flat 30–80 c/mm stretch. Overlay and fix.
- [ ] **Step 3: Halation inputs.** Search for published values: 35 mm acetate camera-negative base
  thickness; reflectance of rem-jet-backed vs unbacked base (halation literature, Kodak H-1/H-845,
  CineStill notes). Record what you find with URLs. Where nothing is published, use these documented
  assumptions and say so in the source entry:
  `baseThicknessMm` 0.127 (5 mil acetate); `reflectanceOff` = diffuse internal reflectance of a
  base/air interface, `1 − (1 − 0.092)/n²` with n = 1.48 (cellulose triacetate) ≈ 0.585, times the
  assumed ratio of red light reaching the base to red light absorbed, 0.5 → 0.29; `reflectanceOn` 0.01
  (carbon-black rem-jet). `layerSplit` is derived by `build_stock.py`, not typed in:
  `w_c = Σ_{600..700 nm} s_c(λ) / Σ_{600..700 nm} s_red(λ)` on the stock's linear sensitivities (light
  reaching the base is what the upper layers did not absorb: red).
- [ ] **Step 4: Write the failing tests** in `tests/core/test_stock.cpp`:
  - `TEST_CASE("5219 carries MTF and halation data")`: each of 3 curves has ≥ 8 points, f strictly
    ascending, first response in [0.9, 1.3], every response > 0; `halation` present;
    `0.05 < baseThicknessMm < 0.3`; `reflectanceOff > reflectanceOn > 0`; `layerSplit[0] == 1`,
    `layerSplit[0] > layerSplit[1] > layerSplit[2] >= 0`.
  - `TEST_CASE("2383 carries MTF, no halation")`: 3 curves as above; `!halation`.
  - `TEST_CASE("malformed MTF is rejected")`: a copy of the 5219 JSON with one curve's frequencies
    reversed → `loadStock` throws `std::runtime_error` naming `mtf`; a negative without `halation` throws.
- [ ] **Step 5: Run to see them fail** — `ctest --test-dir build -R core --output-on-failure`; expected FAIL
  (no `mtf` member).
- [ ] **Step 6: Extend schema, meta, `build_stock.py`, loader.** Schema: `mtf` required for every
  stock (`{red, green, blue}` → arrays of `[f > 0, response > 0]`); `halation` required when `type` is
  `negative` (`baseThicknessMm`, `remjetReflectance {on, off}`, `layerSplit [r,g,b]`, `sources`).
  `build_stock.py` converts percent → fraction, keeps the digitised points (no resampling), computes
  `layerSplit`. `loadStock` maps `remjetReflectance.on/off` to `reflectanceOn/Off` and validates as the
  tests demand. Regenerate both stock files (`build_stock.py` must print `valid`). README: document the
  MTF figures and the halation block.
- [ ] **Step 7: Run tests** — all `core` tests PASS.
- [ ] **Step 8: Commit** — `feat(data): datasheet MTF curves and halation inputs`.

### Task 2: Sharpness fit, optics and film formats

**Files:**
- Create: `src/core/optics.hpp`, `src/core/optics.cpp`, `tests/core/test_optics.cpp`
- Modify: `src/core/lut.hpp`, `src/core/lut.cpp` (`BakedFilm` gains `optics`), `CMakeLists.txt`

**Interfaces:**
- Consumes: `Stock::mtf`, `Stock::halation` (Task 1).
- Produces (`optics.hpp`):
  ```cpp
  struct Lobe { double weight; double sigmaMm; };           // a normalised Gaussian carrying `weight` of the light
  struct LayerSharpness { std::vector<Lobe> scatter; double edgeK = 0; double edgeSigmaMm = 0; };
  double gaussianMtf(double sigmaMm, double cyclesPerMm);   // exp(-2 pi^2 sigma^2 f^2)
  double sharpnessResponse(const LayerSharpness&, double cyclesPerMm);  // MTF_s(f) * (1 + k (1 - MTF_A(f)))
  LayerSharpness fitSharpness(const MtfCurve& mtf, bool withEdgeEffect);
  struct FilmOptics {
    std::array<LayerSharpness, 3> negative;   // scatter + edge effect
    std::array<LayerSharpness, 3> print;      // scatter only (edgeK = 0)
    Halation halation;                        // from the negative stock
  };
  FilmOptics makeOptics(const Stock& negative, const Stock& print);
  std::vector<Lobe> halationLobes(double baseThicknessMm, double radiusMultiplier);
  std::vector<Lobe> bloomLobes(double radiusMm);
  enum class FilmFormat { Super35, Academy35, Super16, Std16, Mm65, Custom };
  double gateWidthMm(FilmFormat, double customMm);          // custom clamped to [5, 80]; non-finite -> 24.89
  double pixelsPerMm(double imageWidthPx, double gateMm);
  ```
  `BakedFilm` gains `FilmOptics optics;`, filled by `bakeFilm`.

- [ ] **Step 1: Write the failing tests** (`test_optics.cpp`):
  - `"fit matches the 5219 datasheet"`: for each layer, `fitSharpness(mtf, true)`; every digitised
    point satisfies `|sharpnessResponse − response| ≤ 0.05`; lobe weights sum to 1 within 1e-9; all
    sigmas > 0; `edgeK ≥ 0`.
  - `"fit matches the 2383 datasheet"`: same with `withEdgeEffect = false`; `edgeK == 0`.
  - `"fit is deterministic"`: two fits of the same curve are bit-identical.
  - `"gaussian mtf"`: `gaussianMtf(s, 0) == 1`; at `f = sqrt(ln 2 / (2 pi^2)) / s` it is 0.5 within 1e-12.
  - `"lobes are normalised"`: `halationLobes(0.127, 1)` and `bloomLobes(0.5)` weights sum to 1;
    `halationLobes(t, 2)` sigmas are exactly twice `halationLobes(t, 1)`'s.
  - `"formats"`: Super35 24.89, Academy35 21.95, Super16 12.52, Std16 10.26, Mm65 52.48; Custom 3 → 5,
    100 → 80, NaN → 24.89; `pixelsPerMm(3840, 24.89)` = 3840/24.89.
  - `"bake carries optics"`: the 5219/2383 bake's `optics.negative[0].scatter` is non-empty.
- [ ] **Step 2: Run to see them fail** (missing header) — `ctest -R core`.
- [ ] **Step 3: Implement.** Pinned values:
  - Scatter = 2 lobes `{w, σ1}, {1−w, σ2}`; edge = one Gaussian `σ_A` with gain `k`. Negative fit: 5
    parameters (w, σ1, σ2, k, σ_A); print fit: 3 (k = 0).
  - Fitter: Nelder–Mead on unconstrained parameters (w via logistic, σ and k via exp), objective = sum
    of squared response errors over the digitised points; deterministic starts: σ1 ∈ {0.5, 1, 2} ×
    `s50`, σ2 = 4·σ1, w = 0.8, k ∈ {0, 0.2}, σ_A = 0.02 mm, where `s50 = sqrt(ln2/(2π²)) / f50` and
    `f50` is the first frequency where the curve falls below 0.5 (or its last frequency). Keep the best;
    max 4000 iterations per start; tolerance 1e-12.
  - `halationLobes(t, R)` = `{0.5, 1.0·t·R}, {0.5, 2.0·t·R}` (the total-internal-reflection ring of a
    base of thickness t sits near radius 1.83·t; the two lobes bracket it).
  - `bloomLobes(r)` = `{0.7, 0.5·r}, {0.3, 1.5·r}`.
- [ ] **Step 4: Run tests** — all PASS. If a datasheet curve cannot meet 0.05, stop and report the
  worst point (the spec's bar; do not loosen it in the test).
- [ ] **Step 5: Commit** — `feat(core): fit per-layer sharpness to datasheet MTF; film formats`.

### Task 3: Gaussian spread library

**Files:**
- Create: `src/core/blur.hpp`, `src/core/blur.cpp`, `tests/core/test_blur.cpp`
- Modify: `CMakeLists.txt`

**Interfaces:**
- Produces (`blur.hpp`):
  ```cpp
  struct Image4f { int width = 0, height = 0; std::vector<float> px; float* at(int x, int y); const float* at(int x, int y) const; };  // RGBA interleaved
  struct PxLobe { float weight; std::array<float, 3> sigmaPx; };   // per channel
  // out[c] = Σ_lobes weight * (G_{sigmaPx[c]} * in)[c] for c < 3; alpha copied. Edge clamp at image borders.
  void spread(const Image4f& in, Image4f& out, const std::vector<PxLobe>& lobes);
  // One lobe's blur of RGB, used by spread() and by the halation/edge passes.
  void gaussianBlur(const Image4f& in, Image4f& out, const std::array<float, 3>& sigmaPx);
  int blurRadiusPx(float sigmaPx);           // ceil(3 sigma); 0 when sigma < 0.25 (identity)
  int downsampleFactor(float sigmaPx);       // 1 if sigma <= 4, else largest power of two f <= 32 with sigma / f >= 2
  ```
- Pinned algorithm (the GPU must reproduce it exactly):
  - Taps: `exp(-i²/(2σ²))` for `|i| ≤ blurRadiusPx(σ)`, normalised to sum 1, separable (horizontal
    then vertical), clamp-to-edge addressing. σ < 0.25 px → copy.
  - Per lobe, the factor uses the smallest σ of its three channels. For f > 1: box-average f×f blocks
    (partial edge blocks average only their real pixels) into a `ceil(w/f) × ceil(h/f)` image; blur it
    with `σ_c' = sqrt(max(σ_c² − (f²−1)/12 − f²/6, 0.0625)) / f`; upsample by bilinear interpolation at
    fine-pixel centres `((x + 0.5)/f − 0.5)`, clamp-to-edge.

- [ ] **Step 1: Write the failing tests** (`test_blur.cpp`):
  - `"light is conserved"`: a single unit pixel in the centre of a 512×512 image, lobes with σ 0.3, 3,
    20, 90 px: `Σ out = Σ in` within 1e-4 relative, per channel.
  - `"matches a continuous Gaussian"`: a vertical step edge, σ = 6 px and σ = 60 px: the output profile
    is within 0.01 (of step height 1) of `0.5·erfc(-x/(σ√2))` everywhere.
  - `"tiny sigma is identity"`: σ = 0.2 → output bit-identical to input.
  - `"per-channel sigmas"`: σ = {1, 4, 16}: the three channels' second moments of a point response
    are σ² within 3 %.
  - `"kernel larger than image"`: 64×36 image and 1×1 image, σ = 500 px: finite, sum conserved within
    1e-3, no sanitizer error (build the test with `-fsanitize=address` locally once).
  - `"downsample factor"`: `downsampleFactor(4) == 1`, `(4.1) == 2`, `(16) == 8`, `(1000) == 32`.
- [ ] **Step 2: Run to see them fail** — `ctest -R core`.
- [ ] **Step 3: Implement** with threads over rows, as `film_renderer.cpp` does.
- [ ] **Step 4: Run tests** — PASS.
- [ ] **Step 5: Commit** — `feat(core): Gaussian spread with reduced-resolution wide lobes`.

### Task 4: CPU spatial pipeline

**Files:**
- Create: `src/core/film_spatial.hpp`, `src/core/film_spatial.cpp`, `tests/core/test_film_spatial.cpp`
- Modify: `CMakeLists.txt`

**Interfaces:**
- Consumes: `BakedFilm` (+`optics`), `LookControls`, `renderFilmCpu`, `exposureGain`,
  `printOffsetsLogE` (M3); Tasks 2–3.
- Produces (`film_spatial.hpp`):
  ```cpp
  struct SpatialControls {
    bool sharpness = true;
    bool halation = true; bool remjet = true; double halationStrength = 1; double halationRadius = 1;
    bool bloom = false; double bloomAmount = 0.25; double bloomRadiusMm = 0.5;
    double pxPerMm = 3840 / 24.89;
    bool anyOn() const;   // sharpness || halation || bloom
  };
  SpatialControls sanitize(SpatialControls);   // non-finite -> default; clamp to the spec ranges; pxPerMm <= 0 or non-finite -> default
  int spatialMarginPx(const FilmOptics&, const SpatialControls&);  // sum over active stages of the largest blurRadiusPx, full-res px
  // Calls renderFilmCpu when !anyOn(); otherwise the staged chain below. Same window/alpha/NaN rules as renderFilmCpu.
  void renderFilm(const ConstImageView& src, const ImageView& dst, const RenderWindow& win, const BakedFilm& film,
                  const LookControls& look, const SpatialControls& spatial);
  // Stages (public for tests and as the GPU's reference). Working pixel (0,0) is image pixel (rect.x1, rect.y1).
  struct WorkRect { int x1, y1, x2, y2; };     // window grown by the margin, cut to the source bounds
  WorkRect workRect(const ConstImageView& src, const RenderWindow& win, int marginPx);
  void exposeStage(const ConstImageView& src, const WorkRect&, const BakedFilm&, float gain, Image4f& E);   // RGB = (10^lutE − eps)·gain, A = alpha
  void exposureEffects(Image4f& E, const FilmOptics&, const SpatialControls&);   // bloom, scatter, halation in that order
  void developStage(Image4f& EtoD, const BakedFilm&);                           // log10(max(E,1e-30)) -> negCurves
  void edgeEffect(Image4f& D, const FilmOptics&, const SpatialControls&);
  void printExposeStage(Image4f& DtoP, const BakedFilm&);                       // lutPE -> P = 10^pe
  void printSharpness(Image4f& P, const FilmOptics&, const SpatialControls&);
  void finishStage(const Image4f& P, const WorkRect&, const BakedFilm&, const Vec3& offsets, const ImageView& dst, const RenderWindow& win);
  ```
- Pinned maths: bloom `E ← (1−a)E + a·spread(E, bloomLobes)`; scatter `E_c ← spread_c(E, negative[c].scatter)`;
  halation `E_c ← E_c + h·w_c·(H∗E_r)` with `h = (remjet ? reflectanceOn : reflectanceOff) · Strength`,
  `H = halationLobes(baseThicknessMm, Radius)`; edge `D_c ← D_c + k_c(D_c − G_{σA,c}∗D_c)`;
  print `P_c ← spread_c(P, print[c].scatter)`; finish `log10(max(P,1e-30)) + offsets → printCurves → lutD
  → 10^q → displayMatrix → encodeDisplay`, written only inside `win`; window pixels outside the source
  are (0,0,0,0) as in M3. Lobe σ_px = σ_mm · pxPerMm.

- [ ] **Step 1: Write the failing tests** (`test_film_spatial.cpp`; use the real 5219/2383 bake, as `test_film_renderer.cpp` does):
  - `"all off is M3"`: random 97×61 image incl. NaN/negative pixels and a sub-window: `renderFilm` with
    all three off is bit-identical to `renderFilmCpu`.
  - `"staged chain without effects matches M3"`: run the stages directly with no effect applied:
    max |diff| ≤ 1e-5, alpha bit-exact.
  - `"measured sharpness matches the datasheet"`: Halation and Bloom off, Sharpness on,
    `pxPerMm = 4096 / 5` (gate Custom 5 mm). For each digitised 5219 frequency ≤ 200 c/mm and each layer:
    a horizontal sine of exposure, mean = mid-grey exposure, amplitude 2 %, through `exposeStage →
    exposureEffects → developStage → edgeEffect`; response = density modulation with Sharpness on ÷
    with it off (measured by projecting onto the sine over the central 50 % of a 4096-wide row).
    Every response within 0.07 of the datasheet point.
  - `"halation physics"`: a single pixel of exposure 1000× mid-grey on black, Bloom and Sharpness off.
    Halo energy (Σ E_out − Σ E_in per channel after `exposureEffects`) has red > green > blue > 0;
    energy(rem-jet off) / energy(rem-jet on) = `reflectanceOff / reflectanceOn` within 5 %; Halation off
    → `E` bit-identical to its input.
  - `"edge effect keeps mean density"`: a 256×256 density image with a constant 64-px border and noise
    inside: mean before/after within 1e-4.
  - `"format scaling"`: halo second-moment radius (px) of a point at gate 24.89 is 2× that at 49.78
    within 1 px; same for bloom.
  - `"render scale matches the downsampled render"`: smooth 512×288 image (soft highlight blobs, σ ≥ 8 px)
    rendered at pxPerMm p; its 2×2 box-downsample vs the 2×-downsampled input rendered at p/2: mean
    |diff| over the interior (16 px border excluded) ≤ 2e-3.
  - `"sub-window equals full frame"` (Review Focus 3): window = central 40 % with the full source,
    every effect on: pixels equal the full-frame render's within 1e-6. Source cut to the window only
    (host ignored the ROI): finite output.
  - `"extreme highlights stay finite"` (Review Focus 1): DI 1.0 at +10 stops, Strength 5, Radius 4,
    Bloom 1 / 5 mm: all outputs finite and in [0, 1].
  - `"tiny image and huge kernels"` (Review Focus 2): 64×36 and 1×1 with gate Std16, Bloom 5 mm,
    Radius 4: finite output.
  - `"sanitize"` (Review Focus 5): NaN Strength → 1; Radius −1 → 0.25; Bloom Amount 7 → 1; pxPerMm NaN
    or 0 → default.
- [ ] **Step 2: Run to see them fail** — `ctest -R core`.
- [ ] **Step 3: Implement** the stages and `renderFilm`.
- [ ] **Step 4: Run tests** — PASS (whole `core` suite).
- [ ] **Step 5: Commit** — `feat(core): spatial film pipeline — sharpness, edge effect, halation, bloom`.

### Task 5: Metal spatial passes

**Files:**
- Create: `src/metal/film_common.h` (table lookups and curves shared by both .metal files),
  `src/metal/spatial.metal`
- Modify: `src/metal/film.metal` (include `film_common.h`; behaviour unchanged), `src/metal/shader_types.h`,
  `src/metal/metal_renderer.h`, `src/metal/metal_renderer.mm`, `CMakeLists.txt` (add `spatial.metal`
  to `openfilm_add_metallib`), `tests/gpu/test_gpu_vs_cpu.mm`

**Interfaces:**
- Consumes: Task 4's stages and `SpatialControls`, `spatialMarginPx`, `workRect`.
- Produces: `bool MetalRenderer::render(void* commandQueue, void* srcBuffer, const ImageGeometry& srcGeom,
  void* dstBuffer, const ImageGeometry& dstGeom, const RenderWindow& win, const std::shared_ptr<const BakedFilm>& film,
  const LookControls& look, const SpatialControls& spatial);` — calls the M3 path unchanged when
  `!spatial.anyOn()`. The old overload stays (M3 tests use it).
- One GPU kernel per CPU stage/pass (expose, horizontal blur, vertical blur(+accumulate), box down,
  bilinear up(+accumulate), halation add, develop, edge combine, print expose, finish), same order and
  maths as Task 4. Blur taps are computed on the host with the CPU's function and uploaded.
- Intermediate images: float4 `MTLBuffer`s from a per-device pool; a render checks out a set big enough
  for its `WorkRect`, and returns it in the command buffer's completion handler.
- Fold in the deferred M3 items: `static_assert`s on `OFFilmUniforms` (and the new uniform structs)
  size and offsets in `shader_types.h` for the C++ side; threadgroup size from
  `pso.maxTotalThreadsPerThreadgroup` / `threadExecutionWidth` instead of hard-coded 16×16.

- [ ] **Step 1: Write the failing tests** (`test_gpu_vs_cpu.mm`): compare against `renderFilm` with
  every effect on (Bloom on), and each effect alone:
  - `"spatial: points, edges, noise"`: max |diff| ≤ 1e-3 per channel, alpha bit-exact.
  - `"spatial: odd sizes, padded rows, sub-windows"`: 1×1, 3×97, 1001×17; rowBytes padding; window
    inside the image — same bar.
  - `"spatial off runs the M3 kernel"`: output bit-identical to the old overload.
  - `"spatial: extreme highlights finite"` (Review Focus 1).
  - `"spatial: concurrent renders of different sizes"` (Review Focus 4): 8 threads, sizes 64×64 to
    1920×1080, each compared with its CPU render — same bar.
- [ ] **Step 2: Run to see them fail** — `ctest --test-dir build -R gpu --output-on-failure`.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run tests** — `gpu` and `core` PASS; M3's GPU tests unchanged and passing.
- [ ] **Step 5: Commit** — `feat(metal): spatial passes mirroring the CPU pipeline`.

### Task 6: UHD speed budget

**Files:**
- Modify: `src/metal/*`, `src/core/blur.cpp` / `film_spatial.cpp` (only if a lever changes the maths the
  CPU must mirror), `tests/gpu/test_gpu_vs_cpu.mm`

- [ ] **Step 1: Write the failing benchmark** — `TEST_CASE("benchmark: UHD all M4 effects under 20 ms")`:
  3840×2160 uniform-noise frame (the M3 worst case), Super 35, every control at its default with Bloom
  enabled, tables already uploaded; median of 50 frames ≤ 20 ms. Print the median and the time of each
  pass (GPU start/end per pass in a profiling run of the benchmark).
- [ ] **Step 2: Run it** — record the median and per-pass times in the report. If already ≤ 20 ms, go to Step 4.
- [ ] **Step 3: Apply levers in this order, re-measuring after each, stopping at ≤ 20 ms:**
  1. Fuse each pointwise stage into the pass before it (accumulate in the vertical blur / upsample pass;
     develop into the last exposure-domain pass; print-expose into the edge combine; skip identity lobes).
  2. Half-float (`half4`) intermediate images, with the CPU mirroring the rounding (store through
     `_Float16`) so the GPU-vs-CPU bar still holds.
  3. Colour-table re-budget (M3's deferred item): half-float `lutE` or a smaller GPU `lutE`, with M3's
     accuracy test (`max ΔE2000 < 0.5`) re-run and passing.
  After each lever all `core` and `gpu` tests must pass.
- [ ] **Step 4: Run all tests** — benchmark PASS, everything else PASS.
- [ ] **Step 5: Commit** — `perf(metal): M4 passes within 20 ms UHD`.

### Task 7: OFX controls, regions of interest, docs, Resolve check

**Files:**
- Modify: `src/ofx/openfilm_plugin.cpp`, `tests/ofx/test_bundle_loads.cpp`, `docs/film-model.md`, `README.md`

**Interfaces:**
- Consumes: `SpatialControls`, `sanitize`, `spatialMarginPx`, `renderFilm`, `MetalRenderer::render`
  (spatial overload), `FilmFormat`, `gateWidthMm`, `pixelsPerMm`.

- [ ] **Step 1: Write the failing test** — `test_bundle_loads.cpp`: the described plugin has params
  `filmFormat` (6 options, default Super 35), `gateWidth` (5–80), `sharpnessEnable` (on),
  `halationEnable` (on), `halationRemjet` (on), `halationStrength` (0–5, 1), `halationRadius` (0.25–4, 1),
  `bloomEnable` (off), `bloomAmount` (0–1, 0.25), `bloomRadius` (0.05–5, 0.5), in groups "Film format",
  "Sharpness", "Halation", "Bloom", with the spec's labels.
- [ ] **Step 2: Run to see it fail** — `ctest -R bundle`.
- [ ] **Step 3: Implement.**
  - Params per the spec table; `gateWidth` enabled only when Format is Custom (`changedParam`, wrapped).
  - `readLook` also fills a `SpatialControls` (then `sanitize`). `pxPerMm = pixelsPerMm(source RoD width
    (canonical) × args.renderScale.x, gateWidthMm(format, gateWidth))`.
  - `getRegionsOfInterest`: source ROI = render window grown by `spatialMarginPx / renderScale.x`
    (canonical units). Wrapped; rendering stays correct if the host ignores it (Task 4's clamp).
    Log once per session whether the host called it and the render scale it passed.
  - Both render paths call the spatial entry points (`renderFilm`, the spatial `MetalRenderer::render`).
- [ ] **Step 4: Run all tests** — `ctest --test-dir build --output-on-failure`: all PASS, bundle
  signature valid.
- [ ] **Step 5: Docs** — `docs/film-model.md`: what each effect is physically, where it acts, the
  format menu, the halation inputs and which are assumptions (from Task 1's sources), the fitted MTF
  error per layer, the speed figures from Task 6. README: the new controls.
- [ ] **Step 6: Commit** — `feat(ofx): M4 controls, film format and regions of interest`.
- [ ] **Step 7: User review (stop and ask):** send the user the install commands and the
  `build/digitize-review/{7219,2383}_mtf.png` overlays. Checklist: overlays sit on the printed lines;
  real-time UHD with everything on; halation subtle with rem-jet, strong without; Super 16 vs Super 35
  visibly different in halo size and softness; effects off equals M3; whether Resolve called
  `getRegionsOfInterest` and passed render scale ≠ 1 on proxy playback (from the log).
