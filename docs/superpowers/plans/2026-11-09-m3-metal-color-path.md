# M3 — Metal Colour Path Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the M2 film look on the GPU in real time, with printer lights and Print Density applied per pixel (never re-baking).

**Architecture:** The print table is split at the print's exposure: negative density → print log E (3D, `lutPE`), + per-pixel offsets, → three 1D print curves → print density → display code (3D, `lutD`). The CPU renderer and one fused Metal kernel run the identical chain over the identical tables, with the same tetrahedral interpolation. The Metal renderer uploads a bake's tables once and reuses them.

**Tech Stack:** C++17, Objective-C++ (Metal host), Metal Shading Language, CMake, doctest, OpenFX Support library.

**Spec:** `docs/superpowers/specs/2026-11-09-m3-metal-color-path-design.md`

## Global Constraints

- macOS 12+, arm64. Target machine: M4 Pro.
- The look must not change. The direct model (`FilmModel::evaluate`) stays the reference: baked CPU render within **max ΔE2000 < 0.5** of it (M2's bar), GPU within **1e-3 code value per channel** of the CPU render, alpha bit-exact.
- Speed: GPU colour path **≤ 5 ms per UHD (3840×2160) frame**, median of 50, tables already uploaded.
- Control ranges (from the plugin): Exposure −10…+10 stops (shown −5…+5), Printer light R/G/B −20…+20 points, Print Density −20…+20 points. One printer point = 0.025 log E.
- Cache key = negative id, print id, gamut compression only. Exposure, printer points and Print Density are per-render.
- Non-finite control values are treated as 0. Non-finite pixel RGB is treated as 0 (the single NaN guard). Alpha is copied.
- `src/core` must not depend on Metal or OFX.
- Any OFX host-suite call that is not essential is wrapped like `showError()`/`clearError()` (Resolve may throw `kOfxStatErrUnsupported`).
- Commits use the repo's local noreply identity, with the trailer `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. Never commit datasheet PDFs. Never merge or push without the user asking.
- Comments are written for a colorist reading the code: say what the step means in film terms.

## Review Focus

1. **Concurrent first use of a bake on the GPU.** Resolve renders frames on several threads. Two threads first-using one bake must not double-upload or race; every frame must still match the CPU. Test: Task 3, "concurrent renders share one upload".
2. **Stock switching.** Evicted bakes must release their GPU tables, so memory does not grow as stocks change. Test: Task 3, "gpu tables are released with their bake".
3. **Typed-in extremes and NaN keyframes on the controls.** Points ±50 (beyond the slider), NaN exposure or points: output finite; NaN behaves as 0. Tests: Task 2, "extreme and non-finite controls"; Task 3, "extreme controls on the GPU".
4. **Degenerate windows.** Empty window, window outside the source bounds, 1-pixel-wide image: no crash, no write outside the window. Test: Task 3, "degenerate windows".
5. **Print side at the domain edges.** A completely clear negative printed with +20 points, and a dense one with −20 points, push print log E off the print curve: it must plateau (clamp), not extrapolate. Test: Task 2, "print exposure beyond the print curve clamps".

---

### Task 1: Split the print model API (core)

**Files:**
- Modify: `src/core/print.hpp`, `src/core/print.cpp`, `src/core/film_model.hpp`, `src/core/film_model.cpp`
- Modify: `src/core/lut.cpp` (call-site only, see Step 4), `src/ofx/openfilm_plugin.cpp` (call-site only, see Step 4)
- Test: `tests/core/test_film_model.cpp`, `tests/core/test_print.cpp`

**Interfaces:**
- Produces (in `film_model.hpp`):
  ```cpp
  struct LookControls {            // per-render; never part of the bake
    double exposureStops = 0;
    Vec3 printerPoints{0, 0, 0};   // R, G, B points
    double printDensity = 0;       // points added to all three
  };
  // balance + 0.025 * (points + density), per channel. Non-finite control values count as 0.
  Vec3 printOffsetsLogE(const Vec3& balance, const LookControls& c);
  ```
  `FilmParams` loses `printerLights` and `printDensity` (keeps `negativeId`, `printId`, `gamutCompression`); `hash()` covers those three.
  New and changed `FilmModel` members:
  ```cpp
  Vec3 printLogExposureBase(const Vec3& negStatusMDensity) const;       // print log E, zero offsets
  Vec3 printDensity(const Vec3& printLogE) const;                        // Status A, from the print curves
  Vec3 displayFromPrintDensity(const Vec3& printStatusADensity) const;   // projector -> D65 -> 709 -> anchor -> BT.1886
  std::array<double, 2> printLogERange(int ch) const;                    // print curve's digitised log E span
  std::array<double, 2> printDensityRange(int ch) const;                 // print curve's min..max density
  Vec3 printerOffsets(const LookControls& c) const;                      // printOffsetsLogE(balance(), c)
  Vec3 printOutput(const Vec3& negStatusMDensity, const LookControls& c) const;
  Vec3 evaluate(const Vec3& diCode, const LookControls& c) const;
  ```
  The old `printerOffsets()`, `printOutput(d)` and `evaluate(di, stops)` are removed.
- Produces (in `print.hpp`): `Xyz PrintModel::projectedXyzFromDensity(const Vec3& printStatusADensity) const;`; `projectedXyz(logE)` becomes `projectedXyzFromDensity(printDensity(logE))`.

- [ ] **Step 1: Write the failing tests** in `tests/core/test_film_model.cpp`. Update every existing use of `printerLights`, `printDensity`, `printerOffsets()`, `printOutput(d)` and `evaluate(di, stops)` to the `LookControls` forms, keeping their assertions. Replace "params hash covers every field" with the third test below. Add:

```cpp
TEST_CASE("print output factors through print log E and print density") {
  // printOutput must equal the split chain exactly: this is what lets the bake split the table.
  const FilmModel m(res(), params());
  std::mt19937 rng(5);
  std::uniform_real_distribution<double> ud(0.1, 2.5), up(-20, 20);
  for (int i = 0; i < 200; ++i) {
    const Vec3 d{ud(rng), ud(rng), ud(rng)};
    LookControls c; c.printerPoints = {up(rng), up(rng), up(rng)}; c.printDensity = up(rng);
    const Vec3 off = m.printerOffsets(c);
    Vec3 e = m.printLogExposureBase(d);
    for (int k = 0; k < 3; ++k) e[k] += off[k];
    const Vec3 split = m.displayFromPrintDensity(m.printDensity(e));
    const Vec3 direct = m.printOutput(d, c);
    for (int k = 0; k < 3; ++k) CHECK(split[k] == doctest::Approx(direct[k]).epsilon(1e-12));
  }
}

TEST_CASE("print offsets: balance plus 0.025 per point, non-finite counts as zero") {
  LookControls c; c.printerPoints = {4, 0, -2}; c.printDensity = 1;
  const Vec3 o = printOffsetsLogE({0.1, 0.2, 0.3}, c);
  CHECK(o[0] == doctest::Approx(0.1 + 0.025 * 5));
  CHECK(o[1] == doctest::Approx(0.2 + 0.025 * 1));
  CHECK(o[2] == doctest::Approx(0.3 + 0.025 * -1));
  c.printerPoints[1] = std::numeric_limits<double>::quiet_NaN(); c.printDensity = INFINITY;
  const Vec3 n = printOffsetsLogE({0, 0, 0}, c);
  CHECK(n[1] == 0.0);
  CHECK(n[0] == doctest::Approx(0.1));
}

TEST_CASE("params hash holds only bake inputs") {
  FilmParams a = params(), b = params();
  CHECK(a.hash() == b.hash());
  b.gamutCompression = 0.6; CHECK(a.hash() != b.hash());
  b = params(); b.printId = "x"; CHECK(a.hash() != b.hash());
  b = params(); b.negativeId = "x"; CHECK(a.hash() != b.hash());
}
```

In `tests/core/test_print.cpp` add "projected XYZ from density matches from log E": for 20 print log E triples, `projectedXyz(e)` equals `projectedXyzFromDensity(printDensity(e))` exactly.

- [ ] **Step 2: Run to see them fail**

Run: `cmake --build build --target test_core 2>&1 | tail -5`
Expected: compile errors naming `LookControls`, `printLogExposureBase`, `projectedXyzFromDensity`.

- [ ] **Step 3: Implement** the interfaces above. `printOutput` must *be* `displayFromPrintDensity(printDensity(printLogExposureBase(d) + printerOffsets(c)))`, one code path, so Step 1's equality is exact. `displayFromPrintDensity` = `projectedXyzFromDensity` → `adapt_` → `toRec709_` → × `anchorScale_` → `bt1886Encode`. The balance solve keeps its own residual path; `anchorScale_` must not change (the existing "grey prints neutral at code 0.409" test guards it). `printDensityRange(ch)`: min and max of the print stock's `curve[ch]` density values.

- [ ] **Step 4: Keep the build green.** In `lut.cpp`, bake `lutP` with `model.printOutput(d, LookControls{})` (temporary; Task 2 replaces the print table). In `openfilm_plugin.cpp`, stop writing the removed `FilmParams` fields. Until Task 2 the CPU render ignores printer points and Print Density; say so in the commit message.

- [ ] **Step 5: Run the suite**

Run: `cmake --build build && ctest --test-dir build --output-on-failure 2>&1 | tail -5`
Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/core tests/core src/ofx/openfilm_plugin.cpp
git commit -m "refactor(core): split print output at print exposure; printer controls leave the bake key"
```

---

### Task 2: Split bake, per-pixel printer offsets in the CPU renderer

**Files:**
- Modify: `src/core/lut.hpp`, `src/core/lut.cpp`, `src/core/film_renderer.hpp`, `src/core/film_renderer.cpp`, `src/ofx/openfilm_plugin.cpp`
- Test: `tests/core/test_film_renderer.cpp`

**Interfaces:**
- Consumes: Task 1's `LookControls`, `printOffsetsLogE`, the `FilmModel` split members.
- Produces:
  ```cpp
  struct BakedFilm {
    Lut3D lutE;                         // unchanged: DI -> log10(E + 1e-3), 113^3, t^2 nodes
    std::array<Curve1D, 3> negCurves;   // unchanged
    Lut3D lutPE;                        // neg Status M density (reachable domain) -> print log E, zero offsets
    std::array<Curve1D, 3> printCurves; // print log E over [print curve min - 1, max + 1] -> Status A density
    Lut3D lutD;                         // print density over [printDensityRange -/+ 0.02, floor 0] -> display code
    Vec3 balance;                       // auto-balance offsets (log E)
  };
  std::shared_ptr<const BakedFilm> bakeFilm(const FilmModel& model, int lutSizeE = 113, int lutSizePE = 97, int lutSizeD = 97);
  float exposureGain(double stops);    // 2^clamp(stops, -64, 64), non-finite -> 1; shared with the GPU host code
  void renderFilmCpu(const ConstImageView& src, const ImageView& dst, const RenderWindow& win,
                     const BakedFilm& film, const LookControls& controls);
  ```
  Per pixel: `lutE` → `(10^q − 1e-3) × gain` → log10 (floor 1e-30) → `negCurves` → `lutPE` → `+ offsets` → `printCurves` → `lutD`. Offsets are computed once per render, in float, from `printOffsetsLogE(film.balance, controls)`. `lutP` is removed.

- [ ] **Step 1: Update and add tests** in `tests/core/test_film_renderer.cpp`. `renderOne` takes a `LookControls`. The two existing accuracy tests draw printer points and Print Density uniformly from −20…+20 in addition to exposure, and compare against `model.evaluate(in, c)`; the bar stays `CHECK(maxDe < 0.5)`. "baked tables are finite…" checks `lutPE`, `lutD`, `printCurves` (all finite; `lutD` ≥ 0). Add:

```cpp
TEST_CASE("printer lights and print density need no rebake") {
  const auto film = cache().get(params());
  const float c = float(diEncode(0.18));
  LookControls none, red; red.printerPoints = {5, 0, 0};
  const Vec3 a = renderOne(*film, c, c, c, none), b = renderOne(*film, c, c, c, red);
  CHECK(cache().get(params()) == film);  // same bake object
  CHECK(b[0] < a[0] - 0.01);             // R +5 -> more cyan dye -> less red
}

TEST_CASE("extreme and non-finite controls") {
  const auto film = cache().get(params());
  LookControls big; big.printerPoints = {50, -50, 50}; big.printDensity = -50; big.exposureStops = 10;
  const Vec3 o = renderOne(*film, 0.5f, 0.2f, 0.9f, big);
  for (double v : o) CHECK(std::isfinite(v));
  LookControls nan; nan.printerPoints = {NAN, 0, 0}; nan.printDensity = INFINITY; nan.exposureStops = NAN;
  const Vec3 n = renderOne(*film, 0.4f, 0.4f, 0.4f, nan), z = renderOne(*film, 0.4f, 0.4f, 0.4f, LookControls{});
  for (int k = 0; k < 3; ++k) CHECK(n[k] == z[k]);
}

TEST_CASE("print exposure beyond the print curve clamps") {
  // Clear negative (DI 0) at +20 points and a dense one at -20: print log E leaves the curve's range.
  const FilmModel model(res(), params());
  const auto film = cache().get(params());
  for (double pts : {-20.0, 20.0})
    for (float v : {0.f, 1.f}) {
      LookControls c; c.printerPoints = {pts, pts, pts}; c.printDensity = pts; c.exposureStops = pts > 0 ? 10 : -10;
      const Vec3 baked = renderOne(*film, v, v, v, c), direct = model.evaluate({v, v, v}, c);
      CHECK(deltaE2000(toLab(direct), toLab(baked)) < 0.5);
    }
}
```

- [ ] **Step 2: Run to see them fail**

Run: `cmake --build build --target test_core 2>&1 | tail -5`
Expected: compile errors (`lutPE`, `printCurves`, `renderFilmCpu` with `LookControls`).

- [ ] **Step 3: Implement** `bakeFilm`, `exposureGain` and `renderFilmCpu` as specified. `lutPE` domain: the reachable negative density box, computed as `lutP`'s was (neg curve min/max per layer ± 0.02, floor 0). `printCurves` sample `model.printDensity({e,e,e})[c]` at 4096 points (layers are independent, as for the negative). `lutD` bakes `model.displayFromPrintDensity(d)`. `balance = model.balance()`.

- [ ] **Step 4: Update the plugin's CPU branch** to build `LookControls` from Exposure, the three printer lights and Print Density, and call the new `renderFilmCpu`. Non-finite handling now lives in `printOffsetsLogE` and `exposureGain`, so the plugin's `finiteOrZero` for these goes.

- [ ] **Step 5: Run the suite and read the accuracy printouts**

Run: `cmake --build build && ctest --test-dir build --output-on-failure 2>&1 | tail -20; ./build/test_core -tc="baked matches*" -s 2>&1 | grep -E "max dE|p50|dark|bake"`
Expected: all pass; max ΔE2000 < 0.5 in both accuracy tests. If not, raise `lutSizePE` and/or `lutSizeD` in steps of 16 up to 129, and re-run. Report the final sizes, the measured max, and the bake time (it was ~1.4 s).

- [ ] **Step 6: Commit**

```bash
git add src/core tests/core src/ofx/openfilm_plugin.cpp
git commit -m "feat(core): split print table; printer lights and print density applied per pixel"
```

---

### Task 3: Metal film kernel and GPU renderer

**Files:**
- Create: `src/metal/film.metal`
- Delete: `src/metal/exposure.metal`, `src/core/cpu_renderer.hpp`, `src/core/cpu_renderer.cpp`, `src/core/render_params.hpp`, `tests/core/test_cpu_renderer.cpp` (M1's test exposure path; nothing uses it once this task lands)
- Modify: `src/metal/shader_types.h`, `src/metal/metal_renderer.h`, `src/metal/metal_renderer.mm`, `cmake/OpenFilmBundle.cmake`, `CMakeLists.txt`, `src/ofx/openfilm_plugin.cpp` (GPU branch call-site; Metal stays declared unsupported until Task 4)
- Test: `tests/gpu/test_gpu_vs_cpu.mm` (rewritten)

**Interfaces:**
- Consumes: Task 2's `BakedFilm`, `LookControls`, `printOffsetsLogE`, `exposureGain`, `renderFilmCpu`; `Lut3D::data()` layout `((b*N + g)*N + r)*3`, node `i` at `(i/(N-1))^2` when dense-darks.
- Produces:
  ```cpp
  // metal_renderer.h
  bool MetalRenderer::render(void* commandQueue, void* srcBuffer, const ImageGeometry& srcGeom,
                             void* dstBuffer, const ImageGeometry& dstGeom, const RenderWindow& win,
                             const std::shared_ptr<const BakedFilm>& film, const LookControls& controls);
  size_t MetalRenderer::cachedFilmCount() const;  // live GPU table sets; for tests
  ```
  `shader_types.h` gains `OFLut3D {int size; int denseDarks; float lo[3]; float hi[3];}`, `OFCurve {float lo, hi; int offset, count;}` and `OFFilmUniforms {OFImageGeom src, dst; int winX1, winY1, winX2, winY2; float gain; float offsets[3]; OFLut3D lutE, lutPE, lutD; OFCurve neg[3], print[3];}` (plain C types only). Kernel buffers: 0 src, 1 dst, 2 uniforms (`setBytes`), 3 lutE, 4 lutPE, 5 lutD, 6 all six curves packed one after another.

- [ ] **Step 1: Rewrite `tests/gpu/test_gpu_vs_cpu.mm`.** `runBoth(srcGeom, dstGeom, win, controls, fill)` renders the same source through `renderFilmCpu` and `MetalRenderer::render` (`waitUntilCompleted`), and returns the max abs RGB difference and whether alpha matched bit-for-bit. The film comes from a `FilmCache` over `FilmResources::load(OPENFILM_SOURCE_DATA_DIR, OPENFILM_COEFF_PATH)` (add both compile definitions to `test_gpu`, and `add_dependencies(test_gpu openfilm_coeff)`). Cases, each `CHECK(maxDiff <= 1e-3)` and `CHECK(alphaExact)`:
  - "grey ramp": 256×16, DI 0…1, zero controls.
  - "saturated primaries and DWG edges": random DI in [0,1]³, seed 7, 512×512.
  - "exposure ±8 and points ±20": controls {stops +8, points (20, −20, 20), density −20} and the mirror.
  - "non-finite and negative pixels": NaN, ±inf, −1, 2 in RGB; alpha 0.25.
  - "offset bounds, padded rows, sub-window": as the M1 test (bounds not at 0, rowBytes + 64, window inside).
  - "odd sizes": 1×1, 3×7, 1921×1081.
  - **"extreme controls on the GPU"**: points ±50, NaN exposure; GPU equals CPU (Review Focus 3).
  - **"degenerate windows"**: empty window (x1 == x2), window entirely outside src bounds, 1-pixel-wide image; `render` returns true and nothing outside the window changes (Review Focus 4).
  - **"concurrent renders share one upload"**: a fresh bake; 4 threads each render the same frame on their own command queue at once; all match the CPU; `cachedFilmCount()` grew by exactly 1 (Review Focus 1).
  - **"gpu tables are released with their bake"**: render with a bake held only by a local `shared_ptr` (bake it directly with `bakeFilm`, not through the cache), drop it, render with another bake; `cachedFilmCount()` no longer counts the dropped one (Review Focus 2).
  - "invalid metallib path" (kept).
  - **"benchmark: UHD under 5 ms"**: 3840×2160, tables already uploaded (one warm-up frame), 50 frames, GPU time per frame from `GPUEndTime − GPUStartTime`; print the median; `CHECK(median <= 5e-3)`.

- [ ] **Step 2: Run to see them fail**

Run: `cmake --build build --target test_gpu 2>&1 | tail -5`
Expected: compile errors (new `render` signature, `cachedFilmCount`).

- [ ] **Step 3: Write `film.metal`.** One kernel, `film_kernel`, one thread per window pixel. It mirrors `renderRows` in `film_renderer.cpp` operation for operation, in float: the same NaN guard; the same `Lut3D::lookup` (clamp, the dense-darks `sqrt` of the normalised coordinate, the same cell index clamp, the same sorted-fraction tetrahedral walk); the same `Curve1D::eval`; the same floors. Window pixels outside `src` bounds write (0,0,0,0) like the CPU. Header comment: "GPU twin of renderFilmCpu(); any change there must be made here too."

- [ ] **Step 4: Compile shaders without fast math.** In `openfilm_add_metallib`, add `-fno-fast-math` to the `metal -c` command (closes the deferred M1 minor; fast-math `pow`/`log10` would drift from the CPU). Replace `exposure.metal` with `film.metal` in `CMakeLists.txt`. Remove `cpu_renderer.cpp` from `openfilm_core` and `test_cpu_renderer.cpp` from `test_core`.

- [ ] **Step 5: Implement the renderer.** Per-device pipeline as now. A GPU table set (`lutE`, `lutPE`, `lutD`, packed curves; `MTLResourceStorageModeShared` buffers) is looked up by `(id<MTLDevice>, const BakedFilm*)` and stored with a `std::weak_ptr<const BakedFilm>`; expired entries are erased on each lookup, which also means a new bake that reuses a freed address is never mistaken for the old one. Creation happens under the renderer's mutex, so concurrent first use uploads once. Encoding: `setBytes` for the uniforms (gain from `exposureGain`, offsets from `printOffsetsLogE`, as the CPU computes them), `dispatchThreads` over the window (empty window: no dispatch, return true), `commit`, no wait.

- [ ] **Step 6: Update the plugin's GPU branch** to the new `render` signature (get the bake from the cache, as the CPU branch does). Leave `setSupportsMetalRender(false)`; Task 4 turns it on.

- [ ] **Step 7: Run everything**

Run: `cmake --build build && ctest --test-dir build --output-on-failure 2>&1 | tail -20; ./build/test_gpu -tc="benchmark*" -s 2>&1 | grep -i median`
Expected: all tests pass; the printed median ≤ 5 ms. Put the measured median and the worst GPU-vs-CPU difference in the report.

- [ ] **Step 8: Commit**

```bash
git add -A src/metal src/core tests CMakeLists.txt cmake src/ofx/openfilm_plugin.cpp
git commit -m "feat(metal): film kernel and GPU renderer matching the CPU path; drop M1 exposure path"
```

---

### Task 4: Turn on Metal in the plugin, docs, and the Resolve check

**Files:**
- Modify: `src/ofx/openfilm_plugin.cpp`, `docs/film-model.md`, `README.md`
- Test: `tests/ofx/test_bundle_loads.cpp` and the bundle signature test (unchanged; must still pass)

**Interfaces:**
- Consumes: Task 3's `MetalRenderer::render`, Task 2's `LookControls`.

- [ ] **Step 1: One place reads the look.** Factor the parameter reads into one member, `void readLook(const OFX::RenderArguments&, const LoadedFilm&, openfilm::FilmParams&, openfilm::LookControls&)`, used by both branches, so the CPU and GPU paths can never read the controls differently.

- [ ] **Step 2: Enable Metal.** `desc.setSupportsMetalRender(true)`. GPU branch: film-data load failure, metallib failure, a bake exception, or `false` from `render` → `showError(...)` with a specific message and `throwSuiteStatusException(kOfxStatFailed)`; success → `clearError()`. Update the file's top comment and the M2 "Metal off" comments.

- [ ] **Step 3: Build, run all tests, stage the install**

Run: `cmake --build build && ctest --test-dir build --output-on-failure 2>&1 | tail -8`
Expected: all pass, including `bundle` and `bundle_signature`.
The install needs sudo, so the controller gives the user these commands rather than running them: `sudo rm -rf /Library/OFX/Plugins/OpenFilm.ofx.bundle && sudo cp -R build/OpenFilm.ofx.bundle /Library/OFX/Plugins/`, then delete `~/Library/Application Support/Blackmagic Design/DaVinci Resolve/OFXPluginCacheV2.xml` with Resolve closed.

- [ ] **Step 4: Docs.** `docs/film-model.md`: printer lights and Print Density are applied per pixel (no re-bake); the print table is split at print exposure (one colorist-facing paragraph); GPU rendering, with the CPU as the reference. `README.md` WIP note: M3 done (GPU rendering, real-time); still to come: halation, bloom, grain.

- [ ] **Step 5: Commit**

```bash
git add src/ofx/openfilm_plugin.cpp docs/film-model.md README.md
git commit -m "feat(ofx): render on Metal in Resolve; one parameter reader for both paths"
```

- [ ] **Step 6: User check in Resolve (stop and ask).** The controller asks the user to confirm, on a UHD timeline with the M2 node setup:
  1. the node changes the image (GPU path active, not pass-through);
  2. UHD plays back in real time;
  3. dragging a printer light updates instantly; R +5 → cyan;
  4. switching the negative stock waits ~1.4 s, then shows the new look;
  5. the M2 colour checks still hold (grey ≈ 41 %, exposure in stops, saturated reds/magentas smooth).
  If the node does nothing, look in `~/Library/Application Support/Blackmagic Design/DaVinci Resolve/logs/ResolveDebug.txt` for an OFX status error first (M2's failure was an unsupported host call).
