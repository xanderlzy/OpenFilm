# OpenFilm Milestone 2 (Core Colour Model, CPU) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** OpenFilm renders a spectral 500T (5219-type) → 2383-type print → Rec.709 gamma 2.4 film
look in Resolve, on the CPU, from digitised Kodak datasheet data.

**Architecture:**
- **Core model.** New `src/core/` units, each with one job: spectral maths, colour spaces, gamut compression, RGB→spectrum, stock data, densitometry, the negative, the print/display, and the LUT bake/cache. A `FilmModel` composes them into a direct per-pixel evaluation. `renderFilmCpu` runs that model as baked LUTs.
- **Plugin.** It gains the M2 parameters and uses the film renderer with Metal declared off.
- **Data.** Datasheet curves are produced offline by Python tools in `tools/digitize/` and committed as JSON. The RGB→spectrum coefficient table is generated at build time by a vendored C++ tool.

**Tech Stack:** C++17, CMake, doctest 2.4.11, nlohmann/json 3.11.3 (FetchContent), OpenFX 1.5.1, Python 3.9 venv (PyMuPDF 1.26.5, numpy, matplotlib, jsonschema) for offline digitising only.

**Spec:** `docs/superpowers/specs/2026-11-08-m2-color-model-design.md` (parent: `docs/superpowers/specs/2026-11-08-film-emulator-design.md`)

## Global Constraints

- Everything from M1 still applies:
  - MIT licence; plugin id `org.openfilm.OpenFilm`.
  - arm64, macOS 12.0.
  - Float RGBA only.
  - `src/core/` has no OFX/Apple includes.
  - Comments are written for a colorist reader.
  - Commits end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Spectral grid: **380–780 nm, 5 nm, 81 samples**. Every spectrum in core uses it.
- Input: **DaVinci Intermediate** (DWG primaries). Output: **Rec.709 primaries, BT.1886 γ 2.4**. Each is a choice param with exactly one entry.
- Printer point = **0.025 log E**. Printer lights R/G/B and Print density are in **−20…+20**, default 0. Gamut compression is in **0…1**, default **0.5**.
- Mid-grey anchor: 18 % grey at 0 stops gives output code **0.409** (display-linear 0.409^2.4 = 0.11706).
- White: the projector illuminant is **CIE D55**, Bradford-adapted to D65. Scene illuminant for 500T: **3200 K blackbody**. Printer lamp: **3200 K blackbody**.
- Stock display names: **"Tungsten 500T (5219-type)"** and **"Print 2383-type"**. Stock ids: `kodak-5219-type` and `kodak-2383-type`.
- LUT sizes: **65³** for both LUT_E and LUT_P. LUT_E domain is DI code [0,1]³. LUT_P domain is negative Status M density [0,4]³.
- Plugin declares **Metal render unsupported** in M2.
- Datasheet PDFs are never committed; each stock JSON cites its URLs:
  - `https://www.kodak.com/content/products-brochures/Film/VISION3-500T-Color-Negative-Film-7219-TECHNICAL-DATA.pdf` (H-1-5219t; 7219 is the 16 mm format of the 5219 emulsion)
  - `https://www.kodak.com/content/products-brochures/Film/KODAK-VISION-Color-Print-Film-2383-3383-data-sheet.pdf` (H-1-2383, rev. 3-22)
  - Status M/A source: ANSI/NAPM IT2.18-1996 (= ISO 5-3:1995), Tables 3 and 4, `https://law.resource.org/pub/us/cfr/ibr/001/aimm.it2.18.1996.pdf`

### Spec corrections (decided while planning; recorded here so executors follow them)
1. The spec's test "Planck 6504 K ≈ D65 ±0.003" is physically wrong: D65 lies on the daylight locus, about 0.005 off the Planckian locus. Replaced by **Planck 2856 K = Illuminant A, xy (0.44757, 0.40745) ±0.0005**.
2. `baseDensity` is **optional** in stock JSON. 7219 publishes a minimum-density spectrum; 2383 does not. When it is absent, the base is zero and D-min is carried by the dye amounts solved from the characteristic curve.
3. Spectral data comes from the CIE "S0/S1/S2 daylight components" plus CMFs. D65 and D55 are **computed** from the daylight model (CIE 15) rather than stored separately.

## Review Focus

1. **Input outside the LUT domain**: DI code < 0 (negative scene-linear noise) or > 1 (extreme highlights). Expected: clamped to the domain edge, finite output, no NaN. Test in Task 9.
2. **NaN/Inf input pixels**: Expected: finite RGB output (treated as black) and alpha passed through, never NaN spreading into LUT lookups. Test in Task 9.
3. **Pure black / zero light**: RGB 0 gives a zero spectrum and log10(0). Expected: clamped to the curve toe (D-min), giving the print's maximum black, finite. Tests in Task 7 and Task 8.
4. **Extreme DWG colours** (pure DWG primaries, e.g. code for linear (0,0,1)) through the full pipeline. Expected: finite, non-negative output; compression amount 0 vs 1 both finite. Test in Task 8.
5. **Concurrent renders with different parameters**: Resolve renders several frames or nodes at once, e.g. two nodes with different printer lights. Expected: each gets its own correct LUT from the cache, with no torn state. Test in Task 9.

---

## File Structure

```
data/spectral/cie_1931_2deg_cmf.csv      CIE 018:2019 dataset, 380–780 @5 nm (resampled if needed)
data/spectral/cie_daylight_components.csv S0,S1,S2 (CIE 15:2018 Table), 380–780 @5 nm
data/spectral/iso5_3_status_a.csv        Table 3, log10 spectral products, 400–750 @10 nm
data/spectral/iso5_3_status_m.csv        Table 4, log10 spectral products, 400–770 @10 nm
data/stocks/schema.json                  JSON Schema for stock files
data/stocks/kodak-5219-type.json         digitised 500T
data/stocks/kodak-2383-type.json         digitised 2383
tools/digitize/requirements.txt          pymupdf==1.26.5 numpy matplotlib jsonschema
tools/digitize/README.md                 how to (re)digitise, for contributors
tools/digitize/pdf_vectors.py            vector-path extraction + axis calibration → CSV per curve
tools/digitize/overlay.py                plots digitised points over the rendered PDF figure (PNG)
tools/digitize/build_stock.py            curve CSVs + metadata → stock JSON (resampled, validated)
tools/digitize/figures/*.json            per-figure config: pdf page, axis calibration, curve picks
tools/digitize/points/*.csv              digitised curve points (inputs to build_stock.py)
tools/rgb2spec/rgb2spec_opt.cpp          vendored Mitsuba rgb2spec optimiser (BSD-3, header kept)
tools/rgb2spec/LICENSE                   its licence
THIRD_PARTY.md                           rgb2spec (BSD-3), nlohmann/json (MIT), doctest (MIT), OpenFX (BSD-3)
src/core/spectral.hpp/.cpp               grid, Spectrum, tables, Planck, daylight, integrate, chromaticity
src/core/color.hpp/.cpp                  Vec3/Mat3, primaries, DI, Bradford, BT.1886
src/core/gamut.hpp/.cpp                  gamut compression into Rec.2020
src/core/rgb2spec.hpp/.cpp               coefficient table loader + RGB→spectrum
src/core/stock.hpp/.cpp                  Stock struct + JSON loader/validation
src/core/densitometry.hpp/.cpp           Status A/M densities; densities ↔ dye amounts
src/core/negative.hpp/.cpp               NegativeModel
src/core/print.hpp/.cpp                  PrintModel (print + display) + auto-balance
src/core/film_model.hpp/.cpp             FilmResources, FilmParams, FilmModel (direct evaluation)
src/core/lut.hpp/.cpp                    Lut3D (tetrahedral), Curve1D
src/core/film_renderer.hpp/.cpp          BakedFilm, FilmCache, renderFilmCpu
src/core/cpu_renderer.*                  renamed function: renderExposureCpu (M1 path, kept for GPU tests)
src/ofx/openfilm_plugin.cpp              M2 params, film renderer, Metal off
tests/core/test_*.cpp                    one test file per core unit
docs/film-model.md                       colorist-facing explanation of the model and controls
```

---

### Task 1: Spectral foundation

**Files:**
- Create: `data/spectral/cie_1931_2deg_cmf.csv`, `data/spectral/cie_daylight_components.csv`, `src/core/spectral.hpp`, `src/core/spectral.cpp`
- Modify: `CMakeLists.txt`: add sources to `openfilm_core` and add test files to `test_core`. Define `OPENFILM_SOURCE_DATA_DIR="${CMAKE_SOURCE_DIR}/data"` for `test_core`.
- Test: `tests/core/test_spectral.cpp`

**Interfaces** (namespace `openfilm`):
- Produces:
  - `constexpr int kSpectralSamples = 81; constexpr double kLambdaMin = 380, kLambdaStep = 5;` `double wavelength(int i);`
  - `using Spectrum = std::array<double, kSpectralSamples>;` `struct Xyz { double X, Y, Z; };` `struct Xy { double x, y; };`
  - `Spectrum resample(const std::vector<std::array<double,2>>& nmValue, double outside);` linear interpolation; samples outside the data range get `outside`.
  - `class SpectralTables { public: static SpectralTables load(const std::string& dataDir); const Spectrum& xBar() const; const Spectrum& yBar() const; const Spectrum& zBar() const; Spectrum daylight(double cctKelvin) const; };`
    - `load` throws `std::runtime_error` naming the file on any missing or bad CSV.
    - `daylight` uses the CIE 15 D-series formula (xD from CCT, M1/M2, S0+M1·S1+M2·S2), normalised to 100 at 560 nm.
  - `Spectrum blackbody(double kelvin);` Planck's law, normalised to 1 at 560 nm.
  - `Xyz integrate(const Spectrum& s, const SpectralTables& t);` = Σ s·x̄ / Σ ȳ (an equal-energy spectrum of 1 → Y = 1).
  - `Xy chromaticity(const Xyz&);` `std::array<double,2> uvPrime(const Xyz&);`
  - `double sum(const Spectrum&)`, element-wise `Spectrum operator*(const Spectrum&, const Spectrum&)` and scalar `*`.
- CSV format: `#` comment lines (source and licence note), then a header row, then `nm,values…` rows.

- [ ] **Step 1: Fetch the CIE data.**
  - Download the CIE 1931 2° CMF CSV (CIE dataset DOI 10.25039/CIE.DS.xvudnb9b) and the CIE daylight S0/S1/S2 components (from CIE 15 / CIE dataset "CIE D-series components").
  - Keep 380–780 nm rows at 5 nm. If the source is at 1 nm, take every 5th row. If it is at 10 nm, linearly interpolate.
  - Put the DOI or URL and the download date in the `#` header.
  - Save the raw downloads in the scratchpad only.
- [ ] **Step 2: Write failing tests** in `tests/core/test_spectral.cpp`, loading from `OPENFILM_SOURCE_DATA_DIR "/spectral"`:
  - `"equal-energy spectrum integrates to Y=1 and x≈y≈1/3"`: xy within 0.0005 of (0.33333, 0.33333).
  - `"daylight 6504 K is D65"`: xy (0.3127, 0.3290) ±0.0001.
  - `"daylight 5503 K is D55"`: xy (0.3324, 0.3474) ±0.0002.
  - `"blackbody 2856 K is Illuminant A"`: xy (0.44757, 0.40745) ±0.0005.
  - `"resample interpolates and fills outside"`: points {(400,1),(410,3)} → sample at 405 nm == 2, at 380 nm == `outside`.
  - `"load reports missing file"`: `SpectralTables::load("/nonexistent")` throws, and the message contains `"cie_1931_2deg_cmf.csv"`.
- [ ] **Step 3: Run and confirm the build fails.** Run `cmake --build build --target test_core`. Expected: `spectral.hpp` not found.
- [ ] **Step 4: Implement** `spectral.hpp/.cpp` to the interfaces above.
- [ ] **Step 5: Run and confirm the tests pass.** Run `cmake --build build && ctest --test-dir build --output-on-failure`. Expected: all pass.
- [ ] **Step 6: Commit** `feat(core): spectral grid, CIE tables, illuminants`.

---

### Task 2: Colour spaces

**Files:** Create `src/core/color.hpp`, `src/core/color.cpp`; Test `tests/core/test_color.cpp`

**Interfaces:**
- Produces:
  - `using Vec3 = std::array<double,3>; using Mat3 = std::array<Vec3,3>;` `Vec3 mul(const Mat3&, const Vec3&); Mat3 mul(const Mat3&, const Mat3&); Mat3 inverse(const Mat3&);`
  - `struct Primaries { Xy r, g, b, white; };` with constants:
    - `kDaVinciWideGamut`: r (0.8000, 0.3130), g (0.1682, 0.9877), b (0.0790, −0.1155), white D65
    - `kRec2020`: r (0.708, 0.292), g (0.170, 0.797), b (0.131, 0.046), white D65
    - `kRec709`: r (0.640, 0.330), g (0.300, 0.600), b (0.150, 0.060), white D65
    - D65 white = (0.3127, 0.3290)
  - `Mat3 rgbToXyz(const Primaries&);` (white Y = 1)
  - `Mat3 bradford(const Xyz& srcWhite, const Xyz& dstWhite);`
  - `double diDecode(double code); double diEncode(double linear);`, the DaVinci Intermediate curve:
    - constants: A = 0.0075, B = 7.0, C = 0.07329248, M = 10.44426855, LIN_CUT = 0.00262409, LOG_CUT = 0.02740668
    - encode: `lin <= LIN_CUT ? lin*M : (log2(lin + A) + B) * C`
    - decode: `code <= LOG_CUT ? code/M : exp2(code/C − B) − A`
  - `double bt1886Encode(double displayLinear);` = `pow(max(L,0), 1/2.4)`. Values above 1 are not clamped.

- [ ] **Step 1: Write failing tests:**
  - `"DI round trip"`: for 1000 values in [−0.01, 100] (log-spaced plus negatives), `diDecode(diEncode(x))` is within 1e-6·max(1,|x|).
  - `"DI reference points"`: `diEncode(0.18)` = 0.3360 ±0.001 and `diEncode(0)` = 0.
  - `"rgbToXyz maps white to D65"`: `mul(rgbToXyz(P), {1,1,1})` has xy (0.3127, 0.3290) ±1e-6 and Y = 1, for all three primaries.
  - `"Rec709 matrix matches the published one"`: row 0 is (0.4124, 0.3576, 0.1805) ±1e-4.
  - `"bradford identity and white mapping"`: `bradford(w,w)` ≈ I, and `bradford(D55,D65)` maps D55 to D65 within 1e-9.
  - `"bt1886"`: 0.11706 → 0.409 ±0.0005; −1 → 0.
- [ ] **Step 2: Run and confirm failure** (header missing).
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run and confirm pass:** `ctest --test-dir build -R core --output-on-failure`.
- [ ] **Step 5: Commit** `feat(core): colour spaces (DWG/Intermediate, Rec.2020, Rec.709, Bradford, BT.1886)`.

---

### Task 3: Gamut compression into Rec.2020

**Files:** Create `src/core/gamut.hpp/.cpp`; Test `tests/core/test_gamut.cpp`

**Interfaces:**
- Consumes: `Vec3`, `Primaries`, `rgbToXyz`, `inverse` (Task 2).
- Produces:
  - `Vec3 dwgLimitsInRec2020();` For each channel, the largest "distance" (defined below) that any DWG colour inside the DWG gamut reaches when expressed in Rec.2020. Computed from the three DWG primaries converted to Rec.2020, each limit at least 1.0001.
  - `Vec3 gamutCompress(const Vec3& rec2020Linear, double amount, const Vec3& limits);` This is the ACES Reference Gamut Compression form, applied per channel in Rec.2020.
    - `ach = max(r,g,b)`. If `ach <= 0`, return {0,0,0}.
    - `d_c = (ach − c)/ach`.
    - Threshold `t = 1 − 0.35·amount` (amount 0 → t = 1).
    - For `d_c > t`, compress with power p = 1.2 so that `limits[c]` maps to exactly 1.0:
      - scale `s = (lim − t) / pow(pow((1 − t)/(lim − t), −p) − 1, 1/p)`
      - `d' = t + s·((d−t)/s) / pow(1 + pow((d−t)/s, p), 1/p)`
    - When amount = 0 (t = 1), hard-clip: `d' = min(d, 1)`.
    - Result `c' = ach − d'·ach`. Distances are clamped to ≤ 1, so every output is ≥ 0.

- [ ] **Step 1: Write failing tests:**
  - `"in-threshold colours unchanged"`: amount 0.5; random colours with all d ≤ 0.8 come back unchanged to 1e-12.
  - `"output always inside Rec.2020"`: random DWG colours converted to Rec.2020 (including the three DWG primaries), amount ∈ {0, 0.5, 1} → all channels ≥ −1e-12.
  - `"monotonic along a saturation ramp"`: blending from grey to a DWG primary in 100 steps, the output distance is non-decreasing.
  - `"amount 0 is hard clip"`: (1, −0.5, 0.2) → (1, 0, 0.2).
  - `"amount 1 compresses more than 0.5"`: for an out-of-gamut colour, output distance at amount 1 ≤ output distance at amount 0.5.
  - `"black stays black"`: {0,0,0} → {0,0,0}.
- [ ] **Step 2: Run and confirm failure.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run and confirm pass.**
- [ ] **Step 5: Commit** `feat(core): smooth gamut compression into Rec.2020`.

---

### Task 4: RGB → spectrum (Jakob–Hanika)

**Files:**
- Create: `tools/rgb2spec/rgb2spec_opt.cpp` (vendored), `tools/rgb2spec/LICENSE`, `THIRD_PARTY.md`, `src/core/rgb2spec.hpp/.cpp`
- Modify: `CMakeLists.txt`:
  - executable `rgb2spec_opt`;
  - custom command producing `${CMAKE_BINARY_DIR}/resources/rec2020.coeff` via `rgb2spec_opt 64 <out> REC2020`, with target `openfilm_coeff`;
  - `test_core` depends on `openfilm_coeff` and gets `OPENFILM_COEFF_PATH`.
- Test: `tests/core/test_rgb2spec.cpp`

**Interfaces:**
- Consumes: `Spectrum`, `SpectralTables`, `integrate`, `wavelength` (Task 1). `Vec3`, `rgbToXyz(kRec2020)` (Task 2).
- Produces:
  - `class Rgb2Spec { public: static Rgb2Spec load(const std::string& coeffPath); std::array<double,3> coefficients(const Vec3& rgb01) const; Spectrum toSpectrum(const Vec3& rec2020Linear) const; };`
    - The coeff file format is Mitsuba's: `"SPEC"`, `uint32 res`, `float scale[res]`, `float data[3][res][res][res][3]`. `coefficients` follows Mitsuba's `rgb2spec_fetch` (largest-channel slice, `scale` lookup, trilinear interpolation).
    - The spectrum value at λ is `sigmoid(c0·λ² + c1·λ + c2)`, with `sigmoid(x) = 0.5 + x/(2·sqrt(1+x²))` and λ in nm (as in Mitsuba).
    - `toSpectrum` handles scene-linear brightness: `m = max(rgb)`. If m ≤ 0, return zeros. Otherwise fit `rgb/(2m)` and multiply the spectrum by `2m`.
    - `load` throws `std::runtime_error` on a missing file, bad magic or size mismatch.

- [ ] **Step 1: Vendor the optimiser.**
  - Download `rgb2spec_opt.cpp` and `LICENSE` from `https://github.com/mitsuba-renderer/rgb2spec` (master).
  - Keep its header and licence. Add a top comment: "Vendored for OpenFilm; changes: <list>".
  - Add a `REC2020` gamut if it is missing, using the primaries from Task 2 and D65 white.
  - Make it compile without OpenMP (guard the `#pragma omp`).
  - Write `THIRD_PARTY.md` listing rgb2spec (BSD-3, Wenzel Jakob), nlohmann/json (MIT), doctest (MIT) and OpenFX (BSD-3), each with its URL.
- [ ] **Step 2: Write failing tests** (load `OPENFILM_COEFF_PATH`; integrate under D65 via `SpectralTables`; convert XYZ to Rec.2020 with the inverse matrix):
  - `"round trip inside gamut"`: 1000 random Rec.2020 colours in [0,1]³ with all gamut distances ≤ 0.8 (seed 7). Each re-integrated colour (spectrum × D65, normalised so a flat 1 gives Y = 1) matches the input within 1e-3 absolute.
  - `"round trip full cube"`: 1000 random colours in [0,1]³ → max error ≤ 0.02.
  - `"scene-linear brightness"`: (4, 2, 1) round-trips within 4e-3 and (0.001, 0.001, 0.001) within 1e-5.
  - `"reflectance bounds"`: for inputs in [0,1]³, spectrum values lie in [0, 1+1e-9].
  - `"black"`: {0,0,0} → all zeros.
  - `"bad file"`: `load("/nonexistent")` throws.
- [ ] **Step 3: Run and confirm failure.**
- [ ] **Step 4: Implement** the CMake rule and `rgb2spec.hpp/.cpp`. Generating the table may take a while: if the 64 res build takes more than 3 minutes, record a ruling and pass `-O3` to the tool.
- [ ] **Step 5: Run and confirm pass.**
- [ ] **Step 6: Commit** `feat(core): Jakob–Hanika RGB→spectrum with build-time Rec.2020 table`.

---

### Task 5: Datasheet data (densitometry tables, 500T, 2383)

**Files:**
- Create:
  - `data/spectral/iso5_3_status_a.csv`, `data/spectral/iso5_3_status_m.csv`
  - `data/stocks/schema.json`, `data/stocks/kodak-5219-type.json`, `data/stocks/kodak-2383-type.json`
  - `tools/digitize/{requirements.txt, README.md, pdf_vectors.py, overlay.py, build_stock.py}`
  - `tools/digitize/figures/*.json`, `tools/digitize/points/*.csv`
- Review output: overlay PNGs in `build/digitize-review/`, not committed.

**Interfaces:**
- Produces the **stock JSON schema** (`schema.json`, JSON Schema draft 2020-12):
  ```
  id: string, displayName: string, type: "negative"|"print",
  densitometry: "statusM"|"statusA",
  sources: [{title, url, retrieved (YYYY-MM-DD), figures: [string]}],
  balanceIlluminantK: number            (required if negative),
  midGreyLogExposure: number            (required if negative),
  labAimDensity: [R,G,B] Status A       (required if print),
  layers: {red|green|blue: {logSensitivity: [[nm, v]...], characteristicCurve: [[logE, D]...]}},
  dyeDensity: {cyan|magenta|yellow: [[nm, v]...]},
  baseDensity: [[nm, v]...]             (optional),
  interlayer: 3×3 number array          (identity in M2)
  ```
  Arrays are sorted ascending by their first element. Spectra are at 5 nm within 380–780, and only the digitised range is stored. Characteristic curves are resampled to 0.05 log E steps.
- **Status CSV:** columns `nm,blue,green,red` holding log10 spectral products as printed (peak 5.000), with missing cells left empty. The header records that edge "slope" extrapolations from the tables were not applied (that adds negligible weight).

- [ ] **Step 1: Transcribe Status A (Table 3, p.16) and Status M (Table 4, p.17)** from IT2.18-1996.
  - Render each page at 200 dpi in the scratchpad and read the values. Cross-check against the PDF's OCR text from `page.get_text()`.
  - Every row must have a peak of exactly 5.000 per channel, and values must rise to that peak and then fall.
- [ ] **Step 2: Write the tools.**
  - `pdf_vectors.py`:
    - Takes `figures/<name>.json` = `{pdf, page, clip:[x0,y0,x1,y1], xAxis:{p0,v0,p1,v1,log:bool}, yAxis:{…}, curves:{name:{color:[r,g,b], dashes:str|null, xRange:[min,max]}}}`.
    - Uses `page.get_drawings()` inside `clip` and keeps the paths that match each curve's stroke colour and dash pattern.
    - Maps PDF coordinates to data units by linear calibration, with x and y axis tick positions taken from `page.get_text("dict")` spans.
    - Writes `points/<figure>_<curve>.csv`.
  - `overlay.py`:
    - Renders the figure clip at 300 dpi and draws the points (from vectors or hand-entered CSV) in magenta on top.
    - Writes `build/digitize-review/<figure>.png`.
  - `build_stock.py`:
    - Takes `--meta <json>` plus point CSVs.
    - Resamples (spectra to the 5 nm grid within the data range; curves to 0.05 log E), validates against `schema.json` with `jsonschema`, and writes `data/stocks/<id>.json`.
- [ ] **Step 3: Digitise 500T from the 7219 PDF** (vector paths on pp.4–5):
  - Sensitometric curves → R/G/B characteristic curves. They are Status M; `densitometry: "statusM"`.
  - Spectral sensitivity → yellow/magenta/cyan-forming layers, mapped to the blue/green/red layers.
  - Spectral dye density → cyan, magenta and yellow dyes. If the figure has a minimum-density curve, it becomes `baseDensity`.
  - `midGreyLogExposure`: the log E at the "Camera Stops 0" tick, read from the vector text positions where the two axes line up.
  - `balanceIlluminantK`: 3200.
- [ ] **Step 4: Digitise 2383 from the 2383 PDF** (raster images, pp.4–5):
  - Render each figure at 400 dpi and read points by eye: every 10 nm for spectra, and every 0.1 log E for the characteristic curves.
  - Write them to `points/*.csv` and check them with `overlay.py`. Iterate until the magenta points sit on the curves.
  - `densitometry: "statusA"`.
  - `labAimDensity`: from the LAD text on p.3 (Status A R, G, B).
- [ ] **Step 5: Validate.**
  - Run `python -I tools/digitize/build_stock.py …` for both stocks. Expected: "valid" for each.
  - Run `overlay.py` for every figure. Expected: one PNG per figure in `build/digitize-review/`.
  - Then look at each PNG yourself: points must lie on the printed curves within about one line width.
- [ ] **Step 6: Commit** `data: digitised 500T (5219-type), 2383-type and ISO 5-3 Status A/M tables`. The commit includes the tools, figure configs, points and JSON; the PDFs and PNGs are excluded.

---

### Task 6: Stock loader + densitometry

**Files:**
- Create: `src/core/stock.hpp/.cpp`, `src/core/densitometry.hpp/.cpp`
- Modify: `CMakeLists.txt`:
  - FetchContent `nlohmann_json` from `https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz`
  - link `nlohmann_json::nlohmann_json` into `openfilm_core`
- Test: `tests/core/test_stock.cpp`, `tests/core/test_densitometry.cpp`

**Interfaces:**
- Consumes: `Spectrum`, `resample`, `SpectralTables` (Task 1); stock JSON (Task 5).
- Produces:
  - `enum class StockType { Negative, Print }; enum class Densitometry { StatusM, StatusA };`
  - `struct Curve { std::vector<double> logE, density; double eval(double logE) const; double minLogE() const; double maxLogE() const; };`
    - Linear interpolation, clamped to the end values outside the range. The ends act as toe and shoulder plateaus.
    - `double inverse(double density) const;` Monotonic inverse, clamped.
  - `struct Stock`:
    - `std::string id, displayName; StockType type; Densitometry densitometry;`
    - `std::array<Spectrum,3> sensitivity;` linear (10^logSensitivity), 0 outside the data range, index 0 = red layer
    - `std::array<Curve,3> curve; std::array<Spectrum,3> dye;` index 0 = cyan
    - `Spectrum base;` zeros if absent
    - `double balanceIlluminantK = 0, midGreyLogExposure = 0; Vec3 labAimDensity{};`
  - `Stock loadStock(const std::string& jsonPath);` Throws `std::runtime_error("<path>: <what>")` on a missing field, wrong type, unsorted array, wrong type-specific field, or a curve that is not non-decreasing in density.
  - `std::vector<Stock> loadStocks(const std::string& stocksDir);` loads every `*.json` except `schema.json`, sorted by `id`.
  - `class Densitometer`:
    - `static Densitometer load(const std::string& spectralDir, Densitometry);` reads the status CSV and resamples it to the grid; the weights are `10^value`, 0 outside the range.
    - `Vec3 density(const Spectrum& spectralDensity) const;` Each channel i is `−log10( Σ wᵢ·10^(−D(λ)) / Σ wᵢ )`. Output order is R, G, B.
  - `Vec3 dyeAmountsForDensity(const Densitometer&, const std::array<Spectrum,3>& dye, const Spectrum& base, const Vec3& targetRGB);`
    - Newton iteration on the amounts c (C, M, Y).
    - The start comes from the linearised 3×3 matrix (status density of each unit dye).
    - Converges to |error| < 1e-6 within at most 20 iterations, otherwise returns the best result.
  - `Spectrum spectralDensity(const std::array<Spectrum,3>& dye, const Spectrum& base, const Vec3& amounts);` = base + Σ cⱼ·dyeⱼ.

- [ ] **Step 1: Write failing tests:**
  - **test_stock:**
    - `"loads 5219-type"`: type Negative, StatusM, midGrey finite, balance 3200, each curve has ≥ 20 points and is non-decreasing; each sensitivity peaks in the right place: blue layer 400–500 nm, green 500–600, red 600–700.
    - `"loads 2383-type"`: type Print, StatusA, labAimDensity values each in [0.8, 1.3].
    - `"rejects malformed"`: temp files missing `layers`, with an unsorted `characteristicCurve`, or with `type:"print"` but no `labAimDensity` → each throws, and the message contains the field name.
    - `"loadStocks lists both, sorted"`: ids == {"kodak-2383-type", "kodak-5219-type"}.
    - `"curve eval and inverse"`: clamping at both ends; `inverse(eval(x)) ≈ x` inside a strictly increasing region.
  - **test_densitometry:**
    - `"zero density is zero"`.
    - `"flat density reads flat"`: a constant 1.0 spectrum gives (1, 1, 1) ±1e-9 for both Status A and M.
    - `"dye amounts reproduce target"`: for 5219 dyes and targets {(0.5, 0.8, 1.0), (1.2, 1.6, 2.0), (2.5, 2.8, 3.0)}, `density(spectralDensity(c)) == target` ±1e-5.
- [ ] **Step 2: Run and confirm failure.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run and confirm pass.**
- [ ] **Step 5: Commit** `feat(core): stock loader and Status A/M densitometry`.

---

### Task 7: Negative model

**Files:** Create `src/core/negative.hpp/.cpp`; Test `tests/core/test_negative.cpp`

**Interfaces:**
- Consumes: `Rgb2Spec`, `SpectralTables`, `blackbody`, `Stock`, `Densitometer`, `dyeAmountsForDensity`, `spectralDensity`, `gamutCompress`, `dwgLimitsInRec2020`, `diDecode`, `rgbToXyz`.
- Produces:
  - `class NegativeModel`:
    - Constructor: `NegativeModel(const Stock&, const SpectralTables&, const Rgb2Spec&, const Densitometer& statusM, double gamutAmount);`
    - `Vec3 logExposure(const Vec3& diCode) const;` steps 1–4:
      1. Decode DI to linear DWG.
      2. Convert DWG to Rec.2020 and compress (`gamutAmount`).
      3. `Rgb2Spec::toSpectrum`, multiplied by `blackbody(stock.balanceIlluminantK)`.
      4. Per layer i: `Eᵢ = Σ S·sensᵢ / Σ (illuminant·sensᵢ)`, giving `logEᵢ = log10(max(Eᵢ, 1e-10)) + midGreyLogExposure − log10(0.18)`.

      Result: a flat 18 % reflectance gives exactly `midGreyLogExposure` on every layer.
    - `Vec3 density(const Vec3& logE) const;` per layer `curve[i].eval(logE[i])` (Status M R, G, B), then the `interlayer` matrix applied to (D − Dmin) + Dmin. Identity in M2.
    - `Spectrum transmittanceDensity(const Vec3& statusMDensity) const;` via `dyeAmountsForDensity` with base → spectral density.

- [ ] **Step 1: Write failing tests:**
  - `"18% grey hits mid-grey log E on all layers"`: `logExposure({diEncode(0.18)×3})` == midGreyLogExposure ±1e-4 each.
  - `"+1 stop is +0.30103"`: `diEncode(0.36)` vs `diEncode(0.18)` differ by 0.30103 ±1e-4 per layer.
  - `"black is finite and at toe"`: `logExposure({0,0,0})` is finite (≈ −10 + offset); `density(that)` == the curve's minimum density ±1e-9.
  - `"density monotonic in exposure"`: a grey ramp −8…+8 stops gives non-decreasing density per channel.
  - `"transmittance reproduces density"`: for the mid-grey density, `Densitometer(StatusM).density(transmittanceDensity(D))` == D ±1e-5.
  - `"extreme DWG primary is finite"`: DWG (0,0,1) and (1,0,0) linear encoded to DI, at amounts 0 and 1 → all outputs finite.
- [ ] **Step 2: Run and confirm failure.** **Step 3: Implement.** **Step 4: Run and confirm pass.**
- [ ] **Step 5: Commit** `feat(core): spectral negative model`.

---

### Task 8: Print + display model, auto-balance, FilmModel

**Files:** Create `src/core/print.hpp/.cpp`, `src/core/film_model.hpp/.cpp`; Test `tests/core/test_print.cpp`, `tests/core/test_film_model.cpp`

**Interfaces:**
- Consumes: `NegativeModel` (Task 7) and everything before it.
- Produces:
  - `class PrintModel`:
    - Constructor: `PrintModel(const Stock& print, const SpectralTables&, const Densitometer& statusA);`
    - `Vec3 printLogExposure(const Spectrum& negSpectralDensity, const Vec3& offsetsLogE) const;`
      - Lamp = `blackbody(3200)`.
      - Per layer j: `log10( Σ lamp·10^(−Dneg)·sensⱼ / Σ lamp·sensⱼ ) + offsetsLogE[j]`.
    - `Xyz projectedXyz(const Vec3& printLogE) const;`
      1. Print curves → Status A density.
      2. Density → dye amounts → spectral density.
      3. T = 10^(−D).
      4. × `daylight(5503)` → `integrate`, normalised so the open-gate projector light gives Y = 1.
    - `Vec3 printDensity(const Vec3& printLogE) const;` Status A, for the balance.
  - `struct FilmParams { std::string negativeId, printId; double gamutCompression = 0.5; Vec3 printerLights{0,0,0}; double printDensity = 0; };`
    - `uint64_t hash() const;` covers every field.
  - `struct FilmResources`:
    - Fields: `SpectralTables tables; Rgb2Spec rgb2spec; Densitometer statusM, statusA; std::vector<Stock> stocks;`
    - `static std::shared_ptr<const FilmResources> load(const std::string& resourcesDir);` The layout is `<dir>/data/spectral`, `<dir>/data/stocks` and `<dir>/rec2020.coeff`.
    - Overload for tests: `load(dataDir, coeffPath)`.
    - `const Stock& stock(const std::string& id) const;` throws on an unknown id.
  - `class FilmModel`:
    - Constructor: `FilmModel(std::shared_ptr<const FilmResources>, const FilmParams&);` builds the NegativeModel and PrintModel, then solves the balance.
    - `Vec3 negativeLogExposure(const Vec3& diCode) const; Vec3 negativeDensity(const Vec3& logE) const;`
    - `Vec3 printOutput(const Vec3& negStatusMDensity) const;`
      - `offsets = balance + 0.025·(printerLights + printDensity)` → `projectedXyz`.
      - Bradford (D55 white → D65), then the XYZ→Rec.709 matrix.
      - × `anchorScale`, then `bt1886Encode` per channel.
    - `Vec3 evaluate(const Vec3& diCode, double exposureStops) const;` = `printOutput(negativeDensity(negativeLogExposure(di) + stops·log10(2)))`
    - `const Vec3& balance() const; double anchorScale() const;`
    - **Balance solve:**
      - The grey density `Dg` is the negative density at mid-grey.
      - The unknowns are three log E offsets b.
      - The residuals are `(u′−u′_D65, v′−v′_D65, (printDensity_G − labAim_G)/10)` for the grey at zero user offsets.
      - Start where each channel's print density equals `labAim` via the curve inverse. Use Newton with a finite-difference Jacobian (h = 1e-4) and damping 0.5 on divergence, stopping at |r| < 1e-7 or after 50 iterations. If it hasn't converged, throw `std::runtime_error("print balance did not converge")`.
      - **Anchor:** `anchorScale = 0.11706 / Y_adapted(grey at balance)`.

- [ ] **Step 1: Write failing tests** (resources via the test overload, from `OPENFILM_SOURCE_DATA_DIR` and `OPENFILM_COEFF_PATH`):
  - `"grey prints neutral at code 0.409"`: `evaluate({diEncode(0.18)×3}, 0)` gives all three channels within 0.005 of 0.409, and the u′v′ of the adapted XYZ is within 0.001 of D65.
  - `"printer point adds 0.025 log E"`: comparing printerLights.r = 1 with 0, the red print log E differs by 0.025 ±1e-9.
  - `"print density moves all three"`: printDensity = +1 equals printerLights = (1,1,1).
  - `"grey ramp neutral and monotonic"`:
    - for −2…+2 stops in 0.25 steps, u′v′ distance from D65 is < 0.003;
    - for −6…+6 stops, each output channel is non-decreasing as exposure goes up.
  - `"black is finite"`: input code 0 → finite, ≥ 0, and darker than −6 stops grey.
  - `"DWG primaries finite"`: DWG (1,0,0), (0,1,0) and (0,0,1) at gamut amount 0 and 1 → finite, ≥ 0.
  - `"unknown stock id throws"`.
- [ ] **Step 2: Run and confirm failure.** **Step 3: Implement.** **Step 4: Run and confirm pass.**
- [ ] **Step 5: Commit** `feat(core): print/display model with auto-balanced printer lights`.

---

### Task 9: LUT baking, cache and the film CPU renderer

**Files:**
- Create: `src/core/lut.hpp/.cpp`, `src/core/film_renderer.hpp/.cpp`
- Modify: `src/core/cpu_renderer.hpp/.cpp` (rename `renderCpu` → `renderExposureCpu`), `tests/core/test_cpu_renderer.cpp`, `tests/gpu/test_gpu_vs_cpu.mm`, `src/ofx/openfilm_plugin.cpp` (call sites)
- Test: `tests/core/test_lut.cpp`, `tests/core/test_film_renderer.cpp`

**Interfaces:**
- Consumes: `FilmModel`, `FilmResources`, `FilmParams` (Task 8); image views (M1).
- Produces:
  - `class Lut3D`:
    - Constructor: `Lut3D(int size, Vec3 domainMin, Vec3 domainMax);`
    - `void bake(const std::function<Vec3(const Vec3&)>& f);` multithreaded over slices.
    - `std::array<float,3> lookup(float, float, float) const;` tetrahedral interpolation, inputs clamped to the domain.
    - `const std::vector<float>& data() const;` RGB interleaved, red index fastest. This is the layout M3 uploads.
  - `struct Curve1D { float min, max; std::vector<float> values; float eval(float) const; };` 4096 samples, clamped.
  - `struct BakedFilm { Lut3D lutE; std::array<Curve1D,3> negCurves; Lut3D lutP; };`
    - `lutE` covers DI [0,1]³ → log E.
    - `negCurves` covers log E over [curve.minLogE − 1, curve.maxLogE + 1] → Status M density.
    - `lutP` covers density [0,4]³ → output code.
  - `std::shared_ptr<const BakedFilm> bakeFilm(const FilmModel&, int lutSize = 65);`
  - `class FilmCache`:
    - Constructor: `explicit FilmCache(std::shared_ptr<const FilmResources>);`
    - `std::shared_ptr<const BakedFilm> get(const FilmParams&);` holds up to 4 entries in LRU order, keyed by `FilmParams::hash()`, protected by a mutex. Baking happens while holding the lock.
  - `void renderFilmCpu(const ConstImageView& src, const ImageView& dst, const RenderWindow& win, const BakedFilm&, double exposureStops);`
    - Uses the same window, bounds and padding semantics as M1.
    - Alpha is copied.
    - Out-of-bounds source pixels become (0,0,0,0).
    - Non-finite RGB is treated as 0 before lookup.
  - `renderExposureCpu` keeps the M1 behaviour exactly; only the name changes.

- [ ] **Step 1: Rename** `renderCpu` → `renderExposureCpu` at every call site. Run `cmake --build build && ctest --test-dir build`. Expected: all 4 existing suites still pass, the rename being the only change.
- [ ] **Step 2: Write failing tests:**
  - **test_lut:**
    - `"identity LUT is exact at nodes and close between"`: a 17³ identity → node lookups exact, random lookups within 1e-6.
    - `"tetrahedral reproduces linear functions exactly"`: f(x) = A·x + b → error < 1e-5.
    - `"inputs clamp to domain"`: lookup(−1, 2, 0.5) == lookup(0, 1, 0.5).
  - **test_film_renderer:**
    - `"baked matches direct within ΔE2000 0.5"`: 2000 random DI codes in [0,1]³ (seed 11) plus exposures in [−3, 3]. Convert the BT.1886-decoded outputs of `evaluate` vs `renderFilmCpu` to CIELAB (D65) and compute ΔE2000 (implement the formula in the test helper); the maximum is < 0.5.
    - `"exposure needs no rebake"`: the same `BakedFilm` pointer is returned for params that differ only in exposure. Exposure isn't in `FilmParams`, so check `cache.get(p) == cache.get(p)` and that rendering at different exposures changes the output.
    - `"window, bounds and padding respected"`: the M1 offset-bounds and padded-row geometry case through `renderFilmCpu` → pixels outside the window untouched, alpha copied, no NaN from the padding.
    - `"out-of-domain and non-finite inputs"`: codes −0.5 and 1.7 and NaN, +Inf → finite outputs; the −0.5 output equals the 0 output.
    - `"concurrent cache use"`: 8 threads × 20 iterations, each fetching params with printer lights r ∈ {−5, 0, 5} by thread index and rendering a 16×16 grey frame; every result equals the single-threaded reference for its params.
- [ ] **Step 3: Run and confirm failure.** **Step 4: Implement.** **Step 5: Run and confirm pass.**
- [ ] **Step 6: Commit** `feat(core): baked film LUTs, cache and CPU film renderer`.

---

### Task 10: Plugin wiring, bundle resources, docs

**Files:**
- Modify:
  - `src/ofx/openfilm_plugin.cpp`
  - `CMakeLists.txt` and `cmake/OpenFilmBundle.cmake`: an `openfilm_resources` target copies `data/spectral/*.csv`, `data/stocks/*.json` and `rec2020.coeff` into `${CMAKE_BINARY_DIR}/resources/`, and the bundle post-build copies that tree into `Contents/Resources/`, with a stamp file in `LINK_DEPENDS`.
  - `tests/ofx/test_bundle_loads.cpp`, `README.md`
- Create: `docs/film-model.md`

**Interfaces:**
- Consumes: `FilmResources::load`, `FilmCache`, `FilmParams`, `renderFilmCpu` (Task 9).
- Produces: OFX parameters (names are API, so later milestones keep them):
  - Group `colour` "Colour": choice `inputSpace` ["DaVinci Wide Gamut / Intermediate"], choice `outputSpace` ["Rec.709 Gamma 2.4"], double `gamutCompression` "Gamut compression" 0…1, default 0.5.
  - Group `negative` "Negative": choice `negativeStock` (display names of the negative stocks, in `id` order), double `exposure` (unchanged).
  - Group `print` "Print": choice `printStock`, doubles `printerLightR`/`printerLightG`/`printerLightB` "Printer light R/G/B" −20…20, increment 1, default 0, double `printDensity` "Print density" −20…20, default 0.

- [ ] **Step 1: Write failing tests** (additions to `test_bundle_loads.cpp`):
  - `"bundle ships film resources"`: `Contents/Resources/data/stocks/kodak-5219-type.json`, `…/kodak-2383-type.json`, `…/data/spectral/iso5_3_status_m.csv` and `Contents/Resources/rec2020.coeff` all exist.
  - `"film resources load from the bundle"`: `FilmResources::load(<bundle>/Contents/Resources)` succeeds and lists 2 stocks.
- [ ] **Step 2: Run and confirm failure.** Expected: the resources are missing from the bundle.
- [ ] **Step 3: Implement the plugin changes:**
  - `setSupportsMetalRender(false)`, with a comment that M3 re-enables it.
  - Process-wide `FilmResources` and a `FilmCache` (function-local statics). `describeInContext` loads resources to fill the stock choices. If that fails, describe a single choice entry "(stock data missing)", and render shows the persistent error `"OpenFilm: film data failed to load: <what>"` and fails with `kOfxStatFailed`.
  - `render` builds `FilmParams` from the params at `args.time`, gets the bake from the cache, and calls `renderFilmCpu`. On success it calls `clearPersistentMessage()` (fixes the M1 deferred minor).
- [ ] **Step 4: Write `docs/film-model.md`** for colorists:
  - each pipeline stage in film terms;
  - what each control does, with ranges: printer points as a lab timer knows them, 0 = balanced, and more light in a channel puts more of the complementary dye in the print;
  - why playback is slow in M2;
  - how the stock data was obtained, with citations;
  - known approximations: no interlayer effects yet, Status-table edge slopes ignored, D55 projector, 3200 K lamp.

  Link the doc from `README.md`.
- [ ] **Step 5: Run and confirm pass:** `cmake --build build && ctest --test-dir build --output-on-failure`. Expected: every suite passes, and `bundle_signature` passes.
- [ ] **Step 6: Manual check in Resolve** (the user runs this):
  1. `scripts/install.sh` and restart Resolve.
  2. On a DWG/Intermediate timeline, the OpenFilm node shows a film print look. A grey card at 0 stops reads about 41 %.
  3. Printer light R +5 adds red printing light, which forms more cyan dye in the print, so the image shifts toward cyan, exactly like a lab timer's trim. Exposure behaves in stops.
  4. Overlay PNGs in `build/digitize-review/` look right against the PDFs.
- [ ] **Step 7: Commit** `feat(ofx): film colour model in the plugin (CPU), bundle resources, film-model docs`.

---

## Milestone exit criteria
- `ctest --test-dir build` passes: core (all unit files), gpu (M1 exposure kernel), bundle, bundle_signature.
- The user has reviewed the overlays and the Resolve look.
- Inputs for M3: `Lut3D::data()` layout, `Curve1D`, `BakedFilm` and `renderFilmCpu` as the reference for the Metal port.
