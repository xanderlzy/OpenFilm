# OpenFilm — Design Spec (2026-11-08)

## Context
Build an **open-source (MIT) OpenFX plugin for DaVinci Resolve** that emulates motion-picture
film (in the spirit of FilmBox Pro / Dehancer) by simulating the photochemical behaviour of
celluloid — a spectral negative → print model with physically-motivated halation, bloom and grain —
rather than applying a look LUT. The project folder is empty (greenfield). The user is primarily a
colorist, so code and docs must explain the C++/OFX/Metal plumbing clearly.

### Decisions captured
| Topic | Decision |
|---|---|
| Purpose | Open-source, MIT licence |
| Platform | macOS only; Metal GPU path + CPU path (shared C++ math, reference + fallback) |
| Fidelity | Spectral / physical model |
| v1 effects | Negative→print pipeline, halation + bloom, grain (gate weave/flicker/dust/vignette later) |
| Colour mgmt | Selectable input (DWG/Intermediate, ACES AP0, ACEScct, ARRI LogC3, LogC4, Rec.709) and output (Rec.709 display, or back to a log space) |
| Stock data | Hand-digitised from public datasheets → JSON, sources cited, generic names |
| Name | **OpenFilm** |
| Stocks v1 | 2 negatives (500T 5219-type, 250D 5207-type) + 1 print (2383-type) |
| Output v1 | Projected print only. Negative-scan and print-scan modes later — stage interface designed for them now |
| Processing | Push/pull development (parametric: contrast, shadow speed, fog, grain; documented as approximation) |
| Workflow | One node at the end of the tree (replaces output transform); grade before it in log |
| Shooting | Scene illuminant (colour temp) + camera filter (none/85/80A); unfiltered mismatch shows as on real film; printer lights correct |
| Halation | Physical (red-layer scatter), strength + radius + optional hue nudge |
| Grain | Physically driven (density, stock, format, push); knobs only scale amount/size. Full: stochastic silver-grain; Preview: fitted noise |
| Highlights | Stock shoulder roll-off + a **highlight latitude** knob |
| Sharpness | Per-layer MTF fitted to datasheets in v1; DIR adjacency effects later |
| Performance | User-selectable **Quality: Preview / Full**; target 24 fps HD Preview on M1/M2 (4K best-effort) |
| Resolve | Studio only |
| Success bar | **Model fidelity** vs datasheets, proven by an automated datasheet report |
| Reference | Datasheets only (no real film scans) |
| UI | Stock pickers + physical knobs; no presets/expert mode in v1 |
| Architecture | **A — baked spectral LUTs on CPU + GPU spatial passes** |

## Architecture

### Per-frame pipeline (one plugin node)
```
Input image (chosen input space)
 1. Linearize → scene-linear in a working gamut            [GPU, analytic]
 2. LUT_E (3D, baked): scene RGB → 3 layer log-exposures    [GPU lookup]
      (spectral upsampling × scene illuminant × camera filter × negative
       spectral sensitivities × exposure stops)
 3. Bloom (emulsion/lens diffusion) + Halation             [GPU, in linear exposure]
      halation = blurred highlight exposure re-entering mainly the red-sensitive layer
    Per-layer MTF blur (fitted to datasheet MTF)            [GPU, in exposure]
 4. Negative H&D curves (1D per layer, modified by push/pull and highlight
    latitude) + interlayer (DIR) matrix → CMY densities     [GPU]
 5. Grain on per-layer densities (Preview: noise; Full: stochastic grains) [GPU]
 6. LUT_P (3D, baked): negative densities → print exposure (printer lights) →
      print densities → projected spectrum → XYZ → output space            [GPU lookup]
Output image
```
Why this split: steps 2 and 6 depend only on each pixel's own value, so they bake *exactly* into
LUTs (the expensive spectral integration runs on CPU only when stock/knobs change). Steps 3–5
depend on neighbouring pixels and must happen in the right physical domain (light scatter adds
*exposure*; grain lives in *density*), so they are GPU passes between the LUTs.

Quality setting controls: LUT size (33³ vs 65³), blur quality, grain model.

### Physical model (CPU, `src/core/`)
- Spectral grid 380–780 nm @ 5 nm; CIE 1931 CMFs; illuminants (camera-side scene white, printer
  lamp, projector xenon).
- RGB→spectrum: Jakob–Hanika (2019) sigmoid upsampling, coefficient table precomputed offline
  for the working gamut (handles wide-gamut input); scene-linear brightness scaled out.
- Negative: layer exposure Eᵢ = ∫ S(λ)·sᵢ(λ) dλ · 2^stops → densities via per-layer characteristic
  curve + 3×3 interlayer matrix; spectral transmittance T(λ)=10^-(Σ Dᵢ·dyeᵢ(λ)+base(λ)).
- Print: print exposure ∫ lamp(λ)·T(λ)·s_print_j(λ) with printer-light offsets
  (1 point = 0.025 log E) → print characteristic curves → print dye spectrum → viewed under
  projector illuminant → XYZ → output space. Default printer lights auto-balanced so an 18%
  grey card at 0 stops prints neutral at nominal density.
- Grain parameters from datasheet RMS granularity + nominal grain size; scaled to pixels via
  **film format** (35mm / Super16 / 65mm) and image width.

### OFX / Metal plumbing (`src/ofx/`, `src/metal/`)
- OpenFX SDK as git submodule (AcademySoftwareFoundation/openfx), using its C++ Support library.
- Declares Metal render support; Resolve passes `MTLCommandQueue` + `MTLBuffer`s. LUTs uploaded as
  buffers, tetrahedral interpolation done in-kernel. Kernels compiled with `xcrun metal` into a
  `.metallib` inside the bundle's Resources.
- CPU path runs the same pipeline in C++ (multithreaded) — also the reference for GPU tests.
- LUT cache keyed by hash of (stocks, colour spaces, exposure, printer lights, quality) so
  scrubbing doesn't re-bake.
- Grain seeded by frame time → temporally varying, render-deterministic.

Step 6 is an **output stage** interface (`OutputStage::bake(negDensity→out)`); v1 implements
`ProjectedPrint`. `NegativeScan` (Status M → Cineon log) and `PrintScan` are later drop-ins.

### Parameters (UI)
Input space · Output space · Film format · Quality ·
**Shooting**: scene illuminant (K), camera filter ·
**Negative**: stock, exposure (stops), push/pull (−1…+2), highlight latitude ·
**Print**: stock, printer lights R/G/B (points), print density ·
**Halation**: enable, amount, radius, hue · **Bloom**: enable, amount, radius ·
**Sharpness (MTF)**: enable · **Grain**: enable, amount, size multiplier.

### Stock data (`data/stocks/*.json`)
Per stock: id, display name (generic, e.g. "Tungsten 500T"), type (negative/print), spectral
log-sensitivities per layer, characteristic curves (log E→D) per layer, spectral dye densities
(+ base/mask), interlayer matrix, RMS granularity, grain size, halation params, source citations.
`tools/` holds Python scripts that convert WebPlotDigitizer CSV exports to resampled JSON, generate
the Jakob–Hanika table, and plot curves for sanity checks. A JSON schema validates files.

### Repository layout
```
CMakeLists.txt  LICENSE (MIT)  README.md
external/openfx/            (submodule)
src/core/                   spectral, color_spaces, stock, negative, print, lut_baker, grain_model
src/ofx/                    plugin entry, params, render dispatch (CPU/Metal), LUT cache
src/metal/                  *.metal kernels (linearize, lut3d, blur, halation, hd_curves, grain)
data/stocks/  data/spectral/  (CMFs, illuminants, upsampling table)
tools/                      Python digitizing/plotting/table generators
tests/                      doctest unit tests + Metal-vs-CPU comparison harness
docs/                       colorist-oriented: how film works here, build/install, adding a stock
```

### Testing
- **Unit (core, doctest):** spectral integration of flat spectrum, colour-space round-trips,
  monotonic H&D curves, neutral axis (grey ramp stays neutral within ΔE threshold with balanced
  printer lights), +1 printer point shifts density by expected amount, LUT vs direct eval error.
- **Datasheet fidelity report (primary success bar):** `tools/fidelity_report.py` drives the core
  model via a small CLI, re-derives each datasheet plot (characteristic curves, spectral
  sensitivity, dye density, MTF, RMS granularity, 18% grey neutrality), overlays it on the digitised
  data, writes an HTML report with error metrics, and fails CI if errors exceed thresholds.
- **GPU vs CPU harness:** standalone executable runs each Metal kernel on test images and compares
  against the CPU path (max error threshold).
- **In Resolve:** install bundle, apply to a ColorChecker / grey-ramp / highlight test clip; check
  playback fps in Preview, render in Full; compare stock pairs.

## Milestones (each ends with a working, testable plugin)
1. **Scaffold** — CMake, OFX submodule, passthrough plugin with CPU + Metal paths, visible in Resolve.
2. **Core colour model (CPU)** — colour spaces, spectral upsampling, stock loader, negative + print
   model, LUT baking; one provisional stock pair; unit tests.
3. **Metal colour path** — LUT/curve kernels, LUT cache, GPU-vs-CPU harness.
4. **Halation, bloom, MTF** — CPU + Metal, film-format scaling.
5. **Grain** — Preview noise model, then Full stochastic model; temporal seeding.
6. **Processing & shooting** — push/pull, highlight latitude, scene illuminant + filters.
7. **Stock data + fidelity** — digitise 500T, 250D, 2383-type; fidelity report in CI; auto-balance; docs.


## Verification (end-to-end)
- `cmake --build` produces `OpenFilm.ofx.bundle`; copy to `/Library/OFX/Plugins`; restart Resolve.
- Plugin appears under OpenFX; renders identically (within tolerance) on CPU and Metal.
- `ctest` passes core unit tests and GPU-vs-CPU harness.
- Visual check on grey ramp + ColorChecker + specular highlight clip for each stock pair.
