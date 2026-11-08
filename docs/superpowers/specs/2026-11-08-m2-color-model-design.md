# OpenFilm Milestone 2 — Core Colour Model (CPU): Design Spec (2026-11-08)

Parent spec: `2026-11-08-film-emulator-design.md`. This document refines Milestone 2 and records
the decisions made for it. Where they differ, this document wins for M2.

## Goal
OpenFilm renders a real spectral film look in Resolve: camera negative (500T, 5219-type) printed
to 2383-type print stock and viewed on a Rec.709 gamma 2.4 display. Rendering runs on the CPU only
in this milestone; Milestone 3 ports it to Metal.

## Decisions
| Topic | Decision |
|---|---|
| Visibility | Wired into the plugin. Metal render support is **declared off** during M2 (CPU-only, slow playback). |
| Stocks | Real datasheet data for **500T (5219-type)** and **2383-type print**. 250D follows in M7. |
| Digitising | Claude digitises the curves. It extracts vector paths from the PDF where possible and otherwise reads from page images, then produces overlay plots for the user to review. |
| Datasheets | Downloaded from Kodak's website. URLs are cited in each stock JSON. PDFs are **not** committed. |
| Input space | DaVinci Wide Gamut / DaVinci Intermediate only (choice param with one entry). |
| Output | Rec.709 primaries, BT.1886 gamma 2.4 encoding (choice param with one entry). |
| Extreme colours | Smooth gamut compression into Rec.2020, with a user **amount** knob. |
| Printer lights | R/G/B **offsets from the auto-balanced print** (0 = neutral). 1 point = 0.025 log E. **Print density** = master offset applied to all three. |
| Display mapping | **Mid-grey anchored**: an 18% grey card at 0 stops renders at Rec.709 code value 0.409 (what a standard Rec.709 OETF gives 0.18). |
| White point | Projector white is chromatically adapted (Bradford) to D65. |
| Stock names | Generic + type code: "Tungsten 500T (5219-type)", "Print 2383-type". |
| Scene light | The stock's balance illuminant (3200 K blackbody for 500T). User control arrives in M6. |
| Interlayer effects | Identity matrix in M2 (datasheets don't publish it); tuned in M7. |

## Pipeline
```
DaVinci Intermediate (input)
 1. decode → linear DWG
 2. DWG → Rec.2020 linear; gamut-compress (amount knob)             ┐
 3. Jakob–Hanika RGB→spectrum (Rec.2020 table) × scene illuminant    │ baked: LUT_E (3D)
 4. ∫ × negative spectral sensitivities → 3 layer log exposures      ┘
 5. + exposureStops · log10(2)                    (exact, not baked)
 6. negative characteristic curves (1D per layer) → Status M densities
 7. densities → dye amounts → spectral transmittance                ┐
 8. printer lamp × T(λ) × print sensitivities → print log exposures  │
    + balance + printer-light offsets                                │ baked: LUT_P (3D)
 9. print characteristic curves → print densities → dye amounts      │
    → print spectral transmittance × projector illuminant → XYZ      │
10. Bradford adapt projector white → D65; mid-grey anchor scale      │
    → Rec.709 linear → BT.1886 (gamma 2.4) encode                   ┘
```
- **LUT_E** domain: DaVinci Intermediate code values, 65³. Output: 3 log exposures. Depends on stock and compression amount.
- **LUT_P** domain: negative Status M densities in [0, 4], 65³. Depends on both stocks, the printer lights and print density.
- Both LUTs are cached by a hash of their inputs. Exposure never triggers a re-bake.
- Lookups use tetrahedral interpolation. Steps 5–6 run per pixel.
- **Auto-balance:** for a stock pair, solve the 3 printer-light balance values (Newton / damped least squares) so that an 18% grey at 0 stops:
  - prints neutral, i.e. its adapted chromaticity is D65;
  - has its print green Status A density at the lab aim, which is stored in the print stock JSON.

  The result is cached per stock pair.
- **Densities → dye amounts:** solve the 3×3 system relating unit-dye Status M (negative) or Status A (print) densities to measured densities. Each stock JSON provides its dye spectra normalised per the datasheet; the matrix is computed at load time.

## Data
- `data/spectral/`:
  - CIE 1931 2° CMFs and CIE D65 / D55 SPDs, both from CIE public tables.
  - ISO 5-3 Status M and Status A spectral products.
  - The source of each table is noted in a header comment.
- Blackbody SPDs (scene and printer lamp) are computed from Planck's law.
- Spectral grid: 380–780 nm at 5 nm (81 samples). Data on other grids is resampled with linear interpolation.
- `data/stocks/<id>.json`, one per stock. The JSON Schema lives in `data/stocks/schema.json`. Fields:
  - `id`, `displayName`, `type` (negative|print)
  - `sources`: an array of {title, url, retrieved}
  - `balanceIlluminant` (negative), `midGreyLogExposure` (negative), `labAimDensity` (print)
  - per layer: `spectralSensitivity` (log sensitivity vs nm) and `characteristicCurve` (log E → density)
  - `dyeDensity` (C, M, Y spectra) and `baseDensity` (spectrum)
  - `interlayer` (3×3, identity in M2)
- `tools/digitize/`:
  - Python scripts (run in a local venv) that pull vector paths from the PDFs or take manual point lists.
  - They map those to data units using the axis calibration, resample, and write the JSON.
  - They plot overlays (PNG) for review.
- RGB→spectrum table: a C++ tool `tools/rgb2spec_opt.cpp` is built and run at build time. It is a port of Mitsuba's rgb2spec optimiser (BSD-3, credited in the file and `THIRD_PARTY.md`). It writes `build/rec2020.coeff`, which is copied into the bundle Resources.

## Plugin (M2)
- Parameters, in groups:
  - **Colour**: Input space, Output, Gamut compression (0–1, default 0.5).
  - **Negative**: Stock, Exposure.
  - **Print**: Stock, Printer lights R/G/B (−20…+20 points, default 0), Print density (−20…+20, default 0).
- `setSupportsMetalRender(false)` for M2. The Metal exposure kernel and its tests stay in the repo but are unused by the plugin until M3.
- The stock JSON, spectral tables and coeff table are loaded from bundle Resources once per process.

  If loading fails, show a persistent error message and fail the render.

## Testing
- **Spectral:**
  - an equal-energy spectrum integrates to XYZ white;
  - D65 gives xy (0.3127, 0.3290) ±1e-4;
  - Planck 6504 K gives xy near D65 ±0.003.
- **Colour spaces:**
  - DaVinci Intermediate round-trips (1e-6), and published reference points hold: 0.0 lin → 0.0 code (per the BMD whitepaper formula) and 0.18 lin → 0.336 code ±0.001.
  - DWG→XYZ: the matrix rows sum to the D65 white.
- **RGB→spectrum:** for 1000 random in-gamut Rec.2020 colours, upsampling then re-integrating gives back the same colour within 1e-3, and spectra stay within [0, 1] reflectance.
- **Gamut compression:**
  - colours inside the threshold are unchanged;
  - the output is always inside Rec.2020;
  - it is monotonic along a saturation ramp;
  - amount = 0 means hard clip and amount = 1 means maximum compression.
- **Negative:**
  - an 18% grey at 0 stops has log E equal to `midGreyLogExposure` on each layer;
  - +1 stop gives +0.30103;
  - densities increase monotonically with exposure.
- **Print / auto-balance:**
  - an 18% grey at 0 stops renders neutral (|u′v′ − D65| < 0.001) at code value 0.409 ±0.005;
  - +1 red printer point adds +0.025 to red print log E;
  - the grey ramp −6…+6 stops is neutral within 0.003 u′v′ through the mid-range (−2…+2) and monotonic everywhere.
- **LUTs:** for 2000 random inputs, the baked LUT result vs direct evaluation has max ΔE2000 < 0.5.
- **Data:** each stock JSON validates against the schema, curves are monotonic where physics requires it, and the loader rejects malformed files with a clear message.
- **Digitising review:** overlay PNGs for every digitised plot, which the user reviews against the PDFs before M2 is closed.
- **In Resolve (user):** the film look shows on a DWG/Intermediate timeline; printer lights and exposure behave like a lab timer would expect.
