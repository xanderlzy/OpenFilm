# M3 — Metal colour path: design

Milestone 3 of [the OpenFilm design](2026-11-08-film-emulator-design.md). Builds on M2
([design](2026-11-08-m2-color-model-design.md)), which the user verified in Resolve on 2026-11-09.

## Goal

Run the M2 film look on the GPU (Metal) so it plays back in real time in Resolve, with the look
unchanged. Printer lights and Print Density become as interactive as Exposure: no re-bake.

## Decisions (agreed with the user)

| Topic | Decision |
|---|---|
| Hardware target | M4 Pro MacBook (the user's machine) |
| Speed | UHD (3840×2160) at ≥ 24 fps in Resolve; the colour path ≤ 5 ms per UHD frame on the GPU, leaving the rest of the 41 ms budget for M4/M5 |
| Printer lights / Print Density | Per-pixel offsets, never re-bake (option A) |
| Stock change | The render waits for the ~1.4 s bake, then shows the correct look; the last 4 bakes stay cached (option A) |
| Quality control | Not in M3 — arrives with M4/M5 where it changes blur/grain cost. Colour always uses full-size tables |
| GPU method | One fused kernel, tables in plain `MTLBuffer`s, the same tetrahedral interpolation as the CPU (no hardware texture filtering) |
| Look | Unchanged from M2 within the tolerances below; the CPU path stays the reference and the CPU-render fallback |

Out of scope: halation, bloom, MTF (M4); grain (M5); push/pull, scene illuminant (M6); new stocks (M7).

## Why the print table can be split exactly

Today `lutP` maps negative density straight to display code with the printer offsets baked in.
In the model (`FilmModel::printOutput`) the offsets — auto-balance + 0.025 log E × (printer points +
Print Density) — are only *added per channel to the print log exposure*
(`PrintModel::printLogExposure`). Everything after that is a function of the three print log
exposures alone, and the mid-grey anchor (`anchorScale_`) depends only on the auto-balance, not on
the user's points. So the chain splits with no approximation beyond table interpolation:

```
DI code ─[lutE 3D]→ log10(E+eps) ─(× 2^Exposure)→ log E ─[3 neg curves 1D]→ neg density (Status M)
  ─[lutPE 3D]→ print log E (no offsets) ─(+ offsets)→ ─[3 print curves 1D]→ print density (Status A)
  ─[lutD 3D]→ Rec.709 / BT.1886 display code
```

## Components

### Core (`src/core/`, CPU)

- **`PrintModel`**: expose the pieces the split needs — print log exposure with zero offsets (already
  `printLogExposure(spec, {0,0,0})`), print density from print log E (already `printDensity`), and
  display code from print density (new: the projected-XYZ → D65 → Rec.709 → anchor → BT.1886 part of
  `FilmModel::printOutput`, refactored so the direct model and the bake call the same code).
- **`FilmModel`**: `printOutput` keeps its signature and stays the direct reference.
  `printerOffsets()` stays; the bake no longer bakes it in.
- **`BakedFilm`** becomes:
  - `lutE` — unchanged (113³, t²-spaced, log10(E + 1e-3)).
  - `negCurves[3]` — unchanged.
  - `lutPE` — negative density (per-channel reachable domain, as `lutP` today) → print log E without offsets.
  - `printCurves[3]` — print log E → print Status A density. 4096-sample `Curve1D`. Domain: the print
    stock's curve range ±1 log E, matching `negCurves`.
  - `lutD` — print density over [print D-min, print D-max] per channel → display code.
  - `balance` — the auto-balance offsets (so a render needs no `FilmModel`).
  - Starting sizes: `lutPE` 97³, `lutD` 97³. Raise either if the accuracy test needs it.
- **`FilmParams::hash()`** covers only the two stock ids and gamut compression. Printer points and
  Print Density move to per-render values, next to exposure.
- **`renderFilmCpu`** takes `exposureStops` and a `Vec3 printOffsets` (the full log-E offsets:
  balance + 0.025 × (points + density)) and runs the chain above.

### GPU (`src/metal/`)

- **`film.metal`** — one kernel, `film_kernel`, one thread per pixel, running exactly the chain
  above in 32-bit float: tetrahedral `Lut3D` lookup (same node spacing, including t² for `lutE`,
  same clamping), linear `Curve1D` lookup, the same NaN/inf → 0 input guard, alpha passed through.
  Replaces `exposure.metal`.
- **`shader_types.h`** — `OFFilmUniforms`: image geometries, window, gain, offsets[3], and each
  table's size and domain min/max.
- **`MetalRenderer`** — `render(…, const GpuFilm&, gain, offsets)`. A **`GpuFilm`** holds the
  tables as `MTLBuffer`s, uploaded once per bake; it is created lazily the first time a bake is used
  on the GPU and lives as long as the `BakedFilm` it came from (kept in the cache entry), so
  frames reuse it. Encoding stays thread-safe; Resolve may render frames concurrently.

### OFX (`src/ofx/openfilm_plugin.cpp`)

- `setSupportsMetalRender(true)`.
- Metal render: get the cached bake (waits for it on a stock change), get or create its `GpuFilm`,
  encode on `args.pMetalCmdQ`, return. No CPU round-trip.
- Errors (missing metallib, failed upload, encode failure) show in the node through the existing
  `showError()` wrapper and fail that render — never a silently wrong frame.
- CPU render (host asks for CPU) uses `renderFilmCpu` with the same per-render offsets.
- Any new host-suite call is wrapped like the persistent-message calls (Resolve may not support it).

## Testing

| Test | Pass bar |
|---|---|
| Split bake vs direct model (`FilmModel::evaluate`), 2000 random DI codes × exposure −4…+4 stops × printer points −12…+12 | ≤ 0.4 ΔE2000, same metric and sample scheme as M2's LUT test |
| Printer lights: offsets applied per pixel equal the old baked-in result | within the same tolerance; R +5 still shifts toward cyan |
| Cache: changing points or Print Density never re-bakes | bake count unchanged |
| GPU vs CPU (`test_gpu`): grey ramp, saturated primaries, ±8 stops, points ±20, NaN/inf/negative pixels, a window smaller than the image, odd widths, rowBytes padding | max abs difference ≤ 1e-3 code value per channel; alpha bit-exact |
| Speed (`test_gpu` benchmark): UHD frame, tables already uploaded, median of 50 | ≤ 5 ms on the M4 Pro |
| Bundle tests | still pass; signature still valid |
| In Resolve (user) | UHD real-time playback; printer lights instant; stock change waits ~1.4 s then correct; M2 colour checks still pass; GPU and CPU-mode renders match by eye |

A first in-Resolve check happens as soon as the kernel renders the M2 look, before the speed work —
M2 showed host behaviour can only be confirmed in Resolve.

## Risks

- **Three 3D tables compound interpolation error.** Mitigation: the accuracy test above; grow the
  table sizes (memory is not a constraint: 113³ + 2 × 97³ RGB float nodes ≈ 39 MB per bake).
- **Resolve's Metal buffers** may use a row stride or window that differs from the CPU path.
  Mitigation: the GPU test covers padded rows and partial windows; early in-Resolve check.
- **float vs double**: the CPU path computes `pow`/`log10` in float already; the kernel uses the
  same operations so drift stays far under 1e-3.
