# M4 — Halation, bloom and sharpness: design

Milestone 4 of [the OpenFilm design](2026-11-08-film-emulator-design.md). Builds on M3
([design](2026-11-09-m3-metal-color-path-design.md)): the colour path runs on Metal in real time,
with the CPU path as reference.

## Goal

Add the film's spatial behaviour — emulsion sharpness with development edge effects, halation, and
optional in-camera diffusion — each acting where it physically happens, scaled by the film format,
on CPU and GPU, while keeping UHD real time.

## Decisions (agreed with the user)

| Topic | Decision |
|---|---|
| Halation default | Physically accurate 5219: rem-jet on → faint warm edge on hard highlights. A **Rem-jet** switch (off → strong red glow, CineStill-like) |
| Halation controls | Enable, Rem-jet on/off, Strength (1 = physical), Radius (multiplier, 1 = physical). No hue control: colour comes from the physics |
| Bloom | In-camera diffusion (lens / Pro-Mist-like) acting on the light before the negative; off by default; Amount, Radius (mm). Brightness-driven, not detail-driven |
| Emulsion scatter | Part of Sharpness (MTF), per layer, fitted to the datasheet |
| Edge (adjacency) effects | In M4, per layer, in developed density; fitted together with the scatter so the total sharpness matches the datasheet curve including the hump above 100 %. Follows the Sharpness switch; no separate control |
| Print | 2383 gets its own sharpness from its datasheet curve; no print halation (2383's anti-halation dye makes it negligible) |
| Film format | Menu: Super 35 (24.89 mm, default), 35mm Academy (21.95 mm), Super 16 (12.52 mm), 16mm (10.26 mm), 65mm 5-perf (52.48 mm), Custom + "Gate width (mm)". Image width ↔ gate width |
| Speed | UHD real time at 24 fps with all M4 effects on: colour + M4 ≤ 20 ms per UHD frame on the M4 Pro, worst-case noise frame included. No Quality control yet (arrives with M5 grain) |
| Blur method | Each spread is a sum of 2–3 separable Gaussians; wide ones computed at reduced resolution. CPU runs the same steps |
| Off = M3 | With Sharpness, Halation and Bloom all off, output equals M3's exactly |

Out of scope: grain (M5); push/pull and its effect on edge effects (M6); more stocks (M7);
gate weave, flicker, dust, vignette.

## Pipeline

The fused M3 kernel becomes a sequence of passes. Each spatial step acts in the domain where it
happens physically.

```
DI ─[lutE]→ per-layer linear exposure E (× 2^Exposure)          image: E_r, E_g, E_b
 ├ Bloom (if on):  E ← (1−a)·E + a·(G_bloom ∗ E)                 same kernel all layers
 ├ Scatter (if Sharpness): E_c ← S_c ∗ E_c                       per-layer kernel S_c
 └ Halation (if on): E_c ← E_c + h·w_c·(H ∗ E_r)                 light back from the base
─[neg curves]→ negative density D_c
 └ Edge effect (if Sharpness): D_c ← D_c + k_c·(D_c − A_c ∗ D_c) per-layer
─[lutPE]→ print log E → linear print exposure P
 └ Print sharpness (if Sharpness): P_c ← S^print_c ∗ P_c
─(log, + printer offsets)→ print curves → lutD → display matrix → clip + gamma
```

- Bloom: `a` = Amount; `G_bloom` is a 2-Gaussian sum whose overall width is Radius (mm).
  Diffusing scene light before the film equals diffusing each layer's exposure, because exposure is
  linear in light.
- Halation: light that passed through the emulsion reflects from the base back into it. The
  red-sensitive layer's exposure is the proxy for the light reaching the base (it is the deepest
  layer). `H` is a sum of Gaussians sized from the stock's base thickness × Radius. `h` = rem-jet
  reflectance (on/off value from the stock data) × Strength. `w_c` = how the returning light splits
  between the layers (red largest), from stock data. Halation off adds exactly nothing.
- Scatter and edge effect are fitted jointly per layer: the combined frequency response
  `MTF_scatter(f) · (1 + k·(1 − MTF_A(f)))` must match the datasheet MTF curve.
- Print sharpness acts on linear print exposure (light through the negative onto the print).
- Every kernel used for spreading (`G_bloom`, `S_c`, `H`, `A_c`, `S^print_c`) is normalised to
  sum to 1, so spreading moves light without creating it. The edge effect keeps the mean density.
- Kernels are specified in millimetres on the film and converted to pixels with
  `px_per_mm = image width in pixels / gate width (mm)`, using the OFX render scale so lower-
  resolution previews match.
- Wide Gaussians are computed on a downsampled image and upsampled, identically on CPU and GPU.
  The width threshold is set in the plan.

## Data

- `data/stocks/kodak-5219-type.json` gains:
  - `mtf`: per-layer `[[cycles/mm, response]...]`, digitised from the 7219 datasheet (vector
    paths where available, otherwise hand-read), reviewed with `overlay.py` like M2's figures.
  - `halation`: `baseThicknessMm`, `remjetReflectance {on, off}`, `layerSplit [r, g, b]`, each with
    a `sources` entry (datasheet, or published literature where the datasheet is silent).
- `data/stocks/kodak-2383-type.json` gains `mtf` per layer from the 2383 datasheet.
- `schema.json` extended accordingly. Fitted Gaussian parameters are derived at load time (not
  stored), so the data stays the datasheet's.

## Controls (OFX)

| Group | Control | Range / default |
|---|---|---|
| Film format | Format | menu above, Super 35 |
| | Gate width (mm) | 5–80, used when Custom |
| Sharpness | Enable | on |
| Halation | Enable | on |
| | Rem-jet | on |
| | Strength | 0–5, 1 |
| | Radius | 0.25–4, 1 |
| Bloom | Enable | off |
| | Amount | 0–1, 0.25 |
| | Radius (mm) | 0.05–5, 0.5 |

None of these trigger a re-bake of the colour tables. Non-finite values are treated as the default.

## Testing

| Test | Pass bar |
|---|---|
| All M4 effects off | output identical to the M3 path (bit-exact on CPU; GPU within M3's 1e-3) |
| Sharpness fit per layer, negative and print | fitted response within 0.05 of every digitised datasheet MTF point, including the >100 % hump |
| Measured sharpness of the full CPU pipeline (sine targets through the renderer, Halation and Bloom off) | within 0.07 of the datasheet negative curve at the digitised frequencies |
| Spreading conserves light | each normalised kernel: sum of output = sum of input within 1e-4 relative; edge effect preserves mean density within 1e-4 |
| Halation physics | a single bright point gets a halo with red > green > blue energy; rem-jet off gives more halo energy than on by the stock data's ratio (±5 %); Halation off adds exactly nothing |
| Format scaling | doubling the gate width halves the halo / bloom radius in pixels (±1 px); a half-resolution render scale matches the downsampled full render (within 2e-3) |
| GPU vs CPU | max difference ≤ 1e-3 per channel, alpha bit-exact, on test images incl. points, edges, noise, odd sizes, sub-windows |
| Speed | colour + all M4 effects, UHD, worst-case noise frame: median ≤ 20 ms on the M4 Pro |
| Digitised data | overlay PNGs reviewed by the user, as in M2 |
| In Resolve (user) | real-time UHD with everything on; halation subtle with rem-jet, strong without; Super 16 vs Super 35 visibly different in halo size and softness; with effects off the image equals M3 |

## Risks

- **Halation inputs the datasheet does not give** (rem-jet reflectance, layer split). Mitigation:
  cite published measurements; keep them in the stock JSON with sources; Strength exists for taste.
- **Edge effects need a density-domain neighbourhood pass**, splitting the colour chain in two;
  per-pass memory traffic at UHD is the main speed risk. Mitigation: half-float intermediate images
  if the budget requires it, with the GPU-vs-CPU bar re-checked.
- **Window edges**: spreads need pixels outside the render window. The plugin requests a source
  region enlarged by the largest kernel radius (OFX regions of interest) and edge-clamps beyond the
  source image. Resolve's support for regions of interest is checked in Resolve early.
- **M3's speed headroom** on noisy frames (colour path ~9 ms worst case): the plan re-budgets
  before adding passes (half-float tables or a smaller GPU exposure table, with accuracy re-checked).
