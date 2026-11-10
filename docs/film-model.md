# How OpenFilm's colour model works

This page describes what OpenFilm (Milestone 4) actually does to your picture, in film terms. It is
written for colorists; no code reading needed. Everything here is the current build, including
its shortcuts (see "Known approximations").

## The chain, stage by stage

OpenFilm follows the photochemical route: **scene → camera negative → printer → print film →
projector → screen.**

1. **Input (DaVinci Wide Gamut / Intermediate).** Your node is assumed to be in DaVinci
   Intermediate. Mid-grey (18 %) sits at the usual DI code value. The only input choice today is
   DWG/Intermediate.
2. **Gamut compression.** DWG can describe colours no light source can produce (neon, deep LED
   blue). Those are pulled smoothly back to the Rec.2020 boundary (the widest set of colours a real
   light source can approach) before anything else happens. See the *Gamut compression* control.
3. **Scene light as a spectrum.** Each pixel's colour is turned into a plausible full spectrum
   (380-780 nm, every 5 nm), because film responds to wavelengths, not to RGB. A spectrum has to be
   invented from three numbers, so OpenFilm uses a published smooth-spectrum method (Jakob and
   Hanika, 2019). For very saturated colours the method is deliberately tamed
   (saturation-dependent brightness of the spectrum) so it stays inside the range it can
   represent. This changes the look of extremely saturated reds and magentas compared with a naive
   upsampler; they are a little calmer. Please check them on your footage.
4. **Tungsten scene light.** The negative is **Kodak VISION3 500T (5219/7219)**, balanced for 3200 K.
   Scene white is therefore treated as a 3200 K blackbody. A neutral grey at 18 % lands on the
   negative at **-1.50 log E**, the speed point read from the datasheet's sensitometric curve
   axis labels.
5. **Exposure on the negative.** Each of the three dye layers (red-, green-, blue-sensitive) adds up
   the light it sees through its own spectral sensitivity curve. The *Exposure* control shifts all
   three in stops here, just like opening the lens.
6. **Development.** Each layer's exposure becomes density along its own H&D (characteristic)
   curve, digitised from the datasheet. This is where highlight roll-off, toe and the layers'
   slightly different contrasts come from.
7. **Dyes.** The cyan, magenta and yellow dye clouds have measured spectral density curves.
   OpenFilm works out how much of each dye is needed to give the densities the curves ask for, so
   dye overlap (cyan eating some green, and so on) is respected. The dye curves are held at their
   end values outside the digitised range rather than dropping to clear film.
8. **Printer.** The negative is printed onto **Kodak VISION 2383** using a 3200 K tungsten lamp
   behind a Wratten 2B-style UV cut, as the datasheet's printing condition describes. The printer
   lights (below) act here, as exposure changes in the print's three layers.
9. **Auto-balance.** Before your printer lights are applied, OpenFilm balances the printer so that
   an 18 % grey comes out neutral at display level 0.409 (that is 11.7 % display-linear). 2383's
   aim is Kodak's LAD (Laboratory Aim Density) of 1.09 / 1.06 / 1.03 Status A, from Kodak
   publication H-61B. So "0" on the printer lights means *balanced*, not "no light".
10. **Print development and dyes.** Same idea as steps 6-7, with 2383's own curves and dyes.
11. **Projection.** The print is lit by a projector lamp at CIE D55 (about 5500 K, a typical
    film-projection white) and its transmitted colour measured. White is then adapted (Bradford) to
    D65.
12. **Output (Rec.709, Gamma 2.4).** The result is encoded for a Rec.709 BT.1886 display. 18 % grey
    at 0 stops reads about **0.409** (roughly 41 %).

## Controls

| Group | Control | Range (default) | What it does |
|---|---|---|---|
| Colour | Input space | one entry | DaVinci Wide Gamut / Intermediate |
| Colour | Output space | one entry | Rec.709, Gamma 2.4 |
| Colour | Gamut compression | 0 to 1 (0.5) | 0 hard-clips colours outside what real light can make; 1 starts a soft roll-in earlier. Leave near the middle unless neon or LED colours misbehave. |
| Negative | Negative stock | list | Tungsten 500T (5219-type) |
| Negative | Exposure | -10 to +10 stops (0) | Over- or underexposes the negative. Applied per pixel, instant. |
| Print | Print stock | list | Print 2383-type |
| Print | Printer light R / G / B | -20 to +20 points (0) | Lab-timer style trim, applied per pixel, instant. One point is 0.025 log E (about 1/12 stop). 0 is the balanced, grey-neutral setting. |
| Print | Print density | -20 to +20 points (0) | Points added to all three lights together: overall printing density. |
| Film format | Format | list (Super 35) | Gate width of the film the picture is laid onto: Super 35 (24.89 mm), 35mm Academy (21.95 mm), Super 16 (12.52 mm), 16mm (10.26 mm), 65mm 5-perf (52.48 mm) or Custom. It sets the real size, in millimetres, of every effect below. |
| Film format | Gate width (mm) | 5 to 80 (24.89) | Only active when Format is Custom. |
| Sharpness | Enable | on | Emulsion light scatter in the negative and the print, and the negative's development edge effect. |
| Halation | Enable | on | Light that passes through the emulsion, bounces off the back of the film base and exposes the layers again. |
| Halation | Rem-jet | on | On: the stock's anti-halation backing is present (faint halo). Off: the backing was removed before exposure, CineStill-style (strong red glow). |
| Halation | Strength | 0 to 5 (1) | 1 is the physical amount; above 1 is for taste. |
| Halation | Radius | 0.25 to 4 (1) | Multiplier on the physical radius. |
| Bloom | Enable | off | In-camera diffusion (lens or diffusion filter) before the light reaches the negative. |
| Bloom | Amount | 0 to 1 (0.25) | Share of the light that is diffused. |
| Bloom | Radius (mm) | 0.05 to 5 (0.5) | Diffusion radius measured on the film. |
| Grain | Enable | on | Dye-cloud grain on the negative and on the print. Off is identical to a version without grain. |
| Grain | Amount | 0 to 3 (1) | 1 is the datasheet's rms granularity; 0 is the same as off; above 1 is for taste. |
| Grain | Size | 0.5 to 2 (1) | Scales the size of the dye clouds, so the texture gets coarser or finer. The datasheet's rms strength stays the same. |
| Grain | Quality | Preview, Full (Preview) | Preview is real time. Full is the exact simulation, slower: set it by hand before final renders. |

**How the printer lights behave.** More light in a channel exposes that print layer more, so it
forms *more* of the complementary dye. Red light up, more cyan dye, so the picture moves toward
cyan (and loses red), exactly as a lab timer's trim does. Print density up makes the whole print
denser and darker; down makes it lighter.

## Sharpness, halation, bloom and film format

These effects act in the place of the chain where they happen in real film, not as a filter on the
finished picture. Sizes are in millimetres on the film, so the **Format** menu matters: the same
halo is a bigger part of a Super 16 frame than of a Super 35 one, and a 65mm frame is cleaner.
Switching format changes how big these effects are, not how the colour looks. With all three
effects off, you get the colour-only path. Since Milestone 4 that path reads the exposure table at half
precision, at most 0.444 dE2000 from the direct model (see "Exposure table precision" below).

* **Bloom** (optional, off by default) happens first, in front of the negative: some of the scene
  light is spread by a diffusion filter or lens. It acts on exposure, so bright highlights bloom
  much more than shadows, as they do in the camera.
* **Sharpness, negative.** Light scatters inside each emulsion layer before it is recorded, which
  softens fine detail; each of the three layers has its own softness. During development the
  chemistry then exaggerates edges slightly (the adjacency or edge effect), which restores some
  bite. Both are fitted to the datasheet's MTF (sharpness versus detail size) curves.
* **Halation** acts on the exposure too. Light that gets through the emulsion reflects off the back
  of the film base and re-exposes the layers around the bright spot. The returned light is red
  because the red-sensitive layer is the deepest, nearest the base, and is the one that light
  mostly reaches, so highlights get a red-orange
  halo. With Rem-jet on, the dark backing absorbs nearly all of it (about 1 % comes back); with
  it off, much more does (about 29 %), giving the classic glow.
* **Sharpness, print.** The print film scatters light again (three scatter lobes rather than two).
  The print is modelled as scatter only, with no edge effect of its own. A fitted print edge effect
  put a wide green-tinted halo (about 120 px at UHD) on every mid-contrast edge, so it was left out.

### What is measured and what is assumed

* **Fitted to data.** The scatter lobes (and the negative's edge effect) come from the digitised
  datasheet MTF curves. The worst point of each fit, in response units (1.0 = 100 %): 5219 red
  0.017, green 0.022, blue 0.007; 2383 red 0.065, green 0.075, blue 0.072.
* **Why the print fit is looser.** The worst 2383 points are all at the lowest frequency
  (2.54 cycles/mm), where the datasheet curves sit 5-6 % above 100 %. Scatter alone can't go above
  100 %, so the print can't reach those points. Past them the fit follows the curves closely.
* **Limits on the negative's edge effect.** The edge reach is capped at 0.5 mm and the edge gain at
  0.3. Larger values only cancelled the dip on sine-wave test targets and would put a visible fringe
  on real edges.
* **Assumed, not published.** No manufacturer publishes halation values, so these are assumptions:
  the base is 0.127 mm acetate (the datasheet says acetate base with a rem-jet backing); the light
  returned is 1 % with rem-jet and 29 % without; and how the halo is shared between the layers is
  derived from the 5219 sensitivities, which in practice means red only. **Strength** exists so
  you can adjust it to taste.

### Window and render scale

The effects need to see a little beyond the area being drawn, so OpenFilm asks Resolve for a
slightly larger source region than the one it renders (the margin depends on which effects are on
and on the format; with everything off it asks for nothing extra). At the edge of the frame the
picture is extended by repeating its border pixels. On proxy or reduced-resolution playback the
sizes are kept in proportion: a halo that is 1/50 of the width at full resolution is also 1/50 of
the width at half resolution.

## Grain

**What is simulated.** Film grain is dye. After development each layer holds small clouds of
dye, and where clouds overlap their densities add, as light through two filters does. OpenFilm
scatters clouds of a fixed radius at random over the film, so the density at a point is the
mean density plus or minus a random count of clouds. The texture has the same character at every
density of a layer, and the mean density is untouched: grain adds texture, not a shift. What
changes with density is the strength, and that follows the datasheet exactly.

**Where it sits in the chain.** On the negative, grain is added to the density after development
(and after the halation and sharpness effects), so it is a property of the negative like the real
thing, and the print then copies it. The print has its own grain, added to the print density after
print development, so the print's grain is not softened by the print's scatter.

**Strength: measured.** Strength follows the stock's rms granularity curve from the datasheet
(Kodak 7219 / 5219 type for the negative, Status M density; 2383 type for the print, Status A
density), measured with a 48 micrometre circular aperture, at the actual density of each pixel.
Amount 1 reproduces those numbers at every digitised density. The curves are read per layer, so
the colour of the grain follows the stock.

**Texture: assumed.** The datasheets give strength, not shape, so two things are assumptions:

* **Cloud radius:** 5 micrometres for the negative and 2 micrometres for the print. No primary
  source for either was found. The only figure found is secondary: T. Vitale's 2007 notes give colour
  dye clouds of roughly 10 to 15 micrometres, without saying whether that is radius or diameter.
  The 5 micrometre negative radius is our assumption (the low end of that range, read as a
  diameter), and the 2 micrometre print radius is likewise ours. **Size** multiplies these radii. It changes the
  scale of the texture and not the 48 micrometre rms, so Size is a pure look control.
* **Mean overlap:** on average 4 clouds cover any point. This sets how clumpy or smooth the
  texture is, and the datasheet cannot fix it, so it is an assumption.

**Format.** Grain is sized in millimetres on the film, like the other effects. A smaller gauge
spreads the same grain over a larger part of the picture, so Super 16 looks coarser than Super 35
at the same output resolution, and 65mm looks finer.

**New grain every frame.** The grain pattern is seeded from the frame's time, so it changes on every
frame, and rendering the same frame twice gives the same grain.

**Quality: Preview and Full.**

* **Preview** (default) reads the grain from precomputed, tiling plates, one per layer and stage,
  so it plays in real time. It uses the same strength curve, and its statistics match Full: the rms
  differs by about 1 % in our tests, against a 10 % bar. On the GPU it costs about 3.8 ms per UHD
  frame on top of the other effects.
* **Full** simulates the clouds exactly for each pixel and frame. It takes about 0.3 s per UHD
  frame on the GPU (about 6.5 s on the CPU path). **Resolve does not switch this for you: set
  Quality to Full by hand before a final render, and back to Preview for grading.**
* **Waits.** The first time grain is used, and whenever the format, the gate width or Size change,
  the plates are rebuilt, about 0.2 s on the GPU (a few seconds on the CPU path). Playback stalls
  briefly once, then continues.

**Grain off, or Amount 0,** gives exactly the same picture as without grain.

## How it renders: GPU, with no waiting on the printer lights

OpenFilm renders on the GPU (Metal in Resolve), so playback is real time. The CPU version of the
same maths is kept as the reference: the GPU is checked against it, and they agree to well below
anything visible.

To keep each frame fast, the spectral maths is pre-computed ("baked") into lookup tables. The
negative side is one table (DI code to exposure). The print side is computed in two tables, split
at the point where the print is exposed. The printer lights act exactly at that split, like a
printer's light valves sitting between the negative and the print stock. The first table takes the
negative to the light that reaches the print; your lights and Print density then shift that light
per pixel; the second table takes it on to the screen.

Because of this split, **Exposure, the three printer lights and Print density are all applied per
pixel with no recompute**: drag them freely during playback and the picture follows immediately.
The last table stores projected light on a log scale (so deep shadows keep their detail), and the
screen's black clip and Gamma 2.4 are applied per pixel afterwards.

Only a change of stock or Gamut compression needs a new bake, which takes about **1.5 s**; playback
then shows the new look. The last four looks are cached, so returning to one of them (the same
exact settings) is instant. While one bakes, frames that use an already-baked look keep rendering.

## Where the stock data comes from

All numbers are digitised by hand/script from Kodak datasheets (the PDFs themselves are not
included in the repository). Each stock's JSON file in `data/stocks/` lists its sources.

* **Tungsten 500T (5219-type).** KODAK VISION3 500T Color Negative Film 5219 / 7219 / SO-219,
  Technical Data H-1-5219t (Revised 7-15): sensitometric curves, spectral sensitivity, spectral dye
  density. <https://www.kodak.com/content/products-brochures/Film/VISION3-500T-Color-Negative-Film-7219-TECHNICAL-DATA.pdf>
* **Print 2383-type.** KODAK VISION Color Print Film 2383 / 3383 data sheet, H-1-2383 (Revised
  3-22): same three curve sets.
  <https://www.kodak.com/content/products-brochures/Film/KODAK-VISION-Color-Print-Film-2383-3383-data-sheet.pdf>
* **2383 aim density (LAD).** LAD for KODAK VISION Color Print Film, H-61B (the 2383 sheet names the
  LAD but gives no numbers): Status A 1.09 / 1.06 / 1.03.
  <https://www.kodak.com/content/products-brochures/Film/LAD-for-KODAK-VISION-Color-Print-Film-H-61b.pdf>
* **Status A / Status M densitometry.** ANSI/NAPM IT2.18-1996 (ISO 5-3 weighting tables).
  Camera negatives are read in Status M, prints in Status A.
* **Colour matching and daylight.** CIE 1931 2-degree observer and the CIE daylight model
  (S0/S1/S2), used to compute D55 and D65.

Curve digitising was checked by overlaying the extracted curves on the datasheet pages; see the
review images generated in `build/digitize-review/`.

## Known approximations

* **No interlayer effects yet.** Real negatives use inhibition between layers (the DIR couplers) to
  boost saturation. This version has no such term.
* **Status-table edge slopes are ignored.** The Status A/M weighting tables are used as tabulated.
* **Projector white is D55.** Real cinemas vary (xenon is nearer 5500-6000 K); the print is judged
  under D55, Bradford-adapted to D65.
* **Printer lamp is a 3200 K blackbody** with a Wratten 2B-style UV cut; no printer heat glass, no
  lamp ageing.
* **Grey-ramp neutrality.** Because of the unmodelled printer heat glass and digitising tolerance, a
  grey ramp is neutral to within **0.008 u'v'** inside +/-1 stop and **0.012 u'v'** inside +/-2
  stops of mid-grey; beyond that a faint tint appears. This is to be revisited in M7.
* **Digitising tolerance.** Curves come from chart images, with a tolerance of roughly 0.02
  density.
* **Dye and base spectra beyond the digitised range** (below 400 nm and above 750 nm or so) are held
  flat at their end values.
* **2383 UV-sensitivity branches excluded.** The print stock's sensitivity below about 400 nm
  (the 372-399 nm branches on the datasheet) is not modelled; the Wratten-2B-style cut stands in
  for it.
* **No print D-min or base spectrum.** The print is modelled as dye density only; the clear
  base and fog of the real print film are not added.
* **Grain, halation, bloom and sharpness** are in (see their sections); the cloud radii and overlap of the grain, and the halation inputs, are assumptions.
* **Lookup tables.** The exposure table is 129 points per side, the print-exposure and print
  tables 97 (their points are packed toward clear film, where the eye is most sensitive).
* **Baked versus the full spectral model.** Across Exposure +/-5 stops, with printer lights and
  Print density over their full +/-20 range, the worst difference is **0.44 dE2000**; in the deep
  shadows it is 0.19.
* **Dark, saturated colours with extreme lights.** With extreme printer lights or Print density
  these can be off by up to about **0.57 dE2000**, still below a just-noticeable difference (~1).
* **Extreme Exposure.** Beyond +/-5 stops, the deepest shadows are checked less tightly.
* **GPU versus CPU reference.** They agree within 1/1000 of a code value (measured 0.00008).
* **Speed.** The colour path alone takes about 4 ms per UHD frame on an M4 Pro for normal footage.
  With every effect on, UHD takes about **19.0-19.9 ms** depending on load (median, on a pure-noise
  frame, the hard case; the target is 20 ms), which is real time at 24 fps. While rendering, one UHD
  frame uses about 500 MB of GPU memory. At most three frames' worth are in use at once (a fourth
  frame waits for one to finish). Spare sets are kept so the next frame starts fast. All but one
  are released after a second of idling, and the system can reclaim any of them if it needs memory.
* **Exposure table precision.** The exposure table is stored rounded to half precision on both the
  CPU and the GPU; against the direct model the worst difference is **0.444 dE2000**.

## Glossary

* **H&D curve** (Hurter and Driffield): a film layer's characteristic curve, plotting the density
  it develops to against the log of the exposure it received.
* **dE2000 (Delta E 2000):** the CIE's measure of how different two colours look; about 1 is a
  just-noticeable difference.
* **u'v':** a chromaticity coordinate system (CIE 1976) where equal distances look roughly equally
  different; used here to measure how neutral a grey is.
* **Bradford:** the standard method for adjusting colours from one white point to another (here
  D55 to D65).
