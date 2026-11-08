# How OpenFilm's colour model works

This page describes what OpenFilm (Milestone 2) actually does to your picture, in film terms. It is
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
| Negative | Exposure | -10 to +10 stops (0) | Over- or underexposes the negative. Changing it never causes a re-bake. |
| Print | Print stock | list | Print 2383-type |
| Print | Printer light R / G / B | -20 to +20 points (0) | Lab-timer style trim. One point is 0.025 log E (about 1/12 stop). 0 is the balanced, grey-neutral setting. |
| Print | Print density | -20 to +20 points (0) | Points added to all three lights together: overall printing density. |

**How the printer lights behave.** More light in a channel exposes that print layer more, so it
forms *more* of the complementary dye. Red light up, more cyan dye, so the picture moves toward
cyan (and loses red), exactly as a lab timer's trim does. Print density up makes the whole print
denser and darker; down makes it lighter.

## Why playback is slow in this milestone

M2 runs entirely on the CPU. Resolve's GPU (Metal) path is switched off for this plugin until M3,
so expect slow scrubbing and playback; this is a look-development build.

To keep each frame manageable, the spectral maths is pre-computed ("baked") into lookup tables:
a 113 x 113 x 113 table for the negative (DI code to exposure) and a 97 x 97 x 97 table for the
print (negative density to display code). The sizes were chosen so the tables agree with the full
spectral calculation to within 0.4 dE2000 (measured worst case 0.38 over 2000 random colours and
Exposure from -5 to +5 stops, the range the slider shows; 0.34 for deep shadows pushed the same
amount). Exposure beyond +/-5 stops is still correct in the midtones and highlights but is not
checked as tightly in the deepest shadows. Building them takes about **1.4 s**, and happens
whenever the printer lights, print density, stocks or gamut compression change. Exposure never
re-bakes, so it is the control to ride while playing. The last four looks are cached, so returning
to one of them (the same exact settings) is instant. A slider drag passes through many new looks,
and each one is a fresh bake; while one bakes, frames that use an already-baked look keep
rendering.

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
* **No halation, bloom or grain** yet; those come in later milestones.

## Glossary

* **H&D curve** (Hurter and Driffield): a film layer's characteristic curve, plotting the density
  it develops to against the log of the exposure it received.
* **dE2000 (Delta E 2000):** the CIE's measure of how different two colours look; about 1 is a
  just-noticeable difference.
* **u'v':** a chromaticity coordinate system (CIE 1976) where equal distances look roughly equally
  different; used here to measure how neutral a grey is.
* **Bradford:** the standard method for adjusting colours from one white point to another (here
  D55 to D65).
