<p align="center">
  <img src="branding/openfilm-logo.png" alt="OpenFilm — physical film emulation" width="640">
</p>

# OpenFilm

OpenFilm is an open-source OpenFX plugin for DaVinci Resolve Studio (macOS, Metal) that emulates
motion-picture film by simulating its physical behaviour: a spectral camera-negative → print
pipeline with halation, bloom and grain, built from published film-stock datasheets.

> **⚠️ Work in progress.** OpenFilm is under active development and not ready for production use. Milestones 1 (plugin scaffold), 2 (spectral 500T → 2383 colour model), 3 (GPU rendering on Metal, real-time playback) and 4 (halation, bloom, sharpness, film format) are done; grain is still to come. Expect breaking changes.

## Controls

* **Colour / Negative / Print:** input and output space, gamut compression, negative and print
  stock, exposure, printer lights and print density.
* **Film format:** Super 35, Academy, Super 16, 16mm, 65mm or Custom (with a gate width in mm). All
  spatial effects are sized in millimetres on the film, so the format changes their scale.
* **Sharpness:** one switch for the emulsion scatter of the negative and the print (the print is scatter-only) and the negative's development edge effect.
* **Halation:** on/off, Rem-jet (on = faint halo, off = strong red glow), Strength (1 = physical) and
  Radius (multiplier).
* **Bloom:** on/off (off by default), Amount and Radius (mm).

Quick start: `cmake -S . -B build && cmake --build build && ctest --test-dir build && scripts/install.sh`, then restart Resolve.
Full guide: [docs/building.md](docs/building.md).
How the film colour model works: [docs/film-model.md](docs/film-model.md).
Design: [docs/superpowers/specs/2026-11-08-film-emulator-design.md](docs/superpowers/specs/2026-11-08-film-emulator-design.md).

Built against OpenFX `OFX_Release_1.5.1` (git submodule in `external/openfx`).

Licence: MIT.
