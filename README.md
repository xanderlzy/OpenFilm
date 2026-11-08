# OpenFilm

OpenFilm is an open-source OpenFX plugin for DaVinci Resolve Studio (macOS, Metal) that emulates
motion-picture film by simulating its physical behaviour: a spectral camera-negative → print
pipeline with halation, bloom and grain, built from published film-stock datasheets.

> **⚠️ Work in progress.** OpenFilm is under active development and not ready for production use. Milestone 1 (plugin scaffold) is done; Milestone 2 (spectral colour model, CPU) is in progress on the `m2-color-model` branch. Expect breaking changes.

Quick start: `cmake -S . -B build && cmake --build build && ctest --test-dir build && scripts/install.sh`, then restart Resolve.
Full guide: [docs/building.md](docs/building.md).
How the film colour model works: [docs/film-model.md](docs/film-model.md).
Design: [docs/superpowers/specs/2026-11-08-film-emulator-design.md](docs/superpowers/specs/2026-11-08-film-emulator-design.md).

Built against OpenFX `OFX_Release_1.5.1` (git submodule in `external/openfx`).

Licence: MIT.
