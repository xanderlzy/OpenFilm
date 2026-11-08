# OpenFilm

OpenFilm is an open-source OpenFX plugin for DaVinci Resolve Studio (macOS, Metal) that emulates
motion-picture film by simulating its physical behaviour: a spectral camera-negative → print
pipeline with halation, bloom and grain, built from published film-stock datasheets.

Status: early development (Milestone 1 — plugin scaffold).

Quick start: `cmake -S . -B build && cmake --build build && ctest --test-dir build && scripts/install.sh`, then restart Resolve.
Full guide: [docs/building.md](docs/building.md).
Design: [docs/superpowers/specs/2026-11-08-film-emulator-design.md](docs/superpowers/specs/2026-11-08-film-emulator-design.md).

Built against OpenFX `OFX_Release_1.5.1` (git submodule in `external/openfx`).

Licence: MIT.
