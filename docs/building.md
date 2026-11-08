# Building, testing and installing OpenFilm

This guide assumes you are comfortable in Resolve but not necessarily with C++ toolchains.
Every command below is typed in Terminal.

## 1. What you need

- **macOS 12 or newer on Apple Silicon.**
- **Xcode** from the App Store. Open it once so it finishes installing its components, then run
  `xcode-select --install` for the command-line tools. Xcode provides the C++ compiler and Apple's
  Metal shader compiler.
- **CMake** 3.24 or newer: `brew install cmake` (with [Homebrew](https://brew.sh)).
- **DaVinci Resolve Studio.** OpenFilm targets Studio.

## 2. Get the code

```sh
git clone --recursive <repository-url> OpenFilm
cd OpenFilm
```

`--recursive` also downloads the OpenFX SDK into `external/openfx`. If you forgot it, run
`git submodule update --init`.

## 3. Build

```sh
cmake -S . -B build
cmake --build build
```

The first command prepares a `build/` folder (and downloads the small `doctest` test library);
the second compiles everything. The result is the plugin bundle `build/OpenFilm.ofx.bundle`.

## 4. Test

```sh
ctest --test-dir build --output-on-failure
```

Three test suites run:

| Suite | What it proves |
|---|---|
| `core` | The CPU film maths does exactly what it should (exposure, image edges, padding). |
| `gpu` | The Metal (GPU) path produces the same numbers as the CPU path. |
| `bundle` | The built plugin loads like Resolve loads it and identifies itself as OpenFilm. |

All three should report `Passed`.

## 5. Install

```sh
scripts/install.sh
```

It copies the bundle to `/Library/OFX/Plugins` (it asks for your Mac password). Restart Resolve.

## 6. Use it in Resolve

Color page → **Effects** panel → **OpenFX** → **OpenFilm**. Drag it onto a node. The
**Negative → Exposure** knob changes exposure in stops (Milestone 1 has only this control).

## Troubleshooting

- **OpenFilm is not in the list.** Resolve caches its plugin list. Quit Resolve, delete
  `~/Library/Application Support/Blackmagic Design/DaVinci Resolve/OFXPluginCacheV2.xml`, and start
  Resolve again. Also check that `/Library/OFX/Plugins/OpenFilm.ofx.bundle` exists.
- **Red message "OpenFilm: GPU shaders failed to load".** The file
  `OpenFilm.ofx.bundle/Contents/Resources/OpenFilm.metallib` is missing or damaged. Rebuild and
  reinstall. Copy the whole bundle folder, never just the inner file.
- **Resolve says the plugin is damaged or refuses to load it.** The bundle must stay signed. Always
  install with `scripts/install.sh` after a full build; editing files inside the bundle breaks the
  signature.

## How the code is organised

```
src/core/    The film maths, in plain C++. No Resolve, no GPU. This is the "reference".
src/metal/   The same maths written for the GPU (Metal), plus the code that sends it to the GPU.
src/ofx/     The thin layer that talks to Resolve through OpenFX: knobs, image formats, and
             choosing CPU or GPU for each frame.
tests/       Automatic checks for each layer (see "Test" above).
```

**Why there is a CPU path at all.** Resolve normally renders on the GPU with Metal, which is fast.
The CPU version is slower, but it is easier to read and to test. It is the definition of what is
correct: every GPU kernel is checked against it automatically. Resolve also uses the CPU path if
GPU rendering is unavailable.

**How a frame flows.** Resolve calls OpenFilm's `render` with the source image, an empty output
image, and the current knob values. With Metal enabled, the images are GPU buffers and OpenFilm
queues its GPU kernels on Resolve's command queue. Otherwise, OpenFilm runs the CPU version on the
same data.
