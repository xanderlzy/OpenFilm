#!/usr/bin/env bash
# Installs the built OpenFilm plugin where DaVinci Resolve looks for OpenFX plugins.
# Usage (from anywhere): scripts/install.sh    — asks for your password (sudo).
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
bundle="$repo_root/build/OpenFilm.ofx.bundle"
dest="/Library/OFX/Plugins"

if [[ ! -d "$bundle" ]]; then
  echo "error: $bundle not found. Build first: cmake -S . -B build && cmake --build build" >&2
  exit 1
fi

sudo mkdir -p "$dest"
sudo rm -rf "$dest/OpenFilm.ofx.bundle"
sudo cp -R "$bundle" "$dest/"
echo "Installed OpenFilm to $dest/OpenFilm.ofx.bundle"
echo "Restart DaVinci Resolve to load OpenFilm."
