"""Assemble a stock JSON (data/stocks/<id>.json) from a meta file plus digitised point CSVs.

Usage:  python -I tools/digitize/build_stock.py --meta tools/digitize/meta/<id>.json [--out-dir data/stocks]

The meta file holds everything the schema asks for. Where the schema wants a curve or spectrum, the
meta gives instead the path of a points CSV (relative to tools/digitize/). This script:
  - resamples spectra onto the 5 nm grid, only inside the range that was digitised (and 380-780 nm);
  - resamples characteristic curves every 0.05 log E;
  - clamps dye and base densities at 0 (tiny negative tails come from Kodak's D-min subtraction);
  - makes characteristic curves non-decreasing (a real film curve never loses density with more
    exposure; any dip is digitising noise) and refuses to build if a dip is bigger than 0.02 D;
  - keeps MTF points as digitised (percent -> fraction, repeated frequencies averaged, no resampling);
  - granularity: takes each grain-versus-log-E point through the figure's own density curve to
    [density, sigma D] (see granularity_layer); the cloud radius and sources are copied from the meta;
  - derives halation.layerSplit from the layer sensitivities (negatives only);
  - validates the result against data/stocks/schema.json and prints "valid".
"""
import argparse
import json
import math
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jsonschema  # noqa: E402
import numpy as np  # noqa: E402

from figure import REPO_ROOT, TOOLS_DIR, read_points  # noqa: E402

NM_MIN, NM_MAX, NM_STEP = 380, 780, 5
LOGE_STEP = 0.05
MAX_MONOTONE_FIX = 0.02
GRAN_MERGE_D = 0.01  # granularity points closer in density than this are averaged
SCHEMA = os.path.join(REPO_ROOT, "data", "stocks", "schema.json")


def _clean(pts):
    """Sort by x and average any repeated x, so linear interpolation is well defined."""
    xs = {}
    for x, y in pts:
        xs.setdefault(x, []).append(y)
    x = np.array(sorted(xs))
    return x, np.array([np.mean(xs[k]) for k in x])


def resample_spectrum(pts, clamp_zero):
    x, y = _clean(pts)
    lo = max(NM_MIN, NM_STEP * math.ceil(x[0] / NM_STEP - 1e-9))
    hi = min(NM_MAX, NM_STEP * math.floor(x[-1] / NM_STEP + 1e-9))
    grid = np.arange(lo, hi + 1, NM_STEP)
    v = np.interp(grid, x, y)
    clamped = int((v < 0).sum())
    if clamp_zero:
        v = np.maximum(v, 0.0)
    return [[int(g), round(float(a), 4) + 0.0] for g, a in zip(grid, v)], clamped


def resample_curve(pts, label):
    x, y = _clean(pts)
    k0 = math.ceil(round(x[0] / LOGE_STEP, 6))
    k1 = math.floor(round(x[-1] / LOGE_STEP, 6))
    grid = np.array([k * LOGE_STEP for k in range(k0, k1 + 1)])
    v = np.interp(grid, x, y)
    mono = np.maximum.accumulate(v)
    fix = float((mono - v).max())
    if fix > MAX_MONOTONE_FIX:
        raise SystemExit(f"{label}: density falls by {fix:.3f} D somewhere; re-check the digitised points")
    return [[round(float(g), 2) + 0.0, round(float(a), 4) + 0.0] for g, a in zip(grid, mono)], fix


def layer_split(layers):
    """Relative weight of back-reflected light per layer: sum of linear sensitivity over 600-700 nm,
    divided by the red layer's. The light that reaches the base is what the upper layers did not
    absorb, i.e. red."""
    sums = []
    for ch in ("red", "green", "blue"):
        sums.append(sum(10 ** v for nm, v in layers[ch]["logSensitivity"] if 600 <= nm <= 700))
    split = [float(f"{s / sums[0]:.4g}") for s in sums]
    return split, f"layerSplit (600-700 nm sensitivity ratio): {split}"


def granularity_layer(grain_pts, density_pts, label):
    """[[density, sigma]] for one layer. The datasheet plots grain against relative log E beside the
    figure's own density curves; each grain point is taken to density by linear interpolation on
    the layer's density curve (points outside it are dropped). Points whose densities fall within
    GRAN_MERGE_D of each other (the flat toe, where the curve is a plateau) are averaged so density is
    strictly ascending."""
    dx, dy = _clean(density_pts)
    kept = [(float(np.interp(x, dx, dy)), s) for x, s in grain_pts if dx[0] <= x <= dx[-1]]
    dropped = len(grain_pts) - len(kept)
    kept.sort()
    merged = []
    for d, s in kept:
        if merged and d - merged[-1][-1][0] < GRAN_MERGE_D:
            merged[-1].append((d, s))
        else:
            merged.append([(d, s)])
    pairs = [[round(float(np.mean([d for d, _ in g])), 4), round(float(np.mean([s for _, s in g])), 4)] for g in merged]
    return pairs, f"{label}: {len(pairs)} points from {len(grain_pts)} ({dropped} outside the density curve), " \
                  f"density {pairs[0][0]}..{pairs[-1][0]}, sigma {min(p[1] for p in pairs)}..{max(p[1] for p in pairs)}"


def build_granularity(g):
    out = {"apertureUm": g["apertureUm"], "cloudRadiusUm": g["cloudRadiusUm"]}
    lines = []
    for ch in ("red", "green", "blue"):
        out[ch], line = granularity_layer(_load(g[ch]["grain"]), _load(g[ch]["density"]), f"{ch} granularity")
        lines.append(line)
    out["sources"] = g["sources"]
    return out, lines


def _load(rel):
    return read_points(os.path.join(TOOLS_DIR, rel))


def build(meta):
    out = {k: v for k, v in meta.items() if k not in ("layers", "dyeDensity", "baseDensity", "comment", "mtf", "halation", "granularity")}
    report = []
    out["layers"] = {}
    for ch in ("red", "green", "blue"):
        src = meta["layers"][ch]
        sens, _ = resample_spectrum(_load(src["logSensitivity"]), clamp_zero=False)
        curve, fix = resample_curve(_load(src["characteristicCurve"]), f"{ch} characteristic curve")
        out["layers"][ch] = {"logSensitivity": sens, "characteristicCurve": curve}
        report.append(f"{ch}: sensitivity {sens[0][0]}-{sens[-1][0]} nm, curve {curve[0][0]}..{curve[-1][0]} log E"
                      f" (monotone fix {fix:.4f} D)")
    out["dyeDensity"] = {}
    for dye in ("cyan", "magenta", "yellow"):
        spec, clamped = resample_spectrum(_load(meta["dyeDensity"][dye]), clamp_zero=True)
        out["dyeDensity"][dye] = spec
        report.append(f"{dye} dye: {spec[0][0]}-{spec[-1][0]} nm, {clamped} samples clamped to 0")
    if "baseDensity" in meta:
        spec, clamped = resample_spectrum(_load(meta["baseDensity"]), clamp_zero=True)
        out["baseDensity"] = spec
        report.append(f"base density: {spec[0][0]}-{spec[-1][0]} nm, {clamped} samples clamped to 0")
    out["mtf"] = {}
    for ch in ("red", "green", "blue"):
        f, pct = _clean(_load(meta["mtf"][ch]))
        out["mtf"][ch] = [[round(float(a), 4), round(float(b) / 100.0, 4)] for a, b in zip(f, pct)]
        report.append(f"{ch} MTF: {len(f)} points, {f[0]:.2f}-{f[-1]:.1f} c/mm")
    if "halation" in meta:
        hal = dict(meta["halation"])
        hal["layerSplit"], note = layer_split(out["layers"])
        report.append(note)
        out["halation"] = {k: hal[k] for k in ("baseThicknessMm", "remjetReflectance", "layerSplit", "sources")}
    if "granularity" in meta:
        out["granularity"], lines = build_granularity(meta["granularity"])
        report += lines
    # Keep the schema's key order so the files diff cleanly.
    order = ["id", "displayName", "type", "densitometry", "sources", "balanceIlluminantK",
             "midGreyLogExposure", "labAimDensity", "layers", "dyeDensity", "baseDensity", "mtf", "halation", "granularity", "interlayer"]
    return {k: out[k] for k in order if k in out}, report


def check_sorted(doc):
    def arrays(node):
        if isinstance(node, list) and node and all(isinstance(p, list) and len(p) == 2 for p in node):
            yield node
        elif isinstance(node, dict):
            for v in node.values():
                yield from arrays(v)
    for arr in arrays({"layers": doc["layers"], "dye": doc["dyeDensity"], "base": doc.get("baseDensity", []),
                    "gran": {k: doc["granularity"][k] for k in ("red", "green", "blue")} if "granularity" in doc else {}}):
        xs = [p[0] for p in arr]
        if xs != sorted(set(xs)):
            raise SystemExit("an [x, y] array is not strictly ascending")


def dumps(doc):
    """JSON with one [x, y] pair per line: readable and diff-friendly."""
    text = json.dumps(doc, indent=2)
    # Collapse the innermost two-number lists json.dumps spread over four lines.
    return re.sub(r"\[\s+(-?[\d.eE+-]+),\s+(-?[\d.eE+-]+)\s+\]", r"[\1, \2]", text) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--meta", required=True)
    ap.add_argument("--out-dir", default=os.path.join(REPO_ROOT, "data", "stocks"))
    args = ap.parse_args()
    with open(args.meta) as f:
        meta = json.load(f)
    doc, report = build(meta)
    with open(SCHEMA) as f:
        schema = json.load(f)
    jsonschema.Draft202012Validator(schema).validate(doc)
    check_sorted(doc)
    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, doc["id"] + ".json")
    with open(out, "w") as f:
        f.write(dumps(doc))
    for line in report:
        print("  " + line)
    print(f"{doc['id']}: valid -> {os.path.relpath(out, REPO_ROOT)}")


if __name__ == "__main__":
    main()
