"""Assemble a stock JSON (data/stocks/<id>.json) from a meta file plus digitised point CSVs.

Usage:  python -I tools/digitize/build_stock.py --meta tools/digitize/meta/<id>.json [--out-dir data/stocks]

The meta file holds everything the schema asks for. Where the schema wants a curve or spectrum, the
meta gives instead the path of a points CSV (relative to tools/digitize/). This script:
  - resamples spectra onto the 5 nm grid, only inside the range that was digitised (and 380-780 nm);
  - resamples characteristic curves every 0.05 log E;
  - clamps dye and base densities at 0 (tiny negative tails come from Kodak's D-min subtraction);
  - makes characteristic curves non-decreasing (a real film curve never loses density with more
    exposure; any dip is digitising noise) and refuses to build if a dip is bigger than 0.02 D;
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


def _load(rel):
    return read_points(os.path.join(TOOLS_DIR, rel))


def build(meta):
    out = {k: v for k, v in meta.items() if k not in ("layers", "dyeDensity", "baseDensity", "comment")}
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
    # Keep the schema's key order so the files diff cleanly.
    order = ["id", "displayName", "type", "densitometry", "sources", "balanceIlluminantK",
             "midGreyLogExposure", "labAimDensity", "layers", "dyeDensity", "baseDensity", "interlayer"]
    return {k: out[k] for k in order if k in out}, report


def check_sorted(doc):
    def arrays(node):
        if isinstance(node, list) and node and all(isinstance(p, list) and len(p) == 2 for p in node):
            yield node
        elif isinstance(node, dict):
            for v in node.values():
                yield from arrays(v)
    for arr in arrays({"layers": doc["layers"], "dye": doc["dyeDensity"], "base": doc.get("baseDensity", [])}):
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
