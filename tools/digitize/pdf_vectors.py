"""Read curves straight out of a vector-drawn datasheet graph (e.g. the Kodak 7219 PDF).

Usage:  python -I tools/digitize/pdf_vectors.py --pdf-dir <dir with the PDFs> <figure> [<figure>...]

For each curve in figures/<figure>.json, keeps the page drawings inside the figure clip whose stroke
colour and dash pattern match (and, when every curve is drawn the same way, whose bounding box matches
the curve's "bbox"), flattens them to points, converts PDF coordinates to data units and writes
points/<figure>_<curve>.csv.

Axis calibration: each axis names two reference values v0, v1. Their PDF positions p0, p1 are found
from the tick labels in page.get_text("dict"), then snapped to the nearest drawn tick or frame line
(the labels are typeset by hand and can sit a point or two off their tick). An explicit "p0"/"p1" in
the config overrides the automatic value.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fitz  # PyMuPDF  # noqa: E402

from figure import Axis, load_figure, points_path, write_points  # noqa: E402

BEZIER_STEPS = 12
SNAP_PT = 3.0


def _parse_number(text):
    try:
        return float(text.strip().replace("−", "-"))
    except ValueError:
        return None


def _label_centre(page, clip, value, horizontal):
    """PDF position of the tick label reading `value` (x for a horizontal axis, y for vertical).

    `clip` is the axis's "labelClip": a strip holding only that axis's labels, so a number that also
    appears on the other axis cannot be picked up by mistake.
    """
    hits = []
    for block in page.get_text("dict", clip=clip)["blocks"]:
        for line in block.get("lines", []):
            for span in line["spans"]:
                if _parse_number(span["text"]) == value:
                    x0, y0, x1, y1 = span["bbox"]
                    hits.append(((x0 + x1) / 2, (y0 + y1) / 2))
    if len(hits) != 1:
        raise SystemExit(f"expected one tick label '{value}' in {clip}, found {len(hits)}")
    return hits[0][0] if horizontal else hits[0][1]


def _tick_lines(drawings, horizontal):
    """Positions of drawn straight segments perpendicular to the axis (ticks, grid and frame lines)."""
    out = []
    for d in drawings:
        # Ticks, grid lines and frames are one- or two-item paths; anything longer is a data curve,
        # whose steep segments must never be mistaken for a tick.
        if len(d["items"]) > 2:
            continue
        for it in d["items"]:
            if it[0] == "l":
                a, b = it[1], it[2]
                if horizontal and abs(a.x - b.x) < 0.05:
                    out.append(a.x)
                elif not horizontal and abs(a.y - b.y) < 0.05:
                    out.append(a.y)
            elif it[0] == "re":
                r = it[1]
                if r.width >= 1.0 and r.height >= 1.0:  # a frame contributes its edges
                    out += [r.x0, r.x1] if horizontal else [r.y0, r.y1]
                elif horizontal and r.width < 1.0:  # thin upright filled rect = x tick
                    out.append((r.x0 + r.x1) / 2)
                elif not horizontal and r.height < 1.0:  # thin flat filled rect = y tick
                    out.append((r.y0 + r.y1) / 2)
    return out


def calibrate(page, clip, drawings, spec, horizontal):
    spec = dict(spec)
    clip = fitz.Rect(spec.get("labelClip", clip))
    lines = _tick_lines(drawings, horizontal)
    for k in ("0", "1"):
        if "p" + k in spec:
            continue
        label = _label_centre(page, clip, float(spec["v" + k]), horizontal)
        near = [p for p in lines if abs(p - label) <= SNAP_PT]
        spec["p" + k] = min(near, key=lambda p: abs(p - label)) if near else label
        print(f"  axis {'x' if horizontal else 'y'} value {spec['v' + k]}: label at {label:.2f} -> {spec['p' + k]:.2f}")
    return Axis(spec)


def _flatten(item):
    kind = item[0]
    if kind == "l":
        return [item[1], item[2]]
    if kind == "c":
        p0, c1, c2, p3 = item[1:5]
        pts = []
        for i in range(BEZIER_STEPS + 1):
            t = i / BEZIER_STEPS
            u = 1 - t
            pts.append(p0 * (u ** 3) + c1 * (3 * u * u * t) + c2 * (3 * u * t * t) + p3 * (t ** 3))
        return pts
    return []


def _matches(d, curve):
    col = d.get("color")
    if col is None or any(abs(a - b) > 0.02 for a, b in zip(col, curve["color"])):
        return False
    want = " ".join((curve.get("dashes") or "[] 0").split())
    if " ".join((d.get("dashes") or "[] 0").split()) != want:
        return False
    bbox = curve.get("bbox")
    if bbox is not None and any(abs(a - b) > 1.0 for a, b in zip(d["rect"], bbox)):
        return False
    return True


def figure_axes(page, fig, drawings=None):
    """The x Axis and a dict of y Axis objects for a figure (explicit p0/p1 win; otherwise labels
    snapped to ticks). The dict has "y" and, for figures with a second scale on the other side
    (the granularity graphs), "y2" from the config's "yAxis2"; a curve picks its scale with "axis"."""
    clip = fitz.Rect(fig["clip"])
    if drawings is None:
        drawings = [d for d in page.get_drawings() if clip.contains(d["rect"])]
    ys = {"y": calibrate(page, clip, drawings, fig["yAxis"], False)}
    if "yAxis2" in fig:
        ys["y2"] = calibrate(page, clip, drawings, fig["yAxis2"], False)
    return calibrate(page, clip, drawings, fig["xAxis"], True), ys


def curve_y_axis(ys, curve):
    return ys[curve.get("axis", "y")]


def _subpath(items, n):
    """The n-th pen-up-separated run of a path's items (a drawing can hold several curves)."""
    runs, prev = [[]], None
    for it in items:
        start = it[1]
        if prev is not None and (abs(start.x - prev.x) > 0.01 or abs(start.y - prev.y) > 0.01):
            runs.append([])
        runs[-1].append(it)
        prev = it[2] if it[0] == "l" else it[4]
    if n >= len(runs):
        raise SystemExit(f"subpath {n} requested but the path has {len(runs)}")
    return runs[n]


def extract(fig, pdf_dir):
    doc = fitz.open(os.path.join(pdf_dir, fig["pdf"]))
    page = doc[fig["page"] - 1]
    clip = fitz.Rect(fig["clip"])
    drawings = [d for d in page.get_drawings() if clip.contains(d["rect"])]
    print(f"{fig['name']}: {len(drawings)} drawings in clip")
    xa, ys = figure_axes(page, fig, drawings)
    for name, curve in fig["curves"].items():
        if curve.get("hand"):
            continue
        hits = [d for d in drawings if _matches(d, curve)]
        if len(hits) != 1:
            raise SystemExit(f"{fig['name']}/{name}: expected 1 matching path, found {len(hits)}")
        ya = curve_y_axis(ys, curve)
        items = hits[0]["items"]
        if "subpath" in curve:
            items = _subpath(items, curve["subpath"])
        pts = []
        for it in items:
            pts += [(xa.to_value(p.x), ya.to_value(p.y)) for p in _flatten(it)]
        lo, hi = curve.get("xRange", [-1e9, 1e9])
        pts = sorted({(round(x, 4), round(y, 4)) for x, y in pts if lo <= x <= hi})
        out = points_path(fig["name"], name)
        write_points(out, curve.get("columns", fig.get("columns", ["x", "y"])), pts,
                     f"{fig['name']} / {name}: vector path from {fig['pdf']} p.{fig['page']} (pdf_vectors.py)")
        print(f"  {name}: {len(pts)} points, x {pts[0][0]:.3f}..{pts[-1][0]:.3f} -> {os.path.relpath(out)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pdf-dir", required=True, help="directory holding the datasheet PDFs (never committed)")
    ap.add_argument("figures", nargs="+")
    args = ap.parse_args()
    for f in args.figures:
        extract(load_figure(f), args.pdf_dir)


if __name__ == "__main__":
    main()
