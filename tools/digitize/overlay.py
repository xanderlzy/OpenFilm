"""Draw digitised points back onto the datasheet graph so a person can check them by eye.

Usage:  python -I tools/digitize/overlay.py --pdf-dir <dir with the PDFs> <figure> [<figure>...]

Renders the figure clip at 300 dpi and plots every point of points/<figure>_<curve>.csv in magenta
(vector-extracted points as small dots, hand-read points as rings). Writes
build/digitize-review/<figure>.png. A good digitisation puts the magenta on the printed line,
within about one line width, all along the curve.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fitz  # PyMuPDF  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from figure import REPO_ROOT, load_figure, points_path, read_points  # noqa: E402
from pdf_vectors import figure_axes  # noqa: E402

DPI = 300
MAGENTA = (255, 0, 255)


def overlay(fig, pdf_dir, out_dir):
    doc = fitz.open(os.path.join(pdf_dir, fig["pdf"]))
    clip = fitz.Rect(fig["clip"])
    page = doc[fig["page"] - 1]
    pix = page.get_pixmap(dpi=DPI, clip=clip)
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    draw = ImageDraw.Draw(img)
    scale = DPI / 72.0
    xa, ya = figure_axes(page, fig)
    for name, curve in fig["curves"].items():
        path = points_path(fig["name"], name)
        if not os.path.exists(path):
            print(f"  {name}: no points file, skipped")
            continue
        r = 4 if curve.get("hand") else 1.5
        for x, y in read_points(path):
            px = (xa.to_pdf(x) - clip.x0) * scale
            py = (ya.to_pdf(y) - clip.y0) * scale
            if curve.get("hand"):
                draw.ellipse([px - r, py - r, px + r, py + r], outline=MAGENTA, width=2)
            else:
                draw.ellipse([px - r, py - r, px + r, py + r], fill=MAGENTA)
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, fig["name"] + ".png")
    img.save(out)
    print(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pdf-dir", required=True, help="directory holding the datasheet PDFs (never committed)")
    ap.add_argument("--out-dir", default=os.path.join(REPO_ROOT, "build", "digitize-review"))
    ap.add_argument("figures", nargs="+")
    args = ap.parse_args()
    for f in args.figures:
        overlay(load_figure(f), args.pdf_dir, args.out_dir)


if __name__ == "__main__":
    main()
