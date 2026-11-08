# Datasheet digitiser

These scripts turn the graphs in published film datasheets into the stock files in `data/stocks/`.
Every number OpenFilm uses for a stock can be traced back through this folder to a page of a
manufacturer's PDF.

The PDFs themselves are never committed. Download them yourself into a directory of their own:

| File name  | Source |
|------------|--------|
| `7219.pdf` | https://www.kodak.com/content/products-brochures/Film/VISION3-500T-Color-Negative-Film-7219-TECHNICAL-DATA.pdf |
| `2383.pdf` | https://www.kodak.com/content/products-brochures/Film/KODAK-VISION-Color-Print-Film-2383-3383-data-sheet.pdf |

The ISO 5-3 Status A and Status M tables in `data/spectral/iso5_3_status_{a,m}.csv` were typed in
from ANSI/NAPM IT2.18-1996 (https://law.resource.org/pub/us/cfr/ibr/001/aimm.it2.18.1996.pdf),
Tables 3 and 4, and cross-checked against that PDF's OCR text.

## Setup

```sh
python3 -m venv <scratch>/venv
<scratch>/venv/bin/pip install -r tools/digitize/requirements.txt
```

Run every script with `python -I`: the PDFs are untrusted input, and `-I` stops Python picking up
modules from the current directory or the environment.

## Workflow

1. **Describe the figure** in `figures/<figure>.json` (format below).
2. **Get the points.**
   - Vector graphs (the 7219 PDF): `python -I tools/digitize/pdf_vectors.py --pdf-dir <pdfs> <figure>...`
     reads the drawn paths and writes `points/<figure>_<curve>.csv`.
   - Raster graphs (the 2383 PDF): read the points by eye from a 400 dpi render and type them into
     `points/<figure>_<curve>.csv`. Mark those curves `"hand": true` in the figure config.
3. **Check them.** `python -I tools/digitize/overlay.py --pdf-dir <pdfs> <figure>...` draws the points
   in magenta over a 300 dpi render of the graph and writes `build/digitize-review/<figure>.png`.
   Look at every image: the magenta must sit on the printed line, within about one line width.
   Fix and repeat.
4. **Build the stock.** `python -I tools/digitize/build_stock.py --meta tools/digitize/meta/<id>.json`
   resamples the points (spectra to 5 nm, characteristic curves to 0.05 log E), validates the result
   against `data/stocks/schema.json` and writes `data/stocks/<id>.json`. It prints `valid` on success.

## Figure config (`figures/<figure>.json`)

```json
{
  "comment": "what the graph is, how it was read, anything ambiguous",
  "pdf": "7219.pdf",
  "page": 4,
  "clip": [x0, y0, x1, y1],
  "columns": ["logE", "density"],
  "xAxis": {"v0": -4.0, "v1": 1.0, "labelClip": [x0, y0, x1, y1]},
  "yAxis": {"v0": 0.0, "v1": 3.0, "p0": 306.8, "p1": 122.3, "log": false},
  "curves": {
    "red": {"color": [0.011, 0.016, 0.017], "dashes": null, "bbox": [x0, y0, x1, y1], "xRange": [min, max]},
    "blue": {"hand": true}
  }
}
```

- Coordinates are PDF points (1/72 inch, origin top-left). `page` counts from 1.
- **Axes.** Each axis maps two reference values `v0`, `v1` to PDF positions `p0`, `p1`. Set `"log": true`
  for a logarithmic axis. If `p0`/`p1` are left out, `pdf_vectors.py` finds the tick labels reading
  `v0` and `v1` inside `labelClip` (a strip that holds only that axis's labels). It then snaps each
  one to the nearest drawn tick or frame line, because typeset labels can sit a point or two off
  their tick. For raster graphs, give `p0`/`p1` explicitly, measured from the tick marks on the render.
- **Curves.** A vector curve is the single path in the clip that matches `color` (stroke RGB, ±0.02),
  `dashes` (PyMuPDF dash string such as `"[ 1.8001 1.8001 ] 0"`, or `null` for solid) and, when every curve is
  drawn in the same colour and dash, `bbox` (the path's bounding box, ±1 pt). `xRange` optionally
  trims the points. A `"hand": true` curve is read by eye and skipped by `pdf_vectors.py`.

## Stock meta (`meta/<id>.json`)

This file holds the same fields as the stock schema. Where the schema wants a curve or a spectrum,
the meta file gives the path of a points CSV (relative to this folder) instead. `comment` is kept
here and not copied into the stock file.

`build_stock.py` clamps dye and base densities at 0, because Kodak's D-min subtraction leaves tails
of about -0.04. It also makes characteristic curves non-decreasing, and refuses to build if any dip
is larger than 0.02 D.
