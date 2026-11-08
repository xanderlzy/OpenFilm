"""Shared helpers for the datasheet digitiser: figure configs, axis calibration, point CSVs.

A figure config (figures/<name>.json) describes one graph in a datasheet PDF: where it sits on the
page, how PDF coordinates map to data units on each axis, and which curves to read from it.
"""
import csv
import json
import math
import os

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(TOOLS_DIR))
FIGURES_DIR = os.path.join(TOOLS_DIR, "figures")
POINTS_DIR = os.path.join(TOOLS_DIR, "points")


def load_figure(name_or_path):
    path = name_or_path if name_or_path.endswith(".json") else os.path.join(FIGURES_DIR, name_or_path + ".json")
    with open(path) as f:
        fig = json.load(f)
    fig["name"] = os.path.splitext(os.path.basename(path))[0]
    return fig


class Axis:
    """Linear (or log) map between a PDF coordinate and a data value, fixed by two reference ticks."""

    def __init__(self, spec):
        self.p0, self.p1 = float(spec["p0"]), float(spec["p1"])
        self.log = bool(spec.get("log", False))
        v0, v1 = float(spec["v0"]), float(spec["v1"])
        # A log axis is linear in log10(value), so calibrate in that space.
        self.v0, self.v1 = (math.log10(v0), math.log10(v1)) if self.log else (v0, v1)

    def to_value(self, p):
        v = self.v0 + (p - self.p0) * (self.v1 - self.v0) / (self.p1 - self.p0)
        return 10 ** v if self.log else v

    def to_pdf(self, v):
        v = math.log10(v) if self.log else v
        return self.p0 + (v - self.v0) * (self.p1 - self.p0) / (self.v1 - self.v0)


def points_path(fig_name, curve):
    return os.path.join(POINTS_DIR, f"{fig_name}_{curve}.csv")


def write_points(path, header, pts, note):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        f.write(f"# {note}\n")
        w = csv.writer(f)
        w.writerow(header)
        for x, y in pts:
            w.writerow([f"{x:.4f}", f"{y:.4f}"])


def read_points(path):
    """Read an [x, y] CSV, skipping '#' comment lines and the header row."""
    with open(path) as f:
        rows = [r for r in csv.reader(line for line in f if not line.startswith("#")) if r]
    return [(float(a), float(b)) for a, b in rows[1:]]
