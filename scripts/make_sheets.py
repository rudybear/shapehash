#!/usr/bin/env python3
"""Visual comparison sheets from results/thumbs: one per budget (rows =
models, columns = original + methods) and one per model (rows = budgets)."""
import glob
import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # works under python -I
from common import RESULTS  # noqa: E402

METHODS = ["box", "impostor", "kdop", "sh", "dct", "ellip", "mesh", "vox", "voxs"]
T = 112
PADL, PADT = 150, 18


def cell(arr, view=0):
    if arr is None:
        return np.full((T, T, 3), 230, np.uint8)
    return np.asarray(Image.fromarray(arr[view]).resize((T, T), Image.LANCZOS))


def sheet(rows, row_labels, col_labels, path):
    h, w = PADT + T * len(rows), PADL + T * len(col_labels)
    img = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(img)
    for j, c in enumerate(col_labels):
        d.text((PADL + j * T + 4, 3), c, fill="black")
    for i, (lab, cells) in enumerate(zip(row_labels, rows)):
        d.text((4, PADT + i * T + T // 2 - 5), lab, fill="black")
        for j, c in enumerate(cells):
            img.paste(Image.fromarray(c), (PADL + j * T, PADT + i * T))
    img.save(path)


def main():
    out = os.path.join(RESULTS, "sheets")
    os.makedirs(out, exist_ok=True)
    thumbs = {os.path.basename(p)[:-4]: np.load(p) for p in sorted(glob.glob(os.path.join(RESULTS, "thumbs", "*.npz")))}
    rows_all = [r for p in glob.glob(os.path.join(RESULTS, "raw", "*.json")) for r in json.load(open(p))]
    budgets = sorted({r["budget"] for r in rows_all if "budget" in r})
    methods = [m for m in METHODS if any(r.get("method") == m for r in rows_all)]
    for b in budgets:
        for view in (0, 1):
            rows = [[cell(t["orig"], view)] + [cell(t[f"{m}_{b}"] if f"{m}_{b}" in t else None, view) for m in methods] for t in thumbs.values()]
            sheet(rows, list(thumbs), ["original"] + methods, os.path.join(out, f"budget_{b:03d}_view{view}.png"))
    for name, t in thumbs.items():
        rows = [[cell(t["orig"])] + [cell(t[f"{m}_{b}"] if f"{m}_{b}" in t else None) for m in methods] for b in budgets]
        sheet(rows, [f"{b} bytes" for b in budgets], ["original"] + methods, os.path.join(out, f"model_{name}.png"))
    print("sheets written to", out)


if __name__ == "__main__":
    main()
