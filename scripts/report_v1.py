#!/usr/bin/env python3
"""Report for the frozen ShapeHash v1 format: results/v1/summary.md and
comparison sheets, against the phase-1 exploration numbers."""
import csv
import glob
import json
import os
import sys
from collections import Counter

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # works under python -I
from common import ROOT  # noqa: E402

R = os.path.join(ROOT, "results")
METRICS = [("score", "Score", "{:.3f}", 1), ("chamfer", "Chamfer x100", "{:.2f}", 100), ("f05", "F@0.05", "{:.3f}", 1), ("normal", "Normal", "{:.3f}", 1),
           ("deltaE", "dE", "{:.1f}", 1), ("sil_iou", "Sil IoU", "{:.3f}", 1), ("ssim", "SSIM", "{:.3f}", 1), ("psnr_blur", "PSNR-blur", "{:.1f}", 1)]
LABEL = {"voxs": "phase 1: voxs (best single)", "mesh": "phase 1: mesh", "sh": "phase 1: sh", "auto": "phase 1: auto (oracle over 9 methods)",
         "vox1": "v1 vox, smooth decode", "vox1c": "v1 vox, cube decode", "sh1": "v1 sh", "mesh1": "v1 mesh",
         "union1": "v1 union, Python encoder", "ts1": "v1 union, TypeScript encoder"}


def score(r):
    return 0.25 * (r["f05"] + r["sil_iou"] + r["ssim"] + (1 - min(r["deltaE"], 40.0) / 40.0)) if "ssim" in r else 0.0


def fill(r):
    r["score"] = score(r)
    if "ssim" not in r:
        r.update(chamfer=1.0, f05=0.0, normal=0.0, deltaE=40.0, sil_iou=0.0, ssim=0.0, psnr_blur=5.0, tris=0, bytes=0, enc_s=0.0, dec_ms=0.0)
    return r


def load(d):
    return [fill(r) for p in sorted(glob.glob(os.path.join(R, d, "raw", "*.json"))) for r in json.load(open(p))]


def main():
    rows = load("v1") + load("v1ts")
    for r in csv.DictReader(open(os.path.join(R, "results.csv"))):
        if r["method"] in ("voxs", "mesh", "sh", "auto"):
            rows.append({k: (float(v) if k not in ("model", "method", "hash", "picked") and v != "" else v) for k, v in r.items()})
    for r in rows:
        r["budget"] = int(r["budget"])
    models = sorted({r["model"] for r in rows})
    split = {"all": models, "tuning half": models[0::2], "held-out half": models[1::2]}
    budgets = [32, 64, 128, 256, 512]
    order = ["voxs", "mesh", "sh", "auto", "vox1c", "vox1", "sh1", "mesh1", "union1", "ts1"]

    def sel(m, b, keep):
        return [r for r in rows if r["method"] == m and r["budget"] == b and r["model"] in keep]

    md = ["# ShapeHash v1: evaluation\n", f"{len(models)} models, same harness and metrics as the phase-1 exploration (see README).",
          "Composite score = mean of F-score@0.05, silhouette IoU, SSIM and 1 - min(dE,40)/40.\n"]
    for name, keep in split.items():
        md.append(f"\n## Composite score, {name} ({len(keep)} models)\n")
        md.append("| method | " + " | ".join(f"{b} B" for b in budgets) + " |")
        md.append("|---|" + "---|" * len(budgets))
        for m in order:
            cells = []
            for b in budgets:
                rs = sel(m, b, keep)
                cells.append(f"{np.mean([r['score'] for r in rs]):.3f}" if len(rs) == len(keep) else "-")
            md.append(f"| {LABEL[m]} | " + " | ".join(cells) + " |")
    for b in (64, 128, 256):
        md.append(f"\n## All metrics at {b} bytes (28 models)\n")
        md.append("| method | bytes | " + " | ".join(m[1] for m in METRICS) + " | triangles | enc s | dec ms (Python) |")
        md.append("|---|---|" + "---|" * (len(METRICS) + 3))
        for m in order:
            rs = sel(m, b, models)
            if len(rs) != len(models):
                continue
            md.append(f"| {LABEL[m]} | {np.mean([float(r['bytes']) for r in rs]):.0f} | " + " | ".join(f.format(np.mean([float(r[k]) for r in rs]) * s) for k, _, f, s in METRICS)
                      + f" | {np.mean([float(r['tris']) for r in rs]):.0f} | {np.mean([float(r['enc_s']) for r in rs]):.2f} | {np.mean([float(r['dec_ms']) for r in rs]):.1f} |")
    md.append("\n## Which method the union encoder keeps (count of 28 models)\n")
    md.append("| budget | encoder | vox | sh | mesh | oracle pick (full composite) vox / sh / mesh | union score | oracle score |")
    md.append("|---|---|---|---|---|---|---|---|")
    for b in budgets:
        for m, lab in (("union1", "Python"), ("ts1", "TypeScript")):
            rs = sel(m, b, models)
            if not rs:
                continue
            c = Counter(r.get("picked") for r in rs)
            orc, osc = Counter(), []
            for model in models:
                cand = {k: [r for r in rows if r["model"] == model and r["budget"] == b and r["method"] == k] for k in ("vox1", "sh1", "mesh1")}
                sc = {k: v[0]["score"] for k, v in cand.items() if v}
                best = max(sc, key=sc.get)
                orc[best[:-1]] += 1
                osc.append(sc[best])
            md.append(f"| {b} | {lab} | {c['vox']} | {c['sh']} | {c['mesh']} | {orc['vox']} / {orc['sh']} / {orc['mesh']} | {np.mean([r['score'] for r in rs]):.3f} | {np.mean(osc):.3f} |")
    md.append("\n## Per-model composite at 128 bytes\n")
    md.append("| model | phase 1 voxs | phase 1 auto | v1 vox | v1 sh | v1 mesh | v1 union (picked) | TypeScript (picked) |")
    md.append("|---|---|---|---|---|---|---|---|")
    for model in models:
        g = {m: (sel(m, 128, [model]) or [None])[0] for m in order}
        md.append(f"| {model} | " + " | ".join(f"{g[m]['score']:.3f}" if g[m] else "-" for m in ("voxs", "auto", "vox1", "sh1", "mesh1"))
                  + f" | {g['union1']['score']:.3f} ({g['union1'].get('picked')}) | {g['ts1']['score']:.3f} ({g['ts1'].get('picked')}) |")
    bench = os.path.join(R, "v1", "ts_decode_bench.json")
    if os.path.exists(bench):
        bj = json.load(open(bench))
        md.append(f"\n## TypeScript decoder: size and speed (Node {bj['node']})\n")
        md.append("| bundle | minified bytes | gzip bytes |\n|---|---|---|")
        for k, v in bj["sizes"].items():
            md.append(f"| {k} | {v['bytes']} | {v['gzip']} |")
        md.append("\n| budget | method | hashes | median ms | max ms | median triangles |\n|---|---|---|---|---|---|")
        for r in bj["rows"]:
            md.append(f"| {r['budget']} | {r['method']} | {r['hashes']} | {r['median_ms']} | {r['max_ms']} | {r['median_triangles']} |")
    tsj = os.path.join(R, "v1", "ts_encoded.json")
    if os.path.exists(tsj):
        tr = [r for r in json.load(open(tsj)) if "hash" in r]
        md.append("\n## TypeScript encoder time (Node, single thread)\n")
        md.append("| budget | mean ms | max ms | one-off load + sampling ms (mean) |\n|---|---|---|---|")
        for b in sorted({r["budget"] for r in tr}):
            rs = [r for r in tr if r["budget"] == b]
            md.append(f"| {b} | {np.mean([r['enc_ms'] for r in rs]):.0f} | {max(r['enc_ms'] for r in rs)} | {np.mean([r['prep_ms'] for r in rs]):.0f} |")
    open(os.path.join(R, "v1", "summary.md"), "w").write("\n".join(md) + "\n")

    # ---- sheets
    T, PADL, PADT = 112, 150, 18
    cols = [("orig", "original", "thumbs"), ("voxs", "phase1 voxs", "thumbs"), ("vox1c", "v1 vox cubes", "v1/thumbs"), ("vox1", "v1 vox smooth", "v1/thumbs"),
            ("sh1", "v1 sh", "v1/thumbs"), ("mesh1", "v1 mesh", "v1/thumbs"), ("union1", "v1 union (py)", "v1/thumbs"), ("ts1", "v1 union (ts)", "v1ts/thumbs")]
    os.makedirs(os.path.join(R, "v1", "sheets"), exist_ok=True)
    cache = {}

    def thumb(d, model, key):
        p = os.path.join(R, d, model + ".npz")
        if p not in cache:
            cache[p] = np.load(p) if os.path.exists(p) else {}
        z = cache[p]
        return np.asarray(Image.fromarray(z[key][0]).resize((T, T), Image.LANCZOS)) if key in z else np.full((T, T, 3), 230, np.uint8)

    for b in (64, 128, 256):
        img = Image.new("RGB", (PADL + T * len(cols), PADT + T * len(models)), "white")
        d = ImageDraw.Draw(img)
        for j, (_, lab, _) in enumerate(cols):
            d.text((PADL + j * T + 4, 3), lab, fill="black")
        for i, model in enumerate(models):
            d.text((4, PADT + i * T + T // 2 - 5), model, fill="black")
            for j, (key, _, dd) in enumerate(cols):
                img.paste(Image.fromarray(thumb(dd, model, "orig" if key == "orig" else f"{key}_{b}")), (PADL + j * T, PADT + i * T))
        img.save(os.path.join(R, "v1", "sheets", f"v1_budget_{b:03d}.png"))
    print(open(os.path.join(R, "v1", "summary.md")).read())


if __name__ == "__main__":
    main()
