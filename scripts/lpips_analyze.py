#!/usr/bin/env python3
"""LPIPS experiment, stage 3: tables and sheets (results/lpips/*.md, *.png).
Reads results/lpips/metrics.csv (stage 2) and the existing evaluation rows."""
import csv
import glob
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np
from PIL import Image, ImageDraw
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # works under python -I
from common import ROOT  # noqa: E402

R = os.path.join(ROOT, "results")
OUT = os.path.join(R, "lpips")
BUD = (64, 128, 256)
P1 = ["box", "impostor", "kdop", "sh", "dct", "ellip", "mesh", "vox", "voxs"]
V1 = ["vox1c", "vox1", "sh1", "mesh1", "union1"]
PM = ["crop_alex", "crop_vgg", "crop_dists", "lp_alex", "lp_vgg", "grey_alex", "grey_vgg", "full_alex", "full_vgg", "blur_alex"]


def score(r):
    return 0.25 * (r["f05"] + r["sil_iou"] + r["ssim"] + (1 - min(r["deltaE"], 40.0) / 40.0)) if "ssim" in r else 0.0


def load():
    ev = {}
    for d in ("raw", "v1/raw", "v1ts/raw"):
        for p in glob.glob(os.path.join(R, d, "*.json")):
            for r in json.load(open(p)):
                if "ssim" in r:
                    r["score"] = score(r)
                    ev[(r["model"], r["method"], r["budget"])] = r
    rows = []
    for r in csv.DictReader(open(os.path.join(OUT, "metrics.csv"))):
        r = {k: (v if k in ("model", "method") else float(v)) for k, v in r.items()}
        r["budget"] = int(r["budget"])
        e = ev.get((r["model"], r["method"], r["budget"]))
        if e:
            r.update({k: e[k] for k in ("score", "f05", "sil_iou", "ssim", "deltaE", "chamfer", "normal", "psnr_blur")})
            r["picked"] = e.get("picked")
        rows.append(r)
    return rows


def thumbs(model, key, views=(0, 4), size=128):
    z = thumbs.cache.get(model)
    if z is None:
        z = thumbs.cache[model] = np.load(os.path.join(OUT, "cache", model + ".npz"))
    out = []
    for v in views:
        if key == "orig":
            a, s, m = z["orig/a"][v], z["orig/s"][v], z["orig/m"][v]
        else:
            a, s, m = z[key + "/a"][v], z[key + "/s"][v], z[key + "/m"][v]
        img = (a.astype(np.float32) * s[..., None] / 255).astype(np.uint8)
        img[~m] = 255
        ys, xs = np.nonzero(z["orig/m"][v] | m)
        if len(xs):
            cx, cy = (xs.min() + xs.max()) // 2, (ys.min() + ys.max()) // 2
            h = int(max(xs.max() - xs.min(), ys.max() - ys.min()) * 0.56) + 4
            x0, y0 = max(0, cx - h), max(0, cy - h)
            img = img[y0:cy + h, x0:cx + h]
        out.append(np.asarray(Image.fromarray(img).resize((size, size), Image.LANCZOS)))
    return np.concatenate(out, axis=1)


thumbs.cache = {}


def sheet(lines, path, size=128):
    """lines: list of (row label, [(caption, model, key) ...])"""
    ncol = max(len(c) for _, c in lines)
    W, H = 190 + ncol * (2 * size + 6), len(lines) * (size + 16) + 4
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    for i, (label, cells) in enumerate(lines):
        y = i * (size + 16) + 2
        for k, part in enumerate(label.split("\n")):
            d.text((4, y + 30 + 12 * k), part, fill="black")
        for j, (cap, model, key) in enumerate(cells):
            x = 190 + j * (2 * size + 6)
            d.text((x + 2, y), cap, fill="black")
            try:
                img.paste(Image.fromarray(thumbs(model, key, size=size)), (x, y + 13))
            except KeyError:
                pass
    img.save(path)


def table(header, lines):
    return ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + ["| " + " | ".join(str(c) for c in ln) + " |" for ln in lines]


def main():
    rows = load()
    models = sorted({r["model"] for r in rows})
    by = {(r["model"], r["method"], r["budget"]): r for r in rows}

    def get(model, method, b):
        return by.get((model, method, b)) or by.get((model, method, 0))

    def mean(method, b, col):
        v = [get(m, method, b) for m in models]
        v = [r[col] for r in v if r and col in r]
        return float(np.mean(v)) if len(v) >= 26 else float("nan")

    md = ["# LPIPS on ShapeHash renders: sanity checks\n",
          "Renders: orthographic, 256x256, white background, colour = albedo x (0.35 + 0.65 |n.v|) (headlight), six fixed views",
          "(azimuth/elevation 30/20, 120/20, 210/20, 300/20, 75/65, 255/-40). Values are means over views and over the 28 models; lower is better.",
          "Variants: `full` = whole frame; `crop` = cropped to the union of both silhouettes and resized to 224 px; `lp` = crop, reference texture",
          "low-passed on the surface; `blur` = crop, reference image blurred (sigma 4 px); `grey` = crop, both with uniform grey albedo.\n",
          "## Baselines and methods at 128 bytes\n"]
    cols = [f"{v}_{n}" for v in ("full", "crop", "lp", "blur", "grey") for n in ("alex", "vgg", "dists")]
    names = [("blank", "blank white image"), ("box", "bounding box, mean colour"), ("sphere", "ellipsoid in the bbox, mean colour"), ("impostor", "box + 6 face images"),
             ("kdop", "convex hull (k-DOP)"), ("orig_flat", "ORIGINAL geometry, one flat colour"), ("orig_lp", "ORIGINAL geometry, low-passed texture")] + \
            [(m, "phase 1 " + m) for m in P1[3:]] + [(m, "v1 " + m) for m in V1]
    md += table(["", "composite"] + cols, [[lab, f"{mean(m, 128, 'score'):.3f}" if m in P1 + V1 else "-"] + [f"{mean(m, 128, c):.3f}" for c in cols] for m, lab in names])
    md.append(f"\nMean fraction of the 256 px frame covered by the object: {np.mean([get(m, 'blank', 0)['coverage'] for m in models]):.3f}.\n")
    md.append("## Monotonic with budget? (crop variant)\n")
    for c in ("crop_alex", "crop_vgg", "crop_dists", "grey_alex"):
        md.append(f"\n`{c}`\n")
        md += table(["method"] + [f"{b} B" for b in (32, 64, 128, 256, 512)], [[m] + [f"{mean(m, b, c):.3f}" for b in (32, 64, 128, 256, 512)] for m in P1[2:] + V1])
    bad = 0
    tot = 0
    for m in P1[3:] + V1[:4]:
        for model in models:
            for b0, b1 in ((64, 128), (128, 256)):
                a, b = get(model, m, b0), get(model, m, b1)
                if a and b:
                    tot += 1
                    bad += b["crop_alex"] > a["crop_alex"] + 1e-9
    md.append(f"\nPer model: doubling the budget (64->128, 128->256) made `crop_alex` worse in {bad} of {tot} method/model cases.\n")
    md.append("## Is it texture or geometry?\n")
    md.append("`orig_lp` is the exact original geometry with the texture low-passed, `orig_flat` the exact geometry in one colour. If LPIPS were dominated by missing")
    md.append("texture detail (as in 2-D) these would score close to the placeholders. Per-model `crop_alex`:\n")
    md += table(["model", "coverage", "orig_lp", "orig_flat", "v1 union 128 B", "sphere", "blank"],
                [[m, f"{get(m,'blank',0)['coverage']:.2f}"] + [f"{get(m, k, 128)['crop_alex']:.3f}" for k in ("orig_lp", "orig_flat", "union1", "sphere", "blank")] for m in models])
    cov = [get(m, "blank", 0)["coverage"] for m in models]
    for c in ("full_alex", "full_vgg", "crop_alex", "crop_vgg", "crop_dists"):
        md.append(f"\nSpearman(coverage, {c}) across models for v1 union at 128 B: {spearmanr(cov, [get(m, 'union1', 128)[c] for m in models])[0]:+.2f}; "
                  f"for the blank image: {spearmanr(cov, [get(m, 'blank', 0)[c] for m in models])[0]:+.2f}")
    md.append("\n## The blank-image check\n")
    md.append("Number of models (of 28) on which a blank white image scores *better* than the baseline:\n")
    md += table(["metric", "vs box", "vs sphere", "vs k-DOP 128 B", "vs v1 union 64 B"],
                [[c] + [sum(get(m, "blank", 0)[c] < get(m, k, b)[c] for m in models) for k, b in (("box", 128), ("sphere", 0), ("kdop", 128), ("union1", 64))] for c in cols])
    open(os.path.join(OUT, "sanity.md"), "w").write("\n".join(md) + "\n")

    # ------------------------------------------------------------ agreement
    md = ["# LPIPS vs the composite score\n", "## Mean rank of each phase-1 method (1 = best of 9), per metric\n"]
    metrics = [("score", True)] + [(c, False) for c in PM]

    def ranks(b, col, hi, methods):
        acc = defaultdict(list)
        for model in models:
            vals = {m: get(model, m, b) for m in methods}
            if any(v is None or col not in v for v in vals.values()):
                continue
            order = sorted(methods, key=lambda m: -vals[m][col] if hi else vals[m][col])
            for i, m in enumerate(order):
                acc[m].append(i + 1)
        return {m: float(np.mean(v)) for m, v in acc.items()}

    for b in BUD:
        md.append(f"\n{b} bytes\n")
        rk = {c: ranks(b, c, hi, P1) for c, hi in metrics}
        md += table(["method", "composite"] + PM, [[m] + [f"{rk[c][m]:.2f}" for c, _ in metrics] for m in P1])
    md.append("\n## Mean value per method (all 28 models)\n")
    for b in BUD:
        md.append(f"\n{b} bytes\n")
        md += table(["method", "composite (higher better)", "F@0.05", "sil IoU", "SSIM", "dE"] + PM[:7],
                    [[m, f"{mean(m,b,'score'):.3f}", f"{mean(m,b,'f05'):.3f}", f"{mean(m,b,'sil_iou'):.3f}", f"{mean(m,b,'ssim'):.3f}", f"{mean(m,b,'deltaE'):.1f}"] + [f"{mean(m,b,c):.3f}" for c in PM[:7]] for m in P1 + V1])
    md.append("\n## Rank correlation (Spearman)\n")
    md.append("Within each model x budget, across the 9 phase-1 methods + 4 v1 methods (13 placeholders), then averaged over 28 models x 3 budgets. Signs flipped so +1 = same ranking.\n")
    allm = P1 + V1[:4]
    cols2 = ["score", "f05", "sil_iou", "ssim", "deltaE", "chamfer", "normal", "psnr_blur", "crop_alex", "crop_vgg", "crop_dists", "lp_alex", "grey_alex", "full_alex"]
    sign = {"score": 1, "f05": 1, "sil_iou": 1, "ssim": 1, "normal": 1, "psnr_blur": 1}
    C = np.zeros((len(cols2), len(cols2)))
    n = 0
    for model in models:
        for b in BUD:
            g = [get(model, m, b) for m in allm]
            if any(x is None or "score" not in x for x in g):
                continue
            M = np.array([[sign.get(c, -1) * x[c] for c in cols2] for x in g])
            C += spearmanr(M)[0]
            n += 1
    C /= n
    md += table([""] + cols2, [[a] + [f"{C[i, j]:+.2f}" for j in range(len(cols2))] for i, a in enumerate(cols2)])
    md.append(f"\n({n} groups.)\n")

    # disagreements on the best phase-1 method
    md.append("\n## Where the composite and LPIPS pick a different best method (phase-1 methods)\n")
    lines, tab = [], []
    cnt = Counter()
    for b in BUD:
        for model in models:
            g = {m: get(model, m, b) for m in P1}
            if any(v is None or "score" not in v for v in g.values()):
                continue
            pc = max(P1, key=lambda m: g[m]["score"])
            pa, pv, pd = (min(P1, key=lambda m: g[m][c]) for c in ("crop_alex", "crop_vgg", "crop_dists"))
            cnt[(b, "alex")] += pc != pa
            cnt[(b, "vgg")] += pc != pv
            cnt[(b, "dists")] += pc != pd
            cnt[(b, "n")] += 1
            if pc != pa or pc != pv:
                tab.append([model, b, pc, pa, pv, pd, f"{g[pc]['score']:.3f} / {g[pa]['score']:.3f}", f"{g[pc]['crop_alex']:.3f} / {g[pa]['crop_alex']:.3f}"])
                if b == 128:
                    lines.append((f"{model}\n{b} B", [("original", model, "orig"), (f"composite: {pc}", model, f"{pc}_{b}"), (f"alex: {pa}", model, f"{pa}_{b}"), (f"vgg: {pv}", model, f"{pv}_{b}"), (f"dists: {pd}", model, f"{pd}_{b}")]))
    md += table(["budget", "models", "LPIPS-alex differs", "LPIPS-vgg differs", "DISTS differs"], [[b, cnt[(b, "n")], cnt[(b, "alex")], cnt[(b, "vgg")], cnt[(b, "dists")]] for b in BUD])
    md.append("")
    md += table(["model", "budget", "composite picks", "alex picks", "vgg picks", "dists picks", "composite of (composite pick / alex pick)", "crop_alex of (composite pick / alex pick)"], tab)
    md.append("\nSheet (128 B rows): `disagree_phase1_128.png`.\n")
    sheet(lines, os.path.join(OUT, "disagree_phase1_128.png"))
    for b in BUD:
        wins = {c: Counter(max(P1, key=lambda m: get(model, m, b)["score"]) if c == "score" else min(P1, key=lambda m: get(model, m, b)[c]) for model in models) for c in ("score", "crop_alex", "crop_vgg", "crop_dists")}
        md.append(f"\nBest-method counts at {b} B: " + "; ".join(f"{c}: " + ", ".join(f"{k} {v}" for k, v in w.most_common()) for c, w in wins.items()))
    open(os.path.join(OUT, "agreement.md"), "w").write("\n".join(md) + "\n")

    # specific-artefact sheets
    sheet([(f"{m}\n128 B", [("original", m, "orig"), ("ellip", m, "ellip_128"), ("voxs", m, "voxs_128"), ("mesh", m, "mesh_128")]) for m in ("Duck", "Suzanne", "DamagedHelmet", "Avocado", "Corset", "CesiumMan")] +
          [(f"{m}\n512 B", [("original", m, "orig"), ("ellip", m, "ellip_512"), ("voxs", m, "voxs_512"), ("mesh", m, "mesh_512")]) for m in ("Duck", "Suzanne", "DamagedHelmet")],
          os.path.join(OUT, "ellip_cases.png"))
    sheet([(f"{m}\n128 B", [("original", m, "orig"), ("v1 vox smooth", m, "vox1_128"), ("v1 vox cubes", m, "vox1c_128"), ("v1 mesh", m, "mesh1_128"), ("v1 sh", m, "sh1_128")]) for m in ("SheenChair", "ProcTable", "Lantern", "AntiqueCamera", "ChairDamaskPurplegold", "SunglassesKhronos", "ProcTorus")],
          os.path.join(OUT, "thin_and_nonstar_cases.png"))
    md = ["# Specific artefacts\n", "## ellip (\"pile of scales\")\n", "Mean rank among the 9 phase-1 methods (1 = best):\n"]
    md += table(["budget", "composite", "F@0.05 alone"] + PM[:7], [[b, f"{ranks(b,'score',True,P1)['ellip']:.2f}", f"{ranks(b,'f05',True,P1)['ellip']:.2f}"] + [f"{ranks(b,c,False,P1)['ellip']:.2f}" for c in PM[:7]] for b in (64, 128, 256, 512)])
    md.append("\nModels (of 28) on which ellip beats voxs / beats mesh:\n")
    md += table(["budget", "metric", "ellip beats voxs", "ellip beats mesh"],
                [[b, c, sum((get(m, "ellip", b)[c] > get(m, "voxs", b)[c]) == hi for m in models), sum((get(m, "ellip", b)[c] > get(m, "mesh", b)[c]) == hi for m in models)]
                 for b in (128, 256, 512) for c, hi in (("score", True), ("crop_alex", False), ("crop_vgg", False), ("crop_dists", False), ("grey_vgg", False))])
    md.append("\n## Thin rods: v1 vox smooth vs cubes on thin-legged models (128 B)\n")
    md += table(["model", "composite smooth / cubes", "sil IoU smooth / cubes", "crop_alex smooth / cubes", "crop_vgg smooth / cubes", "crop_dists smooth / cubes"],
                [[m] + [f"{get(m,'vox1',128)[c]:.3f} / {get(m,'vox1c',128)[c]:.3f}" for c in ("score", "sil_iou", "crop_alex", "crop_vgg", "crop_dists")] for m in ("SheenChair", "ProcTable", "Lantern", "AntiqueCamera", "ChairDamaskPurplegold", "SunglassesKhronos")])
    md.append("\n## sh on shapes that are not star-shaped (128 B): rank of v1 sh among v1 vox / sh / mesh\n")
    md += table(["model", "composite: sh / vox / mesh", "crop_alex: sh / vox / mesh", "crop_vgg: sh / vox / mesh"],
                [[m] + [" / ".join(f"{get(m,k,128)[c]:.3f}" for k in ("sh1", "vox1", "mesh1")) for c in ("score", "crop_alex", "crop_vgg")] for m in ("ProcTable", "ProcTorus", "SheenChair", "SunglassesKhronos", "ChairDamaskPurplegold", "Lantern")])
    open(os.path.join(OUT, "artefacts.md"), "w").write("\n".join(md) + "\n")

    # ------------------------------------------------------------ selection
    md = ["# LPIPS as the union selection criterion (v1: vox smooth / sh / mesh)\n",
          "Candidates are the three strings the Python v1 encoder produces per model and budget. `shipped` here is the selection score the encoder had",
          "before this experiment: mean of F-score@0.05, colour term and 24x24 silhouette IoU. `new score` adds normal consistency (what ships now).",
          "Each alternative picks by one render metric. Judged by everything, including metrics not used to select.\n"]
    cand = ("vox1", "sh1", "mesh1")
    terms = {(r["model"], r["method"], r["budget"]): r for r in json.load(open(os.path.join(OUT, "selection_terms.json")))}
    sels = {"shipped": None, "new score (+NC)": "nc", "crop_alex": "crop_alex", "crop_vgg": "crop_vgg", "crop_dists": "crop_dists", "lp_alex": "lp_alex", "grey_vgg": "grey_vgg", "oracle composite": "score"}
    judge = ["score", "f05", "sil_iou", "ssim", "deltaE", "chamfer", "normal", "crop_alex", "crop_vgg", "crop_dists", "grey_alex"]
    half = {"all 28": models, "tuning half": models[0::2], "held-out half": models[1::2]}
    lines = []
    for b in (32,) + BUD + (512,):
        md.append(f"\n## {b} bytes\n")
        picks = {}
        for name, col in sels.items():
            picks[name] = {}
            for model in models:
                g = {k: get(model, k, b) for k in cand}
                g = {k: v for k, v in g.items() if v and "score" in v}
                if name == "shipped":
                    picks[name][model] = max(g, key=lambda k: sum(terms[(model, k, b)][t] for t in ("f", "col", "sil")) / 3)
                elif col == "nc":
                    picks[name][model] = max(g, key=lambda k: sum(terms[(model, k, b)][t] for t in ("f", "col", "sil", "nc")) / 4)
                elif col == "score":
                    picks[name][model] = max(g, key=lambda k: g[k]["score"])
                else:
                    picks[name][model] = min(g, key=lambda k: g[k][col])
        tab = []
        for name in sels:
            ch = sum(picks[name][m] != picks["shipped"][m] for m in models)
            c = Counter(picks[name].values())
            tab.append([name, f"{c['vox1']}/{c['sh1']}/{c['mesh1']}", ch] + [f"{np.mean([get(m, picks[name][m], b)[j] for m in models]) * (100 if j == 'chamfer' else 1):.3f}" for j in judge])
        md += table(["selector", "vox/sh/mesh", "picks changed vs shipped"] + judge, tab)
        md.append("\nHeld-out half only (14 models):\n")
        md += table(["selector"] + judge, [[name] + [f"{np.mean([get(m, picks[name][m], b)[j] for m in half['held-out half']]) * (100 if j == 'chamfer' else 1):.3f}" for j in judge] for name in sels])
        ch = [(m, picks["shipped"][m], picks["crop_alex"][m], picks["crop_vgg"][m], picks["crop_dists"][m]) for m in models if len({picks[k][m] for k in ("shipped", "crop_alex", "crop_vgg", "crop_dists")}) > 1]
        md.append("\nModels where any of alex / vgg / dists would change the pick:\n")
        md += table(["model", "shipped", "alex", "vgg", "dists", "composite shipped -> alex pick", "F@0.05 shipped -> alex pick", "sil IoU shipped -> alex pick"],
                    [[m, s, a, v, d] + [f"{get(m,s,b)[j]:.3f} -> {get(m,a,b)[j]:.3f}" for j in ("score", "f05", "sil_iou")] for m, s, a, v, d in ch])
        if b in BUD:
            for m, s, a, v, d in ch:
                lines.append((f"{m}\n{b} B", [("original", m, "orig"), (f"shipped: {s}", m, f"{s}_{b}"), (f"alex: {a}", m, f"{a}_{b}"), (f"vgg: {v}", m, f"{v}_{b}"), (f"dists: {d}", m, f"{d}_{b}")]))
    sheet(lines, os.path.join(OUT, "selection_changes.png"))
    md.append("\nSheet of every changed pick at 64 / 128 / 256 B: `selection_changes.png`.\n")
    open(os.path.join(OUT, "selection.md"), "w").write("\n".join(md) + "\n")
    print("wrote sanity.md agreement.md artefacts.md selection.md and sheets to", OUT)


if __name__ == "__main__":
    main()
