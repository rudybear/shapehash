#!/usr/bin/env python3
"""LPIPS experiment, stage 5: encoder-choice tuning and selection-score fitting
(results/lpips/tuning.md, selection_score.md and sheets)."""
import csv
import glob
import itertools
import json
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # works under python -I
import lpips_analyze as A  # noqa: E402
from common import ROOT  # noqa: E402

OUT = os.path.join(ROOT, "results", "lpips")
B = (64, 128, 256)
W = (0.0, 0.15, 0.3, 0.4, 0.5, 0.6, 0.7, 0.85, 1.0)
NETS = ("crop_alex", "crop_vgg", "crop_dists")


def readcsv(name):
    return {(r["model"], r["method"], int(r["budget"])): {k: float(v) for k, v in r.items() if k not in ("model", "method")} for r in csv.DictReader(open(os.path.join(OUT, name)))}


def main():
    meta = json.load(open(os.path.join(OUT, "candidates.json")))
    mc, m0 = readcsv("metrics_cand.csv"), readcsv("metrics.csv")
    models = sorted({r["model"] for r in meta})
    tune, held = models[0::2], models[1::2]
    wk = lambda w: f"voxw{int(round(w * 100)):03d}"  # noqa: E731
    md = ["# LPIPS to tune encoder choices\n",
          "Tuning half = every other model alphabetically starting with the first (14), held-out half = the rest (14).\n",
          "## vox: weight between geometry and colour in the grid / colour-mode search\n",
          "The encoder keeps the candidate (grid size, colour cell size, colour mode) maximising `(1-w) * F-score + w * colour term`; it ships w = 0.5.",
          "Each row re-selects with another w from the same candidate set, decodes (smooth) and renders. Mean over models, lower is better.\n"]
    for b in B:
        md.append(f"\n{b} bytes\n")
        lines = []
        for w in W:
            rs = [r for r in meta if r["kind"] == "vox" and r["budget"] == b and abs(r["w"] - w) < 1e-9]
            lines.append([f"{w:.2f}" + (" (shipped)" if w == 0.5 else "")] + [f"{np.mean([mc[(m, wk(w), b)][c] for m in ms]):.4f}" for ms in (tune, held) for c in NETS]
                         + [f"{np.mean([max(r['dims']) for r in rs]):.1f}", f"{np.mean([r['f'] for r in rs]):.2f}", sum(r["shipped"] for r in rs)])
        md += A.table(["w"] + [f"{h} {c}" for h in ("tune", "held-out") for c in NETS] + ["mean grid size", "mean colour cell", "same string as w=0.5 (of 28)"], lines)
        for c in NETS:
            # ties resolved towards the shipped w
            order = sorted(W, key=lambda w: abs(w - 0.5))
            best = Counter(min(order, key=lambda w: round(mc[(m, wk(w), b)][c], 6)) for m in models)
            gain = np.mean([mc[(m, wk(0.5), b)][c] - min(mc[(m, wk(w), b)][c] for w in W) for m in models])
            md.append(f"\nPer-model best w by `{c}` (ties go to 0.5): " + ", ".join(f"{k:.2f}: {v}" for k, v in sorted(best.items())) + f"; mean gain of a per-model oracle over w=0.5: {gain:.4f}")
    md.append("\n## mesh: position precision (5-8 bits per component, more bits = fewer vertices)\n")
    lines = []
    for b in B:
        rs = [r for r in meta if r["kind"] == "mesh" and r["budget"] == b]
        ship = Counter(r["pb"] for r in rs if r["shipped"])
        for c in NETS:
            pick, gain_t, gain_h, diff = Counter(), [], [], 0
            for m in models:
                cs = [r for r in rs if r["model"] == m]
                if not cs:
                    continue
                key = lambda r: mc[(m, r["key"].rsplit("_", 1)[0], b)][c]  # noqa: E731
                sh = [r for r in cs if r["shipped"]][0]
                bst = min(cs, key=lambda r: (round(key(r), 6), not r["shipped"]))
                pick[bst["pb"]] += 1
                diff += bst["pb"] != sh["pb"]
                (gain_t if m in tune else gain_h).append(key(sh) - key(bst))
            lines.append([b, c, " / ".join(str(ship[p]) for p in (5, 6, 7, 8)), " / ".join(str(pick[p]) for p in (5, 6, 7, 8)), diff, f"{np.mean(gain_t):.4f}", f"{np.mean(gain_h):.4f}"])
    md += A.table(["budget", "metric", "encoder keeps 5/6/7/8 bits", "metric prefers 5/6/7/8 bits", "models where they differ", "oracle gain, tune", "oracle gain, held-out"], lines)
    md.append("\nMean over models of each fixed precision:\n")
    md += A.table(["budget", "bits", "mean vertices", "crop_alex", "crop_vgg", "crop_dists"],
                  [[b, pb, f"{np.mean([r['verts'] for r in meta if r['kind']=='mesh' and r['budget']==b and r['pb']==pb]):.1f}"] + [f"{np.mean([mc[(m, f'meshpb{pb}', b)][c] for m in models if (m, f'meshpb{pb}', b) in mc]):.4f}" for c in NETS] for b in B for pb in (5, 6, 7, 8)])
    A.thumbs.cache = {}
    cache0 = os.path.join(OUT, "cache")
    lines = []
    for m in ("Duck", "ProcTorus", "CesiumMilkTruck", "Fox", "SheenChair", "BoxTextured"):
        lines.append((f"{m}\n128 B", [("original", m, "orig")] + [(f"w={w}", m, f"{wk(w)}_128") for w in (0.0, 0.15, 0.5, 0.85, 1.0)]))

    def thumbs_cand(model, key, views=(0, 4), size=128):
        z = thumbs_cand.c.get(model)
        if z is None:
            z = thumbs_cand.c[model] = dict(np.load(os.path.join(OUT, "cache_cand", model + ".npz")))
            z.update({k: v for k, v in np.load(os.path.join(cache0, model + ".npz")).items() if k.startswith("orig")})
        A.thumbs.cache[model] = z
        return orig_thumbs(model, key, views, size)

    thumbs_cand.c = {}
    orig_thumbs = A.thumbs
    A.thumbs = thumbs_cand
    A.thumbs.cache = orig_thumbs.cache
    A.sheet(lines, os.path.join(OUT, "tuning_vox_weight_128.png"))
    A.thumbs = orig_thumbs
    A.thumbs.cache = {}
    md.append("\nSheet: `tuning_vox_weight_128.png` (w = 0, 0.15, 0.5, 0.85, 1 for six models).\n")
    open(os.path.join(OUT, "tuning.md"), "w").write("\n".join(md) + "\n")

    # ------------------------------------------------ selection score fitting
    ev = {}
    for p in glob.glob(os.path.join(ROOT, "results", "v1", "raw", "*.json")):
        for r in json.load(open(p)):
            if "ssim" in r:
                ev[(r["model"], r["method"], r["budget"])] = r
    F = {(r["model"], r["method"], r["budget"]): r for r in meta if r["kind"] == "union"}
    feats = ["f0.05", "f0.025", "f0.0125", "col", "nc", "sil24", "sil48", "sil96"]

    def total(fn, ms):
        acc = {k: [] for k in ("alex", "vgg", "dists", "comp")}
        agree = 0
        for b in B:
            for m in ms:
                c = [k for k in ("vox1", "sh1", "mesh1") if (m, k, b) in F]
                pick = max(c, key=lambda k: fn(F[(m, k, b)]))
                agree += pick == min(c, key=lambda k: m0[(m, k, b)]["crop_alex"])
                acc["alex"].append(m0[(m, pick, b)]["crop_alex"])
                acc["vgg"].append(m0[(m, pick, b)]["crop_vgg"])
                acc["dists"].append(m0[(m, pick, b)]["crop_dists"])
                acc["comp"].append(A.score(ev[(m, pick, b)]))
        return [f"{np.mean(acc['alex']):.4f}", f"{agree}/{len(ms) * len(B)}", f"{np.mean(acc['vgg']):.4f}", f"{np.mean(acc['dists']):.4f}", f"{np.mean(acc['comp']):.4f}"]

    named = [("F@0.05 + colour + sil24 (before this experiment)", lambda r: r["f0.05"] + r["col"] + r["sil24"]),
             ("F@0.05 + colour + sil24 + normal consistency (ships now)", lambda r: r["f0.05"] + r["col"] + r["sil24"] + r["nc"]),
             ("F@0.025 + colour + sil24", lambda r: r["f0.025"] + r["col"] + r["sil24"]),
             ("F@0.025 + colour + sil48 + nc", lambda r: r["f0.025"] + r["col"] + r["sil48"] + r["nc"]),
             ("colour + sil48 + nc (no F-score)", lambda r: r["col"] + r["sil48"] + r["nc"]),
             ("sil48 only", lambda r: r["sil48"]), ("normal consistency only", lambda r: r["nc"]),
             ("always vox", lambda r: 1.0 if r["method"] == "vox1" else 0.0),
             ("LPIPS-alex itself (upper bound for agreement)", lambda r: -m0[(r["model"], r["method"], r["budget"])]["crop_alex"])]
    md = ["# A network-free selection score that follows LPIPS\n",
          "Features are point-based (30 000 samples per side): F-score at three thresholds, colour term, normal consistency, 3-view point-splat silhouette IoU at three",
          "resolutions. Each selector picks among v1 vox / sh / mesh at 64, 128 and 256 B; columns report the picked strings' mean crop LPIPS / DISTS, how often the pick equals",
          "the LPIPS-AlexNet pick, and the phase-1 composite. The form was chosen on the tuning half; the held-out half is the test.\n"]
    md += A.table(["selector", "tune alex", "tune = alex pick", "tune vgg", "tune dists", "tune composite", "held alex", "held = alex pick", "held vgg", "held dists", "held composite"],
                  [[n] + total(fn, tune) + total(fn, held) for n, fn in named])
    res = []
    for w in itertools.product((0, 1, 2, 3), repeat=len(feats)):
        if sum(w) == 0:
            continue
        fn = lambda r, w=w: sum(wi * r[f] for wi, f in zip(w, feats))  # noqa: E731
        t = total(fn, tune)
        res.append((float(t[0]), -int(t[1].split("/")[0]), sum(w), w))
    res.sort()
    md.append(f"\nExhaustive search over integer weights 0-3 for the 8 features ({len(res)} combinations), best on the tuning half (ties: fewest total weight):\n")
    md += A.table(["weights " + "/".join(feats), "tune alex", "tune = alex pick", "held alex", "held = alex pick", "held vgg", "held dists", "held composite"],
                  [["/".join(map(str, w))] + total(lambda r, w=w: sum(wi * r[f] for wi, f in zip(w, feats)), tune)[:2] + (lambda h: [h[0], h[1], h[2], h[3], h[4]])(total(lambda r, w=w: sum(wi * r[f] for wi, f in zip(w, feats)), held)) for _, _, _, w in res[:6]])
    open(os.path.join(OUT, "selection_score.md"), "w").write("\n".join(md) + "\n")
    print("wrote tuning.md, selection_score.md")


if __name__ == "__main__":
    main()
