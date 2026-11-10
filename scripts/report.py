#!/usr/bin/env python3
"""Aggregate results/raw/*.json -> results/results.csv, results/summary.md,
results/plots/*.png and results/examples/ (decoded GLBs + their hash strings)."""
import csv
import glob
import json
import os
import sys
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # works under python -I
from common import RESULTS  # noqa: E402
from hash3d.codecs import REGISTRY, decode_any  # noqa: E402
from hash3d.mesh import export_glb  # noqa: E402

METHODS = ["box", "impostor", "kdop", "sh", "dct", "ellip", "mesh", "vox", "voxs"]
# fixed categorical order (colour follows the method, never its rank); baselines are neutral
COLORS = {"kdop": "#2a78d6", "sh": "#eb6834", "dct": "#1baf7a", "ellip": "#eda100", "mesh": "#e87ba4", "vox": "#008300", "voxs": "#6250d6",
          "box": "#b5b4ab", "impostor": "#7d7c75", "auto": "#1a1a19"}
MARKERS = {"kdop": "v", "sh": "o", "dct": "s", "ellip": "D", "mesh": "^", "vox": "P", "voxs": "X", "box": ".", "impostor": ".", "auto": "*"}
METRICS = [  # key, label, higher is better, format
    ("chamfer", "Chamfer x100 (lower)", False, lambda v: f"{v*100:.2f}"),
    ("f05", "F-score@0.05", True, lambda v: f"{v:.3f}"),
    ("normal", "Normal cons.", True, lambda v: f"{v:.3f}"),
    ("viou", "Vol. IoU", True, lambda v: f"{v:.3f}"),
    ("deltaE", "Colour dE (lower)", False, lambda v: f"{v:.1f}"),
    ("sil_iou", "Silhouette IoU", True, lambda v: f"{v:.3f}"),
    ("psnr", "PSNR dB", True, lambda v: f"{v:.2f}"),
    ("psnr_blur", "PSNR-blur dB", True, lambda v: f"{v:.2f}"),
    ("ssim", "SSIM", True, lambda v: f"{v:.3f}"),
    ("score", "Score", True, lambda v: f"{v:.3f}"),
]


def placeholder_score(r):
    """Ad-hoc composite in [0,1] used for ranking and for the per-model 'auto'
    selector: mean of F-score@0.05, silhouette IoU, SSIM and a colour term."""
    if "ssim" not in r:
        return 0.0
    return 0.25 * (r["f05"] + r["sil_iou"] + r["ssim"] + (1 - min(r["deltaE"], 40.0) / 40.0))


def load():
    rows = [r for p in sorted(glob.glob(os.path.join(RESULTS, "raw", "*.json"))) for r in json.load(open(p))]
    for r in rows:
        if "error" in r:
            print("ERROR row:", r.get("model"), r.get("method"), r.get("budget"), r["error"].strip().splitlines()[-1])
        r["score"] = placeholder_score(r)
        if "ssim" not in r:  # empty decode: count as total failure rather than dropping the row
            r.update(chamfer=1.0, f05=0.0, normal=0.0, viou=0.0, deltaE=40.0, sil_iou=0.0, psnr=5.0, psnr_blur=5.0, ssim=0.0, failed=True)
            r.setdefault("bytes", 0), r.setdefault("chars", 0), r.setdefault("enc_s", 0), r.setdefault("dec_ms", 0), r.setdefault("tris", 0)
    return rows


def add_auto(rows):
    by = defaultdict(list)
    for r in rows:
        by[(r["model"], r["budget"])].append(r)
    out = []
    for (model, budget), rs in by.items():
        best = max(rs, key=lambda r: r["score"])
        a = dict(best)
        a["picked"] = best["method"]
        a["method"] = "auto"
        # an auto encoder runs every encoder
        a["enc_s"] = sum(r.get("enc_s", 0) for r in rs)
        out.append(a)
    return out


def mean(rs, k):
    return float(np.mean([r[k] for r in rs]))


def main():
    rows = load()
    auto = add_auto(rows)
    allrows = rows + auto
    budgets = sorted({r["budget"] for r in rows})
    models = sorted({r["model"] for r in rows})
    methods = [m for m in METHODS if any(r["method"] == m for r in rows)] + ["auto"]
    os.makedirs(os.path.join(RESULTS, "plots"), exist_ok=True)

    keys = ["model", "method", "budget", "bytes", "chars", "enc_s", "dec_ms", "tris", "verts"] + [m[0] for m in METRICS] + ["f10", "rgb_rmse", "picked", "hash"]
    with open(os.path.join(RESULTS, "results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(sorted(allrows, key=lambda r: (r["model"], r["budget"], r["method"])))

    sel = lambda m, b: [r for r in allrows if r["method"] == m and r["budget"] == b]  # noqa: E731
    md = []
    md.append(f"Models: {len(models)} ({', '.join(models)})\n")
    for b in budgets:
        md.append(f"\n#### Budget {b} bytes ({-(-b*4//3)} base64url chars) - mean over {len(models)} models\n")
        md.append("| method | bytes used | " + " | ".join(m[1] for m in METRICS) + " | wins | tris | enc s | dec ms |")
        md.append("|---|---|" + "---|" * (len(METRICS) + 4))
        wins = defaultdict(int)
        for a in auto:
            if a["budget"] == b:
                wins[a["picked"]] += 1
        table = {m: sel(m, b) for m in methods}
        best = {}
        for k, _, hi, _ in METRICS:
            vals = {m: mean(table[m], k) for m in methods if m != "auto"}
            best[k] = (max if hi else min)(vals, key=vals.get)
        for m in methods:
            rs = table[m]
            cells = []
            for k, _, hi, fmt in METRICS:
                s = fmt(mean(rs, k))
                cells.append(f"**{s}**" if best[k] == m else s)
            md.append(f"| {m} | {mean(rs,'bytes'):.0f} | " + " | ".join(cells) + f" | {wins[m] if m != 'auto' else '-'} | {mean(rs,'tris'):.0f} | {mean(rs,'enc_s'):.2f} | {mean(rs,'dec_ms'):.1f} |")
    md.append("\n#### Which method `auto` picks (count of models per budget)\n")
    md.append("| budget | " + " | ".join(methods[:-1]) + " |")
    md.append("|---|" + "---|" * (len(methods) - 1))
    for b in budgets:
        c = defaultdict(int)
        for a in auto:
            if a["budget"] == b:
                c[a["picked"]] += 1
        md.append(f"| {b} | " + " | ".join(str(c[m]) for m in methods[:-1]) + " |")
    for b in budgets:
        md.append(f"\n#### Per-model composite score at {b} bytes\n")
        md.append("| model | " + " | ".join(methods[:-1]) + " | best |")
        md.append("|---|" + "---|" * len(methods))
        for model in models:
            sc = {r["method"]: r["score"] for r in rows if r["model"] == model and r["budget"] == b}
            bm = max(sc, key=sc.get)
            md.append(f"| {model} | " + " | ".join((f"**{sc[m]:.3f}**" if m == bm else f"{sc[m]:.3f}") for m in methods[:-1]) + f" | {bm} |")
    md.append("\n#### Mean composite score by budget\n")
    md.append("| method | " + " | ".join(f"{b} B" for b in budgets) + " |")
    md.append("|---|" + "---|" * len(budgets))
    for m in methods:
        md.append(f"| {m} | " + " | ".join(f"{mean(sel(m,b),'score'):.3f}" for b in budgets) + " |")
    md.append("\n#### Connectivity / payload accounting\n")
    md.append("| budget | mesh verts | mesh tris | sh/dct/vox notes |")
    md.append("|---|---|---|---|")
    for b in budgets:
        ms = sel("mesh", b)
        md.append(f"| {b} | {mean(ms,'verts'):.1f} | {mean(ms,'tris'):.1f} | vox bytes used {mean(sel('vox',b),'bytes'):.0f}, dct {mean(sel('dct',b),'bytes'):.0f}, sh {mean(sel('sh',b),'bytes'):.0f} |")
    with open(os.path.join(RESULTS, "summary.md"), "w") as f:
        f.write("\n".join(md) + "\n")

    # ---- plots: small multiples, one metric per panel, shared categorical colours
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e5df", "grid.linewidth": 0.6,
                         "axes.edgecolor": "#8a897f", "axes.labelcolor": "#3b3b37", "xtick.color": "#3b3b37", "ytick.color": "#3b3b37"})
    panels = [("score", "Composite placeholder score (higher is better)"), ("chamfer", "Chamfer distance x100 (lower is better)"), ("f05", "F-score @ 0.05 (higher is better)"),
              ("sil_iou", "Silhouette IoU (higher is better)"), ("ssim", "Render SSIM (higher is better)"), ("psnr_blur", "Blurred-render PSNR, dB (higher is better)"),
              ("deltaE", "Colour error, CIELAB dE (lower is better)"), ("viou", "Volumetric IoU (higher is better)"), ("normal", "Normal consistency (higher is better)")]
    fig, axes = plt.subplots(3, 3, figsize=(15, 12.5))
    for ax, (k, title) in zip(axes.ravel(), panels):
        for m in methods:
            ys = [mean(sel(m, b), k) * (100 if k == "chamfer" else 1) for b in budgets]
            base = m in ("box", "impostor")
            ax.plot(budgets, ys, color=COLORS[m], marker=MARKERS[m], markersize=7 if not base else 5, linewidth=2 if not base else 1.2,
                    linestyle="--" if m == "auto" else ("-" if not base else ":"), label=m, markeredgecolor="white", markeredgewidth=0.8, zorder=3 if not base else 2)
        if k == "chamfer":
            ax.set_ylim(0, 22)
        if k in ("score", "sil_iou", "ssim", "normal"):
            ax.set_ylim(0.3, None)
        if k == "psnr_blur":
            ax.set_ylim(10, None)
        ax.set_xscale("log", base=2)
        ax.set_xticks(budgets)
        ax.set_xticklabels([str(b) for b in budgets])
        ax.set_title(title, loc="left", fontsize=10, color="#1a1a19")
        ax.set_xlabel("payload budget (bytes)")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper left", bbox_to_anchor=(0.005, 0.972), ncol=len(methods), frameon=False, fontsize=10)
    fig.suptitle(f"hash3d: quality vs payload size, mean over {len(models)} models (mesh cannot fit anything in 16 B: off-scale)", y=0.995, fontsize=12, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(os.path.join(RESULTS, "plots", "quality_vs_bytes.png"), dpi=110)
    plt.close(fig)

    # per-model score heat table at each budget would be large; instead: score spread per method
    fig, axes = plt.subplots(1, len(budgets), figsize=(3.4 * len(budgets), 4.2), sharey=True)
    for ax, b in zip(np.atleast_1d(axes), budgets):
        ms = methods[:-1]
        for i, m in enumerate(ms):
            ys = [r["score"] for r in sel(m, b)]
            ax.scatter(np.full(len(ys), i) + np.linspace(-0.18, 0.18, len(ys)), sorted(ys), s=14, color=COLORS[m], edgecolor="white", linewidth=0.4, zorder=3)
            ax.plot([i - 0.3, i + 0.3], [np.mean(ys)] * 2, color="#1a1a19", linewidth=1.6, zorder=4)
        ax.set_xticks(range(len(ms)))
        ax.set_xticklabels(ms, rotation=60, ha="right")
        ax.set_title(f"{b} bytes", loc="left", fontsize=10)
    np.atleast_1d(axes)[0].set_ylabel("composite score per model (bar = mean)")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "plots", "score_spread.png"), dpi=110)
    plt.close(fig)

    # ---- examples: decoded GLBs with their hash strings
    ex = os.path.join(RESULTS, "examples")
    os.makedirs(ex, exist_ok=True)
    lines = ["# Example hashes (decode with: python -I scripts/hash3d_cli.py decode <hash> out.glb)\n"]
    want = [m for m in ("Duck", "Fox", "SheenChair", "DamagedHelmet", "Lantern", "CesiumMilkTruck") if m in models]
    for model in want:
        for b in budgets:
            cands = [r for r in rows if r["model"] == model and r["budget"] == b and "hash" in r]
            picks = {"auto": max(cands, key=lambda r: r["score"])}
            for m in ("voxs", "vox", "sh", "mesh"):
                picks.update({m: r for r in cands if r["method"] == m})
            done = set()
            for label, r in picks.items():
                if label != "auto" and b not in (64, 256):
                    continue
                if (r["method"]) in done:
                    continue
                done.add(r["method"])
                fn = f"{model}_{b:03d}B_{r['method']}.glb"
                export_glb(decode_any(r["hash"]), os.path.join(ex, fn))
                lines.append(f"- `{fn}` ({r['bytes']} bytes, {r['chars']} chars, method `{r['method']}`{', picked by auto' if label == 'auto' else ''}, score {r['score']:.3f})\n\n  `{r['hash']}`\n")
    with open(os.path.join(ex, "HASHES.md"), "w") as f:
        f.write("\n".join(lines))
    print("\n".join(md[: 14 + 0]))
    print("wrote summary.md, results.csv, plots/, examples/")


if __name__ == "__main__":
    main()
