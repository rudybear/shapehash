#!/usr/bin/env python3
"""LPIPS experiment, stage 2: perceptual metrics on the cached renders.

    TORCH_HOME=$PWD/.cache/torch python -I scripts/lpips_metrics.py [cache_dir] [out.csv]

Needs torch, lpips and piq (not in requirements.txt: only this experiment uses
them).  Variants (all: mean over the six views, lower is better):
  full   textured original vs placeholder, whole 256 px frame, white background
  crop   same, cropped to the union bounding box of both silhouettes, resized to 224 px
  lp     crop, original rendered with its texture low-passed on the surface (~0.1 units)
  blur   crop, original image Gaussian-blurred (sigma 4 px) - the 2-D experiment's fix
  grey   crop, both rendered with a uniform grey albedo (shape and shading only)
Networks: LPIPS-AlexNet, LPIPS-VGG, DISTS.
"""
import csv
import glob
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # works under python -I
from common import ROOT  # noqa: E402

VARIANTS = ("full", "crop", "lp", "blur", "grey")
NETS = ("alex", "vgg", "dists")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
_nets = {}


def nets():
    if not _nets:
        import lpips
        import piq

        _nets["alex"] = lpips.LPIPS(net="alex", verbose=False).to(DEV).eval()
        _nets["vgg"] = lpips.LPIPS(net="vgg", verbose=False).to(DEV).eval()
        _nets["dists"] = piq.DISTS(reduction="none").to(DEV).eval()
    return _nets


@torch.no_grad()
def distances(a, b, bs=192):
    """a, b: (N,3,H,W) float in [0,1] on DEV -> {net: (N,) numpy}"""
    out = {k: [] for k in NETS}
    n = nets()
    for i in range(0, len(a), bs):
        x, y = a[i:i + bs], b[i:i + bs]
        out["alex"].append(n["alex"](x * 2 - 1, y * 2 - 1).flatten())
        out["vgg"].append(n["vgg"](x * 2 - 1, y * 2 - 1).flatten())
        out["dists"].append(n["dists"](x, y).flatten())
    return {k: torch.cat(v).cpu().numpy() for k, v in out.items()}


def compose(alb, shade, mask, grey=False):
    """-> (V,3,H,W) float tensor, white background."""
    a = torch.from_numpy(np.ascontiguousarray(alb)).to(DEV).float() / 255
    s = torch.from_numpy(np.ascontiguousarray(shade)).to(DEV).float() / 255
    m = torch.from_numpy(np.ascontiguousarray(mask)).to(DEV)
    if grey:
        a = torch.full_like(a, 0.7)
    img = a * s[..., None]
    img = torch.where(m[..., None], img, torch.ones_like(img))
    return img.permute(0, 3, 1, 2)


def crop_boxes(m0, m1, pad=0.06):
    """Square crop boxes around the union of two mask stacks (V,H,W)."""
    boxes = []
    H, W = m0.shape[1:]
    for a, b in zip(m0, m1):
        u = a | b
        if not u.any():
            boxes.append((0, 0, W, H))
            continue
        ys, xs = np.nonzero(u)
        cx, cy = (xs.min() + xs.max() + 1) / 2, (ys.min() + ys.max() + 1) / 2
        half = max(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1) / 2 * (1 + pad) + 1
        half = min(half, W / 2, H / 2)
        cx, cy = min(max(cx, half), W - half), min(max(cy, half), H - half)
        boxes.append((int(round(cx - half)), int(round(cy - half)), int(round(cx + half)), int(round(cy + half))))
    return boxes


def crop(img, boxes, size=224):
    return torch.cat([F.interpolate(img[i:i + 1, :, y0:y1, x0:x1], size=(size, size), mode="bilinear", antialias=True, align_corners=False)
                      for i, (x0, y0, x1, y1) in enumerate(boxes)])


def blur(img, sigma=4.0):
    k = int(6 * sigma) | 1
    x = torch.arange(k, device=img.device) - k // 2
    g = torch.exp(-x.float() ** 2 / (2 * sigma * sigma))
    g = (g / g.sum()).view(1, 1, 1, k)
    c = img.shape[1]
    img = F.conv2d(F.pad(img, (k // 2, k // 2, 0, 0), mode="replicate"), g.expand(c, 1, 1, k), groups=c)
    return F.conv2d(F.pad(img, (0, 0, k // 2, k // 2), mode="replicate"), g.transpose(2, 3).expand(c, 1, k, 1), groups=c)


def pairs_for(z, key):
    """-> {variant: (ref, test)} tensors for one cached mesh."""
    om, os_ = z["orig/m"], z["orig/s"]
    if key == "blank":
        da, ds, dm = np.full_like(z["orig/a"], 255), np.zeros_like(os_), np.zeros_like(om)
    elif key == "orig_lp":
        da, ds, dm = z["orig_lp/a"], os_, om
    elif key == "orig_flat":
        mean = z["orig/a"][om].mean(axis=0)
        da, ds, dm = np.broadcast_to(mean.astype(np.uint8), z["orig/a"].shape), os_, om
    else:
        da, ds, dm = z[key + "/a"], z[key + "/s"], z[key + "/m"]
    ref, ref_lp, ref_g = compose(z["orig/a"], os_, om), compose(z["orig_lp/a"], os_, om), compose(z["orig/a"], os_, om, grey=True)
    tst, tst_g = compose(da, ds, dm), compose(da, ds, dm, grey=True)
    bx = crop_boxes(om, dm)
    rc, tc = crop(ref, bx), crop(tst, bx)
    cover = float(np.mean([(m.sum() / m.size) for m in om]))
    return {"full": (ref, tst), "crop": (rc, tc), "lp": (crop(ref_lp, bx), tc), "blur": (blur(rc), tc), "grey": (crop(ref_g, bx), crop(tst_g, bx))}, cover


def measure(z, keys):
    """-> {key: {variant_net: value}}"""
    refs = {v: [] for v in VARIANTS}
    tsts = {v: [] for v in VARIANTS}
    cover = 0.0
    for key in keys:
        p, cover = pairs_for(z, key)
        for v in VARIANTS:
            refs[v].append(p[v][0])
            tsts[v].append(p[v][1])
    out = {k: {"coverage": cover} for k in keys}
    nv = z["orig/m"].shape[0]
    for v in VARIANTS:
        d = distances(torch.cat(refs[v]), torch.cat(tsts[v]))
        for net, vals in d.items():
            vals = vals.reshape(len(keys), nv).mean(axis=1)
            for k, x in zip(keys, vals):
                out[k][f"{v}_{net}"] = float(x)
    return out


def main():
    cache = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "results", "lpips", "cache")
    dst = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "results", "lpips", "metrics.csv")
    rows = []
    for p in sorted(glob.glob(os.path.join(cache, "*.npz"))):
        model = os.path.basename(p)[:-4]
        z = dict(np.load(p))
        keys = sorted({k.split("/")[0] for k in z if k.endswith("/s") and not k.startswith("orig")}) + ["blank", "orig_lp", "orig_flat"]
        res = measure(z, keys)
        for k, vals in res.items():
            method, _, budget = k.rpartition("_")
            if not budget.isdigit():
                method, budget = k, "0"
            rows.append({"model": model, "method": method, "budget": int(budget), **vals})
        print(model, len(keys), flush=True)
    cols = ["model", "method", "budget", "coverage"] + [f"{v}_{n}" for v in VARIANTS for n in NETS]
    with open(dst, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print("wrote", dst, len(rows), "rows on", DEV)


if __name__ == "__main__":
    main()
