#!/usr/bin/env python3
"""LPIPS experiment, stage 4: candidates the encoder considered but did not keep.

For every model and budget this enumerates
  * vox: the best candidate under (1-w) * geometry + w * colour for a range of
    weights w (the encoder ships w = 0.5),
  * mesh: the best-fitting mesh for each position precision 5..8 bits,
and computes hand-written selection features for the three union candidates.
Renders go to results/lpips/cache_cand/, metadata to results/lpips/candidates.json.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json  # noqa: E402
import sys  # noqa: E402
import traceback  # noqa: E402
from multiprocessing import Pool  # noqa: E402

import numpy as np  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # works under python -I
from common import ROOT, find_models  # noqa: E402
from hash3d.asset import load_asset  # noqa: E402
from hash3d.v1 import decode  # noqa: E402
from hash3d.v1 import encode as E  # noqa: E402
from lpips_render import render_strings  # noqa: E402

BUDGETS = (64, 128, 256)
WEIGHTS = (0.0, 0.15, 0.3, 0.4, 0.5, 0.6, 0.7, 0.85, 1.0)
OUT = os.path.join(ROOT, "results", "lpips")


def sil(p, res):
    out = []
    for ax in range(3):
        o = [k for k in range(3) if k != ax]
        m = np.zeros((res, res), bool)
        ij = np.clip(np.floor((p[:, o] + 1.05) / 2.1 * res).astype(int), 0, res - 1)
        m[ij[:, 0], ij[:, 1]] = True
        out.append(m)
    return out


def features(src, h, n=30000):
    """Point-based quantities a network-free selection score could use."""
    d = decode(h)
    p, nr, c = E.sample_mesh(d.positions, d.indices.astype(np.int64), d.colors, n, 7)
    g, gn, gc = src.pts[100000:100000 + n], src.nrm[100000:100000 + n], src.col[100000:100000 + n]
    d1, i1 = cKDTree(p).query(g)
    d2, i2 = cKDTree(g).query(p)
    out = {}
    for tau in (0.05, 0.025, 0.0125):
        pr, rc = float((d2 < tau).mean()), float((d1 < tau).mean())
        out[f"f{tau}"] = 2 * pr * rc / max(pr + rc, 1e-9)
    out["chamfer"] = float(0.5 * (d1.mean() + d2.mean()))
    lg, lp = E.rgb_to_lab(gc), E.rgb_to_lab(c)
    de = 0.5 * (np.linalg.norm(lg - lp[i1], axis=1).mean() + np.linalg.norm(lp - lg[i2], axis=1).mean())
    out["col"] = float(1 - min(de, 40.0) / 40.0)
    out["nc"] = float(0.5 * (np.abs((gn * nr[i1]).sum(1)).mean() + np.abs((nr * gn[i2]).sum(1)).mean()))
    for res in (24, 48, 96):
        out[f"sil{res}"] = float(np.mean([(a & b).sum() / max((a | b).sum(), 1) for a, b in zip(sil(g, res), sil(p, res))]))
    out["shipped_score"] = float(E.score(src, d, full=True))
    return out


def run(args):
    name, path = args
    try:
        asset = load_asset(path, name)
        src = E.Source.from_asset(asset)
        meta, strings = [], {}
        for b in BUDGETS:
            cands = []
            shipped = E.encode_vox(src, b, collect=cands)
            for w in WEIGHTS:
                best = max(cands, key=lambda c: (1 - w) * c["g"] + w * c["c"])
                key = f"voxw{int(round(w * 100)):03d}_{b}"
                strings[key] = ("vox1", best["hash"])
                meta.append({"model": name, "budget": b, "kind": "vox", "w": w, "key": key, "shipped": best["hash"] == shipped,
                             **{k: best[k] for k in ("dims", "f", "cm", "g", "c")}, "ncand": len(cands)})
            mc = []
            ms = E.encode_mesh(src, b, collect=mc)
            for c in mc:
                key = f"meshpb{c['pb']}_{b}"
                strings[key] = ("mesh1", c["hash"])
                meta.append({"model": name, "budget": b, "kind": "mesh", "pb": c["pb"], "key": key, "j": c["j"], "shipped": c["hash"] == ms,
                             "verts": int(len(decode(c["hash"]).positions))})
            sh = E.encode_sh(src, b)
            for kind, h in (("vox1", shipped), ("sh1", sh), ("mesh1", ms)):
                if h is not None:
                    meta.append({"model": name, "budget": b, "kind": "union", "method": kind, "hash": h, **features(src, h)})
        render_strings(asset, strings, os.path.join(OUT, "cache_cand", name + ".npz"))
        return meta
    except Exception:  # noqa
        print(name, traceback.format_exc(), flush=True)
        return []


def main():
    models = find_models()
    meta = []
    with Pool(28) as pool:
        for m in pool.imap_unordered(run, models):
            meta += m
            if m:
                print(m[0]["model"], len(m), flush=True)
    json.dump(meta, open(os.path.join(OUT, "candidates.json"), "w"))


if __name__ == "__main__":
    main()
