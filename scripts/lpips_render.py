#!/usr/bin/env python3
"""LPIPS experiment, stage 1: render originals and every decoded placeholder.

For each model writes results/lpips/cache/<model>.npz with, per key, six
256x256 views: albedo (uint8 RGB), shade (uint8, the headlight factor) and
mask.  Keys: orig, orig_lp (texture low-passed on the surface), sphere,
<method>_<budget> for every hash string found in results/raw, results/v1/raw
and results/v1ts/raw.  Rendering = the harness renderer: orthographic, white
background, colour = albedo * (0.35 + 0.65 |n.v|), six fixed views.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import traceback  # noqa: E402
from multiprocessing import Pool  # noqa: E402

import numpy as np  # noqa: E402
import trimesh  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # works under python -I
from common import ROOT, find_models  # noqa: E402
from hash3d.asset import load_asset  # noqa: E402
from hash3d.codecs import REGISTRY  # noqa: E402
from hash3d.mesh import ColorMesh, icosphere  # noqa: E402
from hash3d.render import VIEWS, _rays, cast  # noqa: E402

RES = 256
BUDGETS = (32, 64, 128, 256, 512)
CACHE = os.path.join(ROOT, "results", "lpips", "cache")
SOURCES = (("raw", None), (os.path.join("v1", "raw"), None), (os.path.join("v1ts", "raw"), None))


def layers_mesh(cm):
    """-> albedo (V,H,W,3) u8, shade (V,H,W) u8, mask (V,H,W) bool"""
    A = np.full((len(VIEWS), RES * RES, 3), 255, np.uint8)
    S = np.zeros((len(VIEWS), RES * RES), np.uint8)
    M = np.zeros((len(VIEWS), RES * RES), bool)
    if len(cm.faces) == 0:
        return A.reshape(-1, RES, RES, 3), S.reshape(-1, RES, RES), M.reshape(-1, RES, RES)
    mesh = trimesh.Trimesh(cm.vertices, cm.faces, process=False)
    fn = cm.face_normals()
    vn = None if cm.flat else cm.vertex_normals()
    for vi, view in enumerate(VIEWS):
        o, dirs, d = _rays(view, RES)
        tri, bary = cast(mesh, o, dirs)
        hit = tri >= 0
        if not hit.any():
            continue
        f = cm.faces[tri[hit]]
        alb = np.einsum("nk,nkc->nc", bary[hit], cm.colors[f])
        if vn is None:
            n = fn[tri[hit]]
        else:
            n = np.einsum("nk,nkc->nc", bary[hit], vn[f])
            n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)
        A[vi, hit] = np.clip(alb * 255 + 0.5, 0, 255).astype(np.uint8)
        S[vi, hit] = np.clip((0.35 + 0.65 * np.abs(n @ d)) * 255 + 0.5, 0, 255).astype(np.uint8)
        M[vi, hit] = True
    return A.reshape(-1, RES, RES, 3), S.reshape(-1, RES, RES), M.reshape(-1, RES, RES)


def layers_asset(asset):
    """-> albedo, low-passed albedo, shade, mask for the original."""
    A = np.full((len(VIEWS), RES * RES, 3), 255, np.uint8)
    L = A.copy()
    S = np.zeros((len(VIEWS), RES * RES), np.uint8)
    M = np.zeros((len(VIEWS), RES * RES), bool)
    pts, _, col = asset.sample(40000, seed=5)
    tree = cKDTree(pts)
    for vi, view in enumerate(VIEWS):
        o, dirs, d = _rays(view, RES)
        tri, bary = cast(asset.mesh, o, dirs)
        hit = tri >= 0
        if not hit.any():
            continue
        alb = asset.color_at(tri[hit], bary[hit])
        n = np.einsum("nk,nkc->nc", bary[hit], asset.corner_nrm[tri[hit]])
        ln = np.linalg.norm(n, axis=1, keepdims=True)
        n = np.where(ln > 1e-9, n / np.maximum(ln, 1e-9), asset.face_normal[tri[hit]])
        p = np.einsum("nk,nkc->nc", bary[hit], asset.vertices[asset.faces[tri[hit]]])
        _, idx = tree.query(p, k=128)  # ~0.1 units radius: the colour resolution a placeholder can have
        A[vi, hit] = np.clip(alb * 255 + 0.5, 0, 255).astype(np.uint8)
        L[vi, hit] = np.clip(col[idx].mean(axis=1) * 255 + 0.5, 0, 255).astype(np.uint8)
        S[vi, hit] = np.clip((0.35 + 0.65 * np.abs(n @ d)) * 255 + 0.5, 0, 255).astype(np.uint8)
        M[vi, hit] = True
    sh = (-1, RES, RES)
    return A.reshape(sh + (3,)), L.reshape(sh + (3,)), S.reshape(sh), M.reshape(sh)


def render_strings(asset, strings, path):
    """strings: {key: (method, hash)}; writes the npz."""
    out = {}
    a, l, s, m = layers_asset(asset)
    out.update({"orig/a": a, "orig_lp/a": l, "orig/s": s, "orig/m": m})
    dirs, faces = icosphere(3)
    sph = ColorMesh(dirs * asset.extents, faces, np.tile(asset.mean_color(), (len(dirs), 1)))
    items = [("sphere", sph)]
    for key, (method, h) in strings.items():
        try:
            items.append((key, REGISTRY[method].decode(h)))
        except Exception:  # noqa
            print("decode failed", asset.name, key, traceback.format_exc().splitlines()[-1], flush=True)
    for key, cm in items:
        a, s, m = layers_mesh(cm)
        out.update({f"{key}/a": a, f"{key}/s": s, f"{key}/m": m})
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, **out)
    return len(items)


def run(args):
    name, path = args
    try:
        asset = load_asset(path, name)
        strings = {}
        for d, _ in SOURCES:
            p = os.path.join(ROOT, "results", d, name + ".json")
            if not os.path.exists(p):
                continue
            for r in json.load(open(p)):
                if "hash" in r and r["budget"] in BUDGETS:
                    strings[f"{r['method']}_{r['budget']}"] = (r["method"], r["hash"])
        extra = os.path.join(ROOT, "results", "lpips", "ts_before_strings.json")  # TypeScript picks before the selection-score change
        if os.path.exists(extra):
            strings.update({k: tuple(v) for k, v in json.load(open(extra)).get(name, {}).items()})
        n = render_strings(asset, strings, os.path.join(CACHE, name + ".npz"))
        return f"{name}: {n} meshes"
    except Exception:  # noqa
        return f"{name}: FAILED {traceback.format_exc()}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*")
    ap.add_argument("--jobs", type=int, default=28)
    a = ap.parse_args()
    models = [(n, p) for n, p in find_models() if not a.models or n in a.models]
    with Pool(a.jobs) as pool:
        for msg in pool.imap_unordered(run, models):
            print(msg, flush=True)


if __name__ == "__main__":
    main()
