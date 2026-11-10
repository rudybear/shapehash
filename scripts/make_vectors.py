#!/usr/bin/env python3
"""Generate cross-validation vectors for the TypeScript implementation.

Reads hashes produced by the Python encoder (results/v1/raw/*.json), decodes
them with the Python reference and writes ts/test/vectors/vectors.json.
"""
import base64
import glob
import hashlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # works under python -I
from common import RESULTS, ROOT  # noqa: E402
from hash3d.v1 import decode, parse  # noqa: E402


def sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def b64(a):
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()


def mesh_record(d, full=False):
    """Exact digests always; full float arrays only where a tolerance applies (sh)."""
    rec = {"nverts": int(len(d.positions)), "ntris": int(len(d.indices)), "flat": bool(d.flat),
           "positions_sha": sha(d.positions.astype("<f4")), "normals_sha": sha(d.normals.astype("<f4")),
           "indices_sha": sha(d.indices.astype("<u4")), "colors_sha": sha(d.colors.astype(np.uint8))}
    if full:
        rec["positions"] = b64(d.positions.astype("<f4"))
        rec["colors"] = b64(d.colors.astype(np.uint8))
    return rec


def record(h, meta):
    p = parse(h)
    rec = {"hash": h, **meta, "method": int(p["method"]), "extents_q": [int(v) for v in p["extents_q"]]}
    if p["method"] == 0:
        rec["vox"] = {"dims": [int(v) for v in p["dims"]], "f": int(p["f"]), "color_mode": int(p["color_mode"]),
                      "occ_sha": sha(p["occ"].astype(np.uint8)), "active_sha": sha(p["active"].astype(np.uint8)),
                      "cell_colors_sha": sha(p["cell_colors"].astype(np.uint8))}
        rec["smooth"] = mesh_record(decode(h, vox="smooth"))
        rec["cubes"] = mesh_record(decode(h, vox="cubes"))
    elif p["method"] == 1:
        rec["sh"] = {"L": p["L"], "Ly": p["Ly"], "Lc": p["Lc"], "center_q": p["center_q"],
                     "groups": [{"dc": g["dc"], "scale_q": g["scale_q"], "ac": [int(v) for v in g["ac"]]} for g in p["groups"]]}
        rec["mesh"] = mesh_record(decode(h), full=True)
    else:
        rec["meshc"] = {"pos_bits": int(p["pos_bits"]), "pos_q_sha": sha(p["pos_q"].astype("<u2")), "col_q_sha": sha(p["col_q"].astype(np.uint8)),
                        "tris_sha": sha(p["tris"].astype("<u4"))}
        rec["mesh"] = mesh_record(decode(h))
    return rec


def main():
    src = os.path.join(RESULTS, "v1", "raw") if os.path.isdir(os.path.join(RESULTS, "v1", "raw")) else os.path.join(RESULTS, "raw")
    rows = [r for p in sorted(glob.glob(os.path.join(src, "*.json"))) for r in json.load(open(p)) if "hash" in r]
    want_models = ["Duck", "Fox", "SheenChair", "DamagedHelmet", "Lantern", "BoxVertexColors", "ProcTorus", "SunglassesKhronos", "CesiumMilkTruck", "Avocado"]
    out, seen, modes = [], set(), set()
    for r in rows:
        if r["method"] not in ("vox1", "sh1", "mesh1") or r["hash"] in seen:
            continue
        big = r["budget"] > 128
        if r["model"] not in want_models or (big and r["model"] not in want_models[:3]):
            continue
        seen.add(r["hash"])
        out.append(record(r["hash"], {"model": r["model"], "budget": r["budget"]}))
        if out[-1]["method"] == 0:
            modes.add(out[-1]["vox"]["color_mode"])
    # make sure every vox colour mode that occurs anywhere is covered
    for r in sorted(rows, key=lambda r: r["budget"]):
        if r["method"] == "vox1" and r["hash"] not in seen:
            cm = parse(r["hash"])["color_mode"]
            if cm not in modes:
                modes.add(cm)
                seen.add(r["hash"])
                out.append(record(r["hash"], {"model": r["model"], "budget": r["budget"]}))
    d = os.path.join(ROOT, "ts", "test", "vectors")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "vectors.json"), "w") as f:
        json.dump({"format": "shapehash-v1", "float_tolerance": 1e-5, "vectors": out}, f)
    print(f"{len(out)} vectors, vox colour modes covered: {sorted(modes)}, {os.path.getsize(os.path.join(d, 'vectors.json'))/1e6:.1f} MB")


if __name__ == "__main__":
    main()
