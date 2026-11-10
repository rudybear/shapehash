#!/usr/bin/env python3
"""Evaluate every codec on every model at equal payload budgets.

    python -I scripts/run_eval.py [--budgets 32 64 128 256 512] [--models Duck Fox] [--jobs 16]

Writes results/raw/<model>.json (one row per model x method x budget, with the
actual hash string) and results/thumbs/<model>.npz (renders for the sheets).
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from multiprocessing import Pool  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # works under python -I
from common import RESULTS, find_models  # noqa: E402
from hash3d.asset import load_asset  # noqa: E402
from hash3d.codecs import REGISTRY  # noqa: E402
from hash3d.metrics import GroundTruth, evaluate  # noqa: E402
from hash3d.render import render_asset, render_mesh  # noqa: E402

RES = 160
THUMB_VIEWS = (0, 4)


def _thumb(renders):
    return np.stack([(renders[i][0] * 255).astype(np.uint8) for i in THUMB_VIEWS])


def run_model(args):
    name, path, budgets, methods = args
    rows, thumbs = [], {}
    try:
        asset = load_asset(path, name)
        ref = render_asset(asset, res=RES)
        gt = GroundTruth(asset, ref)
        thumbs["orig"] = _thumb(ref)
    except Exception:  # noqa
        return name, [{"model": name, "error": traceback.format_exc()}]
    for budget in budgets:
        for mname in methods:
            codec = REGISTRY[mname]
            row = {"model": name, "method": mname, "budget": budget}
            try:
                t = time.perf_counter()
                s = codec.encode(asset, budget)
                row["enc_s"] = time.perf_counter() - t
                nbytes = (len(s) * 3) // 4
                assert nbytes <= budget, f"over budget: {nbytes} > {budget}"
                t = time.perf_counter()
                cm = codec.decode(s)  # decoding sees ONLY the string
                row["dec_ms"] = (time.perf_counter() - t) * 1000
                row.update(hash=s, bytes=nbytes, chars=len(s))
                info = getattr(codec, "last_info", None)
                if info and mname in ("union1", "ts1"):
                    row["picked"] = info["method"]
                    row["sel_scores"] = info["scores"]
                rd = render_mesh(cm, res=RES)
                row.update(evaluate(gt, cm, rd))
                thumbs[f"{mname}_{budget}"] = _thumb(rd)
            except Exception:  # noqa
                row["error"] = traceback.format_exc()
            rows.append(row)
    os.makedirs(os.path.join(RESULTS, "raw"), exist_ok=True)
    os.makedirs(os.path.join(RESULTS, "thumbs"), exist_ok=True)
    with open(os.path.join(RESULTS, "raw", name + ".json"), "w") as f:
        json.dump(rows, f, indent=1)
    np.savez_compressed(os.path.join(RESULTS, "thumbs", name + ".npz"), **thumbs)
    return name, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budgets", type=int, nargs="+", default=[32, 64, 128, 256, 512])
    ap.add_argument("--models", nargs="*")
    ap.add_argument("--methods", nargs="*", default=[m for m in REGISTRY if not m.endswith("1") and m != "vox1c"])
    ap.add_argument("--jobs", type=int, default=min(16, os.cpu_count() or 1))
    a = ap.parse_args()
    models = [(n, p) for n, p in find_models() if not a.models or n in a.models]
    t0 = time.time()
    with Pool(a.jobs) as pool:
        for name, rows in pool.imap_unordered(run_model, [(n, p, a.budgets, a.methods) for n, p in models]):
            errs = [r for r in rows if "error" in r]
            print(f"[{time.time()-t0:6.1f}s] {name}: {len(rows)} rows, {len(errs)} errors", flush=True)
            for r in errs[:3]:
                print("   ", r.get("method"), r.get("budget"), r["error"].strip().splitlines()[-1])


if __name__ == "__main__":
    main()
