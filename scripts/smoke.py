#!/usr/bin/env python3
"""Quick round-trip smoke test: python -I scripts/smoke.py [Model] [budget...]"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # works under python -I
from common import find_models  # noqa: E402
from hash3d.asset import load_asset  # noqa: E402
from hash3d.codecs import REGISTRY  # noqa: E402
from hash3d.metrics import GroundTruth, evaluate  # noqa: E402
from hash3d.render import render_asset  # noqa: E402


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "Duck"
    budgets = [int(x) for x in sys.argv[2:]] or [32, 128, 512]
    asset = load_asset(dict(find_models())[name], name)
    gt = GroundTruth(asset, render_asset(asset))
    only = os.environ.get("METHODS", "").split(",") if os.environ.get("METHODS") else None
    for b in budgets:
        for cname, codec in REGISTRY.items():
            if only and cname not in only:
                continue
            t = time.time()
            s = codec.encode(asset, b)
            te = time.time() - t
            t = time.time()
            cm = codec.decode(s)
            td = time.time() - t
            m = evaluate(gt, cm)
            print(f"{b:4d} {cname:9s} {(len(s)*3)//4:4d}B enc {te:6.2f}s dec {td*1000:7.1f}ms tris {m['tris']:6d} "
                  + " ".join(f"{k}={m[k]:.3f}" for k in ("chamfer", "f05", "normal", "viou", "deltaE", "sil_iou", "psnr", "ssim") if k in m))


if __name__ == "__main__":
    main()
