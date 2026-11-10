#!/usr/bin/env python3
"""Load every model, print stats, write results/originals.png (sanity check)."""
import os
import sys
import time

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # works under python -I
from common import RESULTS, find_models  # noqa: E402
from hash3d.asset import load_asset  # noqa: E402
from hash3d.render import render_asset  # noqa: E402


def main():
    os.makedirs(RESULTS, exist_ok=True)
    rows = []
    for name, path in find_models():
        t = time.time()
        try:
            a = load_asset(path, name)
        except Exception as e:  # noqa
            print(f"{name:28s} LOAD FAILED: {e}")
            continue
        r = render_asset(a, res=128)
        tex = sum(m[1] is not None for m in a.materials)
        print(f"{name:28s} F={len(a.faces):7d} parts={len(a.materials):3d} tex={tex:3d} ext={np.round(a.extents,2)} "
              f"col={np.round(a.mean_color(),2)} {time.time()-t:.1f}s")
        rows.append(np.concatenate([im for im, _ in r], axis=1))
    Image.fromarray((np.concatenate(rows, 0) * 255).astype(np.uint8)).save(os.path.join(RESULTS, "originals.png"))


if __name__ == "__main__":
    main()
