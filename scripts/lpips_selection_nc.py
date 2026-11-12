#!/usr/bin/env python3
"""LPIPS experiment: re-run the union selection on the stored v1 candidate
strings with the encoder's current selection score, recording its terms.
Writes results/lpips/selection_terms.json."""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json  # noqa: E402
import sys  # noqa: E402
from multiprocessing import Pool  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # works under python -I
from common import ROOT, find_models  # noqa: E402
from hash3d.asset import load_asset  # noqa: E402
from hash3d.v1 import decode  # noqa: E402
from hash3d.v1 import encode as E  # noqa: E402


def run(args):
    name, path = args
    src = E.Source.from_asset(load_asset(path, name))
    out = []
    for r in json.load(open(os.path.join(ROOT, "results", "v1", "raw", name + ".json"))):
        if r["method"] in ("vox1", "sh1", "mesh1") and "hash" in r:
            parts = {}
            E.score(src, decode(r["hash"]), full=True, parts=parts)
            out.append({"model": name, "budget": r["budget"], "method": r["method"], **parts})
    return out


def main():
    rows = []
    with Pool(28) as pool:
        for r in pool.imap_unordered(run, find_models()):
            rows += r
    json.dump(rows, open(os.path.join(ROOT, "results", "lpips", "selection_terms.json"), "w"))
    print(len(rows), "rows")


if __name__ == "__main__":
    main()
