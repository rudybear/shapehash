#!/usr/bin/env python3
"""Cross-validate strings produced by the TypeScript encoder.

Every candidate string in results/v1/ts_encoded.json (written by
`node ts/dist/cli.js batch`) is decoded by the Python reference and the
result is compared with the digests of the TypeScript decode stored next to
it.  Exit status 1 on any mismatch.
"""
import hashlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # works under python -I
from common import RESULTS  # noqa: E402
from hash3d.v1 import decode  # noqa: E402


def sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(RESULTS, "v1", "ts_encoded.json")
    rows = json.load(open(path))
    n = bad = over = 0
    stats = {}
    for r in rows:
        if "error" in r:
            print("TS encode error:", r["model"], r["error"])
            bad += 1
            continue
        for method, h in r["candidates"].items():
            if (len(h) * 3) // 4 > r["budget"]:
                over += 1
            for mode in ("smooth", "cubes"):
                d = decode(h, vox=mode)
                exp = r["digests"][method][mode]
                got = {"nverts": len(d.positions), "ntris": len(d.indices), "positions_sha": sha(d.positions.astype("<f4")), "indices_sha": sha(d.indices.astype("<u4")),
                       "colors_sha": sha(d.colors.astype(np.uint8)), "normals_sha": sha(d.normals.astype("<f4"))}
                assert d.method == method, (d.method, method)
                ok = got == exp
                s = stats.setdefault(method, [0, 0])
                s[0] += 1
                s[1] += ok
                n += 1
                if not ok:
                    bad += 1
                    print("MISMATCH", r["model"], r["budget"], method, mode, {k: (got[k], exp[k]) for k in got if got[k] != exp[k]})
    print(f"{n} decodes of TypeScript-encoded strings checked against the Python reference; bit-identical per method: "
          + ", ".join(f"{k} {v[1]}/{v[0]}" for k, v in sorted(stats.items())) + f"; over budget: {over}; failures: {bad}")
    return 1 if bad or over else 0


if __name__ == "__main__":
    sys.exit(main())
