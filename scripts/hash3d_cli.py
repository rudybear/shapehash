#!/usr/bin/env python3
"""Python reference command line (the production CLI is the TypeScript one in ts/).

    python -I scripts/hash3d_cli.py encode model.glb --budget 128 [--method auto|vox|sh|mesh]
    python -I scripts/hash3d_cli.py decode <hash> out.glb [--cubes]

ShapeHash v1 by default.  --legacy selects the phase-1 exploratory codecs
(method names box, impostor, kdop, sh, dct, ellip, mesh, vox, voxs).
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # works under python -I
import common  # noqa: E402,F401
from hash3d.asset import load_asset  # noqa: E402
from hash3d.mesh import export_glb  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("encode")
    e.add_argument("model")
    e.add_argument("--budget", type=int, default=128, help="payload bytes (string is 4/3 of that)")
    e.add_argument("--method", default="auto")
    e.add_argument("--legacy", action="store_true")
    d = sub.add_parser("decode")
    d.add_argument("hash")
    d.add_argument("out")
    d.add_argument("--cubes", action="store_true", help="vox payloads: cube faces instead of surface nets")
    d.add_argument("--legacy", action="store_true")
    a = ap.parse_args()
    if a.cmd == "encode":
        asset = load_asset(a.model)
        if a.legacy:
            from hash3d.codecs import REGISTRY

            s, m = REGISTRY[a.method].encode(asset, a.budget), a.method
        else:
            from hash3d.v1 import encode as v1e

            s, info = v1e.encode(v1e.Source.from_asset(asset), a.budget, a.method)
            m = info["method"]
        print(s)
        print(f"# method={m} payload={(len(s) * 3) // 4} bytes, {len(s)} chars", file=sys.stderr)
    else:
        if a.legacy:
            from hash3d.codecs import decode_any

            cm = decode_any(a.hash)
        else:
            from hash3d.codecs.v1 import to_colormesh
            from hash3d.v1 import decode

            cm = to_colormesh(decode(a.hash, vox="cubes" if a.cubes else "smooth"))
        export_glb(cm, a.out)
        print(f"wrote {a.out}: {len(cm.vertices)} vertices, {len(cm.faces)} triangles", file=sys.stderr)


if __name__ == "__main__":
    main()
