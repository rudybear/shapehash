"""Shared helpers for the scripts (path setup, model discovery)."""
import glob
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

DATA = os.path.join(ROOT, "data")
RESULTS = os.environ.get("HASH3D_RESULTS") or os.path.join(ROOT, "results")  # override for side experiments


def find_models():
    """-> sorted list of (name, path) for downloaded + procedural models."""
    out = {}
    for p in glob.glob(os.path.join(DATA, "models", "*.glb")) + glob.glob(os.path.join(DATA, "procedural", "*.glb")):
        out[os.path.splitext(os.path.basename(p))[0]] = p
    for p in glob.glob(os.path.join(DATA, "models", "*", "*.gltf")):
        out[os.path.splitext(os.path.basename(p))[0]] = p
    return sorted(out.items())
