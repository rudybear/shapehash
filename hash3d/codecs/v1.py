"""Adapters exposing the frozen ShapeHash v1 format to the evaluation harness."""
import numpy as np

from ..mesh import ColorMesh
from ..v1 import decode as v1_decode
from ..v1 import encode as v1e
from .base import Codec, register


def _source(asset):
    if "v1src" not in asset._cache:
        asset._cache["v1src"] = v1e.Source.from_asset(asset)
    return asset._cache["v1src"]


def to_colormesh(d):
    return ColorMesh(d.positions.astype(np.float64), d.indices.astype(np.int64), d.colors / 255.0, flat=d.flat)


class _V1(Codec):
    method = "auto"
    vox_mode = "smooth"

    def encode(self, asset, budget_bytes):
        key = ("v1enc", self.method, budget_bytes)
        if key not in asset._cache:  # vox1 / vox1c / union1 share encodes
            asset._cache[key] = v1e.encode(_source(asset), budget_bytes, self.method)
        s, info = asset._cache[key]
        self.last_info = info
        return s

    def decode(self, s):
        return to_colormesh(v1_decode(s, vox=self.vox_mode))


@register
class V1Vox(_V1):
    name, tag, method, decoder_cost = "vox1", 101, "vox", "medium"


@register
class V1VoxCubes(_V1):
    name, tag, method, vox_mode, decoder_cost = "vox1c", 102, "vox", "cubes", "low"


@register
class V1SH(_V1):
    name, tag, method, decoder_cost = "sh1", 103, "sh", "low"


@register
class V1Mesh(_V1):
    name, tag, method, decoder_cost = "mesh1", 104, "mesh", "trivial"


@register
class V1Union(_V1):
    name, tag, method, decoder_cost = "union1", 105, "auto", "medium"


@register
class TSEncoded(_V1):
    """Strings produced by the TypeScript encoder (ts/dist/cli.js batch),
    decoded and scored by the Python reference."""

    name, tag, decoder_cost = "ts1", 106, "medium"
    _rows = None

    def encode(self, asset, budget_bytes):
        import json
        import os

        if TSEncoded._rows is None:
            path = os.environ.get("HASH3D_TS_JSON") or os.path.join(os.path.dirname(__file__), "..", "..", "results", "v1", "ts_encoded.json")
            TSEncoded._rows = {(r["model"], r["budget"]): r for r in json.load(open(path)) if "hash" in r}
        row = TSEncoded._rows[(asset.name, budget_bytes)]
        self.last_info = {"method": row["method"], "scores": row["scores"]}
        return row["hash"]
