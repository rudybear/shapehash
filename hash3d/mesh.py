"""Decoded placeholder mesh container + GLB exporter (no trimesh needed)."""
import json
import struct
from dataclasses import dataclass

import numpy as np


@dataclass
class ColorMesh:
    vertices: np.ndarray  # (V,3) float, normalised frame
    faces: np.ndarray  # (F,3) int
    colors: np.ndarray  # (V,3) float sRGB in [0,1]
    flat: bool = False  # True: render with face normals (faceted look intended)

    def __post_init__(self):
        self.vertices = np.asarray(self.vertices, dtype=np.float64).reshape(-1, 3)
        self.faces = np.asarray(self.faces, dtype=np.int64).reshape(-1, 3)
        self.colors = np.clip(np.asarray(self.colors, dtype=np.float64).reshape(-1, 3), 0, 1)

    def face_normals(self):
        t = self.vertices[self.faces]
        n = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
        return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-20)

    def face_areas(self):
        t = self.vertices[self.faces]
        return 0.5 * np.linalg.norm(np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]), axis=1)

    def vertex_normals(self):
        t = self.vertices[self.faces]
        fn = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])  # area weighted
        vn = np.zeros_like(self.vertices)
        for k in range(3):
            np.add.at(vn, self.faces[:, k], fn)
        return vn / np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-20)

    def sample(self, n, seed=0):
        rng = np.random.default_rng(seed)
        a = self.face_areas()
        if len(a) == 0 or a.sum() <= 0:
            return np.zeros((n, 3)), np.tile([0, 0, 1.0], (n, 1)), np.zeros((n, 3))
        fi = rng.choice(len(a), size=n, p=a / a.sum())
        u, v = rng.random(n), rng.random(n)
        flip = u + v > 1
        u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
        bary = np.stack([1 - u - v, u, v], 1)
        f = self.faces[fi]
        pts = np.einsum("nk,nkc->nc", bary, self.vertices[f])
        col = np.einsum("nk,nkc->nc", bary, self.colors[f])
        return pts, self.face_normals()[fi], col


def srgb_to_linear(c):
    c = np.asarray(c, dtype=np.float64)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def export_glb(mesh: ColorMesh, path, scale=1.0, translate=(0, 0, 0)):
    """Write a minimal GLB: POSITION, COLOR_0 (linear, as glTF requires),
    NORMAL for smooth meshes (flat meshes omit normals -> viewers flat-shade),
    one rough double-sided material."""
    v = (mesh.vertices * scale + np.asarray(translate)).astype("<f4")
    c = srgb_to_linear(mesh.colors).astype("<f4")
    idx_type, idx_code = ("<u2", 5123) if len(v) < 65535 else ("<u4", 5125)
    idx = mesh.faces.astype(idx_type).ravel()
    blobs, views, accessors = [], [], []

    def add(arr, target, ctype, atype, minmax=False):
        raw = arr.tobytes()
        offset = sum(len(b) for b in blobs)
        blobs.append(raw + b"\0" * (-len(raw) % 4))
        views.append({"buffer": 0, "byteOffset": offset, "byteLength": len(raw), "target": target})
        acc = {"bufferView": len(views) - 1, "componentType": ctype, "count": int(arr.size if atype == "SCALAR" else len(arr)), "type": atype}
        if minmax:
            acc["min"], acc["max"] = arr.min(axis=0).tolist(), arr.max(axis=0).tolist()
        accessors.append(acc)
        return len(accessors) - 1

    attrs = {"POSITION": add(v, 34962, 5126, "VEC3", True)}
    if not mesh.flat:
        attrs["NORMAL"] = add(mesh.vertex_normals().astype("<f4"), 34962, 5126, "VEC3")
    attrs["COLOR_0"] = add(c, 34962, 5126, "VEC3")
    ia = add(idx, 34963, idx_code, "SCALAR")
    gltf = {
        "asset": {"version": "2.0", "generator": "hash3d"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [{"primitives": [{"attributes": attrs, "indices": ia, "material": 0, "mode": 4}]}],
        "materials": [{"pbrMetallicRoughness": {"baseColorFactor": [1, 1, 1, 1], "metallicFactor": 0.0, "roughnessFactor": 0.9}, "doubleSided": True}],
        "buffers": [{"byteLength": sum(len(b) for b in blobs)}],
        "bufferViews": views,
        "accessors": accessors,
    }
    js = json.dumps(gltf, separators=(",", ":")).encode()
    js += b" " * (-len(js) % 4)
    binb = b"".join(blobs)
    with open(path, "wb") as f:
        f.write(struct.pack("<4sII", b"glTF", 2, 12 + 8 + len(js) + 8 + len(binb)))
        f.write(struct.pack("<I4s", len(js), b"JSON") + js)
        f.write(struct.pack("<I4s", len(binb), b"BIN\0") + binb)


# ------------------------------------------------------------ shared shapes
_ICO_CACHE = {}


def icosphere(subdiv):
    """Unit icosphere (deterministic; a JS decoder would bake the same table)."""
    if subdiv in _ICO_CACHE:
        return _ICO_CACHE[subdiv]
    t = (1 + 5 ** 0.5) / 2
    v = [(-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0), (0, -1, t), (0, 1, t), (0, -1, -t), (0, 1, -t), (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1)]
    f = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
         (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9), (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)]
    v = [np.array(p, dtype=np.float64) / np.linalg.norm(p) for p in v]
    for _ in range(subdiv):
        mid, nf = {}, []

        def m(a, b):
            k = (min(a, b), max(a, b))
            if k not in mid:
                p = v[a] + v[b]
                v.append(p / np.linalg.norm(p))
                mid[k] = len(v) - 1
            return mid[k]

        for a, b, c in f:
            ab, bc, ca = m(a, b), m(b, c), m(c, a)
            nf += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        f = nf
    _ICO_CACHE[subdiv] = (np.array(v), np.array(f, dtype=np.int64))
    return _ICO_CACHE[subdiv]


def fibonacci_sphere(n):
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    th = np.pi * (1 + 5 ** 0.5) * i
    return np.stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)], 1)
