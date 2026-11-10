#!/usr/bin/env python3
"""Procedural test models (network-free fallback + controlled hard cases).

Written as GLB with vertex colours so they take the same path as downloads.
"""
import os
import sys

import numpy as np
import trimesh

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # works under python -I
from common import DATA  # noqa: E402


def colored(mesh, fn):
    mesh = mesh.subdivide().subdivide() if len(mesh.vertices) < 2000 else mesh
    c = np.clip(fn(mesh.vertices), 0, 1)
    mesh.visual = trimesh.visual.ColorVisuals(mesh, vertex_colors=(np.c_[c, np.ones(len(c))] * 255).astype(np.uint8))
    return mesh


def torus():
    m = trimesh.creation.torus(1.0, 0.3, major_sections=64, minor_sections=24)

    def fn(v):
        a = np.arctan2(v[:, 1], v[:, 0])
        stripe = (np.floor(a / (np.pi / 4)) % 2)[:, None]
        return stripe * np.array([0.9, 0.3, 0.1]) + (1 - stripe) * np.array([0.1, 0.4, 0.8])

    return colored(m, fn)


def table():
    parts = [trimesh.creation.box(extents=(2.0, 0.12, 1.2), transform=trimesh.transformations.translation_matrix((0, 1.0, 0)))]
    for sx in (-0.9, 0.9):
        for sz in (-0.5, 0.5):
            parts.append(trimesh.creation.box(extents=(0.1, 1.0, 0.1), transform=trimesh.transformations.translation_matrix((sx, 0.5, sz))))
    out = []
    for i, p in enumerate(parts):
        for _ in range(3):
            p = p.subdivide()
        col = np.array([0.55, 0.33, 0.15]) if i == 0 else np.array([0.2, 0.2, 0.22])
        p.visual = trimesh.visual.ColorVisuals(p, vertex_colors=np.tile((np.r_[col, 1] * 255).astype(np.uint8), (len(p.vertices), 1)))
        out.append(p)
    return trimesh.util.concatenate(out)


def dumbbell():
    a = trimesh.creation.icosphere(subdivisions=3, radius=0.5)
    b = a.copy()
    a.apply_translation((-1.2, 0, 0))
    b.apply_translation((1.2, 0, 0))
    bar = trimesh.creation.cylinder(radius=0.12, height=2.0, sections=24)
    bar.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, (0, 1, 0)))
    cols = [(0.85, 0.1, 0.1), (0.1, 0.6, 0.2), (0.8, 0.8, 0.82)]
    out = []
    for p, col in zip((a, b, bar), cols):
        p.visual = trimesh.visual.ColorVisuals(p, vertex_colors=np.tile((np.r_[col, 1] * 255).astype(np.uint8), (len(p.vertices), 1)))
        out.append(p)
    return trimesh.util.concatenate(out)


def main():
    d = os.path.join(DATA, "procedural")
    os.makedirs(d, exist_ok=True)
    for name, fn in (("ProcTorus", torus), ("ProcTable", table), ("ProcDumbbell", dumbbell)):
        fn().export(os.path.join(d, name + ".glb"))
        print("wrote", name)


if __name__ == "__main__":
    main()
