"""Convex k-DOP: support distances along a baked set of directions, one
colour per plane.  Decoder intersects the half-spaces."""
import numpy as np
from scipy.spatial import HalfspaceIntersection, cKDTree

from ..bitio import dequant, quant_extents, read_extents, read_rgb, write_extents, write_rgb
from ..mesh import ColorMesh, fibonacci_sphere
from .base import Codec, register

HMAX = 1.75
AXES = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], dtype=np.float64)


def directions(k):
    return np.concatenate([AXES, fibonacci_sphere(k)]) if k else AXES.copy()


@register
class KDopCodec(Codec):
    """Payload: tag(8) | K' 8 | extents 18 | 6 x RGB444 (axis planes) | K' x
    [support distance 6 bits, RGB444]."""

    name = "kdop"
    tag = 7
    decoder_cost = "medium"
    decoder_needs = "half-space intersection (= 3-D convex hull of K dual points)"

    def encode(self, asset, budget_bytes):
        k = int(np.clip((budget_bytes * 8 - 8 - 8 - 18 - 72) // 18, 0, 255))
        dirs = directions(k)
        e = quant_extents(asset.extents)
        verts = asset.vertices[np.unique(asset.faces)]
        h = (verts @ dirs.T).max(axis=0)
        hq = np.clip(np.ceil(h / HMAX * 63), 1, 63).astype(int)
        hd = np.concatenate([np.repeat(e, 2), hq[6:] / 63 * HMAX])
        pts, _, col = asset.sample(20000, seed=9)
        lab = np.argmax(pts @ dirs.T - hd, axis=1)
        mean = col.mean(0)
        w = self.start()
        w.write(k, 8)
        write_extents(w, asset.extents)
        for i in range(len(dirs)):
            if i >= 6:
                w.write(int(hq[i]), 6)
            write_rgb(w, col[lab == i].mean(0) if (lab == i).any() else mean)
        return self.finish(w, budget_bytes)

    def decode_payload(self, r, nibble):
        k = r.read(8)
        e = read_extents(r)
        dirs = directions(k)
        h = np.repeat(e, 2).tolist()
        col = []
        for i in range(len(dirs)):
            if i >= 6:
                h.append(max(r.read(6), 1) / 63 * HMAX)
            col.append(read_rgb(r))
        h = np.array(h)
        hs = HalfspaceIntersection(np.c_[dirs, -h], np.zeros(3))
        pts = hs.intersections
        pts = pts[np.all(np.isfinite(pts), axis=1)]
        # merge numerically identical vertices
        tree = cKDTree(pts)
        rep = np.array([min(g) for g in tree.query_ball_point(pts, 1e-7)])
        pts = pts[np.unique(rep)]
        V, F, C = [], [], []
        n = 0
        for i in range(len(dirs)):
            on = np.nonzero(np.abs(pts @ dirs[i] - h[i]) < 1e-6)[0]
            if len(on) < 3:
                continue
            p = pts[on]
            c = p.mean(0)
            a = np.cross(dirs[i], [1, 0, 0] if abs(dirs[i][0]) < 0.9 else [0, 1, 0])
            a /= np.linalg.norm(a)
            b = np.cross(dirs[i], a)
            order = np.argsort(np.arctan2((p - c) @ b, (p - c) @ a))
            p = p[order]
            V.append(p)
            m = len(p)
            F.append(np.stack([np.full(m - 2, n), n + np.arange(1, m - 1), n + np.arange(2, m)], 1))
            C.append(np.tile(col[i], (m, 1)))
            n += m
        return ColorMesh(np.concatenate(V), np.concatenate(F), np.concatenate(C), flat=True)
