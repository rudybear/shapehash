"""Trivial baselines: single-colour bounding box, and the box 'impostor'
(bounding box whose six faces carry BlurHash-like 2-D DCT colour images)."""
import numpy as np
from scipy import ndimage

from ..bitio import quant_extents, read_coeffs, read_extents, read_rgb, rgb_to_ycc, write_coeffs, write_extents, write_rgb, ycc_to_rgb
from ..mesh import ColorMesh
from ..render import cast
from .base import Codec, register

# face k: (axis, sign, u axis, v axis)
FACES = [(0, 1, 1, 2), (0, -1, 1, 2), (1, 1, 0, 2), (1, -1, 0, 2), (2, 1, 0, 1), (2, -1, 0, 1)]


def box_grid(e, g):
    """Box of half extents e, each face a g x g quad grid (unshared verts).
    -> vertices, faces, (face id, u, v in [0,1]) per vertex."""
    V, F, meta = [], [], []
    s = np.linspace(0, 1, g + 1)
    uu, vv = np.meshgrid(s, s, indexing="ij")
    for k, (ax, sg, ua, va) in enumerate(FACES):
        p = np.zeros((g + 1, g + 1, 3))
        p[..., ax] = sg * e[ax]
        p[..., ua] = (uu * 2 - 1) * e[ua]
        p[..., va] = (vv * 2 - 1) * e[va]
        base = sum(len(x) for x in V)
        V.append(p.reshape(-1, 3))
        meta.append(np.stack([np.full(uu.size, k), uu.ravel(), vv.ravel()], 1))
        i, j = np.meshgrid(np.arange(g), np.arange(g), indexing="ij")
        a = base + (i * (g + 1) + j).ravel()
        b, c, d = a + (g + 1), a + (g + 2), a + 1
        F.append(np.stack([a, b, c], 1))
        F.append(np.stack([a, c, d], 1))
    return np.concatenate(V), np.concatenate(F), np.concatenate(meta)


@register
class BoxCodec(Codec):
    """Payload: tag(8) + extents(18) + RGB555(15) = 41 bits -> 6 bytes, whatever the budget."""

    name = "box"
    tag = 0
    decoder_cost = "trivial"
    decoder_needs = "nothing (12 triangles)"

    def encode(self, asset, budget_bytes):
        w = self.start()
        write_extents(w, asset.extents)
        write_rgb(w, asset.mean_color(), (5, 5, 5))
        return self.finish(w, budget_bytes)

    def decode_payload(self, r, nibble):
        e = read_extents(r)
        col = read_rgb(r, (5, 5, 5))
        v, f, _ = box_grid(e, 1)
        return ColorMesh(v, f, np.tile(col, (len(v), 1)), flat=True)


def _dct_basis(n, x):
    """cos(pi * a * x) for a in 0..n-1, x in [0,1] -> (len(x), n)"""
    return np.cos(np.pi * np.outer(x, np.arange(n)))


# budget class -> (luma components per axis, chroma components per axis)
IMP_CLASSES = [(1, 1), (2, 1), (2, 2), (3, 2), (4, 2), (5, 3), (6, 3), (7, 4), (8, 5), (10, 6), (12, 7)]
IMP_YB, IMP_CB = 4, 3


def _imp_bits(ny, nc):
    per_face = (6 + (6 + (ny * ny - 1) * IMP_YB if ny > 1 else 0)) + 2 * (5 + (5 + (nc * nc - 1) * IMP_CB if nc > 1 else 0))
    return 8 + 18 + 6 * per_face


@register
class ImpostorCodec(Codec):
    """Bounding box + six tiny DCT face images (the literal 'six BlurHashes').

    Payload: tag(4) class(4) | extents 18 | per face: Y DC 6, [Y scale 6, Y AC
    4 bits each], Co/Cg DC 5 each, [scale 5, AC 3 bits each].
    """

    name = "impostor"
    tag = 1
    decoder_cost = "low"
    decoder_needs = "2-D inverse DCT per face (or 6 textures on a cube)"
    RES = 32

    def encode(self, asset, budget_bytes):
        cls = max(i for i, c in enumerate(IMP_CLASSES) if _imp_bits(*c) <= budget_bytes * 8)
        ny, nc = IMP_CLASSES[cls]
        w = self.start(cls)
        write_extents(w, asset.extents)
        e = quant_extents(asset.extents)
        n = self.RES
        s = (np.arange(n) + 0.5) / n
        uu, vv = np.meshgrid(s, s, indexing="ij")
        mean = rgb_to_ycc(asset.mean_color())
        for ax, sg, ua, va in FACES:
            o = np.zeros((n * n, 3))
            o[:, ua] = (uu.ravel() * 2 - 1) * e[ua]
            o[:, va] = (vv.ravel() * 2 - 1) * e[va]
            d = np.zeros((n * n, 3))
            d[:, ax] = sg
            tri, bary = cast(asset.mesh, o + d * 4.0, -d)
            hit = tri >= 0
            img = np.tile(mean, (n * n, 1))
            if hit.any():
                col = np.tile(asset.mean_color(), (n * n, 1))
                col[hit] = asset.color_at(tri[hit], bary[hit])
                # background pixels take the nearest foreground colour (no ringing at silhouettes)
                if not hit.all():
                    ind = ndimage.distance_transform_edt(~hit.reshape(n, n), return_distances=False, return_indices=True)
                    col = col.reshape(n, n, 3)[ind[0], ind[1]].reshape(-1, 3)
                img = rgb_to_ycc(col)
            img = img.reshape(n, n, 3)
            for ch, (k, dcb, acb) in enumerate(((ny, 6, IMP_YB), (nc, 5, IMP_CB), (nc, 5, IMP_CB))):
                B = _dct_basis(k, s)  # (n,k)
                norm = np.where(np.arange(k) == 0, 1.0, 2.0)
                co = (B.T @ img[..., ch] @ B) / (n * n) * np.outer(norm, norm)
                lo, hi = (0.0, 1.0) if ch == 0 else (-0.5, 0.5)
                w.write(int(np.clip(np.rint((co[0, 0] - lo) / (hi - lo) * ((1 << dcb) - 1)), 0, (1 << dcb) - 1)), dcb)
                if k > 1:
                    write_coeffs(w, co.ravel()[1:], acb, dcb, 1.0)
        return self.finish(w, budget_bytes)

    def decode_payload(self, r, nibble):
        ny, nc = IMP_CLASSES[nibble]
        e = read_extents(r)
        g = 4 if ny <= 2 else (8 if ny <= 5 else 16)
        v, f, meta = box_grid(e, g)
        col = np.zeros((len(v), 3))
        for k in range(6):
            sel = meta[:, 0] == k
            ycc = []
            for ch, (kk, dcb, acb) in enumerate(((ny, 6, IMP_YB), (nc, 5, IMP_CB), (nc, 5, IMP_CB))):
                lo, hi = (0.0, 1.0) if ch == 0 else (-0.5, 0.5)
                co = np.zeros(kk * kk)
                co[0] = lo + r.read(dcb) / ((1 << dcb) - 1) * (hi - lo)
                if kk > 1:
                    co[1:] = read_coeffs(r, kk * kk - 1, acb, dcb, 1.0)
                co = co.reshape(kk, kk)
                ycc.append(np.einsum("na,ab,nb->n", _dct_basis(kk, meta[sel, 1]), co, _dct_basis(kk, meta[sel, 2])))
            col[sel] = ycc_to_rgb(np.stack(ycc, 1))
        return ColorMesh(v, f, col, flat=True)
