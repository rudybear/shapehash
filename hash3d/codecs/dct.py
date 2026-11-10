"""'BlurHash in 3-D': low-frequency cosine series of a truncated signed
distance field (geometry) and of a colour field (fitted on the surface only),
decoded with marching cubes.

f(u,v,w) = sum_n a_n cos(pi a u) cos(pi b v) cos(pi c w),  (u,v,w) in [0,1]^3
spanning the bounding box enlarged by PAD.  Coefficient order (baked): by
a^2+b^2+c^2, then lexicographic.
"""
import numpy as np
from scipy import ndimage
from scipy.fft import dctn
from skimage.measure import marching_cubes

from ..bitio import dequant, qcoeffs, quant, quant_extents, read_coeffs, read_extents, rgb_to_ycc, write_coeffs, write_extents, ycc_to_rgb
from ..mesh import ColorMesh
from .base import Codec, register

PAD = 1.15
KMAX = 20
FINE = 64
GRID = 32  # decoder grid
GB, YB, CB = 4, 4, 3
HEAD = 8 + 18 + 4

_abc = np.array([(a, b, c) for a in range(KMAX) for b in range(KMAX) for c in range(KMAX)])
ORDER = _abc[np.lexsort((_abc[:, 2], _abc[:, 1], _abc[:, 0], (_abc**2).sum(1)))]


def _make_classes():
    out = []
    for budget in (12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024):
        bits = budget * 8 - HEAD
        ng = max(1, int((0.58 * bits - 13) // GB) + 1)
        ny = max(1, int((0.22 * bits - 12) // YB) + 1)
        nc = max(1, int((0.10 * bits - 11) // CB) + 1)
        out.append((ng, ny, nc))
    return out


CLASSES = _make_classes()


def _bits(ng, ny, nc):
    return HEAD + 7 + (6 + (ng - 1) * GB if ng > 1 else 0) + 6 + (6 + (ny - 1) * YB if ny > 1 else 0) + 2 * (5 + (6 + (nc - 1) * CB if nc > 1 else 0))


def basis_points(u, n):
    """Cosine basis at arbitrary points u (N,3) in [0,1] -> (N, n)."""
    k = int(ORDER[:n].max()) + 1
    cx = np.cos(np.pi * np.outer(u[:, 0], np.arange(k)))
    cy = np.cos(np.pi * np.outer(u[:, 1], np.arange(k)))
    cz = np.cos(np.pi * np.outer(u[:, 2], np.arange(k)))
    o = ORDER[:n]
    return cx[:, o[:, 0]] * cy[:, o[:, 1]] * cz[:, o[:, 2]]


def field_grid(coef, m):
    """Evaluate the series on an m^3 grid of cell centres (separable IDCT)."""
    n = len(coef)
    k = int(ORDER[:n].max()) + 1
    A = np.zeros((k, k, k))
    o = ORDER[:n]
    A[o[:, 0], o[:, 1], o[:, 2]] = coef
    C = np.cos(np.pi * np.outer((np.arange(m) + 0.5) / m, np.arange(k)))
    return np.einsum("ia,jb,kc,abc->ijk", C, C, C, A, optimize=True)


def _iso(q):
    return -0.3 + q / 15.0 * 0.9


@register
class DCTCodec(Codec):
    """Payload: tag(4) class(4) | extents 18 | iso 4 | SDF: DC 7, scale 6, AC 4
    bits each | luma: DC 6, scale 6, AC 4 each | Co, Cg: DC 5, scale 6, AC 3
    each.  Coefficient counts come from the baked class table."""

    name = "dct"
    tag = 3
    decoder_cost = "high"
    decoder_needs = "3-D inverse DCT on a 32^3 grid + marching cubes (or ray-march the series in a shader)"

    def encode(self, asset, budget_bytes):
        cls = max(i for i, c in enumerate(CLASSES) if _bits(*c) <= budget_bytes * 8)
        ng, ny, nc = CLASSES[cls]
        w = self.start(cls)
        write_extents(w, asset.extents)
        e = quant_extents(asset.extents)
        occ = asset.solid(FINE, -PAD * e, PAD * e)
        # thin sheets (open shells, fabric, leaves) would be erased by the
        # low-pass: thicken them first so they survive as fat sheets
        thin = occ & ~ndimage.binary_opening(occ, iterations=1)
        occ = occ | ndimage.binary_dilation(thin, iterations=1)
        sdf = ndimage.distance_transform_edt(~occ) - ndimage.distance_transform_edt(occ)
        sdf = sdf - np.sign(sdf) * 0.5
        f = np.tanh(sdf / 5.0)
        c = dctn(f, type=2, norm="ortho")
        k = int(ORDER[:ng].max()) + 1
        s = np.where(np.arange(k) == 0, np.sqrt(1.0 / FINE), np.sqrt(2.0 / FINE))
        amp = c[:k, :k, :k] * s[:, None, None] * s[None, :, None] * s[None, None, :]
        o = ORDER[:ng]
        cg = amp[o[:, 0], o[:, 1], o[:, 2]]
        # quantise exactly as the decoder will see it, then pick the best iso level
        dc_q = int(quant(cg[0], -1.0, 1.0, 7))
        cq = np.concatenate([[dequant(dc_q, -1.0, 1.0, 7)], qcoeffs(cg[1:], GB, 6, 2.0) if ng > 1 else []])
        fd = field_grid(cq, GRID)
        f2 = FINE // GRID
        target = occ.reshape(GRID, f2, GRID, f2, GRID, f2).mean(axis=(1, 3, 5))
        tb = target >= 0.5
        if not tb.any():
            tb = target > 0
        best, best_q = -1.0, 5
        for q in range(16):
            ob = fd < _iso(q)
            iou = (ob & tb).sum() / max((ob | tb).sum(), 1)
            # soft IoU against the fractional target keeps thin shells alive
            soft = np.minimum(ob, target).sum() / max(np.maximum(ob, target).sum(), 1e-9)
            score = 0.5 * iou + 0.5 * soft
            if score > best:
                best, best_q = score, q
        w.write(best_q, 4)
        w.write(dc_q, 7)
        if ng > 1:
            write_coeffs(w, cg[1:], GB, 6, 2.0)
        # colour: least squares on surface samples
        pts, nrm, col = asset.sample(20000, seed=3)
        # the decoded surface is not exactly on the true one: make the colour
        # field locally constant across the surface by fitting jittered copies
        rng = np.random.default_rng(3)
        off = rng.normal(0, 0.05, (2, len(pts), 1)) * nrm[None] + rng.normal(0, 0.02, (2, len(pts), 3))
        pts = np.concatenate([pts, pts + off[0], pts + off[1]])
        col = np.concatenate([col, col, col])
        u = np.clip((pts / (PAD * e) + 1) / 2, 0, 1)
        ycc = rgb_to_ycc(col)
        for ch, (n, b, dcb, lo, hi) in enumerate(((ny, YB, 6, 0.0, 1.0), (nc, CB, 5, -0.5, 0.5), (nc, CB, 5, -0.5, 0.5))):
            B = basis_points(u, n)
            freq = (ORDER[:n] ** 2).sum(1).astype(np.float64)
            reg = 1e-2 * len(u) * freq / max(freq.max(), 1.0)
            co = np.linalg.solve(B.T @ B + np.diag(reg + 1e-6), B.T @ ycc[:, ch])
            w.write(int(quant(co[0], lo, hi, dcb)), dcb)
            if n > 1:
                write_coeffs(w, co[1:], b, 6, 2.0)
        return self.finish(w, budget_bytes)

    def decode_payload(self, r, nibble):
        ng, ny, nc = CLASSES[nibble]
        e = read_extents(r)
        iso = _iso(r.read(4))
        cg = np.zeros(ng)
        cg[0] = dequant(r.read(7), -1.0, 1.0, 7)
        if ng > 1:
            cg[1:] = read_coeffs(r, ng - 1, GB, 6, 2.0)
        f = np.pad(field_grid(cg, GRID), 1, constant_values=1.0)
        if f.min() >= iso or f.max() <= iso:
            return ColorMesh(np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 3)))
        v, faces, _, _ = marching_cubes(f, level=iso)
        u = np.clip((v - 1 + 0.5) / GRID, 0, 1)
        ycc = []
        for n, b, dcb, lo, hi in ((ny, YB, 6, 0.0, 1.0), (nc, CB, 5, -0.5, 0.5), (nc, CB, 5, -0.5, 0.5)):
            co = np.zeros(n)
            co[0] = dequant(r.read(dcb), lo, hi, dcb)
            if n > 1:
                co[1:] = read_coeffs(r, n - 1, b, 6, 2.0)
            ycc.append(basis_points(u, n) @ co)
        return ColorMesh((u * 2 - 1) * PAD * e, faces, ycc_to_rgb(np.stack(ycc, 1)))
