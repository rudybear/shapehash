"""Spherical-harmonic radius + colour (star-shaped placeholder).

r(dir) = distance from a centre to the OUTERMOST surface along dir, colour(dir)
= colour of that outermost point; both expanded in real spherical harmonics.
The expansion is done in bbox-normalised coordinates (bbox stretched to a
cube) so elongated objects do not waste coefficients on their aspect ratio.
"""
import numpy as np
import trimesh
from scipy.special import gammaln, lpmv

from ..bitio import dequant, quant, quant_extents, read_coeffs, read_extents, rgb_to_ycc, write_coeffs, write_extents, ycc_to_rgb
from ..mesh import ColorMesh, fibonacci_sphere, icosphere
from ..render import outer_radius
from .base import Codec, register


def real_sh(L, dirs):
    """Real orthonormal SH basis up to degree L. dirs (N,3) unit -> (N,(L+1)^2)"""
    z = np.clip(dirs[:, 2], -1, 1)
    phi = np.arctan2(dirs[:, 1], dirs[:, 0])
    out = np.zeros((len(dirs), (L + 1) ** 2))
    for l in range(L + 1):
        for m in range(0, l + 1):
            norm = np.sqrt((2 * l + 1) / (4 * np.pi) * np.exp(gammaln(l - m + 1) - gammaln(l + m + 1)))
            p = lpmv(m, l, z) * norm
            if m == 0:
                out[:, l * l + l] = p
            else:
                out[:, l * l + l + m] = np.sqrt(2) * p * np.cos(m * phi)
                out[:, l * l + l - m] = np.sqrt(2) * p * np.sin(m * phi)
    return out


GB, YB, CB = 5, 4, 3  # bits per AC coefficient: geometry, luma, chroma
HEAD = 8 + 18 + 15


def _bits(L, Ly, Lc):
    def grp(l, b, dc):
        return dc + (6 + ((l + 1) ** 2 - 1) * b if l > 0 else 0)

    return HEAD + grp(L, GB, 7) + grp(Ly, YB, 6) + 2 * grp(Lc, CB, 5)


def _make_classes():
    """Deterministic table: budget class -> (L, Ly, Lc). Baked into the codec."""
    out = []
    for budget in (12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024):
        bits = budget * 8
        best = (0, 0, 0)
        for L in range(0, 26):
            Ly, Lc = int(round(0.62 * L)), int(round(0.38 * L))
            if _bits(L, Ly, Lc) <= bits:
                best = (L, Ly, Lc)
        L, Ly, Lc = best
        while Ly < L and _bits(L, Ly + 1, Lc) <= bits:
            Ly += 1
        while Lc < Ly and _bits(L, Ly, Lc + 1) <= bits:
            Lc += 1
        out.append((L, Ly, Lc))
    return out


CLASSES = _make_classes()


def _fit(B, y, wgt, L, lam=1e-4):
    """Weighted ridge least squares with a Laplacian (l(l+1))^2 smoothness prior."""
    ls = np.concatenate([[l] * (2 * l + 1) for l in range(L + 1)]).astype(np.float64)
    A = B * wgt[:, None]
    reg = lam * wgt.sum() * (ls * (ls + 1)) ** 2 / max(1.0, (L * (L + 1.0)) ** 2)
    return np.linalg.solve(A.T @ B + np.diag(reg + 1e-9), A.T @ y)


@register
class SHCodec(Codec):
    """Payload: tag(4) class(4) | extents 18 | centre 3x5 | radius: DC 7, scale
    6, AC 5 bits each | luma: DC 6, scale 6, AC 4 each | Co, Cg: DC 5, scale 6,
    AC 3 each.  Degrees (L, Ly, Lc) come from the baked class table."""

    name = "sh"
    tag = 2
    decoder_cost = "low"
    decoder_needs = "SH evaluation on a fixed icosphere (matrix can be precomputed)"

    def encode(self, asset, budget_bytes):
        cls = max(i for i, c in enumerate(CLASSES) if _bits(*c) <= budget_bytes * 8)
        L, Ly, Lc = CLASSES[cls]
        w = self.start(cls)
        write_extents(w, asset.extents)
        e = quant_extents(asset.extents)
        # centre = centroid of the solid volume, in cube coordinates
        occ = asset.solid(48, -e, e)
        idx = np.argwhere(occ)
        c = ((idx.mean(axis=0) + 0.5) / 48 * 2 - 1) if len(idx) else np.zeros(3)
        cq = quant(c, -0.5, 0.5, 5)
        for v in cq:
            w.write(int(v), 5)
        c = dequant(cq, -0.5, 0.5, 5)
        cube = trimesh.Trimesh(asset.vertices / e, asset.faces, process=False)
        dirs = fibonacci_sphere(6000)
        r, tri, bary, ok = outer_radius(cube, c, dirs)
        B = real_sh(L, dirs)
        # misses count as radius ~0 (the shape is not star-shaped there)
        cr = _fit(B, np.maximum(r, 0.02), np.ones(len(dirs)), L)
        col = np.tile(asset.mean_color(), (len(dirs), 1))
        if ok.any():
            col[ok] = asset.color_at(tri[ok], bary[ok])
        ycc = rgb_to_ycc(col)
        wgt = np.where(ok, 1.0, 0.02)
        s4 = np.sqrt(4 * np.pi)
        for ch, (l, b, dcb, lo, hi) in enumerate(((L, GB, 7, 0.0, 2.0), (Ly, YB, 6, 0.0, 1.0), (Lc, CB, 5, -0.5, 0.5), (Lc, CB, 5, -0.5, 0.5))):
            if ch == 0:
                co = cr
            else:
                co = _fit(B[:, : (l + 1) ** 2], ycc[:, ch - 1], wgt, l)
            w.write(int(quant(co[0] / s4, lo, hi, dcb)), dcb)
            if l > 0:
                write_coeffs(w, co[1:], b, 6, 4.0)
        return self.finish(w, budget_bytes)

    def decode_payload(self, r, nibble):
        L, Ly, Lc = CLASSES[nibble]
        e = read_extents(r)
        c = dequant(np.array([r.read(5) for _ in range(3)]), -0.5, 0.5, 5)
        dirs, faces = icosphere(3 if L <= 8 else 4)
        B = real_sh(L, dirs)
        s4 = np.sqrt(4 * np.pi)
        vals = []
        for l, b, dcb, lo, hi in ((L, GB, 7, 0.0, 2.0), (Ly, YB, 6, 0.0, 1.0), (Lc, CB, 5, -0.5, 0.5), (Lc, CB, 5, -0.5, 0.5)):
            co = np.zeros((l + 1) ** 2)
            co[0] = dequant(r.read(dcb), lo, hi, dcb) * s4
            if l > 0:
                co[1:] = read_coeffs(r, len(co) - 1, b, 6, 4.0)
            vals.append(B[:, : len(co)] @ co)
        rad = np.maximum(vals[0], 0.02)
        v = (c + dirs * rad[:, None]) * e
        return ColorMesh(v, faces, ycc_to_rgb(np.stack(vals[1:], 1)))
